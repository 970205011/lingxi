// L1 GDN（线性注意力）相关算子。

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// 深度可分离因果 1D 卷积 + silu：x[T, C], w[K, C]（行主序，第 k 行是第 k 个 tap）
// y[t][c] = silu(Σ_k x[t-(K-1-k)*D][c] * w[k][c]) 越界视为 0
// D = 膨胀系数（GDN 卷积 D=1；PLE 的 short conv D=ngram_size=3）
extern "C" __global__ void lingxi_causal_conv1d_silu_bf16(const bf16* __restrict__ x,
                                                          const bf16* __restrict__ w,
                                                          bf16* __restrict__ y,
                                                          int T,
                                                          int C,
                                                          int K,
                                                          int D) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * C) {
        return;
    }
    const int t = (int)(i / C);
    const int c = (int)(i % C);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int src = t - (K - 1 - k) * D;
        if (src >= 0) {
            acc += __bfloat162float(x[(size_t)src * C + c]) * __bfloat162float(w[(size_t)k * C + c]);
        }
    }
    y[i] = __float2bfloat16(acc / (1.0f + __expf(-acc)));
}

// **融合版**因果卷积：把「`hist`/`cur` 拼进 `cin` → conv → 再把末尾 `kh` 行拷回 `hist`」
// 合成**一次 launch**。原版每层要 **3 次 `memcpy_dtod` + 1 次 conv = 4 次 launch**，
// `LINGXI_KTS` 实测那段 **28 µs/层**，而数据只有 ~5 µs —— 按 launch ≈ 6.6 µs 算，**几乎全是 CPU 发射开销**。
//
// 语义与 `lingxi_causal_conv1d_silu_bf16` **逐位一致**：
// 原式是 `src = i - (K-1-k)*D`（`i` 是 `cin` 里的绝对行），调用方只用最后 `t` 行，
// 即 `i = kh + tt` ⇒ `src = (kh + tt) - (K-1-k)*D`，**k 仍按 0..K-1 升序累加**、silu 同式。
// 历史推进 `new_hist[j] = (hist; cur)[t + j]`，与原来的 `cin[t*C .. ] → k_hist` 等价。
extern "C" __global__ void lingxi_causal_conv1d_silu_hist_bf16(const bf16* __restrict__ hist,
                                                               const bf16* __restrict__ cur,
                                                               const bf16* __restrict__ w,
                                                               bf16* __restrict__ y,
                                                               bf16* __restrict__ new_hist,
                                                               int kh, int t, int C, int K, int D) {
    const int stride = (int)(gridDim.x * blockDim.x);
    const long long base = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    // 1) 只算 t 行输出
    for (long long i = base; i < (long long)t * C; i += stride) {
        const int tt = (int)(i / C);
        const int c = (int)(i % C);
        float acc = 0.0f;
#pragma unroll 4
        for (int k = 0; k < K; ++k) {
            const int src = (kh + tt) - (K - 1 - k) * D;
            if (src >= 0) {
                const float xv = (src < kh) ? __bfloat162float(hist[(size_t)src * C + c])
                                            : __bfloat162float(cur[(size_t)(src - kh) * C + c]);
                acc += xv * __bfloat162float(w[(size_t)k * C + c]);
            }
        }
        y[i] = __float2bfloat16(acc / (1.0f + __expf(-acc)));
    }
    // 2) 历史推进：new_hist[j] = (hist; cur)[t + j]
    for (long long i = base; i < (long long)kh * C; i += stride) {
        const int j = (int)(i / C);
        const int c = (int)(i % C);
        const int src = t + j;
        new_hist[i] = (src < kh) ? hist[(size_t)src * C + c] : cur[(size_t)(src - kh) * C + c];
    }
}

// 与 host 参考 `lingxi-model/src/gdn.rs::recurrent_core` **逐条对应**（同一运算顺序，
// 便于对拍）。一个 block 负责一个 v-head，`blockDim.x == hv`，逐 token 串行推进。
//
// 每步（head vh、token i）：
// decay = exp(-exp(a_log[vh]) * softplus(a[i,vh] + dt_bias[vh])) beta = sigmoid(b[i,vh])
// S *= decay; kv_mem[dv] = Σ_dk S[dk][dv]·k̂[dk]
// delta[dv] = (v[dv] - kv_mem[dv])·beta; S[dk][dv] += k̂[dk]·delta[dv]
// o[dv] = Σ_dk S[dk][dv]·q̂[dk]
// gated = o · rsqrt(mean(o²)+eps) · norm_w[dv] · act(z[dv])
// 其中 q̂ = q / ‖q_head‖ / √hk，k̂ = k / ‖k_head‖（头内 l2，q/k 按 k-head 分，rep 复制到 v-head）。
// `s_all` 是**持久设备状态**，就地推进；`gate_act`：0=silu 1=sigmoid。
// ★ P15-㉘（2026-10-17）：`use_smem` 时把本 v-head 的 S（hk×hv fp32，本模型 64KB）
// 一次性搬进**动态 shared memory 驻留**，T 循环全程打 smem，结束再写回全局。
// 收益账：S 原来每 token 被全局内存读+写共 4 遍（decay 读写 + update 读写），
// T=6 一轮 = 4×3MB×6 ≈ 75MB/层；smem 驻留后每轮只有 in/out 各一次 = 6MB，砍 12×。
// **逐位等价**：每个线程的 dk 串行累加次序、归约结构、写回值全部不变，
// 只是 S 的驻留位置从全局换成 smem（初值同一字节、写回同一字节）。
// ★ P15-㉟ 并发批（P1）：`n_seq > 1` 时一个 batch 里有 N 条序列各推 1 行。
// - 状态布局变 seq-major：`s_all[(seq*num_v_heads + vh) * hk*hv]`（调用方把 N 条的
// GdnState.s_all 连续排列，或传各自的基址差固定 stride）。
// - 输入行按 seq 交错：`conv[(i*n_seq + seq) * conv_dim]`（i 是本步内的行序号）。
// - 输出同理：`gated[(i*n_seq + seq) * value_dim]`。
// `n_seq==1` 时全部退化为老布局 ⇒ **逐位等价**（单序列调用传 1 即可）。
extern "C" __global__ void lingxi_gdn_recurrent_bf16(
    const bf16* __restrict__ conv,     // [T*n_seq, conv_dim]（已过 silu 的因果卷积输出）
    const bf16* __restrict__ z,        // [T*n_seq, value_dim]
    const bf16* __restrict__ agate,    // [T*n_seq, num_v_heads] (a)
    const bf16* __restrict__ bgate,    // [T*n_seq, num_v_heads] (b)
    const bf16* __restrict__ norm_w,   // [hv]（bf16 原值，f32 乘）
    const float* __restrict__ a_log,   // [num_v_heads]
    const float* __restrict__ dt_bias, // [num_v_heads]
    float* __restrict__ s_all,         // [n_seq, num_v_heads, hk*hv] 持久状态（就地，seq-major）
    bf16* __restrict__ gated,          // [T*n_seq, value_dim]
    int T, int hk, int hv, int num_v_heads, int num_k_heads, int gate_act, int row_off,
    int use_smem, int n_seq,
    int prefetch) {  // 补3：wrapper 统一传参；本（旧）核忽略之，保持 launch 形参一致
    const int nvh = num_v_heads;
    const int vh = blockIdx.x % nvh;
    const int seq = blockIdx.x / nvh;
    const int dv = threadIdx.x;
    const int key_dim = num_k_heads * hk;
    const int value_dim = num_v_heads * hv;
    const int conv_dim = 2 * key_dim + value_dim;
    const int rep = num_v_heads / num_k_heads;
    const int hh = vh / rep;  // 该 v-head 所属的 k-head

    __shared__ float ssum[32];
    extern __shared__ float sS[];
    float* S = s_all + ((size_t)seq * nvh + vh) * hk * hv;
    float* Sw = S;
    if (use_smem) {
        for (int idx = threadIdx.x; idx < hk * hv; idx += blockDim.x) {
            sS[idx] = S[idx];
        }
        __syncthreads();
        Sw = sS;
    }

    for (int i = 0; i < T; ++i) {
        const bf16* conv_i = conv + (size_t)(i * n_seq + seq) * conv_dim;
        const bf16* qrow = conv_i + (size_t)hh * hk;
        const bf16* krow = conv_i + key_dim + (size_t)hh * hk;

        // 头内 l2 范数（block 内各线程算同一段，读为广播）
        float qss = 0.0f, kss = 0.0f;
        // ★ P15-㉘：48 block × 4 warp（2.4 warp/SM）是**延迟受限**；且原循环里
        // `S[dk]=s` 的**写后读**让编译器无法证明相邻迭代不别名（hk/hv 是运行时值），
        // load 被串行化在 store 之后 ⇒ 每迭代吃满一次访存延迟（隔离实测 35µs/token
        // ≈ 128×2×250cyc，正是串行延迟账）。修法 = **显式 16 批量预载**（先纯 load
        // 16 个进寄存器，再算+写），把在飞 load 提到 16。kv_mem/acc/qss/kss 的累加
        // 次序不变（仍按 dk 升序）⇒ **逐位一致**。
        {
            int d = 0;
            for (; d + 16 <= hk; d += 16) {
                float qv[16], kv[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    qv[u] = __bfloat162float(qrow[d + u]);
                    kv[u] = __bfloat162float(krow[d + u]);
                }
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    qss += qv[u] * qv[u];
                    kss += kv[u] * kv[u];
                }
            }
            for (; d < hk; ++d) {
                const float qv = __bfloat162float(qrow[d]);
                const float kv = __bfloat162float(krow[d]);
                qss += qv * qv;
                kss += kv * kv;
            }
        }
        const float qscale = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)hk);
        const float kscale = 1.0f / sqrtf(kss + 1e-6f);

        const int r_i = i * n_seq + seq;  // 交错行号（n_seq==1 时 = i，退化为老布局）
        const float av = __bfloat162float(agate[(size_t)r_i * num_v_heads + vh]) + dt_bias[vh];
        const float sp = (av > 20.0f) ? av : log1pf(expf(av));
        const float decay = expf(-expf(a_log[vh]) * sp);
        const float bv = __bfloat162float(bgate[(size_t)r_i * num_v_heads + vh]);
        const float beta = 1.0f / (1.0f + expf(-bv));

        // S *= decay；同时算 kv_mem = Σ_dk S[dk][dv]·k̂[dk]
        float kv_mem = 0.0f;
        {
            int dk = 0;
            for (; dk + 16 <= hk; dk += 16) {
                float sv[16], kn[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    sv[u] = Sw[(size_t)(dk + u) * hv + dv];
                    kn[u] = __bfloat162float(krow[dk + u]) * kscale;
                }
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const float s = sv[u] * decay;
                    Sw[(size_t)(dk + u) * hv + dv] = s;
                    kv_mem += s * kn[u];
                }
            }
            for (; dk < hk; ++dk) {
                const float s = Sw[(size_t)dk * hv + dv] * decay;
                Sw[(size_t)dk * hv + dv] = s;
                kv_mem += s * (__bfloat162float(krow[dk]) * kscale);
            }
        }
        const float v = __bfloat162float(conv_i[2 * key_dim + (size_t)vh * hv + dv]);
        const float delta = (v - kv_mem) * beta;
        // S += k̂ ⊗ delta；同时算 o = Σ_dk S[dk][dv]·q̂[dk]（用**更新后**的 S）
        float acc = 0.0f;
        {
            int dk = 0;
            for (; dk + 16 <= hk; dk += 16) {
                float su0[16], qn[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    su0[u] = Sw[(size_t)(dk + u) * hv + dv];
                    qn[u] = __bfloat162float(qrow[dk + u]) * qscale;
                }
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const float su = su0[u] + __bfloat162float(krow[dk + u]) * kscale * delta;
                    Sw[(size_t)(dk + u) * hv + dv] = su;
                    acc += su * qn[u];
                }
            }
            for (; dk < hk; ++dk) {
                const float kn = __bfloat162float(krow[dk]) * kscale;
                const float su = Sw[(size_t)dk * hv + dv] + kn * delta;
                Sw[(size_t)dk * hv + dv] = su;
                acc += su * (__bfloat162float(qrow[dk]) * qscale);
            }
        }

        // RMSNormGated：跨 dv 归约 mean(o²)
        float sq = acc * acc;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, off);
        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        if (lane == 0) ssum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            const int nwarp = (blockDim.x + 31) >> 5;
            float v2 = (lane < nwarp) ? ssum[lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v2 += __shfl_down_sync(0xffffffffu, v2, off);
            if (lane == 0) ssum[0] = v2;
        }
        __syncthreads();
        const float r = 1.0f / sqrtf(ssum[0] / (float)hv + 1e-6f);

        const float zz = __bfloat162float(z[(size_t)r_i * value_dim + (size_t)vh * hv + dv]);
        const float actv = (gate_act == 0) ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
        gated[(size_t)r_i * value_dim + (size_t)vh * hv + dv] =
            __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
        __syncthreads();  // ssum 复用：下一轮归约前必须全体同步
    }
    // smem 驻留：把推进后的 S 写回全局（值与逐 token 写全局的路径逐位相同）。
    if (use_smem) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < hk * hv; idx += blockDim.x) {
            S[idx] = sS[idx];
        }
    }
}

// ★ 2026-10-24GDN 递推 **state 常驻寄存器版**（hk=hv=128 特化，逐位相同）。
//
// 分母（`tests/t16_gdn_rec_bw.rs`，同形探针）把旧核的账算清了（t=6，S = 48×128×128×f32 = 3.00 MiB）：
// * 旧核实测 **82.6 µs**，每 token 对 S 做 2 读 + 2 写 ⇒ 72 MiB/层；
// * **同形 2R+2W 纯搬运（不做递推）只要 41.3 µs**、同形纯读 2 遍 19.3 µs、
// 1R+1W 纯搬运 8.5 µs ⇒ **旧核只有"它自己那份地址流"的 49%**，取数结构远没到顶；
// * 更狠的一条：**q/k 的 3 遍广播 load**（范数遍 + 两次递推遍，每线程 384 次 load + 384 次 cvt）
// 才是真瓶颈 —— 把它们换成"每 token 每线程只读 1 个 q + 1 个 k 进 smem"，再把 state 全程
// 驻留寄存器（S 只 1 读 + 1 写），同一个算式从 82.6 → **32.1 µs（2.6×）**。
//
// 逐位：累加次序（dk 升序）、行串行、RMSNorm 的 shuffle 树、decay 的每一次舍入**一字未改**；
// 唯一变化是 S 的驻留位置（global → 寄存器）与 q/k 的取数来源（global → smem 一次性搬运）。
// 证据：`t16_gdn_rec_bits` 对 s_all(786432) 与 gated(36864) **0 位不同**。
//
// 形状：只支持 hk == hv == 128（本模型），其余形状由 `wrappers.rs` 回退旧核。
// `LINGXI_NO_GDN_REC_REG=1` 回退（A/B 用）。
#define GDN_REC_HK 128
#define GDN_REC_HV 128

// 流水版的头内 l2 范数（与寄存器核同一 float4 次序、同一累加链）。
__device__ __forceinline__ void gdn_rec_l2norm(const float* sQ, const float* sK,
                                               float& qss, float& kss) {
    qss = 0.0f;
    kss = 0.0f;
#pragma unroll
    for (int d = 0; d < GDN_REC_HK; d += 4) {
        const float4 q4 = *(const float4*)(sQ + d);
        const float4 k4 = *(const float4*)(sK + d);
        qss += q4.x * q4.x;
        qss += q4.y * q4.y;
        qss += q4.z * q4.z;
        qss += q4.w * q4.w;
        kss += k4.x * k4.x;
        kss += k4.y * k4.y;
        kss += k4.z * k4.z;
        kss += k4.w * k4.w;
    }
}

// （2026-10-27）GDN rec **流水版**核体：`pipe=1` 档（`LINGXI_GDN_PIPE` 门控，wrapper 传入）。
// 与寄存器核**逐位等价**：所有算术表达式、舍入、dk 升序累加链、RMSNorm 的 shuffle 树
// 一字未改；只改**调度**——
// 1. `al = expf(a_log[vh])` 提出 T 循环（循环不变量，同函数同输入 ⇒ 同值）；
// 2. 每步 5 个 __syncthreads → 3 个：sQ/sK/sQN/sKN 双缓冲（按 i&1 取槽）消掉
// "写后全员可读"的串行配对，ssum[2][32] 按步 parity 消掉尾部"复用保护"同步；
// 3. RMSNorm 跨 warp 合并由 warp0 单算 + ssum[0] 广播，改为**每个 warp 各自算同一棵
// shuffle 树**（同一批 ssum 值、同一归约次序 ⇒ 同一 v2）+ __shfl_sync(warp 内广播
// lane0)——消掉第二个 broadcast 同步；
// 4. 门（decay/beta）改"提前一拍"算好放旋转寄存器（表达式与旧核逐字相同），其
// expf 链与 ③④ 的 dk 串行链无数据依赖 ⇒ 编译器把它们交叠进 FMA 链的停顿槽；
// 5. 下一步 q/k 行的装载+范数放在本步 ④ 之后、两个同步之间（与旧核同一累加次序）。
// ★ 负结果（判决实验，已删码留档）：每 block 装 2 个 v-head（256 线程 = 每 SMSP
// 2 warp，想让两个 head 的流互填 spill/transcendental 停顿槽）实测 rec 2587~2591µs
// **慢于**基线 2296µs——8 warp 块内 barrier 把两个 head 锁步、arrival skew 全暴露，
// 且只占 24/80 SM。判决否决。
__device__ __forceinline__ void gdn_rec_reg_pipe_body(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    const int nvh = num_v_heads;
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    const int key_dim = num_k_heads * GDN_REC_HK;
    const int value_dim = nvh * GDN_REC_HV;
    const int conv_dim = 2 * key_dim + value_dim;
    const int rep = nvh / num_k_heads;
    const int hh = vh / rep;

    __shared__ float ssum[2][4];
    __shared__ float sQ[2][GDN_REC_HV], sK[2][GDN_REC_HV];
    __shared__ float sQN[2][GDN_REC_HV], sKN[2][GDN_REC_HV];
    float* S = s_all + (size_t)vh * GDN_REC_HK * GDN_REC_HV;

    float st[GDN_REC_HK];
#pragma unroll
    for (int d = 0; d < GDN_REC_HK; ++d) st[d] = S[(size_t)d * GDN_REC_HV + dv];

    const float al = expf(a_log[vh]);  //：循环不变量提出（同函数同输入 ⇒ 逐位同值）

    // prologue：第 0 步的 q/k 范数 + scaled（与旧核同一累加次序、同一舍入）。
    {
        const bf16* qrow = conv + (size_t)hh * GDN_REC_HK;
        const bf16* krow = conv + key_dim + (size_t)hh * GDN_REC_HK;
        sQ[0][dv] = __bfloat162float(qrow[dv]);
        sK[0][dv] = __bfloat162float(krow[dv]);
        __syncthreads();
        float qss, kss;
        gdn_rec_l2norm(sQ[0], sK[0], qss, kss);
        const float qs0 = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)GDN_REC_HK);
        const float ks0 = 1.0f / sqrtf(kss + 1e-6f);
        sQN[0][dv] = sQ[0][dv] * qs0;
        sKN[0][dv] = sK[0][dv] * ks0;
    }
    // 旋转寄存器：v_c/zz_c 持"当前步"原始值；decay_c/beta_c 持"当前步"门计算值。
    float v_c = __bfloat162float(conv[2 * key_dim + (size_t)vh * GDN_REC_HV + dv]);
    float zz_c = __bfloat162float(z[(size_t)vh * GDN_REC_HV + dv]);
    {
        const float av0 = __bfloat162float(agate[vh]) + dt_bias[vh];
        const float bv0 = __bfloat162float(bgate[vh]);
        const float sp0 = (av0 > 20.0f) ? av0 : log1pf(expf(av0));
        float decay_c = expf(-al * sp0);
        float beta_c = 1.0f / (1.0f + expf(-bv0));
        __syncthreads();  // sQN[0]/sKN[0] 写后全员可读
        for (int i = 0; i < T; ++i) {
            const int p = i & 1;
            const float decay = decay_c;
            const float beta = beta_c;

            //：提前发下一拍门标量 load 并随即算好下一拍门（表达式逐字相同；
            // expf 链与 ③④ 的 dk 串行链无数据依赖 ⇒ 交叠进 FMA 停顿槽），
            // avn/bvn 立刻死亡，不横跨 ③④。
            if (i + 1 < T) {
                const float avn = __bfloat162float(agate[(size_t)(i + 1) * nvh + vh]) + dt_bias[vh];
                const float bvn = __bfloat162float(bgate[(size_t)(i + 1) * nvh + vh]);
                const float spn = (avn > 20.0f) ? avn : log1pf(expf(avn));
                decay_c = expf(-al * spn);
                beta_c = 1.0f / (1.0f + expf(-bvn));
            }

            // ③ S *= decay；kv_mem = Σ S·k̂（dk 升序、逐项一次舍入——与旧核逐字相同）
            float kv_mem = 0.0f;
#pragma unroll
            for (int d = 0; d < GDN_REC_HK; d += 4) {
                const float4 kn = *(const float4*)(sKN[p] + d);
                const float v0 = st[d] * decay;
                const float v1 = st[d + 1] * decay;
                const float v2 = st[d + 2] * decay;
                const float v3 = st[d + 3] * decay;
                st[d] = v0;
                st[d + 1] = v1;
                st[d + 2] = v2;
                st[d + 3] = v3;
                kv_mem += v0 * kn.x;
                kv_mem += v1 * kn.y;
                kv_mem += v2 * kn.z;
                kv_mem += v3 * kn.w;
            }
            const float delta = (v_c - kv_mem) * beta;
            if (i + 1 < T) {  // v 消费完毕，预载下一拍（同地址同字节）
                v_c = __bfloat162float(
                    conv[(size_t)(i + 1) * conv_dim + 2 * key_dim + (size_t)vh * GDN_REC_HV + dv]);
            }

            // ④ S += k̂⊗delta；o = Σ S_new·q̂（dk 升序——与旧核逐字相同）
            float acc = 0.0f;
#pragma unroll
            for (int d = 0; d < GDN_REC_HK; d += 4) {
                const float4 kn = *(const float4*)(sKN[p] + d);
                const float4 qn = *(const float4*)(sQN[p] + d);
                const float u0 = st[d] + kn.x * delta;
                const float u1 = st[d + 1] + kn.y * delta;
                const float u2 = st[d + 2] + kn.z * delta;
                const float u3 = st[d + 3] + kn.w * delta;
                st[d] = u0;
                st[d + 1] = u1;
                st[d + 2] = u2;
                st[d + 3] = u3;
                acc += u0 * qn.x;
                acc += u1 * qn.y;
                acc += u2 * qn.z;
                acc += u3 * qn.w;
            }

            // ⑤ RMSNormGated：warp 内 shuffle 树与旧核相同；跨 warp 改为每 warp 自算
            // 同一棵树（同值）+ lane0 广播——只 1 个 __syncthreads。
            float sq = acc * acc;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, off);
            const int lane = threadIdx.x & 31;
            const int wid = threadIdx.x >> 5;
            ssum[p][wid & 3] = sq;
            __syncthreads();
            float v2 = (lane < 4) ? ssum[p][lane] : 0.0f;  // nwarp 恒为 4（hv=128）
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v2 += __shfl_down_sync(0xffffffffu, v2, off);
            v2 = __shfl_sync(0xffffffffu, v2, 0);  // warp 内广播 lane0 的总值（与旧核 ssum[0] 同值）
            const float r = 1.0f / sqrtf(v2 / (float)GDN_REC_HV + 1e-6f);

            const float actv =
                (gate_act == 0) ? (zz_c / (1.0f + __expf(-zz_c)))
                                : (1.0f / (1.0f + __expf(-zz_c)));
            gated[(size_t)i * value_dim + (size_t)vh * GDN_REC_HV + dv] =
                __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
            if (i + 1 < T) {  // zz 消费完毕，预载下一拍
                zz_c = __bfloat162float(
                    z[(size_t)(i + 1) * value_dim + (size_t)vh * GDN_REC_HV + dv]);
            }

            //：下一拍 q/k 进双缓冲另一槽 → sync → 范数（同序）→ scaled → sync。
            if (i + 1 < T) {
                const int np = p ^ 1;
                const bf16* conv_n = conv + (size_t)(i + 1) * conv_dim;
                sQ[np][dv] = __bfloat162float(conv_n[(size_t)hh * GDN_REC_HK + dv]);
                sK[np][dv] = __bfloat162float(conv_n[key_dim + (size_t)hh * GDN_REC_HK + dv]);
                __syncthreads();
                float qss, kss;
                gdn_rec_l2norm(sQ[np], sK[np], qss, kss);
                const float qs1 = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)GDN_REC_HK);
                const float ks1 = 1.0f / sqrtf(kss + 1e-6f);
                sQN[np][dv] = sQ[np][dv] * qs1;
                sKN[np][dv] = sK[np][dv] * ks1;
                __syncthreads();
            }
        }
    }

#pragma unroll
    for (int d = 0; d < GDN_REC_HK; ++d) S[(size_t)d * GDN_REC_HV + dv] = st[d];
}

extern "C" __global__ void __launch_bounds__(GDN_REC_HV, 1) lingxi_gdn_recurrent_reg_bf16(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int hk, int hv, int num_v_heads,
    int num_k_heads, int gate_act, int row_off, int use_smem, int n_seq,
    int prefetch, int pipe) {
    //：流水版（`LINGXI_GDN_PIPE=1`；逐位等价论证见 gdn_rec_reg_pipe_body 头注）。
    // 负结果留档：每 block 2 v-head（H=2，每 SMSP 2 warp 互填停顿槽）实测 rec
    // 2587~2591µs **慢于基线** 2296µs——8 warp 块内 barrier 把两个 head 锁步、
    // skew 全暴露，且只占 24/80 SM ⇒ 判决否决，H=2 实例已删。
    if (pipe) {
        gdn_rec_reg_pipe_body(conv, z, agate, bgate, norm_w, a_log, dt_bias,
                              s_all, gated, T, num_v_heads, num_k_heads, gate_act);
        return;
    }
    const int nvh = num_v_heads;
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    const int key_dim = num_k_heads * GDN_REC_HK;
    const int value_dim = nvh * GDN_REC_HV;
    const int conv_dim = 2 * key_dim + value_dim;
    const int rep = nvh / num_k_heads;
    const int hh = vh / rep;

    __shared__ float ssum[32];
    __shared__ float sQ[GDN_REC_HV], sK[GDN_REC_HV], sQN[GDN_REC_HV], sKN[GDN_REC_HV];
    float* S = s_all + (size_t)vh * GDN_REC_HK * GDN_REC_HV;

    // 本线程的一整列 state（hk 个 f32）全程驻留寄存器：整段只 1 读 + 1 写。
    float st[GDN_REC_HK];
#pragma unroll
    for (int d = 0; d < GDN_REC_HK; ++d) st[d] = S[(size_t)d * GDN_REC_HV + dv];

    // ★ 补3（2026-10-27）：`prefetch=1` 时手动**提前一拍**预载下一 step 的
    // 门/v/z 标量 + q/k 行 L2 预取。动机：循环体内的 `__syncthreads` 会阻止编译器
    // 跨迭代提升 load，而每 step 关键路径上有 ~3 次全局访存延迟（agate/bgate/v/z）
    // 裸奔（block 仅 4 warp/SM，无其他 warp 可切换）。**逐位等价**：只是把同一地址
    // 的同一字节提前取进寄存器/缓存，任何算术表达式、累加次序、归约树一字未动。
    // 实测开关 = `LINGXI_GDN_PREFETCH`（wrapper 传入）。
    float av_c = 0.0f, bv_c = 0.0f, v_c = 0.0f, zz_c = 0.0f;
    if (prefetch && 0 < T) {
        av_c = __bfloat162float(agate[vh]) + dt_bias[vh];
        bv_c = __bfloat162float(bgate[vh]);
        v_c = __bfloat162float(conv[2 * key_dim + (size_t)vh * GDN_REC_HV + dv]);
        zz_c = __bfloat162float(z[(size_t)vh * GDN_REC_HV + dv]);
    }

    for (int i = 0; i < T; ++i) {
        const bf16* conv_i = conv + (size_t)i * conv_dim;
        const bf16* qrow = conv_i + (size_t)hh * GDN_REC_HK;
        const bf16* krow = conv_i + key_dim + (size_t)hh * GDN_REC_HK;

        // 补3：q/k 行的 L2 预取（下一拍用；prefetch 指令是纯提示）。
        if (prefetch && i + 1 < T) {
            const bf16* conv_n = conv + (size_t)(i + 1) * conv_dim;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(conv_n + (size_t)hh * GDN_REC_HK + dv));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(conv_n + key_dim + (size_t)hh * GDN_REC_HK + dv));
        }

        // ① q/k 只搬一次进 smem（raw 供范数、scaled 供递推）；scaled 的值与旧核逐位相同
        // （同一个 `cvt(bf16) * scale`，同一个 f32 舍入）。
        sQ[dv] = __bfloat162float(qrow[dv]);
        sK[dv] = __bfloat162float(krow[dv]);
        __syncthreads();

        // ② 头内 l2 范数：dk 升序（与旧核同一累加次序），smem 读走 float4。
        float qss = 0.0f, kss = 0.0f;
#pragma unroll
        for (int d = 0; d < GDN_REC_HK; d += 4) {
            const float4 q4 = *(const float4*)(sQ + d);
            const float4 k4 = *(const float4*)(sK + d);
            qss += q4.x * q4.x;
            qss += q4.y * q4.y;
            qss += q4.z * q4.z;
            qss += q4.w * q4.w;
            kss += k4.x * k4.x;
            kss += k4.y * k4.y;
            kss += k4.z * k4.z;
            kss += k4.w * k4.w;
        }
        const float qscale = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)GDN_REC_HK);
        const float kscale = 1.0f / sqrtf(kss + 1e-6f);
        sQN[dv] = sQ[dv] * qscale;
        sKN[dv] = sK[dv] * kscale;
        __syncthreads();

        // 补3：门标量用旋转寄存器（prefetch=1 时上一拍已载好；=0 时即时装载，
        // 表达式与旧核逐字相同）。
        float av, bv;
        if (prefetch) {
            av = av_c;
            bv = bv_c;
        } else {
            av = __bfloat162float(agate[(size_t)i * nvh + vh]) + dt_bias[vh];
            bv = __bfloat162float(bgate[(size_t)i * nvh + vh]);
        }
        const float sp = (av > 20.0f) ? av : log1pf(expf(av));
        const float decay = expf(-expf(a_log[vh]) * sp);
        const float beta = 1.0f / (1.0f + expf(-bv));
        // 补3：decay/beta 算完即覆写旋转寄存器，预载下一拍门标量。
        if (prefetch && i + 1 < T) {
            av_c = __bfloat162float(agate[(size_t)(i + 1) * nvh + vh]) + dt_bias[vh];
            bv_c = __bfloat162float(bgate[(size_t)(i + 1) * nvh + vh]);
        }

        // ③ S *= decay；kv_mem = Σ S·k̂（dk 升序、逐项一次舍入）
        float kv_mem = 0.0f;
#pragma unroll
        for (int d = 0; d < GDN_REC_HK; d += 4) {
            const float4 kn = *(const float4*)(sKN + d);
            const float v0 = st[d] * decay;
            const float v1 = st[d + 1] * decay;
            const float v2 = st[d + 2] * decay;
            const float v3 = st[d + 3] * decay;
            st[d] = v0;
            st[d + 1] = v1;
            st[d + 2] = v2;
            st[d + 3] = v3;
            kv_mem += v0 * kn.x;
            kv_mem += v1 * kn.y;
            kv_mem += v2 * kn.z;
            kv_mem += v3 * kn.w;
        }
        float v;
        if (prefetch) {
            v = v_c;
        } else {
            v = __bfloat162float(conv_i[2 * key_dim + (size_t)vh * GDN_REC_HV + dv]);
        }
        const float delta = (v - kv_mem) * beta;
        // 补3：v 消费完毕，预载下一拍的 v。
        if (prefetch && i + 1 < T) {
            v_c = __bfloat162float(
                conv[(size_t)(i + 1) * conv_dim + 2 * key_dim + (size_t)vh * GDN_REC_HV + dv]);
        }

        // ④ S += k̂⊗delta；o = Σ S_new·q̂
        float acc = 0.0f;
#pragma unroll
        for (int d = 0; d < GDN_REC_HK; d += 4) {
            const float4 kn = *(const float4*)(sKN + d);
            const float4 qn = *(const float4*)(sQN + d);
            const float u0 = st[d] + kn.x * delta;
            const float u1 = st[d + 1] + kn.y * delta;
            const float u2 = st[d + 2] + kn.z * delta;
            const float u3 = st[d + 3] + kn.w * delta;
            st[d] = u0;
            st[d + 1] = u1;
            st[d + 2] = u2;
            st[d + 3] = u3;
            acc += u0 * qn.x;
            acc += u1 * qn.y;
            acc += u2 * qn.z;
            acc += u3 * qn.w;
        }

        // ⑤ RMSNormGated（与旧核同一 shuffle 树 / 同一 ssum 复用）
        float sq = acc * acc;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, off);
        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        if (lane == 0) ssum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            const int nwarp = (blockDim.x + 31) >> 5;
            float v2 = (lane < nwarp) ? ssum[lane] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v2 += __shfl_down_sync(0xffffffffu, v2, off);
            if (lane == 0) ssum[0] = v2;
        }
        __syncthreads();
        const float r = 1.0f / sqrtf(ssum[0] / (float)GDN_REC_HV + 1e-6f);

        // 补3：z 用旋转寄存器（表达式与旧核逐字相同）。
        float zz;
        if (prefetch) {
            zz = zz_c;
        } else {
            zz = __bfloat162float(z[(size_t)i * value_dim + (size_t)vh * GDN_REC_HV + dv]);
        }
        const float actv = (gate_act == 0) ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
        gated[(size_t)i * value_dim + (size_t)vh * GDN_REC_HV + dv] =
            __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
        // 补3：zz 消费完毕，预载下一拍的 z。
        if (prefetch && i + 1 < T) {
            zz_c = __bfloat162float(z[(size_t)(i + 1) * value_dim + (size_t)vh * GDN_REC_HV + dv]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int d = 0; d < GDN_REC_HK; ++d) S[(size_t)d * GDN_REC_HV + dv] = st[d];
}

// ㉟+1 GDN 递推 **多 warp 占用率版**（2026-10-22，第二十一轮，M5 便宜形态）。
//
// 立项依据（PLAN_NEXT .A.1 总原则"算力用满" + M5 重开条件）：旧核 grid=48 block、
// 每 block 仅 hv 线程（4 warp、~6% 占用），T 步串行的每步两次跨 dk 归约全压在一个
// warp 的访存延迟上（P15-㉘ 批量预载只把在飞 load 提到 16，warp 数没变）。递推的
// 跨 token 依赖是本质的，但**跨 dk 是并行的**——本核把 dk 维拆到 SPL=4 组线程
// （block = hv×SPL = 512 线程 = 16 warp），每步三次跨 split 归约走 shared：
// * 组内仍按 dk 升序串行累加（同旧核的寄存器次序）；
// * 跨 split 按 split 编号升序相加 ⇒ 确定性的**有损档**（归约次序与旧核不同，
// PLAN_NEXT 判据：首 token 相等 + 质量抽检）。
// 同步账：每步 3 次 __syncthreads（≈百 ns 级），换 4× 在飞访存。
// 只支持 n_seq==1（prefill 形态；decode 的并发批走旧核）。S 直接全局读写
// （㉘ 实测 smem 驻留档在批量预载后无收益，不再保留该分支）。
template <int SPL>
__device__ __forceinline__ void gdn_recurrent_mw_body(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int hk, int hv, int num_v_heads,
    int num_k_heads, int gate_act) {
    const int nvh = num_v_heads;
    const int vh = blockIdx.x % nvh;
    const int dv = threadIdx.x % hv;   // 列
    const int sp = threadIdx.x / hv;   // dk 分组
    const int key_dim = num_k_heads * hk;
    const int value_dim = nvh * hv;
    const int conv_dim = 2 * key_dim + value_dim;
    const int rep = num_v_heads / num_k_heads;
    const int hh = vh / rep;

    // 本 split 的 dk 区间 [d0, d1)。
    const int d0 = sp * hk / SPL;
    const int d1 = (sp + 1) * hk / SPL;

    __shared__ float red[3][SPL][128]; // 每步三次归约的部分和（hv ≤ 128）
    __shared__ float ssum[32];

    float* S = s_all + (size_t)vh * hk * hv;

    for (int i = 0; i < T; ++i) {
        const bf16* conv_i = conv + (size_t)i * conv_dim;
        const bf16* qrow = conv_i + (size_t)hh * hk;
        const bf16* krow = conv_i + key_dim + (size_t)hh * hk;

        // ---- 头内 l2 范数：split 部分和 → 升序合并 ----
        float qss = 0.0f, kss = 0.0f;
        {
            int d = d0;
            for (; d + 16 <= d1; d += 16) {
                float qv[16], kv[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    qv[u] = __bfloat162float(qrow[d + u]);
                    kv[u] = __bfloat162float(krow[d + u]);
                }
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    qss += qv[u] * qv[u];
                    kss += kv[u] * kv[u];
                }
            }
            for (; d < d1; ++d) {
                const float qv = __bfloat162float(qrow[d]);
                const float kv = __bfloat162float(krow[d]);
                qss += qv * qv;
                kss += kv * kv;
            }
        }
        red[0][sp][dv] = qss;
        red[1][sp][dv] = kss;
        __syncthreads();
        qss = red[0][0][dv];
        kss = red[1][0][dv];
#pragma unroll
        for (int s = 1; s < SPL; ++s) {
            qss += red[0][s][dv];
            kss += red[1][s][dv];
        }
        const float qscale = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)hk);
        const float kscale = 1.0f / sqrtf(kss + 1e-6f);

        const float av = __bfloat162float(agate[(size_t)i * nvh + vh]) + dt_bias[vh];
        const float spg = (av > 20.0f) ? av : log1pf(expf(av));
        const float decay = expf(-expf(a_log[vh]) * spg);
        const float bv = __bfloat162float(bgate[(size_t)i * nvh + vh]);
        const float beta = 1.0f / (1.0f + __expf(-bv));

        // ---- S *= decay；kv_mem = Σ S·k̂（split 部分和 → 升序合并）----
        float kv_mem = 0.0f;
        {
            int d = d0;
            for (; d + 16 <= d1; d += 16) {
                float sv[16], kn[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    sv[u] = S[(size_t)(d + u) * hv + dv];
                    kn[u] = __bfloat162float(krow[d + u]) * kscale;
                }
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const float s = sv[u] * decay;
                    S[(size_t)(d + u) * hv + dv] = s;
                    kv_mem += s * kn[u];
                }
            }
            for (; d < d1; ++d) {
                const float s = S[(size_t)d * hv + dv] * decay;
                S[(size_t)d * hv + dv] = s;
                kv_mem += s * (__bfloat162float(krow[d]) * kscale);
            }
        }
        red[2][sp][dv] = kv_mem;
        __syncthreads();
        kv_mem = red[2][0][dv];
#pragma unroll
        for (int s = 1; s < SPL; ++s) {
            kv_mem += red[2][s][dv];
        }

        const float v = __bfloat162float(conv_i[2 * key_dim + (size_t)vh * hv + dv]);
        const float delta = (v - kv_mem) * beta;

        // ---- S += k̂⊗delta；acc = Σ S_new·q̂（split 部分和 → 升序合并）----
        float acc = 0.0f;
        {
            int d = d0;
            for (; d + 16 <= d1; d += 16) {
                float su0[16], qn[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    su0[u] = S[(size_t)(d + u) * hv + dv];
                    qn[u] = __bfloat162float(qrow[d + u]) * qscale;
                }
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const float su = su0[u] + __bfloat162float(krow[d + u]) * kscale * delta;
                    S[(size_t)(d + u) * hv + dv] = su;
                    acc += su * qn[u];
                }
            }
            for (; d < d1; ++d) {
                const float kn = __bfloat162float(krow[d]) * kscale;
                const float su = S[(size_t)d * hv + dv] + kn * delta;
                S[(size_t)d * hv + dv] = su;
                acc += su * (__bfloat162float(qrow[d]) * qscale);
            }
        }
        // 复用 red[0] 做第三次归约（此刻 red[0] 的 qss 已被全体读完，上一轮同步后无人再读）
        red[0][sp][dv] = acc;
        __syncthreads();
        acc = red[0][0][dv];
#pragma unroll
        for (int s = 1; s < SPL; ++s) {
            acc += red[0][s][dv];
        }

        // ---- RMSNormGated：跨 dv 归约（仅 sp==0 的 hv 个线程参与，与旧核同构）----
        if (sp == 0) {
            float sq = acc * acc;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, off);
            const int lane = dv & 31;
            const int wid = dv >> 5;
            if (lane == 0) ssum[wid] = sq;
        }
        __syncthreads();
        if (sp == 0 && dv < 32) {
            const int nwarp = hv >> 5;
            float v2 = (dv < nwarp) ? ssum[dv] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v2 += __shfl_down_sync(0xffffffffu, v2, off);
            if (dv == 0) ssum[0] = v2;
        }
        __syncthreads();
        const float r = 1.0f / sqrtf(ssum[0] / (float)hv + 1e-6f);

        if (sp == 0) {
            const float zz = __bfloat162float(z[(size_t)i * value_dim + (size_t)vh * hv + dv]);
            const float actv = (gate_act == 0) ? (zz / (1.0f + __expf(-zz)))
                                               : (1.0f / (1.0f + __expf(-zz)));
            gated[(size_t)i * value_dim + (size_t)vh * hv + dv] =
                __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
        }
        // red[0] 下轮复用前必须全体读完（上面 sp==0 分支已sync过，这里再补一个统一尾部）
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(512, 1)
lingxi_gdn_recurrent_mw_bf16(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int hk, int hv, int num_v_heads,
    int num_k_heads, int gate_act) {
    gdn_recurrent_mw_body<4>(conv, z, agate, bgate, norm_w, a_log, dt_bias, s_all,
                             gated, T, hk, hv, num_v_heads, num_k_heads, gate_act);
}
