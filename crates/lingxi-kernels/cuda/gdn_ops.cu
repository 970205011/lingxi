// L1 GDN（线性注意力）相关算子。

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// 深度可分离因果 1D 卷积 + silu：x[T, C], w[K, C]（行主序，第 k 行是第 k 个 tap）
//   y[t][c] = silu( Σ_k x[t-(K-1-k)*D][c] * w[k][c] )   越界视为 0
//   D = 膨胀系数（GDN 卷积 D=1；PLE 的 short conv D=ngram_size=3）
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
//   decay = exp(-exp(a_log[vh]) * softplus(a[i,vh] + dt_bias[vh]))     beta = sigmoid(b[i,vh])
//   S *= decay ; kv_mem[dv] = Σ_dk S[dk][dv]·k̂[dk]
//   delta[dv] = (v[dv] - kv_mem[dv])·beta ; S[dk][dv] += k̂[dk]·delta[dv]
//   o[dv] = Σ_dk S[dk][dv]·q̂[dk]
//   gated = o · rsqrt(mean(o²)+eps) · norm_w[dv] · act(z[dv])
// 其中 q̂ = q / ‖q_head‖ / √hk ，k̂ = k / ‖k_head‖（头内 l2，q/k 按 k-head 分，rep 复制到 v-head）。
// `s_all` 是**持久设备状态**，就地推进；`gate_act`：0=silu 1=sigmoid。
// ★ P15-㉘（2026-10-17）：`use_smem` 时把本 v-head 的 S（hk×hv fp32，本模型 64KB）
// 一次性搬进**动态 shared memory 驻留**，T 循环全程打 smem，结束再写回全局。
// 收益账：S 原来每 token 被全局内存读+写共 4 遍（decay 读写 + update 读写），
// T=6 一轮 = 4×3MB×6 ≈ 75MB/层；smem 驻留后每轮只有 in/out 各一次 = 6MB，砍 12×。
// **逐位等价**：每个线程的 dk 串行累加次序、归约结构、写回值全部不变，
// 只是 S 的驻留位置从全局换成 smem（初值同一字节、写回同一字节）。
// ★ P15-㉟ 并发批（P1）：`n_seq > 1` 时一个 batch 里有 N 条序列各推 1 行。
//   - 状态布局变 seq-major：`s_all[(seq*num_v_heads + vh) * hk*hv]`（调用方把 N 条的
//     GdnState.s_all 连续排列，或传各自的基址差固定 stride）。
//   - 输入行按 seq 交错：`conv[(i*n_seq + seq) * conv_dim]`（i 是本步内的行序号）。
//   - 输出同理：`gated[(i*n_seq + seq) * value_dim]`。
//   `n_seq==1` 时全部退化为老布局 ⇒ **逐位等价**（单序列调用传 1 即可）。
extern "C" __global__ void lingxi_gdn_recurrent_bf16(
    const bf16* __restrict__ conv,     // [T*n_seq, conv_dim]（已过 silu 的因果卷积输出）
    const bf16* __restrict__ z,        // [T*n_seq, value_dim]
    const bf16* __restrict__ agate,    // [T*n_seq, num_v_heads]  (a)
    const bf16* __restrict__ bgate,    // [T*n_seq, num_v_heads]  (b)
    const bf16* __restrict__ norm_w,   // [hv]（bf16 原值，f32 乘）
    const float* __restrict__ a_log,   // [num_v_heads]
    const float* __restrict__ dt_bias, // [num_v_heads]
    float* __restrict__ s_all,         // [n_seq, num_v_heads, hk*hv] 持久状态（就地，seq-major）
    bf16* __restrict__ gated,          // [T*n_seq, value_dim]
    int T, int hk, int hv, int num_v_heads, int num_k_heads, int gate_act, int row_off,
    int use_smem, int n_seq) {
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

        // S *= decay ；同时算 kv_mem = Σ_dk S[dk][dv]·k̂[dk]
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
        // S += k̂ ⊗ delta ；同时算 o = Σ_dk S[dk][dv]·q̂[dk]（用**更新后**的 S）
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
