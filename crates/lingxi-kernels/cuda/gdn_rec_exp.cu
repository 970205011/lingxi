// **实验核**（2026-10-24）：GDN 递推 `gdn_recurrent_bf16`（t=6）的**分母探针** +
// 一个"state 全程寄存器"候选。只有 `tests/t16_gdn_rec_bw.rs` 会调；生产核一个字节没动。
//
// 背景：`rec` 是 36/48 层都要跑的核（层均 78.6~92.5 µs），以前被判"到顶"的依据是
// "bit-exact + 行串行 + smem 档更差"，**但从来没有量过同形纯读/纯搬运的分母**。
//
// 字节账（本模型 48 v-head × hk128 × hv128 × f32 = 3.00 MiB）：
// 生产核每 token 对 S 做 **2 读 + 2 写**（decay 遍读+写、update 遍读+写）
// ⇒ 4 × 3.00 MiB × 6 token = **72 MiB/层**
// k/v/q 流量：每 block 每 token 读 q(128) + k(2×128) + v(128) ≈ 0.9 KB ⇒ 48 block × 6 ≈ 0.26 MB
// z + gated：每 token 12 KB + 12 KB ⇒ 0.15 MB
// ⇒ 生产核 **~76 MB/层**，其中 S 占 99%。
//
// 探针（grid=48、block=128，与生产核逐字节相同的地址映射；hk=hv=128）：
// 0 prod 生产核逐字复刻（自检：应与 `lingxi_gdn_recurrent_bf16` 逐位相同）
// 1 read2 纯读 S 两遍（同形地址流，只读不写）
// 2 read1 纯读 S 一遍
// 3 copy2 就地 2R+2W 纯搬运（同形地址流，不做递推）
// 4 copy1 就地 1R+1W 纯搬运（整段一遍 = 6.00 MiB）
// 5 reg **候选**：state 全程驻留寄存器，每个 head 只 1 读 + 1 写（6.00 MiB 总）
// 6 reg_nos 候选但不碰 S（纯算 + q/k + 输出 ⇒ ALU/发射地板）
// 7 reg_tok 候选但**每 token 把 state 写回一次**（只写不读 ⇒ 量"逐 token 写回"的代价）
//
// 逐位：5/6/7 的算式与生产核**逐字相同**（同一表达式、同一累加次序 dk 升序、同一 rk 归约），
// 只把 `S[(dk)*hv+dv]` 的读写位置从 global 换成寄存器。6 不写 s_all，仅用于计时。

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

#define HK128 128
#define HV128 128

__device__ __forceinline__ float gdn_l2_scale(float ss, int hk, int q) {
    return q ? ((1.0f / sqrtf(ss + 1e-6f)) / sqrtf((float)hk)) : (1.0f / sqrtf(ss + 1e-6f));
}

// ───────────────────────── 0：生产核逐字复刻 ─────────────────────────
// 与 `gdn_ops.cu::lingxi_gdn_recurrent_bf16` 的单序列、`use_smem=0` 分支逐字对应。
extern "C" __global__ void lingxi_gdn_rec_prod(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int hk, int hv, int num_v_heads,
    int num_k_heads, int gate_act) {
    const int nvh = num_v_heads;
    const int vh = blockIdx.x % nvh;
    const int dv = threadIdx.x;
    const int key_dim = num_k_heads * hk;
    const int value_dim = nvh * hv;
    const int conv_dim = 2 * key_dim + value_dim;
    const int rep = nvh / num_k_heads;
    const int hh = vh / rep;

    __shared__ float ssum[32];
    float* S = s_all + (size_t)vh * hk * hv;

    for (int i = 0; i < T; ++i) {
        const bf16* conv_i = conv + (size_t)i * conv_dim;
        const bf16* qrow = conv_i + (size_t)hh * hk;
        const bf16* krow = conv_i + key_dim + (size_t)hh * hk;

        float qss = 0.0f, kss = 0.0f;
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

        const float av = __bfloat162float(agate[(size_t)i * num_v_heads + vh]) + dt_bias[vh];
        const float sp = (av > 20.0f) ? av : log1pf(expf(av));
        const float decay = expf(-expf(a_log[vh]) * sp);
        const float bv = __bfloat162float(bgate[(size_t)i * num_v_heads + vh]);
        const float beta = 1.0f / (1.0f + expf(-bv));

        float kv_mem = 0.0f;
        {
            int dk = 0;
            for (; dk + 16 <= hk; dk += 16) {
                float sv[16], kn[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    sv[u] = S[(size_t)(dk + u) * hv + dv];
                    kn[u] = __bfloat162float(krow[dk + u]) * kscale;
                }
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const float s = sv[u] * decay;
                    S[(size_t)(dk + u) * hv + dv] = s;
                    kv_mem += s * kn[u];
                }
            }
            for (; dk < hk; ++dk) {
                const float s = S[(size_t)dk * hv + dv] * decay;
                S[(size_t)dk * hv + dv] = s;
                kv_mem += s * (__bfloat162float(krow[dk]) * kscale);
            }
        }
        const float v = __bfloat162float(conv_i[2 * key_dim + (size_t)vh * hv + dv]);
        const float delta = (v - kv_mem) * beta;
        float acc = 0.0f;
        {
            int dk = 0;
            for (; dk + 16 <= hk; dk += 16) {
                float su0[16], qn[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    su0[u] = S[(size_t)(dk + u) * hv + dv];
                    qn[u] = __bfloat162float(qrow[dk + u]) * qscale;
                }
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const float su = su0[u] + __bfloat162float(krow[dk + u]) * kscale * delta;
                    S[(size_t)(dk + u) * hv + dv] = su;
                    acc += su * qn[u];
                }
            }
            for (; dk < hk; ++dk) {
                const float kn = __bfloat162float(krow[dk]) * kscale;
                const float su = S[(size_t)dk * hv + dv] + kn * delta;
                S[(size_t)dk * hv + dv] = su;
                acc += su * (__bfloat162float(qrow[dk]) * qscale);
            }
        }

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

        const float zz = __bfloat162float(z[(size_t)i * value_dim + (size_t)vh * hv + dv]);
        const float actv = (gate_act == 0) ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
        gated[(size_t)i * value_dim + (size_t)vh * hv + dv] =
            __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
        __syncthreads();
    }
}

// ───────────────────────── 1/2：同形纯读 ─────────────────────────
// 地址流与生产核的 S 访问完全一致（每 token 两遍、dk 升序、16 批量）。
extern "C" __global__ void lingxi_gdn_rec_read2(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int hk, int hv, int num_v_heads,
    int num_k_heads, int gate_act) {
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    const float* S = s_all + (size_t)vh * hk * hv;
    float acc = 0.0f;
    for (int i = 0; i < T; ++i) {
        for (int pass = 0; pass < 2; ++pass) {
            int dk = 0;
            for (; dk + 16 <= hk; dk += 16) {
                float sv[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) sv[u] = S[(size_t)(dk + u) * hv + dv];
#pragma unroll
                for (int u = 0; u < 16; ++u) acc += sv[u];
            }
            for (; dk < hk; ++dk) acc += S[(size_t)dk * hv + dv];
        }
    }
    if (acc == 1.2345e-31f) gated[threadIdx.x] = __float2bfloat16(acc);
}

extern "C" __global__ void lingxi_gdn_rec_read1(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int hk, int hv, int num_v_heads,
    int num_k_heads, int gate_act) {
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    const float* S = s_all + (size_t)vh * hk * hv;
    float acc = 0.0f;
    for (int i = 0; i < T; ++i) {
        int dk = 0;
        for (; dk + 16 <= hk; dk += 16) {
            float sv[16];
#pragma unroll
            for (int u = 0; u < 16; ++u) sv[u] = S[(size_t)(dk + u) * hv + dv];
#pragma unroll
            for (int u = 0; u < 16; ++u) acc += sv[u];
        }
        for (; dk < hk; ++dk) acc += S[(size_t)dk * hv + dv];
    }
    if (acc == 1.2345e-31f) gated[threadIdx.x] = __float2bfloat16(acc);
}

// ───────────────────────── 3/4：同形纯搬运（无递推） ─────────────────────────
extern "C" __global__ void lingxi_gdn_rec_copy2(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int hk, int hv, int num_v_heads,
    int num_k_heads, int gate_act) {
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    float* S = s_all + (size_t)vh * hk * hv;
    for (int i = 0; i < T; ++i) {
        const float f = 1.0f / (1.0f + (float)i);  // 与数据无关，只为挡住常量折叠
        for (int pass = 0; pass < 2; ++pass) {
            int dk = 0;
            for (; dk + 16 <= hk; dk += 16) {
                float sv[16];
#pragma unroll
                for (int u = 0; u < 16; ++u) sv[u] = S[(size_t)(dk + u) * hv + dv];
#pragma unroll
                for (int u = 0; u < 16; ++u) S[(size_t)(dk + u) * hv + dv] = sv[u] * f;
            }
            for (; dk < hk; ++dk) S[(size_t)dk * hv + dv] *= f;
        }
    }
}

extern "C" __global__ void lingxi_gdn_rec_copy1(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int hk, int hv, int num_v_heads,
    int num_k_heads, int gate_act) {
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    float* S = s_all + (size_t)vh * hk * hv;
    const float f = 1.0f / (1.0f + (float)T);
    int dk = 0;
    for (; dk + 16 <= hk; dk += 16) {
        float sv[16];
#pragma unroll
        for (int u = 0; u < 16; ++u) sv[u] = S[(size_t)(dk + u) * hv + dv];
#pragma unroll
        for (int u = 0; u < 16; ++u) S[(size_t)(dk + u) * hv + dv] = sv[u] * f;
    }
    for (; dk < hk; ++dk) S[(size_t)dk * hv + dv] *= f;
}

// ───────────────────────── 5/6/7：候选（state 常驻寄存器） ─────────────────────────
// MODE 0 = 只在开头读一次、结尾写一次（候选）；1 = 完全不碰 S（地板）；2 = 每 token 只写回。
template <int MODE>
__device__ __forceinline__ void gdn_rec_reg_body(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    const int nvh = num_v_heads;
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    const int key_dim = num_k_heads * HK128;
    const int value_dim = nvh * HV128;
    const int conv_dim = 2 * key_dim + value_dim;
    const int rep = nvh / num_k_heads;
    const int hh = vh / rep;

    __shared__ float ssum[32];
    float* S = s_all + (size_t)vh * HK128 * HV128;

    // 本线程的一整列 state（hk 个 f32）驻留寄存器
    float st[HK128];
    if (MODE != 1) {
#pragma unroll
        for (int d = 0; d < HK128; ++d) st[d] = S[(size_t)d * HV128 + dv];
    } else {
#pragma unroll
        for (int d = 0; d < HK128; ++d) st[d] = 0.0f;
    }

    for (int i = 0; i < T; ++i) {
        const bf16* conv_i = conv + (size_t)i * conv_dim;
        const bf16* qrow = conv_i + (size_t)hh * HK128;
        const bf16* krow = conv_i + key_dim + (size_t)hh * HK128;

        float qss = 0.0f, kss = 0.0f;
        {
            int d = 0;
            for (; d + 16 <= HK128; d += 16) {
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
        }
        const float qscale = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)HK128);
        const float kscale = 1.0f / sqrtf(kss + 1e-6f);

        const float av = __bfloat162float(agate[(size_t)i * nvh + vh]) + dt_bias[vh];
        const float sp = (av > 20.0f) ? av : log1pf(expf(av));
        const float decay = expf(-expf(a_log[vh]) * sp);
        const float bv = __bfloat162float(bgate[(size_t)i * nvh + vh]);
        const float beta = 1.0f / (1.0f + expf(-bv));

        float kv_mem = 0.0f;
#pragma unroll
        for (int d = 0; d < HK128; ++d) {
            const float sv = st[d] * decay;
            st[d] = sv;
            kv_mem += sv * (__bfloat162float(krow[d]) * kscale);
        }
        const float v = __bfloat162float(conv_i[2 * key_dim + (size_t)vh * HV128 + dv]);
        const float delta = (v - kv_mem) * beta;

        float acc = 0.0f;
#pragma unroll
        for (int d = 0; d < HK128; ++d) {
            const float su = st[d] + __bfloat162float(krow[d]) * kscale * delta;
            st[d] = su;
            acc += su * (__bfloat162float(qrow[d]) * qscale);
        }

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
        const float r = 1.0f / sqrtf(ssum[0] / (float)HV128 + 1e-6f);

        const float zz = __bfloat162float(z[(size_t)i * value_dim + (size_t)vh * HV128 + dv]);
        const float actv = (gate_act == 0) ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
        gated[(size_t)i * value_dim + (size_t)vh * HV128 + dv] =
            __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
        if (MODE == 2) {
#pragma unroll
            for (int d = 0; d < HK128; ++d) S[(size_t)d * HV128 + dv] = st[d];
        }
        __syncthreads();
    }
    if (MODE == 0) {
#pragma unroll
        for (int d = 0; d < HK128; ++d) S[(size_t)d * HV128 + dv] = st[d];
    }
}

extern "C" __global__ void __launch_bounds__(HV128, 1) lingxi_gdn_rec_reg(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    gdn_rec_reg_body<0>(conv, z, agate, bgate, norm_w, a_log, dt_bias, s_all, gated,
                        T, num_v_heads, num_k_heads, gate_act);
}

extern "C" __global__ void __launch_bounds__(HV128, 1) lingxi_gdn_rec_reg_nos(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    gdn_rec_reg_body<1>(conv, z, agate, bgate, norm_w, a_log, dt_bias, s_all, gated,
                        T, num_v_heads, num_k_heads, gate_act);
}

extern "C" __global__ void __launch_bounds__(HV128, 1) lingxi_gdn_rec_reg_tok(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    gdn_rec_reg_body<2>(conv, z, agate, bgate, norm_w, a_log, dt_bias, s_all, gated,
                        T, num_v_heads, num_k_heads, gate_act);
}

// ───────────────────────── 8/9：smem 驻留（1R+1W 全局） ─────────────────────────
// MODE 0 = decay 在第二遍**重算**（乘法确定性 ⇒ 与"存下来再读"逐位相同），smem 只 2R+1W；
// MODE 1 = 老档（smem 2R+2W，复刻 `use_smem=1` 的 117.7 µs 口径）。
template <int MODE>
__device__ __forceinline__ void gdn_rec_smem_body(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    const int nvh = num_v_heads;
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    const int key_dim = num_k_heads * HK128;
    const int value_dim = nvh * HV128;
    const int conv_dim = 2 * key_dim + value_dim;
    const int rep = nvh / num_k_heads;
    const int hh = vh / rep;

    __shared__ float ssum[32];
    extern __shared__ float sS[];
    float* S = s_all + (size_t)vh * HK128 * HV128;
    for (int idx = threadIdx.x; idx < HK128 * HV128; idx += blockDim.x) sS[idx] = S[idx];
    __syncthreads();

    for (int i = 0; i < T; ++i) {
        const bf16* conv_i = conv + (size_t)i * conv_dim;
        const bf16* qrow = conv_i + (size_t)hh * HK128;
        const bf16* krow = conv_i + key_dim + (size_t)hh * HK128;

        float qss = 0.0f, kss = 0.0f;
        {
            int d = 0;
            for (; d + 16 <= HK128; d += 16) {
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
        }
        const float qscale = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)HK128);
        const float kscale = 1.0f / sqrtf(kss + 1e-6f);

        const float av = __bfloat162float(agate[(size_t)i * nvh + vh]) + dt_bias[vh];
        const float sp = (av > 20.0f) ? av : log1pf(expf(av));
        const float decay = expf(-expf(a_log[vh]) * sp);
        const float bv = __bfloat162float(bgate[(size_t)i * nvh + vh]);
        const float beta = 1.0f / (1.0f + expf(-bv));

        float kv_mem = 0.0f;
#pragma unroll 16
        for (int d = 0; d < HK128; ++d) {
            const float sv = sS[(size_t)d * HV128 + dv] * decay;
            if (MODE == 1) sS[(size_t)d * HV128 + dv] = sv;
            kv_mem += sv * (__bfloat162float(krow[d]) * kscale);
        }
        const float v = __bfloat162float(conv_i[2 * key_dim + (size_t)vh * HV128 + dv]);
        const float delta = (v - kv_mem) * beta;

        float acc = 0.0f;
#pragma unroll 16
        for (int d = 0; d < HK128; ++d) {
            const float s1 = __fmul_rn(sS[(size_t)d * HV128 + dv], decay);  // 显式舍入，禁止收缩成 FMA
            const float su0 = (MODE == 1) ? sS[(size_t)d * HV128 + dv] : s1;
            const float su = su0 + __bfloat162float(krow[d]) * kscale * delta;
            sS[(size_t)d * HV128 + dv] = su;
            acc += su * (__bfloat162float(qrow[d]) * qscale);
        }

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
        const float r = 1.0f / sqrtf(ssum[0] / (float)HV128 + 1e-6f);

        const float zz = __bfloat162float(z[(size_t)i * value_dim + (size_t)vh * HV128 + dv]);
        const float actv = (gate_act == 0) ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
        gated[(size_t)i * value_dim + (size_t)vh * HV128 + dv] =
            __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
        __syncthreads();
    }
    for (int idx = threadIdx.x; idx < HK128 * HV128; idx += blockDim.x) S[idx] = sS[idx];
}

extern "C" __global__ void __launch_bounds__(HV128, 1) lingxi_gdn_rec_smem(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    gdn_rec_smem_body<0>(conv, z, agate, bgate, norm_w, a_log, dt_bias, s_all, gated,
                         T, num_v_heads, num_k_heads, gate_act);
}

extern "C" __global__ void __launch_bounds__(HV128, 1) lingxi_gdn_rec_smem_old(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    gdn_rec_smem_body<1>(conv, z, agate, bgate, norm_w, a_log, dt_bias, s_all, gated,
                         T, num_v_heads, num_k_heads, gate_act);
}

// ───────────────── 11/12：q/k 走 smem（scaled 版预先算好）+ state 半寄存器 ─────────────────
// 动机：reg 档（探针 6）只剩 61 µs、且"完全不碰 S"的地板仍 55.5 µs ⇒ 递推的**算式/取数延迟**
// 才是大头，S 流量只值 ~6 µs。每 token 每个线程原本要读 3 遍 q/k（qss 遍 + 两遍递推）
// = 384 次广播 load + 384 次 cvt；改成：每 token 每个线程只从 global 读 1 个 q + 1 个 k
// 填进 smem（raw + scaled 各一份），其余全走 smem。
// HYB=0：state 全寄存器（128 个 f32/线程，ptxas 会溢出 ~83 个到 local）
// HYB=1：state 前 64 个 dk 驻留寄存器、后 64 个驻留 smem（每 token smem 2R+1W、无溢出）
// VEC=1：q/k 的 smem 读改成 float4（每 4 个 dk 一次 LDS.128，广播地址 ⇒ 无冲突）
template <int HYB, int VEC>
__device__ __forceinline__ void gdn_rec_v2_body(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    const int nvh = num_v_heads;
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    const int key_dim = num_k_heads * HK128;
    const int value_dim = nvh * HV128;
    const int conv_dim = 2 * key_dim + value_dim;
    const int rep = nvh / num_k_heads;
    const int hh = vh / rep;

    __shared__ float ssum[32];
    __shared__ float sQ[HV128], sK[HV128], sQN[HV128], sKN[HV128];
    extern __shared__ float sS[];  // HYB=1：后 64 个 dk（32 KB）
    float* S = s_all + (size_t)vh * HK128 * HV128;

    float st[HYB ? 64 : HK128];
#pragma unroll
    for (int d = 0; d < 64; ++d) st[d] = S[(size_t)d * HV128 + dv];
    if (HYB) {
#pragma unroll
        for (int d = 64; d < HK128; ++d) sS[(size_t)(d - 64) * HV128 + dv] = S[(size_t)d * HV128 + dv];
    } else {
#pragma unroll
        for (int d = 64; d < HK128; ++d) st[d] = S[(size_t)d * HV128 + dv];
    }

    for (int i = 0; i < T; ++i) {
        const bf16* conv_i = conv + (size_t)i * conv_dim;
        const bf16* qrow = conv_i + (size_t)hh * HK128;
        const bf16* krow = conv_i + key_dim + (size_t)hh * HK128;

        // ① 每个线程只读 1 个 q + 1 个 k（原来每线程 384 次广播 load）
        sQ[dv] = __bfloat162float(qrow[dv]);
        sK[dv] = __bfloat162float(krow[dv]);
        __syncthreads();

        // ② 头内 l2 范数：dk 升序（与生产核同一累加次序）
        float qss = 0.0f, kss = 0.0f;
        if (VEC) {
#pragma unroll
            for (int d = 0; d < HK128; d += 4) {
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
        } else {
#pragma unroll 16
            for (int d = 0; d < HK128; ++d) {
                const float qv = sQ[d];
                const float kv = sK[d];
                qss += qv * qv;
                kss += kv * kv;
            }
        }
        const float qscale = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)HK128);
        const float kscale = 1.0f / sqrtf(kss + 1e-6f);
        sQN[dv] = sQ[dv] * qscale;
        sKN[dv] = sK[dv] * kscale;
        __syncthreads();

        const float av = __bfloat162float(agate[(size_t)i * nvh + vh]) + dt_bias[vh];
        const float sp = (av > 20.0f) ? av : log1pf(expf(av));
        const float decay = expf(-expf(a_log[vh]) * sp);
        const float bv = __bfloat162float(bgate[(size_t)i * nvh + vh]);
        const float beta = 1.0f / (1.0f + expf(-bv));

        float kv_mem = 0.0f;
        if (HYB) {
#pragma unroll
            for (int d = 0; d < 64; ++d) {
                const float sv = st[d] * decay;
                st[d] = sv;
                kv_mem += sv * sKN[d];
            }
#pragma unroll
            for (int d = 64; d < HK128; ++d) {
                const float sv = sS[(size_t)(d - 64) * HV128 + dv] * decay;
                kv_mem += sv * sKN[d];
            }
        } else if (VEC) {
#pragma unroll
            for (int d = 0; d < HK128; d += 4) {
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
        } else {
#pragma unroll
            for (int d = 0; d < HK128; ++d) {
                const float sv = st[d] * decay;
                st[d] = sv;
                kv_mem += sv * sKN[d];
            }
        }
        const float v = __bfloat162float(conv_i[2 * key_dim + (size_t)vh * HV128 + dv]);
        const float delta = (v - kv_mem) * beta;

        float acc = 0.0f;
        if (HYB) {
#pragma unroll
            for (int d = 0; d < 64; ++d) {
                const float su = st[d] + sKN[d] * delta;
                st[d] = su;
                acc += su * sQN[d];
            }
#pragma unroll
            for (int d = 64; d < HK128; ++d) {
                const float s1 = __fmul_rn(sS[(size_t)(d - 64) * HV128 + dv], decay);  // 显式舍入
                const float su = s1 + sKN[d] * delta;
                sS[(size_t)(d - 64) * HV128 + dv] = su;
                acc += su * sQN[d];
            }
        } else if (VEC) {
#pragma unroll
            for (int d = 0; d < HK128; d += 4) {
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
        } else {
#pragma unroll
            for (int d = 0; d < HK128; ++d) {
                const float su = st[d] + sKN[d] * delta;
                st[d] = su;
                acc += su * sQN[d];
            }
        }

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
        const float r = 1.0f / sqrtf(ssum[0] / (float)HV128 + 1e-6f);

        const float zz = __bfloat162float(z[(size_t)i * value_dim + (size_t)vh * HV128 + dv]);
        const float actv = (gate_act == 0) ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
        gated[(size_t)i * value_dim + (size_t)vh * HV128 + dv] =
            __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
        __syncthreads();
    }

#pragma unroll
    for (int d = 0; d < 64; ++d) S[(size_t)d * HV128 + dv] = st[d];
    if (HYB) {
#pragma unroll
        for (int d = 64; d < HK128; ++d) S[(size_t)d * HV128 + dv] = sS[(size_t)(d - 64) * HV128 + dv];
    } else {
#pragma unroll
        for (int d = 64; d < HK128; ++d) S[(size_t)d * HV128 + dv] = st[d];
    }
}

extern "C" __global__ void __launch_bounds__(HV128, 1) lingxi_gdn_rec_v2(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    gdn_rec_v2_body<0, 0>(conv, z, agate, bgate, norm_w, a_log, dt_bias, s_all, gated,
                          T, num_v_heads, num_k_heads, gate_act);
}

extern "C" __global__ void __launch_bounds__(HV128, 1) lingxi_gdn_rec_v3(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    gdn_rec_v2_body<0, 1>(conv, z, agate, bgate, norm_w, a_log, dt_bias, s_all, gated,
                          T, num_v_heads, num_k_heads, gate_act);
}

extern "C" __global__ void __launch_bounds__(HV128, 1) lingxi_gdn_rec_hyb(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    gdn_rec_v2_body<1, 0>(conv, z, agate, bgate, norm_w, a_log, dt_bias, s_all, gated,
                          T, num_v_heads, num_k_heads, gate_act);
}

// ───────────────── 13：消融（q/k 全用常量，量化"取数延迟"占多少） ─────────────────
extern "C" __global__ void __launch_bounds__(HV128, 1) lingxi_gdn_rec_nok(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated, int T, int num_v_heads, int num_k_heads,
    int gate_act) {
    const int nvh = num_v_heads;
    const int vh = blockIdx.x;
    const int dv = threadIdx.x;
    const int value_dim = nvh * HV128;

    __shared__ float ssum[32];
    float* S = s_all + (size_t)vh * HK128 * HV128;

    float st[HK128];
#pragma unroll
    for (int d = 0; d < HK128; ++d) st[d] = S[(size_t)d * HV128 + dv];

    for (int i = 0; i < T; ++i) {
        float qss = 0.0f, kss = 0.0f;
#pragma unroll 16
        for (int d = 0; d < HK128; ++d) {
            const float qv = 0.25f + (float)(d & 3);
            const float kv = 0.5f - (float)(d & 7);
            qss += qv * qv;
            kss += kv * kv;
        }
        const float qscale = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)HK128);
        const float kscale = 1.0f / sqrtf(kss + 1e-6f);

        const float av = __bfloat162float(agate[(size_t)i * nvh + vh]) + dt_bias[vh];
        const float sp = (av > 20.0f) ? av : log1pf(expf(av));
        const float decay = expf(-expf(a_log[vh]) * sp);
        const float bv = __bfloat162float(bgate[(size_t)i * nvh + vh]);
        const float beta = 1.0f / (1.0f + expf(-bv));

        float kv_mem = 0.0f;
#pragma unroll
        for (int d = 0; d < HK128; ++d) {
            const float sv = st[d] * decay;
            st[d] = sv;
            kv_mem += sv * ((0.5f - (float)(d & 7)) * kscale);
        }
        const float v = __bfloat162float(conv[2 * (num_k_heads * HK128) + (size_t)vh * HV128 + dv]);
        const float delta = (v - kv_mem) * beta;

        float acc = 0.0f;
#pragma unroll
        for (int d = 0; d < HK128; ++d) {
            const float su = st[d] + (0.5f - (float)(d & 7)) * kscale * delta;
            st[d] = su;
            acc += su * ((0.25f + (float)(d & 3)) * qscale);
        }

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
        const float r = 1.0f / sqrtf(ssum[0] / (float)HV128 + 1e-6f);

        const float zz = __bfloat162float(z[(size_t)i * value_dim + (size_t)vh * HV128 + dv]);
        const float actv = (gate_act == 0) ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
        gated[(size_t)i * value_dim + (size_t)vh * HV128 + dv] =
            __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
        __syncthreads();
    }
#pragma unroll
    for (int d = 0; d < HK128; ++d) S[(size_t)d * HV128 + dv] = st[d];
}
