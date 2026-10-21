// L1 GR（GatedResidual / hyper-connection）相关算子。
// 约定：外部接口一律 bf16；kernel 内部按 float 计算后回落 bf16（对齐 HF 的 float 中间精度）。
// 纯函数：无全局态、无 alloc，stream 由调用方显式传入。

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// 分组 RMSNorm（gamma 已烘焙，直接乘）：x[T, dim] → 每 (row, group) 独立归一化。
// 与 HF `Qwen4ExpTextRMSNorm._norm`（reshape(-1, group_size) 后 rms + (1+w)）等价。
extern "C" __global__ void lingxi_grouped_rmsnorm_bf16(const bf16* __restrict__ x,
                                                       const bf16* __restrict__ gamma,
                                                       bf16* __restrict__ y,
                                                       int dim,
                                                       int group,
                                                       float eps) {
    const int row = blockIdx.x;
    const int gi = blockIdx.y;
    const bf16* xr = x + (size_t)row * (size_t)dim + (size_t)gi * (size_t)group;
    bf16* yr = y + (size_t)row * (size_t)dim + (size_t)gi * (size_t)group;

    extern __shared__ float sdata[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        acc += v * v;
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            sdata[threadIdx.x] += sdata[threadIdx.x + st];
        }
        __syncthreads();
    }
    const float rms = rsqrtf(sdata[0] / (float)group + eps);
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]) * rms * __bfloat162float(gamma[gi * group + i]);
        yr[i] = __float2bfloat16(v);
    }
}

// 就地：x = silu(x * scale)
extern "C" __global__ void lingxi_silu_scale_inplace_bf16(bf16* __restrict__ x,
                                                          float scale,
                                                          int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = __bfloat162float(x[i]) * scale;
        x[i] = __float2bfloat16(v / (1.0f + __expf(-v)));
    }
}

// 就地：x = sigmoid(x * scale) * mul
extern "C" __global__ void lingxi_sigmoid_scale_mul_inplace_bf16(bf16* __restrict__ x,
                                                                 float scale,
                                                                 float mul,
                                                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = __bfloat162float(x[i]) * scale;
        x[i] = __float2bfloat16(mul / (1.0f + __expf(-v)));
    }
}

// out[row, i] = (1/hc) * Σ_g w[row, g*H+i] * xn[row, g*H+i]
// 逐行核改成**一次处理所有行**（`blockIdx.y` = 行）：原来在 Rust 侧 `for row in 0..t`
// 每行发一次 launch，t>1 时每层多出 2(t-1)~(4)(t-1) 个 kernel × ~15 µs 的 GPU 固定开销
// （t=8 时折合 ~20 ms/token，正好解释了 t=8 前向 159 ms 里那条"随 t 线性增长"的成本）。
extern "C" __global__ void lingxi_mix_mean_bf16(const bf16* __restrict__ w,
                                                const bf16* __restrict__ xn,
                                                bf16* __restrict__ out,
                                                int hc,
                                                int hidden,
                                                int row_base) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = row_base + (int)blockIdx.y;
    if (i < hidden) {
        const size_t base = (size_t)row * (size_t)hc * (size_t)hidden;
        float acc = 0.0f;
        for (int g = 0; g < hc; ++g) {
            const size_t idx = base + (size_t)g * (size_t)hidden + (size_t)i;
            acc += __bfloat162float(w[idx]) * __bfloat162float(xn[idx]);
        }
        out[(size_t)row * (size_t)hidden + (size_t)i] = __float2bfloat16(acc / (float)hc);
    }
}

// 就地残差写回：hyper[row, g*H+i] += attn[row, i] * injw[row, g]
extern "C" __global__ void lingxi_gr_write_inplace_bf16(bf16* __restrict__ hyper,
                                                        const bf16* __restrict__ attn,
                                                        const bf16* __restrict__ injw,
                                                        int hc,
                                                        int hidden,
                                                        int row_base) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = row_base + (int)blockIdx.y;
    if (i < hidden) {
        const size_t hbase = (size_t)row * (size_t)hc * (size_t)hidden;
        const float a = __bfloat162float(attn[(size_t)row * (size_t)hidden + (size_t)i]);
        for (int g = 0; g < hc; ++g) {
            const size_t idx = hbase + (size_t)g * (size_t)hidden + (size_t)i;
            hyper[idx] = __float2bfloat16(__bfloat162float(hyper[idx])
                                          + a * __bfloat162float(injw[(size_t)row * (size_t)hc + (size_t)g]));
        }
    }
}

// bf16 逐元素乘：y = a * b
extern "C" __global__ void lingxi_mul_bf16(const bf16* __restrict__ a,
                                           const bf16* __restrict__ b,
                                           bf16* __restrict__ y,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2bfloat16(__bfloat162float(a[i]) * __bfloat162float(b[i]));
    }
}

// ───────────────────────── 融合 GR（decode t==1 专用） ─────────────────────────
//
// 动机（实测，见 docs/decode-speed-profile.md §3.2/§3.12）：一个 GR 块原本是
// 5 个 kernel（grouped_rmsnorm → gemv(mix_down) → gemv(mix_up) → mix_mean →
// gemv(block_inject)），每层两个块 = 10 个 kernel；48 层 × 2 块 ≈ 960 次 launch，
// 每次 ~14 µs 的 GPU 固定开销 ≈ 13 ms/token，是单流 decode 的最大单项。
// 下面把 5 个融成 2 个：**每块只读一遍权重、数值不变**（只是求和顺序换了分组的先合后总）。
//
// 关键技巧：**把分组 RMSNorm 折进 GEMV 的读通路**（借自 DeepGEMM
// `sm90_tf32_hc_prenorm_gemm` 把 reduce(x²) 寄生在投影 A 读上的思路）：
//   xn[c] = x[c]·rms_{c/H}·gamma[c]  ⇒  Σ_c W[r,c]·xn[c] = Σ_g rms_g · S_{r,g}
//   S_{r,g} = Σ_{c∈g} W[r,c]·x[c]·gamma[c]
// 于是既不用把 xn 落成 10240 个 bf16，也省掉一个只有 4 个 block 的低并行度 kernel。

#define LINGXI_GR_MAXG 8

// 第一阶段：mix_down（+ block_inject）一行一个 block。
//   r < lowrank → d[r]              = silu(dot / hc)
//   r ≥ lowrank → injw[r-lowrank]   = 2·sigmoid(dot / hc)
// 同时 block 0 把 `rsqrt(Σx²/H + eps)` 写进 `rmsg[g]`，供第二阶段算 xn 用。
extern "C" __global__ void lingxi_gr_down_fuse_bf16(const bf16* __restrict__ x,
                                                    const bf16* __restrict__ gamma,
                                                    const bf16* __restrict__ wd,
                                                    const bf16* __restrict__ wi,
                                                    bf16* __restrict__ d_out,
                                                    bf16* __restrict__ injw_out,
                                                    float* __restrict__ rmsg,
                                                    int dim, int hidden, int ngroups,
                                                    int lowrank, int hc, float inv_hc, float eps) {
    const int r = blockIdx.x;
    const bool is_down = r < lowrank;
    const bf16* W = is_down ? (wd + (size_t)r * (size_t)dim)
                            : (wi + (size_t)(r - lowrank) * (size_t)dim);
    float S[LINGXI_GR_MAXG];
    float Q[LINGXI_GR_MAXG];
#pragma unroll
    for (int g = 0; g < LINGXI_GR_MAXG; ++g) {
        S[g] = 0.0f;
        Q[g] = 0.0f;
    }
    const float inv_h = 1.0f / (float)hidden;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const float xv = __bfloat162float(x[c]);
        const float wv = __bfloat162float(W[c]);
        const float gv = __bfloat162float(gamma[c]);
        const int g = c / hidden;
        // ⚠️ 必须**展开成编译期常量下标**：写成 `S[g] += ...`（g 是运行期值）时
        // nvcc 会把 S/Q 落到 **local memory**，每个元素变成一次 local load+store，
        // 实测让整个 kernel 慢一倍（108 µs/块 → 目标 ~30 µs）。
#pragma unroll
        for (int gg = 0; gg < LINGXI_GR_MAXG; ++gg) {
            if (gg == g) {
                S[gg] += wv * xv * gv;
                Q[gg] += xv * xv;
            }
        }
    }
    // warp 内归约
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int g = 0; g < LINGXI_GR_MAXG; ++g) {
            S[g] += __shfl_down_sync(0xffffffffu, S[g], off);
            Q[g] += __shfl_down_sync(0xffffffffu, Q[g], off);
        }
    }
    __shared__ float sS[LINGXI_GR_MAXG][32];
    __shared__ float sQ[LINGXI_GR_MAXG][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < LINGXI_GR_MAXG; ++g) {
            sS[g][warp] = S[g];
            sQ[g][warp] = Q[g];
        }
    }
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x >> 5;
        float aS[LINGXI_GR_MAXG];
        float aQ[LINGXI_GR_MAXG];
#pragma unroll
        for (int g = 0; g < LINGXI_GR_MAXG; ++g) {
            float s = (lane < nw) ? sS[g][lane] : 0.0f;
            float q = (lane < nw) ? sQ[g][lane] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_down_sync(0xffffffffu, s, off);
                q += __shfl_down_sync(0xffffffffu, q, off);
            }
            aS[g] = s;
            aQ[g] = q;
        }
        if (lane == 0) {
            float dot = 0.0f;
            for (int g = 0; g < ngroups; ++g) {
                const float rms = rsqrtf(aQ[g] * inv_h + eps);
                if (blockIdx.x == 0) {
                    rmsg[g] = rms;
                }
                dot += rms * aS[g];
            }
            const float v = dot * inv_hc;
            if (is_down) {
                d_out[r] = __float2bfloat16(v / (1.0f + __expf(-v)));
            } else {
                injw_out[r - lowrank] = __float2bfloat16(2.0f / (1.0f + __expf(-v)));
            }
        }
    }
}

// 第二阶段：mix_up + sigmoid + mix_mean —— **不落 imw**（省 40 KB/层 的写+读）。
// **一个 block 一个输出 i，block 内每个 warp 负责一个 g**（并行度必须和旧 mix_up GEMV
// 对齐：旧核 1280 block × 256 线程 = 32 万线程；最初写成"一 warp 一个输出"只有 8 万线程，
// 访存延迟藏不住，实测反而慢 11%）。
//   mixed[i] = (1/hc) Σ_g sigmoid( Σ_j mix_up[g·hidden+i, j]·d[j] ) · xn[g·hidden+i]
//   xn[g·hidden+i] = x[..]·rmsg[g]·gamma[..]
extern "C" __global__ void lingxi_gr_up_mix_bf16(const bf16* __restrict__ d,
                                                 const bf16* __restrict__ x,
                                                 const bf16* __restrict__ gamma,
                                                 const bf16* __restrict__ wu,
                                                 const float* __restrict__ rmsg,
                                                 bf16* __restrict__ mixed,
                                                 int dim, int hidden, int lowrank,
                                                 int ngroups, float inv_hc) {
    extern __shared__ float sd[];
    const int i = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int g = tid >> 5;
    for (int j = tid; j < lowrank; j += blockDim.x) {
        sd[j] = __bfloat162float(d[j]);
    }
    __syncthreads();
    float p = 0.0f;
    if (g < ngroups) {
        const bf16* row = wu + ((size_t)g * (size_t)hidden + (size_t)i) * (size_t)lowrank;
        for (int j = lane; j < lowrank; j += 32) {
            p += __bfloat162float(row[j]) * sd[j];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            p += __shfl_down_sync(0xffffffffu, p, off);
        }
    }
    __shared__ float sdot[LINGXI_GR_MAXG];
    if (lane == 0 && g < LINGXI_GR_MAXG) {
        sdot[g] = p;
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int gg = 0; gg < ngroups; ++gg) {
            const float sg = 1.0f / (1.0f + __expf(-sdot[gg]));
            const int c = gg * hidden + i;
            const float xn = __bfloat162float(x[c]) * rmsg[gg] * __bfloat162float(gamma[c]);
            acc += sg * xn;
        }
        mixed[i] = __float2bfloat16(acc * inv_hc);
    }
}

// 主干入口：把 [rows, H] 的 embedding 沿特征维重复 hc 次 → [rows, hc*H]
// （HF: `hidden_states.repeat(1, 1, hc_count)`；out[r, g*H + i] = emb[r, i]）
extern "C" __global__ void lingxi_hc_repeat_bf16(const bf16* __restrict__ emb,
                                                 bf16* __restrict__ out,
                                                 int rows,
                                                 int hidden,
                                                 int hc) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)rows * hc * hidden;
    if (i < n) {
        const int col = (int)(i % (long long)hidden);
        const long long row = i / ((long long)hc * hidden);
        out[i] = emb[row * hidden + col];
    }
}
