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
// 动机：一个 GR 块原本是
// 5 个 kernel（grouped_rmsnorm → gemv(mix_down) → gemv(mix_up) → mix_mean →
// gemv(block_inject)），每层两个块 = 10 个 kernel；48 层 × 2 块 ≈ 960 次 launch，
// 每次 ~14 µs 的 GPU 固定开销 ≈ 13 ms/token，是单流 decode 的最大单项。
// 下面把 5 个融成 2 个：**每块只读一遍权重、数值不变**（只是求和顺序换了分组的先合后总）。
//
// 关键技巧：**把分组 RMSNorm 折进 GEMV 的读通路**（借自 DeepGEMM
// `sm90_tf32_hc_prenorm_gemm` 把 reduce(x²) 寄生在投影 A 读上的思路）：
// xn[c] = x[c]·rms_{c/H}·gamma[c] ⇒ Σ_c W[r,c]·xn[c] = Σ_g rms_g · S_{r,g}
// S_{r,g} = Σ_{c∈g} W[r,c]·x[c]·gamma[c]
// 于是既不用把 xn 落成 10240 个 bf16，也省掉一个只有 4 个 block 的低并行度 kernel。

#define LINGXI_GR_MAXG 8

// 第一阶段：mix_down（+ block_inject）一行一个 block。
// r < lowrank → d[r] = silu(dot / hc)
// r ≥ lowrank → injw[r-lowrank] = 2·sigmoid(dot / hc)
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
// mixed[i] = (1/hc) Σ_g sigmoid(Σ_j mix_up[g·hidden+i, j]·d[j]) · xn[g·hidden+i]
// xn[g·hidden+i] = x[..]·rmsg[g]·gamma[..]
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

// ───────────── 通用档（rows ≥ 2）融合：5 发 → 3 发 ─────────────
//
// 背景（2026-10-24 实测，`t9_gr_shape_bw`）：decode 验证批 rows=6 时一个 GR 块
// 5 发合计 **92.6 µs**，其中
// rmsnorm 7.8 / mix_down 22.8 / mix_up 45.6 / mix_mean 6.5 / block_inject 13.4；
// 而同 grid/block 的**空核**每条要 **6.1~6.5 µs** ⇒ 5 发里有 **~31.5 µs（34%）是发射地板**。
// 下面两个核只做"合发"，**不改任何计算次序**（逐位不变）：
// (1) mix_down + block_inject → 1 发（同一 `gemm_skinny_row_bf16` 的体，按行分流 epi）
// (2) mix_up + sigmoid + mix_mean → 1 发（warp→列、lane→k 的次序与 `gemm_skinny` 逐字相同）
// 闸门由 host 侧控制（`rows 2..8`、无 f32x、有 block_inject）。

// (1) down + inject 合发。行 r < lowrank 走 mix_down（epi=1, a=inv_hc），
// r ≥ lowrank 走 block_inject（epi=2, a=inv_hc, b=2）。
// **逐位等价论证**：线程→列映射（c0 = tid*8 递增）、8 元素内 z 次序、vec/tail 两路分支、
// warp shfl + smem 两级归约树与 `lingxi_gemm_skinny_row_bf16` 完全同构；两路输出各自的
// 行索引/步长按原 n 计算（d 用 lowrank、injw 用 hc），与两次独立 launch 写到同一地址。
extern "C" __global__ void lingxi_gr_down_inject_bf16(const bf16* __restrict__ wd,
                                                      const bf16* __restrict__ wi,
                                                      const bf16* __restrict__ x,
                                                      bf16* __restrict__ d,
                                                      bf16* __restrict__ injw,
                                                      int lowrank, int hc,
                                                      int k, int m, float inv_hc) {
    const int row = blockIdx.x;
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const bool is_down = row < lowrank;
    const bf16* wr = is_down ? (wd + (size_t)row * (size_t)k)
                             : (wi + (size_t)(row - lowrank) * (size_t)k);
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    int c0 = tid * 8;
    for (; c0 + 8 <= k; c0 += (int)blockDim.x * 8) {
        const int4 wv = *reinterpret_cast<const int4*>(wr + c0);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= mt) {
                break;
            }
            const int4 xv = *reinterpret_cast<const int4*>(x + (size_t)(r0 + s) * (size_t)k + c0);
            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
            float dd = 0.0f;
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                dd += __bfloat162float(wp[z]) * __bfloat162float(xp[z]);
            }
            acc[s] += dd;
        }
    }
    for (; c0 < k; c0 += (int)blockDim.x * 8) {
        for (int i = 0; i < 8 && c0 + i < k; ++i) {
            const float wf = __bfloat162float(wr[c0 + i]);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= mt) {
                    break;
                }
                acc[s] += wf * __bfloat162float(x[(size_t)(r0 + s) * (size_t)k + c0 + i]);
            }
        }
    }
    __shared__ float sred[32][8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= mt) {
            break;
        }
        float v = acc[s];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) {
            sred[warp][s] = v;
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int s = 0; s < mt; ++s) {
            float v = (lane < nwarp) ? sred[lane][s] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                float rr = v;
                if (is_down) {
                    const float t2 = rr * inv_hc;
                    rr = t2 / (1.0f + __expf(-t2));
                    d[(size_t)(r0 + s) * (size_t)lowrank + row] = __float2bfloat16(rr);
                } else {
                    const float t2 = rr * inv_hc;
                    rr = (1.0f / (1.0f + __expf(-t2))) * 2.0f;
                    injw[(size_t)(r0 + s) * (size_t)hc + (row - lowrank)] = __float2bfloat16(rr);
                }
            }
        }
    }
}

// (2) mix_up + sigmoid(imw round bf16) + mix_mean 合发（不落 imw）。
// block = `32 * hc * opb` 线程：warp → (ii, g)，`i = blockIdx.x*opb + ii`。
// grid.x = hidden/opb。
// **逐位等价论证**：
// * 每 (g,i) 列的点积仍是 `for (j = lane; j < lowrank; j += 32)`，每 lane 的累加序列、
// 8 元素内 z 次序、shfl_down 16..1 归约树与 `lingxi_gemm_skinny_*` 逐字相同；
// `d` 预转 f32 进 smem（bf16→f32 精确 ⇒ 值不变，只换搬运方式），每 token 一行。
// * imw 先 `__float2bfloat16` round（= 原路径写 imw bf16 张量的舍入点），再从 bf16 读回。
// * 求和沿用 `lingxi_mix_mean_bf16` 的循环：g 升序累加、再 `/(float)hc` 后 round。
extern "C" __global__ void lingxi_gr_up_mix_fuse_bf16(const bf16* __restrict__ wu,
                                                      const bf16* __restrict__ d,
                                                      const bf16* __restrict__ xn,
                                                      bf16* __restrict__ mixed,
                                                      int lowrank, int hidden, int hc,
                                                      int m, int opb) {
    extern __shared__ float sd[];  // m * lowrank（每 token 一行，bf16→f32 精确）
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = warp % hc;
    const int i = (int)blockIdx.x * opb + warp / hc;
    for (int j = tid; j < m * lowrank; j += (int)blockDim.x) {
        sd[j] = __bfloat162float(d[j]);
    }
    __syncthreads();
    const bf16* wr = wu + ((size_t)g * (size_t)hidden + (size_t)i) * (size_t)lowrank;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r] = 0.0f;
    }
    for (int j = lane; j < lowrank; j += 32) {
        const float wv = __bfloat162float(wr[j]);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < m) {
                acc[r] += wv * sd[r * lowrank + j];
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }
    // imw 的 bf16 舍入点（= 原路径写/读 imw 张量）
    __shared__ bf16 simw[8][8 * 8];
    const int ii = warp / hc;
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < m) {
                const float t2 = acc[r] * 1.0f;
                simw[r][ii * hc + g] =
                    __float2bfloat16((1.0f / (1.0f + __expf(-t2))) * 1.0f);
            }
        }
    }
    __syncthreads();
    // 尾：本 block 的 **opb 个输出** 各自把 hc 个组的 imw 与 xn 混合（每个线程负责一个
    // (ii, r) 对）。求和次序仍与 `lingxi_mix_mean_bf16` 逐字相同（g 升序 → /hc → round）。
    if (tid < m * opb) {
        const int r = tid % m;
        const int ii2 = tid / m;
        const int i2 = (int)blockIdx.x * opb + ii2;
        const size_t base = (size_t)r * (size_t)hc * (size_t)hidden;
        float sum = 0.0f;
        for (int gg = 0; gg < hc; ++gg) {
            const size_t idx = base + (size_t)gg * (size_t)hidden + (size_t)i2;
            sum += __bfloat162float(simw[r][ii2 * hc + gg]) * __bfloat162float(xn[idx]);
        }
        mixed[(size_t)r * (size_t)hidden + (size_t)i2] = __float2bfloat16(sum / (float)hc);
    }
}

// 主干入口：把 [rows, H] 的 embedding 沿特征维重复 hc 次 → [rows, hc*H]
// （HF: `hidden_states.repeat(1, 1, hc_count)`；out[r, g*H+i] = emb[r, i]）
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
