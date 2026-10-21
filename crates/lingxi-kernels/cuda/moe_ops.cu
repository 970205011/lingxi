// L1 MoE（稀疏专家）相关算子。全部 bf16，内部 float 计算。

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// 融合 SwiGLU：x[m, 2*inter] → y[m, inter]，y = silu(x[:,0:inter]) * x[:,inter:2*inter]
extern "C" __global__ void lingxi_swiglu_bf16(const bf16* __restrict__ x,
                                              bf16* __restrict__ y,
                                              int m,
                                              int inter) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)m * inter) {
        const int row = (int)(i / inter);
        const int col = (int)(i % inter);
        const bf16* xr = x + (size_t)row * (size_t)(2 * inter);
        const float g = __bfloat162float(xr[col]);
        const float u = __bfloat162float(xr[inter + col]);
        y[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
    }
}

// 按行 gather：dst[i, :] = src[idx[idx_off + i], :]
// `idx_off`：索引/输出行在各自数组里的起始偏移（一次分配整层索引后逐专家切片用）。
extern "C" __global__ void lingxi_gather_rows_bf16(const bf16* __restrict__ src,
                                                   const int* __restrict__ idx,
                                                   bf16* __restrict__ dst,
                                                   int m,
                                                   int k,
                                                   int idx_off) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)m * k) {
        const int row = (int)(i / k);
        const int col = (int)(i % k);
        dst[i] = src[(size_t)idx[idx_off + row] * (size_t)k + (size_t)col];
    }
}

// 按行加权累加：dst[idx[idx_off+i], :] += w[idx_off+i] * src[i, :]
// 同一次调用内 idx 互不相同（top-k 选的是不同专家），故无需原子。
extern "C" __global__ void lingxi_scatter_add_rows_bf16(bf16* __restrict__ dst,
                                                        const int* __restrict__ idx,
                                                        const float* __restrict__ w,
                                                        const bf16* __restrict__ src,
                                                        int m,
                                                        int k,
                                                        int idx_off) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)m * k) {
        const int row = (int)(i / k);
        const int col = (int)(i % k);
        const size_t d = (size_t)idx[idx_off + row] * (size_t)k + (size_t)col;
        dst[d] = __float2bfloat16(__bfloat162float(dst[d])
                                  + w[idx_off + row] * __bfloat162float(src[i]));
    }
}

// 合并：`expert_out[r, :] = Σ_{s : rows_of[s] == r} w[s] * o[s, :]`
// （decode 批量路径用：各专家的输出先写进 `o[S, H]`，再一次按路由权重归并回 token 行。）
// 与逐专家 `scatter_add_rows` 的区别：这里在 f32 里累加、最后只 round 一次（更准，非逐位相同）。
extern "C" __global__ void lingxi_moe_combine_rows_bf16(const bf16* __restrict__ o,
                                                        const int* __restrict__ rows_of,
                                                        const float* __restrict__ w,
                                                        bf16* __restrict__ expert_out,
                                                        int S,
                                                        int H) {
    const int r = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        if (rows_of[s] == r) {
            acc += w[s] * __bfloat162float(o[(size_t)s * H + h]);
        }
    }
    expert_out[(size_t)r * H + h] = __float2bfloat16(acc);
}

// 合并+残差**一发融合**（P15-㊶，差距#9）：`out[r,:] = bf16(f32(bf16(Σ w·o)) + f32(shared[r,:]))`。
// 两个舍入（combine 的 round + add 的 round）与"combine 一发 + add_bf16 一发"**完全同序**
// ⇒ 逐位相同；省一次 5120 元素 kernel launch × 48 层。
extern "C" __global__ void lingxi_moe_combine_add_rows_bf16(const bf16* __restrict__ o,
                                                            const int* __restrict__ rows_of,
                                                            const float* __restrict__ w,
                                                            const bf16* __restrict__ shared,
                                                            bf16* __restrict__ out,
                                                            int S,
                                                            int H) {
    const int r = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        if (rows_of[s] == r) {
            acc += w[s] * __bfloat162float(o[(size_t)s * H + h]);
        }
    }
    const bf16 e = __float2bfloat16(acc);
    out[(size_t)r * H + h] =
        __float2bfloat16(__bfloat162float(e) + __bfloat162float(shared[(size_t)r * H + h]));
}

// ㊾ 合并+残差 CSR 版（batch 路径，`S=对数` 可以是几千——旧核逐元素扫全 S 是
// O(rows×H×S)，rows=887 实测 19ms/层，占专家段 76%）：
//   `out[r,:] = bf16(f32(bf16(Σ_j w[col[j]]·o[col[j],:])) + f32(shared[r,:]))`
// `col[row_off[r]..row_off[r+1])` = 该行所有对的原下标 s，**升序**（host 稳定计数
// 排序保证）⇒ 每行的累加次序与旧核"升序 s 扫描"完全一致 ⇒ 逐位相同。
extern "C" __global__ void lingxi_moe_combine_add_csr_bf16(
    const bf16* __restrict__ o, const int* __restrict__ col,
    const int* __restrict__ row_off, const float* __restrict__ w,
    const bf16* __restrict__ shared, bf16* __restrict__ out, int H) {
    const int r = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    float acc = 0.0f;
    for (int j = row_off[r]; j < row_off[r + 1]; ++j) {
        const int s = col[j];
        acc += w[s] * __bfloat162float(o[(size_t)s * H + h]);
    }
    const bf16 e = __float2bfloat16(acc);
    out[(size_t)r * H + h] =
        __float2bfloat16(__bfloat162float(e) + __bfloat162float(shared[(size_t)r * H + h]));
}

// bf16 GEMV：`y[n] = W[n,k] · x[k]`（W 行主序）。T=1 的 decode 用。
// 一个 warp 管一行、lane 沿 k 分片 → 访存合并；x 常驻 L1/L2（几 KB）。
// 动机：cuBLAS 的 `gemm_ex` 每次调用约 **26 µs CPU 开销**（实测），T=1 下远超其计算量。
__device__ __forceinline__ float lingxi_dot_bf16(const bf16* __restrict__ wr,
                                                 const bf16* __restrict__ x,
                                                 int k, int lane) {
    float acc = 0.0f;
    for (int j = lane; j < k; j += 32) {
        acc += __bfloat162float(wr[j]) * __bfloat162float(x[j]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    return acc;
}

extern "C" __global__ void lingxi_gemv_bf16(const bf16* __restrict__ w,
                                            const bf16* __restrict__ x,
                                            bf16* __restrict__ y,
                                            int n,
                                            int k) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const float acc = lingxi_dot_bf16(w + (size_t)warp * (size_t)k, x, k, lane);
    if (lane == 0) {
        y[warp] = __float2bfloat16(acc);
    }
}

// 带**尾融合**的 bf16 GEMV：把紧跟其后的逐元素 op 折进同一个 kernel，省掉一次
// kernel 发起 + 一次读写往返。decode 是"kernel 数"受限（每层 40 来个 kernel、
// 每个 ~10-30 µs 的固定成本），尾融合是最划算的一刀。
//   epi = 0：`y = acc`
//   epi = 1：`y = silu(acc * a)`              （GR mix_down：先除 hc 再 silu）
//   epi = 2：`y = sigmoid(acc * a) * b`       （GR mix_up / block_inject）
//   epi = 3：`y = acc * a + b`                （仿射）
//   epi = 4：`y = silu(g) * u`，g/u 取 W 的第 i 与第 i+n/2 行（MoE swiglu）
extern "C" __global__ void lingxi_gemv_bf16_epi(const bf16* __restrict__ w,
                                                const bf16* __restrict__ x,
                                                bf16* __restrict__ y,
                                                int n,
                                                int k,
                                                int epi,
                                                float a,
                                                float b) {
    const int half = (epi == 4) ? (n >> 1) : n;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= half) {
        return;
    }
    if (epi == 4) {
        const float g = lingxi_dot_bf16(w + (size_t)warp * (size_t)k, x, k, lane);
        const float u = lingxi_dot_bf16(w + (size_t)(warp + half) * (size_t)k, x, k, lane);
        if (lane == 0) {
            const float s = g / (1.0f + __expf(-g));
            y[warp] = __float2bfloat16(s * u);
        }
        return;
    }
    float acc = lingxi_dot_bf16(w + (size_t)warp * (size_t)k, x, k, lane);
    if (lane != 0) {
        return;
    }
    if (epi == 1) {
        const float v = acc * a;
        acc = v / (1.0f + __expf(-v));
    } else if (epi == 2) {
        const float v = acc * a;
        acc = (1.0f / (1.0f + __expf(-v))) * b;
    } else if (epi == 3) {
        acc = acc * a + b;
    }
    y[warp] = __float2bfloat16(acc);
}

// **一个 block 负责一行**的高并行度 GEMV（1024 线程 = 32 warp 一起算一行，warp 间用
// shared 归约）。专治"n 小、k 大"的形状：
//
// 动机（engine 内实测，`lingxi-bench launch-bench`）：
//   [320,10240] 6.2 MB，warps-per-row 只有 40 个 block → **26.1 µs（251 GB/s）**
//   [   4,10240] 0.1 MB，只有 **1 个 block** → **13.6 µs（6 GB/s）**
//   [8192,2560] 40 MB（1024 blocks）→ 57.9 µs（725 GB/s）
// 原因不是带宽，是**每行的并行度**：warps-per-row 下每 lane 要串行走 k/32 次依赖 load，
// k=10240 就是 320 次 → 访存延迟串成链。本 kernel 把每行的并行度提到 32 warp，
// k=10240 只需 10 次 load 链；n 个 block 也保证了 SM 占用。
//
// 语义与 `lingxi_gemv_bf16_epi` 相同（含同样的尾融合），差别只在**累加顺序**
// （warp 间先归约再跨 warp），因此数值会有最后一两位的差异。
extern "C" __global__ void lingxi_gemv_bf16_block_epi(const bf16* __restrict__ w,
                                                      const bf16* __restrict__ x,
                                                      bf16* __restrict__ y,
                                                      int n,
                                                      int k,
                                                      int epi,
                                                      float a,
                                                      float b) {
    const int row = blockIdx.x;
    if (row >= n) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc = 0.0f;
    for (int j = warp * 32 + lane; j < k; j += nwarp * 32) {
        acc += __bfloat162float(wr[j]) * __bfloat162float(x[j]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    __shared__ float sred[32];
    if (lane == 0) {
        sred[warp] = acc;
    }
    __syncthreads();
    if (tid >= 32) {
        return;
    }
    float v = (lane < nwarp) ? sred[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane != 0) {
        return;
    }
    if (epi == 1) {
        const float t = v * a;
        v = t / (1.0f + __expf(-t));
    } else if (epi == 2) {
        const float t = v * a;
        v = (1.0f / (1.0f + __expf(-t))) * b;
    } else if (epi == 3) {
        v = v * a + b;
    }
    y[row] = __float2bfloat16(v);
}

// **向量化 block GEMV**：一行一个 block，每线程一次读 **16 字节（8 个 bf16）**。
//
// 动机（实测，`lingxi-bench launch-bench` 的形状表）：
//   [320,10240] 6.2 MB 走 `gemv_bf16_ptr`（warps-per-row）只有 **40 block → 254 GB/s**；
//   [4,10240] 只有 1 block → 6 GB/s。瓶颈不是带宽，而是**每线程只发 2 字节的标量 load**
//   （k=10240 时每 lane 要走 320 次依赖 load），访存延迟串成链、指令发射被 load 打满。
// 本 kernel 把每线程的 load 次数压到 `k/8/blockDim`（k=10240、256 线程时只要 5 次），
// 让访存真正跑在带宽上。语义与 `lingxi_gemv_bf16_block_epi` 相同（只是归约顺序不同）。
// 要求 `k % 8 == 0`（否则尾部走标量兜底）。
extern "C" __global__ void lingxi_gemv_bf16_row8_epi(const bf16* __restrict__ w,
                                                     const bf16* __restrict__ x,
                                                     bf16* __restrict__ y,
                                                     int n, int k, int epi, float a, float b) {
    const int row = blockIdx.x;
    if (row >= n) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc = 0.0f;
    const int vec_k = k & ~7;
    for (int j = tid * 8; j < vec_k; j += (int)blockDim.x * 8) {
        const int4 wv = *reinterpret_cast<const int4*>(wr + j);
        const int4 xv = *reinterpret_cast<const int4*>(x + j);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
        const bf16* xp = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc += __bfloat162float(wp[t]) * __bfloat162float(xp[t]);
        }
    }
    for (int j = vec_k + tid; j < k; j += blockDim.x) {
        acc += __bfloat162float(wr[j]) * __bfloat162float(x[j]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    __shared__ float sred[32];
    if (lane == 0) {
        sred[warp] = acc;
    }
    __syncthreads();
    if (tid != 0) {
        return;
    }
    float v = 0.0f;
    for (int wp2 = 0; wp2 < nwarp; ++wp2) {
        v += sred[wp2];
    }
    if (epi == 1) {
        const float t = v * a;
        v = t / (1.0f + __expf(-t));
    } else if (epi == 2) {
        const float t = v * a;
        v = (1.0f / (1.0f + __expf(-t))) * b;
    } else if (epi == 3) {
        v = v * a + b;
    }
    y[row] = __float2bfloat16(v);
}

// **一行一个 block 的瘦长 GEMM**（t>1 用）：与 row8 同款并行度（n 个 block），128 位 load，
// m 个独立累加器。
//
// ⚠️ 旧版 `lingxi_gemm_skinny_m*` 的 grid 是 **n/8 个 block**（warps-per-row），
// n 小的时候并行度差 8 倍：GR 的 `block_inject`（n=4）只有 **1 个 block**、
// `mix_down`（n=320）只有 40 个。实测 t=2 时 GR 从 3.4 ms 涨到 10.3 ms（3.0×），
// 而理论上"权重只读一遍"应该接近 1.1×。
// 本核把 grid 拉回 n 个 block（与 t=1 的 row8 一致），并把每线程的 load 数除以 8。
// 语义与 `lingxi_gemm_skinny_m*` 完全一致（`y[m][n] = W[n,k]·x[m,k]^T` + 同样的尾融合）。
// `epi == 4`（swiglu）不走本核（调用方保证）。
extern "C" __global__ void lingxi_gemm_skinny_row_bf16(const bf16* __restrict__ w,
                                                       const bf16* __restrict__ x,
                                                       bf16* __restrict__ y,
                                                       int n, int k, int m, int epi,
                                                       float a, float b) {
    const int row = blockIdx.x;
    if (row >= n || m <= 0) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const bf16* wr = w + (size_t)row * (size_t)k;
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
            if (s >= m) {
                break;
            }
            const int4 xv = *reinterpret_cast<const int4*>(x + (size_t)s * (size_t)k + c0);
            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
            float d = 0.0f;
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                d += __bfloat162float(wp[z]) * __bfloat162float(xp[z]);
            }
            acc[s] += d;
        }
    }
    for (; c0 < k; c0 += (int)blockDim.x * 8) {
        for (int i = 0; i < 8 && c0 + i < k; ++i) {
            const float wf = __bfloat162float(wr[c0 + i]);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= m) {
                    break;
                }
                acc[s] += wf * __bfloat162float(x[(size_t)s * (size_t)k + c0 + i]);
            }
        }
    }
    __shared__ float sred[32][8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= m) {
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
        for (int s = 0; s < m; ++s) {
            float v = (lane < nwarp) ? sred[lane][s] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                float r = v;
                if (epi == 1) {
                    const float t2 = r * a;
                    r = t2 / (1.0f + __expf(-t2));
                } else if (epi == 2) {
                    const float t2 = r * a;
                    r = (1.0f / (1.0f + __expf(-t2))) * b;
                } else if (epi == 3) {
                    r = r * a + b;
                }
                y[(size_t)s * (size_t)n + row] = __float2bfloat16(r);
            }
        }
    }
}

// ★ P15-⑨（2026-10-17）：`gemm_skinny_row` 的 **x 预转 fp32** 版（"所有后面计算过程
// 要转化的提前转化"）。skinny_row 在 m>1 时带宽腰斩（m=8 实测 350-415 GB/s vs m=1 750+）
// 的根因：内层每 (元素,token) 重复 bf16→f32 CVT（m=8 时每 16B 权重 ~200 条指令，
// issue 塞爆）。这里 w 的 CVT 每轮只做 8 次（wf[8]），x 由调用方一次性预转
// （`convert_bf16_f32`，bf16→f32 精确 ⇒ **逐位等价**），内层每 token 只剩 2×LDG.128 + 8 FMA。
// 尾部/归约/epi 与 `lingxi_gemm_skinny_row_bf16` 完全同构。
extern "C" __global__ void lingxi_gemm_skinny_row_f32x_bf16w(const bf16* __restrict__ w,
                                                             const float* __restrict__ x,
                                                             bf16* __restrict__ y,
                                                             int n, int k, int m, int epi,
                                                             float a, float b) {
    const int row = blockIdx.x;
    if (row >= n || m <= 0) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    int c0 = tid * 8;
    for (; c0 + 8 <= k; c0 += (int)blockDim.x * 8) {
        const int4 wv = *reinterpret_cast<const int4*>(wr + c0);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
        float wf[8];
#pragma unroll
        for (int z = 0; z < 8; ++z) {
            wf[z] = __bfloat162float(wp[z]);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= m) {
                break;
            }
            const float4 x0 = *reinterpret_cast<const float4*>(x + (size_t)s * (size_t)k + c0);
            const float4 x1 = *reinterpret_cast<const float4*>(x + (size_t)s * (size_t)k + c0 + 4);
            float d = 0.0f;
            d += wf[0] * x0.x;
            d += wf[1] * x0.y;
            d += wf[2] * x0.z;
            d += wf[3] * x0.w;
            d += wf[4] * x1.x;
            d += wf[5] * x1.y;
            d += wf[6] * x1.z;
            d += wf[7] * x1.w;
            acc[s] += d;
        }
    }
    for (; c0 < k; c0 += (int)blockDim.x * 8) {
        for (int i = 0; i < 8 && c0 + i < k; ++i) {
            const float wf = __bfloat162float(wr[c0 + i]);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= m) {
                    break;
                }
                acc[s] += wf * x[(size_t)s * (size_t)k + c0 + i];
            }
        }
    }
    __shared__ float sred[32][8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= m) {
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
        for (int s = 0; s < m; ++s) {
            float v = (lane < nwarp) ? sred[lane][s] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                float r = v;
                if (epi == 1) {
                    const float t2 = r * a;
                    r = t2 / (1.0f + __expf(-t2));
                } else if (epi == 2) {
                    const float t2 = r * a;
                    r = (1.0f / (1.0f + __expf(-t2))) * b;
                } else if (epi == 3) {
                    r = r * a + b;
                }
                y[(size_t)s * (size_t)n + row] = __float2bfloat16(r);
            }
        }
    }
}

// ★ P15-⑲（2026-10-17）：`gemm_skinny_row_bf16` 的**多输出行/块**版（R=4）。
// m>1 带宽腰斩（m=8 实测 347-416 GB/s vs m=1 750+）有两个叠加根因：
//   ① 内层每 (元素,token) 重复 bf16→f32 CVT —— ⑨ 的 f32x 预转解掉了 CVT，但 x 变
//      fp32 后每 block 的 x 流量翻倍 ⇒ 整机净亏（P15-⑨ 判否）；
//   ② 每 block 只算 1 个输出行 ⇒ x[m,k] 被 n 个 block 重复读（L2）+ 重复 CVT。
// 本核：一块算 R 个输出行，x 按 2048 元素分块进**动态 smem（转好 f32）**，行间共享
// ⇒ x 的 L2 读与 CVT 同除 R；权重流量不变。CVT 与 L2 一起解，避开 f32x 的翻倍坑。
//
// **逐位等价**（对 skinny_row，BLOCK=256 同款启动）论证：每线程 acc[r][s] 的累加
// 序列与单行版**完全相同**——分块只改 x 的搬运时机，不改每线程访问元素的全局次序
// （c0 = cb + tid*8，cb 是步长 BLOCK*8=2048 的倍数 ⇒ 每 chunk 每线程恰一个位置，全局
// 递增）；8 元素组内 z 次序、vec/tail 两路分支条件（c0+8<=k）与 warp/smem 两级归约
// 全部照搬；chunk 边界是 8 的倍数 ⇒ 不切断 vec 组。bf16→f32 精确，smem 里的就是
// 原核每轮现转的那个值。
#define SKINNY_MULTI_R 4
#define SKINNY_MULTI_CHUNK 2048
// ★ bank 冲突消解：xs 行内地址加 swizzle（每 32 float 插 1 个 padding word）。
// 未 swizzle 时 32 线程读 c = t*8+z 只落在 4 个 bank（8-way 冲突，实测整体慢 9×）；
// swizzle 后 (c + c/32) mod 32 对 t=0..31 两两全不同 ⇒ 无冲突。
#define SKINNY_MULTI_XS_ROW (SKINNY_MULTI_CHUNK + (SKINNY_MULTI_CHUNK >> 5))
#define XS_SW(c) ((c) + ((c) >> 5))
extern "C" __global__ void lingxi_gemm_skinny_row_multi_bf16(const bf16* __restrict__ w,
                                                             const bf16* __restrict__ x,
                                                             bf16* __restrict__ y,
                                                             int n, int k, int m, int epi,
                                                             float a, float b) {
    const int row0 = blockIdx.x * SKINNY_MULTI_R;
    if (row0 >= n || m <= 0) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int R = min(SKINNY_MULTI_R, n - row0);
    extern __shared__ float xs[];  // [m][SKINNY_MULTI_XS_ROW] f32（行内 swizzle）

    float acc[SKINNY_MULTI_R][8];
#pragma unroll
    for (int r = 0; r < SKINNY_MULTI_R; ++r) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[r][s] = 0.0f;
        }
    }

    for (int cb = 0; cb < k; cb += SKINNY_MULTI_CHUNK) {
        const int clen = min(SKINNY_MULTI_CHUNK, k - cb);
        __syncthreads();  // 等上一轮 compute 把 xs 读完（首圈无竞争）
        for (int j = tid; j < m * clen; j += (int)blockDim.x) {
            const int s = j / clen;
            const int c = j - s * clen;
            xs[s * SKINNY_MULTI_XS_ROW + XS_SW(c)] = __bfloat162float(x[(size_t)s * (size_t)k + cb + c]);
        }
        __syncthreads();
        const int c0 = cb + tid * 8;
        if (c0 >= cb + clen) {
            continue;  // 尾巴 chunk（clen < BLOCK*8）里本线程没有位置
        }
        const int lc = c0 - cb;  // chunk 内位置（0 ≤ lc < 2048）
        if (c0 + 8 <= k) {
            // 先把 R 行权重的 8 元素组转好进寄存器（wf[r][z]，编译期下标）
            float wf[SKINNY_MULTI_R][8];
#pragma unroll
            for (int r = 0; r < SKINNY_MULTI_R; ++r) {
                if (r >= R) {
                    break;
                }
                const int4 wv =
                    *reinterpret_cast<const int4*>(w + (size_t)(row0 + r) * (size_t)k + c0);
                const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    wf[r][z] = __bfloat162float(wp[z]);
                }
            }
            // 每个 token 的 x 组只从 smem 读一次（进 xf[8] 寄存器），4 行共享
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= m) {
                    break;
                }
                float xf[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    xf[z] = xs[s * SKINNY_MULTI_XS_ROW + XS_SW(lc + z)];
                }
#pragma unroll
                for (int r = 0; r < SKINNY_MULTI_R; ++r) {
                    if (r >= R) {
                        break;
                    }
                    float d = 0.0f;
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        d += wf[r][z] * xf[z];
                    }
                    acc[r][s] += d;
                }
            }
        } else {
            for (int i = 0; i < 8 && c0 + i < k; ++i) {
                float wf[SKINNY_MULTI_R];
#pragma unroll
                for (int r = 0; r < SKINNY_MULTI_R; ++r) {
                    if (r >= R) {
                        break;
                    }
                    wf[r] = __bfloat162float(w[(size_t)(row0 + r) * (size_t)k + c0 + i]);
                }
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s >= m) {
                        break;
                    }
#pragma unroll
                    for (int r = 0; r < SKINNY_MULTI_R; ++r) {
                        if (r >= R) {
                            break;
                        }
                        acc[r][s] += wf[r] * xs[s * SKINNY_MULTI_XS_ROW + XS_SW(lc + i)];
                    }
                }
            }
        }
    }

    __shared__ float sred[32][8];
#pragma unroll
    for (int r = 0; r < SKINNY_MULTI_R; ++r) {
        if (r >= R) {
            break;
        }
        __syncthreads();  // 保护上一轮 r 对 sred 的读取
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= m) {
                break;
            }
            float v = acc[r][s];
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
            for (int s = 0; s < m; ++s) {
                float v = (lane < nwarp) ? sred[lane][s] : 0.0f;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    v += __shfl_down_sync(0xffffffffu, v, off);
                }
                if (lane == 0) {
                    float rr = v;
                    if (epi == 1) {
                        const float t2 = rr * a;
                        rr = t2 / (1.0f + __expf(-t2));
                    } else if (epi == 2) {
                        const float t2 = rr * a;
                        rr = (1.0f / (1.0f + __expf(-t2))) * b;
                    } else if (epi == 3) {
                        rr = rr * a + b;
                    }
                    y[(size_t)s * (size_t)n + row0 + r] = __float2bfloat16(rr);
                }
            }
        }
    }
}

// **瘦长 GEMM**（模板版）：`y[m, n] += W[n, k] · x[m, k]^T`（W 行主序 [n,k]）。
//
// 一个 warp 负责一行输出，把 W 的该行**只读一遍**，同时对 m 个输入行做 m 个点积。
//
// 动机与**为什么必须按 m 特化**（实测，见 launch-bench 的 skinny 对比）：
//   gemv   [8192,2560]     → 58 µs（724 GB/s）
//   skinny [8192,2560] m=1 → 195 µs（**215 GB/s**）
// 原实现固定开 8 个累加器、每元素发 8 次 LDG+FFMA，即使 m=1 也照发 → 变成指令发射受限，
// 权重带宽掉到 1/3.4。特化后 m=1 只有 1 套 load/FMA，应回到 ~700 GB/s。
// 实测 m=1/2/4 的耗时几乎相同（215/215/214 GB/s）说明"多出来的列是白送的"——
// 只要别再让指令数随 8 走。这也正是 T>1（prefill / 投机验证）能不能快起来的关键。
template <int M>
__device__ __forceinline__ void skinny_body(const bf16* __restrict__ w,
                                            const bf16* __restrict__ x,
                                            bf16* __restrict__ y,
                                            int n,
                                            int k,
                                            int epi,
                                            float a,
                                            float b) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const bf16* wr = w + (size_t)warp * (size_t)k;
    float acc[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        acc[r] = 0.0f;
    }
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            acc[r] += wv * __bfloat162float(x[(size_t)r * (size_t)k + j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            float v = acc[r];
            if (epi == 1) {
                const float t = v * a;
                v = t / (1.0f + __expf(-t));
            } else if (epi == 2) {
                const float t = v * a;
                v = (1.0f / (1.0f + __expf(-t))) * b;
            } else if (epi == 3) {
                v = v * a + b;
            }
            y[(size_t)r * (size_t)n + warp] = __float2bfloat16(v);
        }
    }
}

#define LINGXI_SKINNY_M(M)                                                              \
    extern "C" __global__ void lingxi_gemm_skinny_m##M(                                 \
        const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y,   \
        int n, int k, int epi, float a, float b) {                                      \
        skinny_body<M>(w, x, y, n, k, epi, a, b);                                       \
    }

LINGXI_SKINNY_M(1)
LINGXI_SKINNY_M(2)
LINGXI_SKINNY_M(3)
LINGXI_SKINNY_M(4)
LINGXI_SKINNY_M(5)
LINGXI_SKINNY_M(6)
LINGXI_SKINNY_M(7)
LINGXI_SKINNY_M(8)

// PLE 门控（替代 host 参考实现）：每个 (token i, group g) 块算一个点积，再过
// 有符号开方 + sigmoid 得到门控标量，最后用它缩放 value[i, :]。
//   gate = dot(key_n[blk], query_n[blk]) * inv
//   s    = sign(gate);  g2 = s * sqrt(max(|gate|, 1e-6));  sg = sigmoid(g2)
//   out[blk, j] = sg * value[i, j]
// 原实现每层要 3 次 D2H + 1 次 H2D（host 上做同样的事），48 层/ token 累积是 decode 的大头。
extern "C" __global__ void lingxi_ple_gate_bf16(const bf16* __restrict__ key_n,
                                                const bf16* __restrict__ query_n,
                                                const bf16* __restrict__ value,
                                                bf16* __restrict__ out,
                                                int hc,
                                                int hidden,
                                                float inv) {
    const int blk = blockIdx.x; // = i * hc + g
    const int i = blk / hc;
    const bf16* kk = key_n + (size_t)blk * (size_t)hidden;
    const bf16* qq = query_n + (size_t)blk * (size_t)hidden;

    float dot = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        dot += __bfloat162float(kk[j]) * __bfloat162float(qq[j]);
    }

    __shared__ float sred[32];
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)(threadIdx.x >> 5);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, off);
    }
    if (lane == 0) {
        sred[warp] = dot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = (int)((blockDim.x + 31) >> 5);
        float acc = 0.0f;
        for (int w = 0; w < nw; ++w) {
            acc += sred[w];
        }
        sred[0] = acc;
    }
    __syncthreads();
    const float gate = sred[0] * inv;
    const float s = (gate > 0.0f) ? 1.0f : ((gate < 0.0f) ? -1.0f : 0.0f);
    const float g2 = s * sqrtf(fmaxf(fabsf(gate), 1e-6f));
    const float sg = 1.0f / (1.0f + expf(-g2));

    const bf16* vv = value + (size_t)i * (size_t)hidden;
    bf16* oo = out + (size_t)blk * (size_t)hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        oo[j] = __float2bfloat16(sg * __bfloat162float(vv[j]));
    }
}

// 行缩放（广播）：x[row, :] *= s[row]
extern "C" __global__ void lingxi_row_scale_inplace_bf16(bf16* __restrict__ x,
                                                         const bf16* __restrict__ s,
                                                         int rows,
                                                         int cols) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)rows * cols) {
        const int row = (int)(i / cols);
        x[i] = __float2bfloat16(__bfloat162float(x[i]) * __bfloat162float(s[row]));
    }
}

// y = a + b（逐元素）
extern "C" __global__ void lingxi_add_bf16(const bf16* __restrict__ a,
                                           const bf16* __restrict__ b,
                                           bf16* __restrict__ y,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}
