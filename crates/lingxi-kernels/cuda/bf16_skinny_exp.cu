// **实验核**（2026-10-24）：bf16 瘦长 GEMM 与草稿 lm_head（Q4 稠密 GEMV）的取数探针。
// 只有 `tests/t15_bf16_skinny_bw.rs` 会调；生产核（`moe_ops.cu` / `gr_ops.cu` /
// `dequant_ops.cu`）一个字节都没动。
//
// 背景：前三次"只换取数结构、算式一字未改"各白捡 1.27~2.0×（//）。
// 还没查的两块是：
// A. bf16 瘦长 GEMM（`gemm_skinny_row_bf16` 的一行一 block 体 / `gemm_skinny_m8` 的一 warp
// 一行体）。线上形状 n=10240·k=320（GR mix_up）、n=1280·k=2560（shared gu）、
// n=2560·k=640（shared dn），m=6。
// B. 草稿 lm_head：`gemv_q4g64_dense_epi` n=248320·k=2560、m=1（每步读 ~318 MB payload
// + 19.9 MB scale）。
//
// 口径：GB/s 一律按**权重 payload 字节**（A: n·k·2；B: n·k_pad/2 + n·(k_pad/64)·2）折算，
// 与 / 的表同口径。
//
// ── A 族探针（m8 体：一 warp 一行输出，j = lane; j < k; j += 32；grid = ceil(n/8) 块 × 256 线程）
// lingxi_bf16_m8_read1 纯读 2 B/lane（生产取数形状）
// lingxi_bf16_m8_read16 纯读 16 B/lane uint4（结构上限）
// lingxi_bf16_m8_alu 2 B/lane + cvt + m×FFMA（x 用寄存器常量 ⇒ 不含 x 读的地板）
// lingxi_bf16_m8_prod 生产内层逐字复刻（自检：应与 `lingxi_gemm_skinny_m8` 逐位相同）
// lingxi_bf16_m8_r{2,4,8} 一个 warp 连算 R 行输出（x 的 load+cvt 被 R 行共用）**逐位等价**
// lingxi_bf16_m8_r4pf R=4 + 权重 PF=2（下一迭代的 4 个行权重先发）
// lingxi_bf16_m8_r4x R=4 + x 先期转 f32 进 smem（隔离 x 的 global 读）
//
// ── A 族探针（row 体：一 block 一行输出，c0 = tid*8，uint4 取数；grid = n 块 × 256 线程）
// lingxi_bf16_row_read16 纯读 16 B/lane（生产取数形状）
// lingxi_bf16_row_alu 16 B/lane + 8×cvt + m×8 FFMA（x 常量）
// lingxi_bf16_row_prod 生产内层逐字复刻（自检：应与 `lingxi_gemm_skinny_row_bf16` 逐位相同）
// lingxi_bf16_row_r2 一个 block 连算 2 行输出（x 的 load+cvt 被 2 行共用）**逐位等价**
// lingxi_bf16_row_r2pf R=2 + 权重 PF=2
// lingxi_bf16_row_r2x R=2 + x 先期转 f32 进 smem
//
// ── B 族探针（grid = ceil(n/8) 块 × 256 线程，一 warp 一行，c0 = lane*8；c0 += 256）
// lingxi_q4_head_read1 纯读 payload 4 B/lane + 每 group 一个 scale（生产取数形状）
// lingxi_q4_head_read16 纯读 payload 16 B/lane（结构上限，不含 scale）
// lingxi_q4_head_prod 生产内层逐字复刻（自检：应与 `lingxi_gemv_q4g64_dense_epi` 逐位相同）

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

// ═══════════════════════════ A 族：m8 体（一 warp 一行） ═══════════════════════════

__device__ __forceinline__ void m8_row_of(int& row, int& lane) {
    row = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    lane = (int)(threadIdx.x & 31);
}

// 纯读 2 B/lane（生产取数形状）。整数累加，不做 cvt。
extern "C" __global__ void lingxi_bf16_m8_read1(const bf16* __restrict__ w,
                                                const bf16* __restrict__ x,
                                                bf16* __restrict__ y,
                                                int n, int k, int m) {
    int row, lane;
    m8_row_of(row, lane);
    if (row >= n) {
        return;
    }
    const std::uint16_t* wr = reinterpret_cast<const std::uint16_t*>(w + (size_t)row * (size_t)k);
    unsigned iacc = 0;
    for (int j = lane; j < k; j += 32) {
        iacc += (unsigned)wr[j];
    }
    if (lane == 0 && iacc == 0xFFFFFFFFu) {
        y[row] = __float2bfloat16(1.0f);
    }
}

// 纯读 16 B/lane uint4（结构上限）。
extern "C" __global__ void lingxi_bf16_m8_read16(const bf16* __restrict__ w,
                                                 const bf16* __restrict__ x,
                                                 bf16* __restrict__ y,
                                                 int n, int k, int m) {
    int row, lane;
    m8_row_of(row, lane);
    if (row >= n) {
        return;
    }
    const std::uint8_t* wr =
        reinterpret_cast<const std::uint8_t*>(w + (size_t)row * (size_t)k);
    const int row_bytes = k * 2;
    unsigned iacc = 0;
    for (int o = lane * 16; o + 16 <= row_bytes; o += 512) {
        const uint4 v = *reinterpret_cast<const uint4*>(wr + o);
        iacc += v.x + v.y + v.z + v.w;
    }
    if (lane == 0 && iacc == 0xFFFFFFFFu) {
        y[row] = __float2bfloat16(1.0f);
    }
}

// 2 B/lane + cvt + m×FFMA（x 用寄存器常量）⇒ 不含 x 读的 ALU 地板。
extern "C" __global__ void lingxi_bf16_m8_alu(const bf16* __restrict__ w,
                                              const bf16* __restrict__ x,
                                              bf16* __restrict__ y,
                                              int n, int k, int m) {
    int row, lane;
    m8_row_of(row, lane);
    if (row >= n) {
        return;
    }
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
        float xc = (float)(j & 7) * 0.125f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                acc[s] += wv * xc;
            }
        }
    }
    float tot = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        tot += acc[s];
    }
    if (lane == 0 && tot == 1e30f) {
        y[row] = __float2bfloat16(1.0f);
    }
}

// 生产内层逐字复刻（epi=0、grid.y=1、m≤8）。
extern "C" __global__ void lingxi_bf16_m8_prod(const bf16* __restrict__ w,
                                               const bf16* __restrict__ x,
                                               bf16* __restrict__ y,
                                               int n, int k, int m) {
    int row, lane;
    m8_row_of(row, lane);
    if (row >= n) {
        return;
    }
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    const bf16* wr = w + (size_t)row * (size_t)k;
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                acc[s] += wv * __bfloat162float(x[(size_t)s * (size_t)k + j]);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[s] += __shfl_down_sync(0xffffffffu, acc[s], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                y[(size_t)s * (size_t)n + row] = __float2bfloat16(acc[s]);
            }
        }
    }
}

// **候选结构**：一个 warp 连算 R 行输出，x 的 load+cvt 被 R 行共用。
// **逐位等价**：每行 r 的 lane→列集合（j = lane, lane+32, …）、累加次序（j 升序）、
// warp shfl_down 16..1 归约树与 `skinny_body<8>` **逐字不变**；只是把同一个 `xv[s]`
// （同一个 bf16 的精确展宽）复用给 R 行 ⇒ 不改任何一次 FMA 的输入与次序。
template <int R>
__device__ __forceinline__ void m8_r_body(const bf16* __restrict__ w,
                                          const bf16* __restrict__ x,
                                          bf16* __restrict__ y,
                                          int n, int k, int m) {
    const int gwarp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row0 = gwarp * R;
    if (row0 >= n) {
        return;
    }
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    for (int j = lane; j < k; j += 32) {
        float xv[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                xv[s] = __bfloat162float(x[(size_t)s * (size_t)k + j]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < n) {
                const float wv = __bfloat162float(w[(size_t)row * (size_t)k + j]);
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s < m) {
                        acc[r][s] += wv * xv[s];
                    }
                }
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                acc[r][s] += __shfl_down_sync(0xffffffffu, acc[r][s], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < n) {
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s < m) {
                        y[(size_t)s * (size_t)n + row] = __float2bfloat16(acc[r][s]);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void lingxi_bf16_m8_r2(const bf16* __restrict__ w,
                                             const bf16* __restrict__ x,
                                             bf16* __restrict__ y, int n, int k, int m) {
    m8_r_body<2>(w, x, y, n, k, m);
}
extern "C" __global__ void lingxi_bf16_m8_r4(const bf16* __restrict__ w,
                                             const bf16* __restrict__ x,
                                             bf16* __restrict__ y, int n, int k, int m) {
    m8_r_body<4>(w, x, y, n, k, m);
}
extern "C" __global__ void lingxi_bf16_m8_r8(const bf16* __restrict__ w,
                                             const bf16* __restrict__ x,
                                             bf16* __restrict__ y, int n, int k, int m) {
    m8_r_body<8>(w, x, y, n, k, m);
}

// R=4 + 权重 PF=2：下一个 j 的 4 个行权重先发（独立 load），当前 j 的 FMAs 与之重叠。
template <int R>
__device__ __forceinline__ void m8_rpf_body(const bf16* __restrict__ w,
                                            const bf16* __restrict__ x,
                                            bf16* __restrict__ y,
                                            int n, int k, int m) {
    const int gwarp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row0 = gwarp * R;
    if (row0 >= n) {
        return;
    }
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    const bf16* w0 = w + (size_t)row0 * (size_t)k;
    float wbuf[R];
    if (lane < k) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            wbuf[r] = __bfloat162float(w0[(size_t)r * (size_t)k + lane]);
        }
    }
    for (int j = lane; j < k; j += 32) {
        float wc[R];
        float xv[8];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            wc[r] = wbuf[r];
        }
        // 先把下一迭代的 R 个权重发出去（独立 load，不等当前 FMA）
        const int jn = j + 32;
        if (jn < k) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                wbuf[r] = __bfloat162float(w0[(size_t)r * (size_t)k + jn]);
            }
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                xv[s] = __bfloat162float(x[(size_t)s * (size_t)k + j]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s < m) {
                    acc[r][s] += wc[r] * xv[s];
                }
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                acc[r][s] += __shfl_down_sync(0xffffffffu, acc[r][s], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < n) {
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s < m) {
                        y[(size_t)s * (size_t)n + row] = __float2bfloat16(acc[r][s]);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void lingxi_bf16_m8_r4pf(const bf16* __restrict__ w,
                                               const bf16* __restrict__ x,
                                               bf16* __restrict__ y, int n, int k, int m) {
    m8_rpf_body<4>(w, x, y, n, k, m);
}

// R=4 + x 先期转 f32 进 smem（每 block 一份；bf16→f32 精确 ⇒ 逐位不变）。
extern "C" __global__ void lingxi_bf16_m8_r4x(const bf16* __restrict__ w,
                                               const bf16* __restrict__ x,
                                               bf16* __restrict__ y, int n, int k, int m) {
    constexpr int R = 4;
    extern __shared__ float sx[];  // [8][k]
    const int tid = (int)threadIdx.x;
    for (int i = tid; i < m * k; i += (int)blockDim.x) {
        const int s = i / k;
        const int j = i - s * k;
        sx[i] = __bfloat162float(x[(size_t)s * (size_t)k + j]);
    }
    __syncthreads();
    const int gwarp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row0 = gwarp * R;
    if (row0 >= n) {
        return;
    }
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    for (int j = lane; j < k; j += 32) {
        float xv[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                xv[s] = sx[s * k + j];
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < n) {
                const float wv = __bfloat162float(w[(size_t)row * (size_t)k + j]);
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s < m) {
                        acc[r][s] += wv * xv[s];
                    }
                }
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                acc[r][s] += __shfl_down_sync(0xffffffffu, acc[r][s], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < n) {
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s < m) {
                        y[(size_t)s * (size_t)n + row] = __float2bfloat16(acc[r][s]);
                    }
                }
            }
        }
    }
}

// ═══════════════ A 族：row 体（一 block 一行，c0 = tid*8，uint4） ═══════════════

// 纯读 16 B/lane（生产取数形状）。
extern "C" __global__ void lingxi_bf16_row_read16(const bf16* __restrict__ w,
                                                  const bf16* __restrict__ x,
                                                  bf16* __restrict__ y,
                                                  int n, int k, int m) {
    const int row = (int)blockIdx.x;
    if (row >= n) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const std::uint8_t* wr =
        reinterpret_cast<const std::uint8_t*>(w + (size_t)row * (size_t)k);
    unsigned iacc = 0;
    int c0 = tid * 16;
    for (; c0 + 16 <= k * 2; c0 += (int)blockDim.x * 16) {
        const uint4 v = *reinterpret_cast<const uint4*>(wr + c0);
        iacc += v.x + v.y + v.z + v.w;
    }
    if (tid == 0 && iacc == 0xFFFFFFFFu) {
        y[row] = __float2bfloat16(1.0f);
    }
}

// 16 B/lane + 8 cvt + m×8 FFMA（x 用寄存器常量）。
extern "C" __global__ void lingxi_bf16_row_alu(const bf16* __restrict__ w,
                                               const bf16* __restrict__ x,
                                               bf16* __restrict__ y,
                                               int n, int k, int m) {
    const int row = (int)blockIdx.x;
    if (row >= n) {
        return;
    }
    const int tid = (int)threadIdx.x;
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
        float xc = (float)(c0 & 7) * 0.125f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    acc[s] += wf[z] * xc;
                }
            }
        }
    }
    float tot = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        tot += acc[s];
    }
    if (tid == 0 && tot == 1e30f) {
        y[row] = __float2bfloat16(1.0f);
    }
}

// 生产内层逐字复刻（epi=0、grid.y=1）。
extern "C" __global__ void lingxi_bf16_row_prod(const bf16* __restrict__ w,
                                                const bf16* __restrict__ x,
                                                bf16* __restrict__ y,
                                                int n, int k, int m) {
    const int row = (int)blockIdx.x;
    if (row >= n) {
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
            if (s < m) {
                const int4 xv =
                    *reinterpret_cast<const int4*>(x + (size_t)s * (size_t)k + c0);
                const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                float d = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    d += __bfloat162float(wp[z]) * __bfloat162float(xp[z]);
                }
                acc[s] += d;
            }
        }
    }
    __shared__ float sred[32][8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= m) {
            break;
        }
        float vv = acc[s];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            vv += __shfl_down_sync(0xffffffffu, vv, off);
        }
        if (lane == 0) {
            sred[warp][s] = vv;
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int s = 0; s < m; ++s) {
            float vv = (lane < nwarp) ? sred[lane][s] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                vv += __shfl_down_sync(0xffffffffu, vv, off);
            }
            if (lane == 0) {
                y[(size_t)s * (size_t)n + row] = __float2bfloat16(vv);
            }
        }
    }
}

// **候选结构**：一个 block 连算 R 行输出（c0 = tid*8 不变），x 的 uint4 load 与
// 8×cvt 被 R 行共用。**逐位等价**：每行 r 的线程→列映射、`d` 的 z 升序、`acc[s] += d`
// 次序、两级归约树全同（见 B 的同一论证）。
template <int R, int PF>
__device__ __forceinline__ void row_r_body(const bf16* __restrict__ w,
                                           const bf16* __restrict__ x,
                                           bf16* __restrict__ y,
                                           int n, int k, int m) {
    const int row0 = (int)blockIdx.x * R;
    if (row0 >= n) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    const int step = (int)blockDim.x * 8;
    if (PF == 0) {
        int c0 = tid * 8;
        for (; c0 + 8 <= k; c0 += step) {
            float wf[R][8];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int4 wv = *reinterpret_cast<const int4*>(
                    w + (size_t)(row0 + r) * (size_t)k + c0);
                const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    wf[r][z] = __bfloat162float(wp[z]);
                }
            }
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s < m) {
                    const int4 xv =
                        *reinterpret_cast<const int4*>(x + (size_t)s * (size_t)k + c0);
                    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                    float xf[8];
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        xf[z] = __bfloat162float(xp[z]);
                    }
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        float d = 0.0f;
#pragma unroll
                        for (int z = 0; z < 8; ++z) {
                            d += wf[r][z] * xf[z];
                        }
                        acc[r][s] += d;
                    }
                }
            }
        }
    } else {
        // PF=2：先发下一个 chunk 的 R 个权重 uint4（独立 load），与本 chunk 的 cvt/FMA 重叠
        int c0 = tid * 8;
        int4 wbuf[R];
        if (c0 + 8 <= k) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                wbuf[r] = *reinterpret_cast<const int4*>(
                    w + (size_t)(row0 + r) * (size_t)k + c0);
            }
        }
        for (; c0 + 8 <= k; c0 += step) {
            int4 wc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                wc[r] = wbuf[r];
            }
            const int cn = c0 + step;
            if (cn + 8 <= k) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    wbuf[r] = *reinterpret_cast<const int4*>(
                        w + (size_t)(row0 + r) * (size_t)k + cn);
                }
            }
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s < m) {
                    const int4 xv =
                        *reinterpret_cast<const int4*>(x + (size_t)s * (size_t)k + c0);
                    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                    float xf[8];
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        xf[z] = __bfloat162float(xp[z]);
                    }
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const bf16* wp = reinterpret_cast<const bf16*>(&wc[r]);
                        float d = 0.0f;
#pragma unroll
                        for (int z = 0; z < 8; ++z) {
                            d += __bfloat162float(wp[z]) * xf[z];
                        }
                        acc[r][s] += d;
                    }
                }
            }
        }
    }
    __shared__ float sred[32][4][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        if (row >= n) {
            break;
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= m) {
                break;
            }
            float vv = acc[r][s];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                vv += __shfl_down_sync(0xffffffffu, vv, off);
            }
            if (lane == 0) {
                sred[warp][r][s] = vv;
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= n) {
                break;
            }
            for (int s = 0; s < m; ++s) {
                float vv = (lane < nwarp) ? sred[lane][r][s] : 0.0f;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    vv += __shfl_down_sync(0xffffffffu, vv, off);
                }
                if (lane == 0) {
                    y[(size_t)s * (size_t)n + row] = __float2bfloat16(vv);
                }
            }
        }
    }
}

extern "C" __global__ void lingxi_bf16_row_r2(const bf16* __restrict__ w,
                                              const bf16* __restrict__ x,
                                              bf16* __restrict__ y, int n, int k, int m) {
    row_r_body<2, 0>(w, x, y, n, k, m);
}
extern "C" __global__ void lingxi_bf16_row_r2pf(const bf16* __restrict__ w,
                                                const bf16* __restrict__ x,
                                                bf16* __restrict__ y, int n, int k, int m) {
    row_r_body<2, 1>(w, x, y, n, k, m);
}
extern "C" __global__ void lingxi_bf16_row_r4(const bf16* __restrict__ w,
                                              const bf16* __restrict__ x,
                                              bf16* __restrict__ y, int n, int k, int m) {
    row_r_body<4, 0>(w, x, y, n, k, m);
}

// 只做 x 的读（每 token 16 B/lane，与 row 体同形），不读权重 ⇒ 直接量 x 侧自己的账。
extern "C" __global__ void lingxi_bf16_row_xonly(const bf16* __restrict__ w,
                                                 const bf16* __restrict__ x,
                                                 bf16* __restrict__ y, int n, int k, int m) {
    const int row = (int)blockIdx.x;
    if (row >= n) {
        return;
    }
    const int tid = (int)threadIdx.x;
    unsigned iacc = 0;
    int c0 = tid * 8;
    for (; c0 + 8 <= k; c0 += (int)blockDim.x * 8) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                const int4 xv = *reinterpret_cast<const int4*>(x + (size_t)s * (size_t)k + c0);
                iacc += (unsigned)xv.x + (unsigned)xv.y + (unsigned)xv.z + (unsigned)xv.w;
            }
        }
    }
    if (tid == 0 && iacc == 0xFFFFFFFFu) {
        y[row] = __float2bfloat16(1.0f);
    }
}

// ═════════════ m8 体的补充变体（x 侧账 / 预取 / x 进 smem / 逐字复刻对照） ═════════════

// **生产体逐字复刻**（含 `r0 = blockIdx.y*8` 与 `mt`）：用来定位"生产核 vs 本文件的
// 简化复刻"为何差 1.5×（2026-10-24 实测：156 GB/s vs 238 GB/s，两边都是 48 寄存器）。
extern "C" __global__ void lingxi_bf16_m8_prod_r0(const bf16* __restrict__ w,
                                                  const bf16* __restrict__ x,
                                                  bf16* __restrict__ y,
                                                  int n, int k, int m) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const bf16* wr = w + (size_t)warp * (size_t)k;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r] = 0.0f;
    }
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r0 + r < m) {
                acc[r] += wv * __bfloat162float(x[(size_t)(r0 + r) * (size_t)k + j]);
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
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r0 + r >= m) {
                continue;
            }
            y[(size_t)(r0 + r) * (size_t)n + warp] = __float2bfloat16(acc[r]);
        }
    }
}

// **r0 保留、但把 x 的行基址提到循环外**（只改取数结构/地址算式，不改任何累加）：
// 生产体在循环里算 `x + (size_t)(r0+r)*(size_t)k + j`，r0 是运行期值 ⇒ 每轮 8 个
// 64 位 IMAD 重建 8 个地址（还挡住了 MLP）。这里把 `xr = x + r0*k` 提出去，行偏移
// `r*k` 变成编译期常量，内层只剩「基址 + j」。
extern "C" __global__ void lingxi_bf16_m8_r0hoist(const bf16* __restrict__ w,
                                                  const bf16* __restrict__ x,
                                                  bf16* __restrict__ y,
                                                  int n, int k, int m) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const bf16* wr = w + (size_t)warp * (size_t)k;
    const bf16* xr = x + (size_t)r0 * (size_t)k;   // ← 唯一的改动：行基址提到循环外
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r] = 0.0f;
    }
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < mt) {
                acc[r] += wv * __bfloat162float(xr[(size_t)r * (size_t)k + j]);
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
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= mt) {
                continue;
            }
            y[(size_t)(r0 + r) * (size_t)n + warp] = __float2bfloat16(acc[r]);
        }
    }
}

// **m 编译期特化到 6**（= `gemm_skinny_m6` 的形态；生产入口目前固定发 M=8）。
// 位安全：每个 acc[r] 的 j 序列与归约树不变，只是少掉 2 个空累加器及其谓词/空 load 槽。
extern "C" __global__ void lingxi_bf16_m8_m6(const bf16* __restrict__ w,
                                             const bf16* __restrict__ x,
                                             bf16* __restrict__ y,
                                             int n, int k, int m) {
    constexpr int M = 6;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n || m < M) {
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
            y[(size_t)r * (size_t)n + warp] = __float2bfloat16(acc[r]);
        }
    }
}


// 只读 x（每 token 2 B/lane，与 m8 体同形），不读权重。
extern "C" __global__ void lingxi_bf16_m8_xonly(const bf16* __restrict__ w,
                                                const bf16* __restrict__ x,
                                                bf16* __restrict__ y, int n, int k, int m) {
    int row, lane;
    m8_row_of(row, lane);
    if (row >= n) {
        return;
    }
    unsigned iacc = 0;
    for (int j = lane; j < k; j += 32) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                iacc += (unsigned)reinterpret_cast<const std::uint16_t*>(
                    x + (size_t)s * (size_t)k)[j];
            }
        }
    }
    if (lane == 0 && iacc == 0xFFFFFFFFu) {
        y[row] = __float2bfloat16(1.0f);
    }
}

// R=1 + 权重 PF=4 移位预取（的同一招：滚动窗口把 4 轮 load 一次挂上）。
extern "C" __global__ void lingxi_bf16_m8_r1pf(const bf16* __restrict__ w,
                                               const bf16* __restrict__ x,
                                               bf16* __restrict__ y, int n, int k, int m) {
    int row, lane;
    m8_row_of(row, lane);
    if (row >= n) {
        return;
    }
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    constexpr int PF = 4;
    float wbuf[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) {
        const int jp = lane + p * 32;
        wbuf[p] = (jp < k) ? __bfloat162float(wr[jp]) : 0.0f;
    }
    for (int j = lane; j < k; j += 32) {
        const float wv = wbuf[0];
        float xv[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                xv[s] = __bfloat162float(x[(size_t)s * (size_t)k + j]);
            }
        }
#pragma unroll
        for (int p = 0; p + 1 < PF; ++p) {
            wbuf[p] = wbuf[p + 1];
        }
        const int jn = j + PF * 32;
        wbuf[PF - 1] = (jn < k) ? __bfloat162float(wr[jn]) : 0.0f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                acc[s] += wv * xv[s];
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[s] += __shfl_down_sync(0xffffffffu, acc[s], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                y[(size_t)s * (size_t)n + row] = __float2bfloat16(acc[s]);
            }
        }
    }
}

// R=1 + x 先期转 f32 进 smem（每 block 一份；bf16→f32 精确）。j 维读 smem 是**无冲突**的
// （lane 读连续 float）。⇒ 内层从「6×LDG.U16 + 6 cvt」降到「6×LDS.32」。
extern "C" __global__ void lingxi_bf16_m8_r1x(const bf16* __restrict__ w,
                                              const bf16* __restrict__ x,
                                              bf16* __restrict__ y, int n, int k, int m) {
    extern __shared__ float sx[];  // [8][k]
    const int tid = (int)threadIdx.x;
    for (int i = tid; i < m * k; i += (int)blockDim.x) {
        const int s = i / k;
        const int j = i - s * k;
        sx[i] = __bfloat162float(x[(size_t)s * (size_t)k + j]);
    }
    __syncthreads();
    int row, lane;
    m8_row_of(row, lane);
    if (row >= n) {
        return;
    }
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                acc[s] += wv * sx[s * k + j];
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[s] += __shfl_down_sync(0xffffffffu, acc[s], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                y[(size_t)s * (size_t)n + row] = __float2bfloat16(acc[s]);
            }
        }
    }
}

// R=2 + x 进 smem f32（x 侧 LDS 与权重 load 都减半）。
extern "C" __global__ void lingxi_bf16_m8_r2x(const bf16* __restrict__ w,
                                              const bf16* __restrict__ x,
                                              bf16* __restrict__ y, int n, int k, int m) {
    constexpr int R = 2;
    extern __shared__ float sx[];  // [8][k]
    const int tid = (int)threadIdx.x;
    for (int i = tid; i < m * k; i += (int)blockDim.x) {
        const int s = i / k;
        const int j = i - s * k;
        sx[i] = __bfloat162float(x[(size_t)s * (size_t)k + j]);
    }
    __syncthreads();
    const int gwarp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row0 = gwarp * R;
    if (row0 >= n) {
        return;
    }
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    for (int j = lane; j < k; j += 32) {
        float xv[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < m) {
                xv[s] = sx[s * k + j];
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < n) {
                const float wv = __bfloat162float(w[(size_t)row * (size_t)k + j]);
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s < m) {
                        acc[r][s] += wv * xv[s];
                    }
                }
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                acc[r][s] += __shfl_down_sync(0xffffffffu, acc[r][s], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < n) {
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s < m) {
                        y[(size_t)s * (size_t)n + row] = __float2bfloat16(acc[r][s]);
                    }
                }
            }
        }
    }
}

// ═════════════ B 族：草稿 lm_head（Q4G64 稠密 GEMV，一 warp 一行，c0 = lane*8） ═════════════
//
// 生产核 `lingxi_gemv_q4g64_dense_epi` 的主段：每个 lane 一次 32 位 load 取 8 个 nibble
// （`c0 = lane*8`，整 warp 覆盖 256 列 = 一个 group 的 32 字节），每 group 再读一个 fp16 scale。

// 纯读 payload 4 B/lane + 每 group 一个 scale（= 生产核的取数形状，不解包）。
extern "C" __global__ void lingxi_q4_head_read1(const std::uint8_t* __restrict__ w,
                                                const bf16* __restrict__ x,
                                                bf16* __restrict__ y,
                                                int n, int k, int k_pad, int row_off,
                                                long long scale_off) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const int row = row_off + warp;
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* payload = w + (long long)row * row_bytes;
    const __half* scales =
        reinterpret_cast<const __half*>(w + scale_off) + (long long)row * row_groups;
    unsigned iacc = 0;
    for (int c0 = lane * 8; c0 + 8 <= k; c0 += 256) {
        const int g = c0 >> 6;
        const int t = c0 & 63;
        const std::uint32_t word =
            *reinterpret_cast<const std::uint32_t*>(payload + g * 32 + (t >> 1));
        iacc += word + (unsigned)__half_as_ushort(scales[g]);
    }
    if (lane == 0 && iacc == 0xFFFFFFFFu) {
        y[warp] = __float2bfloat16(1.0f);
    }
}

// 纯读 payload 16 B/lane（结构上限；不含 scale）。
extern "C" __global__ void lingxi_q4_head_read16(const std::uint8_t* __restrict__ w,
                                                 const bf16* __restrict__ x,
                                                 bf16* __restrict__ y,
                                                 int n, int k, int k_pad, int row_off,
                                                 long long scale_off) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const int row = row_off + warp;
    const int row_bytes = k_pad >> 1;
    const std::uint8_t* payload = w + (long long)row * row_bytes;
    unsigned iacc = 0;
    for (int o = lane * 16; o + 16 <= row_bytes; o += 512) {
        const uint4 v = *reinterpret_cast<const uint4*>(payload + o);
        iacc += v.x + v.y + v.z + v.w;
    }
    if (lane == 0 && iacc == 0xFFFFFFFFu) {
        y[warp] = __float2bfloat16(1.0f);
    }
}

// 生产内层复刻 + **权重 PF=2**（10 次迭代的滚动窗口；只提前 load，不改任何算式）。
extern "C" __global__ void lingxi_q4_head_pf(const std::uint8_t* __restrict__ w,
                                             const bf16* __restrict__ x,
                                             bf16* __restrict__ y,
                                             int n, int k, int k_pad, int row_off,
                                             long long scale_off) {
    __shared__ float2 c_lut[256];
    for (int lb = (int)threadIdx.x; lb < 256; lb += (int)blockDim.x) {
        const int lo = (lb & 0xF) >= 8 ? (lb & 0xF) - 16 : (lb & 0xF);
        const int hi = (lb >> 4) >= 8 ? (lb >> 4) - 16 : (lb >> 4);
        c_lut[lb] = make_float2((float)lo, (float)hi);
    }
    __syncthreads();
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const int row = row_off + warp;
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* payload = w + (long long)row * row_bytes;
    const __half* scales =
        reinterpret_cast<const __half*>(w + scale_off) + (long long)row * row_groups;
    float acc = 0.0f;
    constexpr int PF = 2;
    std::uint32_t wbuf[PF];
    float sbuf[PF];
    int4 xbuf[PF];
    int c0 = lane * 8;
#pragma unroll
    for (int p = 0; p < PF; ++p) {
        const int cp = c0 + p * 256;
        if (cp + 8 <= k) {
            const int g = cp >> 6;
            const int t = cp & 63;
            wbuf[p] = *reinterpret_cast<const std::uint32_t*>(payload + g * 32 + (t >> 1));
            sbuf[p] = __half2float(scales[g]);
            xbuf[p] = *reinterpret_cast<const int4*>(x + cp);
        }
    }
    for (; c0 + 8 <= k; c0 += 256) {
        const std::uint32_t wc = wbuf[0];
        const float scc = sbuf[0];
        const int4 xc = xbuf[0];
#pragma unroll
        for (int p = 0; p + 1 < PF; ++p) {
            wbuf[p] = wbuf[p + 1];
            sbuf[p] = sbuf[p + 1];
            xbuf[p] = xbuf[p + 1];
        }
        const int cn = c0 + PF * 256;
        if (cn + 8 <= k) {
            const int g = cn >> 6;
            const int t = cn & 63;
            wbuf[PF - 1] =
                *reinterpret_cast<const std::uint32_t*>(payload + g * 32 + (t >> 1));
            sbuf[PF - 1] = __half2float(scales[g]);
            xbuf[PF - 1] = *reinterpret_cast<const int4*>(x + cn);
        }
        const bf16* xp = reinterpret_cast<const bf16*>(&xc);
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int b8 = (int)((wc >> (8 * i)) & 0xFFu);
            const float2 c2 = c_lut[b8];
            dot = fmaf(c2.x, __bfloat162float(xp[2 * i]), dot);
            dot = fmaf(c2.y, __bfloat162float(xp[2 * i + 1]), dot);
        }
        acc += scc * dot;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        y[warp] = __float2bfloat16(acc);
    }
}

// 生产内层复刻 + **一个 warp 连算 R=2 行**（x 的 16 B load 被 2 行共用；每行的
// lane→列集合、8 个 nibble 的 z 次序、`acc += scale*dot` 次序、shfl 树全同 ⇒ 逐位等价）。
extern "C" __global__ void lingxi_q4_head_r2(const std::uint8_t* __restrict__ w,
                                             const bf16* __restrict__ x,
                                             bf16* __restrict__ y,
                                             int n, int k, int k_pad, int row_off,
                                             long long scale_off) {
    __shared__ float2 c_lut[256];
    for (int lb = (int)threadIdx.x; lb < 256; lb += (int)blockDim.x) {
        const int lo = (lb & 0xF) >= 8 ? (lb & 0xF) - 16 : (lb & 0xF);
        const int hi = (lb >> 4) >= 8 ? (lb >> 4) - 16 : (lb >> 4);
        c_lut[lb] = make_float2((float)lo, (float)hi);
    }
    __syncthreads();
    constexpr int R = 2;
    const int gwarp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int w0 = gwarp * R;
    if (w0 >= n) {
        return;
    }
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0f;
    }
    for (int c0 = lane * 8; c0 + 8 <= k; c0 += 256) {
        const int g = c0 >> 6;
        const int t = c0 & 63;
        const int4 xv = *reinterpret_cast<const int4*>(x + c0);
        const bf16* xp = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row_off + w0 + r;
            if (w0 + r >= n) {
                break;
            }
            const std::uint8_t* payload = w + (long long)row * row_bytes;
            const __half* scales =
                reinterpret_cast<const __half*>(w + scale_off) + (long long)row * row_groups;
            const std::uint32_t word =
                *reinterpret_cast<const std::uint32_t*>(payload + g * 32 + (t >> 1));
            const float sc = __half2float(scales[g]);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int b8 = (int)((word >> (8 * i)) & 0xFFu);
                const float2 c2 = c_lut[b8];
                dot = fmaf(c2.x, __bfloat162float(xp[2 * i]), dot);
                dot = fmaf(c2.y, __bfloat162float(xp[2 * i + 1]), dot);
            }
            acc[r] += sc * dot;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (w0 + r < n) {
                y[w0 + r] = __float2bfloat16(acc[r]);
            }
        }
    }
}

// 生产内层逐字复刻（epi=0）⇒ 自检：应与 `lingxi_gemv_q4g64_dense_epi` 逐位相同。
extern "C" __global__ void lingxi_q4_head_prod(const std::uint8_t* __restrict__ w,
                                               const bf16* __restrict__ x,
                                               bf16* __restrict__ y,
                                               int n, int k, int k_pad, int row_off,
                                               long long scale_off) {
    __shared__ float2 c_lut[256];
    for (int lb = (int)threadIdx.x; lb < 256; lb += (int)blockDim.x) {
        const int lo = (lb & 0xF) >= 8 ? (lb & 0xF) - 16 : (lb & 0xF);
        const int hi = (lb >> 4) >= 8 ? (lb >> 4) - 16 : (lb >> 4);
        c_lut[lb] = make_float2((float)lo, (float)hi);
    }
    __syncthreads();
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const int row = row_off + warp;
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* payload = w + (long long)row * row_bytes;
    const __half* scales =
        reinterpret_cast<const __half*>(w + scale_off) + (long long)row * row_groups;
    float acc = 0.0f;
    for (int c0 = lane * 8; c0 + 8 <= k; c0 += 256) {
        const int g = c0 >> 6;
        const int t = c0 & 63;
        const std::uint32_t word =
            *reinterpret_cast<const std::uint32_t*>(payload + g * 32 + (t >> 1));
        const int4 xv = *reinterpret_cast<const int4*>(x + c0);
        const bf16* xp = reinterpret_cast<const bf16*>(&xv);
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int b8 = (int)((word >> (8 * i)) & 0xFFu);
            const float2 c2 = c_lut[b8];
            dot = fmaf(c2.x, __bfloat162float(xp[2 * i]), dot);
            dot = fmaf(c2.y, __bfloat162float(xp[2 * i + 1]), dot);
        }
        acc += __half2float(scales[g]) * dot;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        y[warp] = __float2bfloat16(acc);
    }
}
