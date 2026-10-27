// **实验核**（2026-10-24）：稠密 Q4G64 瘦长 GEMM（`gemm_q4_skinny_bf16`）的取数结构探针。
// 只有 `tests/t14_q4_dense_skinny_bw.rs` 会调；生产路径（`dequant_ops.cu` 的
// `lingxi_gemm_q4_skinny_bf16` / `_s8` / `_ab`）一个字节都没动。
//
// 背景：专家段 Q4/Q3 GEMV 只换取数结构（算式一字未改、逐位相同）就白捡 1.6~2.0×
// （/）。稠密投影是**另一个核**（m>1 的瘦长 GEMM，每 warp 一行输出、
// x 走 shared 舞台、每 lane 每次 4 B 权重 + m 个 token 的 smem 读 + FMA），
// 实测 m=6 只有 125~132 GB/s。本文件把这条差距拆成四层，**地址映射与生产核逐字节相同**：
//
// mode 0 = 权重纯读 4 B/lane（生产核的取数指令形状，不解包）
// mode 1 = 权重纯读 16 B/lane `uint4`（同一批字节的结构上限）
// mode 2 = 4 B/lane + nibble 解包 + I2F（**ALU 地板**，不碰 x）
// mode 4 = staging（x→shared）+ 权重 4 B 纯读（隔离舞台/同步的代价）
// mode 5 = 4 B + 解包 + I2F + 8 FMA/token（x 用常量 ⇒ 隔离 smem x 读的代价）
// mode 6 = **只做 staging**（不读权重）⇒ 直接量出"x 舞台"自己的账
// mode 7 = 生产内层，但 x **直接从 global 读**（无舞台、无 barrier）⇒ 隔离"舞台 vs 直读"
// mode 3 = 生产内层全量（staging + 4 B + 解包 + smem x + FMA）⇒ 应与生产核同速
//
// 判据：GB/s 按**权重 payload 字节**（n·k/2）折算，与 的表同口径。
// ⚠️ mode 0/1/2/5 不分配 dynamic smem（纯取数结构，别被 16 KB 舞台压住占用）；
// mode 3/4 分配 `8 × 1024 × 2 B`（与生产核的 `shared_mem_bytes` 逐字节相同）。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace {

// 与生产核的 `Q4G_ROWS` / `Q4G_KTILE` 相同（dequant_ops.cu）。
constexpr int kRows = 8;
constexpr int kKtile = 1024;

__device__ __forceinline__ void q4d_probe_body(const std::uint8_t* __restrict__ wq,
                                               const bf16* __restrict__ x, bf16* __restrict__ y,
                                               int n, int k, int k_pad, int m, int row_off,
                                               long long scale_off, int mode) {
    extern __shared__ bf16 sx[]; // [8, kKtile]，布局与生产核相同
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = (int)blockIdx.x * kRows + warp;
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const bool valid = row < n;
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* wp = wq + (long long)(row_off + row) * (long long)row_bytes;
    const __half* scp = reinterpret_cast<const __half*>(wq + scale_off) +
                        (long long)(row_off + row) * (long long)row_groups;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }

    if (mode == 1) {
        // 16 B/lane：32 lane × 16 B = 512 B / 指令（同一行的同一批字节，结构上限）
        int iacc = 0;
        for (int o = lane * 16; o + 16 <= row_bytes; o += 512) {
            const uint4 v = *reinterpret_cast<const uint4*>(wp + o);
            iacc += (int)(v.x & 0xFFFFu) + (int)(v.y & 0xFFFFu) + (int)(v.z & 0xFFFFu) +
                    (int)(v.w & 0xFFFFu);
        }
        acc[0] = (float)iacc;
    } else {
        const bool stage = (mode == 3 || mode == 4 || mode == 6);
        for (int k0 = 0; k0 < k; k0 += kKtile) {
            const int kn = (k - k0) < kKtile ? (k - k0) : kKtile;
            if (stage) {
                // 与生产核逐字相同的 x 舞台
                const int nvec_per_tok = (kn + 7) >> 3;
                const int nvec = nvec_per_tok * mt;
                for (int i = tid; i < nvec; i += (int)blockDim.x) {
                    const int s = i / nvec_per_tok;
                    const int c = (i % nvec_per_tok) << 3;
                    const bf16* xs = x + (size_t)(r0 + s) * (size_t)k + (size_t)k0 + c;
                    bf16* dst = sx + s * kKtile + c;
                    if (c + 8 <= kn) {
                        *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(xs);
                    } else {
                        for (int z = 0; z < 8; ++z) {
                            dst[z] = (c + z < kn) ? xs[z] : __float2bfloat16(0.0f);
                        }
                    }
                }
                __syncthreads();
            }
            if (valid && mode != 6) {
                const int ncol = (kn + 7) & ~7;
                int iacc = 0;
                for (int c0 = lane * 8; c0 < ncol; c0 += 256) {
                    const int c = k0 + c0;
                    const int g = c >> 6;
                    const int t = c & 63;
                    const std::uint32_t word =
                        *reinterpret_cast<const std::uint32_t*>(wp + g * 32 + (t >> 1));
                    if (mode == 0) {
                        iacc += (int)word;
                        continue;
                    }
                    const float sc = __half2float(scp[g]);
                    float cd[8];
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        const int nb = (int)((word >> (4 * z)) & 0xFu);
                        cd[z] = (float)(nb >= 8 ? nb - 16 : nb);
                    }
                    if (mode == 2) {
                        // ALU 地板：解包 + I2F + 8 次串行 FADD（不碰 x）
#pragma unroll
                        for (int z = 0; z < 8; ++z) {
                            acc[0] += cd[z];
                        }
                        continue;
                    }
#pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        if (s >= mt) {
                            break;
                        }
                        if (mode == 5) {
                            // 8 FFMA/token，但 x 用常量（不读 shared）
#pragma unroll
                            for (int z = 0; z < 8; ++z) {
                                acc[s] += sc * cd[z] * (float)(z + 1);
                            }
                        } else {
                            // mode 3/4：x 走 shared 舞台；mode 7：x 直接从 global 读（无舞台）
                            const int4 xv = (mode == 7)
                                ? *reinterpret_cast<const int4*>(
                                      x + (size_t)(r0 + s) * (size_t)k + (size_t)k0 + c0)
                                : *reinterpret_cast<const int4*>(sx + s * kKtile + c0);
                            const bf16* xb = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
                            for (int z = 0; z < 8; ++z) {
                                acc[s] += sc * cd[z] * __bfloat162float(xb[z]);
                            }
                        }
                    }
                }
                acc[0] += (float)iacc; // 只保证 mode 0 的累加不被优化掉
            }
            if (stage) {
                __syncthreads();
            }
        }
    }

    // 收尾：与生产核逐字相同的 warp 归约 + lane==0 写 bf16（探针里 epi 恒 0）
    if (valid) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= mt) {
                break;
            }
            float rr = acc[s];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                rr += __shfl_down_sync(0xffffffffu, rr, off);
            }
            if (lane == 0) {
                y[(size_t)(r0 + s) * (size_t)n + row] = __float2bfloat16(rr);
            }
        }
    }
}

} // namespace

extern "C" __global__ void lingxi_q4_dense_probe(const std::uint8_t* __restrict__ wq,
                                                 const bf16* __restrict__ x, bf16* __restrict__ y,
                                                 int n, int k, int k_pad, int m, int row_off,
                                                 long long scale_off, int mode) {
    q4d_probe_body(wq, x, y, n, k, k_pad, m, row_off, scale_off, mode);
}

// ───────────────────── 多行/warp 变体（逐位等价，只有结构不同）─────────────────────
//
// 动机（探针量出来的账）：生产核每 4 B 权重（8 个 code）要为**每个 token** 付
// 1 次 smem 读 + 8 次 bf16→f32 + 8 次 FFMA；x 的读取与转换被 R 个输出行重复付账，
// 同时每 block 只摊 8 个输出行 ⇒ x 的 global 舞台流量是权重的 m/2 倍（m=6 时 3 倍）。
// 本变体让**一个 warp 连算 R 个输出行**（行号 `warp*R + r`，仍是"一 lane 8 列"）：
// * x 的 smem 读与 bf16→f32 只做一次、被 R 行共用；
// * 每 block 摊 8R 个输出行 ⇒ 舞台流量与 barrier 次数都 ÷R；
// * R 个权重字是**独立 load**（专家段 t2 那招的同一形态）。
//
// **位安全论据**：每行的 lane→列集合（`lane*8 + 256*i`）、chunk 次序、z=0..7 的串行
// 累加次序、`sc*cd*xf` 的乘序、以及收尾的 shfl 归约树全部与 `gemm_q4_skinny_bf16`
// 逐字相同；`xf[z]` 只是把同一个 bf16 **精确**展宽成 f32（bf16→f32 无损）后复用。
template <int R, int W>
__device__ __forceinline__ void q4d_rows_body(const std::uint8_t* __restrict__ wq,
                                              const bf16* __restrict__ x, bf16* __restrict__ y,
                                              int n, int k, int k_pad, int m, int row_off,
                                              long long scale_off) {
    extern __shared__ bf16 sx[]; // [8, kKtile]
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int r0 = (int)blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const int base = ((int)blockIdx.x * W + warp) * R;
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;

    const std::uint8_t* wp[R];
    const __half* scp[R];
    bool ok[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = base + r;
        ok[r] = row < n;
        const long long gi = (long long)(row_off + (ok[r] ? row : 0));
        wp[r] = wq + gi * (long long)row_bytes;
        scp[r] = reinterpret_cast<const __half*>(wq + scale_off) + gi * (long long)row_groups;
    }

    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[r][s] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < k; k0 += kKtile) {
        const int kn = (k - k0) < kKtile ? (k - k0) : kKtile;
        {
            // 与生产核逐字相同的 x 舞台
            const int nvec_per_tok = (kn + 7) >> 3;
            const int nvec = nvec_per_tok * mt;
            for (int i = tid; i < nvec; i += (int)blockDim.x) {
                const int s = i / nvec_per_tok;
                const int c = (i % nvec_per_tok) << 3;
                const bf16* xs = x + (size_t)(r0 + s) * (size_t)k + (size_t)k0 + c;
                bf16* dst = sx + s * kKtile + c;
                if (c + 8 <= kn) {
                    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(xs);
                } else {
                    for (int z = 0; z < 8; ++z) {
                        dst[z] = (c + z < kn) ? xs[z] : __float2bfloat16(0.0f);
                    }
                }
            }
        }
        __syncthreads();
        const int ncol = (kn + 7) & ~7;
        for (int c0 = lane * 8; c0 < ncol; c0 += 256) {
            const int c = k0 + c0;
            const int g = c >> 6;
            const int t = c & 63;
            std::uint32_t word[R];
            float sc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                word[r] = *reinterpret_cast<const std::uint32_t*>(wp[r] + g * 32 + (t >> 1));
                sc[r] = __half2float(scp[r][g]);
            }
            float cd[R][8];
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    const int nb = (int)((word[r] >> (4 * z)) & 0xFu);
                    cd[r][z] = (float)(nb >= 8 ? nb - 16 : nb);
                }
            }
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= mt) {
                    break;
                }
                const int4 xv = *reinterpret_cast<const int4*>(sx + s * kKtile + c0);
                const bf16* xb = reinterpret_cast<const bf16*>(&xv);
                float xf[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    xf[z] = __bfloat162float(xb[z]);
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        acc[r][s] += sc[r] * cd[r][z] * xf[z];
                    }
                }
            }
        }
        __syncthreads();
    }

    // 写回：与生产核逐字相同的 shfl 归约树 + bf16 round 点
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (!ok[r]) {
            continue;
        }
        const int row = base + r;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= mt) {
                break;
            }
            float rr = acc[r][s];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                rr += __shfl_down_sync(0xffffffffu, rr, off);
            }
            if (lane == 0) {
                y[(size_t)(r0 + s) * (size_t)n + row] = __float2bfloat16(rr);
            }
        }
    }
}

// 五个独立符号（**必须分开**：合在一个符号里 ptxas 按最坏变体分配寄存器，R4 的 104 个
// 寄存器会把 R1/R2 的占用一起拖下去）。grid/blockDim 由调用方按 (R, W) 折算：
// `grid.x = ceil(n / (W*R))`、`blockDim = 32*W`。
#define LINGXI_Q4D_ROWS(NAME, R, W)                                                           \
    extern "C" __global__ void NAME(const std::uint8_t* __restrict__ wq,                      \
                                    const bf16* __restrict__ x, bf16* __restrict__ y, int n,    \
                                    int k, int k_pad, int m, int row_off,                      \
                                    long long scale_off) {                                     \
        q4d_rows_body<R, W>(wq, x, y, n, k, k_pad, m, row_off, scale_off);                      \
    }

LINGXI_Q4D_ROWS(lingxi_q4_dense_rows_r1, 1, 8)
LINGXI_Q4D_ROWS(lingxi_q4_dense_rows_r2, 2, 8)
LINGXI_Q4D_ROWS(lingxi_q4_dense_rows_r4, 4, 8)
LINGXI_Q4D_ROWS(lingxi_q4_dense_rows_r1w16, 1, 16)
LINGXI_Q4D_ROWS(lingxi_q4_dense_rows_r2w16, 2, 16)

#endif // sm_70
