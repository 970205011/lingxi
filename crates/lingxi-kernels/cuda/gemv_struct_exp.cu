// **实验核**（2026-10-24）：专家段 Q4G64 GEMV 的"乘加结构"微基准。
// 只有 `tests/t12_moe_gemv_struct_bw.rs` 会调；生产路径（`slotsm_rpw_*`）一个字节都没动。
//
// 目的： 量到"纯读 783 → +解包 602 → +I2F 596 → +x读/FFMA/scale **271**"，
// 596 之后掉的那一半到底是什么、换乘加结构能不能拿回来。两组核：
//
// A. 阶梯探针（`q4_probe_body<STAGE>`，与生产核**逐字节相同的访存形状**：
// 4 B/lane、lane 的第 i 块 = 列 `lane*8 + 256*i`）：
// P0 只读 P1 +nibble 解包（整数） P2 +I2F
// P3 +x 载入（8 FADD，无乘法） P4 +8 FFMA（x 用常量，不载入）
// P5 +x 载入 + 8 FFMA（无 scale） P6 +scale（= 生产算式）
// P7 = P6 但解包用"移位对"（2 条指令/值，生产是 4 条）
// P8 = P6 但 scale 由 block 预载进 smem P9 = P8 + x 预转 f32 进 smem
//
// B. 结构变体（都用 uint4 = 16 B/lane 搬权重，一行 Q4 = 1280 B）：
// u128_r1 / r2 / r4：一个 warp 负责 1/2/4 个输出行，k 方向 32 lane 并行；
// contig：每 lane 连续 16 个 Q4 word（= 128 权重）再归约；
// u128_r2_sm：r2 + scale 与 x(f32) 都进 smem；
// u128_r1_dp4a：16 B 载入 + int8 激活 + `__dp4a`（整数累加、最后一次性 I2F，
// **非位等价**，只用于判断"CVT/FFMA 是不是那 300 GB/s 的锅"）。
//
// 归约/写出与生产核一致（warp 内 shuffle 归约、`y[(s0+s)*n + row] = bf16(acc)`）；
// 列集合与累加次序**变**（结构本来就不同）⇒ 除 r1/r2/r4 的算式等价外，都不保证逐位相同。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

// ───────────────────────── 0) 公共小工具 ─────────────────────────

// DEC=0：生产核的解包（右移+与+符号选择，4~5 条/值）
// DEC=1：移位对（`(int)(w<<(28-4z))>>28`，2 条/值）
template <int DEC>
__device__ __forceinline__ void q4_decode8(std::uint32_t wc, float* code) {
#pragma unroll
    for (int z = 0; z < 8; ++z) {
        if (DEC == 0) {
            const int nb = (int)((wc >> (4 * z)) & 0xFu);
            code[z] = (float)(nb >= 8 ? nb - 16 : nb);
        } else {
            code[z] = (float)((int)(wc << (28 - 4 * z)) >> 28);
        }
    }
}

__device__ __forceinline__ void warp_reduce_store(bf16* __restrict__ y, long long idx, float v,
                                                  int lane) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
        y[idx] = __float2bfloat16(v);
    }
}

// ───────────────────────── 1) 阶梯探针 ─────────────────────────
template <int STAGE>
__device__ __forceinline__ void q4_probe_body(const std::uint8_t* __restrict__ slots,
                                              const int* __restrict__ slot_ids,
                                              const bf16* __restrict__ x,
                                              const int* __restrict__ group_off,
                                              bf16* __restrict__ y, int E, int n, int k,
                                              int k_pad, int a_stride, long long blob,
                                              long long seg_off, long long scale_off) {
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int m = group_off[e + 1] - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp >= n) {
        return;
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const int xrow = (a_stride == 0) ? 0 : s0;
    const bf16* xp = x + (size_t)xrow * (size_t)a_stride;
    float acc = 0.0f;
    int iacc = 0;
    int c0 = lane * 8;
    for (; c0 + 8 <= k; c0 += 256) {
        const std::uint32_t wc = *reinterpret_cast<const std::uint32_t*>(
            payload + (long long)warp * row_bytes + (c0 >> 1));
        if (STAGE == 0) {
            iacc += (int)(wc & 0xFFFFu) + (int)(wc >> 16);
        } else if (STAGE == 1) {
            int t = 0;
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                const int nb = (int)((wc >> (4 * z)) & 0xFu);
                t += (nb >= 8 ? nb - 16 : nb);
            }
            iacc += t;
        } else if (STAGE == 2) {
            float code[8];
            q4_decode8<0>(wc, code);
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                acc += code[z];
            }
        } else if (STAGE == 3) {
            float code[8];
            q4_decode8<0>(wc, code);
            const int4 xv = *reinterpret_cast<const int4*>(xp + c0);
            const bf16* xq = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                acc += code[z] + __bfloat162float(xq[z]);
            }
        } else if (STAGE == 4) {
            const float xc = 0.01f;
            float code[8];
            q4_decode8<0>(wc, code);
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                acc += code[z] * xc;
            }
        } else {
            float code[8];
            q4_decode8<(STAGE == 7) ? 1 : 0>(wc, code);
            const int4 xv = *reinterpret_cast<const int4*>(xp + c0);
            const bf16* xq = reinterpret_cast<const bf16*>(&xv);
            float d = 0.0f;
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                d += code[z] * __bfloat162float(xq[z]);
            }
            if (STAGE == 5) {
                acc += d;
            } else {
                const int g = c0 >> 6;
                const float scc = __half2float(scales[(long long)warp * row_groups + g]);
                acc += scc * d;
            }
        }
    }
    acc += (float)iacc;
    warp_reduce_store(y, (long long)s0 * n + warp, acc, lane);
}

#define LINGXI_Q4_PROBE(S)                                                                  \
    extern "C" __global__ void lingxi_q4g64_p##S(                                          \
        const std::uint8_t* __restrict__ slots, const int* __restrict__ slot_ids,            \
        const bf16* __restrict__ x, const int* __restrict__ group_off,                       \
        bf16* __restrict__ y, int E, int n, int k, int k_pad, int a_stride,                  \
        long long blob, long long seg_off, long long scale_off) {                            \
        q4_probe_body<S>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride, blob,    \
                         seg_off, scale_off);                                                \
    }

LINGXI_Q4_PROBE(0)
LINGXI_Q4_PROBE(1)
LINGXI_Q4_PROBE(2)
LINGXI_Q4_PROBE(3)
LINGXI_Q4_PROBE(4)
LINGXI_Q4_PROBE(5)
LINGXI_Q4_PROBE(6)
LINGXI_Q4_PROBE(7)

// ── P8：scale 由 block 预载进 smem（判定"每轮的 scale LDG.16 是不是暗坑"）──
// 每 block 8 个 warp = 8 个输出行，每行 row_groups（gu=40）个 fp16 scale ⇒ ≤1 KB。
extern "C" __global__ void lingxi_q4g64_p8_smsc(const std::uint8_t* __restrict__ slots,
                                                const int* __restrict__ slot_ids,
                                                const bf16* __restrict__ x,
                                                const int* __restrict__ group_off,
                                                bf16* __restrict__ y, int E, int n, int k,
                                                int k_pad, int a_stride, long long blob,
                                                long long seg_off, long long scale_off) {
    __shared__ float ssc[8][64];
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int m = group_off[e + 1] - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int wl = (int)(threadIdx.x >> 5);              // warp 在 block 内序号
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    if (row_groups > 64) {
        return;  // 微基准守卫（gu 的 row_groups=40）
    }
    const int row0 = (int)blockIdx.x * 8;                // 本 block 的首行（= warp - wl）
    for (int i = (int)threadIdx.x; i < 8 * row_groups; i += (int)blockDim.x) {
        const int r = i / row_groups;
        const int g = i - r * row_groups;
        if (row0 + r < n) {
            ssc[r][g] = __half2float(scales[(long long)(row0 + r) * row_groups + g]);
        }
    }
    __syncthreads();
    if (warp >= n) {
        return;
    }
    const int xrow = (a_stride == 0) ? 0 : s0;
    const bf16* xp = x + (size_t)xrow * (size_t)a_stride;
    float acc = 0.0f;
    int c0 = lane * 8;
    for (; c0 + 8 <= k; c0 += 256) {
        const std::uint32_t wc = *reinterpret_cast<const std::uint32_t*>(
            payload + (long long)warp * row_bytes + (c0 >> 1));
        float code[8];
        q4_decode8<0>(wc, code);
        const int4 xv = *reinterpret_cast<const int4*>(xp + c0);
        const bf16* xq = reinterpret_cast<const bf16*>(&xv);
        float d = 0.0f;
#pragma unroll
        for (int z = 0; z < 8; ++z) {
            d += code[z] * __bfloat162float(xq[z]);
        }
        acc += ssc[wl][c0 >> 6] * d;
    }
    warp_reduce_store(y, (long long)s0 * n + warp, acc, lane);
}

// ── P9：P8 + x 预转 f32 进 smem（只对 a_stride==0 的单行激活成立）──
// 每 block 8 个 warp 共用同一行 x（m=1 的线上形状）⇒ x 的 global/L1 流量 ÷8。
extern "C" __global__ void lingxi_q4g64_p9_smscx(const std::uint8_t* __restrict__ slots,
                                                 const int* __restrict__ slot_ids,
                                                 const bf16* __restrict__ x,
                                                 const int* __restrict__ group_off,
                                                 bf16* __restrict__ y, int E, int n, int k,
                                                 int k_pad, int a_stride, long long blob,
                                                 long long seg_off, long long scale_off) {
    __shared__ float ssc[8][64];
    extern __shared__ float sxf[];
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int m = group_off[e + 1] - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int wl = (int)(threadIdx.x >> 5);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    if (row_groups > 64 || a_stride != 0) {
        return;  // 微基准守卫（只管 m=1 的线上形状）
    }
    const int row0 = (int)blockIdx.x * 8;
    for (int i = (int)threadIdx.x; i < 8 * row_groups; i += (int)blockDim.x) {
        const int r = i / row_groups;
        const int g = i - r * row_groups;
        if (row0 + r < n) {
            ssc[r][g] = __half2float(scales[(long long)(row0 + r) * row_groups + g]);
        }
    }
    for (int i = (int)threadIdx.x; i < k; i += (int)blockDim.x) {
        sxf[i] = __bfloat162float(x[(size_t)s0 * (size_t)a_stride + i]);
    }
    __syncthreads();
    if (warp >= n) {
        return;
    }
    float acc = 0.0f;
    int c0 = lane * 8;
    for (; c0 + 8 <= k; c0 += 256) {
        const std::uint32_t wc = *reinterpret_cast<const std::uint32_t*>(
            payload + (long long)warp * row_bytes + (c0 >> 1));
        float code[8];
        q4_decode8<0>(wc, code);
        float d = 0.0f;
#pragma unroll
        for (int z = 0; z < 8; ++z) {
            d += code[z] * sxf[c0 + z];
        }
        acc += ssc[wl][c0 >> 6] * d;
    }
    warp_reduce_store(y, (long long)s0 * n + warp, acc, lane);
}

// ───────────────────────── 2) 结构变体：uint4（16 B/lane）─────────────────────────
// 一个 warp 负责 R 个输出行；lane 的第 i 块 = 列 `lane*32 + 1024*i`（32 个权重 = 16 B），
// 每块恰好落在**一个** 64 权重 group 内（16 B 对齐 ⇒ c0>>6 就是 scale 下标）。
// 权重只读一遍；x 被 R 行共用（每个 8 元素组只载入一次）。
template <int R, int MB>
__device__ __forceinline__ void q4_u128_body(const std::uint8_t* __restrict__ slots,
                                             const int* __restrict__ slot_ids,
                                             const bf16* __restrict__ x,
                                             const int* __restrict__ group_off,
                                             bf16* __restrict__ y, int E, int n, int k,
                                             int k_pad, int a_stride, long long blob,
                                             long long seg_off, long long scale_off) {
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int m = group_off[e + 1] - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row0 = ((int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5)) * R;
    if (row0 >= n) {
        return;
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    // 行数（末尾块可能不足 R 行）
    const int nr = (n - row0) < R ? (n - row0) : R;
    for (int sb = 0; sb < m; sb += MB) {
        const int mb = (m - sb) < MB ? (m - sb) : MB;
        const bf16* xbp = x + (size_t)(s0 + sb) * (size_t)a_stride;
        float acc[R][MB];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                acc[r][s] = 0.0f;
            }
        }
        for (int c0 = lane * 32; c0 + 32 <= k; c0 += 1024) {
            const int g = c0 >> 6;
            std::uint32_t wq[R][4];
            float scv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r >= nr) {
                    break;
                }
                const uint4 wv = *reinterpret_cast<const uint4*>(
                    payload + (long long)(row0 + r) * row_bytes + (c0 >> 1));
                wq[r][0] = wv.x;
                wq[r][1] = wv.y;
                wq[r][2] = wv.z;
                wq[r][3] = wv.w;
                scv[r] = __half2float(scales[(long long)(row0 + r) * row_groups + g]);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float code[R][8];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r >= nr) {
                        break;
                    }
                    q4_decode8<1>(wq[r][q], code[r]);
                }
#pragma unroll
                for (int s = 0; s < MB; ++s) {
                    if (s >= mb) {
                        break;
                    }
                    const int4 xv =
                        *reinterpret_cast<const int4*>(xbp + (size_t)s * (size_t)a_stride + c0 +
                                                       q * 8);
                    const bf16* xq = reinterpret_cast<const bf16*>(&xv);
                    float xf[8];
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        xf[z] = __bfloat162float(xq[z]);
                    }
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (r >= nr) {
                            break;
                        }
                        float d = 0.0f;
#pragma unroll
                        for (int z = 0; z < 8; ++z) {
                            d += code[r][z] * xf[z];
                        }
                        acc[r][s] += scv[r] * d;
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r >= nr) {
                break;
            }
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                if (s >= mb) {
                    break;
                }
                warp_reduce_store(y, (long long)(s0 + sb + s) * n + row0 + r, acc[r][s], lane);
            }
        }
    }
}

extern "C" __global__ void lingxi_q4g64_u128_r1(const std::uint8_t* __restrict__ slots,
                                                const int* __restrict__ slot_ids,
                                                const bf16* __restrict__ x,
                                                const int* __restrict__ group_off,
                                                bf16* __restrict__ y, int E, int n, int k,
                                                int k_pad, int a_stride, long long blob,
                                                long long seg_off, long long scale_off) {
    q4_u128_body<1, 8>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride, blob, seg_off,
                       scale_off);
}

extern "C" __global__ void lingxi_q4g64_u128_r2(const std::uint8_t* __restrict__ slots,
                                                const int* __restrict__ slot_ids,
                                                const bf16* __restrict__ x,
                                                const int* __restrict__ group_off,
                                                bf16* __restrict__ y, int E, int n, int k,
                                                int k_pad, int a_stride, long long blob,
                                                long long seg_off, long long scale_off) {
    q4_u128_body<2, 4>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride, blob, seg_off,
                       scale_off);
}

extern "C" __global__ void lingxi_q4g64_u128_r4(const std::uint8_t* __restrict__ slots,
                                                const int* __restrict__ slot_ids,
                                                const bf16* __restrict__ x,
                                                const int* __restrict__ group_off,
                                                bf16* __restrict__ y, int E, int n, int k,
                                                int k_pad, int a_stride, long long blob,
                                                long long seg_off, long long scale_off) {
    q4_u128_body<4, 2>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride, blob, seg_off,
                       scale_off);
}

// ── contig：每 lane 连续 16 个 Q4 word（64 B = 128 权重）再归约 ──
// 行内 `lanes_per_row = row_bytes/64` 条 lane 参与（gu: 20/32）；每个 lane 两块 group
// （64 B = 2 × 32 B），用 4 条独立累加链（每 word 一条）拉长无依赖链。
extern "C" __global__ void lingxi_q4g64_contig(const std::uint8_t* __restrict__ slots,
                                               const int* __restrict__ slot_ids,
                                               const bf16* __restrict__ x,
                                               const int* __restrict__ group_off,
                                               bf16* __restrict__ y, int E, int n, int k,
                                               int k_pad, int a_stride, long long blob,
                                               long long seg_off, long long scale_off) {
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int m = group_off[e + 1] - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp >= n) {
        return;
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const int lanes_per_row = row_bytes >> 6;  // 64 B / lane
    const bool active = lane < lanes_per_row;  // ⚠️ 不 early-return：shuffle 归约要全 lane 参与
    const int c0 = lane * 128;                    // 本 lane 的起始列
    const int g0 = (c0 >> 6);                     // 前 64 权重的 group
    const float sc0 = active ? __half2float(scales[(long long)warp * row_groups + g0]) : 0.0f;
    const float sc1 = active ? __half2float(scales[(long long)warp * row_groups + g0 + 1]) : 0.0f;
    constexpr int MB = 8;
    for (int sb = 0; sb < m; sb += MB) {
        const int mb = (m - sb) < MB ? (m - sb) : MB;
        const bf16* xbp = x + (size_t)(s0 + sb) * (size_t)a_stride;
        float acc[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            acc[s] = 0.0f;
        }
        uint4 wv[4];
        if (active) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                wv[q] = *reinterpret_cast<const uint4*>(
                    payload + (long long)warp * row_bytes + (c0 >> 1) + q * 16);
            }
        }
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            if (!active) {
                break;
            }
            const bf16* xp = xbp + (size_t)s * (size_t)a_stride + c0;
            float d0 = 0.0f;  // 前 64 权重（word 0..7）
            float d1 = 0.0f;  // 后 64 权重（word 8..15）
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const std::uint32_t* wp = reinterpret_cast<const std::uint32_t*>(&wv[q]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float code[8];
                    q4_decode8<1>(wp[j], code);
                    const int4 xvv = *reinterpret_cast<const int4*>(xp + q * 32 + j * 8);
                    const bf16* xq = reinterpret_cast<const bf16*>(&xvv);
                    float d = 0.0f;
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        d += code[z] * __bfloat162float(xq[z]);
                    }
                    if (q < 2) {
                        d0 += d;
                    } else {
                        d1 += d;
                    }
                }
            }
            acc[s] += (sc0 * d0 + sc1 * d1);
        }
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            warp_reduce_store(y, (long long)(s0 + sb + s) * n + warp, acc[s], lane);
        }
    }
}

// ───────────────────────── 3) 生产算式 × 预取结构 ─────────────────────────
// P6/P7 已经证明：**同一份算式**（逐位相同）在"每轮直接 load、不搞软件流水"的写法下能到
// 459/555 GB/s，而生产核只有 286 —— 差的不是数学，是 `slotsm_body` 的 PF=4
// 移位预取（`wbuf[p]=wbuf[p+1]` 串成依赖链 + 74 寄存器）。
// 下面把三种预取结构放在**同一个多对批处理骨架**（MB=8，生产同款）上量：
// `unr1`：每轮直 load（= P6 的骨架 + 多对）
// `unr2/4`：一次发 2/4 个**独立** load 再算（无移位链）
// `shift4`：逐字复刻生产的 PF=4 移位管线
// DEC: 0 = 生产解包 / 1 = 移位对。全部**逐位等价**（code 值与累加次序不变）。
template <int UNR, int DEC>
__device__ __forceinline__ void q4_slotsm_unroll_body(
        const std::uint8_t* __restrict__ slots, const int* __restrict__ slot_ids,
        const bf16* __restrict__ x, const int* __restrict__ group_off, bf16* __restrict__ y,
        int E, int n, int k, int k_pad, int a_stride, long long blob, long long seg_off,
        long long scale_off) {
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int m = group_off[e + 1] - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp >= n) {
        return;
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* prow = payload + (long long)warp * row_bytes;
    const __half* srow = scales + (long long)warp * row_groups;
    constexpr int MB = 8;
    for (int sb = 0; sb < m; sb += MB) {
        const int mb = (m - sb) < MB ? (m - sb) : MB;
        const bf16* xbp = x + (size_t)(s0 + sb) * (size_t)a_stride;
        float acc[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            acc[s] = 0.0f;
        }
        const int step = 256 * UNR;
        int c0 = lane * 8;
        for (; c0 + 256 * (UNR - 1) + 8 <= k; c0 += step) {
            std::uint32_t wv[UNR];
            float scv[UNR];
#pragma unroll
            for (int u = 0; u < UNR; ++u) {
                wv[u] = *reinterpret_cast<const std::uint32_t*>(prow + ((c0 + 256 * u) >> 1));
                scv[u] = __half2float(srow[(c0 + 256 * u) >> 6]);
            }
#pragma unroll
            for (int u = 0; u < UNR; ++u) {
                float code[8];
                q4_decode8<DEC>(wv[u], code);
#pragma unroll
                for (int s = 0; s < MB; ++s) {
                    if (s >= mb) {
                        break;
                    }
                    const int4 xv = *reinterpret_cast<const int4*>(
                        xbp + (size_t)s * (size_t)a_stride + c0 + 256 * u);
                    const bf16* xq = reinterpret_cast<const bf16*>(&xv);
                    float d = 0.0f;
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        d += code[z] * __bfloat162float(xq[z]);
                    }
                    acc[s] += scv[u] * d;
                }
            }
        }
        for (; c0 + 8 <= k; c0 += 256) {
            const std::uint32_t wc = *reinterpret_cast<const std::uint32_t*>(prow + (c0 >> 1));
            const float scc = __half2float(srow[c0 >> 6]);
            float code[8];
            q4_decode8<DEC>(wc, code);
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                if (s >= mb) {
                    break;
                }
                const int4 xv = *reinterpret_cast<const int4*>(
                    xbp + (size_t)s * (size_t)a_stride + c0);
                const bf16* xq = reinterpret_cast<const bf16*>(&xv);
                float d = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    d += code[z] * __bfloat162float(xq[z]);
                }
                acc[s] += scc * d;
            }
        }
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            warp_reduce_store(y, (long long)(s0 + sb + s) * n + warp, acc[s], lane);
        }
    }
}

/// 生产核 PF=4 移位管线的**逐字复刻**（判定"预取实现"值多少）。
template <int DEC>
__device__ __forceinline__ void q4_slotsm_shift4_body(
        const std::uint8_t* __restrict__ slots, const int* __restrict__ slot_ids,
        const bf16* __restrict__ x, const int* __restrict__ group_off, bf16* __restrict__ y,
        int E, int n, int k, int k_pad, int a_stride, long long blob, long long seg_off,
        long long scale_off) {
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int m = group_off[e + 1] - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp >= n) {
        return;
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* prow = payload + (long long)warp * row_bytes;
    const __half* srow = scales + (long long)warp * row_groups;
    constexpr int MB = 8;
    constexpr int PF = 4;
    for (int sb = 0; sb < m; sb += MB) {
        const int mb = (m - sb) < MB ? (m - sb) : MB;
        const bf16* xbp = x + (size_t)(s0 + sb) * (size_t)a_stride;
        float acc[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            acc[s] = 0.0f;
        }
        std::uint32_t wbuf[PF];
        float sbuf[PF];
        int c0 = lane * 8;
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cp = c0 + p * 256;
            if (cp + 8 <= k) {
                wbuf[p] = *reinterpret_cast<const std::uint32_t*>(prow + (cp >> 1));
                sbuf[p] = __half2float(srow[cp >> 6]);
            }
        }
        for (; c0 + 8 <= k; c0 += 256) {
            const std::uint32_t wc = wbuf[0];
            const float scc = sbuf[0];
#pragma unroll
            for (int p = 0; p + 1 < PF; ++p) {
                wbuf[p] = wbuf[p + 1];
                sbuf[p] = sbuf[p + 1];
            }
            const int cn = c0 + PF * 256;
            if (cn + 8 <= k) {
                wbuf[PF - 1] = *reinterpret_cast<const std::uint32_t*>(prow + (cn >> 1));
                sbuf[PF - 1] = __half2float(srow[cn >> 6]);
            }
            float code[8];
            q4_decode8<DEC>(wc, code);
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                if (s >= mb) {
                    break;
                }
                const int4 xv =
                    *reinterpret_cast<const int4*>(xbp + (size_t)s * (size_t)a_stride + c0);
                const bf16* xq = reinterpret_cast<const bf16*>(&xv);
                float d = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    d += code[z] * __bfloat162float(xq[z]);
                }
                acc[s] += scc * d;
            }
        }
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            warp_reduce_store(y, (long long)(s0 + sb + s) * n + warp, acc[s], lane);
        }
    }
}

#define LINGXI_Q4_TIER(NAME, UNR, DEC)                                                      \
    extern "C" __global__ void NAME(                                                       \
        const std::uint8_t* __restrict__ slots, const int* __restrict__ slot_ids,            \
        const bf16* __restrict__ x, const int* __restrict__ group_off,                       \
        bf16* __restrict__ y, int E, int n, int k, int k_pad, int a_stride,                  \
        long long blob, long long seg_off, long long scale_off) {                            \
        q4_slotsm_unroll_body<UNR, DEC>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,     \
                                        a_stride, blob, seg_off, scale_off);                 \
    }

LINGXI_Q4_TIER(lingxi_q4g64_tier_unr1, 1, 1)
LINGXI_Q4_TIER(lingxi_q4g64_tier_unr2, 2, 1)
LINGXI_Q4_TIER(lingxi_q4g64_tier_unr4, 4, 1)
LINGXI_Q4_TIER(lingxi_q4g64_tier_unr4_proddec, 4, 0)

extern "C" __global__ void lingxi_q4g64_tier_shift4(const std::uint8_t* __restrict__ slots,
                                                    const int* __restrict__ slot_ids,
                                                    const bf16* __restrict__ x,
                                                    const int* __restrict__ group_off,
                                                    bf16* __restrict__ y, int E, int n, int k,
                                                    int k_pad, int a_stride, long long blob,
                                                    long long seg_off, long long scale_off) {
    q4_slotsm_shift4_body<1>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride, blob,
                             seg_off, scale_off);
}

// ── u128_r2_sm：R=2 + scale 与 x(f32) 都在 smem（a_stride==0 的单行激活）──
extern "C" __global__ void lingxi_q4g64_u128_r2_sm(const std::uint8_t* __restrict__ slots,
                                                   const int* __restrict__ slot_ids,
                                                   const bf16* __restrict__ x,
                                                   const int* __restrict__ group_off,
                                                   bf16* __restrict__ y, int E, int n, int k,
                                                   int k_pad, int a_stride, long long blob,
                                                   long long seg_off, long long scale_off) {
    __shared__ float ssc[16][64];
    extern __shared__ float sxf[];
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int m = group_off[e + 1] - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int wl = (int)(threadIdx.x >> 5);
    const int block_row0 = (int)blockIdx.x * warps_per_block * 2;  // 本 block 首行
    const int row0 = block_row0 + wl * 2;                           // 本 warp 首行
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    if (row_groups > 64 || a_stride != 0 || m > 4) {
        return;  // 微基准守卫
    }
    const int bnr = warps_per_block * 2;  // 本 block 覆盖的行数（smem 槽位数）
    for (int i = (int)threadIdx.x; i < bnr * row_groups; i += (int)blockDim.x) {
        const int r = i / row_groups;
        const int g = i - r * row_groups;
        if (block_row0 + r < n) {
            ssc[r][g] = __half2float(scales[(long long)(block_row0 + r) * row_groups + g]);
        }
    }
    for (int i = (int)threadIdx.x; i < k; i += (int)blockDim.x) {
        sxf[i] = __bfloat162float(x[s0 * (size_t)a_stride + i]);
    }
    __syncthreads();
    if (row0 >= n) {
        return;
    }
    const int nr = (n - row0) < 2 ? (n - row0) : 2;
    float acc[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    for (int c0 = lane * 32; c0 + 32 <= k; c0 += 1024) {
        const int g = c0 >> 6;
        std::uint32_t wq[2][4];
        const float scv0 = ssc[wl * 2 + 0][g];
        const float scv1 = ssc[wl * 2 + 1][g];
        const uint4 wv0 = *reinterpret_cast<const uint4*>(
            payload + (long long)(row0 + 0) * row_bytes + (c0 >> 1));
        wq[0][0] = wv0.x;
        wq[0][1] = wv0.y;
        wq[0][2] = wv0.z;
        wq[0][3] = wv0.w;
        if (nr > 1) {
            const uint4 wv1 = *reinterpret_cast<const uint4*>(
                payload + (long long)(row0 + 1) * row_bytes + (c0 >> 1));
            wq[1][0] = wv1.x;
            wq[1][1] = wv1.y;
            wq[1][2] = wv1.z;
            wq[1][3] = wv1.w;
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float code0[8];
            float code1[8];
            q4_decode8<1>(wq[0][q], code0);
            if (nr > 1) {
                q4_decode8<1>(wq[1][q], code1);
            }
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                if (s >= m) {
                    break;
                }
                float xf[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    xf[z] = sxf[c0 + q * 8 + z];
                }
                float d0 = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    d0 += code0[z] * xf[z];
                }
                acc[0][s] += scv0 * d0;
                if (nr > 1) {
                    float d1 = 0.0f;
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        d1 += code1[z] * xf[z];
                    }
                    acc[1][s] += scv1 * d1;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (r >= nr) {
            break;
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s >= m) {
                break;
            }
            warp_reduce_store(y, (long long)(s0 + s) * n + row0 + r, acc[r][s], lane);
        }
    }
}

// ── u128_r1_dp4a：16 B 载入 + int8 激活（`__dp4a` 整数累加，最后一次性 I2F）──
// ⚠️ **非位等价**：x 先按行量化成 int8（±127），误差 ~2^-8。
extern "C" __global__ void lingxi_q4g64_u128_r1_dp4a(const std::uint8_t* __restrict__ slots,
                                                     const int* __restrict__ slot_ids,
                                                     const bf16* __restrict__ x,
                                                     const int* __restrict__ group_off,
                                                     bf16* __restrict__ y, int E, int n, int k,
                                                     int k_pad, int a_stride, long long blob,
                                                     long long seg_off, long long scale_off,
                                                     const signed char* __restrict__ xq,
                                                     const float* __restrict__ sx) {
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int m = group_off[e + 1] - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp >= n) {
        return;
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    constexpr int MB = 8;
    for (int sb = 0; sb < m; sb += MB) {
        const int mb = (m - sb) < MB ? (m - sb) : MB;
        float ssx[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            ssx[s] = sx[(a_stride == 0) ? 0 : (s0 + sb + s)];
        }
        float acc[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            acc[s] = 0.0f;
        }
        for (int c0 = lane * 32; c0 + 32 <= k; c0 += 1024) {
            const int g = c0 >> 6;
            const uint4 wv = *reinterpret_cast<const uint4*>(
                payload + (long long)warp * row_bytes + (c0 >> 1));
            const float scc = __half2float(scales[(long long)warp * row_groups + g]);
            const std::uint32_t* wp = reinterpret_cast<const std::uint32_t*>(&wv);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const std::uint32_t wc = wp[q];
                const std::uint32_t se =
                    __vsub4((wc & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
                const std::uint32_t so =
                    __vsub4(((wc >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
                const std::uint32_t a_lo = __byte_perm(se, so, 0x5140);
                const std::uint32_t a_hi = __byte_perm(se, so, 0x7362);
#pragma unroll
                for (int s = 0; s < MB; ++s) {
                    if (s >= mb) {
                        break;
                    }
                    const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                    const uint2 xv = *reinterpret_cast<const uint2*>(
                        xq + (size_t)xrow * (size_t)k + c0 + q * 8);
                    int d = __dp4a((int)a_lo, (int)xv.x, 0);
                    d = __dp4a((int)a_hi, (int)xv.y, d);
                    acc[s] += scc * (ssx[s] * (float)d);
                }
            }
        }
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            warp_reduce_store(y, (long long)(s0 + sb + s) * n + warp, acc[s], lane);
        }
    }
}
