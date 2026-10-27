// **实验核**（2026-10-24）：Q3G128 专家 GEMV 的取数结构微基准。
// 只有 `tests/t13_moe_q3_gemv_bw.rs` 会调；生产路径（`gemv_q3g128_slotsm_*`）一个字节都没动。
//
// 背景：Q4 的 `slotsm_t2_bf16` 把 PF=4 移位预取换成"两个独立 load + 移位对解包"，
// gu E40 m=1 317→453 GB/s；Q3 仍是老体（`gemv_q3g128_slotsm_body`：每 8 列 **3 次
// 1 字节 `__ldg`** + PF=4 移位链 + 4~5 条/值的解包）。本文件把 Q3 的差距拆开量：
//
// A. 同形纯读探针（`lingxi_q3_read_probe`，与生产核**同一地址映射**）：
// mode 0 = 每次 3×1 B `__ldg`（生产核的取数指令形状）
// mode 1 = 每次 2 个对齐 u32 + `__funnelshift_r`（候选结构的取数指令形状；
// ⚠️ 单个 4 B 未对齐 load 在 V100 上直接 MISALIGNED_ADDRESS，这条路不存在）
// mode 2 = 16 B/lane `uint4`（同一批字节的"结构上限"，Q4 那边是 783 GB/s）
// mode 3 = 对齐 u32 对 + 移位对解包 + I2F（再叠一层 ALU，判解码代价）
//
// B. 计算变体（`q3_slotsm_body<UNR, DEC, LOAD>`，全部**逐位等价**于生产核）：
// UNR = 每轮发出 1/2/4 个 chunk 的权重与 scale（独立 load，取代 PF=4 移位链）
// DEC = 0 生产解包（右移+与+符号选择）/ 1 移位对（`(int)(w<<(29-3z))>>29`）
// LOAD = 0 三字节 __ldg / 2 两个对齐 u32 + `__funnelshift_r`
//
// 位安全论据（与 Q4 t2 同款）：lane 的列集合（`lane*8 + 256*round`）、chunk 次序、
// 每 chunk 内 z=0..7 的串行累加次序、`acc[s] += scc*d` 的次序全部不变；
// 三种取数/两种解包解出的都是**同一个整数 code**（3-bit 二补码）⇒ 逐位相同。
//
// ⚠️ LOAD=1/2 会多读 1~4 字节：偏移是行内 `3*(c0>>3)`，最后一 chunk 读到的行尾之后
// 仍是**同一段的 scale 平面**（布局 `[rows*row_bytes][align256][scales]`，scale 平面
// 恒非空）⇒ 不会越出该槽的 blob。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

// ───────────────────────── 0) 解码与取数 ─────────────────────────

// DEC=0：生产式（与 `dequant_ops.cu` 的 `lingxi_q3_code8` 逐字相同）
// DEC=1：移位对（2 条指令/值；解出的是同一个整数）
template <int DEC>
__device__ __forceinline__ float q3_code8(std::uint32_t w24, int z) {
    if (DEC == 0) {
        const int w3 = (int)((w24 >> (3 * z)) & 7u);
        return (float)(w3 >= 4 ? w3 - 8 : w3);
    }
    return (float)((int)(w24 << (29 - 3 * z)) >> 29);
}

// 行内字节偏移 `B`（3 的倍数）处的 24 位窗口（bits [3c0, 3c0+24)）。
// ⚠️ LOAD=1「一个未对齐 4 B load」**在 V100 上直接 CUDA_ERROR_MISALIGNED_ADDRESS**
// （实测：mode 1 纯读即崩）——`B` 是 3 的倍数，4 B 访问天然做不到对齐 ⇒ 这条路不存在。
// 可行的是 LOAD=2：两个**对齐** u32（`4*(B>>2)` 与 +4）+ `__funnelshift_r` 拼出窗口。
// row_bytes = groups*48 恒为 16 的倍数 ⇒ 行首 16 B 对齐，两个 word 都合法。
template <int LOAD>
__device__ __forceinline__ std::uint32_t q3_window(const std::uint8_t* __restrict__ row, int B) {
    if (LOAD == 0) {
        const std::uint32_t b0 = (std::uint32_t)__ldg(row + B);
        const std::uint32_t b1 = (std::uint32_t)__ldg(row + B + 1);
        const std::uint32_t b2 = (std::uint32_t)__ldg(row + B + 2);
        return b0 | (b1 << 8) | (b2 << 16);
    } else {
        const int w0 = B >> 2;
        const std::uint32_t a = *reinterpret_cast<const std::uint32_t*>(row + (w0 << 2));
        const std::uint32_t b = *reinterpret_cast<const std::uint32_t*>(row + (w0 << 2) + 4);
        return __funnelshift_r(a, b, (B & 3) * 8);
    }
}

__device__ __forceinline__ void q3_warp_reduce_store(bf16* __restrict__ y, long long idx, float v,
                                                     int lane) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
        y[idx] = __float2bfloat16(v);
    }
}

// ───────────────────────── 1) 同形纯读探针 ─────────────────────────
// 地址映射与生产核逐字节相同（warp 一行、lane 的 chunk = `lane*8 + 256*i`）。
extern "C" __global__ void lingxi_q3_read_probe(const std::uint8_t* __restrict__ slots,
                                                float* __restrict__ out,
                                                int rows_per_expert,
                                                int row_bytes,
                                                int k,
                                                long long blob,
                                                int mode) {
    const int e = blockIdx.y;
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp >= rows_per_expert) {
        return;
    }
    const std::uint8_t* row = slots + (long long)e * blob + (long long)warp * row_bytes;
    float acc = 0.0f;
    int iacc = 0;
    if (mode == 2) {
        // 16 B/lane ⇒ 32 lane × 16 B = 512 B / 指令（同一批字节，纯结构上限）
        for (int off = lane * 16; off + 16 <= row_bytes; off += 512) {
            const uint4 v = *reinterpret_cast<const uint4*>(row + off);
            iacc += (int)(v.x & 0xFFFFu) + (int)(v.y & 0xFFFFu) + (int)(v.z & 0xFFFFu)
                    + (int)(v.w & 0xFFFFu);
        }
    } else {
        for (int c0 = lane * 8; c0 + 8 <= k; c0 += 256) {
            const int B = 3 * (c0 >> 3);
            if (mode == 0) {
                // 生产核的取数指令形状：3 次 1 字节
                const std::uint32_t b0 = (std::uint32_t)__ldg(row + B);
                const std::uint32_t b1 = (std::uint32_t)__ldg(row + B + 1);
                const std::uint32_t b2 = (std::uint32_t)__ldg(row + B + 2);
                iacc += (int)((b0 | (b1 << 8) | (b2 << 16)) & 0xFFFFFFu);
            } else if (mode == 1) {
                const int w0 = B >> 2;
                const std::uint32_t a = *reinterpret_cast<const std::uint32_t*>(row + (w0 << 2));
                const std::uint32_t b = *reinterpret_cast<const std::uint32_t*>(row + (w0 << 2) + 4);
                const std::uint32_t w = __funnelshift_r(a, b, (B & 3) * 8);
                iacc += (int)(w & 0xFFFFu) + (int)((w >> 16) & 0xFFu);
            } else {
                const std::uint32_t w = q3_window<2>(row, B);
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    acc += q3_code8<1>(w, z);
                }
            }
        }
    }
    acc += (float)iacc;
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)e * rows_per_expert + warp] = acc;
    }
}

// ───────────────────────── 2) 计算变体（逐位等价）─────────────────────────
// 骨架照抄生产 `lingxi_gemv_q3g128_slotsm_body<1>`（MB=8 分批、xf==nullptr 的 bf16 现转、
// `acc[s] += scc*d`），只把"取数 + 解包"换掉。
template <int UNR, int DEC, int LOAD, int MB>
__device__ __forceinline__ void q3_slotsm_body(const std::uint8_t* __restrict__ slots,
                                               const int* __restrict__ slot_ids,
                                               const bf16* __restrict__ x,
                                               const int* __restrict__ group_off,
                                               bf16* __restrict__ y, int E, int n, int k, int k_pad,
                                               int a_stride, long long blob, long long seg_off,
                                               long long scale_off,
                                               const int* __restrict__ gidx) {
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int gi = (gidx != nullptr) ? gidx[e] : e;
    const int s0 = group_off[gi];
    const int s1 = group_off[gi + 1];
    const int m = s1 - s0;
    if (m <= 0 || slot_ids[gi] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp >= n) {
        return;
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[gi] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = (k_pad / 128) * 48;
    const int row_groups = k_pad / 128;
    const std::uint8_t* prow = payload + (long long)warp * row_bytes;
    const __half* srow = scales + (long long)warp * row_groups;
    for (int sb = 0; sb < m; sb += MB) {
        const int mb = (m - sb) < MB ? (m - sb) : MB;
        const bf16* xbp = x + (size_t)(s0 + sb) * (size_t)a_stride;
        float acc[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            acc[s] = 0.0f;
        }
        int c0 = lane * 8;
        // UNR 个 chunk 一组：权重与 scale **一次全发出去**（独立 load，无移位链）
        for (; c0 + 256 * (UNR - 1) + 8 <= k; c0 += 256 * UNR) {
            std::uint32_t wv[UNR];
            float scv[UNR];
#pragma unroll
            for (int u = 0; u < UNR; ++u) {
                const int cc = c0 + 256 * u;
                wv[u] = q3_window<LOAD>(prow, 3 * (cc >> 3));
                scv[u] = __half2float(srow[cc >> 7]);
            }
#pragma unroll
            for (int u = 0; u < UNR; ++u) {
                float code[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    code[z] = q3_code8<DEC>(wv[u], z);
                }
#pragma unroll
                for (int s = 0; s < MB; ++s) {
                    if (s >= mb) {
                        break;
                    }
                    const int4 xv = *reinterpret_cast<const int4*>(
                        xbp + (size_t)s * (size_t)a_stride + c0 + 256 * u);
                    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                    float d = 0.0f;
                    d += code[0] * __bfloat162float(xp[0]);
                    d += code[1] * __bfloat162float(xp[1]);
                    d += code[2] * __bfloat162float(xp[2]);
                    d += code[3] * __bfloat162float(xp[3]);
                    d += code[4] * __bfloat162float(xp[4]);
                    d += code[5] * __bfloat162float(xp[5]);
                    d += code[6] * __bfloat162float(xp[6]);
                    d += code[7] * __bfloat162float(xp[7]);
                    acc[s] += scv[u] * d;
                }
            }
        }
        // 尾部：单 chunk（chunk 次序与生产核的 `c0 += 256` 完全一致）
        for (; c0 + 8 <= k; c0 += 256) {
            const float scc = __half2float(srow[c0 >> 7]);
            const std::uint32_t wc = q3_window<LOAD>(prow, 3 * (c0 >> 3));
            float code[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                code[z] = q3_code8<DEC>(wc, z);
            }
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                if (s >= mb) {
                    break;
                }
                const int4 xv =
                    *reinterpret_cast<const int4*>(xbp + (size_t)s * (size_t)a_stride + c0);
                const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                float d = 0.0f;
                d += code[0] * __bfloat162float(xp[0]);
                d += code[1] * __bfloat162float(xp[1]);
                d += code[2] * __bfloat162float(xp[2]);
                d += code[3] * __bfloat162float(xp[3]);
                d += code[4] * __bfloat162float(xp[4]);
                d += code[5] * __bfloat162float(xp[5]);
                d += code[6] * __bfloat162float(xp[6]);
                d += code[7] * __bfloat162float(xp[7]);
                acc[s] += scc * d;
            }
        }
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            q3_warp_reduce_store(y, (long long)(s0 + sb + s) * n + warp, acc[s], lane);
        }
    }
}

#define LINGXI_Q3_EXP(NAME, UNR, DEC, LOAD)                                                     \
    extern "C" __global__ void NAME(const std::uint8_t* __restrict__ slots,                     \
                                    const int* __restrict__ slot_ids,                           \
                                    const bf16* __restrict__ x,                                 \
                                    const int* __restrict__ group_off, bf16* __restrict__ y,    \
                                    int E, int n, int k, int k_pad, int a_stride,                \
                                    long long blob, long long seg_off, long long scale_off,     \
                                    const int* __restrict__ gidx) {                             \
        q3_slotsm_body<UNR, DEC, LOAD, 8>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,      \
                                          a_stride, blob, seg_off, scale_off, gidx);             \
    }

// MB=1 特化：**只在"每组恰好 1 对"时可用**（rows=1 的 decode 恒成立）。
// 与 MB=8 逐位相同（少掉的只是 `s>=mb` 的 7 次空转），但省 7×2 条比较/分支 + 7 个累加器。
#define LINGXI_Q3_EXP_MB1(NAME, UNR, DEC, LOAD)                                                 \
    extern "C" __global__ void NAME(const std::uint8_t* __restrict__ slots,                     \
                                    const int* __restrict__ slot_ids,                           \
                                    const bf16* __restrict__ x,                                 \
                                    const int* __restrict__ group_off, bf16* __restrict__ y,    \
                                    int E, int n, int k, int k_pad, int a_stride,                \
                                    long long blob, long long seg_off, long long scale_off,     \
                                    const int* __restrict__ gidx) {                             \
        q3_slotsm_body<UNR, DEC, LOAD, 1>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,      \
                                          a_stride, blob, seg_off, scale_off, gidx);             \
    }

// 单 chunk（只去掉 PF=4 移位链）
LINGXI_Q3_EXP(lingxi_q3g128_exp_unr1_b3_pdec, 1, 0, 0)
// 2 chunk × 生产式（只去掉移位链）
LINGXI_Q3_EXP(lingxi_q3g128_exp_unr2_b3_pdec, 2, 0, 0)
// 2 chunk × 生产解包 × 对齐 u32 对（只换取数宽度）
LINGXI_Q3_EXP(lingxi_q3g128_exp_unr2_u32_pdec, 2, 0, 2)
// 2 chunk × 移位对解包 × 三字节 load（只换解包）
LINGXI_Q3_EXP(lingxi_q3g128_exp_unr2_b3_sdec, 2, 1, 0)
// ★ 候选 t2：2 chunk × 移位对解包 × 对齐 u32 对
LINGXI_Q3_EXP(lingxi_q3g128_exp_unr2_u32_sdec, 2, 1, 2)
// 1 / 4 / 8 chunk 的候选（判 UNR 该多大）
LINGXI_Q3_EXP(lingxi_q3g128_exp_unr1_u32_sdec, 1, 1, 2)
LINGXI_Q3_EXP(lingxi_q3g128_exp_unr4_u32_sdec, 4, 1, 2)
LINGXI_Q3_EXP(lingxi_q3g128_exp_unr8_u32_sdec, 8, 1, 2)
// MB=1 特化（只在每组 1 对时用；线上 rows=1 decode 恒成立）
LINGXI_Q3_EXP_MB1(lingxi_q3g128_exp_mb1_unr1_u32_sdec, 1, 1, 2)
LINGXI_Q3_EXP_MB1(lingxi_q3g128_exp_mb1_unr2_u32_sdec, 2, 1, 2)
LINGXI_Q3_EXP_MB1(lingxi_q3g128_exp_mb1_unr4_u32_sdec, 4, 1, 2)
LINGXI_Q3_EXP_MB1(lingxi_q3g128_exp_mb1_unr2_b3_sdec, 2, 1, 0)
