// **实验核**（2026-10-24）：判定专家段 GEMV 的 275 GB/s 天花板到底是"结构上限"
// 还是"并行度/占用率不足"。**只有 microbench（`tests/t11_moe_gemv_par_bw.rs`）会调**，
// 生产路径（`slotsm_rpw`）一个字节都没动。
//
// 三个探针：
// 1. `lingxi_gemv_q4g64_slotsm_splitk_bf16` —— k 方向切 S 段（gridDim.z），
// 每段算部分和写 fp32，再由 `lingxi_gemv_splitk_reduce_bf16` 求和；
// 2. `lingxi_read_probe_slots` —— 按**和 slotsm 完全相同的访存形状**只读不算，
// 量"这个形状下硬件能给多少"；
// 3. `lingxi_read_probe_flat` —— 经典 grid-stride uint4 纯读，量机器的读带宽上限。
//
// ⚠️ split-k **不保证与 `slotsm` 逐位相同**（求和次序变了），只用于有损档评估。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

// ───────────────────────── 1) split-k Q4G64 GEMV ─────────────────────────
// 布局/解码/归约与 `lingxi_gemv_q4g64_slotsm_body<1>` 完全一致（lane 的列集合、
// 每列所属 group、nibble 抽取次序都不变），**唯一区别**是 k 方向的 256 列 chunk
// 按 blockIdx.z 切开 ⇒ 每个 block 只算一段，部分和落 fp32。
extern "C" __global__ void lingxi_gemv_q4g64_slotsm_splitk_bf16(
    const std::uint8_t* __restrict__ slots,
    const int* __restrict__ slot_ids,
    const bf16* __restrict__ x,
    const int* __restrict__ group_off,
    float* __restrict__ part,
    int E,
    int n,
    int k,
    int k_pad,
    int a_stride,
    long long blob,
    long long seg_off,
    long long scale_off,
    int splits,
    const float* xf) {
    const int e = blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int s1 = group_off[e + 1];
    const int m = s1 - s0;
    if (m <= 0 || slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp >= n) {
        return;
    }
    const int row = warp;
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const int chunks = (k + 255) >> 8;
    const int c_lo = (int)(((long long)chunks * blockIdx.z) / splits);
    const int c_hi = (int)(((long long)chunks * (blockIdx.z + 1)) / splits);

    constexpr int MB = 8;
    for (int sb = 0; sb < m; sb += MB) {
        const int mb = (m - sb) < MB ? (m - sb) : MB;
        const bf16* xbp = x + (size_t)(s0 + sb) * (size_t)a_stride;
        float acc[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            acc[s] = 0.0f;
        }
        for (int ci = c_lo; ci < c_hi; ++ci) {
            const int c0 = ci * 256 + lane * 8;
            if (c0 + 8 <= k) {
                const int g = c0 >> 6;
                const int t = c0 & 63;
                const std::uint32_t wc = *reinterpret_cast<const std::uint32_t*>(
                    payload + (long long)row * row_bytes + g * 32 + (t >> 1));
                const float scc = __half2float(scales[(long long)row * row_groups + g]);
                float code[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    const int nb = (int)((wc >> (4 * z)) & 0xFu);
                    code[z] = (float)(nb >= 8 ? nb - 16 : nb);
                }
#pragma unroll
                for (int s = 0; s < MB; ++s) {
                    if (s >= mb) {
                        break;
                    }
                    float d = 0.0f;
                    if (xf != nullptr) {
                        const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                        const float4 x0 = *reinterpret_cast<const float4*>(xf + (size_t)xrow * (size_t)k + c0);
                        const float4 x1 = *reinterpret_cast<const float4*>(xf + (size_t)xrow * (size_t)k + c0 + 4);
                        d += code[0] * x0.x;
                        d += code[1] * x0.y;
                        d += code[2] * x0.z;
                        d += code[3] * x0.w;
                        d += code[4] * x1.x;
                        d += code[5] * x1.y;
                        d += code[6] * x1.z;
                        d += code[7] * x1.w;
                    } else {
                        const int4 xv = *reinterpret_cast<const int4*>(
                            xbp + (size_t)s * (size_t)a_stride + c0);
                        const bf16* xp = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
                        for (int z = 0; z < 8; ++z) {
                            d += code[z] * __bfloat162float(xp[z]);
                        }
                    }
                    acc[s] += scc * d;
                }
            } else {
                for (int i = 0; i < 8 && c0 + i < k; ++i) {
                    const int c = c0 + i;
                    const int g = c >> 6;
                    const int t = c & 63;
                    const std::uint8_t byte = payload[(long long)row * row_bytes + g * 32 + (t >> 1)];
                    const int nb = (t & 1) ? (byte >> 4) : (byte & 0x0F);
                    const int code = nb >= 8 ? nb - 16 : nb;
                    const float sc = __half2float(scales[(long long)row * row_groups + g]);
#pragma unroll
                    for (int s = 0; s < MB; ++s) {
                        if (s >= mb) {
                            break;
                        }
                        if (xf != nullptr) {
                            const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                            acc[s] += (float)code * sc * xf[(size_t)xrow * (size_t)k + c];
                        } else {
                            acc[s] += (float)code * sc
                                      * __bfloat162float(xbp[(size_t)s * (size_t)a_stride + c]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            float a2 = acc[s];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a2 += __shfl_down_sync(0xffffffffu, a2, off);
            }
            if (lane == 0) {
                const long long slot = ((long long)(s0 + sb + s) * n + row) * splits + blockIdx.z;
                part[slot] = a2;
            }
        }
    }
}

// S 段 fp32 部分和 → bf16 输出（按 ks=0..S-1 顺序累加）。
extern "C" __global__ void lingxi_gemv_splitk_reduce_bf16(const float* __restrict__ part,
                                                          bf16* __restrict__ y,
                                                          long long cnt,
                                                          int splits) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < cnt;
         i += (long long)gridDim.x * blockDim.x) {
        float a = 0.0f;
        for (int ks = 0; ks < splits; ++ks) {
            a += part[i * splits + ks];
        }
        y[i] = __float2bfloat16(a);
    }
}

// ───────────────── 2) 与 slotsm 同形状的"只读不算"探针 ─────────────────
// mode 0：和 `slotsm_body` 完全相同的 4 B/lane（warp 覆盖 128 B 连续），只做整数累加；
// mode 1：16 B/lane（warp 覆盖 512 B 连续）—— 量"每条指令搬更多字节"是否有用；
// mode 2：4 B/lane + **完整 nibble 抽取**（8 个 nibble 的整数和）⇒ 加进"解码 ALU"的代价；
// mode 3：4 B/lane + nibble 抽取 + **I2F 的符号扩展码**（float 累加，×1.0）⇒ 再量 CVT。
// ⇒ mode0/1 = 纯读上限，mode2−mode0 = 抽取指令代价，mode3−mode2 = 转换指令代价，
// 真核 − mode3 = x 读 + FFMA 的代价。
extern "C" __global__ void lingxi_read_probe_slots(const std::uint8_t* __restrict__ slots,
                                                   float* __restrict__ out,
                                                   int rows_per_expert,
                                                   int row_bytes,
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
    if (mode == 1) {
        // 32 lane × 16 B = 512 B / 指令
        for (int off = lane * 16; off + 16 <= row_bytes; off += 512) {
            const uint4 v = *reinterpret_cast<const uint4*>(row + off);
            iacc += (int)(v.x & 0xFFFFu) + (int)(v.y & 0xFFFFu) + (int)(v.z & 0xFFFFu)
                    + (int)(v.w & 0xFFFFu);
        }
    } else {
        // 32 lane × 4 B = 128 B / 指令（与生产核的地址映射逐字节相同）
        for (int off = 0; off + 128 <= row_bytes; off += 128) {
            const std::uint32_t w =
                *reinterpret_cast<const std::uint32_t*>(row + off + ((lane >> 3) * 32 + (lane & 7) * 4));
            if (mode == 0) {
                iacc += (int)(w & 0xFFFFu) + (int)(w >> 16);
            } else if (mode == 2) {
                // 8 个 nibble 的整数和（只量抽取，不量转换）
                int t = 0;
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    const int nb = (int)((w >> (4 * z)) & 0xFu);
                    t += (nb >= 8 ? nb - 16 : nb);
                }
                iacc += t;
            } else {
                // 抽取 + 符号扩展 + I2F（与真核的 code[] 计算逐条同形）
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    const int nb = (int)((w >> (4 * z)) & 0xFu);
                    acc += (float)(nb >= 8 ? nb - 16 : nb);
                }
            }
        }
    }
    // 归约 + 写出（防止 DCE）
    acc += (float)iacc;
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)e * rows_per_expert + warp] = acc;
    }
}

// ───────────────── 3) 经典 grid-stride 纯读（机器读带宽上限）─────────────────
extern "C" __global__ void lingxi_read_probe_flat(const uint4* __restrict__ buf,
                                                  float* __restrict__ out,
                                                  long long n16) {
    float acc = 0.0f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16;
         i += (long long)gridDim.x * blockDim.x) {
        const uint4 v = buf[i];
        acc += (float)(v.x & 0xFFFFu) + (float)(v.y & 0xFFFFu) + (float)(v.z & 0xFFFFu)
               + (float)(v.w & 0xFFFFu);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if ((threadIdx.x & 31) == 0) {
        out[(long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = acc;
    }
}
