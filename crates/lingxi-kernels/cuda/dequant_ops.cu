// L1 权重解码：`row-split-k128-v1` + `Q4G64_F16S`（对称分组量化）→ bf16。
//
// 物理布局（照搬 ninfer `tools/artifact/layouts.py`，见 ARCHITECTURE）：
// k_pad = align_up(k, 128)；groups = k_pad / 64
// base 平面 @ 0，每行 k_pad/2 字节；每字节低 nibble = 组内偶元素、高 nibble = 奇元素（int8 二补码）
// scale 平面 @ align_up(n * k_pad/2, 256)，每行 groups 个 fp16（每 64 个元素一个 scale）
// value = code_signed * scale

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

extern "C" __global__ void lingxi_dequant_rowsplit_q4g64_bf16(const std::uint8_t* __restrict__ payload,
                                                              bf16* __restrict__ out,
                                                              int n,
                                                              int k,
                                                              int k_pad,
                                                              long long scale_offset) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * k) {
        return;
    }
    const int r = (int)(idx / k);
    const int c = (int)(idx % k);
    const int g = c >> 6;          // 每组 64
    const int e = c & 63;
    const std::uint8_t byte = payload[(long long)r * (k_pad / 2) + g * 32 + (e >> 1)];
    const int nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
    const int code = nib >= 8 ? nib - 16 : nib;   // 4bit 二补码
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_offset);
    const float s = __half2float(scales[(long long)r * (k_pad / 64) + g]);
    out[idx] = __float2bfloat16((float)code * s);
}

// **融合反量化的 Q4G64 GEMV**：直接从设备槽读 Q4 权重，不落 bf16 中转。
// `y[e, row] = Σ_c dequant(w_q4[e, row, c]) * x[e * a_stride + c]`
// 一个 warp 管一行；lane 沿列分片 → 同组 32 列落在同一 32B 块内，访存合并。
// 动机（实测）：原「批量解量化成 bf16（98MB/层）× 再 batched GEMM」路径里，解量化 kernel
// 每层写 65MB+33MB bf16、且是 1 元素/线程的低效形态，实测占 MoE 专家段的大头。
// **逐行体（模板：RPW = 每 warp 处理的行数，必须是编译期常量）**。
// ⚠️ 绝不能把 RPW 做成运行时参数：那样外层循环无法被编译器消除，实测会让
// dn 从 31 µs 退化到 41.6 µs（-25%）、gu 从 46 µs 退化到 56 µs（对照 slotsm
// 同形状 41.6 完全一致，说明是代码生成退化而非时钟）。RPW=1 展开后与原
// 「一 warp 一行」核**逐位等价、性能等价**（`row == warp`，循环被完全消除）。
template <int RPW>
__device__ __forceinline__ void lingxi_gemv_q4g64_slots_body(
        const std::uint8_t* __restrict__ payload,
        const __half* __restrict__ scales,
        const bf16* __restrict__ xr,
        bf16* __restrict__ y,
        int e, int n, int k, int row_bytes, int row_groups,
        int lane, int warp, int total_warps, bool vec_ok) {
    // ══ nibble → float 查表（放 shared）══════════════════════════════════════════
    // SASS 实测：每解码一列（4 bit）要 **7 条指令**——
    // SHF(取位) + LOP3(掩码) + ISETP(判负) + @P LOP3(符号扩展) + I2F + PRMT(x:bf16→f32) + FFMA
    // 其中只有最后一条 FFMA 是真正在算。V100 每个 sub-partition 只有 **16 条 FP32 通道**，
    // 一条 32 线程的 warp 指令要占 **2 个 cycle** ⇒ 这个核卡在**发射率**上。
    // 「字节(含 2 个 nibble)当索引查 shared」把解码降到 ~1.5 条/列。
    // ⚠️ 已实测：**换成纯 ALU 的符号扩展（零 shared）反而更慢**（gu 439→420、dn 316→306 GB/s）——
    // 单行核每迭代只有 4 次 `LDS.64`，shared 流量根本不是瓶颈；查表省下的 2 条指令更值钱。
    // （4 行版才会被 shared 挤爆，见。）
    // ⚠️ **位安全**：表里存的就是 `(float)code`（与 I2F 对 [-8,7] 逐位相同），且累加仍是
    // `dot = fma(code, x, dot)` 的**同一顺序**、不做任何重结合 ⇒ 逐位一致。
    __shared__ float2 c_lut[256];
    {
        const int b = (int)threadIdx.x;
        const int lo = (b & 0xF) >= 8 ? (b & 0xF) - 16 : (b & 0xF);
        const int hi = (b >> 4) >= 8 ? (b >> 4) - 16 : (b >> 4);
        c_lut[b] = make_float2((float)lo, (float)hi);
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = warp + r * total_warps;
        if (row >= n) {
            break;
        }
        float acc = 0.0f;
    // ⚠️ **一次 32 位 load 取 8 个 nibble**（与稠密版 `lingxi_gemv_q4g64_dense_epi` 同一手法）。
    // 原实现是「每 lane 一次只取 1 字节（两个 lane 共用一个字节）」：
    // k=2560 时每 lane 要走 80 次**依赖**的 1 字节 load，实测只有 **197 GB/s**
    // （专家段 gu+down 合计 137 µs/层 × 48 层 = **6.58 ms/token**，是 decode 单块最大的内核开销）。
    // 改成 `c0 = lane*8` 后：lane 之间是**连续 4 字节**（整 warp 覆盖 256 列 = 一个 group 的 32 字节，
    // 完美合并），每 lane 的权重 load 次数降到 1/8。与稠密版一致。
    // 对齐自检：向量段要求 payload 起始 4 字节对齐（slots 基址 256 对齐、row_bytes 是 64 的倍数，
    // 所以只需 `blob`/`seg_off` 是 4 的倍数）。不满足就退回逐元素，绝不做非对齐 u32 load。
    // （`vec_ok` 由调用方算好传入；x 侧还要 16 字节对齐 —— 一次读 8 个 bf16。）
    if (vec_ok) {
        // **软件流水（㊸：预取深度 1→4）**：EVPROF 实测 decode 专家段占 token 的 67%、
        // 命中层也只有 ~200 GB/s（V100 900 GB/s 的 22%）——根因是全局在途字节数不够：
        // 打满 900 GB/s @ ~600 cyc 延迟需要 ~540 KB 在途，而深度 1 × (gu 1280+dn 640) warps
        // 只有 ~130 KB。滚动窗口一次性挂 4 轮 load，权重在途 ×4。
        // ⚠️ 位安全：lane 的列集合、每列所属 group、`acc += sc*dot` 的次序全部不变，
        // 只是 load 的**发射时刻**提前 ⇒ 逐位一致（同上一版深度 1 的论证）。
        const std::uint8_t* wp = payload + (long long)row * row_bytes;
        const __half* sp = scales + (long long)row * row_groups;
        constexpr int PF = 4;
        std::uint32_t wbuf[PF];
        float sbuf[PF];
        int4 xbuf[PF];
        int c0 = lane * 8;
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cp = c0 + p * 256;
            if (cp + 8 <= k) {
                wbuf[p] = *reinterpret_cast<const std::uint32_t*>(wp + ((cp >> 6) << 5) + ((cp & 63) >> 1));
                sbuf[p] = __half2float(sp[cp >> 6]);
                xbuf[p] = *reinterpret_cast<const int4*>(xr + cp);
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
                wbuf[PF - 1] = *reinterpret_cast<const std::uint32_t*>(wp + ((cn >> 6) << 5) + ((cn & 63) >> 1));
                sbuf[PF - 1] = __half2float(sp[cn >> 6]);
                xbuf[PF - 1] = *reinterpret_cast<const int4*>(xr + cn);
            }
            const bf16* xp = reinterpret_cast<const bf16*>(&xc);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // 一次取整个字节（含 2 个 nibble）当索引 → 一次 LDS.64 拿回两个 float。
                const int b8 = (int)((wc >> (8 * i)) & 0xFFu);
                const float2 c2 = c_lut[b8];
                dot = fmaf(c2.x, __bfloat162float(xp[2 * i]), dot);
                dot = fmaf(c2.y, __bfloat162float(xp[2 * i + 1]), dot);
            }
            acc += scc * dot;
        }
        // 尾部（k 不是 8 的倍数时）：退回逐元素，列下标照旧
        for (; c0 < k; c0 += 256) {
            for (int i = 0; i < 8 && c0 + i < k; ++i) {
                const int c = c0 + i;
                const int g = c >> 6;
                const int t = c & 63;
                const std::uint8_t byte = payload[(long long)row * row_bytes + g * 32 + (t >> 1)];
                const int nib = (t & 1) ? (byte >> 4) : (byte & 0x0F);
                const int code = nib >= 8 ? nib - 16 : nib;
                const float sc = __half2float(scales[(long long)row * row_groups + g]);
                acc += (float)code * sc * __bfloat162float(xr[c]);
            }
        }
    } else {
        for (int c = lane; c < k; c += 32) {
            const int g = c >> 6;
            const int t = c & 63;
            const std::uint8_t byte = payload[(long long)row * row_bytes + g * 32 + (t >> 1)];
            const int nib = (t & 1) ? (byte >> 4) : (byte & 0x0F);
            const int code = nib >= 8 ? nib - 16 : nib;
            const float sc = __half2float(scales[(long long)row * row_groups + g]);
            acc += (float)code * sc * __bfloat162float(xr[c]);
        }
    }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) {
            y[(long long)e * n + row] = __float2bfloat16(acc);
        }
    }
}

/// 一 warp 一行（decode t=1 的黄金路径；与历史版本逐位一致）。
extern "C" __global__ void lingxi_gemv_q4g64_slots_bf16(const std::uint8_t* __restrict__ slots,
                                                         const int* __restrict__ slot_ids,
                                                         const bf16* __restrict__ x,
                                                         bf16* __restrict__ y,
                                                         int S,
                                                         int n,
                                                         int k,
                                                         int k_pad,
                                                         int a_stride,
                                                         long long blob,
                                                         long long seg_off,
                                                         long long scale_off) {
    const int e = blockIdx.y;
    if (e >= S) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const bf16* xr = x + (long long)e * a_stride;
    const bool vec_ok = ((blob & 3) == 0) && ((seg_off & 3) == 0) && ((a_stride & 7) == 0);
    lingxi_gemv_q4g64_slots_body<1>(payload, scales, xr, y, e, n, k, k_pad >> 1, k_pad >> 6,
                                    lane, warp, total_warps, vec_ok);
}

/// 多行/warp 版：**每个 RPW 一个独立核符号**。
/// ⚠️ 不能把多个 RPW 塞进同一个核用 `switch` 分派：那样寄存器/占用率会按最大的
/// RPW 分配，实测让 RPW=1 路径从 31.5 µs 退化到 41.5 µs（-24%）。独立核各自最优。
#define LINGXI_DEF_SLOTS_RPW(NAME, RPW)                                                       \
    extern "C" __global__ void NAME(const std::uint8_t* __restrict__ slots,                    \
                                    const int* __restrict__ slot_ids,                          \
                                    const bf16* __restrict__ x,                                \
                                    bf16* __restrict__ y,                                      \
                                    int S, int n, int k, int k_pad, int a_stride,              \
                                    long long blob, long long seg_off, long long scale_off) {  \
        const int e = blockIdx.y;                                                              \
        if (e >= S) {                                                                          \
            return;                                                                            \
        }                                                                                      \
        const int lane = (int)(threadIdx.x & 31);                                              \
        const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);                  \
        const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);                          \
        const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;         \
        const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);           \
        const bf16* xr = x + (long long)e * a_stride;                                          \
        const bool vec_ok = ((blob & 3) == 0) && ((seg_off & 3) == 0) && ((a_stride & 7) == 0);\
        lingxi_gemv_q4g64_slots_body<RPW>(payload, scales, xr, y, e, n, k, k_pad >> 1,         \
                                          k_pad >> 6, lane, warp, total_warps, vec_ok);        \
    }

LINGXI_DEF_SLOTS_RPW(lingxi_gemv_q4g64_slots_r2_bf16, 2)
LINGXI_DEF_SLOTS_RPW(lingxi_gemv_q4g64_slots_r4_bf16, 4)
LINGXI_DEF_SLOTS_RPW(lingxi_gemv_q4g64_slots_r8_bf16, 8)

// **稠密权重**的 Q4G64 融合反量化 GEMV（无槽间接）：`y[n] = W_q4[n,k] · x[k]`。
// 与专家槽版同一套解码契约，只是权重直接来自一整块稠密 payload（base + scale 两平面）。
// 动机：decode 的稠密投影在全 BF16 下是 7.97 GiB/token 的 HBM 流量，实测这些大形状
// 全都在 ~700 GB/s 的带宽墙上（launch-bench 分形状表）→ 把字节砍到 1/3.77 直接省时间。
// 尾融合语义与 `lingxi_gemv_bf16_epi` 一致（0 原样 / 1 silu(acc*a) / 2 sigmoid(acc*a)*b / 3 acc*a+b）。
extern "C" __global__ void lingxi_gemv_q4g64_dense_epi(const std::uint8_t* __restrict__ w,
                                                       const bf16* __restrict__ x,
                                                       bf16* __restrict__ y,
                                                       int n,
                                                       int k,
                                                       int k_pad,
                                                       int row_off,
                                                       long long scale_off,
                                                       int epi,
                                                       float a,
                                                       float b) {
    // nibble→float 查表（同 `lingxi_gemv_q4g64_slots_body` 的长注释）：把「SHF+LOP3+ISETP+@P LOP3+I2F」
    // 5 条解码指令换成 1 条 `LDS.64`，每行列的指令数从 7 降到 ~3.5 ⇒ 从发射率瓶颈里解出来。
    // ⚠️ 必须放在 `warp >= n` 早退**之前**——那是块内**非一致**退出，其后不能有 `__syncthreads`。
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
    const __half* scales = reinterpret_cast<const __half*>(w + scale_off) + (long long)row * row_groups;
    float acc = 0.0f;
    // **每次处理 8 列**：这 8 列的 nibble 恰好是 4 个连续字节（group 内 t 是 8 的倍数 →
    // `g*32 + (t>>1)` 4 字节对齐），一次 32 位 load 拿到全部 8 个 nibble。
    // 这样 lane 之间是连续 4 字节、整 warp 覆盖一个 group 的 32 字节 → 完美合并访存；
    // 原实现每 lane 只取 1 字节（两个 lane 共用一个字节），实测有效带宽只有 bf16 GEMV 的一半。
    const int g0b = k_pad >> 1;
    int c0 = lane * 8;
    // ⚠️ **实测否决**（2026-10-16）：这里也试过「预取下一迭代的 word / x / scale」
    // （与单行专家核、`slotsm` 同一手法），**六个形状全平或更慢**：
    // `[16384,2560]` 523→513、`[2560,6144]` 366→365、`[13312,2560]` 510→503、
    // `[ 320,10240]` 123→110、`[10240, 320]` 234→216、`[2560,640]` 165→160 GB/s。
    // 差别在于**稠密的 k 大**（2560~10240，主段有 10~40 次迭代），循环自身就能把
    // HBM 延迟盖住、且已有足量 warp；预取只白占寄存器、压占用率。**别再往这个核里加预取。**
    // 主段：权重 4 字节一次、x 16 字节一次（指针不越界，绝不读 k 之后）
    for (; c0 + 8 <= k; c0 += 256) {
        const int g = c0 >> 6;
        const int t = c0 & 63;
        const std::uint32_t word =
            *reinterpret_cast<const std::uint32_t*>(payload + g * 32 + (t >> 1));
        const int4 xv = *reinterpret_cast<const int4*>(x + c0);
        const bf16* xp = reinterpret_cast<const bf16*>(&xv);
        // 注：试过拆 4 个累加器打散 FMA 依赖链，实测无收益（[2560,6144] 314→317 GB/s），
        // 且会改求和顺序 → 回退成单累加器。
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
    // 尾段（k 不是 8 的倍数）：同一个 chunk 的 4 字节 word 只取一次，按 i 抽 nibble，
    // 累加进 dot 后再统一缩放（与主段同一套舍入顺序）
    for (; c0 < k; c0 += 256) {
        const int g = c0 >> 6;
        const int t0 = c0 & 63;
        const int bi = g * 32 + (t0 >> 1);
        std::uint8_t tmp[4] = {0, 0, 0, 0};
        const int lim = g0b - bi;
        for (int z = 0; z < 4 && z < lim; ++z) {
            tmp[z] = payload[bi + z];
        }
        const std::uint32_t word = (std::uint32_t)tmp[0] | ((std::uint32_t)tmp[1] << 8)
                                 | ((std::uint32_t)tmp[2] << 16) | ((std::uint32_t)tmp[3] << 24);
        float dot = 0.0f;
        for (int i = 0; i < 8 && c0 + i < k; ++i) {
            const int nib = (int)((word >> (4 * i)) & 0xFu);
            const int code = nib >= 8 ? nib - 16 : nib;
            dot += (float)code * __bfloat162float(x[c0 + i]);
        }
        acc += __half2float(scales[g]) * dot;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane != 0) {
        return;
    }
    float v = acc;
    if (epi == 1) {
        const float t = v * a;
        v = t / (1.0f + __expf(-t));
    } else if (epi == 2) {
        const float t = v * a;
        v = (1.0f / (1.0f + __expf(-t))) * b;
    } else if (epi == 3) {
        v = v * a + b;
    }
    y[warp] = __float2bfloat16(v);
}

// **按"唯一专家"分组的专家槽 Q4 GEMV**（批量路径用；`t=1` 时退化成上一个核）。
//
// ⚠️ **上一版（"对"路径）不做专家去重**：`t` 个 token × top_k 个专家 = `t·top_k` 个"对"，
// 同一个专家的**整块权重被读 `t` 遍**。t=8 时专家段流量因此是 t=1 的 ~3.3 倍（本应 ~1.2 倍），
// 实测 MoE 在 t=8 要 60 ms（占整趟前向的 38%）。
//
// 改法：`blockIdx.y` 变成**唯一专家下标**，块内循环该专家的 `m` 个 token（`group_off[e]..group_off[e+1]`），
// 权重行只 load 一次、被 m 个 x 行共用。**数学完全不变**（同一批 (token, 专家) 乘加，只是循环次序）。
// `a_stride` 是 x 的行步长：rows==1 时传 0（所有 pair 共用第 0 行），批量时传 hidden。
// ★ 2026-10-16：把 `slotsm` 抽成 **RPW（rows-per-warp）模板**，并加 R=2 变体。
//
// 为什么（实测依据）：`dn` 段 `k=640` 时每 warp 只有 ~2.5 次迭代，HBM ~600 cyc 的访存延迟
// 盖不住；而 **t=1 路径的 `slots` 早就靠 `rpw=2/4/8` 多行版解决同一问题**
// （见上面 `LINGXI_DEF_SLOTS_RPW` 的注释："dn 段 k=640 每行只有 320 B 的 Q4，逐行版会把全局
// warp 数撑到 n×S，线程太多、每行太少；多行版让一个 warp 顺序扫多行，减少线程数并把跨行访存
// 并行起来"）。**但 t>1 的 `slotsm` 一直是一格一行。**
// 我们实测专家段合计只有 **~270 GB/s**（顺序路径 790）⇒ 这是 25% 的整轮时间所在。
//
// ⚠️ **位安全**：每个输出行的 lane 列集合（`lane*8 + 256*i`）、每列所属 group、
// `acc[s] += scc*dot` 的次序、`code[z]` 的抽取次序**全部不变**，只是同一 warp 里多扫了几行。
// ⚠️ **不能用一个核 + switch 分派**（与 `slots` 同因）：寄存器/占用率会按最大 RPW 分配。
//
// ★ 2026-10-24 组压缩（`gidx`）：`gidx != nullptr` 时 `blockIdx.y` 是**本格式组列表的下标**，
// 真实组下标 = `gidx[e]`，边界与槽号都从全表按它取（`group_off[gi]`/`group_off[gi+1]`/
// `slot_ids[gi]`）⇒ 每个真实组读到的三元组与全 `u` 网格**逐项相同**，只是不再给对侧格式的
// 组发空块。⚠️ 不能改成"传紧凑 group_off"：两种格式的组在 pair 表里交错，紧凑列表中相邻
// 两项不是本组的起点/终点。
template <int RPW>
__device__ __forceinline__ void lingxi_gemv_q4g64_slotsm_body(const std::uint8_t* __restrict__ slots,
                                                              const int* __restrict__ slot_ids,
                                                              const bf16* __restrict__ x,
                                                              const int* __restrict__ group_off,
                                                              bf16* __restrict__ y,
                                                              int E,
                                                              int n,
                                                              int k,
                                                              int k_pad,
                                                              int a_stride,
                                                              long long blob,
                                                              long long seg_off,
                                                              long long scale_off,
                                                              const float* xf,
                                                              const int* __restrict__ gidx) {
    const int e = blockIdx.y;
    if (e >= E) {
        return;
    }
    const int gi = (gidx != nullptr) ? gidx[e] : e;
    const int s0 = group_off[gi];
    const int s1 = group_off[gi + 1];
    const int m = s1 - s0;
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);
    if (m <= 0) {
        return;
    }
    if (slot_ids[gi] < 0) {
        return; // 混档单 launch：他格式组的哨兵槽，跳过
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[gi] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const bf16* xb = x + (size_t)s0 * (size_t)a_stride;
    (void)xb;
    const bool vec_ok = ((blob & 3) == 0) && ((seg_off & 3) == 0) && ((a_stride & 7) == 0);
    // ★ 否决记录（P15-⑥，2026-10-17）：试过把 `slots_body` 的字节查表解码（c_lut[256]）
    // 搬到本核 —— **微基准全面大涨（m=1: 403→512、m=2: 422→488）但整机 fwd t=7 两次
    // 实测 −3%（79→82ms）**。根因是测量假象：bw-ceiling 的权重是 `alloc_zeros` 全零，
    // 查表索引恒定 ⇒ 全 lane 同地址广播无冲突；真实随机权重下 float2[256] 查表是
    // **2-way bank 冲突**（idx*2 mod 32），4 次 LDS.64 在关键路径上反而比 20 条
    // 独立 ALU 解码慢（ALU 各 lane 互不干扰，可双发射）。⇒ 保持纯 ALU 解码。
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
    // 行索引与 `slots_body` 同一约定（交错：`warp + r*total_warps`）
    const int row = warp + r * total_warps;
    if (row >= n) {
        break;
    }
    // ★★ 正确性修复（2026-10-18，`acc[8]` 只覆盖 8 个对）：**m > MB 的组必须分批跑**。
    // 旧版没有这层循环 ⇒ 每组只算前 8 个对，第 9 个及以后的 `y` 是**上一轮的脏值**
    // （实测：867 token 单 chunk、layer 47、232 组 8670 对，只有 **80 个对**真被写出；
    // `gu_all` 11,097,600 个元素里 10,995,200 个仍是 NaN 填充值）。
    // 这就是"同输入/同路由/同权重，不同卡配置首 token 不同"的总根因（脏值随分配历史变）。
    // 每个 pair 的累加次序（k 方向 + z=0..7）与旧版完全一致 ⇒ **m≤MB 时逐位不变**。
    // `MB` 取 8：这是**寄存器宽度**，直接决定占用率 —— 实测 t=7 验证轮里
    // `acc[16]` 比 `acc[8]` 的专家段层均 **0.64 → 0.59 ms（−8%）**（占用率 4 块→3 块），
    // 而 prefill 大 chunk 下 16 只比 8 快 1% ⇒ **取 8**（decode/验证是热路径）。
    // decode（rows=1）走的是另一支 `slots` 核，与本核无关。
    // ★ 2026-10-18 实测否决：MB 从 8 提到 16（组均 m≈22 ⇒ 少一批权重重读/解码）
    // 在真实 23.7K prefill 上 123.4 → 124.2s（无收益，寄存器占用率抵消了收益）⇒ 保持 8。
    constexpr int MB = 8;
    for (int sb = 0; sb < m; sb += MB) {
    const int mb = (m - sb) < MB ? (m - sb) : MB;
    const bf16* xbp = x + (size_t)(s0 + sb) * (size_t)a_stride;
    float acc[MB];
#pragma unroll
    for (int s = 0; s < MB; ++s) {
        acc[s] = 0.0f;
    }
    if (vec_ok) {
        // **软件流水（㊸：预取深度 1→4，同 slots_body 的论证）**：批量路径是大头那一趟，
        // x 侧每 (迭代, s) 的 load 保持原位（m≤8 行 × 16 B 已够宽），权重/scale 滚动预取 4 轮。
        // ⚠️ **位安全**：lane 的列集合、每列所属 group、`acc[s] += sc*d` 的次序、
        // `code[z]` 的抽取次序全部不变，只是 load 的**发射时刻**提前 ⇒ 与旧版逐位一致。
        // 预取深度扫描（2026-10-18）：PF=2/4/8 → m=2: 319.6/318.3/301.2、m=7: 262.8/255.7/227.1
        // ⇒ 深度不是杠杆（8 因寄存器掉占用率反而更差）；保持 4（历史验证最充分）。
        constexpr int PF = 4;
        std::uint32_t wbuf[PF];
        float sbuf[PF];
        int c0 = lane * 8;
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cp = c0 + p * 256;
            if (cp + 8 <= k) {
                const int g = cp >> 6;
                const int t = cp & 63;
                wbuf[p] = *reinterpret_cast<const std::uint32_t*>(
                    payload + (long long)row * row_bytes + g * 32 + (t >> 1));
                sbuf[p] = __half2float(scales[(long long)row * row_groups + g]);
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
                const int gn = cn >> 6;
                const int tn = cn & 63;
                wbuf[PF - 1] = *reinterpret_cast<const std::uint32_t*>(
                    payload + (long long)row * row_bytes + gn * 32 + (tn >> 1));
                sbuf[PF - 1] = __half2float(scales[(long long)row * row_groups + gn]);
            }
            // 权重行只 load 一次，且 **nibble 反量化也只做一次**（此前每个 token 重复抽 8 次
            // nibble，m=8 时 ALU 是 FMA 的 4 倍，把核压成了计算瓶颈）。每个 s 内 z 的
            // 乘加次序与旧版逐位一致：code 值本就是精确整数，d 仍是 z=0..7 串行累加。
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
                    // ★ P15-⑤b ①（2026-10-16 第四版）：host 已把 x **提前转成 fp32**
                    // （"所有后面计算过程要转化的提前转化"）。直接 2×LDG.128 float4
                    // （L1/L2 命中），省掉每 (word,token) 8 条 CVT。行序与 bf16 路径一致
                    // （a_stride==0 时所有对共用第 0 行），z 次序与累加次序逐位不变。
                    const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                    const float* xfp = xf + (size_t)xrow * (size_t)k + c0;
                    const float4 x0 = *reinterpret_cast<const float4*>(xfp);
                    const float4 x1 = *reinterpret_cast<const float4*>(xfp + 4);
                    d += code[0] * x0.x;
                    d += code[1] * x0.y;
                    d += code[2] * x0.z;
                    d += code[3] * x0.w;
                    d += code[4] * x1.x;
                    d += code[5] * x1.y;
                    d += code[6] * x1.z;
                    d += code[7] * x1.w;
                } else {
                    const int4 xv = *reinterpret_cast<const int4*>(xbp + (size_t)s * (size_t)a_stride + c0);
                    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                    d += code[0] * __bfloat162float(xp[0]);
                    d += code[1] * __bfloat162float(xp[1]);
                    d += code[2] * __bfloat162float(xp[2]);
                    d += code[3] * __bfloat162float(xp[3]);
                    d += code[4] * __bfloat162float(xp[4]);
                    d += code[5] * __bfloat162float(xp[5]);
                    d += code[6] * __bfloat162float(xp[6]);
                    d += code[7] * __bfloat162float(xp[7]);
                }
                acc[s] += scc * d;
            }
        }
        for (; c0 < k; c0 += 256) {
            for (int i = 0; i < 8 && c0 + i < k; ++i) {
                const int c = c0 + i;
                const int g = c >> 6;
                const int t = c & 63;
                const std::uint8_t byte = payload[(long long)row * row_bytes + g * 32 + (t >> 1)];
                const int nb = (t & 1) ? (byte >> 4) : (byte & 0x0F);
                const int code = nb >= 8 ? nb - 16 : nb;
                const float sc = __half2float(scales[(long long)row * row_groups + g]);
                // ⚠️ 必须 `#pragma unroll + break`（索引编译期）——`for (s < m)` 运行期
                // 下标会把 acc[8] 打进**局部内存**（ptxas "32 bytes stack frame"），
                // 连主循环的 acc 读写都走 local ⇒ 整机 −20% 量级（P15-⑦ 实测定案）。
#pragma unroll
                for (int s = 0; s < MB; ++s) {
                    if (s >= mb) {
                        break;
                    }
                    if (xf != nullptr) {
                        const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                        acc[s] += (float)code * sc * xf[(size_t)xrow * (size_t)k + c];
                    } else {
                        acc[s] += (float)code * sc * __bfloat162float(xbp[(size_t)s * (size_t)a_stride + c]);
                    }
                }
            }
        }
    } else {
        for (int c = lane; c < k; c += 32) {
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
                acc[s] += (float)code * sc * __bfloat162float(xbp[(size_t)s * (size_t)a_stride + c]);
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
            y[(size_t)(s0 + sb + s) * (size_t)n + row] = __float2bfloat16(a2);
        }
    }
    }  // for sb (pairs in batches of 8)
    }  // for r (RPW)
}

// bf16 → fp32 行转换（`slotsm` 的 `xf` 预转输入；P15-⑤b ①"提前转化"）。
// n%8==0 时向量化（int4 = 8 bf16）；bf16→f32 精确 ⇒ 下游 GEMV 逐位不变。
extern "C" __global__ void lingxi_convert_bf16_f32(const bf16* __restrict__ x,
                                                   float* __restrict__ y,
                                                   long long n) {
    if (((n & 7) == 0) && (((long long)x & 15) == 0) && (((long long)y & 15) == 0)) {
        const long long n8 = n >> 3;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n8;
             i += (long long)gridDim.x * blockDim.x) {
            const int4 xv = *reinterpret_cast<const int4*>(x + (i << 3));
            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
            float out[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                out[z] = __bfloat162float(xp[z]);
            }
            *reinterpret_cast<float4*>(y + (i << 3)) = *reinterpret_cast<float4*>(&out[0]);
            *reinterpret_cast<float4*>(y + (i << 3) + 4) = *reinterpret_cast<float4*>(&out[4]);
        }
    } else {
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += (long long)gridDim.x * blockDim.x) {
            y[i] = __bfloat162float(x[i]);
        }
    }
}

// RPW=1：与原 `slotsm` **逐位等价、行为完全不变**（`row == warp`，r 循环被完全展开消掉）。
extern "C" __global__ void lingxi_gemv_q4g64_slotsm_bf16(const std::uint8_t* __restrict__ slots,
                                                          const int* __restrict__ slot_ids,
                                                          const bf16* __restrict__ x,
                                                          const int* __restrict__ group_off,
                                                          bf16* __restrict__ y,
                                                          int E,
                                                          int n,
                                                          int k,
                                                          int k_pad,
                                                          int a_stride,
                                                          long long blob,
                                                           long long seg_off,
                                                           long long scale_off,
                                                           const float* xf,
                                                           const int* gidx) {
    lingxi_gemv_q4g64_slotsm_body<1>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride,
                                     blob, seg_off, scale_off, xf, gidx);
}

// RPW=2：一个 warp 顺序扫两行（行距 = total_warps，与 `slots_body` 同约定）。
extern "C" __global__ void lingxi_gemv_q4g64_slotsm_r2_bf16(const std::uint8_t* __restrict__ slots,
                                                             const int* __restrict__ slot_ids,
                                                             const bf16* __restrict__ x,
                                                             const int* __restrict__ group_off,
                                                             bf16* __restrict__ y,
                                                             int E,
                                                             int n,
                                                             int k,
                                                             int k_pad,
                                                             int a_stride,
                                                             long long blob,
                                                              long long seg_off,
                                                              long long scale_off,
                                                             const float* xf,
                                                             const int* gidx) {
    lingxi_gemv_q4g64_slotsm_body<2>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride,
                                     blob, seg_off, scale_off, xf, gidx);
}

// ★ 2026-10-24`LINGXI_MOE_GEMV_TIER=1` 的可选档：`slotsm_body<1>` 的**逐位等价**
// 改写，只动"取数结构"，不动任何一条算式：
// ① 每轮把两个 chunk（`c0`、`c0+256`）的权重与 scale **一次发出去**（两个独立 load）。
// 生产版的 PF=4 是**移位预取**（`wbuf[p]=wbuf[p+1]`）：读下一轮值要等上一轮移位串完，
// 还多占 8 个寄存器（实测 74 regs / 24 warp 占用）；
// ② nibble 解包改"移位对"：`(int)(w << (28-4z)) >> 28`，2 条指令/值（生产是
// 右移+与+符号选择，4~5 条）。**解出的 code 是同一个整数**，故不影响数值。
// `acc[s] += scc*d` 的 chunk 次序、每 chunk 内 z=0..7 的串行次序、lane 的列集合全部不变
// ⇒ 与 `lingxi_gemv_q4g64_slotsm_bf16`（RPW=1 默认档）**逐位相同**
//（`tests/t12_moe_gemv_struct_bw.rs` 的 check：0/51200 元素不同）。
// 实测（E40 gu n1280 k2560、单线程）：m=1 317→453 GB/s、m≈1.5 235→340、E7 m=1 269→374。
// 只覆盖 `xf == nullptr`（bf16 现转）——线上 rows=1 的混档 decode 正是这一支。
// 核体（2026-10-25 从 `..._t2_bf16` 里**逐字搬出**，供下面两个入口共用：
// ① `..._t2_bf16`：精确网格 `grid.y = E`（生产默认）；
// ② `..._t2_ng_bf16`：静态网格 + 设备侧组数上界。
// 搬移是纯文本移动，`gi` 由调用方解析好传进来 ⇒ 两条路径逐位相同。
__device__ __forceinline__ void lingxi_gemv_q4g64_slotsm_t2_body(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        int gi) {
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
        int c0 = lane * 8;
        // 两个 chunk 一组（k=2560 → 5 组；k=640 → 2 组 + 尾巴）
        for (; c0 + 264 <= k; c0 += 512) {
            std::uint32_t w0, w1;
            float s0v, s1v;
            {
                const int c = c0;
                w0 = *reinterpret_cast<const std::uint32_t*>(prow + (c >> 1));
                s0v = __half2float(srow[c >> 6]);
                const int c2 = c + 256;
                w1 = *reinterpret_cast<const std::uint32_t*>(prow + (c2 >> 1));
                s1v = __half2float(srow[c2 >> 6]);
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const std::uint32_t wc = (u == 0) ? w0 : w1;
                const float scc = (u == 0) ? s0v : s1v;
                const int cc = c0 + 256 * u;
                float code[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    code[z] = (float)((int)(wc << (28 - 4 * z)) >> 28);
                }
#pragma unroll
                for (int s = 0; s < MB; ++s) {
                    if (s >= mb) {
                        break;
                    }
                    const int4 xv =
                        *reinterpret_cast<const int4*>(xbp + (size_t)s * (size_t)a_stride + cc);
                    const bf16* xp2 = reinterpret_cast<const bf16*>(&xv);
                    float d = 0.0f;
                    d += code[0] * __bfloat162float(xp2[0]);
                    d += code[1] * __bfloat162float(xp2[1]);
                    d += code[2] * __bfloat162float(xp2[2]);
                    d += code[3] * __bfloat162float(xp2[3]);
                    d += code[4] * __bfloat162float(xp2[4]);
                    d += code[5] * __bfloat162float(xp2[5]);
                    d += code[6] * __bfloat162float(xp2[6]);
                    d += code[7] * __bfloat162float(xp2[7]);
                    acc[s] += scc * d;
                }
            }
        }
        for (; c0 + 8 <= k; c0 += 256) {
            const std::uint32_t wc = *reinterpret_cast<const std::uint32_t*>(prow + (c0 >> 1));
            const float scc = __half2float(srow[c0 >> 6]);
            float code[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                code[z] = (float)((int)(wc << (28 - 4 * z)) >> 28);
            }
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                if (s >= mb) {
                    break;
                }
                const int4 xv =
                    *reinterpret_cast<const int4*>(xbp + (size_t)s * (size_t)a_stride + c0);
                const bf16* xp2 = reinterpret_cast<const bf16*>(&xv);
                float d = 0.0f;
                d += code[0] * __bfloat162float(xp2[0]);
                d += code[1] * __bfloat162float(xp2[1]);
                d += code[2] * __bfloat162float(xp2[2]);
                d += code[3] * __bfloat162float(xp2[3]);
                d += code[4] * __bfloat162float(xp2[4]);
                d += code[5] * __bfloat162float(xp2[5]);
                d += code[6] * __bfloat162float(xp2[6]);
                d += code[7] * __bfloat162float(xp2[7]);
                acc[s] += scc * d;
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
                y[(size_t)(s0 + sb + s) * (size_t)n + warp] = __float2bfloat16(a2);
            }
        }
    }
}

// ① 生产入口（精确网格 `grid.y = E`）。
extern "C" __global__ void lingxi_gemv_q4g64_slotsm_t2_bf16(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const int* __restrict__ gidx) {
    const int e = blockIdx.y;
    if (e >= E) {
        return;
    }
    const int gi = (gidx != nullptr) ? gidx[e] : e;
    lingxi_gemv_q4g64_slotsm_t2_body(slots, slot_ids, x, group_off, y, n, k, k_pad, a_stride,
                                     blob, seg_off, scale_off, gi);
}

// ② **静态网格 + 设备侧组数上界**：与 ① 同一个 body、
// 逐位相同；只是允许 `grid.y = s`（上界）而不必先等设备写出 `npri/nalt`
// （那个等待就是每层 345~566 µs 的 `route_spin`）。
extern "C" __global__ void lingxi_gemv_q4g64_slotsm_t2_ng_bf16(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const int* __restrict__ gidx,
        const int* __restrict__ ng_dev) {
    const int e = blockIdx.y;
    if (ng_dev != nullptr && e >= *ng_dev) {
        return;
    }
    if (e >= E) {
        return;
    }
    const int gi = (gidx != nullptr) ? gidx[e] : e;
    lingxi_gemv_q4g64_slotsm_t2_body(slots, slot_ids, x, group_off, y, n, k, k_pad, a_stride,
                                     blob, seg_off, scale_off, gi);
}
extern "C" __global__ void lingxi_gemv_q4g64_slotsm_r4_bf16(const std::uint8_t* __restrict__ slots,
                                                             const int* __restrict__ slot_ids,
                                                             const bf16* __restrict__ x,
                                                             const int* __restrict__ group_off,
                                                             bf16* __restrict__ y,
                                                             int E,
                                                             int n,
                                                             int k,
                                                             int k_pad,
                                                             int a_stride,
                                                             long long blob,
                                                              long long seg_off,
                                                              long long scale_off,
                                                             const float* xf,
                                                             const int* gidx) {
    lingxi_gemv_q4g64_slotsm_body<4>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride,
                                     blob, seg_off, scale_off, xf, gidx);
}

// ═══════════════ ★ Q3G128F16S：Q4G64F16S 的平行 3-bit 变体（2026-10-19）═══════════════
//
// 布局（上游 layouts.py `_pack_three_bit` 同一契约，dense LSB-first 位流）：
// 元素 c 占**行内** bits [3c, 3c+3)（每 128 值 = 48 B，行间无填充）；
// code = w >= 4 ? w - 8: w（3-bit 二补码），value = code * scale（fp16，每 128 元素一个）。
// row_bytes = k_pad / 128 * 48；row_groups = k_pad / 128。
// 结构与 Q4 版**完全同构**：同一 RPW 模板 / 行交错索引 / MB=8 分批 / PF=4 软件流水 /
// xf fp32 快路 / vec_ok 守卫 / 尾部逐列。每 8 列 = 24 bit = 3 字节，行内字节偏移
// `3*(c0>>3)`（c0 是 8 的倍数 ⇒ 精确对齐）。
// ⚠️ 3-bit **不做字节查表**（查表地址空间 512 项，bank 冲突不划算）——纯 ALU 解码。

// 行内偏移 `byte_off` 处装载 3 字节、拼成 24 位位流（bits [3c, 3c+3) 的容器）。
__device__ __forceinline__ std::uint32_t lingxi_q3_load24(const std::uint8_t* __restrict__ p) {
    const std::uint32_t b0 = (std::uint32_t)__ldg(p);
    const std::uint32_t b1 = (std::uint32_t)__ldg(p + 1);
    const std::uint32_t b2 = (std::uint32_t)__ldg(p + 2);
    return b0 | (b1 << 8) | (b2 << 16);
}

// 24 位位流里第 z 个 3-bit code（z=0..7）→ float（code ∈ [-4,3] 精确整数）。
__device__ __forceinline__ float lingxi_q3_code8(std::uint32_t w24, int z) {
    const int w3 = (int)((w24 >> (3 * z)) & 7u);
    return (float)(w3 >= 4 ? w3 - 8 : w3);
}

// 逐列版：列 c 的 3-bit code（3c 可能跨字节 ⇒ 2 字节拼 u16 再取 [bit_off, bit_off+3)）。
__device__ __forceinline__ int lingxi_q3_code_at(const std::uint8_t* rowp, int c) {
    const int byte_off = (3 * c) >> 3;
    const int bit_off = (3 * c) & 7;
    const std::uint32_t w16 = (std::uint32_t)rowp[byte_off]
                            | ((std::uint32_t)rowp[byte_off + 1] << 8);
    const int w3 = (int)((w16 >> bit_off) & 7u);
    return w3 >= 4 ? w3 - 8 : w3;
}

// **批量解量化**：Q3G128 RowSplit → bf16（混档 6.2-2 逐专家回退路径用；与 Q4 版逐行体同构：
// 1 元素/线程，n×k 网格）。`payload` 指向段首，scale 平面在 `scale_offset`。
extern "C" __global__ void lingxi_dequant_rowsplit_q3g128_bf16(
        const std::uint8_t* __restrict__ payload,
        bf16* __restrict__ out,
        int n,
        int k,
        int k_pad,
        long long scale_offset) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * k) {
        return;
    }
    const int r = (int)(idx / k);
    const int c = (int)(idx % k);
    const int g = c >> 7;          // 每组 128
    const std::uint8_t* rowp = payload + (long long)r * ((k_pad / 128) * 48);
    const int code = lingxi_q3_code_at(rowp, c);
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_offset);
    const float s = __half2float(scales[(long long)r * (k_pad / 128) + g]);
    out[idx] = __float2bfloat16((float)code * s);
}

// **融合反量化的 Q3G128 GEMV 逐行体**（模板/行交错/流水与 `lingxi_gemv_q4g64_slots_body`
// 同构；Q4 的 c_lut 查表换成纯 ALU 3-bit 抽取，无 shared、无 __syncthreads）。
template <int RPW>
__device__ __forceinline__ void lingxi_gemv_q3g128_slots_body(
        const std::uint8_t* __restrict__ payload,
        const __half* __restrict__ scales,
        const bf16* __restrict__ xr,
        bf16* __restrict__ y,
        int e, int n, int k, int row_bytes, int row_groups,
        int lane, int warp, int total_warps, bool vec_ok) {
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = warp + r * total_warps;
        if (row >= n) {
            break;
        }
        float acc = 0.0f;
    if (vec_ok) {
        // 软件流水 PF=4（同 Q4 版）：lane 每轮 8 列 = 3 字节权重 + 16 字节 x + 1 个 scale，
        // 滚动预取。位安全论据与 Q4 版相同（只改 load 发射时刻，不改列集合/累加次序）。
        const std::uint8_t* wp = payload + (long long)row * row_bytes;
        const __half* sp = scales + (long long)row * row_groups;
        constexpr int PF = 4;
        std::uint32_t wbuf[PF];
        float sbuf[PF];
        int4 xbuf[PF];
        int c0 = lane * 8;
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cp = c0 + p * 256;
            if (cp + 8 <= k) {
                wbuf[p] = lingxi_q3_load24(wp + 3 * (cp >> 3));
                sbuf[p] = __half2float(sp[cp >> 7]);
                xbuf[p] = *reinterpret_cast<const int4*>(xr + cp);
            }
        }
        for (; c0 + 8 <= k; c0 += 256) {
            const std::uint32_t w24 = wbuf[0];
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
                wbuf[PF - 1] = lingxi_q3_load24(wp + 3 * (cn >> 3));
                sbuf[PF - 1] = __half2float(sp[cn >> 7]);
                xbuf[PF - 1] = *reinterpret_cast<const int4*>(xr + cn);
            }
            const bf16* xp = reinterpret_cast<const bf16*>(&xc);
            float dot = 0.0f;
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                // 3-bit 纯 ALU 解码（7 条指令/列，同 Q4 未查表版的量级）；
                // code 是精确整数 ⇒ z 串行 fma 次序与 Q4 版逐位同构。
                dot = fmaf(lingxi_q3_code8(w24, z), __bfloat162float(xp[z]), dot);
            }
            acc += scc * dot;
        }
        // 尾部（k 不是 8 的倍数）：逐列，列下标照旧
        for (; c0 < k; c0 += 256) {
            const std::uint8_t* rowp = payload + (long long)row * row_bytes;
            for (int i = 0; i < 8 && c0 + i < k; ++i) {
                const int c = c0 + i;
                const float code = (float)lingxi_q3_code_at(rowp, c);
                const float sc = __half2float(scales[(long long)row * row_groups + (c >> 7)]);
                acc += code * sc * __bfloat162float(xr[c]);
            }
        }
    } else {
        const std::uint8_t* rowp = payload + (long long)row * row_bytes;
        for (int c = lane; c < k; c += 32) {
            const float code = (float)lingxi_q3_code_at(rowp, c);
            const float sc = __half2float(scales[(long long)row * row_groups + (c >> 7)]);
            acc += code * sc * __bfloat162float(xr[c]);
        }
    }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) {
            y[(long long)e * n + row] = __float2bfloat16(acc);
        }
    }
}

/// 一 warp 一行（decode t=1 黄金路径；参数表与 `lingxi_gemv_q4g64_slots_bf16` 完全一致）。
extern "C" __global__ void lingxi_gemv_q3g128_slots_bf16(const std::uint8_t* __restrict__ slots,
                                                         const int* __restrict__ slot_ids,
                                                         const bf16* __restrict__ x,
                                                         bf16* __restrict__ y,
                                                         int S,
                                                         int n,
                                                         int k,
                                                         int k_pad,
                                                         int a_stride,
                                                         long long blob,
                                                         long long seg_off,
                                                         long long scale_off) {
    const int e = blockIdx.y;
    if (e >= S) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const bf16* xr = x + (long long)e * a_stride;
    const bool vec_ok = ((blob & 3) == 0) && ((seg_off & 3) == 0) && ((a_stride & 7) == 0);
    lingxi_gemv_q3g128_slots_body<1>(payload, scales, xr, y, e, n, k, (k_pad / 128) * 48,
                                     k_pad / 128, lane, warp, total_warps, vec_ok);
}

/// 多行/warp 版：**每个 RPW 一个独立核符号**（同 Q4 版，不能 switch 分派）。
#define LINGXI_DEF_SLOTS_Q3_RPW(NAME, RPW)                                                    \
    extern "C" __global__ void NAME(const std::uint8_t* __restrict__ slots,                    \
                                    const int* __restrict__ slot_ids,                          \
                                    const bf16* __restrict__ x,                                \
                                    bf16* __restrict__ y,                                      \
                                    int S, int n, int k, int k_pad, int a_stride,              \
                                    long long blob, long long seg_off, long long scale_off) {  \
        const int e = (int)blockIdx.y;                                                          \
        if (e >= S) {                                                                          \
            return;                                                                            \
        }                                                                                      \
        const int lane = (int)(threadIdx.x & 31);                                              \
        const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);                  \
        const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);                          \
        const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;         \
        const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);           \
        const bf16* xr = x + (long long)e * a_stride;                                          \
        const bool vec_ok = ((blob & 3) == 0) && ((seg_off & 3) == 0) && ((a_stride & 7) == 0);\
        lingxi_gemv_q3g128_slots_body<RPW>(payload, scales, xr, y, e, n, k, (k_pad / 128) * 48,\
                                           k_pad / 128, lane, warp, total_warps, vec_ok);      \
    }

LINGXI_DEF_SLOTS_Q3_RPW(lingxi_gemv_q3g128_slots_r2_bf16, 2)
LINGXI_DEF_SLOTS_Q3_RPW(lingxi_gemv_q3g128_slots_r4_bf16, 4)
LINGXI_DEF_SLOTS_Q3_RPW(lingxi_gemv_q3g128_slots_r8_bf16, 8)

// **按唯一专家分组**的 Q3G128 批量体（模板结构照抄 `lingxi_gemv_q4g64_slotsm_body`：
// MB=8 分批、PF=4 滚动预取、xf fp32 快路、acc[s] += scc*d 次序逐位同构；
// 权重装载 3×__ldg 拼 24 位位流，解码一次被 m 个 token 共用）。
template <int RPW>
__device__ __forceinline__ void lingxi_gemv_q3g128_slotsm_body(const std::uint8_t* __restrict__ slots,
                                                               const int* __restrict__ slot_ids,
                                                               const bf16* __restrict__ x,
                                                               const int* __restrict__ group_off,
                                                               bf16* __restrict__ y,
                                                               int E,
                                                               int n,
                                                               int k,
                                                               int k_pad,
                                                               int a_stride,
                                                               long long blob,
                                                               long long seg_off,
                                                               long long scale_off,
                                                               const float* xf,
                                                               const int* __restrict__ gidx) {
    const int e = blockIdx.y;
    if (e >= E) {
        return;
    }
    const int gi = (gidx != nullptr) ? gidx[e] : e;
    const int s0 = group_off[gi];
    const int s1 = group_off[gi + 1];
    const int m = s1 - s0;
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);
    if (m <= 0) {
        return;
    }
    if (slot_ids[gi] < 0) {
        return; // 混档单 launch：他格式组的哨兵槽，跳过
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[gi] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = (k_pad / 128) * 48;
    const int row_groups = k_pad / 128;
    const bf16* xb = x + (size_t)s0 * (size_t)a_stride;
    (void)xb;
    const bool vec_ok = ((blob & 3) == 0) && ((seg_off & 3) == 0) && ((a_stride & 7) == 0);
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
    const int row = warp + r * total_warps;
    if (row >= n) {
        break;
    }
    constexpr int MB = 8;
    for (int sb = 0; sb < m; sb += MB) {
    const int mb = (m - sb) < MB ? (m - sb) : MB;
    const bf16* xbp = x + (size_t)(s0 + sb) * (size_t)a_stride;
    float acc[MB];
#pragma unroll
    for (int s = 0; s < MB; ++s) {
        acc[s] = 0.0f;
    }
    if (vec_ok) {
        constexpr int PF = 4;
        std::uint32_t wbuf[PF];
        float sbuf[PF];
        int c0 = lane * 8;
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cp = c0 + p * 256;
            if (cp + 8 <= k) {
                wbuf[p] = lingxi_q3_load24(
                    payload + (long long)row * row_bytes + 3 * (cp >> 3));
                sbuf[p] = __half2float(scales[(long long)row * row_groups + (cp >> 7)]);
            }
        }
        for (; c0 + 8 <= k; c0 += 256) {
            const std::uint32_t w24 = wbuf[0];
            const float scc = sbuf[0];
#pragma unroll
            for (int p = 0; p + 1 < PF; ++p) {
                wbuf[p] = wbuf[p + 1];
                sbuf[p] = sbuf[p + 1];
            }
            const int cn = c0 + PF * 256;
            if (cn + 8 <= k) {
                wbuf[PF - 1] = lingxi_q3_load24(
                    payload + (long long)row * row_bytes + 3 * (cn >> 3));
                sbuf[PF - 1] = __half2float(scales[(long long)row * row_groups + (cn >> 7)]);
            }
            // 权重行只 load 一次、3-bit 解码也只一次，被 mb 个 token 共用；
            // 每个 s 内 z=0..7 串行累加次序与 Q4 版逐位同构。
            float code[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                code[z] = lingxi_q3_code8(w24, z);
            }
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                if (s >= mb) {
                    break;
                }
                float d = 0.0f;
                if (xf != nullptr) {
                    const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                    const float* xfp = xf + (size_t)xrow * (size_t)k + c0;
                    const float4 x0 = *reinterpret_cast<const float4*>(xfp);
                    const float4 x1 = *reinterpret_cast<const float4*>(xfp + 4);
                    d += code[0] * x0.x;
                    d += code[1] * x0.y;
                    d += code[2] * x0.z;
                    d += code[3] * x0.w;
                    d += code[4] * x1.x;
                    d += code[5] * x1.y;
                    d += code[6] * x1.z;
                    d += code[7] * x1.w;
                } else {
                    const int4 xv = *reinterpret_cast<const int4*>(xbp + (size_t)s * (size_t)a_stride + c0);
                    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                    d += code[0] * __bfloat162float(xp[0]);
                    d += code[1] * __bfloat162float(xp[1]);
                    d += code[2] * __bfloat162float(xp[2]);
                    d += code[3] * __bfloat162float(xp[3]);
                    d += code[4] * __bfloat162float(xp[4]);
                    d += code[5] * __bfloat162float(xp[5]);
                    d += code[6] * __bfloat162float(xp[6]);
                    d += code[7] * __bfloat162float(xp[7]);
                }
                acc[s] += scc * d;
            }
        }
        for (; c0 < k; c0 += 256) {
            const std::uint8_t* rowp = payload + (long long)row * row_bytes;
            for (int i = 0; i < 8 && c0 + i < k; ++i) {
                const int c = c0 + i;
                const float code = (float)lingxi_q3_code_at(rowp, c);
                const float sc = __half2float(scales[(long long)row * row_groups + (c >> 7)]);
#pragma unroll
                for (int s = 0; s < MB; ++s) {
                    if (s >= mb) {
                        break;
                    }
                    if (xf != nullptr) {
                        const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                        acc[s] += code * sc * xf[(size_t)xrow * (size_t)k + c];
                    } else {
                        acc[s] += code * sc * __bfloat162float(xbp[(size_t)s * (size_t)a_stride + c]);
                    }
                }
            }
        }
    } else {
        const std::uint8_t* rowp = payload + (long long)row * row_bytes;
        for (int c = lane; c < k; c += 32) {
            const float code = (float)lingxi_q3_code_at(rowp, c);
            const float sc = __half2float(scales[(long long)row * row_groups + (c >> 7)]);
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                if (s >= mb) {
                    break;
                }
                acc[s] += code * sc * __bfloat162float(xbp[(size_t)s * (size_t)a_stride + c]);
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
            y[(size_t)(s0 + sb + s) * (size_t)n + row] = __float2bfloat16(a2);
        }
    }
    }  // for sb (pairs in batches of 8)
    }  // for r (RPW)
}

// RPW=1/2/4 包装（参数表与 `lingxi_gemv_q4g64_slotsm_*_bf16` 完全一致）。
extern "C" __global__ void lingxi_gemv_q3g128_slotsm_bf16(const std::uint8_t* __restrict__ slots,
                                                          const int* __restrict__ slot_ids,
                                                          const bf16* __restrict__ x,
                                                          const int* __restrict__ group_off,
                                                          bf16* __restrict__ y,
                                                          int E,
                                                          int n,
                                                          int k,
                                                          int k_pad,
                                                          int a_stride,
                                                          long long blob,
                                                          long long seg_off,
                                                          long long scale_off,
                                                          const float* xf,
                                                          const int* gidx) {
    lingxi_gemv_q3g128_slotsm_body<1>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride,
                                      blob, seg_off, scale_off, xf, gidx);
}

extern "C" __global__ void lingxi_gemv_q3g128_slotsm_r2_bf16(const std::uint8_t* __restrict__ slots,
                                                             const int* __restrict__ slot_ids,
                                                             const bf16* __restrict__ x,
                                                             const int* __restrict__ group_off,
                                                             bf16* __restrict__ y,
                                                             int E,
                                                             int n,
                                                             int k,
                                                             int k_pad,
                                                             int a_stride,
                                                             long long blob,
                                                             long long seg_off,
                                                             long long scale_off,
                                                             const float* xf,
                                                             const int* gidx) {
    lingxi_gemv_q3g128_slotsm_body<2>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride,
                                      blob, seg_off, scale_off, xf, gidx);
}

extern "C" __global__ void lingxi_gemv_q3g128_slotsm_r4_bf16(const std::uint8_t* __restrict__ slots,
                                                             const int* __restrict__ slot_ids,
                                                             const bf16* __restrict__ x,
                                                             const int* __restrict__ group_off,
                                                             bf16* __restrict__ y,
                                                             int E,
                                                             int n,
                                                             int k,
                                                             int k_pad,
                                                             int a_stride,
                                                             long long blob,
                                                             long long seg_off,
                                                             long long scale_off,
                                                             const float* xf,
                                                             const int* gidx) {
    lingxi_gemv_q3g128_slotsm_body<4>(slots, slot_ids, x, group_off, y, E, n, k, k_pad, a_stride,
                                      blob, seg_off, scale_off, xf, gidx);
}

// ═══════════ ★ 2026-10-24Q3 的取数结构档（`LINGXI_NO_Q3_T2` 回退）═══════════
//
// Q4 的 `slotsm_t2_bf16` 已在线上白捡 58%（gu E40 m=1 286→453 GB/s）；Q3 有**两处**不同的贵：
// ① 每 8 列 3 次 1 字节 `__ldg`（`lingxi_q3_load24`）：3 条 LDG + 2 条 OR/SHL 才拼出 24 位；
// ⚠️ 这里**不能**换成一个未对齐 4 B load —— V100 直接 `CUDA_ERROR_MISALIGNED_ADDRESS`
// （t13 实测，纯读探针 mode 1 即崩）；3-bit 位流的行内偏移 `3*(c0>>3)` 天然 3 对齐。
// ⇒ 用**两个对齐 u32**（`4*(B>>2)`、+4）+ `__funnelshift_r` 拼窗口：2 条 LDG + 1 条 SHF。
// 多读的 1~5 B 落在同一段的 scale 平面里（布局 `[rows*row_bytes][align256][scales]`，
// 该平面恒非空）⇒ 不越出槽 blob。
// ② 解包 `lingxi_q3_code8`（右移+与+符号选择，3~4 条/值）⇒ 移位对
// `(int)(w24 << (29-3z)) >> 29`（2 条/值，ptxas 出 BFE）。解出的是**同一个整数**。
// 另加两条与算式无关的结构改动（都不改数值）：
// ③ 每轮把 UNR 个 chunk 的权重与 scale **一次全发出去**（独立 load），取代 PF=4 移位预取
// （`wbuf[p]=wbuf[p+1]` 是串行依赖链，且多占寄存器）；
// ④ `MB=1` 特化：`slotsm_body` 的 MB=8 骨架在 m=1 时仍要跑 7 次 `if (s >= mb) break`
// （7×2 条比较/分支 + 7 个空累加器）。线上 rows=1 的 decode **每组恰好 1 对**，
// 用 `MB=1` 实例即可（由调用方按 `tmax == 1` 选核，见 wrappers）。
//
// 位安全（逐位相同，t13 对拍 0/6400、0/19200、0/12800）：lane 的列集合
// （`lane*8 + 256*round`）、chunk 次序、每 chunk 内 z=0..7 的串行 fma 次序、
// `acc[s] += scc*d` 的次序全部不变；解的 code 是同一个整数 ⇒ 逐位相同。
//
// 实测（t13，单线程 `--test-threads=1`）：gu E10 m=1 161→303 GB/s（1.88×）、
// E3 m=1 164→251、dn E10 m=1 119→196；m≈1.5 档 137→197（MB=8 实例）。
// 只覆盖 `xf == nullptr`（bf16 现转）——线上 rows=1 的混档 decode 正是这一支。

// 行内字节偏移 `B`（3 的倍数）处的 24 位窗口：两个对齐 u32 + `__funnelshift_r`。
__device__ __forceinline__ std::uint32_t lingxi_q3_window32(const std::uint8_t* __restrict__ row,
                                                            int B) {
    const int w0 = B >> 2;
    const std::uint32_t a = *reinterpret_cast<const std::uint32_t*>(row + (w0 << 2));
    const std::uint32_t b = *reinterpret_cast<const std::uint32_t*>(row + (w0 << 2) + 4);
    return __funnelshift_r(a, b, (B & 3) * 8);
}

// 移位对解包：code ∈ [-4,3] 的精确整数（与 `lingxi_q3_code8` 同值）。
__device__ __forceinline__ float lingxi_q3_code8_s(std::uint32_t w24, int z) {
    return (float)((int)(w24 << (29 - 3 * z)) >> 29);
}

template <int UNR, int MB>
__device__ __forceinline__ void lingxi_gemv_q3g128_slotsm_t2_body(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const int* __restrict__ gidx) {
    const int e = blockIdx.y;
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
        // UNR 个 chunk 一组：权重与 scale 一次全发（独立 load，无移位链）
        for (; c0 + 256 * (UNR - 1) + 8 <= k; c0 += 256 * UNR) {
            std::uint32_t wv[UNR];
            float scv[UNR];
#pragma unroll
            for (int u = 0; u < UNR; ++u) {
                const int cc = c0 + 256 * u;
                wv[u] = lingxi_q3_window32(prow, 3 * (cc >> 3));
                scv[u] = __half2float(srow[cc >> 7]);
            }
#pragma unroll
            for (int u = 0; u < UNR; ++u) {
                float code[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    code[z] = lingxi_q3_code8_s(wv[u], z);
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
        // 尾部：单 chunk（chunk 次序与 `slotsm_body` 的 `c0 += 256` 一致）
        for (; c0 + 8 <= k; c0 += 256) {
            const float scc = __half2float(srow[c0 >> 7]);
            const std::uint32_t wc = lingxi_q3_window32(prow, 3 * (c0 >> 3));
            float code[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                code[z] = lingxi_q3_code8_s(wc, z);
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
            float a2 = acc[s];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a2 += __shfl_down_sync(0xffffffffu, a2, off);
            }
            if (lane == 0) {
                y[(size_t)(s0 + sb + s) * (size_t)n + warp] = __float2bfloat16(a2);
            }
        }
    }
}

// 通用档（每组 1~8 对，UNR=2）—— 与 `slotsm_bf16`（RPW=1）逐位相同。
extern "C" __global__ void lingxi_gemv_q3g128_slotsm_t2_bf16(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const int* __restrict__ gidx) {
    lingxi_gemv_q3g128_slotsm_t2_body<2, 8>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,
                                            a_stride, blob, seg_off, scale_off, gidx);
}

// ★ 2026-10-25 **设备侧组数上界档**：
// 与上面那个核**同一个 body、逐位相同**，唯一区别是入口多一道
// `e >= *ng_dev` 早退 —— 于是主机可以发**静态网格** `grid.y = s`（= rows×top_k 的上界，
// 60）而不再需要先等设备写出 `npri/nalt`（那个等待就是每层 345~566 µs 的 `route_spin`）。
// 为什么不能靠"把 gidx 补齐"：补最后一个合法索引会让**同一个组被多个 block 重复写 y**
// （数字直接错），补哨兵则 `group_off[-1]` 越界。
// 代价：多出来的空 block 只做一次 4 B 设备读 + 一次比较就退出（实测见 `t13_q3_static_grid`）。
extern "C" __global__ void lingxi_gemv_q3g128_slotsm_t2_ng_bf16(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const int* __restrict__ gidx,
        const int* __restrict__ ng_dev) {
    if (ng_dev != nullptr && (int)blockIdx.y >= *ng_dev) {
        return;
    }
    lingxi_gemv_q3g128_slotsm_t2_body<2, 8>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,
                                            a_stride, blob, seg_off, scale_off, gidx);
}

// ─────────── Q3G128 + **int8 激活 / `__dp4a`**（2026-10-25，非位等价）───────────
//
// ⛔ **实测负结果，未接生产**（，`t13_q3_dp4a_m5`）：
// gu E10 m=1 254.5→179.8（0.71×）、gu E40 m=5 88.8→81.9（0.92×）、dn E40 m=5 64.5→63.5（0.98×）。
// 原因是这些形状**延迟受限**（m=5 E40 的 t2 只有 88.8 GB/s，分母 255）——砍 ALU 换不来带宽，
// 而 3-bit→int8 的打包税（22 条/8 列）与 float 解包相当。核与用例保留，只作为判据留档。
//
// 原始动机（已证伪）：线上验证批 m≈5，t2 体每 token 要 8 条 FFMA + 8 条 CVT，
// 换成 int8 x 后每 token 只要 1 LDG.64 + 2 dp4a + 1 I2F + 1 FFMA。
//
// 量纲（若将来在别的形状复用，必须按 group 分段，不能跨组整数相加）：
// group 内 128 个权重共享一个 fp16 scale `scc`；x 每行一个 `ssx`（amax/127）
// ⇒ `d = Σ_group code·xq`（int32，|d| ≤ 4·127·128 = 65024，不溢出）
// `acc[s] += scc · (ssx · (float)d)`（与 Q4 的 `slotsm_sxq_dp4a` 同构）。
//
// ⚠️ **非位等价**（x 量化到 int8，误差 ~2^-7.5，比 Q3 权重自身的 2^-2.5 小 32×）。
// ⚠️ 只支持 `k_pad == k && k % 8 == 0`（本题 gu 2560 / dn 640 都满足；wrapper 里断言）。
__device__ __forceinline__ std::uint32_t lingxi_q3_pack4(std::uint32_t w24, int base) {
    const std::uint32_t c0 = (std::uint32_t)((int)(w24 << (29 - 3 * (base + 0))) >> 29);
    const std::uint32_t c1 = (std::uint32_t)((int)(w24 << (29 - 3 * (base + 1))) >> 29);
    const std::uint32_t c2 = (std::uint32_t)((int)(w24 << (29 - 3 * (base + 2))) >> 29);
    const std::uint32_t c3 = (std::uint32_t)((int)(w24 << (29 - 3 * (base + 3))) >> 29);
    const std::uint32_t p01 = __byte_perm(c0, c1, 0x0040);  // 低 2 字节 = [c0, c1]
    const std::uint32_t p23 = __byte_perm(c2, c3, 0x0040);
    return __byte_perm(p01, p23, 0x5410);                   // [c0, c1, c2, c3]
}

extern "C" __global__ void lingxi_gemv_q3g128_slotsm_dp4a(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const int* __restrict__ gidx,
        const signed char* __restrict__ xq,
        const float* __restrict__ sx) {
    const int e = (int)blockIdx.y;
    if (e >= E) {
        return;
    }
    const int gi = (gidx != nullptr) ? gidx[e] : e;
    const int s0 = group_off[gi];
    const int m = group_off[gi + 1] - s0;
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
    constexpr int MB = 8;
    for (int sb = 0; sb < m; sb += MB) {
        const int mb = (m - sb) < MB ? (m - sb) : MB;
        float ssx[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
            ssx[s] = sx[xrow];
        }
        float acc[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            acc[s] = 0.0f;
        }
        for (int c0 = lane * 8; c0 + 8 <= k; c0 += 256) {
            const std::uint32_t w24 = lingxi_q3_window32(prow, 3 * (c0 >> 3));
            const float scc = __half2float(srow[c0 >> 7]);
            const std::uint32_t a_lo = lingxi_q3_pack4(w24, 0);
            const std::uint32_t a_hi = lingxi_q3_pack4(w24, 4);
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                if (s >= mb) {
                    break;
                }
                const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                const uint2 xv = *reinterpret_cast<const uint2*>(
                    xq + (size_t)xrow * (size_t)k + c0);
                int d = __dp4a((int)a_lo, (int)xv.x, 0);
                d = __dp4a((int)a_hi, (int)xv.y, d);
                acc[s] += scc * (ssx[s] * (float)d);
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
                y[(size_t)(s0 + sb + s) * (size_t)n + warp] = __float2bfloat16(a2);
            }
        }
    }
}

// `MB=1` 特化（**只在每组恰好 1 对时可用**；UNR=2 给长 k，UNR=1 给短 k 的 dn 段）。
extern "C" __global__ void lingxi_gemv_q3g128_slotsm_t2_mb1u2_bf16(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const int* __restrict__ gidx) {
    lingxi_gemv_q3g128_slotsm_t2_body<2, 1>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,
                                            a_stride, blob, seg_off, scale_off, gidx);
}

extern "C" __global__ void lingxi_gemv_q3g128_slotsm_t2_mb1u1_bf16(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const int* __restrict__ gidx) {
    lingxi_gemv_q3g128_slotsm_t2_body<1, 1>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,
                                            a_stride, blob, seg_off, scale_off, gidx);
}

// ─────────── ★ P15-㉖：`slotsm` 的 dp4a/int8-x 变体（2026-10-17）───────────
//
// 动机（bw-ceiling 纠偏后实测）：(e) m=7 只有 ~279 GB/s，而 m=1 有 ~380 —— 差距的
// 主体是**每个 token 每个权重 word 重读 32B fp32 x**（L1 流量是权重字节的 8×），
// m=7 时 L1 带宽已被 x 重读打满（核从"计算受限"进一步变成"L1 受限"）。int8 x
//（每行一个 amax scale，host 提前转化）把 x 流量砍 4×；权重侧 SIMD 字节道解码
// + __dp4a（2 条 dp4a 取代 8 条 FP32 FMA）把指令数近乎砍半。
//
// ⚠️ **非位等价**：x 被量化到 int8（每行 amax/127，相对误差 ~2^-8）。权重本身
// 已是 Q4（相对误差 ~2^-4）⇒ x 的量化误差比权重小 16 倍，期望贪心序列不变；
// 判据走 纪律（首 token + 固定贪心序列逐位不变 + spec 接受率），
// 开关 `LINGXI_MOE_DP4A`（probes 登记），默认关。
//
// 数学契约：d_int = Σ_z code[z]·xq[z]（**整数精确**，|code|≤8、|xq|≤127、8 项
// ⇒ |d_int| ≤ 8128，远不溢出）；acc[s] += scc·(sx[s]·float(d_int))。
// 行集合 / group 归属 / 列步长 / 归约结构 与 `slotsm_body` 完全一致。
template <int RPW>
__device__ __forceinline__ void lingxi_gemv_q4g64_slotsm_dp4a_body(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const signed char* __restrict__ xq,
        const float* __restrict__ sx,
        const int* __restrict__ gidx) {
    const int e = blockIdx.y;
    if (e >= E) {
        return;
    }
    const int gi = (gidx != nullptr) ? gidx[e] : e;
    const int s0 = group_off[gi];
    const int s1 = group_off[gi + 1];
    const int m = s1 - s0;
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);
    if (m <= 0) {
        return;
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[gi] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const bf16* xb = x + (size_t)s0 * (size_t)a_stride;
    const bool vec_ok = ((blob & 3) == 0) && ((seg_off & 3) == 0) && ((a_stride & 7) == 0)
                        && ((k & 7) == 0);
    // 每 token 的行 scale（循环不变，提到 c 循环外）。
    float ssx[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= m) {
            break;
        }
        const int xrow = (a_stride == 0) ? 0 : (s0 + s);
        ssx[s] = sx[xrow];
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
    const int row = warp + r * total_warps;
    if (row >= n) {
        break;
    }
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    if (vec_ok) {
        int c0 = lane * 8;
        std::uint32_t w = 0;
        float sC = 0.0f;
        if (c0 + 8 <= k) {
            const int g = c0 >> 6;
            const int t = c0 & 63;
            w = *reinterpret_cast<const std::uint32_t*>(
                payload + (long long)row * row_bytes + g * 32 + (t >> 1));
            sC = __half2float(scales[(long long)row * row_groups + g]);
        }
        for (; c0 + 8 <= k; c0 += 256) {
            const std::uint32_t wc = w;
            const float scc = sC;
            const int cn = c0 + 256;
            if (cn + 8 <= k) {
                const int gn = cn >> 6;
                const int tn = cn & 63;
                w = *reinterpret_cast<const std::uint32_t*>(
                    payload + (long long)row * row_bytes + gn * 32 + (tn >> 1));
                sC = __half2float(scales[(long long)row * row_groups + gn]);
            }
            // ★ SIMD 字节道解码：偶 nibble（z=0,2,4,6）→ se 字节道，奇 nibble → so。
            // (b^8)-8 mod 256 = int8 补码（__vsub4 无跨字节借位）⇒ code 整数值与
            // 旧路径逐位相同；PRMT 把偶/奇字节道交错回 z 序（a_lo=[z0..z3]，a_hi=[z4..z7]）。
            const std::uint32_t se = __vsub4((wc & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
            const std::uint32_t so = __vsub4(((wc >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
            const std::uint32_t a_lo = __byte_perm(se, so, 0x5140);
            const std::uint32_t a_hi = __byte_perm(se, so, 0x7362);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= m) {
                    break;
                }
                const int xrow = (a_stride == 0) ? 0 : (s0 + s);
                // 8×int8 一行（LDG.64）——x 流量是 fp32 路径的 1/4。
                const uint2 xqv =
                    *reinterpret_cast<const uint2*>(xq + (size_t)xrow * (size_t)k + c0);
                int d = __dp4a((int)a_lo, (int)xqv.x, 0);
                d = __dp4a((int)a_hi, (int)xqv.y, d);
                // 变体实测（P15-㉖）：I2F 235.5 vs 魔数法 216.8 GB/s（(l) m=7）⇒ 留 I2F。
                acc[s] += scc * (ssx[s] * (float)d);
            }
        }
        for (; c0 < k; c0 += 256) {
            for (int i = 0; i < 8 && c0 + i < k; ++i) {
                const int c = c0 + i;
                const int g = c >> 6;
                const int t = c & 63;
                const std::uint8_t byte = payload[(long long)row * row_bytes + g * 32 + (t >> 1)];
                const int nb = (t & 1) ? (byte >> 4) : (byte & 0x0F);
                const int code = nb >= 8 ? nb - 16 : nb;
                const float sc = __half2float(scales[(long long)row * row_groups + g]);
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s >= m) {
                        break;
                    }
                    const int xrow = (a_stride == 0) ? 0 : (s0 + s);
                    acc[s] += (float)code * sc * (ssx[s] * (float)xq[(size_t)xrow * (size_t)k + c]);
                }
            }
        }
    } else {
        for (int c = lane; c < k; c += 32) {
            const int g = c >> 6;
            const int t = c & 63;
            const std::uint8_t byte = payload[(long long)row * row_bytes + g * 32 + (t >> 1)];
            const int nb = (t & 1) ? (byte >> 4) : (byte & 0x0F);
            const int code = nb >= 8 ? nb - 16 : nb;
            const float sc = __half2float(scales[(long long)row * row_groups + g]);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= m) {
                    break;
                }
                acc[s] += (float)code * sc * __bfloat162float(xb[(size_t)s * (size_t)a_stride + c]);
            }
        }
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= m) {
            break;
        }
        float a2 = acc[s];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a2 += __shfl_down_sync(0xffffffffu, a2, off);
        }
        if (lane == 0) {
            y[(size_t)(s0 + s) * (size_t)n + row] = __float2bfloat16(a2);
        }
    }
    }  // for r (RPW)
}

extern "C" __global__ void lingxi_gemv_q4g64_slotsm_dp4a(const std::uint8_t* __restrict__ slots,
                                                         const int* __restrict__ slot_ids,
                                                         const bf16* __restrict__ x,
                                                         const int* __restrict__ group_off,
                                                         bf16* __restrict__ y,
                                                         int E,
                                                         int n,
                                                         int k,
                                                         int k_pad,
                                                         int a_stride,
                                                         long long blob,
                                                         long long seg_off,
                                                         long long scale_off,
                                                         const signed char* __restrict__ xq,
                                                         const float* __restrict__ sx,
                                                         const int* gidx) {
    lingxi_gemv_q4g64_slotsm_dp4a_body<1>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,
                                          a_stride, blob, seg_off, scale_off, xq, sx, gidx);
}

extern "C" __global__ void lingxi_gemv_q4g64_slotsm_r2_dp4a(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const signed char* __restrict__ xq,
        const float* __restrict__ sx,
        const int* gidx) {
    lingxi_gemv_q4g64_slotsm_dp4a_body<2>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,
                                          a_stride, blob, seg_off, scale_off, xq, sx, gidx);
}

// ═══════════════ ★ smem-x：激活按 block 一次性搬进 shared（LINGXI_GEMV_SMEM_X）═══════════════
//
// 判因（`tests/t3_gemv_bw.rs::bench_var`，2026-10-23）：gu(E49 n1280 k2560) m=6 时
// bf16 97.2 / dp4a 115.9 GB/s；「x 广播(a_stride=0)」155.3 / 288.4；权重同槽 118.9 / 198.9；
// 两者都去掉 173.6 / 301.1。⇒ **每个 token 的激活重读**是与权重 HBM 流并列的大头：
// 一个 block 的 8 个 warp 各自把同一份 m×k 激活从 global/L2 再读一遍（x 只有 ~1 MB，
// 停在 L2 进不了 L1）。
// 改法：本 block 需要的 x（当前 batch 的 mb ≤ cap 个 token × k）一次性搬进 dynamic shared，
// block 内所有 warp 从 shared 读 ⇒ 激活的 global/L2 流量 ÷8。
// 位安全：搬的是同一批 bf16 值（逐位相同），列集合 / 每个 s 的累加次序 / code 抽取次序不变。
// `xf`（fp32 预转）在本核**不使用**：bf16→f32 精确 ⇒ 与 `xf` 路径逐位一致。
// smem 预算：cap(≤8) × k × 2B（gu k=2560 → ≤40 KB，dn k=1280 → ≤20 KB），< 48 KB 动态上限。
template <int RPW>
__device__ __forceinline__ void lingxi_gemv_q4g64_slotsm_sxb_body(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        int cap) {
    extern __shared__ __align__(16) unsigned char sx_sh_b[];
    bf16* sxb = reinterpret_cast<bf16*>(sx_sh_b);
    const int e = blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int s1 = group_off[e + 1];
    const int m = s1 - s0;
    if (m <= 0) {
        return;
    }
    if (slot_ids[e] < 0) {
        return; // 混档单 launch：他格式组的哨兵槽，跳过
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);
    const int tid = (int)threadIdx.x;
    const int nth = (int)blockDim.x;
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const bool vec_ok = ((blob & 3) == 0) && ((seg_off & 3) == 0) && ((a_stride & 7) == 0);
    const int KP = (k + 7) & ~7;               // 每行 smem 步长（16B 对齐）
    const int mcap = cap < 1 ? 1 : (cap > 8 ? 8 : cap);
    const int n8 = k >> 3;
    constexpr int MB = 8;
    for (int sb = 0; sb < m; sb += mcap) {
        const int mb = (m - sb) < mcap ? (m - sb) : mcap;
        __syncthreads();
        {
            // staging 用最小寄存器足迹：32 位索引 + 不展开。
#pragma unroll 1
            for (int s = 0; s < mb; ++s) {
                const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                const bf16* src = x + xrow * a_stride;
                bf16* dst = sxb + s * KP;
                if (vec_ok) {
#pragma unroll 1
                    for (int i = tid; i < n8; i += nth) {
                        *reinterpret_cast<int4*>(dst + (i << 3)) =
                            *reinterpret_cast<const int4*>(src + (i << 3));
                    }
                    for (int j = (n8 << 3) + tid; j < k; j += nth) {
                        dst[j] = src[j];
                    }
                } else {
#pragma unroll 1
                    for (int i = tid; i < k; i += nth) {
                        dst[i] = src[i];
                    }
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = warp + r * total_warps;
            if (row >= n) {
                break;
            }
            float acc[MB];
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                acc[s] = 0.0f;
            }
            if (vec_ok) {
                constexpr int PF = 4;
                std::uint32_t wbuf[PF];
                float sbuf[PF];
                int c0 = lane * 8;
#pragma unroll
                for (int p = 0; p < PF; ++p) {
                    const int cp = c0 + p * 256;
                    if (cp + 8 <= k) {
                        const int g = cp >> 6;
                        const int t = cp & 63;
                        wbuf[p] = *reinterpret_cast<const std::uint32_t*>(
                            payload + (long long)row * row_bytes + g * 32 + (t >> 1));
                        sbuf[p] = __half2float(scales[(long long)row * row_groups + g]);
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
                        const int gn = cn >> 6;
                        const int tn = cn & 63;
                        wbuf[PF - 1] = *reinterpret_cast<const std::uint32_t*>(
                            payload + (long long)row * row_bytes + gn * 32 + (tn >> 1));
                        sbuf[PF - 1] = __half2float(scales[(long long)row * row_groups + gn]);
                    }
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
                        // ★ smem-x：同一份激活被本 block 的 8 个 warp 共用，只从 shared 读。
                        const bf16* xsp = sxb + s * KP;
                        const int4 xv = *reinterpret_cast<const int4*>(xsp + c0);
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
                for (; c0 < k; c0 += 256) {
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
                            const bf16* xsp = sxb + s * KP;
                            acc[s] += (float)code * sc * __bfloat162float(xsp[c]);
                        }
                    }
                }
            } else {
                for (int c = lane; c < k; c += 32) {
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
                        const bf16* xsp = sxb + s * KP;
                        acc[s] += (float)code * sc * __bfloat162float(xsp[c]);
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
                    y[(size_t)(s0 + sb + s) * (size_t)n + row] = __float2bfloat16(a2);
                }
            }
        }
    }
}

// smem-x 的 dp4a/int8 版本（`LINGXI_MOE_DP4A=1` + `LINGXI_GEMV_SMEM_X=2`）：
// 结构与上面同源，只是 x 是 `xq`（每行 k 个 int8、行 scale `sx`），内积走 `__dp4a`。
// 本核**支持 m > 8**（外层 sb 分批，acc 仍是 8 宽）—— 设备全路由下 group 大小不可预知。
template <int RPW>
__device__ __forceinline__ void lingxi_gemv_q4g64_slotsm_sxq_dp4a_body(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const signed char* __restrict__ xq,
        const float* __restrict__ sx,
        int cap) {
    extern __shared__ __align__(16) unsigned char sx_sh_q[];
    signed char* sxq = reinterpret_cast<signed char*>(sx_sh_q);
    const int e = blockIdx.y;
    if (e >= E) {
        return;
    }
    const int s0 = group_off[e];
    const int s1 = group_off[e + 1];
    const int m = s1 - s0;
    if (m <= 0) {
        return;
    }
    if (slot_ids[e] < 0) {
        return;
    }
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);
    const int tid = (int)threadIdx.x;
    const int nth = (int)blockDim.x;
    const std::uint8_t* payload = slots + (long long)slot_ids[e] * blob + seg_off;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const bf16* xb = x + s0 * a_stride;
    const bool vec_ok = ((blob & 3) == 0) && ((seg_off & 3) == 0) && ((a_stride & 7) == 0)
                        && ((k & 7) == 0);
    const int KP = (k + 7) & ~7;
    const int mcap = cap < 1 ? 1 : (cap > 8 ? 8 : cap);
    const int n16 = k >> 4;
    constexpr int MB = 8;
    for (int sb = 0; sb < m; sb += mcap) {
        const int mb = (m - sb) < mcap ? (m - sb) : mcap;
        __syncthreads();
        {
#pragma unroll 1
            for (int s = 0; s < mb; ++s) {
                const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
                const signed char* src = xq + xrow * k;
                signed char* dst = sxq + s * KP;
#pragma unroll 1
                for (int i = tid; i < n16; i += nth) {
                    *reinterpret_cast<int4*>(dst + (i << 4)) =
                        *reinterpret_cast<const int4*>(src + (i << 4));
                }
                for (int j = (n16 << 4) + tid; j < k; j += nth) {
                    dst[j] = src[j];
                }
            }
        }
        __syncthreads();
        float ssx[MB];
#pragma unroll
        for (int s = 0; s < MB; ++s) {
            if (s >= mb) {
                break;
            }
            const int xrow = (a_stride == 0) ? 0 : (s0 + sb + s);
            ssx[s] = sx[xrow];
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = warp + r * total_warps;
            if (row >= n) {
                break;
            }
            float acc[MB];
#pragma unroll
            for (int s = 0; s < MB; ++s) {
                acc[s] = 0.0f;
            }
            if (vec_ok) {
                int c0 = lane * 8;
                std::uint32_t w = 0;
                float sC = 0.0f;
                if (c0 + 8 <= k) {
                    const int g = c0 >> 6;
                    const int t = c0 & 63;
                    w = *reinterpret_cast<const std::uint32_t*>(
                        payload + (long long)row * row_bytes + g * 32 + (t >> 1));
                    sC = __half2float(scales[(long long)row * row_groups + g]);
                }
                for (; c0 + 8 <= k; c0 += 256) {
                    const std::uint32_t wc = w;
                    const float scc = sC;
                    const int cn = c0 + 256;
                    if (cn + 8 <= k) {
                        const int gn = cn >> 6;
                        const int tn = cn & 63;
                        w = *reinterpret_cast<const std::uint32_t*>(
                            payload + (long long)row * row_bytes + gn * 32 + (tn >> 1));
                        sC = __half2float(scales[(long long)row * row_groups + gn]);
                    }
                    const std::uint32_t se = __vsub4((wc & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
                    const std::uint32_t so = __vsub4(((wc >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
                    const std::uint32_t a_lo = __byte_perm(se, so, 0x5140);
                    const std::uint32_t a_hi = __byte_perm(se, so, 0x7362);
#pragma unroll
                    for (int s = 0; s < MB; ++s) {
                        if (s >= mb) {
                            break;
                        }
                        const signed char* xsp = sxq + s * KP;
                        const uint2 xqv = *reinterpret_cast<const uint2*>(xsp + c0);
                        int d = __dp4a((int)a_lo, (int)xqv.x, 0);
                        d = __dp4a((int)a_hi, (int)xqv.y, d);
                        acc[s] += scc * (ssx[s] * (float)d);
                    }
                }
                for (; c0 < k; c0 += 256) {
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
                            const signed char* xsp = sxq + s * KP;
                            acc[s] += (float)code * sc * (ssx[s] * (float)xsp[c]);
                        }
                    }
                }
            } else {
                for (int c = lane; c < k; c += 32) {
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
                        acc[s] += (float)code * sc
                                  * __bfloat162float(xb[(size_t)(sb + s) * (size_t)a_stride + c]);
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
                    y[(size_t)(s0 + sb + s) * (size_t)n + row] = __float2bfloat16(a2);
                }
            }
        }
    }
}

extern "C" __global__ void lingxi_gemv_q4g64_slotsm_sxb_bf16(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        int cap) {
    lingxi_gemv_q4g64_slotsm_sxb_body<1>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,
                                         a_stride, blob, seg_off, scale_off, cap);
}

extern "C" __global__ void lingxi_gemv_q4g64_slotsm_sxq_dp4a(
        const std::uint8_t* __restrict__ slots,
        const int* __restrict__ slot_ids,
        const bf16* __restrict__ x,
        const int* __restrict__ group_off,
        bf16* __restrict__ y,
        int E,
        int n,
        int k,
        int k_pad,
        int a_stride,
        long long blob,
        long long seg_off,
        long long scale_off,
        const signed char* __restrict__ xq,
        const float* __restrict__ sx,
        int cap) {
    lingxi_gemv_q4g64_slotsm_sxq_dp4a_body<1>(slots, slot_ids, x, group_off, y, E, n, k, k_pad,
                                              a_stride, blob, seg_off, scale_off, xq, sx, cap);
}

// bf16 → int8 **行量化**（dp4a 的"提前转化"）：每行一个 scale = amax/127，
// xq = rintf(x/scale) 截断到 [-127,127]。grid.y = 行，block = 256 线程。
extern "C" __global__ void lingxi_convert_bf16_q8_rs(const bf16* __restrict__ x,
                                                     signed char* __restrict__ y,
                                                     float* __restrict__ sx,
                                                     int cols) {
    const int row = blockIdx.y;
    const bf16* xr = x + (size_t)row * cols;
    signed char* yr = y + (size_t)row * cols;
    float m = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        m = fmaxf(m, fabsf(__bfloat162float(xr[c])));
    }
    __shared__ float red[8];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    }
    if ((threadIdx.x & 31) == 0) {
        red[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        m = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        }
        if (threadIdx.x == 0) {
            const float s = m > 0.0f ? m / 127.0f : 1.0f;
            red[0] = s;
            sx[row] = s;
        }
    }
    __syncthreads();
    const float s = red[0];
    const float inv = 1.0f / s;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        int q = (int)rintf(__bfloat162float(xr[c]) * inv);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        yr[c] = (signed char)q;
    }
}

// ─────────── 分块 Q4 瘦长 GEMM（t>1 专用）───────────
//
// ⚠️ 修的是 根因 1：t>1 时原来走 `dequant_q4g64_dense_bf16` **把整块权重解回 bf16**
// 再喂给瘦长 GEMM —— `qkvz`（21.2 MB Q4）变成「读 21 + 写 42 + 再读 42」= **105 MB**，
// 而 t=1 只要 21 MB。实测 attn 在 t=2 就从 9.1 ms 涨到 38.5 ms（4.2×），主因就是它。
//
// 本核**直接吃 Q4**：一个 block 管 256 个输出行（一线程一行），k 方向按 256 分块，
// 每个 k 块把 `x` 的 `[m, 256]` 协同搬进 shared（否则每个 block 都要重读 x，
// n=16384 时是 655 MB 的 L2 流量）。每个线程的 8 个累加器对应 m 个 token。
//
// 关键收益：**8 个 nibble 的解码只做一次，被 m 个 token 共用** —— 每 64 个 MAC
// 只要 ~113 条指令（1.77 条/MAC），而 t=1 的逐 token GEMV 是 7.25 条/MAC。
// 权重行只读一遍 → 流量回到 21.2 MB。**数学与旧路径一致（求和次序不同）。**
// 结构（第一版写成"一线程一行 → grid=ceil(n/256)=64 个 block"，只有 ~6 warp/SM，
// 实测**比旧路还慢 15%**；改成 warp-per-row 后并行度拉回 1024+ 个 block）：
// * grid = `ceil(n / 8)`，blockDim = 256（**8 个 warp**），**warp w 负责第 `blockIdx.x*8+w` 行**；
// （⚠️ 第一版这里写成 16 行/warp 但 blockDim 只有 256 → 后 8 行根本没算，首 token 从 11751 变 13307）
// * k 方向按 `Q4G_KTILE` 分块，每块把 `x[m][KTILE]` 协同搬进 shared（被本 block 的 8 行共用）；
// * 每个 lane 一次吃 8 列（1 个 u32 = 8 个 nibble），**解码一次、被 m 个 token 复用**。
// **逐元素版**（只用于 `LINGXI_Q4G_DBG=1` 时定位问题）：一线程一行、逐字节解码、f64 累加。
// 它和 `lingxi_dequant_q4g64_dense_bf16` 的取址逐字一致，**若它也错，说明问题在参数/指针**；
// 若它对，说明问题在优化过的内层循环。
extern "C" __global__ void lingxi_gemm_q4_skinny_dbg(const std::uint8_t* __restrict__ wq,
                                                     const bf16* __restrict__ x,
                                                     bf16* __restrict__ y,
                                                     int n, int k, int k_pad, int m,
                                                     int row_off, long long scale_off) {
    const int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= n) {
        return;
    }
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* wp = wq + (long long)(row_off + row) * (long long)row_bytes;
    const __half* scp =
        reinterpret_cast<const __half*>(wq + scale_off) + (long long)(row_off + row) * (long long)row_groups;
    for (int s = 0; s < m; ++s) {
        const bf16* xr = x + (size_t)s * (size_t)k;
        double acc = 0.0;
        for (int c = 0; c < k; ++c) {
            const int g = c >> 6;
            const int t = c & 63;
            const std::uint8_t byte = wp[g * 32 + (t >> 1)];
            const int nib = (t & 1) ? (byte >> 4) : (byte & 0x0F);
            const int code = nib >= 8 ? nib - 16 : nib;
            acc += (double)code * (double)__half2float(scp[g]) * (double)__bfloat162float(xr[c]);
        }
        y[(size_t)s * (size_t)n + row] = __float2bfloat16((float)acc);
    }
}

#define Q4G_ROWS 8
#define Q4G_KTILE 1024

extern "C" __global__ void lingxi_gemm_q4_skinny_bf16(const std::uint8_t* __restrict__ wq,
                                                       const bf16* __restrict__ x,
                                                       bf16* __restrict__ y,
                                                       int n, int k, int k_pad, int m,
                                                       int row_off, long long scale_off,
                                                       int epi, float a, float b) {
    extern __shared__ bf16 sx[];  // [8, Q4G_KTILE]
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = (int)blockIdx.x * Q4G_ROWS + warp;
    // ★ 2026-10-22 grid.y 行分块：blockIdx.y 每块接 8 行（m 任意大）。每块 8 行子块与
    // 旧"逐片 launch"数学完全同构（同一 k 分块/共享舞台/累加次序）⇒ 逐位等价；
    // 并发块共享 Q4 权重读 ⇒ DRAM 流量从 ⌈m/8⌉ 遍降到 ~1 遍。grid.y==1 ⇒ 与旧版一致。
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const bool valid = row < n;
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* wp = wq + (long long)(row_off + row) * (long long)row_bytes;
    const __half* scp =
        reinterpret_cast<const __half*>(wq + scale_off) + (long long)(row_off + row) * (long long)row_groups;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    for (int k0 = 0; k0 < k; k0 += Q4G_KTILE) {
        const int kn = (k - k0) < Q4G_KTILE ? (k - k0) : Q4G_KTILE;
        const int nvec_per_tok = (kn + 7) >> 3;
        const int nvec = nvec_per_tok * mt;  // ★ 分块后是**本块**行数（旧版 = m ≤ 8 恰好相同）
        for (int i = tid; i < nvec; i += (int)blockDim.x) {
            const int s = i / nvec_per_tok;
            const int c = (i % nvec_per_tok) << 3;
            const bf16* xs = x + (size_t)(r0 + s) * (size_t)k + (size_t)k0 + c;
            bf16* dst = sx + s * Q4G_KTILE + c;
            // ⚠️ 早先写成 `bf16 tmp[8]` 中转，`reinterpret_cast<int4*>(tmp)` 是**未对齐的
            // 128 位访问**（nvcc 只保证 local 数组 2 字节对齐）→ 实测**全部输出错误**。
            // 直接 global→shared 拷（对齐有保证），尾块走标量。
            if (c + 8 <= kn) {
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(xs);
            } else {
                for (int z = 0; z < 8; ++z) {
                    dst[z] = (c + z < kn) ? xs[z] : __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();
        if (valid) {
            const int ncol = (kn + 7) & ~7;  // 本 tile 的 8 对齐列数
            for (int c0 = lane * 8; c0 < ncol; c0 += 256) {
                const int c = k0 + c0;
                const int g = c >> 6;
                const int t = c & 63;
                const std::uint32_t word =
                    *reinterpret_cast<const std::uint32_t*>(wp + g * 32 + (t >> 1));
                const float sc = __half2float(scp[g]);
                float cd[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    const int nb = (int)((word >> (4 * z)) & 0xFu);
                    cd[z] = (float)(nb >= 8 ? nb - 16 : nb);
                }
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s >= mt) {
                        break;
                    }
                    // ㊽+10：x 向量化读——8 个 bf16 一次 int4（c0 是 8 的倍数、
                    // 舞台行 1024 对齐 ⇒ 16B 对齐），把每权重的 smem load 指令
                    // 从 8 条标量降到 1 条向量；解压/累加次序全不动 ⇒ **逐位等价**。
                    const int4 xv = *reinterpret_cast<const int4*>(sx + s * Q4G_KTILE + c0);
                    const bf16* xb = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        acc[s] += sc * cd[z] * __bfloat162float(xb[z]);
                    }
                }
            }
        }
        __syncthreads();
    }
    // ⚠️ **必须做 warp 内归约**：每个 lane 只负责 `4 次 × 8 列 = 32 列`（共 1024 列），
    // 少了这一步就是 32 个 lane 各写一份 1/32 的部分和到同一个 `y[s*n+row]`，
    // 谁最后写谁生效 —— 实测表现为"所有输出都错、量级只有正确值的 1/10"。
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
                if (epi == 1) {
                    const float t2 = rr * a;
                    rr = t2 / (1.0f + __expf(-t2));
                } else if (epi == 2) {
                    const float t2 = rr * a;
                    rr = (1.0f / (1.0f + __expf(-t2))) * b;
                } else if (epi == 3) {
                    rr = rr * a + b;
                }
                y[(size_t)(r0 + s) * (size_t)n + row] = __float2bfloat16(rr);
            }
        }
    }
}

// ─────────── QSA 四段投影合发（q | k | gate | v，一次 launch）───────────
//
// 背景（2026-10-23 事件口径，）：QSA 层均 0.355 ms 里 `proj` 段 **150~161 µs**
// 是四发（n=6144/512/6144/512，grid=768/64/768/64 块）。后两发只有 64 块 < 80 SM，
// 每发都被 ~6~8 µs 的发射地板吃掉；反事实微基准把四段并成 `n=13312` 一发 **122.3 µs**
// ⇒ 省 ~39 µs/层（−24%）。
//
// **逐位等价论证**：本核与 `lingxi_gemm_q4_skinny_bf16` 的线程→列映射（`row =
// blockIdx.x*8 + warp`、`c0 = lane*8` 步进 256）、k 分块（`Q4G_KTILE`）、x 暂存 shared 的
// 写法、8 元素内 z 次序、warp shfl + `lane==0` 写、以及累加次序**逐字相同**；
// 唯一区别是**写哪个 y**：按全局行号落回原来那一段各自的 `y[seg]`（行内索引 = row - 段起点），
// 与四次独立 launch 写到同一地址、同一个 bf16 round 点。
// 权重按**整块矩阵的绝对行号**（`row_off + row`）取，与旧路四段各自的 `row_off` 之和一致。
extern "C" __global__ void lingxi_gemm_q4_skinny4_bf16(const std::uint8_t* __restrict__ wq,
                                                        const bf16* __restrict__ x,
                                                        bf16* __restrict__ y0,
                                                        bf16* __restrict__ y1,
                                                        bf16* __restrict__ y2,
                                                        bf16* __restrict__ y3,
                                                        int n0, int n1, int n2, int n3,
                                                        int k, int k_pad, int m,
                                                        int row_off, long long scale_off,
                                                        int epi, float a, float b) {
    extern __shared__ bf16 sx[];  // [8, Q4G_KTILE]
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = (int)blockIdx.x * Q4G_ROWS + warp;
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const int n = n0 + n1 + n2 + n3;
    const bool valid = row < n;
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* wp = wq + (long long)(row_off + row) * (long long)row_bytes;
    const __half* scp =
        reinterpret_cast<const __half*>(wq + scale_off) + (long long)(row_off + row) * (long long)row_groups;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    for (int k0 = 0; k0 < k; k0 += Q4G_KTILE) {
        const int kn = (k - k0) < Q4G_KTILE ? (k - k0) : Q4G_KTILE;
        const int nvec_per_tok = (kn + 7) >> 3;
        const int nvec = nvec_per_tok * mt;
        for (int i = tid; i < nvec; i += (int)blockDim.x) {
            const int s = i / nvec_per_tok;
            const int c = (i % nvec_per_tok) << 3;
            const bf16* xs = x + (size_t)(r0 + s) * (size_t)k + (size_t)k0 + c;
            bf16* dst = sx + s * Q4G_KTILE + c;
            if (c + 8 <= kn) {
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(xs);
            } else {
                for (int z = 0; z < 8; ++z) {
                    dst[z] = (c + z < kn) ? xs[z] : __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();
        if (valid) {
            const int ncol = (kn + 7) & ~7;
            for (int c0 = lane * 8; c0 < ncol; c0 += 256) {
                const int c = k0 + c0;
                const int g = c >> 6;
                const int t = c & 63;
                const std::uint32_t word =
                    *reinterpret_cast<const std::uint32_t*>(wp + g * 32 + (t >> 1));
                const float sc = __half2float(scp[g]);
                float cd[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    const int nb = (int)((word >> (4 * z)) & 0xFu);
                    cd[z] = (float)(nb >= 8 ? nb - 16 : nb);
                }
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s >= mt) {
                        break;
                    }
                    const int4 xv = *reinterpret_cast<const int4*>(sx + s * Q4G_KTILE + c0);
                    const bf16* xb = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        acc[s] += sc * cd[z] * __bfloat162float(xb[z]);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (valid) {
        // 段归属（行区间）：q | k | gate | v —— 与旧路四次 launch 的 n/row_off 完全一致。
        bf16* yseg;
        int lrow;
        int nseg;
        if (row < n0) {
            yseg = y0; lrow = row; nseg = n0;
        } else if (row < n0 + n1) {
            yseg = y1; lrow = row - n0; nseg = n1;
        } else if (row < n0 + n1 + n2) {
            yseg = y2; lrow = row - n0 - n1; nseg = n2;
        } else {
            yseg = y3; lrow = row - n0 - n1 - n2; nseg = n3;
        }
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
                if (epi == 1) {
                    const float t2 = rr * a;
                    rr = t2 / (1.0f + __expf(-t2));
                } else if (epi == 2) {
                    const float t2 = rr * a;
                    rr = (1.0f / (1.0f + __expf(-t2))) * b;
                } else if (epi == 3) {
                    rr = rr * a + b;
                }
                yseg[(size_t)(r0 + s) * (size_t)nseg + lrow] = __float2bfloat16(rr);
            }
        }
    }
}

// ─────────── GDN：qkvz（Q4 行）+ a/b（bf16 行）合发 ───────────
//
// 背景：GDN 层均 301~307 µs 里 `a、b 两个 tiny GEMV` **17.5 µs**，
// 而它们读的权重只有 96×2560×2B = **0.49 MB** ⇒ 几乎全是每发 ~6~8 µs 的 launch 地板
// + 两次排空。这里把 a/b 的 96 行当成 `qkvz`（n=conv_dim=10240）那次 launch 的**尾巴**：
// grid.x = `ceil(n_q4/8) + 2*nvh`，前 `ceil(n_q4/8)` 块跑 Q4 体，后 `2*nvh` 块跑
// **`lingxi_gemm_skinny_row_bf16` 的行体**（a 的 48 行 + b 的 48 行，各自输出到 a/b）。
//
// **逐位等价论证**（两段都逐字保留）：
// * Q4 段：与 `lingxi_gemm_q4_skinny_bf16` 同一份代码（线程→列映射 / k 分块 / smem 舞台 /
// 8 元素 z 次序 / warp 归约 / bf16 round 点全部不动），只是本核输出恒为 `yq`。
// * bf16 段：与 `lingxi_gemm_skinny_row_bf16` 同一份代码（`c0 = tid*8` 步进 `blockDim*8`、
// 每 8 列 `d` 累加、warp shfl + `sred[32][8]` 两级归约、归一化后 round bf16）。
// ⚠️ `sred` 复用动态 smem（`sx` 的前 1 KB）而不是另开 `__shared__`：静态 smem 会
// 把本核的占用从 16 KB/块抬到 17 KB/块，连带拖慢 Q4 段（那才是大的那块）。
// * 差异只在"哪块跑哪段"和输出指针/行号，与两次独立 launch 写到同一地址等价。
extern "C" __global__ void lingxi_gemm_q4_skinny_ab_bf16(const std::uint8_t* __restrict__ wq,
                                                         const bf16* __restrict__ x,
                                                         bf16* __restrict__ yq,
                                                         const bf16* __restrict__ wab,
                                                         bf16* __restrict__ ya,
                                                         bf16* __restrict__ yb,
                                                         int n_q4, int nvh,
                                                         int k, int k_pad, int m,
                                                         int row_off, long long scale_off) {
    extern __shared__ bf16 sx[];  // Q4 段：[8, Q4G_KTILE]；bf16 段：前 1KB 当 float[32][8]
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const int q4_blocks = (n_q4 + Q4G_ROWS - 1) / Q4G_ROWS;
    if ((int)blockIdx.x >= q4_blocks) {
        // ── a/b 行体（逐字同 `lingxi_gemm_skinny_row_bf16`）──
        const int rb = (int)blockIdx.x - q4_blocks;   // 0..2*nvh-1
        const bool is_a = rb < nvh;
        const int lrow = is_a ? rb : rb - nvh;
        bf16* yr = is_a ? ya : yb;
        const bf16* wr = wab + (size_t)rb * (size_t)k;
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
                    if (s >= mt) {
                        break;
                    }
                    acc[s] += wf * __bfloat162float(x[(size_t)(r0 + s) * (size_t)k + c0 + i]);
                }
            }
        }
        float (*sred)[8] = reinterpret_cast<float (*)[8]>(sx);
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
                    yr[(size_t)(r0 + s) * (size_t)nvh + lrow] = __float2bfloat16(v);
                }
            }
        }
        return;
    }

    // ── Q4 段（逐字同 `lingxi_gemm_q4_skinny_bf16`，epi 恒 0）──
    const int row = (int)blockIdx.x * Q4G_ROWS + warp;
    const bool valid = row < n_q4;
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* wp = wq + (long long)(row_off + row) * (long long)row_bytes;
    const __half* scp =
        reinterpret_cast<const __half*>(wq + scale_off) + (long long)(row_off + row) * (long long)row_groups;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    for (int k0 = 0; k0 < k; k0 += Q4G_KTILE) {
        const int kn = (k - k0) < Q4G_KTILE ? (k - k0) : Q4G_KTILE;
        const int nvec_per_tok = (kn + 7) >> 3;
        const int nvec = nvec_per_tok * mt;
        for (int i = tid; i < nvec; i += (int)blockDim.x) {
            const int s = i / nvec_per_tok;
            const int c = (i % nvec_per_tok) << 3;
            const bf16* xs = x + (size_t)(r0 + s) * (size_t)k + (size_t)k0 + c;
            bf16* dst = sx + s * Q4G_KTILE + c;
            if (c + 8 <= kn) {
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(xs);
            } else {
                for (int z = 0; z < 8; ++z) {
                    dst[z] = (c + z < kn) ? xs[z] : __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();
        if (valid) {
            const int ncol = (kn + 7) & ~7;
            for (int c0 = lane * 8; c0 < ncol; c0 += 256) {
                const int c = k0 + c0;
                const int g = c >> 6;
                const int t = c & 63;
                const std::uint32_t word =
                    *reinterpret_cast<const std::uint32_t*>(wp + g * 32 + (t >> 1));
                const float sc = __half2float(scp[g]);
                float cd[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    const int nb = (int)((word >> (4 * z)) & 0xFu);
                    cd[z] = (float)(nb >= 8 ? nb - 16 : nb);
                }
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s >= mt) {
                        break;
                    }
                    const int4 xv = *reinterpret_cast<const int4*>(sx + s * Q4G_KTILE + c0);
                    const bf16* xb = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        acc[s] += sc * cd[z] * __bfloat162float(xb[z]);
                    }
                }
            }
        }
        __syncthreads();
    }
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
                yq[(size_t)(r0 + s) * (size_t)n_q4 + row] = __float2bfloat16(rr);
            }
        }
    }
}

// ─────────── 多行/warp 的稠密 Q4 瘦长 GEMM（2026-10-24，**逐位等价**）───────────
//
// 动机（docs 的探针账，t14_q4_dense_skinny_bw）：旧核对每 4 B 权重（8 个 code）
// 要**为每个 token** 付 1 次 smem 读 + 8 次 bf16→f32 + 8 次 FFMA，而每 block 只摊 8 个
// 输出行 ⇒ x 的 global 舞台流量是权重 payload 的 m/2 倍（m=6 时 3 倍），且每 k-tile 两道
// barrier 把 staging 与 compute 完全串起来。同形纯读 4 B/lane 有 360~430 GB/s、ALU 地板
// （解包+I2F+FFMA）约 200 GB/s，旧核只有 125~133 ⇒ **不是 HBM 的账，是"每字节指令数 +
// x 重复付费"的账**（与专家段 同源）。
//
// 改法（算式一字未改，只有取数/复用结构变）：
// * **一个 warp 连算 R=2 个输出行**：x 的 smem 读与 bf16→f32 只做一次、被 2 行共用；
// 每 block 摊 16 个输出行 ⇒ 舞台流量与 barrier 次数 ÷2；
// * 2 个权重字是**独立 load**（没有移位/预取链）。
//
// **位安全论据**：每行的 lane→列集合（`lane*8 + 256*i`）、chunk 次序（c0 升序）、每 chunk
// 内 z=0..7 的串行累加次序、`sc*cd*xf` 的乘序、warp shfl 归约树、bf16 round 点**全部与
// `lingxi_gemm_q4_skinny_bf16` 逐字相同**；`xf[z]` 只是把同一个 bf16 **精确**展宽成 f32 后
// 复用（bf16→f32 无损）。输出落点按行区间分到 y0..y3，与 `_skinny4` 的写法一致。
// 判据见 `tests/t14_q4_dense_skinny_bw.rs`（n=1024、n=1000 非整除边界、m=5/6、R=1/2/4）。
#ifndef Q4G_R2
#define Q4G_R2 2
#endif

template <int R>
__device__ __forceinline__ void q4_skinny_r2_body(
    const std::uint8_t* __restrict__ wq, const bf16* __restrict__ x,
    bf16* __restrict__ y0, bf16* __restrict__ y1, bf16* __restrict__ y2,
    bf16* __restrict__ y3, int n0, int n1, int n2, int n3, int k, int k_pad, int m,
    int row_off, long long scale_off, int epi, float ea, float eb) {
    extern __shared__ bf16 sx[];  // [8, Q4G_KTILE]，布局同旧核
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const int n = n0 + n1 + n2 + n3;
    const int base = ((int)blockIdx.x * (int)(blockDim.x >> 5) + warp) * R;
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

    for (int k0 = 0; k0 < k; k0 += Q4G_KTILE) {
        const int kn = (k - k0) < Q4G_KTILE ? (k - k0) : Q4G_KTILE;
        {
            // x 舞台：与旧核逐字相同（含尾块的标量路）
            const int nvec_per_tok = (kn + 7) >> 3;
            const int nvec = nvec_per_tok * mt;
            for (int i = tid; i < nvec; i += (int)blockDim.x) {
                const int s = i / nvec_per_tok;
                const int c = (i % nvec_per_tok) << 3;
                const bf16* xs = x + (size_t)(r0 + s) * (size_t)k + (size_t)k0 + c;
                bf16* dst = sx + s * Q4G_KTILE + c;
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
                const int4 xv = *reinterpret_cast<const int4*>(sx + s * Q4G_KTILE + c0);
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

    // 写回：段归属（行区间）与 `_skinny4` 一致；归约树 / bf16 round 点与旧核逐字相同。
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (!ok[r]) {
            continue;
        }
        const int row = base + r;
        bf16* yseg;
        int lrow;
        int nseg;
        if (row < n0) {
            yseg = y0; lrow = row; nseg = n0;
        } else if (row < n0 + n1) {
            yseg = y1; lrow = row - n0; nseg = n1;
        } else if (row < n0 + n1 + n2) {
            yseg = y2; lrow = row - n0 - n1; nseg = n2;
        } else {
            yseg = y3; lrow = row - n0 - n1 - n2; nseg = n3;
        }
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
                if (epi == 1) {
                    const float t2 = rr * ea;
                    rr = t2 / (1.0f + __expf(-t2));
                } else if (epi == 2) {
                    const float t2 = rr * ea;
                    rr = (1.0f / (1.0f + __expf(-t2))) * eb;
                } else if (epi == 3) {
                    rr = rr * ea + eb;
                }
                yseg[(size_t)(r0 + s) * (size_t)nseg + lrow] = __float2bfloat16(rr);
            }
        }
    }
}

// 单段版（= `gemm_q4_skinny_bf16` 的替身）：grid.x = ceil(n/(8*R))、block 256。
extern "C" __global__ void lingxi_gemm_q4_skinny_r2_bf16(
    const std::uint8_t* __restrict__ wq, const bf16* __restrict__ x, bf16* __restrict__ y,
    int n, int k, int k_pad, int m, int row_off, long long scale_off, int epi, float a, float b) {
    q4_skinny_r2_body<Q4G_R2>(wq, x, y, nullptr, nullptr, nullptr, n, 0, 0, 0, k, k_pad, m,
                              row_off, scale_off, epi, a, b);
}

// 四段版（= `gemm_q4_skinny4_bf16` 的替身，QSA q|k|gate|v）
extern "C" __global__ void lingxi_gemm_q4_skinny4_r2_bf16(
    const std::uint8_t* __restrict__ wq, const bf16* __restrict__ x, bf16* __restrict__ y0,
    bf16* __restrict__ y1, bf16* __restrict__ y2, bf16* __restrict__ y3, int n0, int n1, int n2,
    int n3, int k, int k_pad, int m, int row_off, long long scale_off) {
    q4_skinny_r2_body<Q4G_R2>(wq, x, y0, y1, y2, y3, n0, n1, n2, n3, k, k_pad, m, row_off,
                              scale_off, 0, 0.0f, 0.0f);
}

// GDN qkvz + a/b 尾巴版（= `gemm_q4_skinny_ab_bf16` 的替身）：前 `ceil(n_q4/(8R))` 块跑 Q4 体，
// 后 `2*nvh` 块跑 `gemm_skinny_row_bf16` 的行体（**逐字照抄旧核的尾巴**）。
extern "C" __global__ void lingxi_gemm_q4_skinny_ab_r2_bf16(
    const std::uint8_t* __restrict__ wq, const bf16* __restrict__ x, bf16* __restrict__ yq,
    const bf16* __restrict__ wab, bf16* __restrict__ ya, bf16* __restrict__ yb, int n_q4,
    int nvh, int k, int k_pad, int m, int row_off, long long scale_off) {
    extern __shared__ bf16 sx[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const int q4_blocks = (n_q4 + (Q4G_ROWS * Q4G_R2) - 1) / (Q4G_ROWS * Q4G_R2);
    if ((int)blockIdx.x >= q4_blocks) {
        // ── a/b 行体（与 `lingxi_gemm_skinny_row_bf16` 逐字相同）──
        const int rb = (int)blockIdx.x - q4_blocks;
        const bool is_a = rb < nvh;
        const int lrow = is_a ? rb : rb - nvh;
        bf16* yr = is_a ? ya : yb;
        const bf16* wr = wab + (size_t)rb * (size_t)k;
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
                    if (s >= mt) {
                        break;
                    }
                    acc[s] += wf * __bfloat162float(x[(size_t)(r0 + s) * (size_t)k + c0 + i]);
                }
            }
        }
        float (*sred)[8] = reinterpret_cast<float (*)[8]>(sx);
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
                    yr[(size_t)(r0 + s) * (size_t)nvh + lrow] = __float2bfloat16(v);
                }
            }
        }
        return;
    }

    q4_skinny_r2_body<Q4G_R2>(wq, x, yq, nullptr, nullptr, nullptr, n_q4, 0, 0, 0, k, k_pad, m,
                              row_off, scale_off, 0, 0.0f, 0.0f);
}

// ㊽+9 **M≤8 行槽版** Q4 瘦长 GEMM（`LINGXI_Q4GEMM_S8=1`，m∈2..8 时替换 skinny）。
//
// 旧 skinny 的账：每 lane 负责 8 列、对 **m 行串行** 累加 ⇒ 每读 1 个权重字
// （8 权重）要配 8×m 次 x smem 读 + bf16→f32；m=6 时 smem 读是 FMA 的 6 倍，
// 实测只有 60~83GB/s（KTS 分解）。f32 舞台改良死於 smem 占用（16→32KB），
// 大 tile MMA 下放死於 padding 白算（2.4×慢）——两条改良路全负后只剩新写。
//
// 本核：lane = （行槽 rs=lane&7, 列槽 cs=lane>>3) —— 每 lane **只累加自己那一行**
// 的 k 片（acc 单标量）：每权重字配 8 次 x 读且 8 个 bf16 连续 = **一次 int4
// 向量化读**；同一字被 8 个行槽 lane 广播读（同地址 ⇒ L1 单事务，全局流量仍 1 遍）。
// 收尾同行槽 4 个列槽蝴蝶归约（shfl 8/16），cs==0 的 lane 写 y[rs][n]。
// 归约次序 ≠ 旧核（列片升序 + 4 路树）⇒ **有损档**（判据 PLAN_NEXT：首 token
// 相等 + 质量抽检）。x 舞台布局/掩码与旧核完全相同。
extern "C" __global__ void lingxi_gemm_q4_s8_bf16(const std::uint8_t* __restrict__ wq,
                                                  const bf16* __restrict__ x,
                                                  bf16* __restrict__ y,
                                                  int n, int k, int k_pad, int m,
                                                  int row_off, long long scale_off,
                                                  int epi, float a, float b) {
    extern __shared__ bf16 sx[];  // [8, Q4G_KTILE]，布局同旧核
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = (int)blockIdx.x * Q4G_ROWS + warp;
    const bool valid = row < n;
    const int rs = lane & 7;   // 行槽
    const int cs = lane >> 3;  // 列槽（4 个）
    const bool row_ok = rs < m;

    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* wp = wq + (long long)(row_off + row) * (long long)row_bytes;
    const __half* scp =
        reinterpret_cast<const __half*>(wq + scale_off) + (long long)(row_off + row) * (long long)row_groups;

    float acc = 0.0f;
    for (int k0 = 0; k0 < k; k0 += Q4G_KTILE) {
        const int kn = (k - k0) < Q4G_KTILE ? (k - k0) : Q4G_KTILE;
        const int nvec_per_tok = (kn + 7) >> 3;
        const int nvec = nvec_per_tok * m;
        for (int i = tid; i < nvec; i += (int)blockDim.x) {
            const int s = i / nvec_per_tok;
            const int c = (i % nvec_per_tok) << 3;
            const bf16* xs = x + (size_t)s * (size_t)k + (size_t)k0 + c;
            bf16* dst = sx + s * Q4G_KTILE + c;
            if (c + 8 <= kn) {
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(xs);
            } else {
                for (int z = 0; z < 8; ++z) {
                    dst[z] = (c + z < kn) ? xs[z] : __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();
        if (valid && row_ok) {
            const int ncol = (kn + 7) & ~7;  // 本 tile 的 8 对齐列数
            const bf16* xr = sx + rs * Q4G_KTILE;
            for (int c0 = cs * 8; c0 < ncol; c0 += 32) {
                const int c = k0 + c0;
                const int g = c >> 6;
                const int t = c & 63;
                const std::uint32_t word =
                    *reinterpret_cast<const std::uint32_t*>(wp + g * 32 + (t >> 1));
                const float sc = __half2float(scp[g]);
                const int4 xv = *reinterpret_cast<const int4*>(xr + c0);  // 8×bf16 一次读
                const bf16* xb = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    const int nb = (int)((word >> (4 * z)) & 0xFu);
                    const float cd = (float)(nb >= 8 ? nb - 16 : nb);
                    acc += sc * cd * __bfloat162float(xb[z]);
                }
            }
        }
        __syncthreads();
    }
    // 行槽内 4 列槽归约：写者 lane=rs 最终持有 Σ_{cs=0..3} acc。
#pragma unroll
    for (int off = 8; off <= 16; off <<= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (valid && row_ok && cs == 0) {
        float rr = acc;
        if (epi == 1) {
            const float t2 = rr * a;
            rr = t2 / (1.0f + __expf(-t2));
        } else if (epi == 2) {
            const float t2 = rr * a;
            rr = (1.0f / (1.0f + __expf(-t2))) * b;
        } else if (epi == 3) {
            rr = rr * a + b;
        }
        y[(size_t)rs * (size_t)n + row] = __float2bfloat16(rr);
    }
}

// **一个 warp 算 4 行**的版本：x 的 8 个值（含 8 次 bf16→f32 转换）只读/换一次，
// 却被 4 行共用；nibble 提取与 FMA 仍按行做。
//
// 动机（实测）：单行版 Q4 GEMV 只有 **328 GB/s**，而 bf16 GEMV 有 724 GB/s。
// Q4 每读 0.5 字节权重就要配 1 次 x load + 1 次 FMA + ~3 次 nibble 位运算 → 指令/字节
// 天生是 bf16 的 ~3 倍，于是"字节省 3.77x"被"指令多 3x"吃掉一半。把 x 侧的
// load+转换按 4 行摊薄后，这部分压力降到 1/4。
extern "C" __global__ void lingxi_gemv_q4g64_dense4_epi(const std::uint8_t* __restrict__ w,
                                                        const bf16* __restrict__ x,
                                                        bf16* __restrict__ y,
                                                        int n,
                                                        int k,
                                                        int k_pad,
                                                        int row_off,
                                                        long long scale_off,
                                                        int epi,
                                                        float a,
                                                        float b) {
    // R=8 试过：微基准 [16384,2560] 348 vs 343 GB/s（略好），但**整机 p10 反而
    // 从 0.0270 掉到 0.0285**——warp 数减半（4096→2048）让并行度不足。保持 4。
    constexpr int R = 4;
    // nibble→float 查表（同 `lingxi_gemv_q4g64_slots_body` 的长注释）。
    // ⚠️ 放在 `base >= n` 早退**之前**——那是块内非一致退出。
    __shared__ float2 c_lut[256];
    for (int lb = (int)threadIdx.x; lb < 256; lb += (int)blockDim.x) {
        const int lo = (lb & 0xF) >= 8 ? (lb & 0xF) - 16 : (lb & 0xF);
        const int hi = (lb >> 4) >= 8 ? (lb >> 4) - 16 : (lb >> 4);
        c_lut[lb] = make_float2((float)lo, (float)hi);
    }
    __syncthreads();
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int base = warp * R;
    if (base >= n) {
        return;
    }
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* pay[R];
    const __half* sca[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row_off + base + r;
        pay[r] = w + (long long)row * row_bytes;
        sca[r] = reinterpret_cast<const __half*>(w + scale_off) + (long long)row * row_groups;
    }
    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0f;
    }
    for (int c0 = lane * 8; c0 < k; c0 += 256) {
        const int g = c0 >> 6;
        const int t = c0 & 63;
        const int bi = g * 32 + (t >> 1);
        const bool full = (c0 + 8 <= k);
        float xv[8];
        // 注：这里试过改成一次 int4 读 8 个 bf16，实测**反而变慢**
        //（[16384,2560] 372→343、[13312,2560] 352→332 GB/s）——4 行共享 xv[8] 时
        // 寄存器已经紧张，再多一组向量寄存器就掉占用率。保持 8 次标量 load。
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            xv[i] = (c0 + i < k) ? __bfloat162float(x[c0 + i]) : 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            std::uint32_t word;
            if (full) {
                word = *reinterpret_cast<const std::uint32_t*>(pay[r] + bi);
            } else {
                std::uint8_t tmp[4] = {0, 0, 0, 0};
                const int lim = row_bytes - bi;
                for (int z = 0; z < 4 && z < lim; ++z) {
                    tmp[z] = pay[r][bi + z];
                }
                word = (std::uint32_t)tmp[0] | ((std::uint32_t)tmp[1] << 8)
                     | ((std::uint32_t)tmp[2] << 16) | ((std::uint32_t)tmp[3] << 24);
            }
            const float sc = __half2float(sca[r][g]);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int b8 = (int)((word >> (8 * i)) & 0xFFu);
                const float2 c2 = c_lut[b8];
                dot = fmaf(c2.x, xv[2 * i], dot);
                dot = fmaf(c2.y, xv[2 * i + 1], dot);
            }
            // 把 per-group 的 scale 提到内层之外：8 个元素只做 1 次缩放乘，而不是 8 次
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
            if (base + r >= n) {
                break;
            }
            float v = acc[r];
            if (epi == 1) {
                const float t2 = v * a;
                v = t2 / (1.0f + __expf(-t2));
            } else if (epi == 2) {
                const float t2 = v * a;
                v = (1.0f / (1.0f + __expf(-t2))) * b;
            } else if (epi == 3) {
                v = v * a + b;
            }
            y[base + r] = __float2bfloat16(v);
        }
    }
}

// 稠密 Q4G64 → bf16 整块解量化（`t>1` 的批量路径用：把 W4 权重还原成 bf16 再走 GEMM）。
// 这样权重**只需常驻 Q4 一份**，省下的显存正好给专家槽（decode 的另一条腿）。
extern "C" __global__ void lingxi_dequant_q4g64_dense_bf16(const std::uint8_t* __restrict__ w,
                                                           bf16* __restrict__ out,
                                                           int rows,
                                                           int cols,
                                                           int k_pad,
                                                           int row_off,
                                                           long long scale_off) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cols) {
        return;
    }
    const int r = (int)(i / cols);
    const int c = (int)(i % cols);
    const int row = row_off + r;
    const int row_bytes = k_pad >> 1;
    const int row_groups = k_pad >> 6;
    const std::uint8_t* payload = w + (long long)row * row_bytes;
    const __half* scales =
        reinterpret_cast<const __half*>(w + scale_off) + (long long)row * row_groups;
    const int g = c >> 6;
    const int t = c & 63;
    const std::uint8_t byte = payload[g * 32 + (t >> 1)];
    const int nib = (t & 1) ? (byte >> 4) : (byte & 0x0F);
    const int code = nib >= 8 ? nib - 16 : nib;
    out[i] = __float2bfloat16((float)code * __half2float(scales[g]));
}

// **批量版**：一次 launch 解量化 S 个专家（槽区里的等大 blob），输出紧凑 `[S, n, k]`。
// `slots`：设备槽区基址；`slot_ids[s]`：第 s 个专家的槽号；`blob`：槽字节数；
// `seg_off`：该段（gu 或 down）在 blob 内的偏移；`scale_off`：scale 平面相对**该段起点**的偏移。
// 与单体版逐位一致（同一 nibble/二补码/scale 语义），只是把专家维交给 blockIdx.y。
extern "C" __global__ void lingxi_dequant_rowsplit_q4g64_batch(const std::uint8_t* __restrict__ slots,
                                                               const int* __restrict__ slot_ids,
                                                               long long blob,
                                                               long long seg_off,
                                                               bf16* __restrict__ out,
                                                               int S,
                                                               int n,
                                                               int k,
                                                               int k_pad,
                                                               long long scale_off) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (idx >= (long long)n * k || s >= S) {
        return;
    }
    const std::uint8_t* payload = slots + (long long)slot_ids[s] * blob + seg_off;
    const int r = (int)(idx / k);
    const int c = (int)(idx % k);
    const int g = c >> 6;
    const int e = c & 63;
    const std::uint8_t byte = payload[(long long)r * (k_pad / 2) + g * 32 + (e >> 1)];
    const int nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
    const int code = nib >= 8 ? nib - 16 : nib;
    const __half* scales = reinterpret_cast<const __half*>(payload + scale_off);
    const float sc = __half2float(scales[(long long)r * (k_pad / 64) + g]);
    out[((long long)s * n + r) * k + c] = __float2bfloat16((float)code * sc);
}
