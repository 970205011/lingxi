// ㊽+3 Q3 专家张量核 GEMM —— prefill 大 tile 版（Volta mma.sync.m8n8k4，QPN 映射）。
//
// 动机（2026-10-22，moe-gemm-bench 证据链）：混档 arena 里 **Q3 是唯一还在走 SIMT
// slotsm 的格式**（每 8 行重读全权重，t≥128 时实测 0.1~6 GB/s），而 q4_tcp 同形状
// 77~104 GB/s（×17~38）。SKIP 剥离实锤 Q3 slotsm 是 prefill MoE GPU 11.4s（51%）
// 的主体。本核把 q4_tcp.cu 的结构平移到 Q3G128F16S：
// * kGroupK = 128（Q3 的 scale 粒度），x 舞台 64 行 × 128 k fp16；
// * 权重行 48 B / 组（128 个 3-bit code），每 lane 每 word 取 3 B（8 个 code）；
// * 3-bit 解码用和 Q4 同构的"魔数 fp16"技巧：code = (w3 ^ 4) - 4 ⇒ code+4 = w3^4，
// fp16 位 = 0x6400 | (w3^4) ⇒ 值 = 1024 + (w3^4)，减 bias 1028.0 即得 code
// （两条 3-bit 拼一个 half2，一次 hsub2 + hmul2(group scale)）。
//
// 数值口径（有损档，与 q4_tcp.cu 相同判据）：权重 code×fp16 group-scale 进 fp16
// 张量核、fp32 累加；x 进舞台时 bf16→fp16 舍入（~2^-11）。判据 = 首 token 相等 +
// 质量抽检；SIMT 档（gemv_q3g128_slotsm_*）保留。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace {

struct Tcp3S {
    static constexpr int kWarps = 8;      // 每 CTA warp 数（2 行 × 4 列）
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kRowsPerCta = 64;  // CTA 的 token 行数（M 轴 tile）
    static constexpr int kColsPerCta = 128; // CTA 的输出行数（N 轴 tile）
    static constexpr int kWarpRows = 32;    // 每 warp 的 token 行数（4 个 mma M-tile）
    static constexpr int kWarpCols = 32;    // 每 warp 的输出行数（QPN quadpair）
    static constexpr int kTiles = kWarpRows / 8;
    static constexpr int kGroupK = 128;     // Q3 的 scale 粒度
    static constexpr int kShPad = 8;        // shared 行填充（半元素），错开 16B 读的 bank
    static constexpr int kShStride = kGroupK + kShPad; // 136
    static constexpr int kRowBytes = 48;    // 每行每组 48 code B（128 × 3-bit）
};

// 同 q4_tcp.cu：一条 mma 一条 k slice（4 个真实 k）。
__device__ __forceinline__ void mma_qpn(float (&d)[8], unsigned a0, unsigned a1,
                                        unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
          "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// 24 位位流（bits [3c, 3c+3)，LSB-first）第 z 个 3-bit code（z=0..7）→ fp16。
// code = (w3 ^ 4) - 4；fp16(1024 + (w3^4)) = 0x6400 | (w3^4)（低 4 位空闲，OR==ADD），
// 调用方统一减 bias 1028.0。
__device__ __forceinline__ __half q3_code_h(std::uint32_t w24, int z) {
    const std::uint32_t w3 = ((w24 >> (3 * z)) & 7u) ^ 4u;
    return __ushort_as_half((unsigned short)(0x6400u | w3));
}

// 网格形态两档（，与 q4_tcp.cu 同构）：旧档 grid.z = ceil(tmax/64) 由 host 发；
// 乐观档 `lingxi_gemv_q3g128_slots_tcp_bf16_dyn` grid = (ceil(n/128), ng_ub, 1)，
// 行块核内按本组 t 循环（免 host 计数）。两档每 (组, 行块) 算术逐字相同（同一 body）。
__device__ __forceinline__ void q3_slots_tcp_body(
    const std::uint8_t* __restrict__ slots, const std::int32_t* __restrict__ slot_ids,
    const std::int32_t* __restrict__ group_off, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int e, int row0,
    int n, int k, int a_stride, int k_pad,
    std::int64_t blob, std::int64_t seg_off, std::int64_t scale_off) {
    using S = Tcp3S;
    const int s0 = group_off[e];
    const int t = group_off[e + 1] - s0;
    if (t <= 0 || row0 >= t) {
        return;
    }
    const int slot = slot_ids[e];
    if (slot < 0) {
        return; // 混档单 launch：他格式组的哨兵槽，跳过
    }
    const int groups = k_pad / S::kGroupK;

    const std::uint8_t* base = slots + static_cast<std::int64_t>(slot) * blob + seg_off;
    const std::uint8_t* codes = base;
    const std::uint16_t* scales = reinterpret_cast<const std::uint16_t*>(base + scale_off);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    // QPN lane 映射（同 q4_tcp.cu）：qp 分 N 四段，r 取段内 8 行。
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
    // CTA 内 warp 排布：2（行）× 4（列）。
    const int wr = warp & 1;
    const int wc = warp >> 1;

    const int colw = static_cast<int>(blockIdx.x) * S::kColsPerCta + wc * S::kWarpCols +
                     qp * 8 + r;
    const bool good = colw < n;
    const int cbase = good ? colw : 0;

    const std::uint8_t* crow = codes + static_cast<std::int64_t>(cbase) * groups * S::kRowBytes;
    const std::uint16_t* srow = scales + static_cast<std::int64_t>(cbase) * groups;

    float c[S::kTiles][8];
#pragma unroll
    for (int tile = 0; tile < S::kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[tile][i] = 0.0f;
        }
    }

    // x 的 k-group 舞台（fp16）：64 行 × 128 k，行填充 8 避 bank 冲突。
    __shared__ __half xs[S::kRowsPerCta][S::kShStride];

    const half2 bias = __half2half2(__ushort_as_half(0x6404)); // 1028.0
    constexpr int kWords = S::kGroupK / 8; // 每组 16 个 8-k word

    for (int g = 0; g < groups; ++g) {
        const int kg = g * S::kGroupK;
        if (kg >= k) {
            continue; // 补齐组：权重为 0，整组跳过
        }
        // ---- 舞台 x：本 CTA 的 64 行 × 本组 128 k，bf16→fp16 一次 CVT ----
        // 每线程 32 个元素：row = tid>>2，列起点 = (tid&3)*32。
        {
            const int srow_i = tid >> 2;          // 0..63
            const int scol = (tid & 3) * 32;      // 0/32/64/96
            const int grow = row0 + srow_i;       // 全局对下标
            __half tmp[32];
            if (grow < t) {
                const __nv_bfloat16* xrow =
                    x + static_cast<std::int64_t>(s0 + grow) * a_stride + kg + scol;
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(xrow + v * 8);
                    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        tmp[v * 8 + j] = __float2half(__bfloat162float(src[j]));
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < 32; ++j) {
                    tmp[j] = __ushort_as_half(0);
                }
            }
            __syncthreads(); // 等全 CTA 读完上一组
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                *reinterpret_cast<uint4*>(&xs[srow_i][scol + v * 8]) =
                    *reinterpret_cast<const uint4*>(tmp + v * 8);
            }
            __syncthreads(); // 本组舞台完成
        }

        // ---- 本组 scale 与本 lane 权重行窗口（48 B = 3×uint4 对齐读，寄存器内抽 24-bit）----
        const half2 sc2 = __half2half2(__ushort_as_half(good ? srow[g] : 0));
        const std::uint8_t* wrow = crow + static_cast<std::int64_t>(g) * S::kRowBytes;
        // 行起始 16B 对齐（48 = 3×16 的组步进 + 基址同 Q4 核的 256 对齐前提）。
        const uint4 cw0 = __ldg(reinterpret_cast<const uint4*>(wrow));
        const uint4 cw1 = __ldg(reinterpret_cast<const uint4*>(wrow + 16));
        const uint4 cw2 = __ldg(reinterpret_cast<const uint4*>(wrow + 32));
        const std::uint32_t w12[12] = {cw0.x, cw0.y, cw0.z, cw0.w, cw1.x, cw1.y,
                                       cw1.z, cw1.w, cw2.x, cw2.y, cw2.z, cw2.w};
        // 字节流 [3w, 3w+3) 的 24-bit 窗口：u32 对齐读 + 跨字拼接（尾部 wi+1 越界守卫，
        // 低 24 位不受影响）。
        auto load24 = [&w12](int byte_off) -> std::uint32_t {
            const int wi = byte_off >> 2;
            const int sh = (byte_off & 3) * 8;
            std::uint32_t v = w12[wi] >> sh;
            if (sh && wi + 1 < 12) {
                v |= w12[wi + 1] << (32 - sh);
            }
            return v & 0xFFFFFFu;
        };

#pragma unroll
        for (int w = 0; w < kWords; ++w) {
            const std::uint32_t w24 = load24(3 * w);
            half2 b[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __half lo = q3_code_h(w24, 2 * j);
                const __half hi = q3_code_h(w24, 2 * j + 1);
                // (1024 + (w3^4)) - 1028 = code；再乘 group scale。
                b[j] = __hmul2(__hsub2(__halves2half2(lo, hi), bias), sc2);
            }
            const unsigned* B = reinterpret_cast<const unsigned*>(b);
            const int lbase = w * 8; // 组内 k 起点
#pragma unroll
            for (int tile = 0; tile < S::kTiles; ++tile) {
                // A 片：本 lane 负责的 token 行 × 8 个 k，一条 uint4 = 4 个 half2。
                const int rb = wr * S::kWarpRows + tile * 8 + r;
                const uint4 av = *reinterpret_cast<const uint4*>(&xs[rb][lbase]);
                const unsigned* A = reinterpret_cast<const unsigned*>(&av);
                mma_qpn(c[tile], A[0], A[1], B[0], B[1]);
                mma_qpn(c[tile], A[2], A[3], B[2], B[3]);
            }
        }
    }

    // ---- 直接写回（无 K 拆分，不需要跨 warp 归约）----
    const int colb = static_cast<int>(blockIdx.x) * S::kColsPerCta + wc * S::kWarpCols +
                     qp * 8;
#pragma unroll
    for (int tile = 0; tile < S::kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            const int grow = row0 + wr * S::kWarpRows + tile * 8 + rr;
            const int gcol = colb + cl;
            if (grow < t && gcol < n) {
                y[static_cast<std::int64_t>(s0 + grow) * n + gcol] =
                    __float2bfloat16(c[tile][i]);
            }
        }
    }
}

} // namespace

extern "C" __global__ void __launch_bounds__(Tcp3S::kThreads, 2)
lingxi_gemv_q3g128_slots_tcp_bf16(
    const std::uint8_t* __restrict__ slots, const std::int32_t* __restrict__ slot_ids,
    const std::int32_t* __restrict__ group_off, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int n, int k, int a_stride, int k_pad,
    std::int64_t blob, std::int64_t seg_off, std::int64_t scale_off) {
    q3_slots_tcp_body(slots, slot_ids, group_off, x, y,
                      static_cast<int>(blockIdx.y),
                      static_cast<int>(blockIdx.z) * Tcp3S::kRowsPerCta,
                      n, k, a_stride, k_pad, blob, seg_off, scale_off);
}

// 乐观发射档（语义与 `lingxi_gemv_q4g64_slots_tcp_bf16_dyn` 完全一致，见其注记）：
// grid = (ceil(n/128), ng_ub, 1)，ng_ub 是 host 上界，尾组靠 group_off 清零早退。
extern "C" __global__ void __launch_bounds__(Tcp3S::kThreads, 2)
lingxi_gemv_q3g128_slots_tcp_bf16_dyn(
    const std::uint8_t* __restrict__ slots, const std::int32_t* __restrict__ slot_ids,
    const std::int32_t* __restrict__ group_off, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int n, int k, int a_stride, int k_pad,
    std::int64_t blob, std::int64_t seg_off, std::int64_t scale_off) {
    const int e = static_cast<int>(blockIdx.y);
    const int t = group_off[e + 1] - group_off[e];
    for (int row0 = 0; row0 < t; row0 += Tcp3S::kRowsPerCta) {
        q3_slots_tcp_body(slots, slot_ids, group_off, x, y, e, row0,
                          n, k, a_stride, k_pad, blob, seg_off, scale_off);
    }
}

#endif // sm_70
