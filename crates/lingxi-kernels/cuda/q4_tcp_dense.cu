// ㊽+6 q4_tcp_dense —— Q4 稠密投影的"TCP MMA + 大 m 摊销"核（2026-10-22，第十九轮立项）。
//
// 动机（接 q4_mlarge 结案教训）：attn 段 Q4 投影（GDN qkvz/o、QSA qkgv/o）t=512 实测
// 10.7ms/层。q4_mlarge（标量 FMA + 核内反量化）实测回退 +21%——dequant ALU 链
// （~80 ALU/8 MAC）没被摊销。**x 重读账 ≠ 胜利条件，MMA 摊销才是**。
//
// 本核 = q4_tcp.cu 的 body（Volta mma.m8n8k4 QPN 映射 + nibble→fp16 魔数解码 +
// x bf16→fp16 一次 CVT 进 shared）去 slots/group_off、改稠密 W 行指针 + row_off：
// CTA = 64 token 行（M 轴 tile，grid.y 分块）× 128 输出行（N 轴 tile，grid.x），
// 每 warp 32 行 × 32 列（QPN quadpair），权重行被 64 token 行摊销，
// x 每 k-group 整块进 shared 只读一遍（vs gemm_q4_skinny 的 n/8 遍重读）。
// k-tile 64 = Q4G64 的一个 scale 组，核内反量化后直接把 fp16 B 片段喂 MMA——
// 反量化 ALU 被 MMA 摊销（这正是 q4_mlarge 缺的那一环）。
//
// 数值口径与 q4_tcp 完全一致（有损档，PLAN_NEXT）：权重 nibble×fp16 scale 进
// fp16 张量核、fp32 累加；x 进 shared 时 bf16→fp16 舍入。判据 = 首 token 相等 +
// 质量抽检。`LINGXI_NO_Q4TCPDENSE=1` 回退 gemm_q4_skinny。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace {

struct TcdS {
    static constexpr int kWarps = 8;      // 每 CTA warp 数（2 行 × 4 列）
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kRowsPerCta = 64;  // CTA 的 token 行数（M 轴 tile）
    static constexpr int kColsPerCta = 128; // CTA 的输出行数（N 轴 tile）
    static constexpr int kWarpRows = 32;    // 每 warp 的 token 行数（4 个 mma M-tile）
    static constexpr int kWarpCols = 32;    // 每 warp 的输出行数（QPN quadpair）
    static constexpr int kTiles = kWarpRows / 8;
    static constexpr int kGroupK = 64;
    static constexpr int kShPad = 8;      // shared 行填充（半元素），错开 16B 读的 bank
    static constexpr int kShStride = kGroupK + kShPad; // 72
    static constexpr int kRowWarps = kRowsPerCta / kWarpRows; // 2
};

// M≤8 变体（2026-10-23）：verify 批 t=6 时 M64 的 tile 白算 8 倍（实测比 skinny 慢
// 2.4×）。这里把 M 轴 tile 压到 8（= mma 的最小 M），CTA = 4 warp × 32 列 = 128 列，
// 权重仍只读一遍、dequant B 片段照样喂 MMA，仅去掉 padding 的工作量。
struct TcdM8 {
    static constexpr int kRowsPerCta = 8;   // M 轴 tile = 8（一条 mma 的高度）
    static constexpr int kWarpRows = 8;
    static constexpr int kWarpsCols = 4;    // 每 CTA 4 个列 warp
    static constexpr int kRowWarps = 1;
    static constexpr int kWarps = kRowWarps * kWarpsCols;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kColsPerCta = kWarpsCols * 32;
    static constexpr int kWarpCols = 32;
    static constexpr int kTiles = kWarpRows / 8;
    static constexpr int kGroupK = 64;
    static constexpr int kShPad = 8;
    static constexpr int kShStride = kGroupK + kShPad;
};

__device__ __forceinline__ void mma_qpn(float (&d)[8], unsigned a0, unsigned a1,
                                        unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
          "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float tcd_epi(float v, int epi, float a, float b) {
    if (epi == 1) {
        const float t2 = v * a;
        return t2 / (1.0f + __expf(-t2));
    }
    if (epi == 2) {
        const float t2 = v * a;
        return (1.0f / (1.0f + __expf(-t2))) * b;
    }
    if (epi == 3) {
        return v * a + b;
    }
    return v;
}

// grid = (ceil(n/kColsPerCta), ceil(m/kRowsPerCta), 1)，block = S::kThreads。
// 每 CTA：token 行 [m0, m0+kRowsPerCta) × 输出行 [col0, col0+kColsPerCta)。
template <int kBlk, typename S>
__device__ __forceinline__ void q4_tcp_dense_body(
    const std::uint8_t* __restrict__ wq, const bf16* __restrict__ x,
    bf16* __restrict__ y, int n, int k, int k_pad, int m,
    int row_off, std::int64_t scale_off, int epi, float ea, float eb) {
    const int m0 = static_cast<int>(blockIdx.y) * S::kRowsPerCta;
    if (m0 >= m) {
        return;
    }
    const int groups = k_pad / S::kGroupK;

    // 稠密 Q4G64 行布局（与 gemm_q4_skinny 同）：每行 groups×32 code B；scale 平面 fp16。
    const std::uint8_t* codes = wq + static_cast<std::int64_t>(row_off) * groups * 32;
    const std::uint16_t* scales =
        reinterpret_cast<const std::uint16_t*>(wq + scale_off) +
        static_cast<std::int64_t>(row_off) * groups;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
    const int wr = warp % S::kRowWarps;
    const int wc = warp / S::kRowWarps;

    const int colw = static_cast<int>(blockIdx.x) * S::kColsPerCta + wc * S::kWarpCols +
                     qp * 8 + r;
    const bool good = colw < n;
    const int cbase = good ? colw : 0;

    const std::uint8_t* crow =
        codes + static_cast<std::int64_t>(cbase) * groups * 32; // 每行每组 32 code B
    const std::uint16_t* srow = scales + static_cast<std::int64_t>(cbase) * groups;

    float c[S::kTiles][8];
#pragma unroll
    for (int tile = 0; tile < S::kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[tile][i] = 0.0f;
        }
    }

    __shared__ __half xs[S::kRowsPerCta][S::kShStride];

    const half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032.0
    uint4 cw[2 * kBlk];
    std::uint16_t scv[kBlk];

    for (int gb = 0; gb < groups; gb += kBlk) {
        const int blk = min(kBlk, groups - gb);
        const int last = groups - 1;
#pragma unroll
        for (int q = 0; q < kBlk; ++q) {
            const int g = min(gb + q, last);
            const std::uint8_t* p = crow + static_cast<std::int64_t>(g) * 32;
            cw[2 * q] = __ldg(reinterpret_cast<const uint4*>(p));
            cw[2 * q + 1] = __ldg(reinterpret_cast<const uint4*>(p + 16));
            scv[q] = good ? srow[g] : static_cast<std::uint16_t>(0);
        }
        for (int q = 0; q < blk; ++q) {
            const int g = gb + q;
            const int kg = g * S::kGroupK;
            if (kg >= k) {
                continue; // 补齐组：权重为 0，整组跳过
            }
            // ---- 舞台 x：本 CTA 的 kRowsPerCta 行 × 本组 64 k，bf16→fp16 一次 CVT ----
            {
                const int srow_i = tid >> 2;          // 0..kRowsPerCta-1（M≤8 时只有前 32 线程）
                const int scol = (tid & 3) * 16;      // 0/16/32/48
                const int grow = m0 + srow_i;         // 全局 token 行
                __half tmp[16];
                if (grow < m && srow_i < S::kRowsPerCta) {
                    const bf16* xrow =
                        x + static_cast<std::int64_t>(grow) * k + kg + scol;
                    const uint4 raw0 = *reinterpret_cast<const uint4*>(xrow);
                    const uint4 raw1 = *reinterpret_cast<const uint4*>(xrow + 8);
                    const bf16* src0 = reinterpret_cast<const bf16*>(&raw0);
                    const bf16* src1 = reinterpret_cast<const bf16*>(&raw1);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        tmp[j] = __float2half(__bfloat162float(src0[j]));
                        tmp[8 + j] = __float2half(__bfloat162float(src1[j]));
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        tmp[j] = __ushort_as_half(0);
                    }
                }
                __syncthreads(); // 等全 CTA 读完上一组
                if (srow_i < S::kRowsPerCta) { // M≤8 变体只有前 32 线程有对应行
                    *reinterpret_cast<uint4*>(&xs[srow_i][scol]) =
                        *reinterpret_cast<const uint4*>(tmp);
                    *reinterpret_cast<uint4*>(&xs[srow_i][scol + 8]) =
                        *reinterpret_cast<const uint4*>(tmp + 8);
                }
                __syncthreads(); // 本组舞台完成
            }

            // ---- 本组 64 k：8 个 8-k word，每 word 2 条 mma × kTiles ----
            const half2 sc2 = __half2half2(__ushort_as_half(scv[q]));
            const std::uint32_t words[8] = {cw[2 * q].x, cw[2 * q].y, cw[2 * q].z, cw[2 * q].w,
                                            cw[2 * q + 1].x, cw[2 * q + 1].y,
                                            cw[2 * q + 1].z, cw[2 * q + 1].w};
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const std::uint32_t p4 = words[w] ^ 0x88888888u;
                half2 b[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const std::uint32_t byte = (p4 >> (8 * j)) & 0xffu;
                    std::uint32_t bits = ((byte & 0x0fu) | ((byte & 0xf0u) << 12)) | 0x64006400u;
                    b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
                }
                const unsigned* B = reinterpret_cast<const unsigned*>(b);
                const int lbase = w * 8; // 组内 k 起点
#pragma unroll
                for (int tile = 0; tile < S::kTiles; ++tile) {
                    const int rb = wr * S::kWarpRows + tile * 8 + r;
                    const uint4 av = *reinterpret_cast<const uint4*>(&xs[rb][lbase]);
                    const unsigned* A = reinterpret_cast<const unsigned*>(&av);
                    mma_qpn(c[tile], A[0], A[1], B[0], B[1]);
                    mma_qpn(c[tile], A[2], A[3], B[2], B[3]);
                }
            }
        }
    }

    // ---- 写回（epi 融合，语义同 skinny 族）----
    const int colb = static_cast<int>(blockIdx.x) * S::kColsPerCta + wc * S::kWarpCols +
                     qp * 8;
#pragma unroll
    for (int tile = 0; tile < S::kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            const int grow = m0 + wr * S::kWarpRows + tile * 8 + rr;
            const int gcol = colb + cl;
            if (grow < m && gcol < n) {
                y[static_cast<std::int64_t>(grow) * n + gcol] =
                    __float2bfloat16(tcd_epi(c[tile][i], epi, ea, eb));
            }
        }
    }
}

} // namespace

extern "C" __global__ void __launch_bounds__(TcdS::kThreads, 2)
lingxi_gemm_q4_tcp_dense(
    const std::uint8_t* __restrict__ wq, const bf16* __restrict__ x,
    bf16* __restrict__ y, int n, int k, int k_pad, int m,
    int row_off, std::int64_t scale_off, int epi, float a, float b) {
    q4_tcp_dense_body<4, TcdS>(wq, x, y, n, k, k_pad, m, row_off, scale_off, epi, a, b);
}

// M≤8 变体：verify 批（t=6）专用，M 轴 tile = 8。grid = (ceil(n/128), 1, 1)。
extern "C" __global__ void __launch_bounds__(TcdM8::kThreads, 4)
lingxi_gemm_q4_tcp_dense_m8(
    const std::uint8_t* __restrict__ wq, const bf16* __restrict__ x,
    bf16* __restrict__ y, int n, int k, int k_pad, int m,
    int row_off, std::int64_t scale_off, int epi, float a, float b) {
    q4_tcp_dense_body<4, TcdM8>(wq, x, y, n, k, k_pad, m, row_off, scale_off, epi, a, b);
}

#endif // sm_70
