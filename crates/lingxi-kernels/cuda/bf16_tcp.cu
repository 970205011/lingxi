// bf16_tcp —— 稠密 bf16 投影的 Volta MMA 核（2026-10-27， 立项）。
//
// 动机（判决）：attn 投影（qkvz/zproj/out/proj，t=512 合计 ~3.5ms/层 ×48 层）已
// 到**指令发射地板**——SIMT FMA 每线程每指令 1 FMA，每 FMA 搭 ~1.5 个 smem 读；
// mlarge32 宽行重排（x 重读 ÷4）实测零差异 ⇒ 不是流量账。剩下唯一一刀 = 张量核化。
//
// 本核 = q4_tcp_dense.cu 的骨架去反量化、改稠密 bf16 W 行指针：
// CTA = 64 token 行（M 轴 tile，grid.y 分块）× 128 输出行（N 轴 tile，grid.x），
// 每 warp 32 行 × 32 列（QPN quadpair），权重行被 64 token 行摊销，
// x 每 k-tile(64) 整块进 shared 只读一遍（bf16→fp16 一次 CVT，同 q4_tcp 口径）。
// 内层每 k8：lane 读自己 W 行 8 个 bf16 → CVT fp16 喂 mma.m8n8k4（×2 覆盖 k8），
// A 片段直接从 fp16 shared 读 uint4 —— SIMT FMA 的 ~160 issue slot/64 FMA 压到
// ~22（8 mma + 8 CVT + 少量 load）。
//
// 数值口径（有损档）：V100 无 bf16 HMMA、无 fp32
// 输入 HMMA —— 乘数走 fp16（x/W 各一次 bf16→fp16 舍入，尾数 8→10 位是放大不缩，
// |值|>65504 会 inf 的理论风险与 q4_tcp 同），**fp32 累加**、epi fp32、写出 bf16。
// 判据 = 首 token 相等 + spec_bench 接受率 + 文本质量抽检。
// `LINGXI_PROJ_TC=1` 才启用（**默认关**，gpu.rs 与 mlarge 同形状门禁）。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace {

struct TcpS {
    static constexpr int kWarps = 8;        // 每 CTA warp 数（2 行 × 4 列）
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kRowsPerCta = 64;  // CTA 的 token 行数（M 轴 tile）
    static constexpr int kColsPerCta = 128; // CTA 的输出行数（N 轴 tile）
    static constexpr int kWarpRows = 32;    // 每 warp 的 token 行数（4 个 mma M-tile）
    static constexpr int kWarpCols = 32;    // 每 warp 的输出行数（QPN quadpair）
    static constexpr int kTiles = kWarpRows / 8;
    static constexpr int kKt = 64;          // k 分块（x 舞台粒度）
    static constexpr int kShPad = 8;        // shared 行填充（半元素），错开 16B 读的 bank
    static constexpr int kShStride = kKt + kShPad; // 72
    static constexpr int kRowWarps = kRowsPerCta / kWarpRows; // 2
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

__device__ __forceinline__ float tcp_epi(float v, int epi, float a, float b) {
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

} // namespace

// grid = (ceil(n/128), ceil(m/64), 1)，block = 256。
// 每 CTA：token 行 [m0, m0+64) × 输出行 [col0, col0+128)。要求 k % 64 == 0（调用方门禁）。
extern "C" __global__ void __launch_bounds__(TcpS::kThreads, 2)
lingxi_gemm_bf16_tcp(const bf16* __restrict__ w, const bf16* __restrict__ x,
                     bf16* __restrict__ y, int n, int k, int m, int epi,
                     float a, float b) {
    const int m0 = static_cast<int>(blockIdx.y) * TcpS::kRowsPerCta;
    if (m0 >= m) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
    const int wr = warp % TcpS::kRowWarps;
    const int wc = warp / TcpS::kRowWarps;

    const int colw = static_cast<int>(blockIdx.x) * TcpS::kColsPerCta + wc * TcpS::kWarpCols +
                     qp * 8 + r;
    const bool good = colw < n;
    const int cbase = good ? colw : 0;

    const bf16* wrow = w + static_cast<std::int64_t>(cbase) * k;

    float c[TcpS::kTiles][8];
#pragma unroll
    for (int tile = 0; tile < TcpS::kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[tile][i] = 0.0f;
        }
    }

    __shared__ __half xs[TcpS::kRowsPerCta][TcpS::kShStride];

    for (int k0 = 0; k0 < k; k0 += TcpS::kKt) {
        // ---- 协同 stage x：64 行 × 64 k，bf16→fp16 一次 CVT（溢出口径同 q4_tcp）----
        {
            const int sr = tid >> 2;            // 0..63
            const int sc = (tid & 3) * 16;      // 0/16/32/48
            const int grow = m0 + sr;
            __half tmp[16];
            if (grow < m) {
                const bf16* xrow = x + static_cast<std::int64_t>(grow) * k + k0 + sc;
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
            __syncthreads(); // 等全 CTA 读完上一 tile
            *reinterpret_cast<uint4*>(&xs[sr][sc]) = *reinterpret_cast<const uint4*>(tmp);
            *reinterpret_cast<uint4*>(&xs[sr][sc + 8]) =
                *reinterpret_cast<const uint4*>(tmp + 8);
            __syncthreads(); // 本 tile 舞台完成
        }

        // ---- 本 tile 64 k：8 个 8-k word，每 word 2 条 mma × kTiles ----
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const int lbase = w * 8; // tile 内 k 起点
            // B 片段：本 lane 的 W 行 8 个 bf16 → fp16（k8 降序无关，逐元素 CVT）
            const uint4 wv = __ldg(reinterpret_cast<const uint4*>(wrow + k0 + lbase));
            const bf16* wsrc = reinterpret_cast<const bf16*>(&wv);
            __half2 b[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                b[j] = __floats2half2_rn(__bfloat162float(wsrc[2 * j]),
                                         __bfloat162float(wsrc[2 * j + 1]));
            }
            const unsigned* B = reinterpret_cast<const unsigned*>(b);
#pragma unroll
            for (int tile = 0; tile < TcpS::kTiles; ++tile) {
                const int rb = wr * TcpS::kWarpRows + tile * 8 + r;
                const uint4 av = *reinterpret_cast<const uint4*>(&xs[rb][lbase]);
                const unsigned* A = reinterpret_cast<const unsigned*>(&av);
                mma_qpn(c[tile], A[0], A[1], B[0], B[1]);
                mma_qpn(c[tile], A[2], A[3], B[2], B[3]);
            }
        }
    }

    // ---- 写回（epi 融合，语义同 skinny 族）----
    const int colb = static_cast<int>(blockIdx.x) * TcpS::kColsPerCta + wc * TcpS::kWarpCols +
                     qp * 8;
#pragma unroll
    for (int tile = 0; tile < TcpS::kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            const int grow = m0 + wr * TcpS::kWarpRows + tile * 8 + rr;
            const int gcol = colb + cl;
            if (grow < m && gcol < n) {
                y[static_cast<std::int64_t>(grow) * n + gcol] =
                    __float2bfloat16(tcp_epi(c[tile][i], epi, a, b));
            }
        }
    }
}

#endif // sm_70
