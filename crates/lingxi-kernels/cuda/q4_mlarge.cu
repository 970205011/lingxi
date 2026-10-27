// ㊽+5 q4_mlarge —— Q4 稠密投影大 m 摊销核（2026-10-22，第十八轮立项）。
//
// `gemm_bf16_mlarge`（mlarge.cu）的 Q4G64F16S 版：同 tiling（CTA=8 输出行 × 64
// token 行、k-tile 64 进 smem ⇒ x 一遍、权重摊销），权重在核内反量化——
// k-tile 64 恰好 = Q4G64 的一个 scale 组：每 lane 每 tile 读 32 B（8 u32，同址
// 广播）+ 1 个 fp16 scale，nibble→fp16 用 q4_tcp 的魔数技巧
//（nibble^8 → fp16 尾数 → −1032 → ×scale）。每元素 k 升序串行 fp32 累加
//（有损档，判据 PLAN_NEXT；`LINGXI_NO_Q4MLARGE=1` 回退 gemm_q4_skinny）。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace {

constexpr int kNr = 8;
constexpr int kMr = 64;
constexpr int kKt = 64;   // = Q4G64 的 scale 组
constexpr int kPad = 8;
constexpr int kStride = kKt + kPad;

__device__ __forceinline__ float q4ml_epi(float v, int epi, float a, float b) {
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

extern "C" __global__ void __launch_bounds__(256, 4)
lingxi_gemm_q4_mlarge(const std::uint8_t* __restrict__ wq, const bf16* __restrict__ x,
                      bf16* __restrict__ y, int n, int k, int m,
                      int row_off, long long scale_off, int epi, float a, float b) {
    __shared__ bf16 xs[kMr][kStride];
    const int n0 = blockIdx.x * kNr;
    const int m0 = blockIdx.y * kMr;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nrow = n0 + warp;
    const bool ngood = nrow < n;
    const int r0 = m0 + lane * 2;
    const int r1 = r0 + 1;
    const bool g0 = r0 < m;
    const bool g1 = r1 < m;

    // Q4G64 行布局（与 gemm_q4_skinny 同）：每行 groups×32 code B；scale 平面 fp16。
    const int groups = (k + 63) >> 6;   // k 是 64 倍数（host 保证）⇒ = k/64
    const int lrow = ngood ? (row_off + nrow) : row_off;
    const std::uint8_t* crow = wq + (long long)lrow * (groups * 32);
    const __half* srow = reinterpret_cast<const __half*>(wq + scale_off) +
                         (long long)lrow * groups;

    float acc0 = 0.0f, acc1 = 0.0f;
    const half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032.0

    for (int k0 = 0; k0 < k; k0 += kKt) {
        {
            const int sr = tid >> 2;
            const int sc = (tid & 3) * 16;
            const int gr = m0 + sr;
            bf16 tmp[16];
            if (gr < m) {
                const bf16* xr = x + (long long)gr * k + k0 + sc;
#pragma unroll
                for (int v = 0; v < 2; ++v) {
                    const int4 raw = *reinterpret_cast<const int4*>(xr + v * 8);
                    const bf16* src = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
                    for (int j = 0; j < 8; ++j) tmp[v * 8 + j] = src[j];
                }
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j) tmp[j] = __float2bfloat16(0.0f);
            }
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                *reinterpret_cast<int4*>(&xs[sr][sc + v * 8]) =
                    *reinterpret_cast<const int4*>(tmp + v * 8);
            }
        }
        __syncthreads();

        if (ngood) {
            const int g = k0 >> 6;
            const half2 sc2 = __half2half2(srow[g]);
            const std::uint8_t* wp = crow + (long long)g * 32;
            const bf16* xa = &xs[lane * 2][0];
            const bf16* xb = &xs[lane * 2 + 1][0];
            // 8 u32 = 64 个 nibble；每 u32 → 4 个 half2（8 个 k）
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const std::uint32_t word =
                    *reinterpret_cast<const std::uint32_t*>(wp + 4 * w);
                const std::uint32_t p4 = word ^ 0x88888888u;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const std::uint32_t byte = (p4 >> (8 * j)) & 0xffu;
                    const std::uint32_t bits =
                        ((byte & 0x0fu) | ((byte & 0xf0u) << 12)) | 0x64006400u;
                    const half2 h2 = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&bits), bias), sc2);
                    const float2 f = __half22float2(h2);
                    const int kk = w * 8 + j * 2;
                    acc0 = fmaf(f.x, __bfloat162float(xa[kk]), acc0);
                    acc1 = fmaf(f.x, __bfloat162float(xb[kk]), acc1);
                    acc0 = fmaf(f.y, __bfloat162float(xa[kk + 1]), acc0);
                    acc1 = fmaf(f.y, __bfloat162float(xb[kk + 1]), acc1);
                }
            }
        }
        __syncthreads();
    }

    if (ngood) {
        if (g0) y[(long long)r0 * n + nrow] = __float2bfloat16(q4ml_epi(acc0, epi, a, b));
        if (g1) y[(long long)r1 * n + nrow] = __float2bfloat16(q4ml_epi(acc1, epi, a, b));
    }
}

#endif // sm_70
