// ㊽+4 gemm_bf16_mlarge —— t>1 稠密投影大 m 摊销核（2026-10-22，第十六/十七轮立项）。
//
// 动机：attn 段的 qkvz/o/z 投影在 t=512 实测 14.5ms/层（~2-3 GB/s 有效带宽）。
// 根因（第十六轮实测账）：`gemm_skinny_row_bf16` 的 CTA = 1 输出行 × 8 token 行，
// x 被重读 n/8 遍（out_proj：2048/8=256 遍 ⇒ 5.2GB L2 流量/层）。m 族 grid.y 也不救
// （每 warp 串行 k 循环、latency-bound 不摊销）。P15-⑲ 否决的 row_multi 是 m=8 口径，
// m=512 regime 不同（收益大 32 倍）⇒ 本核按"一次 pass 摊销全部 m"重造：
//
// CTA = 8 输出行（1 warp/行）× 64 token 行；k 分块 64。
// * x 的 k-tile 协同进 shared（64×64×2B = 8.5KB），被 8 个输出行摊销 ⇒
// x 流量 = m×k×2 × (n/8)/(n/8) = **一遍**（vs row 核 n/8 遍）；
// * 权重每 warp 读自己的行、一遍（lane 同址广播，1 次事务）；
// * 每 lane 2 个独立 acc 链（行 2l / 2l+1），k-tile 内 64 步、tile 间顺序衔接
// ⇒ 每元素 k 升序串行累加（与 host 参考同序；相对当前 row 核是**有损档**
// （归约方式不同），判据 = 首 token 相等 + 质量抽检（PLAN_NEXT）。
//
// grid = (ceil(n/8), ceil(m/64))，block = 256（8 warp）。epi 在写出时融合（同 skinny 族语义）。

#include <cuda_bf16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace {

constexpr int kNr = 8;    // 每 CTA 输出行数（1 warp/行）
constexpr int kMr = 64;   // 每 CTA token 行数
constexpr int kKt = 64;   // k 分块
constexpr int kPad = 8;   // smem 行填充（bf16 元素），错开 bank
constexpr int kStride = kKt + kPad;

__device__ __forceinline__ float mlarge_epi(float v, int epi, float a, float b) {
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
lingxi_gemm_bf16_mlarge(const bf16* __restrict__ w, const bf16* __restrict__ x,
                        bf16* __restrict__ y, int n, int k, int m, int epi,
                        float a, float b) {
    __shared__ bf16 xs[kMr][kStride];
    const int n0 = blockIdx.x * kNr;            // 本 CTA 的输出行起点
    const int m0 = blockIdx.y * kMr;            // 本 CTA 的 token 行起点
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;                  // 0..7 → 输出行 n0+warp
    const int nrow = n0 + warp;                 // 全局输出行
    const bool ngood = nrow < n;

    // 每 lane 负责 token 行 2l / 2l+1（64 行 / 32 lane）
    const int r0 = m0 + lane * 2;
    const int r1 = r0 + 1;
    const bool g0 = r0 < m;
    const bool g1 = r1 < m;

    float acc0 = 0.0f, acc1 = 0.0f;
    const bf16* wrow = w + (std::int64_t)(ngood ? nrow : 0) * k;

    for (int k0 = 0; k0 < k; k0 += kKt) {
        // ---- 协同 stage x：64 行 × 64 k（host 保证 k % 64 == 0，无尾部）----
        {
            // 256 线程 × 16 元素 = 4096 = 64×64
            const int sr = tid >> 2;            // 0..63
            const int sc = (tid & 3) * 16;      // 0/16/32/48
            const int gr = m0 + sr;
            bf16 tmp[16];
            if (gr < m) {
                const bf16* xr = x + (std::int64_t)gr * k + k0 + sc;
#pragma unroll
                for (int v = 0; v < 2; ++v) {
                    const int4 raw = *reinterpret_cast<const int4*>(xr + v * 8);
                    const bf16* src = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        tmp[v * 8 + j] = src[j];
                    }
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

        // ---- 本 k-tile：lane 内 64 步，W 同址广播 + smem x ----
        if (ngood) {
            const bf16* wp = wrow + k0;
            const bf16* xa = &xs[lane * 2][0];
            const bf16* xb = &xs[lane * 2 + 1][0];
#pragma unroll 8
            for (int kk = 0; kk < kKt; ++kk) {
                const float wv = __bfloat162float(wp[kk]);
                acc0 = fmaf(wv, __bfloat162float(xa[kk]), acc0);
                acc1 = fmaf(wv, __bfloat162float(xb[kk]), acc1);
            }
        }
        __syncthreads();
    }

    // ---- 写出（epi 融合，语义同 skinny 族）----
    if (ngood) {
        if (g0) y[(std::int64_t)r0 * n + nrow] = __float2bfloat16(mlarge_epi(acc0, epi, a, b));
        if (g1) y[(std::int64_t)r1 * n + nrow] = __float2bfloat16(mlarge_epi(acc1, epi, a, b));
    }
}

#endif // sm_70
