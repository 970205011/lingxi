// ── 2026-10-24 MoE **router 投影**专用：一行一 block、m ≤ 6 **编译期特化**的 bf16 瘦长 GEMM ──
//
// 动机（`tests/t20_route_chain_bw.rs`，同进程背靠背 best_of3）：
// `lingxi_gemm_skinny_row_bf16`(n=512,k=2560,m=6) **16.0 µs（164 GB/s）**，而同 grid/block、
// 同地址流的**纯读**只要 **4.7 µs（555 GB/s）** ⇒ 11 µs 是取数之外的东西（CVT/FFMA 发射）。
// 旧核内层对**每个 token 行**都重新 `__bfloat162float(wp[z])`（8 次 w 转换 × 6 行 = 48 次），
// 且行偏移 `(r0+s)*k` 是运行期量、每轮重建 6 个 64 位地址。
//
// 本核只做三件**不改算式**的事：
// 1. m 编译期特化到 6（`s >= mt` 的运行期截断仍在，m ≤ 6 都正确）；
// 2. w 的 8 个元素**每轮只转一次**（`__bfloat1622float2`，一次两个；bf16→f32 精确）；
// 3. 行偏移 `s*k` 变编译期常量。
// ⇒ 每个 8 元素组的指令数从 ~1+6 load + 96 cvt + 48 FFMA 降到 7 load + 28 cvt + 48 FFMA。
//
// **逐位相同**的依据：每线程的列划分（`c0 = tid*8; c0 += blockDim*8`）、8 元素组内
// `d += w[z]*x[z]` 的加法次序、`acc[s] += d`、两级 warp 归约树（`shfl_down` 16..1 →
// `sred[warp][s]` → warp0 再 `shfl_down`）、以及 epi 全部与 `lingxi_gemm_skinny_row_bf16`
// 逐字同构；bf16→f32 是精确转换（`__bfloat1622float2` 与逐个 `__bfloat162float` 同值）。
// 判据：`tests/t22_router_m6_bitid.rs`（m=2..6 × 多组随机 w/x，logits 逐位比）。

#include <cuda_bf16.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ void m6_cvt8(const int4& v, float* f) {
    const bf16* p = reinterpret_cast<const bf16*>(&v);
#pragma unroll
    for (int z2 = 0; z2 < 4; ++z2) {
        const float2 t = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(p)[z2]);
        f[2 * z2] = t.x;
        f[2 * z2 + 1] = t.y;
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) lingxi_gemm_skinny_row_m6_bf16(
    const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y,
    int n, int k, int m, int epi, float a, float b) {
    const int row = blockIdx.x;
    if (row >= n || m <= 0 || m > 6) {
        return;
    }
    const int mt = m;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc[6];
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        acc[s] = 0.0f;
    }
    int c0 = tid * 8;
    for (; c0 + 8 <= k; c0 += (int)blockDim.x * 8) {
        const int4 wv = *reinterpret_cast<const int4*>(wr + c0);
        float wf[8];
        m6_cvt8(wv, wf);
#pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s >= mt) {
                break;
            }
            const int4 xv = *reinterpret_cast<const int4*>(x + (size_t)s * (size_t)k + c0);
            float xf[8];
            m6_cvt8(xv, xf);
            float d = 0.0f;
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                d += wf[z] * xf[z];
            }
            acc[s] += d;
        }
    }
    for (; c0 < k; c0 += (int)blockDim.x * 8) {
        for (int i = 0; i < 8 && c0 + i < k; ++i) {
            const float wf = __bfloat162float(wr[c0 + i]);
#pragma unroll
            for (int s = 0; s < 6; ++s) {
                if (s >= mt) {
                    break;
                }
                acc[s] += wf * __bfloat162float(x[(size_t)s * (size_t)k + c0 + i]);
            }
        }
    }
    __shared__ float sred[32][6];
#pragma unroll
    for (int s = 0; s < 6; ++s) {
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
                float r = v;
                if (epi == 1) {
                    const float t2 = r * a;
                    r = t2 / (1.0f + __expf(-t2));
                } else if (epi == 2) {
                    const float t2 = r * a;
                    r = (1.0f / (1.0f + __expf(-t2))) * b;
                } else if (epi == 3) {
                    r = r * a + b;
                }
                y[(size_t)s * (size_t)n + row] = __float2bfloat16(r);
            }
        }
    }
}
