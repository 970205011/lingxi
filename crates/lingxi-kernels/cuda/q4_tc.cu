// ㊼ Q4 专家张量核 GEMM（Volta mma.sync.m8n8k4，移植自本地 ninfer 的
// `src/ops/linear/q4/q4_volta_qpn_gemm.cuh`，作者自有代码）。
//
// 为什么：我们的 Q4 专家段一直是 SIMT GEMV（每权重 4bit 解包 + FP32 FMA），真实形状下
// 有效带宽只有 ~180-208 GB/s（V100 900 的 20%），且**张量核利用率为 0**。V100 的张量核
// 指令是 `mma.sync.m8n8k4`（M=8），⇒ **T=8 就能用满**（不需要 Ampere 的 m16n8k8）。
//
// QPN 映射（quadpair-split-N）：一条 mma 是四个独立的 8x8x4，四个 quadpair 分 N 不分 M，
// warp tile = 8(T) x 32(N)。T=8 时 A 行利用率 100%（旧的 32x8 映射在 T=8 只用到 25%）。
//
// 三件"装载"事实（照抄 ninfer，别重新推导）：
//  1. **我们的 Q4 存储顺序天然就是 B-fragment 顺序**：一个 code byte 解成 half2(k, k+1)，
//     `((b & 0x0f) | ((b & 0xf0) << 12)) | 0x64006400` 再减 1032 就得到 (n^8)-8 —— 无需预打包。
//  2. 一个 fp16 group scale 正好覆盖 64 个 k = 16 个 mma slice。
//  3. 权重读是 DRAM 主流量且**完全不合并**（32 lane 读 32 行）⇒ 每个 lane 一次读满 128B
//     （kBlk=4 组），把 2*kBlk 个 load 放在飞。
//
// 与我们在 SIMT 路径的数值差异：这里是 fp16(权重×group scale) 进 fp16 张量核、fp32 累加；
// SIMT 是 int4 精确值 × f32 scale。差 ~2^-11 相对（远小于 bf16 输出舍入 2^-8）⇒
// **有损档**，必须 env 门控 + 48token/质量双验证。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace {

struct QpnS {
    static constexpr int kWarps = 4;
    static constexpr int kColsPerCta = 32; // 每个 CTA 的输出行（mma 的 N 轴）
    static constexpr int kRowsPerTile = 8; // 每个 A tile 的 token 数（mma 的 M 轴）
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kGroupK = 64;
    static constexpr int kCodeB = 32;   // 每组 code 字节
    static constexpr int kScaleB = 2;   // 每组 scale 字节（f16）
};

// 一条 mma 一个 k slice（4 个真实 k）。
__device__ __forceinline__ void mma_qpn(float (&d)[8], unsigned a0, unsigned a1,
                                        unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
          "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// slot 版：一个 CTA 服务**一个专家**（blockIdx.y），输出该专家命中的 t 行（t = 组大小）。
// grid = (ceil(n/32), n_experts)，block = 128。
template <int kTiles, int kBlk>
__device__ __forceinline__ void q4_slots_tc_body(
    const std::uint8_t* __restrict__ slots, const std::int32_t* __restrict__ slot_ids,
    const std::int32_t* __restrict__ group_off, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int n, int k, int a_stride, int k_pad,
    std::int64_t blob, std::int64_t seg_off, std::int64_t scale_off) {
    using S = QpnS;
    const int e = static_cast<int>(blockIdx.y);
    const int s0 = group_off[e];
    const int t = group_off[e + 1] - s0;
    if (t <= 0) {
        return;
    }
    const int slot = slot_ids[e];
    const int groups = k_pad / S::kGroupK; // 组数按**补齐后**的 k 算
    const int kpad_groups = (k + S::kGroupK - 1) / S::kGroupK;
    (void)kpad_groups;

    const std::uint8_t* base = slots + static_cast<std::int64_t>(slot) * blob + seg_off;
    const std::uint8_t* codes = base;
    const std::uint16_t* scales = reinterpret_cast<const std::uint16_t*>(base + scale_off);

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const bool good = col < n;
    const int cbase = good ? col : 0;

    // 四个 warp 按**整组**切 K。
    const int gq = groups / S::kWarps;
    const int g0 = warp * gq;
    const int gend = (warp == S::kWarps - 1) ? groups : g0 + gq;

    const std::uint8_t* crow =
        codes + static_cast<std::int64_t>(cbase) * groups * S::kCodeB;
    const std::uint16_t* srow = reinterpret_cast<const std::uint16_t*>(
        reinterpret_cast<const std::uint8_t*>(scales) +
        static_cast<std::int64_t>(cbase) * groups * S::kScaleB);

    float c[kTiles][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[tile][i] = 0.0f;
        }
    }

    const half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032.0
    uint4 cw[2 * kBlk];
    std::uint16_t scv[kBlk];

    for (int gb = g0; gb < gend; gb += kBlk) {
        const int blk = min(kBlk, gend - gb);
        const int last = gend - 1;
#pragma unroll
        for (int q = 0; q < kBlk; ++q) {
            const int g = min(gb + q, last);
            const std::uint8_t* p = crow + static_cast<std::int64_t>(g) * S::kCodeB;
            cw[2 * q] = __ldg(reinterpret_cast<const uint4*>(p));
            cw[2 * q + 1] = __ldg(reinterpret_cast<const uint4*>(p + 16));
            scv[q] = good ? srow[g] : static_cast<std::uint16_t>(0);
        }
        for (int q = 0; q < blk; ++q) {
            const int g = gb + q;
            const half2 sc2 = __half2half2(__ushort_as_half(scv[q]));
            const std::uint32_t words[8] = {cw[2 * q].x, cw[2 * q].y, cw[2 * q].z, cw[2 * q].w,
                                            cw[2 * q + 1].x, cw[2 * q + 1].y, cw[2 * q + 1].z,
                                            cw[2 * q + 1].w};
            const int kg = g * S::kGroupK;
            // 补齐区（k_pad > k）：整组跳过（权重为 0，且激活不保证有效）。
            if (kg >= k) {
                continue;
            }
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
                const int kbase = kg + w * 8;
#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    const int row = tile * S::kRowsPerTile + r;
                    half2 a[4];
                    if (row < t) {
                        const __nv_bfloat16* xrow =
                            x + static_cast<std::int64_t>(s0 + row) * a_stride + kbase;
                        const uint4 raw = *reinterpret_cast<const uint4*>(xrow);
                        const auto* src = reinterpret_cast<const __nv_bfloat16*>(&raw);
                        __half tmp[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            tmp[j] = __float2half(__bfloat162float(src[j]));
                        }
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a[j] = *reinterpret_cast<const half2*>(tmp + 2 * j);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a[j] = __half2half2(__ushort_as_half(0));
                        }
                    }
                    const unsigned* A = reinterpret_cast<const unsigned*>(a);
                    mma_qpn(c[tile], A[0], A[1], B[0], B[1]);
                    mma_qpn(c[tile], A[2], A[3], B[2], B[3]);
                }
            }
        }
    }

    __shared__ float cs[QpnS::kWarps][kTiles * QpnS::kRowsPerTile * QpnS::kColsPerCta];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(tile * QpnS::kRowsPerTile + row) * QpnS::kColsPerCta + qp * 8 + cl] =
                c[tile][i];
        }
    }
    __syncthreads();

    constexpr int kOut = kTiles * QpnS::kRowsPerTile * QpnS::kColsPerCta;
    for (int idx = static_cast<int>(threadIdx.x); idx < kOut; idx += QpnS::kThreads) {
        const int row = idx / QpnS::kColsPerCta;
        const int cl = idx % QpnS::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * QpnS::kColsPerCta + cl;
        if (row < t && ocol < n) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < QpnS::kWarps; ++w) {
                v += cs[w][idx];
            }
            y[static_cast<std::int64_t>(s0 + row) * n + ocol] = __float2bfloat16(v);
        }
    }
}

} // namespace

extern "C" __global__ void __launch_bounds__(QpnS::kThreads, 8)
lingxi_gemv_q4g64_slots_tc_bf16(
    const std::uint8_t* __restrict__ slots, const std::int32_t* __restrict__ slot_ids,
    const std::int32_t* __restrict__ group_off, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int n, int k, int a_stride, int k_pad,
    std::int64_t blob, std::int64_t seg_off, std::int64_t scale_off) {
    q4_slots_tc_body<1, 4>(slots, slot_ids, group_off, x, y, n, k, a_stride, k_pad, blob,
                           seg_off, scale_off);
}

extern "C" __global__ void __launch_bounds__(QpnS::kThreads, 8)
lingxi_gemv_q4g64_slots_tc2_bf16(
    const std::uint8_t* __restrict__ slots, const std::int32_t* __restrict__ slot_ids,
    const std::int32_t* __restrict__ group_off, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int n, int k, int a_stride, int k_pad,
    std::int64_t blob, std::int64_t seg_off, std::int64_t scale_off) {
    q4_slots_tc_body<2, 4>(slots, slot_ids, group_off, x, y, n, k, a_stride, k_pad, blob,
                           seg_off, scale_off);
}

#endif // sm_70
