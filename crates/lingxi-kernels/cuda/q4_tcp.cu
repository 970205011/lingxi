// ㊽ Q4 专家张量核 GEMM —— **prefill 大 tile 版**（Volta mma.sync.m8n8k4，QPN 映射）。
//
// 为什么（2026-10-19，PLAN_NEXT 拍板：判据放开为「首 token 相等 + 质量抽检」）：
// 旧的 kTiles=4 实验（q4_tc.cu，实测 +8% 更慢被否决）死在三个地方——
// 1. CTA 只覆盖 32 行（M 轴不分块），t 上百的行组要么丢行要么每组一次 launch；
// 2. A 片（x）每个 mma 都从 global 现读 bf16 + 逐元素 CVT fp16，CVT 比 MMA 还多；
// 3. 权重（DRAM 主流量）不做行摊销：一 warp 一个输出行，同一批 x 被反复重读。
// 本核的对应修法：
// 1. **M 轴分块进 grid.z**（每块 64 行），任意大的组一次 launch 全覆盖；
// 2. x 按 k-group（64）**整块进 shared 并一次性 CVT 成 fp16**（每专家行摊 n/128 次，
// CVT 总量与 SIMT 预转档同量级），A 片从 shared 按 uint4 取、零冲突；
// 3. 每 warp 算 32 行 × 32 列（QPN quadpair），CTA = 2×4 warp = **64 行 × 128 列**，
// 权重行的读取被 64 个 token 行摊销（权重重读因子 = t/64）。
//
// 数值口径与 q4_tc.cu 一致（有损档）：权重 nibble × fp16 group-scale 进 fp16 张量核、
// fp32 累加；x 在进 shared 时 bf16→fp16 舍入（~2^-11 相对，远小于 Q4 权重本身的 2^-4
// 量化误差）。判据 = 首 token 相等 + 质量抽检，env `LINGXI_MOE_TC=1` 门控，SIMT 档保留。
//
// 复用 q4_tc.cu 的三件"装载"事实（QPN 映射 / nibble→fp16 位技巧 / 128B 满读），不重推导。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace {

struct TcpS {
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
};

// 同 q4_tc.cu：一条 mma 一条 k slice（4 个真实 k）。
__device__ __forceinline__ void mma_qpn(float (&d)[8], unsigned a0, unsigned a1,
                                        unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
          "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// 网格形态两档：
// 旧档 `lingxi_gemv_q4g64_slots_tcp_bf16`：grid = (ceil(n/128), n_experts, ceil(tmax/64))，
// 每 CTA 本专家（blockIdx.y）的 [row0, row0+64) 行 × [col0, col0+128) 列；
// 乐观档 `lingxi_gemv_q4g64_slots_tcp_bf16_dyn`：grid = (ceil(n/128), ng_ub, 1)，行块在
// 核内按**本组大小 t** 循环（免 host 计数 ⇒ 可在 pair 建表核完成前发射）。
// 两档每 (组, 行块) 的算术逐字相同（同一 body），只是行块的驱动方式不同。
template <int kBlk>
__device__ __forceinline__ void q4_slots_tcp_body(
    const std::uint8_t* __restrict__ slots, const std::int32_t* __restrict__ slot_ids,
    const std::int32_t* __restrict__ group_off, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int e, int row0,
    int n, int k, int a_stride, int k_pad,
    std::int64_t blob, std::int64_t seg_off, std::int64_t scale_off) {
    using S = TcpS;
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
    // QPN lane 映射（同 q4_tc.cu）：qp 分 N 四段，r 取段内 8 行。
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
    // CTA 内 warp 排布：2（行）× 4（列）。
    const int wr = warp & 1;
    const int wc = warp >> 1;

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

    // x 的 k-group 舞台（fp16）：64 行 × 64 k，行填充 8 避 bank 冲突。
    __shared__ __half xs[S::kRowsPerCta][S::kShStride];

    const half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032.0
    // 本 warp 权重行的 64k 窗口预取（kBlk 组在飞）。
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
            // ---- 舞台 x：本 CTA 的 64 行 × 本组 64 k，bf16→fp16 一次 CVT ----
            // 每线程 16 个元素：row = tid>>2，列起点 = (tid&3)*16。
            {
                const int srow_i = tid >> 2;          // 0..63
                const int scol = (tid & 3) * 16;      // 0/16/32/48
                const int grow = row0 + srow_i;       // 全局对下标
                __half tmp[16];
                if (grow < t) {
                    const __nv_bfloat16* xrow =
                        x + static_cast<std::int64_t>(s0 + grow) * a_stride + kg + scol;
                    const uint4 raw0 = *reinterpret_cast<const uint4*>(xrow);
                    const uint4 raw1 = *reinterpret_cast<const uint4*>(xrow + 8);
                    const __nv_bfloat16* src0 = reinterpret_cast<const __nv_bfloat16*>(&raw0);
                    const __nv_bfloat16* src1 = reinterpret_cast<const __nv_bfloat16*>(&raw1);
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
                *reinterpret_cast<uint4*>(&xs[srow_i][scol]) =
                    *reinterpret_cast<const uint4*>(tmp);
                *reinterpret_cast<uint4*>(&xs[srow_i][scol + 8]) =
                    *reinterpret_cast<const uint4*>(tmp + 8);
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
                    // A 片：本 lane 负责的 token 行 × 8 个 k，一条 uint4 = 4 个 half2。
                    const int rb = wr * S::kWarpRows + tile * 8 + r;
                    const uint4 av = *reinterpret_cast<const uint4*>(&xs[rb][lbase]);
                    const unsigned* A = reinterpret_cast<const unsigned*>(&av);
                    mma_qpn(c[tile], A[0], A[1], B[0], B[1]);
                    mma_qpn(c[tile], A[2], A[3], B[2], B[3]);
                }
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

extern "C" __global__ void __launch_bounds__(TcpS::kThreads, 2)
lingxi_gemv_q4g64_slots_tcp_bf16(
    const std::uint8_t* __restrict__ slots, const std::int32_t* __restrict__ slot_ids,
    const std::int32_t* __restrict__ group_off, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int n, int k, int a_stride, int k_pad,
    std::int64_t blob, std::int64_t seg_off, std::int64_t scale_off) {
    q4_slots_tcp_body<4>(slots, slot_ids, group_off, x, y,
                         static_cast<int>(blockIdx.y),
                         static_cast<int>(blockIdx.z) * TcpS::kRowsPerCta,
                         n, k, a_stride, k_pad, blob, seg_off, scale_off);
}

// 乐观发射档（`LINGXI_ROUTE_FULLDEV_BIG` 路径）：grid = (ceil(n/128), ng_ub, 1)。
// ng_ub 组的网格是 **host 侧上界**（min(E, S)）——真实组数 ng 由 pair 建表核写在
// group_off[0..ng]（尾部已清零）里，e >= ng 的块 t <= 0 早退；行块按本组 t 核内循环，
// 不需要 host 的 tmax。因此本核可以在 pair 核完成**之前**入队（同流序 ⇒ 读到建成后的表），
// 把 host 从"钉在 pair 完成上才发专家段"里解放出来。算术与旧档逐 (组,行块) 相同。
extern "C" __global__ void __launch_bounds__(TcpS::kThreads, 2)
lingxi_gemv_q4g64_slots_tcp_bf16_dyn(
    const std::uint8_t* __restrict__ slots, const std::int32_t* __restrict__ slot_ids,
    const std::int32_t* __restrict__ group_off, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int n, int k, int a_stride, int k_pad,
    std::int64_t blob, std::int64_t seg_off, std::int64_t scale_off) {
    const int e = static_cast<int>(blockIdx.y);
    const int t = group_off[e + 1] - group_off[e];
    for (int row0 = 0; row0 < t; row0 += TcpS::kRowsPerCta) {
        q4_slots_tcp_body<4>(slots, slot_ids, group_off, x, y, e, row0,
                             n, k, a_stride, k_pad, blob, seg_off, scale_off);
    }
}

#endif // sm_70
