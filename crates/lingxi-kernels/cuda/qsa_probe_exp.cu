// **实验核**（2026-10-24）：QSA 的 idxprep / idxsel / attn 三段的**分母探针** + 候选。
// 只有 `tests/t17_qsa_bw.rs` 会调；生产核一个字节没动（`attn_ops.cu` 保持原样）。
//
// 线上形状（`LINGXI_ATTN_SYNC=1`，t=6、len≈118、nblocks=29、budget=2048、compress=4）：
// attn: grid=(6, 24/NH)=36 block × 256 线程，NH=4；K/V 各读一遍 ⇒ 4.4 MB/层
// idxsel: scores(174 活跃线程) + rank + sel_build 三发，只读 7 KB ⇒ 纯发射/延迟
//
// 本文件的探针全部保持**生产核的 grid/block/地址映射**，只把"计算"换成"搬运"或常量，
// 用来回答"现状是同形分母的百分之几 / 差在重复搬运还是串行依赖"。

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// ───────────────────────────── attn gather ─────────────────────────────
//
// 生产核 `attn_ops.cu::attn_gather_body<NH>` 的逐字复刻 + 各种消融。MODE：
// 0 prod 生产核逐字复刻（自检：应与 `lingxi_attn_gather4_bf16` 逐位相同）
// 1 read_kv 同形纯读 K + V（只读不写分数、不算点积）
// 2 read_k 只读 K（第一遍的地址流）
// 3 read_v 只读 V（第二遍的地址流）
// 4 constq prod 但 q 用常量（不读 q）
// 5 constkv prod 但 K/V 用常量（不读 K/V ⇒ ALU/发射地板）
// 6 pass1 只做第一遍（点积 + 行内 max）
// 7 pass2 只做第二遍（V 加权和；分数用常量填）
// 8 v2 **候选**：第二遍按"选块 → 块内 4 token"重排（blk_tab 每 4 token 只读一次）
// 9 v2pf **候选**：v2 + 4 个 V 值先取齐再累加（纯发射顺序调整，逐位相同）
// 10 empty 同 grid 空核（纯发射地板）
//
// ⚠️ 红线：4/5/6/7 只用于计时（会改数学），8/9 必须与 0 逐位相同。

__device__ __forceinline__ int kv_phys_row_probe(int pos, const int* __restrict__ blk_tab,
                                                 int cshift, int ctok) {
    return blk_tab[pos >> cshift] * ctok + (pos & (ctok - 1));
}

template <int NH, int MODE>
__device__ __forceinline__ void attn_probe_body(const bf16* __restrict__ q,
                                                const bf16* __restrict__ k,
                                                const bf16* __restrict__ v,
                                                const int* __restrict__ blk_tab,
                                                const int* __restrict__ sel,
                                                const int* __restrict__ sel_cnt,
                                                bf16* __restrict__ out,
                                                int heads, int kv_heads, int head_dim,
                                                float scale, int vis_base, int nblocks,
                                                int cshift, int stride) {
    extern __shared__ float sp[];
    const int i = blockIdx.x;
    const int h0 = blockIdx.y * NH;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int kh = h0 / (heads / kv_heads);
    const int vis = vis_base + i + 1;
    const int nsel = sel_cnt[i];
    const int tail0 = nblocks << cshift;
    const int ctok = 1 << cshift;
    const int span = stride * ctok + ctok;
    const int* srow = sel + (size_t)i * stride;

    if (MODE == 10) {
        if (tid == 0 && h0 == 0) out[i] = __float2bfloat16((float)nsel);
        return;
    }

    int4 qreg[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) {
        qreg[g] = make_int4(0, 0, 0, 0);
        if (MODE == 4) {
            // 常量 q（不做 global 读）
            qreg[g] = make_int4(0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00);
        } else if (head_dim == 256) {
            const bf16* qv = q + ((size_t)i * heads + h0 + g) * head_dim;
            qreg[g] = *reinterpret_cast<const int4*>(qv + lane * 8);
        }
    }

    const int ntail = (vis > tail0) ? (vis - tail0) : 0;
    const int total = nsel * ctok + ntail;
    auto pos_of = [&](int idx) -> int {
        const int nb = nsel * ctok;
        if (idx < nb) return srow[idx >> cshift] * ctok + (idx & (ctok - 1));
        return tail0 + (idx - nb);
    };

    // ── 纯读探针：只把 K/V 的字节累加进寄存器，最后写一个 bf16 防 DCE ──
    if (MODE == 1 || MODE == 2 || MODE == 3 || MODE == 11) {
        float acc = 0.0f;
        if (MODE != 3) {
            for (int idx = warp; idx < total; idx += nwarp) {
                const int pos = pos_of(idx);
                if (pos >= vis) continue;
                const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
                const bf16* kv = k + ((size_t)phys * kv_heads + kh) * head_dim;
                const int4 x4 = *reinterpret_cast<const int4*>(kv + lane * 8);
                const bf16* xp = reinterpret_cast<const bf16*>(&x4);
                acc += __bfloat162float(xp[0]) + __bfloat162float(xp[7]);
            }
        }
        if (MODE == 11) {
            // 与候选 v2 相同的"按选块分组"地址流（每个 block 只读一次 blk_tab）
            const size_t vstride = (size_t)kv_heads * head_dim;
            for (int j = 0; j < nsel; ++j) {
                const int blk = srow[j];
                const bf16* vrow =
                    v + (size_t)blk_tab[blk] * ctok * vstride + (size_t)kh * head_dim + tid;
                for (int kk = 0; kk < ctok; ++kk) acc += __bfloat162float(vrow[(size_t)kk * vstride]);
            }
            for (int t2 = 0; t2 < ntail; ++t2) {
                const int pos = tail0 + t2;
                const bf16* vv = v + (size_t)kv_phys_row_probe(pos, blk_tab, cshift, ctok) * vstride
                                 + (size_t)kh * head_dim + tid;
                acc += __bfloat162float(vv[0]);
            }
        } else if (MODE != 2) {
            for (int idx = 0; idx < total; ++idx) {
                const int pos = pos_of(idx);
                const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
                const bf16* vv = v + ((size_t)phys * kv_heads + kh) * head_dim + tid;
                acc += __bfloat162float(vv[0]);
            }
        }
        out[((size_t)i * heads + h0) * head_dim + tid] = __float2bfloat16(acc);
        return;
    }

    // ── 第一遍：点积 + 行内 max ──
    float lmax[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) lmax[g] = -INFINITY;
    if (MODE == 5) {
        // K 用常量：仍走同样的 idx 枚举与 shuffle，只是不读 global
#pragma unroll 1
        for (int idx = warp; idx < total; idx += nwarp) {
            const int pos = pos_of(idx);
            if (pos >= vis) {
                if (lane == 0) {
#pragma unroll
                    for (int g = 0; g < NH; ++g) sp[g * span + idx] = -INFINITY;
                }
                continue;
            }
#pragma unroll
            for (int g = 0; g < NH; ++g) {
                float s = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) s += 0.001f * 0.002f;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                s = __shfl_sync(0xffffffffu, s, 0);
                if (lane == 0) {
                    const float sv = s * scale;
                    sp[g * span + idx] = sv;
                    lmax[g] = fmaxf(lmax[g], sv);
                }
            }
        }
    } else if (MODE != 7) {
        for (int idx = warp; idx < total; idx += nwarp) {
            const int pos = pos_of(idx);
            if (pos >= vis) {
                if (lane == 0) {
#pragma unroll
                    for (int g = 0; g < NH; ++g) sp[g * span + idx] = -INFINITY;
                }
                continue;
            }
            const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
            const bf16* kv = k + ((size_t)phys * kv_heads + kh) * head_dim;
            const int4 kv4 = *reinterpret_cast<const int4*>(kv + lane * 8);
            const bf16* kp = reinterpret_cast<const bf16*>(&kv4);
#pragma unroll
            for (int g = 0; g < NH; ++g) {
                const bf16* qp = reinterpret_cast<const bf16*>(&qreg[g]);
                float s = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) s += __bfloat162float(qp[z]) * __bfloat162float(kp[z]);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                s = __shfl_sync(0xffffffffu, s, 0);
                if (lane == 0) {
                    const float sv = s * scale;
                    sp[g * span + idx] = sv;
                    lmax[g] = fmaxf(lmax[g], sv);
                }
            }
        }
    } else {
        // MODE 7：跳过第一遍，分数填常量
        for (int idx = tid; idx < total; idx += (int)blockDim.x) {
#pragma unroll
            for (int g = 0; g < NH; ++g) sp[g * span + idx] = 0.5f;
        }
    }
    __syncthreads();
    if (MODE == 6) {   // 只做第一遍：把行内 max 写出去防 DCE
        if (tid == 0) {
#pragma unroll
            for (int g = 0; g < NH; ++g)
                out[((size_t)i * heads + h0 + g) * head_dim] = __float2bfloat16(lmax[g]);
        }
        return;
    }

    __shared__ float sred[NH][32];
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < NH; ++g) sred[g][warp] = lmax[g];
    }
    __syncthreads();
    __shared__ float smx[NH];
    if (tid < 32) {
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            float w = (tid < nwarp) ? sred[g][tid] : -INFINITY;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) w = fmaxf(w, __shfl_down_sync(0xffffffffu, w, off));
            if (tid == 0) smx[g] = w;
        }
    }
    __syncthreads();
    if (MODE != 7) {
        for (int idx = tid; idx < total; idx += (int)blockDim.x) {
#pragma unroll
            for (int g = 0; g < NH; ++g) sp[g * span + idx] = __expf(sp[g * span + idx] - smx[g]);
        }
    }
    __syncthreads();
    __shared__ float slsum[NH];
    if (tid == 0) {
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            float ls = 0.0f;
            for (int idx = 0; idx < total; ++idx) ls += sp[g * span + idx];
            slsum[g] = ls;
        }
    }
    __syncthreads();

    // ── 第二遍：V 加权和 ──
    if (MODE == 8 || MODE == 9) {
        float acc[NH];
#pragma unroll
        for (int g = 0; g < NH; ++g) acc[g] = 0.0f;
        const size_t vstride = (size_t)kv_heads * head_dim;
        int idx = 0;
        for (int j = 0; j < nsel; ++j) {
            const int blk = srow[j];
            const bf16* vrow = v + (size_t)blk_tab[blk] * ctok * vstride + (size_t)kh * head_dim + tid;
            if (MODE == 9 && ctok == 4) {
                // 4 个地址一起发出（纯发射顺序，累加次序仍是 idx 升序）。
                // ⚠️ 与生产核一样，**不可见 token 也照样读 V 并乘 0 分**（sp 已被第一遍置 0），
                // 这样连"多读一个 V"的地址流都不变。
                float vx[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) vx[k] = __bfloat162float(vrow[(size_t)k * vstride]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
#pragma unroll
                    for (int g = 0; g < NH; ++g) acc[g] += sp[g * span + idx + k] * vx[k];
                }
                idx += ctok;
            } else {
#pragma unroll 4
                for (int kk = 0; kk < ctok; ++kk, ++idx) {
                    const float vx = __bfloat162float(vrow[(size_t)kk * vstride]);
#pragma unroll
                    for (int g = 0; g < NH; ++g) acc[g] += sp[g * span + idx] * vx;
                }
            }
        }
        for (int t2 = 0; t2 < ntail; ++t2, ++idx) {
            const int pos = tail0 + t2;
            const bf16* vv = v + (size_t)kv_phys_row_probe(pos, blk_tab, cshift, ctok) * vstride
                             + (size_t)kh * head_dim + tid;
            const float vx = __bfloat162float(vv[0]);
#pragma unroll
            for (int g = 0; g < NH; ++g) acc[g] += sp[g * span + idx] * vx;
        }
        if (tid < head_dim) {
#pragma unroll
            for (int g = 0; g < NH; ++g) {
                out[((size_t)i * heads + h0 + g) * head_dim + tid] =
                    __float2bfloat16(acc[g] / slsum[g]);
            }
        }
        return;
    }

    if (tid < head_dim) {
        float acc[NH];
#pragma unroll
        for (int g = 0; g < NH; ++g) acc[g] = 0.0f;
        for (int idx = 0; idx < total; ++idx) {
            const int pos = pos_of(idx);
            float vx;
            if (MODE == 5) {
                vx = 0.002f;
            } else {
                const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
                const bf16* vv = v + ((size_t)phys * kv_heads + kh) * head_dim + tid;
                vx = __bfloat162float(vv[0]);
            }
#pragma unroll
            for (int g = 0; g < NH; ++g) acc[g] += sp[g * span + idx] * vx;
        }
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            out[((size_t)i * heads + h0 + g) * head_dim + tid] =
                __float2bfloat16(acc[g] / slsum[g]);
        }
    }
}

#define ATTN_ARGS                                                                     \
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, \
    const int* __restrict__ blk_tab, const int* __restrict__ sel,                      \
    const int* __restrict__ sel_cnt, bf16* __restrict__ out, int heads, int kv_heads,   \
    int head_dim, float scale, int vis_base, int nblocks, int cshift, int stride

#define ATTN_CALL(NH, MODE)                                                                  \
    attn_probe_body<NH, MODE>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, \
                              scale, vis_base, nblocks, cshift, stride)

extern "C" __global__ void lingxi_qsa_attn_prod4(ATTN_ARGS) { ATTN_CALL(4, 0); }
extern "C" __global__ void lingxi_qsa_attn_readkv4(ATTN_ARGS) { ATTN_CALL(4, 1); }
extern "C" __global__ void lingxi_qsa_attn_readk4(ATTN_ARGS) { ATTN_CALL(4, 2); }
extern "C" __global__ void lingxi_qsa_attn_readv4(ATTN_ARGS) { ATTN_CALL(4, 3); }
extern "C" __global__ void lingxi_qsa_attn_constq4(ATTN_ARGS) { ATTN_CALL(4, 4); }
extern "C" __global__ void lingxi_qsa_attn_constkv4(ATTN_ARGS) { ATTN_CALL(4, 5); }
extern "C" __global__ void lingxi_qsa_attn_pass1_4(ATTN_ARGS) { ATTN_CALL(4, 6); }
extern "C" __global__ void lingxi_qsa_attn_pass2_4(ATTN_ARGS) { ATTN_CALL(4, 7); }
extern "C" __global__ void lingxi_qsa_attn_v2_4(ATTN_ARGS) { ATTN_CALL(4, 8); }
extern "C" __global__ void lingxi_qsa_attn_v2pf_4(ATTN_ARGS) { ATTN_CALL(4, 9); }
extern "C" __global__ void lingxi_qsa_attn_empty4(ATTN_ARGS) { ATTN_CALL(4, 10); }
extern "C" __global__ void lingxi_qsa_attn_readkv_v2_4(ATTN_ARGS) { ATTN_CALL(4, 11); }
// 并行度候选：同一份生产算式，NH=2 / NH=1（grid.y 变 12 / 24 ⇒ block 数 72 / 144）
extern "C" __global__ void lingxi_qsa_attn_prod2(ATTN_ARGS) { ATTN_CALL(2, 0); }
extern "C" __global__ void lingxi_qsa_attn_prod1(ATTN_ARGS) { ATTN_CALL(1, 0); }
extern "C" __global__ void lingxi_qsa_attn_v2pf_2(ATTN_ARGS) { ATTN_CALL(2, 9); }
extern "C" __global__ void lingxi_qsa_attn_v2pf_1(ATTN_ARGS) { ATTN_CALL(1, 9); }

// 候选 v3 复用生产体（`attn_gather_v3.cuh`，与生产核同一份实现，逐位等价）。
#include "attn_gather_v3.cuh"

extern "C" __global__ void lingxi_qsa_attn_v3_1(ATTN_ARGS) { attn_gather_v3_body<1>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }
extern "C" __global__ void lingxi_qsa_attn_v3_2(ATTN_ARGS) { attn_gather_v3_body<2>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }
extern "C" __global__ void lingxi_qsa_attn_v3_4(ATTN_ARGS) { attn_gather_v3_body<4>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }

// ───────────── 候选 v5（global 分数缓冲，smem 归零 → 满占用）─────────────
//
// v3 的教训：smem 分数数组（NH=4 时 32.8KB/块）把占用压到 2 块/SM（16 warp），
// 纯读分母只有 0.76 TB/s；NH=1（8.2KB → 8 块/SM 满占用）同地址流能跑 2.0 TB/s
// ——墙壁是**每 warp 延迟 × 驻留 warp 数**，不是 L2 容量（4.2MB 全驻留对照 C 同速）。
// v5 = NH 头共享（流量 ÷NH）+ 分数走 global scratch（写 32.8KB + tid0 顺序和读回，
// 额外流量 <1%）+ smem 只留 max/sum 标量 → 占用回 8 块/SM。
//
// 逐位等价论证（与 prod 对拍）：
// ① 点积/树/max：与 v3 完全相同（max 本身次序不敏感）；
// ② exp 由 tid0 按 idx 升序就地带入：__expf(raw - smx) 与 prod 的并行 exp
// **同输入同函数** ⇒ 逐位相同；ls[g] 的累加链严格 idx 升序（同 prod）；
// ③ pass2 读回的是②写回的 exp 分值（同一数值），FMA 链严格 idx 升序；
// ④ 块内 __syncthreads 保证 global 写对全块可见。
// spg 布局：[blockIdx.y * gridDim.x + row] * span * NH + idx * NH + g（块私有，无原子）。
template <int NH>
__device__ __forceinline__ void attn_v5_body(const bf16* __restrict__ q,
                                             const bf16* __restrict__ k,
                                             const bf16* __restrict__ v,
                                             const int* __restrict__ blk_tab,
                                             const int* __restrict__ sel,
                                             const int* __restrict__ sel_cnt,
                                             bf16* __restrict__ out,
                                             int heads, int kv_heads, int head_dim,
                                             float scale, int vis_base, int nblocks,
                                             int cshift, int stride,
                                             float* __restrict__ spg) {
    const int i = blockIdx.x;
    const int h0 = blockIdx.y * NH;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int kh = h0 / (heads / kv_heads);
    const int vis = vis_base + i + 1;
    const int nsel = sel_cnt[i];
    const int tail0 = nblocks << cshift;
    const int ctok = 1 << cshift;
    const int span = stride * ctok + ctok;
    const int* srow = sel + (size_t)i * stride;
    float* sp = spg + ((size_t)blockIdx.y * gridDim.x + i) * ((size_t)span * NH);
    int4 qreg[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) {
        qreg[g] = make_int4(0, 0, 0, 0);
        if (head_dim == 256 && lane < 32) {
            const bf16* qv = q + ((size_t)i * heads + h0 + g) * head_dim;
            qreg[g] = *reinterpret_cast<const int4*>(qv + lane * 8);
        }
    }
    const int ntail = (vis > tail0) ? (vis - tail0) : 0;
    const int total = nsel * ctok + ntail;
    auto pos_of = [&](int idx) -> int {
        const int nb = nsel * ctok;
        if (idx < nb) return srow[idx >> cshift] * ctok + (idx & (ctok - 1));
        return tail0 + (idx - nb);
    };
    auto load_k = [&](int idx, bool& ok, int4& kv4) {
        const int pos = pos_of(idx);
        ok = pos < vis;
        if (ok) {
            const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
            const bf16* kp = k + ((size_t)phys * kv_heads + kh) * head_dim;
            kv4 = *reinterpret_cast<const int4*>(kp + lane * 8);
        }
    };

    // ── 第 1 遍：2 深流水点积；分数写 global scratch（原始值 sv = s*scale）──
    float lmax[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) lmax[g] = -INFINITY;
    int idx = warp;
    bool oka = false;
    int4 kva = make_int4(0, 0, 0, 0);
    if (idx < total) load_k(idx, oka, kva);
    for (; idx < total; idx += nwarp) {
        const int nidx = idx + nwarp;
        bool okb = false;
        int4 kvb = make_int4(0, 0, 0, 0);
        if (nidx < total) load_k(nidx, okb, kvb);
        if (oka) {
            const bf16* kp = reinterpret_cast<const bf16*>(&kva);
            float kf[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) kf[z] = __bfloat162float(kp[z]);
#pragma unroll
            for (int g = 0; g < NH; ++g) {
                const bf16* qp = reinterpret_cast<const bf16*>(&qreg[g]);
                float s = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) s += __bfloat162float(qp[z]) * kf[z];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                s = __shfl_sync(0xffffffffu, s, 0);
                if (lane == 0) {
                    const float sv = s * scale;
                    sp[idx * NH + g] = sv;
                    lmax[g] = fmaxf(lmax[g], sv);
                }
            }
        } else if (lane == 0) {
#pragma unroll
            for (int g = 0; g < NH; ++g) sp[idx * NH + g] = -INFINITY;
        }
        oka = okb;
        kva = kvb;
    }
    __shared__ float sred[NH][32];
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < NH; ++g) sred[g][warp] = lmax[g];
    }
    __syncthreads();
    __shared__ float smx[NH];
    if (tid < 32) {
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            float w = (tid < nwarp) ? sred[g][tid] : -INFINITY;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) w = fmaxf(w, __shfl_down_sync(0xffffffffu, w, off));
            if (tid == 0) smx[g] = w;
        }
    }
    __syncthreads();

    // ── exp（tid0 按 idx 升序就地写回）+ 严格升序分母和 ──
    __shared__ float slsum[NH];
    if (tid == 0) {
        float ls[NH];
#pragma unroll
        for (int g = 0; g < NH; ++g) ls[g] = 0.0f;
        for (int idx2 = 0; idx2 < total; ++idx2) {
#pragma unroll
            for (int g = 0; g < NH; ++g) {
                const float e = __expf(sp[idx2 * NH + g] - smx[g]);
                sp[idx2 * NH + g] = e;
                ls[g] += e;
            }
        }
#pragma unroll
        for (int g = 0; g < NH; ++g) slsum[g] = ls[g];
    }
    __syncthreads();

    // ── 第 2 遍：V 加权和（global 分数、块表预取、向量读）──
    if (tid < head_dim) {
        float acc[NH];
#pragma unroll
        for (int g = 0; g < NH; ++g) acc[g] = 0.0f;
        const size_t vstride = (size_t)kv_heads * head_dim;
        int pblk = 0;
        if (nsel > 0) pblk = blk_tab[srow[0]];
        for (int j = 0; j < nsel; ++j) {
            const int phys = pblk;
            if (j + 1 < nsel) pblk = blk_tab[srow[j + 1]];
            const bf16* vrow =
                v + (size_t)phys * ctok * vstride + (size_t)kh * head_dim + tid;
            const int idx2 = j * ctok;
            if (ctok == 4) {
                float vx[4];
#pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) vx[k2] = __bfloat162float(vrow[(size_t)k2 * vstride]);
                if (NH == 4) {
#pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) {
                        const float4 s4 = *reinterpret_cast<const float4*>(&sp[(idx2 + k2) * 4]);
                        acc[0] += s4.x * vx[k2];
                        acc[1] += s4.y * vx[k2];
                        acc[2] += s4.z * vx[k2];
                        acc[3] += s4.w * vx[k2];
                    }
                } else if (NH == 2) {
#pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) {
                        const float2 s2 = *reinterpret_cast<const float2*>(&sp[(idx2 + k2) * 2]);
                        acc[0] += s2.x * vx[k2];
                        acc[1] += s2.y * vx[k2];
                    }
                } else {
                    const float4 s4 = *reinterpret_cast<const float4*>(&sp[idx2]);
                    acc[0] += s4.x * vx[0];
                    acc[0] += s4.y * vx[1];
                    acc[0] += s4.z * vx[2];
                    acc[0] += s4.w * vx[3];
                }
            } else {
                for (int k2 = 0; k2 < ctok; ++k2) {
                    const float vx = __bfloat162float(vrow[(size_t)k2 * vstride]);
#pragma unroll
                    for (int g = 0; g < NH; ++g) acc[g] += sp[(idx2 + k2) * NH + g] * vx;
                }
            }
        }
        for (int t2 = 0; t2 < ntail; ++t2) {
            const int pos = tail0 + t2;
            const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
            const bf16* vv = v + ((size_t)phys * kv_heads + kh) * head_dim + tid;
            const float vx = __bfloat162float(vv[0]);
            const int idx2 = nsel * ctok + t2;
#pragma unroll
            for (int g = 0; g < NH; ++g) acc[g] += sp[idx2 * NH + g] * vx;
        }
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            out[((size_t)i * heads + h0 + g) * head_dim + tid] =
                __float2bfloat16(acc[g] / slsum[g]);
        }
    }
}

#define ATTN_V5_ARGS ATTN_ARGS, float* __restrict__ spg
#define ATTN_V5_CALL(NH) \
    attn_v5_body<NH>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, \
                     scale, vis_base, nblocks, cshift, stride, spg)

extern "C" __global__ void lingxi_qsa_attn_v5_1(ATTN_V5_ARGS) { ATTN_V5_CALL(1); }
extern "C" __global__ void lingxi_qsa_attn_v5_2(ATTN_V5_ARGS) { ATTN_V5_CALL(2); }
extern "C" __global__ void lingxi_qsa_attn_v5_4(ATTN_V5_ARGS) { ATTN_V5_CALL(4); }

// ───────────── 候选 v6（grid 转置：同 row 的 head 块调度相邻）─────────────
//
// v3/v5 的判决：NH≥2 时内核已贴"同形纯读"地板（8.5 vs 8.8ms），ALU 不是墙；
// 纯读地板本身是**调度顺序**问题——grid=(t_q, heads/NH) 时线性分发顺序按
// blockIdx.x（行）走：先扫完全部 512 行的第 0 头组再扫第 1 头组……同 row 的
// 24 个头块在分发序上相隔 512 块 ⇒ 同一份 K/V 被 12 个头块分散重读时 L2 早已
// 淘汰（工作集 512 行 × 2MB）。v6 把 grid 转置成 (heads/NH, t_q)：同 row 的
// 24 个头块**相邻分发**、并发驻留，共享同一份 K/V 的取数窗口（行工作集 2MB ≪ L2）
// ⇒ 行内 12 头共享的 K/V 流量塌缩。算术/访存地址/累加次序与 v3 逐字相同，
// 只换 blockIdx 的角色 ⇒ 逐位等价。
template <int NH>
__device__ __forceinline__ void attn_v6_body(const bf16* __restrict__ q,
                                             const bf16* __restrict__ k,
                                             const bf16* __restrict__ v,
                                             const int* __restrict__ blk_tab,
                                             const int* __restrict__ sel,
                                             const int* __restrict__ sel_cnt,
                                             bf16* __restrict__ out,
                                             int heads, int kv_heads, int head_dim,
                                             float scale, int vis_base, int nblocks,
                                             int cshift, int stride) {
    extern __shared__ float sp[];   // v3 布局：[idx*NH + g]
    const int i = blockIdx.y;       // ★ 转置：行 = y
    const int h0 = blockIdx.x * NH; // ★ 转置：头组 = x
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int kh = h0 / (heads / kv_heads);
    const int vis = vis_base + i + 1;
    const int nsel = sel_cnt[i];
    const int tail0 = nblocks << cshift;
    const int ctok = 1 << cshift;
    const int* srow = sel + (size_t)i * stride;
    int4 qreg[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) {
        qreg[g] = make_int4(0, 0, 0, 0);
        if (head_dim == 256 && lane < 32) {
            const bf16* qv = q + ((size_t)i * heads + h0 + g) * head_dim;
            qreg[g] = *reinterpret_cast<const int4*>(qv + lane * 8);
        }
    }
    const int ntail = (vis > tail0) ? (vis - tail0) : 0;
    const int total = nsel * ctok + ntail;
    auto pos_of = [&](int idx) -> int {
        const int nb = nsel * ctok;
        if (idx < nb) return srow[idx >> cshift] * ctok + (idx & (ctok - 1));
        return tail0 + (idx - nb);
    };
    auto load_k = [&](int idx, bool& ok, int4& kv4) {
        const int pos = pos_of(idx);
        ok = pos < vis;
        if (ok) {
            const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
            const bf16* kp = k + ((size_t)phys * kv_heads + kh) * head_dim;
            kv4 = *reinterpret_cast<const int4*>(kp + lane * 8);
        }
    };

    float lmax[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) lmax[g] = -INFINITY;
    int idx = warp;
    bool oka = false;
    int4 kva = make_int4(0, 0, 0, 0);
    if (idx < total) load_k(idx, oka, kva);
    for (; idx < total; idx += nwarp) {
        const int nidx = idx + nwarp;
        bool okb = false;
        int4 kvb = make_int4(0, 0, 0, 0);
        if (nidx < total) load_k(nidx, okb, kvb);
        if (oka) {
            const bf16* kp = reinterpret_cast<const bf16*>(&kva);
            float kf[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) kf[z] = __bfloat162float(kp[z]);
#pragma unroll
            for (int g = 0; g < NH; ++g) {
                const bf16* qp = reinterpret_cast<const bf16*>(&qreg[g]);
                float s = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) s += __bfloat162float(qp[z]) * kf[z];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                s = __shfl_sync(0xffffffffu, s, 0);
                if (lane == 0) {
                    const float sv = s * scale;
                    sp[idx * NH + g] = sv;
                    lmax[g] = fmaxf(lmax[g], sv);
                }
            }
        } else if (lane == 0) {
#pragma unroll
            for (int g = 0; g < NH; ++g) sp[idx * NH + g] = -INFINITY;
        }
        oka = okb;
        kva = kvb;
    }
    __shared__ float sred[NH][32];
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < NH; ++g) sred[g][warp] = lmax[g];
    }
    __syncthreads();
    __shared__ float smx[NH];
    if (tid < 32) {
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            float w = (tid < nwarp) ? sred[g][tid] : -INFINITY;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) w = fmaxf(w, __shfl_down_sync(0xffffffffu, w, off));
            if (tid == 0) smx[g] = w;
        }
    }
    __syncthreads();
    for (int idx2 = tid; idx2 < total; idx2 += (int)blockDim.x) {
#pragma unroll
        for (int g = 0; g < NH; ++g) sp[idx2 * NH + g] = __expf(sp[idx2 * NH + g] - smx[g]);
    }
    __syncthreads();
    __shared__ float slsum[NH];
    if (tid == 0) {
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            float ls = 0.0f;
            for (int idx2 = 0; idx2 < total; ++idx2) ls += sp[idx2 * NH + g];
            slsum[g] = ls;
        }
    }
    __syncthreads();
    if (tid < head_dim) {
        float acc[NH];
#pragma unroll
        for (int g = 0; g < NH; ++g) acc[g] = 0.0f;
        const size_t vstride = (size_t)kv_heads * head_dim;
        int pblk = 0;
        if (nsel > 0) pblk = blk_tab[srow[0]];
        for (int j = 0; j < nsel; ++j) {
            const int phys = pblk;
            if (j + 1 < nsel) pblk = blk_tab[srow[j + 1]];
            const bf16* vrow =
                v + (size_t)phys * ctok * vstride + (size_t)kh * head_dim + tid;
            const int idx2 = j * ctok;
            if (ctok == 4) {
                float vx[4];
#pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) vx[k2] = __bfloat162float(vrow[(size_t)k2 * vstride]);
                if (NH == 4) {
#pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) {
                        const float4 s4 = *reinterpret_cast<const float4*>(&sp[(idx2 + k2) * 4]);
                        acc[0] += s4.x * vx[k2];
                        acc[1] += s4.y * vx[k2];
                        acc[2] += s4.z * vx[k2];
                        acc[3] += s4.w * vx[k2];
                    }
                } else if (NH == 2) {
#pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) {
                        const float2 s2 = *reinterpret_cast<const float2*>(&sp[(idx2 + k2) * 2]);
                        acc[0] += s2.x * vx[k2];
                        acc[1] += s2.y * vx[k2];
                    }
                } else {
                    const float4 s4 = *reinterpret_cast<const float4*>(&sp[idx2]);
                    acc[0] += s4.x * vx[0];
                    acc[0] += s4.y * vx[1];
                    acc[0] += s4.z * vx[2];
                    acc[0] += s4.w * vx[3];
                }
            } else {
                for (int k2 = 0; k2 < ctok; ++k2) {
                    const float vx = __bfloat162float(vrow[(size_t)k2 * vstride]);
#pragma unroll
                    for (int g = 0; g < NH; ++g) acc[g] += sp[(idx2 + k2) * NH + g] * vx;
                }
            }
        }
        for (int t2 = 0; t2 < ntail; ++t2) {
            const int pos = tail0 + t2;
            const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
            const bf16* vv = v + ((size_t)phys * kv_heads + kh) * head_dim + tid;
            const float vx = __bfloat162float(vv[0]);
            const int idx2 = nsel * ctok + t2;
#pragma unroll
            for (int g = 0; g < NH; ++g) acc[g] += sp[idx2 * NH + g] * vx;
        }
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            out[((size_t)i * heads + h0 + g) * head_dim + tid] =
                __float2bfloat16(acc[g] / slsum[g]);
        }
    }
}

extern "C" __global__ void lingxi_qsa_attn_v6_1(ATTN_ARGS) { attn_v6_body<1>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }
extern "C" __global__ void lingxi_qsa_attn_v6_2(ATTN_ARGS) { attn_v6_body<2>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }
extern "C" __global__ void lingxi_qsa_attn_v6_4(ATTN_ARGS) { attn_v6_body<4>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }

// ───────────── 候选 v7（pass2 V 走 smem 站台：生产者用快流型）─────────────
//
// 分母判决（t25 探针 6a/6b/6c）：K 流（warp-per-token LDG.128，512B/warp）
// 1.48 TB/s；V 流（线程↔dim 映射的逐 token 2B load，64B/warp）只有 0.45 TB/s
// ——同字节数慢 3.3×，V 流是整个 gather 的墙（B 场景纯读 8.8ms 里 V 占 7.2ms）。
// 但 pass2 的"线程固定 dim、acc 严格 idx 升序"是逐位铁律，映射不能换。
// v7 = 生产者/消费者：8 个 warp 轮流用 **K 流的快形态**（一 warp 一 token、
// LDG.128 512B 全宽）把 V 行站台进 smem（vsm），消费者照旧线程↔dim 从 smem 读
// 并严格升序累加——数值与累加次序与 prod 逐位相同，只换 V 的取数路径。
// smem：分数（v3 布局 [idx*NH+g]）+ vsm 单缓冲 CH 个 token（CH·512B）≤ 40KB
// ⇒ 保持 2 块/SM。cp.async 是 sm_80+，V100（sm_70）只能手工站台。
#define AK_V7_CH 16

template <int NH>
__device__ __forceinline__ void attn_v7_body(const bf16* __restrict__ q,
                                             const bf16* __restrict__ k,
                                             const bf16* __restrict__ v,
                                             const int* __restrict__ blk_tab,
                                             const int* __restrict__ sel,
                                             const int* __restrict__ sel_cnt,
                                             bf16* __restrict__ out,
                                             int heads, int kv_heads, int head_dim,
                                             float scale, int vis_base, int nblocks,
                                             int cshift, int stride) {
    extern __shared__ float sp[];              // [idx*NH + g] 分数（v3 布局）
    const int ctok = 1 << cshift;
    bf16* vsm = reinterpret_cast<bf16*>(sp + (size_t)(stride * ctok + ctok) * NH);
    const int i = blockIdx.x;
    const int h0 = blockIdx.y * NH;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int kh = h0 / (heads / kv_heads);
    const int vis = vis_base + i + 1;
    const int nsel = sel_cnt[i];
    const int tail0 = nblocks << cshift;
    const int* srow = sel + (size_t)i * stride;
    int4 qreg[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) {
        qreg[g] = make_int4(0, 0, 0, 0);
        if (head_dim == 256 && lane < 32) {
            const bf16* qv = q + ((size_t)i * heads + h0 + g) * head_dim;
            qreg[g] = *reinterpret_cast<const int4*>(qv + lane * 8);
        }
    }
    const int ntail = (vis > tail0) ? (vis - tail0) : 0;
    const int total = nsel * ctok + ntail;
    auto pos_of = [&](int idx) -> int {
        const int nb = nsel * ctok;
        if (idx < nb) return srow[idx >> cshift] * ctok + (idx & (ctok - 1));
        return tail0 + (idx - nb);
    };
    auto load_k = [&](int idx, bool& ok, int4& kv4) {
        const int pos = pos_of(idx);
        ok = pos < vis;
        if (ok) {
            const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
            const bf16* kp = k + ((size_t)phys * kv_heads + kh) * head_dim;
            kv4 = *reinterpret_cast<const int4*>(kp + lane * 8);
        }
    };

    // ── 第 1 遍：与 v3 相同（2 深流水点积 → 分数写 smem）──
    float lmax[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) lmax[g] = -INFINITY;
    int idx = warp;
    bool oka = false;
    int4 kva = make_int4(0, 0, 0, 0);
    if (idx < total) load_k(idx, oka, kva);
    for (; idx < total; idx += nwarp) {
        const int nidx = idx + nwarp;
        bool okb = false;
        int4 kvb = make_int4(0, 0, 0, 0);
        if (nidx < total) load_k(nidx, okb, kvb);
        if (oka) {
            const bf16* kp = reinterpret_cast<const bf16*>(&kva);
            float kf[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) kf[z] = __bfloat162float(kp[z]);
#pragma unroll
            for (int g = 0; g < NH; ++g) {
                const bf16* qp = reinterpret_cast<const bf16*>(&qreg[g]);
                float s = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) s += __bfloat162float(qp[z]) * kf[z];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                s = __shfl_sync(0xffffffffu, s, 0);
                if (lane == 0) {
                    const float sv = s * scale;
                    sp[idx * NH + g] = sv;
                    lmax[g] = fmaxf(lmax[g], sv);
                }
            }
        } else if (lane == 0) {
#pragma unroll
            for (int g = 0; g < NH; ++g) sp[idx * NH + g] = -INFINITY;
        }
        oka = okb;
        kva = kvb;
    }
    __shared__ float sred[NH][32];
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < NH; ++g) sred[g][warp] = lmax[g];
    }
    __syncthreads();
    __shared__ float smx[NH];
    if (tid < 32) {
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            float w = (tid < nwarp) ? sred[g][tid] : -INFINITY;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) w = fmaxf(w, __shfl_down_sync(0xffffffffu, w, off));
            if (tid == 0) smx[g] = w;
        }
    }
    __syncthreads();
    for (int idx2 = tid; idx2 < total; idx2 += (int)blockDim.x) {
#pragma unroll
        for (int g = 0; g < NH; ++g) sp[idx2 * NH + g] = __expf(sp[idx2 * NH + g] - smx[g]);
    }
    __syncthreads();
    __shared__ float slsum[NH];
    if (tid == 0) {
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            float ls = 0.0f;
            for (int idx2 = 0; idx2 < total; ++idx2) ls += sp[idx2 * NH + g];
            slsum[g] = ls;
        }
    }
    __syncthreads();

    // ── 第 2 遍：V 生产者/消费者（消费者累加次序 = idx 严格升序，与 prod 相同）──
    if (tid < head_dim) {
        float acc[NH];
#pragma unroll
        for (int g = 0; g < NH; ++g) acc[g] = 0.0f;
        const size_t vstride = (size_t)kv_heads * head_dim;
        const int nchunk = (total + AK_V7_CH - 1) / AK_V7_CH;
        for (int c = 0; c < nchunk; ++c) {
            const int base = c * AK_V7_CH;
            const int cn = min(AK_V7_CH, total - base);
            // 站台：warp w 负责 token base + w + r*nwarp（LDG.128 全宽 512B）
            for (int t = warp; t < cn; t += nwarp) {
                const int pos = pos_of(base + t);
                const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
                const bf16* vv = v + ((size_t)phys * kv_heads + kh) * head_dim;
                *reinterpret_cast<int4*>(&vsm[(size_t)t * head_dim + lane * 8]) =
                    *reinterpret_cast<const int4*>(vv + lane * 8);
            }
            __syncthreads();
            // 消费：线程↔dim，token 升序
            for (int t = 0; t < cn; ++t) {
                const float vx = __bfloat162float(vsm[(size_t)t * head_dim + tid]);
                const int idx2 = base + t;
                if (NH == 4) {
                    const float4 s4 = *reinterpret_cast<const float4*>(&sp[idx2 * 4]);
                    acc[0] += s4.x * vx;
                    acc[1] += s4.y * vx;
                    acc[2] += s4.z * vx;
                    acc[3] += s4.w * vx;
                } else if (NH == 2) {
                    const float2 s2 = *reinterpret_cast<const float2*>(&sp[idx2 * 2]);
                    acc[0] += s2.x * vx;
                    acc[1] += s2.y * vx;
                } else {
                    acc[0] += sp[idx2] * vx;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            out[((size_t)i * heads + h0 + g) * head_dim + tid] =
                __float2bfloat16(acc[g] / slsum[g]);
        }
    }
}

extern "C" __global__ void lingxi_qsa_attn_v7_1(ATTN_ARGS) { attn_v7_body<1>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }
extern "C" __global__ void lingxi_qsa_attn_v7_2(ATTN_ARGS) { attn_v7_body<2>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }
extern "C" __global__ void lingxi_qsa_attn_v7_4(ATTN_ARGS) { attn_v7_body<4>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }

// ───────────── 候选 v9（V 行地址一次预计算进 smem + j×2 流水）─────────────
//
// v7（生产者/消费者）负结果的教训：__syncthreads 把 8 个 warp 锁进同步节奏，
// 站台延迟反而暴露在关键路径上。回到 v3 结构，攻剩下的依赖链：pass2 内圈每
// 个 j 都要走 srow(global)→blk_tab(global)→V(global) 三级串行。但 **V 行的
// phys 与 pass1 的 K 行完全相同**（同一 pos、同一 blk_tab）——pass1 顺手把
// phys 写进 smem（span×4B），pass2 内圈变 LDS→LDG 两级且 j×2 展开把 8 个
// V load 同时在飞。数值/累加次序与 v3 逐字相同 ⇒ 逐位等价。
// smem：分数 span·NH f32 + phys span i32（NH=4 时 41KB ⇒ 仍 2 块/SM）。
template <int NH>
__device__ __forceinline__ void attn_v9_body(const bf16* __restrict__ q,
                                             const bf16* __restrict__ k,
                                             const bf16* __restrict__ v,
                                             const int* __restrict__ blk_tab,
                                             const int* __restrict__ sel,
                                             const int* __restrict__ sel_cnt,
                                             bf16* __restrict__ out,
                                             int heads, int kv_heads, int head_dim,
                                             float scale, int vis_base, int nblocks,
                                             int cshift, int stride) {
    extern __shared__ float sp[];              // [idx*NH + g] 分数（v3 布局）
    const int ctok = 1 << cshift;
    const int span = stride * ctok + ctok;
    int* sphys = reinterpret_cast<int*>(sp + (size_t)span * NH);   // [idx] V/K 行号
    const int i = blockIdx.x;
    const int h0 = blockIdx.y * NH;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int kh = h0 / (heads / kv_heads);
    const int vis = vis_base + i + 1;
    const int nsel = sel_cnt[i];
    const int tail0 = nblocks << cshift;
    const int* srow = sel + (size_t)i * stride;
    int4 qreg[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) {
        qreg[g] = make_int4(0, 0, 0, 0);
        if (head_dim == 256 && lane < 32) {
            const bf16* qv = q + ((size_t)i * heads + h0 + g) * head_dim;
            qreg[g] = *reinterpret_cast<const int4*>(qv + lane * 8);
        }
    }
    const int ntail = (vis > tail0) ? (vis - tail0) : 0;
    const int total = nsel * ctok + ntail;
    auto pos_of = [&](int idx) -> int {
        const int nb = nsel * ctok;
        if (idx < nb) return srow[idx >> cshift] * ctok + (idx & (ctok - 1));
        return tail0 + (idx - nb);
    };

    // ── 第 1 遍：2 深流水点积；lane0 顺手把 phys（K=V 同一行）写进 smem ──
    float lmax[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) lmax[g] = -INFINITY;
    int idx = warp;
    bool oka = false;
    int phys_a = 0;
    int4 kva = make_int4(0, 0, 0, 0);
    {
        if (idx < total) {
            const int pos = pos_of(idx);
            oka = pos < vis;
            phys_a = kv_phys_row_probe(pos, blk_tab, cshift, ctok);   // phys 恒算（V 流要用）
            if (oka) {
                const bf16* kp = k + ((size_t)phys_a * kv_heads + kh) * head_dim;
                kva = *reinterpret_cast<const int4*>(kp + lane * 8);
            }
        }
    }
    for (; idx < total; idx += nwarp) {
        const int nidx = idx + nwarp;
        bool okb = false;
        int phys_b = 0;
        int4 kvb = make_int4(0, 0, 0, 0);
        if (nidx < total) {
            const int pos = pos_of(nidx);
            okb = pos < vis;
            phys_b = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
            if (okb) {
                const bf16* kp = k + ((size_t)phys_b * kv_heads + kh) * head_dim;
                kvb = *reinterpret_cast<const int4*>(kp + lane * 8);
            }
        }
        if (lane == 0) sphys[idx] = phys_a;   // 不可见 token 也写（pass2 V 流照读，分=0）
        if (oka) {
            const bf16* kp = reinterpret_cast<const bf16*>(&kva);
            float kf[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) kf[z] = __bfloat162float(kp[z]);
#pragma unroll
            for (int g = 0; g < NH; ++g) {
                const bf16* qp = reinterpret_cast<const bf16*>(&qreg[g]);
                float s = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) s += __bfloat162float(qp[z]) * kf[z];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                s = __shfl_sync(0xffffffffu, s, 0);
                if (lane == 0) {
                    const float sv = s * scale;
                    sp[idx * NH + g] = sv;
                    lmax[g] = fmaxf(lmax[g], sv);
                }
            }
        } else if (lane == 0) {
#pragma unroll
            for (int g = 0; g < NH; ++g) sp[idx * NH + g] = -INFINITY;
        }
        oka = okb;
        phys_a = phys_b;
        kva = kvb;
    }
    __shared__ float sred[NH][32];
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < NH; ++g) sred[g][warp] = lmax[g];
    }
    __syncthreads();
    __shared__ float smx[NH];
    if (tid < 32) {
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            float w = (tid < nwarp) ? sred[g][tid] : -INFINITY;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) w = fmaxf(w, __shfl_down_sync(0xffffffffu, w, off));
            if (tid == 0) smx[g] = w;
        }
    }
    __syncthreads();
    for (int idx2 = tid; idx2 < total; idx2 += (int)blockDim.x) {
#pragma unroll
        for (int g = 0; g < NH; ++g) sp[idx2 * NH + g] = __expf(sp[idx2 * NH + g] - smx[g]);
    }
    __syncthreads();
    __shared__ float slsum[NH];
    if (tid == 0) {
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            float ls = 0.0f;
            for (int idx2 = 0; idx2 < total; ++idx2) ls += sp[idx2 * NH + g];
            slsum[g] = ls;
        }
    }
    __syncthreads();

    // ── 第 2 遍：phys 走 smem（LDS→LDG 两级），j×2 流水 8 个 V load 在飞 ──
    if (tid < head_dim) {
        float acc[NH];
#pragma unroll
        for (int g = 0; g < NH; ++g) acc[g] = 0.0f;
        const size_t vstride = (size_t)kv_heads * head_dim;
        const size_t voff = (size_t)kh * head_dim + tid;
        int j = 0;
        // prologue：预取 j=0 的 phys
        int ph0 = 0;
        if (j < nsel) ph0 = sphys[j * ctok];
        for (; j < nsel; ++j) {
            const int phys_a = ph0;   // 本块首 token 的物理行号；块内 token 物理连续
            int ph1 = 0;
            if (j + 1 < nsel) ph1 = sphys[(j + 1) * ctok];   // 预取下一行号
            const bf16* vrow = v + (size_t)phys_a * vstride + voff;
            const int idx2 = j * ctok;
            if (ctok == 4) {
                float vx[4];
#pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) vx[k2] = __bfloat162float(vrow[(size_t)k2 * vstride]);
                if (NH == 4) {
#pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) {
                        const float4 s4 = *reinterpret_cast<const float4*>(&sp[(idx2 + k2) * 4]);
                        acc[0] += s4.x * vx[k2];
                        acc[1] += s4.y * vx[k2];
                        acc[2] += s4.z * vx[k2];
                        acc[3] += s4.w * vx[k2];
                    }
                } else if (NH == 2) {
#pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) {
                        const float2 s2 = *reinterpret_cast<const float2*>(&sp[(idx2 + k2) * 2]);
                        acc[0] += s2.x * vx[k2];
                        acc[1] += s2.y * vx[k2];
                    }
                } else {
                    const float4 s4 = *reinterpret_cast<const float4*>(&sp[idx2]);
                    acc[0] += s4.x * vx[0];
                    acc[0] += s4.y * vx[1];
                    acc[0] += s4.z * vx[2];
                    acc[0] += s4.w * vx[3];
                }
            } else {
                for (int k2 = 0; k2 < ctok; ++k2) {
                    const float vx = __bfloat162float(vrow[(size_t)k2 * vstride]);
#pragma unroll
                    for (int g = 0; g < NH; ++g) acc[g] += sp[(idx2 + k2) * NH + g] * vx;
                }
            }
            ph0 = ph1;
        }
        for (int t2 = 0; t2 < ntail; ++t2) {
            const int pos = tail0 + t2;
            const int phys = kv_phys_row_probe(pos, blk_tab, cshift, ctok);
            const bf16* vv = v + ((size_t)phys * kv_heads + kh) * head_dim + tid;
            const float vx = __bfloat162float(vv[0]);
            const int idx2 = nsel * ctok + t2;
#pragma unroll
            for (int g = 0; g < NH; ++g) acc[g] += sp[idx2 * NH + g] * vx;
        }
#pragma unroll
        for (int g = 0; g < NH; ++g) {
            out[((size_t)i * heads + h0 + g) * head_dim + tid] =
                __float2bfloat16(acc[g] / slsum[g]);
        }
    }
}

extern "C" __global__ void lingxi_qsa_attn_v9_1(ATTN_ARGS) { attn_v9_body<1>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }
extern "C" __global__ void lingxi_qsa_attn_v9_2(ATTN_ARGS) { attn_v9_body<2>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }
extern "C" __global__ void lingxi_qsa_attn_v9_4(ATTN_ARGS) { attn_v9_body<4>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim, scale, vis_base, nblocks, cshift, stride); }

// ───────────────────────── indexer 打分 ─────────────────────────
//
// 生产核 `lingxi_qsa_indexer_scores_bf16` 的复刻 + 消融。MODE：
// 0 prod 逐字复刻（自检）
// 1 read 同形纯读 q + pooled（不做乘加）
// 2 constq prod 但 q 用常量（只读权重）
// 3 constp prod 但 pooled 用常量（只读 q）
// 4 v2 **候选**：heads 展开成 4 条独立累加链（每个 head 的 d 升序不变 ⇒ 逐位相同）
// 5 v2pf **候选**：v2 + pooled 的 4 个 8 宽段先取齐（更省指令）
//
// grid = (ceil(nblocks/BLOCK), t_q)、block = 256（与生产核一致）。

template <int MODE>
__device__ __forceinline__ void idx_scores_body(const bf16* __restrict__ q,
                                                const bf16* __restrict__ pooled,
                                                float* __restrict__ scores,
                                                int t_q, int nblocks, int heads, int hd,
                                                float sqrt_hd) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (b >= nblocks || row >= t_q) return;
    const bf16* qr = q + (size_t)row * heads * hd;
    const bf16* p = pooled + (size_t)b * hd;
    float s = 0.0f;
    if (MODE == 1) {
        float acc = 0.0f;
        for (int h = 0; h < heads; ++h) {
            const bf16* qh = qr + (size_t)h * hd;
            for (int d = 0; d < hd; ++d) acc += __bfloat162float(qh[d]) + __bfloat162float(p[d]);
        }
        scores[(size_t)row * nblocks + b] = acc;
        return;
    }
    if (MODE == 4 || MODE == 5) {
        // 假设 heads==4（生产配置）；否则退回通用循环
        if (heads == 4 && hd >= 8) {
            const bf16* q0 = qr;
            const bf16* q1 = qr + hd;
            const bf16* q2 = qr + 2 * hd;
            const bf16* q3 = qr + 3 * hd;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            if (MODE == 5) {
                int d = 0;
                for (; d + 8 <= hd; d += 8) {
                    const int4 q0v = *reinterpret_cast<const int4*>(q0 + d);
                    const int4 q1v = *reinterpret_cast<const int4*>(q1 + d);
                    const int4 q2v = *reinterpret_cast<const int4*>(q2 + d);
                    const int4 q3v = *reinterpret_cast<const int4*>(q3 + d);
                    const int4 pv4 = *reinterpret_cast<const int4*>(p + d);
                    const unsigned short* pv = reinterpret_cast<const unsigned short*>(&pv4);
                    const unsigned short* a0 = reinterpret_cast<const unsigned short*>(&q0v);
                    const unsigned short* a1 = reinterpret_cast<const unsigned short*>(&q1v);
                    const unsigned short* a2 = reinterpret_cast<const unsigned short*>(&q2v);
                    const unsigned short* a3 = reinterpret_cast<const unsigned short*>(&q3v);
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        // ⚠️ 必须 `__ushort_as_bfloat16`（按位重解释）；直接
                        // `__bfloat162float((unsigned short)x)` 是把整数**数值**转 bf16，错了。
                        const float pvv = __bfloat162float(__ushort_as_bfloat16(pv[u]));
                        d0 += __bfloat162float(__ushort_as_bfloat16(a0[u])) * pvv;
                        d1 += __bfloat162float(__ushort_as_bfloat16(a1[u])) * pvv;
                        d2 += __bfloat162float(__ushort_as_bfloat16(a2[u])) * pvv;
                        d3 += __bfloat162float(__ushort_as_bfloat16(a3[u])) * pvv;
                    }
                }
                for (; d < hd; ++d) {
                    const float pvv = __bfloat162float(p[d]);
                    d0 += __bfloat162float(q0[d]) * pvv;
                    d1 += __bfloat162float(q1[d]) * pvv;
                    d2 += __bfloat162float(q2[d]) * pvv;
                    d3 += __bfloat162float(q3[d]) * pvv;
                }
            } else {
                for (int d = 0; d < hd; ++d) {
                    const float pvv = __bfloat162float(p[d]);
                    d0 += __bfloat162float(q0[d]) * pvv;
                    d1 += __bfloat162float(q1[d]) * pvv;
                    d2 += __bfloat162float(q2[d]) * pvv;
                    d3 += __bfloat162float(q3[d]) * pvv;
                }
            }
            s = fmaxf(d0, 0.0f);
            s += fmaxf(d1, 0.0f);
            s += fmaxf(d2, 0.0f);
            s += fmaxf(d3, 0.0f);
            scores[(size_t)row * nblocks + b] = s / sqrt_hd;
            return;
        }
    }
    const bool qconst = (MODE == 2);
    const bool pconst = (MODE == 3);
    for (int h = 0; h < heads; ++h) {
        const bf16* qh = qr + (size_t)h * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) {
            const float qv = qconst ? 0.01f : __bfloat162float(qh[d]);
            const float pv = pconst ? 0.02f : __bfloat162float(p[d]);
            dot += qv * pv;
        }
        s += fmaxf(dot, 0.0f);
    }
    scores[(size_t)row * nblocks + b] = s / sqrt_hd;
}

extern "C" __global__ void lingxi_qsa_idx_scores_prod(const bf16* __restrict__ q,
                                                      const bf16* __restrict__ pooled,
                                                      float* __restrict__ scores,
                                                      int t_q, int nblocks, int heads, int hd,
                                                      float sqrt_hd) {
    idx_scores_body<0>(q, pooled, scores, t_q, nblocks, heads, hd, sqrt_hd);
}
extern "C" __global__ void lingxi_qsa_idx_scores_read(const bf16* __restrict__ q,
                                                      const bf16* __restrict__ pooled,
                                                      float* __restrict__ scores,
                                                      int t_q, int nblocks, int heads, int hd,
                                                      float sqrt_hd) {
    idx_scores_body<1>(q, pooled, scores, t_q, nblocks, heads, hd, sqrt_hd);
}
extern "C" __global__ void lingxi_qsa_idx_scores_cq(const bf16* __restrict__ q,
                                                    const bf16* __restrict__ pooled,
                                                    float* __restrict__ scores,
                                                    int t_q, int nblocks, int heads, int hd,
                                                    float sqrt_hd) {
    idx_scores_body<2>(q, pooled, scores, t_q, nblocks, heads, hd, sqrt_hd);
}
extern "C" __global__ void lingxi_qsa_idx_scores_cp(const bf16* __restrict__ q,
                                                    const bf16* __restrict__ pooled,
                                                    float* __restrict__ scores,
                                                    int t_q, int nblocks, int heads, int hd,
                                                    float sqrt_hd) {
    idx_scores_body<3>(q, pooled, scores, t_q, nblocks, heads, hd, sqrt_hd);
}
extern "C" __global__ void lingxi_qsa_idx_scores_v2(const bf16* __restrict__ q,
                                                    const bf16* __restrict__ pooled,
                                                    float* __restrict__ scores,
                                                    int t_q, int nblocks, int heads, int hd,
                                                    float sqrt_hd) {
    idx_scores_body<4>(q, pooled, scores, t_q, nblocks, heads, hd, sqrt_hd);
}
extern "C" __global__ void lingxi_qsa_idx_scores_v2pf(const bf16* __restrict__ q,
                                                      const bf16* __restrict__ pooled,
                                                      float* __restrict__ scores,
                                                      int t_q, int nblocks, int heads, int hd,
                                                      float sqrt_hd) {
    idx_scores_body<5>(q, pooled, scores, t_q, nblocks, heads, hd, sqrt_hd);
}
