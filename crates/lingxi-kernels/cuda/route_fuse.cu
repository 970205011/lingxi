// ── ㊿++ 2026-10-24 **top-k + pair 建表合成一发**（`LINGXI_NO_TKPAIR_FUSE=1` 可回退）──
//
// 动机（微基准 `tests/t20_route_chain_bw.rs`，同进程背靠背 best_of3）：
// router 16.0 µs（空核地板 6.2）/ topk 13.4（5.9）/ pair 12.0（6.1）⇒ 三发整链 37.8 µs，
// 其中 **17.8 µs 是同 grid 空核的地板**（每发 ~6 µs）。topk 与 pair 是纯串行依赖
// （pair 要吃 topk 的 ids），两者的数据量都只有几十 KB ⇒ 合成一发直接省掉一次发射地板。
//
// 逐位等价（与 `lingxi_route_topk_rows_bf16` + `lingxi_route_pairs_bf16` 两发相比）：
// * topk 段：**一 warp 管一行**（本核 block=512 = 16 warp，rows ≤ 16）。
// - max / 逐元素 expf / IEEE 除法：与旧核同一算式（expf 用同一份 glibc 复刻）；
// - `sum` 仍是**单线程按专家下标升序串行累加**（bit 语义的一部分，不许并行归约）；
// - 名次仍是「(p 降序, 下标升序) 全序」逐轮取「键 < 上一轮阈值」的最大键 ——
// 键唯一 ⇒ 名次序列唯一，与旧核的 256 线程 block 归约取到的是**同一批、同一序**；
// - norm 仍按名次顺序串行累加；输出仍按专家升序（数「比自己下标小的选中项个数」）。
// ⇒ 唯一变化是「谁来算这个最大值」——max/键比较都是精确/唯一运算，结果不受归约形状影响。
// * pair 段：核体与 `lingxi_route_pairs_bf16` **逐字相同**（512 线程 = 每线程一个专家，
// 五段 inclusive 前缀和 + 专家升序写 pair/组/CSR/双视图槽号/压缩组下标）。
// 只是把「另一次 launch」换成「本 block 里 `__syncthreads` 之后的续算」。
//
// 线程映射说明：topk 段只用 warp 0..rows-1（每行一个 warp，行内 lane 沿专家下标 strided）；
// pair 段用满 512 线程（与旧核一字不差）。

#include <cuda_bf16.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

// ── glibc expf 的**逐位复刻**（与 `topk_ops.cu` 同一份；该文件里是 A2 设备路由用） ──
// 移植自 glibc `sysdeps/ieee754/flt-32/e_expf.c` + `e_exp2f_data.c`（Arm optimized-routines，
// 无 FMA）。⚠️ 必须用 `__d*_rn`（双精度内建）。
__constant__ uint64_t EXP2F_TAB_F[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL, 0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL, 0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL, 0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};

__device__ __forceinline__ float glibc_expf_fuse(float x) {
    const uint32_t abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= (__float_as_uint(88.0f) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (__float_as_uint(INFINITY) >> 20)) return x + x;
        if (x > 0x1.62e42ep6f) return __int_as_float(0x7f800000);
        if (x < -0x1.9fe368p6f) return 0.0f;
        if (x < -0x1.9d1d9ep6f) return 0.0f;
    }
    const double invln2n = 0x1.71547652b82fep+0 * 32.0;
    const double shift = 0x1.8p+52;
    const double c0 = 0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0;
    const double c1 = 0x1.ebfce50fac4f3p-3 / 32.0 / 32.0;
    const double c2 = 0x1.62e42ff0c52d6p-1 / 32.0;
    const double xd = (double)x;
    const double z = __dmul_rn(invln2n, xd);
    double kd = __dadd_rn(z, shift);
    const uint64_t ki = __double_as_longlong(kd);
    kd = __dsub_rn(kd, shift);
    const double r = __dsub_rn(z, kd);
    uint64_t t = EXP2F_TAB_F[ki % 32];
    t += ki << (52 - 5);
    const double s = __longlong_as_double(t);
    const double zz = __dadd_rn(__dmul_rn(c0, r), c1);
    const double r2 = __dmul_rn(r, r);
    double y = __dadd_rn(__dmul_rn(c2, r), 1.0);
    y = __dadd_rn(__dmul_rn(zz, r2), y);
    y = __dmul_rn(y, s);
    return (float)y;
}

#define FUSE_MAX_ROWS 16
#define FUSE_KMAX 64

// ── top-k 单行的**核体**（一 warp 一行）：合成核与独立核共用这一份实现 ──
// 与 `lingxi_route_topk_rows_bf16` **逐位相同**：max/键比较都是精确或唯一运算；
// `sum`（专家下标升序）与 `norm`（名次序）仍是**单线程串行**累加；输出仍按专家升序。
__device__ __forceinline__ void route_topk_row_warp(
    const bf16* __restrict__ lr, int E, int top_k,
    float* prow, int* sel, float* ssum1, float* snorm1,
    int* __restrict__ ids_row, float* __restrict__ wts_row) {
    const int lane = (int)(threadIdx.x & 31);
    const float NEG_INF = -__int_as_float(0x7f800000);
        // ⚠️ 本核只服务 E ≤ 512（wrapper 里门控）⇒ 每 lane 最多 16 个元素。
        unsigned long long myk[16];
        float mx;
        // 每 lane 一段**连续**的 `per = E/32` 个元素：E=512 ⇒ 16 个 = 2 个 int4 ⇒
        // 整行只发**两次 16B load**（旧写法是 16 次 2B 标量 load 的依赖链，实测吃掉 ~9 µs）。
        // 归属映射改了（lane 持有连续段而非 strided），但 max 与键比较都与归属无关，
        // `sum`/`norm` 仍从 smem 按**下标升序/名次序**串行 ⇒ 结果逐位不变。
        if ((E & 255) == 0) {
            const int per = E >> 5;   // 8 或 16（E=256/512）
            const int nv = per >> 3;  // int4 个数（1 或 2）
            const int4* base = reinterpret_cast<const int4*>(lr) + (size_t)lane * nv;
            int4 v0 = base[0];
            int4 v1 = (nv > 1) ? base[1] : v0;
            const bf16* b0 = reinterpret_cast<const bf16*>(&v0);
            const bf16* b1 = reinterpret_cast<const bf16*>(&v1);
            float lv = NEG_INF;
#pragma unroll
            for (int z = 0; z < 8; ++z) lv = fmaxf(lv, __bfloat162float(b0[z]));
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                if (nv > 1) lv = fmaxf(lv, __bfloat162float(b1[z]));
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) lv = fmaxf(lv, __shfl_down_sync(0xffffffffu, lv, off));
            mx = __shfl_sync(0xffffffffu, lv, 0);
            // 同一份寄存器里的 logits 直接算 exp（只读一遍全局）
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                prow[lane * per + z] = glibc_expf_fuse(__bfloat162float(b0[z]) - mx);
            }
            if (nv > 1) {
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    prow[lane * per + 8 + z] = glibc_expf_fuse(__bfloat162float(b1[z]) - mx);
                }
            }
        } else {
            // 兜底（E 非 256 倍数，本模型走不到）：strided + 4 路独立 load
            float lv = NEG_INF;
            int i = lane;
            for (; i + 96 < E; i += 128) {
                const float a0 = __bfloat162float(lr[i]);
                const float a1 = __bfloat162float(lr[i + 32]);
                const float a2 = __bfloat162float(lr[i + 64]);
                const float a3 = __bfloat162float(lr[i + 96]);
                lv = fmaxf(lv, fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)));
            }
            for (; i < E; i += 32) lv = fmaxf(lv, __bfloat162float(lr[i]));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) lv = fmaxf(lv, __shfl_down_sync(0xffffffffu, lv, off));
            mx = __shfl_sync(0xffffffffu, lv, 0);
            for (int i2 = lane; i2 < E; i2 += 32) {
                prow[i2] = glibc_expf_fuse(__bfloat162float(lr[i2]) - mx);
            }
        }
        __syncwarp();
        // 1.3 **串行**和（按专家下标升序；与旧核 thread0 的加法次序完全相同）
        if (lane == 0) {
            float s = 0.0f;
            int i = 0;
            for (; i + 4 <= E; i += 4) {
                const float a0 = prow[i], a1 = prow[i + 1], a2 = prow[i + 2], a3 = prow[i + 3];
                s += a0; s += a1; s += a2; s += a3;
            }
            for (; i < E; ++i) s += prow[i];
            ssum1[0] = s;
        }
        __syncwarp();
        const float sum = ssum1[0];
        // 1.4 p_i = e_i / sum；全序键进寄存器（元素归属与上面的取数映射一致）
        const bool chunked = ((E & 255) == 0);
        const int per = chunked ? (E >> 5) : 0;
#pragma unroll
        for (int q = 0; q < 16; ++q) {
            const int i = chunked ? (lane * per + q) : (lane + q * 32);
            const bool live = chunked ? (q < per) : (i < E);
            unsigned long long key = 0ull;
            if (live) {
                const float pi = prow[i] / sum;
                prow[i] = pi;
                key = isfinite(pi)
                          ? ((((unsigned long long)__float_as_uint(pi)) << 32) |
                             (unsigned long long)(0xFFFFFFFFu - (unsigned int)i))
                          : 0ull;
            }
            myk[q] = key;
        }
        // 1.5 逐轮取名次（键唯一 ⇒ 名次序列唯一，与旧核同一批、同一序）
        unsigned long long thr = ~0ull;
        int nsel = 0;
        for (int r = 0; r < top_k; ++r) {
            unsigned long long bk = 0ull;
            int bi = -1;
#pragma unroll
            for (int q = 0; q < 16; ++q) {
                const unsigned long long kk = myk[q];
                if (kk > bk && kk < thr) {
                    bk = kk;
                    bi = chunked ? (lane * per + q) : (lane + q * 32);
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const unsigned long long ok = __shfl_down_sync(0xffffffffu, bk, off);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (ok > bk) {
                    bk = ok;
                    bi = oi;
                }
            }
            bk = __shfl_sync(0xffffffffu, bk, 0);
            bi = __shfl_sync(0xffffffffu, bi, 0);
            if (lane == 0 && bi >= 0) {
                thr = bk;
                sel[nsel] = bi;
                ++nsel;
            }
            thr = __shfl_sync(0xffffffffu, thr, 0);
            nsel = __shfl_sync(0xffffffffu, nsel, 0);
        }
        // 1.6 norm：按**名次顺序**串行累加（与旧核同序同值）
        if (lane == 0) {
            float norm = 0.0f;
            for (int r = 0; r < nsel; ++r) norm += prow[sel[r]];
            snorm1[0] = norm;
        }
        __syncwarp();
        const float norm = snorm1[0];
        // 1.7 输出：按专家升序（= 数「选中项里比自己下标小的个数」得槽位）
        for (int j = lane; j < nsel; j += 32) {
            const int v = sel[j];
            int w = 0;
            for (int r = 0; r < nsel; ++r) w += (sel[r] < v) ? 1 : 0;
            ids_row[w] = v;
            wts_row[w] = prow[v] / norm;
        }
        // 1.8 补齐占位（非有限 logits 使选中数 < top_k；host 侧本会报错拒绝）
        for (int j = nsel + lane; j < top_k; j += 32) {
            ids_row[j] = 0;
            wts_row[j] = 0.0f;
        }}

// `logits` = router 的 bf16 logits [rows, E]；其余参数/语义与 `lingxi_route_pairs_bf16` 相同。
extern "C" __global__ void __launch_bounds__(512, 1) lingxi_route_topk_pairs_bf16(
    const bf16* __restrict__ logits, int* __restrict__ ids, float* __restrict__ wts,
    const int* __restrict__ slot_map, const int* __restrict__ slot_map_alt,
    const int* __restrict__ fmt_tab, int* __restrict__ pair_rows,
    float* __restrict__ pair_wts, int* __restrict__ group_slots,
    int* __restrict__ group_slots_alt, int* __restrict__ group_off,
    int* __restrict__ uniq_experts, int* __restrict__ col, int* __restrict__ roff,
    unsigned int* __restrict__ counts, int* __restrict__ hit_pri,
    int* __restrict__ hit_alt, int* __restrict__ miss_ids,
    int* __restrict__ pri_gidx, int* __restrict__ alt_gidx,
    int E, int rows, int top_k) {
    // topk 段：每行一份 p（exp 值 → 除 sum 后的概率）+ 名次→专家下标
    __shared__ float sp[FUSE_MAX_ROWS * 512];
    __shared__ int ssel[FUSE_MAX_ROWS * FUSE_KMAX];
    __shared__ float ssum[FUSE_MAX_ROWS];
    __shared__ float snorm[FUSE_MAX_ROWS];
    // pair 段（与 `lingxi_route_pairs_bf16` 同名同义）
    __shared__ int s_cnt[512];
    __shared__ int s_nz[512];
    __shared__ int s_ms[512];
    __shared__ int s_pnz[512];
    __shared__ int s_anz[512];
    __shared__ int s_tmax;
    __shared__ int s_nprot[2];

    const int tid = (int)threadIdx.x;
    const int wid = tid >> 5;
    const float NEG_INF = -__int_as_float(0x7f800000);

    // ── 1) top-k：一 warp 一行（核体见上面的 `route_topk_row_warp`）──
    if (wid < rows) {
        const int row = wid;
        route_topk_row_warp(logits + (size_t)row * (size_t)E, E, top_k,
                            sp + row * 512, ssel + row * FUSE_KMAX,
                            &ssum[row], &snorm[row],
                            ids + (size_t)row * top_k, wts + (size_t)row * top_k);
    }
    __syncthreads();

    // ── 2) pair / 组 / CSR 建表：与 `lingxi_route_pairs_bf16` **逐字相同** ──
    const int S = rows * top_k;
    const int e = tid;
    const bool has_e = e < E;
    int c = 0;
    if (has_e) {
        for (int j = 0; j < S; ++j) {
            if (ids[j] == e) {
                ++c;
            }
        }
    }
    s_cnt[tid] = c;
    const int is_alt = (has_e && fmt_tab != nullptr) ? fmt_tab[e] : 0;
    const int slm = has_e ? (is_alt ? slot_map_alt[e] : slot_map[e]) : 0;
    const bool miss = has_e && c > 0 && slm <= 0;
    s_nz[tid] = (c > 0) ? 1 : 0;
    s_ms[tid] = miss ? 1 : 0;
    s_pnz[tid] = (c > 0 && !is_alt) ? 1 : 0;
    s_anz[tid] = (c > 0 && is_alt) ? 1 : 0;
    if (tid == 0) {
        s_tmax = 0;
        s_nprot[0] = 0;
        s_nprot[1] = 0;
    }
    for (int r = tid; r < rows + 1; r += 512) {
        roff[r] = r * top_k;
    }
    __syncthreads();
    for (int off = 1; off < 512; off <<= 1) {
        const int v1 = (tid >= off) ? s_cnt[tid - off] : 0;
        const int v2 = (tid >= off) ? s_nz[tid - off] : 0;
        const int v3 = (tid >= off) ? s_ms[tid - off] : 0;
        const int v4 = (tid >= off) ? s_pnz[tid - off] : 0;
        const int v5 = (tid >= off) ? s_anz[tid - off] : 0;
        __syncthreads();
        s_cnt[tid] += v1;
        s_nz[tid] += v2;
        s_ms[tid] += v3;
        s_pnz[tid] += v4;
        s_anz[tid] += v5;
        __syncthreads();
    }
    const int base = s_cnt[tid] - c;
    const int g = has_e ? (s_nz[tid] - (c > 0 ? 1 : 0)) : 0;
    const int mrank = has_e ? (s_ms[tid] - (miss ? 1 : 0)) : 0;
    const int prank = (has_e && c > 0 && !is_alt) ? (s_pnz[tid] - 1) : 0;
    const int arank = (has_e && c > 0 && is_alt) ? (s_anz[tid] - 1) : 0;
    const int U = s_nz[511];
    const int nmiss = s_ms[511];
    const int npri = s_pnz[511];
    const int nalt = s_anz[511];
    if (has_e && c > 0) {
        group_off[g] = base;
        uniq_experts[g] = e;
        atomicMax(&s_tmax, c);
        const int slot = slm - 1;
        group_slots[g] = is_alt ? -1 : slot;
        if (group_slots_alt != nullptr) {
            group_slots_alt[g] = is_alt ? slot : -1;
        }
        if (!is_alt) {
            if (pri_gidx != nullptr) {
                pri_gidx[prank] = g;
            }
        } else if (alt_gidx != nullptr) {
            alt_gidx[arank] = g;
        }
        if (slot >= 0) {
            const int pi = atomicAdd(&s_nprot[is_alt ? 1 : 0], 1);
            if (is_alt) {
                hit_alt[pi] = slot;
            } else {
                hit_pri[pi] = slot;
            }
        }
        int s = base;
        for (int j = 0; j < S; ++j) {
            if (ids[j] != e) {
                continue;
            }
            pair_rows[s] = j / top_k;
            pair_wts[s] = wts[j];
            col[j] = s;
            ++s;
        }
        if (miss) {
            miss_ids[mrank] = e;
        }
    }
    __syncthreads();
    // ★：与 `lingxi_route_pairs_bf16` 逐字相同的尾部清零（乐观发射档界内保护，见
    // moe_ops.cu 同位置注记；本核走不到乐观路径，纯保两核文本一致）。
    for (int g2 = U + 1 + tid; g2 <= E; g2 += 512) {
        group_off[g2] = 0;
    }
    {
        const int tot = s_cnt[511];
        for (int j = tot + tid; j < S; j += 512) {
            pair_rows[j] = 0;
            pair_wts[j] = 0.0f;
            col[j] = 0;
        }
    }
    if (tid == 0) {
        group_off[U] = S;
        counts[0] = (unsigned)S;
        counts[1] = (unsigned)U;
        counts[2] = (unsigned)nmiss;
        counts[3] = (unsigned)s_tmax;
        counts[4] = (unsigned)s_nprot[0];
        counts[5] = (unsigned)s_nprot[1];
        counts[6] = (unsigned)npri;
        counts[7] = (unsigned)nalt;
    }
}

// ── 独立一发：`route_topk_row_warp` 一行一 block（block = 32 = 一个 warp）──
// 与 `lingxi_route_topk_rows_bf16`（一行一 block、256 线程、两级归约 + 逐轮 block 归约）
// **逐位相同**，但每轮归约只在 warp 内做（无 `__syncthreads`）⇒ 省掉逐轮的 block 同步。
// 门控 `LINGXI_NO_TOPK_WARP=1` 回退（wrapper 内）。要求 E ≤ 512、top_k ≤ 64（wrapper 保证）。
extern "C" __global__ void __launch_bounds__(32, 8) lingxi_route_topk_row_warp_bf16(
    const bf16* __restrict__ logits, int* __restrict__ ids, float* __restrict__ wts,
    int E, int top_k) {
    const int row = (int)blockIdx.x;
    __shared__ float prow[512];
    __shared__ int sel[FUSE_KMAX];
    __shared__ float ssum1, snorm1;
    route_topk_row_warp(logits + (size_t)row * (size_t)E, E, top_k, prow, sel, &ssum1, &snorm1,
                        ids + (size_t)row * top_k, wts + (size_t)row * top_k);
}
