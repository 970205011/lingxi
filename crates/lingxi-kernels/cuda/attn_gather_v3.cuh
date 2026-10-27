// ★ （2026-10-26）：QSA gather 注意力 **prefill 形态 v3 核**（生产 + 探针共用体）。
//
// 背景：
// prefill t=512 实测 attn gather ~3.04ms/层，K/V 流量 ~200MB/层 ⇒ DRAM 地板 ~0.4ms。
// t25 分母探针判决：墙壁不是 L2 容量（4.2MB 全驻留对照同速）、不是调度顺序
// （grid 转置 v6 中性）、不是 ALU（constkv 地板仅 37~42%）——是 **V 流的取数形态**
// （线程↔dim 映射下逐 token 2B load，64B/warp，纯读只有 0.45TB/s，K 流的
// warp-per-token 512B/warp 形态是 1.48TB/s）。逐位铁律禁止改 pass2 的线程映射
// 与累加次序，因此 v3 在保持每输出元素"idx 严格升序"的前提下做四件事：
// ① pass1 取数 2 深流水（idx+nwarp 的 pos/K 在当前 token 计算前发出）；
// ② K 行 bf16→f32 每 token 只转一次，NH 个 head 共用；
// ③ sp 分数布局 [idx*NH+g]（值不变），pass2 的分数读变向量（NH=4 每 token
// 一条 LDS.128；NH=1 一条 LDS.128 拿 4 个 token）；
// ④ pass2 的 blk_tab 项预取一拍。
// t=512 微基准（best_of3，B 场景 total=2048/行）：prod NH=1 13.2ms → v3 NH=4
// 8.5ms（−36%）；场景 A（total=1024）：5.97 → 3.86ms（−35%）。全部候选
// （v3/v5/v6/v7/v9）与生产核在 t=512 两场景逐位对拍 0 差异（t25_attn_prefill_bits）。
//
// **逐位等价论证**（与 `attn_gather_body` 对拍）：
// - 每个 head 的点积仍是 lane 内 z=0..7 顺序累加 + 同一 shfl_down 树；
// - softmax 的 max / 分母和 / 加权和仍按 idx 升序、每 head 独立；
// - sp 只是改地址布局，写入值与时机不变；FMA 的操作数值与次序逐字相同；
// - NH 只合并取数（同一 kv-group 的 NH 个 q-head 共享 K/V 行），不改任何
// 每-head 算术 ⇒ NH=2/4 与 NH=1 输出逐位相同（t25 对拍实测 0 差异）。
//
// 调用契约与 `lingxi_attn_gather_bf16` 完全相同（grid=(t_q, heads/NH)、256 线程、
// smem = span·NH·4B）；仅 t≥16 的 prefill 形态由 wrapper 选路，decode（t 小、
// 块数少，NH=1 栅格更优）与旧核路径不受影响。

#ifndef LINGXI_ATTN_GATHER_V3_CUH
#define LINGXI_ATTN_GATHER_V3_CUH

#ifndef LINGXI_V3_KV_PHYS_ROW
#define LINGXI_V3_KV_PHYS_ROW
__device__ __forceinline__ int kv_phys_row_v3(int pos, const int* __restrict__ blk_tab,
                                              int cshift, int ctok) {
    return blk_tab[pos >> cshift] * ctok + (pos & (ctok - 1));
}
#endif

template <int NH>
__device__ __forceinline__ void attn_gather_v3_body(const bf16* __restrict__ q,
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
            const int phys = kv_phys_row_v3(pos, blk_tab, cshift, ctok);
            const bf16* kp = k + ((size_t)phys * kv_heads + kh) * head_dim;
            kv4 = *reinterpret_cast<const int4*>(kp + lane * 8);
        }
    };

    // ── 第 1 遍：2 深流水点积 + 行内 max（K 只读一次，NH 个 head 共用）──
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
        if (nidx < total) load_k(nidx, okb, kvb);   // 预取：与当前 token 计算重叠
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

    // ── exp 每 token 一次，就地写回 shared（v3 布局）──
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

    // ── 第 2 遍：V 加权和（v3 布局向量 LDS + 下一块地址预取）──
    if (tid < head_dim) {
        float acc[NH];
#pragma unroll
        for (int g = 0; g < NH; ++g) acc[g] = 0.0f;
        const size_t vstride = (size_t)kv_heads * head_dim;
        int pblk = 0;
        if (nsel > 0) pblk = blk_tab[srow[0]];
        for (int j = 0; j < nsel; ++j) {
            const int phys = pblk;
            if (j + 1 < nsel) pblk = blk_tab[srow[j + 1]];   // 预取下一块
            const bf16* vrow =
                v + (size_t)phys * ctok * vstride + (size_t)kh * head_dim + tid;
            const int idx2 = j * ctok;
            if (ctok == 4) {
                float vx[4];
#pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) vx[k2] = __bfloat162float(vrow[(size_t)k2 * vstride]);
                if (NH == 4) {
                    // 每 token 一条 LDS.128 = 该 token 全部 NH 个分
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
                    // NH=1：一条 LDS.128 = 4 个 token 的分；FMA 仍逐 token 升序
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
            const int phys = kv_phys_row_v3(pos, blk_tab, cshift, ctok);
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

#endif // LINGXI_ATTN_GATHER_V3_CUH
