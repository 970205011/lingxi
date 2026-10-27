// L1 MoE（稀疏专家）相关算子。全部 bf16，内部 float 计算。

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// 融合 SwiGLU：x[m, 2*inter] → y[m, inter]，y = silu(x[:,0:inter]) * x[:,inter:2*inter]
extern "C" __global__ void lingxi_swiglu_bf16(const bf16* __restrict__ x,
                                              bf16* __restrict__ y,
                                              int m,
                                              int inter) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)m * inter) {
        const int row = (int)(i / inter);
        const int col = (int)(i % inter);
        const bf16* xr = x + (size_t)row * (size_t)(2 * inter);
        const float g = __bfloat162float(xr[col]);
        const float u = __bfloat162float(xr[inter + col]);
        y[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
    }
}

// 按行 gather：dst[i,:] = src[idx[idx_off + i],:]
// `idx_off`：索引/输出行在各自数组里的起始偏移（一次分配整层索引后逐专家切片用）。
extern "C" __global__ void lingxi_gather_rows_bf16(const bf16* __restrict__ src,
                                                   const int* __restrict__ idx,
                                                   bf16* __restrict__ dst,
                                                   int m,
                                                   int k,
                                                   int idx_off) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)m * k) {
        const int row = (int)(i / k);
        const int col = (int)(i % k);
        dst[i] = src[(size_t)idx[idx_off + row] * (size_t)k + (size_t)col];
    }
}

// 按行加权累加：dst[idx[idx_off+i],:] += w[idx_off+i] * src[i,:]
// 同一次调用内 idx 互不相同（top-k 选的是不同专家），故无需原子。
extern "C" __global__ void lingxi_scatter_add_rows_bf16(bf16* __restrict__ dst,
                                                        const int* __restrict__ idx,
                                                        const float* __restrict__ w,
                                                        const bf16* __restrict__ src,
                                                        int m,
                                                        int k,
                                                        int idx_off) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)m * k) {
        const int row = (int)(i / k);
        const int col = (int)(i % k);
        const size_t d = (size_t)idx[idx_off + row] * (size_t)k + (size_t)col;
        dst[d] = __float2bfloat16(__bfloat162float(dst[d])
                                  + w[idx_off + row] * __bfloat162float(src[i]));
    }
}

// ── 批量设备路由的 **pair 构建核**（A 方案 t>1，`LINGXI_ROUTE_FULLDEV=1`） ──────
// 输入 = `route_topk_rows` 的输出 `ids[rows, top_k]`（每行**专家升序**）+ `wts`，以及
// 每层槽位表（slot+1、0=未缓存）。输出与 legacy host 簿记**逐项同序**：
// · pair 组按**专家升序**排列；组内按 (row 升序, j 升序) —— host 遍历序；
// · `group_off[g]`/`uniq_experts[g]` 紧凑（只含非空专家），`[U] = S`；
// · 行 CSR `col[r*top_k + j]` = 该 (r,j) 的 pair 下标（升序），`roff[r] = r*top_k`；
// · 槽号按各专家格式分流到 `group_slots`（主）/`group_slots_alt`（次），另一侧写 -1
// （与 legacy `slot_ids` 同口径：**每组一项**，-1 = 该视图跳过）。
// · ★ 2026-10-24（组压缩）：另出**按格式压缩的组下标表**，让两个视图各自只发自己的组：
// `pri_gidx[npri]`（Q4 组）、`alt_gidx[nalt]`（Q3 组），均按组下标升序。
// GEMV 用 `gidx[e]` 间接取全表的 `group_off[gidx[e]]`/`group_off[gidx[e]+1]`/`slot_ids[gidx[e]]`
// ⇒ 每个真实组读到的 (s0, s1, slot) **逐位不变**，只是网格 y 从 `U` 缩到 `npri`/`nalt`
// （原来对侧格式那些组靠 -1 哨兵空跑）。
// ⚠️ 为什么**不能**直接给一张紧凑的 `group_off`：两种格式的组在 pair 表里是**交错**的，
// 紧凑列表里相邻两项不是"本组起点/终点"，`off[e+1]-off[e]` 会跨过对侧组 ⇒ 必须留 gidx。
// `counts`（映射 host，免 sync 读）8 项：{S, U, nmiss, tmax, nprot_pri, nprot_alt, npri, nalt}。
// 注意 `nprot_*` 数的是**已在槽里的组数**（槽号 ≥ 0），`npri/nalt` 才是**每格式的组数**；
// 稳态（nmiss==0）两者相等。
// `miss_ids`（升序、紧凑）与 `hit_*`（**顺序无关**的保护集）也写映射 host。
// 块大小固定 512（= E），每线程负责一个专家 ⇒ 所有写互不重叠、结果确定。
extern "C" __global__ void __launch_bounds__(512, 1) lingxi_route_pairs_bf16(
    const int* __restrict__ ids, const float* __restrict__ wts,
    const int* __restrict__ slot_map, const int* __restrict__ slot_map_alt,
    const int* __restrict__ fmt_tab, int* __restrict__ pair_rows,
    float* __restrict__ pair_wts, int* __restrict__ group_slots,
    int* __restrict__ group_slots_alt, int* __restrict__ group_off,
    int* __restrict__ uniq_experts, int* __restrict__ col, int* __restrict__ roff,
    unsigned int* __restrict__ counts, int* __restrict__ hit_pri,
    int* __restrict__ hit_alt, int* __restrict__ miss_ids,
    int* __restrict__ pri_gidx, int* __restrict__ alt_gidx,
    int E, int rows, int top_k) {
    __shared__ int s_cnt[512];   // 每专家 pair 数 → 之后变 inclusive 前缀和
    __shared__ int s_nz[512];    // 非空标志 → inclusive 前缀
    __shared__ int s_ms[512];    // 未命中标志 → inclusive 前缀
    __shared__ int s_pnz[512];   // 主格式非空标志 → inclusive 前缀（= pri 组紧凑秩）
    __shared__ int s_anz[512];   // 次格式非空标志 → inclusive 前缀（= alt 组紧凑秩）
    __shared__ int s_tmax;
    __shared__ int s_nprot[2];
    const int tid = (int)threadIdx.x;
    const int S = rows * top_k;
    const int e = tid;
    const bool has_e = e < E;
    // 1) 计数（同时统计每格式 pair 数）
    int c = 0;
    if (has_e) {
        for (int j = 0; j < S; ++j) {
            if (ids[j] == e) {
                ++c;
            }
        }
    }
    s_cnt[tid] = c;
    // 槽号：0 = 未命中；其余 = slot+1
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
    // 行 CSR 右端与每行起点（与 pair 无关，纯算术）
    for (int r = tid; r < rows + 1; r += 512) {
        roff[r] = r * top_k;
    }
    __syncthreads();
    // 2) 五段 inclusive 前缀和（pair 数 / 非空 / 未命中 / 主格式非空 / 次格式非空）
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
    const int base = s_cnt[tid] - c;                 // 本专家在 pair 表里的起点
    const int g = has_e ? (s_nz[tid] - (c > 0 ? 1 : 0)) : 0; // 组下标（非空专家升序）
    const int mrank = has_e ? (s_ms[tid] - (miss ? 1 : 0)) : 0;
    // 每格式的紧凑秩：本组在该格式**组列表**里的下标（升序，与全表同序）
    const int prank = (has_e && c > 0 && !is_alt) ? (s_pnz[tid] - 1) : 0;
    const int arank = (has_e && c > 0 && is_alt) ? (s_anz[tid] - 1) : 0;
    const int U = s_nz[511];
    const int nmiss = s_ms[511];
    const int npri = s_pnz[511];
    const int nalt = s_anz[511];
    // 3) 写 pair（顺序 = 专家升序 → 行升序 → j 升序，与 host 完全一致）
    if (has_e && c > 0) {
        group_off[g] = base;
        uniq_experts[g] = e;
        atomicMax(&s_tmax, c);
        const int slot = slm - 1;
        group_slots[g] = is_alt ? -1 : slot;
        if (group_slots_alt != nullptr) {
            group_slots_alt[g] = is_alt ? slot : -1;
        }
        // 按格式压缩的组下标：本组在自己格式的组列表里占 `prank`/`arank`（升序、无空洞）。
        // GEMV 拿它间接索引全表，边界/槽号因此逐项不变。
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
            col[j] = s; // j = r*top_k + jj ⇒ 该行第 jj 个 pair 的下标
            ++s;
        }
        if (miss) {
            miss_ids[mrank] = e;
        }
    }
    __syncthreads();
    // ★ （乐观发射档的界内保护）：组表/pair 表/CSR 的**尾部清零**——乐观网格按上界
    // u_max=min(E,S) 发射，尾组必须读到 t<=0 才会早退；病态 top-k（ids 残渣，见
    // LINGXI_FULLDEV_CHK）下也保证 gather/combine 的索引读在界内。对旧路径无影响：
    // 旧读法只碰 [0,U] / [0,S) 的有效前缀（group_off[U]=S 保持，尾部本就不被读）。
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

// 合并：`expert_out[r,:] = Σ_{s: rows_of[s] == r} w[s] * o[s,:]`
// （decode 批量路径用：各专家的输出先写进 `o[S, H]`，再一次按路由权重归并回 token 行。）
// 与逐专家 `scatter_add_rows` 的区别：这里在 f32 里累加、最后只 round 一次（更准，非逐位相同）。
extern "C" __global__ void lingxi_moe_combine_rows_bf16(const bf16* __restrict__ o,
                                                        const int* __restrict__ rows_of,
                                                        const float* __restrict__ w,
                                                        bf16* __restrict__ expert_out,
                                                        int S,
                                                        int H) {
    const int r = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        if (rows_of[s] == r) {
            acc += w[s] * __bfloat162float(o[(size_t)s * H + h]);
        }
    }
    expert_out[(size_t)r * H + h] = __float2bfloat16(acc);
}

// 合并+残差**一发融合**（P15-㊶，差距#9）：`out[r,:] = bf16(f32(bf16(Σ w·o)) + f32(shared[r,:]))`。
// 两个舍入（combine 的 round + add 的 round）与"combine 一发 + add_bf16 一发"**完全同序**
// ⇒ 逐位相同；省一次 5120 元素 kernel launch × 48 层。
extern "C" __global__ void lingxi_moe_combine_add_rows_bf16(const bf16* __restrict__ o,
                                                            const int* __restrict__ rows_of,
                                                            const float* __restrict__ w,
                                                            const bf16* __restrict__ shared,
                                                            bf16* __restrict__ out,
                                                            int S,
                                                            int H) {
    const int r = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        if (rows_of[s] == r) {
            acc += w[s] * __bfloat162float(o[(size_t)s * H + h]);
        }
    }
    const bf16 e = __float2bfloat16(acc);
    out[(size_t)r * H + h] =
        __float2bfloat16(__bfloat162float(e) + __bfloat162float(shared[(size_t)r * H + h]));
}

// ㊾ 合并+残差 CSR 版（batch 路径，`S=对数` 可以是几千——旧核逐元素扫全 S 是
// O(rows×H×S)，rows=887 实测 19ms/层，占专家段 76%）：
// `out[r,:] = bf16(f32(bf16(Σ_j w[col[j]]·o[col[j],:])) + f32(shared[r,:]))`
// `col[row_off[r]..row_off[r+1])` = 该行所有对的原下标 s，**升序**（host 稳定计数
// 排序保证）⇒ 每行的累加次序与旧核"升序 s 扫描"完全一致 ⇒ 逐位相同。
extern "C" __global__ void lingxi_moe_combine_add_csr_bf16(
    const bf16* __restrict__ o, const int* __restrict__ col,
    const int* __restrict__ row_off, const float* __restrict__ w,
    const bf16* __restrict__ shared, bf16* __restrict__ out, int H) {
    const int r = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    float acc = 0.0f;
    for (int j = row_off[r]; j < row_off[r + 1]; ++j) {
        const int s = col[j];
        acc += w[s] * __bfloat162float(o[(size_t)s * H + h]);
    }
    const bf16 e = __float2bfloat16(acc);
    out[(size_t)r * H + h] =
        __float2bfloat16(__bfloat162float(e) + __bfloat162float(shared[(size_t)r * H + h]));
}

// bf16 GEMV：`y[n] = W[n,k] · x[k]`（W 行主序）。T=1 的 decode 用。
// 一个 warp 管一行、lane 沿 k 分片 → 访存合并；x 常驻 L1/L2（几 KB）。
// 动机：cuBLAS 的 `gemm_ex` 每次调用约 **26 µs CPU 开销**（实测），T=1 下远超其计算量。
__device__ __forceinline__ float lingxi_dot_bf16(const bf16* __restrict__ wr,
                                                 const bf16* __restrict__ x,
                                                 int k, int lane) {
    float acc = 0.0f;
    for (int j = lane; j < k; j += 32) {
        acc += __bfloat162float(wr[j]) * __bfloat162float(x[j]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    return acc;
}

extern "C" __global__ void lingxi_gemv_bf16(const bf16* __restrict__ w,
                                            const bf16* __restrict__ x,
                                            bf16* __restrict__ y,
                                            int n,
                                            int k) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const float acc = lingxi_dot_bf16(w + (size_t)warp * (size_t)k, x, k, lane);
    if (lane == 0) {
        y[warp] = __float2bfloat16(acc);
    }
}

// 带**尾融合**的 bf16 GEMV：把紧跟其后的逐元素 op 折进同一个 kernel，省掉一次
// kernel 发起 + 一次读写往返。decode 是"kernel 数"受限（每层 40 来个 kernel、
// 每个 ~10-30 µs 的固定成本），尾融合是最划算的一刀。
// epi = 0：`y = acc`
// epi = 1：`y = silu(acc * a)` （GR mix_down：先除 hc 再 silu）
// epi = 2：`y = sigmoid(acc * a) * b` （GR mix_up / block_inject）
// epi = 3：`y = acc * a + b` （仿射）
// epi = 4：`y = silu(g) * u`，g/u 取 W 的第 i 与第 i+n/2 行（MoE swiglu）
extern "C" __global__ void lingxi_gemv_bf16_epi(const bf16* __restrict__ w,
                                                const bf16* __restrict__ x,
                                                bf16* __restrict__ y,
                                                int n,
                                                int k,
                                                int epi,
                                                float a,
                                                float b) {
    const int half = (epi == 4) ? (n >> 1) : n;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= half) {
        return;
    }
    if (epi == 4) {
        const float g = lingxi_dot_bf16(w + (size_t)warp * (size_t)k, x, k, lane);
        const float u = lingxi_dot_bf16(w + (size_t)(warp + half) * (size_t)k, x, k, lane);
        if (lane == 0) {
            const float s = g / (1.0f + __expf(-g));
            y[warp] = __float2bfloat16(s * u);
        }
        return;
    }
    float acc = lingxi_dot_bf16(w + (size_t)warp * (size_t)k, x, k, lane);
    if (lane != 0) {
        return;
    }
    if (epi == 1) {
        const float v = acc * a;
        acc = v / (1.0f + __expf(-v));
    } else if (epi == 2) {
        const float v = acc * a;
        acc = (1.0f / (1.0f + __expf(-v))) * b;
    } else if (epi == 3) {
        acc = acc * a + b;
    }
    y[warp] = __float2bfloat16(acc);
}

// **一个 block 负责一行**的高并行度 GEMV（1024 线程 = 32 warp 一起算一行，warp 间用
// shared 归约）。专治"n 小、k 大"的形状：
//
// 动机（engine 内实测，`lingxi-bench launch-bench`）：
// [320,10240] 6.2 MB，warps-per-row 只有 40 个 block → **26.1 µs（251 GB/s）**
// [ 4,10240] 0.1 MB，只有 **1 个 block** → **13.6 µs（6 GB/s）**
// [8192,2560] 40 MB（1024 blocks）→ 57.9 µs（725 GB/s）
// 原因不是带宽，是**每行的并行度**：warps-per-row 下每 lane 要串行走 k/32 次依赖 load，
// k=10240 就是 320 次 → 访存延迟串成链。本 kernel 把每行的并行度提到 32 warp，
// k=10240 只需 10 次 load 链；n 个 block 也保证了 SM 占用。
//
// 语义与 `lingxi_gemv_bf16_epi` 相同（含同样的尾融合），差别只在**累加顺序**
// （warp 间先归约再跨 warp），因此数值会有最后一两位的差异。
extern "C" __global__ void lingxi_gemv_bf16_block_epi(const bf16* __restrict__ w,
                                                      const bf16* __restrict__ x,
                                                      bf16* __restrict__ y,
                                                      int n,
                                                      int k,
                                                      int epi,
                                                      float a,
                                                      float b) {
    const int row = blockIdx.x;
    if (row >= n) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc = 0.0f;
    for (int j = warp * 32 + lane; j < k; j += nwarp * 32) {
        acc += __bfloat162float(wr[j]) * __bfloat162float(x[j]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    __shared__ float sred[32];
    if (lane == 0) {
        sred[warp] = acc;
    }
    __syncthreads();
    if (tid >= 32) {
        return;
    }
    float v = (lane < nwarp) ? sred[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane != 0) {
        return;
    }
    if (epi == 1) {
        const float t = v * a;
        v = t / (1.0f + __expf(-t));
    } else if (epi == 2) {
        const float t = v * a;
        v = (1.0f / (1.0f + __expf(-t))) * b;
    } else if (epi == 3) {
        v = v * a + b;
    }
    y[row] = __float2bfloat16(v);
}

// **向量化 block GEMV**：一行一个 block，每线程一次读 **16 字节（8 个 bf16）**。
//
// 动机（实测，`lingxi-bench launch-bench` 的形状表）：
// [320,10240] 6.2 MB 走 `gemv_bf16_ptr`（warps-per-row）只有 **40 block → 254 GB/s**；
// [4,10240] 只有 1 block → 6 GB/s。瓶颈不是带宽，而是**每线程只发 2 字节的标量 load**
// （k=10240 时每 lane 要走 320 次依赖 load），访存延迟串成链、指令发射被 load 打满。
// 本 kernel 把每线程的 load 次数压到 `k/8/blockDim`（k=10240、256 线程时只要 5 次），
// 让访存真正跑在带宽上。语义与 `lingxi_gemv_bf16_block_epi` 相同（只是归约顺序不同）。
// 要求 `k % 8 == 0`（否则尾部走标量兜底）。
extern "C" __global__ void lingxi_gemv_bf16_row8_epi(const bf16* __restrict__ w,
                                                     const bf16* __restrict__ x,
                                                     bf16* __restrict__ y,
                                                     int n, int k, int epi, float a, float b) {
    const int row = blockIdx.x;
    if (row >= n) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc = 0.0f;
    const int vec_k = k & ~7;
    for (int j = tid * 8; j < vec_k; j += (int)blockDim.x * 8) {
        const int4 wv = *reinterpret_cast<const int4*>(wr + j);
        const int4 xv = *reinterpret_cast<const int4*>(x + j);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
        const bf16* xp = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc += __bfloat162float(wp[t]) * __bfloat162float(xp[t]);
        }
    }
    for (int j = vec_k + tid; j < k; j += blockDim.x) {
        acc += __bfloat162float(wr[j]) * __bfloat162float(x[j]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    __shared__ float sred[32];
    if (lane == 0) {
        sred[warp] = acc;
    }
    __syncthreads();
    if (tid != 0) {
        return;
    }
    float v = 0.0f;
    for (int wp2 = 0; wp2 < nwarp; ++wp2) {
        v += sred[wp2];
    }
    if (epi == 1) {
        const float t = v * a;
        v = t / (1.0f + __expf(-t));
    } else if (epi == 2) {
        const float t = v * a;
        v = (1.0f / (1.0f + __expf(-t))) * b;
    } else if (epi == 3) {
        v = v * a + b;
    }
    y[row] = __float2bfloat16(v);
}

// **一行一个 block 的瘦长 GEMM**（t>1 用）：与 row8 同款并行度（n 个 block），128 位 load，
// m 个独立累加器。
//
// ⚠️ 旧版 `lingxi_gemm_skinny_m*` 的 grid 是 **n/8 个 block**（warps-per-row），
// n 小的时候并行度差 8 倍：GR 的 `block_inject`（n=4）只有 **1 个 block**、
// `mix_down`（n=320）只有 40 个。实测 t=2 时 GR 从 3.4 ms 涨到 10.3 ms（3.0×），
// 而理论上"权重只读一遍"应该接近 1.1×。
// 本核把 grid 拉回 n 个 block（与 t=1 的 row8 一致），并把每线程的 load 数除以 8。
// 语义与 `lingxi_gemm_skinny_m*` 完全一致（`y[m][n] = W[n,k]·x[m,k]^T` + 同样的尾融合）。
// `epi == 4`（swiglu）不走本核（调用方保证）。
extern "C" __global__ void lingxi_gemm_skinny_row_bf16(const bf16* __restrict__ w,
                                                       const bf16* __restrict__ x,
                                                       bf16* __restrict__ y,
                                                       int n, int k, int m, int epi,
                                                       float a, float b) {
    const int row = blockIdx.x;
    // ★ 2026-10-22 **grid.y 行分块**：blockIdx.y 每块接 8 行（m 任意大）。
    // 每块的 8 行子块与旧"逐片 launch"**数学完全同构**（同线程 k 划分、同归约树、
    // 同 epi）⇒ 逐位等价；并发块共享同一份权重读 ⇒ DRAM 流量从 ⌈m/8⌉ 遍降到 ~1 遍
    // （旧 host 循环是顺序 64 发、每发重读全权重，实测 GR mix_down 1.75ms/层）。
    // grid.y == 1 时 r0=0、mt=m，行为与旧版逐字节一致。
    const int r0 = blockIdx.y * 8;
    if (row >= n || r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    int c0 = tid * 8;
    for (; c0 + 8 <= k; c0 += (int)blockDim.x * 8) {
        const int4 wv = *reinterpret_cast<const int4*>(wr + c0);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= mt) {
                break;
            }
            const int4 xv = *reinterpret_cast<const int4*>(x + (size_t)(r0 + s) * (size_t)k + c0);
            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
            float d = 0.0f;
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                d += __bfloat162float(wp[z]) * __bfloat162float(xp[z]);
            }
            acc[s] += d;
        }
    }
    for (; c0 < k; c0 += (int)blockDim.x * 8) {
        for (int i = 0; i < 8 && c0 + i < k; ++i) {
            const float wf = __bfloat162float(wr[c0 + i]);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= mt) {
                    break;
                }
                acc[s] += wf * __bfloat162float(x[(size_t)(r0 + s) * (size_t)k + c0 + i]);
            }
        }
    }
    __shared__ float sred[32][8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
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
                y[(size_t)(r0 + s) * (size_t)n + row] = __float2bfloat16(r);
            }
        }
    }
}

// ★ P15-⑨（2026-10-17）：`gemm_skinny_row` 的 **x 预转 fp32** 版（"所有后面计算过程
// 要转化的提前转化"）。skinny_row 在 m>1 时带宽腰斩（m=8 实测 350-415 GB/s vs m=1 750+）
// 的根因：内层每 (元素,token) 重复 bf16→f32 CVT（m=8 时每 16B 权重 ~200 条指令，
// issue 塞爆）。这里 w 的 CVT 每轮只做 8 次（wf[8]），x 由调用方一次性预转
// （`convert_bf16_f32`，bf16→f32 精确 ⇒ **逐位等价**），内层每 token 只剩 2×LDG.128 + 8 FMA。
// 尾部/归约/epi 与 `lingxi_gemm_skinny_row_bf16` 完全同构。
extern "C" __global__ void lingxi_gemm_skinny_row_f32x_bf16w(const bf16* __restrict__ w,
                                                             const float* __restrict__ x,
                                                             bf16* __restrict__ y,
                                                             int n, int k, int m, int epi,
                                                             float a, float b) {
    const int row = blockIdx.x;
    // grid.y 行分块：与 `lingxi_gemm_skinny_row_bf16` 同一改动、同一逐位等价论证。
    const int r0 = blockIdx.y * 8;
    if (row >= n || r0 >= m) {
        return;
    }
    const int mt = (m - r0) < 8 ? (m - r0) : 8;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        acc[s] = 0.0f;
    }
    int c0 = tid * 8;
    for (; c0 + 8 <= k; c0 += (int)blockDim.x * 8) {
        const int4 wv = *reinterpret_cast<const int4*>(wr + c0);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
        float wf[8];
#pragma unroll
        for (int z = 0; z < 8; ++z) {
            wf[z] = __bfloat162float(wp[z]);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= mt) {
                break;
            }
            const float4 x0 = *reinterpret_cast<const float4*>(x + (size_t)(r0 + s) * (size_t)k + c0);
            const float4 x1 = *reinterpret_cast<const float4*>(x + (size_t)(r0 + s) * (size_t)k + c0 + 4);
            float d = 0.0f;
            d += wf[0] * x0.x;
            d += wf[1] * x0.y;
            d += wf[2] * x0.z;
            d += wf[3] * x0.w;
            d += wf[4] * x1.x;
            d += wf[5] * x1.y;
            d += wf[6] * x1.z;
            d += wf[7] * x1.w;
            acc[s] += d;
        }
    }
    for (; c0 < k; c0 += (int)blockDim.x * 8) {
        for (int i = 0; i < 8 && c0 + i < k; ++i) {
            const float wf = __bfloat162float(wr[c0 + i]);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= mt) {
                    break;
                }
                acc[s] += wf * x[(size_t)(r0 + s) * (size_t)k + c0 + i];
            }
        }
    }
    __shared__ float sred[32][8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
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
                y[(size_t)(r0 + s) * (size_t)n + row] = __float2bfloat16(r);
            }
        }
    }
}

// ★ P15-⑲（2026-10-17）：`gemm_skinny_row_bf16` 的**多输出行/块**版（R=4）。
// m>1 带宽腰斩（m=8 实测 347-416 GB/s vs m=1 750+）有两个叠加根因：
// ① 内层每 (元素,token) 重复 bf16→f32 CVT —— ⑨ 的 f32x 预转解掉了 CVT，但 x 变
// fp32 后每 block 的 x 流量翻倍 ⇒ 整机净亏（P15-⑨ 判否）；
// ② 每 block 只算 1 个输出行 ⇒ x[m,k] 被 n 个 block 重复读（L2）+ 重复 CVT。
// 本核：一块算 R 个输出行，x 按 2048 元素分块进**动态 smem（转好 f32）**，行间共享
// ⇒ x 的 L2 读与 CVT 同除 R；权重流量不变。CVT 与 L2 一起解，避开 f32x 的翻倍坑。
//
// **逐位等价**（对 skinny_row，BLOCK=256 同款启动）论证：每线程 acc[r][s] 的累加
// 序列与单行版**完全相同**——分块只改 x 的搬运时机，不改每线程访问元素的全局次序
// （c0 = cb + tid*8，cb 是步长 BLOCK*8=2048 的倍数 ⇒ 每 chunk 每线程恰一个位置，全局
// 递增）；8 元素组内 z 次序、vec/tail 两路分支条件（c0+8<=k）与 warp/smem 两级归约
// 全部照搬；chunk 边界是 8 的倍数 ⇒ 不切断 vec 组。bf16→f32 精确，smem 里的就是
// 原核每轮现转的那个值。
#define SKINNY_MULTI_R 4
#define SKINNY_MULTI_CHUNK 2048
// ★ bank 冲突消解：xs 行内地址加 swizzle（每 32 float 插 1 个 padding word）。
// 未 swizzle 时 32 线程读 c = t*8+z 只落在 4 个 bank（8-way 冲突，实测整体慢 9×）；
// swizzle 后 (c + c/32) mod 32 对 t=0..31 两两全不同 ⇒ 无冲突。
#define SKINNY_MULTI_XS_ROW (SKINNY_MULTI_CHUNK + (SKINNY_MULTI_CHUNK >> 5))
#define XS_SW(c) ((c) + ((c) >> 5))
extern "C" __global__ void lingxi_gemm_skinny_row_multi_bf16(const bf16* __restrict__ w,
                                                             const bf16* __restrict__ x,
                                                             bf16* __restrict__ y,
                                                             int n, int k, int m, int epi,
                                                             float a, float b) {
    const int row0 = blockIdx.x * SKINNY_MULTI_R;
    if (row0 >= n || m <= 0) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int R = min(SKINNY_MULTI_R, n - row0);
    extern __shared__ float xs[];  // [m][SKINNY_MULTI_XS_ROW] f32（行内 swizzle）

    float acc[SKINNY_MULTI_R][8];
#pragma unroll
    for (int r = 0; r < SKINNY_MULTI_R; ++r) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[r][s] = 0.0f;
        }
    }

    for (int cb = 0; cb < k; cb += SKINNY_MULTI_CHUNK) {
        const int clen = min(SKINNY_MULTI_CHUNK, k - cb);
        __syncthreads();  // 等上一轮 compute 把 xs 读完（首圈无竞争）
        for (int j = tid; j < m * clen; j += (int)blockDim.x) {
            const int s = j / clen;
            const int c = j - s * clen;
            xs[s * SKINNY_MULTI_XS_ROW + XS_SW(c)] = __bfloat162float(x[(size_t)s * (size_t)k + cb + c]);
        }
        __syncthreads();
        const int c0 = cb + tid * 8;
        if (c0 >= cb + clen) {
            continue;  // 尾巴 chunk（clen < BLOCK*8）里本线程没有位置
        }
        const int lc = c0 - cb;  // chunk 内位置（0 ≤ lc < 2048）
        if (c0 + 8 <= k) {
            // 先把 R 行权重的 8 元素组转好进寄存器（wf[r][z]，编译期下标）
            float wf[SKINNY_MULTI_R][8];
#pragma unroll
            for (int r = 0; r < SKINNY_MULTI_R; ++r) {
                if (r >= R) {
                    break;
                }
                const int4 wv =
                    *reinterpret_cast<const int4*>(w + (size_t)(row0 + r) * (size_t)k + c0);
                const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    wf[r][z] = __bfloat162float(wp[z]);
                }
            }
            // 每个 token 的 x 组只从 smem 读一次（进 xf[8] 寄存器），4 行共享
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= m) {
                    break;
                }
                float xf[8];
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    xf[z] = xs[s * SKINNY_MULTI_XS_ROW + XS_SW(lc + z)];
                }
#pragma unroll
                for (int r = 0; r < SKINNY_MULTI_R; ++r) {
                    if (r >= R) {
                        break;
                    }
                    float d = 0.0f;
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        d += wf[r][z] * xf[z];
                    }
                    acc[r][s] += d;
                }
            }
        } else {
            for (int i = 0; i < 8 && c0 + i < k; ++i) {
                float wf[SKINNY_MULTI_R];
#pragma unroll
                for (int r = 0; r < SKINNY_MULTI_R; ++r) {
                    if (r >= R) {
                        break;
                    }
                    wf[r] = __bfloat162float(w[(size_t)(row0 + r) * (size_t)k + c0 + i]);
                }
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s >= m) {
                        break;
                    }
#pragma unroll
                    for (int r = 0; r < SKINNY_MULTI_R; ++r) {
                        if (r >= R) {
                            break;
                        }
                        acc[r][s] += wf[r] * xs[s * SKINNY_MULTI_XS_ROW + XS_SW(lc + i)];
                    }
                }
            }
        }
    }

    __shared__ float sred[32][8];
#pragma unroll
    for (int r = 0; r < SKINNY_MULTI_R; ++r) {
        if (r >= R) {
            break;
        }
        __syncthreads();  // 保护上一轮 r 对 sred 的读取
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= m) {
                break;
            }
            float v = acc[r][s];
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
            for (int s = 0; s < m; ++s) {
                float v = (lane < nwarp) ? sred[lane][s] : 0.0f;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    v += __shfl_down_sync(0xffffffffu, v, off);
                }
                if (lane == 0) {
                    float rr = v;
                    if (epi == 1) {
                        const float t2 = rr * a;
                        rr = t2 / (1.0f + __expf(-t2));
                    } else if (epi == 2) {
                        const float t2 = rr * a;
                        rr = (1.0f / (1.0f + __expf(-t2))) * b;
                    } else if (epi == 3) {
                        rr = rr * a + b;
                    }
                    y[(size_t)s * (size_t)n + row0 + r] = __float2bfloat16(rr);
                }
            }
        }
    }
}

// **瘦长 GEMM**（模板版）：`y[m, n] += W[n, k] · x[m, k]^T`（W 行主序 [n,k]）。
//
// 一个 warp 负责一行输出，把 W 的该行**只读一遍**，同时对 m 个输入行做 m 个点积。
//
// 动机与**为什么必须按 m 特化**（实测，见 launch-bench 的 skinny 对比）：
// gemv [8192,2560] → 58 µs（724 GB/s）
// skinny [8192,2560] m=1 → 195 µs（**215 GB/s**）
// 原实现固定开 8 个累加器、每元素发 8 次 LDG+FFMA，即使 m=1 也照发 → 变成指令发射受限，
// 权重带宽掉到 1/3.4。特化后 m=1 只有 1 套 load/FMA，应回到 ~700 GB/s。
// 实测 m=1/2/4 的耗时几乎相同（215/215/214 GB/s）说明"多出来的列是白送的"——
// 只要别再让指令数随 8 走。这也正是 T>1（prefill / 投机验证）能不能快起来的关键。
template <int M>
__device__ __forceinline__ void skinny_body(const bf16* __restrict__ w,
                                            const bf16* __restrict__ x,
                                            bf16* __restrict__ y,
                                            int n,
                                            int k,
                                            int m,
                                            int epi,
                                            float a,
                                            float b) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    // ★ 2026-10-22 grid.y 行分块：blockIdx.y 每块接 M 行（m 任意大）。有效行的累加
    // 次序与旧版逐字节一致（acc[r] 的 j 序列、归约树不变）⇒ 逐位等价；并发块共享
    // 权重读 ⇒ DRAM 流量 ⌈m/M⌉ 遍 → ~1 遍。grid.y==1 且 m==M ⇒ 与旧版完全一致。
    const int r0 = blockIdx.y * M;
    if (r0 >= m) {
        return;
    }
    const bf16* wr = w + (size_t)warp * (size_t)k;
    float acc[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        acc[r] = 0.0f;
    }
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            if (r0 + r < m) {
                acc[r] += wv * __bfloat162float(x[(size_t)(r0 + r) * (size_t)k + j]);
            }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            if (r0 + r >= m) {
                continue;
            }
            float v = acc[r];
            if (epi == 1) {
                const float t = v * a;
                v = t / (1.0f + __expf(-t));
            } else if (epi == 2) {
                const float t = v * a;
                v = (1.0f / (1.0f + __expf(-t))) * b;
            } else if (epi == 3) {
                v = v * a + b;
            }
            y[(size_t)(r0 + r) * (size_t)n + warp] = __float2bfloat16(v);
        }
    }
}

#define LINGXI_SKINNY_M(M)                                                              \
    extern "C" __global__ void lingxi_gemm_skinny_m##M(                                \
        const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y,   \
        int n, int k, int m, int epi, float a, float b) {                               \
        skinny_body<M>(w, x, y, n, k, m, epi, a, b);                                    \
    }

LINGXI_SKINNY_M(1)
LINGXI_SKINNY_M(2)
LINGXI_SKINNY_M(3)
LINGXI_SKINNY_M(4)
LINGXI_SKINNY_M(5)
LINGXI_SKINNY_M(6)
LINGXI_SKINNY_M(7)
LINGXI_SKINNY_M(8)

// ★ 2026-10-27：`skinny_body<8>` 的**小 k 大 m 宽行版**（mwide）。
// 目标形状：mix_up [n=20480, k=320, m=512]——旧核 grid.y=m/8=64 分块，权重 DRAM 重读 64 遍
// （832MB ⇒ 实测 2.60ms/层）。本核把每 warp 的行块放大到 MC=32 行（acc 全驻寄存器）：
// CTA = 4 warp = 4 输出行；grid = (ceil(n/4), ceil(m/MC))；**零 smem**——x 直接从
// L1/L2 读（行块 20KB 驻 L1，w 行 640B 广播读），不占 smem ⇒ occupancy 不降。
// （v1 曾用 x 预转 f32 进动态 smem：staging 开销 ≈ 算力且 40KB/block 把 occupancy
// 压到 12.5%，实测 3× 更慢——负结果留档于此。）
// **逐位等价**（判据不需"有损档"验收）：每个输出元素的累加 = lane 步进 j 偏分
// （j = lane, lane+32, … 升序串行）+ 16/8/4/2/1 shuffle 归约树，与 skinny_body 逐字相同；
// 行分块只改"哪些行在一个 warp 里算"，每行独立 ⇒ 任意行划分逐位不变。
// 门控：gpu.rs `skinny_call` 仅在 `k ≤ 4096`（L1 行块预算）且 `m ≥ 64` 时分流；
// `LINGXI_NO_MWIDE=1` 回退旧核。
#define SKINNY_MW_MC 32
extern "C" __global__ void __launch_bounds__(128)
lingxi_gemm_skinny_mwide_bf16(const bf16* __restrict__ w,
                              const bf16* __restrict__ x,
                              bf16* __restrict__ y,
                              int n, int k, int m, int epi,
                              float a, float b) {
    const int warp = (int)(blockIdx.x * (blockDim.x >> 5)) + ((int)threadIdx.x >> 5);
    const int lane = (int)threadIdx.x & 31;
    if (warp >= n) {
        return;
    }
    const int r0 = blockIdx.y * SKINNY_MW_MC;
    const int mrows = min(SKINNY_MW_MC, m - r0);
    if (mrows <= 0) {
        return;
    }
    const bf16* wr = w + (size_t)warp * (size_t)k;
    const bf16* xb = x + (size_t)r0 * (size_t)k;
    float acc[SKINNY_MW_MC];
#pragma unroll
    for (int r = 0; r < SKINNY_MW_MC; ++r) {
        acc[r] = 0.0f;
    }
    const bool full = (mrows == SKINNY_MW_MC);
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
        if (full) {
#pragma unroll
            for (int r = 0; r < SKINNY_MW_MC; ++r) {
                acc[r] += wv * __bfloat162float(xb[(size_t)r * (size_t)k + (size_t)j]);
            }
        } else {
#pragma unroll 4
            for (int r = 0; r < SKINNY_MW_MC; ++r) {
                if (r < mrows) {
                    acc[r] += wv * __bfloat162float(xb[(size_t)r * (size_t)k + (size_t)j]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < SKINNY_MW_MC; ++r) {
        if (!full && r >= mrows) {
            continue;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
        if (lane == 0) {
            float v = acc[r];
            if (epi == 1) {
                const float t = v * a;
                v = t / (1.0f + __expf(-t));
            } else if (epi == 2) {
                const float t = v * a;
                v = (1.0f / (1.0f + __expf(-t))) * b;
            } else if (epi == 3) {
                v = v * a + b;
            }
            y[(size_t)(r0 + r) * (size_t)n + (size_t)warp] = __float2bfloat16(v);
        }
    }
}

// ★ 2026-10-24：`skinny_body<8>` 的 **grid.y==1 扁平成**（1 ≤ m ≤ 8）。
//
// 算式、线程映射、累加次序、归约树与 `lingxi_gemm_skinny_m8` **逐字相同**（⇒ 逐位相同），
// 只去掉两样纯取数/地址侧的东西：
// * `r0 = blockIdx.y * 8`（grid.y 恒为 1 ⇒ 恒 0）与其 `r0 + r < m` 谓词；
// 行偏移因此是编译期常量 `r*k`，内层只剩「基址 + j」，不再每轮重建 8 个 64 位地址；
// * `mt` 的运行期截断（用 `r < m`）。
// 实测（`tests/t15_bf16_skinny_bw.rs` 同进程 6 轮交替，m=6）：
// n=10240 k=320：42.5 → **28.0 µs**（154 → 234 GB/s，1.52×）
// n=2560 k=640：25.0 → **12.9 µs**（131 → 254 GB/s，1.94×）
// （原始 `skinny_body<8>` 逐字复刻 = 47.2 / 18.0 µs ⇒ 生产入口与逐字复刻一致，
// 差异确实来自上面的行分块形态，不是测量假象：同一进程内交替 6 轮）。
// 门控：`LINGXI_NO_BF16_FLAT=1` 回退到 `lingxi_gemm_skinny_m8`（见 wrappers）。
extern "C" __global__ void lingxi_gemm_skinny_m8_flat_bf16(const bf16* __restrict__ w,
                                                           const bf16* __restrict__ x,
                                                           bf16* __restrict__ y,
                                                           int n, int k, int m, int epi,
                                                           float a, float b) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n || m <= 0) {
        return;
    }
    const bf16* wr = w + (size_t)warp * (size_t)k;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r] = 0.0f;
    }
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < m) {
                acc[r] += wv * __bfloat162float(x[(size_t)r * (size_t)k + j]);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= m) {
                break;
            }
            float v = acc[r];
            if (epi == 1) {
                const float t = v * a;
                v = t / (1.0f + __expf(-t));
            } else if (epi == 2) {
                const float t = v * a;
                v = (1.0f / (1.0f + __expf(-t))) * b;
            } else if (epi == 3) {
                v = v * a + b;
            }
            y[(size_t)r * (size_t)n + warp] = __float2bfloat16(v);
        }
    }
}

// ★ 2026-10-24 **shared expert 融合核 1/2**：gate_up 的 gate 行 i 与 up 行 i+I 由**同一
// block** 算（各 8 个 token 累加器），归约后各自 `__float2bfloat16` round 一次（= 原路径
// 写 bf16 tmp 的舍入点），再做 `silu(g)*u`（g/u 都从 bf16 读回 f32，= `lingxi_swiglu_bf16`
// 的算法）⇒ 与 [gemm_skinny_row(epi=0) → bf16 tmp → lingxi_swiglu_bf16] **逐位相同**。
// blockIdx.x == inter 的那一块顺带算 shared_gate 行（n=1、epi=2，与
// `lingxi_gemm_skinny_row_bf16` 同线程映射/同归约树）⇒ 省掉一次独立 launch。
// 每行/每 token 的列划分、累加次序、两级 warp 归约树均与 `lingxi_gemm_skinny_row_bf16` 逐字一致；
// grid.y 仍按 8 token 行分块（与 skinny_row 同一改动）。
extern "C" __global__ void lingxi_gemm_skinny_row_gu_swiglu_sg_bf16(
    const bf16* __restrict__ w, const bf16* __restrict__ wsg,
    const bf16* __restrict__ x, bf16* __restrict__ y, bf16* __restrict__ sg,
    int inter, int k, int m, float a, float b) {
    const int row = blockIdx.x;
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const int mt = ((m - r0) < 8) ? (m - r0) : 8;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    __shared__ float sred_g[32][8];
    __shared__ float sred_u[32][8];

    if (row >= inter) {
        // ── shared_gate 行（n=1，epi=2：sigmoid(v*a)*b）──
        float acc[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            acc[s] = 0.0f;
        }
        int c0 = tid * 8;
        for (; c0 + 8 <= k; c0 += (int)blockDim.x * 8) {
            const int4 wv = *reinterpret_cast<const int4*>(wsg + c0);
            const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= mt) {
                    break;
                }
                const int4 xv =
                    *reinterpret_cast<const int4*>(x + (size_t)(r0 + s) * (size_t)k + c0);
                const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                float d = 0.0f;
#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    d += __bfloat162float(wp[z]) * __bfloat162float(xp[z]);
                }
                acc[s] += d;
            }
        }
        for (; c0 < k; c0 += (int)blockDim.x * 8) {
            for (int i = 0; i < 8 && c0 + i < k; ++i) {
                const float wf = __bfloat162float(wsg[c0 + i]);
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s >= mt) {
                        break;
                    }
                    acc[s] += wf * __bfloat162float(x[(size_t)(r0 + s) * (size_t)k + c0 + i]);
                }
            }
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= mt) {
                break;
            }
            float v = acc[s];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                sred_g[warp][s] = v;
            }
        }
        __syncthreads();
        if (warp == 0) {
            for (int s = 0; s < mt; ++s) {
                float v = (lane < nwarp) ? sred_g[lane][s] : 0.0f;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    v += __shfl_down_sync(0xffffffffu, v, off);
                }
                if (lane == 0) {
                    const float t2 = v * a;
                    sg[r0 + s] = __float2bfloat16((1.0f / (1.0f + __expf(-t2))) * b);
                }
            }
        }
        return;
    }

    // ── gate 行 row 与 up 行 row+inter ──
    const bf16* wg = w + (size_t)row * (size_t)k;
    const bf16* wu = w + (size_t)(row + inter) * (size_t)k;
    float accg[8];
    float accu[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        accg[s] = 0.0f;
        accu[s] = 0.0f;
    }
    int c0 = tid * 8;
    for (; c0 + 8 <= k; c0 += (int)blockDim.x * 8) {
        const int4 wgv = *reinterpret_cast<const int4*>(wg + c0);
        const int4 wuv = *reinterpret_cast<const int4*>(wu + c0);
        const bf16* wgp = reinterpret_cast<const bf16*>(&wgv);
        const bf16* wup = reinterpret_cast<const bf16*>(&wuv);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= mt) {
                break;
            }
            const int4 xv = *reinterpret_cast<const int4*>(x + (size_t)(r0 + s) * (size_t)k + c0);
            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
            float dg = 0.0f;
            float du = 0.0f;
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                const float xf = __bfloat162float(xp[z]);
                dg += __bfloat162float(wgp[z]) * xf;
                du += __bfloat162float(wup[z]) * xf;
            }
            accg[s] += dg;
            accu[s] += du;
        }
    }
    for (; c0 < k; c0 += (int)blockDim.x * 8) {
        for (int i = 0; i < 8 && c0 + i < k; ++i) {
            const float wgf = __bfloat162float(wg[c0 + i]);
            const float wuf = __bfloat162float(wu[c0 + i]);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (s >= mt) {
                    break;
                }
                const float xf = __bfloat162float(x[(size_t)(r0 + s) * (size_t)k + c0 + i]);
                accg[s] += wgf * xf;
                accu[s] += wuf * xf;
            }
        }
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= mt) {
            break;
        }
        float vg = accg[s];
        float vu = accu[s];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            vg += __shfl_down_sync(0xffffffffu, vg, off);
            vu += __shfl_down_sync(0xffffffffu, vu, off);
        }
        if (lane == 0) {
            sred_g[warp][s] = vg;
            sred_u[warp][s] = vu;
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int s = 0; s < mt; ++s) {
            float vg = (lane < nwarp) ? sred_g[lane][s] : 0.0f;
            float vu = (lane < nwarp) ? sred_u[lane][s] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                vg += __shfl_down_sync(0xffffffffu, vg, off);
                vu += __shfl_down_sync(0xffffffffu, vu, off);
            }
            if (lane == 0) {
                // 两次 round（= 写/读 bf16 tmp 的舍入点），再按 swiglu 的 f32 算式。
                const float g = __bfloat162float(__float2bfloat16(vg));
                const float u = __bfloat162float(__float2bfloat16(vu));
                y[(size_t)(r0 + s) * (size_t)inter + row] =
                    __float2bfloat16((g / (1.0f + __expf(-g))) * u);
            }
        }
    }
}

// ★ 2026-10-24 **shared expert 融合核 2/2**：`lingxi_gemm_skinny_m8`（dn 投影）+ row_scale 尾。
// 逐位等价：先 `__float2bfloat16(acc)`（= 原路径写 `shared` 的舍入），再
// `bf16 * bf16 → f32 → bf16`（= `lingxi_row_scale_inplace_bf16` 的算式）。
extern "C" __global__ void lingxi_gemm_skinny_m8_scale_bf16(
    const bf16* __restrict__ w, const bf16* __restrict__ x,
    bf16* __restrict__ y, const bf16* __restrict__ s,
    int n, int k, int m) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const int r0 = blockIdx.y * 8;
    if (r0 >= m) {
        return;
    }
    const bf16* wr = w + (size_t)warp * (size_t)k;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r] = 0.0f;
    }
    for (int j = lane; j < k; j += 32) {
        const float wv = __bfloat162float(wr[j]);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r0 + r < m) {
                acc[r] += wv * __bfloat162float(x[(size_t)(r0 + r) * (size_t)k + j]);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r0 + r >= m) {
                continue;
            }
            const bf16 v = __float2bfloat16(acc[r]);
            y[(size_t)(r0 + r) * (size_t)n + warp] =
                __float2bfloat16(__bfloat162float(v) * __bfloat162float(s[r0 + r]));
        }
    }
}

// PLE 门控（替代 host 参考实现）：每个 (token i, group g) 块算一个点积，再过
// 有符号开方 + sigmoid 得到门控标量，最后用它缩放 value[i,:]。
// gate = dot(key_n[blk], query_n[blk]) * inv
// s = sign(gate); g2 = s * sqrt(max(|gate|, 1e-6)); sg = sigmoid(g2)
// out[blk, j] = sg * value[i, j]
// 原实现每层要 3 次 D2H + 1 次 H2D（host 上做同样的事），48 层/ token 累积是 decode 的大头。
extern "C" __global__ void lingxi_ple_gate_bf16(const bf16* __restrict__ key_n,
                                                const bf16* __restrict__ query_n,
                                                const bf16* __restrict__ value,
                                                bf16* __restrict__ out,
                                                int hc,
                                                int hidden,
                                                float inv) {
    const int blk = blockIdx.x; // = i * hc + g
    const int i = blk / hc;
    const bf16* kk = key_n + (size_t)blk * (size_t)hidden;
    const bf16* qq = query_n + (size_t)blk * (size_t)hidden;

    float dot = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        dot += __bfloat162float(kk[j]) * __bfloat162float(qq[j]);
    }

    __shared__ float sred[32];
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)(threadIdx.x >> 5);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, off);
    }
    if (lane == 0) {
        sred[warp] = dot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = (int)((blockDim.x + 31) >> 5);
        float acc = 0.0f;
        for (int w = 0; w < nw; ++w) {
            acc += sred[w];
        }
        sred[0] = acc;
    }
    __syncthreads();
    const float gate = sred[0] * inv;
    const float s = (gate > 0.0f) ? 1.0f : ((gate < 0.0f) ? -1.0f : 0.0f);
    const float g2 = s * sqrtf(fmaxf(fabsf(gate), 1e-6f));
    const float sg = 1.0f / (1.0f + expf(-g2));

    const bf16* vv = value + (size_t)i * (size_t)hidden;
    bf16* oo = out + (size_t)blk * (size_t)hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        oo[j] = __float2bfloat16(sg * __bfloat162float(vv[j]));
    }
}

// 行缩放（广播）：x[row,:] *= s[row]
extern "C" __global__ void lingxi_row_scale_inplace_bf16(bf16* __restrict__ x,
                                                         const bf16* __restrict__ s,
                                                         int rows,
                                                         int cols) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)rows * cols) {
        const int row = (int)(i / cols);
        x[i] = __float2bfloat16(__bfloat162float(x[i]) * __bfloat162float(s[row]));
    }
}

// y = a + b（逐元素）
extern "C" __global__ void lingxi_add_bf16(const bf16* __restrict__ a,
                                           const bf16* __restrict__ b,
                                           bf16* __restrict__ y,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}
