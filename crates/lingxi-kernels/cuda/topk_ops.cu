// 每行 argmax / top-K（bf16 logits，GPU 侧完成，免去 t×vocab 的 D2H + host 遍历）。
//
// 位等价口径（与 host `argmax_rows_bf16` / `topk_bf16_row` 完全一致）：
//   比较键 = (f32 值 **降序**, 下标 **升序**) 的**全序** ⇒ top-K 选择结果唯一，
//   任何按同一全序做正确选择/合并的实现都逐位相同。bf16→f32 转换精确且保序；
//   NaN 在两侧都"严格比较恒假" ⇒ 永远不会胜出，行为一致。

#include <cuda_bf16.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

// ── glibc expf 的**逐位复刻**（A2 设备路由用） ──────────────────────────────
// 为什么必须复刻：host 路由用的是 glibc 2.41 的 `expf`，它**不是正确舍入**的
// （实测 200 万个 [-20,0] 随机值里 1277 个与「f64 exp 再舍入」差 1 ulp）⇒ 直接在核里
// 用 exp(double) 会让 top-k 权重差 1 ulp，进而改变贪心 token 流。
// 移植自 glibc `sysdeps/ieee754/flt-32/e_expf.c` + `e_exp2f_data.c`（Arm optimized-routines，
// 无 FMA）。⚠️ 必须用 `__d*_rn`（双精度内建）；`__fmul_rn/__fadd_rn` 是 **float** 版
// （实测会把整个多项式算成单精度，结果错得离谱）。
// 4M 随机输入（[-100,0]，偏向 [-40,0]）对比 host expf：**0 处不一致**。
__constant__ uint64_t EXP2F_TAB[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL, 0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL, 0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL, 0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};

__device__ __forceinline__ float lingxi_glibc_expf(float x) {
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
    uint64_t t = EXP2F_TAB[ki % 32];
    t += ki << (52 - 5);
    const double s = __longlong_as_double(t);
    const double zz = __dadd_rn(__dmul_rn(c0, r), c1);
    const double r2 = __dmul_rn(r, r);
    double y = __dadd_rn(__dmul_rn(c2, r), 1.0);
    y = __dadd_rn(__dmul_rn(zz, r2), y);
    y = __dmul_rn(y, s);
    return (float)y;
}

// ── top-K（K ≤ 32）：两段式 ──
// 全序 = (f32 值 **降序**, 下标 **升序**)——与 host `topk_bf16_row` 的选择结果**唯一且逐位相同**
// （host 语义：新元素只在「严格大于当前第 K 名」时进入，并插到**第一个严格小于它的元素之前**
// ⇒ 同值元素按下标升序排列 ⇒ 正是该全序的前 K 名）。任何按同一全序正确选择/合并的实现
// 都逐位一致。NaN 两侧都"严格比较恒假"⇒ 永不中选；非有限值会计进 `bad`，host 侧报错
// （与 host 的 `ensure!(is_finite)` 同效）。
//
// part：grid (g, rows) block 128。每线程对自己的 strided 子段维护一个局部降序 top-k
//       （数组全部经 `#pragma unroll + break` 以编译期下标访问，防 local memory 化——
//        与 slotsm 尾部循环同一纪律）；再树形合并 128 个列表（每轮半数线程两两归并）。
// fin ：grid (rows) block 128。把 g 个部分列表同样树形归并成最终 top-k 写出。
// 哨兵 = (-inf, INT_MAX)：参与比较但永远打不过真值；vocab ≥ k 时不会留在结果里。

#define TOPK_KMAX 32

__device__ __forceinline__ void topk_better(float av, int ai, float bv, int bi,
                                            float* ov, int* oi) {
    if (bv > av || (bv == av && bi < ai)) {
        *ov = bv;
        *oi = bi;
    } else {
        *ov = av;
        *oi = ai;
    }
}

// 局部降序 top-k 插入：仅当「未满 或 严格大于当前第 k 名」时进入。
__device__ __forceinline__ void topk_insert(float* lv, int* li, int& cnt, int k,
                                            float v, int idx) {
    if (cnt == k && !(v > lv[k - 1])) {
        return;
    }
    int pos = cnt;  // 第一个 v > lv[s] 的槽；无（追加）则为 cnt
#pragma unroll
    for (int s = 0; s < TOPK_KMAX; ++s) {
        if (s >= cnt) {
            break;
        }
        if (v > lv[s]) {
            pos = s;
            break;
        }
    }
    const int newcnt = min(cnt + 1, k);
    // [pos+1, newcnt) 右移一格（j 编译期，pos/newcnt 运行期 → break/continue）
#pragma unroll
    for (int j = TOPK_KMAX - 1; j >= 1; --j) {
        if (j >= newcnt) {
            continue;
        }
        if (j <= pos) {
            break;
        }
        lv[j] = lv[j - 1];
        li[j] = li[j - 1];
    }
    if (pos < newcnt) {
        lv[pos] = v;
        li[pos] = idx;
    }
    cnt = newcnt;
}

// 两个降序 k 列表（含哨兵尾）归并出降序 top-k，写入 (ov, oi)。
__device__ __forceinline__ void topk_merge(const float* av, const int* ai,
                                           const float* bv, const int* bi,
                                           float* ov, int* oi, int k) {
    int ap = 0;
    int bp = 0;
#pragma unroll
    for (int j = 0; j < TOPK_KMAX; ++j) {
        if (j >= k) {
            break;
        }
        topk_better(av[ap], ai[ap], bv[bp], bi[bp], &ov[j], &oi[j]);
        // 推进被选中的那一边的游标。真值 (值,下标) 在全序下唯一 ⇒ 两边同头只可能是
        // 哨兵重复；此时确定性推 a，保证结果与归并顺序无关。
        const bool is_a = (ov[j] == av[ap]) && (oi[j] == ai[ap]);
        const bool is_b = (ov[j] == bv[bp]) && (oi[j] == bi[bp]);
        if (is_b && !is_a) {
            bp++;
        } else {
            ap++;
        }
    }
}

extern "C" __global__ void lingxi_topk_rows_part_bf16(
    const bf16* __restrict__ x,
    float* __restrict__ pval,
    int* __restrict__ pidx,
    int* __restrict__ bad,
    int vocab,
    int g,
    int k) {
    const int row = blockIdx.y;
    const int chunk = (vocab + g - 1) / g;
    const int begin = blockIdx.x * chunk;
    const int end = min(begin + chunk, vocab);
    const bf16* xr = x + (size_t)row * (size_t)vocab;

    float lv[TOPK_KMAX];
    int li[TOPK_KMAX];
    int cnt = 0;
    for (int i = begin + (int)threadIdx.x; i < end; i += (int)blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        if (!isfinite(v)) {
            atomicAdd(bad, 1);
            continue;
        }
        topk_insert(lv, li, cnt, k, v, i);
    }
    // 不足 k 个的列表用哨兵补齐（chunk 小 / 空块时），保证归并逻辑统一
    for (int s = cnt; s < k; ++s) {
        lv[s] = -__int_as_float(0x7f800000);
        li[s] = 0x7fffffff;
    }

    __shared__ float sv[128][TOPK_KMAX];
    __shared__ int si[128][TOPK_KMAX];
#pragma unroll
    for (int s = 0; s < TOPK_KMAX; ++s) {
        if (s >= k) {
            break;
        }
        sv[threadIdx.x][s] = lv[s];
        si[threadIdx.x][s] = li[s];
    }
    __syncthreads();

    // 树形归并：128 → 64 → … → 1
    for (int half = 64; half > 0; half >>= 1) {
        if (threadIdx.x < (unsigned)half) {
            const int t = threadIdx.x;
            float tv[TOPK_KMAX];
            int ti[TOPK_KMAX];
            topk_merge(sv[t], si[t], sv[t + half], si[t + half], tv, ti, k);
#pragma unroll
            for (int s = 0; s < TOPK_KMAX; ++s) {
                if (s >= k) {
                    break;
                }
                sv[t][s] = tv[s];
                si[t][s] = ti[s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float* pv = pval + ((size_t)row * g + blockIdx.x) * k;
        int* pi = pidx + ((size_t)row * g + blockIdx.x) * k;
        for (int s = 0; s < k; ++s) {
            pv[s] = sv[0][s];
            pi[s] = si[0][s];
        }
    }
}

extern "C" __global__ void lingxi_topk_rows_fin(
    const float* __restrict__ pval,
    const int* __restrict__ pidx,
    float* __restrict__ oval,
    int* __restrict__ oidx,
    int g,
    int k) {
    const int row = blockIdx.x;
    __shared__ float sv[128][TOPK_KMAX];
    __shared__ int si[128][TOPK_KMAX];
    const int tid = (int)threadIdx.x;
    // g 个部分列表装进 smem（g ≤ 128；不足的槽给哨兵）
    if (tid < g) {
        const float* pv = pval + ((size_t)row * g + tid) * k;
        const int* pi = pidx + ((size_t)row * g + tid) * k;
#pragma unroll
        for (int s = 0; s < TOPK_KMAX; ++s) {
            if (s >= k) {
                break;
            }
            sv[tid][s] = pv[s];
            si[tid][s] = pi[s];
        }
    } else {
#pragma unroll
        for (int s = 0; s < TOPK_KMAX; ++s) {
            if (s >= k) {
                break;
            }
            sv[tid][s] = -__int_as_float(0x7f800000);
            si[tid][s] = 0x7fffffff;
        }
    }
    __syncthreads();
    for (int half = 64; half > 0; half >>= 1) {
        if (tid < half) {
            float tv[TOPK_KMAX];
            int ti[TOPK_KMAX];
            topk_merge(sv[tid], si[tid], sv[tid + half], si[tid + half], tv, ti, k);
#pragma unroll
            for (int s = 0; s < TOPK_KMAX; ++s) {
                if (s >= k) {
                    break;
                }
                sv[tid][s] = tv[s];
                si[tid][s] = ti[s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        float* ov = oval + (size_t)row * k;
        int* oi = oidx + (size_t)row * k;
        for (int s = 0; s < k; ++s) {
            ov[s] = sv[0][s];
            oi[s] = si[0][s];
        }
    }
}


// ── argmax：两段式 ──
// part：grid (g, rows)，block 128。每块把自己那 chunk（连续段）扫成一个 (max, 最小下标)，
//       写 partials。chunk 内线程按 128 步长交错扫（访存合并）。
// fin ：grid (rows)，block 128。把 g 个 partial 归约成 1 个，写 out[row]。

__device__ __forceinline__ void better(float av, int ai, float bv, int bi,
                                       float* ov, int* oi) {
    // 全序比较：b 严格更好 = 值更大，或值相等且下标更小
    if (bv > av || (bv == av && bi < ai)) {
        *ov = bv;
        *oi = bi;
    } else {
        *ov = av;
        *oi = ai;
    }
}

extern "C" __global__ void lingxi_argmax_rows_part_bf16(
    const bf16* __restrict__ x,
    float* __restrict__ pval,
    int* __restrict__ pidx,
    int vocab,
    int g) {
    const int row = blockIdx.y;
    const int chunk = (vocab + g - 1) / g;
    const int begin = blockIdx.x * chunk;
    const int end = min(begin + chunk, vocab);
    const bf16* xr = x + (size_t)row * (size_t)vocab;

    float bv = -__int_as_float(0x7f800000);  // -inf
    int bi = begin;                           // 平局取下标最小 ⇒ 初值取本 chunk 首下标
    bool seen = false;
    for (int i = begin + (int)threadIdx.x; i < end; i += (int)blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        if (!seen) {
            bv = v;
            bi = i;
            seen = true;
        } else {
            float ov;
            int oi;
            better(bv, bi, v, i, &ov, &oi);
            bv = ov;
            bi = oi;
        }
    }
    if (!seen) {
        bv = -__int_as_float(0x7f800000);
        bi = begin;
    }

    // block 归约（smem，值+下标交错两数组）
    __shared__ float sv[128];
    __shared__ int si[128];
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            float ov;
            int oi;
            better(sv[threadIdx.x], si[threadIdx.x],
                   sv[threadIdx.x + off], si[threadIdx.x + off], &ov, &oi);
            sv[threadIdx.x] = ov;
            si[threadIdx.x] = oi;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pval[(size_t)row * g + blockIdx.x] = sv[0];
        pidx[(size_t)row * g + blockIdx.x] = si[0];
    }
}

extern "C" __global__ void lingxi_argmax_rows_fin(
    const float* __restrict__ pval,
    const int* __restrict__ pidx,
    int* __restrict__ out,
    int g) {
    const int row = blockIdx.x;
    __shared__ float sv[128];
    __shared__ int si[128];
    const int tid = (int)threadIdx.x;
    float bv = -__int_as_float(0x7f800000);
    int bi = 0x7fffffff;
    // g ≤ 1024：前 128 线程各吃 ceil(g/128) 个 partial 再做块归约
    const int per = (g + 127) / 128;
    for (int j = 0; j < per; ++j) {
        const int slot = tid * per + j;
        if (slot < g) {
            const float v = pval[(size_t)row * g + slot];
            const int i = pidx[(size_t)row * g + slot];
            if (i != 0x7fffffff) {
                float ov;
                int oi;
                better(bv, bi, v, i, &ov, &oi);
                bv = ov;
                bi = oi;
            }
        }
    }
    sv[tid] = bv;
    si[tid] = (bi == 0x7fffffff) ? 0 : bi;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) {
        if (tid < off) {
            float ov;
            int oi;
            better(sv[tid], si[tid], sv[tid + off], si[tid + off], &ov, &oi);
            sv[tid] = ov;
            si[tid] = oi;
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = si[0];
    }
}

// ─────────────────────── ㊺ A2 设备路由：logits → top-k → 槽位查表（单 block 全 GPU） ───────────────────────
//
// 把 MoE 每层的「router GEMV → 同步 → host softmax/topk → host 查槽 → staging」整条
// host 决策链压成一个核：数学**逐位复刻** moe.rs 的 host 路径——
//   e_i = expf(v_i − m)（exp 走 f64 路径对齐 glibc expf 的正确舍入；v_i−m 与 host 同为 f32）；
//   sum = Σe 按 i 升序**串行**（thread0，与 host 同序同值）；p_i = e_i / sum（IEEE 除法）；
//   选择 = 插入式 top-k（严格 `>`、全序 (p 降序, 下标升序)）；norm = Σp_选中 按**专家升序**串行；
//   w = p / norm。输出全部按专家升序（与 host `sel.sort_by_key` 后的组序一致 ⇒ combine 归并次序相同）。
// miss（slot_map[e]==0 ⇒ 槽内写 -1）时 *miss_flag=1 并记录缺专家 id —— host 见 flag 即整体回退 legacy。
extern "C" __global__ void __launch_bounds__(256, 1) lingxi_route_topk_slots_bf16(
    const bf16* __restrict__ logits,      // [E]
    const int* __restrict__ slot_map,     // [E]（slot+1，0=未缓存）
    int* __restrict__ ids,                // [top_k] 固定槽（诊断）
    float* __restrict__ wts,              // [top_k] 固定槽
    int* __restrict__ slot_ids,           // [top_k] 固定槽 = 图体 (gkey,5) 的 qin
    int* __restrict__ rows,               // [top_k] 固定槽（全 0）
    int* __restrict__ host_slots,         // mapped host i32[top_k]：每个选中专家的槽（-1=miss）
    unsigned int* __restrict__ miss_flag, // mapped host u32：**未命中个数**（0=全中）
    int* __restrict__ miss_ids,           // mapped host i32[top_k]（紧凑、专家升序）
    int E, int top_k) {
    __shared__ float p[1024];       // 概率（E ≤ 1024）
    __shared__ unsigned char in_sel[1024];
    __shared__ int sel_idx[72];     // top_k ≤ 64，+1 给插入移位
    __shared__ int srank[1024];     // 每个元素的排名（并行 top-k 用）
    __shared__ float red[32];
    __shared__ float s_m, s_sum;
    const int tid = (int)threadIdx.x;
    const int nth = (int)blockDim.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    // 1) f32 转换 + block max（max 与归约次序无关，精确）
    // ⚠️ E（本模型 512）**可以大于 blockDim（256）** ⇒ 必须跨步循环，不能只处理 tid<E。
    float lv = -INFINITY;
    for (int i = tid; i < E; i += nth) lv = fmaxf(lv, __bfloat162float(logits[i]));
    float mx = lv;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < (nth >> 5)) ? red[lane] : -INFINITY;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
        if (lane == 0) s_m = v;
    }
    __syncthreads();
    const float m = s_m;
    // 2) exp（**glibc expf 逐位复刻**——见文件头的说明）
    for (int i = tid; i < E; i += nth) {
        p[i] = lingxi_glibc_expf(__bfloat162float(logits[i]) - m);
    }
    __syncthreads();
    // 3) 串行和（thread0，升序——与 host 同序同值）
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < E; ++i) s += p[i];
        s_sum = s;
    }
    __syncthreads();
    const float sum = s_sum;
    // 诊断（`LINGXI_ROUTE_DEV_CHECK=1` 读）：把 m / sum 的位模式塞进 miss_ids[0..2]。
    // 有 miss 时 miss_ids[0..nmiss) 会被覆盖，而诊断只在全中时读 ⇒ 无冲突。
    if (tid == 0) {
        miss_ids[0] = __float_as_int(m);
        miss_ids[1] = __float_as_int(sum);
    }
    // 4) p_i = e_i / sum（IEEE 除法按位确定）
    for (int i = tid; i < E; i += nth) {
        p[i] = p[i] / sum;
        in_sel[i] = 0u;
    }
    __syncthreads();
    // 5) **并行** top-k：每个元素算"更优于它的元素个数"（全序 = p 降序、下标升序）。
    //    旧版是 thread0 的插入式选择（512 次 × 最多 10 次移位，全在 shared 上串行依赖）
    //    ——实测 ~0.1 ms/层、24 层 ×2 卡 = 每 token 10 ms 级，比 host 路径还慢。
    //    这里改成 E×E 广播比较：所有线程同一时刻读同一个 p[j]（broadcast，无 bank 冲突），
    //    512 次循环 ≈ 数千周期。排名 → 专家下标由 thread0 回收（E 次扫描，很便宜）。
    //    ⚠️ 语义仍是 host 的**总序**，选择集合与顺序逐位相同。
    {
        for (int i = tid; i < E; i += nth) {
            const float pi = p[i];
            int c = 0;
            for (int j = 0; j < E; ++j) {
                const float pj = p[j];
                if (pj > pi || (pj == pi && j < i)) ++c;
            }
            srank[i] = c;
            in_sel[i] = (c < top_k) ? 1u : 0u;
        }
    }
    __syncthreads();
    if (tid == 0) {
        // 6) 排名 → 专家下标（**降序序列**）；norm 必须沿这个顺序累加（与 host 同序）
        for (int r = 0; r < top_k; ++r) sel_idx[r] = -1;
        for (int i = 0; i < E; ++i) {
            if (in_sel[i]) sel_idx[srank[i]] = i;
        }
        float norm = 0.0f;
        for (int r = 0; r < top_k; ++r) norm += p[sel_idx[r]];
        miss_ids[2] = __float_as_int(norm);
        // 7) 输出（专家升序）：w = p/norm；槽位直查；miss 上报
        unsigned nmiss = 0u;
        int w = 0;
        for (int i = 0; i < E && w < top_k; ++i) {
            if (!in_sel[i]) continue;
            const int slm = slot_map[i];            // slot+1，0=miss
            const int slot = (slm > 0) ? (slm - 1) : -1;
            ids[w] = i;
            wts[w] = p[i] / norm;
            slot_ids[w] = slot;
            host_slots[w] = slot;
            rows[w] = 0;
            if (slm <= 0) miss_ids[nmiss++] = i;
            ++w;
        }
        *miss_flag = nmiss;
    }
}
