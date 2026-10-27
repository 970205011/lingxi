// 每行 argmax / top-K（bf16 logits，GPU 侧完成，免去 t×vocab 的 D2H + host 遍历）。
//
// 位等价口径（与 host `argmax_rows_bf16` / `topk_bf16_row` 完全一致）：
// 比较键 = (f32 值 **降序**, 下标 **升序**) 的**全序** ⇒ top-K 选择结果唯一，
// 任何按同一全序做正确选择/合并的实现都逐位相同。bf16→f32 转换精确且保序；
// NaN 在两侧都"严格比较恒假" ⇒ 永远不会胜出，行为一致。

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
// （数组全部经 `#pragma unroll + break` 以编译期下标访问，防 local memory 化——
// 与 slotsm 尾部循环同一纪律）；再树形合并 128 个列表（每轮半数线程两两归并）。
// fin：grid (rows) block 128。把 g 个部分列表同样树形归并成最终 top-k 写出。
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


// ── top-K part 的**寄存器版**（2026-10-24，`LINGXI_NO_TOPKROWS_FAST=1` 可回退） ──
//
// 为什么另起一个核而**不动上面两个已提交的核**：
// `spec.rs` 的草稿 head 每步都要对 vocab=248320 做一次 top-20（t=1 时只有 64 个 block）。
// 微基准实测 `part+fin` = **155.7 µs**（`tests/t7_mtp_shape_bw.rs`）——对 0.5 MB 的输入来说
// 差 2 个数量级。根因：旧 `topk_insert` 用**运行期下标** `lv[pos]` / 移位循环写局部数组
// ⇒ `lv/li` 落 local memory，每线程 30 个元素 × 每次 ~20 条 local ld/st。
//
// 本核把局部插入改成**无动态下标**的「逐槽比较-交换」（槽号 s 是编译期常量 ⇒ 全在寄存器），
// 语义与 `topk_insert` **逐位一致**：
// * 只在**严格大于**时前移（同值保持"先来后到"⇒ 天然下标升序，与旧核/全序定义相同）；
// * 局部数组**预填哨兵** (-inf, INT_MAX) —— 与旧核 `for s in cnt..k` 的补齐值完全相同；
// * 线程映射、每线程扫的 strided 子段、128 路树形归并（`topk_merge`）**一字未改**
// ⇒ 输出 `pval/pidx` 与旧核逐位相同（判据见 `tests/t7_mtp_shape_bw.rs::t7_topk_part_bitid`）。
__device__ __forceinline__ void topk_insert_reg(float* lv, int* li, int k,
                                                float v, int idx) {
    // 与旧 `topk_insert` **同一套动作**（找第一个严格小于 v 的槽 → 右移一格 → 写入），
    // 只是把「运行期下标」全部换成「编译期槽号 s + 谓词」：
    // * 旧核的 `lv[pos] = v` / `lv[j] = lv[j-1]` 用运行期下标 ⇒ 局部数组整个落 local memory；
    // * 这里 s 是常量，赋值用 `if (s == pos)` / `if (s > pos)` 谓词完成 ⇒ 全在寄存器。
    // 语义一字未改（严格大于才前移、同值保持先后 ⇒ 下标升序），因此局部表逐位相同。
    int pos = k; // 不可插入时保持 k
#pragma unroll
    for (int s = 0; s < TOPK_KMAX; ++s) {
        const bool hit = (s < k) && (pos == k) && (v > lv[s]);
        if (hit) {
            pos = s;
        }
    }
#pragma unroll
    for (int s = TOPK_KMAX - 1; s >= 1; --s) {
        const bool mv = (s < k) && (s > pos);
        const float pv = lv[s - 1];
        const int pi = li[s - 1];
        if (mv) {
            lv[s] = pv;
            li[s] = pi;
        }
    }
#pragma unroll
    for (int s = 0; s < TOPK_KMAX; ++s) {
        if (s == pos) {
            lv[s] = v;
            li[s] = idx;
        }
    }
}

extern "C" __global__ void lingxi_topk_rows_part_fast_bf16(
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
    const float NEG_INF = -__int_as_float(0x7f800000);
#pragma unroll
    for (int s = 0; s < TOPK_KMAX; ++s) {
        lv[s] = NEG_INF;
        li[s] = 0x7fffffff;
    }
    // 4 路展开：4 个**独立** load 先发出去（打散依赖链；每线程只有 ~30 个元素、
    // 每 SM 只有 4 个 warp ⇒ 单靠 warp 级并行盖不住 HBM 延迟）。
    // 插入顺序仍是「下标升序」（i, i+128, i+256, i+384）⇒ 同值并列时的先后与逐元素版**相同**，
    // 因此局部表/输出逐位不变。
    const int stride = (int)blockDim.x;
    int i = begin + (int)threadIdx.x;
    for (; i + 3 * stride < end; i += 4 * stride) {
        const float v0 = __bfloat162float(xr[i]);
        const float v1 = __bfloat162float(xr[i + stride]);
        const float v2 = __bfloat162float(xr[i + 2 * stride]);
        const float v3 = __bfloat162float(xr[i + 3 * stride]);
        if (isfinite(v0)) { topk_insert_reg(lv, li, k, v0, i); } else { atomicAdd(bad, 1); }
        if (isfinite(v1)) { topk_insert_reg(lv, li, k, v1, i + stride); } else { atomicAdd(bad, 1); }
        if (isfinite(v2)) { topk_insert_reg(lv, li, k, v2, i + 2 * stride); } else { atomicAdd(bad, 1); }
        if (isfinite(v3)) { topk_insert_reg(lv, li, k, v3, i + 3 * stride); } else { atomicAdd(bad, 1); }
    }
    for (; i < end; i += stride) {
        const float v = __bfloat162float(xr[i]);
        if (!isfinite(v)) {
            atomicAdd(bad, 1);
            continue;
        }
        topk_insert_reg(lv, li, k, v, i);
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

// ── argmax：两段式 ──
// part：grid (g, rows)，block 128。每块把自己那 chunk（连续段）扫成一个 (max, 最小下标)，
// 写 partials。chunk 内线程按 128 步长交错扫（访存合并）。
// fin：grid (rows)，block 128。把 g 个 partial 归约成 1 个，写 out[row]。

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
// e_i = expf(v_i − m)（exp 走 f64 路径对齐 glibc expf 的正确舍入；v_i−m 与 host 同为 f32）；
// sum = Σe 按 i 升序**串行**（thread0，与 host 同序同值）；p_i = e_i / sum（IEEE 除法）；
// 选择 = 插入式 top-k（严格 `>`、全序 (p 降序, 下标升序)）；norm = Σp_选中 按**专家升序**串行；
// w = p / norm。输出全部按专家升序（与 host `sel.sort_by_key` 后的组序一致 ⇒ combine 归并次序相同）。
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
    // 旧版是 thread0 的插入式选择（512 次 × 最多 10 次移位，全在 shared 上串行依赖）
    // ——实测 ~0.1 ms/层、24 层 ×2 卡 = 每 token 10 ms 级，比 host 路径还慢。
    // 这里改成 E×E 广播比较：所有线程同一时刻读同一个 p[j]（broadcast，无 bank 冲突），
    // 512 次循环 ≈ 数千周期。排名 → 专家下标由 thread0 回收（E 次扫描，很便宜）。
    // ⚠️ 语义仍是 host 的**总序**，选择集合与顺序逐位相同。
    {
        for (int i = tid; i < E; i += nth) {
            const float pi = p[i];
            int c = 0;
            for (int j = 0; j < E; ++j) {
                const float pj = p[j];
                if (pj > pi || (pj == pi && j < i)) ++c;
            }
            srank[i] = c;
            // ⚠️ 非有限值**永不入选择集合**：NaN 参与比较时全假 ⇒ c=0、会被"选中"，
            // 于是 sel_idx[0] 被多个元素抢占、其余槽留 -1；随后 `p[sel_idx[r]]`
            // 读到 `p[-1]`（shared 窗口外的负偏移）⇒ **CUDA_ERROR_ILLEGAL_ADDRESS**
            // （2026-10-23 t2_route_topk 压力测试定位；生产 logits 只要出现一个 NaN/inf 就中）。
            // host 侧对非有限 logits 直接报错，这里保持"不选"的健壮语义即可。
            in_sel[i] = (c < top_k && isfinite(pi)) ? 1u : 0u;
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
        for (int r = 0; r < top_k; ++r) {
            if (sel_idx[r] >= 0) norm += p[sel_idx[r]];
        }
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
        // 补齐占位（同批量版）：非有限 logits ⇒ 选中数 < top_k，尾部留残渣会让下游误读。
        for (; w < top_k; ++w) {
            ids[w] = 0;
            wts[w] = 0.0f;
            slot_ids[w] = -1;
            host_slots[w] = -1;
            rows[w] = 0;
        }
        *miss_flag = nmiss;
    }
}

// ─────────── ㊿+ 2026-10-22 批量路由：grid.x = 行，每 block 复刻一行 host softmax/topk ───────────
//
// rows>1（prefill/投机验证）的 host 路径是「D2H 全部 logits → 每行 host softmax +
// 插入式 topk + 按专家升序拼对」。本核把每行数学**逐位复刻**（与上面单块版同一套：
// glibc-expf 复刻、升序串行 sum、IEEE 除法、总序 (p 降序, 下标升序) 的 top_k、
// 选择序 norm、w = p/norm），grid.x 每 block 管一行，输出按**专家升序**：
// ids[row*top_k + j] / wts[row*top_k + j]
// host 侧按行升序展开即得到与旧实现**完全相同**的 per_expert/pair 列表（逐位）。
//
// ── 2026-10-23 提速（clock64 段定位后重写「排名 + 输出」，数学一字未改） ──────────
// 定位（rows=6 / E=512 / top_k=10，旧核总 ≈114k SM tick ≈ 90 µs，microbench 实测 89.8 µs）：
// max 1.2k / exp 1.6k / sum 3.4k / div 0.5k / **rank 42k** / **out 65k** tick
// ⇒ 旧核 94% 的时间在「每线程串行 O(E²) 排名」与「tid==0 串行扫 512 项输出/建 sel_idx」上。
// 改后同口径 14.8k tick（microbench 13.7 µs，6.6×）：rank 7.3k / sum 2.8k / 其余 ~2.3k。
//
// 新的等价写法（只换算法，不换语义）：
// 1) 把「(p 降序, 下标升序) 全序」编码成 u64 键
// key_i = (sortable_u32(p_i) << 32) | (0xFFFFFFFF - i)
// p_i ≥ 0（expf 结果，且 x = logit-max ≤ 0 ⇒ 不可能 +inf）⇒ float 比较与 sortable
// 位序一致；索引取反 ⇒ 同值按下标升序。非有限 p（只会是 NaN）键置 0，低于任何真实
// 元素 ⇒ 既不中选也不参与计数，与旧版「NaN 严格比较恒假 + isfinite 防护」等价。
// 2) 名次：**逐轮取「严格小于上一轮键」的最大键**（键唯一 ⇒ 第 1..top_k 名的顺序唯一），
// 每轮一次 block 归约 —— O(top_k·E) 而不是 O(E²)。
// 3) 选中集：不再是「数出精确名次 c 再判 c < top_k」，而是逐轮取出的那 top_k 个本身
// （取不满 = 存在非有限值 ⇒ 只输出取到的这些，其余写 (0, 0.0) 占位，同旧版）。
// 4) norm 仍按**名次顺序串行**累加（与旧版 `sel_idx[r]` 的顺序逐位相同）。
// 5) 输出仍按**专家升序**：每个选中项自己数「有几个选中项下标更小」⇒ 命中槽位；
// 权重仍为 `p[v] / norm`（同一 IEEE 除法）。
// 6) `sum` 仍是**顺序**累加（bit 语义的一部分，禁止并行归约），只把相邻 4 个的 load
// 提前发出（加法次序不变，实测 3.4k → 2.8k tick）。
// 逐位对拍：`tests/t4_route_topk_bench.rs`（旧/新摘要一致）+ `tests/t2_route_topk.rs`
// （对 host 参考，4000 轮 × rows≤16，含并列 / -inf / NaN 分布）。旧实现保留为
// `lingxi_route_topk_rows_bf16_legacy`，`LINGXI_NO_ROUTE_TOPK_FAST=1` 可整核回退。
extern "C" __global__ void __launch_bounds__(256, 1) lingxi_route_topk_rows_bf16(
    const bf16* __restrict__ logits,      // [rows, E]
    int* __restrict__ ids,                // [rows, top_k] 专家升序
    float* __restrict__ wts,              // [rows, top_k] 与 ids 对齐
    int E, int top_k) {
    __shared__ float p[1024];
    __shared__ unsigned long long ukey[1024];   // 全序键（见上）
    __shared__ int sel_r[72];                   // 名次 → 专家下标（top_k ≤ 64）
    __shared__ unsigned long long red_k[32];
    __shared__ int red_i[32];
    __shared__ float red[32];
    __shared__ float s_m, s_sum, s_norm;
    __shared__ unsigned long long s_thr;        // 当前名次阈值（严格小于它才可能是下一名）
    __shared__ int s_n;                         // 已取出的名次数
    const int row = (int)blockIdx.x;
    const bf16* lr = logits + (size_t)row * (size_t)E;
    const int tid = (int)threadIdx.x;
    const int nth = (int)blockDim.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    float lv = -INFINITY;
    for (int i = tid; i < E; i += nth) lv = fmaxf(lv, __bfloat162float(lr[i]));
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
    for (int i = tid; i < E; i += nth) {
        p[i] = lingxi_glibc_expf(__bfloat162float(lr[i]) - m);
    }
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        // ⚠️ 顺序累加是 bit 语义的一部分（= host 的 `for i: s += p[i]`），不得并行归约；
        // 这里只是把「相邻 4 个的 load」提前发出来（加法的次序完全没变）。
        int i = 0;
        for (; i + 4 <= E; i += 4) {
            const float a0 = p[i], a1 = p[i + 1], a2 = p[i + 2], a3 = p[i + 3];
            s += a0; s += a1; s += a2; s += a3;
        }
        for (; i < E; ++i) s += p[i];
        s_sum = s;
    }
    __syncthreads();
    const float sum = s_sum;
    for (int i = tid; i < E; i += nth) {
        const float pi = p[i] / sum;
        p[i] = pi;
        ukey[i] = isfinite(pi)
                    ? ((((unsigned long long)__float_as_uint(pi)) << 32) |
                       (unsigned long long)(0xFFFFFFFFu - (unsigned int)i))
                    : 0ull;
    }
    if (tid == 0) { s_thr = ~0ull; s_n = 0; }
    __syncthreads();
    // 逐轮取名次：每轮取「键 < 上一轮阈值」的最大键（键唯一 ⇒ 名次顺序唯一）。
    // 键先落到寄存器（nth=256、E≤1024 ⇒ 每线程 ≤4 项），轮内不再读 shared。
    unsigned long long myk[4];
    int myi[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int i = tid + q * nth;
        myk[q] = (i < E) ? ukey[i] : 0ull;
        myi[q] = (i < E) ? i : -1;
    }
    const int nw = nth >> 5;
    for (int r = 0; r < top_k; ++r) {
        const unsigned long long thr = s_thr;
        unsigned long long bk = 0ull;
        int bi = -1;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned long long kk = myk[q];
            if (kk > bk && kk < thr) { bk = kk; bi = myi[q]; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned long long ok = __shfl_down_sync(0xffffffffu, bk, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ok > bk) { bk = ok; bi = oi; }
        }
        if (lane == 0) { red_k[wid] = bk; red_i[wid] = bi; }
        __syncthreads();
        if (wid == 0) {
            unsigned long long k2 = (lane < nw) ? red_k[lane] : 0ull;
            int i2 = (lane < nw) ? red_i[lane] : -1;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const unsigned long long ok = __shfl_down_sync(0xffffffffu, k2, off);
                const int oi = __shfl_down_sync(0xffffffffu, i2, off);
                if (ok > k2) { k2 = ok; i2 = oi; }
            }
            if (lane == 0 && i2 >= 0) { s_thr = k2; sel_r[s_n++] = i2; }
        }
        __syncthreads();
    }
    if (tid == 0) {
        float norm = 0.0f;
        for (int r = 0; r < s_n; ++r) norm += p[sel_r[r]];   // 名次序串行（= 旧版 sel_idx 序）
        s_norm = norm;
    }
    __syncthreads();
    // 输出：专家升序。s_n ≤ top_k ≤ 64，每个选中项自己数「有几个选中项下标更小」⇒ 直接得槽位。
    const float norm = s_norm;
    if (tid < s_n) {
        const int v = sel_r[tid];
        int w = 0;
        for (int r = 0; r < s_n; ++r) w += (sel_r[r] < v) ? 1 : 0;
        ids[(size_t)row * top_k + w] = v;
        wts[(size_t)row * top_k + w] = p[v] / norm;
    }
    // 补齐：非有限 logits 使选中数 < top_k（host 侧本会报错拒绝），写 (0, 0.0) 占位，
    // 保证下游每次 `ids` 查表都落在 [0,E) 内——否则 ids 尾部留池内残渣，
    // `moe_ops.cu` 的 pair 核按 `slot_map[ids[j]]` 查表就会越界（ILLEGAL_ADDRESS）。
    if (tid < top_k - s_n) {
        ids[(size_t)row * top_k + s_n + tid] = 0;
        wts[(size_t)row * top_k + s_n + tid] = 0.0f;
    }
}

// ── 旧实现（2026-10-22 版），保留作逐位基准 / 回退（`LINGXI_NO_ROUTE_TOPK_FAST=1`） ──
extern "C" __global__ void __launch_bounds__(256, 1) lingxi_route_topk_rows_bf16_legacy(
    const bf16* __restrict__ logits,      // [rows, E]
    int* __restrict__ ids,                // [rows, top_k] 专家升序
    float* __restrict__ wts,              // [rows, top_k] 与 ids 对齐
    int E, int top_k) {
    __shared__ float p[1024];
    __shared__ unsigned char in_sel[1024];
    __shared__ int srank[1024];
    __shared__ int sel_idx[72];
    __shared__ float red[32];
    __shared__ float s_m, s_sum;
    const int row = (int)blockIdx.x;
    const bf16* lr = logits + (size_t)row * (size_t)E;
    const int tid = (int)threadIdx.x;
    const int nth = (int)blockDim.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    float lv = -INFINITY;
    for (int i = tid; i < E; i += nth) lv = fmaxf(lv, __bfloat162float(lr[i]));
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
    for (int i = tid; i < E; i += nth) {
        p[i] = lingxi_glibc_expf(__bfloat162float(lr[i]) - m);
    }
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < E; ++i) s += p[i];
        s_sum = s;
    }
    __syncthreads();
    const float sum = s_sum;
    for (int i = tid; i < E; i += nth) {
        p[i] = p[i] / sum;
        in_sel[i] = 0u;
    }
    __syncthreads();
    for (int i = tid; i < E; i += nth) {
        const float pi = p[i];
        int c = 0;
        for (int j = 0; j < E; ++j) {
            const float pj = p[j];
            if (pj > pi || (pj == pi && j < i)) ++c;
        }
        srank[i] = c;
        // 同单块版的非有限值防护（否则 `p[sel_idx[r]]` 读 `p[-1]` ⇒ ILLEGAL_ADDRESS）
        in_sel[i] = (c < top_k && isfinite(pi)) ? 1u : 0u;
    }
    __syncthreads();
    if (tid == 0) {
        for (int r = 0; r < top_k; ++r) sel_idx[r] = -1;
        for (int i = 0; i < E; ++i) {
            if (in_sel[i]) sel_idx[srank[i]] = i;
        }
        float norm = 0.0f;
        for (int r = 0; r < top_k; ++r) {
            if (sel_idx[r] >= 0) norm += p[sel_idx[r]];
        }
        int w = 0;
        for (int i = 0; i < E && w < top_k; ++i) {
            if (!in_sel[i]) continue;
            ids[(size_t)row * top_k + w] = i;
            wts[(size_t)row * top_k + w] = p[i] / norm;
            ++w;
        }
        // 补齐：非有限 logits 使选中数 < top_k（host 侧本会报错拒绝），这里写**(0, 0.0) 占位**，
        // 保证下游每次 `ids` 查表都落在 [0,E) 内——否则 ids 尾部留池内残渣，
        // `moe_ops.cu` 的 pair 核按 `slot_map[ids[j]]` 查表就会越界（ILLEGAL_ADDRESS）。
        for (; w < top_k; ++w) {
            ids[(size_t)row * top_k + w] = 0;
            wts[(size_t)row * top_k + w] = 0.0f;
        }
    }
}
