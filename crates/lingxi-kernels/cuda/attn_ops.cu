// L1 QSA（稀疏全注意力）相关算子。

#include <cuda_bf16.h>
#include <math.h>

using bf16 = __nv_bfloat16;

// ── E4M3(FP8) 编解码（P15-㊶ KV 压缩 `LINGXI_KV_K8V4`，有损、env 门控）────────
// sm_70 没有 cvt 指令 ⇒ 手写 satfinite + RNE。码本与 PLE 表同一 E4M3（±448、
// denorm 步长 2^-9）。encode 以"值/scale"量化（scale = amax/448，按 (行, kv头) 存 f32）。
// 位运算版（免 exp2f SFU——发射率受限下 SFU 比 ALU 贵 4×，㊶ 实测教训）：
// normal(e≥1): (1+m/8)·2^(e-7) ⇒ 阶码域 e+120、尾数 m<<20（3 int op）；
// denorm(e=0): m·2^-9，(float)m 精确（I2F+mul）；符号用 -0.0 加法折叠。
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    const unsigned int s = (unsigned int)(b & 0x80u) << 24;
    const unsigned int e = ((unsigned int)b >> 3) & 15u;
    const unsigned int m = (unsigned int)b & 7u;
    const unsigned int nbits = (m << 20) | ((e + 120u) << 23);
    const float df = (float)m * 0.001953125f;
    const float val = (e == 0) ? df : __uint_as_float(nbits);
    return __uint_as_float(s) + val;
}

__device__ __forceinline__ unsigned char f32_to_e4m3(float x) {
    const bool neg = signbit(x) != 0;
    float a = fabsf(x);
    a = fminf(a, 448.0f);
    unsigned int r;
    if (a < 0.015625f) {                                  // < 2^-6：e4m3 denorm（步长 2^-9）
        const float u = a * 512.0f;                       // 以 2^-9 为单位
        const float fl = floorf(u);
        const float rc = u - fl;
        r = (unsigned int)fl + ((rc > 0.5f || (rc == 0.5f && ((unsigned int)fl & 1u))) ? 1u : 0u);
        if (r >= 8u) {                                    // RNE 越过 denorm 上限 ⇒ 最小 normal 2^-6
            return (unsigned char)((neg ? 0x80u : 0u) | 0x08u);
        }
    } else {
        unsigned int b = __float_as_uint(a);
        b += 0x0007FFFFu + ((b >> 20) & 1u);              // 丢掉 20 位尾数，RNE
        unsigned int e = (b >> 23);
        if (e > 135u) { e = 135u; b = (b & 0x807FFFFFu) | (135u << 23); }  // satfinite 护栏
        r = ((e - 120u) << 3) | ((b >> 20) & 7u);
    }
    return (unsigned char)(r | (neg ? 0x80u : 0u));
}

// bf16 [rows, heads, head_dim] → E4M3 字节（同平坦形状）+ 每 (row, head) 一个 f32 scale。
// 一个 block 处理一个 (row, head)；blockDim=256（head_dim=256 时一线程一元素）。
extern "C" __global__ void lingxi_quant_rows_e4m3_bf16(const bf16* __restrict__ x,
                                                       unsigned char* __restrict__ q,
                                                       float* __restrict__ sc,
                                                       int rows,
                                                       int heads,
                                                       int head_dim) {
    (void)rows;
    const int rh = blockIdx.x;
    const int lane = (int)threadIdx.x;
    const int nthr = (int)blockDim.x;
    const bf16* src = x + ((size_t)rh) * head_dim;
    unsigned char* dst = q + ((size_t)rh) * head_dim;
    float av = 0.0f;
    for (int d = lane; d < head_dim; d += nthr) {
        av = fmaxf(av, fabsf(__bfloat162float(src[d])));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        av = fmaxf(av, __shfl_down_sync(0xffffffffu, av, off));
    }
    __shared__ float wmax[32];
    const int warp = lane >> 5;
    const int wl = lane & 31;
    if (wl == 0) {
        wmax[warp] = av;
    }
    __syncthreads();
    if (lane < 32) {
        float m = (lane < (nthr >> 5)) ? wmax[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        }
        if (lane == 0) {
            wmax[0] = m;
        }
    }
    __syncthreads();
    const float amax = wmax[0];
    const float s = amax > 0.0f ? (amax / 448.0f) : 1.0f;   // 全零行 scale=1（避免除零）
    if (lane == 0) {
        sc[rh] = s;
    }
    const float inv = 1.0f / s;
    for (int d = lane; d < head_dim; d += nthr) {
        dst[d] = f32_to_e4m3(__bfloat162float(src[d]) * inv);
    }
}

// 部分 RoPE（NEOX 配对，就地）：x[rows, heads, head_dim]，只旋转前 rot_dim 维。
// cos/sin 是覆盖全序列的表（BF16）；第 row 行对应**绝对位置** row+pos_offset
// （整段前向 pos_offset=0；增量 decode 时 = 已缓存 token 数）。
// HF 的 cos/sin 是 [f0..f_{D/2-1}] 复制两份，与 rotate_half 配对等价于 (d, d + D/2)。
extern "C" __global__ void lingxi_rope_partial_inplace_bf16(bf16* __restrict__ x,
                                                            const bf16* __restrict__ cos,
                                                            const bf16* __restrict__ sin,
                                                            int rows,
                                                            int heads,
                                                            int head_dim,
                                                            int rot_dim,
                                                            int pos_offset) {
    const int half = rot_dim / 2;
    // 每个线程处理 (row, head, d) 的一个 d ∈ [0, half)
    const long long total = (long long)rows * heads * half;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int d = (int)(i % half);
    const int head = (int)((i / half) % heads);
    const int row = (int)(i / ((long long)half * heads));
    const int pos = row + pos_offset;

    bf16* base = x + ((size_t)row * heads + head) * head_dim;
    const float c = __bfloat162float(cos[(size_t)pos * rot_dim + d]);
    const float s = __bfloat162float(sin[(size_t)pos * rot_dim + d]);
    const float x0 = __bfloat162float(base[d]);
    const float x1 = __bfloat162float(base[d + half]);
    base[d] = __float2bfloat16(x0 * c - x1 * s);
    base[d + half] = __float2bfloat16(x1 * c + x0 * s);
}

// 同一 RoPE，但行位置按 `pos_base + row * pos_stride` 取（QSA 索引器池化块：
// 块起点位置 = b * compress，不是连续行）。数学与原核逐项相同。
extern "C" __global__ void lingxi_rope_partial_stride_bf16(bf16* __restrict__ x,
                                                           const bf16* __restrict__ cos,
                                                           const bf16* __restrict__ sin,
                                                           int rows,
                                                           int head_dim,
                                                           int rot_dim,
                                                           int pos_base,
                                                           int pos_stride) {
    const int half = rot_dim / 2;
    const long long total = (long long)rows * half;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int d = (int)(i % half);
    const int row = (int)(i / half);
    const int pos = pos_base + row * pos_stride;
    bf16* base = x + (size_t)row * head_dim;
    const float c = __bfloat162float(cos[(size_t)pos * rot_dim + d]);
    const float s = __bfloat162float(sin[(size_t)pos * rot_dim + d]);
    const float x0 = __bfloat162float(base[d]);
    const float x1 = __bfloat162float(base[d + half]);
    base[d] = __float2bfloat16(x0 * c - x1 * s);
    base[d + half] = __float2bfloat16(x1 * c + x0 * s);
}

// 带加性掩码的多头注意力（GQA，token-major 布局，batch=1）：
//   q[T, H, D], k/v[Tk, KVH, D], mask[1, T, Tk]（float 加性），scale
// 布局按 token-major 是为了直接消费投影结果，避免额外转置。
// ─────────── 稀疏注意力（重写版，2026-10-15）───────────
//
// ⚠️ **上一版是个 O(head_dim²·Tk) 的算法事故**：它的第三段（加权求和）写成
//   `for (d = tid; d < head_dim; d += blockDim) { for (j = 0; j < Tk; ++j) { s = Σ_dd q·k;  ← 重算点积
//      acc += exp(s)·v[j][d]; } }`
// 也就是**每个输出维度都把所有 key 的 QK 点积重算一遍**。head_dim=256 时，
// 每个 (head, token) 要做 `Tk·head_dim² = 192·65536 ≈ 12.6 M` 次 FMA，而正确量是 `Tk·head_dim = 49 K`
// ——**256 倍**。实测 `tk=192` 时单核 **1240 µs**（tk=4096 时 32.6 ms），
// 12 个 QSA 层 × 1.24 ms ≈ **15 ms/token，占整机 44%**，也正是"上下文一长就掉到 6 tok/s"的根因。
//
// 重写版：**两遍 + 分 tile**，每个 warp 负责一个 key（lane 沿 head_dim 分片 + warp 内归约），
// 点积只算两次（一遍求 max，一遍出 exp 并顺便做加权和），exp 值落在 shared。
// tile 循环保证任意 Tk 都能跑（不受 shared 容量限制）；`head_dim ≤ blockDim` 由调用方保证
// （不满足时回退旧核，见 `lingxi_attn_masked_bf16`）。
// P15-㊵：attn2 同样模板化，SPARSE=true 为 decode(t<4) 的跳块变体（attn2s）。
// 三态语义与 attn3_body 完全一致（见 state_j 注释）。
template <bool SPARSE>
__device__ __forceinline__ void attn2_body(const bf16* __restrict__ q,
                                           const bf16* __restrict__ k,
                                           const bf16* __restrict__ v,
                                           const float* __restrict__ mask,
                                           const unsigned char* __restrict__ keep,
                                           bf16* __restrict__ out,
                                           int heads,
                                           int kv_heads,
                                           int T,
                                           int Tk,
                                           int head_dim,
                                           float scale,
                                           int tile,
                                           int vis_base,
                                           int nblocks,
                                           int cshift,
                                           // ㊶ KV 量化（非空即启用；K/V 必须同时给）：
                                           const unsigned char* __restrict__ k_q8 = nullptr,
                                           const float* __restrict__ k_sc = nullptr,
                                           const unsigned char* __restrict__ v_q8 = nullptr,
                                           const float* __restrict__ v_sc = nullptr) {
    extern __shared__ float smem[];
    float* sp = smem;          // [tile]：本 tile 的 exp 值
    float* sred = smem + tile; // [32]：跨 warp 归约
    const int h = blockIdx.y;
    const int i = blockIdx.x;
    if (i >= T) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int kh = h / (heads / kv_heads);
    const bool q8 = (k_q8 != nullptr);
    const bf16* qv = q + ((size_t)i * heads + h) * head_dim;
    const float* mrow = mask + (size_t)i * Tk;

    // 跳块状态（仅 SPARSE；语义与 attn3_body 的 state_j 相同）
    auto state_j = [&](int pos) -> int {
        const bool before_vis = pos < vis_base + i + 1;
        const int tail = nblocks << cshift;
        const bool sel = pos >= tail
            || keep[(size_t)i * nblocks + (pos >> cshift)] != 0u;
        if (before_vis) return sel ? 0 : 1;
        return sel ? 1 : 2;
    };

    // ── 第 1 遍：全局 max（分 tile）──
    // 点积用 **128 位 load**：lane 一次吃 8 个 bf16（q 侧只在循环外读一次进寄存器）。
    // head_dim=256 = 32 lane × 8，正好一轮覆盖；更大时按 256 步进。
    float mx = -INFINITY;
    for (int j0 = 0; j0 < Tk; j0 += tile) {
        const int jn = (Tk - j0) < tile ? (Tk - j0) : tile;
        for (int j = warp; j < jn; j += nwarp) {
            const int st = SPARSE ? state_j(j0 + j) : 0;
            float s = 0.0f;
            if (st == 0) {
                if (q8) {
                    // ㊶ 量化 K：8 字节 load + 解码；scale 提到点积外（有损路径，无需保序）
                    const unsigned char* kqp = k_q8 + ((size_t)(j0 + j) * kv_heads + kh) * head_dim;
                    const float ksc = k_sc[(size_t)(j0 + j) * kv_heads + kh];
                    float sd = 0.0f;
                    for (int d0 = lane * 8; d0 < head_dim; d0 += 256) {
                        const int4 qv4 = *reinterpret_cast<const int4*>(qv + d0);
                        const uint2 kv2 = *reinterpret_cast<const uint2*>(kqp + d0);
                        const bf16* qp = reinterpret_cast<const bf16*>(&qv4);
                        const unsigned char* kp = reinterpret_cast<const unsigned char*>(&kv2);
#pragma unroll
                        for (int z = 0; z < 8; ++z) {
                            sd += __bfloat162float(qp[z]) * e4m3_to_f32(kp[z]);
                        }
                    }
                    s += sd * ksc;
                } else {
                const bf16* kv = k + ((size_t)(j0 + j) * kv_heads + kh) * head_dim;
                for (int d0 = lane * 8; d0 < head_dim; d0 += 256) {
                    const int4 qv4 = *reinterpret_cast<const int4*>(qv + d0);
                    const int4 kv4 = *reinterpret_cast<const int4*>(kv + d0);
                    const bf16* qp = reinterpret_cast<const bf16*>(&qv4);
                    const bf16* kp = reinterpret_cast<const bf16*>(&kv4);
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        s += __bfloat162float(qp[z]) * __bfloat162float(kp[z]);
                    }
                }
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_down_sync(0xffffffffu, s, off);
            }
            s = __shfl_sync(0xffffffffu, s, 0);
            if (lane == 0) {
                sp[j] = (st == 0)
                    ? s * scale + (SPARSE ? 0.0f : mrow[j0 + j])
                    : (st == 1 ? -3.402823466e+38F : -INFINITY);
            }
        }
        __syncthreads();
        float lm = -INFINITY;
        for (int j = tid; j < jn; j += blockDim.x) {
            lm = fmaxf(lm, sp[j]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            lm = fmaxf(lm, __shfl_down_sync(0xffffffffu, lm, off));
        }
        if (lane == 0) {
            sred[warp] = lm;
        }
        __syncthreads();
        if (tid < 32) {
            float w = (tid < nwarp) ? sred[tid] : -INFINITY;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                w = fmaxf(w, __shfl_down_sync(0xffffffffu, w, off));
            }
            if (tid == 0) {
                sred[0] = w;
            }
        }
        __syncthreads();
        mx = fmaxf(mx, sred[0]);
        __syncthreads();
    }

    // ── 第 2 遍：exp 落 shared，同时累积分母与加权和 ──
    float acc = 0.0f;   // 本线程负责的那个 d（head_dim ≤ blockDim 时恰好一个）
    float lsum = 0.0f;
    for (int j0 = 0; j0 < Tk; j0 += tile) {
        const int jn = (Tk - j0) < tile ? (Tk - j0) : tile;
        for (int j = warp; j < jn; j += nwarp) {
            const int st = SPARSE ? state_j(j0 + j) : 0;
            float s = 0.0f;
            if (st == 0) {
                if (q8) {
                    const unsigned char* kqp = k_q8 + ((size_t)(j0 + j) * kv_heads + kh) * head_dim;
                    const float ksc = k_sc[(size_t)(j0 + j) * kv_heads + kh];
                    float sd = 0.0f;
                    for (int d0 = lane * 8; d0 < head_dim; d0 += 256) {
                        const int4 qv4 = *reinterpret_cast<const int4*>(qv + d0);
                        const uint2 kv2 = *reinterpret_cast<const uint2*>(kqp + d0);
                        const bf16* qp = reinterpret_cast<const bf16*>(&qv4);
                        const unsigned char* kp = reinterpret_cast<const unsigned char*>(&kv2);
#pragma unroll
                        for (int z = 0; z < 8; ++z) {
                            sd += __bfloat162float(qp[z]) * e4m3_to_f32(kp[z]);
                        }
                    }
                    s += sd * ksc;
                } else {
                const bf16* kv = k + ((size_t)(j0 + j) * kv_heads + kh) * head_dim;
                for (int d0 = lane * 8; d0 < head_dim; d0 += 256) {
                    const int4 qv4 = *reinterpret_cast<const int4*>(qv + d0);
                    const int4 kv4 = *reinterpret_cast<const int4*>(kv + d0);
                    const bf16* qp = reinterpret_cast<const bf16*>(&qv4);
                    const bf16* kp = reinterpret_cast<const bf16*>(&kv4);
#pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        s += __bfloat162float(qp[z]) * __bfloat162float(kp[z]);
                    }
                }
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_down_sync(0xffffffffu, s, off);
            }
            s = __shfl_sync(0xffffffffu, s, 0);
            if (lane == 0) {
                sp[j] = (st == 0)
                    ? __expf(s * scale + (SPARSE ? 0.0f : mrow[j0 + j]) - mx)
                    : __expf(((st == 1) ? -3.402823466e+38F : -INFINITY) - mx);
            }
        }
        __syncthreads();
        float ls = 0.0f;
        for (int j = tid; j < jn; j += blockDim.x) {
            ls += sp[j];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ls += __shfl_down_sync(0xffffffffu, ls, off);
        }
        if (lane == 0) {
            sred[warp] = ls;
        }
        __syncthreads();
        if (tid < 32) {
            float w = (tid < nwarp) ? sred[tid] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                w += __shfl_down_sync(0xffffffffu, w, off);
            }
            if (tid == 0) {
                sred[0] = w;
            }
        }
        __syncthreads();
        lsum += sred[0];
        if (tid < head_dim) {
            const bf16* vbase = v + ((size_t)kh) * head_dim + tid;
            const unsigned char* vqp = v_q8 + ((size_t)kh) * head_dim + tid;
            float a = 0.0f;
            for (int j = 0; j < jn; ++j) {
                // V 读取跳过（bit-safe 论据同 attn3_body）
                if (!SPARSE || sp[j] != 0.0f) {
                    const float vv = q8
                        ? e4m3_to_f32(vqp[(size_t)(j0 + j) * kv_heads * head_dim]) *
                          v_sc[(size_t)(j0 + j) * kv_heads + kh]
                        : __bfloat162float(vbase[(size_t)(j0 + j) * kv_heads * head_dim]);
                    a += sp[j] * vv;
                }
            }
            acc += a;
        }
        __syncthreads();
    }
    if (tid < head_dim) {
        out[((size_t)i * heads + h) * head_dim + tid] = __float2bfloat16(acc / lsum);
    }
}

// 密集版入口（行为与历史 attn2 完全一致）
extern "C" __global__ void lingxi_attn2_masked_bf16(const bf16* __restrict__ q,
                                                    const bf16* __restrict__ k,
                                                    const bf16* __restrict__ v,
                                                    const float* __restrict__ mask,
                                                    bf16* __restrict__ out,
                                                    int heads,
                                                    int kv_heads,
                                                    int T,
                                                    int Tk,
                                                    int head_dim,
                                                    float scale,
                                                    int tile) {
    attn2_body<false>(q, k, v, mask, nullptr, out, heads, kv_heads, T, Tk, head_dim, scale, tile, 0, 0, 0);
}

// 跳块稀疏 decode 版入口（P15-㊵）：语义见 attn5 入口注释
extern "C" __global__ void lingxi_attn2s_sparse_bf16(const bf16* __restrict__ q,
                                                     const bf16* __restrict__ k,
                                                     const bf16* __restrict__ v,
                                                     const unsigned char* __restrict__ keep,
                                                     bf16* __restrict__ out,
                                                     int heads,
                                                     int kv_heads,
                                                     int T,
                                                     int Tk,
                                                     int head_dim,
                                                     float scale,
                                                     int tile,
                                                     int vis_base,
                                                     int nblocks,
                                                     int cshift) {
    attn2_body<true>(q, k, v, nullptr, keep, out, heads, kv_heads, T, Tk, head_dim, scale, tile,
                     vis_base, nblocks, cshift);
}

// ㊶ KV 量化（E4M3）入口：k_q8/k_sc/v_q8/v_sc 非空即启用；k/v 形参仅在 bf16 路径读。
extern "C" __global__ void lingxi_attn2_masked_kvq8_bf16(const bf16* __restrict__ q,
                                                         const bf16* __restrict__ k,
                                                         const bf16* __restrict__ v,
                                                         const float* __restrict__ mask,
                                                         bf16* __restrict__ out,
                                                         int heads,
                                                         int kv_heads,
                                                         int T,
                                                         int Tk,
                                                         int head_dim,
                                                         float scale,
                                                         int tile,
                                                         const unsigned char* __restrict__ k_q8,
                                                         const float* __restrict__ k_sc,
                                                         const unsigned char* __restrict__ v_q8,
                                                         const float* __restrict__ v_sc) {
    attn2_body<false>(q, k, v, mask, nullptr, out, heads, kv_heads, T, Tk, head_dim, scale,
                      tile, 0, 0, 0, k_q8, k_sc, v_q8, v_sc);
}

extern "C" __global__ void lingxi_attn2s_sparse_kvq8_bf16(const bf16* __restrict__ q,
                                                          const bf16* __restrict__ k,
                                                          const bf16* __restrict__ v,
                                                          const unsigned char* __restrict__ keep,
                                                          bf16* __restrict__ out,
                                                          int heads,
                                                          int kv_heads,
                                                          int T,
                                                          int Tk,
                                                          int head_dim,
                                                          float scale,
                                                          int tile,
                                                          int vis_base,
                                                          int nblocks,
                                                          int cshift,
                                                          const unsigned char* __restrict__ k_q8,
                                                          const float* __restrict__ k_sc,
                                                          const unsigned char* __restrict__ v_q8,
                                                          const float* __restrict__ v_sc) {
    attn2_body<true>(q, k, v, nullptr, keep, out, heads, kv_heads, T, Tk, head_dim, scale, tile,
                     vis_base, nblocks, cshift, k_q8, k_sc, v_q8, v_sc);
}

// ─────────── 行块平铺版（attn3，2026-10-17，P15-㊳）───────────
//
// 动机（实测，17K 上下文 prefill）：attn2 的 grid=(T, heads)，**每个 (行,头) 块独自
// 流式扫全量 K（两遍）+V**，per-chunk attention 流量 = O(chunk×T)。17K 语料实测
// prefill 边际成本 9.9→24.5 ms/行（8.5K→33K，~0.6ms/行/10K 上下文），外推 256K
// = 数小时——正是"256K 要 2 小时"的元凶。
//
// 改法：一个 block 负责 R 行 × 1 头，K/V 只流一遍喂 R 行 ⇒ 流量 ÷R。
// **bit-safe 论证**：每行内部的全部归约次序与 attn2 逐项相同——
//   ① 点积：lane 沿 d0=lane*8 的 8 元 FMA 链 + 同一 shuffle 树（R 行只是复用同一
//      次 kv load 各算各的，互不干扰）；
//   ② max：fmax 无舍入，同集合任意次序结果逐位相同；
//   ③ exp/lsum/acc：sp 值相同，求和循环（strided-by-blockDim / 顺序 j）原样保留。
// 只改"哪些行共享 K 读"，不改任何一行的数学。
//
// 约束：head_dim == 256 == blockDim（调用方保证，否则回退 attn2）；T 任意（尾部行守卫）。
#define LINGXI_ATTN3_R 4
#define LINGXI_ATTN3_TILE 512
#define LINGXI_ATTN3_KPF 4

// P15-㊵ 跳块稀疏变体（SPARSE=true）：按 keep[]+因果谓词**跳过未选中位置的点积**。
// bit-safe 论证：被跳位置在密集版里的贡献恰为 0——mask=f32::MIN(+s 被吸附) ⇒ exp≡0；
// lsum/acc 的加 0 精确不变（x+±0=x，唯一例外是 acc 恰停于 ∓0 的测零概率边角，判据回归覆盖）；
// lsum/acc 循环保持全扫 ⇒ 求和分区与次序与密集版逐项相同。参考实现（ninfer-dev
// qsa_attend_volta 的 kv_gather 方案）同思路，我们选"原地跳"省掉 gather 拷贝。
template <bool SPARSE>
__device__ __forceinline__ void attn3_body(const bf16* __restrict__ q,
                                           const bf16* __restrict__ k,
                                           const bf16* __restrict__ v,
                                           const float* __restrict__ mask,
                                           const unsigned char* __restrict__ keep,
                                           bf16* __restrict__ out,
                                           int heads,
                                           int kv_heads,
                                           int T,
                                           int Tk,
                                           int head_dim,
                                           float scale,
                                           int vis_base,
                                           int nblocks,
                                           int cshift,
                                           // ㊶ KV 量化（非空即启用；K/V 必须同时给）
                                           const unsigned char* __restrict__ k_q8 = nullptr,
                                           const float* __restrict__ k_sc = nullptr,
                                           const unsigned char* __restrict__ v_q8 = nullptr,
                                           const float* __restrict__ v_sc = nullptr) {
    constexpr int R = LINGXI_ATTN3_R;
    constexpr int TILE = LINGXI_ATTN3_TILE;
    constexpr int KPF = LINGXI_ATTN3_KPF;
    extern __shared__ float smem[];
    float* sp = smem;              // [R][TILE]：本 tile 每行的 exp 值
    float* sred = smem + R * TILE; // [32]：跨 warp 归约
    const int h = blockIdx.y;
    const int i0 = blockIdx.x * R; // 本块的首行
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const bool q8 = (k_q8 != nullptr);
    const int nwarp = (int)(blockDim.x >> 5);
    const int kh = h / (heads / kv_heads);
    // 本块实际行数（尾部块可能不足 R）
    int nr = T - i0; if (nr > R) nr = R;

    // 跳块状态（仅 SPARSE 使用）：对绝对位置 pos、行 i0+r 返回
    //   0 = 计算点积（pos < vis 且 [尾区 ∨ keep 选中]）
    //   1 = 跳过、mask 值 -FLT_MAX（dense 的 s*scale 被吸附后的精确等价：可见未选中 /
    //       超 vis 但选中 —— 两者的 dense mask 都是恰 -FLT_MAX）
    //   2 = 跳过、mask 值 -inf（超 vis 且未选中）
    // dense 版对跳过位置 sp1=mask 值（吸附）、sp2=__expf(mask-mx)——非退化 mx 有限 ⇒ 0，
    // 退化（整行无可见位，mx=-FLT_MAX）⇒ (1)=1 / (2)=0，本写法逐位复刻（含退化）。
    auto state_j = [&](int r, int pos) -> int {
        if (r >= nr) return 2;
        const bool before_vis = pos < vis_base + i0 + r + 1;
        const int tail = nblocks << cshift;
        const bool sel = pos >= tail
            || keep[(size_t)(i0 + r) * nblocks + (pos >> cshift)] != 0u;
        if (before_vis) return sel ? 0 : 1;
        return sel ? 1 : 2;
    };
    auto sel_j = [&](int r, int pos) -> bool { return state_j(r, pos) == 0; };

    // q 预载进寄存器：lane 持有各行 d0=lane*8 处的 int4（head_dim=256 一轮覆盖）。
    // 越界行的地址会被守卫（不读不写），但寄存器槽位照常存在以保循环结构恒定。
    int4 qreg[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        qreg[r] = make_int4(0, 0, 0, 0);
    }
    if (head_dim == 256) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < nr) {
                const bf16* qv = q + ((size_t)(i0 + r) * heads + h) * head_dim + lane * 8;
                qreg[r] = *reinterpret_cast<const int4*>(qv);
            }
        }
    }

    float mx[R], lsum[R], acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { mx[r] = -INFINITY; lsum[r] = 0.0f; acc[r] = 0.0f; }

    // ── 第 1 遍：全局 max（分 tile）── 结构与 attn2 相同，只是 K 一次 load 喂 R 行。
    // 内层每轮吃 KPF 个 key：先把 KPF 个 load 全部发出（延迟重叠），再逐 key 算点积。
    // 每 key 的 FMA 链与 shuffle 树与 attn2 逐项相同（bit-safe，只是交错发射）。
    for (int j0 = 0; j0 < Tk; j0 += TILE) {
        const int jn = (Tk - j0) < TILE ? (Tk - j0) : TILE;
        for (int jb = warp * KPF; jb < jn; jb += nwarp * KPF) {
            int4 kv4[KPF];
            uint2 kv2[KPF];   // ㊶ 量化 K 的 8 字节槽位（q8 时用）
            float ksc[KPF];   // ㊶ 每 (pos, kv头) 的 K scale
            int st[KPF][R];      // 三态：0 算 / 1 跳(-FLT_MAX) / 2 跳(-inf)
            bool any_on[KPF];
#pragma unroll
            for (int u = 0; u < KPF; ++u) {
                const int j = jb + u;
                any_on[u] = false;
                if (j < jn) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        st[u][r] = SPARSE ? state_j(r, j0 + j) : 0;
                        any_on[u] |= (st[u][r] == 0);
                    }
                    if (any_on[u]) {
                        if (q8) {
                            kv2[u] = *reinterpret_cast<const uint2*>(
                                k_q8 + ((size_t)(j0 + j) * kv_heads + kh) * head_dim + lane * 8);
                            ksc[u] = k_sc[(size_t)(j0 + j) * kv_heads + kh];
                        } else {
                            kv4[u] = *reinterpret_cast<const int4*>(
                                k + ((size_t)(j0 + j) * kv_heads + kh) * head_dim + lane * 8);
                        }
                    }
                }
            }
            float s[KPF][R];
#pragma unroll
            for (int u = 0; u < KPF; ++u) {
                const int j = jb + u;
                if (j < jn) {
                    const bf16* kp = reinterpret_cast<const bf16*>(&kv4[u]);
                    const unsigned char* kqp = reinterpret_cast<const unsigned char*>(&kv2[u]);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const bf16* qp = reinterpret_cast<const bf16*>(&qreg[r]);
                        s[u][r] = 0.0f;
                        if (st[u][r] == 0) {
#pragma unroll
                            for (int z = 0; z < 8; ++z) {
                                s[u][r] += q8
                                    ? __bfloat162float(qp[z]) * e4m3_to_f32(kqp[z])
                                    : __bfloat162float(qp[z]) * __bfloat162float(kp[z]);
                            }
                            if (q8) {
                                s[u][r] *= ksc[u];
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
                for (int u = 0; u < KPF; ++u) {
                    const int j = jb + u;
                    if (j < jn) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            s[u][r] += __shfl_down_sync(0xffffffffu, s[u][r], off);
                        }
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < KPF; ++u) {
                const int j = jb + u;
                if (j < jn) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        s[u][r] = __shfl_sync(0xffffffffu, s[u][r], 0);
                        // 跳过位写 dense 精确 mask 值（-FLT_MAX / -inf）：tile-max 循环全扫会读。
                        // 可见位 mask 值恰为 0（dense 同），SPARSE 下不读 mask（nullptr）
                        if (lane == 0) {
                            sp[r * TILE + j] = (st[u][r] == 0)
                                ? s[u][r] * scale + (SPARSE ? 0.0f : mask[(size_t)(i0 + r) * Tk + j0 + j])
                                : (st[u][r] == 1 ? -3.402823466e+38F : -INFINITY);
                        }
                    }
                }
            }
        }
        __syncthreads();
        // 每行一次 tile 内 max（fmax 无舍入 ⇒ 与 attn2 的分块方式结果逐位相同）
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float lm = -INFINITY;
            for (int j = tid; j < jn; j += blockDim.x) {
                lm = fmaxf(lm, sp[r * TILE + j]);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lm = fmaxf(lm, __shfl_down_sync(0xffffffffu, lm, off));
            }
            if (lane == 0) {
                sred[warp] = lm;
            }
            __syncthreads();
            if (tid < 32) {
                float w = (tid < nwarp) ? sred[tid] : -INFINITY;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    w = fmaxf(w, __shfl_down_sync(0xffffffffu, w, off));
                }
                if (tid == 0) {
                    sred[0] = w;
                }
            }
            __syncthreads();
            mx[r] = fmaxf(mx[r], sred[0]);
            __syncthreads();
        }
    }

    // ── 第 2 遍：exp 落 shared，同时累积分母与加权和 ──（同 KPF 批处理结构）
    for (int j0 = 0; j0 < Tk; j0 += TILE) {
        const int jn = (Tk - j0) < TILE ? (Tk - j0) : TILE;
        for (int jb = warp * KPF; jb < jn; jb += nwarp * KPF) {
            int4 kv4[KPF];
            uint2 kv2[KPF];   // ㊶ 量化 K 的 8 字节槽位（q8 时用）
            float ksc[KPF];   // ㊶ 每 (pos, kv头) 的 K scale
            int st[KPF][R];
            bool any_on[KPF];
#pragma unroll
            for (int u = 0; u < KPF; ++u) {
                const int j = jb + u;
                any_on[u] = false;
                if (j < jn) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        st[u][r] = SPARSE ? state_j(r, j0 + j) : 0;
                        any_on[u] |= (st[u][r] == 0);
                    }
                    if (any_on[u]) {
                        if (q8) {
                            kv2[u] = *reinterpret_cast<const uint2*>(
                                k_q8 + ((size_t)(j0 + j) * kv_heads + kh) * head_dim + lane * 8);
                            ksc[u] = k_sc[(size_t)(j0 + j) * kv_heads + kh];
                        } else {
                            kv4[u] = *reinterpret_cast<const int4*>(
                                k + ((size_t)(j0 + j) * kv_heads + kh) * head_dim + lane * 8);
                        }
                    }
                }
            }
            float s[KPF][R];
#pragma unroll
            for (int u = 0; u < KPF; ++u) {
                const int j = jb + u;
                if (j < jn) {
                    const bf16* kp = reinterpret_cast<const bf16*>(&kv4[u]);
                    const unsigned char* kqp = reinterpret_cast<const unsigned char*>(&kv2[u]);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const bf16* qp = reinterpret_cast<const bf16*>(&qreg[r]);
                        s[u][r] = 0.0f;
                        if (st[u][r] == 0) {
#pragma unroll
                            for (int z = 0; z < 8; ++z) {
                                s[u][r] += q8
                                    ? __bfloat162float(qp[z]) * e4m3_to_f32(kqp[z])
                                    : __bfloat162float(qp[z]) * __bfloat162float(kp[z]);
                            }
                            if (q8) {
                                s[u][r] *= ksc[u];
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
                for (int u = 0; u < KPF; ++u) {
                    const int j = jb + u;
                    if (j < jn) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            s[u][r] += __shfl_down_sync(0xffffffffu, s[u][r], off);
                        }
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < KPF; ++u) {
                const int j = jb + u;
                if (j < jn) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        s[u][r] = __shfl_sync(0xffffffffu, s[u][r], 0);
                        // 跳过位写 dense 精确值 __expf(maskval - mx)：非退化 ⇒ 0；
                        // 退化（mx=-FLT_MAX）⇒ (1)位=1 / (2)位=0，与密集版逐位一致
                        if (lane == 0) {
                            sp[r * TILE + j] = (st[u][r] == 0)
                                ? __expf(s[u][r] * scale
                                         + (SPARSE ? 0.0f : mask[(size_t)(i0 + r) * Tk + j0 + j]) - mx[r])
                                : __expf(((st[u][r] == 1) ? -3.402823466e+38F : -INFINITY) - mx[r]);
                        }
                    }
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float ls = 0.0f;
            for (int j = tid; j < jn; j += blockDim.x) {
                ls += sp[r * TILE + j];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ls += __shfl_down_sync(0xffffffffu, ls, off);
            }
            if (lane == 0) {
                sred[warp] = ls;
            }
            __syncthreads();
            if (tid < 32) {
                float w = (tid < nwarp) ? sred[tid] : 0.0f;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    w += __shfl_down_sync(0xffffffffu, w, off);
                }
                if (tid == 0) {
                    sred[0] = w;
                }
            }
            __syncthreads();
            lsum[r] += sred[0];
            __syncthreads();
        }
        // 加权 V 求和：V 每 j 只读一次，喂 R 行（acc 次序：每行仍是顺序 j 累加 ⇒ 与 attn2 相同）。
        // ★ P15-㊵续 V 读取跳过（bit-safe，同 K 点积论据）：sp==0 的位跳过 vv 读取与乘加——
        // 被跳项是 +0×v = ±0，加不进 acc（唯一例外 acc 恰为 ∓0 的测零概率边角）；
        // 退化情形（mx=-FLT_MAX）masked 位 sp∈{1,0}，非零位照常读 V ⇒ 逐位一致。
        if (tid < head_dim) {
            const bf16* vbase = v + ((size_t)kh) * head_dim + tid;
            const unsigned char* vqp = v_q8 + ((size_t)kh) * head_dim + tid;
            float a[R];
#pragma unroll
            for (int r = 0; r < R; ++r) { a[r] = 0.0f; }
            for (int j = 0; j < jn; ++j) {
                bool any = !SPARSE;
#pragma unroll
                for (int r = 0; r < R; ++r) { any |= (sp[r * TILE + j] != 0.0f); }
                if (any) {
                    const float vv = q8
                        ? e4m3_to_f32(vqp[(size_t)(j0 + j) * kv_heads * head_dim]) *
                          v_sc[(size_t)(j0 + j) * kv_heads + kh]
                        : __bfloat162float(vbase[(size_t)(j0 + j) * kv_heads * head_dim]);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const float s = sp[r * TILE + j];
                        if (!SPARSE || s != 0.0f) {
                            a[r] += s * vv;
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) { acc[r] += a[r]; }
        }
        __syncthreads();
    }
    if (tid < head_dim) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < nr) {
                out[((size_t)(i0 + r) * heads + h) * head_dim + tid] =
                    __float2bfloat16(acc[r] / lsum[r]);
            }
        }
    }
}

// 密集版入口（行为与历史 attn3 完全一致：SPARSE 分支被编译期消去）
extern "C" __global__ void __launch_bounds__(256, 2) lingxi_attn3_masked_bf16(const bf16* __restrict__ q,
                                                    const bf16* __restrict__ k,
                                                    const bf16* __restrict__ v,
                                                    const float* __restrict__ mask,
                                                    bf16* __restrict__ out,
                                                    int heads,
                                                    int kv_heads,
                                                    int T,
                                                    int Tk,
                                                    int head_dim,
                                                    float scale) {
    attn3_body<false>(q, k, v, mask, nullptr, out, heads, kv_heads, T, Tk, head_dim, scale, 0, 0, 0);
}

// 跳块稀疏版入口（P15-㊵）：keep=[行×块] 选中表，vis_base=已缓存 token 数，
// nblocks/cshift 描述池化几何（tail = nblocks<<cshift）。不读 mask。
extern "C" __global__ void __launch_bounds__(256, 2) lingxi_attn5_sparse_bf16(const bf16* __restrict__ q,
                                                    const bf16* __restrict__ k,
                                                    const bf16* __restrict__ v,
                                                    const unsigned char* __restrict__ keep,
                                                    bf16* __restrict__ out,
                                                    int heads,
                                                    int kv_heads,
                                                    int T,
                                                    int Tk,
                                                    int head_dim,
                                                    float scale,
                                                    int vis_base,
                                                    int nblocks,
                                                    int cshift) {
    attn3_body<true>(q, k, v, nullptr, keep, out, heads, kv_heads, T, Tk, head_dim, scale,
                     vis_base, nblocks, cshift);
}

// ㊶ KV 量化（E4M3）入口：k_q8/k_sc/v_q8/v_sc 非空即启用；k/v 形参仅在 bf16 路径读。
extern "C" __global__ void __launch_bounds__(256, 2) lingxi_attn3_masked_kvq8_bf16(const bf16* __restrict__ q,
                                                    const bf16* __restrict__ k,
                                                    const bf16* __restrict__ v,
                                                    const float* __restrict__ mask,
                                                    bf16* __restrict__ out,
                                                    int heads,
                                                    int kv_heads,
                                                    int T,
                                                    int Tk,
                                                    int head_dim,
                                                    float scale,
                                                    const unsigned char* __restrict__ k_q8,
                                                    const float* __restrict__ k_sc,
                                                    const unsigned char* __restrict__ v_q8,
                                                    const float* __restrict__ v_sc) {
    attn3_body<false>(q, k, v, mask, nullptr, out, heads, kv_heads, T, Tk, head_dim, scale,
                      0, 0, 0, k_q8, k_sc, v_q8, v_sc);
}

extern "C" __global__ void __launch_bounds__(256, 2) lingxi_attn5_sparse_kvq8_bf16(const bf16* __restrict__ q,
                                                    const bf16* __restrict__ k,
                                                    const bf16* __restrict__ v,
                                                    const unsigned char* __restrict__ keep,
                                                    bf16* __restrict__ out,
                                                    int heads,
                                                    int kv_heads,
                                                    int T,
                                                    int Tk,
                                                    int head_dim,
                                                    float scale,
                                                    int vis_base,
                                                    int nblocks,
                                                    int cshift,
                                                    const unsigned char* __restrict__ k_q8,
                                                    const float* __restrict__ k_sc,
                                                    const unsigned char* __restrict__ v_q8,
                                                    const float* __restrict__ v_sc) {
    attn3_body<true>(q, k, v, nullptr, keep, out, heads, kv_heads, T, Tk, head_dim, scale,
                     vis_base, nblocks, cshift, k_q8, k_sc, v_q8, v_sc);
}

// 旧实现（O(head_dim²·Tk)，仅作 `head_dim > blockDim` 的兜底；正常路径不要用）
extern "C" __global__ void lingxi_attn_masked_bf16(const bf16* __restrict__ q,
                                                   const bf16* __restrict__ k,
                                                   const bf16* __restrict__ v,
                                                   const float* __restrict__ mask,
                                                   bf16* __restrict__ out,
                                                   int heads,
                                                   int kv_heads,
                                                   int T,
                                                   int Tk,
                                                   int head_dim,
                                                   float scale) {
    extern __shared__ float sdata[];
    const int h = blockIdx.y;
    const int i = blockIdx.x;
    if (i >= T) {
        return;
    }
    const int kh = h / (heads / kv_heads);   // GQA：多个 q 头共享同一个 kv 头

    const bf16* qv = q + ((size_t)i * heads + h) * head_dim;
    const float* mrow = mask + (size_t)i * Tk;

    // 求最大值（数值稳定）
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < Tk; j += blockDim.x) {
        float s = 0.0f;
        const bf16* kv = k + ((size_t)j * kv_heads + kh) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            s += __bfloat162float(qv[d]) * __bfloat162float(kv[d]);
        }
        s = s * scale + mrow[j];
        local_max = fmaxf(local_max, s);
    }
    // 归约 max
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + st]);
        }
        __syncthreads();
    }
    const float mx = sdata[0];
    __syncthreads();

    // 求 exp 和
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < Tk; j += blockDim.x) {
        float s = 0.0f;
        const bf16* kv = k + ((size_t)j * kv_heads + kh) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            s += __bfloat162float(qv[d]) * __bfloat162float(kv[d]);
        }
        s = s * scale + mrow[j];
        local_sum += __expf(s - mx);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            sdata[threadIdx.x] += sdata[threadIdx.x + st];
        }
        __syncthreads();
    }
    const float inv = 1.0f / sdata[0];
    __syncthreads();

    // 加权求和
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < Tk; ++j) {
            float s = 0.0f;
            const bf16* kv = k + ((size_t)j * kv_heads + kh) * head_dim;
            for (int dd = 0; dd < head_dim; ++dd) {
                s += __bfloat162float(qv[dd]) * __bfloat162float(kv[dd]);
            }
            s = s * scale + mrow[j];
            const float p = __expf(s - mx) * inv;
            const bf16* vv = v + ((size_t)j * kv_heads + kh) * head_dim;
            acc += p * __bfloat162float(vv[d]);
        }
        out[((size_t)i * heads + h) * head_dim + d] = __float2bfloat16(acc);
    }
}

// ───────────────────── QSA indexer 选块（GPU 版，替换 host 路径） ─────────────────────
//
// 与 host 参考 `lingxi-model/src/qsa.rs::build_indexer_mask` 逐条对应：
//   池化(mean→bf16) → RMSNorm → RoPE(块起点位置) → 打分 Σ_h relu(q·p)/√hd → top-k 块 → 掩码。
// 四步各一个 kernel，全在同一 stream 上，无 D2H 同步。

// 1) 把一个压缩块的 raw key 求均值并 round 到 bf16：pooled[d] = bf16(Σ_j rk[row0+j][d] / compress)
extern "C" __global__ void lingxi_qsa_pool_block_bf16(const bf16* __restrict__ rk,
                                                      bf16* __restrict__ pooled,
                                                      int row0, int compress, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hd) {
        return;
    }
    float acc = 0.0f;
    for (int j = 0; j < compress; ++j) {
        acc += __bfloat162float(rk[(size_t)(row0 + j) * hd + d]);
    }
    pooled[d] = __float2bfloat16(acc / (float)compress);
}

// 1b) **批量池化**：一次 launch 处理 nb 个新块（原来每块一条链 4 个 launch ⇒ 128 块
//     = 512 次 launch/层/chunk，256K 下光 launch 就要 ~10 分钟）。数学与逐块版逐项相同。
extern "C" __global__ void lingxi_qsa_pool_blocks_bf16(const bf16* __restrict__ rk,
                                                       bf16* __restrict__ out,   // [nb, hd] 临时缓冲
                                                       int b0, int nb, int compress, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hd) {
        return;
    }
    const int lb = blockIdx.y;                 // 局部块号 0..nb-1
    const int b = b0 + lb;                     // 绝对块号（只用于读 rk）
    const int row0 = b * compress;
    float acc = 0.0f;
    for (int j = 0; j < compress; ++j) {
        acc += __bfloat162float(rk[(size_t)(row0 + j) * hd + d]);
    }
    // ⚠️ 写**局部下标**：调用方传的是 nb*hd 的临时缓冲，不是整张 pooled。
    out[(size_t)lb * hd + d] = __float2bfloat16(acc / (float)compress);
}

// 2) 每个 (query 行, 块) 一个线程：scores[row][b] = (Σ_h max(Σ_d q[row,h,d]·p[b,d], 0)) / √hd
//    grid = (nblocks 方向, t_q)；`q` 是 [t_q, heads, hd]
extern "C" __global__ void lingxi_qsa_indexer_scores_bf16(const bf16* __restrict__ q,
                                                          const bf16* __restrict__ pooled,
                                                          float* __restrict__ scores,
                                                          int t_q, int nblocks, int heads, int hd,
                                                          float sqrt_hd) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (b >= nblocks || row >= t_q) {
        return;
    }
    const bf16* qr = q + (size_t)row * heads * hd;
    const bf16* p = pooled + (size_t)b * hd;
    float s = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const bf16* qh = qr + (size_t)h * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) {
            dot += __bfloat162float(qh[d]) * __bfloat162float(p[d]);
        }
        s += fmaxf(dot, 0.0f);
    }
    scores[(size_t)row * nblocks + b] = s / sqrt_hd;
}

// 3) 名次：rank(row,b) = #{b' : score[row][b'] > score[row][b] 或 (相等且 b' < b)}；rank < topk 则保留
//    grid = (nblocks 方向, t_q)
extern "C" __global__ void lingxi_qsa_indexer_rank_f32(const float* __restrict__ scores,
                                                       unsigned char* __restrict__ keep,
                                                       int t_q, int nblocks, int topk) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (b >= nblocks || row >= t_q) {
        return;
    }
    const float* sc = scores + (size_t)row * nblocks;
    const float sb = sc[b];
    int rank = 0;
    for (int b2 = 0; b2 < nblocks; ++b2) {
        const float s2 = sc[b2];
        if (s2 > sb || (s2 == sb && b2 < b)) {
            ++rank;
        }
    }
    keep[(size_t)row * nblocks + b] = (rank < topk) ? 1u : 0u;
}

// ───────────────── 3') 精确 top-k 选块（O(nblocks)，替代 rank 的 O(nblocks²)）─────────────────
//
// 动机（2026-10-18，256K 可用性）：`lingxi_qsa_indexer_rank_f32` 每个 (row, blk) 线程都要
// 扫完 nblocks 条分数 ⇒ 单 chunk O(t·nblocks²)，跨 chunk 累加 ≈ O(S³)。256K 下 nblocks=65536，
// 这一项单独就要几十分钟；同时 `scores` 要 t×nblocks×4B（2048×65536=537MB/层×12 层 ≈ 6.4GB）
// 把显存顶爆。本实现把"选块"做成**每行一个 block 的流式基数选择**：
//   - 分数只算不存（5 遍：3 遍直方图定位阈值 + 2 遍填 keep），零 scores 缓冲；
//   - 分数恒 ≥ 0 ⇒ IEEE-754 正浮点位型与数值同序，可直接对 u32 位型做 11+11+10 位基数选择；
//   - 并列语义与旧 rank 完全一致：同分按块下标升序取前 (topk - cnt_gt) 个。
// 开关 `LINGXI_QSA_KEEP_RADIX`（默认开；=0 回旧 scores+rank 路径做 A/B）。

__device__ __forceinline__ float qsa_score1(const bf16* __restrict__ qr,
                                            const bf16* __restrict__ p,
                                            int heads, int hd, float sqrt_hd) {
    float s = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const bf16* qh = qr + (size_t)h * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) {
            dot += __bfloat162float(qh[d]) * __bfloat162float(p[d]);
        }
        s += fmaxf(dot, 0.0f);
    }
    return s / sqrt_hd;
}

// **4 个块并行打分**（ILP）：单线程一条 512 MAC 的依赖链是选块打分的瓶颈
// （每元素 2 次 bf16→f32 转换 + FMA，延迟 4~6 周期）。四个独立累加链交错发射可把
// 延迟藏起来；**每个点积内部的加序与 `qsa_score1` 逐项相同**（bit-exact）。
__device__ __forceinline__ void qsa_score4(const bf16* __restrict__ qr,
                                           const bf16* __restrict__ p0,
                                           const bf16* __restrict__ p1,
                                           const bf16* __restrict__ p2,
                                           const bf16* __restrict__ p3,
                                           int heads, int hd, float sqrt_hd, float* out) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const bf16* qh = qr + (size_t)h * hd;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        for (int d = 0; d < hd; ++d) {
            const float q = __bfloat162float(qh[d]);
            d0 += q * __bfloat162float(p0[d]);
            d1 += q * __bfloat162float(p1[d]);
            d2 += q * __bfloat162float(p2[d]);
            d3 += q * __bfloat162float(p3[d]);
        }
        s0 += fmaxf(d0, 0.0f);
        s1 += fmaxf(d1, 0.0f);
        s2 += fmaxf(d2, 0.0f);
        s3 += fmaxf(d3, 0.0f);
    }
    out[0] = s0 / sqrt_hd;
    out[1] = s1 / sqrt_hd;
    out[2] = s2 / sqrt_hd;
    out[3] = s3 / sqrt_hd;
}

// 阈值选择：每行一个 block（grid = t_q），3 遍打分 + 直方图。输出 thr_key[row]/cnt_gt[row]。
extern "C" __global__ void lingxi_qsa_keep_select_bf16(const bf16* __restrict__ q,
                                                       const bf16* __restrict__ pooled,
                                                       unsigned int* __restrict__ thr_key,
                                                       int* __restrict__ cnt_gt,
                                                       int nblocks, int heads, int hd,
                                                       float sqrt_hd, int topk) {
    __shared__ unsigned int hist[2048];
    __shared__ int s_bin[4];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const bf16* qr = q + (size_t)row * heads * hd;

    if (nblocks <= topk) {  // 全保留：fill 核特判，这里只写占位
        if (tid == 0) { thr_key[row] = 0u; cnt_gt[row] = 0; }
        return;
    }

    // ── 第 1 遍：高 11 位（每线程 4 块并行，打 ILP）──
    for (int i = tid; i < 2048; i += nthr) hist[i] = 0u;
    __syncthreads();
    for (int b = tid * 4; b < nblocks; b += nthr * 4) {
        if (b + 3 < nblocks) {
            float sc[4];
            qsa_score4(qr, pooled + (size_t)b * hd, pooled + (size_t)(b + 1) * hd,
                       pooled + (size_t)(b + 2) * hd, pooled + (size_t)(b + 3) * hd,
                       heads, hd, sqrt_hd, sc);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                atomicAdd(&hist[(__float_as_uint(sc[u]) >> 21) & 2047u], 1u);
            }
        } else {
            for (int bb = b; bb < nblocks; ++bb) {
                const unsigned int kk = __float_as_uint(qsa_score1(qr, pooled + (size_t)bb * hd, heads, hd, sqrt_hd));
                atomicAdd(&hist[(kk >> 21) & 2047u], 1u);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        int cum = 0, bin = 0;
        for (int i = 2047; i >= 0; --i) {
            const int h = (int)hist[i];
            if (cum + h >= topk) { bin = i; break; }
            cum += h;
        }
        s_bin[0] = bin; s_bin[1] = cum;
    }
    __syncthreads();
    const int bin1 = s_bin[0], cum1 = s_bin[1];

    // ── 第 2 遍：中 11 位（仅高 11 位命中 bin1 的元素）──
    for (int i = tid; i < 2048; i += nthr) hist[i] = 0u;
    __syncthreads();
    for (int b = tid * 4; b < nblocks; b += nthr * 4) {
        if (b + 3 < nblocks) {
            float sc[4];
            qsa_score4(qr, pooled + (size_t)b * hd, pooled + (size_t)(b + 1) * hd,
                       pooled + (size_t)(b + 2) * hd, pooled + (size_t)(b + 3) * hd,
                       heads, hd, sqrt_hd, sc);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned int kk = __float_as_uint(sc[u]);
                if ((int)((kk >> 21) & 2047u) == bin1) atomicAdd(&hist[(kk >> 10) & 2047u], 1u);
            }
        } else {
            for (int bb = b; bb < nblocks; ++bb) {
                const unsigned int kk = __float_as_uint(qsa_score1(qr, pooled + (size_t)bb * hd, heads, hd, sqrt_hd));
                if ((int)((kk >> 21) & 2047u) == bin1) atomicAdd(&hist[(kk >> 10) & 2047u], 1u);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        int cum = 0, bin = 0;
        const int need = topk - cum1;
        for (int i = 2047; i >= 0; --i) {
            const int h = (int)hist[i];
            if (cum + h >= need) { bin = i; break; }
            cum += h;
        }
        s_bin[2] = bin; s_bin[3] = cum;
    }
    __syncthreads();
    const int bin2 = s_bin[2], cum2 = s_bin[3];

    // ── 第 3 遍：低 10 位（仅前 22 位全命中）──
    for (int i = tid; i < 1024; i += nthr) hist[i] = 0u;
    __syncthreads();
    const unsigned int hi22 = ((unsigned int)bin1 << 11) | (unsigned int)bin2;
    for (int b = tid * 4; b < nblocks; b += nthr * 4) {
        if (b + 3 < nblocks) {
            float sc[4];
            qsa_score4(qr, pooled + (size_t)b * hd, pooled + (size_t)(b + 1) * hd,
                       pooled + (size_t)(b + 2) * hd, pooled + (size_t)(b + 3) * hd,
                       heads, hd, sqrt_hd, sc);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned int kk = __float_as_uint(sc[u]);
                if ((kk >> 10) == hi22) atomicAdd(&hist[kk & 1023u], 1u);
            }
        } else {
            for (int bb = b; bb < nblocks; ++bb) {
                const unsigned int kk = __float_as_uint(qsa_score1(qr, pooled + (size_t)bb * hd, heads, hd, sqrt_hd));
                if ((kk >> 10) == hi22) atomicAdd(&hist[kk & 1023u], 1u);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        int cum = 0, bin = 0;
        const int need = topk - cum1 - cum2;
        for (int i = 1023; i >= 0; --i) {
            const int h = (int)hist[i];
            if (cum + h >= need) { bin = i; break; }
            cum += h;
        }
        thr_key[row] = (hi22 << 10) | (unsigned int)bin;
        cnt_gt[row] = cum1 + cum2 + cum;
    }
}

// ───────────────── 3'') 融合选块（2026-10-18）─────────────────
//
// 问题：radix 版为了省 `t×nblocks×4B` 的 scores 缓冲，把**同一份打分重算了 4 遍**
// （3 遍直方图定位阈值 + 1 遍填 keep）。32K 下每层 4 × 34 GMAC，单层 ~300ms，
// 占 prefill 总时间的 40%+（`LINGXI_QSA_PROF` 实测：select+fill 293~335ms，
// 而 gather 注意力只有 58~62ms）。
//
// 修法：nblocks 不太大时，把分数**落 shared**（nblocks×4B ≤ 36KB），之后三遍直方图和
// 填 keep 都读 shared，总打分次数 4 → 1。分数、阈值、并列规则与 radix 版**逐位一致**
// （同一 `qsa_score4`、同一 11+11+10 位、同一"同分按块下标升序"语义）。
// nblocks 超过 `LINGXI_QSA_FUSED_MAX`（默认 9216）时调用方回退旧路径。
extern "C" __global__ void lingxi_qsa_keep_fused_bf16(const bf16* __restrict__ q,
                                                      const bf16* __restrict__ pooled,
                                                      unsigned char* __restrict__ keep,
                                                      int nblocks, int heads, int hd,
                                                      float sqrt_hd, int topk) {
    extern __shared__ float sc[];          // nblocks 条分数
    __shared__ unsigned int hist[2048];
    __shared__ int s_bin[4];
    __shared__ int cnt[256];
    __shared__ int off[256];
    __shared__ unsigned int s_thr;
    __shared__ int s_need_eq;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const bf16* qr = q + (size_t)row * heads * hd;
    unsigned char* kp = keep + (size_t)row * nblocks;

    if (nblocks <= topk) {                 // 全保留
        for (int b = tid; b < nblocks; b += nthr) kp[b] = 1u;
        return;
    }

    // ── 1) 打分一次，落 shared（与 qsa_score4/qsa_score1 逐项同序）──
    for (int b = tid * 4; b < nblocks; b += nthr * 4) {
        if (b + 3 < nblocks) {
            float s4[4];
            qsa_score4(qr, pooled + (size_t)b * hd, pooled + (size_t)(b + 1) * hd,
                       pooled + (size_t)(b + 2) * hd, pooled + (size_t)(b + 3) * hd,
                       heads, hd, sqrt_hd, s4);
            sc[b] = s4[0]; sc[b + 1] = s4[1]; sc[b + 2] = s4[2]; sc[b + 3] = s4[3];
        } else {
            for (int bb = b; bb < nblocks; ++bb)
                sc[bb] = qsa_score1(qr, pooled + (size_t)bb * hd, heads, hd, sqrt_hd);
        }
    }
    __syncthreads();

    // ── 2) 第 1 遍直方图：高 11 位 ──
    for (int i = tid; i < 2048; i += nthr) hist[i] = 0u;
    __syncthreads();
    for (int b = tid; b < nblocks; b += nthr)
        atomicAdd(&hist[(__float_as_uint(sc[b]) >> 21) & 2047u], 1u);
    __syncthreads();
    if (tid == 0) {
        int cum = 0, bin = 0;
        for (int i = 2047; i >= 0; --i) {
            const int h = (int)hist[i];
            if (cum + h >= topk) { bin = i; break; }
            cum += h;
        }
        s_bin[0] = bin; s_bin[1] = cum;
    }
    __syncthreads();
    const int bin1 = s_bin[0], cum1 = s_bin[1];

    // ── 3) 第 2 遍：中 11 位 ──
    for (int i = tid; i < 2048; i += nthr) hist[i] = 0u;
    __syncthreads();
    for (int b = tid; b < nblocks; b += nthr) {
        const unsigned int kk = __float_as_uint(sc[b]);
        if ((int)((kk >> 21) & 2047u) == bin1) atomicAdd(&hist[(kk >> 10) & 2047u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        int cum = 0, bin = 0;
        const int need = topk - cum1;
        for (int i = 2047; i >= 0; --i) {
            const int h = (int)hist[i];
            if (cum + h >= need) { bin = i; break; }
            cum += h;
        }
        s_bin[2] = bin; s_bin[3] = cum;
    }
    __syncthreads();
    const int bin2 = s_bin[2], cum2 = s_bin[3];

    // ── 4) 第 3 遍：低 10 位 ──
    for (int i = tid; i < 1024; i += nthr) hist[i] = 0u;
    __syncthreads();
    const unsigned int hi22 = ((unsigned int)bin1 << 11) | (unsigned int)bin2;
    for (int b = tid; b < nblocks; b += nthr) {
        const unsigned int kk = __float_as_uint(sc[b]);
        if ((kk >> 10) == hi22) atomicAdd(&hist[kk & 1023u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        int cum = 0, bin = 0;
        const int need = topk - cum1 - cum2;
        for (int i = 1023; i >= 0; --i) {
            const int h = (int)hist[i];
            if (cum + h >= need) { bin = i; break; }
            cum += h;
        }
        s_thr = (hi22 << 10) | (unsigned int)bin;
        s_need_eq = topk - (cum1 + cum2 + cum);
    }
    __syncthreads();
    const unsigned int thr = s_thr;
    const int need_eq = s_need_eq;

    // ── 5) 填 keep：并列按块下标升序（与旧 fill 核同一规则）──
    const int per = (nblocks + nthr - 1) / nthr;
    const int lo = tid * per;
    const int hi = min(nblocks, lo + per);
    int c = 0;
    for (int b = lo; b < hi; ++b) c += (__float_as_uint(sc[b]) == thr);
    cnt[tid] = c;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nthr; ++i) { off[i] = acc; acc += cnt[i]; }
    }
    __syncthreads();
    int r = off[tid];
    for (int b = lo; b < hi; ++b) {
        const unsigned int kk = __float_as_uint(sc[b]);
        unsigned char v;
        if (kk > thr) {
            v = 1u;
        } else if (kk == thr) {
            v = (r < need_eq) ? 1u : 0u;
            ++r;
        } else {
            v = 0u;
        }
        kp[b] = v;
    }
}

// ───────────────── 3''') 大 nblocks 选块：分数已在全局，只做直方图 + 填 keep ─────────────────
//
// 为什么需要（256K）：融合核把分数放 shared，只能覆盖 nblocks ≤ 9216（≈49K 上下文）。
// 更大的上下文改走"先并行打分落全局（1 遍）+ 本核只做 3 遍直方图 + 1 遍填 keep"。
// 与 radix 版**逐位一致**：分数来自同一个 `qsa_score1` 公式（`qsa_indexer_scores_bf16`），
// 阈值/并列规则与融合核逐行相同。每行一个 block，分数 4B/块读 4 遍，代价远小于重算。
extern "C" __global__ void lingxi_qsa_keep_from_scores_bf16(const float* __restrict__ scores,
                                                           unsigned char* __restrict__ keep,
                                                           int nblocks, int topk) {
    __shared__ unsigned int hist[2048];
    __shared__ int s_bin[4];
    __shared__ int cnt[256];
    __shared__ int off[256];
    __shared__ unsigned int s_thr;
    __shared__ int s_need_eq;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const float* sc = scores + (size_t)row * nblocks;
    unsigned char* kp = keep + (size_t)row * nblocks;

    if (nblocks <= topk) {
        for (int b = tid; b < nblocks; b += nthr) kp[b] = 1u;
        return;
    }

    // 第 1 遍：高 11 位
    for (int i = tid; i < 2048; i += nthr) hist[i] = 0u;
    __syncthreads();
    for (int b = tid; b < nblocks; b += nthr)
        atomicAdd(&hist[(__float_as_uint(sc[b]) >> 21) & 2047u], 1u);
    __syncthreads();
    if (tid == 0) {
        int cum = 0, bin = 0;
        for (int i = 2047; i >= 0; --i) {
            const int h = (int)hist[i];
            if (cum + h >= topk) { bin = i; break; }
            cum += h;
        }
        s_bin[0] = bin; s_bin[1] = cum;
    }
    __syncthreads();
    const int bin1 = s_bin[0], cum1 = s_bin[1];

    // 第 2 遍：中 11 位
    for (int i = tid; i < 2048; i += nthr) hist[i] = 0u;
    __syncthreads();
    for (int b = tid; b < nblocks; b += nthr) {
        const unsigned int kk = __float_as_uint(sc[b]);
        if ((int)((kk >> 21) & 2047u) == bin1) atomicAdd(&hist[(kk >> 10) & 2047u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        int cum = 0, bin = 0;
        const int need = topk - cum1;
        for (int i = 2047; i >= 0; --i) {
            const int h = (int)hist[i];
            if (cum + h >= need) { bin = i; break; }
            cum += h;
        }
        s_bin[2] = bin; s_bin[3] = cum;
    }
    __syncthreads();
    const int bin2 = s_bin[2], cum2 = s_bin[3];

    // 第 3 遍：低 10 位
    for (int i = tid; i < 1024; i += nthr) hist[i] = 0u;
    __syncthreads();
    const unsigned int hi22 = ((unsigned int)bin1 << 11) | (unsigned int)bin2;
    for (int b = tid; b < nblocks; b += nthr) {
        const unsigned int kk = __float_as_uint(sc[b]);
        if ((kk >> 10) == hi22) atomicAdd(&hist[kk & 1023u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        int cum = 0, bin = 0;
        const int need = topk - cum1 - cum2;
        for (int i = 1023; i >= 0; --i) {
            const int h = (int)hist[i];
            if (cum + h >= need) { bin = i; break; }
            cum += h;
        }
        s_thr = (hi22 << 10) | (unsigned int)bin;
        s_need_eq = topk - (cum1 + cum2 + cum);
    }
    __syncthreads();
    const unsigned int thr = s_thr;
    const int need_eq = s_need_eq;

    // 填 keep：并列按块下标升序
    const int per = (nblocks + nthr - 1) / nthr;
    const int lo = tid * per;
    const int hi = min(nblocks, lo + per);
    int c = 0;
    for (int b = lo; b < hi; ++b) c += (__float_as_uint(sc[b]) == thr);
    cnt[tid] = c;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nthr; ++i) { off[i] = acc; acc += cnt[i]; }
    }
    __syncthreads();
    int r = off[tid];
    for (int b = lo; b < hi; ++b) {
        const unsigned int kk = __float_as_uint(sc[b]);
        unsigned char v;
        if (kk > thr) {
            v = 1u;
        } else if (kk == thr) {
            v = (r < need_eq) ? 1u : 0u;
            ++r;
        } else {
            v = 0u;
        }
        kp[b] = v;
    }
}

// 填 keep：先数"等于阈值"的个数并做块内前缀和（并列按块下标升序），再走一遍写 0/1。
extern "C" __global__ void lingxi_qsa_keep_fill_bf16(const bf16* __restrict__ q,
                                                     const bf16* __restrict__ pooled,
                                                     const unsigned int* __restrict__ thr_key,
                                                     const int* __restrict__ cnt_gt,
                                                     unsigned char* __restrict__ keep,
                                                     int nblocks, int heads, int hd,
                                                     float sqrt_hd, int topk) {
    __shared__ int cnt[1024];
    __shared__ int off[1024];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const bf16* qr = q + (size_t)row * heads * hd;
    unsigned char* kp = keep + (size_t)row * nblocks;

    if (nblocks <= topk) {
        for (int b = tid; b < nblocks; b += nthr) kp[b] = 1u;
        return;
    }
    const unsigned int thr = thr_key[row];
    const int need_eq = topk - cnt_gt[row];
    const int per = (nblocks + nthr - 1) / nthr;
    const int lo = tid * per;
    const int hi = min(nblocks, lo + per);
    int c = 0;
    for (int b = lo; b < hi; ) {
        if (b + 3 < hi) {
            float sc[4];
            qsa_score4(qr, pooled + (size_t)b * hd, pooled + (size_t)(b + 1) * hd,
                       pooled + (size_t)(b + 2) * hd, pooled + (size_t)(b + 3) * hd,
                       heads, hd, sqrt_hd, sc);
#pragma unroll
            for (int u = 0; u < 4; ++u) c += (__float_as_uint(sc[u]) == thr);
            b += 4;
        } else {
            const unsigned int kk = __float_as_uint(qsa_score1(qr, pooled + (size_t)b * hd, heads, hd, sqrt_hd));
            c += (kk == thr);
            ++b;
        }
    }
    cnt[tid] = c;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nthr; ++i) { off[i] = acc; acc += cnt[i]; }
    }
    __syncthreads();
    int r = off[tid];
    for (int b = lo; b < hi; ) {
        if (b + 3 < hi) {
            float sc[4];
            qsa_score4(qr, pooled + (size_t)b * hd, pooled + (size_t)(b + 1) * hd,
                       pooled + (size_t)(b + 2) * hd, pooled + (size_t)(b + 3) * hd,
                       heads, hd, sqrt_hd, sc);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned int kk = __float_as_uint(sc[u]);
                unsigned char v;
                if (kk > thr) {
                    v = 1u;
                } else if (kk == thr) {
                    v = (r < need_eq) ? 1u : 0u;
                    ++r;
                } else {
                    v = 0u;
                }
                kp[b + u] = v;
            }
            b += 4;
        } else {
            const unsigned int kk = __float_as_uint(qsa_score1(qr, pooled + (size_t)b * hd, heads, hd, sqrt_hd));
            unsigned char v;
            if (kk > thr) {
                v = 1u;
            } else if (kk == thr) {
                v = (r < need_eq) ? 1u : 0u;
                ++r;
            } else {
                v = 0u;
            }
            kp[b] = v;
            ++b;
        }
    }
}

// ───────────────── decode 选块：并行直方图版（t==1）─────────────────
//
// 为什么需要：radix 版是"每行一个 block"，t=1 时整个 GPU 只有一个 block 在跑
// （实测 8K 上下文 decode 97ms/token，30K 时 254ms/token，随 nblocks 线性涨）。
// decode 的正确形态是**先并行打分落盘（1 遍），再用网格并行的直方图**做同样的
// 11+11+10 位基数选择 —— 3 遍直方图 + 1 遍填 keep 都只读 4B/块，且全 GPU 并行。
// 语义与 radix 版逐位一致（同一阈值、同一并列规则）。

// 直方图：`(key >> shift) & bmask`，只统计 `(key & fmask) == fval` 的元素。
extern "C" __global__ void lingxi_qsa_hist_f32(const float* __restrict__ scores,
                                               unsigned int* __restrict__ hist,
                                               int nblocks, int shift, unsigned int bmask,
                                               unsigned int fmask, unsigned int fval) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned int kk = __float_as_uint(scores[b]);
    if ((kk & fmask) != fval) return;
    atomicAdd(&hist[(kk >> shift) & bmask], 1u);
}

// 后缀计数 → 阈值所在的 bin 与"严格高于它的个数"。单线程扫 nbins。
extern "C" __global__ void lingxi_qsa_bin_finalize(const unsigned int* __restrict__ hist,
                                                   int nbins, int need,
                                                   int* __restrict__ out /* [bin, cum] */) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int cum = 0, bin = 0;
    for (int i = nbins - 1; i >= 0; --i) {
        const int h = (int)hist[i];
        if (cum + h >= need) { bin = i; break; }
        cum += h;
    }
    out[0] = bin;
    out[1] = cum;
}

// 填 keep（读已算好的 scores）：并列按块下标升序取前 (topk - cnt_gt) 个。
extern "C" __global__ void lingxi_qsa_keep_fill_scores(const float* __restrict__ scores,
                                                       unsigned char* __restrict__ keep,
                                                       int nblocks, int topk,
                                                       unsigned int thr, int need_eq) {
    __shared__ int cnt[1024];
    __shared__ int off[1024];
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int per = (nblocks + nthr - 1) / nthr;
    const int lo = tid * per;
    const int hi = min(nblocks, lo + per);
    int c = 0;
    for (int b = lo; b < hi; ++b) c += (__float_as_uint(scores[b]) == thr);
    cnt[tid] = c;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nthr; ++i) { off[i] = acc; acc += cnt[i]; }
    }
    __syncthreads();
    int r = off[tid];
    for (int b = lo; b < hi; ++b) {
        const unsigned int kk = __float_as_uint(scores[b]);
        unsigned char v;
        if (kk > thr) {
            v = 1u;
        } else if (kk == thr) {
            v = (r < need_eq) ? 1u : 0u;
            ++r;
        } else {
            v = 0u;
        }
        keep[b] = v;
    }
}

// ───────────────── 选块列表（gather 路径）─────────────────
//
// 动机（2026-10-18，256K）：`attn3_body<SPARSE>` 虽然跳过未选中块的**点积**，但仍要
// 逐 key 扫全量 Tk（每个 key 每行都要查 keep + 分支）⇒ 单 chunk O(t·Tk)，跨 chunk 累计
// O(S²)。256K 下这是 prefill 的主项（实测 16K 上下文 143ms/层/chunk，外推 256K ≈ 2.3s）。
// 改成"先把选中块压成升序列表，再只遍历列表"⇒ 每行恒定 2048 token，与上下文长度无关。
//
// 语义：列表里只放 keep=1 的块；因果可见性与尾部不完整块由注意力核按 `vis` 过滤/补扫。

// 3a) 每行一个 block：把 keep=1 的块号按升序压进 `sel[row*stride + j]`，个数写 `sel_cnt[row]`。
extern "C" __global__ void lingxi_qsa_sel_build(const unsigned char* __restrict__ keep,
                                                int* __restrict__ sel,
                                                int* __restrict__ sel_cnt,
                                                int nblocks, int stride) {
    __shared__ int cnt[1024];
    __shared__ int off[1024];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const unsigned char* kp = keep + (size_t)row * nblocks;
    int* sp = sel + (size_t)row * stride;
    const int per = (nblocks + nthr - 1) / nthr;
    const int lo = tid * per;
    const int hi = min(nblocks, lo + per);
    int c = 0;
    for (int b = lo; b < hi; ++b) c += (kp[b] != 0u);
    cnt[tid] = c;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nthr; ++i) { off[i] = acc; acc += cnt[i]; }
        sel_cnt[row] = acc;
    }
    __syncthreads();
    int w = off[tid];
    for (int b = lo; b < hi; ++b) {
        if (kp[b] != 0u) sp[w++] = b;
    }
}

// 3b) gather 注意力：每个 (row, head) 一个 block（256 线程 = head_dim）。
//     第 1 遍按列表算点积并把分数留在 shared（不重读 K）；第 2 遍只读 V 做加权和。
//     `vis = vis_base + row + 1`；列表里超出 vis 的块跳过；尾部不完整块（pos ≥ nblocks<<cshift）
//     恒可见，放在列表之后补扫。
//
// ★ 2026-10-18：**NH 个 q-head 合并进一个 block**。本模型 heads=24 / kv_heads=2 ⇒
//   12 个 q-head 共用同一份 K/V；旧版每个 (row, head) 各读一遍 K/V（实测 gather 60ms/层，
//   49GB/层 ≈ 816GB/s 正好打满 HBM）⇒ **12× 冗余搬运**。现在一个 block 处理同一 kv-group
//   内的 NH 个 head：K/V 只读一次、喂 NH 个点积/累加（NH=4 ⇒ K/V 流量 ÷4）。
//   **逐位一致**：每个 head 的点积仍是 lane 内 z=0..7 顺序累加 + 同一 warp shuffle；
//   softmax 的 max/分母/加权和仍按 idx 升序、每个 head 独立。
// P2 Stage 1（2026-10-20）：KV 块表间接寻址——`blk_tab[逻辑块号] = 物理槽号`。
// Stage 1 表恒等映射（槽号==逻辑块号，零行为变化），为 Stage 2 host 分页铺路；
// 间接粒度 = indexer 块（ctok = 1<<cshift，与 sel 单位 1:1）。
__device__ __forceinline__ int kv_phys_row(int pos, const int* __restrict__ blk_tab,
                                           int cshift, int ctok) {
    return blk_tab[pos >> cshift] * ctok + (pos & (ctok - 1));
}

template <int NH>
__device__ __forceinline__ void attn_gather_body(const bf16* __restrict__ q,
                                                 const bf16* __restrict__ k,
                                                 const bf16* __restrict__ v,
                                                 const int* __restrict__ blk_tab,
                                                 const int* __restrict__ sel,
                                                 const int* __restrict__ sel_cnt,
                                                 bf16* __restrict__ out,
                                                 int heads, int kv_heads, int head_dim,
                                                 float scale, int vis_base, int nblocks,
                                                 int cshift, int stride) {
    extern __shared__ float sp[];   // [NH][stride*ctok + ctok] 分数
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
    int4 qreg[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) {
        qreg[g] = make_int4(0, 0, 0, 0);
        if (head_dim == 256 && lane < 32) {
            const bf16* qv = q + ((size_t)i * heads + h0 + g) * head_dim;
            qreg[g] = *reinterpret_cast<const int4*>(qv + lane * 8);
        }
    }

    // 枚举第 idx 个候选 token 的绝对位置（列表块 + 尾部）
    const int ntail = (vis > tail0) ? (vis - tail0) : 0;
    const int total = nsel * ctok + ntail;
    auto pos_of = [&](int idx) -> int {
        const int nb = nsel * ctok;
        if (idx < nb) return srow[idx >> cshift] * ctok + (idx & (ctok - 1));
        return tail0 + (idx - nb);
    };

    // ── 第 1 遍：点积 + 行内 max（K 只读一次，NH 个 head 共用）──
    float lmax[NH];
#pragma unroll
    for (int g = 0; g < NH; ++g) lmax[g] = -INFINITY;
    for (int idx = warp; idx < total; idx += nwarp) {
        const int pos = pos_of(idx);
        if (pos >= vis) {  // 列表里可能有超出因果前沿的块
            if (lane == 0) {
#pragma unroll
                for (int g = 0; g < NH; ++g) sp[g * span + idx] = -INFINITY;
            }
            continue;
        }
        const int phys = kv_phys_row(pos, blk_tab, cshift, ctok);
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

    // ── 第 2 遍：exp **每 token 只算一次**并就地写回 shared ──
    // 旧版把 `__expf(sp[idx]-mx)` 放在"每线程固定一个 d"的循环里 ⇒ 256 个线程
    // 对同一个 idx 各算一遍（**256× 重复计算**）。现在并行算一次、各自读 shared。
    for (int idx = tid; idx < total; idx += (int)blockDim.x) {
#pragma unroll
        for (int g = 0; g < NH; ++g) sp[g * span + idx] = __expf(sp[g * span + idx] - smx[g]);
    }
    __syncthreads();
    // 分母用 tid==0 顺序累加（未选中 = exp(-inf) = 0 精确不加）
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
    if (tid < head_dim) {
        float acc[NH];
#pragma unroll
        for (int g = 0; g < NH; ++g) acc[g] = 0.0f;
        // 每个线程固定一个 d，按 idx 升序累加；V 只读一次、NH 个 head 共用
        for (int idx = 0; idx < total; ++idx) {
            const int pos = pos_of(idx);
            const int phys = kv_phys_row(pos, blk_tab, cshift, ctok);
            const bf16* vv = v + ((size_t)phys * kv_heads + kh) * head_dim + tid;
            const float vx = __bfloat162float(vv[0]);
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

extern "C" __global__ void lingxi_attn_gather_bf16(const bf16* __restrict__ q,
                                                   const bf16* __restrict__ k,
                                                   const bf16* __restrict__ v,
                                                   const int* __restrict__ blk_tab,
                                                   const int* __restrict__ sel,
                                                   const int* __restrict__ sel_cnt,
                                                   bf16* __restrict__ out,
                                                   int heads,
                                                   int kv_heads,
                                                   int head_dim,
                                                   float scale,
                                                   int vis_base,
                                                   int nblocks,
                                                   int cshift,
                                                   int stride) {
    attn_gather_body<1>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim,
                        scale, vis_base, nblocks, cshift, stride);
}

// NH=4：同一 kv-group 的 4 个 q-head 合并（K/V 流量 ÷4）。
extern "C" __global__ void lingxi_attn_gather4_bf16(const bf16* __restrict__ q,
                                                    const bf16* __restrict__ k,
                                                    const bf16* __restrict__ v,
                                                    const int* __restrict__ blk_tab,
                                                    const int* __restrict__ sel,
                                                    const int* __restrict__ sel_cnt,
                                                    bf16* __restrict__ out,
                                                    int heads,
                                                    int kv_heads,
                                                    int head_dim,
                                                    float scale,
                                                    int vis_base,
                                                    int nblocks,
                                                    int cshift,
                                                    int stride) {
    attn_gather_body<4>(q, k, v, blk_tab, sel, sel_cnt, out, heads, kv_heads, head_dim,
                        scale, vis_base, nblocks, cshift, stride);
}

// 4) 掩码填充：可见 = 因果前缀 ∧ (选中块 ∨ 尾部不完整块)；未选中再叠加 min
//    grid = (t_kv 方向, t_q)；`vis = len + row + 1`
extern "C" __global__ void lingxi_qsa_mask_fill_f32(const unsigned char* __restrict__ keep,
                                                    float* __restrict__ mask,
                                                    int t_q, int t_kv, int nblocks, int compress,
                                                    int len, float minv) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (j >= t_kv || row >= t_q) {
        return;
    }
    const int vis = len + row + 1;
    float v = (j < vis) ? 0.0f : minv;
    const bool tail = (j >= nblocks * compress);
    const bool sel = tail || (nblocks > 0 && keep[(size_t)row * nblocks + j / compress] != 0u);
    if (!sel) {
        v += minv;
    }
    mask[(size_t)row * t_kv + j] = v;
}
