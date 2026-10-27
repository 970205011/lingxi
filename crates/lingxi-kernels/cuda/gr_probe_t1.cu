// GR（GatedResidual）**t=1（草稿占位）** 的取数分母探针：只有 microbench 会调。
//
// 为什么需要：`t9_gr_shape_bw` 已经把 rows=1 的 5 个核逐计时（rmsnorm / mix_down /
// mix_up / mix_mean / block_inject ≈ 8.3/9.5/23.5/7.6/7.7 µs），但**没有同形分母**，
// 于是"这个核慢不慢"只能靠空核地板猜。这里给两条与生产核**逐字相同的地址流**、
// 只读不算的版本：
// * `lingxi_probe_read_row8_bf16`：对应 `lingxi_gemv_bf16_row8_epi`（mix_down / block_inject
// 的 t=1 路径）——一行一 block、每线程 int4（8×bf16）步进、warp shfl + smem 两级归约；
// * `lingxi_probe_read_warp_bf16`：对应 `lingxi_gemv_bf16_epi`（mix_up，k=320 → warp 一行、
// lane 沿 k 标量步进）。
// 只把"乘 x + 累加"换成"只累加"，其余（grid/block/向量宽度/步长/归约树）一字不动。

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// 一行一个 block；地址流 = `lingxi_gemv_bf16_row8_epi`（grid = n、block = 1024）。
extern "C" __global__ void lingxi_probe_read_row8_bf16(const bf16* __restrict__ w,
                                                       float* __restrict__ out,
                                                       int k) {
    const int row = blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const bf16* wr = w + (size_t)row * (size_t)k;
    float acc = 0.0f;
    const int vec_k = k & ~7;
    for (int j = tid * 8; j < vec_k; j += (int)blockDim.x * 8) {
        const int4 wv = *reinterpret_cast<const int4*>(wr + j);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc += __bfloat162float(wp[t]);
        }
    }
    for (int j = vec_k + tid; j < k; j += (int)blockDim.x) {
        acc += __bfloat162float(wr[j]);
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
    if (tid == 0) {
        float v = 0.0f;
        for (int i = 0; i < (int)(blockDim.x >> 5); ++i) {
            v += sred[i];
        }
        out[row] = v;
    }
}

// 一个 warp 一行；地址流 = `lingxi_gemv_bf16_epi`（grid = n/32、block = 1024，k=320）。
extern "C" __global__ void lingxi_probe_read_warp_bf16(const bf16* __restrict__ w,
                                                       float* __restrict__ out,
                                                       int n, int k) {
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (warp >= n) {
        return;
    }
    const bf16* wr = w + (size_t)warp * (size_t)k;
    float acc = 0.0f;
    for (int j = lane; j < k; j += 32) {
        acc += __bfloat162float(wr[j]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[warp] = acc;
    }
}

// 同字节的**纯流式读**（grid = k/8 个线程块，128 位读）：给"这些字节在一个能打满
// HBM 的形状下最快多久"当分母（对照上面两条"线上形状"的地址流）。
extern "C" __global__ void lingxi_probe_read_stream_bf16(const bf16* __restrict__ w,
                                                         float* __restrict__ out,
                                                         long long n) {
    const long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float acc = 0.0f;
    if (i + 8 <= n) {
        const int4 wv = *reinterpret_cast<const int4*>(w + i);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc += __bfloat162float(wp[t]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = acc;
    }
}
