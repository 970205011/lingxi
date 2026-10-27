// L1 RMSNorm（f32 契约测试版）：y[r,i] = x[r,i] / rms(x[r,:]) * gamma[i]
// rms(x) = sqrt(mean_i x^2 + eps)
// 一 block 处理一行，block 内归约。纯函数；gamma 已由转换器烘焙好。

extern "C" __global__ void lingxi_rmsnorm_f32(const float* __restrict__ x,
                                              const float* __restrict__ gamma,
                                              float* __restrict__ y,
                                              int dim,
                                              float eps) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * (size_t)dim;
    float* yr = y + (size_t)row * (size_t)dim;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = xr[i];
        acc += v * v;
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float rms = rsqrtf(sdata[0] / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        yr[i] = xr[i] * rms * gamma[i];
    }
}
