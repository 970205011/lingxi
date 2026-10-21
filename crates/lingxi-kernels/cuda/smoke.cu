// L1 smoke kernel —— 打通 nvcc → cubin → cudarc launch 链路。
// 规范：纯函数，无全局、无 alloc，stream 由调用方显式传入。

extern "C" __global__ void lingxi_smoke_scale(const float* __restrict__ x,
                                              float a,
                                              float* __restrict__ y,
                                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * a;
    }
}
