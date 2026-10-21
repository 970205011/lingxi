// GDN n_seq=2 批跑 ≡ 两条 n_seq=1 各自单跑（逐位对拍，P1 第一步）
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using bf16 = __nv_bfloat16;

extern "C" __global__ void gdn_seq(
    const bf16* __restrict__ conv, const bf16* __restrict__ z,
    const bf16* __restrict__ agate, const bf16* __restrict__ bgate,
    const bf16* __restrict__ norm_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ s_all,
    bf16* __restrict__ gated,
    int T, int hk, int hv, int num_v_heads, int num_k_heads, int gate_act, int row_off,
    int use_smem, int n_seq) {
    const int nvh = num_v_heads;
    const int vh = blockIdx.x % nvh;
    const int seq = blockIdx.x / nvh;
    const int dv = threadIdx.x;
    const int key_dim = num_k_heads * hk;
    const int value_dim = num_v_heads * hv;
    const int conv_dim = 2 * key_dim + value_dim;
    const int rep = num_v_heads / num_k_heads;
    const int hh = vh / rep;
    __shared__ float ssum[32];
    float* S = s_all + ((size_t)seq * nvh + vh) * hk * hv;
    float* Sw = S;
    for (int i = 0; i < T; ++i) {
        const bf16* conv_i = conv + (size_t)(i * n_seq + seq) * conv_dim;
        const bf16* qrow = conv_i + (size_t)hh * hk;
        const bf16* krow = conv_i + key_dim + (size_t)hh * hk;
        float qss = 0.0f, kss = 0.0f;
        for (int d = 0; d < hk; ++d) {
            const float qv = __bfloat162float(qrow[d]);
            const float kv = __bfloat162float(krow[d]);
            qss += qv * qv; kss += kv * kv;
        }
        const float qscale = (1.0f / sqrtf(qss + 1e-6f)) / sqrtf((float)hk);
        const float kscale = 1.0f / sqrtf(kss + 1e-6f);
        const int r_i = i * n_seq + seq;
        const float av = __bfloat162float(agate[(size_t)r_i * num_v_heads + vh]) + dt_bias[vh];
        const float sp = (av > 20.0f) ? av : log1pf(expf(av));
        const float decay = expf(-expf(a_log[vh]) * sp);
        const float bv = __bfloat162float(bgate[(size_t)r_i * num_v_heads + vh]);
        const float beta = 1.0f / (1.0f + expf(-bv));
        float kv_mem = 0.0f;
        for (int dk = 0; dk < hk; ++dk) {
            const float s = S[(size_t)dk * hv + dv] * decay;
            S[(size_t)dk * hv + dv] = s;
            kv_mem += s * (__bfloat162float(krow[dk]) * kscale);
        }
        const float v = __bfloat162float(conv_i[2 * key_dim + (size_t)vh * hv + dv]);
        const float delta = (v - kv_mem) * beta;
        float acc = 0.0f;
        for (int dk = 0; dk < hk; ++dk) {
            const float kn = __bfloat162float(krow[dk]) * kscale;
            const float su = S[(size_t)dk * hv + dv] + kn * delta;
            S[(size_t)dk * hv + dv] = su;
            acc += su * (__bfloat162float(qrow[dk]) * qscale);
        }
        float sq = acc * acc;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, off);
        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        if (lane == 0) ssum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            const int nwarp = (blockDim.x + 31) >> 5;
            float v2 = (lane < nwarp) ? ssum[lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v2 += __shfl_down_sync(0xffffffffu, v2, off);
            if (lane == 0) ssum[0] = v2;
        }
        __syncthreads();
        const float r = 1.0f / sqrtf(ssum[0] / (float)hv + 1e-6f);
        const float zz = __bfloat162float(z[(size_t)r_i * value_dim + (size_t)vh * hv + dv]);
        const float actv = (gate_act == 0) ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
        gated[(size_t)r_i * value_dim + (size_t)vh * hv + dv] =
            __float2bfloat16(acc * r * __bfloat162float(norm_w[dv]) * actv);
        __syncthreads();
    }
}

int main() {
    const int hk = 128, hv = 128, nvh = 48, nkh = 16;
    const int key_dim = nkh * hk, value_dim = nvh * hv, conv_dim = 2 * key_dim + value_dim;
    const int T = 7;              // 模拟一次 spec verify 窗
    const int NS = 2;
    bf16 *conv, *z, *ag, *bg, *nw, *g1, *g2;
    float *alog, *dtb, *s1, *s2;
    cudaMalloc(&conv, (size_t)T * NS * conv_dim * 2);
    cudaMalloc(&z, (size_t)T * NS * value_dim * 2);
    cudaMalloc(&ag, (size_t)T * NS * nvh * 2);
    cudaMalloc(&bg, (size_t)T * NS * nvh * 2);
    cudaMalloc(&nw, hv * 2);
    cudaMalloc(&g1, (size_t)T * NS * value_dim * 2);
    cudaMalloc(&g2, (size_t)T * NS * value_dim * 2);
    cudaMalloc(&alog, nvh * 4);
    cudaMalloc(&dtb, nvh * 4);
    cudaMalloc(&s1, (size_t)NS * nvh * hk * hv * 4);
    cudaMalloc(&s2, (size_t)NS * nvh * hk * hv * 4);
    bf16 *hconv = (bf16*)malloc((size_t)T * NS * conv_dim * 2);
    bf16 *hz = (bf16*)malloc((size_t)T * NS * value_dim * 2);
    bf16 *hag = (bf16*)malloc((size_t)T * NS * nvh * 2);
    bf16 *hbg = (bf16*)malloc((size_t)T * NS * nvh * 2);
    bf16 *hnw = (bf16*)malloc(hv * 2);
    float *halog = (float*)malloc(nvh * 4);
    float *hdtb = (float*)malloc(nvh * 4);
    srand(7);
    auto fr = []() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; };
    for (int i = 0; i < T * NS * conv_dim; ++i) hconv[i] = __float2bfloat16(fr());
    for (int i = 0; i < T * NS * value_dim; ++i) hz[i] = __float2bfloat16(fr());
    for (int i = 0; i < T * NS * nvh; ++i) { hag[i] = __float2bfloat16(fr()); hbg[i] = __float2bfloat16(fr()); }
    for (int i = 0; i < hv; ++i) hnw[i] = __float2bfloat16(0.5f + fr() * 0.5f);
    for (int i = 0; i < nvh; ++i) { halog[i] = fr(); hdtb[i] = fr(); }
    cudaMemcpy(conv, hconv, (size_t)T * NS * conv_dim * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(z, hz, (size_t)NS * T * value_dim * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(ag, hag, (size_t)NS * T * nvh * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(bg, hbg, (size_t)NS * T * nvh * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(nw, hnw, hv * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(alog, halog, nvh * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtb, hdtb, nvh * 4, cudaMemcpyHostToDevice);
    // 参考：seq0 与 seq1 各自 n_seq=1 跑（输入取各自交错行）
    cudaMemset(s1, 0, (size_t)NS * nvh * hk * hv * 4);
    cudaMemset(s2, 0, (size_t)NS * nvh * hk * hv * 4);
    for (int sq = 0; sq < NS; ++sq) {
        gdn_seq<<<nvh, hv>>>(conv + (size_t)sq * T * conv_dim, z + (size_t)sq * T * value_dim,
                             ag + (size_t)sq * T * nvh, bg + (size_t)sq * T * nvh,
                             nw, alog, dtb, s1 + (size_t)sq * nvh * hk * hv,
                             g1 + (size_t)sq * T * value_dim,
                             T, hk, hv, nvh, nkh, 0, 0, 0, 1);
    }
    // 批跑：n_seq=2，交错布局
    gdn_seq<<<nvh * NS, hv>>>(conv, z, ag, bg, nw, alog, dtb, s2, g2, T, hk, hv, nvh, nkh, 0, 0, 0, NS);
    cudaDeviceSynchronize();
    bf16 *hg1 = (bf16*)malloc((size_t)T * NS * value_dim * 2);
    bf16 *hg2 = (bf16*)malloc((size_t)T * NS * value_dim * 2);
    cudaMemcpy(hg1, g1, (size_t)T * NS * value_dim * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hg2, g2, (size_t)T * NS * value_dim * 2, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < T * NS * value_dim && bad < 5; ++i) {
        if (__bfloat162float(hg1[i]) != __bfloat162float(hg2[i])) {
            printf("gated MISMATCH @%d: solo=%f batch=%f\n", i,
                   __bfloat162float(hg1[i]), __bfloat162float(hg2[i]));
            ++bad;
        }
    }
    if (!bad) printf("gated: n_seq=2 与两条单跑 **逐位一致** (%d elems)\n", T * NS * value_dim);
    float *hs1 = (float*)malloc((size_t)NS * nvh * hk * hv * 4);
    float *hs2 = (float*)malloc((size_t)NS * nvh * hk * hv * 4);
    cudaMemcpy(hs1, s1, (size_t)NS * nvh * hk * hv * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hs2, s2, (size_t)NS * nvh * hk * hv * 4, cudaMemcpyDeviceToHost);
    int sbad = 0;
    for (int i = 0; i < NS * nvh * hk * hv && sbad < 5; ++i) {
        if (hs1[i] != hs2[i]) { printf("S MISMATCH @%d: %f vs %f\n", i, hs1[i], hs2[i]); ++sbad; }
    }
    if (!sbad) printf("S: 逐位一致 (%d elems)\n", NS * nvh * hk * hv);
    printf(bad || sbad ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return 0;
}
