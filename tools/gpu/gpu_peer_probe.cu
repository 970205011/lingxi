// 跨卡（GPU0 ↔ GPU1）稳定性/带宽探针 —— 决定"GPU1 拿 32 GB 做专家常驻"这条路能不能走。
//
// 为什么需要：
// 单流 26.8 ms/token 里 **26% 花在 PCIe 上**（专家 Q4 不在显存里，7000 槽 / 24576 专家，
// 稳态 93% 命中 ⇒ 每 token 89 MB 过 PCIe）。单卡加槽到 9500 就 OOM ⇒ 单卡无解。
// 唯一还有 >1.5× 潜力的结构改动是把专家摊到两块卡（2×32 GB）。
// **但 project_memory 记着：GPU1（index1 = PCI 44:00.0，x8 链路）在高跨卡流量下会触发
// PCIe CmpltTO 导致物理掉线。** 本探针要回答三件事：
// ① P2P（`cudaDeviceCanAccessPeer` / `cudaMemcpyPeerAsync`）在这对卡上到底能不能用；
// ② 设计口径（每层 ~56 KB、每 token ~2.7 MB、约 80 MB/s）长时间跑稳不稳、带宽多少；
// ③ 「把专家灌进 GPU1」（host→GPU1 一次性 ~30 GB）这个动作能不能扛。
//
// 用法（CT141 内）：
// /usr/local/cuda-12.8/bin/nvcc -O2 -arch=sm_70 -o /tmp/gpu_peer_probe gpu_peer_probe.cu
// /tmp/gpu_peer_probe 2000 8 # 迭代 token 数、host→GPU1 的 GiB
// 之后在宿主上看 `dmesg | tail -50` 有没有 Xid / CmpltTO。
//
// ⚠️ 本探针**只读**：不碰 /dev/shm、不碰 arena、不写任何模型文件。GPU1 掉卡只影响本进程。
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>

static const char* err(cudaError_t e) { return cudaGetErrorString(e); }

// 记录所有错误（不早退：要看到"第几次开始出错"）
struct Tally {
    long ok = 0, bad = 0;
    cudaError_t first = cudaSuccess;
    long first_at = -1;
    void note(cudaError_t e, long idx, const char* what) {
        if (e == cudaSuccess) { ok++; return; }
        if (bad == 0) { first = e; first_at = idx; printf("  ✗ 首次失败 %s @%ld: %s\n", what, idx, err(e)); }
        bad++;
        if (bad <= 5) printf("  ✗ %s @%ld: %s\n", what, idx, err(e));
    }
    void report(const char* tag) {
        printf("  [%s] ok=%ld bad=%ld", tag, ok, bad);
        if (bad) printf("  首次错误=%s @%ld", err(first), first_at);
        printf("\n");
    }
};

// GPU0 上的"计算负载"：模拟真实 decode 时 GPU0 是忙的（GPU1 只跑专家 GEMV）。
__global__ void busy_kernel(float* p, int n, int iters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = p[i];
    for (int k = 0; k < iters; ++k) v = fmaf(v, 1.0000001f, 1e-7f);
    p[i] = v;
}

int main(int argc, char** argv) {
    const long n_tok   = (argc > 1) ? atol(argv[1]) : 2000;   // 模拟多少 token
    const double gpu1_gib = (argc > 2) ? atof(argv[2]) : 8.0; // host→GPU1 灌多少 GiB

    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 2) { printf("需要 ≥2 张卡，实际 %d\n", nd); return 1; }

    int cap01 = 0, cap10 = 0;
    cudaDeviceCanAccessPeer(&cap01, 0, 1);
    cudaDeviceCanAccessPeer(&cap10, 1, 0);
    printf("== ① 能力 ==\n  cudaDeviceCanAccessPeer(0→1)=%d  (1→0)=%d\n", cap01, cap10);
    int peer_on = 0, peer_on10 = 0;
    if (cap01) { cudaSetDevice(0); peer_on = (cudaDeviceEnablePeerAccess(1, 0) == cudaSuccess); }
    if (cap10) { cudaSetDevice(1); peer_on10 = (cudaDeviceEnablePeerAccess(0, 0) == cudaSuccess); }
    printf("  EnablePeerAccess 0→1=%d 1→0=%d\n", peer_on, peer_on10);

    cudaDeviceProp p0{}, p1{};
    cudaGetDeviceProperties(&p0, 0);
    cudaGetDeviceProperties(&p1, 1);
    printf("  GPU0=%s  总线 %04x:%02x  sm %d.%d  显存 %.1f GiB\n", p0.name, p0.pciDomainID, p0.pciBusID, p0.major, p0.minor, p0.totalGlobalMem / 1073741824.0);
    printf("  GPU1=%s  总线 %04x:%02x  sm %d.%d  显存 %.1f GiB\n", p1.name, p1.pciDomainID, p1.pciBusID, p1.major, p1.minor, p1.totalGlobalMem / 1073741824.0);

    // ── 两块卡上的缓冲 ──
    const size_t SMALL = 5120;              // 5 KB：t×hidden bf16（一行 hidden 的激活）
    const size_t BLOB  = 2611200;           // 2.61 MB：一个专家的 Q4 blob（真值）
    const size_t BIG   = 64u << 20;         // 64 MB
    float *g0 = nullptr, *g1 = nullptr;
    cudaSetDevice(0); cudaMalloc(&g0, BIG); cudaMemset(g0, 1, BIG);
    char *s0 = nullptr, *s1 = nullptr, *b0 = nullptr, *b1 = nullptr, *big0 = nullptr, *big1 = nullptr;
    cudaMalloc(&s0, SMALL); cudaMalloc(&b0, BLOB); cudaMalloc(&big0, BIG); cudaMemset(s0, 1, SMALL); cudaMemset(b0, 1, BLOB); cudaMemset(big0, 2, BIG);
    cudaSetDevice(1); cudaMalloc(&s1, SMALL); cudaMalloc(&b1, BLOB); cudaMalloc(&big1, BIG);

    auto bench = [&](const char* tag, void* dst, void* src, size_t bytes, int flag) -> double {
        cudaSetDevice(0);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        // ⚠️ 只跑 5 次：2026-10-16 实测这台机器的 **IOMMU（DMAR）会挡掉 P2P DMA**
        // （`DMAR: [DMA Write NO_PASID] Request device [44:00.0] fault ... PTE Write access is not set`），
        // 每次失败都在刷内核日志 ⇒ 次数必须小，别把 dmesg 刷爆。
        const int R = 5;
        cudaError_t e0 = cudaMemcpyPeerAsync(dst, 1, src, 0, bytes, 0); // 预热 + 取错误码
        cudaError_t e1 = cudaDeviceSynchronize();
        if (e0 != cudaSuccess || e1 != cudaSuccess) {
            printf("  %-28s ❌ peer 不可用：调用=%s sync=%s\n", tag, err(e0), err(e1));
            cudaEventDestroy(a); cudaEventDestroy(b);
            return -1;
        }
        cudaEventRecord(a);
        for (int i = 0; i < R; ++i) cudaMemcpyPeerAsync(dst, 1, src, 0, bytes, 0);
        cudaEventRecord(b);
        (void)flag;
        if (cudaDeviceSynchronize() != cudaSuccess) { printf("  [%s] 失败: %s\n", tag, err(cudaGetLastError())); return -1; }
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        const double gbs = (double)bytes * R / (ms / 1e3) / 1e9;
        printf("  %-28s %8.1f µs/次  →  %7.2f GB/s\n", tag, ms * 1e3 / R, gbs);
        cudaEventDestroy(a); cudaEventDestroy(b);
        return gbs;
    };
    printf("\n== ② 单向带宽（GPU0 → GPU1）==\n");
    double bw5 = bench("peer 5 KB (激活)", s1, s0, SMALL, 0);
    if (bw5 > 0) bench("peer 2.61 MB (单专家 blob)", b1, b0, BLOB, 0);
    // ⚠️ `CanAccessPeer` 与 `EnablePeerAccess` **都返回 1**，但实测 5 KB 要 49 ms（0.0001 GB/s）——
    // DMA 撞 IOMMU（DMAR）后走的是"故障+回退"路。所以**门槛必须卡在带宽上**，
    // 不能只看那两个 API 的返回值。<0.1 GB/s = 实际上不可用。
    const bool peer_ok = (bw5 > 0.1);
    printf("  ⇒ P2P 可用性：%s\n", peer_ok ? "可用" : "**实际不可用（IOMMU 挡 DMA；API 说可用但带宽 0.0001 GB/s）**");

    double bw_local = -1;
    if (peer_ok) {
        printf("== 反向 + 本机对照 ==\n");
        cudaSetDevice(0);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        const int R = 5;
        for (int rep = 0; rep < 2; ++rep) {
            const bool rev = (rep == 0);
            void* d = rev ? b0 : b0; void* s = rev ? b1 : big0;
            const int dstdev = rev ? 0 : 0, srcdev = rev ? 1 : 0;
            cudaDeviceSynchronize();
            cudaEventRecord(a);
            for (int i = 0; i < R; ++i) {
                if (rev) cudaMemcpyPeerAsync(d, dstdev, s, srcdev, BLOB, 0);
                else     cudaMemcpyAsync(d, s, BLOB, cudaMemcpyDeviceToDevice, 0);
            }
            cudaEventRecord(b); cudaDeviceSynchronize();
            float ms = 0; cudaEventElapsedTime(&ms, a, b);
            printf("  %-28s %8.1f µs/次  →  %7.2f GB/s\n",
                   rev ? "peer 2.61 MB (GPU1→GPU0)" : "本机 HBM 2.61 MB (对照)",
                   ms * 1e3 / R, (double)BLOB * R / (ms / 1e3) / 1e9);
            if (!rev) bw_local = (double)BLOB * R / (ms / 1e3) / 1e9;
        }
        cudaEventDestroy(a); cudaEventDestroy(b);
    }

    // ── ②' **host 中转**（P2P 被 IOMMU 挡死后的唯一选择）：GPU0→锁页host→GPU1 ──
    // 这是"专家分片到 GPU1"真正要付的代价：**每层一次同步往返的延迟**。
    // 判据：`48 × 往返延迟` 必须远小于它要省下的 7 ms/token（专家 H2D），否则不值。
    {
        printf("\n== ②' host 中转（GPU0→pinned→GPU1）—— 同步往返延迟 ==\n");
        const size_t RET = 51200;
        void *hp_s = nullptr, *hp_r = nullptr;
        cudaHostAlloc(&hp_s, SMALL, cudaHostAllocDefault);
        cudaHostAlloc(&hp_r, RET, cudaHostAllocDefault);
        char* r0 = nullptr; cudaSetDevice(1); char* r1 = nullptr;
        cudaMalloc(&r1, RET); cudaMemset(r1, 3, RET);
        cudaSetDevice(0); cudaMalloc(&r0, RET); cudaMemset(r0, 4, RET);

        const int N = 200;
        double t_go = 0, t_back = 0;
        // 去：GPU0 → host → GPU1（5 KB）
        auto a = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            cudaSetDevice(0); cudaMemcpyAsync(hp_s, s0, SMALL, cudaMemcpyDeviceToHost, 0); cudaDeviceSynchronize();
            cudaSetDevice(1); cudaMemcpyAsync(s1, hp_s, SMALL, cudaMemcpyHostToDevice, 0); cudaDeviceSynchronize();
        }
        t_go = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() / N * 1e6;
        // 回：GPU1 → host → GPU0（51.2 KB）
        a = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            cudaSetDevice(1); cudaMemcpyAsync(hp_r, r1, RET, cudaMemcpyDeviceToHost, 0); cudaDeviceSynchronize();
            cudaSetDevice(0); cudaMemcpyAsync(r0, hp_r, RET, cudaMemcpyHostToDevice, 0); cudaDeviceSynchronize();
        }
        t_back = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() / N * 1e6;
        printf("  去 5 KB  往返延迟 %8.1f µs/层次\n", t_go);
        printf("  回 51 KB 往返延迟 %8.1f µs/层次\n", t_back);
        printf("  ⇒ 48 层/token 合计 **%.2f ms/token**（对比：它要替掉的专家 H2D = 7 ms/token）\n",
               (t_go + t_back) * 48 / 1000.0);
        printf("  ⇒ 判定：%s\n", ((t_go + t_back) * 48 / 1000.0 < 3.0)
               ? "**可能值**（延迟 ≤3 ms/token）" : "**不值**（延迟吃掉大半收益）");
        cudaFreeHost(hp_s); cudaFreeHost(hp_r); cudaFree(r0); cudaFree(r1);
    }

    // ── ③ 设计口径的持续压力：每层 56 KB（5 KB 去 + 51 KB 回），48 层/token ──
    // 同时 GPU0 上有常驻计算负载（真实 decode 时 GPU0 是忙的）。
    // ⚠️ P2P 不可用时**整段跳过**：否则每层每 token 都去撞 IOMMU，dmesg 会被刷爆。
    Tally ts;
    if (peer_ok) {
    printf("\n== ③ 设计口径持续压力：%ld token × 48 层 × (5 KB 去 + 51 KB 回) = %.1f GB 跨卡 ==\n",
           n_tok, (double)n_tok * 48 * (SMALL + 51200) / 1e9);
    const size_t RET = 51200; // 10 行 × 2560 × bf16 = 51.2 KB
    char* ret1 = nullptr; char* ret0 = nullptr;
    cudaSetDevice(1); cudaMalloc(&ret1, RET); cudaMemset(ret1, 3, RET);
    cudaSetDevice(0); cudaMalloc(&ret0, RET);

    // GPU0 的常驻计算负载（另开一条 stream，别把拷贝和计算串在一起）。
    // 强度刻意压在 ~70% 占空比：太满会把跨卡拷贝完全掩住，就看不出跨卡本身的问题了。
    cudaStream_t cs;
    cudaSetDevice(0); cudaStreamCreate(&cs);
    volatile bool stop_load = false;
    std::thread load([&] {
        cudaSetDevice(0);
        while (!stop_load) {
            busy_kernel<<<1024, 256, 0, cs>>>(g0, 1 << 20, 30);
            cudaStreamSynchronize(cs);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    auto t_start = std::chrono::steady_clock::now();
    long n_calls = 0;
    for (long tk = 0; tk < n_tok; ++tk) {
        for (int L = 0; L < 48; ++L) {
            cudaError_t e1 = cudaMemcpyPeerAsync(s1, 1, s0, 0, SMALL, 0);
            cudaError_t e2 = cudaMemcpyPeerAsync(ret0, 0, ret1, 1, RET, 0);
            ts.note(e1, n_calls, "5KB 去"); ts.note(e2, n_calls, "51KB 回");
            n_calls++;
            if (ts.bad > 20) { printf("  ⚠️ 错误过多，提前停下\n"); goto done; }
        }
        if (tk % 200 == 0) {
            cudaError_t es = cudaDeviceSynchronize();
            ts.note(es, tk, "sync");
            printf("    %5ld/%ld token  ok=%ld bad=%ld\n", tk, n_tok, ts.ok, ts.bad);
            fflush(stdout);
        }
    }
done:
    cudaDeviceSynchronize();
    {
        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        printf("  完成 %ld 层对 %ld 次拷贝，用时 %.1f s  ⇒  单 token 跨卡 %.2f MB 时约 %.1f ms/token\n",
               n_calls * 2, n_calls, dt, (SMALL + RET) * 48.0 / 1e6, dt / (n_tok ? n_tok : 1) * 1e3);
    }
    ts.report("③ 设计口径");
    stop_load = true;
    load.join();
    cudaStreamDestroy(cs);
    } else {
        printf("\n== ③ 设计口径持续压力：**跳过**（P2P 不可用；继续撞 IOMMU 只会刷爆 dmesg）==\n");
        printf("  ⇒ 跨卡只能走 **host 中转**（GPU0→host→GPU1）：多一次 PCIe 往返、带宽约减半，\n");
        printf("     但对本设计（80 MB/s）仍然够用；真正的代价是**延迟**（每层多 ~20~40 µs）。\n");
    }

    // ── ④ host → GPU1 灌专家（一次性 ~30 GB 的动作能不能扛）──
    if (gpu1_gib <= 0) goto skip4;
    printf("\n== ④ host → GPU1 灌 %.1f GiB（模拟把专家装进 GPU1）==\n", gpu1_gib);
    {
        const size_t CH = 256u << 20;              // 256 MB/块
        void* hp = nullptr;
        cudaSetDevice(0); cudaHostAlloc(&hp, CH, cudaHostAllocDefault);
        memset(hp, 7, CH);
        cudaSetDevice(1);
        void* d = nullptr; cudaMalloc(&d, CH);
        Tally t4;
        const long nch = (long)(gpu1_gib * 1073741824.0 / CH);
        auto a = std::chrono::steady_clock::now();
        for (long i = 0; i < nch; ++i) {
            cudaError_t e = cudaMemcpyAsync(d, hp, CH, cudaMemcpyHostToDevice, 0);
            t4.note(e, i, "H2D");
            cudaError_t s = cudaDeviceSynchronize();
            t4.note(s, i, "sync");
            if (i % 8 == 0) { printf("    %4ld/%ld (%.1f GiB)\n", i, nch, (i + 1) * CH / 1073741824.0); fflush(stdout); }
            if (t4.bad) break;
        }
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        printf("  用 %.1f s ⇒ %.2f GB/s\n", dt, nch * (double)CH / dt / 1e9);
        t4.report("④ host→GPU1");
        cudaFree(d); cudaFreeHost(hp);
    }

skip4:
    // ── ⑤ 压力上限：64 MB 乒乓，尽量快（找断点）—— 只在 P2P 可用时做 ──
    if (peer_ok) {
    printf("\n== ⑤ 压力上限：64 MB 乒乓 20 s ==\n");
    {
        Tally t5; long i = 0;
        auto a = std::chrono::steady_clock::now();
        cudaSetDevice(0);
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() < 20.0) {
            cudaError_t e1 = cudaMemcpyPeerAsync(big1, 1, big0, 0, BIG, 0);
            cudaError_t e2 = cudaMemcpyPeerAsync(big0, 0, big1, 1, BIG, 0);
            t5.note(e1, i, "64MB 去"); t5.note(e2, i, "64MB 回");
            i++;
            if (i % 64 == 0) { t5.note(cudaDeviceSynchronize(), i, "sync"); }
            if (t5.bad > 10) break;
        }
        cudaDeviceSynchronize();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        printf("  %ld 对 64 MB 往返 / %.1f s ⇒ 双向 %.2f GB/s\n", i, dt, i * 2.0 * BIG / dt / 1e9);
        t5.report("⑤ 压力上限");
    }
    } else {
        printf("\n== ⑤ 压力上限：**跳过**（P2P 不可用）==\n");
    }

    printf("\n== 结论提示 ==\n");
    printf("  · ① 若 CanAccessPeer=0 ⇒ 只能走 host 中转（GPU0→host→GPU1，约 1/2 带宽但稳定）。\n");
    printf("  · ③ 全部 ok 且带宽 ≥ 2 GB/s ⇒ 设计口径（80 MB/s）有 25× 余量，跨卡专家常驻可行。\n");
    printf("  · ⑤ 若不稳（掉卡/Xid）而 ③ 稳 ⇒ 别做重灌，只走③的轻流量。\n");
    printf("  · 跑完请在**宿主**上看：dmesg | tail -50 （找 Xid / CmpltTO / DMAR）。\n");
    return 0;
}
