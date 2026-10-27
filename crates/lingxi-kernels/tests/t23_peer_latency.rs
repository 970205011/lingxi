//! **跨卡小消息的往返延迟**（docs 的前置判决实验）。
//!
//! 要决定的事：把 MoE 的专家按 id 二分到两张卡（稠密仍层切分）⇒ 每层要交换
//! ①`x`（激活，t×hidden×2 B ≈ 30 KB）②两半的部分和（同量级），**每轮 96 次**。
//! 若一次交换是 50 µs，96 次就是 4.8 ms —— 吃掉大半收益；若是 5 µs，只有 0.5 ms。
//!
//! 三条路都量：
//! A. **P2P 直发**（`cuMemcpyPeerAsync` + 目标流事件），包大小 30 KB；
//! B. **P2P 直发但不等**（连发 96 次再等一次）—— 看的是"流水化之后的单次成本"；
//! C. **现有的 host staging 路**（`cross_copy_staged`：D2H→memcpy→H2D，两次流同步）。
//!
//! 判据（写进）：只要 A ≤ 15 µs，96 次/轮 ≤ 1.5 ms，这条路就值得开工；
//! 若 A ≥ 40 µs，收益会被交换吃掉一半以上，得先想别的（合并成每 N 层一次）。
//!
//! 跑法：
//! `bash
//! export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib   # cublas/nvrtc 不在默认路径时按需追加
//! cargo test --release -p lingxi-kernels --features gpu --test t23_peer_latency \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

/// 一次交换的载荷：`t=6` 行 × hidden 2560 × bf16 = 30 KB（与 verify 批一致）。
const ROWS: usize = 6;
const HIDDEN: usize = 2560;
const N: usize = ROWS * HIDDEN;
/// 一轮的交换次数（48 层 × 2 次：发 x + 回部分和）。
const PER_ROUND: usize = 96;

/// 裸 `cuMemcpyPeerAsync`：`dst(dev1) ← src(dev0)`，排在 `stream` 上。
///
/// 用 `cudarc::driver::result` 的原始入口（safe 层没有 peer 封装）。
unsafe fn peer_copy(
    k_from: &Kernels,
    src: &CudaSlice<bf16>,
    k_to: &Kernels,
    dst: &mut CudaSlice<bf16>,
) -> anyhow::Result<()> {
    let (sp, _g0) = src.device_ptr(&k_from.stream);
    let (dp, _g1) = dst.device_ptr_mut(&k_to.stream);
    // 签名：`(dst_ctx, dst, src_ctx, src, num_bytes, stream)`
    cudarc::driver::result::memcpy_peer_async(
        k_to.ctx.cu_ctx(),
        dp,
        k_from.ctx.cu_ctx(),
        sp,
        N * 2,
        k_to.stream.cu_stream(),
    )?;
    Ok(())
}

fn time_it(iters: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..3 {
        f(i);
    }
    let t0 = Instant::now();
    for i in 0..iters {
        f(i);
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

#[test]
fn t23_peer_latency() {
    let k0 = Kernels::load(0).expect("GPU0");
    let k1 = Kernels::load(1).expect("GPU1");
    let mut a0: CudaSlice<bf16> = k0.stream.alloc_zeros::<bf16>(N).unwrap();
    let mut b1: CudaSlice<bf16> = k1.stream.alloc_zeros::<bf16>(N).unwrap();
    let mut a0b: CudaSlice<bf16> = k0.stream.alloc_zeros::<bf16>(N).unwrap();
    let mut x1: CudaSlice<bf16> = k1.stream.alloc_zeros::<bf16>(N).unwrap();
    // 先写点东西（避免全 0 页被特殊处理）
    k0.stream.memset_zeros(&mut a0).unwrap();
    k0.stream.synchronize().unwrap();

    // ── A：P2P 直发 + 等目标流（一次往返的真实成本）───────────────────────
    let a_us = match unsafe { peer_copy(&k0, &a0, &k1, &mut b1) } {
        Ok(()) => {
            k1.stream.synchronize().unwrap();
            time_it(200, |i| {
                if i % 2 == 0 {
                    unsafe { peer_copy(&k0, &a0, &k1, &mut b1) }.unwrap();
                    k1.stream.synchronize().unwrap();
                } else {
                    unsafe { peer_copy(&k1, &x1, &k0, &mut a0b) }.unwrap();
                    k0.stream.synchronize().unwrap();
                }
            })
        }
        Err(e) => {
            println!("[t23] ❌ P2P 不可用：{e:#}（这条路直接否掉）");
            f64::INFINITY
        }
    };
    if a_us.is_finite() {
        println!(
            "[t23] A. P2P 直发 + 同步（30 KB，单向往返）：{a_us:.1} µs/次 \
             ⇒ 一轮 {PER_ROUND} 次 = {:.2} ms",
            a_us * PER_ROUND as f64 / 1e3
        );
    }

    // ── B：连发 96 次再等一次（流水化后的摊薄成本）────────────────────────
    if a_us.is_finite() {
        let t0 = Instant::now();
        for round in 0..20 {
            for j in 0..PER_ROUND {
                if j % 2 == 0 {
                    unsafe { peer_copy(&k0, &a0, &k1, &mut b1) }.unwrap();
                } else {
                    unsafe { peer_copy(&k1, &x1, &k0, &mut a0b) }.unwrap();
                }
            }
            if round % 2 == 0 {
                k1.stream.synchronize().unwrap();
            } else {
                k0.stream.synchronize().unwrap();
            }
        }
        k0.stream.synchronize().unwrap();
        k1.stream.synchronize().unwrap();
        let per_round = t0.elapsed().as_secs_f64() * 1e6 / 20.0;
        println!(
            "[t23] B. 连发 {PER_ROUND} 次（双向交替，只等一次）：{per_round:.0} µs/轮 \
             = {:.1} µs/次（含主机排队）",
            per_round / PER_ROUND as f64
        );
    }

    // ── C：现有 host staging 路的等价物（D2H → 主机 memcpy → H2D，两次流同步）──────
    // `cross_copy_staged` 在 lingxi-model 里（本 crate 不能依赖它），这里用同样的三步裸写，
    // 差别只是少了 pinned staging（非锁页会再慢一点 ⇒ 下面的数字是**下界**）。
    let mut host = vec![bf16::ZERO; N];
    let c_us = time_it(100, |_| {
        k0.stream.memcpy_dtoh(&a0, &mut host).unwrap();
        k0.stream.synchronize().unwrap();
        k1.stream.memcpy_htod(&host, &mut b1).unwrap();
        k1.stream.synchronize().unwrap();
    });
    println!(
        "[t23] C. host staging（D2H→memcpy→H2D，两次同步）：{c_us:.1} µs/次 \
         ⇒ 一轮 {PER_ROUND} 次 = {:.2} ms",
        c_us * PER_ROUND as f64 / 1e3
    );

    if a_us.is_finite() {
        println!(
            "[t23] ⇒ P2P 比 staging 快 **{:.1}×**；一轮交换成本 {:.2} ms（P2P）vs {:.2} ms（staging）",
            c_us / a_us,
            a_us * PER_ROUND as f64 / 1e3,
            c_us * PER_ROUND as f64 / 1e3
        );
    }
}
