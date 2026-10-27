//! ** 残留风险补拍**（2026-10-27）：`bf16_tcp` 的 **m=71 尾块数值对拍**。
//!
//! 背景： 的 bf16_tcp 核对 m%64≠0 尾块（t=71 ⇒ 第 2 个 64 行 chunk 只剩 7 行，
//! 靠零填充 + 写回守卫处理）但未单独对拍。本测试把 t26 的数值对拍原样搬到 m=71，
//! 生产形状不变：qkvz/zproj/out/QSA proj。
//!
//! 判据（同 档位）：0 NaN/Inf；不一致位比例与 max_abs 在 实测量级内
//! （m=512：不一致位 0.04~0.17%、max_abs ≤3.9e-3）。
//!
//! `bash
//! cargo test --release -p lingxi-kernels --features gpu --test t27_bf16_tcp_tail71 \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

fn rndbf16(n: usize, seed: u32) -> Vec<bf16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_f32(((s >> 8) as f32 / 16777216.0 - 0.5) * 0.2)
        })
        .collect()
}

#[test]
fn bf16_tcp_m71_tail_chunk_vs_mlarge() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    const M: usize = 71; // 64+7：第二个行 chunk 只有 7 行有效（尾块路径）
    for (n, kdim) in [
        (2560usize, 2560usize), // attn qkvz
        (2048, 2560),           // attn zproj
        (2560, 2048),           // attn out（GDN 档近似）
        (6144, 2560),           // QSA proj
        (10240, 2560),          // GDN qkvz 真形状
        (2560, 6144),           // GDN out（k=value_dim）
    ] {
        let w = k.stream.clone_htod(&rndbf16(n * kdim, 7)).unwrap();
        let x = k.stream.clone_htod(&rndbf16(M * kdim, 9)).unwrap();
        let mut y0 = k.stream.clone_htod(&rndbf16(M * n, 11)).unwrap();
        let mut y1 = k.stream.clone_htod(&rndbf16(M * n, 11)).unwrap();
        {
            let (wp, _a) = w.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            let (p0, _c) = y0.device_ptr_mut(&k.stream);
            k.gemm_bf16_mlarge_ptr(wp, xp, p0, n, kdim, M, 0, 0.0, 0.0).unwrap();
            let (p1, _c) = y1.device_ptr_mut(&k.stream);
            k.gemm_bf16_tcp_ptr(wp, xp, p1, n, kdim, M, 0, 0.0, 0.0).unwrap();
            k.stream.synchronize().unwrap();
        }
        let h0: Vec<bf16> = k.stream.clone_dtoh(&y0).unwrap();
        let h1: Vec<bf16> = k.stream.clone_dtoh(&y1).unwrap();
        // 单独统计尾块行（row ≥ 64）与正块行，确认守卫没有把垃圾写进有效行、也没漏写。
        let mut max_abs = 0f64;
        let mut nan = 0usize;
        let mut ndiff = 0usize;
        let mut ndiff_tail = 0usize;
        let mut ntail = 0usize;
        for (i, (a, b)) in h0.iter().zip(h1.iter()).enumerate() {
            let (fa, fb) = (a.to_f32() as f64, b.to_f32() as f64);
            if !fa.is_finite() || !fb.is_finite() {
                nan += 1;
                continue;
            }
            let d = (fa - fb).abs();
            max_abs = max_abs.max(d);
            if d > 0.0 {
                ndiff += 1;
                if i / n >= 64 {
                    ndiff_tail += 1;
                }
            }
            if i / n >= 64 {
                ntail += 1;
            }
        }
        let pct = ndiff as f64 / h0.len() as f64 * 100.0;
        println!(
            "[n{n} k{kdim} m{M}] 数值：不一致位 {ndiff}/{}（{pct:.3}%）NaN/Inf {nan} \
             max_abs {max_abs:.3e}｜尾块行 {ntail} 行内不一致 {ndiff_tail}",
            h0.len()
        );
        assert_eq!(nan, 0, "[n{n} k{kdim}] 出现 NaN/Inf —— 尾块守卫有 bug");
        assert!(
            max_abs <= 1e-2,
            "[n{n} k{kdim}] max_abs {max_abs:.3e} 超  同档量级（≤3.9e-3 附近放宽一档）"
        );
    }
}
