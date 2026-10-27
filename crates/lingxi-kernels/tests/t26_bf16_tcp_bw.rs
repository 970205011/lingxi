//! ** 投影张量核化微基准**（2026-10-27）：稠密 bf16 投影 `bf16_tcp`（Volta
//! mma.m8n8k4，有损档）vs 生产 `mlarge`（SIMT FMA）——同形状背靠背对时 + 数值口径对拍。
//!
//! 形状取 attn 投影 t=512 生产形状：qkvz [n=2560 k=2560]、zproj [n=2048 k=2560]、
//! out [n=2560 k=2048]（GDN 档近似）、QSA proj [n=6144 k=2560]，全部 m=512。
//!
//! 数值口径：fp16 乘 + fp32 累加 vs f32 FMA——**非逐位**是预期；判读 =
//! 无 NaN/Inf、最大相对误差 ≲1e-2（bf16 输出量化 ~2^-8 相对步进同量级）。
//!
//! `bash
//! cargo test --release -p lingxi-kernels --features gpu --test t26_bf16_tcp_bw \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
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

fn time_it(k: &Kernels, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    k.stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    k.stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() / iters as f64
}

fn best_of3(k: &Kernels, iters: usize, mut f: impl FnMut()) -> f64 {
    let mut v = [0f64; 3];
    for s in v.iter_mut() {
        *s = time_it(k, iters, &mut f);
    }
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn run_shape(k: &Kernels, n: usize, kdim: usize, m: usize, iters: usize) {
    let w = k.stream.clone_htod(&rndbf16(n * kdim, 7)).unwrap();
    let x = k.stream.clone_htod(&rndbf16(m * kdim, 9)).unwrap();
    let mut y0 = k.stream.clone_htod(&rndbf16(m * n, 11)).unwrap();
    let mut y1 = k.stream.clone_htod(&rndbf16(m * n, 11)).unwrap();

    // ---- 数值对拍：mlarge（f32 FMA 档）vs bf16_tcp（fp16 MMA 档）----
    {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (p0, _c) = y0.device_ptr_mut(&k.stream);
        k.gemm_bf16_mlarge_ptr(wp, xp, p0, n, kdim, m, 0, 0.0, 0.0).unwrap();
        let (p1, _c) = y1.device_ptr_mut(&k.stream);
        k.gemm_bf16_tcp_ptr(wp, xp, p1, n, kdim, m, 0, 0.0, 0.0).unwrap();
        k.stream.synchronize().unwrap();
    }
    let h0: Vec<bf16> = k.stream.clone_dtoh(&y0).unwrap();
    let h1: Vec<bf16> = k.stream.clone_dtoh(&y1).unwrap();
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut nan = 0usize;
    let mut ndiff = 0usize;
    for (a, b) in h0.iter().zip(h1.iter()) {
        let (fa, fb) = (a.to_f32() as f64, b.to_f32() as f64);
        if !fa.is_finite() || !fb.is_finite() {
            nan += 1;
            continue;
        }
        let d = (fa - fb).abs();
        if d > 0.0 {
            ndiff += 1;
        }
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / fa.abs().max(1e-6));
    }
    println!(
        "[n{n} k{kdim} m{m}] 数值：不一致位 {ndiff}/{}、NaN/Inf {nan}、max_abs {max_abs:.3e}、max_rel {max_rel:.3e}",
        h0.len()
    );
    assert_eq!(nan, 0, "出现 NaN/Inf —— MMA 片段映射有 bug");

    // ---- 对时（best_of3，背靠背同一进程）----
    let t_mlarge = best_of3(k, iters, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (p, _c) = y0.device_ptr_mut(&k.stream);
        k.gemm_bf16_mlarge_ptr(wp, xp, p, n, kdim, m, 0, 0.0, 0.0).unwrap();
    });
    let t_tcp = best_of3(k, iters, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (p, _c) = y1.device_ptr_mut(&k.stream);
        k.gemm_bf16_tcp_ptr(wp, xp, p, n, kdim, m, 0, 0.0, 0.0).unwrap();
    });
    let gbs = |t: f64| (n * kdim * 2) as f64 / t / 1e9;
    println!(
        "[n{n} k{kdim} m{m}] mlarge {:8.1} µs（{:5.1} GB/s） vs bf16_tcp {:8.1} µs（{:5.1} GB/s） ⇒ {:.2}×",
        t_mlarge * 1e6,
        gbs(t_mlarge),
        t_tcp * 1e6,
        gbs(t_tcp),
        t_mlarge / t_tcp
    );
}

#[test]
fn bf16_tcp_vs_mlarge_shapes() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    // 预热：拉时钟 + JIT/缓存就绪
    {
        let w = k.stream.clone_htod(&rndbf16(2560 * 2560, 3)).unwrap();
        let x = k.stream.clone_htod(&rndbf16(512 * 2560, 4)).unwrap();
        let mut y = k.stream.clone_htod(&rndbf16(512 * 2560, 5)).unwrap();
        for _ in 0..5 {
            let (wp, _a) = w.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            let (p, _c) = y.device_ptr_mut(&k.stream);
            k.gemm_bf16_tcp_ptr(wp, xp, p, 2560, 2560, 512, 0, 0.0, 0.0).unwrap();
        }
        k.stream.synchronize().unwrap();
    }
    for (n, kdim) in [
        (2560usize, 2560usize),
        (2048, 2560),
        (2560, 2048),
        (6144, 2560),
        (10240, 2560), // GDN qkvz 真形状（conv_dim=2*2048+6144）
        (6144, 2560),  // zproj（value_dim=48*128）
        (2560, 6144),  // GDN out（k=value_dim）
    ] {
        run_shape(&k, n, kdim, 512, 20);
    }
}
