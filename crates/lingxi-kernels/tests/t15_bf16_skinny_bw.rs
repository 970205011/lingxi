//! **bf16 瘦长 GEMM 与草稿 lm_head 的取数结构微基准**（2026-10-24，）。
//!
//! 前三次只换"取数结构"（算式一字未改、逐位相同）各白捡 1.27~2.0×（~）。
//! 本轮先给剩下两块量分母：
//!
//! **目标 1**：bf16 瘦长 GEMM（线上 m=6）
//! * `n=10240 k=320`（GR `mix_up`）与 `n=2560 k=640`（shared `dn`）：走 `gemm_skinny_bf16`
//! = `skinny_body<8>`，**一 warp 一行、每 lane 每轮只读 2 B 权重**；
//! * `n=1280 k=2560`（shared `gu`）：走 `gemm_skinny_row_bf16`，**一 block 一行、16 B/lane**。
//! 探针（`cuda/bf16_skinny_exp.cu`，grid/block/地址映射与生产核逐字节相同）：
//! m8 体 0 = 纯读 2 B/lane、1 = 纯读 16 B/lane、2 = 2 B + cvt + m×FFMA（x 常量）、
//! 3 = 生产内层复刻（自检）、4/5/6 = 一 warp 连算 R=2/4/8 行、7 = R=4+PF、8 = R=4+x smem；
//! row 体 9 = 纯读 16 B、10 = +cvt+FFMA（x 常量）、11 = 生产复刻（自检）、12/13 = 一 block 2 行。
//!
//! **目标 2**：草稿 `lm_head` = `gemv_q4g64_dense_epi`（`n=248320 k=2560`、m=1、Q4G64）。
//! 探针 0 = 纯读 4 B/lane + 每 group 一个 scale（生产取数形状）、1 = 纯读 16 B/lane（上限）、
//! 2 = 生产内层复刻（自检）。
//!
//! 口径：GB/s 按**权重 payload 字节**折算（A 族 n·k·2；B 族 n·k_pad/2 + n·(k_pad/64)·2），
//! 与 / 的表同口径。**`--test-threads=1` 是硬要求**（并行只剩 ~1/3 带宽）。
//!
//! `bash
//! export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib   # cublas/nvrtc 不在默认路径时按需追加
//! cargo test --release -p lingxi-kernels --features gpu --test t15_bf16_skinny_bw \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

struct Q4 {
    packed: CudaSlice<u8>,
    k_pad: i32,
    scale_off: u64,
}

fn q4w(k: &Kernels, rows: usize, kdim: usize) -> Q4 {
    let k_pad = ((kdim + 127) / 128 * 128) as i32;
    let row_bytes = k_pad as usize / 2;
    let base = rows * row_bytes;
    let scale_off = ((base + 255) / 256 * 256) as u64;
    let groups = k_pad as usize / 64;
    let total = scale_off as usize + rows * groups * 2;
    let mut host = vec![0u8; total];
    let mut s: u32 = 0x1234_5678;
    for v in host.iter_mut().take(base) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *v = (s & 0xFF) as u8;
    }
    for i in 0..rows * groups {
        let h = half::f16::from_f32(0.002 + 0.0005 * ((i % 45) as f32));
        let off = scale_off as usize + 2 * i;
        host[off..off + 2].copy_from_slice(&h.to_le_bytes());
    }
    Q4 { packed: k.stream.clone_htod(&host).unwrap(), k_pad, scale_off }
}

fn rndbf16(n: usize, seed: u32) -> Vec<bf16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_f32(((s >> 8) as f32 / 16777216.0 - 0.5) * 0.1)
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

/// **背靠背**：同一进程内 3 轮交替，取**最好**（剔除时钟爬坡/抢占的离群）。
fn best_of3(k: &Kernels, iters: usize, mut f: impl FnMut()) -> f64 {
    let mut v = [0f64; 3];
    for s in v.iter_mut() {
        *s = time_it(k, iters, &mut f);
    }
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// 启动预热：把时钟拉起来（否则进程里**第一发**测量会慢 1.5× —— 本轮实测
/// `gemm_skinny_bf16` 首发 42 µs、同核复刻 27.6 µs 的假差距就是这么来的）。
fn warmup(k: &Kernels) {
    let w = k.stream.clone_htod(&rndbf16(8192 * 2048, 99)).unwrap();
    let x = k.stream.clone_htod(&rndbf16(6 * 2048, 98)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(6 * 8192).unwrap();
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    for _ in 0..40 {
        k.bf16_skinny_probe(1, wp, xp, yp, 8192, 2048, 6).unwrap();
    }
    k.stream.synchronize().unwrap();
}

fn report(tag: &str, bytes: f64, dt: f64) {
    eprintln!(
        "[bf16-bw] {tag:<44}｜{:8.2} µs｜{:7.1} GB/s｜{:6.2} MB",
        dt * 1e6,
        bytes / dt / 1e9,
        bytes / 1e6
    );
}

/// 生产核 ---------------------------------------------------------------

fn bench_prod_m8(
    k: &Kernels, tag: &str, w: &CudaSlice<bf16>, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
    n: usize, kdim: usize, m: usize, bytes: f64, iters: usize,
) {
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let dt = best_of3(k, iters, || {
        k.gemm_skinny_bf16_ptr(wp, xp, yp, n, kdim, m, 0, 0.0, 0.0).unwrap();
    });
    report(tag, bytes, dt);
}

fn bench_prod_row(
    k: &Kernels, tag: &str, w: &CudaSlice<bf16>, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
    n: usize, kdim: usize, m: usize, bytes: f64, iters: usize,
) {
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let dt = best_of3(k, iters, || {
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, n, kdim, m, 0, 0.0, 0.0).unwrap();
    });
    report(tag, bytes, dt);
}

fn bench_probe(
    k: &Kernels, tag: &str, which: usize, w: &CudaSlice<bf16>, x: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>, n: usize, kdim: usize, m: usize, bytes: f64, iters: usize,
) {
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let dt = best_of3(k, iters, || {
        k.bf16_skinny_probe(which, wp, xp, yp, n, kdim, m).unwrap();
    });
    report(tag, bytes, dt);
}

fn bits_eq(a: &[bf16], b: &[bf16]) -> usize {
    a.iter().zip(b.iter()).filter(|(u, v)| u.to_bits() != v.to_bits()).count()
}

/// 逐位自检：生产复刻（3 / 11）与变体（4..8 / 12,13）都对生产核。
fn check_bits(
    k: &Kernels, w: &CudaSlice<bf16>, n: usize, kdim: usize, m: usize,
) {
    let x = k.stream.clone_htod(&rndbf16(m * kdim, 7)).unwrap();
    let mut yref = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    let row_prod = kdim >= 1024;
    {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = yref.device_ptr_mut(&k.stream);
        if row_prod {
            k.gemm_skinny_row_bf16_ptr(wp, xp, yp, n, kdim, m, 0, 0.0, 0.0).unwrap();
        } else {
            k.gemm_skinny_bf16_ptr(wp, xp, yp, n, kdim, m, 0, 0.0, 0.0).unwrap();
        }
    }
    k.stream.synchronize().unwrap();
    let href: Vec<bf16> = k.stream.clone_dtoh(&yref).unwrap();
    let which: &[(usize, &str)] = if row_prod {
        &[(11, "row_prod"), (12, "row_r2"), (13, "row_r2pf"), (19, "row_r4")]
    } else {
        &[
            (3, "m8_prod"),
            (4, "m8_r2"),
            (5, "m8_r4"),
            (6, "m8_r8"),
            (7, "m8_r4pf"),
            (8, "m8_r4x"),
            (15, "m8_r1pf"),
            (16, "m8_r1x"),
            (17, "m8_r2x"),
        ]
    };
    for &(wh, nm) in which {
        let mut y = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
        {
            let (wp, _a) = w.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            let (yp, _c) = y.device_ptr_mut(&k.stream);
            k.bf16_skinny_probe(wh, wp, xp, yp, n, kdim, m).unwrap();
        }
        k.stream.synchronize().unwrap();
        let h: Vec<bf16> = k.stream.clone_dtoh(&y).unwrap();
        let nd = bits_eq(&href, &h);
        eprintln!("[bf16-bw] 自检 {nm} vs 生产核（n{n} k{kdim} m{m}）：{nd} / {} 元素不同", h.len());
        assert_eq!(nd, 0, "{nm} 与生产核不逐位相同");
    }
}

/// 目标 1：一条形状的现状 / 分母 / 候选。
fn shape_target1(k: &Kernels, n: usize, kdim: usize, m: usize) {
    let w = k.stream.clone_htod(&rndbf16(n * kdim, 3)).unwrap();
    let x = k.stream.clone_htod(&rndbf16(m * kdim, 5)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    let bytes = (n * kdim * 2) as f64;
    let row_prod = kdim >= 1024;
    eprintln!(
        "── 目标1 n{n} k{kdim} m{m}（payload {:.2} MB，线上走 {}）──",
        bytes / 1e6,
        if row_prod { "gemm_skinny_row_bf16" } else { "gemm_skinny_bf16(m8)" }
    );
    let iters = if n * kdim > 8_000_000 { 60 } else { 200 };
    if row_prod {
        bench_prod_row(k, "现状 gemm_skinny_row_bf16", &w, &x, &mut y, n, kdim, m, bytes, iters);
    } else {
        bench_prod_m8(k, "现状 gemm_skinny_bf16(m8)", &w, &x, &mut y, n, kdim, m, bytes, iters);
    }
    if row_prod {
        for &(wh, nm) in
            &[(9usize, "①row 纯读 16 B/lane"), (18, "①b只读 x（staging 的原料）"),
              (10, "②row +cvt+m·8FFMA（x 常量）"),
              (11, "③row 生产复刻（自检）"), (12, "④row 一 block 2 行"), (13, "⑤row 2 行+PF2"),
              (19, "⑥row 一 block 4 行")][..]
        {
            bench_probe(k, nm, wh, &w, &x, &mut y, n, kdim, m, bytes, iters);
        }
        // 对照组：同形状若走 m8 体会怎样
        bench_prod_m8(k, "·参照 gemm_skinny_bf16(m8)", &w, &x, &mut y, n, kdim, m, bytes, iters);
    } else {
        for &(wh, nm) in &[
            (0usize, "①m8 纯读 2 B/lane（生产形状）"),
            (1, "②m8 纯读 16 B/lane"),
            (14, "③m8 只读 x（2 B/lane × m）"),
            (2, "④m8 +cvt+m·FFMA（x 常量）"),
            (3, "⑤m8 生产复刻（自检）"),
            (20, "⑥m8 生产体逐字复刻（含 r0）"),
            (21, "⑦m8 m=6 编译期特化"),
            (15, "⑥b m8 R1 + 权重 PF=4"),
            (16, "⑦b m8 R1 + x smem f32"),
            (17, "⑧m8 R2 + x smem f32"),
            (4, "⑨m8 一 warp 2 行"),
            (5, "⑩m8 一 warp 4 行"),
            (6, "⑪m8 一 warp 8 行"),
            (7, "⑫m8 4 行 + PF2"),
            (8, "⑬m8 4 行 + x smem"),
        ][..]
        {
            bench_probe(k, nm, wh, &w, &x, &mut y, n, kdim, m, bytes, iters);
        }
    }
    check_bits(k, &w, n, kdim, m);
}

/// 目标 2：草稿 lm_head。
fn target2_head(k: &Kernels, n: usize, kdim: usize) {
    let w = q4w(k, n, kdim);
    let x = k.stream.clone_htod(&rndbf16(kdim, 11)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(n).unwrap();
    let row_bytes = w.k_pad as usize / 2;
    let groups = w.k_pad as usize / 64;
    let payload = (n * row_bytes) as f64;
    let total = payload + (n * groups * 2) as f64;
    eprintln!(
        "── 目标2 lm_head n{n} k{kdim}（payload {:.1} MB + scale {:.1} MB = {:.1} MB）──",
        payload / 1e6,
        (total - payload) / 1e6,
        total / 1e6
    );
    let (wp, _a) = w.packed.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let dt = best_of3(k, 40, || {
        k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, n, kdim, w.k_pad, 0, w.scale_off, 0, 0.0, 0.0)
            .unwrap();
    });
    report("现状 gemv_q4g64_dense_epi", total, dt);
    eprintln!(
        "[bf16-bw]   ↑ 若按 payload-only 折算 = {:.1} GB/s",
        payload / dt / 1e9
    );
    for &(wh, nm) in
        &[(0usize, "①纯读 4 B/lane + scale（生产形状）"), (1, "②纯读 16 B/lane"),
          (2, "③生产复刻（自检）"), (3, "④生产复刻 + 权重 PF=2"), (4, "⑤一 warp 连算 2 行")][..]
    {
        let dt = best_of3(k, 40, || {
            k.q4_head_probe(wh, wp, xp, yp, n, kdim, w.k_pad, 0, w.scale_off).unwrap();
        });
        report(nm, total, dt);
    }
    // 逐位自检：生产复刻（2）vs 生产核
    let mut yref = k.stream.alloc_zeros::<bf16>(n).unwrap();
    {
        let (yp, _c) = yref.device_ptr_mut(&k.stream);
        k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, n, kdim, w.k_pad, 0, w.scale_off, 0, 0.0, 0.0)
            .unwrap();
    }
    let mut ychk = k.stream.alloc_zeros::<bf16>(n).unwrap();
    {
        let (yp, _c) = ychk.device_ptr_mut(&k.stream);
        k.q4_head_probe(2, wp, xp, yp, n, kdim, w.k_pad, 0, w.scale_off).unwrap();
    }
    k.stream.synchronize().unwrap();
    let a: Vec<bf16> = k.stream.clone_dtoh(&yref).unwrap();
    let b: Vec<bf16> = k.stream.clone_dtoh(&ychk).unwrap();
    let nd = bits_eq(&a, &b);
    eprintln!("[bf16-bw] 自检 head 生产复刻 vs 生产核：{nd} / {} 元素不同", a.len());
    assert_eq!(nd, 0, "head 探针的生产复刻与生产核不逐位相同");
    for &(wh, nm) in &[(3usize, "head_pf"), (4, "head_r2")][..] {
        let mut y2 = k.stream.alloc_zeros::<bf16>(n).unwrap();
        {
            let (yp, _c) = y2.device_ptr_mut(&k.stream);
            k.q4_head_probe(wh, wp, xp, yp, n, kdim, w.k_pad, 0, w.scale_off).unwrap();
        }
        k.stream.synchronize().unwrap();
        let h: Vec<bf16> = k.stream.clone_dtoh(&y2).unwrap();
        let nd = bits_eq(&a, &h);
        eprintln!("[bf16-bw] 自检 {nm} vs 生产核：{nd} / {} 元素不同", h.len());
        assert_eq!(nd, 0, "{nm} 与生产核不逐位相同");
    }
}

/// **背靠背 A/B：生产核 vs 三个"同算式"变体**（同一条命令内 6 轮交替，每轮独立 30 迭代）。
/// 动机：第一版测量里 `gemm_skinny_bf16(m8)` 42 µs、本文件的简化复刻 27.6 µs —— 两边都是
/// 48 寄存器、算式逐字相同，必须先证伪"测量假象"再谈收益。
fn ab_prod_vs_copy(k: &Kernels, n: usize, kdim: usize, m: usize) {
    let w = k.stream.clone_htod(&rndbf16(n * kdim, 3)).unwrap();
    let x = k.stream.clone_htod(&rndbf16(m * kdim, 5)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    let bytes = (n * kdim * 2) as f64;
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let mut acc = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    eprintln!("[bf16-ab] n{n} k{kdim} m{m}：6 轮交替（单位 µs）");
    for _ in 0..6 {
        acc[0].push(time_it(k, 30, || {
            k.gemm_skinny_bf16_ptr(wp, xp, yp, n, kdim, m, 0, 0.0, 0.0).unwrap();
        }));
        acc[1].push(time_it(k, 30, || {
            k.bf16_skinny_probe(3, wp, xp, yp, n, kdim, m).unwrap();
        }));
        acc[2].push(time_it(k, 30, || {
            k.bf16_skinny_probe(20, wp, xp, yp, n, kdim, m).unwrap();
        }));
        acc[3].push(time_it(k, 30, || {
            k.bf16_skinny_probe(21, wp, xp, yp, n, kdim, m).unwrap();
        }));
        acc[4].push(time_it(k, 30, || {
            k.bf16_skinny_probe(22, wp, xp, yp, n, kdim, m).unwrap();
        }));
        eprintln!(
            "[bf16-ab]   生产 {:.2}｜简化 {:.2}｜含r0 {:.2}｜m=6 {:.2}｜**r0+x基址外提** {:.2}",
            acc[0].last().unwrap() * 1e6,
            acc[1].last().unwrap() * 1e6,
            acc[2].last().unwrap() * 1e6,
            acc[3].last().unwrap() * 1e6,
            acc[4].last().unwrap() * 1e6
        );
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let names = ["生产", "简化复刻", "逐字复刻(r0)", "m=6 特化", "r0+x基址外提"];
    let meds: Vec<f64> = acc.iter_mut().map(|v| med(v)).collect();
    for (i, nm) in names.iter().enumerate() {
        eprintln!(
            "[bf16-ab] {nm:<14} 中位 {:6.2} µs｜{:6.1} GB/s",
            meds[i] * 1e6,
            bytes / meds[i] / 1e9
        );
    }
}

#[test]
fn t15_ab_m8_copy() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    warmup(&k);
    ab_prod_vs_copy(&k, 10240, 320, 6);
    ab_prod_vs_copy(&k, 2560, 640, 6);
}

/// 生产入口（默认已切到 `_flat`）vs 旧 `skinny_body<8>` 逐字复刻（探针 20）：
/// m=1..8 × 两个 n（一个非 8 的整数倍，覆盖尾巴 block）全逐位相同才算过。
fn check_flat_vs_old(k: &Kernels) {
    let kdim = 320usize;
    for &n in &[1000usize, 1024] {
        let w = k.stream.clone_htod(&rndbf16(n * kdim, 21)).unwrap();
        for m in 1..=8usize {
            let x = k.stream.clone_htod(&rndbf16(m * kdim, 22 + m as u32)).unwrap();
            let mut y_new = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
            let mut y_old = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
            {
                let (wp, _a) = w.device_ptr(&k.stream);
                let (xp, _b) = x.device_ptr(&k.stream);
                let (yp, _c) = y_new.device_ptr_mut(&k.stream);
                k.gemm_skinny_bf16_ptr(wp, xp, yp, n, kdim, m, 0, 0.0, 0.0).unwrap();
            }
            {
                let (wp, _a) = w.device_ptr(&k.stream);
                let (xp, _b) = x.device_ptr(&k.stream);
                let (yp, _c) = y_old.device_ptr_mut(&k.stream);
                k.bf16_skinny_probe(20, wp, xp, yp, n, kdim, m).unwrap();
            }
            k.stream.synchronize().unwrap();
            let a: Vec<bf16> = k.stream.clone_dtoh(&y_new).unwrap();
            let b: Vec<bf16> = k.stream.clone_dtoh(&y_old).unwrap();
            let nd = bits_eq(&a, &b);
            eprintln!("[bf16-bw] 自检 flat vs 旧体（n{n} k{kdim} m{m}）：{nd} / {} 元素不同", a.len());
            assert_eq!(nd, 0, "flat 与旧体不逐位相同（n{n} m{m}）");
        }
    }
}

#[test]
fn t15_flat_bits() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    check_flat_vs_old(&k);
}

#[test]
fn t15_bf16_skinny_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    warmup(&k);
    // 目标 1：三条线上形状（m=6 是 verify 批；kd=5）
    shape_target1(&k, 10240, 320, 6); // GR mix_up
    shape_target1(&k, 1280, 2560, 6); // shared gu
    shape_target1(&k, 2560, 640, 6); // shared dn
    // 目标 2：草稿 lm_head
    target2_head(&k, 248320, 2560);
}
