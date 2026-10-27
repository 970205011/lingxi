//! 专家段 GEMV 的**带宽微基准**（2026-10-23）。
//!
//! 动机：整机实测专家段每层约 0.53 ms，按 mix64 的实际字节数折算只有 ~220 GB/s，
//! 远低于 V100 的可达带宽。这个用例把真实形状（E=49 组、n=1280、k=2560、组均 m≈1）
//! 单独拉出来跑，量出核本身的带宽上限，再决定要不要改核。
//!
//! 门控与其它 GPU 用例一致：
//! `LD_LIBRARY_PATH=... cargo test --release -p lingxi-kernels --features gpu --test t3_gemv_bw`

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

/// 组均 m 行、E 组，量一次 launch 的等效带宽。
///
/// `same_slot=true` 时所有组都指向 0 号槽（1.6 MB ⇒ 全部命中 6 MB L2）：
/// 用来分离「HBM 带宽」与「解码 ALU/延迟」——若时间几乎不变，说明瓶颈不在带宽。
fn bench(tag: &str, k: &Kernels, e: usize, n: usize, kdim: usize, m: usize, blob: usize, same_slot: bool) {
    let slot_ids: Vec<i32> = if same_slot {
        vec![0; e]
    } else {
        (0..e as i32).collect()
    };
    let mut group_off: Vec<i32> = Vec::with_capacity(e + 1);
    for g in 0..=e {
        group_off.push((g * m) as i32);
    }
    let s = e * m;
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); s * kdim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(s * n).unwrap();
    let slots = unsafe { k.stream.alloc::<u8>(e * blob) }.unwrap();
    // 槽内容未初始化没关系：只量带宽（Kernels 的 alloc 不做 memset，读到的是显存原值）。
    let ids = k.stream.clone_htod(&slot_ids).unwrap();
    let goff = k.stream.clone_htod(&group_off).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);

    let mut warm: Vec<bf16> = k.stream.clone_dtoh(&y).unwrap();
    warm[0] = warm[0]; // 防止被优化掉
    for _ in 0..3 {
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &slots, &ids_v, &x, &goff_v, &mut y, e, n, kdim, kdim as i32, kdim,
            blob as u64, 0, 0, 1, 0,
        )
        .unwrap();
    }
    k.stream.synchronize().unwrap();
    let iters = 30;
    let t0 = Instant::now();
    for _ in 0..iters {
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &slots, &ids_v, &x, &goff_v, &mut y, e, n, kdim, kdim as i32, kdim,
            blob as u64, 0, 0, 1, 0,
        )
        .unwrap();
    }
    k.stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    // 每组读 n 行、每行 k_pad/2 字节（Q4G64）
    let bytes = (e * n * (kdim / 2)) as f64;
    eprintln!(
        "[gemv-bw] {tag}: E{e} n{n} k{kdim} m{m}｜{:.3} ms/次｜{:.1} GB/s",
        dt * 1e3,
        bytes / dt / 1e9
    );
}

/// 三条 **x 供给路径**的统一微基准（同核同形状，只改 m 与 x 表示）：
/// `mode 0` = bf16 现转（历史 `bench` 的口径）；`1` = **f32 预转（生产默认，`xf`）**；
/// `2` = **int8 行 scale + dp4a**（`LINGXI_MOE_DP4A` 的核）。
///
/// 目的（2026-10-23 判决）：把「每 token 的反量化/转换/FMA 指令」与「权重流」分账——
/// 若 dp4a 把 m=6 从 ~95 拉到 ≥180 GB/s，说明 m 依赖项就是 ALU 发射率（不是带宽）。
fn bench_mode(tag: &str, k: &Kernels, e: usize, n: usize, kdim: usize, m: usize, blob: usize,
              seg_off: u64, scale_off: u64, mode: u8) {
    let slot_ids: Vec<i32> = (0..e as i32).collect();
    let mut group_off: Vec<i32> = Vec::with_capacity(e + 1);
    for g in 0..=e {
        group_off.push((g * m) as i32);
    }
    let s = e * m;
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); s * kdim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(s * n).unwrap();
    let slots = unsafe { k.stream.alloc::<u8>(e * blob + scale_off as usize
        + if blob == 0 { n * kdim / 64 * 2 } else { 0 }) }.unwrap();
    let ids = k.stream.clone_htod(&slot_ids).unwrap();
    let goff = k.stream.clone_htod(&group_off).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let sz = kdim / 2;
    let xf_b = if mode == 1 {
        Some(k.stream.clone_htod(&vec![0.01f32; s * kdim]).unwrap())
    } else { None };
    let xq_b = if mode == 2 {
        Some(unsafe { k.stream.alloc::<u8>(s * kdim) }.unwrap())
    } else { None };
    let sx_b = if mode == 2 { Some(k.stream.alloc_zeros::<f32>(s).unwrap()) } else { None };
    let xf_p = match &xf_b {
        Some(v) => v.device_ptr(&k.stream).0,
        None => 0,
    };
    let run = |y: &mut cudarc::driver::CudaSlice<bf16>| {
        match mode {
            2 => k.gemv_q4g64_slotsm_rpw_dp4a_view(
                &slots, &ids_v, &x, &goff_v, y, e, n, kdim, kdim as i32, kdim,
                blob as u64, seg_off, scale_off, 1,
                xq_b.as_ref().unwrap(), sx_b.as_ref().unwrap(),
            ).unwrap(),
            _ => k.gemv_q4g64_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x, &goff_v, y, e, n, kdim, kdim as i32, kdim,
                blob as u64, seg_off, scale_off, 1, xf_p,
            ).unwrap(),
        }
    };
    for _ in 0..3 {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let iters = 30;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let bytes = (e * n * sz) as f64;
    let mm = match mode { 0 => "bf16x", 1 => "f32x", _ => "dp4a" };
    eprintln!(
        "[path-bw] {tag}〔{mm}〕: E{e} n{n} k{kdim} m{m}｜{:.3} ms/次｜{:.1} GB/s",
        dt * 1e3,
        bytes / dt / 1e9
    );
}

#[test]
fn t3_gemv_bw_paths() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let blob = 8 * 1024 * 1024;
    for &m in &[1usize, 2, 4, 6, 8] {
        for &mode in &[0u8, 1, 2] {
            bench_mode("gu", &k, 49, 1280, 2560, m, blob, 0, 0, mode);
            bench_mode("dn", &k, 49, 640, 1280, m, blob, 0, 0, mode);
        }
    }
    // 稠密投影同口径：E=1 单块权重（scale 平面跟在 base 平面后）。
    for &m in &[2usize, 4, 6, 8] {
        for &mode in &[0u8, 1, 2] {
            let (n, kd) = (10240usize, 2560usize);
            bench_mode("dense", &k, 1, n, kd, m, 0, 0,
                       (n * kd / 2) as u64, mode);
        }
    }
}

/// 判因变体：`rpw` = 每 warp 行数；`share_x` = 所有 token 共用第 0 行 x（a_stride=0，
/// 把「每 token 的激活读」从 L1 重读变成广播）；`same_slot` = 49 组指向同一槽（权重进 6MB L2）。
fn bench_var(tag: &str, k: &Kernels, e: usize, n: usize, kdim: usize, m: usize, blob: usize,
             mode: u8, rpw: usize, share_x: bool, same_slot: bool) {
    let slot_ids: Vec<i32> = if same_slot {
        vec![0; e]
    } else {
        (0..e as i32).collect()
    };
    let mut group_off: Vec<i32> = Vec::with_capacity(e + 1);
    for g in 0..=e {
        group_off.push((g * m) as i32);
    }
    let a_stride = if share_x { 0 } else { kdim };
    let s = e * m;
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); s * kdim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(s * n).unwrap();
    let slots = unsafe { k.stream.alloc::<u8>(e * blob) }.unwrap();
    let ids = k.stream.clone_htod(&slot_ids).unwrap();
    let goff = k.stream.clone_htod(&group_off).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let xf_b = if mode == 1 { Some(k.stream.clone_htod(&vec![0.01f32; s * kdim]).unwrap())
        } else { None };
    let xq_b = if mode == 2 { Some(unsafe { k.stream.alloc::<u8>(s * kdim) }.unwrap())
        } else { None };
    let sx_b = if mode == 2 { Some(k.stream.alloc_zeros::<f32>(s).unwrap()) } else { None };
    let xf_p = match &xf_b { Some(v) => v.device_ptr(&k.stream).0, None => 0 };
    let run = |y: &mut cudarc::driver::CudaSlice<bf16>| {
        match mode {
            2 => k.gemv_q4g64_slotsm_rpw_dp4a_view(
                &slots, &ids_v, &x, &goff_v, y, e, n, kdim, kdim as i32, a_stride,
                blob as u64, 0, 0, rpw,
                xq_b.as_ref().unwrap(), sx_b.as_ref().unwrap(),
            ).unwrap(),
            _ => k.gemv_q4g64_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x, &goff_v, y, e, n, kdim, kdim as i32, a_stride,
                blob as u64, 0, 0, rpw, xf_p,
            ).unwrap(),
        }
    };
    for _ in 0..3 {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let iters = 30;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let bytes = (e * n * (kdim / 2)) as f64;
    let mm = match mode { 0 => "bf16x", 1 => "f32x", _ => "dp4a" };
    eprintln!(
        "[var-bw] {tag}〔{mm} rpw{rpw}{}{}〕: m{m}｜{:.3} ms/次｜{:.1} GB/s",
        if share_x { " x广播" } else { "" },
        if same_slot { " 同槽" } else { "" },
        dt * 1e3, bytes / dt / 1e9
    );
}

#[test]
fn t3_gemv_bw_var() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let blob = 8 * 1024 * 1024;
    // smem-x A/B：本用例同进程只跑一次 ⇒ 用 `LINGXI_GEMV_SMEM_X=1|2` 分进程跑两遍，
    // 对比 `[var-bw]` 行的 GB/s（默认关 = 改前）。m=1 用来确认不退化。
    for &m in &[1usize, 2, 4, 6, 8] {
        for &mode in &[0u8, 2] {
            bench_var("gu", &k, 49, 1280, 2560, m, blob, mode, 1, false, false);
            bench_var("dn", &k, 49, 640, 1280, m, blob, mode, 1, false, false);
        }
    }
    // 判因锚点（m=6）：x 广播 / 权重同槽 / 两者都去掉 —— 与改造后的值同口径对照。
    for &mode in &[0u8, 2] {
        bench_var("gu", &k, 49, 1280, 2560, 6, blob, mode, 1, true, false);
        bench_var("gu", &k, 49, 1280, 2560, 6, blob, mode, 1, false, true);
        bench_var("gu", &k, 49, 1280, 2560, 6, blob, mode, 2, false, false);
        bench_var("gu", &k, 49, 1280, 2560, 6, blob, mode, 1, true, true);
    }
}

#[test]
fn t3_gemv_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let blob = 8 * 1024 * 1024; // 槽距放大到 8 MiB，保证各组不在同一 L2 行上
    for &m in &[1usize, 2, 4, 8] {
        bench("gu(2*inter)", &k, 49, 1280, 2560, m, blob, false);
        bench("dn", &k, 49, 640, 1280, m, blob, false);
        bench_q3("gu-q3", &k, 49, 1280, 2560, m, blob, 0.75);
        bench_tc("gu-tc", &k, 49, 1280, 2560, m, blob, 2);
        if m == 1 {
            bench("gu(L2常驻)", &k, 49, 1280, 2560, m, blob, true);
            bench("dn(L2常驻)", &k, 49, 640, 1280, m, blob, true);
        }
    }
    // 稠密投影的真实形状：与专家段同核（E=1）vs 现役 skinny GEMM，背靠背量核本身带宽。
    for &m in &[1usize, 5, 6] {
        for &(n, kd) in &[(10240usize, 2560usize), (6144, 2560), (2560, 6144)] {
            bench_dense("skinny", &k, n, kd, m, None);
            bench_dense("dense-view r1", &k, n, kd, m, Some(1));
            bench_dense("dense-tc", &k, n, kd, m, Some(0));
        }
    }
}

/// 稠密 Q4G64 投影：一块连续权重（`n` 行 × `kdim`）+ `m` 行激活。
/// `None` 走现役 `gemm_q4_skinny_bf16_ptr`；`Some(1)` 走专家段 `slotsm`（E=1）；
/// `Some(0)` 走专家段张量核槽版。
fn bench_dense(tag: &str, k: &Kernels, n: usize, kdim: usize, m: usize, rpw: Option<usize>) {
    let row_bytes = kdim / 2;
    let scale_off = n * row_bytes; // 256B 对齐：n*row_bytes 已是 256 的倍数
    let w = unsafe { k.stream.alloc::<u8>(scale_off + n * kdim / 64 * 2) }.unwrap();
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); m * kdim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    let ids = k.stream.clone_htod(&[0i32]).unwrap();
    let goff = k.stream.clone_htod(&[0i32, m as i32]).unwrap();
    let ids_v = ids.slice(0..1);
    let goff_v = goff.slice(0..2);

    let run = |y: &mut cudarc::driver::CudaSlice<bf16>| {
        match rpw {
            None => {
                let (wp, _a) = w.device_ptr(&k.stream);
                let (xp, _b) = x.device_ptr(&k.stream);
                let (yp, _c) = y.device_ptr_mut(&k.stream);
                unsafe {
                    k.gemm_q4_skinny_bf16_ptr(wp, xp, yp, n, kdim, kdim as i32, m, 0, scale_off as u64, 0, 0.0, 0.0)
                }
                .unwrap();
            }
            Some(0) => {
                k.gemv_q4g64_slots_tc_bf16(
                    &w, &ids_v, &goff_v, &x, y, 1, n, kdim, kdim, kdim as i32,
                    0, 0, scale_off as u64, 2,
                )
                .unwrap();
            }
            Some(r) => {
                k.gemv_q4g64_slotsm_rpw_bf16_view(
                    &w, &ids_v, &x, &goff_v, y, 1, n, kdim, kdim as i32, kdim,
                    0, 0, scale_off as u64, r, 0,
                )
                .unwrap();
            }
        }
    };
    for _ in 0..3 {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let bytes = (n * row_bytes) as f64;
    eprintln!(
        "[dense-bw] {tag}: n{n} k{kdim} m{m}｜{:.3} ms/次｜{:.1} GB/s",
        dt * 1e3,
        bytes / dt / 1e9
    );
}

/// Q3G128 SIMT 版（混档 arena 里多数槽是 Q3 ⇒ 这才是真实主力）。
fn bench_q3(tag: &str, k: &Kernels, e: usize, n: usize, kdim: usize, m: usize, blob: usize, _r: f64) {
    let slot_ids: Vec<i32> = (0..e as i32).collect();
    let mut group_off: Vec<i32> = Vec::with_capacity(e + 1);
    for g in 0..=e {
        group_off.push((g * m) as i32);
    }
    let s = e * m;
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); s * kdim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(s * n).unwrap();
    let slots = unsafe { k.stream.alloc::<u8>(e * blob) }.unwrap();
    let ids = k.stream.clone_htod(&slot_ids).unwrap();
    let goff = k.stream.clone_htod(&group_off).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let run = |y: &mut cudarc::driver::CudaSlice<bf16>| {
        k.gemv_q3g128_slotsm_rpw_bf16_view(
            &slots, &ids_v, &x, &goff_v, y, e, n, kdim, kdim as i32, kdim,
            blob as u64, 0, 0, 1, 0,
        )
        .unwrap();
    };
    for _ in 0..3 {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let iters = 30;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let bytes = (e * n * (kdim / 8 * 3)) as f64;
    eprintln!(
        "[gemv-bw] {tag}: E{e} n{n} k{kdim} m{m}｜{:.3} ms/次｜{:.1} GB/s",
        dt * 1e3,
        bytes / dt / 1e9
    );
}

/// Q4G64 张量核版（`LINGXI_MOE_TC` 生产档：gu 用、dn 不用）。
fn bench_tc(tag: &str, k: &Kernels, e: usize, n: usize, kdim: usize, m: usize, blob: usize, tiles: usize) {
    let slot_ids: Vec<i32> = (0..e as i32).collect();
    let mut group_off: Vec<i32> = Vec::with_capacity(e + 1);
    for g in 0..=e {
        group_off.push((g * m) as i32);
    }
    let s = e * m;
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); s * kdim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(s * n).unwrap();
    let slots = unsafe { k.stream.alloc::<u8>(e * blob) }.unwrap();
    let ids = k.stream.clone_htod(&slot_ids).unwrap();
    let goff = k.stream.clone_htod(&group_off).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let run = |y: &mut cudarc::driver::CudaSlice<bf16>| {
        k.gemv_q4g64_slots_tc_bf16(
            &slots, &ids_v, &goff_v, &x, y, e, n, kdim, kdim, kdim as i32,
            blob as u64, 0, 0, tiles,
        )
        .unwrap();
    };
    for _ in 0..3 {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let iters = 30;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let bytes = (e * n * (kdim / 2)) as f64;
    eprintln!(
        "[gemv-bw] {tag}: E{e} n{n} k{kdim} m{m}｜{:.3} ms/次｜{:.1} GB/s",
        dt * 1e3,
        bytes / dt / 1e9
    );
}
