//! **MoE shared expert 线上真实形状的微基准**（2026-10-24）。
//!
//! 动机：`LINGXI_SEG3_SYNC=1` 的分段表里 shared 段（`moe/forward.rs` 157~191）
//! 层均 **75~94 µs**，而三块权重合计只有 ~9.8 MB（bf16）⇒ ~100-130 GB/s，明显低于
//! 同形状稠密 bf16 的上限。本用例只复刻**线上形状与分派**、不改核，回答：
//! "78 µs 花在哪、每个候选核在同样形状上能到多少"。
//!
//! ⚠️ 事实校正（与任务书的前提不同）：`MoeWeights::shared_gate_up/shared_down` 在产物里
//! 是 **BF16**（`MoeWeights` 的字段类型就是 `CudaSlice<bf16>`，`upload_bf16` 直传），
//! 不是 Q4G64。所以默认路径走的是 `gemm_skinny_row_bf16` / `gemm_skinny_bf16`，
//! **不经过** `gemm_q4_skinny_bf16`。Q4 候选核在本文件里用合成 Q4 权重量，仅作
//! "如果权重是 Q4 会怎样"的对照（换 Q4 会改数值 ⇒ 与逐位约束不兼容）。
//!
//! 门控：
//! `cargo test --release -p lingxi-kernels --features gpu --test t8_moe_shared_bw -- --nocapture`

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

const HIDDEN: usize = 2560;
const INTER: usize = 640;

fn time_launches(k: &Kernels, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    k.stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    k.stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
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

fn report(tag: &str, us: f64, bytes: usize) {
    eprintln!(
        "[shared-bw] {tag:<44} {us:8.1} µs｜{:6.1} GB/s｜{:5.2} MB",
        bytes as f64 / (us * 1e-6) / 1e9,
        bytes as f64 / 1e6
    );
}

/// 稠密 Q4G64（与 t6 同款几何）：payload = base 平面 + scale 平面。
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
    let total = scale_off as usize + rows * (k_pad as usize / 64) * 2;
    Q4 { packed: unsafe { k.stream.alloc::<u8>(total) }.unwrap(), k_pad, scale_off }
}

/// 线上 shared 段（`moe/forward.rs` 默认分支）逐核复刻：gu(epi=4→GEMM+swiglu) / down / sg / row_scale。
fn shared_chain(k: &Kernels, rows: usize) -> f64 {
    let x = k.stream.clone_htod(&rndbf16(rows * HIDDEN, 2)).unwrap();
    let wgu = k.stream.clone_htod(&rndbf16(2 * INTER * HIDDEN, 3)).unwrap();
    let wdn = k.stream.clone_htod(&rndbf16(HIDDEN * INTER, 4)).unwrap();
    let wsg = k.stream.clone_htod(&rndbf16(HIDDEN, 5)).unwrap();
    let mut act_full = k.stream.alloc_zeros::<bf16>(rows * 2 * INTER).unwrap();
    let mut act = k.stream.alloc_zeros::<bf16>(rows * INTER).unwrap();
    let mut shared = k.stream.alloc_zeros::<bf16>(rows * HIDDEN).unwrap();
    let mut sg = k.stream.alloc_zeros::<bf16>(rows).unwrap();
    let bytes = 2 * INTER * HIDDEN * 2 + HIDDEN * INTER * 2 + HIDDEN * 2;
    let us = time_launches(k, 50, || {
        {
            let (wp, _a) = wgu.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            let (afp, _c) = act_full.device_ptr_mut(&k.stream);
            k.gemm_skinny_row_bf16_ptr(wp, xp, afp, 2 * INTER, HIDDEN, rows, 0, 0.0, 0.0)
                .unwrap();
        }
        k.swiglu_bf16(&act_full, &mut act, rows, INTER).unwrap();
        {
            let (ap, _d) = act.device_ptr(&k.stream);
            let (dp, _e) = wdn.device_ptr(&k.stream);
            let (sp, _f) = shared.device_ptr_mut(&k.stream);
            k.gemm_skinny_bf16_ptr(dp, ap, sp, HIDDEN, INTER, rows, 0, 0.0, 0.0).unwrap();
        }
        {
            let (gp, _g) = wsg.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            let (sgp, _h) = sg.device_ptr_mut(&k.stream);
            k.gemm_skinny_row_bf16_ptr(gp, xp, sgp, 1, HIDDEN, rows, 2, 1.0, 1.0).unwrap();
        }
        k.row_scale_inplace_bf16(&mut shared, &sg, rows, HIDDEN).unwrap();
    });
    report(&format!("【线上链】m={rows} gu+swiglu+dn+sg+scale"), us, bytes);
    // 逐核拆开
    let us_gu = time_launches(k, 50, || {
        let (wp, _a) = wgu.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (afp, _c) = act_full.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, afp, 2 * INTER, HIDDEN, rows, 0, 0.0, 0.0)
            .unwrap();
    });
    report(&format!("  · gu skinny_row m={rows}"), us_gu, 2 * INTER * HIDDEN * 2);
    let us_sw = time_launches(k, 50, || {
        k.swiglu_bf16(&act_full, &mut act, rows, INTER).unwrap();
    });
    report(&format!("  · swiglu m={rows}"), us_sw, rows * INTER * 2 * 2);
    let us_dn = time_launches(k, 50, || {
        let (dp, _e) = wdn.device_ptr(&k.stream);
        let (ap, _d) = act.device_ptr(&k.stream);
        let (sp, _f) = shared.device_ptr_mut(&k.stream);
        k.gemm_skinny_bf16_ptr(dp, ap, sp, HIDDEN, INTER, rows, 0, 0.0, 0.0).unwrap();
    });
    report(&format!("  · dn skinny(n/8块) m={rows}"), us_dn, HIDDEN * INTER * 2);
    let us_sg = time_launches(k, 50, || {
        let (gp, _g) = wsg.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (sgp, _h) = sg.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(gp, xp, sgp, 1, HIDDEN, rows, 2, 1.0, 1.0).unwrap();
    });
    report(&format!("  · sg n=1 m={rows}"), us_sg, HIDDEN * 2);
    let us_rs = time_launches(k, 50, || {
        k.row_scale_inplace_bf16(&mut shared, &sg, rows, HIDDEN).unwrap();
    });
    report(&format!("  · row_scale m={rows}"), us_rs, rows * HIDDEN * 2 * 2);
    us
}

/// gate_up `[1280,2560]` 的 bf16 候选核（m=rows）。
fn gu_candidates(k: &Kernels, rows: usize) {
    let x = k.stream.clone_htod(&rndbf16(rows * HIDDEN, 12)).unwrap();
    let mut xf = k.stream.alloc_zeros::<f32>(rows * HIDDEN).unwrap();
    let w = k.stream.clone_htod(&rndbf16(2 * INTER * HIDDEN, 13)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(rows * 2 * INTER).unwrap();
    let bytes = 2 * INTER * HIDDEN * 2;
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, 2 * INTER, HIDDEN, rows, 0, 0.0, 0.0).unwrap();
    });
    report(&format!("gu 现状 gemm_skinny_row m={rows}"), us, bytes);
    k.convert_bf16_f32(&x, &mut xf, rows * HIDDEN).unwrap();
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xfp, _d) = xf.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_f32x_ptr(wp, xfp, yp, 2 * INTER, HIDDEN, rows, 0, 0.0, 0.0).unwrap();
    });
    report(&format!("gu 候选 f32x          m={rows}"), us, bytes);
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_bf16_ptr(wp, xp, yp, 2 * INTER, HIDDEN, rows, 0, 0.0, 0.0).unwrap();
    });
    report(&format!("gu 候选 skinny(n/8块) m={rows}"), us, bytes);
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_multi_bf16_ptr(wp, xp, yp, 2 * INTER, HIDDEN, rows, 0, 0.0, 0.0)
            .unwrap();
    });
    report(&format!("gu 候选 row_multi R4  m={rows}"), us, bytes);
    let us_us = time_launches(k, 50, || {
        k.convert_bf16_f32(&x, &mut xf, rows * HIDDEN).unwrap();
    });
    report(&format!("  · convert_bf16_f32 x  m={rows}"), us_us, rows * HIDDEN * 2);
}

/// shared_down `[2560,640]`（k=640）的 bf16 候选核。
fn dn_candidates(k: &Kernels, rows: usize) {
    let x = k.stream.clone_htod(&rndbf16(rows * INTER, 22)).unwrap();
    let mut xf = k.stream.alloc_zeros::<f32>(rows * INTER).unwrap();
    let w = k.stream.clone_htod(&rndbf16(HIDDEN * INTER, 23)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(rows * HIDDEN).unwrap();
    let bytes = HIDDEN * INTER * 2;
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_bf16_ptr(wp, xp, yp, HIDDEN, INTER, rows, 0, 0.0, 0.0).unwrap();
    });
    report(&format!("dn 现状 skinny(n/8块) m={rows}"), us, bytes);
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, HIDDEN, INTER, rows, 0, 0.0, 0.0).unwrap();
    });
    report(&format!("dn 候选 skinny_row    m={rows}"), us, bytes);
    k.convert_bf16_f32(&x, &mut xf, rows * INTER).unwrap();
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xfp, _d) = xf.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_f32x_ptr(wp, xfp, yp, HIDDEN, INTER, rows, 0, 0.0, 0.0).unwrap();
    });
    report(&format!("dn 候选 row f32x      m={rows}"), us, bytes);
}

/// shared_gate 行 `[1,2560]`（epi=2）。
fn sg_candidates(k: &Kernels, rows: usize) {
    let x = k.stream.clone_htod(&rndbf16(rows * HIDDEN, 32)).unwrap();
    let mut xf = k.stream.alloc_zeros::<f32>(rows * HIDDEN).unwrap();
    let w = k.stream.clone_htod(&rndbf16(HIDDEN, 33)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(rows).unwrap();
    let bytes = HIDDEN * 2;
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, 1, HIDDEN, rows, 2, 1.0, 1.0).unwrap();
    });
    report(&format!("sg 现状 row(n=1)      m={rows}"), us, bytes);
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_bf16_ptr(wp, xp, yp, 1, HIDDEN, rows, 2, 1.0, 1.0).unwrap();
    });
    report(&format!("sg 候选 skinny(n/8块)m={rows}"), us, bytes);
    k.convert_bf16_f32(&x, &mut xf, rows * HIDDEN).unwrap();
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.device_ptr(&k.stream);
        let (xfp, _d) = xf.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_f32x_ptr(wp, xfp, yp, 1, HIDDEN, rows, 2, 1.0, 1.0).unwrap();
    });
    report(&format!("sg 候选 row f32x      m={rows}"), us, bytes);
}

/// **仅作对照**：假如 shared 权重是 Q4G64（会把数值改掉，与逐位约束不兼容）。
fn q4_hypothetical(k: &Kernels, rows: usize) {
    let x = k.stream.clone_htod(&rndbf16(rows * HIDDEN, 42)).unwrap();
    let w = q4w(k, 2 * INTER, HIDDEN);
    let mut y = k.stream.alloc_zeros::<bf16>(rows * 2 * INTER).unwrap();
    let bytes = 2 * INTER * HIDDEN / 2;
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.packed.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_q4_skinny_bf16_ptr(
            wp, xp, yp, 2 * INTER, HIDDEN, w.k_pad, rows, 0, w.scale_off, 0, 0.0, 0.0,
        )
        .unwrap();
    });
    report(&format!("gu Q4 skinny(n/8块)   m={rows}"), us, bytes);
    let ids_d = k.stream.clone_htod(&[0i32]).unwrap();
    let goff_d = k.stream.clone_htod(&[0i32, rows as i32]).unwrap();
    let (ids_v, goff_v) = (ids_d.slice(0..1), goff_d.slice(0..2));
    for rpw in [1usize, 2, 4] {
        let us = time_launches(k, 50, || {
            let (wp, _a) = w.packed.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            k.gemv_q4g64_slotsm_rpw_bf16_view(
                &w.packed, &ids_v, &x, &goff_v, &mut y, 1, 2 * INTER, HIDDEN, w.k_pad, HIDDEN,
                0, 0, w.scale_off, rpw, 0,
            )
            .unwrap();
            let _ = (wp, xp);
        });
        report(&format!("gu Q4 slotsm rpw={rpw}     m={rows}"), us, bytes);
    }
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.packed.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemm_q4_tcp_dense_ptr(
            wp, xp, yp, 2 * INTER, HIDDEN, w.k_pad as usize, rows, 0, w.scale_off, 0, 0.0, 0.0,
        )
        .unwrap();
    });
    report(&format!("gu Q4 tcp_dense       m={rows}"), us, bytes);
    let us = time_launches(k, 50, || {
        let (wp, _a) = w.packed.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        for r in 0..rows {
            k.gemv_q4g64_dense_epi_ptr(
                wp, xp + (r * HIDDEN * 2) as u64, yp + (r * 2 * INTER * 2) as u64,
                2 * INTER, HIDDEN, w.k_pad, 0, w.scale_off, 0, 0.0, 0.0,
            )
            .unwrap();
        }
    });
    report(&format!("gu Q4 dense_epi×m     m={rows}"), us, bytes);
}

#[allow(clippy::too_many_arguments)]
fn old_chain(
    k: &Kernels, x: &CudaSlice<bf16>, wgu: &CudaSlice<bf16>, wdn: &CudaSlice<bf16>,
    wsg: &CudaSlice<bf16>, rows: usize, act_full: &mut CudaSlice<bf16>,
    act: &mut CudaSlice<bf16>, sh: &mut CudaSlice<bf16>, sg: &mut CudaSlice<bf16>,
) {
    {
        let (wp, _a) = wgu.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (afp, _c) = act_full.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, afp, 2 * INTER, HIDDEN, rows, 0, 0.0, 0.0).unwrap();
    }
    k.swiglu_bf16(act_full, act, rows, INTER).unwrap();
    {
        let (dp, _e) = wdn.device_ptr(&k.stream);
        let (ap, _d) = act.device_ptr(&k.stream);
        let (sp, _f) = sh.device_ptr_mut(&k.stream);
        k.gemm_skinny_bf16_ptr(dp, ap, sp, HIDDEN, INTER, rows, 0, 0.0, 0.0).unwrap();
    }
    {
        let (gp, _g) = wsg.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (sgp, _h) = sg.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(gp, xp, sgp, 1, HIDDEN, rows, 2, 1.0, 1.0).unwrap();
    }
    k.row_scale_inplace_bf16(sh, sg, rows, HIDDEN).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn new_chain(
    k: &Kernels, x: &CudaSlice<bf16>, wgu: &CudaSlice<bf16>, wdn: &CudaSlice<bf16>,
    wsg: &CudaSlice<bf16>, rows: usize, act: &mut CudaSlice<bf16>, sh: &mut CudaSlice<bf16>,
    sg: &mut CudaSlice<bf16>,
) {
    {
        let (wp, _a) = wgu.device_ptr(&k.stream);
        let (gp, _g) = wsg.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (ap, _c) = act.device_ptr_mut(&k.stream);
        let (sgp, _h) = sg.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_gu_swiglu_sg_bf16(wp, gp, xp, ap, sgp, INTER, HIDDEN, rows, 1.0, 1.0)
            .unwrap();
    }
    {
        let (dp, _e) = wdn.device_ptr(&k.stream);
        let (ap, _d) = act.device_ptr(&k.stream);
        let (sp, _f) = sh.device_ptr_mut(&k.stream);
        let (sgp, _h) = sg.device_ptr(&k.stream);
        k.gemm_skinny_m8_scale_bf16(dp, ap, sp, sgp, HIDDEN, INTER, rows).unwrap();
    }
}

/// 新融合链 vs 原 5 发链：**逐位对拍 + 计时**（rows > 1 才走融合）。
fn fused_vs_old(k: &Kernels, rows: usize) {
    let x = k.stream.clone_htod(&rndbf16(rows * HIDDEN, 52)).unwrap();
    let wgu = k.stream.clone_htod(&rndbf16(2 * INTER * HIDDEN, 53)).unwrap();
    let wdn = k.stream.clone_htod(&rndbf16(HIDDEN * INTER, 54)).unwrap();
    let wsg = k.stream.clone_htod(&rndbf16(HIDDEN, 55)).unwrap();
    let mut act_full = k.stream.alloc_zeros::<bf16>(rows * 2 * INTER).unwrap();
    let mut act_a = k.stream.alloc_zeros::<bf16>(rows * INTER).unwrap();
    let mut act_b = k.stream.alloc_zeros::<bf16>(rows * INTER).unwrap();
    let mut sh_a = k.stream.alloc_zeros::<bf16>(rows * HIDDEN).unwrap();
    let mut sh_b = k.stream.alloc_zeros::<bf16>(rows * HIDDEN).unwrap();
    let mut sg_a = k.stream.alloc_zeros::<bf16>(rows).unwrap();
    let mut sg_b = k.stream.alloc_zeros::<bf16>(rows).unwrap();
    let bytes = 2 * INTER * HIDDEN * 2 + HIDDEN * INTER * 2;

    old_chain(k, &x, &wgu, &wdn, &wsg, rows, &mut act_full, &mut act_a, &mut sh_a, &mut sg_a);
    new_chain(k, &x, &wgu, &wdn, &wsg, rows, &mut act_b, &mut sh_b, &mut sg_b);
    k.stream.synchronize().unwrap();
    let ha = k.stream.clone_dtoh(&sh_a).unwrap();
    let hb = k.stream.clone_dtoh(&sh_b).unwrap();
    let aa = k.stream.clone_dtoh(&act_a).unwrap();
    let ab = k.stream.clone_dtoh(&act_b).unwrap();
    let ga = k.stream.clone_dtoh(&sg_a).unwrap();
    let gb = k.stream.clone_dtoh(&sg_b).unwrap();
    let mut d_shared = 0usize;
    let mut d_act = 0usize;
    let mut d_sg = 0usize;
    for i in 0..rows * HIDDEN {
        if ha[i].to_bits() != hb[i].to_bits() {
            if d_shared < 4 {
                eprintln!(
                    "[shared-bw]   shared diff@{i}: {:#x} vs {:#x}",
                    ha[i].to_bits(), hb[i].to_bits()
                );
            }
            d_shared += 1;
        }
    }
    for i in 0..rows * INTER {
        if aa[i].to_bits() != ab[i].to_bits() {
            d_act += 1;
        }
    }
    for i in 0..rows {
        if ga[i].to_bits() != gb[i].to_bits() {
            d_sg += 1;
        }
    }
    eprintln!(
        "[shared-bw] 逐位对拍 m={rows}：shared {d_shared}/{}、act {d_act}/{}、sg {d_sg}/{rows} 不一致",
        rows * HIDDEN,
        rows * INTER
    );
    assert_eq!((d_shared, d_act, d_sg), (0, 0, 0), "融合核与旧链不逐位相同");
    let us_old = time_launches(k, 50, || {
        old_chain(k, &x, &wgu, &wdn, &wsg, rows, &mut act_full, &mut act_a, &mut sh_a, &mut sg_a)
    });
    report(&format!("【融合对比】旧链 5 发 m={rows}"), us_old, bytes);
    let us_new = time_launches(k, 50, || {
        new_chain(k, &x, &wgu, &wdn, &wsg, rows, &mut act_b, &mut sh_b, &mut sg_b)
    });
    report(&format!("【融合对比】新链 2 发 m={rows}"), us_new, bytes);
    // 融合核逐发
    let us_k1 = time_launches(k, 50, || {
        let (wp, _a) = wgu.device_ptr(&k.stream);
        let (gp, _g) = wsg.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (ap, _c) = act_b.device_ptr_mut(&k.stream);
        let (sgp, _h) = sg_b.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_gu_swiglu_sg_bf16(wp, gp, xp, ap, sgp, INTER, HIDDEN, rows, 1.0, 1.0)
            .unwrap();
    });
    report(
        &format!("  · 融合核1 gu+swiglu+sg m={rows}"),
        us_k1,
        2 * INTER * HIDDEN * 2 + HIDDEN * 2,
    );
    let us_k2 = time_launches(k, 50, || {
        let (dp, _e) = wdn.device_ptr(&k.stream);
        let (ap, _d) = act_b.device_ptr(&k.stream);
        let (sp, _f) = sh_b.device_ptr_mut(&k.stream);
        let (sgp, _h) = sg_b.device_ptr(&k.stream);
        k.gemm_skinny_m8_scale_bf16(dp, ap, sp, sgp, HIDDEN, INTER, rows).unwrap();
    });
    report(&format!("  · 融合核2 dn+scale   m={rows}"), us_k2, HIDDEN * INTER * 2);
}

/// gate_up 核的 **n 方向标度**（回答"1280 行的 232 GB/s 是网格不够还是核本身的地板"）。
fn gu_n_scaling(k: &Kernels, rows: usize) {
    let x = k.stream.clone_htod(&rndbf16(rows * HIDDEN, 62)).unwrap();
    for n in [1280usize, 2560, 5120, 10240] {
        let w = k.stream.clone_htod(&rndbf16(n * HIDDEN, 63)).unwrap();
        let mut y = k.stream.alloc_zeros::<bf16>(rows * n).unwrap();
        let us = time_launches(k, 30, || {
            let (wp, _a) = w.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            let (yp, _c) = y.device_ptr_mut(&k.stream);
            k.gemm_skinny_row_bf16_ptr(wp, xp, yp, n, HIDDEN, rows, 0, 0.0, 0.0).unwrap();
        });
        report(&format!("gu 标度 n={n:<6} k=2560 m={rows}"), us, n * HIDDEN * 2);
    }
}

#[test]
fn t8_moe_shared_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    eprintln!(
        "[shared-bw] 权重是 **bf16**（9.84 MB/层），任务书里的 Q4 2.95 MB 前提不成立；\
         本表按 bf16 现状量，Q4 段仅作对照"
    );
    for rows in [5usize, 6] {
        shared_chain(&k, rows);
        gu_candidates(&k, rows);
        dn_candidates(&k, rows);
        sg_candidates(&k, rows);
        q4_hypothetical(&k, rows);
        fused_vs_old(&k, rows);
        gu_n_scaling(&k, rows);
    }
}
