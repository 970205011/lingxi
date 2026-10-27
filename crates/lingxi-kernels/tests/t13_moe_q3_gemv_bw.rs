//! **Q3G128 专家 GEMV 取数结构微基准**（2026-10-24，）。
//!
//! Q4 的 `slotsm_t2_bf16` 已在线上白捡 58%（gu E40 m=1 286→453 GB/s）；Q3 仍是老体
//! （每 8 列 **3 次 1 字节 `__ldg`** + PF=4 移位链 + 4~5 条/值的解包）。本用例量三件事：
//!
//! 1. **现状核** `gemv_q3g128_slotsm_rpw_bf16_view`（rpw=1、xf=0，= 线上混档 decode 那一支）：
//! gu（n=1280 k=2560）与 dn（n=2560 k=640），E=5/10、m=1 与 m≈1.5；
//! 2. **同形纯读**（`gemv_q3_exp.cu` 的 `q3_read_probe`，地址映射与生产核逐字节相同）：
//! mode 0 = 3×1 B（生产取数形状）/ 1 = 4 B 未对齐 / 2 = 16 B uint4 / 3 = 4 B + 解包 + I2F；
//! 3. **候选结构**（`q3_slotsm_body<UNR, DEC, LOAD>`，全部与生产核**逐位相同**）：
//! UNR=1/2/4 个独立 load 取代 PF=4 移位链；DEC=移位对解包；LOAD=4 B / 两对齐 u32。
//!
//! 口径：GB/s 按**权重字节**（e·n·k·3/8）折算。`--test-threads=1` 是硬要求
//! 。
//!
//! `bash
//! export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib   # cublas/nvrtc 不在默认路径时按需追加
//! cargo test --release -p lingxi-kernels --features gpu --test t13_moe_q3_gemv_bw \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaView, DevicePtr};
use half::bf16;
use lingxi_kernels::Kernels;

const HIDDEN: usize = 2560;
const INTER: usize = 640;
/// gu 段：n = 2*inter = 1280 行、k = hidden = 2560（每行 Q3 = 960 B）。
const GU_N: usize = 2 * INTER;
const GU_K: usize = HIDDEN;

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

/// `matrix_layout` 的同公式（Q3G128，每 128 值 48 B）：blob / gu_scale / dn_seg / dn_scale。
fn geo_q3() -> (u64, u64, u64, u64) {
    let layout = |rows: u64, cols: u64| -> (u64, u64) {
        let k_pad = align_up(cols, 128);
        let groups = k_pad / 128;
        let base = rows * groups * 48;
        let scale_off = align_up(base, 256);
        (scale_off, scale_off + rows * groups * 2)
    };
    let (gu_scale, gu_payload) = layout(GU_N as u64, GU_K as u64);
    let (dn_scale, dn_payload) = layout(HIDDEN as u64, INTER as u64);
    (gu_payload + dn_payload, gu_scale, gu_payload, dn_scale)
}

fn group_off(e: usize, pat: u8) -> Vec<i32> {
    let mut v = Vec::with_capacity(e + 1);
    v.push(0i32);
    for g in 0..e {
        let m = match pat {
            1 => 1,
            2 => 2,
            5 => 5,
            _ => {
                if g % 2 == 0 {
                    1
                } else {
                    2
                }
            }
        };
        v.push(v[g] + m);
    }
    v
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

fn report(tag: &str, bytes: f64, dt: f64, extra: &str) {
    eprintln!(
        "[q3-bw] {tag:<46}｜{:8.3} ms｜{:7.1} GB/s｜{:6.2} MB{extra}",
        dt * 1e3,
        bytes / dt / 1e9,
        bytes / 1e6
    );
}

/// 随机权重字节 + **合理 scale**（fp16 0.002~0.024；随机字节的 fp16 多为 NaN/denorm）。
fn fill_slots(e: usize, blob: u64, gu_scale: u64, dn_seg: u64, dn_scale: u64) -> Vec<u8> {
    let mut b = vec![0u8; e * blob as usize];
    let mut s: u32 = 0x1234_5678;
    for v in b.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *v = (s & 0xFF) as u8;
    }
    let put = |b: &mut Vec<u8>, base: u64, rows: usize, groups: usize, ei: usize| {
        for i in 0..rows * groups {
            let h = half::f16::from_f32(0.002 + 0.001 * ((i % 23) as f32));
            let off = ei * blob as usize + base as usize + 2 * i;
            b[off..off + 2].copy_from_slice(&h.to_le_bytes());
        }
    };
    for ei in 0..e {
        put(&mut b, gu_scale, GU_N, GU_K / 128, ei);
        put(&mut b, dn_seg + dn_scale, HIDDEN, INTER / 128, ei);
    }
    b
}

fn make_x(s: usize, k: usize) -> Vec<bf16> {
    (0..s * k)
        .map(|i| bf16::from_f32(((i * 37 % 101) as f32 / 101.0) - 0.5))
        .collect()
}

struct Ctx {
    e: usize,
    slots: CudaSlice<u8>,
    ids: Vec<i32>,
    goff: Vec<i32>,
    s: usize,
}

impl Ctx {
    fn new(k: &Kernels, e: usize, pat: u8, gu_scale: u64, dn_seg: u64, dn_scale: u64) -> Self {
        let (blob, _, _, _) = geo_q3();
        let host = fill_slots(e, blob, gu_scale, dn_seg, dn_scale);
        let slots = k.stream.clone_htod(&host).unwrap();
        let ids: Vec<i32> = (0..e as i32).collect();
        let goff = group_off(e, pat);
        let s = *goff.last().unwrap() as usize;
        Ctx { e, slots, ids, goff, s }
    }
    fn views<'a>(
        &'a self, ids: &'a CudaSlice<i32>, goff: &'a CudaSlice<i32>,
    ) -> (CudaView<'a, i32>, CudaView<'a, i32>) {
        (ids.slice(0..self.e), goff.slice(0..=self.e))
    }
}

/// 现状核（生产默认档：rpw=1、xf=0）。
#[allow(clippy::too_many_arguments)]
fn bench_prod(k: &Kernels, tag: &str, c: &Ctx, n: usize, kk: usize, a_stride: usize, seg: u64, sc: u64) {
    let (blob, _, _, _) = geo_q3();
    let ids = k.stream.clone_htod(&c.ids).unwrap();
    let goff = k.stream.clone_htod(&c.goff).unwrap();
    let (ids_v, goff_v) = c.views(&ids, &goff);
    let x = k.stream.clone_htod(&make_x(c.s, kk)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
    let dt = time_it(k, 60, || {
        k.gemv_q3g128_slotsm_rpw_bf16_view(
            &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, n, kk, kk as i32, a_stride, blob, seg, sc,
            1, 0,
        )
        .unwrap();
    });
    let bytes = c.e as f64 * n as f64 * kk as f64 * 3.0 / 8.0;
    report(tag, bytes, dt, &format!("（m={:.1}）", c.s as f64 / c.e as f64));
}

/// 候选变体（`which` 见 wrapper 注释）。
#[allow(clippy::too_many_arguments)]
fn bench_exp(k: &Kernels, tag: &str, c: &Ctx, which: usize, n: usize, kk: usize, a_stride: usize,
             seg: u64, sc: u64) {
    let (blob, _, _, _) = geo_q3();
    let ids = k.stream.clone_htod(&c.ids).unwrap();
    let goff = k.stream.clone_htod(&c.goff).unwrap();
    let (ids_v, goff_v) = c.views(&ids, &goff);
    let x = k.stream.clone_htod(&make_x(c.s, kk)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
    let dt = time_it(k, 60, || {
        k.gemv_q3g128_slotsm_exp(
            which, &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, n, kk, kk as i32, a_stride, blob,
            seg, sc, 0,
        )
        .unwrap();
    });
    let bytes = c.e as f64 * n as f64 * kk as f64 * 3.0 / 8.0;
    report(tag, bytes, dt, &format!("（m={:.1}）", c.s as f64 / c.e as f64));
}

/// 同形纯读（地址映射与生产核逐字节相同）。
fn bench_read(k: &Kernels, tag: &str, c: &Ctx, n: usize, kk: usize, mode: i32) {
    let (blob, _, _, _) = geo_q3();
    let row_bytes = (align_up(kk as u64, 128) as usize / 128) * 48;
    let mut out = k.stream.alloc_zeros::<f32>(c.e * n).unwrap();
    let dt = time_it(k, 60, || {
        k.q3_read_probe(&c.slots, &mut out, c.e, n, row_bytes, kk, blob, mode).unwrap();
    });
    let bytes = c.e as f64 * n as f64 * kk as f64 * 3.0 / 8.0;
    report(tag, bytes, dt, "");
}

/// 逐位对照：变体 vs 生产核（gu 形状）。
fn check(k: &Kernels, which: usize, c: &Ctx) {
    check_shape(k, which, c, GU_N, GU_K, GU_K, 0, geo_q3().1);
}

/// 逐位对照（任意形状）：`gemv_q3g128_slotsm_exp` 变体 vs 生产核。
#[allow(clippy::too_many_arguments)]
fn check_shape(k: &Kernels, which: usize, c: &Ctx, n: usize, kk: usize, a_stride: usize,
               seg: u64, sc: u64) {
    let (blob, _, _, _) = geo_q3();
    let ids = k.stream.clone_htod(&c.ids).unwrap();
    let goff = k.stream.clone_htod(&c.goff).unwrap();
    let (ids_v, goff_v) = c.views(&ids, &goff);
    let x = k.stream.clone_htod(&make_x(c.s, kk)).unwrap();
    let mut y_ref = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
    let mut y_cmp = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
    k.gemv_q3g128_slotsm_rpw_bf16_view(
        &c.slots, &ids_v, &x, &goff_v, &mut y_ref, c.e, n, kk, kk as i32, a_stride, blob, seg, sc,
        1, 0,
    )
    .unwrap();
    k.gemv_q3g128_slotsm_exp(
        which, &c.slots, &ids_v, &x, &goff_v, &mut y_cmp, c.e, n, kk, kk as i32, a_stride, blob,
        seg, sc, 0,
    )
    .unwrap();
    k.stream.synchronize().unwrap();
    let hr: Vec<bf16> = k.stream.clone_dtoh(&y_ref).unwrap();
    let hc: Vec<bf16> = k.stream.clone_dtoh(&y_cmp).unwrap();
    let mut n_diff = 0usize;
    let mut max_abs = 0.0f32;
    for (a, b) in hr.iter().zip(hc.iter()) {
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
        }
        let (af, bf) = (a.to_f32(), b.to_f32());
        if af.is_finite() && bf.is_finite() {
            max_abs = max_abs.max((af - bf).abs());
        }
    }
    eprintln!(
        "[check] 变体 {which}（n{n} k{kk}）vs 生产核：{n_diff} / {} 元素逐位不同｜max|Δ|={max_abs:.6e}",
        hr.len()
    );
    assert_eq!(n_diff, 0, "变体 {which} 与生产核不逐位相同");
}

/// **生产 t2 档**（`gemv_q3g128_slotsm_t2_view_gidx`，即线上真正走的那条）vs 旧核：逐位对照。
fn check_t2(k: &Kernels, tag: &str, c: &Ctx, n: usize, kk: usize, a_stride: usize, seg: u64,
            sc: u64, all_m1: bool) {
    let (blob, _, _, _) = geo_q3();
    let ids = k.stream.clone_htod(&c.ids).unwrap();
    let goff = k.stream.clone_htod(&c.goff).unwrap();
    let (ids_v, goff_v) = c.views(&ids, &goff);
    let x = k.stream.clone_htod(&make_x(c.s, kk)).unwrap();
    let mut yr = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
    let mut yc = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
    k.gemv_q3g128_slotsm_rpw_bf16_view(
        &c.slots, &ids_v, &x, &goff_v, &mut yr, c.e, n, kk, kk as i32, a_stride, blob, seg, sc, 1,
        0,
    )
    .unwrap();
    k.gemv_q3g128_slotsm_t2_view_gidx(
        &c.slots, &ids_v, &x, &goff_v, &mut yc, c.e, n, kk, kk as i32, a_stride, blob, seg, sc, 1,
        0, 0, all_m1,
    )
    .unwrap();
    k.stream.synchronize().unwrap();
    let hr: Vec<bf16> = k.stream.clone_dtoh(&yr).unwrap();
    let hc: Vec<bf16> = k.stream.clone_dtoh(&yc).unwrap();
    let nd = hr.iter().zip(hc.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    eprintln!("[check] 生产 t2 档 {tag}（all_m1={all_m1}）：{nd} / {} 元素逐位不同", hr.len());
    assert_eq!(nd, 0, "生产 t2 档 {tag} 与旧核不逐位相同");
}

/// **背靠背 A/B**：同一条命令里交替跑旧核 / 新核（t2 生产档），各 3 轮，逐轮打印（剔除冷启动）。
#[allow(clippy::too_many_arguments)]
fn bench_ab(k: &Kernels, tag: &str, c: &Ctx, n: usize, kk: usize, a_stride: usize, seg: u64,
            sc: u64, all_m1: bool) {
    let (blob, _, _, _) = geo_q3();
    let ids = k.stream.clone_htod(&c.ids).unwrap();
    let goff = k.stream.clone_htod(&c.goff).unwrap();
    let (ids_v, goff_v) = c.views(&ids, &goff);
    let x = k.stream.clone_htod(&make_x(c.s, kk)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
    let mut old = Vec::new();
    let mut new = Vec::new();
    for _ in 0..3 {
        old.push(time_it(k, 40, || {
            k.gemv_q3g128_slotsm_rpw_bf16_view(
                &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, n, kk, kk as i32, a_stride, blob, seg,
                sc, 1, 0,
            )
            .unwrap();
        }));
        new.push(time_it(k, 40, || {
            k.gemv_q3g128_slotsm_t2_view_gidx(
                &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, n, kk, kk as i32, a_stride, blob,
                seg, sc, 1, 0, 0, all_m1,
            )
            .unwrap();
        }));
    }
    let bytes = c.e as f64 * n as f64 * kk as f64 * 3.0 / 8.0;
    let g = |dt: f64| bytes / dt / 1e9;
    // 中位数（3 轮取中）作为口径，同时把全部样本打出来
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (mo, mn) = (med(&mut old), med(&mut new));
    eprintln!(
        "[q3-ab] {tag:<28}｜旧 {:6.1} µs（{:5.1} GB/s）｜新 {:6.1} µs（{:5.1} GB/s）｜{:4.2}×\
         ｜样本 旧{:?} 新{:?}",
        mo * 1e6,
        g(mo),
        mn * 1e6,
        g(mn),
        mo / mn,
        old.iter().map(|v| (v * 1e6 * 10.0).round() / 10.0).collect::<Vec<_>>(),
        new.iter().map(|v| (v * 1e6 * 10.0).round() / 10.0).collect::<Vec<_>>(),
    );
}

#[test]
fn t13_moe_q3_gemv_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let (blob, gu_scale, dn_seg, dn_scale) = geo_q3();
    eprintln!(
        "[q3-bw] 几何：Q3 blob {blob} B｜gu n{GU_N} k{GU_K} 每行 {} B（单专家 {:.3} MB）｜\
         dn n{HIDDEN} k{INTER} 每行 {} B（{:.3} MB）",
        GU_K / 8 * 3,
        GU_N as f64 * GU_K as f64 * 3.0 / 8.0 / 1e6,
        INTER / 8 * 3,
        HIDDEN as f64 * INTER as f64 * 3.0 / 8.0 / 1e6
    );

    let c5_m1 = Ctx::new(&k, 5, 1, gu_scale, dn_seg, dn_scale);
    let c10_m1 = Ctx::new(&k, 10, 1, gu_scale, dn_seg, dn_scale);
    let c10_m15 = Ctx::new(&k, 10, 3, gu_scale, dn_seg, dn_scale);
    let c3_m1 = Ctx::new(&k, 3, 1, gu_scale, dn_seg, dn_scale);

    let names = [
        "v0 unr1/三字节/生产解包",
        "v1 unr2/三字节/生产解包",
        "v2 unr2/对齐u32对/生产解包",
        "v3 unr2/三字节/移位对",
        "v4 unr2/对齐u32对/移位对（t2 候选）",
        "v5 unr1/对齐u32对/移位对",
        "v6 unr4/对齐u32对/移位对",
        "v7 unr8/对齐u32对/移位对",
        "v8 MB=1 unr1/对齐u32对/移位对",
        "v9 MB=1 unr2/对齐u32对/移位对",
        "v10 MB=1 unr4/对齐u32对/移位对",
        "v11 MB=1 unr2/三字节/移位对",
    ];
    // MB=1 档只在"每组恰好 1 对"时成立 ⇒ 只参与 m=1 的行
    let mb1_ok = |pat_is_m1: bool| -> bool { pat_is_m1 };

    eprintln!("── ① gu（n1280 k2560）现状核 ──");
    for (tag, c) in [("E5 m=1", &c5_m1), ("E10 m=1", &c10_m1), ("E10 m≈1.5", &c10_m15)] {
        bench_prod(&k, &format!("{tag} 生产核"), c, GU_N, GU_K, GU_K, 0, gu_scale);
    }
    eprintln!("── ② gu 同形纯读（分母）──");
    for (tag, c) in [("E3 m=1", &c3_m1), ("E5 m=1", &c5_m1), ("E10 m=1", &c10_m1)] {
        for (m, nm) in [
            (0i32, "纯读 3×1B（生产取数形状）"),
            (1, "纯读 对齐u32对（候选取数形状）"),
            (2, "纯读 16B uint4"),
            (3, "纯读 对齐u32对 + 解包 + I2F"),
        ] {
            bench_read(&k, &format!("{tag} {nm}"), c, GU_N, GU_K, m);
        }
    }
    eprintln!("── ③ gu 候选结构（全部逐位等价；v8+ 是 MB=1 特化，只在 m=1 成立）──");
    for (tag, c, m1) in [
        ("E5 m=1", &c5_m1, true),
        ("E10 m=1", &c10_m1, true),
        ("E10 m≈1.5", &c10_m15, false),
    ] {
        for (w, nm) in names.iter().enumerate() {
            if w >= 8 && !mb1_ok(m1) {
                continue;
            }
            bench_exp(&k, &format!("{tag} {nm}"), c, w, GU_N, GU_K, GU_K, 0, gu_scale);
        }
    }
    eprintln!("── ④ gu 线上形状参照（E3 m=1，每层 Q3 组数量级）──");
    bench_prod(&k, "E3 m=1 生产核", &c3_m1, GU_N, GU_K, GU_K, 0, gu_scale);
    for &w in &[4usize, 9, 10] {
        bench_exp(&k, &format!("E3 m=1 {}", names[w]), &c3_m1, w, GU_N, GU_K, GU_K, 0, gu_scale);
    }

    eprintln!("── ⑤ dn（n2560 k640，每行 240 B）──");
    for (tag, c) in [("E5 m=1", &c5_m1), ("E10 m=1", &c10_m1), ("E10 m≈1.5", &c10_m15)] {
        bench_prod(&k, &format!("{tag} dn 生产核"), c, HIDDEN, INTER, INTER, dn_seg, dn_scale);
    }
    for (m, nm) in [
        (0i32, "纯读 3×1B"),
        (1, "纯读 对齐u32对"),
        (2, "纯读 16B uint4"),
        (3, "纯读 对齐u32对 + 解包 + I2F"),
    ] {
        bench_read(&k, &format!("E3 m=1 dn {nm}"), &c3_m1, HIDDEN, INTER, m);
    }
    for (m, nm) in [
        (0i32, "纯读 3×1B"),
        (1, "纯读 对齐u32对"),
        (2, "纯读 16B uint4"),
        (3, "纯读 对齐u32对 + 解包 + I2F"),
    ] {
        bench_read(&k, &format!("E10 m=1 dn {nm}"), &c10_m1, HIDDEN, INTER, m);
    }
    for (tag, c, m1) in [
        ("E5 m=1", &c5_m1, true),
        ("E10 m=1", &c10_m1, true),
        ("E10 m≈1.5", &c10_m15, false),
    ] {
        for (w, nm) in names.iter().enumerate() {
            if w >= 8 && !mb1_ok(m1) {
                continue;
            }
            bench_exp(&k, &format!("{tag} dn {nm}"), c, w, HIDDEN, INTER, INTER, dn_seg, dn_scale);
        }
    }

    eprintln!("── ⑥ 逐位对照（gu，E5 m=1 / E10 m≈1.5）──");
    for w in 0..names.len() {
        check(&k, w, &c5_m1);
    }
    for w in [4usize, 6] {
        check(&k, w, &c10_m15);
    }
    eprintln!("── ⑦ 逐位对照（dn 形状，E5 m=1）──");
    for w in [4usize, 6, 9, 10] {
        check_shape(&k, w, &c5_m1, HIDDEN, INTER, INTER, dn_seg, dn_scale);
    }

    eprintln!("── ⑧ 背靠背 A/B（旧核 vs t2 生产档，交替各 3 轮 × 40 迭代）──");
    bench_ab(&k, "gu E3 m=1（线上组数）", &c3_m1, GU_N, GU_K, GU_K, 0, gu_scale, true);
    bench_ab(&k, "gu E5 m=1", &c5_m1, GU_N, GU_K, GU_K, 0, gu_scale, true);
    bench_ab(&k, "gu E10 m=1", &c10_m1, GU_N, GU_K, GU_K, 0, gu_scale, true);
    bench_ab(&k, "gu E10 m≈1.5", &c10_m15, GU_N, GU_K, GU_K, 0, gu_scale, false);
    bench_ab(&k, "dn E3 m=1", &c3_m1, HIDDEN, INTER, INTER, dn_seg, dn_scale, true);
    bench_ab(&k, "dn E10 m=1", &c10_m1, HIDDEN, INTER, INTER, dn_seg, dn_scale, true);
    bench_ab(&k, "dn E10 m≈1.5", &c10_m15, HIDDEN, INTER, INTER, dn_seg, dn_scale, false);

    eprintln!("── ⑨ 生产 t2 档逐位对照（真正上线的那条路径）──");
    check_t2(&k, "gu E5 m=1", &c5_m1, GU_N, GU_K, GU_K, 0, gu_scale, true);
    check_t2(&k, "gu E10 m≈1.5", &c10_m15, GU_N, GU_K, GU_K, 0, gu_scale, false);
    check_t2(&k, "dn E5 m=1", &c5_m1, HIDDEN, INTER, INTER, dn_seg, dn_scale, true);
    check_t2(&k, "dn E10 m≈1.5", &c10_m15, HIDDEN, INTER, INTER, dn_seg, dn_scale, false);
}

// ───────────────── ④ Q3 + int8 激活 + dp4a（线上 m≈5 的真实形状） ─────────────────
//
// 线上 `[evprof] 专家段 … rows=5`：每层约 43 个唯一专家、每个专家 5 个 token。
// t2 体在 m=5 时每 group ≈ 24(固定) + 8.5×5 条，其中 42.5 条是**每 token 的 FP32 发射**
// （8 CVT + 8 FFMA）—— 这就是 m 越大带宽越低的机制。dp4a 档把每 token 压到
// 1 LDG.64 + 2 dp4a + 1 I2F + 1 FFMA ≈ 5 条，理论上限 ≈ 1.4×。
//
// 判据：gu ≥ 280 GB/s 且 ≥ 1.35× 同形状 t2；dn ≥ 240 且 ≥ 1.35×。
#[test]
fn t13_q3_dp4a_m5() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let (blob, gu_scale, dn_seg, dn_scale) = geo_q3();
    for (tag, c, n, kk, seg, sc) in [
        ("gu n1280 k2560 E10 m=1（自检，应 ≈300）", Ctx::new(&k, 10, 1, gu_scale, dn_seg, dn_scale), GU_N, GU_K, 0u64, gu_scale),
        ("gu n1280 k2560", Ctx::new(&k, 40, 5, gu_scale, dn_seg, dn_scale), GU_N, GU_K, 0u64, gu_scale),
        ("dn n2560 k640", Ctx::new(&k, 40, 5, gu_scale, dn_seg, dn_scale), HIDDEN, INTER, dn_seg, dn_scale),
    ] {
        let ids = k.stream.clone_htod(&c.ids).unwrap();
        let goff = k.stream.clone_htod(&c.goff).unwrap();
        let (ids_v, goff_v) = c.views(&ids, &goff);
        let x = k.stream.clone_htod(&make_x(c.s, kk)).unwrap();
        let mut y = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
        let bytes = c.e as f64 * n as f64 * kk as f64 * 3.0 / 8.0;
        let dt_t2 = time_it(&k, 40, || {
            k.gemv_q3g128_slotsm_t2_view_gidx(
                &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, n, kk, kk as i32, kk, blob, seg, sc,
                1, 0, 0, false,
            )
            .unwrap();
        });
        // int8 行量化（dp4a 的"提前转化"；每行一个 scale）
        let mut xq = k.stream.alloc_zeros::<u8>(c.s * kk).unwrap();
        let mut sx = k.stream.alloc_zeros::<f32>(c.s).unwrap();
        k.convert_bf16_q8_rs(&x, &mut xq, &mut sx, c.s, kk).unwrap();
        let dt_dp = time_it(&k, 40, || {
            k.gemv_q3g128_slotsm_dp4a_view(
                &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, n, kk, kk as i32, kk, blob, seg, sc,
                0, &xq, &sx,
            )
            .unwrap();
        });
        report(&format!("{tag} m=5 t2（生产默认）"), bytes, dt_t2, "");
        report(&format!("{tag} m=5 dp4a(int8 激活)"), bytes, dt_dp, "");
        eprintln!(
            "[q3-bw]   ⇒ {tag}：{:.2}× （{:.0} → {:.0} GB/s）",
            dt_t2 / dt_dp,
            bytes / dt_t2 / 1e9,
            bytes / dt_dp / 1e9
        );
    }
}

// ───────────────── ⑤ 专家槽的**地址散乱**是不是那 1.4× 的差距？（2026-10-25） ─────────────────
//
// 线上专家段 0.427 ms/层 ⇒ ~185 GB/s；同形微基准（E=40 连续槽）给 255~326。
// 唯一还没验证的差异是**寻址**：生产 arena 是 23 GiB 的常驻池，43 个专家按 slot 表跳读；
// 微基准里 slot_ids = 0..E-1 是连续 100 MB。若 TLB/局部性是主因，散开就会掉。
//
// 做法：同形状（gu n1280 k2560，E=40，m=1）只改 `slot_ids` 的散列度与池子大小：
// 连续 40 槽（100 MB） / 散开 256 槽（640 MB） / 散开 1024 槽（2.5 GB，接近线上量级）。
// 权重内容无关带宽 ⇒ 大池用同一块 blob 重复铺（memcpy，避免逐字节随机填充）。
#[test]
fn t13_q3_slot_locality() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let (blob, gu_scale, dn_seg, dn_scale) = geo_q3();
    let e = 40usize;
    let n = GU_N;
    let kk = GU_K;
    let bytes = e as f64 * n as f64 * kk as f64 * 3.0 / 8.0;
    let goff = k.stream.clone_htod(&{
        let mut v = vec![0i32];
        for g in 0..e {
            v.push(v[g] + 1);
        }
        v
    }).unwrap();
    let goff_v = goff.slice(0..=e);
    let x = k.stream.clone_htod(&make_x(e, kk)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(e * n).unwrap();
    // 一块"正常"的 blob（scale 合法），大池就复制它
    let one = fill_slots(1, blob, gu_scale, dn_seg, dn_scale);
    for (tag, n_slots, stride) in [("连续 40 槽（100 MB）", 40usize, 1usize),
                                   ("散开 256 槽（640 MB）", 256, 7),
                                   ("散开 1024 槽（2.5 GB）", 1024, 29)] {
        let mut host = Vec::with_capacity(n_slots * blob as usize);
        for _ in 0..n_slots {
            host.extend_from_slice(&one);
        }
        let slots = k.stream.clone_htod(&host).unwrap();
        drop(host);
        let ids: Vec<i32> = (0..e).map(|g| ((g * stride) % n_slots) as i32).collect();
        let ids_d = k.stream.clone_htod(&ids).unwrap();
        let ids_v = ids_d.slice(0..e);
        let dt = time_it(&k, 40, || {
            k.gemv_q3g128_slotsm_t2_view_gidx(
                &slots, &ids_v, &x, &goff_v, &mut y, e, n, kk, kk as i32, kk, blob, 0, gu_scale,
                1, 0, 0, false,
            )
            .unwrap();
        });
        report(tag, bytes, dt, "");
    }
}

// ───────────────── ⑥ 静态网格 + 设备组数上界：与精确网格**逐位相同**、速度不亏 ─────────────────
//
// A'的地基：主机要能"先发牌、后核对"，就必须用 `grid.y = s`（上界）
// 而不是精确的 `ng = npri/nalt`。这个用例回答两个问题：
// ① 逐位：两档输出必须完全一致（同一个 body，只多一道入口早退）；
// ② 代价：多出来的空 block（每块一次 4 B 设备读 + 退出）值多少 GB/s。
#[test]
fn t13_q3_static_grid() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let (blob, gu_scale, dn_seg, dn_scale) = geo_q3();
    let n = GU_N;
    let kk = GU_K;
    // 两种形状：m=1（s=40，超发 1.3×，≈线上）与 m=5（s=200，超发 6.7×，最坏情况）
    for (tag, c, ng, tol) in [
        ("m=1 s=40 ng=30（≈线上 1.3× 超发）", Ctx::new(&k, 40, 1, gu_scale, dn_seg, dn_scale), 30usize, 1.05f64),
        ("m=5 s=200 ng=30（最坏 6.7× 超发）", Ctx::new(&k, 40, 5, gu_scale, dn_seg, dn_scale), 30, 1.15),
    ] {
        let ids = k.stream.clone_htod(&c.ids).unwrap();
        let goff = k.stream.clone_htod(&c.goff).unwrap();
        let (ids_v, goff_v) = c.views(&ids, &goff);
        let gidx_h: Vec<i32> = (0..ng as i32).collect();
        let gidx_d = k.stream.clone_htod(&gidx_h).unwrap();
        let (gidx_p, _g1) = gidx_d.device_ptr(&k.stream);
        let ng_d = k.stream.clone_htod(&[ng as i32]).unwrap();
        let (ng_p, _g2) = ng_d.device_ptr(&k.stream);
        let x = k.stream.clone_htod(&make_x(c.s, kk)).unwrap();
        let mut y_exact = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
        let mut y_static = k.stream.alloc_zeros::<bf16>(c.s * n).unwrap();
        let bytes = ng as f64 * n as f64 * kk as f64 * 3.0 / 8.0;
        let dt_e = time_it(&k, 40, || {
            k.gemv_q3g128_slotsm_t2_view_gidx(
                &c.slots, &ids_v, &x, &goff_v, &mut y_exact, ng, n, kk, kk as i32, kk, blob, 0,
                gu_scale, 1, 0, gidx_p, false,
            )
            .unwrap();
        });
        let dt_s = time_it(&k, 40, || {
            k.gemv_q3g128_slotsm_t2_ng_view_gidx(
                &c.slots, &ids_v, &x, &goff_v, &mut y_static, c.s, n, kk, kk as i32, kk, blob, 0,
                gu_scale, gidx_p, ng_p,
            )
            .unwrap();
        });
        let a: Vec<bf16> = k.stream.clone_dtoh(&y_exact).unwrap();
        let b: Vec<bf16> = k.stream.clone_dtoh(&y_static).unwrap();
        let nd = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        report(&format!("{tag} 精确网格（e_uniq=ng）"), bytes, dt_e, "");
        report(&format!("{tag} 静态网格（e_max=s + 设备上界）"), bytes, dt_s, "");
        eprintln!(
            "[q3-bw]   ⇒ {tag}：逐位不同 {nd} / {} ｜速度比 {:.3}×",
            a.len(),
            dt_e / dt_s
        );
        assert_eq!(nd, 0, "{tag}: 静态网格与精确网格不逐位相同");
        assert!(
            dt_s < dt_e * tol,
            "{tag}: 静态网格慢过 {:.0}%（{dt_s:.4} vs {dt_e:.4}）",
            (tol - 1.0) * 100.0
        );
    }
}
