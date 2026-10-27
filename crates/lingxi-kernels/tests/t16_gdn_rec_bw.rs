//! **GDN 递推 `gdn_recurrent_bf16`（t=6）的分母表 + state 常驻候选**（2026-10-24，）。
//!
//! `rec` 是 36/48 层都跑的核（`LINGXI_ATTN_SYNC=1` 层均 78.6~92.5 µs），以前被"bit-exact +
//! 行串行 + smem 更差"判为到顶，**但从没量过同形纯读/纯搬运的分母**。本用例只做这件事：
//!
//! | 探针 | 含义 | S 字节/层 |
//! |---|---|---|
//! | 0 prod | 生产核逐字复刻（自检） | 72 MiB（2R+2W × 6 token）|
//! | 1 read2 | 同形纯读两遍/ token | 36 MiB |
//! | 2 read1 | 同形纯读一遍/ token | 18 MiB |
//! | 3 copy2 | 就地 2R+2W 纯搬运（无递推） | 72 MiB |
//! | 4 copy1 | 就地 1R+1W 纯搬运（整段一遍） | 6 MiB |
//! | 5 reg | **候选**：state 全程驻留寄存器 | 6 MiB |
//! | 6 reg_nos | 候选但不碰 S（ALU/发射地板） | 0 |
//! | 7 reg_tok | 候选 + 每 token 写回一次 | 18 MiB |
//! | 8 smem | **候选**：state 驻留 smem，decay 第二遍重算 | 6 MiB |
//! | 9 smem_old | 老档（smem 2R+2W，复刻 117.7 µs 口径） | 6 MiB |
//!
//! 口径：`--test-threads=1`；同进程**预热**（首发被时钟爬坡吃掉 1.5× 是本仓踩过的坑）+
//! `best_of3`（每轮独立 N 次连发，取最好）+ 同一进程内交错。逐位自检见 `t16_gdn_rec_bits`。
//!
//! `bash
//! export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib   # cublas/nvrtc 不在默认路径时按需追加
//! cargo test --release -p lingxi-kernels --features gpu --test t16_gdn_rec_bw \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use half::bf16;
use lingxi_kernels::Kernels;

const NVH: usize = 48;
const NKH: usize = 16;
const HK: usize = 128;
const HV: usize = 128;
const T: usize = 6;

struct Buf {
    conv: CudaSlice<bf16>,
    z: CudaSlice<bf16>,
    a: CudaSlice<bf16>,
    b: CudaSlice<bf16>,
    nw: CudaSlice<bf16>,
    alog: CudaSlice<f32>,
    dtb: CudaSlice<f32>,
    s: CudaSlice<f32>,
    gated: CudaSlice<bf16>,
}

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16777216.0 - 0.5) * scale
        })
        .collect()
}

fn bufs(k: &Kernels) -> Buf {
    bufs_t(k, T)
}

fn bufs_t(k: &Kernels, tmax: usize) -> Buf {
    let conv_dim = 2 * NKH * HK + NVH * HV;
    let value_dim = NVH * HV;
    let conv: Vec<bf16> =
        rnd(tmax * conv_dim, 11, 0.25).iter().map(|v| bf16::from_f32(*v)).collect();
    let z: Vec<bf16> = rnd(tmax * value_dim, 12, 0.5).iter().map(|v| bf16::from_f32(*v)).collect();
    let a: Vec<bf16> = rnd(tmax * NVH, 13, 0.25).iter().map(|v| bf16::from_f32(*v)).collect();
    let b: Vec<bf16> = rnd(tmax * NVH, 14, 0.25).iter().map(|v| bf16::from_f32(*v)).collect();
    Buf {
        conv: k.stream.clone_htod(&conv).unwrap(),
        z: k.stream.clone_htod(&z).unwrap(),
        a: k.stream.clone_htod(&a).unwrap(),
        b: k.stream.clone_htod(&b).unwrap(),
        nw: k.stream.clone_htod(&vec![bf16::from_f32(1.0); HV]).unwrap(),
        // a_log = -10 ⇒ decay ≈ 1（state 既不衰减到 0 也不炸，整段计时多次复跑稳定）
        alog: k.stream.clone_htod(&vec![-10.0f32; NVH]).unwrap(),
        dtb: k.stream.clone_htod(&vec![0.0f32; NVH]).unwrap(),
        s: k.stream.clone_htod(&rnd(NVH * HK * HV, 15, 0.1)).unwrap(),
        gated: k.stream.alloc_zeros::<bf16>(tmax * value_dim).unwrap(),
    }
}

/// 生产核（`gdn_recurrent_bf16`，hk=hv=128、nvh=48、nkh=16、gate_act=1、n_seq=1）。
/// 注意：2026-10-24 之后它默认派发到 **state 常驻寄存器版**。
fn run_prod(k: &Kernels, b: &mut Buf) {
    run_prod_t(k, b, T)
}

fn run_prod_t(k: &Kernels, b: &mut Buf, t: usize) {
    k.gdn_recurrent_bf16(
        &b.conv, &b.z, &b.a, &b.b, &b.nw, &b.alog, &b.dtb, &mut b.s, &mut b.gated,
        t, HK, HV, NVH, NKH, 1, 0, 1,
    )
    .unwrap();
}

/// 6 个 int 的探针（prod / read / copy：带 hk、hv、nvh、nkh、gate_act）。
fn run6(k: &Kernels, f: &CudaFunction, b: &mut Buf, t: usize) {
    let cfg = LaunchConfig {
        grid_dim: (NVH as u32, 1, 1),
        block_dim: (HV as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        k.stream
            .launch_builder(f)
            .arg(&b.conv).arg(&b.z).arg(&b.a).arg(&b.b).arg(&b.nw)
            .arg(&b.alog).arg(&b.dtb).arg(&mut b.s).arg(&mut b.gated)
            .arg(&(t as i32)).arg(&(HK as i32)).arg(&(HV as i32))
            .arg(&(NVH as i32)).arg(&(NKH as i32)).arg(&1i32)
            .launch(cfg)
            .unwrap();
    }
}

/// 4 个 int 的探针（reg / smem：hk、hv 编译期固定 128）。
fn run4(k: &Kernels, f: &CudaFunction, b: &mut Buf, smem: u32, t: usize) {
    let cfg = LaunchConfig {
        grid_dim: (NVH as u32, 1, 1),
        block_dim: (HV as u32, 1, 1),
        shared_mem_bytes: smem,
    };
    unsafe {
        k.stream
            .launch_builder(f)
            .arg(&b.conv).arg(&b.z).arg(&b.a).arg(&b.b).arg(&b.nw)
            .arg(&b.alog).arg(&b.dtb).arg(&mut b.s).arg(&mut b.gated)
            .arg(&(t as i32)).arg(&(NVH as i32)).arg(&(NKH as i32)).arg(&1i32)
            .launch(cfg)
            .unwrap();
    }
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

/// 同一进程内 3 轮独立连发，取**最好**（剔除时钟爬坡/抢占离群）。
fn best_of3(k: &Kernels, iters: usize, mut f: impl FnMut()) -> f64 {
    let mut v = [0f64; 3];
    for s in v.iter_mut() {
        *s = time_it(k, iters, &mut f);
    }
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn report(tag: &str, bytes: f64, dt: f64) {
    eprintln!(
        "[rec-bw] {tag:<40}｜{:8.2} µs｜{:7.1} GB/s｜S 流量 {:6.2} MB",
        dt * 1e6,
        if bytes > 0.0 { bytes / dt / 1e9 } else { 0.0 },
        bytes / 1e6
    );
}

fn s_bytes(passes: f64, per_token: bool) -> f64 {
    let one = (NVH * HK * HV * 4) as f64; // 一次全量 S
    one * passes * if per_token { T as f64 } else { 1.0 }
}

#[test]
fn t16_gdn_rec_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut b = bufs(&k);
    // 预热：先跑够 GPU 工作量把时钟拉起来（1100 发 × ~85 µs ≈ 94 ms）
    for _ in 0..1100 {
        run_prod(&k, &mut b);
    }
    k.stream.synchronize().unwrap();

    let g = &k.gdn_rec_exp;
    let iters = 300;
    let dt_prod = best_of3(&k, iters, || run_prod(&k, &mut b));
    report("0 现状 gdn_recurrent_bf16", s_bytes(4.0, true), dt_prod);
    let dt = best_of3(&k, iters, || run6(&k, &g.prod, &mut b, T));
    report("1 prod 复刻（自检）", s_bytes(4.0, true), dt);
    let dt_read2 = best_of3(&k, iters, || run6(&k, &g.read2, &mut b, T));
    report("2 同形纯读 2 遍/token", s_bytes(2.0, true), dt_read2);
    let dt_read1 = best_of3(&k, iters, || run6(&k, &g.read1, &mut b, T));
    report("3 同形纯读 1 遍/token", s_bytes(1.0, true), dt_read1);
    let dt_copy2 = best_of3(&k, iters, || run6(&k, &g.copy2, &mut b, T));
    report("4 同形纯搬运 2R+2W/token", s_bytes(4.0, true), dt_copy2);
    let dt_copy1 = best_of3(&k, iters, || run6(&k, &g.copy1, &mut b, T));
    report("5 同形纯搬运 1R+1W（整段）", s_bytes(1.0, false), dt_copy1);

    let dt_reg = best_of3(&k, iters, || run4(&k, &g.reg, &mut b, 0, T));
    report("6 候选 reg（state 常驻寄存器）", s_bytes(1.0, false), dt_reg);
    let dt_reg_nos = best_of3(&k, iters, || run4(&k, &g.reg_nos, &mut b, 0, T));
    report("7 候选 reg 但不碰 S（ALU 地板）", 0.0, dt_reg_nos);
    let dt_reg_tok = best_of3(&k, iters, || run4(&k, &g.reg_tok, &mut b, 0, T));
    report("8 候选 reg + 每 token 写回", s_bytes(1.0, true), dt_reg_tok);
    let dt_smem = best_of3(&k, iters, || run4(&k, &g.smem, &mut b, (HK * HV * 4) as u32, T));
    report("9 候选 smem（2R+1W smem）", s_bytes(1.0, false), dt_smem);
    let dt_smem_old = best_of3(&k, iters, || run4(&k, &g.smem_old, &mut b, (HK * HV * 4) as u32, T));
    report("10 候选 smem 老档（2R+2W）", s_bytes(1.0, false), dt_smem_old);
    let dt_v2 = best_of3(&k, iters, || run4(&k, &g.v2, &mut b, 0, T));
    report("11 候选 v2（q/k smem + state 全寄存器）", s_bytes(1.0, false), dt_v2);
    let dt_hyb = best_of3(&k, iters, || run4(&k, &g.hyb, &mut b, 32 * 1024, T));
    report("12 候选 hyb（q/k smem + state 半寄存器）", s_bytes(1.0, false), dt_hyb);
    let dt_nok = best_of3(&k, iters, || run4(&k, &g.nok, &mut b, 0, T));
    report("13 消融 nok（q/k 全常量）", 0.0, dt_nok);
    let dt_v3 = best_of3(&k, iters, || run4(&k, &g.v3, &mut b, 0, T));
    report("14 候选 v3（v2 + q/k float4）", s_bytes(1.0, false), dt_v3);

    eprintln!(
        "[rec-bw] 结论账：现状 {:.1} µs = 同形 2R+2W 纯搬运的 {:.0}%｜纯读 2 遍的 {:.0}%｜\
         1R+1W 纯搬运的 {:.0}%｜reg 候选的 {:.0}%",
        dt_prod * 1e6,
        100.0 * dt_copy2 / dt_prod,
        100.0 * dt_read2 / dt_prod,
        100.0 * dt_copy1 / dt_prod,
        100.0 * dt_reg / dt_prod
    );
    eprintln!(
        "[rec-bw] 候选：reg {:.1} µs（{:.2}×）/ smem {:.1} µs（{:.2}×）/ reg_tok {:.1} µs（{:.2}×）",
        dt_reg * 1e6,
        dt_prod / dt_reg,
        dt_smem * 1e6,
        dt_prod / dt_smem,
        dt_reg_tok * 1e6,
        dt_prod / dt_reg_tok
    );
}

fn bits_eq(a: &[bf16], b: &[bf16]) -> usize {
    a.iter().zip(b.iter()).filter(|(u, v)| u.to_bits() != v.to_bits()).count()
}

fn fbits_eq(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b.iter()).filter(|(u, v)| u.to_bits() != v.to_bits()).count()
}

/// **t 扫描**：线上 t 不止 6（draft 路径是 t=1 逐 token、replay 是 t≈4、prefill 是大 t）。
/// 旧核每 token 2R+2W、新核整段 1R+1W ⇒ 新核在小 t 上省得少（摊销不变但绝对量小）。
#[test]
fn t16_gdn_rec_tsweep() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut b = bufs_t(&k, 32);
    for _ in 0..1100 {
        run_prod(&k, &mut b);
    }
    k.stream.synchronize().unwrap();
    for &t in &[1usize, 2, 4, 6, 7, 16, 32] {
        let iters = if t <= 7 { 300 } else { 60 };
        let old = best_of3(&k, iters, || run6(&k, &k.gdn_rec_exp.prod, &mut b, t));
        let new = best_of3(&k, iters, || run_prod_t(&k, &mut b, t));
        eprintln!(
            "[rec-t] t={t:<3} 旧核 {:7.2} µs（{:.2}/token）｜新核 {:7.2} µs（{:.2}/token）｜{:5.2}×",
            old * 1e6,
            old * 1e6 / t as f64,
            new * 1e6,
            new * 1e6 / t as f64,
            old / new
        );
    }
}

/// 逐位自检：生产复刻 / reg / smem 三条路径的 `gated` 与 `s_all` 必须与生产核**逐位相同**。
#[test]
fn t16_gdn_rec_bits() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    // bufs 用固定种子 ⇒ 每次调用内容一致（省掉拷贝克隆）
    let mut rb = bufs(&k);
    k.gdn_recurrent_bf16(
        &rb.conv, &rb.z, &rb.a, &rb.b, &rb.nw, &rb.alog, &rb.dtb, &mut rb.s, &mut rb.gated,
        T, HK, HV, NVH, NKH, 1, 0, 1,
    )
    .unwrap();
    k.stream.synchronize().unwrap();
    let hs: Vec<f32> = k.stream.clone_dtoh(&rb.s).unwrap();
    let hg: Vec<bf16> = k.stream.clone_dtoh(&rb.gated).unwrap();
    eprintln!("[rec-bits] 参考：s_all 前 3 = {:?}｜gated 前 3 = {:?}", &hs[..3], &hg[..3]);

    // 逐位必须相同的：生产复刻 + 两个候选（v3/v2）+ smem 老档（逐字节复刻）
    // `hyb`（半寄存器半 smem）实测**不逐位**（smem 那半的 decay 重算被 ptxas 收缩成 FMA，
    // 用 __fmul_rn 也仍未对齐，见 负结果）⇒ 只报数、不作候选。
    let cases: [(&str, &CudaFunction, u32, bool, bool); 7] = [
        ("prod 复刻", &k.gdn_rec_exp.prod, 0, true, true),
        ("v3 候选", &k.gdn_rec_exp.v3, 0, false, true),
        ("v2 候选", &k.gdn_rec_exp.v2, 0, false, true),
        ("smem 候选", &k.gdn_rec_exp.smem, (HK * HV * 4) as u32, false, true),
        ("smem 老档", &k.gdn_rec_exp.smem_old, (HK * HV * 4) as u32, false, true),
        ("reg 候选", &k.gdn_rec_exp.reg, 0, false, true),
        ("hyb 候选", &k.gdn_rec_exp.hyb, 32 * 1024, false, false),
    ];
    for (nm, f, smem, six, strict) in cases {
        let mut bb = bufs(&k);
        if six {
            run6(&k, f, &mut bb, T);
        } else {
            run4(&k, f, &mut bb, smem, T);
        }
        k.stream.synchronize().unwrap();
        let hs2: Vec<f32> = k.stream.clone_dtoh(&bb.s).unwrap();
        let hg2: Vec<bf16> = k.stream.clone_dtoh(&bb.gated).unwrap();
        let nds = fbits_eq(&hs, &hs2);
        let ndg = bits_eq(&hg, &hg2);
        eprintln!(
            "[rec-bits] {nm:<10}：s_all {nds} / {} 位不同｜gated {ndg} / {} 位不同{}",
            hs.len(),
            hg.len(),
            if strict { "" } else { "（不作候选）" }
        );
        if strict {
            assert_eq!(nds, 0, "{nm} 的 s_all 与生产核不逐位相同");
            assert_eq!(ndg, 0, "{nm} 的 gated 与生产核不逐位相同");
        }
    }
}
