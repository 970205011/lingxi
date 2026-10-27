//! **QSA 的 idxprep / idxsel / attn 三段：先量分母**（2026-10-24）。
//!
//! 背景：`LINGXI_ATTN_SYNC=1` 层均（本模型 6 层/轮/GPU、t=6、len≈118、nblocks=29）：
//! `idxprep 31~33` / `idxsel 47~48` / `attn 63~65` µs，合计占 QSA 层 attn 段的 45%。
//! 这三个以前都只被"段均"看过，**没有同形分母**。本用例照 （GDN 递推）的拆法：
//!
//! - `attn`（`tests/../cuda/qsa_probe_exp.cu`，grid=(6,6)=36 block × 256 线程、NH=4）：
//! 0 prod 复刻（自检）/ 1 同形纯读 K+V / 2 只读 K / 3 只读 V / 4 q 常量 / 5 K/V 常量 /
//! 6 只做第一遍 / 7 只做第二遍 / 8/9 候选（第二遍按选块重排、4 值预取）/ 10 空核 /
//! 11~14 候选在 NH=2/1 下的并行度（block 数 72/144）。
//! - `idxprep`：两发 bf16 瘦长 GEMM（640×2560、128×2560，m=6）+ norm + rope。
//! 分母直接用 的 row 体探针（`bf16_skinny_probe` 9/10/11/18/12/19）。
//! - `idxsel`：scores（174 活跃线程）+ rank + sel_build + 池化链；打分另有 6 个探针。
//!
//! 口径：`--test-threads=1`；同进程**预热** + `best_of3` + 同进程内交错。
//!
//! `bash
//! export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib   # cublas/nvrtc 不在默认路径时按需追加
//! cargo test --release -p lingxi-kernels --features gpu --test t17_qsa_bw \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

// 线上形状（golden meta：[`artifacts/golden_f32/meta.json`]）
const T: usize = 6;
const HEADS: usize = 24;
const KVH: usize = 2;
const HD: usize = 256;
const IDX_HEADS: usize = 4;
const IDX_HD: usize = 128;
const COMPRESS: usize = 4;
const BUDGET: usize = 2048;
const HIDDEN: usize = 2560;
const IDX_QDIM: usize = IDX_HEADS * IDX_HD; // 512
const TOPK: usize = BUDGET / COMPRESS; // 512
const LEN: usize = 116; // 线上 len≈116~118

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16777216.0 - 0.5) * scale
        })
        .collect()
}

fn tobf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|x| bf16::from_f32(*x)).collect()
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
        "[qsa-bw] {tag:<44}｜{:8.2} µs｜{:7.1} GB/s｜流量 {:6.2} MB",
        dt * 1e6,
        if bytes > 0.0 { bytes / dt / 1e9 } else { 0.0 },
        bytes / 1e6
    );
}

struct Attn {
    qn: CudaSlice<bf16>,
    kn: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    blk_tab: CudaSlice<i32>,
    sel: CudaSlice<i32>,
    sel_cnt: CudaSlice<i32>,
    out: CudaSlice<bf16>,
    nblocks: usize,
    /// `vis_base`：线上 = `len`；这里取 `nblocks*compress`（⇒ vis = tail0+1+r，尾块 1~6 个）
    vis_base: usize,
    /// 每个 block（行×头组）的 token 数 = vis
    total: Vec<usize>,
}

fn attn_bufs(k: &Kernels, nblocks: usize) -> Attn {
    let cap = nblocks * COMPRESS + 64;
    let qn = k.stream.clone_htod(&tobf16(&rnd(T * HEADS * HD, 21, 0.3))).unwrap();
    let kn = k.stream.clone_htod(&tobf16(&rnd(cap * KVH * HD, 22, 0.3))).unwrap();
    let v = k.stream.clone_htod(&tobf16(&rnd(cap * KVH * HD, 23, 0.3))).unwrap();
    let blk_tab = k.stream.clone_htod(&(0..(nblocks + 16) as i32).collect::<Vec<_>>()).unwrap();
    let selcnt = nblocks.min(TOPK);
    let mut sel = vec![0i32; T * TOPK];
    for r in 0..T {
        for b in 0..selcnt {
            sel[r * TOPK + b] = b as i32;
        }
    }
    let sel = k.stream.clone_htod(&sel).unwrap();
    let sel_cnt = k.stream.clone_htod(&vec![selcnt as i32; T]).unwrap();
    let out = k.stream.alloc_zeros::<bf16>(T * HEADS * HD).unwrap();
    // 生产核里 total = nsel*ctok + ntail，而 ntail = vis - tail0 ⇒ total == vis
    let vis_base = nblocks * COMPRESS;
    let total = (0..T).map(|r| vis_base + r + 1).collect();
    Attn { qn, kn, v, blk_tab, sel, sel_cnt, out, nblocks, vis_base, total }
}

fn attn_bytes(a: &Attn, nh: usize) -> f64 {
    let groups = (HEADS / KVH / nh) * KVH; // 每个 query 行的 block 数
    let toks: usize = a.total.iter().sum(); // 已对 6 行求和
    (groups * toks * HD * 2) as f64 // 单侧（K 或 V）
}

fn run_attn_probe(k: &Kernels, a: &mut Attn, which: usize) {
    let scale = 1.0f32 / (HD as f32).sqrt();
    k.qsa_attn_probe(
        which, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
        &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
        T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks, COMPRESS.trailing_zeros() as usize, TOPK,
    )
    .unwrap();
}

fn run_attn_prod(k: &Kernels, a: &mut Attn) {
    let scale = 1.0f32 / (HD as f32).sqrt();
    k.attn_gather_bf16(
        &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
        &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
        T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks, COMPRESS.trailing_zeros() as usize, TOPK, 4,
    )
    .unwrap();
}

#[test]
fn t17_attn_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut a = attn_bufs(&k, 29);
    for _ in 0..800 {
        run_attn_prod(&k, &mut a);
    }
    k.stream.synchronize().unwrap();

    let kb = attn_bytes(&a, 4);
    eprintln!(
        "[qsa-bw] attn 形状：grid=(6,6)=36 block×256 线程、NH=4；每 block token={:?}；\
         K+V 各 {:.2} MB（NH=4）",
        a.total,
        kb / 1e6
    );
    let iters = 300;
    let dt_prod = best_of3(&k, iters, || run_attn_prod(&k, &mut a));
    report("0 现状 attn_gather_bf16（NH=4）", 2.0 * kb, dt_prod);
    let dt = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 0));
    report("1 prod 复刻（自检）", 2.0 * kb, dt);
    let dt_empty = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 10));
    report("2 空核（同 grid 发射地板）", 0.0, dt_empty);
    let dt_rkv = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 1));
    report("3 同形纯读 K+V", 2.0 * kb, dt_rkv);
    let dt_rk = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 2));
    report("4 只读 K（第一遍地址流）", kb, dt_rk);
    let dt_rv = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 3));
    report("5 只读 V（第二遍地址流）", kb, dt_rv);
    let dt_cq = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 4));
    report("6 q 常量（不读 q）", 2.0 * kb, dt_cq);
    let dt_ckv = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 5));
    report("7 K/V 常量（不读 K/V ⇒ ALU 地板）", 0.0, dt_ckv);
    let dt_p1 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 6));
    report("8 只做第一遍（点积+max）", kb, dt_p1);
    let dt_p2 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 7));
    report("9 只做第二遍（V 加权和）", kb, dt_p2);
    let dt_v2 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 8));
    report("10 候选 v2（第二遍按选块重排）", 2.0 * kb, dt_v2);
    let dt_v2pf = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 9));
    report("11 候选 v2pf（v2 + 4 值预取）", 2.0 * kb, dt_v2pf);
    let dt_p2 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 11));
    report("12 候选 NH=2（72 block）", 2.0 * kb, dt_p2);
    let dt_p1 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 12));
    report("13 候选 NH=1（144 block）", 2.0 * kb, dt_p1);
    let dt_p2 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 13));
    report("14 候选 v2pf + NH=2", 2.0 * kb, dt_p2);
    let dt_p1 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 14));
    report("15 候选 v2pf + NH=1", 2.0 * attn_bytes(&a, 1), dt_p1);
    let kb1 = attn_bytes(&a, 1);
    let dt_empty1 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 16));
    report("16 空核（NH=1 的 144 block 栅格）", 0.0, dt_empty1);
    let dt_rkv2 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 15));
    report("17 同形纯读 K+V（v2 地址流，NH=4）", 2.0 * kb, dt_rkv2);
    let dt_rkv2_1 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 17));
    report("18 同形纯读 K+V（v2 地址流，NH=1）", 2.0 * kb1, dt_rkv2_1);

    eprintln!(
        "[qsa-bw] 结论账：现状 {:.1} µs = 同形纯读 K+V 的 {:.0}%｜空核地板的 {:.0}%｜\
         ALU 地板(不读 K/V) 的 {:.0}%",
        dt_prod * 1e6,
        100.0 * dt_rkv / dt_prod,
        100.0 * dt_empty / dt_prod,
        100.0 * dt_ckv / dt_prod
    );
}

/// **上下文扫描**：NH=1/2/4 的选择必须按 nblocks 定（NH=1 的 K/V 流量 ×4，
/// 只在"延迟受限 + K/V 全在 L2"的短上下文划算）。
#[test]
fn t17_attn_sweep() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut a0 = attn_bufs(&k, 29);
    for _ in 0..800 {
        run_attn_prod(&k, &mut a0);
    }
    k.stream.synchronize().unwrap();
    eprintln!("[qsa-sweep] nblocks（len=4×nblocks）｜prod(NH4) ｜v2pf NH4 ｜v2pf NH2 ｜v2pf NH1");
    for &nb in &[29usize, 512, 1024, 2048, 4096] {
        let mut a = attn_bufs(&k, nb);
        let iters = if nb >= 1024 { 40 } else { 200 };
        let p = best_of3(&k, iters, || run_attn_prod(&k, &mut a));
        let h4 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 9));
        let h2 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 13));
        let h1 = best_of3(&k, iters, || run_attn_probe(&k, &mut a, 14));
        eprintln!(
            "[qsa-sweep] nb={nb:<5} ｜{:.1}｜{:.1}｜{:.1}｜{:.1} µs",
            p * 1e6, h4 * 1e6, h2 * 1e6, h1 * 1e6
        );
    }
}

/// 逐位自检：候选 v2/v2pf（含 NH=2/1）必须与生产核**逐位相同**。
#[test]
fn t17_attn_bits() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    for &nb in &[7usize, 29, 130] {
        let mut a = attn_bufs(&k, nb);
        run_attn_prod(&k, &mut a);
        k.stream.synchronize().unwrap();
        let r: Vec<bf16> = k.stream.clone_dtoh(&a.out).unwrap();
        for which in [0usize, 8, 9, 11, 12, 13, 14] {
            a.out = k.stream.alloc_zeros::<bf16>(T * HEADS * HD).unwrap();
            run_attn_probe(&k, &mut a, which);
            k.stream.synchronize().unwrap();
            let o: Vec<bf16> = k.stream.clone_dtoh(&a.out).unwrap();
            let nd = r.iter().zip(o.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            eprintln!("[qsa-bits] nblocks={nb:<4} which={which:<3}：{nd} / {} 位不同", r.len());
            assert_eq!(nd, 0, "nblocks={nb} which={which} 与生产核不逐位相同");
        }
    }
}

// ─────────────────────────── idxprep ───────────────────────────

/// bf16 瘦长 GEMM（row 体）的分母：两发投影 + norm + rope。
#[test]
fn t17_idxprep_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let x = k.stream.clone_htod(&tobf16(&rnd(T * HIDDEN, 31, 0.2))).unwrap();
    let wq = k.stream.clone_htod(&tobf16(&rnd(IDX_QDIM * HIDDEN, 32, 0.1))).unwrap();
    let wk = k.stream.clone_htod(&tobf16(&rnd(IDX_HD * HIDDEN, 33, 0.1))).unwrap();
    let mut iq = k.stream.alloc_zeros::<bf16>(T * IDX_QDIM).unwrap();
    let mut rk = k.stream.alloc_zeros::<bf16>(T * IDX_HD).unwrap();
    let gam = k.stream.clone_htod(&tobf16(&vec![1.0f32; IDX_HD])).unwrap();
    let mut iqn = k.stream.alloc_zeros::<bf16>(T * IDX_QDIM).unwrap();
    let cos = k.stream.clone_htod(&tobf16(&rnd(4096 * 32, 34, 1.0))).unwrap();
    let sin = k.stream.clone_htod(&tobf16(&rnd(4096 * 32, 35, 1.0))).unwrap();

    let (wqp, _a) = wq.device_ptr(&k.stream);
    let (wkp, _b) = wk.device_ptr(&k.stream);
    let (xp, _c) = x.device_ptr(&k.stream);
    let (iqp, _d) = iq.device_ptr(&k.stream);
    let (rkp, _e) = rk.device_ptr(&k.stream);

    let run_pair = |k: &Kernels| {
        k.gemm_skinny_row_bf16_ptr(wqp, xp, iqp, IDX_QDIM, HIDDEN, T, 0, 0.0, 0.0).unwrap();
        k.gemm_skinny_row_bf16_ptr(wkp, xp, rkp, IDX_HD, HIDDEN, T, 0, 0.0, 0.0).unwrap();
    };
    let mut run_tail = |k: &Kernels| {
        k.grouped_rmsnorm_bf16(&iq, &gam, &mut iqn, (T * IDX_HEADS) as u32, IDX_HD, IDX_HD, 1e-6)
            .unwrap();
        k.rope_partial_inplace_bf16(&mut iqn, &cos, &sin, T, IDX_HEADS, IDX_HD, 64, LEN).unwrap();
    };
    for _ in 0..600 {
        run_pair(&k);
        run_tail(&k);
    }
    k.stream.synchronize().unwrap();

    let bq = (IDX_QDIM * HIDDEN * 2) as f64;
    let bk = (IDX_HD * HIDDEN * 2) as f64;
    let iters = 500;

    let dt_q = best_of3(&k, iters, || {
        k.gemm_skinny_row_bf16_ptr(wqp, xp, iqp, IDX_QDIM, HIDDEN, T, 0, 0.0, 0.0).unwrap();
    });
    report("idp 0 现状 proj iq 640×2560 m6", bq, dt_q);
    let dt_k = best_of3(&k, iters, || {
        k.gemm_skinny_row_bf16_ptr(wkp, xp, rkp, IDX_HD, HIDDEN, T, 0, 0.0, 0.0).unwrap();
    });
    report("idp 1 现状 proj k 128×2560 m6", bk, dt_k);
    let dt_pair = best_of3(&k, iters, || run_pair(&k));
    report("idp 2 现状两发合计", bq + bk, dt_pair);
    let dt_tail = best_of3(&k, iters, || run_tail(&k));
    report("idp 3 现状 norm+rope 两发", 0.0, dt_tail);
    let dt_all = best_of3(&k, iters, || {
        run_pair(&k);
        run_tail(&k);
    });
    report("idp 4 现状整段（4 发）", bq + bk, dt_all);

    // 的 row 体探针（同一份生产内层，只换取数结构）：n/k/m 可任意
    for &(tag, n, bytes, wp) in &[("iq(640)", IDX_QDIM, bq, wqp), ("k(128)", IDX_HD, bk, wkp)] {
        let d9 = best_of3(&k, iters, || {
            k.bf16_skinny_probe(9, wp, xp, iqp, n, HIDDEN, T).unwrap();
        });
        report(&format!("idp 5 row 纯读16B {tag}"), bytes, d9);
        let d10 = best_of3(&k, iters, || {
            k.bf16_skinny_probe(10, wp, xp, iqp, n, HIDDEN, T).unwrap();
        });
        report(&format!("idp 6 row ALU（x 常量）{tag}"), bytes, d10);
        let d11 = best_of3(&k, iters, || {
            k.bf16_skinny_probe(11, wp, xp, iqp, n, HIDDEN, T).unwrap();
        });
        report(&format!("idp 7 row 生产复刻 {tag}"), bytes, d11);
        let d18 = best_of3(&k, iters, || {
            k.bf16_skinny_probe(18, wp, xp, iqp, n, HIDDEN, T).unwrap();
        });
        report(&format!("idp 8 row 只读 x {tag}"), 0.0, d18);
        let d19 = best_of3(&k, iters, || {
            k.bf16_skinny_probe(19, wp, xp, iqp, n, HIDDEN, T).unwrap();
        });
        report(&format!("idp 9 row R=4 {tag}"), bytes, d19);
    }
    eprintln!(
        "[qsa-bw] 结论账：整段 {:.1} µs = 权重 payload {:.2} MB ⇒ {:.0} GB/s；\
         两发 GEMM {:.1}、norm+rope {:.1}、其余 {:.1} µs",
        dt_all * 1e6,
        (bq + bk) / 1e6,
        (bq + bk) / dt_all / 1e9,
        dt_pair * 1e6,
        dt_tail * 1e6,
        (dt_all - dt_pair - dt_tail) * 1e6
    );
}

// ─────────────────────────── idxsel ───────────────────────────

struct Sel {
    iq: CudaSlice<bf16>,
    pooled: CudaSlice<bf16>,
    scores: CudaSlice<f32>,
    keep: CudaSlice<u8>,
    sel: CudaSlice<i32>,
    scnt: CudaSlice<i32>,
    nblocks: usize,
}

fn sel_bufs(k: &Kernels, nblocks: usize) -> Sel {
    Sel {
        iq: k.stream.clone_htod(&tobf16(&rnd(T * IDX_QDIM, 41, 0.3))).unwrap(),
        pooled: k.stream.clone_htod(&tobf16(&rnd((nblocks + 16) * IDX_HD, 42, 0.3))).unwrap(),
        scores: k.stream.alloc_zeros::<f32>(T * (nblocks + 16)).unwrap(),
        keep: k.stream.alloc_zeros::<u8>(T * (nblocks + 16)).unwrap(),
        sel: k.stream.alloc_zeros::<i32>(T * TOPK).unwrap(),
        scnt: k.stream.alloc_zeros::<i32>(T).unwrap(),
        nblocks,
    }
}

#[test]
fn t17_idxsel_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut s = sel_bufs(&k, 29);
    let sq = (IDX_HD as f32).sqrt();
    let run_prod3 = |k: &Kernels, s: &mut Sel| {
        k.qsa_indexer_scores_bf16(&s.iq, &s.pooled, &mut s.scores, T, s.nblocks, IDX_HEADS, IDX_HD, sq)
            .unwrap();
        k.qsa_indexer_rank_f32(&s.scores, &mut s.keep, T, s.nblocks, TOPK).unwrap();
        k.qsa_sel_build(&mut s.keep, &mut s.sel, &mut s.scnt, T, s.nblocks, TOPK, false).unwrap();
    };
    for _ in 0..800 {
        run_prod3(&k, &mut s);
    }
    k.stream.synchronize().unwrap();
    let iters = 500;
    // 读字节：q 6×512×2 + pooled 29×128×2 + scores 6×29×4 ×(rank 读 + sel 读)
    let rb = (T * IDX_QDIM * 2 + 29 * IDX_HD * 2) as f64;

    let dt = best_of3(&k, iters, || {
        k.qsa_indexer_scores_bf16(
            &s.iq, &s.pooled, &mut s.scores, T, s.nblocks, IDX_HEADS, IDX_HD, sq,
        )
        .unwrap();
    });
    report("ixs 0 现状 scores（grid 6 块×256，174 活跃线程）", rb, dt);
    for (which, tag) in [
        (0usize, "1 scores prod 复刻（自检）"),
        (1, "2 scores 同形纯读 q+pooled"),
        (2, "3 scores q 常量（只读权重）"),
        (3, "4 scores pooled 常量（只读 q）"),
        (4, "5 候选 heads 展开 4 条链"),
        (5, "6 候选 heads 展开 + 8 宽取数"),
    ] {
        let d = best_of3(&k, iters, || {
            k.qsa_idx_scores_probe(
                which, &s.iq.slice(..), &s.pooled.slice(..), &mut s.scores.slice_mut(..),
                T, s.nblocks, IDX_HEADS, IDX_HD, sq,
            )
            .unwrap();
        });
        report(&format!("ixs {tag}"), rb, d);
    }
    let dt_rank = best_of3(&k, iters, || {
        k.qsa_indexer_rank_f32(&s.scores, &mut s.keep, T, s.nblocks, TOPK).unwrap();
    });
    report("ixs 7 现状 rank（O(nblocks²)）", (T * 29 * 4) as f64, dt_rank);
    let dt_build = best_of3(&k, iters, || {
        k.qsa_sel_build(&mut s.keep, &mut s.sel, &mut s.scnt, T, s.nblocks, TOPK, false).unwrap();
    });
    report("ixs 8 现状 sel_build", (T * 29) as f64, dt_build);
    let dt3 = best_of3(&k, iters, || run_prod3(&k, &mut s));
    report("ixs 9 现状三发合计", rb, dt3);

    // 池化链（每 4 个 token 一次；nb=1）
    let rk = k.stream.clone_htod(&tobf16(&rnd((29 * COMPRESS + 16) * IDX_HD, 43, 0.3))).unwrap();
    let mut pool = k.stream.alloc_zeros::<bf16>(32 * IDX_HD).unwrap();
    let mut pool2 = k.stream.alloc_zeros::<bf16>(32 * IDX_HD).unwrap();
    let gam = k.stream.clone_htod(&tobf16(&vec![1.0f32; IDX_HD])).unwrap();
    let cos = k.stream.clone_htod(&tobf16(&rnd(4096 * 32, 44, 1.0))).unwrap();
    let sin = k.stream.clone_htod(&tobf16(&rnd(4096 * 32, 45, 1.0))).unwrap();
    let mut run_pool = |k: &Kernels| {
        k.qsa_pool_blocks_bf16(&rk, &mut pool, 28, 1, COMPRESS, IDX_HD).unwrap();
        k.grouped_rmsnorm_bf16(&pool, &gam, &mut pool2, 1, IDX_HD, IDX_HD, 1e-6).unwrap();
        k.rope_partial_stride_bf16(&mut pool2, &cos, &sin, 1, IDX_HD, 64, 112, COMPRESS).unwrap();
        k.stream.memcpy_dtod(&pool2.slice(0..IDX_HD), &mut pool.slice_mut(0..IDX_HD)).unwrap();
    };
    let dt_pool = best_of3(&k, iters, || run_pool(&k));
    report("ixs 10 池化链（每 4 token 一次，4 发）", 0.0, dt_pool);
    eprintln!(
        "[qsa-bw] 结论账（nblocks=29）：三发 {:.1} µs = scores {:.1} + rank {:.1} + sel_build {:.1}；\
         池化链另有 {:.1} µs（len118 时约 30 次/层 ⇒ 摊 {:.1} µs/层）",
        dt3 * 1e6,
        dt * 1e6,
        dt_rank * 1e6,
        dt_build * 1e6,
        dt_pool * 1e6,
        dt_pool * 1e6 * 24.0 / 6.0
    );
}

/// **合发核的 µs 账**（落地的 4 处小发射合并：前后都是生产入口）。
#[test]
fn t17_fuse_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let iters = 500;
    let warm = |f: &mut dyn FnMut()| {
        for _ in 0..600 {
            f();
        }
        k.stream.synchronize().unwrap();
    };

    // ── 1) 池化链：4 发 vs 1 发（nb=1，线上 decode 每次 1 块）──
    let rk = k.stream.clone_htod(&tobf16(&rnd(29 * COMPRESS * IDX_HD, 51, 0.3))).unwrap();
    let gam = k.stream.clone_htod(&tobf16(&vec![1.0f32; IDX_HD])).unwrap();
    let cos = k.stream.clone_htod(&tobf16(&rnd(4096 * 64, 52, 1.0))).unwrap();
    let sin = k.stream.clone_htod(&tobf16(&rnd(4096 * 64, 53, 1.0))).unwrap();
    let mut pool = k.stream.alloc_zeros::<bf16>(32 * IDX_HD).unwrap();
    let mut pool2 = k.stream.alloc_zeros::<bf16>(32 * IDX_HD).unwrap();
    let mut pooled = k.stream.alloc_zeros::<bf16>(32 * IDX_HD).unwrap();
    let mut old4 = || {
        k.qsa_pool_blocks_bf16(&rk, &mut pool, 28, 1, COMPRESS, IDX_HD).unwrap();
        k.grouped_rmsnorm_bf16(&pool, &gam, &mut pool2, 1, IDX_HD, IDX_HD, 1e-6).unwrap();
        k.rope_partial_stride_bf16(&mut pool2, &cos, &sin, 1, IDX_HD, 64, 112, COMPRESS).unwrap();
        k.stream.memcpy_dtod(&pool2.slice(0..IDX_HD), &mut pool.slice_mut(0..IDX_HD)).unwrap();
    };
    warm(&mut old4);
    let d_old = best_of3(&k, iters, || old4());
    let mut new1 = || {
        k.qsa_pool_norm_rope_bf16(
            &rk, &mut pooled, &gam, &cos, &sin, 28, 1, COMPRESS, IDX_HD, 64, 1e-6,
        )
        .unwrap();
    };
    warm(&mut new1);
    let d_new = best_of3(&k, iters, || new1());
    report("fz 0 池化链 4 发（旧）", 0.0, d_old);
    report("fz 1 池化链 1 发（新）", 0.0, d_new);

    // ── 2) indexer q/k 投影：2 发 vs 1 发 ──
    let x = k.stream.clone_htod(&tobf16(&rnd(T * HIDDEN, 54, 0.2))).unwrap();
    let w1 = k.stream.clone_htod(&tobf16(&rnd(IDX_QDIM * HIDDEN, 55, 0.1))).unwrap();
    let w2 = k.stream.clone_htod(&tobf16(&rnd(IDX_HD * HIDDEN, 56, 0.1))).unwrap();
    let mut y1 = k.stream.alloc_zeros::<bf16>(T * IDX_QDIM).unwrap();
    let mut y2 = k.stream.alloc_zeros::<bf16>(T * IDX_HD).unwrap();
    let (wp1, _a) = w1.device_ptr(&k.stream);
    let (wp2, _b) = w2.device_ptr(&k.stream);
    let (xp, _c) = x.device_ptr(&k.stream);
    let (yp1, _d) = y1.device_ptr(&k.stream);
    let (yp2, _e) = y2.device_ptr(&k.stream);
    let mut old2 = || {
        k.gemm_skinny_row_bf16_ptr(wp1, xp, yp1, IDX_QDIM, HIDDEN, T, 0, 0.0, 0.0).unwrap();
        k.gemm_skinny_row_bf16_ptr(wp2, xp, yp2, IDX_HD, HIDDEN, T, 0, 0.0, 0.0).unwrap();
    };
    warm(&mut old2);
    let d2 = best_of3(&k, iters, || old2());
    let mut newp = || {
        k.gemm_bf16_row_pair_ptr(wp1, yp1, IDX_QDIM, wp2, yp2, IDX_HD, xp, HIDDEN, T).unwrap();
    };
    warm(&mut newp);
    let dp = best_of3(&k, iters, || newp());
    report("fz 2 q/k 投影 2 发（旧）", 0.0, d2);
    report("fz 3 q/k 投影 1 发（新）", 0.0, dp);

    // ── 3) indexer norm+rope：2 发 vs 1 发 ──
    let mut iqn = k.stream.alloc_zeros::<bf16>(T * IDX_QDIM).unwrap();
    let mut old2b = || {
        k.grouped_rmsnorm_bf16(&y1, &gam, &mut iqn, (T * IDX_HEADS) as u32, IDX_HD, IDX_HD, 1e-6)
            .unwrap();
        k.rope_partial_inplace_bf16(&mut iqn, &cos, &sin, T, IDX_HEADS, IDX_HD, 64, LEN).unwrap();
    };
    warm(&mut old2b);
    let d2b = best_of3(&k, iters, || old2b());
    let mut nr = || {
        k.rmsnorm_rope_bf16(
            &y1, &gam, &cos, &sin, &mut iqn, T, IDX_HEADS, IDX_HD, 64, LEN, 1e-6,
        )
        .unwrap();
    };
    warm(&mut nr);
    let dnr = best_of3(&k, iters, || nr());
    report("fz 4 norm+rope 2 发（旧）", 0.0, d2b);
    report("fz 5 norm+rope 1 发（新）", 0.0, dnr);

    // ── 4) 选块：scores+rank+sel_build 三发 vs sel_build(all) 一发 ──
    let mut s = sel_bufs(&k, 29);
    let sq = (IDX_HD as f32).sqrt();
    let mut old3 = || {
        k.qsa_indexer_scores_bf16(&s.iq, &s.pooled, &mut s.scores, T, 29, IDX_HEADS, IDX_HD, sq)
            .unwrap();
        k.qsa_indexer_rank_f32(&s.scores, &mut s.keep, T, 29, TOPK).unwrap();
        k.qsa_sel_build(&mut s.keep, &mut s.sel, &mut s.scnt, T, 29, TOPK, false).unwrap();
    };
    warm(&mut old3);
    let d3 = best_of3(&k, iters, || old3());
    let mut newk = || {
        k.qsa_sel_build(&mut s.keep, &mut s.sel, &mut s.scnt, T, 29, TOPK, true).unwrap();
    };
    warm(&mut newk);
    let dk = best_of3(&k, iters, || newk());
    report("fz 6 选块 3 发（旧）", 0.0, d3);
    report("fz 7 选块 1 发（新，keep 全 1 档）", 0.0, dk);

    eprintln!(
        "[qsa-bw] 合发账：池化 {:.1}→{:.1}｜qk 投影 {:.1}→{:.1}｜norm+rope {:.1}→{:.1}｜\
         选块 {:.1}→{:.1} µs（合计 {:.1}→{:.1}）",
        d_old * 1e6, d_new * 1e6, d2 * 1e6, dp * 1e6, d2b * 1e6, dnr * 1e6, d3 * 1e6, dk * 1e6,
        (d_old + d2 + d2b + d3) * 1e6,
        (d_new + dp + dnr + dk) * 1e6
    );
}

/// **合发核逐位自检**：池化链（含 `b0>0`、`nb>1`）与 norm+rope 两发必须与旧路径逐位相同。
/// （线上短上下文 nblocks ≤ budget ⇒ keep 恒全 1 ⇒ 池化结果**不参与**输出，
/// 端到端 md5 根本照不到这里的错；必须单独对拍。2026-10-24 就是这么抓到 `lb*hd` vs `b*hd` 的。）
#[test]
fn t17_fuse_bits() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let nb = 3usize;
    let b0 = 7usize;
    let rows = (b0 + nb) * COMPRESS;
    let rk = k.stream.clone_htod(&tobf16(&rnd(rows * IDX_HD, 61, 0.3))).unwrap();
    let gam = k.stream.clone_htod(&tobf16(&rnd(IDX_HD, 62, 0.2))).unwrap();
    let cos = k.stream.clone_htod(&tobf16(&rnd(4096 * 64, 63, 1.0))).unwrap();
    let sin = k.stream.clone_htod(&tobf16(&rnd(4096 * 64, 64, 1.0))).unwrap();
    // 旧路：pool → rmsnorm → rope → memcpy（只改 [b0, b0+nb) 段）
    let mut a = k.stream.alloc_zeros::<bf16>(rows * IDX_HD).unwrap();
    let mut t1 = k.stream.alloc_zeros::<bf16>(nb * IDX_HD).unwrap();
    let mut t2 = k.stream.alloc_zeros::<bf16>(nb * IDX_HD).unwrap();
    k.qsa_pool_blocks_bf16(&rk, &mut t1, b0, nb, COMPRESS, IDX_HD).unwrap();
    k.grouped_rmsnorm_bf16(&t1, &gam, &mut t2, nb as u32, IDX_HD, IDX_HD, 1e-6).unwrap();
    k.rope_partial_stride_bf16(&mut t2, &cos, &sin, nb, IDX_HD, 64, b0 * COMPRESS, COMPRESS).unwrap();
    k.stream
        .memcpy_dtod(&t2.slice(..), &mut a.slice_mut(b0 * IDX_HD..(b0 + nb) * IDX_HD))
        .unwrap();
    // 新路：一发
    let mut b = k.stream.alloc_zeros::<bf16>(rows * IDX_HD).unwrap();
    k.qsa_pool_norm_rope_bf16(&rk, &mut b, &gam, &cos, &sin, b0, nb, COMPRESS, IDX_HD, 64, 1e-6)
        .unwrap();
    k.stream.synchronize().unwrap();
    let av: Vec<bf16> = k.stream.clone_dtoh(&a).unwrap();
    let bv: Vec<bf16> = k.stream.clone_dtoh(&b).unwrap();
    let nd = av.iter().zip(bv.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    eprintln!(
        "[qsa-bits] 池化合发 b0={b0} nb={nb}：{nd} / {} 位不同（只比 [{}..{}) 段）",
        av.len(),
        b0 * IDX_HD,
        (b0 + nb) * IDX_HD
    );
    assert_eq!(nd, 0, "池化合发与四发不逐位相同");

    // norm+rope 合发 vs 两发（q 形状：t=6 / heads=4 / hd=128 / rot=64）
    let x = k.stream.clone_htod(&tobf16(&rnd(T * IDX_QDIM, 65, 0.3))).unwrap();
    let mut y_old = k.stream.alloc_zeros::<bf16>(T * IDX_QDIM).unwrap();
    let mut y_new = k.stream.alloc_zeros::<bf16>(T * IDX_QDIM).unwrap();
    k.grouped_rmsnorm_bf16(&x, &gam, &mut y_old, (T * IDX_HEADS) as u32, IDX_HD, IDX_HD, 1e-6)
        .unwrap();
    k.rope_partial_inplace_bf16(&mut y_old, &cos, &sin, T, IDX_HEADS, IDX_HD, 64, LEN).unwrap();
    k.rmsnorm_rope_bf16(&x, &gam, &cos, &sin, &mut y_new, T, IDX_HEADS, IDX_HD, 64, LEN, 1e-6)
        .unwrap();
    k.stream.synchronize().unwrap();
    let av: Vec<bf16> = k.stream.clone_dtoh(&y_old).unwrap();
    let bv: Vec<bf16> = k.stream.clone_dtoh(&y_new).unwrap();
    let nd = av.iter().zip(bv.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    eprintln!("[qsa-bits] norm+rope 合发：{nd} / {} 位不同", av.len());
    assert_eq!(nd, 0, "norm+rope 合发与两发不逐位相同");
}

/// 逐位自检：打分候选（4/5）必须与生产核逐位相同。
#[test]
fn t17_idxsel_bits() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    for &nb in &[29usize, 130] {
        let mut s = sel_bufs(&k, nb);
        let sq = (IDX_HD as f32).sqrt();
        k.qsa_indexer_scores_bf16(&s.iq, &s.pooled, &mut s.scores, T, nb, IDX_HEADS, IDX_HD, sq)
            .unwrap();
        k.stream.synchronize().unwrap();
        let r: Vec<f32> = k.stream.clone_dtoh(&s.scores).unwrap();
        for which in [0usize, 4, 5] {
            let mut sc = k.stream.alloc_zeros::<f32>(T * (nb + 16)).unwrap();
            k.qsa_idx_scores_probe(
                which, &s.iq.slice(..), &s.pooled.slice(..), &mut sc.slice_mut(..),
                T, nb, IDX_HEADS, IDX_HD, sq,
            )
            .unwrap();
            k.stream.synchronize().unwrap();
            let o: Vec<f32> = k.stream.clone_dtoh(&sc).unwrap();
            let nd = r.iter().zip(o.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            eprintln!("[qsa-bits] scores nblocks={nb:<4} which={which}：{nd} / {} 位不同", r.len());
            assert_eq!(nd, 0, "scores nblocks={nb} which={which} 不逐位相同");
        }
    }
}
