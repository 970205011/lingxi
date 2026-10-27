//! `lingxi_route_topk_rows_bf16` 的**独立微基准 + 位级摘要**（2026-10-23）。
//!
//! 动机：整机 decode 里 rows=6 / E=512 / top_k=10 的这个核量到层均 ≈100 µs，
//! 而工作量只有 6 行 × 512 项选 10 —— 先把核本身单独拉出来量清楚（背靠背 launch，
//! 排除 per-boundary `cudaStreamSynchronize` 的固定开销），再决定改哪一段。
//!
//! 门控：
//! `cargo test --release -p lingxi-kernels --features gpu --test t4_route_topk_bench -- --nocapture`
//!
//! 用例：
//! - `t4_route_topk_bench`：背靠背 2000 次平均（另有 per-call sync 口径对照生产探针）；
//! - `t4_route_topk_digest`：20 组定种随机输入 → 打印 ids/wts 的位级摘要。
//! 同一二进制跑两遍（`LINGXI_NO_ROUTE_TOPK_FAST=1` 开/关）比较两行摘要即完成 A/B 对拍。

#![cfg(feature = "gpu")]

use std::time::Instant;

use half::bf16;
use lingxi_kernels::Kernels;

const ROWS: usize = 6;
const E: usize = 512;
const TOPK: usize = 10;

/// xorshift 随机 logits（与 t2 同分布：[-20, 20]，bf16 量化后再喂核）。
fn rand_logits(seed: u64, n: usize) -> Vec<bf16> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 40.0 - 20.0
    };
    (0..n).map(|_| bf16::from_f32(rnd())).collect()
}

#[test]
fn t4_route_topk_bench() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let host = rand_logits(20261023, ROWS * E);
    let logits = k.stream.clone_htod(&host).unwrap();
    let mut ids = k.stream.alloc_zeros::<i32>(ROWS * TOPK).unwrap();
    let mut wts = k.stream.alloc_zeros::<f32>(ROWS * TOPK).unwrap();

    for _ in 0..5 {
        k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, ROWS, E, TOPK).unwrap();
    }
    k.stream.synchronize().unwrap();

    // 口径 A：背靠背 launch，一次 sync（纯 GPU 时间 + 入队开销）
    let iters = 2000;
    let t0 = Instant::now();
    for _ in 0..iters {
        k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, ROWS, E, TOPK).unwrap();
    }
    k.stream.synchronize().unwrap();
    let a = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;

    // 口径 B：每次 launch 后 sync（复刻生产探针：含固定 sync 开销）
    let iters_b = 300;
    let t1 = Instant::now();
    for _ in 0..iters_b {
        k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, ROWS, E, TOPK).unwrap();
        k.stream.synchronize().unwrap();
    }
    let b = t1.elapsed().as_secs_f64() / iters_b as f64 * 1e6;

    // 空 port：同口径量一次「launch+sync」的底噪（同 stream 上一个空转小核不好找，
    // 用 route_topk_rows 的 rows=1 做参照即可看出固定开销量级）
    let mut ids1 = k.stream.alloc_zeros::<i32>(TOPK).unwrap();
    let mut wts1 = k.stream.alloc_zeros::<f32>(TOPK).unwrap();
    let iters_c = 300;
    let t2 = Instant::now();
    for _ in 0..iters_c {
        k.route_topk_rows_bf16(&logits, &mut ids1, &mut wts1, 1, E, TOPK).unwrap();
        k.stream.synchronize().unwrap();
    }
    let c = t2.elapsed().as_secs_f64() / iters_c as f64 * 1e6;

    eprintln!(
        "[topk-bench] rows={ROWS} E={E} top_k={TOPK}｜背靠背 {iters} 次：{a:.2} µs/次｜\
         每次 sync {iters_b} 次：{b:.2} µs/次｜rows=1 每次 sync：{c:.2} µs/次（固定开销参照）"
    );
}

/// 位级摘要：20 组定种随机输入 × 6 行 × (10 ids + 10 wts)。
/// 输出一行 `idshash/wtshash`（+ 前 3 组明细，便于肉眼比对）。
#[test]
fn t4_route_topk_digest() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut ids = k.stream.alloc_zeros::<i32>(ROWS * TOPK).unwrap();
    let mut wts = k.stream.alloc_zeros::<f32>(ROWS * TOPK).unwrap();
    let mut h_ids: u64 = 0xcbf29ce484222325;
    let mut h_wts: u64 = 0xcbf29ce484222325;
    let mut detail = String::new();
    let mut n_ids = 0usize;
    let mut n_wts = 0usize;
    for g in 0..20u64 {
        let host = rand_logits(0x20261023 + g, ROWS * E);
        let logits = k.stream.clone_htod(&host).unwrap();
        k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, ROWS, E, TOPK).unwrap();
        let gi: Vec<i32> = k.stream.clone_dtoh(&ids).unwrap();
        let gw: Vec<f32> = k.stream.clone_dtoh(&wts).unwrap();
        for v in &gi {
            h_ids = (h_ids ^ *v as u32 as u64).wrapping_mul(0x100000001b3);
            n_ids += 1;
        }
        for v in &gw {
            h_wts = (h_wts ^ v.to_bits() as u64).wrapping_mul(0x100000001b3);
            n_wts += 1;
        }
        if g < 3 {
            detail.push_str(&format!("  g{g} ids={gi:?}\n  g{g} wts={gw:?}\n"));
        }
    }
    eprintln!(
        "[topk-digest] 20 组 × {ROWS} 行｜{n_ids} 个 ids、{n_wts} 个 wts｜\
         ids={h_ids:016x} wts={h_wts:016x}"
    );
    eprintln!("{detail}");
}
