//! **MoE 路由链三发的「地板 vs 真活」微基准**（2026-10-24）。
//!
//! 动机：`LINGXI_SEG3_SYNC=1` 的层均是 shared 58~60 / router 27~28 / topk 25~27 /
//! pair 35~37 µs，而三块的数据量都只有几十 KB~2.6 MB ⇒ 怀疑大头是「发数 × 地板」。
//! 本用例把三个核单独拉出来背靠背量（同进程预热 + best_of3），并对每个核用
//! **同 grid/block 的空核**（`empty_probe_scale`，`n=0`）量出该网格形状的发射地板。
//!
//! 门控：
//! `cargo test --release -p lingxi-kernels --features gpu --test t20_route_chain_bw -- --nocapture --test-threads=1`
#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

const ROWS: usize = 6;
const E: usize = 512;
const HIDDEN: usize = 2560;
const TOPK: usize = 10;

fn rndbf16(n: usize, seed: u32) -> Vec<bf16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_f32(((s >> 8) as f32 / 16777216.0 - 0.5) * 0.1)
        })
        .collect()
}

/// 背靠背 iters 发 + 一次 sync；同进程 3 轮取最好（min）。
fn bench<F: FnMut()>(k: &Kernels, tag: &str, bytes: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..20 {
        f();
    }
    k.stream.synchronize().unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        k.stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        best = best.min(us);
    }
    eprintln!(
        "[route-chain] {tag:<44} {best:8.1} µs｜{:6.1} GB/s｜{:6.3} MB",
        bytes as f64 / (best * 1e-6) / 1e9,
        bytes as f64 / 1e6
    );
    best
}

/// 每次 launch 后 sync（复刻 `LINGXI_SEG3_SYNC` 的口径），3 轮取最好。
fn bench_sync<F: FnMut()>(k: &Kernels, tag: &str, iters: usize, mut f: F) -> f64 {
    for _ in 0..10 {
        f();
        k.stream.synchronize().unwrap();
    }
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
            k.stream.synchronize().unwrap();
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        best = best.min(us);
    }
    eprintln!("[route-chain] {tag:<44} {best:8.1} µs（每次 sync，SEG3 口径）");
    best
}

#[test]
fn t20_route_chain_floor() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    // 合成核**默认关**（负结果）⇒ 本节显式打开，才能与「三发」同口径对比。
    unsafe { std::env::set_var("LINGXI_TKPAIR_FUSE", "1") };
    let zx = k.stream.alloc_zeros::<f32>(1).unwrap();
    let mut zy = k.stream.alloc_zeros::<f32>(1).unwrap();

    // ── router：gemm_skinny_row（n=512, k=2560, m=6）grid(512,1) blk256 ──
    let w_r = k.stream.clone_htod(&rndbf16(E * HIDDEN, 11)).unwrap();
    let x = k.stream.clone_htod(&rndbf16(ROWS * HIDDEN, 12)).unwrap();
    let mut logits = k.stream.alloc_zeros::<bf16>(ROWS * E).unwrap();
    // 裸指针取一次（guard 立即 forget：本用例自己 sync；下面还要不可变借 logits）
    let yp = {
        let (p, g) = logits.device_ptr_mut(&k.stream);
        std::mem::forget(g);
        p
    };
    let mut ids = k.stream.alloc_zeros::<i32>(ROWS * TOPK).unwrap();
    let mut wts = k.stream.alloc_zeros::<f32>(ROWS * TOPK).unwrap();

    eprintln!(
        "[route-chain] ── 线上形状 rows={ROWS} E={E} hidden={HIDDEN} top_k={TOPK} ──"
    );
    let router = {
        let (wp, _g1) = w_r.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        bench(&k, "router gemm_skinny_row(512,2560,m6)", E * HIDDEN * 2, 500, || {
            k.gemm_skinny_row_bf16_ptr(wp, xp, yp, E, HIDDEN, ROWS, 0, 0.0, 0.0)
                .unwrap();
        })
    };
    let router_floor = bench(&k, "  ↑空核同 grid(512,1) blk256", 0, 1000, || {
        k.empty_probe_scale(&zx, &mut zy, (E as u32, 1), 256).unwrap();
    });
    eprintln!("[route-chain]   ⇒ router 净活 {:.1} µs", router - router_floor);

    // ── topk：route_topk_rows（6 行 × 512 → 10）grid(6,1) blk256 ──
    let topk = bench(&k, "topk route_topk_rows(6,512,10)", ROWS * E * 2, 1000, || {
        k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, ROWS, E, TOPK)
            .unwrap();
    });
    let topk_floor = bench(&k, "  ↑空核同 grid(6,1) blk256", 0, 1000, || {
        k.empty_probe_scale(&zx, &mut zy, (ROWS as u32, 1), 256).unwrap();
    });
    eprintln!("[route-chain]   ⇒ topk 净活 {:.1} µs", topk - topk_floor);

    // topk 的「一 warp 一行」版（grid=rows, block=32，逐位相同）
    let topk_warp = bench(&k, "topk warp 版(6,512,10)", ROWS * E * 2, 2000, || {
        let ok = k
            .route_topk_row_warp_bf16(&logits, &mut ids, &mut wts, ROWS, E, TOPK)
            .unwrap();
        debug_assert!(ok);
    });
    let topk_warp_floor = bench(&k, "  ↑空核同 grid(6,1) blk32", 0, 2000, || {
        k.empty_probe_scale(&zx, &mut zy, (ROWS as u32, 1), 32).unwrap();
    });
    eprintln!(
        "[route-chain]   ⇒ topk warp 版 {topk_warp:.1} µs（旧核 {topk:.1}，−{:.1} µs / {:.0}%；\
         地板 {topk_warp_floor:.1}，净活 {:.1}）",
        topk - topk_warp,
        100.0 * (topk - topk_warp) / topk,
        topk_warp - topk_warp_floor
    );

    // ── pair：route_pairs（单 block 512 线程，E=512）──
    let slot_map = k.stream.clone_htod(&vec![1i32; E]).unwrap();
    let mut pair_rows = k.stream.alloc_zeros::<i32>(ROWS * TOPK).unwrap();
    let mut pair_wts = k.stream.alloc_zeros::<f32>(ROWS * TOPK).unwrap();
    let mut gslots = k.stream.alloc_zeros::<i32>(E).unwrap();
    let mut gslots_alt = k.stream.alloc_zeros::<i32>(E).unwrap();
    let mut goff = k.stream.alloc_zeros::<i32>(E + 1).unwrap();
    let mut uniq = k.stream.alloc_zeros::<i32>(E).unwrap();
    let mut col = k.stream.alloc_zeros::<i32>(ROWS * TOPK).unwrap();
    let mut roff = k.stream.alloc_zeros::<i32>(ROWS + 1).unwrap();
    let mut counts = k.stream.alloc_zeros::<u32>(8).unwrap();
    let mut hit_pri = k.stream.alloc_zeros::<i32>(E).unwrap();
    let mut hit_alt = k.stream.alloc_zeros::<i32>(E).unwrap();
    let mut miss_ids = k.stream.alloc_zeros::<i32>(E).unwrap();
    let mut pri_gidx = k.stream.alloc_zeros::<i32>(E).unwrap();
    let mut alt_gidx = k.stream.alloc_zeros::<i32>(E).unwrap();
    let pair = {
        let (sp, _g) = slot_map.device_ptr(&k.stream);
        let (cd, _g2) = counts.device_ptr(&k.stream);
        let (hp, _g3) = hit_pri.device_ptr(&k.stream);
        let (ha, _g4) = hit_alt.device_ptr(&k.stream);
        let (mi, _g5) = miss_ids.device_ptr(&k.stream);
        let (ga, _g6) = gslots_alt.device_ptr(&k.stream);
        bench(&k, "pair route_pairs(E=512,rows=6,k=10)", ROWS * E * 2, 1000, || {
            k.route_pairs_bf16(
                &ids, &wts, sp, 0, 0, &mut pair_rows, &mut pair_wts, &mut gslots, ga,
                &mut goff, &mut uniq, &mut col, &mut roff, cd, hp, ha, mi,
                &mut pri_gidx, &mut alt_gidx, E, ROWS, TOPK,
            )
            .unwrap();
        })
    };
    let pair_floor = bench(&k, "  ↑空核同 grid(1,1) blk512", 0, 1000, || {
        k.empty_probe_scale(&zx, &mut zy, (1, 1), 512).unwrap();
    });
    eprintln!("[route-chain]   ⇒ pair 净活 {:.1} µs", pair - pair_floor);

    // ── 同形纯读分母（只读不算；地址流与 router 核逐字相同）──
    {
        let mut rout = k.stream.alloc_zeros::<f32>(E).unwrap();
        let (wp, _g1) = w_r.device_ptr(&k.stream);
        bench(&k, "纯读 row8 形 grid(512) k2560", E * HIDDEN * 2, 500, || {
            k.probe_read_row8_bf16(wp, &mut rout, E, HIDDEN).unwrap();
        });
        bench(&k, "理想流式读 2.62MB", E * HIDDEN * 2, 500, || {
            k.probe_read_stream_bf16(wp, &mut rout, E * HIDDEN).unwrap();
        });
    }

    // ── router 的 m≤6 特化核（`router_m6.cu`）──
    let router_m6 = {
        let (wp, _g1) = w_r.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        bench(&k, "router m6 特化核(512,2560,m6)", E * HIDDEN * 2, 500, || {
            k.gemm_skinny_row_m6_bf16_ptr(wp, xp, yp, E, HIDDEN, ROWS, 0, 0.0, 0.0)
                .unwrap();
        })
    };
    eprintln!(
        "[route-chain]   ⇒ router 特化核 {router_m6:.1} µs（行核 {router:.1}，−{:.1} µs / {:.0}%）",
        router - router_m6,
        100.0 * (router - router_m6) / router
    );

    // ── topk + pair 合成一发（`route_fuse.cu`）──
    let fused = {
        let (sp, _g) = slot_map.device_ptr(&k.stream);
        let (cd, _g2) = counts.device_ptr(&k.stream);
        let (hp, _g3) = hit_pri.device_ptr(&k.stream);
        let (ha, _g4) = hit_alt.device_ptr(&k.stream);
        let (mi, _g5) = miss_ids.device_ptr(&k.stream);
        let (ga, _g6) = gslots_alt.device_ptr(&k.stream);
        bench(&k, "【合成一发】topk+pair", ROWS * E * 2, 1000, || {
            let ok = k
                .route_topk_pairs_bf16(
                    &logits, &mut ids, &mut wts, sp, 0, 0, &mut pair_rows, &mut pair_wts,
                    &mut gslots, ga, &mut goff, &mut uniq, &mut col, &mut roff, cd, hp, ha,
                    mi, &mut pri_gidx, &mut alt_gidx, E, ROWS, TOPK,
                )
                .unwrap();
            debug_assert!(ok);
        })
    };
    eprintln!(
        "[route-chain]   ⇒ 合成一发 {fused:.1} µs（vs 两发 {:.1} = topk+pair，省 {:.1}）",
        topk + pair,
        topk + pair - fused
    );

    // ── 三发连发（真 CPU 侧连发，无中间 sync）──
    let chain = {
        let (wp, _g1) = w_r.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        let (sp, _g) = slot_map.device_ptr(&k.stream);
        let (cd, _g2b) = counts.device_ptr(&k.stream);
        let (hp, _g3b) = hit_pri.device_ptr(&k.stream);
        let (ha, _g4) = hit_alt.device_ptr(&k.stream);
        let (mi, _g5) = miss_ids.device_ptr(&k.stream);
        let (ga, _g6) = gslots_alt.device_ptr(&k.stream);
        bench(&k, "【三发整链】router+topk+pair", E * HIDDEN * 2, 500, || {
            k.gemm_skinny_row_bf16_ptr(wp, xp, yp, E, HIDDEN, ROWS, 0, 0.0, 0.0)
                .unwrap();
            k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, ROWS, E, TOPK)
                .unwrap();
            k.route_pairs_bf16(
                &ids, &wts, sp, 0, 0, &mut pair_rows, &mut pair_wts, &mut gslots, ga,
                &mut goff, &mut uniq, &mut col, &mut roff, cd, hp, ha, mi,
                &mut pri_gidx, &mut alt_gidx, E, ROWS, TOPK,
            )
            .unwrap();
        })
    };
    let chain_floor = bench(&k, "  ↑空核同三发（512/6/1）", 0, 500, || {
        k.empty_probe_scale(&zx, &mut zy, (E as u32, 1), 256).unwrap();
        k.empty_probe_scale(&zx, &mut zy, (ROWS as u32, 1), 256).unwrap();
        k.empty_probe_scale(&zx, &mut zy, (1, 1), 512).unwrap();
    });
    eprintln!(
        "[route-chain]   ⇒ 三发整链 {chain:.1} µs（地板 {chain_floor:.1}，净活 {:.1}）",
        chain - chain_floor
    );

    // ── 线上整链（router 特化核 + topk + pair = 默认三发）──
    let chain_prod = {
        let (wp, _g1) = w_r.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        let (sp, _g) = slot_map.device_ptr(&k.stream);
        let (cd, _g2b) = counts.device_ptr(&k.stream);
        let (hp, _g3b) = hit_pri.device_ptr(&k.stream);
        let (ha, _g4) = hit_alt.device_ptr(&k.stream);
        let (mi, _g5) = miss_ids.device_ptr(&k.stream);
        let (ga, _g6) = gslots_alt.device_ptr(&k.stream);
        bench(&k, "【线上整链】router_m6+topk_warp+pair", E * HIDDEN * 2, 500, || {
            k.gemm_skinny_row_m6_bf16_ptr(wp, xp, yp, E, HIDDEN, ROWS, 0, 0.0, 0.0)
                .unwrap();
            k.route_topk_row_warp_bf16(&logits, &mut ids, &mut wts, ROWS, E, TOPK)
                .unwrap();
            k.route_pairs_bf16(
                &ids, &wts, sp, 0, 0, &mut pair_rows, &mut pair_wts, &mut gslots, ga,
                &mut goff, &mut uniq, &mut col, &mut roff, cd, hp, ha, mi,
                &mut pri_gidx, &mut alt_gidx, E, ROWS, TOPK,
            )
            .unwrap();
        })
    };
    eprintln!(
        "[route-chain]   ⇒ **线上整链 {chain_prod:.1} µs**（原三发 {chain:.1}，−{:.1} µs / {:.0}%）",
        chain - chain_prod,
        100.0 * (chain - chain_prod) / chain
    );

    // ── 两发整链（router + 合成核）──
    let chain2 = {
        let (wp, _g1) = w_r.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        let (sp, _g) = slot_map.device_ptr(&k.stream);
        let (cd, _g2b) = counts.device_ptr(&k.stream);
        let (hp, _g3b) = hit_pri.device_ptr(&k.stream);
        let (ha, _g4) = hit_alt.device_ptr(&k.stream);
        let (mi, _g5) = miss_ids.device_ptr(&k.stream);
        let (ga, _g6) = gslots_alt.device_ptr(&k.stream);
        bench(&k, "【两发整链】router+合成核", E * HIDDEN * 2, 500, || {
            k.gemm_skinny_row_bf16_ptr(wp, xp, yp, E, HIDDEN, ROWS, 0, 0.0, 0.0)
                .unwrap();
            let ok = k
                .route_topk_pairs_bf16(
                    &logits, &mut ids, &mut wts, sp, 0, 0, &mut pair_rows, &mut pair_wts,
                    &mut gslots, ga, &mut goff, &mut uniq, &mut col, &mut roff, cd, hp, ha,
                    mi, &mut pri_gidx, &mut alt_gidx, E, ROWS, TOPK,
                )
                .unwrap();
            debug_assert!(ok);
        })
    };
    let chain2_floor = bench(&k, "  ↑空核同两发（512/1）", 0, 500, || {
        k.empty_probe_scale(&zx, &mut zy, (E as u32, 1), 256).unwrap();
        k.empty_probe_scale(&zx, &mut zy, (1, 1), 512).unwrap();
    });
    eprintln!(
        "[route-chain]   ⇒ **两发整链 {chain2:.1} µs**（地板 {chain2_floor:.1}，净活 {:.1}）｜\
         相对三发 {chain:.1} −{:.1} µs（{:.0}%）",
        chain2 - chain2_floor,
        chain - chain2,
        100.0 * (chain - chain2) / chain
    );

    // ── 对照：SEG3 口径（每发 sync）──
    let s_router = {
        let (wp, _g1) = w_r.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        let (yp, _g3) = logits.device_ptr_mut(&k.stream);
        bench_sync(&k, "router（每次 sync）", 300, || {
            k.gemm_skinny_row_bf16_ptr(wp, xp, yp, E, HIDDEN, ROWS, 0, 0.0, 0.0)
                .unwrap();
        })
    };
    let s_topk = bench_sync(&k, "topk（每次 sync）", 300, || {
        k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, ROWS, E, TOPK)
            .unwrap();
    });
    let s_pair = {
        let (sp, _g) = slot_map.device_ptr(&k.stream);
        let (cd, _g2) = counts.device_ptr(&k.stream);
        let (hp, _g3) = hit_pri.device_ptr(&k.stream);
        let (ha, _g4) = hit_alt.device_ptr(&k.stream);
        let (mi, _g5) = miss_ids.device_ptr(&k.stream);
        let (ga, _g6) = gslots_alt.device_ptr(&k.stream);
        bench_sync(&k, "pair（每次 sync）", 300, || {
            k.route_pairs_bf16(
                &ids, &wts, sp, 0, 0, &mut pair_rows, &mut pair_wts, &mut gslots, ga,
                &mut goff, &mut uniq, &mut col, &mut roff, cd, hp, ha, mi,
                &mut pri_gidx, &mut alt_gidx, E, ROWS, TOPK,
            )
            .unwrap();
        })
    };
    let s_empty = bench_sync(&k, "空核 grid(1,1) blk256（每次 sync）", 300, || {
        k.empty_probe_scale(&zx, &mut zy, (1, 1), 256).unwrap();
    });
    eprintln!(
        "[route-chain] ── 汇总（背靠背 / SEG3 口径）──\n\
         [route-chain]   router {router:6.1} / {s_router:6.1} µs（地板 {router_floor:5.1}）\n\
         [route-chain]   topk   {topk:6.1} / {s_topk:6.1} µs（地板 {topk_floor:5.1}）\n\
         [route-chain]   pair   {pair:6.1} / {s_pair:6.1} µs（地板 {pair_floor:5.1}）\n\
         [route-chain]   三发整链 {chain:6.1} µs（地板 {chain_floor:5.1}）｜每次 sync 空核 {s_empty:5.1}"
    );
}
