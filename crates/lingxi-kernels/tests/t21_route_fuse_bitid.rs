//! `lingxi_route_topk_pairs_bf16`（top-k + pair 建表**合成一发**）的**逐位对拍**。
//!
//! 判据：与「`lingxi_route_topk_rows_bf16` → `lingxi_route_pairs_bf16` 两发」相比，
//! **所有输出表逐位相同**（`ids/wts` 是 token 流的直接输入；`pair_*/col/roff/group_*`
//! 决定 expert GEMV 与 CSR 归并的次序，任何一项不同都会改数字）。
//!
//! 覆盖：rows 2..16、E=512、top_k 10、单/双视图（`fmt_tab` + `slot_map_alt`）、
//! 含 miss（槽号为 0）与不含 miss、以及并列 / -inf / NaN 三种 logits 分布。
//!
//! `hit_pri`/`hit_alt` 的**内容是原子序**（旧核自己两次跑也不同）⇒ 只比「排序后的多重集」。
//!
//! 门控：
//! `cargo test --release -p lingxi-kernels --features gpu --test t21_route_fuse_bitid -- --nocapture --test-threads=1`

#![cfg(feature = "gpu")]

use cudarc::driver::{CudaSlice, DevicePtr};
use half::bf16;
use lingxi_kernels::Kernels;

const E: usize = 512;
const TOPK: usize = 10;

struct Case {
    logits: Vec<bf16>,
    rows: usize,
    fmt: Option<Vec<i32>>,
    slot_pri: Vec<i32>,
    slot_alt: Vec<i32>,
}

fn rnd(seed: u64, n: usize, kind: u32) -> Vec<bf16> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / 9007199254740992.0
    };
    (0..n)
        .map(|_| match kind {
            // 0：连续均匀
            0 => bf16::from_f32(rnd() * 40.0 - 20.0),
            // 1：大量并列（只有 8 个不同值）
            1 => bf16::from_f32((rnd() * 8.0).floor() - 4.0),
            // 2：混入 -inf / NaN（非有限 logits 的健壮性）
            _ => {
                let r = rnd();
                if r < 0.01 {
                    bf16::from_f32(f32::NEG_INFINITY)
                } else if r < 0.02 {
                    bf16::from_f32(f32::NAN)
                } else {
                    bf16::from_f32(r * 20.0 - 10.0)
                }
            }
        })
        .collect()
}

/// 一组输出（两条路径各一份）
struct Out {
    ids: CudaSlice<i32>,
    wts: CudaSlice<f32>,
    pair_rows: CudaSlice<i32>,
    pair_wts: CudaSlice<f32>,
    gslots: CudaSlice<i32>,
    gslots_alt: CudaSlice<i32>,
    goff: CudaSlice<i32>,
    uniq: CudaSlice<i32>,
    col: CudaSlice<i32>,
    roff: CudaSlice<i32>,
    counts: CudaSlice<u32>,
    hit_pri: CudaSlice<i32>,
    hit_alt: CudaSlice<i32>,
    miss_ids: CudaSlice<i32>,
    pri_gidx: CudaSlice<i32>,
    alt_gidx: CudaSlice<i32>,
}

impl Out {
    fn new(k: &Kernels, rows: usize) -> Self {
        let s = rows * TOPK;
        Self {
            ids: k.stream.alloc_zeros::<i32>(s).unwrap(),
            wts: k.stream.alloc_zeros::<f32>(s).unwrap(),
            pair_rows: k.stream.alloc_zeros::<i32>(s).unwrap(),
            pair_wts: k.stream.alloc_zeros::<f32>(s).unwrap(),
            gslots: k.stream.alloc_zeros::<i32>(E).unwrap(),
            gslots_alt: k.stream.alloc_zeros::<i32>(E).unwrap(),
            goff: k.stream.alloc_zeros::<i32>(E + 1).unwrap(),
            uniq: k.stream.alloc_zeros::<i32>(E).unwrap(),
            col: k.stream.alloc_zeros::<i32>(s).unwrap(),
            roff: k.stream.alloc_zeros::<i32>(rows + 1).unwrap(),
            counts: k.stream.alloc_zeros::<u32>(8).unwrap(),
            hit_pri: k.stream.alloc_zeros::<i32>(E).unwrap(),
            hit_alt: k.stream.alloc_zeros::<i32>(E).unwrap(),
            miss_ids: k.stream.alloc_zeros::<i32>(E).unwrap(),
            pri_gidx: k.stream.alloc_zeros::<i32>(E).unwrap(),
            alt_gidx: k.stream.alloc_zeros::<i32>(E).unwrap(),
        }
    }
}

fn eq_i32(k: &Kernels, a: &CudaSlice<i32>, b: &CudaSlice<i32>, n: usize, tag: &str, bad: &mut usize) {
    let av: Vec<i32> = k.stream.clone_dtoh(&a.slice(0..n)).unwrap();
    let bv: Vec<i32> = k.stream.clone_dtoh(&b.slice(0..n)).unwrap();
    for i in 0..n {
        if av[i] != bv[i] {
            if *bad < 6 {
                eprintln!("[fuse-bitid] {tag}[{i}] 两发={} 一发={}", av[i], bv[i]);
            }
            *bad += 1;
        }
    }
}

/// 独立核 `lingxi_route_topk_row_warp_bf16`（一 warp 一行）vs 旧核 `route_topk_rows_bf16`。
#[test]
fn t21_topk_warp_bitid() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut total_bad = 0usize;
    let mut cases = 0usize;
    for seed in 0..32u64 {
        let e = [512usize, 256, 512, 128][seed as usize % 4];
        let rows = 1 + (seed as usize % 16);
        let top_k = [10usize, 1, 64, 32][seed as usize % 4];
        let kind = (seed % 3) as u32;
        let host = rnd(0x770000 + seed, rows * e, kind);
        let logits = k.stream.clone_htod(&host).unwrap();
        let mut a_ids = k.stream.alloc_zeros::<i32>(rows * top_k).unwrap();
        let mut a_wts = k.stream.alloc_zeros::<f32>(rows * top_k).unwrap();
        let mut b_ids = k.stream.alloc_zeros::<i32>(rows * top_k).unwrap();
        let mut b_wts = k.stream.alloc_zeros::<f32>(rows * top_k).unwrap();
        k.route_topk_rows_bf16(&logits, &mut a_ids, &mut a_wts, rows, e, top_k).unwrap();
        let warp = k
            .route_topk_row_warp_bf16(&logits, &mut b_ids, &mut b_wts, rows, e, top_k)
            .unwrap();
        assert!(warp, "warp 版没生效");
        k.stream.synchronize().unwrap();
        let ai: Vec<i32> = k.stream.clone_dtoh(&a_ids).unwrap();
        let bi: Vec<i32> = k.stream.clone_dtoh(&b_ids).unwrap();
        let aw: Vec<f32> = k.stream.clone_dtoh(&a_wts).unwrap();
        let bw: Vec<f32> = k.stream.clone_dtoh(&b_wts).unwrap();
        let mut bad = 0usize;
        for i in 0..rows * top_k {
            if ai[i] != bi[i] || aw[i].to_bits() != bw[i].to_bits() {
                if bad < 4 {
                    eprintln!(
                        "[topk-warp] e{e} rows{rows} k{top_k} kind{kind} [{i}] ids {}→{} wts {:#x}→{:#x}",
                        ai[i], bi[i], aw[i].to_bits(), bw[i].to_bits()
                    );
                }
                bad += 1;
            }
        }
        cases += 1;
        total_bad += bad;
    }
    eprintln!("[topk-warp] {cases} 组（E=512/256/128 × rows 1..16 × top_k 1/10/32/64 × 含并列/NaN）⇒ **不一致 {total_bad} 处**");
    assert_eq!(total_bad, 0, "warp 版 top-k 与旧核不逐位相同");
}

#[test]
fn t21_route_fuse_bitid() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    // 合成路径**默认关**（负结果，见 `wrappers.rs` 的 `tkpair_fuse_on`）⇒ 本用例显式打开它。
    unsafe { std::env::set_var("LINGXI_TKPAIR_FUSE", "1") };
    let mut total_bad = 0usize;
    let mut cases = 0usize;
    for seed in 0..25u64 {
        let rows = [2usize, 3, 5, 6, 8, 11, 16][seed as usize % 7];
        let kind = (seed % 3) as u32;
        let dual = seed % 2 == 1;
        let with_miss = seed % 4 == 3;
        let c = Case {
            logits: rnd(0x20261024 + seed, rows * E, kind),
            rows,
            fmt: if dual { Some((0..E).map(|i| ((i as u64 * 7 + seed) % 2) as i32).collect()) } else { None },
            slot_pri: (0..E)
                .map(|i| {
                    if with_miss && i % 37 == 0 {
                        0
                    } else {
                        1 + (i as i32 % 60)
                    }
                })
                .collect(),
            slot_alt: (0..E).map(|i| 1 + ((i as i32 * 3) % 60)).collect(),
        };
        let logits = k.stream.clone_htod(&c.logits).unwrap();
        let slot_pri = k.stream.clone_htod(&c.slot_pri).unwrap();
        let slot_alt = k.stream.clone_htod(&c.slot_alt).unwrap();
        let fmt = c.fmt.as_ref().map(|f| k.stream.clone_htod(f).unwrap());
        let mut a = Out::new(&k, rows);
        let mut b = Out::new(&k, rows);
        let s = rows * TOPK;
        {
            let (sp, _g1) = slot_pri.device_ptr(&k.stream);
            let (sa, _g2) = slot_alt.device_ptr(&k.stream);
            let fm_hold;
            let fm_p = match fmt.as_ref() {
                Some(f) => {
                    let (p, g) = f.device_ptr(&k.stream);
                    fm_hold = Some(g);
                    p
                }
                None => {
                    fm_hold = None;
                    0u64
                }
            };
            let _ = &fm_hold;
            let (cd_a, _ga) = a.counts.device_ptr(&k.stream);
            let (hp_a, _gb) = a.hit_pri.device_ptr(&k.stream);
            let (ha_a, _gc) = a.hit_alt.device_ptr(&k.stream);
            let (mi_a, _gd) = a.miss_ids.device_ptr(&k.stream);
            let (gsa_a, _ge) = a.gslots_alt.device_ptr(&k.stream);
            // ── A：两发（旧路径）──
            k.route_topk_rows_bf16(&logits, &mut a.ids, &mut a.wts, rows, E, TOPK).unwrap();
            k.route_pairs_bf16(
                &a.ids, &a.wts, sp, if dual { sa } else { 0 }, fm_p,
                &mut a.pair_rows, &mut a.pair_wts, &mut a.gslots, gsa_a,
                &mut a.goff, &mut a.uniq, &mut a.col, &mut a.roff,
                cd_a, hp_a, ha_a, mi_a, &mut a.pri_gidx, &mut a.alt_gidx,
                E, rows, TOPK,
            )
            .unwrap();
            // ── B：一发（合成核）──
            let (cd_b, _ga) = b.counts.device_ptr(&k.stream);
            let (hp_b, _gb) = b.hit_pri.device_ptr(&k.stream);
            let (ha_b, _gc) = b.hit_alt.device_ptr(&k.stream);
            let (mi_b, _gd) = b.miss_ids.device_ptr(&k.stream);
            let (gsa_b, _ge) = b.gslots_alt.device_ptr(&k.stream);
            let fused = k
                .route_topk_pairs_bf16(
                    &logits, &mut b.ids, &mut b.wts, sp, if dual { sa } else { 0 }, fm_p,
                    &mut b.pair_rows, &mut b.pair_wts, &mut b.gslots, gsa_b,
                    &mut b.goff, &mut b.uniq, &mut b.col, &mut b.roff,
                    cd_b, hp_b, ha_b, mi_b, &mut b.pri_gidx, &mut b.alt_gidx,
                    E, rows, TOPK,
                )
                .unwrap();
            assert!(fused, "合成核没有生效（形状越界？）");
        }
        k.stream.synchronize().unwrap();
        let ca: Vec<u32> = k.stream.clone_dtoh(&a.counts).unwrap();
        let cb: Vec<u32> = k.stream.clone_dtoh(&b.counts).unwrap();
        let mut bad = 0usize;
        for i in [0usize, 1, 2, 3, 4, 5, 6, 7] {
            if ca[i] != cb[i] {
                if bad < 6 {
                    eprintln!("[fuse-bitid] counts[{i}] 两发={} 一发={}", ca[i], cb[i]);
                }
                bad += 1;
            }
        }
        eq_i32(&k, &a.ids, &b.ids, s, "ids", &mut bad);
        eq_i32(&k, &a.pair_rows, &b.pair_rows, ca[1] as usize, "pair_rows", &mut bad);
        eq_i32(&k, &a.col, &b.col, s, "col", &mut bad);
        eq_i32(&k, &a.roff, &b.roff, rows + 1, "roff", &mut bad);
        eq_i32(&k, &a.goff, &b.goff, ca[1] as usize + 1, "group_off", &mut bad);
        eq_i32(&k, &a.uniq, &b.uniq, ca[1] as usize, "uniq_experts", &mut bad);
        eq_i32(&k, &a.gslots, &b.gslots, ca[1] as usize, "group_slots", &mut bad);
        eq_i32(&k, &a.gslots_alt, &b.gslots_alt, ca[1] as usize, "group_slots_alt", &mut bad);
        eq_i32(&k, &a.pri_gidx, &b.pri_gidx, ca[6] as usize, "pri_gidx", &mut bad);
        eq_i32(&k, &a.alt_gidx, &b.alt_gidx, ca[7] as usize, "alt_gidx", &mut bad);
        eq_i32(&k, &a.miss_ids, &b.miss_ids, ca[2] as usize, "miss_ids", &mut bad);
        {
            let av: Vec<f32> = k.stream.clone_dtoh(&a.pair_wts.slice(0..ca[1] as usize)).unwrap();
            let bv: Vec<f32> = k.stream.clone_dtoh(&b.pair_wts.slice(0..ca[1] as usize)).unwrap();
            for i in 0..av.len() {
                if av[i].to_bits() != bv[i].to_bits() {
                    if bad < 6 {
                        eprintln!("[fuse-bitid] pair_wts[{i}] {:#x} vs {:#x}", av[i].to_bits(), bv[i].to_bits());
                    }
                    bad += 1;
                }
            }
            let aw: Vec<f32> = k.stream.clone_dtoh(&a.wts).unwrap();
            let bw: Vec<f32> = k.stream.clone_dtoh(&b.wts).unwrap();
            for i in 0..s {
                if aw[i].to_bits() != bw[i].to_bits() {
                    if bad < 6 {
                        eprintln!("[fuse-bitid] wts[{i}] {:#x} vs {:#x}", aw[i].to_bits(), bw[i].to_bits());
                    }
                    bad += 1;
                }
            }
        }
        // 保护集：内容顺序是原子序 ⇒ 只比多重集
        for (tag, x, y, n) in [
            ("hit_pri", &a.hit_pri, &b.hit_pri, ca[4] as usize),
            ("hit_alt", &a.hit_alt, &b.hit_alt, ca[5] as usize),
        ] {
            let mut av: Vec<i32> = k.stream.clone_dtoh(&x.slice(0..n)).unwrap();
            let mut bv: Vec<i32> = k.stream.clone_dtoh(&y.slice(0..n)).unwrap();
            av.sort_unstable();
            bv.sort_unstable();
            if av != bv {
                eprintln!("[fuse-bitid] {tag} 多重集不同（两发 {:?} / 一发 {:?}）", &av[..av.len().min(8)], &bv[..bv.len().min(8)]);
                bad += 1;
            }
        }
        cases += 1;
        if bad > 0 {
            eprintln!(
                "[fuse-bitid] ✗ case{seed} rows{rows} kind{kind} dual{dual} miss{with_miss}：{bad} 处不一致"
            );
            total_bad += bad;
        }
    }
    eprintln!(
        "[fuse-bitid] {cases} 组用例（rows 2..16 / E{E} / top_k{TOPK} / 单双视图 / 含 miss / 并列+-inf+NaN）\
         ⇒ **不一致 {total_bad} 处**"
    );
    assert_eq!(total_bad, 0, "合成核与两发路径不逐位相同");
}
