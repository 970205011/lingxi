//! **QSA gather 的 prefill（t=512）形态分母**（，2026-10-26）。
//!
//! 背景（attn_prof t=512 事件口径，parity prompt 4167 tok）：QSA attn 段
//! len=0（total≈512/行）≈ 2.77 ms/层、len=512（total≈1024/行）≈ 5.96 ms/层，
//! NH=4 后 2.00 / 4.45 ms——K/V 读取量（行×头 放大）按 ~2.2 TB/s 折算正好贴 L2
//! 带宽 ⇒ 主嫌疑 = L2 流量放大。本用例在生产形状（t=512 / heads24 / kv2 / hd256 /
//! compress4 / topk512）下量：prod NH=1/2/4、同形纯读、空核、pass1/pass2 分拆、
//! ALU 地板（constkv/constq），并逐位对拍 NH=1/2/4。
//!
//! 口径：`--test-threads=1`；同进程**预热** + `best_of3` + 同进程内交错（同 t17）。
//!
//! `bash
//! cd /models/lingxi && QE_NVCC=/models/cuda-12.4/bin/nvcc \
//! LD_LIBRARY_PATH=/models/cuda-12.4/targets/x86_64-linux/lib \
//! cargo test --release -p lingxi-kernels --features gpu --test t25_qsa_prefill_bw \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

// prefill 线上形状
const T: usize = 512;
const HEADS: usize = 24;
const KVH: usize = 2;
const HD: usize = 256;
const COMPRESS: usize = 4;
const BUDGET: usize = 2048;
const TOPK: usize = BUDGET / COMPRESS; // 512

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

struct Attn {
    qn: CudaSlice<bf16>,
    kn: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    blk_tab: CudaSlice<i32>,
    sel: CudaSlice<i32>,
    sel_cnt: CudaSlice<i32>,
    out: CudaSlice<bf16>,
    nblocks: usize,
    vis_base: usize,
    total: usize,
}

/// `scenario`：
/// * A（all_kept，len512 chunk）：nblocks=256 ≤ topk ⇒ sel=恒等、cnt=256、total=1024；
/// * B（selected，len3584 chunk）：nblocks=1024、每行 sorted 512 个伪随机块、total=2048；
/// * C（all_kept 满额，len1536 chunk）：nblocks=512、sel=恒等、total=2048——唯一 4.2MB
/// **全量驻留 L2**（6MB）的对照：区分"容量颠簸"与"延迟/占用"两种墙壁。
fn attn_bufs(k: &Kernels, scenario: char) -> Attn {
    let (nblocks, selcnt, vis_base) = match scenario {
        'A' => (256usize, 256usize, 512usize),
        'B' => (1024usize, 512usize, 3584usize),
        _ => (512usize, 512usize, 1536usize),
    };
    let cap = nblocks * COMPRESS + 64;
    let qn = k.stream.clone_htod(&tobf16(&rnd(T * HEADS * HD, 21, 0.3))).unwrap();
    let kn = k.stream.clone_htod(&tobf16(&rnd(cap * KVH * HD, 22, 0.3))).unwrap();
    let v = k.stream.clone_htod(&tobf16(&rnd(cap * KVH * HD, 23, 0.3))).unwrap();
    let blk_tab = k.stream.clone_htod(&(0..(nblocks + 16) as i32).collect::<Vec<_>>()).unwrap();
    let mut sel = vec![0i32; T * TOPK];
    for r in 0..T {
        if scenario == 'A' || scenario == 'C' {
            for b in 0..selcnt {
                sel[r * TOPK + b] = b as i32;
            }
        } else {
            // 每行确定性伪随机 512 个不同块，升序（与 sel_build 输出同形）
            let mut s = (r as u32 + 1) | 1;
            let mut taken = vec![0u8; nblocks];
            let mut cnt = 0usize;
            while cnt < selcnt {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let b = (s >> 8) as usize % nblocks;
                if taken[b] == 0 {
                    taken[b] = 1;
                    cnt += 1;
                }
            }
            let mut w = 0;
            for b in 0..nblocks {
                if taken[b] == 1 {
                    sel[r * TOPK + w] = b as i32;
                    w += 1;
                }
            }
        }
    }
    let sel = k.stream.clone_htod(&sel).unwrap();
    let sel_cnt = k.stream.clone_htod(&vec![selcnt as i32; T]).unwrap();
    let out = k.stream.alloc_zeros::<bf16>(T * HEADS * HD).unwrap();
    let total = selcnt * COMPRESS;
    Attn { qn, kn, v, blk_tab, sel, sel_cnt, out, nblocks, vis_base, total }
}

/// 单侧（K 或 V）读取字节：行 × (heads/nh) 组 × total token × 512B。
fn attn_bytes(a: &Attn, nh: usize) -> f64 {
    ((HEADS / nh) * T * a.total * HD * 2) as f64
}

fn run_probe(k: &Kernels, a: &mut Attn, which: usize) {
    let scale = 1.0f32 / (HD as f32).sqrt();
    k.qsa_attn_probe(
        which, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
        &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
        T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks, COMPRESS.trailing_zeros() as usize, TOPK,
    )
    .unwrap();
}

fn run_prod(k: &Kernels, a: &mut Attn, nh: usize) {
    let scale = 1.0f32 / (HD as f32).sqrt();
    k.attn_gather_bf16(
        &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
        &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
        T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks, COMPRESS.trailing_zeros() as usize, TOPK, nh,
    )
    .unwrap();
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

fn report(tag: &str, bytes: f64, dt: f64) {
    eprintln!(
        "[ak-bw] {tag:<46}｜{:8.2} µs｜{:7.1} GB/s｜流量 {:7.2} MB",
        dt * 1e6,
        if bytes > 0.0 { bytes / dt / 1e9 } else { 0.0 },
        bytes / 1e6
    );
}

#[test]
fn t25_attn_prefill_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    for scenario in ['A', 'B', 'C'] {
        let mut a = attn_bufs(&k, scenario);
        // 预热（含时钟爬坡）
        for _ in 0..50 {
            run_prod(&k, &mut a, 1);
        }
        k.stream.synchronize().unwrap();
        let iters = 50;
        eprintln!(
            "[ak-bw] ===== 场景 {scenario}：nblocks={} total/行={} K+V 唯一 {:.1} MB =====",
            a.nblocks, a.total, 2.0 * a.nblocks as f64 * COMPRESS as f64 * KVH as f64 * HD as f64 * 2.0 / 1e6
        );
        let d1 = best_of3(&k, iters, || run_prod(&k, &mut a, 1));
        report("0 现状 prod NH=1", 2.0 * attn_bytes(&a, 1), d1);
        let d2 = best_of3(&k, iters, || run_prod(&k, &mut a, 2));
        report("1 现状 prod NH=2", 2.0 * attn_bytes(&a, 2), d2);
        let d4 = best_of3(&k, iters, || run_prod(&k, &mut a, 4));
        report("2 现状 prod NH=4", 2.0 * attn_bytes(&a, 4), d4);
        let d_e1 = best_of3(&k, iters, || run_probe(&k, &mut a, 16));
        report("3 空核 NH=1（grid 512×24）", 0.0, d_e1);
        let d_e4 = best_of3(&k, iters, || run_probe(&k, &mut a, 10));
        report("4 空核 NH=4（grid 512×6）", 0.0, d_e4);
        let d_rk4 = best_of3(&k, iters, || run_probe(&k, &mut a, 1));
        report("6 同形纯读 K+V NH=4", 2.0 * attn_bytes(&a, 4), d_rk4);
        let d_rk_k = best_of3(&k, iters, || run_probe(&k, &mut a, 2));
        report("6a 只读 K NH=4（pass1 地址流）", attn_bytes(&a, 4), d_rk_k);
        let d_rk_v = best_of3(&k, iters, || run_probe(&k, &mut a, 3));
        report("6b 只读 V NH=4（pass2 地址流）", attn_bytes(&a, 4), d_rk_v);
        let d_rk_v2 = best_of3(&k, iters, || run_probe(&k, &mut a, 15));
        report("6c 只读 V v2 地址流 NH=4（按选块分组）", attn_bytes(&a, 4), d_rk_v2);
        let d_ck4 = best_of3(&k, iters, || run_probe(&k, &mut a, 5));
        report("7 ALU 地板 constkv NH=4", 0.0, d_ck4);
        let d_cq4 = best_of3(&k, iters, || run_probe(&k, &mut a, 4));
        report("8 constq NH=4（不读 q）", 2.0 * attn_bytes(&a, 4), d_cq4);
        let d_p1 = best_of3(&k, iters, || run_probe(&k, &mut a, 6));
        report("9 只做第一遍 pass1 NH=4", attn_bytes(&a, 4), d_p1);
        let d_p2 = best_of3(&k, iters, || run_probe(&k, &mut a, 7));
        report("10 只做第二遍 pass2 NH=4", attn_bytes(&a, 4), d_p2);
        let d_v2pf1 = best_of3(&k, iters, || run_probe(&k, &mut a, 14));
        report("11 v2pf NH=1（候选=prod 结构对拍）", 2.0 * attn_bytes(&a, 1), d_v2pf1);
        let d_v3_1 = best_of3(&k, iters, || run_probe(&k, &mut a, 18));
        report("12  v3 NH=1（流水化+向量LDS）", 2.0 * attn_bytes(&a, 1), d_v3_1);
        let d_v3_2 = best_of3(&k, iters, || run_probe(&k, &mut a, 19));
        report("13  v3 NH=2", 2.0 * attn_bytes(&a, 2), d_v3_2);
        let d_v3_4 = best_of3(&k, iters, || run_probe(&k, &mut a, 20));
        report("14  v3 NH=4", 2.0 * attn_bytes(&a, 4), d_v3_4);
        // v5：global 分数缓冲（layout 对三种 NH 都是 gridY*t*span*NH = 24*T*span）
        let span = TOPK * COMPRESS + COMPRESS;
        let mut spg = k.stream.alloc_zeros::<f32>(HEADS * T * span).unwrap();
        let run_v5 = |k: &Kernels, a: &mut Attn, which: usize, spg: &mut CudaSlice<f32>| {
            let scale = 1.0f32 / (HD as f32).sqrt();
            k.qsa_attn_v5_probe(
                which, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
                &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
                &mut spg.slice_mut(..),
                T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks,
                COMPRESS.trailing_zeros() as usize, TOPK,
            )
            .unwrap();
        };
        let mut spg1 = spg.clone();
        let d_v5_1 = best_of3(&k, iters, || run_v5(&k, &mut a, 0, &mut spg1));
        report("15  v5 NH=1（global 分数+满占用）", 2.0 * attn_bytes(&a, 1), d_v5_1);
        let mut spg2 = spg.clone();
        let d_v5_2 = best_of3(&k, iters, || run_v5(&k, &mut a, 1, &mut spg2));
        report("16  v5 NH=2", 2.0 * attn_bytes(&a, 2), d_v5_2);
        let d_v5_4 = best_of3(&k, iters, || run_v5(&k, &mut a, 2, &mut spg));
        report("17  v5 NH=4", 2.0 * attn_bytes(&a, 4), d_v5_4);
        let run_v6 = |k: &Kernels, a: &mut Attn, which: usize| {
            let scale = 1.0f32 / (HD as f32).sqrt();
            k.qsa_attn_v6_probe(
                which, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
                &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
                T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks,
                COMPRESS.trailing_zeros() as usize, TOPK,
            )
            .unwrap();
        };
        let d_v6_1 = best_of3(&k, iters, || run_v6(&k, &mut a, 0));
        report("18  v6 NH=1（grid 转置）", 2.0 * attn_bytes(&a, 1), d_v6_1);
        let d_v6_2 = best_of3(&k, iters, || run_v6(&k, &mut a, 1));
        report("19  v6 NH=2（grid 转置）", 2.0 * attn_bytes(&a, 2), d_v6_2);
        let d_v6_4 = best_of3(&k, iters, || run_v6(&k, &mut a, 2));
        report("20  v6 NH=4（grid 转置）", 2.0 * attn_bytes(&a, 4), d_v6_4);
        let run_v7 = |k: &Kernels, a: &mut Attn, which: usize| {
            let scale = 1.0f32 / (HD as f32).sqrt();
            k.qsa_attn_v7_probe(
                which, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
                &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
                T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks,
                COMPRESS.trailing_zeros() as usize, TOPK,
            )
            .unwrap();
        };
        let d_v7_1 = best_of3(&k, iters, || run_v7(&k, &mut a, 0));
        report("21  v7 NH=1（V smem 站台）", 2.0 * attn_bytes(&a, 1), d_v7_1);
        let d_v7_2 = best_of3(&k, iters, || run_v7(&k, &mut a, 1));
        report("22  v7 NH=2（V smem 站台）", 2.0 * attn_bytes(&a, 2), d_v7_2);
        let d_v7_4 = best_of3(&k, iters, || run_v7(&k, &mut a, 2));
        report("23  v7 NH=4（V smem 站台）", 2.0 * attn_bytes(&a, 4), d_v7_4);
        let run_v9 = |k: &Kernels, a: &mut Attn, which: usize| {
            let scale = 1.0f32 / (HD as f32).sqrt();
            k.qsa_attn_v9_probe(
                which, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
                &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
                T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks,
                COMPRESS.trailing_zeros() as usize, TOPK,
            )
            .unwrap();
        };
        let d_v9_1 = best_of3(&k, iters, || run_v9(&k, &mut a, 0));
        report("24  v9 NH=1（phys smem+j×2）", 2.0 * attn_bytes(&a, 1), d_v9_1);
        let d_v9_2 = best_of3(&k, iters, || run_v9(&k, &mut a, 1));
        report("25  v9 NH=2（phys smem+j×2）", 2.0 * attn_bytes(&a, 2), d_v9_2);
        let d_v9_4 = best_of3(&k, iters, || run_v9(&k, &mut a, 2));
        report("26  v9 NH=4（phys smem+j×2）", 2.0 * attn_bytes(&a, 4), d_v9_4);
        eprintln!(
            "[ak-bw] 场景 {scenario} 账：prod NH1 {:.0}｜NH4 {:.0}｜纯读NH4 {:.0}（{:.0}%）｜\
             ALU 地板 {:.0}（{:.0}%）｜pass1 {:.0} + pass2 {:.0}",
            d1 * 1e6, d4 * 1e6, d_rk4 * 1e6, 100.0 * d_rk4 / d4, d_ck4 * 1e6,
            100.0 * d_ck4 / d4, d_p1 * 1e6, d_p2 * 1e6
        );
    }
}

/// 逐位自检：t=512 形状下 prod NH=1/2/4 与 v2pf 候选必须**逐位相同**
/// （铁律：gather 归约次序 = 每输出元素按 idx 升序，NH 只合并取数不改次序）。
#[test]
fn t25_attn_prefill_bits() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    for scenario in ['A', 'B'] {
        let mut a = attn_bufs(&k, scenario);
        run_prod(&k, &mut a, 1);
        k.stream.synchronize().unwrap();
        let r: Vec<bf16> = k.stream.clone_dtoh(&a.out).unwrap();
        let span = TOPK * COMPRESS + COMPRESS;
        let mut spg = k.stream.alloc_zeros::<f32>(HEADS * T * span).unwrap();
        let mut all_ok = true;
        for (tag, which) in [
            ("prod NH=2", 99usize), ("prod NH=4", 98), ("v2pf NH=1", 14),
            ("v3 NH=1", 18), ("v3 NH=2", 19), ("v3 NH=4", 20),
            ("v5 NH=1", 97), ("v5 NH=2", 96), ("v5 NH=4", 95),
            ("v6 NH=1", 94), ("v6 NH=2", 93), ("v6 NH=4", 92),
            ("v7 NH=1", 91), ("v7 NH=2", 90), ("v7 NH=4", 89),
            ("v9 NH=1", 88), ("v9 NH=2", 87), ("v9 NH=4", 86),
        ] {
            a.out = k.stream.alloc_zeros::<bf16>(T * HEADS * HD).unwrap();
            match which {
                99 => run_prod(&k, &mut a, 2),
                98 => run_prod(&k, &mut a, 4),
                95..=97 => {
                    let w = 97 - which; // 97→0(NH1) 96→1(NH2) 95→2(NH4)
                    let scale = 1.0f32 / (HD as f32).sqrt();
                    k.qsa_attn_v5_probe(
                        w, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
                        &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
                        &mut spg.slice_mut(..),
                        T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks,
                        COMPRESS.trailing_zeros() as usize, TOPK,
                    )
                    .unwrap();
                }
                92..=94 => {
                    let w = 94 - which; // 94→0(NH1) 93→1(NH2) 92→2(NH4)
                    let scale = 1.0f32 / (HD as f32).sqrt();
                    k.qsa_attn_v6_probe(
                        w, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
                        &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
                        T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks,
                        COMPRESS.trailing_zeros() as usize, TOPK,
                    )
                    .unwrap();
                }
                89..=91 => {
                    let w = 91 - which; // 91→0(NH1) 90→1(NH2) 89→2(NH4)
                    let scale = 1.0f32 / (HD as f32).sqrt();
                    k.qsa_attn_v7_probe(
                        w, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
                        &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
                        T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks,
                        COMPRESS.trailing_zeros() as usize, TOPK,
                    )
                    .unwrap();
                }
                86..=88 => {
                    let w = 88 - which; // 88→0(NH1) 87→1(NH2) 86→2(NH4)
                    let scale = 1.0f32 / (HD as f32).sqrt();
                    k.qsa_attn_v9_probe(
                        w, &a.qn.slice(..), &a.kn.slice(..), &a.v.slice(..), &a.blk_tab.slice(..),
                        &a.sel.slice(..), &a.sel_cnt.slice(..), &mut a.out.slice_mut(..),
                        T, HEADS, KVH, HD, scale, a.vis_base, a.nblocks,
                        COMPRESS.trailing_zeros() as usize, TOPK,
                    )
                    .unwrap();
                }
                w => run_probe(&k, &mut a, w),
            }
            k.stream.synchronize().unwrap();
            let o: Vec<bf16> = k.stream.clone_dtoh(&a.out).unwrap();
            let nd = r.iter().zip(o.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            if nd > 0 {
                let pos = r.iter().zip(o.iter()).position(|(x, y)| x.to_bits() != y.to_bits()).unwrap();
                let (row, rem) = (pos / (HEADS * HD), pos % (HEADS * HD));
                eprintln!(
                    "[ak-bits] 场景 {scenario} {tag}：{nd} / {} 位不同；首个 @ row{row} head{} dim{}：ref={:?} got={:?}",
                    r.len(), rem / HD, rem % HD, r[pos], o[pos]
                );
            } else {
                eprintln!("[ak-bits] 场景 {scenario} {tag}：0 / {} 位不同", r.len());
            }
            if nd > 0 {
                all_ok = false;
            }
        }
        assert!(all_ok, "场景 {scenario} 存在不逐位相同的候选");
    }
}
