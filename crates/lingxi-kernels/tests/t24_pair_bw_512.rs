//! ** 勘察用**：pair 建表核（`lingxi_route_pairs_bf16`）在 prefill 形状
//! rows=512 / E=512 / top_k=10（S=5120 对）下的孤立 GPU 时间。
//!
//! 动机：batch_dev 设备路由扩 rows≤512 后整机 prefill 反而 +4s（双 V100 测试机
//! 干净环境 A/B 实锤），需要把差额钉到具体哪一发。本用例只量 pair 核 + 同 grid
//! 空核地板，与 t20 同口径（背靠背 best_of3）。
//!
//! 门控：
//! `cargo test --release -p lingxi-kernels --features gpu --test t24_pair_bw_512 -- --nocapture --test-threads=1`
#![cfg(feature = "gpu")]

use cudarc::driver::{DevicePtr, DevicePtrMut};
use lingxi_kernels::Kernels;

const ROWS: usize = 512;
const E: usize = 512;
const TOPK: usize = 10;

fn bench<F: FnMut()>(k: &Kernels, tag: &str, iters: usize, mut f: F) -> f64 {
    for _ in 0..20 {
        f();
    }
    k.stream.synchronize().unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        k.stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        best = best.min(us);
    }
    eprintln!("[pair512] {tag:<44} {best:8.1} µs");
    best
}

#[test]
fn t24_pair_bw_512() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut s = 1u32;
    let mut rnd = move || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s
    };
    // 每行 top_k 个**互异**专家（贴近真实路由）：行 r 选 (r*7 + j*53) % E
    let mut ids_h = vec![0i32; ROWS * TOPK];
    for r in 0..ROWS {
        for j in 0..TOPK {
            ids_h[r * TOPK + j] = ((r * 7 + j * 53) % E) as i32;
        }
    }
    let _ = rnd;
    let ids = k.stream.clone_htod(&ids_h).unwrap();
    let wts = k.stream.clone_htod(&vec![0.1f32; ROWS * TOPK]).unwrap();
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
    let zx = k.stream.alloc_zeros::<f32>(1).unwrap();
    let mut zy = k.stream.alloc_zeros::<f32>(1).unwrap();

    let pair = {
        let (sp, _g) = slot_map.device_ptr(&k.stream);
        let (cd, _g2) = counts.device_ptr(&k.stream);
        let (hp, _g3) = hit_pri.device_ptr(&k.stream);
        let (ha, _g4) = hit_alt.device_ptr(&k.stream);
        let (mi, _g5) = miss_ids.device_ptr(&k.stream);
        let (ga, _g6) = gslots_alt.device_ptr(&k.stream);
        bench(&k, "pair route_pairs(E=512,rows=512,k=10)", 200, || {
            k.route_pairs_bf16(
                &ids, &wts, sp, 0, 0, &mut pair_rows, &mut pair_wts, &mut gslots, ga,
                &mut goff, &mut uniq, &mut col, &mut roff, cd, hp, ha, mi,
                &mut pri_gidx, &mut alt_gidx, E, ROWS, TOPK,
            )
            .unwrap();
        })
    };
    let floor = bench(&k, "  ↑空核同 grid(1,1) blk512", 1000, || {
        k.empty_probe_scale(&zx, &mut zy, (1, 1), 512).unwrap();
    });
    eprintln!("[pair512]   ⇒ pair 净活 {:.1} µs", pair - floor);
}
