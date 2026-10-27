//! `lingxi_gemm_skinny_row_m6_bf16`（router 投影的 m≤6 特化核）与
//! `lingxi_gemm_skinny_row_bf16` 的**逐位对拍**。
//!
//! 判据：同一 (w, x) 下 logits 的**每一个 bf16 都相同**（router logits 是 top-k/softmax 的
//! 唯一输入，1 ulp 就可能翻名次 ⇒ 必须位级一致）。
//! 覆盖 m=2..6（生产 verify 批 rows=2..6）、多组随机 w/x、k=2560 与几个非 8 倍数的 k（走尾部）。

#![cfg(feature = "gpu")]

use cudarc::driver::{CudaSlice, DevicePtr};
use half::bf16;
use lingxi_kernels::Kernels;

fn rndbf16(n: usize, seed: u32, scale: f32) -> Vec<bf16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_f32(((s >> 8) as f32 / 16777216.0 - 0.5) * scale)
        })
        .collect()
}

fn run(k: &Kernels, w: &CudaSlice<bf16>, x: &CudaSlice<bf16>, m6: bool, n: usize, kk: usize, m: usize) -> Vec<bf16> {
    let mut y = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    let (wp, _g1) = w.device_ptr(&k.stream);
    let (xp, _g2) = x.device_ptr(&k.stream);
    let (yp, _g3) = y.device_ptr(&k.stream);
    if m6 {
        k.gemm_skinny_row_m6_bf16_ptr(wp, xp, yp, n, kk, m, 0, 0.0, 0.0).unwrap();
    } else {
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, n, kk, m, 0, 0.0, 0.0).unwrap();
    }
    k.stream.synchronize().unwrap();
    k.stream.clone_dtoh(&y).unwrap()
}

#[test]
fn t22_router_m6_bitid() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut total_bad = 0usize;
    let mut cases = 0usize;
    // ⚠️ k 必须是 8 的倍数（两核都用 `int4` 读 x 的 8 元素组 ⇒ 行基址要 16B 对齐；
    // 生产 hidden=2560 满足。k=2559 这种形状**两个核都会** MISALIGNED_ADDRESS）。
    for &(n, kk) in &[(512usize, 2560usize), (512, 2048), (256, 1024), (64, 640), (512, 5120)] {
        for m in 2..=6usize {
            for seed in [7u32, 11] {
                let w = k.stream.clone_htod(&rndbf16(n * kk, seed, 0.1)).unwrap();
                let x = k.stream.clone_htod(&rndbf16(m * kk, seed + 1, 4.0)).unwrap();
                let a = run(&k, &w, &x, false, n, kk, m);
                let b = run(&k, &w, &x, true, n, kk, m);
                let mut bad = 0usize;
                for i in 0..a.len() {
                    if a[i].to_bits() != b[i].to_bits() {
                        if bad < 4 {
                            eprintln!(
                                "[router-m6] n{n} k{kk} m{m} seed{seed} y[{i}] 行核={}({:04x}) m6核={}({:04x})",
                                a[i].to_f32(), a[i].to_bits(), b[i].to_f32(), b[i].to_bits()
                            );
                        }
                        bad += 1;
                    }
                }
                cases += 1;
                if bad > 0 {
                    eprintln!("[router-m6] ✗ n{n} k{kk} m{m} seed{seed}：{bad}/{} 不一致", a.len());
                    total_bad += bad;
                }
            }
        }
    }
    eprintln!(
        "[router-m6] {cases} 组（m=2..6 × 6 种 (n,k)，含 k 非 8 倍数的尾部）⇒ **不一致 {total_bad} 处**"
    );
    assert_eq!(total_bad, 0, "m6 特化核与行核不逐位相同");
}
