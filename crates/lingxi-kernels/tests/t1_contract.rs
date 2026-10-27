//! T1 kernel 契约测试（验证体系）：随机输入 vs torch 参考（同一 pocenv 环境），rel < 2e-3。
//!
//! 门控：`cargo test -p lingxi-kernels --features gpu`
//! 环境：`LINGXI_PYTHON` 指向 pocenv 的 python（默认 /models/pocenv/bin/python）。

#![cfg(feature = "gpu")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn python() -> String {
    std::env::var("LINGXI_PYTHON").unwrap_or_else(|_| "/models/pocenv/bin/python".to_string())
}

fn oracle_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lingxi_t1_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run_oracle(dir: &Path, op: &str, params: &str) {
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/golden/kernel_oracle.py");
    let status = Command::new(python())
        .arg(script)
        .arg(dir)
        .arg(op)
        .arg(params)
        .status()
        .expect("无法启动 pocenv python");
    assert!(status.success(), "oracle 生成失败: op={op}");
}

fn read_f32(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("读 {} 失败: {e}", p.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// 相对 L2 误差
fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let num: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
    let den: f32 = want.iter().map(|b| b * b).sum::<f32>().sqrt();
    num / den.max(1e-12)
}

const TOL: f32 = 2e-3;

#[test]
fn t1_smoke_scale() {
    let k = lingxi_kernels::Kernels::load(0).expect("CUDA 初始化失败");
    let n = 4096usize;
    let a = 1.75f32;
    let dir = oracle_dir("smoke_scale");
    run_oracle(&dir, "smoke_scale", &format!(r#"{{"n":{n},"a":{a},"seed":1}}"#));

    let x = read_f32(&dir.join("in_x.f32"));
    let want = read_f32(&dir.join("ref.f32"));

    let dx = k.stream.clone_htod(&x).unwrap();
    let mut dy = k.stream.alloc_zeros::<f32>(n).unwrap();
    k.smoke_scale(&dx, a, &mut dy, n as u32).unwrap();
    let got = k.stream.clone_dtoh(&dy).unwrap();

    let rel = rel_l2(&got, &want);
    assert!(rel < TOL, "smoke_scale rel={rel} >= {TOL}");
    println!("T1 smoke_scale rel = {rel:.2e}");
}

#[test]
fn t1_rmsnorm_f32() {
    let k = lingxi_kernels::Kernels::load(0).expect("CUDA 初始化失败");
    let rows = 64usize;
    let dim = 2560usize;
    let eps = 1e-6f32;
    let dir = oracle_dir("rmsnorm");
    run_oracle(&dir, "rmsnorm", &format!(r#"{{"rows":{rows},"dim":{dim},"eps":{eps},"seed":2}}"#));

    let x = read_f32(&dir.join("in_x.f32"));
    let gamma = read_f32(&dir.join("in_gamma.f32"));
    let want = read_f32(&dir.join("ref.f32"));

    let dx = k.stream.clone_htod(&x).unwrap();
    let dg = k.stream.clone_htod(&gamma).unwrap();
    let mut dy = k.stream.alloc_zeros::<f32>(rows * dim).unwrap();
    k.rmsnorm_f32(&dx, &dg, &mut dy, rows as u32, dim, eps).unwrap();
    let got = k.stream.clone_dtoh(&dy).unwrap();

    let rel = rel_l2(&got, &want);
    assert!(rel < TOL, "rmsnorm rel={rel} >= {TOL}");
    println!("T1 rmsnorm_f32 rel = {rel:.2e}");
}

/// cuBLAS 用"带偏移的 view"当操作数是否正确（MoE 专家权重就靠这个）。
#[test]
fn t1_gemm_view_offset() {
    use half::bf16;
    let k = lingxi_kernels::Kernels::load(0).expect("CUDA 初始化失败");
    for m in [3usize, 1usize] {
    let (n, kk) = (4usize, 5usize);
    let a: Vec<f32> = (0..m * kk).map(|i| (i % 7) as f32 * 0.1).collect();
    // W 是 [2N, K] 的大张量，第二个专家块在行 N..2N
    let big: Vec<f32> = (0..2 * n * kk).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
    let a_bf: Vec<bf16> = a.iter().map(|v| bf16::from_f32(*v)).collect();
    let big_bf: Vec<bf16> = big.iter().map(|v| bf16::from_f32(*v)).collect();
    let da = k.stream.clone_htod(&a_bf).unwrap();
    let dbig = k.stream.clone_htod(&big_bf).unwrap();

    let expect = |off: usize| -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for p in 0..kk {
                    s += big[(off + j) * kk + p] * a[i * kk + p];
                }
                c[i * n + j] = s;
            }
        }
        c
    };

    for off in [0usize, n] {
        let w_view = dbig.slice(off * kk..(off + n) * kk);
        let mut dc = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
        k.gemm_bf16_nt(&da, &w_view, &mut dc, m as u32, n as u32, kk as u32).unwrap();
        let got: Vec<f32> = k.stream.clone_dtoh(&dc).unwrap().iter().map(|v| v.to_f32()).collect();
        let want = expect(off);
        println!("  m={m} off={off}: got={got:?}");
        println!("            want={want:?}");
        assert!(rel_l2(&got, &want) < 2e-2, "带偏移 view 的 gemm 错（m={m} off={off}）");
    }
    }
    println!("T1 gemm_view_offset OK");
}
#[test]
fn t1_gather_scatter_rows() {
    use half::bf16;
    let k = lingxi_kernels::Kernels::load(0).expect("CUDA 初始化失败");
    let (m_src, kk, m) = (6usize, 4usize, 3usize);
    let src_h: Vec<f32> = (0..m_src * kk).map(|i| i as f32).collect();
    // 取 src 的第 4、1、5 行
    let idx_h: Vec<i32> = vec![4, 1, 5];
    let w_h: Vec<f32> = vec![0.5, 2.0, 1.0];

    let src_bf: Vec<bf16> = src_h.iter().map(|v| bf16::from_f32(*v)).collect();
    let dsrc = k.stream.clone_htod(&src_bf).unwrap();
    let didx = k.stream.clone_htod(&idx_h).unwrap();
    let dw = k.stream.clone_htod(&w_h).unwrap();

    let mut dg = k.stream.alloc_zeros::<bf16>(m * kk).unwrap();
    k.gather_rows_bf16(&dsrc, &didx, &mut dg, m, kk, 0).unwrap();
    let got_g: Vec<f32> = k.stream.clone_dtoh(&dg).unwrap().iter().map(|v| v.to_f32()).collect();
    let want_g: Vec<f32> = vec![16., 17., 18., 19., 4., 5., 6., 7., 20., 21., 22., 23.];
    assert!(rel_l2(&got_g, &want_g) < 1e-6, "gather 错: {got_g:?}");

    let mut dst = k.stream.alloc_zeros::<bf16>(m_src * kk).unwrap();
    k.scatter_add_rows_bf16(&mut dst, &didx, &dw, &dg, m, kk, 0).unwrap();
    let got_s: Vec<f32> = k.stream.clone_dtoh(&dst).unwrap().iter().map(|v| v.to_f32()).collect();
    let mut want_s = vec![0f32; m_src * kk];
    for i in 0..m {
        for c in 0..kk {
            want_s[idx_h[i] as usize * kk + c] += w_h[i] * want_g[i * kk + c];
        }
    }
    assert!(rel_l2(&got_s, &want_s) < 1e-6, "scatter 错: got={got_s:?} want={want_s:?}");
    println!("T1 gather/scatter OK");
}
