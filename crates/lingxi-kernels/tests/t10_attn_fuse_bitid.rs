//! **attn 段两处"合发"的位级对拍 + 计时**（2026-10-24）。
//!
//! 对应生产改动：
//! 1. QSA 的 `q | k | gate | v` 四发 → 一发（`lingxi_gemm_q4_skinny4_bf16`）；
//! 2. GDN 的 `qkvz`(Q4) + `a/b`(bf16) → 一发（`lingxi_gemm_q4_skinny_ab_bf16`）。
//!
//! 判据：**逐位相等**（bf16 的 16 位原样比），不是"误差很小"。
//! 形状取线上真值：hidden=2560、QSA 13312 行（6144/512/6144/512）、
//! GDN qkvz 10240 行、a/b 各 48 行、m=5 与 m=6（投机验证批）。
//!
//! `cargo test --release -p lingxi-kernels --features gpu --test t10_attn_fuse_bitid -- --nocapture`

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

const HIDDEN: usize = 2560;

fn time_launches(k: &Kernels, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    k.stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    k.stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn rndbf16(n: usize, seed: u32) -> Vec<bf16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_f32(((s >> 8) as f32 / 16777216.0 - 0.5) * 0.1)
        })
        .collect()
}

/// 稠密 Q4G64 权重：payload = 随机 nibble 的 base 平面 + scale 平面（scale 全填 1.0）。
fn q4w_seeded(k: &Kernels, rows: usize, kdim: usize) -> (CudaSlice<u8>, i32, u64) {
    let k_pad = ((kdim + 127) / 128 * 128) as i32;
    let row_bytes = k_pad as usize / 2;
    let base = rows * row_bytes;
    let scale_off = ((base + 255) / 256 * 256) as u64;
    let nsc = rows * (k_pad as usize / 64);
    let total = scale_off as usize + nsc * 2;
    let mut buf = vec![0u8; total];
    let mut s: u32 = 0x1234_5678;
    for b in buf.iter_mut().take(base) {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 13) as u8;
    }
    // half 1.0 = 0x3C00（小端两字节）
    for i in 0..nsc {
        buf[scale_off as usize + 2 * i] = 0x00;
        buf[scale_off as usize + 2 * i + 1] = 0x3C;
    }
    (k.stream.clone_htod(&buf).unwrap(), k_pad, scale_off)
}

fn bits_of(k: &Kernels, y: &CudaSlice<bf16>) -> Vec<u16> {
    k.stream.clone_dtoh(y).unwrap().iter().map(|v| v.to_bits()).collect()
}

/// 逐段比较并打印（返回不一致数）。
fn cmp(tag: &str, a: &[u16], b: &[u16]) -> usize {
    assert_eq!(a.len(), b.len(), "{tag}: 长度不同");
    let bad = a.iter().zip(b).filter(|(x, y)| x != y).count();
    if bad == 0 {
        eprintln!("[fuse-bitid] {tag:<34} 0/{} 不一致 ✓", a.len());
    } else {
        let first = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
        eprintln!(
            "[fuse-bitid] {tag:<34} **{bad}/{} 不一致** ✗（首个 @{first}: {:#06x} vs {:#06x}）",
            a.len(),
            a[first],
            b[first]
        );
    }
    bad
}

/// 1) QSA：4 发 vs 1 发。
fn qsa_case(k: &Kernels, m: usize) -> usize {
    let (q_dim, kv_dim) = (6144usize, 512usize);
    let ns = [q_dim, kv_dim, q_dim, kv_dim];
    let rows = 2 * q_dim + 2 * kv_dim; // 13312
    let (w, k_pad, scale_off) = q4w_seeded(k, rows, HIDDEN);
    let x = k.stream.clone_htod(&rndbf16(m * HIDDEN, 7)).unwrap();

    let mut old: Vec<CudaSlice<bf16>> =
        ns.iter().map(|&n| k.stream.alloc_zeros::<bf16>(m * n).unwrap()).collect();
    let mut new: Vec<CudaSlice<bf16>> =
        ns.iter().map(|&n| k.stream.alloc_zeros::<bf16>(m * n).unwrap()).collect();

    let (wp, _g0) = w.device_ptr(&k.stream);
    let (xp, _g1) = x.device_ptr(&k.stream);
    let offs = [0usize, ns[0], ns[0] + ns[1], ns[0] + ns[1] + ns[2]];
    let mut oldp = [0u64; 4];
    let mut newp = [0u64; 4];
    {
        // 指针守卫只在这个块里活着 —— 出去后才能把 old/new 借给 `bits_of` 读回。
        let mut guards = Vec::new();
        for (i, y) in old.iter_mut().enumerate() {
            let (p, g) = y.device_ptr_mut(&k.stream);
            oldp[i] = p;
            guards.push(g);
        }
        for (i, y) in new.iter_mut().enumerate() {
            let (p, g) = y.device_ptr_mut(&k.stream);
            newp[i] = p;
            guards.push(g);
        }
        for (i, &n) in ns.iter().enumerate() {
            k.gemm_q4_skinny_bf16_ptr(
                wp, xp, oldp[i], n, HIDDEN, k_pad, m, offs[i], scale_off, 0, 0.0, 0.0,
            )
            .unwrap();
        }
        k.gemm_q4_skinny4_bf16_ptr(
            wp, xp, newp[0], newp[1], newp[2], newp[3],
            ns[0], ns[1], ns[2], ns[3], HIDDEN, k_pad, m, 0, scale_off,
        )
        .unwrap();
        k.stream.synchronize().unwrap();
    }

    let names = ["q", "k", "gate", "v"];
    let mut bad = 0;
    for i in 0..4 {
        let a = bits_of(k, &old[i]);
        let b = bits_of(k, &new[i]);
        bad += cmp(&format!("QSA m={m} {}", names[i]), &a, &b);
    }
    let run_old = || {
        for (i, &n) in ns.iter().enumerate() {
            k.gemm_q4_skinny_bf16_ptr(
                wp, xp, oldp[i], n, HIDDEN, k_pad, m, offs[i], scale_off, 0, 0.0, 0.0,
            )
            .unwrap();
        }
    };
    let run_new = || {
        k.gemm_q4_skinny4_bf16_ptr(
            wp, xp, newp[0], newp[1], newp[2], newp[3],
            ns[0], ns[1], ns[2], ns[3], HIDDEN, k_pad, m, 0, scale_off,
        )
        .unwrap();
    };
    let dt_old = time_launches(k, 100, run_old);
    let dt_new = time_launches(k, 100, run_new);
    eprintln!(
        "[fuse-bitid] QSA m={m}: 四发 **{dt_old:.1} µs** → 一发 **{dt_new:.1} µs**（{:+.1} µs, {:+.0}%）",
        dt_new - dt_old,
        (dt_new / dt_old - 1.0) * 100.0
    );
    bad
}

/// 2) GDN：qkvz(Q4) + a + b（3 发）vs 一发。
fn gdn_case(k: &Kernels, m: usize) -> usize {
    let (conv_dim, nvh) = (10240usize, 48usize);
    let (w, k_pad, scale_off) = q4w_seeded(k, 16384, HIDDEN);
    let wab = k.stream.clone_htod(&rndbf16(2 * nvh * HIDDEN, 11)).unwrap();
    let x = k.stream.clone_htod(&rndbf16(m * HIDDEN, 13)).unwrap();

    let mut qkv_old = k.stream.alloc_zeros::<bf16>(m * conv_dim).unwrap();
    let mut a_old = k.stream.alloc_zeros::<bf16>(m * nvh).unwrap();
    let mut b_old = k.stream.alloc_zeros::<bf16>(m * nvh).unwrap();
    let mut qkv_new = k.stream.alloc_zeros::<bf16>(m * conv_dim).unwrap();
    let mut a_new = k.stream.alloc_zeros::<bf16>(m * nvh).unwrap();
    let mut b_new = k.stream.alloc_zeros::<bf16>(m * nvh).unwrap();

    let (wp, _g0) = w.device_ptr(&k.stream);
    let (xp, _g1) = x.device_ptr(&k.stream);
    let (wabp, _g2) = wab.device_ptr(&k.stream);
    let mut p = [0u64; 6]; // [qkv_old, a_old, b_old, qkv_new, a_new, b_new]
    {
        let mut guards = Vec::new();
        for (i, y) in [&mut qkv_old, &mut a_old, &mut b_old, &mut qkv_new, &mut a_new, &mut b_new]
            .into_iter()
            .enumerate()
        {
            let (pp, g) = y.device_ptr_mut(&k.stream);
            p[i] = pp;
            guards.push(g);
        }
        // 旧路：`proj_q4_epi(qkvz)` + 两次 `proj_t1(a/b)` = `gemm_skinny_row`（n=48 各一发）
        k.gemm_q4_skinny_bf16_ptr(
            wp, xp, p[0], conv_dim, HIDDEN, k_pad, m, 0, scale_off, 0, 0.0, 0.0,
        )
        .unwrap();
        k.gemm_skinny_row_bf16_ptr(wabp, xp, p[1], nvh, HIDDEN, m, 0, 0.0, 0.0).unwrap();
        k.gemm_skinny_row_bf16_ptr(
            wabp + (nvh * HIDDEN * 2) as u64, xp, p[2], nvh, HIDDEN, m, 0, 0.0, 0.0,
        )
        .unwrap();
        // 新路：一发
        k.gemm_q4_skinny_ab_bf16_ptr(
            wp, xp, p[3], wabp, p[4], p[5], conv_dim, nvh, HIDDEN, k_pad, m, 0, scale_off,
        )
        .unwrap();
        k.stream.synchronize().unwrap();
    }

    let mut bad = 0;
    bad += cmp(&format!("GDN m={m} qkvz(10240)"), &bits_of(k, &qkv_old), &bits_of(k, &qkv_new));
    bad += cmp(&format!("GDN m={m} a(48)"), &bits_of(k, &a_old), &bits_of(k, &a_new));
    bad += cmp(&format!("GDN m={m} b(48)"), &bits_of(k, &b_old), &bits_of(k, &b_new));
    let run_old = || {
        k.gemm_q4_skinny_bf16_ptr(
            wp, xp, p[0], conv_dim, HIDDEN, k_pad, m, 0, scale_off, 0, 0.0, 0.0,
        )
        .unwrap();
        k.gemm_skinny_row_bf16_ptr(wabp, xp, p[1], nvh, HIDDEN, m, 0, 0.0, 0.0).unwrap();
        k.gemm_skinny_row_bf16_ptr(
            wabp + (nvh * HIDDEN * 2) as u64, xp, p[2], nvh, HIDDEN, m, 0, 0.0, 0.0,
        )
        .unwrap();
    };
    let run_new = || {
        k.gemm_q4_skinny_ab_bf16_ptr(
            wp, xp, p[3], wabp, p[4], p[5], conv_dim, nvh, HIDDEN, k_pad, m, 0, scale_off,
        )
        .unwrap();
    };
    let dt_old = time_launches(k, 100, run_old);
    let dt_new = time_launches(k, 100, run_new);
    eprintln!(
        "[fuse-bitid] GDN m={m}: qkvz+a+b 三发 **{dt_old:.1} µs** → 一发 **{dt_new:.1} µs**（{:+.1} µs）",
        dt_new - dt_old
    );
    let dt_a = time_launches(k, 100, || {
        k.gemm_skinny_row_bf16_ptr(wabp, xp, p[1], nvh, HIDDEN, m, 0, 0.0, 0.0).unwrap();
    });
    eprintln!("[fuse-bitid]   · 单发 a（n=48, m={m}）**{dt_a:.1} µs** ← 被省掉的那一发");
    bad
}

#[test]
fn qsa_gdn_fuse_bit_exact() {
    let k = Kernels::load(0).expect("CUDA 设备 0");
    let mut bad = 0;
    for m in [5usize, 6] {
        bad += qsa_case(&k, m);
    }
    for m in [5usize, 6] {
        bad += gdn_case(&k, m);
    }
    assert_eq!(bad, 0, "合发核与原核输出**不是逐位相同**（{bad} 个元素不同）");
}
