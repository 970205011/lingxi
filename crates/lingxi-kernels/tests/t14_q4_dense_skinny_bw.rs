//! **稠密 Q4 瘦长 GEMM 的取数结构微基准**（2026-10-24，）。
//!
//! 专家段 Q4/Q3 GEMV 只换取数结构（算式一字未改、逐位相同）就白捡 1.6~2.0×
//! （/）。稠密投影是**另一个核**（`gemm_q4_skinny_bf16`，m>1 的瘦长 GEMM）
//! ——实测 m=6 只有 125~132 GB/s（B 表）。本用例先量它的**分母**：
//!
//! 1. 现状核 `gemm_q4_skinny_bf16_ptr`（m=5/6）；
//! 2. 同形纯读（`q4_dense_exp.cu` 的 `q4_dense_probe`，grid/block/地址映射与生产核逐字节相同）：
//! mode 0 = 4 B/lane（生产取数形状）/ 1 = 16 B/lane `uint4`（结构上限）；
//! 3. 纯读 + nibble 解包 + I2F（**ALU 地板**，mode 2）；
//! 4. 归因用：mode 4 = staging + 纯读（隔离 x 舞台/同步）、mode 5 = 解包 + FFMA（x 用常量，
//! 隔离 smem x 读）、mode 3 = 生产内层全量（应与现状核同速，作自检）。
//!
//! 口径：GB/s 按**权重 payload 字节**（n·k/2）折算，与 的表相同。
//! **`--test-threads=1` 是硬要求**（并行只剩 ~1/3 带宽，见 D）。
//!
//! `bash
//! export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib   # cublas/nvrtc 不在默认路径时按需追加
//! cargo test --release -p lingxi-kernels --features gpu --test t14_q4_dense_skinny_bw \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

/// 稠密 Q4G64 权重（`[rows, kdim]`，base 平面 + 对齐 256 的 fp16 scale 平面）。
/// 布局与 `lingxi-model/src/q4.rs` 的 `quantize_bf16_bits` 完全一致。
struct Q4 {
    packed: CudaSlice<u8>,
    k_pad: i32,
    scale_off: u64,
}

fn q4w(k: &Kernels, rows: usize, kdim: usize) -> Q4 {
    let k_pad = ((kdim + 127) / 128 * 128) as i32;
    let row_bytes = k_pad as usize / 2;
    let base = rows * row_bytes;
    let scale_off = ((base + 255) / 256 * 256) as u64;
    let groups = k_pad as usize / 64;
    let total = scale_off as usize + rows * groups * 2;
    let mut host = vec![0u8; total];
    // code 平面：xorshift 随机字节（0..15 半字节天然合法）
    let mut s: u32 = 0x1234_5678;
    for v in host.iter_mut().take(base) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *v = (s & 0xFF) as u8;
    }
    // scale 平面：fp16 0.002~0.024（随机字节的 fp16 多为 NaN/denorm，会让 FMA 走慢路）
    for i in 0..rows * groups {
        let h = half::f16::from_f32(0.002 + 0.0005 * ((i % 45) as f32));
        let off = scale_off as usize + 2 * i;
        host[off..off + 2].copy_from_slice(&h.to_le_bytes());
    }
    Q4 { packed: k.stream.clone_htod(&host).unwrap(), k_pad, scale_off }
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

fn report(tag: &str, bytes: f64, dt: f64, extra: &str) {
    eprintln!(
        "[q4d-bw] {tag:<42}｜{:8.2} µs｜{:7.1} GB/s｜{:6.2} MB{extra}",
        dt * 1e6,
        bytes / dt / 1e9,
        bytes / 1e6
    );
}

/// 现状核（生产档：`gemm_q4_skinny_bf16`，epi=0、row_off=0）。
fn bench_prod(k: &Kernels, tag: &str, w: &Q4, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
              n: usize, kdim: usize, m: usize, bytes: f64) {
    let (wp, _a) = w.packed.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let dt = time_it(k, 40, || {
        k.gemm_q4_skinny_bf16_ptr(wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off, 0, 0.0, 0.0)
            .unwrap();
    });
    report(tag, bytes, dt, "");
}

/// **背靠背 A/B**：同一条命令里交替跑旧核 / R2 生产核（各 3 轮 × 30 迭代，取中位）。
fn bench_ab_prod(k: &Kernels, w: &Q4, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
                 n: usize, kdim: usize, m: usize, bytes: f64) {
    let (wp, _a) = w.packed.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let mut old = Vec::new();
    let mut new = Vec::new();
    for _ in 0..3 {
        old.push(time_it(k, 30, || {
            k.gemm_q4_skinny_bf16_ptr(wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off, 0, 0.0, 0.0)
                .unwrap();
        }));
        new.push(time_it(k, 30, || {
            k.gemm_q4_skinny_r2_bf16_ptr(
                wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off, 0, 0.0, 0.0,
            )
            .unwrap();
        }));
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (mo, mn) = (med(&mut old), med(&mut new));
    eprintln!(
        "[q4d-ab] n{n} k{kdim} m{m}｜旧 {:6.1} µs（{:6.1} GB/s）｜R2 {:6.1} µs（{:6.1} GB/s）｜**{:.2}×**",
        mo * 1e6,
        bytes / mo / 1e9,
        mn * 1e6,
        bytes / mn / 1e9,
        mo / mn
    );
}

/// 同形探针。
fn bench_probe(k: &Kernels, tag: &str, w: &Q4, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
               n: usize, kdim: usize, m: usize, bytes: f64, mode: i32) {
    let (wp, _a) = w.packed.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let dt = time_it(k, 40, || {
        k.q4_dense_probe(wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off, mode).unwrap();
    });
    report(tag, bytes, dt, "");
}

/// 逐位自检：mode 3（探针里逐字复刻的生产内层）vs 现状核。
fn check_mode3(k: &Kernels, w: &Q4, n: usize, kdim: usize, m: usize) {
    let x = k.stream.clone_htod(&make_x(m, kdim)).unwrap();
    let mut yr = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    let mut yc = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    let (wp, _a) = w.packed.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    {
        let (yp, _c) = yr.device_ptr_mut(&k.stream);
        k.gemm_q4_skinny_bf16_ptr(wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off, 0, 0.0, 0.0)
            .unwrap();
    }
    {
        let (yp, _c) = yc.device_ptr_mut(&k.stream);
        k.q4_dense_probe(wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off, 3).unwrap();
    }
    k.stream.synchronize().unwrap();
    let hr: Vec<bf16> = k.stream.clone_dtoh(&yr).unwrap();
    let hc: Vec<bf16> = k.stream.clone_dtoh(&yc).unwrap();
    let nd = hr.iter().zip(hc.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    eprintln!("[q4d-bw] 自检 mode3 vs 现状核（n{n} k{kdim} m{m}）：{nd} / {} 元素不同", hr.len());
    assert_eq!(nd, 0, "探针的 mode 3 与现状核不逐位相同");
}

/// 变体表：(下标, 名字)。
const VARIANTS: [(usize, &str); 5] = [
    (0, "R1W8"),
    (1, "R2W8"),
    (2, "R4W8"),
    (3, "R1W16"),
    (4, "R2W16"),
];

/// 逐位自检：多行变体 vs 现状核。
fn check_rows(k: &Kernels, w: &Q4, n: usize, kdim: usize, m: usize) {
    let x = k.stream.clone_htod(&make_x(m, kdim)).unwrap();
    let mut yr = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    {
        let (wp, _a) = w.packed.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = yr.device_ptr_mut(&k.stream);
        k.gemm_q4_skinny_bf16_ptr(wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off, 0, 0.0, 0.0)
            .unwrap();
    }
    k.stream.synchronize().unwrap();
    let hr: Vec<bf16> = k.stream.clone_dtoh(&yr).unwrap();
    for &(which, nm) in VARIANTS.iter() {
        let mut yc = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
        {
            let (wp, _a) = w.packed.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            let (yp, _c) = yc.device_ptr_mut(&k.stream);
            k.q4_dense_rows_probe(which, wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off).unwrap();
        }
        k.stream.synchronize().unwrap();
        let hc: Vec<bf16> = k.stream.clone_dtoh(&yc).unwrap();
        let nd = hr.iter().zip(hc.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!(
            "[q4d-bw] 自检 {nm} vs 现状核（n{n} k{kdim} m{m}）：{nd} / {} 元素不同",
            hr.len()
        );
        assert_eq!(nd, 0, "多行变体 {nm} 与现状核不逐位相同");
    }
}

/// 多行变体。
fn bench_rows(k: &Kernels, w: &Q4, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
              n: usize, kdim: usize, m: usize, bytes: f64, which: usize, nm: &str) {
    let (wp, _a) = w.packed.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let dt = time_it(k, 40, || {
        k.q4_dense_rows_probe(which, wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off).unwrap();
    });
    report(&format!("⑨ 多行 {nm}"), bytes, dt, "");
}

fn make_x(m: usize, k: usize) -> Vec<bf16> {
    (0..m * k)
        .map(|i| bf16::from_f32(((i * 37 % 101) as f32 / 101.0) - 0.5))
        .collect()
}

/// 生产三发（旧核 vs R2 档）的逐位对照：
/// 单段（= z/out 投影）、四段（QSA q|k|gate|v 合发）、`qkvz + a/b` 尾巴（GDN 合发）。
/// ⚠️ 测试进程里 **强制 `LINGXI_NO_Q4D_R2=1`** ⇒ `gemm_q4_*_bf16_ptr` 走旧核；
/// R2 走显式入口（`OnceLock` 一旦读到开关就不能在同一进程里换）。
fn check_prod(k: &Kernels) {
    let (m, kdim) = (6usize, 2560usize);
    // ── 单段 ──
    let n = 1000usize; // 非 16 的整数倍：覆盖尾巴 block 的 ok[r] 掩码
    let w = q4w(&k, n, kdim);
    let x = k.stream.clone_htod(&make_x(m, kdim)).unwrap();
    let mut y_old = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    let mut y_new = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
    let (wp, _a) = w.packed.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    {
        let (yp, _c) = y_old.device_ptr_mut(&k.stream);
        k.gemm_q4_skinny_bf16_ptr(wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off, 0, 0.0, 0.0)
            .unwrap();
    }
    {
        let (yp, _c) = y_new.device_ptr_mut(&k.stream);
        k.gemm_q4_skinny_r2_bf16_ptr(wp, xp, yp, n, kdim, w.k_pad, m, 0, w.scale_off, 0, 0.0, 0.0)
            .unwrap();
    }
    k.stream.synchronize().unwrap();
    let a: Vec<bf16> = k.stream.clone_dtoh(&y_old).unwrap();
    let b: Vec<bf16> = k.stream.clone_dtoh(&y_new).unwrap();
    let nd = a.iter().zip(b.iter()).filter(|(u, v)| u.to_bits() != v.to_bits()).count();
    eprintln!("[q4d-bw] 生产对照 单段 R2 vs 旧核（n{n} k{kdim} m{m}）：{nd} / {} 元素不同", a.len());
    assert_eq!(nd, 0, "单段 R2 与旧核不逐位相同");

    // ── 四段（QSA 线上形状 6144/512/6144/512，+ row_off 非零）──
    let ns = [6144usize, 512, 6144, 512];
    let w4 = q4w(&k, 13312 + 256, kdim);
    let (wp4, _a4) = w4.packed.device_ptr(&k.stream);
    let mut yo: Vec<CudaSlice<bf16>> =
        ns.iter().map(|&nn| k.stream.alloc_zeros::<bf16>(m * nn).unwrap()).collect();
    let mut yn: Vec<CudaSlice<bf16>> =
        ns.iter().map(|&nn| k.stream.alloc_zeros::<bf16>(m * nn).unwrap()).collect();
    let ptrs = |v: &mut Vec<CudaSlice<bf16>>| -> [u64; 4] {
        [0, 1, 2, 3].map(|i| v[i].device_ptr_mut(&k.stream).0)
    };
    let po = ptrs(&mut yo);
    let pn = ptrs(&mut yn);
    k.gemm_q4_skinny4_bf16_ptr(
        wp4, xp, po[0], po[1], po[2], po[3], ns[0], ns[1], ns[2], ns[3], kdim, w4.k_pad, m, 0,
        w4.scale_off,
    )
    .unwrap();
    k.gemm_q4_skinny4_r2_bf16_ptr(
        wp4, xp, pn[0], pn[1], pn[2], pn[3], ns[0], ns[1], ns[2], ns[3], kdim, w4.k_pad, m, 0,
        w4.scale_off,
    )
    .unwrap();
    k.stream.synchronize().unwrap();
    for i in 0..4 {
        let a: Vec<bf16> = k.stream.clone_dtoh(&yo[i]).unwrap();
        let b: Vec<bf16> = k.stream.clone_dtoh(&yn[i]).unwrap();
        let nd = a.iter().zip(b.iter()).filter(|(u, v)| u.to_bits() != v.to_bits()).count();
        eprintln!(
            "[q4d-bw] 生产对照 四段 R2 vs 旧核（段{i} n{} k{kdim} m{m}）：{nd} / {} 元素不同",
            ns[i],
            a.len()
        );
        assert_eq!(nd, 0, "四段 R2 段{i} 与旧核不逐位相同");
    }

    // ── qkvz + a/b 尾巴（GDN 线上形状 n_q4=10240、nvh=48）──
    let (n_q4, nvh) = (10240usize, 48usize);
    let wq4 = q4w(&k, n_q4, kdim);
    let wab = k
        .stream
        .clone_htod(&(0..96 * kdim).map(|i| bf16::from_f32((i % 37) as f32 / 37.0)).collect::<Vec<_>>())
        .unwrap();
    let mut yq_o = k.stream.alloc_zeros::<bf16>(m * n_q4).unwrap();
    let mut ya_o = k.stream.alloc_zeros::<bf16>(m * nvh).unwrap();
    let mut yb_o = k.stream.alloc_zeros::<bf16>(m * nvh).unwrap();
    let mut yq_n = k.stream.alloc_zeros::<bf16>(m * n_q4).unwrap();
    let mut ya_n = k.stream.alloc_zeros::<bf16>(m * nvh).unwrap();
    let mut yb_n = k.stream.alloc_zeros::<bf16>(m * nvh).unwrap();
    let (wpq, _aq) = wq4.packed.device_ptr(&k.stream);
    let (wabp, _ab) = wab.device_ptr(&k.stream);
    {
        let (p0, _0) = yq_o.device_ptr_mut(&k.stream);
        let (p1, _1) = ya_o.device_ptr_mut(&k.stream);
        let (p2, _2) = yb_o.device_ptr_mut(&k.stream);
        k.gemm_q4_skinny_ab_bf16_ptr(wpq, xp, p0, wabp, p1, p2, n_q4, nvh, kdim, wq4.k_pad, m, 0,
                                     wq4.scale_off)
            .unwrap();
    }
    {
        let (p0, _0) = yq_n.device_ptr_mut(&k.stream);
        let (p1, _1) = ya_n.device_ptr_mut(&k.stream);
        let (p2, _2) = yb_n.device_ptr_mut(&k.stream);
        k.gemm_q4_skinny_ab_r2_bf16_ptr(wpq, xp, p0, wabp, p1, p2, n_q4, nvh, kdim, wq4.k_pad, m,
                                        0, wq4.scale_off)
            .unwrap();
    }
    k.stream.synchronize().unwrap();
    for (nm, o, nn) in [
        ("qkvz", &yq_o, &yq_n),
        ("a", &ya_o, &ya_n),
        ("b", &yb_o, &yb_n),
    ] {
        let a: Vec<bf16> = k.stream.clone_dtoh(o).unwrap();
        let b: Vec<bf16> = k.stream.clone_dtoh(nn).unwrap();
        let nd = a.iter().zip(b.iter()).filter(|(u, v)| u.to_bits() != v.to_bits()).count();
        eprintln!("[q4d-bw] 生产对照 qkvz+a/b R2 vs 旧核（{nm}）：{nd} / {} 元素不同", a.len());
        assert_eq!(nd, 0, "qkvz+a/b R2 的 {nm} 与旧核不逐位相同");
    }
}

#[test]
fn t14_q4_dense_skinny_bw() {
    // 同进程 A/B 的前提：生产入口 `gemm_q4_*_bf16_ptr` 必须走**旧核**（R2 走显式入口）。
    std::env::set_var("LINGXI_NO_Q4D_R2", "1");
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let shapes = [
        (10240usize, 2560usize, "GDN qkvz"),
        (6144, 2560, "GDN z / QSA q"),
        (2560, 6144, "out_proj"),
    ];
    let modes = [
        (0i32, "① 纯读 4 B/lane（生产取数形状）"),
        (1, "② 纯读 16 B/lane uint4（结构上限）"),
        (2, "③ 纯读 + 解包 + I2F（ALU 地板）"),
        (5, "④ 解包 + FFMA（x 用常量，无 smem 读）"),
        (6, "⑦ 只做 staging（不读权重）"),
        (7, "⑧ 生产内层但 x 直读 global（无舞台）"),
        (4, "⑤ staging + 纯读 4 B（隔离舞台）"),
        (3, "⑥ 生产内层全量（自检）"),
    ];

    // 逐位自检先跑（形状小的一半，快）
    {
        let w = q4w(&k, 1024, 2560);
        check_mode3(&k, &w, 1024, 2560, 6);
        check_rows(&k, &w, 1024, 2560, 6);
        // 行数不是 8R 整数倍的边界（n=1000、m=5）
        let w2 = q4w(&k, 1000, 2560);
        check_rows(&k, &w2, 1000, 2560, 5);
    }
    check_prod(&k);

    for &m in &[5usize, 6] {
        for &(n, kdim, name) in shapes.iter() {
            let w = q4w(&k, n, kdim);
            let x = k.stream.clone_htod(&make_x(m, kdim)).unwrap();
            let mut y = k.stream.alloc_zeros::<bf16>(m * n).unwrap();
            let bytes = (n * kdim / 2) as f64;
            eprintln!(
                "── {name}（n{n} k{kdim}，payload {:.2} MB）m={m} ──",
                bytes / 1e6
            );
            bench_prod(&k, "现状核 gemm_q4_skinny_bf16", &w, &x, &mut y, n, kdim, m, bytes);
            for &(mode, nm) in modes.iter() {
                bench_probe(&k, nm, &w, &x, &mut y, n, kdim, m, bytes, mode);
            }
            for &(which, nm) in VARIANTS.iter() {
                bench_rows(&k, &w, &x, &mut y, n, kdim, m, bytes, which, nm);
            }
            bench_ab_prod(&k, &w, &x, &mut y, n, kdim, m, bytes);
        }
    }
}
