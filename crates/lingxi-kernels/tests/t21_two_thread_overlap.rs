//! **两张卡、两个 host 线程能不能真重叠**。
//!
//! 背景：投机验证的一轮 = GPU0 半场 + GPU1 半场，**串行**（同一份 hidden 脊柱，深度 1 级）。
//! 的结论是"跨流重叠在同一个主机线程里不可能"——那就换一个主机线程。
//! 这条路线能完全绕开 `any_miss` 与 `spec_restore`（路由保持现有的同步版本，spin 在各自线程里无害）。
//!
//! 但这条路成立的前提必须先证：**两个线程各驱一张卡时，GPU 工作真的并行**。
//! 判据：同样的总工作量，两线程并行墙钟 ≈ 单线程串行的 1/2（≥1.7× 算过）。
//!
//! 跑法（**必须单线程跑测试框架**，避免测试框架自己的并行干扰计时）：
//! `bash
//! export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib   # cublas/nvrtc 不在默认路径时按需追加
//! cargo test --release -p lingxi-kernels --features gpu --test t21_two_thread_overlap \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

/// 每个线程的载荷：一张 248320×2560 的 Q4 稠密权重 + 一次 GEMV。
/// 选这个形状是因为它和线上 `lm_head` 同形（318 MB，纯 HBM 流），
/// 能代表"每张卡各读自己那份 HBM"的真实情形。
const N: usize = 248320;
const K: usize = 2560;

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

fn layout(rows: u64, cols: u64) -> (u64, u64, u64) {
    let k_pad = align_up(cols, 128);
    let groups = k_pad / 64;
    let row_bytes = groups * 32;
    let base = rows * row_bytes;
    let scale_off = align_up(base, 256);
    (row_bytes, scale_off, scale_off + rows * groups * 2)
}

fn rnd_bytes(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFF) as u8
        })
        .collect()
}

struct Side {
    w: CudaSlice<u8>,
    x: CudaSlice<bf16>,
    y: CudaSlice<bf16>,
    scale_off: u64,
}

fn build(k: &Kernels) -> Side {
    let (rb, scale_off, pay) = layout(N as u64, K as u64);
    let mut host = rnd_bytes(pay as usize, 0x1234);
    // scale 填合理值（随机字节当 fp16 多为 NaN/denorm）
    for i in 0..N * (K / 64) {
        let h = half::f16::from_f32(0.002 + 0.001 * ((i % 23) as f32));
        let off = scale_off as usize + 2 * i;
        host[off..off + 2].copy_from_slice(&h.to_le_bytes());
    }
    let _ = rb;
    let w = k.stream.clone_htod(&host).unwrap();
    drop(host);
    let x = k
        .stream
        .clone_htod(&(0..K).map(|i| bf16::from_f32(i as f32 * 1e-4)).collect::<Vec<_>>())
        .unwrap();
    let y = k.stream.alloc_zeros::<bf16>(N).unwrap();
    Side { w, x, y, scale_off }
}

/// 跑 `reps` 次 GEMV 并等它做完，返回墙钟秒数。
fn run(k: &Kernels, s: &mut Side, reps: usize) -> f64 {
    let (wp, _a) = s.w.device_ptr(&k.stream);
    let (xp, _b) = s.x.device_ptr(&k.stream);
    let (yp, _c) = s.y.device_ptr_mut(&k.stream);
    let t0 = Instant::now();
    for _ in 0..reps {
        k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, N, K, K as i32, 0, s.scale_off, 0, 0.0, 0.0)
            .unwrap();
    }
    k.stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64()
}

#[test]
fn t21_two_thread_overlap() {
    let k0 = Kernels::load(0).expect("GPU0 初始化失败");
    let k1 = Kernels::load(1).expect("GPU1 初始化失败");
    let mut a = build(&k0);
    let mut b = build(&k1);
    const REPS: usize = 8;

    // 预热 + 单线程串行基线
    run(&k0, &mut a, 2);
    run(&k1, &mut b, 2);
    let seq = run(&k0, &mut a, REPS) + run(&k1, &mut b, REPS);

    // 两线程并行（每线程只碰自己那张卡的 Kernels / slice）
    let par = {
        let t0 = Instant::now();
        std::thread::scope(|s| {
            s.spawn(|| {
                let _ = run(&k0, &mut a, REPS);
            });
            s.spawn(|| {
                let _ = run(&k1, &mut b, REPS);
            });
        });
        t0.elapsed().as_secs_f64()
    };

    let bytes = 2.0 * REPS as f64 * (N * K) as f64 / 2.0;
    eprintln!(
        "[t21] 每卡 {N}x{K} Q4 GEMV × {REPS}（每卡 {:.0} MB）",
        (N * K) as f64 / 2.0 / 1e6
    );
    eprintln!(
        "[t21] 串行（同一线程依次跑两卡）：{:7.1} ms｜{:6.1} GB/s（两卡合计）",
        seq * 1e3,
        bytes / seq / 1e9
    );
    eprintln!(
        "[t21] 并行（两线程各一卡）：      {:7.1} ms｜{:6.1} GB/s（两卡合计）",
        par * 1e3,
        bytes / par / 1e9
    );
    eprintln!("[t21] ⇒ 并行加速比 **{:.2}×**", seq / par);
    assert!(
        seq / par > 1.7,
        "两线程没有得到预期的并行度（{:.2}×）—— D' 路线的前提不成立",
        seq / par
    );
}
