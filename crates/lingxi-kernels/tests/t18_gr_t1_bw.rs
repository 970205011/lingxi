//! **GR（t=1，草稿占位）的取数分母**：同形纯读 + 空核地板 + 冷/热 L2 + 连发/突发。
//!
//! 动机（2026-10-24）：`LINGXI_MTP_PROF` 把 GR 的 510 µs/步报成最大剩余项，但那是**探针
//! 错位**造成的（见 `draft_prof.rs` 顶部；正确数字 ≈ 114 µs/步）。本用例只回答一件事：
//! **t=1 的 5 个核，各自离开它的"地址流地板"还有多远**——
//! * 同形纯读（`gr_probe_t1.cu`，grid/block/向量宽度/归约树与生产核相同，只读不算）
//! * 同字节的理想流式读（能打满 HBM 的形状）
//! * 同 grid 的空核（发射地板）
//! * 冷 L2（先写 48 MB 脏缓冲）+ 单次计时（草稿每一步的真实上下文：上一轮主干刚把 L2 冲掉）
//!
//! 门控：
//! `cargo test --release -p lingxi-kernels --features gpu --test t18_gr_t1_bw -- --nocapture`
#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

const DIM: usize = 10240; // hc * hidden
const HIDDEN: usize = 2560;
const LOWRANK: usize = 320;
const HC: usize = 4;
const EPS: f32 = 1e-6;

fn rndbf16(n: usize, seed: u32) -> Vec<bf16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_f32(((s >> 8) as f32 / 16777216.0 - 0.5) * 0.1)
        })
        .collect()
}

fn gbs(bytes: usize, us: f64) -> f64 {
    bytes as f64 / (us * 1e-6) / 1e9
}

/// 热循环（同一份地址流反复跑，流水不排空）
fn bench<F: FnMut()>(k: &Kernels, tag: &str, bytes: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..5 {
        f();
    }
    k.stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    k.stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    eprintln!("[gr-t1] {tag:<34} {us:7.1} µs｜{:6.1} GB/s", gbs(bytes, us));
    us
}

/// 冷 L2 + 单次计时（5 次取 min）：先写 48 MB 脏缓冲把 L2 冲掉，再量**一发**。
fn bench_cold<F: FnMut()>(
    k: &Kernels, dirty: &mut CudaSlice<u8>, tag: &str, bytes: usize, mut f: F,
) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..5 {
        k.stream.memset_zeros(dirty).unwrap();
        k.stream.synchronize().unwrap();
        let t0 = Instant::now();
        f();
        k.stream.synchronize().unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e6);
    }
    eprintln!("[gr-t1] {tag:<34} {best:7.1} µs｜{:6.1} GB/s（冷 L2 单发 min）", gbs(bytes, best));
    best
}

struct Bufs {
    x: CudaSlice<bf16>,
    gamma: CudaSlice<bf16>,
    wd: CudaSlice<bf16>,
    wu: CudaSlice<bf16>,
    wi: CudaSlice<bf16>,
    xn: CudaSlice<bf16>,
    d: CudaSlice<bf16>,
    imw: CudaSlice<bf16>,
    mixed: CudaSlice<bf16>,
    injw: CudaSlice<bf16>,
}

impl Bufs {
    fn new(k: &Kernels) -> Self {
        Self {
            x: k.stream.clone_htod(&rndbf16(DIM, 2)).unwrap(),
            gamma: k.stream.clone_htod(&rndbf16(DIM, 3)).unwrap(),
            wd: k.stream.clone_htod(&rndbf16(LOWRANK * DIM, 4)).unwrap(),
            wu: k.stream.clone_htod(&rndbf16(DIM * LOWRANK, 5)).unwrap(),
            wi: k.stream.clone_htod(&rndbf16(HC * DIM, 6)).unwrap(),
            xn: k.stream.alloc_zeros::<bf16>(DIM).unwrap(),
            d: k.stream.alloc_zeros::<bf16>(LOWRANK).unwrap(),
            imw: k.stream.alloc_zeros::<bf16>(DIM).unwrap(),
            mixed: k.stream.alloc_zeros::<bf16>(HIDDEN).unwrap(),
            injw: k.stream.alloc_zeros::<bf16>(HC).unwrap(),
        }
    }
}

const INV_HC: f32 = 1.0 / HC as f32;

/// 线上 t=1 的 5 发整链（`gated_residual_forward_into` 的**默认路**，rows=1）。
///
/// ⚠️ 线上 t=1 走的是 `gpu::proj_t1_epi` ⇒ `gemv_bf16_row8_epi`（k≥1024）/`gemv_bf16_epi`，
/// **不是** `t9_gr_shape_bw` 里那条 `gemm_skinny_*`（那是 rows≥2 的路）——这是本用例存
/// 在的原因之一。
fn chain(k: &Kernels, b: &mut Bufs) {
    k.grouped_rmsnorm_bf16(&b.x, &b.gamma, &mut b.xn, 1, DIM, HIDDEN, EPS).unwrap();
    {
        let (wp, _a) = b.wd.device_ptr(&k.stream);
        let (xp, _b) = b.xn.device_ptr(&k.stream);
        let (yp, _c) = b.d.device_ptr_mut(&k.stream);
        k.gemv_bf16_row8_epi_ptr(wp, xp, yp, LOWRANK, DIM, 1, INV_HC, 0.0).unwrap();
    }
    {
        let (wp, _a) = b.wu.device_ptr(&k.stream);
        let (xp, _b) = b.d.device_ptr(&k.stream);
        let (yp, _c) = b.imw.device_ptr_mut(&k.stream);
        k.gemv_bf16_epi_ptr(wp, xp, yp, DIM, LOWRANK, 2, 1.0, 1.0).unwrap();
    }
    k.mix_mean_bf16(&b.imw, &b.xn, &mut b.mixed, HC, HIDDEN, 1).unwrap();
    {
        let (wp, _a) = b.wi.device_ptr(&k.stream);
        let (xp, _b) = b.xn.device_ptr(&k.stream);
        let (yp, _c) = b.injw.device_ptr_mut(&k.stream);
        k.gemv_bf16_row8_epi_ptr(wp, xp, yp, HC, DIM, 2, INV_HC, 2.0).unwrap();
    }
}

#[test]
fn t18_gr_t1_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut b = Bufs::new(&k);
    let mut rout = k.stream.alloc_zeros::<f32>(HIDDEN).unwrap();
    let mut dirty = k.stream.alloc_zeros::<u8>(48 * 1024 * 1024).unwrap();
    let zx = k.stream.alloc_zeros::<f32>(1).unwrap();
    let mut zy = k.stream.alloc_zeros::<f32>(1).unwrap();

    let b_down = LOWRANK * DIM * 2; // 6.55 MB
    let b_up = DIM * LOWRANK * 2; // 1.64 MB
    let b_inj = HC * DIM * 2; // 0.08 MB
    let b_all = b_down + b_up + b_inj;

    eprintln!("[gr-t1] 线上 t=1 形状：dim={DIM} hidden={HIDDEN} lowrank={LOWRANK} hc={HC}");

    // ── A. 生产核（热循环）──
    eprintln!("[gr-t1] ── A. 生产核（热循环，host 计时） ──");
    bench(&k, "rmsnorm  grid(1,4)", DIM * 2 * 2, 100, || {
        k.grouped_rmsnorm_bf16(&b.x, &b.gamma, &mut b.xn, 1, DIM, HIDDEN, EPS).unwrap();
    });
    bench(&k, "mix_down row8 grid(320) k10240", b_down, 100, || {
        let (wp, _a) = b.wd.device_ptr(&k.stream);
        let (xp, _b) = b.xn.device_ptr(&k.stream);
        let (yp, _c) = b.d.device_ptr_mut(&k.stream);
        k.gemv_bf16_row8_epi_ptr(wp, xp, yp, LOWRANK, DIM, 1, INV_HC, 0.0).unwrap();
    });
    bench(&k, "  ↑对照：gemm_skinny_row 同形", b_down, 100, || {
        let (wp, _a) = b.wd.device_ptr(&k.stream);
        let (xp, _b) = b.xn.device_ptr(&k.stream);
        let (yp, _c) = b.d.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, LOWRANK, DIM, 1, 1, INV_HC, 0.0).unwrap();
    });
    bench(&k, "mix_up gemv grid(80) k320", b_up, 100, || {
        let (wp, _a) = b.wu.device_ptr(&k.stream);
        let (xp, _b) = b.d.device_ptr(&k.stream);
        let (yp, _c) = b.imw.device_ptr_mut(&k.stream);
        k.gemv_bf16_epi_ptr(wp, xp, yp, DIM, LOWRANK, 2, 1.0, 1.0).unwrap();
    });
    bench(&k, "mix_mean grid(10)", DIM * 2 * 2, 100, || {
        k.mix_mean_bf16(&b.imw, &b.xn, &mut b.mixed, HC, HIDDEN, 1).unwrap();
    });
    bench(&k, "block_inject row8 grid(4)", b_inj, 100, || {
        let (wp, _a) = b.wi.device_ptr(&k.stream);
        let (xp, _b) = b.xn.device_ptr(&k.stream);
        let (yp, _c) = b.injw.device_ptr_mut(&k.stream);
        k.gemv_bf16_row8_epi_ptr(wp, xp, yp, HC, DIM, 2, INV_HC, 2.0).unwrap();
    });
    bench(&k, "【整链 5 发 · 连发】", b_all, 100, || chain(&k, &mut b));

    // ── B. 同形纯读（只读不算；地址流与生产核逐字相同）──
    eprintln!("[gr-t1] ── B. 同形纯读分母（grid/block/地址流与 A 相同，只读不算） ──");
    {
        let (wdp, _a) = b.wd.device_ptr(&k.stream);
        bench(&k, "纯读 down 形 grid(320)", b_down, 100, || {
            k.probe_read_row8_bf16(wdp, &mut rout, LOWRANK, DIM).unwrap();
        });
        bench(&k, "纯读 inject 形 grid(4)", b_inj, 100, || {
            k.probe_read_row8_bf16(wdp, &mut rout, HC, DIM).unwrap();
        });
        bench(&k, "理想流式读 6.55MB", b_down, 100, || {
            k.probe_read_stream_bf16(wdp, &mut rout, b.wd.len()).unwrap();
        });
    }
    {
        let (wup, _a) = b.wu.device_ptr(&k.stream);
        bench(&k, "纯读 up 形 grid(80) k320", b_up, 100, || {
            k.probe_read_warp_bf16(wup, &mut rout, DIM, LOWRANK).unwrap();
        });
        bench(&k, "理想流式读 1.64MB", b_up, 100, || {
            k.probe_read_stream_bf16(wup, &mut rout, b.wu.len()).unwrap();
        });
    }

    // ── C. 同 grid 空核（发射地板）──
    eprintln!("[gr-t1] ── C. 空核地板（同 grid/block） ──");
    for (tag, grid, block) in [
        ("空核 grid(320) blk1024", (LOWRANK as u32, 1u32), 1024u32),
        ("空核 grid(80)  blk1024", ((DIM / 32) as u32, 1), 1024),
        ("空核 grid(4)   blk1024", (HC as u32, 1), 1024),
        ("空核 grid(10)  blk256", (10, 1), 256),
        ("空核 grid(1,4) blk256", (1, 4), 256),
    ] {
        bench(&k, tag, 0, 200, || {
            k.empty_probe_scale(&zx, &mut zy, grid, block).unwrap();
        });
    }

    // ── D. 冷 L2 单发（草稿每一步的真实上下文：上一轮主干刚冲掉 L2）──
    eprintln!("[gr-t1] ── D. 冷 L2 + 单发 ──");
    {
        let (wdp, _a) = b.wd.device_ptr(&k.stream);
        let (xp, _b) = b.xn.device_ptr(&k.stream);
        let (yp, _c) = b.d.device_ptr_mut(&k.stream);
        bench_cold(&k, &mut dirty, "冷 L2 mix_down", b_down, || {
            k.gemv_bf16_row8_epi_ptr(wdp, xp, yp, LOWRANK, DIM, 1, INV_HC, 0.0).unwrap();
        });
    }
    {
        let (wup, _a) = b.wu.device_ptr(&k.stream);
        let (dp, _b) = b.d.device_ptr(&k.stream);
        let (yp, _c) = b.imw.device_ptr_mut(&k.stream);
        bench_cold(&k, &mut dirty, "冷 L2 mix_up", b_up, || {
            k.gemv_bf16_epi_ptr(wup, dp, yp, DIM, LOWRANK, 2, 1.0, 1.0).unwrap();
        });
    }
    bench_cold(&k, &mut dirty, "【整链 5 发 · 冷 L2 单拍】", b_all, || chain(&k, &mut b));
    // 冷 L2 下的**同形纯读地板**：判断 above 的 27/32 µs 是"核慢"还是"冷 L2 的地址流地板"
    {
        let (wdp, _a) = b.wd.device_ptr(&k.stream);
        bench_cold(&k, &mut dirty, "冷 L2 纯读 down 形 grid(320)", b_down, || {
            k.probe_read_row8_bf16(wdp, &mut rout, LOWRANK, DIM).unwrap();
        });
        bench_cold(&k, &mut dirty, "冷 L2 理想流式读 6.55MB", b_down, || {
            k.probe_read_stream_bf16(wdp, &mut rout, b.wd.len()).unwrap();
        });
    }
    {
        let (wup, _a) = b.wu.device_ptr(&k.stream);
        bench_cold(&k, &mut dirty, "冷 L2 纯读 up 形 grid(80) k320", b_up, || {
            k.probe_read_warp_bf16(wup, &mut rout, DIM, LOWRANK).unwrap();
        });
        bench_cold(&k, &mut dirty, "冷 L2 理想流式读 1.64MB", b_up, || {
            k.probe_read_stream_bf16(wup, &mut rout, b.wu.len()).unwrap();
        });
    }
}
