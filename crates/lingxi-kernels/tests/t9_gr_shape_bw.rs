//! **GR（gated residual）线上真实形状的微基准 + launch 地板对照**（2026-10-24）。
//!
//! 动机：`LINGXI_EVPROF=1` 的逐段表里 `入口+PLE+attn_GR`（seg0）与 `mlp_GR`（seg2）
//! 层均 **90~102 µs**，而一个 GR 块只读 13 MB（mix_down 6.55 + mix_up 6.55 + inject 0.08）。
//! 本用例按**线上分派**逐核量 µs/GB/s，并对每个核用一个**同 grid/block 的空核**
//! （`empty_probe_scale`，`n=0`）量出该网格形状的 launch 地板 ⇒ 回答
//! "9x µs 里多少是活、多少是每发的地板"。
//!
//! 门控：
//! `cargo test --release -p lingxi-kernels --features gpu --test t9_gr_shape_bw -- --nocapture`
#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

const DIM: usize = 10240; // hc * hidden = 4 * 2560
const HIDDEN: usize = 2560;
const LOWRANK: usize = 320;
const HC: usize = 4;
const EPS: f32 = 1e-6;

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
    eprintln!(
        "[gr-bw] {tag:<38} {us:8.1} µs｜{:6.1} GB/s｜{:6.2} MB",
        bytes as f64 / (us * 1e-6) / 1e9,
        bytes as f64 / 1e6
    );
    us
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

struct Gr {
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

impl Gr {
    fn new(k: &Kernels, rows: usize) -> Self {
        Self {
            x: k.stream.clone_htod(&rndbf16(rows * DIM, 2)).unwrap(),
            gamma: k.stream.clone_htod(&rndbf16(DIM, 3)).unwrap(),
            wd: k.stream.clone_htod(&rndbf16(LOWRANK * DIM, 4)).unwrap(),
            wu: k.stream.clone_htod(&rndbf16(DIM * LOWRANK, 5)).unwrap(),
            wi: k.stream.clone_htod(&rndbf16(HC * DIM, 6)).unwrap(),
            xn: k.stream.alloc_zeros::<bf16>(rows * DIM).unwrap(),
            d: k.stream.alloc_zeros::<bf16>(rows * LOWRANK).unwrap(),
            imw: k.stream.alloc_zeros::<bf16>(rows * DIM).unwrap(),
            mixed: k.stream.alloc_zeros::<bf16>(rows * HIDDEN).unwrap(),
            injw: k.stream.alloc_zeros::<bf16>(rows * HC).unwrap(),
        }
    }
}

/// 线上 GR 块的 5 发（`gated_residual_forward_into` 的默认路径，rows=5~6）。
fn gr_chain(k: &Kernels, g: &mut Gr, rows: usize) {
    let inv_hc = 1.0f32 / HC as f32;
    k.grouped_rmsnorm_bf16(&g.x, &g.gamma, &mut g.xn, rows as u32, DIM, HIDDEN, EPS).unwrap();
    {
        let (wp, _a) = g.wd.device_ptr(&k.stream);
        let (xp, _b) = g.xn.device_ptr(&k.stream);
        let (yp, _c) = g.d.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, LOWRANK, DIM, rows, 1, inv_hc, 0.0).unwrap();
    }
    {
        let (wp, _a) = g.wu.device_ptr(&k.stream);
        let (xp, _b) = g.d.device_ptr(&k.stream);
        let (yp, _c) = g.imw.device_ptr_mut(&k.stream);
        k.gemm_skinny_bf16_ptr(wp, xp, yp, DIM, LOWRANK, rows, 2, 1.0, 1.0).unwrap();
    }
    k.mix_mean_bf16(&g.imw, &g.xn, &mut g.mixed, HC, HIDDEN, rows as u32).unwrap();
    {
        let (wp, _a) = g.wi.device_ptr(&k.stream);
        let (xp, _b) = g.xn.device_ptr(&k.stream);
        let (yp, _c) = g.injw.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, HC, DIM, rows, 2, inv_hc, 2.0).unwrap();
    }
}

fn per_kernel(k: &Kernels, rows: usize) {
    let mut g = Gr::new(k, rows);
    let inv_hc = 1.0f32 / HC as f32;
    // 空核对照的缓冲
    let zx = k.stream.alloc_zeros::<f32>(1).unwrap();
    let mut zy = k.stream.alloc_zeros::<f32>(1).unwrap();

    eprintln!("[gr-bw] ── rows={rows}：逐核 + 空核地板（同 grid/block） ──");
    // 0 rmsnorm：grid (rows, dim/hidden)
    let us = bench(k, "rmsnorm", rows * DIM * 2 * 2, 100, || {
        k.grouped_rmsnorm_bf16(&g.x, &g.gamma, &mut g.xn, rows as u32, DIM, HIDDEN, EPS).unwrap();
    });
    let e = bench(k, "  ↑空核同 grid(rows,4)", 0, 200, || {
        k.empty_probe_scale(&zx, &mut zy, (rows as u32, 4), 256).unwrap();
    });
    eprintln!("[gr-bw]   ⇒ rmsnorm 净活 {:.1} µs", us - e);

    // 1 down：grid (320, 1)
    let us = bench(k, "mix_down", LOWRANK * DIM * 2, 100, || {
        let (wp, _a) = g.wd.device_ptr(&k.stream);
        let (xp, _b) = g.xn.device_ptr(&k.stream);
        let (yp, _c) = g.d.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, LOWRANK, DIM, rows, 1, inv_hc, 0.0).unwrap();
    });
    let e = bench(k, "  ↑空核同 grid(320,1)", 0, 200, || {
        k.empty_probe_scale(&zx, &mut zy, (LOWRANK as u32, 1), 256).unwrap();
    });
    eprintln!("[gr-bw]   ⇒ mix_down 净活 {:.1} µs", us - e);

    // 2 up：grid (1280, 1)
    let us = bench(k, "mix_up", DIM * LOWRANK * 2, 100, || {
        let (wp, _a) = g.wu.device_ptr(&k.stream);
        let (xp, _b) = g.d.device_ptr(&k.stream);
        let (yp, _c) = g.imw.device_ptr_mut(&k.stream);
        k.gemm_skinny_bf16_ptr(wp, xp, yp, DIM, LOWRANK, rows, 2, 1.0, 1.0).unwrap();
    });
    let e = bench(k, "  ↑空核同 grid(1280,1)", 0, 200, || {
        k.empty_probe_scale(&zx, &mut zy, ((DIM / 8) as u32, 1), 256).unwrap();
    });
    eprintln!("[gr-bw]   ⇒ mix_up 净活 {:.1} µs", us - e);

    // 3 mix：grid (10, rows)
    let us = bench(k, "mix_mean", rows * DIM * 2 * 2, 100, || {
        k.mix_mean_bf16(&g.imw, &g.xn, &mut g.mixed, HC, HIDDEN, rows as u32).unwrap();
    });
    let e = bench(k, "  ↑空核同 grid(10,rows)", 0, 200, || {
        k.empty_probe_scale(&zx, &mut zy, (10, rows as u32), 256).unwrap();
    });
    eprintln!("[gr-bw]   ⇒ mix_mean 净活 {:.1} µs", us - e);

    // 4 inject：grid (4, 1)
    let us = bench(k, "block_inject(n=4)", HC * DIM * 2, 100, || {
        let (wp, _a) = g.wi.device_ptr(&k.stream);
        let (xp, _b) = g.xn.device_ptr(&k.stream);
        let (yp, _c) = g.injw.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, HC, DIM, rows, 2, inv_hc, 2.0).unwrap();
    });
    let e = bench(k, "  ↑空核同 grid(4,1)", 0, 200, || {
        k.empty_probe_scale(&zx, &mut zy, (HC as u32, 1), 256).unwrap();
    });
    eprintln!("[gr-bw]   ⇒ block_inject 净活 {:.1} µs", us - e);

    // 5 整链
    let bytes = LOWRANK * DIM * 2 + DIM * LOWRANK * 2 + HC * DIM * 2;
    let us = bench(k, "【线上 5 发整链】", bytes, 100, || gr_chain(k, &mut g, rows));
    eprintln!("[gr-bw]   ⇒ 5 发整链 {us:.1} µs（含 5 次发射）");
}

/// 融合档（新 3 发）vs 旧 5 发：**逐位对拍 + 计时**。
fn fused_vs_old(k: &Kernels, rows: usize) {
    let mut g = Gr::new(k, rows);
    let mut d_b = k.stream.alloc_zeros::<bf16>(rows * LOWRANK).unwrap();
    let mut mixed_b = k.stream.alloc_zeros::<bf16>(rows * HIDDEN).unwrap();
    let mut injw_b = k.stream.alloc_zeros::<bf16>(rows * HC).unwrap();
    let inv_hc = 1.0f32 / HC as f32;
    // 旧链（参考）
    gr_chain(k, &mut g, rows);
    // 新链：rmsnorm → down+inject → up+mix。**每个 opb 都要对拍**（opb>1 时尾部的
    // `tid < m*opb` 映射是另一条代码路径；只测 opb=1 会漏掉"半个 hidden 没写"这类 bug）。
    for opb in [1usize, 2, 4, 8] {
        k.stream.memset_zeros(&mut mixed_b).unwrap();
        k.grouped_rmsnorm_bf16(&g.x, &g.gamma, &mut g.xn, rows as u32, DIM, HIDDEN, EPS).unwrap();
        k.gr_down_inject_bf16(&g.wd, &g.wi, &g.xn, &mut d_b, &mut injw_b, LOWRANK, HC, DIM, rows, inv_hc)
            .unwrap();
        k.gr_up_mix_fuse_bf16(&g.wu, &g.d, &g.xn, &mut mixed_b, LOWRANK, HIDDEN, HC, rows, opb).unwrap();
        k.stream.synchronize().unwrap();
        let (da, db) = (
            k.stream.clone_dtoh(&g.d).unwrap(),
            k.stream.clone_dtoh(&d_b).unwrap(),
        );
        let (ma, mb) = (
            k.stream.clone_dtoh(&g.mixed).unwrap(),
            k.stream.clone_dtoh(&mixed_b).unwrap(),
        );
        let (ia, ib) = (
            k.stream.clone_dtoh(&g.injw).unwrap(),
            k.stream.clone_dtoh(&injw_b).unwrap(),
        );
        let cnt = |a: &[bf16], b: &[bf16]| {
            a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
        };
        let (dd, dm, di) = (cnt(&da, &db), cnt(&ma, &mb), cnt(&ia, &ib));
        let mut shown = 0;
        for idx in 0..rows * HIDDEN {
            if ma[idx].to_bits() != mb[idx].to_bits() && shown < 4 {
                eprintln!(
                    "[gr-bw]   mixed diff@{idx} (r={} i={}): 旧 {:.7} vs 新 {:.7}",
                    idx / HIDDEN, idx % HIDDEN, ma[idx].to_f32(), mb[idx].to_f32()
                );
                shown += 1;
            }
        }
        eprintln!(
            "[gr-bw] 逐位对拍 rows={rows} opb={opb}：d {dd}/{}、mixed {dm}/{}、injw {di}/{} 不一致",
            rows * LOWRANK,
            rows * HIDDEN,
            rows * HC
        );
        assert_eq!((dd, dm, di), (0, 0, 0), "opb={opb}：融合档与旧 5 发链不逐位相同");
    }

    let bytes = LOWRANK * DIM * 2 + DIM * LOWRANK * 2 + HC * DIM * 2;
    let us_old = bench(k, "【对比】旧 5 发", bytes, 100, || gr_chain(k, &mut g, rows));
    for opb in [1usize, 2, 4, 8] {
        let us = bench(k, &format!("【对比】新 3 发 opb={opb}"), bytes, 100, || {
            k.grouped_rmsnorm_bf16(&g.x, &g.gamma, &mut g.xn, rows as u32, DIM, HIDDEN, EPS).unwrap();
            k.gr_down_inject_bf16(&g.wd, &g.wi, &g.xn, &mut d_b, &mut injw_b, LOWRANK, HC, DIM, rows, inv_hc)
                .unwrap();
            k.gr_up_mix_fuse_bf16(&g.wu, &g.d, &g.xn, &mut mixed_b, LOWRANK, HIDDEN, HC, rows, opb).unwrap();
        });
        eprintln!("[gr-bw]   ⇒ opb={opb} 相对旧链 {:.1}%", 100.0 * (us / us_old - 1.0));
    }
    // 新核逐发
    bench(k, "  · 新 down+inject 单发", LOWRANK * DIM * 2 + HC * DIM * 2, 100, || {
        k.gr_down_inject_bf16(&g.wd, &g.wi, &g.xn, &mut d_b, &mut injw_b, LOWRANK, HC, DIM, rows, inv_hc)
            .unwrap();
    });
    for opb in [1usize, 2, 4] {
        bench(k, &format!("  · 新 up+mix 单发 opb={opb}"), DIM * LOWRANK * 2, 100, || {
            k.gr_up_mix_fuse_bf16(&g.wu, &g.d, &g.xn, &mut mixed_b, LOWRANK, HIDDEN, HC, rows, opb).unwrap();
        });
    }
}

/// up+mix（新）与 mix_up（旧）随 m 的标度：判断"每多一个 token 行"要付多少钱。
fn m_scaling(k: &Kernels) {
    let mut g = Gr::new(k, 8);
    eprintln!("[gr-bw] ── up 随 m 的标度（同权重量，只有 x 行数变） ──");
    for m in 1usize..=8 {
        let mut imw = k.stream.alloc_zeros::<bf16>(m * DIM).unwrap();
        bench(k, &format!("旧 mix_up m={m}"), DIM * LOWRANK * 2, 100, || {
            let (wp, _a) = g.wu.device_ptr(&k.stream);
            let (xp, _b) = g.d.device_ptr(&k.stream);
            let (yp, _c) = imw.device_ptr_mut(&k.stream);
            k.gemm_skinny_bf16_ptr(wp, xp, yp, DIM, LOWRANK, m, 2, 1.0, 1.0).unwrap();
        });
    }
    for m in 2usize..=8 {
        bench(k, &format!("新 up+mix opb=2 m={m}"), DIM * LOWRANK * 2, 100, || {
            k.gr_up_mix_fuse_bf16(&g.wu, &g.d, &g.xn, &mut g.mixed, LOWRANK, HIDDEN, HC, m, 2).unwrap();
        });
    }
}

#[test]
fn t9_gr_shape_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    eprintln!("[gr-bw] GR 块真实形状：dim={DIM} hidden={HIDDEN} lowrank={LOWRANK} hc={HC}");
    for rows in [1usize, 5, 6, 7] {
        per_kernel(&k, rows);
    }
    for rows in [5usize, 6, 7] {
        fused_vs_old(&k, rows);
    }
    m_scaling(&k);
}
