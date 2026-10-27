//! **MTP 草稿步（t=1）真实形状的带宽微基准**（2026-10-24）。
//!
//! 动机：`LINGXI_MTP_PROF=1` 把 draft 的一步拆成
//! `stem / attn_gr / attn / mlp_gr / moe / 收尾`，但"某段 200 µs 是高还是低"必须
//! 拿**核本身在同样形状上的上限**作对照。本用例按 MTP 层（哨兵 100）的真实形状、
//! `t=1`、`E=10` 逐核量 µs 与 GB/s，并顺带量 **head**（层尾 mixer + lm_head Q4 +
//! top-k）——它不在 MTP 层内（落在 `lingxi-bench::spec`），但每步都跑一次。
//!
//! 门控与其它 GPU 用例一致：
//! `LD_LIBRARY_PATH=... cargo test --release -p lingxi-kernels --features gpu --test t7_mtp_shape_bw -- --nocapture`
//!
//! 形状来源（产物 `mtp/*`，见 `lingxi-bench list`）：
//! * stem：`fc_embedding/fc_hidden` [2560,2560]，`hidden_norm` [10240]
//! * QSA：`query_key_gate_value` [13312,2560]（q 6144 / k 512 / gate 6144 / v 512 四发）、
//! `output` [2560,6144]、`index_qk_proj` [640,2560]
//! * GR（attn_gr / mlp_gr / mixer）：`mix_down` [320,10240]、`mix_up` [10240,320]、
//! `block_inject` [4,10240]
//! * MoE：`router_shared_gate` [513,2560]、`shared_gate_up` [1280,2560]、
//! `shared_down` [2560,640]；专家 Q4 槽：gu [1280,2560] + dn [2560,640] × E=10
//! * head：`lm_head` Q4 [248320,2560]（vocab 248320）

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

/// 跑 `iters` 次取均值，打印 µs 与 GB/s（`bytes` = 该次 launch 真正从 HBM 读的字节）。
fn bench<F: FnMut()>(k: &Kernels, tag: &str, bytes: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    k.stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    k.stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let gbs = bytes as f64 / (us * 1e-6) / 1e9;
    eprintln!("[mtp-bw] {tag:<40} {us:8.1} µs｜{gbs:6.1} GB/s｜{:6.1} MB", bytes as f64 / 1e6);
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

/// 与 `gpu::proj_t1` 完全同一条分派（t=1）：`k>=1024 && k%8==0` 走 row8，否则 warps-per-row。
/// （`proj_t1` 在 `lingxi-model`，本 crate 不能反向依赖 ⇒ 这里逐行复刻分派条件。）
fn proj_t1(k: &Kernels, x: &CudaSlice<bf16>, w: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
           n: usize, kdim: usize, t: usize) {
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let row8 = kdim >= 1024 && kdim % 8 == 0;
    if t == 1 {
        if row8 {
            k.gemv_bf16_row8_epi_ptr(wp, xp, yp, n, kdim, 0, 0.0, 0.0).unwrap()
        } else {
            k.gemv_bf16_ptr(wp, xp, yp, n, kdim).unwrap()
        }
    } else if row8 {
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, n, kdim, t, 0, 0.0, 0.0).unwrap()
    } else {
        k.gemm_skinny_bf16_ptr(wp, xp, yp, n, kdim, t, 0, 0.0, 0.0).unwrap()
    }
}

/// 同上，带尾融合（`gpu::proj_t1_epi` 的 t=1 分派）。
fn proj_t1_epi(k: &Kernels, x: &CudaSlice<bf16>, w: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
               n: usize, kdim: usize, t: usize, epi: i32, a: f32, b: f32) {
    assert_eq!(t, 1, "本微基准只量 decode 档（t=1）");
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    if kdim >= 1024 && kdim % 8 == 0 && epi != 4 {
        k.gemv_bf16_row8_epi_ptr(wp, xp, yp, n, kdim, epi, a, b).unwrap()
    } else {
        k.gemv_bf16_epi_ptr(wp, xp, yp, n, kdim, epi, a, b).unwrap()
    }
}

/// 与 `gpu::proj_t1` 完全同一条分派（t=1）：`k>=1024 && k%8==0` 走 row8，否则 warps-per-row。
fn gemv_t1(k: &Kernels, tag: &str, n: usize, kdim: usize) {
    let w = k.stream.clone_htod(&rndbf16(n * kdim, 1)).unwrap();
    let x = k.stream.clone_htod(&rndbf16(kdim, 2)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(n).unwrap();
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    let row8 = kdim >= 1024 && kdim % 8 == 0;
    bench(k, tag, n * kdim * 2, 30, || {
        if row8 {
            k.gemv_bf16_row8_epi_ptr(wp, xp, yp, n, kdim, 0, 0.0, 0.0).unwrap()
        } else {
            k.gemv_bf16_ptr(wp, xp, yp, n, kdim).unwrap()
        }
    });
}

/// GR 一段（`gated_residual_forward` 的 5 个 kernel，t=1、`LINGXI_GR_FUSE` 关）。
/// `dim=hc*hidden`、`lowrank`、`hidden` 与线上一致；`inject=false` 对应层尾 mixer（无 block_inject）。
fn gr_5k(k: &Kernels, tag: &str, dim: usize, hidden: usize, lowrank: usize, hc: usize, inject: bool) -> f64 {
    let x = k.stream.clone_htod(&rndbf16(dim, 3)).unwrap();
    let gamma = k.stream.clone_htod(&rndbf16(dim, 4)).unwrap();
    let wd = k.stream.clone_htod(&rndbf16(lowrank * dim, 5)).unwrap();
    let wu = k.stream.clone_htod(&rndbf16(dim * lowrank, 6)).unwrap();
    let wi = k.stream.clone_htod(&rndbf16(hc * dim, 7)).unwrap();
    let mut xn = k.stream.alloc_zeros::<bf16>(dim).unwrap();
    let mut d = k.stream.alloc_zeros::<bf16>(lowrank).unwrap();
    let mut imw = k.stream.alloc_zeros::<bf16>(dim).unwrap();
    let mut mixed = k.stream.alloc_zeros::<bf16>(hidden).unwrap();
    let mut injw = k.stream.alloc_zeros::<bf16>(hc).unwrap();
    let inv_hc = 1.0 / hc as f32;
    // 读的字节：mix_down + mix_up + block_inject + 小核的 x/gamma
    let bytes = if inject { lowrank * dim * 2 + dim * lowrank * 2 + hc * dim * 2 } else { lowrank * dim * 2 + dim * lowrank * 2 };
    bench(k, tag, bytes, 30, || {
        k.grouped_rmsnorm_bf16(&x, &gamma, &mut xn, 1, dim, hidden, 1e-6).unwrap();
        proj_t1_epi(k, &xn, &wd, &mut d, lowrank, dim, 1, 1, inv_hc, 0.0);
        proj_t1_epi(k, &d, &wu, &mut imw, dim, lowrank, 1, 2, 1.0, 1.0);
        k.mix_mean_bf16(&imw, &xn, &mut mixed, hc, hidden, 1).unwrap();
        if inject {
            proj_t1_epi(k, &xn, &wi, &mut injw, hc, dim, 1, 2, inv_hc, 2.0);
        }
    })
}

/// GR 融合档（`LINGXI_GR_FUSE=1` 的 2 个 kernel），供 A/B。
fn gr_2k(k: &Kernels, tag: &str, dim: usize, hidden: usize, lowrank: usize, hc: usize) {
    let x = k.stream.clone_htod(&rndbf16(dim, 3)).unwrap();
    let gamma = k.stream.clone_htod(&rndbf16(dim, 4)).unwrap();
    let wd = k.stream.clone_htod(&rndbf16(lowrank * dim, 5)).unwrap();
    let wu = k.stream.clone_htod(&rndbf16(dim * lowrank, 6)).unwrap();
    let wi = k.stream.clone_htod(&rndbf16(hc * dim, 7)).unwrap();
    let mut d = k.stream.alloc_zeros::<bf16>(lowrank).unwrap();
    let mut injw = k.stream.alloc_zeros::<bf16>(hc).unwrap();
    let mut rmsg = k.stream.alloc_zeros::<f32>(hc).unwrap();
    let mut mixed = k.stream.alloc_zeros::<bf16>(hidden).unwrap();
    let ngroups = dim / hidden;
    let inv_hc = 1.0 / hc as f32;
    let bytes = lowrank * dim * 2 + dim * lowrank * 2 + hc * dim * 2;
    bench(k, tag, bytes, 30, || {
        k.gr_down_fuse_bf16(&x, &gamma, &wd, &wi, &mut d, &mut injw, &mut rmsg,
                            dim, hidden, ngroups, lowrank, hc, inv_hc, 1e-6).unwrap();
        k.gr_up_mix_bf16(&d, &x, &gamma, &wu, &rmsg, &mut mixed,
                         dim, hidden, lowrank, ngroups, inv_hc);
    });
}

/// 稠密 Q4G64 GEMV（`t=1` 的 `proj_q4_epi`）——head 与主干 Q4 投影走的就是它。
fn gemv_q4_dense(k: &Kernels, tag: &str, n: usize, kdim: usize, iters: usize) {
    let row_bytes = kdim / 2;
    let scale_off = n * row_bytes; // 256B 对齐
    let mut w = unsafe { k.stream.alloc::<u8>(scale_off + n * kdim / 64 * 2) }.unwrap();
    // 填 0x88（nibble=8、scale=0）⇒ 值是 0，避免读未初始化内存看运气
    k.stream.memset_zeros(&mut w).unwrap();
    let x = k.stream.clone_htod(&rndbf16(kdim, 9)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(n).unwrap();
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y.device_ptr_mut(&k.stream);
    bench(k, tag, scale_off + n * kdim / 64 * 2, iters, || {
        k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, n, kdim, kdim as i32, 0, scale_off as u64, 0, 0.0, 0.0)
            .unwrap()
    });
}

/// 层尾 head 的 top-k（`topk_rows_part_bf16` + `topk_rows_fin`），t=1、vocab=248320。
fn head_topk(k: &Kernels, vocab: usize) {
    let logits = k.stream.clone_htod(&rndbf16(vocab, 11)).unwrap();
    const G: usize = 64;
    const KK: usize = 20;
    let mut pval = k.stream.alloc_zeros::<f32>(G * KK).unwrap();
    let mut pidx = k.stream.alloc_zeros::<i32>(G * KK).unwrap();
    let mut oidx = k.stream.alloc_zeros::<i32>(KK).unwrap();
    let mut oval = k.stream.alloc_zeros::<f32>(KK).unwrap();
    let mut bad = k.stream.alloc_zeros::<i32>(1).unwrap();
    bench(k, "head topk(part+fin)", vocab * 2, 30, || {
        k.topk_rows_part_bf16(&logits, &mut pval, &mut pidx, &mut bad, 1, vocab, G, KK).unwrap();
        k.topk_rows_fin(&pval, &pidx, &mut oval, &mut oidx, 1, G, KK);
    });
}

/// MoE 的 shared 专家（t=1、E 无关）：gate_up swiglu 融合 + down + shared_gate。
fn moe_shared(k: &Kernels, inter: usize, hidden: usize) {
    let x = k.stream.clone_htod(&rndbf16(hidden, 12)).unwrap();
    let wgu = k.stream.clone_htod(&rndbf16(2 * inter * hidden, 13)).unwrap();
    let wdn = k.stream.clone_htod(&rndbf16(hidden * inter, 14)).unwrap();
    let wsg = k.stream.clone_htod(&rndbf16(hidden, 15)).unwrap();
    let mut act = k.stream.alloc_zeros::<bf16>(inter).unwrap();
    let mut shared = k.stream.alloc_zeros::<bf16>(hidden).unwrap();
    let mut sg = k.stream.alloc_zeros::<bf16>(1).unwrap();
    bench(k, "moe shared(gu+dn+sgate)", 2 * inter * hidden * 2 + hidden * inter * 2 + hidden * 2, 30, || {
        proj_t1_epi(k, &x, &wgu, &mut act, 2 * inter, hidden, 1, 4, 0.0, 0.0);
        proj_t1(k, &act, &wdn, &mut shared, hidden, inter, 1);
        proj_t1_epi(k, &x, &wsg, &mut sg, 1, hidden, 1, 2, 1.0, 1.0);
        k.row_scale_inplace_bf16(&mut shared, &sg, 1, hidden);
    });
}

/// 专家段（t=1、E=10 唯一专家、Q4 槽、`slotsm` 单行核）：gu 一发 + dn 一发。
fn moe_experts(k: &Kernels, e: usize, inter: usize, hidden: usize) {
    let blob = 4 * 1024 * 1024; // 槽距放大，确保 10 组不共享 L2 行
    let mut slots = unsafe { k.stream.alloc::<u8>(e * blob) }.unwrap();
    k.stream.memset_zeros(&mut slots).unwrap();
    let ids: Vec<i32> = (0..e as i32).collect();
    let mut goff = Vec::with_capacity(e + 1);
    for g in 0..=e {
        goff.push(g as i32); // 每组 1 行
    }
    let ids_d = k.stream.clone_htod(&ids).unwrap();
    let goff_d = k.stream.clone_htod(&goff).unwrap();
    let ids_v = ids_d.slice(0..e);
    let goff_v = goff_d.slice(0..=e);
    let x = k.stream.clone_htod(&rndbf16(hidden, 16)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(e * 2 * inter).unwrap();
    bench(k, "moe experts gu ×10 (Q4)", e * 2 * inter * hidden / 2, 30, || {
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &slots, &ids_v, &x, &goff_v, &mut y, e, 2 * inter, hidden, hidden as i32, hidden,
            blob as u64, 0, 0, 1, 0,
        ).unwrap();
    });
    let mut act = k.stream.alloc_zeros::<bf16>(e * inter).unwrap();
    let mut o = k.stream.alloc_zeros::<bf16>(e * hidden).unwrap();
    bench(k, "moe experts dn ×10 (Q4)", e * hidden * inter / 2, 30, || {
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &slots, &ids_v, &act, &goff_v, &mut o, e, hidden, inter, inter as i32, inter,
            blob as u64, 0, 0, 1, 0,
        );
    });
}

/// stem（t=1）：两条 [N,N] GEMV + 两个 RMSNorm + hc_repeat + add。
fn stem(k: &Kernels, n: usize, hc: usize) {
    let emb = k.stream.clone_htod(&rndbf16(n, 17)).unwrap();
    let rin = k.stream.clone_htod(&rndbf16(hc * n, 18)).unwrap();
    let wfc = k.stream.clone_htod(&rndbf16(n * n, 19)).unwrap();
    let gn = k.stream.clone_htod(&rndbf16(n, 20)).unwrap();
    let gh = k.stream.clone_htod(&rndbf16(hc * n, 21)).unwrap();
    let mut en = k.stream.alloc_zeros::<bf16>(n).unwrap();
    let mut e2 = k.stream.alloc_zeros::<bf16>(n).unwrap();
    let mut hn = k.stream.alloc_zeros::<bf16>(hc * n).unwrap();
    let mut h2 = k.stream.alloc_zeros::<bf16>(hc * n).unwrap();
    let mut e2b = k.stream.alloc_zeros::<bf16>(hc * n).unwrap();
    let mut r = k.stream.alloc_zeros::<bf16>(hc * n).unwrap();
    bench(k, "stem(2×[N,N]+4小核)", 2 * n * n * 2, 30, || {
        k.grouped_rmsnorm_bf16(&emb, &gn, &mut en, 1, n, n, 1e-6).unwrap();
        proj_t1(k, &en, &wfc, &mut e2, n, n, 1);
        k.grouped_rmsnorm_bf16(&rin, &gh, &mut hn, 1, hc * n, n, 1e-6).unwrap();
        // fc_hidden 的输入是 [hc, N] ⇒ rows=hc（t=hc），走瘦长 GEMM
        proj_t1(k, &hn, &wfc, &mut h2, n, n, hc);
        k.hc_repeat_bf16(&e2, &mut e2b, 1, n, hc).unwrap();
        k.add_bf16(&h2, &e2b, &mut r, hc * n);
    });
}

#[test]
fn t7_mtp_shape_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    eprintln!("[mtp-bw] === MTP draft 步（t=1）逐核上限 ===");
    stem(&k, 2560, 4);
    gr_5k(&k, "GR attn_gr 5核 (dim10240)", 10240, 2560, 320, 4, true);
    gr_5k(&k, "GR mlp_gr  5核 (dim10240)", 10240, 2560, 320, 4, true);
    gr_2k(&k, "GR 融合 2核 (dim10240)", 10240, 2560, 320, 4);
    gr_5k(&k, "head mixer GR 5核", 10240, 2560, 320, 4, false);
    gemv_t1(&k, "attn qkv+gate 四发 [6144/512/6144/512,2560]", 6144, 2560);
    gemv_t1(&k, "attn  q [6144,2560]", 6144, 2560);
    gemv_t1(&k, "attn  k/v [512,2560]", 512, 2560);
    gemv_t1(&k, "attn out [2560,6144]", 2560, 6144);
    gemv_t1(&k, "attn index_qk [640,2560]", 640, 2560);
    moe_shared(&k, 640, 2560);
    moe_experts(&k, 10, 640, 2560);
    eprintln!("[mtp-bw] === head（每步一次，不在 MTP 层内）===");
    gemv_q4_dense(&k, "head lm_head Q4 [248320,2560] m=1", 248320, 2560, 10);
    head_topk(&k, 248320);
}

// 【已删除·负结果留档】原 `t7_q4gemv_rows`：稠密 Q4 GEMV 的"行循环版"与旧核逐位对拍 + 带宽扫描。
// 2026-10-24 实测：n=248320 时旧核 591 GB/s，行循环版按 grid 160→5120 只有
// 278/437/420/505/518/537 GB/s ⇒ "限 grid 省 block 调度开销"这个猜想是错的，大 grid 更好。
// 核与封装已删除，测试随之移除；数据留在。

/// `topk_rows_part_bf16` 的**寄存器版 vs 旧版逐位对拍 + 速度**。
/// 判据：`pval`（f32 bits）与 `pidx` 逐元素全等，且 `topk_rows_fin` 的最终输出也全等。
/// 数据集覆盖面：随机 / 大量并列（量化到 8 档）/ 含 NaN、+Inf / 全相等。
#[test]
fn t7_topk_part_bitid() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    const VOCAB: usize = 248320;
    const G: usize = 64;
    const KK: usize = 20;
    for &rows in &[1usize, 6] {
        for &mode in &[0u8, 1, 2, 3] {
            let mut v: Vec<bf16> = rndbf16(rows * VOCAB, 0x51 + mode as u32);
            match mode {
                1 => {
                    // 大量并列：只保留 8 个不同值
                    for x in v.iter_mut() {
                        let q = (x.to_f32() * 64.0).round() / 64.0;
                        *x = bf16::from_f32(q);
                    }
                }
                2 => {
                    for (i, x) in v.iter_mut().enumerate() {
                        if i % 9973 == 0 {
                            *x = bf16::NAN;
                        } else if i % 6007 == 0 {
                            *x = bf16::INFINITY;
                        }
                    }
                }
                _ => {
                    for x in v.iter_mut() {
                        *x = bf16::from_f32(0.5);
                    }
                }
            }
            let logits = k.stream.clone_htod(&v).unwrap();
            let npart = rows * G * KK;
            let mut pa = k.stream.alloc_zeros::<f32>(npart).unwrap();
            let mut ia = k.stream.alloc_zeros::<i32>(npart).unwrap();
            let mut pb = k.stream.alloc_zeros::<f32>(npart).unwrap();
            let mut ib = k.stream.alloc_zeros::<i32>(npart).unwrap();
            let mut ba = k.stream.alloc_zeros::<i32>(1).unwrap();
            let mut bb = k.stream.alloc_zeros::<i32>(1).unwrap();
            k.topk_rows_part_bf16_legacy(&logits, &mut pa, &mut ia, &mut ba, rows, VOCAB, G, KK)
                .unwrap();
            k.topk_rows_part_bf16(&logits, &mut pb, &mut ib, &mut bb, rows, VOCAB, G, KK)
                .unwrap();
            let (ha, hb) = (k.stream.clone_dtoh(&pa).unwrap(), k.stream.clone_dtoh(&pb).unwrap());
            let (ja, jb) = (k.stream.clone_dtoh(&ia).unwrap(), k.stream.clone_dtoh(&ib).unwrap());
            let baa = k.stream.clone_dtoh(&ba).unwrap()[0];
            let bab = k.stream.clone_dtoh(&bb).unwrap()[0];
            let mut diff = 0usize;
            for i in 0..npart {
                if ha[i].to_bits() != hb[i].to_bits() || ja[i] != jb[i] {
                    if diff < 6 {
                        eprintln!(
                            "[topk-bitid]   diff@{i}: 旧 {:.9}/{:#x}/idx{} vs 新 {:.9}/{:#x}/idx{}",
                            ha[i], ha[i].to_bits(), ja[i], hb[i], hb[i].to_bits(), jb[i]
                        );
                    }
                    diff += 1;
                }
            }
            if diff > 0 {
                let blk = 1092 / KK;
                let mut oldl: Vec<(u32, i32)> =
                    (0..KK).map(|s| (ha[blk * KK + s].to_bits(), ja[blk * KK + s])).collect();
                let mut newl: Vec<(u32, i32)> =
                    (0..KK).map(|s| (hb[blk * KK + s].to_bits(), jb[blk * KK + s])).collect();
                eprintln!("[topk-bitid]   block {blk} 旧={:?}", oldl.iter().map(|x| x.1).collect::<Vec<_>>());
                eprintln!("[topk-bitid]   block {blk} 新={:?}", newl.iter().map(|x| x.1).collect::<Vec<_>>());
                // 真全序（值降序、下标升序）
                let key = |x: &(u32, i32)| (x.0, x.1);
                oldl.sort_by(|a, b| key(b).partial_cmp(&key(a)).unwrap());
                newl.sort_by(|a, b| key(b).partial_cmp(&key(a)).unwrap());
                eprintln!(
                    "[topk-bitid]   同集合={}",
                    oldl == newl
                );
            }
            // 最终归并也同一份输入（pval/pidx 已逐位相同 ⇒ fin 必然相同，这里仍跑一遍）
            let mut oa = k.stream.alloc_zeros::<f32>(rows * KK).unwrap();
            let mut oia = k.stream.alloc_zeros::<i32>(rows * KK).unwrap();
            let mut ob = k.stream.alloc_zeros::<f32>(rows * KK).unwrap();
            let mut oib = k.stream.alloc_zeros::<i32>(rows * KK).unwrap();
            k.topk_rows_fin(&pa, &ia, &mut oa, &mut oia, rows, G, KK).unwrap();
            k.topk_rows_fin(&pb, &ib, &mut ob, &mut oib, rows, G, KK).unwrap();
            let fa = k.stream.clone_dtoh(&oa).unwrap();
            let fb = k.stream.clone_dtoh(&ob).unwrap();
            let ga = k.stream.clone_dtoh(&oia).unwrap();
            let gb = k.stream.clone_dtoh(&oib).unwrap();
            let fin_same = fa.iter().zip(fb.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
                && ga == gb;
            eprintln!(
                "[topk-bitid] rows={rows} mode={mode} 不一致={diff}/{npart}｜fin 相同={fin_same}｜bad {baa} vs {bab}"
            );
            assert_eq!(diff, 0, "rows={rows} mode={mode}：寄存器版与旧版不一致");
            assert!(fin_same, "rows={rows} mode={mode}：fin 输出不一致");
            assert_eq!(baa, bab);
        }
        // 速度（真实形状：t=1 线上就是 64 个 block）
        let v = rndbf16(rows * VOCAB, 7);
        let logits = k.stream.clone_htod(&v).unwrap();
        let mut p = k.stream.alloc_zeros::<f32>(rows * G * KK).unwrap();
        let mut i = k.stream.alloc_zeros::<i32>(rows * G * KK).unwrap();
        let mut b = k.stream.alloc_zeros::<i32>(1).unwrap();
        bench(&k, &format!("topk part 旧核 rows={rows}"), VOCAB * 2, 10, || {
            k.topk_rows_part_bf16_legacy(&logits, &mut p, &mut i, &mut b, rows, VOCAB, G, KK)
                .unwrap();
        });
        bench(&k, &format!("topk part 寄存器版 rows={rows}"), VOCAB * 2, 10, || {
            k.topk_rows_part_bf16(&logits, &mut p, &mut i, &mut b, rows, VOCAB, G, KK).unwrap();
        });
    }
}
