//! **草稿相位（MTP 一层，t=1）两个"异常"的分母**（2026-10-24）。
//!
//! 背景：`LINGXI_DRAFT_PROF=1` 的分账里 attn 段 ≈200 µs、MoE ≈216 µs。先量分母再判断：
//!
//! * **A. MoE 专家段（`rows==1`）**：MTP arena 是 Q4-only（`q4_alt=None`）⇒ 线上走
//! `gemv_q4g64_slots_rows_bf16`（`slots` 体：c_lut 字节查表 + PF=4 移位预取），
//! **没有**吃到 的 `slotsm_t2_bf16`（那条只在 `slotsm` 分组核里）。
//! 本用例给出：旧核 / `slotsm`（→ t2 档）/ 同形纯读（`read_probe_slots`，地址流与
//! `slots` 体逐字节相同）三者的 µs 与 GB/s，并做**逐位对拍**（旧核 vs slotsm）。
//! * **B. QSA gather 注意力**：t=1 的行数只有主干验证批的 1/6（grid=(t, heads/NH) ⇒
//! 24 vs 144 block）。同一 topk=512 下量 t=1 / t=6 的生产核，回答"行数少反而慢"
//! 到底是核的问题还是并行度/工作量的问题；附同字节流式读上限。
//!
//! 口径：`--test-threads=1`，同进程预热 + 交错 A/B + best_of3。
//!
//! `bash
//! export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib   # cublas/nvrtc 不在默认路径时按需追加
//! cargo test --release -p lingxi-kernels --features gpu --test t19_draft_t1_bw \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::{bf16, f16};
use lingxi_kernels::Kernels;

// ── 线上形状（MTP 专家 = 主干同值）──
const HIDDEN: usize = 2560;
const INTER: usize = 640;
const GU_N: usize = 2 * INTER; // gu：n=1280、k=2560
const E: usize = 10; // 草稿一步的唯一专家数

// ── QSA ──
const HEADS: usize = 24;
const KVH: usize = 2;
const HD: usize = 256;
const COMPRESS: usize = 4;
const TOPK: usize = 2048 / COMPRESS; // budget/compress = 512

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

/// `matrix_layout`（Q4G64）的同一公式：payload 段 = [rows×row_bytes | scales]。
fn layout(rows: u64, cols: u64) -> (u64, u64, u64) {
    let k_pad = align_up(cols, 128);
    let groups = k_pad / 64;
    let row_bytes = groups * 32;
    let base = rows * row_bytes;
    let scale_off = align_up(base, 256);
    let payload = scale_off + rows * groups * 2;
    (row_bytes, scale_off, payload)
}

fn rnd_bytes(n: usize) -> Vec<u8> {
    let mut s: u32 = 0x1234_5678;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFF) as u8
        })
        .collect()
}

/// 填两个 payload 段（[row_bytes×rows | scales]），scale 取合理值（避免 fp16 NaN/denorm）。
fn fill_payload(b: &mut [u8], rows: usize, groups: usize, scale_off: u64) {
    for i in 0..rows * groups {
        let h = f16::from_f32(0.002 + 0.001 * ((i % 23) as f32));
        let off = scale_off as usize + 2 * i;
        b[off..off + 2].copy_from_slice(&h.to_le_bytes());
    }
}

fn rnd_bf16(n: usize, seed: u32) -> Vec<bf16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_f32(((s >> 8) as f32 / 16777216.0 - 0.5) * 0.2)
        })
        .collect()
}

fn best_of<F: FnMut()>(k: &Kernels, iters: usize, mut f: F) -> f64 {
    for _ in 0..5 {
        f();
    }
    k.stream.synchronize().unwrap();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        k.stream.synchronize().unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    best
}

/// 冷 L2 + 单发（草稿每步的真实上下文：上一段刚把 L2 冲掉）。先写 48 MB 脏缓冲再量一发，5 次取 min。
fn best_cold<F: FnMut()>(k: &Kernels, dirty: &mut CudaSlice<u8>, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..5 {
        k.stream.memset_zeros(dirty).unwrap();
        k.stream.synchronize().unwrap();
        let t0 = Instant::now();
        f();
        k.stream.synchronize().unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e6);
    }
    best
}

fn report(tag: &str, bytes: f64, us: f64) {
    eprintln!(
        "[t19] {tag:<46}｜{us:8.2} µs｜{:7.1} GB/s",
        bytes / (us * 1e-6) / 1e9
    );
}

fn bits_diff(a: &[bf16], b: &[bf16]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

// ───────────────────────── A. MoE（rows==1） ─────────────────────────
#[test]
fn t19_moe_t1_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let (gu_rb, gu_scale, gu_pay) = layout(GU_N as u64, HIDDEN as u64);
    let (dn_rb, dn_scale, dn_pay) = layout(HIDDEN as u64, INTER as u64);
    // 与 MtpArena 同布局：每专家一块 blob = [gu payload | dn payload]
    let blob = gu_pay + dn_pay;
    let mut host = rnd_bytes((blob * E as u64) as usize);
    for e in 0..E {
        let base = e * blob as usize;
        fill_payload(&mut host[base..base + gu_pay as usize], GU_N, (gu_rb * 2 / 64) as usize, 0);
        let dn_base = base + gu_pay as usize;
        fill_payload(
            &mut host[dn_base..dn_base + dn_pay as usize],
            HIDDEN,
            (dn_rb * 2 / 64) as usize,
            0,
        );
    }
    let slots: CudaSlice<u8> = k.stream.clone_htod(&host).unwrap();
    // 单发给纯读探针的 dn 副本（探针按 e*blob 取基址，不能带偏移）
    let mut dn_only = vec![0u8; (dn_pay * E as u64) as usize];
    for e in 0..E {
        let src = e * blob as usize + gu_pay as usize;
        let dst = e * dn_pay as usize;
        dn_only[dst..dst + dn_pay as usize]
            .copy_from_slice(&host[src..src + dn_pay as usize]);
    }
    let dn_slots: CudaSlice<u8> = k.stream.clone_htod(&dn_only).unwrap();

    let ids: Vec<i32> = (0..E as i32).collect();
    let ids_d = k.stream.clone_htod(&ids).unwrap();
    let ids_v = ids_d.slice(0..E);
    let goff_h: Vec<i32> = (0..=E as i32).collect();
    let goff_d = k.stream.clone_htod(&goff_h).unwrap();
    let goff_v = goff_d.slice(0..E + 1);

    let x_gu_h = rnd_bf16(HIDDEN, 11);
    let x_gu = k.stream.clone_htod(&x_gu_h).unwrap();
    let x_dn_h = rnd_bf16(E * INTER, 12);
    let x_dn = k.stream.clone_htod(&x_dn_h).unwrap();

    let mut y_old = k.stream.alloc_zeros::<bf16>(E * GU_N).unwrap();
    let mut y_new = k.stream.alloc_zeros::<bf16>(E * GU_N).unwrap();
    let mut y_ref = k.stream.alloc_zeros::<bf16>(E * GU_N).unwrap();

    // ── 逐位对拍：旧 slots 核 vs slotsm（TIER 默认 1 ⇒ t2 档）──
    k.gemv_q4g64_slots_rows_bf16(
        &slots, &ids_d, &x_gu, &mut y_ref, E, GU_N, HIDDEN, HIDDEN as i32, 0, blob, 0, gu_scale, 1,
    )
    .unwrap();
    k.gemv_q4g64_slotsm_rpw_bf16_view(
        &slots, &ids_v, &x_gu, &goff_v, &mut y_new, E, GU_N, HIDDEN, HIDDEN as i32, 0, blob, 0,
        gu_scale, 1, 0,
    )
    .unwrap();
    k.stream.synchronize().unwrap();
    let a: Vec<bf16> = k.stream.clone_dtoh(&y_ref).unwrap();
    let b: Vec<bf16> = k.stream.clone_dtoh(&y_new).unwrap();
    eprintln!("[t19] gu 逐位对拍（slots vs slotsm_t2）：{} / {} 元素不同", bits_diff(&a, &b), a.len());

    // dn：a_stride=inter（每个对读自己那行激活）
    let mut o_old = k.stream.alloc_zeros::<bf16>(E * HIDDEN).unwrap();
    let mut o_new = k.stream.alloc_zeros::<bf16>(E * HIDDEN).unwrap();
    k.gemv_q4g64_slots_rows_bf16(
        &slots, &ids_d, &x_dn, &mut o_old, E, HIDDEN, INTER, INTER as i32, INTER, blob, gu_pay,
        dn_scale, 1,
    )
    .unwrap();
    k.gemv_q4g64_slotsm_rpw_bf16_view(
        &slots, &ids_v, &x_dn, &goff_v, &mut o_new, E, HIDDEN, INTER, INTER as i32, INTER, blob,
        gu_pay, dn_scale, 1, 0,
    )
    .unwrap();
    k.stream.synchronize().unwrap();
    let c: Vec<bf16> = k.stream.clone_dtoh(&o_old).unwrap();
    let d: Vec<bf16> = k.stream.clone_dtoh(&o_new).unwrap();
    eprintln!("[t19] dn 逐位对拍（slots vs slotsm_t2）：{} / {} 元素不同", bits_diff(&c, &d), c.len());

    let gu_bytes = (gu_pay * E as u64) as f64;
    let dn_bytes = (dn_pay * E as u64) as f64;
    let mut rout = k.stream.alloc_zeros::<f32>(E * GU_N).unwrap();

    // ── 交错 A/B（每轮 old→new，取各自 min）──
    let (mut t_old, mut t_new, mut t_probe0, mut t_probe1) =
        (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
    for _ in 0..3 {
        t_old = t_old.min(best_of(&k, 20, || {
            k.gemv_q4g64_slots_rows_bf16(
                &slots, &ids_d, &x_gu, &mut y_old, E, GU_N, HIDDEN, HIDDEN as i32, 0, blob, 0,
                gu_scale, 1,
            )
            .unwrap();
        }));
        t_new = t_new.min(best_of(&k, 20, || {
            k.gemv_q4g64_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x_gu, &goff_v, &mut y_new, E, GU_N, HIDDEN, HIDDEN as i32, 0,
                blob, 0, gu_scale, 1, 0,
            )
            .unwrap();
        }));
        t_probe0 = t_probe0.min(best_of(&k, 20, || {
            k.read_probe_slots(&slots, &mut rout, E, GU_N, gu_rb as usize, blob, 0).unwrap();
        }));
        t_probe1 = t_probe1.min(best_of(&k, 20, || {
            k.read_probe_slots(&slots, &mut rout, E, GU_N, gu_rb as usize, blob, 1).unwrap();
        }));
    }
    eprintln!("[t19] ── gu（n={GU_N} k={HIDDEN}，E={E}，m=1）──");
    report("旧核 slots（线上 t=1 走的）", gu_bytes, t_old);
    report("slotsm（t2 档，逐位相同）", gu_bytes, t_new);
    report("同形纯读 mode0（4B/lane，与 slots 同地址流）", gu_bytes, t_probe0);
    report("同形纯读 mode1（16B/lane 上限）", gu_bytes, t_probe1);
    for rpw in [2usize, 4] {
        let us = best_of(&k, 20, || {
            k.gemv_q4g64_slots_rows_bf16(
                &slots, &ids_d, &x_gu, &mut y_old, E, GU_N, HIDDEN, HIDDEN as i32, 0, blob, 0,
                gu_scale, rpw,
            )
            .unwrap();
        });
        report(&format!("旧核 slots rpw={rpw}"), gu_bytes, us);
        let us = best_of(&k, 20, || {
            k.gemv_q4g64_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x_gu, &goff_v, &mut y_new, E, GU_N, HIDDEN, HIDDEN as i32, 0,
                blob, 0, gu_scale, rpw, 0,
            )
            .unwrap();
        });
        report(&format!("slotsm rpw={rpw}"), gu_bytes, us);
    }

    let mut t_old = f64::MAX;
    let mut t_new = f64::MAX;
    let mut t_probe = f64::MAX;
    for _ in 0..3 {
        t_old = t_old.min(best_of(&k, 20, || {
            k.gemv_q4g64_slots_rows_bf16(
                &slots, &ids_d, &x_dn, &mut o_old, E, HIDDEN, INTER, INTER as i32, INTER, blob,
                gu_pay, dn_scale, 1,
            )
            .unwrap();
        }));
        t_new = t_new.min(best_of(&k, 20, || {
            k.gemv_q4g64_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x_dn, &goff_v, &mut o_new, E, HIDDEN, INTER, INTER as i32, INTER,
                blob, gu_pay, dn_scale, 1, 0,
            )
            .unwrap();
        }));
        t_probe = t_probe.min(best_of(&k, 20, || {
            k.read_probe_slots(&dn_slots, &mut rout, E, HIDDEN, dn_rb as usize, dn_pay, 0).unwrap();
        }));
    }
    eprintln!("[t19] ── dn（n={HIDDEN} k={INTER}，E={E}，m=1）──");
    report("旧核 slots", dn_bytes, t_old);
    report("slotsm（t2 档，逐位相同）", dn_bytes, t_new);
    report("同形纯读 mode0", dn_bytes, t_probe);
    for rpw in [2usize, 4] {
        let us = best_of(&k, 20, || {
            k.gemv_q4g64_slots_rows_bf16(
                &slots, &ids_d, &x_dn, &mut o_old, E, HIDDEN, INTER, INTER as i32, INTER, blob,
                gu_pay, dn_scale, rpw,
            )
            .unwrap();
        });
        report(&format!("旧核 slots rpw={rpw}"), dn_bytes, us);
        let us = best_of(&k, 20, || {
            k.gemv_q4g64_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x_dn, &goff_v, &mut o_new, E, HIDDEN, INTER, INTER as i32, INTER,
                blob, gu_pay, dn_scale, rpw, 0,
            )
            .unwrap();
        });
        report(&format!("slotsm rpw={rpw}"), dn_bytes, us);
    }

    eprintln!(
        "[t19] 合计权重 payload：gu {:.2} MB + dn {:.2} MB = {:.2} MB",
        gu_bytes / 1e6,
        dn_bytes / 1e6,
        (gu_bytes + dn_bytes) / 1e6
    );

    // ── 冷 L2 单发：草稿每步的真实上下文（attention_step 刚读过 100 MB）──
    let mut dirty = k.stream.alloc_zeros::<u8>(48 * 1024 * 1024).unwrap();
    eprintln!("[t19] ── 冷 L2 单发（先 memset 48 MB 脏缓冲；5 次取 min）──");
    let c_old = best_cold(&k, &mut dirty, || {
        k.gemv_q4g64_slots_rows_bf16(
            &slots, &ids_d, &x_gu, &mut y_old, E, GU_N, HIDDEN, HIDDEN as i32, 0, blob, 0,
            gu_scale, 1,
        )
        .unwrap();
    });
    let c_new = best_cold(&k, &mut dirty, || {
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &slots, &ids_v, &x_gu, &goff_v, &mut y_new, E, GU_N, HIDDEN, HIDDEN as i32, 0, blob,
            0, gu_scale, 1, 0,
        )
        .unwrap();
    });
    let c_probe = best_cold(&k, &mut dirty, || {
        k.read_probe_slots(&slots, &mut rout, E, GU_N, gu_rb as usize, blob, 0).unwrap();
    });
    report("冷 · gu 旧核 slots", gu_bytes, c_old);
    report("冷 · gu slotsm（t2 档）", gu_bytes, c_new);
    report("冷 · gu 同形纯读 mode0", gu_bytes, c_probe);
    let c_old = best_cold(&k, &mut dirty, || {
        k.gemv_q4g64_slots_rows_bf16(
            &slots, &ids_d, &x_dn, &mut o_old, E, HIDDEN, INTER, INTER as i32, INTER, blob, gu_pay,
            dn_scale, 1,
        )
        .unwrap();
    });
    let c_probe = best_cold(&k, &mut dirty, || {
        k.read_probe_slots(&dn_slots, &mut rout, E, HIDDEN, dn_rb as usize, dn_pay, 0).unwrap();
    });
    report("冷 · dn 旧核 slots", dn_bytes, c_old);
    report("冷 · dn 同形纯读 mode0", dn_bytes, c_probe);
}

// ───────────────────────── B. QSA gather（t=1 vs t=6） ─────────────────────────

/// **MTP 注意力投影（t=1，bf16）**：MTP 层的 `query_key_gate_value_q4/output_q4` 都是 `None`
/// （`build_mtp_layer`）⇒ 走 `proj_t1` 的 `gemv_bf16_row8_epi`，权重是 **bf16**
/// （13312×2560 = 68.2 MB + 2560×6144 = 31.5 MB）。这里量它离"同形纯读"多远。
const N_QKVG: usize = 13312; // 2*6144(q) + 2*512(k/v)
const K_ATTN: usize = 2560;
const N_OUT: usize = 2560;
const K_OUT: usize = 6144;

#[test]
fn t19_mtp_attn_proj_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let mut dirty = k.stream.alloc_zeros::<u8>(48 * 1024 * 1024).unwrap();
    for (n, kk, tag) in [(N_QKVG, K_ATTN, "q|k|gate|v 13312×2560"), (N_OUT, K_OUT, "out 2560×6144")] {
        let bytes = (n * kk * 2) as f64;
        let w = k.stream.clone_htod(&rnd_bf16(n * kk, 31)).unwrap();
        let x = k.stream.clone_htod(&rnd_bf16(kk, 32)).unwrap();
        let mut y = k.stream.alloc_zeros::<bf16>(n).unwrap();
        let mut out = k.stream.alloc_zeros::<f32>(n).unwrap();
        let (wp, _g1) = w.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        let (yp, _g3) = y.device_ptr_mut(&k.stream);
        let hot = best_of(&k, 20, || {
            k.gemv_bf16_row8_epi_ptr(wp, xp, yp, n, kk, 0, 0.0, 0.0).unwrap();
        });
        let rd = best_of(&k, 20, || {
            k.probe_read_row8_bf16(wp, &mut out, n, kk).unwrap();
        });
        let st = best_of(&k, 20, || {
            k.probe_read_stream_bf16(wp, &mut out, n * kk).unwrap();
        });
        let cold = best_cold(&k, &mut dirty, || {
            k.gemv_bf16_row8_epi_ptr(wp, xp, yp, n, kk, 0, 0.0, 0.0).unwrap();
        });
        let cold_rd = best_cold(&k, &mut dirty, || {
            k.probe_read_row8_bf16(wp, &mut out, n, kk).unwrap();
        });
        eprintln!("[t19] ── MTP attn proj t=1：{tag}（{:.1} MB bf16）──", bytes / 1e6);
        report("生产 gemv_bf16_row8_epi", bytes, hot);
        report("同形纯读 row8", bytes, rd);
        report("同字节流式读（机器上限）", bytes, st);
        report("冷 L2 · 生产核", bytes, cold);
        report("冷 L2 · 同形纯读", bytes, cold_rd);
    }
}

fn attn_case(k: &Kernels, nb: usize, t: usize, iters: usize, cold: bool) -> (f64, f64) {
    let len = nb * COMPRESS;
    // 末尾再留 t 行给"尾部不足一块"的 token（vis_base = len-1 ⇒ 行 i 有 i 个尾部 token）
    let kn_v = rnd_bf16((len + t) * KVH * HD, 21);
    let kn = k.stream.clone_htod(&kn_v).unwrap();
    let v = k.stream.clone_htod(&kn_v).unwrap();
    let tab_h: Vec<i32> = (0..nb as i32).collect();
    let tab = k.stream.clone_htod(&tab_h).unwrap();
    let sel_h: Vec<i32> = (0..t * nb).map(|i| (i % nb) as i32).collect();
    let sel = k.stream.clone_htod(&sel_h).unwrap();
    let cnt_h = vec![nb as i32; t];
    let cnt = k.stream.clone_htod(&cnt_h).unwrap();
    let q = k.stream.alloc_zeros::<bf16>(t * HEADS * HD).unwrap();
    let mut o = k.stream.alloc_zeros::<bf16>(t * HEADS * HD).unwrap();
    let vis = len.saturating_sub(1); // 块全部可见
    let scale = 1.0f32 / (HD as f32).sqrt();

    let qv = q.slice(0..t * HEADS * HD);
    let mut ov = o.slice_mut(0..t * HEADS * HD);
    let kv = kn.slice(0..(len + t) * KVH * HD);
    let vv = v.slice(0..(len + t) * KVH * HD);
    let tabv = tab.slice(0..nb);
    let selv = sel.slice(0..t * nb);
    let cntv = cnt.slice(0..t);
    let us = if cold {
        let mut dirty = k.stream.alloc_zeros::<u8>(48 * 1024 * 1024).unwrap();
        best_cold(k, &mut dirty, || {
            k.attn_gather_bf16(
                &qv, &kv, &vv, &tabv, &selv, &cntv, &mut ov, t, HEADS, KVH, HD, scale, vis, nb,
                COMPRESS.trailing_zeros() as usize, nb, 1,
            )
            .unwrap();
        })
    } else {
        best_of(k, iters, || {
            k.attn_gather_bf16(
                &qv, &kv, &vv, &tabv, &selv, &cntv, &mut ov, t, HEADS, KVH, HD, scale, vis, nb,
                COMPRESS.trailing_zeros() as usize, nb, 1,
            )
            .unwrap();
        })
    };
    // HBM 口径：每行 K+V = 2 × (nb*COMPRESS) × kv_heads × head_dim × 2 B
    let bytes = (t * 2 * len * KVH * HD * 2) as f64;
    (us, bytes)
}

#[test]
fn t19_qsa_gather_t1_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    eprintln!("[t19] ── QSA gather（heads=24 kv=2 hd=256 NH=1，grid=(t,24)）──");
    for (nb, tag) in [(TOPK, "topk=512（预算打满）"), (5, "nb=5（短上下文，草稿实际）"), (29, "nb=29")] {
        let (u1, b1) = attn_case(&k, nb, 1, 20, false);
        let (u6, b6) = attn_case(&k, nb, 6, 20, false);
        report(&format!("{tag} t=1"), b1, u1);
        report(&format!("{tag} t=6"), b6, u6);
        eprintln!(
            "[t19]   ⇒ t=1 {u1:.1} µs / t=6 {u6:.1} µs（t=6 每行 {:.1} µs，t=1 每行 {:.1} µs）",
            u6 / 6.0,
            u1
        );
    }
    let (c1, b1) = attn_case(&k, TOPK, 1, 20, true);
    report("冷 L2 · topk=512 t=1", b1, c1);
    // 同字节流式读上限（t=1、topk=512 的那 4.2 MB）
    let len = TOPK * COMPRESS;
    let kn_v = rnd_bf16(len * KVH * HD, 21);
    let kn = k.stream.clone_htod(&kn_v).unwrap();
    let mut out = k.stream.alloc_zeros::<f32>(4096).unwrap();
    let (p, _g) = kn.device_ptr(&k.stream);
    let us = best_of(&k, 20, || {
        k.probe_read_stream_bf16(p, &mut out, len * KVH * HD).unwrap();
    });
    report("同字节流式读（2.1 MB，机器上限）", (len * KVH * HD * 2) as f64, us);
}

// ───────────────── D. 草稿 head（lm_head，248320×2560，m=1）的三条候选 ─────────────────
//
// 为什么单列一节：草稿一步 `LINGXI_DRAFT_PROF=1` 的**层外 = 1.16~1.29 ms**，其中
// `lm_head` 是绝对大头（E 的 in-situ 事件口径 911 µs / 348 GB/s），而它是
// **唯一"草稿允许非逐位"的大件**（被接受的前缀永远来自主干 argmax ⇒ 换核不改文本）。
// 这里把三条候选放在**同进程、同一份权重、热循环背靠背**的尺子上：
// * 生产 `gemv_q4g64_dense_epi`（4 B/lane + c_lut 字节查表，**逐位**）
// * `u128_r1`（16 B/lane，bf16 激活，重结合）
// * `u128_r1_dp4a`（16 B/lane + int8 激活 + `__dp4a`，有损）
// 外加同形纯读分母（`read_probe_slots` mode 0/1，地址流与生产核 4 B/lane 相同）。
const HEAD_N: usize = 248320;
const HEAD_K: usize = 2560;

#[test]
fn t19_draft_head_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let (rb, scale_off, pay) = layout(HEAD_N as u64, HEAD_K as u64);
    let mut host = rnd_bytes(pay as usize);
    fill_payload(&mut host, HEAD_N, HEAD_K / 64, scale_off);
    let w = k.stream.clone_htod(&host).unwrap();
    drop(host);
    let x_v = rnd_bf16(HEAD_K, 41);
    let x = k.stream.clone_htod(&x_v).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(HEAD_N).unwrap();
    // 生产核走 `_ptr` 入口（`yp` 会独占 `y` 的可变借用），单独给一块输出
    let mut y2 = k.stream.alloc_zeros::<bf16>(HEAD_N).unwrap();
    let mut rout = k.stream.alloc_zeros::<f32>(HEAD_N).unwrap();
    let ids1 = k.stream.clone_htod(&[0i32]).unwrap();
    let goff = k.stream.clone_htod(&[0i32, 1i32]).unwrap();
    let ids_v = ids1.slice(0..1);
    let go_v = goff.slice(0..2);
    let mut xq = k.stream.alloc_zeros::<u8>(HEAD_K).unwrap();
    let mut sx = k.stream.alloc_zeros::<f32>(1).unwrap();
    k.convert_bf16_q8_rs(&x, &mut xq, &mut sx, 1, HEAD_K).unwrap();
    let payload_bytes = (HEAD_N * rb as usize) as f64;
    let total_bytes = payload_bytes + (HEAD_N * (HEAD_K / 64) * 2) as f64;
    let (wp, _a) = w.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yp, _c) = y2.device_ptr_mut(&k.stream);
    let mut dirty = k.stream.alloc_zeros::<u8>(48 * 1024 * 1024).unwrap();

    eprintln!(
        "[t19] ── 草稿 head {HEAD_N}x{HEAD_K}（payload {:.1} MB + scale {:.1} MB）──",
        payload_bytes / 1e6,
        total_bytes / 1e6 - payload_bytes / 1e6
    );
    let prep = best_of(&k, 20, || {
        k.probe_read_stream_bf16(wp, &mut rout, HEAD_N * HEAD_K / 8).unwrap();
    });
    report("①同字节流式读（1 B/权重，机器上限 900）", payload_bytes, prep);
    for (mode, tag) in [(0, "②同形纯读 mode0（4 B/lane，生产核地址流）"), (1, "③同形纯读 mode1（+scale）")] {
        let us = best_of(&k, 10, || {
            k.read_probe_slots(&w, &mut rout, 1, HEAD_N, rb as usize, 0, mode).unwrap();
        });
        report(tag, total_bytes, us);
    }
    let us_prod = best_of(&k, 10, || {
        k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, HEAD_N, HEAD_K, HEAD_K as i32, 0, scale_off, 0, 0.0, 0.0)
            .unwrap();
    });
    let us_u128 = best_of(&k, 10, || {
        k.gemv_q4g64_struct_exp(
            10, &w, &ids_v, &x, &go_v, &mut y, 1, HEAD_N, HEAD_K, HEAD_K as i32, 0, 0, 0, scale_off,
        )
        .unwrap();
    });
    let us_dp4a = best_of(&k, 10, || {
        k.gemv_q4g64_struct_exp_dp4a(
            &w, &ids_v, &x, &go_v, &mut y, 1, HEAD_N, HEAD_K, HEAD_K as i32, 0, 0, 0, scale_off,
            &xq, &sx,
        )
        .unwrap();
    });
    report("④生产 gemv_q4g64_dense_epi（逐位）", total_bytes, us_prod);
    report("⑤u128_r1（16 B/lane，bf16 激活）", total_bytes, us_u128);
    report("⑥u128_r1_dp4a（16 B/lane，int8 激活）", total_bytes, us_dp4a);
    let c_prod = best_cold(&k, &mut dirty, || {
        k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, HEAD_N, HEAD_K, HEAD_K as i32, 0, scale_off, 0, 0.0, 0.0)
            .unwrap();
    });
    let c_u128 = best_cold(&k, &mut dirty, || {
        k.gemv_q4g64_struct_exp(
            10, &w, &ids_v, &x, &go_v, &mut y, 1, HEAD_N, HEAD_K, HEAD_K as i32, 0, 0, 0, scale_off,
        )
        .unwrap();
    });
    report("冷 L2 · 生产", total_bytes, c_prod);
    report("冷 L2 · u128_r1", total_bytes, c_u128);
    eprintln!(
        "[t19] ⇒ 相对生产：u128 {:.1}×、dp4a {:.1}×（热循环）；冷 L2 u128 {:.1}×",
        us_prod / us_u128,
        us_prod / us_dp4a,
        c_prod / c_u128
    );

    // ── ⑦ Q3G128 同形档──
    // 几何：row_bytes = (k/128)*48 = 960；scales = n*(k/128)*2 = 9,932,800；
    // payload = n*960 = 238,387,200 ⇒ 合计 248,320,000 B（Q4 的 0.735×）。
    // 判据（写进）：≤553.7 µs 才打平 u128，≤520 µs 才值得真做。
    {
        let groups = HEAD_K / 128;
        let row_bytes = groups * 48;
        let q3_pay = HEAD_N * row_bytes;
        let q3_scale = HEAD_N * groups * 2;
        let mut h = rnd_bytes(q3_pay + q3_scale);
        for i in 0..(HEAD_N * groups) {
            let v: u16 = 0x2C00; // f16 0.0625（有限、非 0）
            h[q3_pay + 2 * i] = (v & 0xff) as u8;
            h[q3_pay + 2 * i + 1] = (v >> 8) as u8;
        }
        let w3 = k.stream.clone_htod(&h).unwrap();
        drop(h);
        let q3_bytes = (q3_pay + q3_scale) as f64;
        let mut y3 = k.stream.alloc_zeros::<bf16>(HEAD_N).unwrap();
        let us_q3 = best_of(&k, 10, || {
            k.gemv_q3g128_slotsm_t2_view_gidx(
                &w3, &ids_v, &x, &go_v, &mut y3, 1, HEAD_N, HEAD_K, HEAD_K as i32, HEAD_K,
                q3_bytes as u64, 0, q3_pay as u64, 1, 0, 0, true,
            )
            .unwrap();
        });
        report("⑦Q3G128 t2_view（E=1 稠密，MB1/UNR2）", q3_bytes, us_q3);
        eprintln!(
            "[t19] ⇒ Q3 vs u128 = {:.2}×、vs dp4a = {:.2}×（要 ≤1.00 才值得做；≤0.94 才好过 dp4a）",
            us_u128 / us_q3,
            us_dp4a / us_q3
        );
    }
}


// ───────────────── E. Q4 专家 GEMV 的静态网格档 ─────────────────
//
// 与 Q3 的同名用例同构（`t13_q3_static_grid`）：同核体、只多一道 `e >= *ng_dev` 早退，
// 主机因此可以发 `grid.y = s`（上界）而不必先等 `npri/nalt`。这里量两件事：
// ① 逐位：tier 档（精确网格）vs `_ng` 档（静态网格）必须完全相同；
// ② 代价：多出来的空 block 值多少 GB/s（线上超发比 ≈ s/ng ≈ 1.4×）。
#[test]
fn t19_q4_static_grid() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    const E: usize = 40;
    const NG: usize = 30;
    const N: usize = 1280; // gu 形状：n = 2*inter
    const KK: usize = 2560;
    let (rb, scale_off, pay) = layout(N as u64, KK as u64);
    let mut one = rnd_bytes(pay as usize);
    fill_payload(&mut one, N, KK / 64, scale_off);
    let mut all = Vec::with_capacity(E * pay as usize);
    for _ in 0..E {
        all.extend_from_slice(&one);
    }
    let slots = k.stream.clone_htod(&all).unwrap();
    drop(all);
    let ids: Vec<i32> = (0..E as i32).collect();
    let ids_d = k.stream.clone_htod(&ids).unwrap();
    let ids_v = ids_d.slice(0..E);
    // 每组恰好 1 对 ⇒ s = E = 40（线上 s≈60/ng≈36，超发比同为 ~1.4×）
    let goff: Vec<i32> = (0..=E).map(|i| i as i32).collect();
    let goff_d = k.stream.clone_htod(&goff).unwrap();
    let goff_v = goff_d.slice(0..=E);
    let gidx_h: Vec<i32> = (0..NG as i32).collect();
    let gidx_d = k.stream.clone_htod(&gidx_h).unwrap();
    let (gidx_p, _g1) = gidx_d.device_ptr(&k.stream);
    let ng_d = k.stream.clone_htod(&[NG as i32]).unwrap();
    let (ng_p, _g2) = ng_d.device_ptr(&k.stream);
    let x = k.stream.clone_htod(&rnd_bf16(E * KK, 5)).unwrap();
    let mut y_e = k.stream.alloc_zeros::<bf16>(E * N).unwrap();
    let mut y_s = k.stream.alloc_zeros::<bf16>(E * N).unwrap();
    let bytes = NG as f64 * N as f64 * KK as f64 / 2.0;
    eprintln!(
        "[t19] ── Q4 静态网格 vs 精确网格（n{N} k{KK} E{E} ng{NG}，每行 {:.0} B）──",
        rb as f64
    );
    let us_e = best_of(&k, 40, || {
        k.gemv_q4g64_slotsm_rpw_bf16_view_gidx(
            &slots, &ids_v, &x, &goff_v, &mut y_e, NG, N, KK, KK as i32, KK, pay, 0, scale_off,
            1, 0, gidx_p,
        )
        .unwrap();
    });
    let us_s = best_of(&k, 40, || {
        k.gemv_q4g64_slotsm_t2_ng_view_gidx(
            &slots, &ids_v, &x, &goff_v, &mut y_s, E, N, KK, KK as i32, KK, pay, 0, scale_off,
            gidx_p, ng_p,
        )
        .unwrap();
    });
    let a: Vec<bf16> = k.stream.clone_dtoh(&y_e).unwrap();
    let b: Vec<bf16> = k.stream.clone_dtoh(&y_s).unwrap();
    let nd = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    report("Q4 精确网格（tier 档，e_uniq=ng）", bytes, us_e);
    report("Q4 静态网格（e_max=s + 设备上界）", bytes, us_s);
    eprintln!(
        "[t19]   ⇒ 逐位不同 {nd} / {} ｜速度比 {:.3}×",
        a.len(),
        us_e / us_s
    );
    assert_eq!(nd, 0, "Q4 静态网格与精确网格不逐位相同");
    assert!(us_s < us_e * 1.10, "Q4 静态网格慢过 10%（{us_s:.2} vs {us_e:.2}）");
}
