//! **attn 段（evprof seg1）线上真实形状的微基准**（2026-10-23）。
//!
//! 动机：`LINGXI_ATTN_SYNC=1` 的分段表
//! 显示 attn 的 6 段里，投影类占 ~45%、GDN 递推 ~25%、QSA 稀疏 attention ~16%。
//! 这份用例只复刻**线上形状**、不改核，回答"每个核离它自己的天花板还有多远"：
//!
//! - `q4`: 稠密 Q4G64 瘦长 GEMM（`gemm_q4_skinny_bf16_ptr`，m=6 生产档）——
//! GDN `qkvz(10240×2560)` / `z(6144×2560)` / `out(2560×6144)`，
//! QSA `q(6144)` / `k(512)` / `gate(6144)` / `v(512)`，以及"4 发 vs 1 发"的合并对照。
//! - `rec`: GDN 递推（`gdn_recurrent_bf16`，48 v-head × 128 线程，t=6）。
//! - `attn`: QSA 稀疏 gather attention（`attn_gather_bf16`，t=6 / 24 heads / kv2 / hd256 /
//! compress4 / topk512 / nh=4）。
//! - `conv`: GDN 因果卷积（`causal_conv1d_silu_hist_bf16`）。
//!
//! 门控：
//! `cargo test --release -p lingxi-kernels --features gpu --test t6_attn_shape_bw -- --nocapture`
//!
//! 对照天花板（本机实测，见 docs /）：
//! 稠密 Q4 m6 = 141~149 GB/s（`t3_gemv_bw` 同核）、dp4a 196.9、m=1 GEMV 275.6；
//! bf16 ALU 上限 177 GB/s。

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
    t0.elapsed().as_secs_f64() / iters as f64
}

/// 一碗稠密 Q4G64 权重（`[rows, kdim]`，base 平面 + scale 平面）。
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
    let total = scale_off as usize + rows * (k_pad as usize / 64) * 2;
    Q4 { packed: unsafe { k.stream.alloc::<u8>(total) }.unwrap(), k_pad, scale_off }
}

/// 线上真的会这样发：**同一块权重、若干段 row_off/n 各一发**（GDN=2 发，QSA=4 发）。
/// `segs` 里每一段写进**自己的紧凑输出缓冲**（生产就是这样，4 个 scratch 各一块）。
/// 返回总字节（只算 base 平面 payload，与小节其它表同口径）。
fn bench_segs(tag: &str, k: &Kernels, rows_total: usize, kdim: usize, m: usize, segs: &[(usize, usize)]) {
    let w = q4w(k, rows_total, kdim);
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); m * kdim]).unwrap();
    let mut ys: Vec<CudaSlice<bf16>> =
        segs.iter().map(|&(_, n)| k.stream.alloc_zeros::<bf16>(m * n).unwrap()).collect();
    let (wp, _g0) = w.packed.device_ptr(&k.stream);
    let (xp, _g1) = x.device_ptr(&k.stream);
    let mut yp: Vec<u64> = Vec::new();
    for y in ys.iter_mut() {
        yp.push(y.device_ptr_mut(&k.stream).0);
    }
    let run = || {
        for (i, &(off, n)) in segs.iter().enumerate() {
            k.gemm_q4_skinny_bf16_ptr(
                wp, xp, yp[i], n, kdim, w.k_pad, m, off, w.scale_off, 0, 0.0, 0.0,
            )
            .unwrap();
        }
    };
    let dt = time_launches(k, 50, run);
    k.stream.synchronize().unwrap();
    let payload: usize = segs.iter().map(|&(_, n)| n * kdim / 2).sum();
    let nm: usize = segs.iter().map(|&(_, n)| n).sum();
    eprintln!(
        "[attn-bw] {tag}: {} 发 合计行{nm} k{kdim} m{m}｜{:.1} µs｜{:.1} GB/s｜payload {:.2} MB",
        segs.len(),
        dt * 1e6,
        payload as f64 / dt / 1e9,
        payload as f64 / 1e6
    );
    // 每段单独一发（同样的缓冲，一次只发一段）——用来分离"小 n 的一发"本身的成本
    if segs.len() > 1 {
        for (i, &(off, n)) in segs.iter().enumerate() {
            let dt1 = time_launches(k, 50, || {
                k.gemm_q4_skinny_bf16_ptr(
                    wp, xp, yp[i], n, kdim, w.k_pad, m, off, w.scale_off, 0, 0.0, 0.0,
                )
                .unwrap();
            });
            let b = n * kdim / 2;
            eprintln!(
                "[attn-bw]   · 单发 n{n:<6} m{m}｜{:.1} µs｜{:.1} GB/s",
                dt1 * 1e6,
                b as f64 / dt1 / 1e9
            );
        }
    }
    let _ = &mut ys;
}

/// GDN 递推：48 v-head × 128 线程、t 步串行（生产核，smem 关）。
fn bench_gdn_rec(k: &Kernels, t: usize) {
    let (nvh, nkh, hk, hv) = (48usize, 16usize, 128usize, 128usize);
    let conv_dim = 2 * nkh * hk + nvh * hv;
    let value_dim = nvh * hv;
    let conv = k.stream.clone_htod(&vec![bf16::from_f32(0.01); t * conv_dim]).unwrap();
    let z = k.stream.clone_htod(&vec![bf16::from_f32(0.01); t * value_dim]).unwrap();
    let a = k.stream.clone_htod(&vec![bf16::from_f32(0.01); t * nvh]).unwrap();
    let b = k.stream.clone_htod(&vec![bf16::from_f32(0.01); t * nvh]).unwrap();
    let nw = k.stream.clone_htod(&vec![bf16::from_f32(1.0); hv]).unwrap();
    let alog = k.stream.clone_htod(&vec![0.1f32; nvh]).unwrap();
    let dtb = k.stream.clone_htod(&vec![0.0f32; nvh]).unwrap();
    let mut s = k.stream.alloc_zeros::<f32>(nvh * hk * hv).unwrap();
    let mut gated = k.stream.alloc_zeros::<bf16>(t * value_dim).unwrap();
    let dt = time_launches(k, 100, || {
        k.gdn_recurrent_bf16(
            &conv, &z, &a, &b, &nw, &alog, &dtb, &mut s, &mut gated, t, hk, hv, nvh, nkh, 1, 0, 1,
        )
        .unwrap();
    });
    k.stream.synchronize().unwrap();
    // 有效带宽：S 每 token 被读+写 3 遍（decay+update 读写、kv_mem 读、o 读）≈ 6 遍
    let s_mb = (nvh * hk * hv * 4) as f64 / 1e6;
    eprintln!(
        "[attn-bw] gdn_rec t{t}: 48 head×128 dv｜{:.1} µs（{:.1} µs/token）｜S={s_mb:.1} MB → 折合 {:.0} GB/s",
        dt * 1e6,
        dt * 1e6 / t as f64,
        6.0 * s_mb * 1e3 / (dt * 1e6)
    );
}

/// QSA 稀疏 gather attention：t=6 / heads24 / kv2 / hd256 / compress4 / topk512 / nh=4。
fn bench_attn_gather(k: &Kernels, nblocks: usize) {
    let (t, heads, kvh, hd, compress, topk, nh) = (6usize, 24usize, 2usize, 256usize, 4usize, 512usize, 4usize);
    let ctok = compress;
    let vis = nblocks * compress;
    let cap = vis + 64;
    let kn = k.stream.clone_htod(&vec![bf16::from_f32(0.01); cap * kvh * hd]).unwrap();
    let v = k.stream.clone_htod(&vec![bf16::from_f32(0.01); cap * kvh * hd]).unwrap();
    let qn = k.stream.clone_htod(&vec![bf16::from_f32(0.01); t * heads * hd]).unwrap();
    let blk: Vec<i32> = (0..(nblocks + 16) as i32).collect();
    let blk_tab = k.stream.clone_htod(&blk).unwrap();
    let selcnt = nblocks.min(topk);
    let mut sel = vec![0i32; t * topk];
    for r in 0..t {
        for b in 0..selcnt {
            sel[r * topk + b] = b as i32;
        }
    }
    let sel = k.stream.clone_htod(&sel).unwrap();
    let scnt = k.stream.clone_htod(&vec![selcnt as i32; t]).unwrap();
    let mut out = k.stream.alloc_zeros::<bf16>(t * heads * hd).unwrap();
    let scale = 1.0f32 / (hd as f32).sqrt();
    let dt = time_launches(k, 100, || {
        k.attn_gather_bf16(
            &qn.slice(..), &kn.slice(..), &v.slice(..), &blk_tab.slice(..),
            &sel.slice(..), &scnt.slice(..), &mut out.slice_mut(..),
            t, heads, kvh, hd, scale, vis, nblocks, 2, topk, nh,
        )
        .unwrap();
    });
    k.stream.synchronize().unwrap();
    // K/V 读字节：每行 × 每 kv-head × (heads/kvh/nh) 组 × (sel*ctok+ctok) key × hd × 2B × 2(K,V)
    let keys = selcnt * ctok + ctok;
    let bytes = t * kvh * (heads / kvh / nh) * keys * hd * 2 * 2;
    eprintln!(
        "[attn-bw] attn_gather len{}（nblocks{nblocks} sel{selcnt} keys{keys}）｜{:.1} µs｜{:.1} GB/s｜K/V 读 {:.2} MB",
        vis,
        dt * 1e6,
        bytes as f64 / dt / 1e9,
        bytes as f64 / 1e6
    );
}

/// GDN 因果卷积（融合历史推进版）：conv_dim=10240、t=6、K=4。
fn bench_conv(k: &Kernels, t: usize) {
    let conv_dim = 10240usize;
    let kh = 3usize;
    let hist = k.stream.alloc_zeros::<bf16>(kh * conv_dim).unwrap();
    let hist_next = k.stream.alloc_zeros::<bf16>(kh * conv_dim).unwrap();
    let qkv = k.stream.clone_htod(&vec![bf16::from_f32(0.01); t * conv_dim]).unwrap();
    let w = k.stream.clone_htod(&vec![bf16::from_f32(0.1); 4 * conv_dim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(t * conv_dim).unwrap();
    let (hp, _a) = hist.device_ptr(&k.stream);
    let (qp, _b) = qkv.device_ptr(&k.stream);
    let (wp, _c) = w.device_ptr(&k.stream);
    let (yp, _d) = y.device_ptr_mut(&k.stream);
    let (np, _e) = hist_next.device_ptr(&k.stream);
    let dt = time_launches(k, 200, || {
        k.causal_conv1d_silu_hist_bf16(hp, qp, wp, yp, np, kh, t, conv_dim, 4, 1).unwrap();
    });
    k.stream.synchronize().unwrap();
    let bytes = t * conv_dim * 2 * 3;
    eprintln!(
        "[attn-bw] gdn_conv t{t} conv_dim{conv_dim}｜{:.1} µs｜{:.1} GB/s（含 3×hist 拷贝）",
        dt * 1e6,
        bytes as f64 / dt / 1e9
    );
}

/// QSA indexer 选块的两个小核 + sel_build（线上 len≈118 ⇒ nblocks=29、topk=512）。
fn bench_idxsel(k: &Kernels, nblocks: usize) {
    let (t, heads, hd, topk, compress) = (6usize, 4usize, 128usize, 512usize, 4usize);
    let iq = k.stream.clone_htod(&vec![bf16::from_f32(0.01); t * heads * hd]).unwrap();
    let pooled = k.stream.clone_htod(&vec![bf16::from_f32(0.01); (nblocks + 8) * hd]).unwrap();
    let mut scores = k.stream.alloc_zeros::<f32>(t * (nblocks + 8)).unwrap();
    let mut keep = k.stream.alloc_zeros::<u8>(t * (nblocks + 8)).unwrap();
    let mut sel = k.stream.alloc_zeros::<i32>(t * topk).unwrap();
    let mut scnt = k.stream.alloc_zeros::<i32>(t).unwrap();
    let sq = (hd as f32).sqrt();
    let dt_s = time_launches(k, 100, || {
        k.qsa_indexer_scores_bf16(&iq, &pooled, &mut scores, t, nblocks, heads, hd, sq).unwrap();
    });
    let dt_r = time_launches(k, 100, || {
        k.qsa_indexer_rank_f32(&scores, &mut keep, t, nblocks, topk).unwrap();
    });
    let dt_b = time_launches(k, 100, || {
        k.qsa_sel_build(&mut keep, &mut sel, &mut scnt, t, nblocks, topk, false).unwrap();
    });
    let dt_all = time_launches(k, 100, || {
        k.qsa_indexer_scores_bf16(&iq, &pooled, &mut scores, t, nblocks, heads, hd, sq).unwrap();
        k.qsa_indexer_rank_f32(&scores, &mut keep, t, nblocks, topk).unwrap();
        k.qsa_sel_build(&mut keep, &mut sel, &mut scnt, t, nblocks, topk, false).unwrap();
    });
    // 每 4 个 token 才做一次的池化链（nb=1）
    let rk = k.stream.clone_htod(&vec![bf16::from_f32(0.01); (nblocks * compress + 8) * hd]).unwrap();
    let mut pool = k.stream.alloc_zeros::<bf16>((nblocks + 8) * hd).unwrap();
    let nw = k.stream.clone_htod(&vec![bf16::from_f32(1.0); hd]).unwrap();
    let mut p2 = k.stream.alloc_zeros::<bf16>((nblocks + 8) * hd).unwrap();
    let dt_pool = time_launches(k, 100, || {
        k.qsa_pool_blocks_bf16(&rk, &mut pool, nblocks - 1, 1, compress, hd).unwrap();
        k.grouped_rmsnorm_bf16(&pool, &nw, &mut p2, 1, hd, hd, 1e-6).unwrap();
    });
    eprintln!(
        "[attn-bw] qsa_idxsel nblocks{nblocks}: scores {:.1} + rank {:.1} + sel_build {:.1} = 三发 {:.1} µs｜池化链(每 4 token 一次) {:.1} µs",
        dt_s * 1e6, dt_r * 1e6, dt_b * 1e6, dt_all * 1e6, dt_pool * 1e6
    );
}

#[test]
fn t6_attn_shape_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    eprintln!("[attn-bw] 天花板（本机实测）：稠密 Q4 m6 141~149 GB/s / m1 GEMV 275.6 / dp4a m6 196.9");

    // ── GDN：qkvz(10240) + z(6144) 两发，与 out_proj(2560×6144) ──
    for &m in &[1usize, 6] {
        bench_segs("GDN qkvz+z", &k, 16384, HIDDEN, m, &[(0, 10240), (10240, 6144)]);
        bench_segs("GDN out(2560x6144)", &k, 2560, 6144, m, &[(0, 2560)]);
    }
    // ── QSA：q(6144)+k(512)+gate(6144)+v(512) 四发 vs 合并一发 n=13312 ──
    for &m in &[1usize, 6] {
        bench_segs(
            "QSA 4 发(生产)",
            &k,
            13312,
            HIDDEN,
            m,
            &[(0, 6144), (6144, 512), (6656, 6144), (12800, 512)],
        );
        bench_segs("QSA 合并 1 发(反事实)", &k, 13312, HIDDEN, m, &[(0, 13312)]);
        bench_segs("QSA out(2560x6144)", &k, 2560, 6144, m, &[(0, 2560)]);
    }

    // ── GDN 递推 ──
    for &t in &[1usize, 6] {
        bench_gdn_rec(&k, t);
    }

    // ── QSA 稀疏 attention（短上下文 len118 / 中上下文 len520） ──
    for &nb in &[29usize, 130] {
        bench_attn_gather(&k, nb);
    }

    // ── QSA indexer 选块（3 发 + 每 4 token 一次的池化链） ──
    for &nb in &[29usize, 130] {
        bench_idxsel(&k, nb);
    }

    // ── GDN 卷积 ──
    bench_conv(&k, 6);
}
