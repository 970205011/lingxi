//! **专家段 GEMV 天花板判定微基准**（2026-10-24）。
//!
//! 问题：线上 `MoE:专家段` 按字节折算只有 ~275 GB/s（m=1）/ 264（m≈1.5，TC 核），
//! 而 V100-32GB 的 HBM2 是 900 GB/s ⇒ 只有 30%。这个 275 是**我们自己核的上限**，
//! 到底卡在"访问模式/指令"，还是"并行度不够"（40 个唯一专家 = gridDim.y 只有 40）？
//!
//! 这个用例只做**判定**，不改生产路径：
//! 1. 现状核（`gemv_q4g64_slotsm_rpw_bf16_view`，E=40，m=1 / m≈1.5）；
//! 2. **同一核、E 人为放大**到 80/160/320（把同一份权重拷到更多槽，只为测 block 数的影响）；
//! 3. **k 方向 split-k 2/4/8 段**（`gemv_par_exp.cu` 的新实验核，允许不逐位相同）；
//! 4. 两个纯读基线：与 `slotsm` 同形状的只读核（4 B/lane、16 B/lane）、经典 grid-stride uint4。
//!
//! 门控：`cargo test --release -p lingxi-kernels --features gpu --test t11_moe_gemv_par_bw -- --nocapture`

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

const HIDDEN: usize = 2560;
const INTER: usize = 640;
/// gu 段：n = 2*inter = 1280 行、k = hidden = 2560（每行 Q4 = 1280 B）。
const GU_N: usize = 2 * INTER;
const GU_K: usize = HIDDEN;

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

/// `matrix_layout` 的同公式（Q4G64，PLANE_ALIGNMENT=256）：blob / gu_scale / dn_seg / dn_scale。
fn geo_q4() -> (u64, u64, u64, u64) {
    let layout = |rows: u64, cols: u64| -> (u64, u64, u64) {
        let k_pad = align_up(cols, 128);
        let groups = k_pad / 64;
        let row_bytes = groups * 32;
        let base = rows * row_bytes;
        let scale_off = align_up(base, 256);
        let payload = scale_off + rows * groups * 2;
        (base, scale_off, payload)
    };
    let (_, gu_scale, gu_payload) = layout(GU_N as u64, GU_K as u64);
    let (_, dn_scale, dn_payload) = layout(HIDDEN as u64, INTER as u64);
    (gu_payload + dn_payload, gu_scale, gu_payload, dn_scale)
}

/// Q3G128 的同公式（每 128 值 48 B）：blob / gu_scale / dn_seg / dn_scale。
fn geo_q3() -> (u64, u64, u64, u64) {
    let layout = |rows: u64, cols: u64| -> (u64, u64) {
        let k_pad = align_up(cols, 128);
        let groups = k_pad / 128;
        let base = rows * groups * 48;
        let scale_off = align_up(base, 256);
        (scale_off, scale_off + rows * groups * 2)
    };
    let (gu_scale, gu_payload) = layout(GU_N as u64, GU_K as u64);
    let (dn_scale, dn_payload) = layout(HIDDEN as u64, INTER as u64);
    (gu_payload + dn_payload, gu_scale, gu_payload, dn_scale)
}

fn group_off(e: usize, pat: u8) -> Vec<i32> {
    let mut v = Vec::with_capacity(e + 1);
    v.push(0i32);
    for g in 0..e {
        let m = match pat {
            1 => 1,
            2 => 2,
            _ => {
                if g % 2 == 0 {
                    1
                } else {
                    2
                }
            }
        };
        v.push(v[g] + m);
    }
    v
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

struct Buf {
    slots: CudaSlice<u8>,
    ids: Vec<i32>,
    goff: Vec<i32>,
    s: usize,
}

fn buf(k: &Kernels, e: usize, blob: u64, pat: u8, same_slot: bool) -> Buf {
    let slots = unsafe { k.stream.alloc::<u8>(e * blob as usize) }.unwrap();
    let ids: Vec<i32> = if same_slot { vec![0; e] } else { (0..e as i32).collect() };
    let goff = group_off(e, pat);
    let s = *goff.last().unwrap() as usize;
    Buf { slots, ids, goff, s }
}

fn report(tag: &str, bytes: f64, dt: f64, extra: &str) {
    eprintln!(
        "[par-bw] {tag:<46}｜{:8.3} ms｜{:7.1} GB/s｜{:7.2} MB{extra}",
        dt * 1e3,
        bytes / dt / 1e9,
        bytes / 1e6
    );
}

/// 现状核（生产默认路径）gu 段：E 组、组均 m 由 pat 决定。
fn bench_base(k: &Kernels, tag: &str, e: usize, pat: u8, rpw: usize, same_slot: bool) {
    let (blob, gu_scale, _, _) = geo_q4();
    let b = buf(k, e, blob, pat, same_slot);
    let ids = k.stream.clone_htod(&b.ids).unwrap();
    let goff = k.stream.clone_htod(&b.goff).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); b.s * GU_K]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(b.s * GU_N).unwrap();
    let dt = time_it(k, 60, || {
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &b.slots, &ids_v, &x, &goff_v, &mut y, e, GU_N, GU_K, GU_K as i32, GU_K,
            blob, 0, gu_scale, rpw, 0,
        )
        .unwrap();
    });
    k.stream.synchronize().unwrap();
    let bytes = e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    report(
        tag,
        bytes,
        dt,
        &format!("（m={:.1} rpw{rpw} 块={}）", b.s as f64 / e as f64, (GU_N + 7) / 8 * e),
    );
}

/// split-k 实验核（S 段）。
fn bench_splitk(k: &Kernels, tag: &str, e: usize, pat: u8, splits: usize) {
    let (blob, gu_scale, _, _) = geo_q4();
    let b = buf(k, e, blob, pat, false);
    let ids = k.stream.clone_htod(&b.ids).unwrap();
    let goff = k.stream.clone_htod(&b.goff).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); b.s * GU_K]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(b.s * GU_N).unwrap();
    let mut part = k.stream.alloc_zeros::<f32>(b.s * GU_N * splits).unwrap();
    let dt = time_it(k, 60, || {
        k.gemv_q4g64_slotsm_splitk_bf16(
            &b.slots, &ids_v, &x, &goff_v, &mut part, &mut y, e, GU_N, GU_K, GU_K as i32, GU_K,
            blob, 0, gu_scale, splits, 0,
        )
        .unwrap();
    });
    k.stream.synchronize().unwrap();
    // 权重字节 + 部分和写/读（fp32）
    let wbytes = e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    let pbytes = 2.0 * b.s as f64 * GU_N as f64 * splits as f64 * 4.0;
    report(
        tag,
        wbytes,
        dt,
        &format!("（权重 {:.1} MB + part 读写 {:.1} MB）", wbytes / 1e6, pbytes / 1e6),
    );
}

/// 现状核的变体：`xf` = 输入预转 fp32（省 bf16→f32 的 CVT）、`dp4a` = int8 输入 + dp4a、
/// `tc` = 张量核（`LINGXI_MOE_TC=1` 的生产档，仅 1<rows≤16）。
fn bench_base_variant(k: &Kernels, tag: &str, e: usize, pat: u8, xf: bool, tc: bool) {
    let (blob, gu_scale, _, _) = geo_q4();
    let b = buf(k, e, blob, pat, false);
    let ids = k.stream.clone_htod(&b.ids).unwrap();
    let goff = k.stream.clone_htod(&b.goff).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); b.s * GU_K]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(b.s * GU_N).unwrap();
    let xf_b = if xf {
        Some(k.stream.clone_htod(&vec![0.01f32; b.s * GU_K]).unwrap())
    } else {
        None
    };
    let xf_p = match &xf_b {
        Some(v) => {
            use cudarc::driver::DevicePtr;
            v.device_ptr(&k.stream).0
        }
        None => 0,
    };
    let dt = time_it(k, 60, || {
        if tc {
            // ⚠️ `s` 是**唯一组数**（生产传 `cnt`），不是对数：组内行由 group_off 给出。
            k.gemv_q4g64_slots_tc_bf16(
                &b.slots, &ids_v, &goff_v, &x, &mut y, e, GU_N, GU_K, GU_K, GU_K as i32,
                blob, 0, gu_scale, if b.s / e > 8 { 2 } else { 1 },
            )
            .unwrap();
        } else {
            k.gemv_q4g64_slotsm_rpw_bf16_view(
                &b.slots, &ids_v, &x, &goff_v, &mut y, e, GU_N, GU_K, GU_K as i32, GU_K,
                blob, 0, gu_scale, 1, xf_p,
            )
            .unwrap();
        }
    });
    let bytes = e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    report(tag, bytes, dt, &format!("（m={:.1}）", b.s as f64 / e as f64));
}

/// dp4a 档（int8 输入）：`LINGXI_MOE_DP4A=1` 的生产核。
fn bench_dp4a(k: &Kernels, tag: &str, e: usize, pat: u8) {
    let (blob, gu_scale, _, _) = geo_q4();
    let b = buf(k, e, blob, pat, false);
    let ids = k.stream.clone_htod(&b.ids).unwrap();
    let goff = k.stream.clone_htod(&b.goff).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); b.s * GU_K]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(b.s * GU_N).unwrap();
    let xq = k.stream.clone_htod(&vec![1u8; b.s * GU_K]).unwrap();
    let sx = k.stream.clone_htod(&vec![0.001f32; b.s]).unwrap();
    let dt = time_it(k, 60, || {
        k.gemv_q4g64_slotsm_rpw_dp4a_view(
            &b.slots, &ids_v, &x, &goff_v, &mut y, e, GU_N, GU_K, GU_K as i32, GU_K,
            blob, 0, gu_scale, 1, &xq, &sx,
        )
        .unwrap();
    });
    let bytes = e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    report(tag, bytes, dt, &format!("（m={:.1}）", b.s as f64 / e as f64));
}

/// 槽距（blob）扫描：生产线上的混档 arena 槽距是 2,611,200 B（**非 2 的幂**），
/// 而历史微基准（`t3_gemv_bw`）用的是 8 MiB（**2 的幂**）——若两者差很多，
/// 说明老尺子量的是"分区 camping"而不是核本身。
fn bench_stride(k: &Kernels, tag: &str, e: usize, blob: u64, dp4a: bool) {
    let (_, gu_scale, _, _) = geo_q4();
    let slots = unsafe { k.stream.alloc::<u8>(e * blob as usize) }.unwrap();
    let ids_h: Vec<i32> = (0..e as i32).collect();
    let goff_h = group_off(e, 1);
    let ids = k.stream.clone_htod(&ids_h).unwrap();
    let goff = k.stream.clone_htod(&goff_h).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); e * GU_K]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(e * GU_N).unwrap();
    let xq = k.stream.clone_htod(&vec![1u8; e * GU_K]).unwrap();
    let sx = k.stream.clone_htod(&vec![0.001f32; e]).unwrap();
    let dt = time_it(k, 60, || {
        if dp4a {
            k.gemv_q4g64_slotsm_rpw_dp4a_view(
                &slots, &ids_v, &x, &goff_v, &mut y, e, GU_N, GU_K, GU_K as i32, GU_K,
                blob, 0, gu_scale, 1, &xq, &sx,
            )
            .unwrap();
        } else {
            k.gemv_q4g64_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x, &goff_v, &mut y, e, GU_N, GU_K, GU_K as i32, GU_K,
                blob, 0, gu_scale, 1, 0,
            )
            .unwrap();
        }
    });
    let bytes = e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    report(tag, bytes, dt, &format!("（槽距 {:.2} MB）", blob as f64 / 1e6));
}

/// **复刻 `t3_gemv_bw` 的尺子**（E=49、blob=8 MiB、`seg_off=0`、`scale_off=0`），
/// 用来解释"同一核、同一名义形状，t3 报 ~100 GB/s、本用例报 ~293 GB/s"的 3× 差。
fn bench_t3_replica(k: &Kernels, tag: &str, e: usize, blob: u64, scale_off: u64) {
    let slots = unsafe { k.stream.alloc::<u8>(e * blob as usize) }.unwrap();
    let ids_h: Vec<i32> = (0..e as i32).collect();
    let goff_h = group_off(e, 1);
    let ids = k.stream.clone_htod(&ids_h).unwrap();
    let goff = k.stream.clone_htod(&goff_h).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); e * GU_K]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(e * GU_N).unwrap();
    let dt = time_it(k, 60, || {
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &slots, &ids_v, &x, &goff_v, &mut y, e, GU_N, GU_K, GU_K as i32, GU_K,
            blob, 0, scale_off, 1, 0,
        )
        .unwrap();
    });
    let bytes = e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    report(tag, bytes, dt, &format!("（blob {:.1} MB / scale_off {scale_off}）", blob as f64 / 1e6));
}

/// **dn 段**（n=hidden=2560、k=inter=640，每行只有 320 B）：现核 vs dp4a vs 纯读。
/// dn 的 k 小 ⇒ 每 warp 只 ~1.25 个 256 列 chunk，本来就更难喂满。
fn bench_dn(k: &Kernels, tag: &str, e: usize, mode: u8) {
    let (blob, _, dn_seg, dn_scale) = geo_q4();
    let slots = unsafe { k.stream.alloc::<u8>(e * blob as usize) }.unwrap();
    let ids_h: Vec<i32> = (0..e as i32).collect();
    let goff_h = group_off(e, 3);
    let ids = k.stream.clone_htod(&ids_h).unwrap();
    let goff = k.stream.clone_htod(&goff_h).unwrap();
    let ids_v = ids.slice(0..e);
    let goff_v = goff.slice(0..=e);
    let s = *goff_h.last().unwrap() as usize;
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); s * INTER]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(s * HIDDEN).unwrap();
    let xq = k.stream.clone_htod(&vec![1u8; s * INTER]).unwrap();
    let sx = k.stream.clone_htod(&vec![0.001f32; s]).unwrap();
    // `dn_scale` 是**相对 dn 段起点**的偏移（与 `Q4Consts` 同口径），直接传。
    let sub = dn_scale;
    let dt = time_it(k, 60, || {
        match mode {
            0 => k.gemv_q4g64_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x, &goff_v, &mut y, e, HIDDEN, INTER, INTER as i32, INTER,
                blob, dn_seg, sub, 1, 0,
            ),
            1 => k.gemv_q4g64_slotsm_rpw_dp4a_view(
                &slots, &ids_v, &x, &goff_v, &mut y, e, HIDDEN, INTER, INTER as i32, INTER,
                blob, dn_seg, sub, 1, &xq, &sx,
            ),
            _ => {
                let mut out = k.stream.alloc_zeros::<f32>(e * HIDDEN).unwrap();
                k.read_probe_slots(&slots, &mut out, e, HIDDEN, INTER / 2, blob, 0)
            }
        }
        .unwrap();
    });
    let bytes = e as f64 * HIDDEN as f64 * (INTER as f64 / 2.0);
    report(tag, bytes, dt, &format!("（m={:.1}）", s as f64 / e as f64));
}

/// **线上 decode 形状**（rows=1、s=10 对、7 组 Q4 + 3 组 Q3、两发分开、`a_stride=0`）。
/// 这是判断"微基准的 1.6× 能不能兑现到整机"的关键尺子：evprof 的专家段层均 0.286 ms，
/// 按字节折算只有 ~60-90 GB/s —— 若这个形状本身就喂不满核，换内层指令收益就有限。
fn bench_prod(k: &Kernels, tag: &str, dp4a: bool) {
    let (b4, s4off, d4seg, d4sc) = geo_q4();
    let (b3, s3off, d3seg, d3sc) = geo_q3();
    let e4 = 7usize;
    let e3 = 3usize;
    let slots4 = unsafe { k.stream.alloc::<u8>(e4 * b4 as usize) }.unwrap();
    let slots3 = unsafe { k.stream.alloc::<u8>(e3 * b3 as usize) }.unwrap();
    let ids4 = k.stream.clone_htod(&(0..e4 as i32).collect::<Vec<_>>()).unwrap();
    let ids3 = k.stream.clone_htod(&(0..e3 as i32).collect::<Vec<_>>()).unwrap();
    let go4 = k.stream.clone_htod(&(0..=e4 as i32).collect::<Vec<_>>()).unwrap();
    let go3 = k.stream.clone_htod(&(0..=e3 as i32).collect::<Vec<_>>()).unwrap();
    let ids4v = ids4.slice(0..e4);
    let ids3v = ids3.slice(0..e3);
    let go4v = go4.slice(0..=e4);
    let go3v = go3.slice(0..=e3);
    // rows=1 ⇒ 所有对共用第 0 行
    let xg = k.stream.clone_htod(&vec![bf16::from_f32(0.01); GU_K]).unwrap();
    let xd = k.stream.clone_htod(&vec![bf16::from_f32(0.01); INTER]).unwrap();
    let xqg = k.stream.clone_htod(&vec![1u8; GU_K]).unwrap();
    let xqd = k.stream.clone_htod(&vec![1u8; INTER]).unwrap();
    let sx = k.stream.clone_htod(&vec![0.001f32; 1]).unwrap();
    let mut yg = k.stream.alloc_zeros::<bf16>(e4 * GU_N).unwrap();
    let mut yg3 = k.stream.alloc_zeros::<bf16>(e3 * GU_N).unwrap();
    let mut yd = k.stream.alloc_zeros::<bf16>(e4 * HIDDEN).unwrap();
    let mut yd3 = k.stream.alloc_zeros::<bf16>(e3 * HIDDEN).unwrap();
    let xgv = xg.slice(0..xg.len());
    let xdv = xd.slice(0..xd.len());
    let dt = time_it(k, 100, || {
        // gu：Q4（7 组）+ Q3（3 组）两次 launch，背靠背
        if dp4a {
            k.gemv_q4g64_slotsm_rpw_dp4a_view(
                &slots4, &ids4v, &xg, &go4v, &mut yg, e4, GU_N, GU_K, GU_K as i32, 0,
                b4, 0, s4off, 1, &xqg, &sx,
            )
            .unwrap();
        } else {
            let mut yv = yg.slice_mut(0..yg.len());
            k.gemv_q4g64_slots_rows_bf16_view(
                &slots4, &ids4v, &xgv, &mut yv, e4, GU_N, GU_K, GU_K as i32, 0,
                b4, 0, s4off, 1,
            )
            .unwrap();
        }
        let mut y3v = yg3.slice_mut(0..yg3.len());
        k.gemv_q3g128_slots_rows_bf16_view(
            &slots3, &ids3v, &xgv, &mut y3v, e3, GU_N, GU_K, GU_K as i32, 0,
            b3, 0, s3off, 1,
        )
        .unwrap();
        // dn
        if dp4a {
            k.gemv_q4g64_slotsm_rpw_dp4a_view(
                &slots4, &ids4v, &xd, &go4v, &mut yd, e4, HIDDEN, INTER, INTER as i32, 0,
                b4, d4seg, d4sc, 1, &xqd, &sx,
            )
            .unwrap();
        } else {
            let mut yv = yd.slice_mut(0..yd.len());
            k.gemv_q4g64_slots_rows_bf16_view(
                &slots4, &ids4v, &xdv, &mut yv, e4, HIDDEN, INTER, INTER as i32, 0,
                b4, d4seg, d4sc, 1,
            )
            .unwrap();
        }
        let mut y3v = yd3.slice_mut(0..yd3.len());
        k.gemv_q3g128_slots_rows_bf16_view(
            &slots3, &ids3v, &xdv, &mut y3v, e3, HIDDEN, INTER, INTER as i32, 0,
            b3, d3seg, d3sc, 1,
        )
        .unwrap();
    });
    let bytes = e4 as f64 * (GU_N * GU_K / 2 + HIDDEN * INTER / 2) as f64
        + e3 as f64 * (GU_N * 960 + HIDDEN * 240) as f64;
    report(tag, bytes, dt, &format!("（{} 组 Q4 + {} 组 Q3，rows=1）", e4, e3));
}

/// 线上形状的**逐发拆账**：四发各自多少 µs（判"是否被 launch/尾巴吃掉了"）。
fn bench_prod_parts(k: &Kernels) {
    let (b4, s4off, d4seg, d4sc) = geo_q4();
    let (b3, s3off, d3seg, d3sc) = geo_q3();
    let (e4, e3) = (7usize, 3usize);
    let slots4 = unsafe { k.stream.alloc::<u8>(e4 * b4 as usize) }.unwrap();
    let slots3 = unsafe { k.stream.alloc::<u8>(e3 * b3 as usize) }.unwrap();
    let ids4 = k.stream.clone_htod(&(0..e4 as i32).collect::<Vec<_>>()).unwrap();
    let ids3 = k.stream.clone_htod(&(0..e3 as i32).collect::<Vec<_>>()).unwrap();
    let go4 = k.stream.clone_htod(&(0..=e4 as i32).collect::<Vec<_>>()).unwrap();
    let go3 = k.stream.clone_htod(&(0..=e3 as i32).collect::<Vec<_>>()).unwrap();
    let ids4v = ids4.slice(0..e4);
    let ids3v = ids3.slice(0..e3);
    let go4v = go4.slice(0..=e4);
    let go3v = go3.slice(0..=e3);
    let xg = k.stream.clone_htod(&vec![bf16::from_f32(0.01); GU_K]).unwrap();
    let xd = k.stream.clone_htod(&vec![bf16::from_f32(0.01); INTER]).unwrap();
    let mut yg = k.stream.alloc_zeros::<bf16>(e4 * GU_N).unwrap();
    let mut yg3 = k.stream.alloc_zeros::<bf16>(e3 * GU_N).unwrap();
    let mut yd = k.stream.alloc_zeros::<bf16>(e4 * HIDDEN).unwrap();
    let mut yd3 = k.stream.alloc_zeros::<bf16>(e3 * HIDDEN).unwrap();
    let xgv = xg.slice(0..xg.len());
    let xdv = xd.slice(0..xd.len());
    for (tag, n, e, mb, f) in [
        ("gu Q4 7组", GU_N, e4, e4 as f64 * (GU_N * GU_K / 2) as f64, 0u8),
        ("gu Q3 3组", GU_N, e3, e3 as f64 * (GU_N * 960) as f64, 1),
        ("dn Q4 7组", HIDDEN, e4, e4 as f64 * (HIDDEN * INTER / 2) as f64, 2),
        ("dn Q3 3组", HIDDEN, e3, e3 as f64 * (HIDDEN * 240) as f64, 3),
    ] {
        let dt = time_it(k, 100, || match f {
            0 => {
                let mut yv = yg.slice_mut(0..yg.len());
                k.gemv_q4g64_slots_rows_bf16_view(
                    &slots4, &ids4v, &xgv, &mut yv, e4, GU_N, GU_K, GU_K as i32, 0,
                    b4, 0, s4off, 1,
                )
                .unwrap();
            }
            1 => {
                let mut yv = yg3.slice_mut(0..yg3.len());
                k.gemv_q3g128_slots_rows_bf16_view(
                    &slots3, &ids3v, &xgv, &mut yv, e3, GU_N, GU_K, GU_K as i32, 0,
                    b3, 0, s3off, 1,
                )
                .unwrap();
            }
            2 => {
                let mut yv = yd.slice_mut(0..yd.len());
                k.gemv_q4g64_slots_rows_bf16_view(
                    &slots4, &ids4v, &xdv, &mut yv, e4, HIDDEN, INTER, INTER as i32, 0,
                    b4, d4seg, d4sc, 1,
                )
                .unwrap();
            }
            _ => {
                let mut yv = yd3.slice_mut(0..yd3.len());
                k.gemv_q3g128_slots_rows_bf16_view(
                    &slots3, &ids3v, &xdv, &mut yv, e3, HIDDEN, INTER, INTER as i32, 0,
                    b3, d3seg, d3sc, 1,
                )
                .unwrap();
            }
        });
        let _ = n;
        report(&format!("拆分 {tag}"), mb, dt, "");
    }
}

/// 与 slotsm 同形状的纯读探针。
fn bench_read_slots(k: &Kernels, tag: &str, e: usize, mode: i32) {
    let (blob, _, _, _) = geo_q4();
    let slots = unsafe { k.stream.alloc::<u8>(e * blob as usize) }.unwrap();
    let mut out = k.stream.alloc_zeros::<f32>(e * GU_N).unwrap();
    let dt = time_it(k, 60, || {
        k.read_probe_slots(&slots, &mut out, e, GU_N, GU_K / 2, blob, mode).unwrap();
    });
    k.stream.synchronize().unwrap();
    let bytes = e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    report(tag, bytes, dt, "");
}

/// 经典 grid-stride uint4 纯读（同一批字节）。
fn bench_read_flat(k: &Kernels, tag: &str, bytes: usize) {
    let slots = unsafe { k.stream.alloc::<u8>(bytes) }.unwrap();
    let mut out = k.stream.alloc_zeros::<f32>(4096).unwrap();
    let dt = time_it(k, 60, || {
        k.read_probe_flat(&slots, &mut out).unwrap();
    });
    k.stream.synchronize().unwrap();
    report(tag, bytes as f64, dt, "");
}

#[test]
fn t11_moe_gemv_par_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let (blob, gu_scale, _, _) = geo_q4();
    eprintln!(
        "[par-bw] 几何：Q4 blob {} B｜gu n{} k{} 每行 {} B｜单专家 gu 字节 {:.2} MB",
        blob,
        GU_N,
        GU_K,
        GU_K / 2,
        GU_N as f64 * (GU_K as f64 / 2.0) / 1e6
    );

    eprintln!("── ① 现状核（生产默认 rpw=1） ──");
    for &pat in &[1u8, 3] {
        for &rpw in &[1usize, 2] {
            bench_base(&k, &format!("baseline E40 m={}", if pat == 1 { "1.0" } else { "1.5" }), 40, pat, rpw, false);
        }
    }
    bench_base(&k, "baseline E40 m=1.0 同槽(L2)", 40, 1, 1, true);

    eprintln!("── ② 同一核、E 放大（block 数 ×2/×4/×8） ──");
    for &e in &[40usize, 80, 160, 320] {
        bench_base(&k, &format!("E扩 m=1.0 E{e}"), e, 1, 1, false);
    }
    for &e in &[40usize, 80, 160] {
        bench_base(&k, &format!("E扩 m=1.5 E{e}"), e, 3, 1, false);
    }

    eprintln!("── ③ split-k（k=2560 切 S 段） ──");
    for &splits in &[2usize, 4, 8] {
        bench_splitk(&k, &format!("split-k S{splits} E40 m=1.0"), 40, 1, splits);
    }
    for &splits in &[2usize, 4] {
        bench_splitk(&k, &format!("split-k S{splits} E40 m=1.5"), 40, 3, splits);
    }

    eprintln!("── ⑦ 线上 decode 形状（rows=1、7 组 Q4 + 3 组 Q3、两发分开） ──");
    bench_prod(&k, "线上形状 现核（slots_rows）", false);
    bench_prod(&k, "线上形状 Q4 改 dp4a", true);
    bench_prod_parts(&k);

    eprintln!("── ③c dn 段（n=2560 k=640，每行 320 B） ──");
    bench_dn(&k, "dn 现核 E40 m=1.5", 40, 0);
    bench_dn(&k, "dn dp4a E40 m=1.5", 40, 1);
    bench_dn(&k, "dn 纯读 E40 m=1.5", 40, 2);

    eprintln!("── ③b 别的计算档（同形状：少指令/不同指令能不能突破） ──");
    for &pat in &[1u8, 3] {
        let m = if pat == 1 { "1.0" } else { "1.5" };
        bench_base_variant(&k, &format!("现核 + xf(fp32 输入) m={m}"), 40, pat, true, false);
        bench_dp4a(&k, &format!("dp4a(int8 输入) m={m}"), 40, pat);
        bench_base_variant(&k, &format!("张量核 TC m={m}"), 40, pat, false, true);
    }
    // 投机验证轮的线上形状（rows=7 ⇒ 每对 m≈2、E≈35 组、a_stride=hidden）：
    // 生产 gu-Q4 走 TC（`LINGXI_MOE_TC=1` 且 1<rows≤16），所以这里 TC 才是基准。
    for &e in &[35usize] {
        bench_base_variant(&k, &format!("E{e} m=2.0 张量核 TC（生产 gu-Q4 档）"), e, 2, false, true);
        bench_dp4a(&k, &format!("E{e} m=2.0 dp4a"), e, 2);
        bench_base(&k, &format!("E{e} m=2.0 现核 SIMT"), e, 2, 1, false);
    }

    eprintln!("── ⑤ 槽距（blob）扫描：旧尺子用 8 MiB（2 的幂），生产用 2.61 MB ──");
    for &blob in &[
        blob,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        8 * 1024 * 1024 + 256,
        16 * 1024 * 1024,
    ] {
        bench_stride(&k, &format!("现核 E40 m=1 blob={blob}"), 40, blob, false);
        bench_stride(&k, &format!("dp4a E40 m=1 blob={blob}"), 40, blob, true);
    }

    eprintln!("── ⑤b 组数扫描（线上每层只有 7~12 个真组）：dp4a 的 1.6× 还在不在 ──");
    for &e in &[7usize, 10, 20, 40] {
        bench_stride(&k, &format!("现核 E{e} m=1"), e, blob, false);
        bench_stride(&k, &format!("dp4a E{e} m=1"), e, blob, true);
    }

    eprintln!("── ⑥ 复刻 t3 尺子（E49 / blob 8 MiB / scale_off 0）──");
    bench_t3_replica(&k, "t3复刻 E49 blob8M scale0", 49, 8 * 1024 * 1024, 0);
    bench_t3_replica(&k, "t3复刻 E49 blob8M 真scale", 49, 8 * 1024 * 1024, gu_scale);
    bench_t3_replica(&k, "t3复刻 E40 blob2.6M 真scale", 40, blob, gu_scale);

    eprintln!("── ④ 纯读基线（同样字节、不算） ──");
    for &mode in &[0i32, 1, 2, 3] {
        bench_read_slots(&k, &format!("只读/解码 同形状 mode{mode} E40"), 40, mode);
    }
    for &e in &[80usize, 160] {
        bench_read_slots(&k, &format!("只读 同形状 mode0 E{e}"), e, 0);
    }
    bench_read_flat(
        &k,
        "只读 grid-stride uint4（同 65.5 MB 权重字节）",
        40 * GU_N * (GU_K / 2),
    );
}
