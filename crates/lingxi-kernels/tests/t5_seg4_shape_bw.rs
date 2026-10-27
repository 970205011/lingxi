//! **seg4（专家段）线上真实形状的带宽微基准**（2026-10-23）。
//!
//! 动机：整机 evprof 的 seg4 层均 0.488~0.492 ms，按混档 arena 的实际字节折算只有
//! ~165 GB/s，而 `t3_gemv_bw` 里同核 m=1 的天花板是 275.6 GB/s。差的那 40% 到底在
//! "核本身"还是"线上形状"（两个视图各一发、组数少、m 混排、dn 的 k 只有 640）？
//!
//! 这个用例只复刻**线上形状**，不改核：
//! - `E=40` 组（唯一专家）、`m` 按 1/2 交替（组均 1.5，对应线上 s≈60）；
//! - `gu`：`n=2*inter=1280`、`k=hidden=2560`；`dn`：`n=hidden=2560`、`k=inter=640`；
//! - 槽距用**线上混档 arena 的真实值**：Q4G64 2,611,200 B / Q3G128 1,920,000 B
//! （由 `expert_arena::matrix_layout` 的公式推得，与 的 2.61/1.92 MB 一致）；
//! - 混档按线上比例切：热 Q4 5 组 + 冷 Q3 35 组（64/512 热占比 12.5%，40 组 ≈ 5 热）。
//!
//! 门控：
//! `cargo test --release -p lingxi-kernels --features gpu --test t5_seg4_shape_bw -- --nocapture`

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

const HIDDEN: usize = 2560;
const INTER: usize = 640;

/// 一个格式（Q4G64 / Q3G128）的槽内几何（`matrix_layout` 同公式，PLANE_ALIGNMENT=256）。
#[derive(Clone, Copy)]
struct Geo {
    blob: u64,
    gu_seg: u64,
    dn_seg: u64,
    gu_scale: u64,
    dn_scale: u64,
    gu_k_pad: i32,
    dn_k_pad: i32,
}

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

/// `bits=4`（G64）或 `3`（G128）。
fn geo(bits: u64) -> Geo {
    let group = if bits == 4 { 64 } else { 128 };
    let layout = |rows: u64, cols: u64| -> (u64, u64, u64) {
        let k_pad = align_up(cols, 128);
        let groups_per_row = k_pad / group;
        let row_bytes = groups_per_row * if bits == 3 { group * 3 / 8 } else { group / 2 };
        let base = rows * row_bytes;
        let scale_off = align_up(base, 256);
        let payload = scale_off + rows * groups_per_row * 2;
        (base, scale_off, payload)
    };
    let (gu_base, gu_scale, gu_payload) = layout((2 * INTER) as u64, HIDDEN as u64);
    let (dn_base, dn_scale_rel, dn_payload) = layout(HIDDEN as u64, INTER as u64);
    let _ = gu_base;
    let _ = dn_base;
    Geo {
        blob: gu_payload + dn_payload,
        gu_seg: 0,
        dn_seg: gu_payload,
        gu_scale,
        // ⚠️ 与 `Q4Consts` 同口径：`dn_scale` 是**相对 dn 段起点**的偏移（核里
        // `scales = slots + id*blob + dn_seg + dn_scale`）。
        dn_scale: dn_scale_rel,
        gu_k_pad: align_up(HIDDEN as u64, 128) as i32,
        dn_k_pad: align_up(INTER as u64, 128) as i32,
    }
}

/// 组内行数模式：`1` = 全 m=1；`2` = 全 m=2；`3` = 1/2 交替（组均 1.5，线上情形）。
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

struct Buf {
    slots: CudaSlice<u8>,
    ids: CudaSlice<i32>,
    goff: CudaSlice<i32>,
    s: usize,
}

fn buf(k: &Kernels, e: usize, g: Geo, pat: u8, same_slot: bool) -> Buf {
    let slots = unsafe { k.stream.alloc::<u8>(e * g.blob as usize) }.unwrap();
    let ids_h: Vec<i32> = if same_slot { vec![0; e] } else { (0..e as i32).collect() };
    let goff_h = group_off(e, pat);
    let ids = k.stream.clone_htod(&ids_h).unwrap();
    let goff = k.stream.clone_htod(&goff_h).unwrap();
    Buf { slots, ids, goff, s: *goff_h.last().unwrap() as usize }
}

/// 一次 launch 的耗时（背靠背、末尾一次 sync；不含 per-launch sync 的固定开销）。
#[allow(clippy::too_many_arguments)]
fn bench(
    tag: &str, k: &Kernels, bits: u64, dn: bool, e: usize, pat: u8, same_slot: bool, xf: bool,
    rpw: usize,
) {
    let g = geo(bits);
    let b = buf(k, e, g, pat, same_slot);
    let (n, kdim, seg, scale, k_pad) = if dn {
        (HIDDEN, INTER, g.dn_seg, g.dn_scale, g.dn_k_pad)
    } else {
        (2 * INTER, HIDDEN, g.gu_seg, g.gu_scale, g.gu_k_pad)
    };
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); b.s * kdim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(b.s * n).unwrap();
    let xf_b = if xf { Some(k.stream.clone_htod(&vec![0.01f32; b.s * kdim]).unwrap()) } else { None };
    let xf_p = match &xf_b {
        Some(v) => {
            use cudarc::driver::DevicePtr;
            v.device_ptr(&k.stream).0
        }
        None => 0,
    };
    let ids_v = b.ids.slice(0..e);
    let goff_v = b.goff.slice(0..=e);
    let run = |y: &mut CudaSlice<bf16>| {
        let r = if bits == 3 {
            k.gemv_q3g128_slotsm_rpw_bf16_view(
                &b.slots, &ids_v, &x, &goff_v, y, e, n, kdim, k_pad, kdim, g.blob, seg, scale,
                rpw, xf_p,
            )
        } else {
            k.gemv_q4g64_slotsm_rpw_bf16_view(
                &b.slots, &ids_v, &x, &goff_v, y, e, n, kdim, k_pad, kdim, g.blob, seg, scale,
                rpw, xf_p,
            )
        };
        r.unwrap();
    };
    for _ in 0..3 {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let iters = 60;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let row_bytes = if bits == 3 {
        (k_pad as u64 / 128) * 48
    } else {
        k_pad as u64 / 2
    };
    let bytes = e as f64 * n as f64 * row_bytes as f64;
    eprintln!(
        "[seg4-bw] {tag} rpw{rpw}: {} E{e} n{n} k{kdim} m{a}｜{:.3} ms/次｜{:.1} GB/s｜{:.2} MB",
        if bits == 3 { "Q3G128" } else { "Q4G64" },
        dt * 1e3,
        bytes / dt / 1e9,
        bytes / 1e6,
        a = (b.s as f64) / e as f64,
    );
}

/// 混档一发「同一 stage 的两个视图」（线上就是这个形状：热 Q4 + 冷 Q3 各一次 launch）。
fn bench_mixed(tag: &str, k: &Kernels, dn: bool, e4: usize, e3: usize, rpw: usize) {
    let g4 = geo(4);
    let g3 = geo(3);
    let b4 = buf(k, e4, g4, 3, false);
    let b3 = buf(k, e3, g3, 3, false);
    let (n, kdim) = if dn { (HIDDEN, INTER) } else { (2 * INTER, HIDDEN) };
    let s4 = b4.s;
    let s3 = b3.s;
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); (s4 + s3) * kdim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>((s4 + s3) * n).unwrap();
    let ids4 = b4.ids.slice(0..e4);
    let goff4 = b4.goff.slice(0..=e4);
    let ids3 = b3.ids.slice(0..e3);
    let goff3 = b3.goff.slice(0..=e3);
    let k_pad4 = if dn { g4.dn_k_pad } else { g4.gu_k_pad };
    let k_pad3 = if dn { g3.dn_k_pad } else { g3.gu_k_pad };
    let run = |y: &mut CudaSlice<bf16>| {
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &b4.slots, &ids4, &x, &goff4, y, e4, n, kdim, k_pad4, kdim, g4.blob,
            if dn { g4.dn_seg } else { g4.gu_seg },
            if dn { g4.dn_scale } else { g4.gu_scale },
            rpw, 0,
        )
        .unwrap();
        k.gemv_q3g128_slotsm_rpw_bf16_view(
            &b3.slots, &ids3, &x, &goff3, y, e3, n, kdim, k_pad3, kdim, g3.blob,
            if dn { g3.dn_seg } else { g3.gu_seg },
            if dn { g3.dn_scale } else { g3.gu_scale },
            rpw, 0,
        )
        .unwrap();
    };
    for _ in 0..3 {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let iters = 60;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let rb4 = k_pad4 as u64 / 2;
    let rb3 = (k_pad3 as u64 / 128) * 48;
    let bytes = (e4 as u64 * n as u64 * rb4 + e3 as u64 * n as u64 * rb3) as f64;
    eprintln!(
        "[seg4-bw] {tag} rpw{rpw}: Q4E{e4}+Q3E{e3} n{n} k{kdim}｜{:.3} ms/次｜{:.1} GB/s｜{:.2} MB",
        dt * 1e3,
        bytes / dt / 1e9,
        bytes / 1e6,
    );
}

#[test]
fn t5_seg4_shape_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let g4 = geo(4);
    let g3 = geo(3);
    eprintln!(
        "[seg4-bw] 几何：Q4 blob {} B（gu_scale {} / dn_seg {} / dn_scale {}）｜Q3 blob {} B",
        g4.blob, g4.gu_scale, g4.dn_seg, g4.dn_scale, g3.blob
    );
    // ★ 生产配置是 `slots_rpw`（默认 1），而 `slotsm_rpw` 的注释说 2 是甜点 ——
    // 先把 rpw 扫出来，别把"参数不对"误判成"核到顶"。
    for &rpw in &[1usize, 2, 4] {
        for &pat in &[1u8, 3] {
            for &bits in &[4u64, 3] {
                bench("gu", &k, bits, false, 40, pat, false, false, rpw);
                bench("dn", &k, bits, true, 40, pat, false, false, rpw);
            }
        }
    }
    // 判因：同槽（40 组都指 0 号槽 ⇒ 权重全在 L2）+ xf 预转（f32 输入）。
    for &bits in &[4u64, 3] {
        bench("gu 同槽", &k, bits, false, 40, 3, true, false, 1);
        bench("gu xf", &k, bits, false, 40, 3, false, true, 1);
        bench("dn 同槽", &k, bits, true, 40, 3, true, false, 1);
        bench("dn xf", &k, bits, true, 40, 3, false, true, 1);
    }
    // 线上混档：热 5 组 Q4 + 冷 35 组 Q3，两发背靠背（各 rpw 一遍）。
    for &rpw in &[1usize, 2, 4] {
        bench_mixed("gu 混档", &k, false, 5, 35, rpw);
        bench_mixed("dn 混档", &k, true, 5, 35, rpw);
        bench_mixed("gu 混档36", &k, false, 5, 31, rpw);
        bench_mixed("dn 混档36", &k, true, 5, 31, rpw);
    }
    // 单发小网格的真实开销（组数少 ⇒ 网格小）：Q4 只 5 组。
    bench("gu(仅5组)", &k, 4, false, 5, 3, false, false, 1);
    bench("dn(仅5组)", &k, 4, true, 5, 3, false, false, 1);
}

/// 线上真的会这样发：**同一个 launch 里两种格式的组混排、他格式用 -1 哨兵跳过**
/// （`batch_dev` 的 views 循环对每个视图都按完整 `0..u` 发网格）。这里量"哨兵组"多贵：
/// 同样 `e_real` 个真组，一种是中间插 `e_sent` 个哨兵（线上形状），一种是紧凑。
fn bench_sentinel(tag: &str, k: &Kernels, bits: u64, dn: bool, e_real: usize, e_sent: usize) {
    bench_sentinel_ex(tag, k, bits, dn, e_real, e_sent, false)
}

/// `tc=true` 走**生产上真正用的 Q4 张量核**（`LINGXI_MOE_TC=1`，`start.sh` 默认开；
/// 只在 `gu` 分支生效，`dn` 恒 SIMT）。
fn bench_sentinel_ex(
    tag: &str, k: &Kernels, bits: u64, dn: bool, e_real: usize, e_sent: usize, tc: bool,
) {
    let g = geo(bits);
    let (n, kdim, seg, scale, k_pad) = if dn {
        (HIDDEN, INTER, g.dn_seg, g.dn_scale, g.dn_k_pad)
    } else {
        (2 * INTER, HIDDEN, g.gu_seg, g.gu_scale, g.gu_k_pad)
    };
    let total = e_real + e_sent;
    let mut ids_h = vec![-1i32; total];
    let mut goff_h = vec![0i32; total + 1];
    let mut real = 0usize;
    for g_i in 0..total {
        // 真组按比例均匀插在 total 个位置里（模拟 Q4/Q3 混排）
        let next_real = (real + 1) * total / e_real;
        if real < e_real && g_i == next_real - 1 {
            ids_h[g_i] = real as i32;
            real += 1;
        }
        let m = if g_i % 2 == 0 { 1 } else { 2 };
        goff_h[g_i + 1] = goff_h[g_i] + m;
    }
    let s = *goff_h.last().unwrap() as usize;
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); s * kdim]).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(s * n).unwrap();
    let slots = unsafe { k.stream.alloc::<u8>(e_real * g.blob as usize) }.unwrap();
    let ids = k.stream.clone_htod(&ids_h).unwrap();
    let goff = k.stream.clone_htod(&goff_h).unwrap();
    let ids_v = ids.slice(0..total);
    let goff_v = goff.slice(0..=total);
    let run = |y: &mut CudaSlice<bf16>| {
        if tc {
            // 生产 gu 分支：LINGXI_MOE_TC=1 ⇒ 张量核槽版（tmax≤8 ⇒ tiles=1）
            k.gemv_q4g64_slots_tc_bf16(
                &slots, &ids_v, &goff_v, &x, y, total, n, kdim, kdim, k_pad, g.blob, seg, scale, 1,
            )
            .unwrap();
        } else if bits == 3 {
            k.gemv_q3g128_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x, &goff_v, y, total, n, kdim, k_pad, kdim, g.blob, seg, scale,
                1, 0,
            )
            .unwrap();
        } else {
            k.gemv_q4g64_slotsm_rpw_bf16_view(
                &slots, &ids_v, &x, &goff_v, y, total, n, kdim, k_pad, kdim, g.blob, seg, scale,
                1, 0,
            )
            .unwrap();
        }
    };
    for _ in 0..3 {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let iters = 60;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut y);
    }
    k.stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let rb = if bits == 3 { (k_pad as u64 / 128) * 48 } else { k_pad as u64 / 2 };
    eprintln!(
        "[seg4-bw] {tag}: {} 真{e_real}+哨兵{e_sent}｜{:.3} ms/次（真组折算 {:.1} GB/s）",
        if bits == 3 { "Q3G128" } else { "Q4G64" },
        dt * 1e3,
        e_real as f64 * n as f64 * rb as f64 / dt / 1e9,
    );
}

fn time_launches(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    dt
}

/// seg4 里那几个**数据量很小**的核（gather / swiglu / combine）单独量一次：
/// 背靠背 launch、末尾一次 sync ⇒ 量的是"串在流上真实占的时间"（含入队）。
#[test]
fn t5_seg4_small_kernels() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let (rows, s, hidden, inter, topk) = (6usize, 60usize, HIDDEN, INTER, 10usize);

    // gather：[s, hidden] bf16 ← x 的 pair 行
    let x = k.stream.clone_htod(&vec![bf16::from_f32(0.01); rows * hidden]).unwrap();
    let idx_h: Vec<i32> = (0..s).map(|i| (i % rows) as i32).collect();
    let idx = k.stream.clone_htod(&idx_h).unwrap();
    let mut xs = k.stream.alloc_zeros::<bf16>(s * hidden).unwrap();
    let dt = time_launches(200, || k.gather_rows_bf16(&x, &idx, &mut xs, s, hidden, 0).unwrap());
    k.stream.synchronize().unwrap();
    eprintln!(
        "[seg4-small] gather: s{s} k{hidden}｜{:.3} ms/次｜{:.1} GB/s（读写各 {:.0} KB）",
        dt * 1e3,
        (2 * s * hidden * 2) as f64 / dt / 1e9,
        (s * hidden * 2) as f64 / 1e3
    );

    // swiglu：[s, 2*inter] → [s, inter]
    let gu = k.stream.clone_htod(&vec![bf16::from_f32(0.01); s * 2 * inter]).unwrap();
    let mut act = k.stream.alloc_zeros::<bf16>(s * inter).unwrap();
    let dt = time_launches(200, || k.swiglu_bf16(&gu, &mut act, s, inter).unwrap());
    k.stream.synchronize().unwrap();
    eprintln!(
        "[seg4-small] swiglu: s{s} inter{inter}｜{:.3} ms/次｜{:.1} GB/s",
        dt * 1e3,
        (3 * s * inter * 2) as f64 / dt / 1e9
    );

    // combine（CSR）：[s, hidden] → [rows, hidden]
    let o = k.stream.clone_htod(&vec![bf16::from_f32(0.01); s * hidden]).unwrap();
    let col_h: Vec<i32> = (0..s as i32).collect();
    let roff_h: Vec<i32> = (0..=rows).map(|r| (r * topk) as i32).collect();
    let w = k.stream.clone_htod(&vec![0.1f32; s]).unwrap();
    let shared = k.stream.clone_htod(&vec![bf16::from_f32(0.02); rows * hidden]).unwrap();
    let mut fo = k.stream.alloc_zeros::<bf16>(rows * hidden).unwrap();
    let col = k.stream.clone_htod(&col_h).unwrap();
    let roff = k.stream.clone_htod(&roff_h).unwrap();
    let dt = time_launches(
        200,
        || k.moe_combine_add_csr_bf16(&o, &col, &roff, &w, &shared, &mut fo, rows, hidden).unwrap(),
    );
    k.stream.synchronize().unwrap();
    eprintln!(
        "[seg4-small] combine: rows{rows} topk{topk} H{hidden}｜{:.3} ms/次｜{:.1} GB/s",
        dt * 1e3,
        ((s + 2 * rows) * hidden * 2) as f64 / dt / 1e9,
    );

    // bf16→f32 预转（混档时生产**不走**这条，只作对照）
    let mut xf = k.stream.alloc_zeros::<f32>(s * hidden).unwrap();
    let dt = time_launches(200, || k.convert_bf16_f32(&xs, &mut xf, s * hidden).unwrap());
    k.stream.synchronize().unwrap();
    eprintln!("[seg4-small] convert_bf16_f32: n{}｜{:.3} ms/次", s * hidden, dt * 1e3);

    // 纯 launch 开销基准（同一条 cudarc 路径）：n=256 的单 block 逐元素核
    let a = k.stream.alloc_zeros::<bf16>(256).unwrap();
    let mut bb = k.stream.alloc_zeros::<bf16>(256).unwrap();
    let dt = time_launches(500, || k.add_bf16(&a, &a, &mut bb, 256).unwrap());
    k.stream.synchronize().unwrap();
    eprintln!("[seg4-small] add_bf16 n=256（≈纯 launch 开销）｜{:.3} ms/次", dt * 1e3);
}

/// 哨兵组开销：线上每个视图都按**完整 u** 发网格，他格式的组靠 -1 哨兵直接 return。
#[test]
fn t5_seg4_sentinel() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    // 实测现场（`[fulldev]` trace）：u≈43，主视图 26 真 + 次视图 17 真
    bench_sentinel("gu Q4 线上(26真+17哨)", &k, 4, false, 26, 17);
    bench_sentinel("gu Q4 紧凑(26真)", &k, 4, false, 26, 0);
    bench_sentinel("dn Q4 线上(26真+17哨)", &k, 4, true, 26, 17);
    bench_sentinel("dn Q4 紧凑(26真)", &k, 4, true, 26, 0);
    bench_sentinel("gu Q3 线上(17真+26哨)", &k, 3, false, 17, 26);
    bench_sentinel("gu Q3 紧凑(17真)", &k, 3, false, 17, 0);
    bench_sentinel("dn Q3 线上(17真+26哨)", &k, 3, true, 17, 26);
    bench_sentinel("dn Q3 紧凑(17真)", &k, 3, true, 17, 0);
    // 生产 gu-Q4 走张量核（LINGXI_MOE_TC=1）：同口径再看一遍
    bench_sentinel_ex("gu Q4-TC 线上(26真+17哨)", &k, 4, false, 26, 17, true);
    bench_sentinel_ex("gu Q4-TC 紧凑(26真)", &k, 4, false, 26, 0, true);
}

/// 组数扫描：线上每个视图**只有 15~30 个真组**。若 per-group 时间随组数变化，
/// 说明 `E=40` 的单点测量不能直接外推（这也是本轮的判因点）。
#[test]
fn t5_seg4_groups_sweep() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    for &bits in &[4u64, 3] {
        for &e in &[5usize, 10, 15, 20, 26, 30, 40] {
            bench("gu", &k, bits, false, e, 3, false, false, 1);
            bench("dn", &k, bits, true, e, 3, false, false, 1);
        }
    }
}
