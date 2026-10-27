//! **专家段 GEMV「乘加结构」微基准**（2026-10-24，）。
//!
//! 上一轮（`t11_moe_gemv_par_bw`）量出：同一份槽布局、同形只读 783 GB/s、+nibble 解包 602、
//! +I2F 596，但**真核只有 271** —— 掉的那一半在"乘加循环"里。本用例把它拆开：
//!
//! * **阶梯探针 P0~P9**（`gemv_struct_exp.cu`，与生产核**逐字节相同的访存形状**）：
//! P0 只读 → P1 +解包 → P2 +I2F → P3 +x 载入 → P4 +FFMA(常量 x) → P5 +x 载入+FFMA
//! → P6 +scale(= 生产算式) → P7 移位对解包 → P8 scale 进 smem → P9 scale+x 进 smem。
//! * **结构变体**：uint4(16 B/lane) 的 r1/r2/r4、lane 连续 16 word、r2+smem、r1+dp4a。
//! * **参照**：生产核（SIMT / dp4a / TC）与 t11 的纯读探针。
//!
//! 口径：GB/s 按**权重字节**（E·n·k/2）折算，与 t11 一致；`--test-threads=1` 是硬要求
//! 。
//!
//! `bash
//! export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib   # cublas/nvrtc 不在默认路径时按需追加
//! cargo test --release -p lingxi-kernels --features gpu --test t12_moe_gemv_struct_bw \
//! -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaView};
use half::{bf16, f16};
use lingxi_kernels::Kernels;

const HIDDEN: usize = 2560;
const INTER: usize = 640;
/// gu 段：n = 2*inter = 1280 行、k = hidden = 2560（每行 Q4 = 1280 B）。
const GU_N: usize = 2 * INTER;
const GU_K: usize = HIDDEN;
/// 参照点（2026-10-24 t11 实测，m=1 生产 SIMT）——用来给每个变体打"相对 271 的倍数"。
const BASE_GBS: f64 = 271.0;

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

fn report(tag: &str, bytes: f64, dt: f64, extra: &str) {
    let gbs = bytes / dt / 1e9;
    eprintln!(
        "[struct] {tag:<44}｜{:8.3} ms｜{:7.1} GB/s｜{:5.2}×271{extra}",
        dt * 1e3,
        gbs,
        gbs / BASE_GBS
    );
}

/// 随机字节 + **合理的 scale**（fp16 在 0.002~0.024；随机字节的 fp16 可能全是 NaN/denorm，
/// 那会让"数值"完全没意义，正确性对照也读不出来）。
fn fill_slots(e: usize, blob: u64, scale_off: u64, rows: usize, groups: usize) -> Vec<u8> {
    let mut b = vec![0u8; e * blob as usize];
    let mut s: u32 = 0x1234_5678;
    for v in b.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *v = (s & 0xFF) as u8;
    }
    for ei in 0..e {
        for i in 0..rows * groups {
            let h = f16::from_f32(0.002 + 0.001 * ((i % 23) as f32));
            let off = ei * blob as usize + scale_off as usize + 2 * i;
            b[off..off + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
    b
}

fn make_x(s: usize, k: usize) -> Vec<bf16> {
    (0..s * k)
        .map(|i| bf16::from_f32(((i * 37 % 101) as f32 / 101.0) - 0.5))
        .collect()
}

struct Ctx {
    e: usize,
    slots: CudaSlice<u8>,
    ids: Vec<i32>,
    goff: Vec<i32>,
    s: usize,
}

impl Ctx {
    fn new(k: &Kernels, e: usize, blob: u64, scale_off: u64, pat: u8) -> Self {
        let host = fill_slots(e, blob, scale_off, GU_N, GU_K / 64);
        let slots = k.stream.clone_htod(&host).unwrap();
        let ids: Vec<i32> = (0..e as i32).collect();
        let goff = group_off(e, pat);
        let s = *goff.last().unwrap() as usize;
        Ctx { e, slots, ids, goff, s }
    }
    fn ids_v(&self, k: &Kernels) -> (CudaSlice<i32>, CudaSlice<i32>) {
        let ids = k.stream.clone_htod(&self.ids).unwrap();
        let goff = k.stream.clone_htod(&self.goff).unwrap();
        (ids, goff)
    }
}

/// 生产核参照。
fn bench_prod(k: &Kernels, tag: &str, c: &Ctx, a_stride: usize, mode: u8, xq_d: &CudaSlice<u8>, sx_d: &CudaSlice<f32>) {
    let (blob, gu_scale, _, _) = geo_q4();
    let ids = k.stream.clone_htod(&c.ids).unwrap();
    let goff = k.stream.clone_htod(&c.goff).unwrap();
    let ids_v = ids.slice(0..c.e);
    let goff_v = goff.slice(0..=c.e);
    let x = k.stream.clone_htod(&make_x(c.s, GU_K)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(c.s * GU_N).unwrap();
    let dt = time_it(k, 60, || match mode {
        0 => k
            .gemv_q4g64_slotsm_rpw_bf16_view(
                &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, GU_N, GU_K, GU_K as i32, a_stride,
                blob, 0, gu_scale, 1, 0,
            )
            .unwrap(),
        1 => k
            .gemv_q4g64_slotsm_rpw_dp4a_view(
                &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, GU_N, GU_K, GU_K as i32, a_stride,
                blob, 0, gu_scale, 1, xq_d, sx_d,
            )
            .unwrap(),
        _ => k
            .gemv_q4g64_slots_tc_bf16(
                &c.slots, &ids_v, &goff_v, &x, &mut y, c.e, GU_N, GU_K, GU_K, GU_K as i32, blob,
                0, gu_scale, 1,
            )
            .unwrap(),
    });
    let bytes = c.e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    report(tag, bytes, dt, &format!("（m={:.1}）", c.s as f64 / c.e as f64));
}

/// `gemv_struct_exp.cu` 的变体（`which` 见 wrapper 注释）。
fn bench_variant(k: &Kernels, tag: &str, c: &Ctx, which: usize, a_stride: usize) {
    let (blob, gu_scale, _, _) = geo_q4();
    let (ids, goff) = c.ids_v(k);
    let ids_v: CudaView<'_, i32> = ids.slice(0..c.e);
    let goff_v: CudaView<'_, i32> = goff.slice(0..=c.e);
    let x = k.stream.clone_htod(&make_x(c.s, GU_K)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(c.s * GU_N).unwrap();
    let dt = time_it(k, 60, || {
        k.gemv_q4g64_struct_exp(
            which, &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, GU_N, GU_K, GU_K as i32,
            a_stride, blob, 0, gu_scale,
        )
        .unwrap();
    });
    let bytes = c.e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    report(tag, bytes, dt, &format!("（m={:.1}）", c.s as f64 / c.e as f64));
}

fn bench_dp4a_u128(k: &Kernels, tag: &str, c: &Ctx, a_stride: usize, xq_d: &CudaSlice<u8>, sx_d: &CudaSlice<f32>) {
    let (blob, gu_scale, _, _) = geo_q4();
    let (ids, goff) = c.ids_v(k);
    let ids_v: CudaView<'_, i32> = ids.slice(0..c.e);
    let goff_v: CudaView<'_, i32> = goff.slice(0..=c.e);
    let x = k.stream.clone_htod(&make_x(c.s, GU_K)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(c.s * GU_N).unwrap();
    let dt = time_it(k, 60, || {
        k.gemv_q4g64_struct_exp_dp4a(
            &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, GU_N, GU_K, GU_K as i32, a_stride, blob,
            0, gu_scale, xq_d, sx_d,
        )
        .unwrap();
    });
    let bytes = c.e as f64 * GU_N as f64 * (GU_K as f64 / 2.0);
    report(tag, bytes, dt, &format!("（m={:.1}）", c.s as f64 / c.e as f64));
}

/// dn 段（n=hidden=2560、k=inter=640）：生产核 vs 变体（`which=None` = 生产核）。
fn bench_dn(k: &Kernels, tag: &str, c: &Ctx, a_stride: usize, which: Option<usize>) {
    let (blob, _, dn_seg, dn_scale) = geo_q4();
    let (ids, goff) = c.ids_v(k);
    let ids_v: CudaView<'_, i32> = ids.slice(0..c.e);
    let goff_v: CudaView<'_, i32> = goff.slice(0..=c.e);
    let x = k.stream.clone_htod(&make_x(c.s, INTER)).unwrap();
    let mut y = k.stream.alloc_zeros::<bf16>(c.s * HIDDEN).unwrap();
    let dt = time_it(k, 60, || match which {
        None => k
            .gemv_q4g64_slotsm_rpw_bf16_view(
                &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, HIDDEN, INTER, INTER as i32,
                a_stride, blob, dn_seg, dn_scale, 1, 0,
            )
            .unwrap(),
        Some(w) => k
            .gemv_q4g64_struct_exp(w, &c.slots, &ids_v, &x, &goff_v, &mut y, c.e, HIDDEN, INTER,
                                   INTER as i32, a_stride, blob, dn_seg, dn_scale)
            .unwrap(),
    });
    let bytes = c.e as f64 * HIDDEN as f64 * (INTER as f64 / 2.0);
    report(tag, bytes, dt, &format!("（m={:.1}）", c.s as f64 / c.e as f64));
}

/// 正确性：变体 vs 生产核（首对 8 行、m=1）。
fn check_variant(k: &Kernels, which: Option<usize>, c: &Ctx) {
    let (blob, gu_scale, _, _) = geo_q4();
    let (ids, goff) = c.ids_v(k);
    let ids_v: CudaView<'_, i32> = ids.slice(0..c.e);
    let goff_v: CudaView<'_, i32> = goff.slice(0..=c.e);
    let xh = make_x(c.s, GU_K);
    let x = k.stream.clone_htod(&xh).unwrap();
    let mut y_ref = k.stream.alloc_zeros::<bf16>(c.s * GU_N).unwrap();
    let mut y_cmp = k.stream.alloc_zeros::<bf16>(c.s * GU_N).unwrap();
    k.gemv_q4g64_slotsm_rpw_bf16_view(
        &c.slots, &ids_v, &x, &goff_v, &mut y_ref, c.e, GU_N, GU_K, GU_K as i32, 0, blob, 0,
        gu_scale, 1, 0,
    )
    .unwrap();
    match which {
        Some(w) => k
            .gemv_q4g64_struct_exp(w, &c.slots, &ids_v, &x, &goff_v, &mut y_cmp, c.e, GU_N, GU_K,
                                   GU_K as i32, 0, blob, 0, gu_scale)
            .unwrap(),
        None => {
            let xq = k.stream.clone_htod(&vec![1u8; c.s * GU_K]).unwrap();
            let sx = k.stream.clone_htod(&vec![0.001f32; c.s]).unwrap();
            k.gemv_q4g64_slotsm_rpw_dp4a_view(
                &c.slots, &ids_v, &x, &goff_v, &mut y_cmp, c.e, GU_N, GU_K, GU_K as i32, 0, blob,
                0, gu_scale, 1, &xq, &sx,
            )
            .unwrap();
        }
    }
    k.stream.synchronize().unwrap();
    let hr: Vec<bf16> = k.stream.clone_dtoh(&y_ref).unwrap();
    let hc: Vec<bf16> = k.stream.clone_dtoh(&y_cmp).unwrap();
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut n_diff = 0usize;
    for (a, b) in hr.iter().zip(hc.iter()) {
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
        }
        let (af, bf) = (a.to_f32(), b.to_f32());
        if af.is_finite() && bf.is_finite() {
            max_abs = max_abs.max((af - bf).abs());
            let den = af.abs().max(1e-6);
            max_rel = max_rel.max((af - bf).abs() / den);
        } else if af.to_bits() != bf.to_bits() {
            eprintln!("[check] 非有限值不一致: {af} vs {bf}");
        }
    }
    eprintln!(
        "[check] 变体 {:?} vs 生产核：{} / {} 元素逐位不同｜max|Δ|={max_abs:.6}｜max rel={max_rel:.3e}",
        which,
        n_diff,
        hr.len()
    );
}

#[test]
fn t12_moe_gemv_struct_bw() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let (blob, gu_scale, _, _) = geo_q4();
    eprintln!(
        "[struct] 几何：Q4 blob {blob} B｜gu n{GU_N} k{GU_K}｜单专家 gu {:.2} MB｜单发 E40 = 65.5 MB \
         ｜参照 271 GB/s",
        GU_N as f64 * (GU_K as f64 / 2.0) / 1e6
    );

    let c40_m1 = Ctx::new(&k, 40, blob, gu_scale, 1);
    let c40_m15 = Ctx::new(&k, 40, blob, gu_scale, 3);

    // dp4a 参照用：int8 激活（恒 1，与 t11 同口径——只量带宽，不判数值）
    let xq_d = k.stream.clone_htod(&vec![1u8; c40_m1.s * GU_K]).unwrap();
    let sx_d = k.stream.clone_htod(&vec![0.001f32; c40_m1.s]).unwrap();

    eprintln!("── 参照：生产核（E40 / gu n1280 k2560）──");
    bench_prod(&k, "生产 SIMT m=1", &c40_m1, 0, 0, &xq_d, &sx_d);
    bench_prod(&k, "生产 SIMT m≈1.5", &c40_m15, GU_K, 0, &xq_d, &sx_d);
    bench_prod(&k, "生产 dp4a m=1", &c40_m1, 0, 1, &xq_d, &sx_d);
    bench_prod(&k, "生产 TC   m=1", &c40_m1, 0, 2, &xq_d, &sx_d);

    eprintln!("── ① 阶梯探针（与生产核逐字节同访存形状，m=1）──");
    let names = [
        "P0 只读（4B/lane）",
        "P1 +nibble 解包（整数）",
        "P2 +I2F（浮点累加）",
        "P3 +x 载入（8 FADD）",
        "P4 +8 FFMA（x=常量，不载入）",
        "P5 +x 载入 + 8 FFMA（无 scale）",
        "P6 +scale（= 生产算式）",
        "P7 P6 + 移位对解包",
        "P8 P6 + scale 进 smem",
        "P9 P8 + x(f32) 进 smem",
    ];
    for (w, nm) in names.iter().enumerate() {
        bench_variant(&k, nm, &c40_m1, w, 0);
    }
    eprintln!("── ①b 同一批探针（m≈1.5）⚠️ 阶梯探针只算每组的**第一对**（m=1 骨架）⇒ 这一档只作定性 ──");
    for &w in &[6usize, 8] {
        bench_variant(&k, names[w], &c40_m15, w, GU_K);
    }

    eprintln!("── ② 结构变体：uint4（16 B/lane）──");
    bench_variant(&k, "u128 r1（1 行/warp）", &c40_m1, 10, 0);
    bench_variant(&k, "u128 r2（2 行/warp）", &c40_m1, 11, 0);
    bench_variant(&k, "u128 r4（4 行/warp）", &c40_m1, 12, 0);
    bench_variant(&k, "contig（lane 连续 16 word）", &c40_m1, 13, 0);
    bench_variant(&k, "u128 r2 + smem(scale+x)", &c40_m1, 14, 0);
    eprintln!("── ②b 结构变体（m≈1.5）──");
    bench_variant(&k, "u128 r1 m≈1.5", &c40_m15, 10, GU_K);
    bench_variant(&k, "u128 r2 m≈1.5", &c40_m15, 11, GU_K);
    bench_variant(&k, "contig m≈1.5", &c40_m15, 13, GU_K);

    eprintln!("── ③ 整数累加档（dp4a，非位等价，只判 CVT 是不是那 300 GB/s 的锅）──");
    bench_dp4a_u128(&k, "u128 r1 + dp4a（int8 激活）", &c40_m1, 0, &xq_d, &sx_d);

    eprintln!("── ③b 生产算式 × 预取结构（MB=8 多对骨架；**全部与生产核逐位等价**）──");
    let c35_m2 = Ctx::new(&k, 35, blob, gu_scale, 2);
    let c7_m1 = Ctx::new(&k, 7, blob, gu_scale, 1);
    for (tag, c, a_stride) in [
        ("E40 m=1", &c40_m1, 0usize),
        ("E40 m≈1.5", &c40_m15, GU_K),
        ("E35 m=2（投机验证轮）", &c35_m2, GU_K),
        ("E7 m=1（线上每层 Q4 组数）", &c7_m1, 0),
    ] {
        // 生产核在同一形状上做对照（口径完全一样）
        bench_prod(&k, &format!("{tag} 生产 SIMT"), c, a_stride, 0, &xq_d, &sx_d);
        for &(w, nm) in &[
            (19usize, "shift4（复刻生产 PF=4）"),
            (15, "unr1（每轮直 load）"),
            (16, "unr2（2 个独立 load）"),
            (17, "unr4（4 个独立 load）"),
            (18, "unr4 + 生产解包"),
        ] {
            bench_variant(&k, &format!("{tag} {nm}"), c, w, a_stride);
        }
    }

    eprintln!("── ④ 正确性（m=1、随机权重/激活）──");
    for w in [6usize, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] {
        check_variant(&k, Some(w), &c40_m1);
    }

    eprintln!("── ⑤ dn 段（n=2560 k=640；线上 rows>1 时 dn 仍走 SIMT，tier 也作用于它）──");
    for (tag, c, a_stride) in [
        ("E7 m=1", &c7_m1, 0usize),
        ("E35 m=2", &c35_m2, GU_K),
    ] {
        bench_dn(&k, &format!("{tag} dn 生产核"), c, a_stride, None);
        for &(w, nm) in &[(15usize, "dn tier unr1"), (16, "dn tier unr2"), (19, "dn shift4")] {
            bench_dn(&k, &format!("{tag} {nm}"), c, a_stride, Some(w));
        }
    }

    eprintln!("── ⑥ u128 结构在**投机验证轮形状**（E35 m=2 / E7 m=1，对照 TC 268~284）──");
    for (tag, c, a_stride) in [
        ("E35 m=2", &c35_m2, GU_K),
        ("E7 m=1", &c7_m1, 0usize),
    ] {
        bench_variant(&k, &format!("{tag} u128 r1"), c, 10, a_stride);
        bench_variant(&k, &format!("{tag} u128 r2"), c, 11, a_stride);
    }
}
