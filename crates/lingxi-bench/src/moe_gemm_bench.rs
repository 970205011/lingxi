//! Q4 专家 GEMM **隔离基准**（判决实验，不进热路径）。
//!
//! 背景：prefill 张量核的**旧 kTiles=4 实验**（q4_tc.cu）整机实测 +8% 更慢被否决
//!（git `829e402`），死在三个已知缺陷（CTA 只盖 32 行 / x 逐 mma 现读现 CVT /
//! 权重无行摊销）。`q4_tcp.cu` 是为修这三个缺陷重写的大 tile 版，但一直缺
//! **单核、固定形状、同一份权重**的隔离判决——本基准补上。四条路径同场对比：
//!
//! 1. `simt` —— 现行 prefill 生产路径（`slotsm rpw=2` + xf32 预转）
//! 2. `tc` —— `q4_tcp` 融合张量核（`mma.m8n8k4`，dequant 在核内转 fp16）
//! 3. `deq` —— 只跑反量化（Q4 → bf16 中转缓冲），量 dequant 本身的开销
//! 4. `deq+cublas` —— 两段式：反量化 + cuBLAS bf16 GEMM，以及 `cublas` 单独一档
//! （权重已是 bf16 时的纯 GEMM 上限）
//!
//! 形态档位（每组 = 一个唯一专家的路由行数）：
//! * `u128t*` = 真实路由量级（23.7K chunk 下 ~217 唯一专家、组均 m≈8~100）；
//! * `u8t*` = 集中形态，看 SIMT 结构性重读权重的放大倍数；
//! * `zipf` = **单 launch 内混合组大小**（1 个热点 2048 + 长尾），真实 prefill
//! 路由就是这种形态；`q4_tcp` 靠核内 `row0 >= t` 早退（q4_tcp.cu:66）避免
//! 小组被 tmax 拉满 grid.z，本档验证该早退在真实长尾下没有悬崖。
//!
//! 判决口径（先定好，免得跑完再找借口）：
//! * 看各档 **墙钟等效速度**（GB/s = Q4 权重字节 ÷ 时间；TF/s 辅助）；
//! * `tc` 在**全部档位** ≥ simt ×1.15 ⇒ 进引擎做整机验证；
//! * 任一档反而更慢 ⇒ 记录"此路不通"，prefill 维持 SIMT，不再整机试。
//!
//! 口径说明：
//! * 权重是**随机字节**（payload 面）+ 常量 scale（0.01）——scale 必须合法，
//! 随机 uint16 一半是 NaN/Inf 会毒化数值对比；payload 随机已足够代表解码 ALU。
//! * 两段式实际流量是 Q4 的 ~4×（bf16 中转写+读），它的 GB/s 天然吃亏，
//! 判决只看相对倍数，不看它的 GB/s。
//! * 计时含入队（rep 次排队后一次同步），与 `bw-ceiling` 同口径，互相可比。
//! * xf32 预转在生产里每层一次、gu/dn 共享摊销；基准里在计时外，单独打印参考。

use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, DevicePtr};
use half::{bf16, f16};
use lingxi_artifact::expert_arena::{matrix_layout, BlobFormat};
use lingxi_kernels::Kernels;

/// 真实形状（hidden=2560, inter=640）：gu 投影 n=2·inter、k=hidden；dn 投影反之。
const SHAPES: [(&str, usize, usize); 2] = [("gu", 1280, 2560), ("dn", 2560, 640)];

fn regimes() -> Vec<(&'static str, Vec<usize>)> {
    let uniform = |e: usize, t: usize| vec![t; e];
    let mut zipf = vec![2048usize];
    zipf.extend(std::iter::repeat(256).take(8));
    zipf.extend(std::iter::repeat(64).take(24));
    zipf.extend(std::iter::repeat(8).take(96));
    vec![
        ("u128t8", uniform(128, 8)),
        ("u128t32", uniform(128, 32)),
        ("u128t128", uniform(128, 128)),
        ("u8t512", uniform(8, 512)),
        ("u8t4096", uniform(8, 4096)),
        ("zipf", zipf),
    ]
}

fn rnd_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed | 1;
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push(s as u8);
    }
    v
}

fn rnd_bf16(n: usize, seed: u64) -> Vec<bf16> {
    let b = rnd_bytes(n * 2, seed);
    b.chunks_exact(2)
        .map(|c| {
            // 取 16 位构造 ±[0,1) 的数：低 15 位小数、符号随机。
            let bits = (u16::from_le_bytes([c[0], c[1]]) & 0x3FFF) | 0x3C00; // 1.x 区间
            let mag: f32 = f16::from_bits(bits).to_f32() - 1.0; // [0,1)
            let sign = if c[1] & 0x40 != 0 { -1.0f32 } else { 1.0 };
            bf16::from_f32(sign * mag)
        })
        .collect()
}

/// 计时口径同 `bw-ceiling`：3 次预热，rep 次排队后一次同步。
fn bw(k: &Kernels, rep: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..3 {
        f()?;
    }
    k.stream.synchronize()?;
    let t = std::time::Instant::now();
    for _ in 0..rep {
        f()?;
    }
    k.stream.synchronize()?;
    Ok(t.elapsed().as_secs_f64() / rep as f64)
}

pub(crate) fn run(args: &[String]) -> Result<()> {
    let dev: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let rep: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let k = Kernels::load(dev)?;
    eprintln!("moe-gemm-bench: device {dev}, rep {rep}");

    for &(tag, n, kk) in &SHAPES {
        let lay = matrix_layout(BlobFormat::Q4G64F16S, n as u64, kk as u64)
            .context("Q4G64F16S 几何非法")?;
        let blob = lay.payload_bytes as usize;
        for (label, sizes) in regimes() {
            run_case(&k, tag, n, kk, blob, lay.scale_offset, label, &sizes, rep)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    k: &Kernels,
    tag: &str,
    n: usize,
    kk: usize,
    blob: usize,
    scale_off: u64,
    label: &str,
    sizes: &[usize],
    rep: usize,
) -> Result<()> {
    let e = sizes.len();
    let tmax = *sizes.iter().max().unwrap_or(&0);
    let uniform = sizes.windows(2).all(|w| w[0] == w[1]);
    let rows: usize = sizes.iter().sum();
    let payload_bytes = e * blob;
    // ── 权重：e 个 blob 连续排，payload 随机、scale 面统一 0.01（合法 fp16）──
    let mut host = rnd_bytes(payload_bytes, 0x51 + n as u64 + rows as u64);
    let sc_bits = f16::from_f32(0.01).to_bits().to_le_bytes();
    let sc_bytes = lay_scale_bytes(n, kk);
    for ei in 0..e {
        let base = ei * blob + scale_off as usize;
        for i in (0..sc_bytes).step_by(2) {
            host[base + i] = sc_bits[0];
            host[base + i + 1] = sc_bits[1];
        }
    }
    let slots: CudaSlice<u8> = k.stream.clone_htod(&host)?;
    // ── 路由形态：前缀和组偏移；x 每行自己的向量（a_stride = k）──
    let mut offs = Vec::with_capacity(e + 1);
    offs.push(0i32);
    for &s in sizes {
        offs.push(offs.last().unwrap() + s as i32);
    }
    let ids = k.stream.clone_htod(&(0..e as i32).collect::<Vec<i32>>())?;
    let group = k.stream.clone_htod(&offs)?;
    let x = k.stream.clone_htod(&rnd_bf16(rows * kk, 0x77 + rows as u64))?;
    let mut y_simt: CudaSlice<bf16> = k.stream.alloc_zeros(rows * n)?;
    let mut y_tc: CudaSlice<bf16> = k.stream.alloc_zeros(rows * n)?;
    let mut y_blas: CudaSlice<bf16> = k.stream.alloc_zeros(rows * n)?;
    let mut w_bf: CudaSlice<bf16> = k.stream.alloc_zeros(e * n * kk)?;
    // xf32 预转（生产里每层一次、gu/dn 共享；此处计时外，打印一次耗时参考）。
    let mut xf: CudaSlice<f32> = k.stream.alloc_zeros(rows * kk)?;
    let t_xf = bw(k, rep, || k.convert_bf16_f32(&x, &mut xf, rows * kk))?;

    let ids_v = ids.slice(0..e);
    let grp_v = group.slice(0..=e);
    let (xf_p, _g) = xf.device_ptr(&k.stream);

    let fmt = |secs: f64| -> String {
        let gbs = payload_bytes as f64 / secs / 1e9;
        let tflops = 2.0 * rows as f64 * n as f64 * kk as f64 / secs / 1e12;
        let ms = secs * 1e3;
        format!("{ms:8.3} ms  {gbs:7.1} GB/s  {tflops:6.2} TF/s")
    };

    // 1) SIMT（生产路径）
    let t_simt = bw(k, rep, || {
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &slots, &ids_v, &x, &grp_v, &mut y_simt, e, n, kk, kk as i32, kk,
            blob as u64, 0, scale_off, 2, xf_p,
        )
    })?;
    // 2) TC 融合核（prefill 大 tile 版；组内早退由核自己做）
    let t_tc = bw(k, rep, || {
        k.gemv_q4g64_slots_tcp_bf16_view(
            &slots, &ids_v, &grp_v, &x, &mut y_tc, e, n, kk, kk, kk as i32,
            blob as u64, 0, scale_off, tmax,
        )
    })?;
    // 3) 只反量化（e 个 blob → bf16 中转）
    let t_deq = bw(k, rep, || {
        for ei in 0..e {
            let src = slots.slice(ei * blob..(ei + 1) * blob);
            let mut dst = w_bf.slice_mut(ei * n * kk..(ei + 1) * n * kk);
            k.dequant_rowsplit_q4g64_bf16_view(&src, &mut dst, n, kk, scale_off)?;
        }
        Ok(())
    })?;
    // 4) 只 cuBLAS：仅组大小一致时可 batched（混合组退化为逐专家计时，仅供参考）。
    let t_blas = if uniform {
        let t = sizes[0];
        Some(bw(k, rep, || {
            k.gemm_bf16_nt_batched(
                &x, &w_bf, &mut y_blas, t as u32, n as u32, kk as u32, e as u32,
                (t * kk) as i64, (n * kk) as i64, (t * n) as i64,
            )
        })?)
    } else {
        None
    };

    // ── 数值对拍：simt 为基准（int 精确解码 × f32 scale）──
    k.stream.synchronize()?;
    let h_simt: Vec<bf16> = k.stream.clone_dtoh(&y_simt)?;
    // TC 重跑一次取输出（上面计时最后跑的是 blas）。
    k.gemv_q4g64_slots_tcp_bf16_view(
        &slots, &ids_v, &grp_v, &x, &mut y_tc, e, n, kk, kk, kk as i32,
        blob as u64, 0, scale_off, tmax,
    )?;
    // blas 参考：uniform 用 batched；混合组逐专家（不计时，仅参考）。
    if uniform {
        let t = sizes[0];
        k.gemm_bf16_nt_batched(
            &x, &w_bf, &mut y_blas, t as u32, n as u32, kk as u32, e as u32,
            (t * kk) as i64, (n * kk) as i64, (t * n) as i64,
        )?;
    } else {
        for (ei, &s) in sizes.iter().enumerate() {
            let xa = x.slice(offs[ei] as usize * kk..offs[ei + 1] as usize * kk);
            let wa = w_bf.slice(ei * n * kk..(ei + 1) * n * kk);
            let mut ya = y_blas.slice_mut(offs[ei] as usize * n..offs[ei + 1] as usize * n);
            k.gemm_bf16_nt(&xa, &wa, &mut ya, s as u32, n as u32, kk as u32)?;
        }
    }
    k.stream.synchronize()?;
    let h_tc: Vec<bf16> = k.stream.clone_dtoh(&y_tc)?;
    let h_blas: Vec<bf16> = k.stream.clone_dtoh(&y_blas)?;
    let rel = |a: &[bf16], b: &[bf16]| -> (f32, f32) {
        let mut max_rel = 0f32;
        let mut max_abs = 0f32;
        for (u, v) in a.iter().zip(b.iter()) {
            let (fu, fv) = (u.to_f32(), v.to_f32());
            let abs = (fu - fv).abs();
            let rel = abs / fu.abs().max(1e-6);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
        }
        (max_rel, max_abs)
    };
    let (rel_tc, abs_tc) = rel(&h_simt, &h_tc);
    let (rel_blas, abs_blas) = rel(&h_simt, &h_blas);

    eprintln!();
    eprintln!("═══ {tag}  n={n} k={kk}  {label}（E={e} 组, {rows} 行, tmax={tmax}）═══");
    eprintln!("  simt  slotsm rpw2+xf32 : {}", fmt(t_simt));
    eprintln!("  tc    q4_tcp 融合核    : {}  ×{:.2} vs simt", fmt(t_tc), t_simt / t_tc);
    eprintln!("  deq   反量化→bf16      : {}", fmt(t_deq));
    if let Some(t_blas) = t_blas {
        eprintln!(
            "  blas  cuBLAS bf16 GEMM  : {}  ×{:.2} vs simt",
            fmt(t_blas),
            t_simt / t_blas
        );
        eprintln!(
            "  两段式 deq+cublas 合计  : {}  ×{:.2} vs simt",
            fmt(t_deq + t_blas),
            t_simt / (t_deq + t_blas)
        );
    } else {
        eprintln!("  blas  （混合组，逐专家调用，不计时）");
    }
    eprintln!("  [参考] xf32 预转 {t_xf:.3} ms/次（{:.1}% of simt）", t_xf / t_simt * 100.0);
    eprintln!("  [数值] simt vs tc   : max_rel={rel_tc:.2e} max_abs={abs_tc:.2e}");
    eprintln!("  [数值] simt vs blas : max_rel={rel_blas:.2e} max_abs={abs_blas:.2e}");
    Ok(())
}

/// scale 面字节数（= rows × groups_per_row × 2，与 `matrix_layout` 一致）。
fn lay_scale_bytes(n: usize, k: usize) -> usize {
    let k_pad = (k + 127) / 128 * 128;
    n * (k_pad / 64) * 2
}
