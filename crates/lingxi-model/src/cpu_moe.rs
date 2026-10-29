//! **host 侧 Q4G64 专家 GEMV**（给"装不进显存的专家"做兜底）。
//!
//! 为什么要它（2026-10-16 实测，`docs/PLAN.md` §P10.5）：
//! 真语料下命中率 80.6% ⇒ **每轮仍有 382 个 miss × 2.61 MB = 997 MB 要过 PCIe**，
//! 在 ~10 GB/s 上就是 **~100 ms/轮**，占真语料整轮（173 ms）的 **58%**。
//! 而同样的 997 MB 从 **RAM** 读只要 ~10 ms，且 CPU 算力够（1.88 G MAC/轮 ÷ 85 ms ≈ 22 GMAC/s）。
//! ⇒ **把那部分专家交给 CPU 算，跨 PCIe 只传激活**（每轮 ~0.7 MB，少 1400 倍）。
//! 这也正是参考实现必须有 CPU 专家池（它文档写明 `--no-pool` 根本起不来）的原因。
//!
//! **布局契约与 GPU 核 `lingxi_gemv_q4g64_slotsm_body` 完全一致**（照抄 CUDA 侧）：
//! - 每个专家的 blob = `[gu 段 | dn 段]`；段内前面是 4-bit 载荷、紧跟 fp16 scale 平面；
//! - 段内行 `r` 的载荷在 `payload + r*(k_pad/2)`，scale 在 `payload + r*(k_pad/64)` 个 fp16；
//! - 组 `g` 占 32 字节载荷，字节 `z` 里**低 nibble 是列 `2z`、高 nibble 是列 `2z+1`**；
//! - nibble 是**带符号** 4-bit（0..7 → 0..7，8..15 → −8..−1）；
//! - `seg_off` / `scale_off` 都相对**槽/blob 起点**（与 CUDA 核同口径：`payload = slot + seg_off`，
//!   `scales = payload + scale_off`）；
//! - 累加次序：组内 `dot` 串行、组间 `acc += scale*dot` 串行 —— 与 GPU 核同序。
//!
//! ## ⚠️ 结论（2026-10-16 实测）：**这条路暂时不划算，默认关**
//! `throughput_go_nogo` 实测（真尺寸 hidden=2560/inter=640）：**单线程 284 MB/s，20 线程 1.88 GB/s**。
//! 真语料 miss 997 MB/轮 ÷ 1.88 GB/s = **530 ms**，而走 PCIe 那条路只要 **~100 ms**
//! ⇒ **CPU 兜底比"搬过 PCIe"更慢**。根因：GPU 的 Q4 反量化 GEMV 实测能跑到 **271 GB/s**，
//! CPU 即使 AVX2 拉满（10 GB/s 级）也追不上 —— "DDR3 比 PCIe 快 10 倍"这个前提在**算力**上不成立。
//! ⇒ **miss 的唯一解是"更多显存"（两卡 ⇒ 槽位 7000→14000 ⇒ miss 19.4%→~3%）**，
//!    而不是"换个地方算"。本模块保留原因：它是**两卡也装不下**（长会话跨多话题、槽位被填满）时
//!    唯一的兜底；正确性已由 2 个单测锁死（gu 行级 + 整条 gu→swiglu→dn，都对独立手算参照）。

use half::f16;

/// Q4 组大小（每 64 列一个 fp16 scale）。
const G: usize = 64;

#[inline]
fn code4(nib: u32) -> f32 {
    let v = nib as i32;
    (if v >= 8 { v - 16 } else { v }) as f32
}

#[inline]
fn f16_at(bytes: &[u8], i: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[i], bytes[i + 1]])).to_f32()
}

/// 单行：`y[r] = W[n=r, 0..k] · x[0..k]`（Q4G64）。
/// `seg` = 载荷起点（已含 `seg_off` 的切片），`scales` = scale 平面起点（已含 `seg_off+scale_off`）。
#[inline]
pub fn gemv_row(seg: &[u8], scales: &[u8], k_pad: usize, k: usize, r: usize, x: &[f32]) -> f32 {
    let rb = k_pad / 2;
    let rg = k_pad / G;
    let pay = &seg[r * rb..];
    let sc = &scales[r * rg * 2..];
    let ncol = k.min(k_pad);
    let ng = ncol.div_ceil(G);
    let mut acc = 0.0f32;
    for g in 0..ng {
        let base = g * G;
        let nb = (ncol - base).min(G).div_ceil(2);
        let gp = &pay[g * 32..g * 32 + nb];
        let mut dot = 0.0f32;
        for (z, &byte) in gp.iter().enumerate() {
            let c0 = base + 2 * z;
            dot += code4((byte & 0x0F) as u32) * x[c0];
            if c0 + 1 < ncol {
                dot += code4((byte >> 4) as u32) * x[c0 + 1];
            }
        }
        acc += f16_at(sc, g * 2) * dot;
    }
    acc
}

/// 一个专家的一个 token 行：`gu → swiglu → dn`，结果**累加**进 `out[0..hidden]`。
///
/// `blob` 从**该槽的 gu 段起点**开始的整块（= `arena` 里 `gu.start..down.end` 的字节）；
/// 四个偏移都相对 `blob` 起点（`gu_seg` 通常为 0）。
#[allow(clippy::too_many_arguments)]
pub fn expert_row_forward(
    blob: &[u8],
    gu_seg: usize,
    gu_scale: usize,
    gu_k_pad: usize,
    dn_seg: usize,
    dn_scale: usize,
    dn_k_pad: usize,
    hidden: usize,
    inter: usize,
    x: &[f32],
    tmp_act: &mut [f32],
    out: &mut [f32],
) {
    debug_assert!(tmp_act.len() >= inter && out.len() >= hidden && x.len() >= hidden);
    let gu = &blob[gu_seg..];
    let gu_sc = &blob[gu_seg + gu_scale..];
    for i in 0..inter {
        let gate = gemv_row(gu, gu_sc, gu_k_pad, hidden, i, x);
        let up = gemv_row(gu, gu_sc, gu_k_pad, hidden, inter + i, x);
        let s = gate / (1.0 + (-gate).exp()); // silu(gate)
        tmp_act[i] = s * up;
    }
    let dn = &blob[dn_seg..];
    let dn_sc = &blob[dn_seg + dn_scale..];
    for r in 0..hidden {
        out[r] += gemv_row(dn, dn_sc, dn_k_pad, inter, r, tmp_act);
    }
}

/// 并行版：对一组 `(专家, 行)` 请求做 host 前向，结果写进 `outs`（每个请求一行 `[hidden]`）。
///
/// - `jobs`：`(blob, 请求下标, 该请求要用的 x 行起点)`（`xrows` 是 `[nrows, hidden]` 扁平）
/// - 线程数默认 `可用并行度`；**按专家分组**保证同一 blob 的权重只从 RAM 读一遍。
pub struct CpuJob<'a> {
    pub blob: &'a [u8],
    pub job: usize,
    pub row: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn forward_jobs(
    jobs: &[CpuJob<'_>],
    xrows: &[f32],
    hidden: usize,
    inter: usize,
    gu_seg: usize,
    gu_scale: usize,
    gu_k_pad: usize,
    dn_seg: usize,
    dn_scale: usize,
    dn_k_pad: usize,
    outs: &mut [f32],
) {
    let n = jobs.len();
    if n == 0 {
        return;
    }
    let nthreads = std::thread::available_parallelism().map_or(4, |v| v.get()).min(n);
    let chunk = n.div_ceil(nthreads);
    // 每个线程写**自己的**结果缓冲，最后串行拷回 —— 避免对 `outs` 的多重可变借用。
    let parts: Vec<Vec<f32>> = std::thread::scope(|sc| {
        let mut hs = Vec::with_capacity(nthreads);
        for part in jobs.chunks(chunk) {
            hs.push(sc.spawn(move || {
                let mut res = vec![0.0f32; part.len() * hidden];
                let mut act = vec![0.0f32; inter];
                for (k, j) in part.iter().enumerate() {
                    let x = &xrows[j.row * hidden..(j.row + 1) * hidden];
                    let o = &mut res[k * hidden..(k + 1) * hidden];
                    expert_row_forward(
                        j.blob, gu_seg, gu_scale, gu_k_pad, dn_seg, dn_scale, dn_k_pad, hidden, inter,
                        x, &mut act, o,
                    );
                }
                res
            }));
        }
        hs.into_iter().map(|h| h.join().expect("cpu_moe 线程 panic")).collect()
    });
    for (ci, part) in jobs.chunks(chunk).enumerate() {
        for (k, j) in part.iter().enumerate() {
            outs[j.job * hidden..(j.job + 1) * hidden]
                .copy_from_slice(&parts[ci][k * hidden..(k + 1) * hidden]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个小 blob（1 个专家，`hidden=128 inter=32`），**按布局契约手工摆位**。
    /// 返回 `(blob, gu_scale 偏移, dn_seg 偏移, dn_scale 偏移(相对 dn 段), gu_k_pad, dn_k_pad)`。
    /// ⚠️ 全部是**偏移**不是长度（第一版就是把这个搞混，测试直接炸）。
    fn build_blob(hidden: usize, inter: usize, gu_w: f32, dn_w: f32) -> (Vec<u8>, usize, usize, usize, usize, usize) {
        let gu_k_pad = hidden;
        let dn_k_pad = inter.max(G);
        let rb_g = gu_k_pad / 2;
        let rg_g = gu_k_pad / G;
        let rb_d = dn_k_pad / 2;
        let rg_d = dn_k_pad / G;
        let gu_rows = 2 * inter;
        let dn_rows = hidden;
        let gu_pay = gu_rows * rb_g;
        let gu_sc = gu_rows * rg_g * 2;
        let dn_pay = dn_rows * rb_d;
        let dn_sc = dn_rows * rg_d * 2;
        let mut blob = vec![0u8; gu_pay + gu_sc + dn_pay + dn_sc];
        // 造权重：可被 4bit 精确表示的整数（w_g/w_d 仅作文档——实际编码走下方 put 的闭包）
        let _w_g = |r: usize, c: usize| (((r * 7 + c * 3) % 15) as i32 - 7) as f32 * gu_w;
        let _w_d = |r: usize, c: usize| (((r * 5 + c * 11) % 15) as i32 - 7) as f32 * dn_w;
        // 但 4bit 载荷只能存整数码、scale 是 fp16 ⇒ 用 scale=1.0、权重取整数，便于手算参照
        let put = |blob: &mut Vec<u8>, base: usize, rb: usize, rg: usize, rows: usize, kk: usize, wf: &dyn Fn(usize, usize) -> i32| {
            for r in 0..rows {
                for c in 0..kk {
                    let g = c / G;
                    let z = (c % G) / 2;
                    let nib = (wf(r, c) & 0xF) as u8;
                    let b = &mut blob[base + r * rb + g * 32 + z];
                    if c % 2 == 0 {
                        *b = (*b & 0xF0) | nib;
                    } else {
                        *b = (*b & 0x0F) | (nib << 4);
                    }
                }
                for g in 0..rg {
                    let off = base + rows * rb + r * rg * 2 + g * 2;
                    blob[off..off + 2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
                }
            }
        };
        put(&mut blob, 0, rb_g, rg_g, gu_rows, hidden, &|r, c| ((r * 7 + c * 3) % 15) as i32 - 7);
        let dn_seg = gu_pay + gu_sc;
        put(&mut blob, dn_seg, rb_d, rg_d, dn_rows, inter, &|r, c| ((r * 5 + c * 11) % 15) as i32 - 7);
        // 三段偏移：gu 段内 scale 平面的偏移 = gu 载荷长度；dn 段起点 = 整块 gu；
        // dn 段内 scale 平面偏移 = dn 载荷长度。⚠️ 全是**相对所在段**的偏移（与 CUDA 核同口径）。
        (blob, gu_rows * rb_g, dn_seg, dn_rows * rb_d, gu_k_pad, dn_k_pad)
    }

    /// **吞吐判定**（不是正确性）—— 实测结论：**不划算**（见模块头）。：按真模型尺寸（hidden=2560/inter=640）量单线程与多线程吞吐，
    /// 决定"CPU 兜底 miss 专家"这条路成不成立 —— 需要 ≥ 12 GB/s 的 Q4 处理速度
    /// （真语料 miss 997 MB/轮，整轮预算 ~85 ms）。
    #[test]
    fn throughput_go_nogo() {
        let (hidden, inter) = (2560usize, 640usize);
        let (blob, gu_scale, dn_seg, dn_scale, gu_k_pad, dn_k_pad) = build_blob(hidden, inter, 1.0, 1.0);
        // 造大 blob：只造 1 个专家就够（重复用同一份权重，量的是**每字节的处理速度**）
        let x: Vec<f32> = (0..hidden).map(|i| ((i % 7) as f32) * 0.01 - 0.03).collect();
        let reps = 8;
        let t0 = std::time::Instant::now();
        let mut act = vec![0.0f32; inter];
        let mut o = vec![0.0f32; hidden];
        for _ in 0..reps {
            o.iter_mut().for_each(|v| *v = 0.0);
            expert_row_forward(&blob, 0, gu_scale, gu_k_pad, dn_seg, dn_scale, dn_k_pad, hidden, inter, &x, &mut act, &mut o);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let blob_mb = blob.len() as f64 / 1048576.0;
        eprintln!("[cpu_moe] 单线程：1 个专家-行 {:.2} ms ⇒ **{:.1} MB/s**（blob {:.2} MB）",
                  dt * 1e3, blob_mb / dt, blob_mb);
        // 多线程：20 个"专家-行"任务
        let jobs: Vec<CpuJob> = (0..20).map(|j| CpuJob { blob: &blob, job: j, row: 0 }).collect();
        let xrows = vec![0.0f32; hidden];
        let mut outs = vec![0.0f32; 20 * hidden];
        let t1 = std::time::Instant::now();
        forward_jobs(&jobs, &xrows, hidden, inter, 0, gu_scale, gu_k_pad, dn_seg, dn_scale, dn_k_pad, &mut outs);
        let dt2 = t1.elapsed().as_secs_f64();
        eprintln!("[cpu_moe] 20 线程：20 个专家-行 {:.1} ms ⇒ **{:.1} MB/s**；线程数={}",
                  dt2 * 1e3, 20.0 * blob_mb / dt2,
                  std::thread::available_parallelism().map_or(0, |v| v.get()));
    }

    /// 判据：与"逐列扫、不复用实现取字节方式"的参照实现一致（能抓住 nibble 顺序/符号搞反）。
    #[test]
    fn gu_row_matches_reference() {
        let (hidden, inter) = (128usize, 32usize);
        let (blob, gu_scale, _dn_seg, _dn_scale, k_pad, _) = build_blob(hidden, inter, 1.0, 1.0);
        let x: Vec<f32> = (0..hidden).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
        let rb = k_pad / 2;
        let rg = k_pad / G;
        for r in [0usize, 5, 63] {
            let mut want = 0.0f32;
            for g in 0..rg {
                let mut dot = 0.0f32;
                for c in 0..hidden {
                    if c / G != g {
                        continue;
                    }
                    let z = (c % G) / 2;
                    let byte = blob[r * rb + g * 32 + z];
                    let nib = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    dot += code4(nib as u32) * x[c];
                }
                want += f16_at(&blob, (2 * inter) * rb + r * rg * 2 + g * 2) * dot;
            }
            let got = gemv_row(&blob, &blob[gu_scale..], k_pad, hidden, r, &x);
            assert!((got - want).abs() <= 1e-3 * want.abs().max(1.0), "r={r} got={got} want={want}");
        }
    }

    /// 判据：整条 `gu→swiglu→dn` 与手算参照一致（抓住 swiglu 用错半段 / dn 段偏移搞错）。
    #[test]
    fn expert_forward_matches_reference() {
        let (hidden, inter) = (128usize, 32usize);
        let (blob, gu_scale, dn_seg, dn_scale, gu_k_pad, dn_k_pad) = build_blob(hidden, inter, 1.0, 1.0);
        let x: Vec<f32> = (0..hidden).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
        // 参照：直接用与实现相同的 gemv_row 造 act，再算 dn —— 但 act 用**独立**的逐列公式重算
        let rb_g = gu_k_pad / 2;
        let rg_g = gu_k_pad / G;
        let _wv = |r: usize, c: usize| ((r * 7 + c * 3) % 15) as i32 - 7;
        let _wd = |r: usize, c: usize| ((r * 5 + c * 11) % 15) as i32 - 7;
        let seg = |r: usize, c: usize| -> f32 {
            let g = c / G;
            let z = (c % G) / 2;
            let byte = blob[r * rb_g + g * 32 + z];
            let nib = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
            code4(nib as u32)
        };
        let mut act = vec![0.0f32; inter];
        for i in 0..inter {
            let gate: f32 = (0..hidden).map(|c| seg(i, c) * x[c]).sum();
            let up: f32 = (0..hidden).map(|c| seg(inter + i, c) * x[c]).sum();
            act[i] = (gate / (1.0 + (-gate).exp())) * up;
        }
        let rb_d = dn_k_pad / 2;
        let rg_d = dn_k_pad / G;
        let _ = (rg_g, rg_d);
        let mut want = vec![0.0f32; hidden];
        for r in 0..hidden {
            let _acc = 0.0f32;
            for g in 0..inter.div_ceil(G) {   // ⚠️ 不是 inter/G：inter=32 < G=64 时整除会得 0，参照恒为 0
                let mut dot = 0.0f32;
                for c in 0..inter {
                    if c / G != g {
                        continue;
                    }
                    let z = (c % G) / 2;
                    let byte = blob[dn_seg + r * rb_d + g * 32 + z];
                    let nib = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    dot += code4(nib as u32) * act[c];
                }
                want[r] += f16_at(&blob, dn_seg + dn_scale + r * (dn_k_pad / G) * 2 + g * 2) * dot;
            }
        }
        let mut got = vec![0.0f32; hidden];
        let mut tmp = vec![0.0f32; inter];
        expert_row_forward(
            &blob, 0, gu_scale, gu_k_pad, dn_seg, dn_scale, dn_k_pad, hidden, inter, &x, &mut tmp, &mut got,
        );
        for r in 0..hidden {
            assert!(
                (got[r] - want[r]).abs() <= 1e-2 * want[r].abs().max(1.0),
                "r={r} got={} want={}",
                got[r],
                want[r]
            );
        }
    }
}
