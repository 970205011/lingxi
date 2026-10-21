//! 主干装配：`embedding → hc 广播 → 48 层 → hyper_connection_mixer → lm_head`。
//!
//! 严格对齐 HF `Qwen4ExpTextModel.forward` + `Qwen4ExpForCausalLM.forward`：
//! ```text
//! emb   = embed_tokens(ids)                     # [T, H]
//! hyper = emb.repeat(1, 1, hc)                  # [T, hc*H]
//! for i in 0..n_layers: hyper = layer_i(hyper)  # 单层见 `layer` 模块
//! mixed = hyper_connection_mixer(hyper)         # GR(use_combine=False) → [T, H]
//! logits = lm_head(mixed)                       # [T, vocab]（无最终 norm）
//! ```
//!
//! **权重分层流式**：48 层合计 228 GiB，不可能同时常驻 32G 显存；因此本模块
//! 不持有层权重，而是通过 `load_layer` 回调**逐层取用**（调用方决定常驻/流式）。

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

use crate::layer::{LayerEnv, LayerShape, LayerWeights};
use crate::{gr, layer, moe};

/// 与层无关的公共输入（cos/sin/因果掩码/token id）。
pub struct TrunkEnv<'a> {
    pub cos: &'a CudaSlice<bf16>,
    pub sin: &'a CudaSlice<bf16>,
    pub cos_host: &'a [f32],
    pub sin_host: &'a [f32],
    pub causal_host: &'a [f32],
    pub ids: &'a [i64],
    /// MoE 瘦身路径的按需专家源；`None` = 层权重预装全量专家（对拍路径）
    pub experts: Option<&'a dyn moe::ExpertSource>,
    /// **跨卡 MoE**（`LINGXI_MOE_SPLIT`）：见 `moe::RemoteMoe`。`None` = 现状。
    pub remote: Option<&'a moe::RemoteMoe<'a>>,
}

/// 主干两端的权重（小，可常驻）：embedding / mixer / lm_head。
pub struct TrunkWeights {
    /// [vocab, H]
    pub token_embedding: CudaSlice<bf16>,
    /// `use_combine=False`（`block_inject` 为 `None`）
    pub mixer: gr::GrWeights,
    /// [vocab, H]
    pub output_head: CudaSlice<bf16>,
    /// 同上矩阵的 **W4G64** 版本（可选，`LINGXI_HEAD_Q4=1`）。
    ///
    /// 动机：`lm_head` 是 1.18 GiB 的 bf16，每个 token 全读一遍 —— 是 decode 里
    /// **单块最大的读**（M=1 GEMV，实测 ~2.4 ms/token，占 6%）。W4 后 0.31 GiB，
    /// 顺带把 0.87 GiB 显存让给专家槽（≈330 个槽）。
    pub output_head_q4: Option<crate::q4::Q4W>,
}

/// `load_layer` 回调的返回：层权重 + 该层 indexer 的 `k_layernorm`（host f32；GDN 层为空）。
pub struct LoadedLayer {
    pub weights: LayerWeights,
    pub k_layernorm_host: Vec<f32>,
}

/// 主干前向的两个可对拍输出（都已搬到 host、f32）。
pub struct TrunkOut {
    /// `hyper_connection_mixer` 的输出 `[t, hidden]`（logits 之前的最后一站）
    pub mixed: Vec<f32>,
    /// `lm_head` 的 logits `[t, vocab]`
    pub logits: Vec<f32>,
}

/// 跑一遍主干，返回 `(mixer 输出, logits)`（host，f32，行主序）。
///
/// `load_layer(i)` 每层调用一次；返回的权重在本层用完后即被丢弃（调用方无需缓存）。
pub fn trunk_forward(
    k: &Kernels,
    tw: &TrunkWeights,
    s: &LayerShape,
    n_layers: u32,
    vocab: usize,
    t: usize,
    env: &TrunkEnv,
    load_layer: &mut dyn FnMut(u32) -> Result<LoadedLayer>,
) -> Result<TrunkOut> {
    let (hidden, hc) = (s.hidden, s.hc);
    let dim = hc * hidden;

    // emb = embed_tokens(ids)
    let idx_host: Vec<i32> = env.ids.iter().map(|&v| v as i32).collect();
    let didx = k.stream.clone_htod(&idx_host)?;
    let mut emb = k.stream.alloc_zeros::<bf16>(t * hidden)?;
    k.gather_rows_bf16(&tw.token_embedding, &didx, &mut emb, t, hidden, 0)?;

    // hyper = emb.repeat(1, 1, hc)
    let mut hyper = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.hc_repeat_bf16(&emb, &mut hyper, t, hidden, hc)?;

    for i in 0..n_layers {
        let loaded = load_layer(i)?;
        let lenv = LayerEnv {
            cos: env.cos,
            sin: env.sin,
            cos_host: env.cos_host,
            sin_host: env.sin_host,
            remote: env.remote,
            causal_host: env.causal_host,
            k_layernorm_host: &loaded.k_layernorm_host,
            ids: env.ids,
            experts: env.experts,
        };
        let next = layer::layer_forward(k, &loaded.weights, s, &hyper, t, &lenv, i)?;
        hyper = next;
    }

    // mixed = hyper_connection_mixer(hyper)（GR，无 block_inject）
    let (mixed, _) =
        gr::gated_residual_forward(k, &tw.mixer, &hyper, t as u32, dim, hidden, hc, s.lowrank, s.eps)?;

    // logits = lm_head(mixed)
    let mut logits = k.stream.alloc_zeros::<bf16>(t * vocab)?;
    if let Some(wq) = &tw.output_head_q4 {
        crate::gpu::proj_q4_epi(k, &mixed, wq, &mut logits, vocab, hidden, 0, t, 0, 0.0, 0.0)?;
    } else {
        crate::gpu::proj_t1(k, &mixed, &tw.output_head, &mut logits, vocab, hidden, t)?;
    }
    let mixed_h: Vec<f32> = k.stream.clone_dtoh(&mixed)?.iter().map(|v| v.to_f32()).collect();
    let logits_h: Vec<f32> = k.stream.clone_dtoh(&logits)?.iter().map(|v| v.to_f32()).collect();
    Ok(TrunkOut { mixed: mixed_h, logits: logits_h })
}

/// 每行 logits 的 argmax（贪心下一个 token）。
pub fn argmax_rows(logits: &[f32], rows: usize, vocab: usize) -> Vec<u32> {
    (0..rows)
        .map(|r| {
            let row = &logits[r * vocab..(r + 1) * vocab];
            let mut best = 0usize;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = i;
                }
            }
            best as u32
        })
        .collect()
}

/// 同上，但直接吃 **bf16** logits —— 省掉调用方 `map(to_f32)` 出的那一整份 f32（t×vocab×4 B；
/// t=9、vocab=248320 时是 **9 MB 分配 + 220 万次转换**，每轮一次）。
///
/// **逐位等价**：bf16→f32 是精确且保序的（同指数同尾数，仅补零），所以
/// `a > b` 在 bf16 位模式上与在 f32 值上完全一致 ⇒ 结果与 [`argmax_rows`] 逐位相同。
pub fn argmax_rows_bf16(logits: &[bf16], rows: usize, vocab: usize) -> Vec<u32> {
    (0..rows)
        .map(|r| {
            let row = &logits[r * vocab..(r + 1) * vocab];
            let mut best = 0usize;
            let mut bv = row[0];
            for (i, v) in row.iter().enumerate() {
                if *v > bv {
                    bv = *v;
                    best = i;
                }
            }
            best as u32
        })
        .collect()
}

/// 每行的 **top-K 下标**（按值降序；同值取更小下标）。
///
/// 口径与 [`argmax_rows`] **完全一致**：两者都是"严格 `>` + 按下标升序扫"，
/// 所以 `topk_rows(.., 1)[r][0] == argmax_rows(..)[r]` 恒成立。
///
/// 为什么要它：`top-1 命中率`会**低估草稿头的能力** —— 草稿分布可能完全对，
/// 只是 argmax 差一两位。用 top-N（N=1..20）的命中率才能回答
/// "**该不该做多候选（树）草稿**"：若 top-8 命中率显著高于 top-1，
/// 那树/多候选就有真收益；若 top-20 也只比 top-1 高一点，那就是分布本身偏了。
///
/// 实现是**单趟 O(vocab) + 早退**（不是 K 趟 argmax）：`top` 保持降序定长，
/// 只在 `v > top[K-1].0` 时才做一次插入。实测 24.8 万词表约 0.25 ms/行，
/// 比"K 趟全扫"快 ~20×（K 趟是 5M 次比较 ≈ 5 ms/行）。
/// **一趟**从 bf16 logits 单行取 top-K，并顺带校验有限性。
///
/// 口径与 [`topk_rows`] **逐位相同**（同值取更小下标；满 K 后只接受**严格更大**），
/// 所以它是 `topk_rows` 的位等价替代。
///
/// 为什么要它（`docs/PLAN.md` §P13.5）：`mtp_step` 每步要对 248K 个 logits 做
/// 「建 1 MB f32 Vec + argmax 一趟 + topk 一趟」，KTS 实测 **0.90 ms/步**
/// × 5 步 = **4.5 ms/轮（3.2%）**，而且**每步都让 GPU 空等**。
/// 这里改成单趟直接从 bf16 读（bf16→f32 是精确的，所以比较结果不变）。
pub fn topk_bf16_row(row: &[half::bf16], k: usize) -> Result<Vec<(f32, u32)>> {
    let k = k.max(1);
    let mut top: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (i, &bv) in row.iter().enumerate() {
        let v = bv.to_f32();
        anyhow::ensure!(v.is_finite(), "logits 出现非有限值（下标 {i}）");
        if top.len() == k && !(v > top[k - 1].0) {
            continue;
        }
        let pos = top.iter().position(|&(tv, _)| v > tv).unwrap_or(top.len());
        top.insert(pos, (v, i as u32));
        if top.len() > k {
            top.pop();
        }
    }
    Ok(top)
}

pub fn topk_rows(logits: &[f32], rows: usize, vocab: usize, k: usize) -> Vec<Vec<u32>> {
    let k = k.max(1);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &logits[r * vocab..(r + 1) * vocab];
        let mut top: Vec<(f32, u32)> = Vec::with_capacity(k);
        for (i, &v) in row.iter().enumerate() {
            if top.len() == k && !(v > top[k - 1].0) {
                continue;
            }
            let pos = top.iter().position(|&(tv, _)| v > tv).unwrap_or(top.len());
            top.insert(pos, (v, i as u32));
            if top.len() > k {
                top.pop();
            }
        }
        out.push(top.into_iter().map(|(_, i)| i).collect());
    }
    out
}

#[cfg(test)]
mod topk_tests {
    use super::{argmax_rows, topk_rows};

    /// 判据：`topk_rows(..,1)` 的每行首元素必须与 `argmax_rows` **逐位相同**
    /// （同值取更小下标）。这条不成立的话，所有 top-N 统计都不能信。
    #[test]
    fn top1_matches_argmax() {
        let (rows, vocab) = (5usize, 37usize);
        let mut logits = vec![0f32; rows * vocab];
        for r in 0..rows {
            for c in 0..vocab {
                // 故意造重复峰值（平局）与负值
                logits[r * vocab + c] = ((r * 7 + c * 13) % 11) as f32 - 5.0;
            }
        }
        let am = argmax_rows(&logits, rows, vocab);
        let tk = topk_rows(&logits, rows, vocab, 1);
        for r in 0..rows {
            assert_eq!(tk[r].len(), 1);
            assert_eq!(tk[r][0], am[r], "行 {r} 的 top1 与 argmax 不一致");
        }
    }

    /// top-N 必须真是**最大**的 N 个（不重不漏），且降序。
    #[test]
    fn topk_is_sorted_and_distinct() {
        let (rows, vocab) = (3usize, 64usize);
        let mut logits = vec![0f32; rows * vocab];
        for r in 0..rows {
            for c in 0..vocab {
                logits[r * vocab + c] = (c as f32 * 0.25) - (r as f32);
            }
        }
        let k = 8usize;
        let tk = topk_rows(&logits, rows, vocab, k);
        for r in 0..rows {
            assert_eq!(tk[r].len(), k);
            let vals: Vec<f32> = tk[r].iter().map(|&i| logits[r * vocab + i as usize]).collect();
            for w in vals.windows(2) {
                assert!(w[0] >= w[1], "top-N 不是降序: {vals:?}");
            }
            let uniq: std::collections::HashSet<u32> = tk[r].iter().copied().collect();
            assert_eq!(uniq.len(), k, "top-N 有重复下标");
            // 与"暴力排序取前 k"的**值多重集**一致
            let mut all: Vec<f32> = logits[r * vocab..(r + 1) * vocab].to_vec();
            all.sort_by(|a, b| b.partial_cmp(a).unwrap());
            for (a, b) in vals.iter().zip(all.iter()) {
                assert_eq!(a, b, "top-N 的取值与暴力排序不一致");
            }
        }
    }
}

#[cfg(test)]
mod topk_bf16_tests {
    use super::{argmax_rows, topk_bf16_row, topk_rows};

    /// 判据：**单趟 bf16 版与原来的 `topk_rows` 逐位相同**（含并列的 tie-break），
    /// 且 top-1 必须等于 `argmax_rows`。这条不成立就不能换掉旧路径。
    #[test]
    fn topk_bf16_row_matches_f32() {
        let (rows, vocab, k) = (3usize, 4096usize, 19usize);
        // 造**大量并列**的数据（最容易暴露 tie-break 差异），且都在 bf16 精确可表示范围内
        let bits: Vec<u16> = (0..rows * vocab)
            .map(|i| half::bf16::from_f32(((i * 7919) % 1000) as f32 * 0.01 - 5.0).to_bits())
            .collect();
        let lf: Vec<f32> = bits.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
        let want = topk_rows(&lf, rows, vocab, k);
        let am = argmax_rows(&lf, rows, vocab);
        for r in 0..rows {
            let row: Vec<half::bf16> = bits[r * vocab..(r + 1) * vocab]
                .iter()
                .map(|&b| half::bf16::from_bits(b))
                .collect();
            let got = topk_bf16_row(&row, k).expect("topk_bf16_row");
            let gi: Vec<u32> = got.iter().map(|&(_, i)| i).collect();
            assert_eq!(gi, want[r], "row {r}: top-K 下标与 f32 版不一致");
            assert_eq!(gi[0], am[r], "row {r}: top-1 ≠ argmax_rows");
            // 值也要一致（供 probs 用）
            for &(v, i) in got.iter() {
                assert_eq!(v, lf[r * vocab + i as usize], "row {r}: 值不一致 at {i}");
            }
        }
    }
}
