//! MTP 投机解码运行时（`serve` 用）。
//!
//! 从 `main.rs` 的 `mtp-accept` 分支移植的**精简版**：只保留生产需要的
//! 「草稿链 → 主干验证 → 提交 → MTP 重放」四步，去掉所有诊断/后缀查表/AB 段落。
//! 语义与 `mtp-accept` **逐位一致**（同一套 `mtp_step` 调用次序、同一 min_p 闸门）。
//!
//! 关键约定（照抄 mtp-accept，别改）：
//! - MTP 层号必须用哨兵 `100`（避免与主干 layer 0 共用 CUDA Graph/固定缓冲）。
//! - 每轮草稿从 `(h_last, pending)` 起链，`pending` = KV 里最后喂进去的 token。
//! - 提交后 MTP 的 KV 要回卷到轮前，再用**主干 h** 重放 `keep` 个 cell
//!   `(h_j, tok_{j+1})`，否则下一轮建立在错 KV 上。
//! - `h_last` 每轮更新为 `hyper_rows[keep-1]`。

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use lingxi_artifact::Artifact;
use lingxi_kernels::Kernels;
use lingxi_model::layer::{AttnState, LayerEnv, LayerStates};

/// 一轮投机的产出（`tokens` 已按主干真实 argmax 提交，可直接对外发）。
pub struct Round {
    pub tokens: Vec<i64>,
    pub accepted: usize,
    /// 本轮送进验证的行数（= `chunk.len()`），供上层算"每验证一行的产出"
    pub verified: usize,
}

pub struct SpecRuntime<'a> {
    k: &'a Kernels,
    g: lingxi_model::mtp::MtpWeights,
    ml: crate::MtpLayer,
    arena: Option<crate::mtp_arena::MtpArena<'a>>,
    mst: LayerStates,
    tables: &'a lingxi_driver::RopeTables,
    endpoints: &'a lingxi_model::trunk::TrunkWeights,
    shape: &'a lingxi_model::layer::LayerShape,
    hidden: usize,
    dim: usize,
    vocab: usize,
    kd: usize,
    min_p: f32,
}

impl<'a> SpecRuntime<'a> {
    pub fn load(
        k: &'a Kernels,
        mtp_path: &str,
        tables: &'a lingxi_driver::RopeTables,
        endpoints: &'a lingxi_model::trunk::TrunkWeights,
        shape: &'a lingxi_model::layer::LayerShape,
        rope_len: usize,
    ) -> Result<Self> {
        let mtp_art = Artifact::open(mtp_path)?;
        let g = lingxi_model::mtp::load(k, &mtp_art)?;
        // ★ serve 默认 **Q4 专家**（`: `LINGXI_MTP_Q4_EXPERTS=0` 可关）：bf16 的 512 专家要 5.03 GiB，
        //   在"钉表 8.3 GiB + arena 21.9 GiB"的生产配置下会把 GPU0 顶满（实测 32.4/32.8 GB 卡死）。
        //   Q4 版只占 1.34 GiB。
        let q4 = std::env::var("LINGXI_MTP_Q4_EXPERTS").map(|v| v != "0").unwrap_or(true);
        let ml = crate::build_mtp_layer(k, &mtp_art, q4)?;
        let arena = if q4 { Some(crate::mtp_arena::MtpArena::build(k, &mtp_art)?) } else { None };
        let mst = LayerStates::for_kind(
            k,
            lingxi_ids::LayerKind::QsaSparseAttention,
            shape,
            rope_len + 8,
            None,
        )?;
        let kd: usize = std::env::var("LINGXI_SERVE_SPEC_KD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let min_p: f32 = std::env::var("LINGXI_SPEC_MIN_P")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);
        let hidden = shape.hidden;
        let dim = shape.hc * hidden;
        let vocab = lingxi_model::DIMS.vocab as usize;
        Ok(Self { k, g, ml, arena, mst, tables, endpoints, shape, hidden, dim, vocab, kd, min_p })
    }

    /// 新请求：把 MTP 的注意力计数退回 0（KV 是 append-only，回卷后旧行不会再被读到）。
    pub fn reset(&mut self) {
        if let AttnState::Qsa(q) = &mut self.mst.attn {
            q.rewind(0);
        }
    }

    pub fn kd(&self) -> usize {
        self.kd
    }

    pub fn embed1(&self, id: i64) -> Result<CudaSlice<bf16>> {
        let k = self.k;
        let d = k.stream.clone_htod(&[id as i32])?;
        let mut e = k.stream.alloc_zeros::<bf16>(self.hidden)?;
        k.gather_rows_bf16(&self.endpoints.token_embedding, &d, &mut e, 1, self.hidden, 0)?;
        Ok(e)
    }

    /// MTP 一步：`(hyper[t,dim], emb[t,H])` → (argmax, top1 概率, 层输出 `o`)。
    fn step(
        &mut self,
        hyper: &CudaSlice<bf16>,
        emb: &CudaSlice<bf16>,
        t: usize,
        ids: &[i64],
        need_head: bool,
    ) -> Result<(Vec<u32>, Vec<f32>, CudaSlice<bf16>)> {
        let k = self.k;
        let r = lingxi_model::mtp::stem(k, &self.g, hyper, emb, t, self.shape.eps)?;
        let env = LayerEnv {
            cos: &self.tables.cos,
            sin: &self.tables.sin,
            cos_host: &self.tables.cos_host,
            sin_host: &self.tables.sin_host,
            remote: None,
            causal_host: &[],
            k_layernorm_host: &self.ml.k_layernorm_host,
            ids,
            experts: self.arena.as_ref().map(|a| a as &dyn lingxi_model::moe::ExpertSource),
        };
        const MTP_LAYER_SENTINEL: u32 = 100;
        let out = lingxi_model::layer::layer_step(
            k, &self.ml.weights, self.shape, &r, t, &env, &mut self.mst, MTP_LAYER_SENTINEL,
        )?;
        lingxi_model::scratch::put_bf16(r);
        if !need_head {
            return Ok((Vec::new(), Vec::new(), out));
        }
        let (d, probs) = self.head(&out, t)?;
        Ok((d, probs, out))
    }

    /// 层尾 mixer + lm_head + top-20 → (argmax, prob1)。行间独立 ⇒ 可只对**单行**调用。
    fn head(&self, out: &CudaSlice<bf16>, t: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        let k = self.k;
        let fmix: &lingxi_model::gr::GrWeights = if lingxi_model::flag("LINGXI_MTP_GMIX") {
            &self.endpoints.mixer
        } else {
            &self.ml.mixer
        };
        let (mixed, injw) = lingxi_model::gr::gated_residual_forward(
            k, fmix, out, t as u32, self.dim, self.hidden, self.shape.hc, self.shape.lowrank,
            self.shape.eps,
        )?;
        if let Some(j) = injw {
            lingxi_model::scratch::put_bf16(j);
        }
        let mut logits = k.stream.alloc_zeros::<bf16>(t * self.vocab)?;
        if let Some(wq) = &self.endpoints.output_head_q4 {
            lingxi_model::gpu::proj_q4_epi(
                k, &mixed, wq, &mut logits, self.vocab, self.hidden, 0, t, 0, 0.0, 0.0,
            )?;
        } else {
            lingxi_model::gpu::proj_t1(
                k, &mixed, &self.endpoints.output_head, &mut logits, self.vocab, self.hidden, t,
            )?;
        }
        lingxi_model::scratch::put_bf16(mixed);
        // GPU 两段式 top-K（与 `topk_bf16_row` 同一全序；只取 top-20 算 prob1）
        const G: usize = 64;
        const KK: usize = 20;
        let mut pval = lingxi_model::scratch::f32(k, t * G * KK)?;
        let mut pidx = lingxi_model::scratch::i32(k, t * G * KK)?;
        let mut oval = lingxi_model::scratch::f32(k, t * KK)?;
        let mut oidx = lingxi_model::scratch::i32(k, t * KK)?;
        let mut bad = k.stream.alloc_zeros::<i32>(1)?;
        k.topk_rows_part_bf16(&logits, &mut pval, &mut pidx, &mut bad, t, self.vocab, G, KK)?;
        k.topk_rows_fin(&pval, &pidx, &mut oval, &mut oidx, t, G, KK)?;
        let vals: Vec<f32> = k.stream.clone_dtoh(&oval)?;
        let idxs: Vec<i32> = k.stream.clone_dtoh(&oidx)?;
        let bad_h: Vec<i32> = k.stream.clone_dtoh(&bad)?;
        anyhow::ensure!(bad_h[0] == 0, "MTP logits 出现非有限值");
        lingxi_model::scratch::put_f32(pval);
        lingxi_model::scratch::put_i32(pidx);
        lingxi_model::scratch::put_f32(oval);
        lingxi_model::scratch::put_i32(oidx);
        let mut d = Vec::with_capacity(t);
        let mut probs = Vec::with_capacity(t);
        for r in 0..t {
            let row = &vals[r * KK..(r + 1) * KK];
            let l0 = row[0];
            let mut s = 0f64;
            for &v in row.iter().skip(1) {
                s += ((v - l0) as f64).exp();
            }
            probs.push((1.0 / (1.0 + s)) as f32);
            d.push(idxs[r * KK] as u32);
        }
        Ok((d, probs))
    }

    /// prompt 相位：喂整个 prompt 的 `(h_i, ids[i+1])`，返回最后一个位置的草稿。
    ///
    /// ⚠️ 两个关键点：
    /// 1. **只对最后一行跑 head**：`step(need_head=false)` 已推进 MTP 的 KV，
    ///    而 head 逐行独立 —— 对 `prompt_len` 行全跑要多算 `prompt_len × vocab` logits + topk。
    /// 2. **分块喂**（默认 128 行）：MTP 层是因果的、位置由 QSA 的 `len` 自动推进，
    ///    分块与整段等价（与主干的 prefill 分块同一依据）；同时把峰值显存压到可预测。
    ///    实测整段 867 行会让 MTP head 的 logits 出现非有限值（疑似大 t 的缓冲/路径问题）。
    pub fn prime(
        &mut self,
        hyper_prompt: &CudaSlice<bf16>,
        ids: &[i64],
        emb_prompt: CudaSlice<bf16>,
    ) -> Result<i64> {
        let k = self.k;
        // ⚠️ 行数取 `h`/`emb` 的实际行数（= prompt 长度），**不是** `ids.len()`：
        //    `ids` 比它们多一个（prefill 已把首个生成 token 推进 ids）。
        let n = hyper_prompt.len() / self.dim;
        anyhow::ensure!(
            emb_prompt.len() == n * self.hidden && ids.len() >= n,
            "prime 形状不匹配：h={} 行、emb={} 行、ids={}",
            n,
            emb_prompt.len() / self.hidden,
            ids.len()
        );
        let chunk: usize = std::env::var("LINGXI_MTP_PRIME_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128)
            .max(1);
        let mut off = 0usize;
        let mut last_out: Option<CudaSlice<bf16>> = None;
        while off < n {
            let c = chunk.min(n - off);
            let mut hb = lingxi_model::scratch::bf16(k, c * self.dim)?;
            k.stream.memcpy_dtod(
                &hyper_prompt.slice(off * self.dim..(off + c) * self.dim),
                &mut hb,
            )?;
            let mut eb = lingxi_model::scratch::bf16(k, c * self.hidden)?;
            k.stream.memcpy_dtod(
                &emb_prompt.slice(off * self.hidden..(off + c) * self.hidden),
                &mut eb,
            )?;
            let (_, _, o) = self.step(&hb, &eb, c, ids, false)?;
            lingxi_model::scratch::put_bf16(hb);
            lingxi_model::scratch::put_bf16(eb);
            if let Some(prev) = last_out.take() {
                lingxi_model::scratch::put_bf16(prev);
            }
            last_out = Some(o);
            off += c;
        }
        lingxi_model::scratch::put_bf16(emb_prompt);
        let o = last_out.expect("prompt 非空");
        let last_c = if n % chunk == 0 { chunk.min(n) } else { n % chunk };
        let mut last = lingxi_model::scratch::bf16(k, self.dim)?;
        k.stream.memcpy_dtod(
            &o.slice((last_c - 1) * self.dim..last_c * self.dim),
            &mut last,
        )?;
        lingxi_model::scratch::put_bf16(o);
        let (d, _) = self.head(&last, 1)?;
        lingxi_model::scratch::put_bf16(last);
        Ok(d[0] as i64)
    }

    /// prompt 的 `ids[1..]` embedding（与 mtp-accept 的 `emb_p` 同构造）。
    pub fn prompt_emb(&self, ids: &[i64]) -> Result<CudaSlice<bf16>> {
        let k = self.k;
        let idx: Vec<i32> = ids[1..].iter().map(|&v| v as i32).collect();
        let n = idx.len();
        let d = k.stream.clone_htod(&idx)?;
        let mut e = k.stream.alloc_zeros::<bf16>(n * self.hidden)?;
        k.gather_rows_bf16(&self.endpoints.token_embedding, &d, &mut e, n, self.hidden, 0)?;
        Ok(e)
    }

    /// 一轮投机。返回本轮**可以对外发**的 token（不含已经发过的 `pending`）。
    #[allow(clippy::too_many_arguments)]
    pub fn round(
        &mut self,
        seq: &mut lingxi_driver::Sequence,
        eng: &lingxi_driver::Engine<'_>,
        provider: &mut dyn lingxi_driver::LayerProvider,
        sb: &mut lingxi_driver::SpecBuffers,
        h_last: &mut CudaSlice<bf16>,
    ) -> Result<Round> {
        let k = self.k;
        seq.spec_snapshot(eng, sb)?;
        let pending = *seq.ids.last().expect("round 需要非空序列");
        let mlen0 = match &self.mst.attn {
            AttnState::Qsa(q) => q.len,
            _ => 0,
        };
        let mut h_in = lingxi_model::scratch::bf16(k, self.dim)?;
        k.stream.memcpy_dtod(h_last, &mut h_in)?;
        let mut drafts: Vec<i64> = Vec::with_capacity(self.kd);
        let mut tok = pending;
        for _ in 0..self.kd {
            let emb = self.embed1(tok)?;
            let (d, pr, o) = self.step(&h_in, &emb, 1, &seq.ids, true)?;
            lingxi_model::scratch::put_bf16(emb);
            lingxi_model::scratch::put_bf16(h_in);
            let p1 = pr.first().copied().unwrap_or(1.0);
            drafts.push(d[0] as i64);
            tok = d[0] as i64;
            h_in = o;
            if self.min_p > 0.0 && p1 < self.min_p {
                break;
            }
        }
        // ② 主干一次批前向验证
        let chunk: Vec<i64> = std::iter::once(pending).chain(drafts.iter().copied()).collect();
        seq.spec_chunk(eng, &chunk, provider, sb)?;
        let mut acc = 0usize;
        for (j, &d) in drafts.iter().enumerate() {
            if d == sb.argmax[j] as i64 {
                acc += 1;
            } else {
                break;
            }
        }
        let keep = acc + 1;
        // ③ 提交
        seq.spec_commit(eng, provider, sb, keep)?;
        // ④ MTP：回卷到轮前，再用主干 h 批量重放 keep 个 cell
        if let AttnState::Qsa(q) = &mut self.mst.attn {
            q.rewind(mlen0);
        }
        let bonus = *seq.ids.last().unwrap();
        let mut hb = lingxi_model::scratch::bf16(k, keep * self.dim)?;
        k.stream.memcpy_dtod(&sb.hyper_rows.slice(0..keep * self.dim), &mut hb)?;
        let toks: Vec<i32> = (0..keep)
            .map(|j| if j + 1 < keep { chunk[j + 1] as i32 } else { bonus as i32 })
            .collect();
        let d_toks = k.stream.clone_htod(&toks)?;
        let mut emb = k.stream.alloc_zeros::<bf16>(keep * self.hidden)?;
        k.gather_rows_bf16(&self.endpoints.token_embedding, &d_toks, &mut emb, keep, self.hidden, 0)?;
        let (_, _, o) = self.step(&hb, &emb, keep, &seq.ids, false)?;
        lingxi_model::scratch::put_bf16(o);
        lingxi_model::scratch::put_bf16(emb);
        lingxi_model::scratch::put_bf16(hb);
        lingxi_model::scratch::put_bf16(h_in);
        // 新 h_last = 被接受前缀最后一行的主干 h
        *h_last = lingxi_model::scratch::bf16(k, self.dim)?;
        k.stream
            .memcpy_dtod(&sb.hyper_rows.slice((keep - 1) * self.dim..keep * self.dim), h_last)?;
        // chunk[0]（pending）上一轮已发；本轮发 chunk[1..keep] + 奖励 token
        let mut tokens: Vec<i64> = chunk[1..keep].to_vec();
        tokens.push(bonus);
        Ok(Round { tokens, accepted: acc, verified: chunk.len() })
    }
}
