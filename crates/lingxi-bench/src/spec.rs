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
//! `(h_j, tok_{j+1})`，否则下一轮建立在错 KV 上。
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
    /// 本轮送进验证的行数（= `chunk.len`），供上层算"每验证一行的产出"
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
    /// 自适应 kd：尾随接受率 EMA（α=0.35/轮，初值 kd）。草稿只跑到
    /// `kd_eff = clamp(⌈acc_ema×2⌉, 3, kd)` —— 接受率高的内容（英文 4.1）恒顶到 kd，
    /// 接受率低的内容（中文 1.5）~7 轮内降到 3 ⇒ 中文 +12%（66.2 vs 59.2，实测）。
    /// α 取 0.35 是为了扛住英文自己的冷启动轮（α=0.5 实测 −1.3%）。倍率 2.0 让
    /// kd_eff=7 的线落在 ema>3.0（英文冷启动谷底 ~3.5 安全，中文稳态 1.47 → 3）。
    /// `LINGXI_SPEC_ADAPT_KD=0` 关闭（恒 kd）。
    acc_ema: f64,
    /// `gemv_q4g64_struct_exp*` 的 E=1 分组常量（slot id = 0；group_off = [0,1]）
    slot1: CudaSlice<i32>,
    goff2: CudaSlice<i32>,
    /// 第二张卡（层切分的 GPU1）。草稿相位它整段空闲 ⇒ 拿来做 `lm_head` 的行二分。
    k1: Option<&'a Kernels>,
    /// GPU1 上的 `lm_head` 后半（行 `[vocab/2, vocab)`），`None` = 不分卡
    head_q4_hi: Option<lingxi_model::q4::Q4W>,
    /// GPU1 上那份 E=1 分组常量（`head_q4_hi` 存在时才有）
    slot1_hi: Option<CudaSlice<i32>>,
    goff2_hi: Option<CudaSlice<i32>>,
    /// 分卡 head 的复用缓冲（**必须在 `load` 里一次分配**：实测每步重新 `alloc_zeros`
    /// 两个 248 KB 要 45 µs、12 个小缓冲要 ~40 µs，全是 cudaMalloc 的地板，白吃 16%）
    hb: Option<HeadBufs>,
}

/// 分卡 head 的常驻缓冲（两张卡各一套）
struct HeadBufs {
    x1: CudaSlice<bf16>,   // GPU1 上的激活（5 KB）
    lo: CudaSlice<bf16>,   // GPU0 的 logits 前半
    hi: CudaSlice<bf16>,   // GPU1 的 logits 后半
    pv0: CudaSlice<f32>,
    pi0: CudaSlice<i32>,
    ov0: CudaSlice<f32>,
    oi0: CudaSlice<i32>,
    bad0: CudaSlice<i32>,
    pv1: CudaSlice<f32>,
    pi1: CudaSlice<i32>,
    ov1: CudaSlice<f32>,
    oi1: CudaSlice<i32>,
    bad1: CudaSlice<i32>,
}

/// 把稠密 Q4 权重的**尾部行** `[lo, rows)` 搬到另一张卡，并按"payload | 对齐 | scales"
/// 重建它自己的 `scale_off`（原缓冲是整块 [rows×row_bytes | pad | rows×groups×2]，
/// 切出来的两段在新缓冲里必须重新排布）。
///
/// 走 host（D2H → H2D）而不是 device-to-device：一是 cudarc 的跨卡 D2D 需要 P2P 上下文，
/// 二是本项目对 x8 链路上的持续跨卡流量有明确禁令。
/// 一次性 ~170 MB，之后每步只有 5 KB。
fn q4_tail_to(
    k_src: &Kernels, wq: &lingxi_model::q4::Q4W, k_dst: &Kernels, lo: usize,
) -> Result<lingxi_model::q4::Q4W> {
    let row_bytes = (wq.k_pad as usize) / 2;
    let groups = (wq.k_pad as usize) / 64;
    let n_tail = wq.rows - lo;
    let pay_off = lo * row_bytes;
    let pay_len = n_tail * row_bytes;
    let sc_off = wq.scale_off as usize + lo * groups * 2;
    let sc_len = n_tail * groups * 2;
    anyhow::ensure!(
        sc_off + sc_len <= wq.packed.len(),
        "q4_tail_to: 形状越界（packed {} < {}）",
        wq.packed.len(),
        sc_off + sc_len
    );
    let pay: Vec<u8> = k_src.stream.clone_dtoh(&wq.packed.slice(pay_off..pay_off + pay_len))?;
    let sc: Vec<u8> = k_src.stream.clone_dtoh(&wq.packed.slice(sc_off..sc_off + sc_len))?;
    let scale_off = ((pay_len + 255) / 256 * 256) as u64;
    let mut buf = Vec::with_capacity(scale_off as usize + sc_len);
    buf.extend_from_slice(&pay);
    buf.resize(scale_off as usize, 0);
    buf.extend_from_slice(&sc);
    let packed = k_dst.stream.clone_htod(&buf)?;
    eprintln!(
        "[draft-head-split] GPU1 驻留 lm_head 尾部：行 {lo}..{}（payload {:.1} MB + scale {:.1} MB）",
        wq.rows,
        pay_len as f64 / 1e6,
        sc_len as f64 / 1e6
    );
    Ok(lingxi_model::q4::Q4W {
        packed,
        rows: n_tail,
        cols: wq.cols,
        k_pad: wq.k_pad,
        scale_off,
    })
}

impl<'a> SpecRuntime<'a> {
    pub fn load(
        k: &'a Kernels,
        k1: Option<&'a Kernels>,
        mtp_path: &str,
        tables: &'a lingxi_driver::RopeTables,
        endpoints: &'a lingxi_model::trunk::TrunkWeights,
        shape: &'a lingxi_model::layer::LayerShape,
        rope_len: usize,
    ) -> Result<Self> {
        let mtp_art = Artifact::open(mtp_path)?;
        let g = lingxi_model::mtp::load(k, &mtp_art)?;
        // ★ serve 默认 **Q4 专家**（`: `LINGXI_MTP_Q4_EXPERTS=0` 可关）：bf16 的 512 专家要 5.03 GiB，
        // 在"钉表 8.3 GiB + arena 21.9 GiB"的生产配置下会把 GPU0 顶满（实测 32.4/32.8 GB 卡死）。
        // Q4 版只占 1.34 GiB。
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
        // ★ 2026-10-25：**kd 默认 5 → 7**。
        // 最优点右移是因为**每步边际成本从 5.1 掉到 3.5 ms**（全驻留专家 + 草稿 head 二分），
        // 而"多提交一枚"的边际接受率还有 0.49 ⇒ 续投划算了。实测（200 tok，交替 A/B）：
        // 容器档 91.9 → **96.0 tok/s**（44→39 轮、接受 3.55→4.08、轮 p50 52→58 ms），
        // 宿主档 88.6 → **94.9**；kd=8/9/10 反而回落到 85.6~88.2（接受饱和在 4.08）。
        // 文本 md5 `62b508f8` 与 kd=5/8/9 逐位相同 ⇒ 纯速度，不是行为变化。
        let kd: usize = std::env::var("LINGXI_SERVE_SPEC_KD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(7);
        let min_p: f32 = std::env::var("LINGXI_SPEC_MIN_P")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);
        let hidden = shape.hidden;
        let dim = shape.hc * hidden;
        let vocab = lingxi_model::DIMS.vocab as usize;
        // `gemv_q4g64_struct_exp[_dp4a]` 的"唯一专家"分组常量（E=1，整块稠密权重当一个专家）
        let slot1 = k.stream.clone_htod(&[0i32])?;
        let goff2 = k.stream.clone_htod(&[0i32, 1i32])?;
        // ★ 2026-10-25 `LINGXI_DRAFT_HEAD_SPLIT`（有第二张卡时**默认开**）：草稿 head 的
        // `lm_head` 按 vocab 行二分到两张卡。草稿相位只有 GPU0 在跑（`SpecRuntime` 一直只拿
        // `k`），另一张卡在草稿的 ~8 ms 里整段空转 —— 这是那份空闲显存/带宽唯一的用法。
        // 逐位不变：同一个核、同一行、同一归约次序；top-K 的全序是 (值降序, 下标升序)，
        // 两半各取 top-20 再按同一全序归并 ≡ 全局 top-20（F.1）。
        let split_on = std::env::var("LINGXI_DRAFT_HEAD_SPLIT").map(|v| v != "0").unwrap_or(true);
        let head_q4_hi = match (k1, endpoints.output_head_q4.as_ref()) {
            (Some(k1), Some(wq)) if split_on && wq.rows > 1 && wq.rows % 2 == 0 => {
                let lo = wq.rows / 2;
                Some(q4_tail_to(k, wq, k1, lo)?)
            }
            _ => None,
        };
        // GPU1 上的 E=1 分组常量（与 GPU0 的 `slot1/goff2` 同值，但必须各驻留一份）
        let (slot1_hi, goff2_hi) = if head_q4_hi.is_some() {
            let k1 = k1.expect("head_q4_hi 存在必有 k1");
            (Some(k1.stream.clone_htod(&[0i32])?), Some(k1.stream.clone_htod(&[0i32, 1i32])?))
        } else {
            (None, None)
        };
        // 分卡 head 的常驻缓冲（一次分配；见 `HeadBufs` 的注释——每步重开要白吃 85 µs）
        let hb = if head_q4_hi.is_some() {
            const G: usize = 64;
            const KK: usize = 20;
            let k1 = k1.expect("head_q4_hi 存在必有 k1");
            let half = vocab / 2;
            Some(HeadBufs {
                x1: k1.stream.alloc_zeros::<bf16>(hidden)?,
                lo: k.stream.alloc_zeros::<bf16>(half)?,
                hi: k1.stream.alloc_zeros::<bf16>(half)?,
                pv0: k.stream.alloc_zeros::<f32>(G * KK)?,
                pi0: k.stream.alloc_zeros::<i32>(G * KK)?,
                ov0: k.stream.alloc_zeros::<f32>(KK)?,
                oi0: k.stream.alloc_zeros::<i32>(KK)?,
                bad0: k.stream.alloc_zeros::<i32>(1)?,
                pv1: k1.stream.alloc_zeros::<f32>(G * KK)?,
                pi1: k1.stream.alloc_zeros::<i32>(G * KK)?,
                ov1: k1.stream.alloc_zeros::<f32>(KK)?,
                oi1: k1.stream.alloc_zeros::<i32>(KK)?,
                bad1: k1.stream.alloc_zeros::<i32>(1)?,
            })
        } else {
            None
        };
        Ok(Self { k, k1, head_q4_hi, slot1_hi, goff2_hi, hb, g, ml, arena, mst, tables, endpoints,
                  shape, hidden, dim, vocab, kd, min_p, acc_ema: (kd as f64 * 0.6).max(3.0), slot1, goff2 })
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
    fn head(&mut self, out: &CudaSlice<bf16>, t: usize) -> Result<(Vec<u32>, Vec<f32>)> {
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
        // ★ 2026-10-25：整表 `logits`（`t×vocab` bf16 ≈ 0.5 MB）**只有单卡路才用**。
        // 分卡路（默认开）的两半 top-20 已在各自卡上算完并归并 ⇒ 这里连分配都不需要。
        // 之前无条件 `alloc_zeros`：每步 1 次 cudaMalloc + memset + free，6 步/轮 ≈ 0.3 ms/轮。
        let mut logits: Option<CudaSlice<bf16>> = None;
        macro_rules! logits_mut {
            () => {{
                if logits.is_none() {
                    logits = Some(k.stream.alloc_zeros::<bf16>(t * self.vocab)?);
                }
                logits.as_mut().unwrap()
            }};
        }
        // ★ 2026-10-25 `LINGXI_DRAFT_HEAD_FAST`：草稿 head（t=1）走**非逐位**快档。
        //
        // 为什么允许非逐位：贪心投机里被接受的前缀永远是**主干自己的 argmax**，草稿只决定
        // "这一轮能接受几个" ⇒ 换核不改变生成文本（判据 = md5 与不开时相同；接受率可能微动，
        // 掉了就关掉）。这条授权**只对草稿 head 成立**，主干 verify 的 head 必须逐位。
        //
        // 0 = 关（走 `proj_q4_epi`，逐位）；1 = `u128_r1`（16 B/lane，bf16 激活，只改求和次序，
        // **默认**）；2 = `u128_r1_dp4a`（int8 激活行量化 + `__dp4a`，整数累加）。
        //
        // 实测（`tests/t19_draft_t1_bw.rs::t19_draft_head_bw`，同进程同权重背靠背，
        // 248320×2560、payload 317.8 MB + scale 19.9 MB，分母 = 同形纯读 382 µs / 883 GB/s）：
        // 生产 727.8 µs（464，52% 分母）/ u128_r1 553.7（610，72%）/ dp4a 521.9（647，79%）；
        // 冷 L2 同向（720 → 567）。⇒ u128 1.31×、dp4a 1.39×。
        // 取 1 而不是 2：dp4a 只多 6%，但要付 int8 激活量化（有损）；u128 只改求和次序。
        // 端到端（128 token 三对）：接受率 2.79/轮 两档完全相同、解码文本 md5 全同。
        let fast_mode: u8 = std::env::var("LINGXI_DRAFT_HEAD_FAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        // 形状守卫：`u128_r1` 的列循环是 `c0 + 32 <= k` 的 32 列步进（不做尾部），
        // hidden 不是 32 的倍数或 k_pad != hidden 时一律回退逐位路。
        let fast_ok = fast_mode > 0
            && t == 1
            && self.hidden % 32 == 0
            && self.endpoints.output_head_q4.as_ref().is_some_and(|w| w.k_pad as usize == self.hidden);
        let mut head_done = false;
        // 分卡路径的产出（已按 (值降序, 下标升序) 归并到全局 top-K）；`Some` 时跳过下面的 topk
        let mut merged: Option<(Vec<f32>, Vec<i32>)> = None;
        if let Some(wq) = &self.endpoints.output_head_q4 {
            if fast_ok && self.hb.is_some() {
                if let (Some(k1), Some(wq_hi), Some(ids1), Some(go1)) =
                    (self.k1, self.head_q4_hi.as_ref(), self.slot1_hi.as_ref(), self.goff2_hi.as_ref())
                {
                    // 缓冲是 `&mut` 的：临时从 `self` 摘出来，算完放回去（借检查器）
                    let mut hb = self.hb.take();
                    let r = match hb.as_mut() {
                        Some(b) => self.head_split(k, wq, k1, wq_hi, ids1, go1, &mixed, b),
                        None => unreachable!("hb.is_some() 已判过"),
                    };
                    self.hb = hb;
                    match r {
                        Ok(m) => {
                            merged = Some(m);
                            head_done = true;
                        }
                        Err(e) => {
                            static WARN2: std::sync::Once = std::sync::Once::new();
                            WARN2.call_once(|| {
                                eprintln!("[draft-head-split] ⚠️ 分卡失败，回退单卡快档: {e:#}");
                            });
                        }
                    }
                }
            }
            if !head_done && fast_ok {
                let ids_v = self.slot1.slice(0..1);
                let go_v = self.goff2.slice(0..2);
                // `LINGXI_Q4GEMV_PROF=1`（与 `proj_q4_epi` 的 t=1 同口径）：流内事件报真 GPU 时间，
                // 用来判"快档到底快了多少"——层外墙钟含 host 往返，量不出内核差。
                let prof = lingxi_model::flag("LINGXI_Q4GEMV_PROF");
                let mut evs = Vec::new();
                if prof {
                    let f = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                    evs.push(k.ctx.new_event(f)?);
                    evs.push(k.ctx.new_event(f)?);
                    evs[0].record(&k.stream)?;
                }
                let r = if fast_mode == 2 {
                    let mut xq = unsafe { k.stream.alloc::<u8>(self.hidden) }?;
                    let mut sx = unsafe { k.stream.alloc::<f32>(1) }?;
                    k.convert_bf16_q8_rs(&mixed, &mut xq, &mut sx, 1, self.hidden)?;
                    k.gemv_q4g64_struct_exp_dp4a(
                        &wq.packed, &ids_v, &mixed, &go_v, logits_mut!(),
                        1, self.vocab, self.hidden, wq.k_pad, 0, 0, 0, wq.scale_off, &xq, &sx,
                    )
                } else {
                    k.gemv_q4g64_struct_exp(
                        10, &wq.packed, &ids_v, &mixed, &go_v, logits_mut!(),
                        1, self.vocab, self.hidden, wq.k_pad, 0, 0, 0, wq.scale_off,
                    )
                };
                if prof && r.is_ok() {
                    evs[1].record(&k.stream)?;
                    k.stream.synchronize()?;
                    if let Ok(ms) = evs[0].elapsed_ms(&evs[1]) {
                        eprintln!(
                            "[q4gemv-prof] 快档{fast_mode} n{} k{} = {:.1} µs｜{:.1} GB/s",
                            self.vocab,
                            self.hidden,
                            ms as f64 * 1e3,
                            (self.vocab * self.hidden / 2) as f64 / (ms as f64 * 1e6)
                        );
                    }
                }
                head_done = match r {
                    Ok(()) => true,
                    Err(e) => {
                        // 静默回退会伪装成"改了但没效果"（实测踩过：一次 misaligned 就白跑一轮）
                        static WARN: std::sync::Once = std::sync::Once::new();
                        WARN.call_once(|| {
                            eprintln!("[draft-head-fast] ⚠️ 快档启动失败，回退逐位路: {e:#}");
                        });
                        false
                    }
                };
            }
            if !head_done {
                lingxi_model::gpu::proj_q4_epi(
                    k, &mixed, wq, logits_mut!(), self.vocab, self.hidden, 0, t, 0, 0.0, 0.0,
                )?;
            }
        } else {
            lingxi_model::gpu::proj_t1(
                k, &mixed, &self.endpoints.output_head, logits_mut!(), self.vocab, self.hidden, t,
            )?;
        }
        lingxi_model::scratch::put_bf16(mixed);
        // GPU 两段式 top-K（与 `topk_bf16_row` 同一全序；只取 top-20 算 prob1）
        const G: usize = 64;
        const KK: usize = 20;
        // 分卡路径已经在两张卡上各取过 top-20 并按同一全序归并好，这里直接用
        let (vals, idxs): (Vec<f32>, Vec<i32>) = if let Some(m) = merged {
            m
        } else {
            let mut pval = lingxi_model::scratch::f32(k, t * G * KK)?;
            let mut pidx = lingxi_model::scratch::i32(k, t * G * KK)?;
            let mut oval = lingxi_model::scratch::f32(k, t * KK)?;
            let mut oidx = lingxi_model::scratch::i32(k, t * KK)?;
            let mut bad = k.stream.alloc_zeros::<i32>(1)?;
            // 走到这里一定是单卡路 ⇒ `logits` 必然已分配（见上面的宏）。
            let logits = logits.as_ref().expect("单卡路必须已分配 logits");
            k.topk_rows_part_bf16(logits, &mut pval, &mut pidx, &mut bad, t, self.vocab, G, KK)?;
            k.topk_rows_fin(&pval, &pidx, &mut oval, &mut oidx, t, G, KK)?;
            let vals: Vec<f32> = k.stream.clone_dtoh(&oval)?;
            let idxs: Vec<i32> = k.stream.clone_dtoh(&oidx)?;
            let bad_h: Vec<i32> = k.stream.clone_dtoh(&bad)?;
            anyhow::ensure!(bad_h[0] == 0, "MTP logits 出现非有限值");
            lingxi_model::scratch::put_f32(pval);
            lingxi_model::scratch::put_i32(pidx);
            lingxi_model::scratch::put_f32(oval);
            lingxi_model::scratch::put_i32(oidx);
            (vals, idxs)
        };
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

    /// **`lm_head` 行二分到两张卡**（草稿相位 GPU1 空闲）。返回**已归并的全局 top-20**。
    ///
    /// 为什么逐位不变：
    /// * 每行 logits 由**同一个核、同一行**算出（GPU1 上只是把权重尾部单独搬了一份，
    /// `row_bytes`/归约次序都相同）⇒ 数值逐位一致；
    /// * top-K 的全序是「值降序, 下标升序」（`topk_ops.cu` 的注释），而全局 top-20 的每一项
    /// 必在其所在半区的 top-20 里 ⇒ 「两半各取 top-20 再按同一全序归并」≡ 全局 top-20。
    ///
    /// 成本：每步一次 5 KB 的 D2H→H2D（激活）与 160 B×2 的回传；一次性 170 MB 权重搬迁在 `load`。
    #[allow(clippy::too_many_arguments)]
    fn head_split(
        &self, k: &Kernels, wq: &lingxi_model::q4::Q4W, k1: &Kernels,
        wq_hi: &lingxi_model::q4::Q4W, ids1: &CudaSlice<i32>, go1: &CudaSlice<i32>,
        mixed: &CudaSlice<bf16>, b: &mut HeadBufs,
    ) -> Result<(Vec<f32>, Vec<i32>)> {
        const G: usize = 64;
        const KK: usize = 20;
        let half = self.vocab / 2;
        // 分阶段主机计时（`LINGXI_DRAFT_PROF=1`）：每段都是阻塞点，差值就是真实串行开销。
        let tim = lingxi_model::draft_prof::on();
        let mut t0 = std::time::Instant::now();
        macro_rules! lap {
            ($tag:expr) => {
                if tim {
                    let now = std::time::Instant::now();
                    eprintln!("[head-split] {:<10} {:7.1} µs", $tag, now.duration_since(t0).as_secs_f64() * 1e6);
                    t0 = now;
                }
            };
        }
        // 激活过一次 PCIe（hidden×2 B = 5 KB）：GPU0 的 stream 上 D2H，GPU1 上 H2D
        let hv: Vec<bf16> = k.stream.clone_dtoh(mixed)?;
        k1.stream.memcpy_htod(&hv, &mut b.x1)?;
        lap!("x过卡");
        let ids0 = self.slot1.slice(0..1);
        let go0 = self.goff2.slice(0..2);
        let ids1v = ids1.slice(0..1);
        let go1v = go1.slice(0..2);
        k.gemv_q4g64_struct_exp(
            10, &wq.packed, &ids0, mixed, &go0, &mut b.lo,
            1, half, self.hidden, wq.k_pad, 0, 0, 0, wq.scale_off,
        )?;
        k1.gemv_q4g64_struct_exp(
            10, &wq_hi.packed, &ids1v, &b.x1, &go1v, &mut b.hi,
            1, half, self.hidden, wq_hi.k_pad, 0, 0, 0, wq_hi.scale_off,
        )?;
        // 两半各取 top-20（两条 stream 各自排队，天然并行）
        k.topk_rows_part_bf16(&b.lo, &mut b.pv0, &mut b.pi0, &mut b.bad0, 1, half, G, KK)?;
        k.topk_rows_fin(&b.pv0, &b.pi0, &mut b.ov0, &mut b.oi0, 1, G, KK)?;
        k1.topk_rows_part_bf16(&b.hi, &mut b.pv1, &mut b.pi1, &mut b.bad1, 1, half, G, KK)?;
        k1.topk_rows_fin(&b.pv1, &b.pi1, &mut b.ov1, &mut b.oi1, 1, G, KK)?;
        lap!("发射");
        // 回传：GPU0 先到先读，GPU1 的那份在它自己的 stream 上本来就在并行算
        let v0: Vec<f32> = k.stream.clone_dtoh(&b.ov0)?;
        let i0: Vec<i32> = k.stream.clone_dtoh(&b.oi0)?;
        let b0: Vec<i32> = k.stream.clone_dtoh(&b.bad0)?;
        lap!("GPU0回传");
        let v1: Vec<f32> = k1.stream.clone_dtoh(&b.ov1)?;
        let i1: Vec<i32> = k1.stream.clone_dtoh(&b.oi1)?;
        let b1: Vec<i32> = k1.stream.clone_dtoh(&b.bad1)?;
        lap!("GPU1回传");
        anyhow::ensure!(b0[0] == 0 && b1[0] == 0, "MTP logits 出现非有限值");
        let mut all: Vec<(f32, i32)> = Vec::with_capacity(2 * KK);
        for j in 0..KK {
            all.push((v0[j], i0[j]));
        }
        for j in 0..KK {
            all.push((v1[j], i1[j] + half as i32));
        }
        // 全序：值降序 → 下标升序（与 `topk_ops.cu::topk_better` 完全一致）
        all.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1))
        });
        all.truncate(KK);
        let (vals, idxs): (Vec<f32>, Vec<i32>) = all.into_iter().unzip();
        Ok((vals, idxs))
    }

    /// prompt 相位：喂整个 prompt 的 `(h_i, ids[i+1])`，返回最后一个位置的草稿。
    ///
    /// ⚠️ 两个关键点：
    /// 1. **只对最后一行跑 head**：`step(need_head=false)` 已推进 MTP 的 KV，
    /// 而 head 逐行独立 —— 对 `prompt_len` 行全跑要多算 `prompt_len × vocab` logits + topk。
    /// 2. **分块喂**（默认 128 行）：MTP 层是因果的、位置由 QSA 的 `len` 自动推进，
    /// 分块与整段等价（与主干的 prefill 分块同一依据）；同时把峰值显存压到可预测。
    /// 实测整段 867 行会让 MTP head 的 logits 出现非有限值（疑似大 t 的缓冲/路径问题）。
    pub fn prime(
        &mut self,
        hyper_prompt: &CudaSlice<bf16>,
        ids: &[i64],
        emb_prompt: CudaSlice<bf16>,
    ) -> Result<i64> {
        let k = self.k;
        // ⚠️ 行数取 `h`/`emb` 的实际行数（= prompt 长度），**不是** `ids.len`：
        // `ids` 比它们多一个（prefill 已把首个生成 token 推进 ids）。
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
        // `LINGXI_SPEC_PROF=1`：一轮的四相位墙上时钟（snapshot/草稿/验证/提交+重放），
        // 投机轮成本结构诊断用。
        let prof = lingxi_model::flag("LINGXI_SPEC_PROF");
        let tp = std::time::Instant::now();
        let mut lap = |tag: &str, t0: std::time::Instant| -> std::time::Instant {
            if prof {
                eprintln!("    [spec-prof] {tag}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
            std::time::Instant::now()
        };
        seq.spec_snapshot(eng, sb)?;
        let mut t = lap("snapshot", tp);
        let pending = *seq.ids.last().expect("round 需要非空序列");
        let mlen0 = match &self.mst.attn {
            AttnState::Qsa(q) => q.len,
            _ => 0,
        };
        let mut h_in = lingxi_model::scratch::bf16(k, self.dim)?;
        k.stream.memcpy_dtod(h_last, &mut h_in)?;
        let mut drafts: Vec<i64> = Vec::with_capacity(self.kd);
        let mut tok = pending;
        // 自适应 kd：EMA 决定本轮草稿链长（上限仍是 self.kd，缓冲按 kd+1 已备好）。
        // 默认开；`LINGXI_SPEC_ADAPT_KD=0` 关闭（每轮一次 env 读取，不构成热点）。
        let adapt = std::env::var("LINGXI_SPEC_ADAPT_KD")
            .map(|v| v != "0")
            .unwrap_or(true);
        let kd_eff = if adapt {
            (self.acc_ema * 2.0).ceil().clamp(3.0, self.kd as f64) as usize
        } else {
            self.kd
        };
        for _ in 0..kd_eff {
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
        t = lap("draft", t);
        let chunk: Vec<i64> = std::iter::once(pending).chain(drafts.iter().copied()).collect();
        // `LINGXI_VERIFY_HEAD_SPLIT`：verify head 的词表行二分到两卡。
        // 判据见 lib.rs 的 `VerifyHeadAlt`：`gemm_q4_skinny_bf16` 无 split-k ⇒ 逐位等价；
        // 端到端 md5 与不开时相同。GPU1 的尾半权重已由 `head_q4_hi` 常驻。
        let head_alt = match (self.k1, self.head_q4_hi.as_ref()) {
            (Some(k1), Some(wq_hi))
                if std::env::var("LINGXI_VERIFY_HEAD_SPLIT").map(|v| v != "0").unwrap_or(true) =>
            {
                Some(lingxi_driver::VerifyHeadAlt { k1, wq_hi })
            }
            _ => None,
        };
        seq.spec_chunk(eng, &chunk, provider, sb, head_alt.as_ref())?;
        t = lap("verify", t);
        // ★ `LINGXI_SPEC_SELFTEST=1`：**验证 A' 的回退原语**（回卷到快照 + 用同一批 token 重跑）。
        // 判据：两次 argmax 必须逐位相同 —— 这条不通，"乐观执行 + 整轮重算"就没有可依的回退。
        // 代价：本轮 verify 跑两遍（只在自检开关下）。跑完状态回到"验证后"，正常流程继续。
        if lingxi_model::flag("LINGXI_SPEC_SELFTEST") {
            let a0 = sb.argmax.clone();
            seq.spec_restore(eng, provider, sb)?;
            seq.spec_chunk(eng, &chunk, provider, sb, head_alt.as_ref())?;
            let same = sb.argmax == a0;
            eprintln!(
                "[spec-selftest] 回卷+重跑：argmax {}（{:?} vs {:?}）",
                if same { "逐位相同 ✓" } else { "**不一致 ✗**" },
                &a0[..a0.len().min(8)],
                &sb.argmax[..sb.argmax.len().min(8)]
            );
            anyhow::ensure!(same, "spec_restore 后重跑的 argmax 与首跑不一致");
        }
        let mut acc = 0usize;
        for (j, &d) in drafts.iter().enumerate() {
            if d == sb.argmax[j] as i64 {
                acc += 1;
            } else {
                break;
            }
        }
        let keep = acc + 1;
        // 自适应 kd 的 EMA 更新（α=0.35：慢到扛住英文自己的冷启动轮——实测 α=0.5 时
        // 前 5 轮冷 miss 会把 ema 砸到 3.1 ⇒ kd_eff=6，英文接受 4.08→3.71、−1.3%）。
        self.acc_ema = 0.65 * self.acc_ema + 0.35 * (acc as f64);
        // ③ 提交
        seq.spec_commit(eng, provider, sb, keep)?;
        t = lap("commit", t);
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
        lap("replay", t);
        Ok(Round { tokens, accepted: acc, verified: chunk.len() })
    }
}
