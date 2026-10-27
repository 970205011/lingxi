//! 单层主干前向组装。数学对齐 HF `Qwen4ExpTextDecoderLayer.forward`：
//! `text
//! hyper = hc_count 路残差流（[T, hc*H]）
//! mixed, injw = attn_gr(hyper) # GR 读出单流
//! a = attn(mixed) | gdn(mixed) # QSA 或 GDN
//! hyper += injw ⊗ a # 注入回 hc 路
//! mixed, injw = mlp_gr(hyper)
//! m = moe(mixed)
//! hyper += injw ⊗ m
//! `
//! PLE（仅个别层）在 GR 之前对 `hyper` 做一次加性注入，见 `plet` 模块。

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

use crate::{gdn, gr, graph, moe, plet, qsa, scratch};

/// 一个层的注意力分支：QSA（全注意力）或 GDN（线性注意力）。
pub enum AttnWeights {
    Qsa { attn: qsa::AttentionWeights, indexer: qsa::IndexerWeights },
    Gdn(gdn::GdnWeights),
}

/// 一个层全部权重（都已烘焙 / 无需运行时转换）。
pub struct LayerWeights {
    pub attn_gr: gr::GrWeights,
    pub mlp_gr: gr::GrWeights,
    pub attn: AttnWeights,
    pub moe: moe::MoeWeights,
    /// 仅 layer 1（`ple_layer_ids=[2]`）非空
    pub ple: Option<crate::plet::PleWeights>,
}

/// 一个层的形状/超参。
#[derive(Debug, Clone, Copy)]
pub struct LayerShape {
    pub hidden: usize,
    pub hc: usize,
    pub lowrank: usize,
    pub eps: f32,
    pub attn: qsa::AttnCfg,
    pub indexer: qsa::IndexerCfg,
    pub gdn: gdn::GdnCfg,
    pub num_experts: usize,
    pub top_k: usize,
    pub inter: usize,
}

/// 与层无关的公共输入（QSA 用；GDN 层忽略）。
pub struct LayerEnv<'a> {
    pub cos: &'a CudaSlice<bf16>,
    pub sin: &'a CudaSlice<bf16>,
    pub cos_host: &'a [f32],
    pub sin_host: &'a [f32],
    /// 因果掩码 `[T, T]`（f32 加性）
    pub causal_host: &'a [f32],
    /// indexer 池化 key 的归一化 gamma（host）
    pub k_layernorm_host: &'a [f32],
    /// 原始 token id（PLE 层需要）
    pub ids: &'a [i64],
    /// MoE 瘦身路径的按需专家源；`None` = 用层权重里预装的全量专家张量
    pub experts: Option<&'a dyn moe::ExpertSource>,
    /// **跨卡 MoE**（`LINGXI_MOE_SPLIT`）：`Some` 时把落在 `layers` 区间内的层的 MoE
    /// 交给 GPU1 跑。`None` = 现状逐位不变。
    pub remote: Option<&'a moe::RemoteMoe<'a>>,
}

/// `x[T, hc*H] → y[T, hc*H]`。`x` 不被修改（内部复制一份残差流）。
pub fn layer_forward(
    k: &Kernels,
    w: &LayerWeights,
    s: &LayerShape,
    x: &CudaSlice<bf16>,
    t: usize,
    env: &LayerEnv,
    layer: u32,
) -> Result<CudaSlice<bf16>> {
    let dim = s.hc * s.hidden;
    let mut hyper = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.stream.memcpy_dtod(x, &mut hyper)?;
    let dbg = if crate::flag("LINGXI_LAYER_DEBUG_DIR") {
        std::env::var("LINGXI_LAYER_DEBUG_DIR").ok()
    } else {
        None
    };

    // ── PLE（仅 layer 1）：先把 PLE 输出加到残差流上 ──
    if let Some(pw) = &w.ple {
        let ple_out = crate::plet::ple_forward(k, pw, &hyper, env.ids, t)?;
        let mut summed = k.stream.alloc_zeros::<bf16>(t * dim)?;
        k.add_bf16(&hyper, &ple_out, &mut summed, t * dim)?;
        hyper = summed;
        if let Some(d) = &dbg {
            dump(d, "ple_out", k, &ple_out)?;
            dump(d, "after_ple", k, &hyper)?;
        }
    }

    // ── 注意力块 ──
    let (mixed, injw) = gr::gated_residual_forward(
        k, &w.attn_gr, &hyper, t as u32, dim, s.hidden, s.hc, s.lowrank, s.eps,
    )?;
    if let Some(d) = &dbg {
        dump(d, "mixed_attn", k, &mixed)?;
    }
    let attn_out = match &w.attn {
        AttnWeights::Qsa { attn, indexer } => {
            let mask = qsa::indexer_mask(
                k, indexer, &mixed, env.cos, env.sin, env.cos_host, env.sin_host,
                env.causal_host, env.k_layernorm_host, t, s.hidden, &s.indexer,
            )?;
            if let Some(d) = &dbg {
                dump_f32(d, "indexer_mask", k, &mask)?;
            }
            qsa::attention_forward(k, attn, &mixed, env.cos, env.sin, &mask, t, &s.attn)?
        }
        AttnWeights::Gdn(g) => gdn::gdn_forward(k, g, &mixed, t, &s.gdn)?,
    };
    if let Some(d) = &dbg {
        dump(d, "attn_out", k, &attn_out)?;
    }
    let injw = injw.expect("attn_gr 必须带 block_inject");
    k.gr_write_inplace_bf16(&mut hyper, &attn_out, &injw, s.hc, s.hidden, t as u32)?;

    // ── MLP 块 ──
    let (mixed, injw) = gr::gated_residual_forward(
        k, &w.mlp_gr, &hyper, t as u32, dim, s.hidden, s.hc, s.lowrank, s.eps,
    )?;
    if let Some(d) = &dbg {
        dump(d, "mixed_mlp", k, &mixed)?;
    }
    let moe_out = moe::sparse_moe_forward(
        k, &w.moe, &mixed, t, s.hidden, s.num_experts, s.top_k, s.inter, None,
        layer, env.experts, env.remote, None,
    )?
    .out;
    let injw = injw.expect("mlp_gr 必须带 block_inject");
    k.gr_write_inplace_bf16(&mut hyper, &moe_out, &injw, s.hc, s.hidden, t as u32)?;

    Ok(hyper)
}

// ───────────────────────────── 增量（driver 用） ─────────────────────────────

/// 一层的**跨 token 状态**：注意力分支状态（QSA 的 KV/indexer cache 或 GDN 的
/// 卷积历史 + S）+ PLE 的膨胀卷积历史（仅 layer 1）。
pub enum AttnState {
    Qsa(qsa::QsaState),
    Gdn(gdn::GdnState),
}

/// 一层在 decode 期间要带进下一步的全部可变状态。GR/MoE 无状态。
pub struct LayerStates {
    pub attn: AttnState,
    pub ple: Option<plet::PleState>,
}

impl LayerStates {
    /// 按权重种类构造空状态。`cap` 是 QSA KV cache 容量（需 ≥ 序列最大长度）。
    pub fn new(k: &Kernels, w: &LayerWeights, s: &LayerShape, cap: usize) -> Result<Self> {
        let attn = match &w.attn {
            AttnWeights::Qsa { .. } => AttnState::Qsa(qsa::QsaState::new(k, &s.attn, &s.indexer, cap)?),
            AttnWeights::Gdn(_) => AttnState::Gdn(gdn::GdnState::new(k, &s.gdn)?),
        };
        let ple = match &w.ple {
            Some(pw) => Some(plet::PleState::for_weights(k, pw)?),
            None => None,
        };
        Ok(Self { attn, ple })
    }

    /// 不读权重、只按层种类建空状态（driver 一次建 48 层状态用）。
    /// `ple = Some((conv_kernel, dilation))` 表示该层带 PLE。
    pub fn for_kind(
        k: &Kernels,
        kind: lingxi_ids::LayerKind,
        s: &LayerShape,
        cap: usize,
        ple: Option<(usize, usize)>,
    ) -> Result<Self> {
        let attn = match kind {
            lingxi_ids::LayerKind::QsaSparseAttention => {
                AttnState::Qsa(qsa::QsaState::new(k, &s.attn, &s.indexer, cap)?)
            }
            lingxi_ids::LayerKind::GdnLinearAttention => AttnState::Gdn(gdn::GdnState::new(k, &s.gdn)?),
        };
        let pst = match ple {
            Some((kk, dilation)) => {
                Some(plet::PleState::new(k, s.hc * s.hidden, kk, dilation)?)
            }
            None => None,
        };
        Ok(Self { attn, ple: pst })
    }
}

/// 增量步进：`x[t, hc*H] → y[t, hc*H]`，`states` 就地推进 t 个 token。
///
/// 与 [`layer_forward`] **同一份数学**：GR/MoE 本就逐 token 无状态，注意力分支走
/// `gdn_step` / `attention_step`，PLE 走 `ple_step`。prefill 时以**空状态**传入
/// `t=整段长度, len=0`；decode 时每步 `t=1, len=已确认 token 数`。
///
/// 已缓存 token 数由各 `states` 自身持有（QSA/GDN 的 `state.len`）；调用方只需保证
/// `env.cos`/`sin`（device）与 `env.cos_host`/`sin_host` 覆盖到当前末尾，且
/// `env.ids` 是包含本批 t 个新 token 的全量 token id（PLE 用）。
/// `LINGXI_SKIP=attn,moe,gr` 逗号列表：诊断用，跳掉对应子块以剥离耗时。
fn skip_has(what: &str) -> bool {
    use std::sync::OnceLock;
    static LIST: OnceLock<String> = OnceLock::new();
    let l = LIST.get_or_init(|| std::env::var("LINGXI_SKIP").unwrap_or_default());
    !l.is_empty() && l.split(',').any(|x| x.trim() == what)
}

/// 单层前向（decode 用）。`LINGXI_KT2=1` 时用同步括号量出**该层独占的 GPU+CPU 时间**，
/// 用来回答"每层到底花在 GPU 上多少"。
pub fn layer_step(
    k: &Kernels,
    w: &LayerWeights,
    s: &LayerShape,
    x: &CudaSlice<bf16>,
    t: usize,
    env: &LayerEnv,
    states: &mut LayerStates,
    layer: u32,
) -> Result<CudaSlice<bf16>> {
    layer_step_rec(k, w, s, x, t, env, states, layer, None)
}

/// 与 [`layer_step`] 相同，但把 **GDN 的内部输入（投影的输出 `qkv/z/a/b`）录下来**
/// （`rec.replay == false`）或**从录像重放**（`rec.replay == true`）。见 [`crate::gdn::GdnRec`]。
///
/// 为什么要它：投机验证是"验完 t 行 → 回卷 → 把 GDN 状态挪到 `accept` 行处"。
/// `gdn_step` 的三步（投影 / 卷积 / 递推）**全是 `mixed` 的纯函数** ⇒ 把**投影的输出**录下来，
/// 重放时就只跑「卷积 + 递推」两个小核，**跳过投影那笔最大的权重读**（~30 MB/层）。
/// 录 5 MB（`t_max=7` × 36 层），比"逐行快照整个状态"那条（500 MB）省 99%。
#[allow(clippy::too_many_arguments)]
pub fn layer_step_rec(
    k: &Kernels,
    w: &LayerWeights,
    s: &LayerShape,
    x: &CudaSlice<bf16>,
    t: usize,
    env: &LayerEnv,
    states: &mut LayerStates,
    layer: u32,
    rec: Option<crate::gdn::GdnRec<'_>>,
) -> Result<CudaSlice<bf16>> {
    let kt2 = crate::flag("LINGXI_KT2");
    if kt2 {
        k.stream.synchronize()?;
    }
    let t_kt = std::time::Instant::now();
    let r = layer_step_inner(k, w, s, x, t, env, states, layer, rec);
    if kt2 {
        k.stream.synchronize()?;
        eprintln!(
            "[kt2] 层 {layer:2} 单层 GPU+CPU = **{:.3} ms**（t={t}）",
            t_kt.elapsed().as_secs_f64() * 1e3
        );
    }
    r
}

/// [`layer_dense_step`] 的产物：残差流 + 已经取出、**必须**由 [`layer_moe_finish`]
/// （或 [`layer_moe_apply`]）归还的 MoE GR 固定槽。`t_*` 只给 `LINGXI_PROF` 用。
pub struct DenseStepOut {
    pub hyper: CudaSlice<bf16>,
    pub mixed_m: CudaSlice<bf16>,
    pub injw_m: CudaSlice<bf16>,
    pub(crate) t0: std::time::Instant,
    pub(crate) t_agr: std::time::Duration,
    pub(crate) t_attn: std::time::Duration,
    pub(crate) t_mgr: std::time::Duration,
}

/// [`layer_dense_step`] + MoE 的产物（`hyper` 已加上 MoE 输出）。
pub struct MoeFinishOut {
    pub hyper: CudaSlice<bf16>,
    pub(crate) t_moe: std::time::Instant,
}

/// EP 用的"半场 MoE"产物：`out` 是本卡负责的专家区间的**部分和**（卡 0 另含 shared）。
pub struct MoePartialOut {
    pub out: CudaSlice<bf16>,
    pub(crate) t_moe: std::time::Instant,
}

/// **稠密段**（evprof 段 0..2）：入口 + attn_GR、attn、gr_write + mlp_GR。
/// 原 `layer_step_inner` 的 seg0..seg2 原样搬过来；返回残差流与尚未归还的 MoE GR 槽，
/// 数学/事件顺序与拆分前逐位一致。
#[allow(clippy::too_many_arguments)]
pub fn layer_dense_step(
    k: &Kernels,
    w: &LayerWeights,
    s: &LayerShape,
    x: &CudaSlice<bf16>,
    t: usize,
    env: &LayerEnv,
    states: &mut LayerStates,
    layer: u32,
    rec: Option<crate::gdn::GdnRec<'_>>,
) -> Result<DenseStepOut> {
    let dim = s.hc * s.hidden;
    let t0 = std::time::Instant::now();
    // `LINGXI_EVPROF=1`：逐段记 GPU 事件（**不排空流水**，口径干净；见 lingxi_kernels::evprof）
    let ep = k.evprof.as_ref();
    if let Some(p) = ep {
        p.begin(layer, 0, &k.stream);
    }
    let mut hyper = scratch::bf16(k, t * dim)?;
    // [探针] 地址稳定性检查：CUDA Graph 捕获会把**设备指针烘死**，因此要求
    // decode 路径上同一逻辑缓冲在 token 之间拿到同一个地址。不清 pool 时这是
    // 确定性的（同尺寸 LIFO），这里把它打出来做实证。
    if crate::flag("LINGXI_PTRPROBE") && (layer == 0 || layer == 47) {
        use cudarc::driver::DevicePtr;
        let (hp, _g) = hyper.device_ptr(&k.stream);
        eprintln!("    [ptrprobe] 层 {layer:2} hyper=0x{hp:x}");
    }
    k.stream.memcpy_dtod(x, &mut hyper)?;
    let _t_prep = t0.elapsed();

    // ── PLE（仅 layer 1）：全量 ids + 卷积历史 ──
    // 诊断：`LINGXI_SKIP=ple` 关掉 PLE。这一层每 token 会从 95 GiB 大表按 ngram 哈希
    // 随机取行（表在机械盘上时是**随机读**），会让"跳过子块"的对拍测量串味。
    if let Some(pw) = &w.ple {
        if !skip_has("ple") {
            let kts_ple = crate::flag("LINGXI_KTS");
            if kts_ple {
                k.stream.synchronize()?;
            }
            let t_ple0 = std::time::Instant::now();
            let pst = states.ple.as_mut().expect("PLE 层必须带 PleState");
            crate::gr_prof::ple_begin(k)?;
            let ple_out = plet::ple_step(k, pw, &hyper, env.ids, t, pst)?;
            let mut summed = scratch::bf16(k, t * dim)?;
            k.add_bf16(&hyper, &ple_out, &mut summed, t * dim)?;
            crate::gr_prof::ple_end(k)?;
            if kts_ple {
                k.stream.synchronize()?;
                use std::sync::atomic::{AtomicU32, Ordering};
                static PLE_N: AtomicU32 = AtomicU32::new(0);
                if PLE_N.fetch_add(1, Ordering::Relaxed) < 400 {
                    eprintln!("[kts-ple] PLE 单层 **{:.1} µs**（t={t}）", t_ple0.elapsed().as_secs_f64() * 1e6);
                }
            }
            scratch::put_bf16(ple_out);
            scratch::put_bf16(hyper); // PLE 分支把主残差流换成了 summed
            hyper = summed;
        }
    }

    // ── 注意力块 ──
    // GR 的**输出缓冲按层固定**（CUDA Graph 会把设备指针烘死，不能用共享 scratch 池）。
    let gbase = (layer as u64) * 8;
    // ★ P15-⑩（2026-10-17）：键**带 t**（`(1+t) << 40`）。根因：spec 下 t=1 草稿链与 t=7
    // 验证批交替，键只按 `layer*8` 时 t=7 会把 t=1 的 GR 图录死的输出缓冲（len=dim）
    // 以 `len=7*dim` 重新分配、旧块还给驱动 ⇒ t=1 图写出到野内存 = 静默数值错
    // （实测 spec+graph 接受率 3.9→0.17，罪魁=GR 图；二分开关 LINGXI_NO_GR_GRAPH）。
    let tkey = gbase.wrapping_add((1u64 + t as u64) << 40);
    let mut mixed_a = graph::take_fixed(k, tkey, 0, t * s.hidden)?;
    let mut injw_a = graph::take_fixed(k, tkey, 1, t * s.hc)?;
    let mut mixed_m = graph::take_fixed(k, tkey, 2, t * s.hidden)?;
    let mut injw_m = graph::take_fixed(k, tkey, 3, t * s.hc)?;
    // 诊断：`LINGXI_SKIP=gr` 把两个 GR 块整个跳过（输出直接置 0，不分配、kernel 数一致）。
    if skip_has("gr") {
        k.stream.memset_zeros(&mut mixed_a)?;
        k.stream.memset_zeros(&mut injw_a)?;
        k.stream.memset_zeros(&mut mixed_m)?;
        k.stream.memset_zeros(&mut injw_m)?;
    } else {
    crate::gr_prof::begin(k, t as u32, crate::gr_prof::Chain::Attn)?;
    gr::gated_residual_forward_into(
        k, &w.attn_gr, &hyper, &mut mixed_a, Some(&mut injw_a), t as u32,
        dim, s.hidden, s.hc, s.lowrank, s.eps, tkey + 1,
    )?;
    }
    // `LINGXI_KTS=1`：给 `[step]` 的四段插同步括号 ⇒ 变成**真 GPU 时间**（否则只是 CPU 发射时间）。
    if crate::flag("LINGXI_KTS") {
        k.stream.synchronize()?;
    }
    crate::mtp_prof::mark(k, 1)?; // 段 1 = attn_gr 结束（仅 MTP 哨兵层生效）
    crate::draft_prof::mark(k, 1)?; // 同上（`LINGXI_DRAFT_PROF`，正确边界版）
    let t_agr = t0.elapsed();
    if let Some(p) = ep {
        p.end(layer, 0, &k.stream);
        p.begin(layer, 1, &k.stream);
    }
    let attn_out = if skip_has("attn") {
        k.stream.alloc_zeros::<bf16>(t * s.hidden)?
    } else {
        // `LINGXI_ATTN_SYNC=1`：attn 段六分探针（流内事件，层末一次 sync；见 `attn_prof`）
        crate::attn_prof::info(
            t,
            match &states.attn {
                AttnState::Qsa(st) => st.len,
                AttnState::Gdn(_) => 0,
            },
        );
        crate::attn_prof::begin(
            k,
            layer,
            if matches!(w.attn, AttnWeights::Qsa { .. }) {
                crate::attn_prof::Kind::Qsa
            } else {
                crate::attn_prof::Kind::Gdn
            },
        )?;
        match (&w.attn, &mut states.attn) {
            (AttnWeights::Qsa { attn, indexer }, AttnState::Qsa(st)) => qsa::attention_step(
                k, attn, indexer, &mixed_a, env.cos, env.sin, env.cos_host, env.sin_host,
                env.k_layernorm_host, t, &s.attn, &s.indexer, st,
            )?,
            // GDN：把录像槽透传下去 —— `replay=false` 时录、`replay=true` 时从录像读并跳过投影
            // （见 [`layer_step_rec`] 与 [`crate::gdn::GdnRec`]）。QSA 层不需要（提交只改计数）。
            (AttnWeights::Gdn(g), AttnState::Gdn(st)) => {
                gdn::gdn_step_rec(k, g, &mixed_a, t, &s.gdn, st, rec)?
            }
            _ => unreachable!("层权重种类与状态种类必须一致"),
        }
    };
    if crate::flag("LINGXI_KTS") {
        k.stream.synchronize()?;
    }
    crate::mtp_prof::mark(k, 2)?; // 段 2 = attention 结束
    crate::draft_prof::mark(k, 2)?;
    let t_attn = t0.elapsed();
    if let Some(p) = ep {
        p.end(layer, 1, &k.stream);
        p.begin(layer, 2, &k.stream);
    }
    k.gr_write_inplace_bf16(&mut hyper, &attn_out, &injw_a, s.hc, s.hidden, t as u32)?;
    if layer == 0 || layer == 47 {
        scratch::probe_ptr(k, &format!("层{layer} mixed_a"), &mixed_a);
        scratch::probe_ptr(k, &format!("层{layer} attn_out"), &attn_out);
    }
    graph::give_fixed(k, tkey, 0, mixed_a);
    graph::give_fixed(k, tkey, 1, injw_a);
    scratch::put_bf16(attn_out);

    // ── MLP 块 ──
    if !skip_has("gr") {
        crate::gr_prof::begin(k, t as u32, crate::gr_prof::Chain::Mlp)?;
        gr::gated_residual_forward_into(
            k, &w.mlp_gr, &hyper, &mut mixed_m, Some(&mut injw_m), t as u32,
            dim, s.hidden, s.hc, s.lowrank, s.eps, tkey + 2,
        )?;
    }
    if crate::flag("LINGXI_KTS") {
        k.stream.synchronize()?;
    }
    crate::mtp_prof::mark(k, 3)?; // 段 3 = mlp_gr 结束（含 attn 输出的 gr_write）
    crate::draft_prof::mark(k, 3)?;
    let t_mgr = t0.elapsed();
    if let Some(p) = ep {
        p.end(layer, 2, &k.stream);
    }
    // `LINGXI_GR_PROF=1`：层末**一次** sync 读回两条 GR 链 + PLE 的逐核事件。
    crate::gr_prof::flush(k, layer)?;
    Ok(DenseStepOut { hyper, mixed_m, injw_m, t0, t_agr, t_attn, t_mgr })
}

/// **MoE 段**（evprof 段 3..4）：路由 + host 段 + 专家段。输出是"本卡半边"的部分和；
/// `ep = Some(..)` 时只算分配表里属于本卡的专家（见 `moe::sparse_moe_forward`）。
#[allow(clippy::too_many_arguments)]
pub fn layer_moe_partial(
    k: &Kernels,
    w: &LayerWeights,
    s: &LayerShape,
    mixed_m: &CudaSlice<bf16>,
    t: usize,
    env: &LayerEnv,
    layer: u32,
    ep: Option<moe::EpSlice<'_>>,
) -> Result<MoePartialOut> {
    if let Some(p) = k.evprof.as_ref() {
        // 段 3 = MoE 的"路由 + host 段"，段 4 = "专家段"（边界记在 `moe::sparse_moe_forward` 里）
        p.begin(layer, 3, &k.stream);
    }
    let moe_out = if skip_has("moe") {
        k.stream.alloc_zeros::<bf16>(t * s.hidden)?
    } else {
        let mo = moe::sparse_moe_forward(
            k, &w.moe, mixed_m, t, s.hidden, s.num_experts, s.top_k, s.inter, None,
            layer, env.experts, env.remote, ep,
        )?;
        if let Some(l) = mo.router_logits {
            scratch::put_bf16(l);
        }
        scratch::put_bf16(mo.shared);
        scratch::put_bf16(mo.expert_out);
        mo.out
    };
    if crate::flag("LINGXI_KTS") {
        k.stream.synchronize()?;
    }
    let t_moe = std::time::Instant::now();
    Ok(MoePartialOut { out: moe_out, t_moe })
}

/// **收尾段**（evprof 段 5）：把 MoE 输出写回残差流，并归还 MoE 的两个 GR 固定槽。
#[allow(clippy::too_many_arguments)]
pub fn layer_moe_apply(
    k: &Kernels,
    s: &LayerShape,
    mut hyper: CudaSlice<bf16>,
    moe_out: CudaSlice<bf16>,
    mixed_m: CudaSlice<bf16>,
    injw_m: CudaSlice<bf16>,
    t: usize,
    layer: u32,
) -> Result<CudaSlice<bf16>> {
    let tkey = (layer as u64) * 8 + ((1u64 + t as u64) << 40);
    if let Some(p) = k.evprof.as_ref() {
        p.end(layer, 4, &k.stream);
        p.begin(layer, 5, &k.stream);
    }
    k.gr_write_inplace_bf16(&mut hyper, &moe_out, &injw_m, s.hc, s.hidden, t as u32)?;
    // ★ P15-⑩：键必须与 take 侧一致（`tkey`）。漏 give 的话块会在函数末尾 drop 还给驱动，
    // 下轮 take 重新分配新地址 ⇒ 图（烘死地址）写出到野内存（spec 下 NaN/数值错的最后一环）。
    graph::give_fixed(k, tkey, 2, mixed_m);
    graph::give_fixed(k, tkey, 3, injw_m);
    scratch::put_bf16(moe_out);
    if let Some(p) = k.evprof.as_ref() {
        p.end(layer, 5, &k.stream);
    }
    if crate::flag("LINGXI_LAYER_NAN_CHK") {
        // 诊断（`LINGXI_LAYER_NAN_CHK=1`）：逐层核 hidden/hyper 是否有非有限值并报出行号。
        // 用途：定位"某行 NaN 从哪一层开始"（NaN 会让设备 top-k 的 ids 尾部留残渣，
        // 进而把后续 pair 核带到越界读）。每层一次 D2H，仅排障时开。
        let dim = s.hc * s.hidden;
        let v: Vec<bf16> = k.stream.clone_dtoh(&hyper)?;
        let mut rows_bad = Vec::new();
        for r in 0..t {
            if v[r * dim..(r + 1) * dim].iter().any(|x| !x.to_f32().is_finite()) {
                rows_bad.push(r);
            }
        }
        if !rows_bad.is_empty() {
            eprintln!("[nanchk] layer {layer} t={t} rows_bad={rows_bad:?}");
        }
    }
    Ok(hyper)
}

/// **MoE + 收尾**（evprof 段 3..5）：等价于 `layer_moe_partial` + `layer_moe_apply`。
/// 非 EP 路径用它，保持与拆分前同一调用序列/同一数学。
#[allow(clippy::too_many_arguments)]
pub fn layer_moe_finish(
    k: &Kernels,
    w: &LayerWeights,
    s: &LayerShape,
    hyper: CudaSlice<bf16>,
    mixed_m: CudaSlice<bf16>,
    injw_m: CudaSlice<bf16>,
    t: usize,
    env: &LayerEnv,
    layer: u32,
    ep: Option<moe::EpSlice<'_>>,
) -> Result<MoeFinishOut> {
    let part = layer_moe_partial(k, w, s, &mixed_m, t, env, layer, ep)?;
    crate::mtp_prof::mark(k, 4)?; // 段 4 = MoE（路由+shared+专家+归并）
    crate::draft_prof::mark(k, 4)?;
    let hyper = layer_moe_apply(k, s, hyper, part.out, mixed_m, injw_m, t, layer)?;
    crate::mtp_prof::mark(k, 5)?; // 段 5 = 收尾（MoE 输出的 gr_write）
    crate::draft_prof::mark(k, 5)?;
    Ok(MoeFinishOut { hyper, t_moe: part.t_moe })
}

/// 见 [`layer_step_rec`]。
fn layer_step_inner(
    k: &Kernels,
    w: &LayerWeights,
    s: &LayerShape,
    x: &CudaSlice<bf16>,
    t: usize,
    env: &LayerEnv,
    states: &mut LayerStates,
    layer: u32,
    rec: Option<crate::gdn::GdnRec<'_>>,
) -> Result<CudaSlice<bf16>> {
    let d = layer_dense_step(k, w, s, x, t, env, states, layer, rec)?;
    let mo = layer_moe_finish(k, w, s, d.hyper, d.mixed_m, d.injw_m, t, env, layer, None)?;
    if crate::flag("LINGXI_PROF") {
        eprintln!(
            "    [step] 层 {layer:2} 入口+attn_gr {:.1} attn {:.1} mlp_gr {:.1} moe {:.1} 合计 {:.1} µs",
            d.t_agr.as_secs_f64() * 1e6,
            (d.t_attn - d.t_agr).as_secs_f64() * 1e6,
            (d.t_mgr - d.t_attn).as_secs_f64() * 1e6,
            (mo.t_moe - d.t0).saturating_sub(d.t_mgr).as_secs_f64() * 1e6,
            (mo.t_moe - d.t0).as_secs_f64() * 1e6,
        );
    }
    Ok(mo.hyper)
}

/// 调试：把显存张量写成 `<dir>/<name>.f32`（仅 `LINGXI_LAYER_DEBUG_DIR` 打开时调用）。
fn dump(dir: &str, name: &str, k: &Kernels, dev: &CudaSlice<bf16>) -> Result<()> {
    let v: Vec<f32> = k.stream.clone_dtoh(dev)?.iter().map(|x| x.to_f32()).collect();
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(std::path::Path::new(dir).join(format!("{name}.f32")), bytes)?;
    Ok(())
}

fn dump_f32(dir: &str, name: &str, k: &Kernels, dev: &CudaSlice<f32>) -> Result<()> {
    let v: Vec<f32> = k.stream.clone_dtoh(dev)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(std::path::Path::new(dir).join(format!("{name}.f32")), bytes)?;
    Ok(())
}
