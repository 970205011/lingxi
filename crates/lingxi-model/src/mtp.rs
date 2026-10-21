//! MTP（多 token 预测）草稿层：权重加载 + **stem**。
//!
//! 规格与权威参考见 `docs/decode-speed-profile.md` §3.52：
//! - 前向顺序取自 **Strata 的 `src/core/mtp.cpp` L669–693**（跑过 `--mtp`、已验证）；
//! - 图结构对照 llama.cpp `qwen3next.cpp::graph_mtp`（L627–755）；
//! - **不照抄 ninfer**（用户明确：它的 MTP 接受率是错的）。
//!
//! 权重来自 `lingxi-bench mtp-pack` 抽出的独立小产物（生产产物里没有 `mtp/` 张量）。

use anyhow::{Context, Result};
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_artifact::Artifact;
use lingxi_kernels::Kernels;

use crate::gpu::upload_any_bf16;
use crate::scratch;

/// MTP 的几何 —— **全部从产物张量形状推出**，不写死任何数。
#[derive(Clone, Copy, Debug)]
pub struct MtpGeom {
    /// hidden（2560）
    pub n: usize,
    /// hyper-connection 的 stream 数（4）
    pub hc: usize,
    /// q 投影的输出维（`query_key_gate_value` 的前段）
    pub q_dim: usize,
    /// k/v 投影的输出维
    pub kv_dim: usize,
    /// MoE 的中间维（= 主干同值 640）
    pub inter: usize,
    /// 专家数（= 主干同值 512）
    pub num_experts: usize,
}

/// MTP 的权重（**只含 stem**）。
///
/// 层内/层尾的权重由装配层（`build_mtp_layer`）构造 —— 那里本来就是主干层权重的那一套。
/// ⚠️ 这里**故意不碰** `mtp/layer/mlp/routed_*`（3.35 + 1.68 GiB）：早先版本把它们也上传了，
/// 结果与装配层重复占 5 GiB 显存，prefill 到第 43 层就 OOM。几何只需要**形状**，不需要数据。
pub struct MtpWeights {
    pub geom: MtpGeom,
    /// stem · embedding 分支
    pub fc_embedding: CudaSlice<bf16>,   // [N, N]
    pub embedding_norm: CudaSlice<bf16>, // [N]
    /// stem · hidden 分支（输入是 HC*N 的**残差栈**）
    pub fc_hidden: CudaSlice<bf16>,      // [N, N]
    pub hidden_norm: CudaSlice<bf16>,    // [HC*N]
}

fn shape_of(art: &Artifact, name: &str) -> Result<Vec<u64>> {
    let obj = art.find(name).with_context(|| format!("MTP 产物缺少对象 {name}"))?;
    let (_, _, shape) = obj.tensor_parts().with_context(|| format!("{name} 不是 tensor"))?;
    Ok(shape.to_vec())
}

fn rank2(shape: &[u64], name: &str) -> Result<(usize, usize)> {
    if shape.len() != 2 {
        anyhow::bail!("{name} 需要 rank-2，得到 {shape:?}");
    }
    Ok((shape[0] as usize, shape[1] as usize))
}

/// 从 `mtp-pack` 产物里加载全部 MTP 权重。
pub fn load(k: &Kernels, art: &Artifact) -> Result<MtpWeights> {
    let up = |name: &str| -> Result<CudaSlice<bf16>> {
        let full = format!("mtp/{name}");
        upload_any_bf16(k, art, &full).with_context(|| format!("加载 {full} 失败"))
    };

    // ── 几何（从形状推，不写死） ──
    let sh_v = shape_of(art, "mtp/layer/attention/query_key_gate_value")?;
    let (qkv_rows, n) = rank2(&sh_v, "mtp/layer/attention/query_key_gate_value")?;
    let sh_out = shape_of(art, "mtp/layer/attention/output")?;
    let (_o_rows, q_dim) = rank2(&sh_out, "mtp/layer/attention/output")?;
    // `query_key_gate_value` 的排布是 [q(q_dim) | k | gate(q_dim) | v]，故行数 = 2*q_dim + 2*kv_dim。
    let kv_dim = (qkv_rows as usize)
        .checked_sub(2 * q_dim)
        .context("query_key_gate_value 行数 < 2*q_dim，形状不符合 QSA 契约")?
        / 2;
    let sh_norm = shape_of(art, "mtp/hidden_norm")?;
    let hc = (sh_norm[0] as usize) / n;
    anyhow::ensure!(hc >= 1 && hc * n == sh_norm[0] as usize, "hidden_norm 形状与 N 不整除");
    let sh_rg = shape_of(art, "mtp/layer/mlp/routed_gate_up")?;
    // 专家数 = `router_shared_gate` 的行数 − 1（最后一行是 `shared_expert_gate`）
    let sh_rtr = shape_of(art, "mtp/layer/mlp/router_shared_gate")?;
    let num_experts = (sh_rtr[0] as usize) - 1;
    // ⚠️ `routed_gate_up` 是 `[E*2I, H]` ⇒ **I = rows/(2E)**（不是 cols/2 —— 一开始写错成
    // `cols/2` 得到 1280，看着也"像对的"；用 `routed_down` 交叉校验把这类静默错误钉死）。
    anyhow::ensure!(
        sh_rg[0] as usize % (2 * num_experts) == 0,
        "routed_gate_up 行数 {} 不是 2E(= {}) 的倍数",
        sh_rg[0],
        2 * num_experts
    );
    let inter = (sh_rg[0] as usize) / (2 * num_experts);
    let sh_rd = shape_of(art, "mtp/layer/mlp/routed_down")?;
    anyhow::ensure!(
        sh_rd[0] as usize == num_experts * n && sh_rd[1] as usize == inter,
        "routed_down {sh_rd:?} 与 (E={num_experts}, H={n}, I={inter}) 不符"
    );
    let geom = MtpGeom { n, hc, q_dim, kv_dim, inter, num_experts };

    // ── 只上传 stem 的 4 个张量（layer/* 与 mixer/* 交给 build_mtp_layer，别重复占显存） ──
    let mut fc_embedding = up("fc_embedding")?;
    let mut fc_hidden = up("fc_hidden")?;
    // ⚠️ 诊断开关：`fc_embedding` 与 `fc_hidden` **形状完全相同（[N,N]）**，
    //    一旦产物里两者命名/顺序被搞反，症状正好是"能跑、量级正常、但全错"。
    //    `LINGXI_MTP_SWAP_FC=1` 交换一下，用来一刀裁掉这个可能。
    if crate::flag("LINGXI_MTP_SWAP_FC") {
        std::mem::swap(&mut fc_embedding, &mut fc_hidden);
        eprintln!("[mtp] ⚠️ LINGXI_MTP_SWAP_FC=1：已交换 fc_embedding / fc_hidden（诊断用）");
    }
    let embedding_norm = up("embedding_norm")?;
    let hidden_norm = up("hidden_norm")?;

    Ok(MtpWeights { geom, fc_embedding, embedding_norm, fc_hidden, hidden_norm })
}

/// **MTP stem**：`(Rin[T, HC*N], emb[T, N]) -> R[T, HC*N]`。
///
/// 逐字对应 §3.52（Strata `src/core/mtp.cpp` L669–693）：
/// ```text
/// en = RMSNorm_N(emb, embedding_norm)            // 整段 N 宽
/// e2 = fc_embedding @ en                         // [T, N]
/// hn = 逐 stream RMSNorm_N(Rin, hidden_norm)     // HC 片各 norm
/// h2 = fc_hidden @ hn                            // **每片各投一份**
/// R  = h2 + repeat(e2, HC)                       // **广播相加**（不是 concat、不是只加一路）
/// ```
pub fn stem(
    k: &Kernels,
    w: &MtpWeights,
    rin: &CudaSlice<bf16>,
    emb: &CudaSlice<bf16>,
    t: usize,
    eps: f32,
) -> Result<CudaSlice<bf16>> {
    let n = w.geom.n;
    let hc = w.geom.hc;
    let rows = t * hc; // 把 [T, HC, N] 当成 T*HC 行

    // embedding 分支（整段 N 宽的一次 norm ⇒ group == dim）
    let mut en = scratch::bf16(k, t * n)?;
    k.grouped_rmsnorm_bf16(emb, &w.embedding_norm, &mut en, t as u32, n, n, eps)?;
    let mut e2 = scratch::bf16(k, t * n)?;
    crate::gpu::proj_t1(k, &en, &w.fc_embedding, &mut e2, n, n, t)?;
    scratch::put_bf16(en);

    // hidden 分支：`pre_fc_norm_hidden` 是 **[HC*N]（10240）** 的 gamma，**两种 norm 分组**：
    //
    // ★★ 2026-10-16 **翻案**：默认改成 **逐 stream**（`LINGXI_MTP_HNORM=pooled` 可退回）。
    //   原注释写着"最初我按逐 stream 实现（group = N），实测接受率 0/64 —— 就是这里错了"，
    //   但**那次对照与 stem 的 `+1` gamma 修正同期**（`tools/golden/mtp_fix_stem_norm.py`）
    //   ⇒ 整个比较作废。重做的证据（交错 A/B，各跑 2 轮，n=200，`min_p=0` 全链）：
    //     | 模式 | 每轮接受 | tok/s | 主干验证 ms/轮 |
    //     | pooled | 3.47/6 | 38.6 / 37.2 | 83.7 / 86.6 |
    //     | **stream** | **3.67/6** | **41.0 / 41.0** | **81.6 / 81.6** |
    //   逐位条件接受率（位 0→5）：pooled 0.98/0.68/0.83/1.00/0.76/0.68；
    //                              **stream 0.98/0.79/0.79/0.96/0.72/0.78** ⇒ 位 1 是主要改善点。
    //   旁证：① Strata `src/core/mtp.cpp:686` 的注释明写逐 stream 才是
    //        **"as llama.cpp's qwen4exp MTP graph does"**；② 同机同模型的「顺子」
    //        开着 `--mtp-hnorm stream`（峰值 +3.7%）且 MTP 接受率 93~99%。
    //   ⚠️ 测量陷阱：第一次 A/B 是"pooled 先跑、stream 后跑"，stream 那轮温度上来了
    //      （主干验证 89→149 ms）⇒ 差点得出"stream 更慢 45%"的错结论。**A/B 必须交错。**
    let pooled = std::env::var("LINGXI_MTP_HNORM").map(|v| v == "pooled").unwrap_or(false);
    let mut hn = scratch::bf16(k, rows * n)?;
    if pooled {
        // 整段：一条 10240 的 RMS，gamma 全场乘一遍（Strata 的默认分支）
        k.grouped_rmsnorm_bf16(rin, &w.hidden_norm, &mut hn, t as u32, hc * n, hc * n, eps)?;
    } else {
        // 逐 stream：`[T, HC*N]` 当作 `[T*HC, N]` 的行，每行单独 norm；
        // `grouped_rmsnorm` 的 gamma 按**绝对列**索引（`gamma[gi*group + i]`）⇒ 正好等于
        // gamma 的第 gi 片 ⇒ 与 Strata 的 `native_qsa_rms_norm_grouped(N, HC, T*HC)` 同义。
        k.grouped_rmsnorm_bf16(rin, &w.hidden_norm, &mut hn, t as u32, hc * n, n, eps)?;
    }
    let mut h2 = scratch::bf16(k, rows * n)?;
    crate::gpu::proj_t1(k, &hn, &w.fc_hidden, &mut h2, n, n, rows)?;
    scratch::put_bf16(hn);

    // 广播相加
    let mut e2b = scratch::bf16(k, rows * n)?;
    k.hc_repeat_bf16(&e2, &mut e2b, t, n, hc)?;
    scratch::put_bf16(e2);
    let mut r = scratch::bf16(k, rows * n)?;
    k.add_bf16(&h2, &e2b, &mut r, rows * n)?;
    scratch::put_bf16(h2);
    scratch::put_bf16(e2b);
    Ok(r)
}
