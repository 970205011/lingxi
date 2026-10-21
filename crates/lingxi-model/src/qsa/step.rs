//! QSA GPU 增量路径 `attention_step`：投影/norm/rope（GPU）+ 选块（radix/融合/big_prefill）
//! + gather 稀疏注意力 + KV 追加。host 参考实现见 [`crate::qsa::oracle`]。

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

use super::oracle::build_indexer_mask;
use super::core_::{attn_gather_on, keep_radix_on, AttentionWeights, AttnCfg, IndexerCfg, IndexerWeights, QsaState};
use crate::gpu::{proj_q4_epi, proj_t1};
use crate::scratch;

/// 增量步进：`x[T, H] → out[T, H]`，`state` 就地追加 T 个 token 并前移。
///
/// 与 [`attention_forward`] + [`indexer_mask`] 的关系：数学相同，区别只在
/// ① q/k 的 RoPE 用**绝对位置**（`dev_cos`/`dev_sin` 是覆盖全序列的表，offset=`len`）；
/// ② k/v 追加进 cache，注意力在 `T_q=T, T_kv=len+T` 上做；
/// ③ indexer 在**全量 key** 上选块（`build_indexer_mask`，同一份真源）。
///
/// `dev_cos`/`dev_sin` 与 `cos_host`/`sin_host` 都需覆盖到 `len+T`（块起点 RoPE 用）。
#[allow(clippy::too_many_arguments)]
pub fn attention_step(
    k: &Kernels,
    w: &AttentionWeights,
    wi: &IndexerWeights,
    x: &CudaSlice<bf16>,
    dev_cos: &CudaSlice<bf16>,
    dev_sin: &CudaSlice<bf16>,
    cos_host: &[f32],
    sin_host: &[f32],
    k_layernorm_host: &[f32],
    t: usize,
    c: &AttnCfg,
    ci: &IndexerCfg,
    state: &mut QsaState,
) -> Result<CudaSlice<bf16>> {
    let q_dim = c.heads * c.head_dim;
    let kv_dim = c.kv_heads * c.head_dim;
    let fused = 2 * q_dim + 2 * kv_dim;
    let h = c.hidden;
    let scale = 1.0f32 / (c.head_dim as f32).sqrt();
    let len = state.len;
    let t_kv = len + t;
    // ㊿-P1 弹性 KV：容量不足先几何增长（冷路径；对数学逐位无影响），不再硬性 assert。
    if t_kv > state.cap {
        state.ensure_cap(k, t_kv)?;
    }
    // 层内细标记（只给 decode 用；prefill 调用太多会把日志淹没）
    let prof = crate::flag("LINGXI_QSA_PROF");
    let dtrace = prof && t <= 4;
    macro_rules! dmark {
        ($tag:expr) => {
            if dtrace {
                k.stream.synchronize()?;
                eprintln!("[qsa-dbg] t={t} len={len} step={}", $tag);
            }
        };
    }

    // 1) 融合投影（与整段同）
    let mut q = scratch::bf16(k, t * q_dim)?;
    let mut kraw = scratch::bf16(k, t * kv_dim)?;
    let mut gate = scratch::bf16(k, t * q_dim)?;
    let mut v = scratch::bf16(k, t * kv_dim)?;
    // four 段在同一块矩阵里连续：q | k | gate | v，用 row_off 直接切，避免拆成 4 个张量。
    // ⚠️ `slice()` 必须在非 W4 分支里做：W4 模式下 bf16 字段只是 1 元素占位，切了会 panic。
    if let Some(wq) = &w.query_key_gate_value_q4 {
        proj_q4_epi(k, x, wq, &mut q, q_dim, h, 0, t, 0, 0.0, 0.0)?;
        proj_q4_epi(k, x, wq, &mut kraw, kv_dim, h, q_dim, t, 0, 0.0, 0.0)?;
        proj_q4_epi(k, x, wq, &mut gate, q_dim, h, q_dim + kv_dim, t, 0, 0.0, 0.0)?;
        proj_q4_epi(k, x, wq, &mut v, kv_dim, h, 2 * q_dim + kv_dim, t, 0, 0.0, 0.0)?;
    } else {
        let q_w = w.query_key_gate_value.slice(0..q_dim * h);
        let k_w = w.query_key_gate_value.slice(q_dim * h..(q_dim + kv_dim) * h);
        let g_w = w.query_key_gate_value.slice((q_dim + kv_dim) * h..(2 * q_dim + kv_dim) * h);
        let v_w = w.query_key_gate_value.slice((2 * q_dim + kv_dim) * h..fused * h);
        proj_t1(k, x, &q_w, &mut q, q_dim, h, t)?;
        proj_t1(k, x, &k_w, &mut kraw, kv_dim, h, t)?;
        proj_t1(k, x, &g_w, &mut gate, q_dim, h, t)?;
        proj_t1(k, x, &v_w, &mut v, kv_dim, h, t)?;
    }

    // 2) 每头 RMSNorm + partial RoPE（用新 token 的绝对位置）
    let mut qn = scratch::bf16(k, t * q_dim)?;
    k.grouped_rmsnorm_bf16(&q, &w.query_norm, &mut qn, (t * c.heads) as u32, c.head_dim, c.head_dim, c.eps)?;
    scratch::put_bf16(q);
    let mut kn_new = scratch::bf16(k, t * kv_dim)?;
    k.grouped_rmsnorm_bf16(&kraw, &w.key_norm, &mut kn_new, (t * c.kv_heads) as u32, c.head_dim, c.head_dim, c.eps)?;
    scratch::put_bf16(kraw);
    if c.rot_dim > 0 {
        k.rope_partial_inplace_bf16(&mut qn, dev_cos, dev_sin, t, c.heads, c.head_dim, c.rot_dim, len)?;
        k.rope_partial_inplace_bf16(&mut kn_new, dev_cos, dev_sin, t, c.kv_heads, c.head_dim, c.rot_dim, len)?;
    }

    dmark!("proj");
    // P2 分页：写前把 [len, t_kv) 覆盖的块物化出 GPU 槽（本步写入 + gather 尾部读它）。
    state.kv_materialize_write(k, len, t)?;
    // 3) 追加进 cache（bf16 直拷，或 ㊶ E4M3 量化后写字节缓冲 + scale）
    if state.kn.is_some() {
        if state.paged {
            state.kv_append_paged(k, &kn_new, &v, len, t)?;
        } else {
            k.stream.memcpy_dtod(&kn_new, &mut state.kn.as_mut().unwrap().slice_mut(len * kv_dim..t_kv * kv_dim))?;
            k.stream.memcpy_dtod(&v, &mut state.v.as_mut().unwrap().slice_mut(len * kv_dim..t_kv * kv_dim))?;
        }
    } else {
        let nkh = c.kv_heads;
        let mut kqv = state.kn_q8.as_mut().unwrap().slice_mut(len * kv_dim..t_kv * kv_dim);
        let mut ksc = state.kn_sc.as_mut().unwrap().slice_mut(len * nkh..t_kv * nkh);
        k.quant_rows_e4m3_bf16(&kn_new, &mut kqv, &mut ksc, t, nkh, c.head_dim)?;
        let mut vqv = state.v_q8.as_mut().unwrap().slice_mut(len * kv_dim..t_kv * kv_dim);
        let mut vsc = state.v_sc.as_mut().unwrap().slice_mut(len * nkh..t_kv * nkh);
        k.quant_rows_e4m3_bf16(&v, &mut vqv, &mut vsc, t, nkh, c.head_dim)?;
    }
    scratch::put_bf16(kn_new);
    scratch::put_bf16(v);
    dmark!("kv_append");

    // 4) indexer：新 token 的 q（norm+rope）与 raw key（追加到 host 侧）
    let iq_dim = ci.heads * ci.head_dim;
    let iq_w = wi.index_qk_proj.slice(0..iq_dim * h);
    let ik_w = wi.index_qk_proj.slice(iq_dim * h..(iq_dim + ci.head_dim) * h);
    let mut iq = scratch::bf16(k, t * iq_dim)?;
    let mut rk_new = scratch::bf16(k, t * ci.head_dim)?;
    // ★ P15-⑨：t>1 时 x 预转 fp32，indexer 两个 bf16 投影共用一份（主投影已走 Q4 不受影响）
    match crate::gpu::prep_f32x(k, x, t, t * h, 2)? {
        Some(xf) => {
            crate::gpu::proj_f32x(k, &iq_w, &xf, &mut iq, iq_dim, h, t, 0, 0.0, 0.0)?;
            crate::gpu::proj_f32x(k, &ik_w, &xf, &mut rk_new, ci.head_dim, h, t, 0, 0.0, 0.0)?;
        }
        None => {
            proj_t1(k, x, &iq_w, &mut iq, iq_dim, h, t)?;
            proj_t1(k, x, &ik_w, &mut rk_new, ci.head_dim, h, t)?;
        }
    }
    dmark!("idx_proj");
    let mut iqn = scratch::bf16(k, t * iq_dim)?;
    k.grouped_rmsnorm_bf16(&iq, &wi.q_layernorm, &mut iqn, (t * ci.heads) as u32, ci.head_dim, ci.head_dim, ci.eps)?;
    scratch::put_bf16(iq);
    dmark!("idx_norm");
    if ci.rot_dim > 0 {
        k.rope_partial_inplace_bf16(&mut iqn, dev_cos, dev_sin, t, ci.heads, ci.head_dim, ci.rot_dim, len)?;
    }
    dmark!("idx_rope");
    // 4) indexer 选块：GPU 路径（增量池化 + 打分 + 名次 + 掩码，全同流无同步）；
    //    `LINGXI_QSA_HOST=1` 回退 host 参考对拍。
    let host_fallback = crate::flag("LINGXI_QSA_HOST");
    // nblocks_new / keep 提升到分支外：sparse 注意力在分支后也要用
    let nblocks_new = t_kv / ci.compress;
    // ㊸ 持久复用（grow-only）：不再每 token `alloc_zeros`。两个核都全量覆写 ⇒ 无需清零。
    //
    // ★ 选块实现按 t 分派（2026-10-18 实测）：
    //   t≥16（prefill chunk）走 radix（O(nblocks)/行、零 scores 缓冲，256K 省 537MB/层）；
    //   1<t<16 走旧 rank。
    //   t==1（decode）走**并行直方图**
    //   （先并行打分落盘，再网格并行做 3 遍直方图 + 1 遍填 keep）：t=1 时 radix 只有 1 个
    //   block 在跑（8K 97ms/token、30K 254ms/token），rank 又是 O(nblocks²)（256K 外推 22s/token）。
    // ★ 2026-10-18：prefill 的选块统一走**并行直方图**：
    //   ① 分数用 `qsa_indexer_scores_bf16` 一次算完（grid 并行 over (行, 块)，全 SM 打满）；
    //   ② 分数落一次性 scratch（t×nblocks×4B，池化复用，峰值一块）；
    //   ③ `qsa_keep_from_scores_bf16` 每行一个 block 做 3 遍直方图 + 填 keep（只读 4B/块）。
    //   32K 实测 select+fill **41ms/层**，比 radix 的 293~335ms 快 7~8×，也比 shared 融合核
    //   （68ms，每行串行扫全部块）更快。256K（nblocks=65536）同样适用 ⇒ 不再有 49K 边界。
    //   `LINGXI_QSA_FUSED_MAX=<n>` 可回退到 shared 融合核做 A/B（默认 0 = 不用它）。
    let fused_max: usize = std::env::var("LINGXI_QSA_FUSED_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let big_prefill = keep_radix_on()
        && t >= 16
        && nblocks_new > fused_max
        && std::env::var("LINGXI_QSA_BIGPAR").map(|v| v != "0").unwrap_or(true);
    let dec_par = keep_radix_on() && t == 1;
    let use_radix = keep_radix_on() && t >= 16;
    let idx_need = t * nblocks_new.max(1);
    if state.keep_buf.len() < idx_need {
        state.keep_buf = k.stream.alloc_zeros::<u8>(idx_need)?;
    }
    // 常驻 scores 只给 decode（t==1）与旧 rank 路；大 prefill 用一次性 scratch。
    if ((keep_radix_on() && t == 1) || !use_radix) && state.scores_buf.len() < idx_need {
        // 这两条路径都需要 t×nblocks 的分数缓冲（decode 只有几百 KB）。
        state.scores_buf = k.stream.alloc_zeros::<f32>(idx_need)?;
    }
    if use_radix && state.thr_key_buf.len() < t {
        state.thr_key_buf = k.stream.alloc_zeros::<u32>(t)?;
        state.cnt_gt_buf = k.stream.alloc_zeros::<i32>(t)?;
    }
    let p0 = std::time::Instant::now();
    let mut p_pool = std::time::Duration::ZERO;
    let mut p_sel = std::time::Duration::ZERO;
    let mut gather_ok = false;
    let mut mask_opt: Option<cudarc::driver::CudaSlice<f32>> = None;
    if host_fallback {
        let rk_new_host: Vec<f32> = k.stream.clone_dtoh(&rk_new)?.iter().map(|v| v.to_f32()).collect();
        state.raw_keys_host.extend_from_slice(&rk_new_host);
        let iq_host: Vec<f32> = k.stream.clone_dtoh(&iqn)?.iter().map(|v| v.to_f32()).collect();
        let mask_host = build_indexer_mask(
            &iq_host, &state.raw_keys_host, cos_host, sin_host, k_layernorm_host, t, t_kv, len, ci,
        );
        mask_opt = Some(k.stream.clone_htod(&mask_host)?);
    } else {
        // raw key 追加进设备缓存
        k.stream
            .memcpy_dtod(&rk_new, &mut state.rk_dev.slice_mut(len * ci.head_dim..t_kv * ci.head_dim))?;
        dmark!("idx_rk");
        // 增量池化新完成的块：mean → RMSNorm → RoPE（块起点绝对位置）。
        // ★ 批量版（2026-10-18）：一次 launch 处理全部新块。原来每块 4 个 launch，
        //    256K 下 65K 块 × 4 × 12 层 ≈ 300 万次 launch ≈ 10 分钟纯 driver 开销。
        let b0 = state.nblocks;
        let nb = nblocks_new.saturating_sub(b0);
        if nb > 0 {
            let need = nb * ci.head_dim;
            if state.pool_tmp.len() < need {
                state.pool_tmp = k.stream.alloc_zeros::<bf16>(need)?;
                state.pool_tmp2 = k.stream.alloc_zeros::<bf16>(need)?;
            }
            if dtrace {
                eprintln!(
                    "[qsa-dbg] pool_in nb={nb} b0={b0} nblk_new={nblocks_new} need={need} \
                     pooled={} pool_tmp={} pool_tmp2={} rk_dev={} cos={} rot_dim={}",
                    state.pooled.len(), state.pool_tmp.len(), state.pool_tmp2.len(),
                    state.rk_dev.len(), dev_cos.len(), ci.rot_dim
                );
            }
            k.qsa_pool_blocks_bf16(&state.rk_dev, &mut state.pool_tmp, b0, nb, ci.compress, ci.head_dim)?;
            dmark!("pool_k");
            k.grouped_rmsnorm_bf16(
                &state.pool_tmp, &wi.k_layernorm, &mut state.pool_tmp2,
                nb as u32, ci.head_dim, ci.head_dim, ci.eps,
            )?;
            dmark!("pool_norm");
            if ci.rot_dim > 0 {
                k.rope_partial_stride_bf16(
                    &mut state.pool_tmp2, dev_cos, dev_sin, nb, ci.head_dim, ci.rot_dim,
                    b0 * ci.compress, ci.compress,
                )?;
            }
            dmark!("pool_rope");
            k.stream.memcpy_dtod(
                &state.pool_tmp2.slice(0..need),
                &mut state.pooled.slice_mut(b0 * ci.head_dim..nblocks_new * ci.head_dim),
            )?;
            dmark!("pool_copy");
        }
        state.nblocks = nblocks_new;
        if prof {
            k.stream.synchronize()?;
        }
        p_pool = p0.elapsed();
        dmark!("pool");

        if nblocks_new > 0 {
            let topk = ci.budget / ci.compress;
            let sqrt_hd = (ci.head_dim as f32).sqrt();
            if dec_par {
                // 并行：1 遍打分落盘 + 3 遍网格并行直方图（11+11+10 位）+ 1 遍填 keep。
                k.qsa_indexer_scores_bf16(
                    &iqn, &state.pooled, &mut state.scores_buf, t, nblocks_new, ci.heads, ci.head_dim, sqrt_hd,
                )?;
                let mut bin1 = 0i32;
                let mut cum1 = 0i32;
                let mut bin2 = 0i32;
                let mut cum2 = 0i32;
                for stage in 0..3 {
                    k.stream.memset_zeros(&mut state.hist_buf)?;
                    let (shift, bmask, fmask, fval, nbins, need) = match stage {
                        0 => (21u32, 2047u32, 0u32, 0u32, 2048usize, topk as i32),
                        1 => (10u32, 2047u32, 0xFFE0_0000u32, (bin1 as u32) << 21, 2048usize,
                              topk as i32 - cum1),
                        _ => (0u32, 1023u32, 0xFFFF_FC00u32,
                              ((bin1 as u32) << 21) | ((bin2 as u32) << 10), 1024usize,
                              topk as i32 - cum1 - cum2),
                    };
                    k.qsa_hist_f32(&state.scores_buf, &mut state.hist_buf, nblocks_new, shift, bmask, fmask, fval)?;
                    {
                        let mut bc = state.bin_buf.slice_mut(0..2);
                        k.qsa_bin_finalize(&state.hist_buf, nbins, need, &mut bc)?;
                    }
                    let bc: Vec<i32> = k.stream.clone_dtoh(&state.bin_buf)?.to_vec();
                    match stage {
                        0 => { bin1 = bc[0]; cum1 = bc[1]; }
                        1 => { bin2 = bc[0]; cum2 = bc[1]; }
                        _ => {
                            let hi22 = ((bin1 as u32) << 11) | (bin2 as u32);
                            let thr = (hi22 << 10) | (bc[0] as u32);
                            let need_eq = topk as i32 - cum1 - cum2 - bc[1];
                            k.qsa_keep_fill_scores(
                                &state.scores_buf, &mut state.keep_buf, nblocks_new, topk, thr, need_eq,
                            )?;
                        }
                    }
                }
            } else if big_prefill {
                // 大 nblocks（≈>49K 上下文）：1 遍并行打分落一次性 scratch + 一个
                // 每行 block 的"只读分数"选块核（3 直方图 + 填 keep）。
                // 分数来自与 radix 完全相同的 `qsa_score1` 公式 ⇒ 逐位一致。
                let mut s = crate::scratch::f32(k, idx_need)?;
                k.qsa_indexer_scores_bf16(
                    &iqn, &state.pooled, &mut s, t, nblocks_new, ci.heads, ci.head_dim, sqrt_hd,
                )?;
                k.qsa_keep_from_scores_bf16(&s, &mut state.keep_buf, t, nblocks_new, topk)?;
                crate::scratch::put_f32(s);
            } else if use_radix {
                // ★ 2026-10-18：nblocks 不大时走**融合选块**（一次打分 + shared 直方图），
                //   把旧路径的 4 遍打分（3 遍直方图 + 1 遍填 keep）压成 1 遍。
                //   32K（nblocks≈8184）实测 select+fill 300ms → 见 docs/BACKLOG §3.3 R。
                if nblocks_new <= fused_max {
                    k.qsa_keep_fused_bf16(
                        &iqn, &state.pooled, &mut state.keep_buf,
                        t, nblocks_new, ci.heads, ci.head_dim, sqrt_hd, topk,
                    )?;
                } else {
                    k.qsa_keep_select_bf16(
                        &iqn, &state.pooled, &mut state.thr_key_buf, &mut state.cnt_gt_buf,
                        t, nblocks_new, ci.heads, ci.head_dim, sqrt_hd, topk,
                    )?;
                    k.qsa_keep_fill_bf16(
                        &iqn, &state.pooled, &state.thr_key_buf, &state.cnt_gt_buf, &mut state.keep_buf,
                        t, nblocks_new, ci.heads, ci.head_dim, sqrt_hd, topk,
                    )?;
                }
            } else {
                k.qsa_indexer_scores_bf16(
                    &iqn, &state.pooled, &mut state.scores_buf, t, nblocks_new, ci.heads, ci.head_dim, sqrt_hd,
                )?;
                k.qsa_indexer_rank_f32(&state.scores_buf, &mut state.keep_buf, t, nblocks_new, topk)?;
            }
            if prof {
                k.stream.synchronize()?;
                p_sel = p0.elapsed() - p_pool;
            }
            dmark!("select");
        }
        // 选块列表（gather 路径）：把 keep 压成升序块号。bf16 KV 全走（prefill + decode）——
        // decode 也必须有：旧"原地跳"仍逐 key 扫全量 Tk，30K 上下文实测 decode 边际成本
        // ~4.8µs/上下文 token（256K 外推 >1s/token）；gather 后与上下文长度解耦。
        gather_ok = attn_gather_on()
            && state.kn.is_some()
            && ci.compress.is_power_of_two()
            && c.head_dim == 256
            && nblocks_new > 0;
        // P2：分页模式（Stage 2 起生效）硬前提——raw KV 只有 gather 路径按块读，
        // dense/masked 读全量行与分页互斥；K8V4 档无 gather 亦互斥。提前炸，不留脏值。
        if crate::flag("LINGXI_QSA_PAGED") {
            anyhow::ensure!(
                gather_ok,
                "LINGXI_QSA_PAGED 要求 gather 路径（LINGXI_ATTN_GATHER on + bf16 KV + \
                 2 的幂 compress + head_dim=256），当前配置不兼容"
            );
        }
        if gather_ok {
            let topk = ci.budget / ci.compress;
            let need = t * topk;
            if state.sel_buf.len() < need {
                state.sel_buf = k.stream.alloc_zeros::<i32>(need)?;
            }
            if state.sel_cnt_buf.len() < t {
                state.sel_cnt_buf = k.stream.alloc_zeros::<i32>(t)?;
            }
            k.qsa_sel_build(&state.keep_buf, &mut state.sel_buf, &mut state.sel_cnt_buf, t, nblocks_new, topk)?;
            dmark!("sel");
            // P2 分页准入挪到 gather 处按行批做（prefill 整步并集 ≈ 上下文，批并集 ~topk+扩散）。
        }
        // 掩码延迟到 attention 处按需构建（sparse 路径完全不需要 t×t_kv 掩码）
    }

    // 5) 注意力：T_q = t、T_kv = t_kv（cache 行主序连续，直接传整块 + tk 限定行数）
    let mut attn = scratch::bf16(k, t * q_dim)?;
    // ★ 跳块稀疏：keep 非空且 compress 为 2 的幂时走 attn_sparse（免 t×t_kv 掩码
    // 分配+填充，未选中块按 dense 精确语义跳过）——对拍开关 LINGXI_QSA_DENSE=1 回旧路径。
    let sparse = !host_fallback
        && nblocks_new > 0
        && ci.compress.is_power_of_two()
        && !crate::flag("LINGXI_QSA_DENSE");
    if sparse && gather_ok {
        let topk = ci.budget / ci.compress;
        // ★ 2026-10-18：同一 kv-group 的 4 个 q-head 合并进一个 block（K/V 流量 ÷4）。
        //   本模型 heads=24 / kv_heads=2 ⇒ 12 个 q-head 共用一份 K/V，旧版按 (row,head)
        //   各读一遍（实测 49GB/层 ≈ 816GB/s 打满 HBM）。逐位一致（每 head 的归约次序不变）。
        let ctok = 1usize << ci.compress.trailing_zeros();
        let span = topk * ctok + ctok;
        let nh = if std::env::var("LINGXI_ATTN_GATHER_H").map(|v| v != "0").unwrap_or(true)
            && c.heads % 4 == 0
            && c.kv_heads > 0
            && (c.heads / c.kv_heads) % 4 == 0
            && span * 4 * 4 <= 48 * 1024
        {
            4
        } else {
            1
        };
        // P2 分页 + prefill：按行批 "admit → gather" 成对发射——每批只保自己的选中
        // 并集（~topk+批内扩散）常驻，窗口 ≥ 批并集即可；整步并集 ≈ 上下文（不可驻留）。
        // decode（t=1）与非分页：单批（行为同旧）。
        let batch = if state.paged && t > 1 {
            std::env::var("LINGXI_QSA_ADMIT_BATCH")
                .ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0)
                .unwrap_or(64)
        } else {
            t
        };
        let mut r0 = 0usize;
        while r0 < t {
            let bn = (t - r0).min(batch);
            // （clone 是 Arc 级浅拷贝，只为避免 &mut state 与字段借用打架）
            let sel_ref = state.sel_buf.clone();
            let cnt_ref = state.sel_cnt_buf.clone();
            state.kv_admit(k, &sel_ref, &cnt_ref, r0, bn, topk, len, t)?;
            k.attn_gather_bf16(
                &qn.slice(r0 * q_dim..(r0 + bn) * q_dim),
                &state.kn.as_ref().unwrap().slice(..),
                &state.v.as_ref().unwrap().slice(..),
                &state.blk_tab.slice(..),
                &state.sel_buf.slice(r0 * topk..(r0 + bn) * topk),
                &state.sel_cnt_buf.slice(r0..r0 + bn),
                &mut attn.slice_mut(r0 * q_dim..(r0 + bn) * q_dim),
                bn, c.heads, c.kv_heads, c.head_dim, scale,
                len + r0, nblocks_new, ci.compress.trailing_zeros() as usize, topk, nh,
            )?;
            r0 += bn;
        }
    } else if sparse {
        match (&state.kn, &state.v) {
            (Some(kn), Some(v)) => {
                k.attn_sparse_bf16(
                    &qn, kn, v, &state.keep_buf, len, nblocks_new,
                    ci.compress.trailing_zeros(), &mut attn, c.heads, c.kv_heads, t, t_kv,
                    c.head_dim, scale,
                )?;
            }
            _ => {
                // ㊶ 量化档：bf16 形参传 1 元素占位（内核 q8 分支从不读），KV 读字节缓冲
                k.attn_sparse_kvq8_bf16(
                    &qn, &state.stub_kn, &state.stub_v, &state.keep_buf, len, nblocks_new,
                    ci.compress.trailing_zeros(), &mut attn, c.heads, c.kv_heads, t, t_kv,
                    c.head_dim, scale,
                    state.kn_q8.as_ref().unwrap(), state.kn_sc.as_ref().unwrap(),
                    state.v_q8.as_ref().unwrap(), state.v_sc.as_ref().unwrap(),
                )?;
            }
        }
    } else if host_fallback {
        // host 参考路径：掩码来自 host build_indexer_mask（keep 在此分支未算，不能用）
        let mask = mask_opt.as_ref().expect("host_fallback 分支必已构建 mask");
        match (&state.kn, &state.v) {
            (Some(kn), Some(v)) => {
                k.attn_masked_bf16(&qn, kn, v, mask, &mut attn, c.heads, c.kv_heads, t, t_kv, c.head_dim, scale)?;
            }
            _ => {
                k.attn_masked_kvq8_bf16(
                    &qn, &state.stub_kn, &state.stub_v, mask, &mut attn, c.heads, c.kv_heads,
                    t, t_kv, c.head_dim, scale,
                    state.kn_q8.as_ref().unwrap(), state.kn_sc.as_ref().unwrap(),
                    state.v_q8.as_ref().unwrap(), state.v_sc.as_ref().unwrap(),
                )?;
            }
        }
    } else {
        let mut m = k.stream.alloc_zeros::<f32>(t * t_kv)?;
        k.qsa_mask_fill_f32(&state.keep_buf, &mut m, t, t_kv, nblocks_new, ci.compress, len, f32::MIN)?;
        match (&state.kn, &state.v) {
            (Some(kn), Some(v)) => {
                k.attn_masked_bf16(&qn, kn, v, &m, &mut attn, c.heads, c.kv_heads, t, t_kv, c.head_dim, scale)?;
            }
            _ => {
                k.attn_masked_kvq8_bf16(
                    &qn, &state.stub_kn, &state.stub_v, &m, &mut attn, c.heads, c.kv_heads,
                    t, t_kv, c.head_dim, scale,
                    state.kn_q8.as_ref().unwrap(), state.kn_sc.as_ref().unwrap(),
                    state.v_q8.as_ref().unwrap(), state.v_sc.as_ref().unwrap(),
                )?;
            }
        }
    }
    dmark!("attn");
    // [判决性探针] `LINGXI_QSA_PAGED_NAN=1`：gather 后同步回读，分别统计
    // 输入（kn_new/v）与输出（attn）的 NaN——钉死第一层翻车的调用序号与方向。
    if crate::flag("LINGXI_QSA_PAGED_NAN") && state.paged {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NAN_N: AtomicU64 = AtomicU64::new(0);
        let n = NAN_N.fetch_add(1, Ordering::Relaxed);
        k.stream.synchronize()?;
        let bad = |h: &[bf16]| h.iter().filter(|v| v.is_nan()).count();
        if let Ok(a) = scratch::download_bf16(k, &attn) {
            let an = bad(&a);
            if an > 0 {
                use cudarc::driver::DevicePtr;
                let mut c0 = vec![0i32; t.min(8)];
                {
                    let (p, _g) = state.sel_cnt_buf.device_ptr(&k.stream);
                    unsafe { cudarc::driver::result::memcpy_dtoh_sync::<i32>(&mut c0, p).ok() };
                }
                let mut tab0 = vec![0i32; 8];
                {
                    let (p, _g) = state.blk_tab.device_ptr(&k.stream);
                    unsafe { cudarc::driver::result::memcpy_dtoh_sync::<i32>(&mut tab0, p).ok() };
                }
                let mut sel0 = vec![0i32; 8];
                {
                    let (p, _g) = state.sel_buf.device_ptr(&k.stream);
                    unsafe { cudarc::driver::result::memcpy_dtoh_sync::<i32>(&mut sel0, p).ok() };
                }
                let host_min = state.tab_host.iter().take(16).cloned().collect::<Vec<_>>();
                eprintln!(
                    "[paged-nan] 调用#{n} t={t} len={len} attn NaN={an}/{} sel_cnt[:8]={c0:?} sel[0][:8]={sel0:?} dev_tab[:8]={tab0:?} host_tab[:16]={host_min:?}",
                    a.len(),
                );
            }
        }
    }
    if prof {
        k.stream.synchronize()?;
        use std::sync::atomic::{AtomicU32, Ordering};
        static QP_N: AtomicU32 = AtomicU32::new(0);
        // decode（t≤4）全打；prefill 只打前 40 次，避免淹没日志
        // `LINGXI_QSA_PROF_ALL=1` 时不限量（用于按上下文长度看 pool/select/attn 的伸缩）
        static QP_LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let lim = *QP_LIMIT.get_or_init(|| {
            if crate::flag("LINGXI_QSA_PROF_ALL") { u32::MAX } else { 40 }
        });
        if t <= 4 || QP_N.fetch_add(1, Ordering::Relaxed) < lim {
            eprintln!(
                "[qsa-prof] t={t} len={len} nblocks={nblocks_new}  pool {:.1}ms  select+fill {:.1}ms  attn {:.1}ms",
                p_pool.as_secs_f64() * 1e3,
                p_sel.as_secs_f64() * 1e3,
                (p0.elapsed() - p_pool - p_sel).as_secs_f64() * 1e3
            );
        }
    }
    scratch::put_bf16(qn);
    scratch::put_bf16(iqn);
    scratch::put_bf16(rk_new);

    // 6) 门控 + o_proj
    k.sigmoid_scale_mul_inplace_bf16(&mut gate, 1.0, 1.0, t * q_dim)?;
    let mut gated = scratch::bf16(k, t * q_dim)?;
    k.mul_bf16(&attn, &gate, &mut gated, t * q_dim)?;
    scratch::put_bf16(gate);
    scratch::put_bf16(attn);
    let mut out = scratch::bf16(k, t * h)?;
    if let Some(wq) = &w.output_q4 {
        proj_q4_epi(k, &gated, wq, &mut out, h, q_dim, 0, t, 0, 0.0, 0.0)?;
    } else {
        proj_t1(k, &gated, &w.output, &mut out, h, q_dim, t)?;
    }
    scratch::put_bf16(gated);
    dmark!("out");

    state.len = t_kv;
    Ok(out)
}
