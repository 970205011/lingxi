//! QSA host 参考实现（oracle）：`indexer_mask`/`build_indexer_mask`/`attention_forward`。
//! 纯 host、可逐位对拍；`LINGXI_QSA_HOST=1` 等降级/对拍路径用。GPU 增量路径见 [`crate::qsa::step`]。

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

use super::core_::{AttnCfg, AttentionWeights, IndexerCfg, IndexerWeights};
use crate::gpu::proj_t1;  // oracle 用

fn to_bf16(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn rms_norm_f32(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(gamma).map(|(v, g)| to_bf16(v * r * g)).collect()
}

/// 部分 RoPE（NEOX，就地作用于 f32 切片的前 `rot_dim` 维）。
fn rope_partial_f32(x: &mut [f32], cos: &[f32], sin: &[f32], head_dim: usize, rot_dim: usize) {
    let half = rot_dim / 2;
    for h in (0..x.len()).step_by(head_dim) {
        for d in 0..half {
            let c0 = cos[d];
            let s0 = sin[d];
            let c1 = cos[d + half];
            let s1 = sin[d + half];
            let (x0, x1) = (x[h + d], x[h + d + half]);
            x[h + d] = to_bf16(x0 * c0 - x1 * s0);
            x[h + d + half] = to_bf16(x1 * c1 + x0 * s1);
        }
    }
}

/// 生成**组合掩码** `[T, T]`（f32 加性）：因果掩码 + indexer 选块掩码。
///
/// 数学对齐 HF `Qwen4ExpTextQSAIndexer.forward`。选块（打分/topk）在 host 上做，
/// 投影/norm/rope 在 GPU 上做 —— 短序列下这是最省事且可逐位对齐的形态。
#[allow(clippy::too_many_arguments)]
pub fn indexer_mask(
    k: &Kernels,
    w: &IndexerWeights,
    x: &CudaSlice<bf16>,
    dev_cos: &CudaSlice<bf16>,
    dev_sin: &CudaSlice<bf16>,
    cos_host: &[f32],
    sin_host: &[f32],
    // 因果掩码（整段路径下由外部给定；增量路径由绝对位置自行推出，故此处保留形参仅为签名稳定）
    _causal_host: &[f32],
    k_layernorm_host: &[f32],
    t: usize,
    hidden: usize,
    c: &IndexerCfg,
) -> Result<CudaSlice<f32>> {
    let q_dim = c.heads * c.head_dim;
    // 1) 投影：q 段与 k 段各一次 gemm（按权重行切，避免行内切分）
    let q_w = w.index_qk_proj.slice(0..q_dim * hidden);
    let k_w = w.index_qk_proj.slice(q_dim * hidden..(q_dim + c.head_dim) * hidden);
    let mut q = k.stream.alloc_zeros::<bf16>(t * q_dim)?;
    proj_t1(k, x, &q_w, &mut q, q_dim, hidden, t)?;
    let mut raw_keys = k.stream.alloc_zeros::<bf16>(t * c.head_dim)?;
    proj_t1(k, x, &k_w, &mut raw_keys, c.head_dim, hidden, t)?;

    // 2) q 的每头归一化 + rope
    let mut qn = k.stream.alloc_zeros::<bf16>(t * q_dim)?;
    k.grouped_rmsnorm_bf16(&q, &w.q_layernorm, &mut qn, (t * c.heads) as u32, c.head_dim, c.head_dim, c.eps)?;
    if c.rot_dim > 0 {
        k.rope_partial_inplace_bf16(&mut qn, dev_cos, dev_sin, t, c.heads, c.head_dim, c.rot_dim, 0)?;
    }

    // 3) 取回 host（很小）
    let q_host: Vec<f32> = k.stream.clone_dtoh(&qn)?.iter().map(|v| v.to_f32()).collect();
    let rk_host: Vec<f32> = k.stream.clone_dtoh(&raw_keys)?.iter().map(|v| v.to_f32()).collect();

    // 4) host 侧选块（与增量路径共用 `build_indexer_mask`；整段时 len=0、key 数 = t）
    let combined = build_indexer_mask(
        &q_host, &rk_host, cos_host, sin_host, k_layernorm_host, t, t, 0, c,
    );

    Ok(k.stream.clone_htod(&combined)?)
}

/// 组合掩码构造（**唯一真源**）：`t_q` 行 query（绝对位置 `len..len+t_q`）对 `t_kv` 个 key。
///
/// 因果可见集恒为前缀 `0..=len+i`；indexer 在其上按 `compress` 切块、池化、打分、top-k 选块；
/// 未被选中的可见项置 `f32::MIN`（加性掩码）。尾部不足一块的 token 始终可见（与 HF 一致）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_indexer_mask(
    q_host: &[f32],
    rk_host: &[f32],
    cos_host: &[f32],
    sin_host: &[f32],
    k_layernorm_host: &[f32],
    t_q: usize,
    t_kv: usize,
    len: usize,
    c: &IndexerCfg,
) -> Vec<f32> {
    let hd = c.head_dim;
    let block_topk = c.budget / c.compress;
    let sqrt_hd = (hd as f32).sqrt();
    let min_dtype = f32::MIN;
    let mut combined = vec![0f32; t_q * t_kv];

    for i in 0..t_q {
        let vis = len + i + 1; // 可见前缀长度（含自身）
        for j in 0..t_kv {
            combined[i * t_kv + j] = if j < vis { 0.0 } else { min_dtype };
        }
        let nblocks = vis / c.compress;
        let mut keep = vec![false; t_kv];
        if nblocks > 0 {
            // pooled = mean over block → bf16 → k_layernorm → rope（块起点位置）
            let mut pooled = vec![0f32; nblocks * hd];
            for b in 0..nblocks {
                for d in 0..hd {
                    let mut acc = 0f32;
                    for j in 0..c.compress {
                        acc += rk_host[(b * c.compress + j) * hd + d];
                    }
                    pooled[b * hd + d] = to_bf16(acc / c.compress as f32);
                }
            }
            for b in 0..nblocks {
                let normed = rms_norm_f32(&pooled[b * hd..(b + 1) * hd], k_layernorm_host, c.eps);
                pooled[b * hd..(b + 1) * hd].copy_from_slice(&normed);
            }
            for b in 0..nblocks {
                let pos = b * c.compress;
                let cos = &cos_host[pos * c.rot_dim..(pos + 1) * c.rot_dim];
                let sin = &sin_host[pos * c.rot_dim..(pos + 1) * c.rot_dim];
                let mut one = pooled[b * hd..(b + 1) * hd].to_vec();
                rope_partial_f32(&mut one, cos, sin, hd, c.rot_dim);
                pooled[b * hd..(b + 1) * hd].copy_from_slice(&one);
            }
            // scores = Σ_h relu(q_i,h · pooled_b) / sqrt(head_dim)
            let scores: Vec<f32> = (0..nblocks)
                .map(|b| {
                    let mut s = 0f32;
                    for h in 0..c.heads {
                        let mut dot = 0f32;
                        for d in 0..hd {
                            dot += q_host[(i * c.heads + h) * hd + d] * pooled[b * hd + d];
                        }
                        s += dot.max(0.0);
                    }
                    s / sqrt_hd
                })
                .collect();
            let take = block_topk.min(nblocks);
            let mut order: Vec<usize> = (0..nblocks).collect();
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            for &b in &order[..take] {
                for j in b * c.compress..(b + 1) * c.compress {
                    keep[j] = true;
                }
            }
        }
        // 尾部不完整块始终可见
        for j in nblocks * c.compress..vis {
            keep[j] = true;
        }
        for j in 0..t_kv {
            if !keep[j] {
                combined[i * t_kv + j] += min_dtype;
            }
        }
    }
    combined
}

pub fn attention_forward(
    k: &Kernels,
    w: &AttentionWeights,
    x: &CudaSlice<bf16>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    mask: &CudaSlice<f32>,
    t: usize,
    c: &AttnCfg,
) -> Result<CudaSlice<bf16>> {
    let q_dim = c.heads * c.head_dim;
    let kv_dim = c.kv_heads * c.head_dim;
    let fused = 2 * q_dim + 2 * kv_dim;
    let h = c.hidden;
    let scale = 1.0f32 / (c.head_dim as f32).sqrt();

    // 1) 融合投影：按权重行切 4 段，各做一次 gemm
    let q_w = w.query_key_gate_value.slice(0..q_dim * h);
    let k_w = w.query_key_gate_value.slice(q_dim * h..(q_dim + kv_dim) * h);
    let g_w = w.query_key_gate_value.slice((q_dim + kv_dim) * h..(2 * q_dim + kv_dim) * h);
    let v_w = w.query_key_gate_value.slice((2 * q_dim + kv_dim) * h..fused * h);

    let mut q = k.stream.alloc_zeros::<bf16>(t * q_dim)?;
    proj_t1(k, x, &q_w, &mut q, q_dim, h, t)?;
    let mut kraw = k.stream.alloc_zeros::<bf16>(t * kv_dim)?;
    proj_t1(k, x, &k_w, &mut kraw, kv_dim, h, t)?;
    let mut gate = k.stream.alloc_zeros::<bf16>(t * q_dim)?;
    proj_t1(k, x, &g_w, &mut gate, q_dim, h, t)?;
    let mut v = k.stream.alloc_zeros::<bf16>(t * kv_dim)?;
    proj_t1(k, x, &v_w, &mut v, kv_dim, h, t)?;

    // 2) 每头 RMSNorm（gamma 已烘焙）
    let mut qn = k.stream.alloc_zeros::<bf16>(t * q_dim)?;
    k.grouped_rmsnorm_bf16(&q, &w.query_norm, &mut qn, (t * c.heads) as u32, c.head_dim, c.head_dim, c.eps)?;
    let mut kn = k.stream.alloc_zeros::<bf16>(t * kv_dim)?;
    k.grouped_rmsnorm_bf16(&kraw, &w.key_norm, &mut kn, (t * c.kv_heads) as u32, c.head_dim, c.head_dim, c.eps)?;

    // 3) partial RoPE
    if c.rot_dim > 0 {
        k.rope_partial_inplace_bf16(&mut qn, cos, sin, t, c.heads, c.head_dim, c.rot_dim, 0)?;
        k.rope_partial_inplace_bf16(&mut kn, cos, sin, t, c.kv_heads, c.head_dim, c.rot_dim, 0)?;
    }

    // 4) 带加性掩码的注意力（GQA 在 kernel 内做头映射）
    let mut attn = k.stream.alloc_zeros::<bf16>(t * q_dim)?;
    k.attn_masked_bf16(&qn, &kn, &v, mask, &mut attn, c.heads, c.kv_heads, t, t, c.head_dim, scale)?;

    // 5) 门控 + o_proj
    k.sigmoid_scale_mul_inplace_bf16(&mut gate, 1.0, 1.0, t * q_dim)?;
    let mut gated = k.stream.alloc_zeros::<bf16>(t * q_dim)?;
    k.mul_bf16(&attn, &gate, &mut gated, t * q_dim)?;
    let mut out = k.stream.alloc_zeros::<bf16>(t * h)?;
    proj_t1(k, &gated, &w.output, &mut out, h, q_dim, t)?;
    Ok(out)
}
