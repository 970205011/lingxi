//! QSA 全注意力（layer ∈ {3,7,…,47}）。数学对齐 HF `Qwen4ExpTextAttention`：
//! ```text
//! qkgv = x @ qkgv_w^T                    # [T, 13312] = q(6144) | k(512) | gate(6144) | v(512)
//! q    = q_norm(q)   (每 head_dim 归一化)  k = k_norm(k)
//! q, k = rope(q, k)  (partial, rot_dim=64, NEOX)
//! o    = masked_attention(q, k, v, mask)  # mask = 因果 ∧ indexer 选块（加性）
//! out  = o_proj( (o * sigmoid(gate)) )
//! ```
//!
//! R3 拆分（2026-10-21）：原 1639 行单文件按职责分 5 模块——
//! - [`core_`]：权重/形状类型 + 能力开关 + `QsaState` 状态与生命周期；
//! - [`paging`]：host 锁页块池 / 驱逐 / 物化 / 按块检查点 + 沉取账本（**独占 host 池句柄**）；
//! - [`admit`]：`kv_admit` 行批选中准入（**独占选中缓冲**）；
//! - [`step`]：`attention_step`（GPU 增量路径：选块 / gather / 诊断）；
//! - [`oracle`]：host 参考实现（`indexer_mask`/`attention_forward`，对拍与降级用）。
//! 本文件只做模块声明与再导出——**对外 `qsa::` 路径全部不变**（调用方零改动）。

mod admit;
mod core_;
mod oracle;
mod paging;
mod step;

pub use core_::*;
pub use oracle::*;
pub use paging::*;
pub use step::*;


