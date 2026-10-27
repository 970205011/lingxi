//! `lingxi-driver` — L3 调度层：slot 池 · 准入 · 静态批 decode 循环 · MTP · 前缀共享 · 取消。
//!
//! 职责边界（见 `ARCHITECTURE.md`）：**不做 HTTP、不做计算。**
//! 计算全部在 `lingxi-model`（同一份 `layer_step` 既跑 prefill 也跑 decode）；
//! 本 crate 只编排：状态在谁手里、每步喂什么、何时停。
//!
//! 当前实现：**单序列贪心**（batch=1，无采样/MTP/前缀共享；后续在 [`Slot`] 上扩展）。

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_ids::{SessionId, SlotId};
use lingxi_kernels::Kernels;
use lingxi_model::layer::LayerShape;
use lingxi_model::moe::ExpertSource;
use lingxi_model::trunk::{LoadedLayer, TrunkWeights};

pub mod kv_budget;
pub use kv_budget::KvBudget;

/// `LINGXI_FIXED_IDS="13,561,6511,..."`：decode 时强制喂固定 token 序列（诊断用，见
/// [`Sequence::decode_token`]）。设了它之后，所有配置处理的输入完全相同，A/B 才可信。
fn fixed_seq() -> &'static [i64] {
    static V: std::sync::OnceLock<Vec<i64>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        std::env::var("LINGXI_FIXED_IDS")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse::<i64>().ok()).collect())
            .unwrap_or_default()
    })
}

fn next_fixed() -> Option<i64> {
    let v = fixed_seq();
    if v.is_empty() {
        return None;
    }
    thread_local! {
        static FI: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }
    let i = FI.with(|c| {
        let i = c.get() % v.len();
        c.set(i + 1);
        i
    });
    Some(v[i])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Empty,
    Prefill,
    Decode,
    Waiting,
}

/// 请求槽（未来 slot 池用；当前单序列路径见 [`Sequence`]）。
#[derive(Debug, Clone)]
pub struct Slot {
    pub id: SlotId,
    pub state: SlotState,
    /// 已确认 token 数（整个系统核心运行时变量）
    pub len: u32,
    pub draft_len: u32,
    pub session: SessionId,
}

impl Slot {
    pub fn empty(id: SlotId) -> Self {
        Self { id, state: SlotState::Empty, len: 0, draft_len: 0, session: SessionId(0) }
    }
}

/// RoPE 表（device + host 各一份，覆盖到容量上限）。host 份给 indexer 选块算块起点
/// RoPE 用；device 份给 q/k 的 RoPE kernel 用（按绝对位置索引）。
pub struct RopeTables {
    pub cos: CudaSlice<bf16>,
    pub sin: CudaSlice<bf16>,
    pub cos_host: Vec<f32>,
    pub sin_host: Vec<f32>,
}

/// 引擎只读句柄：权重两端 + 形状 + RoPE 表。层权重经 `load_layer` 回调逐层取用
/// （调用方决定常驻/流式，driver 不碰 IO）。
pub struct Engine<'a> {
    pub k: &'a Kernels,
    pub shape: &'a LayerShape,
    pub n_layers: u32,
    pub vocab: usize,
    pub endpoints: &'a TrunkWeights,
    pub tables: &'a RopeTables,
    /// 可选：MoE 瘦身专家源（只传命中专家）。`None` = 层权重必须预装全量专家
    pub experts: Option<&'a dyn ExpertSource>,
    /// **跨卡 MoE**（`LINGXI_MOE_SPLIT`）：`Some` 时把落在 `layers` 区间内的层的 MoE 交给 GPU1
    /// 跑。`None` = 现状逐位不变。
    pub remote: Option<&'a lingxi_model::moe::RemoteMoe<'a>>,
    /// **全量 layer-split**：`Some` 时
    /// `[lo, n_layers)` 的**整层**（稠密+注意力+MoE+状态）都跑在 GPU1 上。
    /// 与 `remote`（只搬 MoE）互斥；两者都不设 = 单卡逐位不变。
    pub k1: Option<&'a Kernels>,
    /// 切分点（`k1` 为 `Some` 时有效）
    pub split_lo: u32,
    /// GPU1 侧的 RoPE 表（device 副本必须在 GPU1 上）
    pub tables1: Option<&'a RopeTables>,
    /// GPU1 侧的专家源（后一半层的 arena）
    pub experts1: Option<&'a dyn ExpertSource>,
    /// **EP（专家并行，`LINGXI_EP=1`）**：`Some` 时两卡各跑**全部** 48 层的稠密段，
    /// 路由全量各算一份，专家按**均衡分配表**（[`EpCtx::expert_assign`]）分给两卡，
    /// 每层交换 MoE 部分和。`None` = 旧路径逐位不变。
    pub ep: Option<&'a EpCtx<'a>>,
}

/// EP 模式下第二张卡需要的全部句柄。
///
/// 与 layer-split 的 `k1/tables1/experts1` 不同：EP 里 GPU1 **不是**只跑后 24 层，
/// 而是每层都跑一遍稠密段 + 本卡那半专家，所以它需要一份完整的层权重缓存
/// （`layers1`，0..n_layers 全量）。
pub struct EpCtx<'a> {
    pub k1: &'a Kernels,
    /// GPU1 上的 RoPE 表
    pub tables1: &'a RopeTables,
    /// GPU1 侧的专家源（服务**全部**层，但只会被请求分配表里属于本卡的专家 id）
    pub experts1: &'a dyn ExpertSource,
    /// GPU1 的稠密层权重缓存（**全部** n_layers 层；运行期必须先预热）
    pub layers1: &'a dyn LayerCache,
    /// `[n_layers × num_experts]` 的 `u8` 表：每层 `expert -> card(0/1)`。
    /// 由 [`lingxi_model::moe::build_ep_expert_assign`] **构建一次**，两卡读同一份
    /// ⇒ 同一 token 的同一专家只落在一张卡上（路由不会分叉）。
    pub expert_assign: &'a [u8],
    /// 每层专家数（`expert_assign` 的步长）。
    pub num_experts: usize,
}

impl<'a> EpCtx<'a> {
    /// 第 `layer` 层的 `expert -> card` 表（长度 `num_experts`）。
    #[inline]
    pub fn assign(&self, layer: u32) -> &[u8] {
        let n = self.num_experts;
        let b = layer as usize * n;
        &self.expert_assign[b..b + n]
    }
}

impl<'a> Engine<'a> {
    /// 这一层跑在哪张卡上（不切分时恒为 `self.k`）。
    #[inline]
    pub fn k_of(&self, layer: u32) -> &Kernels {
        // EP：k1 只是"第二张卡"的句柄，**不**承担整层——card0 仍是所有层的主卡
        if self.ep.is_some() {
            return self.k;
        }
        match self.k1 {
            Some(k1) if layer >= self.split_lo => k1,
            _ => self.k,
        }
    }

    /// 是否开了全量 layer-split（EP 模式下 `k1` 存在但**不是** layer-split）。
    #[inline]
    pub fn split_active(&self) -> bool {
        self.ep.is_none() && self.k1.is_some()
    }

    /// 这一层的 RoPE 表（device 部分要跟层同卡）。
    #[inline]
    pub fn tables_of(&self, layer: u32) -> &RopeTables {
        if self.ep.is_some() {
            return self.tables;
        }
        match self.tables1 {
            Some(t1) if layer >= self.split_lo => t1,
            _ => self.tables,
        }
    }

    /// 这一层的专家源。
    #[inline]
    pub fn experts_of(&self, layer: u32) -> Option<&dyn ExpertSource> {
        if self.ep.is_some() {
            return self.experts;
        }
        match (self.experts1, self.k1) {
            (Some(e1), Some(_)) if layer >= self.split_lo => Some(e1),
            _ => self.experts,
        }
    }
}

/// 层权重供给策略（调用方实现）：
/// - **流式**（对拍）：每次新建一层、用后丢弃；
/// - **显存常驻**（生产 decode）：dense 权重只构建一次、永久驻留显存，
/// 跨 decode 步直接按引用复用（CudaSlice 不能 clone——那会真复制显存）。
pub trait LayerProvider {
    fn layer(&mut self, i: u32) -> Result<&LoadedLayer>;
    /// 只读访问器（线程版跨卡流水用）：**必须先预热全部层**，之后 `peek` 只读缓存表。
    fn as_cache(&self) -> Option<&dyn LayerCache> {
        None
    }
}

/// 只读层缓存（`Sync`）：供"按卡分线程"（㊿-①）里两个线程同时取权重。
pub trait LayerCache: Sync {
    fn peek(&self, i: u32) -> Option<&LoadedLayer>;
}

/// **验证批 head 的第二半**：`lm_head` 词表尾部的 Q4 权重，常驻在 GPU1 上。
///
/// 为什么要它：`head_argmax_all` 要读 248320×2560 的 Q4 权重（约 338 MB）≈ 1.8 ms，
/// 而那时 **GPU1 正好空闲**（它刚跑完最后一层）。按词表行二分后两卡各读一半 ⇒ ~0.9 ms。
/// 逐位性：`proj_q4_epi` 在 t=6 走 `gemm_q4_skinny_bf16`（grid.y 行分块、**无 split-k**）
/// ⇒ 每行的 k 归约次序与词表行数无关，二分前后**逐位相同**；argmax 两半再按
/// 「值降序、下标升序」归并也等价于整表 argmax（平局取更小下标 ⇒ 必落在 lo 半）。
pub struct VerifyHeadAlt<'a> {
    /// GPU1（`eng.k1` 的同一份）
    pub k1: &'a lingxi_kernels::Kernels,
    /// 词表尾部 `[vocab/2, vocab)` 的 Q4 权重（GPU1 显存）
    pub wq_hi: &'a lingxi_model::q4::Q4W,
}

/// **投机（speculative decoding）用的可变状态快照/录像。**
///
/// 只需支撑"把状态退回本批之前 + 只重放被接受前缀的 GDN"：
/// - **QSA**：append-only，回卷只要改计数（`QsaState::rewind`）—— 超出的行不会被再读；
/// - **GDN**：循环状态（`s_all` + 卷积历史），必须**备份**；部分接受后**重放**被接受前缀
/// （重放用验证批里录下的每层输入，不重算整个主干）。
///
/// 缓冲一次分配、逐轮复用（36 个 GDN 层 ≈ 116 MB + 每层 `t_max` 行输入）。

// CODE_MAP Phase 4 刀P4-2：lib.rs → lib（API 面）+ checkpoint（存档/序列化）+ sequence（前向编排）。
// 只搬家不改行为；对外 lingxi_driver:: 路径全部经本文件 re-export。

mod checkpoint;
mod sequence;

pub use checkpoint::{HostAlt, HostCheckpoint, SpecBuffers, StateCheckpoint};
pub use sequence::Sequence;

pub(crate) use checkpoint::{QsaHostKv, d2h_f32, d2h_u16, h2d_f32, h2d_u16};

