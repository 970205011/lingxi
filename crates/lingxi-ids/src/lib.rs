//! `lingxi-ids` — 标识与连接层：所有跨模块契约的唯一来源。
//!
//! 铁律（见 `ARCHITECTURE.md`）：模块之间只经本层通信；换模型 / 换 kernel / 加功能只改本层注册表。
//!
//! 词表部分（[`TensorFormat`] / [`TensorLayout`] / [`encoded_size`]）**照搬 ninfer 的注册表语义**
//! （`tools/artifact/numeric.py` / `layouts.py`），不自造简化版——这是"复用优于自研"铁律的落点。

use serde::{Deserialize, Serialize};

// ───────────────────────────── 数值格式（照搬 ninfer numeric 注册表） ─────────────────────────────

/// 持久化数值格式。名字与 ninfer 注册表逐字一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorFormat {
    // 定长直接格式
    Bf16,
    Fp32,
    I32,
    // 分组量化（signed grouped codes + 每组一个 binary16 scale）
    Q3G128F16S,
    Q4G64F16S,
    Q5G64F16S,
    Q6G64F16S,
    W8G32F16S,
    // E2M1 + E4M3FN 块量化
    Nvfp4,
    // E4M3FN + 行 BF16 scale
    Fp8E4m3fnRowBf16s,
}

impl TensorFormat {
    pub const ALL: [TensorFormat; 10] = [
        TensorFormat::Bf16,
        TensorFormat::Fp32,
        TensorFormat::I32,
        TensorFormat::Q3G128F16S,
        TensorFormat::Q4G64F16S,
        TensorFormat::Q5G64F16S,
        TensorFormat::Q6G64F16S,
        TensorFormat::W8G32F16S,
        TensorFormat::Nvfp4,
        TensorFormat::Fp8E4m3fnRowBf16s,
    ];

    pub fn name(self) -> &'static str {
        match self {
            TensorFormat::Bf16 => "BF16",
            TensorFormat::Fp32 => "FP32",
            TensorFormat::I32 => "I32",
            TensorFormat::Q3G128F16S => "Q3G128_F16S",
            TensorFormat::Q4G64F16S => "Q4G64_F16S",
            TensorFormat::Q5G64F16S => "Q5G64_F16S",
            TensorFormat::Q6G64F16S => "Q6G64_F16S",
            TensorFormat::W8G32F16S => "W8G32_F16S",
            TensorFormat::Nvfp4 => "NVFP4",
            TensorFormat::Fp8E4m3fnRowBf16s => "FP8_E4M3FN_ROW_BF16S",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }

    /// 定长格式的每元素字节数（量化格式返回 None）。
    pub fn word_bytes(self) -> Option<u64> {
        match self {
            TensorFormat::Bf16 => Some(2),
            TensorFormat::Fp32 | TensorFormat::I32 => Some(4),
            _ => None,
        }
    }

    /// 量化参数 `(bits, group_size, qmin, qmax)`；非分组量化返回 None。
    pub fn quant(self) -> Option<QuantParams> {
        let p = match self {
            TensorFormat::Q3G128F16S => (3, 128, -4, 3),
            TensorFormat::Q4G64F16S => (4, 64, -8, 7),
            TensorFormat::Q5G64F16S => (5, 64, -16, 15),
            TensorFormat::Q6G64F16S => (6, 64, -32, 31),
            TensorFormat::W8G32F16S => (8, 32, -127, 127),
            _ => return None,
        };
        Some(QuantParams { bits: p.0, group: p.1, qmin: p.2, qmax: p.3 })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantParams {
    pub bits: u32,
    pub group: u32,
    pub qmin: i32,
    pub qmax: i32,
}

// ───────────────────────────── 张量布局（照搬 ninfer layouts 注册表） ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockscaleK16M128x4V1,
    RowScaleV1,
}

impl TensorLayout {
    pub fn name(self) -> &'static str {
        match self {
            TensorLayout::ContiguousLeV1 => "contiguous-le-v1",
            TensorLayout::RowSplitK128V1 => "row-split-k128-v1",
            TensorLayout::BlockscaleK16M128x4V1 => "blockscale-k16-m128x4-v1",
            TensorLayout::RowScaleV1 => "row-scale-v1",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        [
            TensorLayout::ContiguousLeV1,
            TensorLayout::RowSplitK128V1,
            TensorLayout::BlockscaleK16M128x4V1,
            TensorLayout::RowScaleV1,
        ]
        .into_iter()
        .find(|l| l.name() == name)
    }

    pub fn alignment(self) -> u64 {
        256
    }
}

pub const PLANE_ALIGNMENT: u64 = 256;
pub const K_ALIGNMENT: u64 = 128;

pub fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) / a * a
}

/// tensor 的物理字节数（照搬 ninfer `encoded_size` 语义）。格式/布局不匹配时 panic。
pub fn encoded_size(format: TensorFormat, layout: TensorLayout, shape: &[u64]) -> u64 {
    try_encoded_size(format, layout, shape).unwrap_or_else(|e| panic!("encoded_size: {e}"))
}

/// 可失败版本：校验 format 与 layout 的兼容性（解析产物时必须走这个）。
pub fn try_encoded_size(
    format: TensorFormat,
    layout: TensorLayout,
    shape: &[u64],
) -> Result<u64, String> {
    match layout {
        TensorLayout::ContiguousLeV1 => format
            .word_bytes()
            .map(|w| shape.iter().product::<u64>() * w)
            .ok_or_else(|| "contiguous-le-v1 需要定长格式（BF16/FP32/I32）".to_string()),
        TensorLayout::RowSplitK128V1 => {
            if shape.len() != 2 {
                return Err("row-split-k128-v1 需要 rank-2 形状".into());
            }
            let q = format
                .quant()
                .ok_or_else(|| "row-split-k128-v1 需要分组量化格式".to_string())?;
            Ok(row_split_payload_bytes(q, shape))
        }
        TensorLayout::BlockscaleK16M128x4V1 => {
            if format != TensorFormat::Nvfp4 {
                return Err("blockscale-k16-m128x4-v1 需要 NVFP4".into());
            }
            let (n, k) = (shape[0], shape[1]);
            let code = n * k / 2;
            let scale_off = align_up(code, PLANE_ALIGNMENT);
            Ok(scale_off + n * k / 16 + 4)
        }
        TensorLayout::RowScaleV1 => {
            if format != TensorFormat::Fp8E4m3fnRowBf16s {
                return Err("row-scale-v1 需要 FP8_E4M3FN_ROW_BF16S".into());
            }
            let (n, k) = (shape[0], shape[1]);
            Ok(align_up(n * k, PLANE_ALIGNMENT) + n * 2)
        }
    }
}

fn row_split_payload_bytes(q: QuantParams, shape: &[u64]) -> u64 {
    let (n, k) = (shape[0], shape[1]);
    let k_pad = align_up(k, K_ALIGNMENT);
    let groups = k_pad / q.group as u64;
    let (base_per_group, high_per_group) = if q.bits == 8 {
        (q.group as u64, 0)
    } else if q.bits == 2 || q.bits == 3 {
        (q.group as u64 * q.bits as u64 / 8, 0)
    } else {
        (
            q.group as u64 / 2,
            if q.bits == 4 { 0 } else { q.group as u64 * (q.bits as u64 - 4) / 8 },
        )
    };
    let base_bytes = n * groups * base_per_group;
    let high_bytes = n * groups * high_per_group;
    let scale_bytes = n * groups * 2;
    let high_off = align_up(base_bytes, PLANE_ALIGNMENT);
    let scale_off = high_off + align_up(high_bytes, PLANE_ALIGNMENT);
    scale_off + scale_bytes
}

// ───────────────────────────── tensor 角色 ─────────────────────────────

/// tensor 在计算中的角色。让模块按角色取用，不必知道具体名字。
/// **由 canonical 名字在加载期推导**（不经产物），见 `lingxi-model`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorRole {
    TokenEmbedding,
    OutputHead,
    AttentionProjection,
    AttentionOutput,
    AttentionNorm,
    /// QSA indexer 的 qk 投影（`.../attention/indexer/index_qk_proj`）
    AttentionIndexerProjection,
    LinearAttentionProjection,
    LinearAttentionState,
    LinearAttentionNorm,
    MoeRouter,
    MoeRoutedGateUp,
    MoeRoutedDown,
    MoeSharedGateUp,
    MoeSharedDown,
    HyperConnectionNorm,
    HyperConnectionInject,
    HyperConnectionMix,
    PleProjection,
    PleNorm,
    /// MTP stem 投影（`mtp/fc_embedding` / `mtp/fc_hidden`）
    MtpStemProjection,
    /// MTP stem 归一化（`mtp/embedding_norm` / `mtp/hidden_norm`）
    MtpStemNorm,
    Other,
}

// ───────────────────────────── 身份 ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    GdnLinearAttention,
    QsaSparseAttention,
}

/// 门控激活：`RMSNormGated` 里 gate 分支的激活函数（模型 config 的 `output_gate_type`）。
/// GDN 层用它，未来换模型只改这里。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateActivation {
    Silu,
    Sigmoid,
}

impl GateActivation {
    pub fn name(self) -> &'static str {
        match self {
            GateActivation::Silu => "silu",
            GateActivation::Sigmoid => "sigmoid",
        }
    }
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "silu" => Some(GateActivation::Silu),
            "sigmoid" => Some(GateActivation::Sigmoid),
            _ => None,
        }
    }
    pub fn apply(self, x: f32) -> f32 {
        match self {
            GateActivation::Silu => x / (1.0 + (-x).exp()),
            GateActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpId {
    RmsNorm,
    RmsNormGrouped,
    Gemv,
    Gemm,
    Rope,
    GdnStep,
    QsaAttention,
    MoeRouter,
    MoeExpert,
    GrRead,
    GrWrite,
    Sample,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelId {
    Smoke,
    RmsNormF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeTag {
    SmallTokens,
    DecodeStep,
    PrefillChunk,
    Quantized,
}

// ───────────────────────────── 结构描述 ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dims {
    pub hidden: u32,
    pub n_layers: u32,
    pub n_mtp: u32,
    pub hyper_connection: u32,
    pub hyper_connection_dim: u32,
    pub qsa_head_dim: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub vocab: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSpec {
    pub layer: LayerId,
    pub kind: LayerKind,
}

/// tensor 元信息。`name` 只在本层与 artifact 层持有；其它模块只拿 [`TensorId`]。
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub format: TensorFormat,
    pub layout: TensorLayout,
    pub role: TensorRole,
}

impl TensorInfo {
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }
    pub fn storage_bytes(&self) -> u64 {
        encoded_size(self.format, self.layout, &self.shape)
    }
    pub fn is_quantized(&self) -> bool {
        self.format.quant().is_some()
    }
    pub fn is_row_split(&self) -> bool {
        self.layout == TensorLayout::RowSplitK128V1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub offset: u64,
    pub len: u64,
    pub stride: u64,
}

// ───────────────────────────── catalog ─────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct TensorCatalog {
    infos: Vec<TensorInfo>,
}

impl TensorCatalog {
    pub fn new(infos: Vec<TensorInfo>) -> Self {
        Self { infos }
    }
    pub fn len(&self) -> usize {
        self.infos.len()
    }
    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }
    pub fn id_of(&self, name: &str) -> Option<TensorId> {
        self.infos.iter().position(|i| i.name == name).map(|i| TensorId(i as u32))
    }
    pub fn info(&self, id: TensorId) -> Option<&TensorInfo> {
        self.infos.get(id.0 as usize)
    }
    pub fn iter(&self) -> impl Iterator<Item = (TensorId, &TensorInfo)> {
        self.infos.iter().enumerate().map(|(i, info)| (TensorId(i as u32), info))
    }
    pub fn by_role(&self, role: TensorRole) -> impl Iterator<Item = TensorId> + '_ {
        self.iter().filter(move |(_, i)| i.role == role).map(|(id, _)| id)
    }
}

// ───────────────────────────── 可替换接口 ─────────────────────────────

pub trait TensorResolver {
    fn span(&self, id: TensorId) -> Span;
}

pub trait KernelRegistry {
    fn kernel(&self, op: OpId, shape: ShapeTag) -> KernelId;
}

pub trait ModelSpec {
    fn id(&self) -> ModelId;
    fn name(&self) -> &'static str;
    fn dims(&self) -> Dims;
    fn layers(&self) -> &[LayerSpec];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_names_roundtrip_like_ninfer() {
        for f in TensorFormat::ALL {
            assert_eq!(TensorFormat::from_name(f.name()), Some(f));
        }
        assert_eq!(TensorFormat::Q4G64F16S.quant().unwrap().group, 64);
        assert_eq!(TensorFormat::Q3G128F16S.quant().unwrap().bits, 3);
    }

    #[test]
    fn contiguous_size_matches_words() {
        assert_eq!(encoded_size(TensorFormat::Bf16, TensorLayout::ContiguousLeV1, &[248320, 2560]),
                   248320 * 2560 * 2);
        assert_eq!(encoded_size(TensorFormat::Fp32, TensorLayout::ContiguousLeV1, &[48]), 192);
    }

    #[test]
    fn row_split_k128_pads_k_to_128() {
        // Q4G64_F16S，n=2560, k=6144：k 已是 128 的倍数
        let bytes = encoded_size(TensorFormat::Q4G64F16S, TensorLayout::RowSplitK128V1, &[2560, 6144]);
        // base = 2560 * (6144/64) * (64/2) = 2560*96*32
        let base = 2560u64 * 96 * 32;
        let scale = 2560u64 * 96 * 2;
        assert_eq!(bytes, align_up(base, 256) + scale);
        // k=48 会被补到 128 → 2 组
        let small = encoded_size(TensorFormat::Q4G64F16S, TensorLayout::RowSplitK128V1, &[4, 48]);
        assert_eq!(small, 4 * 2 * 32 + 4 * 2 * 2);
    }
}
