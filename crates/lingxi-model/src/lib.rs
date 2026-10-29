//! `lingxi-model` — L2 前向逻辑 + 本模型的静态描述与张量角色推导。
//!
//! 职责边界（见 `ARCHITECTURE.md`）：**只做数学编排与模型知识；不做调度、不做 IO。**
//!
//! 张量角色（[`TensorRole`]）是**模型知识**，由 canonical 名字在加载期推导——
//! 产物本身不含 role，因此不需要改动 ninfer 的产物（复用铁律）。

use lingxi_artifact::Artifact;
use lingxi_ids::{Dims, LayerId, LayerKind, LayerSpec, ModelId, ModelSpec, TensorCatalog, TensorInfo, TensorRole};

pub mod attn_prof;
pub mod cpu_moe;
pub mod decode;
pub mod draft_prof;
pub mod gdn;
pub mod gpu;
pub mod gr;
pub mod gr_prof;
pub mod graph;
pub mod graph_timer;
pub mod layer;
pub mod moe;
pub mod mtp;
pub mod mtp_prof;
pub mod plet;
pub mod probes;
pub mod q4;
pub mod qsa;
pub mod rope;
pub mod scratch;
pub mod suffix;
pub mod trunk;

pub const MODEL_ID: ModelId = ModelId(1);
pub const MODEL_NAME: &str = "Qwen3.8-Flash-Next-Uncensored";

/// 环境开关**带缓存**查询（热路径专用）。
///
/// `std::env::var` 每次都要加锁并线性扫全部环境变量、再构造一个 `String`；
/// 而我们对 `LINGXI_PROF` / `LINGXI_GDN_HOST` / `LINGXI_QSA_HOST` 这类开关是**每层**
/// 查一次的（48 层 × 多重 = 每 token 数百次），实测是可观的纯 CPU 浪费。
/// 进程内首次查询后缓存结果（这些开关本来就不该在运行中改变）。
pub fn flag(name: &str) -> bool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let m = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // 单线程快路径：先用 get 试探，命中就不再构造 String
    if let Ok(g) = m.try_lock() {
        if let Some(v) = g.get(name) {
            return *v;
        }
    }
    let v = std::env::var(name).is_ok();
    if let Ok(mut g) = m.lock() {
        g.insert(name.to_string(), v);
    }
    v
}

/// **值感知**开关查询（2026-10-27 教训复发后引入）：未设 = `default`；设了按值判
/// （"0"/"false"/"no"/"off" = 关，其余 = 开）。`flag()` 是"存在即真"——
/// `export LINGXI_X=0` 对它**照样是开**，值语义开关必须走这里才允许外部 `=0` 覆盖。
pub fn flag_val(name: &str, default: bool) -> bool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let m = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(g) = m.try_lock() {
        if let Some(v) = g.get(name) {
            return *v;
        }
    }
    let v = match std::env::var(name) {
        Err(_) => default,
        Ok(s) => !matches!(s.as_str(), "0" | "false" | "no" | "off" | ""),
    };
    if let Ok(mut g) = m.lock() {
        g.insert(name.to_string(), v);
    }
    v
}

/// 逗号分隔的整数表（诊断用）：判断 `v` 是否在 `name` 环境变量列出的集合里。
/// 例：`LINGXI_FULLDEV_OFF=26,27` ⇒ 这两层强制走 legacy 路径。
pub fn env_list_contains(name: &str, v: u32) -> bool {
    match std::env::var(name) {
        Ok(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<u32>().ok())
            .any(|x| x == v),
        Err(_) => false,
    }
}

/// 静态形状（编译期常量集中地）。来自 HF `config.json` 的 `text_config`。
pub const DIMS: Dims = Dims {
    hidden: 2560,
    n_layers: 48,
    n_mtp: 1,
    hyper_connection: 4,
    hyper_connection_dim: 10240, // 4 * 2560
    qsa_head_dim: 256,
    q_heads: 24,
    kv_heads: 2,
    vocab: 248320,
};

/// 全注意力（QSA）层：0-based 3,7,11,…,47（每 4 层一个，共 12）。其余为 GDN。
pub fn layer_kind(index: u32) -> LayerKind {
    if index % 4 == 3 {
        LayerKind::QsaSparseAttention
    } else {
        LayerKind::GdnLinearAttention
    }
}

/// 由 canonical 名字推导张量角色。**先具体后一般**，避免误判。
pub fn role_of(name: &str) -> TensorRole {
    // 顶层
    if name == "text/token_embedding" {
        return TensorRole::TokenEmbedding;
    }
    if name == "text/output_head" {
        return TensorRole::OutputHead;
    }
    // Hyper-connection（GR）：三种子类
    if name.ends_with("hc_norm") {
        return TensorRole::HyperConnectionNorm;
    }
    if name.ends_with("block_inject") {
        return TensorRole::HyperConnectionInject;
    }
    if name.ends_with("mix_down") || name.ends_with("mix_up") {
        return TensorRole::HyperConnectionMix;
    }
    // 全注意力（QSA）
    if name.ends_with("/attention/query_key_gate_value") {
        return TensorRole::AttentionProjection;
    }
    if name.ends_with("/attention/output") {
        return TensorRole::AttentionOutput;
    }
    if name.ends_with("/indexer/index_qk_proj") {
        return TensorRole::AttentionIndexerProjection;
    }
    if name.ends_with("/attention/query_norm")
        || name.ends_with("/attention/key_norm")
        || name.ends_with("/indexer/q_layernorm")
        || name.ends_with("/indexer/k_layernorm")
    {
        return TensorRole::AttentionNorm;
    }
    // 线性注意力（GDN）
    if name.contains("/gdn/") {
        if name.ends_with("/gdn/norm") {
            return TensorRole::LinearAttentionNorm;
        }
        if name.ends_with("/gdn/a_log") || name.ends_with("/gdn/dt_bias") || name.ends_with("/gdn/convolution") {
            return TensorRole::LinearAttentionState;
        }
        return TensorRole::LinearAttentionProjection;
    }
    // MoE（trunk 用 `/moe/`，MTP 用 `/mlp/`，故只按后缀判）
    if name.ends_with("/router_shared_gate") {
        return TensorRole::MoeRouter;
    }
    if name.ends_with("/routed_gate_up") {
        return TensorRole::MoeRoutedGateUp;
    }
    if name.ends_with("/routed_down") {
        return TensorRole::MoeRoutedDown;
    }
    if name.ends_with("/shared_gate_up") {
        return TensorRole::MoeSharedGateUp;
    }
    if name.ends_with("/shared_down") {
        return TensorRole::MoeSharedDown;
    }
    // MTP stem
    if name.starts_with("mtp/") && (name.ends_with("fc_embedding") || name.ends_with("fc_hidden")) {
        return TensorRole::MtpStemProjection;
    }
    if name == "mtp/embedding_norm" || name == "mtp/hidden_norm" {
        return TensorRole::MtpStemNorm;
    }
    // PLE
    if name.starts_with("text/ple/") {
        return if name.contains("norm") {
            TensorRole::PleNorm
        } else {
            TensorRole::PleProjection
        };
    }
    TensorRole::Other
}

/// 从现成产物构建张量目录（`TensorId ↔ TensorInfo(name,shape,format,layout,role)`）。
/// **名字解析只在这里发生**；其它模块之后只用 [`lingxi_ids::TensorId`]。
pub fn catalog_from_artifact(a: &Artifact) -> TensorCatalog {
    let infos: Vec<TensorInfo> = a
        .tensors()
        .map(|o| {
            let (format, layout, shape) = o.tensor_parts().expect("tensors() 只返回 tensor");
            TensorInfo {
                name: o.name().to_string(),
                shape: shape.to_vec(),
                format,
                layout,
                role: role_of(o.name()),
            }
        })
        .collect();
    TensorCatalog::new(infos)
}

/// 模型规格实现。换模型 = 新加一个 `impl ModelSpec` + 一个 catalog，其它模块不动。
pub struct FlashNextModel {
    dims: Dims,
    layers: Vec<LayerSpec>,
}

impl FlashNextModel {
    pub fn new() -> Self {
        let layers = (0..DIMS.n_layers)
            .map(|i| LayerSpec { layer: LayerId(i), kind: layer_kind(i) })
            .collect();
        Self { dims: DIMS, layers }
    }
}

impl Default for FlashNextModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelSpec for FlashNextModel {
    fn id(&self) -> ModelId {
        MODEL_ID
    }
    fn name(&self) -> &'static str {
        MODEL_NAME
    }
    fn dims(&self) -> Dims {
        self.dims
    }
    fn layers(&self) -> &[LayerSpec] {
        &self.layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qsa_layers_are_every_fourth_starting_at_three() {
        let m = FlashNextModel::new();
        assert_eq!(m.layers().len(), 48);
        let qsa: Vec<u32> = m
            .layers()
            .iter()
            .filter(|l| l.kind == LayerKind::QsaSparseAttention)
            .map(|l| l.layer.0)
            .collect();
        assert_eq!(qsa, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
    }

    #[test]
    fn canonical_names_map_to_roles() {
        assert_eq!(role_of("text/token_embedding"), TensorRole::TokenEmbedding);
        assert_eq!(role_of("text/output_head"), TensorRole::OutputHead);
        assert_eq!(role_of("text/layers/0/attn_gr/hc_norm"), TensorRole::HyperConnectionNorm);
        assert_eq!(role_of("text/layers/0/attn_gr/mix_down"), TensorRole::HyperConnectionMix);
        assert_eq!(role_of("text/layers/3/attention/query_key_gate_value"), TensorRole::AttentionProjection);
        assert_eq!(role_of("text/layers/3/attention/query_norm"), TensorRole::AttentionNorm);
        assert_eq!(role_of("text/layers/0/gdn/query_key_value_z"), TensorRole::LinearAttentionProjection);
        assert_eq!(role_of("text/layers/0/gdn/a_log"), TensorRole::LinearAttentionState);
        assert_eq!(role_of("text/layers/0/moe/routed_gate_up"), TensorRole::MoeRoutedGateUp);
        assert_eq!(role_of("text/layers/0/moe/router_shared_gate"), TensorRole::MoeRouter);
        assert_eq!(role_of("text/ple/norm_key"), TensorRole::PleNorm);
        assert_eq!(role_of("mtp/layers/0/mlp_gr/hc_norm"), TensorRole::HyperConnectionNorm);
        assert_eq!(role_of("text/layers/3/attention/indexer/index_qk_proj"), TensorRole::AttentionIndexerProjection);
        assert_eq!(role_of("mtp/fc_embedding"), TensorRole::MtpStemProjection);
        assert_eq!(role_of("mtp/hidden_norm"), TensorRole::MtpStemNorm);
        assert_eq!(role_of("mtp/layer/mlp/routed_down"), TensorRole::MoeRoutedDown);
        assert_eq!(role_of("mtp/layer/mlp/shared_gate_up"), TensorRole::MoeSharedGateUp);
    }
}
