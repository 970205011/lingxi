//! `lingxi-plan` — 由 [`TensorCatalog`] 生成运行时内存布局（offset / [`Span`]）。
//!
//! 职责边界（见 `ARCHITECTURE.md`）：**只算地址，不做计算、不碰 CUDA。
//! ** 它是 `TensorResolver` 的实现方：model/kernel 只拿 [`TensorId`] 换 [`Span`]。

use lingxi_ids::{Span, TensorCatalog, TensorId, TensorResolver};

/// 设备/主机侧对齐粒度（cubin、向量化读取都要求高位对齐）。
pub const ALIGN: u64 = 256;

/// 权重区布局：按 catalog 顺序给每个 tensor 分配对齐后的地址。
#[derive(Debug, Clone)]
pub struct WeightLayout {
    base: u64,
    spans: Vec<Span>,
    total_bytes: u64,
}

impl WeightLayout {
    pub fn plan(catalog: &TensorCatalog, base: u64) -> Self {
        let mut cursor = base;
        let mut spans = Vec::with_capacity(catalog.len());
        for (_, info) in catalog.iter() {
            cursor = align_up(cursor, ALIGN);
            let len = info.storage_bytes();
            spans.push(Span { offset: cursor, len, stride: 0 });
            cursor += len;
        }
        WeightLayout { base, spans, total_bytes: cursor - base }
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// 运行期工作区大小（后续按层临时量最大值补全；v1 先占位）。
    pub fn workspace_bytes(&self) -> u64 {
        0
    }
}

impl TensorResolver for WeightLayout {
    fn span(&self, id: TensorId) -> Span {
        self.spans[id.0 as usize]
    }
}

pub fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) / a * a
}

#[cfg(test)]
mod tests {
    use super::*;
    use lingxi_ids::{TensorFormat, TensorInfo, TensorLayout, TensorRole};

    #[test]
    fn spans_are_aligned_and_contiguous() {
        let cat = TensorCatalog::new(vec![
            TensorInfo { name: "a".into(), format: TensorFormat::Bf16, layout: TensorLayout::ContiguousLeV1, shape: vec![3], role: TensorRole::Other },
            TensorInfo { name: "b".into(), format: TensorFormat::Fp32, layout: TensorLayout::ContiguousLeV1, shape: vec![1], role: TensorRole::Other },
        ]);
        let l = WeightLayout::plan(&cat, 4096);
        let s0 = l.span(TensorId(0));
        assert_eq!(s0.offset, 4096);
        assert_eq!(s0.len, 6);
        let s1 = l.span(TensorId(1));
        assert_eq!(s1.offset, 4096 + 256); // 对齐到 256
        assert_eq!(s1.len, 4);
        assert_eq!(l.total_bytes(), 256 + 4);
    }
}
