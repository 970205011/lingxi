//! 产物张量解码：把 ninfer v2 产物的物理字节还原为逻辑 f32。
//!
//! 这是引擎读取权重的**基础**：布局语义照搬 ninfer
//! （`contiguous-le-v1` / `row-split-k128-v1`，见 `lingxi-ids::encoded_size`）。
//! 当前实现覆盖现成产物实际用到的格式：`BF16` / `FP32` / `Q4G64_F16S`。

use anyhow::{bail, Result};
use lingxi_artifact::Object;
use lingxi_ids::{align_up, TensorFormat, TensorLayout, K_ALIGNMENT, PLANE_ALIGNMENT};

/// 解码一个 tensor 对象为逻辑 f32（行主序，去掉 K 方向的补零）。
pub fn decode_f32(obj: &Object, bytes: &[u8]) -> Result<Vec<f32>> {
    let (format, layout, shape) = obj
        .tensor_parts()
        .ok_or_else(|| anyhow::anyhow!("不是 tensor 对象"))?;
    if bytes.len() as u64 != lingxi_ids::encoded_size(format, layout, shape) {
        bail!("载荷长度与 format/layout 不符: {}", obj.name());
    }
    match layout {
        TensorLayout::ContiguousLeV1 => decode_contiguous(format, shape, bytes),
        TensorLayout::RowSplitK128V1 => decode_row_split_q4(format, shape, bytes),
        other => bail!("暂不支持的布局: {}", other.name()),
    }
}

fn decode_contiguous(format: TensorFormat, shape: &[u64], bytes: &[u8]) -> Result<Vec<f32>> {
    let n: u64 = shape.iter().product();
    match format {
        TensorFormat::Bf16 => Ok((0..n as usize)
            .map(|i| {
                let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect()),
        TensorFormat::Fp32 => Ok((0..n as usize)
            .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect()),
        other => bail!("contiguous-le-v1 暂不支持 {}", other.name()),
    }
}

/// `row-split-k128-v1`（分组量化）：三平面 base/high/scale，K 补零到 128。
/// 现成产物只用 Q4（base = 每组 32 字节，无 high 平面）。
fn decode_row_split_q4(format: TensorFormat, shape: &[u64], bytes: &[u8]) -> Result<Vec<f32>> {
    let q = format.quant().ok_or_else(|| anyhow::anyhow!("row-split 需要分组量化格式"))?;
    if q.bits != 4 {
        bail!("row-split 解码暂只实现 4bit（得到 {}bit）", q.bits);
    }
    let (n, k) = (shape[0], shape[1]);
    let k_pad = align_up(k, K_ALIGNMENT);
    let groups = (k_pad / q.group as u64) as usize;
    let base_row_bytes = groups * (q.group as usize / 2); // 每组 32B（两个 4bit/字节）
    let scale_row_bytes = groups * 2;
    let base_bytes = n * base_row_bytes as u64;
    let scale_offset = align_up(base_bytes, PLANE_ALIGNMENT) as usize;

    let mut out = vec![0f32; (n * k) as usize];
    for r in 0..n as usize {
        for g in 0..groups {
            // scale（binary16，来自 scale 平面）
            let so = scale_offset + r * scale_row_bytes + g * 2;
            let scale = half::f16::from_bits(u16::from_le_bytes([bytes[so], bytes[so + 1]])).to_f32();
            // 64 个 4bit 码（来自 base 平面）
            let bo = r * base_row_bytes + g * (q.group as usize / 2);
            for e in 0..q.group as usize {
                let byte = bytes[bo + e / 2];
                let nib = if e % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let code = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                let col = g as u64 * q.group as u64 + e as u64;
                if col < k {
                    out[r * k as usize + col as usize] = code as f32 * scale;
                }
            }
        }
    }
    Ok(out)
}
