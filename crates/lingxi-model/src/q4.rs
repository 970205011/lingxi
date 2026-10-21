//! 稠密权重的 **W4G64**（group 64、FP16 scale、nibble 二补码）在线量化与设备句柄。
//!
//! 动机（实测，见 `docs/decode-speed-profile.md` §1/§4）：decode 每 token 要从 HBM 读
//! **7.97 GiB 的 bf16 稠密权重**，其中最大的几块（`gdn/query_key_value_z` 84 MB/层、
//! `qsa/query_key_gate_value`、各 `output`）实测都在 ~700 GB/s 的带宽墙上
//! （`launch-bench` 分形状表）。**这些是纯带宽受限的**，把字节数砍到 1/3.77 就是
//! 直接省时间——而小 kernel 的固定开销/调度都动不了它。
//!
//! 布局与专家 arena 的 `Q4G64_F16S` 完全一致（复用同一套解码契约，
//! 见 `lingxi_artifact::expert_arena::matrix_layout` 与 `decode_q4g64_f16s`）：
//! ```text
//! payload = [ base 平面: rows × (k_pad/2) 字节 ]      nibble 打包（偶列=低半字节）
//!         | pad 到 256 对齐
//!         | [ scale 平面: rows × (k_pad/64) × 2 字节 ]  fp16
//! k_pad   = align_up(cols, 128)；groups = k_pad/64
//! value   = ((nib ^ 8) - 8) * scale        （即 nibble 为二补码）
//! ```
//! 量化是 **RTN（round-to-nearest）**，按 64 一组取 `amax` 定 scale、码位限在 [-7,7]
//! （保留 4 bit 二补码的合法范围，避开 -8 边界）。这是"先拿到带宽收益"的第一刀；
//! `docs/weight-compression-plan.md` 规划的 AWQ+GPTQ 是精修版（同一容器，只换量化器）。

use anyhow::{Context, Result};
use cudarc::driver::CudaSlice;
use lingxi_kernels::Kernels;

pub const K_ALIGN: u64 = 128;
pub const GROUPS_PER_ROW_DIV: u64 = 64;
const PLANE_ALIGN: u64 = 256;

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

/// 一个稠密矩阵的 Q4G64 设备句柄。
pub struct Q4W {
    /// 打包后的 payload（base + scale 两个平面）
    pub packed: CudaSlice<u8>,
    pub rows: usize,
    pub cols: usize,
    /// 补齐后的列数（= 每行 nibble 数的 2 倍）
    pub k_pad: i32,
    /// scale 平面相对 payload 起点的字节偏移
    pub scale_off: u64,
}

/// bf16 位形 → `(payload, k_pad, scale_off)`。
pub fn quantize_bf16_bits(bits: &[u16], rows: usize, cols: usize) -> Result<(Vec<u8>, i32, u64)> {
    anyhow::ensure!(bits.len() >= rows * cols, "量化输入长度不足: {} < {}", bits.len(), rows * cols);
    let k_pad = align_up(cols as u64, K_ALIGN);
    let groups = k_pad / GROUPS_PER_ROW_DIV;
    let row_bytes = (k_pad / 2) as usize;
    let base_bytes = rows as u64 * row_bytes as u64;
    let scale_off = align_up(base_bytes, PLANE_ALIGN);
    let scale_bytes = rows as u64 * groups * 2;
    let total = (scale_off + scale_bytes) as usize;
    let mut out = vec![0u8; total];

    for r in 0..rows {
        let src = &bits[r * cols..r * cols + cols];
        let base_row = r * row_bytes;
        for g in 0..groups as usize {
            let c0 = g * 64;
            let mut amax = 0f32;
            for i in 0..64 {
                let c = c0 + i;
                if c >= cols {
                    break;
                }
                let v = half::bf16::from_bits(src[c]).to_f32().abs();
                if v > amax {
                    amax = v;
                }
            }
            // 码位限在 [-7,7]；scale = amax/7 保证不越界（amax=0 时给 1 避免除零）
            let scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
            let inv = 1.0f32 / scale;
            let sh = half::f16::from_f32(scale).to_bits();
            let so = (scale_off as usize) + (r * groups as usize + g) * 2;
            out[so] = (sh & 0xFF) as u8;
            out[so + 1] = (sh >> 8) as u8;
            for i in 0..64 {
                let c = c0 + i;
                if c >= cols {
                    break;
                }
                let v = half::bf16::from_bits(src[c]).to_f32() * inv;
                let code = v.round().clamp(-7.0, 7.0) as i32; // [-7,7]
                let nib = (code & 0x0F) as u8; // 二补码
                let bi = base_row + g * 32 + (i >> 1);
                if i & 1 == 0 {
                    out[bi] = (out[bi] & 0xF0) | nib;
                } else {
                    out[bi] = (out[bi] & 0x0F) | (nib << 4);
                }
            }
        }
    }
    Ok((out, k_pad as i32, scale_off))
}

/// 量化并上传到设备。
///
/// ⚠️ 这条是"每次都现算"的形态。要**提前算好、之后只取**请用
/// [`crate::gpu::upload_q4_named`]（按张量名寻址的磁盘缓存，命中时连 bf16 都不用读）。
pub fn upload_from_bf16_bits(
    k: &Kernels, bits: &[u16], rows: usize, cols: usize,
) -> Result<Q4W> {
    let (payload, k_pad, scale_off) = quantize_bf16_bits(bits, rows, cols)
        .with_context(|| format!("Q4 量化失败 rows={rows} cols={cols}"))?;
    let packed = k.stream.clone_htod(&payload)?;
    Ok(Q4W { packed, rows, cols, k_pad, scale_off })
}

/// 设备上 Q4 占的字节数（用于日志/显存预算）。
impl Q4W {
    pub fn bytes(&self) -> usize {
        let groups = self.k_pad as u64 / GROUPS_PER_ROW_DIV;
        self.scale_off as usize + self.rows * groups as usize * 2
    }
}
