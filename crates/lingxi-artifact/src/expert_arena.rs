//! `Q4EARN1` 专家 arena 只读读取器（qwen4_exp MoE 专家权重容器）。
//!
//! **归属**（开源合规）：设计参考 Strata（`Niko1221/Strata`, MIT）
//! `src/core/pinned.cu` 的 shared-file arena 与 `include/strata/core/expert_cache.hpp`
//! 的常驻表/槽位思想；容器布局 `Q4EARN1` 与量化档（Q4G64_F16S 等）来自 ninfer-dev
//! `src/targets/qwen4_exp/impl/kernels/moe_expert_arena.h`（灵犀分支）。
//! 本文件只做**只读解析 + 段几何**：不量化、不重转、不碰 CUDA。
//!
//! 文件布局（little endian）：
//! ```text
//! ArenaHeader(72B) | routed index [n_layers*n_expert * ArenaBlobRef(16B)]
//!                  | shared index [n_layers * ArenaBlobRef(16B)]
//!                  | pad→slot_align | data（各 blob 按 slot_align 对齐）
//! routed blob = [gate_up][down]；shared blob = [gate][up][down]（各补齐到公共步长）
//! ArenaBlobRef = { i64 offset(相对 data 区起点), i32 bytes, i32 format }
//! format: 0=BF16 1=Q4G64_F16S 2=Q5G64_F16S 3=Q6G64_F16S 4=W8G32_F16S
//! ```
//! 每个 blob 自带格式字 → 同一 arena 可混合精度（见 ninfer M2.6）。

use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

pub const ARENA_MAGIC: &[u8; 8] = b"Q4EARN1\0";
pub const ARENA_HEADER_BYTES: usize = 72;
pub const ARENA_REF_BYTES: usize = 16;
/// 头部 flags 位 0：`ArenaBlobRef::format` 是该 blob 的量化格式（0 仍表示原样 BF16）。
pub const FLAG_BLOB_FORMATS: i32 = 1;

const K_ALIGNMENT: u64 = 128;
const PLANE_ALIGNMENT: u64 = 256;

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

/// 单个 blob 的权重格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobFormat {
    /// 原样 BF16（本引擎专家核可直接读）
    Bf16Raw,
    /// Q4，group 64，FP16 scale，RowSplit
    Q4G64F16S,
    /// Q5，group 64，FP16 scale，RowSplit
    Q5G64F16S,
    /// Q6，group 64，FP16 scale，RowSplit
    Q6G64F16S,
    /// W8，group 32，FP16 scale，RowSplit
    W8G32F16S,
    /// Q3，group 128，FP16 scale，RowSplit（dense LSB-first 位流：128 值 = 48B，
    /// 元素 i 占 bits [3i,3i+3)，无 high 平面；与上游 layouts.py `_pack_three_bit` 一致）
    Q3G128F16S,
}

impl BlobFormat {
    pub fn from_code(v: i32) -> Option<Self> {
        Some(match v {
            0 => Self::Bf16Raw,
            1 => Self::Q4G64F16S,
            2 => Self::Q5G64F16S,
            3 => Self::Q6G64F16S,
            4 => Self::W8G32F16S,
            5 => Self::Q3G128F16S,
            _ => return None,
        })
    }

    pub fn code(self) -> i32 {
        match self {
            Self::Bf16Raw => 0,
            Self::Q4G64F16S => 1,
            Self::Q5G64F16S => 2,
            Self::Q6G64F16S => 3,
            Self::W8G32F16S => 4,
            Self::Q3G128F16S => 5,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16Raw => "BF16",
            Self::Q4G64F16S => "Q4G64_F16S",
            Self::Q5G64F16S => "Q5G64_F16S",
            Self::Q6G64F16S => "Q6G64_F16S",
            Self::W8G32F16S => "W8G32_F16S",
            Self::Q3G128F16S => "Q3G128_F16S",
        }
    }

    /// 每权重名义位数（BF16 记 16）。
    pub fn bits(self) -> u32 {
        match self {
            Self::Bf16Raw => 16,
            Self::Q4G64F16S => 4,
            Self::Q5G64F16S => 5,
            Self::Q6G64F16S => 6,
            Self::W8G32F16S => 8,
            Self::Q3G128F16S => 3,
        }
    }

    /// 量化组宽（BF16 无组概念，返回 0）。
    pub fn group(self) -> u64 {
        match self {
            Self::Bf16Raw => 0,
            Self::Q4G64F16S | Self::Q5G64F16S | Self::Q6G64F16S => 64,
            Self::W8G32F16S => 32,
            Self::Q3G128F16S => 128,
        }
    }
}

/// 一个 `[rows, cols]` 矩阵在给定格式下的 RowSplit 段几何。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    pub rows: u64,
    pub cols: u64,
    /// 补零后的列数（`align_up(cols, 128)`）
    pub k_pad: u64,
    pub groups_per_row: u64,
    pub base_row_bytes: u64,
    pub base_offset: u64,
    pub base_bytes: u64,
    pub high_offset: u64,
    pub high_bytes: u64,
    pub scale_offset: u64,
    pub scale_bytes: u64,
    pub payload_bytes: u64,
}

/// 与 ninfer `arena_matrix_layout` 逐字段一致。
pub fn matrix_layout(f: BlobFormat, rows: u64, cols: u64) -> Option<MatrixLayout> {
    if rows == 0 || cols == 0 {
        return None;
    }
    if f == BlobFormat::Bf16Raw {
        let base_bytes = rows * cols * 2;
        return Some(MatrixLayout {
            rows,
            cols,
            k_pad: cols,
            groups_per_row: 0,
            base_row_bytes: cols * 2,
            base_offset: 0,
            base_bytes,
            high_offset: base_bytes,
            high_bytes: 0,
            scale_offset: base_bytes, // 无 scale 平面：偏移即结尾
            scale_bytes: 0,
            payload_bytes: base_bytes,
        });
    }
    let (group, bits) = (f.group(), f.bits() as u64);
    let k_pad = align_up(cols, K_ALIGNMENT);
    if k_pad % group != 0 {
        return None;
    }
    let groups_per_row = k_pad / group;
    // base 平面每行字节：4/5/6bit 是 group/2，W8 是 group，**Q3 是 group×3/8（48B/128值）**。
    let base_row_bytes = groups_per_row
        * if bits == 8 {
            group
        } else if bits == 3 {
            group * 3 / 8
        } else {
            group / 2
        };
    // high 平面：Q4/Q8 无；Q5/Q6 有；**Q3 无**（48B 全在 base，dense LSB-first）。
    let high_per_group = if bits == 4 || bits == 8 || bits == 3 {
        0
    } else {
        group * (bits - 4) / 8
    };
    let base_bytes = rows * base_row_bytes;
    let high_bytes = rows * groups_per_row * high_per_group;
    let scale_bytes = rows * groups_per_row * 2;
    let high_offset = align_up(base_bytes, PLANE_ALIGNMENT);
    let scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT);
    Some(MatrixLayout {
        rows,
        cols,
        k_pad,
        groups_per_row,
        base_row_bytes,
        base_offset: 0,
        base_bytes,
        high_offset,
        high_bytes,
        scale_offset,
        scale_bytes,
        payload_bytes: scale_offset + scale_bytes,
    })
}

/// 索引里的一个引用（数据区相对偏移）。
#[derive(Debug, Clone, Copy)]
struct BlobRef {
    offset: i64,
    bytes: i32,
    format: BlobFormat,
}

/// 一个 routed 专家 blob 在 mmap 中的绝对范围。
#[derive(Debug, Clone)]
pub struct ExpertBlob {
    pub format: BlobFormat,
    /// gate_up `[2*n_ff, n_in]`
    pub gu: Range<usize>,
    /// down `[n_in, n_ff]`
    pub down: Range<usize>,
}

pub struct ExpertArena {
    pub path: PathBuf,
    pub n_layers: u32,
    pub n_experts: u32,
    pub n_in: u32,
    pub n_ff: u32,
    pub slot_align: u64,
    pub flags: i32,
    /// ⚠️ **缺省格式几何**（R1 几何单源化，2026-10-21）：open 时按**首个 routed ref 的格式**
    /// 推导的 gu/down 段 payload 字节数。**同 arena 内格式可能混合，本字段只代表缺省格式**——
    /// 任何拿到了 blob 自己格式（`routed()` 的 `BlobRef.format`）的消费侧都必须用
    /// `matrix_layout(fmt, rows, cols).payload_bytes` 自算，禁止读本字段（10-19/10-20 两次
    /// dn 段偏移雷的根源：混档下按缺省格式取几何）。本字段仅保留给"整 arena 单格式"的
    /// 簿记路径（重钉/A2 补槽，均有 !mixed 闸门）。
    pub gu_bytes: u64,
    pub down_bytes: u64,
    data_offset: usize,
    routed_index_off: usize,
    /// shared index 区起始（layer → 单个 shared blob），暂未消费
    pub shared_index_off: usize,
    data_bytes: u64,
    map: memmap2::Mmap,
}

fn rd_i32(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn rd_i64(b: &[u8], o: usize) -> i64 {
    i64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

impl ExpertArena {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .with_context(|| format!("打开专家 arena 失败: {}", path.display()))?;
        let file_bytes = file.metadata()?.len();
        if file_bytes < ARENA_HEADER_BYTES as u64 {
            bail!("arena 短于头部");
        }
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let h = &map[..ARENA_HEADER_BYTES];
        if &h[0..8] != ARENA_MAGIC {
            bail!("arena magic 不是 Q4EARN1");
        }
        let version = rd_i32(h, 8);
        if version != 1 {
            bail!("arena version {version} 不受支持");
        }
        let n_layers = rd_i32(h, 12);
        let n_experts = rd_i32(h, 16);
        let n_in = rd_i32(h, 20);
        let n_ff = rd_i32(h, 24);
        let slot_align = rd_i32(h, 28) as u64;
        let flags = rd_i32(h, 32);
        if flags & !FLAG_BLOB_FORMATS != 0 {
            bail!("arena flags 含未知位（0x{flags:x}）");
        }
        let data_offset = rd_i64(h, 40);
        let routed_index_off = rd_i64(h, 48);
        let shared_index_off = rd_i64(h, 56);
        let data_bytes = rd_i64(h, 64);
        if n_layers <= 0 || n_experts <= 0 || n_in <= 0 || n_ff <= 0 {
            bail!("arena 维度非法: {n_layers}x{n_experts} {n_in}x{n_ff}");
        }
        let data_offset = data_offset as u64;
        let end = data_offset + data_bytes as u64;
        if end > file_bytes {
            bail!("arena 声明 {} 字节（data {} + {}），实际 {}——文件被截断", end, data_offset, data_bytes, file_bytes);
        }

        // 段几何缺省：生产 arena 是**均匀档**（全 Q4G64 或全 Q3G128），按首个 routed
        // ref 的格式推导；混档 arena 逐 blob 以自己的格式为准（见 routed()），
        // 这里仅是 gu_bytes/down_bytes 缺省值。
        let default_fmt = if flags & FLAG_BLOB_FORMATS != 0 {
            let o = routed_index_off as usize;
            if o + ARENA_REF_BYTES > map.len() {
                bail!("arena routed 索引越界（offset {o}）");
            }
            BlobFormat::from_code(rd_i32(&map[o..o + ARENA_REF_BYTES], 12))
                .unwrap_or(BlobFormat::Q4G64F16S)
        } else {
            BlobFormat::Q4G64F16S
        };
        let gu_bytes = matrix_layout(default_fmt, 2 * n_ff as u64, n_in as u64)
            .context("routed gate_up 几何非法")?
            .payload_bytes;
        let down_bytes = matrix_layout(default_fmt, n_in as u64, n_ff as u64)
            .context("routed down 几何非法")?
            .payload_bytes;

        Ok(Self {
            path,
            n_layers: n_layers as u32,
            n_experts: n_experts as u32,
            n_in: n_in as u32,
            n_ff: n_ff as u32,
            slot_align,
            flags,
            gu_bytes,
            down_bytes,
            data_offset: data_offset as usize,
            routed_index_off: routed_index_off as usize,
            shared_index_off: shared_index_off as usize,
            data_bytes: data_bytes as u64,
            map,
        })
    }

    fn read_ref(&self, index_off: usize, i: usize) -> Result<BlobRef> {
        let o = index_off + i * ARENA_REF_BYTES;
        if o + ARENA_REF_BYTES > self.map.len() {
            bail!("arena 索引越界（entry {i}）");
        }
        let b = &self.map[o..o + ARENA_REF_BYTES];
        let format = if self.flags & FLAG_BLOB_FORMATS != 0 {
            BlobFormat::from_code(rd_i32(b, 12))
                .with_context(|| format!("未知 blob 格式码 {}", rd_i32(b, 12)))?
        } else {
            BlobFormat::Bf16Raw
        };
        Ok(BlobRef { offset: rd_i64(b, 0), bytes: rd_i32(b, 8), format })
    }

    /// `(layer, expert)` 的 routed blob 绝对范围；缺席（稀疏 arena）返回 `None`。
    pub fn routed(&self, layer: u32, expert: u32) -> Result<Option<ExpertBlob>> {
        if layer >= self.n_layers || expert >= self.n_experts {
            bail!("routed 越界: layer {layer} expert {expert}");
        }
        let i = (layer as usize) * (self.n_experts as usize) + expert as usize;
        let r = self.read_ref(self.routed_index_off, i)?;
        if r.offset < 0 {
            return Ok(None);
        }
        let base = self.data_offset + r.offset as usize;
        let bytes = r.bytes as usize;
        if r.offset as u64 + bytes as u64 > self.data_bytes {
            bail!("routed blob 越过 data 区");
        }
        // routed blob = [gate_up][down]，段长按该 blob 的格式推导
        let gu_b = matrix_layout(r.format, 2 * self.n_ff as u64, self.n_in as u64)
            .context("routed gu 几何非法")?
            .payload_bytes as usize;
        let dn_b = matrix_layout(r.format, self.n_in as u64, self.n_ff as u64)
            .context("routed down 几何非法")?
            .payload_bytes as usize;
        if gu_b + dn_b != bytes {
            bail!(
                "routed blob 段长与声明不符: {} + {} != {} (fmt={})",
                gu_b, dn_b, bytes, r.format.name()
            );
        }
        Ok(Some(ExpertBlob {
            format: r.format,
            gu: base..base + gu_b,
            down: base + gu_b..base + gu_b + dn_b,
        }))
    }

    /// 数据区（用于 `cudaHostRegisterReadOnly` 整段 pin）。
    pub fn data_region(&self) -> &[u8] {
        &self.map[self.data_offset..self.data_offset + self.data_bytes as usize]
    }

    /// 数据区起始指针（页对齐，mmap 保证）。
    pub fn data_ptr(&self) -> *const u8 {
        self.map[self.data_offset..].as_ptr()
    }

    /// 数据区在文件内的偏移（`mmap_shared_rw` 拿到的别名映射要用**同一个偏移**）。
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// 数据区字节数。
    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    pub fn bytes(&self, r: &Range<usize>) -> &[u8] {
        &self.map[r.clone()]
    }
}

/// 再开一份**可写**的 `MAP_SHARED` 映射（与 `open` 的只读映射是**同一份页**，不额外占内存）。
///
/// 为什么要它（docs/PLAN.md §P13.9b）：`CU_MEMHOSTREGISTER_READ_ONLY` 的语义是"让设备**读**这块
/// host 内存"（zero-copy 映射），**不是把页锁好给 DMA 用** —— 从它发起的 H2D 实测只有 8.6 GB/s，
/// 而且**与 kernel 完全不可重叠**（把拷贝挪到第二条流，专家段"每 miss 增量"一点没变）。
/// 而 `DEFAULT/PORTABLE` 注册要求 VMA **可写**（对 `PROT_READ` 映射必返回 `invalid argument`，
/// 见 `lingxi_kernels::pin_host_readonly` 的注释）⇒ 必须先有这份可写映射。
///
/// ⚠️ **只注册、绝不写**：可写只是为了满足注册的前置条件；写入会破坏 arena。
pub fn mmap_shared_rw(path: impl AsRef<Path>) -> Result<memmap2::MmapMut> {
    let path = path.as_ref();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| {
            format!(
                "以读写打开 arena 失败（DMA 注册需要可写 VMA）: {}",
                path.display()
            )
        })?;
    // SAFETY: 与 `open` 同一约定（文件是预生成的只读产物，进程内只读不写）。
    let m = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };
    Ok(m)
}

// ───────────────────────── 主机侧参考解码（用于精度门禁/对拍） ─────────────────────────

/// IEEE binary16 → f32（保持 bit 语义，不用额外依赖）。
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = -1i32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let exp32 = (127 - 15 + e + 1) as u32;
            (sign << 31) | (exp32 << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// `Q4G64_F16S` 的 RowSplit 解码（与 ninfer `decode_q4_f64` 同一契约）：
/// 高半字节是高列号；`value = ((nib ^ 8) - 8) * scale`。
pub fn decode_q4g64_f16s(payload: &[u8], rows: u64, cols: u64) -> Option<Vec<f32>> {
    let g = matrix_layout(BlobFormat::Q4G64F16S, rows, cols)?;
    if payload.len() < g.payload_bytes as usize {
        return None;
    }
    let mut out = vec![0f32; (rows * cols) as usize];
    for r in 0..rows {
        let code_row = &payload[(r * g.base_row_bytes) as usize..];
        let scale_row = &payload[(g.scale_offset + r * g.groups_per_row * 2) as usize..];
        for gi in 0..g.groups_per_row {
            let scale = f16_to_f32(u16::from_le_bytes([
                scale_row[(gi * 2) as usize],
                scale_row[(gi * 2 + 1) as usize],
            ]));
            for i in 0..64u64 {
                let kk = gi * 64 + i;
                if kk >= cols {
                    break;
                }
                let byte = code_row[(gi * 32 + i / 2) as usize];
                let nib = if i & 1 == 1 { byte >> 4 } else { byte & 0x0f };
                let code = ((nib ^ 0x08) as i32) - 0x08;
                out[(r * cols + kk) as usize] = code as f32 * scale;
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_ninfer_q4g64() {
        // 与 tools/golden/q4_arena_check.py 实测值一致
        let gu = matrix_layout(BlobFormat::Q4G64F16S, 1280, 2560).unwrap();
        assert_eq!(gu.groups_per_row, 40);
        assert_eq!(gu.base_row_bytes, 1280);
        assert_eq!(gu.base_bytes, 1_638_400);
        assert_eq!(gu.high_bytes, 0);
        assert_eq!(gu.scale_offset, 1_638_400);
        assert_eq!(gu.scale_bytes, 102_400);
        assert_eq!(gu.payload_bytes, 1_740_800);

        let dn = matrix_layout(BlobFormat::Q4G64F16S, 2560, 640).unwrap();
        assert_eq!(dn.groups_per_row, 10);
        assert_eq!(dn.payload_bytes, 870_400);

        let bf = matrix_layout(BlobFormat::Bf16Raw, 1280, 2560).unwrap();
        assert_eq!(bf.payload_bytes, 6_553_600);
    }

    #[test]
    fn geometry_q3g128() {
        // 与上游 layouts.py _pack_three_bit（128 值 = 48B dense LSB-first）及
        // quantize_expert_arena.py 的编码器自推几何一致：routed blob = 1,920,000 B。
        let gu = matrix_layout(BlobFormat::Q3G128F16S, 1280, 2560).unwrap();
        assert_eq!(gu.groups_per_row, 20);
        assert_eq!(gu.base_row_bytes, 20 * 48);
        assert_eq!(gu.base_bytes, 1_228_800);
        assert_eq!(gu.high_bytes, 0); // Q3 无 high 平面
        assert_eq!(gu.scale_offset, 1_228_800);
        assert_eq!(gu.scale_bytes, 1280 * 20 * 2);
        assert_eq!(gu.payload_bytes, 1_280_000);

        let dn = matrix_layout(BlobFormat::Q3G128F16S, 2560, 640).unwrap();
        assert_eq!(dn.groups_per_row, 5);
        assert_eq!(dn.payload_bytes, 640_000);

        assert_eq!(BlobFormat::Q3G128F16S.code(), 5);
        assert_eq!(BlobFormat::from_code(5), Some(BlobFormat::Q3G128F16S));
    }
}
