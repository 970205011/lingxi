//! `lingxi-artifact` — ninfer v2 容器（`.ninfer`）的读取器/写入器。
//!
//! **格式与 ninfer 完全兼容，直接读现成产物**（`ARCHITECTURE.md` §0.5 复用铁律）：
//! ```text
//! magic(8B "NINFER\0\x02") | json_bytes(u64 LE) | JSON directory | pad→4096 | payload
//! directory = {"identity":{"model_id","weights_id"},"objects":[...]}
//!   tensor   : {name,kind:"tensor",shape,format,layout,offset,bytes}
//!   resource : {name,kind:"resource",encoding,offset,bytes}
//! ```
//! offset 相对 payload 起点；tensor 按 layout 的 256B 对齐，resource 按 1B。
//!
//! 职责边界：**只做容器解析与结构校验；不量化、不重转、不碰 CUDA。**
//! 张量角色（role）由 `lingxi-model` 在加载期从 canonical 名字推导，产物本身不改。

pub mod expert_arena;

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use lingxi_ids::{TensorFormat, TensorLayout};
use serde_json::Value;

pub const MAGIC: &[u8; 8] = b"NINFER\x00\x02";
pub const PREFIX_BYTES: u64 = 16;
pub const PAYLOAD_ALIGNMENT: u64 = 4096;

pub const ENCODING_RAW_BYTES_V1: &str = "raw-bytes-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub model_id: String,
    pub weights_id: String,
}

/// 目录中的一个对象。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Object {
    Tensor {
        name: String,
        shape: Vec<u64>,
        format: String,
        layout: String,
        offset: u64,
        bytes: u64,
    },
    Resource {
        name: String,
        encoding: String,
        offset: u64,
        bytes: u64,
    },
}

impl Object {
    pub fn name(&self) -> &str {
        match self {
            Object::Tensor { name, .. } | Object::Resource { name, .. } => name,
        }
    }
    pub fn offset(&self) -> u64 {
        match self {
            Object::Tensor { offset, .. } | Object::Resource { offset, .. } => *offset,
        }
    }
    pub fn bytes(&self) -> u64 {
        match self {
            Object::Tensor { bytes, .. } | Object::Resource { bytes, .. } => *bytes,
        }
    }
    pub fn alignment(&self) -> u64 {
        match self {
            Object::Tensor { layout, .. } => TensorLayout::from_name(layout)
                .expect("已校验 layout")
                .alignment(),
            Object::Resource { .. } => 1,
        }
    }
    /// 该对象的 `(format, layout, shape)`（resource 返回 None）。
    pub fn tensor_parts(&self) -> Option<(TensorFormat, TensorLayout, &[u64])> {
        match self {
            Object::Tensor { shape, format, layout, .. } => Some((
                TensorFormat::from_name(format).expect("已校验 format"),
                TensorLayout::from_name(layout).expect("已校验 layout"),
                shape,
            )),
            Object::Resource { .. } => None,
        }
    }
}

// ───────────────────────────── 解析（严格，照搬 ninfer 契约） ─────────────────────────────

fn need_str(v: &Value, field: &str) -> Result<String> {
    v.get(field)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .with_context(|| format!("{field} 必须是非空字符串"))
}

fn need_u64(v: &Value, field: &str, positive: bool) -> Result<u64> {
    let n = v
        .get(field)
        .and_then(Value::as_u64)
        .with_context(|| format!("{field} 必须是非负整数"))?;
    if positive && n == 0 {
        bail!("{field} 必须为正");
    }
    Ok(n)
}

fn need_exact_keys(v: &Value, kind: &str, keys: &[&str]) -> Result<()> {
    let obj = v.as_object().with_context(|| format!("{kind} 必须是 JSON 对象"))?;
    if obj.len() != keys.len() || !keys.iter().all(|k| obj.contains_key(*k)) {
        bail!("{kind} 成员必须恰为 {keys:?}");
    }
    Ok(())
}

fn parse_object(v: &Value) -> Result<Object> {
    let kind = v.get("kind").and_then(Value::as_str).unwrap_or("");
    match kind {
        "tensor" => {
            need_exact_keys(v, "tensor", &["name", "kind", "shape", "format", "layout", "offset", "bytes"])?;
            let name = need_str(v, "name")?;
            let shape = v["shape"]
                .as_array()
                .context("shape 必须是数组")?
                .iter()
                .map(|d| d.as_u64().filter(|n| *n > 0).context("shape 维度必须为正整数"))
                .collect::<Result<Vec<u64>>>()?;
            let format = need_str(v, "format")?;
            let layout = need_str(v, "layout")?;
            let offset = need_u64(v, "offset", false)?;
            let bytes = need_u64(v, "bytes", true)?;
            let fmt = TensorFormat::from_name(&format)
                .with_context(|| format!("未知数值格式: {format}"))?;
            let lay = TensorLayout::from_name(&layout)
                .with_context(|| format!("未知布局: {layout}"))?;
            let expected = lingxi_ids::try_encoded_size(fmt, lay, &shape)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            if bytes != expected {
                bail!("tensor {name} 存 {bytes} 字节，layout 要求 {expected}");
            }
            Ok(Object::Tensor { name, shape, format, layout, offset, bytes })
        }
        "resource" => {
            need_exact_keys(v, "resource", &["name", "kind", "encoding", "offset", "bytes"])?;
            let name = need_str(v, "name")?;
            let encoding = need_str(v, "encoding")?;
            if encoding != ENCODING_RAW_BYTES_V1 {
                bail!("未知 resource 编码: {encoding}");
            }
            Ok(Object::Resource {
                name,
                encoding,
                offset: need_u64(v, "offset", false)?,
                bytes: need_u64(v, "bytes", true)?,
            })
        }
        other => bail!("object kind 必须是 'tensor' 或 'resource'（得到 {other:?}）"),
    }
}

fn parse_directory(data: &[u8]) -> Result<(Identity, Vec<Object>)> {
    let root: Value = serde_json::from_slice(data).context("directory JSON 解析失败")?;
    need_exact_keys(&root, "directory", &["identity", "objects"])?;
    let idv = &root["identity"];
    need_exact_keys(idv, "identity", &["model_id", "weights_id"])?;
    let identity = Identity {
        model_id: need_str(idv, "model_id")?,
        weights_id: need_str(idv, "weights_id")?,
    };
    let arr = root["objects"].as_array().context("objects 必须是数组")?;
    if arr.is_empty() {
        bail!("objects 不能为空");
    }
    let mut objects = Vec::with_capacity(arr.len());
    let mut seen = std::collections::HashSet::new();
    for item in arr {
        let obj = parse_object(item)?;
        if !seen.insert(obj.name().to_string()) {
            bail!("重复的 object 名: {}", obj.name());
        }
        objects.push(obj);
    }
    Ok((identity, objects))
}

// ───────────────────────────── 读取 ─────────────────────────────

pub struct Artifact {
    pub path: PathBuf,
    pub identity: Identity,
    pub objects: Vec<Object>,
    pub payload_offset: u64,
    pub file_bytes: u64,
    map: memmap2::Mmap,
    index: HashMap<String, usize>,
}

impl Artifact {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).with_context(|| format!("打开产物失败: {}", path.display()))?;
        let file_bytes = file.metadata()?.len();
        if file_bytes < PREFIX_BYTES {
            bail!("产物短于 v2 前缀");
        }

        use std::io::Read;
        let mut prefix = [0u8; PREFIX_BYTES as usize];
        file.read_exact(&mut prefix)?;
        if &prefix[0..8] != MAGIC {
            bail!("magic 不是 NInfer v2");
        }
        let json_bytes = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
        if json_bytes == 0 {
            bail!("json_bytes 必须为正");
        }
        let metadata_end = PREFIX_BYTES + json_bytes;
        let payload_offset = lingxi_ids::align_up(metadata_end, PAYLOAD_ALIGNMENT);
        if metadata_end > file_bytes || payload_offset > file_bytes {
            bail!("声明的 JSON / payload 起点越界");
        }
        let mut directory = vec![0u8; json_bytes as usize];
        file.read_exact(&mut directory)?;
        let (identity, objects) = parse_directory(&directory)?;

        // 结构校验：顺序、对齐、越界
        let payload_bytes = file_bytes - payload_offset;
        let mut cursor = 0u64;
        let mut index = HashMap::new();
        for (i, obj) in objects.iter().enumerate() {
            let align = obj.alignment();
            if obj.offset() < cursor {
                bail!("object {} 重叠或乱序", obj.name());
            }
            if obj.offset() % align != 0 {
                bail!("object {} 未按 {align} 字节对齐", obj.name());
            }
            let end = obj.offset() + obj.bytes();
            if end > payload_bytes {
                bail!("object {} 越过文件尾", obj.name());
            }
            cursor = end;
            index.insert(obj.name().to_string(), i);
        }

        let map = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Artifact { path, identity, objects, payload_offset, file_bytes, map, index })
    }

    pub fn find(&self, name: &str) -> Option<&Object> {
        self.index.get(name).map(|i| &self.objects[*i])
    }

    /// 取对象原始字节（mmap 零拷贝，按需换页）。
    pub fn payload(&self, name: &str) -> Option<&[u8]> {
        let i = *self.index.get(name)?;
        let o = &self.objects[i];
        let begin = (self.payload_offset + o.offset()) as usize;
        Some(&self.map[begin..begin + o.bytes() as usize])
    }

    pub fn tensors(&self) -> impl Iterator<Item = &Object> {
        self.objects.iter().filter(|o| matches!(o, Object::Tensor { .. }))
    }

    pub fn resources(&self) -> impl Iterator<Item = &Object> {
        self.objects.iter().filter(|o| matches!(o, Object::Resource { .. }))
    }

    /// 结构自检（T0）：逐对象核对 offset/对齐/尺寸（解析期已做），此处再核一遍 payload 可达。
    pub fn verify(&self) -> Result<()> {
        let payload_bytes = self.file_bytes - self.payload_offset;
        for o in &self.objects {
            if o.offset() + o.bytes() > payload_bytes {
                bail!("T0: object {} 越界", o.name());
            }
            let begin = (self.payload_offset + o.offset()) as usize;
            let _ = &self.map[begin..begin + o.bytes() as usize];
        }
        Ok(())
    }
}

// ───────────────────────────── 写入（我们自己的产物，格式与 ninfer 兼容） ─────────────────────────────

pub struct Writer {
    file: File,
    pub path: PathBuf,
    pub objects: Vec<Object>,
    pub payload_offset: u64,
    cursor: u64,
    next: usize,
}

impl Writer {
    pub fn create(path: impl AsRef<Path>, identity: &Identity, objects: Vec<Object>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        // 计算对齐后的 offset（入参 offset 被忽略，重新规划）
        let mut planned = Vec::with_capacity(objects.len());
        let mut cursor = 0u64;
        for o in &objects {
            let align = o.alignment();
            let offset = lingxi_ids::align_up(cursor, align);
            let bytes = o.bytes();
            planned.push(match o {
                Object::Tensor { name, shape, format, layout, .. } => Object::Tensor {
                    name: name.clone(), shape: shape.clone(), format: format.clone(),
                    layout: layout.clone(), offset, bytes,
                },
                Object::Resource { name, encoding, .. } => Object::Resource {
                    name: name.clone(), encoding: encoding.clone(), offset, bytes,
                },
            });
            cursor = offset + bytes;
        }
        let directory = encode_directory(identity, &planned)?;
        let payload_offset = lingxi_ids::align_up(PREFIX_BYTES + directory.len() as u64, PAYLOAD_ALIGNMENT);
        let mut file = File::create(&path)?;
        file.write_all(MAGIC)?;
        file.write_all(&(directory.len() as u64).to_le_bytes())?;
        file.write_all(&directory)?;
        file.write_all(&vec![0u8; (payload_offset - PREFIX_BYTES - directory.len() as u64) as usize])?;
        Ok(Writer { file, path, objects: planned, payload_offset, cursor: 0, next: 0 })
    }

    pub fn write(&mut self, name: &str, payload: &[u8]) -> Result<()> {
        let obj = self.objects.get(self.next).context("已无待写对象")?;
        if obj.name() != name {
            bail!("期望写 {}，收到 {name}", obj.name());
        }
        if payload.len() as u64 != obj.bytes() {
            bail!("{} 载荷 {} 字节，应为 {}", name, payload.len(), obj.bytes());
        }
        if obj.offset() > self.cursor {
            self.file.write_all(&vec![0u8; (obj.offset() - self.cursor) as usize])?;
        }
        self.file.write_all(payload)?;
        self.cursor = obj.offset() + obj.bytes();
        self.next += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if self.next != self.objects.len() {
            bail!("缺少对象 {}", self.objects[self.next].name());
        }
        self.file.set_len(self.payload_offset + self.cursor)?;
        self.file.flush()?;
        Ok(())
    }
}

fn encode_directory(identity: &Identity, objects: &[Object]) -> Result<Vec<u8>> {
    let objs: Vec<Value> = objects
        .iter()
        .map(|o| match o {
            Object::Tensor { name, shape, format, layout, offset, bytes } => serde_json::json!({
                "name": name, "kind": "tensor", "shape": shape,
                "format": format, "layout": layout, "offset": offset, "bytes": bytes,
            }),
            Object::Resource { name, encoding, offset, bytes } => serde_json::json!({
                "name": name, "kind": "resource", "encoding": encoding,
                "offset": offset, "bytes": bytes,
            }),
        })
        .collect();
    let root = serde_json::json!({
        "identity": { "model_id": identity.model_id, "weights_id": identity.weights_id },
        "objects": objs,
    });
    Ok(serde_json::to_vec(&root)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_v2_container() {
        let objs = vec![
            Object::Resource { name: "frontend/tokenizer.json".into(), encoding: ENCODING_RAW_BYTES_V1.into(), offset: 0, bytes: 5 },
            Object::Tensor { name: "text/layers/0/attn_gr/hc_norm".into(), shape: vec![10240], format: "BF16".into(), layout: "contiguous-le-v1".into(), offset: 0, bytes: 10240 * 2 },
        ];
        let id = Identity { model_id: "qwen4_exp".into(), weights_id: "bf16".into() };
        let p = std::env::temp_dir().join("lingxi_artifact_rt.ninfer");
        let mut w = Writer::create(&p, &id, objs).unwrap();
        w.write("frontend/tokenizer.json", b"hello").unwrap();
        w.write("text/layers/0/attn_gr/hc_norm", &vec![0u8; 10240 * 2]).unwrap();
        w.finish().unwrap();

        let a = Artifact::open(&p).unwrap();
        a.verify().unwrap();
        assert_eq!(a.identity.model_id, "qwen4_exp");
        assert_eq!(a.payload("frontend/tokenizer.json").unwrap(), b"hello");
        assert_eq!(a.tensors().count(), 1);
    }

    #[test]
    fn rejects_bad_layout_size() {
        // 声明 BF16 + row-split（布局不接受定长格式）→ 尺寸计算/校验必须失败
        let bad = serde_json::json!({
            "identity": {"model_id":"m","weights_id":"w"},
            "objects": [{"name":"x","kind":"tensor","shape":[4,64],
                         "format":"BF16","layout":"row-split-k128-v1","offset":0,"bytes":8}]
        });
        let r = parse_directory(&serde_json::to_vec(&bad).unwrap());
        assert!(r.is_err());
    }
}
