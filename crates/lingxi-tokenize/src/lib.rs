//! `lingxi-tokenize` — 纯 host 的文本↔token 桥（HF `tokenizer.json` 语义）。
//!
//! 职责边界（见 `ARCHITECTURE.md`）：**只做分词，不做 IO、不碰 CUDA、不依赖任何项目内 crate。**
//! tokenizer.json 字节由调用方从产物 resource（`frontend/tokenizer.json`）取出后传入，
//! 本 crate 不知道产物存在，server 与 bench 都可复用。
//!
//! 复用优于自研（铁律 0.5）：直接使用 Hugging Face 官方 Rust 库 `tokenizers`，
//! 不自己实现 byte-level BPE / added tokens 解码，那套边界规则手写必错。

use anyhow::{Context, Result};
use tokenizers::Tokenizer as HfTokenizer;

/// 文本分词器：从产物内 `frontend/tokenizer.json` 构造，进程内复用。
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    /// 从 tokenizer.json 的原始字节构造。
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let inner = HfTokenizer::from_bytes(bytes.to_vec())
            .map_err(|e| anyhow::anyhow!("解析 tokenizer.json 失败: {e}"))?;
        Ok(Self { inner })
    }

    /// 文本 → token id（自动加该分词器声明的 special tokens；Qwen3 系无 BOS）。
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("encode 失败: {e}"))?;
        Ok(enc.get_ids().iter().map(|&v| v as i64).collect())
    }

    /// token id → 文本（含 added token 还原、byte-level 解码、空白重建）。
    pub fn decode(&self, ids: &[i64]) -> Result<String> {
        let ids32: Vec<u32> = ids.iter().map(|&v| v as u32).collect();
        self.inner
            .decode(&ids32, true)
            .map_err(|e| anyhow::anyhow!("decode 失败: {e}"))
            .context("tokenizer decode")
    }

    /// 按符号查 special token id（如 `<|im_end|>`）。不存在返回 None。
    pub fn token_to_id(&self, token: &str) -> Option<i64> {
        self.inner.token_to_id(token).map(|v| v as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 无产物时的退化行为：坏字节应报错而非 panic。
    #[test]
    fn garbage_bytes_error() {
        assert!(Tokenizer::from_json(b"not a tokenizer").is_err());
    }
}
