//! R4 拆分（2026-10-21）：原 serve.rs 3152 行按职责分段，本目录为其中一段。
//! 模块总览见 `serve/mod.rs`。段间共享类型一律 `pub(crate)`。
//! CODE_MAP Phase 4 收尾刀：proto.rs → proto/{chat,sse}.rs（只搬家不改行为）。

pub(crate) mod chat;
pub(crate) mod sse;

pub(crate) use chat::*;
pub(crate) use sse::*;
