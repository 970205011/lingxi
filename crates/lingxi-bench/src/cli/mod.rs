//! CLI 骨架（CODE_MAP Phase 1）：子命令入口从 main.rs 按域拆到 `cmd/*`。
//! 原则：只搬家不改行为；共享装配助手放 `common`。

pub mod cmd;
pub mod common;
