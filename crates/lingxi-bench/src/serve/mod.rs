//! HTTP serve 入口与请求处理。R4 拆分（2026-10-21）：原 3152 行单文件 → 4 段——
//! - [`cache`]：前缀缓存 + 检查点存储（PrefillCache/CkptStore/账本/保存分派）；
//! - [`sessions`]：会话表（Sessions/Sunk/LiveSeq，sid 路由与空闲记账）；
//! - [`proto`]：协议层（OpenAI chat 请求解析/build_prompt/SSE/工具块/StopFilter/Detok）；
//! - [`engine`]：引擎闭环（run/run_concurrent/engine_loop/Gen 步进/conn_loop）。
//! `mod.rs` 只做声明与再导出——**对外 `serve::run`/`serve::ServeOpts` 路径不变**。

pub(crate) mod cache;
pub(crate) mod engine;
pub(crate) mod proto;
pub(crate) mod sessions;

pub use engine::{run, ServeOpts};


