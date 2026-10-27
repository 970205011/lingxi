//! `vram-budget` 入口薄封装：预算逻辑本体在 crate 根 `vram.rs`（CODE_MAP 刀1）。

use anyhow::Result;

pub fn run(args: &[String]) -> Result<()> {
    crate::vram::cmd_vram_budget(args)
}
