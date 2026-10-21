//! `lingxi-server` — L4 服务层（axum / OpenAI 兼容 / SSE）。M3 落地。
//! 职责边界：**不做计算、不做调度决策。** 铁律：改 server 绝不重编 kernel。

use lingxi_ids::ModelSpec;

fn main() -> anyhow::Result<()> {
    let model = lingxi_model::FlashNextModel::new();
    println!("lingxi-server 骨架 — model = {}", lingxi_model::MODEL_NAME);
    println!("dims = {:?}", model.dims());
    println!("HTTP 服务（/v1/chat/completions · SSE）将在 M3 接入。");
    Ok(())
}
