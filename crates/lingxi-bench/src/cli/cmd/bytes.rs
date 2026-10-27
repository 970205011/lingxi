//! `bytes <artifact> [top_n]` — payload 字节构成统计（从 main.rs 搬出，未改逻辑）。

use std::collections::BTreeMap;

use anyhow::Result;
use lingxi_artifact::Artifact;

fn categorize(name: &str) -> &'static str {
    if name == "text/token_embedding" {
        return "embedding";
    }
    if name == "text/output_head" {
        return "output_head";
    }
    if name.starts_with("text/ple/") {
        return "ple";
    }
    if name.starts_with("mtp/") {
        return "mtp";
    }
    if name.ends_with("/routed_gate_up") || name.ends_with("/routed_down") {
        return "moe_routed_experts";
    }
    if name.ends_with("/router_shared_gate") {
        return "moe_router";
    }
    if name.ends_with("/shared_gate_up") || name.ends_with("/shared_down") {
        return "moe_shared";
    }
    if name.contains("/indexer/") {
        return "qsa_indexer";
    }
    if name.contains("/attention/") {
        return "qsa_attention";
    }
    if name.contains("/gdn/") {
        return "gdn";
    }
    if name.ends_with("hc_norm")
        || name.ends_with("block_inject")
        || name.ends_with("mix_down")
        || name.ends_with("mix_up")
    {
        return "gr";
    }
    "other"
}

/// 把 canonical 名归到所属层（`text/layers/N`、`mtp/...`、`text/ple`、`top`）。
fn layer_of(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("text/layers/") {
        if let Some(idx) = rest.split('/').next() {
            return format!("text/layers/{idx}");
        }
    }
    if name.starts_with("mtp/") {
        return "mtp".to_string();
    }
    if name.starts_with("text/ple/") {
        return "text/ple".to_string();
    }
    "top".to_string()
}

pub fn run(args: &[String]) -> Result<()> {
    // 用法: bytes <artifact> [top_n]
    // 统计 payload 字节构成，用于部署内存策略（什么常驻 VRAM / 什么流式读盘）。
    let path = args.get(2).expect("用法: lingxi-bench bytes <artifact.ninfer> [top_n]");
    let top_n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let art = Artifact::open(path)?;
    let catalog = lingxi_model::catalog_from_artifact(&art);

    let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
    let mut total = 0u64;
    let mut by_format: BTreeMap<String, (u64, usize)> = BTreeMap::new();
    let mut by_cat: BTreeMap<String, (u64, usize)> = BTreeMap::new();
    let mut by_layer: BTreeMap<String, u64> = BTreeMap::new();
    let mut tensors: Vec<(u64, String)> = Vec::new();

    for (_, info) in catalog.iter() {
        let b = info.storage_bytes();
        total += b;
        let f = by_format.entry(info.format.name().to_string()).or_insert((0, 0));
        f.0 += b;
        f.1 += 1;

        let cat = categorize(&info.name);
        let c = by_cat.entry(cat.to_string()).or_insert((0, 0));
        c.0 += b;
        c.1 += 1;

        *by_layer.entry(layer_of(&info.name)).or_default() += b;
        tensors.push((b, info.name.clone()));
    }

    println!("总 payload: {:.2} GiB ({total} bytes)", gb(total));
    println!("\n== 按量化格式 ==");
    for (k, (b, n)) in &by_format {
        println!("  {k:<16} {:>8.2} GiB  ({n} 个)", gb(*b));
    }
    println!("\n== 按类别 ==");
    let mut cats: Vec<_> = by_cat.iter().collect();
    cats.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (k, (b, n)) in cats {
        println!(
            "  {k:<24} {:>8.2} GiB  {:>5.1}%  ({n} 个)",
            gb(*b),
            100.0 * *b as f64 / total as f64
        );
    }
    println!("\n== 按层（text 主干 + mtp）==");
    let mut layers: Vec<_> = by_layer.iter().collect();
    layers.sort_by(|a, b| b.1.cmp(a.1));
    for (k, b) in layers.iter().take(55) {
        println!("  {k:<28} {:>8.2} GiB", gb(**b));
    }
    // text 48 层是否等大（决定能否共享一套“单层常驻”方案）
    let text_layers: Vec<u64> = (0..48)
        .filter_map(|i| by_layer.get(&format!("text/layers/{i}")).copied())
        .collect();
    if !text_layers.is_empty() {
        let mn = *text_layers.iter().min().unwrap();
        let mx = *text_layers.iter().max().unwrap();
        println!(
            "  text 主干 48 层：min {:.2} GiB / max {:.2} GiB（差 {:.2} GiB），合计 {:.2} GiB",
            gb(mn),
            gb(mx),
            gb(mx - mn),
            gb(text_layers.iter().sum())
        );
    }
    println!("\n== 最大的 {top_n} 个张量 ==");
    tensors.sort_by(|a, b| b.0.cmp(&a.0));
    for (b, name) in tensors.iter().take(top_n) {
        println!("  {:>8.2} GiB  {name}", gb(*b));
    }
    Ok(())
}
