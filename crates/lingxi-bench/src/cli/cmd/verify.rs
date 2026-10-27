//! `verify <artifact.ninfer>` — 结构自检 + 目录统计（从 main.rs 搬出，未改逻辑）。

use std::collections::BTreeMap;

use anyhow::Result;
use lingxi_artifact::Artifact;

pub fn run(args: &[String]) -> Result<()> {
    let path = args.get(2).expect("用法: lingxi-bench verify <artifact.ninfer>");
    let t0 = std::time::Instant::now();
    let art = Artifact::open(path)?;
    art.verify()?;
    let catalog = lingxi_model::catalog_from_artifact(&art);

    let mut formats: BTreeMap<String, usize> = BTreeMap::new();
    let mut layouts: BTreeMap<String, usize> = BTreeMap::new();
    let mut roles: BTreeMap<String, usize> = BTreeMap::new();
    let mut total = 0u64;
    for (_, info) in catalog.iter() {
        *formats.entry(info.format.name().to_string()).or_default() += 1;
        *layouts.entry(info.layout.name().to_string()).or_default() += 1;
        *roles.entry(format!("{:?}", info.role)).or_default() += 1;
        total += info.storage_bytes();
    }
    println!(
        "OK: identity={}/{} objects={} (tensor {} + resource {}) payload={:.2}GB 用时 {:.2}s",
        art.identity.model_id,
        art.identity.weights_id,
        art.objects.len(),
        art.tensors().count(),
        art.resources().count(),
        total as f64 / 1e9,
        t0.elapsed().as_secs_f64(),
    );
    println!("formats: {formats:?}");
    println!("layouts: {layouts:?}");
    let other = roles.get("Other").copied().unwrap_or(0);
    println!("roles: {roles:?}  (Other={other})");
    Ok(())
}
