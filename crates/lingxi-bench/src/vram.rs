//! 显存预算：自动槽数推算（`autotune_arena_slots`）与 `vram-budget` 子命令。
//!
//! 从 main.rs 搬出（CODE_MAP Phase 1 刀1，只搬家不改行为）。

use anyhow::Result;
use lingxi_artifact::expert_arena::BlobFormat;
use lingxi_artifact::Artifact;

/// `vram-budget <artifact> [arena] [split_lo] [kv_cap]`
/// 只读 catalog/arena 与设备总量，不加载权重、不分配专家槽。
pub fn cmd_vram_budget(args: &[String]) -> Result<()> {
    let path = args
        .get(2)
        .expect("用法: vram-budget <artifact> [arena] [split_lo] [kv_cap]");
    let arena = args
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("/dev/shm/qwen4_exp.expert-arena.mix64.bin");
    let split_lo: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(24);
    let kv_cap: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(32768);
    let art = Artifact::open(path)?;
    println!("artifact={path}\narena={arena}\nsplit_lo={split_lo} kv_cap={kv_cap}");
    for dev in 0..2 {
        let _ctx = cudarc::driver::CudaContext::new(dev)?;
        match autotune_arena_slots(&art, arena, dev, split_lo, kv_cap) {
            Ok(Some((slots, ledger))) => println!("{ledger}\n  dev{dev}: slots={slots}"),
            Ok(None) => println!("dev{dev}: autotune disabled"),
            Err(e) => println!("dev{dev}: ERROR {e:#}"),
        }
    }
    Ok(())
}

pub fn autotune_arena_slots(
    art: &Artifact,
    arena_path: &str,
    dev: usize,
    split_lo: u32,
    kv_cap: usize,
) -> Result<Option<(usize, String)>> {
    use lingxi_ids::TensorRole;
    if std::env::var("LINGXI_NO_VRAM_AUTOTUNE").is_ok() {
        return Ok(None);
    }
    let arena = lingxi_artifact::expert_arena::ExpertArena::open(arena_path)?;
    let n_layers = arena.n_layers;
    let (lo, hi) = if dev == 0 {
        (0, split_lo.min(n_layers))
    } else {
        (split_lo.min(n_layers), n_layers)
    };
    // 按**每个 blob 自己的格式**统计本卡服务区间的专家数与平均字节数。
    // 混档 arena 不能再用缺省 Q4 blob 估算——那会把 Q3 槽也按 Q4 大小算，
    // 白白少算几千个槽（2026-10-22 修正）。
    let mut n_q4 = 0usize;
    let mut n_q3 = 0usize;
    let mut q4_bytes = 0u64;
    let mut q3_bytes = 0u64;
    for l in lo..hi {
        for e in 0..arena.n_experts {
            if let Some(b) = arena.routed(l, e)? {
                let sz = (b.gu.len() + b.down.len()) as u64;
                match b.format {
                    BlobFormat::Q4G64F16S => {
                        n_q4 += 1;
                        q4_bytes += sz;
                    }
                    BlobFormat::Q3G128F16S => {
                        n_q3 += 1;
                        q3_bytes += sz;
                    }
                    _ => {}
                }
            }
        }
    }
    let blob_q4 = if n_q4 > 0 {
        q4_bytes / n_q4 as u64
    } else {
        arena.gu_bytes + arena.down_bytes
    };
    let blob_q3 = if n_q3 > 0 { q3_bytes / n_q3 as u64 } else { 0 };
    let mixed = n_q4 > 0 && n_q3 > 0;
    let catalog = lingxi_model::catalog_from_artifact(art);
    let mut w = [0u64; 2];
    for (_tid, info) in catalog.iter() {
        // 专家路由权重：Q4 arena 路径下不进显存（bf16 原档留在 host mmap）。
        if matches!(info.role, TensorRole::MoeRoutedGateUp | TensorRole::MoeRoutedDown) {
            continue;
        }
        let gpu = info
            .name
            .split("layers/")
            .nth(1)
            .and_then(|rest| rest.split('/').next())
            .and_then(|s| s.parse::<u32>().ok())
            .map_or(0usize, |l| if l >= split_lo { 1 } else { 0 });
        w[gpu] += info.storage_bytes();
    }
    // P2 分页：KV 设备侧只开常驻窗口（`LINGXI_QSA_WIN_TOKENS`，默认 32K token）——
    // 预算按窗口口径算，省下的全额 KV 直接变成专家槽。（indexer 的 rk/pooled 是
    // 逻辑缓冲但懒增长到 len，不进预算。）
    let cap = if lingxi_model::flag("LINGXI_QSA_PAGED") {
        let win: usize = std::env::var("LINGXI_QSA_WIN_TOKENS")
            .ok().and_then(|s| s.parse().ok()).filter(|&n| n >= 1024)
            .unwrap_or(32768);
        kv_cap.min(win)
    } else {
        kv_cap // 调用方给 KV 池的实际容量（serve=serve_ctx+528，talk=rope_len+8）
    };
    const QSA_PER_DEV: usize = 6; // 12 层 full-attn 双卡各半
    const GDN_PER_DEV: usize = 18; // 36 层 linear-attn 双卡各半
    let kv_per_token_layer: u64 = if lingxi_model::flag("LINGXI_KV_K8V4") { 1184 } else { 2368 };
    let kv = cap as u64 * kv_per_token_layer * QSA_PER_DEV as u64;
    let gdn = 3_220_000u64 * GDN_PER_DEV as u64; // S 3.1MB + 卷积历史双缓冲 ~3.22MB/层
    let reserve: u64 = if dev == 0 { 3 << 30 } else { 2500 << 20 }; // scratch/图/投机缓冲
    let margin: u64 = 1 << 30; // 驱动/碎片边距
    let dev_raw = cudarc::driver::result::device::get(dev as i32)?;
    let total = unsafe { cudarc::driver::result::device::total_mem(dev_raw)? } as u64;
    let budget = total.saturating_sub(margin);
    let fixed = w[dev] + kv + gdn + reserve;
    let remain = budget.saturating_sub(fixed);
    let gib = |b: u64| b as f64 / (1 << 30) as f64;
    let (slots, ledger) = if mixed {
        let q4_slots = n_q4.min((remain / blob_q4) as usize);
        let remain2 = remain.saturating_sub(q4_slots as u64 * blob_q4);
        let q3_slots = n_q3.min((remain2 / blob_q3) as usize);
        let slots = q4_slots + q3_slots;
        let used = q4_slots as u64 * blob_q4 + q3_slots as u64 * blob_q3;
        let all = q4_slots == n_q4 && q3_slots == n_q3;
        let ledger = format!(
            "[vram-budget] GPU{dev}: 总量 {:.1} − 边距 {:.1} − 权重 {:.1} − KV {:.2} \
             − GDN {:.2} − 预留 {:.1} = 余 {:.1} GiB | Q4 {}/{} (blob {:.2}MB) + \
             Q3 {}/{} (blob {:.2}MB) ⇒ 槽 {}，占用 {:.1} GiB{}",
            gib(total), gib(margin), gib(w[dev]), gib(kv), gib(gdn), gib(reserve),
            gib(remain),
            q4_slots, n_q4, blob_q4 as f64 / (1 << 20) as f64,
            q3_slots, n_q3, blob_q3 as f64 / (1 << 20) as f64,
            slots, gib(used),
            if all { "（全部专家可驻留）" } else { "" },
        );
        (slots, ledger)
    } else {
        let blob = if n_q4 > 0 { blob_q4 } else if n_q3 > 0 { blob_q3 } else { arena.gu_bytes + arena.down_bytes };
        let slots = (remain / blob) as usize;
        let ledger = format!(
            "[vram-budget] GPU{dev}: 总量 {:.1} − 边距 {:.1} − 权重 {:.1} − KV {:.2} \
             − GDN {:.2} − 预留 {:.1} = 余 {:.1} GiB ⇒ 槽 {} × blob {:.2}MB",
            gib(total), gib(margin), gib(w[dev]), gib(kv), gib(gdn), gib(reserve),
            gib(remain), slots, blob as f64 / (1 << 20) as f64,
        );
        (slots, ledger)
    };
    if slots == 0 {
        anyhow::bail!(
            "{ledger}\nautotune: GPU{dev} 预算连 1 个专家槽都装不下。降 LINGXI_SERVE_MAX_CTX、\
             显式给 LINGXI_ARENA_SLOTS/LINGXI_LAYER_SPLIT_SLOTS，或检查权重分类是否漏了新角色。"
        );
    }
    Ok(Some((slots, ledger)))
}
