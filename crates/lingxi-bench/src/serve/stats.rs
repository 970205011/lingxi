//! Lightweight live stats for the optional `/dashboard` panel.
//!
//! Only counters and hardware numbers — never prompt or output content.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

struct Stats {
    start: OnceLock<Instant>,
    requests: AtomicU64,
    gen_tokens: AtomicU64,
    gen_ms: AtomicU64,
    last_tps_milli: AtomicU64,
    prefix_hits: AtomicU64,
    expert_hits: AtomicU64,
    expert_misses: AtomicU64,
    spec_rounds: AtomicU64,
    spec_accepted: AtomicU64,
}

static S: Stats = Stats {
    start: OnceLock::new(),
    requests: AtomicU64::new(0),
    gen_tokens: AtomicU64::new(0),
    gen_ms: AtomicU64::new(0),
    last_tps_milli: AtomicU64::new(0),
    prefix_hits: AtomicU64::new(0),
    expert_hits: AtomicU64::new(0),
    expert_misses: AtomicU64::new(0),
    spec_rounds: AtomicU64::new(0),
    spec_accepted: AtomicU64::new(0),
};

pub(crate) fn init() {
    let _ = S.start.set(Instant::now());
}

/// 面板开关：`LINGXI_NO_DASHBOARD=1` 时 `/stats` + `/dashboard` 路由关闭（404），
/// 计数仍照记（开销可忽略），重新开面板不需要重启引擎之外的操作。
pub(crate) fn disabled() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("LINGXI_NO_DASHBOARD").map(|v| v == "1").unwrap_or(false))
}

pub(crate) fn record(
    gen_tokens: usize,
    gen_secs: f64,
    prefix_hit: bool,
    expert_hit: u64,
    expert_miss: u64,
    spec_rounds: u64,
    spec_accepted: u64,
) {
    S.requests.fetch_add(1, Ordering::Relaxed);
    S.gen_tokens.fetch_add(gen_tokens as u64, Ordering::Relaxed);
    let ms = (gen_secs.max(0.0) * 1000.0).round() as u64;
    S.gen_ms.fetch_add(ms, Ordering::Relaxed);
    if gen_secs > 0.0 {
        S.last_tps_milli
            .store((gen_tokens as f64 / gen_secs * 1000.0).round() as u64, Ordering::Relaxed);
    }
    if prefix_hit {
        S.prefix_hits.fetch_add(1, Ordering::Relaxed);
    }
    S.expert_hits.fetch_add(expert_hit, Ordering::Relaxed);
    S.expert_misses.fetch_add(expert_miss, Ordering::Relaxed);
    S.spec_rounds.fetch_add(spec_rounds, Ordering::Relaxed);
    S.spec_accepted.fetch_add(spec_accepted, Ordering::Relaxed);
}

fn pct(hit: u64, miss: u64) -> f64 {
    let total = hit + miss;
    if total == 0 { 0.0 } else { hit as f64 * 100.0 / total as f64 }
}

fn gpu_snapshot() -> Vec<serde_json::Value> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let mut it = line.split(',').map(|s| s.trim());
            let index: u32 = it.next()?.parse().ok()?;
            let name = it.next()?.to_string();
            let used: u64 = it.next()?.parse().ok()?;
            let total: u64 = it.next()?.parse().ok()?;
            let util: u64 = it.next()?.parse().ok()?;
            Some(serde_json::json!({
                "index": index,
                "name": name,
                "used_mib": used,
                "total_mib": total,
                "util_pct": util,
            }))
        })
        .collect()
}

pub(crate) fn json() -> String {
    let uptime = S.start.get().map(|t| t.elapsed().as_secs()).unwrap_or(0);
    let gen_ms = S.gen_ms.load(Ordering::Relaxed);
    let gen_tokens = S.gen_tokens.load(Ordering::Relaxed);
    let avg_tps = if gen_ms == 0 { 0.0 } else { gen_tokens as f64 * 1000.0 / gen_ms as f64 };
    let last_tps = S.last_tps_milli.load(Ordering::Relaxed) as f64 / 1000.0;
    let eh = S.expert_hits.load(Ordering::Relaxed);
    let em = S.expert_misses.load(Ordering::Relaxed);
    let rounds = S.spec_rounds.load(Ordering::Relaxed);
    let accepted = S.spec_accepted.load(Ordering::Relaxed);
    let spec_rate = if rounds == 0 { 0.0 } else { accepted as f64 * 100.0 / rounds as f64 };
    let body = serde_json::json!({
        "ok": true,
        "service": "lingxi",
        "uptime_secs": uptime,
        "requests": S.requests.load(Ordering::Relaxed),
        "gen_tokens": gen_tokens,
        "avg_tok_s": (avg_tps * 10.0).round() / 10.0,
        "last_tok_s": (last_tps * 10.0).round() / 10.0,
        "prefix_hit_pct": (pct(S.prefix_hits.load(Ordering::Relaxed), S.requests.load(Ordering::Relaxed) - S.prefix_hits.load(Ordering::Relaxed)) * 10.0).round() / 10.0,
        "expert_hit_pct": (pct(eh, em) * 10.0).round() / 10.0,
        "spec_accept_pct": (spec_rate * 10.0).round() / 10.0,
        "gpus": gpu_snapshot(),
    });
    body.to_string()
}

pub(crate) fn dashboard() -> &'static str {
    include_str!("dashboard.html")
}
