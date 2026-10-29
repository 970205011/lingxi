//! 专家 arena 直读源：`Q4EARN1` arena → 设备 Q4 槽缓存 →（视图解量化）→ 紧凑 bf16 scratch。
//!
//! **归属**：arena 容器与"每 blob 格式字"来自 ninfer-dev（灵犀分支）；设计参考
//! Strata（`Niko1221/Strata`, MIT）的 pinned arena + 常驻槽位思想。
//!
//! **磁盘纪律（事故见 project_memory / 磁盘归属 README）**：默认**不 pin**。
//! 整段 `cudaHostRegister` 会逼内核把整个 arena major-fault 进内存 = 对机械盘持续
//! 顺序读，15+ 分钟不停会触发该盘固件假死。惰性缺页天然分块（每专家 ~2.6 MB）。
//!
//! **两层缓存**：
//!   1. **设备 Q4 槽缓存**（`slot`）：热专家的 Q4 blob 常驻显存，命中即跳过 H2D——
//!      这是 decode 的主瓶颈（每 token 480 专家 × 2.6 MB ≈ 1.25 GB 过 PCIe）；
//!   2. host 侧 pinned arena（可选）：`LINGXI_ARENA_PIN_GIB`，源在 RAM 时注册瞬时不触发磁盘读
//!      （介质判定在 `pin`）。
//!
//! CODE_MAP Phase 2 刀P2-1：文件 → 目录模块（slot/pin/source），只搬家不改行为。

mod pin;
mod prefetch;
mod slot;
mod source;

pub use source::ArenaExpertSource;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// 命中/未命中 = 设备槽缓存；unique = 本请求实际路由到多少个不同的 (层, 专家)。
pub(crate) static EXPERT_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static EXPERT_MISSES: AtomicU64 = AtomicU64::new(0);
static EXPERT_TRACK: AtomicBool = AtomicBool::new(false);
pub(crate) const EXPERT_WORDS: usize = (48 * 512) / 64;

fn expert_seen() -> &'static [AtomicU64] {
    static SEEN: std::sync::OnceLock<Vec<AtomicU64>> = std::sync::OnceLock::new();
    SEEN.get_or_init(|| (0..EXPERT_WORDS).map(|_| AtomicU64::new(0)).collect())
}

/// 请求开始时清零（serve 每请求调用一次）。
pub fn expert_track_reset() {
    EXPERT_HITS.store(0, Ordering::Relaxed);
    EXPERT_MISSES.store(0, Ordering::Relaxed);
    for w in expert_seen() {
        w.store(0, Ordering::Relaxed);
    }
    EXPERT_TRACK.store(true, Ordering::Relaxed);
}

/// 请求结束：返回 `(命中, 未命中, 去重专家数)` 并停止记录。
pub fn expert_track_finish() -> (u64, u64, usize) {
    EXPERT_TRACK.store(false, Ordering::Relaxed);
    let hits = EXPERT_HITS.load(Ordering::Relaxed);
    let misses = EXPERT_MISSES.load(Ordering::Relaxed);
    let unique = expert_seen()
        .iter()
        .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
        .sum();
    (hits, misses, unique)
}

#[inline]
pub(crate) fn mark_expert(layer: u32, e: u32) {
    if !EXPERT_TRACK.load(Ordering::Relaxed) {
        return;
    }
    let idx = layer as usize * 512 + e as usize;
    if idx < 48 * 512 {
        expert_seen()[idx / 64].fetch_or(1u64 << (idx % 64), Ordering::Relaxed);
    }
}

