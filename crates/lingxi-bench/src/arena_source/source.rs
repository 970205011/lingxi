//! ArenaExpertSource 主逻辑：arena → 设备槽缓存 → 视图解量化 → 紧凑 scratch
//!（CODE_MAP Phase 2 刀P2-1 从 arena_source.rs 搬入，只搬家不改行为；
//! prewarm/重钉/mix 的方法实现仍在本文件，按方法级拆分是 P2 后续刀）。

use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_artifact::expert_arena::{matrix_layout, BlobFormat, ExpertArena};
use lingxi_kernels::Kernels;

use super::pin::backing_fs_type;
use super::slot::SlotCache;
use super::{mark_expert, EXPERT_HITS, EXPERT_MISSES};

pub struct ArenaExpertSource<'a> {
    k: &'a Kernels,
    arena: ExpertArena,
    /// **可写别名映射**（`LINGXI_ARENA_PIN_DMA=1`，见）：与 `arena` 是**同一份 tmpfs 页**，
    /// 只是 VMA 可写 ⇒ 才能用 `PORTABLE` 做 DMA 注册。`None` = 老行为（READ_ONLY 注册）。
    alias: Option<memmap2::MmapMut>,
    cache: UnsafeCell<SlotCache>,
    /// **混档 arena**（Q4G64+Q3G128 并存，6.2-2）的第二种格式槽缓存。`None` = 单格式 arena。
    /// v1 设计：两 cache 各保**等大**槽（Q4 槽 2613248 / Q3 槽 1921024），load_hits 按
    /// blob 格式分流；Q4 直读/A2 设备路由/钉表在混合态**全部禁用**（回退 host 路径 +
    /// 逐专家解量化），先保正确性与质量判据，快路径双视图留 v2。
    cache_alt: Option<UnsafeCell<SlotCache>>,
    /// 次格式（混合态 = Q3G128F16S；单格式 = None）
    fmt2: Option<BlobFormat>,
    /// 主/次格式各自的槽步长（字节）
    blob_pri: usize,
    blob_alt: usize,
    /// arena 是否混格式（Q4G64 与 Q3G128 并存）
    mixed: bool,
    scratch: UnsafeCell<Scratch>,
    /// `LINGXI_ARENA_HIST=<path>`：每 `(层, 专家)` 的**请求计数**（含命中与未命中）。
    /// 为什么要它：P1「静态钉住每层最热的 N 个专家」必须先知道**路由有多集中**、
    /// 以及**每层该钉多少个**才够覆盖 x% 的请求。凭感觉定 N 会要么钉太多（挤掉冷尾）、
    /// 要么钉太少（白改）。每 480 次 `load_hits` 覆写一次落盘（进程被杀也不会丢尽）。
    hist: UnsafeCell<HashMap<(u32, u32), u64>>,
    /// **本实例**的 `tick_stats` 计数（2026-10-18 修）。
    /// 以前用函数内 `static`：双卡时两个实例共享一个计数器，而 480 是 48（每 token 层数）
    /// 的整数倍 ⇒ 每次 `n % 480 == 0` 落在同一个实例（层 47 = GPU1 那份）⇒ **GPU0 的
    /// 0..23 层永远不落盘**，采出来的直方图只有一半层（实测 24..47）。改成按实例计数。
    ticks: std::cell::Cell<u64>,
    /// `LINGXI_ARENA_REPIN=<窗口>`：**在线自适应重钉**的窗口（单位 = `load_hits` 次数）。
    /// 0 = 关（行为与以前完全一致）。>0 = 每累计这么多次请求，就按"最近热度"重钉一次
    /// （进 top-N 的搬进来钉住、掉出的解钉还给 LRU），并把热度计数减半。
    /// 为什么需要：钉表是**离线静态**的（`LINGXI_ARENA_PREWARM` 用一份历史直方图），
    /// 实测同一配置 n=80 有 46 tok/s、**n=300 掉到 29.6**（生成超出采集话题后覆盖率就掉）。
    repin_window: u64,
    /// 重钉时每层保留的 top-N（复用 `LINGXI_ARENA_PREWARM_N`）
    repin_n: usize,
    /// 只给 `layers` 区间内的层分槽 / 钉住（默认 `0..48`；配合 `LINGXI_MOE_SPLIT` 用：
    /// GPU0 那份拿 `0..split_lo`、GPU1 那份拿 `split_lo..48`）
    layers: std::ops::Range<u32>,
    /// ㊺ A2 设备路由：`[层, E]` 的槽位表（slot+1、0=未缓存）。路由核在设备端直查它，
    /// 省掉每层"sync → host top-k → host 查槽 → staging"的整条决策链。
    slot_map: CudaSlice<i32>,
    /// 混档次格式（Q3）cache 的 `[层, E]` 槽位表；单格式时全 0（不读）。
    slot_map_alt: CudaSlice<i32>,
    /// 混档每 (层, 专家) 格式表：0 = 主（Q4）、1 = 次（Q3）。**静态**，启动刷一次。
    fmt_dev: CudaSlice<i32>,
    /// 上表逐层上传用的**锁页暂存**（每层一块：异步 H2D 的源不能被下一层的写覆掉）
    slot_map_pins: UnsafeCell<Vec<cudarc::driver::PinnedHostSlice<i32>>>,
    /// 次格式槽位表的逐层锁页暂存
    slot_map_alt_pins: UnsafeCell<Vec<cudarc::driver::PinnedHostSlice<i32>>>,
    /// Q4 段几何常量（与 `Q4Slots` 同源）
    q4c: lingxi_model::moe::Q4Consts,
    /// 次格式（混档 = Q3）段几何常量
    q4c_alt: Option<lingxi_model::moe::Q4Consts>,
    /// A2 设备路由可用（主格式是 Q4G64/Q3G128；混档**也**可用——双视图 + 格式表）
    q4_ok: bool,
    /// **簿记串行锁**（2026-10-25，双会话并发 `LINGXI_TALK_SPEC_BATCH`）：SlotCache 的
    /// victim/insert/touch 是 host 侧裸可变（UnsafeCell），两个线程在同一卡上同时 miss
    /// 会双发同一槽 ⇒ 权重互相覆写、token 流静默分叉。锁内**不含 GPU 等待**（miss 的
    /// blob DMA 只是入队），命中路径持锁 ~µs。scratch 双缓冲的"隔层必同步"余量是
    /// **每线程路径内**的论证（同一条 k0 FIFO + 每层路由同步在各自路径上排干），
    /// 与并发无关 ⇒ 不需要为 scratch 加锁。
    op_lock: std::sync::Mutex<()>,
}

/// 全部按需扩容：decode 每层槽数固定，正常只分配一次；
/// prefill 的命中专家数可达 `rows * top_k`，必须能长大。
struct Scratch {
    gu: Option<CudaSlice<bf16>>,
    dn: Option<CudaSlice<bf16>>,
    /// 槽号设备数组（批量解量化用）。**两份轮换**（见 `ids_phase` 与下面的说明）。
    slot_ids: [Option<CudaSlice<i32>>; 2],
    /// 混档 Q3 cache 的槽号设备数组（6.2-2 Stage B；同样双缓冲）。
    slot_ids_alt: [Option<CudaSlice<i32>>; 2],
    /// 槽号的**锁页**宿主暂存（可分页源会让 `cuMemcpyHtoDAsync` 退化成同步）。同样**两份轮换**。
    ids_pin: [Option<cudarc::driver::PinnedHostSlice<i32>>; 2],
    /// 混档 Q3 cache 的锁页暂存。
    ids_pin_alt: [Option<cudarc::driver::PinnedHostSlice<i32>>; 2],
    /// 轮换相位（每次 `load_hits` 取反）。
    ///
    /// ★ **为什么必须双缓冲**：
    /// `slot_ids` 被上一个 `load_hits` **异步 H2D** 上去后，**上一层的内核还在异步读它**；
    /// 单缓冲时下一次 `load_hits` 会直接把内容覆写成新一层的槽号 ⇒ 上一层读到**别的层的槽号**
    /// ⇒ 取错专家、token 流静默分叉（实测：投机路径第 103 个 token 起与纯贪心不同，而同权重
    /// 的 t=1 路径完全正常）。**只加"设备侧等 event"治不好**（实测 `MOE_OVERLAP` 无效），
    /// 加 host 同步能治好 ⇒ 是主机侧覆写时序。双缓冲给出一整层的余量，且**不付同步的钱**。
    ids_phase: u8,
    /// Q4 直读路径下 `ExpertHits::gate_up/down` 的占位（1 元素）
    stub_gu: Option<CudaSlice<bf16>>,
    stub_dn: Option<CudaSlice<bf16>>,
}

unsafe impl Send for ArenaExpertSource<'_> {}
unsafe impl Sync for ArenaExpertSource<'_> {}


impl<'a> ArenaExpertSource<'a> {
    /// `layers`：本份 arena **只服务**的层区间（钉表/重钉都只看它）。
    /// 开 `LINGXI_MOE_SPLIT` 时 GPU0 那份给 `0..split_lo`、GPU1 那份给 `split_lo..48` ——
    /// 不这么切的话，GPU1 会照读同一份钉表、把不相干层的槽钉死（灾难）。
    /// `prewarm_off=true`：**完全不钉**（纯 LRU）；`layers` 已经能隔离范围，所以现在基本不用。
    pub fn new(
        k: &'a Kernels,
        arena_path: &str,
        slots: usize,
        prewarm_off: bool,
        layers: std::ops::Range<u32>,
    ) -> Result<Self> {
        let arena = ExpertArena::open(arena_path)?;
        // ── pin 策略：**按介质判定**（2026-10-16 改；事故见 project_memory）──────────
        // 旧策略"默认 0 = 不 pin + 允许给任意 GiB 上限"有两个问题（都已实测）：
        // 1. 分开源 ⇒ 驱动只能走内部 staging 的**同步**拷贝 ⇒ 慢 28% 且 p50 掉到 8.8 tok/s
        // （实测 34.4 ms/29 tok/s vs 整段锁页 26.7 ms/37.4 tok/s，p50 29 ms/34.5 tok/s）；
        // 2. **部分 pin 是坏的** ⇒ `LINGXI_ARENA_PIN_GIB=16` 会在 decode 第 15 个 token 崩
        // `arena blob H2D: CUDA_ERROR_INVALID_VALUE`（同一 mmap 只注册一段时，
        // 落到注册区间之外的拷贝源会被驱动拒）。
        // 新策略：
        // * 介质是 **RAM（tmpfs/ramfs）** ⇒ 注册不触发任何磁盘读（前提：swap 关）
        // ⇒ **允许且默认整段 pin**（`LINGXI_ARENA_PIN_GIB` 不设 = 整段）；
        // * 介质**不是 RAM** ⇒ **硬禁**（2026-10-14 那次整机重启就是机械盘被 60 GiB
        // major-fault 持续顺序读 15 min+ 触发固件假死）——要关 pin 请显式 `=0`；
        // * 只支持"整段或 0"：给中间值直接报错（不再留到运行中段崩）。
        let fs_type = backing_fs_type(arena_path);
        let on_ram = matches!(fs_type.as_deref(), Some("tmpfs") | Some("ramfs"));
        let arena_bytes = arena.data_region().len() as u64;
        let pin_env: Option<u64> = std::env::var("LINGXI_ARENA_PIN_GIB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());
        let want_bytes: u64 = match pin_env {
            Some(0) => 0,
            Some(g) => g.saturating_mul(1 << 30),
            None => {
                if on_ram {
                    arena_bytes
                } else {
                    0
                }
            }
        };
        if want_bytes > 0 && !on_ram {
            bail!(
                "arena 不在 RAM 上（fstype={:?}）：**禁止 pin**——注册会逼内核把整段 arena \
                 major-fault 进内存，对机械盘持续顺序读会触发固件假死（2026-10-14 事故）。\
                 把 arena 放到 /dev/shm 重启，或显式 `LINGXI_ARENA_PIN_GIB=0` 关闭 pin。",
                fs_type
            );
        }
        if want_bytes > 0 && want_bytes < arena_bytes {
            bail!(
                "只支持**整段** pin 或不 pin：请求 {} GiB < arena {:.1} GiB。\
                 部分 pin 已被实测证明会崩（decode 中途 CUDA_ERROR_INVALID_VALUE：\
                 同一 mmap 只注册一段时，区间外的拷贝源被拒）。请给 ≥ arena 的 GiB 数或 0。",
                want_bytes / (1 << 30),
                arena_bytes as f64 / 1073741824.0
            );
        }
        // ★ **DMA 注册**（`LINGXI_ARENA_PIN_DMA=1`，）：先拿一份**可写**别名映射，
        // 再用 `PORTABLE` 注册 ⇒ 从它发起的 H2D 才是 copy-engine DMA（**可重叠**）。
        // 老行为（READ_ONLY 注册）实测：8.6 GB/s 且与 kernel 完全不可重叠。
        let alias = if lingxi_model::flag("LINGXI_ARENA_PIN_DMA") && want_bytes > 0 {
            Some(
                lingxi_artifact::expert_arena::mmap_shared_rw(arena_path)
                    .context("LINGXI_ARENA_PIN_DMA：可写别名映射建立失败")?,
            )
        } else {
            None
        };
        let pinned = want_bytes > 0
            && match alias.as_ref() {
                Some(al) => {
                    let base = (al.as_ptr() as usize + arena.data_offset()) as *const u8;
                    lingxi_kernels::pin_host_dma(base, arena.data_bytes() as usize)
                }
                None => lingxi_kernels::pin_host_readonly(arena.data_ptr(), arena_bytes as usize),
            };
        // ── 格式扫描（6.2-2 混档：Q4G64+Q3G128 并存，2026-10-20） ──
        // 单格式：主格式 = 全 arena 格式（行为与以前完全一致）。
        // 混格式：主 = Q4G64F16S（热专家）、次 = Q3G128F16S（冷专家）；两格式各建一个
        // **等大槽** SlotCache，load_hits 按 blob 格式分流。直读/A2/钉表在混合态禁用（v1）。
        let mut fmt_set: Vec<BlobFormat> = Vec::new();
        for l in layers.start..layers.end.min(arena.n_layers) {
            for e in 0..arena.n_experts {
                match arena.routed(l, e) {
                    Ok(Some(b)) => {
                        if !fmt_set.contains(&b.format) {
                            fmt_set.push(b.format);
                        }
                    }
                    _ => anyhow::bail!("arena 索引缺 routed 条目（layer {l} expert {e}）"),
                }
            }
        }
        let mixed = fmt_set.len() > 1;
        if mixed
            && !(fmt_set.len() == 2
                && fmt_set.contains(&BlobFormat::Q4G64F16S)
                && fmt_set.contains(&BlobFormat::Q3G128F16S))
        {
            anyhow::bail!(
                "arena 混格式暂只支持 Q4G64F16S + Q3G128F16S 两种（实际 {fmt_set:?}）"
            );
        }
        let pri_fmt = if mixed {
            BlobFormat::Q4G64F16S
        } else {
            fmt_set.first().copied().unwrap_or(BlobFormat::Q4G64F16S)
        };
        let fmt2: Option<BlobFormat> = if mixed { Some(BlobFormat::Q3G128F16S) } else { None };
        let lay = |f: BlobFormat, r: u64, c: u64| {
            matrix_layout(f, r, c).map(|l| l.payload_bytes).context("段几何非法")
        };
        let blob_pri = (lay(pri_fmt, 2 * arena.n_ff as u64, arena.n_in as u64)?
            + lay(pri_fmt, arena.n_in as u64, arena.n_ff as u64)?) as usize;
        let blob_alt = match fmt2 {
            Some(f) => (lay(f, 2 * arena.n_ff as u64, arena.n_in as u64)?
                + lay(f, arena.n_in as u64, arena.n_ff as u64)?) as usize,
            None => 0,
        };
        // 混合态槽预算分流：Q4 cache 默认 = 服务层区间内 Q4 专家数（热集即钉集，全给槽），
        // env `LINGXI_ARENA_MIX_Q4_SLOTS` 可显式覆盖；其余全给 Q3 cache（LRU）。
        let (pri_slots, alt_slots) = if mixed {
            let n_q4_range = (layers.start..layers.end.min(arena.n_layers))
                .map(|l| {
                    (0..arena.n_experts)
                        .filter(|&e| {
                            matches!(
                                arena.routed(l, e),
                                Ok(Some(b)) if b.format == BlobFormat::Q4G64F16S
                            )
                        })
                        .count()
                })
                .sum::<usize>();
            let want: usize = std::env::var("LINGXI_ARENA_MIX_Q4_SLOTS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(n_q4_range)
                .min(slots);
            (want, slots - want)
        } else {
            (slots, 0)
        };
        let mut cache = SlotCache::new(
            k,
            if mixed { pri_slots } else { slots },
            blob_pri,
            arena.n_layers as usize,
            arena.n_experts as usize,
        )?;
        // ★ 2026-10-25：预算按"余量吃满"算（ledger 上"占用 = 余"），槽位本身是一次性大块
        // `cudaMalloc` ⇒ 偶发 `CUDA_ERROR_OUT_OF_MEMORY`（实测 200 token 跑里约 1/20 起不来，
        // 栈顶就在 `SlotCache::new`）。`SlotCache::new` 现在 OOM 会逐级退让，这里**回读实际槽数**，
        // 让下游（钉表/ledger/命中统计）用真值而不是请求值。
        let pri_want = pri_slots;
        let pri_slots = cache.n_slots;
        let mut cache_alt = if mixed {
            Some(UnsafeCell::new(SlotCache::new(
                k,
                alt_slots,
                blob_alt,
                arena.n_layers as usize,
                arena.n_experts as usize,
            )?))
        } else {
            None
        };
        let alt_want = alt_slots;
        let alt_slots = match &cache_alt {
            Some(c) => unsafe { (*c.get()).n_slots },
            None => 0,
        };
        if pri_slots != pri_want || alt_slots != alt_want {
            eprintln!(
                "[arena] ⚠️ 槽位分配 OOM，已向下夹紧：Q4 {pri_slots}/{pri_want} + \
                 Q3 {alt_slots}/{alt_want}（预算按「余量吃满」算，属已知紧度）"
            );
        }
        // ── `LINGXI_ARENA_PREWARM=<hist.csv>`：**预热 + 钉住**每层最热的 N 个专家 ──
        // 依据：**工作集 ≈ 槽数**（bench 负载 6774 个
        // (层,专家) / 7000 槽）⇒ 纯 LRU 会抖，稳定命中率只有 93%，而且那 7% 里绝大部分是
        // **首触（compulsory）**——LRU 再聪明也躲不掉"第一次见到它"。
        // 直方图显示每层 top-145 覆盖 99.65% 的请求 ⇒ 把这批**提前搬进显存并钉住**，
        // 把首触也省掉。直方图由 `LINGXI_ARENA_HIST` 在同一批语料上产出。
        // ⚠️ **强依赖工作负载**：JSON/工具类上下文里每层 top-145 只覆盖 85.5%。
        // 不设本开关 = 行为与以前完全一致（零风险）。
        // `layers`：**只钉这个区间内的层**（见 `new` 的参数说明）。
        // 为什么要它：开了 `LINGXI_MOE_SPLIT` 之后，后一半层的专家由 **GPU1 的 arena** 负责，
        // GPU0 这张表就不该再给那些层留槽（否则一半的钉位是白占的），反之亦然。
        // 配合把 `PREWARM_N` 调大（9000/(HI-LO) ≈ 375）⇒ 每层覆盖率从 83.8% 抬到 ~98%。
        let (lo, hi) = (layers.start, layers.end);
        // 重钉时每层保留的 top-N：与预热同口径（`LINGXI_ARENA_PREWARM_N`）
        let repin_n: usize = std::env::var("LINGXI_ARENA_PREWARM_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(145);
        // `prewarm_off=true`（GPU1 的那份 arena 用）：**完全不钉**，纯 LRU。
        // 理由：GPU1 只服务后一半层，若照读同一份钉表会把不相干层的槽钉死（灾难）。
        // 而 GPU1 的槽数给到"接近每层全部专家"时，纯 LRU 的自然命中率就已经很高。
        if !mixed {
            // 单格式：blob 步长 = 主格式几何（与原行为一致；`blob` 名保留以减少 diff）。
            let blob = blob_pri;
            if let Some(csv) = std::env::var("LINGXI_ARENA_PREWARM").ok().filter(|_| !prewarm_off) {
            let n_pin: usize = std::env::var("LINGXI_ARENA_PREWARM_N")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100);
            let txt = std::fs::read_to_string(&csv)
                .with_context(|| format!("PREWARM 读不到直方图 {csv}"))?;
            let mut per: HashMap<u32, Vec<(u64, u32)>> = HashMap::new();
            for line in txt.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let mut it = line.split(',');
                let (Some(l), Some(e), Some(c)) = (it.next(), it.next(), it.next()) else { continue };
                if let (Ok(l), Ok(e), Ok(c)) =
                    (l.parse::<u32>(), e.parse::<u32>(), c.parse::<u64>())
                {
                    if l < lo || l >= hi {
                        continue;
                    }
                    per.entry(l).or_default().push((c, e));
                }
            }
            // ★ **钉哪些槽：两种口径**（`LINGXI_PREWARM_GLOBAL=1` 走全局贪心，默认仍是一刀切）
            //
            // 为什么全局贪心更对：一刀切"每层 top-`n_pin`"是**按层平均**分配槽，
            // 于是 `distinct` 少的层会把槽花在 `count=1` 的长尾上，而 `distinct` 多的层
            // （实测 182~411，差一倍多）的 `count=2~3` 专家反而钉不上。
            // 静态钉表下"最小化总 miss 次数"的最优贪心是：**把全部 (层,专家) 按 count 降序，
            // 取前 `budget` 个** —— 层配额自然浮现，不需要人工拍。
            // 这里 `budget` 取与旧口径**相同**的槽数（`Σ min(distinct_l, n_pin)`），
            // 这样 A/B 只变"钉哪些"、不变"钉多少"，才是干净的对照。
            let global = lingxi_model::flag("LINGXI_PREWARM_GLOBAL");
            let budget: usize = per.values().map(|v| v.len().min(n_pin)).sum();
            let mut want: Vec<(u32, u32)> = Vec::new();
            if global {
                let mut all: Vec<(u64, u32, u32)> = Vec::new();
                for (&l, v) in per.iter() {
                    for &(c, e) in v.iter() {
                        all.push((c, l, e));
                    }
                }
                // count 降序；同 count 按 (层,专家) 升序 ⇒ 完全确定（不依赖 HashMap 迭代序）
                all.sort_unstable_by(|a, b| {
                    b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2)))
                });
                all.truncate(budget);
                want.extend(all.into_iter().map(|(_, l, e)| (l, e)));
            } else {
                for (&l, v) in per.iter() {
                    let mut vv = v.clone();
                    vv.sort_unstable_by(|a, b| b.cmp(a));
                    for &(_, e) in vv.iter().take(n_pin) {
                        want.push((l, e));
                    }
                }
            }
            want.sort_unstable();
            eprintln!(
                "[prewarm] 钉表口径 = {}（钉 {} 个；直方图 distinct 共 {}）",
                if global { "全局 count 降序（LINGXI_PREWARM_GLOBAL=1）" } else { "每层 top-N 一刀切" },
                want.len(),
                per.values().map(|v| v.len()).sum::<usize>()
            );
            // `LINGXI_PREWARM_REVERSE=1`：**只反转槽位映射、不改钉住的集合**。
            // 判决性实验：若 token 流因此改变 ⇒ 专家 GEMV 的结果**依赖槽位号**（内核越界读/写），
            // 是真 bug；若不变 ⇒ bug 在"钉进去的字节"而不是"钉在哪个槽"。
            if lingxi_model::flag("LINGXI_PREWARM_REVERSE") {
                want.reverse();
            }
            let plain = want.len();
            let t0 = std::time::Instant::now();
            // ⚠️ 见 `bind_ctx` 的说明：裸 `result::memcpy_*` 不会自己绑上下文。
            k.ctx.bind_to_thread()?;
            let mut done = 0usize;
            for (i, &(l, e)) in want.iter().enumerate() {
                if i >= cache.n_slots {
                    eprintln!("[prewarm] ⚠️ 槽不够：只预热了 {i}/{plain}（把 PREWARM_N 调小或加 SLOTS）");
                    break;
                }
                let b = arena
                    .routed(l, e)?
                    .with_context(|| format!("预热缺专家 layer {l} expert {e}"))?;
                let sv = cache.slots.slice(i * blob..(i + 1) * blob);
                let (dptr, _g) = sv.device_ptr(&k.stream);
                // 源：优先从**已注册的别名映射**取（同一份页；地址在注册区间内才走 DMA）
                let src = match alias.as_ref() {
                    Some(al) => &al[b.gu.start..b.down.end],
                    None => arena.bytes(&(b.gu.start..b.down.end)),
                };
                // ⚠️ 与 `maybe_repin` 同一条纪律：预热的字节数**必须**正好一个 blob。
                // 缺这条断言时（2026-10-16 排查中发现），一旦 `gu.start..down.end` 不是
                // 恰好 blob 长，就会**静默多拷/少拷**，钉进去的专家权重与懒加载不一致 ⇒
                // token 流在几十个 token 后分叉。这里补上（并且是真检查，不只是打印）。
                if done == 0 {
                    eprintln!(
                        "[prewarm] 首个 blob 长度核对：src.len()={} blob={} gu.start={} down.end={} gu_bytes={} down_bytes={}",
                        src.len(), blob, b.gu.start, b.down.end,
                        arena.gu_bytes, arena.down_bytes
                    );
                }
                anyhow::ensure!(
                    src.len() == blob,
                    "预热字节数不符：src.len()={} blob={}（layer {l} expert {e}，\
                     gu.start={} down.end={}）",
                    src.len(), blob, b.gu.start, b.down.end
                );
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(dptr, src, k.stream.cu_stream())
                        .context("prewarm blob H2D")?;
                }
                drop(_g);
                cache.index.insert((l, e), i);
                cache.rev.insert(i, (l, e));
                cache.flat_set((l, e), i);
                cache.pinned[i] = true;
                cache.pinned_q.push_back(i);
                done += 1;
            }
            cache.next_free = done;
            k.stream.synchronize()?;
            eprintln!(
                "[prewarm] 钉住 {done} 个专家（每层 top-{n_pin}，{:.1} GiB）用时 {:.1}s；LRU 剩 {} 槽给尾巴",
                (done * blob) as f64 / 1073741824.0,
                t0.elapsed().as_secs_f64(),
                cache.n_slots - done
            );
            }
        } else {
            // 混档 v2（2026-10-20）：钉表按**专家自己的格式**分流入两个 cache——
            // Q4 进主 cache、Q3 进次 cache，各用各的槽编号空间/槽步长。
            // 收益与单格式钉表同源：消灭 decode 的首触 DMA miss（实测 miss ≈ 350µs/个）。
            let csv = std::env::var("LINGXI_ARENA_PREWARM").ok().filter(|_| !prewarm_off);
            match csv {
                None => eprintln!(
                    "[prewarm] 混档 arena：未设 LINGXI_ARENA_PREWARM，纯 LRU（Q4 {} 槽 / Q3 {} 槽）",
                    pri_slots, alt_slots
                ),
                Some(csv) => {
            let n_pin: usize = std::env::var("LINGXI_ARENA_PREWARM_N")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100);
            let txt = std::fs::read_to_string(&csv)
                .with_context(|| format!("PREWARM 读不到直方图 {csv}"))?;
            let mut per: HashMap<u32, Vec<(u64, u32)>> = HashMap::new();
            for line in txt.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let mut it = line.split(',');
                let (Some(l), Some(e), Some(c)) = (it.next(), it.next(), it.next()) else { continue };
                if let (Ok(l), Ok(e), Ok(c)) =
                    (l.parse::<u32>(), e.parse::<u32>(), c.parse::<u64>())
                {
                    if l < lo || l >= hi {
                        continue;
                    }
                    per.entry(l).or_default().push((c, e));
                }
            }
            // 与单格式同口径：一刀切 top-N（全局贪心开关同样生效）。
            let global = lingxi_model::flag("LINGXI_PREWARM_GLOBAL");
            let budget: usize = per.values().map(|v| v.len().min(n_pin)).sum();
            let mut want: Vec<(u32, u32)> = Vec::new();
            if global {
                let mut all: Vec<(u64, u32, u32)> = Vec::new();
                for (&l, v) in per.iter() {
                    for &(c, e) in v.iter() {
                        all.push((c, l, e));
                    }
                }
                all.sort_unstable_by(|a, b| {
                    b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2)))
                });
                all.truncate(budget);
                want.extend(all.into_iter().map(|(_, l, e)| (l, e)));
            } else {
                for (&l, v) in per.iter() {
                    let mut vv = v.clone();
                    vv.sort_unstable_by(|a, b| b.cmp(a));
                    for &(_, e) in vv.iter().take(n_pin) {
                        want.push((l, e));
                    }
                }
            }
            want.sort_unstable();
            eprintln!(
                "[prewarm] 混档钉表：钉 {} 个（直方图 distinct 共 {}，Q4 槽 {} / Q3 槽 {}）",
                want.len(),
                per.values().map(|v| v.len()).sum::<usize>(),
                pri_slots, alt_slots
            );
            k.ctx.bind_to_thread()?;
            let mut done = [0usize; 2]; // 0=主(Q4) 1=次(Q3)
            let t0 = std::time::Instant::now();
            for &(l, e) in want.iter() {
                let b = arena
                    .routed(l, e)?
                    .with_context(|| format!("预热缺专家 layer {l} expert {e}"))?;
                // 按**blob 自己的格式**选 cache（与 load_hits 同一逻辑）。
                let which = if Some(b.format) == fmt2 { 1usize } else { 0 };
                let (cache, cblob): (&mut SlotCache, usize) = if which == 1 {
                    (cache_alt.as_mut().unwrap().get_mut(), blob_alt)
                } else {
                    (&mut cache, blob_pri)
                };
                let i = done[which];
                if i >= cache.n_slots {
                    done[which] += 1;
                    if done[which] == cache.n_slots + 1 {
                        eprintln!("[prewarm] ⚠️ {} cache 槽不够：跳过其余钉表项（调小 PREWARM_N 或加 SLOTS）",
                            if which == 1 { "Q3" } else { "Q4" });
                    }
                    continue;
                }
                let src = match alias.as_ref() {
                    Some(al) => &al[b.gu.start..b.down.end],
                    None => arena.bytes(&(b.gu.start..b.down.end)),
                };
                anyhow::ensure!(
                    src.len() == cblob,
                    "预热字节数不符：src.len()={} cblob={}（layer {l} expert {e} fmt={:?}）",
                    src.len(), cblob, b.format
                );
                let sv = cache.slots.slice(i * cblob..(i + 1) * cblob);
                let (dptr, _g) = sv.device_ptr(&k.stream);
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(dptr, src, k.stream.cu_stream())
                        .context("prewarm blob H2D")?;
                }
                drop(_g);
                cache.index.insert((l, e), i);
                cache.rev.insert(i, (l, e));
                cache.flat_set((l, e), i);
                cache.pinned[i] = true;
                cache.pinned_q.push_back(i);
                done[which] += 1;
            }
            cache.next_free = done[0].min(cache.n_slots);
            let alt_c = cache_alt.as_mut().unwrap().get_mut();
            alt_c.next_free = done[1].min(alt_c.n_slots);
            k.stream.synchronize()?;
            eprintln!(
                "[prewarm] 混档钉住 Q4 {} 个 + Q3 {} 个（{:.1} GiB + {:.1} GiB）用时 {:.1}s",
                done[0].min(pri_slots), done[1].min(alt_slots),
                (done[0].min(pri_slots) * blob_pri) as f64 / 1073741824.0,
                (done[1].min(alt_slots) * blob_alt) as f64 / 1073741824.0,
                t0.elapsed().as_secs_f64(),
            );
            }
            }
        }
        // ── ★ 2026-10-25`LINGXI_ARENA_PRELOAD_ALL`：把本卡服务层的
        // **全部专家**一次搬进显存（幂等：已驻留的跳过）。
        //
        // 动机（实测）：`[htime-spike]` 里那笔"prep 4.4 ms/轮"其实是**首触补槽** ——
        // `prep` 从 `t_pre_pairs` 起算，而 miss 时第 2 次 attempt 会把它连
        // `load_missing` 的整段 DMA 一起算进去（`[prep-prof]` 四个子步骤全是 0 µs 就是铁证）。
        // 全驻留后 decode 期 miss=0 ⇒ 这笔钱变成启动时一次性搬完。
        // 代价：启动多 5~7 s（本卡 ~32 GB over PCIe）；而 tok/s 口径**不含启动** ⇒ 净赚。
        if lingxi_model::flag("LINGXI_ARENA_PRELOAD_ALL") && !prewarm_off {
            let t0 = std::time::Instant::now();
            k.ctx.bind_to_thread()?;
            let (mut n_new, mut n_skip) = (0usize, 0usize);
            for l in lo..hi.min(arena.n_layers) {
                for e in 0..arena.n_experts {
                    let Some(b) = arena.routed(l, e)? else { continue };
                    let use_alt = mixed && Some(b.format) == fmt2;
                    let (cache, cblob): (&mut SlotCache, usize) = if use_alt {
                        (cache_alt.as_mut().unwrap().get_mut(), blob_alt)
                    } else {
                        (&mut cache, blob_pri)
                    };
                    if cache.flat_get((l, e)).is_some() {
                        n_skip += 1;
                        continue;
                    }
                    // 池满了就**停**，不要用 `victim` 去淘汰 —— 那会把 prewarm 钉住的
                    // 热专家挤出去（`victim` 的兜底会解钉最旧的那个）。剩下的懒加载即可。
                    if cache.next_free >= cache.n_slots {
                        break;
                    }
                    let Some(sl) = cache.victim(&[]) else { break };
                    let src = match alias.as_ref() {
                        Some(al) => &al[b.gu.start..b.down.end],
                        None => arena.bytes(&(b.gu.start..b.down.end)),
                    };
                    anyhow::ensure!(
                        src.len() == cblob,
                        "preload 字节数不符：src.len()={} cblob={cblob}（layer {l} expert {e}）",
                        src.len()
                    );
                    {
                        let sv = cache.slots.slice(sl * cblob..(sl + 1) * cblob);
                        let (dptr, _g) = sv.device_ptr(&k.stream);
                        unsafe {
                            cudarc::driver::result::memcpy_htod_async(
                                dptr, src, k.stream.cu_stream(),
                            )
                            .context("preload blob H2D")?;
                        }
                    }
                    cache.insert((l, e), sl);
                    n_new += 1;
                }
            }
            k.stream.synchronize()?;
            eprintln!(
                "[preload] 全驻留：新搬 {n_new}／已驻留 {n_skip}（Q4 槽 {} / Q3 槽 {}）用时 {:.1}s",
                cache.n_slots,
                // SAFETY：循环已结束、没有任何存活的 &mut/& 借用（与 loader 其它处同一约定）。
                cache_alt.as_ref().map(|c| unsafe { (*c.get()).n_slots }).unwrap_or(0),
                t0.elapsed().as_secs_f64()
            );
        }
        eprintln!(
            "arena: {} | {} 层 x {} 专家 | blob {} B{} | 设备槽 {} ({:.1} GiB) | fstype={:?} | arena_pinned={} ({:.1} GiB) | 服务层 {:?}",
            arena_path,
            arena.n_layers,
            arena.n_experts,
            blob_pri,
            if mixed { format!(" + {} B(Q3)", blob_alt) } else { String::new() },
            if mixed { format!("{}/{}", pri_slots, alt_slots) } else { format!("{}", slots) },
            if mixed {
                (pri_slots * blob_pri + alt_slots * blob_alt) as f64 / 1073741824.0
            } else {
                (blob_pri * slots) as f64 / 1073741824.0
            },
            fs_type,
            pinned as i32,
            if pinned { arena_bytes as f64 / 1073741824.0 } else { 0.0 },
            layers,
        );
        // ── ㊺ A2 设备路由：槽位表 + Q4/Q3 几何 ──
        let n_layers = arena.n_layers as usize;
        let n_experts = arena.n_experts as usize;
        let rows_gu = 2 * arena.n_ff as usize;
        let rows_dn = arena.n_in as usize;
        // 格式已在上方扫描（含混档双 cache 分流）。A2 设备路由在主格式是 Q4G64/Q3G128 时可用；
        // 混档走**双视图**（主/次各一张槽表 + 格式表），单格式时次视图全 0。
        let arena_fmt = pri_fmt;
        let q4_ok = matches!(
            arena_fmt,
            BlobFormat::Q4G64F16S | BlobFormat::Q3G128F16S
        );
        let gu_lay = matrix_layout(arena_fmt, rows_gu as u64, arena.n_in as u64)
            .context("gu 段几何非法")?;
        let dn_lay = matrix_layout(arena_fmt, rows_dn as u64, arena.n_ff as u64)
            .context("dn 段几何非法")?;
        let kpad = |x: usize| ((x as u64 + 127) / 128 * 128) as i32;
        let q4c = lingxi_model::moe::Q4Consts {
            blob: blob_pri as u64,
            gu_seg: 0,
            // R1 几何单源化：与本文件所有消费点同纪律——段几何用上面 gu_lay/dn_lay
            // （pri_fmt 的 matrix_layout），不用 arena.gu_bytes 缺省字段。
            dn_seg: gu_lay.payload_bytes,
            gu_scale: gu_lay.scale_offset,
            dn_scale: dn_lay.scale_offset,
            gu_k_pad: kpad(arena.n_in as usize),
            dn_k_pad: kpad(arena.n_ff as usize),
            bits: arena_fmt.bits(),
        };
        // 混档案：次格式（Q3）的段几何常量 + 每 (层,专家) 格式表（静态，启动刷一次）。
        let q4c_alt = match fmt2 {
            Some(f) => {
                let gu2 = matrix_layout(f, rows_gu as u64, arena.n_in as u64)
                    .context("alt gu 段几何非法")?;
                let dn2 = matrix_layout(f, rows_dn as u64, arena.n_ff as u64)
                    .context("alt dn 段几何非法")?;
                Some(lingxi_model::moe::Q4Consts {
                    blob: blob_alt as u64,
                    gu_seg: 0,
                    dn_seg: gu2.payload_bytes,
                    gu_scale: gu2.scale_offset,
                    dn_scale: dn2.scale_offset,
                    gu_k_pad: kpad(arena.n_in as usize),
                    dn_k_pad: kpad(arena.n_ff as usize),
                    bits: f.bits(),
                })
            }
            None => None,
        };
        let slot_map = k.stream.alloc_zeros::<i32>(n_layers * n_experts)?;
        let slot_map_alt = k.stream.alloc_zeros::<i32>(n_layers * n_experts)?;
        // 格式表：0 = 主（Q4），1 = 次（Q3）；服务区间外的层填 -1（永远不会被读）。
        let mut fmt_host = vec![-1i32; n_layers * n_experts];
        if mixed {
            for l in layers.start..layers.end.min(arena.n_layers) {
                for e in 0..arena.n_experts {
                    let is_alt = matches!(
                        arena.routed(l, e),
                        Ok(Some(b)) if b.format == fmt2.unwrap()
                    );
                    fmt_host[l as usize * n_experts + e as usize] = if is_alt { 1 } else { 0 };
                }
            }
        }
        let mut fmt_dev = k.stream.alloc_zeros::<i32>(n_layers * n_experts)?;
        k.stream.memcpy_htod(&fmt_host, &mut fmt_dev)?;
        let mut slot_map_pins = Vec::with_capacity(n_layers);
        let mut slot_map_alt_pins = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            slot_map_pins.push(unsafe {
                k.stream.context().alloc_pinned_with_flags::<i32>(n_experts.max(1), 0)?
            });
            slot_map_alt_pins.push(unsafe {
                k.stream.context().alloc_pinned_with_flags::<i32>(n_experts.max(1), 0)?
            });
        }
        let me = Self {
            k,
            arena,
            alias,
            cache: UnsafeCell::new(cache),
            cache_alt,
            fmt2,
            blob_pri,
            blob_alt,
            mixed,
            hist: UnsafeCell::new(HashMap::new()),
            ticks: std::cell::Cell::new(0),
            scratch: UnsafeCell::new(Scratch {
                gu: None,
                dn: None,
                slot_ids: [None, None],
                slot_ids_alt: [None, None],
                ids_pin: [None, None],
                ids_pin_alt: [None, None],
                ids_phase: 0,
                stub_gu: None,
                stub_dn: None,
            }),
            // `LINGXI_ARENA_REPIN=<窗口>`：在线自适应重钉（单位 = load_hits 次数；0 = 关）
            repin_window: std::env::var("LINGXI_ARENA_REPIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            repin_n,
            layers,
            slot_map,
            slot_map_alt,
            fmt_dev,
            slot_map_pins: UnsafeCell::new(slot_map_pins),
            slot_map_alt_pins: UnsafeCell::new(slot_map_alt_pins),
            q4c,
            q4c_alt,
            q4_ok,
            op_lock: std::sync::Mutex::new(()),
        };
        // 初始表：把预热后的全部层刷一次（之后只有 legacy load_hits / load_missing 会改它）。
        me.slot_map_sync_dirty()?;
        k.stream.synchronize()?;
        eprintln!(
            "[arena] A2 设备路由{}：槽位表 {} 层 x {} 专家（{:.0} KiB/表）",
            if q4_ok {
                if mixed { "可用（混档双视图）" } else { "可用（全 Q4G64 / 全 Q3G128）" }
            } else {
                "禁用（非量化格式）"
            },
            n_layers,
            n_experts,
            (n_layers * n_experts * 4) as f64 / 1024.0,
        );
        Ok(me)
    }

    /// ㊺ A2：把 `layer` 这一层的槽位表（E 项）同步上卡。
    /// 只在表**会被路由核读取**时需要：legacy `load_hits` / A2 `load_missing` 末尾，
    /// 以及构造期把预热结果刷一次。2 KB/层，锁页 H2D（每层独立暂存，异步安全）。
    fn slot_map_sync_layer(&self, layer: u32) -> Result<()> {
        let cache = unsafe { &*self.cache.get() };
        let e = cache.n_experts;
        if layer as usize >= self.arena.n_layers as usize {
            return Ok(());
        }
        let base = layer as usize * e;
        let pins = unsafe { &mut *self.slot_map_pins.get() };
        {
            let buf = pins[layer as usize].as_mut_slice()?;
            for j in 0..e {
                buf[j] = cache.flat[base + j] as i32;
            }
        }
        self.bind_ctx()?;
        let (dp, _g) = self.slot_map.device_ptr(&self.k.stream);
        let src = pins[layer as usize].as_slice()?;
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                dp + (base * 4) as u64,
                &src[..e],
                self.k.stream.cu_stream(),
            )
            .context("A2 slot_map H2D")?;
        }
        // 混档：次格式 cache 的同一层也要刷（两者是**独立**槽号空间）。
        if let Some(alt) = self.cache_alt.as_ref() {
            let cache_alt = unsafe { &*alt.get() };
            let pins_alt = unsafe { &mut *self.slot_map_alt_pins.get() };
            {
                let buf = pins_alt[layer as usize].as_mut_slice()?;
                for j in 0..e {
                    buf[j] = cache_alt.flat[base + j] as i32;
                }
            }
            let (dp_alt, _g2) = self.slot_map_alt.device_ptr(&self.k.stream);
            let src_alt = pins_alt[layer as usize].as_slice()?;
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    dp_alt + (base * 4) as u64,
                    &src_alt[..e],
                    self.k.stream.cu_stream(),
                )
                .context("A2 slot_map_alt H2D")?;
            }
        }
        Ok(())
    }

    /// ㊺ A2：把**所有变过的层**（insert/victim 都可能改到别的层）刷上卡，然后清脏位。
    fn slot_map_sync_dirty(&self) -> Result<()> {
        let cache = unsafe { &*self.cache.get() };
        for l in 0..cache.dirty.len() {
            if cache.dirty[l] {
                self.slot_map_sync_layer(l as u32)?;
            }
        }
        let cache = unsafe { &mut *self.cache.get() };
        cache.dirty.fill(false);
        if let Some(alt) = self.cache_alt.as_ref() {
            let cache_alt = unsafe { &mut *alt.get() };
            for l in 0..cache_alt.dirty.len() {
                if cache_alt.dirty[l] {
                    self.slot_map_sync_sublayer(l as u32)?;
                }
            }
            cache_alt.dirty.fill(false);
        }
        Ok(())
    }

    /// 只刷**次格式** cache 的某一层（上表已刷主格式；避免 `slot_map_sync_dirty` 里重复刷主表）。
    fn slot_map_sync_sublayer(&self, layer: u32) -> Result<()> {
        let Some(alt) = self.cache_alt.as_ref() else {
            return Ok(());
        };
        let cache_alt = unsafe { &*alt.get() };
        let e = cache_alt.n_experts;
        if layer as usize >= self.arena.n_layers as usize {
            return Ok(());
        }
        let base = layer as usize * e;
        let pins_alt = unsafe { &mut *self.slot_map_alt_pins.get() };
        {
            let buf = pins_alt[layer as usize].as_mut_slice()?;
            for j in 0..e {
                buf[j] = cache_alt.flat[base + j] as i32;
            }
        }
        self.bind_ctx()?;
        let (dp_alt, _g2) = self.slot_map_alt.device_ptr(&self.k.stream);
        let src_alt = pins_alt[layer as usize].as_slice()?;
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                dp_alt + (base * 4) as u64,
                &src_alt[..e],
                self.k.stream.cu_stream(),
            )
            .context("A2 slot_map_alt H2D")?;
        }
        Ok(())
    }

    /// **绑上下文**：cudarc 的**裸驱动调用**（`result::memcpy_*`）不会 `bind_to_thread`，
    /// 而所有安全封装（`alloc` / `memcpy_htod` / 内核 launch）都会 —— 单卡时"当前上下文"
    /// 恒等于唯一那张卡，所以一直没暴露；一旦开第二张卡（`LINGXI_MOE_SPLIT`），
    /// 线程的当前上下文可能仍是 **GPU0**，此时对 **GPU1** 的槽发拷贝 ⇒
    /// `CUDA_ERROR_INVALID_VALUE`（2026-10-16 实测：prefill 跑到 layer 24 就崩在
    /// `arena blob H2D`）。⇒ 每个裸拷贝**之前**先绑一次（`cuCtxSetCurrent`，开销可忽略）。
    fn bind_ctx(&self) -> Result<()> {
        self.k.ctx.bind_to_thread()?;
        Ok(())
    }

    /// 取一段 host 字节：**优先从"已注册的别名映射"取**。
    /// 别名与 `arena` 是同一份 tmpfs 页（内容逐字节相同，**不改任何数学**），
    /// 区别只在于**它的地址在 `cuMemHostRegister` 的注册区间内** ⇒ 拷贝才走 copy-engine DMA。
    /// （`LINGXI_ARENA_PIN_DMA=1` 时为 `Some`，否则退回 `arena.bytes`。）
    fn src_bytes(&self, r: &std::ops::Range<usize>) -> &[u8] {
        match self.alias.as_ref() {
            Some(al) => &al[r.clone()],
            None => self.arena.bytes(r),
        }
    }

    /// **在线自适应重钉**：每累计 `repin_window` 次请求，就按"最近热度"把每层的 top-N 重排一次
    /// —— 新进 top-N 的搬进来钉住，掉出 top-N 的解钉（还给 LRU），热度计数**减半**（滑动窗口）。
    ///
    /// 为什么需要（实测）：钉表是离线静态的，同一配置 n=80 有 46 tok/s、**n=300 掉到 29.6**
    /// —— 生成一旦超出采集话题的范围，覆盖率就掉。这对应"应该跟着话题自动调整"。
    fn maybe_repin(&self, inter: usize, hidden: usize) -> Result<()> {
        if self.mixed {
            return Ok(()); // 混档 v1：重钉禁用（静态钉表同样禁用，见 new）
        }
        if self.repin_window == 0 {
            return Ok(());
        }
        let blob = (self.arena.gu_bytes + self.arena.down_bytes) as usize;
        let _ = (inter, hidden);
        let cache = unsafe { &mut *self.cache.get() };
        let total = cache.hits + cache.misses;
        if total < cache.recent_at + self.repin_window {
            return Ok(());
        }
        let n_pin = self.repin_n.max(1);
        // 1) 每层按最近热度取 top-N
        let mut per: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for (&(l, e), &c) in cache.recent.iter() {
            if self.layers.contains(&l) {
                per.entry(l).or_default().push((c, e));
            }
        }
        let mut want: HashSet<(u32, u32)> = HashSet::new();
        for (l, mut v) in per {
            v.sort_unstable_by(|a, b| b.cmp(a));
            for &(_, e) in v.iter().take(n_pin) {
                want.insert((l, e));
            }
        }
        // 2) 现在钉住的是哪些 key
        let have: HashSet<(u32, u32)> = cache
            .index
            .iter()
            .filter(|(_, &sl)| cache.pinned[sl])
            .map(|(k, _)| *k)
            .collect();
        let to_unpin: Vec<(u32, u32)> = have.difference(&want).copied().collect();
        let n_unpin = to_unpin.len();
        // 每次重钉最多改这么多（防止话题切换时一次搬几 GB —— 实测 `take(600)` + 短窗口
        // 会把 17 GB 额外流量灌进去，比它省下的 miss 还贵）
        let churn_cap: usize = std::env::var("LINGXI_ARENA_REPIN_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(96);
        let to_pin: Vec<(u32, u32)> = want.difference(&have).copied().take(churn_cap).collect();
        let mut done = 0usize;
        if !to_pin.is_empty() {
            self.bind_ctx()?;
            // 3) 解钉：槽留着（还给 LRU），只是不再受保护
            for k in to_unpin {
                if let Some(sl) = cache.index.get(&k).copied() {
                    cache.pinned[sl] = false;
                    // ⚠️ **必须放回 LRU 队**：`victim` 只从 `order` 取，钉住的槽从不进队，
                    // 所以解钉后不入队 ⇒ `order` 会逐渐空 ⇒ `victim` 走到 `return 0` 的兜底
                    // ⇒ 多个 key 撞在 0 号槽（实测 `驻留 9166/9000` 就是这么来的）。
                    cache.touch(k, sl);
                }
            }
            // 4) 新钉：**已经在缓存里的直接钉住（零搬运）**，不在的才取槽 + H2D
            //
            // ⚠️ 第一版没有这一步，而且踩了个真坑：`(l,e)` 可能已在 `index` 里（挂在未钉的槽上），
            // 这时再 `victim` 拿新槽会**覆盖 `index[key]` 而旧槽永远回收不了** ⇒ 槽泄漏
            // （实测 `驻留 9000 → 6299` 一路掉、命中率 84.8% → 73.9%、tok/s 32.5 → 19.7）。
            // 本次重钉已发出去的槽同样要保护（理由见 `victim` 的 `prot` 说明）。
            let mut prot: Vec<usize> = Vec::new();
            for (l, e) in to_pin {
                if let Some(&sl) = cache.index.get(&(l, e)) {
                    if !cache.pinned[sl] {
                        cache.pinned[sl] = true;
                        cache.pinned_q.push_back(sl);
                    }
                    prot.push(sl);
                    done += 1;
                    continue;
                }
                // `routed` 返回 `Option`（arena 里没有这个专家时是 None）⇒ 跳过，不钉
                let Ok(Some(b)) = self.arena.routed(l, e) else { continue };
                let Some(sl) = cache.victim(&prot) else { break };
                prot.push(sl);
                let src = self.src_bytes(&(b.gu.start..b.down.end));
                anyhow::ensure!(src.len() == blob, "重钉字节数不符");
                {
                    let sv = cache.slots.slice_mut(sl * blob..(sl + 1) * blob);
                    use cudarc::driver::DevicePtr;
                    let (dp, _g) = sv.device_ptr(&self.k.stream);
                    unsafe {
                        cudarc::driver::result::memcpy_htod_async(
                            dp,
                            src,
                            self.k.stream.cu_stream(),
                        )?;
                    }
                }
                cache.insert((l, e), sl);
                cache.pinned[sl] = true;
                cache.pinned_q.push_back(sl);
                done += 1;
            }
            self.k.stream.synchronize()?;
        }
        // 5) 滑动窗口：热度减半（让"最近"的权重高于历史累计）
        for v in cache.recent.values_mut() {
            *v >>= 1;
        }
        cache.recent.retain(|_, v| *v > 0);
        cache.recent_at = total;
        eprintln!(
            "[arena-repin] 第 {} 次重钉：钉住 +{done}（目标 {} 个/层 × 48），解钉 {}，窗口 {} 次请求",
            total / self.repin_window.max(1),
            n_pin,
            n_unpin,
            self.repin_window
        );
        Ok(())
    }

    /// 命中率统计（`LINGXI_ARENA_STATS=1`）。**必须在快路径 return 之前调用**：
    /// Q4 直读分支是提前返回的，统计代码放在函数尾部会永远不执行。
    fn tick_stats(&self) {
        let stats_on = lingxi_model::flag("LINGXI_ARENA_STATS");
        let hist_on = lingxi_model::flag("LINGXI_ARENA_HIST");
        // ⚠️ 两个开关**各管各的**：以前这里是"STATS 没开就直接 return"，
        // 那样单开 HIST 会静默什么都不落（本轮写完才发现，已修）。
        if !stats_on && !hist_on {
            return;
        }
        // 按**实例**计数（见字段 `ticks` 的注释）：双卡下两个实例各自 480 次落一次盘。
        let n = self.ticks.get() + 1;
        self.ticks.set(n);
        if n % 480 == 0 {
            let c = unsafe { &*self.cache.get() };
            let tot = c.hits + c.misses;
            if stats_on {
                eprintln!(
                    "[arena] 命中率 {:.1}%  ({}/{}) 驻留 {}/{} 槽",
                    if tot > 0 { 100.0 * c.hits as f64 / tot as f64 } else { 0.0 },
                    c.hits,
                    tot,
                    c.index.len(),
                    c.n_slots,
                );
            }
            let _ = tot;
            // `LINGXI_ARENA_HIST=<path>`：把 (层,专家,请求数) 落盘。
            // ⚠️ **必须合并而不是整文件覆写**（2026-10-18 实测踩到）：双卡 layer-split 下
            // 有**两个** ArenaExpertSource（GPU0 管 0..24、GPU1 管 24..48），各自只统计自己
            // 那段层；整文件覆写会让后写的那份把另一半层抹掉 ⇒ 用这种残缺表做 prewarm
            // 实测 43.9 tok/s（完整表 48.5）。这里改成"读回旧表 → 只替换本实例负责的层 →
            // 写回"，并用全局锁串行化两个实例的写入。
            if let Ok(path) = std::env::var("LINGXI_ARENA_HIST") {
                use std::sync::{Mutex, OnceLock};
                static W: OnceLock<Mutex<()>> = OnceLock::new();
                let _g = W.get_or_init(|| Mutex::new(())).lock().unwrap();
                let mut merged: std::collections::BTreeMap<(u32, u32), u64> =
                    std::collections::BTreeMap::new();
                if let Ok(old) = std::fs::read_to_string(&path) {
                    for line in old.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }
                        let mut it = line.split(',');
                        if let (Some(l), Some(e), Some(c)) = (it.next(), it.next(), it.next()) {
                            if let (Ok(l), Ok(e), Ok(c)) =
                                (l.parse::<u32>(), e.parse::<u32>(), c.parse::<u64>())
                            {
                                // 旧表里**不属于本实例**的层原样保留
                                if !self.layers.contains(&l) {
                                    merged.insert((l, e), c);
                                }
                            }
                        }
                    }
                }
                let h = unsafe { &*self.hist.get() };
                for (&(l, e), &c) in h.iter() {
                    if self.layers.contains(&l) {
                        merged.insert((l, e), c);
                    }
                }
                let mut s = String::with_capacity(merged.len() * 16);
                s.push_str("# layer,expert,count\n");
                for ((l, e), c) in merged {
                    s.push_str(&format!("{l},{e},{c}\n"));
                }
                if let Err(e) = std::fs::write(&path, s) {
                    eprintln!("[arena] HIST 落盘失败 {path}: {e}");
                }
            }
        }
    }
}

impl<'a> lingxi_model::moe::ExpertSource for ArenaExpertSource<'a> {
    fn q4_dev(&self) -> Option<lingxi_model::moe::Q4DevView<'_>> {
        if !self.q4_ok {
            return None;
        }
        let cache = unsafe { &*self.cache.get() };
        let alt = match (self.cache_alt.as_ref(), self.q4c_alt) {
            (Some(c), Some(cs)) => {
                let ca = unsafe { &*c.get() };
                Some(lingxi_model::moe::Q4DevAlt {
                    slots: &ca.slots,
                    slot_map: &self.slot_map_alt,
                    consts: cs,
                })
            }
            _ => None,
        };
        Some(lingxi_model::moe::Q4DevView {
            slots: &cache.slots,
            slot_map: &self.slot_map,
            consts: self.q4c,
            alt,
            fmt: if self.mixed { Some(&self.fmt_dev) } else { None },
        })
    }

    fn slot_map_sync_layer(&self, layer: u32) -> Result<()> {
        ArenaExpertSource::slot_map_sync_layer(self, layer)
    }

    /// EP 均衡分配：直接给该 (层, 专家) 的槽步长（Q4 热 2.6 MB / Q3 冷 1.92 MB）。
    /// 不碰设备、不做任何 IO（arena 索引常驻）⇒ 构建一次分配表几乎零成本。
    fn slot_bytes(&self, layer: u32, expert: u32) -> Option<u64> {
        if layer >= self.arena.n_layers || expert >= self.arena.n_experts {
            return None;
        }
        let b = match self.arena.routed(layer, expert) {
            Ok(Some(b)) => b,
            _ => return None,
        };
        Some(if Some(b.format) == self.fmt2 {
            self.blob_alt as u64
        } else {
            self.blob_pri as u64
        })
    }

    /// ㊺ A2：补槽。与 `load_hits` 的 miss 分支同一条纪律（victim 带保护集、blob 整段 H2D），
    /// 但**只搬缺的**、不做解量化、不填 scratch。返回时槽位表已对设备可见。
    /// 混档按各专家自己的格式落到主/次 cache（两者的槽号空间独立、保护集也分开）。
    fn load_missing(
        &self,
        layer: u32,
        experts: &[u32],
        prot_pri: &[usize],
        prot_alt: &[usize],
    ) -> Result<()> {
        if experts.is_empty() {
            return Ok(());
        }
        // 双会话并发下的簿记串行化（见字段 `op_lock` 的注释）。
        let _op = self.op_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.bind_ctx()?;
        let mut prot: Vec<usize> = prot_pri.to_vec();
        let mut prot2: Vec<usize> = prot_alt.to_vec();
        let mut n_new = 0usize;
        for &e in experts {
            mark_expert(layer, e);
            let key = (layer, e);
            let b = self
                .arena
                .routed(layer, e)?
                .with_context(|| format!("arena 缺专家 layer {layer} expert {e}"))?;
            let is_alt = self.fmt2 == Some(b.format);
            // 两 cache 是**不同**的 UnsafeCell：分别取可变引用（同一时刻只动一个）。
            let (cache, blob, prot_ref): (&mut SlotCache, usize, &mut Vec<usize>) = if is_alt {
                (
                    unsafe { &mut *self.cache_alt.as_ref().unwrap().get() },
                    self.blob_alt,
                    &mut prot2,
                )
            } else {
                (unsafe { &mut *self.cache.get() }, self.blob_pri, &mut prot)
            };
            if let Some(sl) = cache.flat_get(key) {
                EXPERT_HITS.fetch_add(1, Ordering::Relaxed);
                cache.touch(key, sl);
                continue;
            }
            EXPERT_MISSES.fetch_add(1, Ordering::Relaxed);
            let sl = cache.victim(prot_ref).ok_or_else(|| {
                anyhow::anyhow!(
                    "arena 槽位耗尽（A2 补槽）：layer {layer} 需要新槽，但 {} 个槽全被本批/钉表占用",
                    cache.n_slots
                )
            })?;
            prot_ref.push(sl);
            let src = self.src_bytes(&(b.gu.start..b.down.end));
            anyhow::ensure!(src.len() == blob, "A2 补槽字节数不符");
            {
                let sv = cache.slots.slice(sl * blob..(sl + 1) * blob);
                let (dp, _g) = sv.device_ptr(&self.k.stream);
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(dp, src, self.k.stream.cu_stream())
                        .context("A2 补槽 blob H2D")?;
                }
            }
            cache.insert(key, sl);
            cache.misses += 1;
            n_new += 1;
        }
        if n_new > 0 {
            self.slot_map_sync_dirty()?;
        }
        Ok(())
    }

    fn load_hits<'b>(
        &'b self,
        layer: u32,
        experts: &[u32],
        inter: usize,
        hidden: usize,
        want_q4: bool,
        overlap: bool,
    ) -> Result<lingxi_model::moe::ExpertHits<'b>> {
        let s = experts.len();
        if s == 0 {
            bail!("ArenaExpertSource::load_hits 收到空专家列表");
        }
        // 双会话并发下的簿记串行化（见字段 `op_lock` 的注释）。
        let _op = self.op_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.tick_stats();
        // `LINGXI_ARENA_HIST`：记每 `(层, 专家)` 的**请求**次数（含命中与未命中）。
        // 这是"每层该钉多少个热专家"的唯一依据（P1）。
        if lingxi_model::flag("LINGXI_ARENA_HIST") {
            let h = unsafe { &mut *self.hist.get() };
            for &e in experts.iter() {
                *h.entry((layer, e)).or_insert(0) += 1;
            }
        }
        let repin_on = self.repin_window > 0;
        // 在线自适应重钉：窗口到了就按"最近热度"重排每层 top-N（详见 `maybe_repin`）
        self.maybe_repin(inter, hidden)?;
        if hidden != self.arena.n_in as usize || inter != self.arena.n_ff as usize {
            bail!(
                "维度不符：模型 hidden={hidden} inter={inter}，arena {}/{}",
                self.arena.n_in,
                self.arena.n_ff
            );
        }
        let rows_gu = 2 * inter;
        let rows_dn = hidden;
        let per_gu = rows_gu * hidden;
        let per_dn = rows_dn * inter;
        // 直读快路径（仅单格式可达）的槽步长；混合态走逐专家解量化，按各 cache 自己的步长。
        let blob = self.blob_pri;

        // scratch 按需扩容（prefill 的 rows*top_k）。
        // ⚠️ **只在非 Q4 路径才分配 bf16 解量化 scratch**：批量路径下 `s` 可到 `rows*top_k`
        // （长 prompt 上万），按 `s * 2*inter*hidden` 预分配会是几十 GB，直接 OOM。
        // 需要时在下面的 else 分支里再分配。

        // 1) 槽位定位：命中直接用；未命中 → H2D 填进（未用/被淘汰的）槽
    // ★ P15-⑱：本批簿记走 thread_local 复用（原来每次 4 个容器分配）；`prot` 改 `Vec`
    // 线性扫（本批对数 ≤ 64，`victim` 里 HashSet 哈希反而慢）。语义全同。
    thread_local! {
        static LH_WS: std::cell::RefCell<(
            Vec<BlobFormat>, Vec<usize>, Vec<usize>, Vec<bool>, Vec<u8>, Vec<usize>,
        )> = const { std::cell::RefCell::new((
            Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        )) };
    }
    let mut lh = LH_WS.with(|c| c.replace((
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
    )));
    let (fmts, slot_ids, prot, hit_flag, which, prot_alt) = &mut lh;
    fmts.clear();
    slot_ids.clear();
    prot.clear();
    hit_flag.clear();
    which.clear();
    prot_alt.clear();
    // **本批已分配的槽**（传给它 `victim`，见那里的说明）：同一次调用里已经发出去的槽
    // 绝不能又被"淘汰"给另一个专家 —— 否则两个专家共用一块权重、数学静默出错。
    // 每个条目是命中还是未命中 —— `LINGXI_MOE_OVERLAP` 按它切"命中组 / 未命中组"。
    let mut n_miss = 0usize;
        // 裸 `memcpy_htod_async` 前必须绑本源的上下文（见 `bind_ctx`）。
        self.bind_ctx()?;
        // ★ **专家段重叠**（`LINGXI_MOE_OVERLAP`）：把 miss 的 blob H2D 挪到**第二流**上飞，
        // compute 流先算命中组 ⇒ 两边并行（；`NOH2D` 实测这一趟值 28 ms/轮）。
        let ov_stream = if overlap { Some(self.k.stream2.clone()) } else { None };
        if let Some(s2) = ov_stream.as_ref() {
            // 跨流门：`stream2` 上的覆写必须等 compute 流上**更早那些读同一个槽的 GEMV** 跑完。
            // （现在它是空转：MoE 的路由读取每层都会 `stream.synchronize`，队列本来就是空的；
            // 留着是为了不依赖那个巧合。）
            self.k.ev_mo_pre.record(&self.k.stream)?;
            s2.wait(&self.k.ev_mo_pre)?;
        }
        {
            for &e in experts.iter() {
                mark_expert(layer, e);
                let key = (layer, e);
                let b = self
                    .arena
                    .routed(layer, e)?
                    .with_context(|| format!("arena 缺专家 layer {layer} expert {e}"))?;
                fmts.push(b.format);
                // 混档：按 blob 自己的格式选 cache（主=Q4 / 次=Q3），槽步长随之。
                // 单格式 arena：`mixed=false`，全部走主 cache（行为与以前一致）。
                let use_alt = self.mixed && Some(b.format) == self.fmt2;
                which.push(use_alt as u8);
                let (cache, cblob): (&mut SlotCache, usize) = if use_alt {
                    (
                        unsafe { &mut *self.cache_alt.as_ref().unwrap().get() },
                        self.blob_alt,
                    )
                } else {
                    (unsafe { &mut *self.cache.get() }, self.blob_pri)
                };
                if repin_on {
                    *cache.recent.entry(key).or_insert(0) += 1;
                }
                if let Some(sl) = cache.flat_get(key) {
                    cache.hits += 1;
                    EXPERT_HITS.fetch_add(1, Ordering::Relaxed);
                    cache.touch(key, sl);
                    if use_alt {
                        prot_alt.push(sl);
                    } else {
                        prot.push(sl);
                    }
                    slot_ids.push(sl);
                    hit_flag.push(true);
                    continue;
                }
                cache.misses += 1;
                EXPERT_MISSES.fetch_add(1, Ordering::Relaxed);
                n_miss += 1;
                // victim 的保护集是**本 cache 内**的槽号空间 ⇒ 主/次 cache 各用各的 prot。
                let sl = {
                    let prot_ref: &[usize] = if use_alt { prot_alt } else { prot };
                    cache.victim(prot_ref).ok_or_else(|| {
                        anyhow::anyhow!(
                            "arena 槽位耗尽：layer {layer} expert {e} 需要新槽，但{} cache 的 {} 个槽\
                             全被本批/钉表占用{}",
                            if use_alt { " Q3" } else { "" },
                            cache.n_slots,
                            if self.mixed {
                                "（混档：可用 LINGXI_ARENA_MIX_Q4_SLOTS 调整 Q4/Q3 槽配比）"
                            } else {
                                "（把 LINGXI_ARENA_PREWARM_N 调小或加 LINGXI_ARENA_SLOTS）"
                            },
                        )
                    })?
                };
                if use_alt {
                    prot_alt.push(sl);
                } else {
                    prot.push(sl);
                }
                // 诊断：`LINGXI_ARENA_NOH2D=1` 只记槽不真拷 DMA。
                // 槽内容因此是旧的（数学结果作废），但**LRU/命中统计/槽位编号完全不变**，
                // 于是能干净地回答"每层这 1ms 到底是不是被专家 DMA 占掉的"。
                if !lingxi_model::flag("LINGXI_ARENA_NOH2D") {
                    let src = self.src_bytes(&(b.gu.start..b.down.end));
                    // 混档纪律：净字节数必须正好等于**该格式**的槽步长。
                    anyhow::ensure!(
                        src.len() == cblob,
                        "blob 字节数与该格式槽步长不符：src.len()={} cblob={}（layer {} expert {} fmt={:?}）",
                        src.len(), cblob, layer, e, b.format
                    );
                    let sv = cache.slots.slice(sl * cblob..(sl + 1) * cblob);
                    let (dptr, _g) = sv.device_ptr(&self.k.stream);
                    let cs = match ov_stream.as_ref() {
                        Some(x) => x.cu_stream(),
                        None => self.k.stream.cu_stream(),
                    };
                    unsafe {
                        cudarc::driver::result::memcpy_htod_async(dptr, src, cs)
                            .context("arena blob H2D")?;
                    }
                    // [探针] `LINGXI_ARENA_SLOT_CHECK=<layer>`：把刚灌进去的槽回读 64 B 与 arena 源字节比对，
                    // 并在 slot_ids 上传后核对其回读 —— 一次判掉"槽内容/槽号到底对不对"。
                    if lingxi_model::flag("LINGXI_ARENA_SLOT_CHECK")
                        && std::env::var("LINGXI_ARENA_SLOT_CHECK")
                            .ok()
                            .and_then(|v| v.parse::<u32>().ok())
                            == Some(layer)
                    {
                        self.k.stream.synchronize()?;
                        let mut back = vec![0u8; cblob];
                        unsafe {
                            cudarc::driver::result::memcpy_dtoh_sync::<u8>(&mut back, dptr)
                        }?;
                        let bad = back.iter().zip(src.iter()).filter(|(a, b)| a != b).count();
                        if bad > 0 {
                            use std::sync::atomic::{AtomicU32, Ordering};
                            static NBAD: AtomicU32 = AtomicU32::new(0);
                            let k = NBAD.fetch_add(1, Ordering::Relaxed);
                            if k < 5 {
                                let first = back.iter().zip(src.iter()).position(|(a, b)| a != b);
                                eprintln!(
                                    "[slot-check] layer={layer} e={e} slot={sl} **不一致 {bad}/{cblob}** 首个差异={first:?}"
                                );
                            }
                        } else {
                            use std::sync::atomic::{AtomicU32, Ordering};
                            static NOK: AtomicU32 = AtomicU32::new(0);
                            NOK.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    // ★ 判决性诊断（`LINGXI_ARENA_SYNC_H2D=1`）：每次 miss 的 H2D 立刻同步。
                    // 用途：如果"同一份权重、只差命中/未命中"就能让 token 流分叉，那要么是
                    // **字节不同**、要么是 **H2D 与后续 GEMV 之间没有真正可见**。这条开关把
                    // 后一种可能一刀排除（同步之后就不可能有"还没到位"这回事）。
                    if lingxi_model::flag("LINGXI_ARENA_SYNC_H2D") {
                        self.k.stream.synchronize()?;
                    }
                }
                cache.insert(key, sl);
                slot_ids.push(sl);
                hit_flag.push(false);
            }
        }
        // ㊺ A2：槽位表可能被本批的 miss 改过（也可能挤掉了**别的层**的 key）⇒ 刷脏行。
        // 无脏行时是 48 次 bool 检查（零 H2D），所以无条件调用。
        self.slot_map_sync_dirty()?;
        // 本层所有 miss 的 blob 都发完了 ⇒ 在第二流上打个点（调用方跑 miss 组之前等它）
        if n_miss > 0 {
            if let Some(s2) = ov_stream.as_ref() {
                self.k.ev_mo_miss.record(s2)?;
            }
        }
        // `LINGXI_EVPROF=1`：把本层 miss 数记进计时器（用来和"专家段耗时"算相关性）
        if let Some(p) = self.k.evprof.as_ref() {
            p.set_info(layer, 3, n_miss as u32);
        }
        // [诊断] `LINGXI_MOE_OV_TRACE=1`：打印前 200 层的"命中/未命中 + miss 字节"。
        // 为什么要它：重叠能不能帮上忙**完全取决于 miss 的分布** ——
        // 若 miss 是"每层 2~3 个"（散布），命中组的 GEMV（~0.5 ms）足够把它盖住；
        // 若是"少数层一次 30 个"（突发），GEMV 根本盖不住那 7 ms 的 H2D。
        if std::env::var("LINGXI_MOE_OV_TRACE").is_ok() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 200 {
                eprintln!(
                    "[moe-ov] 层 {layer:2}: 条目 {s}（命中 {} / 未命中 {n_miss}）miss {:.2} MB",
                    s - n_miss,
                    (n_miss as f64) * (blob as f64) / 1e6
                );
            }
        }

        // 2) 从**设备槽**解量化：全 Q4G64 / 全 Q3G128（统一格式）走**批量版**（2 次 launch/层，
        // 而非 2S 次）；否则逐专家（含 BF16 原样直落）。
        let all_q4 = !fmts.is_empty()
            && fmts.iter().all(|f| *f == fmts[0])
            && matches!(fmts[0], BlobFormat::Q4G64F16S | BlobFormat::Q3G128F16S);
        // 提前拷出本批统一格式（BlobFormat: Copy）：后面要把 `lh` 簿记还回 thread_local，
        // 此后不能再借 `fmts`。
        let q4_fmt: Option<BlobFormat> = if all_q4 { Some(fmts[0]) } else { None };
        // 把"命中/未命中"切出来给调用方（只在 Q4 直读路径上才有意义）。
        let mut overlap_info = if overlap && n_miss > 0 {
            let mut hit_runs: Vec<(u32, u32)> = Vec::new();
            let mut miss_groups: Vec<u32> = Vec::new();
            let mut i = 0usize;
            while i < s {
                if hit_flag[i] {
                    let a = i;
                    while i < s && hit_flag[i] {
                        i += 1;
                    }
                    hit_runs.push((a as u32, i as u32));
                } else {
                    miss_groups.push(i as u32);
                    i += 1;
                }
            }
            Some(lingxi_model::moe::MoOverlap { hit_runs, miss_groups })
        } else {
            None
        };
        if overlap_info.is_some() && !(all_q4 && want_q4) {
            // ⚠️ **安全网**：这一步的 H2D 已经发在第二流上了，但调用方要走的是**非 Q4 直读**
            // 路径（它不会等事件）⇒ 这里在 compute 流上补一次等待，然后退回老行为。
            // 这条支路目前只在"arena 里混了非 Q4 blob"时才会走到（默认全 Q4G64）。
            self.k.stream.wait(&self.k.ev_mo_miss)?;
            overlap_info = None;
        }
        // [探针] 一次性打印走了哪条路（判断 Q4 直读是否真的生效）。
        if lingxi_model::flag("LINGXI_ARENA_ROUTE") {
            use std::sync::atomic::{AtomicBool, Ordering};
            static ONCE: AtomicBool = AtomicBool::new(false);
            if !ONCE.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[arena-route] s={s} all_q4={all_q4} want_q4={want_q4} fmts0={:?}",
                    fmts.first()
                );
            }
        }
        // 双缓冲：本层用 `ph` 号那份，下一层用另一份 ⇒ 本层的内核读完之前，
        // 本层的 `slot_ids` 不会被覆写（见 `Scratch::ids_phase` 的说明）。
        let ph;
        // 混档 arena 即使是单格式批次也**不走本臂**：本臂的 dn_seg 用头字段 gu_bytes
        // （按首个 ref 的格式推导，混档下是错的）、blob/cache 固定主格式 —— 单格式批次
        // 交给下面的混档双视图臂（runs 单段，天然兼容）。
        if all_q4 && want_q4 && !self.mixed {
            {
                let sc = unsafe { &mut *self.scratch.get() };
                ph = (sc.ids_phase & 1) as usize;
                sc.ids_phase ^= 1;
                if sc.slot_ids[ph].as_ref().map_or(0, |v| v.len()) < s {
                    sc.slot_ids[ph] = Some(unsafe { self.k.stream.alloc::<i32>(s) }?);
                }
                if sc.ids_pin[ph].as_ref().map_or(0, |p| p.len()) < s {
                    sc.ids_pin[ph] = Some(unsafe {
                        self.k.stream.context().alloc_pinned_with_flags::<i32>(s, 0)
                    }?);
                }
                // Q4 直读路径下 `ExpertHits::gate_up/down` 只是**占位**（调用方不会读它们），
                // 给一对 1 元素 stub 即可 —— 绝不能按 `s * 2*inter*hidden` 去要 bf16 scratch
                // （批量路径 s 可达上万，那是几十 GB）。
                if sc.stub_gu.is_none() {
                    sc.stub_gu = Some(unsafe { self.k.stream.alloc::<bf16>(1) }?);
                }
                if sc.stub_dn.is_none() {
                    sc.stub_dn = Some(unsafe { self.k.stream.alloc::<bf16>(1) }?);
                }
                // ★ `LINGXI_ARENA_SYNC_IDS=1`（判决性诊断，2026-10-16）：**在覆写复用缓冲之前先同步**。
                //
                // 背景（已实测）：投机路径在「同权重、只差命中/未命中」时会分叉出不同的 token 流，
                // 而**给每个 miss 的 blob H2D 加同步就治好了**；**换成设备侧 event 等待（MOE_OVERLAP）
                // 治不好** ⇒ 说明这是**主机侧**的时序问题，不是设备侧的可见性问题。
                // 头号嫌疑就在下面这段：`sc.ids_pin`（锁页 host）与 `sc.slot_ids`（device）是
                // **跨层复用**的，`slot_ids` 的内容被下一个 `load_hits` 覆写，而上一层的内核
                // 还在异步读它 —— 只在"设备侧等一个 event"时盖不住。
                // 这条开关把嫌疑一刀切开：同步之后再覆写。
                if lingxi_model::flag("LINGXI_ARENA_SYNC_IDS") {
                    self.k.stream.synchronize()?;
                }
                {
                    let pin = sc.ids_pin[ph].as_mut().unwrap();
                    let buf = pin.as_mut_slice()?;
                    for (j, &x) in slot_ids.iter().enumerate() {
                        buf[j] = x as i32;
                    }
                }
                let (dp, _g) = sc.slot_ids[ph].as_mut().unwrap().device_ptr_mut(&self.k.stream);
                let src = sc.ids_pin[ph].as_ref().unwrap().as_slice()?;
                self.bind_ctx()?;
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        dp,
                        &src[..s],
                        self.k.stream.cu_stream(),
                    )?;
                }
                // [探针] 同上：核对**设备侧 slot_ids** 与 host 意图是否一致。
                if lingxi_model::flag("LINGXI_ARENA_SLOT_CHECK")
                    && std::env::var("LINGXI_ARENA_SLOT_CHECK")
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok())
                        == Some(layer)
                {
                    self.k.stream.synchronize()?;
                    let mut back = vec![0i32; s];
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_sync::<i32>(&mut back, dp)
                    }?;
                    let bad = back
                        .iter()
                        .zip(slot_ids.iter())
                        .filter(|(a, b)| **a as usize != **b)
                        .count();
                    eprintln!(
                        "[slot-check] layer={layer} s={s} slot_ids 回读不一致={bad} dev={:?} host={:?}",
                        &back[..4.min(s)],
                        &slot_ids[..4.min(s)]
                    );
                }
            }
            // P15-⑱：归还本批簿记工作区（取走容量，下一 call 复用）。
            // 先取走 bits_sel（消费侧切格式段用），再归还。
            let bits_sel: Vec<u8> = fmts.iter().map(|f| f.bits() as u8).collect();
            LH_WS.with(|c| {
                c.replace(lh);
            });
            let cache = unsafe { &*self.cache.get() };
            let sc = unsafe { &*self.scratch.get() };
            let hits_fmt = q4_fmt.expect("all_q4 已保证本批统一为 Q4G64/Q3G128");
            let gu_lay = matrix_layout(hits_fmt, rows_gu as u64, hidden as u64)
                .context("gu 段几何非法")?;
            let dn_lay = matrix_layout(hits_fmt, rows_dn as u64, inter as u64)
                .context("dn 段几何非法")?;
            // **直读 Q4**：不再解量化到 bf16 scratch（原路径每层写 98MB 且解量化 kernel 只有
            // 1 元素/线程、访存不合并，占 MoE 专家段的大头）。把槽句柄交给 MoE 用融合 GEMV。
            // `gate_up`/`down` 仅作占位，此路径下调用方不会读它们。
            let kpad = |k: usize| ((k as u64 + 127) / 128 * 128) as i32;
            return Ok(lingxi_model::moe::ExpertHits {
                gate_up: sc.stub_gu.as_ref().unwrap(),
                down: sc.stub_dn.as_ref().unwrap(),
                q4: Some(lingxi_model::moe::Q4Slots {
                    slots: &cache.slots,
                    slot_ids: sc.slot_ids[ph].as_ref().unwrap(),
                    blob: blob as u64,
                    gu_seg: 0,
                    // R1 几何单源化（2026-10-21）：段几何一律从**本批格式**的 matrix_layout 取，
                    // 不再用 arena.gu_bytes（那是 open 时按首个 ref 格式推导的缺省值——混档下
                    // 是错的，10-19/10-20 两次 dn 段偏移雷的根源；本臂虽有 !self.mixed 闸门，
                    // 但把正确性寄托在闸门上就是债务）。
                    dn_seg: gu_lay.payload_bytes,
                    gu_scale: gu_lay.scale_offset,
                    dn_scale: dn_lay.scale_offset,
                    gu_k_pad: kpad(hidden),
                    dn_k_pad: kpad(inter),
                    bits: hits_fmt.bits(),
                    overlap: overlap_info,
                }),
                q4_alt: None,
                bits_sel,
            });
        } else if self.mixed
            && want_q4
            && !fmts.is_empty()
            && fmts.iter().all(|f| {
                matches!(f, BlobFormat::Q4G64F16S | BlobFormat::Q3G128F16S)
            })
        {
            // ── 混档双视图直读（6.2-2 Stage B，2026-10-20）──
            // 两个 cache 各上传一份与命中列表对齐的 slot_ids（本格式段有效、他格式段填 0）；
            // 消费侧按 bits_sel 切同格式连续段，各段用对应视图跑 GEMV——buffer 全局按对
            // 下标写、每对恰好被一个格式写 ⇒ combine 无需改动、无累加语义问题。
            // v1：混档禁用 hit/miss 重叠（overlap=None）。
            let sc = unsafe { &mut *self.scratch.get() };
            ph = (sc.ids_phase & 1) as usize;
            sc.ids_phase ^= 1;
            if sc.slot_ids[ph].as_ref().map_or(0, |v| v.len()) < s {
                sc.slot_ids[ph] = Some(unsafe { self.k.stream.alloc::<i32>(s) }?);
            }
            if sc.slot_ids_alt[ph].as_ref().map_or(0, |v| v.len()) < s {
                sc.slot_ids_alt[ph] = Some(unsafe { self.k.stream.alloc::<i32>(s) }?);
            }
            if sc.ids_pin[ph].as_ref().map_or(0, |p| p.len()) < s {
                sc.ids_pin[ph] = Some(unsafe {
                    self.k.stream.context().alloc_pinned_with_flags::<i32>(s, 0)
                }?);
            }
            if sc.ids_pin_alt[ph].as_ref().map_or(0, |p| p.len()) < s {
                sc.ids_pin_alt[ph] = Some(unsafe {
                    self.k.stream.context().alloc_pinned_with_flags::<i32>(s, 0)
                }?);
            }
            if sc.stub_gu.is_none() {
                sc.stub_gu = Some(unsafe { self.k.stream.alloc::<bf16>(1) }?);
            }
            if sc.stub_dn.is_none() {
                sc.stub_dn = Some(unsafe { self.k.stream.alloc::<bf16>(1) }?);
            }
            if lingxi_model::flag("LINGXI_ARENA_SYNC_IDS") {
                self.k.stream.synchronize()?;
            }
            {
                let pin = sc.ids_pin[ph].as_mut().unwrap();
                let buf = pin.as_mut_slice()?;
                let pin_a = sc.ids_pin_alt[ph].as_mut().unwrap();
                let buf_a = pin_a.as_mut_slice()?;
                for j in 0..s {
                    // ★ 2026-10-22 单 launch 化：他格式段从哨兵 0 改成 **-1**（0 是合法槽号，
                    // 会在整段单 launch 下被当真槽读）。核里 `slot < 0` 早退跳过。
                    if which[j] == 0 {
                        buf[j] = slot_ids[j] as i32;
                        buf_a[j] = -1;
                    } else {
                        buf[j] = -1;
                        buf_a[j] = slot_ids[j] as i32;
                    }
                }
            }
            if std::env::var("LINGXI_MIX_DEBUG").is_ok() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 3 {
                    eprintln!(
                        "[mix-dbg] layer={layer} s={s} which={:?} slot_ids={:?} fmts={:?}",
                        &which[..s.min(16)],
                        &slot_ids[..s.min(16)],
                        &fmts[..s.min(16)],
                    );
                }
            }
            self.bind_ctx()?;
            for (dev, pin) in [
                (sc.slot_ids[ph].as_mut().unwrap(), sc.ids_pin[ph].as_ref().unwrap()),
                (
                    sc.slot_ids_alt[ph].as_mut().unwrap(),
                    sc.ids_pin_alt[ph].as_ref().unwrap(),
                ),
            ] {
                let (dp, _g) = dev.device_ptr_mut(&self.k.stream);
                let src = pin.as_slice()?;
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        dp,
                        &src[..s],
                        self.k.stream.cu_stream(),
                    )?;
                }
            }
            if std::env::var("LINGXI_MIX_DEBUG").is_ok() {
                // 回读校验：设备侧 slot_ids 必须与宿主意图一致。
                self.k.stream.synchronize()?;
                for (tag, dev) in [
                    ("pri", sc.slot_ids[ph].as_ref().unwrap()),
                    ("alt", sc.slot_ids_alt[ph].as_ref().unwrap()),
                ] {
                    use cudarc::driver::DevicePtr;
                    let (p, _g) = dev.device_ptr(&self.k.stream);
                    let mut back = vec![0i32; s];
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_sync::<i32>(&mut back, p)
                    }?;
                    eprintln!("[mix-dbg] layer={layer} {tag} dev={:?}", &back[..s.min(16)]);
                }
                // 槽内容回读：抽 2 个 Q4 + 2 个 Q3 专家，设备槽字节 vs arena 源字节。
                 if layer == 2 && s == 10 {
                    let mut checked = (0u32, 0u32);
                    for j in 0..s {
                        let want_q4 = which[j] == 0;
                        let cnt = if want_q4 { &mut checked.0 } else { &mut checked.1 };
                        if *cnt >= 2 {
                            continue;
                        }
                        *cnt += 1;
                        let e = experts[j];
                        let b = self.arena.routed(layer, e)?.unwrap();
                        let src = self.src_bytes(&(b.gu.start..b.down.end));
                        let (cache, cblob) = if want_q4 {
                            (unsafe { &*self.cache.get() }, self.blob_pri)
                        } else {
                            (unsafe { &*self.cache_alt.as_ref().unwrap().get() }, self.blob_alt)
                        };
                        let off = slot_ids[j] * cblob;
                        let sv = cache.slots.slice(off..off + src.len());
                        let (p, _g) = sv.device_ptr(&self.k.stream);
                        let mut back = vec![0u8; src.len()];
                        unsafe {
                            cudarc::driver::result::memcpy_dtoh_sync::<u8>(&mut back, p)
                        }?;
                        let bad = back.iter().zip(src.iter()).filter(|(a, b)| a != b).count();
                        eprintln!(
                            "[mix-dbg] layer={layer} e={e} q4={want_q4} slot={} 槽内容不符字节={}/{}",
                            slot_ids[j], bad, src.len()
                        );
                    }
                }
            }
            // 先取走 bits_sel，再归还簿记。
            let bits_sel: Vec<u8> = fmts.iter().map(|f| f.bits() as u8).collect();
            if std::env::var("LINGXI_MIX_DEBUG").is_ok() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 3 {
                    eprintln!(
                        "[mix-dbg] layer={layer} s={s} which={:?} slot_ids={:?} fmts={:?}",
                        &which[..s.min(16)],
                        &slot_ids[..s.min(16)],
                        &fmts[..s.min(16)],
                    );
                }
            }
            LH_WS.with(|c| {
                c.replace(lh);
            });
            let _kpad = |k: usize| ((k as u64 + 127) / 128 * 128) as i32;
            let lay = |f: BlobFormat, r: u64, c: u64| {
                matrix_layout(f, r, c).context("混档段几何非法")
            };
            let gu4 = lay(BlobFormat::Q4G64F16S, rows_gu as u64, hidden as u64)?;
            let dn4 = lay(BlobFormat::Q4G64F16S, rows_dn as u64, inter as u64)?;
            let gu3 = lay(BlobFormat::Q3G128F16S, rows_gu as u64, hidden as u64)?;
            let dn3 = lay(BlobFormat::Q3G128F16S, rows_dn as u64, inter as u64)?;
            let cache_pri = unsafe { &*self.cache.get() };
            let cache_alt = unsafe { &*self.cache_alt.as_ref().unwrap().get() };
            let sc = unsafe { &*self.scratch.get() };
            fn mk_q4slots<'a>(
                slots: &'a cudarc::driver::CudaSlice<u8>,
                ids: &'a cudarc::driver::CudaSlice<i32>,
                blob: usize,
                gu: &lingxi_artifact::expert_arena::MatrixLayout,
                dn: &lingxi_artifact::expert_arena::MatrixLayout,
                bits: u32,
                hidden: usize,
                inter: usize,
            ) -> lingxi_model::moe::Q4Slots<'a> {
                let kpad = |k: usize| ((k as u64 + 127) / 128 * 128) as i32;
                lingxi_model::moe::Q4Slots {
                    slots,
                    slot_ids: ids,
                    blob: blob as u64,
                    gu_seg: 0,
                    dn_seg: gu.payload_bytes,
                    gu_scale: gu.scale_offset,
                    dn_scale: dn.scale_offset,
                    gu_k_pad: kpad(hidden),
                    dn_k_pad: kpad(inter),
                    bits,
                    overlap: None,
                }
            }
            let q4_view = mk_q4slots(
                &cache_pri.slots,
                sc.slot_ids[ph].as_ref().unwrap(),
                self.blob_pri,
                &gu4,
                &dn4,
                4,
                hidden,
                inter,
            );
            let q4a_view = mk_q4slots(
                &cache_alt.slots,
                sc.slot_ids_alt[ph].as_ref().unwrap(),
                self.blob_alt,
                &gu3,
                &dn3,
                3,
                hidden,
                inter,
            );
            return Ok(lingxi_model::moe::ExpertHits {
                gate_up: sc.stub_gu.as_ref().unwrap(),
                down: sc.stub_dn.as_ref().unwrap(),
                q4: Some(q4_view),
                q4_alt: Some(q4a_view),
                bits_sel,
            });
        } else {
            let sc = unsafe { &mut *self.scratch.get() };
            if sc.gu.as_ref().map_or(0, |v| v.len()) < s * per_gu {
                sc.gu = Some(unsafe { self.k.stream.alloc::<bf16>(s * per_gu) }?);
            }
            if sc.dn.as_ref().map_or(0, |v| v.len()) < s * per_dn {
                sc.dn = Some(unsafe { self.k.stream.alloc::<bf16>(s * per_dn) }?);
            }
            let gu = sc.gu.as_mut().unwrap();
            let dn = sc.dn.as_mut().unwrap();
            self.bind_ctx()?;
            for i in 0..s {
                // 混档：逐专家回到**自己的** cache（槽步长/段几何随 blob 格式）。
                // `which[i]` 与上方槽定位循环同源（0=主/Q4，1=次/Q3）；单格式全 0。
                let use_alt = which[i] != 0;
                let (cache, cblob): (&SlotCache, usize) = if use_alt {
                    (
                        unsafe { &*self.cache_alt.as_ref().unwrap().get() },
                        self.blob_alt,
                    )
                } else {
                    (unsafe { &*self.cache.get() }, self.blob_pri)
                };
                let mut og = gu.slice_mut(i * per_gu..(i + 1) * per_gu);
                let mut od = dn.slice_mut(i * per_dn..(i + 1) * per_dn);
                match fmts[i] {
                    BlobFormat::Bf16Raw => {
                        // 未量化：槽里就是原始字节，两段各 dtod 到 scratch。
                        // 元素类型不同（u8 源 / bf16 目标）→ 走原始指针版。
                        // 段长用 **BF16 自己的** 布局（混档下不能用默认格式的 gu_bytes）。
                        let gu_b = matrix_layout(BlobFormat::Bf16Raw, rows_gu as u64, hidden as u64)
                            .context("gu 段几何非法")?
                            .payload_bytes as usize;
                        let dn_b = matrix_layout(BlobFormat::Bf16Raw, rows_dn as u64, inter as u64)
                            .context("dn 段几何非法")?
                            .payload_bytes as usize;
                        let base = slot_ids[i] * cblob;
                        {
                            let (gp, _a) = og.device_ptr_mut(&self.k.stream);
                            let gsrc = cache.slots.slice(base..base + gu_b);
                            let (sp, _b) = gsrc.device_ptr(&self.k.stream);
                            unsafe {
                                cudarc::driver::result::memcpy_dtod_async(
                                    gp,
                                    sp,
                                    gu_b,
                                    self.k.stream.cu_stream(),
                                )?;
                            }
                        }
                        {
                            let (dp, _c) = od.device_ptr_mut(&self.k.stream);
                            let dsrc = cache.slots.slice(base + gu_b..base + gu_b + dn_b);
                            let (sq, _d) = dsrc.device_ptr(&self.k.stream);
                            unsafe {
                                cudarc::driver::result::memcpy_dtod_async(
                                    dp,
                                    sq,
                                    dn_b,
                                    self.k.stream.cu_stream(),
                                )?;
                            }
                        }
                    }
                    fmt @ (BlobFormat::Q4G64F16S | BlobFormat::Q3G128F16S) => {
                        let gu_lay = matrix_layout(fmt, rows_gu as u64, hidden as u64)
                            .context("gu 段几何非法")?;
                        let dn_lay = matrix_layout(fmt, rows_dn as u64, inter as u64)
                            .context("dn 段几何非法")?;
                        let base = slot_ids[i] * cblob;
                        let gv = cache.slots.slice(base..base + gu_lay.payload_bytes as usize);
                        // dn 段紧跟**本格式自己的** gu 净长之后（混档下不能用 arena 默认格式的 gu_bytes）。
                        let dv = cache.slots.slice(
                            base + gu_lay.payload_bytes as usize
                                ..base + gu_lay.payload_bytes as usize + dn_lay.payload_bytes as usize,
                        );
                        if fmt == BlobFormat::Q4G64F16S {
                            self.k.dequant_rowsplit_q4g64_bf16_view(
                                &gv, &mut og, rows_gu, hidden, gu_lay.scale_offset,
                            )?;
                            self.k.dequant_rowsplit_q4g64_bf16_view(
                                &dv, &mut od, rows_dn, inter, dn_lay.scale_offset,
                            )?;
                        } else {
                            self.k.dequant_rowsplit_q3g128_bf16_view(
                                &gv, &mut og, rows_gu, hidden, gu_lay.scale_offset,
                            )?;
                            self.k.dequant_rowsplit_q3g128_bf16_view(
                                &dv, &mut od, rows_dn, inter, dn_lay.scale_offset,
                            )?;
                        }
                    }
                    fmt => {
                        anyhow::bail!("不支持的 blob 格式（逐专家回退）: {:?}", fmt);
                    }
                }
            }
        }

        // P15-⑱：归还本批簿记工作区。
        LH_WS.with(|c| {
            c.replace(lh);
        });
        let sc = unsafe { &*self.scratch.get() };
        Ok(lingxi_model::moe::ExpertHits {
            gate_up: sc.gu.as_ref().unwrap(),
            down: sc.dn.as_ref().unwrap(),
            q4: None,
            q4_alt: None,
            bits_sel: Vec::new(),
        })
    }
}
