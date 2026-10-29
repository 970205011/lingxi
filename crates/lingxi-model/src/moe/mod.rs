//! MoE（稀疏专家）前向。数学对齐 HF `Qwen4ExpTextSparseMoeBlock`：
//! ```text
//! logits = router(hs)                     # [n, E]
//! probs  = softmax(logits.float); top-k -> 归一化（norm_topk_prob=True）
//! for e in 命中的专家:
//!     gate, up = split(gate_up_e @ hs_e)   # [n_e, 2I] → chunk(2, -1)
//!     h = silu(gate) * up
//!     out[idx] += probs_e * (down_e @ h)
//! shared = sigmoid(shared_expert_gate(hs)) * down(swiglu(gate_up(hs)))
//! out = expert_out + shared
//! ```

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

// CODE_MAP Phase 3 刀P3-1：moe.rs → 目录模块（mod=API 面 / router=路由核 / forward=主前向）。
// 对外 `moe::` 路径全部经本文件 re-export，调用方零改动。

mod batch_dev;
mod forward;
mod router;

pub use forward::sparse_moe_forward;
pub(crate) use forward::slots_rpw;
pub(crate) use router::route_dev_attempt;

/// **seg3 段内三分探针**（`LINGXI_SEG3_SYNC=1`，默认关）。
///
/// 为什么要有它：evprof 的 seg3（路由+host）把「shared 专家 / router+topk / 设备端建 pair 表」
/// 全包在一起，而这段在跨运行之间会从 0.28 跳到 0.85 ms（见 docs §十二 B），
/// 用两次不同运行的差值去归因是**不可信**的（§十一 D 的 0.19 ms 就是这么来的，已作废）。
/// 这里在每个边界插一次流同步，用**同一次运行内**的主机钟量出三段的真耗时。
///
/// ⚠️ 同步会显著拖慢整轮 ⇒ **只在归因时开**，不要用它跑吞吐。
pub mod s3 {
    use super::*;
    use std::cell::RefCell;
    use std::time::Instant;

    thread_local! {
        /// (上一刀的时刻, [shared, router gemm, top-k, pair建表, 调用次数, 缺 t0 次数] × 设备)
        static ST: RefCell<(Option<Instant>, [[f64; 6]; 2])> =
            const { RefCell::new((None, [[0.0; 6]; 2])) };
    }

    #[inline]
    pub fn on() -> bool {
        crate::flag("LINGXI_SEG3_SYNC")
    }

    /// `LINGXI_SEG3_LAYER=<轮号>`：只在**指定的那一轮**（48 层）逐层打出来，
    /// 用来定位"某些层特别慢"是慢在哪一段。与 `LINGXI_SEG3_SYNC` 一起用。
    #[inline]
    fn layer_round() -> Option<u32> {
        std::env::var("LINGXI_SEG3_LAYER").ok().and_then(|v| v.parse::<u32>().ok())
    }

    thread_local! {
        static SEEN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        /// 本层各段已量到的 µs（`mark` 是多次独立调用，不能放在局部变量里）
        static CUR: std::cell::Cell<[f64; 4]> = const { std::cell::Cell::new([0.0; 4]) };
    }

    /// `idx`：0=段3起点，1=shared 完，2=router gemm 完，3=top-k 完，4=pair 建表完。
    pub fn mark(k: &Kernels, layer: u32, idx: usize) -> Result<()> {
        if !on() {
            return Ok(());
        }
        k.stream.synchronize()?;
        let now = Instant::now();
        let dev = k.ctx.ordinal() & 1;
        let mut cur = [0f64; 4];
        ST.with(|c| {
            let mut g = c.borrow_mut();
            let (last, acc) = &mut *g;
            if idx == 0 {
                *last = Some(now);
                CUR.with(|c| c.set([0.0; 4]));
                return;
            }
            // ★ 2026-10-25 修：没有 t0 也必须**计数**。旧写法把整段记账放在
            //   `let Some(t0) = *last else { return }` 之后 ⇒ 一旦 `mark(0)` 与
            //   `mark(2..4)` 不在同一条语句链上（或 mark(0) 没跑），后三发直接早退，
            //   表现为"同步照付钱、`[s3]` 一行不打"（200 token 实测 96 → 64.6 tok/s 却零输出）。
            //   缺 t0 的次数单独记在 acc[dev][5]，一跑就能证实/证伪这条假设。
            match *last {
                Some(t0) => {
                    let us = now.duration_since(t0).as_secs_f64() * 1e6;
                    acc[dev][idx - 1] += us;
                    CUR.with(|c| {
                        let mut v = c.get();
                        v[idx - 1] = us;
                        c.set(v);
                        cur = v;
                    });
                }
                None => acc[dev][5] += 1.0,
            }
            *last = Some(now);
            if idx == 4 {
                acc[dev][4] += 1.0;
                let n = acc[dev][4];
                if let Some(want) = layer_round() {
                    let s = SEEN.with(|c| {
                        let v = c.get();
                        c.set(v + 1);
                        v
                    });
                    if s / 48 == want {
                        eprintln!(
                            "[s3L] #{s:02} dev{dev} L{layer:<2} shared {:6.1}／router {:6.1}／topk {:6.1}／pair {:6.1} µs",
                            cur[0], cur[1], cur[2], cur[3]
                        );
                    }
                }
                if n >= 240.0 {
                    let [a, b, c2, d, _, miss] = acc[dev];
                    eprintln!(
                        "[s3] dev{dev} L{layer} n={n:.0} 层均 µs：shared {:.1}／router {:.1}／topk {:.1}／pair {:.1}（缺 t0 {miss:.0} 次）",
                        a / n, b / n, c2 / n, d / n
                    );
                    acc[dev] = [0.0; 6];
                }
            }
        });
        Ok(())
    }
}

pub trait ExpertSource: Sync {
    /// **专家段串行化锁**（two-pass 层间流水用，默认无）：
    /// 层间流水下两个线程会并发驱动**同一只** expert source（一个跑本命层、一个跑
    /// 远程层的内层调用）——`load_hits` 内部的簿记锁只保护调用本身，**返回后、
    /// GEMV 入队前**仍存在"对方把槽驱逐覆写、而本线程读该槽的核还在队列里"的窗口。
    /// 返回 `Some` 时，`sparse_moe_forward` 会把「load_hits ~ 专家 GEMV 全部入队 ~
    /// combine」整段持锁，封掉该窗口（锁内无 host 同步 ⇒ 不破坏双卡流水）。
    /// 持有检测走 [`expert_acquire`] 的线程级标记 ⇒ 与 `load_hits` 内部取同一把锁
    /// 不会死锁（嵌套调用自动跳过）。
    fn expert_lock(&self) -> Option<&std::sync::Mutex<()>> {
        None
    }

    /// 按命中专家 id 列表（moe 保证已去重、按 id 升序）填充并返回**紧凑**权重。
    /// 允许（也应当）复用内部 scratch：host 聚合缓冲与 device 张量都只分配一次、
    /// 之后逐层覆写——decode 每层槽位数相同，重复分配大缓冲会造成缺页/cuMalloc 风暴。
    /// `want_q4`：调用方是否打算走「Q4 直读」快路径（decode, rows==1）。为真且数据全为 Q4G64 时，
    /// source **可以跳过解量化到 bf16**，改为在 [`ExpertHits::q4`] 里直接给槽句柄；
    /// 为假时必须填好 `gate_up`/`down`（prefill 与逐专家路径依赖它们）。
    /// `overlap`：调用方**是否会把未命中组单独排在 `k.ev_mo_miss` 之后**（`LINGXI_MOE_OVERLAP`，
    /// 只在 `rows>1` 的组列表路径上成立）。为真时 source 可以把 miss 的 blob H2D 挪到
    /// `k.stream2` 上飞、并顺手记录事件；**为假时必须照旧放在 compute 流上**（否则调用方
    /// 不等待 ⇒ GEMV 会读到还没拷完的槽）。
    fn load_hits<'a>(
        &'a self,
        layer: u32,
        experts: &[u32],
        inter: usize,
        hidden: usize,
        want_q4: bool,
        overlap: bool,
    ) -> Result<ExpertHits<'a>>;

    /// router lookahead 预取的 **router_logits 信号供给**（`LINGXI_ARENA_PREFETCH=1`
    /// 且 `LINGXI_ARENA_PREFETCH_SIGNAL=router_logits`）：在 **router GEMV 已入队、
    /// 首个同步尚未发生**的窗口调用。实现方登记本层 router 权重的设备地址，并在
    /// **独立流**上异步发起"层 L+1 的 router 权重 × 本层 router 输入 x"的预测
    /// GEMV（真 logits，radiance GUIDE §4.3），预测结果在同一层随后的
    /// `load_hits`（pf_issue）里消费、发起 stream2 预取 DMA。因为发射点在同步前，
    /// 本调用不得含任何流同步。默认空实现 = 非 arena 源（/ 旧信号档）零开销。
    fn pf_predict_enqueue(
        &self,
        _layer: u32,
        _x: &CudaSlice<bf16>,
        _router_w_dev: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// ㊺ A2 设备路由：设备侧 Q4 直读视图（槽区 + 每层槽位表 + 段几何常量）。
    /// `Some` = 支持设备端 top-k 直查（免 host 同步决策）；`None` = 回退 host 路径。
    fn q4_dev(&self) -> Option<Q4DevView<'_>> {
        None
    }

    /// ㊺ A2：把 `experts` 里**尚未入槽**的搬进设备槽（已在槽里的是空操作），并保证
    /// `prot_pri`/`prot_alt`（本批已在用的槽，按 cache 分开）以及本批新分配的槽不会被
    /// `victim()` 选中。实现方**必须**让槽位表的改动对设备可见（含被淘汰的**其它层**的槽位行）。
    /// 混档 arena 按各专家自己的格式落到对应 cache；`prot_alt` 在单格式实现里忽略。
    fn load_missing(
        &self,
        _layer: u32,
        _experts: &[u32],
        _prot_pri: &[usize],
        _prot_alt: &[usize],
    ) -> Result<()> {
        Ok(())
    }

    /// ㊺ A2：把 `layer` 这一层的槽位表（E 项）同步上卡。**只在 legacy（miss 回退）
    /// load_hits 之后调用**——设备路由快路径全中时 load_hits 根本不被调用，表无需改动。
    fn slot_map_sync_layer(&self, _layer: u32) -> Result<()> {
        Ok(())
    }

    /// **EP 均衡分配**：该 `(层, 专家)` 的设备槽字节数（混档 arena 下 Q4 热 ≈2.6 MB /
    /// Q3 冷 ≈1.92 MB）。默认 `None` = 不知道 ⇒ 分配退回 id 二分（旧行为）。
    fn slot_bytes(&self, _layer: u32, _expert: u32) -> Option<u64> {
        None
    }
}

/// **EP 分片**（`LINGXI_EP=1`）：本卡只算 `assign[e] == card` 的专家。
///
/// `assign` 是**每层一张**、长度 `num_experts` 的 `u8` 表（见 [`build_ep_expert_assign`]）；
/// 两卡读同一份表 ⇒ 同一 token 的同一专家只会落在一张卡上，路由结果不会分叉。
#[derive(Clone, Copy)]
pub struct EpSlice<'a> {
    pub assign: &'a [u8],
    pub card: u8,
}

/// 按「专家槽字节 × 命中频次」做**贪心均衡**，产出每层 `expert -> card(0/1)` 表。
///
/// 为什么不再按 id 二分：混档 arena 里热专家（Q4，2.6 MB）几乎全落在小 id 侧，按 id 切会
/// 把一边压成另一边 ~2 倍的专家段耗时（实测 1.009 vs 0.470 ms/层）。
/// 做法：每层所有专家按 `bytes × (1 + 历史命中数)` 降序，逐个放进当前**代价更轻**的卡 ⇒
/// 热专家天然交错（第 1 个→卡0、第 2 个→卡1 …），且两边总字节近似相等。
///
/// `slot_bytes(l, e)`：来自 [`ExpertSource::slot_bytes`]；任一层拿不到字节（非 arena 源）
/// 就对该层退回 id 二分。频次取自 `LINGXI_ARENA_PREWARM` 的 `layer,expert,count` CSV
/// （读不到按 1 计，即纯按字节均衡）。全部是确定性计算、只构建一次 ⇒ 两卡逐位同表。
pub fn build_ep_expert_assign(
    n_layers: u32,
    num_experts: usize,
    slot_bytes: impl Fn(u32, u32) -> Option<u64>,
) -> Vec<u8> {
    use std::cmp::Ordering;
    let n = n_layers as usize;
    // `LINGXI_EP_ASSIGN=id`：退回旧的 id 二分（A/B 定位用；默认按均衡表）。
    if std::env::var("LINGXI_EP_ASSIGN").as_deref() == Ok("id") {
        let mut out = vec![0u8; n * num_experts];
        for l in 0..n {
            for e in 0..num_experts {
                out[l * num_experts + e] = if e < num_experts / 2 { 0 } else { 1 };
            }
        }
        eprintln!("[ep-balance] LINGXI_EP_ASSIGN=id：退回 id 二分（仅定位用）");
        return out;
    }
    // 历史命中频次（可选）：`# layer,expert,count`
    let mut hits = vec![0u32; n * num_experts];
    if let Ok(path) = std::env::var("LINGXI_ARENA_PREWARM") {
        if let Ok(txt) = std::fs::read_to_string(&path) {
            for line in txt.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let mut it = line.split(',');
                let (Some(l), Some(e), Some(c)) = (it.next(), it.next(), it.next()) else {
                    continue;
                };
                if let (Ok(l), Ok(e), Ok(c)) =
                    (l.parse::<u32>(), e.parse::<usize>(), c.parse::<u64>())
                {
                    if (l as usize) < n && e < num_experts {
                        hits[l as usize * num_experts + e] =
                            c.min(u32::MAX as u64) as u32;
                    }
                }
            }
        }
    }
    let mut out = vec![0u8; n * num_experts];
    let mut bytes_card = [0f64; 2];
    let mut cost_card = [0f64; 2];
    let mut worst = 0f64;
    for l in 0..n {
        let row = &hits[l * num_experts..(l + 1) * num_experts];
        let mut w = vec![0f64; num_experts];
        let mut known = true;
        for e in 0..num_experts {
            match slot_bytes(l as u32, e as u32) {
                Some(b) => w[e] = b as f64 * (1.0 + row[e] as f64),
                None => {
                    known = false;
                    break;
                }
            }
        }
        let base = l * num_experts;
        if !known {
            for e in 0..num_experts {
                out[base + e] = if e < num_experts / 2 { 0 } else { 1 };
            }
            continue;
        }
        let mut order: Vec<usize> = (0..num_experts).collect();
        order.sort_by(|&a, &b| {
            w[b].partial_cmp(&w[a]).unwrap_or(Ordering::Equal).then(a.cmp(&b))
        });
        let (mut s0, mut s1) = (0f64, 0f64);
        let (mut b0, mut b1) = (0f64, 0f64);
        for e in order {
            let c = if s0 <= s1 { 0 } else { 1 };
            out[base + e] = c;
            if c == 0 {
                s0 += w[e];
                b0 += slot_bytes(l as u32, e as u32).unwrap_or(0) as f64;
            } else {
                s1 += w[e];
                b1 += slot_bytes(l as u32, e as u32).unwrap_or(0) as f64;
            }
        }
        bytes_card[0] += b0;
        bytes_card[1] += b1;
        cost_card[0] += s0;
        cost_card[1] += s1;
        worst = worst.max(s0.max(s1) / s0.min(s1).max(1e-9));
    }
    eprintln!(
        "[ep-balance] 每层 {num_experts} 专家贪心均衡（字节×频次）：槽字节 卡0 {:.2} GiB / 卡1 {:.2} GiB；\
         代价 卡0 {:.2} / 卡1 {:.2}；最差层比 {worst:.3}×（目标 ≤1.15）",
        bytes_card[0] / 1073741824.0,
        bytes_card[1] / 1073741824.0,
        cost_card[0] / 1073741824.0,
        cost_card[1] / 1073741824.0,
    );
    out
}

/// ㊺ A2 设备路由用的 Q4 常量（`Q4Slots` 里不随命中变化的部分）。
#[derive(Clone, Copy)]
pub struct Q4Consts {
    pub blob: u64,
    pub gu_seg: u64,
    pub dn_seg: u64,
    pub gu_scale: u64,
    pub dn_scale: u64,
    pub gu_k_pad: i32,
    pub dn_k_pad: i32,
    /// 权重位宽（4 = Q4G64F16S，3 = Q3G128F16S）——选核用（Q3 无 TC/dp4a 档）。
    pub bits: u32,
}

/// ㊺ A2 设备路由视图：`slot_map` 是 `[层, E]` 的 i32 表（存 slot+1、0=未缓存）。
/// 混档 arena（Q4G64+Q3G128）再给 `alt`（次格式 cache 的槽区/表/几何）与 `fmt`
/// （`[层, E]`：0=主视图、1=次视图）——批量设备路由据此一次建两组槽号（互斥 -1 哨兵）。
pub struct Q4DevView<'a> {
    pub slots: &'a cudarc::driver::CudaSlice<u8>,
    pub slot_map: &'a cudarc::driver::CudaSlice<i32>,
    pub consts: Q4Consts,
    pub alt: Option<Q4DevAlt<'a>>,
    pub fmt: Option<&'a cudarc::driver::CudaSlice<i32>>,
}

/// 混档次格式 cache 的设备视图（与主视图同构）。
pub struct Q4DevAlt<'a> {
    pub slots: &'a cudarc::driver::CudaSlice<u8>,
    pub slot_map: &'a cudarc::driver::CudaSlice<i32>,
    pub consts: Q4Consts,
}

/// `LINGXI_MOE_OVERLAP` 是否**真的生效**：它依赖"专家段不受 CUDA Graph 捕获"（跨流 + 事件
/// 在捕获里是另一套语义）。所以要求同时设 `LINGXI_NO_GRAPHS=1`（实测关图本来就更快，§P13.3）。
/// 只设了本开关却没关图时**大声告警并自动退回老路径**（宁可慢，也不静默错）。
pub fn moe_overlap_on() -> bool {
    if !crate::flag("LINGXI_MOE_OVERLAP") {
        return false;
    }
    // ★ P15-㊶：判定从"env 设了 NO_GRAPHS"改成"graph 真的启用"。原判定有洞：
    // 双卡自动 layer-split 时图是被 force_disable() 关掉的（不要求用户设 env），
    // 于是 overlap 在生产默认配置下被误报"失效"——其实图根本没开。
    if crate::graph::enabled() {
        static WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[moe-overlap] ⚠️ LINGXI_MOE_OVERLAP=1 但 CUDA Graph 启用中 ⇒ **本开关自动失效**\
                 （跨流+事件在 capture 里语义不同）。请同时设 LINGXI_NO_GRAPHS=1。"
            );
        }
        return false;
    }
    true
}

// ─────────────────────── ㊺ A2 设备路由：辅助 + 快路径 ───────────────────────

/// 诊断对拍用：FNV-1a 64 位哈希（`LINGXI_PAIR_DUMP` 比较 host/设备构建的 pair/CSR 表）。
pub(crate) fn hash_i32(v: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in v {
        h = (h ^ (x as u32 as u64)).wrapping_mul(0x100000001b3);
    }
    h
}

/// 诊断对拍用：pair 行号 + 权重位型（两段分别滚动，保证 `(行, 权重)` 对齐后哈希一致）。
pub(crate) fn hash_pair_tables(rows: &[i32], wts: &[f32]) -> u64 {
    let mut h = hash_i32(rows);
    for &w in wts {
        h = (h ^ (w.to_bits() as u64)).wrapping_mul(0x100000001b3);
    }
    h
}



/// 一次 [`ExpertSource::load_hits`] 的紧凑权重借用：gate_up `[S, 2I, H]`、
/// down `[S, H, I]`，槽位 `s` 对应请求的 `experts[s]`。
///
/// 借用绑在 source 上：source 内部复用同一对 scratch 缓冲（跨层跨步覆写），
/// 故调用方必须在**下一次 load_hits 之前**用完（单层 MoE 内天然满足）。
pub struct ExpertHits<'a> {
    pub gate_up: &'a CudaSlice<bf16>,
    pub down: &'a CudaSlice<bf16>,
    /// **Q4 直读通道**（可选）：提供时 decode 批量路径**跳过 bf16 中转**，用融合反量化的
    /// GEMV 直接从设备槽读 Q4 权重。此时 `gate_up`/`down` 不被使用。
    pub q4: Option<Q4Slots<'a>>,
    /// **混档次格式（Q3）直读视图**（6.2-2 Stage B）：`Some` = 本批专家含 Q3G128，
    /// 视图与 `q4` 同构但 `slots`/`slot_ids`/`blob`/段几何属 Q3 cache。
    /// 消费侧按 `bits_sel` 把命中列表切成同格式连续段、各段用对应视图跑（buffer 全局
    /// 按对下标写 ⇒ combine 不用改、无累加问题）。
    pub q4_alt: Option<Q4Slots<'a>>,
    /// 与传给 `load_hits` 的命中列表**对齐**的每专家位宽（4=Q4G64 / 3=Q3G128；
    /// 直读路径用它切格式段；bf16 回退路径忽略）。直读两视图时长度 = 列表长。
    pub bits_sel: Vec<u8>,
}

/// 设备槽里的 Q4G64 专家权重句柄（配套 [`ExpertHits::q4`]）。
pub struct Q4Slots<'a> {
    pub slots: &'a CudaSlice<u8>,
    pub slot_ids: &'a CudaSlice<i32>,
    /// 单个槽（gu+dn）的字节数
    pub blob: u64,
    /// gu / dn 段在 blob 内的起点
    pub gu_seg: u64,
    pub dn_seg: u64,
    /// gu / dn 段内 scale 平面偏移
    pub gu_scale: u64,
    pub dn_scale: u64,
    /// 列数补齐（Q4 每行字节数 = `k_pad / 2`、Q3 = `k_pad / 128 * 48`）
    pub gu_k_pad: i32,
    pub dn_k_pad: i32,
    /// 权重位宽（4 = Q4G64F16S，3 = Q3G128F16S）——decode/prefill 选核用。
    pub bits: u32,
    /// **专家段 hit/miss 重叠**（`LINGXI_MOE_OVERLAP=1`）：`Some` 时 miss 的 blob H2D 是在
    /// **第二流**上飞的，调用方必须把"未命中组"的 GEMV 排在 `k.ev_mo_miss` 之后。
    /// `None` = 老行为（H2D 与 GEMV 都在 compute 流上、严格串行）。
    pub overlap: Option<MoOverlap>,
}

/// 专家段"命中/未命中"的分组切分（配合 [`Q4Slots::overlap`]）。**只在 `rows > 1` 的
/// 组列表路径上有意义**（那时 `slot_ids[i]` 与"唯一专家 i"一一对应）。
///
/// 为什么能这么切而**不改变数学**：`gemv_q4g64_slotsm_*` 里每个组的读写在
/// **绝对对下标**上（`xb = x + s0*a_stride`、`y[(s0+s)*n + row]`，见 dequant_ops.cu），
/// 所以把组的连续区间拆成几次 launch，**每个对的算式、列次序、累加次序逐位不变**；
/// 而"哪些对落进哪次 launch"只影响**执行的先后**，不影响归并（`moe_combine_rows` 仍按
/// 原来的对序累加）。⇒ 判据（首 token + 固定序列）必须逐位不变。
pub struct MoOverlap {
    /// **命中组的连续区间**（组下标，升序、两两不相邻）：这些组的权重早已在槽里，
    /// 可以**不等** miss 的 H2D 就先算。
    pub hit_runs: Vec<(u32, u32)>,
    /// 未命中组的下标（升序）。空 = 本层没有 miss，无需等待。
    pub miss_groups: Vec<u32>,
}

/// 一个 MoE 块的权重（全部 bf16，无需运行时转换）。
pub struct MoeWeights {
    /// [E+1, H]：前 E 行是 router，最后一行是 `shared_expert_gate`
    pub router_shared_gate: CudaSlice<bf16>,
    /// [E*2I, H]：专家主序（每个专家 `[gate(I) ; up(I)]`）。
    /// `None` = 瘦身路径：不预装全量专家，运行时经 [`ExpertSource`] 只取命中专家。
    pub routed_gate_up: Option<CudaSlice<bf16>>,
    /// [E*H, I]；`None` 含义同 `routed_gate_up`
    pub routed_down: Option<CudaSlice<bf16>>,
    /// [2I, H]：`[gate(I) ; up(I)]`
    pub shared_gate_up: CudaSlice<bf16>,
    /// [H, I]
    pub shared_down: CudaSlice<bf16>,
    /// **W4 共享专家档**（`LINGXI_SHARED_Q4=1`，2026-10-29）：两矩阵的 Q4G64 平面；
    /// `Some` 时前向走 dequant GEMV（bf16 字段退化为 1 元素占位）。
    /// 共享专家是 dense（每 token 都参与），量化走 **GPTQ 校准**（tools/arena-gen/
    /// gptq_shared.py 产物，经 LINGXI_Q4_CACHE 名寻址缓存加载），不用 RTN。
    pub shared_gate_up_q4: Option<crate::q4::Q4W>,
    pub shared_down_q4: Option<crate::q4::Q4W>,
}

/// MoE 前向结果（含中间量，便于分段对拍）。
pub struct MoeOutput {
    /// [rows, H]
    pub out: CudaSlice<bf16>,
    /// [rows, E]；**零拷贝路径下为 `None`**（logits 直接落在映射 host 内存里，不占显存）
    pub router_logits: Option<CudaSlice<bf16>>,
    /// [rows, H]（已乘 `sigmoid(shared_expert_gate)`）
    pub shared: CudaSlice<bf16>,
    /// [rows, H]（仅稀疏专家分支）
    pub expert_out: CudaSlice<bf16>,
    /// 每个专家命中的 `(token 行, 权重)`
    pub per_expert: Vec<Vec<(i32, f32)>>,
    /// `(token 行, 专家, 权重)` 全量选择（按 token 顺序）
    pub sel_rows: Vec<(u32, i32, f32)>,
}

// ───────────── 专家段段锁（two-pass 层间流水的跨线程槽保护）─────────────

thread_local! {
    /// 本线程是否已持有某只 source 的段锁/`op_lock`（嵌套取锁的跳过判据）。
    static EXPERT_HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 专家段段锁开关（`LINGXI_PREFILL_TWOPASS=1` + `LINGXI_TWOPASS_PIPE=1` 的层间流水期间置真）。
/// 关掉时 `sparse_moe_forward` 不取段锁 ⇒ 默认/talk_spec 路径行为与以前逐字一致。
static EXPERT_LOCKS_ON: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 层间流水编排入口调：进出各一次（true/false）。
pub fn set_expert_locks_on(on: bool) {
    EXPERT_LOCKS_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// 段锁当前是否生效（`sparse_moe_forward` 查询）。
pub fn expert_locks_on() -> bool {
    EXPERT_LOCKS_ON.load(std::sync::atomic::Ordering::Relaxed)
}

/// 取专家锁（可嵌套安全）：本线程已持有（外层段锁）⇒ 返回 `None`（内部簿记已被保护）；
/// 否则取锁并打线程级持有标记，guard drop 时释放并清标记。
/// poison 容忍（`into_inner`）——与 arena 内部既有的取锁口径一致。
pub fn expert_acquire(m: &std::sync::Mutex<()>) -> Option<ExpertGuard<'_>> {
    if EXPERT_HELD.with(|c| c.get()) {
        return None;
    }
    let g = m.lock().unwrap_or_else(|e| e.into_inner());
    EXPERT_HELD.with(|c| c.set(true));
    Some(ExpertGuard { _g: g })
}

/// [`expert_acquire`] 的守卫（Drop 清线程级持有标记）。
pub struct ExpertGuard<'a> {
    // 下划线前缀：字段本体只为持有 MutexGuard（RAII 解锁），从不显式读——
    // 不叫 `_g` 的话 release 构建会报 dead_code。
    _g: std::sync::MutexGuard<'a, ()>,
}

impl<'a> Drop for ExpertGuard<'a> {
    fn drop(&mut self) {
        EXPERT_HELD.with(|c| c.set(false));
    }
}

/// `x[n, H] → out[n, H]`。
#[allow(clippy::too_many_arguments)]
/// **跨卡跑 MoE**（`LINGXI_MOE_SPLIT`，设计见 `docs/PLAN.md` §P13.1c）。
///
/// 为什么：真语料下**整轮 43% 的时间**花在"专家 miss 的 PCIe 搬运"上（603 MB/轮 @10 GB/s），
/// 根因是 GPU0 的专家槽位不够（9000 槽 / 每层 183 个 ⇒ 覆盖 88%，而 9200 就 OOM）。
/// 把**一半层的 MoE 挪到全空的 GPU1**（它只有 24 层的槽位 ⇒ 每层 500+ ⇒ 覆盖 ~100%、miss≈0），
/// 就能把这一半的 miss 消掉；跨卡只过 MoE 的**输入/输出激活**（每层 `t×hidden` bf16 = 36 KB ×2）。
///
/// **位等价**：GPU1 上跑的是同一内核、同一权重字节、同一输入 ⇒ 输出必须逐位相同
/// （判据：首 token `11751` + 固定贪心序列不变）。因此这条改动**不碰精度**。
pub struct RemoteMoe<'a> {
    /// GPU1 的 kernels
    pub k: &'a Kernels,
    /// GPU1 侧、**只服务这些层**的专家源（它的槽只装这些层的专家 ⇒ 覆盖率高、几乎不 miss）
    pub experts: &'a dyn ExpertSource,
    /// 交给 GPU1 跑的层区间
    pub layers: std::ops::Range<u32>,
    /// 按层号索引的 **GPU1 侧 MoE 权重**（`None` = 该层不在 GPU1 上）
    pub moe: &'a [Option<MoeWeights>],
    /// `(跨卡次数, 累计 µs, 其中搬运 µs)` —— 便于量"跨卡本身值多少"
    pub stat: &'a [std::sync::atomic::AtomicU64; 3],
}

/// ② 路由 host 簿记的**复用工作区**（P15-⑧ 行动清单第 1 条：消灭每层 ~512 个 Vec 分配）。
/// 为什么值得做：路由段在 `synchronize()`（取回 logits）之后、专家 GEMV 入队之前，
/// **全程压在 GPU 空闲的关键路径上**——这里每省 1 µs 就是 1 µs 的 GPU 死时间（48 层累乘）。
/// 原来这层新分配 `per_expert`（512 个内层 Vec）+ 8 张簿记表 ≈ 520 次分配，实测 ~30 µs/层。
/// 用 `take`/`replace` 取出归还（不长期持锁）；意外 early-return 丢一份也只是下次重建。
#[derive(Default)]
struct RouteWs {
    per_expert: Vec<Vec<(i32, f32)>>,
    sel_rows: Vec<(u32, i32, f32)>,
    probs: Vec<f32>,
    sel: Vec<(usize, f32)>,
    hit_ids: Vec<u32>,
    pair_experts: Vec<u32>,
    pair_rows: Vec<i32>,
    pair_wts: Vec<f32>,
    uniq_experts: Vec<u32>,
    group_off: Vec<i32>,
    rows_of: Vec<i32>,
    wts_1: Vec<f32>,
}

thread_local! {
    pub(crate) static ROUTE_WS: std::cell::RefCell<RouteWs> = const { std::cell::RefCell::new(RouteWs {
        per_expert: Vec::new(),
        sel_rows: Vec::new(),
        probs: Vec::new(),
        sel: Vec::new(),
        hit_ids: Vec::new(),
        pair_experts: Vec::new(),
        pair_rows: Vec::new(),
        pair_wts: Vec::new(),
        uniq_experts: Vec::new(),
        group_off: Vec::new(),
        rows_of: Vec::new(),
        wts_1: Vec::new(),
    }) };
}

// [探针 2026-10-22] router 段三项分解（host-bound 诊断：把 MoE 括号里的
// "等 GPU 积压"从真实 host 工作中剥出来）：
//   SYNC_WAIT_NS   = 锁页 D2H + sync 的累计（GPU 积压等待的主体）
//   ROUTER_ALLOC_NS = scratch::bf16(logits) 分配
//   ROUTER_LAUNCH_NS = router GEMM 发射（device_ptr ×3 + launch）
pub(crate) static SYNC_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static ROUTER_ALLOC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static ROUTER_LAUNCH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

