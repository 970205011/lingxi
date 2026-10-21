//! **K2 graph_timer 图内探针**（借鉴 DeepInfer `csrc/graph_timer/`，2026-10-21 落地）。
//!
//! DeepInfer 的招法：**捕获时用带 `CU_EVENT_RECORD_EXTERNAL` 的 event record**（录进图里
//! 成为 event-record 节点，而不是外部依赖），图实例化后用
//! `cuGraphExecEventRecordNodeSetEvent` 在**每次重放前**把新事件换进节点（ping-pong 双
//! 缓冲——上一代事件还能异步读回），从而拿到 **CUDA graph 内部各分段的零开销计时**。
//!
//! 本模块把它 Rust 化并挂进 `graph::run_cached`：
//! - **开关**：`LINGXI_GRAPH_TIMER=1`（默认关）。关掉时零开销（不建事件、不包 body）。
//! - **自动探针**：开关打开时 `run_cached` 在 body 前后各录一个 begin/end 探针——图路径下
//!   这两个 record 成为图的首/尾 event-record 节点；直跑路径（paged / 降级）就是普通流
//!   事件，同一套读出口径。
//! - **手动探针**：`probe(k, "name")` 可在 body 内部插点（分段时间 = 相邻探针对），同样
//!   图/直跑两用。
//! - **数值无涉**：事件不改变任何 kernel、缓冲或顺序；只读计时。
//!
//! 容错纪律（探针绝不能把 decode 搞挂）：任何驱动调用失败 → 打一次日志、全局自关
//! （`OFF`），之后一切调用变 no-op，只损计时数据不损正确性。

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::sys;
use lingxi_kernels::Kernels;

/// `LINGXI_GRAPH_TIMER=1` 打开。默认关；探针类开关，绝不进生产默认路径。
pub fn enabled() -> bool {
    !OFF.load(Ordering::Relaxed) && crate::flag("LINGXI_GRAPH_TIMER")
}

/// 任何驱动错误后永久自关。
static OFF: AtomicBool = AtomicBool::new(false);

fn check(r: sys::CUresult, what: &str) -> bool {
    if r.result().is_ok() {
        return true;
    }
    eprintln!("⚠️ [graph_timer] {what} 失败（{r:?}）—— 计时器自动关闭，只损计时不损正确性");
    OFF.store(true, Ordering::Relaxed);
    false
}

// ── 裸句柄层（absorb/drain 用，不构造 Ev 临时包装，避免误销毁已挂进图的事件） ──

fn query_ready_raw(e: sys::CUevent) -> bool {
    let r = unsafe { sys::cuEventQuery(e) };
    if r.result().is_ok() {
        return true; // CUDA_SUCCESS：已完成
    }
    if matches!(r, sys::CUresult::CUDA_ERROR_NOT_READY) {
        let _ = r.result(); // 清 sticky error 状态（NOT_READY 会留 sticky）
        return false;
    }
    check(r, "cuEventQuery");
    false
}

fn elapsed_raw(a: sys::CUevent, b: sys::CUevent) -> Option<f32> {
    let mut ms: f32 = 0.0;
    if check(unsafe { sys::cuEventElapsedTime(&mut ms, a, b) }, "cuEventElapsedTime") {
        Some(ms)
    } else {
        None
    }
}

/// 计时事件包装：拥有句柄，Drop 销毁。必须 `CU_EVENT_DEFAULT`（timing 开启，
/// `cuEventElapsedTime` 依赖它）。
struct Ev(sys::CUevent);

impl Ev {
    fn new(what: &str) -> Option<Ev> {
        let mut e: sys::CUevent = std::ptr::null_mut();
        if check(
            unsafe { sys::cuEventCreate(&mut e, sys::CUevent_flags::CU_EVENT_DEFAULT as u32) },
            what,
        ) {
            Some(Ev(e))
        } else {
            None
        }
    }
    /// 录到流上。`external=true`（捕获内）→ 成为图的 event-record 节点。
    fn record(&self, k: &Kernels, external: bool) -> bool {
        let flags = if external {
            sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL as u32
        } else {
            sys::CUevent_record_flags::CU_EVENT_RECORD_DEFAULT as u32
        };
        check(
            unsafe { sys::cuEventRecordWithFlags(self.0, k.stream.cu_stream(), flags) },
            "cuEventRecordWithFlags",
        )
    }
}

impl Drop for Ev {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let _ = unsafe { sys::cuEventDestroy_v2(self.0) };
        }
    }
}

/// 把一串 (名, 裸事件句柄) 里**已完成的相邻对**计入统计环。最后一个事件未完成则
/// 整串留待下次（事件按流序完成，尾事件没好就整串不算，避免半遍污染）。
fn absorb(s: &mut KeyState, names: &[String], evs: &[sys::CUevent]) {
    if evs.len() < 2 || !query_ready_raw(*evs.last().unwrap()) {
        return;
    }
    // 整段 begin→end（只有首尾自动探针时就是唯一段）。
    if let (Some(&first), Some(&last)) = (evs.first(), evs.last()) {
        if let Some(ms) = elapsed_raw(first, last) {
            push_ring(&mut s.spans, ms);
        }
    }
    // 相邻分段。
    for (ew, nw) in evs.windows(2).zip(names.windows(2)) {
        if let Some(ms) = elapsed_raw(ew[0], ew[1]) {
            let key = format!("{}→{}", nw[0], nw[1]);
            push_ring(s.segs.entry(key).or_default(), ms);
        }
    }
}

const RING_CAP: usize = 512;

fn push_ring(r: &mut VecDeque<f32>, v: f32) {
    if r.len() >= RING_CAP {
        r.pop_front();
    }
    r.push_back(v);
}

/// 一个捕获单元的计时状态（键与 `graph::CACHE` 同构：`(base, epoch)`）。
struct KeyState {
    /// 探针名（与 `nodes`/`gens` 对齐；顺序 = 捕获注册序，经烘焙句柄匹配回填）。
    names: Vec<String>,
    /// 图里的 event-record 节点（枚举序——图节点序不保证等于注册序，所以名字靠
    /// 烘焙事件的句柄匹配回填，见 `on_captured`）。
    nodes: Vec<sys::CUgraphNode>,
    /// ping-pong 两代事件：每次重放前把 `gens[next_gen]` 换进节点，上一代留给异步读回。
    gens: [Vec<Ev>; 2],
    next_gen: usize,
    /// 最近一次重放用的是哪一代（尚未被 drain 消费）。
    last_used: Option<usize>,
    /// 直跑路径：本遍已录的 (名, 事件)。
    direct: Vec<(String, Ev)>,
    /// 统计环：整段 begin→end。
    spans: VecDeque<f32>,
    /// 统计环：相邻探针对 "a→b"。
    segs: HashMap<String, VecDeque<f32>>,
}

impl KeyState {
    fn new() -> Self {
        KeyState {
            names: Vec::new(),
            nodes: Vec::new(),
            gens: [Vec::new(), Vec::new()],
            next_gen: 0,
            last_used: None,
            direct: Vec::new(),
            spans: VecDeque::new(),
            segs: HashMap::new(),
        }
    }
}

thread_local! {
    /// `(base, epoch) -> 状态`，与 `graph::CACHE` 同寿命；`purge` 同步回收。
    static GT: RefCell<HashMap<(u64, u64), KeyState>> = RefCell::new(HashMap::new());
    /// 当前 run_cached 的键栈（支持嵌套：内层 enter/leave 恢复外层归属）。
    static CURRENT: RefCell<Vec<(u64, u64)>> = const { RefCell::new(Vec::new()) };
    /// 捕获期内登记的探针（带键：嵌套捕获时内层 on_captured 只取自己的那份）。
    static PENDING: RefCell<Vec<((u64, u64), String, sys::CUevent)>> =
        const { RefCell::new(Vec::new()) };
}

/// run_cached 入口登记（嵌套安全）。
pub fn enter(key: (u64, u64)) {
    if enabled() {
        CURRENT.with(|c| c.borrow_mut().push(key));
    }
}

/// run_cached 出口（与 enter 配对）。
pub fn leave() {
    if enabled() {
        CURRENT.with(|c| {
            c.borrow_mut().pop();
        });
    }
}

/// 手动探针：body 内部插点。图捕获内 → 录成节点；直跑 → 录成流事件。
/// 开关关 / 无 CURRENT 时是 no-op。
pub fn probe(k: &Kernels, name: &str) {
    if !enabled() {
        return;
    }
    let key = match CURRENT.with(|c| c.borrow().last().copied()) {
        Some(k) => k,
        None => return,
    };
    let external = crate::graph::is_capturing();
    let ev = match Ev::new("cuEventCreate") {
        Some(e) => e,
        None => return,
    };
    if !ev.record(k, external) {
        return;
    }
    if external {
        // 句柄所有权转交 PENDING → on_captured 的 gens[0]；forget 防 Drop 销毁。
        PENDING.with(|p| p.borrow_mut().push((key, name.to_string(), ev.0)));
        std::mem::forget(ev);
    } else {
        GT.with(|m| m.borrow_mut().entry(key).or_insert_with(KeyState::new).direct.push((name.to_string(), ev)));
    }
}

/// 图捕获成功并实例化后调用：枚举 event-record 节点，按**烘焙事件句柄**匹配回探针名
/// （图节点枚举序 ≠ 注册序，句柄是唯一可靠关联），建立 ping-pong 两代事件。
pub fn on_captured(key: (u64, u64), g: &cudarc::driver::CudaGraph) {
    // 无论成功与否，本键的 pending 都清掉（句柄要么被 gens[0] 接管，要么归还销毁）。
    let pend: Vec<(String, sys::CUevent)> = PENDING.with(|p| {
        p.borrow_mut().drain(..).filter(|(k, _, _)| *k == key).map(|(_, n, e)| (n, e)).collect()
    });
    if !enabled() {
        drop(pend.into_iter().map(|(_, e)| Ev(e)).collect::<Vec<_>>());
        return;
    }
    // 1. 枚举图节点，挑出 event-record 节点。
    let mut num: usize = 0;
    if !check(
        unsafe { sys::cuGraphGetNodes(g.cu_graph(), std::ptr::null_mut(), &mut num) },
        "cuGraphGetNodes(计数)",
    ) {
        return;
    }
    let mut raw_nodes = vec![std::ptr::null_mut(); num];
    if num > 0
        && !check(
            unsafe { sys::cuGraphGetNodes(g.cu_graph(), raw_nodes.as_mut_ptr(), &mut num) },
            "cuGraphGetNodes",
        )
    {
        return;
    }
    // 2. 逐节点 GetEvent → 在 pending 里按句柄找名字（O(n²) 无所谓，n=探针数≤几十）。
    let mut names = Vec::new();
    let mut nodes = Vec::new();
    let mut baked: Vec<Ev> = Vec::new();
    let mut matched = vec![false; pend.len()];
    for (i, node) in raw_nodes.into_iter().enumerate() {
        let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD;
        if !check(unsafe { sys::cuGraphNodeGetType(node, &mut ty) }, "cuGraphNodeGetType") {
            return;
        }
        if ty != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD {
            continue;
        }
        let mut ev: sys::CUevent = std::ptr::null_mut();
        if !check(
            unsafe { sys::cuGraphEventRecordNodeGetEvent(node, &mut ev) },
            "cuGraphEventRecordNodeGetEvent",
        ) {
            return;
        }
        let pos = pend.iter().position(|(_, pe)| *pe == ev);
        if let Some(p) = pos {
            matched[p] = true;
        }
        names.push(pos.map(|p| pend[p].0.clone()).unwrap_or_else(|| format!("probe#{i}")));
        nodes.push(node);
        baked.push(Ev(ev)); // 所有权从图节点接管（SetEvent 换走前图持有同一指针）
    }
    // 没匹配上的 pending（理论上不存在）——句柄归还销毁。
    let _unmatched: Vec<Ev> = pend
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !matched[*i])
        .map(|(_, (_, pe))| Ev(pe))
        .collect();
    // 3. 建第二代事件，写状态。
    let n = nodes.len();
    let mut gen1 = Vec::with_capacity(n);
    for _ in 0..n {
        match Ev::new("cuEventCreate(ping-pong)") {
            Some(e) => gen1.push(e),
            None => return,
        }
    }
    GT.with(|m| {
        let mut m = m.borrow_mut();
        let s = m.entry(key).or_insert_with(KeyState::new);
        s.names = names;
        s.nodes = nodes;
        s.gens = [baked, gen1];
        s.next_gen = 1; // 第一次重放换入 gen1；gen0（烘焙）捕获期没执行过，可直接被读回逻辑跳过
        s.last_used = None;
        s.direct.clear();
    });
}

/// 每次重放前：把新一代事件换进图节点（上一代已被前一次重放写完，可异步读回）。
/// 返回 false = 本键图未就绪/换入失败（调用方照放图，只是没计时）。
pub fn before_replay(key: (u64, u64), g: &cudarc::driver::CudaGraph) -> bool {
    if !enabled() {
        return false;
    }
    // 顺手消费上一代（完成了就计入统计环）。
    maybe_drain(key);
    let (nodes, gen) = match GT.with(|m| {
        let mut m = m.borrow_mut();
        match m.get_mut(&key) {
            Some(s) if !s.nodes.is_empty() => Some((s.nodes.clone(), s.next_gen)),
            _ => None,
        }
    }) {
        Some(v) => v,
        None => return false,
    };
    let evs: Vec<sys::CUevent> = match GT.with(|m| {
        let m = m.borrow();
        m.get(&key).map(|s| s.gens[gen].iter().map(|e| e.0).collect::<Vec<_>>())
    }) {
        Some(e) if e.len() == nodes.len() => e,
        _ => return false,
    };
    for (node, ev) in nodes.iter().zip(evs.iter()) {
        if !check(
            unsafe { sys::cuGraphExecEventRecordNodeSetEvent(g.cu_graph_exec(), *node, *ev) },
            "cuGraphExecEventRecordNodeSetEvent",
        ) {
            return false;
        }
    }
    GT.with(|m| {
        if let Some(s) = m.borrow_mut().get_mut(&key) {
            s.last_used = Some(gen);
            s.next_gen ^= 1;
        }
    });
    true
}

/// 消费上一代/直跑缓冲里已完成的事件对，计入统计环。随时可安全调用。
pub fn maybe_drain(key: (u64, u64)) {
    if !enabled() {
        return;
    }
    GT.with(|m| {
        let mut m = m.borrow_mut();
        let s = match m.get_mut(&key) {
            Some(s) => s,
            None => return,
        };
        if let Some(g) = s.last_used.take() {
            let names = s.names.clone();
            let evs: Vec<sys::CUevent> = s.gens[g].iter().map(|e| e.0).collect();
            if evs.len() >= 2 && query_ready_raw(*evs.last().unwrap()) {
                absorb(s, &names, &evs);
            } else {
                s.last_used = Some(g); // 没完成，留着下次
            }
        }
        if !s.direct.is_empty() {
            let names: Vec<String> = s.direct.iter().map(|(n, _)| n.clone()).collect();
            let evs: Vec<sys::CUevent> = s.direct.iter().map(|(_, e)| e.0).collect();
            if evs.len() >= 2 && query_ready_raw(*evs.last().unwrap()) {
                absorb(s, &names, &evs);
                s.direct.clear();
            }
        }
    });
}

/// 回收某 epoch 的全部计时状态（与 `graph::epoch_purge` 同步调）。
pub fn purge(old: u64) {
    if old == 0 {
        return;
    }
    GT.with(|m| m.borrow_mut().retain(|k, _| k.1 != old));
    PENDING.with(|p| p.borrow_mut().retain(|(k, _, _)| k.1 != old));
    CURRENT.with(|c| c.borrow_mut().retain(|k| k.1 != old));
}

/// 汇总打印（bench 收尾时调；开关关则 no-op）。按 base 键聚合各 epoch，
/// 按总耗时（avg×n 求和）降序，top 40。
pub fn report() {
    if !enabled() {
        return;
    }
    // 先尽力 drain 所有键（最后一次重放的数据也收进来）。
    let keys: Vec<(u64, u64)> = GT.with(|m| m.borrow().keys().copied().collect());
    for k in &keys {
        maybe_drain(*k);
    }
    // 聚合成 (base) -> (spans, segs)。
    let mut agg: HashMap<u64, (Vec<f32>, HashMap<String, Vec<f32>>)> = HashMap::new();
    GT.with(|m| {
        for ((base, _), s) in m.borrow().iter() {
            let e = agg.entry(*base).or_default();
            e.0.extend(s.spans.iter().copied());
            for (name, ring) in &s.segs {
                e.1.entry(name.clone()).or_default().extend(ring.iter().copied());
            }
        }
    });
    if agg.is_empty() {
        eprintln!("[graph_timer] 无数据（开关开了但没有任何探针被录到？）");
        return;
    }
    let mut rows: Vec<(u64, usize, f32, f32, String)> = Vec::new(); // (base, n, avg, max, seg)
    let mut total_span = Vec::new();
    for (base, (spans, segs)) in &agg {
        total_span.extend(spans.iter().copied());
        let mut seg_rows: Vec<(String, Vec<f32>)> =
            segs.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        seg_rows.sort_by(|a, b| b.1.iter().sum::<f32>().total_cmp(&a.1.iter().sum::<f32>()));
        for (name, v) in seg_rows {
            let n = v.len();
            let avg = v.iter().sum::<f32>() / n.max(1) as f32;
            let max = v.iter().copied().fold(0.0f32, f32::max);
            rows.push((*base, n, avg, max, name));
        }
    }
    // base 键按总贡献（avg×n 求和）降序。
    let mut order: Vec<u64> = agg.keys().copied().collect();
    order.sort_by(|a, b| {
        let sa: f32 = rows.iter().filter(|r| r.0 == *a).map(|r| r.2 * r.1 as f32).sum();
        let sb: f32 = rows.iter().filter(|r| r.0 == *b).map(|r| r.2 * r.1 as f32).sum();
        sb.total_cmp(&sa)
    });
    eprintln!("\n[graph_timer] 各捕获单元计时（最近 ≤{RING_CAP} 次，ms；图内探针零开销）");
    for (printed, base) in order.iter().enumerate() {
        if printed >= 40 {
            eprintln!("  …（另有 {} 个键未显示）", order.len() - 40);
            break;
        }
        eprintln!("  key=0x{base:016x}:");
        for r in rows.iter().filter(|r| r.0 == *base) {
            eprintln!("      {:<28} n={:>5}  avg={:>8.3}  max={:>8.3}", r.4, r.1, r.2, r.3);
        }
    }
    if !total_span.is_empty() {
        let avg: f32 = total_span.iter().sum::<f32>() / total_span.len() as f32;
        eprintln!("  ── begin→end 合计：{avg:.3} ms/遍（n={}）──", total_span.len());
    }
}
