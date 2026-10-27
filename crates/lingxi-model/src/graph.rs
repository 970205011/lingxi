//! CUDA Graph 缓存 + **固定地址工作区**。
//!
//! 动机：decode 每层 ~30 个 kernel，其中
//! 低占用率的小 kernel 有 **10~26 µs 的 GPU 固定成本**（[4,10240] 只有 6 GB/s、
//! [320,10240] 251 GB/s，而 [8192,2560] 有 725 GB/s）。整层 144 MB 的聚合带宽只有
//! ~150 GB/s。把一串 kernel 预先录进一张图里一次提交，是唯一能同时消掉
//! 「CPU 逐条入队」和「逐 kernel 起的流水线空隙」的手段，而且**完全不改数值**
//! （录进去的就是同一批 kernel、同一批参数、同一批缓冲）。
//!
//! 两个前提：
//! 1. **真 stream**：`cuStreamBeginCapture` 不允许捕获 legacy NULL stream
//! （`Kernels::load` 已改用 `ctx.new_stream`）。
//! 2. **地址稳定**：图把设备指针烘死，所以被录的代码**不能**用共享 scratch 池
//! （池的 LIFO 轮转会被别处的调用打乱）。这里给每个捕获单元一份**私有固定缓冲**
//! （`take_fixed` / `give_fixed`，按 `(key, slot)` 键控）。
//!
//! 三段式：第 1 次调用→建缓冲并正常执行；第 2 次→捕获；第 3 次起→重放。
//! `LINGXI_GRAPHS=1` 打开；失败会回落到普通路径（不做静默兜底：出错直接返回错误）。

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::Result;
use cudarc::driver::{CudaGraph, CudaSlice};
use half::bf16;
use lingxi_kernels::Kernels;

thread_local! {
    /// `(base key, epoch) -> graph`。v1.5 起键升为二元组：base 键自带分桶语义
    /// （层号 / GR 的 `(1+rows)<<40` / MoE gkey），epoch 是会话维——二元组杜绝一切
    /// 位打包撞键（旧 `base + epoch<<40` 与 GR rows 段重叠，跨 epoch 理论撞键）。
    static CACHE: RefCell<HashMap<(u64, u64), CudaGraph>> = RefCell::new(HashMap::new());
    /// `(base key, epoch) -> 已走过几个阶段`（0 建缓冲 / 1 待捕获 / 2 已捕获可重放）
    static PHASE: RefCell<HashMap<(u64, u64), u8>> = RefCell::new(HashMap::new());
    /// 固定地址工作区：`(设备, key, slot) -> 设备缓冲`（**带设备**，见 `dev` 的说明）
    static FIXED: RefCell<HashMap<(usize, u64, usize), CudaSlice<bf16>>> =
        RefCell::new(HashMap::new());
}

/// 默认**开启**（本轮已把三个真 bug 修完，48/192 token 输出与 baseline 逐位一致，实测 −4%）；
/// `LINGXI_NO_GRAPHS=1` 可关。开关名从 `LINGXI_GRAPHS` 改成反向，避免"忘了设就没收益"。
pub fn enabled() -> bool {
    !FORCE_OFF.load(std::sync::atomic::Ordering::Relaxed)
        && SCOPE_OFF.load(std::sync::atomic::Ordering::Relaxed) == 0
        && !crate::flag("LINGXI_NO_GRAPHS")
}

/// **作用域禁用**计数（路线 A two-pass prefill 用）：同一线程先后在两卡上跑同一层
/// （pass1 卡 A 录了 key=(层,t) 的图，pass2 卡 B 同 key 会命中 PHASE=2 直接**重放卡 A 的图**
/// ——图缓存键不含设备）。two-pass 的两个 pass 全程持 guard ⇒ 一个键都不会录/重放，
/// decode 期 guard 已 drop ⇒ 行为与现状逐字一致。
static SCOPE_OFF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// 取得一个**作用域图禁用**守卫（Drop 时恢复）。可嵌套。
pub fn scoped_disable() -> GraphScopeGuard {
    SCOPE_OFF.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GraphScopeGuard
}

pub struct GraphScopeGuard;
impl Drop for GraphScopeGuard {
    fn drop(&mut self) {
        SCOPE_OFF.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// ★ P15-㉜（2026-10-17）：双卡自动 layer-split 时**内部强制关图**（跨卡内核不能被
/// GPU0 的 CUDA Graph 捕获）——不必要求用户手设 `LINGXI_NO_GRAPHS=1`。
/// 只影响本进程，幂等。
pub fn force_disable() {
    FORCE_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
}
static FORCE_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 「只用私有缓冲、**不录图**」的保留 key。
///
/// ⚠️ **绝不能再用 0 表示这个语义**：层的 key 是 `layer*8`，layer 0 恰好就是 0，
/// 而主干末端那个 `hyper_connection_mixer` 每 token 也走一次 GR。两者共用 key 0
/// 会让同一 `(key, slot)` 在 `dim=10240` 与 `hidden=2560` 之间来回翻——`take_fixed`
/// 每次长度对不上就**重新分配**，地址逐 token 漂移；图里烘死的指针随即指向错的地方
/// （实测：token 2 之后立刻发散，且 layer 0 的 `mixed_a` 地址以 2 为周期摆动）。
pub const NO_CAPTURE: u64 = u64::MAX / 2;

thread_local! {
    /// 捕获期间置真：此时**禁止任何分配**（`cuMemAllocAsync` 在捕获里不受支持/会被驱动
    /// 报成含义不明的 `CUDA_ERROR_INVALID_VALUE`）。缺缓冲就直接报错、把 key/slot 说清楚。
    static CAPTURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 取一块**专属于 `(key, slot)`** 的 bf16 缓冲（取走后由调用方持有，用完 `give_fixed` 归还）。
///
/// ⚠️ **长度校验用 `>=` 而不是 `==`**（2026-10-16 修）：投机解码里每轮的 `t` 都在变，
/// 而 `layer.rs` 的 key 是 `layer*8`（**不含 t**）⇒ 用 `==` 时每轮每层 4 块长度全对不上，
/// 于是 `cudaMalloc`+`cudaFree` 各 192 次/轮（`cudaFree` 走 legacy 分配器会**隐式同步整条流**）。
/// 实测 `LINGXI_PTRPROBE=1`：一个 24-token 的投机跑里 `[fixed]` 重新分配被打出 **3315 次**。
/// 改成 `>=` 后池里留的是"曾经用过的最大块"，地址从此稳定（对图也更友好），
/// 多占的显存上界 = `t_max × hidden × 2 B × 192` ≈ 9 MB。
/// **池键必须带设备**（`k.ctx.cu_ctx`）。
///
/// 为什么（2026-10-16 顺手修，为 P15 的 layer-split 扫雷）：第一版只按 `(key, slot)` 键控，
/// 而 `layer.rs` 的 key 是 `layer*8` —— **layer-split 之后同一张卡上跑的是不同的层**，
/// 于是 GPU1 上的层会**拿到 GPU0 之前放回池里的那块显存** ⇒ 又是
/// `CUDA_ERROR_ILLEGAL_ADDRESS` / 静默错（与 `scratch` 池、`MappedHost` 是**同一类坑**，
/// 那两个已经在跨卡上各咬过一口）。
#[inline]
fn dev(k: &Kernels) -> usize {
    k.ctx.cu_ctx() as usize
}

pub fn take_fixed(k: &Kernels, key: u64, slot: usize, len: usize) -> Result<CudaSlice<bf16>> {
    let kk = (dev(k), key, slot);
    let got = FIXED.with(|m| m.borrow_mut().remove(&kk));
    match got {
        Some(s) if s.len() >= len => {
            if crate::flag("LINGXI_PTRPROBE") {
                eprintln!("      [fixed] key={key:?} slot={slot} len={len} → 复用（现存 {}）", s.len());
            }
            Ok(s)
        }
        other => {
            if CAPTURING.with(|c| c.get()) {
                let had = other.map(|s| s.len());
                anyhow::bail!(
                    "捕获期间缺少固定缓冲 key={key:?} slot={slot} len={len}（现存长度={had:?}）——\
                     说明第 1 遍没有把它建出来"
                );
            }
            if crate::flag("LINGXI_PTRPROBE") {
                eprintln!(
                    "      [fixed] key={key:?} slot={slot} len={len} → 重新分配（旧长度={:?}）",
                    other.as_ref().map(|s| s.len())
                );
            }
            Ok(unsafe { k.stream.alloc::<bf16>(len) }?)
        }
    }
}

pub fn give_fixed(k: &Kernels, key: u64, slot: usize, s: CudaSlice<bf16>) {
    let kk = (dev(k), key, slot);
    FIXED.with(|m| {
        m.borrow_mut().insert(kk, s);
    });
}

thread_local! {
    /// `f32` 版固定区（GR 融合核的 `rmsg` 等小量）。**同样按设备分键**（理由见 `dev`）。
    static FIXED_F32: RefCell<HashMap<(usize, u64, usize), CudaSlice<f32>>> =
        RefCell::new(HashMap::new());
}

pub fn take_fixed_f32(k: &Kernels, key: u64, slot: usize, len: usize) -> Result<CudaSlice<f32>> {
    let kk = (dev(k), key, slot);
    let got = FIXED_F32.with(|m| m.borrow_mut().remove(&kk));
    match got {
        Some(s) if s.len() >= len => Ok(s),
        other => {
            if CAPTURING.with(|c| c.get()) {
                anyhow::bail!("捕获期间缺少 f32 固定缓冲 key={key:?} slot={slot} len={len}（现存={:?}）",
                    other.map(|s| s.len()));
            }
            if crate::flag("LINGXI_PTRPROBE") {
                eprintln!("      [fixed-f32] key={key:?} slot={slot} len={len} → 重新分配（旧长度={:?}）",
                    other.as_ref().map(|s| s.len()));
            }
            Ok(unsafe { k.stream.alloc::<f32>(len) }?)
        }
    }
}

pub fn give_fixed_f32(k: &Kernels, key: u64, slot: usize, s: CudaSlice<f32>) {
    let kk = (dev(k), key, slot);
    FIXED_F32.with(|m| {
        m.borrow_mut().insert(kk, s);
    });
}

thread_local! {
    /// `i32` 版固定区（MoE rows==1 图的 `slot_ids` stage 用）。**按设备分键**（理由见 `dev`）。
    static FIXED_I32: RefCell<HashMap<(usize, u64, usize), CudaSlice<i32>>> =
        RefCell::new(HashMap::new());
}

pub fn take_fixed_i32(k: &Kernels, key: u64, slot: usize, len: usize) -> Result<CudaSlice<i32>> {
    let kk = (dev(k), key, slot);
    let got = FIXED_I32.with(|m| m.borrow_mut().remove(&kk));
    match got {
        Some(s) if s.len() >= len => Ok(s),
        other => {
            if CAPTURING.with(|c| c.get()) {
                anyhow::bail!("捕获期间缺少 i32 固定缓冲 key={key:?} slot={slot} len={len}（现存={:?}）",
                    other.map(|s| s.len()));
            }
            if crate::flag("LINGXI_PTRPROBE") {
                eprintln!("      [fixed-i32] key={key:?} slot={slot} len={len} → 重新分配（旧长度={:?}）",
                    other.as_ref().map(|s| s.len()));
            }
            Ok(unsafe { k.stream.alloc::<i32>(len) }?)
        }
    }
}

pub fn give_fixed_i32(k: &Kernels, key: u64, slot: usize, s: CudaSlice<i32>) {
    let kk = (dev(k), key, slot);
    FIXED_I32.with(|m| {
        m.borrow_mut().insert(kk, s);
    });
}

thread_local! {
    /// ㊿-P1 会话 epoch：CUDA Graph 把设备指针（工作区/固定槽）烘死在图体里，**只属于录制时
    /// 的那个 Sequence**。serve 在物化另一个会话前 bump 一次，旧图自然作废并按新 epoch
    /// 重新录制（下一 token 每核一遍，摊销可忽略）；不 bump 会跨会话重放旧地址（必错）。
    ///
    /// v1.5（2026-10-20，多物化）：epoch 升级为 **per-session 键**——每个物化会话持有
    /// 自己的 epoch 值，切换只是 `epoch_switch`（图集各自保留、交替不重录）；只有**新建/
    /// 唤醒**（设备缓冲重建）才领新 epoch。CACHE/PHASE 键是 `(base, epoch)` 二元组，
    /// 与 base 键的位打包（层号 / GR rows 段）彻底分离。CACHE 按存活会话数有界；
    /// 会话沉没时 `epoch_purge` 回收其图。
    static EPOCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// 全局 epoch 分配器（per-process，单调 +1；0 保留给"无会话"默认态）。
static NEXT_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// 领一个新 epoch 并切换过去：**新建/唤醒**会话时用（设备缓冲是新地址，旧图必须作废）。
pub fn epoch_fresh() -> u64 {
    let e = NEXT_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    EPOCH.with(|c| c.set(e));
    e
}

/// 切换到某个已存活会话的 epoch（多物化交替：图集各自保留，零重录）。
pub fn epoch_switch(e: u64) {
    EPOCH.with(|c| c.set(e));
}

/// 当前 epoch（诊断/记账用）。
pub fn epoch_get() -> u64 {
    EPOCH.with(|c| c.get())
}

/// 回收指定 epoch 的图与相位（会话沉没/丢弃时调用，防 CACHE 无界增长）。
pub fn epoch_purge(old: u64) {
    if old == 0 {
        return;
    }
    CACHE.with(|c| c.borrow_mut().retain(|k, _| k.1 != old));
    PHASE.with(|p| p.borrow_mut().retain(|k, _| k.1 != old));
    crate::graph_timer::purge(old);
}

/// 兼容旧调用点（qsa.rs 恢复设备检查点后）：语义 = 领新 epoch。
pub fn epoch_bump() {
    epoch_fresh();
}

/// 捕获 / 重放。`body` 必须是**同一条 stream 上、无同步、无 host 回读、无共享池分配**的一段 kernel 序列。
///
/// 健壮性（2026-10-19/20 海森堡 bug 处置，见 PLAN_NEXT 事故记录）：捕获/重放对驱动时序敏感，
/// 间歇 `CUDA_ERROR_INVALID_VALUE` 不再向上炸会话——失败的键**永久降级为直跑**（phase=3，
/// 只损性能不损正确性），并打响亮日志保留根因线索。body 为 FnMut：捕获已结束但
/// upload/首放失败时**重跑一遍直算**补回输出（捕获期只录不跑，不重跑会得到脏值）。
/// 捕获期探针（graph_timer）查询用。
pub(crate) fn is_capturing() -> bool {
    CAPTURING.with(|c| c.get())
}

pub fn run_cached<F>(k: &Kernels, key: u64, body: F) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    // P2 分页：QSA attention_step 体内有 host 同步/D2H/H2D（选块准入、下沉、预取、
    // 刷块表）——**与 CUDA Graph 捕获互斥**（实测 end_capture 炸 STREAM_CAPTURE_UNJOINED、
    // 且可把流留在捕获态诱发后续 EINVAL）。分页模式下本函数一概直跑（只损图收益）。
    if crate::flag("LINGXI_QSA_PAGED") {
        // K2 探针：直跑路径同样计时（普通流事件，非图节点）。
        let mut body = body;
        if crate::graph_timer::enabled() {
            let key2 = (key, EPOCH.with(|e| e.get()));
            crate::graph_timer::enter(key2);
            crate::graph_timer::probe(k, "begin");
            let r = body();
            crate::graph_timer::probe(k, "end");
            crate::graph_timer::leave();
            return r;
        }
        return body();
    }
    // K2 graph_timer：开关打开时登记键归属 + body 首尾自动探针（图路径下成为图的
    // 首/尾 event-record 节点；直跑路径是普通流事件）。关闭时零开销。
    let gt = crate::graph_timer::enabled();
    let key2 = (key, EPOCH.with(|e| e.get()));
    if gt {
        crate::graph_timer::enter(key2);
    }
    let mut body = body;
    let r = run_cached_inner(k, key2, &mut body, gt);
    if gt {
        crate::graph_timer::leave();
    }
    r
}

fn run_cached_inner<F>(k: &Kernels, key: (u64, u64), body: &mut F, gt: bool) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    use cudarc::driver::sys;
    // 键已由外层 run_cached 升级为 `(base, epoch)` 二元组（K2 graph_timer 一并使用）。
    let set_phase = |v: u8| PHASE.with(|p| { p.borrow_mut().insert(key, v); });
    let phase = PHASE.with(|p| {
        let mut m = p.borrow_mut();
        let e = m.entry(key).or_insert(0u8);
        let v = *e;
        if v < 2 {
            *e = v + 1;
        }
        v
    });
    // K2：body 首尾自动探针（仅计时开关打开时非空操作）。
    let timed = |body: &mut F| -> Result<()> {
        if gt {
            crate::graph_timer::probe(k, "begin");
        }
        let r = body();
        if gt {
            crate::graph_timer::probe(k, "end");
        }
        r
    };
    match phase {
        // 第 1 遍：正常执行（同时把私有缓冲建出来 —— 捕获期间不再做任何分配）
        0 => timed(body),
        // 第 2 遍：录图
        1 => {
            // 先静默：`host_cache` 的预读线程可能还在同一条 stream 上收尾，
            // 带着跨线程的在途工作开始捕获会直接 INVALID_VALUE。
            k.stream.synchronize()?;
            // ⚠️ **必须用 RELAXED，不能用 THREAD_LOCAL**：`host_cache` 的预读线程也会
            // 在**同一条 stream** 上做权重上传（`k.stream.clone_htod`），
            // THREAD_LOCAL 校验会因此直接 `CUDA_ERROR_INVALID_VALUE`。
            // RELAXED 只放弃跨线程的依赖校验，对我们（上传都发生在 decode 之前）语义不变。
            let began = k.stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .is_ok()
                || {
                    // 在途跨线程工作竞态：同步后重试一次（实测可消掉间歇 EINVAL）。
                    let _ = k.stream.synchronize();
                    k.stream
                        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                        .is_ok()
                };
            if !began {
                // 重试仍失败：本键永久降级直跑（性能换确定性），留日志追根因。
                eprintln!("⚠️ [graph] key={key:?} begin_capture 重试仍失败 —— 该键永久降级为直跑");
                set_phase(3);
                return body();
            }
            CAPTURING.with(|c| c.set(true));
            // 捕获内探针必须带 EXTERNAL flag（graph_timer 内部按 is_capturing 处理），
            // 录进图成为首/尾 event-record 节点。
            if gt {
                crate::graph_timer::probe(k, "begin");
            }
            let r = body();
            if gt {
                crate::graph_timer::probe(k, "end");
            }
            CAPTURING.with(|c| c.set(false));
            let cap = k.stream.end_capture(sys::CUgraphInstantiate_flags(0));
            if crate::flag("LINGXI_GRAPHS_VERBOSE") {
                eprintln!("    [graph] key={key:?} body={:?} end_capture_ok={}", r.as_ref().err().map(|e| e.to_string()), cap.is_ok());
            }
            match (r, cap) {
                (Ok(()), Ok(Some(g))) => {
                    // ⚠️ **捕获期只“录”不“跑”**：`cuStreamBeginCapture` 期间调用的 kernel
                    // 只会被记进图里，**不会真的执行**。所以这一遍必须把刚录好的图**立刻放一次**，
                    // 否则这一 token 的 GR 输出全是上一轮的脏值 —— 首 token 之后立刻发散成乱码
                    // （实测：token 2 从 561 变成 11，之后路由乱跳、DMA 风暴，速度掉到 0.2 s/token）。
                    // upload 或首放失败的图必须**丢弃**：实测 upload 失败后 launch 会在
                    // libcuda 里 segfault（NULL+0xe8，dmesg 实锤）——丢图重跑直算补输出。
                    if g.upload().is_err() {
                        eprintln!("⚠️ [graph] key={key:?} upload 失败 —— 丢弃该图，本键降级直跑");
                        set_phase(3);
                        return timed(body);
                    }
                    if g.launch().is_err() {
                        eprintln!("⚠️ [graph] key={key:?} 首放失败 —— 丢弃该图，本键降级直跑");
                        set_phase(3);
                        return timed(body);
                    }
                    // K2：图已成——枚举 event-record 节点、建 ping-pong 事件（仅计时开时有效）。
                    if gt {
                        crate::graph_timer::on_captured(key, &g);
                    }
                    CACHE.with(|c| {
                        c.borrow_mut().insert(key, g);
                    });
                    Ok(())
                }
                (Err(e), _) => { set_phase(3); Err(e) }
                (_, Ok(None)) => {
                    eprintln!("⚠️ [graph] key={key:?} end_capture 未产出图 —— 本键降级直跑");
                    set_phase(3);
                    timed(body)
                }
                (_, Err(e)) => {
                    eprintln!("⚠️ [graph] key={key:?} end_capture 失败（{e:?}）—— 本键降级直跑");
                    set_phase(3);
                    timed(body)
                }
            }
        }
        // 永久降级：直跑（捕获竞争/重放失败的键落这里）。
        3 => timed(body),
        // 第 3 遍起：重放；失败/缺失 → 永久降级直跑（body 未被消费，可回退）。
        _ => {
            let launched = CACHE.with(|c| match c.borrow().get(&key) {
                Some(g) => {
                    // K2：重放前把新一代探针事件换进图节点（上一代留异步读回）。
                    if gt {
                        crate::graph_timer::before_replay(key, g);
                    }
                    g.launch().is_ok()
                }
                None => false,
            });
            if launched {
                Ok(())
            } else {
                eprintln!("⚠️ [graph] key={key:?} 重放失败或图缺失 —— 该键永久降级为直跑");
                set_phase(3);
                timed(body)
            }
        }
    }
}

/// 供日志：已录了多少张图。
pub fn graph_count() -> usize {
    CACHE.with(|c| c.borrow().len())
}
