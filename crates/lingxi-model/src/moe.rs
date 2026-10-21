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
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

use crate::gpu::{proj_t1, proj_t1_epi};
use crate::scratch;

/// 专家 GEMV 的**每 warp 行数**（rows-per-warp）。
///
/// 默认 1 = 逐行版，与历史逐位一致（`gemv_q4g64_slots_rows_bf16(.., 1)` 就是它）。
/// `LINGXI_Q4_SLOTS_RPW=2/4/8` 切到多行版：`dn` 段 `k=640`（每行只有 320 B 的 Q4），
/// 逐行版会把全局 warp 数撑到 `n×S = 25600` 个（每 warp 只搬 320 B），线程太多、每行太少；
/// 多行版让一个 warp 顺序扫多行，减少线程数并把跨行访存并行起来。
fn slots_rpw() -> usize {
    static RPW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *RPW.get_or_init(|| {
        std::env::var("LINGXI_Q4_SLOTS_RPW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    })
}

/// **批量路径（`slotsm`）的每 warp 行数**。**默认 2**（实测甜点，见下）。
///
/// `LINGXI_Q4_SLOTSM_RPW=1` 退回逐行（与历史逐位一致）；`=4` 实测更差。
///
/// 交错 A/B（各 2 轮，n=200，kd=6/min_p=0.7/钉N=145）：
/// | RPW | tok/s | 主干验证 ms/轮 |
/// | --- | --- | --- |
/// | 1 | 44.0 / 46.1 | 66.8 / 63.8 |
/// | **2** | **48.2 / 48.5** | **60.7 / 60.3** |
/// | 4 | 46.5 / 45.3 | 63.9 / 65.7 |
/// ⇒ 一格从"1 个输出行"变成"顺序扫 2 行"，用更少的 warp 换更多跨行访存并行 ——
/// 与 t=1 路径的 `slots_rpw` 同一手法、同一理由（`dn` 段 k=640 每 warp 仅 ~2.5 次迭代，
/// HBM ~600 cyc 延迟盖不住）。
/// **位安全**：每行的列集合与累加次序不变（见 `dequant_ops.cu` 的 `slotsm_body` 注释）。
fn slotsm_rpw() -> usize {
    static RPW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *RPW.get_or_init(|| {
        std::env::var("LINGXI_Q4_SLOTSM_RPW")
            .ok()
            .and_then(|v| v.parse().ok())
            // ★ P15-⑦（2026-10-17）：默认 2 → 1。ptxas 实测寄存器：rpw=2 用 104 个
            // （⇒ 每 SM 只 2 块 = 16 warps），rpw=1 用 64 个（⇒ 4 块 = 32 warps）——
            // 这就是 slots(48 regs/40 warps, 398 GB/s) vs slotsm(302 GB/s) 差距的主因
            // （占用率 = 延迟隐藏能力）。bw-ceiling 真实权重实测 rpw=1 与 rpw=2 在所有
            // m 下打平（m=1: 303/302、m=2: 297/297、m=7: 264/264）——
            // 旧扫描里"rpw=2 快 10%"是全零权重时代的噪声/假象。
            .unwrap_or(1)
    })
}

/// 稀疏专家**按需取权重**源（P1 瘦身）：路由算完后，只把命中专家的字节搬到显存。
///
/// model 不做 IO——本 trait 由装配层（bench/server）实现：mmap 切片、页缓存、
/// 显存槽缓存等策略全在实现侧，model 只在路由结果出来后调用一次。
/// 数学约束：返回的每个专家字节必须与全量张量中对应专家段**逐位相同**，
/// 槽位顺序见 [`ExpertSource::load_hits`]。
pub trait ExpertSource: Sync {
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

    /// ㊺ A2 设备路由：设备侧 Q4 直读视图（槽区 + 每层槽位表 + 段几何常量）。
    /// `Some` = 支持设备端 top-k 直查（免 host 同步决策）；`None` = 回退 host 路径。
    fn q4_dev(&self) -> Option<Q4DevView<'_>> {
        None
    }

    /// ㊺ A2：把 `experts` 里**尚未入槽**的搬进设备槽（已在槽里的是空操作），并保证
    /// `protect`（本批已在用的槽）以及本批新分配的槽不会被 `victim()` 选中。
    /// 实现方**必须**让槽位表的改动对设备可见（含被淘汰的**其它层**的槽位行）。
    fn load_missing(&self, _layer: u32, _experts: &[u32], _protect: &[usize]) -> Result<()> {
        Ok(())
    }

    /// ㊺ A2：把 `layer` 这一层的槽位表（E 项）同步上卡。**只在 legacy（miss 回退）
    /// load_hits 之后调用**——设备路由快路径全中时 load_hits 根本不被调用，表无需改动。
    fn slot_map_sync_layer(&self, _layer: u32) -> Result<()> {
        Ok(())
    }
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
pub struct Q4DevView<'a> {
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

/// A2 路由核的三块**映射 host 内存**（thread_local 按上下文隔离，同 `MAP` 的坑与解法）：
/// 未命中计数、未命中专家 id（紧凑）、每个选中专家的槽号（-1=miss）。
struct RouteMapped {
    flag_dev: u64,
    flag_host: *mut u32,
    miss_ids_dev: u64,
    miss_ids_host: *mut i32,
    slots_dev: u64,
    slots_host: *mut i32,
}

fn route_miss_mapped(k: &Kernels, top_k: usize) -> Result<RouteMapped> {
    type Slot = (
        usize,
        crate::scratch::MappedHost<u32>,
        crate::scratch::MappedHost<i32>,
        crate::scratch::MappedHost<i32>,
    );
    thread_local! {
        static MM: std::cell::RefCell<Option<Slot>> = const { std::cell::RefCell::new(None) };
    }
    MM.with(|c| -> Result<RouteMapped> {
        let mut g = c.borrow_mut();
        let ctx_id = k.ctx.cu_ctx() as usize;
        let reuse = g.as_ref().map_or(false, |m| {
            m.0 == ctx_id && m.1.len() >= 1 && m.2.len() >= top_k && m.3.len() >= top_k
        });
        if !reuse {
            *g = Some((
                ctx_id,
                unsafe { crate::scratch::MappedHost::<u32>::new(k, 1)? },
                unsafe { crate::scratch::MappedHost::<i32>::new(k, top_k.max(1))? },
                unsafe { crate::scratch::MappedHost::<i32>::new(k, top_k.max(1))? },
            ));
        }
        let m = g.as_mut().unwrap();
        let flag_host = m.1.slice_mut()?.as_mut_ptr();
        let miss_ids_host = m.2.slice_mut()?.as_mut_ptr();
        let slots_host = m.3.slice_mut()?.as_mut_ptr();
        Ok(RouteMapped {
            flag_dev: m.1.dev,
            flag_host,
            miss_ids_dev: m.2.dev,
            miss_ids_host,
            slots_dev: m.3.dev,
            slots_host,
        })
    })
}

/// 记录事件并**自旋**等待（`cuEventQuery` 忙等，不线程让出——比阻塞式
/// `stream.synchronize()` 省 ~5-15µs/次的唤醒延迟 ×48 层）。事件按上下文缓存。
fn route_spin(k: &Kernels, timed: bool) -> Result<Option<f32>> {
    use cudarc::driver::{result, sys};
    thread_local! {
        static EV: std::cell::Cell<Option<(usize, sys::CUevent)>> =
            const { std::cell::Cell::new(None) };
    }
    let ctx_id = k.ctx.cu_ctx() as usize;
    let ev = EV.with(|c| {
        let cur = c.get();
        match cur {
            Some((id, ev)) if id == ctx_id => Ok::<sys::CUevent, anyhow::Error>(ev),
            _ => {
                let ev = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)?;
                c.set(Some((ctx_id, ev)));
                Ok(ev)
            }
        }
    })?;
    unsafe { result::event::record(ev, k.stream.cu_stream())? };
    if crate::flag("LINGXI_ROUTE_DEV_SYNC") {
        // 诊断/对照：用整条流同步替代自旋（验证"自旋本身是否拖慢"）。
        if timed {
            if let Some((_, e0)) = ROUTE_EV0.with(|c| c.get()) {
                let mut ms = 0f32;
                unsafe { sys::cuEventElapsedTime(&mut ms, e0, ev) };
                k.stream.synchronize()?;
                return Ok(Some(ms));
            }
        }
        k.stream.synchronize()?;
        return Ok(None);
    }
    loop {
        let r = unsafe { sys::cuEventQuery(ev) };
        if r == sys::CUresult::CUDA_SUCCESS {
            break;
        }
        if r != sys::CUresult::CUDA_ERROR_NOT_READY {
            anyhow::bail!("route_spin: cuEventQuery -> {r:?}");
        }
        std::hint::spin_loop();
    }
    if timed {
        if let Some((_, e0)) = ROUTE_EV0.with(|c| c.get()) {
            let mut ms = 0f32;
            unsafe { sys::cuEventElapsedTime(&mut ms, e0, ev) };
            return Ok(Some(ms));
        }
    }
    Ok(None)
}

thread_local! {
    /// `route_mark` 打的前置事件（诊断计时；按上下文隔离）。
    static ROUTE_EV0: std::cell::Cell<Option<(usize, cudarc::driver::sys::CUevent)>> =
        const { std::cell::Cell::new(None) };
}

/// 诊断用：在路由核 launch **之前**打一个事件（与 [`route_spin`] 的结束事件配对）。
fn route_mark(k: &Kernels) -> Result<()> {
    use cudarc::driver::{result, sys};
    let ctx_id = k.ctx.cu_ctx() as usize;
    let ev = ROUTE_EV0.with(|c| -> Result<sys::CUevent> {
        match c.get() {
            Some((id, ev)) if id == ctx_id => Ok::<sys::CUevent, anyhow::Error>(ev),
            _ => {
                let ev = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)?;
                c.set(Some((ctx_id, ev)));
                Ok(ev)
            }
        }
    })?;
    unsafe { result::event::record(ev, k.stream.cu_stream())? };
    Ok(())
}

/// ㊺ A2 设备路由快路径（rows==1 专用）。命中全中 → `Some(out)`；任一选中专家未入槽
/// → `None`（调用方落回 legacy；本函数写过的设备缓冲全部无副作用）。
///
/// 与 legacy hit 路径的**逐位等价性**靠三点：
/// ① 路由核逐位复刻 host softmax/插入式 top-k/norm（见 topk_ops.cu 头注）；
/// ② 专家段 3 核与 legacy 同键（`gkey=layer*8+7`）同体（同核同参同序），直接复用图缓存；
/// ③ 输出/权重/槽号全部按专家升序，与 host `sel.sort_by_key` 后的组序一致。
#[allow(clippy::too_many_arguments)]
fn route_dev_attempt(
    k: &Kernels,
    layer: u32,
    w: &MoeWeights,
    x: &CudaSlice<bf16>,
    shared: &CudaSlice<bf16>,
    src: &dyn ExpertSource,
    num_experts: usize,
    top_k: usize,
    inter: usize,
    hidden: usize,
) -> Result<Option<CudaSlice<bf16>>> {
    // `LINGXI_NO_PAIR_MOE=1` 时 legacy 走 bf16 紧凑权重路径（不吃 q4）：A2 必须同步关掉，
    // 否则图体（同 gkey）会被两种不同的核序列争用同一个图缓存。
    if crate::flag("LINGXI_NO_PAIR_MOE") {
        return Ok(None);
    }
    let dv = match src.q4_dev() {
        Some(v) => v,
        None => return Ok(None),
    };
    let tw0 = std::time::Instant::now();
    let qc = dv.consts;
    let s = top_k;
    let gkey = (layer as u64) * 8 + 7;
    let rkey = (layer as u64) * 8 + 6;
    // 路由核输出固定槽：logits(rkey,0) / ids(rkey,1) / wts(rkey,2) / rows(rkey,4) /
    // slot_ids 直写图体输入槽 (gkey,5)（省一次 D2D staging）。
    //
    // ⚠️ router 投影必须用 **legacy rows==1 同一条核**（`gemv_bf16_ptr`，不是 `proj_t1`
    //    的 row8 分派）：两核累加次序不同，换核就不再逐位等价。
    let mut logits = crate::graph::take_fixed(k, rkey, 0, num_experts)?;
    let router_w = w.router_shared_gate.slice(0..num_experts * hidden);
    {
        let (wp, _g1) = router_w.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        let (yp, _g3) = logits.device_ptr_mut(&k.stream);
        k.gemv_bf16_ptr(wp, xp, yp, num_experts, hidden)?;
    }
    let mut ids = crate::graph::take_fixed_i32(k, rkey, 1, s)?;
    let mut wts = crate::graph::take_fixed_f32(k, rkey, 2, s)?;
    let mut slots = crate::graph::take_fixed_i32(k, gkey, 5, s)?;
    let mut rowsb = crate::graph::take_fixed_i32(k, rkey, 4, s)?;
    let m = route_miss_mapped(k, s)?;
    // 槽位表按层偏移取裸指针（本层 E 项）；guard 活到 launch 之后。
    let map_base = layer as usize * num_experts;
    let (map_ptr, _map_g) = dv.slot_map.device_ptr(&k.stream);
    let map_dev = map_ptr + (map_base * 4) as u64;
    // 最多两趟：① 全中 → 直接算；② 有未命中 → 补槽 + 刷表（含被挤掉的**别的层**行）
    // → 重跑路由核（logits 未变 ⇒ 选择/权重逐位相同，只是槽号变新）。
    let mut nmiss = 0u32;
    let trace = crate::flag("LINGXI_ROUTE_DEV_TRACE");
    let mut dev_ms = 0f32;
    for attempt in 0..2 {
        unsafe { std::ptr::write_volatile(m.flag_host, 0u32) };
        if trace {
            route_mark(k)?;
        }
        k.route_topk_slots_bf16(
            &logits, map_dev, &mut ids, &mut wts, &mut slots, &mut rowsb, m.slots_dev,
            m.flag_dev, m.miss_ids_dev, num_experts, s,
        )?;
        if let Some(ms) = route_spin(k, trace)? {
            dev_ms += ms;
        }
        nmiss = unsafe { std::ptr::read_volatile(m.flag_host) };
        if nmiss == 0 || attempt == 1 {
            break;
        }
        // 映射内存直读（内核已完成 ⇒ host 可见）：未命中专家（升序）+ 本批已占用槽。
        let mut miss_list: Vec<u32> = Vec::with_capacity(nmiss as usize);
        let mut prot: Vec<usize> = Vec::with_capacity(s);
        for wi in 0..s {
            let sl = unsafe { std::ptr::read_volatile(m.slots_host.add(wi)) };
            if sl >= 0 {
                prot.push(sl as usize);
            }
        }
        for j in 0..nmiss as usize {
            miss_list.push(unsafe { std::ptr::read_volatile(m.miss_ids_host.add(j)) } as u32);
        }
        src.load_missing(layer, &miss_list, &prot)?;
    }
    // `LINGXI_ROUTE_DEV_CHECK=1`：逐位核对路由核 vs host 参考（会同步，仅诊断用）。
    if crate::flag("LINGXI_ROUTE_DEV_CHECK") && nmiss == 0 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 2 {
            let lh: Vec<f32> = k.stream.clone_dtoh(&logits)?.iter().map(|v| v.to_f32()).collect();
            let ih: Vec<i32> = k.stream.clone_dtoh(&ids)?.iter().take(s).copied().collect();
            let wh: Vec<f32> = k.stream.clone_dtoh(&wts)?.iter().take(s).copied().collect();
            // host 参考（与 legacy 同代码）
            let mh = lh.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f32> = lh.iter().map(|v| (v - mh).exp()).collect();
            let sum: f32 = p.iter().sum();
            for v in p.iter_mut() {
                *v /= sum;
            }
            let mut sel: Vec<(usize, f32)> = Vec::new();
            for (i, &pi) in p.iter().enumerate() {
                if sel.len() == s && !(pi > sel[sel.len() - 1].1) {
                    continue;
                }
                let pos = sel.iter().position(|&(_, v)| pi > v).unwrap_or(sel.len());
                sel.insert(pos, (i, pi));
                if sel.len() > s {
                    sel.pop();
                }
            }
            let norm: f32 = sel.iter().map(|(_, v)| *v).sum();
            sel.sort_by_key(|(i, _)| *i);
            let mut bad = 0;
            for (j, &(e, pv)) in sel.iter().enumerate() {
                if ih[j] != e as i32 || wh[j].to_bits() != (pv / norm).to_bits() {
                    if bad < 4 {
                        eprintln!(
                            "[route-check] 层{layer} j={j} 专家 dev={} host={} w dev={:.9e}({:08x}) host={:.9e}({:08x})",
                            ih[j], e, wh[j], wh[j].to_bits(), pv / norm, (pv / norm).to_bits()
                        );
                    }
                    bad += 1;
                }
            }
            eprintln!(
                "[route-check] 层{layer} s={s} 不一致 {bad}/{s}；logits[0..3]={:?}",
                &lh[..3]
            );
            let dev_m = f32::from_bits(unsafe { std::ptr::read_volatile(m.miss_ids_host.add(0)) } as u32);
            let dev_sum = f32::from_bits(unsafe { std::ptr::read_volatile(m.miss_ids_host.add(1)) } as u32);
            let dev_norm = f32::from_bits(unsafe { std::ptr::read_volatile(m.miss_ids_host.add(2)) } as u32);
            eprintln!(
                "[route-check] 层{layer} m dev={dev_m}({:08x}) host={mh}({:08x}) ; sum dev={dev_sum}({:08x}) host={sum}({:08x}) ; norm dev={dev_norm}({:08x}) host={norm}({:08x})",
                dev_m.to_bits(), mh.to_bits(), dev_sum.to_bits(), sum.to_bits(), dev_norm.to_bits(), norm.to_bits()
            );
        }
    }
    crate::graph::give_fixed(k, rkey, 0, logits);
    crate::graph::give_fixed_i32(k, rkey, 1, ids);
    crate::graph::give_fixed_i32(k, gkey, 5, slots);
    let spin_us = tw0.elapsed().as_secs_f64() * 1e6;
    // `LINGXI_ROUTE_DEV_TRACE=1`：每 480 次（= 10 token × 48 层）打一行命中率。
    if crate::flag("LINGXI_ROUTE_DEV_TRACE") {
        use std::sync::atomic::{AtomicU64, Ordering};
        static HIT: AtomicU64 = AtomicU64::new(0);
        static MISS: AtomicU64 = AtomicU64::new(0);
        static FAIL: AtomicU64 = AtomicU64::new(0);
        if nmiss == 0 {
            HIT.fetch_add(1, Ordering::Relaxed);
        } else {
            MISS.fetch_add(1, Ordering::Relaxed);
            FAIL.fetch_add(1, Ordering::Relaxed);
        }
        let n = HIT.load(Ordering::Relaxed) + MISS.load(Ordering::Relaxed);
        if n % 480 == 0 {
            eprintln!(
                "[route-dev] 层调用 {n}：首趟全中 {} / 需补槽 {}（补后仍失败 {}）｜路由核 {:.1} µs/层，决策段 wall {:.1} µs/层",
                HIT.load(Ordering::Relaxed),
                MISS.load(Ordering::Relaxed),
                FAIL.load(Ordering::Relaxed),
                1000.0 * dev_ms as f64 / 480.0,
                spin_us,
            );
        }
    }
    if nmiss != 0 {
        // 补槽后仍 miss（例如槽位耗尽）：回退 legacy，**不留设备侧副作用**。
        crate::graph::give_fixed_f32(k, rkey, 2, wts);
        crate::graph::give_fixed_i32(k, rkey, 4, rowsb);
        return Ok(None);
    }
    // ③ x → 图体固定槽（与 legacy 同：图里烘的是固定地址，不能被池借还打乱）
    let t_stage = std::time::Instant::now();
    {
        let mut xs = crate::graph::take_fixed(k, gkey, 4, hidden)?;
        k.stream.memcpy_dtod(x, &mut xs)?;
        crate::graph::give_fixed(k, gkey, 4, xs);
    }
    let stage_us = t_stage.elapsed().as_secs_f64() * 1e6;
    let body = || -> Result<()> {
        let mut gu_all = crate::graph::take_fixed(k, gkey, 0, s * 2 * inter)?;
        let mut act_all = crate::graph::take_fixed(k, gkey, 1, s * inter)?;
        let mut o_all = crate::graph::take_fixed(k, gkey, 2, s * hidden)?;
        let xin = crate::graph::take_fixed(k, gkey, 4, hidden)?;
        let qin = crate::graph::take_fixed_i32(k, gkey, 5, s)?;
        let rpw = slots_rpw();
        if qc.bits == 3 {
            // Q3G128：dense LSB-first 3-bit 位流核（A2 路由只有 rows==1 逐对路径，无 TC/dp4a 档）。
            k.gemv_q3g128_slots_rows_bf16(
                dv.slots, &qin, &xin, &mut gu_all, s, 2 * inter, hidden, qc.gu_k_pad, 0,
                qc.blob, qc.gu_seg, qc.gu_scale, rpw,
            )?;
            k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
            k.gemv_q3g128_slots_rows_bf16(
                dv.slots, &qin, &act_all, &mut o_all, s, hidden, inter, qc.dn_k_pad, inter,
                qc.blob, qc.dn_seg, qc.dn_scale, rpw,
            )?;
        } else {
        k.gemv_q4g64_slots_rows_bf16(
            dv.slots, &qin, &xin, &mut gu_all, s, 2 * inter, hidden, qc.gu_k_pad, 0,
            qc.blob, qc.gu_seg, qc.gu_scale, rpw,
        )?;
        k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
        k.gemv_q4g64_slots_rows_bf16(
            dv.slots, &qin, &act_all, &mut o_all, s, hidden, inter, qc.dn_k_pad, inter,
            qc.blob, qc.dn_seg, qc.dn_scale, rpw,
        )?;
        }
        crate::graph::give_fixed(k, gkey, 0, gu_all);
        crate::graph::give_fixed(k, gkey, 1, act_all);
        crate::graph::give_fixed(k, gkey, 2, o_all);
        crate::graph::give_fixed(k, gkey, 4, xin);
        crate::graph::give_fixed_i32(k, gkey, 5, qin);
        Ok(())
    };
    let t_body = std::time::Instant::now();
    if crate::graph::enabled() && !crate::flag("LINGXI_NO_MOE_GRAPH") {
        crate::graph::run_cached(k, gkey, body)?;
    } else {
        body()?;
    }
    let body_us = t_body.elapsed().as_secs_f64() * 1e6;
    // combine 图外（dst 走 scratch 池；`rows`/`wts` 是路由核直写的固定槽）
    let t_comb = std::time::Instant::now();
    let o_fix = crate::graph::take_fixed(k, gkey, 2, s * hidden)?;
    let mut fo = scratch::bf16(k, hidden)?;
    k.moe_combine_add_rows_bf16(&o_fix, &rowsb, &wts, shared, &mut fo, s, 1, hidden)?;
    crate::graph::give_fixed(k, gkey, 2, o_fix);
    crate::graph::give_fixed_f32(k, rkey, 2, wts);
    crate::graph::give_fixed_i32(k, rkey, 4, rowsb);
    let comb_us = t_comb.elapsed().as_secs_f64() * 1e6;
    if trace {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        static US: AtomicU64 = AtomicU64::new(0);
        static ST: AtomicU64 = AtomicU64::new(0);
        static BD: AtomicU64 = AtomicU64::new(0);
        static CB: AtomicU64 = AtomicU64::new(0);
        let t = tw0.elapsed().as_secs_f64() * 1e6;
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        let acc = US.fetch_add(t as u64, Ordering::Relaxed) + t as u64;
        let _ = ST.fetch_add(stage_us as u64, Ordering::Relaxed);
        let _ = BD.fetch_add(body_us as u64, Ordering::Relaxed);
        let _ = CB.fetch_add(comb_us as u64, Ordering::Relaxed);
        if n % 480 == 0 {
            eprintln!(
                "[route-dev] 整段 {:.1} µs/层（决策 {:.1}｜stage {:.1} body {:.1} combine {:.1}）",
                acc as f64 / n as f64,
                spin_us,
                ST.load(Ordering::Relaxed) as f64 / n as f64,
                BD.load(Ordering::Relaxed) as f64 / n as f64,
                CB.load(Ordering::Relaxed) as f64 / n as f64,
            );
        }
    }
    Ok(Some(fo))
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
    static ROUTE_WS: std::cell::RefCell<RouteWs> = const { std::cell::RefCell::new(RouteWs {
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

pub fn sparse_moe_forward(
    k: &Kernels,
    w: &MoeWeights,
    x: &CudaSlice<bf16>,
    rows: usize,
    hidden: usize,
    num_experts: usize,
    top_k: usize,
    inter: usize,
    debug_dir: Option<&std::path::Path>,
    layer: u32,
    src: Option<&dyn ExpertSource>,
    remote: Option<&RemoteMoe>,
) -> Result<MoeOutput> {
    // ★ **跨卡**：这些层的 MoE 在 GPU1 上跑（`LINGXI_MOE_SPLIT`）。跨卡只过激活。
    if let Some(rm) = remote {
        if rm.layers.contains(&layer) {
            let w1 = rm
                .moe
                .get(layer as usize)
                .and_then(|o| o.as_ref())
                .ok_or_else(|| anyhow::anyhow!("MoE 跨卡：层 {layer} 的 GPU1 权重缺失"))?;
            let t0 = std::time::Instant::now();
            // ⚠️ cudarc 的错误是**粘在上下文上**的（`record_err` 只是存进 `error_state`，
            //    要等**下一次** `bind_to_thread` 才吐出来）⇒ 跨卡路径上"报错的位置"常常不是
            //    "出错的位置"（实测：真凶在前面，却在几步之后的 `scratch::upload_i32` 才炸）。
            //    ⇒ 每个阶段后主动 `check_err`（它取走并清空错误），把报错点钉回真凶。
            //    每次只一个原子 swap，开销可忽略。
            let chk = |tag: &str| -> Result<()> {
                rm.k
                    .ctx
                    .check_err()
                    .map_err(|e| anyhow::anyhow!("[moe-split/{tag}] GPU1 上下文报错: {e}"))
            };
            chk("enter")?;
            // ── 两块跨卡缓冲的分配 ──
            // ⚠️ 必须走 `without_pool`：scratch 池的键只有 (类型, 长度)、**没有设备**，
            //    直接 `scratch::bf16(rm.k, …)` 会拿到一块 GPU0 的缓冲（实测 ILLEGAL_ADDRESS）。
            let mut x1 = crate::scratch::without_pool(|| crate::scratch::bf16(rm.k, x.len()))?;
            let mut out =
                crate::scratch::without_pool(|| crate::scratch::bf16(k, rows * hidden))?;
            let rt0 = std::time::Instant::now();
            // 输入激活 → GPU1（走**两张卡各自的锁页 staging**，不再是可分页 Vec）
            crate::scratch::cross_copy(k, x, rm.k, &mut x1)?;
            let rt1 = std::time::Instant::now();
            chk("激活上传")?;
            // 在 GPU1 上原样重跑（权重/专家源都换成 GPU1 的；`remote=None` 防递归）
            let mo1 = crate::scratch::without_pool(|| {
                sparse_moe_forward(
                    rm.k, w1, &x1, rows, hidden, num_experts, top_k, inter, debug_dir, layer,
                    Some(rm.experts), None,
                )
            })?;
            let rt2 = std::time::Instant::now();
            chk("MoE 计算")?;
            // 输出 → GPU0
            crate::scratch::cross_copy(rm.k, &mo1.out, k, &mut out)?;
            chk("激活回传")?;
            {
                // 诊断字段（`shared`/`expert_out`）在跨卡路径下是**占位**：需要它们就别开 MOE_SPLIT。
                use std::sync::atomic::Ordering::Relaxed;
                rm.stat[0].fetch_add(1, Relaxed);
                rm.stat[1].fetch_add(t0.elapsed().as_micros() as u64, Relaxed);
                rm.stat[2]
                    .fetch_add((rt1 - rt0 + rt2.elapsed()).as_micros() as u64, Relaxed);
            }
            return Ok(MoeOutput {
                out,
                router_logits: None,
                shared: unsafe { k.stream.alloc::<bf16>(1) }?,
                expert_out: unsafe { k.stream.alloc::<bf16>(1) }?,
                per_expert: mo1.per_expert,
                sel_rows: mo1.sel_rows,
            });
        }
    }
    // ───── shared expert ─────
    let prof = crate::flag("LINGXI_PROF");
    let t_start = std::time::Instant::now();
    let mut act = scratch::bf16(k, rows * inter)?;
    let mut shared = scratch::bf16(k, rows * hidden)?;
    // shared_expert_gate = router_shared_gate 的最后一行（sigmoid 折进 GEMV 尾）
    let sg_row = w.router_shared_gate.slice(num_experts * hidden..(num_experts + 1) * hidden);
    let mut sg = scratch::bf16(k, rows)?;
    // ★ P15-⑨（t>1）：x 预转 fp32 一次，shared 三个投影共用（m>1 时内层免重复 CVT，
    // 带宽 ~350→~700 GB/s；epi=4 拆成"裸 GEMM + swiglu"，与 proj_t1_epi 内部同构、逐位等价）。
    let xf = crate::gpu::prep_f32x(k, x, rows, rows * hidden, 1)?;
    if let Some(xf) = xf.as_ref() {
        let mut act_full = scratch::bf16(k, rows * 2 * inter)?;
        crate::gpu::proj_f32x(k, &w.shared_gate_up, xf, &mut act_full, 2 * inter, hidden, rows, 0, 0.0, 0.0)?;
        k.swiglu_bf16(&act_full, &mut act, rows, inter)?;
        scratch::put_bf16(act_full);
        // ⚠️ shared_down 的输入是 **act**（swiglu 输出），不是 MoE 入口 x ——
        //    这里保持 bf16 现转（P15-⑨ 第一轮实 bug：误传 xf 导致首 token 漂移）。
        proj_t1(k, &act, &w.shared_down, &mut shared, hidden, inter, rows)?;
        crate::gpu::proj_f32x(k, &sg_row, xf, &mut sg, 1, hidden, rows, 2, 1.0, 1.0)?;
    } else {
        // swiglu 折进 GEMV 尾：`shared_gate_up` 的前 I 行是 gate、后 I 行是 up，一次算完。
        proj_t1_epi(k, x, &w.shared_gate_up, &mut act, 2 * inter, hidden, rows, 4, 0.0, 0.0)?;
        proj_t1(k, &act, &w.shared_down, &mut shared, hidden, inter, rows)?;
        proj_t1_epi(k, x, &sg_row, &mut sg, 1, hidden, rows, 2, 1.0, 1.0)?;
    }
    scratch::put_bf16(act);
    k.row_scale_inplace_bf16(&mut shared, &sg, rows, hidden)?;
    scratch::put_bf16(sg);
    // `LINGXI_KTS=1`：给 `[moe]` 的三段插同步括号，让它们变成**真 GPU 时间**（否则只是 CPU 发射时间）。
    if crate::flag("LINGXI_KTS") {
        k.stream.synchronize()?;
    }
    let t_shared = t_start.elapsed();

    // ㊺ A2 设备路由：rows==1 时把「router GEMV → 同步 → host softmax/topk → host 查槽 →
    // staging」整条 host 决策链压成一个 GPU 核 + 一次事件自旋。命中全中 → 直接出结果；
    // 任一选中专家未入槽（miss）→ None，落回 legacy 全路径（本函数写过的设备缓冲无副作用）。
    // `LINGXI_ROUTE_DEV=1` 开启；判据未过前默认关。
    let route_dev_on = rows == 1
        && crate::flag("LINGXI_ROUTE_DEV")
        && !crate::flag("LINGXI_MOE_DEBUG")
        && !crate::flag("LINGXI_MOE_DUMP");
    if route_dev_on {
        if let Some(s) = src {
            if let Some(out) = route_dev_attempt(
                k, layer, w, x, &shared, s, num_experts, top_k, inter, hidden,
            )? {
                if let Some(p) = k.evprof.as_ref() {
                    p.end(layer, 3, &k.stream);
                    p.begin(layer, 4, &k.stream);
                    p.set_info(layer, 0, top_k as u32);
                    p.set_info(layer, 1, top_k as u32);
                    p.set_info(layer, 2, 1);
                    p.end(layer, 4, &k.stream);
                }
                return Ok(MoeOutput {
                    out,
                    router_logits: None,
                    shared,
                    expert_out: unsafe { k.stream.alloc::<bf16>(1)? },
                    per_expert: Vec::new(),
                    sel_rows: Vec::new(),
                });
            }
            // miss：继续走 legacy 全路径（下面原样）
        }
    }

    // ───── router（在 host 上按 f32 做 softmax / top-k / 归一化） ─────
    // [探针] LINGXI_MOE_DUMP=<dir>：单调 call 计数落 MoE 入口 x、router logits、
    // 每行 top-k 选择（默认关，不改变热路径分配/时序）。仅用于 engine vs HF 定位。
    let dump_dir = if crate::flag("LINGXI_MOE_DUMP") {
        std::env::var("LINGXI_MOE_DUMP").ok().map(std::path::PathBuf::from)
    } else {
        None
    };
    let call_id: u64 = if dump_dir.is_some() {
        static CALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    } else {
        0
    };
    if let Some(d) = &dump_dir {
        let xh: Vec<f32> = k.stream.clone_dtoh(x)?.iter().map(|v| v.to_f32()).collect();
        let bytes: Vec<u8> = xh.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(d.join(format!("moe{call_id:03}_x.f32")), bytes)?;
    }
    let router_w = w.router_shared_gate.slice(0..num_experts * hidden);
    let need = rows * num_experts;
    let t_rg = std::time::Instant::now();
    // ── 取回 host：**零拷贝**路径（所有 rows）──
    // 让 GEMM/GEMV 直接把 logits 写进「映射到设备的锁页 host 内存」，host 侧只剩一次
    // `cuStreamSynchronize`。原路径（`cuMemcpyDtoHAsync` + sync）实测**光这 1 KB 的拷贝
    // 就让同步多等 ~0.8 ms/层**（见 docs/decode-speed-profile.md 的 [d2h] 拆解），
    // 48 层累积 ~38 ms/token。`LINGXI_ROUTER_NODMAP=1` 可回退。
    // 【本轮修正】原来这里是 `rows == 1 && ...` —— 于是 **t>1 又跌回 D2H 老路**，
    // 而 §3.42 用「剥块后看 Δ(t=2−t=1) 是否塌缩」实测：**t=2 悬崖 +35.8 ms 里有 +23.4 ms 在 MoE**，
    // 正是这条 0.8 ms/层 × 48。现在 rows>1 也走映射（用瘦长 GEMM 直写映射内存，免掉 D2H）。
    let use_map = !crate::flag("LINGXI_ROUTER_NODMAP");
    let mut logits_dev: Option<CudaSlice<bf16>> = None;
    let logits_host: Vec<bf16> = if use_map {
        thread_local! {
            static MAP: std::cell::RefCell<Option<scratch::MappedHost<bf16>>> =
                const { std::cell::RefCell::new(None) };
        }
        MAP.with(|c| -> Result<Vec<bf16>> {
            let mut g = c.borrow_mut();
            // ⚠️ `MappedHost` 里的设备别名是 `cuMemHostGetDevicePointer` 在**它创建时的上下文**
            //    里取的 ⇒ 线程级缓存必须按上下文隔离，否则 GPU1 的核会拿到 GPU0 的别名
            //    （双卡实测：`CUDA_ERROR_ILLEGAL_ADDRESS`）。
            let reuse = g.as_ref().map_or(false, |m| {
                std::sync::Arc::ptr_eq(m.ctx(), k.stream.context()) && m.len() == need
            });
            if !reuse {
                *g = Some(unsafe { scratch::MappedHost::<bf16>::new(k, need)? });
            }
            let m = g.as_mut().unwrap();
            let (wp, _a) = router_w.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            if rows == 1 {
                k.gemv_bf16_ptr(wp, xp, m.dev, num_experts, hidden)?;
            } else {
                // rows>1：瘦长 GEMM 直写映射内存。`hidden=2560 >= 1024` ⇒ 走 `gemm_skinny_row`
                //（一行一 block + k 方向 128 位读），与 `proj_t1` 的分派一致。
                // ⚠️ 该核**只支持 1≤m≤8**（见 kernels 侧的校验）⇒ 必须按 ≤8 行切片，否则
                //    长 prompt（例如 23 行命中同一专家）会直接失败：
                //    `Error: skinny-row 只支持 1≤m≤8（给到 23）`。
                let mut done = 0usize;
                while done < rows {
                    let mrow = (rows - done).min(8);
                    // 行步长按 **bf16 = 2 字节** 算：x 是 [rows, hidden]、y 是 [rows, num_experts]，
                    // 与 `proj_t1` 里 `xp + done*kdim*2` / `yp + done*n*2` 同一口径。
                    k.gemm_skinny_row_bf16_ptr(
                        wp,
                        xp + (done * hidden * 2) as u64,
                        m.dev + (done * num_experts * 2) as u64,
                        num_experts,
                        hidden,
                        mrow,
                        0,
                        0.0,
                        0.0,
                    )?;
                    done += mrow;
                }
            }
            k.stream.synchronize()?;
            Ok(m.slice()?[..need].to_vec())
        })?
    } else {
        let mut logits = scratch::bf16(k, rows * num_experts)?;
    match crate::gpu::prep_f32x(k, x, rows, rows * hidden, 1)? {
        // ★ P15-⑨：t>1 时 x 预转 fp32（与上面的 shared 段各转一份 scratch，池化复用、几乎免费）
        Some(xf) => crate::gpu::proj_f32x(k, &router_w, &xf, &mut logits, num_experts, hidden, rows, 0, 0.0, 0.0)?,
        None => proj_t1(k, x, &router_w, &mut logits, num_experts, hidden, rows)?,
    }
        let lh = {
            thread_local! {
                static PIN: std::cell::RefCell<Option<cudarc::driver::PinnedHostSlice<bf16>>> =
                    const { std::cell::RefCell::new(None) };
            }
            PIN.with(|c| -> Result<Vec<bf16>> {
                let mut g = c.borrow_mut();
                // ⚠️ 同 `scratch::same_ctx`：线程级锁页缓存**按上下文隔离** ——
                //    跨上下文用会 `cuStreamWaitEvent` 失败（INVALID_VALUE，双卡实测）。
                let reuse = g.as_ref().map_or(false, |p| {
                    std::sync::Arc::ptr_eq(p.context(), k.stream.context()) && p.len() >= need
                });
                if !reuse {
                    *g = Some(unsafe {
                        k.stream.context().alloc_pinned_with_flags::<bf16>(need, 0)
                    }?);
                }
                let buf = g.as_mut().unwrap();
                k.stream.memcpy_dtoh(&logits, &mut buf.as_mut_slice()?[..need])?;
                k.stream.synchronize()?; // `memcpy_dtoh` 是异步的，必须等完成再读 host 缓冲
                Ok(buf.as_slice()?[..need].to_vec())
            })?
        };
        logits_dev = Some(logits);
        lh
    };
    let t_rd = std::time::Instant::now();
    let t_rt = t_rd;

    // ★ P15-⑧①：簿记表全部走复用工作区（稳态零分配；见 `RouteWs` 的注释）。
    // `probs` 每行会被整行覆写（`probs[i]=e` 对所有 i），无需清零。
    let mut ws = ROUTE_WS.with(|c| c.replace(RouteWs::default()));
    if ws.per_expert.len() != num_experts {
        ws.per_expert.clear();
        ws.per_expert.resize_with(num_experts, Vec::new);
    }
    for v in ws.per_expert.iter_mut() {
        v.clear();
    }
    ws.sel_rows.clear();
    ws.probs.resize(num_experts, 0.0);
    ws.sel.clear();
    ws.hit_ids.clear();
    ws.pair_experts.clear();
    ws.pair_rows.clear();
    ws.pair_wts.clear();
    ws.uniq_experts.clear();
    ws.group_off.clear();
    ws.rows_of.clear();
    ws.wts_1.clear();
    let RouteWs {
        per_expert,
        sel_rows,
        probs,
        sel,
        rows_of,
        wts_1,
        ..
    } = &mut ws;
    // ⚠️ 这里原来是 `order = vec![0..num_experts]` + `sort_by(...)`：每行分配 4 KB、
    // 跑满 `512·log512` 次**带闭包**的比较。第二版是 top_k 趟全扫 selection（`top_k × 512`
    // 次无闭包比较）。★ P15-⑰ 改 **单趟插入式 top-k**：「严格大于第 k 名才进、插到第一个
    // 严格更小的元素之前」⇒ 同值按下标升序 ⇒ 与旧实现选出的集合与顺序**逐位相同**
    // （全序 (值降序, 下标升序) 的前 top_k 名唯一）。绝大多数元素一次比较即退，
    // t=3 实测 topk ~80µs → ~10µs（`LINGXI_PROF` 的 `topk` 列）。
    for r in 0..rows {
        let lr = &logits_host[r * num_experts..(r + 1) * num_experts];
        let m = lr.iter().map(|v| v.to_f32()).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (i, v) in lr.iter().enumerate() {
            let e = (v.to_f32() - m).exp();
            probs[i] = e;
            sum += e;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        sel.clear();
        for i in 0..num_experts {
            let p = probs[i];
            if sel.len() == top_k && !(p > sel[sel.len() - 1].1) {
                continue;
            }
            let pos = sel.iter().position(|&(_, v)| p > v).unwrap_or(sel.len());
            sel.insert(pos, (i, p));
            if sel.len() > top_k {
                sel.pop();
            }
        }
        let norm: f32 = sel.iter().map(|(_, p)| *p).sum();
        // `per_expert` 按专家下标升序枚举，这里排一下保持同一顺序
        sel.sort_by_key(|(i, _)| *i);
        for &(e, p) in sel.iter() {
            per_expert[e].push((r as i32, p / norm));
            sel_rows.push((r as u32, e as i32, p / norm));
        }
    }
    if let Some(d) = &dump_dir {
        // logits（f32，rows*num_experts）+ 每行 top-k 专家 id（i32）
        let lf: Vec<f32> = logits_host.iter().map(|v| v.to_f32()).collect();
        let lb: Vec<u8> = lf.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(d.join(format!("moe{call_id:03}_logits.f32")), lb)?;
        let sel: Vec<i32> = sel_rows.iter().map(|(_, e, _)| *e).collect();
        let sb: Vec<u8> = sel.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(d.join(format!("moe{call_id:03}_sel.i32")), sb)?;
    }

    // [探针] `LINGXI_ARENA_FIXED_EXPERTS=1`：把路由**钉死**成固定的前 top_k 个专家。
    // 数学作废，但**kernel 数、launch 数、缓冲、同步点全都与正常路径完全一致**，
    // 只是专家槽在第一次之后 100% 命中（不再有 H2D DMA），且路由不再乱跳。
    // 与 `LINGXI_FIXED_IDS` 组合使用，就能干净地量出"每层那点开销到底来自主机交互还是 GPU 本身"。
    // 🚫 以前这里写着"与 `LINGXI_ROUTER_NOSYNC=1` 组合"——**那个开关代码里根本没有实现**
    //    （已登记为 Ghost，见 probes.rs）。照它做实验等于什么都没变，别再用。
    if crate::flag("LINGXI_ARENA_FIXED_EXPERTS") {
        for v in per_expert.iter_mut() {
            v.clear();
        }
        sel_rows.clear();
        let wgt = 1.0f32 / top_k as f32;
        for e in 0..top_k {
            per_expert[e].push((0i32, wgt));
            sel_rows.push((0u32, e as i32, wgt));
        }
    }

    // ───── 稀疏专家 ─────
    let t_rk = std::time::Instant::now();
    if crate::flag("LINGXI_KTS") {
        k.stream.synchronize()?;
    }
    let t_router = t_start.elapsed();
    // 权重两路径，数学完全相同（同字节、同专家升序遍历，仅存放布局不同）：
    //  Full    ：w.routed_* 是 [E,…] 全量张量，按专家 id 直接切片（对拍路径）
    //  Sparse  ：路由后经 ExpertSource 只取回命中专家的 [S,…] 紧凑张量，按槽位切片
    let hit_ids: Vec<u32> = per_expert
        .iter()
        .enumerate()
        .filter(|(_, hits)| !hits.is_empty())
        .map(|(e, _)| e as u32)
        .collect();
    // **按 (专家, 行) 对**展开。`rows == 1` 时每个专家恰好命中一次 → 与 `hit_ids` 完全相同；
    // `rows > 1` 时一对一个条目，**允许同一专家出现多次**（不同 token 命中同一专家）。
    // 这样"批量"路径可以完全复用 decode 的 Q4 直读核：把每条对当成一个独立 GEMV 行，
    // 先把 x 按行号 gather 出来，再算、再按行号加权归并。
    // 原来 rows>1 走的是**逐专家**路径（每层 ~2×非空专家个数 次 cuBLAS GEMV，
    // 22 token 的 prompt 每层要几百个 kernel），这是 prefill 慢的根因。
    let pair_experts: Vec<u32> = per_expert
        .iter()
        .enumerate()
        .flat_map(|(e, hits)| hits.iter().map(move |_| e as u32))
        .collect();
    let pair_rows: Vec<i32> =
        per_expert.iter().flat_map(|hits| hits.iter().map(|(r, _)| *r)).collect();
    let pair_wts: Vec<f32> =
        per_expert.iter().flat_map(|hits| hits.iter().map(|(_, w)| *w)).collect();
    // **唯一专家**（升序）+ 每个唯一专家在"对"列表里的区间 `[group_off[i], group_off[i+1])`。
    // `pair_experts` 本来就是按专家升序拼出来的，所以区间天然连续。
    // 批量路径靠它把"同一个专家的权重读 t 遍"变成"读一遍、被 t 个 token 共用"（§3.21 根因 3）。
    let mut uniq_experts: Vec<u32> = Vec::new();
    let mut group_off: Vec<i32> = Vec::new();
    {
        let mut acc = 0i32;
        for (e, hits) in per_expert.iter().enumerate() {
            if hits.is_empty() {
                continue;
            }
            uniq_experts.push(e as u32);
            group_off.push(acc);
            acc += hits.len() as i32;
        }
        group_off.push(acc);
    }
    let compact_gu;
    let compact_dn;
    let q4_hits;
    let q4_alt_hits;
    let bits_sel;
    // ★ **专家段 hit/miss 重叠**（`LINGXI_MOE_OVERLAP`，§P4；㊶ 起 **rows==1 也生效**）：
    //   rows>1 时 slot_ids[i] 与唯一专家 i 一一对应；rows==1 时对列表==专家列表，
    //   同样一一对应 ⇒ 两种行数都能按"槽条目"切 hit/miss（arena_source 按 overlap 参数
    //   把 miss 的 blob H2D 挪去 stream2，见 arena_source.rs）。
    let mo_ov = moe_overlap_on();
    // `LINGXI_EVPROF=1`：MoE 内部再切一刀 —— 段 3（路由+host）到此为止，段 4（查槽+专家 GEMV+归并）开始。
    // 同时把"现场"记进计时器（对数 / 组数 / rows），好让 dump 直接算出相关性。
    if let Some(p) = k.evprof.as_ref() {
        p.end(layer, 3, &k.stream);
        p.begin(layer, 4, &k.stream);
        p.set_info(layer, 0, pair_experts.len() as u32);
        p.set_info(layer, 1, uniq_experts.len() as u32);
        p.set_info(layer, 2, rows as u32);
    }
    // ㊽ 段 4 六事件剖面链（`LINGXI_MOE_PROF=1`）：A=段起点 B=run(gu) 前 C=gu 后
    // D=swiglu+xf 后 E=run(dn) 后 F=combine 后。A→B 兜住 upload/gather/load_hits/xf，
    // E→F 兜住 combine+收尾。事件只记不同步。
    let prof4 = crate::flag("LINGXI_MOE_PROF");
    let mut evs: Vec<cudarc::driver::CudaEvent> = Vec::new();
    if prof4 {
        let f = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        for _ in 0..6 {
            evs.push(k.ctx.new_event(f)?);
        }
        evs[0].record(&k.stream)?;
    }
    // ★ P15-⑰：uploads + gather **提前到 load_hits 之前**——它们只依赖路由结果
    // （pair_rows/pair_wts/x），不依赖专家权重。让 gather 的 GPU 执行与 load_hits 的
    // host 查找（实测 ~50µs/层）重叠，消一段 GPU 死时间。只调执行顺序，stream 上
    // 依赖关系不变 ⇒ 逐位不变。回退路径（NO_PAIR_MOE / 无 src / 空对）不预做。
    // 与 pair_path 的条件差在 `q4_hits`（load_hits 才知道）⇒ 没用上时在函数尾归还池。
    let mut pre: Option<(
        CudaSlice<i32>,
        CudaSlice<f32>,
        CudaSlice<i32>,
        CudaSlice<i32>,
        Option<CudaSlice<bf16>>,
    )> = None;
    if src.is_some() && !pair_experts.is_empty() && !crate::flag("LINGXI_NO_PAIR_MOE") {
        let s = pair_experts.len();
        let d_rows = scratch::upload_i32(k, &pair_rows)?;
        let d_wts = scratch::upload_f32(k, &pair_wts)?;
        // ㊾ 行 CSR（稳定计数排序，O(s+rows)）：`col[row_off[r]..row_off[r+1])` = 该行所有
        // 对的原下标 s 且**升序** ⇒ CSR combine 核的累加次序与旧核"升序 s 扫描"逐位相同。
        // rows==1 时 S 小（旧核扫描便宜），不上传 CSR，combine 走旧核。
        let (d_col, d_roff) = if rows > 1 {
            let mut roff = vec![0i32; rows + 1];
            for &r in pair_rows.iter() {
                roff[r as usize + 1] += 1;
            }
            for i in 0..rows {
                roff[i + 1] += roff[i];
            }
            let mut col = vec![0i32; s];
            let mut cur: Vec<i32> = roff[..rows].to_vec();
            for (si, &r) in pair_rows.iter().enumerate() {
                let rr = r as usize;
                col[cur[rr] as usize] = si as i32;
                cur[rr] += 1;
            }
            (scratch::upload_i32(k, &col)?, scratch::upload_i32(k, &roff)?)
        } else {
            (scratch::upload_i32(k, &[])?, scratch::upload_i32(k, &[])?)
        };
        let xg = if rows == 1 {
            None
        } else {
            let mut g = scratch::bf16(k, s * hidden)?;
            k.gather_rows_bf16(x, &d_rows, &mut g, s, hidden, 0)?;
            Some(g)
        };
        pre = Some((d_rows, d_wts, d_col, d_roff, xg));
    }
    let t_lh = std::time::Instant::now();
    if let Some(s) = src {
        // `rows == 1`：对列表与专家列表等价，沿用旧核（不引入 group_off 上传）。
        // `rows > 1`：传**唯一专家**列表，`slot_ids[i]` 与 `blockIdx.y` 一一对应。
        // `want_q4` 必须与"调用方会不会走 Q4 直读"一致：只有会走时才允许返回 q4 占位，
        // 否则调用方会拿 1 元素的 stub 当真权重读（会非法访问）。
        let want_q4 = !crate::flag("LINGXI_NO_PAIR_MOE");
        let hit_list: &[u32] = if rows == 1 { &pair_experts } else { &uniq_experts };
        let hits = s.load_hits(layer, hit_list, inter, hidden, want_q4, mo_ov)?;
        if prof {
            eprintln!("    [moe] 层 {layer:2} load_hits {:.1} µs (uniq {})",
                t_lh.elapsed().as_secs_f64() * 1e6, uniq_experts.len());
        }
        compact_gu = Some(hits.gate_up);
        compact_dn = Some(hits.down);
        q4_hits = hits.q4;
        q4_alt_hits = hits.q4_alt;
        bits_sel = hits.bits_sel;
    } else {
        compact_gu = None;
        compact_dn = None;
        q4_hits = None;
        q4_alt_hits = None;
        bits_sel = Vec::new();
    }
    // 混档双视图（6.2-2 Stage B）：把命中列表切成**同格式连续段**——每对只被一个
    // 格式写、buffer 全局按对下标 ⇒ combine 无需改动。单格式 ⇒ runs4 = [0, len)。
    // 注意段下标与 slot_ids/组的编号同源（都对着 hit_list = rows==1 ? pair : uniq）。
    let (mut runs4, mut runs3): (Vec<(u32, u32)>, Vec<(u32, u32)>) = (Vec::new(), Vec::new());
    {
        let n = bits_sel.len();
        let mut i = 0usize;
        while i < n {
            let b = bits_sel[i];
            let mut j = i + 1;
            while j < n && bits_sel[j] == b {
                j += 1;
            }
            (if b == 3 { &mut runs3 } else { &mut runs4 }).push((i as u32, j as u32));
            i = j;
        }
    }
    let mut dual_q4 = q4_alt_hits.is_some();
    // 判决性诊断：`LINGXI_MIX_FORCE_DUAL=1` 让单格式 arena 也走双视图循环
    // （单视图、整段 [0,len)）——对照实验：双调机制本身是否有 bug。
    if !dual_q4 && crate::flag("LINGXI_MIX_FORCE_DUAL") && q4_hits.is_some() {
        dual_q4 = true;
        let n = bits_sel.len() as u32;
        // 默认两段拆（验非零起点切片）；=1 时整段（验视图直通）。
        let split: u32 = std::env::var("LINGXI_MIX_FORCE_DUAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
            .max(1);
        if split <= 1 || n < 2 {
            runs4 = vec![(0, n)];
        } else {
            let mid = n / 2;
            runs4 = vec![(0, mid), (mid, n)];
        }
    }
    // `slot_of` 只在"逐专家慢路径"里用到（快路径整层 5 个 kernel）。原来无条件构造：
    // 每层一次 4 KB 分配 + 512 次写，而 decode 永远走快路径 → 纯浪费。改成按需构造。
    let pair_path = src.is_some()
        && q4_hits.is_some()
        && !pair_experts.is_empty()
        && !crate::flag("LINGXI_NO_PAIR_MOE");
    let slot_of: Vec<usize> = if pair_path {
        Vec::new()
    } else {
        let mut v = vec![usize::MAX; num_experts];
        for (slot, &e) in hit_ids.iter().enumerate() {
            v[e as usize] = slot;
        }
        v
    };

    // 快路径由 `moe_combine_rows_bf16` **整体赋值**写出；逐专家路径是
    // `scatter_add_rows` 累加，必须从零起（故两路分配方式不同）。
    // 注意：两条路都必须在 scratch 池里成对 take/put，否则池子会逐层长出新块、
    // "同一逻辑缓冲"的地址就会逐 token 漂移（实测 [ptrprobe] 里 +3/层）。
    let fast_path = pair_path || (src.is_some() && rows == 1 && !hit_ids.is_empty());
    let mut expert_out = if fast_path {
        scratch::bf16(k, rows * hidden)?
    } else {
        k.stream.alloc_zeros::<bf16>(rows * hidden)?
    };
    // 诊断字段开关（原在函数尾，㊶提前）：开着时专家段保持"combine 写 expert_out +
    // add 写 out"老两步（诊断要读 expert_out）；关时两条快路径用**融合核**一发直写 out。
    let dbg_routes = std::env::var("LINGXI_MOE_DEBUG").is_ok() || dump_dir.is_some();
    // 快路径融合核直写的 `out`（Some = 函数尾不再 add_bf16）。
    let mut out_fused: Option<CudaSlice<bf16>> = None;

    // ── 批量快路径：**5 个 kernel 搞定整层专家段**（与 rows 无关）──
    // gather(按行取 x) → Q4 直读 gemv(gu) → swiglu → Q4 直读 gemv(dn) → 按行加权归并
    // prof 分解用：专家段 host 发射起点 / uploads 耗时（pair_path 块内赋值）。
    let t_enq = std::time::Instant::now();
    let mut t_up_done = 0.0f64;
    if pair_path {
        // `LINGXI_KTS=1` 诊断：专家段的**分段 GPU 时间**（插同步括号；会破坏流水，只为定位）。
        let kts_m = crate::flag("LINGXI_KTS");
        if kts_m {
            k.stream.synchronize()?;
        }
        let t_m0 = std::time::Instant::now();
        let s = pair_experts.len();
        let q = q4_hits.as_ref().unwrap();
        // ★ P15-⑰：uploads/gather 已在 load_hits 之前预做（见上），这里只取回；
        // `up` 列从此记账 0（工作仍在，只是挪去和 load_hits 重叠了）。
        let t_up = std::time::Instant::now();
        let (d_rows, d_wts, d_col, d_roff, xg) = pre.take().expect("预做块与 pair_path 条件一致");
        t_up_done = t_up.elapsed().as_secs_f64() * 1e6;
        let xs: &CudaSlice<bf16> = xg.as_ref().unwrap_or(x);
        // [探针] 核对**设备侧的行号/权重表**：gather 只按行号取 x，行号若错会"总量不变、分组变"
        if crate::flag("LINGXI_MOE_OUT_DUMP") && {
            let want: u32 = std::env::var("LINGXI_MOE_OUT_DUMP_LAYER")
                .ok().and_then(|v| v.parse().ok())
                .map(|v: u32| v.wrapping_sub(1))
                .unwrap_or(crate::DIMS.n_layers as u32 - 1);
            layer == want
        } {
            use cudarc::driver::DevicePtr;
            let (p, _g) = d_rows.device_ptr(&k.stream);
            let mut br = vec![0i32; d_rows.len()];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync::<i32>(&mut br, p) }?;
            let bad = br.iter().zip(pair_rows.iter()).filter(|(a, b)| **a != **b).count();
            eprintln!(
                "[moe-out] 层{layer} d_rows 回读不一致={bad}/{} 前8 dev={:?} host={:?}",
                br.len(), &br[..8.min(br.len())], &pair_rows[..8.min(pair_rows.len())]
            );
            let (pw, _g2) = d_wts.device_ptr(&k.stream);
            let mut bw = vec![0f32; d_wts.len()];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync::<f32>(&mut bw, pw) }?;
            let badw = bw.iter().zip(pair_wts.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            eprintln!("[moe-out] 层{layer} d_wts 回读不一致={badw}/{}", bw.len());
        }
        // [探针] 核对 gather 出来的专家输入（`d_rows` 若错，这里先露馅）
        if crate::flag("LINGXI_MOE_OUT_DUMP") && {
            let want: u32 = std::env::var("LINGXI_MOE_OUT_DUMP_LAYER")
                .ok().and_then(|v| v.parse().ok())
                .map(|v: u32| v.wrapping_sub(1))
                .unwrap_or(crate::DIMS.n_layers as u32 - 1);
            layer == want
        } {
            if let Ok(h) = scratch::download_bf16(k, xs) {
                let mut s0 = 0f64;
                let mut mx = 0f32;
                for x in &h {
                    let f = x.to_f32();
                    s0 += f as f64;
                    if f.abs() > mx { mx = f.abs(); }
                }
                eprintln!("[moe-out] 层{layer} gather_x_sum={s0:.6} maxabs={mx:.4} n={}", h.len());
            }
        }
        // `a_stride` 是 GEMV 里 x 的**行步长**：rows==1 时所有对共用第 0 行（stride=0）；
        // rows>1 时 x 是 gather 出来的 [S, hidden]，必须用 hidden 当步长，否则每对都读第 0 行。
        let gu_stride: usize = if rows == 1 { 0 } else { hidden };

        let mut gu_all = scratch::bf16(k, s * 2 * inter)?;
        let mut act_all = scratch::bf16(k, s * inter)?;
        let mut o_all = scratch::bf16(k, s * hidden)?;
        // [探针] `LINGXI_MOE_ZERO_OUT=1`：专家段输出先清零 —— 判"是否有行没被写出"
        //   （若清零后两个配置的结果一致 ⇒ 差异来自**未初始化行**，不是数学）。
        if crate::flag("LINGXI_MOE_ZERO_OUT") {
            k.stream.memset_zeros(&mut gu_all)?;
            k.stream.memset_zeros(&mut act_all)?;
            k.stream.memset_zeros(&mut o_all)?;
        }
        // [探针] `LINGXI_MOE_NAN_FILL=1`：把三个输出缓冲先填 NaN（0x7FC0）—— 跑完数 NaN
        //   就能**逐元素定位"哪些位置没被写出"**（正确性判据，不是性能开关）。
        if crate::flag("LINGXI_MOE_NAN_FILL") {
            let nan = half::bf16::from_bits(0x7FC0);
            for buf in [&mut gu_all, &mut act_all, &mut o_all] {
                let v = vec![nan; buf.len()];
                k.stream.memcpy_htod(&v, buf)?;
            }
        }
        // `rows == 1`：每个专家恰好一个对 → 沿用"逐对"旧核（零额外上传）。
        // `rows > 1`：走"按唯一专家分组"的新核 —— 权重行只 load 一次、被该专家命中的
        // 所有 token 共用。t=8 时专家段流量因此从 ~3.3× 降到 ~1.2×（§3.21 根因 3）。
        if rows == 1 {
            let rpw = slots_rpw();
            if dual_q4 {
                // 混档：按格式段循环（每段用对应 cache 视图；段间对区间互不相交，
                // 与 ㊶ 拆 launch 同一论据——有损的是 Q3 档本身，不是拆分）。
                if std::env::var("LINGXI_MIX_DEBUG").is_ok() {
                    use std::sync::atomic::{AtomicU32, Ordering};
                    static ND: AtomicU32 = AtomicU32::new(0);
                    if ND.fetch_add(1, Ordering::Relaxed) < 200 {
                        eprintln!(
                            "[mix-dbg moe] layer={layer} rows={rows} s={s} runs4={runs4:?} runs3={runs3:?} bits_sel={bits_sel:?}"
                        );
                        for (tag, v) in [("pri", Some(q)), ("alt", q4_alt_hits.as_ref())] {
                            if let Some(v) = v {
                                eprintln!(
                                    "[mix-dbg view {tag}] bits={} blob={} gu_seg={} dn_seg={} gu_scale={} dn_scale={} gu_k_pad={} dn_k_pad={}",
                                    v.bits, v.blob, v.gu_seg, v.dn_seg, v.gu_scale, v.dn_scale, v.gu_k_pad, v.dn_k_pad
                                );
                            }
                        }
                    }
                }
                let mut views: Vec<(&Q4Slots, &Vec<(u32, u32)>)> = vec![(q, &runs4)];
                if let Some(qa) = q4_alt_hits.as_ref() {
                    views.push((qa, &runs3));
                }
                // 诊断：`LINGXI_MIX_ONLY=3/4` 只跑单格式段（数学不完整，只看该段输出是否 NaN）。
                if let Ok(v) = std::env::var("LINGXI_MIX_ONLY") {
                    views.retain(|(view, _)| view.bits.to_string() == v);
                }
                for (view, runs) in views.iter().copied() {
                    // gu 段（全部 run）
                    for &(a, b) in runs {
                        if b <= a {
                            continue;
                        }
                        let (a, b) = (a as usize, b as usize);
                        let ids = view.slot_ids.slice(a..b);
                        let xv = xs.slice(0..xs.len());
                        let mut gy = gu_all.slice_mut(a * 2 * inter..b * 2 * inter);
                        if view.bits == 3 {
                            k.gemv_q3g128_slots_rows_bf16_view(
                                view.slots, &ids, &xv, &mut gy, b - a, 2 * inter, hidden,
                                view.gu_k_pad, gu_stride, view.blob, view.gu_seg, view.gu_scale, rpw,
                            )?;
                        } else {
                            k.gemv_q4g64_slots_rows_bf16_view(
                                view.slots, &ids, &xv, &mut gy, b - a, 2 * inter, hidden,
                                view.gu_k_pad, gu_stride, view.blob, view.gu_seg, view.gu_scale, rpw,
                            )?;
                        }
                    }
                }
                // swiglu 整批（与 rows>1 同构：全量一次）
                k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                // [探针] `LINGXI_MIX_SYNC_DN=1`：dn 前全流同步 —— 判"步 1 输出坏是否
                //   异步上传/H2D 未就绪竞态"（同步后转好 ⇒ 坐实）。
                if crate::flag("LINGXI_MIX_SYNC_DN") {
                    k.stream.synchronize()?;
                }
                for (view, runs) in views {
                    // dn 段（全部 run）
                    for &(a, b) in runs {
                        if b <= a {
                            continue;
                        }
                        let (a, b) = (a as usize, b as usize);
                        let ids = view.slot_ids.slice(a..b);
                        let ac = act_all.slice(a * inter..b * inter);
                        let mut oy = o_all.slice_mut(a * hidden..b * hidden);
                        if view.bits == 3 {
                            k.gemv_q3g128_slots_rows_bf16_view(
                                view.slots, &ids, &ac, &mut oy, b - a, hidden, inter,
                                view.dn_k_pad, inter, view.blob, view.dn_seg, view.dn_scale, rpw,
                            )?;
                        } else {
                            k.gemv_q4g64_slots_rows_bf16_view(
                                view.slots, &ids, &ac, &mut oy, b - a, hidden, inter,
                                view.dn_k_pad, inter, view.blob, view.dn_seg, view.dn_scale, rpw,
                            )?;
                        }
                    }
                }
                // [探针] `LINGXI_MIX_DUMP_DN=1`：dn 循环刚结束立刻回读 o_all（merge 之前），
                // 区分"dn gemv 本身坏" vs "后续 merge 覆写"。
                if crate::flag("LINGXI_MIX_DUMP_DN") {
                    k.stream.synchronize()?;
                    if let Ok(h) = scratch::download_bf16(k, &o_all) {
                        let nans = h.iter().filter(|v| v.is_nan()).count();
                        let tot: f64 = h.iter().filter(|v| !v.is_nan()).map(|v| v.to_f32() as f64).sum();
                        eprintln!("[mix-dbg] layer={layer} dn刚完 o_all nan={nans}/{} 有限和={tot:.4}", h.len());
                    }
                }
            } else {
            if std::env::var("LINGXI_MIX_DEBUG").is_ok() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static NE: AtomicU32 = AtomicU32::new(0);
                if NE.fetch_add(1, Ordering::Relaxed) < 60 {
                    eprintln!(
                        "[mix-dbg] layer={layer} rows=1 非dual臂 bits={} s={s} alt={}",
                        q.bits, q4_alt_hits.is_some()
                    );
                }
            }
            // ① 无重叠：整批一次（老行为，逐位基准）。
            // ② **hit/miss 重叠**（㊶，t=1 启用）：命中条目 gu→swiglu→dn 一口气推进
            //    （与 stream2 上飞的 miss H2D 并行），dn 算完再 `wait(ev_mo_miss)`，
            //    最后补未命中条目。bit-safe 论据与 slotsm 拆 launch 相同：每个对的
            //    输出区间互不相交、点积/元素级算式不变，发射边界不影响任何对的数值；
            //    swiglu 元素级按对独立，切段逐位不变。
            match q.overlap.as_ref() {
                None => {
                    if q.bits == 3 {
                        k.gemv_q3g128_slots_rows_bf16(
                            q.slots, q.slot_ids, xs, &mut gu_all, s, 2 * inter, hidden,
                            q.gu_k_pad, gu_stride, q.blob, q.gu_seg, q.gu_scale, rpw,
                        )?;
                        k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                        k.gemv_q3g128_slots_rows_bf16(
                            q.slots, q.slot_ids, &act_all, &mut o_all, s, hidden, inter,
                            q.dn_k_pad, inter, q.blob, q.dn_seg, q.dn_scale, rpw,
                        )?;
                    } else {
                    k.gemv_q4g64_slots_rows_bf16(
                        q.slots, q.slot_ids, xs, &mut gu_all, s, 2 * inter, hidden, q.gu_k_pad, gu_stride,
                        q.blob, q.gu_seg, q.gu_scale, rpw,
                    )?;
                    k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                    k.gemv_q4g64_slots_rows_bf16(
                        q.slots, q.slot_ids, &act_all, &mut o_all, s, hidden, inter, q.dn_k_pad, inter,
                        q.blob, q.dn_seg, q.dn_scale, rpw,
                    )?;
                    }
                }
                Some(ov) => {
                    let miss_runs: Vec<(u32, u32)> =
                        ov.miss_groups.iter().map(|&g| (g, g + 1)).collect();
                    // 命中段：gu→swiglu→dn 一口气推进（与 stream2 上飞的 miss H2D 并行）；
                    // 未命中段：等事件到位再同样三段。顺序内联（非闭包）避免同一缓冲的
                    // mut/immut 借用冲突；每段区间互不相交 ⇒ 逐位不变。
                    let mut segs: Vec<(u32, u32)> = ov.hit_runs.clone();
                    segs.extend(miss_runs.iter().copied());
                    for (i, &(a, b)) in segs.iter().enumerate() {
                        if i == ov.hit_runs.len() {
                            // 第一段未命中之前：等 stream2 的 blob H2D 到位
                            k.stream.wait(&k.ev_mo_miss)?;
                        }
                        let (a, b) = (a as usize, b as usize);
                        let ids = q.slot_ids.slice(a..b);
                        let xv = xs.slice(0..xs.len());
                        let mut gy = gu_all.slice_mut(a * 2 * inter..b * 2 * inter);
                        if q.bits == 3 {
                            k.gemv_q3g128_slots_rows_bf16_view(
                                q.slots, &ids, &xv, &mut gy, b - a, 2 * inter, hidden, q.gu_k_pad,
                                gu_stride, q.blob, q.gu_seg, q.gu_scale, rpw,
                            )?;
                        } else {
                        k.gemv_q4g64_slots_rows_bf16_view(
                            q.slots, &ids, &xv, &mut gy, b - a, 2 * inter, hidden, q.gu_k_pad,
                            gu_stride, q.blob, q.gu_seg, q.gu_scale, rpw,
                        )?;
                        }
                        let gs = gu_all.slice(a * 2 * inter..b * 2 * inter);
                        let mut ac = act_all.slice_mut(a * inter..b * inter);
                        k.swiglu_bf16_view(&gs, &mut ac, b - a, inter)?;
                        let ids = q.slot_ids.slice(a..b);
                        let ac = act_all.slice(a * inter..b * inter);
                        let mut oy = o_all.slice_mut(a * hidden..b * hidden);
                        if q.bits == 3 {
                            k.gemv_q3g128_slots_rows_bf16_view(
                                q.slots, &ids, &ac, &mut oy, b - a, hidden, inter, q.dn_k_pad,
                                inter, q.blob, q.dn_seg, q.dn_scale, rpw,
                            )?;
                        } else {
                        k.gemv_q4g64_slots_rows_bf16_view(
                            q.slots, &ids, &ac, &mut oy, b - a, hidden, inter, q.dn_k_pad,
                            inter, q.blob, q.dn_seg, q.dn_scale, rpw,
                        )?;
                        }
                    }
                }
            }
            }
        } else {
            let d_group = scratch::upload_i32(k, &group_off)?;
            let e_uniq = uniq_experts.len();
            let sm_rpw = slotsm_rpw();
            // P15-⑤b ①（"所有后面计算过程要转化的提前转化"）：x **提前转 fp32**。
            // gu 用 xs 的副本；dn 用 act_all（swiglu 输出）的副本，在 swiglu 后转换。
            // 逐位等价：bf16→f32 精确、GEMV 内 z 次序与累加次序不变；省每 (word,token) 8 条 CVT。
            // `LINGXI_MOE_NO_XF32=1` 回退 bf16 现转（同二进制 A/B 用）。
            // ★ P15-㉖：`LINGXI_MOE_DP4A=1` 时改走 **int8 行量化 + dp4a 内积**（x 流量 ÷4，
            // 判据见 PLAN；非位等价 ⇒ 默认关，与 xf32 互斥）。
            // ⚠️ dp4a 核（`slotsm_dp4a_body`）与修复前的 bf16 版一样只覆盖每组前 8 个对
            //   （`acc[8]` 且无分批循环，2026-10-18 审计发现）⇒ **任一组 m>8 时强制退回 bf16**，
            //   否则会静默算错。dp4a 本来就非位等价、默认关，这里只加安全网。
            // ⚠️ Q3G128 **不支持 dp4a / TC 档**（只有 Q4 布局的变体核）⇒ bits==3 恒走
            //    普通 bf16 slotsm（xf32 预转仍可用：它只是 bf16→f32 精确转换）。
            let q3_on = q.bits == 3;
            let dp4a_on = crate::flag("LINGXI_MOE_DP4A")
                && group_off.windows(2).all(|w| (w[1] - w[0]) <= 8)
                && !q3_on
                && !dual_q4;
            // ㊽ prefill 大 tile 张量核（LINGXI_MOE_TC 复用同一开关）：rows>16 的批量路径，
            // gu/dn 的 k（hidden/inter）都是 64 的倍数。有损档（~2^-11），判据 = 首 token
            // 相等 + 质量抽检（PLAN_NEXT §3 拍板）。开着时不需要 xf/dp4a 预转。
            let tcp_on = crate::flag("LINGXI_MOE_TC") && rows > 16
                && hidden % 64 == 0 && inter % 64 == 0
                && !q3_on && !dual_q4;
            let dp4a_on = dp4a_on && !tcp_on;
            let xf_on = !crate::flag("LINGXI_MOE_NO_XF32") && !dp4a_on && !tcp_on && !dual_q4;
            let mut xf_gu = scratch::f32(k, s * hidden)?;
            let mut xf_gu_p: u64 = 0;
            let mut _g_gu = None;
            if xf_on {
                k.convert_bf16_f32(xs, &mut xf_gu, s * hidden)?;
                let (p, g) = xf_gu.device_ptr(&k.stream);
                xf_gu_p = p;
                _g_gu = Some(g);
            }
            let mut xq_gu_b = if dp4a_on { Some(scratch::u8(k, s * hidden)?) } else { None };
            let mut sx_gu_b = if dp4a_on { Some(scratch::f32(k, s)?) } else { None };
            if let Some((xq, sx)) = xq_gu_b.as_mut().zip(sx_gu_b.as_mut()) {
                k.convert_bf16_q8_rs(xs, xq, sx, s, hidden)?;
            }
            let mut xq_dn_b = if dp4a_on { Some(scratch::u8(k, s * inter)?) } else { None };
            let mut sx_dn_b = if dp4a_on { Some(scratch::f32(k, s)?) } else { None };
            let mut xf_dn = scratch::f32(k, s * inter)?;
            // 注意：`device_ptr` 的 guard 不能进闭包（E0521），所以在 match 臂里内联取指针。
            // 跑一批"组区间"的 gu/dn GEMV。**位安全**：核里每组的读写在绝对对下标上
            // （`xb = x + s0*a_stride`、`y[(s0+s)*n + row]`），拆 launch 不改任何一条算式
            // 的列次序与累加次序；`moe_combine_rows` 仍按原来的对序累加 ⇒ 逐位不变。
            let run = |runs: &[(u32, u32)],
                       gu: bool,
                       gu_all: &mut CudaSlice<bf16>,
                       act_all: &CudaSlice<bf16>,
                       o_all: &mut CudaSlice<bf16>,
                       xf: u64,
                       dp: Option<(&CudaSlice<u8>, &CudaSlice<f32>)>,
                       qv: &Q4Slots|
             -> Result<()> {
                // 混档（dual_q4）：每次调用带自己的视图 ⇒ 选核常量按视图位宽重算。
                // 单格式：qv==q，shadow 后行为与以前完全一致。
                let q = qv;
                let q3_on = q.bits == 3;
                let tcp_on = tcp_on && !dual_q4;
                for &(a, b) in runs {
                    if b <= a {
                        continue;
                    }
                    let gsel = d_group.slice(a as usize..=b as usize);
                    let ssel = q.slot_ids.slice(a as usize..b as usize);
                    let cnt = (b - a) as usize;
                    if gu {
                        // ㊽ prefill 大 tile 张量核档：一次 launch 覆盖整个专家区间，
                        // M 轴在 grid.z 里分块（tmax = 区间最大组行数）。
                        if tcp_on {
                            let tmax = (a as usize..b as usize)
                                .map(|g| (group_off[g + 1] - group_off[g]) as usize)
                                .max()
                                .unwrap_or(1);
                            k.gemv_q4g64_slots_tcp_bf16_view(
                                q.slots, &ssel, &gsel, xs, gu_all, cnt, 2 * inter, hidden,
                                gu_stride, q.gu_k_pad, q.blob, q.gu_seg, q.gu_scale, tmax,
                            )?;
                        } else {
                        // ㊼ 张量核档（`LINGXI_MOE_TC=1`）：gu 的 k=hidden 是 64 的整数倍，
                        // 布局与 SIMT 完全相同（group64），只把内积搬到 m8n8k4。
                        // ⚠️ 张量核核只覆盖每组 ≤16 行（kTiles≤2）；prefill 的长 prompt 会出现
                        //    t 上百的行组 ⇒ 逐组判定，超限的组仍走 SIMT（否则会**静默丢掉行**）。
                        // ★ 2026-10-18：只在 **1 < rows ≤ 16**（投机验证批）时用 TC ——
                        //    t=1 时 M=8 tile 只用 1 行（−3%），prefill 的大 chunk 会退化成
                        //    "每组一次 launch"（几百次/层）。实测 t=7 验证轮专家段 0.59→0.55
                        //    ms/层（−7%），整轮 −5%。
                        // ⚠️ 2026-10-18 实测否决：把 TC 扩到 prefill（kTiles=4，≤32 行/组，
                        //    整份 unique-expert 列表一次 launch）反而 **23.7K 120.5 → 130.5s
                        //    （+8%）**：fp16 x 的逐次 CVT + 小行组（m<8）填充浪费吃掉了 MMA 收益。
                        //    ⇒ prefill 仍用 SIMT；TC 只服务 1<rows≤16 的验证批。
                        let tc_ok = crate::flag("LINGXI_MOE_TC")
                            && rows > 1
                            && rows <= 16
                            && hidden % 64 == 0
                            && !q3_on;
                        if tc_ok {
                            let tmax = (a as usize..b as usize)
                                .map(|g| (group_off[g + 1] - group_off[g]) as usize)
                                .max()
                                .unwrap_or(1);
                            if tmax <= 16 {
                                // 快路径：整个区间所有组都 ≤16 行 ⇒ 一次 launch 搞定（decode/投机常走这里）。
                                let tiles = if tmax > 8 { 2 } else { 1 };
                                k.gemv_q4g64_slots_tc_bf16(
                                    q.slots, &ssel, &gsel, xs, gu_all, cnt, 2 * inter, hidden,
                                    gu_stride, q.gu_k_pad, q.blob, q.gu_seg, q.gu_scale, tiles,
                                )?;
                            } else {
                                for g in a as usize..b as usize {
                                let cnt = (group_off[g + 1] - group_off[g]) as usize;
                                let gsel1 = d_group.slice(g..=g + 1);
                                let ssel1 = q.slot_ids.slice(g..g + 1);
                                if cnt <= 16 {
                                    let tiles = if cnt > 8 { 2 } else { 1 };
                                    k.gemv_q4g64_slots_tc_bf16(
                                        q.slots, &ssel1, &gsel1, xs, gu_all, 1, 2 * inter,
                                        hidden, gu_stride, q.gu_k_pad, q.blob, q.gu_seg,
                                        q.gu_scale, tiles,
                                    )?;
                                } else {
                                    match dp {
                                        Some((xq, sx)) => k.gemv_q4g64_slotsm_rpw_dp4a_view(
                                            q.slots, &ssel1, xs, &gsel1, gu_all, 1, 2 * inter,
                                            hidden, q.gu_k_pad, gu_stride, q.blob, q.gu_seg,
                                            q.gu_scale, sm_rpw, xq, sx,
                                        )?,
                                        None if q3_on => k.gemv_q3g128_slotsm_rpw_bf16_view(
                                            q.slots, &ssel1, xs, &gsel1, gu_all, 1, 2 * inter,
                                            hidden, q.gu_k_pad, gu_stride, q.blob, q.gu_seg,
                                            q.gu_scale, sm_rpw, xf,
                                        )?,
                                        None => k.gemv_q4g64_slotsm_rpw_bf16_view(
                                            q.slots, &ssel1, xs, &gsel1, gu_all, 1, 2 * inter,
                                            hidden, q.gu_k_pad, gu_stride, q.blob, q.gu_seg,
                                            q.gu_scale, sm_rpw, xf,
                                        )?,
                                    }
                                }
                            }
                            }
                        } else {
                            match dp {
                                Some((xq, sx)) => k.gemv_q4g64_slotsm_rpw_dp4a_view(
                                    q.slots, &ssel, xs, &gsel, gu_all, cnt, 2 * inter, hidden,
                                    q.gu_k_pad, gu_stride, q.blob, q.gu_seg, q.gu_scale, sm_rpw,
                                    xq, sx,
                                )?,
                                None if q3_on => k.gemv_q3g128_slotsm_rpw_bf16_view(
                                    q.slots, &ssel, xs, &gsel, gu_all, cnt, 2 * inter, hidden,
                                    q.gu_k_pad, gu_stride, q.blob, q.gu_seg, q.gu_scale, sm_rpw,
                                    xf,
                                )?,
                                None => k.gemv_q4g64_slotsm_rpw_bf16_view(
                                    q.slots, &ssel, xs, &gsel, gu_all, cnt, 2 * inter, hidden,
                                    q.gu_k_pad, gu_stride, q.blob, q.gu_seg, q.gu_scale, sm_rpw,
                                    xf,
                                )?,
                            }
                        }
                        }
                    } else {
                        // ㊽ dn 同样走大 tile 张量核（k=inter 是 64 的倍数）。
                        if tcp_on {
                            let tmax = (a as usize..b as usize)
                                .map(|g| (group_off[g + 1] - group_off[g]) as usize)
                                .max()
                                .unwrap_or(1);
                            k.gemv_q4g64_slots_tcp_bf16_view(
                                q.slots, &ssel, &gsel, act_all, o_all, cnt, hidden, inter,
                                inter, q.dn_k_pad, q.blob, q.dn_seg, q.dn_scale, tmax,
                            )?;
                        } else {
                        match dp {
                            Some((xq, sx)) => k.gemv_q4g64_slotsm_rpw_dp4a_view(
                                q.slots, &ssel, act_all, &gsel, o_all, cnt, hidden, inter,
                                q.dn_k_pad, inter, q.blob, q.dn_seg, q.dn_scale, sm_rpw, xq, sx,
                            )?,
                            None if q3_on => k.gemv_q3g128_slotsm_rpw_bf16_view(
                                q.slots, &ssel, act_all, &gsel, o_all, cnt, hidden, inter,
                                q.dn_k_pad, inter, q.blob, q.dn_seg, q.dn_scale, sm_rpw, xf,
                            )?,
                            None => k.gemv_q4g64_slotsm_rpw_bf16_view(
                                q.slots, &ssel, act_all, &gsel, o_all, cnt, hidden, inter,
                                q.dn_k_pad, inter, q.blob, q.dn_seg, q.dn_scale, sm_rpw, xf,
                            )?,
                        }
                        }
                    }
                }
                Ok(())
            };
            let dp_gu = xq_gu_b.as_ref().zip(sx_gu_b.as_ref());
            match q.overlap.as_ref() {
                None => {
                     let all = [(0u32, e_uniq as u32)];
                     if prof4 {
                         evs[1].record(&k.stream)?;
                     }
                     if dual_q4 {
                         // 混档：每格式各跑自己的组段（buffer 全局按组写，combine 不变）。
                         let mut views: Vec<(&Q4Slots, &Vec<(u32, u32)>)> = vec![(q, &runs4)];
                         if let Some(qa) = q4_alt_hits.as_ref() {
                             views.push((qa, &runs3));
                         }
                         for (qv, runs) in views {
                             if runs.is_empty() {
                                 continue;
                             }
                             run(runs, true, &mut gu_all, &act_all, &mut o_all, 0, None, qv)?;
                         }
                     } else {
                         run(&all, true, &mut gu_all, &act_all, &mut o_all, xf_gu_p, dp_gu, q)?;
                     }
                     if prof4 {
                         evs[2].record(&k.stream)?;
                     }
                    k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                    let mut xf_dn_p: u64 = 0;
                    let mut _g_dn = None;
                    if xf_on {
                        k.convert_bf16_f32(&act_all, &mut xf_dn, s * inter)?;
                        let (p, g) = xf_dn.device_ptr(&k.stream);
                        xf_dn_p = p;
                        _g_dn = Some(g);
                    }
                    if let Some((xq, sx)) = xq_dn_b.as_mut().zip(sx_dn_b.as_mut()) {
                        k.convert_bf16_q8_rs(&act_all, xq, sx, s, inter)?;
                    }
                    if prof4 {
                        evs[3].record(&k.stream)?;
                    }
                    if dual_q4 {
                         let mut views: Vec<(&Q4Slots, &Vec<(u32, u32)>)> = vec![(q, &runs4)];
                         if let Some(qa) = q4_alt_hits.as_ref() {
                             views.push((qa, &runs3));
                         }
                         for (qv, runs) in views {
                            if runs.is_empty() {
                                continue;
                            }
                            run(runs, false, &mut gu_all, &act_all, &mut o_all, 0, None, qv)?;
                            // [探针] 每个视图 dn 跑完立刻数 NaN —— 定位坏在哪个格式视图。
                            if crate::flag("LINGXI_MIX_DUMP_DN") {
                                k.stream.synchronize()?;
                                if let Ok(h) = scratch::download_bf16(k, &o_all) {
                                    let nans = h.iter().filter(|v| v.is_nan()).count();
                                    eprintln!("[mix-dbg] 层{layer} rows={rows} dn后 view bits={} nan={nans}/{}", qv.bits, h.len());
                                }
                            }
                        }
                     } else {
                         run(&all, false, &mut gu_all, &act_all, &mut o_all, xf_dn_p,
                             xq_dn_b.as_ref().zip(sx_dn_b.as_ref()), q)?;
                     }
                    if prof4 {
                        evs[4].record(&k.stream)?;
                    }
                }
                Some(ov) => {
                    // ① 命中组先算：此时 miss 的 blob H2D 正在**第二流**上飞（两边并行）
                    run(&ov.hit_runs, true, &mut gu_all, &act_all, &mut o_all, xf_gu_p, dp_gu, q)?;
                    // ② 等未命中组的 blob 到位，再算它们（miss 组通常只有 2~3 个 ⇒ 一次 launch/组）
                    k.stream.wait(&k.ev_mo_miss)?;
                    let miss_runs: Vec<(u32, u32)> =
                        ov.miss_groups.iter().map(|&g| (g, g + 1)).collect();
                    run(&miss_runs, true, &mut gu_all, &act_all, &mut o_all, xf_gu_p, dp_gu, q)?;
                    k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                    let mut xf_dn_p: u64 = 0;
                    let mut _g_dn = None;
                    if xf_on {
                        k.convert_bf16_f32(&act_all, &mut xf_dn, s * inter)?;
                        let (p, g) = xf_dn.device_ptr(&k.stream);
                        xf_dn_p = p;
                        _g_dn = Some(g);
                    }
                    if let Some((xq, sx)) = xq_dn_b.as_mut().zip(sx_dn_b.as_mut()) {
                        k.convert_bf16_q8_rs(&act_all, xq, sx, s, inter)?;
                    }
                    let dp_dn = xq_dn_b.as_ref().zip(sx_dn_b.as_ref());
                    run(&ov.hit_runs, false, &mut gu_all, &act_all, &mut o_all, xf_dn_p, dp_dn, q)?;
                    run(&miss_runs, false, &mut gu_all, &act_all, &mut o_all, xf_dn_p, dp_dn, q)?;
                }
            }
            scratch::put_i32(d_group);
            // ★ 顺手修池对称性：xf_gu/xf_dn 此前用完直接 drop（每 call 白吃两块 + 池地址漂移）。
            // 先 drop 指针守卫（借用着 xf_gu；xf_dn 的守卫在 match 臂内已随作用域结束），再归还池。
            drop(_g_gu);
            scratch::put_f32(xf_gu);
            scratch::put_f32(xf_dn);
            if let Some((xq, sx)) = xq_gu_b.take().zip(sx_gu_b.take()) {
                scratch::put_u8(xq);
                scratch::put_f32(sx);
            }
            if let Some((xq, sx)) = xq_dn_b.take().zip(sx_dn_b.take()) {
                scratch::put_u8(xq);
                scratch::put_f32(sx);
            }
        }
        // [探针] 按组（唯一专家）核对专家 GEMV 的输出：与 gather_x / 权重一起构成完整判据。
        if crate::flag("LINGXI_MOE_OUT_DUMP") && {
            let want: u32 = std::env::var("LINGXI_MOE_OUT_DUMP_LAYER")
                .ok().and_then(|v| v.parse().ok())
                .map(|v: u32| v.wrapping_sub(1))
                .unwrap_or(crate::DIMS.n_layers as u32 - 1);
            layer == want
        } {
            for (nm, buf) in [("gu_all", &gu_all), ("act_all", &act_all), ("o_all", &o_all)] {
                if let Ok(h) = scratch::download_bf16(k, buf) {
                    let tot: f64 = h.iter().map(|v| v.to_f32() as f64).sum();
                    let nans = h.iter().filter(|v| v.is_nan()).count();
                    let first = h.iter().position(|v| v.is_nan());
                    eprintln!(
                        "[moe-out] 层{layer} {nm} 全和={tot:.6} n={} nan={nans} first_nan={first:?}",
                        h.len()
                    );
                }
            }
            if let Ok(h) = scratch::download_bf16(k, &o_all) {
                let tot: f64 = h.iter().map(|v| v.to_f32() as f64).sum();
                eprintln!("[moe-out] 层{layer} o_all 全和={tot:.6} n={}", h.len());
                // 混档诊断：逐组 NaN 计数（揪出哪些专家的 dn 输出是 NaN）。
                for g in 0..uniq_experts.len() {
                    let a = group_off[g] as usize;
                    let b = group_off[g + 1] as usize;
                    let nan_n = h[a * hidden..b * hidden].iter().filter(|v| v.is_nan()).count();
                    if nan_n > 0 {
                        eprintln!(
                            "[moe-out] 层{layer} **NaN 组** g={g} expert={} m={} nan={}/{}",
                            uniq_experts[g], b - a, nan_n, (b - a) * hidden
                        );
                    }
                }
                for g in 0..3.min(uniq_experts.len()) {
                    let a = group_off[g] as usize;
                    let b = group_off[g + 1] as usize;
                    let (mut s0, mut mx) = (0f64, 0f32);
                    for v in &h[a * hidden..b * hidden] {
                        let f = v.to_f32();
                        s0 += f as f64;
                        if f.abs() > mx {
                            mx = f.abs();
                        }
                    }
                    eprintln!(
                        "[moe-out] 层{layer} 组{g} expert={} m={} o_sum={s0:.6} maxabs={mx:.4}",
                        uniq_experts[g], b - a
                    );
                }
            }
        }
        if kts_m {
            k.stream.synchronize()?;
            let t_m1 = std::time::Instant::now();
            k.moe_combine_rows_bf16(&o_all, &d_rows, &d_wts, &mut expert_out, s, rows, hidden)?;
            k.stream.synchronize()?;
            use std::sync::atomic::{AtomicU32, Ordering};
            static M_N: AtomicU32 = AtomicU32::new(0);
            if M_N.fetch_add(1, Ordering::Relaxed) < 400 {
                eprintln!(
                    "[kts-moe] rows={} s={} 前置(upload/gather) {:.1} | 专家核+combine {:.1} µs",
                    rows, s,
                    (t_m1 - t_m0).as_secs_f64() * 1e6,
                    t_m1.elapsed().as_secs_f64() * 1e6,
                );
            }
        } else {
            // ㊶：诊断关时 combine+残差一发融合（两次舍入同序 ⇒ 逐位相同），省一次 launch/层。
            if dbg_routes {
                k.moe_combine_rows_bf16(&o_all, &d_rows, &d_wts, &mut expert_out, s, rows, hidden)?;
            } else if rows == 1 {
                // S 小，旧核 O(H×S) 扫描即可。
                let mut fo = scratch::bf16(k, rows * hidden)?;
                k.moe_combine_add_rows_bf16(
                    &o_all, &d_rows, &d_wts, &shared, &mut fo, s, rows, hidden,
                )?;
                out_fused = Some(fo);
            } else {
                // ㊾ batch 路径走 CSR combine（O(H×top_k)/行）：rows=887 实测 19ms → <1ms。
                let mut fo = scratch::bf16(k, rows * hidden)?;
                k.moe_combine_add_csr_bf16(
                    &o_all, &d_col, &d_roff, &d_wts, &shared, &mut fo, rows, hidden,
                )?;
                out_fused = Some(fo);
            }
        }
        if prof4 && evs.len() == 6 {
            evs[5].record(&k.stream)?;
            if rows > 1 {
                let ms = |i: usize| evs[i].elapsed_ms(&evs[i + 1]).unwrap_or(0.0);
                eprintln!(
                    "[moe-prof4] 层{layer} rows={rows} s={s} 前置 {:.2} | gu {:.2} | \
                     swiglu+xf {:.2} | dn {:.2} | combine+收尾 {:.2} ms",
                    ms(0), ms(1), ms(2), ms(3), ms(4),
                );
            }
        }

        // [探针] `LINGXI_MOE_DUMP`（目录）+ `LINGXI_MOE_DUMP_ACT=1`：dump 每对专家输入
        // （gu: 按行 gather 的 x，dn: swiglu 中间激活）+ 对→(专家,行) 名单——离线累加
        // H = X^T X 做 GPTQ 校准（6.2-3）。只 dump rows>1（prefill 批；decode 对数太少）。
        // 注意：同步 D2H，校准跑别开 CUDA Graph 捕获本段（prefill pair 路径本就不捕获）。
        if dump_dir.is_some() && rows > 1 && crate::flag("LINGXI_MOE_DUMP_ACT") {
            let d = dump_dir.as_ref().unwrap();
            let act_h = scratch::download_bf16(k, &act_all)?;
            let x_h = scratch::download_bf16(k, xs)?;
            let pe: Vec<u8> = pair_experts.iter().flat_map(|v| v.to_le_bytes()).collect();
            let pr: Vec<u8> = pair_rows.iter().flat_map(|v| v.to_le_bytes()).collect();
            let xh: Vec<u8> = x_h.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            let ah: Vec<u8> = act_h.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            std::fs::write(d.join(format!("moe{call_id:03}_xpair.bf16")), xh)?;
            std::fs::write(d.join(format!("moe{call_id:03}_act.bf16")), ah)?;
            std::fs::write(d.join(format!("moe{call_id:03}_pairs.pe.u32")), pe)?;
            std::fs::write(d.join(format!("moe{call_id:03}_pairs.pr.i32")), pr)?;
            std::fs::write(
                d.join(format!("moe{call_id:03}_meta.json")),
                format!(
                    r#"{{"layer":{layer},"rows":{rows},"s":{s},"topk":{top_k},"hidden":{hidden},"inter":{inter}}}"#
                ),
            )?;
        }

        scratch::put_bf16(gu_all);
        scratch::put_bf16(act_all);
        scratch::put_bf16(o_all);
        scratch::put_i32(d_rows);
        scratch::put_f32(d_wts);
        scratch::put_i32(d_col);
        scratch::put_i32(d_roff);
        if let Some(g) = xg {
            scratch::put_bf16(g);
        }
    } else if src.is_some() && rows == 1 && !hit_ids.is_empty() {
        let s = hit_ids.len();
        for &e in hit_ids.iter() {
            let (r, wgt) = per_expert[e as usize][0];
            rows_of.push(r);
            wts_1.push(wgt);
        }
        // 走**锁页**中转：可分页源会让 `cuMemcpyHtoDAsync` 退化成驱动 staging 的同步拷贝。
        let d_rows = scratch::upload_i32(k, rows_of)?;
        let d_wts = scratch::upload_f32(k, wts_1)?;

        // ★ P15-⑩（2026-10-17）：图内**所有**缓冲改走 `take_fixed` 按层固定槽。
        // 根因：spec 的 t=7 验证批会打乱 scratch 池借还序列，闭包外 scratch take 的
        // x/gu/act/o 地址逐轮漂移 ⇒ 图（烘死地址）读到旧块 = **静默数值错**（实测
        // spec+graph 接受率 4.14→0.33，不 NaN）。手法与 gr.rs 相同：输入 stage 进固定槽
        // （图外拷入）、中间缓冲 take/give_fixed、combine 移出图（dst 是 scratch）。
        let gkey = (layer as u64) * 8 + 7;
        let mut xs = crate::graph::take_fixed(k, gkey, 4, rows * hidden)?;
        k.stream.memcpy_dtod(x, &mut xs)?;
        crate::graph::give_fixed(k, gkey, 4, xs);
        // slot_ids 也 stage：arena 的 slot_ids 是**双缓冲**（ids_phase 层间交替），
        // t=7 验证批的 load_hits 会把层 l 的 slot_ids 翻到另一块 ⇒ 图录死的地址读到别层内容。
        if let Some(q) = q4_hits.as_ref() {
            let mut ids_fix = crate::graph::take_fixed_i32(k, gkey, 5, s)?;
            k.stream.memcpy_dtod(q.slot_ids, &mut ids_fix)?;
            crate::graph::give_fixed_i32(k, gkey, 5, ids_fix);
        }
        let body = || -> Result<()> {
            let mut gu_all = crate::graph::take_fixed(k, gkey, 0, s * 2 * inter)?;
            let mut act_all = crate::graph::take_fixed(k, gkey, 1, s * inter)?;
            let mut o_all = crate::graph::take_fixed(k, gkey, 2, s * hidden)?;
            let xin = crate::graph::take_fixed(k, gkey, 4, rows * hidden)?;
            let qin = crate::graph::take_fixed_i32(k, gkey, 5, s)?;
            if let Some(q) = q4_hits.as_ref() {
                // **直读设备槽里的 Q4/Q3**：融合反量化，省掉每层 98MB 的 bf16 中转与 2 次低效解量化。
                let rpw = slots_rpw();
                if q.bits == 3 {
                    k.gemv_q3g128_slots_rows_bf16(
                        q.slots, &qin, &xin, &mut gu_all, s, 2 * inter, hidden, q.gu_k_pad, 0,
                        q.blob, q.gu_seg, q.gu_scale, rpw,
                    )?;
                    k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                    k.gemv_q3g128_slots_rows_bf16(
                        q.slots, &qin, &act_all, &mut o_all, s, hidden, inter, q.dn_k_pad, inter,
                        q.blob, q.dn_seg, q.dn_scale, rpw,
                    )?;
                } else {
                k.gemv_q4g64_slots_rows_bf16(
                    q.slots, &qin, &xin, &mut gu_all, s, 2 * inter, hidden, q.gu_k_pad, 0,
                    q.blob, q.gu_seg, q.gu_scale, rpw,
                )?;
                k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                k.gemv_q4g64_slots_rows_bf16(
                    q.slots, &qin, &act_all, &mut o_all, s, hidden, inter, q.dn_k_pad, inter,
                    q.blob, q.dn_seg, q.dn_scale, rpw,
                )?;
                }
            } else {
                let gu_w = compact_gu.expect("batched 路径需要紧凑专家权重");
                let dn_w = compact_dn.expect("batched 路径需要紧凑专家权重");
                k.gemm_bf16_nt_batched(
                    &xin, gu_w, &mut gu_all, 1, (2 * inter) as u32, hidden as u32, s as u32,
                    0, (2 * inter * hidden) as i64, (2 * inter) as i64,
                )?;
                k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                k.gemm_bf16_nt_batched(
                    &act_all, dn_w, &mut o_all, 1, hidden as u32, inter as u32, s as u32,
                    inter as i64, (hidden * inter) as i64, hidden as i64,
                )?;
            }
            crate::graph::give_fixed(k, gkey, 0, gu_all);
            crate::graph::give_fixed(k, gkey, 1, act_all);
            crate::graph::give_fixed(k, gkey, 2, o_all);
            crate::graph::give_fixed(k, gkey, 4, xin);
            crate::graph::give_fixed_i32(k, gkey, 5, qin);
            Ok(())
        };
        // `LINGXI_NO_MOE_GRAPH=1`：只禁 rows==1 专家段图（二分"spec+graph 数值错"的罪魁用）。
        if crate::graph::enabled() && !crate::flag("LINGXI_NO_MOE_GRAPH") {
            crate::graph::run_cached(k, gkey, body)?;
        } else {
            body()?;
        }
        // combine 图外：o_all 从固定槽取出（重放时它正是图写的块），dst 是 scratch expert_out。
        // ㊶：非诊断时融合核直写 out（combine 的 round + add 的 round 同序，逐位相同）。
        let o_fix = crate::graph::take_fixed(k, gkey, 2, s * hidden)?;
        if dbg_routes {
            k.moe_combine_rows_bf16(&o_fix, &d_rows, &d_wts, &mut expert_out, s, rows, hidden)?;
        } else {
            let mut fo = scratch::bf16(k, rows * hidden)?;
            k.moe_combine_add_rows_bf16(
                &o_fix, &d_rows, &d_wts, &shared, &mut fo, s, rows, hidden,
            )?;
            out_fused = Some(fo);
        }
        crate::graph::give_fixed(k, gkey, 2, o_fix);
        scratch::put_i32(d_rows);
        scratch::put_f32(d_wts);
    } else {

    // ── 每专家 scratch 与索引：**每次调用只分配一次**，逐专家切片 ──
    // 原实现在专家循环内 `alloc_zeros`（每 token 上千次 cuMalloc/cuMemFree + clone_htod
    // 反复换新），实测是 decode 的主要开销。这里按 `rows` 上限一次性分配、循环复用；
    // kernel 只读写前 m 行，传 m 即可。
    let mut xg = unsafe { k.stream.alloc::<bf16>(rows * hidden) }?;
    let mut gu_m = unsafe { k.stream.alloc::<bf16>(rows * 2 * inter) }?;
    let mut act_m = unsafe { k.stream.alloc::<bf16>(rows * inter) }?;
    let mut o_m = unsafe { k.stream.alloc::<bf16>(rows * hidden) }?;
    // 整层命中的 `(行, 权重)` 打包成两块、**一次** H2D；逐专家只传起始偏移。
    // （原来每专家两次 `clone_htod` = 每 token 近千次 cuMalloc，随显存占用上升明显变慢。）
    let mut all_idx: Vec<i32> = Vec::with_capacity(rows * top_k + 1);
    let mut all_wt: Vec<f32> = Vec::with_capacity(rows * top_k + 1);
    let mut off: Vec<usize> = vec![0; num_experts];
    for (e, hits) in per_expert.iter().enumerate() {
        off[e] = all_idx.len();
        for (t, wgt) in hits {
            all_idx.push(*t);
            all_wt.push(*wgt);
        }
    }
    let didx_all = k.stream.clone_htod(&all_idx)?;
    let dwt_all = k.stream.clone_htod(&all_wt)?;

    for (e, hits) in per_expert.iter().enumerate() {
        if hits.is_empty() {
            continue;
        }
        let m = hits.len();
        let o = off[e];

        k.gather_rows_bf16(x, &didx_all, &mut xg, m, hidden, o)?;

        // `slice` 收**元素**下标；专家 e 的行号是 e*2I，故元素偏移 = e*2I*H。
        // 瘦身路径下全量张量不存在，用槽位偏移切紧凑张量。
        let (gu_base, dn_base) = if src.is_some() {
            (slot_of[e], slot_of[e])
        } else {
            (e, e)
        };
        let gu_w = w.routed_gate_up.as_ref().or(compact_gu).expect(
            "routed_gate_up 缺失：全量权重与 ExpertSource 至少要有一个",
        );
        let dn_w = w.routed_down.as_ref().or(compact_dn).expect(
            "routed_down 缺失：全量权重与 ExpertSource 至少要有一个",
        );
        let gu_e = gu_w.slice(
            gu_base * 2 * inter * hidden..(gu_base + 1) * 2 * inter * hidden,
        );
        proj_t1(k, &xg, &gu_e, &mut gu_m, 2 * inter, hidden, m)?;

        k.swiglu_bf16(&gu_m, &mut act_m, m, inter)?;

        let down_e = dn_w.slice(dn_base * hidden * inter..(dn_base + 1) * hidden * inter);
        proj_t1(k, &act_m, &down_e, &mut o_m, hidden, inter, m)?;

        if let Some(dir) = debug_dir {
            let dump = |tag: String, v: &[f32]| -> Result<()> {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("dbg_{tag}.f32")), bytes)?;
                Ok(())
            };
            dump(format!("e{e}_idx"), &all_idx[o..o + m].iter().map(|v| *v as f32).collect::<Vec<_>>())?;
            dump(format!("e{e}_w"), &all_wt[o..o + m])?;
            let gu_h: Vec<f32> = k.stream.clone_dtoh(&gu_m)?[..m * 2 * inter]
                .iter().map(|v| v.to_f32()).collect();
            let act_h: Vec<f32> = k.stream.clone_dtoh(&act_m)?[..m * inter]
                .iter().map(|v| v.to_f32()).collect();
            let om_h: Vec<f32> = k.stream.clone_dtoh(&o_m)?[..m * hidden]
                .iter().map(|v| v.to_f32()).collect();
            dump(format!("e{e}_gu"), &gu_h)?;
            dump(format!("e{e}_act"), &act_h)?;
            dump(format!("e{e}_om"), &om_h)?;
        }

        k.scatter_add_rows_bf16(&mut expert_out, &didx_all, &dwt_all, &o_m, m, hidden, o)?;
    }
    }

    // `LINGXI_EVPROF=1`：段 4（专家段）到此为止。
    if let Some(p) = k.evprof.as_ref() {
        p.end(layer, 4, &k.stream);
    }
    // ───── out = expert_out + shared ─────
    // ㊶：两条快路径已用融合核直写 out；其余路径（逐专家 scatter 等）照旧两步。
    let out = match out_fused {
        Some(fo) => fo,
        None => {
            let mut o = scratch::bf16(k, rows * hidden)?;
            k.add_bf16(&expert_out, &shared, &mut o, rows * hidden)?;
            o
        }
    };
    if prof {
        if crate::flag("LINGXI_KTS") {
            k.stream.synchronize()?;
        }
        let t_all = t_start.elapsed();
        eprintln!(
            "    [moe] 层 {layer:2} shared {:.1} router {:.1} (gemm {:.1} D2H {:.1} topk {:.1}) 专家 {:.1} 合计 {:.1} µs | enq {:.1} up {:.1}",
            t_shared.as_secs_f64() * 1e6,
            (t_router - t_shared).as_secs_f64() * 1e6,
            (t_rd - t_rg).as_secs_f64() * 1e6,
            (t_rt - t_rd).as_secs_f64() * 1e6,
            (t_rk - t_rt).as_secs_f64() * 1e6,
            (t_all - t_router).as_secs_f64() * 1e6,
            t_all.as_secs_f64() * 1e6,
            t_enq.elapsed().as_secs_f64() * 1e6,
            t_up_done,
        );
    }
    // 诊断字段（component-check 的 `LINGXI_MOE_DEBUG` / `LINGXI_MOE_DUMP` 用）：
    // 生产路径不带出（`Vec::new` 不分配），需要时才 clone（调试无热点可言）。
    // （`dbg_routes` 在专家段入口已算好。）
    let per_expert_out = if dbg_routes { per_expert.clone() } else { Vec::new() };
    let sel_rows_out = if dbg_routes { std::mem::take(sel_rows) } else { Vec::new() };
    // P15-⑰：预做的 uploads/gather 若最终没走 pair_path（q4_hits=None 的回退），在这里归还池。
    if let Some((d_rows, d_wts, d_col, d_roff, xg)) = pre {
        scratch::put_i32(d_rows);
        scratch::put_f32(d_wts);
        scratch::put_i32(d_col);
        scratch::put_i32(d_roff);
        if let Some(g) = xg {
            scratch::put_bf16(g);
        }
    }
    ROUTE_WS.with(|c| {
        c.replace(ws);
    });
    // [探针] `LINGXI_MOE_OUT_DUMP=1`：只打**最后一层的 MoE 输出/共享专家**指纹，
    // 用于定位"同输入同路由、输出却不同"（线程/分层/命中态相关）的差异落在哪一段。
    if crate::flag("LINGXI_MOE_OUT_DUMP") && {
            let want: u32 = std::env::var("LINGXI_MOE_OUT_DUMP_LAYER")
                .ok().and_then(|v| v.parse().ok())
                .map(|v: u32| v.wrapping_sub(1))
                .unwrap_or(crate::DIMS.n_layers as u32 - 1);
            layer == want
        } {
        let fsum = |v: &[half::bf16]| -> (f64, f32) {
            let mut s = 0f64;
            let mut mx = 0f32;
            for x in v {
                let f = x.to_f32();
                s += f as f64;
                if f.abs() > mx {
                    mx = f.abs();
                }
            }
            (s, mx)
        };
        if let Ok(h) = scratch::download_bf16(k, &out) {
            let (s, mx) = fsum(&h);
            eprintln!("[moe-out] 层{layer} rows={rows} s={} out_sum={s:.6} maxabs={mx:.4}", pair_experts.len());
        }
        if let Ok(h) = scratch::download_bf16(k, &shared) {
            let (s, mx) = fsum(&h);
            eprintln!("[moe-out] 层{layer} shared_sum={s:.6} maxabs={mx:.4}");
        }
    }
    Ok(MoeOutput { out, router_logits: logits_dev, shared, expert_out, per_expert: per_expert_out, sel_rows: sel_rows_out })
}
