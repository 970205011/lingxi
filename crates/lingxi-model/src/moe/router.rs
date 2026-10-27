//! 路由段：A2 映射 host 内存 / 自旋等待 / 路由核 launch（`route_*`）。
//! CODE_MAP Phase 3 刀P3-1 从 moe.rs 搬入，只搬家不改行为。

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

use crate::scratch;

use super::{slots_rpw, ExpertSource, MoeWeights};

/// A2 路由核的三块**映射 host 内存**（thread_local 按上下文隔离，同 `MAP` 的坑与解法）：
/// 未命中计数、未命中专家 id（紧凑）、每个选中专家的槽号（-1=miss）。
pub(crate) struct RouteMapped {
    flag_dev: u64,
    flag_host: *mut u32,
    miss_ids_dev: u64,
    miss_ids_host: *mut i32,
    slots_dev: u64,
    slots_host: *mut i32,
}

pub(crate) fn route_miss_mapped(k: &Kernels, top_k: usize) -> Result<RouteMapped> {
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
/// `stream.synchronize` 省 ~5-15µs/次的唤醒延迟 ×48 层）。事件按上下文缓存。
pub(crate) fn route_spin(k: &Kernels, timed: bool) -> Result<Option<f32>> {
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
pub(crate) fn route_mark(k: &Kernels) -> Result<()> {
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
pub(crate) fn route_dev_attempt(
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
    // t=1 设备路由尚未接混档双视图（`alt`）——混档仍走 legacy host 路径；
    // 混档的设备快路径在 `batch_dev`（rows>1）。
    if dv.alt.is_some() {
        return Ok(None);
    }
    let tw0 = std::time::Instant::now();
    let qc = dv.consts;
    let s = top_k;
    let gkey = (layer as u64) * 8 + 7;
    let rkey = (layer as u64) * 8 + 6;
    // 路由核输出固定槽：logits(rkey,0) / ids(rkey,1) / wts(rkey,2) / rows(rkey,4) /
    // slot_ids 直写图体输入槽 (gkey,5)（省一次 D2D staging）。
    //
    // ⚠️ router 投影必须用 **legacy rows==1 同一条核**（`gemv_bf16_ptr`，不是 `proj_t1`
    // 的 row8 分派）：两核累加次序不同，换核就不再逐位等价。
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
        src.load_missing(layer, &miss_list, &prot, &[])?;
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

