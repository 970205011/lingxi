//! MoE 主前向 `sparse_moe_forward`（含 slots_rpw/slotsm_rpw 形状助手）。
//! CODE_MAP Phase 3 刀P3-1 从 moe.rs 搬入，只搬家不改行为。

use std::sync::atomic::Ordering;

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

use crate::gpu::{proj_t1, proj_t1_epi};
use crate::scratch;

use super::{
    moe_overlap_on, route_dev_attempt, ExpertSource, MoeOutput, MoeWeights, Q4Slots,
    RemoteMoe, RouteWs, ROUTE_WS, SYNC_WAIT_NS, ROUTER_ALLOC_NS, ROUTER_LAUNCH_NS,
};

/// 专家 GEMV 的**每 warp 行数**（rows-per-warp）。
///
/// 默认 1 = 逐行版，与历史逐位一致（`gemv_q4g64_slots_rows_bf16(.., 1)` 就是它）。
/// `LINGXI_Q4_SLOTS_RPW=2/4/8` 切到多行版：`dn` 段 `k=640`（每行只有 320 B 的 Q4），
/// 逐行版会把全局 warp 数撑到 `n×S = 25600` 个（每 warp 只搬 320 B），线程太多、每行太少；
/// 多行版让一个 warp 顺序扫多行，减少线程数并把跨行访存并行起来。
pub(crate) fn slots_rpw() -> usize {
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
pub(crate) fn slotsm_rpw() -> usize {
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

// 稀疏专家**按需取权重**源（P1 瘦身）：路由算完后，只把命中专家的字节搬到显存。
//
// model 不做 IO——本 trait 由装配层（bench/server）实现：mmap 切片、页缓存、
// 显存槽缓存等策略全在实现侧，model 只在路由结果出来后调用一次。
// 数学约束：返回的每个专家字节必须与全量张量中对应专家段**逐位相同**，
// 槽位顺序见 [`ExpertSource::load_hits`]。

thread_local! {
    /// `LINGXI_FULLDEV_SHADOW=1` 对拍的重入保护：嵌套递归那一趟**必须走 legacy**。
    static SHADOW_NESTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[allow(clippy::too_many_arguments)]
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
    // EP 分片（`LINGXI_EP=1`）：只算 `assign[e] == card` 的对；`out` 是本卡那半边的部分和
    // （card 0 另含 shared，card 1 的 shared 恒为零）。路由全量照旧，两卡逐位一致。
    ep: Option<super::EpSlice<'_>>,
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
            // 池键自 P15-㉞ 起为 (设备号, 长度)（scratch.rs 头部说明），两张卡各用各的
            // 桶 ⇒ 这里**池化**分配是跨卡安全的（旧注释"键无设备"已过时；without_pool
            // 会让每次远程调用都退化成 cudaMalloc/free 风暴）。用完显式 put 回池复用。
            let mut x1 = crate::scratch::bf16(rm.k, x.len())?;
            let mut out = crate::scratch::bf16(k, rows * hidden)?;
            let rt0 = std::time::Instant::now();
            // 输入激活 → GPU1（走**两张卡各自的锁页 staging**，不再是可分页 Vec）
            crate::scratch::cross_copy(k, x, rm.k, &mut x1)?;
            let rt1 = std::time::Instant::now();
            chk("激活上传")?;
            // 在 GPU1 上原样重跑（权重/专家源都换成 GPU1 的；`remote=None` 防递归）
            let mo1 = sparse_moe_forward(
                rm.k, w1, &x1, rows, hidden, num_experts, top_k, inter, debug_dir, layer,
                Some(rm.experts), None, None,
            )?;
            crate::scratch::put_bf16(x1);
            let rt2 = std::time::Instant::now();
            chk("MoE 计算")?;
            // 输出 → GPU0
            crate::scratch::cross_copy(rm.k, &mo1.out, k, &mut out)?;
            crate::scratch::put_bf16(mo1.out);
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
    super::s3::mark(k, layer, 0)?;
    let prof = crate::flag("LINGXI_PROF");
    let t_start = std::time::Instant::now();
    let mut shared = scratch::bf16(k, rows * hidden)?;
    // ★ EP：shared expert 只在 **card 0** 上算；另一卡输出零（本卡半边部分和语义）。
    // 合并时两卡都做 out0 + out1（同序），结果 = shared + 全量专家。
    let shared_on = ep.map_or(true, |e| e.card == 0);
    if shared_on {
    let sg_row = w.router_shared_gate.slice(num_experts * hidden..(num_experts + 1) * hidden);
    let mut act = scratch::bf16(k, rows * inter)?;
    let mut sg = scratch::bf16(k, rows)?;
    // ★ 2026-10-24 shared expert **融合核**（默认开，`LINGXI_NO_SHARED_FUSE=1` 回退）。
    // 原路径 5 发：gu GEMM / swiglu / dn / sg / row_scale —— 后三发的核本身只有 5~60 KB
    // 的活，每发 ~7 µs 全是 launch 地板（微基准 `t8_moe_shared_bw`：m=6 链 71.9 µs，
    // 其中 gu 28.2 + dn 25.0，swiglu/sg/row_scale 合计 23.1）。
    // 融合后 2 发：① gate 行 i 与 up 行 i+I 同 block 算 + swiglu + shared_gate 行；
    // ② dn + row_scale 尾。中间张量（act_full / sg 的读写）不再各自走一次 HBM 往返。
    // **逐位等价**：两核的线程→列映射、累加次序、两级归约树与
    // `gemm_skinny_row_bf16`/`gemm_skinny_m8` 逐字相同，swiglu/scale 的舍入点显式保留
    // （见 `moe_ops.cu` 两核头注）。
    // 只对 `2 ≤ rows ≤ 64` 生效：
    //  * rows == 1 走 GEMV（累加次序不同，必须原样）；
    //  * rows > 64 时旧路会切到 `gemm_bf16_mlarge`（有损档、k 升序串行累加，见 `proj_t1`），
    //    与这里的 skinny 归约次序不同 ⇒ 为了**任何输入都逐位不变**，那一档保持旧路。
    // `hidden < 1024 / hidden % 8 != 0 / LINGXI_NO_SKINNY_ROW` 时旧路也不走 skinny_row；
    // `LINGXI_F32X_PROJ` 是既有诊断档（它会改走 f32x 核），让它优先、融合让位。
    let fuse_shared = (2..=64).contains(&rows)
        && hidden >= 1024
        && hidden % 8 == 0
        && !crate::flag("LINGXI_NO_SKINNY_ROW")
        && !crate::flag("LINGXI_F32X_PROJ")
        && !crate::flag("LINGXI_NO_SHARED_FUSE");
    if let (Some(gu_q4), Some(dn_q4)) = (&w.shared_gate_up_q4, &w.shared_down_q4) {
        // ★ W4 共享专家档（2026-10-29，LINGXI_SHARED_Q4）：Q4 侧没有"gu+swiglu 单核"
        // 融合（那是 bf16 skinny 核族），拆三发走通用 dequant GEMV（与注意力 W4 同核
        // 族 proj_q4_epi）：gu → swiglu → dn；sg（shared gate 行）照旧 bf16。
        // 无损判据不适用（有损 W4 注定不逐位）——验收走质量判据（长文档 QA/needle）
        // + GPTQ 校准产物（gptq_shared.py）。
        let mut act_full = scratch::bf16(k, rows * 2 * inter)?;
        crate::gpu::proj_q4_epi(k, x, gu_q4, &mut act_full, 2 * inter, hidden, 0, rows, 0, 0.0, 0.0)?;
        k.swiglu_bf16(&act_full, &mut act, rows, inter)?;
        scratch::put_bf16(act_full);
        crate::gpu::proj_q4_epi(k, &act, dn_q4, &mut shared, hidden, inter, 0, rows, 0, 0.0, 0.0)?;
        proj_t1_epi(k, x, &sg_row, &mut sg, 1, hidden, rows, 2, 1.0, 1.0)?;
        scratch::put_bf16(act);
        k.row_scale_inplace_bf16(&mut shared, &sg, rows, hidden)?;
        scratch::put_bf16(sg);
    } else if fuse_shared {
        {
            let (wp, _a) = w.shared_gate_up.device_ptr(&k.stream);
            let (sp, _b) = sg_row.device_ptr(&k.stream);
            let (xp, _c) = x.device_ptr(&k.stream);
            let (ap, _d) = act.device_ptr_mut(&k.stream);
            let (gp, _e) = sg.device_ptr_mut(&k.stream);
            k.gemm_skinny_row_gu_swiglu_sg_bf16(
                wp, sp, xp, ap, gp, inter, hidden, rows, 1.0, 1.0,
            )?;
        }
        {
            let (wp, _a) = w.shared_down.device_ptr(&k.stream);
            let (ap, _b) = act.device_ptr(&k.stream);
            let (yp, _c) = shared.device_ptr_mut(&k.stream);
            let (gp, _d) = sg.device_ptr(&k.stream);
            k.gemm_skinny_m8_scale_bf16(wp, ap, yp, gp, hidden, inter, rows)?;
        }
        scratch::put_bf16(act);
        scratch::put_bf16(sg);
    } else {
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
    }
    } else {
        k.stream.memset_zeros(&mut shared)?;
    }
    // `LINGXI_KTS=1`：给 `[moe]` 的三段插同步括号，让它们变成**真 GPU 时间**（否则只是 CPU 发射时间）。
    if crate::flag("LINGXI_KTS") {
        k.stream.synchronize()?;
    }
    let t_shared = t_start.elapsed();
    super::s3::mark(k, layer, 1)?;

    // ㊺ A2 设备路由：rows==1 时把「router GEMV → 同步 → host softmax/topk → host 查槽 →
    // staging」整条 host 决策链压成一个 GPU 核 + 一次事件自旋。命中全中 → 直接出结果；
    // 任一选中专家未入槽（miss）→ None，落回 legacy 全路径（本函数写过的设备缓冲无副作用）。
    // `LINGXI_ROUTE_DEV=1` 开启；判据未过前默认关。
    let route_dev_on = rows == 1
        && crate::flag("LINGXI_ROUTE_DEV")
        && !crate::flag("LINGXI_MOE_DEBUG")
        && !crate::flag("LINGXI_MOE_DUMP")
        // EP：设备路由在**核内**就完成专家查表/取权重，没给 host 留过滤专家表的缝；
        // 而且它的输出直接含 shared（本卡半边语义不成立）。EP 下强制走 legacy host 路，
        // 两卡同核同数据 ⇒ 路由结果逐位相同（这是"两卡选中对一致"的前置条件）。
        && ep.is_none();
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

    // A 方案（t>1，`LINGXI_ROUTE_FULLDEV=1`）：批量设备路由——路由/top-k/pair/CSR 全在
    // 设备，host 只读映射计数（零 sync），仅在真未命中时补槽重跑。逐位等价性见 batch_dev.rs 头注。
    if rows > 1
        && crate::flag("LINGXI_ROUTE_FULLDEV")
        && !crate::flag("LINGXI_MOE_DEBUG")
        && !crate::flag("LINGXI_MOE_DUMP")
        && !crate::env_list_contains("LINGXI_FULLDEV_OFF", layer)
        && !SHADOW_NESTED.with(|c| c.get())
        // EP：批量设备路由把 gather/pair/CSR **全在设备上**建好（见 batch_dev.rs），
        // 没有插入专家分配表过滤的位置；强行过滤会改 CSR 顺序 ⇒ 破坏两卡一致性。
        // EP 下回退 legacy host 路（路由结果本身仍然全量、逐位一致，只是慢一点点）。
        && ep.is_none()
    {
        if let Some(s) = src {
            if let Some(out) = super::batch_dev::route_dev_batch_attempt(
                k, layer, w, x, &shared, s, num_experts, top_k, inter, hidden, rows,
            )? {
                // `LINGXI_FULLDEV_SHADOW=1`：同一进程内把 **legacy** 再算一遍当参考，
                // 逐元素比 bit（不改变对外行为，返回 legacy 结果）。
                if crate::flag("LINGXI_FULLDEV_SHADOW") {
                    SHADOW_NESTED.with(|c| c.set(true));
                    let mo = sparse_moe_forward(
                        k, w, x, rows, hidden, num_experts, top_k, inter, debug_dir, layer, src, None,
                        ep,
                    );
                    SHADOW_NESTED.with(|c| c.set(false));
                    let mo = mo?;
                    let a: Vec<bf16> = k.stream.clone_dtoh(&out)?;
                    let b: Vec<bf16> = k.stream.clone_dtoh(&mo.out)?;
                    let n = a.len().min(b.len());
                    let mut diff = 0usize;
                    let mut worst = 0f32;
                    let mut first = None;
                    for i in 0..n {
                        if a[i].to_bits() != b[i].to_bits() {
                            diff += 1;
                            if first.is_none() {
                                first = Some(i);
                            }
                            let d = (a[i].to_f32() - b[i].to_f32()).abs();
                            if d > worst {
                                worst = d;
                            }
                        }
                    }
                    eprintln!(
                        "[fulldev-shadow] L{layer} rows{rows} n{n} diff={diff} worst={worst:.6} first={:?}",
                        first.map(|i| (i / hidden, i % hidden))
                    );
                    return Ok(mo);
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
            // 未命中两次 / 前置不符：继续走 legacy 全路径
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
    // ★ router lookahead（router_logits 信号）：在 router GEMV 入队后、首个同步**之前**，
    // 把本层 router W 的设备地址登记进 ExpertSource 并异步发起"层 L+1 router × 本层
    // 输入 h"的预测 GEMV（独立流，不延长本层同步；消费点 = 本层 load_hits 的 pf_issue）。
    // 默认空实现 ⇒ 非 arena 源 / topk_set 信号档零开销。
    if let Some(s) = src {
        let (wpf, _gpf) = router_w.device_ptr(&k.stream);
        s.pf_predict_enqueue(layer, x, wpf)?;
    }
    let need = rows * num_experts;
    let t_rg = std::time::Instant::now();
    // ── 取回 host：**零拷贝**路径（所有 rows）──
    // 让 GEMM/GEMV 直接把 logits 写进「映射到设备的锁页 host 内存」，host 侧只剩一次
    // `cuStreamSynchronize`。原路径（`cuMemcpyDtoHAsync` + sync）实测**光这 1 KB 的拷贝
    // 就让同步多等 ~0.8 ms/层**（见 docs/decode-speed-profile.md 的 [d2h] 拆解），
    // 48 层累积 ~38 ms/token。`LINGXI_ROUTER_NODMAP=1` 可回退。
    // 【本轮修正】原来这里是 `rows == 1 && ...` —— 于是 **t>1 又跌回 D2H 老路**，
    // 而 §3.42 用「剥块后看 Δ(t=2−t=1) 是否塌缩」实测：**t=2 悬崖 +35.8 ms 里有 +23.4 ms 在 MoE**，
    // 正是这条 0.8 ms/层 × 48。
    // ★ 2026-10-22 二次修正：rows>1 **收回映射路径**。35f4246 放开时只测了 t=2（1 发切片）；
    // 长 prompt（t=512 ⇒ 64 发 ≤8 行切片）每发直写映射 host 内存 = PCIe write-through，
    // 实测 router gemm 段 24.5ms/层、占 prefill wall 约 1/3。rows>1 改走**位等价**设备写回：
    // 同样的核 / 同样的 8 行切片 / 同样的逐元素累加次序（⇒ logits 逐位不变、首 token
    // 锚点安全），只把输出去向从映射 host 内存换成设备缓冲 + 一次锁页 D2H。
    let no_dmap = crate::flag("LINGXI_ROUTER_NODMAP");
    let use_map = !no_dmap && rows == 1;
    // ★ 2026-10-22 ㊿+ 批量设备路由（**默认关**，`LINGXI_ROUTE_DEV=1` 开启）：
    // softmax+插入式 topk 可整体搬上 GPU（`route_topk_rows_bf16` 逐位复刻 host 数学——
    // glibc-expf 复刻 / 升序串行 sum / IEEE 除法 / 同全序选择 / 同序 norm，首 token
    // 332/97149 实锤位级一致），D2H 从 512KB logits 缩到 41KB 选中表。
    // ⚠️ **默认关的原因（实测 A/B）**：parity 21.6→22.7s、needle 52.6→56.2s——更慢。
    // host softmax/topk 期间 GPU 在消化上一层积压（host-bound 下的免费重叠）；改设备
    // 路由后 host 在小 D2H 的 stream.sync 上**干等 GPU 排空**，重叠结构被破坏。
    // 与 t=1 版 route_dev_attempt 当年「实测 −8% 先 park」同病：要启用必须先解决
    // 管道重叠（pair/CSR 也搬设备、或 D2H 与后续 host 簿记重排），届时再开。
    let route_dev = rows > 1
        && crate::flag("LINGXI_ROUTE_DEV")
        && !no_dmap
        && !crate::flag("LINGXI_ROUTE_HOST")
        // EP：设备 top-k 的结果同样要给 host 过滤专家分配表（可以兼容），但为了两卡路由
        // **逐位同源**、少一个变量，EP 下统一走 legacy host 路由。
        && ep.is_none();
    let mut logits_dev: Option<CudaSlice<bf16>> = None;
    let (logits_host, sel_dev): (Vec<bf16>, Option<(Vec<i32>, Vec<f32>)>) = if use_map {
        thread_local! {
            static MAP: std::cell::RefCell<Option<scratch::MappedHost<bf16>>> =
                const { std::cell::RefCell::new(None) };
        }
        MAP.with(|c| -> Result<(Vec<bf16>, Option<(Vec<i32>, Vec<f32>)>)> {
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
            k.gemv_bf16_ptr(wp, xp, m.dev, num_experts, hidden)?;
            k.stream.synchronize()?;
            Ok((m.slice()?[..need].to_vec(), None))
        })?
    } else {
        let t_alloc = std::time::Instant::now();
        // [probe-§AB] 判决实验：logits 改走**经典 cuMemAlloc**（without_pool）。
        // 假设：池化（cuMemAllocAsync/VMM）缓冲上的 CE D2H 被 IOMMU/VMM 慢路径拖到
        // ~21MB/s（实测 sync 等时 = payload/21MB/s + 固定 ~5ms；经典分配的独立微基准 6GB/s）。
        let mut logits = if crate::flag("LINGXI_ROUTER_CLASSIC_ALLOC") {
            crate::scratch::without_pool(|| scratch::bf16(k, rows * num_experts))?
        } else {
            scratch::bf16(k, rows * num_experts)?
        };
        let alloc_us = t_alloc.elapsed().as_nanos() as u64;
    if !no_dmap {
        // rows>1 位等价设备写回（见上面 2026-10-22 注释）：同核同累加次序，
        // 输出去向 = 设备缓冲；2026-10-22 二改：grid.y 行分块后**一发全覆盖**
        // （不再 64 发切片）。guard 活到 PIN 块内的 D2H 同步之后。
        let t_launch = std::time::Instant::now();
        let (wp, _a) = router_w.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = logits.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, num_experts, hidden, rows, 0, 0.0, 0.0)?;
        let launch_us = t_launch.elapsed().as_nanos() as u64;
        ROUTER_ALLOC_NS.fetch_add(alloc_us, Ordering::Relaxed);
        ROUTER_LAUNCH_NS.fetch_add(launch_us, Ordering::Relaxed);
    } else {
    match crate::gpu::prep_f32x(k, x, rows, rows * hidden, 1)? {
        // ★ P15-⑨：t>1 时 x 预转 fp32（与上面的 shared 段各转一份 scratch，池化复用、几乎免费）
        Some(xf) => crate::gpu::proj_f32x(k, &router_w, &xf, &mut logits, num_experts, hidden, rows, 0, 0.0, 0.0)?,
        None => proj_t1(k, x, &router_w, &mut logits, num_experts, hidden, rows)?,
    }
    }
        // 批量设备路由：核内 softmax+topk → 只回 41KB 选中表（ids 专家升序 + 对齐 wts）。
        // 锁页缓冲按上下文隔离（双卡实测教训，同下方 PIN 块）。
        let mut sel_dev: Option<(Vec<i32>, Vec<f32>)> = None;
        let (lh, _sel_from_arm): (Vec<bf16>, Option<(Vec<i32>, Vec<f32>)>) = if route_dev {
            let mut ids = scratch::i32(k, rows * top_k)?;
            let mut wts = scratch::f32(k, rows * top_k)?;
            k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, rows, num_experts, top_k)?;
            let out = {
                thread_local! {
                    static PIN_R: std::cell::RefCell<Option<(
                        cudarc::driver::PinnedHostSlice<i32>,
                        cudarc::driver::PinnedHostSlice<f32>,
                    )>> = const { std::cell::RefCell::new(None) };
                }
                PIN_R.with(|c| -> Result<(Vec<i32>, Vec<f32>)> {
                    let mut g = c.borrow_mut();
                    let need_r = rows * top_k;
                    let reuse = g.as_ref().map_or(false, |(pi, pw)| {
                        std::sync::Arc::ptr_eq(pi.context(), k.stream.context())
                            && pi.len() >= need_r
                            && pw.len() >= need_r
                    });
                    if !reuse {
                        *g = Some(unsafe {
                            (
                                k.stream.context().alloc_pinned_with_flags::<i32>(need_r, 0)?,
                                k.stream.context().alloc_pinned_with_flags::<f32>(need_r, 0)?,
                            )
                        });
                    }
                    let (pi, pw) = g.as_mut().unwrap();
                    k.stream.memcpy_dtoh(&ids, &mut pi.as_mut_slice()?[..need_r])?;
                    k.stream.memcpy_dtoh(&wts, &mut pw.as_mut_slice()?[..need_r])?;
                    k.stream.synchronize()?; // 异步拷贝，必须等完成再读
                    Ok((
                        pi.as_slice()?[..need_r].to_vec(),
                        pw.as_slice()?[..need_r].to_vec(),
                    ))
                })?
            };
            scratch::put_i32(ids);
            scratch::put_f32(wts);
            sel_dev = Some(out);
            (Vec::new(), None) // logits 不再回 host（dump 路径见下方守卫）
        } else {
            thread_local! {
                static PIN: std::cell::RefCell<Option<cudarc::driver::PinnedHostSlice<bf16>>> =
                    const { std::cell::RefCell::new(None) };
            }
            PIN.with(|c| -> Result<(Vec<bf16>, Option<(Vec<i32>, Vec<f32>)>)> {
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
                // [探针] 等的是"router GEMM + 全流积压"。注意 `memcpy_dtoh` **自身就会
                // 排空流**（cudarc 同步版）⇒ 计时必须包住它而不是后面的 synchronize。
                // [probe-§AB] GPU 侧事件对横跨同步：区分"GPU 真有积压" vs "同步调用本身慢"。
                // 事件只记不同步；span = D2H 完成时刻 − 事件入队被 GPU 执行到的时刻。
                let evp = if prof {
                    let f = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                    Some((k.ctx.new_event(f)?, k.ctx.new_event(f)?))
                } else {
                    None
                };
                if let Some((b, _)) = &evp {
                    b.record(&k.stream)?;
                }
                // [probe-§AB] 细分：copy 返回 vs synchronize 返回，定位 25ms 躺在哪半。
                let t_syn = std::time::Instant::now();
                k.stream.memcpy_dtoh(&logits, &mut buf.as_mut_slice()?[..need])?;
                let t_copy_done = std::time::Instant::now();
                if crate::flag("LINGXI_EVT_SYNC") {
                    // 对照实验：stream 同步换事件同步（拷贝是流上最后一发，语义等价）。
                    let f = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                    let ev = k.ctx.new_event(f)?;
                    ev.record(&k.stream)?;
                    ev.synchronize()?;
                } else if crate::flag("LINGXI_EVT_POLL") {
                    // [probe-§AB] 判决实验：busy-poll cuEventQuery 替代 synchronize。
                    // 若 poll 很快返回 ⇒ 完成信号本身可见、是 sync 的通知路径慢；
                    // 若 poll 也等 25ms ⇒ 完成信号就是延迟的（CE 中断路径）。
                    let f = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                    let ev = k.ctx.new_event(f)?;
                    ev.record(&k.stream)?;
                    let mut spins = 0u64;
                    loop {
                        match unsafe {
                            cudarc::driver::result::event::query(ev.cu_event())
                        } {
                            Ok(()) => break,
                            Err(e) => {
                                if e.0
                                    != cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY
                                {
                                    Err(e)?;
                                }
                            }
                        }
                        spins += 1;
                        std::hint::spin_loop();
                    }
                    let _ = spins;
                } else if crate::flag("LINGXI_KICK") {
                    // [probe-§AB] 判决实验：enqueue 后立刻 cuStreamQuery 强制 doorbell/flush，
                    // 再同步。若 sync 塌缩 ⇒ 病根是提交批次化延迟（门铃晚响）。
                    let r = unsafe {
                        cudarc::driver::sys::cuStreamQuery(k.stream.cu_stream())
                    };
                    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS
                        && r != cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY
                    {
                        Err(cudarc::driver::DriverError(r))?;
                    }
                    k.stream.synchronize()?;
                } else {
                    k.stream.synchronize()?; // `memcpy_dtoh` 是异步的，必须等完成再读 host 缓冲
                }
                SYNC_WAIT_NS.fetch_add(t_syn.elapsed().as_nanos() as u64, Ordering::Relaxed);
                // [probe-§AB] 判决实验 5：同步刚返回时立刻测一块**专用预热缓冲**的同尺寸
                // D2H（流此时是空的）。若它也慢 ⇒ CE 全局被拖（外部竞争）；若快 ⇒ 慢的是
                // logits 这块缓冲/此刻的流状态。
                if crate::flag("LINGXI_D2H_BW") {
                    thread_local! {
                        static PROBE_BUF: std::cell::RefCell<Option<(cudarc::driver::CudaSlice<bf16>, cudarc::driver::PinnedHostSlice<bf16>)>> =
                            const { std::cell::RefCell::new(None) };
                    }
                    PROBE_BUF.with(|c| -> Result<()> {
                        let mut g = c.borrow_mut();
                        let reuse = g.as_ref().map_or(false, |(d, _)| {
                            std::sync::Arc::ptr_eq(d.context(), k.stream.context()) && d.len() >= need
                        });
                        if !reuse {
                            let mut d = unsafe { k.stream.alloc::<bf16>(need) }?;
                            let h = unsafe {
                                k.stream.context().alloc_pinned_with_flags::<bf16>(need, 0)
                            }?;
                            k.stream.memcpy_htod(&vec![bf16::from_f32(1.0); need], &mut d)?;
                            k.stream.synchronize()?;
                            *g = Some((d, h));
                        }
                        let (d, h) = g.as_mut().unwrap();
                        let tb = std::time::Instant::now();
                        k.stream.memcpy_dtoh(&*d, &mut h.as_mut_slice()?[..need])?;
                        k.stream.synchronize()?;
                        eprintln!(
                            "    [d2h-bw] L{layer:2} rows{rows} clean_copy={:.3}ms",
                            tb.elapsed().as_secs_f64() * 1e3
                        );
                        Ok(())
                    })?;
                }
                if let Some((b, e)) = &evp {
                    e.record(&k.stream)?;
                    let gpu_ms = b.elapsed_ms(e).unwrap_or(-1.0);
                    eprintln!(
                        "    [moe-sync-probe] L{layer:2} rows{rows} host_spin={:.3}ms copy={:.3}ms sync={:.3}ms gpu_span={gpu_ms:.3}ms",
                        t_syn.elapsed().as_secs_f64() * 1e3,
                        (t_copy_done - t_syn).as_secs_f64() * 1e3,
                        (t_copy_done.elapsed()).as_secs_f64() * 1e3,
                    );
                }
                Ok((buf.as_slice()?[..need].to_vec(), None))
            })?
        };
        logits_dev = Some(logits);
        (lh, sel_dev)
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
    // 批量设备路由命中：选中表已按 (行升序, 专家升序) 排好，直接展开——与下方 host
    // 循环产生的 per_expert/sel_rows **逐项相同**（同序同值，w 位级一致）。
    if let Some((ids_h, wts_h)) = &sel_dev {
        for r in 0..rows {
            for j in 0..top_k {
                let e = ids_h[r * top_k + j] as usize;
                let w = wts_h[r * top_k + j];
                per_expert[e].push((r as i32, w));
                sel_rows.push((r as u32, e as i32, w));
            }
        }
    } else {
        // ★ §AG（2026-10-27）：host topk+softmax 逐行并行化。t=512 时该循环 ~3.8ms/层是
        // 纯 host CPU（512 行互相独立，实测占层 wall 的 ~15%）。两阶段、**逐位等价**：
        //   A（rayon 全局池，§AG-补 起免每 call spawn）：每行复刻旧循环的
        //     max/exp/sum/归一/topk/renorm 算术（每线程独立 scratch，同值同序），
        //     产出 (专家升序, p/norm) 平铺表；
        //   B（串行）：按行升序合并进 per_expert/sel_rows —— 与旧循环的 push 次序
        //     **逐项相同**（行升序 × 行内专家升序 × 同 w 位级）。
        // `LINGXI_NO_TOPK_PAR=1` 回退旧单线程循环（对拍用）。
        let par = !crate::flag("LINGXI_NO_TOPK_PAR") && rows >= 32;
        if par {
            use rayon::prelude::*;
            // §AG-补：rayon 全局池（进程常驻线程，免每 call spawn ~0.5ms）。
            let nthreads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .clamp(2, 16);
            let mut flat: Vec<(i32, f32)> = vec![(0i32, 0.0f32); rows * top_k];
            let chunk = rows.div_ceil(nthreads);
            let num_experts = num_experts;
            flat.par_chunks_mut(chunk * top_k)
                .enumerate()
                .try_for_each(|(pi, part)| -> Result<()> {
                    let r0 = pi * chunk;
                    let mut probs: Vec<f32> = vec![0.0f32; num_experts];
                    let mut sel: Vec<(usize, f32)> = Vec::with_capacity(top_k + 1);
                    let nrows = part.len() / top_k;
                    for j in 0..nrows {
                        let r = r0 + j;
                        if r >= rows {
                            break;
                        }
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
                        // 与旧循环一致：per_expert 按专家下标升序枚举
                        sel.sort_by_key(|(i, _)| *i);
                        for (t, &(e, p)) in sel.iter().enumerate() {
                            part[j * top_k + t] = (e as i32, p / norm);
                        }
                    }
                    Ok(())
                })?;
            // B：串行合并（行升序，与旧循环 push 次序逐项相同）
            for r in 0..rows {
                for j in 0..top_k {
                    let (e, w) = flat[r * top_k + j];
                    per_expert[e as usize].push((r as i32, w));
                    sel_rows.push((r as u32, e, w));
                }
            }
        } else {
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
        }
    }
    if let Some(d) = &dump_dir {
        if sel_dev.is_some() {
            // 批量设备路由下 logits 不回 host，跳过 logits dump（sel dump 仍有效）。
            eprintln!("[moe] dump: route_dev 模式跳过 logits dump（LINGXI_ROUTE_HOST=1 可回退）");
        }
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

    // ★ EP（`LINGXI_EP=1`）：路由是**全量**算出来的（两卡同核同数据 ⇒ 选出的
    // (行, 专家, 权重) 逐位一致）；这里只把分配表里不属于本卡的**对**剔掉。
    // 下游所有簿记（hit_ids / pair_* / uniq_experts / group_off / CSR）都从 `per_expert`
    // 派生，因此在这一处过滤 == "只加载、只计算本卡的专家"。
    if let Some(ep) = ep {
        anyhow::ensure!(
            ep.assign.len() >= num_experts,
            "EP 专家分配表太短：{} < num_experts={num_experts}",
            ep.assign.len()
        );
        for (e, hits) in per_expert.iter_mut().enumerate() {
            if ep.assign[e] != ep.card {
                hits.clear();
            }
        }
        sel_rows.retain(|&(_, e, _)| ep.assign[e as usize] == ep.card);
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
    // 对拍（`LINGXI_PAIR_DUMP=1`）：与 `moe/batch_dev.rs` 设备侧构建的 pair/CSR 逐项比对。
    if crate::flag("LINGXI_PAIR_DUMP") {
        let h = super::hash_pair_tables(&pair_rows, &pair_wts);
        eprintln!(
            "[pair-host] L{layer} rows{rows} n{} h{h:016x} rows8={:?} wts8={:?}",
            pair_rows.len(),
            &pair_rows[..pair_rows.len().min(8)],
            &pair_wts[..pair_wts.len().min(8)]
        );
    }
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
    if crate::flag("LINGXI_PAIR_DUMP") {
        eprintln!(
            "[pair-host] L{layer} go{:?} ue8={:?}",
            &group_off[..group_off.len().min(32)],
            &uniq_experts[..uniq_experts.len().min(8)]
        );
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
            if crate::flag("LINGXI_PAIR_DUMP") {
                let h = super::hash_i32(&col) ^ super::hash_i32(&roff);
                eprintln!(
                    "[pair-host] L{layer} col n{} h{h:016x} col8={:?} roff{:?}",
                    col.len(),
                    &col[..col.len().min(8)],
                    &roff[..roff.len().min(8)]
                );
            }
            (scratch::upload_i32(k, &col)?, scratch::upload_i32(k, &roff)?)
        } else {
            // rows==1：combine 走 `moe_combine_add_rows_bf16`，CSR 的 `col`/`roff` **根本不读**
            // ⇒ 只留两块空占位，**不做 H2D**（每步省两次 0 字节异步拷贝 + 两次事件记录）。
            (scratch::i32(k, 0)?, scratch::i32(k, 0)?)
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
        // ★ EP：本层命中专家可能**全在对面卡**（EP 按 id 二分，路由仍全量）。
        // 此时 hit_list 为空是合法状态 —— 本卡无专家工作，输出零，靠层内交换并到另一半。
        // 不能把空列表喂给 load_hits：ArenaExpertSource 会直接报"空专家列表"。
        if hit_list.is_empty() {
            compact_gu = None;
            compact_dn = None;
            q4_hits = None;
            q4_alt_hits = None;
            bits_sel = Vec::new();
        } else {
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
        }
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
                if prof4 {
                    evs[1].record(&k.stream)?;
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
                if prof4 {
                    evs[2].record(&k.stream)?;
                }
                // swiglu 整批（与 rows>1 同构：全量一次）
                k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                if prof4 {
                    evs[3].record(&k.stream)?;
                }
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
                if prof4 {
                    evs[4].record(&k.stream)?;
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
                    // `LINGXI_MOE_PROF=1`：rows==1 单格式档也切四段（原来只有混档/批量档有事件，
                    // 草稿 t=1 的 [moe-prof4] 只剩一个"合计"）。纯诊断，不改变执行次序。
                    if prof4 {
                        evs[1].record(&k.stream)?;
                    }
                    if q.bits == 3 {
                        k.gemv_q3g128_slots_rows_bf16(
                            q.slots, q.slot_ids, xs, &mut gu_all, s, 2 * inter, hidden,
                            q.gu_k_pad, gu_stride, q.blob, q.gu_seg, q.gu_scale, rpw,
                        )?;
                        if prof4 {
                            evs[2].record(&k.stream)?;
                        }
                        k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                        if prof4 {
                            evs[3].record(&k.stream)?;
                        }
                        k.gemv_q3g128_slots_rows_bf16(
                            q.slots, q.slot_ids, &act_all, &mut o_all, s, hidden, inter,
                            q.dn_k_pad, inter, q.blob, q.dn_seg, q.dn_scale, rpw,
                        )?;
                        if prof4 {
                            evs[4].record(&k.stream)?;
                        }
                    } else {
                    k.gemv_q4g64_slots_rows_bf16(
                        q.slots, q.slot_ids, xs, &mut gu_all, s, 2 * inter, hidden, q.gu_k_pad, gu_stride,
                        q.blob, q.gu_seg, q.gu_scale, rpw,
                    )?;
                    if prof4 {
                        evs[2].record(&k.stream)?;
                    }
                    k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
                    if prof4 {
                        evs[3].record(&k.stream)?;
                    }
                    k.gemv_q4g64_slots_rows_bf16(
                        q.slots, q.slot_ids, &act_all, &mut o_all, s, hidden, inter, q.dn_k_pad, inter,
                        q.blob, q.dn_seg, q.dn_scale, rpw,
                    )?;
                    if prof4 {
                        evs[4].record(&k.stream)?;
                    }
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
            // ★ 2026-10-22（moe-gemm-bench 隔离判决 12 档全过：×3.3~×33，zipf ×15.7）：
            // 取消 `!dual_q4` 门控 —— 混档时 Q4 视图（runs4）照样可以走 TCP，Q3 视图由
            // run() 内按位宽退回 SIMT。这是 prefill 混档默认提速的最后一块拼图。
            let tcp_on = crate::flag("LINGXI_MOE_TC") && rows > 16
                && hidden % 64 == 0 && inter % 64 == 0
                && !q3_on;
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
                // Q4 视图（含混档 runs4）走 Q4 TCP；Q3 视图走 ㊽+3 新 Q3 TCP 大 tile 核
                //（有损档，判据同 q4_tcp：首 token 相等 + 质量抽检；`LINGXI_NO_Q3TCP=1` 回退 SIMT）。
                let tcp_on = (tcp_on && !q3_on)
                    || (q3_on
                        && crate::flag("LINGXI_MOE_TC")
                        && rows > 16
                        && q.gu_k_pad % 128 == 0
                        && q.dn_k_pad % 128 == 0
                        && !crate::flag("LINGXI_NO_Q3TCP"));
                for &(a, b) in runs {
                    if b <= a {
                        continue;
                    }
                    let gsel = d_group.slice(a as usize..=b as usize);
                    let ssel = q.slot_ids.slice(a as usize..b as usize);
                    let cnt = (b - a) as usize;
                    // Q3 t2 档的 `MB=1` 实例要求"每组恰好 1 对"（`group_off` 在 host 上，直接查）。
                    let q3_all_m1 = (a as usize..b as usize)
                        .all(|g| group_off[g + 1] - group_off[g] == 1);
                    if gu {
                        // ㊽ prefill 大 tile 张量核档：一次 launch 覆盖整个专家区间，
                        // M 轴在 grid.z 里分块（tmax = 区间最大组行数）。
                        if tcp_on {
                            let tmax = (a as usize..b as usize)
                                .map(|g| (group_off[g + 1] - group_off[g]) as usize)
                                .max()
                                .unwrap_or(1);
                            if std::env::var("LINGXI_TCP_PROBE").is_ok() {
                                eprintln!("[tcp-probe] gu launch: cnt={cnt} tmax={tmax}");
                            }
                            if q3_on {
                                k.gemv_q3g128_slots_tcp_bf16_view(
                                    q.slots, &ssel, &gsel, xs, gu_all, cnt, 2 * inter, hidden,
                                    gu_stride, q.gu_k_pad, q.blob, q.gu_seg, q.gu_scale, tmax,
                                )?;
                            } else {
                            k.gemv_q4g64_slots_tcp_bf16_view(
                                q.slots, &ssel, &gsel, xs, gu_all, cnt, 2 * inter, hidden,
                                gu_stride, q.gu_k_pad, q.blob, q.gu_seg, q.gu_scale, tmax,
                            )?;
                            }
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
                                        None if q3_on => k.gemv_q3g128_slotsm_t2_view_gidx(
                                            q.slots, &ssel1, xs, &gsel1, gu_all, 1, 2 * inter,
                                            hidden, q.gu_k_pad, gu_stride, q.blob, q.gu_seg,
                                            q.gu_scale, sm_rpw, xf, 0, cnt == 1,
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
                                None if q3_on => k.gemv_q3g128_slotsm_t2_view_gidx(
                                    q.slots, &ssel, xs, &gsel, gu_all, cnt, 2 * inter, hidden,
                                    q.gu_k_pad, gu_stride, q.blob, q.gu_seg, q.gu_scale, sm_rpw,
                                    xf, 0, q3_all_m1,
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
                            if std::env::var("LINGXI_TCP_PROBE").is_ok() {
                                eprintln!("[tcp-probe] dn launch: cnt={cnt} tmax={tmax}");
                            }
                            if q3_on {
                                k.gemv_q3g128_slots_tcp_bf16_view(
                                    q.slots, &ssel, &gsel, act_all, o_all, cnt, hidden, inter,
                                    inter, q.dn_k_pad, q.blob, q.dn_seg, q.dn_scale, tmax,
                                )?;
                            } else {
                            k.gemv_q4g64_slots_tcp_bf16_view(
                                q.slots, &ssel, &gsel, act_all, o_all, cnt, hidden, inter,
                                inter, q.dn_k_pad, q.blob, q.dn_seg, q.dn_scale, tmax,
                            )?;
                            }
                        } else {
                        match dp {
                            Some((xq, sx)) => k.gemv_q4g64_slotsm_rpw_dp4a_view(
                                q.slots, &ssel, act_all, &gsel, o_all, cnt, hidden, inter,
                                q.dn_k_pad, inter, q.blob, q.dn_seg, q.dn_scale, sm_rpw, xq, sx,
                            )?,
                            None if q3_on => k.gemv_q3g128_slotsm_t2_view_gidx(
                                q.slots, &ssel, act_all, &gsel, o_all, cnt, hidden, inter,
                                q.dn_k_pad, inter, q.blob, q.dn_seg, q.dn_scale, sm_rpw, xf, 0,
                                q3_all_m1,
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
                        // 混档：**每视图一次 launch 覆盖全部组**（2026-10-22 单 launch 化）。
                        // 视图的 slot_ids 是他格式段 -1 哨兵（arena_source 上传），核里
                        // `slot < 0` 早退；组间零交互 ⇒ 逐组数学与按 runs 切分时完全一致。
                        // （原按 runs 切是"哨兵=0"时代的产物——0 是合法槽号；现已改 -1。）
                        // runs4/runs3 只用于"该格式是否有组"的过滤，避免全哨兵空 launch。
                        let full = [(0u32, e_uniq as u32)];
                        let mut views: Vec<(&Q4Slots, bool)> = vec![(q, !runs4.is_empty())];
                        if let Some(qa) = q4_alt_hits.as_ref() {
                            views.push((qa, !runs3.is_empty()));
                        }
                        for (qv, has) in views {
                            if !has {
                                continue;
                            }
                            run(&full, true, &mut gu_all, &act_all, &mut o_all, 0, None, qv)?;
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
                        // 同 gu：每视图一次 launch（哨兵 -1 早退，数学按组不变）。
                        let full = [(0u32, e_uniq as u32)];
                        let mut views: Vec<(&Q4Slots, bool)> = vec![(q, !runs4.is_empty())];
                        if let Some(qa) = q4_alt_hits.as_ref() {
                            views.push((qa, !runs3.is_empty()));
                        }
                        for (qv, has) in views {
                            if !has {
                                continue;
                            }
                            run(&full, false, &mut gu_all, &act_all, &mut o_all, 0, None, qv)?;
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
                    if prof4 {
                        evs[1].record(&k.stream)?;
                    }
                    // ① 命中组先算：此时 miss 的 blob H2D 正在**第二流**上飞（两边并行）
                    run(&ov.hit_runs, true, &mut gu_all, &act_all, &mut o_all, xf_gu_p, dp_gu, q)?;
                    if prof4 {
                        evs[2].record(&k.stream)?;
                    }
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
                    if prof4 {
                        evs[3].record(&k.stream)?;
                    }
                    run(&ov.hit_runs, false, &mut gu_all, &act_all, &mut o_all, xf_dn_p, dp_dn, q)?;
                    run(&miss_runs, false, &mut gu_all, &act_all, &mut o_all, xf_dn_p, dp_dn, q)?;
                    if prof4 {
                        evs[4].record(&k.stream)?;
                    }
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
        // `LINGXI_FULLDEV_OUT_DUMP` 复用同一块（对拍 A 方案三段缓冲），但不触发
        // `route_dev_batch_attempt` 的"诊断开关即回退"（LINGXI_MOE_OUT_DUMP 会让 A 路径直接关掉）。
        if (crate::flag("LINGXI_MOE_OUT_DUMP") || crate::flag("LINGXI_FULLDEV_OUT_DUMP")) && {
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
                    let mut bits: u64 = 0xcbf29ce484222325;
                    for v in h.iter() {
                        bits = (bits ^ (v.to_bits() as u64)).wrapping_mul(0x100000001b3);
                    }
                    eprintln!(
                        "[moe-out] 层{layer} {nm} 全和={tot:.6} n={} nan={nans} first_nan={first:?} h{bits:016x}",
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
            {
                let ms = |i: usize| evs[i].elapsed_ms(&evs[i + 1]).unwrap_or(0.0);
                let tot = evs[0].elapsed_ms(&evs[5]).unwrap_or(0.0);
                eprintln!(
                    "[moe-prof4] 层{layer} rows={rows} s={s} 前置 {:.3} | gu {:.3} | \
                     swiglu+xf {:.3} | dn {:.3} | combine+收尾 {:.3} | 合计 {:.3} ms",
                    ms(0), ms(1), ms(2), ms(3), ms(4), tot,
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
        // 自上次打印以来的三项分解（全局 atomic；打印处清零，单线程近似无竞争）。
        let sync_us = SYNC_WAIT_NS.swap(0, Ordering::Relaxed) as f64 / 1e3;
        let alloc_us = ROUTER_ALLOC_NS.swap(0, Ordering::Relaxed) as f64 / 1e3;
        let launch_us = ROUTER_LAUNCH_NS.swap(0, Ordering::Relaxed) as f64 / 1e3;
        eprintln!(
            "    [moe] 层 {layer:2} shared {:.1} router {:.1} (gemm {:.1} D2H {:.1} topk {:.1}) 专家 {:.1} 合计 {:.1} µs | enq {:.1} up {:.1} | alloc {:.1} launch {:.1} sync {:.1}",
            t_shared.as_secs_f64() * 1e6,
            (t_router - t_shared).as_secs_f64() * 1e6,
            (t_rd - t_rg).as_secs_f64() * 1e6,
            (t_rt - t_rd).as_secs_f64() * 1e6,
            (t_rk - t_rt).as_secs_f64() * 1e6,
            (t_all - t_router).as_secs_f64() * 1e6,
            t_all.as_secs_f64() * 1e6,
            t_enq.elapsed().as_secs_f64() * 1e6,
            t_up_done,
            alloc_us,
            launch_us,
            sync_us,
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

