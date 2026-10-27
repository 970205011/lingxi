//! A 方案（t>1，`LINGXI_ROUTE_FULLDEV=1`）：**整条路由与 pair/CSR 构建留在设备**。
//!
//! 动机：投机验证批每层在
//! `router logits D2H + stream.synchronize` 处把 host 钉死在单层，随后 ~0.27ms 的
//! host 簿记（softmax/top-k/查槽/pair/CSR/H2D）与 GPU 完全串行 ⇒ 轮 wall 80ms 里
//! ~13ms 是这条链。这里把同一批决策搬上 GPU，host 只读**映射内存**里的计数（零 sync），
//! 只在真未命中时补槽（`load_missing`）再重跑一次。
//!
//! 逐位等价的三块：
//! ① router 投影用 legacy rows>1 **同一条核**（`gemm_skinny_row_bf16_ptr`），
//! softmax+插入式 top-k 用 `route_topk_rows_bf16`（已与 host 逐位对拍过）；
//! ② pair 表/组表/行 CSR 的**次序**由设备核按 host 遍历序重建（专家升序 → 行升序 →
//! j 升序）——`moe_combine_add_csr_bf16` 的累加次序因此不变；
//! ③ 专家段仍用同一批核、同一参数体系（TC/SIMT 按 legacy 同规则择一）；
//! 混档双视图各发**自己格式的组**（`pri_gidx`/`alt_gidx` 压缩下标 + `ng = npri/nalt`，
//! 见 2026-10-24 的组压缩，`LINGXI_NO_GROUP_COMPACT=1` 回退到全 `u` + `-1` 哨兵直通）。

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

use crate::scratch;

use super::forward::{slots_rpw, slotsm_rpw};
use super::router::route_spin;
use super::{ExpertSource, MoeWeights, Q4Consts};

/// pair 构建核的映射 host 区：计数 8 项 + 两个保护集 + 未命中专家表。
struct PairMapped {
    counts_dev: u64,
    counts_host: *mut u32,
    hit_pri_dev: u64,
    hit_pri_host: *mut i32,
    hit_alt_dev: u64,
    hit_alt_host: *mut i32,
    miss_ids_dev: u64,
    miss_ids_host: *mut i32,
}

/// 计数 8 项：{S, U, nmiss, tmax, nprot_pri, nprot_alt, npri, nalt}
/// （`nprot_*` = 已进槽的组数；`npri`/`nalt` = **每格式的组数**，两者在 nmiss==0 时相等）
const COUNTS_N: usize = 8;

/// 诊断：`LINGXI_FULLDEV_SYNC=1` 时每次 launch 后整条流同步，把异步的
/// `CUDA_ERROR_ILLEGAL_ADDRESS` 钉到具体那一发（`label` 会写进错误信息）。
fn sync_mark(k: &Kernels, label: &str) -> Result<()> {
    if crate::flag("LINGXI_FULLDEV_SYNC") {
        k.stream.synchronize().map_err(|e| anyhow::anyhow!("[fulldev-sync] {label}: {e}"))?;
    }
    Ok(())
}

/// **seg4（专家段）内部九分探针**（`LINGXI_SEG4_SYNC=1`，默认关）。
///
/// 为什么要有它：evprof 把"gather → 预转 → gu → swiglu → 预转 → dn → combine"整段包在
/// 一个事件对里（层均 ~0.49 ms），只看总数分不出主次。
///
/// **口径**：与 evprof 同款——每个边界在**流内**记一个带计时的事件（不排空流水），
/// 层末统一 sync 一次读出 9 段。为什么不照 `s3` 那样每个边界 `synchronize`：
/// 实测（2026-10-23）那样会把 gather/swiglu/combine 这种 ~3 µs 的小核量成 12~17 µs
/// （空队列下的 launch 唤醒延迟全算进去了），份额被放大 4 倍；小核的绝对值请看
/// `crates/lingxi-kernels/tests/t5_seg4_shape_bw.rs` 的 `t5_seg4_small_kernels`。
///
/// ⚠️ 层末一次 sync 仍会排空流水、拖慢整轮 ⇒ **只看份额**，不要用它评吞吐。
mod s4 {
    use super::Kernels;
    use anyhow::Result;
    use cudarc::driver::{sys, CudaEvent};
    use std::cell::RefCell;

    pub const NSEG: usize = 9;
    const NAMES: [&str; NSEG] = [
        "gather", "xq_gu", "gu_q4", "gu_q3", "swiglu", "xq_dn", "dn_q4", "dn_q3", "combine",
    ];

    struct Dev {
        ctx: usize,
        /// `NSEG + 1` 个带计时事件：第 `s` 段的区间 = `evs[s] → evs[s+1]`
        evs: Vec<CudaEvent>,
    }

    struct St {
        /// 本层各段 µs
        cur: [f64; NSEG],
        /// [设备] × [段] 累计
        sum: [[f64; NSEG]; 2],
        n: [f64; 2],
        seen: u32,
        /// 上一次 mark 的段号（补记被跳过的边界用）
        last_idx: usize,
        /// 现场：s / 组数 / rows / 主视图对数 / 次视图对数
        info: [u32; 5],
        devs: Vec<Dev>,
    }

    thread_local! {
        static ST: RefCell<St> = RefCell::new(St {
            cur: [0.0; NSEG], sum: [[0.0; NSEG]; 2], n: [0.0; 2], seen: 0, last_idx: 0,
            info: [0; 5], devs: Vec::new(),
        });
    }

    /// seg4 起点记现场（s / 组数 / rows / 主视图对数 / 次视图对数），供逐层行打印。
    pub fn info(s: usize, u: usize, rows: usize, npair_pri: u32, npair_alt: u32) {
        if !on() {
            return;
        }
        ST.with(|c| {
            c.borrow_mut().info = [s as u32, u as u32, rows as u32, npair_pri, npair_alt];
        });
    }

    #[inline]
    pub fn on() -> bool {
        crate::flag("LINGXI_SEG4_SYNC")
    }

    #[inline]
    fn layer_round() -> u32 {
        std::env::var("LINGXI_SEG4_LAYER")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0)
    }

    /// `idx`：0=seg4 起点，1..=9=各段结束（1 gather / 2 预转gu / 3 gu-q4 /
    /// 4 gu-q3 / 5 swiglu / 6 预转dn / 7 dn-q4 / 8 dn-q3 / 9 combine）。
    pub fn mark(k: &Kernels, layer: u32, idx: usize) -> Result<()> {
        if !on() {
            return Ok(());
        }
        let dev = (k.ctx.ordinal() & 1) as usize;
        let ctx = k.ctx.cu_ctx() as usize;
        ST.with(|c| -> Result<()> {
        let mut g = c.borrow_mut();
        let di = match g.devs.iter().position(|d| d.ctx == ctx) {
            Some(i) => i,
            None => {
                // ⚠️ 必须开 timing（cudarc 的 `new_event(None)` 是 DISABLE_TIMING）
                let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
                let mut evs = Vec::with_capacity(NSEG + 1);
                for _ in 0..=NSEG {
                    evs.push(k.ctx.new_event(flags)?);
                }
                g.devs.push(Dev { ctx, evs });
                g.devs.len() - 1
            }
        };
        // ⚠️ 某些段可能整段不存在（例如混档层 alt 命中 0 ⇒ 没有 gu-q3/dn-q3 那一发）。
        // 若不补记，该段的区间会跨越"上一层的旧事件"⇒ 出现 ±900 µs 的假值。
        // 这里把跳过的边界**按当前位置补记**，跳过的段即为 0 宽（必须**先**补记再记 idx）。
        let last = g.last_idx;
        for j in last + 1..idx.min(NSEG) {
            g.devs[di].evs[j].record(&k.stream)?;
        }
        g.devs[di].evs[idx.min(NSEG)].record(&k.stream)?;
        g.last_idx = idx.min(NSEG);
        if idx != NSEG {
            return Ok(());
        }
        // 层末一次 sync：只为读 elapsed（事件已全部完成）
        k.stream.synchronize()?;
        let mut cur = [0f64; NSEG];
        {
            let d = &g.devs[di];
            for s in 0..NSEG {
                if let Ok(ms) = d.evs[s].elapsed_ms(&d.evs[s + 1]) {
                    cur[s] = ms as f64 * 1e3;
                }
            }
        }
        g.cur = cur;
        for (a, b) in g.sum[dev].iter_mut().zip(cur.iter()) {
            *a += *b;
        }
        g.n[dev] += 1.0;
        g.seen += 1;
        if g.seen / 48 == layer_round() {
            let tot: f64 = cur.iter().sum();
            let i = g.info;
            let mut line = String::new();
            for (j, name) in NAMES.iter().enumerate() {
                line.push_str(&format!("{name} {:6.1}／", cur[j]));
            }
            eprintln!(
                "[s4L] #{} dev{dev} L{layer:<2} s{} 组{} rows{} pri{} alt{}｜{line}合计 {tot:.1} µs",
                g.seen, i[0], i[1], i[2], i[3], i[4]
            );
        }
        let n = g.n[dev];
        if n >= 240.0 {
            let a = g.sum[dev];
            let tot: f64 = a.iter().sum();
            let mut line = String::new();
            for (j, name) in NAMES.iter().enumerate() {
                line.push_str(&format!("{name} {:6.1}（{:4.1}%）／", a[j] / n, 100.0 * a[j] / tot.max(1e-9)));
            }
            eprintln!("[s4] dev{dev} L{layer} n={n:.0} 层均 µs：{line}合计 {:.1}", tot / n);
            g.sum[dev] = [0.0; NSEG];
            g.n[dev] = 0.0;
        }
        Ok(())
        })
    }
}

/// 诊断：`LINGXI_FULLDEV_HTIME=1` 时**逐段累计主机侧 µs**（launch / 自旋 / 读计数），
/// 每 240 次调用按设备打一行。目的：evprof 的"路由+host 段"是流时间轴，分不清
/// "GPU 在算" 与 "主机在排队"——这个探针量的是主机自己花的墙钟。
mod htime {
    use std::cell::Cell;
    use std::time::Instant;

    thread_local! {
        /// [dev(0/1)] 六项累计 µs：router+topk launch、pairs launch、自旋、读计数、其余 host、次数
        static ACC: Cell<[[f64; 6]; 2]> = const { Cell::new([[0.0; 6]; 2]) };
    }

    #[inline]
    pub fn on() -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| std::env::var("LINGXI_FULLDEV_HTIME").map(|s| s == "1").unwrap_or(false))
    }

    pub fn add(dev: usize, idx: usize, t: Instant) {
        let us = t.elapsed().as_secs_f64() * 1e6;
        ACC.with(|c| {
            let mut a = c.get();
            a[dev & 1][idx] += us;
            c.set(a);
        });
    }

    pub fn tick(dev: usize, layer: u32) {
        ACC.with(|c| {
            let mut a = c.get();
            a[dev & 1][5] += 1.0;
            c.set(a);
            let n = a[dev & 1][5];
            if n >= 240.0 {
                let [rl, pl, sp, rd, oh, _] = a[dev & 1];
                let per = |v: f64| v / n;
                eprintln!(
                    "[htime] dev{dev} L{layer} n={n:.0} 层均 µs：launch(r+t) {:.1}／pairs {:.1}／自旋 {:.1}／读计数 {:.1}／其余 {:.1}｜合计 {:.1}",
                    per(rl), per(pl), per(sp), per(rd), per(oh),
                    per(rl + pl + sp + rd + oh)
                );
                a[dev & 1] = [0.0; 6];
                c.set(a);
            }
        });
    }
}

fn pair_mapped(k: &Kernels, e: usize) -> Result<PairMapped> {
    struct Slot {
        ctx: usize,
        counts: scratch::MappedHost<u32>,
        hit_pri: scratch::MappedHost<i32>,
        hit_alt: scratch::MappedHost<i32>,
        miss_ids: scratch::MappedHost<i32>,
    }
    thread_local! {
        /// **按上下文多槽**（不是单槽）：layer-split 下每层交替 GPU0/GPU1，单槽缓存会在
        /// 每次换卡时 `cuMemFreeHost` 掉另一张卡正在写的映射区 —— 那正是
        /// `cuEventQuery -> CUDA_ERROR_ILLEGAL_ADDRESS` 的来源（异步、`CUDA_LAUNCH_BLOCKING=1`
        /// 就消失）。两块都留着，进程内不再释放。
        static MM: std::cell::RefCell<Vec<Slot>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    MM.with(|c| -> Result<PairMapped> {
        let mut g = c.borrow_mut();
        let ctx_id = k.ctx.cu_ctx() as usize;
        let mut idx = None;
        for (i, m) in g.iter().enumerate() {
            if m.ctx == ctx_id {
                idx = Some(i);
                break;
            }
        }
        let needs_alloc = match idx {
            Some(i) => {
                let m = &g[i];
                m.counts.len() < COUNTS_N || m.hit_pri.len() < e || m.hit_alt.len() < e || m.miss_ids.len() < e
            }
            None => true,
        };
        if needs_alloc {
            // 只 **push 不替换**：旧块可能还有另一张卡的核在写，进程内一律不释放。
            g.push(Slot {
                ctx: ctx_id,
                counts: unsafe { scratch::MappedHost::<u32>::new(k, COUNTS_N)? },
                hit_pri: unsafe { scratch::MappedHost::<i32>::new(k, e.max(1))? },
                hit_alt: unsafe { scratch::MappedHost::<i32>::new(k, e.max(1))? },
                miss_ids: unsafe { scratch::MappedHost::<i32>::new(k, e.max(1))? },
            });
            idx = Some(g.len() - 1);
        }
        let i = idx.unwrap();
        let m = &mut g[i];
        Ok(PairMapped {
            counts_dev: m.counts.dev,
            counts_host: m.counts.slice_mut()?.as_mut_ptr(),
            hit_pri_dev: m.hit_pri.dev,
            hit_pri_host: m.hit_pri.slice_mut()?.as_mut_ptr(),
            hit_alt_dev: m.hit_alt.dev,
            hit_alt_host: m.hit_alt.slice_mut()?.as_mut_ptr(),
            miss_ids_dev: m.miss_ids.dev,
            miss_ids_host: m.miss_ids.slice_mut()?.as_mut_ptr(),
        })
    })
}

/// 跑整层批量设备路由。返回 `Some(out)` 表示**专家段已全部算完**（含残差+shared 的融合
/// 归并已在流上），`None` = 前置条件不满足或两次尝试后仍有未命中（调用方落回 legacy）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_dev_batch_attempt(
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
    rows: usize,
) -> Result<Option<CudaSlice<bf16>>> {
    // rows 上限：默认 16（投机验证批形态）；`LINGXI_ROUTE_FULLDEV_BIG=1` 扩到 512
    // （prefill 形态，）。设备核 rows-敏感性勘察结论：topk 核 grid.x=rows、smem/寄存器
    // 布局只随 E；pair 核单块 512 线程、循环上界 S=rows·top_k（计数 O(E·S) 扫描变
    // 4096 次/线程，~10-20µs/层，可接受）；router 投影 m>6 自动回落行核（grid.y 行分块
    // 无上限）；slotsm 组大小动态（group_off）、a_stride i32 容 s=4096；combine CSR
    // grid.y=rows——**全部无需改 CUDA**。有损档：验收 = 首 token 链 + md5 链（先例 PLAN_NEXT）。
    let big = crate::flag_val("LINGXI_ROUTE_FULLDEV_BIG", true);
    let rows_max = if big { 512 } else { 16 };
    if rows < 2 || rows > rows_max || num_experts > 512 {
        return Ok(None);
    }
    if crate::flag("LINGXI_NO_PAIR_MOE")
        || crate::flag("LINGXI_MOE_DEBUG")
        || crate::flag("LINGXI_MOE_DUMP")
        || crate::flag("LINGXI_MOE_OUT_DUMP")
    {
        return Ok(None);
    }
    let dv = match src.q4_dev() {
        Some(v) => v,
        None => return Ok(None),
    };
    let mixed = dv.alt.is_some();
    let s = rows * top_k;
    let l0 = layer as usize * num_experts;

    // ── 1) router 投影 + 设备 top-k（与 legacy rows>1 同核同参）──
    let ht = htime::on();
    let dev = k.ctx.ordinal();
    // ★：`LINGXI_ROUTE_FULLDEV_BIG=1`（prefill 形态）时，层内 scratch 走
    // `scratch::*_keep` 常驻复用——bench/serve 入口的全局 `disable_pool_forever`
    // 下原本每层 ~110MB 直连 VMM 分配/释放全暴露在 pair 核后的 GPU 空窗（+1.35s
    // 的主嫌疑）；池键已含设备号，split 安全。`fo` 要移交所有权给调用方，**不**走 keep。
    // BIG 关（含 rows≤16 已验证档）时下方宏全部退化为原 `scratch::*`，一字不动。
    macro_rules! tk_bf16 {
        ($n:expr) => { if big { scratch::bf16_keep(k, $n) } else { scratch::bf16(k, $n) } };
    }
    macro_rules! tk_f32 {
        ($n:expr) => { if big { scratch::f32_keep(k, $n) } else { scratch::f32(k, $n) } };
    }
    macro_rules! tk_i32 {
        ($n:expr) => { if big { scratch::i32_keep(k, $n) } else { scratch::i32(k, $n) } };
    }
    macro_rules! tk_u8 {
        ($n:expr) => { if big { scratch::u8_keep(k, $n) } else { scratch::u8(k, $n) } };
    }
    macro_rules! pt_bf16 {
        ($b:expr) => { if big { scratch::put_bf16_keep($b) } else { scratch::put_bf16($b) } };
    }
    macro_rules! pt_f32 {
        ($b:expr) => { if big { scratch::put_f32_keep($b) } else { scratch::put_f32($b) } };
    }
    macro_rules! pt_i32 {
        ($b:expr) => { if big { scratch::put_i32_keep($b) } else { scratch::put_i32($b) } };
    }
    macro_rules! pt_u8 {
        ($b:expr) => { if big { scratch::put_u8_keep($b) } else { scratch::put_u8($b) } };
    }
    let t_call = std::time::Instant::now();
    let t_rt = std::time::Instant::now();
    // ★ 2026-10-24 router 投影：m ≤ 6 时走**编译期特化核**（`router_m6.cu`，与行核逐位相同；
    // 微基准 `t20`：同 grid/block 的纯读 4.7 µs vs 行核 16.0 µs，差在「每轮对 6 个 token 行
    // 各转一遍 w」的 CVT/FFMA 发射）。`LINGXI_NO_ROUTER_M6=1` 回退（wrapper 内门控），
    // m > 6 自动回退到 `gemm_skinny_row_bf16`。
    let mut logits = tk_bf16!(rows * num_experts)?;
    let router_w = w.router_shared_gate.slice(0..num_experts * hidden);
    {
        let (wp, _g1) = router_w.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        let (yp, _g3) = logits.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_m6_bf16_ptr(wp, xp, yp, num_experts, hidden, rows, 0, 0.0, 0.0)?;
    }
    sync_mark(k, &format!("router-gemm L{layer} rows{rows} s{s}"))?;
    super::s3::mark(k, layer, 2)?;
    let mut ids = tk_i32!(s)?;
    let mut wts = tk_f32!(s)?;
    // ★ 2026-10-24 **top-k 与 pair 建表合成一发**（`route_fuse.cu`，逐位相同）。**默认关**
    // （`LINGXI_TKPAIR_FUSE=1` 才开）——实测是负结果：top-k 要靠「一行一 block」的行间
    // 并行，塞进单 block 后 6 行只剩 6 个 warp ⇒ 合成核 27.5 µs > 两发 25.5 µs
    // （连链 37.7 → 41.7 µs）。留作负结果与「末块续算」式融合的基线。
    // 段账口径（`LINGXI_SEG3_SYNC`）：合成时整核落在 **pair** 段里 ⇒ `topk` 列 ≈ 0。
    let fused = crate::flag("LINGXI_TKPAIR_FUSE")
        && num_experts <= 512
        && rows <= 16
        && top_k <= 64;
    // ★ 处方 A（**乐观发射**，仅 BIG=1）：rows>16 且**全视图都走 TCP 档**时，
    // pair 建表核入队后不自旋，按组数上界 min(E, s) 立即发射整个专家段（dyn 核的
    // 组数守卫/行块循环全在设备侧，见 q4_tcp.cu / q3_tcp.cu 的 `_dyn` 档与 pair 核的
    // 尾部清零）——CPU 不再钉在 pair 完成上。
    // ⚠ **实测负结果（保留档注记）**：双 V100 测试机 12 对成对 A/B（开/关交错）干净档位墙钟无差异
    // （开 10.2~10.4 / 关 10.1~10.4，均众数 ~10.3s；decode md5 开=关=BIG0 逐位同）——
    // 估的 pair 0.41ms×432 ≈ 0.18s 是**串行 GPU 时间本身**不是发射空窗（CPU 层均
    // ~1ms 对 GPU 层均 ~25ms 始终领先 ≥20ms，spin 只节流 CPU 不制造 GPU 空窗）。
    // 保留为 `LINGXI_ROUTE_OPT=1` 显式开启的实验路径（默认关 = 行为一字不变）。
    let opt_ok = big && rows > 16
        && crate::flag("LINGXI_ROUTE_OPT")
        && crate::flag("LINGXI_MOE_TC") && hidden % 64 == 0 && inter % 64 == 0;
    if opt_ok {
        let pri_q3_gate = dv.consts.bits == 3;
        let pri_tcp = !pri_q3_gate
            || (dv.consts.gu_k_pad % 128 == 0 && dv.consts.dn_k_pad % 128 == 0
                && !crate::flag("LINGXI_NO_Q3TCP"));
        let alt_tcp = dv.alt.as_ref().map(|a| {
            a.consts.gu_k_pad % 128 == 0 && a.consts.dn_k_pad % 128 == 0
                && !crate::flag("LINGXI_NO_Q3TCP")
        }).unwrap_or(true);
        if pri_tcp && alt_tcp {
            return route_dev_batch_attempt_opt(
                k, layer, w, x, shared, src, num_experts, top_k, inter, hidden, rows,
            );
        }
    }
    if !fused {
        // 默认仍是旧核。2026-10-24 试过「一 warp 一行」版（`route_fuse.cu` 的
        // `lingxi_route_topk_row_warp_bf16`，逐位相同）：微基准 15.5 µs > 13.4 µs
        // —— top-k 的净活主要是**逐元素 expf/建键**，6 行 × 256 线程的并行度
        // （每线程 2 个元素）比 6 × 32（每线程 16 个元素）划算 ⇒ 负结果，未接生产。
        k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, rows, num_experts, top_k)?;
    }
    let t_after_topk = std::time::Instant::now();
    let us_rt = t_after_topk.duration_since(t_rt).as_secs_f64() * 1e6;
    if ht {
        htime::add(dev, 0, t_rt);
    }
    if !fused {
        sync_mark(k, &format!("topk L{layer} rows{rows} s{s} E{num_experts} k{top_k} logits_len{} ids_len{} wts_len{}",
            logits.len(), ids.len(), wts.len()))?;
    }
    // ⚠️ 段账口径（`LINGXI_SEG3_SYNC`）：合成核整个落在 **pair** 段里 ⇒ fused 时
    // `topk` 列 ≈ 0、`pair` 列 = topk+pair 合计。逐段 A/B 请看微基准
    // （`lingxi-kernels/tests/t20_route_chain_bw.rs`）。
    super::s3::mark(k, layer, 3)?;

    // ── 2) pair/组/CSR 构建（设备；最多两趟：未命中先补槽再重跑）──
    let mut pair_rows = tk_i32!(s)?;
    let mut pair_wts = tk_f32!(s)?;
    let mut group_slots = tk_i32!(num_experts)?;
    let mut group_slots_alt = tk_i32!(num_experts)?;
    let mut group_off = tk_i32!(num_experts + 1)?;
    // 按格式压缩的**组下标表**（建表核顺手写，零额外 launch）：`pri_gidx[k]`/`alt_gidx[k]`
    // 是该格式第 k 组的全表组下标（升序）。GEMV 用它们间接取 group_off/slot_ids ⇒ 真实组
    // 读到的三元组逐位不变，只是网格不再覆盖对侧格式的空组。
    let mut pri_gidx = tk_i32!(num_experts)?;
    let mut alt_gidx = tk_i32!(num_experts)?;
    let mut uniq = tk_i32!(num_experts)?;
    let mut col = tk_i32!(s)?;
    let mut roff = tk_i32!(rows + 1)?;
    let m = pair_mapped(k, num_experts)?;
    let map_pri_s = dv.slot_map.slice(l0..l0 + num_experts);
    let map_alt_s = dv.alt.as_ref().map(|a| a.slot_map.slice(l0..l0 + num_experts));
    let fmt_s = dv.fmt.map(|f| f.slice(l0..l0 + num_experts));
    let t_pre_pairs = std::time::Instant::now();
    let mut counts: [u32; COUNTS_N] = [0; COUNTS_N];
    let mut us_prep = 0.0f64;
    let mut us_launch = 0.0f64;
    let mut us_spin = 0.0f64;
    let mut us_read = 0.0f64;
    // `LINGXI_PREP_PROF=1`：把 prep（6 次 `device_ptr` + 1 次 volatile 写）**逐步**计时。
    // 背景：`[htime-spike]` 的 prep 加起来 1192 层 132 ms ⇒ **4.4 ms/轮**，
    // 而这段只有 5~6 次 `device_ptr` —— 物理上讲不通，必须量到具体是哪一步。
    let pp = crate::flag("LINGXI_PREP_PROF");
    let mut us = [0f64; 8];
    let mut tries = 0usize;
    for attempt in 0..2 {
        tries += 1;
        let mut tp0 = std::time::Instant::now();
        macro_rules! lap0 {
            ($i:expr) => {
                if pp {
                    let now = std::time::Instant::now();
                    us[$i] = now.duration_since(tp0).as_secs_f64() * 1e6;
                    tp0 = now;
                }
            };
        }
        unsafe { std::ptr::write_volatile(m.counts_host.add(2), 0u32) };
        lap0!(0);
        // 指针守卫必须活到 launch 之后（`device_ptr` 借用切片；guard 在同一作用域内）。
        let (map_pri_p, _gm) = map_pri_s.device_ptr(&k.stream);
        lap0!(1);
        let (map_alt_p, fmt_p, _ga, _gf) = match (map_alt_s.as_ref(), fmt_s.as_ref()) {
            (Some(a), Some(f)) => {
                let (p, ga) = a.device_ptr(&k.stream);
                let (q, gf) = f.device_ptr(&k.stream);
                (p, q, Some(ga), Some(gf))
            }
            _ => (0u64, 0u64, None, None),
        };
        lap0!(2);
        let (gs_alt_dev, _gsa) = group_slots_alt.device_ptr(&k.stream);
        lap0!(3);
        let t_p = std::time::Instant::now();
        // 合成路径（`LINGXI_TKPAIR_FUSE=1`，默认关）：第 0 趟发**一发**（top-k + 建表），
        // 第 1 趟（补槽后重跑）只重建表（ids/wts 由第 0 趟算出、确定不变）。
        if fused && attempt == 0 {
            k.route_topk_pairs_bf16(
                &logits, &mut ids, &mut wts, map_pri_p, map_alt_p, fmt_p,
                &mut pair_rows, &mut pair_wts, &mut group_slots, gs_alt_dev,
                &mut group_off, &mut uniq, &mut col, &mut roff,
                m.counts_dev, m.hit_pri_dev, m.hit_alt_dev, m.miss_ids_dev,
                &mut pri_gidx, &mut alt_gidx,
                num_experts, rows, top_k,
            )?;
        } else {
            k.route_pairs_bf16(
                &ids, &wts, map_pri_p, map_alt_p, fmt_p,
                &mut pair_rows, &mut pair_wts, &mut group_slots, gs_alt_dev,
                &mut group_off, &mut uniq, &mut col, &mut roff,
                m.counts_dev, m.hit_pri_dev, m.hit_alt_dev, m.miss_ids_dev,
                &mut pri_gidx, &mut alt_gidx,
                num_experts, rows, top_k,
            )?;
        }
        us_prep = t_p.duration_since(t_pre_pairs).as_secs_f64() * 1e6;
        us_launch = t_p.elapsed().as_secs_f64() * 1e6;
        if ht {
            htime::add(dev, 1, t_p);
        }
        sync_mark(k, "pairs")?;
        let t_s = std::time::Instant::now();
        route_spin(k, false)?;
        us_spin = t_s.elapsed().as_secs_f64() * 1e6;
        if ht {
            // ★ 2026-10-23 修：此前**从没累加过自旋项**（只 add 了 0/1/3），所以汇总行永远打
            // 「自旋 0.0」——那是探针漏计，不是实测为 0。实测自旋常在 300~900 µs。
            htime::add(dev, 2, t_s);
        }
        let t_c = std::time::Instant::now();
        for (i, c) in counts.iter_mut().enumerate() {
            *c = unsafe { std::ptr::read_volatile(m.counts_host.add(i)) };
        }
        us_read = t_c.elapsed().as_secs_f64() * 1e6;
        if ht {
            htime::add(dev, 3, t_c);
            htime::tick(dev, layer);
        }
        let nmiss = counts[2];
        super::s3::mark(k, layer, 4)?;
        if nmiss == 0 {
            break;
        }
        if attempt == 1 {
            // 两趟仍缺 ⇒ 落回 legacy（本函数写过的缓冲/槽位表都无副作用）。
            return Ok(None);
        }
        let miss_list: Vec<u32> = (0..nmiss as usize)
            .map(|j| unsafe { std::ptr::read_volatile(m.miss_ids_host.add(j)) } as u32)
            .collect();
        let prot_pri: Vec<usize> = (0..counts[4] as usize)
            .map(|j| unsafe { std::ptr::read_volatile(m.hit_pri_host.add(j)) } as usize)
            .collect();
        let prot_alt: Vec<usize> = (0..counts[5] as usize)
            .map(|j| unsafe { std::ptr::read_volatile(m.hit_alt_host.add(j)) } as usize)
            .collect();
        src.load_missing(layer, &miss_list, &prot_pri, &prot_alt)?;
    }
    let u = counts[1] as usize;
    let tmax = counts[3] as usize;
    // 诊断（`LINGXI_FULLDEV_CHK=1`）：确认 router logits 里到底有没有非有限值（NaN/inf 会让
    // top-k 少选 ⇒ ids 尾部留池内残渣 ⇒ 后续 pair 核按 ids 查 slot_map 越界）。
    // ⚠️ 2026-10-24：从 top-k 之后**挪到建表之后**——合成一发时 ids/wts 是建表那一次 launch
    // 的产物，挪到此处两条路径都能查（检查内容一字未改）。
    if crate::flag("LINGXI_FULLDEV_CHK") {
        let lh: Vec<bf16> = k.stream.clone_dtoh(&logits)?;
        let ih: Vec<i32> = k.stream.clone_dtoh(&ids)?;
        let wh: Vec<f32> = k.stream.clone_dtoh(&wts)?;
        let mut bad = 0usize;
        let mut worst = 0f32;
        for (i, v) in lh.iter().enumerate().take(rows * num_experts) {
            let f = v.to_f32();
            if f.is_finite() {
                worst = worst.max(f.abs());
            } else {
                if bad < 3 {
                    eprintln!("[chk] L{layer} row{} idx{} val={f}", i / num_experts, i % num_experts);
                }
                bad += 1;
            }
        }
        let ibad = ih.iter().take(s).filter(|&&v| v < 0 || v as usize >= num_experts).count();
        let wbad = wh.iter().take(s).filter(|v| !v.is_finite()).count();
        eprintln!(
            "[chk] L{layer} rows{rows} nonfinite_logits={bad} max|logit|={worst:.3} bad_ids={ibad} bad_wts={wbad}"
        );
    }
    // ★ 2026-10-24 组压缩：每个视图只发**自己格式的组**（`ng = npri`/`nalt`），不再按完整 `u`
    // 发再让对侧组 `slot_ids[e] < 0` 空跑。`LINGXI_NO_GROUP_COMPACT=1` 回退旧行为。
    // 逐位不变的依据：核内按 `gidx[e]` 间接取的 `group_off[gi]`/`group_off[gi+1]`/`slot_ids[gi]`
    // 与全 `u` 网格下该组读到的三元组逐项相同 —— 被删掉的只是"算不出任何东西"的空组。
    let compact = !crate::flag("LINGXI_NO_GROUP_COMPACT");
    let npri = counts[6] as usize;
    let nalt = counts[7] as usize;
    // dp4a 与设备全路由**共存**：旧版这里把 `LINGXI_MOE_DP4A` 直接退回 legacy，是因为设备路径
    // 只接了 bf16 GEMV、没有把 gather 后的 `xs` 量化成 int8 —— 不是 pair/哨兵格式不兼容。
    // 现在在同一份 `xs`/`act_all` 上补 `convert_bf16_q8_rs`（与 legacy 同核同参）即可。
    // 约束：dp4a 核只有 8 宽 acc ⇒ 需要 `tmax ≤ 8`；混档（alt 视图）与 legacy 一样不接 dp4a。
    // ★：rows>16（prefill）+ `LINGXI_MOE_TC`（start.sh 默认 1）⇒ 专家段与 legacy
    // prefill **同款 TCP 大 tile 张量核**（有损档 ~2^-11，判据同 q4_tcp：首 token 相等 +
    // 质量抽检）。缺了这一档时 SIMT slotsm 在 s=5120 上实测 3.5× 慢（gu_q3 10.3ms vs
    // legacy TCP 合计 3.2ms），是 FULLDEV_BIG 第一版 prefill +4s 的全部差额。
    // TCP 核无组压缩（gidx）入口 ⇒ 按**全表 u** 发网格、对侧格式组 slot=-1 早退
    // （与 legacy 混档双视图口径一致）；tmax 用全局值（grid.z 分块上界，≥每视图真值）。
    let tcp_outer = crate::flag("LINGXI_MOE_TC") && rows > 16
        && hidden % 64 == 0 && inter % 64 == 0;
    let dp4a_on = crate::flag("LINGXI_MOE_DP4A") && !mixed && tmax <= 8 && !tcp_outer;
    // 对拍（`LINGXI_PAIR_DUMP=1`）：把设备构建的 pair/CSR/组表拉回 host，与 legacy 的同名表逐项比。
    // 目的：把「1 ulp 差」钉到**路由表次序**还是**专家段算式**（2026-10-23，TC 开/关 diff 完全相同 ⇒ 非 TC）。
    if crate::flag("LINGXI_PAIR_DUMP") {
        let rh: Vec<i32> = k.stream.clone_dtoh(&pair_rows)?;
        let wh: Vec<f32> = k.stream.clone_dtoh(&pair_wts)?;
        let ch: Vec<i32> = k.stream.clone_dtoh(&col)?;
        let ofh: Vec<i32> = k.stream.clone_dtoh(&roff)?;
        let goh: Vec<i32> = k.stream.clone_dtoh(&group_off)?;
        let gsh: Vec<i32> = k.stream.clone_dtoh(&group_slots)?;
        eprintln!(
            "[pair-dev] L{layer} rows{rows} n{} h{:016x} rows8={:?} wts8={:?}",
            rh.len(),
            super::hash_pair_tables(&rh, &wh),
            &rh[..rh.len().min(8)],
            &wh[..wh.len().min(8)]
        );
        eprintln!(
            "[pair-dev] L{layer} col n{} h{:016x} col8={:?} roff{:?} go{:?} gs8={:?}",
            ch.len(),
            super::hash_i32(&ch) ^ super::hash_i32(&ofh),
            &ch[..ch.len().min(8)],
            &ofh[..ofh.len().min(8)],
            &goh[..goh.len().min(u + 1)],
            &gsh[..gsh.len().min(8)]
        );
        // 组压缩表自检：`pri_gidx`/`alt_gidx` 必须是 0..u 的**分区**（升序、互斥、无漏），
        // 且各自指向的槽号确实是本格式（≥0 / 对侧 -1）。这是"真实组三元组不变"的判据。
        if mixed {
            let ash: Vec<i32> = k.stream.clone_dtoh(&group_slots_alt)?;
            let pg: Vec<i32> = k.stream.clone_dtoh(&pri_gidx)?;
            let ag: Vec<i32> = k.stream.clone_dtoh(&alt_gidx)?;
            let inrange = pg[..npri].iter().chain(ag[..nalt].iter())
                .all(|&g| g >= 0 && (g as usize) < u);
            let (mut partition, mut asc, mut side) = (inrange, inrange, inrange);
            if inrange {
                let mut seen = vec![0u8; u];
                for &g in pg[..npri].iter().chain(ag[..nalt].iter()) {
                    seen[g as usize] += 1;
                }
                partition = seen.iter().all(|&x| x == 1);
                asc = pg[..npri].windows(2).all(|w| w[0] < w[1])
                    && ag[..nalt].windows(2).all(|w| w[0] < w[1]);
                side = pg[..npri].iter().all(|&g| gsh[g as usize] >= 0)
                    && ag[..nalt].iter().all(|&g| ash[g as usize] >= 0);
            }
            eprintln!(
                "[pair-dev] L{layer} 压缩 npri{npri}/nalt{nalt} 分区={partition} 升序={asc} 槽侧={side}"
            );
        }
    }
    if crate::flag("LINGXI_ROUTE_FULLDEV_TRACE") {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 4 || n % 48 == 0 {
            eprintln!(
                "[fulldev] call {n} layer {layer} rows {rows} s {s} u {u} tmax {tmax} mixed {mixed} npair_pri {} npair_alt {} → 网格 pri {npri}/alt {nalt}（compact={compact}）",
                counts[4], counts[5]
            );
        }
    }
    if u == 0 {
        return Ok(None);
    }
    // `LINGXI_FULLDEV_HTIME=1`：主机侧总耗时 >250µs 的层逐条打出分解（定位 s3 尖峰）。
    if pp && us_prep > 50.0 {
        eprintln!(
            "[prep-prof] dev{dev} L{layer} prep {us_prep:.0}µs（attempt={tries}）：volatile写 {:.0}／\
             pri_ptr {:.0}／alt+fmt {:.0}／gs_alt {:.0}",
            us[0], us[1], us[2], us[3]
        );
    }
    if ht {
        let tot = t_call.elapsed().as_secs_f64() * 1e6;
        if tot > 250.0 {
            let us_alloc = t_pre_pairs.duration_since(t_after_topk).as_secs_f64() * 1e6;
            let rest = tot - (us_rt + us_alloc + us_prep + us_launch + us_spin + us_read);
            eprintln!(
                "[htime-spike] dev{dev} L{layer} rows{rows} 总{tot:.0}µs：router+topk {us_rt:.0}／alloc {us_alloc:.0}／prep {us_prep:.0}／pairs {us_launch:.0}／自旋 {us_spin:.0}／读 {us_read:.0}／其余 {rest:.0}"
            );
        }
    }
    // evprof：段 3（路由+host）到此为止，段 4（专家 GEMV+归并）开始（与 legacy 同口径）。
    if let Some(p) = k.evprof.as_ref() {
        p.end(layer, 3, &k.stream);
        p.begin(layer, 4, &k.stream);
        p.set_info(layer, 0, s as u32);
        p.set_info(layer, 1, u as u32);
        p.set_info(layer, 2, rows as u32);
    }
    s4::info(s, u, rows, counts[4], counts[5]);
    s4::mark(k, layer, 0)?;

    // ── 3) gather（按 pair 行号取 x；与 legacy 同核同参）──
    let mut xs = tk_bf16!(s * hidden)?;
    k.gather_rows_bf16(x, &pair_rows, &mut xs, s, hidden, 0)?;
    sync_mark(k, "gather")?;
    s4::mark(k, layer, 1)?;
    let mut gu_all = tk_bf16!(s * 2 * inter)?;
    let mut act_all = tk_bf16!(s * inter)?;
    let mut o_all = tk_bf16!(s * hidden)?;
    // ★：rows>16（prefill 形态，FULLDEV_BIG）用 legacy 批量同款 `slotsm_rpw`（默认 2）——
    // batch_dev 历史上沿用 decode 的 `slots_rpw`（默认 1），prefill 形状实测慢 ~2×（专家段
    // k 窄、每 warp 行数翻倍摊访存延迟，判据见 forward.rs 的 slotsm_rpw 注记）；rows≤16 档
    // 一字不动（保持已验证行为）。
    let sm_rpw = if rows > 16 { slotsm_rpw() } else { slots_rpw() };
    // Q3 t2 档的 `MB=1` 实例要求**每组恰好 1 对**：`tmax` 是本次 launch 的最大组大小 ⇒ `tmax == 1`
    // 正是该条件（rows=1 的 decode 恒成立）。t2 核与旧核逐位相同，只换取数结构（`LINGXI_NO_Q3_T2=1` 回退）。
    let q3_all_m1 = tmax == 1;
    let gu_stride = hidden;
    // legacy：混档时 xf 预转关（`!dual_q4`）；单格式 rows≤16 时开。逐位等价，仅影响速度。
    // dp4a 档不用 fp32 舞台（它需要的是 int8 量化行），与 legacy 一致关掉。
    let xf_on = !crate::flag("LINGXI_MOE_NO_XF32") && !mixed && !dp4a_on;
    let mut xf_gu = tk_f32!(s * hidden)?;
    let mut xf_gu_p = 0u64;
    let mut g_gu = None;
    if xf_on {
        k.convert_bf16_f32(&xs, &mut xf_gu, s * hidden)?;
        let (p, g) = xf_gu.device_ptr(&k.stream);
        xf_gu_p = p;
        g_gu = Some(g);
    }
    // dp4a：把 gather 后的激活按行量化（每行一个 scale），与 legacy 同核同参。
    let mut xq_gu = if dp4a_on { Some(tk_u8!(s * hidden)?) } else { None };
    let mut sx_gu = if dp4a_on { Some(tk_f32!(s)?) } else { None };
    if let (Some(xq), Some(sx)) = (xq_gu.as_mut(), sx_gu.as_mut()) {
        k.convert_bf16_q8_rs(&xs, xq, sx, s, hidden)?;
    }
    s4::mark(k, layer, 2)?;
    // 视图表：(slots, ssel, gsel, consts, q3, 本视图的组数 ng, gidx 设备指针)；
    // 单格式只有主视图。默认 `ng = npri/nalt` + `gidx` 间接索引（对侧格式的组不发块）；
    // `LINGXI_NO_GROUP_COMPACT=1` 回退成 `ng = u`、`gidx = 0`（改动前的行为，用于 A/B 与兜底）。
    // `_slot`/`_off` 始终是**全表**（紧凑只体现在 gidx 与网格 y 上）。
    struct ViewSel<'a> {
        slots: &'a CudaSlice<u8>,
        ssel: cudarc::driver::CudaView<'a, i32>,
        gsel: cudarc::driver::CudaView<'a, i32>,
        consts: Q4Consts,
        q3: bool,
        ng: usize,
        gidx: u64,
    }
    let gsel_full = group_off.slice(0..=u);
    let (pri_gsel, alt_gsel) = (gsel_full, group_off.slice(0..=u));
    let ssel_pri = group_slots.slice(0..u);
    let ssel_alt = group_slots_alt.slice(0..u);
    let (gp_p, _gp_g) = pri_gidx.device_ptr(&k.stream);
    let (gp_a, _ga_g) = alt_gidx.device_ptr(&k.stream);
    let mut views: Vec<ViewSel> = Vec::new();
    // ★ 2026-10-25：主视图的格式**不能写死 Q4**。单格式全 Q3 arena（`q3g128.bin` 等）的
    // 主格式就是 Q3G128，旧代码把它按 Q4 核跑 ⇒ 直接吐垃圾（实测输出一长串 `!!!!`、
    // 接受率 0、被关闸 ① 打回 AR）。判据用 `Q4Consts.bits`（4=Q4G64 / 3=Q3G128）。
    let pri_q3 = dv.consts.bits == 3;
    if compact {
        views.push(ViewSel { slots: dv.slots, ssel: ssel_pri, gsel: pri_gsel, consts: dv.consts, q3: pri_q3, ng: npri, gidx: gp_p });
        if let Some(alt) = dv.alt.as_ref() {
            views.push(ViewSel { slots: alt.slots, ssel: ssel_alt, gsel: alt_gsel, consts: alt.consts, q3: true, ng: nalt, gidx: gp_a });
        }
    } else {
        views.push(ViewSel { slots: dv.slots, ssel: ssel_pri, gsel: pri_gsel, consts: dv.consts, q3: pri_q3, ng: u, gidx: 0 });
        if let Some(alt) = dv.alt.as_ref() {
            views.push(ViewSel { slots: alt.slots, ssel: ssel_alt, gsel: alt_gsel, consts: alt.consts, q3: true, ng: u, gidx: 0 });
        }
    }
    let tc_flag = crate::flag("LINGXI_MOE_TC") && hidden % 64 == 0 && tmax <= 16;
    let tiles = if tmax > 8 { 2 } else { 1 };
    {
        // ── 4) gu（每视图一次 grouped launch）──
        for v in views.iter() {
            let (slots, ssel, gsel, c, q3, ng, gidx) =
                (v.slots, &v.ssel, &v.gsel, v.consts, v.q3, v.ng, v.gidx);
            if ng == 0 {
                continue;
            }
            let tcp_on = (tcp_outer && !q3)
                || (q3 && tcp_outer
                    && c.gu_k_pad % 128 == 0 && c.dn_k_pad % 128 == 0
                    && !crate::flag("LINGXI_NO_Q3TCP"));
            if tcp_on {
                if q3 {
                    k.gemv_q3g128_slots_tcp_bf16_view(
                        slots, ssel, gsel, &xs, &mut gu_all, u, 2 * inter, hidden,
                        gu_stride, c.gu_k_pad, c.blob, c.gu_seg, c.gu_scale, tmax,
                    )?;
                } else {
                    k.gemv_q4g64_slots_tcp_bf16_view(
                        slots, ssel, gsel, &xs, &mut gu_all, u, 2 * inter, hidden,
                        gu_stride, c.gu_k_pad, c.blob, c.gu_seg, c.gu_scale, tmax,
                    )?;
                }
            } else if tc_flag && !q3 {
                k.gemv_q4g64_slots_tc_bf16_gidx(
                    slots, ssel, gsel, &xs, &mut gu_all, ng, 2 * inter, hidden, gu_stride,
                    c.gu_k_pad, c.blob, c.gu_seg, c.gu_scale, tiles, gidx,
                )?;
            } else if q3 {
                k.gemv_q3g128_slotsm_t2_view_gidx(
                    slots, ssel, &xs, gsel, &mut gu_all, ng, 2 * inter, hidden, c.gu_k_pad,
                    gu_stride, c.blob, c.gu_seg, c.gu_scale, sm_rpw, xf_gu_p, gidx, q3_all_m1,
                )?;
            } else if dp4a_on {
                k.gemv_q4g64_slotsm_rpw_dp4a_view_gidx(
                    slots, ssel, &xs, gsel, &mut gu_all, ng, 2 * inter, hidden, c.gu_k_pad,
                    gu_stride, c.blob, c.gu_seg, c.gu_scale, sm_rpw,
                    xq_gu.as_ref().unwrap(), sx_gu.as_ref().unwrap(), gidx,
                )?;
            } else {
                k.gemv_q4g64_slotsm_rpw_bf16_view_gidx(
                    slots, ssel, &xs, gsel, &mut gu_all, ng, 2 * inter, hidden, c.gu_k_pad,
                    gu_stride, c.blob, c.gu_seg, c.gu_scale, sm_rpw, xf_gu_p, gidx,
                )?;
            }
            sync_mark(k, if q3 { "gu-q3" } else { "gu-q4" })?;
            s4::mark(k, layer, if q3 { 4 } else { 3 })?;
        }
    }
    k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
    sync_mark(k, "swiglu")?;
    s4::mark(k, layer, 5)?;
    // ── 5) dn（同规则；xf 预转 act_all）──
    let mut xf_dn = tk_f32!(s * inter)?;
    let mut xf_dn_p = 0u64;
    let mut g_dn = None;
    if xf_on {
        k.convert_bf16_f32(&act_all, &mut xf_dn, s * inter)?;
        let (p, g) = xf_dn.device_ptr(&k.stream);
        xf_dn_p = p;
        g_dn = Some(g);
    }
    let mut xq_dn = if dp4a_on { Some(tk_u8!(s * inter)?) } else { None };
    let mut sx_dn = if dp4a_on { Some(tk_f32!(s)?) } else { None };
    if let (Some(xq), Some(sx)) = (xq_dn.as_mut(), sx_dn.as_mut()) {
        k.convert_bf16_q8_rs(&act_all, xq, sx, s, inter)?;
    }
    s4::mark(k, layer, 6)?;
    {
        for v in views.iter() {
            let (slots, ssel, gsel, c, q3, ng, gidx) =
                (v.slots, &v.ssel, &v.gsel, v.consts, v.q3, v.ng, v.gidx);
            if ng == 0 {
                continue;
            }
            // ⚠️ dn **不跟 gu 一样走小档 TC**：legacy 的 dn 段只有 TCP（rows>16）与 SIMT 两档，
            // rows≤16 时恒走 SIMT（forward.rs 的 `slots_tc_bf16` 仅两处调用，都在 gu 分支）。
            // 2026-10-23 对拍实证：dn 也开 TC ⇒ o_all 每层 3~8% 元素差 1 ulp（gu/act 哈希仍一致）。
            // （rows>16 的 TCP 大 tile 档 gu/dn 都走——与 legacy 口径一致，见上。）
            let tcp_on = (tcp_outer && !q3)
                || (q3 && tcp_outer
                    && c.gu_k_pad % 128 == 0 && c.dn_k_pad % 128 == 0
                    && !crate::flag("LINGXI_NO_Q3TCP"));
            if tcp_on {
                if q3 {
                    k.gemv_q3g128_slots_tcp_bf16_view(
                        slots, ssel, gsel, &act_all, &mut o_all, u, hidden, inter,
                        inter, c.dn_k_pad, c.blob, c.dn_seg, c.dn_scale, tmax,
                    )?;
                } else {
                    k.gemv_q4g64_slots_tcp_bf16_view(
                        slots, ssel, gsel, &act_all, &mut o_all, u, hidden, inter,
                        inter, c.dn_k_pad, c.blob, c.dn_seg, c.dn_scale, tmax,
                    )?;
                }
            } else if q3 {
                k.gemv_q3g128_slotsm_t2_view_gidx(
                    slots, ssel, &act_all, gsel, &mut o_all, ng, hidden, inter, c.dn_k_pad,
                    inter, c.blob, c.dn_seg, c.dn_scale, sm_rpw, xf_dn_p, gidx, q3_all_m1,
                )?;
            } else if dp4a_on {
                k.gemv_q4g64_slotsm_rpw_dp4a_view_gidx(
                    slots, ssel, &act_all, gsel, &mut o_all, ng, hidden, inter, c.dn_k_pad,
                    inter, c.blob, c.dn_seg, c.dn_scale, sm_rpw,
                    xq_dn.as_ref().unwrap(), sx_dn.as_ref().unwrap(), gidx,
                )?;
            } else {
                k.gemv_q4g64_slotsm_rpw_bf16_view_gidx(
                    slots, ssel, &act_all, gsel, &mut o_all, ng, hidden, inter, c.dn_k_pad,
                    inter, c.blob, c.dn_seg, c.dn_scale, sm_rpw, xf_dn_p, gidx,
                )?;
            }
            sync_mark(k, if q3 { "dn-q3" } else { "dn-q4" })?;
            s4::mark(k, layer, if q3 { 8 } else { 7 })?;
        }
    }
    // ── 6) 按行 CSR 归并 + shared 残差（与 legacy 同核）──
    let mut fo = scratch::bf16(k, rows * hidden)?;
    k.moe_combine_add_csr_bf16(&o_all, &col, &roff, &pair_wts, shared, &mut fo, rows, hidden)?;
    sync_mark(k, "combine")?;
    s4::mark(k, layer, 9)?;
    // 对拍（与 legacy 的 `LINGXI_MOE_OUT_DUMP` 同口径）：三段缓冲的位哈希 + 和 + NaN 数，
    // 用来把差异钉在 gu / act(swiglu) / dn(o_all) 哪一段。
    if crate::flag("LINGXI_FULLDEV_OUT_DUMP") {
        let want: u32 = std::env::var("LINGXI_MOE_OUT_DUMP_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(|v: u32| v.wrapping_sub(1))
            .unwrap_or(crate::DIMS.n_layers as u32 - 1);
        if layer == want {
            for (nm, buf) in [("gu_all", &gu_all), ("act_all", &act_all), ("o_all", &o_all)] {
                let h: Vec<bf16> = k.stream.clone_dtoh(buf)?;
                let tot: f64 = h.iter().map(|v| v.to_f32() as f64).sum();
                let nans = h.iter().filter(|v| v.is_nan()).count();
                let mut bits: u64 = 0xcbf29ce484222325;
                for v in h.iter() {
                    bits = (bits ^ (v.to_bits() as u64)).wrapping_mul(0x100000001b3);
                }
                eprintln!(
                    "[fulldev-out] 层{layer} {nm} 全和={tot:.6} n={} nan={nans} h{bits:016x}",
                    h.len()
                );
            }
        }
    }
    if let Some(p) = k.evprof.as_ref() {
        p.end(layer, 4, &k.stream);
    }

    drop(views);
    drop(_gp_g);
    drop(_ga_g);
    drop(g_gu);
    drop(g_dn);
    pt_bf16!(logits);
    pt_i32!(ids);
    pt_f32!(wts);
    pt_i32!(pair_rows);
    pt_f32!(pair_wts);
    pt_i32!(group_slots);
    pt_i32!(group_slots_alt);
    pt_i32!(group_off);
    pt_i32!(pri_gidx);
    pt_i32!(alt_gidx);
    pt_i32!(uniq);
    pt_i32!(col);
    pt_i32!(roff);
    pt_bf16!(xs);
    pt_bf16!(gu_all);
    pt_bf16!(act_all);
    pt_bf16!(o_all);
    pt_f32!(xf_gu);
    pt_f32!(xf_dn);
    if let Some(b) = xq_gu {
        pt_u8!(b);
    }
    if let Some(b) = sx_gu {
        pt_f32!(b);
    }
    if let Some(b) = xq_dn {
        pt_u8!(b);
    }
    if let Some(b) = sx_dn {
        pt_f32!(b);
    }
    Ok(Some(fo))
}

/// 处方 A：**pair 核组数上界乐观发射**（仅 `LINGXI_ROUTE_FULLDEV_BIG=1` 且全视图
/// TCP 档由 [`route_dev_batch_attempt`] 的 gate 分流进来）。
///
/// 与默认路径的差异只有**发射时序**：pair 建表核入队后**不自旋**，立即按 host 上界
/// `ng_ub = min(E, s)` 发射 gather → 预转 → gu → swiglu → 预转 → dn → combine；
/// `*_tcp_bf16_dyn` 核的组数守卫（group_off 尾部清零 ⇒ t<=0 早退）与行块循环
/// （按本组大小，免 host 的 tmax）全在设备侧，同流序保证读到建成后的表。全部发射完
/// 才自旋读计数——CPU 不再钉在 pair 完成上， 残留的 post-spin ~10 发 launch 空窗
/// 被 GPU 流水吃掉。逐位性：dyn 核与旧 TCP 核每 (组, 行块) 的算术逐字相同（同一
/// body），其余核同核同参；网格只多"算不出任何东西"的空块。
///
/// miss 重试 = 补槽后 pair + 整段**重发一遍**（流序覆盖，同输入同表 ⇒ 同结果；
/// 稳态 prefill miss 罕见，浪费一层段 GPU）。两次仍 miss / u==0 ⇒ `None` 落回
/// legacy（本函数写过的设备缓冲全是 scratch/keep 池，无副作用）。
#[allow(clippy::too_many_arguments)]
fn route_dev_batch_attempt_opt(
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
    rows: usize,
) -> Result<Option<CudaSlice<bf16>>> {
    let dv = src.q4_dev().unwrap();
    let mixed = dv.alt.is_some();
    let s = rows * top_k;
    let l0 = layer as usize * num_experts;
    let ht = htime::on();
    let dev = k.ctx.ordinal();
    let t_call = std::time::Instant::now();
    let t_rt = std::time::Instant::now();
    // ── 1) router 投影 + 设备 top-k（与非乐观路径同核同参，见该路径注记）──
    let mut logits = scratch::bf16_keep(k, rows * num_experts)?;
    let router_w = w.router_shared_gate.slice(0..num_experts * hidden);
    {
        let (wp, _g1) = router_w.device_ptr(&k.stream);
        let (xp, _g2) = x.device_ptr(&k.stream);
        let (yp, _g3) = logits.device_ptr_mut(&k.stream);
        k.gemm_skinny_row_m6_bf16_ptr(wp, xp, yp, num_experts, hidden, rows, 0, 0.0, 0.0)?;
    }
    sync_mark(k, &format!("router-gemm L{layer} rows{rows} s{s}"))?;
    super::s3::mark(k, layer, 2)?;
    let mut ids = scratch::i32_keep(k, s)?;
    let mut wts = scratch::f32_keep(k, s)?;
    k.route_topk_rows_bf16(&logits, &mut ids, &mut wts, rows, num_experts, top_k)?;
    if ht {
        htime::add(dev, 0, t_rt);
    }
    sync_mark(k, &format!("topk L{layer} rows{rows} s{s} E{num_experts} k{top_k}"))?;
    super::s3::mark(k, layer, 3)?;

    // ── 2) pair/组/CSR 建表缓冲（与默认路径同池同形状；计数无关，循环外一次 take）──
    let mut pair_rows = scratch::i32_keep(k, s)?;
    let mut pair_wts = scratch::f32_keep(k, s)?;
    let mut group_slots = scratch::i32_keep(k, num_experts)?;
    let mut group_slots_alt = scratch::i32_keep(k, num_experts)?;
    let mut group_off = scratch::i32_keep(k, num_experts + 1)?;
    let mut col = scratch::i32_keep(k, s)?;
    let mut roff = scratch::i32_keep(k, rows + 1)?;
    // 建表核形参要求的伴随表（本路径不读，仅为保持核签名一致；keep 池常驻）。
    let mut uniq = scratch::i32_keep(k, num_experts)?;
    let mut pri_gidx = scratch::i32_keep(k, num_experts)?;
    let mut alt_gidx = scratch::i32_keep(k, num_experts)?;
    let m = pair_mapped(k, num_experts)?;
    let map_pri_s = dv.slot_map.slice(l0..l0 + num_experts);
    let map_alt_s = dv.alt.as_ref().map(|a| a.slot_map.slice(l0..l0 + num_experts));
    let fmt_s = dv.fmt.map(|f| f.slice(l0..l0 + num_experts));
    // 乐观网格的组数上界：u ≤ min(E, s)（每组 ≥1 对、每专家 ≤1 组）。
    let ng_ub = num_experts.min(s);
    let mut counts: [u32; COUNTS_N] = [0; COUNTS_N];

    // ── 3) 乐观发射 + 自旋读计数（miss 时补槽重发整段，最多两趟）──
    let mut fo_opt: Option<CudaSlice<bf16>> = None;
    for attempt in 0..2 {
        if attempt == 1 {
            // 第 0 趟有未命中：补槽（与本路径默认实现同一例程），下趟重建。
            let nmiss0 = counts[2] as usize;
            let miss_list: Vec<u32> = (0..nmiss0)
                .map(|j| unsafe { std::ptr::read_volatile(m.miss_ids_host.add(j)) } as u32)
                .collect();
            let prot_pri: Vec<usize> = (0..counts[4] as usize)
                .map(|j| unsafe { std::ptr::read_volatile(m.hit_pri_host.add(j)) } as usize)
                .collect();
            let prot_alt: Vec<usize> = (0..counts[5] as usize)
                .map(|j| unsafe { std::ptr::read_volatile(m.hit_alt_host.add(j)) } as usize)
                .collect();
            src.load_missing(layer, &miss_list, &prot_pri, &prot_alt)?;
        }
        unsafe { std::ptr::write_volatile(m.counts_host.add(2), 0u32) };
        let t_p = std::time::Instant::now();
        {
            // guard 只需活到 launch 返回（缓冲本体活到函数尾）；块尾即 drop，
            // 本轮末尾的失败回收（put）才能拿回报有所有权。
            let (map_pri_p, _gm) = map_pri_s.device_ptr(&k.stream);
            let (map_alt_p, fmt_p, _ga, _gf) = match (map_alt_s.as_ref(), fmt_s.as_ref()) {
                (Some(a), Some(f)) => {
                    let (p, ga) = a.device_ptr(&k.stream);
                    let (q, gf) = f.device_ptr(&k.stream);
                    (p, q, Some(ga), Some(gf))
                }
                _ => (0u64, 0u64, None, None),
            };
            let (gs_alt_dev, _gsa) = group_slots_alt.device_ptr(&k.stream);
            k.route_pairs_bf16(
                &ids, &wts, map_pri_p, map_alt_p, fmt_p,
                &mut pair_rows, &mut pair_wts, &mut group_slots, gs_alt_dev,
                &mut group_off, &mut uniq, &mut col, &mut roff,
                m.counts_dev, m.hit_pri_dev, m.hit_alt_dev, m.miss_ids_dev,
                &mut pri_gidx, &mut alt_gidx,
                num_experts, rows, top_k,
            )?;
        }
        if ht {
            htime::add(dev, 1, t_p);
        }
        sync_mark(k, "pairs")?;
        // evprof 段 3 → 段 4 的边界事件必须在专家段发射**之前**记进流。
        if let Some(p) = k.evprof.as_ref() {
            p.end(layer, 3, &k.stream);
            p.begin(layer, 4, &k.stream);
        }
        s4::mark(k, layer, 0)?;

        // ── 3a) 专家段（乐观上界发射；dyn 核设备侧守卫，读到建成后的表）──
        let mut xs = scratch::bf16_keep(k, s * hidden)?;
        k.gather_rows_bf16(x, &pair_rows, &mut xs, s, hidden, 0)?;
        sync_mark(k, "gather")?;
        s4::mark(k, layer, 1)?;
        let mut gu_all = scratch::bf16_keep(k, s * 2 * inter)?;
        let mut act_all = scratch::bf16_keep(k, s * inter)?;
        let mut o_all = scratch::bf16_keep(k, s * hidden)?;
        // xf 预转口径与非乐观路径在 TCP 档下相同（dp4a 档不接入此路径）。
        // TCP dyn 核不吃 xf 指针（核内自做 bf16→fp16），但**保留预转 launch 本身**——
        // 与非乐观路径的 GPU 工作量一致，A/B 的墙钟差才纯是发射时序的贡献。
        let xf_on = !crate::flag("LINGXI_MOE_NO_XF32") && !mixed;
        let mut xf_gu = scratch::f32_keep(k, s * hidden)?;
        if xf_on {
            k.convert_bf16_f32(&xs, &mut xf_gu, s * hidden)?;
        }
        s4::mark(k, layer, 2)?;
        // 视图表（只接 TCP dyn 档；gsel/ssel 传全表，组数守卫在核内）。
        let gsel_full = group_off.slice(0..=num_experts);
        let ssel_pri = group_slots.slice(0..num_experts);
        let ssel_alt = group_slots_alt.slice(0..num_experts);
        let pri_q3 = dv.consts.bits == 3;
        struct TcpView<'a> {
            slots: &'a CudaSlice<u8>,
            ssel: cudarc::driver::CudaView<'a, i32>,
            consts: Q4Consts,
            q3: bool,
        }
        let mut views: Vec<TcpView> = vec![TcpView {
            slots: dv.slots,
            ssel: ssel_pri,
            consts: dv.consts,
            q3: pri_q3,
        }];
        if let Some(alt) = dv.alt.as_ref() {
            views.push(TcpView {
                slots: alt.slots,
                ssel: ssel_alt,
                consts: alt.consts,
                q3: true,
            });
        }
        // gu（每视图一次 dyn launch）
        for v in views.iter() {
            let c = v.consts;
            if v.q3 {
                k.gemv_q3g128_slots_tcp_bf16_dyn_view(
                    v.slots, &v.ssel, &gsel_full, &xs, &mut gu_all, ng_ub, 2 * inter, hidden,
                    hidden, c.gu_k_pad, c.blob, c.gu_seg, c.gu_scale,
                )?;
            } else {
                k.gemv_q4g64_slots_tcp_bf16_dyn_view(
                    v.slots, &v.ssel, &gsel_full, &xs, &mut gu_all, ng_ub, 2 * inter, hidden,
                    hidden, c.gu_k_pad, c.blob, c.gu_seg, c.gu_scale,
                )?;
            }
            sync_mark(k, if v.q3 { "gu-q3" } else { "gu-q4" })?;
            s4::mark(k, layer, if v.q3 { 4 } else { 3 })?;
        }
        k.swiglu_bf16(&gu_all, &mut act_all, s, inter)?;
        sync_mark(k, "swiglu")?;
        s4::mark(k, layer, 5)?;
        let mut xf_dn = scratch::f32_keep(k, s * inter)?;
        if xf_on {
            k.convert_bf16_f32(&act_all, &mut xf_dn, s * inter)?;
        }
        s4::mark(k, layer, 6)?;
        // dn（dyn；与 gu 同规则）
        for v in views.iter() {
            let c = v.consts;
            if v.q3 {
                k.gemv_q3g128_slots_tcp_bf16_dyn_view(
                    v.slots, &v.ssel, &gsel_full, &act_all, &mut o_all, ng_ub, hidden, inter,
                    inter, c.dn_k_pad, c.blob, c.dn_seg, c.dn_scale,
                )?;
            } else {
                k.gemv_q4g64_slots_tcp_bf16_dyn_view(
                    v.slots, &v.ssel, &gsel_full, &act_all, &mut o_all, ng_ub, hidden, inter,
                    inter, c.dn_k_pad, c.blob, c.dn_seg, c.dn_scale,
                )?;
            }
            sync_mark(k, if v.q3 { "dn-q3" } else { "dn-q4" })?;
            s4::mark(k, layer, if v.q3 { 8 } else { 7 })?;
        }
        // 按行 CSR 归并 + shared 残差（与非乐观路径同核同参；fo 直连移交调用方）。
        let mut fo = scratch::bf16(k, rows * hidden)?;
        k.moe_combine_add_csr_bf16(&o_all, &col, &roff, &pair_wts, shared, &mut fo, rows, hidden)?;
        sync_mark(k, "combine")?;
        s4::mark(k, layer, 9)?;
        if let Some(p) = k.evprof.as_ref() {
            p.end(layer, 4, &k.stream);
        }
        // 对拍（与非乐观路径同口径；在 put 回池前读，flag 关时是零开销）。
        if crate::flag("LINGXI_FULLDEV_OUT_DUMP") {
            let want: u32 = std::env::var("LINGXI_MOE_OUT_DUMP_LAYER")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(|v: u32| v.wrapping_sub(1))
                .unwrap_or(crate::DIMS.n_layers as u32 - 1);
            if layer == want {
                for (nm, buf) in [("gu_all", &gu_all), ("act_all", &act_all), ("o_all", &o_all)] {
                    let h: Vec<bf16> = k.stream.clone_dtoh(buf)?;
                    let tot: f64 = h.iter().map(|v| v.to_f32() as f64).sum();
                    let nans = h.iter().filter(|v| v.is_nan()).count();
                    let mut bits: u64 = 0xcbf29ce484222325;
                    for v in h.iter() {
                        bits = (bits ^ (v.to_bits() as u64)).wrapping_mul(0x100000001b3);
                    }
                    eprintln!(
                        "[fulldev-out] 层{layer} {nm} 全和={tot:.6} n={} nan={nans} h{bits:016x}",
                        h.len()
                    );
                }
            }
        }

        // ── 3b) 全部发射完才自旋：CPU 不钉在 pair 完成上 ──
        let t_s = std::time::Instant::now();
        route_spin(k, false)?;
        if ht {
            htime::add(dev, 2, t_s);
        }
        let t_c = std::time::Instant::now();
        for (i, c) in counts.iter_mut().enumerate() {
            *c = unsafe { std::ptr::read_volatile(m.counts_host.add(i)) };
        }
        if ht {
            htime::add(dev, 3, t_c);
            htime::tick(dev, layer);
        }
        super::s3::mark(k, layer, 4)?;
        // 缓冲回收（keep 池同流复用安全；fo 成功趟带走、失败趟放回）。
        drop(views);
        scratch::put_bf16_keep(xs);
        scratch::put_bf16_keep(gu_all);
        scratch::put_bf16_keep(act_all);
        scratch::put_bf16_keep(o_all);
        scratch::put_f32_keep(xf_gu);
        scratch::put_f32_keep(xf_dn);
        let nmiss = counts[2];
        if nmiss == 0 {
            fo_opt = Some(fo);
            break;
        }
        scratch::put_bf16(fo);
        if attempt == 1 {
            // 两趟仍缺 ⇒ 落回 legacy（写过的全是 scratch/keep 池缓冲，无副作用）。
            scratch::put_i32_keep(pair_rows);
            scratch::put_f32_keep(pair_wts);
            scratch::put_i32_keep(group_slots);
            scratch::put_i32_keep(group_slots_alt);
            scratch::put_i32_keep(group_off);
            scratch::put_i32_keep(col);
            scratch::put_i32_keep(roff);
            scratch::put_i32_keep(uniq);
            scratch::put_i32_keep(pri_gidx);
            scratch::put_i32_keep(alt_gidx);
            scratch::put_bf16_keep(logits);
            scratch::put_i32_keep(ids);
            scratch::put_f32_keep(wts);
            return Ok(None);
        }
    }
    let fo = fo_opt.expect("两趟循环必有一趟 break 或 return");
    let u = counts[1] as usize;
    let tmax = counts[3] as usize;

    // ── 4) 计数后诊断（与非乐观路径同内容；此时专家段已在流上跑/跑完）──
    if crate::flag("LINGXI_FULLDEV_CHK") {
        let lh: Vec<bf16> = k.stream.clone_dtoh(&logits)?;
        let ih: Vec<i32> = k.stream.clone_dtoh(&ids)?;
        let wh: Vec<f32> = k.stream.clone_dtoh(&wts)?;
        let mut bad = 0usize;
        let mut worst = 0f32;
        for (i, v) in lh.iter().enumerate().take(rows * num_experts) {
            let f = v.to_f32();
            if f.is_finite() {
                worst = worst.max(f.abs());
            } else {
                if bad < 3 {
                    eprintln!("[chk] L{layer} row{} idx{} val={f}", i / num_experts, i % num_experts);
                }
                bad += 1;
            }
        }
        let ibad = ih.iter().take(s).filter(|&&v| v < 0 || v as usize >= num_experts).count();
        let wbad = wh.iter().take(s).filter(|v| !v.is_finite()).count();
        eprintln!(
            "[chk] L{layer} rows{rows} nonfinite_logits={bad} max|logit|={worst:.3} bad_ids={ibad} bad_wts={wbad}"
        );
    }
    if crate::flag("LINGXI_PAIR_DUMP") {
        let rh: Vec<i32> = k.stream.clone_dtoh(&pair_rows)?;
        let wh: Vec<f32> = k.stream.clone_dtoh(&pair_wts)?;
        let ch: Vec<i32> = k.stream.clone_dtoh(&col)?;
        let ofh: Vec<i32> = k.stream.clone_dtoh(&roff)?;
        let goh: Vec<i32> = k.stream.clone_dtoh(&group_off)?;
        let gsh: Vec<i32> = k.stream.clone_dtoh(&group_slots)?;
        eprintln!(
            "[pair-dev] L{layer} rows{rows} n{} h{:016x} rows8={:?} wts8={:?}",
            rh.len(),
            super::hash_pair_tables(&rh, &wh),
            &rh[..rh.len().min(8)],
            &wh[..wh.len().min(8)]
        );
        eprintln!(
            "[pair-dev] L{layer} col n{} h{:016x} col8={:?} roff{:?} go{:?} gs8={:?}",
            ch.len(),
            super::hash_i32(&ch) ^ super::hash_i32(&ofh),
            &ch[..ch.len().min(8)],
            &ofh[..ofh.len().min(8)],
            &goh[..goh.len().min(u + 1)],
            &gsh[..gsh.len().min(8)]
        );
    }
    if crate::flag("LINGXI_ROUTE_FULLDEV_TRACE") {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 4 || n % 48 == 0 {
            eprintln!(
                "[fulldev] call {n} layer {layer} rows {rows} s {s} u {u} tmax {tmax} mixed {mixed} npair_pri {} npair_alt {}（opt 乐观发射）",
                counts[4], counts[5]
            );
        }
    }
    if u == 0 {
        scratch::put_i32_keep(pair_rows);
        scratch::put_f32_keep(pair_wts);
        scratch::put_i32_keep(group_slots);
        scratch::put_i32_keep(group_slots_alt);
        scratch::put_i32_keep(group_off);
        scratch::put_i32_keep(col);
        scratch::put_i32_keep(roff);
        scratch::put_i32_keep(uniq);
        scratch::put_i32_keep(pri_gidx);
        scratch::put_i32_keep(alt_gidx);
        scratch::put_bf16_keep(logits);
        scratch::put_i32_keep(ids);
        scratch::put_f32_keep(wts);
        return Ok(None);
    }
    if let Some(p) = k.evprof.as_ref() {
        p.set_info(layer, 0, s as u32);
        p.set_info(layer, 1, u as u32);
        p.set_info(layer, 2, rows as u32);
    }
    s4::info(s, u, rows, counts[4], counts[5]);
    if ht {
        let tot = t_call.elapsed().as_secs_f64() * 1e6;
        if tot > 250.0 {
            eprintln!("[htime-spike] dev{dev} L{layer} rows{rows} 总{tot:.0}µs（opt 乐观发射口径）");
        }
    }
    scratch::put_i32_keep(pair_rows);
    scratch::put_f32_keep(pair_wts);
    scratch::put_i32_keep(group_slots);
    scratch::put_i32_keep(group_slots_alt);
    scratch::put_i32_keep(group_off);
    scratch::put_i32_keep(col);
    scratch::put_i32_keep(roff);
    scratch::put_i32_keep(uniq);
    scratch::put_i32_keep(pri_gidx);
    scratch::put_i32_keep(alt_gidx);
    scratch::put_bf16_keep(logits);
    scratch::put_i32_keep(ids);
    scratch::put_f32_keep(wts);
    Ok(Some(fo))
}
