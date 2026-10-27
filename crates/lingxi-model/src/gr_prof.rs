//! **GR 块（decode 的 seg0 里的 attn_GR + seg2 的 mlp_GR）内部的逐核探针**
//! （`LINGXI_GR_PROF=1`，默认关）。
//!
//! 为什么要有它：`LINGXI_EVPROF=1` 只把 `入口+PLE+attn_GR`（层均 93~102 µs）和
//! `mlp_GR`（90~94 µs）各包在**一个**事件对里，看不出 5 个 kernel 谁是大头、
//! 也分不出 PLE 与 GR。口径与 `attn_prof` / `moe::batch_dev::s4` 完全一致：
//! **每个边界在流内记一个带计时的事件（不排空流水），层末只 sync 一次**。
//! `LINGXI_KTS` 那种"每段 synchronize"会把小核放大 4 倍（见
//!），不要用。
//!
//! 段定义（按**执行次序**，`mark` 由 `gr.rs` 在算子边界打）：
//! * 旧路（5 发）：0 rmsnorm / 1 down / 2 up / 3 mix / 4 inject
//! * 融合路（3 发，`rows 2..8`）：0 rmsnorm / 1 down+inject / 2 up+mix
//! * PLE（仅 layer 1，由 `layer.rs` 打 begin/end）单独一段。
//!
//! ⚠️ 层末那次 sync 会排空流水 ⇒ **只看份额，不要用它评吞吐**（吞吐看 evprof/尺子）。

use anyhow::Result;
use cudarc::driver::{sys, CudaEvent};
use lingxi_kernels::Kernels;
use std::cell::RefCell;

/// 每条链最多 6 个边界（= 5 段）；融合路只用 4 个（= 3 段）。
const MAXB: usize = 6;
/// 每条链的事件数：`MAXB + 1` 个边界。
const MAXEV: usize = MAXB + 1;
/// 槽位：0..2 = PLE 的两个边界；2..=8 = attn_gr；8..=14 = mlp_gr。
const NSLOT: usize = 2 + 2 * MAXEV;
/// 段号（累计数组下标）：0 = PLE，1..=5 = attn_gr，6..=10 = mlp_gr。
const NSEG: usize = 1 + 2 * MAXB;

/// 一条 GR 链（一层两个：attn_GR / mlp_GR）。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Chain {
    Attn = 0,
    Mlp = 1,
}

const OLD_NAMES: [&str; MAXB] = ["rmsnorm", "down", "up", "mix", "inject", ""];
const FUSE_NAMES: [&str; MAXB] = ["rmsnorm", "down+inject", "up+mix", "", "", ""];

struct Dev {
    ctx: usize,
    evs: Vec<CudaEvent>,
}

struct St {
    /// 累计（只统计 rows≥2 的验证批）
    sum: [f64; NSEG],
    n: f64,
    ple_n: f64,
    /// rows==1（草稿占位）的累计与次数（`LINGXI_GR_PROF_T1`）
    sum1: [f64; NSEG],
    n1: f64,
    /// 本层已记录的链（bits: 0 = PLE，1 = attn，2 = mlp）
    have: u8,
    /// 每条链本层的段数（0 = 本层没跑该链）
    nseg: [usize; 2],
    /// 当前链（`begin` 设置）
    key: Option<Chain>,
    last_idx: usize,
    /// 累计 flush 次数（含草稿批）与**验证批**层数
    seen: u32,
    vseen: u32,
    info: [u32; 2],
    devs: Vec<Dev>,
}

thread_local! {
    static ST: RefCell<St> = RefCell::new(St {
        sum: [0.0; NSEG], n: 0.0, ple_n: 0.0, sum1: [0.0; NSEG], n1: 0.0,
        have: 0, nseg: [0; 2], key: None, last_idx: 0, seen: 0, vseen: 0,
        info: [0; 2], devs: Vec::new(),
    });
}

#[inline]
pub fn on() -> bool {
    crate::flag("LINGXI_GR_PROF") || t1_on()
}

/// `LINGXI_GR_PROF_T1=1`：额外**统计并打印 rows==1（MTP 草稿占位）**的逐核表。
/// 原来 rows<2 既不累计也不打印，草稿 t=1 的 attn_gr / mlp_gr 只有段均、看不到 5 个核。
#[inline]
fn t1_on() -> bool {
    crate::flag("LINGXI_GR_PROF_T1")
}

/// 逐层打印的层数（默认 24 = 双卡 layer-split 下一整个验证批）。
#[inline]
fn prof_n() -> u32 {
    std::env::var("LINGXI_GR_PROF_N")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(24)
}

/// PLE 段开始（`layer.rs` 在 `plet::ple_step` 前调用）。
pub fn ple_begin(k: &Kernels) -> Result<()> {
    if !on() {
        return Ok(());
    }
    with_dev(k, |d| Ok(d.evs[0].record(&k.stream)?))
}

/// PLE 段结束（含 `ple_step` + `add_bf16`）。
pub fn ple_end(k: &Kernels) -> Result<()> {
    if !on() {
        return Ok(());
    }
    ST.with(|c| c.borrow_mut().have |= 1);
    with_dev(k, |d| Ok(d.evs[1].record(&k.stream)?))
}

/// 一条 GR 链开始：记第 0 个边界。`rows` 只用于打印现场（区分草稿 t=1 与验证批）。
pub fn begin(k: &Kernels, rows: u32, chain: Chain) -> Result<()> {
    if !on() {
        return Ok(());
    }
    let base = 2 + (chain as usize) * MAXEV;
    ST.with(|c| {
        let mut g = c.borrow_mut();
        g.key = Some(chain);
        g.last_idx = 0;
        g.nseg[chain as usize] = 0;
        g.have |= 1 << (1 + chain as usize);
        g.info = [rows, 0];
    });
    with_dev(k, |d| Ok(d.evs[base].record(&k.stream)?))
}

/// 段边界。`idx`：1..=nseg 依次为各段结束；未跑的边界按当前位置补记（零宽段）。
/// 最后一次 `mark` 的 `idx` 就是本链的段数（融合路 3、旧路 5）。
pub fn mark(k: &Kernels, idx: usize) -> Result<()> {
    if !on() {
        return Ok(());
    }
    let (chain, nseg, last) = ST.with(|c| {
        let g = c.borrow();
        match g.key {
            Some(ch) => (ch, g.nseg[ch as usize], g.last_idx),
            None => return (Chain::Attn, 0, 0),
        }
    });
    if last == 0 && ST.with(|c| c.borrow().key.is_none()) {
        return Ok(());
    }
    let base = 2 + (chain as usize) * MAXEV;
    let idx = idx.min(MAXB);
    with_dev(k, |d| {
        for j in last + 1..=idx {
            d.evs[base + j].record(&k.stream)?;
        }
        Ok(())
    })?;
    let _ = nseg;
    ST.with(|c| {
        let mut g = c.borrow_mut();
        g.last_idx = idx;
        g.nseg[chain as usize] = idx;
    });
    Ok(())
}

/// 层末汇总：**一次 sync**，读回本层两条链 + PLE，累计并打印。
/// 由 `layer.rs` 在 seg2（mlp_GR）之后调用。
pub fn flush(k: &Kernels, layer: u32) -> Result<()> {
    if !on() {
        return Ok(());
    }
    let (have, nseg, info) = ST.with(|c| {
        let g = c.borrow();
        (g.have, g.nseg, g.info)
    });
    if have == 0 {
        return Ok(());
    }
    k.stream.synchronize()?;
    let rows = info[0];
    let mut cur = [0f64; NSEG];
    {
        let di = dev_index(k)?;
        ST.with(|c| {
            let g = c.borrow();
            let d = &g.devs[di];
            if have & 1 != 0 {
                if let Ok(ms) = d.evs[0].elapsed_ms(&d.evs[1]) {
                    cur[0] = ms as f64 * 1e3;
                }
            }
            for ch in 0..2usize {
                if have & (1 << (1 + ch)) == 0 {
                    continue;
                }
                let base = 2 + ch * MAXEV;
                for s in 0..nseg[ch] {
                    if let Ok(ms) = d.evs[base + s].elapsed_ms(&d.evs[base + s + 1]) {
                        cur[1 + ch * MAXB + s] = ms as f64 * 1e3;
                    }
                }
            }
        });
    }
    // 逐链的现场描述（验证批与草稿占位共用）
    let tot: f64 = cur.iter().sum();
    let mut line = String::new();
    for ch in 0..2usize {
        if have & (1 << (1 + ch)) == 0 {
            continue;
        }
        let names = if nseg[ch] >= 5 { OLD_NAMES } else { FUSE_NAMES };
        line.push_str(if ch == 0 { "attn_gr[" } else { "mlp_gr[" });
        for s in 0..nseg[ch] {
            line.push_str(&format!("{} {:5.1} ", names[s], cur[1 + ch * MAXB + s]));
        }
        line.push_str("] ");
    }
    if have & 1 != 0 {
        line.push_str(&format!("ple {:5.1}", cur[0]));
    }
    ST.with(|c| {
        let mut g = c.borrow_mut();
        g.seen += 1;
        if rows >= 2 {
            for (a, b) in g.sum.iter_mut().zip(cur.iter()) {
                *a += *b;
            }
            g.n += 1.0;
            g.vseen += 1;
            if have & 1 != 0 {
                g.ple_n += 1.0;
            }
        }
        // 逐层证据：默认打**一个完整验证批**的前 24 层（`LINGXI_GR_PROF_N` 可改）。
        // ⚠️ 不能用 `seen/48` 判轮：双卡 layer-split 下每批只有 24 层（48 是单卡数）。
        if rows >= 2 && g.vseen <= prof_n() {
            eprintln!("[grL] #{:<3} L{layer:<2} rows{rows}｜{line}｜合计 {tot:.1} µs", g.seen);
        }
        // 草稿占位（rows==1）：`LINGXI_GR_PROF_T1=1` 时同样逐层打 + 每 24 次（= 24 个草稿步）打一次均值
        if rows < 2 && t1_on() {
            for (a, b) in g.sum1.iter_mut().zip(cur.iter()) {
                *a += *b;
            }
            g.n1 += 1.0;
            if g.n1 <= prof_n() as f64 {
                eprintln!("[grT1] #{:<3} L{layer:<2} rows{rows}｜{line}｜合计 {tot:.1} µs", g.seen);
            }
            if g.n1 % 24.0 == 0.0 {
                let n = g.n1;
                let s = g.sum1;
                let mut attn = String::new();
                let mut mlp = String::new();
                for ch in 0..2usize {
                    let names = if nseg[ch] >= 5 { OLD_NAMES } else { FUSE_NAMES };
                    let dst = if ch == 0 { &mut attn } else { &mut mlp };
                    for i in 0..nseg[ch] {
                        dst.push_str(&format!("{} {:.1}  ", names[i], s[1 + ch * MAXB + i] / n));
                    }
                }
                eprintln!(
                    "[grT1-avg] n={n:.0} 步均 µs｜attn_gr {attn}｜mlp_gr {mlp}｜两段合计 {:.1}",
                    s.iter().skip(1).sum::<f64>() / n
                );
                g.sum1 = [0.0; NSEG];
                g.n1 = 0.0;
            }
        }
        g.have = 0;
        g.key = None;
        g.last_idx = 0;
        // 每 48 个验证层（本机 ≈ 2 个批）打一次层均，然后清零
        if rows >= 2 && g.n >= 48.0 {
            let n = g.n;
            let s = g.sum;
            let tot: f64 = s.iter().sum();
            let mut attn = String::new();
            let mut mlp = String::new();
            for ch in 0..2usize {
                let names = if nseg[ch] >= 5 { OLD_NAMES } else { FUSE_NAMES };
                let dst = if ch == 0 { &mut attn } else { &mut mlp };
                for i in 0..nseg[ch] {
                    dst.push_str(&format!("{} {:.1}  ", names[i], s[1 + ch * MAXB + i] / n));
                }
            }
            eprintln!(
                "[gr] n={n:.0} 层均 µs｜attn_gr {attn}｜mlp_gr {mlp}｜ple {:.1}（{:.0} 次）｜两段合计 {:.1}",
                s[0] / g.ple_n.max(1.0),
                g.ple_n,
                tot / n
            );
            g.sum = [0.0; NSEG];
            g.n = 0.0;
            g.ple_n = 0.0;
        }
        Ok(())
    })
}

fn with_dev<R>(k: &Kernels, f: impl FnOnce(&Dev) -> Result<R>) -> Result<R> {
    let di = dev_index(k)?;
    ST.with(|c| {
        let g = c.borrow();
        f(&g.devs[di])
    })
}

/// 取（或懒建）本上下文的事件槽。
fn dev_index(k: &Kernels) -> Result<usize> {
    let ctx = k.ctx.cu_ctx() as usize;
    let mk = |k: &Kernels| -> Result<usize> {
        // ⚠️ 必须开 timing（cudarc 的 `new_event(None)` 是 DISABLE_TIMING）
        let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let mut evs = Vec::with_capacity(NSLOT);
        for _ in 0..NSLOT {
            evs.push(k.ctx.new_event(flags)?);
        }
        ST.with(|c| {
            let mut g = c.borrow_mut();
            g.devs.push(Dev { ctx, evs });
            Ok(g.devs.len() - 1)
        })
    };
    let existing = ST.with(|c| c.borrow().devs.iter().position(|d| d.ctx == ctx));
    match existing {
        Some(i) => Ok(i),
        None => mk(k),
    }
}
