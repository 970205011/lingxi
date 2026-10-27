//! **MTP 草稿一层（哨兵层 100）的分段事件探针（正确边界版）**（`LINGXI_DRAFT_PROF=1`）。
//!
//! 为什么另起一个而不是改 `mtp_prof`：`mtp_prof::mark` 记边界事件时用的是
//! `for j in last..=idx { evs[j+1].record }` —— 当段号连续推进时它会把**上一个边界
//! 再记一遍**（`last` 停在「上一个已完成的段」，而 `evs[last+1]` 正是那个边界的终点）。
//! 事件是就地覆写的，于是**每个边界都被后一次 mark 的时间戳覆盖** ⇒ 整张表**整体前移一段**：
//! 表里的 "attn_gr" 其实是 attn、"attn" 其实是 mlp_gr、"mlp_gr" 其实是 MoE……
//! （2026-10-24 用 `LINGXI_PROF=1 LINGXI_KTS=1` 的主机侧真时间逐段对拍确认：
//! 宿主 attn_gr 115.2／attn 217.6／mlp_gr 73.6／moe 325.0 与 mtp_prof 的
//! stem 148.5／attn_gr 217.1／attn 72.7／mlp_gr 324.6 一一错位一格。）
//! `mtp_prof` 属于"已提交、本轮不动"的清单，所以这里给一个**边界写法正确**的复本，
//! 段定义与它逐字相同，只把记录的边界算对：
//! * `begin` 记 `evs[0]`，并把「下一个要记的边界」设成 1；
//! * `mark(seg)` 记 `evs[next_b..=seg+1]`（被跳过的段在这里补成 0 宽），再 `next_b = seg+2`。
//!
//! 段（与 `mtp_prof` 同义）：0 stem / 1 attn_gr / 2 attn / 3 mlp_gr / 4 MoE / 5 收尾。
//! 层外的 head（mixer GR + lm_head + top-k + D2H）不在层内，用**主机墙钟**从上一个
//! `mark(5)` 到下一次 `begin` 的间隔量（与 `mtp_prof` 的 `[mtp-prof-head]` 同口径）。
//!
//! ⚠️ 与 `mtp_prof` 一样：一步结束处会 `synchronize` 一次 ⇒ **只用来分账，不要用它评吞吐**。

use anyhow::Result;
use cudarc::driver::{sys, CudaEvent};
use lingxi_kernels::Kernels;
use std::cell::RefCell;

pub const NSEG: usize = 6;
const NAMES: [&str; NSEG] = ["stem", "attn_gr", "attn", "mlp_gr", "moe", "收尾"];

struct Dev {
    ctx: usize,
    evs: Vec<CudaEvent>,
}

struct St {
    devs: Vec<Dev>,
    active: bool,
    /// 下一个要记的边界事件号（1..=NSEG）
    next_b: usize,
    t: u32,
    /// 上一次 mark 的主机时刻（层外 head 的起点）
    tp: Option<std::time::Instant>,
    cur: [f64; NSEG],
    cur_h: [f64; NSEG],
    sum: [f64; NSEG],
    n: u64,
}

thread_local! {
    static ST: RefCell<St> = const { RefCell::new(St {
        devs: Vec::new(), active: false, next_b: 1, t: 0, tp: None,
        cur: [0.0; NSEG], cur_h: [0.0; NSEG], sum: [0.0; NSEG], n: 0,
    }) };
    static ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[inline]
pub fn on() -> bool {
    ON.with(|c| match c.get() {
        Some(v) => v,
        None => {
            let v = crate::flag("LINGXI_DRAFT_PROF");
            c.set(Some(v));
            v
        }
    })
}

fn dev_index(k: &Kernels, g: &mut St, ctx: usize) -> Result<usize> {
    if let Some(i) = g.devs.iter().position(|d| d.ctx == ctx) {
        return Ok(i);
    }
    let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
    let mut evs = Vec::with_capacity(NSEG + 1);
    for _ in 0..=NSEG {
        evs.push(k.ctx.new_event(flags)?);
    }
    g.devs.push(Dev { ctx, evs });
    Ok(g.devs.len() - 1)
}

/// 一步开始（`mtp::stem` 入口）。
pub fn begin(k: &Kernels, t: usize) -> Result<()> {
    if !on() {
        return Ok(());
    }
    let ctx = k.ctx.cu_ctx() as usize;
    ST.with(|c| -> Result<()> {
        let mut g = c.borrow_mut();
        let di = dev_index(k, &mut g, ctx)?;
        // 上一步已完整跑完 ⇒ 这段主机间隔就是层外（head+embed+簿记）
        if g.n > 0 && !g.active {
            if let Some(t0) = g.tp {
                eprintln!(
                    "[draft-prof-head] #{} t{} 层外（head+embed+簿记）= {:.1} µs",
                    g.n,
                    g.t,
                    t0.elapsed().as_secs_f64() * 1e6
                );
            }
        }
        g.active = true;
        g.next_b = 1;
        g.t = t as u32;
        g.tp = Some(std::time::Instant::now());
        g.cur = [0.0; NSEG];
        g.cur_h = [0.0; NSEG];
        g.devs[di].evs[0].record(&k.stream)?;
        Ok(())
    })
}

/// 段边界：`seg` = 刚结束的段号。未 `begin` 时 no-op（主干 0..47 层一律 no-op）。
pub fn mark(k: &Kernels, seg: usize) -> Result<()> {
    if !on() {
        return Ok(());
    }
    let ctx = k.ctx.cu_ctx() as usize;
    ST.with(|c| -> Result<()> {
        let mut g = c.borrow_mut();
        if !g.active {
            return Ok(());
        }
        let di = dev_index(k, &mut g, ctx)?;
        let idx = seg.min(NSEG - 1);
        // 主机墙钟（与事件口径并排看：区分"GPU 忙"与"等主机发射"）
        if let Some(t0) = g.tp {
            let now = std::time::Instant::now();
            g.cur_h[idx] = now.duration_since(t0).as_secs_f64() * 1e6;
            g.tp = Some(now);
        }
        // **只记还没记过的边界**：evs[next_b ..= idx+1]。被跳过的段在这里补成 0 宽。
        for b in g.next_b..=idx + 1 {
            g.devs[di].evs[b].record(&k.stream)?;
        }
        g.next_b = idx + 2;
        if idx + 1 < NSEG {
            return Ok(());
        }
        // 一步结束：一次 sync 读回各段
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
        g.active = false;
        g.n += 1;
        for (a, b) in g.sum.iter_mut().zip(cur.iter()) {
            *a += *b;
        }
        let tot: f64 = cur.iter().sum();
        let mut line = String::new();
        let mut line_h = String::new();
        for (j, name) in NAMES.iter().enumerate() {
            line.push_str(&format!("{name} {:7.1}／", cur[j]));
            line_h.push_str(&format!("{name} {:7.1}／", g.cur_h[j]));
        }
        let tot_h: f64 = g.cur_h.iter().sum();
        eprintln!(
            "[draft-prof] #{} t{}｜{line}合计 {tot:.1} µs｜累计步均 {:.1} µs",
            g.n,
            g.t,
            g.sum.iter().sum::<f64>() / g.n as f64
        );
        eprintln!("[draft-prof-host] #{} t{}｜{line_h}合计 {tot_h:.1} µs", g.n, g.t);
        Ok(())
    })
}
