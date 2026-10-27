//! **MTP 草稿层（哨兵层号 100）的分段事件探针**（`LINGXI_MTP_PROF=1`，默认关）。
//!
//! 动机：`LINGXI_SPEC_PROF` 只给四相位（snapshot/draft/verify/commit+replay），
//! 而 draft 才是要降的那块（~8.8 ms/轮）。要回答"1.5 ms/token 花在哪"，
//! 必须把 MTP 的一步（stem → attn_gr → attn → mlp_gr → MoE → 收尾）拆开。
//!
//! **口径**（与 `attn_prof`、`moe/batch_dev.rs::s4` 完全一致）：每个边界在**流内**记一个
//! 带计时的事件（不排空流水），一步结束处只 `synchronize` 一次读出各段。
//! ⚠️ 绝不能用"每段 sync + 主机钟"——实测
//! 会把 3 µs 的小核放大 4 倍。
//!
//! 只对**哨兵层 100**（MTP 伪层）激活：`begin` 由 `mtp::stem` 打，`mark(i)` 在
//! `layer.rs` / `layer_moe_finish` 的段边界打；主干 0..47 一律 no-op。
//!
//! 段定义（`mark(seg)` = 第 `seg` 段结束）：
//! * 0 stem（入口：embedding/hidden 两支 RMSNorm + 两个 [N,N] 投影 + 广播相加）
//! * 1 attn_gr（GR 读出）
//! * 2 attn（QSA 整段：投影/norm/rope/KV/indexer/稀疏注意力/out_proj）
//! * 3 mlp_gr（含上一步 attn 输出的 gr_write）
//! * 4 MoE（路由 + shared + 专家 GEMV + 归并；内部再分用 `LINGXI_PROF` / `LINGXI_MOE_PROF`）
//! * 5 收尾（MoE 输出的 gr_write）
//!
//! 层尾的 head（mixer + lm_head + top-k，落在 `linxi-bench::spec` 里）不计入本探针 ⇒
//! **head ≈ draft 单步墙钟 − 本探针合计**。

use anyhow::Result;
use cudarc::driver::{sys, CudaEvent};
use lingxi_kernels::Kernels;
use std::cell::RefCell;

pub const NSEG: usize = 6;
const NAMES: [&str; NSEG] = ["stem", "attn_gr", "attn", "mlp_gr", "moe", "收尾"];

struct Dev {
    ctx: usize,
    /// `NSEG + 1` 个带计时事件：第 `s` 段 = `evs[s] → evs[s+1]`
    evs: Vec<CudaEvent>,
}

struct St {
    devs: Vec<Dev>,
    /// 当前这一步是否已 `begin`（否则其它调用点直接 no-op）
    active: bool,
    /// 上一次 `mark` 的段号（补记被跳过的边界用）
    last: usize,
    /// 当前步的 t
    t: u32,
    /// 上一次 `mark` 的主机时刻（诊断：区分"GPU 真的在跑"与"等主机发射"）
    tp: Option<std::time::Instant>,
    /// 当前步各段的主机墙钟 µs（含发射与等在途工作）
    cur_h: [f64; NSEG],
    /// 当前步各段 µs
    cur: [f64; NSEG],
    /// 累计与步数
    sum: [f64; NSEG],
    n: u64,
}

thread_local! {
    static ST: RefCell<St> = const { RefCell::new(St {
        devs: Vec::new(), active: false, last: 0, t: 0, tp: None,
        cur_h: [0.0; NSEG], cur: [0.0; NSEG], sum: [0.0; NSEG], n: 0,
    }) };
    /// `on` 的线程级缓存（主干每层会调 3~5 次 `mark`，别每次都走 env 查表）
    static ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[inline]
pub fn on() -> bool {
    ON.with(|c| match c.get() {
        Some(v) => v,
        None => {
            let v = crate::flag("LINGXI_MTP_PROF");
            c.set(Some(v));
            v
        }
    })
}

fn dev_index(k: &Kernels, g: &mut St, ctx: usize) -> Result<usize> {
    if let Some(i) = g.devs.iter().position(|d| d.ctx == ctx) {
        return Ok(i);
    }
    // ⚠️ 必须开 timing（`new_event(None)` 是 DISABLE_TIMING）
    let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
    let mut evs = Vec::with_capacity(NSEG + 1);
    for _ in 0..=NSEG {
        evs.push(k.ctx.new_event(flags)?);
    }
    g.devs.push(Dev { ctx, evs });
    Ok(g.devs.len() - 1)
}

/// 一步开始时调用（`mtp::stem` 入口）。仅 `LINGXI_MTP_PROF=1` 时生效。
pub fn begin(k: &Kernels, t: usize) -> Result<()> {
    if !on() {
        return Ok(());
    }
    let ctx = k.ctx.cu_ctx() as usize;
    ST.with(|c| -> Result<()> {
        let mut g = c.borrow_mut();
        let di = dev_index(k, &mut g, ctx)?;
        // 上一步已经跑完（`last` 停在第 5 段）⇒ 这段间隔就是**层外**的部分：
        // `head`（mixer GR + lm_head + top-k + 3 次 D2H）+ `embed1` + spec 里的簿记。
        // 它落在 `lingxi-bench::spec` 里（不能改），只能这样从外面量。
        if g.n > 0 && g.last + 1 == NSEG {
            if let Some(t0) = g.tp {
                eprintln!(
                    "[mtp-prof-head] #{} t{} 层外（head+embed+簿记）= {:.1} µs",
                    g.n,
                    g.t,
                    t0.elapsed().as_secs_f64() * 1e6
                );
            }
        }
        g.active = true;
        g.last = 0;
        g.t = t as u32;
        g.tp = Some(std::time::Instant::now());
        g.cur = [0.0; NSEG];
        g.cur_h = [0.0; NSEG];
        g.devs[di].evs[0].record(&k.stream)?;
        Ok(())
    })
}

/// 段边界。`seg` = 刚结束的段号（0..NSEG-1）。未 `begin` 时是 no-op。
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
        // 主机墙钟（诊断用：与事件口径并排看，区分"GPU 忙"与"等主机"）
        if let Some(t0) = g.tp {
            let now = std::time::Instant::now();
            g.cur_h[idx] = now.duration_since(t0).as_secs_f64() * 1e6;
            g.tp = Some(now);
        }
        // 事件下标 = 段号 + 1（`evs[0]` 是 begin）。跳过的段按当前位置补记 ⇒ 0 宽。
        for j in g.last..=idx {
            g.devs[di].evs[j + 1].record(&k.stream)?;
        }
        g.last = idx;
        if idx + 1 < NSEG {
            return Ok(());
        }
        // 一步结束：这里 sync 一次只为读 elapsed。注意 draft 每步末尾本来就有一次
        // argmax 的 D2H 同步，所以这次 sync 不会额外排空多少工作。
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
            "[mtp-prof] #{} t{}｜{line}合计 {tot:.1} µs｜累计步均 {:.1} µs",
            g.n,
            g.t,
            g.sum.iter().sum::<f64>() / g.n as f64
        );
        eprintln!("[mtp-prof-host] #{} t{}｜{line_h}合计 {tot_h:.1} µs", g.n, g.t);
        Ok(())
    })
}
