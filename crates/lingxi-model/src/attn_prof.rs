//! **attn 段（evprof seg1）内部的六分探针**（`LINGXI_ATTN_SYNC=1`，默认关）。
//!
//! 为什么要有它：evprof 把 QSA/GDN 的整段注意力包在**一个**事件对里（层均 0.30 / 0.40 ms），
//! 只看总数分不出"投影 / 卷积 / 递推 / indexer / 稀疏注意力"谁是大头。旧的 `LINGXI_KTS`
//! 口径是**主机钟 + 每段 `synchronize`**——实测会把小核放大 4 倍（见
//!），且 GDN 三段之和 116~140 µs 对不上
//! attn 段的 300 µs ⇒ 旧数据不能再用。
//!
//! **口径**（与 `moe/batch_dev.rs::s4`、`lingxi_kernels::evprof` 完全一致）：
//! 每个边界在**流内**记一个带计时的事件（不排空流水），层末统一 sync 一次读出 6 段。
//! ⚠️ 层末那次 sync 仍会排空流水、拖慢整轮 ⇒ **只看份额**，不要用它评吞吐。
//!
//! 段定义（`begin` 由 `layer.rs` 在 attn 调用前打，`mark(i)` 在各段边界打）：
//! * GDN：0 qkvz 投影 / 1 z 投影 / 2 a、b 两个 tiny GEMV / 3 卷积 / 4 递推 / 5 out_proj
//! * QSA：0 qkv+gate 投影 / 1 k norm+rope+KV 追加 / 2 indexer qk 投影+norm+rope /
//! 3 indexer 池化+打分+选块 / 4 稀疏(gather) attention 主核 / 5 gate+out_proj

use anyhow::Result;
use cudarc::driver::{sys, CudaEvent};
use lingxi_kernels::Kernels;
use std::cell::RefCell;

pub const NSEG: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Gdn = 0,
    Qsa = 1,
}

fn names(kind: Kind) -> [&'static str; NSEG] {
    match kind {
        Kind::Gdn => ["qkvz", "zproj", "ab", "conv", "rec", "out"],
        Kind::Qsa => ["proj", "kv", "idxprep", "idxsel", "attn", "out"],
    }
}

struct Dev {
    ctx: usize,
    /// `NSEG + 1` 个带计时事件：第 `s` 段的区间 = `evs[s] → evs[s+1]`
    evs: Vec<CudaEvent>,
}

struct St {
    /// 本层各段 µs
    cur: [f64; NSEG],
    /// [设备] × [种类] × [段] 累计
    sum: [[[f64; NSEG]; 2]; 2],
    n: [[f64; 2]; 2],
    /// [设备] × [种类] 的上下文长度累计（打印层均 len）
    len_sum: [[f64; 2]; 2],
    seen: u32,
    /// 当前 `begin` 打过的 (层, 种类)；`None` = 未激活（其它调用点直接走 no-op）
    key: Option<(u32, Kind)>,
    /// 上一次 mark 的段号（补记被跳过的边界用）
    last_idx: usize,
    /// 现场：t / len
    info: [u32; 2],
    devs: Vec<Dev>,
}

thread_local! {
    static ST: RefCell<St> = RefCell::new(St {
        cur: [0.0; NSEG], sum: [[[0.0; NSEG]; 2]; 2], n: [[0.0; 2]; 2],
        len_sum: [[0.0; 2]; 2], seen: 0, key: None,
        last_idx: 0, info: [0; 2], devs: Vec::new(),
    });
}

#[inline]
pub fn on() -> bool {
    crate::flag("LINGXI_ATTN_SYNC")
}

#[inline]
fn layer_round() -> u32 {
    std::env::var("LINGXI_ATTN_LAYER")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0)
}

/// 记录本步现场（t / 已缓存 token 数），供逐层行打印。
pub fn info(t: usize, len: usize) {
    if !on() {
        return;
    }
    ST.with(|c| c.borrow_mut().info = [t as u32, len as u32]);
}

/// attn 调用入口：激活探针并记第 0 个事件。由 `layer.rs` 在分派到
/// `qsa::attention_step` / `gdn::gdn_step_rec` **之前**调用。
pub fn begin(k: &Kernels, layer: u32, kind: Kind) -> Result<()> {
    if !on() {
        return Ok(());
    }
    let dev = (k.ctx.ordinal() & 1) as usize;
    let ctx = k.ctx.cu_ctx() as usize;
    ST.with(|c| -> Result<()> {
        let mut g = c.borrow_mut();
        let di = dev_index(k, &mut g, ctx, dev)?;
        g.key = Some((layer, kind));
        g.last_idx = 0;
        g.devs[di].evs[0].record(&k.stream)?;
        Ok(())
    })
}

/// 段边界。`idx`：1..=5 = 各段结束，6 = attn 段结束（层末 sync + 汇总打印）。
/// 未 `begin`（例如 `sequence.rs` 的草稿重放直接调 `gdn_step_rec`）时是 no-op。
pub fn mark(k: &Kernels, idx: usize) -> Result<()> {
    if !on() {
        return Ok(());
    }
    let dev = (k.ctx.ordinal() & 1) as usize;
    let ctx = k.ctx.cu_ctx() as usize;
    ST.with(|c| -> Result<()> {
        let mut g = c.borrow_mut();
        let Some((layer, kind)) = g.key else { return Ok(()) };
        let di = dev_index(k, &mut g, ctx, dev)?;
        // 某些段可能整段不存在（例如 GDN 重放跳过投影）⇒ 把跳过的边界**按当前位置补记**，
        // 跳过的段即为 0 宽（必须先补记再记 idx）。
        let last = g.last_idx;
        for j in last + 1..idx.min(NSEG) {
            g.devs[di].evs[j].record(&k.stream)?;
        }
        g.devs[di].evs[idx.min(NSEG)].record(&k.stream)?;
        g.last_idx = idx.min(NSEG);
        if idx < NSEG {
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
        let kk = kind as usize;
        // 只统计 48 层主干（`L100` 是 MTP 伪层的 attn，t/len 语义不同，单独看逐层行）。
        // （2026-10-27）：**汇总再按 t 分档**——t=1（decode）与 t≥2（prefill）混在一个
        // 均值里会把 145µs 的 decode 层和 5.5ms 的 prefill 层搅成一锅；prefill 战役只
        // 看 t≥2 档（t=1 档仍可从逐层行 `[attnL]` 读）。
        let real = (layer as usize) < 48 && g.info[0] >= 2;
        if real {
            for (a, b) in g.sum[dev][kk].iter_mut().zip(cur.iter()) {
                *a += *b;
            }
            g.n[dev][kk] += 1.0;
            g.len_sum[dev][kk] += g.info[1] as f64;
        }
        g.seen += 1;
        g.key = None; // 一次调用只量一层
        let nm = names(kind);
        if g.seen / 48 == layer_round() {
            let tot: f64 = cur.iter().sum();
            let mut line = String::new();
            for (j, name) in nm.iter().enumerate() {
                line.push_str(&format!("{name} {:6.1}／", cur[j]));
            }
            eprintln!(
                "[attnL] #{} dev{dev} L{layer:<2} {} t{} len{}｜{line}合计 {tot:.1} µs",
                g.seen,
                match kind {
                    Kind::Gdn => "GDN",
                    Kind::Qsa => "QSA",
                },
                g.info[0],
                g.info[1],
            );
        }
        let n = g.n[dev][kk];
        if real && n >= 120.0 {
            let a = g.sum[dev][kk];
            let tot: f64 = a.iter().sum();
            let avg_len = g.len_sum[dev][kk] / n;
            let mut line = String::new();
            for (j, name) in nm.iter().enumerate() {
                line.push_str(&format!(
                    "{name} {:6.1}（{:4.1}%）／",
                    a[j] / n,
                    100.0 * a[j] / tot.max(1e-9)
                ));
            }
            eprintln!(
                "[attn] dev{dev} {} L{layer} n={n:.0} len均={avg_len:.0} 层均 µs：{line}合计 {:.1}",
                match kind {
                    Kind::Gdn => "GDN",
                    Kind::Qsa => "QSA",
                },
                tot / n
            );
            g.sum[dev][kk] = [0.0; NSEG];
            g.n[dev][kk] = 0.0;
            g.len_sum[dev][kk] = 0.0;
        }
        Ok(())
    })
}

/// 取（或懒建）本上下文的事件槽。
fn dev_index(k: &Kernels, g: &mut St, ctx: usize, _dev: usize) -> Result<usize> {
    if let Some(i) = g.devs.iter().position(|d| d.ctx == ctx) {
        return Ok(i);
    }
    // ⚠️ 必须开 timing（cudarc 的 `new_event(None)` 是 DISABLE_TIMING）
    let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
    let mut evs = Vec::with_capacity(NSEG + 1);
    for _ in 0..=NSEG {
        evs.push(k.ctx.new_event(flags)?);
    }
    g.devs.push(Dev { ctx, evs });
    Ok(g.devs.len() - 1)
}
