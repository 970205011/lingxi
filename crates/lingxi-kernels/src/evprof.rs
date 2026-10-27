//! **不改数学的分段 GPU 计时**（`LINGXI_EVPROF=1`）—— 每层每个段前后各记一个**带计时**的事件，
//! 跑完用 `cuEventElapsedTime` 求和。
//!
//! **为什么不能用现成的两种口径**（都是踩过的坑）：
//! * `LINGXI_KTS` 的同步括号**结尾是一次 `stream.synchronize`** ⇒ 量到的是"**排空**时间"
//! 而不是"这段的时间"（文档记的偏差 +25%，而且段与段会互相借时间）；
//! * `LINGXI_ARENA_NOH2D` 那类**会改变数学**的消融 ⇒ 输出变了 ⇒ 路由/接受率/每层唯一专家数
//! 全变 ⇒ 量的是**另一个负载**（本轮实测：它"省下"的 28 ms 是假的）。
//!
//! 事件记在**流内**、**不排空流水**（`cuEventRecord` 只是插一个标记，`cuStreamWaitEvent` 才等待），
//! 所以既拿到真 GPU 时间、又不扰动调度。代价：每段 2 次 `cuEventRecord`（~1~2 µs），
//! 只在探针打开时有。
//!
//! **汇总口径**：每个 `(层, 段)` 一对事件、每轮覆写 ⇒ `dump` 给的是**最后一轮的逐段账**
//! （对一个 300 token 的跑，最后一轮与稳态没有区别）。

use anyhow::Result;
use cudarc::driver::{sys, CudaContext, CudaEvent, CudaStream};
use std::sync::Arc;

/// 每层的段数（下标即 `seg`）。
/// `3`/`4` 是 **MoE 内部的再切**：`3` 从"进 MoE"到"路由算完、槽还没查"（shared+router+host），
/// `4` 是"查槽 + 专家 GEMV + 归并"。这两段一旦分开，就能回答"MoE 到底是**主机/同步受限**
/// 还是 **GEMV 带宽受限**"—— 2026-10-16 实测前者：MoE 占主干 78%，但 GEMV 的字节只值 0.6 ms/层。
pub const NSEG: usize = 7;
/// 段名（打印用）。
pub const SEG_NAMES: [&str; NSEG] = [
    "入口+PLE+attn_GR",
    "attn(QSA/GDN)",
    "mlp_GR",
    "MoE:路由+host段",
    "MoE:专家段",
    "gr_write(收尾)",
    "head(lm_head+采样)",
];
/// 预留的最大层数（事件对是这个数量 × `NSEG`）。
/// `MAX_LAYERS-1` 号层是 **head 伪层**（decode 的 lm_head+argmax 计时，seq 6）。
const MAX_LAYERS: usize = 64;
/// head 伪层的层号 = 真实层数（调用方传 `n_layers`），段号用 6。
pub const HEAD_SEG: usize = 6;

pub struct EvProf {
    /// `(begin, end)`，下标 = `layer * NSEG + seg`
    pairs: Vec<(CudaEvent, CudaEvent)>,
    /// 每层的**现场信息**，`[0]=s(对数) [1]=e_uniq(唯一专家/组数) [2]=rows [3]=n_miss`。
    /// 为什么要它：专家段**层间差 8×**（0.57↔3.59 ms），而"字节量"只差 1.6× ⇒ 一定有别的东西在变。
    /// 把"现场"和"耗时"并排打出来，就能一次回答"是 miss 数、还是组数、还是行数"。
    /// `UnsafeCell`：单线程逐层调用（与 `ArenaExpertSource` 的 cache 同一前提）。
    info: Vec<[std::sync::atomic::AtomicU32; 4]>,
    /// per-token trace 输出（`LINGXI_EVPROF_TRACE=<路径>`）：每 decode token 一行 JSON，
    /// 含 wall/各段和/逐 (层,段) 明细，给 HTML 分布报告用。`None` = 关（零开销）。
    trace: Option<std::sync::Mutex<std::fs::File>>,
}

impl EvProf {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        // ⚠️ **必须开 timing**：cudarc 的 `new_event(None)` 建的是 `CU_EVENT_DISABLE_TIMING`，
        // 拿它求 `cuEventElapsedTime` 会直接失败。
        let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let mut pairs = Vec::with_capacity(MAX_LAYERS * NSEG);
        for _ in 0..MAX_LAYERS * NSEG {
            pairs.push((ctx.new_event(flags)?, ctx.new_event(flags)?));
        }
        // trace 文件**追加**开：双卡各持一个 EvProf、每 token 各写一行（调用方顺序调用，
        // 同一进程内不会交错）。
        let trace = std::env::var("LINGXI_EVPROF_TRACE").ok().map(|path| {
            std::sync::Mutex::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .unwrap_or_else(|e| panic!("LINGXI_EVPROF_TRACE 打不开 {path}: {e}")),
            )
        });
        Ok(Self {
            pairs,
            info: (0..MAX_LAYERS)
                .map(|_| std::array::from_fn(|_| std::sync::atomic::AtomicU32::new(0)))
                .collect(),
            trace,
        })
    }

    /// 每 decode token 落一行 JSON：`{gpu,tok,wall_ms,busy_ms,head_ms,segs:[7],layers:[[l,seg,ms,i0..i3]]}`。
    ///
    /// 调用点保证：本 token 的事件**已全部完成**（decode_token 末尾的 argmax D2H 是同步
    /// 拷贝，排在同流全部工作之后）⇒ 读 elapsed 不需要额外 sync。但**对侧 GPU** 的流不受
    /// 本流 D2H 约束 ⇒ `stream` 参数用于先同步（trace 是剖剖析模式，每 token 一次 sync 可接受）。
    pub fn trace_token(
        &self,
        stream: &CudaStream,
        gpu: u32,
        tok: usize,
        wall_ms: f64,
        n_layers: u32,
    ) {
        let Some(f) = self.trace.as_ref() else { return };
        stream.synchronize().ok();
        let nl = (n_layers as usize + 1).min(MAX_LAYERS); // +1 = head 伪层
        let mut segs = [0f64; NSEG];
        let mut layers_json = String::with_capacity(nl * NSEG * 24);
        layers_json.push('[');
        let mut first = true;
        let mut busy = 0f64;
        // ㊽+11 每层 wall（seg0 begin → 该层最后一个非零段的 end）⇒ 层内气泡 =
        // lwall − 段和；层间气泡 = verify wall − Σ lwall。气泡二分的基础设施。
        let mut lwall_json = String::with_capacity(nl * 10);
        lwall_json.push('[');
        for l in 0..nl {
            let mut lwall = 0f64;
            if let Some((b0, _)) = self.pairs.get(l * NSEG) {
                for s in 0..NSEG {
                    if let Some((_, e)) = self.pairs.get(l * NSEG + s) {
                        if let Ok(ms) = b0.elapsed_ms(e) {
                            lwall = lwall.max(ms as f64);
                        }
                    }
                }
            }
            if l > 0 {
                lwall_json.push(',');
            }
            lwall_json.push_str(&format!("{lwall:.4}"));
            for s in 0..NSEG {
                let Some((b, e)) = self.pairs.get(l * NSEG + s) else { continue };
                let Ok(ms) = b.elapsed_ms(e) else { continue };
                let ms = ms as f64;
                if ms <= 0.0 {
                    continue;
                }
                segs[s] += ms;
                busy += ms;
                if !first {
                    layers_json.push(',');
                }
                first = false;
                let i = self.info_of(l as u32);
                layers_json.push_str(&format!(
                    "[{l},{s},{ms:.4},{},{},{},{}]",
                    i[0], i[1], i[2], i[3]
                ));
            }
        }
        layers_json.push(']');
        lwall_json.push(']');
        let head = segs[HEAD_SEG];
        let line = format!(
            "{{\"gpu\":{gpu},\"tok\":{tok},\"wall_ms\":{wall_ms:.4},\"busy_ms\":{busy:.4},\"head_ms\":{head:.4},\"segs\":[{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}],\"lwall\":{lwall_json},\"layers\":{layers_json}}}\n",
            segs[0], segs[1], segs[2], segs[3], segs[4], segs[5], segs[6]
        );
        use std::io::Write;
        if let Ok(mut f) = f.lock() {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// 记一条现场信息（`idx`：0=s、1=组数、2=rows、3=miss 数）。
    #[inline]
    pub fn set_info(&self, layer: u32, idx: usize, val: u32) {
        if let Some(c) = self.info.get(layer as usize) {
            c[idx].store(val, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn info_of(&self, layer: u32) -> [u32; 4] {
        match self.info.get(layer as usize) {
            Some(c) => std::array::from_fn(|i| c[i].load(std::sync::atomic::Ordering::Relaxed)),
            None => [0; 4],
        }
    }

    #[inline]
    pub fn begin(&self, layer: u32, seg: usize, stream: &CudaStream) {
        if let Some((b, _)) = self.pairs.get(layer as usize * NSEG + seg) {
            let _ = b.record(stream);
        }
    }

    #[inline]
    pub fn end(&self, layer: u32, seg: usize, stream: &CudaStream) {
        if let Some((_, e)) = self.pairs.get(layer as usize * NSEG + seg) {
            let _ = e.record(stream);
        }
    }

    /// 汇总打印（每个 `(层,段)` 取**最后一次记录**）。`n_layers` = 实际层数。
    pub fn dump(&self, n_layers: u32) {
        let mut sum = [0f64; NSEG];
        let mut cnt = [0usize; NSEG];
        let mut lo = [f64::MAX; NSEG];
        let mut hi = [0f64; NSEG];
        for l in 0..n_layers as usize {
            for s in 0..NSEG {
                let idx = l * NSEG + s;
                let Some((b, e)) = self.pairs.get(idx) else { continue };
                let Ok(ms) = b.elapsed_ms(e) else { continue };
                let ms = ms as f64;
                if ms <= 0.0 {
                    continue; // 没记录过的对（例如该层没有这一段）会返回 0
                }
                sum[s] += ms;
                cnt[s] += 1;
                lo[s] = lo[s].min(ms);
                hi[s] = hi[s].max(ms);
            }
        }
        let tot: f64 = sum.iter().sum();
        eprintln!(
            "  [evprof] 逐段真 GPU 时间（**最后一轮**、逐层求和；事件不排空流水，口径干净）"
        );
        for s in 0..NSEG {
            if cnt[s] == 0 {
                continue;
            }
            eprintln!(
                "    {:<18} 合计 {:8.2} ms｜层均 {:7.3} ms｜min {:6.3} / max {:6.3}｜占 {:4.1}%（{} 层）",
                SEG_NAMES[s],
                sum[s],
                sum[s] / cnt[s] as f64,
                lo[s],
                hi[s],
                100.0 * sum[s] / tot.max(1e-9),
                cnt[s],
            );
        }
        eprintln!("    {:<18} 合计 {:8.2} ms（= 该轮主干 GPU 时间）", "★ 五段相加", tot);
        // 逐层最慢的 8 个（只报"专家段"）：**层间 8× 的方差**是最大的线索 ——
        // 它说明问题不在"总共搬了多少字节"，而在"某些层跑了另一条路"。
        let mut v: Vec<(f64, u32)> = (0..n_layers)
            .filter_map(|l| {
                let (b, e) = &self.pairs[l as usize * NSEG + 4];
                b.elapsed_ms(e).ok().map(|ms| (ms as f64, l))
            })
            .collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("  [evprof] 专家段：最慢 8 层（后面括号里是现场 s/组数/rows/miss）");
        for (ms, l) in v.iter().take(8) {
            let i = self.info_of(*l);
            eprintln!("    层{l:2} = {ms:5.2} ms  [s={} 组={} rows={} miss={}]", i[0], i[1], i[2], i[3]);
        }
        // ★ **相关性**：把"现场"分组求均值，一次回答"是 miss 数在作怪，还是组数/行数"。
        let mut by_miss = [(0f64, 0usize); 4]; // miss=0 / 1 / 2 / >=3
        let mut by_grp = [(0f64, 0usize); 4]; // 组数 <26 / 26-30 / 31-34 / >=35
        for (ms, l) in v.iter() {
            let i = self.info_of(*l);
            let mi = (i[3] as usize).min(3);
            by_miss[mi].0 += ms;
            by_miss[mi].1 += 1;
            let gi = match i[1] {
                0..=25 => 0,
                26..=30 => 1,
                31..=34 => 2,
                _ => 3,
            };
            by_grp[gi].0 += ms;
            by_grp[gi].1 += 1;
        }
        let fmt = |a: &[(f64, usize)], names: &[&str]| -> String {
            let mut out: Vec<String> = Vec::new();
            for i in 0..a.len().min(names.len()) {
                if a[i].1 == 0 {
                    continue;
                }
                out.push(format!("{}:{:5.2}({}层)", names[i], a[i].0 / a[i].1 as f64, a[i].1));
            }
            out.join("  ")
        };
        eprintln!("  [evprof] 专家段按 miss 分组层均 ms：{}", fmt(&by_miss, &["miss=0", "miss=1", "miss=2", "miss>=3"]));
        eprintln!("  [evprof] 专家段按组数分组层均 ms：{}", fmt(&by_grp, &["组<26", "组26-30", "组31-34", "组>=35"]));
    }
}
