//! Router lookahead 预取的自适应控制器（`LINGXI_ARENA_PREFETCH`）与预测信号
//! （`LINGXI_ARENA_PREFETCH_SIGNAL`）。
//!
//! **预测信号（为什么换）**：第一版用「层 L 的 top-k 输出预测层 L+1 选同样专家」
//! （`topk_set`），实测命中率仅 **2.4%** —— 信号错了。换成同类开源推理引擎 next-layer 预取实践的
//! 原设计（`router_logits`，默认）：**用层 L+1 自己的 router 权重 W 作用在层 L 的
//! router 输入 h 上**，算真 logits 再取 top-k——不是照搬层 L 的 top-k 集合。
//!
//! **执行结构**（`router_logits` 信号）：
//!   1. 发射：`sparse_moe_forward` 在 router GEMV 入队后、首个同步前调用
//!      `ExpertSource::pf_predict_enqueue` —— 登记本层 router W 的设备地址，并在
//!      **stream2** 上异步发预测 GEMV（`logits_pred = W_{L+1}·h_L`，直写映射 host
//!      内存，完成记线程级事件）。放 stream2 是因为路由段末尾有一次全流同步，
//!      预测 GEMV 若挂 compute 流会把 ~50µs 加进**每层**的同步等待；
//!   2. 消费：同一层随后的 `load_hits` → `pf_issue`：事件自旋（此刻层 L 专家段已
//!      在 compute 流排了 ~0.5 ms ⇒ 事件必然就绪，自旋 ≈ 0）→ host top-k → 至多
//!      [`PF_MAX`] 个未驻留/在途专家经 stream2 异步 DMA 预取。
//!
//! claim 转正 / 在途槽不进路由表 / promote 两种信号完全同机制（见 `source.rs`）。
//!
//! **自适应休眠（判据 = 命中节省 vs 预测成本）**：每预取一个专家记一次"预测"并
//! 记入 DMA 机会成本 [`PF_DMA_PENALTY_NS`]；预测 GEMV / top-k 的实测 host 开销
//! 按 pass 记进 [`PfCtl::note_pred_cost`]。采样窗 = [`WINDOW`] 次预测，窗尾比较
//! `win_hit × MISS_SAVE_NS`（命中节省，一次 miss DMA 实测 350~450µs）与
//! `win_cost_ns`（预测成本）——节省 ≤ 成本 ⇒ 休眠 [`SLEEP_PASSES`] 个 pass
//! （一次 `load_hits` 调用 = 一个 pass，期间白占带宽不如不预取），睡醒打一行账。
//! 命中率统计与 DMA 完全解耦：命中在 claim（load_hits / load_missing 实际路由到
//! 该专家）时记。本模型 48 层全是 MoE（layer.rs 每层的 `w.moe`）⇒ 路由层间隔 = 1，
//! lookahead 目标 = L+1（层 47 回绕 0 = 下一 token 的首路由层）。

use half::bf16;

/// 采样窗：这么多次**预测**（每预取一个专家算一次）结一次账。
pub(crate) const WINDOW: u32 = 2048;
/// 节省 ≤ 成本时，休眠这么多个 **pass**（`load_hits` 调用）。
pub(crate) const SLEEP_PASSES: u64 = 8192;
/// 一张一批的预取上限（同款"至多 2 个"纪律）。
pub(crate) const PF_MAX: usize = 2;
/// 一次 miss DMA 的代价（实测 350~450µs）⇒ 一次预取命中的节省按此估算。
pub(crate) const MISS_SAVE_NS: u64 = 400_000;
/// 每个预取决策记入的 DMA **机会成本**（stream2/PCIe 带宽占用 + LRU 槽位搅动）。
/// 没有这笔账，"host 几乎零成本"的旧信号（topk_set）在经济判据下永不休眠、会
/// 白灌 PCIe —— topk_set 实测命中率只有 2.4%，正是这笔账让它被自动闲置。
pub(crate) const PF_DMA_PENALTY_NS: u64 = 50_000;
/// 同层 write-through 预取（`LINGXI_ARENA_TOPK_FETCH`）一张一批的上限。
/// 与 prefetch 同款"至多 2 个"纪律：这些 DMA 必被本批用到（非预测），
/// 但同发太多会齐挤 PCIe / copy engine，且 claim 时 compute 流要等整批 `pf_done`。
pub(crate) const TOPK_FETCH_MAX: usize = PF_MAX;

/// 预测信号（`LINGXI_ARENA_PREFETCH_SIGNAL`）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PfSignal {
    /// 原设计（默认）：层 L+1 的 router 权重 × 层 L 的 router 输入。
    RouterLogits,
    /// 旧信号（对照/消融）：层 L 的 top-k 集合原样预测层 L+1。实测命中率 2.4%。
    TopkSet,
}

impl PfSignal {
    pub(crate) fn from_env() -> Self {
        match std::env::var("LINGXI_ARENA_PREFETCH_SIGNAL").ok().as_deref() {
            None | Some("router_logits") => Self::RouterLogits,
            Some("topk_set") => Self::TopkSet,
            Some(other) => {
                eprintln!(
                    "[arena-prefetch] ⚠️ 未知 LINGXI_ARENA_PREFETCH_SIGNAL={other:?}，\
                     回退默认 router_logits"
                );
                Self::RouterLogits
            }
        }
    }

    /// 启动日志用的一句话描述。
    pub(crate) fn describe(self) -> &'static str {
        match self {
            Self::RouterLogits => {
                "router_logits（层 L+1 的 router W × 层 L 的 router 输入 h）"
            }
            Self::TopkSet => "topk_set（层 L top-k 原样预测层 L+1，旧信号，实测命中率 2.4%）",
        }
    }
}

/// 纯函数：预测 logits 的 top-k（插入式，全序 = **值降序、同值下标升序**——与 host
/// 路由选择规则同源；softmax 单调 ⇒ 对原始 logits 取 top-k 与对 probs 取逐位相同）。
pub(crate) fn pred_topk(logits: &[bf16], k: usize) -> Vec<u32> {
    let mut sel: Vec<(usize, f32)> = Vec::with_capacity(k.saturating_add(1).min(logits.len()));
    for (i, &v) in logits.iter().enumerate() {
        let p = v.to_f32();
        if sel.len() == k && !(p > sel[sel.len() - 1].1) {
            continue;
        }
        let pos = sel.iter().position(|&(_, v)| p > v).unwrap_or(sel.len());
        sel.insert(pos, (i, p));
        if sel.len() > k {
            sel.pop();
        }
    }
    sel.into_iter().map(|(i, _)| i as u32).collect()
}

/// 纯函数（`router_logits` 信号）：按预测 logits 的全序取至多 `max` 个
/// 「未驻留且不在途」的候选专家（跳过已不可用的，继续看下一个最优）。
pub(crate) fn pred_candidates(
    logits: &[bf16],
    max: usize,
    unavailable: &mut dyn FnMut(u32) -> bool,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(max);
    for e in pred_topk(logits, logits.len()) {
        if out.len() >= max {
            break;
        }
        if unavailable(e) {
            continue;
        }
        out.push(e);
    }
    out
}

/// 纯函数（`topk_set` 信号，旧）：层 L 的命中列表原样作为层 L+1 的候选（保持
/// 原 pf_issue 的遍历次序），同样滤掉已驻留/在途。
pub(crate) fn topk_set_candidates(
    hit_list: &[u32],
    max: usize,
    unavailable: &mut dyn FnMut(u32) -> bool,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(max);
    for &e in hit_list {
        if out.len() >= max {
            break;
        }
        if unavailable(e) {
            continue;
        }
        out.push(e);
    }
    out
}

/// 纯函数（`LINGXI_ARENA_TOPK_FETCH`）：同层 write-through 预取的候选选择 ——
/// 本层路由 top-k 列表里「**未驻留且不在途**」的其余专家，保持列表次序，至多
/// `max` 个。`unavailable` 谓词语义与 `source.rs::pf_unavailable` 相同（主/次
/// cache 任一已驻留 **或** `pf_pending` 在途 ⇒ 跳过）。topk-fetch 与 prefetch
/// **共用**在途表且同 key 互斥（HashMap insert 会静默覆盖旧条目）⇒ 在途者必须
/// 被滤掉，否则两者会双占 LRU 槽、丢在途条目。
pub(crate) fn topk_fetch_candidates(
    hit_list: &[u32],
    max: usize,
    unavailable: &mut dyn FnMut(u32) -> bool,
) -> Vec<u32> {
    topk_set_candidates(hit_list, max, unavailable)
}

pub(crate) struct PfCtl {
    /// 本实例是否启用（`LINGXI_ARENA_PREFETCH` 值感知开启 **且** 未开
    /// `LINGXI_ARENA_PRELOAD_ALL`——全驻留时无 miss 可省，预取自动闲置）。
    pub(crate) enabled: bool,
    pred_total: u64,
    hit_total: u64,
    /// 当前采样窗内的预测/命中计数（到窗尾清零重评）。
    win_pred: u32,
    win_hit: u32,
    /// pass 计数（每次预取决策点 +1，即每次 `load_hits` 调用）。
    passes: u64,
    /// 0 = 醒着；否则睡到 `passes` 达到该值（预测不划算的冷却期）。
    sleep_until: u64,
    n_sleep: u32,
    /// 预测成本总账（ns）= 实测 host 开销（GEMV 发射 / top-k / 自旋）+ 每笔
    /// 决策的 DMA 机会成本（[`PF_DMA_PENALTY_NS`]）。
    cost_total_ns: u64,
    /// 当前采样窗内的预测成本。
    win_cost_ns: u64,
    /// 上一 pass 的预测成本（`allow` 时结算，打印用）。
    last_pass_cost_ns: u64,
    cost_at_pass_start: u64,
}

impl PfCtl {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            pred_total: 0,
            hit_total: 0,
            win_pred: 0,
            win_hit: 0,
            passes: 0,
            sleep_until: 0,
            n_sleep: 0,
            cost_total_ns: 0,
            win_cost_ns: 0,
            last_pass_cost_ns: 0,
            cost_at_pass_start: 0,
        }
    }

    /// 本次 `load_hits` 是否允许发预测。pass 计数与**上 pass 成本结算**在这里
    /// 推进（休眠期也计，否则永远睡不醒）。休眠醒来时打一行账。
    pub(crate) fn allow(&mut self) -> bool {
        self.last_pass_cost_ns = self.cost_total_ns - self.cost_at_pass_start;
        self.cost_at_pass_start = self.cost_total_ns;
        self.passes += 1;
        if !self.enabled {
            return false;
        }
        if self.sleep_until > 0 {
            if self.passes < self.sleep_until {
                return false;
            }
            self.sleep_until = 0;
            eprintln!(
                "[arena-prefetch] 休眠 {SLEEP_PASSES} pass 结束：恢复预测（累计预测 {} / 命中 {}｜\
                 预测成本 {:.2} ms vs 命中节省估计 {:.2} ms）",
                self.pred_total,
                self.hit_total,
                self.cost_total_ns as f64 / 1e6,
                self.hit_total.saturating_mul(MISS_SAVE_NS) as f64 / 1e6,
            );
        }
        true
    }

    /// 发射端（`pf_predict_enqueue`）查询：当前是否在休眠冷却期（不推进 pass）。
    /// 休眠期连预测 GEMV 都不发 —— 白占 stream2。
    pub(crate) fn sleeping(&self) -> bool {
        self.sleep_until > self.passes
    }

    /// 预测成本的实测 host 开销记账（GEMV 发射 / 事件自旋 + top-k）。
    pub(crate) fn note_pred_cost(&mut self, ns: u64) {
        self.cost_total_ns += ns;
        self.win_cost_ns += ns;
    }

    /// 每预取一个专家记一次预测；到窗尾结一次账：命中节省 vs 预测成本，不划算 ⇒ 休眠。
    pub(crate) fn note_predicted(&mut self) {
        self.pred_total += 1;
        self.win_pred += 1;
        // 每笔决策记入 DMA 机会成本（与信号无关：就算 host 零成本，预取 DMA 本身
        // 也占 stream2/PCIe 与 LRU 槽位）。
        self.cost_total_ns += PF_DMA_PENALTY_NS;
        self.win_cost_ns += PF_DMA_PENALTY_NS;
        if self.win_pred >= WINDOW {
            let rate = self.win_hit as f64 / self.win_pred as f64;
            let benefit_ns = self.win_hit as u64 * MISS_SAVE_NS;
            eprintln!(
                "[arena-prefetch] 窗口账：命中率 {:.1}%（{}/{}）｜命中节省 ≈ {:.2} ms \
                 vs 预测成本 {:.2} ms（上 pass {:.1} µs）{}",
                rate * 100.0,
                self.win_hit,
                self.win_pred,
                benefit_ns as f64 / 1e6,
                self.win_cost_ns as f64 / 1e6,
                self.last_pass_cost_ns as f64 / 1e3,
                if benefit_ns <= self.win_cost_ns { "⇒ 不划算" } else { "" },
            );
            if benefit_ns <= self.win_cost_ns {
                self.n_sleep += 1;
                self.sleep_until = self.passes + SLEEP_PASSES;
                eprintln!(
                    "[arena-prefetch] ⚠️ 命中节省 ≤ 预测成本（{:.1}% 命中）：休眠 {SLEEP_PASSES} 个 \
                     pass（第 {} 次休眠；累计预测 {} / 命中 {}｜预测成本 {:.2} ms vs 节省估计 {:.2} ms）",
                    rate * 100.0,
                    self.n_sleep,
                    self.pred_total,
                    self.hit_total,
                    self.cost_total_ns as f64 / 1e6,
                    self.hit_total.saturating_mul(MISS_SAVE_NS) as f64 / 1e6,
                );
            }
            self.win_pred = 0;
            self.win_hit = 0;
            self.win_cost_ns = 0;
        }
    }

    /// 预取命中：实际路由到了预取过的专家（claim 时记）。
    pub(crate) fn note_hit(&mut self) {
        self.hit_total += 1;
        self.win_hit += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits_of(vals: &[f32]) -> Vec<bf16> {
        vals.iter().map(|&v| bf16::from_f32(v)).collect()
    }

    #[test]
    fn 关闭时永远不允许() {
        let mut c = PfCtl::new(false);
        for _ in 0..10_000 {
            assert!(!c.allow());
        }
    }

    #[test]
    fn 全命中不休眠() {
        let mut c = PfCtl::new(true);
        for _ in 0..(WINDOW as u64 * 3) {
            assert!(c.allow());
            c.note_predicted();
            c.note_hit();
        }
        assert_eq!(c.sleep_until, 0);
    }

    #[test]
    fn 低命中休眠后自动醒来() {
        let mut c = PfCtl::new(true);
        let mut slept = false;
        let mut awake_after_sleep = 0u64;
        for i in 0..(WINDOW as u64 * 2 + SLEEP_PASSES + 8) {
            if !c.allow() {
                slept = true;
                continue;
            }
            c.note_predicted(); // 从不 note_hit ⇒ 命中率 0%
            if slept {
                awake_after_sleep += 1;
            }
            let _ = i;
        }
        assert!(slept, "低命中必须触发休眠");
        // 醒来后必须能继续预测（至少 WINDOW 次评下一个窗）。
        assert!(awake_after_sleep >= WINDOW as u64);
    }

    #[test]
    fn 高命中不休眠低命中休眠() {
        let mut c = PfCtl::new(true);
        // 第一窗 100% 命中。
        for _ in 0..WINDOW {
            assert!(c.allow());
            c.note_predicted();
            c.note_hit();
        }
        assert_eq!(c.sleep_until, 0);
        // 第二窗 0% 命中 ⇒ 进休眠。
        for _ in 0..WINDOW {
            assert!(c.allow());
            c.note_predicted();
        }
        assert!(c.sleep_until > 0);
    }

    #[test]
    fn 命中率不低但预测成本更贵也休眠() {
        // 判据是"命中节省 vs 预测成本"，不是单纯命中率：50% 命中、但每次预测
        // 的实测成本 ≈ 一次 miss 的节省 ⇒ 不划算 ⇒ 休眠。
        let mut c = PfCtl::new(true);
        for i in 0..WINDOW {
            assert!(c.allow());
            c.note_pred_cost(MISS_SAVE_NS); // 实测 host 成本 ≈ 一次 miss
            c.note_predicted();
            if i % 2 == 0 {
                c.note_hit(); // 50% 命中
            }
        }
        // 节省 = 1024×400µs ≈ 410 ms ≤ 成本 = 2048×(400+50)µs ≈ 922 ms ⇒ 休眠。
        assert!(c.sleep_until > 0, "节省 ≤ 成本必须休眠（哪怕命中率 50%）");
    }

    #[test]
    fn 休眠期sleeping为真且不推进pass() {
        let mut c = PfCtl::new(true);
        for _ in 0..WINDOW {
            c.allow();
            c.note_predicted(); // 0% 命中 ⇒ 休眠
        }
        assert!(c.sleep_until > 0);
        let passes = { c.passes };
        assert!(c.sleeping());
        assert_eq!(c.passes, passes, "sleeping() 不得推进 pass");
    }

    #[test]
    fn pred_topk_选择正确() {
        // 值降序取 top-k。
        let l = logits_of(&[0.1, 0.9, 0.5, 0.3, 0.8]);
        assert_eq!(pred_topk(&l, 1), vec![1]);
        assert_eq!(pred_topk(&l, 2), vec![1, 4]);
        assert_eq!(pred_topk(&l, 3), vec![1, 4, 2]);
        // k ≥ len ⇒ 全序。
        assert_eq!(pred_topk(&l, 5), vec![1, 4, 2, 3, 0]);
        // 全负值也行（max 归一化前的原始 logits）。
        let l2 = logits_of(&[-3.0, -1.0, -2.0]);
        assert_eq!(pred_topk(&l2, 2), vec![1, 2]);
    }

    #[test]
    fn pred_topk_并列按下标升序() {
        // 同值 ⇒ 下标升序（与 host 路由的全序一致）。
        let l = logits_of(&[0.5, 0.9, 0.5, 0.9, 0.1]);
        assert_eq!(pred_topk(&l, 3), vec![1, 3, 0]);
        assert_eq!(pred_topk(&l, 4), vec![1, 3, 0, 2]);
    }

    #[test]
    fn pred_candidates_跳过驻留与在途并保持全序() {
        // logits：专家 3 > 1 > 4 > 0 > 2。专家 3 已驻留、专家 1 在途 ⇒
        // 候选 = [4, 0]（跳过不可用的，继续取下一个最优），至多 PF_MAX=2 个。
        let l = logits_of(&[0.3, 0.9, 0.1, 1.0, 0.7]);
        let unavailable = |e: u32| e == 3 || e == 1;
        assert_eq!(pred_candidates(&l, PF_MAX, &mut unavailable.clone()), vec![4, 0]);
        // 全不可用 ⇒ 空。
        assert!(pred_candidates(&l, PF_MAX, &mut |_| true).is_empty());
        // max=1 ⇒ 只取最优可用。
        assert_eq!(pred_candidates(&l, 1, &mut unavailable.clone()), vec![4]);
    }

    #[test]
    fn topk_set候选保持hit_list次序() {
        // 旧信号：hit_list 次序即候选次序（原 pf_issue 语义），只滤已驻留/在途。
        let unavailable = |e: u32| e == 10;
        assert_eq!(
            topk_set_candidates(&[7, 10, 3, 12], PF_MAX, &mut unavailable.clone()),
            vec![7, 3]
        );
        assert_eq!(
            topk_set_candidates(&[10, 11], PF_MAX, &mut unavailable.clone()),
            vec![11]
        );
        assert!(topk_set_candidates(&[], PF_MAX, &mut |_| false).is_empty());
    }

    #[test]
    fn topk_fetch候选_减已驻留减在途() {
        // top-k [5, 2, 9, 7, 3]；专家 2/9 已驻留、专家 7 在途 ⇒ 候选 = [5, 3]
        // （跳过不可用的、保持列表次序、至多 TOPK_FETCH_MAX 个）。谓词 = source.rs
        // `pf_unavailable` 的纯函数版（驻留 ∨ 在途）。
        let resident = [2u32, 9];
        let pending = [7u32];
        assert_eq!(
            topk_fetch_candidates(&[5, 2, 9, 7, 3], TOPK_FETCH_MAX, &mut |e| {
                resident.contains(&e) || pending.contains(&e)
            }),
            vec![5, 3]
        );
        // 全驻留/在途 ⇒ 空。
        assert!(topk_fetch_candidates(&[2, 9, 7], TOPK_FETCH_MAX, &mut |e| {
            resident.contains(&e) || pending.contains(&e)
        })
        .is_empty());
        // max=1 ⇒ 只取首个可用。
        assert_eq!(
            topk_fetch_candidates(&[2, 5, 3], 1, &mut |e| resident.contains(&e)),
            vec![5]
        );
    }

    #[test]
    fn topk_fetch与prefetch共用pf_pending互不双发() {
        // 模拟两者**并发共用一张在途表**：prefetch 先为层 L 选候选并登记在途，
        // topk-fetch 之后为同一层同一批 top-k 再选 —— 在途者必须被跳过，两边
        // 选出的集合不相交（同 key 双发会双占 LRU 槽 / 覆盖 pf_pending 条目）。
        let layer = 3u32;
        let topk: Vec<u32> = vec![1, 6, 4, 2, 8, 5];
        let mut resident: std::collections::HashSet<u32> = [1u32, 4].into_iter().collect();
        let mut pf_pending: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::new();
        // prefetch 先挑（层 L 预测候选），登记在途。
        let pf_sel = topk_fetch_candidates(&topk, TOPK_FETCH_MAX, &mut |e| {
            resident.contains(&e) || pf_pending.contains_key(&(layer, e))
        });
        assert_eq!(pf_sel, vec![6, 2]);
        for (i, &e) in pf_sel.iter().enumerate() {
            pf_pending.insert((layer, e), 100 + i); // 槽号随意
        }
        // topk-fetch 后挑同一列表：在途的 6/2 必须跳过，且与 prefetch 的选择不相交。
        let tf_sel = topk_fetch_candidates(&topk, TOPK_FETCH_MAX, &mut |e| {
            resident.contains(&e) || pf_pending.contains_key(&(layer, e))
        });
        assert!(tf_sel.iter().all(|e| !pf_sel.contains(e)), "两路选择必须不相交");
        assert_eq!(tf_sel, vec![8, 5]);
        // 再登记后第三次选择 ⇒ 空（无重复、无覆盖）。
        for (i, &e) in tf_sel.iter().enumerate() {
            pf_pending.insert((layer, e), 200 + i);
        }
        assert!(
            topk_fetch_candidates(&topk, TOPK_FETCH_MAX, &mut |e| {
                resident.contains(&e) || pf_pending.contains_key(&(layer, e))
            })
            .is_empty()
        );
        assert_eq!(pf_pending.len(), 4, "同 key 不得互相覆盖");
        // 在途表键值完好：prefetch 登记的槽号未被 topk-fetch 覆盖。
        assert_eq!(pf_pending[&(layer, 6)], 100);
        assert_eq!(pf_pending[&(layer, 2)], 101);
        // claim 转正（驻留集扩大）同样被跳过：8 转正后再选 ⇒ 只剩在途的 5 也不可选。
        resident.insert(8);
        assert!(
            topk_fetch_candidates(&topk, TOPK_FETCH_MAX, &mut |e| {
                resident.contains(&e) || pf_pending.contains_key(&(layer, e))
            })
            .is_empty()
        );
    }
}
