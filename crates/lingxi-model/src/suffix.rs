//! **后缀查表草稿器**（suffix / prompt-lookup drafter）——零权重、零 GPU、微秒级。
//!
//! 逐字对齐参考实现 Strata 的 `include/strata/spec/suffix_drafter.hpp`
//! （**它在解码里是默认开启的**：`--suffix-draft 3`，且 `--lookup-chain` 建在它之上）。
//!
//! 动机：**一层 MTP 头的自喂链本来就只有 1~3 位有效**（我们实测位 0 = 0.98、位 1 = 0.79，
//! vLLM 对 Qwen 系 MTP 也只开 `num_speculative_tokens=1`，GLM-5.2 用 3）。
//! 而"输出在复述输入"的场景（改代码、重命名、引号里的原文、工具调用 JSON、枚举/列表）
//! **续写往往是原样复制** —— 这时从上下文里查表就能白拿一长串高接受率的草稿。
//!
//! 规则（与参考实现同口径）：
//! - 索引**历史里每个 trigram** → 最近的 `WAYS` 个结束位置（定长、开放寻址，内存 ≈ 20 B/历史 token）；
//! - 提议时把"当前后缀（+ 已产出的草稿）"当查询，逐个候选**向后延伸**比对，取最长（同长取最近）；
//! - 匹配长度 < `min_match` ⇒ 不提议；延伸上界 `max_match`。
//!
//! ⚠️ 它**只改"提哪些草稿"**，被采纳的 token 始终由主干验证决定 ⇒
//! 打开它**不应该改变任何对外输出**（判据：首 token + 固定贪心序列不变）。

use std::collections::HashMap;

/// 一个 trigram 保留多少个候选结束位置（参考实现 `WAYS = 4`）。
const WAYS: usize = 4;

pub struct SuffixDrafter {
    /// 历史 token（prompt + 每个被采纳的 token）
    hist: Vec<i32>,
    /// trigram 哈希 → 结束位置（最近在前，最多 `WAYS` 个）
    idx: HashMap<u64, Vec<u32>>,
    /// 少于这个匹配长度就什么都不提（参考实现默认 3）
    min_match: usize,
    /// 向后延伸的上界（参考实现默认 32）
    max_match: usize,
}

fn tri_hash(a: i32, b: i32, c: i32) -> u64 {
    // 简单可用的混合；不必抗碰撞（碰撞只会多几个候选、由逐元素比对兜住）
    let mut h = (a as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (b as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= (c as u32 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h
}

impl SuffixDrafter {
    pub fn new(min_match: usize) -> Self {
        Self { hist: Vec::new(), idx: HashMap::new(), min_match, max_match: 32 }
    }

    pub fn len(&self) -> usize {
        self.hist.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hist.is_empty()
    }

    /// 往历史里追加 token（prompt，然后是每个被采纳的 token）。
    pub fn append(&mut self, tokens: &[i32]) {
        for &t in tokens {
            self.hist.push(t);
            let n = self.hist.len();
            if n >= 3 {
                let k = tri_hash(self.hist[n - 3], self.hist[n - 2], self.hist[n - 1]);
                let v = self.idx.entry(k).or_default();
                v.insert(0, (n - 1) as u32);
                v.truncate(WAYS);
            }
        }
    }

    pub fn append_one(&mut self, t: i32) {
        self.append(&[t]);
    }

    /// 提议**不超过 `max_k` 个**后继 token（写在 `out` 末尾）；返回写进去的个数。
    ///
    /// `extra`：已经产出的草稿（`--lookup-chain`）——它和"历史后缀"一起构成被匹配的后缀，
    /// 所以查表会从"那些草稿结束的地方"继续。历史本身不变。
    pub fn propose_after(&self, extra: &[i32], max_k: usize, out: &mut Vec<i32>) -> usize {
        if max_k == 0 || self.hist.len() < 3 {
            return 0;
        }
        let n = self.hist.len();
        // 查询 = 历史后缀 + extra；只需要"尾部 L 个"参与比对，L 上界 = max_match
        let lmax = (self.min_match + extra.len()).max(self.max_match.min(n + extra.len()));
        let at = |i: usize| -> i32 {
            // i 从查询尾部往前数（0 = 最后一个 token）
            let back = i + 1;
            if back <= extra.len() {
                extra[extra.len() - back]
            } else {
                let hb = back - extra.len();
                if hb <= n {
                    self.hist[n - hb]
                } else {
                    i32::MIN // 越界哨兵：绝不会等于真实 token，从而截断延伸
                }
            }
        };
        // 用查询最后一个 trigram 找候选
        let (q2, q1, q0) = (at(2), at(1), at(0));
        if q2 == i32::MIN || q1 == i32::MIN || q0 == i32::MIN {
            return 0;
        }
        let cands = match self.idx.get(&tri_hash(q2, q1, q0)) {
            Some(v) => v,
            None => return 0,
        };
        let mut best_end: Option<usize> = None;
        let mut best_len = 0usize;
        for &e in cands {
            let e = e as usize;
            // ⚠️ **必须跳过"自己"**：查询的后缀一定能在历史的尾巴上平凡匹配到自己
            //    （而且因为延伸上界是 max_match，它永远是最长的那个）⇒
            //    若不跳过，最佳候选恒为尾部自身、提议恒为空。
            //    （第一版就是漏了这条：单元测试报"应当命中，实得 0"。）
            if e + 1 >= n {
                continue;
            }
            // 候选的"结束位置" e；向后延伸比对（i = 1,2,... 对应更早的 token）
            let mut m = 1usize;
            while m < lmax {
                let q = at(m);
                if q == i32::MIN || e < m {
                    break;
                }
                if self.hist[e - m] != q {
                    break;
                }
                m += 1;
            }
            // `m` = 匹配上的**全部** token 数（含 `extra` 那部分）—— 参考实现的
            // `--lookup-chain-min` 口径就是"in tokens, drafts included"。
            if m >= self.min_match && m > best_len {
                best_len = m;
                best_end = Some(e);
            }
        }
        let Some(e) = best_end else { return 0 };
        // 提议：历史里 e 之后、且**不与查询重叠**的 token
        let start = e + 1;
        let mut got = 0usize;
        while got < max_k && start + got < n.saturating_sub(extra.len()) {
            out.push(self.hist[start + got]);
            got += 1;
        }
        got
    }

    /// 不带 `extra` 的版本（等价于 `propose_after(&[], …)`）。
    pub fn propose(&self, max_k: usize, out: &mut Vec<i32>) -> usize {
        self.propose_after(&[], max_k, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_its_own_input() {
        // 历史里出现过 "capital of France is Paris"；现在写到 "capital of France is "，应能提 "Paris"
        let mut sd = SuffixDrafter::new(3);
        let h: Vec<i32> = "The capital of France is Paris. The".bytes().map(|b| b as i32).collect();
        sd.append(&h);
        let q: Vec<i32> = "capital of France is ".bytes().map(|b| b as i32).collect();
        sd.append(&q);
        let mut out = Vec::new();
        let n = sd.propose(8, &mut out);
        assert!(n >= 3, "应当命中，实得 {n}");
        let want: Vec<i32> = "Paris".bytes().map(|b| b as i32).collect();
        assert_eq!(&out[..want.len()], &want[..], "应续出 Paris，实得 {out:?}");
    }

    #[test]
    fn no_match_proposes_nothing() {
        let mut sd = SuffixDrafter::new(3);
        sd.append(&[1, 2, 3, 4, 5]);
        let mut out = Vec::new();
        assert_eq!(sd.propose_after(&[900, 901], 4, &mut out), 0);
    }

    #[test]
    fn extra_extends_the_query() {
        // 历史 "A B C D"，当前后缀 + extra = "A B C D" → 应在这里命中自己并续出后面的内容
        let mut sd = SuffixDrafter::new(3);
        sd.append(&[10, 20, 30, 40, 50, 60, 10, 20]);
        let mut out = Vec::new();
        // 后缀 "10 20" + extra "30 40" ⇒ 完整匹配到 [10,20,30,40] ⇒ 续 50,60
        let n = sd.propose_after(&[30, 40], 4, &mut out);
        assert_eq!(n, 2, "实得 {out:?}");
        assert_eq!(out, vec![50, 60]);
    }
}
