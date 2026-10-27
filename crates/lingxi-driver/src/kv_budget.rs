//! ㊿-P1 弹性 KV：**统一预算 + 准入控制**的记账核心（纯逻辑，无 GPU；serve 调度器消费）。
//!
//! 原则（对应 PLAN_NEXT P1）：不定死路数——一个 token 池，按"请求长度 + 冷热程度"
//! 弹性分配；预算不足时按冷度挤占（整会话下沉 host 或拒绝新会话）。
//! 懒分配的 `QsaState::ensure_cap`（lingxi-model）负责单会话内的几何增长，
//! 本模块负责**跨会话**的预算账本与准入裁决。

use std::collections::HashMap;

/// 一条驻留会话的预算占用项
#[derive(Debug, Clone, Copy)]
pub struct Occupant {
    /// 当前已用 token（= QsaState.len，含草稿）
    pub used: usize,
    /// 已承诺上限（= max_cap；记账口径按承诺算，防止超卖）
    pub pledged: usize,
    /// 冷度分：距上次访问的秒数（调度器每步刷新；越大越先被挤占）
    pub idle_secs: u64,
}

/// 统一 KV token 预算（每 GPU 各一份；双卡 split 下每会话只占其那半层的预算，
/// 但层间分布均匀 ⇒ 直接按全量 token 记账即可，不必乘层数）
#[derive(Debug)]
pub struct KvBudget {
    total: usize,
    occ: HashMap<u64, Occupant>,
}

impl KvBudget {
    pub fn new(total_tokens: usize) -> Self {
        Self { total: total_tokens, occ: HashMap::new() }
    }

    pub fn total(&self) -> usize { self.total }

    /// 已承诺总量
    pub fn pledged(&self) -> usize { self.occ.values().map(|o| o.pledged).sum() }

    /// 剩余可承诺量
    pub fn remaining(&self) -> usize { self.total.saturating_sub(self.pledged()) }

    pub fn contains(&self, id: u64) -> bool { self.occ.contains_key(&id) }

    /// 新会话准入：`need` = 请求的上下文上限（prompt + 预期生成长度）。
    /// 预算够 ⇒ 直接承诺；不够 ⇒ 找最冷且 pledged>used 的会话挤占，仍不够 ⇒ 拒绝。
    /// 返回被挤占的会话 id（调用方负责把它们的 KV 下沉 host / 结束）。
    pub fn admit(&mut self, id: u64, need: usize, min_ctx: usize) -> Result<Vec<u64>, ()> {
        if need > self.total {
            return Err(()); // 单请求就超池（如 256K 请求进小池）
        }
        let mut evicted = Vec::new();
        while need > self.remaining() {
            // 挤占候选：最冷优先；只挤"承诺多于实际用"的（先收缩承诺，不动数据）
            let cand = self.occ.iter()
                .filter(|(k, o)| **k != id && o.pledged > o.used.max(min_ctx))
                .max_by_key(|(_, o)| o.idle_secs)
                .map(|(k, _)| *k);
            match cand {
                Some(k) => {
                    let o = self.occ.get_mut(&k).unwrap();
                    // 收缩到"实际用量（不低于 min_ctx）"，不是踢出——数据还在设备上
                    o.pledged = o.used.max(min_ctx);
                    evicted.push(k);
                }
                None => {
                    // 没有可收缩的承诺：只能整会话下沉（pledged 归零 = 释放整份）
                    let cand = self.occ.iter()
                        .filter(|(k, _)| **k != id)
                        .max_by_key(|(_, o)| o.idle_secs)
                        .map(|(k, _)| *k);
                    match cand {
                        Some(k) => {
                            self.occ.remove(&k);
                            evicted.push(k);
                        }
                        None => return Err(()),
                    }
                }
            }
        }
        self.occ.insert(id, Occupant { used: 0, pledged: need, idle_secs: 0 });
        Ok(evicted)
    }

    /// 会话增长（QsaState 几何增长前先记账；超承诺则先尝试收缩他人承诺，同 admit 逻辑）
    pub fn grow(&mut self, id: u64, new_pledge: usize, min_ctx: usize) -> Result<Vec<u64>, ()> {
        if !self.occ.contains_key(&id) {
            return self.admit(id, new_pledge, min_ctx);
        }
        if new_pledge <= self.occ[&id].pledged {
            self.occ.get_mut(&id).unwrap().pledged = new_pledge.max(self.occ[&id].used);
            return Ok(Vec::new());
        }
        let delta = new_pledge - self.occ[&id].pledged;
        let mut evicted = Vec::new();
        while delta > self.remaining() {
            let cand = self.occ.iter()
                .filter(|(k, o)| **k != id && o.pledged > o.used.max(min_ctx))
                .max_by_key(|(_, o)| o.idle_secs)
                .map(|(k, _)| *k);
            match cand {
                Some(k) => {
                    let o = self.occ.get_mut(&k).unwrap();
                    o.pledged = o.used.max(min_ctx);
                    evicted.push(k);
                }
                None => return Err(()),
            }
        }
        self.occ.get_mut(&id).unwrap().pledged = new_pledge;
        Ok(evicted)
    }

    /// 每步刷新用量与冷度（调度器在 decode 轮转时调用）
    pub fn tick(&mut self, id: u64, used: usize, idle_secs: u64) {
        if let Some(o) = self.occ.get_mut(&id) {
            o.used = used;
            o.idle_secs = idle_secs;
        }
    }

    /// 会话结束（正常退出或拒绝后清理）
    pub fn release(&mut self, id: u64) -> bool { self.occ.remove(&id).is_some() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn occ_ids(b: &KvBudget) -> Vec<u64> {
        let mut v: Vec<u64> = b.occ.keys().copied().collect();
        v.sort();
        v
    }

    #[test]
    fn admit_basic() {
        let mut b = KvBudget::new(100_000);
        assert!(b.admit(1, 40_000, 1_000).is_ok());
        assert!(b.admit(2, 40_000, 1_000).is_ok());
        assert_eq!(b.remaining(), 20_000);
        assert!(b.admit(3, 110_000, 1_000).is_err()); // 单请求超池
        // （need ≤ total 时 admit 总能靠收缩/下沉既有会话腾够 ⇒ 唯一拒绝口径是超池）
        // 两会话都用满（无冗余）时进 25K：必须把其中一个整会话挤掉
        b.tick(1, 40_000, 0);
        b.tick(2, 40_000, 0);
        let ev = b.admit(3, 25_000, 1_000).unwrap();
        assert_eq!(ev.len(), 1);
        assert_eq!(b.pledged(), 65_000); // 剩一个 40K + 新 25K
        assert_eq!(b.remaining(), 35_000);
    }

    #[test]
    fn admit_shrinks_coldest_overcommitted() {
        let mut b = KvBudget::new(100_000);
        b.admit(1, 40_000, 1_000).unwrap();
        b.admit(2, 40_000, 1_000).unwrap();
        b.tick(1, 40_000, 100); // 1 号用满且很冷
        b.tick(2, 5_000, 5);    // 2 号冷度低、承诺多
        // 1 号承诺=实际（无冗余）⇒ 不在收缩候选里；2 号有 35K 冗余 ⇒ 被收缩
        let ev = b.admit(3, 50_000, 1_000).unwrap();
        assert_eq!(ev, vec![2]);
        assert_eq!(b.remaining(), 5_000); // 40K(1) + 5K(2 收缩后) + 50K(3) = 95K
    }

    #[test]
    fn admit_evicts_coldest_first() {
        let mut b = KvBudget::new(100_000);
        b.admit(1, 40_000, 1_000).unwrap();
        b.admit(2, 40_000, 1_000).unwrap();
        b.tick(1, 40_000, 500); // 1 号最冷
        b.tick(2, 40_000, 5);
        let ev = b.admit(3, 30_000, 1_000).unwrap();
        // 30K > remaining 20K ⇒ 先收缩最冷的 1 号到 used(40K) 无冗余……不可收缩
        // ⇒ 整会话下沉 1 号
        assert_eq!(ev, vec![1]);
        assert_eq!(occ_ids(&b), vec![2, 3]);
        assert_eq!(b.remaining(), 30_000);
    }

    #[test]
    fn grow_needs_budget() {
        let mut b = KvBudget::new(100_000);
        b.admit(1, 30_000, 1_000).unwrap();
        b.tick(1, 30_000, 0);
        let ev = b.grow(1, 80_000, 1_000).unwrap(); // 池够，无需挤占
        assert!(ev.is_empty());
        assert!(b.grow(1, 120_000, 1_000).is_err()); // 超池
    }

    #[test]
    fn shrink_pledge_on_growth() {
        let mut b = KvBudget::new(100_000);
        b.admit(1, 60_000, 1_000).unwrap();
        b.tick(1, 10_000, 300);
        b.admit(2, 30_000, 1_000).unwrap();
        b.tick(2, 30_000, 0);
        assert_eq!(b.remaining(), 10_000);
        // 2 号增长 25K：挤占最冷且冗余的 1 号（60K 承诺 vs 10K 实际）
        let ev = b.grow(2, 55_000, 1_000).unwrap();
        assert_eq!(ev, vec![1]);
        assert_eq!(b.pledged(), 65_000);
    }

    #[test]
    fn release_and_reject_oversize() {
        let mut b = KvBudget::new(50_000);
        assert!(b.admit(1, 60_000, 1_000).is_err()); // 单请求超池
        b.admit(1, 20_000, 1_000).unwrap();
        assert!(b.release(1));
        assert_eq!(b.remaining(), 50_000);
    }
}
