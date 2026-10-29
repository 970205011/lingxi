//! R4 拆分（2026-10-21）：原 serve.rs 3152 行按职责分段，本模块为其中一段。
//! 模块总览见 `serve/mod.rs`。段间共享类型一律 `pub(crate)`。

use anyhow::Result;
use std::time::Instant;


use super::cache::*;


pub(crate) struct Sunk {
    pub(crate) cp: lingxi_driver::HostCheckpoint,
    pub(crate) bytes: usize,
    pub(crate) at: Instant,
}

/// 物化中的会话（v1.5 多物化：每个常驻会话一份）。
pub(crate) struct LiveSeq {
    pub(crate) seq: lingxi_driver::Sequence,
    pub(crate) epoch: u64,
    pub(crate) at: Instant, // 最近活跃（LRU 记账）
}

pub struct Sessions {
    cur_sid: Option<String>,
    live: std::collections::HashMap<String, LiveSeq>,
    sunk: std::collections::HashMap<String, Sunk>,
    budget: lingxi_driver::KvBudget,
    sunk_cap: usize,
    sunk_bytes: usize,
    max_mat: usize,
}

/// 预算记账用的会话键（稳定 hash）
pub(crate) fn sid_key(sid: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    sid.hash(&mut h);
    h.finish()
}

impl Sessions {
    pub(crate) fn new(max_ctx: usize) -> Self {
        let sunk_cap = std::env::var("LINGXI_SERVE_SUNK_MB")
            .ok().and_then(|s| s.parse().ok()).filter(|&m| m > 0)
            .unwrap_or(32768)
            * 1024 * 1024;
        // v1.5 多物化上限：同时物化的 Sequence 数（KV 预算外的显存零和由 LINGXI_SERVE_MAX_CTX
        // 的 pledge 口径管着；每个物化会话都吃自己的 pledge）。
        let max_mat: usize = std::env::var("LINGXI_SERVE_MATERIALIZED")
            .ok().and_then(|s| s.parse().ok()).filter(|&m| m > 0)
            .unwrap_or(2);
        Self {
            cur_sid: None,
            live: std::collections::HashMap::new(),
            sunk: std::collections::HashMap::new(),
            budget: lingxi_driver::KvBudget::new(max_ctx),
            sunk_cap,
            sunk_bytes: 0,
            max_mat,
        }
    }

    pub(crate) fn current_mut(&mut self) -> Option<&mut lingxi_driver::Sequence> {
        self.cur_sid.as_ref().and_then(|s| self.live.get_mut(s)).map(|l| &mut l.seq)
    }

    /// 引擎步进前调用：把当前指针 + CUDA Graph epoch 切到已物化的 sid（零搬迁）。
    pub(crate) fn activate(&mut self, sid: &str) {
        if self.cur_sid.as_deref() == Some(sid) {
            return;
        }
        if let Some(l) = self.live.get(sid) {
            lingxi_model::graph::epoch_switch(l.epoch);
        }
        self.cur_sid = Some(sid.to_string());
    }

    /// 确保 sid 已物化；返回前完成 沉没当前 / 唤醒目标 / 新建 三选一并做好预算与图隔离。
    pub(crate) fn ensure(
        &mut self,
        sid: &str,
        need: usize,
        eng: &lingxi_driver::Engine<'_>,
        mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
        cache: &mut PrefillCache,
        protect: &[String],
    ) -> Result<()> {
        const MIN_CTX: usize = 1024;
    // 记录缓存属主：Dev 检查点按 owner 判有效性（v1.5 私有工作区，切换不覆盖别家）。
    cache.set_owner(sid);
    if self.cur_sid.as_deref() == Some(sid) {
        // 同会话：只可能 pledge 增长
        self.budget.grow(sid_key(sid), need, MIN_CTX)
            .map_err(|_| anyhow::anyhow!("KV 预算不足（会话 {sid} 增长到 {need}）"))?;
        if let Some(l) = self.live.get_mut(sid) { l.at = Instant::now(); }
        return Ok(());
    }
    // 已物化 ⇒ 切换 = 换 epoch（图集各自保留、零重录），零数据搬迁。
    // 别家会话的 Dev 检查点**不作废**——KV 工作区是各会话私有的（v1 时代的一刀切
    // invalidate 会让刚离开又回来的会话丢掉全部热检查点，长文档多轮实测踩过）。
    if let Some(l) = self.live.get_mut(sid) {
        let ep = l.epoch;
        l.at = Instant::now();
        lingxi_model::graph::epoch_switch(ep);
        self.cur_sid = Some(sid.to_string());
        self.budget.grow(sid_key(sid), need, MIN_CTX)
            .map_err(|_| anyhow::anyhow!("KV 预算不足（会话 {sid} 增长到 {need}）"))?;
        return Ok(());
    }
    // 1) 未物化：若已达多物化上限，先把最久未动的 live 会话沉掉（图 epoch 随之回收）。
    //    `protect` = 正在生成中的会话（并发路径传入）——绝不当牺牲者。
    if self.live.len() >= self.max_mat {
        let victim = self
            .live
            .iter()
            .filter(|(k, _)| !protect.iter().any(|p| p == *k))
            .min_by_key(|(_, l)| l.at)
            .map(|(k, _)| k.clone());
        if let Some(v) = victim {
            self.sink_live(&v, eng, cache)?;
        }
    }
        // 2) 沉没池超上限：按 LRU 丢弃（数据损失，响亮日志）
        while self.sunk_bytes > self.sunk_cap {
            match self.sunk.iter().max_by_key(|(_, s)| s.at).map(|(k, _)| k.clone()) {
                Some(k) => {
                    if let Some(s) = self.sunk.remove(&k) {
                        self.sunk_bytes -= s.bytes;
                        eprintln!("[serve] ⚠ 沉没池超限：丢弃最旧会话 '{k}'（数据丢失）");
                    }
                }
                None => break,
            }
        }
        // 3) 目标在沉没层 → 唤醒
        if let Some(s) = self.sunk.remove(sid) {
            self.sunk_bytes = self.sunk_bytes.saturating_sub(s.bytes);
            let len = s.cp.ids.len();
            // pledge 至少盖住现有用量（+64 草稿余量），且不超池
            let pledge = need.max(len + 64).min(self.budget.total());
            self.budget.admit(sid_key(sid), pledge, MIN_CTX)
                .map_err(|_| anyhow::anyhow!("KV 预算不足（唤醒会话 {sid}）"))?;
            let mut seq = mk_seq()?;
            seq.restore_host_checkpoint_at(eng, &s.cp, len)?;
            seq.ids = s.cp.ids.clone();
            self.budget.tick(sid_key(sid), len, 0);
            eprintln!("[serve] 会话 '{sid}' 唤醒（{len} tok 从 host 灌回）");
            self.materialize(sid, seq);
            return Ok(());
        }
        // 4) 全新会话
        self.budget.admit(sid_key(sid), need, MIN_CTX)
            .map_err(|_| anyhow::anyhow!("KV 预算不足（新会话 {sid} 需 {need}）"))?;
        let seq = mk_seq()?;
        self.materialize(sid, seq);
        Ok(())
    }

    /// 把一个 live 会话沉到 host（多物化超配挤占用）。其图集随 epoch 回收。
    /// 设备 KV 工作区随之释放 ⇒ 该会话的 Dev 检查点全部作废（host/disk 档自包含，保留）。
    pub(crate) fn sink_live(
        &mut self,
        sid: &str,
        eng: &lingxi_driver::Engine<'_>,
        cache: &mut PrefillCache,
    ) -> Result<()> {
        let Some(l) = self.live.remove(sid) else { return Ok(()) };
        cache.invalidate_owner(sid);
        let cp = l.seq.save_host_checkpoint(eng, l.seq.ids.len())?;
        // host 字节粗算：KV bf16 ~4KB/token（双卡合并口径）+ GDN 固定态
        let bytes = cp.ids.len() * 4096 + 256 * 1024 * 1024;
        self.sunk_bytes += bytes;
        let n = cp.ids.len();
        self.sunk.insert(sid.to_string(), Sunk { cp, bytes, at: Instant::now() });
        self.budget.release(sid_key(sid));
        lingxi_model::graph::epoch_purge(l.epoch);
        if self.cur_sid.as_deref() == Some(sid) {
            self.cur_sid = None;
        }
        eprintln!("[serve] 会话 '{sid}' 沉没（{n} tok → host，图 epoch {} 已回收）", l.epoch);
        Ok(())
    }

    pub(crate) fn materialize(&mut self, sid: &str, seq: lingxi_driver::Sequence) {
        // 新物化 = 设备缓冲全新地址 ⇒ 领全进程唯一新 epoch（旧图绝不可重放）。
        // 注意：**不要**在这里 epoch_switch——Kernels/stream 是 thread_local，
        // 真正跑前向的引擎线程要用 `activate` 自己切。
        let epoch = lingxi_model::graph::epoch_fresh();
        self.cur_sid = Some(sid.to_string());
        self.live.insert(sid.to_string(), LiveSeq { seq, epoch, at: Instant::now() });
    }

    /// 退出时把全部 live 会话落盘（旧行为只落当前会话）。
    pub(crate) fn offload_all_live(
        &mut self,
        eng: &lingxi_driver::Engine<'_>,
        cache: &mut PrefillCache,
    ) {
        let sids: Vec<String> = self.live.keys().cloned().collect();
        for sid in sids {
            self.activate(&sid);
            cache.set_owner(&sid);
            if let Some(seq) = self.current_mut() {
                if let Err(e) = cache.offload_resident(eng, seq) {
                    eprintln!("[serve] 退出落盘失败（{sid}）: {e:#}");
                }
            }
        }
    }
}
