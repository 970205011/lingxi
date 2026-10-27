//! 设备 Q4 槽缓存：`n_slots` 个等大 blob 槽 + LRU（CODE_MAP Phase 2 刀P2-1
//! 从 arena_source.rs 搬入，只搬家不改行为）。

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::Result;
use cudarc::driver::CudaSlice;
use lingxi_kernels::Kernels;

/// 设备 Q4 槽缓存：`n_slots` 个等大 blob 槽 + LRU。
pub(crate) struct SlotCache {
    /// `n_slots * blob` 字节的 Q4 blob 区
    pub(crate) slots: CudaSlice<u8>,
    pub(crate) n_slots: usize,
    pub(crate) index: HashMap<(u32, u32), usize>,
    /// LRU 序：`(key, 版本, 槽号)`。**只含非钉住的槽**（钉住的在构造期一次填好后永不进队）。
    pub(crate) order: VecDeque<((u32, u32), u64, usize)>,
    pub(crate) ver: HashMap<(u32, u32), u64>,
    pub(crate) next_free: usize,
    /// **钉住的槽号**（`LINGXI_ARENA_PREWARM` 预热的那批）：不进 LRU 队 ⇒ `victim` 永不返回它们。
    /// 为什么要"钉"而不是"只预热"：实测工作集 ≈ 槽数（bench 负载 6774 个 (层,专家) / 7000 槽）
    /// ⇒ 纯 LRU 会**抖**（把很快要用的热专家换出去），实测命中率只有 93%。
    /// 钉住最热的一批、让 LRU 只管理尾巴，尾巴就能装下（4800 钉 + 2200 尾 ≥ 尾巴 distinct）。
    /// ★ P15-⑱：HashSet<usize> → `Vec<bool>`（按槽号 O(1) 判钉）。`touch`/`victim` 热路径
    /// 每次命中都要查一次，HashSet 哈希 ~20ns × 每层数十命中 ⇒ Vec 下标 ~2ns。
    pub(crate) pinned: Vec<bool>,
    /// **钉住槽的入钉顺序**（`victim` 兜底时用它取"最旧的钉槽"）。
    /// 为什么必须有：钉表 + 尾槽超过总槽数时，LRU 队会**被掏空**，而钉住的槽从不进队 ⇒
    /// `victim` 走到 `return 0` 那一步，把 0 号槽上钉住的专家**静默覆盖**（数学出错、
    /// 无人知晓）。2026-10-16 实测：`PREWARM_N=200 × 48 层 = 9600 > 9000 槽` ⇒ 9000 槽全钉
    /// ⇒ 主干首 token 判据当场不再命中（英文提示词的固定贪心序列直接变了）。
    pub(crate) pinned_q: VecDeque<usize>,
    /// **反向索引** `槽号 → key`。为什么要它（2026-10-16，性能坑）：`victim` 的兜底
    /// （"槽全被钉住"时解钉最旧的钉槽）需要"这个槽属于哪个 key"，而第一版写成
    /// `self.index.iter.find(|(_, &s)| s == sl)` —— **O(n_slots) 的 HashMap 扫描**（9000 项指针追逐
    /// ≈ 数百 µs）。而"钉表 ≈ 槽数"时**每个 miss 都会走兜底** ⇒ 那 0.29 ms/miss 的"拷贝代价"
    /// 其实是**这段扫描**（实测：换注册方式、换拷贝流，斜率一点没变）。这里加个 O(1) 的反查。
    pub(crate) rev: HashMap<usize, (u32, u32)>,
    pub hits: u64,
    pub misses: u64,
    /// **在线自适应重钉**（`LINGXI_ARENA_REPIN=<窗口>`）用的"最近热度"计数。
    /// 每窗口结束按它重排 top-N，并把计数**减半**（滑动窗口）⇒ 话题切了就跟着换。
    /// ⚠️ 只有 `LINGXI_ARENA_REPIN>0` 时才被填（否则恒空、完全不参与任何逻辑）。
    pub(crate) recent: HashMap<(u32, u32), u32>,
    /// 上次重钉时的 `hits+misses` 水位
    pub(crate) recent_at: u64,
    /// ㊹ B3：**扁平直查表** `(layer*E+e) → 槽号+1`（0=不在缓存）。热路径 `load_hits` 每 token
    /// 480 次 `HashMap<(u32,u32),usize>` 查找（~50-100ns 哈希+探测）⇒ 换 O(1) 下标（~2ns）。
    /// `index`/`ver`/`rev` 保留（victim/重钉簿记仍用）；变更点在 insert/victim/prewarm 同步维护。
    pub(crate) flat: Vec<u32>,
    pub(crate) n_experts: usize,
    /// ㊺ A2：哪些层的槽位表行自上次同步后变过（insert/victim 都会改 flat）。
    /// 被淘汰的 key 可能属于**别的层**——那一层也必须重传，否则设备表会指向已被覆盖的槽。
    pub(crate) dirty: Vec<bool>,
}

impl SlotCache {
    pub(crate) fn new(k: &Kernels, n_slots: usize, blob: usize, n_layers: usize, n_experts: usize) -> Result<Self> {
        // ★ 2026-10-25 **OOM 就逐级退让**：`autotune_arena_slots` 的预算是静态估算，
        // 槽位会把"余量"吃满（ledger 上「占用 = 余」）⇒ 这一次性大块 `cudaMalloc`
        // 偶发 `CUDA_ERROR_OUT_OF_MEMORY`（实测 200 token 跑里约 1/20 起不来）。
        // 失败的 `cudaMalloc` 不污染上下文，退 6.25% 重试即可；少几十个槽 < 1%，
        // 比"启动随机失败"划算得多。
        // ★★ 2026-10-25 **先等再砍**：连续两次跑挨得近时，**上一个进程的显存是异步释放的**
        // ⇒ 第一次分配可能撞到"还没还回来"。实测那次退让一路砍到 1357 槽（−87%），
        // 随后连 `GdnState` 的小分配都 OOM（整跑起不来）。所以每个尺寸先原地重试 3 次
        // （间隔 400 ms），三次都不成才真的缩。
        let want = n_slots;
        let mut n = n_slots;
        let mut fails = 0usize;
        let mut last_err: Option<anyhow::Error> = None;
        let slots = loop {
            let mut hit: Option<cudarc::driver::CudaSlice<u8>> = None;
            for attempt in 0..3 {
                match unsafe { k.stream.alloc::<u8>(blob * n.max(1)) } {
                    Ok(s) => {
                        hit = Some(s);
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e.into());
                        if attempt == 0 && fails == 0 {
                            eprintln!(
                                "[arena] ⚠️ 槽位分配 OOM（{n} 槽 × {blob} B）——先等 400 ms 重试同一尺寸\
                                 （多半是上一进程的显存还没释放完）"
                            );
                        }
                        std::thread::sleep(std::time::Duration::from_millis(if attempt < 2 { 400 } else { 0 }));
                    }
                }
            }
            if let Some(s) = hit {
                break s;
            }
            let next = n * 15 / 16;
            if n <= 1 || next == n || fails >= 96 {
                return Err(last_err.unwrap_or_else(|| anyhow::anyhow!("arena 槽位分配失败")));
            }
            // 只打第一次：真实场景（预算差几百个槽）退 1~2 次就成，刷屏没意义
            if fails == 0 {
                eprintln!(
                    "[arena] ⚠️ 三次重试仍 OOM：{n}/{want} 个 × {blob} B 起退（每步 −6.25%）"
                );
            }
            fails += 1;
            n = next;
        };
        Ok(Self {
            slots,
            n_slots: n,
            index: HashMap::new(),
            order: VecDeque::new(),
            ver: HashMap::new(),
            next_free: 0,
            pinned: vec![false; n_slots],
            pinned_q: VecDeque::new(),
            rev: HashMap::new(),
            hits: 0,
            misses: 0,
            recent: HashMap::new(),
            recent_at: 0,
            flat: vec![0u32; n_layers * n_experts],
            n_experts,
            dirty: vec![false; n_layers],
        })
    }

    #[inline]
    pub(crate) fn flat_get(&self, key: (u32, u32)) -> Option<usize> {
        let v = self.flat[key.0 as usize * self.n_experts + key.1 as usize];
        if v == 0 { None } else { Some((v - 1) as usize) }
    }
    #[inline]
    pub(crate) fn flat_set(&mut self, key: (u32, u32), sl: usize) {
        self.flat[key.0 as usize * self.n_experts + key.1 as usize] = (sl + 1) as u32;
        self.dirty[key.0 as usize] = true;
    }
    #[inline]
    pub(crate) fn flat_clr(&mut self, key: (u32, u32)) {
        self.flat[key.0 as usize * self.n_experts + key.1 as usize] = 0;
        self.dirty[key.0 as usize] = true;
    }

    pub(crate) fn touch(&mut self, key: (u32, u32), sl: usize) {
        // 钉住的槽不进 LRU 队：`victim` 只从队里取 ⇒ 永不淘汰它们
        if self.pinned[sl] {
            return;
        }
        let v = {
            let e = self.ver.entry(key).or_insert(0);
            *e += 1;
            *e
        };
        self.order.push_back((key, v, sl));
        if self.order.len() > self.index.len() * 4 + 64 {
            // ⚠️ `self.ver[kk]` 会**直接 panic**（`HashMap: Index` 用 `expect`）：`index` 里
            // 有**预热/重钉**塞进去的 key（它们只 `index.insert`、从不 `touch` ⇒ 不入 `ver`）。
            // 2026-10-16 实测踩到：`LINGXI_MOE_SPLIT` + 7000 槽 + `REPIN` 那条路上
            // `touch` 的压缩分支 `panic at arena_source.rs:88: option::expect_failed`。
            // 这里按"版本号缺失 = 0"处理（预热的槽从不进 `order`，所以 `retain` 只会把它们
            // 之外的陈旧项删掉 —— 语义与原来一致，但不再 panic）。
            let cur: HashSet<((u32, u32), u64, usize)> = self
                .index
                .iter()
                .map(|(kk, &s)| (*kk, self.ver.get(kk).copied().unwrap_or(0), s))
                .collect();
            self.order.retain(|ent| cur.contains(ent));
        }
    }

    pub(crate) fn insert(&mut self, key: (u32, u32), sl: usize) {
        self.index.insert(key, sl);
        self.rev.insert(sl, key);
        self.flat_set(key, sl);
        self.touch(key, sl);
    }

    /// 取一个可写槽：先用未用过的，再用 LRU 最旧的。
    ///
    /// `prot`（**保护集**）：本批 `load_hits` **已经分配出去**的槽 —— 必须跳过。
    /// 为什么（2026-10-16 抓到的真 bug）：原来这里是"LRU 队里随便取一个"。而 `load_hits`
    /// 是**边取槽边 insert** 的 ⇒ 同一次调用里先为 e1 拿到槽 S、`insert(e1,S)` 把 S 推到 LRU 队尾；
    /// 接着为 e2 取槽时，若队里只剩 S（尾槽耗尽/全钉住时必然如此），就会把 **S 再分配给 e2**
    /// ⇒ S 里装的是 e2 的权重，而 `slot_ids` 里 e1 仍指着 S ⇒ **两个专家共用一块权重**：
    /// 数学静默出错（实测：`PREWARM_N=200×48 > 9000 槽` 时首 token 从 11751 变成 303）。
    /// `None` = 实在没有可淘汰的槽（调用方必须报错，不能再像以前那样 `return 0` 硬来）。
    pub(crate) fn victim(&mut self, prot: &[usize]) -> Option<usize> {
        if self.next_free < self.n_slots {
            let s = self.next_free;
            self.next_free += 1;
            return Some(s);
        }
        // ① 先找**未被钉、且不在本批**的最旧项（跳过的按原顺序放回队头，保持 LRU 序）
        // ★ P15-⑱：`prot` 从 HashSet 改 `&[usize]` 线性扫（本批对数 ≤ 64，哈希反而慢）。
        let mut skipped: Vec<((u32, u32), u64, usize)> = Vec::new();
        let mut picked = None;
        while let Some(ent) = self.order.pop_front() {
            let (key, v, sl) = ent;
            if self.pinned[sl] || prot.contains(&sl) {
                skipped.push(ent);
                continue;
            }
            if self.ver.get(&key) == Some(&v) && self.index.get(&key) == Some(&sl) {
                self.index.remove(&key);
                self.ver.remove(&key);
                self.rev.remove(&sl);
                self.flat_clr(key);
                picked = Some(sl);
                break;
            }
        }
        for e in skipped.into_iter().rev() {
            self.order.push_front(e);
        }
        if picked.is_some() {
            return picked;
        }
        // ② 兜底：队空了或只剩钉住/本批的 ⇒ **解钉最旧的**那个（钉 = 优先不淘汰，不是永不淘汰）。
        // 以前这里直接 `return 0`，会**静默覆盖 0 号槽上钉住的专家**。
        let mut keep: Vec<usize> = Vec::new();
        let mut got = None;
        while let Some(sl) = self.pinned_q.pop_front() {
            if prot.contains(&sl) {
                keep.push(sl);
                continue;
            }
            if !self.pinned[sl] {
                continue; // 已经解钉过的陈旧项
            }
            self.pinned[sl] = false;
            // O(1) 反查（第一版是 `index.iter.find(...)` = O(n_slots) ⇒ 每个 miss 数百 µs）
            if let Some(key) = self.rev.remove(&sl) {
                self.index.remove(&key);
                self.ver.remove(&key);
                self.flat_clr(key);
            }
            got = Some(sl);
            break;
        }
        for sl in keep.into_iter().rev() {
            self.pinned_q.push_front(sl);
        }
        if got.is_some() {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[arena] ⚠️ 槽位全被钉住（钉表 ≥ 槽数）⇒ 被迫淘汰**钉住的**槽。\
                     数学仍然正确（不再覆盖 0 号槽、也不会把本批在用的槽分给别人），\
                     但说明 `LINGXI_ARENA_PREWARM_N` 过大：钉表应当留出足够的 LRU 尾巴。"
                );
            }
        }
        got
    }
}
