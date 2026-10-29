//! R4 拆分（2026-10-21）：原 serve.rs 3152 行按职责分段，本模块为其中一段。
//! 模块总览见 `serve/mod.rs`。段间共享类型一律 `pub(crate)`。

/// Radix 前缀树块长（token 数）。纯 host 元数据结构，**与 KV 页大小无关**：
/// 粒度只影响节点数与分裂频率。16 tok ⇒ 119K token 文档最多 ~7.4K 节点，
/// 每节点约百字节；太大（如整段 4K）退化成旧的"整段线性比较"，
/// 太小节点/map 开销放大。对标 SGLang RadixAttention 但按 token 切块。
const RADIX_BLK: usize = 16;

/// **Radix 前缀树**（阶段 1，2026-10-28）：条目的查找索引，对标 SGLang
/// RadixAttention。**只负责"查找加速 + 跨会话前缀段元数据去重"**——
/// 共享前缀（同一系统提示/同一份文档）在树里只建一份路径；
/// 每个会话各自的物理检查点（Dev/Disk）仍挂在同一路径的节点上，
/// 命中后各 restore 各的，Dev/Disk 语义与 owner 有效性规则一字不改。
///
/// 节点存活判定 = `attached` 非空或有子节点（引用计数的派生形式）：
/// 条目删除时沿路径摘除其引用，空节点自底向上回收（arena 槽位进空闲表复用）。
/// 另有 key_map 把树里的条目 key 解析回 `entries` 下标；调用方绕过
/// `PrefillCache` 直接删条目（如 engine 的 drop_stale_dev）时引用会变
/// "陈旧"，`best()` 遇到 key 对不上就惰性摘除并回收，保证正确性。
struct RadixTree {
    /// arena；`nodes[0]` 是根（空块，start=0）。
    nodes: Vec<RadixNode>,
    /// 回收的槽位，alloc 时复用（复用前字段全清，无悬空引用）。
    free: Vec<usize>,
    /// 条目 key → `PrefillCache::entries` 下标。
    key_map: std::collections::HashMap<u64, usize>,
}

struct RadixNode {
    /// token 块（≤ `RADIX_BLK` 个；根为空）。子节点索引以块首 token 为键。
    block: Vec<i64>,
    /// 根到本节点块起点的路径长（token 数）。
    start: usize,
    parent: usize,
    children: std::collections::HashMap<i64, usize>,
    /// 挂在本节点的检查点：`(条目长度, 条目 key)`。条目长度落在
    /// `(start, start+block.len()]` 内——主长度挂在末尾节点，
    /// 较短前缀档（alt_pairs）可能挂在中途的块节点上。
    attached: Vec<(usize, u64)>,
}

impl RadixNode {
    fn empty() -> Self {
        Self {
            block: Vec::new(),
            start: 0,
            parent: 0,
            children: std::collections::HashMap::new(),
            attached: Vec::new(),
        }
    }
}

impl RadixTree {
    fn new() -> Self {
        Self {
            nodes: vec![RadixNode::empty()],
            free: Vec::new(),
            key_map: std::collections::HashMap::new(),
        }
    }

    fn alloc(&mut self) -> usize {
        if let Some(id) = self.free.pop() {
            self.nodes[id] = RadixNode::empty();
            id
        } else {
            self.nodes.push(RadixNode::empty());
            self.nodes.len() - 1
        }
    }

    /// 沿 `ids` 下行，返回覆盖 `ids` 末尾的节点；沿途缺的块新建、
    /// 分叉的块分裂。`ids` 在已有块中间结束时不分裂（挂在中途块节点上）。
    fn insert_path(&mut self, ids: &[i64]) -> usize {
        let mut node = 0usize;
        let mut pos = 0usize;
        while pos < ids.len() {
            let tok = ids[pos];
            let Some(&c) = self.nodes[node].children.get(&tok) else {
                let end = (pos + RADIX_BLK).min(ids.len());
                let id = self.alloc();
                self.nodes[id].block = ids[pos..end].to_vec();
                self.nodes[id].start = pos;
                self.nodes[id].parent = node;
                self.nodes[node].children.insert(tok, id);
                node = id;
                pos = end;
                continue;
            };
            let cstart = self.nodes[c].start;
            let blk_len = self.nodes[c].block.len();
            let mut k = 0usize;
            while k < blk_len && pos + k < ids.len() && self.nodes[c].block[k] == ids[pos + k] {
                k += 1;
            }
            if k == blk_len || pos + k == ids.len() {
                // 整块匹配，或插入前缀在 c 的块中间结束：直接落 c
                node = c;
                pos += k;
                continue;
            }
            // 块内分叉：前段 [..k] 给新节点 split，后段 [k..] 留在 c
            debug_assert!(k > 0); // 首 token 命中 map 键，公共前缀至少 1
            let split = self.alloc();
            let front: Vec<i64> = self.nodes[c].block[..k].to_vec();
            let back: Vec<i64> = self.nodes[c].block[k..].to_vec();
            self.nodes[split].block = front;
            self.nodes[split].start = cstart;
            self.nodes[split].parent = node;
            // attached 按长度切分：len ≤ cstart+k 属于前段 split，其余留在 c
            let mut front_att = Vec::new();
            let mut back_att = Vec::new();
            for a in std::mem::take(&mut self.nodes[c].attached) {
                if a.0 <= cstart + k {
                    front_att.push(a);
                } else {
                    back_att.push(a);
                }
            }
            self.nodes[split].attached = front_att;
            self.nodes[c].attached = back_att;
            self.nodes[c].block = back;
            self.nodes[c].start = cstart + k;
            self.nodes[c].parent = split;
            let back_tok = self.nodes[c].block[0];
            self.nodes[split].children.insert(back_tok, c);
            self.nodes[node].children.insert(tok, split);
            node = split;
            pos += k;
        }
        node
    }

    /// 摘除一份条目的全部引用：沿 `ids` 路径删掉所有 `(len, key)`，
    /// 再自底向上回收空节点。`ids` 必须与插入时一致（条目的主长度路径）。
    fn remove_path(&mut self, ids: &[i64], key: u64) {
        self.key_map.remove(&key);
        let mut path = vec![0usize];
        let mut node = 0usize;
        let mut pos = 0usize;
        while pos < ids.len() {
            let Some(&c) = self.nodes[node].children.get(&ids[pos]) else { break };
            let blk_len = self.nodes[c].block.len();
            let mut k = 0usize;
            while k < blk_len && pos + k < ids.len() && self.nodes[c].block[k] == ids[pos + k] {
                k += 1;
            }
            node = c;
            pos += k;
            path.push(node);
            if k < blk_len {
                break;
            }
        }
        for &n in &path {
            self.nodes[n].attached.retain(|&(_, k)| k != key);
        }
        for &n in path.iter().skip(1).rev() {
            self.prune_up(n);
        }
    }

    /// 摘掉某节点上给定下标的 attached 引用，再自底向上回收空节点。
    fn detach(&mut self, node: usize, attached_idx: &[usize]) {
        {
            let attached = &mut self.nodes[node].attached;
            for &ai in attached_idx.iter().rev() {
                attached.remove(ai);
            }
        }
        self.prune_up(node);
    }

    /// 节点既无 attached 也无子节点时从父摘除并进空闲表（其子孙必然已空：
    /// 任何存活条目都会把路径上的节点保持在"非空"状态）。
    fn prune_up(&mut self, mut n: usize) {
        // block 为空 = 已回收的槽位（避免对同一路径重复向上回收时二次访问）
        while n != 0
            && !self.nodes[n].block.is_empty()
            && self.nodes[n].attached.is_empty()
            && self.nodes[n].children.is_empty()
        {
            let p = self.nodes[n].parent;
            let tok = self.nodes[n].block[0];
            self.nodes[p].children.remove(&tok);
            self.nodes[n] = RadixNode::empty();
            self.free.push(n);
            n = p;
        }
    }
}

/// **跨请求智能前缀缓存**（C3，2026-10-18）：多轮对话里下一轮的 prompt 通常以
/// “上一轮 prompt + 回复 + 新消息”开头 ⇒ 已进 KV 的那段不用重算。
///
/// 与 C2（只认完全匹配）的区别：
/// - 缓存里放 **1~2 个状态检查点**（`StateCheckpoint`：GDN/PLE 真拷贝 + QSA 长度）；
/// - 请求进来时找**最长匹配**的检查点（不要求最后一个字也对上）；
/// - 另外在“最后一条消息之前”存一个**消息边界检查点**：用户只改最后一句时，
///   前面的历史直接复用，只重读新消息。
///
/// 默认**开**（`LINGXI_SERVE_PREFIX_CACHE=0` 可关）。
///
/// 阶段 1（2026-10-28，本分支）：条目索引改为 Radix 前缀树（`RADIX_BLK`=16 token
/// 块，纯 host 元数据）。树只做"查找加速 + 跨会话前缀段去重索引"：Dev/Disk 存储
/// 语义、owner 有效性规则（ed7ef0a）原样保留；物理 KV 共享（COW 页池）是阶段 2，
/// 见 docs/radix-prefix-tree-design.md。
pub(crate) enum CkptStore {
    /// 会话还在显存里：QSA KV 就地有效，只存了 GDN/PLE 的快照。
    Dev(lingxi_driver::StateCheckpoint),
    /// 会话已落盘（/dev/shm 或真实磁盘）：进程退出后还在，下次命中再读回显存。
    Disk(std::path::PathBuf),
}

pub(crate) struct PrefillCache {
    pub(crate) entries: Vec<CkptEntry>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) clock: u64,
    /// 落盘缓存已用字节
    pub(crate) cache_bytes: usize,
    /// 落盘缓存上限（`LINGXI_SERVE_CACHE_GIB`，默认 24 GiB；0 = 关）
    pub(crate) cache_limit: usize,
    /// 只缓存“够长”的会话：短上下文重算更便宜，而 GDN/PLE 状态固定 ~113MB/份，
    /// 存 18 个 token 也要 226MB（`LINGXI_SERVE_CACHE_MIN_TOKENS`，默认 8192）
    pub(crate) cache_min_tokens: usize,
    /// 落盘目录（`LINGXI_SERVE_CACHE_DIR`，默认 `/tmp/lingxi-prefix-cache`；空 = 关）
    pub(crate) cache_dir: Option<std::path::PathBuf>,
    /// 当前**设备 KV** 里对应的 token（最近一次成功请求结束时的 `seq.ids`）。
    /// 只有它是某个条目的前缀时，该 Dev 条目才仍然有效（否则设备 KV 已被别的会话覆盖）。
    pub(crate) device_ids: Vec<i64>,
    /// 当前会话（v1.5 多物化：每个 live 会话有私有 KV 工作区，切换不覆盖别家数据）。
    /// Dev 检查点按 owner 判有效性：只有 owner 仍是当前会话才可恢复。
    pub(crate) cur_owner: Option<String>,
    /// Radix 前缀树：条目的查找索引（跨会话共享前缀段只建一份路径）。
    tree: RadixTree,
    /// 条目 key 发号器（1 起；树 key_map 用）。0 = 未入树的哨兵。
    next_key: u64,
}

impl Default for PrefillCache {
    fn default() -> Self {
        let dir = std::env::var("LINGXI_SERVE_CACHE_DIR")
            .unwrap_or_else(|_| "/tmp/lingxi-prefix-cache".into());
        let gib = std::env::var("LINGXI_SERVE_CACHE_GIB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(24);
        let min_tokens = std::env::var("LINGXI_SERVE_CACHE_MIN_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8192);
        let mut cache = Self {
            entries: Vec::new(),
            hits: 0,
            misses: 0,
            clock: 0,
            cache_bytes: 0,
            cache_limit: gib * 1024 * 1024 * 1024,
            cache_min_tokens: min_tokens,
            cache_dir: if dir.is_empty() || gib == 0 { None } else { Some(dir.into()) },
            device_ids: Vec::new(),
            cur_owner: None,
            tree: RadixTree::new(),
            next_key: 1,
        };
        if let Some(d) = cache.cache_dir.clone() {
            let _ = std::fs::create_dir_all(&d);
            cache.load_disk_cache(&d);
        }
        cache
    }
}

pub(crate) struct CkptEntry {
    pub(crate) kind: CkptKind,
    /// 已喂进 KV 的 token（= 检查点长度）
    pub(crate) ids: Vec<i64>,
    /// 这个前缀的下一个 token（检查点保存时已经算出来了；**完全命中原 prompt** 时直接用它，
    /// 再把它 push 进 ids 当 pending，后续 decode 的约定就与正常 prefill 完全一致）。
    pub(crate) next: i64,
    pub(crate) store: CkptStore,
    /// Dev 条目的属主会话：KV 工作区是它的，只有它当值时才可恢复（v1.5 多物化）。
    /// Disk 条目无属主（数据自包含），恒有效。
    pub(crate) owner: Option<String>,
    /// 落盘占用（Dev 条目为 0）
    pub(crate) bytes: usize,
    /// LRU 时钟
    pub(crate) used: u64,
    /// 同一会话的较短前缀档：`(长度, next)`（KV 共享，只有 GDN/PLE 状态不同）
    pub(crate) alt_pairs: Vec<(usize, i64)>,
    /// 树索引用的唯一 key（`next_key` 发号）。entry 从 `entries` 里被删、
    /// 但树引用尚未摘除时会变"陈旧"，`best()` 按 key 对不上惰性清理。
    pub(crate) key: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]

pub(crate) enum CkptKind {
    /// 最后一条消息之前（用户改最后一句时命中）
    Boundary,
    /// 本轮 prompt 刚喂完、还没生成（下一轮原样重发时命中）
    Prompt,
    /// 本轮生成结束（最长的历史，下一轮追加新消息时命中）
    End,
}

/// 检查点级别：`LINGXI_SERVE_CKPTS`（默认 3）
/// 3 = 边界+prompt+末尾（≈345 MB）；2 = prompt+末尾；1 = 只 prompt；0 = 不存（只复用已有的）。
pub(crate) fn ckpt_level() -> usize {
    std::env::var("LINGXI_SERVE_CKPTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
        .min(3)
}

impl PrefillCache {
    pub(crate) fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// 记录当前会话（ensure/activate 时调用）。Dev 检查点只有 owner == 当前会话
    /// 才可恢复——v1.5 多物化下每个 live 会话是私有 KV 工作区，切换不覆盖别家数据，
    /// 所以条目按 owner 存活，不再一刀切作废。
    pub(crate) fn set_owner(&mut self, sid: &str) {
        self.cur_owner = Some(sid.to_string());
    }

    /// 这个 Dev 条目还跟当前设备 KV 对得上吗？（别的会话 prefill 过就作废）
    pub(crate) fn dev_valid(&self, e: &CkptEntry) -> bool {
        match &e.store {
            // v1.5 私有工作区：owner 还当着、条目非空即有效（owner 自己重 prefill
            // 时 run_prefill 会调 invalidate_owner 清掉旧条目）。
            CkptStore::Dev(_) => {
                !e.ids.is_empty()
                    && e.owner.is_some()
                    && e.owner.as_deref() == self.cur_owner.as_deref()
            }
            CkptStore::Disk(_) => !e.ids.is_empty(),
        }
    }

    pub(crate) fn next_at(&self, e: &CkptEntry, len: usize) -> i64 {
        if len == e.ids.len() {
            e.next
        } else {
            e.alt_pairs
                .iter()
                .find(|(l, _)| *l == len)
                .map(|(_, n)| *n)
                .unwrap_or(e.next)
        }
    }

    /// 把 `entries[idx]` 挂进树：主长度挂末尾节点，`alt_pairs` 的短前缀档
    /// 挂各自长度所在节点。空 ids 不入树。
    fn index_insert(&mut self, idx: usize) {
        let (key, idlen, alt_lens) = {
            let e = &self.entries[idx];
            if e.ids.is_empty() {
                return;
            }
            (
                e.key,
                e.ids.len(),
                e.alt_pairs.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
            )
        };
        self.tree.key_map.insert(key, idx);
        let node = self.tree.insert_path(&self.entries[idx].ids);
        self.tree.nodes[node].attached.push((idlen, key));
        for l in alt_lens {
            if l > 0 && l <= idlen {
                let n = self.tree.insert_path(&self.entries[idx].ids[..l]);
                self.tree.nodes[n].attached.push((l, key));
            }
        }
    }

    /// `entries.retain` 的同步维护版：被删条目先从树里摘除（引用回收），
    /// 再动 `entries`。cache.rs 内所有批量删除一律走这里。
    fn retain_entries<F: FnMut(&CkptEntry) -> bool>(&mut self, mut pred: F) {
        let dead: Vec<(u64, Vec<i64>)> = self
            .entries
            .iter()
            .filter(|e| !pred(e))
            .map(|e| (e.key, e.ids.clone()))
            .collect();
        self.entries.retain(pred);
        for (key, ids) in dead {
            self.tree.remove_path(&ids, key);
        }
    }

    /// 找“最长且完整匹配”的检查点，返回 `(下标, 可复用长度)`。
    ///
    /// Radix 树下行最长匹配：从根按 token 块逐块比对，只访问与查询共享前缀
    /// 的路径节点——O(块深) 而不是旧的 O(条目数 × 前缀比较)；共享前缀的
    /// 多份条目挂在同一节点，各自解析、各判有效性。
    ///
    /// Dev 条目必须仍然是当前设备 KV 的前缀；Disk 条目永远有效（数据在盘上）。
    /// Disk 条目除了主长度外，还可能带若干**较短前缀档**（KV 共享，只多存 GDN/PLE）。
    pub(crate) fn best(&mut self, ids: &[i64]) -> Option<(usize, usize)> {
        let dbg = std::env::var("LINGXI_SERVE_CACHE_DEBUG").map(|v| v != "0").unwrap_or(false);
        if ids.is_empty() {
            return None;
        }
        let mut best: Option<(usize, usize)> = None;
        let mut node = 0usize;
        let mut matched = 0usize;
        loop {
            // 1) 本节点 token 块与查询的公共前缀
            let start = matched;
            let blk_len = self.tree.nodes[node].block.len();
            let mut k = 0usize;
            while k < blk_len && start + k < ids.len() && self.tree.nodes[node].block[k] == ids[start + k]
            {
                k += 1;
            }
            matched = start + k;
            // 2) 挂在该节点的检查点候选：条目长度须落在已匹配范围内
            let mut stale: Vec<usize> = Vec::new();
            let mut dead_keys: Vec<u64> = Vec::new();
            for (ai, &(l, key)) in self.tree.nodes[node].attached.iter().enumerate() {
                if l > matched {
                    continue;
                }
                let idx = match self.tree.key_map.get(&key) {
                    Some(&idx) if self.entries.get(idx).map_or(false, |e| e.key == key) => idx,
                    _ => {
                        // 调用方绕过 PrefillCache 直接删过条目（engine 的
                        // drop_stale_dev / 恢复失败摘除）：下标错位或条目已死。
                        // 先按 key 找回真身并修正映射（错位是永久的，修一次就好）；
                        // 找不到才是陈旧引用，惰性摘除并回收路径。
                        match self.entries.iter().position(|e| e.key == key) {
                            Some(real) => {
                                self.tree.key_map.insert(key, real);
                                real
                            }
                            None => {
                                stale.push(ai);
                                dead_keys.push(key);
                                if dbg {
                                    eprintln!("[cache] 树引用陈旧 key={key} len={l} 摘除回收");
                                }
                                continue;
                            }
                        }
                    }
                };
                let e = &self.entries[idx];
                let kname = match e.kind {
                    CkptKind::Boundary => "边界",
                    CkptKind::Prompt => "prompt",
                    CkptKind::End => "末尾",
                };
                let sname = if matches!(e.store, CkptStore::Dev(_)) { "dev" } else { "disk" };
                if matches!(e.store, CkptStore::Dev(_)) && !self.dev_valid(e) {
                    if dbg {
                        eprintln!(
                            "[cache] #{idx} {kname}/{sname} len={l} owner={:?} 跳过(dev无效 cur={:?})",
                            e.owner, self.cur_owner
                        );
                    }
                    continue;
                }
                if dbg {
                    eprintln!("[cache] #{idx} {kname}/{sname} len={l} owner={:?} 命中", e.owner);
                }
                if best.map_or(true, |(_, bl)| l > bl) {
                    best = Some((idx, l));
                }
            }
            if !stale.is_empty() {
                for k in dead_keys {
                    self.tree.key_map.remove(&k);
                }
                self.tree.detach(node, &stale);
            }
            // 3) 整块匹配且查询还有剩余才继续下行；否则最长匹配到此为止
            if k < blk_len || matched >= ids.len() {
                break;
            }
            match self.tree.nodes[node].children.get(&ids[matched]) {
                Some(&c) => node = c,
                None => break,
            }
        }
        if dbg {
            eprintln!(
                "[cache] best={best:?}（共 {} 份，树 {} 节点/{} 空闲槽）",
                self.entries.len(),
                self.tree.nodes.len(),
                self.tree.free.len()
            );
        }
        best
    }

    /// 某个会话的 Dev 检查点全部作废（其 KV 工作区已不存在：沉没/重生）。
    /// 其余会话的 Dev 条目不受影响——v1.5 私有工作区互不覆盖。
    pub(crate) fn invalidate_owner(&mut self, sid: &str) {
        self.retain_entries(|e| {
            !matches!(e.store, CkptStore::Dev(_)) || e.owner.as_deref() != Some(sid)
        });
        if self.cur_owner.as_deref() == Some(sid) {
            self.device_ids.clear();
        }
    }

    /// 插入/替换**当前会话**的 Dev 检查点（每种只留一份）。
    pub(crate) fn push_dev(
        &mut self,
        kind: CkptKind,
        ids: Vec<i64>,
        next: i64,
        cp: lingxi_driver::StateCheckpoint,
        owner: Option<String>,
    ) {
        // 每种（kind）每会话只留一份：只替换本 owner 的旧档，别家不动。
        self.retain_entries(|e| {
            !(e.kind == kind
                && matches!(e.store, CkptStore::Dev(_))
                && e.owner == owner)
        });
        let used = self.tick();
        let key = self.next_key;
        self.next_key += 1;
        self.entries.push(CkptEntry {
            kind,
            ids,
            next,
            store: CkptStore::Dev(cp),
            owner,
            bytes: 0,
            used,
            alt_pairs: Vec::new(),
            key,
        });
        let idx = self.entries.len() - 1;
        self.index_insert(idx);
    }

    /// 把一份完整检查点写到盘上，并换成 Disk 条目（内存里的副本随即释放）。
    ///
    /// 盘目录默认 `/tmp/lingxi-prefix-cache`（tmpfs，进程退出后还在）；写真实磁盘
    /// 也能用 `LINGXI_SERVE_CACHE_DIR` 指过去。超过 `LINGXI_SERVE_CACHE_GIB` 按 LRU 删。
    pub(crate) fn push_disk(
        &mut self,
        kind: CkptKind,
        ids: Vec<i64>,
        next: i64,
        cp: lingxi_driver::HostCheckpoint,
    ) {
        let Some(dir) = self.cache_dir.clone() else { return };
        let bytes = cp.bytes;
        if bytes > self.cache_limit {
            eprintln!(
                "[serve] 单份缓存 {:.1} GB 超过上限 {:.1} GB，丢弃",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                self.cache_limit as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            return;
        }
        let alt_pairs: Vec<(usize, i64)> = cp
            .alt_lens()
            .into_iter()
            .map(|l| (l, cp.alt_next(l).unwrap_or(next)))
            .collect();
        let used = self.tick();
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        // 文件名带专用前缀：扫描/清理只认这个模式，绝不碰目录里的其他文件
        let path = dir.join(format!("lingxi-ckpt-{ms:013}-{used:06}.lxc"));
        match cp.write_to(&path, next) {
            Ok(file_bytes) => {
                drop(cp);
                // 同 ids 的旧文件/条目删掉，避免重复
                let mut replaced = 0usize;
                self.retain_entries(|e| {
                    let same = e.ids == ids && matches!(e.store, CkptStore::Disk(_));
                    if same {
                        replaced += e.bytes;
                        if let CkptStore::Disk(p) = &e.store {
                            let _ = std::fs::remove_file(p);
                        }
                    }
                    !same
                });
                self.cache_bytes = self.cache_bytes + file_bytes as usize - replaced.min(self.cache_bytes);
                let key = self.next_key;
                self.next_key += 1;
                self.entries.push(CkptEntry {
                    kind,
                    ids,
                    next,
                    store: CkptStore::Disk(path),
                    owner: None,
                    bytes: file_bytes as usize,
                    used,
                    alt_pairs,
                    key,
                });
                let idx = self.entries.len() - 1;
                self.index_insert(idx);
                self.evict_disk();
                eprintln!(
                    "[serve] 缓存落盘：{:.1} MB（{:.1}/{:.1} GB，{} 份）",
                    file_bytes as f64 / (1024.0 * 1024.0),
                    self.cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    self.cache_limit as f64 / (1024.0 * 1024.0 * 1024.0),
                    self.entries.iter().filter(|e| matches!(e.store, CkptStore::Disk(_))).count()
                );
            }
            Err(e) => eprintln!("[serve] 缓存落盘失败: {e:#}"),
        }
    }

    pub(crate) fn evict_disk(&mut self) {
        while self.cache_bytes > self.cache_limit {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| matches!(e.store, CkptStore::Disk(_)))
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    let (key, ids) = {
                        let e = &self.entries[i];
                        (e.key, e.ids.clone())
                    };
                    self.tree.remove_path(&ids, key);
                    let e = self.entries.remove(i);
                    if let CkptStore::Disk(p) = &e.store {
                        let _ = std::fs::remove_file(p);
                    }
                    self.cache_bytes = self.cache_bytes.saturating_sub(e.bytes);
                }
                None => break,
            }
        }
    }

    /// 启动时扫描落盘目录：只读每个文件的头（ids / next / 短前缀档），不读 payload。
    pub(crate) fn load_disk_cache(&mut self, dir: &std::path::Path) {
        use std::collections::HashMap;
        let mut keep: HashMap<Vec<i64>, (std::path::PathBuf, i64, Vec<(usize, i64)>, u64, u64)> =
            HashMap::new();
        let mut files: Vec<(u64, std::path::PathBuf)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for ent in rd.flatten() {
                let p = ent.path();
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                // 只碰我们自己命名的文件；目录里的其他东西一律忽略
                if !is_our_cache_file(name) {
                    continue;
                }
                if name.ends_with(".lxc.tmp") {
                    let _ = std::fs::remove_file(&p);
                    continue;
                }
                let mt = ent
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                files.push((mt, p));
            }
        }
        files.sort_by_key(|(t, _)| *t);
        for (mt, p) in files {
            match lingxi_driver::HostCheckpoint::read_header(&p) {
                Ok((ids, next, alt_pairs, bytes)) => {
                    // 策略变严后留下的短条目：直接清掉
                    if ids.len() < self.cache_min_tokens {
                        let _ = std::fs::remove_file(&p);
                        continue;
                    }
                    // 同一份 ids 只留最新的
                    if let Some(old) = keep.insert(ids, (p, next, alt_pairs, bytes, mt)) {
                        let _ = std::fs::remove_file(old.0);
                    }
                }
                Err(_) => {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
        let mut items: Vec<_> = keep
            .into_iter()
            .map(|(ids, (p, next, alts, bytes, mt))| (ids, p, next, alts, bytes, mt))
            .collect();
        items.sort_by_key(|(_, _, _, _, _, mt)| *mt);
        let mut total = 0usize;
        for (ids, path, next, alt_pairs, bytes, _) in items {
            total += bytes as usize;
            let used = self.tick();
            let key = self.next_key;
            self.next_key += 1;
            self.entries.push(CkptEntry {
                kind: CkptKind::End,
                ids,
                next,
                store: CkptStore::Disk(path),
                owner: None,
                bytes: bytes as usize,
                used,
                alt_pairs,
                key,
            });
            let idx = self.entries.len() - 1;
            self.index_insert(idx);
        }
        self.cache_bytes = total;
        self.evict_disk();
        eprintln!(
            "[serve] 前缀缓存: 目录 {}，恢复 {} 份 / {:.1} GB（上限 {:.1} GB）",
            dir.display(),
            self.entries.iter().filter(|e| matches!(e.store, CkptStore::Disk(_))).count(),
            self.cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.cache_limit as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    /// **把设备上当前会话整段换出到盘上**（新请求没命中时调用）。
    ///
    /// 设备 KV 马上会被新 prompt 覆盖，所以必须先拷走；拷完就把 Dev 副本丢掉，
    /// 不再占显存。另一个会话下次命中时再从盘上拉回来。
    pub(crate) fn offload_resident(
        &mut self,
        eng: &lingxi_driver::Engine<'_>,
        seq: &lingxi_driver::Sequence,
    ) -> anyhow::Result<()> {
        if self.cache_dir.is_none() {
            self.drop_owner_dev();
            return Ok(());
        }
        let pos = self
            .entries
            .iter()
            .position(|e| e.kind == CkptKind::End && matches!(e.store, CkptStore::Dev(_)));
        let Some(pos) = pos else { return Ok(()) };
        if !self.dev_valid(&self.entries[pos]) {
            // 设备 KV 已经不是它的了 ⇒ 这份 Dev 快照作废（只清当前属主，别家不动）
            self.drop_owner_dev();
            return Ok(());
        }
        let ids = self.entries[pos].ids.clone();
        let next = self.entries[pos].next;
        // 短上下文：GDN/PLE 固定 ~113MB/份，存 18 tok 也要 226MB，不如重算
        if ids.len() < self.cache_min_tokens {
            self.drop_owner_dev();
            eprintln!(
                "[serve] 会话 {} tok < 落盘阈值 {}，不缓存（重算更便宜）",
                ids.len(),
                self.cache_min_tokens
            );
            return Ok(());
        }
        // 同一会话的“消息边界 / prompt 刚喂完”两档只迁 GDN/PLE 状态，KV 共享最长那份
        let mut alts = Vec::new();
        for e in &self.entries {
            if e.kind == CkptKind::End
                || !matches!(e.store, CkptStore::Dev(_))
                || e.ids.len() >= ids.len()
                || ids[..e.ids.len()] != e.ids[..]
            {
                continue;
            }
            if let CkptStore::Dev(cp) = &e.store {
                alts.push(cp.to_host_alt(eng, e.next)?);
            }
        }
        let mut cp = seq.save_host_checkpoint(eng, ids.len())?;
        for a in alts {
            cp.push_alt(a);
        }
        // 只清当前属主的 Dev 档（其 KV 工作区马上被本请求重灌覆盖）；
        // 别家会话的 Dev 档指向各自的私有工作区，仍然有效（v1.5 多物化）。
        self.drop_owner_dev();
        self.push_disk(CkptKind::End, ids, next, cp);
        Ok(())
    }

    /// 删掉当前属主（cur_owner）的全部 Dev 条目；Disk 档与其余属主不动。
    fn drop_owner_dev(&mut self) {
        let owner = self.cur_owner.clone();
        self.retain_entries(|e| !matches!(e.store, CkptStore::Dev(_)) || e.owner != owner);
    }
}

/// 检查点保存统一入口。**QSA 分页模式下 dev 检查点语义不成立**（`reset` 的
/// `rewind(0)` 已把块表/槽/host 坑全部归还，而 dev 检查点只存 GDN/PLE，QSA 靠
/// "KV 还在设备上" 的隐含假设）——2026-10-20 实测：serve+paged 下第二次请求
/// （命中 dev 检查点恢复）输出退化为乱码。故 paged 时强制走
/// `save_host_checkpoint` → Disk 档（`paged_kv_to_host` 按块表收拢全量 KV）。
pub(crate) fn save_ckpt_auto(
    seq: &lingxi_driver::Sequence,
    eng: &lingxi_driver::Engine,
    len: usize,
    kind: CkptKind,
    next: i64,
    cache: &mut PrefillCache,
    what: &str,
    owner: &str,
) {
    if lingxi_model::flag("LINGXI_QSA_PAGED") {
        match seq.save_host_checkpoint(eng, len) {
            Ok(cp) => cache.push_disk(kind, seq.ids[..len].to_vec(), next, cp),
            Err(e) => eprintln!("[serve] {what} host 检查点保存失败: {e:#}"),
        }
    } else {
        match seq.save_checkpoint(eng, len) {
            Ok(cp) => cache.push_dev(
                kind,
                seq.ids[..len].to_vec(),
                next,
                cp,
                Some(owner.to_string()),
            ),
            Err(e) => eprintln!("[serve] {what} 检查点保存失败: {e:#}"),
        }
    }
}

/// ⚠️ **默认开**（`LINGXI_SERVE_PREFIX_CACHE=0` 才关）。
/// 与投机模式互斥（`LINGXI_SERVE_SPEC=1` 时强制关缓存：MTP 的 KV 也要跟着前缀走，
/// 目前还没把 MTP 状态纳入检查点）。
pub(crate) fn prefix_cache_enabled() -> bool {
    std::env::var("LINGXI_SERVE_PREFIX_CACHE").map(|v| v != "0").unwrap_or(true)
}

// ───────────────────────── 请求模板（对齐产物 chat_template.jinja）─────────────────────────
//
// 产物里的 `chat_template.jinja` 是唯一权威：
// - 思考：`<|im_start|>assistant\n<think>\n` 开场，模型只写思考内容，再自己收 `</think>`；
//   `enable_thinking=false` 时改成 `<think>\n\n</think>\n\n` 直接进入正文。
// - 工具：system 段注入 `<tools>` + 格式说明；模型输出
//   `<tool_call><function=名><parameter=键>值</parameter></function></tool_call>`。
// - 历史：assistant 的 `reasoning_content` 回填 `<think>`；`tool_calls` 重放成 XML；
//   连续 `role:"tool"` 合并进同一个 `<|im_start|>user` 的 `<tool_response>`。

/// 只认我们自己生成的文件名：`lingxi-ckpt-<13位毫秒>-<6位序号>.lxc`（可带 `.lxc.tmp`）。
/// 缓存目录里的其他文件一律不读、不删。
pub(crate) fn is_our_cache_file(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("lingxi-ckpt-") else {
        return false;
    };
    let stem = rest
        .strip_suffix(".lxc.tmp")
        .or_else(|| rest.strip_suffix(".lxc"));
    let Some(stem) = stem else { return false };
    let mut it = stem.split('-');
    let (Some(a), Some(b), None) = (it.next(), it.next(), it.next()) else {
        return false;
    };
    a.len() == 13
        && a.bytes().all(|c| c.is_ascii_digit())
        && b.len() == 6
        && b.bytes().all(|c| c.is_ascii_digit())
}

pub(crate) fn store_kind_name(e: &CkptEntry) -> &'static str {
    match (&e.store, e.kind) {
        (CkptStore::Dev(_), CkptKind::Boundary) => "dev:boundary",
        (CkptStore::Dev(_), CkptKind::Prompt) => "dev:prompt",
        (CkptStore::Dev(_), CkptKind::End) => "dev:end",
        (CkptStore::Disk(_), _) => "disk",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_cache() -> PrefillCache {
        PrefillCache {
            entries: Vec::new(),
            hits: 0,
            misses: 0,
            clock: 0,
            cache_bytes: 0,
            cache_limit: 0,
            cache_min_tokens: 0,
            cache_dir: None,
            device_ids: Vec::new(),
            cur_owner: None,
            tree: RadixTree::new(),
            next_key: 1,
        }
    }

    /// 确定性 token 序列（故意散布，避免 HashMap 聚集）。
    fn ids(n: usize, seed: i64) -> Vec<i64> {
        (0..n as i64).map(|i| (i * 31 + seed * 7) % 50_000 + 3).collect()
    }

    fn share_prefix(a: &mut Vec<i64>, b: &mut Vec<i64>, n: usize, seed: i64) {
        let p = ids(n, seed);
        a[..n].copy_from_slice(&p);
        b[..n].copy_from_slice(&p);
    }

    /// 测试用插入：等价于 push_disk 的内存部分（不写盘），返回 key。
    fn insert_disk(
        cache: &mut PrefillCache,
        kind: CkptKind,
        idv: Vec<i64>,
        alts: Vec<(usize, i64)>,
    ) -> u64 {
        let key = cache.next_key;
        cache.next_key += 1;
        cache.entries.push(CkptEntry {
            kind,
            ids: idv,
            next: 7,
            store: CkptStore::Disk(std::path::PathBuf::from("/nonexistent/x.lxc")),
            owner: None,
            bytes: 0,
            used: 0,
            alt_pairs: alts,
            key,
        });
        let idx = cache.entries.len() - 1;
        cache.index_insert(idx);
        key
    }

    /// 树上从根可达的节点数（含根）。
    fn reachable_nodes(tree: &RadixTree) -> usize {
        fn dfs(tree: &RadixTree, n: usize, seen: &mut std::collections::HashSet<usize>) -> usize {
            if !seen.insert(n) {
                return 0;
            }
            1 + tree.nodes[n].children.values().map(|&c| dfs(tree, c, seen)).sum::<usize>()
        }
        dfs(tree, 0, &mut std::collections::HashSet::new())
    }

    #[test]
    fn insert_and_longest_match() {
        let mut c = mk_cache();
        let mut a = ids(64, 1);
        let mut b = ids(96, 2);
        share_prefix(&mut a, &mut b, 48, 9); // a/b 共享前 48 tok
        // a 带 48 的短前缀档；b 只有主长 96
        let _ = insert_disk(&mut c, CkptKind::End, a.clone(), vec![(48, 123)]);
        insert_disk(&mut c, CkptKind::End, b.clone(), vec![]);
        // 查询 = b 自身：最长命中应为 96
        let (idx, l) = c.best(&b).expect("应命中");
        assert_eq!(l, 96);
        assert_eq!(c.entries[idx].ids.len(), 96);
        // 查询 = a 自身：命中 64
        assert_eq!(c.best(&a).unwrap().1, 64);
        // 查询 48：命中 a 的 alt 档（KV 共享），next 取 alt 档的 123
        let (idx, l) = c.best(&b[..48]).expect("alt 档应命中");
        assert_eq!(l, 48);
        assert_eq!(c.next_at(&c.entries[idx], l), 123);
        // 查询 60：路径上最长可用档是 48（64 档超过查询长，与旧实现一致）
        assert_eq!(c.best(&b[..60]).unwrap().1, 48);
        // 首 token 不同 / 空查询：miss
        let mut q = b.clone();
        q[0] = 1_000_000;
        assert!(c.best(&q).is_none());
        assert!(c.best(&[]).is_none());
    }

    #[test]
    fn shared_prefix_builds_single_path() {
        let mut c = mk_cache();
        let mut a = ids(80, 1);
        let mut b = ids(72, 2);
        share_prefix(&mut a, &mut b, 48, 42);
        insert_disk(&mut c, CkptKind::End, a.clone(), vec![]);
        insert_disk(&mut c, CkptKind::End, b.clone(), vec![]);
        // 共享 48 tok 只建一份路径：块 16+16+16 共享，之后各 2 块分叉
        // 总节点 = 根 + 3 共享 + 2(a 尾) + 2(b 尾) = 8
        let n = reachable_nodes(&c.tree);
        assert!(n <= 8, "共享前缀应去重，实际 {n} 节点");
        assert!(n >= 5);
        // 各自完整命中
        assert_eq!(c.best(&a).unwrap().1, 80);
        assert_eq!(c.best(&b).unwrap().1, 72);
    }

    #[test]
    fn remove_recycles_path() {
        let mut c = mk_cache();
        let mut a = ids(80, 1);
        let mut b = ids(72, 2);
        share_prefix(&mut a, &mut b, 48, 42);
        let ka = insert_disk(&mut c, CkptKind::End, a.clone(), vec![]);
        let kb = insert_disk(&mut c, CkptKind::End, b.clone(), vec![]);
        let full = reachable_nodes(&c.tree);
        // 删掉 b：私有后缀（48 之后）回收，共享段保留给 a
        c.tree.remove_path(&b, kb);
        let after_b = reachable_nodes(&c.tree);
        assert!(after_b < full, "删 b 后节点应变少");
        assert_eq!(c.best(&a).unwrap().1, 80, "a 不受影响");
        // 再删 a：全部回收（只剩根）
        c.tree.remove_path(&a, ka);
        assert_eq!(reachable_nodes(&c.tree), 1);
        assert!(c.tree.nodes[0].children.is_empty());
        assert!(c.tree.free.len() >= full - 1, "回收的槽位应进空闲表");
    }

    #[test]
    fn stale_ref_skipped_and_pruned() {
        let mut c = mk_cache();
        let mut a = ids(80, 1);
        let mut b = ids(72, 2);
        share_prefix(&mut a, &mut b, 48, 42);
        let ka = insert_disk(&mut c, CkptKind::End, a.clone(), vec![]);
        insert_disk(&mut c, CkptKind::End, b.clone(), vec![]);
        // 模拟 engine 的 drop_stale_dev：绕过 PrefillCache 直接删条目
        let pos = c.entries.iter().position(|e| e.key == ka).unwrap();
        c.entries.remove(pos);
        // 查询 a：陈旧引用被跳过并惰性摘除，best 为 None（b 在 48 处已分叉，
        // 而 a/b 的检查点长度分别是 80/72，共享段内无可用档）
        assert!(c.best(&a).is_none());
        // a 的整条私有路径已回收，只剩 b 的路径
        assert!(!c.tree.key_map.contains_key(&ka), "陈旧 key 应已清理");
        assert_eq!(c.tree.key_map.len(), 1);
        assert_eq!(reachable_nodes(&c.tree), 6, "根 + b 的 5 块");
        // b 不受影响
        assert_eq!(c.best(&b).unwrap().1, 72);
    }

    #[test]
    fn block_split_on_divergence() {
        let mut c = mk_cache();
        let a = ids(32, 5); // 块 [0..16] [16..32]
        insert_disk(&mut c, CkptKind::End, a.clone(), vec![]);
        // c 在 20 tok 处有一份 prompt 档（非块边界，挂在覆盖 16..32 的块节点上）
        insert_disk(&mut c, CkptKind::Prompt, a[..20].to_vec(), vec![]);
        assert_eq!(c.best(&a[..20]).unwrap().1, 20);
        // b 与 a 共享前 20 tok、在第 20 tok 处分叉（块 16..32 内分裂）
        let mut b = a[..20].to_vec();
        b.extend_from_slice(&ids(28, 6));
        b[20] = 999_999;
        insert_disk(&mut c, CkptKind::End, b.clone(), vec![]);
        // 分裂后三份各自命中
        assert_eq!(c.best(&a).unwrap().1, 32);
        assert_eq!(c.best(&b).unwrap().1, 48);
        assert_eq!(c.best(&a[..20]).unwrap().1, 20);
        // 19 tok：没有这么短的档，miss（与旧实现一致）
        assert!(c.best(&a[..19]).is_none());
    }

    #[test]
    fn alt_pairs_shorter_len_hits() {
        let mut c = mk_cache();
        let a = ids(64, 8);
        // 主长 64，短前缀档 20
        insert_disk(&mut c, CkptKind::End, a.clone(), vec![(20, 555)]);
        // 查询 30 tok：主长挂的节点在 64，只命中 alt 档 20
        let (_, l) = c.best(&a[..30]).unwrap();
        assert_eq!(l, 20);
        // next_at 能取到 alt 档的 next
        let (idx, l) = c.best(&a[..30]).unwrap();
        assert_eq!(c.next_at(&c.entries[idx], l), 555);
        // 查询 64 tok：主档命中
        assert_eq!(c.best(&a).unwrap().1, 64);
    }

    #[test]
    fn mid_block_len_attaches_without_split() {
        let mut c = mk_cache();
        let a = ids(64, 11);
        // 长度 20（非 16 整数倍）：挂在覆盖 16..32 的块节点上
        insert_disk(&mut c, CkptKind::Prompt, a[..20].to_vec(), vec![]);
        assert_eq!(c.best(&a[..20]).unwrap().1, 20);
        // 之后再插 64 长的同前缀条目：块节点分裂，两者仍各自命中
        insert_disk(&mut c, CkptKind::End, a.clone(), vec![]);
        assert_eq!(c.best(&a[..20]).unwrap().1, 20);
        assert_eq!(c.best(&a).unwrap().1, 64);
    }

    #[test]
    fn invalidate_owner_keeps_other_entries() {
        let mut c = mk_cache();
        let mut a = ids(48, 1);
        let mut b = ids(48, 2);
        share_prefix(&mut a, &mut b, 32, 3);
        // Disk 条目带 owner 字段也恒有效（数据自包含）——invalidate_owner 只清 Dev
        let ka = insert_disk(&mut c, CkptKind::End, a.clone(), vec![]);
        let kb = insert_disk(&mut c, CkptKind::End, b.clone(), vec![]);
        c.entries[0].owner = Some("s1".into());
        c.set_owner("s2");
        c.invalidate_owner("s1");
        // Dev 条目没有（全是 Disk），两条都还在
        assert_eq!(c.entries.len(), 2);
        assert_eq!(c.best(&a).unwrap().1, 48);
        assert_eq!(c.best(&b).unwrap().1, 48);
        let _ = (ka, kb);
    }
}
