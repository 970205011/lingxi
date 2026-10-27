//! R4 拆分（2026-10-21）：原 serve.rs 3152 行按职责分段，本模块为其中一段。
//! 模块总览见 `serve/mod.rs`。段间共享类型一律 `pub(crate)`。

/// **跨请求智能前缀缓存**（C3，2026-10-18）：多轮对话里下一轮的 prompt 通常以
/// “上一轮 prompt + 回复 + 新消息”开头 ⇒ 已进 KV 的那段不用重算。
///
/// 与 C2（只认完全匹配）的区别：
/// - 缓存里放 **1~2 个状态检查点**（`StateCheckpoint`：GDN/PLE 真拷贝 + QSA 长度）；
/// - 请求进来时找**最长匹配**的检查点（不要求最后一个字也对上）；
/// - 另外在“最后一条消息之前”存一个**消息边界检查点**：用户只改最后一句时，
/// 前面的历史直接复用，只重读新消息。
///
/// 默认**开**（`LINGXI_SERVE_PREFIX_CACHE=0` 可关）。
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
    /// 落盘占用（Dev 条目为 0）
    pub(crate) bytes: usize,
    /// LRU 时钟
    pub(crate) used: u64,
    /// 同一会话的较短前缀档：`(长度, next)`（KV 共享，只有 GDN/PLE 状态不同）
    pub(crate) alt_pairs: Vec<(usize, i64)>,
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

    /// 这个 Dev 条目还跟当前设备 KV 对得上吗？（别的会话 prefill 过就作废）
    pub(crate) fn dev_valid(&self, e: &CkptEntry) -> bool {
        !e.ids.is_empty()
            && self.device_ids.len() >= e.ids.len()
            && self.device_ids[..e.ids.len()] == e.ids[..]
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

    /// 找“最长且完整匹配”的检查点，返回 `(下标, 可复用长度)`。
    ///
    /// Dev 条目必须仍然是当前设备 KV 的前缀；Disk 条目永远有效（数据在盘上）。
    /// Disk 条目除了主长度外，还可能带若干**较短前缀档**（KV 共享，只多存 GDN/PLE）。
    pub(crate) fn best(&self, ids: &[i64]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.ids.is_empty() || e.ids[0] != ids[0] {
                continue;
            }
            if matches!(e.store, CkptStore::Dev(_)) && !self.dev_valid(e) {
                continue;
            }
            let mut cands = vec![e.ids.len()];
            cands.extend(e.alt_pairs.iter().map(|(l, _)| *l));
            for l in cands {
                if l == 0 || l > ids.len() || l > e.ids.len() {
                    continue;
                }
                if ids[..l] != e.ids[..l] {
                    continue;
                }
                if best.map_or(true, |(_, bl)| l > bl) {
                    best = Some((i, l));
                }
            }
        }
        best
    }

    /// ㊿-P1：会话切换 ⇒ 设备 KV 换手。Dev 检查点（引用设备状态）全部作废并释放，
    /// host/disk 检查点按 ids 前缀跨会话仍然有效，保留。
    pub(crate) fn invalidate_device(&mut self) {
        self.entries.retain(|e| !matches!(e.store, CkptStore::Dev(_)));
        self.device_ids.clear();
    }

    /// 插入/替换**当前会话**的 Dev 检查点（每种只留一份）。
    pub(crate) fn push_dev(
        &mut self,
        kind: CkptKind,
        ids: Vec<i64>,
        next: i64,
        cp: lingxi_driver::StateCheckpoint,
    ) {
        self.entries
            .retain(|e| !(e.kind == kind && matches!(e.store, CkptStore::Dev(_))));
        let used = self.tick();
        self.entries.push(CkptEntry {
            kind,
            ids,
            next,
            store: CkptStore::Dev(cp),
            bytes: 0,
            used,
            alt_pairs: Vec::new(),
        });
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
                self.entries.retain(|e| {
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
                self.entries.push(CkptEntry {
                    kind,
                    ids,
                    next,
                    store: CkptStore::Disk(path),
                    bytes: file_bytes as usize,
                    used,
                    alt_pairs,
                });
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
            self.entries.push(CkptEntry {
                kind: CkptKind::End,
                ids,
                next,
                store: CkptStore::Disk(path),
                bytes: bytes as usize,
                used,
                alt_pairs,
            });
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
            self.entries.retain(|e| !matches!(e.store, CkptStore::Dev(_)));
            return Ok(());
        }
        let pos = self
            .entries
            .iter()
            .position(|e| e.kind == CkptKind::End && matches!(e.store, CkptStore::Dev(_)));
        let Some(pos) = pos else { return Ok(()) };
        if !self.dev_valid(&self.entries[pos]) {
            // 设备 KV 已经不是它的了 ⇒ 这份 Dev 快照作废
            self.entries.retain(|e| !matches!(e.store, CkptStore::Dev(_)));
            return Ok(());
        }
        let ids = self.entries[pos].ids.clone();
        let next = self.entries[pos].next;
        // 短上下文：GDN/PLE 固定 ~113MB/份，存 18 tok 也要 226MB，不如重算
        if ids.len() < self.cache_min_tokens {
            self.entries.retain(|e| !matches!(e.store, CkptStore::Dev(_)));
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
        self.entries.retain(|e| !matches!(e.store, CkptStore::Dev(_)));
        self.push_disk(CkptKind::End, ids, next, cp);
        Ok(())
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
) {
    if lingxi_model::flag("LINGXI_QSA_PAGED") {
        match seq.save_host_checkpoint(eng, len) {
            Ok(cp) => cache.push_disk(kind, seq.ids[..len].to_vec(), next, cp),
            Err(e) => eprintln!("[serve] {what} host 检查点保存失败: {e:#}"),
        }
    } else {
        match seq.save_checkpoint(eng, len) {
            Ok(cp) => cache.push_dev(kind, seq.ids[..len].to_vec(), next, cp),
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
// `enable_thinking=false` 时改成 `<think>\n\n</think>\n\n` 直接进入正文。
// - 工具：system 段注入 `<tools>` + 格式说明；模型输出
// `<tool_call><function=名><parameter=键>值</parameter></function></tool_call>`。
// - 历史：assistant 的 `reasoning_content` 回填 `<think>`；`tool_calls` 重放成 XML；
// 连续 `role:"tool"` 合并进同一个 `<|im_start|>user` 的 `<tool_response>`。

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
