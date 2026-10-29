//! QSA 分页：host 锁页块池、槽分配/驱逐、写物化、按块检查点（paged_kv_to/from_host）
//! 与沉取账本。**本模块独占 host 池句柄**——状态所有权显式化（R3 动机：
//! "prefill 双线程 ⇒ scratch 双份"类坑不能再靠人肉追）。

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

use super::core_::QsaState;


/// P2 分页账本：累计下沉 / 预取（冷块命中取回）次数。serve 的 perf 行按请求引用，
/// talk 用 `LINGXI_QSA_PAGED_NAN=1` 时的日志即可观察。选中率应低——高 = 窗口配太小。
pub(crate) static PAGED_SINKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static PAGED_PREFETCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// (累计下沉, 累计预取) —— 进程级账本。
pub fn paged_stats() -> (u64, u64) {
    (
        PAGED_SINKS.load(std::sync::atomic::Ordering::Relaxed),
        PAGED_PREFETCHES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// 弹性 KV 的**初始分配** token 数（`LINGXI_KV_INIT_TOKENS`，默认 8192）：
/// 短对话不再预付 256K 的池子；几何增长（倍增）把摊销拷贝控制在 ≈2×。
pub(crate) fn kv_init_tokens() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LINGXI_KV_INIT_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 64)
            .unwrap_or(8192)
    })
}


impl QsaState {
    pub(crate) fn host_pool_grow(&mut self, k: &Kernels, need: usize) -> Result<()> {
        if need <= self.host_pool_slots {
            return Ok(());
        }
        k.stream.synchronize()?;
        let mut new_slots = self.host_pool_slots.max(1);
        while new_slots < need {
            new_slots *= 2;
        }
        let be = self.blk_elems();
        let mut p = unsafe {
            k.stream
                .context()
                .alloc_pinned_with_flags::<bf16>(new_slots * 2 * be, 0)?
        };
        if let Some(old) = self.host_pool.take() {
            let old_len = old.len();
            let old_s = old.as_slice()?;
            p.as_mut_slice()?[..old_len].copy_from_slice(old_s);
            self.host_free.extend((old_len / (2 * be)) as u32..new_slots as u32);
        } else {
            self.host_free.extend(0..new_slots as u32);
        }
        self.host_pool = Some(p);
        self.host_pool_slots = new_slots;
        Ok(())
    }

    /// 驱逐一个常驻满块（< protect_full）下沉 host，返回其槽号。
    /// - `protect_step`：为 true 时（admit 预取路径），touch==本步 的块（本步选中/本步刚预取）
    ///   **绝不驱逐**——否则选中集覆盖整个窗口时会把"本步要读的块"挤出去（静默 corruption
    ///   的实测根因）。无处可驱逐 ⇒ 响亮报错（窗口 ≤ 选中工作集的明确信号，拒绝静默出错）。

    pub(crate) fn slot_alloc(&mut self, k: &Kernels, protect_full: usize, protect_step: bool) -> Result<i32> {
        if let Some(s) = self.free_slots.pop() {
            return Ok(s);
        }
        let mut victim: Option<usize> = None;
        let mut best = u64::MAX;
        for b in 0..protect_full.min(self.tab_host.len()) {
            if self.tab_host[b] >= 0 && self.blk_touch[b] < best {
                if protect_step && self.blk_touch[b] == self.step_ctr {
                    continue; // 本步选中/刚预取：驱逐它必然让 gather 读到已下沉块
                }
                best = self.blk_touch[b];
                victim = Some(b);
            }
        }
        let vb = victim.ok_or_else(|| {
            anyhow::anyhow!(
                "QSA paged: 常驻窗口耗尽且本步选中集覆盖全部常驻块（win={} 块 ≤ 选中工作集；\
                 调大 LINGXI_QSA_WIN_TOKENS）",
                self.win_blocks
            )
        })?;
        let slot = self.tab_host[vb];
        // 找 host 空坑（必要时倍增池）
        if self.host_free.is_empty() {
            self.host_pool_grow(k, self.host_pool_slots.max(1) * 2)?;
        }
        let h = self.host_free.pop().unwrap() as usize;
        let be = self.blk_elems();
        let kn = self.kn.as_ref().unwrap();
        let v = self.v.as_ref().unwrap();
        let pool = self.host_pool.as_mut().unwrap();
        // D2H 下沉（同流排序；pinned ⇒ 真 DMA）
        // [判决开关] `LINGXI_QSA_PAGED_SINK_NOP=1`：跳过下沉拷贝（池留垃圾）——
        // 用于区分"读路径走池"与"破坏在别处"。
        if !crate::flag("LINGXI_QSA_PAGED_SINK_NOP") {
            let psl = pool.as_mut_slice()?;
            // ⚠️ 按**槽号**切（不是逻辑块号 vb——重映射后二者不同，按 vb 切会越界）
            let su = slot as usize;
            let ksrc = kn.slice(su * be..(su + 1) * be);
            let (sp, _g) = ksrc.device_ptr(&k.stream);
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async::<bf16>(
                    &mut psl[h * 2 * be..h * 2 * be + be],
                    sp,
                    k.stream.cu_stream(),
                )?
            };
            let vsrc = v.slice(su * be..(su + 1) * be);
            let (sp2, _g2) = vsrc.device_ptr(&k.stream);
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async::<bf16>(
                    &mut psl[h * 2 * be + be..h * 2 * be + 2 * be],
                    sp2,
                    k.stream.cu_stream(),
                )?
            };
        }
        self.host_blk[vb] = h as u32;
        self.tab_host[vb] = -1;
        self.n_sunk += 1;
        self.tab_dirty = true;
        PAGED_SINKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(slot)
    }

    /// 表脏时把 host 镜像整体刷到设备（**异步**——同流排序，后续 gather 必然读到；
    /// attention kernel 读的是设备表）。

    pub fn tab_flush(&mut self, k: &Kernels) -> Result<()> {
        if self.paged && self.tab_dirty {
            
            let (p, _g) = self.blk_tab.device_ptr_mut(&k.stream);
            unsafe {
                cudarc::driver::result::memcpy_htod_async::<i32>(
                    p,
                    &self.tab_host,
                    k.stream.cu_stream(),
                )?
            };
            self.tab_dirty = false;
        }
        Ok(())
    }

    /// 把下沉块 b 预取回 GPU 槽（H2D kn+v，回填表，LRU touch，n_sunk−1）。
    /// 调用方保证 tab_host[b] == -1。admit 第 2 遍与写前物化共用。

    pub(crate) fn block_prefetch(&mut self, k: &Kernels, b: usize, protect_full: usize) -> Result<()> {
        let slot = self.slot_alloc(k, protect_full, true)?;
        if crate::flag("LINGXI_QSA_PAGED_NAN") {
            eprintln!("[admit] 预取块 b={b} → 槽 {slot}（host 坑 {}）", self.host_blk[b]);
        }
        let h = self.host_blk[b].max(0) as usize;
        let be = self.blk_elems();
        let psl = self.host_pool.as_ref().unwrap().as_slice()?;
        let kn = self.kn.as_mut().unwrap();
        {
            let mut dst = kn.slice_mut(slot as usize * be..(slot as usize + 1) * be);
            let (dp, _g) = dst.device_ptr_mut(&k.stream);
            unsafe {
                cudarc::driver::result::memcpy_htod_async::<bf16>(
                    dp,
                    &psl[h * 2 * be..h * 2 * be + be],
                    k.stream.cu_stream(),
                )?
            };
        }
        let v = self.v.as_mut().unwrap();
        {
            let mut dst = v.slice_mut(slot as usize * be..(slot as usize + 1) * be);
            let (dp, _g) = dst.device_ptr_mut(&k.stream);
            unsafe {
                cudarc::driver::result::memcpy_htod_async::<bf16>(
                    dp,
                    &psl[h * 2 * be + be..h * 2 * be + 2 * be],
                    k.stream.cu_stream(),
                )?
            };
        }
        self.tab_host[b] = slot;
        self.blk_touch[b] = self.step_ctr;
        self.n_sunk -= 1;
        self.tab_dirty = true;
        PAGED_PREFETCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // [判决性自检] `LINGXI_QSA_PAGED_CHECK=1`：预取后同步并回读 GPU 槽，
        // 与 host 池逐字节比对——钉死"下沉写错"还是"读取错"。
        if crate::flag("LINGXI_QSA_PAGED_CHECK") {
            use cudarc::driver::DevicePtr;
            k.stream.synchronize()?;
            let su = slot as usize;
            let mut back = vec![0u16; 2 * be];
            {
                let src = self.kn.as_ref().unwrap().slice(su * be..(su + 1) * be);
                let (p, _g) = src.device_ptr(&k.stream);
                unsafe { cudarc::driver::result::memcpy_dtoh_sync::<u16>(&mut back[..be], p)? };
            }
            {
                let src = self.v.as_ref().unwrap().slice(su * be..(su + 1) * be);
                let (p, _g) = src.device_ptr(&k.stream);
                unsafe { cudarc::driver::result::memcpy_dtoh_sync::<u16>(&mut back[be..], p)? };
            }
            let psl = self.host_pool.as_ref().unwrap().as_slice()?;
            let mut bad = 0usize;
            for i in 0..be {
                if back[i] != psl[h * 2 * be + i].to_bits() {
                    bad += 1;
                }
                if back[be + i] != psl[h * 2 * be + be + i].to_bits() {
                    bad += 1;
                }
            }
            anyhow::ensure!(bad == 0, "QSA paged CHECK: 预取块 b={b} 槽 {slot} 回读不一致 {bad} 元素");
        }
        Ok(())
    }

    /// 写前物化：保证 [wfrom, wfrom+wt) 覆盖的块都分配了 GPU 槽（本步写入 + gather 尾部都读它）。

    pub fn kv_materialize_write(&mut self, k: &Kernels, wfrom: usize, wt: usize) -> Result<()> {
        if !self.sealed || wt == 0 {
            return Ok(());
        }
        let cp = self.idx_compress;
        let b0 = wfrom / cp;
        let b1 = (wfrom + wt - 1) / cp;
        for b in b0..=b1 {
            if self.tab_host[b] == -1 {
                // 恢复出来的尾部半满块可能是下沉态（Disk 缓存恢复 = 全部块进 host 池）：
                // 先预取回来保留已有内容，再在其后追加写。
                self.block_prefetch(k, b, b0)?;
            }
            if self.tab_host[b] == -2 {
                // protect_step=false：写前物化的驱逐不保护本步选中块——被选中的受害者
                // 由随后 admit 的预取补回（admit 在 gather 前跑）。
                let s = self.slot_alloc(k, b0, false)?;
                self.tab_host[b] = s;
                self.blk_touch[b] = self.step_ctr;
                self.tab_dirty = true;
            }
        }
        self.tab_flush(k)
    }

    /// 选块准入（gather 前）：D2H sel → 选中块预取回填（LRU 记账）→ 刷表。
    /// **快路**：没有任何下沉块时跳过名单 D2H/同步（窗口未满的常见形态零开销）；
    /// LRU 记账在首次驱逐前缺失是安全的（驱逐按 touch 最小选，退化为准 FIFO，自愈）。
    /// `row0`：处理 sel 的行段起点（prefill 行批模式；批 b 只看自己的行，并集小一个量级）。

    pub fn kv_append_paged(&mut self, k: &Kernels, kn_new: &CudaSlice<bf16>,
                           v_new: &CudaSlice<bf16>, wfrom: usize, wt: usize) -> Result<()> {
        let cp = self.idx_compress;
        let end = wfrom + wt;
        let kn = self.kn.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();
        let mut cur = wfrom;
        while cur < end {
            let b = cur / cp;
            let row_end = ((b + 1) * cp).min(end);
            let n = row_end - cur;
            let slot = self.tab_host[b];
            anyhow::ensure!(slot >= 0, "kv_append_paged: 块 {b} 未物化");
            let off = (slot as usize) * cp + (cur - b * cp);
            k.stream.memcpy_dtod(&kn_new.slice((cur - wfrom) * self.kv_dim..(cur - wfrom + n) * self.kv_dim),
                                 &mut kn.slice_mut(off * self.kv_dim..(off + n) * self.kv_dim))?;
            k.stream.memcpy_dtod(&v_new.slice((cur - wfrom) * self.kv_dim..(cur - wfrom + n) * self.kv_dim),
                                 &mut v.slice_mut(off * self.kv_dim..(off + n) * self.kv_dim))?;
            cur = row_end;
        }
        Ok(())
    }

    /// **分页**：把 [0, len) 的逻辑 KV 按块表装配成连续 host 行（host checkpoint 用）。
    /// resident 块 D2H；sunk 块从 host 池直拷。返回 (kn, v) 的 u16 位型行主序。

    pub fn paged_kv_to_host(&self, k: &Kernels, len: usize) -> Result<(Vec<u16>, Vec<u16>)> {
        use cudarc::driver::DevicePtr;
        let kv_dim = self.kv_dim;
        let cp = self.idx_compress;
        let be = self.blk_elems();
        let mut kn_h = vec![0u16; len * kv_dim];
        let mut v_h = vec![0u16; len * kv_dim];
        let mut cur = 0usize;
        while cur < len {
            let b = cur / cp;
            let row_end = ((b + 1) * cp).min(len);
            let n = row_end - cur;
            let dst = cur * kv_dim..(cur + n) * kv_dim;
            match self.tab_host[b] {
                s if s >= 0 => {
                    let row0 = (cur - b * cp) * kv_dim;
                    let src = self.kn.as_ref().unwrap()
                        .slice(s as usize * be + row0..s as usize * be + row0 + n * kv_dim);
                    let (p, _g) = src.device_ptr(&k.stream);
                    unsafe { cudarc::driver::result::memcpy_dtoh_sync::<u16>(&mut kn_h[dst.clone()], p)? };
                    let src = self.v.as_ref().unwrap()
                        .slice(s as usize * be + row0..s as usize * be + row0 + n * kv_dim);
                    let (p, _g) = src.device_ptr(&k.stream);
                    unsafe { cudarc::driver::result::memcpy_dtoh_sync::<u16>(&mut v_h[dst], p)? };
                }
                -1 => {
                    let h = self.host_blk[b] as usize;
                    let row0 = (cur - b * cp) * kv_dim;
                    let psl = self.host_pool.as_ref().unwrap().as_slice()?;
                    for i in 0..n * kv_dim {
                        kn_h[dst.start + i] = psl[h * 2 * be + row0 + i].to_bits();
                        v_h[dst.start + i] = psl[h * 2 * be + be + row0 + i].to_bits();
                    }
                }
                _ => anyhow::bail!("paged_kv_to_host: 块 {b} 未分配"),
            }
            cur = row_end;
        }
        Ok((kn_h, v_h))
    }

    /// **分页**：从连续 host 行（host checkpoint）整体恢复——全部块写进 host 池、
    /// 表项置下沉、GPU 槽全空（首步选块时按需预取）。调用前必须先 `reset`（rewind(0)）。
    /// 2026-10-29 起同时是 `paged_seal` 的落点（prefill 许可模式）：恢复/封印后的
    /// 状态都是"全量在 host 池"的封印态，故本函数末尾置 `sealed=true`。
    pub fn paged_kv_from_host(&mut self, k: &Kernels, kn_h: &[u16], v_h: &[u16]) -> Result<()> {
        let len = kn_h.len() / self.kv_dim.max(1);
        anyhow::ensure!(v_h.len() == kn_h.len(), "paged_kv_from_host: kn/v 长度不一致");
        let cp = self.idx_compress;
        let be = self.blk_elems();
        // 清场：槽/host 坑全归还（rewind(0) 已释放块；这里重置分配器本身）。
        self.free_slots = (0..self.win_blocks as i32).rev().collect();
        self.host_free.clear();
        self.host_free.extend(0..self.host_pool_slots as u32);
        self.host_blk.fill(u32::MAX);
        self.tab_host.fill(-2);
        self.n_sunk = 0;
        let nb = (len + cp - 1) / cp;
        self.host_pool_grow(k, nb.max(1))?;
        let mut cur = 0usize;
        while cur < len {
            let b = cur / cp;
            let row_end = ((b + 1) * cp).min(len);
            let n = row_end - cur;
            let h = self.host_free.pop().unwrap() as usize;
            let row0 = (cur - b * cp) * self.kv_dim;
            let psl = self.host_pool.as_mut().unwrap().as_mut_slice()?;
            for i in 0..n * self.kv_dim {
                psl[h * 2 * be + row0 + i] = bf16::from_bits(kn_h[cur * self.kv_dim + i]);
                psl[h * 2 * be + be + row0 + i] = bf16::from_bits(v_h[cur * self.kv_dim + i]);
            }
            self.host_blk[b] = h as u32;
            self.tab_host[b] = -1;
            self.n_sunk += 1;
            cur = row_end;
        }
        self.tab_dirty = true;
        self.sealed = true;
        self.tab_flush(k)
    }
}
