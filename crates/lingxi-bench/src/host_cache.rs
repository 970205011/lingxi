//! Host 内存权重缓存：用大内存消灭 decode 期机械盘 IO。
//!
//! - **dense 常驻**：每层非专家权重（GR/GDN/QSA/shared/router，~150MB/层、
//!   全 48 层约 8~10GB）首次读后永久驻留匿名内存，并由后台线程**提前一层**
//!   预读——盘 IO 与上一层 GPU 计算重叠。
//! - **专家零拷贝 mmap 视图**：专家权重（225GB 量级）不做匿名副本，直接借
//!   产物 mmap 页；首次访问触发 page fault 读盘（在 device 聚集的并行拷贝
//!   线程里自然并行），之后靠内核 page cache 常驻——256GB 内存足以装下全部
//!   专家页，跨 token、跨 run 全部零盘 IO。比手写 LRU 省一次 memcpy/槽，
//!   且无预算淘汰抖动。
//!
//! 正确性：mmap 字节是产物 payload 本身（bf16 即 u16 位形，逐位一致），
//! 全程只读。

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, DevicePtrMut, PinnedHostSlice};
use half::bf16;
use lingxi_artifact::Artifact;
use lingxi_kernels::Kernels;

/// 非临时（streaming）内存拷贝：NT 写绕过 cache、不触发 RFO，
/// 对"只读源 + 只写大块目标"（如 mmap 权重 → pinned 上传缓冲）远快于
/// 普通 `memcpy`（本机 Ivy Bridge + 虚拟化下普通拷贝单核仅 ~1.6GB/s）。
///
/// # Safety
/// 调用方保证：`src..src+bytes`、`dst..dst+bytes` 有效且不重叠；
/// bytes 为 32 的倍数（调用处槽大小恒为 32 字节对齐）。
#[inline]
unsafe fn copy_stream_nt(src: *const u16, dst: *mut u16, elems: usize) {
    use std::arch::x86_64::*;
    let bytes = elems * 2;
    if is_x86_feature_detected!("avx") {
        #[target_feature(enable = "avx")]
        unsafe fn nt(src: *const u8, dst: *mut u8, bytes: usize) {
            let mut i = 0;
            while i + 64 <= bytes {
                _mm256_stream_si256(
                    dst.add(i) as *mut __m256i,
                    _mm256_loadu_si256(src.add(i) as *const __m256i),
                );
                _mm256_stream_si256(
                    dst.add(i + 32) as *mut __m256i,
                    _mm256_loadu_si256(src.add(i + 32) as *const __m256i),
                );
                i += 64;
            }
            if i + 32 <= bytes {
                _mm256_stream_si256(
                    dst.add(i) as *mut __m256i,
                    _mm256_loadu_si256(src.add(i) as *const __m256i),
                );
                i += 32;
            }
            // 本调用处 bytes 恒为 32 倍数；保险起见处理尾段
            if i < bytes {
                std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), bytes - i);
            }
            // 不在此处 sfence：调用方（worker 池栅栏）会在本核一批拷贝后统一 sfence
        }
        nt(src as *const u8, dst as *mut u8, bytes);
    } else {
        std::ptr::copy_nonoverlapping(src, dst, elems);
    }
}


/// 一份 bf16 权重数据的来源：匿名副本（dense 小张量）或产物 mmap 只读视图。
///
/// `Mmap` 持有 `Arc<Artifact>` 保证映射存活；裸指针只在借用期内解引用。
/// 人为只读共享，故可跨线程传递。
#[derive(Clone)]
pub(crate) enum Blob {
    Owned(Arc<Vec<bf16>>),
    /// 一个专家槽的两段视图：gu `[2I,H]` 与 dn `[H,I]`（在产物里不相邻）。
    Mmap {
        gu: *const bf16,
        dn: *const bf16,
        _art: Arc<Artifact>,
    },
}
unsafe impl Send for Blob {}
unsafe impl Sync for Blob {}

impl Blob {
    /// gu 段（`per_gu` 个 bf16）。
    fn gu(&self, per_gu: usize) -> &[bf16] {
        match self {
            Blob::Owned(v) => &v[..per_gu],
            Blob::Mmap { gu, .. } => unsafe { std::slice::from_raw_parts(*gu, per_gu) },
        }
    }

    /// dn 段（`per_dn` 个 bf16，紧跟 gu 之后）。
    fn dn(&self, per_gu: usize, per_dn: usize) -> &[bf16] {
        match self {
            Blob::Owned(v) => &v[per_gu..per_gu + per_dn],
            Blob::Mmap { dn, .. } => unsafe { std::slice::from_raw_parts(*dn, per_dn) },
        }
    }

    /// 整体连续切片（仅 `Owned`/dense 路径使用）。
    fn as_contiguous(&self) -> &[bf16] {
        match self {
            Blob::Owned(v) => v,
            Blob::Mmap { .. } => unreachable!("mmap 专家视图不是连续单段"),
        }
    }
}

pub struct HostCache {
    art: Arc<Artifact>,
    dense: Mutex<HashMap<String, Blob>>,
    prefetch_tx: OnceLock<Sender<String>>,
}

fn bf16_from_payload(bytes: &[u8]) -> Blob {
    Blob::Owned(Arc::new(
        bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    ))
}

impl HostCache {
    pub fn new(art: Arc<Artifact>) -> Arc<Self> {
        Arc::new(Self {
            art,
            dense: Mutex::new(HashMap::new()),
            prefetch_tx: OnceLock::new(),
        })
    }

    /// 启动后台预读线程（只做 mmap→host，不碰 GPU）。幂等。
    pub fn start_prefetcher(self: &Arc<Self>) {
        self.prefetch_tx.get_or_init(|| {
            let (tx, rx) = channel::<String>();
            let c = Arc::clone(self);
            std::thread::Builder::new()
                .name("lingxi-prefetch".into())
                .spawn(move || {
                    let mut queued: std::collections::HashSet<String> = std::collections::HashSet::new();
                    while let Ok(name) = rx.recv() {
                        queued.remove(&name);
                        // 已缓存则跳过
                        if c.dense.lock().unwrap().contains_key(&name) {
                            continue;
                        }
                        if let Some(bytes) = c.art.payload(&name) {
                            let blob = bf16_from_payload(bytes);
                            c.dense.lock().unwrap().entry(name).or_insert(blob);
                        }
                    }
                })
                .expect("启动预读线程失败");
            tx
        });
    }

    /// 异步请求预读一个 dense 张量（后台线程读 mmap 填充）。
    pub fn request_dense(&self, name: String) {
        if let Some(tx) = self.prefetch_tx.get() {
            // 粗去重：锁内查一次，避免队列堆积重复请求
            if self.dense.lock().unwrap().contains_key(&name) {
                return;
            }
            let _ = tx.send(name);
        }
    }

    /// 取 dense 张量（缓存命中零拷贝返回 Arc；未命中则同步从 mmap 填充）。
    fn dense_blob(&self, name: &str) -> Result<Blob> {
        if let Some(b) = self.dense.lock().unwrap().get(name) {
            return Ok(b.clone());
        }
        let bytes = self.art.payload(name).with_context(|| format!("产物缺少 {name}"))?;
        let blob = bf16_from_payload(bytes);
        Ok(self
            .dense
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_insert(blob)
            .clone())
    }

    /// dense 上传：缓存字节 → H2D。
    pub fn upload_dense(&self, k: &Kernels, name: &str) -> Result<CudaSlice<bf16>> {
        let blob = self.dense_blob(name)?;
        Ok(k.stream.clone_htod(blob.as_contiguous())?)
    }

    /// 构造本层命中专家的 mmap 只读视图（与 `experts` 对齐）。
    ///
    /// 纯元数据操作、零拷贝、零锁；首次触页的 major fault 延迟到 device
    /// 聚集的并行拷贝线程发生（多槽并行 fault，机械盘寻道自然重叠）。
    pub fn fetch_hits(
        &self,
        layer: u32,
        experts: &[u32],
        inter: usize,
        hidden: usize,
    ) -> Result<Vec<Blob>> {
        let per_gu = 2 * inter * hidden;
        let per_dn = hidden * inter;
        let p = format!("text/layers/{layer}/moe/");
        let gu_bytes = self
            .art
            .payload(&format!("{p}routed_gate_up"))
            .with_context(|| format!("缺少 {p}routed_gate_up"))?;
        let dn_bytes = self
            .art
            .payload(&format!("{p}routed_down"))
            .with_context(|| format!("缺少 {p}routed_down"))?;
        let gu_base = gu_bytes.as_ptr() as *const bf16;
        let dn_base = dn_bytes.as_ptr() as *const bf16;
        let mut out = Vec::with_capacity(experts.len());
        for &e in experts {
            let ee = e as usize;
            out.push(Blob::Mmap {
                gu: unsafe { gu_base.add(ee * per_gu) },
                dn: unsafe { dn_base.add(ee * per_dn) },
                _art: Arc::clone(&self.art),
            });
        }
        Ok(out)
    }

    /// 预取一层全部 dense 权重名字（后台线程顺序读）。
    pub fn prefetch_layer_dense(&self, layer: u32) {
        let p = format!("text/layers/{layer}/");
        let kind = lingxi_model::layer_kind(layer);
        let mut names = vec![
            format!("{p}attn_gr/hc_norm"),
            format!("{p}attn_gr/mix_down"),
            format!("{p}attn_gr/mix_up"),
            format!("{p}attn_gr/block_inject"),
            format!("{p}mlp_gr/hc_norm"),
            format!("{p}mlp_gr/mix_down"),
            format!("{p}mlp_gr/mix_up"),
            format!("{p}mlp_gr/block_inject"),
            format!("{p}moe/router_shared_gate"),
            format!("{p}moe/shared_gate_up"),
            format!("{p}moe/shared_down"),
        ];
        match kind {
            lingxi_ids::LayerKind::QsaSparseAttention => {
                names.push(format!("{p}attention/indexer/k_layernorm"));
                names.push(format!("{p}attention/query_key_gate_value"));
                names.push(format!("{p}attention/query_norm"));
                names.push(format!("{p}attention/key_norm"));
                names.push(format!("{p}attention/output"));
                names.push(format!("{p}attention/indexer/index_qk_proj"));
                names.push(format!("{p}attention/indexer/q_layernorm"));
            }
            lingxi_ids::LayerKind::GdnLinearAttention => {
                // a_log/dt_bias 是 FP32 且小，调用方仍走 read_fp32_host（page cache 即可）
                names.push(format!("{p}gdn/query_key_value_z"));
                names.push(format!("{p}gdn/a_b_projection"));
                names.push(format!("{p}gdn/convolution"));
                names.push(format!("{p}gdn/norm"));
                names.push(format!("{p}gdn/output"));
            }
        }
        for n in names {
            self.request_dense(n);
        }
    }
}

/// 一个专家在显存里的完整两张量槽：gu `[2I, H]`（6.25MB）+ dn `[H, I]`（3.125MB）。
struct DevSlot {
    gu: CudaSlice<bf16>,
    dn: CudaSlice<bf16>,
}

/// device 侧专家 LRU：热专家整槽常驻显存（decode 跨步直接复用，零 PCIe、零 host 拷贝）。
///
/// 每步每层把本步命中的专家聚集到一对 scratch 紧凑张量：
/// - **device 命中**：`dtod` 从槽位拷到 scratch 偏移（HBM 内，每槽 9.375MiB）；
/// - **未命中**：host blob 分槽 `htod` 直填 scratch 偏移（不经 host 聚合缓冲），
///   再 `dtod` 回填一个新槽供以后命中。
///
/// 淘汰时 `CudaSlice` 的 Drop 自带流同步/流序回收（cudarc），不会释放仍在流中
/// 被引用的显存。本结构**非线程安全**：driver 单序列只在主线程逐层调用。
pub(crate) struct DeviceExpertLru {
    map: HashMap<(u32, u32), DevSlot>,
    order: VecDeque<((u32, u32), u64)>,
    ver: HashMap<(u32, u32), u64>,
    bytes: usize,
    budget_bytes: usize,
    per_gu: usize,
    per_dn: usize,
    /// 冷槽打包上传通道：host blob 先拷进 WC 锁页缓冲（满速写），
    /// 再异步 DMA 到 packed 冷区（每批仅 2 次 H2D），最后 dtod 落到各槽位。
    /// 替代逐槽 pageable H2D（实测 host 阻塞 ~1.5s/步的根源）。
    pin_gu: PinnedHostSlice<u16>,
    pin_dn: PinnedHostSlice<u16>,
    cold_gu: CudaSlice<bf16>,
    cold_dn: CudaSlice<bf16>,
    batch: usize,
}

impl DeviceExpertLru {
    pub(crate) fn new(k: &Kernels, gb: usize, inter: usize, hidden: usize) -> Result<Self> {
        let per_gu = 2 * inter * hidden;
        let per_dn = hidden * inter;
        // decode 每层最多 top_k=10 冷槽，一批放下；prefill 更多时分批
        let batch = 12;
        // u16：与 bf16 同宽同位形；用可缓存锁页（非 WC）：x86 cache coherent，
        // CPU memcpy 写入速度远高于 WC（WC 仅对 NT 指令写入友好）。
        // 缓冲随后被整体覆写，未初始化安全
        let pin_gu = unsafe {
            k.stream
                .context()
                .alloc_pinned_with_flags::<u16>(batch * per_gu, 0)
        }?;
        let pin_dn = unsafe {
            k.stream
                .context()
                .alloc_pinned_with_flags::<u16>(batch * per_dn, 0)
        }?;
        let cold_gu = unsafe { k.stream.alloc::<bf16>(batch * per_gu) }?;
        let cold_dn = unsafe { k.stream.alloc::<bf16>(batch * per_dn) }?;
        Ok(Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            ver: HashMap::new(),
            bytes: 0,
            budget_bytes: gb * 1024 * 1024 * 1024,
            per_gu,
            per_dn,
            pin_gu,
            pin_dn,
            cold_gu,
            cold_dn,
            batch,
        })
    }


    fn touch(&mut self, key: (u32, u32)) {
        let v = self.ver.entry(key).or_insert(0);
        *v += 1;
        self.order.push_back((key, *v));
        if self.order.len() > self.map.len() * 4 + 64 {
            let cur: std::collections::HashSet<((u32, u32), u64)> =
                self.ver.iter().map(|(k, v)| (*k, *v)).collect();
            self.order.retain(|e| cur.contains(e));
        }
    }

    fn insert(&mut self, key: (u32, u32), slot: DevSlot) {
        let len = (slot.gu.len() + slot.dn.len()) * std::mem::size_of::<bf16>();
        if len > self.budget_bytes {
            return; // 单槽超预算（理论不会）不缓存
        }
        if !self.map.contains_key(&key) {
            self.bytes += len;
        }
        self.touch(key);
        self.map.insert(key, slot);
        while self.bytes > self.budget_bytes {
            let Some((old, v)) = self.order.pop_front() else { break };
            if self.ver.get(&old) != Some(&v) {
                continue;
            }
            if let Some(b) = self.map.remove(&old) {
                self.ver.remove(&old);
                self.bytes -= (b.gu.len() + b.dn.len()) * std::mem::size_of::<bf16>();
            }
        }
    }

    /// 把 `experts` 的权重聚集进 scratch（gu/dn 容量由调用方保证 ≥ S 段），
    /// `blobs` 与 experts 对齐（device 未命中槽的 host 来源）。返回 device 命中数。
    pub(crate) fn fill(
        &mut self,
        k: &Kernels,
        layer: u32,
        experts: &[u32],
        blobs: &[Blob],
        scratch_gu: &mut CudaSlice<bf16>,
        scratch_dn: &mut CudaSlice<bf16>,
    ) -> Result<usize> {
        let pg = self.per_gu;
        let pd = self.per_dn;
        let t0 = std::time::Instant::now();
        let prof = lingxi_model::flag("LINGXI_PROF");

        // 分流 + 提升（先只判存在，拷贝循环里再取引用）
        let mut is_hot = vec![false; experts.len()];
        for (i, &e) in experts.iter().enumerate() {
            if self.map.contains_key(&(layer, e)) {
                is_hot[i] = true;
                self.touch((layer, e));
            }
        }
        let dev_hit = is_hot.iter().filter(|h| **h).count();

        let mut t_dtod = 0.0;
        let mut t_pin = 0.0;
        let mut t_h2d = 0.0;
        let mut t_sync = 0.0;
        let mut t_copy = 0.0;
        // device 热槽：HBM 内 dtod 直落 scratch
        for (i, &e) in experts.iter().enumerate() {
            if !is_hot[i] {
                continue;
            }
            let og = i * pg;
            let od = i * pd;
            let t = std::time::Instant::now();
            let slot = self.map.get(&(layer, e)).unwrap();
            k.stream
                .memcpy_dtod(&slot.gu, &mut scratch_gu.slice_mut(og..og + pg))?;
            k.stream
                .memcpy_dtod(&slot.dn, &mut scratch_dn.slice_mut(od..od + pd))?;
            t_dtod += t.elapsed().as_secs_f64();
        }

        // device 冷槽：分批走 WC pinned 打包通道
        // 每批：sync（保证 pinned 可复用）→ blob 拷入 pinned packed →
        // 2 次异步 H2D 到 packed 冷区 → dtod 落 scratch 槽位 + dtod 回填 LRU 新槽
        let cold: Vec<usize> = is_hot
            .iter()
            .enumerate()
            .filter_map(|(i, h)| (!h).then_some(i))
            .collect();
        for part in cold.chunks(self.batch) {
            // raw 指针按值跨线程传递的 Send 包装（各线程写互不重叠偏移，安全）
            #[derive(Clone, Copy)]
            struct SendPtr(*mut u16);
            unsafe impl Send for SendPtr {}
            impl SendPtr {
                #[inline]
                fn raw(self) -> *mut u16 {
                    self.0
                }
                #[inline]
                fn off(self, n: usize) -> *mut u16 {
                    unsafe { self.0.add(n) }
                }
            }
            let b = part.len();
            let ts = std::time::Instant::now();
            k.stream.synchronize()?;
            t_sync += ts.elapsed().as_secs_f64();

            let t = std::time::Instant::now();
            // 多线程并行填充 pinned（各写各的槽偏移，互不重叠）；
            // 源是 mmap 只读视图时，本步同时并行触发 major page fault（读盘重叠）。
            // pinned 借用只在本 scope 内存活，raw 指针按值分发给线程
            let copy_max = std::sync::atomic::AtomicU64::new(0);
            {
                let pgu = self.pin_gu.as_mut_slice()?;
                let pdn = self.pin_dn.as_mut_slice()?;
                let dst_gu = SendPtr(pgu.as_mut_ptr());
                let dst_dn = SendPtr(pdn.as_mut_ptr());
                std::thread::scope(|s| {
                    for (j, &i) in part.iter().enumerate() {
                        let blob = &blobs[i];
                        let src_gu = SendPtr(blob.gu(pg).as_ptr() as *mut u16);
                        let src_dn = SendPtr(blob.dn(pg, pd).as_ptr() as *mut u16);
                        let cm = &copy_max;
                        s.spawn(move || {
                            let tc = std::time::Instant::now();
                            // bf16/u16 同宽同位形；AVX 非临时写填充 pinned
                            unsafe {
                                copy_stream_nt(src_gu.raw(), dst_gu.off(j * pg), pg);
                                copy_stream_nt(src_dn.raw(), dst_dn.off(j * pd), pd);
                            }
                            use std::sync::atomic::Ordering;
                            let ns = tc.elapsed().as_nanos() as u64;
                            let mut prev = cm.load(Ordering::Relaxed);
                            while ns > prev {
                                match cm.compare_exchange_weak(
                                    prev,
                                    ns,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                ) {
                                    Ok(_) => break,
                                    Err(p) => prev = p,
                                }
                            }
                        });
                    }
                });
            }
            t_pin += t.elapsed().as_secs_f64();
            t_copy += copy_max.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;

            // raw driver 调用：dst 按字节定位，src 用 pinned 的 u16 切片（异步入队，不阻塞 host）
            let t = std::time::Instant::now();
            unsafe {
                let (dgu, _g) = self.cold_gu.device_ptr_mut(&k.stream);
                cudarc::driver::result::memcpy_htod_async(
                    dgu,
                    &self.pin_gu.as_slice()?[..b * pg],
                    k.stream.cu_stream(),
                )?;
                let (ddn, _d) = self.cold_dn.device_ptr_mut(&k.stream);
                cudarc::driver::result::memcpy_htod_async(
                    ddn,
                    &self.pin_dn.as_slice()?[..b * pd],
                    k.stream.cu_stream(),
                )?;
            }
            t_h2d += t.elapsed().as_secs_f64();

            let t = std::time::Instant::now();
            for (j, &i) in part.iter().enumerate() {
                let e = experts[i];
                let og = i * pg;
                let od = i * pd;
                k.stream.memcpy_dtod(
                    &self.cold_gu.slice(j * pg..(j + 1) * pg),
                    &mut scratch_gu.slice_mut(og..og + pg),
                )?;
                k.stream.memcpy_dtod(
                    &self.cold_dn.slice(j * pd..(j + 1) * pd),
                    &mut scratch_dn.slice_mut(od..od + pd),
                )?;
                // packed 冷区 → 新驻留槽（从 packed 拷，不依赖 scratch）
                // bf16 任意位形有效，分配后立即被 dtod 整体覆写
                let mut n_gu = unsafe { k.stream.alloc::<bf16>(pg) }?;
                let mut n_dn = unsafe { k.stream.alloc::<bf16>(pd) }?;
                k.stream.memcpy_dtod(
                    &self.cold_gu.slice(j * pg..(j + 1) * pg),
                    &mut n_gu,
                )?;
                k.stream.memcpy_dtod(
                    &self.cold_dn.slice(j * pd..(j + 1) * pd),
                    &mut n_dn,
                )?;
                self.insert((layer, e), DevSlot { gu: n_gu, dn: n_dn });
            }
            t_dtod += t.elapsed().as_secs_f64();
        }

        if prof {
            let mb = experts.len() * (pg + pd) * 2 / 1000_000;
            eprintln!(
                "    [experts] 层 {layer:2} 命中 {:2}（显存 {dev_hit:2}）{mb:6.1}MB  dtod {t_dtod:.3}s  scope {t_pin:.3}s(拷 {t_copy:.3}) sync {t_sync:.3}s  异步入队 {t_h2d:.3}s  共 {:.3}s  驻留 {} 槽",
                experts.len(),
                t0.elapsed().as_secs_f64(),
                self.map.len()
            );
        }
        Ok(dev_hit)
    }
}
