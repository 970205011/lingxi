//! QSA 核心：权重/形状类型 + 能力开关 + `QsaState` 跨 token 状态与生命周期。
//! 见 [crate::qsa] 的模块总览。

use anyhow::Result;
use super::paging::kv_init_tokens;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_kernels::Kernels;

pub struct AttentionWeights {
    /// `[13312, H]`：`q(6144) | k(512) | gate(6144) | v(512)`
    pub query_key_gate_value: CudaSlice<bf16>,
    /// 同上矩阵的 **W4G64** 版本（可选，decode t==1 优先用；四段用 `row_off` 切）
    pub query_key_gate_value_q4: Option<crate::q4::Q4W>,
    /// `[head_dim]`：每头 q 归一化 gamma
    pub query_norm: CudaSlice<bf16>,
    /// `[head_dim]`：每头 k 归一化 gamma
    pub key_norm: CudaSlice<bf16>,
    /// `[H, 6144]`：o_proj
    pub output: CudaSlice<bf16>,
    /// 同上矩阵的 **W4G64** 版本（可选）
    pub output_q4: Option<crate::q4::Q4W>,
}

/// 形状参数。
#[derive(Debug, Clone, Copy)]
pub struct AttnCfg {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
    pub eps: f32,
}

/// indexer 权重。
pub struct IndexerWeights {
    /// `[640, H]` = `q(4*128) | k(1*128)`
    pub index_qk_proj: CudaSlice<bf16>,
    /// `[128]` 池化后 key 的归一化 gamma
    pub q_layernorm: CudaSlice<bf16>,
    /// `[128]`
    pub k_layernorm: CudaSlice<bf16>,
}

/// indexer 形状参数。
#[derive(Debug, Clone, Copy)]
pub struct IndexerCfg {
    /// `indexer_n_heads`（4）
    pub heads: usize,
    /// `indexer_head_dim`（128）
    pub head_dim: usize,
    /// rope 维数（64）
    pub rot_dim: usize,
    /// `indexer_budget`（2048）
    pub budget: usize,
    /// `indexer_compress_ratio`（4）
    pub compress: usize,
    pub eps: f32,
}


/// ㊶ KV E4M3 量化开关（`LINGXI_KV_K8V4`，有损：K/V 各按 (行, kv头) 量到 FP8，
/// 每 token 每 QSA 层 KV 读流量 ÷2；256K 上下文 12 层 6.3GB → 3.15GB）。
/// 开关必须在**任何 QsaState 创建之前**决定（决定分配哪套缓冲）。
pub fn kv_q8_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::flag("LINGXI_KV_K8V4"))
}

/// **gather 稀疏注意力**：把选中块压成升序列表后只遍历列表（每行恒定 = budget token），
/// 不再逐 key 扫全量 Tk。默认开；`LINGXI_ATTN_GATHER=0` 回"原地跳"路径做 A/B。
pub fn attn_gather_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LINGXI_ATTN_GATHER").map(|v| v != "0").unwrap_or(true)
    })
}

/// indexer 选块走**基数选择**（O(nblocks)/行，无 scores 缓冲）而不是"逐 (行,块) 全排名"
/// （O(nblocks²)）。默认开；`LINGXI_QSA_KEEP_RADIX=0` 回旧路径做逐位 A/B。
pub fn keep_radix_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LINGXI_QSA_KEEP_RADIX").map(|v| v != "0").unwrap_or(true)
    })
}


pub struct QsaState {
    /// 归一化 + RoPE 后的 key `[cap, kv_heads*head_dim]`（bf16 档，`LINGXI_KV_K8V4=0`）
    pub kn: Option<CudaSlice<bf16>>,
    /// value `[cap, kv_heads*head_dim]`（同上）
    pub v: Option<CudaSlice<bf16>>,
    /// ㊶ E4M3 量化档（`LINGXI_KV_K8V4=1`，有损、env 门控）：字节 KV + 每 (行, kv头) scale。
    /// 量化后 **bf16 的 kn/v 为 None**（省一半显存：256K 上下文 12 层 6.3GB → 3.15GB）；
    /// `stub_kn/stub_v` 是给量化核包装器的 1 元素占位（内核 q8 分支从不读它们）。
    pub kn_q8: Option<CudaSlice<u8>>,
    pub kn_sc: Option<CudaSlice<f32>>,
    pub v_q8: Option<CudaSlice<u8>>,
    pub v_sc: Option<CudaSlice<f32>>,
    pub stub_kn: CudaSlice<bf16>,
    pub stub_v: CudaSlice<bf16>,
    /// ㊸ indexer 的 `keep`/`scores` 工作缓冲（grow-only 持久复用）：替代每 token 两次
    /// `alloc_zeros`——V100 实测 alloc+memset+free ≈500µs/次，12 个 QSA 层 × 2 ≈ decode
    /// 每 token ~10ms 纯 driver 开销（layerprof 实测 QSA 层 bubble ~800µs 的主因）。
    /// rank/scores 核**全量覆写**（rank 对每个 (row,b) 都写 1/0）⇒ 零值无语义，无需 memset。
    pub keep_buf: CudaSlice<u8>,
    pub scores_buf: CudaSlice<f32>,
    /// 基数选块的每行阈值（u32 位型）与"严格大于阈值的个数"（`LINGXI_QSA_KEEP_RADIX`）。
    pub thr_key_buf: CudaSlice<u32>,
    pub cnt_gt_buf: CudaSlice<i32>,
    /// gather 注意力的每行选中块号列表（升序）与个数（`LINGXI_ATTN_GATHER`）。
    pub sel_buf: CudaSlice<i32>,
    pub sel_cnt_buf: CudaSlice<i32>,
    /// decode（t==1）并行选块用的直方图（2048 bin）与 [bin, cum]。
    pub hist_buf: CudaSlice<u32>,
    pub bin_buf: CudaSlice<i32>,
    /// **P2 Stage 1**：KV 块表 `[逻辑块号] = 物理槽号`（间接粒度 = indexer 块 =
    /// `compress` token，与 gather 的 sel 单位 1:1）。Stage 1 恒等映射（零行为变化）；
    /// Stage 2 起冷块下沉 host 后表项指向 GPU 空闲槽，被选中时预取回填。
    /// 尺寸按上限 `max_cap` 一次建足（ensure_cap 增长不动它）。
    pub blk_tab: CudaSlice<i32>,
    /// **P2 Stage 2**：分页模式（`LINGXI_QSA_PAGED=1`，new 时定型）。kn/v 设备缓冲
    /// 只开 `win_blocks` 个槽（常驻窗口），逻辑上界仍 max_cap；冷块下沉锁页 host 池。
    pub paged: bool,
    /// GPU 常驻窗口（块数）。kn/v = win_blocks × ctok 行。
    pub win_blocks: usize,
    /// 空闲 GPU 槽栈（块号空间 [0, win_blocks)）。
    pub(crate) free_slots: Vec<i32>,
    /// host 锁页块池（懒倍增）：每槽 2×ctok×kv_dim bf16（前 kn 后 v），只有"已下沉"
    /// 块占坑；rewind 释放块时归还。pinned ⇒ 下沉/预取都是真 DMA。
    pub(crate) host_pool: Option<cudarc::driver::PinnedHostSlice<bf16>>,
    pub(crate) host_pool_slots: usize,
    /// 逻辑块 → host 槽（u32::MAX = 不在 host）。
    pub(crate) host_blk: Vec<u32>,
    pub(crate) host_free: Vec<u32>,
    /// 每逻辑块最近使用步（LRU 记账，步 = kv_admit 调用计数）。
    pub(crate) blk_touch: Vec<u64>,
    pub(crate) step_ctr: u64,
    /// 已下沉块计数：0 时 kv_admit 零开销快路（无 D2H/无同步——窗口未满的常见形态）。
    pub(crate) n_sunk: usize,
    /// kv_admit 的锁页暂存（cnt ‖ sel 一条 pinned 缓冲，一次 DMA + 一次同步）。
    pub(crate) sel_pin: Option<cudarc::driver::PinnedHostSlice<i32>>,
    /// blk_tab 的 host 镜像：-2 未分配 / -1 已下沉 / ≥0 GPU 槽号。脏时整体 H2D
    /// （表 ≤ 数十 KB，每步一次拷贝远小于选块流量）。驱动侧判定一律读本镜像。
    pub tab_host: Vec<i32>,
    pub(crate) tab_dirty: bool,
    /// indexer 的 raw（投影后、未 norm）key，host f32，长度 = `len * indexer_head_dim`。
    /// 池化本身在 host 上做，因此不必留在显存。
    pub raw_keys_host: Vec<f32>,
    /// indexer raw key 的**设备**缓存（bf16，`[cap, indexer_head_dim]`）
    pub rk_dev: CudaSlice<bf16>,
    /// 池化块（bf16，`[cap/compress, indexer_head_dim]`）
    pub pooled: CudaSlice<bf16>,
    /// 已完成的池化块数（= `len / compress`）
    pub nblocks: usize,
    /// 池化链路的小暂存（`[head_dim]` ×2），避免每条链多次 alloc
    pub pool_tmp: CudaSlice<bf16>,
    pub pool_tmp2: CudaSlice<bf16>,
    /// 已缓存 token 数
    pub len: usize,
    /// **当前已分配**的容量（≤ max_cap；㊿-P1 弹性 KV 起为懒分配，按需几何增长）。
    /// 设备缓冲恒为 `cap × 对应维度`（增长时全量拷贝）⇒ `kn.len/cap == kv_dim` 不变式保持。
    pub cap: usize,
    /// 容量上限（= 建局时的 rope 容量；RoPE 表只建到这么长，不能越界）
    pub(crate) max_cap: usize,
    /// `kv_heads * head_dim`（= 每行元素数；分页模式下 kn.len/cap 不再等于它，
    /// 一律以本字段为准）
    pub kv_dim: usize,
    /// `kv_heads`（K8V4 的 scale 平面增长用）
    pub(crate) nkh: usize,
    /// indexer 的 compress（回卷 `nblocks` 要用）
    pub idx_compress: usize,
    /// indexer 的 head_dim（回卷 `raw_keys_host` 要用；= rk_dev 每行元素数）
    pub idx_head_dim: usize,
}

impl QsaState {
    pub fn new(k: &Kernels, c: &AttnCfg, ci: &IndexerCfg, cap: usize) -> Result<Self> {
        let kv_dim = c.kv_heads * c.head_dim;
        let q8 = kv_q8_on();
        // P2 Stage 2：分页模式（host 块池 + 常驻窗口）。硬前提 = gather 路径前提
        // （raw KV 只有 gather 按块读；dense/masked 读全量行，与分页互斥）。
        let paged = crate::flag("LINGXI_QSA_PAGED") && !q8;
        if crate::flag("LINGXI_QSA_PAGED") && q8 {
            anyhow::bail!("LINGXI_QSA_PAGED 与 K8V4 量化 KV（LINGXI_KV_K8V4=1）互斥");
        }
        if paged {
            anyhow::ensure!(
                ci.compress.is_power_of_two() && ci.compress >= 1 && c.head_dim == 256,
                "LINGXI_QSA_PAGED 需要 head_dim=256 + 2 的幂 compress（gather 前提）"
            );
        }
        let ctok = ci.compress.max(1);
        let nblk_max = cap / ctok + 1;
        // 常驻窗口（块数）：默认 32K token；非分页模式槽空间 = 逻辑上界（旧行为）。
        let win_blocks = if paged {
            let win_toks: usize = std::env::var("LINGXI_QSA_WIN_TOKENS")
                .ok().and_then(|s| s.parse().ok())
                .filter(|&n| n >= 4 * ctok)
                .unwrap_or(32768)
                .min(cap)
                / ctok;
            win_toks.max(4)
        } else {
            nblk_max
        };
        // ㊿-P1 弹性 KV：只预付 `min(cap, 初始档)`，余量按需 `ensure_cap` 几何增长。
        // 上限仍是 `cap`（RoPE 表口径）；显式 env `LINGXI_KV_NO_LAZY=1` 回全额预付（对照用）。
        let init = if crate::flag("LINGXI_KV_NO_LAZY") { cap } else { cap.min(kv_init_tokens()) };
        let kn_init = if paged { win_blocks * ctok } else { init };
        Ok(Self {
            kn: if q8 { None } else { Some(k.stream.alloc_zeros::<bf16>(kn_init * kv_dim)?) },
            v: if q8 { None } else { Some(k.stream.alloc_zeros::<bf16>(kn_init * kv_dim)?) },
            kn_q8: if q8 { Some(k.stream.alloc_zeros::<u8>(init * kv_dim)?) } else { None },
            kn_sc: if q8 { Some(k.stream.alloc_zeros::<f32>(init * c.kv_heads)?) } else { None },
            v_q8: if q8 { Some(k.stream.alloc_zeros::<u8>(init * kv_dim)?) } else { None },
            v_sc: if q8 { Some(k.stream.alloc_zeros::<f32>(init * c.kv_heads)?) } else { None },
            stub_kn: k.stream.alloc_zeros::<bf16>(1)?,
            stub_v: k.stream.alloc_zeros::<bf16>(1)?,
            // 占位小尺寸，attention_step 里按需长大（长大是冷路径）
            keep_buf: k.stream.alloc_zeros::<u8>(1 << 16)?,
            scores_buf: k.stream.alloc_zeros::<f32>(1 << 16)?,
            thr_key_buf: k.stream.alloc_zeros::<u32>(1 << 12)?,
            cnt_gt_buf: k.stream.alloc_zeros::<i32>(1 << 12)?,
            sel_buf: k.stream.alloc_zeros::<i32>(1 << 12)?,
            sel_cnt_buf: k.stream.alloc_zeros::<i32>(1 << 12)?,
            hist_buf: k.stream.alloc_zeros::<u32>(2048)?,
            bin_buf: k.stream.alloc_zeros::<i32>(2)?,
            // P2：块表按上限建足。分页模式全 -2（未分配，随写随分配）；否则恒等。
            blk_tab: {
                let mut idt: Vec<i32> = Vec::with_capacity(nblk_max);
                if paged {
                    idt.extend(std::iter::repeat(-2i32).take(nblk_max));
                } else {
                    idt.extend(0..nblk_max as i32);
                }
                let mut t = k.stream.alloc_zeros::<i32>(nblk_max)?;
                {
                    use cudarc::driver::DevicePtrMut;
                    let (dp, _g) = t.device_ptr_mut(&k.stream);
                    unsafe {
                        cudarc::driver::result::memcpy_htod_async::<i32>(
                            dp, &idt[..], k.stream.cu_stream(),
                        )?
                    };
                }
                t
            },
            paged,
            win_blocks,
            free_slots: if paged {
                (0..win_blocks as i32).rev().collect()
            } else {
                Vec::new()
            },
            host_pool: None,
            host_pool_slots: 0,
            host_blk: vec![u32::MAX; nblk_max],
            host_free: Vec::new(),
            blk_touch: vec![0u64; nblk_max],
            step_ctr: 0,
            n_sunk: 0,
            sel_pin: None,
            tab_host: if paged {
                vec![-2i32; nblk_max]
            } else {
                (0..nblk_max as i32).collect()
            },
            tab_dirty: false,
            raw_keys_host: Vec::with_capacity(init * ci.head_dim),
            rk_dev: k.stream.alloc_zeros::<bf16>(init * ci.head_dim)?,
            pooled: k.stream.alloc_zeros::<bf16>((init / ci.compress + 1) * ci.head_dim)?,
            nblocks: 0,
            pool_tmp: k.stream.alloc_zeros::<bf16>(ci.head_dim)?,
            pool_tmp2: k.stream.alloc_zeros::<bf16>(ci.head_dim)?,
            len: 0,
            cap: init,
            max_cap: cap,
            kv_dim,
            nkh: c.kv_heads,
            idx_compress: ci.compress,
            idx_head_dim: ci.head_dim,
        })
    }

    /// ㊿-P1 弹性 KV：按需几何增长（alloc + 全量拷贝 + 换绑）。
    /// - append-only 语义 ⇒ 倍增策略的摊销拷贝 ≈ 2× 最终体积；
    /// - 只在同一条 stream 上排队，顺序由 stream 保证；所有 kernel 按**调用时**指针启动，
    /// 专家段图用的是独立固定槽、不引用本池 ⇒ 换缓冲对数学与图都不可见（逐位等价）；
    /// - `need > max_cap` 按旧口径报错（RoPE 表只建到 max_cap，越界会读到垃圾）。

    pub fn ensure_cap(&mut self, k: &Kernels, need: usize) -> Result<()> {
        if need <= self.cap {
            return Ok(());
        }
        anyhow::ensure!(
            need <= self.max_cap,
            "QsaState 容量不足: {need} > 上限 {}",
            self.max_cap
        );
        let mut new_cap = self.cap.max(1);
        while new_cap < need {
            new_cap *= 2;
        }
        new_cap = new_cap.min(self.max_cap);
        // P2 分页：kn/v 是固定窗口池，不随逻辑增长（冷块在 host 池）；只长逻辑缓冲。
        if self.paged {
            let hd = self.idx_head_dim;
            let cp = self.idx_compress;
            let mut b = k.stream.alloc_zeros::<bf16>(new_cap * hd)?;
            let old = std::mem::replace(&mut self.rk_dev, b);
            let (sp, _g1) = old.device_ptr(&k.stream);
            {
                let (dp, _g2) = self.rk_dev.device_ptr_mut(&k.stream);
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(dp, sp, old.len() * 2, k.stream.cu_stream())?
                };
            }
            b = k.stream.alloc_zeros::<bf16>((new_cap / cp + 1) * hd)?;
            let old = std::mem::replace(&mut self.pooled, b);
            let (sp, _g1) = old.device_ptr(&k.stream);
            {
                let (dp, _g2) = self.pooled.device_ptr_mut(&k.stream);
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(dp, sp, old.len() * 2, k.stream.cu_stream())?
                };
            }
            self.raw_keys_host.reserve(new_cap.saturating_sub(self.cap) * hd);
            self.cap = new_cap;
            // rk_dev/pooled 换绑 ⇒ 图地址失效，必须换 epoch（同非分页口径）。
            crate::graph::epoch_bump();
            return Ok(());
        }
        let kv_dim = self.kv_dim;
        let nkh = self.nkh;
        let hd = self.idx_head_dim;
        let cp = self.idx_compress;
        // 设备缓冲：旧内容按旧容量全量拷贝，追加区保持零初始化（零值无语义：注意力只读 [0,len)）。
        // Option 字段（kn/v 及 K8V4 字节档；q8 关闭时为 None）。
        macro_rules! grow_opt {
            ($field:ident, $ty:ty, $elems:expr) => {
                if let Some(old) = self.$field.take() {
                    let mut b = k.stream.alloc_zeros::<$ty>($elems)?;
                    let bytes = std::mem::size_of::<$ty>() * old.len();
                    let (sp, _g1) = old.device_ptr(&k.stream);
                    {
                        let (dp, _g2) = b.device_ptr_mut(&k.stream);
                        unsafe {
                            cudarc::driver::result::memcpy_dtod_async(
                                dp, sp, bytes, k.stream.cu_stream(),
                            )?
                        };
                    }
                    self.$field = Some(b);
                }
            };
        }
        grow_opt!(kn, bf16, new_cap * kv_dim);
        grow_opt!(v, bf16, new_cap * kv_dim);
        grow_opt!(kn_q8, u8, new_cap * kv_dim);
        grow_opt!(kn_sc, f32, new_cap * nkh);
        grow_opt!(v_q8, u8, new_cap * kv_dim);
        grow_opt!(v_sc, f32, new_cap * nkh);
        // 裸字段（恒存在）：rk_dev / pooled。
        macro_rules! grow_bare {
            ($field:ident, $ty:ty, $elems:expr) => {{
                let b = k.stream.alloc_zeros::<$ty>($elems)?;
                let old = std::mem::replace(&mut self.$field, b);
                let bytes = std::mem::size_of::<$ty>() * old.len();
                let (sp, _g1) = old.device_ptr(&k.stream);
                {
                    let (dp, _g2) = self.$field.device_ptr_mut(&k.stream);
                    unsafe {
                        cudarc::driver::result::memcpy_dtod_async(
                            dp, sp, bytes, k.stream.cu_stream(),
                        )?
                    };
                }
            }};
        }
        grow_bare!(rk_dev, bf16, new_cap * hd);
        grow_bare!(pooled, bf16, (new_cap / cp + 1) * hd);
        // host 侧 raw keys 按需 reserve（Vec 本身随 extend 自增，这里只省重复 realloc）。
        self.raw_keys_host.reserve(new_cap.saturating_sub(self.cap) * hd);
        self.cap = new_cap;
        // ⚠️ 换绑会让**已录进 CUDA Graph 的旧缓冲指针全部作废**（QSA/GR 图烘死的是旧 kn/v/
        // pooled 地址）。必须 bump 本线程的图 epoch：全部图按新地址重新录制，否则重放读到
        // 已释放内存——实测表现是 GR 里间歇性 CUDA_ERROR_INVALID_VALUE（海森堡 bug，
        // 布局/时序敏感，LINGXI_KV_NO_LAZY=1 全额预付即不复现）。
        crate::graph::epoch_bump();
        Ok(())
    }

    /// **回卷到 `new_len`**（投机解码用：一批草稿被部分接受时，把 KV/索引器状态退回接受长度）。
    ///
    /// 为什么便宜：KV 是 append-only，`kn/v/rk_dev` 里超过 `new_len` 的行**不会被再读**
    /// （注意力只读 `[0, len)`），所以只要**改计数**：
    /// - `len` → `new_len`
    /// - `nblocks` → `new_len / compress`（超过的池化块下次推进时会**重算**，见 `for b in state.nblocks..`）
    /// - `raw_keys_host` → 截断到 `new_len * head_dim`（只有 `LINGXI_QSA_HOST` 回退路径用它）

    pub fn rewind(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.len, "rewind 只能往回");
        // P2 分页：释放 [new_len, old_len) 覆盖的块——槽与 host 坑都归还。
        // 语义安全：append-only + rewind-only 恢复（被释放的区间只会被重写，不重读）。
        if self.paged {
            let cp = self.idx_compress;
            let nb_new = (new_len + cp - 1) / cp;
            let nb_old = (self.len + cp - 1) / cp;
            for b in nb_new..nb_old {
                match self.tab_host[b] {
                    s if s >= 0 => self.free_slots.push(s),
                    -1 => {
                        let h = self.host_blk[b];
                        if h != u32::MAX {
                            self.host_free.push(h);
                            self.host_blk[b] = u32::MAX;
                        }
                        self.n_sunk = self.n_sunk.saturating_sub(1);
                    }
                    _ => {}
                }
                self.tab_host[b] = -2;
            }
            self.tab_dirty = true;
        }
        self.len = new_len;
        self.nblocks = new_len / self.idx_compress;
        self.raw_keys_host.truncate(new_len * self.idx_head_dim);
    }

    // ───────────────────── P2 Stage 2：分页管理（仅 paged 模式） ─────────────────────

    /// 每块元素数：`ctok × kv_dim`。

    pub(crate) fn blk_elems(&self) -> usize {
        self.idx_compress * self.kv_dim
    }
}
