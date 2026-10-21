//! PLE（Per-Layer Embedding，**仅 layer 1**）。数学对齐 HF `Qwen4ExpTextPLELayer`：
//! ```text
//! emb            = ngram_embedding(ngram_hash(input_ids))     # [T, ple_embed_dim]
//! key_normed     = norm_key(key_proj(emb))                    # [T, hc, hidden]
//! value          = value_proj(emb)                            # [T, hidden]
//! query_normed   = norm_query(hidden_states)                  # [T, hc, hidden]
//! gate           = Σ_i key_normed*query_normed / sqrt(hidden) # [T, hc, 1]
//! gate          = sign(gate) * sqrt(clamp(|gate|, 1e-6))
//! gated_value    = sigmoid(gate) * value.unsqueeze(-2)        # [T, hc, hidden]
//! out            = gated_value + short_conv(norm_conv(gated_value))
//! ```
//! `short_conv` = 膨胀深度卷积（K=4, dilation=ngram_size=3）+ silu。
//!
//! ngram 哈希与 95GiB 表的查表在 host 上做（按需 `pread`，不整表进内存）；
//! 投影 / 归一化 / 卷积在 GPU 上做。

use std::fs::File;
use std::os::unix::fs::FileExt;

use anyhow::{Context, Result};
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

use crate::gpu::proj_t1;

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10007;

/// PLE 超参（来自 `text_config`）。
#[derive(Debug, Clone)]
pub struct PleCfg {
    pub hidden: usize,
    pub hc: usize,
    pub embed_dim: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    /// `config.ngram_vocab_size_base`（素数门表基准）
    pub ngram_vocab_base: u64,
    /// `config.vocab_size`（哈希乘子用；与 `ngram_vocab_base` 不同！）
    pub unigram_vocab: u64,
    pub divisible_by: u64,
    pub seed: u64,
    pub eos: i64,
    pub conv_kernel: usize,
    pub eps: f32,
    /// PLE 表文件（bf16，每行 `head_dim` 个元素）
    pub table_path: String,
}

/// 由 [`PleCfg`] 派生的常量（素数门表大小 / 哈希乘子）。
pub struct PleDerived {
    pub heads: usize,
    pub head_dim: usize,
    pub dilation: usize,
    pub head_sizes: Vec<i64>,
    pub head_offsets: Vec<i64>,
    pub multipliers: Vec<i64>,
    pub total_vocab: u64,
    pub padded_vocab: u64,
}

fn splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(SPLITMIX_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    v ^ (v >> 31)
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v % 2 == 0 {
        return v == 2;
    }
    let mut d = 3u64;
    while d * d <= v {
        if v % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// `start` 之后的第 `count` 个素数。
fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

impl PleDerived {
    pub fn new(cfg: &PleCfg, ple_layer_index: u64) -> Self {
        let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        let head_dim = cfg.embed_dim / heads;
        let mut head_sizes = Vec::with_capacity(heads);
        let mut head_offsets = Vec::with_capacity(heads);
        let mut total = 0u64;
        for h in 0..heads {
            let global = ple_layer_index * heads as u64 + h as u64;
            let size = nth_prime_after(cfg.ngram_vocab_base - 1, global as usize + 1);
            head_sizes.push(size as i64);
            head_offsets.push(total as i64);
            total += size;
        }
        let padded = total.div_ceil(cfg.divisible_by) * cfg.divisible_by;
        // _build_layer_multipliers：int64 溢出语义（wrapping）
        let max_long = (1u64 << 63) - 1;
        let multiplier_max = max_long / cfg.unigram_vocab.max(1);
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = cfg.seed.wrapping_add(PRIME_1.wrapping_mul(ple_layer_index));
        let mut multipliers = Vec::with_capacity(cfg.ngram_size);
        for index in 0..cfg.ngram_size {
            let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
            multipliers.push((2 * (splitmix64(value) % half_bound) + 1) as i64);
        }
        Self {
            heads,
            head_dim,
            dilation: cfg.ngram_size,
            head_sizes,
            head_offsets,
            multipliers,
            total_vocab: total,
            padded_vocab: padded,
        }
    }
}

/// 95GiB PLE 表的按需读句柄（**带行缓存**）。
///
/// 为什么必须有缓存：表在机械盘上，每 token 要按 ngram 哈希**随机**取
/// `heads` 行（本模型 `heads=120`）。95 GiB 的随机读基本每次都是寻道，
/// 实测单 token 因此抖动到 0.15 s（`docs/decode-speed-profile.md` §3.10）。
/// 但同一段对话里同一批 token 的 ngram 行会**反复命中**（相邻 token 的
/// 2~4-gram 窗口只平移一格），所以进程内缓存一次就够。
///
/// 缓存以**字节上限**控制（`LINGXI_PLE_CACHE_MB`，默认 1024 MB，0 = 关闭），
/// 满了按 FIFO 淘汰。缓存与否**不改变数值**（只省 I/O）。
pub struct PleTable {
    file: File,
    row_bytes: usize,
    /// FP8(E4M3) 模式：`Some(scale)` ⇒ 行是 row_elems 字节的 e4m3 索引，读时查表 ×scale。
    /// 来源 = 官方 NVFP4 版模型自带的 PLE 表（51.2GB，bf16 母表的官方发布格式，
    /// 参考实现 ninfer-dev `ple_fp8_e4m3_dequant_row` 同款语义）。sidecar `<path>.scale` 存在即启用。
    fp8_scale: Option<f32>,
    /// IQ4_NL 模式（官方默认运行时格式，26.8GiB）：行 = row_elems/32 个 18B block
    /// （2B fp16 scale + 16B packed nibbles，split-half：byte j 低半 = 元素 j、高半 = 元素 j+16）。
    /// 解码语义与 ninfer-dev `ple_iq4nl_dequant_row` 逐项一致。sidecar `<path>.iq4nl` 存在即启用。
    iq4nl: bool,
    cache: std::cell::RefCell<RowCache>,
}

/// llama.cpp IQ4_NL 码本（与 ninfer-dev `ple_iq4nl_codebook` 一致）。
const IQ4NL_CB: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// 路径是否在 tmpfs（如 /dev/shm）上——解析 /proc/self/mounts 找最长前缀匹配。
/// tmpfs 上的表预热无意义（本来就在内存），还白抢内存带宽。
fn path_on_tmpfs(path: &str) -> bool {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
    let mounts = std::fs::read_to_string("/proc/self/mounts").unwrap_or_default();
    let mut best: Option<usize> = None;
    for line in mounts.lines() {
        let mut it = line.split_whitespace();
        let (Some(_dev), Some(mp), Some(fstype)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        if fstype != "tmpfs" {
            continue;
        }
        let mp_path = std::path::Path::new(mp);
        if canon.starts_with(mp_path) {
            let n = mp.len();
            if best.map_or(true, |b| n > b) {
                best = Some(n);
            }
        }
    }
    best.is_some()
}

/// e4m3fn 解码表（fn 变体：无无穷，NaN 读 0）——与 ninfer-dev `ple_fp8_e4m3_table` 逐项一致。
fn fp8_e4m3_table() -> &'static [f32; 256] {
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut v = [0f32; 256];
        for x in 0..256usize {
            let s = x >> 7;
            let e = (x >> 3) & 15;
            let m = x & 7;
            let f = if e == 15 && m == 7 {
                0.0
            } else if e == 0 {
                (m as f32 / 8.0) * 2f32.powi(-6)
            } else {
                (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
            };
            v[x] = if s != 0 { -f } else { f };
        }
        v
    })
}

struct RowCache {
    map: std::collections::HashMap<u64, Box<[f32]>>,
    order: std::collections::VecDeque<u64>,
    bytes: usize,
    cap_bytes: usize,
}

impl PleTable {
    pub fn open(path: &str, row_elems: usize) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("打不开 PLE 表 {path}"))?;
        let mb: usize = std::env::var("LINGXI_PLE_CACHE_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        // ★ P15-㊳ 后台顺序预热（默认开，`LINGXI_PLE_PREWARM=0` 关）：
        // 表 102GB 全在机械盘，推理是**随机行读**（冷行 = 寻道，实测单层事件间隔
        // 被拉到 6.5s、17K prefill 从 11.3ms/行 劣化到 60ms/行）。与其让首段语料
        // 替我们随机暖页，不如开一个后台线程**顺序**通读全表喂 page cache——
        // 顺序读 ~200MB/s（HDD RAID）约 9 分钟，与模型加载/早期 chunk 天然重叠；
        // 暖好的页是 host 全局的，后续进程同样受益。纯 I/O，不改任何数值。
        if std::env::var_os("LINGXI_PLE_PREWARM").map(|v| v != "0").unwrap_or(true)
            && !path_on_tmpfs(path)
        {
            if let Ok(f) = file.try_clone() {
                std::thread::spawn(move || {
                    use std::os::unix::fs::FileExt;
                    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
                    let mut buf = vec![0u8; 8 * 1024 * 1024];
                    let mut off = 0u64;
                    let t0 = std::time::Instant::now();
                    while off < len {
                        if f.read_at(&mut buf, off).is_err() {
                            break;
                        }
                        off += buf.len() as u64;
                    }
                    eprintln!(
                        "[ple-prewarm] 顺序预热 {len} 字节完成，用时 {:.1}s",
                        t0.elapsed().as_secs_f64()
                    );
                });
            }
        }
        // sidecar 检测：`.scale` ⇒ 官方 FP8 表；`.iq4nl` ⇒ 官方 IQ4_NL 表；否则 bf16。
        let fp8_scale = std::fs::read(format!("{path}.scale"))
            .ok()
            .and_then(|b| (b.len() == 4).then(|| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        let iq4nl = fp8_scale.is_none() && std::path::Path::new(&format!("{path}.iq4nl")).exists();
        match (fp8_scale, iq4nl) {
            (Some(s), _) => eprintln!(
                "[ple] FP8(E4M3) 表模式：行 {} B × scale {s:?}（官方 NVFP4 发布表）",
                row_elems
            ),
            (None, true) => eprintln!(
                "[ple] IQ4_NL 表模式：行 {} B（官方默认运行时格式，26.8GiB 档）",
                row_elems / 32 * 18
            ),
            _ => {}
        }
        Ok(Self {
            file,
            row_bytes: if fp8_scale.is_some() {
                row_elems
            } else if iq4nl {
                row_elems / 32 * 18
            } else {
                row_elems * 2
            },
            fp8_scale,
            iq4nl,
            cache: std::cell::RefCell::new(RowCache {
                map: std::collections::HashMap::new(),
                order: std::collections::VecDeque::new(),
                bytes: 0,
                cap_bytes: mb * 1024 * 1024,
            }),
        })
    }

    /// 取一行（`head_dim` 个 f32）。命中缓存则不碰磁盘。
    fn read_row(&self, id: u64, out: &mut [f32]) -> Result<()> {
        let mut c = self.cache.borrow_mut();
        if let Some(v) = c.map.get(&id) {
            out.copy_from_slice(v);
            return Ok(());
        }
        let mut buf = vec![0u8; self.row_bytes];
        self.file.read_exact_at(&mut buf, id * self.row_bytes as u64)?;
        if let Some(scale) = self.fp8_scale {
            let t = fp8_e4m3_table();
            for (i, &b) in buf.iter().enumerate() {
                out[i] = t[b as usize] * scale;
            }
        } else if self.iq4nl {
            // 与 ninfer-dev ple_iq4nl_dequant_row 逐项一致：每 18B block = f16 scale + 16B nibbles。
            let mut o = 0usize;
            for blk in buf.chunks_exact(18) {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for j in 0..16 {
                    let lo = (blk[2 + j] & 0x0F) as usize;
                    let hi = (blk[2 + j] >> 4) as usize;
                    out[o + j] = d * IQ4NL_CB[lo] as f32;
                    out[o + j + 16] = d * IQ4NL_CB[hi] as f32;
                }
                o += 32;
            }
        } else {
            for (i, ch) in buf.chunks_exact(2).enumerate() {
                out[i] = bf16::from_bits(u16::from_le_bytes([ch[0], ch[1]])).to_f32();
            }
        }
        if c.cap_bytes > 0 {
            let row_bytes = self.row_bytes;
            if c.bytes + row_bytes > c.cap_bytes {
                while c.bytes + row_bytes > c.cap_bytes {
                    match c.order.pop_front() {
                        Some(old) => {
                            if let Some(v) = c.map.remove(&old) {
                                c.bytes -= v.len() * 4;
                            }
                        }
                        None => break,
                    }
                }
            }
            c.map.insert(id, out.to_vec().into_boxed_slice());
            c.order.push_back(id);
            c.bytes += row_bytes;
        }
        Ok(())
    }
}

/// `_shift_right_ignore_eos`（HF 原样搬）。
fn shift_right_ignore_eos(tokens: &[i64], shift: usize, eos: i64) -> Vec<i64> {
    let n = tokens.len();
    let mut last_eos = -1i64;
    let mut prev_eos = vec![-1i64; n];
    for i in 0..n {
        prev_eos[i] = last_eos;
        if tokens[i] == eos {
            last_eos = i as i64;
        }
    }
    let mut out = vec![eos; n];
    for i in 0..n {
        let pos = i as i64;
        let segment_start = prev_eos[i] + 1;
        let position_in_segment = pos - segment_start;
        let source = pos - shift as i64;
        if position_in_segment >= shift as i64 && source >= 0 {
            out[i] = tokens[source as usize];
        }
    }
    out
}

/// ngram 哈希 → `[t, heads]` 的门表行号（host），**只算末尾 `t` 个位置**。
///
/// ⚠️ 这里曾经对**整段历史**每个位置都算一遍（`O(L·heads·ngram)`），于是
/// decode 的 host 时间随上下文长度线性增长（实测 300 token 上下文时单项
/// 就 ~5 ms/token，见 §3.10）。调用方只要末尾 `t` 行，所以内层循环收窄到
/// `t` 个位置即可 —— 数值完全不变（每个位置的行号只依赖它自己的局部窗口）。
pub fn ngram_ids(cfg: &PleCfg, d: &PleDerived, ids: &[i64], t: usize) -> Vec<i64> {
    assert!(t <= ids.len(), "ngram_ids: t={t} > ids.len()={}", ids.len());
    let context_len = cfg.ngram_size - 1;
    let mut history = vec![cfg.eos; context_len];
    history.extend_from_slice(ids);
    let shifted: Vec<Vec<i64>> = (0..cfg.ngram_size)
        .map(|s| shift_right_ignore_eos(&history, s, cfg.eos))
        .collect();

    let l = history.len();
    let start = l - t;
    let mut out = vec![0i64; t * d.heads];
    let mut mixed = vec![0i64; t];
    for head in 0..d.heads {
        // 该 head 属于哪个 ngram（2..=ngram_size）
        let ngram = 2 + head / cfg.heads_per_ngram;
        for pos in 0..ngram {
            let m = d.multipliers[pos];
            for k in 0..t {
                let term = shifted[pos][start + k].wrapping_mul(m);
                mixed[k] = if pos == 0 { term } else { mixed[k] ^ term };
            }
        }
        let size = d.head_sizes[head];
        let off = d.head_offsets[head];
        for k in 0..t {
            out[k * d.heads + head] = mixed[k].rem_euclid(size) + off;
        }
    }
    out
}

/// PLE 的权重与派生状态。
pub struct PleWeights {
    pub key_proj: CudaSlice<bf16>,
    pub value_proj: CudaSlice<bf16>,
    pub norm_key: CudaSlice<bf16>,
    pub norm_query: CudaSlice<bf16>,
    pub norm_conv: CudaSlice<bf16>,
    /// `[K, hc*hidden]`（已由产物的 `[C, K]` 转置而来）
    pub convolution: CudaSlice<bf16>,
    pub cfg: PleCfg,
    pub derived: PleDerived,
    pub table: PleTable,
}

/// `x[T, hc*H] → out[T, hc*H]`（PLE 的加性注入项）。
pub fn ple_forward(
    k: &Kernels,
    w: &PleWeights,
    x: &CudaSlice<bf16>,
    ids: &[i64],
    t: usize,
) -> Result<CudaSlice<bf16>> {
    let cfg = &w.cfg;
    let d = &w.derived;
    let hidden = cfg.hidden;
    let dim = cfg.hc * hidden;

    // 1) ngram 表查表 → emb [T, embed_dim]（host）
    let rows = ngram_ids(cfg, d, ids, t);
    let mut emb = vec![0f32; t * cfg.embed_dim];
    let mut rowbuf = vec![0f32; d.head_dim];
    for i in 0..t {
        for h in 0..d.heads {
            let id = rows[i * d.heads + h] as u64;
            w.table.read_row(id, &mut rowbuf)?;
            emb[i * cfg.embed_dim + h * d.head_dim..i * cfg.embed_dim + (h + 1) * d.head_dim]
                .copy_from_slice(&rowbuf);
        }
    }
    let emb_bf: Vec<bf16> = emb.iter().map(|v| bf16::from_f32(*v)).collect();
    let demb = k.stream.clone_htod(&emb_bf)?;

    // 2) key = norm_key(key_proj(emb))；value = value_proj(emb)
    let mut key = k.stream.alloc_zeros::<bf16>(t * dim)?;
    proj_t1(k, &demb, &w.key_proj, &mut key, dim, cfg.embed_dim, t)?;
    let mut key_n = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.grouped_rmsnorm_bf16(&key, &w.norm_key, &mut key_n, t as u32, dim, hidden, cfg.eps)?;
    let mut value = k.stream.alloc_zeros::<bf16>(t * hidden)?;
    proj_t1(k, &demb, &w.value_proj, &mut value, hidden, cfg.embed_dim, t)?;

    // 3) query_normed = norm_query(x)
    let mut query_n = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.grouped_rmsnorm_bf16(x, &w.norm_query, &mut query_n, t as u32, dim, hidden, cfg.eps)?;

    // 4) gate（host）：gate[t,g] = Σ_i key_n·query_n / sqrt(hidden) → 逐段处理
    let key_h: Vec<f32> = k.stream.clone_dtoh(&key_n)?.iter().map(|v| v.to_f32()).collect();
    let query_h: Vec<f32> = k.stream.clone_dtoh(&query_n)?.iter().map(|v| v.to_f32()).collect();
    let value_h: Vec<f32> = k.stream.clone_dtoh(&value)?.iter().map(|v| v.to_f32()).collect();
    let inv = 1.0f32 / (hidden as f32).sqrt();
    let mut gated = vec![0f32; t * dim];
    for i in 0..t {
        for g in 0..cfg.hc {
            let base = i * dim + g * hidden;
            let mut dot = 0f32;
            for j in 0..hidden {
                dot += key_h[base + j] * query_h[base + j];
            }
            let mut gate = dot * inv;
            // sign(gate) * sqrt(clamp(|gate|, 1e-6))
            let s = if gate > 0.0 {
                1.0
            } else if gate < 0.0 {
                -1.0
            } else {
                0.0
            };
            gate = s * gate.abs().clamp(1e-6, f32::INFINITY).sqrt();
            let sg = 1.0f32 / (1.0 + (-gate).exp());
            for j in 0..hidden {
                gated[base + j] = sg * value_h[i * hidden + j];
            }
        }
    }
    let gated_bf: Vec<bf16> = gated.iter().map(|v| bf16::from_f32(*v)).collect();
    let dgated = k.stream.clone_htod(&gated_bf)?;

    // 5) norm_conv + 膨胀短卷积 + silu
    let mut gated_n = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.grouped_rmsnorm_bf16(&dgated, &w.norm_conv, &mut gated_n, t as u32, dim, hidden, cfg.eps)?;
    let mut conv = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.causal_conv1d_silu_bf16(&gated_n, &w.convolution, &mut conv, t, dim, cfg.conv_kernel, d.dilation)?;

    // 6) out = gated_value + short_conv(...)
    let mut out = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.add_bf16(&dgated, &conv, &mut out, t * dim)?;
    Ok(out)
}

// ───────────────────────────── 增量（driver 用） ─────────────────────────────

/// PLE 层的**跨 token 状态**：膨胀短卷积的输入历史。
///
/// short conv 是 `K=4, D=ngram_size=3` 的因果膨胀卷积，新 token 要回看
/// 位置 -3 / -6 / -9，因此需保留最近 `(K-1)*D = 9` 行**卷积前**的
/// `norm_conv(gated_value)`。初值全 0，等价于整段前向"越界视为 0"的边界。
pub struct PleState {
    /// 最近 `(K-1)*D` 行卷积输入 `[hist, hc*hidden]`（bf16），按时间顺序。
    pub hist: CudaSlice<bf16>,
    /// 双缓冲（P15-㊵）：融合卷积核「读 hist → 写 new_hist」跨 block 写后读竞态，
    /// 必须换缓冲再 swap（与 GDN 的 `k_hist/k_hist_next` 同范式）。
    pub hist_next: CudaSlice<bf16>,
    pub hist_len: usize,
}

impl PleState {
    /// `dim = hc*hidden`；`conv_kernel`/`dilation` 决定历史行数 `(K-1)*D`。
    pub fn new(k: &Kernels, dim: usize, conv_kernel: usize, dilation: usize) -> Result<Self> {
        let hist_len = (conv_kernel - 1) * dilation;
        Ok(Self {
            hist: k.stream.alloc_zeros::<bf16>(hist_len * dim)?,
            hist_next: k.stream.alloc_zeros::<bf16>(hist_len * dim)?,
            hist_len,
        })
    }

    /// 从 PLE 权重推导形状的便捷构造。
    pub fn for_weights(k: &Kernels, w: &PleWeights) -> Result<Self> {
        Self::new(k, w.cfg.hc * w.cfg.hidden, w.cfg.conv_kernel, w.derived.dilation)
    }
}

/// 增量步进：`x[t, hc*H] → out[t, hc*H]`，`state` 就地推进 t 个 token。
///
/// 与 [`ple_forward`] 的关系：门控/投影逐 token 独立，唯一跨 token 的是膨胀卷积；
/// 这里用 `cin = [hist((K-1)*D 行); gated_n(t 行)]` 重建卷积上下文，跑完只取末尾
/// t 行，再把 `cin` 末尾 `(K-1)*D` 行写回历史。因此「整段一次 t 行（空历史）」与
/// 「逐 token 步进」应逐位一致。
///
/// `ids_all` 是截至当前的**全量** token id（长度 ≥ t；末尾 t 个对应本批新 token），
/// ngram 查表需要完整历史。
pub fn ple_step(
    k: &Kernels,
    w: &PleWeights,
    x: &CudaSlice<bf16>,
    ids_all: &[i64],
    t: usize,
    state: &mut PleState,
) -> Result<CudaSlice<bf16>> {
    let cfg = &w.cfg;
    let d = &w.derived;
    let hidden = cfg.hidden;
    let dim = cfg.hc * hidden;
    assert!(ids_all.len() >= t, "ple_step: ids 长度 {} < t={t}", ids_all.len());

    // 1) ngram 表查表 → emb [T, embed_dim]（host）；只算末尾 t 个位置
    let t_ple_host = std::time::Instant::now();
    let rows_all = ngram_ids(cfg, d, ids_all, t);
    let mut emb = vec![0f32; t * cfg.embed_dim];
    let mut rowbuf = vec![0f32; d.head_dim];
    for i in 0..t {
        for h in 0..d.heads {
            let id = rows_all[i * d.heads + h] as u64;
            w.table.read_row(id, &mut rowbuf)?;
            emb[i * cfg.embed_dim + h * d.head_dim..i * cfg.embed_dim + (h + 1) * d.head_dim]
                .copy_from_slice(&rowbuf);
        }
    }
    let t_ple_read = t_ple_host.elapsed();
    let emb_bf: Vec<bf16> = emb.iter().map(|v| bf16::from_f32(*v)).collect();
    let demb = k.stream.clone_htod(&emb_bf)?;
    if crate::flag("LINGXI_PLE_PROF") {
        eprintln!(
            "[ple-prof] t={t} rows={} 查表+反量化 {:.1}ms  f32→bf16+H2D {:.1}ms",
            t * d.heads,
            t_ple_read.as_secs_f64() * 1e3,
            (t_ple_host.elapsed() - t_ple_read).as_secs_f64() * 1e3
        );
    }

    // 2) key/value 投影
    let mut key = k.stream.alloc_zeros::<bf16>(t * dim)?;
    proj_t1(k, &demb, &w.key_proj, &mut key, dim, cfg.embed_dim, t)?;
    let mut key_n = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.grouped_rmsnorm_bf16(&key, &w.norm_key, &mut key_n, t as u32, dim, hidden, cfg.eps)?;
    let mut value = k.stream.alloc_zeros::<bf16>(t * hidden)?;
    proj_t1(k, &demb, &w.value_proj, &mut value, hidden, cfg.embed_dim, t)?;

    // 3) query_normed = norm_query(x)
    let mut query_n = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.grouped_rmsnorm_bf16(x, &w.norm_query, &mut query_n, t as u32, dim, hidden, cfg.eps)?;

    // 4) gate：**GPU kernel**（原本是 host 参考实现：每层 3 次 D2H + 1 次 H2D，
    //    18 层 PLE × 每 token = 54 次硬同步，是 decode 的主要停顿来源之一）
    let inv = 1.0f32 / (hidden as f32).sqrt();
    let mut dgated = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.ple_gate_bf16(&key_n, &query_n, &value, &mut dgated, t, cfg.hc, hidden, inv)?;

    // 5) norm_conv
    let mut gated_n = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.grouped_rmsnorm_bf16(&dgated, &w.norm_conv, &mut gated_n, t as u32, dim, hidden, cfg.eps)?;

    // 6) 膨胀短卷积：**融合版（一次 launch，P15-㊵）**。
    //    原链「hist→cin 拷 / gated_n→cin 拷 / conv 全量 / cin 尾→hist 拷 / cout 尾→conv 拷」= 5 发
    //    + 2 次 (hl+t)×dim 缓冲分配；现在直接读 (hist, gated_n)、只算 t 行、
    //    历史推进写 hist_next 后 swap（逐位语义与 GDN 已验证的融合核相同：k 升序、silu 同式）。
    let hl = state.hist_len;
    let mut conv = k.stream.alloc_zeros::<bf16>(t * dim)?;
    {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let (hp, _g0) = state.hist.device_ptr(&k.stream);
        let (qp, _g1) = gated_n.device_ptr(&k.stream);
        let (wp, _g2) = w.convolution.device_ptr(&k.stream);
        let (yp, _g3) = conv.device_ptr_mut(&k.stream);
        let (np, _g4) = state.hist_next.device_ptr_mut(&k.stream);
        k.causal_conv1d_silu_hist_bf16(
            hp, qp, wp, yp, np, hl, t, dim, cfg.conv_kernel, d.dilation,
        )?;
    }
    std::mem::swap(&mut state.hist, &mut state.hist_next);

    // 7) out = gated_value + short_conv(...)
    let mut out = k.stream.alloc_zeros::<bf16>(t * dim)?;
    k.add_bf16(&dgated, &conv, &mut out, t * dim)?;
    Ok(out)
}
