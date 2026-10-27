//! GDN（Gated DeltaNet 线性注意力，36 层）。数学对齐 HF `Qwen4ExpTextGatedDeltaNet`
//! 的 **recurrent** 形式（`torch_recurrent_gated_delta_rule`，单序列下与 chunk 形式等价）：
//! `text
//! qkv = in_proj_qkv(x); z = in_proj_z(x); a = in_proj_a(x); b = in_proj_b(x)
//! qkv = silu(causal_conv1d(qkv, K=4)); q,k,v = split(qkv)
//! q,k = l2norm(q,k); q /= sqrt(head_k_dim)
//! beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
//! S = 0; 逐 token: S *= exp(g_t); delta = (v_t - k_t^T S) * beta_t
//! S += k_t ⊗ delta; o_t = q_t^T S
//! out = out_proj(RMSNormGated(o, z))
//! `

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_ids::GateActivation;
use lingxi_kernels::Kernels;

use crate::gpu::proj_t1;
use crate::scratch;

/// `LINGXI_KT=1`：开启引擎内单 kernel GPU 计时（诊断用）。
fn kt_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LINGXI_KT").as_deref() == Ok("1"))
}

pub struct GdnWeights {
    /// `[16384, H]` = `q(2048) | k(2048) | v(6144) | z(6144)`
    pub query_key_value_z: CudaSlice<bf16>,
    /// 同上矩阵的 **W4G64** 版本（可选）。decode（t==1）优先用它 ——
    /// `query_key_value_z` 是每层最大的稠密矩阵（84 MB），纯带宽受限（见 `crate::q4`）。
    pub query_key_value_z_q4: Option<crate::q4::Q4W>,
    /// `[96, H]` = `a(48) | b(48)`
    pub a_b_projection: CudaSlice<bf16>,
    /// `[K, conv_dim]` 深度卷积核
    pub convolution: CudaSlice<bf16>,
    /// `[head_v_dim]` RMSNormGated 的 weight（直接乘，非 1+w）
    pub norm: CudaSlice<bf16>,
    /// `[H, value_dim]` out_proj
    pub output: CudaSlice<bf16>,
    /// 同上矩阵的 **W4G64** 版本（可选，decode t==1 优先用）
    pub output_q4: Option<crate::q4::Q4W>,
    /// `[num_v_heads]`（fp32 原样）
    pub a_log: Vec<f32>,
    /// `[num_v_heads]`
    pub dt_bias: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct GdnCfg {
    pub hidden: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel: usize,
    /// `RMSNormGated` 的 gate 激活（模型 config `output_gate_type`）
    pub gate_activation: GateActivation,
}

fn softplus(x: f32) -> f32 {
    // 与 torch.nn.functional.softplus 一致（beta=1, threshold=20）
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn norm_l2(v: &[f32]) -> Vec<f32> {
    let s: f32 = v.iter().map(|x| x * x).sum();
    let r = 1.0 / (s + 1e-6).sqrt();
    v.iter().map(|x| x * r).collect()
}

/// `recurrent_core` 的完整输出（含调试量，便于与既有诊断脚本对齐）。
struct Recurrent {
    out: Vec<f32>,
    gated: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    beta: Vec<f32>,
    g: Vec<f32>,
}

/// GDN 的 host 侧 recurrent 核心 —— `gdn_forward`（整段）与 `gdn_step`（增量）共用的**唯一真源**。
///
/// `s_all`：`[num_v_heads, head_k_dim*head_v_dim]` 扁平状态，**跨调用保留**（整段前向传全 0）。
/// `conv_h`：`[t, conv_dim]` 已过 `silu(causal_conv1d(...))` 的值（增量时只含新 token）。
#[allow(clippy::too_many_arguments)]
fn recurrent_core(
    conv_h: &[f32],
    z_h: &[f32],
    a_h: &[f32],
    b_h: &[f32],
    norm_w: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    c: &GdnCfg,
    t: usize,
    s_all: &mut [f32],
) -> Recurrent {
    let hk = c.head_k_dim;
    let hv = c.head_v_dim;
    let key_dim = c.num_k_heads * hk;
    let value_dim = c.num_v_heads * hv;
    let conv_dim = 2 * key_dim + value_dim;
    let rep = c.num_v_heads / c.num_k_heads;

    // q/k 先按 head l2norm、q 再除以 sqrt(head_k_dim)
    let mut qn = vec![0f32; t * c.num_v_heads * hk];
    let mut kn = vec![0f32; t * c.num_v_heads * hk];
    for i in 0..t {
        for hh in 0..c.num_k_heads {
            // conv 布局：q 段偏移 0，k 段偏移 key_dim
            let q_src = &conv_h[i * conv_dim + hh * hk..i * conv_dim + (hh + 1) * hk];
            let k_src =
                &conv_h[i * conv_dim + key_dim + hh * hk..i * conv_dim + key_dim + (hh + 1) * hk];
            let q_n = norm_l2(q_src);
            let k_n = norm_l2(k_src);
            for r in 0..rep {
                let dst = hh * rep + r;
                for d in 0..hk {
                    qn[(i * c.num_v_heads + dst) * hk + d] = q_n[d] / (hk as f32).sqrt();
                    kn[(i * c.num_v_heads + dst) * hk + d] = k_n[d];
                }
            }
        }
    }

    let mut out = vec![0f32; t * c.num_v_heads * hv];
    let mut beta_dbg = vec![0f32; t * c.num_v_heads];
    let mut g_dbg = vec![0f32; t * c.num_v_heads];
    for vh in 0..c.num_v_heads {
        let s = &mut s_all[vh * hk * hv..(vh + 1) * hk * hv];
        for i in 0..t {
            let decay = (-(a_log[vh].exp()) as f32 * softplus(a_h[i * c.num_v_heads + vh] + dt_bias[vh])).exp();
            let beta = 1.0 / (1.0 + (-b_h[i * c.num_v_heads + vh]).exp());
            beta_dbg[i * c.num_v_heads + vh] = beta;
            g_dbg[i * c.num_v_heads + vh] = decay;
            for x in s.iter_mut() {
                *x *= decay;
            }
            // kv_mem[dv] = Σ_dk S[dk][dv] * k[i][dk]
            let mut kv_mem = vec![0f32; hv];
            for dk in 0..hk {
                let kv = kn[(i * c.num_v_heads + vh) * hk + dk];
                for dv in 0..hv {
                    kv_mem[dv] += s[dk * hv + dv] * kv;
                }
            }
            // delta = (v - kv_mem) * beta; S += k ⊗ delta
            for dv in 0..hv {
                let v = conv_h[i * conv_dim + 2 * key_dim + vh * hv + dv];
                let delta = (v - kv_mem[dv]) * beta;
                for dk in 0..hk {
                    s[dk * hv + dv] += kn[(i * c.num_v_heads + vh) * hk + dk] * delta;
                }
            }
            // o = q^T S
            for dv in 0..hv {
                let mut acc = 0f32;
                for dk in 0..hk {
                    acc += s[dk * hv + dv] * qn[(i * c.num_v_heads + vh) * hk + dk];
                }
                out[(i * c.num_v_heads + vh) * hv + dv] = acc;
            }
        }
    }

    // RMSNormGated：x * rsqrt(mean(x²)+eps) * weight * act(z)
    // act 由模型 config 的 output_gate_type 决定（本模型 = sigmoid，非 silu）
    let act = c.gate_activation;
    let mut gated = vec![0f32; t * value_dim];
    for i in 0..t {
        for vh in 0..c.num_v_heads {
            let base = (i * c.num_v_heads + vh) * hv;
            let xv = &out[base..base + hv];
            let ms: f32 = xv.iter().map(|v| v * v).sum::<f32>() / hv as f32;
            let r = 1.0 / (ms + 1e-6).sqrt();
            for dv in 0..hv {
                let zz = z_h[i * value_dim + vh * hv + dv];
                gated[i * value_dim + vh * hv + dv] = xv[dv] * r * norm_w[dv] * act.apply(zz);
            }
        }
    }
    Recurrent { out, gated, qn, kn, beta: beta_dbg, g: g_dbg }
}

/// `x[T, H] → out[T, H]`。
pub fn gdn_forward(
    k: &Kernels,
    w: &GdnWeights,
    x: &CudaSlice<bf16>,
    t: usize,
    c: &GdnCfg,
) -> Result<CudaSlice<bf16>> {
    let key_dim = c.num_k_heads * c.head_k_dim;
    let value_dim = c.num_v_heads * c.head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total = conv_dim + value_dim;
    let h = c.hidden;

    // 1) 投影（融合权重按行切段）
    let mut qkv = k.stream.alloc_zeros::<bf16>(t * conv_dim)?;
    proj_t1(k, x, &w.query_key_value_z.slice(0..conv_dim * h), &mut qkv, conv_dim, h, t)?;
    let mut z = k.stream.alloc_zeros::<bf16>(t * value_dim)?;
    proj_t1(k, x, &w.query_key_value_z.slice(conv_dim * h..total * h), &mut z, value_dim, h, t)?;
    let mut a = k.stream.alloc_zeros::<bf16>(t * c.num_v_heads)?;
    proj_t1(k, x, &w.a_b_projection.slice(0..c.num_v_heads * h), &mut a, c.num_v_heads, h, t)?;
    let mut bb = k.stream.alloc_zeros::<bf16>(t * c.num_v_heads)?;
    proj_t1(
        k, x, &w.a_b_projection.slice(c.num_v_heads * h..2 * c.num_v_heads * h),
        &mut bb, c.num_v_heads, h, t,
    )?;

    // 2) 因果卷积 + silu
    let mut conv = k.stream.alloc_zeros::<bf16>(t * conv_dim)?;
    k.causal_conv1d_silu_bf16(&qkv, &w.convolution, &mut conv, t, conv_dim, c.conv_kernel, 1)?;

    // 3) 取回 host（短序列下最小实现，性能优化留到 M3）
    let conv_h: Vec<f32> = k.stream.clone_dtoh(&conv)?.iter().map(|v| v.to_f32()).collect();
    let z_h: Vec<f32> = k.stream.clone_dtoh(&z)?.iter().map(|v| v.to_f32()).collect();
    let a_h: Vec<f32> = k.stream.clone_dtoh(&a)?.iter().map(|v| v.to_f32()).collect();
    let b_h: Vec<f32> = k.stream.clone_dtoh(&bb)?.iter().map(|v| v.to_f32()).collect();
    let norm_w: Vec<f32> = k.stream.clone_dtoh(&w.norm)?.iter().map(|v| v.to_f32()).collect();

    let dbg = crate::flag("LINGXI_GDN_DEBUG");
    if dbg {
        let dump = |tag: &str, v: &[f32]| {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let _ = std::fs::write(format!("/tmp/lingxi_gdn_{tag}.f32"), bytes);
        };
        dump("conv", &conv_h);
        dump("z", &z_h);
        dump("a", &a_h);
        dump("b", &b_h);
        let qkv_h: Vec<f32> = k.stream.clone_dtoh(&qkv)?.iter().map(|v| v.to_f32()).collect();
        dump("qkv_preconv", &qkv_h);
    }

    // 4) host 侧 recurrent gated delta rule（整段前向：状态初值 0）
    let mut s_all = vec![0f32; c.num_v_heads * c.head_k_dim * c.head_v_dim];
    let rec = recurrent_core(
        &conv_h, &z_h, &a_h, &b_h, &norm_w, &w.a_log, &w.dt_bias, c, t, &mut s_all,
    );
    if dbg {
        let dump = |tag: &str, v: &[f32]| {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let _ = std::fs::write(format!("/tmp/lingxi_gdn_{tag}.f32"), bytes);
        };
        dump("out", &rec.out);
        dump("qn", &rec.qn);
        dump("kn", &rec.kn);
        dump("beta", &rec.beta);
        dump("g", &rec.g);
        dump("gated", &rec.gated);
    }
    let gated_bf: Vec<bf16> = rec.gated.iter().map(|v| bf16::from_f32(*v)).collect();
    let dgated = k.stream.clone_htod(&gated_bf)?;

    // 6) out_proj
    let mut y = k.stream.alloc_zeros::<bf16>(t * h)?;
    proj_t1(k, &dgated, &w.output, &mut y, h, value_dim, t)?;
    Ok(y)
}

// ───────────────────────────── 增量（driver 用） ─────────────────────────────

/// GDN 层的**跨 token 状态**：卷积历史 + recurrent 状态 S。
/// 由 driver 的 slot 持有，逐 token 推进。
pub struct GdnState {
    /// 最近 `conv_kernel-1` 个 token 的**卷积前**输入 `[K-1, conv_dim]`（bf16）。
    /// 初值全 0，恰好等价于整段前向里"越界视为 0"的边界。
    pub k_hist: CudaSlice<bf16>,
    /// **双缓冲的第二格**：融合卷积一次 launch 里把「历史推进」写到这一格，然后 host 侧 swap。
    /// 为什么不就地写回 `k_hist`：融合核内部「读 hist → 写 new_hist」跨 block 有写后读竞态
    /// （`new_hist[0..kh)` 与 `hist[1..kh+1)` 重叠），必须换缓冲。
    pub k_hist_next: CudaSlice<bf16>,
    /// `[num_v_heads, head_k_dim*head_v_dim]` 扁平化的 S（**设备 f32**，kernel 就地推进）。
    pub s_all: CudaSlice<f32>,
    /// `a_log` / `dt_bias` 的设备副本（首次使用时一次性上传；两者都很小）。
    pub a_log_d: Option<CudaSlice<f32>>,
    pub dt_bias_d: Option<CudaSlice<f32>>,
}

impl GdnState {
    pub fn new(k: &Kernels, c: &GdnCfg) -> Result<Self> {
        let conv_dim = 2 * c.num_k_heads * c.head_k_dim + c.num_v_heads * c.head_v_dim;
        Ok(Self {
            k_hist: k.stream.alloc_zeros::<bf16>((c.conv_kernel - 1) * conv_dim)?,
            k_hist_next: k.stream.alloc_zeros::<bf16>((c.conv_kernel - 1) * conv_dim)?,
            s_all: k
                .stream
                .alloc_zeros::<f32>(c.num_v_heads * c.head_k_dim * c.head_v_dim)?,
            a_log_d: None,
            dt_bias_d: None,
        })
    }
}

/// 增量步进：`x[T, H] → out[T, H]`，`state` 就地推进 T 个 token。
///
/// 与 [`gdn_forward`] 的关系：**同一份 `recurrent_core`**，区别只在
/// ① 卷积带上了 `state.k_hist` 的 (K-1) 行历史；② S 从 `state.s_all` 续上并写回。
/// 因此「整段一次 T 行」与「逐 token 步进 T 次」应逐位一致（见 `lingxi-bench step-check`）。
pub fn gdn_step(
    k: &Kernels,
    w: &GdnWeights,
    x: &CudaSlice<bf16>,
    t: usize,
    c: &GdnCfg,
    state: &mut GdnState,
) -> Result<CudaSlice<bf16>> {
    gdn_step_rec(k, w, x, t, c, state, None)
}

/// **`gdn_step` 的内部输入录像 / 重放**。
///
/// 为什么要它：投机提交时要把 GDN 的状态挪到 `accept` 行处。`gdn_step` 做三件事
/// `投影(qkv/z/a/b) → 卷积(推进 k_hist) → 递推(推进 s_all)`，**三步全是 `mixed` 的纯函数**。
/// 所以只要把**它的内部输入**（投影的输出）录下来，重放时就只跑「卷积 + 递推」两个小核：
/// **完全跳过投影的权重读**（那是 GDN 层里最大的一笔 HBM 流量，~30 MB/层）。
///
/// 字节账：每行 `qkv(conv_dim) + z(value_dim) + a/b(2·num_v_heads)` ≈ 20.6 KB；
/// `t_max=7` × 36 个 GDN 层 ≈ **5.2 MB**（不是"逐行快照整个状态"那条的 500 MB）。
/// 而且**不需要单独快照 `s_all`/`k_hist`** —— 卷积核与递推核自己会把它们推进到正确位置。
///
/// **位等价**：卷积是因果的、递推是行串行的 ⇒ 用**同样字节**的前 `accept` 行重跑，
/// `cout[0..accept]` 与批量时的逐位相同 ⇒ 状态逐位相同。
pub struct GdnRec<'a> {
    /// `[t_max, conv_dim]`：卷积的输入（= `query_key_value_z` 的 qkv 段投影结果）
    pub qkv: &'a mut CudaSlice<bf16>,
    /// `[t_max, value_dim]`
    pub z: &'a mut CudaSlice<bf16>,
    /// `[t_max, num_v_heads]`：门控 `a`
    pub a: &'a mut CudaSlice<bf16>,
    /// `[t_max, num_v_heads]`：门控 `b`（与 `a` **分开存**：合起来就要逐行切片拷，
    /// `t≤8` 时是十几次小 memcpy —— 又变成发射受限了）
    pub b: &'a mut CudaSlice<bf16>,
    /// 行偏移（chunk 化验证时非 0）
    pub off: usize,
    /// `true` = 从录像**读**并跳过投影；`false` = 把刚算出来的**写**进录像
    pub replay: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_step_rec(
    k: &Kernels,
    w: &GdnWeights,
    x: &CudaSlice<bf16>,
    t: usize,
    c: &GdnCfg,
    state: &mut GdnState,
    mut rec: Option<GdnRec<'_>>,
) -> Result<CudaSlice<bf16>> {
    let key_dim = c.num_k_heads * c.head_k_dim;
    let value_dim = c.num_v_heads * c.head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let total = conv_dim + value_dim;
    let h = c.hidden;
    let kh = c.conv_kernel - 1;

    // 1) 投影（与整段同）
    let prof = crate::flag("LINGXI_PROF");
    // `LINGXI_KTS=1`：GDN 单层的**分段 GPU 时间**（每段插一次同步括号；只为定位，会破坏流水）。
    // 为什么需要它：`[gdn]` 只测 CPU 侧、且**不含 `o_proj`**；而 `LINGXI_KT` 探针在
    // `LINGXI_DENSE_Q4=1` 下会 panic（它引用的是空的 bf16 `query_key_value_z`）。
    let kts = crate::flag("LINGXI_KTS");
    if kts {
        k.stream.synchronize()?;
    }
    let t0 = std::time::Instant::now();
    // 【子段括号】`投影` 段不是"一个 kernel"：它是
    // ① `qkv` 的 Q4 GEMV（[conv_dim, h]，本轮最大的一块）
    // ② `z` 的 Q4 GEMV（[value_dim, h]，同一份权重的另一段，**又读一遍 x**）
    // ③ 两个 tiny 的 bf16 GEMV（`a_b_projection` 的 a 段与 b 段，n=num_v_heads）
    // ④ 5 次 `scratch` 分配 + 视图切片的 CPU 侧开销
    // 冷启动口径下 ①+② 的字节（21.2 MB / 403 GB/s）应 ≈55 µs，而 KTS 实测这一段要 ~86 µs
    // ⇒ 差的 ~31 µs 到底是"多一次 launch/排空"还是"两个 tiny GEMV"还是"CPU 侧"，靠这三个刻度二分。
    let mut t_mark_qkvz = t0;
    let mut t_mark_z = t0;
    let mut t_mark_ab = t0;
    // ★ **重放模式**（`LINGXI_...`）：跳过全部投影，直接把录像里的内部输入拷回来。
    // 投影（尤其 `query_key_value_z` 那 84 MB/层）是 GDN 层最大的 HBM 流量，
    // 而它的输出对同一批行是完全确定的 ⇒ 重放时**没有理由再算一次**。
    let replay = rec.as_ref().map_or(false, |r| r.replay);
    // `LINGXI_GDN_REPLAY_ZERO=1`（诊断）：重放时把这 4 个内部输入缓冲**零初始化**。
    // 用途：判"重放读到了没被完整写过/残留的内存"——若零初始化后抖消失，就是这类问题。
    let zero_init = replay && crate::flag("LINGXI_GDN_REPLAY_ZERO");
    let alc = |n: usize| -> Result<CudaSlice<bf16>> {
        if zero_init {
            Ok(k.stream.alloc_zeros::<bf16>(n)?)
        } else {
            scratch::bf16(k, n)
        }
    };
    let mut qkv = alc(t * conv_dim)?;
    let mut z = alc(t * value_dim)?;
    let mut a = alc(t * c.num_v_heads)?;
    let mut bb = alc(t * c.num_v_heads)?;
    let nvh = c.num_v_heads;
    // ★ 2026-10-24：a/b 是否已并进 `qkvz` 那一发（见下面 `proj_q4_plus_ab`）。
    let mut ab_fused = false;
    // `LINGXI_ATTN_SYNC=1`：分段探针的现场信息（`begin` 已由 layer.rs 打过第 0 个事件）
    crate::attn_prof::info(t, 0);
    if replay {
        let r = rec.as_ref().unwrap();
        let off = r.off;
        k.stream.memcpy_dtod(&r.qkv.slice(off * conv_dim..(off + t) * conv_dim), &mut qkv)?;
        k.stream.memcpy_dtod(&r.z.slice(off * value_dim..(off + t) * value_dim), &mut z)?;
        k.stream.memcpy_dtod(&r.a.slice(off * nvh..(off + t) * nvh), &mut a)?;
        k.stream.memcpy_dtod(&r.b.slice(off * nvh..(off + t) * nvh), &mut bb)?;
        // 重放跳过全部投影 ⇒ 段 1/2 记 0 宽，段 3 = 这四次拷贝
        crate::attn_prof::mark(k, 3)?;
        if kts {
            k.stream.synchronize()?;
            t_mark_qkvz = std::time::Instant::now();
            t_mark_z = t_mark_qkvz;
            t_mark_ab = t_mark_qkvz;
        }
    } else {
    // `query_key_value_z` 是每层最大的稠密矩阵（84 MB），decode 下走 W4 融合反量化 GEMV
    // 可以把它的 HBM 流量砍到 1/2~1/2.2（实测 Q4GEMV 321 GB/s vs bf16 725 GB/s，payload 小 3.77x）。
    if let Some(wq) = &w.query_key_value_z_q4 {
        // ★ 2026-10-24：a/b 的 96 行 bf16 作为**本发 qkvz 的尾巴 block**（合发）。
        // 省掉两次 n=48 的 tiny GEMV（17.5 µs，几乎全是发射地板）；
        // 合发核逐字保留 Q4 体与 `gemm_skinny_row_bf16` 行体 ⇒ 逐位等价。
        // `LINGXI_NO_GDN_AB_FUSE=1` 回退（回退时下面那段还会照旧发两次 `proj_t1`）。
        ab_fused = crate::gpu::proj_q4_plus_ab(
            k, x, wq, &mut qkv, conv_dim,
            &w.a_b_projection, &mut a, &mut bb, nvh, h, t,
        )?;
        if !ab_fused {
            crate::gpu::proj_q4_epi(k, x, wq, &mut qkv, conv_dim, h, 0, t, 0, 0.0, 0.0)?;
        }
        crate::attn_prof::mark(k, 1)?;
        if kts {
            k.stream.synchronize()?;
            t_mark_qkvz = std::time::Instant::now();
        }
        crate::gpu::proj_q4_epi(k, x, wq, &mut z, value_dim, h, conv_dim, t, 0, 0.0, 0.0)?;
        crate::attn_prof::mark(k, 2)?;
        if kts {
            k.stream.synchronize()?;
            t_mark_z = std::time::Instant::now();
        }
    } else {
        let qkv_w = w.query_key_value_z.slice(0..conv_dim * h);
        let z_w = w.query_key_value_z.slice(conv_dim * h..total * h);
        proj_t1(k, x, &qkv_w, &mut qkv, conv_dim, h, t)?;
        crate::attn_prof::mark(k, 1)?;
        if kts {
            k.stream.synchronize()?;
            t_mark_qkvz = std::time::Instant::now();
        }
        proj_t1(k, x, &z_w, &mut z, value_dim, h, t)?;
        crate::attn_prof::mark(k, 2)?;
        if kts {
            k.stream.synchronize()?;
            t_mark_z = std::time::Instant::now();
        }
    }
    if !ab_fused {
        proj_t1(k, x, &w.a_b_projection.slice(0..nvh * h), &mut a, nvh, h, t)?;
        proj_t1(
            k, x, &w.a_b_projection.slice(nvh * h..2 * nvh * h),
            &mut bb, nvh, h, t,
        )?;
    }
    crate::attn_prof::mark(k, 3)?;
    if kts {
        k.stream.synchronize()?;
        t_mark_ab = std::time::Instant::now();
    }
    // 录像模式：把刚算出来的内部输入写进录像（供提交时重放）
    if let Some(r) = rec.as_mut() {
        let off = r.off;
        k.stream.memcpy_dtod(&qkv, &mut r.qkv.slice_mut(off * conv_dim..(off + t) * conv_dim))?;
        k.stream.memcpy_dtod(&z, &mut r.z.slice_mut(off * value_dim..(off + t) * value_dim))?;
        k.stream.memcpy_dtod(&a, &mut r.a.slice_mut(off * nvh..(off + t) * nvh))?;
        k.stream.memcpy_dtod(&bb, &mut r.b.slice_mut(off * nvh..(off + t) * nvh))?;
    }
    }

    // [探针] `LINGXI_GEMV_PROBE=<layer>`：把本层 qkv+z 投影**背靠背连发 8 次**，量"无 CPU 空档"的
    // 每发成本。用途：区分「引擎里 GEMV 真的只有 ~143 GB/s」与「KTS 的单发把 launch/排空延迟
    // 也算进去了」—— launch-bench 同形状的 30 次连发是 526 GB/s，冷启动单发只有 409。
    if t == 1 {
        if let Ok(v) = std::env::var("LINGXI_GEMV_PROBE") {
            if v != "0" {
                use std::sync::atomic::{AtomicU32, Ordering};
                static GP_N: AtomicU32 = AtomicU32::new(0);
                let call = GP_N.fetch_add(1, Ordering::Relaxed);
                if call < 2 {
                    if let Some(wq) = &w.query_key_value_z_q4 {
                        k.stream.synchronize()?;
                        let its = 8usize;
                        let t1 = std::time::Instant::now();
                        for _ in 0..its {
                            crate::gpu::proj_q4_epi(k, x, wq, &mut qkv, conv_dim, h, 0, t, 0, 0.0, 0.0)?;
                            crate::gpu::proj_q4_epi(k, x, wq, &mut z, value_dim, h, conv_dim, t, 0, 0.0, 0.0)?;
                        }
                        k.stream.synchronize()?;
                        let dt = t1.elapsed().as_secs_f64() / its as f64;
                        let bytes = (conv_dim + value_dim) * h / 2;
                        eprintln!(
                            "[gemv-probe] 第 {} 个 GDN 层 背靠背 {its} 次：**{:.1} µs/次 ⇒ {:.0} GB/s**（payload {:.1} MB）",
                            call + 1,
                            dt * 1e6,
                            bytes as f64 / dt / 1e9,
                            bytes as f64 / 1048576.0
                        );
                    }
                }
            }
        }
    }

    // `LINGXI_KT=1`：在引擎内量**单个 kernel 的真实 GPU 时间**，判断引擎里的 kernel 是否也像
    // 微基准那样带宽受限（微基准 [16384,2560] → 108µs / 775 GB/s）。
    // ⚠️ 必须加 `query_key_value_z_q4.is_none` 守卫：开了 `LINGXI_DENSE_Q4` 之后
    // bf16 的 `query_key_value_z` 是空的，下面 `.slice(0..conv_dim*h)` 会 unwrap 到 None 直接 panic
    //（实测：`LINGXI_KT=1` + `LINGXI_DENSE_Q4=1` → 首个 decode token 崩在 cudarc core.rs:2010）。
    if kt_on() && t == 1 && w.query_key_value_z_q4.is_none() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static KT_N: AtomicU32 = AtomicU32::new(0);
        if KT_N.fetch_add(1, Ordering::Relaxed) < 3 {
            k.stream.synchronize()?; // 清空流水（不计入）
            let t_kt = std::time::Instant::now();
            proj_t1(k, x, &w.query_key_value_z.slice(0..conv_dim * h), &mut qkv, conv_dim, h, t)?;
            k.stream.synchronize()?; // 等这个 kernel 真正跑完
            eprintln!(
                "[kt] qkvz 投影（84MB）单 kernel 实测 = **{:.3} ms**（带宽受限应 ≈0.108ms）",
                t_kt.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    let t_proj = if kts {
        k.stream.synchronize()?;
        t0.elapsed()
    } else {
        t0.elapsed()
    };
    // 2) 卷积：**融合版（一次 launch）**。
    // 原来要「hist→cin 拷 / qkv→cin 拷 / conv 算 (kh+t) 行 / 末尾 kh 行→hist 拷」= **4 次 launch**，
    // `LINGXI_KTS` 实测 **28 µs/层**，而数据只有 ~5 µs（launch ≈ 6.6 µs）——几乎全是发射开销。
    // 现在直接读 `(k_hist, qkv)`、只算 t 行、把历史推进写进 `k_hist_next`（双缓冲避开竞态）。
    let mut cout = scratch::bf16(k, t * conv_dim)?;
    if kh > 0 {
        // ⚠️ 指针守卫（`DevicePtrGuard`）在 Drop 时才算归还借用，所以必须**整段括起来**，
        // 不能跨越 `swap`（否则 E0499/E0502：swap 需要再借 `state.k_hist*`）。
        {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (hp, _g0) = state.k_hist.device_ptr(&k.stream);
            let (qp, _g1) = qkv.device_ptr(&k.stream);
            let (wp, _g2) = w.convolution.device_ptr(&k.stream);
            let (yp, _g3) = cout.device_ptr_mut(&k.stream);
            let (np, _g4) = state.k_hist_next.device_ptr_mut(&k.stream);
            k.causal_conv1d_silu_hist_bf16(hp, qp, wp, yp, np, kh, t, conv_dim, c.conv_kernel, 1)?;
        }
        std::mem::swap(&mut state.k_hist, &mut state.k_hist_next);
    } else {
        k.causal_conv1d_silu_bf16(&qkv, &w.convolution, &mut cout, t, conv_dim, c.conv_kernel, 1)?;
    }
    scratch::put_bf16(qkv);

    // 段 4 结束：卷积
    crate::attn_prof::mark(k, 4)?;
    let t_conv = if kts {
        k.stream.synchronize()?;
        t0.elapsed()
    } else {
        t0.elapsed()
    };
    // 3) 递推：**GPU kernel** 就地推进 `state.s_all`。
    // `LINGXI_GDN_HOST=1` 回退到 host 参考（`recurrent_core`），用于对拍。
    let mut dgated = scratch::bf16(k, t * value_dim)?;
    let host_fallback = crate::flag("LINGXI_GDN_HOST");
    if host_fallback {
        let mut s_host: Vec<f32> = k.stream.clone_dtoh(&state.s_all)?;
        let cout_all: Vec<f32> = k.stream.clone_dtoh(&cout)?.iter().map(|v| v.to_f32()).collect();
        let conv_h = &cout_all[..];
        let z_h: Vec<f32> = k.stream.clone_dtoh(&z)?.iter().map(|v| v.to_f32()).collect();
        let a_h: Vec<f32> = k.stream.clone_dtoh(&a)?.iter().map(|v| v.to_f32()).collect();
        let b_h: Vec<f32> = k.stream.clone_dtoh(&bb)?.iter().map(|v| v.to_f32()).collect();
        let norm_w: Vec<f32> = k.stream.clone_dtoh(&w.norm)?.iter().map(|v| v.to_f32()).collect();
        let rec = recurrent_core(
            conv_h, &z_h, &a_h, &b_h, &norm_w, &w.a_log, &w.dt_bias, c, t, &mut s_host,
        );
        let gated_bf: Vec<bf16> = rec.gated.iter().map(|v| bf16::from_f32(*v)).collect();
        k.stream.memcpy_htod(&s_host, &mut state.s_all)?;
        k.stream.memcpy_htod(&gated_bf, &mut dgated)?;
    } else {
        if state.a_log_d.is_none() {
            state.a_log_d = Some(k.stream.clone_htod(&w.a_log)?);
            state.dt_bias_d = Some(k.stream.clone_htod(&w.dt_bias)?);
        }
        let gate_act = match c.gate_activation {
            lingxi_ids::GateActivation::Silu => 0,
            lingxi_ids::GateActivation::Sigmoid => 1,
        };
        // ㉟+1：递推多 warp 占用率版（M5 便宜形态）。t>1 才值得（decode t=1 走旧核），
        // 有损档（跨 split 定序归约 ≠ 旧核单线程升序；PLAN_NEXT 判据），`LINGXI_GDN_MW=1`。
        if t > 1 && crate::flag("LINGXI_GDN_MW") {
            k.gdn_recurrent_mw_bf16(
                &cout, &z, &a, &bb, &w.norm,
                state.a_log_d.as_ref().unwrap(), state.dt_bias_d.as_ref().unwrap(),
                &mut state.s_all, &mut dgated,
                t, c.head_k_dim, c.head_v_dim, c.num_v_heads, c.num_k_heads, gate_act,
            )?;
        } else {
            k.gdn_recurrent_bf16(
                &cout, &z, &a, &bb, &w.norm,
                state.a_log_d.as_ref().unwrap(), state.dt_bias_d.as_ref().unwrap(),
                &mut state.s_all, &mut dgated,
                t, c.head_k_dim, c.head_v_dim, c.num_v_heads, c.num_k_heads, gate_act, 0, 1,
            )?;
        }
    }

    let ts_rec = if kts {
        k.stream.synchronize()?;
        t0.elapsed()
    } else {
        t0.elapsed()
    };
    // 段 5 结束：递推
    crate::attn_prof::mark(k, 5)?;

    if prof {
        let t_all = t0.elapsed();
        eprintln!(
            "      [gdn] 投影 {:.1} 卷积 {:.1} 递推({}) {:.1} 合计 {:.1} µs",
            t_proj.as_secs_f64() * 1e6,
            (t_conv - t_proj).as_secs_f64() * 1e6,
            if host_fallback { "host" } else { "gpu" },
            (t_all - t_conv).as_secs_f64() * 1e6,
            t_all.as_secs_f64(),
        );
    }
    // 5) out_proj
    let mut y = scratch::bf16(k, t * h)?;
    if let Some(wq) = &w.output_q4 {
        crate::gpu::proj_q4_epi(k, &dgated, wq, &mut y, h, value_dim, 0, t, 0, 0.0, 0.0)?;
    } else {
        proj_t1(k, &dgated, &w.output, &mut y, h, value_dim, t)?;
    }
    if kts {
        k.stream.synchronize()?;
        use std::sync::atomic::{AtomicU32, Ordering};
        static KTS_N: AtomicU32 = AtomicU32::new(0);
        if KTS_N.fetch_add(1, Ordering::Relaxed) < 400 {
            let t2 = t0.elapsed();
            eprintln!(
                "[kts] 投影 {:.1} 卷积 {:.1} 递推 {:.1} out_proj {:.1} 合计 **{:.1} µs**",
                t_proj.as_secs_f64() * 1e6,
                (t_conv - t_proj).as_secs_f64() * 1e6,
                (ts_rec - t_conv).as_secs_f64() * 1e6,
                (t2 - ts_rec).as_secs_f64() * 1e6,
                t2.as_secs_f64() * 1e6,
            );
            // 投影段的子段二分（见上面 `t_mark_*` 的注释）
            eprintln!(
                "      [kts-投影] qkvz(alloc+GEMV) {:.1} | z GEMV {:.1} | a/b 两个 tiny GEMV {:.1} | 其它(CPU/KT探针) {:.1} µs",
                (t_mark_qkvz - t0).as_secs_f64() * 1e6,
                (t_mark_z - t_mark_qkvz).as_secs_f64() * 1e6,
                (t_mark_ab - t_mark_z).as_secs_f64() * 1e6,
                // 尾段 = 投影段总时长 − 到 a/b 刻度为止的偏移（两者都是 Duration，不能拿 Instant 相减）
                (t_proj - (t_mark_ab - t0)).as_secs_f64() * 1e6,
            );
        }
    }
    scratch::put_bf16(cout);
    scratch::put_bf16(z);
    scratch::put_bf16(a);
    scratch::put_bf16(bb);
    scratch::put_bf16(dgated);
    // 段 6 结束：out_proj（层末：sync + 汇总打印）
    crate::attn_prof::mark(k, 6)?;
    Ok(y)
}
