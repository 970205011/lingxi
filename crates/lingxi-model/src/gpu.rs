//! GPU 权重上传：把产物张量搬到显存。
//!
//! 当前只支持 `BF16 + contiguous-le-v1`（GR 而了了，/ MoE / 大部分权重都是这个形态）。
//! q4（`row-split-k128-v1`）的按需解码随后补（M2 后半 / M3）。

use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use lingxi_artifact::Artifact;
use lingxi_ids::{TensorFormat, TensorLayout};
use lingxi_kernels::Kernels;

/// decode（t==1）GEMV 内核选择。
///
/// 实测（`lingxi-bench launch-bench` 三核并排，2026-10-15）：
///
/// | 形状 | warps-per-row | block(1024线程/行,标量) | **row8(128-bit)** |
/// |---|---|---|---|
/// | [8192,2560] | 724 | 323 | **754** |
/// | [2560,8192] | 695 | 430 | **790** |
/// | [6144,2560] | 629 | 318 | **729** |
/// | [320,10240] | 252 | 377 | **667** |
/// | [1280,2560] | 551 | 277 | **634** |
/// | [512,2560] | 349 | 250 | **387** |
/// | [10240,320] | **585** | 79 | 270 |
/// | [2560,640] | **493** | 143 | 389 |
///
/// 规律很清楚：**每行字节数够大（k ≥ 1024）时，一行一 block + 128-bit load 全面胜出**
/// —— 瓶颈从来不是带宽而是"每线程只发 2 字节标量 load"的指令/延迟链。
/// 反过来 `k` 小时（如 mix_up 的 k=320、shared_down 的 k=640）每个 block 活太少，
/// 启动开销占主导，warps-per-row 反而更好。
fn use_row8(kdim: usize) -> bool {
    kdim >= 1024 && kdim % 8 == 0 && !crate::flag("LINGXI_NO_ROW8")
}

/// `t>1`（批量）时用哪种瘦长 GEMM。
///
/// 旧版 `gemm_skinny_m*` 的 grid 是 **`n/8` 个 block**（warps-per-row），n 小的时候并行度
/// 差 8 倍（`block_inject` n=4 → **1 个 block**）。实测 t=2 时 GR 从 3.4 ms 涨到 10.3 ms。
/// 新核 `gemm_skinny_row` 把 grid 拉回 n 个 block 并改用 128 位 load —— 判据与 `use_row8` 同源
/// （`k ≥ 1024` 时"一行一 block + 向量 load"才划算）。`LINGXI_NO_SKINNY_ROW=1` 可回退。
fn use_skinny_row(kdim: usize) -> bool {
    kdim >= 1024 && kdim % 8 == 0 && !crate::flag("LINGXI_NO_SKINNY_ROW")
}


/// T>1 时用**瘦长 GEMM** 替掉 cuBLAS；`t > 8` 时按 8 行切片循环（权重每片读一遍）。
/// 为什么不能留给 cuBLAS：实测一次 `t=12` 的前向 **594 ms**（t=8 只要 159 ms）——
/// 这是 t≥9 掉出 `use_skinny` 后的悬崖，也解释了历史 prefill 数字为何那么难看。
/// `LINGXI_NO_SKINNY=1` 可回退做 A/B。
const SKINNY_MAX_M: usize = 8;

/// `t>1` 瘦长 GEMM 的分派：`k ≥ 1024` 走"一行一 block + 128 位 load"的新核，否则旧核。
#[allow(clippy::too_many_arguments)]
fn skinny_call(
    k: &Kernels, wp: u64, xp: u64, yp: u64, n: usize, kdim: usize, m: usize,
    epi: i32, a: f32, b: f32,
) -> Result<()> {
    // ☆ P15-⑲ 否决留档：多行/块版（`gemm_skinny_row_multi_bf16`，x 进 smem 行间共享、
    // 对 row 逐位等价、对拍 0 不一致）实测**全面更慢**（m=8 慢 8×）——V100 的 L2 足够快，
    // smem 往返 + barrier + block 数 ÷4 的延迟隐藏损失超过了 x 流量节省。m=8 的 ~350-420
    // GB/s 就是这个形状的实用地板。核与 launch-bench 对照保留。
    // ★ （2026-10-27）：上面的"m=8 地板"只适用于 m≤8。prefill t=512 时 mix_up
    // [n=20480 k=320 m=512] 走旧核 grid.y=64 ⇒ 权重 DRAM 重读 64 遍（2.60ms/层）；
    // `lingxi_gemm_skinny_mwide_bf16`（warp-per-输出行 × 行块 32 × 零 smem，x 走 L1/L2）
    // 与旧核**逐位等价**（同 lane 步进 j 偏分 + 同 shuffle 树），权重重读降到 m/32 遍。
    // 门控：k≤1024（L1 行块预算）且 m≥64；`LINGXI_NO_MWIDE=1` 回退。
    if m >= 64 && kdim <= 1024 && !crate::flag("LINGXI_NO_MWIDE") {
        k.gemm_skinny_mwide_bf16_ptr(wp, xp, yp, n, kdim, m, epi, a, b)
    } else if use_skinny_row(kdim) {
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, n, kdim, m, epi, a, b)
    } else {
        k.gemm_skinny_bf16_ptr(wp, xp, yp, n, kdim, m, epi, a, b)
    }
}

/// `LINGXI_MLARGE_MIN_K=<n>`：mlarge 门禁的 k 下限（默认 512，行为不变）。
/// 实验用。`LINGXI_MLARGE_MIN_N=<n>`：n 下限（默认 1024）。
/// 判决：mlarge 是 CTA=8 输出行结构，**小 n 时 grid.x=n/8 塌成几 CTA**
/// （n=8 inject → 8 CTA；n=320 down → 320 CTA 但每 CTA 只摊 8 行）⇒ 实测
/// down 4.13ms vs skinny_row **0.99ms**、inject 0.70ms vs **0.037ms**（逐位档！），
/// prefill 16.5s → **10.5s（−35%）**。mlarge retain 给 attn 大 n 投影（设计目标形状）。
fn mlarge_min_k() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LINGXI_MLARGE_MIN_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
    })
}
fn mlarge_min_n() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LINGXI_MLARGE_MIN_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
    })
}

/// 投影张量核化（`bf16_tcp.cu`，Volta mma.m8n8k4）：**有损档、默认关**——
/// `LINGXI_PROJ_TC=1` 才启用。形状门禁与 mlarge 完全同档（t>64 且 k%64==0 且
/// k≥mlarge_min_k 且 n≥mlarge_min_n）： 已判决这批形状是指令发射地板，
/// TC 化直接替换 mlarge 分流；t≤64（含 decode t=1 小 m 形态）一律旧路径不动。
fn use_proj_tc(t: usize, kdim: usize, n: usize) -> bool {
    crate::flag_val("LINGXI_PROJ_TC", true)
        && t > 64
        && kdim % 64 == 0
        && kdim >= mlarge_min_k()
        && n >= mlarge_min_n()
}

/// 投影：**`t==1`（decode）走自研 GEMV，`t>1` 走自研瘦长 GEMM。一律不用 cuBLAS。**
///
/// 为什么连 t>1 也不给 cuBLAS：
/// 1. `gemm_ex` 每次调用约 **26 µs CPU 开销**（普通 launch 只要 ~6.6 µs）；
/// 2. 更要命的是它把形状摆成 `m=n, n=t` —— **cuBLAS 面对 `N=t` 这种极窄 N 会选到病态内核**。
/// 实测 MoE `shared_gate_up` 单层 **50 µs（t=1）→ 500 µs（t=2）**，48 层 = +22.5 ms。
///
/// 自研核在真形状上的实测（launch-bench）：[10240,320] 646 GB/s、[320,10240] 758 GB/s、
/// [2560,640] 514 GB/s —— 都在峰值的 60~85%。用**裸设备指针**传参，`CudaSlice` 与
/// `CudaView` 都能直接喂。
pub fn proj_t1<W: DevicePtr<bf16>>(
    k: &Kernels, x: &CudaSlice<bf16>, wv: &W, y: &mut CudaSlice<bf16>,
    n: usize, kdim: usize, t: usize,
) -> Result<()> {
    if t == 1 {
        let (wp, _a) = wv.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        if use_row8(kdim) {
            k.gemv_bf16_row8_epi_ptr(wp, xp, yp, n, kdim, 0, 0.0, 0.0)
        } else {
            k.gemv_bf16_ptr(wp, xp, yp, n, kdim)
        }
    } else {
        let (wp, _a) = wv.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        // ★ 2026-10-22：一发全覆盖（k≥1024 走 row 核的 grid.y 行分块；k<1024 走
        // gemm_skinny_m8 的 grid.y 行分块——两者都与旧 ≤8 行逐片循环**逐位等价**，
        // 并发块共享权重读 ⇒ DRAM 流量 ⌈t/8⌉ 遍 → ~1 遍）。
        // ★ 2026-10-22 ㊽+4：t>64 且 k%64==0 的大 m 投影改走 `mlarge`（CTA=8 行×64
        // token、k-tile 进 smem ⇒ x 只读一遍 vs row 核的 n/8 遍——第十六轮实测账：
        // out_proj x 重读 256 遍 = 5.2GB L2/层、3.7ms）。有损档（k 升序串行累加，
        // 判据 PLAN_NEXT；`LINGXI_NO_MLARGE=1` 回退 row 核）。
        if t > 64 && kdim % 64 == 0 && kdim >= mlarge_min_k() && n >= mlarge_min_n() && !crate::flag("LINGXI_NO_MLARGE") {
            if use_proj_tc(t, kdim, n) {
                // 张量核化（有损档，默认关）：同形状替换 mlarge。
                k.gemm_bf16_tcp_ptr(wp, xp, yp, n, kdim, t, 0, 0.0, 0.0)
            } else {
                k.gemm_bf16_mlarge_ptr(wp, xp, yp, n, kdim, t, 0, 0.0, 0.0)
            }
        } else {
            skinny_call(k, wp, xp, yp, n, kdim, t, 0, 0.0, 0.0)
        }
    }
}

/// 把紧随投影的**逐元素尾**折进 GEMV（仅 decode `t==1` 生效，`t>1` 时回退成
/// 普通 GEMM + 对应逐元素 kernel）。`epi` 语义见 `gemv_bf16_epi` 的 CUDA 注释：
/// 0=原样、1=`silu(acc*a)`、2=`sigmoid(acc*a)*b`、3=`acc*a+b`、4=swiglu（`n` 为 2I）。
pub fn proj_t1_epi<W: DevicePtr<bf16>>(
    k: &Kernels, x: &CudaSlice<bf16>, wv: &W, y: &mut CudaSlice<bf16>,
    n: usize, kdim: usize, t: usize, epi: i32, a: f32, b: f32,
) -> Result<()> {
    if t == 1 {
        let (wp, _a) = wv.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        if use_row8(kdim) && epi != 4 {
            k.gemv_bf16_row8_epi_ptr(wp, xp, yp, n, kdim, epi, a, b)
        } else {
            k.gemv_bf16_epi_ptr(wp, xp, yp, n, kdim, epi, a, b)
        }
    } else if epi != 4 {
        // T>1 的小批量也走瘦长 GEMM（含同样的尾融合）。
        let (wp, _a) = wv.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        // 一发全覆盖（grid.y 行分块，逐位等价）——同 proj_t1 的 2026-10-22 改动。
        // ㊽+4：大 m 走 mlarge（同 proj_t1，含 epi 融合）。
        if t > 64 && kdim % 64 == 0 && kdim >= mlarge_min_k() && n >= mlarge_min_n() && !crate::flag("LINGXI_NO_MLARGE") {
            if use_proj_tc(t, kdim, n) {
                // 张量核化（有损档，默认关）：同形状替换 mlarge。
                k.gemm_bf16_tcp_ptr(wp, xp, yp, n, kdim, t, epi, a, b)
            } else {
                k.gemm_bf16_mlarge_ptr(wp, xp, yp, n, kdim, t, epi, a, b)
            }
        } else {
            skinny_call(k, wp, xp, yp, n, kdim, t, epi, a, b)
        }
    } else {
        // `epi == 4`（swiglu）是本核唯一不支持的一档，拆成"瘦长 GEMM + swiglu"两步。
        //
        // ⚠️ GEMM **绝不能交给 cuBLAS**（曾经就是）：`gemm_bf16_nt` 把形状摆成 `m=n, n=t`，
        // t>1 时 cuBLAS 看到的是 `N=t` 这种**极窄 N**——它在 N=2 上会选到病态内核。
        // 实测 MoE shared expert 的 `shared_gate_up`（唯一的 epi=4 热路径）单层从
        // **t=1 的 50 µs 暴涨到 t=2 的 500 µs**，48 层 = **+22.5 ms**，是 里
        // t=2 前向 74.7 ms 的最大一块。换自研瘦长 GEMM 后同一形状只要 ~10 µs。
        let half = n / 2;
        let mut tmp = crate::scratch::bf16(k, t * n)?;
        {
            // 取指针的借用守卫必须整段括起来：`_c` 持 `tmp` 的独占借用，
            // 出了这个块才能把 `tmp` 借给 swiglu、再还回池。
            let (wp, _a) = wv.device_ptr(&k.stream);
            let (xp, _b) = x.device_ptr(&k.stream);
            let (tp, _c) = tmp.device_ptr_mut(&k.stream);
            let mut done = 0usize;
            while done < t {
                let m = (t - done).min(SKINNY_MAX_M);
                skinny_call(
                    k, wp,
                    xp + (done * kdim * 2) as u64,
                    tp + (done * n * 2) as u64,
                    n, kdim, m, 0, 0.0, 0.0,
                )?;
                done += m;
            }
        }
        // 直接写进 `y`（`y` 本来就只有 `t*half` 个元素），省掉一块 scratch 与一次 DtoD。
        k.swiglu_bf16(&tmp, y, t, half)?;
        crate::scratch::put_bf16(tmp);
        Ok(())
    }
}

/// ★ P15-⑨：**x 预转 fp32 的投影**（t>1；"所有后面计算过程要转化的提前转化"）。
/// `gemm_skinny_row` 在 m>1 时带宽腰斩（m=8 实测 350-415 GB/s vs m=1 750+）的根因是
/// 内层每 (元素,token) 重复 bf16→f32 CVT；这里 x 由调用方 `convert_bf16_f32` **一次**
/// 预转（同一 x 的多个投影共用一份），内层只剩 w 的 8 次 CVT + 每 token 8 FMA。
/// **逐位等价**：bf16→f32 精确、z 次序与累加次序不变。只支持 epi ∈ {0,1,2,3}。
#[allow(clippy::too_many_arguments)]
pub fn proj_f32x<W: DevicePtr<bf16>>(
    k: &Kernels, wv: &W, xf: &CudaSlice<f32>, y: &mut CudaSlice<bf16>,
    n: usize, kdim: usize, t: usize, epi: i32, a: f32, b: f32,
) -> Result<()> {
    let (wp, _g1) = wv.device_ptr(&k.stream);
    let (xp, _g2) = xf.device_ptr(&k.stream);
    let (yp, _g3) = y.device_ptr_mut(&k.stream);
    // 一发全覆盖（grid.y 行分块，逐位等价于旧逐片循环）——2026-10-22 改动。
    k.gemm_skinny_row_f32x_ptr(wp, xp, yp, n, kdim, t, epi, a, b)
}

/// 便捷入口：`t>1` 且开关开 ⇒ 预转 x 为 fp32 并返回；否则 `None`（调用方走原 bf16 路）。
/// ★ **默认关**（P15-⑨ 整机实测：微基准 m=8 带宽 350→700 GB/s，但整机 fwd t=7 **+2.6 ms**
/// 净亏——x fp32 让每 block 的 x L2 读翻倍 + 每处一次 convert launch）。
/// 设 `LINGXI_F32X_PROJ=1` 才启用；`LINGXI_F32X_SITES` 可**按位**关掉某几处做二分定位。
pub fn prep_f32x(
    k: &Kernels, x: &CudaSlice<bf16>, t: usize, n_elems: usize, site: u8,
) -> Result<Option<CudaSlice<f32>>> {
    if t <= 1 || !crate::flag("LINGXI_F32X_PROJ") {
        return Ok(None);
    }
    static SITES: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let sites = *SITES.get_or_init(|| {
        std::env::var("LINGXI_F32X_SITES")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0b111)
    });
    if sites & site == 0 {
        return Ok(None);
    }
    let mut xf = crate::scratch::f32(k, n_elems)?;
    k.convert_bf16_f32(x, &mut xf, n_elems)?;
    Ok(Some(xf))
}

/// `LINGXI_Q4GEMV_DENSE_VIEW=1|2|4`：把小 m（2..=8）的**稠密 Q4 投影**改走 MoE 专家段
/// 那条 `gemv_q4g64_slotsm_rpw_bf16_view`（E=1 ⇒ 整层 token 当"一个专家的 m 行"）。
/// 值 = rows_per_warp。**默认关**（0）。
fn dense_view_rpw() -> usize {
    static ON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LINGXI_Q4GEMV_DENSE_VIEW")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

thread_local! {
    /// 稠密 `_view` 路径的常量索引缓冲：`[slot_id=0, group_off[0]=0, group_off[1]=t]`。
    /// 值只依赖 `(设备, t)`，缓存起来 ⇒ 每个投影**零 H2D**。
    static DENSE_VIEW_IDX: std::cell::RefCell<Option<((usize, usize), CudaSlice<i32>)>> =
        const { std::cell::RefCell::new(None) };
}

/// 用缓存的索引缓冲发一次 E=1 的 `slotsm_rpw` 稠密 GEMV。
#[allow(clippy::too_many_arguments)]
fn dense_view_gemv(
    k: &Kernels, wq: &crate::q4::Q4W, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
    n: usize, kdim: usize, row_off: usize, t: usize, rpw: usize,
) -> Result<()> {
    let row_bytes = (wq.k_pad as u64) / 2;
    let row_groups = (wq.k_pad as u64) / 64;
    let seg_off = row_off as u64 * row_bytes;
    // 槽版把 scale 平面挂在 `payload + scale_off` 上、并按**段内相对行号** `row` 取
    // `scales[row*groups]`；而稠密布局的 scale 是按**绝对行号**存的 ⇒ 这里要补回
    // `row_off*groups*2` 字节，再减掉 payload 已经前移的 `seg_off`。
    let scale_abs = wq.scale_off + row_off as u64 * row_groups * 2;
    let Some(scale_rel) = scale_abs.checked_sub(seg_off) else {
        anyhow::bail!("dense_view: row_off={row_off} 超出 scale 平面（scale_off={}）", wq.scale_off);
    };
    DENSE_VIEW_IDX.with(|c| -> Result<()> {
        let mut g = c.borrow_mut();
        let key = (k.ctx.ordinal(), t);
        if g.as_ref().map_or(true, |(kk, _)| *kk != key) {
            // [slot_id=0, group_off[0]=0, group_off[1]=t] —— E=1、唯一一组含全部 t 个 token。
            *g = Some((key, k.stream.clone_htod(&[0i32, 0, t as i32])?));
        }
        let buf = &g.as_ref().unwrap().1;
        let ids = buf.slice(0..1);
        let goff = buf.slice(1..3);
        let n_i32 = n as i32;
        k.gemv_q4g64_slotsm_rpw_bf16_view(
            &wq.packed, &ids, x, &goff, y, 1, n, kdim, wq.k_pad, kdim,
            0, seg_off, scale_rel, rpw, 0,
        )
        .map_err(|e| anyhow::anyhow!("dense_view GEMV(n={n_i32}) 失败: {e}"))
    })
}

/// **Q4 稠密权重**投影（decode `t==1` 走融合反量化 GEMV；`t>1` 走**分块 Q4 瘦长 GEMM**）。
/// `row_off` 用于"同一块 Q4 矩阵的某几行"（例如 `query_key_value_z` 的 qkv 段与 z 段）。
#[allow(clippy::too_many_arguments)]
pub fn proj_q4_epi(
    k: &Kernels, x: &CudaSlice<bf16>, wq: &crate::q4::Q4W,
    y: &mut CudaSlice<bf16>, n: usize, kdim: usize, row_off: usize, t: usize,
    epi: i32, a: f32, b: f32,
) -> Result<()> {
    // `LINGXI_Q4GEMV_DENSE_VIEW`：捷径 —— m≤8 的稠密投影直接复用 MoE 专家段的
    // `slotsm_rpw` 核（E=1）。默认关；见 `dense_view_rpw` 的注释。
    let rpw = dense_view_rpw();
    if rpw > 0 && (2..=8).contains(&t) && epi != 4 && kdim % 8 == 0 {
        return dense_view_gemv(k, wq, x, y, n, kdim, row_off, t, rpw);
    }
    // t>1：**直接吃 Q4** 的分块瘦长 GEMM（修 根因 1）。
    // 旧路是"整块解量化成 bf16 再 GEMM"（qkvz 从 21 MB 变 105 MB 的往返）。
    if t > 1 && epi != 4 && kdim % 8 == 0 && !crate::flag("LINGXI_NO_Q4GEMM") {
        let (wp, _a) = wq.packed.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        // 一发全覆盖（grid.y 行分块，逐位等价于旧逐片循环）——2026-10-22 改动，
        // 同 proj_t1 的论证：并发块共享 Q4 权重读，DRAM 流量 ⌈t/8⌉ 遍 → ~1 遍。
        // ㊽+5：q4_mlarge **默认关**（`LINGXI_Q4MLARGE=1` 开启）。
        // ⚠️ 2026-10-22 实测回退：开 needle 43.5→52.8s（+21%）——标量 FMA 路径的
        // 反量化 ALU 链（每 u32 ~80 ALU / 8 MAC）没像 q4_tcp 那样被 MMA 摊销，
        // 比 gemm_q4_skinny 的查表路径慢 2-3×。x 重读账是真的，但 Q4 投影的
        // 正确解法应该是"q4_tcp 式 MMA + mlarge tiling"，不是标量版。核与封装
        // 保留（opt-in）。
        // ㊽+6：q4_tcp_dense = q4_tcp 的 MMA B 片段 + 稠密 W 布局（x 一遍、权重
        // 摊销、dequant 喂 MMA）——Q4 投影的正解。**默认开**，
        // `LINGXI_NO_Q4TCPDENSE=1` 回退。数值有损档（同 q4_tcp 口径），
        // 判据 = 首 token 相等 + 质量抽检。
        // ㊽+7（2026-10-22）：小 t 档（2..64）实测 gemm_q4_skinny 只有 60~83GB/s
        // （GDN 投影 KTS 分解：qkvz 88.8µs/7.4MB、z 55µs/3.3MB）。曾试把 tcp_dense
        // 下放到 t=2..64——**实测反而 2.4× 慢**（24.3/15.2 vs 58.9 tok/s，输出质量无损）：
        // M-tile=64 的 MMA 工作量与 m 无关，m=6 时 58 行 padding 白算，CTA 又少
        // （grid=(n/128,1)，45 块）⇒ 延迟墙更厚。正解 = purpose-built M≤8 MMA 核
        // （strata 的 m8n8k4 填 8 行思路），不是大 tile 下放。留 opt-in 供复测。
        // ㊽+8（2026-10-23）：**M≤8 专用 MMA 变体**（`q4_tcp_dense.cu` 的 m8 版）：
        // 把 M 轴 tile 压到 8（mma 最小高度），权重仍只读一遍、dequant 片段照样喂 MMA。
        // ⚠️ **实测负结果，默认关别开**：61.5→55.5 tok/s（~82ms/轮）。
        // 根因不是寄存器（修 spill 后一个点不变），而是 Q4 dequant ALU 按"每 4B 权重"
        // 计费、与 m 无关——skinny 用 6 条 FMA 摊它，换 MMA 一条也没省，还多付
        // 40 组×2 次 __syncthreads + 每 SM 只剩 4 warp 的延迟墙。留核供 m≥64 档复用。
        if t <= 8 && kdim % 64 == 0 && crate::flag("LINGXI_Q4TC_M8") {
            k.gemm_q4_tcp_dense_m8_ptr(
                wp, xp, yp, n, kdim, wq.k_pad as usize, t, row_off, wq.scale_off, epi, a, b,
            )?;
            return Ok(());
        }
        if t > 1 && kdim % 64 == 0 && crate::flag("LINGXI_Q4TCPDENSE_SMALL") {
            k.gemm_q4_tcp_dense_ptr(
                wp, xp, yp, n, kdim, wq.k_pad as usize, t, row_off, wq.scale_off, epi, a, b,
            )?;
        } else if t > 64 && kdim % 64 == 0 && !crate::flag("LINGXI_NO_Q4TCPDENSE") {
            k.gemm_q4_tcp_dense_ptr(
                wp, xp, yp, n, kdim, wq.k_pad as usize, t, row_off, wq.scale_off, epi, a, b,
            )?;
        } else if t > 64 && kdim % 64 == 0 && crate::flag("LINGXI_Q4MLARGE") {
            k.gemm_q4_mlarge_ptr(wp, xp, yp, n, kdim, t, row_off, wq.scale_off, epi, a, b)?;
        } else {
            k.gemm_q4_skinny_bf16_ptr(
                wp, xp, yp, n, kdim, wq.k_pad, t, row_off, wq.scale_off, epi, a, b,
            )?;
        }
        return Ok(());
    }
    if t == 1 {
        // `LINGXI_Q4GEMV_PROF=1`：t=1 的 Q4 稠密 GEMV 用**流内事件**报一次真 GPU 时间
        // （默认关；给"MTP 草稿步里 lm_head 到底多少"用——它落在 `lingxi-bench::spec`，
        // 外面只能靠 `mtp_prof` 的层外余量反推，这里直接量）。
        let prof = crate::flag("LINGXI_Q4GEMV_PROF");
        let mut evs = Vec::new();
        if prof {
            let f = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
            evs.push(k.ctx.new_event(f)?);
            evs.push(k.ctx.new_event(f)?);
            evs[0].record(&k.stream)?;
        }
        let (wp, _a) = wq.packed.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        let r = k.gemv_q4g64_dense_epi_ptr(
            wp, xp, yp, n, kdim, wq.k_pad, row_off, wq.scale_off, epi, a, b,
        );
        if prof && r.is_ok() {
            evs[1].record(&k.stream)?;
            k.stream.synchronize()?;
            if let Ok(ms) = evs[0].elapsed_ms(&evs[1]) {
                eprintln!(
                    "[q4gemv-prof] n{n} k{kdim} = {:.1} µs｜{:.1} GB/s",
                    ms as f64 * 1e3,
                    (n * (wq.k_pad as usize) / 2) as f64 / (ms as f64 * 1e6)
                );
            }
        }
        r
    } else {
        // t>1：按需把这一片 Q4 解回 bf16 再走 GEMM。权重因此**只需常驻 Q4 一份**，
        // 省下的显存留给专家槽。
        let mut tmp = crate::scratch::bf16(k, n * kdim)?;
        k.dequant_q4g64_dense_bf16(&wq.packed, &mut tmp, n, kdim, wq.k_pad, row_off, wq.scale_off)?;
        let r = proj_t1_epi(k, x, &tmp, y, n, kdim, t, epi, a, b);
        crate::scratch::put_bf16(tmp);
        r
    }
}

/// `proj_q4_epi` 在 `(kdim, t)` 上**是否会落到 `gemm_q4_skinny_bf16`**。
///
/// 为什么需要它：两个"合发"入口（[`proj_q4_epi4`] / [`proj_q4_plus_ab`]）只复刻了
/// `gemm_q4_skinny_bf16` 这一个内核体（逐字，保证逐位等价）。一旦用户的实验开关把
/// 同一形状分派到别的核（m8 / tcp_dense / mlarge / dense_view），合发就会**换数值**
/// ⇒ 那些档一律返回 `false`、调用方走原来的多次 launch。
fn q4_dispatch_is_skinny(kdim: usize, t: usize) -> bool {
    if dense_view_rpw() > 0 && (2..=8).contains(&t) && kdim % 8 == 0 {
        return false;
    }
    if t <= 1 || kdim % 8 != 0 || crate::flag("LINGXI_NO_Q4GEMM") {
        return false;
    }
    if t <= 8 && kdim % 64 == 0 && crate::flag("LINGXI_Q4TC_M8") {
        return false;
    }
    if kdim % 64 == 0 && crate::flag("LINGXI_Q4TCPDENSE_SMALL") {
        return false;
    }
    // t>64 走 tcp_dense / mlarge（t=6 的 decode 不受影响，prefill 保持旧路）
    if t > 64 && kdim % 64 == 0 {
        return false;
    }
    true
}

/// ★ 2026-10-24 **QSA 的 `q | k | gate | v` 四段投影合发**（4 发 → 1 发）。
///
/// 现状（，t=6）：四段分别 6144/512/6144/512 行，grid=768/64/768/64 块；
/// 后两段 64 块 < 80 SM ⇒ 每发都被发射地板吃掉。实测四发 161.5 µs、合并 n=13312 一发
/// 122.3 µs（−39 µs/层）。
///
/// 返回 `Ok(false)` = 本次形状/开关不适合合发（调用方回退 4 次 [`proj_q4_epi`]）。
/// `LINGXI_NO_QSA_PROJ_FUSE=1` 强制回退（A/B 用）。
#[allow(clippy::too_many_arguments)]
pub fn proj_q4_epi4(
    k: &Kernels, x: &CudaSlice<bf16>, wq: &crate::q4::Q4W,
    y0: &mut CudaSlice<bf16>, y1: &mut CudaSlice<bf16>,
    y2: &mut CudaSlice<bf16>, y3: &mut CudaSlice<bf16>,
    ns: [usize; 4], kdim: usize, t: usize,
) -> Result<bool> {
    if crate::flag("LINGXI_NO_QSA_PROJ_FUSE") || !q4_dispatch_is_skinny(kdim, t) {
        return Ok(false);
    }
    let (wp, _a) = wq.packed.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (y0p, _c0) = y0.device_ptr_mut(&k.stream);
    let (y1p, _c1) = y1.device_ptr_mut(&k.stream);
    let (y2p, _c2) = y2.device_ptr_mut(&k.stream);
    let (y3p, _c3) = y3.device_ptr_mut(&k.stream);
    k.gemm_q4_skinny4_bf16_ptr(
        wp, xp, y0p, y1p, y2p, y3p,
        ns[0], ns[1], ns[2], ns[3], kdim, wq.k_pad, t, 0, wq.scale_off,
    )?;
    Ok(true)
}

/// ★ 2026-10-24 **GDN 的 `qkvz`（Q4，n=conv_dim）+ `a/b`（bf16，96 行）合发**。
///
/// 现状：a/b 是两个 `n=48` 的 tiny GEMV（`gemm_skinny_row`），权重合计
/// 0.49 MB 却要 **17.5 µs** —— 几乎全是两发 ~6~8 µs 的 launch 地板 + 排空。
/// 这里把 96 行作为 `qkvz` 那发的**尾巴 block**（grid.x += 96），省掉两发。
///
/// 返回 `Ok(false)` = 本次形状/开关不适合（调用方回退 `proj_q4_epi(qkvz)` + 两次 `proj_t1`）。
/// `LINGXI_NO_GDN_AB_FUSE=1` 强制回退（A/B 用）。
#[allow(clippy::too_many_arguments)]
pub fn proj_q4_plus_ab(
    k: &Kernels, x: &CudaSlice<bf16>, wq: &crate::q4::Q4W,
    yq: &mut CudaSlice<bf16>, n_q4: usize,
    wab: &CudaSlice<bf16>, ya: &mut CudaSlice<bf16>, yb: &mut CudaSlice<bf16>,
    nvh: usize, kdim: usize, t: usize,
) -> Result<bool> {
    if crate::flag("LINGXI_NO_GDN_AB_FUSE") || !q4_dispatch_is_skinny(kdim, t) {
        return Ok(false);
    }
    // a/b 的旧路必须正是 `gemm_skinny_row_bf16`（t=1 走 GEMV、t>64 走 mlarge ⇒ 不合发）
    if !use_skinny_row(kdim) {
        return Ok(false);
    }
    let (wp, _a) = wq.packed.device_ptr(&k.stream);
    let (xp, _b) = x.device_ptr(&k.stream);
    let (yqp, _c) = yq.device_ptr_mut(&k.stream);
    let (wabp, _d) = wab.device_ptr(&k.stream);
    let (yap, _e) = ya.device_ptr_mut(&k.stream);
    let (ybp, _f) = yb.device_ptr_mut(&k.stream);
    k.gemm_q4_skinny_ab_bf16_ptr(
        wp, xp, yqp, wabp, yap, ybp, n_q4, nvh, kdim, wq.k_pad, t, 0, wq.scale_off,
    )?;
    Ok(true)
}

/// 读一个 bf16 张量的**位形**（不解释成 f32），用于在线量化到 Q4。
pub fn read_bf16_bits(art: &Artifact, name: &str) -> Result<(Vec<u16>, usize, usize)> {
    let obj = art.find(name).with_context(|| format!("产物缺少对象 {name}"))?;
    let (format, layout, shape) = obj.tensor_parts().context("不是 tensor 对象")?;
    if format != TensorFormat::Bf16 || layout != TensorLayout::ContiguousLeV1 {
        bail!("read_bf16_bits 只支持 BF16+contiguous-le-v1（{name} 是 {} + {}）", format.name(), layout.name());
    }
    if shape.len() != 2 {
        bail!("{name}: 需要 rank-2，得到 {shape:?}");
    }
    let bytes = art.payload(name).context("取 payload 失败")?;
    let bits: Vec<u16> =
        bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    Ok((bits, shape[0] as usize, shape[1] as usize))
}

/// 带**按张量名**磁盘缓存的 W4 上传 —— "能提前算的都提前算好，之后只取出来就行"。
///
/// 为什么按名字而不是按内容：真正吃掉启动时间的不是量化本身，而是
/// **把 6 GiB bf16 从产物读出来**（+ 算哈希）。按名字寻址可以先看缓存，
/// 命中就**连 bf16 都不读**，直接把 1/4 大小的 Q4 搬上显存 —— 这一步才是"只取"。
///
/// 缓存目录由 `LINGXI_Q4_CACHE=<dir>` 给出；未设则完全走原路径。未命中时
/// 读 bf16 → 量化 → 落盘 → 上传。落盘失败只影响"下次还要算"，不影响正确性。
pub fn upload_q4_named(k: &Kernels, art: &Artifact, name: &str) -> Result<crate::q4::Q4W> {
    let dir = std::env::var("LINGXI_Q4_CACHE").ok();
    if dir.is_none() {
        // ★ P15-㉔ 教训入码：忘设缓存=每次启动重新量化（吃满启动时间还容易让人误判配置）。
        static BANNER: std::sync::Once = std::sync::Once::new();
        BANNER.call_once(|| {
            eprintln!(
                "[启动提示] 未设 LINGXI_Q4_CACHE：W4 权重每次启动都重新量化（~25s+ 且非生产路径）。\
                 生产配置：export LINGXI_Q4_CACHE=/dev/shm/lingxi-q4cache"
            );
        });
    }
    if let Some(d) = dir.as_deref() {
        // FNV-1a 过名字（很短），文件名前缀 `n` 与旧的内容寻址缓存区分开
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let path = std::path::Path::new(d).join(format!("n{h:016x}.q4"));
        if let Ok(buf) = std::fs::read(&path) {
            if buf.len() > 28 {
                let rows = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
                let cols = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
                let k_pad = i32::from_le_bytes(buf[16..20].try_into().unwrap());
                let scale_off = u64::from_le_bytes(buf[20..28].try_into().unwrap());
                let packed = k.stream.clone_htod(&buf[28..])?;
                return Ok(crate::q4::Q4W { packed, rows, cols, k_pad, scale_off });
            }
            eprintln!("[q4cache] 缓存损坏，重算：{}", path.display());
        }
        let (bits, r, c) = read_bf16_bits(art, name)?;
        let (payload, k_pad, scale_off) = crate::q4::quantize_bf16_bits(&bits, r, c)
            .with_context(|| format!("Q4 量化失败 {name}"))?;
        if std::fs::create_dir_all(d).is_ok() {
            let mut out = Vec::with_capacity(28 + payload.len());
            out.extend_from_slice(&(r as u64).to_le_bytes());
            out.extend_from_slice(&(c as u64).to_le_bytes());
            out.extend_from_slice(&k_pad.to_le_bytes());
            out.extend_from_slice(&scale_off.to_le_bytes());
            out.extend_from_slice(&payload);
            if let Err(e) = std::fs::write(&path, &out) {
                eprintln!("[q4cache] 写缓存失败（不影响正确性）：{e}");
            }
        }
        let packed = k.stream.clone_htod(&payload)?;
        return Ok(crate::q4::Q4W { packed, rows: r, cols: c, k_pad, scale_off });
    }
    let (bits, r, c) = read_bf16_bits(art, name)?;
    crate::q4::upload_from_bf16_bits(k, &bits, r, c)
}

/// 上传一个 `BF16 + contiguous-le-v1` 张量到显存。
pub fn upload_bf16(k: &Kernels, art: &Artifact, name: &str) -> Result<CudaSlice<bf16>> {
    let obj = art.find(name).with_context(|| format!("产物缺少对象 {name}"))?;
    let (format, layout, _shape) = obj
        .tensor_parts()
        .with_context(|| format!("{name} 不是 tensor 对象"))?;
    if format != TensorFormat::Bf16 || layout != TensorLayout::ContiguousLeV1 {
        bail!(
            "{name}: 需要 BF16 + contiguous-le-v1，得到 {} + {}",
            format.name(),
            layout.name()
        );
    }
    let bytes = art.payload(name).context("取 payload 失败")?;
    let words: Vec<bf16> = bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Ok(k.stream.clone_htod(&words)?)
}

/// 按名字批量上传。
pub fn upload_many(k: &Kernels, art: &Artifact, names: &[&str]) -> Result<Vec<CudaSlice<bf16>>> {
    names.iter().map(|n| upload_any_bf16(k, art, n)).collect()
}

/// 把产物张量读到 host（f32）。给需要在 CPU 上做小规模计算的场合用（如 indexer 选块）。
pub fn read_f32_host(_k: &Kernels, art: &Artifact, name: &str) -> Result<Vec<f32>> {
    let obj = art.find(name).with_context(|| format!("产物缺少对象 {name}"))?;
    let (format, layout, _shape) = obj.tensor_parts().with_context(|| format!("{name} 不是 tensor"))?;
    if format != TensorFormat::Bf16 || layout != TensorLayout::ContiguousLeV1 {
        bail!("read_f32_host 只支持 BF16 + contiguous-le-v1（{name} 是 {} + {}）", format.name(), layout.name());
    }
    let bytes = art.payload(name).context("取 payload 失败")?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

/// 读一个 `FP32 + contiguous-le-v1` 张量到 host（如 GDN 的 `a_log` / `dt_bias`）。
pub fn read_fp32_host(art: &Artifact, name: &str) -> Result<Vec<f32>> {
    let obj = art.find(name).with_context(|| format!("产物缺少对象 {name}"))?;
    let (format, layout, _shape) = obj.tensor_parts().with_context(|| format!("{name} 不是 tensor"))?;
    if format != TensorFormat::Fp32 || layout != TensorLayout::ContiguousLeV1 {
        bail!("read_fp32_host 只支持 FP32 + contiguous-le-v1（{name} 是 {} + {}）", format.name(), layout.name());
    }
    let bytes = art.payload(name).context("取 payload 失败")?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// 上传任意产物张量到显存（bf16）：自动分派 `BF16+contiguous` 与 `Q4G64_F16S+row-split`。
pub fn upload_any_bf16(k: &Kernels, art: &Artifact, name: &str) -> Result<CudaSlice<bf16>> {
    let obj = art.find(name).with_context(|| format!("产物缺少对象 {name}"))?;
    let (format, layout, shape) = obj
        .tensor_parts()
        .with_context(|| format!("{name} 不是 tensor 对象"))?;
    match (format, layout) {
        (TensorFormat::Bf16, TensorLayout::ContiguousLeV1) => upload_bf16(k, art, name),
        (TensorFormat::Q4G64F16S, TensorLayout::RowSplitK128V1) => {
            if shape.len() != 2 {
                bail!("{name}: row-split 需要 rank-2 形状，得到 {shape:?}");
            }
            let (n, kk) = (shape[0] as usize, shape[1] as usize);
            let payload = art.payload(name).context("取 payload 失败")?;
            let dp = k.stream.clone_htod(payload)?;
            let k_pad = (kk as u64 + 127) / 128 * 128;
            let base_bytes = n as u64 * (k_pad / 2);
            let scale_offset = (base_bytes + 255) / 256 * 256;
            let mut out = k.stream.alloc_zeros::<bf16>(n * kk)?;
            k.dequant_rowsplit_q4g64_bf16(&dp, &mut out, n, kk, scale_offset)?;
            Ok(out)
        }
        _ => bail!(
            "{name}: 暂不支持的格式/布局 {} + {}",
            format.name(),
            layout.name()
        ),
    }
}
