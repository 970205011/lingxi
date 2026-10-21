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
/// | [512,2560]  | 349 | 250 | **387** |
/// | [10240,320] | **585** | 79 | 270 |
/// | [2560,640]  | **493** | 143 | 389 |
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
    // GB/s 就是这个形状的实用地板（见 PLAN §P15-⑲）。核与 launch-bench 对照保留。
    if use_skinny_row(kdim) {
        k.gemm_skinny_row_bf16_ptr(wp, xp, yp, n, kdim, m, epi, a, b)
    } else {
        k.gemm_skinny_bf16_ptr(wp, xp, yp, n, kdim, m, epi, a, b)
    }
}

/// 投影：**`t==1`（decode）走自研 GEMV，`t>1` 走自研瘦长 GEMM。一律不用 cuBLAS。**
///
/// 为什么连 t>1 也不给 cuBLAS：
/// 1. `gemm_ex` 每次调用约 **26 µs CPU 开销**（普通 launch 只要 ~6.6 µs）；
/// 2. 更要命的是它把形状摆成 `m=n, n=t` —— **cuBLAS 面对 `N=t` 这种极窄 N 会选到病态内核**。
///    实测 MoE `shared_gate_up` 单层 **50 µs（t=1）→ 500 µs（t=2）**，48 层 = +22.5 ms（§3.45）。
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
        // 任意 t ≥ 2：按 ≤8 行一片调瘦长 GEMM（权重只读一遍）
        let (wp, _a) = wv.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        let mut done = 0usize;
        while done < t {
            let m = (t - done).min(SKINNY_MAX_M);
            skinny_call(
                k, wp,
                xp + (done * kdim * 2) as u64,
                yp + (done * n * 2) as u64,
                n, kdim, m, 0, 0.0, 0.0,
            )?;
            done += m;
        }
        Ok(())
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
        // T>1 的小批量也走瘦长 GEMM（含同样的尾融合）；t>8 时按 8 行切片
        let (wp, _a) = wv.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        let mut done = 0usize;
        while done < t {
            let m = (t - done).min(SKINNY_MAX_M);
            skinny_call(
                k, wp,
                xp + (done * kdim * 2) as u64,
                yp + (done * n * 2) as u64,
                n, kdim, m, epi, a, b,
            )?;
            done += m;
        }
        Ok(())
    } else {
        // `epi == 4`（swiglu）是本核唯一不支持的一档，拆成"瘦长 GEMM + swiglu"两步。
        //
        // ⚠️ GEMM **绝不能交给 cuBLAS**（曾经就是）：`gemm_bf16_nt` 把形状摆成 `m=n, n=t`，
        // t>1 时 cuBLAS 看到的是 `N=t` 这种**极窄 N**——它在 N=2 上会选到病态内核。
        // 实测 MoE shared expert 的 `shared_gate_up`（唯一的 epi=4 热路径）单层从
        // **t=1 的 50 µs 暴涨到 t=2 的 500 µs**，48 层 = **+22.5 ms**，是 §3.45 里
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
    let mut done = 0usize;
    while done < t {
        let m = (t - done).min(SKINNY_MAX_M);
        k.gemm_skinny_row_f32x_ptr(
            wp, xp + (done * kdim * 4) as u64, yp + (done * n * 2) as u64,
            n, kdim, m, epi, a, b,
        )?;
        done += m;
    }
    Ok(())
}

/// 便捷入口：`t>1` 且开关开 ⇒ 预转 x 为 fp32 并返回；否则 `None`（调用方走原 bf16 路）。
/// ★ **默认关**（P15-⑨ 整机实测：微基准 m=8 带宽 350→700 GB/s，但整机 fwd t=7 **+2.6 ms**
///   净亏——x fp32 让每 block 的 x L2 读翻倍 + 每处一次 convert launch）。
///   设 `LINGXI_F32X_PROJ=1` 才启用；`LINGXI_F32X_SITES` 可**按位**关掉某几处做二分定位。
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

/// **Q4 稠密权重**投影（decode `t==1` 走融合反量化 GEMV；`t>1` 走**分块 Q4 瘦长 GEMM**）。
/// `row_off` 用于"同一块 Q4 矩阵的某几行"（例如 `query_key_value_z` 的 qkv 段与 z 段）。
#[allow(clippy::too_many_arguments)]
pub fn proj_q4_epi(
    k: &Kernels, x: &CudaSlice<bf16>, wq: &crate::q4::Q4W,
    y: &mut CudaSlice<bf16>, n: usize, kdim: usize, row_off: usize, t: usize,
    epi: i32, a: f32, b: f32,
) -> Result<()> {
    // t>1：**直接吃 Q4** 的分块瘦长 GEMM（修 §3.21 根因 1）。
    // 旧路是"整块解量化成 bf16 再 GEMM"（qkvz 从 21 MB 变 105 MB 的往返）。
    if t > 1 && epi != 4 && kdim % 8 == 0 && !crate::flag("LINGXI_NO_Q4GEMM") {
        let (wp, _a) = wq.packed.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        let mut done = 0usize;
        while done < t {
            let m = (t - done).min(SKINNY_MAX_M);
            k.gemm_q4_skinny_bf16_ptr(
                wp,
                xp + (done * kdim * 2) as u64,
                yp + (done * n * 2) as u64,
                n, kdim, wq.k_pad, m, row_off, wq.scale_off, epi, a, b,
            )?;
            done += m;
        }
        return Ok(());
    }
    if t == 1 {
        let (wp, _a) = wq.packed.device_ptr(&k.stream);
        let (xp, _b) = x.device_ptr(&k.stream);
        let (yp, _c) = y.device_ptr_mut(&k.stream);
        k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, n, kdim, wq.k_pad, row_off, wq.scale_off, epi, a, b)
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
                 生产配置：export LINGXI_Q4_CACHE=/dev/shm/lingxi-q4cache（见 HANDOVER §2 / bench.sh）"
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
