//! GR（GatedResidual / hyper-connection）前向。
//!
//! 数学严格对齐 HF `Qwen4ExpTextGatedResidual.forward`：
//! `text
//! xn = hc_norm(hyper_input) # 分组 RMSNorm（gamma 已烘焙）
//! d = mix_down(xn) / hc # [lowrank]
//! imw = sigmoid(mix_up(silu(d))) # [hc*hidden]
//! mixed = mean_g(imw[g*H+i] * xn[g*H+i]) # [hidden]
//! injw = 2 * sigmoid(block_inject(xn) / hc) # [hc]
//! `

use anyhow::{Context as _, Result};
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

use crate::gpu::proj_t1_epi;
use crate::graph;
use crate::scratch;

/// 一个 GR 块的权重（都已烘焙 / 无需运行时转换）。
pub struct GrWeights {
    /// [dim] 分组归一化 gamma
    pub hc_norm: CudaSlice<bf16>,
    /// [lowrank, dim]
    pub mix_down: CudaSlice<bf16>,
    /// [dim, lowrank]
    pub mix_up: CudaSlice<bf16>,
    /// [hc, dim]；`None` = `use_combine=False`（末尾 `hyper_connection_mixer` 即此形态）
    pub block_inject: Option<CudaSlice<bf16>>,
}

/// 返回 `(mixed [rows, hidden], injection_weights [rows, hc] 或 None)`。
///
/// 便利包装：自己向 scratch 池取输出缓冲。**不要**在 CUDA Graph 捕获区里用它
/// （池的轮转会打乱烘死的地址）——捕获路径请用 [`gated_residual_forward_into`]。
pub fn gated_residual_forward(
    k: &Kernels,
    w: &GrWeights,
    x: &CudaSlice<bf16>,
    rows: u32,
    dim: usize,
    hidden: usize,
    hc: usize,
    lowrank: usize,
    eps: f32,
) -> Result<(CudaSlice<bf16>, Option<CudaSlice<bf16>>)> {
    let r = rows as usize;
    let mut mixed = scratch::bf16(k, r * hidden)?;
    let mut injw = if w.block_inject.is_some() { Some(scratch::bf16(k, r * hc)?) } else { None };
    gated_residual_forward_into(
        k, w, x, &mut mixed, injw.as_mut(), rows, dim, hidden, hc, lowrank, eps,
        graph::NO_CAPTURE,
    )?;
    Ok((mixed, injw))
}

/// 同上，但**输出缓冲由调用方提供**，且内部临时缓冲走 `(gkey, slot)` 私有固定区。
///
/// 这样整段 kernel 序列就满足 CUDA Graph 的前提（地址稳定），可以直接录进图里重放。
/// `gkey == 0` 表示"不录图"（仍用固定私有缓冲，只是不捕获）。
#[allow(clippy::too_many_arguments)]
pub fn gated_residual_forward_into(
    k: &Kernels,
    w: &GrWeights,
    x: &CudaSlice<bf16>,
    out_mixed: &mut CudaSlice<bf16>,
    mut out_injw: Option<&mut CudaSlice<bf16>>,
    rows: u32,
    dim: usize,
    hidden: usize,
    hc: usize,
    lowrank: usize,
    eps: f32,
    gkey: u64,
) -> Result<()> {
    let r = rows as usize;
    // ⚠️ **图形必须按「(block, 行数)」分命名空间**：私有缓冲的长度是 `r * dim`，不同 t 长度不同，
    // 共用同一 key 会缺缓冲（实测报 `key=1 slot=0 len=10240（现存 51200）`）。
    // 这里给每个 `rows` 一个独立段（`(1+rows) << 40`），于是**每个 t 都能各自录一张图**。
    // 【本轮修正】原来只对 `rows==1` 开捕获 —— `LINGXI_KTS` 实测那正是 **t=2 悬崖的全部**：
    // GR 块从 **4 µs（t=1，一次图重放）暴涨到 235 µs（t=2，十次裸 launch）**，
    // 每层 +231 µs × 48 = **+11 ms/token**。现在 t>1 也录图（按 t 分键 ⇒ 形状恒定，可录）。
    // ★ P15-⑩（2026-10-17）：**所有 rows 统一分键**（原来 rows==1 豁免）。
    // 根因：spec 下 t=1 草稿链与 t=7 验证批交替，rows==1 与 rows==7 共用同一 gkey 的
    // 固定槽（staged=x 槽 8 等）⇒ rows=7 的 take_fixed(8, 7*dim) 会把 rows==1 图录死的
    // staged 块**重新分配覆盖**（旧块 drop 还给驱动）⇒ rows==1 图重放读已释放内存 =
    // 静默数值错（实测 spec+graph 接受率 4.14→0.17）。分键后各 rows 的槽相互隔离。
    let gkey = if gkey == graph::NO_CAPTURE {
        gkey
    } else {
        gkey.wrapping_add((1u64 + rows as u64) << 40)
    };
    // `LINGXI_NO_GR_GRAPH=1`：只禁 GR 的图（二分"spec+graph 数值错"的罪魁用）。
    // `LINGXI_GR_PROF=1` 时也**不录图**：`cuEventRecord` 不能出现在 capture 区里
    // （会 STREAM_CAPTURE_UNJOINED），而探针本身要逐核事件。
    let capture = graph::enabled()
        && gkey != graph::NO_CAPTURE
        && !crate::flag("LINGXI_NO_GR_GRAPH")
        && !crate::gr_prof::on();
    let take = |slot: usize, len: usize| -> Result<CudaSlice<bf16>> {
        graph::take_fixed(k, gkey, slot, len)
    };

    // ── 融合路径（decode `t==1`）：5 个 kernel 压到 2 个 ──
    // 见 `cuda/gr_ops.cu` 的 `lingxi_gr_down_fuse_bf16` / `lingxi_gr_up_mix_bf16`。
    if rows == 1 && crate::flag("LINGXI_GR_FUSE") && w.block_inject.is_some() {
        let ngroups = dim / hidden;
        let inv_hc = 1.0 / (hc as f32);
        let mut d = take(1, lowrank)?;
        let mut rmsg = graph::take_fixed_f32(k, gkey, 9, ngroups)?;
        let injw = out_injw
            .as_deref_mut()
            .ok_or_else(|| anyhow::anyhow!("GR 融合路径需要 block_inject 输出缓冲"))?;
        k.gr_down_fuse_bf16(
            x, &w.hc_norm, &w.mix_down, w.block_inject.as_ref().unwrap(),
            &mut d, injw, &mut rmsg,
            dim, hidden, ngroups, lowrank, hc, inv_hc, eps,
        ).context("gr: down_fuse")?;
        k.gr_up_mix_bf16(
            &d, x, &w.hc_norm, &w.mix_up, &rmsg, out_mixed,
            dim, hidden, lowrank, ngroups, inv_hc,
        ).context("gr: up_mix")?;
        graph::give_fixed(k, gkey, 1, d);
        graph::give_fixed_f32(k, gkey, 9, rmsg);
        return Ok(());
    }
    // ── 通用档融合（rows 2..=8）：5 发 → 3 发 ──
    // 见 `cuda/gr_ops.cu` 的 `lingxi_gr_down_inject_bf16` / `lingxi_gr_up_mix_fuse_bf16`。
    // 逐位等价（同一个 per-列/per-行的累加次序与归约树、同样保留 imw 的 bf16 舍入点），
    // 微基准 `t9_gr_shape_bw` 在 rows=5/6/7 上对拍 0 不一致。
    // 门控：`LINGXI_NO_GR_FUSE2=1` 回退；`LINGXI_F32X_PROJ` 开时走旧路（保住 f32x 的既有语义）。
    let fuse2 = (2..=8).contains(&rows)
        && w.block_inject.is_some()
        && !crate::flag("LINGXI_NO_GR_FUSE2")
        && !crate::flag("LINGXI_F32X_PROJ")
        && !crate::flag("LINGXI_KTS");
    // ⚠️ **图的输入必须来自私有缓冲**：图把设备指针烘死，而 `x`（残差流）是从 scratch
    // 池按**同尺寸 LIFO** 借的 —— 一个 token 里 `put(旧 hyper)`/`hyper=next` 的轮转会让
    // 各层看到的两块缓冲奇偶交替（实测 layer 47 的 `hyper` 地址每 token 在
    // `0x8e6f56a00`/`0x8e6f4ca00` 之间摆），录进去的图就会读到别的层的数据。
    // 所以在**图外面**先把 x 拷进私有暂存块，图里只认这块固定地址。
    let mut staged = if capture { Some(take(8, r * dim)?) } else { None };
    if let Some(s) = staged.as_mut() {
        k.stream
            .memcpy_dtod(x, s)
            .context("gr: staged memcpy_dtod")?;
    }
    let x: &CudaSlice<bf16> = staged.as_ref().unwrap_or(x);
    let mut body = || -> Result<()> {
        // `LINGXI_KTS=1`（且非捕获路径）时给 5 个算子逐个插同步括号，定位 GR 块内部开销。
        // 捕获中不能插同步，所以只在 !capture（配合 LINGXI_NO_GRAPHS=1 使用）时生效。
        let kts = crate::flag("LINGXI_KTS") && !capture;
        let mut tk = [0f64; 5];
        let mut ta = std::time::Instant::now();
        let mut xn = take(0, r * dim)?;
        k.grouped_rmsnorm_bf16(x, &w.hc_norm, &mut xn, rows, dim, hidden, eps)
            .context("gr: rmsnorm")?;
        crate::gr_prof::mark(k, 1)?;
        if fuse2 {
            // 3 发：rmsnorm（上面）→ down+inject → up+mix
            let inv_hc = 1.0 / (hc as f32);
            let wi = w.block_inject.as_ref().expect("fuse2 要求 block_inject");
            let injw = out_injw
                .as_deref_mut()
                .ok_or_else(|| anyhow::anyhow!("GR 融合档需要 block_inject 输出缓冲"))?;
            let mut d = take(1, r * lowrank)?;
            k.gr_down_inject_bf16(
                &w.mix_down, wi, &xn, &mut d, injw,
                lowrank, hc, dim, r, inv_hc,
            ).context("gr: down+inject fuse")?;
            crate::gr_prof::mark(k, 2)?;
            let opb = if hidden % 2 == 0 { 2 } else { 1 };
            k.gr_up_mix_fuse_bf16(
                &w.mix_up, &d, &xn, out_mixed, lowrank, hidden, hc, r, opb,
            ).context("gr: up+mix fuse")?;
            crate::gr_prof::mark(k, 3)?;
            graph::give_fixed(k, gkey, 0, xn);
            graph::give_fixed(k, gkey, 1, d);
            return Ok(());
        }
        if kts {
            k.stream.synchronize()?;
            tk[0] = ta.elapsed().as_secs_f64() * 1e6;
            ta = std::time::Instant::now();
        }

        // d = mix_down(xn) / hc → silu（尾融合进 GEMV）
        let mut d = take(1, r * lowrank)?;
        // ★ P15-⑨：t>1 时 xn/d 预转 fp32（mix_down+inject 共用 xn 一份、mix_up 用 d 一份）
        let xn_f = crate::gpu::prep_f32x(k, &xn, r, r * dim, 4)?;  // site4=xn 系（mix_down+inject）
        match xn_f.as_ref() {
            Some(xf) => crate::gpu::proj_f32x(k, &w.mix_down, xf, &mut d, lowrank, dim, r, 1, 1.0 / (hc as f32), 0.0)
                .context("gr: mix_down f32x")?,
            None => proj_t1_epi(k, &xn, &w.mix_down, &mut d, lowrank, dim, r, 1, 1.0 / (hc as f32), 0.0)
                .context("gr: mix_down")?,
        }
        if kts {
            k.stream.synchronize()?;
            tk[1] = ta.elapsed().as_secs_f64() * 1e6;
            ta = std::time::Instant::now();
        }

        crate::gr_prof::mark(k, 2)?;
        // imw = sigmoid(mix_up(d))
        let mut imw = take(2, r * dim)?;
        // ★ P15-⑨ **未解之谜**：mix_up 的 f32x（[20480,320] epi=2）在 `f32x-check` 里
        // 与模板版 skinny **0 位不同**，但整机 talk 文本**确定性漂移**（site8 单开即复现）。
        // 保守起见不启用；mix_down/inject（k=dim=20480）已整机验证等价。
        proj_t1_epi(k, &d, &w.mix_up, &mut imw, dim, lowrank, r, 2, 1.0, 1.0)
            .context("gr: mix_up")?;
        if kts {
            k.stream.synchronize()?;
            tk[2] = ta.elapsed().as_secs_f64() * 1e6;
            ta = std::time::Instant::now();
        }

        crate::gr_prof::mark(k, 3)?;
        // mixed = mean_g(imw * xn)（一次覆盖所有行）
        k.mix_mean_bf16(&imw, &xn, out_mixed, hc, hidden, rows)
            .context("gr: mix_mean")?;
        if kts {
            k.stream.synchronize()?;
            tk[3] = ta.elapsed().as_secs_f64() * 1e6;
            ta = std::time::Instant::now();
        }

        crate::gr_prof::mark(k, 4)?;
        // injw = 2 * sigmoid(block_inject(xn) / hc)
        if let (Some(w_inj), Some(dst)) = (&w.block_inject, out_injw.as_deref_mut()) {
            match xn_f.as_ref() {
                Some(xf) => crate::gpu::proj_f32x(k, w_inj, xf, dst, hc, dim, r, 2, 1.0 / (hc as f32), 2.0)
                    .context("gr: inject f32x")?,
                None => proj_t1_epi(k, &xn, w_inj, dst, hc, dim, r, 2, 1.0 / (hc as f32), 2.0)
                    .context("gr: inject")?,
            }
        }
        if kts {
            k.stream.synchronize()?;
            tk[4] = ta.elapsed().as_secs_f64() * 1e6;
            use std::sync::atomic::{AtomicU32, Ordering};
            static GRN: AtomicU32 = AtomicU32::new(0);
            if GRN.fetch_add(1, Ordering::Relaxed) < 200 {
                eprintln!(
                    "[kts-gr] rows={rows} norm {:.1} | mix_down {:.1} | mix_up {:.1} | mix_mean {:.1} | inject {:.1} µs（合计 {:.1}）",
                    tk[0], tk[1], tk[2], tk[3], tk[4], tk.iter().sum::<f64>()
                );
            }
        }

        crate::gr_prof::mark(k, 5)?;

        graph::give_fixed(k, gkey, 0, xn);
        graph::give_fixed(k, gkey, 1, d);
        graph::give_fixed(k, gkey, 2, imw);
        Ok(())
    };
    let r_graph = if capture {
        graph::run_cached(k, gkey, body)
    } else {
        body()
    };
    if let Some(s) = staged {
        graph::give_fixed(k, gkey, 8, s);
    }
    r_graph
}
