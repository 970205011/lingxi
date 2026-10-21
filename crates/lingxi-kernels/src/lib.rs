//! `lingxi-kernels` — L1：CUDA C++ → cubin（build.rs 调 nvcc）+ cudarc 加载/launch。
//!
//! 职责边界（见 `ARCHITECTURE.md` §3）：
//!   * 纯函数 kernel：`(指针 + 参数) → 输出`，无全局态、无隐式 stream 依赖。
//!   * `unsafe` 只允许出现在本 crate 的 launch 封装里。
//!   * 矩阵乘**复用 cuBLAS**（§0.5 复用铁律）；只自研 cuBLAS 不适合的融合/稀疏/注意力算子。

use std::sync::Arc;

pub mod evprof;
use evprof::EvProf;

use anyhow::Result;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr,
    DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use lingxi_ids::{KernelId, KernelRegistry, OpId, ShapeTag};

/// 目标架构（固定 sm_70）。
pub const CUDA_ARCH: &str = "sm_70";

pub static SMOKE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smoke.cubin"));
pub static RMSNORM_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.cubin"));
pub static GR_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gr_ops.cubin"));
pub static MOE_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_ops.cubin"));
pub static ATTN_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_ops.cubin"));
pub static DEQUANT_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dequant_ops.cubin"));
pub static GDN_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_ops.cubin"));
pub static TOPK_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/topk_ops.cubin"));
/// ㊼ Q4 专家张量核 GEMM（Volta m8n8k4）
pub static Q4_TC_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_tc.cubin"));
/// ㊽ Q4 专家张量核 GEMM —— prefill 大 tile 版（M 轴分块 + x 进 shared）
pub static Q4_TCP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_tcp.cubin"));

const BLOCK: u32 = 256;

/// 把只读 mmap 区域整段 pin 住（`cuMemHostRegister_v2` + `CU_MEMHOSTREGISTER_READ_ONLY`）。
///
/// **必须用 READ_ONLY**：对 `PROT_READ` 映射，`cudaHostRegisterDefault` 必返回
/// `cudaErrorInvalidValue`，零拷贝直读会**静默**失效（ninfer-dev `LINGXI_CHANGES.md` C8 实测：
/// Default=invalid argument / ReadOnly=no error；修复后单流 decode 8 → 21~26 tok/s）。
///
/// `Ok(true)` = 注册成功，GPU 可直接 DMA 该区域（省掉每层 miss 的 host memcpy）；
/// `Ok(false)` = 驱动拒绝，调用方**必须回退** gather 路径，不得视为致命错误。
pub fn pin_host_readonly(ptr: *const u8, bytes: usize) -> bool {
    use cudarc::driver::sys;
    if ptr.is_null() || bytes == 0 {
        return false;
    }
    const PAGE: usize = 4096;
    let start = (ptr as usize) & !(PAGE - 1);
    let end = (ptr as usize + bytes + PAGE - 1) & !(PAGE - 1);
    // SAFETY: 调用方保证 [ptr, ptr+bytes) 是已映射的只读页（mmap arena 数据区）；
    // 这里只把地址区间交给驱动注册，不解引用、不写入。
    let r = unsafe {
        sys::cuMemHostRegister_v2(
            start as *mut std::ffi::c_void,
            end - start,
            sys::CU_MEMHOSTREGISTER_READ_ONLY,
        )
    };
    r == sys::CUresult::CUDA_SUCCESS
}

/// 与 [`pin_host_readonly`] 同样的注册，但用 `CU_MEMHOSTREGISTER_PORTABLE`（**要求 VMA 可写**）。
///
/// 为什么要多这一条（2026-10-16，PLAN §P13.9b）：READ_ONLY 注册的语义是"**让设备去读**这块 host
/// 内存"（zero-copy 映射），**不是"把页锁好交给 DMA"**。实测后果有两条，都致命：
/// * 从它发起的 H2D 只有 **8.6 GB/s**；
/// * 它**与 kernel 完全不可重叠** —— 把拷贝挪到第二条流，专家段的"每 miss 增量"一点没变
///   （`miss=0/1/2/≥3` 的层均：0.61/0.91/1.15/2.31 ms，开关第二流前后一模一样）。
/// 而 `DEFAULT/PORTABLE` 要求 VMA 可写（对 `PROT_READ` 映射必返回 `invalid argument`）⇒
/// 必须先拿一份**可写**映射（`expert_arena::mmap_shared_rw`）才谈得上 DMA 注册。
pub fn pin_host_dma(ptr: *const u8, bytes: usize) -> bool {
    use cudarc::driver::sys;
    if ptr.is_null() || bytes == 0 {
        return false;
    }
    const PAGE: usize = 4096;
    let start = (ptr as usize) & !(PAGE - 1);
    let end = (ptr as usize + bytes + PAGE - 1) & !(PAGE - 1);
    // SAFETY: 同 `pin_host_readonly`：只把已映射区间交给驱动注册，不解引用、不写入。
    let r = unsafe {
        sys::cuMemHostRegister_v2(
            start as *mut std::ffi::c_void,
            end - start,
            sys::CU_MEMHOSTREGISTER_PORTABLE,
        )
    };
    r == sys::CUresult::CUDA_SUCCESS
}

/// 加载后的 kernel 句柄集合 + cuBLAS 句柄。构造一次，之后只读复用。
pub struct Kernels {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// **专家段的第二条流**（`LINGXI_MOE_OVERLAP=1`，见 `docs/PLAN.md` §P4）。
    ///
    /// 为什么要它：专家 miss 的 blob H2D（2.49 MB/个，每轮 ~280 MB ≈ **28 ms/轮**）现在**串在
    /// compute 流上**，而它其实只被"未命中那几个组的 GEMV"需要 —— 命中组（每层 ~95%）完全
    /// 可以先算。把 miss 的 H2D 放到这条流上，compute 流先跑命中组，两边**并行**。
    /// 计量口径：`LINGXI_ARENA_NOH2D=1` 实测剥掉这 28 ms 后 40.9 → 48.4 tok/s。
    pub stream2: Arc<CudaStream>,
    /// 跨流顺序的门：`load_hits` 开始时在 compute 流上记一次、`stream2` 等它，
    /// 保证"miss 覆写某个槽"一定发生在"更早那些读该槽的 GEMV"之后（槽复用安全）。
    pub ev_mo_pre: cudarc::driver::CudaEvent,
    /// `stream2` 上"本层所有 miss 的 blob 都拷完"的标记；compute 流在跑**未命中组**之前等它。
    pub ev_mo_miss: cudarc::driver::CudaEvent,
    /// **逐段真 GPU 计时**（`LINGXI_EVPROF=1`）：见 [`evprof`]。默认 `None`（零开销）。
    pub evprof: Option<EvProf>,
    pub blas: CudaBlas,
    smoke_scale: CudaFunction,
    rmsnorm_f32: CudaFunction,
    grouped_rmsnorm_bf16: CudaFunction,
    gr_down_fuse_bf16: CudaFunction,
    gr_up_mix_bf16: CudaFunction,
    silu_scale_inplace_bf16: CudaFunction,
    sigmoid_scale_mul_inplace_bf16: CudaFunction,
    mix_mean_bf16: CudaFunction,
    gr_write_inplace_bf16: CudaFunction,
    hc_repeat_bf16: CudaFunction,
    mul_bf16: CudaFunction,
    swiglu_bf16: CudaFunction,
    gather_rows_bf16: CudaFunction,
    scatter_add_rows_bf16: CudaFunction,
    row_scale_inplace_bf16: CudaFunction,
    moe_combine_rows_bf16: CudaFunction,
    moe_combine_add_rows_bf16: CudaFunction,
    /// ㊾ CSR 版合并+残差（batch 路径，O(rows×H×top_k) 取代旧核 O(rows×H×S)）
    moe_combine_add_csr_bf16: CudaFunction,
    gemv_bf16: CudaFunction,
    gemv_bf16_epi: CudaFunction,
    gemv_bf16_block_epi: CudaFunction,
    gemv_bf16_row8_epi: CudaFunction,
    /// ㊼ Q4 专家张量核 GEMM（T≤8 / T=9..16），`LINGXI_MOE_TC=1` 才用
    q4_tc_t1: CudaFunction,
    q4_tc_t2: CudaFunction,
    /// ㊽ prefill 大 tile 版（任意组大小一次 launch），`LINGXI_MOE_TC=1` 门控
    q4_tcp: CudaFunction,
    /// 瘦长 GEMM，按 m=1..8 各一份（索引 `m-1`）。必须按 m 特化：
    /// 固定开 8 个累加器的版本在 m=1 时只有 215 GB/s，而 GEMV 有 724 GB/s（指令发射受限）。
    gemm_skinny: [CudaFunction; 8],
    gemm_skinny_row: CudaFunction,
    gemm_skinny_row_multi: CudaFunction,
    gemm_skinny_row_f32x: CudaFunction,
    gemm_q4_skinny_bf16: CudaFunction,
    gemm_q4_skinny_dbg: CudaFunction,
    ple_gate_bf16: CudaFunction,
    add_bf16: CudaFunction,
    rope_partial_inplace_bf16: CudaFunction,
    attn_masked_bf16: CudaFunction,
    attn2_masked_bf16: CudaFunction,
    attn2s_sparse_bf16: CudaFunction,
    attn3_masked_bf16: CudaFunction,
    attn5_sparse_bf16: CudaFunction,
    attn2_masked_kvq8_bf16: CudaFunction,
    attn2s_sparse_kvq8_bf16: CudaFunction,
    attn3_masked_kvq8_bf16: CudaFunction,
    attn5_sparse_kvq8_bf16: CudaFunction,
    quant_rows_e4m3_bf16: CudaFunction,
    qsa_pool_block_bf16: CudaFunction,
    qsa_pool_blocks_bf16: CudaFunction,
    rope_partial_stride_bf16: CudaFunction,
    qsa_indexer_scores_bf16: CudaFunction,
    qsa_indexer_rank_f32: CudaFunction,
    qsa_keep_select_bf16: CudaFunction,
    qsa_keep_fill_bf16: CudaFunction,
    qsa_keep_fused_bf16: CudaFunction,
    qsa_keep_from_scores_bf16: CudaFunction,
    qsa_sel_build: CudaFunction,
    qsa_hist_f32: CudaFunction,
    qsa_bin_finalize: CudaFunction,
    qsa_keep_fill_scores: CudaFunction,
    attn_gather_bf16: CudaFunction,
    attn_gather4_bf16: CudaFunction,
    qsa_mask_fill_f32: CudaFunction,
    dequant_rowsplit_q4g64_bf16: CudaFunction,
    dequant_rowsplit_q4g64_batch: CudaFunction,
    dequant_rowsplit_q3g128_bf16: CudaFunction,
    gemv_q4g64_slots_bf16: CudaFunction,
    gemv_q4g64_slots_r2_bf16: CudaFunction,
    gemv_q4g64_slots_r4_bf16: CudaFunction,
    gemv_q4g64_slots_r8_bf16: CudaFunction,
    gemv_q4g64_slotsm_bf16: CudaFunction,
    gemv_q4g64_slotsm_r2_bf16: CudaFunction,
    gemv_q4g64_slotsm_r4_bf16: CudaFunction,
    gemv_q3g128_slots_bf16: CudaFunction,
    gemv_q3g128_slots_r2_bf16: CudaFunction,
    gemv_q3g128_slots_r4_bf16: CudaFunction,
    gemv_q3g128_slots_r8_bf16: CudaFunction,
    gemv_q3g128_slotsm_bf16: CudaFunction,
    gemv_q3g128_slotsm_r2_bf16: CudaFunction,
    gemv_q3g128_slotsm_r4_bf16: CudaFunction,
    gemv_q4g64_slotsm_dp4a: CudaFunction,
    gemv_q4g64_slotsm_r2_dp4a: CudaFunction,
    convert_bf16_f32: CudaFunction,
    convert_bf16_q8_rs: CudaFunction,
    gemv_q4g64_dense_epi: CudaFunction,
    gemv_q4g64_dense4_epi: CudaFunction,
    dequant_q4g64_dense_bf16: CudaFunction,
    causal_conv1d_silu_bf16: CudaFunction,
    causal_conv1d_silu_hist_bf16: CudaFunction,
    gdn_recurrent_bf16: CudaFunction,
    argmax_rows_part_bf16: CudaFunction,
    argmax_rows_fin: CudaFunction,
    topk_rows_part_bf16: CudaFunction,
    topk_rows_fin: CudaFunction,
    route_topk_slots_bf16: CudaFunction,
}

impl Kernels {
    pub fn load(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        // ── 两条**可选**实验开关（默认关闭 = 保持 cudarc 原始语义）──────────────
        // 实测两者对 decode 都**无影响**（0.049→0.049 s/token），故默认不启用；
        // 留作后续排查/多 stream 场景的开关。
        // `LINGXI_DISABLE_EVENT_TRACKING=1`：cudarc 默认每次 `alloc` 都会 cuEventCreate×2。
        if std::env::var("LINGXI_DISABLE_EVENT_TRACKING").is_ok() {
            unsafe { ctx.disable_event_tracking() };
        }
        // `LINGXI_SCHED_SPIN=1`：把 `cuStreamSynchronize` 的调度策略从 AUTO 改成自旋。
        if std::env::var("LINGXI_SCHED_SPIN").is_ok() {
            let _ = ctx.set_flags(cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_SPIN);
        }
        // 用**真 stream**（非 legacy NULL stream）：`cuStreamBeginCapture` 不允许捕获默认流，
        // 而 CUDA Graph 是本项目下一步的主线（把"每层 30 个 kernel 被逐层同步切碎"合并掉）。
        // 单 stream 顺序语义不变 → 数值不变。
        // 换 stream 后 cudarc 会进入 multi-stream 模式，若 event tracking 仍开着，
        // 每次 `.arg(&CudaSlice)` 都要 push/record 事件 → 必须显式关掉（单 stream 无需跨流记账）。
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        // 第二条流 + 两个跨流事件（`LINGXI_MOE_OVERLAP`，见 `stream2` 的说明）。
        let stream2 = ctx.new_stream()?;
        let ev_mo_pre = ctx.new_event(None)?;
        let ev_mo_miss = ctx.new_event(None)?;
        // `LINGXI_EVPROF=1`：建逐段计时事件对（320 对，一次性）。
        let evprof = if std::env::var("LINGXI_EVPROF").is_ok() {
            Some(EvProf::new(&ctx)?)
        } else {
            None
        };
        let smoke = ctx.load_module(Ptx::from_binary(SMOKE_CUBIN.to_vec()))?;
        let rmsnorm = ctx.load_module(Ptx::from_binary(RMSNORM_CUBIN.to_vec()))?;
        let gr = ctx.load_module(Ptx::from_binary(GR_OPS_CUBIN.to_vec()))?;
        let moe = ctx.load_module(Ptx::from_binary(MOE_OPS_CUBIN.to_vec()))?;
        let attn = ctx.load_module(Ptx::from_binary(ATTN_OPS_CUBIN.to_vec()))?;
        let dq = ctx.load_module(Ptx::from_binary(DEQUANT_OPS_CUBIN.to_vec()))?;
        let gdn = ctx.load_module(Ptx::from_binary(GDN_OPS_CUBIN.to_vec()))?;
        let topk = ctx.load_module(Ptx::from_binary(TOPK_OPS_CUBIN.to_vec()))?;
        let q4tc = ctx.load_module(Ptx::from_binary(Q4_TC_CUBIN.to_vec()))?;
        let q4tcp = ctx.load_module(Ptx::from_binary(Q4_TCP_CUBIN.to_vec()))?;
        let blas = CudaBlas::new(stream.clone())?;
        // 瘦长 GEMM 按 m=1..8 各加载一份（必须特化，见 `gemm_skinny` 字段注释）
        let mut skinny_vec: Vec<CudaFunction> = Vec::with_capacity(8);
        for i in 1..=8u32 {
            skinny_vec.push(moe.load_function(&format!("lingxi_gemm_skinny_m{i}"))?);
        }
        let gemm_skinny: [CudaFunction; 8] = skinny_vec
            .try_into()
            .map_err(|_| anyhow::anyhow!("瘦长 GEMM 内核加载数量不对（应 8 个）"))?;
        Ok(Self {
            ctx,
            stream,
            stream2,
            ev_mo_pre,
            ev_mo_miss,
            evprof,
            smoke_scale: smoke.load_function("lingxi_smoke_scale")?,
            rmsnorm_f32: rmsnorm.load_function("lingxi_rmsnorm_f32")?,
            grouped_rmsnorm_bf16: gr.load_function("lingxi_grouped_rmsnorm_bf16")?,
            gr_down_fuse_bf16: gr.load_function("lingxi_gr_down_fuse_bf16")?,
            gr_up_mix_bf16: gr.load_function("lingxi_gr_up_mix_bf16")?,
            silu_scale_inplace_bf16: gr.load_function("lingxi_silu_scale_inplace_bf16")?,
            sigmoid_scale_mul_inplace_bf16: gr.load_function("lingxi_sigmoid_scale_mul_inplace_bf16")?,
            mix_mean_bf16: gr.load_function("lingxi_mix_mean_bf16")?,
            gr_write_inplace_bf16: gr.load_function("lingxi_gr_write_inplace_bf16")?,
            hc_repeat_bf16: gr.load_function("lingxi_hc_repeat_bf16")?,
            mul_bf16: gr.load_function("lingxi_mul_bf16")?,
            swiglu_bf16: moe.load_function("lingxi_swiglu_bf16")?,
            gather_rows_bf16: moe.load_function("lingxi_gather_rows_bf16")?,
            scatter_add_rows_bf16: moe.load_function("lingxi_scatter_add_rows_bf16")?,
            row_scale_inplace_bf16: moe.load_function("lingxi_row_scale_inplace_bf16")?,
            moe_combine_rows_bf16: moe.load_function("lingxi_moe_combine_rows_bf16")?,
            moe_combine_add_rows_bf16: moe.load_function("lingxi_moe_combine_add_rows_bf16")?,
            moe_combine_add_csr_bf16: moe.load_function("lingxi_moe_combine_add_csr_bf16")?,
            gemv_bf16: moe.load_function("lingxi_gemv_bf16")?,
            gemv_bf16_epi: moe.load_function("lingxi_gemv_bf16_epi")?,
            gemv_bf16_block_epi: moe.load_function("lingxi_gemv_bf16_block_epi")?,
            gemv_bf16_row8_epi: moe.load_function("lingxi_gemv_bf16_row8_epi")?,
            q4_tc_t1: q4tc.load_function("lingxi_gemv_q4g64_slots_tc_bf16")?,
            q4_tc_t2: q4tc.load_function("lingxi_gemv_q4g64_slots_tc2_bf16")?,
            q4_tcp: q4tcp.load_function("lingxi_gemv_q4g64_slots_tcp_bf16")?,
            gemm_skinny,
            gemm_skinny_row: moe.load_function("lingxi_gemm_skinny_row_bf16")?,
            // P15-⑲：多行/块版要 64KB 动态 smem（m≤8 × 2048 × 4B），超过 48KB 默认上限
            // ⇒ 加载时抬高 MAX_DYNAMIC_SHARED_SIZE_BYTES（V100 上限 96KB）。
            gemm_skinny_row_multi: {
                let f = moe.load_function("lingxi_gemm_skinny_row_multi_bf16")?;
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    (8 * 2112 * 4) as i32,
                )?;
                f
            },
            gemm_skinny_row_f32x: moe.load_function("lingxi_gemm_skinny_row_f32x_bf16w")?,
            gemm_q4_skinny_bf16: dq.load_function("lingxi_gemm_q4_skinny_bf16")?,
            gemm_q4_skinny_dbg: dq.load_function("lingxi_gemm_q4_skinny_dbg")?,
            ple_gate_bf16: moe.load_function("lingxi_ple_gate_bf16")?,
            add_bf16: moe.load_function("lingxi_add_bf16")?,
            rope_partial_inplace_bf16: attn.load_function("lingxi_rope_partial_inplace_bf16")?,
            attn_masked_bf16: attn.load_function("lingxi_attn_masked_bf16")?,
            attn2_masked_bf16: attn.load_function("lingxi_attn2_masked_bf16")?,
            attn2s_sparse_bf16: attn.load_function("lingxi_attn2s_sparse_bf16")?,
            attn3_masked_bf16: attn.load_function("lingxi_attn3_masked_bf16")?,
            attn5_sparse_bf16: attn.load_function("lingxi_attn5_sparse_bf16")?,
            attn2_masked_kvq8_bf16: attn.load_function("lingxi_attn2_masked_kvq8_bf16")?,
            attn2s_sparse_kvq8_bf16: attn.load_function("lingxi_attn2s_sparse_kvq8_bf16")?,
            attn3_masked_kvq8_bf16: attn.load_function("lingxi_attn3_masked_kvq8_bf16")?,
            attn5_sparse_kvq8_bf16: attn.load_function("lingxi_attn5_sparse_kvq8_bf16")?,
            quant_rows_e4m3_bf16: attn.load_function("lingxi_quant_rows_e4m3_bf16")?,
            qsa_pool_block_bf16: attn.load_function("lingxi_qsa_pool_block_bf16")?,
            qsa_pool_blocks_bf16: attn.load_function("lingxi_qsa_pool_blocks_bf16")?,
            rope_partial_stride_bf16: attn.load_function("lingxi_rope_partial_stride_bf16")?,
            qsa_indexer_scores_bf16: attn.load_function("lingxi_qsa_indexer_scores_bf16")?,
            qsa_indexer_rank_f32: attn.load_function("lingxi_qsa_indexer_rank_f32")?,
            qsa_keep_select_bf16: attn.load_function("lingxi_qsa_keep_select_bf16")?,
            qsa_keep_fill_bf16: attn.load_function("lingxi_qsa_keep_fill_bf16")?,
            qsa_keep_fused_bf16: attn.load_function("lingxi_qsa_keep_fused_bf16")?,
            qsa_keep_from_scores_bf16: attn.load_function("lingxi_qsa_keep_from_scores_bf16")?,
            qsa_sel_build: attn.load_function("lingxi_qsa_sel_build")?,
            qsa_hist_f32: attn.load_function("lingxi_qsa_hist_f32")?,
            qsa_bin_finalize: attn.load_function("lingxi_qsa_bin_finalize")?,
            qsa_keep_fill_scores: attn.load_function("lingxi_qsa_keep_fill_scores")?,
            attn_gather_bf16: attn.load_function("lingxi_attn_gather_bf16")?,
            attn_gather4_bf16: attn.load_function("lingxi_attn_gather4_bf16")?,
            qsa_mask_fill_f32: attn.load_function("lingxi_qsa_mask_fill_f32")?,
            dequant_rowsplit_q4g64_bf16: dq.load_function("lingxi_dequant_rowsplit_q4g64_bf16")?,
            dequant_rowsplit_q4g64_batch: dq.load_function("lingxi_dequant_rowsplit_q4g64_batch")?,
            dequant_rowsplit_q3g128_bf16: dq.load_function("lingxi_dequant_rowsplit_q3g128_bf16")?,
            gemv_q4g64_slots_bf16: dq.load_function("lingxi_gemv_q4g64_slots_bf16")?,
            gemv_q4g64_slots_r2_bf16: dq.load_function("lingxi_gemv_q4g64_slots_r2_bf16")?,
            gemv_q4g64_slots_r4_bf16: dq.load_function("lingxi_gemv_q4g64_slots_r4_bf16")?,
            gemv_q4g64_slots_r8_bf16: dq.load_function("lingxi_gemv_q4g64_slots_r8_bf16")?,
            gemv_q4g64_slotsm_bf16: dq.load_function("lingxi_gemv_q4g64_slotsm_bf16")?,
            gemv_q4g64_slotsm_r2_bf16: dq.load_function("lingxi_gemv_q4g64_slotsm_r2_bf16")?,
            gemv_q4g64_slotsm_r4_bf16: dq.load_function("lingxi_gemv_q4g64_slotsm_r4_bf16")?,
            gemv_q4g64_slotsm_dp4a: dq.load_function("lingxi_gemv_q4g64_slotsm_dp4a")?,
            gemv_q4g64_slotsm_r2_dp4a: dq.load_function("lingxi_gemv_q4g64_slotsm_r2_dp4a")?,
            gemv_q3g128_slots_bf16: dq.load_function("lingxi_gemv_q3g128_slots_bf16")?,
            gemv_q3g128_slots_r2_bf16: dq.load_function("lingxi_gemv_q3g128_slots_r2_bf16")?,
            gemv_q3g128_slots_r4_bf16: dq.load_function("lingxi_gemv_q3g128_slots_r4_bf16")?,
            gemv_q3g128_slots_r8_bf16: dq.load_function("lingxi_gemv_q3g128_slots_r8_bf16")?,
            gemv_q3g128_slotsm_bf16: dq.load_function("lingxi_gemv_q3g128_slotsm_bf16")?,
            gemv_q3g128_slotsm_r2_bf16: dq.load_function("lingxi_gemv_q3g128_slotsm_r2_bf16")?,
            gemv_q3g128_slotsm_r4_bf16: dq.load_function("lingxi_gemv_q3g128_slotsm_r4_bf16")?,
            convert_bf16_f32: dq.load_function("lingxi_convert_bf16_f32")?,
            convert_bf16_q8_rs: dq.load_function("lingxi_convert_bf16_q8_rs")?,
            gemv_q4g64_dense_epi: dq.load_function("lingxi_gemv_q4g64_dense_epi")?,
            gemv_q4g64_dense4_epi: dq.load_function("lingxi_gemv_q4g64_dense4_epi")?,
            dequant_q4g64_dense_bf16: dq.load_function("lingxi_dequant_q4g64_dense_bf16")?,
            causal_conv1d_silu_bf16: gdn.load_function("lingxi_causal_conv1d_silu_bf16")?,
            causal_conv1d_silu_hist_bf16: gdn.load_function("lingxi_causal_conv1d_silu_hist_bf16")?,
            gdn_recurrent_bf16: {
                let f = gdn.load_function("lingxi_gdn_recurrent_bf16")?;
                // P15-㉘：S(hk×hv fp32，本模型 64KB) 动态 smem 驻留。
                // 上限不能顶到 96KB：每 block 预算还要扣静态 smem(ssum) 与 1KB 保留，
                // 顶满会 CUDA_ERROR_INVALID_VALUE（实测）。
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    (96 * 1024 - 2048) as i32,
                )?;
                f
            },
            argmax_rows_part_bf16: topk.load_function("lingxi_argmax_rows_part_bf16")?,
            argmax_rows_fin: topk.load_function("lingxi_argmax_rows_fin")?,
            topk_rows_part_bf16: topk.load_function("lingxi_topk_rows_part_bf16")?,
            topk_rows_fin: topk.load_function("lingxi_topk_rows_fin")?,
            route_topk_slots_bf16: topk.load_function("lingxi_route_topk_slots_bf16")?,
            blas,
        })
    }

    fn grid(n: usize) -> u32 {
        (((n as u64) + BLOCK as u64 - 1) / BLOCK as u64) as u32
    }

    // ───────────────────────── 既有（T1 基线） ─────────────────────────

    pub fn smoke_scale(&self, x: &CudaSlice<f32>, a: f32, y: &mut CudaSlice<f32>, n: u32) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (Self::grid(n as usize), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.smoke_scale)
                .arg(x).arg(&a).arg(y).arg(&(n as i32)).launch(cfg)?;
        }
        Ok(())
    }

    pub fn rmsnorm_f32(
        &self, x: &CudaSlice<f32>, gamma: &CudaSlice<f32>, y: &mut CudaSlice<f32>,
        rows: u32, dim: usize, eps: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (rows, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: BLOCK * 4 };
        unsafe {
            self.stream.launch_builder(&self.rmsnorm_f32)
                .arg(x).arg(gamma).arg(y).arg(&(dim as i32)).arg(&eps).launch(cfg)?;
        }
        Ok(())
    }

    // ───────────────────────── GR（GatedResidual） ─────────────────────────

    /// 分组 RMSNorm（gamma 已烘焙）：`x[rows, dim]`，每 `group` 个元素独立归一化。
    pub fn grouped_rmsnorm_bf16(
        &self, x: &CudaSlice<bf16>, gamma: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
        rows: u32, dim: usize, group: usize, eps: f32,
    ) -> Result<()> {
        assert_eq!(dim % group, 0, "dim 必须是 group 的整数倍");
        let cfg = LaunchConfig {
            grid_dim: (rows, (dim / group) as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * 4,
        };
        unsafe {
            self.stream.launch_builder(&self.grouped_rmsnorm_bf16)
                .arg(x).arg(gamma).arg(y)
                .arg(&(dim as i32)).arg(&(group as i32)).arg(&eps).launch(cfg)?;
        }
        Ok(())
    }

    /// 融合 GR 第一阶段（decode t==1）：分组 RMSNorm 折进 `mix_down`（+`block_inject`）
    /// 的 GEMV 读通路，一次出 `d` 与 `injw`，并把各分组的 `rms` 写进 `rmsg`。
    /// 一行一个 block；见 `cuda/gr_ops.cu` 顶部注释（借自 DeepGEMM 的 prenorm 寄生技巧）。
    #[allow(clippy::too_many_arguments)]
    pub fn gr_down_fuse_bf16(
        &self, x: &CudaSlice<bf16>, gamma: &CudaSlice<bf16>,
        mix_down: &CudaSlice<bf16>, block_inject: &CudaSlice<bf16>,
        d_out: &mut CudaSlice<bf16>, injw_out: &mut CudaSlice<bf16>,
        rmsg: &mut CudaSlice<f32>,
        dim: usize, hidden: usize, ngroups: usize, lowrank: usize, hc: usize,
        inv_hc: f32, eps: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: ((lowrank + hc) as u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gr_down_fuse_bf16)
                .arg(x).arg(gamma).arg(mix_down).arg(block_inject)
                .arg(d_out).arg(injw_out).arg(rmsg)
                .arg(&(dim as i32)).arg(&(hidden as i32)).arg(&(ngroups as i32))
                .arg(&(lowrank as i32)).arg(&(hc as i32)).arg(&inv_hc).arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 融合 GR 第二阶段（decode t==1）：`mix_up` + sigmoid + `mix_mean` 一次出 `mixed`，
    /// **不落 `imw`**。一个 block 一个输出，block 内每 warp 一个 g（并行度对齐旧 GEMV）。
    #[allow(clippy::too_many_arguments)]
    pub fn gr_up_mix_bf16(
        &self, d: &CudaSlice<bf16>, x: &CudaSlice<bf16>, gamma: &CudaSlice<bf16>,
        mix_up: &CudaSlice<bf16>, rmsg: &CudaSlice<f32>, mixed: &mut CudaSlice<bf16>,
        _dim: usize, hidden: usize, lowrank: usize, ngroups: usize, inv_hc: f32,
    ) -> Result<()> {
        let block = ((ngroups.max(1) * 32) as u32).min(1024);
        let cfg = LaunchConfig {
            grid_dim: (hidden as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: (lowrank * 4) as u32,
        };
        unsafe {
            self.stream.launch_builder(&self.gr_up_mix_bf16)
                .arg(d).arg(x).arg(gamma).arg(mix_up).arg(rmsg).arg(mixed)
                .arg(&(0i32)) // dim 占位（内核签名保留）
                .arg(&(hidden as i32)).arg(&(lowrank as i32))
                .arg(&(ngroups as i32)).arg(&inv_hc)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 就地 `x = silu(x * scale)`。
    pub fn silu_scale_inplace_bf16(&self, x: &mut CudaSlice<bf16>, scale: f32, n: usize) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.silu_scale_inplace_bf16)
                .arg(x).arg(&scale).arg(&(n as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// 就地 `x = sigmoid(x * scale) * mul`。
    pub fn sigmoid_scale_mul_inplace_bf16(&self, x: &mut CudaSlice<bf16>, scale: f32, mul: f32, n: usize) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.sigmoid_scale_mul_inplace_bf16)
                .arg(x).arg(&scale).arg(&mul).arg(&(n as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// `out[row, i] = (1/hc) Σ_g w[row, g*H+i] * xn[row, g*H+i]`，**一次覆盖 `rows` 行**
    /// （`grid.y` = 行）。原来 Rust 侧每行发一次 launch，t>1 时每层多出 ~2(t-1) 个 kernel
    /// × ~15 µs 的 GPU 固定开销（t=8 折合 ~10 ms/token）。
    pub fn mix_mean_bf16(
        &self, w: &CudaSlice<bf16>, xn: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>,
        hc: usize, hidden: usize, rows: u32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(hidden), rows.max(1), 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.mix_mean_bf16)
                .arg(w).arg(xn).arg(out).arg(&(hc as i32)).arg(&(hidden as i32)).arg(&0i32).launch(cfg)?;
        }
        Ok(())
    }

    /// 就地残差写回：`hyper[row, g*H+i] += attn[row, i] * injw[row, g]`，**一次覆盖 `rows` 行**。
    pub fn gr_write_inplace_bf16(
        &self, hyper: &mut CudaSlice<bf16>, attn: &CudaSlice<bf16>, injw: &CudaSlice<bf16>,
        hc: usize, hidden: usize, rows: u32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(hidden), rows.max(1), 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gr_write_inplace_bf16)
                .arg(hyper).arg(attn).arg(injw).arg(&(hc as i32)).arg(&(hidden as i32)).arg(&0i32).launch(cfg)?;
        }
        Ok(())
    }

    /// 主干入口广播：`out[r, g*H + i] = emb[r, i]`，`out` 形状 `[rows, hc*H]`。
    pub fn hc_repeat_bf16(
        &self, emb: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>,
        rows: usize, hidden: usize, hc: usize,
    ) -> Result<()> {
        let n = rows * hc * hidden;
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.hc_repeat_bf16)
                .arg(emb).arg(out).arg(&(rows as i32)).arg(&(hidden as i32)).arg(&(hc as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// `y = a * b`（逐元素）。
    pub fn mul_bf16(&self, a: &CudaSlice<bf16>, b: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, n: usize) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.mul_bf16).arg(a).arg(b).arg(y).arg(&(n as i32)).launch(cfg)?;
        }
        Ok(())
    }

    // ───────────────────────── MoE ─────────────────────────

    /// 融合 SwiGLU：`x[m, 2*inter] → y[m, inter]`，`y = silu(x[:, :inter]) * x[:, inter:]`。
    pub fn swiglu_bf16(&self, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, m: usize, inter: usize) -> Result<()> {
        let x_v = x.slice(0..x.len());
        let mut y_v = y.slice_mut(0..y.len());
        self.swiglu_bf16_view(&x_v, &mut y_v, m, inter)
    }

    /// 同 [`Self::swiglu_bf16`]，但源/目标取**视图**（㊶ hit/miss 重叠的逐段 swiglu；
    /// 元素级按对独立 ⇒ 切段逐位不变）。
    pub fn swiglu_bf16_view(
        &self, x: &cudarc::driver::CudaView<'_, bf16>,
        y: &mut cudarc::driver::CudaViewMut<'_, bf16>, m: usize, inter: usize,
    ) -> Result<()> {
        let n = m * inter;
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.swiglu_bf16)
                .arg(x).arg(y).arg(&(m as i32)).arg(&(inter as i32)).launch(cfg)?;
        }
        Ok(())
    }

    pub fn gather_rows_bf16(
        &self, src: &CudaSlice<bf16>, idx: &CudaSlice<i32>, dst: &mut CudaSlice<bf16>,
        m: usize, k: usize, idx_off: usize,
    ) -> Result<()> {
        let n = m * k;
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.gather_rows_bf16)
                .arg(src).arg(idx).arg(dst).arg(&(m as i32)).arg(&(k as i32))
                .arg(&(idx_off as i32)).launch(cfg)?;
        }
        Ok(())
    }

    pub fn scatter_add_rows_bf16(
        &self, dst: &mut CudaSlice<bf16>, idx: &CudaSlice<i32>, w: &CudaSlice<f32>,
        src: &CudaSlice<bf16>, m: usize, k: usize, idx_off: usize,
    ) -> Result<()> {
        let n = m * k;
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.scatter_add_rows_bf16)
                .arg(dst).arg(idx).arg(w).arg(src).arg(&(m as i32)).arg(&(k as i32))
                .arg(&(idx_off as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// 每行 argmax（GPU 两段式，免去 t×vocab 的 D2H + host 遍历）。
    /// **位等价**：与 host `argmax_rows_bf16` 同一全序（f32 值降序、下标升序）⇒ 结果唯一。
    /// `pval`/`pidx` 是调用方给的 `[rows*g]` scratch（part 的部分归约），`out` 是 `[rows]` i32。
    /// `x_off` = 起始行在 `x` 里的元素偏移（取最后一行 argmax 时用，免拷切片）。
    /// 两段的 block 都必须是 128 线程（核里 smem/归约按 128 写死）。
    pub fn argmax_rows_part_bf16(
        &self, x: &CudaSlice<bf16>, x_off: usize, pval: &mut CudaSlice<f32>,
        pidx: &mut CudaSlice<i32>, rows: usize, vocab: usize, g: usize,
    ) -> Result<()> {
        let (xp, _g) = x.device_ptr(&self.stream);
        let cfg = LaunchConfig {
            grid_dim: (g as u32, rows as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.argmax_rows_part_bf16)
                .arg(&(xp + (x_off * 2) as u64)).arg(pval).arg(pidx)
                .arg(&(vocab as i32)).arg(&(g as i32)).launch(cfg)?;
        }
        Ok(())
    }

    pub fn argmax_rows_fin(
        &self, pval: &CudaSlice<f32>, pidx: &CudaSlice<i32>, out: &mut CudaSlice<i32>,
        rows: usize, g: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.argmax_rows_fin)
                .arg(pval).arg(pidx).arg(out)
                .arg(&(g as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// 每行 top-K（GPU 两段式）。**位等价**：与 host `topk_bf16_row` 同一全序
    /// （f32 值降序、下标升序）⇒ 结果唯一。`bad` 计数非有限值（host 侧须检查后报错）。
    /// 调用方保证 `k ≤ 32`、`vocab ≥ k`、`g ≤ 128`。block 128 线程（核内 smem 按 128 写死）。
    pub fn topk_rows_part_bf16(
        &self, x: &CudaSlice<bf16>, pval: &mut CudaSlice<f32>, pidx: &mut CudaSlice<i32>,
        bad: &mut CudaSlice<i32>, rows: usize, vocab: usize, g: usize, k: usize,
    ) -> Result<()> {
        let (xp, _g) = x.device_ptr(&self.stream);
        let cfg = LaunchConfig {
            grid_dim: (g as u32, rows as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.topk_rows_part_bf16)
                .arg(&xp).arg(pval).arg(pidx).arg(bad)
                .arg(&(vocab as i32)).arg(&(g as i32)).arg(&(k as i32)).launch(cfg)?;
        }
        Ok(())
    }

    pub fn topk_rows_fin(
        &self, pval: &CudaSlice<f32>, pidx: &CudaSlice<i32>,
        oval: &mut CudaSlice<f32>, oidx: &mut CudaSlice<i32>,
        rows: usize, g: usize, k: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.topk_rows_fin)
                .arg(pval).arg(pidx).arg(oval).arg(oidx)
                .arg(&(g as i32)).arg(&(k as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// 就地行缩放广播：`x[row, :] *= s[row]`。
    pub fn row_scale_inplace_bf16(
        &self, x: &mut CudaSlice<bf16>, s: &CudaSlice<bf16>, rows: usize, cols: usize,
    ) -> Result<()> {
        let n = rows * cols;
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.row_scale_inplace_bf16)
                .arg(x).arg(s).arg(&(rows as i32)).arg(&(cols as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// PLE 门控（GPU 版，替代 host 参考）：去掉每层 3 次 D2H + 1 次 H2D。
    #[allow(clippy::too_many_arguments)]
    pub fn ple_gate_bf16(
        &self, key_n: &CudaSlice<bf16>, query_n: &CudaSlice<bf16>, value: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>, t: usize, hc: usize, hidden: usize, inv: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: ((t * hc) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.ple_gate_bf16)
                .arg(key_n).arg(query_n).arg(value).arg(out)
                .arg(&(hc as i32)).arg(&(hidden as i32)).arg(&inv)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// bf16 GEMV：`y[n] = W[n,k] · x[k]`（T=1 decode 用；替掉 cuBLAS 的 ~26µs 调用开销）。
    /// 用**裸设备指针**传参，这样 `CudaSlice` 与 `CudaView` 都能直接喂（取 `.device_ptr(stream)`）。
    pub fn gemv_bf16_ptr(&self, w: u64, x: u64, y: u64, n: usize, k: usize) -> Result<()> {
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_bf16)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 带**尾融合**的 bf16 GEMV（见 CUDA 侧注释）。`n` = 权重行数；`epi==4`（swiglu）时
    /// 输出行数是 `n/2`（g/u 分别取第 `i` 与第 `i+n/2` 行）。同样用裸指针，兼容 `CudaView`。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16_epi_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        let out_rows = if epi == 4 { n / 2 } else { n };
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((out_rows + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_bf16_epi)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **高并行度** GEMV：一个 block（1024 线程 = 32 warp）算一行，warp 间用 shared 归约。
    /// 专治 n 小 k 大的形状（实测 [320,10240] 在 warps-per-row 下只有 251 GB/s，
    /// [4,10240] 只有 6 GB/s —— 瓶颈是每行的 load 依赖链，不是带宽）。
    /// 累加顺序与 `gemv_bf16_epi_ptr` 不同 → 数值末位会有差异。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16_block_epi_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_bf16_block_epi)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **向量化 block GEMV**：一行一个 block、128-bit load（一次 8 个 bf16）。
    /// 专治 `n` 小 `k` 大的形状（[320,10240] / [4,10240]），见 CUDA 侧注释。
    /// 累加顺序与 `gemv_bf16_block_epi` 不同（此处 thread0 串行合并 warp 部分和）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16_row8_epi_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_bf16_row8_epi)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **分块 Q4 瘦长 GEMM**（t>1，直接吃 Q4、不落 bf16 中转）：
    /// grid = `ceil(n/256)`、一线程一行、k 分块把 x 搬进 shared。
    /// `LINGXI_NO_Q4GEMM=1` 可回退到旧的"解量化 + 瘦长 GEMM"。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_skinny_bf16_ptr(
        &self, wq: u64, x: u64, y: u64, n: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!((1..=8).contains(&m), "Q4 瘦长 GEMM 只支持 1≤m≤8（给到 {m}）");
        if std::env::var("LINGXI_Q4G_DBG").is_ok() {
            let cfg = LaunchConfig {
                grid_dim: (((n + 255) / 256).max(1) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&self.gemm_q4_skinny_dbg)
                    .arg(&wq).arg(&x).arg(&y)
                    .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                    .arg(&(row_off as i32)).arg(&(scale_off as i64))
                    .launch(cfg)?;
            }
            return Ok(());
        }
        let blocks = ((n + 7) / 8).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 8 * 1024 * 2,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_q4_skinny_bf16)
                .arg(&wq).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&(scale_off as i64))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **一行一个 block 的瘦长 GEMM**（t>1 用，并行度对齐 row8）：grid = n 个 block、128 位 load。
    /// 旧版 `gemm_skinny` 的 grid 只有 `n/8` 个 block，n 小时并行度差 8 倍（见 CUDA 侧注释）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_skinny_row_bf16_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!((1..=8).contains(&m), "skinny-row 只支持 1≤m≤8（给到 {m}）");
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 32 * 8 * 4,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_skinny_row)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ★ P15-⑲：`gemm_skinny_row` 的**多行/块**版（R=4）。一块算 4 个输出行，x 分块进
    /// 动态 smem（转好 f32、行间共享）⇒ m>1 时 x 的 L2 读与 CVT 同除 4，权重流量不变。
    /// **逐位等价**（对 `gemm_skinny_row`）：每线程累加序列/归约结构全同（见 CUDA 注释）。
    /// n 不满 4 的尾巴块按 R=min(4, n-row0) 处理；动态 smem = m×2048×4 ≤ 64KB（加载时已抬属性）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_skinny_row_multi_bf16_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!((1..=8).contains(&m), "skinny-row-multi 只支持 1≤m≤8（给到 {m}）");
        let cfg = LaunchConfig {
            grid_dim: (((n + 3) / 4) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: (m * 2112 * 4) as u32,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_skinny_row_multi)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ★ P15-⑨：`gemm_skinny_row` 的 **x 预转 fp32** 版（"后面要转化的提前转化"）。
    /// x 传 fp32（调用方先 `convert_bf16_f32` 一次、多个投影共用），w 仍 bf16。
    /// m>1 时内层不再每 token 重复 bf16→f32 CVT ⇒ m=8 带宽从 ~350 回到 ~700 GB/s（见 CUDA 注释）。
    /// **逐位等价**：bf16→f32 精确，z 次序与累加次序不变。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_skinny_row_f32x_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!((1..=8).contains(&m), "skinny-row-f32x 只支持 1≤m≤8（给到 {m}）");
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 32 * 8 * 4,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_skinny_row_f32x)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **瘦长 GEMM**（`1 ≤ m ≤ 8`）：一个 warp 一行输出，把权重行**只读一遍**同时算 m 个输入行。
    /// 按 m 特化（m=1..8 各一个内核），否则指令数会随 8 走、权重带宽掉到 1/3。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_skinny_bf16_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!((1..=8).contains(&m), "瘦长 GEMM 只支持 1≤m≤8（给到 {m}）");
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_skinny[m - 1])
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **稠密权重** Q4G64 融合反量化 GEMV（裸指针传参，含尾融合）。
    /// `k_pad` = 每行 nibble 数 × 2（= align_up(cols,128)）；`scale_off` = scale 平面相对 payload 的字节偏移。
    /// **Q4 稠密 GEMV：一个 warp 算 4 行**（x 侧的 load+转换按 4 行摊薄）。
    /// `LINGXI_Q4_ROW1=1` 回退到单行版做 A/B。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_dense_epi_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, k_pad: i32, row_off: usize,
        scale_off: u64, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        // ⚠️ **默认已翻转为单行版**（`LINGXI_Q4_ROW4=1` 可回到 4 行版做 A/B）。
        // 历史：最初 4 行/warp 更快（x 的 load+转换按 4 行摊薄）。但 nibble→float 查表落地后
        // 平衡完全变了：4 行版每次迭代要做 **16 次 `LDS.64`**（4 行 × 4），**shared 带宽被自己挤爆**，
        // launch-bench 实测 `[16384,2560]` 369 GB/s（4 行）vs **482 GB/s（单行）**、
        // `[13312,2560]` 354 vs **514**。而且单行版 REG 只有 32（4 行版 48）。
        let use4 = n >= 4 && k <= 3072 && self.q4_row4();
        let (f, warps) = if use4 {
            (&self.gemv_q4g64_dense4_epi, (n + 3) / 4)
        } else {
            (&self.gemv_q4g64_dense_epi, n)
        };
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((warps + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(f)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(row_off as i32))
                .arg(&(scale_off as i64)).arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `LINGXI_Q4_ROW4=1`：稠密 Q4 GEMV 回到「一个 warp 算 4 行」（做 A/B 用；默认已改成单行版）。
    fn q4_row4(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("LINGXI_Q4_ROW4").is_ok())
    }

    /// 稠密 Q4G64 → bf16 整块解量化（`t>1` 的批量路径用）。
    pub fn dequant_q4g64_dense_bf16(
        &self, w: &CudaSlice<u8>, out: &mut CudaSlice<bf16>, rows: usize, cols: usize,
        k_pad: i32, row_off: usize, scale_off: u64,
    ) -> Result<()> {
        let n = rows * cols;
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.dequant_q4g64_dense_bf16)
                .arg(w).arg(out)
                .arg(&(rows as i32)).arg(&(cols as i32)).arg(&k_pad).arg(&(row_off as i32))
                .arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 合并：`expert_out[r, :] = Σ_{s : rows_of[s] == r} w[s] * o[s, :]`（f32 累加、一次 round）。
    pub fn moe_combine_rows_bf16(
        &self, o: &CudaSlice<bf16>, rows_of: &CudaSlice<i32>, w: &CudaSlice<f32>,
        expert_out: &mut CudaSlice<bf16>, s: usize, rows: usize, hidden: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(hidden), rows as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.moe_combine_rows_bf16)
                .arg(o).arg(rows_of).arg(w).arg(expert_out)
                .arg(&(s as i32)).arg(&(hidden as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 合并+残差一发融合（㊶）：`out[r,:] = bf16(f32(bf16(Σ w·o)) + f32(shared[r,:]))`，
    /// 与 `moe_combine_rows_bf16` + `add_bf16` 两次舍入同序 ⇒ 逐位相同。
    pub fn moe_combine_add_rows_bf16(
        &self, o: &CudaSlice<bf16>, rows_of: &CudaSlice<i32>, w: &CudaSlice<f32>,
        shared: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, s: usize, rows: usize,
        hidden: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(hidden), rows as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.moe_combine_add_rows_bf16)
                .arg(o).arg(rows_of).arg(w).arg(shared).arg(out)
                .arg(&(s as i32)).arg(&(hidden as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ㊾ CSR 版合并+残差：`col/row_off` 为"行 → 升序对下标"的 CSR（host 稳定计数排序）。
    /// 每行累加次序与 [`Self::moe_combine_add_rows_bf16`] 的"升序 s 扫描"一致 ⇒ 逐位相同；
    /// 复杂度 O(rows×H×top_k)，batch 路径（S 数千）用它，旧核留给 S 小的路径。
    pub fn moe_combine_add_csr_bf16(
        &self, o: &CudaSlice<bf16>, col: &CudaSlice<i32>, row_off: &CudaSlice<i32>,
        w: &CudaSlice<f32>, shared: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>,
        rows: usize, hidden: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(hidden), rows as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.moe_combine_add_csr_bf16)
                .arg(o).arg(col).arg(row_off).arg(w).arg(shared).arg(out)
                .arg(&(hidden as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `y = a + b`（逐元素）。
    pub fn add_bf16(&self, a: &CudaSlice<bf16>, b: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, n: usize) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.add_bf16).arg(a).arg(b).arg(y).arg(&(n as i32)).launch(cfg)?;
        }
        Ok(())
    }

    // ───────────────────────── QSA 注意力 ─────────────────────────

    /// 部分 RoPE（NEOX 配对，就地）：`x[rows, heads, head_dim]`，只旋转前 `rot_dim` 维。
    /// `cos`/`sin` 是覆盖全序列的表；`x` 的第 row 行对应绝对位置 `row + pos_offset`
    /// （整段前向传 0；增量 decode 传已缓存 token 数）。
    pub fn rope_partial_inplace_bf16(
        &self, x: &mut CudaSlice<bf16>, cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
        rows: usize, heads: usize, head_dim: usize, rot_dim: usize, pos_offset: usize,
    ) -> Result<()> {
        let n = rows * heads * (rot_dim / 2);
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.rope_partial_inplace_bf16)
                .arg(x).arg(cos).arg(sin)
                .arg(&(rows as i32)).arg(&(heads as i32))
                .arg(&(head_dim as i32)).arg(&(rot_dim as i32))
                .arg(&(pos_offset as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 带加性掩码的多头注意力（token-major，batch 固定 1）：
    /// `q[T,H,D] / kv[Tk,KVH,D] / mask[T,Tk] → out[T,H,D]`。
    /// `k`/`v` 只需是**行主序连续**的基址；`tk` 决定实际读取的行数（缓存可大于已用长度）。
    /// 稀疏/全注意力（**重写版**，点积只算两遍）。
    ///
    /// 旧版 §BUG：加权求和里对**每个输出维度都把全部 key 的 QK 点积重算一遍**，
    /// 复杂度 `O(head_dim²·Tk)`；head_dim=256、Tk=192 时单核 1240 µs，
    /// 12 个 QSA 层 ≈ 15 ms/token（占整机 44%）。本版是 `O(head_dim·Tk)`，见 CUDA 侧注释。
    /// `head_dim > BLOCK` 时回退旧核（本模型 head_dim=256 = BLOCK，不会走到）。
    pub fn attn_masked_bf16(
        &self, q: &CudaSlice<bf16>, k: &CudaSlice<bf16>, v: &CudaSlice<bf16>,
        mask: &CudaSlice<f32>, out: &mut CudaSlice<bf16>,
        heads: usize, kv_heads: usize, t: usize, tk: usize, head_dim: usize, scale: f32,
    ) -> Result<()> {
        if head_dim <= BLOCK as usize && tk > 0 && head_dim % 8 == 0 {
            // ★ P15-㊳ 行块平铺（attn3）：t ≥ 8 且 head_dim==256 时一个 block 喂 8 行，
            // K/V 流量 ÷8——prefill 的 O(chunk×T) attention 是 256K TTFT 的元凶。
            // 每行归约次序与 attn2 逐项相同（bit-safe）；`LINGXI_NO_ATTN3=1` 回退 attn2 对拍。
            const ATT3_R: usize = 4;
            if head_dim == 256 && t >= ATT3_R
                && std::env::var_os("LINGXI_NO_ATTN3").is_none()
            {
                let smem = ATT3_R * 512 * 4 + 32 * 4; // [R][TILE=512]f32 + sred[32]
                let cfg = LaunchConfig {
                    grid_dim: (((t + ATT3_R - 1) / ATT3_R) as u32, heads as u32, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: smem as u32,
                };
                unsafe {
                    self.stream.launch_builder(&self.attn3_masked_bf16)
                        .arg(q).arg(k).arg(v).arg(mask).arg(out)
                        .arg(&(heads as i32)).arg(&(kv_heads as i32))
                        .arg(&(t as i32)).arg(&(tk as i32))
                        .arg(&(head_dim as i32)).arg(&scale)
                        .launch(cfg)?;
                }
                return Ok(());
            }
            let tile = tk.min(4096);
            let cfg = LaunchConfig {
                grid_dim: (t as u32, heads as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: (tile * 4 + 32 * 4) as u32,
            };
            unsafe {
                self.stream.launch_builder(&self.attn2_masked_bf16)
                    .arg(q).arg(k).arg(v).arg(mask).arg(out)
                    .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(t as i32)).arg(&(tk as i32))
                    .arg(&(head_dim as i32)).arg(&scale).arg(&(tile as i32))
                    .launch(cfg)?;
            }
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: (t as u32, heads as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * 4,
        };
        unsafe {
            self.stream.launch_builder(&self.attn_masked_bf16)
                .arg(q).arg(k).arg(v).arg(mask).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(t as i32)).arg(&(tk as i32))
                .arg(&(head_dim as i32)).arg(&scale)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ★ P15-㊵ **跳块稀疏注意力**（QSA）：只算 keep 选中块 + 因果前缀内的点积，
    /// 未选中位置按 dense 精确语义跳过（-FLT_MAX / -inf 两档，含退化整行无可见位情形，
    /// 见 attn_ops.cu 的 state_j 注释）。对拍开关 `LINGXI_QSA_DENSE=1`（走旧掩码路径）。
    /// `vis_base` = 已缓存 token 数（state.len）；`cshift` = log2(compress)，compress 必须 2 的幂。
    pub fn attn_sparse_bf16(
        &self, q: &CudaSlice<bf16>, k: &CudaSlice<bf16>, v: &CudaSlice<bf16>,
        keep: &CudaSlice<u8>, vis_base: usize, nblocks: usize, cshift: u32,
        out: &mut CudaSlice<bf16>,
        heads: usize, kv_heads: usize, t: usize, tk: usize, head_dim: usize, scale: f32,
    ) -> Result<()> {
        assert!(tk > 0 && head_dim == 256, "attn_sparse 仅支持 head_dim=256");
        const ATT3_R: usize = 4;
        if t >= ATT3_R {
            let smem = ATT3_R * 512 * 4 + 32 * 4;
            let cfg = LaunchConfig {
                grid_dim: (((t + ATT3_R - 1) / ATT3_R) as u32, heads as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            unsafe {
                self.stream.launch_builder(&self.attn5_sparse_bf16)
                    .arg(q).arg(k).arg(v).arg(keep).arg(out)
                    .arg(&(heads as i32)).arg(&(kv_heads as i32))
                    .arg(&(t as i32)).arg(&(tk as i32))
                    .arg(&(head_dim as i32)).arg(&scale)
                    .arg(&(vis_base as i32)).arg(&(nblocks as i32)).arg(&(cshift as i32))
                    .launch(cfg)?;
            }
            return Ok(());
        }
        let tile = tk.min(4096);
        let cfg = LaunchConfig {
            grid_dim: (t as u32, heads as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: (tile * 4 + 32 * 4) as u32,
        };
        unsafe {
            self.stream.launch_builder(&self.attn2s_sparse_bf16)
                .arg(q).arg(k).arg(v).arg(keep).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(t as i32)).arg(&(tk as i32))
                .arg(&(head_dim as i32)).arg(&scale).arg(&(tile as i32))
                .arg(&(vis_base as i32)).arg(&(nblocks as i32)).arg(&(cshift as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ㊶ KV 量化（E4M3）版跳块稀疏注意力：`k_q8/k_sc/v_q8/v_sc` 为量化 cache +
    /// 每 (行, kv头) scale。**有损**（`LINGXI_KV_K8V4` 门控），分派结构与
    /// [`Self::attn_sparse_bf16`] 相同（t≥4 → attn5q，否则 attn2sq）。
    #[allow(clippy::too_many_arguments)]
    pub fn attn_sparse_kvq8_bf16(
        &self, q: &CudaSlice<bf16>, k: &CudaSlice<bf16>, v: &CudaSlice<bf16>,
        keep: &CudaSlice<u8>, vis_base: usize, nblocks: usize, cshift: u32,
        out: &mut CudaSlice<bf16>,
        heads: usize, kv_heads: usize, t: usize, tk: usize, head_dim: usize, scale: f32,
        k_q8: &CudaSlice<u8>, k_sc: &CudaSlice<f32>,
        v_q8: &CudaSlice<u8>, v_sc: &CudaSlice<f32>,
    ) -> Result<()> {
        assert!(tk > 0 && head_dim == 256, "attn_sparse_kvq8 仅支持 head_dim=256");
        const ATT3_R: usize = 4;
        if t >= ATT3_R {
            let smem = ATT3_R * 512 * 4 + 32 * 4;
            let cfg = LaunchConfig {
                grid_dim: (((t + ATT3_R - 1) / ATT3_R) as u32, heads as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            unsafe {
                self.stream.launch_builder(&self.attn5_sparse_kvq8_bf16)
                    .arg(q).arg(k).arg(v).arg(keep).arg(out)
                    .arg(&(heads as i32)).arg(&(kv_heads as i32))
                    .arg(&(t as i32)).arg(&(tk as i32))
                    .arg(&(head_dim as i32)).arg(&scale)
                    .arg(&(vis_base as i32)).arg(&(nblocks as i32)).arg(&(cshift as i32))
                    .arg(k_q8).arg(k_sc).arg(v_q8).arg(v_sc)
                    .launch(cfg)?;
            }
            return Ok(());
        }
        let tile = tk.min(4096);
        let cfg = LaunchConfig {
            grid_dim: (t as u32, heads as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: (tile * 4 + 32 * 4) as u32,
        };
        unsafe {
            self.stream.launch_builder(&self.attn2s_sparse_kvq8_bf16)
                .arg(q).arg(k).arg(v).arg(keep).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(t as i32)).arg(&(tk as i32))
                .arg(&(head_dim as i32)).arg(&scale).arg(&(tile as i32))
                .arg(&(vis_base as i32)).arg(&(nblocks as i32)).arg(&(cshift as i32))
                .arg(k_q8).arg(k_sc).arg(v_q8).arg(v_sc)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ㊶ KV 量化（E4M3）版带掩码注意力（host_fallback / dense 回退路径用，有损、门控）。
    #[allow(clippy::too_many_arguments)]
    pub fn attn_masked_kvq8_bf16(
        &self, q: &CudaSlice<bf16>, k: &CudaSlice<bf16>, v: &CudaSlice<bf16>,
        mask: &CudaSlice<f32>, out: &mut CudaSlice<bf16>,
        heads: usize, kv_heads: usize, t: usize, tk: usize, head_dim: usize, scale: f32,
        k_q8: &CudaSlice<u8>, k_sc: &CudaSlice<f32>,
        v_q8: &CudaSlice<u8>, v_sc: &CudaSlice<f32>,
    ) -> Result<()> {
        assert!(tk > 0 && head_dim == 256, "attn_masked_kvq8 仅支持 head_dim=256");
        const ATT3_R: usize = 4;
        if t >= ATT3_R {
            let smem = ATT3_R * 512 * 4 + 32 * 4;
            let cfg = LaunchConfig {
                grid_dim: (((t + ATT3_R - 1) / ATT3_R) as u32, heads as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            unsafe {
                self.stream.launch_builder(&self.attn3_masked_kvq8_bf16)
                    .arg(q).arg(k).arg(v).arg(mask).arg(out)
                    .arg(&(heads as i32)).arg(&(kv_heads as i32))
                    .arg(&(t as i32)).arg(&(tk as i32))
                    .arg(&(head_dim as i32)).arg(&scale)
                    .arg(k_q8).arg(k_sc).arg(v_q8).arg(v_sc)
                    .launch(cfg)?;
            }
            return Ok(());
        }
        let tile = tk.min(4096);
        let cfg = LaunchConfig {
            grid_dim: (t as u32, heads as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: (tile * 4 + 32 * 4) as u32,
        };
        unsafe {
            self.stream.launch_builder(&self.attn2_masked_kvq8_bf16)
                .arg(q).arg(k).arg(v).arg(mask).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(t as i32)).arg(&(tk as i32))
                .arg(&(head_dim as i32)).arg(&scale).arg(&(tile as i32))
                .arg(k_q8).arg(k_sc).arg(v_q8).arg(v_sc)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ㊶ bf16 行组量化到 E4M3：`x[rows, heads, head_dim]` → 字节 `q`（同形）+
    /// 每 (row, head) scale `sc[rows*heads]`（= amax/448，全零行 = 1）。有损、门控。
    /// `q`/`sc` 取**视图**（调用方常传 cache 的行区间切片）。
    pub fn quant_rows_e4m3_bf16(
        &self, x: &CudaSlice<bf16>, q: &mut cudarc::driver::CudaViewMut<'_, u8>,
        sc: &mut cudarc::driver::CudaViewMut<'_, f32>,
        rows: usize, heads: usize, head_dim: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: ((rows * heads) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.quant_rows_e4m3_bf16)
                .arg(x).arg(q).arg(sc)
                .arg(&(rows as i32)).arg(&(heads as i32)).arg(&(head_dim as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    // ───────────────────────── 权重量化解码 ─────────────────────────

    /// `row-split-k128-v1` + `Q4G64_F16S` → bf16：`payload → out[n, k]`。
    pub fn dequant_rowsplit_q4g64_bf16(
        &self, payload: &CudaSlice<u8>, out: &mut CudaSlice<bf16>,
        n: usize, k: usize, scale_offset: u64,
    ) -> Result<()> {
        let k_pad = ((k as u64 + 127) / 128 * 128) as i32;
        let total = n * k;
        let cfg = LaunchConfig { grid_dim: (Self::grid(total), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.dequant_rowsplit_q4g64_bf16)
                .arg(payload).arg(out)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(scale_offset as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 同 [`Self::dequant_rowsplit_q4g64_bf16`]，但源/目标取**视图**。
    /// 用于把一批专家的 Q4 blob 一次 H2D 暂存进大 staging 缓冲后，逐槽解量化
    /// （配合 `ArenaExpertSource`：省掉 host 聚合拷贝）。
    pub fn dequant_rowsplit_q4g64_bf16_view(
        &self,
        payload: &cudarc::driver::CudaView<'_, u8>,
        out: &mut cudarc::driver::CudaViewMut<'_, bf16>,
        n: usize,
        k: usize,
        scale_offset: u64,
    ) -> Result<()> {
        let k_pad = ((k as u64 + 127) / 128 * 128) as i32;
        let total = n * k;
        let cfg = LaunchConfig { grid_dim: (Self::grid(total), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.dequant_rowsplit_q4g64_bf16)
                .arg(payload).arg(out)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(scale_offset as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Q3G128 版 [`Self::dequant_rowsplit_q4g64_bf16_view`]（混档 6.2-2 逐专家回退路径）。
    pub fn dequant_rowsplit_q3g128_bf16_view(
        &self,
        payload: &cudarc::driver::CudaView<'_, u8>,
        out: &mut cudarc::driver::CudaViewMut<'_, bf16>,
        n: usize,
        k: usize,
        scale_offset: u64,
    ) -> Result<()> {
        let k_pad = ((k as u64 + 127) / 128 * 128) as i32;
        let total = n * k;
        let cfg = LaunchConfig { grid_dim: (Self::grid(total), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.dequant_rowsplit_q3g128_bf16)
                .arg(payload).arg(out)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(scale_offset as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    // ───────────────────────── GDN（线性注意力） ─────────────────────────
    /// GDN 递推核心（gated delta rule + RMSNormGated），**就地**推进 `s_all` T 个 token。
    /// 与 host 参考 `recurrent_core` 逐条对应；`gate_act`：0=silu 1=sigmoid。
    #[allow(clippy::too_many_arguments)]
    /// ★ P15-㉟ 并发批：`n_seq > 1` 时一个 batch 里 N 条序列各推 `t` 行（状态 seq-major
    /// 连续排列，输入行按 seq 交错）。`n_seq == 1` 时与旧行为**逐位一致**。
    pub fn gdn_recurrent_bf16(
        &self, conv: &CudaSlice<bf16>, z: &CudaSlice<bf16>,
        a: &CudaSlice<bf16>, b: &CudaSlice<bf16>, norm_w: &CudaSlice<bf16>,
        a_log: &CudaSlice<f32>, dt_bias: &CudaSlice<f32>,
        s_all: &mut CudaSlice<f32>, gated: &mut CudaSlice<bf16>,
        t: usize, hk: usize, hv: usize, num_v_heads: usize, num_k_heads: usize,
        gate_act: i32, row_off: usize, n_seq: usize,
    ) -> Result<()> {
        // P15-㉘：S 一次性驻留动态 smem（≤96KB 时启用；实测批量预载后全局路径 18µs 胜出）
        // ⇒ smem 默认关，仅 LINGXI_GDN_SMEM=1 时开（留档 A/B）。
        let smem = hk * hv * 4;
        let use_smem = smem <= 96 * 1024 && std::env::var("LINGXI_GDN_SMEM").is_ok();
        let cfg = LaunchConfig {
            grid_dim: (num_v_heads as u32 * n_seq.max(1) as u32, 1, 1),
            block_dim: (hv as u32, 1, 1),
            shared_mem_bytes: if use_smem { smem as u32 } else { 0 },
        };
        unsafe {
            self.stream
                .launch_builder(&self.gdn_recurrent_bf16)
                .arg(conv).arg(z).arg(a).arg(b).arg(norm_w)
                .arg(a_log).arg(dt_bias).arg(s_all).arg(gated)
                .arg(&(t as i32)).arg(&(hk as i32)).arg(&(hv as i32))
                .arg(&(num_v_heads as i32)).arg(&(num_k_heads as i32)).arg(&gate_act)
                .arg(&(row_off as i32)).arg(&(use_smem as i32)).arg(&(n_seq as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// batched 版 [`Self::gemm_bf16_nt`]：`c[s][m,n] = a[s][m,k] · w[s][n,k]^T`。
    /// 三个 stride 都是**按元素**的 batch 步长；`stride_a = 0` 表示所有 batch 共用同一个 A
    /// （decode 时所有专家共用同一个 token 向量）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16_nt_batched<A, W, C>(
        &self, a: &A, w: &W, c: &mut C,
        m: u32, n: u32, k: u32, batch: u32,
        stride_a: i64, stride_w: i64, stride_c: i64,
    ) -> Result<()>
    where
        A: DevicePtr<bf16>,
        W: DevicePtr<bf16>,
        C: DevicePtrMut<bf16>,
    {
        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: bf16::from_f32(1.0),
                lda: k as i32,
                ldb: k as i32,
                ldc: n as i32,
                beta: bf16::from_f32(0.0),
            },
            batch_size: batch as i32,
            stride_a: stride_w, // cuBLAS 的 A = w
            stride_b: stride_a, // cuBLAS 的 B = a
            stride_c,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, w, a, c)?; }
        Ok(())
    }

    /// 融合反量化的 Q4G64 GEMV（从设备槽直读，不落 bf16 中转）。
    /// `a_stride = 0`：所有专家共用同一个 `x`（gu 投影）；`= k`：每专家用各自 `x`（dn 投影）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slots_bf16(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaSlice<i32>, x: &CudaSlice<bf16>,
        y: &mut CudaSlice<bf16>, s: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64,
    ) -> Result<()> {
        let rows_per_warp = 1usize;
        let warps_per_block = (BLOCK / 32) as usize;
        let rows_per_block = warps_per_block * rows_per_warp;
        let blocks = ((n + rows_per_block - 1) / rows_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, s as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_q4g64_slots_bf16)
                .arg(slots).arg(slot_ids).arg(x).arg(y)
                .arg(&(s as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 同上的**每 warp 多行**版：`rows_per_warp ∈ {1,2,4,8}`，每个档位对应一个**独立核符号**
    /// （各自独立的寄存器分配/占用率；绝不能合并成一个核用 switch 分派，见 dequant_ops.cu 注释）。
    /// `rows_per_warp == 1` 与逐行版**逐位等价**。用于 dn 这类「行内字节少 → 全局 warp 数爆炸」
    /// 的形状：靠减少线程数 + 跨行访存并行拉带宽。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slots_rows_bf16(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaSlice<i32>, x: &CudaSlice<bf16>,
        y: &mut CudaSlice<bf16>, s: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize,
    ) -> Result<()> {
        let ids_v = slot_ids.slice(0..slot_ids.len());
        let x_v = x.slice(0..x.len());
        let mut y_v = y.slice_mut(0..y.len());
        self.gemv_q4g64_slots_rows_bf16_view(
            slots, &ids_v, &x_v, &mut y_v, s, n, k, k_pad, a_stride, blob, seg_off, scale_off,
            rows_per_warp,
        )
    }

    /// ㊼ Q4 专家**张量核** GEMM（Volta `mma.sync.m8n8k4`，QPN 8×32 映射）。
    /// 一个 CTA 服务一个专家（`blockIdx.y`），`t` = 该专家命中行数（≤8 用 t1，9..16 用 t2）。
    /// 存储布局与 [`Self::gemv_q4g64_slotsm_rpw_bf16`] 完全相同（group64 / 32 code B / 2 scale B），
    /// 所以**不需要换权重产物**；数值是"fp16(权重×scale) 进张量核"，比 SIMT 的 f32 scale 差 ~2^-11
    /// （远小于 bf16 输出舍入）⇒ 有损档，调用方负责门控与质量验证。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slots_tc_bf16(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        group_off: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, s: usize, n: usize, k: usize,
        a_stride: usize, k_pad: i32, blob: u64, seg_off: u64, scale_off: u64,
        tiles: usize,
    ) -> Result<()> {
        let fun = if tiles >= 2 { &self.q4_tc_t2 } else { &self.q4_tc_t1 };
        let x_v = x.slice(0..x.len());
        let mut y_v = y.slice_mut(0..y.len());
        let blocks = ((n + 31) / 32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, s as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(fun)
                .arg(slots)
                .arg(slot_ids)
                .arg(group_off)
                .arg(&x_v)
                .arg(&mut y_v)
                .arg(&(n as i32))
                .arg(&(k as i32))
                .arg(&(a_stride as i32))
                .arg(&k_pad)
                .arg(&(blob as i64))
                .arg(&(seg_off as i64))
                .arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ㊽ Q4 专家张量核 GEMM —— **prefill 大 tile 版**（`LINGXI_MOE_TC=1` 且 rows>16 时用）。
    /// 与 [`Self::gemv_q4g64_slots_tc_bf16`] 同布局（group64，无需换权重产物），区别：
    /// CTA = 64 token 行 × 128 输出行（2×4 warp，每 warp QPN 32×32），M 轴按
    /// `ceil(tmax/64)` 进 grid.z，**任意大的组一次 launch 全覆盖**；x 按 k-group 进
    /// shared 一次 CVT fp16（不再每 mma 现转）。数值同 TC 档（有损 ~2^-11），
    /// 判据 = 首 token 相等 + 质量抽检。`tmax` = 本区间最大组行数（host 预算 grid.z）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slots_tcp_bf16_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        group_off: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, e_uniq: usize, n: usize, k: usize,
        a_stride: usize, k_pad: i32, blob: u64, seg_off: u64, scale_off: u64, tmax: usize,
    ) -> Result<()> {
        let x_v = x.slice(0..x.len());
        let mut y_v = y.slice_mut(0..y.len());
        let blocks = ((n + 127) / 128).max(1);
        let zblocks = ((tmax + 63) / 64).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, zblocks as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.q4_tcp)
                .arg(slots)
                .arg(slot_ids)
                .arg(group_off)
                .arg(&x_v)
                .arg(&mut y_v)
                .arg(&(n as i32))
                .arg(&(k as i32))
                .arg(&(a_stride as i32))
                .arg(&k_pad)
                .arg(&(blob as i64))
                .arg(&(seg_off as i64))
                .arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 同 [`Self::gemv_q4g64_slots_rows_bf16`]，但 `slot_ids`/`x`/`y` 取**视图**——
    /// ㊶ hit/miss 重叠按区间拆 launch 时切子段传（每对的输出区间互不相交 ⇒ 逐位不变）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slots_rows_bf16_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        x: &cudarc::driver::CudaView<'_, bf16>, y: &mut cudarc::driver::CudaViewMut<'_, bf16>,
        s: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize,
    ) -> Result<()> {
        let (fun, rpw) = match rows_per_warp {
            2 => (&self.gemv_q4g64_slots_r2_bf16, 2usize),
            4 => (&self.gemv_q4g64_slots_r4_bf16, 4usize),
            8 => (&self.gemv_q4g64_slots_r8_bf16, 8usize),
            _ => (&self.gemv_q4g64_slots_bf16, 1usize),
        };
        let warps_per_block = (BLOCK / 32) as usize;
        let rows_per_block = warps_per_block * rpw;
        let blocks = ((n + rows_per_block - 1) / rows_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, s as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(fun)
                .arg(slots).arg(slot_ids).arg(x).arg(y)
                .arg(&(s as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// bf16 → fp32 整段转换（`slotsm` 的 `xf` 预转输入，P15-⑤b ①"提前转化"）。
    /// `n` = 元素数；`y` 长度必须 ≥ n。转换精确 ⇒ 下游 GEMV 逐位不变。
    pub fn convert_bf16_f32(
        &self, x: &CudaSlice<bf16>, y: &mut CudaSlice<f32>, n: usize,
    ) -> Result<()> {
        anyhow::ensure!(x.len() >= n && y.len() >= n, "convert_bf16_f32: 长度不足");
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(n / 8 + 1), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.convert_bf16_f32)
                .arg(x).arg(y).arg(&(n as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **按唯一专家分组**的专家槽 Q4 GEMV（批量路径的关键）：`blockIdx.y` = 唯一专家，
    /// 块内循环它命中的 `m` 个 token，权重行只读一遍。`group_off` 长度 `E+1`；
    /// `y` 仍按"对"顺序索引（与 `gemv_q4g64_slots_bf16` 的输出布局一致）。
    /// t=1 时每个专家 m=1，与上一版逐位等价。
    /// `xf` = x 的 **fp32 预转缓冲**（`[对数, k]`，行序与对序一致；0 = 用 bf16 x 现转，
    /// 位等价：bf16→f32 精确，z 次序与累加次序不变 —— P15-⑤b ①）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slotsm_bf16(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaSlice<i32>, x: &CudaSlice<bf16>,
        group_off: &CudaSlice<i32>, y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, xf: u64,
    ) -> Result<()> {
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_q4g64_slotsm_bf16)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(&xf)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **多行/warp 版**（`rows_per_warp` = 1 或 2）：与 [`Self::gemv_q4g64_slotsm_bf16`] 位等价，
    /// 只是让一个 warp 顺序扫 `rows_per_warp` 行（行距 = total_warps），用更少的 warp 数换更多
    /// 跨行访存并行 —— 与 t=1 路径的 `LINGXI_Q4_SLOTS_RPW` 同一手法。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slotsm_rpw_bf16(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaSlice<i32>, x: &CudaSlice<bf16>,
        group_off: &CudaSlice<i32>, y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize, xf: u64,
    ) -> Result<()> {
        // 整块转成视图后走同一份实现（地址与长度一致 ⇒ 行为逐位不变）。
        let ids = slot_ids.slice(0..slot_ids.len());
        let goff = group_off.slice(0..group_off.len());
        self.gemv_q4g64_slotsm_rpw_bf16_view(
            slots, &ids, x, &goff, y, e_uniq, n, k, k_pad, a_stride, blob, seg_off, scale_off,
            rows_per_warp, xf,
        )
    }

    /// 与 [`Self::gemv_q4g64_slotsm_rpw_bf16`] 相同，但 `slot_ids` / `group_off` 可以是
    /// **任意视图** —— `LINGXI_MOE_OVERLAP` 要把"组的连续区间"拆成几次 launch，靠切片传进来
    /// （`LINGXI_ARENA_NOH2D` 之外的形状不变 ⇒ 位等价）。**记好口径**：核里每组的读写在
    /// **绝对对下标**上（`xb = x + s0*a_stride`、`y[(s0+s)*n + row]`），所以切片改的只是
    /// "哪几组参与、谁先跑"，不改任何一条算式的次序。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slotsm_rpw_bf16_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, group_off: &cudarc::driver::CudaView<'_, i32>,
        y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize, xf: u64,
    ) -> Result<()> {
        let (fun, rpw) = match rows_per_warp {
            2 => (&self.gemv_q4g64_slotsm_r2_bf16, 2usize),
            4 => (&self.gemv_q4g64_slotsm_r4_bf16, 4usize),
            _ => (&self.gemv_q4g64_slotsm_bf16, 1usize),
        };
        let warps_per_block = (BLOCK / 32) as usize;
        let rows_per_block = warps_per_block * rpw;
        let blocks = ((n + rows_per_block - 1) / rows_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(fun)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(&xf)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Q3G128 版 `gemv_q4g64_slots_rows_bf16`：同样的槽/参数口径，权重按 dense LSB-first
    /// 3-bit 位流（48 B/128 值）解码。`rows_per_warp ∈ {1,2,4,8}`，各档独立核符号。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slots_rows_bf16(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaSlice<i32>, x: &CudaSlice<bf16>,
        y: &mut CudaSlice<bf16>, s: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize,
    ) -> Result<()> {
        let ids_v = slot_ids.slice(0..slot_ids.len());
        let x_v = x.slice(0..x.len());
        let mut y_v = y.slice_mut(0..y.len());
        self.gemv_q3g128_slots_rows_bf16_view(
            slots, &ids_v, &x_v, &mut y_v, s, n, k, k_pad, a_stride, blob, seg_off, scale_off,
            rows_per_warp,
        )
    }

    /// Q3G128 版 `gemv_q4g64_slots_rows_bf16_view`（㊶ hit/miss 拆段切子段传）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slots_rows_bf16_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        x: &cudarc::driver::CudaView<'_, bf16>, y: &mut cudarc::driver::CudaViewMut<'_, bf16>,
        s: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize,
    ) -> Result<()> {
        let (fun, rpw) = match rows_per_warp {
            2 => (&self.gemv_q3g128_slots_r2_bf16, 2usize),
            4 => (&self.gemv_q3g128_slots_r4_bf16, 4usize),
            8 => (&self.gemv_q3g128_slots_r8_bf16, 8usize),
            _ => (&self.gemv_q3g128_slots_bf16, 1usize),
        };
        let warps_per_block = (BLOCK / 32) as usize;
        let rows_per_block = warps_per_block * rpw;
        let blocks = ((n + rows_per_block - 1) / rows_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, s as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(fun)
                .arg(slots).arg(slot_ids).arg(x).arg(y)
                .arg(&(s as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Q3G128 版 `gemv_q4g64_slotsm_rpw_bf16`：按唯一专家分组的批量 GEMV（权重行只读一遍）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slotsm_rpw_bf16(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaSlice<i32>, x: &CudaSlice<bf16>,
        group_off: &CudaSlice<i32>, y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize, xf: u64,
    ) -> Result<()> {
        // 整块转成视图后走同一份实现（地址与长度一致 ⇒ 行为逐位不变）。
        let ids = slot_ids.slice(0..slot_ids.len());
        let goff = group_off.slice(0..group_off.len());
        self.gemv_q3g128_slotsm_rpw_bf16_view(
            slots, &ids, x, &goff, y, e_uniq, n, k, k_pad, a_stride, blob, seg_off, scale_off,
            rows_per_warp, xf,
        )
    }

    /// Q3G128 版 `gemv_q4g64_slotsm_rpw_bf16_view`：`slot_ids`/`group_off` 可切视图拆 launch。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slotsm_rpw_bf16_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, group_off: &cudarc::driver::CudaView<'_, i32>,
        y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize, xf: u64,
    ) -> Result<()> {
        let (fun, rpw) = match rows_per_warp {
            2 => (&self.gemv_q3g128_slotsm_r2_bf16, 2usize),
            4 => (&self.gemv_q3g128_slotsm_r4_bf16, 4usize),
            _ => (&self.gemv_q3g128_slotsm_bf16, 1usize),
        };
        let warps_per_block = (BLOCK / 32) as usize;
        let rows_per_block = warps_per_block * rpw;
        let blocks = ((n + rows_per_block - 1) / rows_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(fun)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(&xf)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// bf16 → int8 **行量化**（`LINGXI_MOE_DP4A` 的"提前转化"）：每行 scale = amax/127。
    /// `x`/`y` 按 `[rows, cols]` 行主序；`sx` 长度 ≥ rows。
    pub fn convert_bf16_q8_rs(
        &self, x: &CudaSlice<bf16>, y: &mut CudaSlice<u8>, sx: &mut CudaSlice<f32>,
        rows: usize, cols: usize,
    ) -> Result<()> {
        anyhow::ensure!(x.len() >= rows * cols && y.len() >= rows * cols && sx.len() >= rows,
                        "convert_bf16_q8_rs: 长度不足");
        let cfg = LaunchConfig {
            grid_dim: (1, rows as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.convert_bf16_q8_rs)
                .arg(x).arg(y).arg(sx).arg(&(cols as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `slotsm` 的 **dp4a/int8-x** 变体（P15-㉖）：结构同
    /// [`Self::gemv_q4g64_slotsm_rpw_bf16_view`]，但 x 走 int8（`xq`，每行 scale `sx`），
    /// 内积用 `__dp4a`。**非位等价**（x 量化到 int8）⇒ 调用方必须已用判据验收
    /// （`LINGXI_MOE_DP4A` 开关，默认关）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slotsm_rpw_dp4a_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, group_off: &cudarc::driver::CudaView<'_, i32>,
        y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize,
        xq: &CudaSlice<u8>, sx: &CudaSlice<f32>,
    ) -> Result<()> {
        let fun = match rows_per_warp {
            2 => &self.gemv_q4g64_slotsm_r2_dp4a,
            _ => &self.gemv_q4g64_slotsm_dp4a,
        };
        let rpw = if rows_per_warp == 2 { 2usize } else { 1usize };
        let warps_per_block = (BLOCK / 32) as usize;
        let rows_per_block = warps_per_block * rpw;
        let blocks = ((n + rows_per_block - 1) / rows_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(fun)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(xq).arg(sx)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 反量化（row-split Q4G64，**批量版**）：设备槽 → 紧凑 `[S, n, k]` bf16。
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_rowsplit_q4g64_batch(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaSlice<i32>, out: &mut CudaSlice<bf16>,
        s: usize, n: usize, k: usize, blob: u64, seg_off: u64, scale_off: u64,
    ) -> Result<()> {
        let k_pad = ((k as u64 + 127) / 128 * 128) as i32;
        let total = n * k;
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(total), s as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.dequant_rowsplit_q4g64_batch)
                .arg(slots).arg(slot_ids)
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(out)
                .arg(&(s as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    // ───────────────── QSA indexer（GPU 选块，替换 host 路径） ─────────────────
    /// 池化一个压缩块：`pooled[d] = bf16(mean_j rk[row0+j][d])`。
    pub fn qsa_pool_block_bf16(
        &self, rk: &CudaSlice<bf16>, pooled: &mut CudaSlice<bf16>,
        row0: usize, compress: usize, hd: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (Self::grid(hd), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.qsa_pool_block_bf16)
                .arg(rk).arg(pooled)
                .arg(&(row0 as i32)).arg(&(compress as i32)).arg(&(hd as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **批量池化** `nb` 个新块（一次 launch）：`pooled[b][d] = bf16(mean_j rk[b*compress+j][d])`。
    pub fn qsa_pool_blocks_bf16(
        &self, rk: &CudaSlice<bf16>, pooled: &mut CudaSlice<bf16>,
        b0: usize, nb: usize, compress: usize, hd: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(hd), nb as u32, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_pool_blocks_bf16)
                .arg(rk).arg(pooled)
                .arg(&(b0 as i32)).arg(&(nb as i32)).arg(&(compress as i32)).arg(&(hd as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 池化块 RoPE：行位置 = `pos_base + row * pos_stride`。
    #[allow(clippy::too_many_arguments)]
    pub fn rope_partial_stride_bf16(
        &self, x: &mut CudaSlice<bf16>, cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
        rows: usize, head_dim: usize, rot_dim: usize, pos_base: usize, pos_stride: usize,
    ) -> Result<()> {
        let half = rot_dim / 2;
        let total = rows * half;
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(total), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.rope_partial_stride_bf16)
                .arg(x).arg(cos).arg(sin)
                .arg(&(rows as i32)).arg(&(head_dim as i32)).arg(&(rot_dim as i32))
                .arg(&(pos_base as i32)).arg(&(pos_stride as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 打分：`scores[row][b] = (Σ_h max(Σ_d q[row,h,d]·pooled[b,d], 0)) / √hd`。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_indexer_scores_bf16(
        &self, q: &CudaSlice<bf16>, pooled: &CudaSlice<bf16>, scores: &mut CudaSlice<f32>,
        t_q: usize, nblocks: usize, heads: usize, hd: usize, sqrt_hd: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (Self::grid(nblocks), t_q as u32, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.qsa_indexer_scores_bf16)
                .arg(q).arg(pooled).arg(scores)
                .arg(&(t_q as i32)).arg(&(nblocks as i32)).arg(&(heads as i32)).arg(&(hd as i32))
                .arg(&sqrt_hd)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **精确 top-k 选块（O(nblocks)）**：每行一个 block，3 遍打分直方图定阈值，
    /// 输出阈值的 u32 位型与"严格大于阈值的个数"。配合 [`Self::qsa_keep_fill_bf16`]。
    ///
    /// 语义与 [`Self::qsa_indexer_rank_f32`] 完全等价（含并列：同分按块下标升序）。
    /// 不落 scores 缓冲 ⇒ 256K 下省掉 `t×nblocks×4B`（≈537MB/层）。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_keep_select_bf16(
        &self, q: &CudaSlice<bf16>, pooled: &CudaSlice<bf16>,
        thr_key: &mut CudaSlice<u32>, cnt_gt: &mut CudaSlice<i32>,
        t_q: usize, nblocks: usize, heads: usize, hd: usize, sqrt_hd: f32, topk: usize,
    ) -> Result<()> {
        // 每行一个 block；1024 线程尽量摊薄单行的大 nblocks（decode t=1、256K 时 nblocks=65536）。
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_keep_select_bf16)
                .arg(q).arg(pooled).arg(thr_key).arg(cnt_gt)
                .arg(&(nblocks as i32)).arg(&(heads as i32)).arg(&(hd as i32))
                .arg(&sqrt_hd).arg(&(topk as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 填 `keep`（`thr_key`/`cnt_gt` 来自 [`Self::qsa_keep_select_bf16`]）。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_keep_fill_bf16(
        &self, q: &CudaSlice<bf16>, pooled: &CudaSlice<bf16>,
        thr_key: &CudaSlice<u32>, cnt_gt: &CudaSlice<i32>, keep: &mut CudaSlice<u8>,
        t_q: usize, nblocks: usize, heads: usize, hd: usize, sqrt_hd: f32, topk: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_keep_fill_bf16)
                .arg(q).arg(pooled).arg(thr_key).arg(cnt_gt).arg(keep)
                .arg(&(nblocks as i32)).arg(&(heads as i32)).arg(&(hd as i32))
                .arg(&sqrt_hd).arg(&(topk as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 融合选块（2026-10-18）：**一次打分**落 shared，再做 11+11+10 位直方图与填 keep。
    /// 语义与 [`Self::qsa_keep_select_bf16`] + [`Self::qsa_keep_fill_bf16`] 逐位一致。
    /// `nblocks*4B` 走动态 shared；调用方须保证 ≤ 9216（≈36KB，加静态 10KB 仍 < 48KB）。
    pub fn qsa_keep_fused_bf16(
        &self, q: &CudaSlice<bf16>, pooled: &CudaSlice<bf16>, keep: &mut CudaSlice<u8>,
        t_q: usize, nblocks: usize, heads: usize, hd: usize, sqrt_hd: f32, topk: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, 1, 1), block_dim: (256, 1, 1),
            shared_mem_bytes: (nblocks * 4) as u32,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_keep_fused_bf16)
                .arg(q).arg(pooled).arg(keep)
                .arg(&(nblocks as i32)).arg(&(heads as i32)).arg(&(hd as i32))
                .arg(&sqrt_hd).arg(&(topk as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 大 nblocks 选块：分数已在全局（由 `qsa_indexer_scores_bf16` 1 遍算出），
    /// 本核只做 3 遍直方图 + 1 遍填 keep。语义与 radix / 融合核逐位一致。
    pub fn qsa_keep_from_scores_bf16(
        &self, scores: &CudaSlice<f32>, keep: &mut CudaSlice<u8>,
        t_q: usize, nblocks: usize, topk: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_keep_from_scores_bf16)
                .arg(scores).arg(keep)
                .arg(&(nblocks as i32)).arg(&(topk as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// decode 选块（t==1）的并行直方图：`hist[(key>>shift)&bmask] += 1`，只统计
    /// `(key & fmask) == fval` 的元素（调用前 hist 必须清零）。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_hist_f32(
        &self, scores: &CudaSlice<f32>, hist: &mut CudaSlice<u32>,
        nblocks: usize, shift: u32, bmask: u32, fmask: u32, fval: u32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(nblocks), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_hist_f32)
                .arg(scores).arg(hist)
                .arg(&(nblocks as i32)).arg(&(shift as i32)).arg(&bmask).arg(&fmask).arg(&fval)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 后缀计数 → `out[0]=bin, out[1]=严格高于该 bin 的个数`。
    pub fn qsa_bin_finalize(
        &self, hist: &CudaSlice<u32>, nbins: usize, need: i32,
        out: &mut cudarc::driver::CudaViewMut<'_, i32>,
    ) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.qsa_bin_finalize)
                .arg(hist).arg(&(nbins as i32)).arg(&need).arg(out)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 用已算好的 scores + 阈值填 keep（并列按块下标升序）。
    pub fn qsa_keep_fill_scores(
        &self, scores: &CudaSlice<f32>, keep: &mut CudaSlice<u8>,
        nblocks: usize, topk: usize, thr: u32, need_eq: i32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_keep_fill_scores)
                .arg(scores).arg(keep)
                .arg(&(nblocks as i32)).arg(&(topk as i32)).arg(&thr).arg(&need_eq)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 把 `keep` 压成每行的**升序块号列表**（gather 注意力用）。
    pub fn qsa_sel_build(
        &self, keep: &CudaSlice<u8>, sel: &mut CudaSlice<i32>, sel_cnt: &mut CudaSlice<i32>,
        t_q: usize, nblocks: usize, stride: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_sel_build)
                .arg(keep).arg(sel).arg(sel_cnt)
                .arg(&(nblocks as i32)).arg(&(stride as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **gather 稀疏注意力**：只遍历每行的选中块列表（+尾部不完整块），与上下文长度解耦。
    /// 数学 = dense（未选中位贡献 exp=0 精确不参与），仅归约配对顺序不同（~1ulp）。
    /// `nh`：一个 block 合并处理同一 kv-group 的 q-head 数（1 或 4；4 ⇒ K/V 流量 ÷4）。
    #[allow(clippy::too_many_arguments)]
    pub fn attn_gather_bf16(
        &self, q: &CudaView<bf16>, k: &CudaView<bf16>, v: &CudaView<bf16>,
        blk_tab: &CudaView<i32>,
        sel: &CudaView<i32>, sel_cnt: &CudaView<i32>, out: &mut CudaViewMut<bf16>,
        t_q: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        vis_base: usize, nblocks: usize, cshift: usize, stride: usize, nh: usize,
    ) -> Result<()> {
        let ctok = 1usize << cshift;
        let (fun, nhead) = if nh == 4 {
            (&self.attn_gather4_bf16, 4usize)
        } else {
            (&self.attn_gather_bf16, 1usize)
        };
        let smem = (stride * ctok + ctok) * 4 * nhead;
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, (heads / nhead) as u32, 1), block_dim: (256, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        unsafe {
            self.stream.launch_builder(fun)
                .arg(q).arg(k).arg(v).arg(blk_tab).arg(sel).arg(sel_cnt).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(head_dim as i32))
                .arg(&scale).arg(&(vis_base as i32)).arg(&(nblocks as i32))
                .arg(&(cshift as i32)).arg(&(stride as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 名次 → 保留位：`keep[row][b] = rank(row,b) < topk`。
    pub fn qsa_indexer_rank_f32(
        &self, scores: &CudaSlice<f32>, keep: &mut CudaSlice<u8>,
        t_q: usize, nblocks: usize, topk: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (Self::grid(nblocks), t_q as u32, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.qsa_indexer_rank_f32)
                .arg(scores).arg(keep)
                .arg(&(t_q as i32)).arg(&(nblocks as i32)).arg(&(topk as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ㊺ A2 设备路由：单 block 完成 softmax + 插入式 top-k + 槽位查表（数学逐位复刻
    /// moe.rs host 路径）。`miss_flag_dev`/`miss_ids_dev` 是**映射 host 内存的设备别名**
    /// （u64 裸指针，host 自旋事件后直读）。输出按专家升序。
    #[allow(clippy::too_many_arguments)]
    pub fn route_topk_slots_bf16(
        &self, logits: &CudaSlice<bf16>, slot_map_dev: u64,
        ids: &mut CudaSlice<i32>, wts: &mut CudaSlice<f32>,
        slot_ids: &mut CudaSlice<i32>, rows: &mut CudaSlice<i32>,
        host_slots_dev: u64, miss_flag_dev: u64, miss_ids_dev: u64, e: usize, top_k: usize,
    ) -> Result<()> {
        assert!(e <= 1024 && top_k <= 64, "route_topk_slots: E≤1024、top_k≤64");
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.route_topk_slots_bf16)
                .arg(logits).arg(&slot_map_dev)
                .arg(ids).arg(wts).arg(slot_ids).arg(rows)
                .arg(&host_slots_dev).arg(&miss_flag_dev).arg(&miss_ids_dev)
                .arg(&(e as i32)).arg(&(top_k as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 掩码填充：`mask[row][j] = 0`（可见）或 `f32::MIN`（不可见），与 host 参考逐位一致。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_mask_fill_f32(
        &self, keep: &CudaSlice<u8>, mask: &mut CudaSlice<f32>,
        t_q: usize, t_kv: usize, nblocks: usize, compress: usize, len: usize, minv: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (Self::grid(t_kv), t_q as u32, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.qsa_mask_fill_f32)
                .arg(keep).arg(mask)
                .arg(&(t_q as i32)).arg(&(t_kv as i32)).arg(&(nblocks as i32))
                .arg(&(compress as i32)).arg(&(len as i32)).arg(&minv)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 深度可分离因果 1D 卷积 + silu：`x[T,C] / w[K,C] → y[T,C]`。
    /// `dilation`：GDN 卷积为 1；PLE 的 short conv 为 `ngram_size`。
    pub fn causal_conv1d_silu_bf16(
        &self, x: &CudaSlice<bf16>, w: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
        t: usize, c: usize, kernel: usize, dilation: usize,
    ) -> Result<()> {
        let n = t * c;
        let cfg = LaunchConfig { grid_dim: (Self::grid(n), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.causal_conv1d_silu_bf16)
                .arg(x).arg(w).arg(y)
                .arg(&(t as i32)).arg(&(c as i32)).arg(&(kernel as i32)).arg(&(dilation as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **融合版**因果卷积 + 历史推进：一次 launch 取代「hist→cin 拷 / qkv→cin 拷 / conv / 末尾拷回 hist」
    /// 的 **4 次 launch**（`LINGXI_KTS` 实测那段 28 µs/层，数据只有 ~5 µs，几乎全是发射开销）。
    /// 传裸设备指针，方便直接喂子切片/状态缓冲。
    /// ⚠️ `new_hist` **绝不能与 `hist` 同址**：核内「先读 hist、后写 new_hist」跨 block 有写后读竞态
    /// （`new_hist[0..kh)` 与 `hist[1..kh+1)` 重叠），调用方必须双缓冲。
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_silu_hist_bf16(
        &self, hist: u64, cur: u64, w: u64, y: u64, new_hist: u64,
        kh: usize, t: usize, c: usize, kernel: usize, dilation: usize,
    ) -> Result<()> {
        let n = t.max(kh).max(1) * c;
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.causal_conv1d_silu_hist_bf16)
                .arg(&hist).arg(&cur).arg(&w).arg(&y).arg(&new_hist)
                .arg(&(kh as i32)).arg(&(t as i32)).arg(&(c as i32))
                .arg(&(kernel as i32)).arg(&(dilation as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    // ───────────────────────── cuBLAS（复用） ─────────────────────────

    /// 行主序 bf16 GEMM：`C[M,N] = A[M,K] × B[K,N]`（`A/B/C` 可为 slice 视图）。
    pub fn gemm_bf16<A, B, C>(&self, a: &A, b: &B, c: &mut C, m: u32, n: u32, k: u32) -> Result<()>
    where
        A: DevicePtr<bf16>,
        B: DevicePtr<bf16>,
        C: DevicePtrMut<bf16>,
    {
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: bf16::from_f32(1.0),
            lda: n as i32,
            ldb: k as i32,
            ldc: n as i32,
            beta: bf16::from_f32(0.0),
        };
        unsafe { self.blas.gemm(cfg, b, a, c)?; }
        Ok(())
    }

    /// 行主序 bf16 GEMM（**右乘转置**）：`C[M,N] = A[M,K] × W[N,K]^T`。
    ///
    /// 权重在产物里天然是 `[out, in]` 行主序，这正是"行 × 权重"的形态，故最常用。
    pub fn gemm_bf16_nt<A, W, C>(&self, a: &A, w: &W, c: &mut C, m: u32, n: u32, k: u32) -> Result<()>
    where
        A: DevicePtr<bf16>,
        W: DevicePtr<bf16>,
        C: DevicePtrMut<bf16>,
    {
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T, // W 行主序 [N,K] → 列主序 [K,N]，转置后 [N,K]
            transb: cublasOperation_t::CUBLAS_OP_N, // A 行主序 [M,K] → 列主序 [K,M]
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: bf16::from_f32(1.0),
            lda: k as i32,
            ldb: k as i32,
            ldc: n as i32,
            beta: bf16::from_f32(0.0),
        };
        unsafe { self.blas.gemm(cfg, w, a, c)?; }
        Ok(())
    }
}

/// 默认 kernel 选择表：`OpId → KernelId`。换实现只改这里。
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultKernelRegistry;

impl KernelRegistry for DefaultKernelRegistry {
    fn kernel(&self, op: OpId, _shape: ShapeTag) -> KernelId {
        match op {
            OpId::RmsNorm => KernelId::RmsNormF32,
            _ => KernelId::Smoke,
        }
    }
}
