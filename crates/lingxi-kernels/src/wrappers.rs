//! `impl Kernels` 的全部 kernel launch 封装（CODE_MAP Phase 4 刀P4-1
//! 从 lib.rs 搬入，只搬家不改行为）。unsafe 只允许出现在 launch 封装里（crate 铁律）。

use crate::evprof::EvProf;
use anyhow::Result;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtr,
    DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use half::bf16;

use crate::{
    BLOCK,
    Kernels, ATTN_OPS_CUBIN, BF16_SKINNY_EXP_CUBIN, BF16_TCP_CUBIN, Bf16SkinnyExp, DEQUANT_OPS_CUBIN,
    GDN_OPS_CUBIN, GDN_REC_EXP_CUBIN, GdnRecExp, GEMV_PAR_EXP_CUBIN,
    GEMV_STRUCT_EXP_CUBIN, GEMV_Q3_EXP_CUBIN, GemvStructExp, GemvQ3Exp, GR_OPS_CUBIN,
    GR_PROBE_T1_CUBIN, GrProbeT1,
    MLARGE_CUBIN, MOE_OPS_CUBIN, Q3_TCP_CUBIN, Q4_DENSE_EXP_CUBIN, Q4_MLARGE_CUBIN, Q4_TC_CUBIN,
    Q4_TCP_CUBIN, Q4_TCP_DENSE_CUBIN, QSA_PROBE_EXP_CUBIN, QsaProbeExp, RMSNORM_CUBIN,
    SMOKE_CUBIN, TOPK_OPS_CUBIN, ROUTE_FUSE_CUBIN, ROUTER_M6_CUBIN,
};

/// `LINGXI_MOE_GEMV_TIER`（**默认 1** = 新核；设 `0` 回退原核，逐位/逐路径不变）：
/// `1` = `slotsm` 的 Q4 专家 GEMV 改走 `slotsm_t2_bf16`（**与默认核逐位相同**，只改取数结构：
/// 两个独立 load + 移位对解包；实测 gu E40 m=1 317→453 GB/s、线上 E7 m=1 269→374）。
/// 只对 `rows_per_warp==1 && xf==0`（rows=1 的混档 decode 正是这一支）生效，其余形状原样。
/// 2026-10-24 由 opt-in 翻成默认：md5 逐位一致（11+4 跑）、专家段 −8%、tok/s +3%。
fn gemv_tier() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LINGXI_MOE_GEMV_TIER")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1)
    })
}

/// `LINGXI_NO_Q3_T2=1`：Q3 专家 GEMV 回退到旧核（`lingxi_gemv_q3g128_slotsm_body<1>`，PF=4）。
/// 默认走新核（**与旧核逐位相同**，只改取数结构：两对齐 u32 + 移位对解包 + 独立 load）。
fn q3_t2_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LINGXI_NO_Q3_T2").map(|v| v != "0").unwrap_or(false))
}

/// `LINGXI_NO_BF16_FLAT=1`：bf16 瘦长 GEMM（m≤8）回退到 `lingxi_gemm_skinny_m8`
/// （grid.y 行分块形态）。默认走 `_flat` 版（**逐位相同**，只去掉 `r0`/`mt` 的运行期行偏移）。
fn bf16_flat_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LINGXI_NO_BF16_FLAT").map(|v| v != "0").unwrap_or(false))
}

/// `LINGXI_TKPAIR_FUSE=1`：批量设备路由的「top-k + pair 建表」走**合成一发**
/// （`route_fuse.cu`，逐位相同）。**默认关**——2026-10-24 实测为**负结果**：
/// top-k 需要「一行一 block」的行间并行，塞进单 block 后 6 行只剩 6 个 warp ⇒
/// 合成核 26~27 µs，反而比两发（25.5 µs）慢；连链 37.7 → 41.7 µs。
/// 留作可复现的负结果与后续（"末块续算"式零等待融合）的基线。
fn tkpair_fuse_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LINGXI_TKPAIR_FUSE").map(|v| v != "0").unwrap_or(false))
}

/// `LINGXI_GEMV_SMEM_X`（默认 0 = 关，行为逐位不变）：
/// `1` = `slotsm` 的 bf16 档把激活按 block 一次性搬进 shared（每 warp 不再各自重读）；
/// `2` = dp4a/int8 档做同一件事（配合 `LINGXI_MOE_DP4A=1`）。
fn gemv_smem_x() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LINGXI_GEMV_SMEM_X")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// `LINGXI_NO_ROUTE_TOPK_FAST=1`：`route_topk_rows_bf16` 整核回退旧实现（默认走新实现）。
static ROUTE_TOPK_LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// smem-x 的 token 批量上限：按本 launch 的平均组大小估（≤8）。
/// 返回 0 表示**不启用**（所有组 m=1：AR 路径最省的原路，不能退化）。
fn smem_x_cap(y_len: usize, n: usize, e_uniq: usize) -> usize {
    if e_uniq == 0 {
        return 0;
    }
    let s_pairs = y_len / n.max(1);
    if s_pairs <= e_uniq {
        return 0;
    }
    ((s_pairs + e_uniq - 1) / e_uniq).clamp(2, 8)
}

/// `LINGXI_NO_Q4D_R2=1` 回退：稠密 Q4 瘦长 GEMM 走**旧核**（每 warp 一行）。
///
/// 默认（未设/`0`）走 **R2 档**（`dequant_ops.cu` 的 `lingxi_gemm_q4_skinny{,4,ab}_r2_bf16`）：
/// 一个 warp 连算 2 个输出行 ⇒ x 的 smem 读/bf16→f32 被 2 行共用、x 舞台流量与 barrier
/// 次数 ÷2。**与旧核逐位相同**（算式/列集合/累加次序/归约树一字未改），实测 m=6
/// 10240×2560：133 → 173 GB/s（docs §二十七）。
fn no_q4d_r2() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LINGXI_NO_Q4D_R2").map(|v| v != "0").unwrap_or(false))
}

/// R2 档每 block 的输出行数（8 warp × 2 行）与常数（与 CUDA 侧 `Q4G_R2` 一致）。
const Q4D_R2_ROWS: usize = 8 * 2;

/// `LINGXI_NO_GDN_REC_REG=1`：GDN 递推回退旧核（§二十九 state 常驻寄存器版默认开）。
fn no_rec_reg() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LINGXI_NO_GDN_REC_REG").map(|v| v != "0").unwrap_or(false))
}

/// §AG-补3 `LINGXI_GDN_PREFETCH=1`：递推寄存器核的**提前一拍预载**（门/v/z 标量
/// 旋转寄存器 + q/k 行 L2 预取）。逐位等价（纯搬运/预取提示，算术一字未动）；
/// 默认关，A/B 跑赢后翻转。读点 gdn_ops.cu::lingxi_gdn_recurrent_reg_bf16。
fn gdn_prefetch_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LINGXI_GDN_PREFETCH").map(|v| v != "0").unwrap_or(true))
}

/// §AL `LINGXI_GDN_PIPE=1`：递推寄存器核的**流水版**（sync 5→3/步、门提前一拍、
/// 双缓冲 q/k + parity ssum、al 提出循环）。逐位等价（只改调度，算术表达式/累加
/// 次序/shuffle 树一字未动）。**默认开**（A/B 实锤 rec −10.9%），=0 回退（隐含 prefetch 档）。
/// 读点 gdn_ops.cu::gdn_rec_reg_pipe_body。
fn gdn_pipe_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LINGXI_GDN_PIPE").map(|v| v != "0").unwrap_or(true))
}

impl Kernels {
    pub fn load(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        // ── [§AB] `LINGXI_MEMPOOL_KEEP=1`：默认内存池 RELEASE_THRESHOLD 0 → u64::MAX ──
        // 根因（2026-10-27 判决实验）：cudarc 的 `alloc` 在 V100 上走 `cuMemAllocAsync`
        // （默认内存池），池默认阈值 0 ⇒ **每个同步点**把空闲块 cuMemUnmap 还回驱动、
        // 下次分配重新映射触页。prefill t=512 下 s 相关 scratch（xg/gu_all/o_all，
        // 10MB 级）尺寸逐层抖动 + GPU1 层整段 without_pool ⇒ 每层数块 alloc/free churn；
        // 这些延期 VMM 回收在每个同步点（路由 D2H 处是第一个）被集中 drain：
        // host 自旋 25ms 而 GPU 侧仅 0.1ms、minflt 12 万/s、主线程 sys 占 40%
        // （moe-sync-probe 实测）。本开关把阈值提到 MAX，释放的块留在池内复用、
        // 零 VMM 重映射。纯分配器调优：不改任何字节与数值路径；不设 env = 原行为。
        if std::env::var("LINGXI_MEMPOOL_KEEP").map(|v| v == "1").unwrap_or(false) {
            use cudarc::driver::{result, sys};
            let dev: sys::CUdevice = ordinal as _;
            let pool = unsafe { result::device::get_default_mem_pool(dev) }?;
            let mut thresh: u64 = u64::MAX;
            unsafe {
                result::mem_pool::set_attribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut thresh as *mut u64 as *mut core::ffi::c_void,
                )?;
            }
        }
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
        // `LINGXI_EVPROF=1`（或 `LINGXI_EVPROF_TRACE=<路径>`）：建逐段计时事件对
        //（320 对，一次性）。trace 模式隐含开 evprof（per-token JSON 落盘）。
        let evprof = if std::env::var("LINGXI_EVPROF").is_ok()
            || std::env::var("LINGXI_EVPROF_TRACE").is_ok()
        {
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
        let rfuse = ctx.load_module(Ptx::from_binary(ROUTE_FUSE_CUBIN.to_vec()))?;
        let rm6 = ctx.load_module(Ptx::from_binary(ROUTER_M6_CUBIN.to_vec()))?;
        let q4tc = ctx.load_module(Ptx::from_binary(Q4_TC_CUBIN.to_vec()))?;
        let q4tcp = ctx.load_module(Ptx::from_binary(Q4_TCP_CUBIN.to_vec()))?;
        let q3tcp = ctx.load_module(Ptx::from_binary(Q3_TCP_CUBIN.to_vec()))?;
        let mlarge = ctx.load_module(Ptx::from_binary(MLARGE_CUBIN.to_vec()))?;
        let q4ml = ctx.load_module(Ptx::from_binary(Q4_MLARGE_CUBIN.to_vec()))?;
        let q4tcpd = ctx.load_module(Ptx::from_binary(Q4_TCP_DENSE_CUBIN.to_vec()))?;
        let bf16tcp = ctx.load_module(Ptx::from_binary(BF16_TCP_CUBIN.to_vec()))?;
        let exp = ctx.load_module(Ptx::from_binary(GEMV_PAR_EXP_CUBIN.to_vec()))?;
        let stx = ctx.load_module(Ptx::from_binary(GEMV_STRUCT_EXP_CUBIN.to_vec()))?;
        let struct_exp = GemvStructExp {
            probe: [
                stx.load_function("lingxi_q4g64_p0")?,
                stx.load_function("lingxi_q4g64_p1")?,
                stx.load_function("lingxi_q4g64_p2")?,
                stx.load_function("lingxi_q4g64_p3")?,
                stx.load_function("lingxi_q4g64_p4")?,
                stx.load_function("lingxi_q4g64_p5")?,
                stx.load_function("lingxi_q4g64_p6")?,
                stx.load_function("lingxi_q4g64_p7")?,
            ],
            p8_smsc: stx.load_function("lingxi_q4g64_p8_smsc")?,
            p9_smscx: stx.load_function("lingxi_q4g64_p9_smscx")?,
            u128_r1: stx.load_function("lingxi_q4g64_u128_r1")?,
            u128_r2: stx.load_function("lingxi_q4g64_u128_r2")?,
            u128_r4: stx.load_function("lingxi_q4g64_u128_r4")?,
            contig: stx.load_function("lingxi_q4g64_contig")?,
            u128_r2_sm: stx.load_function("lingxi_q4g64_u128_r2_sm")?,
            u128_r1_dp4a: stx.load_function("lingxi_q4g64_u128_r1_dp4a")?,
            tier_unr1: stx.load_function("lingxi_q4g64_tier_unr1")?,
            tier_unr2: stx.load_function("lingxi_q4g64_tier_unr2")?,
            tier_unr4: stx.load_function("lingxi_q4g64_tier_unr4")?,
            tier_unr4_proddec: stx.load_function("lingxi_q4g64_tier_unr4_proddec")?,
            tier_shift4: stx.load_function("lingxi_q4g64_tier_shift4")?,
        };
        let q3x = ctx.load_module(Ptx::from_binary(GEMV_Q3_EXP_CUBIN.to_vec()))?;
        let q4dx = ctx.load_module(Ptx::from_binary(Q4_DENSE_EXP_CUBIN.to_vec()))?;
        let bfx = ctx.load_module(Ptx::from_binary(BF16_SKINNY_EXP_CUBIN.to_vec()))?;
        // GDN 递推的分母探针 + state 常驻候选（`gdn_rec_exp.cu`）
        let grx = ctx.load_module(Ptx::from_binary(GDN_REC_EXP_CUBIN.to_vec()))?;
        let rec_smem = grx.load_function("lingxi_gdn_rec_smem")?;
        rec_smem.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            (128 * 128 * 4) as i32,
        )?;
        let rec_smem_old = grx.load_function("lingxi_gdn_rec_smem_old")?;
        rec_smem_old.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            (128 * 128 * 4) as i32,
        )?;
        let gdn_rec_exp = GdnRecExp {
            prod: grx.load_function("lingxi_gdn_rec_prod")?,
            read2: grx.load_function("lingxi_gdn_rec_read2")?,
            read1: grx.load_function("lingxi_gdn_rec_read1")?,
            copy2: grx.load_function("lingxi_gdn_rec_copy2")?,
            copy1: grx.load_function("lingxi_gdn_rec_copy1")?,
            reg: grx.load_function("lingxi_gdn_rec_reg")?,
            reg_nos: grx.load_function("lingxi_gdn_rec_reg_nos")?,
            reg_tok: grx.load_function("lingxi_gdn_rec_reg_tok")?,
            smem: rec_smem,
            smem_old: rec_smem_old,
            v2: grx.load_function("lingxi_gdn_rec_v2")?,
            hyb: grx.load_function("lingxi_gdn_rec_hyb")?,
            v3: grx.load_function("lingxi_gdn_rec_v3")?,
            nok: grx.load_function("lingxi_gdn_rec_nok")?,
        };
        let bf16_exp = Bf16SkinnyExp {
            m8_read1: bfx.load_function("lingxi_bf16_m8_read1")?,
            m8_read16: bfx.load_function("lingxi_bf16_m8_read16")?,
            m8_alu: bfx.load_function("lingxi_bf16_m8_alu")?,
            m8_prod: bfx.load_function("lingxi_bf16_m8_prod")?,
            m8_r: [
                bfx.load_function("lingxi_bf16_m8_r2")?,
                bfx.load_function("lingxi_bf16_m8_r4")?,
                bfx.load_function("lingxi_bf16_m8_r8")?,
            ],
            m8_r4pf: bfx.load_function("lingxi_bf16_m8_r4pf")?,
            m8_r4x: bfx.load_function("lingxi_bf16_m8_r4x")?,
            row_read16: bfx.load_function("lingxi_bf16_row_read16")?,
            row_alu: bfx.load_function("lingxi_bf16_row_alu")?,
            row_prod: bfx.load_function("lingxi_bf16_row_prod")?,
            row_r2: bfx.load_function("lingxi_bf16_row_r2")?,
            row_r2pf: bfx.load_function("lingxi_bf16_row_r2pf")?,
            q4h_read1: bfx.load_function("lingxi_q4_head_read1")?,
            q4h_read16: bfx.load_function("lingxi_q4_head_read16")?,
            q4h_prod: bfx.load_function("lingxi_q4_head_prod")?,
            q4h_pf: bfx.load_function("lingxi_q4_head_pf")?,
            q4h_r2: bfx.load_function("lingxi_q4_head_r2")?,
            m8_xonly: bfx.load_function("lingxi_bf16_m8_xonly")?,
            m8_r1pf: bfx.load_function("lingxi_bf16_m8_r1pf")?,
            m8_r1x: bfx.load_function("lingxi_bf16_m8_r1x")?,
            m8_r2x: bfx.load_function("lingxi_bf16_m8_r2x")?,
            row_xonly: bfx.load_function("lingxi_bf16_row_xonly")?,
            row_r4: bfx.load_function("lingxi_bf16_row_r4")?,
            m8_prod_r0: bfx.load_function("lingxi_bf16_m8_prod_r0")?,
            m8_m6: bfx.load_function("lingxi_bf16_m8_m6")?,
            m8_r0hoist: bfx.load_function("lingxi_bf16_m8_r0hoist")?,
        };
        let q3_exp = GemvQ3Exp {
            read_probe: q3x.load_function("lingxi_q3_read_probe")?,
            v: [
                q3x.load_function("lingxi_q3g128_exp_unr1_b3_pdec")?,
                q3x.load_function("lingxi_q3g128_exp_unr2_b3_pdec")?,
                q3x.load_function("lingxi_q3g128_exp_unr2_u32_pdec")?,
                q3x.load_function("lingxi_q3g128_exp_unr2_b3_sdec")?,
                q3x.load_function("lingxi_q3g128_exp_unr2_u32_sdec")?,
                q3x.load_function("lingxi_q3g128_exp_unr1_u32_sdec")?,
                q3x.load_function("lingxi_q3g128_exp_unr4_u32_sdec")?,
                q3x.load_function("lingxi_q3g128_exp_unr8_u32_sdec")?,
                q3x.load_function("lingxi_q3g128_exp_mb1_unr1_u32_sdec")?,
                q3x.load_function("lingxi_q3g128_exp_mb1_unr2_u32_sdec")?,
                q3x.load_function("lingxi_q3g128_exp_mb1_unr4_u32_sdec")?,
                q3x.load_function("lingxi_q3g128_exp_mb1_unr2_b3_sdec")?,
            ],
        };
        // QSA 三段的分母探针 + 候选（`qsa_probe_exp.cu`）
        let qpx = ctx.load_module(Ptx::from_binary(QSA_PROBE_EXP_CUBIN.to_vec()))?;
        let qsa_probe = QsaProbeExp {
            attn: [
                qpx.load_function("lingxi_qsa_attn_prod4")?,
                qpx.load_function("lingxi_qsa_attn_readkv4")?,
                qpx.load_function("lingxi_qsa_attn_readk4")?,
                qpx.load_function("lingxi_qsa_attn_readv4")?,
                qpx.load_function("lingxi_qsa_attn_constq4")?,
                qpx.load_function("lingxi_qsa_attn_constkv4")?,
                qpx.load_function("lingxi_qsa_attn_pass1_4")?,
                qpx.load_function("lingxi_qsa_attn_pass2_4")?,
                qpx.load_function("lingxi_qsa_attn_v2_4")?,
                qpx.load_function("lingxi_qsa_attn_v2pf_4")?,
                qpx.load_function("lingxi_qsa_attn_empty4")?,
                qpx.load_function("lingxi_qsa_attn_prod2")?,
                qpx.load_function("lingxi_qsa_attn_prod1")?,
                qpx.load_function("lingxi_qsa_attn_v2pf_2")?,
                qpx.load_function("lingxi_qsa_attn_v2pf_1")?,
                qpx.load_function("lingxi_qsa_attn_readkv_v2_4")?,
                // 16/17 = 保留槽位（旧"空核 NH=1"/"纯读 v2 NH=1"映射仍指向 10/15 号核，
                // 见 `qsa_attn_probe` 的 match；此处加载同句柄占位保持数组下标稳定）
                qpx.load_function("lingxi_qsa_attn_empty4")?,
                qpx.load_function("lingxi_qsa_attn_readkv_v2_4")?,
                qpx.load_function("lingxi_qsa_attn_v3_1")?,
                qpx.load_function("lingxi_qsa_attn_v3_2")?,
                qpx.load_function("lingxi_qsa_attn_v3_4")?,
            ],
            attn_v5: [
                qpx.load_function("lingxi_qsa_attn_v5_1")?,
                qpx.load_function("lingxi_qsa_attn_v5_2")?,
                qpx.load_function("lingxi_qsa_attn_v5_4")?,
            ],
            attn_v6: [
                qpx.load_function("lingxi_qsa_attn_v6_1")?,
                qpx.load_function("lingxi_qsa_attn_v6_2")?,
                qpx.load_function("lingxi_qsa_attn_v6_4")?,
            ],
            attn_v7: [
                qpx.load_function("lingxi_qsa_attn_v7_1")?,
                qpx.load_function("lingxi_qsa_attn_v7_2")?,
                qpx.load_function("lingxi_qsa_attn_v7_4")?,
            ],
            attn_v9: [
                qpx.load_function("lingxi_qsa_attn_v9_1")?,
                qpx.load_function("lingxi_qsa_attn_v9_2")?,
                qpx.load_function("lingxi_qsa_attn_v9_4")?,
            ],
            idx_scores: [
                qpx.load_function("lingxi_qsa_idx_scores_prod")?,
                qpx.load_function("lingxi_qsa_idx_scores_read")?,
                qpx.load_function("lingxi_qsa_idx_scores_cq")?,
                qpx.load_function("lingxi_qsa_idx_scores_cp")?,
                qpx.load_function("lingxi_qsa_idx_scores_v2")?,
                qpx.load_function("lingxi_qsa_idx_scores_v2pf")?,
            ],
        };
        let blas = CudaBlas::new(stream.clone())?;
        // GR t=1 的同形纯读分母（`gr_probe_t1.cu`）
        let grp = ctx.load_module(Ptx::from_binary(GR_PROBE_T1_CUBIN.to_vec()))?;
        let gr_probe_t1 = GrProbeT1 {
            read_row8: grp.load_function("lingxi_probe_read_row8_bf16")?,
            read_warp: grp.load_function("lingxi_probe_read_warp_bf16")?,
            read_stream: grp.load_function("lingxi_probe_read_stream_bf16")?,
        };
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
            gr_down_inject_bf16: gr.load_function("lingxi_gr_down_inject_bf16")?,
            gr_up_mix_fuse_bf16: gr.load_function("lingxi_gr_up_mix_fuse_bf16")?,
            gr_probe_t1,
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
            route_pairs_bf16: moe.load_function("lingxi_route_pairs_bf16")?,
            route_topk_pairs_bf16: rfuse.load_function("lingxi_route_topk_pairs_bf16")?,
            gemm_skinny_row_m6: rm6.load_function("lingxi_gemm_skinny_row_m6_bf16")?,
            route_topk_row_warp: rfuse.load_function("lingxi_route_topk_row_warp_bf16")?,
            gemv_bf16: moe.load_function("lingxi_gemv_bf16")?,
            gemv_bf16_epi: moe.load_function("lingxi_gemv_bf16_epi")?,
            gemv_bf16_block_epi: moe.load_function("lingxi_gemv_bf16_block_epi")?,
            gemv_bf16_row8_epi: moe.load_function("lingxi_gemv_bf16_row8_epi")?,
            q4_tc_t1: q4tc.load_function("lingxi_gemv_q4g64_slots_tc_bf16")?,
            q4_tc_t2: q4tc.load_function("lingxi_gemv_q4g64_slots_tc2_bf16")?,
            q4_tcp: q4tcp.load_function("lingxi_gemv_q4g64_slots_tcp_bf16")?,
            q3_tcp: q3tcp.load_function("lingxi_gemv_q3g128_slots_tcp_bf16")?,
            q4_tcp_dyn: q4tcp.load_function("lingxi_gemv_q4g64_slots_tcp_bf16_dyn")?,
            q3_tcp_dyn: q3tcp.load_function("lingxi_gemv_q3g128_slots_tcp_bf16_dyn")?,
            mlarge: mlarge.load_function("lingxi_gemm_bf16_mlarge")?,
            q4_mlarge: q4ml.load_function("lingxi_gemm_q4_mlarge")?,
            q4_tcp_dense: q4tcpd.load_function("lingxi_gemm_q4_tcp_dense")?,
            q4_tcp_dense_m8: q4tcpd.load_function("lingxi_gemm_q4_tcp_dense_m8")?,
            bf16_tcp: bf16tcp.load_function("lingxi_gemm_bf16_tcp")?,
            gemm_skinny,
            gemm_skinny_row: moe.load_function("lingxi_gemm_skinny_row_bf16")?,
            gemm_skinny_m8_flat: moe.load_function("lingxi_gemm_skinny_m8_flat_bf16")?,
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
            // §AC mwide：小 k 大 m 宽行版（零 smem，v2：v1 动态 smem 版 occupancy 塌方 3× 更慢）。
            gemm_skinny_mwide: moe.load_function("lingxi_gemm_skinny_mwide_bf16")?,
            gemm_skinny_row_gu_swiglu_sg: moe
                .load_function("lingxi_gemm_skinny_row_gu_swiglu_sg_bf16")?,
            gemm_skinny_m8_scale: moe.load_function("lingxi_gemm_skinny_m8_scale_bf16")?,
            gemm_q4_skinny_bf16: dq.load_function("lingxi_gemm_q4_skinny_bf16")?,
            gemm_q4_skinny4_bf16: dq.load_function("lingxi_gemm_q4_skinny4_bf16")?,
            gemm_q4_skinny_ab_bf16: dq.load_function("lingxi_gemm_q4_skinny_ab_bf16")?,
            gemm_q4_skinny_r2_bf16: dq.load_function("lingxi_gemm_q4_skinny_r2_bf16")?,
            gemm_q4_skinny4_r2_bf16: dq.load_function("lingxi_gemm_q4_skinny4_r2_bf16")?,
            gemm_q4_skinny_ab_r2_bf16: dq.load_function("lingxi_gemm_q4_skinny_ab_r2_bf16")?,
            gemm_q4_skinny_dbg: dq.load_function("lingxi_gemm_q4_skinny_dbg")?,
            gemm_q4_s8_bf16: dq.load_function("lingxi_gemm_q4_s8_bf16")?,
            q4_dense_exp: q4dx.load_function("lingxi_q4_dense_probe")?,
            q4_dense_rows_probe: [
                q4dx.load_function("lingxi_q4_dense_rows_r1")?,
                q4dx.load_function("lingxi_q4_dense_rows_r2")?,
                q4dx.load_function("lingxi_q4_dense_rows_r4")?,
                q4dx.load_function("lingxi_q4_dense_rows_r1w16")?,
                q4dx.load_function("lingxi_q4_dense_rows_r2w16")?,
            ],
            bf16_exp,
            gdn_rec_exp,
            qsa_probe,
            ple_gate_bf16: moe.load_function("lingxi_ple_gate_bf16")?,
            add_bf16: moe.load_function("lingxi_add_bf16")?,
            rope_partial_inplace_bf16: attn.load_function("lingxi_rope_partial_inplace_bf16")?,
            rmsnorm_rope_bf16: attn.load_function("lingxi_rmsnorm_rope_bf16")?,
            gemm_bf16_row_pair: attn.load_function("lingxi_gemm_bf16_row_pair")?,
            attn_masked_bf16: attn.load_function("lingxi_attn_masked_bf16")?,
            attn2_masked_bf16: attn.load_function("lingxi_attn2_masked_bf16")?,
            attn2s_sparse_bf16: attn.load_function("lingxi_attn2s_sparse_bf16")?,
            attn3_masked_bf16: attn.load_function("lingxi_attn3_masked_bf16")?,
            attn5_sparse_bf16: attn.load_function("lingxi_attn5_sparse_bf16")?,
            attn2_masked_kvq8_bf16: attn.load_function("lingxi_attn2_masked_kvq8_bf16")?,
            attn2s_sparse_kvq8_bf16: attn.load_function("lingxi_attn2s_sparse_kvq8_bf16")?,
            attn3_masked_kvq8_bf16: attn.load_function("lingxi_attn3_masked_kvq8_bf16")?,
            attn5_sparse_kvq8_bf16: attn.load_function("lingxi_attn5_sparse_kvq8_bf16")?,
            attn2_masked_kvfp8_bf16: attn.load_function("lingxi_attn2_masked_kvfp8_bf16")?,
            attn2s_sparse_kvfp8_bf16: attn.load_function("lingxi_attn2s_sparse_kvfp8_bf16")?,
            attn3_masked_kvfp8_bf16: attn.load_function("lingxi_attn3_masked_kvfp8_bf16")?,
            attn5_sparse_kvfp8_bf16: attn.load_function("lingxi_attn5_sparse_kvfp8_bf16")?,
            quant_rows_fp8g32_bf16: attn.load_function("lingxi_quant_rows_fp8g32_bf16")?,
            quant_rows_fp8g32_res_bf16: attn.load_function("lingxi_quant_rows_fp8g32_res_bf16")?,
            quant_rows_e4m3_bf16: attn.load_function("lingxi_quant_rows_e4m3_bf16")?,
            qsa_pool_block_bf16: attn.load_function("lingxi_qsa_pool_block_bf16")?,
            qsa_pool_blocks_bf16: attn.load_function("lingxi_qsa_pool_blocks_bf16")?,
            qsa_pool_norm_rope_bf16: attn.load_function("lingxi_qsa_pool_norm_rope_bf16")?,
            qsa_keep_all_bf16: attn.load_function("lingxi_qsa_keep_all_bf16")?,
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
            attn_gather2_bf16: attn.load_function("lingxi_attn_gather2_bf16")?,
            attn_gather_v1_bf16: attn.load_function("lingxi_attn_gather_v1_bf16")?,
            attn_gather4_v1_bf16: attn.load_function("lingxi_attn_gather4_v1_bf16")?,
            attn_gather_v3_2_bf16: attn.load_function("lingxi_attn_gather_v3_2_bf16")?,
            attn_gather_v3_4_bf16: attn.load_function("lingxi_attn_gather_v3_4_bf16")?,
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
            gemv_q4g64_slotsm_t2_bf16: dq.load_function("lingxi_gemv_q4g64_slotsm_t2_bf16")?,
            gemv_q4g64_slotsm_t2_ng_bf16:
                dq.load_function("lingxi_gemv_q4g64_slotsm_t2_ng_bf16")?,
            gemv_q4g64_slotsm_dp4a: dq.load_function("lingxi_gemv_q4g64_slotsm_dp4a")?,
            gemv_q4g64_slotsm_r2_dp4a: dq.load_function("lingxi_gemv_q4g64_slotsm_r2_dp4a")?,
            gemv_q4g64_slotsm_sxb_bf16: dq.load_function("lingxi_gemv_q4g64_slotsm_sxb_bf16")?,
            gemv_q4g64_slotsm_sxq_dp4a: dq.load_function("lingxi_gemv_q4g64_slotsm_sxq_dp4a")?,
            gemv_q4g64_slotsm_splitk_bf16: exp
                .load_function("lingxi_gemv_q4g64_slotsm_splitk_bf16")?,
            gemv_splitk_reduce_bf16: exp.load_function("lingxi_gemv_splitk_reduce_bf16")?,
            read_probe_slots: exp.load_function("lingxi_read_probe_slots")?,
            read_probe_flat: exp.load_function("lingxi_read_probe_flat")?,
            struct_exp,
            q3_exp,
            gemv_q3g128_slots_bf16: dq.load_function("lingxi_gemv_q3g128_slots_bf16")?,
            gemv_q3g128_slots_r2_bf16: dq.load_function("lingxi_gemv_q3g128_slots_r2_bf16")?,
            gemv_q3g128_slots_r4_bf16: dq.load_function("lingxi_gemv_q3g128_slots_r4_bf16")?,
            gemv_q3g128_slots_r8_bf16: dq.load_function("lingxi_gemv_q3g128_slots_r8_bf16")?,
            gemv_q3g128_slotsm_bf16: dq.load_function("lingxi_gemv_q3g128_slotsm_bf16")?,
            gemv_q3g128_slotsm_r2_bf16: dq.load_function("lingxi_gemv_q3g128_slotsm_r2_bf16")?,
            gemv_q3g128_slotsm_r4_bf16: dq.load_function("lingxi_gemv_q3g128_slotsm_r4_bf16")?,
            gemv_q3g128_slotsm_t2_bf16: dq.load_function("lingxi_gemv_q3g128_slotsm_t2_bf16")?,
            gemv_q3g128_slotsm_t2_mb1u2_bf16:
                dq.load_function("lingxi_gemv_q3g128_slotsm_t2_mb1u2_bf16")?,
            gemv_q3g128_slotsm_t2_mb1u1_bf16:
                dq.load_function("lingxi_gemv_q3g128_slotsm_t2_mb1u1_bf16")?,
            gemv_q3g128_slotsm_t2_ng_bf16:
                dq.load_function("lingxi_gemv_q3g128_slotsm_t2_ng_bf16")?,
            gemv_q3g128_slotsm_dp4a: dq.load_function("lingxi_gemv_q3g128_slotsm_dp4a")?,
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
            gdn_recurrent_mw_bf16: gdn.load_function("lingxi_gdn_recurrent_mw_bf16")?,
            gdn_recurrent_reg_bf16: gdn.load_function("lingxi_gdn_recurrent_reg_bf16")?,
            argmax_rows_part_bf16: topk.load_function("lingxi_argmax_rows_part_bf16")?,
            argmax_rows_fin: topk.load_function("lingxi_argmax_rows_fin")?,
            topk_rows_part_bf16: topk.load_function("lingxi_topk_rows_part_bf16")?,
            topk_rows_part_fast_bf16: topk.load_function("lingxi_topk_rows_part_fast_bf16")?,
            topk_rows_fin: topk.load_function("lingxi_topk_rows_fin")?,
            route_topk_slots_bf16: topk.load_function("lingxi_route_topk_slots_bf16")?,
            route_topk_rows_bf16: topk.load_function("lingxi_route_topk_rows_bf16")?,
            route_topk_rows_bf16_legacy:
                topk.load_function("lingxi_route_topk_rows_bf16_legacy")?,
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

    /// 微基准专用：**空核对照**。复用 `lingxi_smoke_scale` 但传 `n = 0` ⇒ 一个线程都不做活，
    /// grid/block 由调用方指定成与目标核**完全相同** ⇒ 量到的就是该网格形状的 launch 地板。
    /// （不新增 CUDA 核：`smoke_scale` 的 `if (i < n)` 在 n=0 时恒假。）
    pub fn empty_probe_scale(
        &self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, grid: (u32, u32), block: u32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (grid.0, grid.1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.smoke_scale)
                .arg(x).arg(&1.0f32).arg(y).arg(&0i32).launch(cfg)?;
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

    /// 微基准专用：**同形纯读分母**（`gr_probe_t1.cu`）。
    /// 一行一 block（`grid=n`、`block=1024`），地址流与 `gemv_bf16_row8_epi` 相同，只读不算。
    pub fn probe_read_row8_bf16(&self, w: u64, out: &mut CudaSlice<f32>, n: usize, k: usize) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gr_probe_t1.read_row8)
                .arg(&w).arg(out).arg(&(k as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// 同上，一 warp 一行（地址流 = `gemv_bf16_epi`，`block=1024`）。
    pub fn probe_read_warp_bf16(&self, w: u64, out: &mut CudaSlice<f32>, n: usize, k: usize) -> Result<()> {
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gr_probe_t1.read_warp)
                .arg(&w).arg(out).arg(&(n as i32)).arg(&(k as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// 同字节数的**理想流式读**（每线程 128 位、`grid=ceil(n/8/BLOCK)`）。
    pub fn probe_read_stream_bf16(&self, w: u64, out: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let blocks = ((n + 8 * BLOCK as usize - 1) / (8 * BLOCK as usize)).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gr_probe_t1.read_stream)
                .arg(&w).arg(out).arg(&(n as i64)).launch(cfg)?;
        }
        Ok(())
    }

    /// 通用档（rows 2..8）合发核 1/2：`mix_down` + `block_inject` 一次 launch。
    /// 按行分流（`row < lowrank` 走 down/`silu`，其余走 inject/`2*sigmoid`），
    /// 每行的线程→列映射、累加次序、两级归约树与 `gemm_skinny_row_bf16` **逐字相同**。
    #[allow(clippy::too_many_arguments)]
    pub fn gr_down_inject_bf16(
        &self, wd: &CudaSlice<bf16>, wi: &CudaSlice<bf16>, x: &CudaSlice<bf16>,
        d: &mut CudaSlice<bf16>, injw: &mut CudaSlice<bf16>,
        lowrank: usize, hc: usize, k: usize, m: usize, inv_hc: f32,
    ) -> Result<()> {
        anyhow::ensure!(m >= 1, "gr_down_inject 需要 m≥1（给到 {m}）");
        let cfg = LaunchConfig {
            grid_dim: ((lowrank + hc) as u32, (m as u32 + 7) / 8, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 32 * 8 * 4,
        };
        unsafe {
            self.stream.launch_builder(&self.gr_down_inject_bf16)
                .arg(wd).arg(wi).arg(x).arg(d).arg(injw)
                .arg(&(lowrank as i32)).arg(&(hc as i32))
                .arg(&(k as i32)).arg(&(m as i32)).arg(&inv_hc)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 通用档（rows 2..8）合发核 2/2：`mix_up` + `sigmoid` + `mix_mean` 一次 launch（不落 imw）。
    /// `opb` = 每 block 的隐藏维输出个数（block = `32*hc*opb` 线程，grid = `hidden/opb`）。
    #[allow(clippy::too_many_arguments)]
    pub fn gr_up_mix_fuse_bf16(
        &self, wu: &CudaSlice<bf16>, d: &CudaSlice<bf16>, xn: &CudaSlice<bf16>,
        mixed: &mut CudaSlice<bf16>,
        lowrank: usize, hidden: usize, hc: usize, m: usize, opb: usize,
    ) -> Result<()> {
        anyhow::ensure!((2..=8).contains(&m), "gr_up_mix_fuse 只支持 m∈2..=8（给到 {m}）");
        anyhow::ensure!(opb >= 1 && hidden % opb == 0, "opb 必须整除 hidden（{opb} vs {hidden}）");
        let block = (32 * hc * opb) as u32;
        anyhow::ensure!(block <= 1024, "block 超限（{block}）");
        let cfg = LaunchConfig {
            grid_dim: ((hidden / opb) as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: (lowrank * m * 4) as u32,
        };
        unsafe {
            self.stream.launch_builder(&self.gr_up_mix_fuse_bf16)
                .arg(wu).arg(d).arg(xn).arg(mixed)
                .arg(&(lowrank as i32)).arg(&(hidden as i32)).arg(&(hc as i32))
                .arg(&(m as i32)).arg(&(opb as i32))
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
    ///
    /// 2026-10-24：默认走**寄存器版**（`lingxi_topk_rows_part_fast_bf16`，见 CUDA 侧注释）；
    /// `LINGXI_NO_TOPKROWS_FAST=1` 回退旧核。两者输出逐位相同（`t7` 里对拍）。
    pub fn topk_rows_part_bf16(
        &self, x: &CudaSlice<bf16>, pval: &mut CudaSlice<f32>, pidx: &mut CudaSlice<i32>,
        bad: &mut CudaSlice<i32>, rows: usize, vocab: usize, g: usize, k: usize,
    ) -> Result<()> {
        static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fast = *FAST.get_or_init(|| std::env::var("LINGXI_NO_TOPKROWS_FAST").is_err());
        let (xp, _g) = x.device_ptr(&self.stream);
        let cfg = LaunchConfig {
            grid_dim: (g as u32, rows as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = if fast { &self.topk_rows_part_fast_bf16 } else { &self.topk_rows_part_bf16 };
        unsafe {
            self.stream.launch_builder(f)
                .arg(&xp).arg(pval).arg(pidx).arg(bad)
                .arg(&(vocab as i32)).arg(&(g as i32)).arg(&(k as i32)).launch(cfg)?;
        }
        Ok(())
    }

    /// 旧 `part` 核（`lingxi_topk_rows_part_bf16`）的直通入口：**只给对拍/回退用**
    /// （默认路径已是寄存器版；`LINGXI_NO_TOPKROWS_FAST=1` 可整机回退）。
    pub fn topk_rows_part_bf16_legacy(
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

    /// `gemv_bf16_ptr` 的**指定流**版：router lookahead 预测 GEMV 专用。预测必须
    /// 挂在 compute 流之外（stream2）——路由段末尾有一次全流同步，预测若挂
    /// compute 流会把 ~50µs 加进每层的同步等待。同核同参，仅换发射流。
    pub fn gemv_bf16_ptr_stream(
        &self,
        s: &std::sync::Arc<cudarc::driver::CudaStream>,
        w: u64,
        x: u64,
        y: u64,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            s.launch_builder(&self.gemv_bf16)
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
        if std::env::var("LINGXI_Q4G_DBG").is_ok() {
            anyhow::ensure!((1..=8).contains(&m), "Q4 瘦长 GEMM(dbg) 只支持 1≤m≤8（给到 {m}）");
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
        // 2026-10-22：m 任意大 —— 核按 blockIdx.y 每块接 8 行（逐位等价于旧逐片 launch，
        // 并发块共享 Q4 权重读 ⇒ DRAM 流量 ⌈m/8⌉ 遍 → ~1 遍）。
        anyhow::ensure!(m >= 1, "Q4 瘦长 GEMM 需要 m≥1（给到 {m}）");
        let blocks = ((n + 7) / 8).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, (m as u32 + 7) / 8, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 8 * 1024 * 2,
        };
        // ㊽+9 M≤8 行槽核（`LINGXI_Q4GEMM_S8=1`）：lane=(行槽,列槽) 各累加一行，
        // x smem 读降到 1/6 且 int4 向量化；有损档（归约次序 ≠ 旧核）。
        static S8: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_s8 = *S8.get_or_init(|| std::env::var("LINGXI_Q4GEMM_S8").map(|v| v != "0").unwrap_or(false));
        if use_s8 && (2..=8).contains(&m) {
            unsafe {
                self.stream.launch_builder(&self.gemm_q4_s8_bf16)
                    .arg(&wq).arg(&x).arg(&y)
                    .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                    .arg(&(row_off as i32)).arg(&scale_off)
                    .arg(&epi).arg(&a).arg(&b)
                    .launch(cfg)?;
            }
            return Ok(());
        }
        // ★ 2026-10-24 R2 档（默认）：每 warp 两行的**逐位等价**改写（docs §二十七）。
        if !no_q4d_r2() {
            return self.q4d_r2_launch(wq, x, y, n, k, k_pad, m, row_off, scale_off, epi, a, b);
        }
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

    /// R2 档的四段版（QSA `q | k | gate | v`）。
    #[allow(clippy::too_many_arguments)]
    fn q4d_r2_launch4(
        &self, wq: u64, x: u64, y0: u64, y1: u64, y2: u64, y3: u64,
        n0: usize, n1: usize, n2: usize, n3: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64,
    ) -> Result<()> {
        let n = n0 + n1 + n2 + n3;
        let blocks = ((n + Q4D_R2_ROWS - 1) / Q4D_R2_ROWS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, (m as u32 + 7) / 8, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 8 * 1024 * 2,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_q4_skinny4_r2_bf16)
                .arg(&wq).arg(&x).arg(&y0).arg(&y1).arg(&y2).arg(&y3)
                .arg(&(n0 as i32)).arg(&(n1 as i32)).arg(&(n2 as i32)).arg(&(n3 as i32))
                .arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// R2 档的 `qkvz + a/b` 尾巴版（GDN）。
    #[allow(clippy::too_many_arguments)]
    fn q4d_r2_launch_ab(
        &self, wq: u64, x: u64, yq: u64, wab: u64, ya: u64, yb: u64,
        n_q4: usize, nvh: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64,
    ) -> Result<()> {
        let blocks = (n_q4 + Q4D_R2_ROWS - 1) / Q4D_R2_ROWS + 2 * nvh;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, (m as u32 + 7) / 8, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 8 * 1024 * 2,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_q4_skinny_ab_r2_bf16)
                .arg(&wq).arg(&x).arg(&yq).arg(&wab).arg(&ya).arg(&yb)
                .arg(&(n_q4 as i32)).arg(&(nvh as i32))
                .arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&scale_off)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **R2 档**的稠密 Q4 瘦长 GEMM：与 [`Self::gemm_q4_skinny_bf16_ptr`]（`LINGXI_NO_Q4D_R2=1`）
    /// **逐位相同**，只把每 warp 一行改成每 warp 两行（x 的 smem 读 / bf16→f32 / 舞台流量按 2 行摊薄）。
    /// 生产入口 `gemm_q4_skinny_bf16_ptr` 在开关未设时自动走到这里；这个 pub 版本给 microbench
    /// 做同进程 A/B（`OnceLock` 读到开关后不能在同一进程里换）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_skinny_r2_bf16_ptr(
        &self, wq: u64, x: u64, y: u64, n: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        self.q4d_r2_launch(wq, x, y, n, k, k_pad, m, row_off, scale_off, epi, a, b)
    }

    /// R2 档四段版（供 microbench 做同进程 A/B）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_skinny4_r2_bf16_ptr(
        &self, wq: u64, x: u64, y0: u64, y1: u64, y2: u64, y3: u64,
        n0: usize, n1: usize, n2: usize, n3: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64,
    ) -> Result<()> {
        self.q4d_r2_launch4(wq, x, y0, y1, y2, y3, n0, n1, n2, n3, k, k_pad, m, row_off, scale_off)
    }

    /// R2 档 `qkvz + a/b` 尾巴版（供 microbench 做同进程 A/B）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_skinny_ab_r2_bf16_ptr(
        &self, wq: u64, x: u64, yq: u64, wab: u64, ya: u64, yb: u64,
        n_q4: usize, nvh: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64,
    ) -> Result<()> {
        self.q4d_r2_launch_ab(wq, x, yq, wab, ya, yb, n_q4, nvh, k, k_pad, m, row_off, scale_off)
    }

    /// **R2 档**（默认）的稠密 Q4 瘦长 GEMM：与 [`Self::gemm_q4_skinny_bf16_ptr`] **逐位相同**，
    /// 只把每 warp 一行改成每 warp 两行（x 的 smem 读 / bf16→f32 / 舞台流量按 2 行摊薄）。
    /// 由 `gemm_q4_skinny_bf16_ptr` 在 `LINGXI_NO_Q4D_R2` 未设时自动选中。
    #[allow(clippy::too_many_arguments)]
    fn q4d_r2_launch(
        &self, wq: u64, x: u64, y: u64, n: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        let blocks = ((n + Q4D_R2_ROWS - 1) / Q4D_R2_ROWS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, (m as u32 + 7) / 8, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 8 * 1024 * 2,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_q4_skinny_r2_bf16)
                .arg(&wq).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&(scale_off as i64))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **QSA 的 `q | k | gate | v` 四段合发**（`dequant_ops.cu` 的
    /// `lingxi_gemm_q4_skinny4_bf16`）：与 [`Self::gemm_q4_skinny_bf16_ptr`] 同一个内核体，
    /// 按**全局行号**把结果落回四个输出缓冲（行区间 `n0..n3`）。
    /// 与"同形状的 4 次 `gemm_q4_skinny_bf16_ptr`（row_off = 0 / n0 / n0+n1 / n0+n1+n2）"
    /// **逐位等价**；只是把 4 发并成 1 发（后两段 n=512 只有 64 块 < 80 SM，纯发射地板）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_skinny4_bf16_ptr(
        &self, wq: u64, x: u64, y0: u64, y1: u64, y2: u64, y3: u64,
        n0: usize, n1: usize, n2: usize, n3: usize,
        k: usize, k_pad: i32, m: usize, row_off: usize, scale_off: u64,
    ) -> Result<()> {
        anyhow::ensure!(m >= 1, "Q4 瘦长 GEMM 需要 m≥1（给到 {m}）");
        if !no_q4d_r2() {
            return self.q4d_r2_launch4(
                wq, x, y0, y1, y2, y3, n0, n1, n2, n3, k, k_pad, m, row_off, scale_off,
            );
        }
        let blocks = ((n0 + n1 + n2 + n3 + 7) / 8).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, (m as u32 + 7) / 8, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 8 * 1024 * 2,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_q4_skinny4_bf16)
                .arg(&wq).arg(&x).arg(&y0).arg(&y1).arg(&y2).arg(&y3)
                .arg(&(n0 as i32)).arg(&(n1 as i32)).arg(&(n2 as i32)).arg(&(n3 as i32))
                .arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&scale_off)
                .arg(&0i32).arg(&0.0f32).arg(&0.0f32)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **GDN 的 `qkvz`（Q4，前 `n_q4` 行）+ `a/b`（bf16，`a_b_projection` 的 `2*nvh` 行）合发**
    /// （`dequant_ops.cu` 的 `lingxi_gemm_q4_skinny_ab_bf16`）。
    /// 前 `ceil(n_q4/8)` 个 block 跑 Q4 体（逐字同 `gemm_q4_skinny_bf16`，epi=0），
    /// 后 `2*nvh` 个 block 各跑一行 `gemm_skinny_row_bf16` 的行体（a 的 `nvh` 行 → `ya`，
    /// 再 `nvh` 行 → `yb`）。与"qkvz 一发 + a 一发 + b 一发"逐位等价。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_skinny_ab_bf16_ptr(
        &self, wq: u64, x: u64, yq: u64, wab: u64, ya: u64, yb: u64,
        n_q4: usize, nvh: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64,
    ) -> Result<()> {
        anyhow::ensure!(m >= 1, "Q4 瘦长 GEMM 需要 m≥1（给到 {m}）");
        if !no_q4d_r2() {
            return self.q4d_r2_launch_ab(
                wq, x, yq, wab, ya, yb, n_q4, nvh, k, k_pad, m, row_off, scale_off,
            );
        }
        let blocks = (n_q4 + 7) / 8 + 2 * nvh;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, (m as u32 + 7) / 8, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 8 * 1024 * 2,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_q4_skinny_ab_bf16)
                .arg(&wq).arg(&x).arg(&yq).arg(&wab).arg(&ya).arg(&yb)
                .arg(&(n_q4 as i32)).arg(&(nvh as i32))
                .arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&scale_off)
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
        // 2026-10-22：m 任意大 —— 核按 blockIdx.y 每块接 8 行（逐位等价于旧逐片 launch，
        // 但并发块共享权重读，DRAM 流量 ⌈m/8⌉ 遍 → ~1 遍）。m≤8 且 grid.y=1 时与旧版完全一致。
        anyhow::ensure!(m >= 1, "skinny-row 需要 m≥1（给到 {m}）");
        let cfg = LaunchConfig {
            grid_dim: (n as u32, (m as u32 + 7) / 8, 1),
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
        // 同 `gemm_skinny_row_bf16_ptr`：2026-10-22 起 m 任意大（grid.y 行分块，逐位等价）。
        anyhow::ensure!(m >= 1, "skinny-row-f32x 需要 m≥1（给到 {m}）");
        let cfg = LaunchConfig {
            grid_dim: (n as u32, (m as u32 + 7) / 8, 1),
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

    /// ★ 2026-10-24 shared expert 融合核 1/2：`gate_up[2I,k]` 的 gate 行 i 与 up 行 i+I
    /// 由同一 block 算、归约后各 round 一次再 swiglu；`blockIdx.x == inter` 的块顺带算
    /// shared_gate 行（`sigmoid(v*a)*b`）。与 [skinny_row(epi=0)+swiglu] **逐位等价**。
    /// grid = (inter+1, ceil(m/8))；动态 smem 0（静态 2×32×8×4）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_skinny_row_gu_swiglu_sg_bf16(
        &self, w: u64, wsg: u64, x: u64, y: u64, sg: u64,
        inter: usize, k: usize, m: usize, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!(m >= 1, "gu_swiglu_sg 需要 m≥1（给到 {m}）");
        let cfg = LaunchConfig {
            grid_dim: ((inter as u32 + 1), ((m as u32 + 7) / 8), 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_skinny_row_gu_swiglu_sg)
                .arg(&w).arg(&wsg).arg(&x).arg(&y).arg(&sg)
                .arg(&(inter as i32)).arg(&(k as i32)).arg(&(m as i32))
                .arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ★ 2026-10-24 shared expert 融合核 2/2：`gemm_skinny_m8`（dn）+ row_scale 尾。
    /// 与 [skinny(m8) → row_scale] **逐位等价**（先 round 成 bf16 再乘）。
    /// grid = (ceil(n/8), ceil(m/8))，block = 256（一个 warp 一行输出）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_skinny_m8_scale_bf16(
        &self, w: u64, x: u64, y: u64, s: u64, n: usize, k: usize, m: usize,
    ) -> Result<()> {
        anyhow::ensure!(m >= 1, "skinny_m8_scale 需要 m≥1（给到 {m}）");
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, ((m as u32 + 7) / 8), 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_skinny_m8_scale)
                .arg(&w).arg(&x).arg(&y).arg(&s)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **瘦长 GEMM**（`1 ≤ m ≤ 8`）：一个 warp 一行输出，把权重行**只读一遍**同时算 m 个输入行。
    /// 按 m 特化（m=1..8 各一个内核），否则指令数会随 8 走、权重带宽掉到 1/3。
    /// ★ 2026-10-24 §二十八：`m ≤ 8` 时改走 `lingxi_gemm_skinny_m8_flat_bf16`（grid.y 恒 1 的
    /// 扁平成，**逐位相同**，只去掉 `r0 = blockIdx.y*8` 的运行期行偏移/`mt` 截断：
    /// n=10240 k=320 42.5→28.0 µs、n=2560 k=640 25.0→12.9 µs）。`LINGXI_NO_BF16_FLAT=1` 回退。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_skinny_bf16_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        // 2026-10-22：m 任意大 —— 统一走 M=8 变体 + grid.y 行分块（运行期 m 守卫，
        // 逐位等价于旧 m 特化分派：有效行的 acc 序列与归约树不变）。旧 m1..m7 变体保留
        // 但不再从此入口分发。
        anyhow::ensure!(m >= 1, "瘦长 GEMM 需要 m≥1（给到 {m}）");
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, (m as u32 + 7) / 8, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let flat = m <= 8 && !bf16_flat_off();
        let f = if flat { &self.gemm_skinny_m8_flat } else { &self.gemm_skinny[7] };
        unsafe {
            self.stream.launch_builder(f)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    // 【已删除·负结果留档】稠密 Q4 GEMV 的"行循环版"（限 grid、每 warp 扫多行）——
    // 2026-10-24 实测：n=248320 时旧核 591 GB/s，行循环版按 grid 160→5120 只有
    // 278/437/420/505/518/537 GB/s。**"限 grid 省 block 调度开销"这个猜想是错的，大 grid 更好。**
    // 核、封装、`Kernels` 字段与对拍测试全部删除；细节见 docs/spec-verify-kernel-campaign.md §十八。

    /// §AC mwide：小 k 大 m 瘦长 GEMM（裸指针；逐位等价于 `gemm_skinny_m8` 分块路，
    /// 见 moe_ops.cu 头注）。零 smem，x 走 L1/L2；CTA=4 输出行 × 行块 32，
    /// grid=(⌈n/4⌉, ⌈m/32⌉)。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_skinny_mwide_bf16_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!(m >= 1 && k >= 1, "mwide 需要 m,k≥1（m={m} k={k}）");
        const MC: usize = 32;
        let cfg = LaunchConfig {
            grid_dim: (((n + 3) / 4) as u32, ((m + MC - 1) / MC) as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_skinny_mwide)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
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

    /// **FP8-G32/int8-G32** KV 版跳块稀疏注意力（`LINGXI_KV_FP8` 门控，有损）：
    /// E4M3/int8 数据平面 + 每 32 维一组 f16 scale。分派结构与
    /// [`Self::attn_sparse_kvq8_bf16`] 相同（t≥4 → attn5，否则 attn2s）。
    /// `k_res`/`v_res` + `kv_sink`/`kv_qlo`/`kv_rcap` = KIVI 式残差窗
    /// （bf16 残差平面；`kv_rcap=0` 即关，调用方传 stub）；`f8_mode` 0=E4M3 / 1=int8。
    #[allow(clippy::too_many_arguments)]
    pub fn attn_sparse_kvfp8_bf16(
        &self, q: &CudaSlice<bf16>, k: &CudaSlice<bf16>, v: &CudaSlice<bf16>,
        keep: &CudaSlice<u8>, vis_base: usize, nblocks: usize, cshift: u32,
        out: &mut CudaSlice<bf16>,
        heads: usize, kv_heads: usize, t: usize, tk: usize, head_dim: usize, scale: f32,
        k_f8: &CudaSlice<u8>, k_f8s: &CudaSlice<half::f16>,
        v_f8: &CudaSlice<u8>, v_f8s: &CudaSlice<half::f16>,
        k_res: &CudaSlice<bf16>, v_res: &CudaSlice<bf16>,
        kv_sink: i32, kv_qlo: i32, kv_rcap: i32, f8_mode: i32,
    ) -> Result<()> {
        assert!(tk > 0 && head_dim == 256, "attn_sparse_kvfp8 仅支持 head_dim=256");
        const ATT3_R: usize = 4;
        if t >= ATT3_R {
            let smem = ATT3_R * 512 * 4 + 32 * 4;
            let cfg = LaunchConfig {
                grid_dim: (((t + ATT3_R - 1) / ATT3_R) as u32, heads as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            unsafe {
                self.stream.launch_builder(&self.attn5_sparse_kvfp8_bf16)
                    .arg(q).arg(k).arg(v).arg(keep).arg(out)
                    .arg(&(heads as i32)).arg(&(kv_heads as i32))
                    .arg(&(t as i32)).arg(&(tk as i32))
                    .arg(&(head_dim as i32)).arg(&scale)
                    .arg(&(vis_base as i32)).arg(&(nblocks as i32)).arg(&(cshift as i32))
                    .arg(k_f8).arg(k_f8s).arg(v_f8).arg(v_f8s)
                    .arg(k_res).arg(v_res).arg(&kv_sink).arg(&kv_qlo).arg(&kv_rcap).arg(&f8_mode)
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
            self.stream.launch_builder(&self.attn2s_sparse_kvfp8_bf16)
                .arg(q).arg(k).arg(v).arg(keep).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(t as i32)).arg(&(tk as i32))
                .arg(&(head_dim as i32)).arg(&scale).arg(&(tile as i32))
                .arg(&(vis_base as i32)).arg(&(nblocks as i32)).arg(&(cshift as i32))
                .arg(k_f8).arg(k_f8s).arg(v_f8).arg(v_f8s)
                .arg(k_res).arg(v_res).arg(&kv_sink).arg(&kv_qlo).arg(&kv_rcap).arg(&f8_mode)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **FP8-G32/int8-G32** KV 版带掩码注意力（host_fallback / dense 回退路径用，
    /// 有损、门控）。残差窗参数语义同 [`Self::attn_sparse_kvfp8_bf16`]。
    #[allow(clippy::too_many_arguments)]
    pub fn attn_masked_kvfp8_bf16(
        &self, q: &CudaSlice<bf16>, k: &CudaSlice<bf16>, v: &CudaSlice<bf16>,
        mask: &CudaSlice<f32>, out: &mut CudaSlice<bf16>,
        heads: usize, kv_heads: usize, t: usize, tk: usize, head_dim: usize, scale: f32,
        k_f8: &CudaSlice<u8>, k_f8s: &CudaSlice<half::f16>,
        v_f8: &CudaSlice<u8>, v_f8s: &CudaSlice<half::f16>,
        k_res: &CudaSlice<bf16>, v_res: &CudaSlice<bf16>,
        kv_sink: i32, kv_qlo: i32, kv_rcap: i32, f8_mode: i32,
    ) -> Result<()> {
        assert!(tk > 0 && head_dim == 256, "attn_masked_kvfp8 仅支持 head_dim=256");
        const ATT3_R: usize = 4;
        if t >= ATT3_R {
            let smem = ATT3_R * 512 * 4 + 32 * 4;
            let cfg = LaunchConfig {
                grid_dim: (((t + ATT3_R - 1) / ATT3_R) as u32, heads as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            unsafe {
                self.stream.launch_builder(&self.attn3_masked_kvfp8_bf16)
                    .arg(q).arg(k).arg(v).arg(mask).arg(out)
                    .arg(&(heads as i32)).arg(&(kv_heads as i32))
                    .arg(&(t as i32)).arg(&(tk as i32))
                    .arg(&(head_dim as i32)).arg(&scale)
                    .arg(k_f8).arg(k_f8s).arg(v_f8).arg(v_f8s)
                    .arg(k_res).arg(v_res).arg(&kv_sink).arg(&kv_qlo).arg(&kv_rcap).arg(&f8_mode)
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
            self.stream.launch_builder(&self.attn2_masked_kvfp8_bf16)
                .arg(q).arg(k).arg(v).arg(mask).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(t as i32)).arg(&(tk as i32))
                .arg(&(head_dim as i32)).arg(&scale).arg(&(tile as i32))
                .arg(k_f8).arg(k_f8s).arg(v_f8).arg(v_f8s)
                .arg(k_res).arg(v_res).arg(&kv_sink).arg(&kv_qlo).arg(&kv_rcap).arg(&f8_mode)
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

    /// **FP8-G32/int8-G32** 行组量化（KV FP8 探索档 `LINGXI_KV_FP8`，有损、env 门控）：
    /// `x[rows, heads, head_dim]` → 字节 `q`（同形，E4M3/int8 由 `mode` 定）+
    /// 每 **32 维一组** f16 scale `sc[rows*heads*(head_dim/32)]`（E4M3 = amax/448，
    /// int8 = amax/127，全零组 = 1）。与 [`Self::quant_rows_e4m3_bf16`] 的差别只在
    /// scale 粒度（行级 f32 → 32 维组 f16，对标 radiance fp8_v4/k6_v4）——这是
    /// K8V4 质量翻车后本档的核心改动。`q`/`sc` 取**视图**（调用方常传 cache 的行区间切片）。
    pub fn quant_rows_fp8g32_bf16(
        &self, x: &CudaSlice<bf16>, q: &mut cudarc::driver::CudaViewMut<'_, u8>,
        sc: &mut cudarc::driver::CudaViewMut<'_, half::f16>,
        rows: usize, heads: usize, head_dim: usize, mode: i32,
    ) -> Result<()> {
        assert!(head_dim % 32 == 0, "quant_rows_fp8g32 需要 head_dim%32==0");
        let cfg = LaunchConfig {
            grid_dim: ((rows * heads) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.quant_rows_fp8g32_bf16)
                .arg(x).arg(q).arg(sc)
                .arg(&(rows as i32)).arg(&(heads as i32)).arg(&(head_dim as i32)).arg(&mode)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **FP8-G32/int8-G32 残差窗 flush 量化**（KIVI 式，`LINGXI_KV_FP8_RESIDUAL`）：
    /// 把 bf16 残差平面 `[row0, row0+rows)` 行（源按残差布局寻址：sink 段线性 +
    /// 尾窗环 `sink + (r-sink) % rcap`）量化进压缩平面的**绝对行号**位置。
    /// 一次 launch 天然处理环回绕。`mode` 语义同 [`Self::quant_rows_fp8g32_bf16`]。
    pub fn quant_rows_fp8g32_res_bf16(
        &self, x: &CudaSlice<bf16>, q: &mut cudarc::driver::CudaViewMut<'_, u8>,
        sc: &mut cudarc::driver::CudaViewMut<'_, half::f16>,
        row0: usize, rows: usize, sink: usize, rcap: usize,
        heads: usize, head_dim: usize, mode: i32,
    ) -> Result<()> {
        assert!(head_dim % 32 == 0, "quant_rows_fp8g32_res 需要 head_dim%32==0");
        let cfg = LaunchConfig {
            grid_dim: ((rows * heads) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.quant_rows_fp8g32_res_bf16)
                .arg(x).arg(q).arg(sc)
                .arg(&(row0 as i32)).arg(&(rows as i32))
                .arg(&(sink as i32)).arg(&(rcap as i32))
                .arg(&(heads as i32)).arg(&(head_dim as i32)).arg(&mode)
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
        // ★ §二十九：hk=hv=128、单序列 ⇒ 走 **state 常驻寄存器版**（逐位相同，微基准 2.6×）。
        // 其余形状（并发批 n_seq>1、非 128 头）与 `LINGXI_GDN_SMEM=1` 的 A/B 档走旧核。
        // `LINGXI_NO_GDN_REC_REG=1` 回退（A/B 用）。
        let rec_reg = hk == 128 && hv == 128 && n_seq <= 1 && !use_smem && !no_rec_reg();
        let prefetch: i32 = if rec_reg && gdn_prefetch_on() { 1 } else { 0 };
        // §AL：流水版仅 reg 核承载（隐含 prefetch 语义）；旧核签名一字未动。
        let pipe: i32 = if rec_reg && gdn_pipe_on() { 1 } else { 0 };
        let cfg = LaunchConfig {
            grid_dim: (num_v_heads as u32 * n_seq.max(1) as u32, 1, 1),
            block_dim: (hv as u32, 1, 1),
            shared_mem_bytes: if use_smem { smem as u32 } else { 0 },
        };
        if rec_reg {
            unsafe {
                self.stream
                    .launch_builder(&self.gdn_recurrent_reg_bf16)
                    .arg(conv).arg(z).arg(a).arg(b).arg(norm_w)
                    .arg(a_log).arg(dt_bias).arg(s_all).arg(gated)
                    .arg(&(t as i32)).arg(&(hk as i32)).arg(&(hv as i32))
                    .arg(&(num_v_heads as i32)).arg(&(num_k_heads as i32)).arg(&gate_act)
                    .arg(&(row_off as i32)).arg(&(use_smem as i32)).arg(&(n_seq as i32))
                    .arg(&prefetch).arg(&pipe)
                    .launch(cfg)?;
            }
        } else {
            unsafe {
                self.stream
                    .launch_builder(&self.gdn_recurrent_bf16)
                    .arg(conv).arg(z).arg(a).arg(b).arg(norm_w)
                    .arg(a_log).arg(dt_bias).arg(s_all).arg(gated)
                    .arg(&(t as i32)).arg(&(hk as i32)).arg(&(hv as i32))
                    .arg(&(num_v_heads as i32)).arg(&(num_k_heads as i32)).arg(&gate_act)
                    .arg(&(row_off as i32)).arg(&(use_smem as i32)).arg(&(n_seq as i32))
                    .arg(&prefetch)
                    .launch(cfg)?;
            }
        }
        Ok(())
    }

    /// ㉟+1 GDN 递推**多 warp 占用率版**（`gdn_ops.cu`）：同 [`Self::gdn_recurrent_bf16`]
    /// 语义，但 dk 维拆到 4 组线程（block = hv×4 = 512 线程），每步三次跨 split 定序
    /// 归约走 shared。**有损档**（跨 split 归约次序 ≠ 旧核单线程升序；PLAN_NEXT §3
    /// 判据）。只支持 `n_seq == 1`；要求 hv ≤ 128（shared 归约缓冲）。
    pub fn gdn_recurrent_mw_bf16(
        &self, conv: &CudaSlice<bf16>, z: &CudaSlice<bf16>,
        a: &CudaSlice<bf16>, b: &CudaSlice<bf16>, norm_w: &CudaSlice<bf16>,
        a_log: &CudaSlice<f32>, dt_bias: &CudaSlice<f32>,
        s_all: &mut CudaSlice<f32>, gated: &mut CudaSlice<bf16>,
        t: usize, hk: usize, hv: usize, num_v_heads: usize, num_k_heads: usize,
        gate_act: i32,
    ) -> Result<()> {
        anyhow::ensure!(hv <= 128, "gdn_mw: 需要 hv ≤ 128（给到 {hv}）");
        let cfg = LaunchConfig {
            grid_dim: (num_v_heads as u32, 1, 1),
            block_dim: ((hv * 4) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.gdn_recurrent_mw_bf16)
                .arg(conv).arg(z).arg(a).arg(b).arg(norm_w)
                .arg(a_log).arg(dt_bias).arg(s_all).arg(gated)
                .arg(&(t as i32)).arg(&(hk as i32)).arg(&(hv as i32))
                .arg(&(num_v_heads as i32)).arg(&(num_k_heads as i32)).arg(&gate_act)
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
        self.gemv_q4g64_slots_tc_bf16_gidx(
            slots, slot_ids, group_off, x, y, s, n, k, a_stride, k_pad, blob, seg_off, scale_off,
            tiles, 0,
        )
    }

    /// 同上，但带**组压缩下标表** `gidx`（`0` = 全 `u` 网格）：`blockIdx.y` 走 `gidx[e]`
    /// 间接取 `group_off[gi]`/`group_off[gi+1]`/`slot_ids[gi]`，被跳过的只是对侧格式的空组。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slots_tc_bf16_gidx(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        group_off: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, s: usize, n: usize, k: usize,
        a_stride: usize, k_pad: i32, blob: u64, seg_off: u64, scale_off: u64,
        tiles: usize, gidx: u64,
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
                .arg(&gidx)
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

    /// ㊽+5 Q4G64 版大 m 摊销投影核（`q4_mlarge.cu`）：mlarge tiling + 核内
    /// 反量化（k-tile 64 = 一个 scale 组）。要求 `k % 64 == 0`。
    /// `LINGXI_NO_Q4MLARGE=1` 回退 gemm_q4_skinny。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_mlarge_ptr(
        &self, wq: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        row_off: usize, scale_off: u64, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!(k % 64 == 0 && m >= 1, "q4_mlarge: 需要 k%64==0、m≥1（k={k} m={m}）");
        let cfg = LaunchConfig {
            grid_dim: (((n + 7) / 8) as u32, ((m + 63) / 64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.q4_mlarge)
                .arg(&wq).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&(scale_off as i64))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ㊽+6 Q4 稠密投影 TCP MMA + 大 m 摊销核（`q4_tcp_dense.cu`）：
    /// `y[m,n] = x[m,k] · dequant(Q4G64 W[n,k])ᵀ`，CTA = 128 输出行 × 64 token 行，
    /// x 每 k-group 进 shared 一遍、权重被 64 token 行摊销，dequant 的 B 片段直接喂
    /// Volta MMA（ALU 链被摊销 —— q4_mlarge 缺的那一环）。要求 `k % 64 == 0`
    /// （调用方以 `kdim` 传 k、`wq.k_pad` 传 k_pad）。
    /// 数值口径同 q4_tcp（有损档，判据 PLAN_NEXT §3）。`LINGXI_NO_Q4TCPDENSE=1` 回退。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_tcp_dense_ptr(
        &self, wq: u64, x: u64, y: u64, n: usize, k: usize, k_pad: usize, m: usize,
        row_off: usize, scale_off: u64, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            k % 64 == 0 && k_pad % 64 == 0 && m >= 1,
            "q4_tcp_dense: 需要 k%64==0、k_pad%64==0、m≥1（k={k} k_pad={k_pad} m={m}）"
        );
        let cfg = LaunchConfig {
            grid_dim: (((n + 127) / 128) as u32, ((m + 63) / 64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.q4_tcp_dense)
                .arg(&wq).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(k_pad as i32)).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&(scale_off as i64))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// M≤8 变体（`q4_tcp_dense.cu` 的 `lingxi_gemm_q4_tcp_dense_m8`）：口径同上，
    /// 但 M 轴 tile = 8（mma 最小高度），CTA = 128 输出行 × 8 token 行、block = 128。
    /// 用于 verify 批（t=6）：M64 tile 在这个 m 下白算 8 倍（实测比 skinny 慢 2.4×）。
    /// 要求 `k % 64 == 0`。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_tcp_dense_m8_ptr(
        &self, wq: u64, x: u64, y: u64, n: usize, k: usize, k_pad: usize, m: usize,
        row_off: usize, scale_off: u64, epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            k % 64 == 0 && k_pad % 64 == 0 && m >= 1 && m <= 8,
            "q4_tcp_dense_m8: 需要 k%64==0、k_pad%64==0、1≤m≤8（k={k} k_pad={k_pad} m={m}）"
        );
        let cfg = LaunchConfig {
            grid_dim: (((n + 127) / 128) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.q4_tcp_dense_m8)
                .arg(&wq).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(k_pad as i32)).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&(scale_off as i64))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ㊽+4 稠密投影大 m 摊销核（`mlarge.cu`）：`y[m,n] = x[m,k] · W[n,k]ᵀ`，
    /// CTA = 8 输出行 × 64 token 行、k-tile 64 进 smem。要求 `k % 64 == 0`。
    /// 每元素 k 升序串行累加（有损档 vs row 核，判据 PLAN_NEXT §3）。`LINGXI_NO_MLARGE=1` 回退。
    pub fn gemm_bf16_mlarge_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!(k % 64 == 0 && m >= 1, "mlarge: 需要 k%64==0、m≥1（k={k} m={m}）");
        let cfg = LaunchConfig {
            grid_dim: (((n + 7) / 8) as u32, ((m + 63) / 64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.mlarge)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// §AM 稠密 bf16 投影 Volta MMA 核（`bf16_tcp.cu`）：`y[m,n] = x[m,k] · W[n,k]ᵀ`，
    /// q4_tcp_dense 骨架去反量化——CTA = 128 输出行 × 64 token 行，mma.m8n8k4，
    /// x 舞台 bf16→fp16 一次 CVT、W lane 内 8 元素 CVT 直接喂 B 片段。要求 `k % 64 == 0`。
    /// **有损档**（fp16 乘 + fp32 累加 vs SIMT f32 FMA，判据 PLAN_NEXT §3），
    /// `LINGXI_PROJ_TC=1` 才由 gpu.rs 分流（默认关）。epi 语义同 skinny 族。
    pub fn gemm_bf16_tcp_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        anyhow::ensure!(k % 64 == 0 && m >= 1, "bf16_tcp: 需要 k%64==0、m≥1（k={k} m={m}）");
        let cfg = LaunchConfig {
            grid_dim: (((n + 127) / 128) as u32, ((m + 63) / 64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.bf16_tcp)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ㊽+3 Q3 版 prefill 大 tile 张量核（`q3_tcp.cu`）：语义与参数和
    /// [`Self::gemv_q4g64_slots_tcp_bf16_view`] 完全一致，仅权重布局/反量化不同
    ///（k_pad 必须是 128 的倍数；每行每组 48 code B）。`LINGXI_MOE_TC=1` 门控。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slots_tcp_bf16_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        group_off: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, e_uniq: usize, n: usize, k: usize,
        a_stride: usize, k_pad: i32, blob: u64, seg_off: u64, scale_off: u64, tmax: usize,
    ) -> Result<()> {
        assert!(k_pad as usize % 128 == 0, "q3_tcp: k_pad 必须是 128 的倍数");
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
                .launch_builder(&self.q3_tcp)
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

    /// §AN 乐观发射档（Q4）：与 [`Self::gemv_q4g64_slots_tcp_bf16_view`] 同参同算术，
    /// 但 `e_uniq` 是 **host 上界**（真实组数由 `group_off` 设备侧早退守卫），
    /// 行块按组大小核内循环（无 `tmax` 形参）⇒ 可在 pair 建表核完成前入队。
    /// 调用方保证 `group_off` 尾部已清零（pair 建表核负责，§AN）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slots_tcp_bf16_dyn_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        group_off: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, e_uniq_ub: usize, n: usize, k: usize,
        a_stride: usize, k_pad: i32, blob: u64, seg_off: u64, scale_off: u64,
    ) -> Result<()> {
        let x_v = x.slice(0..x.len());
        let mut y_v = y.slice_mut(0..y.len());
        let blocks = ((n + 127) / 128).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq_ub as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.q4_tcp_dyn)
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

    /// §AN 乐观发射档（Q3）：语义同 [`Self::gemv_q4g64_slots_tcp_bf16_dyn_view`]。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slots_tcp_bf16_dyn_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        group_off: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, e_uniq_ub: usize, n: usize, k: usize,
        a_stride: usize, k_pad: i32, blob: u64, seg_off: u64, scale_off: u64,
    ) -> Result<()> {
        assert!(k_pad as usize % 128 == 0, "q3_tcp_dyn: k_pad 必须是 128 的倍数");
        let x_v = x.slice(0..x.len());
        let mut y_v = y.slice_mut(0..y.len());
        let blocks = ((n + 127) / 128).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq_ub as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.q3_tcp_dyn)
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
        self.gemv_q4g64_slotsm_rpw_bf16_view_gidx(
            slots, slot_ids, x, group_off, y, e_uniq, n, k, k_pad, a_stride, blob, seg_off,
            scale_off, rows_per_warp, xf, 0,
        )
    }

    /// 同上 + **组压缩下标表** `gidx`（`0` = 全 `u` 网格，逐位与旧路径相同）。
    /// `e_uniq` 此时是"本格式组数"，核内按 `gidx[e]` 间接取全表的边界与槽号。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slotsm_rpw_bf16_view_gidx(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, group_off: &cudarc::driver::CudaView<'_, i32>,
        y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize, xf: u64, gidx: u64,
    ) -> Result<()> {
        let (fun, rpw) = match rows_per_warp {
            2 => (&self.gemv_q4g64_slotsm_r2_bf16, 2usize),
            4 => (&self.gemv_q4g64_slotsm_r4_bf16, 4usize),
            _ => (&self.gemv_q4g64_slotsm_bf16, 1usize),
        };
        let warps_per_block = (BLOCK / 32) as usize;
        // ★ smem-x（`LINGXI_GEMV_SMEM_X=1`）：激活按 block 搬进 shared（见 dequant_ops.cu 头注）。
        // 默认关；`m=1`（s_pairs == e_uniq）时 cap=0 ⇒ 走原核，AR 路径不受影响。
        // ⚠️ 该核没有 `gidx` 入参（实验档，默认关）⇒ 带 gidx 时一律走下面原核，保证逐位正确。
        let cap = smem_x_cap(y.len(), n, e_uniq);
        let kp_elems = (k + 7) & !7;
        if gemv_smem_x() == 1
            && gidx == 0
            && rows_per_warp == 1
            && cap > 0
            && kp_elems * cap * 2 <= 48 * 1024
        {
            let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
            let cfg = LaunchConfig {
                grid_dim: (blocks as u32, e_uniq as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: (kp_elems * cap * 2) as u32,
            };
            unsafe {
                self.stream.launch_builder(&self.gemv_q4g64_slotsm_sxb_bf16)
                    .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                    .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                    .arg(&(a_stride as i32))
                    .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                    .arg(&(cap as i32))
                    .launch(cfg)?;
            }
            return Ok(());
        }
        let rows_per_block = warps_per_block * rpw;
        let blocks = ((n + rows_per_block - 1) / rows_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // ★ `LINGXI_MOE_GEMV_TIER=1`：RPW=1 的 bf16 档换 `slotsm_t2_bf16`（**逐位相同**）。
        // 严格限定 `rpw==1 && xf==0`：t2 核没有 xf 快路（fp32 预转），而线上 rows=1 的混档
        // decode 恰好 `xf_on=false`；其余形状（prefill 的 xf 档、rpw=2/4）原样走老核。
        if gemv_tier() >= 1 && rpw == 1 && xf == 0 {
            {
                // 一次性诊断：确认开关真的走到了这个核（不刷屏）。
                static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                if ONCE.set(()).is_ok() {
                    eprintln!(
                        "[gemv-tier] slotsm_t2_bf16 生效：首次 n={n} k={k} k_pad={k_pad} \
                         a_stride={a_stride} e={e_uniq}"
                    );
                }
            }
            unsafe {
                self.stream.launch_builder(&self.gemv_q4g64_slotsm_t2_bf16)
                    .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                    .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                    .arg(&(a_stride as i32))
                    .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                    .arg(&gidx)
                    .launch(cfg)?;
            }
            return Ok(());
        }
        unsafe {
            self.stream.launch_builder(fun)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(&xf).arg(&gidx)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **实验档**（`gemv_par_exp.cu`，只有 microbench 用）：Q4G64 专家 GEMV 的 k 方向
    /// `splits` 段切分。每段算完写 fp32 部分和到 `part[(对,row)*splits + ks]`，再调
    /// [`Self::gemv_splitk_reduce_bf16`] 汇总成 bf16。**与 `slotsm` 不逐位相同**（求和次序变）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slotsm_splitk_bf16(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaView<'_, i32>, x: &CudaSlice<bf16>,
        group_off: &CudaView<'_, i32>, part: &mut CudaSlice<f32>, y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, splits: usize, xf: u64,
    ) -> Result<()> {
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, splits as u32),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_q4g64_slotsm_splitk_bf16)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(&mut *part)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(&(splits as i32)).arg(&xf)
                .launch(cfg)?;
        }
        let cnt = (y.len()) as i64;
        let rcfg = LaunchConfig {
            grid_dim: (((cnt as usize + 255) / 256).min(4096) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_splitk_reduce_bf16)
                .arg(part).arg(y).arg(&cnt).arg(&(splits as i32))
                .launch(rcfg)?;
        }
        Ok(())
    }

    /// **实验档**：按 `slotsm` 的访存形状只读不算（`mode` 0 = 4 B/lane，1 = 16 B/lane）。
    pub fn read_probe_slots(
        &self, slots: &CudaSlice<u8>, out: &mut CudaSlice<f32>,
        e: usize, rows_per_expert: usize, row_bytes: usize, blob: u64, mode: i32,
    ) -> Result<()> {
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((rows_per_expert + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.read_probe_slots)
                .arg(slots).arg(out)
                .arg(&(rows_per_expert as i32)).arg(&(row_bytes as i32))
                .arg(&(blob as i64)).arg(&mode)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **实验档**：经典 grid-stride `uint4` 纯读（机器读带宽上限）。
    pub fn read_probe_flat(&self, buf: &CudaSlice<u8>, out: &mut CudaSlice<f32>) -> Result<()> {
        let n16 = (buf.len() / 16) as i64;
        let blocks = 1024usize;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.read_probe_flat)
                .arg(buf).arg(out).arg(&n16)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **乘加结构实验档**（`gemv_struct_exp.cu`）。`which`：
    /// 0..=7 = 阶梯探针 P0..P7（同访存形状）/ 8 = P8（scale smem）/ 9 = P9（scale+x smem）/
    /// 10 = u128_r1 / 11 = u128_r2 / 12 = u128_r4 / 13 = contig / 14 = u128_r2_sm。
    ///
    /// ⚠️ 除 10（`u128_r1`）外仍是微基准专用；**10 已进生产**：草稿 head 的
    /// `LINGXI_DRAFT_HEAD_FAST=1`（`SpecRuntime::head`，t=1，非逐位——草稿不影响生成文本）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_struct_exp(
        &self, which: usize, slots: &CudaSlice<u8>, slot_ids: &CudaView<'_, i32>,
        x: &CudaSlice<bf16>, group_off: &CudaView<'_, i32>, y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64,
    ) -> Result<()> {
        let (fun, rows_per_block) = match which {
            0..=7 => (&self.struct_exp.probe[which], 8usize),
            8 => (&self.struct_exp.p8_smsc, 8usize),
            9 => (&self.struct_exp.p9_smscx, 8usize),
            10 => (&self.struct_exp.u128_r1, 8usize),
            11 => (&self.struct_exp.u128_r2, 16usize),
            12 => (&self.struct_exp.u128_r4, 32usize),
            13 => (&self.struct_exp.contig, 8usize),
            14 => (&self.struct_exp.u128_r2_sm, 16usize),
            15 => (&self.struct_exp.tier_unr1, 8usize),
            16 => (&self.struct_exp.tier_unr2, 8usize),
            17 => (&self.struct_exp.tier_unr4, 8usize),
            18 => (&self.struct_exp.tier_unr4_proddec, 8usize),
            19 => (&self.struct_exp.tier_shift4, 8usize),
            _ => anyhow::bail!("gemv_q4g64_struct_exp: 未知变体 {which}"),
        };
        // P9 / u128_r2_sm 要 k 个 fp32 的激活缓冲（gu k=2560 → 10 KB，< 48 KB 静态上限）。
        let smem = if which == 9 || which == 14 { k * 4 } else { 0 };
        let blocks = ((n + rows_per_block - 1) / rows_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        unsafe {
            self.stream.launch_builder(fun)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 同上的 **dp4a 变体**（`u128_r1_dp4a`，int8 激活行量化；非位等价）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_struct_exp_dp4a(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaView<'_, i32>, x: &CudaSlice<bf16>,
        group_off: &CudaView<'_, i32>, y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64,
        xq: &CudaSlice<u8>, sx: &CudaSlice<f32>,
    ) -> Result<()> {
        let rows_per_block = 8usize;
        let blocks = ((n + rows_per_block - 1) / rows_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.struct_exp.u128_r1_dp4a)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(xq).arg(sx)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **Q3 取数结构实验档**（`gemv_q3_exp.cu`，只有 microbench 用）。`which`：
    /// 0 = unr1/三字节/生产解包（只去 PF=4）/ 1 = unr2/三字节/生产解包 /
    /// 2 = unr2/对齐 u32 对/生产解包 / 3 = unr2/三字节/移位对 / **4 = unr2/对齐 u32 对/移位对（t2 候选）** /
    /// 5 = unr1/对齐 u32 对/移位对 / 6 = unr4/对齐 u32 对/移位对 / 7 = unr8/对齐 u32 对/移位对 /
    /// 8..=11 = MB=1 特化（**只在每组恰好 1 对时可用**）：unr1/unr2/unr4 对齐 u32 对 + unr2 三字节。
    /// 全部与 [`Self::gemv_q3g128_slotsm_rpw_bf16_view`]（rpw=1、xf=0）**逐位相同**。
    /// ⚠️ 「一个 4 B 未对齐 load」在 V100 上直接 MISALIGNED_ADDRESS（实测）⇒ 只能用对齐 u32 对。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slotsm_exp(
        &self, which: usize, slots: &CudaSlice<u8>, slot_ids: &CudaView<'_, i32>,
        x: &CudaSlice<bf16>, group_off: &CudaView<'_, i32>, y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, gidx: u64,
    ) -> Result<()> {
        anyhow::ensure!(which < self.q3_exp.v.len(), "gemv_q3g128_slotsm_exp: 未知变体 {which}");
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.q3_exp.v[which])
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(&gidx)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **稠密 Q4 瘦长 GEMM 的同形取数探针**（`q4_dense_exp.cu`）：grid/block/地址映射与
    /// `lingxi_gemm_q4_skinny_bf16` 逐字节相同。`mode`：
    /// 0 = 权重纯读 4 B/lane（生产取数形状）/ 1 = 16 B/lane uint4（结构上限）/
    /// 2 = 4 B + nibble 解包 + I2F（ALU 地板）/ 3 = 生产内层全量 / 4 = staging + 纯读 /
    /// 5 = 解包 + I2F + 8 FFMA/token（x 用常量，隔离 smem x 读）/ 6 = 只做 staging /
    /// 7 = 生产内层但 x 直接读 global（无舞台、无 barrier）。
    /// 只有 mode 3/4/6 分配 x 舞台的 16 KB dynamic smem（其余是纯取数结构，别被占用压住）。
    #[allow(clippy::too_many_arguments)]
    pub fn q4_dense_probe(
        &self, wq: u64, x: u64, y: u64, n: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64, mode: i32,
    ) -> Result<()> {
        let blocks = ((n + 7) / 8).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, (m as u32 + 7) / 8, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: if mode == 3 || mode == 4 || mode == 6 { 8 * 1024 * 2 } else { 0 },
        };
        unsafe {
            self.stream.launch_builder(&self.q4_dense_exp)
                .arg(&wq).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&(scale_off as i64)).arg(&mode)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **多行/warp 变体**（`q4_dense_exp.cu`）：一个 warp 连算 R 个输出行，x 的 smem 读与
    /// bf16→f32 被 R 行共用、舞台流量与 barrier 次数 ÷R。`which`：
    /// 0=R1W8 / 1=R2W8 / 2=R4W8 / 3=R1W16 / 4=R2W16（W = 每 block 的 warp 数）。
    /// 与 `lingxi_gemm_q4_skinny_bf16` **逐位相同**（列集合/累加次序/归约树一字未改）。
    #[allow(clippy::too_many_arguments)]
    pub fn q4_dense_rows_probe(
        &self, which: usize, wq: u64, x: u64, y: u64, n: usize, k: usize, k_pad: i32, m: usize,
        row_off: usize, scale_off: u64,
    ) -> Result<()> {
        const VAR: [(usize, usize); 5] = [(1, 8), (2, 8), (4, 8), (1, 16), (2, 16)];
        let which = which.min(VAR.len() - 1);
        let (r, w) = VAR[which];
        let blocks = ((n + w * r - 1) / (w * r)).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, (m as u32 + 7) / 8, 1),
            block_dim: ((32 * w) as u32, 1, 1),
            shared_mem_bytes: 8 * 1024 * 2,
        };
        let f = &self.q4_dense_rows_probe[which];
        unsafe {
            self.stream.launch_builder(f)
                .arg(&wq).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad).arg(&(m as i32))
                .arg(&(row_off as i32)).arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **bf16 瘦长 GEMM 的取数/结构探针**（`bf16_skinny_exp.cu`，只有 microbench 用）。
    /// `which`：0=m8 纯读 2 B/lane、1=m8 纯读 16 B、2=m8 ALU 地板、3=m8 生产复刻、
    /// 4/5/6=m8 一 warp 连算 R=2/4/8 行、7=m8 R=4+PF、8=m8 R=4+x smem、
    /// 9=row 纯读 16 B、10=row ALU、11=row 生产复刻、12/13=row R=2 / R=2+PF。
    pub fn bf16_skinny_probe(
        &self, which: usize, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
    ) -> Result<()> {
        let e = &self.bf16_exp;
        // (核, 每 warp/block 的输出行数, 动态 smem 字节)
        let (f, rows, smem): (&CudaFunction, usize, u32) = match which {
            0 => (&e.m8_read1, 1, 0),
            1 => (&e.m8_read16, 1, 0),
            2 => (&e.m8_alu, 1, 0),
            3 => (&e.m8_prod, 1, 0),
            4 => (&e.m8_r[0], 2, 0),
            5 => (&e.m8_r[1], 4, 0),
            6 => (&e.m8_r[2], 8, 0),
            7 => (&e.m8_r4pf, 4, 0),
            8 => (&e.m8_r4x, 4, (8 * k * 4) as u32),
            9 => (&e.row_read16, 1, 0),
            10 => (&e.row_alu, 1, 0),
            11 => (&e.row_prod, 1, 0),
            12 => (&e.row_r2, 2, 0),
            13 => (&e.row_r2pf, 2, 32 * 2 * 8 * 4),
            14 => (&e.m8_xonly, 1, 0),
            15 => (&e.m8_r1pf, 1, 0),
            16 => (&e.m8_r1x, 1, (8 * k * 4) as u32),
            17 => (&e.m8_r2x, 2, (8 * k * 4) as u32),
            18 => (&e.row_xonly, 1, 0),
            19 => (&e.row_r4, 4, 0),
            20 => (&e.m8_prod_r0, 1, 0),
            21 => (&e.m8_m6, 1, 0),
            22 => (&e.m8_r0hoist, 1, 0),
            _ => anyhow::bail!("bf16_skinny_probe: which={which} 越界"),
        };
        let row_body = which >= 9;
        let blocks = if row_body {
            (n + rows - 1) / rows
        } else {
            let per_block = rows * (BLOCK as usize / 32);
            (n + per_block - 1) / per_block
        }
        .max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            self.stream
                .launch_builder(f)
                .arg(&w)
                .arg(&x)
                .arg(&y)
                .arg(&(n as i32))
                .arg(&(k as i32))
                .arg(&(m as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **草稿 lm_head（Q4G64 稠密 GEMV）的同形探针**（`bf16_skinny_exp.cu`）。
    /// `which`：0 = 纯读 4 B/lane + scale（生产取数形状）/ 1 = 纯读 16 B/lane（上限）/
    /// 2 = 生产内层逐字复刻（自检）。grid = `ceil(n/8)` × 256（一 warp 一行）。
    #[allow(clippy::too_many_arguments)]
    pub fn q4_head_probe(
        &self, which: usize, w: u64, x: u64, y: u64, n: usize, k: usize, k_pad: i32,
        row_off: usize, scale_off: u64,
    ) -> Result<()> {
        let e = &self.bf16_exp;
        let (f, rows): (&CudaFunction, usize) = match which {
            0 => (&e.q4h_read1, 1),
            1 => (&e.q4h_read16, 1),
            3 => (&e.q4h_pf, 1),
            4 => (&e.q4h_r2, 2),
            _ => (&e.q4h_prod, 1),
        };
        let blocks = ((n + 8 * rows - 1) / (8 * rows)).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(f)
                .arg(&w)
                .arg(&x)
                .arg(&y)
                .arg(&(n as i32))
                .arg(&(k as i32))
                .arg(&k_pad)
                .arg(&(row_off as i32))
                .arg(&(scale_off as i64))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **Q3 同形纯读探针**（`gemv_q3_exp.cu`）：`mode` 0 = 3×1 B / 1 = 4 B 未对齐 /
    /// 2 = 16 B uint4 / 3 = 4 B + 移位对解包 + I2F。地址映射与生产核逐字节相同。
    #[allow(clippy::too_many_arguments)]
    pub fn q3_read_probe(
        &self, slots: &CudaSlice<u8>, out: &mut CudaSlice<f32>,
        e: usize, rows_per_expert: usize, row_bytes: usize, k: usize, blob: u64, mode: i32,
    ) -> Result<()> {
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((rows_per_expert + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.q3_exp.read_probe)
                .arg(slots).arg(out)
                .arg(&(rows_per_expert as i32)).arg(&(row_bytes as i32)).arg(&(k as i32))
                .arg(&(blob as i64)).arg(&mode)
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
        self.gemv_q3g128_slotsm_rpw_bf16_view_gidx(
            slots, slot_ids, x, group_off, y, e_uniq, n, k, k_pad, a_stride, blob, seg_off,
            scale_off, rows_per_warp, xf, 0,
        )
    }

    /// 同上 + **组压缩下标表** `gidx`（`0` = 全 `u` 网格）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slotsm_rpw_bf16_view_gidx(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, group_off: &cudarc::driver::CudaView<'_, i32>,
        y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize, xf: u64, gidx: u64,
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
                .arg(&xf).arg(&gidx)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Q3 的**取数结构档**（默认启用；`LINGXI_NO_Q3_T2=1` 回退旧核）。
    ///
    /// 与 [`Self::gemv_q3g128_slotsm_rpw_bf16_view_gidx`] **逐位相同**（t13 对拍 0/N 元素不同），
    /// 只把"3×1 B 取数 + 生产解包 + PF=4 移位预取"换成"两对齐 u32 + `__funnelshift_r`
    /// + 移位对解包 + UNR 个独立 load"（见 `dequant_ops.cu` 该核头注）。
    ///
    /// 门控与 Q4 的 `slotsm_t2_bf16` 同款：只吃 `rows_per_warp == 1 && xf == 0`
    /// （线上 rows=1 的混档 decode 正是这一支；t2 核没有 xf(fp32 预转) 快路，也没有 rpw>1 实例）。
    /// `all_m1` = **本次 launch 的每组恰好 1 对**（调用方按 `tmax == 1` 得出）：届时用 `MB=1`
    /// 实例（省掉 MB=8 骨架里 7 次 `if (s >= mb) break` 与 7 个空累加器，实测 +18%）。
    /// `k >= 1024`（gu 的 2560）用 UNR=2，短 k（dn 的 640）用 UNR=1（t13 实测各自更快）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slotsm_t2_view_gidx(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaView<'_, i32>, x: &CudaSlice<bf16>,
        group_off: &CudaView<'_, i32>, y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize, xf: u64, gidx: u64,
        all_m1: bool,
    ) -> Result<()> {
        if q3_t2_off() || rows_per_warp != 1 || xf != 0 {
            return self.gemv_q3g128_slotsm_rpw_bf16_view_gidx(
                slots, slot_ids, x, group_off, y, e_uniq, n, k, k_pad, a_stride, blob, seg_off,
                scale_off, rows_per_warp, xf, gidx,
            );
        }
        let fun = if !all_m1 {
            &self.gemv_q3g128_slotsm_t2_bf16
        } else if k >= 1024 {
            &self.gemv_q3g128_slotsm_t2_mb1u2_bf16
        } else {
            &self.gemv_q3g128_slotsm_t2_mb1u1_bf16
        };
        {
            // 一次性诊断：确认开关真的走到了新核（不刷屏）。
            static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            if ONCE.set(()).is_ok() {
                eprintln!(
                    "[q3-t2] slotsm_t2 生效：首次 n={n} k={k} k_pad={k_pad} a_stride={a_stride} \
                     e={e_uniq} all_m1={all_m1}"
                );
            }
        }
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
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
                .arg(&gidx)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **Q4 t2 档 + 设备侧组数上界**（A' 的地基，见 docs §三十七 C' / C''）。
    ///
    /// 与 `gemv_q4g64_slotsm_rpw_bf16_view_gidx`（tier 档、rpw=1、xf=0）**同一个核体、逐位相同**；
    /// 区别只有：`grid.y = e_max`（上界）而不是精确的 `e_uniq`，入口按 `*ng_dev` 早退。
    /// 主机因此可以先发牌、后核对（不必等 `npri/nalt` 写出来）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slotsm_t2_ng_view_gidx(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaView<'_, i32>, x: &CudaSlice<bf16>,
        group_off: &CudaView<'_, i32>, y: &mut CudaSlice<bf16>,
        e_max: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, gidx: u64, ng_dev: u64,
    ) -> Result<()> {
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_max as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_q4g64_slotsm_t2_ng_bf16)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_max as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(&gidx).arg(&ng_dev)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **Q3 t2 档 + 设备侧组数上界**（A' 的地基，见 docs §三十七 C'）。
    ///
    /// 与 [`Self::gemv_q3g128_slotsm_t2_view_gidx`] 同一个核体、**逐位相同**；区别只有两点：
    /// * `grid.y = e_max`（调用方给上界，如 `s = rows×top_k`），而不是精确的 `ng`；
    /// * 入口按 `*ng_dev` 早退 ⇒ 多出来的空 block 只做一次 4 B 设备读就退出。
    ///
    /// 为什么需要它：精确 `ng` 只能等设备把 `npri/nalt` 写出来（每层 345~566 µs 的 `route_spin`）；
    /// 有了这个档，主机可以**先发牌、后核对**，主机才可能跑到另一张卡上去（分块流水的前提）。
    /// `ng_dev == 0` 时退化成"不设上界"（等价于 `e_max` 全部有效），仅供对拍。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slotsm_t2_ng_view_gidx(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaView<'_, i32>, x: &CudaSlice<bf16>,
        group_off: &CudaView<'_, i32>, y: &mut CudaSlice<bf16>,
        e_max: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, gidx: u64, ng_dev: u64,
    ) -> Result<()> {
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_max as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_q3g128_slotsm_t2_ng_bf16)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_max as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(&gidx).arg(&ng_dev)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **Q3G128 + int8 激活 + `__dp4a`**（2026-10-25，**非位等价**）。
    ///
    /// 与 [`Self::gemv_q3g128_slotsm_t2_view_gidx`] 同形（同样一 warp 一行、`MB=8` 分批、
    /// `group_off` 分组），只把内层"每 token 8×CVT + 8×FFMA"换成"每 token 2×`__dp4a`"。
    /// 量纲按 group 分段（128 权重一个 fp16 scale），x 走 `xq`(int8)+`sx`(每行 scale)。
    ///
    /// 前置条件（不满足直接 bail，调用方回退 t2）：`k_pad == k && k % 8 == 0`；
    /// `k` 为 128 的倍数（本题 gu 2560 / dn 640 都满足）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3g128_slotsm_dp4a_view(
        &self, slots: &CudaSlice<u8>, slot_ids: &CudaView<'_, i32>, x: &CudaSlice<bf16>,
        group_off: &CudaView<'_, i32>, y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, gidx: u64,
        xq: &CudaSlice<u8>, sx: &CudaSlice<f32>,
    ) -> Result<()> {
        anyhow::ensure!(
            k_pad as usize == k && k % 128 == 0,
            "gemv_q3g128_slotsm_dp4a: 只支持 k_pad==k 且 k%128==0（k={k} k_pad={k_pad}）"
        );
        let warps_per_block = (BLOCK / 32) as usize;
        let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, e_uniq as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemv_q3g128_slotsm_dp4a)
                .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                .arg(&(a_stride as i32))
                .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                .arg(&gidx)
                .arg(xq).arg(sx)
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
        self.gemv_q4g64_slotsm_rpw_dp4a_view_gidx(
            slots, slot_ids, x, group_off, y, e_uniq, n, k, k_pad, a_stride, blob, seg_off,
            scale_off, rows_per_warp, xq, sx, 0,
        )
    }

    /// 同上 + **组压缩下标表** `gidx`（`0` = 全 `u` 网格）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4g64_slotsm_rpw_dp4a_view_gidx(
        &self, slots: &CudaSlice<u8>, slot_ids: &cudarc::driver::CudaView<'_, i32>,
        x: &CudaSlice<bf16>, group_off: &cudarc::driver::CudaView<'_, i32>,
        y: &mut CudaSlice<bf16>,
        e_uniq: usize, n: usize, k: usize, k_pad: i32, a_stride: usize,
        blob: u64, seg_off: u64, scale_off: u64, rows_per_warp: usize,
        xq: &CudaSlice<u8>, sx: &CudaSlice<f32>, gidx: u64,
    ) -> Result<()> {
        let fun = match rows_per_warp {
            2 => &self.gemv_q4g64_slotsm_r2_dp4a,
            _ => &self.gemv_q4g64_slotsm_dp4a,
        };
        let rpw = if rows_per_warp == 2 { 2usize } else { 1usize };
        let warps_per_block = (BLOCK / 32) as usize;
        // ★ smem-x（`LINGXI_GEMV_SMEM_X=2`，配合 `LINGXI_MOE_DP4A=1`）：int8 激活按 block
        // 搬进 shared。默认关；m=1 时 cap=0 ⇒ 原核。
        // ⚠️ 该核没有 `gidx` 入参（实验档，默认关）⇒ 带 gidx 时一律走下面原核，保证逐位正确。
        let cap = smem_x_cap(y.len(), n, e_uniq);
        let kp_elems = (k + 7) & !7;
        if gemv_smem_x() == 2 && gidx == 0 && rpw == 1 && cap > 0 && kp_elems * cap <= 48 * 1024 {
            let blocks = ((n + warps_per_block - 1) / warps_per_block).max(1);
            let cfg = LaunchConfig {
                grid_dim: (blocks as u32, e_uniq as u32, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: (kp_elems * cap) as u32,
            };
            unsafe {
                self.stream.launch_builder(&self.gemv_q4g64_slotsm_sxq_dp4a)
                    .arg(slots).arg(slot_ids).arg(x).arg(group_off).arg(y)
                    .arg(&(e_uniq as i32)).arg(&(n as i32)).arg(&(k as i32)).arg(&k_pad)
                    .arg(&(a_stride as i32))
                    .arg(&(blob as i64)).arg(&(seg_off as i64)).arg(&(scale_off as i64))
                    .arg(xq).arg(sx).arg(&(cap as i32))
                    .launch(cfg)?;
            }
            return Ok(());
        }
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
                .arg(xq).arg(sx).arg(&gidx)
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
        // `LINGXI_IDX_SCORES_VEC=1`：8 宽向量读版（**实测负结果**：长上下文 −23%、
        // 短上下文 +71%，默认关；见 `t6_attn_shape_bw`）
        static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fast = *FAST.get_or_init(|| std::env::var_os("LINGXI_IDX_SCORES_VEC").is_some()) as i32;
        let cfg = LaunchConfig { grid_dim: (Self::grid(nblocks), t_q as u32, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.qsa_indexer_scores_bf16)
                .arg(q).arg(pooled).arg(scores)
                .arg(&(t_q as i32)).arg(&(nblocks as i32)).arg(&(heads as i32)).arg(&(hd as i32))
                .arg(&sqrt_hd).arg(&fast)
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

    /// ★ §三十：**池化 + RMSNorm + RoPE + 落盘 四发合一**（与四发逐位相同）。
    /// grid = `nb`（一个 block 处理一个新块），block=256，动态 smem = 256×4。
    /// 输出直接落 `pooled[b0*hd ..]`（不再需要 pool_tmp/pool_tmp2）。
    pub fn qsa_pool_norm_rope_bf16(
        &self, rk: &CudaSlice<bf16>, pooled: &mut CudaSlice<bf16>,
        gamma: &CudaSlice<bf16>, cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
        b0: usize, nb: usize, compress: usize, hd: usize, rot: usize, eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(hd <= 256 && rot <= hd && rot % 2 == 0, "pool_norm_rope: hd≤256、rot≤hd");
        let cfg = LaunchConfig {
            grid_dim: (nb as u32, 1, 1), block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * 4,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_pool_norm_rope_bf16)
                .arg(rk).arg(pooled).arg(gamma).arg(cos).arg(sin)
                .arg(&(b0 as i32)).arg(&(compress as i32)).arg(&(hd as i32)).arg(&(rot as i32))
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `nblocks ≤ topk` ⇒ keep 全 1（§三十：与 `keep_select`/`keep_from_scores` 的早退同一条件）。
    pub fn qsa_keep_all_bf16(
        &self, keep: &mut CudaSlice<u8>, t_q: usize, nblocks: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(nblocks), t_q as u32, 1), block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_keep_all_bf16)
                .arg(keep).arg(&(nblocks as i32)).arg(&(t_q as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// 升序块号列表构建：`keep=1` 的块压进 `sel[row*stride + j]`，个数写 `sel_cnt[row]`。
    /// `all_kept=1`（调用方保证 `nblocks ≤ stride`）：keep 恒全 1 ⇒ 列表恒为 `[0..nblocks)`，
    /// 一次写 keep + sel + sel_cnt（省掉打分 + 名次两发；结果与正常路径逐位相同）。
    pub fn qsa_sel_build(
        &self, keep: &mut CudaSlice<u8>, sel: &mut CudaSlice<i32>, sel_cnt: &mut CudaSlice<i32>,
        t_q: usize, nblocks: usize, stride: usize, all_kept: bool,
    ) -> Result<()> {
        // `LINGXI_NO_SELBUILD_FAST=1`：回退到 tid==0 的 1024 次串行前缀和（旧实现，做 A/B）
        static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fast = *FAST.get_or_init(|| std::env::var_os("LINGXI_NO_SELBUILD_FAST").is_none()) as i32;
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.qsa_sel_build)
                .arg(keep).arg(sel).arg(sel_cnt)
                .arg(&(nblocks as i32)).arg(&(stride as i32)).arg(&fast)
                .arg(&(all_kept as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **gather 稀疏注意力**：只遍历每行的选中块列表（+尾部不完整块），与上下文长度解耦。
    /// 数学 = dense（未选中位贡献 exp=0 精确不参与），仅归约配对顺序不同（~1ulp）。
    /// `nh`：一个 block 合并处理同一 kv-group 的 q-head 数（1 / 2 / 4；4 ⇒ K/V 流量 ÷4）。
    /// 第二遍默认走"按选块分组取 V"（§三十，`LINGXI_NO_ATTN_V2=1` 回退到逐 token 老档；
    /// 两档逐位相同，`t17_attn_bits` 覆盖 nblocks=7/29/130）。
    #[allow(clippy::too_many_arguments)]
    pub fn attn_gather_bf16(
        &self, q: &CudaView<bf16>, k: &CudaView<bf16>, v: &CudaView<bf16>,
        blk_tab: &CudaView<i32>,
        sel: &CudaView<i32>, sel_cnt: &CudaView<i32>, out: &mut CudaViewMut<bf16>,
        t_q: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        vis_base: usize, nblocks: usize, cshift: usize, stride: usize, nh: usize,
    ) -> Result<()> {
        let ctok = 1usize << cshift;
        let v2 = std::env::var("LINGXI_NO_ATTN_V2").map(|v| v != "1").unwrap_or(true);
        let span = stride * ctok + ctok;
        // §AK v3：prefill 形态（t_q≥16，块数足够摊薄；decode/verify 的小 t 走旧核
        // NH=1 栅格更优）且 smem 放得下时，用重排核（逐位等价，见 attn_gather_v3.cuh）。
        // `LINGXI_NO_ATTN_GATHER_V3=1` 回退旧核路径。
        let v3 = v2
            && t_q >= 16
            && nh >= 2
            && !std::env::var("LINGXI_NO_ATTN_GATHER_V3").map(|v| v == "1").unwrap_or(false)
            && span * 4 * nh <= 47 * 1024;
        let (fun, nhead) = match (nh, v2, v3) {
            (2, true, true) => (&self.attn_gather_v3_2_bf16, 2usize),
            (4, true, true) => (&self.attn_gather_v3_4_bf16, 4usize),
            (2, true, false) => (&self.attn_gather2_bf16, 2usize),
            (4, true, false) => (&self.attn_gather4_bf16, 4usize),
            (4, false, _) => (&self.attn_gather4_v1_bf16, 4usize),
            (_, false, _) => (&self.attn_gather_v1_bf16, 1usize),
            _ => (&self.attn_gather_bf16, 1usize),
        };
        let smem = span * 4 * nhead;
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

    /// ★ §三十：**一次发射**完成两块 bf16 瘦长投影（同一份 x，n1 行 + n2 行）。
    /// 每块的算式/归约/写出与 `gemm_skinny_row_bf16`（grid.y=1、epi=0）逐字相同 ⇒ 逐位相同。
    /// 只要 `m ≤ 8`（grid.y 恒为 1）。
    pub fn gemm_bf16_row_pair_ptr(
        &self, w1: u64, y1: u64, n1: usize, w2: u64, y2: u64, n2: usize,
        x: u64, k: usize, m: usize,
    ) -> Result<()> {
        anyhow::ensure!((1..=8).contains(&m), "row_pair 只支持 1≤m≤8（给到 {m}）");
        let cfg = LaunchConfig {
            grid_dim: ((n1 + n2) as u32, 1, 1), block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_bf16_row_pair)
                .arg(&w1).arg(&y1).arg(&(n1 as i32))
                .arg(&w2).arg(&y2).arg(&(n2 as i32))
                .arg(&x).arg(&(k as i32)).arg(&(m as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ★ §三十：`grouped_rmsnorm_bf16` + `rope_partial_inplace_bf16` **合发**。
    /// grid = `rows*heads`（一格一个 (row, head)），block=256，动态 smem = 256×4。
    /// 归约树/舍入/RoPE 取值都与分开两发逐字相同 ⇒ 逐位相同。
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_rope_bf16(
        &self, x: &CudaSlice<bf16>, gamma: &CudaSlice<bf16>,
        cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>,
        rows: usize, heads: usize, hd: usize, rot: usize, pos_offset: usize, eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(hd <= 512, "rmsnorm_rope: head_dim ≤ 512（给到 {hd}）");
        let cfg = LaunchConfig {
            grid_dim: ((rows * heads) as u32, 1, 1), block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * 4,
        };
        unsafe {
            self.stream.launch_builder(&self.rmsnorm_rope_bf16)
                .arg(x).arg(gamma).arg(cos).arg(sin).arg(y)
                .arg(&(heads as i32)).arg(&(hd as i32)).arg(&(rot as i32))
                .arg(&(pos_offset as i32)).arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **QSA 三段的微基准探针/候选**（`qsa_probe_exp.cu`，只有 microbench 用）。
    /// `which`：0 prod 复刻 / 1 纯读 K+V / 2 只读 K / 3 只读 V / 4 q 常量 / 5 K/V 常量 /
    /// 6 只做第一遍 / 7 只做第二遍 / 8 v2（按选块重排第二遍）/ 9 v2pf（v2+4 值预取）/
    /// 10 空核（发射地板）/ 11 prod NH=2 / 12 prod NH=1 / 13 v2pf NH=2 / 14 v2pf NH=1 /
    /// 15 纯读 K+V（v2 地址流）/ 16 空核（NH=1 的 144 block 栅格）/ 17 纯读 K+V v2（NH=1）。
    /// grid/smem 与生产核同源；NH 由 `which` 决定。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_probe(
        &self, which: usize, q: &CudaView<bf16>, k: &CudaView<bf16>, v: &CudaView<bf16>,
        blk_tab: &CudaView<i32>,
        sel: &CudaView<i32>, sel_cnt: &CudaView<i32>, out: &mut CudaViewMut<bf16>,
        t_q: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        vis_base: usize, nblocks: usize, cshift: usize, stride: usize,
    ) -> Result<()> {
        let (fi, nhead) = match which {
            16 => (10usize, 1usize),
            17 => (15usize, 1usize),
            18 => (18usize, 1usize), // §AK v3 NH=1
            19 => (19, 2),           // §AK v3 NH=2
            20 => (20, 4),           // §AK v3 NH=4
            11 | 13 => (which, 2usize),
            12 | 14 => (which, 1usize),
            _ => (which, 4usize),
        };
        let f = self
            .qsa_probe
            .attn
            .get(fi)
            .ok_or_else(|| anyhow::anyhow!("qsa_attn_probe: which={which} 越界"))?;
        let ctok = 1usize << cshift;
        let smem = (stride * ctok + ctok) * 4 * nhead;
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, (heads / nhead) as u32, 1), block_dim: (256, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        unsafe {
            self.stream.launch_builder(f)
                .arg(q).arg(k).arg(v).arg(blk_tab).arg(sel).arg(sel_cnt).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(head_dim as i32))
                .arg(&scale).arg(&(vis_base as i32)).arg(&(nblocks as i32))
                .arg(&(cshift as i32)).arg(&(stride as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **§AK 候选 v5 探针**（global 分数缓冲 + 满占用，逐位等价于 prod）：`which` 0/1/2 = NH 1/2/4。
    /// `spg` 必须 ≥ `grid.y × t_q × (stride·2^cshift + 2^cshift) × NH` 个 f32（块私有槽，无原子）。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_v5_probe(
        &self, which: usize, q: &CudaView<bf16>, k: &CudaView<bf16>, v: &CudaView<bf16>,
        blk_tab: &CudaView<i32>,
        sel: &CudaView<i32>, sel_cnt: &CudaView<i32>, out: &mut CudaViewMut<bf16>,
        spg: &mut CudaViewMut<f32>,
        t_q: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        vis_base: usize, nblocks: usize, cshift: usize, stride: usize,
    ) -> Result<()> {
        let (fi, nhead) = match which {
            1 => (1usize, 2usize),
            2 => (2, 4usize),
            _ => (0, 1usize),
        };
        let f = self
            .qsa_probe
            .attn_v5
            .get(fi)
            .ok_or_else(|| anyhow::anyhow!("qsa_attn_v5_probe: which={which} 越界"))?;
        let ctok = 1usize << cshift;
        let span = stride * ctok + ctok;
        let need = (heads / nhead) * t_q * span * nhead;
        anyhow::ensure!(spg.len() >= need, "spg 太小: {} < {need}", spg.len());
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, (heads / nhead) as u32, 1), block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(f)
                .arg(q).arg(k).arg(v).arg(blk_tab).arg(sel).arg(sel_cnt).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(head_dim as i32))
                .arg(&scale).arg(&(vis_base as i32)).arg(&(nblocks as i32))
                .arg(&(cshift as i32)).arg(&(stride as i32)).arg(spg)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **§AK 候选 v6 探针**（grid 转置 `(heads/NH, t_q)`，算术与 v3 逐字相同 ⇒ 逐位等价）：
    /// `which` 0/1/2 = NH 1/2/4。其余参数与 [`Self::qsa_attn_probe`] 相同。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_v6_probe(
        &self, which: usize, q: &CudaView<bf16>, k: &CudaView<bf16>, v: &CudaView<bf16>,
        blk_tab: &CudaView<i32>,
        sel: &CudaView<i32>, sel_cnt: &CudaView<i32>, out: &mut CudaViewMut<bf16>,
        t_q: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        vis_base: usize, nblocks: usize, cshift: usize, stride: usize,
    ) -> Result<()> {
        let (fi, nhead) = match which {
            1 => (1usize, 2usize),
            2 => (2, 4usize),
            _ => (0, 1usize),
        };
        let f = self
            .qsa_probe
            .attn_v6
            .get(fi)
            .ok_or_else(|| anyhow::anyhow!("qsa_attn_v6_probe: which={which} 越界"))?;
        let ctok = 1usize << cshift;
        let smem = (stride * ctok + ctok) * 4 * nhead;
        let cfg = LaunchConfig {
            grid_dim: ((heads / nhead) as u32, t_q as u32, 1), block_dim: (256, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        unsafe {
            self.stream.launch_builder(f)
                .arg(q).arg(k).arg(v).arg(blk_tab).arg(sel).arg(sel_cnt).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(head_dim as i32))
                .arg(&scale).arg(&(vis_base as i32)).arg(&(nblocks as i32))
                .arg(&(cshift as i32)).arg(&(stride as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **§AK 候选 v7 探针**（pass2 V 走 smem 生产者/消费者，消费者累加次序不变 ⇒ 逐位等价）：
    /// `which` 0/1/2 = NH 1/2/4。smem = 分数（span·NH f32）+ vsm 站台（16·head_dim bf16）。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_v7_probe(
        &self, which: usize, q: &CudaView<bf16>, k: &CudaView<bf16>, v: &CudaView<bf16>,
        blk_tab: &CudaView<i32>,
        sel: &CudaView<i32>, sel_cnt: &CudaView<i32>, out: &mut CudaViewMut<bf16>,
        t_q: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        vis_base: usize, nblocks: usize, cshift: usize, stride: usize,
    ) -> Result<()> {
        let (fi, nhead) = match which {
            1 => (1usize, 2usize),
            2 => (2, 4usize),
            _ => (0, 1usize),
        };
        let f = self
            .qsa_probe
            .attn_v7
            .get(fi)
            .ok_or_else(|| anyhow::anyhow!("qsa_attn_v7_probe: which={which} 越界"))?;
        let ctok = 1usize << cshift;
        let span = stride * ctok + ctok;
        let smem = span * 4 * nhead + 16 * head_dim * 2;
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, (heads / nhead) as u32, 1), block_dim: (256, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        unsafe {
            self.stream.launch_builder(f)
                .arg(q).arg(k).arg(v).arg(blk_tab).arg(sel).arg(sel_cnt).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(head_dim as i32))
                .arg(&scale).arg(&(vis_base as i32)).arg(&(nblocks as i32))
                .arg(&(cshift as i32)).arg(&(stride as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **§AK 候选 v9 探针**（pass1 顺手缓存 V 行号进 smem，pass2 内圈 LDS→LDG 两级 + j×2
    /// 流水；数值与累加次序与 v3 逐字相同 ⇒ 逐位等价）：`which` 0/1/2 = NH 1/2/4。
    /// smem = 分数（span·NH f32）+ 行号（span i32）。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_v9_probe(
        &self, which: usize, q: &CudaView<bf16>, k: &CudaView<bf16>, v: &CudaView<bf16>,
        blk_tab: &CudaView<i32>,
        sel: &CudaView<i32>, sel_cnt: &CudaView<i32>, out: &mut CudaViewMut<bf16>,
        t_q: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        vis_base: usize, nblocks: usize, cshift: usize, stride: usize,
    ) -> Result<()> {
        let (fi, nhead) = match which {
            1 => (1usize, 2usize),
            2 => (2, 4usize),
            _ => (0, 1usize),
        };
        let f = self
            .qsa_probe
            .attn_v9
            .get(fi)
            .ok_or_else(|| anyhow::anyhow!("qsa_attn_v9_probe: which={which} 越界"))?;
        let ctok = 1usize << cshift;
        let span = stride * ctok + ctok;
        let smem = span * 4 * nhead + span * 4;
        let cfg = LaunchConfig {
            grid_dim: (t_q as u32, (heads / nhead) as u32, 1), block_dim: (256, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        unsafe {
            self.stream.launch_builder(f)
                .arg(q).arg(k).arg(v).arg(blk_tab).arg(sel).arg(sel_cnt).arg(out)
                .arg(&(heads as i32)).arg(&(kv_heads as i32)).arg(&(head_dim as i32))
                .arg(&scale).arg(&(vis_base as i32)).arg(&(nblocks as i32))
                .arg(&(cshift as i32)).arg(&(stride as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// **indexer 打分的微基准探针/候选**（`qsa_probe_exp.cu`）。
    /// `which`：0 prod 复刻 / 1 同形纯读 / 2 q 常量 / 3 pooled 常量 / 4 heads 展开（逐位相同）/
    /// 5 heads 展开 + 8 宽取数（逐位相同）。
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_idx_scores_probe(
        &self, which: usize, q: &CudaView<bf16>, pooled: &CudaView<bf16>,
        scores: &mut CudaViewMut<f32>, t_q: usize, nblocks: usize, heads: usize,
        hd: usize, sqrt_hd: f32,
    ) -> Result<()> {
        let f = self
            .qsa_probe
            .idx_scores
            .get(which)
            .ok_or_else(|| anyhow::anyhow!("qsa_idx_scores_probe: which={which} 越界"))?;
        let cfg = LaunchConfig {
            grid_dim: (Self::grid(nblocks), t_q as u32, 1), block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(f)
                .arg(q).arg(pooled).arg(scores)
                .arg(&(t_q as i32)).arg(&(nblocks as i32)).arg(&(heads as i32))
                .arg(&(hd as i32)).arg(&sqrt_hd)
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

    /// A 方案 t>1（`LINGXI_ROUTE_FULLDEV=1`）：设备端把 `ids/wts` 展开成专家升序的 pair 表
    /// + group_off/uniq + 行 CSR + 双视图槽号（互斥 -1）。`*_dev` 是**映射 host 内存的
    /// 设备别名**（u64 裸指针，0 = 不提供）；`slot_map_alt_dev`/`fmt_dev` 同理 0 = 单格式。
    /// 单 block 512 线程（每线程一个专家），输出次序与 host 簿记逐项一致。
    ///
    /// `pri_gidx`/`alt_gidx` = **按格式压缩的组下标表**（只含本格式的组，升序、无空洞）；
    /// GEMV 用它们间接索引全表 ⇒ 网格 y 从 `U` 降到 `npri`/`nalt`，真实组的三元组一字不改。
    #[allow(clippy::too_many_arguments)]
    pub fn route_pairs_bf16(
        &self, ids: &CudaSlice<i32>, wts: &CudaSlice<f32>,
        slot_map_dev: u64, slot_map_alt_dev: u64, fmt_dev: u64,
        pair_rows: &mut CudaSlice<i32>, pair_wts: &mut CudaSlice<f32>,
        group_slots: &mut CudaSlice<i32>, group_slots_alt_dev: u64,
        group_off: &mut CudaSlice<i32>, uniq_experts: &mut CudaSlice<i32>,
        col: &mut CudaSlice<i32>, roff: &mut CudaSlice<i32>,
        counts_dev: u64, hit_pri_dev: u64, hit_alt_dev: u64, miss_ids_dev: u64,
        pri_gidx: &mut CudaSlice<i32>, alt_gidx: &mut CudaSlice<i32>,
        e: usize, rows: usize, top_k: usize,
    ) -> Result<()> {
        assert!(e <= 512, "route_pairs: E≤512（block 固定 512）");
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.route_pairs_bf16)
                .arg(ids).arg(wts)
                .arg(&slot_map_dev).arg(&slot_map_alt_dev).arg(&fmt_dev)
                .arg(pair_rows).arg(pair_wts).arg(group_slots)
                .arg(&group_slots_alt_dev)
                .arg(group_off).arg(uniq_experts).arg(col).arg(roff)
                .arg(&counts_dev).arg(&hit_pri_dev).arg(&hit_alt_dev).arg(&miss_ids_dev)
                .arg(pri_gidx).arg(alt_gidx)
                .arg(&(e as i32)).arg(&(rows as i32)).arg(&(top_k as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ★ 2026-10-24 **批量路由 top-k：一 warp 一行**（`route_fuse.cu`，逐位相同）。
    ///
    /// ⚠️ **负结果（未接生产）**：微基准实测 **15.5 µs > 旧核 13.4 µs**（同进程 best_of3）。
    /// 原因：top-k 的净活主要是**逐元素 expf + 建键**，旧核 6 行 × 256 线程（每线程 2 个
    /// 元素）的并行度比 6 × 32（每线程 16 个元素）划算；省下的「逐轮 block 归约 +
    /// `__syncthreads`」并不足以抵消并行度的损失。仅保留给 `tests/t20_route_chain_bw.rs`
    /// 做分母对照与后续参考（判据 `tests/t21_route_fuse_bitid.rs::t21_topk_warp_bitid`）。
    pub fn route_topk_row_warp_bf16(
        &self, logits: &CudaSlice<bf16>, ids: &mut CudaSlice<i32>, wts: &mut CudaSlice<f32>,
        rows: usize, e: usize, top_k: usize,
    ) -> Result<bool> {
        if e > 512 || top_k > 64 {
            self.route_topk_rows_bf16(logits, ids, wts, rows, e, top_k)?;
            return Ok(false);
        }
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.route_topk_row_warp)
                .arg(logits).arg(ids).arg(wts)
                .arg(&(e as i32)).arg(&(top_k as i32))
                .launch(cfg)?;
        }
        Ok(true)
    }

    /// ★ 2026-10-24 **MoE router 投影的 m ≤ 6 特化核**（`router_m6.cu`）。
    ///
    /// 动机（`tests/t20_route_chain_bw.rs`）：router 形状 (n=512,k=2560,m=6) 的
    /// `gemm_skinny_row_bf16` 是 16.0 µs（164 GB/s），而**同 grid/block/地址流的纯读**
    /// 只要 4.7 µs（555 GB/s）⇒ 11 µs 花在 CVT/FFMA 发射上（旧核每轮对 6 个 token 行
    /// 各转一遍 8 个 w）。本核：m 编译期 6、w 每轮只转一次、行偏移变常量。
    ///
    /// **逐位相同**（`tests/t22_router_m6_bitid.rs`）：线程→列映射、组内加法次序、
    /// `acc[s] += d`、两级归约树、epi 全部与行核逐字同构；bf16→f32 精确。
    /// `LINGXI_NO_ROUTER_M6=1` 回退；`m > 6` 自动回退。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_skinny_row_m6_bf16_ptr(
        &self, w: u64, x: u64, y: u64, n: usize, k: usize, m: usize,
        epi: i32, a: f32, b: f32,
    ) -> Result<()> {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF.get_or_init(|| std::env::var("LINGXI_NO_ROUTER_M6").map(|v| v != "0").unwrap_or(false));
        if off || m == 0 || m > 6 {
            return self.gemm_skinny_row_bf16_ptr(w, x, y, n, k, m.max(1), epi, a, b);
        }
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.gemm_skinny_row_m6)
                .arg(&w).arg(&x).arg(&y)
                .arg(&(n as i32)).arg(&(k as i32)).arg(&(m as i32))
                .arg(&epi).arg(&a).arg(&b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ★ 2026-10-24 **top-k + pair 建表合成一发**（`route_fuse.cu`）。
    ///
    /// 动机（`tests/t20_route_chain_bw.rs`，同进程背靠背 best_of3）：topk 13.4 µs 与
    /// pair 12.0 µs 各有 ~6 µs 是同 grid 空核的**发射地板**，而两者是纯串行依赖
    /// （pair 吃 topk 的 ids）⇒ 合成一发直接省掉一次地板。
    ///
    /// **逐位相同**：topk 段改成「一 warp 管一行」（max/键比较都是精确或唯一运算，
    /// `sum`/`norm` 仍是单线程按专家下标/名次升序串行），pair 段核体与
    /// `lingxi_route_pairs_bf16` 逐字相同。判据：`tests/t21_route_fuse_bitid.rs`
    /// （25 组随机输入 × rows 2..16 × 单/双视图 × 含 miss，全表逐项比对）。
    ///
    /// 返回 `Ok(true)` = 走了合成核；`Ok(false)` = 已按旧的**两发**跑完（默认，或形状越界），
    /// 调用方据此决定 s3 段标记怎么打。
    #[allow(clippy::too_many_arguments)]
    pub fn route_topk_pairs_bf16(
        &self, logits: &CudaSlice<bf16>, ids: &mut CudaSlice<i32>, wts: &mut CudaSlice<f32>,
        slot_map_dev: u64, slot_map_alt_dev: u64, fmt_dev: u64,
        pair_rows: &mut CudaSlice<i32>, pair_wts: &mut CudaSlice<f32>,
        group_slots: &mut CudaSlice<i32>, group_slots_alt_dev: u64,
        group_off: &mut CudaSlice<i32>, uniq_experts: &mut CudaSlice<i32>,
        col: &mut CudaSlice<i32>, roff: &mut CudaSlice<i32>,
        counts_dev: u64, hit_pri_dev: u64, hit_alt_dev: u64, miss_ids_dev: u64,
        pri_gidx: &mut CudaSlice<i32>, alt_gidx: &mut CudaSlice<i32>,
        e: usize, rows: usize, top_k: usize,
    ) -> Result<bool> {
        let fuse = tkpair_fuse_on() && e <= 512 && rows <= 16 && top_k <= 64;
        if !fuse {
            self.route_topk_rows_bf16(logits, ids, wts, rows, e, top_k)?;
            self.route_pairs_bf16(
                ids, wts, slot_map_dev, slot_map_alt_dev, fmt_dev, pair_rows, pair_wts,
                group_slots, group_slots_alt_dev, group_off, uniq_experts, col, roff,
                counts_dev, hit_pri_dev, hit_alt_dev, miss_ids_dev, pri_gidx, alt_gidx,
                e, rows, top_k,
            )?;
            return Ok(false);
        }
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.route_topk_pairs_bf16)
                .arg(logits).arg(ids).arg(wts)
                .arg(&slot_map_dev).arg(&slot_map_alt_dev).arg(&fmt_dev)
                .arg(pair_rows).arg(pair_wts).arg(group_slots)
                .arg(&group_slots_alt_dev)
                .arg(group_off).arg(uniq_experts).arg(col).arg(roff)
                .arg(&counts_dev).arg(&hit_pri_dev).arg(&hit_alt_dev).arg(&miss_ids_dev)
                .arg(pri_gidx).arg(alt_gidx)
                .arg(&(e as i32)).arg(&(rows as i32)).arg(&(top_k as i32))
                .launch(cfg)?;
        }
        Ok(true)
    }

    /// ㊿+ 2026-10-22 批量设备路由：grid.x = 行，每 block 逐位复刻一行 host
    /// softmax + 插入式 topk；输出 `ids[rows,top_k]`（专家升序）+ 对齐 `wts`。
    /// 供 moe.rs rows>1 路径替换「D2H logits + host softmax/topk」。
    ///
    /// `LINGXI_NO_ROUTE_TOPK_FAST=1`（2026-10-23）：整核回退到旧实现
    /// `lingxi_route_topk_rows_bf16_legacy`（O(E²) 排名 + tid==0 串行输出）。
    pub fn route_topk_rows_bf16(
        &self, logits: &CudaSlice<bf16>, ids: &mut CudaSlice<i32>, wts: &mut CudaSlice<f32>,
        rows: usize, e: usize, top_k: usize,
    ) -> Result<()> {
        assert!(e <= 1024 && top_k <= 64, "route_topk_rows: E≤1024、top_k≤64");
        let legacy = *ROUTE_TOPK_LEGACY.get_or_init(|| std::env::var_os("LINGXI_NO_ROUTE_TOPK_FAST").is_some());
        let f = if legacy { &self.route_topk_rows_bf16_legacy } else { &self.route_topk_rows_bf16 };
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(f)
                .arg(logits).arg(ids).arg(wts)
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

