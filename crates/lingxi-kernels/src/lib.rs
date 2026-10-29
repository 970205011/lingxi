//! `lingxi-kernels` — L1：CUDA C++ → cubin（build.rs 调 nvcc）+ cudarc 加载/launch。
//!
//! 职责边界（见 `ARCHITECTURE.md`）：
//!   * 纯函数 kernel：`(指针 + 参数) → 输出`，无全局态、无隐式 stream 依赖。
//!   * `unsafe` 只允许出现在本 crate 的 launch 封装里。
//!   * 矩阵乘**复用 cuBLAS**（复用铁律）；只自研 cuBLAS 不适合的融合/稀疏/注意力算子。

use std::sync::Arc;

pub mod evprof;
use evprof::EvProf;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream};
use lingxi_ids::{KernelId, KernelRegistry, OpId, ShapeTag};

/// 目标架构（固定 sm_70）。
// CODE_MAP Phase 4 刀P4-1：impl Kernels 的 launch 封装搬到 wrappers.rs（只搬家不改行为）。
mod wrappers;

pub const CUDA_ARCH: &str = "sm_70";

pub static SMOKE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smoke.cubin"));
pub static RMSNORM_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.cubin"));
pub static GR_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gr_ops.cubin"));
pub static MOE_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_ops.cubin"));
pub static ATTN_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_ops.cubin"));
pub static DEQUANT_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dequant_ops.cubin"));
pub static GDN_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_ops.cubin"));
pub static TOPK_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/topk_ops.cubin"));
/// 2026-10-24 top-k + pair 建表的**合成一发**核（`LINGXI_NO_TKPAIR_FUSE=1` 回退两发）。
pub static ROUTE_FUSE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/route_fuse.cubin"));
/// 2026-10-24 MoE router 投影的 **m ≤ 6 编译期特化**核（`LINGXI_NO_ROUTER_M6=1` 回退）。
pub static ROUTER_M6_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/router_m6.cubin"));
/// Q4 专家张量核 GEMM（Volta m8n8k4）
pub static Q4_TC_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_tc.cubin"));
/// Q4 专家张量核 GEMM —— prefill 大 tile 版（M 轴分块 + x 进 shared）
pub static Q4_TCP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_tcp.cubin"));
pub static Q3_TCP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q3_tcp.cubin"));
pub static MLARGE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mlarge.cubin"));
pub static Q4_MLARGE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_mlarge.cubin"));
/// Q4 稠密投影 TCP MMA + 大 m 摊销核（q4_tcp 的 B 片段 + 稠密 W 布局）
pub static Q4_TCP_DENSE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_tcp_dense.cubin"));
/// 稠密 bf16 投影 Volta MMA 核（q4_tcp_dense 骨架去反量化，`LINGXI_PROJ_TC=1` 门控）
pub static BF16_TCP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bf16_tcp.cubin"));
/// 专家段 GEMV 的**实验核**（split-k / 纯读探针）：只有 microbench 会调，生产不用。
pub static GEMV_PAR_EXP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_par_exp.cubin"));
/// 专家段 GEMV 的**乘加结构实验核**（阶梯探针 P0~P9 + uint4 结构变体）：只有 microbench 会调。
pub static GEMV_STRUCT_EXP_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemv_struct_exp.cubin"));
/// Q3G128 专家 GEMV 的**取数结构实验核**（同形纯读 + 逐位等价变体）：只有 microbench 会调。
pub static GEMV_Q3_EXP_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q3_exp.cubin"));
/// 稠密 Q4 瘦长 GEMM 的**取数结构探针**（同形纯读 / ALU 地板 / staging 隔离）：只有 microbench 会调。
pub static Q4_DENSE_EXP_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/q4_dense_exp.cubin"));
/// bf16 瘦长 GEMM + 草稿 lm_head 的**取数结构探针**：只有 microbench 会调。
pub static BF16_SKINNY_EXP_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bf16_skinny_exp.cubin"));
/// GDN 递推（`gdn_recurrent_bf16`）的**分母探针 + state 常驻候选**：只有 microbench 会调。
pub static GDN_REC_EXP_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gdn_rec_exp.cubin"));
/// QSA `idxprep / idxsel / attn` 三段的**分母探针 + 候选**：只有 microbench 会调。
pub static QSA_PROBE_EXP_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/qsa_probe_exp.cubin"));
/// GR `t=1`（草稿占位）的**同形纯读分母**：只有 microbench 会调。
pub static GR_PROBE_T1_CUBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gr_probe_t1.cubin"));

/// `gemv_struct_exp.cu` 的句柄集合（微基准专用，生产路径不引用）。
pub struct GemvStructExp {
    /// P0..P9 阶梯探针（与生产核同访存形状；P7=移位对解包，P8=scale smem，P9=+x smem）
    pub probe: [CudaFunction; 8],
    pub p8_smsc: CudaFunction,
    pub p9_smscx: CudaFunction,
    pub u128_r1: CudaFunction,
    pub u128_r2: CudaFunction,
    pub u128_r4: CudaFunction,
    pub contig: CudaFunction,
    pub u128_r2_sm: CudaFunction,
    pub u128_r1_dp4a: CudaFunction,
    /// 生产算式 × 预取结构（MB=8 多对骨架；全部与生产核逐位等价）
    pub tier_unr1: CudaFunction,
    pub tier_unr2: CudaFunction,
    pub tier_unr4: CudaFunction,
    pub tier_unr4_proddec: CudaFunction,
    pub tier_shift4: CudaFunction,
}

/// `gemv_q3_exp.cu` 的句柄集合（微基准专用，生产路径不引用）。
pub struct GemvQ3Exp {
    /// 同形纯读：mode 0 = 3×1 B / 1 = 对齐 u32 对 / 2 = 16 B uint4 / 3 = 对齐 u32 对 + 解包 + I2F
    pub read_probe: CudaFunction,
    /// 计算变体（**全部与生产核逐位等价**），`which` 见 [`crate::Kernels::gemv_q3g128_slotsm_exp`]
    pub v: [CudaFunction; 12],
}

/// `bf16_skinny_exp.cu` 的句柄集合（微基准专用，生产路径不引用）。
/// 形状与线上一致：A 族 m8 体 grid=`ceil(n/8)`×256（一 warp 一行）、row 体 grid=`n`×256；
/// B 族 grid=`ceil(n/8)`×256（一 warp 一行）。
pub struct Bf16SkinnyExp {
    /// m8 体：纯读 2 B/lane（生产取数形状）
    pub m8_read1: CudaFunction,
    /// m8 体：纯读 16 B/lane uint4（结构上限）
    pub m8_read16: CudaFunction,
    /// m8 体：2 B/lane + cvt + m×FFMA（x 用常量 ⇒ 不含 x 读的地板）
    pub m8_alu: CudaFunction,
    /// m8 体：生产内层逐字复刻（自检用）
    pub m8_prod: CudaFunction,
    /// m8 体：一 warp 连算 R 行（2/4/8，逐位等价）
    pub m8_r: [CudaFunction; 3],
    /// m8 体：R=4 + 权重 PF=2
    pub m8_r4pf: CudaFunction,
    /// m8 体：R=4 + x 转 f32 进 smem
    pub m8_r4x: CudaFunction,
    /// row 体：纯读 16 B/lane（生产取数形状）
    pub row_read16: CudaFunction,
    /// row 体：16 B/lane + cvt + m×8 FFMA（x 用常量）
    pub row_alu: CudaFunction,
    /// row 体：生产内层逐字复刻（自检用）
    pub row_prod: CudaFunction,
    /// row 体：一 block 连算 2 行（逐位等价）/ +PF=2
    pub row_r2: CudaFunction,
    pub row_r2pf: CudaFunction,
    /// 草稿 lm_head：纯读 4 B/lane + scale（生产取数形状）/ 16 B/lane（上限）/ 生产复刻
    pub q4h_read1: CudaFunction,
    pub q4h_read16: CudaFunction,
    pub q4h_prod: CudaFunction,
    /// 草稿 lm_head 的候选结构：权重 PF=2 / 一 warp 连算 2 行（均逐位等价）
    pub q4h_pf: CudaFunction,
    pub q4h_r2: CudaFunction,
    /// x 侧单独记账 / 预取 / x 进 smem
    pub m8_xonly: CudaFunction,
    pub m8_r1pf: CudaFunction,
    pub m8_r1x: CudaFunction,
    pub m8_r2x: CudaFunction,
    pub row_xonly: CudaFunction,
    pub row_r4: CudaFunction,
    /// 生产体逐字复刻（含 blockIdx.y 行分块）/ m 编译期特化到 6
    pub m8_prod_r0: CudaFunction,
    pub m8_m6: CudaFunction,
    pub m8_r0hoist: CudaFunction,
}

pub(crate) const BLOCK: u32 = 256;

/// `gdn_rec_exp.cu` 的句柄集合（微基准专用，生产路径不引用）。
/// 形状固定为线上 GDN 递推：48 v-head × hk128 × hv128 × f32，grid=48 × block=128。
pub struct GdnRecExp {
    /// 生产核逐字复刻（自检：应与 `gdn_recurrent_bf16` 逐位相同）
    pub prod: CudaFunction,
    /// 同形纯读 S 两遍（= 生产核每 token 的读遍数）/ 一遍
    pub read2: CudaFunction,
    pub read1: CudaFunction,
    /// 同形就地纯搬运 2R+2W / 1R+1W（不做递推）
    pub copy2: CudaFunction,
    pub copy1: CudaFunction,
    /// 候选：state 全程驻留寄存器（1R+1W 总）/ 不碰 S（ALU 地板）/ 每 token 写回
    pub reg: CudaFunction,
    pub reg_nos: CudaFunction,
    pub reg_tok: CudaFunction,
    /// 候选：state 驻留 smem（decay 第二遍重算 ⇒ smem 2R+1W）/ 老档（smem 2R+2W）
    pub smem: CudaFunction,
    pub smem_old: CudaFunction,
    /// 候选：q/k 走 smem（scaled 预计算）+ state 全寄存器 / state 半寄存器半 smem
    pub v2: CudaFunction,
    pub hyb: CudaFunction,
    /// 候选 v3：v2 + q/k 的 smem 读改 float4
    pub v3: CudaFunction,
    /// 消融：q/k 全用常量（量化"取数延迟"）
    pub nok: CudaFunction,
}

/// `qsa_probe_exp.cu` 的句柄集合（微基准专用，生产路径不引用）。
/// 形状固定为线上 QSA：t=6 / heads24 / kv2 / hd256 / compress4 / topk512。
pub struct QsaProbeExp {
    /// attn gather 探针，顺序见 [`crate::Kernels::qsa_attn_probe`]
    /// （18..=20 = 候选 v3：pass1/pass2 取数流水化 + sp [idx*NH+g] 布局，逐位等价）
    pub attn: [CudaFunction; 21],
    /// 候选 v5（global 分数缓冲 + 满占用），下标 0/1/2 = NH 1/2/4
    pub attn_v5: [CudaFunction; 3],
    /// 候选 v6（grid 转置：同 row 的头块调度相邻），下标 0/1/2 = NH 1/2/4
    pub attn_v6: [CudaFunction; 3],
    /// 候选 v7（pass2 V 走 smem 生产者/消费者），下标 0/1/2 = NH 1/2/4
    pub attn_v7: [CudaFunction; 3],
    /// 候选 v9（V 行地址预计算进 smem + j×2 流水），下标 0/1/2 = NH 1/2/4
    pub attn_v9: [CudaFunction; 3],
    /// indexer 打分探针，顺序见 [`crate::Kernels::qsa_idx_scores_probe`]
    pub idx_scores: [CudaFunction; 6],
}

/// `gr_probe_t1.cu` 的句柄集合（微基准专用，生产路径不引用）。
/// 三条都是**只读不算**：地址流分别对齐 `gemv_bf16_row8_epi` / `gemv_bf16_epi` /
/// 理想流式读。
pub struct GrProbeT1 {
    /// 一行一 block（对应 mix_down / block_inject 的 t=1 路径）
    pub read_row8: CudaFunction,
    /// 一 warp 一行（对应 mix_up，k=320）
    pub read_warp: CudaFunction,
    /// 同字节数的理想流式读
    pub read_stream: CudaFunction,
}

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
/// 为什么要多这一条（2026-10-16）：READ_ONLY 注册的语义是"**让设备去读**这块 host
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
    /// **专家段的第二条流**（`LINGXI_MOE_OVERLAP=1`）。
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
    pub(crate) smoke_scale: CudaFunction,
    pub(crate) rmsnorm_f32: CudaFunction,
    pub(crate) grouped_rmsnorm_bf16: CudaFunction,
    pub(crate) gr_down_fuse_bf16: CudaFunction,
    pub(crate) gr_up_mix_bf16: CudaFunction,
    /// 通用档（rows 2..8）合发核：`mix_down + block_inject` / `mix_up + sigmoid + mix_mean`
    pub(crate) gr_down_inject_bf16: CudaFunction,
    pub(crate) gr_up_mix_fuse_bf16: CudaFunction,
    /// GR `t=1` 的**同形纯读分母**（微基准专用，见 `gr_probe_t1.cu`）
    pub gr_probe_t1: GrProbeT1,
    pub(crate) silu_scale_inplace_bf16: CudaFunction,
    pub(crate) sigmoid_scale_mul_inplace_bf16: CudaFunction,
    pub(crate) mix_mean_bf16: CudaFunction,
    pub(crate) gr_write_inplace_bf16: CudaFunction,
    pub(crate) hc_repeat_bf16: CudaFunction,
    pub(crate) mul_bf16: CudaFunction,
    pub(crate) swiglu_bf16: CudaFunction,
    pub(crate) gather_rows_bf16: CudaFunction,
    pub(crate) scatter_add_rows_bf16: CudaFunction,
    pub(crate) row_scale_inplace_bf16: CudaFunction,
    pub(crate) moe_combine_rows_bf16: CudaFunction,
    pub(crate) moe_combine_add_rows_bf16: CudaFunction,
    /// CSR 版合并+残差（batch 路径，O(rows×H×top_k) 取代旧核 O(rows×H×S)）
    pub(crate) moe_combine_add_csr_bf16: CudaFunction,
    /// A 方案 t>1：批量设备路由的 pair/CSR 构建核（`moe_ops.cu`）
    pub(crate) route_pairs_bf16: CudaFunction,
    /// 2026-10-24：top-k 与 pair 建表**合成一发**（`route_fuse.cu`，与上面两发逐位相同）
    pub(crate) route_topk_pairs_bf16: CudaFunction,
    /// 2026-10-24：MoE router 投影的 m≤6 特化核（`router_m6.cu`，与行核逐位相同）
    pub(crate) gemm_skinny_row_m6: CudaFunction,
    /// 2026-10-24：批量路由 top-k 的**一 warp 一行**核（`route_fuse.cu`，与旧核逐位相同）
    pub(crate) route_topk_row_warp: CudaFunction,
    pub(crate) gemv_bf16: CudaFunction,
    pub(crate) gemv_bf16_epi: CudaFunction,
    pub(crate) gemv_bf16_block_epi: CudaFunction,
    pub(crate) gemv_bf16_row8_epi: CudaFunction,
    /// Q4 专家张量核 GEMM（T≤8 / T=9..16），`LINGXI_MOE_TC=1` 才用
    pub(crate) q4_tc_t1: CudaFunction,
    pub(crate) q4_tc_t2: CudaFunction,
    /// prefill 大 tile 版（任意组大小一次 launch），`LINGXI_MOE_TC=1` 门控
    pub(crate) q4_tcp: CudaFunction,
    pub(crate) q3_tcp: CudaFunction,
    /// 乐观发射档（BIG 路径）：组数网格按 host 上界发、行块核内循环，
    /// 可在 pair 建表核完成前入队。算术与旧档逐 (组, 行块) 相同。
    pub(crate) q4_tcp_dyn: CudaFunction,
    pub(crate) q3_tcp_dyn: CudaFunction,
    pub(crate) mlarge: CudaFunction,
    pub(crate) q4_mlarge: CudaFunction,
    /// Q4 稠密投影 TCP MMA + 大 m 摊销核（`q4_tcp_dense.cu`）
    pub(crate) q4_tcp_dense: CudaFunction,
    /// M≤8 变体（verify 批 t=6 专用，M 轴 tile = 8）
    pub(crate) q4_tcp_dense_m8: CudaFunction,
    /// 稠密 bf16 投影 Volta MMA 核（`bf16_tcp.cu`，`LINGXI_PROJ_TC=1` 门控）
    pub(crate) bf16_tcp: CudaFunction,
    /// 瘦长 GEMM，按 m=1..8 各一份（索引 `m-1`）。必须按 m 特化：
    /// 固定开 8 个累加器的版本在 m=1 时只有 215 GB/s，而 GEMV 有 724 GB/s（指令发射受限）。
    pub(crate) gemm_skinny: [CudaFunction; 8],
    pub(crate) gemm_skinny_row: CudaFunction,
    /// ★ 2026-10-24：bf16 瘦长 GEMM 的 `grid.y==1` 扁平成（1≤m≤8，逐位等价于
    /// `gemm_skinny[7]`，只去掉 `r0`/`mt` 的运行期行偏移）。`LINGXI_NO_BF16_FLAT=1` 回退。
    pub(crate) gemm_skinny_m8_flat: CudaFunction,
    pub(crate) gemm_skinny_row_multi: CudaFunction,
    pub(crate) gemm_skinny_row_f32x: CudaFunction,
    pub(crate) gemm_skinny_mwide: CudaFunction,
    /// ★ 2026-10-24 shared expert 的两个融合核（见 `moe_ops.cu`）：
    /// 1) gate_up 的 gate/up 配对 + swiglu + shared_gate 行（一次 launch）；
    /// 2) dn 投影 + row_scale 尾（一次 launch）。都与原路径**逐位等价**。
    pub(crate) gemm_skinny_row_gu_swiglu_sg: CudaFunction,
    pub(crate) gemm_skinny_m8_scale: CudaFunction,
    pub(crate) gemm_q4_skinny_bf16: CudaFunction,
    pub(crate) gemm_q4_skinny_dbg: CudaFunction,
    pub(crate) gemm_q4_s8_bf16: CudaFunction,
    /// ★ 2026-10-24 两个"纯合发"核（`dequant_ops.cu`，只并发射、逐位等价）：
    /// 1) QSA 的 `q | k | gate | v` 四段投影 → 1 发（按全局行号落回四个输出）；
    /// 2) GDN 的 `qkvz`（Q4 行）+ `a/b`（bf16 行）→ 1 发。
    pub(crate) gemm_q4_skinny4_bf16: CudaFunction,
    pub(crate) gemm_q4_skinny_ab_bf16: CudaFunction,
    /// ★ 2026-10-24 **R2 档**（一个 warp 连算 2 个输出行；`LINGXI_NO_Q4D_R2=1` 回退）：
    /// 与上面三个旧核**逐位相同**（算式/列集合/累加次序/归约树一字未改），只把 x 的 smem 读、
    /// bf16→f32 与舞台流量按 2 行摊薄。实测 m=6 10240×2560：133 → 173 GB/s。
    pub(crate) gemm_q4_skinny_r2_bf16: CudaFunction,
    pub(crate) gemm_q4_skinny4_r2_bf16: CudaFunction,
    pub(crate) gemm_q4_skinny_ab_r2_bf16: CudaFunction,
    pub(crate) ple_gate_bf16: CudaFunction,
    pub(crate) add_bf16: CudaFunction,
    pub(crate) rope_partial_inplace_bf16: CudaFunction,
    /// ★ 小发射合并：`norm+rope` 一发（`attn_ops.cu`，与分开两发逐位相同）
    pub(crate) rmsnorm_rope_bf16: CudaFunction,
    /// ★ 小发射合并：同一份 x 的两块 bf16 瘦长投影一发（与 `gemm_skinny_row_bf16` 逐位相同）
    pub(crate) gemm_bf16_row_pair: CudaFunction,
    pub(crate) attn_masked_bf16: CudaFunction,
    pub(crate) attn2_masked_bf16: CudaFunction,
    pub(crate) attn2s_sparse_bf16: CudaFunction,
    pub(crate) attn3_masked_bf16: CudaFunction,
    pub(crate) attn5_sparse_bf16: CudaFunction,
    pub(crate) attn2_masked_kvq8_bf16: CudaFunction,
    pub(crate) attn2s_sparse_kvq8_bf16: CudaFunction,
    pub(crate) attn3_masked_kvq8_bf16: CudaFunction,
    pub(crate) attn5_sparse_kvq8_bf16: CudaFunction,
    /// FP8-G32/int8-G32 KV 探索档（`LINGXI_KV_FP8`）：每 32 维一组 f16 scale 的
    /// E4M3/int8 注意力（+ KIVI 式残差窗）
    pub(crate) attn2_masked_kvfp8_bf16: CudaFunction,
    pub(crate) attn2s_sparse_kvfp8_bf16: CudaFunction,
    pub(crate) attn3_masked_kvfp8_bf16: CudaFunction,
    pub(crate) attn5_sparse_kvfp8_bf16: CudaFunction,
    /// FP8-G32/int8-G32 行组量化：bf16 → 字节 + 每 32 维一组 f16 scale
    pub(crate) quant_rows_fp8g32_bf16: CudaFunction,
    /// FP8-G32/int8-G32 残差窗 flush 量化（源 = bf16 残差平面，sink 段 + 尾窗环）
    pub(crate) quant_rows_fp8g32_res_bf16: CudaFunction,
    pub(crate) quant_rows_e4m3_bf16: CudaFunction,
    pub(crate) qsa_pool_block_bf16: CudaFunction,
    pub(crate) qsa_pool_blocks_bf16: CudaFunction,
    /// ★ 池化 + RMSNorm + RoPE + 落盘 四发合一（与四发逐位相同）
    pub(crate) qsa_pool_norm_rope_bf16: CudaFunction,
    /// ★ `nblocks ≤ topk` 时 keep 全 1（省掉打分 + 名次两发）
    pub(crate) qsa_keep_all_bf16: CudaFunction,
    pub(crate) rope_partial_stride_bf16: CudaFunction,
    pub(crate) qsa_indexer_scores_bf16: CudaFunction,
    pub(crate) qsa_indexer_rank_f32: CudaFunction,
    pub(crate) qsa_keep_select_bf16: CudaFunction,
    pub(crate) qsa_keep_fill_bf16: CudaFunction,
    pub(crate) qsa_keep_fused_bf16: CudaFunction,
    pub(crate) qsa_keep_from_scores_bf16: CudaFunction,
    pub(crate) qsa_sel_build: CudaFunction,
    pub(crate) qsa_hist_f32: CudaFunction,
    pub(crate) qsa_bin_finalize: CudaFunction,
    pub(crate) qsa_keep_fill_scores: CudaFunction,
    pub(crate) attn_gather_bf16: CudaFunction,
    pub(crate) attn_gather4_bf16: CudaFunction,
    /// ★ 2026-10-24：gather 第二遍"按选块分组取 V"的档（`LINGXI_NO_ATTN_V2=1` 回退到
    /// 上面的原核；两者逐位相同）+ NH=2 的档（`LINGXI_ATTN_GATHER_H=2`）。
    pub(crate) attn_gather2_bf16: CudaFunction,
    pub(crate) attn_gather_v1_bf16: CudaFunction,
    pub(crate) attn_gather4_v1_bf16: CudaFunction,
    /// gather v3 生产核（prefill 形态重排，逐位等价；`LINGXI_NO_ATTN_GATHER_V3=1` 回退）
    pub(crate) attn_gather_v3_2_bf16: CudaFunction,
    pub(crate) attn_gather_v3_4_bf16: CudaFunction,
    pub(crate) qsa_mask_fill_f32: CudaFunction,
    pub(crate) dequant_rowsplit_q4g64_bf16: CudaFunction,
    pub(crate) dequant_rowsplit_q4g64_batch: CudaFunction,
    pub(crate) dequant_rowsplit_q3g128_bf16: CudaFunction,
    pub(crate) gemv_q4g64_slots_bf16: CudaFunction,
    pub(crate) gemv_q4g64_slots_r2_bf16: CudaFunction,
    pub(crate) gemv_q4g64_slots_r4_bf16: CudaFunction,
    pub(crate) gemv_q4g64_slots_r8_bf16: CudaFunction,
    pub(crate) gemv_q4g64_slotsm_bf16: CudaFunction,
    pub(crate) gemv_q4g64_slotsm_r2_bf16: CudaFunction,
    pub(crate) gemv_q4g64_slotsm_r4_bf16: CudaFunction,
    /// `LINGXI_MOE_GEMV_TIER=1` 的可选档：与 `slotsm_bf16` **逐位相同**，只改取数结构
    /// （两个独立 load + 移位对解包；见 dequant_ops.cu 该核头注）。
    pub(crate) gemv_q4g64_slotsm_t2_bf16: CudaFunction,
    /// Q4 的**设备侧组数上界**档（A' 的地基）：与 `..._t2_bf16` 同一核体，
    /// 入口多一道 `e >= *ng_dev` 早退 ⇒ 主机可发静态网格。
    pub(crate) gemv_q4g64_slotsm_t2_ng_bf16: CudaFunction,
    pub(crate) gemv_q3g128_slots_bf16: CudaFunction,
    pub(crate) gemv_q3g128_slots_r2_bf16: CudaFunction,
    pub(crate) gemv_q3g128_slots_r4_bf16: CudaFunction,
    pub(crate) gemv_q3g128_slots_r8_bf16: CudaFunction,
    pub(crate) gemv_q3g128_slotsm_bf16: CudaFunction,
    pub(crate) gemv_q3g128_slotsm_r2_bf16: CudaFunction,
    pub(crate) gemv_q3g128_slotsm_r4_bf16: CudaFunction,
    /// Q3 的取数结构档（默认启用，`LINGXI_NO_Q3_T2=1` 回退）：与 `slotsm_bf16` **逐位相同**，
    /// 只把"3×1 B 取数 + 生产解包 + PF=4 移位预取"换成"两对齐 u32 + 移位对 + 独立 load"。
    /// 三个实例：通用（MB=8）/ 每组恰 1 对的 MB=1（UNR=2 长 k、UNR=1 短 k）。
    pub(crate) gemv_q3g128_slotsm_t2_bf16: CudaFunction,
    pub(crate) gemv_q3g128_slotsm_t2_mb1u2_bf16: CudaFunction,
    pub(crate) gemv_q3g128_slotsm_t2_mb1u1_bf16: CudaFunction,
    /// 同上的**设备侧组数上界**档（A' 的地基）：入口多一道 `e >= *ng_dev` 早退，
    /// 于是主机可以发静态网格 `grid.y = s` 而不必先等 `npri/nalt`（= 每层 345~566 µs 的 route_spin）。
    pub(crate) gemv_q3g128_slotsm_t2_ng_bf16: CudaFunction,
    /// Q3G128 的 **int8 激活 + `__dp4a`** 档（非位等价；`LINGXI_MOE_DP4A=1` 才用）：
    /// 每 token 的边际成本从 8.5 条压到 ~5 条（m≈5 时理论上限 1.41×）。
    pub(crate) gemv_q3g128_slotsm_dp4a: CudaFunction,
    pub(crate) gemv_q4g64_slotsm_dp4a: CudaFunction,
    pub(crate) gemv_q4g64_slotsm_r2_dp4a: CudaFunction,
    pub(crate) gemv_q4g64_slotsm_sxb_bf16: CudaFunction,
    pub(crate) gemv_q4g64_slotsm_sxq_dp4a: CudaFunction,
    /// 专家段 GEMV 实验档（`gemv_par_exp.cu`）：split-k 与纯读探针（只有 microbench 用）。
    pub(crate) gemv_q4g64_slotsm_splitk_bf16: CudaFunction,
    pub(crate) gemv_splitk_reduce_bf16: CudaFunction,
    pub(crate) read_probe_slots: CudaFunction,
    pub(crate) read_probe_flat: CudaFunction,
    /// 专家段 GEMV **乘加结构**实验档（`gemv_struct_exp.cu`，只有 microbench 用）。
    pub struct_exp: GemvStructExp,
    /// Q3G128 专家 GEMV **取数结构**实验档（`gemv_q3_exp.cu`，只有 microbench 用）。
    pub q3_exp: GemvQ3Exp,
    /// 稠密 Q4 瘦长 GEMM **取数结构**探针（`q4_dense_exp.cu`，只有 microbench 用）。
    pub q4_dense_exp: CudaFunction,
    /// 稠密 Q4 瘦长 GEMM 的**多行/warp 变体**（同上文件；0=R1W8/1=R2W8/2=R4W8/3=R1W16/4=R2W16）。
    pub q4_dense_rows_probe: [CudaFunction; 5],
    /// bf16 瘦长 GEMM 与草稿 lm_head 的**取数结构探针**（`bf16_skinny_exp.cu`，只有 microbench 用）。
    pub bf16_exp: Bf16SkinnyExp,
    /// GDN 递推的分母探针 + state 常驻候选（`gdn_rec_exp.cu`，只有 microbench 用）。
    pub gdn_rec_exp: GdnRecExp,
    /// QSA 三段的探针 + 候选（`qsa_probe_exp.cu`，只有 microbench 用）。
    pub qsa_probe: QsaProbeExp,
    pub(crate) convert_bf16_f32: CudaFunction,
    pub(crate) convert_bf16_q8_rs: CudaFunction,
    pub(crate) gemv_q4g64_dense_epi: CudaFunction,
    pub(crate) gemv_q4g64_dense4_epi: CudaFunction,
    pub(crate) dequant_q4g64_dense_bf16: CudaFunction,
    pub(crate) causal_conv1d_silu_bf16: CudaFunction,
    pub(crate) causal_conv1d_silu_hist_bf16: CudaFunction,
    pub(crate) gdn_recurrent_bf16: CudaFunction,
    /// ★ GDN 递推 **state 常驻寄存器版**（`gdn_ops.cu`，hk=hv=128 特化，逐位相同）
    pub(crate) gdn_recurrent_reg_bf16: CudaFunction,
    /// GDN 递推多 warp 占用率版（`gdn_ops.cu`，M5 便宜形态，`LINGXI_GDN_MW` 门控）
    pub(crate) gdn_recurrent_mw_bf16: CudaFunction,
    pub(crate) argmax_rows_part_bf16: CudaFunction,
    pub(crate) argmax_rows_fin: CudaFunction,
    pub(crate) topk_rows_part_bf16: CudaFunction,
    /// `topk_rows_part_bf16` 的**寄存器版**（2026-10-24，`LINGXI_NO_TOPKROWS_FAST=1` 回退）。
    /// 逐位等价（同线程映射/同全序/同归并），只把局部插入从 local memory 挪到寄存器。
    pub(crate) topk_rows_part_fast_bf16: CudaFunction,
    pub(crate) route_topk_rows_bf16: CudaFunction,
    /// 旧实现（2026-10-22 版）：`LINGXI_NO_ROUTE_TOPK_FAST=1` 时整核回退用。
    pub(crate) route_topk_rows_bf16_legacy: CudaFunction,
    pub(crate) topk_rows_fin: CudaFunction,
    pub(crate) route_topk_slots_bf16: CudaFunction,
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

