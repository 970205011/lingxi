//! **MTP 专家的小 arena**：把 MTP 的 512 个 routed 专家从 **bf16（5.03 GiB）量化成 Q4G64（1.34 GiB）**
//! 放进设备，作为 [`ExpertSource`] 供 MTP 层**直读**（全部常驻、零 DMA、零 LRU、零驱逐）。
//!
//! 目的（`docs/PLAN.md` §P11.3）：给 GPU0 **腾出 ~3.7 GB** ⇒ 换成更多专家槽
//! （真语料：槽位 7000→~8400，每层 145→175，覆盖率 83.8%→~88%，miss 19.4%→~11.7%）。
//! 这比"把 MTP 搬去 GPU1"省事得多，也没有跨卡把驱动搞崩的风险。
//!
//! **为什么零精度风险**：MTP 只产出**草稿**，每个对外发出的 token 都由**主干验证**决定
//! （草稿错了就拒绝、主干自己给 token）⇒ 改 MTP 的专家权重**不可能改变对外输出**，
//! 只可能改变**接受率**。判据仍然是"首 token + 固定贪心序列不变"。
//!
//! 布局沿用 `q4::quantize_bf16_bits` 的产出（与 `ArenaExpertSource` 同源）：
//! 每个专家一块 `blob = [gu 段 | dn 段]`，段内 `[payload 行 | scale 平面]`。

use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, DevicePtrMut};
use half::bf16;
use lingxi_artifact::Artifact;
use lingxi_kernels::Kernels;
use lingxi_model::moe::{ExpertHits, ExpertSource, Q4Slots};

struct MtpIds {
    dev: CudaSlice<i32>,
    pin: cudarc::driver::PinnedHostSlice<i32>,
}

pub struct MtpArena<'a> {
    k: &'a Kernels,
    /// 全部专家的 Q4 blob 连续排布（专家 `e` 的 blob 在 `e * blob` 处）
    slots: CudaSlice<u8>,
    blob: u64,
    dn_seg: u64,
    gu_scale: u64,
    dn_scale: u64,
    gu_k_pad: i32,
    dn_k_pad: i32,
    n_experts: u32,
    /// `gate_up`/`down` 只是**占位**（Q4 直读路径下调用方不读它们）
    stub_gu: CudaSlice<bf16>,
    stub_dn: CudaSlice<bf16>,
    ids: std::cell::UnsafeCell<MtpIds>,
}

impl<'a> MtpArena<'a> {
    /// 从 MTP 产物里读 `mtp/layer/mlp/routed_gate_up|down`（**裸 bf16**）并量化。
    ///
    /// 一次性开销：2.5 G 权重的主机侧量化（约 10 s）+ 1.34 GB H2D。
    pub fn build(k: &'a Kernels, art: &Artifact) -> Result<Self> {
        let p = "mtp/layer/mlp/";
        let gu_name = format!("{p}routed_gate_up");
        let dn_name = format!("{p}routed_down");
        let shape_of = |n: &str| -> Result<(usize, usize)> {
            let obj = art.find(n).with_context(|| format!("产物缺少 {n}"))?;
            let (fmt, lay, shape) =
                obj.tensor_parts().with_context(|| format!("{n} 不是 tensor"))?;
            anyhow::ensure!(
                fmt == lingxi_ids::TensorFormat::Bf16
                    && lay == lingxi_ids::TensorLayout::ContiguousLeV1,
                "{n} 不是裸 bf16 contiguous（本 arena 依赖该前提，否则要改成 device 侧解量化）"
            );
            anyhow::ensure!(shape.len() == 2, "{n} 需要 rank-2，得到 {shape:?}");
            Ok((shape[0] as usize, shape[1] as usize))
        };
        let (gu_rows, hidden) = shape_of(&gu_name)?;
        let (dn_rows, dn_cols) = shape_of(&dn_name)?;
        // 产物里的实际形状（`lingxi-bench list` 实测）：
        //   `routed_gate_up` = `[E*2I, H]` = [655360, 2560]
        //   `routed_down`    = `[E*H, I]`  = [1310720, 640]
        // ⚠️ 这两个的"列/行"关系是**交叉**的（gu 的列 = H；dn 的行/E = H；dn 的列 = I），
        //    第一版按"gu 的列 == dn 的行"校验 ⇒ 直接报错。
        anyhow::ensure!(
            dn_rows % hidden == 0,
            "dn 行数 {dn_rows} 不是 hidden {hidden} 的倍数"
        );
        let inter = dn_cols;
        let n_experts = (dn_rows / hidden) as u32; // dn 行数 = H * E
        anyhow::ensure!(
            gu_rows == 2 * inter * n_experts as usize,
            "gu 行数 {gu_rows} ≠ 2*inter*E = {}（inter={inter}, E={n_experts}）",
            2 * inter * n_experts as usize
        );

        let gu_pay = art.payload(&gu_name).context("取 gu payload 失败")?;
        let dn_pay = art.payload(&dn_name).context("取 dn payload 失败")?;
        let e_gu = 2 * inter * hidden;
        let e_dn = hidden * inter;
        anyhow::ensure!(
            gu_pay.len() == gu_rows * hidden * 2 && dn_pay.len() == dn_rows * inter * 2,
            "payload 字节数与形状不符"
        );
        // 安全性：mmap 只读视图按 u16（bf16 位形）重解释；产物对齐到 4096 ⇒ 2 字节对齐成立。
        let gu_bits: &[u16] =
            unsafe { std::slice::from_raw_parts(gu_pay.as_ptr() as *const u16, gu_pay.len() / 2) };
        let dn_bits: &[u16] =
            unsafe { std::slice::from_raw_parts(dn_pay.as_ptr() as *const u16, dn_pay.len() / 2) };

        let t0 = std::time::Instant::now();
        // 先量化第 0 个专家，拿到段长与偏移（所有专家一致）
        let (g0, gu_k_pad, gu_scale) =
            lingxi_model::q4::quantize_bf16_bits(&gu_bits[..e_gu], 2 * inter, hidden)?;
        let (d0, dn_k_pad, dn_scale) =
            lingxi_model::q4::quantize_bf16_bits(&dn_bits[..e_dn], hidden, inter)?;
        let gu_bytes = g0.len() as u64;
        let dn_bytes = d0.len() as u64;
        let blob = gu_bytes + dn_bytes;
        let total = (blob * n_experts as u64) as usize;
        let mut host = Vec::<u8>::with_capacity(total);
        host.extend_from_slice(&g0);
        host.extend_from_slice(&d0);
        for e in 1..n_experts as usize {
            let (g, _, _) = lingxi_model::q4::quantize_bf16_bits(
                &gu_bits[e * e_gu..(e + 1) * e_gu],
                2 * inter,
                hidden,
            )?;
            let (d, _, _) =
                lingxi_model::q4::quantize_bf16_bits(&dn_bits[e * e_dn..(e + 1) * e_dn], hidden, inter)?;
            anyhow::ensure!(
                g.len() as u64 == gu_bytes && d.len() as u64 == dn_bytes,
                "专家 {e} 段长与第 0 个不一致（布局假设被打破）"
            );
            host.extend_from_slice(&g);
            host.extend_from_slice(&d);
        }
        anyhow::ensure!(host.len() == total, "拼装长度不符 {} ≠ {total}", host.len());
        let slots = k.stream.clone_htod(&host)?;
        drop(host);
        eprintln!(
            "[mtp-arena] {} 个 MTP 专家量化完成：bf16 {} MiB → Q4G64 {} MiB（**省 {:.2} GiB 显存**），用时 {:.1}s",
            n_experts,
            (gu_pay.len() + dn_pay.len()) / 1048576,
            total / 1048576,
            (gu_pay.len() + dn_pay.len()) as f64 / 1073741824.0 - total as f64 / 1073741824.0,
            t0.elapsed().as_secs_f64()
        );

        let ids = MtpIds {
            dev: k.stream.alloc_zeros::<i32>(n_experts as usize)?,
            pin: unsafe {
                k.stream.context().alloc_pinned_with_flags::<i32>(n_experts as usize, 0)?
            },
        };
        Ok(Self {
            k,
            slots,
            blob,
            dn_seg: gu_bytes,
            gu_scale,
            dn_scale,
            gu_k_pad,
            dn_k_pad,
            n_experts,
            stub_gu: unsafe { k.stream.alloc::<bf16>(1) }?,
            stub_dn: unsafe { k.stream.alloc::<bf16>(1) }?,
            ids: std::cell::UnsafeCell::new(ids),
        })
    }

}

// 安全前提与 `ArenaExpertSource` 相同：单序列主线程逐层调用，返回的借用在下一次调用前用完。
unsafe impl Sync for MtpArena<'_> {}

impl ExpertSource for MtpArena<'_> {
    fn load_hits<'b>(
        &'b self,
        _layer: u32,
        experts: &[u32],
        _inter: usize,
        _hidden: usize,
        want_q4: bool,
        _overlap: bool,
    ) -> Result<ExpertHits<'b>> {
        anyhow::ensure!(want_q4, "MtpArena 只支持 Q4 直读路径（不提供 bf16 解量化副本）");
        anyhow::ensure!(!experts.is_empty(), "MtpArena 收到空列表");
        // `slot_ids` 按**请求顺序**（`rows==1` 时是 (token,专家) 对；`rows>1` 时是唯一专家）
        // —— 两种情况下都等于"专家下标本身"，因为全部常驻、槽号 == 专家号。
        {
            let ids = unsafe { &mut *self.ids.get() };
            anyhow::ensure!(experts.len() <= ids.pin.len(), "请求数超过专家总数");
            let buf = ids.pin.as_mut_slice()?;
            for (j, &e) in experts.iter().enumerate() {
                anyhow::ensure!(e < self.n_experts, "专家下标 {e} 越界");
                buf[j] = e as i32;
            }
            let (dp, _g) = ids.dev.device_ptr_mut(&self.k.stream);
            let src = ids.pin.as_slice()?;
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    dp,
                    &src[..experts.len()],
                    self.k.stream.cu_stream(),
                )?;
            }
        }
        let ids = unsafe { &*self.ids.get() };
        Ok(ExpertHits {
            gate_up: &self.stub_gu,
            down: &self.stub_dn,
            q4: Some(Q4Slots {
                slots: &self.slots,
                slot_ids: &ids.dev,
                blob: self.blob,
                gu_seg: 0,
                dn_seg: self.dn_seg,
                gu_scale: self.gu_scale,
                dn_scale: self.dn_scale,
                gu_k_pad: self.gu_k_pad,
                dn_k_pad: self.dn_k_pad,
                // MTP arena 固定 Q4G64F16S。
                bits: 4,
                // MTP 的 512 个专家**全常驻**（没有 miss、没有 H2D）⇒ 没有可重叠的东西。
                overlap: None,
            }),
            q4_alt: None,
            bits_sel: Vec::new(),
        })
    }
}
