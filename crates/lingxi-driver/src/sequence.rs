//! `Sequence`：单序列状态 + prefill/decode 前向编排（impl Sequence 全部方法）。
//! CODE_MAP Phase 4 刀P4-2 从 lib.rs 搬入，只搬家不改行为。

use anyhow::{bail, Result};
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;
use lingxi_model::layer::{AttnState, AttnWeights, LayerEnv, LayerShape, LayerStates};
use lingxi_model::trunk::LoadedLayer;

use crate::{
    next_fixed, d2h_f32, d2h_u16, h2d_f32, h2d_u16,
    Engine, HostCheckpoint, LayerCache, LayerProvider, QsaHostKv, SpecBuffers,
    StateCheckpoint,
};

/// 一条序列的全部可变状态：token 历史 + 每层的跨 token 状态。
///
/// 这是系统里**唯一**的跨 token 可变状态——残差流本身不带入下一步
/// （每步从新 token 的 embedding 重新进 layer 0），跨 token 的信息全在各层 state 里：
/// QSA 的 KV/indexer cache、GDN 的卷积历史 + S、PLE 的膨胀卷积历史。

pub struct Sequence {
    /// 全量已确认 token id（含 prompt 与已生成 token）
    pub ids: Vec<i64>,
    states: Vec<LayerStates>,
    /// **EP**（`LINGXI_EP=1`）：GPU1 上的第二套状态，覆盖**全部** n_layers 层。
    /// 两卡每层跑同一份稠密段 + 各自的半边专家，交换部分和后 hyper 逐位相同 ⇒
    /// 这里的状态与 `states` 始终逐位相同（由 `spec_snapshot`/`spec_commit` 同步维护）。
    states_ep: Option<Vec<LayerStates>>,
    hidden: usize,
    dim: usize,
}

/// `LINGXI_GDN_DUMP_S=1`：提交完一层后，把该层 GDN 的 `s_all`/`k_hist` **头两三个值**打到 stderr。
///
/// 用途：定位"快速重放 vs 老重放"那 1 ulp 到底**从第几轮、哪一层**开始分叉。
/// 只读几个元素（每次一层 8~12 字节），对时间几乎无影响；格式固定，便于两条路的日志直接 diff。
fn dump_gdn_state(
    k: &Kernels,
    layer: usize,
    g_s_all: &CudaSlice<f32>,
    g_k_hist: &CudaSlice<bf16>,
    tag: &str,
) -> Result<()> {
    if !lingxi_model::flag("LINGXI_GDN_DUMP_S") {
        return Ok(());
    }
    use cudarc::driver::DevicePtr;
    let n = 3usize.min(g_s_all.len());
    let mut s = [0f32; 3];
    {
        let (p, _gd) = g_s_all.device_ptr(&k.stream);
        for (j, v) in s.iter_mut().take(n).enumerate() {
            let mut one = [0f32; 1];
            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync::<f32>(&mut one, p + (j * 4) as u64)?;
            }
            *v = one[0];
        }
    }
    let mut hst = [0u16; 3];
    let hn = 3usize.min(g_k_hist.len());
    {
        let (p, _gd) = g_k_hist.device_ptr(&k.stream);
        for (j, v) in hst.iter_mut().take(hn).enumerate() {
            // `bf16` 不是 cudarc 的 `DeviceRepr` ⇒ 按 u16 读同一个地址（位模式一致）
            let mut one = [0u16; 1];
            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync::<u16>(&mut one, p + (j * 2) as u64)?;
            }
            *v = one[0];
        }
    }
    eprintln!(
        "[s:{tag}] 层{layer:2} s=[{:.9e} {:.9e} {:.9e}] kh=[{:#06x} {:#06x} {:#06x}]",
        s[0], s[1], s[2], hst[0], hst[1], hst[2]
    );
    Ok(())
}

/// `LINGXI_GDN_DUMP_FULL=<dir>`：把**层 0 的整个 `s_all`**（约 2 MB f32）按 `tag` 写成
/// `<dir>/<tag>.bin`（**只留第 0 轮**：文件已存在就跳过）。
///
/// 为什么要按 tag 存盘而不是只打头几个值：
/// 头 3 个值显示"老路提交后 == 快照恢复后"（9 位有效数字全同），但光看 3 个数分不清
/// **(a) 老路的重放真的没推进 `s_all`** 还是 **(b) 那 3 个元素恰好不动**。
/// 存全量再逐字节 diff 一次就能定性。
fn dump_gdn_full(
    k: &Kernels,
    layer: usize,
    s_all: &CudaSlice<f32>,
    tag: &str,
) -> Result<()> {
    let dir = match std::env::var("LINGXI_GDN_DUMP_FULL") {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    if layer != 0 {
        return Ok(());
    }
    let path = std::path::Path::new(&dir).join(format!("{tag}.bin"));
    if path.exists() {
        return Ok(()); // 只留第 0 轮
    }
    let v: Vec<f32> = k.stream.clone_dtoh(s_all)?;
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in &v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(&path, &bytes)?;
    eprintln!("[full] tag={tag} 层{layer} {} 个 f32 → {}", v.len(), path.display());
    Ok(())
}

/// 同 [`dump_gdn_state`]，但只打一个 **bf16** 缓冲的头 3 个值（用来 dump 录像本身）。
fn dump_bf16_head(
    k: &Kernels,
    layer: usize,
    t: &CudaSlice<bf16>,
    tag: &str,
) -> Result<()> {
    if !lingxi_model::flag("LINGXI_GDN_DUMP_S") {
        return Ok(());
    }
    use cudarc::driver::DevicePtr;
    let (p, _gd) = t.device_ptr(&k.stream);
    let mut hst = [0u16; 3];
    for (j, v) in hst.iter_mut().enumerate() {
        let mut one = [0u16; 1];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync::<u16>(&mut one, p + (j * 2) as u64)?;
        }
        *v = one[0];
    }
    eprintln!(
        "[{tag}] 层{layer:2} head=[{:#06x} {:#06x} {:#06x}]",
        hst[0], hst[1], hst[2]
    );
    Ok(())
}

/// 组装某层的 **GDN 录像槽**（`replay=false`：录）。
///
/// 为什么不用"每次现建"以外的做法：`GdnRec` 要同时持有 4 个 `&mut` 缓冲，
/// 只能在同一个函数里从 `SpecBuffers` 的**不相交字段**借出来。
fn build_gdn_rec<'a>(
    sb: Option<&'a mut SpecBuffers>,
    li: usize,
    off: usize,
) -> Option<lingxi_model::gdn::GdnRec<'a>> {
    // ⚠️ **只有真要走快速重放时才录**：录制本身是 4 次 memcpy/层 × 36 层/轮，
    // 不开 `LINGXI_GDN_REPLAY_FAST` 时它纯属白付（实测默认会因此从 51.3 掉到 45.3 tok/s）。
    // ⚠️ **默认录**（配合默认走快速重放，见 `spec_commit`）；`LINGXI_GDN_REPLAY_LEGACY=1`
    // 才回退到"不录 + 重算"的老路。录制本身是 4 次 memcpy/层 × 36 层/轮（约 5 MB/轮）。
    if lingxi_model::flag("LINGXI_GDN_REPLAY_LEGACY") {
        return None;
    }
    let sb = sb?;
    let (qkv, z, a, b) = (
        sb.gdn_qkv[li].as_mut()?,
        sb.gdn_z[li].as_mut()?,
        sb.gdn_a[li].as_mut()?,
        sb.gdn_b[li].as_mut()?,
    );
    Some(lingxi_model::gdn::GdnRec { qkv, z, a, b, off, replay: false })
}

/// **按层区间切片的录像视图**：顺序路径给**全表**视图（`base = 0`），
/// 行为与原来的 `Option<&mut SpecBuffers>` 逐字等价（端到端 81.3 tok/s、md5 一致）。
/// 之所以留成"可切片"的形状：两线程分卡那条路要两边各持 `split_at_mut` 的一半、
/// 互不重叠才可能并发 —— 但那条路实测**零重叠**（A）已删除，现在只有全表视图。
pub(crate) struct GdnRecView<'a> {
    pub gdn_in: &'a mut [Option<CudaSlice<bf16>>],
    pub gdn_qkv: &'a mut [Option<CudaSlice<bf16>>],
    pub gdn_z: &'a mut [Option<CudaSlice<bf16>>],
    pub gdn_a: &'a mut [Option<CudaSlice<bf16>>],
    pub gdn_b: &'a mut [Option<CudaSlice<bf16>>],
    /// 这些切片的 `[0]` 对应全局第几层
    pub base: usize,
}

impl SpecBuffers {
    /// 全表视图（顺序路径）。
    pub(crate) fn rec_view(&mut self) -> GdnRecView<'_> {
        GdnRecView {
            gdn_in: &mut self.gdn_in[..],
            gdn_qkv: &mut self.gdn_qkv[..],
            gdn_z: &mut self.gdn_z[..],
            gdn_a: &mut self.gdn_a[..],
            gdn_b: &mut self.gdn_b[..],
            base: 0,
        }
    }
}

/// 同 [`build_gdn_rec`]，但从**按层切片**的视图借（`li - view.base` 定位）。
fn build_gdn_rec_view<'a>(
    v: Option<&'a mut GdnRecView<'_>>,
    li: usize,
    off: usize,
) -> Option<lingxi_model::gdn::GdnRec<'a>> {
    if lingxi_model::flag("LINGXI_GDN_REPLAY_LEGACY") {
        return None;
    }
    let v = v?;
    let i = li.checked_sub(v.base)?;
    let (qkv, z, a, b) = (
        v.gdn_qkv.get_mut(i)?.as_mut()?,
        v.gdn_z.get_mut(i)?.as_mut()?,
        v.gdn_a.get_mut(i)?.as_mut()?,
        v.gdn_b.get_mut(i)?.as_mut()?,
    );
    Some(lingxi_model::gdn::GdnRec { qkv, z, a, b, off, replay: false })
}

/// 组装某层的 **GDN 重放槽**（`replay=true`：从录像读、跳过投影）。只在提交时用。
fn build_gdn_replay_rec<'a>(
    sb: &'a mut SpecBuffers,
    li: usize,
) -> Option<lingxi_model::gdn::GdnRec<'a>> {
    let (qkv, z, a, b) = (
        sb.gdn_qkv[li].as_mut()?,
        sb.gdn_z[li].as_mut()?,
        sb.gdn_a[li].as_mut()?,
        sb.gdn_b[li].as_mut()?,
    );
    Some(lingxi_model::gdn::GdnRec { qkv, z, a, b, off: 0, replay: true })
}

/// 同 [`build_gdn_rec`]，但从**分离的四个槽向量**借（EP 下同一 `SpecBuffers` 里
/// 主卡/第二卡各有一套录像槽，必须按字段分别借才不打架）。槽缺失 ⇒ `None`（自动退回重算）。
fn build_gdn_rec_fields<'a>(
    qkv_v: &'a mut Vec<Option<CudaSlice<bf16>>>,
    z_v: &'a mut Vec<Option<CudaSlice<bf16>>>,
    a_v: &'a mut Vec<Option<CudaSlice<bf16>>>,
    b_v: &'a mut Vec<Option<CudaSlice<bf16>>>,
    li: usize,
    off: usize,
) -> Option<lingxi_model::gdn::GdnRec<'a>> {
    if lingxi_model::flag("LINGXI_GDN_REPLAY_LEGACY") {
        return None;
    }
    let (qkv, z, a, b) = (
        qkv_v.get_mut(li)?.as_mut()?,
        z_v.get_mut(li)?.as_mut()?,
        a_v.get_mut(li)?.as_mut()?,
        b_v.get_mut(li)?.as_mut()?,
    );
    Some(lingxi_model::gdn::GdnRec { qkv, z, a, b, off, replay: false })
}

/// 同 [`build_gdn_replay_rec`]，从分离槽向量借（EP 的第二卡重放）。
fn build_gdn_replay_rec_fields<'a>(
    qkv_v: &'a mut Vec<Option<CudaSlice<bf16>>>,
    z_v: &'a mut Vec<Option<CudaSlice<bf16>>>,
    a_v: &'a mut Vec<Option<CudaSlice<bf16>>>,
    b_v: &'a mut Vec<Option<CudaSlice<bf16>>>,
    li: usize,
) -> Option<lingxi_model::gdn::GdnRec<'a>> {
    let (qkv, z, a, b) = (
        qkv_v.get_mut(li)?.as_mut()?,
        z_v.get_mut(li)?.as_mut()?,
        a_v.get_mut(li)?.as_mut()?,
        b_v.get_mut(li)?.as_mut()?,
    );
    Some(lingxi_model::gdn::GdnRec { qkv, z, a, b, off: 0, replay: true })
}

impl Sequence {
    /// 建空序列。`cap` = QSA KV cache 容量（≥ prompt + 生成长度）；
    /// `ple_layer = Some((层号, conv_kernel, dilation))` 时该层带 PLE 状态（本模型 = layer 1）。
    pub fn new(
        k: &Kernels,
        n_layers: u32,
        shape: &LayerShape,
        cap: usize,
        ple_layer: Option<(u32, usize, usize)>,
    ) -> Result<Self> {
        Self::new_split(k, None, n_layers, shape, cap, ple_layer)
    }

    /// 同上，但 `[lo, n_layers)` 层的**状态建在第二张卡上**。
    ///
    /// 为什么必须在这里分设备：整个 `Sequence` 的跨 token 可变状态（QSA 的 KV/indexer cache、
    /// GDN 的 S + 卷积历史、PLE 历史）都挂在 `states[i]` 上，而 `layer_step` 只认自己那张卡的
    /// 句柄 ⇒ 状态一旦建错卡，第一次 kernel 就是 `ILLEGAL_ADDRESS`（或更糟：静默写串）。
    pub fn new_split(
        k0: &Kernels,
        k1: Option<(&Kernels, u32)>,
        n_layers: u32,
        shape: &LayerShape,
        cap: usize,
        ple_layer: Option<(u32, usize, usize)>,
    ) -> Result<Self> {
        let mut states = Vec::with_capacity(n_layers as usize);
        for i in 0..n_layers {
            let k = match k1 {
                Some((kk, lo)) if i >= lo => kk,
                _ => k0,
            };
            let kind = lingxi_model::layer_kind(i);
            let ple = ple_layer.and_then(|(idx, kk, d)| if idx == i { Some((kk, d)) } else { None });
            states.push(LayerStates::for_kind(k, kind, shape, cap, ple)?);
        }
        Ok(Self {
            ids: Vec::new(),
            states,
            states_ep: None,
            hidden: shape.hidden,
            dim: shape.hc * shape.hidden,
        })
    }

    /// **EP 构造器**（`LINGXI_EP=1`）：主卡状态 `states` 与第二张卡的 `states_ep` 都覆盖
    /// **全部** n_layers 层（两卡每层都跑同一份稠密段，只是专家各算半边）。
    pub fn new_ep(
        k0: &Kernels,
        k1: &Kernels,
        n_layers: u32,
        shape: &LayerShape,
        cap: usize,
        ple_layer: Option<(u32, usize, usize)>,
    ) -> Result<Self> {
        anyhow::ensure!(n_layers > 0, "Sequence::new_ep: n_layers=0");
        let build = |k: &Kernels| -> Result<Vec<LayerStates>> {
            let mut v = Vec::with_capacity(n_layers as usize);
            for i in 0..n_layers {
                let kind = lingxi_model::layer_kind(i);
                let ple =
                    ple_layer.and_then(|(idx, kk, d)| if idx == i { Some((kk, d)) } else { None });
                v.push(LayerStates::for_kind(k, kind, shape, cap, ple)?);
            }
            Ok(v)
        };
        let states = build(k0)?;
        let states_ep = build(k1)?;
        Ok(Self {
            ids: Vec::new(),
            states,
            states_ep: Some(states_ep),
            hidden: shape.hidden,
            dim: shape.hc * shape.hidden,
        })
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 逐（批）层推进 `t` 个 token：`x[t, hc*H] → hyper[t, hc*H]`。
    fn run_layers(
        &mut self,
        eng: &Engine,
        x: CudaSlice<bf16>,
        t: usize,
        provider: &mut dyn LayerProvider,
    ) -> Result<CudaSlice<bf16>> {
        self.run_layers_rec(eng, x, t, provider, None, 0, self.ids.len())
    }

    /// **本轮的 chunk 数**（`LINGXI_VERIFY_CHUNKS`，默认 1 = 老行为）。
    /// 上限夹到 `t`（每 chunk 至少 1 行）。见 `spec_chunk` 里的说明。
    fn verify_chunks_of(t: usize) -> usize {
        let n: usize = std::env::var("LINGXI_VERIFY_CHUNKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        n.clamp(1, t.max(1))
    }

    /// 同上，但**录下每个 GDN 层的输入**（投机解码重放用）。
    ///
    /// 为什么需要录：GDN 是循环状态，验证批一次推进 t 个 token 后，部分接受时必须把
    /// `s_all`/卷积历史退回并**重放**被接受前缀；重放要那份 per-layer 输入。
    fn run_layers_rec(
        &mut self,
        eng: &Engine,
        x: CudaSlice<bf16>,
        t: usize,
        provider: &mut dyn LayerProvider,
        rec: Option<&mut SpecBuffers>,
        // **录像的行偏移**（`LINGXI_VERIFY_CHUNKS>1` 时按 chunk 分段录；0 = 老行为）。
        // `gdn_in[i]` 是 `t_max × dim`，所以 `base + t ≤ t_max` 恒成立。
        rec_base: usize,
        // **PLE 能看到的 ids 个数**（`ple_step` 只用**末尾 t 个**：切 chunk 时本 chunk 的
        // 末尾 t 个必须是它自己那几行 ⇒ 传 `fed + b`；不切就是 `self.ids.len`）。
        ids_len: usize,
    ) -> Result<CudaSlice<bf16>> {
        let mut hyper = x;
        // ★ **EP**（`LINGXI_EP=1`）：两卡都跑全部 n_layers 层；每层稠密段各跑一份，
        // 专家按 id 二分，交换 30 KB 部分和后合并（两卡同序 out0 + out1 ⇒ hyper 逐位相同）。
        // EP 与 layer-split 互斥（CLI 已保证 `ls_on = false`），所以这里先分流。
        if eng.ep.is_some() {
            return self.run_layers_ep(eng, hyper, t, provider, rec, rec_base, ids_len);
        }
        // ★ **全量 layer-split**（`LINGXI_LAYER_SPLIT`）：整层按 `k_of(i)` 决定跑哪张卡；
        // 到切分点把残差流换到另一张卡（`cross_copy` 走两卡各自的锁页 staging，位等价）。
        if eng.split_active() && eng.n_layers > eng.split_lo {
            let k1 = eng.k1.expect("split_lo 有效时 k1 必存在");
            let mut h1 = lingxi_model::scratch::bf16(k1, hyper.len())?;
            lingxi_model::scratch::cross_copy(eng.k, &hyper, k1, &mut h1)?;
            lingxi_model::scratch::put_bf16(hyper);
            hyper = h1;
        }
        hyper = self.run_layers_range(
            eng, 0, eng.n_layers, hyper, t, provider, rec, rec_base, ids_len,
        )?;
        // 最后一段在 GPU1 上 ⇒ 把残差流搬回 GPU0（head/mixer 在 GPU0）
        if eng.split_active() && eng.n_layers > eng.split_lo {
            let k1 = eng.k1.expect("split_lo 有效时 k1 必存在");
            let mut h0 = lingxi_model::scratch::bf16(eng.k, hyper.len())?;
            lingxi_model::scratch::cross_copy(k1, &hyper, eng.k, &mut h0)?;
            drop(hyper);
            hyper = h0;
        }
        Ok(hyper)
    }

    /// **EP 层循环**（`LINGXI_EP=1`）：每层
    /// ① 两卡各跑稠密段（同一份输入/权重，逐位相同）；
    /// ② 两卡各跑本卡半边专家（路由全量各算，逐位相同）；
    /// ③ **一趟双向**交换 30 KB 部分和（`scratch::cross_copy_pair`：两路 D2H 并行、各自同步一次、
    /// host 交叉、两路 H2D）；
    /// ④ 两卡各用**同一加法次序** `out0 + out1` 合并，再各跑 MoE 收尾（gr_write）。
    ///
    /// ④ 之后两卡的 `hyper` 逐位相同 ⇒ 状态也逐位相同，下一层不需要再同步任何状态。
    #[allow(clippy::too_many_arguments)]
    fn run_layers_ep(
        &mut self,
        eng: &Engine,
        x: CudaSlice<bf16>,
        t: usize,
        provider: &mut dyn LayerProvider,
        mut rec: Option<&mut SpecBuffers>,
        rec_base: usize,
        ids_len: usize,
    ) -> Result<CudaSlice<bf16>> {
        use lingxi_model::scratch;
        let epx = eng.ep.expect("run_layers_ep 需要 Engine::ep");
        let (k0, k1) = (eng.k, epx.k1);
        let dim = self.dim;
        let n = eng.n_layers;
        let ids = &self.ids[..ids_len.min(self.ids.len())];
        let states = &mut self.states[..];
        let states_ep = self.states_ep.as_mut().expect("EP 模式必须有 states_ep");
        anyhow::ensure!(
            states.len() == n as usize && states_ep.len() == n as usize,
            "EP：states 层数不匹配"
        );
        let prof = lingxi_model::flag("LINGXI_PROF");
        let mut hyper0 = x;
        let mut hyper1 = scratch::bf16(k1, hyper0.len())?;
        // 只有第一层需要一次单向搬运把残差流初始化到 GPU1；之后两卡 hyper 逐位相同。
        scratch::cross_copy(k0, &hyper0, k1, &mut hyper1)?;
        let mut t_prev = std::time::Instant::now();
        for i in 0..n {
            let li = i as usize;
            // 本层输入（也是上一层残差流）：先移出，层末再归还 scratch 池
            let old0 = hyper0;
            let old1 = hyper1;
            let loaded0: &LoadedLayer = provider.layer(i)?;
            let loaded1: &LoadedLayer = epx
                .layers1
                .peek(i)
                .ok_or_else(|| anyhow::anyhow!("[ep] GPU1 层 {i} 未预热，peek 失败"))?;
            let t_prov = t_prev.elapsed().as_secs_f64();
            // 两卡各录一份本层输入（GPU1 的快速重放用自己那份，避免逐轮跨卡搬输入状态）
            if let Some(sb) = rec.as_deref_mut() {
                let span = rec_base * dim..(rec_base + t) * dim;
                if let Some(dst) = sb.gdn_in[li].as_mut() {
                    k0.stream.memcpy_dtod(&old0, &mut dst.slice_mut(span.clone()))?;
                }
                if let Some(dst) = sb.ep_side.as_mut().and_then(|s| s.gdn_in[li].as_mut()) {
                    k1.stream.memcpy_dtod(&old1, &mut dst.slice_mut(span))?;
                }
            }
            let env0 = LayerEnv {
                cos: &eng.tables.cos,
                sin: &eng.tables.sin,
                cos_host: &eng.tables.cos_host,
                sin_host: &eng.tables.sin_host,
                remote: None,
                causal_host: &[],
                k_layernorm_host: &loaded0.k_layernorm_host,
                ids,
                experts: eng.experts,
            };
            let env1 = LayerEnv {
                cos: &epx.tables1.cos,
                sin: &epx.tables1.sin,
                cos_host: &epx.tables1.cos_host,
                sin_host: &epx.tables1.sin_host,
                remote: None,
                causal_host: &[],
                k_layernorm_host: &loaded1.k_layernorm_host,
                ids,
                experts: Some(epx.experts1),
            };
            // 录像槽（两卡各一套）——只能从 `SpecBuffers` 的**不相交字段**同时借
            let (rec0, rec1) = match rec.as_deref_mut() {
                Some(sb) => {
                    let r0 = build_gdn_rec_fields(
                        &mut sb.gdn_qkv,
                        &mut sb.gdn_z,
                        &mut sb.gdn_a,
                        &mut sb.gdn_b,
                        li,
                        rec_base,
                    );
                    let r1 = match sb.ep_side.as_mut() {
                        Some(side) => build_gdn_rec_fields(
                            &mut side.gdn_qkv,
                            &mut side.gdn_z,
                            &mut side.gdn_a,
                            &mut side.gdn_b,
                            li,
                            rec_base,
                        ),
                        None => None,
                    };
                    (r0, r1)
                }
                None => (None, None),
            };
            let ts = std::time::Instant::now();
            let d0 = lingxi_model::layer::layer_dense_step(
                k0,
                &loaded0.weights,
                eng.shape,
                &old0,
                t,
                &env0,
                &mut states[li],
                i,
                rec0,
            )?;
            let d1 = lingxi_model::layer::layer_dense_step(
                k1,
                &loaded1.weights,
                eng.shape,
                &old1,
                t,
                &env1,
                &mut states_ep[li],
                i,
                rec1,
            )?;
            // ② 两卡各跑本卡那半专家。分配表**两卡同源**（`epx.assign(i)` 同一份切片），
            // 过滤在 `sparse_moe_forward` 里按 `assign[e] == card` 做。
            let assign = epx.assign(i);
            let m0 = lingxi_model::layer::layer_moe_partial(
                k0,
                &loaded0.weights,
                eng.shape,
                &d0.mixed_m,
                t,
                &env0,
                i,
                Some(lingxi_model::moe::EpSlice { assign, card: 0 }),
            )?;
            let m1 = lingxi_model::layer::layer_moe_partial(
                k1,
                &loaded1.weights,
                eng.shape,
                &d1.mixed_m,
                t,
                &env1,
                i,
                Some(lingxi_model::moe::EpSlice { assign, card: 1 }),
            )?;
            // ③ 一趟双向交换（事件只等本层两卡 MoE 的 D2H，不排空整条流；staging 双缓冲
            // 按层奇偶交替，避免上一层的 H2D 还在读就被下一层 host 覆盖）
            let mut mirror0 = scratch::bf16(k0, t * dim)?; // GPU0 上的 out1
            let mut mirror1 = scratch::bf16(k1, t * dim)?; // GPU1 上的 out0
            scratch::cross_copy_pair(
                k0,
                &m0.out,
                &mut mirror0,
                k1,
                &m1.out,
                &mut mirror1,
                (i & 1) as usize,
            )?;
            // ④ 同一加法次序：两卡都是 `out0 + out1`
            let mut merged0 = scratch::bf16(k0, t * dim)?;
            k0.add_bf16(&m0.out, &mirror0, &mut merged0, t * dim)?;
            let mut merged1 = scratch::bf16(k1, t * dim)?;
            k1.add_bf16(&mirror1, &m1.out, &mut merged1, t * dim)?;
            scratch::put_bf16(m0.out);
            scratch::put_bf16(m1.out);
            scratch::put_bf16(mirror0);
            scratch::put_bf16(mirror1);
            hyper0 = lingxi_model::layer::layer_moe_apply(
                k0,
                eng.shape,
                d0.hyper,
                merged0,
                d0.mixed_m,
                d0.injw_m,
                t,
                i,
            )?;
            hyper1 = lingxi_model::layer::layer_moe_apply(
                k1,
                eng.shape,
                d1.hyper,
                merged1,
                d1.mixed_m,
                d1.injw_m,
                t,
                i,
            )?;
            // 上一层残差流已被本层 kernel 读完（同流有序）→ 归还池
            scratch::put_bf16(old0);
            scratch::put_bf16(old1);
            if prof {
                eprintln!(
                    "  [layer] {i:2}/{n} t={t} ep 供层 {t_prov:.3}s 计算 {:.3}s",
                    ts.elapsed().as_secs_f64()
                );
            }
            if lingxi_model::flag("LINGXI_LAYER_PROF") {
                eprintln!(
                    "  [layerprof] 层{i:2} ep wall {:7.1} µs (供层 {:5.1})",
                    t_prev.elapsed().as_secs_f64() * 1e6,
                    t_prov * 1e6
                );
            }
            t_prev = std::time::Instant::now();
        }
        Ok(hyper0)
    }

    /// 跑层区间 `[i0, i1)` 的前向（`run_layers_rec` 的循环体；切 chunk 流水时按半段调用）。
    /// 不做任何跨卡搬运——调用方负责把 `hyper` 放到 `i0` 所在设备的缓冲里，并从 `i1-1`
    /// 所在设备取回结果。
    #[allow(clippy::too_many_arguments)]
    fn run_layers_range(
        &mut self,
        eng: &Engine,
        i0: u32,
        i1: u32,
        hyper: CudaSlice<bf16>,
        t: usize,
        provider: &mut dyn LayerProvider,
        rec: Option<&mut SpecBuffers>,
        rec_base: usize,
        ids_len: usize,
    ) -> Result<CudaSlice<bf16>> {
        let (states, ids, dim) = (&mut self.states[..], &self.ids[..], self.dim);
        let mut recv = rec.map(|sb| sb.rec_view());
        Self::run_layers_range_parts(
            states, 0, dim, ids, eng, i0, i1, hyper, t, Some(provider), None, recv.as_mut(),
            rec_base, ids_len,
        )
    }

#[allow(clippy::too_many_arguments)]
/// 自由函数版：**不借 `&mut Sequence`**，只借本卡负责的那段 `states`（`states[li - base]`）
/// 与只读 `ids`。这是"按卡分线程"（㊿-①）的第一步：GPU0/GPU1 各持一段 `states`
/// （`split_at_mut(lo)`），互不重叠。
fn run_layers_range_parts(
    states: &mut [LayerStates],
    base: usize,
    dim: usize,
    ids: &[i64],
    eng: &Engine,
    i0: u32,
    i1: u32,
    mut hyper: CudaSlice<bf16>,
    t: usize,
    mut provider: Option<&mut dyn LayerProvider>,
    cache: Option<&dyn LayerCache>,
    mut rec: Option<&mut GdnRecView<'_>>,
    rec_base: usize,
    ids_len: usize,
) -> Result<CudaSlice<bf16>> {
    let prof = lingxi_model::flag("LINGXI_PROF");
    for i in i0..i1 {
        let k = eng.k_of(i);
        let tab = eng.tables_of(i);
        let tl = std::time::Instant::now();
        let loaded: &LoadedLayer = match cache {
            Some(c) => c
                .peek(i)
                .ok_or_else(|| anyhow::anyhow!("[pipe-threads] 层 {i} 未预热，peek 失败"))?,
            None => provider
                .as_deref_mut()
                .ok_or_else(|| anyhow::anyhow!("run_layers_range_parts: 无 cache 也无 provider"))?
                .layer(i)?,
        };
        let t_prov = tl.elapsed().as_secs_f64();
        // 录 GDN 层的输入（`hyper` 就是本层输入；必须在**本层所在设备**上录）
        if let Some(r) = rec.as_deref_mut() {
            let rb = r.base;
            if let Some(dst) = (i as usize).checked_sub(rb).and_then(|j| r.gdn_in.get_mut(j))
                .and_then(|o| o.as_mut())
            {
                k.stream.memcpy_dtod(
                    &hyper,
                    &mut dst.slice_mut(rec_base * dim..(rec_base + t) * dim),
                )?;
            }
        }
        let env = LayerEnv {
            cos: &tab.cos,
            sin: &tab.sin,
            cos_host: &tab.cos_host,
            sin_host: &tab.sin_host,
            remote: eng.remote,
            causal_host: &[], // 增量路径按绝对位置自行推出因果集，不用外部掩码
            k_layernorm_host: &loaded.k_layernorm_host,
            ids: &ids[..ids_len.min(ids.len())],
            experts: eng.experts_of(i),
        };
            let ts = std::time::Instant::now();
            // GPU1 的层仍走"禁池"路径（2026-10-18 复测：开池 126.9s vs 禁池 123.4s，
            // 开池没有收益 ⇒ 保持原样，减少风险）。GPU0/主线程的池化照常。
            let on1 = eng.split_active() && i >= eng.split_lo;
            // ★ 投机录像（`gdn_step` 的内部输入，见 `SpecBuffers::gdn_qkv` 与 `gdn::GdnRec`）：
            // 录下来后，提交时只跑「卷积 + 递推」两个小核、跳过投影的权重读。
            // `rec` 的可变借用在每个分支里现取（两个分支各取一次，借用不重叠）。
            let li = i as usize;
            let next = if on1 {
                // ★ 2026-10-26 试验（`LINGXI_ON1_POOL=1`）：GPU1 层走池化。动机：t=512 下
                // 禁池 = 每层每 chunk 数块 10MB 级 cudaFreeAsync→cuMemAllocAsync 的 VMM
                // 映射回收/重映射——prefill 期间实测 minflt 6.4 万次/s（sys CPU 87%）。
                // 2026-10-18 的"开池更慢"是在 33K/7200 槽的旧工况测的，值得在 host 账立案
                // 后复测。`drop(hyper)` 保留：on1 分支的 hyper 上一 chunk 起就不在池里。
                let n1 = if lingxi_model::flag("LINGXI_ON1_POOL") {
                    let r = lingxi_model::layer::layer_step_rec(
                        k,
                        &loaded.weights,
                        eng.shape,
                        &hyper,
                        t,
                        &env,
                        &mut states[li - base],
                        i,
                        build_gdn_rec_view(rec.as_deref_mut(), li, rec_base),
                    );
                    lingxi_model::scratch::put_bf16(hyper);
                    r?
                } else {
                    let r = lingxi_model::scratch::without_pool(|| {
                        lingxi_model::layer::layer_step_rec(
                            k,
                            &loaded.weights,
                            eng.shape,
                            &hyper,
                            t,
                            &env,
                            &mut states[li - base],
                            i,
                            build_gdn_rec_view(rec.as_deref_mut(), li, rec_base),
                        )
                    });
                    drop(hyper); // 禁池路径：put 是 no-op，必须显式释放
                    r?
                };
                n1
            } else {
                let n0 = lingxi_model::layer::layer_step_rec(
                    k,
                    &loaded.weights,
                    eng.shape,
                    &hyper,
                    t,
                    &env,
                    &mut states[li - base],
                    i,
                    build_gdn_rec_view(rec.as_deref_mut(), li, rec_base),
                )?;
                // 上一步的残差流已被本步 enqueue 的 kernel 读完 → 归还 scratch 池复用
                lingxi_model::scratch::put_bf16(hyper);
                n0
            };
            if prof {
                eprintln!(
                    "  [layer] {i:2}/{n} t={t} 供层 {t_prov:.3}s 计算 {:.3}s",
                    ts.elapsed().as_secs_f64(),
                    n = eng.n_layers
                );
            }
            // ㊸：`LINGXI_LAYER_PROF=1` 每层的 **host wall**（含内部所有同步的等待）。
            // GPU busy vs wall 的差就是它——定位"GPU 闲置 20ms/token"挂在哪些层。
            if lingxi_model::flag("LINGXI_LAYER_PROF") {
                eprintln!(
                    "  [layerprof] 层{i:2} wall {:7.1} µs (供层 {:5.1})",
                    tl.elapsed().as_secs_f64() * 1e6,
                    t_prov * 1e6
                );
            }
            hyper = next;
            // [探针] 逐层数值指纹（末行求和 + NaN 数），用于**线程版 vs 顺序版**逐层对比。
            // 只在 t ≤ 1024 时开（整块回读 10~20 MB/层，纯诊断）。
            if t <= 1024 && lingxi_model::flag("LINGXI_PIPE_DUMP") {
                if let Ok(h) = k.stream.clone_dtoh(&hyper) {
                    let (mut nan, mut mx, mut sum) = (0usize, 0f32, 0f64);
                    let mut rows: Vec<usize> = Vec::new();
                    for (idx, v) in h.iter().enumerate() {
                        let f = v.to_f32();
                        if f.is_nan() {
                            nan += 1;
                            if rows.len() < 40 && rows.last() != Some(&(idx / dim)) {
                                rows.push(idx / dim);
                            }
                        } else {
                            mx = mx.max(f.abs());
                            sum += f as f64;
                        }
                    }
                    if nan > 0 {
                        eprintln!(
                            "[pipe-dump] 层{i:2} t={t} nan={nan}/{} maxabs={mx:.4} 行={rows:?}",
                            h.len()
                        );
                    } else {
                        eprintln!("[pipe-dump] 层{i:2} t={t} 全和={sum:.4} maxabs={mx:.4}");
                    }
                }
            }
        }
        Ok(hyper)
    }

/// layer-major prefill 的层内循环：对 `i ∈ [i0,i1)`，把 `buf`（`[n, dim]`）按 `sub`
/// 行一段、**原地**逐段过层。
///
/// 与 [`run_layers_range_parts`] 用**同一套** `layer_step` / `LayerEnv`，唯一区别是
/// 外层是层、内层是行段 ⇒ 每个 (层, 行段) 的算子形状、输入、状态完全一致，
/// 结果与 token-major 逐位相同。`base` 是本段第一行在整段 prompt 中的绝对位置
/// （QSA/PLE 按绝对位置推进）。
#[allow(clippy::too_many_arguments)]
fn run_layers_in_place(
    states: &mut [LayerStates],
    sbase: usize,
    ids: &[i64],
    eng: &Engine,
    i0: u32,
    i1: u32,
    buf: &mut CudaSlice<bf16>,
    n: usize,
    sub: usize,
    base: usize,
    mut provider: Option<&mut dyn LayerProvider>,
    cache: Option<&dyn LayerCache>,
    dim: usize,
) -> Result<()> {
    for i in i0..i1 {
        let k = eng.k_of(i);
        let tab = eng.tables_of(i);
        let loaded: &LoadedLayer = match cache {
            Some(c) => c
                .peek(i)
                .ok_or_else(|| anyhow::anyhow!("[lm] 层 {i} 未预热，peek 失败"))?,
            None => provider
                .as_deref_mut()
                .ok_or_else(|| anyhow::anyhow!("run_layers_in_place: 无 cache 也无 provider"))?
                .layer(i)?,
        };
        let li = i as usize;
        let on1 = eng.split_active() && i >= eng.split_lo;
        let t_layer = std::time::Instant::now();
        let mut c0 = 0usize;
        while c0 < n {
            let c = sub.min(n - c0);
            let env = LayerEnv {
                cos: &tab.cos,
                sin: &tab.sin,
                cos_host: &tab.cos_host,
                sin_host: &tab.sin_host,
                remote: eng.remote,
                causal_host: &[], // 增量路径按绝对位置自行推出因果集
                k_layernorm_host: &loaded.k_layernorm_host,
                ids: &ids[..(base + c0 + c).min(ids.len())],
                experts: eng.experts_of(i),
            };
            let mut x = if on1 {
                lingxi_model::scratch::without_pool(|| lingxi_model::scratch::bf16(k, c * dim))?
            } else {
                lingxi_model::scratch::bf16(k, c * dim)?
            };
            k.stream
                .memcpy_dtod(&buf.slice(c0 * dim..(c0 + c) * dim), &mut x)?;
            let next = if on1 {
                lingxi_model::scratch::without_pool(|| {
                    lingxi_model::layer::layer_step(
                        k,
                        &loaded.weights,
                        eng.shape,
                        &x,
                        c,
                        &env,
                        &mut states[li - sbase],
                        i,
                    )
                })?
            } else {
                lingxi_model::layer::layer_step(
                    k,
                    &loaded.weights,
                    eng.shape,
                    &x,
                    c,
                    &env,
                    &mut states[li - sbase],
                    i,
                )?
            };
            if on1 {
                drop(x);
            } else {
                lingxi_model::scratch::put_bf16(x);
            }
            k.stream
                .memcpy_dtod(&next, &mut buf.slice_mut(c0 * dim..(c0 + c) * dim))?;
            if on1 {
                drop(next);
            } else {
                lingxi_model::scratch::put_bf16(next);
            }
            c0 += c;
        }
        if lingxi_model::flag("LINGXI_LAYER_PROF") {
            eprintln!(
                "[lm-layer] 层{i:2} n={n} sub={sub} 用时 {:8.1} ms",
                t_layer.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    Ok(())
}

    fn embed(&self, eng: &Engine, ids: &[i64]) -> Result<CudaSlice<bf16>> {
        let k = eng.k;
        let t = ids.len();
        let idx_host: Vec<i32> = ids.iter().map(|&v| v as i32).collect();
        let didx = k.stream.clone_htod(&idx_host)?;
        let mut emb = k.stream.alloc_zeros::<bf16>(t * self.hidden)?;
        k.gather_rows_bf16(&eng.endpoints.token_embedding, &didx, &mut emb, t, self.hidden, 0)?;
        let mut x = k.stream.alloc_zeros::<bf16>(t * self.dim)?;
        k.hc_repeat_bf16(&emb, &mut x, t, self.hidden, eng.shape.hc)?;
        Ok(x)
    }

    /// 末端 mixer + lm_head，返回**最后一行**的贪心 token。
    fn head_argmax(&self, eng: &Engine, hyper: &CudaSlice<bf16>, t: usize) -> Result<i64> {
        let k = eng.k;
        // `LINGXI_EVPROF*`：head 计时段（伪层 = n_layers、段 6）。mixer 也在头内
        //（GR 收尾是主干最后一层的 gr_write，与头之间还有跨段 handoff，全记进 head 段）。
        if let Some(ep) = k.evprof.as_ref() {
            ep.begin(eng.n_layers, lingxi_kernels::evprof::HEAD_SEG, &k.stream);
        }
        let r = self.head_argmax_inner(eng, hyper, t);
        if let Some(ep) = k.evprof.as_ref() {
            ep.end(eng.n_layers, lingxi_kernels::evprof::HEAD_SEG, &k.stream);
        }
        r
    }

    fn head_argmax_inner(&self, eng: &Engine, hyper: &CudaSlice<bf16>, t: usize) -> Result<i64> {
        let k = eng.k;
        let s = eng.shape;
        let (mixed, injw) = lingxi_model::gr::gated_residual_forward(
            k,
            &eng.endpoints.mixer,
            hyper,
            t as u32,
            self.dim,
            self.hidden,
            s.hc,
            s.lowrank,
            s.eps,
        )?;
        // **归还 scratch 池**：忘了归还就会让它被 cudarc 释放掉，池子每 token 少一块，
        // 于是"同一逻辑缓冲"的地址每 token 漂移一格（实测：[ptrprobe] 里逐 token +1）。
        // CUDA Graph 会把地址烘死，所以这个归还动作是 graph 化的前提。
        if let Some(j) = injw {
            lingxi_model::scratch::put_bf16(j);
        }
        let mut logits = k.stream.alloc_zeros::<bf16>(t * eng.vocab)?;
        let t0 = std::time::Instant::now();
        if let Some(wq) = &eng.endpoints.output_head_q4 {
            lingxi_model::gpu::proj_q4_epi(
                k, &mixed, wq, &mut logits, eng.vocab, self.hidden, 0, t, 0, 0.0, 0.0,
            )?;
        } else {
            lingxi_model::gpu::proj_t1(
                k,
                &mixed,
                &eng.endpoints.output_head,
                &mut logits,
                eng.vocab,
                self.hidden,
                t,
            )?;
        }
        let t_gemm = t0.elapsed();
        lingxi_model::scratch::put_bf16(mixed);
        // ★ P15-⑭：argmax 移进 GPU（与 `head_argmax_all` 同一对拍口径，见 `argmax-check`），
        // 只 D2H 4 B。`LINGXI_HOST_ARGMAX=1` 回退老路。
        if !lingxi_model::flag("LINGXI_HOST_ARGMAX") {
            const G: usize = 64;
            let mut pval = lingxi_model::scratch::f32(k, G)?;
            let mut pidx = lingxi_model::scratch::i32(k, G)?;
            let mut am = lingxi_model::scratch::i32(k, 1)?;
            k.argmax_rows_part_bf16(&logits, (t - 1) * eng.vocab, &mut pval, &mut pidx, 1, eng.vocab, G)?;
            k.argmax_rows_fin(&pval, &pidx, &mut am, 1, G)?;
            let am_h: Vec<i32> = k.stream.clone_dtoh(&am)?;
            lingxi_model::scratch::put_f32(pval);
            lingxi_model::scratch::put_i32(pidx);
            lingxi_model::scratch::put_i32(am);
            if lingxi_model::flag("LINGXI_PROF") {
                eprintln!("  [head] lm_head {:.4}s  argmax(GPU) 已取回", t_gemm.as_secs_f64());
            }
            return Ok(am_h[0] as i64);
        }
        let t1 = std::time::Instant::now();
        let lh: Vec<bf16> = lingxi_model::scratch::download_bf16(k, &logits)?;
        let t_d2h = t1.elapsed();
        if lingxi_model::flag("LINGXI_PROF") {
            eprintln!(
                "  [head] lm_head {:.4}s  logits D2H(锁页, {} KB) {:.4}s",
                t_gemm.as_secs_f64(),
                lh.len() * 2 / 1024,
                t_d2h.as_secs_f64()
            );
        }
        let last = &lh[(t - 1) * eng.vocab..t * eng.vocab];
        let mut best = 0usize;
        for (i, v) in last.iter().enumerate() {
            if v.to_f32() > last[best].to_f32() {
                best = i;
            }
        }
        Ok(best as i64)
    }

    /// ★ P15-㉞ 诊断：`LINGXI_DUMP_PREFILL_STATE=1` 时 prefill 后 dump 探针层状态
    /// （GDN s_all / QSA KV 校验和 / hyper 末行）——定位 split+分块的跨卡边界发散。
    /// 只读、只打印，不影响数值。
    fn dump_prefill_state(&self, eng: &Engine, hyper: &CudaSlice<bf16>, tag: &str) {
        if !lingxi_model::flag("LINGXI_DUMP_PREFILL_STATE") {
            return;
        }
        let cks = |h: &[bf16]| -> f64 { h.iter().map(|v| v.to_f32() as f64).sum() };
        let cks_bits = |h: &[u16]| -> f64 {
            h.iter().map(|&b| bf16::from_bits(b).to_f32() as f64).sum()
        };
        let qkv_bits = |len: usize, cap: usize| -> usize { len / cap };
        for li in [0usize, 7, 8, 19, 20, 23, 24, 31, 40, 47] {
            let k = eng.k_of(li as u32);
            match &self.states[li].attn {
                AttnState::Gdn(g) => {
                    if let Ok(h) = k.stream.clone_dtoh(&g.s_all) {
                        let s: f64 = h.iter().map(|&v| v as f64).sum();
                        eprintln!("[dump:{tag}] 层{li} GDN s_all sum={s:.6}");
                    }
                }
                AttnState::Qsa(q) => {
                    // clone_dtoh 不收 view ⇒ 整片拷（诊断用途，几 MB 无所谓），
                    // 只累计前 len 行参与校验和。㊶ 量化档 kn/v 为 None ⇒ 跳过（诊断无此需求）。
                    // P2 分页：池布局≠逻辑行 ⇒ 用块表装配后的逻辑行做校验和。
                    if q.paged {
                        if let Ok((kn, v)) = q.paged_kv_to_host(k, q.len) {
                            eprintln!(
                                "[dump:{tag}] 层{li} QSA len={} kn={:.4} v={:.4} nblocks={} (paged)",
                                q.len, cks_bits(&kn), cks_bits(&v), q.nblocks
                            );
                        }
                    } else if let (Some(qkn), Some(qv)) = (&q.kn, &q.v) {
                        if let Ok(kn) = k.stream.clone_dtoh(qkn) {
                            if let Ok(v) = k.stream.clone_dtoh(qv) {
                                let kd = qkv_bits(qkn.len(), q.cap.max(1));
                                let knr = &kn[..(q.len * kd).min(kn.len())];
                                let vr = &v[..(q.len * kd).min(v.len())];
                                eprintln!(
                                    "[dump:{tag}] 层{li} QSA len={} kn={:.4} v={:.4} nblocks={}",
                                    q.len, cks(knr), cks(vr), q.nblocks
                                );
                            }
                        }
                    }
                }
            }
            if let Some(p) = self.states[li].ple.as_ref() {
                let _ = p;
                eprintln!("[dump:{tag}] 层{li} PLE 有状态");
            }
        }
        let rows = hyper.len() / self.dim.max(1);
        if rows > 0 {
            // 同上：整片拷（末 chunk 的行数 × dim）。
            if let Ok(h) = eng.k.stream.clone_dtoh(hyper) {
                eprintln!("[dump:{tag}] hyper 末行 sum={:.4}", cks(&h));
            }
        }
    }

    /// prefill 的**分块粒度**（`LINGXI_PREFILL_CHUNK`，默认 2048；0/1 = 不分块走老路）。
    /// P15-㉝（2026-10-17）：长 prompt 整段前向的激活（t×MoE f32 scratch + GR 图缓冲）
    /// 随 t 线性涨 ⇒ ~2-4K token 就顶爆 32GB VRAM。分块后峰值显存 O(chunk)，
    /// 状态逐 chunk 推进（GDN 递推天然按序、QSA/PLE 按绝对位置）。
    ///
    /// ✅ **逐位等价性（2026-10-18 定案）**：**单卡分块 ≡ 不分块**；**split+分块 ≡ 单卡**
    /// ——23,673 token / 47 个 chunk 的双卡+线程流水与单卡**首 token 完全一致（95835）**。
    /// 旧版"split+分块会发散（chunk 尺寸相关）"的病根是专家核每组只算前 8 个对
    ///：脏值随 chunk 边界/命中态变化，看起来像"状态结转 bug"。
    /// 修好后**不再需要"split 下不分块"这条纪律**。
    fn prefill_chunk_of(&mut self, t: usize, split: bool) -> usize {
        let env = std::env::var("LINGXI_PREFILL_CHUNK").ok().and_then(|v| v.parse::<usize>().ok());
        match env {
            Some(n) => {
                if n <= 1 {
                    t.max(1)
                } else {
                    if split {
                        static WARN: std::sync::Once = std::sync::Once::new();
                        WARN.call_once(|| {
                            eprintln!(
                                "[prefill-chunk] split+分块逐位等价已认证（㉞）——显式指定生效"
                            );
                        });
                    }
                    n.min(t)
                }
            }
            // ★ P15-㉞：split+分块的确定性发散已修复（device-aware 池）⇒ split 下也分块。
            // ★ 但粒度分档：split 时 GPU1 显存余量小（槽位占大头），chunk=2048 的 MoE
            // scratch ~2-3GB 会 OOM（17K 实测 CUDA_ERROR_OUT_OF_MEMORY）⇒ split 默认 512。
            None => {
                if split { 512usize.min(t) } else { 2048usize.min(t) }
            }
        }
    }

    /// **P3 layer-major prefill 开关**（`LINGXI_PREFILL_LM`，默认**关**）。
    ///
    /// 2026-10-18 实测（33,052-token 冷 prefill，7200/7600 槽）：开 = 291.8s，
    /// 关 = 270.6s ⇒ **当前配置下略亏**。原因是 chunk=1024 时专家 H2D 已被计算盖住，
    /// prefill 是**算力瓶颈**（evprof：QSA attn 43% / MoE 专家段 24%，且专家段
    /// 与 miss 数无关 = 纯计算）。它仍是 expert-stationary 的前置框架，且对
    /// **≥128K** 的超长 prompt 有明确收益（H2D 总量 ÷ 外层块数），故保留为 opt-in。
    fn layer_major_enabled() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| std::env::var("LINGXI_PREFILL_LM").map(|s| s == "1").unwrap_or(false))
    }

    /// layer-major 的**外层（整层）分块**：`LINGXI_PREFILL_LM_CHUNK`，默认 16384；
    /// `0` = 整段 prompt 一块。外层越大，每层专家换入次数越少（H2D 越小），
    /// 但要常驻 `outer × dim × 2B` 的激活（16384 token ≈ 320MiB/卡）。
    fn layer_major_outer() -> usize {
        std::env::var("LINGXI_PREFILL_LM_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(16384)
    }

    /// **P3：layer-major prefill**（2026-10-18）。
    ///
    /// 把原来的「外层 chunk × 内层 layer」交换成「外层 layer × 内层 sub-chunk」：
    /// 同一层的专家在整段 prompt 上**只换入一次**，H2D 从 O(chunks × 48 层) 降到
    /// O(outer_chunks × 48 层)。
    ///
    /// 为什么可以做到 **bit 级不变**：这是纯调度重排。每个 (layer, sub-chunk) 的输入行、
    /// 该层状态、算子形状都与 token-major 完全相同（sub-chunk 仍按绝对位置递增，
    /// 注意力只会读到自己之前已写好的 KV），因此数值结果必须逐位一致。
    ///
    /// 显存：外层次块的激活 `outer × dim` 常驻一张卡（16384 ≈ 320MiB），
    /// 层内仍按 `sub` 行分块（MoE scratch 峰值 O(sub)）。
    #[allow(clippy::too_many_arguments)]
    fn prefill_layer_major(
        &mut self,
        eng: &Engine,
        prompt: &[i64],
        provider: &mut dyn LayerProvider,
        start: usize,
        collect: bool,
        sub: usize,
    ) -> Result<(i64, Option<CudaSlice<bf16>>)> {
        let dim = self.dim;
        let total = prompt.len() - start;
        anyhow::ensure!(total > 0, "layer-major prefill：没有新增 token");
        let sub = sub.max(1);
        // 均衡切分外层：避免「最后一个 outer 只有 300 token，却要扫一遍全层专家」。
        let cap = Self::layer_major_outer();
        let n_outer = if cap == 0 { 1 } else { (total + cap - 1) / cap };
        let per = (total + n_outer - 1) / n_outer;

        // ids 一次推满：每个 sub-chunk 用**显式 ids_len** 截断，PLE 只看得到自己之前的历史。
        self.ids.extend_from_slice(&prompt[start..]);

        let lo = if eng.split_active() { eng.split_lo.min(eng.n_layers) } else { eng.n_layers };
        let splitt = eng.split_active() && eng.n_layers > lo;
        let k0 = eng.k;
        let k1 = eng.k1;

        let mut all1: Option<CudaSlice<bf16>> = None;
        let mut tail: Option<CudaSlice<bf16>> = None;
        let mut last_c = 0usize;
        let t_lm = std::time::Instant::now();

        // ── 双卡线程版：GPU0 跑外层前半、GPU1 跑后半，**外层之间流水** ──────────
        // 为什么要线程：layer-major 本身把 H2D 降到 1/outer，但两卡变成串行；
        // 线程让「GPU0 处理后一个 outer」与「GPU1 处理前一个 outer」重叠，
        // 既保住 H2D 收益，又不丢 token-major 线程版的 1.7× 并行。
        if splitt && lingxi_model::flag("LINGXI_PREFILL_THREADS") {
            let kk1 = k1.expect("split 时 k1 必存在");
            // 工作线程只能 peek：先在本线程把 48 层全部建出来。
            for i in 0..eng.n_layers {
                let _ = provider.layer(i)?;
            }
            let cache = provider.as_cache().ok_or_else(|| {
                anyhow::anyhow!("LINGXI_PREFILL_THREADS 需要缓存型 provider（DeviceLayerCache）")
            })?;
            let hidden = self.hidden;
            let plen = prompt.len();
            let (st_lo, st_hi) = self.states.split_at_mut(lo as usize);
            let ids = &self.ids[..];
            let (tx, rx) = std::sync::mpsc::channel::<(usize, usize, CudaSlice<bf16>)>();
            let t_thr = std::time::Instant::now();
            let (tail, last_c, all1) = std::thread::scope(
                |sc| -> Result<(CudaSlice<bf16>, usize, Option<CudaSlice<bf16>>)> {
                    let h_a = sc.spawn(move || -> Result<()> {
                        k0.ctx.bind_to_thread()?;
                        let mut off = start;
                        while off < plen {
                            let n = per.min(plen - off);
                            let idx_host: Vec<i32> =
                                prompt[off..off + n].iter().map(|&v| v as i32).collect();
                            let didx = k0.stream.clone_htod(&idx_host)?;
                            let mut emb = k0.stream.alloc_zeros::<bf16>(n * hidden)?;
                            k0.gather_rows_bf16(
                                &eng.endpoints.token_embedding,
                                &didx,
                                &mut emb,
                                n,
                                hidden,
                                0,
                            )?;
                            let mut h = k0.stream.alloc_zeros::<bf16>(n * dim)?;
                            k0.hc_repeat_bf16(&emb, &mut h, n, hidden, eng.shape.hc)?;
                            drop(emb);
                            Self::run_layers_in_place(
                                st_lo, 0, ids, eng, 0, lo, &mut h, n, sub, off, None, Some(cache),
                                dim,
                            )?;
                            let mut h1 = lingxi_model::scratch::bf16(kk1, n * dim)?;
                            lingxi_model::scratch::cross_copy(k0, &h, kk1, &mut h1)?;
                            drop(h);
                            if tx.send((off, n, h1)).is_err() {
                                break;
                            }
                            off += n;
                        }
                        Ok(())
                    });
                    let h_b = sc.spawn(
                        move || -> Result<(CudaSlice<bf16>, usize, Option<CudaSlice<bf16>>)> {
                            kk1.ctx.bind_to_thread()?;
                            let mut all1 = if collect {
                                Some(lingxi_model::scratch::bf16(kk1, plen * dim)?)
                            } else {
                                None
                            };
                            let mut tail: Option<(CudaSlice<bf16>, usize)> = None;
                            while let Ok((off, n, mut h1)) = rx.recv() {
                                Self::run_layers_in_place(
                                    st_hi,
                                    lo as usize,
                                    ids,
                                    eng,
                                    lo,
                                    eng.n_layers,
                                    &mut h1,
                                    n,
                                    sub,
                                    off,
                                    None,
                                    Some(cache),
                                    dim,
                                )?;
                                if off + n == plen {
                                    let rem = n % sub;
                                    let lc = if rem == 0 { sub.min(n) } else { rem };
                                    let mut tb1 = lingxi_model::scratch::bf16(kk1, lc * dim)?;
                                    kk1.stream.memcpy_dtod(
                                        &h1.slice((n - lc) * dim..n * dim),
                                        &mut tb1,
                                    )?;
                                    let mut tb = lingxi_model::scratch::bf16(k0, lc * dim)?;
                                    lingxi_model::scratch::cross_copy(kk1, &tb1, k0, &mut tb)?;
                                    tail = Some((tb, lc));
                                }
                                if let Some(all) = all1.as_mut() {
                                    kk1.stream.memcpy_dtod(
                                        &h1,
                                        &mut all.slice_mut(off * dim..(off + n) * dim),
                                    )?;
                                }
                                drop(h1);
                                if lingxi_model::flag("LINGXI_PREFILL_TIMING") {
                                    let el = t_thr.elapsed().as_secs_f64();
                                    eprintln!(
                                        "[prefill-lm] {}/{} 行  累计 {el:.1}s ⇒ {:.2} ms/行(均)",
                                        off + n,
                                        plen,
                                        el / (off + n - start) as f64 * 1e3
                                    );
                                }
                            }
                            let (tb, lc) = tail
                                .ok_or_else(|| anyhow::anyhow!("[lm-threads] 没有任何 outer chunk"))?;
                            Ok((tb, lc, all1))
                        },
                    );
                    h_a.join().map_err(|_| anyhow::anyhow!("layer-major GPU0 线程 panic"))??;
                    h_b.join().map_err(|_| anyhow::anyhow!("layer-major GPU1 线程 panic"))?
                },
            )?;
            let token = self.head_argmax(eng, &tail, last_c)?;
            lingxi_model::scratch::put_bf16(tail);
            self.ids.push(token);
            if let Some(a1) = all1 {
                let mut a0 = lingxi_model::scratch::bf16(k0, prompt.len() * dim)?;
                lingxi_model::scratch::cross_copy(kk1, &a1, k0, &mut a0)?;
                return Ok((token, Some(a0)));
            }
            return Ok((token, None));
        }

        {
            let mut off = start;
            while off < prompt.len() {
                let n = per.min(prompt.len() - off);
                let mut h = self.embed(eng, &prompt[off..off + n])?;
                // 拆字段借用：states 可变、ids 只读（同一结构体的不同字段）。
                let (states, ids) = (&mut self.states[..], &self.ids[..]);
                // 前半段层（GPU0）
                Self::run_layers_in_place(
                    states, 0, ids, eng, 0, lo, &mut h, n, sub, off,
                    Some(&mut *provider), None, dim,
                )?;
                if splitt {
                    let kk1 = k1.expect("split 时 k1 必存在");
                    let mut h1 = lingxi_model::scratch::without_pool(|| {
                        lingxi_model::scratch::bf16(kk1, n * dim)
                    })?;
                    // 整段一次性跨卡（比逐 sub-chunk 搬省得多）。
                    lingxi_model::scratch::cross_copy(k0, &h, kk1, &mut h1)?;
                    lingxi_model::scratch::put_bf16(h);
                    Self::run_layers_in_place(
                        states, lo as usize, ids, eng, lo, eng.n_layers, &mut h1, n, sub, off,
                        Some(&mut *provider), None, dim,
                    )?;
                    h = h1; // 后半段结果在 GPU1
                }
                // 最后一个 outer：取末尾 sub 段（head 只需要最后这几行）。
                if off + n == prompt.len() {
                    let rem = n % sub;
                    last_c = if rem == 0 { sub.min(n) } else { rem };
                    let mut tb = lingxi_model::scratch::bf16(k0, last_c * dim)?;
                    k0.stream.memcpy_dtod(
                        &h.slice((n - last_c) * dim..n * dim),
                        &mut tb,
                    )?;
                    tail = Some(tb);
                }
                if collect {
                    if all1.is_none() {
                        let ka = if splitt { k1.expect("split") } else { k0 };
                        all1 = Some(lingxi_model::scratch::without_pool(|| {
                            lingxi_model::scratch::bf16(ka, prompt.len() * dim)
                        })?);
                    }
                    let dst_base = off * dim;
                    let all = all1.as_mut().unwrap();
                    let kk = if splitt { k1.expect("split") } else { k0 };
                    kk.stream.memcpy_dtod(
                        &h,
                        &mut all.slice_mut(dst_base..dst_base + n * dim),
                    )?;
                }
                if splitt {
                    drop(h); // GPU1 的缓冲不进池（与既有 split 路径保持一致）
                } else {
                    lingxi_model::scratch::put_bf16(h);
                }
                off += n;
                if lingxi_model::flag("LINGXI_PREFILL_TIMING") {
                    let el = t_lm.elapsed().as_secs_f64();
                    eprintln!(
                        "[prefill-lm] {off}/{} 行  累计 {el:.1}s ⇒ {:.2} ms/行(均)",
                        prompt.len(),
                        el / (off - start) as f64 * 1e3
                    );
                }
            }
        }

        let tail = tail.expect("prompt 非空 ⇒ 必有 tail");
        let token = self.head_argmax(eng, &tail, last_c)?;
        lingxi_model::scratch::put_bf16(tail);
        self.ids.push(token);

        if collect {
            let all1 = all1.expect("collect ⇒ 必有整段缓冲");
            let mut a0 = lingxi_model::scratch::bf16(k0, prompt.len() * dim)?;
            if splitt {
                lingxi_model::scratch::cross_copy(k1.expect("split"), &all1, k0, &mut a0)?;
            } else {
                k0.stream.memcpy_dtod(&all1, &mut a0)?;
            }
            Ok((token, Some(a0)))
        } else {
            Ok((token, None))
        }
    }

    /// prefill：吃 prompt，推进全部层状态，返回最后一个位置贪心出的下一 token
    /// （已自动追加进 `ids`）。只能在空序列上调用一次。
    /// ★ P15-㉝：长 prompt 自动分块（`prefill_chunk_of`），短 prompt 行为不变。
    pub fn prefill(
        &mut self,
        eng: &Engine,
        prompt: &[i64],
        provider: &mut dyn LayerProvider,
    ) -> Result<i64> {
        assert!(self.ids.is_empty(), "prefill 只能在空序列上调用");
        assert!(!prompt.is_empty(), "prompt 不能为空");
        self.prefill_from(eng, prompt, provider, 0)
    }

    /// **前缀续跑**（C2 前缀缓存，2026-10-18）：`prompt` 必须以**当前 `ids`** 为精确前缀，
    /// 只把 `prompt[ids.len..]` 喂进去。数学上与"整段重新 prefill"逐位相同
    /// （分块不变性 §㊿-② 已证；状态全部按绝对位置推进）。
    ///
    /// ⚠️ 使用前提：`ids` 必须正好等于**已经喂进 KV 的那些 token**。`decode_token` 会把
    /// 新产生的 token 也 push 进 `ids`（它还没被喂）⇒ 调用方要先 [`Self::truncate_unfed`]。
    pub fn prefill_append(
        &mut self,
        eng: &Engine,
        prompt: &[i64],
        provider: &mut dyn LayerProvider,
    ) -> Result<i64> {
        let have = self.ids.len();
        anyhow::ensure!(have > 0, "prefill_append 需要非空前缀（新序列请用 prefill）");
        anyhow::ensure!(prompt.len() > have, "prefill_append：新 prompt 没有新增 token");
        anyhow::ensure!(
            prompt[..have] == self.ids[..],
            "prefill_append：新 prompt 的前缀与已缓存的不一致（必须精确前缀）"
        );
        self.prefill_from(eng, prompt, provider, have)
    }

    /// 丢掉**最后一个"已产生但从未喂进 KV"**的 id（见 [`Self::prefill_append`]）。
    /// 只动 bookkeeping，不动任何设备状态（该位置的 KV/GDN/PLE 本来就没写）。
    pub fn truncate_unfed(&mut self) {
        let _ = self.ids.pop();
    }

    fn prefill_from(
        &mut self,
        eng: &Engine,
        prompt: &[i64],
        provider: &mut dyn LayerProvider,
        start: usize,
    ) -> Result<i64> {
        Ok(self.prefill_from_impl(eng, prompt, provider, start, false)?.0)
    }

    /// `collect=true`：额外返回**整段 prompt 的 hyper**（`[prompt.len, dim]`，在主卡上）。
    /// 供 MTP 投机用（它的 KV 必须覆盖整段 prompt）。
    fn prefill_from_impl(
        &mut self,
        eng: &Engine,
        prompt: &[i64],
        provider: &mut dyn LayerProvider,
        start: usize,
        collect: bool,
    ) -> Result<(i64, Option<CudaSlice<bf16>>)> {
        let chunk = self.prefill_chunk_of(prompt.len(), eng.split_active());
        // [探针] `LINGXI_PREWARM_ALL=1`：顺序路径也先把全部层建出来。
        // 用途：判"线程版的数值差异"到底来自**线程**，还是来自**预热改变了分配历史**
        // （线程版必须预热才能 `peek`；若打开这条后顺序版也变 ⇒ 与线程无关）。
        if lingxi_model::flag("LINGXI_PREWARM_ALL") {
            for i in 0..eng.n_layers {
                let _ = provider.layer(i)?;
            }
        }
        // ★ P3（2026-10-18）：layer-major —— 外层整层、内层行段。
        // 纯调度重排，逐位等价；收益是每层专家在整段 prompt 上只换入一次。
        if Self::layer_major_enabled() {
            return self.prefill_layer_major(eng, prompt, provider, start, collect, chunk);
        }
        let mut hyper = None;
        // `collect` 时的整段 hyper（`[prompt.len, dim]`，主卡）
        let mut all: Option<CudaSlice<bf16>> = None;
        let mut last_c = 0usize;
        let mut off = start;
        #[allow(clippy::let_and_return)]
        let t0 = std::time::Instant::now();
        let _ = &t0;
        // ★ P15-㊿：**outer chunk 成对流水**（split 时）：chunk i 的后半（lo..n，GPU1）与
        // chunk i+1 的前半（0..lo，GPU0）并行。chunk 之间没有数据依赖（每个 chunk 是独立的
        // token 区间），所以这是**无损的调度重排**：每个 chunk 的层区间/ids 可见性/PLE 尾部
        // 都完全相同 ⇒ 判据 = 首 token 不变。
        // 释放上一 chunk 的后半用**事件**同步（不把 10MB 拷回来），只有最后一个 chunk 才回传。
        // ⚠️ 实测（2026-10-17）：即便开了，**没有提速**（530→2120 token 都持平），而且
        // 多头 prompt 下首 token 会变（79648 vs 16）⇒ 还没到能默认开的程度。
        // 根因假设：MoE 每层路由都要 `stream.synchronize`（见 moe.rs），host 在层间
        // 被反复排空 ⇒ 两卡的重叠被 host 串行化吃掉（这正是 A2 设备路由想解决的问题）。
        // 因此当前仅作**opt-in**（`LINGXI_PREFILL_PIPE=1`），默认走原顺序路径。
        // ★ ㊿-① 线程版：GPU0/GPU1 各一线程（全部层预热后 provider 只读 peek）。
        // 每个 chunk：A 跑 0..lo 并跨卡搬到 GPU1 → mpsc 交给 B 跑 lo..n。
        if eng.split_active() && lingxi_model::flag("LINGXI_PREFILL_THREADS") {
            let k1 = eng.k1.expect("split_lo 有效时 k1 必存在");
            let lo = eng.split_lo;
            let n_layers = eng.n_layers;
            for i in 0..n_layers {
                let _ = provider.layer(i)?;
            }
            let cache = provider.as_cache().ok_or_else(|| {
                anyhow::anyhow!("LINGXI_PREFILL_THREADS 需要缓存型 provider（DeviceLayerCache）")
            })?;
            struct ChunkIn {
                off: usize,
                c: usize,
                x: CudaSlice<bf16>,
                ids_len: usize,
            }
            let mut ins: Vec<ChunkIn> = Vec::new();
            while off < prompt.len() {
                let c = chunk.min(prompt.len() - off);
                self.ids.extend_from_slice(&prompt[off..off + c]);
                let x = self.embed(eng, &prompt[off..off + c])?;
                ins.push(ChunkIn { off, c, x, ids_len: self.ids.len() });
                off += c;
            }
            last_c = ins.last().map(|c| c.c).unwrap_or(0);
            if lingxi_model::flag("LINGXI_PIPE_PROF") {
                eprintln!(
                    "[pipe-t] 全部 {} chunk 的 embedding 预计算完成（{:.2}s）",
                    ins.len(),
                    t0.elapsed().as_secs_f64()
                );
            }
            let ids = &self.ids[..];
            let dim = self.dim;
            let plen = prompt.len();
            let (st_lo, st_hi) = self.states.split_at_mut(lo as usize);
            let base_hi = lo as usize;
            let (tx, rx) = std::sync::mpsc::channel::<(usize, usize, CudaSlice<bf16>, usize)>();
            let t_thr = std::time::Instant::now();
            let (hp, all1) = std::thread::scope(
                |sc| -> Result<(CudaSlice<bf16>, Option<CudaSlice<bf16>>)> {
                let h_a = sc.spawn(move || -> Result<()> {
                    // 新线程没有 current CUDA context ⇒ 先绑定（否则第一个驱动调用就 INVALID_VALUE）
                    eng.k.ctx.bind_to_thread()?;
                    // ★ 2026-10-18：**恢复池化**。旧版注释说"线程里开池会出 NaN"，
                    // 那是在池键还只有 (类型,长度)、且 GPU1 禁池的时代；现在键含设备号，
                    // 且 GPU1 也走池（见 run_layers_range_parts）。判据仍看首 token 与长 prompt 逐位。
                    for ch in ins {
                        let h_lo = Self::run_layers_range_parts(
                            st_lo, 0, dim, ids, eng, 0, lo, ch.x, ch.c, None, Some(cache), None, 0,
                            ch.ids_len,
                        )?;
                        let mut h1 = lingxi_model::scratch::bf16(k1, ch.c * dim)?;
                        lingxi_model::scratch::cross_copy(eng.k, &h_lo, k1, &mut h1)?;
                        // [探针] 逐 chunk 检查两半的中间结果（NaN/量级）——定位线程版的数值发散点。
                        if lingxi_model::flag("LINGXI_PIPE_DUMP") {
                            if let Ok(h) = eng.k.stream.clone_dtoh(&h_lo) {
                                let (mut nan, mut mx, mut sum) = (0usize, 0f32, 0f64);
                                for v in &h {
                                    let f = v.to_f32();
                                    if f.is_nan() {
                                        nan += 1;
                                    } else {
                                        mx = mx.max(f.abs());
                                        sum += f as f64;
                                    }
                                }
                                eprintln!(
                                    "[pipe-dump] lo chunk@{} c={} nan={nan} maxabs={mx:.4} sum={sum:.2}",
                                    ch.off, ch.c
                                );
                            }
                        }
                        drop(h_lo);
                        if lingxi_model::flag("LINGXI_PIPE_PROF") {
                            eprintln!("[pipe-t] chunk@{} lo→GPU1（{:.1}s）", ch.off,
                                t_thr.elapsed().as_secs_f64());
                        }
                        let _ = tx.send((ch.off, ch.c, h1, ch.ids_len));
                    }
                    Ok(())
                });
                let h_b = sc.spawn(
                    move || -> Result<(CudaSlice<bf16>, Option<CudaSlice<bf16>>)> {
                    k1.ctx.bind_to_thread()?;
                    // 同上：工作线程恢复池化（pool 是 thread_local，各线程各一套）
                    let mut last: Option<CudaSlice<bf16>> = None;
                    // `collect`：把每个 chunk 的 h 按偏移收进**GPU1 上**的整段缓冲，
                    // 最后一次性回传（比逐块 PCIe 往返省得多）。
                    let mut all1: Option<CudaSlice<bf16>> = None;
                    while let Ok((off, c, h1, ids_len)) = rx.recv() {
                        if lingxi_model::flag("LINGXI_PIPE_DUMP") {
                            if let Ok(h) = k1.stream.clone_dtoh(&h1) {
                                let (mut nan, mut mx, mut sum) = (0usize, 0f32, 0f64);
                                for v in &h {
                                    let f = v.to_f32();
                                    if f.is_nan() {
                                        nan += 1;
                                    } else {
                                        mx = mx.max(f.abs());
                                        sum += f as f64;
                                    }
                                }
                                eprintln!(
                                    "[pipe-dump] hi 收到 c={c} nan={nan} maxabs={mx:.4} sum={sum:.2}"
                                );
                            }
                        }
                        let h_hi = Self::run_layers_range_parts(
                            st_hi, base_hi, dim, ids, eng, lo, n_layers, h1, c, None, Some(cache),
                            None, 0, ids_len,
                        )?;
                        if lingxi_model::flag("LINGXI_PIPE_PROF") {
                            eprintln!("[pipe-t] chunk 后半完成（{:.1}s）",
                                t_thr.elapsed().as_secs_f64());
                        }
                        if collect {
                            if all1.is_none() {
                                all1 = Some(lingxi_model::scratch::bf16(k1, plen * dim)?);
                            }
                            let dst = all1.as_mut().unwrap();
                            k1.stream.memcpy_dtod(
                                &h_hi,
                                &mut dst.slice_mut(off * dim..(off + c) * dim),
                            )?;
                        }
                        last = Some(h_hi);
                    }
                    let last = last.ok_or_else(|| anyhow::anyhow!("[pipe-threads] 没有任何 chunk"))?;
                    Ok((last, all1))
                });
                h_a.join().map_err(|_| anyhow::anyhow!("GPU0 线程 panic"))??;
                h_b.join().map_err(|_| anyhow::anyhow!("GPU1 线程 panic"))?
            })?;
            if let Some(a1) = all1 {
                // 整段 hyper：GPU1 → GPU0 一次搬运；head 用最后一块（拷出来，避免视图类型）
                let mut a0 = lingxi_model::scratch::bf16(eng.k, plen * dim)?;
                lingxi_model::scratch::cross_copy(k1, &a1, eng.k, &mut a0)?;
                drop(a1);
                drop(hp);
                let mut tail = lingxi_model::scratch::bf16(eng.k, last_c * dim)?;
                eng.k.stream.memcpy_dtod(
                    &a0.slice((plen - last_c) * dim..plen * dim),
                    &mut tail,
                )?;
                hyper = Some(tail);
                all = Some(a0);
            } else {
                let mut hb = lingxi_model::scratch::bf16(eng.k, last_c * dim)?;
                lingxi_model::scratch::cross_copy(k1, &hp, eng.k, &mut hb)?;
                drop(hp);
                self.dump_prefill_state(eng, &hb, "last");
                hyper = Some(hb);
            }
        } else if eng.split_active() && lingxi_model::flag("LINGXI_PREFILL_PIPE") {
            anyhow::ensure!(
                !collect,
                "prefill_with_hyper（collect）暂不支持 LINGXI_PREFILL_PIPE；请关掉该开关或用线程版"
            );
            let k1 = eng.k1.expect("split_lo 有效时 k1 必存在");
            let lo = eng.split_lo;
            let mut prev: Option<(CudaSlice<bf16>, cudarc::driver::CudaEvent)> = None;
            while off < prompt.len() {
                let c = chunk.min(prompt.len() - off);
                self.ids.extend_from_slice(&prompt[off..off + c]);
                let x = self.embed(eng, &prompt[off..off + c])?;
                let ids_len = self.ids.len();
                let p0 = std::time::Instant::now();
                let h_lo =
                    self.run_layers_range(eng, 0, lo, x, c, provider, None, off, ids_len)?;
                let p1 = std::time::Instant::now();
                // 前半 → GPU1（阻塞到前半完成；此刻 GPU1 正在跑上一 chunk 的后半 ⇒ 重叠）
                let mut h1 = lingxi_model::scratch::bf16(k1, c * self.dim)?;
                lingxi_model::scratch::cross_copy(eng.k, &h_lo, k1, &mut h1)?;
                let p2 = std::time::Instant::now();
                drop(h_lo);
                let h_hi =
                    self.run_layers_range(eng, lo, eng.n_layers, h1, c, provider, None, off, ids_len)?;
                if lingxi_model::flag("LINGXI_PIPE_PROF") {
                    eprintln!(
                        "[pipeprof] chunk@{off}+{c}  lo {:6.1}ms  xfer {:6.1}ms  hi {:6.1}ms",
                        p1.duration_since(p0).as_secs_f64() * 1e3,
                        p2.duration_since(p1).as_secs_f64() * 1e3,
                        p2.elapsed().as_secs_f64() * 1e3
                    );
                }
                // 回收上一 chunk 的后半（等它真的跑完，再还给池）
                if let Some((ob, oe)) = prev.take() {
                    let _ = oe.synchronize();
                    drop(ob);
                }
                let ev = k1.stream.context().new_event(None)?;
                ev.record(&k1.stream)?;
                prev = Some((h_hi, ev));
                off += c;
                last_c = c;
                if lingxi_model::flag("LINGXI_PREFILL_TIMING") {
                    let el = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "[prefill-pipe] {off}/{} 行  累计 {el:.1}s ⇒ {:.2} ms/行(均)",
                        prompt.len(),
                        el / off as f64 * 1e3
                    );
                }
            }
            let (hp, ev) = prev.take().expect("prompt 非空 ⇒ 至少一个 chunk");
            let _ = ev.synchronize();
            let mut hb = lingxi_model::scratch::bf16(eng.k, last_c * self.dim)?;
            lingxi_model::scratch::cross_copy(k1, &hp, eng.k, &mut hb)?;
            drop(hp);
            self.dump_prefill_state(eng, &hb, "last");
            hyper = Some(hb);
        } else {
        while off < prompt.len() {
            let c = chunk.min(prompt.len() - off);
            // ids 必须**逐 chunk 推**：`run_layers` 给 PLE 传的是 `ids[..ids_len]` 的
            // 末尾 t 个 —— 一次推全会让后续 chunk 的 PLE 看到错误的尾巴。
            self.ids.extend_from_slice(&prompt[off..off + c]);
            let x = self.embed(eng, &prompt[off..off + c])?;
            let h = self.run_layers(eng, x, c, provider)?;
            if collect {
                if all.is_none() {
                    all = Some(lingxi_model::scratch::bf16(eng.k, prompt.len() * self.dim)?);
                }
                let dst = all.as_mut().unwrap();
                eng.k.stream.memcpy_dtod(
                    &h,
                    &mut dst.slice_mut(off * self.dim..(off + c) * self.dim),
                )?;
            }
            off += c;
            last_c = c;
            self.dump_prefill_state(eng, &h, &format!("chunk@{off}"));
            // ★ P15-㉟ 长上下文诊断：逐 chunk 计时（scaling 曲线：ms/行 随上下文增长 ⇒ O(T) culprit）
            if lingxi_model::flag("LINGXI_PREFILL_TIMING") {
                let el = t0.elapsed().as_secs_f64();
                eprintln!(
                    "[prefill] {off}/{} 行  累计 {el:.1}s ⇒ {:.2} ms/行(均)",
                    prompt.len(),
                    el / off as f64 * 1e3
                );
            }
            hyper = Some(h);
        }
        }
        let hyper = hyper.expect("prompt 非空 ⇒ 至少一个 chunk");
        let token = self.head_argmax(eng, &hyper, last_c)?;
        self.ids.push(token);
        if let Some(a) = all {
            // `hyper` 是最后一块的副本（head 已用完），放回池里；对外返回整段
            lingxi_model::scratch::put_bf16(hyper);
            Ok((token, Some(a)))
        } else {
            Ok((token, None))
        }
    }

    /// decode 一步：用当前末尾 token 跑一层所有层，返回贪心出的下一 token
    /// （已自动追加进 `ids`）。
    /// 诊断用：把**喂给下一步的 token** 强制成 `LINGXI_FIXED_IDS` 给定的固定序列，
    /// 让不同配置处理**完全相同的输入**。
    ///
    /// 为什么必须这么做：W4 之类的改动会改变采样结果 → 改变 token 流 → 改变专家路由与
    /// 槽命中率 → 整机耗时波动 ±7%，**把真实增益淹没**（实测：同一配置换一段 token 流
    /// 就能从"快 10%"变成"慢 10%"）。固定 token 流后 A/B 才可信。
    pub fn decode_token(
        &mut self,
        eng: &Engine,
        provider: &mut dyn LayerProvider,
    ) -> Result<i64> {
        let last = *self.ids.last().expect("decode 前序列不能为空");
        let x = self.embed(eng, std::slice::from_ref(&last))?;
        let hyper = self.run_layers(eng, x, 1, provider)?;
        let token = self.head_argmax(eng, &hyper, 1)?;
        // 训练/诊断：只覆盖"下一步吃什么"，返回值仍是真实 argmax
        self.ids.push(next_fixed().unwrap_or(token));
        Ok(token)
    }

    /// 与 [`Self::prefill`] 同步的一步，但**额外交出"层尾 mixer 之前的残差栈"**（`hyper[T, hc*H]`）。
    ///
    /// MTP 草稿头的 `h` 输入就是它（post-norm hidden 同时喂 lm_head 与 MTP 头）。
    pub fn prefill_with_hyper(
        &mut self,
        eng: &Engine,
        prompt: &[i64],
        provider: &mut dyn LayerProvider,
    ) -> Result<(i64, CudaSlice<bf16>)> {
        assert!(self.ids.is_empty(), "prefill 只能在空序列上调用");
        assert!(!prompt.is_empty(), "prompt 不能为空");
        // 与 `prefill` 走**同一条实现**（含线程版）；`collect=true` 让它把每个 chunk 的 h
        // 收进整段缓冲 —— MTP 的 KV 必须覆盖整段 prompt。
        let (token, all) = self.prefill_from_impl(eng, prompt, provider, 0, true)?;
        Ok((token, all.expect("collect=true ⇒ 必有整段 hyper")))
    }

    /// 与 [`Self::decode_token`] 同步的一步，但**额外交出 `hyper`**（同上，供 MTP 用）。
    pub fn decode_token_with_hyper(
        &mut self,
        eng: &Engine,
        provider: &mut dyn LayerProvider,
    ) -> Result<(i64, CudaSlice<bf16>)> {
        let last = *self.ids.last().expect("decode 前序列不能为空");
        let x = self.embed(eng, std::slice::from_ref(&last))?;
        let hyper = self.run_layers(eng, x, 1, provider)?;
        let token = self.head_argmax(eng, &hyper, 1)?;
        self.ids.push(next_fixed().unwrap_or(token));
        Ok((token, hyper))
    }

    /// 预分配投机缓冲。`t_max` = 单批最多喂几个 token。
    pub fn spec_prepare(
        &mut self,
        eng: &Engine,
        provider: &mut dyn LayerProvider,
        t_max: usize,
    ) -> Result<SpecBuffers> {
        let k = eng.k;
        let n = eng.n_layers as usize;
        let dim = self.dim;
        let g = &eng.shape.gdn;
        // 录像的每行宽度（= `gdn_step` 三段的输入宽度）
        let (w_qkv, w_z, w_ab) =
            (2 * g.num_k_heads * g.head_k_dim + g.num_v_heads * g.head_v_dim,
             g.num_v_heads * g.head_v_dim, g.num_v_heads);
        let mut gdn_snap = Vec::with_capacity(n);
        let mut gdn_in = Vec::with_capacity(n);
        let mut gdn_qkv = Vec::with_capacity(n);
        let mut gdn_z = Vec::with_capacity(n);
        let mut gdn_a = Vec::with_capacity(n);
        let mut gdn_b = Vec::with_capacity(n);
        let mut ple_snap = Vec::with_capacity(n);
        // 诊断：把 spec_prepare 的两件事分开（`provider.layer(i)` 全层调用 vs 纯分配），
        // 因为实测"只调 spec_prepare 也会把 decode 打成全 0"⇒ 得先定位是哪一半。
        let skip_alloc = lingxi_model::flag("LINGXI_SPEC_PREP_NOALLOC");
        let skip_prov = lingxi_model::flag("LINGXI_SPEC_PREP_NOPROV");
        for i in 0..n {
            // ★ 全量 layer-split：本层的状态/缓冲必须落在**本层所在的那张卡**上
            let k = eng.k_of(i as u32);
            if skip_prov {
                // 不碰 provider（`lingxi_model::layer_kind` 本来就不需要它）
            } else {
                let _ = provider.layer(i as u32)?;
            }
            if skip_alloc {
                gdn_snap.push(None);
                gdn_in.push(None);
                gdn_qkv.push(None);
                gdn_z.push(None);
                gdn_a.push(None);
                gdn_b.push(None);
                ple_snap.push(None);
                continue;
            }
            let is_gdn = matches!(
                lingxi_model::layer_kind(i as u32),
                lingxi_ids::LayerKind::GdnLinearAttention
            );
            // PLE 也是跨 token 可变状态（卷积历史）⇒ 任何带 PLE 的层都要备份
            ple_snap.push(match &self.states[i].ple {
                Some(pst) => Some(k.stream.alloc_zeros::<bf16>(pst.hist.len())?),
                None => None,
            });
            if !is_gdn {
                if self.states[i].ple.is_some() {
                    bail!("层 {i} 带 PLE 但不是 GDN：投机重放只覆盖 GDN 层，请扩展本函数");
                }
                gdn_snap.push(None);
                gdn_in.push(None);
                gdn_qkv.push(None);
                gdn_z.push(None);
                gdn_a.push(None);
                gdn_b.push(None);
                continue;
            }
            let (sv, sh, shn) = {
                let states = &self.states[i];
                let AttnState::Gdn(g) = &states.attn else {
                    bail!("层 {i} 号称为 GDN 但状态不是 GdnState");
                };
                (
                    k.stream.alloc_zeros::<f32>(g.s_all.len())?,
                    k.stream.alloc_zeros::<bf16>(g.k_hist.len())?,
                    k.stream.alloc_zeros::<bf16>(g.k_hist_next.len())?,
                )
            };
            gdn_snap.push(Some((sv, sh, shn)));
            gdn_in.push(Some(k.stream.alloc_zeros::<bf16>(t_max * dim)?));
            // `gdn_step` 的内部输入录像（跳过投影用），见 `GdnRec`
            gdn_qkv.push(Some(k.stream.alloc_zeros::<bf16>(t_max * w_qkv)?));
            gdn_z.push(Some(k.stream.alloc_zeros::<bf16>(t_max * w_z)?));
            gdn_a.push(Some(k.stream.alloc_zeros::<bf16>(t_max * w_ab)?));
            gdn_b.push(Some(k.stream.alloc_zeros::<bf16>(t_max * w_ab)?));
        }
        // ── EP：GPU1 的镜像录像/快照（**全部** n_layers 层，宽度与主卡相同）──
        // 为什么放在这里：`spec_commit` 部分接受时要让**两张卡**都退回并重放到接受前缀；
        // card1 的录像/快照必须落在 GPU1 自己显存里，否则每层都要跨 PCIe 搬输入。
        let ep_side = match eng.ep {
            None => None,
            Some(epx) => {
                let k1 = epx.k1;
                let states_ep = self.states_ep.as_ref().expect("EP 模式必须有 states_ep");
                anyhow::ensure!(states_ep.len() == n, "EP：states_ep 层数不匹配");
                let mut s = crate::checkpoint::SpecSide {
                    gdn_snap: Vec::with_capacity(n),
                    gdn_in: Vec::with_capacity(n),
                    gdn_qkv: Vec::with_capacity(n),
                    gdn_z: Vec::with_capacity(n),
                    gdn_a: Vec::with_capacity(n),
                    gdn_b: Vec::with_capacity(n),
                    ple_snap: Vec::with_capacity(n),
                };
                for i in 0..n {
                    if skip_alloc {
                        s.gdn_snap.push(None);
                        s.gdn_in.push(None);
                        s.gdn_qkv.push(None);
                        s.gdn_z.push(None);
                        s.gdn_a.push(None);
                        s.gdn_b.push(None);
                        s.ple_snap.push(None);
                        continue;
                    }
                    s.ple_snap.push(match &states_ep[i].ple {
                        Some(pst) => Some(k1.stream.alloc_zeros::<bf16>(pst.hist.len())?),
                        None => None,
                    });
                    let is_gdn = matches!(
                        lingxi_model::layer_kind(i as u32),
                        lingxi_ids::LayerKind::GdnLinearAttention
                    );
                    if !is_gdn {
                        if states_ep[i].ple.is_some() {
                            bail!("层 {i} 带 PLE 但不是 GDN（EP 镜像）：请扩展本函数");
                        }
                        s.gdn_snap.push(None);
                        s.gdn_in.push(None);
                        s.gdn_qkv.push(None);
                        s.gdn_z.push(None);
                        s.gdn_a.push(None);
                        s.gdn_b.push(None);
                        continue;
                    }
                    let AttnState::Gdn(g) = &states_ep[i].attn else {
                        bail!("层 {i} 号称为 GDN 但状态不是 GdnState（EP 镜像）");
                    };
                    s.gdn_snap.push(Some((
                        k1.stream.alloc_zeros::<f32>(g.s_all.len())?,
                        k1.stream.alloc_zeros::<bf16>(g.k_hist.len())?,
                        k1.stream.alloc_zeros::<bf16>(g.k_hist_next.len())?,
                    )));
                    s.gdn_in.push(Some(k1.stream.alloc_zeros::<bf16>(t_max * dim)?));
                    s.gdn_qkv.push(Some(k1.stream.alloc_zeros::<bf16>(t_max * w_qkv)?));
                    s.gdn_z.push(Some(k1.stream.alloc_zeros::<bf16>(t_max * w_z)?));
                    s.gdn_a.push(Some(k1.stream.alloc_zeros::<bf16>(t_max * w_ab)?));
                    s.gdn_b.push(Some(k1.stream.alloc_zeros::<bf16>(t_max * w_ab)?));
                }
                Some(s)
            }
        };
        let hyper_rows = k.stream.alloc_zeros::<bf16>(t_max * dim)?;
        Ok(SpecBuffers {
            kv_len: 0,
            gdn_snap,
            gdn_in,
            gdn_qkv,
            gdn_z,
            gdn_a,
            gdn_b,
            ple_snap,
            ep_side,
            gdn_rec_ok: false,
            argmax: Vec::new(),
            hyper_rows,
            pending: 0,
            t_max,
        })
    }

    /// 快照当前可变状态（**每轮开头**调用）。QSA 只要记长度，GDN/PLE 要真拷。
    pub fn spec_snapshot(&mut self, eng: &Engine, sb: &mut SpecBuffers) -> Result<()> {
        sb.kv_len = self.ids.len().saturating_sub(1);
        // 诊断：`LINGXI_SPEC_NOREC=1` 关掉全部拷贝（用来二分"拷贝本身是否破坏状态"）
        if lingxi_model::flag("LINGXI_SPEC_NOREC") {
            return Ok(());
        }
        for (i, slot) in sb.gdn_snap.iter_mut().enumerate() {
            let Some((sv, sh, shn)) = slot else { continue };
            let AttnState::Gdn(g) = &self.states[i].attn else { continue };
            let k = eng.k_of(i as u32); // 状态在哪张卡，拷贝就用哪张卡的上下文
            k.stream.memcpy_dtod(&g.s_all, sv)?;
            k.stream.memcpy_dtod(&g.k_hist, sh)?;
            k.stream.memcpy_dtod(&g.k_hist_next, shn)?;
        }
        for (i, slot) in sb.ple_snap.iter_mut().enumerate() {
            let Some(ps) = slot else { continue };
            let Some(pst) = self.states[i].ple.as_ref() else { continue };
            let k = eng.k_of(i as u32);
            k.stream.memcpy_dtod(&pst.hist, ps)?;
        }
        // ★ EP：第二张卡的状态快照（状态逐位相同，但拷贝必须在本卡上做、只拷"谁属于快照"的那份）
        if let (Some(epx), Some(side)) = (eng.ep, sb.ep_side.as_mut()) {
            let states_ep = self.states_ep.as_ref().expect("EP 模式必须有 states_ep");
            for (i, slot) in side.gdn_snap.iter_mut().enumerate() {
                let Some((sv, sh, shn)) = slot else { continue };
                let AttnState::Gdn(g) = &states_ep[i].attn else { continue };
                epx.k1.stream.memcpy_dtod(&g.s_all, sv)?;
                epx.k1.stream.memcpy_dtod(&g.k_hist, sh)?;
                epx.k1.stream.memcpy_dtod(&g.k_hist_next, shn)?;
            }
            for (i, slot) in side.ple_snap.iter_mut().enumerate() {
                let Some(ps) = slot else { continue };
                let Some(pst) = states_ep[i].ple.as_ref() else { continue };
                epx.k1.stream.memcpy_dtod(&pst.hist, ps)?;
            }
        }
        Ok(())
    }

    /// **一批前进 `tokens.len` 个 token**（一次批前向），返回**每行**的贪心 argmax。
    ///
    /// 因果性保证：第 j 行只看到 `0..=j` ⇒ **第 j 行的 argmax 只依赖被接受的前缀**，
    /// 这也是投机解码"部分接受仍然等价于贪心"的原理。
    /// 同时把每个 GDN 层的输入录进 `sb.gdn_in`（供 `spec_commit` 重放）。
    #[allow(clippy::too_many_arguments)]
    pub fn spec_chunk(
        &mut self,
        eng: &Engine,
        tokens: &[i64],
        provider: &mut dyn LayerProvider,
        sb: &mut SpecBuffers,
        head_alt: Option<&crate::VerifyHeadAlt<'_>>,
    ) -> Result<()> {
        let t = tokens.len();
        let k = eng.k;
        assert!(t >= 1 && t <= sb.t_max, "spec_chunk: t={t} 超出 t_max={}", sb.t_max);
        assert!(!self.ids.is_empty(), "spec_chunk 前序列不能为空（要先 prefill）");
        // ids 的约定：`ids.len == kv_len + 1`（最后一个是"还没喂进去"的 pending）
        let fed = self.ids.len() - 1;
        self.ids.truncate(fed);
        sb.pending = tokens[0];
        self.ids.extend_from_slice(tokens);
        // ★ 2026-10-25：**验证相位内的三段计时**（`LINGXI_SPEC_PROF=1`，
        // 只在整段 > 200 ms 时打一行）。为什么要它：实测有 ~40% 的跑会出现**一轮 1.5~3.2 s
        // 的软卡**，而四相位探针只把它记在 `verify` 里，看不出是"层前向"还是"head"。
        // 口径：`embed`/`layers` 只是**发牌**（异步），`head` 末尾要读回 argmax ⇒ 真等待落在 head。
        let t_dbg0 = std::time::Instant::now();
        let x = self.embed(eng, tokens)?;
        let t_dbg_emb = t_dbg0.elapsed().as_secs_f64() * 1e3;
        let t_dbg = std::time::Instant::now();
        // ⚠️⚠️ 2026-10-25：**"不录像"这条路是坏的，不是"慢一点"**。
        // `gdn_rec_ok=false` ⇒ `spec_commit` 走 legacy 重放（按 `t=accept` 重算投影），
        // 而实测该路径**数值不对**：基线 + `LINGXI_SPEC_NOREC=1` ⇒ 接受率 3.26→0.67、
        // 文本 md5 直接变（33.9 tok/s 也是假象）。同理 `LINGXI_GDN_REPLAY_LEGACY=1`
        // 会让 `build_gdn_rec` 返回 None ⇒ 同一后果。
        // 所以这两个开关在投机链路上**直接拒绝启动**，免得有人拿它做"少录像的开销"对照
        // 而悄悄跑出错误文本。
        if lingxi_model::flag("LINGXI_SPEC_NOREC") {
            anyhow::bail!(
                "LINGXI_SPEC_NOREC=1 会走坏的 legacy 重放（接受率 3.26→0.67、文本错），\
                 已禁用； "
            );
        }
        if lingxi_model::flag("LINGXI_GDN_REPLAY_LEGACY") {
            anyhow::bail!(
                "LINGXI_GDN_REPLAY_LEGACY=1 在投机链路上等价于不录像（同上，数值不对），已禁用；\
                 见 docs "
            );
        }
        sb.gdn_rec_ok = true;
        let mut rec: Option<&mut SpecBuffers> = Some(&mut *sb);
        match Self::verify_chunks_of(t) {
            1 => {
                let hyper = self.run_layers_rec(eng, x, t, provider, rec, 0, self.ids.len())?;
                let t_dbg_lay = t_dbg.elapsed().as_secs_f64() * 1e3;
                // 诊断：把"批处理之后"的 GDN 状态也 dump 一份。
                // **判据**：当 `keep == t`（本轮全接受）时，"提交后的状态"**必须**等于"批处理后的状态"
                // ⇒ 用它判"快速重放 / 老重放"哪一条是**对**的。
                if lingxi_model::flag("LINGXI_GDN_DUMP_S") {
                    for i in 0..eng.n_layers as usize {
                        let kk = eng.k_of(i as u32);
                        if let AttnState::Gdn(g) = &self.states[i].attn {
                            dump_gdn_state(kk, i, &g.s_all, &g.k_hist, "postbatch")?;
                            dump_gdn_full(kk, i, &g.s_all, "postbatch")?;
                            // 顺便把**录像本身**也打一份：用来判"录制不稳定"还是"读取不稳定"
                            // 
                            if sb.gdn_rec_ok {
                                if let Some(q) = sb.gdn_qkv[i].as_ref() {
                                    dump_bf16_head(kk, i, q, "rec.qkv")?;
                                }
                                if let Some(a) = sb.gdn_a[i].as_ref() {
                                    dump_bf16_head(kk, i, a, "rec.a")?;
                                }
                            }
                        }
                    }
                }
                k.stream
                    .memcpy_dtod(&hyper, &mut sb.hyper_rows.slice_mut(0..t * self.dim))?;
                let t_dbg_cp = t_dbg.elapsed().as_secs_f64() * 1e3;
                sb.argmax = self.head_argmax_all(eng, &hyper, t, head_alt)?;
                let t_dbg_hd = t_dbg.elapsed().as_secs_f64() * 1e3;
                if t_dbg_hd > 200.0 && lingxi_model::flag("LINGXI_SPEC_PROF") {
                    eprintln!(
                        "    [chunk-prof] rows{t} 总 {t_dbg_hd:.0} ms：embed {t_dbg_emb:.1}／\
                         layers(发牌) {:.1}／拷回 {:.1}／head(含等 GPU) {:.1}",
                        t_dbg_lay, t_dbg_cp - t_dbg_lay, t_dbg_hd - t_dbg_cp
                    );
                }
                // 收回 hyper 之前把最后一行的 argmax 当新的 pending
                let pending = sb.argmax[t - 1] as i64;
                lingxi_model::scratch::put_bf16(hyper);
                self.ids.push(pending);
            }
            n_chunks => {
                if eng.split_active() && !lingxi_model::flag("LINGXI_NO_VERIFY_PIPE") {
                    // ★★ **两段流水**（P15 里程碑 B 第 7 步，`LINGXI_LAYER_SPLIT` +
                    // `LINGXI_VERIFY_CHUNKS≥2`）：chunk i 的后半（lo..n，GPU1）与
                    // chunk i+1 的前半（0..lo，GPU0）**并行**。
                    //
                    // ⚠️ 2026-10-23 实测修正：旧版在**循环内**回收上一 chunk 的后半
                    // （`prev.take` → `cross_copy` 会 `k1.stream.synchronize`），而那一刻
                    // GPU1 的流上刚排进 hi_i ⇒ 阻塞等到 hi_i 做完，流水被彻底串行化
                    // （实测 chunk=2 每轮 p50 99.4ms > 单发 80.0ms）。现在改为：
                    // ① 循环内**只做** lo_i 的 D2H（只排空 GPU0，等的是 lo_i 自己）
                    // + H2D 到 GPU1（异步排队），**绝不 sync GPU1**；
                    // ② 所有 hi 结果攒到循环外，最后**一次** join（sync GPU1）再回搬。
                    // 这样 GPU0 的 lo_{i+1} 与 GPU1 的 hi_i 天然并行，唯一的等待点是
                    // "lo_i 自己算完"（本来就必须等）。
                    // 数学与顺序 chunk 完全相同：每 chunk 的层区间/录像行偏移/PLE 可见
                    // ids 都一样，只是"谁在什么时候算"变了 ⇒ 应逐位相同（判据 = token 流）。
                    // staging 每 chunk 轮转一级（`cross_copy_staged`），避免 host 覆写
                    // 还在飞的异步 H2D。
                    let k1 = eng.k1.expect("split_lo 有效时 k1 必存在");
                    let lo = eng.split_lo;
                    let mut hyper = lingxi_model::scratch::bf16(k, t * self.dim)?;
                    // 所有 chunk 的后半输出（在 GPU1 上）+ 各自行区间，最后统一 join。
                    let mut his: Vec<(cudarc::driver::CudaSlice<bf16>, usize, usize)> =
                        Vec::with_capacity(n_chunks);
                    let mut stage = 0usize;
                    let mut a = 0usize;
                    let mut rem = n_chunks;
                    while a < t {
                        let b = a + (t - a).div_ceil(rem);
                        rem -= 1;
                        let tc = b - a;
                        let mut xc = lingxi_model::scratch::bf16(k, tc * self.dim)?;
                        k.stream
                            .memcpy_dtod(&x.slice(a * self.dim..b * self.dim), &mut xc)?;
                        let h_lo = self.run_layers_range(
                            eng, 0, lo, xc, tc, provider, rec.as_deref_mut(), a, fed + b,
                        )?;
                        // 前半结果搬到 GPU1（阻塞到本 chunk 前半完成 ⇒ GPU1 上上一 chunk 的
                        // 后半仍在并行跑；H2D 只是排进 GPU1 的流，不等待）。
                        let mut h1 = lingxi_model::scratch::bf16(k1, tc * self.dim)?;
                        lingxi_model::scratch::cross_copy_staged(
                            k, &h_lo, k1, &mut h1, stage,
                        )?;
                        drop(h_lo);
                        // 后半 enqueue 到 GPU1 流（流序排在上一 chunk 后半之后，天然有序）。
                        let h_hi = self.run_layers_range(
                            eng, lo, eng.n_layers, h1, tc, provider, rec.as_deref_mut(), a, fed + b,
                        )?;
                        his.push((h_hi, a, b));
                        stage = (stage + 1) % lingxi_model::scratch::XFER_STAGES;
                        a = b;
                    }
                    // 唯一 join 点：等 GPU1 把全部后半算完，再按 chunk 回搬到 GPU0。
                    k1.stream.synchronize()?;
                    for (hp, pa, pb) in his.drain(..) {
                        let mut hb = lingxi_model::scratch::bf16(k, (pb - pa) * self.dim)?;
                        lingxi_model::scratch::cross_copy_staged(k1, &hp, k, &mut hb, stage)?;
                        drop(hp);
                        k.stream
                            .memcpy_dtod(&hb, &mut hyper.slice_mut(pa * self.dim..pb * self.dim))?;
                        drop(hb);
                        stage = (stage + 1) % lingxi_model::scratch::XFER_STAGES;
                    }
                    k.stream
                        .memcpy_dtod(&hyper, &mut sb.hyper_rows.slice_mut(0..t * self.dim))?;
                    sb.argmax = self.head_argmax_all(eng, &hyper, t, head_alt)?;
                    let pending = sb.argmax[t - 1] as i64;
                    drop(hyper);
                    self.ids.push(pending);
                } else {
                // ★ **切 chunk 顺序跑**（`LINGXI_VERIFY_CHUNKS`，P15 里程碑 B 的第 6 步）：
                // 目的是先**验证切分不改数学**（QSA 的 KV 是 append-only、GDN 是行串行 ⇒
                // 顺序切分本就等价；MoE 的每行归并也仍按专家升序 ⇒ 逐位不变），
                // 再在此基础上把两块 chunk 铺到两卡做两段流水。
                // 注意：本步**不提速**（同一张卡上只是把一次 t 行前向拆成几次小前向）。
                let mut hyper = lingxi_model::scratch::bf16(k, t * self.dim)?;
                let mut a = 0usize;
                let mut rem = n_chunks;
                while a < t {
                    let b = a + (t - a).div_ceil(rem);
                    rem -= 1;
                    let tc = b - a;
                    // 本 chunk 的 x 拷一份（`run_layers_rec` 会把入参当初始残差流消费掉）
                    let mut xc = lingxi_model::scratch::bf16(k, tc * self.dim)?;
                    k.stream
                        .memcpy_dtod(&x.slice(a * self.dim..b * self.dim), &mut xc)?;
                    let hc = self.run_layers_rec(
                        eng,
                        xc,
                        tc,
                        provider,
                        rec.as_deref_mut(),
                        a,
                        fed + b, // PLE 只看末尾 tc 个 id ⇒ 正好是本 chunk 那几行
                    )?;
                    k.stream
                        .memcpy_dtod(&hc, &mut hyper.slice_mut(a * self.dim..b * self.dim))?;
                    lingxi_model::scratch::put_bf16(hc);
                    a = b;
                }
                k.stream
                    .memcpy_dtod(&hyper, &mut sb.hyper_rows.slice_mut(0.. t * self.dim))?;
                sb.argmax = self.head_argmax_all(eng, &hyper, t, head_alt)?;
                let pending = sb.argmax[t - 1] as i64;
                lingxi_model::scratch::put_bf16(hyper);
                self.ids.push(pending);
                }
            }
        }
        Ok(())
    }

    /// **回到空上下文**（跨请求复用同一 Sequence，serve 用）：QSA 回卷计数到 0、
    /// GDN 的 S 与卷积史、PLE 的卷积史全部清零、已喂 `ids` 清空。
    /// 权重/设备副本不受影响。归零后等价于刚 `new_split` 的状态（初值本来就全 0）。
    pub fn reset(&mut self, eng: &Engine) -> Result<()> {
        for i in 0..eng.n_layers as usize {
            // 状态在 split 下建在 layer 所属的卡上，清它也用同一张卡
            let k = eng.k_of(i as u32);
            if let AttnState::Qsa(q) = &mut self.states[i].attn {
                q.rewind(0);
            }
            if let AttnState::Gdn(g) = &mut self.states[i].attn {
                k.stream.memset_zeros(&mut g.k_hist)?;
                k.stream.memset_zeros(&mut g.k_hist_next)?;
                k.stream.memset_zeros(&mut g.s_all)?;
            }
            if let Some(p) = &mut self.states[i].ple {
                k.stream.memset_zeros(&mut p.hist)?;
                k.stream.memset_zeros(&mut p.hist_next)?;
                p.hist_len = 0;
            }
        }
        // EP：第二卡的状态做同样的清零（两卡状态必须同步）
        if let (Some(epx), Some(states_ep)) = (eng.ep, self.states_ep.as_mut()) {
            for st in states_ep.iter_mut() {
                if let AttnState::Qsa(q) = &mut st.attn {
                    q.rewind(0);
                }
                if let AttnState::Gdn(g) = &mut st.attn {
                    epx.k1.stream.memset_zeros(&mut g.k_hist)?;
                    epx.k1.stream.memset_zeros(&mut g.k_hist_next)?;
                    epx.k1.stream.memset_zeros(&mut g.s_all)?;
                }
                if let Some(p) = &mut st.ple {
                    epx.k1.stream.memset_zeros(&mut p.hist)?;
                    epx.k1.stream.memset_zeros(&mut p.hist_next)?;
                    p.hist_len = 0;
                }
            }
        }
        self.ids.clear();
        Ok(())
    }

    // ───────────────────────── 跨请求状态检查点（智能前缀缓存） ─────────────────────────

    /// **保存一份可恢复的状态检查点**（智能前缀缓存用）。
    ///
    /// 语义：`len` 个 token 已经喂进 KV；QSA 只需记长度（append-only，回卷即丢弃尾巴），
    /// GDN 的 `s_all/k_hist/k_hist_next` 与 PLE 的 `hist/hist_next` 必须**真拷贝**。
    /// 一份约 115 MB（36 个 GDN 层 × 3.1 MB + 小历史），serve 里只留 2~3 份。
    pub fn save_checkpoint(&self, eng: &Engine, len: usize) -> Result<StateCheckpoint> {
        anyhow::ensure!(len <= self.ids.len(), "checkpoint len={len} > ids={}", self.ids.len());
        // R2 检查点统一（2026-10-21）：dev 检查点只存 GDN/PLE，QSA 靠"KV 还在设备上"的
        // 隐含假设——paged 下 reset 的 rewind(0) 会归还块表/槽/host 坑，假设不成立。
        // restore 侧已响亮拒绝（restore_checkpoint），这里把**创建侧**也封死：
        // 不变量在两头都有栅栏，不靠调用方分派自觉。
        for s in &self.states {
            if let AttnState::Qsa(q) = &s.attn {
                anyhow::ensure!(
                    !q.paged,
                    "save_checkpoint(dev) 与 QSA 分页不兼容（dev 档不含 QSA KV，paged 的 \
                     reset 又会释放块表）。请走 save_host_checkpoint / save_ckpt_auto"
                );
            }
        }
        let n = eng.n_layers as usize;
        let mut gdn = Vec::with_capacity(n);
        let mut ple = Vec::with_capacity(n);
        for i in 0..n {
            let k = eng.k_of(i as u32);
            match &self.states[i].attn {
                AttnState::Gdn(g) => {
                    let mut sv = lingxi_model::scratch::f32(k, g.s_all.len())?;
                    let mut sh = lingxi_model::scratch::bf16(k, g.k_hist.len())?;
                    let mut shn = lingxi_model::scratch::bf16(k, g.k_hist_next.len())?;
                    k.stream.memcpy_dtod(&g.s_all, &mut sv)?;
                    k.stream.memcpy_dtod(&g.k_hist, &mut sh)?;
                    k.stream.memcpy_dtod(&g.k_hist_next, &mut shn)?;
                    gdn.push(Some((sv, sh, shn)));
                }
                _ => gdn.push(None),
            }
            match &self.states[i].ple {
                Some(p) => {
                    let mut ph = lingxi_model::scratch::bf16(k, p.hist.len())?;
                    let mut phn = lingxi_model::scratch::bf16(k, p.hist_next.len())?;
                    k.stream.memcpy_dtod(&p.hist, &mut ph)?;
                    k.stream.memcpy_dtod(&p.hist_next, &mut phn)?;
                    ple.push(Some((ph, phn)));
                }
                None => ple.push(None),
            }
        }
        Ok(StateCheckpoint {
            ids: self.ids[..len].to_vec(),
            gdn,
            ple,
        })
    }

    /// 恢复到检查点：QSA 回卷、GDN/PLE 拷回、`ids` 设成检查点里的 token 列表。
    /// **调用前请先 `reset`**（否则旧状态里超出 `len` 的部分不会被清理，虽然 QSA 会忽略）。
    pub fn restore_checkpoint(&mut self, eng: &Engine, cp: &StateCheckpoint) -> Result<()> {
        let n = eng.n_layers as usize;
        anyhow::ensure!(
            cp.gdn.len() == n && cp.ple.len() == n,
            "checkpoint 层数不匹配：{} / {} vs {n}",
            cp.gdn.len(),
            cp.ple.len()
        );
        let len = cp.ids.len();
        for i in 0..n {
            let k = eng.k_of(i as u32);
            if let AttnState::Qsa(q) = &mut self.states[i].attn {
                // P2 分页：dev 检查点只存 GDN/PLE，QSA 靠"KV 还在设备上"的隐含假设——
                // 但 paged 的 rewind(0)（reset）已把块表/槽/host 坑全部归还，恢复出来必是
                // 垃圾（2026-10-20 serve+paged 前缀缓存命中后输出退化实测）。响亮拒绝。
                anyhow::ensure!(
                    !q.paged,
                    "restore_checkpoint(dev) 与 QSA 分页不兼容：块表已在 reset 时释放。\
                     paged 模式请走 HostCheckpoint（serve 前缀缓存会自动落 Disk 档）"
                );
                q.rewind(len);
            }
            if let (Some((sv, sh, shn)), AttnState::Gdn(g)) =
                (cp.gdn[i].as_ref(), &mut self.states[i].attn)
            {
                k.stream.memcpy_dtod(sv, &mut g.s_all)?;
                k.stream.memcpy_dtod(sh, &mut g.k_hist)?;
                k.stream.memcpy_dtod(shn, &mut g.k_hist_next)?;
            }
            if let (Some((ph, phn)), Some(p)) = (cp.ple[i].as_ref(), self.states[i].ple.as_mut()) {
                k.stream.memcpy_dtod(ph, &mut p.hist)?;
                k.stream.memcpy_dtod(phn, &mut p.hist_next)?;
            }
        }
        self.ids.clear();
        self.ids.extend_from_slice(&cp.ids);
        Ok(())
    }

    /// **把整段状态搬到 host**（含 QSA KV/indexer，见 [`HostCheckpoint`]）。
    ///
    /// 只在“会话换出”这种冷路径调用：一份 256K 上下文约 7~9 GB，D2H 约 1 秒；
    /// 换来的是别的请求随便覆盖设备 KV 也不会把这份缓存弄坏。
    pub fn save_host_checkpoint(&self, eng: &Engine, len: usize) -> Result<HostCheckpoint> {
        anyhow::ensure!(len <= self.ids.len(), "checkpoint len={len} > ids={}", self.ids.len());
        let n = eng.n_layers as usize;
        let mut gdn = Vec::with_capacity(n);
        let mut ple = Vec::with_capacity(n);
        let mut qsa = Vec::with_capacity(n);
        let mut bytes = 0usize;
        let want_raw = lingxi_model::flag("LINGXI_QSA_HOST");
        for i in 0..n {
            let k = eng.k_of(i as u32);
            match &self.states[i].attn {
                AttnState::Gdn(g) => {
                    let sv = d2h_f32(k, &g.s_all, g.s_all.len())?;
                    let sh = d2h_u16(k, &g.k_hist, g.k_hist.len())?;
                    let shn = d2h_u16(k, &g.k_hist_next, g.k_hist_next.len())?;
                    bytes += sv.len() * 4 + (sh.len() + shn.len()) * 2;
                    gdn.push(Some((sv, sh, shn)));
                    qsa.push(None);
                }
                AttnState::Qsa(q) => {
                    gdn.push(None);
                    let (kn, v) = match (&q.kn, &q.v) {
                        (Some(kn), Some(v)) => (kn, v),
                        _ => anyhow::bail!("host 检查点不支持 K8V4（LINGXI_KV_K8V4=1）"),
                    };
                    anyhow::ensure!(q.cap > 0, "QSA cap=0");
                    // P2 分页：kv_dim/idx_dim 以 QsaState 记录为准（kn.len/cap 在分页下
                    // 是窗口口径，不再等于行宽）；KV 按块表装配逻辑行。
                    let kv_dim = q.kv_dim;
                    let idx_dim = q.idx_head_dim;
                    let rows = q.len;
                    let (kn_h, v_h) = if rows == 0 {
                        (Vec::new(), Vec::new())
                    } else if q.paged {
                        q.paged_kv_to_host(k, rows)?
                    } else {
                        (d2h_u16(k, kn, rows * kv_dim)?, d2h_u16(k, v, rows * kv_dim)?)
                    };
                    let rk_h = if rows == 0 { Vec::new() } else { d2h_u16(k, &q.rk_dev, rows * idx_dim)? };
                    let pool_h = if q.nblocks == 0 {
                        Vec::new()
                    } else {
                        d2h_u16(k, &q.pooled, q.nblocks * idx_dim)?
                    };
                    let raw_h = if want_raw {
                        let take = (rows * idx_dim).min(q.raw_keys_host.len());
                        q.raw_keys_host[..take].to_vec()
                    } else {
                        Vec::new()
                    };
                    bytes += (kn_h.len() + v_h.len() + rk_h.len() + pool_h.len()) * 2
                        + raw_h.len() * 4;
                    qsa.push(Some(QsaHostKv {
                        len: rows,
                        nblocks: q.nblocks,
                        kv_dim,
                        idx_dim,
                        kn: kn_h,
                        v: v_h,
                        rk: rk_h,
                        pooled: pool_h,
                        raw: raw_h,
                    }));
                }
            }
            match &self.states[i].ple {
                Some(p) => {
                    let ph = d2h_u16(k, &p.hist, p.hist.len())?;
                    let phn = d2h_u16(k, &p.hist_next, p.hist_next.len())?;
                    bytes += (ph.len() + phn.len()) * 2;
                    ple.push(Some((ph, phn)));
                }
                None => ple.push(None),
            }
        }
        Ok(HostCheckpoint { ids: self.ids[..len].to_vec(), bytes, gdn, ple, qsa, alts: Vec::new() })
    }

    /// 从 host 检查点恢复**到指定前缀长度**（调用前先 `reset`）。
    ///
    /// `len` 必须等于主长度或某个 `HostAlt.len`；KV 总是整段写回（更长的尾巴不读）。
    pub fn restore_host_checkpoint_at(
        &mut self,
        eng: &Engine,
        cp: &HostCheckpoint,
        len: usize,
    ) -> Result<()> {
        anyhow::ensure!(len <= cp.ids.len(), "恢复长度 {len} > 检查点 {}", cp.ids.len());
        let (gdn, ple) = if len == cp.ids.len() {
            (&cp.gdn, &cp.ple)
        } else {
            let alt = cp
                .alts
                .iter()
                .find(|a| a.len == len)
                .ok_or_else(|| anyhow::anyhow!("host 检查点没有长度 {len} 的状态档"))?;
            (&alt.gdn, &alt.ple)
        };
        let n = eng.n_layers as usize;
        anyhow::ensure!(
            gdn.len() == n && ple.len() == n && cp.qsa.len() == n,
            "host checkpoint 层数不匹配：{} / {} / {} vs {n}",
            gdn.len(),
            ple.len(),
            cp.qsa.len()
        );
        let want_raw = lingxi_model::flag("LINGXI_QSA_HOST");
        for i in 0..n {
            let k = eng.k_of(i as u32);
            if let (Some((sv, sh, shn)), AttnState::Gdn(g)) = (gdn[i].as_ref(), &mut self.states[i].attn)
            {
                h2d_f32(k, &g.s_all, sv)?;
                h2d_u16(k, &g.k_hist, sh)?;
                h2d_u16(k, &g.k_hist_next, shn)?;
            }
            if let (Some((ph, phn)), Some(p)) = (ple[i].as_ref(), self.states[i].ple.as_mut()) {
                h2d_u16(k, &p.hist, ph)?;
                h2d_u16(k, &p.hist_next, phn)?;
            }
            if let (Some(qh), AttnState::Qsa(q)) =
                (cp.qsa[i].as_ref(), &mut self.states[i].attn)
            {
                anyhow::ensure!(len <= qh.len, "恢复长度 {len} > 检查点 QSA 长度 {}", qh.len);
                // ㊿-P1 弹性 KV：懒分配下恢复前先把池子扩到检查点长度（几何增长，冷路径）。
                // 必须先扩再借 kn/v（ensure_cap 换绑缓冲，借用会拦住 &mut）。
                q.ensure_cap(k, qh.len)?;
                let (kn, v) = match (&q.kn, &q.v) {
                    (Some(kn), Some(v)) => (kn, v),
                    _ => anyhow::bail!("host checkpoint 不支持 K8V4（LINGXI_KV_K8V4=1）"),
                };
                // P2 分页：形状以 QsaState 记录为准；KV 整体进 host 块池（表项置下沉，
                // 不碰 GPU 槽），rk/pooled 照旧直写。
                anyhow::ensure!(
                    qh.kv_dim == q.kv_dim && qh.idx_dim == q.idx_head_dim,
                    "host checkpoint 的 QSA 形状不匹配"
                );
                if q.paged {
                    q.paged_kv_from_host(k, &qh.kn, &qh.v)?;
                } else {
                    h2d_u16(k, kn, &qh.kn)?;
                    h2d_u16(k, v, &qh.v)?;
                }
                h2d_u16(k, &q.rk_dev, &qh.rk)?;
                h2d_u16(k, &q.pooled, &qh.pooled)?;
                q.len = qh.len;
                q.nblocks = qh.nblocks;
                if want_raw && !qh.raw.is_empty() {
                    q.raw_keys_host.clear();
                    q.raw_keys_host.extend_from_slice(&qh.raw);
                }
                q.rewind(len);
            }
        }
        self.ids.clear();
        self.ids.extend_from_slice(&cp.ids[..len]);
        Ok(())
    }

    /// **回卷到本轮快照**（`spec_snapshot` 打下的位置）—— A' 的"整轮重算"就靠它。
    ///
    /// 复用 [`Self::spec_commit`] 传 `accept = 0` 的三件套（QSA 计数回退到 `kv_len`、
    /// GDN 从快照拷回 `s_all/k_hist/k_hist_next`、PLE 卷积历史拷回），
    /// 只额外把 `ids` 截回 `kv_len + 1` —— `spec_chunk` 会把它推到 `kv_len + 1 + t`，
    /// 而 `spec_commit(0)` 不碰 `ids`。
    ///
    /// 判据见 `LINGXI_SPEC_SELFTEST=1`：回卷后拿同一批 token 重跑 `spec_chunk`，
    /// argmax 必须逐位相同（这条不通，A' 的乐观执行就没有回退可依）。
    pub fn spec_restore(
        &mut self,
        eng: &Engine,
        provider: &mut dyn LayerProvider,
        sb: &mut SpecBuffers,
    ) -> Result<()> {
        let kv_len = sb.kv_len;
        self.spec_commit(eng, provider, sb, 0)?;
        anyhow::ensure!(
            self.ids.len() >= kv_len + 1,
            "spec_restore: ids 比快照还短（{} < {}）",
            self.ids.len(),
            kv_len + 1
        );
        self.ids.truncate(kv_len + 1);
        Ok(())
    }

    /// **只保留前 `accept` 个已喂 token**：QSA 回卷计数、GDN 恢复快照并**重放**被接受前缀、
    /// 同步 `ids`（尾元素换成第 `accept-1` 行的 argmax，即"奖励 token"）。
    pub fn spec_commit(
        &mut self,
        eng: &Engine,
        provider: &mut dyn LayerProvider,
        sb: &mut SpecBuffers,
        accept: usize,
    ) -> Result<()> {
        let target = sb.kv_len + accept;
        // K3（DeepInfer 借鉴，2026-10-21）：**全接受 = 零回滚快速路径**。
        // 递推是逐行串行的，全接受时批后态**精确等于**串行推到 target 的状态
        // （lingxi 自己的硬判据「全接受时提交态 == 批后态」LINGXI_SPEC_CHECK 已认证，
        // fast.bin == postbatch 逐字节同）⇒ 快照恢复 + 重放整体跳过。
        // 这与 DeepInfer 的「乐观推进 + 按接受数回填」同源：QSA 侧 lingxi 本来就
        // 是 append-only 只改计数（rewind 即回填）；GDN 侧全接受时连恢复都不需要。
        // 部分接受仍走下面的恢复+重放（循环状态的固有成本，无法覆写）。
        let full_accept = accept == sb.argmax.len();
        for i in 0..eng.n_layers as usize {
            // ★ 全量 layer-split：状态在哪张卡，重放就用哪张卡的上下文
            let k = eng.k_of(i as u32);
            // QSA：只要改计数
            if let AttnState::Qsa(q) = &mut self.states[i].attn {
                q.rewind(target);
                continue;
            }
            if full_accept {
                continue; // 全接受：批后态即正确态，零回滚（见上）
            }
            // GDN：恢复快照 + 用录下的输入重放 accept 个 token
            let Some((sv, sh, shn)) = sb.gdn_snap[i].as_mut() else { continue };
            // 一次拿出 attn/ple 两个**不相交字段**的可变借用（分两次索引 self.states 会被拒）
            let st = &mut self.states[i];
            let (attn_slot, mut ple_slot) = (&mut st.attn, st.ple.as_mut());
            let AttnState::Gdn(g) = attn_slot else { continue };
            k.stream.memcpy_dtod(sv, &mut g.s_all)?;
            k.stream.memcpy_dtod(sh, &mut g.k_hist)?;
            k.stream.memcpy_dtod(shn, &mut g.k_hist_next)?;
            // 诊断：把**快照恢复之后**（= 重放的起点）的状态也 dump 一份 —— 补上二分链的最后一环。
            // 若它在飘 ⇒ 问题在快照/恢复；若它稳定而"提交后"在飘 ⇒ 问题在重放那几颗核。
            dump_gdn_state(k, i, &g.s_all, &g.k_hist, "restored")?;
            dump_gdn_full(k, i, &g.s_all, "restored")?;
            // ⚠️ PLE 卷积历史的**恢复要无条件做**（它也是跨 token 状态）。
            // 之前把它写在 `if accept > 0` 里 ⇒ `commit(0)` 时 PLE 停在"喂过 t 个"的状态，
            // 于是"全拒"还原不回批前 —— 判据 `LINGXI_SPEC_CHECK` 正是这么抓出来的。
            if let (Some(ps), Some(pst)) =
                (sb.ple_snap[i].as_mut(), ple_slot.as_mut().map(|p| &mut **p))
            {
                k.stream.memcpy_dtod(ps, &mut pst.hist)?;
            }
            if accept > 0 {
                let loaded = provider.layer(i as u32)?;
                let s = eng.shape;
                let AttnWeights::Gdn(gw) = &loaded.weights.attn else {
                    bail!("层 {i} 号称为 GDN 但权重不是 Gdn");
                };
                // ★ **只重跑「卷积 + 递推」两个小核**：`gdn_step` 的三步
                // （投影 / 卷积 / 递推）全是 `mixed` 的纯函数，而投影的输出（`qkv/z/a/b`）
                // 验证明时已录下来了 ⇒ 重放时**跳过投影那笔最大的权重读**（~30 MB/层）。
                // 比"逐行快照整个 GDN 状态"（500 MB）省 99% 内存，也不用改 `.cu`。
                // **位等价**：卷积因果、递推行串行 ⇒ 用同样字节的前 `accept` 行重跑，
                // `cout[0..accept]` 与批量时逐位相同 ⇒ 状态逐位相同。
                // ⚠️ PLE 自己的**卷积历史**仍要推进（它是跨 token 状态、喂给下一轮）。
                // ★ **默认走快速重放**（`LINGXI_GDN_REPLAY_LEGACY=1` 一键回退到"重算"老路）。
                // 依据：快速重放喂给递推核的是**批处理当时的投影字节**
                // ⇒ 前 `accept` 行的 `cout/z/a/b` 与批量逐位相同 ⇒ 状态精确等于批量的行串行结果；
                // 而"全接受(`keep==t`)时提交态必须 == 批后态"这条硬判据，实测**通过**
                // （整块拷贝读 + 逐字节比：`fast.bin == postbatch`，同二进制 3 跑全同）。
                // 老路是**按 `t=accept` 重算投影**，没有任何证据能过这条判据。
                if !lingxi_model::flag("LINGXI_GDN_REPLAY_LEGACY")
                    && sb.gdn_rec_ok
                    && sb.gdn_qkv[i].is_some()
                {
                    if let Some(pw) = loaded.weights.ple.as_ref() {
                        let rec_in = sb.gdn_in[i].as_ref().unwrap();
                        let mut xr = lingxi_model::scratch::bf16(k, accept * self.dim)?;
                        k.stream
                            .memcpy_dtod(&rec_in.slice(0..accept * self.dim), &mut xr)?;
                        let pst = ple_slot.take().expect("PLE 层必须带 PleState");
                        let ple_out = lingxi_model::plet::ple_step(
                            k, pw, &xr, &self.ids[..target], accept, pst,
                        )?;
                        lingxi_model::scratch::put_bf16(ple_out);
                        lingxi_model::scratch::put_bf16(xr);
                    }
                    // `x` 在重放模式下**不被使用**（投影被跳过）⇒ 给个等长占位即可
                    let x_unused = lingxi_model::scratch::bf16(k, accept * self.hidden)?;
                    let rec = build_gdn_replay_rec(sb, i);
                    // ★ 诊断（`LINGXI_GDN_REPLAY_NOPOOL=1`）：整段重放**绕开 scratch 池**。
                    // 用途：判定"同长度缓冲在池里被复用"是不是那处 run-to-run 竞态的元凶
                    // （(c)：录制确定、重放输出在飘）。
                    let mut gdn_once = |rc: Option<lingxi_model::gdn::GdnRec<'_>>| {
                        lingxi_model::gdn::gdn_step_rec(k, gw, &x_unused, accept, &s.gdn, g, rc)
                    };
                    let res = if lingxi_model::flag("LINGXI_GDN_REPLAY_NOPOOL") {
                        lingxi_model::scratch::without_pool(|| gdn_once(rec))
                    } else {
                        gdn_once(rec)
                    };
                    let _ = res?;
                    lingxi_model::scratch::put_bf16(x_unused);
                    dump_gdn_state(k, i, &g.s_all, &g.k_hist, "fast")?;
                    dump_gdn_full(k, i, &g.s_all, "fast")?;
                } else {
                    let mixed: CudaSlice<bf16> = {
                    // 退路（`LINGXI_SPEC_PREP_NOALLOC` 等）：老做法——从层输入重算 PLE + attn_gr
                    let rec = sb.gdn_in[i].as_ref().unwrap();
                    let mut xr = lingxi_model::scratch::bf16(k, accept * self.dim)?;
                    k.stream
                        .memcpy_dtod(&rec.slice(0..accept * self.dim), &mut xr)?;
                    let mut attn_in = xr;
                    if let Some(pw) = loaded.weights.ple.as_ref() {
                        let pst = ple_slot.take().expect("PLE 层必须带 PleState");
                        let ple_out = lingxi_model::plet::ple_step(
                            k, pw, &attn_in, &self.ids[..target], accept, pst,
                        )?;
                        let mut summed = lingxi_model::scratch::bf16(k, accept * self.dim)?;
                        k.add_bf16(&attn_in, &ple_out, &mut summed, accept * self.dim)?;
                        lingxi_model::scratch::put_bf16(ple_out);
                        lingxi_model::scratch::put_bf16(attn_in);
                        attn_in = summed;
                    }
                    // ⚠️ **必须和批量走同一套调用**（`_into` + 同一个 `gbase` 私有固定区）。
                    // 原来这里用的是分配版 `gated_residual_forward`（`gkey = NO_CAPTURE`），
                    // 与批量（`gbase+1` 命名空间 + `graph::take_fixed` 的固定缓冲）**不是同一套**，
                    // 于是重算出来的 `mixed` 与批量的**不逐位相同** ⇒ 全接受（`keep == t`）时
                    // "提交后的状态"复现不了"批处理后的状态"。
                    let gbase = (i as u64) * 8;
                    let mut mixed = lingxi_model::graph::take_fixed(k, gbase, 0, accept * self.hidden)?;
                    let mut injw =
                        lingxi_model::graph::take_fixed(k, gbase, 1, accept * s.hc)?;
                    lingxi_model::gr::gated_residual_forward_into(
                        k,
                        &loaded.weights.attn_gr,
                        &attn_in,
                        &mut mixed,
                        Some(&mut injw),
                        accept as u32,
                        self.dim,
                        self.hidden,
                        s.hc,
                        s.lowrank,
                        s.eps,
                        gbase + 1,
                    )?;
                    lingxi_model::graph::give_fixed(k, gbase, 1, injw);
                    lingxi_model::scratch::put_bf16(attn_in);
                    mixed
                    };
                    let _ = lingxi_model::gdn::gdn_step(k, gw, &mixed, accept, &s.gdn, g)?;
                    lingxi_model::graph::give_fixed(k, (i as u64) * 8, 0, mixed);
                    dump_gdn_state(k, i, &g.s_all, &g.k_hist, "legacy")?;
                    dump_gdn_full(k, i, &g.s_all, "legacy")?;
                }
            }
        }
        // ★ **EP**：第二张卡的状态做**完全相同的**回卷/重放。
        // 两卡的 hyper 与状态在批处理时逐位相同 ⇒ 用同样的 `accept`/快照/录像重放，
        // 提交后两卡仍然逐位相同（下一轮不需要任何跨卡状态同步）。
        if let (Some(epx), Some(side)) = (eng.ep, sb.ep_side.as_mut()) {
            let k1 = epx.k1;
            let s = eng.shape;
            let states_ep = self.states_ep.as_mut().expect("EP 模式必须有 states_ep");
            for i in 0..eng.n_layers as usize {
                if let AttnState::Qsa(q) = &mut states_ep[i].attn {
                    q.rewind(target);
                    continue;
                }
                if full_accept {
                    continue;
                }
                let Some((sv, sh, shn)) = side.gdn_snap[i].as_mut() else { continue };
                let st = &mut states_ep[i];
                let (attn_slot, mut ple_slot) = (&mut st.attn, st.ple.as_mut());
                let AttnState::Gdn(g) = attn_slot else { continue };
                k1.stream.memcpy_dtod(sv, &mut g.s_all)?;
                k1.stream.memcpy_dtod(sh, &mut g.k_hist)?;
                k1.stream.memcpy_dtod(shn, &mut g.k_hist_next)?;
                if let (Some(ps), Some(pst)) =
                    (side.ple_snap[i].as_mut(), ple_slot.as_mut().map(|p| &mut **p))
                {
                    k1.stream.memcpy_dtod(ps, &mut pst.hist)?;
                }
                if accept == 0 {
                    continue;
                }
                let loaded = epx
                    .layers1
                    .peek(i as u32)
                    .ok_or_else(|| anyhow::anyhow!("[ep] GPU1 层 {i} 未预热，重放失败"))?;
                let AttnWeights::Gdn(gw) = &loaded.weights.attn else {
                    bail!("层 {i} 号称为 GDN 但权重不是 Gdn（EP 镜像）");
                };
                if !lingxi_model::flag("LINGXI_GDN_REPLAY_LEGACY")
                    && sb.gdn_rec_ok
                    && side.gdn_qkv[i].is_some()
                {
                    // 快速重放：用 card1 自己录下的投影字节（跳过投影的权重读）
                    if let Some(pw) = loaded.weights.ple.as_ref() {
                        let rec_in = side.gdn_in[i].as_ref().expect("录像缺失");
                        let mut xr = lingxi_model::scratch::bf16(k1, accept * self.dim)?;
                        k1.stream
                            .memcpy_dtod(&rec_in.slice(0..accept * self.dim), &mut xr)?;
                        let pst = ple_slot.take().expect("PLE 层必须带 PleState");
                        let ple_out =
                            lingxi_model::plet::ple_step(k1, pw, &xr, &self.ids[..target], accept, pst)?;
                        lingxi_model::scratch::put_bf16(ple_out);
                        lingxi_model::scratch::put_bf16(xr);
                    }
                    let x_unused = lingxi_model::scratch::bf16(k1, accept * self.hidden)?;
                    let rec = build_gdn_replay_rec_fields(
                        &mut side.gdn_qkv,
                        &mut side.gdn_z,
                        &mut side.gdn_a,
                        &mut side.gdn_b,
                        i,
                    );
                    let res = lingxi_model::gdn::gdn_step_rec(
                        k1, gw, &x_unused, accept, &s.gdn, g, rec,
                    );
                    res?;
                    lingxi_model::scratch::put_bf16(x_unused);
                } else {
                    // 退路：从 card1 录下的层输入重算 PLE + attn_GR，再跑 gdn_step
                    let rec_in = side.gdn_in[i].as_ref().expect("录像缺失");
                    let mut attn_in = lingxi_model::scratch::bf16(k1, accept * self.dim)?;
                    k1.stream
                        .memcpy_dtod(&rec_in.slice(0..accept * self.dim), &mut attn_in)?;
                    if let Some(pw) = loaded.weights.ple.as_ref() {
                        let pst = ple_slot.take().expect("PLE 层必须带 PleState");
                        let ple_out = lingxi_model::plet::ple_step(
                            k1, pw, &attn_in, &self.ids[..target], accept, pst,
                        )?;
                        let mut summed = lingxi_model::scratch::bf16(k1, accept * self.dim)?;
                        k1.add_bf16(&attn_in, &ple_out, &mut summed, accept * self.dim)?;
                        lingxi_model::scratch::put_bf16(ple_out);
                        lingxi_model::scratch::put_bf16(attn_in);
                        attn_in = summed;
                    }
                    let gbase = (i as u64) * 8;
                    let mut mixed =
                        lingxi_model::graph::take_fixed(k1, gbase, 0, accept * self.hidden)?;
                    let mut injw =
                        lingxi_model::graph::take_fixed(k1, gbase, 1, accept * s.hc)?;
                    lingxi_model::gr::gated_residual_forward_into(
                        k1,
                        &loaded.weights.attn_gr,
                        &attn_in,
                        &mut mixed,
                        Some(&mut injw),
                        accept as u32,
                        self.dim,
                        self.hidden,
                        s.hc,
                        s.lowrank,
                        s.eps,
                        gbase + 1,
                    )?;
                    lingxi_model::graph::give_fixed(k1, gbase, 1, injw);
                    lingxi_model::scratch::put_bf16(attn_in);
                    lingxi_model::gdn::gdn_step(k1, gw, &mixed, accept, &s.gdn, g)?;
                    lingxi_model::graph::give_fixed(k1, gbase, 0, mixed);
                }
            }
        }
        // ids：前 sb.kv_len 个是旧前缀，接着是 tokens[0..accept]，最后是 pending：
        // accept>0 ⇒ 奖励 token = 第 accept-1 行的 argmax（只依赖被接受前缀，因果保证）
        // accept=0 ⇒ 全拒，pending 还原成本批喂进去的第一个 token
        let next = if accept > 0 { sb.argmax[accept - 1] as i64 } else { sb.pending };
        self.ids.truncate(target);
        self.ids.push(next);
        Ok(())
    }

    /// 验证批 head 的**词表行二分**。
    ///
    /// `lo` 半（全局行 `[0, vocab/2)`）在 GPU0、`hi` 半（`[vocab/2, vocab)`）在 GPU1
    /// （`alt.wq_hi` 是那张尾半权重，加载期已常驻 GPU1）。两半各做 part+fin argmax，
    /// 再按「值降序、下标升序」归并 —— 与整表 `argmax_rows_fin` 同一全序。
    ///
    /// **逐位依据**：`proj_q4_epi` 在 t≤8 且默认开关下走 `gemm_q4_skinny_bf16`（grid.y 行分块、
    /// **无 split-k**）⇒ 每一行的 k 归约次序只取决于该行自己的权重与 x，与词表行数无关。
    /// 平局：`m1 > m0` 才取 hi ⇒ 相等时取更小下标（必在 lo 半），与 fin 核口径一致。
    fn head_argmax_split(
        &self,
        k: &lingxi_kernels::Kernels,
        wq: &lingxi_model::q4::Q4W,
        alt: &crate::VerifyHeadAlt<'_>,
        mixed: &CudaSlice<bf16>,
        t: usize,
        vocab: usize,
    ) -> Result<Vec<u32>> {
        const G: usize = 64;
        anyhow::ensure!(vocab % 2 == 0 && vocab >= 2 * G, "词表 {vocab} 不能二分");
        anyhow::ensure!(
            wq.k_pad as usize == self.hidden && self.hidden % 8 == 0,
            "head 形状不支持二分（k_pad {}／hidden {}）",
            wq.k_pad,
            self.hidden
        );
        let half = vocab / 2;
        let k1 = alt.k1;
        // ① 激活过一次 PCIe（t×hidden×2 B ≈ 30 KB）；D2H 会排空 GPU0，但 mixed 两半都要用，
        // 所以这一刀本来就必须等 —— 它不额外增加串行度。
        let hv: Vec<bf16> = k.stream.clone_dtoh(mixed)?;
        let mut x1 = k1.stream.alloc_zeros::<bf16>(t * self.hidden)?;
        k1.stream.memcpy_htod(&hv, &mut x1)?;
        // ② 两半各发一发（两条 stream 各自排队 ⇒ 天然并行，中间不 sync）
        let mut y_hi = k1.stream.alloc_zeros::<bf16>(t * half)?;
        lingxi_model::gpu::proj_q4_epi(
            k1, &x1, alt.wq_hi, &mut y_hi, half, self.hidden, 0, t, 0, 0.0, 0.0,
        )?;
        let mut y_lo = k.stream.alloc_zeros::<bf16>(t * half)?;
        lingxi_model::gpu::proj_q4_epi(
            k, mixed, wq, &mut y_lo, half, self.hidden, 0, t, 0, 0.0, 0.0,
        )?;
        // ③ 两半各自 part+fin
        let mut pv0 = lingxi_model::scratch::f32(k, t * G)?;
        let mut pi0 = lingxi_model::scratch::i32(k, t * G)?;
        let mut am0 = lingxi_model::scratch::i32(k, t)?;
        k.argmax_rows_part_bf16(&y_lo, 0, &mut pv0, &mut pi0, t, half, G)?;
        k.argmax_rows_fin(&pv0, &pi0, &mut am0, t, G)?;
        let mut pv1 = lingxi_model::scratch::f32(k1, t * G)?;
        let mut pi1 = lingxi_model::scratch::i32(k1, t * G)?;
        let mut am1 = lingxi_model::scratch::i32(k1, t)?;
        k1.argmax_rows_part_bf16(&y_hi, 0, &mut pv1, &mut pi1, t, half, G)?;
        k1.argmax_rows_fin(&pv1, &pi1, &mut am1, t, G)?;
        // ④ 回传：半边的最大 = 它那 G 个 chunk 最大值的最大值（part 核的口径）
        let v0: Vec<f32> = k.stream.clone_dtoh(&pv0)?;
        let a0: Vec<i32> = k.stream.clone_dtoh(&am0)?;
        let v1: Vec<f32> = k1.stream.clone_dtoh(&pv1)?;
        let a1: Vec<i32> = k1.stream.clone_dtoh(&am1)?;
        lingxi_model::scratch::put_f32(pv0);
        lingxi_model::scratch::put_i32(pi0);
        lingxi_model::scratch::put_i32(am0);
        lingxi_model::scratch::put_f32(pv1);
        lingxi_model::scratch::put_i32(pi1);
        lingxi_model::scratch::put_i32(am1);
        let mut out = Vec::with_capacity(t);
        for i in 0..t {
            let s0 = i * G;
            let s1 = (i + 1) * G;
            let m0 = v0[s0..s1].iter().fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a });
            let m1 = v1[s0..s1].iter().fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a });
            let idx = if m1 > m0 { a1[i] + half as i32 } else { a0[i] };
            anyhow::ensure!(idx >= 0 && (idx as usize) < vocab, "head 二分给出越界下标 {idx}");
            out.push(idx as u32);
        }
        Ok(out)
    }

    /// 所有行的贪心 argmax（`head_argmax` 的多行版）。
    fn head_argmax_all(
        &self,
        eng: &Engine,
        hyper: &CudaSlice<bf16>,
        t: usize,
        alt: Option<&crate::VerifyHeadAlt<'_>>,
    ) -> Result<Vec<u32>> {
        let k = eng.k;
        let s = eng.shape;
        let (mixed, injw) = lingxi_model::gr::gated_residual_forward(
            k, &eng.endpoints.mixer, hyper, t as u32, self.dim, self.hidden, s.hc, s.lowrank, s.eps,
        )?;
        if let Some(j) = injw {
            lingxi_model::scratch::put_bf16(j);
        }
        // `LINGXI_VERIFY_HEAD_SPLIT` 走分卡时不分配整表 logits（省 3 MB 往返）。
        let mut logits_slot: Option<CudaSlice<bf16>> = None;
        if let Some(wq) = &eng.endpoints.output_head_q4 {
            // ★ 2026-10-25`LINGXI_VERIFY_HEAD_SPLIT`：词表行二分到两卡。
            // GPU1 这时空闲（刚跑完最后一层），两半各读 ~169 MB ⇒ verify 段少 ~0.9 ms。
            if let Some(a) = alt {
                match self.head_argmax_split(k, wq, a, &mixed, t, eng.vocab) {
                    Ok(v) => {
                        lingxi_model::scratch::put_bf16(mixed);
                        return Ok(v);
                    }
                    Err(e) => {
                        static W: std::sync::Once = std::sync::Once::new();
                        W.call_once(|| {
                            eprintln!("[verify-head-split] ⚠️ 分卡失败，回退单卡逐位: {e:#}")
                        });
                    }
                }
            }
            let mut lg = k.stream.alloc_zeros::<bf16>(t * eng.vocab)?;
            lingxi_model::gpu::proj_q4_epi(
                k, &mixed, wq, &mut lg, eng.vocab, self.hidden, 0, t, 0, 0.0, 0.0,
            )?;
            logits_slot = Some(lg);
        } else {
            let mut lg = k.stream.alloc_zeros::<bf16>(t * eng.vocab)?;
            lingxi_model::gpu::proj_t1(
                k, &mixed, &eng.endpoints.output_head, &mut lg, eng.vocab, self.hidden, t,
            )?;
            logits_slot = Some(lg);
        }
        let logits = logits_slot.expect("head logits 已算出");
        lingxi_model::scratch::put_bf16(mixed);
        // ★ P15-⑭（2026-10-17）：argmax 移进 GPU（两段式 part+fin），只 D2H t×4 B 的列下标。
        // 与 host `argmax_rows_bf16` 同一全序（f32 值降序、下标升序）⇒ 逐位一致（对拍台
        // `argmax-check` 验证，含手工平局/极端值用例）。省：锁页 D2H 4.5 MB（~0.4ms）
        // + host 遍历 t×248k（~1.5ms）。`LINGXI_HOST_ARGMAX=1` 回退老路做 A/B。
        if !lingxi_model::flag("LINGXI_HOST_ARGMAX") {
            const G: usize = 64;
            let mut pval = lingxi_model::scratch::f32(k, t * G)?;
            let mut pidx = lingxi_model::scratch::i32(k, t * G)?;
            let mut am = lingxi_model::scratch::i32(k, t)?;
            k.argmax_rows_part_bf16(&logits, 0, &mut pval, &mut pidx, t, eng.vocab, G)?;
            k.argmax_rows_fin(&pval, &pidx, &mut am, t, G)?;
            let am_h: Vec<i32> = k.stream.clone_dtoh(&am)?;
            lingxi_model::scratch::put_f32(pval);
            lingxi_model::scratch::put_i32(pidx);
            lingxi_model::scratch::put_i32(am);
            return Ok(am_h.iter().map(|&v| v as u32).collect());
        }
        // 2026-10-16：这两行原来是**可分页** D2H + `map(to_f32)` —— t×vocab bf16（t=9 时 4.5 MB）
        // 在 ~300 MB/s 的 bounce buffer 通道上要 10~22 ms，再加 9 MB f32 分配与 220 万次转换。
        // 而且它在**逐层 prof 括号之外** ⇒ 表现为"主干验证比 48 层合计多出 ~29 ms"。
        // 换锁页 D2H + 直接吃 bf16 的 argmax（判据不变，见 `argmax_rows_bf16` 的位等价说明）。
        let lh: Vec<bf16> = lingxi_model::scratch::download_bf16(k, &logits)?;
        Ok(lingxi_model::trunk::argmax_rows_bf16(&lh, t, eng.vocab))
    }

    /// 贪心生成：反复 [`Self::decode_token`]，直到产出 `eos` 或达到 `max_new_tokens`。
    /// 返回新生成的 token（不含 prompt，不含终止 eos）。
    pub fn generate(
        &mut self,
        eng: &Engine,
        max_new_tokens: usize,
        eos: i64,
        provider: &mut dyn LayerProvider,
    ) -> Result<Vec<i64>> {
        let mut out = Vec::new();
        for _ in 0..max_new_tokens {
            let token = self.decode_token(eng, provider)?;
            if token == eos {
                break;
            }
            out.push(token);
        }
        Ok(out)
    }
}

