//! QSA 行批选中准入（`kv_admit`）：把 indexer 选中的块集合按行批收进常驻窗口。

use anyhow::Result;
use cudarc::driver::CudaSlice;
use lingxi_kernels::Kernels;

use super::core_::QsaState;

impl QsaState {
    pub fn kv_admit(&mut self, k: &Kernels, sel: &CudaSlice<i32>, sel_cnt: &CudaSlice<i32>,
                    row0: usize, t: usize, topk: usize, wfrom: usize, wt: usize) -> Result<()> {
        if !self.paged {
            return Ok(());
        }
        if self.n_sunk == 0 {
            return Ok(()); // 无冷块 ⇒ 选中块必然常驻
        }
        use cudarc::driver::DevicePtr;
        self.step_ctr += 1;
        // sel 名单上 host：cnt ‖ sel 拷进一条 pinned 暂存，一次 DMA + 一次同步。
        let need = t + t * topk;
        if self.sel_pin.as_ref().map(|p| p.len()) < Some(need) {
            // 换缓冲前同步：防旧 pinned 被 Drop 时还有在途 DMA（同 host_pool_grow 的坑）。
            k.stream.synchronize()?;
            self.sel_pin = Some(unsafe {
                k.stream.context().alloc_pinned_with_flags::<i32>(need, 0)?
            });
        }
        let pin = self.sel_pin.as_mut().unwrap();
        {
            let src = sel_cnt.slice(row0..row0 + t);
            let (p, _g) = src.device_ptr(&k.stream);
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async::<i32>(
                    &mut pin.as_mut_slice()?[..t], p, k.stream.cu_stream())?
            };
        }
        {
            let src = sel.slice(row0 * topk..(row0 + t) * topk);
            let (p, _g) = src.device_ptr(&k.stream);
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async::<i32>(
                    &mut pin.as_mut_slice()?[t..need], p, k.stream.cu_stream())?
            };
        }
        k.stream.synchronize()?;
        let protect_full = wfrom / self.idx_compress; // 当前满块（写入中）不驱逐
        // 收集有效块号 → 排序去重（prefill t×topk 可达数十万，去重后远小）
        let mut blocks: Vec<i32> = Vec::new();
        {
            let psl = pin.as_slice()?;
            for i in 0..t {
                let n = psl[i].max(0) as usize;
                for j in 0..n.min(topk) {
                    let b = psl[t + i * topk + j];
                    if b >= 0 && (b as usize) < self.tab_host.len() {
                        blocks.push(b);
                    }
                }
            }
        }
        blocks.sort_unstable();
        blocks.dedup();
        if crate::flag("LINGXI_QSA_PAGED_NAN") {
            let presink = blocks.iter().filter(|&&b| self.tab_host[b as usize] == -1).count();
            eprintln!("[admit] t={t} len={wfrom} blocks={} 其中下沉={presink} n_sunk={}", blocks.len(), self.n_sunk);
        }
        // 第 1 遍：先把**所有**选中且常驻的块 touch 到本步——否则第 2 遍预取时的
        // 驱逐可能选中"本步要用、但还没轮到 touch"的常驻块 ⇒ gather 读到已下沉块。
        for &b in &blocks {
            let b = b as usize;
            if self.tab_host[b] >= 0 {
                self.blk_touch[b] = self.step_ctr;
            }
        }
        for b in blocks {
            let b = b as usize;
            if self.tab_host[b] == -1 {
                // 冷块命中：预取回 GPU（罕见；~2.8ms/块量级，同步等待可接受）
                self.block_prefetch(k, b, protect_full)?;
            }
        }
        let _ = wt;
        if crate::flag("LINGXI_QSA_PAGED_NAN") {
            eprintln!("[admit] 完成时 tab[:6]={:?} n_sunk={}", &self.tab_host[..6], self.n_sunk);
        }
        self.tab_flush(k)
    }

}
