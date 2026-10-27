//! 存档与检查点：SpecBuffers / StateCheckpoint / HostCheckpoint + 字节序列化助手。
//! CODE_MAP Phase 4 刀P4-2 从 lib.rs 搬入，只搬家不改行为。

use anyhow::Result;

use crate::Engine;
use cudarc::driver::CudaSlice;
use half::bf16;
use lingxi_kernels::Kernels;

pub struct SpecBuffers {
    /// 本批开始时的 KV 长度
    pub kv_len: usize,
    /// 每层：`Some` = GDN 的 (`s_all`, `k_hist`, `k_hist_next`) 备份
    pub(crate) gdn_snap: Vec<Option<(CudaSlice<f32>, CudaSlice<bf16>, CudaSlice<bf16>)>>,
    /// 每层：`Some` = 本批 GDN 的输入 `[t_max, dim]`（录像，供重放）
    pub(crate) gdn_in: Vec<Option<CudaSlice<bf16>>>,
    /// 每层：`Some` = **`gdn_step` 的内部输入录像**——投影的输出 `qkv/z/a/b`。
    /// 提交时只跑「卷积 + 递推」两个小核、**跳过投影那笔最大的权重读**（~30 MB/层）。
    /// 字节账：`t_max × (conv_dim + value_dim + 2·num_v_heads) × 2 B` ≈ 144 KB/层 ⇒ 36 层 ≈ **5.2 MB**。
    pub(crate) gdn_qkv: Vec<Option<CudaSlice<bf16>>>,
    pub(crate) gdn_z: Vec<Option<CudaSlice<bf16>>>,
    pub(crate) gdn_a: Vec<Option<CudaSlice<bf16>>>,
    pub(crate) gdn_b: Vec<Option<CudaSlice<bf16>>>,
    /// 每层：`Some` = PLE 卷积历史的备份（PLE 也是跨 token 可变状态）
    pub(crate) ple_snap: Vec<Option<CudaSlice<bf16>>>,
    // ───── **EP（`LINGXI_EP=1`）第二张卡的镜像**（全部在 GPU1 上，覆盖全部 n_layers 层）─────
    // 两卡的 hyper/状态逐位相同，但输入录像/快照必须是**各自设备**的缓冲，重放才能在本卡跑；
    // EP 关闭时这里是 `None`，旧路径一个字节都不碰。
    pub(crate) ep_side: Option<SpecSide>,
    /// **本轮是否真的录了上面那四个**（`spec_chunk` 写）。`false` 时 `spec_commit` 只能走
    /// 老路（从层输入重算 PLE + attn_GR + gdn_step）——`LINGXI_SPEC_NOREC=1` 会不录像，
    /// 而缓冲区**是复用的**（不看这个标记就会读到上一轮的陈旧数据）。
    pub(crate) gdn_rec_ok: bool,
    /// 本批每行的贪心 argmax（`spec_chunk` 写、`spec_commit` 读）
    pub argmax: Vec<u32>,
    /// 本批每行的**主干残差栈 h**（`[t_max, dim]`）—— MTP 草稿头要用它重放自己的 KV cell
    pub hyper_rows: CudaSlice<bf16>,
    /// 本批喂进去的第一个 token（= 批前的 pending）—— 全拒时要把它还原回去
    pub(crate) pending: i64,
    pub(crate) t_max: usize,
}

/// **EP 第二张卡**的一整套投机录像/快照缓冲（结构与 `SpecBuffers` 的前 7 个字段一一对应，
/// 全部在 GPU1 上）。为什么要它：部分接受时两张卡都要把状态退回并重放到接受前缀；
/// card1 若靠"从 card0 拷状态"或"每层跨卡搬输入"会把 PCIe 带宽吃光。
pub(crate) struct SpecSide {
    pub(crate) gdn_snap: Vec<Option<(CudaSlice<f32>, CudaSlice<bf16>, CudaSlice<bf16>)>>,
    pub(crate) gdn_in: Vec<Option<CudaSlice<bf16>>>,
    pub(crate) gdn_qkv: Vec<Option<CudaSlice<bf16>>>,
    pub(crate) gdn_z: Vec<Option<CudaSlice<bf16>>>,
    pub(crate) gdn_a: Vec<Option<CudaSlice<bf16>>>,
    pub(crate) gdn_b: Vec<Option<CudaSlice<bf16>>>,
    pub(crate) ple_snap: Vec<Option<CudaSlice<bf16>>>,
}

/// 跨请求状态检查点（见 [`Sequence::save_checkpoint`]）。
///
/// 只存“喂进 KV 的那段”的跨 token 状态：QSA 是 append-only（恢复时回卷计数即可），
/// GDN/PLE 的循环状态必须真拷贝。一份 ≈115 MB（36 个 GDN 层 × 3.1 MB + 小历史）。
pub struct StateCheckpoint {
    /// 已经喂进 KV 的 token（`ids.len` = KV 长度）
    pub ids: Vec<i64>,
    pub(crate) gdn: Vec<Option<(CudaSlice<f32>, CudaSlice<bf16>, CudaSlice<bf16>)>>,
    pub(crate) ple: Vec<Option<(CudaSlice<bf16>, CudaSlice<bf16>)>>,
}

impl Drop for StateCheckpoint {
    fn drop(&mut self) {
        // 归还 scratch 池（同 stream 顺序执行 ⇒ 先前从这些缓冲读的拷贝一定已完成）
        let gdn = std::mem::take(&mut self.gdn);
        for (sv, sh, shn) in gdn.into_iter().flatten() {
            lingxi_model::scratch::put_f32(sv);
            lingxi_model::scratch::put_bf16(sh);
            lingxi_model::scratch::put_bf16(shn);
        }
        let ple = std::mem::take(&mut self.ple);
        for (ph, phn) in ple.into_iter().flatten() {
            lingxi_model::scratch::put_bf16(ph);
            lingxi_model::scratch::put_bf16(phn);
        }
    }
}

/// **主机内存里的完整前缀检查点**（不活跃会话从显存换出用）。
///
/// 与 [`StateCheckpoint`] 的区别：除了 GDN/PLE 循环状态，还把 QSA 的 KV/indexer
/// 数据一起搬到 host —— 否则别的请求一 prefill 就会把设备 KV 覆盖，缓存就废了。
pub struct HostCheckpoint {
    /// 这份检查点覆盖的 token（= 最长的 KV 长度）
    pub ids: Vec<i64>,
    /// host 占用字节数（供 serve 做预算/LRU）
    pub bytes: usize,
    /// 每层 GDN：`(s_all f32, k_hist u16, k_hist_next u16)`；QSA 层为 None
    pub(crate) gdn: Vec<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>>,
    /// 每层 PLE：`(hist u16, hist_next u16)`
    pub(crate) ple: Vec<Option<(Vec<u16>, Vec<u16>)>>,
    /// 每个 QSA 层的 KV/indexer；GDN 层为 None
    pub(crate) qsa: Vec<Option<QsaHostKv>>,
    /// **同一段 KV 的较短前缀**（消息边界 / prompt 刚喂完）只额外存 GDN/PLE 状态，
    /// KV 直接复用最长那份 —— 256K 下省掉两三份 7GB 的重复拷贝。
    pub(crate) alts: Vec<HostAlt>,
}

/// 同一条会话在某个较短长度上的状态（KV 共享主检查点）。
pub struct HostAlt {
    pub len: usize,
    pub next: i64,
    pub bytes: usize,
    pub(crate) gdn: Vec<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>>,
    pub(crate) ple: Vec<Option<(Vec<u16>, Vec<u16>)>>,
}

impl HostCheckpoint {
    /// 追加一个“较短前缀”的状态档（KV 复用主检查点）。
    pub fn push_alt(&mut self, alt: HostAlt) {
        self.bytes += alt.bytes;
        self.alts.retain(|a| a.len != alt.len);
        self.alts.push(alt);
    }

    /// 可命中的长度档：主长度 + 各前缀长度。
    pub fn alt_lens(&self) -> Vec<usize> {
        self.alts.iter().map(|a| a.len).collect()
    }

    pub fn alt_next(&self, len: usize) -> Option<i64> {
        self.alts.iter().find(|a| a.len == len).map(|a| a.next)
    }

    /// **落盘**（应用退出后缓存还在；文件名由调用方定）。
    ///
    /// 格式：`LXCK` + 版本 + 层数 + next + ids + GDN/PLE/QSA + alts，小端、裸数组。
    /// 先写 `.tmp` 再 rename，进程被杀也不会留下半个可读文件。
    pub fn write_to(&self, path: &std::path::Path, next: i64) -> Result<u64> {
        let tmp = path.with_extension("lxc.tmp");
        if let Err(e) = self.write_to_tmp(&tmp, next) {
            // 写一半失败（磁盘满/IO 错）：清掉半成品，别留垃圾
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, path)?;
        Ok(std::fs::metadata(path)?.len())
    }

    fn write_to_tmp(&self, tmp: &std::path::Path, next: i64) -> Result<()> {
        use std::io::Write as _;
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(tmp)?);
            f.write_all(b"LXCK")?;
            f.write_all(&1u32.to_le_bytes())?;
            f.write_all(&(self.gdn.len() as u32).to_le_bytes())?;
            f.write_all(&next.to_le_bytes())?;
            put_ids(&mut f, &self.ids)?;
            // alts 的 (len, next) 放头部：启动扫描只读头就能恢复短前缀档
            f.write_all(&(self.alts.len() as u32).to_le_bytes())?;
            for a in &self.alts {
                f.write_all(&(a.len as u64).to_le_bytes())?;
                f.write_all(&a.next.to_le_bytes())?;
            }
            for g in &self.gdn {
                match g {
                    None => f.write_all(&[0])?,
                    Some((sv, sh, shn)) => {
                        f.write_all(&[1])?;
                        put_f32(&mut f, sv)?;
                        put_u16(&mut f, sh)?;
                        put_u16(&mut f, shn)?;
                    }
                }
            }
            for p in &self.ple {
                match p {
                    None => f.write_all(&[0])?,
                    Some((ph, phn)) => {
                        f.write_all(&[1])?;
                        put_u16(&mut f, ph)?;
                        put_u16(&mut f, phn)?;
                    }
                }
            }
            for q in &self.qsa {
                match q {
                    None => f.write_all(&[0])?,
                    Some(q) => {
                        f.write_all(&[1])?;
                        f.write_all(&(q.len as u64).to_le_bytes())?;
                        f.write_all(&(q.nblocks as u64).to_le_bytes())?;
                        f.write_all(&(q.kv_dim as u64).to_le_bytes())?;
                        f.write_all(&(q.idx_dim as u64).to_le_bytes())?;
                        put_u16(&mut f, &q.kn)?;
                        put_u16(&mut f, &q.v)?;
                        put_u16(&mut f, &q.rk)?;
                        put_u16(&mut f, &q.pooled)?;
                        put_f32(&mut f, &q.raw)?;
                    }
                }
            }
            for a in &self.alts {
                for g in &a.gdn {
                    match g {
                        None => f.write_all(&[0])?,
                        Some((sv, sh, shn)) => {
                            f.write_all(&[1])?;
                            put_f32(&mut f, sv)?;
                            put_u16(&mut f, sh)?;
                            put_u16(&mut f, shn)?;
                        }
                    }
                }
                for p in &a.ple {
                    match p {
                        None => f.write_all(&[0])?,
                        Some((ph, phn)) => {
                            f.write_all(&[1])?;
                            put_u16(&mut f, ph)?;
                            put_u16(&mut f, phn)?;
                        }
                    }
                }
            }
            f.flush()?;
        }
        Ok(())
    }

    /// 只读文件头（ids / next / 短前缀档 / 文件字节数）——启动扫描用，不碰几十 GB 的 payload。
    pub fn read_header(path: &std::path::Path) -> Result<(Vec<i64>, i64, Vec<(usize, i64)>, u64)> {
        use std::io::Read as _;
        let len = std::fs::metadata(path)?.len();
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        anyhow::ensure!(&magic == b"LXCK", "缓存文件 magic 不对");
        let ver = get_u32(&mut f)?;
        anyhow::ensure!(ver == 1, "缓存文件版本 {ver} 不支持");
        let layers = get_u32(&mut f)? as usize;
        anyhow::ensure!(layers <= 256, "缓存文件层数异常: {layers}");
        let next = get_i64(&mut f)?;
        let ids = get_ids(&mut f)?;
        let n_alts = get_u32(&mut f)? as usize;
        anyhow::ensure!(n_alts <= 16, "缓存文件 alts 数量异常: {n_alts}");
        let mut alt_pairs = Vec::with_capacity(n_alts);
        for _ in 0..n_alts {
            let alen = get_u64(&mut f)? as usize;
            let anext = get_i64(&mut f)?;
            alt_pairs.push((alen, anext));
        }
        Ok((ids, next, alt_pairs, len))
    }

    /// 从磁盘读回完整检查点（命中时调用）。
    pub fn read_from(path: &std::path::Path) -> Result<HostCheckpoint> {
        use std::io::Read as _;
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        anyhow::ensure!(&magic == b"LXCK", "缓存文件 magic 不对");
        let ver = get_u32(&mut f)?;
        anyhow::ensure!(ver == 1, "缓存文件版本 {ver} 不支持");
        let layers = get_u32(&mut f)? as usize;
        anyhow::ensure!(layers <= 256, "缓存文件层数异常: {layers}");
        let _next = get_i64(&mut f)?;
        let ids = get_ids(&mut f)?;
        let n_alts = get_u32(&mut f)? as usize;
        anyhow::ensure!(n_alts <= 16, "缓存文件 alts 数量异常: {n_alts}");
        let mut alt_pairs = Vec::with_capacity(n_alts);
        for _ in 0..n_alts {
            let alen = get_u64(&mut f)? as usize;
            let anext = get_i64(&mut f)?;
            alt_pairs.push((alen, anext));
        }
        let mut bytes = 0usize;
        let mut gdn = Vec::with_capacity(layers);
        for _ in 0..layers {
            let tag = get_u8(&mut f)?;
            if tag == 0 {
                gdn.push(None);
            } else {
                let sv = get_f32(&mut f)?;
                let sh = get_u16(&mut f)?;
                let shn = get_u16(&mut f)?;
                bytes += sv.len() * 4 + (sh.len() + shn.len()) * 2;
                gdn.push(Some((sv, sh, shn)));
            }
        }
        let mut ple = Vec::with_capacity(layers);
        for _ in 0..layers {
            let tag = get_u8(&mut f)?;
            if tag == 0 {
                ple.push(None);
            } else {
                let ph = get_u16(&mut f)?;
                let phn = get_u16(&mut f)?;
                bytes += (ph.len() + phn.len()) * 2;
                ple.push(Some((ph, phn)));
            }
        }
        let mut qsa = Vec::with_capacity(layers);
        for _ in 0..layers {
            let tag = get_u8(&mut f)?;
            if tag == 0 {
                qsa.push(None);
            } else {
                let len = get_u64(&mut f)? as usize;
                let nblocks = get_u64(&mut f)? as usize;
                let kv_dim = get_u64(&mut f)? as usize;
                let idx_dim = get_u64(&mut f)? as usize;
                let kn = get_u16(&mut f)?;
                let v = get_u16(&mut f)?;
                let rk = get_u16(&mut f)?;
                let pooled = get_u16(&mut f)?;
                let raw = get_f32(&mut f)?;
                bytes += (kn.len() + v.len() + rk.len() + pooled.len()) * 2 + raw.len() * 4;
                qsa.push(Some(QsaHostKv { len, nblocks, kv_dim, idx_dim, kn, v, rk, pooled, raw }));
            }
        }
        let mut alts = Vec::with_capacity(n_alts);
        for (len, next) in alt_pairs {
            let mut ab = 0usize;
            let mut g = Vec::with_capacity(layers);
            for _ in 0..layers {
                let tag = get_u8(&mut f)?;
                if tag == 0 {
                    g.push(None);
                } else {
                    let sv = get_f32(&mut f)?;
                    let sh = get_u16(&mut f)?;
                    let shn = get_u16(&mut f)?;
                    ab += sv.len() * 4 + (sh.len() + shn.len()) * 2;
                    g.push(Some((sv, sh, shn)));
                }
            }
            let mut p = Vec::with_capacity(layers);
            for _ in 0..layers {
                let tag = get_u8(&mut f)?;
                if tag == 0 {
                    p.push(None);
                } else {
                    let ph = get_u16(&mut f)?;
                    let phn = get_u16(&mut f)?;
                    ab += (ph.len() + phn.len()) * 2;
                    p.push(Some((ph, phn)));
                }
            }
            bytes += ab;
            alts.push(HostAlt { len, next, bytes: ab, gdn: g, ple: p });
        }
        Ok(HostCheckpoint { ids, bytes, gdn, ple, qsa, alts })
    }
}

pub(crate) fn put_ids<W: std::io::Write>(w: &mut W, ids: &[i64]) -> Result<()> {
    w.write_all(&(ids.len() as u64).to_le_bytes())?;
    for id in ids {
        w.write_all(&id.to_le_bytes())?;
    }
    Ok(())
}
pub(crate) fn get_ids<R: std::io::Read>(r: &mut R) -> Result<Vec<i64>> {
    let n = get_u64(r)? as usize;
    // 损坏文件保护：ids 最多 4M（256K 上下文远远够）
    anyhow::ensure!(n <= 4_000_000, "缓存文件 ids 长度异常: {n}");
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(get_i64(r)?);
    }
    Ok(v)
}
pub(crate) fn put_f32<W: std::io::Write>(w: &mut W, v: &[f32]) -> Result<()> {
    w.write_all(&(v.len() as u64).to_le_bytes())?;
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    w.write_all(b)?;
    Ok(())
}
pub(crate) fn get_f32<R: std::io::Read>(r: &mut R) -> Result<Vec<f32>> {
    let n = get_u64(r)? as usize;
    anyhow::ensure!(n <= (1 << 27), "缓存文件 f32 数组长度异常: {n}");
    let mut v = vec![0f32; n];
    let b = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
    r.read_exact(b)?;
    Ok(v)
}
pub(crate) fn put_u16<W: std::io::Write>(w: &mut W, v: &[u16]) -> Result<()> {
    w.write_all(&(v.len() as u64).to_le_bytes())?;
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
    w.write_all(b)?;
    Ok(())
}
pub(crate) fn get_u16<R: std::io::Read>(r: &mut R) -> Result<Vec<u16>> {
    let n = get_u64(r)? as usize;
    anyhow::ensure!(n <= (1 << 27), "缓存文件 u16 数组长度异常: {n}");
    let mut v = vec![0u16; n];
    let b = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 2) };
    r.read_exact(b)?;
    Ok(v)
}
pub(crate) fn get_u8<R: std::io::Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
pub(crate) fn get_u32<R: std::io::Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
pub(crate) fn get_u64<R: std::io::Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
pub(crate) fn get_i64<R: std::io::Read>(r: &mut R) -> Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

impl StateCheckpoint {
    /// 把这份**设备**快照的 GDN/PLE 转成 host 状态（给 [`HostCheckpoint::push_alt`] 用）。
    pub fn to_host_alt(&self, eng: &Engine, next: i64) -> Result<HostAlt> {
        let n = eng.n_layers as usize;
        anyhow::ensure!(self.gdn.len() == n && self.ple.len() == n, "快照层数不匹配");
        let mut gdn = Vec::with_capacity(n);
        let mut ple = Vec::with_capacity(n);
        let mut bytes = 0usize;
        for i in 0..n {
            let k = eng.k_of(i as u32);
            match &self.gdn[i] {
                Some((sv, sh, shn)) => {
                    let sv = d2h_f32(k, sv, sv.len())?;
                    let sh = d2h_u16(k, sh, sh.len())?;
                    let shn = d2h_u16(k, shn, shn.len())?;
                    bytes += sv.len() * 4 + (sh.len() + shn.len()) * 2;
                    gdn.push(Some((sv, sh, shn)));
                }
                None => gdn.push(None),
            }
            match &self.ple[i] {
                Some((ph, phn)) => {
                    let ph = d2h_u16(k, ph, ph.len())?;
                    let phn = d2h_u16(k, phn, phn.len())?;
                    bytes += (ph.len() + phn.len()) * 2;
                    ple.push(Some((ph, phn)));
                }
                None => ple.push(None),
            }
        }
        Ok(HostAlt { len: self.ids.len(), next, bytes, gdn, ple })
    }
}

pub(crate) struct QsaHostKv {
    /// 保存时的 QSA 长度（恢复后按 `ids.len` 回卷）
    pub(crate) len: usize,
    pub(crate) nblocks: usize,
    pub(crate) kv_dim: usize,
    pub(crate) idx_dim: usize,
    pub(crate) kn: Vec<u16>,
    pub(crate) v: Vec<u16>,
    pub(crate) rk: Vec<u16>,
    pub(crate) pooled: Vec<u16>,
    /// 仅 `LINGXI_QSA_HOST=1` 的 host 回退路径需要；默认档存空
    pub(crate) raw: Vec<f32>,
}

pub(crate) fn d2h_f32(k: &Kernels, src: &CudaSlice<f32>, n: usize) -> Result<Vec<f32>> {
    use cudarc::driver::DevicePtr;
    k.ctx.bind_to_thread()?;
    let mut out = vec![0f32; n];
    let (p, _guard) = src.device_ptr(&k.stream);
    unsafe { cudarc::driver::result::memcpy_dtoh_sync::<f32>(&mut out, p)? };
    Ok(out)
}

pub(crate) fn d2h_u16(k: &Kernels, src: &CudaSlice<bf16>, n: usize) -> Result<Vec<u16>> {
    use cudarc::driver::DevicePtr;
    k.ctx.bind_to_thread()?;
    let mut out = vec![0u16; n];
    let (p, _guard) = src.device_ptr(&k.stream);
    unsafe { cudarc::driver::result::memcpy_dtoh_sync::<u16>(&mut out, p)? };
    Ok(out)
}

pub(crate) fn h2d_f32(k: &Kernels, dst: &CudaSlice<f32>, data: &[f32]) -> Result<()> {
    use cudarc::driver::DevicePtr;
    k.ctx.bind_to_thread()?;
    let (p, _guard) = dst.device_ptr(&k.stream);
    unsafe { cudarc::driver::result::memcpy_htod_sync::<f32>(p, data)? };
    Ok(())
}

pub(crate) fn h2d_u16(k: &Kernels, dst: &CudaSlice<bf16>, data: &[u16]) -> Result<()> {
    use cudarc::driver::DevicePtr;
    k.ctx.bind_to_thread()?;
    let (p, _guard) = dst.device_ptr(&k.stream);
    unsafe { cudarc::driver::result::memcpy_htod_sync::<u16>(p, data)? };
    Ok(())
}
