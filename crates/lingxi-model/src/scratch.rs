//! 设备 scratch 池：**按 (元素类型, 元素数) 复用**，专治 decode 热路径上的
//! `cuMemAlloc/cuFree` 风暴。
//!
//! 动机（实测）：cudarc 的 `CudaStream::alloc` 每次都会 `bind_to_thread()` + 驱动分配，
//! `CudaSlice` 析构再释放一次。一层里 GR/MoE 的中间张量加起来二十来个，
//! 一个 token 48 层 → 上千次分配/释放，CPU 侧开销远大于这些小张量的实际算力。
//! `LINGXI_SKIP=attn,moe` 实测纯 GR 地板就有 **0.25 ms/层**（≈12 ms/token）。
//!
//! 用法是显式 take / put：
//! ```ignore
//! let mut xn = scratch::bf16(k, r * dim)?;   // 池里有就复用，没有才真分配
//! k.grouped_rmsnorm_bf16(x, gamma, &mut xn, ...)?;
//! scratch::put_bf16(xn);                      // 放回池里等下次
//! ```
//! **正确性前提**：同一个 CUDA stream 上的操作是顺序执行的，所以"放回池里的缓冲被
//! 后续 kernel 覆写"一定发生在"先前读它的 kernel"之后。反过来，**忘记 put 只是不复用，
//! 不会出错**（池空时自动退回真分配），所以逐步改造是安全的。
//!
//! 与 `LINGXI_ARENA_PIN_GIB` 那类显存/主存策略无关：这里只池化**小中间张量**。

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use lingxi_kernels::Kernels;

/// ★ P15-㉞（2026-10-17）：池键升级为 **(设备号, 元素数)**——layer-split 下两张卡
/// 各用各的桶，跨设备串味的原罪消掉，`disable_pool_forever` 不再是 split 的必需品
/// （禁池才是 split+分块确定性发散的根因：直连 alloc/free 与在飞 kernel 竞争 +
/// 丢失"同 stream 顺序"复用前提）。设备号在 take 时从 `Kernels` 取；
/// `put_*` 拿不到设备号 ⇒ take 登记 ptr→dev 注册表、put 按表归桶。
#[derive(Default)]
struct Pool {
    bf16: HashMap<(usize, usize), Vec<CudaSlice<bf16>>>,
    f32: HashMap<(usize, usize), Vec<CudaSlice<f32>>>,
    i32: HashMap<(usize, usize), Vec<CudaSlice<i32>>>,
    u8: HashMap<(usize, usize), Vec<CudaSlice<u8>>>,
}

thread_local! {
    static POOL: RefCell<Pool> = RefCell::new(Pool::default());
}

/// 设备号：cudarc 0.19 的 `CudaSlice::context()` 携带其分配上下文 ⇒ take/put 两边都能拿到，
/// 不需要任何注册表。
fn dev_of(k: &Kernels) -> usize {
    k.ctx.ordinal()
}
fn dev_of_slice<T>(s: &CudaSlice<T>) -> usize {
    s.context().ordinal()
}

thread_local! {
    /// **禁用池化**（跨设备安全网）。为什么必须有它（2026-10-16，实测崩过）：
    /// 这个池的键**只有 `(类型, 长度)`、没有设备** —— 于是用了第二张卡（`LINGXI_MOE_SPLIT`）时，
    /// GPU1 的核会拿到一块**在 GPU0 上分配**的 scratch，直接 `CUDA_ERROR_ILLEGAL_ADDRESS`。
    /// 正确修法是给池键加设备，但那要动所有 `put_*`（它们拿不到设备号）；
    /// 这里用**保证正确**的最小修法：置上本开关期间 `bf16/f32` 一律直连分配、
    /// `put_*` 直接释放（不进池）。代价只是这段路径少了一点复用。
    static NO_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 在闭包执行期间禁用 scratch 池化（见 `NO_POOL` 的说明）。返回闭包的结果。
pub fn without_pool<R>(f: impl FnOnce() -> R) -> R {
    let prev = NO_POOL.with(|c| c.replace(true));
    let r = f();
    NO_POOL.with(|c| c.set(prev));
    r
}

/// **永久禁用池化**（`LINGXI_LAYER_SPLIT` 时由驱动调用；一旦调用不可撤销）。
///
/// 为什么要有它（2026-10-16，P15 里程碑 A）：池的键**不带设备**，而全量 layer-split 会让
/// **同一段代码在两张卡上跑** —— 逐个去堵"哪个缓冲属于哪张卡"既费劲又容易漏（本会话已经在
/// `scratch` 池、`MappedHost`、`graph` 固定区上各踩过一次）。整轮禁池是**一处改动消掉一整类坑**：
/// 分配直连、归还即释放，代价只是少了小张量的复用。
pub fn disable_pool_forever() {
    POOL_OFF.with(|c| c.set(true));
}

/// ★ P15-㉞ 诊断：`LINGXI_NO_SCRATCH_POOL=1` 时不split也能复现"禁池"状态
/// （二分"split+分块发散"到底是 split 还是 禁池 引起的）。
fn pool_forced_off() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("LINGXI_NO_SCRATCH_POOL").map(|s| s == "1").unwrap_or(false))
}

thread_local! {
    /// 永久禁池标志（见 `disable_pool_forever`）。
    static POOL_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[inline]
fn pool_off() -> bool {
    POOL_OFF.with(|c| c.get())
}

/// 借一块 `bf16` 设备缓冲。
pub fn bf16(k: &Kernels, len: usize) -> Result<CudaSlice<bf16>> {
    if NO_POOL.with(|c| c.get()) || pool_off() || pool_forced_off() {
        return Ok(unsafe { k.stream.alloc::<bf16>(len) }?);
    }
    let dev = dev_of(k);
    let s = POOL.with(|p| {
        p.borrow_mut().bf16.get_mut(&(dev, len)).and_then(|v| v.pop())
    });
    let s = match s {
        Some(s) => s,
        None => unsafe { k.stream.alloc::<bf16>(len) }?,
    };
    Ok(s)
}

pub fn put_bf16(s: CudaSlice<bf16>) {
    if NO_POOL.with(|c| c.get()) || pool_off() || pool_forced_off() {
        return; // 直接释放（调用方已确认这段路径无跨流复用风险）
    }
    let key = (dev_of_slice(&s), s.len());
    POOL.with(|p| p.borrow_mut().bf16.entry(key).or_default().push(s));
}
/// 借一块 `f32` 设备缓冲。
pub fn f32(k: &Kernels, len: usize) -> Result<CudaSlice<f32>> {
    if NO_POOL.with(|c| c.get()) || pool_off() || pool_forced_off() {
        return Ok(unsafe { k.stream.alloc::<f32>(len) }?);
    }
    let dev = dev_of(k);
    let s = POOL.with(|p| {
        p.borrow_mut().f32.get_mut(&(dev, len)).and_then(|v| v.pop())
    });
    let s = match s {
        Some(s) => s,
        None => unsafe { k.stream.alloc::<f32>(len) }?,
    };
    Ok(s)
}

pub fn put_f32(s: CudaSlice<f32>) {
    if NO_POOL.with(|c| c.get()) || pool_off() || pool_forced_off() {
        return; // 直接释放（调用方已确认这段路径无跨流复用风险）
    }
    let key = (dev_of_slice(&s), s.len());
    POOL.with(|p| p.borrow_mut().f32.entry(key).or_default().push(s));
}
/// 借一块 `i32` 设备缓冲。
pub fn i32(k: &Kernels, len: usize) -> Result<CudaSlice<i32>> {
    if NO_POOL.with(|c| c.get()) || pool_off() || pool_forced_off() {
        return Ok(unsafe { k.stream.alloc::<i32>(len) }?);
    }
    let dev = dev_of(k);
    let s = POOL.with(|p| {
        p.borrow_mut().i32.get_mut(&(dev, len)).and_then(|v| v.pop())
    });
    let s = match s {
        Some(s) => s,
        None => unsafe { k.stream.alloc::<i32>(len) }?,
    };
    Ok(s)
}

pub fn put_i32(s: CudaSlice<i32>) {
    if NO_POOL.with(|c| c.get()) || pool_off() || pool_forced_off() {
        return; // 直接释放（调用方已确认这段路径无跨流复用风险）
    }
    let key = (dev_of_slice(&s), s.len());
    POOL.with(|p| p.borrow_mut().i32.entry(key).or_default().push(s));
}
/// 借一块 `u8` 设备缓冲。
pub fn u8(k: &Kernels, len: usize) -> Result<CudaSlice<u8>> {
    if NO_POOL.with(|c| c.get()) || pool_off() || pool_forced_off() {
        return Ok(unsafe { k.stream.alloc::<u8>(len) }?);
    }
    let dev = dev_of(k);
    let s = POOL.with(|p| {
        p.borrow_mut().u8.get_mut(&(dev, len)).and_then(|v| v.pop())
    });
    let s = match s {
        Some(s) => s,
        None => unsafe { k.stream.alloc::<u8>(len) }?,
    };
    Ok(s)
}

pub fn put_u8(s: CudaSlice<u8>) {
    if NO_POOL.with(|c| c.get()) || pool_off() || pool_forced_off() {
        return; // 直接释放（调用方已确认这段路径无跨流复用风险）
    }
    let key = (dev_of_slice(&s), s.len());
    POOL.with(|p| p.borrow_mut().u8.entry(key).or_default().push(s));
}

// ───────────── 锁页 host 暂存：小规模 H2D 必须走锁页，否则驱动会退化成同步拷贝 ─────────────
//
// `CudaStream::clone_htod(&Vec<T>)` 的源是**可分页**内存：`cuMemcpyHtoDAsync` 遇到可分页源
// 会做 driver staging（内部先拷进锁页再 DMA），代价是小拷贝也要几十 µs，而且会阻塞。
// decode 每层都要把 `top_k` 个 `(行, 权重)` 送上去 → 48 层/ token 累积可观。

thread_local! {
    static PIN_I32: RefCell<Option<cudarc::driver::PinnedHostSlice<i32>>> =
        const { RefCell::new(None) };
    static PIN_F32: RefCell<Option<cudarc::driver::PinnedHostSlice<f32>>> =
        const { RefCell::new(None) };
}

/// 线程级锁页缓存的**跨上下文**守卫。
///
/// `PinnedHostSlice` 里带一个**属于创建它的上下文**的 event，而 `memcpy_*` 会先
/// `cuStreamWaitEvent(本次的流, 那个 event)` —— 一旦缓冲建在 **GPU0**、这次却要用在
/// **GPU1 的流**上，就是"跨上下文等事件" ⇒ `CUDA_ERROR_INVALID_VALUE`
/// （2026-10-16 实测：`LINGXI_MOE_SPLIT` 跑到 prefill layer 24 崩在 `scratch::upload_i32`）。
/// ⇒ 线程级缓存必须**按上下文隔离**：上下文一变就重建。
fn same_ctx<T>(p: &cudarc::driver::PinnedHostSlice<T>, k: &Kernels) -> bool {
    std::sync::Arc::ptr_eq(p.context(), k.stream.context())
}

// ⚠️⚠️ **不要跨调用复用单个锁页中转缓冲**（2026-10-16 定位到的真 bug，见 PLAN §P15-C）：
//   `memcpy_htod` 是**异步**的，而这个缓冲是**主机侧**写的 ⇒ 主机完全可能在下一次调用里
//   覆写它（长度相同就复用同一个缓冲），或者因为长度变了**把它 free 掉**——两者都可能
//   发生在"上一次的拷贝还在飞"的时候 ⇒ 设备拿到的不是这一次的字节 ⇒ **静默取错专家/取错行号**。
//   实测症状：同权重下投机路径（t>1 的组列表 MoE）从第 103 个 token 起与纯贪心分叉，
//   而同权重的 t=1 路径完全正常；**加 host 同步能治好、换成设备侧 event 等待治不好**
//   ⇒ 确认是主机侧覆写时序。
//
//   修法：**按长度分桶 + 每桶 N 环轮转**，且**永不 free**。可证正确性：
//   同一个桶里的缓冲要轮满 `PIN_RING` 次才会被再写，而调用方在每一层之间都有同步点
//   （MoE router 的 `stream.synchronize`）⇒ 轮回到同一块时，上一次的拷贝早已完成。
//   换环时再补一次同步（兜底，代价 ~每 N 次上传一次）。
/// 环大小。**不要按长度分桶**（那样长度一变就新建一批锁页缓冲，实测每次上传都在分配，
/// trunk 多花 6~8 ms/轮）；这里每类型只保留 `PIN_RING` 块**按需长大**的缓冲。
const PIN_RING: usize = 4;

macro_rules! pinned_upload_impl {
    ($fname:ident, $pool:ident, $ty:ty) => {
        fn $fname(k: &Kernels, vals: &[$ty], dst: &mut CudaSlice<$ty>) -> Result<()> {
            let n = vals.len();
            $pool.with(|pool| -> Result<()> {
                let mut st = pool.borrow_mut();
                let idx = st.1 % PIN_RING;
                st.1 = st.1.wrapping_add(1);
                // 只涨不缩：容量不够时**重建全部**（很少发生：长度上界是 rows*top_k）
                if st.0.first().map_or(true, |b| b.len() < n) {
                    st.0.clear();
                    for _ in 0..PIN_RING {
                        st.0.push(unsafe {
                            k.stream.context().alloc_pinned_with_flags::<$ty>(n, 0)
                        }?);
                    }
                }
                // ⚠️ 这里**故意不加同步**：环里有 4 块，而单层的上传次数 ≤ 3，且调用方
                //    **每层**都有同步点（MoE router 的 `stream.synchronize`）⇒ 轮回到同一块
                //    时，上一次的拷贝必然早已完成。加同步会把这条本来免费的路径变成
                //    每 4 次上传排空一次流水（实测 trunk +2.9 ms/轮），不划算。
                let buf = &mut st.0[idx];
                buf.as_mut_slice()?[..n].copy_from_slice(vals);
                k.stream.memcpy_htod(&buf.as_slice()?[..n], dst)?;
                Ok(())
            })
        }
    };
}
thread_local! {
    /// `(环形缓冲, 下一个待用下标)`；缓冲**只涨不缩**
    static PIN_I32_POOL: RefCell<(Vec<cudarc::driver::PinnedHostSlice<i32>>, usize)> =
        const { RefCell::new((Vec::new(), 0)) };
    static PIN_F32_POOL: RefCell<(Vec<cudarc::driver::PinnedHostSlice<f32>>, usize)> =
        const { RefCell::new((Vec::new(), 0)) };
}
pinned_upload_impl!(pinned_upload_i32, PIN_I32_POOL, i32);
pinned_upload_impl!(pinned_upload_f32, PIN_F32_POOL, f32);

/// 走锁页中转、从 scratch 池取目标缓冲的 `i32` 小张量上传。
pub fn upload_i32(k: &Kernels, vals: &[i32]) -> Result<CudaSlice<i32>> {
    let mut dst = i32(k, vals.len())?;
    pinned_upload_i32(k, vals, &mut dst)?;
    Ok(dst)
}

/// 同上，`f32`。
pub fn upload_f32(k: &Kernels, vals: &[f32]) -> Result<CudaSlice<f32>> {
    let mut dst = f32(k, vals.len())?;
    pinned_upload_f32(k, vals, &mut dst)?;
    Ok(dst)
}

thread_local! {
    /// 设备→主机下载的**锁页**中转（按需长大到历史最大）。
    static PIN_DL: RefCell<Option<cudarc::driver::PinnedHostSlice<bf16>>> =
        const { RefCell::new(None) };
}

/// 走**锁页**中转的设备→主机下载（bf16），返回拷出的 `Vec`。
///
/// ⚠️ 为什么必须有它（2026-10-16，本轮抓到的**最大单项**）：
/// `stream.clone_dtoh` 的目标是**可分页**内存，驱动只能用小块 bounce buffer 往返搬 ——
/// 实测吞吐只有 **~200~450 MB/s**（x16 链路有 12 GB/s）。而投机解码里有两处大 D2H：
/// - `Sequence::head_argmax_all`（`lingxi-driver`）：`t × vocab` bf16，t=9 时 **4.5 MB**，
///   紧跟着还要 `map(to_f32)` 出 9 MB 的 f32 再 argmax；
/// - `mtp-accept` 的 `mtp_step`：每枚草稿 `1 × vocab` bf16 = 496 KB。
///
/// 两者加起来在 kd=8 下要吃掉 **~20 ms/轮**，而且**完全落在逐层 prof 的括号之外** ——
/// 这正是"主干验证 74 ms vs 48 层 [step] 合计 45 ms"那 ~29 ms 的去处。
/// 换锁页后 4.5 MB 只要 ~0.4 ms。
///
/// 与 `pinned_upload_*` 对称；`memcpy_dtoh` 是异步的，所以必须紧跟一次 `synchronize` 才能读。
pub fn download_bf16(k: &Kernels, src: &CudaSlice<bf16>) -> Result<Vec<bf16>> {
    let n = src.len();
    PIN_DL.with(|c| -> Result<Vec<bf16>> {
        let mut g = c.borrow_mut();
        // ⚠️ 条件写成 `map_or(true, …)`：空池必须**分配**。
        //（我第一版写成 `map_or(usize::MAX, |p| p.len()) < n` —— `usize::MAX < n` 恒假，
        // 于是永远不分配 ⇒ 立刻在原地上 panic。与 `pinned_upload_*` 的 `!= n` 语义对齐。）
        // 另外同 `pinned_upload_*`：**上下文不同必须重建**（见 `same_ctx`）。
        if g.as_ref()
            .map_or(true, |p| !same_ctx(p, k) || p.len() < n)
        {
            *g = Some(unsafe { k.stream.context().alloc_pinned_with_flags::<bf16>(n, 0) }?);
        }
        let buf = g.as_mut().unwrap();
        k.stream.memcpy_dtoh(src, &mut buf.as_mut_slice()?[..n])?;
        k.stream.synchronize()?;
        Ok(buf.as_slice()?[..n].to_vec())
    })
}

// ───────────── 跨卡激活搬运（`LINGXI_MOE_SPLIT`）─────────────
//
// 为什么不能照抄 `clone_dtoh`/`clone_htod`（**实测过的坑**）：那两个的源/目标是**可分页**
// `Vec`，驱动只能走小块 bounce 往返（~200~450 MB/s）⇒ 每层 4 趟 ×36 KB 要 ~0.5 ms，
// 24 层 = 12 ms/轮，正好把跨卡省下的稠密带宽吃回去（实测 39.8 → **32.7 tok/s**）。
//
// 为什么不能两块卡共用一块锁页缓冲：`PinnedHostSlice` 的事件绑在**创建它的上下文**上，
// `memcpy_*` 会 `cuStreamWaitEvent(本流, 那个 event)` —— 跨上下文就是
// `CUDA_ERROR_INVALID_VALUE`（同 `same_ctx` 的说明）。⇒ **每张卡各持一块**，
// 中间隔一次 host `copy_from_slice`（36 KB ≈ 2 µs，可忽略）。
#[derive(Default)]
struct XferSide {
    /// `CUcontext` 句柄（0 = 这个位置还没被占）
    ctx: usize,
    pin: Option<cudarc::driver::PinnedHostSlice<bf16>>,
}

#[derive(Default)]
struct Xfer {
    sides: [XferSide; 2],
}

thread_local! {
    static XFER: RefCell<Xfer> = RefCell::new(Xfer::default());
}

impl Xfer {
    /// 给 `k` 找（或占）一个专属的 staging 位置，并保证容量 ≥ `n`。
    fn side_for(&mut self, k: &Kernels, n: usize) -> Result<usize> {
        let key = k.ctx.cu_ctx() as usize;
        let i = match self.sides.iter().position(|s| s.ctx == key) {
            Some(i) => i,
            None => {
                let i = self
                    .sides
                    .iter()
                    .position(|s| s.ctx == 0)
                    .ok_or_else(|| anyhow::anyhow!("跨卡 staging 只支持两张卡"))?;
                self.sides[i] = XferSide { ctx: key, pin: None };
                i
            }
        };
        let s = &mut self.sides[i];
        if s.pin.as_ref().map_or(true, |p| p.len() < n) {
            s.pin = Some(unsafe { k.stream.context().alloc_pinned_with_flags::<bf16>(n, 0) }?);
        }
        Ok(i)
    }
}

/// **跨卡激活搬运** `src`（在 `k_from` 上）→ `dst`（在 `k_to` 上），走各自卡上的锁页 staging。
/// 同步口径：块内做 3 步（D2H + sync、host memcpy、H2D），`k_to` 的 H2D 是异步的、按流序排队。
pub fn cross_copy(
    k_from: &Kernels,
    src: &CudaSlice<bf16>,
    k_to: &Kernels,
    dst: &mut CudaSlice<bf16>,
) -> Result<()> {
    let n = src.len();
    anyhow::ensure!(dst.len() >= n, "跨卡搬运：目标缓冲太小（{} < {n}）", dst.len());
    XFER.with(|c| -> Result<()> {
        let mut x = c.borrow_mut();
        let a = x.side_for(k_from, n)?;
        let b = x.side_for(k_to, n)?;
        anyhow::ensure!(a != b, "跨卡搬运的源/目标上下文相同：不是跨卡调用");
        // 取两块**不同**元素的 &mut：`a`/`b` 是运行期下标，用 split_at_mut 切成两半再各取一个。
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let (left, right) = x.sides.split_at_mut(hi);
        let (sa, sb) = if a < b {
            (&mut left[lo], &mut right[0])
        } else {
            (&mut right[0], &mut left[lo])
        };
        // 1) 源卡 → 它自己的 staging（`memcpy_dtoh` 异步，必须同步后才能读）
        {
            let buf = sa.pin.as_mut().unwrap().as_mut_slice()?;
            k_from.stream.memcpy_dtoh(src, &mut buf[..n])?;
            k_from.stream.synchronize()?;
        }
        // 2) host 侧两块 staging 之间搬一次（36 KB；`Vec::copy_from_slice` 不走驱动）
        {
            let sbuf = sb.pin.as_mut().unwrap().as_mut_slice()?;
            sbuf[..n].copy_from_slice(&sa.pin.as_ref().unwrap().as_slice()?[..n]);
        }
        // 3) 目标卡的 staging → 目标卡设备（走目标卡自己的流；事件同上下文，合法）
        let sbuf = sb.pin.as_ref().unwrap();
        k_to.stream.memcpy_htod(&sbuf.as_slice()?[..n], dst)?;
        Ok(())
    })
}

/// [探针] 打印一块设备缓冲的地址（`LINGXI_PTRPROBE=1` 时生效）。
/// CUDA Graph 会把设备指针**烘死**，所以必须先证明同一逻辑缓冲在 token 之间地址不变。
pub fn probe_ptr(k: &Kernels, tag: &str, buf: &CudaSlice<bf16>) {
    if !crate::flag("LINGXI_PTRPROBE") {
        return;
    }
    use cudarc::driver::DevicePtr;
    let (p, _g) = buf.device_ptr(&k.stream);
    let len = buf.len();
    let (t, u) = STATS.with(|s| {
        let m = s.borrow();
        m.get(&len).copied().unwrap_or((0, 0))
    });
    eprintln!("    [ptrprobe] {tag} = 0x{p:x}  (len={len} take={t} put={u} live={})", t - u);
}

// 每个尺寸的 bf16 take/put 计数（只在探针打开时维护）。
thread_local! {
    static STATS: RefCell<HashMap<usize, (u64, u64)>> = RefCell::new(HashMap::new());
}

// ───────────── 设备可写的锁页 host 缓冲（零拷贝 D2H 替身）─────────────
//
// 动机（实测，见 docs/decode-speed-profile.md）：decode 每层要把 router logits
// （512×bf16 = 1 KB）取回 host 做 top-k。走 `cuMemcpyDtoHAsync` + `cuStreamSynchronize`
// 时，**这 1 KB 的拷贝本身就让同步多等 ~0.8 ms**（同一份代码里 sync 中位数：
// 有拷贝 1.2 ms / 去掉拷贝 0.4 ms），完全不是带宽能解释的量级。
// 改法：让 kernel **直接写映射到设备的 host 内存**（UVA 零拷贝），host 侧只剩一次
// `cuStreamSynchronize`，拷贝这一步彻底消失。
pub struct MappedHost<T> {
    pin: cudarc::driver::PinnedHostSlice<T>,
    /// `cuMemHostGetDevicePointer` 得到的设备端别名
    pub dev: u64,
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits> MappedHost<T> {
    /// # Safety
    /// 与 `alloc_pinned_with_flags` 相同：内容是未初始化的。
    pub unsafe fn new(k: &Kernels, len: usize) -> Result<Self> {
        use cudarc::driver::sys;
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
        let pin = k.stream.context().alloc_pinned_with_flags::<T>(len, flags)?;
        let host = pin.as_ptr()? as *mut std::ffi::c_void;
        let mut dev: sys::CUdeviceptr = 0;
        // ⚠️ **必须先绑上下文**（2026-10-16，双卡实测）：`cuMemHostGetDevicePointer` 返回的是
        // **当前上下文**里的设备别名 —— 裸驱动调用不会自己 `bind_to_thread`，而开第二张卡
        // （`LINGXI_MOE_SPLIT`）后线程的当前上下文可能还是 GPU0，于是 GPU1 的核会拿到
        // 一个 GPU0 的别名 ⇒ `CUDA_ERROR_ILLEGAL_ADDRESS`。
        k.ctx.bind_to_thread()?;
        let rc = sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0);
        if rc != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuMemHostGetDevicePointer 失败: {rc:?}");
        }
        Ok(Self { pin, dev })
    }

    pub fn len(&self) -> usize {
        self.pin.len()
    }

    /// 创建它的上下文（线程级缓存要按它隔离，见 `moe.rs` 的 `MAP`）。
    pub fn ctx(&self) -> &std::sync::Arc<cudarc::driver::CudaContext> {
        self.pin.context()
    }

    /// 读 host 侧内容（该缓冲从未交给 cudarc 的拷贝 API，事件不会被 record，
    /// 所以这里的 `event.synchronize()` 是空转，不构成额外等待）。
    pub fn slice(&self) -> Result<&[T]> {
        Ok(self.pin.as_slice()?)
    }

    /// ㊺ A2：host 侧可写访问（设备路由的 miss 标志：host 清零、kernel 置位）。
    pub fn slice_mut(&mut self) -> Result<&mut [T]> {
        Ok(self.pin.as_mut_slice()?)
    }
}
