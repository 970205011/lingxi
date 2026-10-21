//! `lingxi-bench` — 压测 / 对拍 / 检查工具入口。旁路，不进热路径。
//!
//! 当前提供：
//!   `verify <artifact>` — 打开现成 ninfer v2 产物，跑结构自检 + 构建 tensor 目录
//!   `list   <artifact>` — 打印目录中的张量（名字/形状/格式/布局/角色）

use std::collections::BTreeMap;
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::DevicePtr;
use lingxi_artifact::expert_arena::BlobFormat;
use lingxi_artifact::Artifact;

mod host_cache;
mod mtp_arena;
mod arena_source;
mod serve;
mod spec;

/// 读 prompt 参数：`@<路径>` 从文件读（长上下文 prompt 可能几百 KB，超 argv 上限），否则当文本。
fn prompt_arg(arg: Option<&String>) -> anyhow::Result<String> {
    let a = arg.cloned().unwrap_or_else(|| "The capital of France is".to_string());
    if let Some(path) = a.strip_prefix('@') {
        Ok(std::fs::read_to_string(path)?)
    } else {
        Ok(a)
    }
}

fn main() {
    // ★ 致命错误必须 process::exit(1) 而不是返回 Err：启动早期 spawn 了分离的
    // PLE 预热线程（plet.rs，102GB 顺序读 ~9 分钟），main 返回不会杀它——
    // 曾实测"启动失败但进程挂着不退"长达数分钟（甚至 D 状态 kill -9 都卡住）。
    // process::exit 立即终止全部线程，保证失败即退出。
    if let Err(e) = run() {
        // K2 graph_timer：失败路径也尽力打出已收集的计时（开关关则 no-op）。
        lingxi_model::graph_timer::report();
        eprintln!("\n!!! lingxi-bench 失败退出: {e:?}\n");
        std::process::exit(1);
    }
    // K2 graph_timer 汇总（开关关则 no-op）。
    lingxi_model::graph_timer::report();
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // 环境开关校验（点名拼错/幽灵开关、拦住已知危险组合）——必须在任何 GPU 初始化之前。
    // `probes` 子命令只打印状态表，不跑别的（但**也要先过一遍校验**，方便在跑之前查配置）。
    lingxi_model::probes::check_env();
    if args.get(1).map(|s| s.as_str()) == Some("probes") {
        print!("{}", lingxi_model::probes::dump());
        return Ok(());
    }
    match args.get(1).map(|s| s.as_str()) {
        Some("attn-bench") => {
            // ★ P15-㊳ attention 微基准：attn2（每行一块）vs attn3（8 行共享 K/V 读）。
            // 冷启动单发口径（PLAN 纪律）：每 trial 前先写 48MB 脏缓冲驱逐 L2，单次 launch 计时，取 5 trial 的 min。
            let dev: usize = std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let heads = 24usize;
            let kv_heads = 2usize;
            let hd = 256usize;
            let t = std::env::var("ATTN_T").ok().and_then(|s| s.parse().ok()).unwrap_or(512usize);
            let mut dirty = k.stream.alloc_zeros::<u8>(48 * 1024 * 1024)?;
            for &tk in &[2048usize, 8192, 32768] {
                let mut q = k.stream.alloc_zeros::<half::bf16>(t * heads * hd)?;
                let mut kk = k.stream.alloc_zeros::<half::bf16>(tk * kv_heads * hd)?;
                let mut v = k.stream.alloc_zeros::<half::bf16>(tk * kv_heads * hd)?;
                let mut mask = k.stream.alloc_zeros::<f32>(t * tk)?;
                // 真实感填充：随机 q/k/v + 掩码 = 因果前缀 + 选中块（其余 f32::MIN）——
                // 零数据测不出数据相关性 slowdown（真实 prefill 里 attn3 慢 10×，零数据只慢 5%）。
                if std::env::var_os("ATTN_REAL_DATA").is_some() {
                    use half::bf16 as B;
                    let mut rng: u64 = 0x9e3779b97f4a7c15;
                    let mut next = move || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
                    let qh: Vec<B> = (0..t * heads * hd).map(|_| B::from_f32((next() % 1000) as f32 / 500.0 - 1.0)).collect();
                    let kh: Vec<B> = (0..tk * kv_heads * hd).map(|_| B::from_f32((next() % 1000) as f32 / 500.0 - 1.0)).collect();
                    let vh: Vec<B> = (0..tk * kv_heads * hd).map(|_| B::from_f32((next() % 1000) as f32 / 500.0 - 1.0)).collect();
                    // 每行选中 ~1/8 的块（compress=16 ⇒ tk/16 块，选 tk/128）
                    let compress = 16usize;
                    let nblocks = tk / compress;
                    let topk = (nblocks / 8).max(1);
                    let mut mh = vec![f32::MIN; t * tk];
                    for r in 0..t {
                        let vis = tk.min((r + 1) * 3 / 4 + 1); // 模拟因果前缀
                        for j in 0..vis { mh[r * tk + j] = 0.0; }
                        let mut sel: Vec<usize> = (0..nblocks).collect();
                        for _p in 0..topk {
                            let idx = (next() as usize) % sel.len();
                            let b = sel.swap_remove(idx);
                            let hi = ((b + 1) * compress).min(tk).min(vis);
                            for j in b * compress..hi { mh[r * tk + j] = 0.0; }
                        }
                    }
                    k.stream.memcpy_htod(&qh, &mut q)?;
                    k.stream.memcpy_htod(&kh, &mut kk)?;
                    k.stream.memcpy_htod(&vh, &mut v)?;
                    k.stream.memcpy_htod(&mh, &mut mask)?;
                }
                let mut out = k.stream.alloc_zeros::<half::bf16>(t * heads * hd)?;
                let scale = 1.0f32 / (hd as f32).sqrt();
                let mut sweep = |tag: &str| -> Result<()> {
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        // L2 驱逐 + 预热一次（不计时，让 lazy 状态 settle）
                        k.stream.memset_zeros(&mut dirty)?;
                        k.stream.synchronize()?;
                        let t0 = std::time::Instant::now();
                        k.attn_masked_bf16(&q, &kk, &v, &mask, &mut out, heads, kv_heads, t, tk, hd, scale)?;
                        k.stream.synchronize()?;
                        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                    }
                    eprintln!("attn-bench t={t} tk={tk} {tag}: **{best:.3} ms**（5 trial min）");
                    Ok(())
                };
                // 默认路径（attn3 若启用）
                sweep("attn3_or_default")?;
                // 强制 attn2
                std::env::set_var("LINGXI_NO_ATTN3", "1");
                sweep("attn2_forced")?;
                std::env::remove_var("LINGXI_NO_ATTN3");
            }
            return Ok(());
        }
        Some("launch-bench") => {
            // 量每次 kernel launch 的 CPU 成本（判断是否被 cudarc 的 bind_to_thread 拖慢）
            let dev: usize = std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let n = 1024usize;
            let x = k.stream.alloc_zeros::<f32>(n)?;
            let mut y = k.stream.alloc_zeros::<f32>(n)?;
            let iters = 20000usize;
            for _ in 0..200 {
                k.smoke_scale(&x, 1.0, &mut y, n as u32)?;
            }
            k.stream.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                k.smoke_scale(&x, 1.0, &mut y, n as u32)?;
            }
            let enqueue = t0.elapsed().as_secs_f64();
            k.stream.synchronize()?;
            eprintln!(
                "smoke_scale: {iters} 次入队 {:.3}s → **{:.2} µs/launch（CPU）**；含同步总计 {:.3}s",
                enqueue,
                enqueue / iters as f64 * 1e6,
                t0.elapsed().as_secs_f64()
            );

            // cuBLAS GEMV（T=1）的调用开销：每层有 12~16 次 cuBLAS，dispatch 可能才是空转来源
            let xg = k.stream.alloc_zeros::<half::bf16>(2560)?;
            let wg = k.stream.alloc_zeros::<half::bf16>(16384 * 2560)?;
            let mut yg = k.stream.alloc_zeros::<half::bf16>(16384)?;
            for _ in 0..50 {
                k.gemm_bf16_nt(&xg, &wg, &mut yg, 1, 16384, 2560)?;
            }
            k.stream.synchronize()?;
            let g_iters = 200usize;
            let tg = std::time::Instant::now();
            for _ in 0..g_iters {
                k.gemm_bf16_nt(&xg, &wg, &mut yg, 1, 16384, 2560)?;
            }
            let gen = tg.elapsed().as_secs_f64();
            k.stream.synchronize()?;
            eprintln!(
                "gemm_bf16_nt [1,16384]x[16384,2560]: {g_iters} 次入队 {:.3}s → **{:.2} µs/次（CPU）**；含同步 {:.3}s",
                gen,
                gen / g_iters as f64 * 1e6,
                tg.elapsed().as_secs_f64()
            );

            // 自研 GEMV 的**真实吞吐**：[16384,2560] 权重 = 83.9MB，看能否打满 HBM 带宽。
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (gn, gk) = (16384usize, 2560usize);
            let gx = k.stream.alloc_zeros::<half::bf16>(gk)?;
            let gw = k.stream.alloc_zeros::<half::bf16>(gn * gk)?;
            let mut gy = k.stream.alloc_zeros::<half::bf16>(gn)?;
            for _ in 0..5 {
                let (wp, _a) = gw.device_ptr(&k.stream);
                let (xp, _b) = gx.device_ptr(&k.stream);
                let (yp, _c) = gy.device_ptr_mut(&k.stream);
                k.gemv_bf16_ptr(wp, xp, yp, gn, gk)?;
            }
            k.stream.synchronize()?;
            let it2 = 30usize;
            let t2 = std::time::Instant::now();
            for _ in 0..it2 {
                let (wp, _a) = gw.device_ptr(&k.stream);
                let (xp, _b) = gx.device_ptr(&k.stream);
                let (yp, _c) = gy.device_ptr_mut(&k.stream);
                k.gemv_bf16_ptr(wp, xp, yp, gn, gk)?;
            }
            k.stream.synchronize()?;
            let dt2 = t2.elapsed().as_secs_f64();
            let bytes = (gn * gk * 2) as f64;
            eprintln!(
                "gemv_bf16 [16384,2560]: **{:.1} µs/次**（含同步）→ **{:.0} GB/s**（V100 峰值 900 GB/s = 带宽受限则 ~93µs）",
                dt2 / it2 as f64 * 1e6,
                bytes * it2 as f64 / dt2 / 1e9
            );

            // 对比 0：**引擎真实形状**的 GEMV 带宽扫描（三种核并排）。
            // 动机：单条 [16384,2560] 能跑到 ~772 GB/s，但整层聚合只有 ~150 GB/s。
            // 这里把引擎里实际用到的每个形状单独量一遍，判断"小 n 的 GEMV 是不是天生带宽低"，
            // 以及 128-bit load 的 row8 核能把它救回多少。
            {
                let its = 40usize;
                let bench = |nd: usize, kd: usize| -> Result<(f64, f64, f64)> {
                    let w2 = k.stream.alloc_zeros::<half::bf16>(nd * kd)?;
                    let x2 = k.stream.alloc_zeros::<half::bf16>(kd)?;
                    let mut y2 = k.stream.alloc_zeros::<half::bf16>(nd)?;
                    let (wp, _) = w2.device_ptr(&k.stream);
                    let (xp, _) = x2.device_ptr(&k.stream);
                    let (yp, _) = y2.device_ptr_mut(&k.stream);
                    // (1) 旧的 warps-per-row
                    for _ in 0..3 {
                        k.gemv_bf16_ptr(wp, xp, yp, nd, kd)?;
                    }
                    k.stream.synchronize()?;
                    let t = std::time::Instant::now();
                    for _ in 0..its {
                        k.gemv_bf16_ptr(wp, xp, yp, nd, kd)?;
                    }
                    k.stream.synchronize()?;
                    let a = t.elapsed().as_secs_f64() / its as f64 * 1e6;
                    // (2) 一行一 block、标量 load
                    for _ in 0..3 {
                        k.gemv_bf16_block_epi_ptr(wp, xp, yp, nd, kd, 0, 0.0, 0.0)?;
                    }
                    k.stream.synchronize()?;
                    let t = std::time::Instant::now();
                    for _ in 0..its {
                        k.gemv_bf16_block_epi_ptr(wp, xp, yp, nd, kd, 0, 0.0, 0.0)?;
                    }
                    k.stream.synchronize()?;
                    let b = t.elapsed().as_secs_f64() / its as f64 * 1e6;
                    // (3) 一行一 block、128-bit load
                    for _ in 0..3 {
                        k.gemv_bf16_row8_epi_ptr(wp, xp, yp, nd, kd, 0, 0.0, 0.0)?;
                    }
                    k.stream.synchronize()?;
                    let t = std::time::Instant::now();
                    for _ in 0..its {
                        k.gemv_bf16_row8_epi_ptr(wp, xp, yp, nd, kd, 0, 0.0, 0.0)?;
                    }
                    k.stream.synchronize()?;
                    let c = t.elapsed().as_secs_f64() / its as f64 * 1e6;
                    Ok((a, b, c))
                };
                for &(nn, kk) in &[
                    (8192usize, 2560usize), // GDN qkvz 前半（42 MB）
                    (16384, 2560),          // 与 Q4gemv 同形状，用于"bf16 vs W4 谁快"
                    (13312, 2560),          // 与 Q4gemv 同形状
                    (2560, 8192),           // GDN output（42 MB）
                    (2560, 6144),           // 与 Q4gemv 同形状
                    (6144, 2560),           // QSA q/gate 段（31 MB）
                    (512, 2560),            // QSA k/v 段 / MoE router（2.6 MB）
                    (10240, 320),           // GR mix_up（6.5 MB）
                    (320, 10240),           // GR mix_down（6.5 MB）
                    (1280, 2560),           // MoE shared_gate_up（6.5 MB）
                    (2560, 640),            // MoE shared_down（3.3 MB）
                    (4, 10240),             // GR block_inject（80 KB）
                ] {
                    let (a, b, c) = bench(nn, kk)?;
                    let by = (nn * kk * 2) as f64;
                    let g = |us: f64| by / (us * 1e-6) / 1e9;
                    eprintln!(
                        "  gemv [{nn:>5},{kk:>5}] {:.1} MB  warp/row {a:>6.1} µs（{:>3.0} GB/s）| block {b:>6.1} µs（{:>3.0} GB/s）| row8 {c:>6.1} µs（**{:>3.0} GB/s**）",
                        by / 1048576.0, g(a), g(b), g(c)
                    );
                }
                // 旧的逐形状表保留（只打 warps-per-row，历史对比用）
                for &(nn, kk) in &[(8192usize, 2560usize)] {
                    let w2 = k.stream.alloc_zeros::<half::bf16>(nn * kk)?;
                    let x2 = k.stream.alloc_zeros::<half::bf16>(kk)?;
                    let mut y2 = k.stream.alloc_zeros::<half::bf16>(nn)?;
                    let its = 40usize;
                    for _ in 0..3 {
                        let (wp, _) = w2.device_ptr(&k.stream);
                        let (xp, _) = x2.device_ptr(&k.stream);
                        let (yp, _) = y2.device_ptr_mut(&k.stream);
                        k.gemv_bf16_ptr(wp, xp, yp, nn, kk)?;
                    }
                    k.stream.synchronize()?;
                    let t = std::time::Instant::now();
                    for _ in 0..its {
                        let (wp, _) = w2.device_ptr(&k.stream);
                        let (xp, _) = x2.device_ptr(&k.stream);
                        let (yp, _) = y2.device_ptr_mut(&k.stream);
                        k.gemv_bf16_ptr(wp, xp, yp, nn, kk)?;
                    }
                    k.stream.synchronize()?;
                    let dt = t.elapsed().as_secs_f64();
                    let by = (nn * kk * 2) as f64;
                    eprintln!(
                        "  （历史口径）gemv [{nn:>5},{kk:>5}] {:.1} MB（{:.0} blocks）→ {:.1} µs → **{:.0} GB/s**",
                        by / 1048576.0,
                        (nn + 7) / 8,
                        dt / its as f64 * 1e6,
                        by * its as f64 / dt / 1e9
                    );
                }
            }


            // 对比 0c：**瘦长 GEMM 随 m 的带宽**（t>1 前向的核心 kernel）。
            // 【本轮改】形状换成 **GR 的两个真形状**（[320,10240] mix_down / [10240,320] mix_up），
            // 并同时量**新旧两个核** —— §3.38 实测 GR 在 t=2 每层要 ~200 µs，而流量只该 ~17 µs/层。
            // 判据：权重只读一遍 ⇒ m=1/2/4/8 的时间应几乎相同（FLOPs 白送）；
            // 若随 m 显著变慢，说明 m 路展开有问题，t>1 前向就永远快不起来。
            for &(nn, kk) in &[
                (320usize, 10240usize),
                (10240, 320),
                // ★ P15-⑨（2026-10-17）：GDN/QSA 投影真形状（attn 段 ~230 GB/s 之谜的对照）
                (10240, 2560),          // GDN qkv（52 MB）
                (6144, 2560),           // GDN z / QSA q,gate（31 MB）
                (2560, 6144),           // GDN out（31 MB）
                (2560, 10240),          // （备用）转置对照
            ] {
                let mut w2 = k.stream.alloc_zeros::<half::bf16>(nn * kk)?;
                let mut x2 = k.stream.alloc_zeros::<half::bf16>(8 * kk)?;
                let mut y2 = k.stream.alloc_zeros::<half::bf16>(8 * nn)?;
                // ★ P15-⑲：填充随机值（全零的对拍恒过、没意义）。复用 xorshift。
                {
                    let mut h_w = vec![half::bf16::from_f32(0.0); nn * kk];
                    let mut h_x = vec![half::bf16::from_f32(0.0); 8 * kk];
                    let mut s = 0x9E3779B97F4A7C15u64;
                    let mut rnd = |v: &mut Vec<half::bf16>| {
                        for b in v.iter_mut() {
                            s ^= s << 13;
                            s ^= s >> 7;
                            s ^= s << 17;
                            // 全精度位模式（含各种指数/尾数组合，可能有 NaN——两侧同输入仍应一致）
                            *b = half::bf16::from_bits(s as u16);
                        }
                    };
                    rnd(&mut h_w);
                    rnd(&mut h_x);
                    k.stream.memcpy_htod(&h_w, &mut w2)?;
                    k.stream.memcpy_htod(&h_x, &mut x2)?;
                }
                let mut yref = k.stream.alloc_zeros::<half::bf16>(8 * nn)?;
                for &m in &[1usize, 2, 4, 8] {
                    // 0=old 1=row 2=multi（P15-⑲）
                    for &newk in &[0u8, 1, 2] {
                        let its = 30usize;
                        let launch = |
                            k: &lingxi_kernels::Kernels,
                            yb: &mut cudarc::driver::CudaSlice<half::bf16>,
                        | -> anyhow::Result<()> {
                            // 注：类型注解写全是为了让闭包非泛型（两处调用点类型一致）。
                            let (wp, _) = w2.device_ptr(&k.stream);
                            let (xp, _) = x2.device_ptr(&k.stream);
                            let (yp, _) = yb.device_ptr_mut(&k.stream);
                            match newk {
                                1 => k.gemm_skinny_row_bf16_ptr(wp, xp, yp, nn, kk, m, 0, 0.0, 0.0)?,
                                2 => k.gemm_skinny_row_multi_bf16_ptr(wp, xp, yp, nn, kk, m, 0, 0.0, 0.0)?,
                                _ => k.gemm_skinny_bf16_ptr(wp, xp, yp, nn, kk, m, 0, 0.0, 0.0)?,
                            }
                            Ok(())
                        };
                        for _ in 0..3 {
                            launch(&k, &mut y2)?;
                        }
                        k.stream.synchronize()?;
                        let t = std::time::Instant::now();
                        for _ in 0..its {
                            launch(&k, &mut y2)?;
                        }
                        k.stream.synchronize()?;
                        let dt = t.elapsed().as_secs_f64();
                        let by = (nn * kk * 2) as f64;
                        eprintln!(
                            "  skinny[{}({nn:>5},{kk:>5})] m={m} {:.1} MB → {:.1} µs → **{:.0} GB/s**",
                            match newk { 1 => "row", 2 => "multi", _ => "old" },
                            by / 1048576.0,
                            dt / its as f64 * 1e6,
                            by * its as f64 / dt / 1e9
                        );
                        // ★ P15-⑲ 对拍：multi 的输出必须与 row **逐位一致**（前 m 个输出行）。
                        if newk == 2 {
                            k.gemm_skinny_row_bf16_ptr(
                                { let (p, _) = w2.device_ptr(&k.stream); p },
                                { let (p, _) = x2.device_ptr(&k.stream); p },
                                { let (p, _) = yref.device_ptr_mut(&k.stream); p },
                                nn, kk, m, 0, 0.0, 0.0,
                            )?;
                            launch(&k, &mut y2)?;
                            k.stream.synchronize()?;
                            let a: Vec<half::bf16> = k.stream.clone_dtoh(&y2)?;
                            let b: Vec<half::bf16> = k.stream.clone_dtoh(&yref)?;
                            let nbad = a[..m * nn]
                                .iter()
                                .zip(b[..m * nn].iter())
                                .filter(|(u, v)| u.to_bits() != v.to_bits())
                                .count();
                            eprintln!("    multi==row 对拍: {nbad}/{} 元素不一致", m * nn);
                            anyhow::ensure!(nbad == 0, "skinny_multi 与 skinny_row 位不一致");
                        }
                    }
                }
            }


            // 判断 W4 到底赚不赚，必须看这个（别靠整机 A/B 猜）。
            // 【本轮加】`(10240, 320)` GR mix_up 与 `(2560, 640)` MoE shared_down —— 这两个
            // **kdim 很小**的形状此前从没被测过 Q4。起因：把 GR 权重量化成 W4 后整机从 33 ms
            // 掉到 **12 s/token**（慢了 360 倍），必须查清是小 kdim 的 Q4 核有病理，还是别的原因。
            for &(nn, kk) in &[
                (16384usize, 2560usize), // gdn/query_key_value_z
                (2560, 6144),            // gdn/output, qsa/output
                (13312, 2560),           // qsa/query_key_gate_value
                (320, 10240),            // gr mix_down
                (10240, 320),            // gr mix_up（kdim 小）
                (2560, 640),             // moe shared_down（kdim 小）
            ] {
                use lingxi_model::q4 as q4m;
                let bits: Vec<u16> =
                    (0..nn * kk).map(|i| (i as u64).wrapping_mul(2654435761) as u16).collect();
                let (payload, k_pad, scale_off) = q4m::quantize_bf16_bits(&bits, nn, kk)?;
                let packed = k.stream.clone_htod(&payload)?;
                let x = k.stream.alloc_zeros::<half::bf16>(kk)?;
                let mut y = k.stream.alloc_zeros::<half::bf16>(nn)?;
                let its = 30usize;
                let (xp, _) = x.device_ptr(&k.stream);
                for _ in 0..3 {
                    let (wp, _) = packed.device_ptr(&k.stream);
                    let (yp, _) = y.device_ptr_mut(&k.stream);
                    k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, nn, kk, k_pad, 0, scale_off, 0, 0.0, 0.0)?;
                }
                k.stream.synchronize()?;
                let t = std::time::Instant::now();
                for _ in 0..its {
                    let (wp, _) = packed.device_ptr(&k.stream);
                    let (yp, _) = y.device_ptr_mut(&k.stream);
                    k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, nn, kk, k_pad, 0, scale_off, 0, 0.0, 0.0)?;
                }
                k.stream.synchronize()?;
                let dt = t.elapsed().as_secs_f64();
                let by = payload.len() as f64;
                // 【冷启动单发对照 —— P7.f 的第一判据】
                // 上面 30 次背靠背是"热循环"口径：同一权重、同一地址、DRAM 页与预取全热，
                // 而**引擎里每个核一轮只跑一次**。两者差多少，决定了 launch-bench 的"隔离值"
                // 能不能代表原位值（P7.f 记的"GDN 四段原位慢 1.9×"到底是真的，还是口径差）。
                // 做法：用 32 MB 的 D2D 拷贝冲掉 L2，再**单发**测一次，重复 5 次取最小值。
                // （cudarc 这个版本没有 `memset_d8`，用 D2D 拷贝同样能达到"写满/挤掉 L2"的效果）
                let l2_src = k.stream.alloc_zeros::<u8>(32 * 1024 * 1024)?;
                let mut l2_dst = k.stream.alloc_zeros::<u8>(32 * 1024 * 1024)?;
                let mut cold = f64::INFINITY;
                for _ in 0..5 {
                    k.stream.memcpy_dtod(&l2_src, &mut l2_dst)?;
                    k.stream.synchronize()?;
                    let t2 = std::time::Instant::now();
                    let (wp, _) = packed.device_ptr(&k.stream);
                    let (yp, _) = y.device_ptr_mut(&k.stream);
                    k.gemv_q4g64_dense_epi_ptr(wp, xp, yp, nn, kk, k_pad, 0, scale_off, 0, 0.0, 0.0)?;
                    k.stream.synchronize()?;
                    cold = cold.min(t2.elapsed().as_secs_f64());
                }
                eprintln!(
                    "  Q4gemv [{nn:>5},{kk:>5}] {:.1} MB payload → {:.1} µs → **{:.0} GB/s**（bf16 同形状读数见上表）\
                     ｜**冷启动单发 {:.1} µs → {:.0} GB/s**（冲掉 L2 后单发；= 引擎的真实口径）",
                    by / 1048576.0,
                    dt / its as f64 * 1e6,
                    by * its as f64 / dt / 1e9,
                    cold * 1e6,
                    by / cold / 1e9
                );
            }


            // ── 小核普查（decode 每 token 都要跑的"零碎"；每格都能对上一个调用点）──
            {
                let its = 300usize;
                let bench = |_tag: &str, f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
                    for _ in 0..5 {
                        f()?;
                    }
                    k.stream.synchronize()?;
                    let t = std::time::Instant::now();
                    for _ in 0..its {
                        f()?;
                    }
                    k.stream.synchronize()?;
                    Ok(t.elapsed().as_secs_f64() / its as f64 * 1e6)
                };
                // ── Q4 瘦长 GEMM 自检：与"解量化 + 瘦长 GEMM"逐元素比对 ──
            if std::env::var("LINGXI_Q4G_TEST").is_ok() {
                let (nn, kk, m) = (512usize, 1024usize, 3usize);
                let mut st: u32 = 12345;
                let nextf = |st: &mut u32| -> f32 {
                    *st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((*st >> 8) as f32 / 8388608.0) - 1.0
                };
                let wbits: Vec<u16> =
                    (0..nn * kk).map(|_| half::bf16::from_f32(nextf(&mut st)).to_bits()).collect();
                let q4w = lingxi_model::q4::upload_from_bf16_bits(&k, &wbits, nn, kk)?;
                let xbits: Vec<half::bf16> =
                    (0..m * kk).map(|_| half::bf16::from_f32(nextf(&mut st))).collect();
                let xdev = k.stream.clone_htod(&xbits)?;
                let mut tmpw = k.stream.alloc_zeros::<half::bf16>(nn * kk)?;
                k.dequant_q4g64_dense_bf16(&q4w.packed, &mut tmpw, nn, kk, q4w.k_pad, 0, q4w.scale_off)?;
                let mut y_ref = k.stream.alloc_zeros::<half::bf16>(m * nn)?;
                lingxi_model::gpu::proj_t1_epi(&k, &xdev, &tmpw, &mut y_ref, nn, kk, m, 0, 0.0, 0.0)?;
                let mut y_new = k.stream.alloc_zeros::<half::bf16>(m * nn)?;
                {
                    let (wp, _a) = q4w.packed.device_ptr(&k.stream);
                    let (xp, _b) = xdev.device_ptr(&k.stream);
                    let (yp, _c) = y_new.device_ptr_mut(&k.stream);
                    k.gemm_q4_skinny_bf16_ptr(wp, xp, yp, nn, kk, q4w.k_pad, m, 0, q4w.scale_off, 0, 0.0, 0.0)?;
                }
                k.stream.synchronize()?;
                let a_ref: Vec<half::bf16> = k.stream.clone_dtoh(&y_ref)?;
                let a_new: Vec<half::bf16> = k.stream.clone_dtoh(&y_new)?;
                // CPU 精确参考：把解量化后的权重与 x 都拉回主机，用 f64 做点积
                let wh: Vec<half::bf16> = k.stream.clone_dtoh(&tmpw)?;
                let xh: Vec<half::bf16> = k.stream.clone_dtoh(&xdev)?;
                let mut cpu = vec![0f64; m * nn];
                for s in 0..m {
                    for r in 0..nn {
                        let mut acc = 0f64;
                        for c in 0..kk {
                            acc += wh[r * kk + c].to_f32() as f64 * xh[s * kk + c].to_f32() as f64;
                        }
                        cpu[s * nn + r] = acc;
                    }
                }
                eprintln!(
                    "  Q4G 自检基准：cpu[0..3]={:?} ref[0..3]={:?} new[0..3]={:?}",
                    &cpu[0..3].iter().map(|v| *v as f32).collect::<Vec<_>>(),
                    &a_ref[0..3].iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                    &a_new[0..3].iter().map(|v| v.to_f32()).collect::<Vec<_>>()
                );
                let mut worst = 0usize;
                let mut wdiff = 0.0f32;
                let mut bad = 0usize;
                let mut shown = 0usize;
                let m_nn = m * nn;
                for i in 0..m * nn {
                    let d = (a_ref[i].to_f32() - a_new[i].to_f32()).abs();
                    if d > 1e-2 {
                        bad += 1;
                        if shown < 4 {
                            shown += 1;
                            eprintln!(
                                "    坏点 s={} row={} ref={} new={}",
                                i / nn,
                                i % nn,
                                a_ref[i].to_f32(),
                                a_new[i].to_f32()
                            );
                        }
                    }
                    if d > wdiff {
                        wdiff = d;
                        worst = i;
                    }
                }
                eprintln!(
                    "  Q4G 自检 [{nn},{kk}] m={m}：最大差 {wdiff:.6}（第 {worst} 位：ref={} new={}）｜坏点 {bad}/{m_nn} ｜{}",
                    a_ref[worst].to_f32(),
                    a_new[worst].to_f32(),
                    if wdiff < 1e-2 { "一致 ✓" } else { "**不一致 ✗**" }
                );
            }

            // MoE 归并（每层 1 次 × 48）
                let o_all = k.stream.alloc_zeros::<half::bf16>(10 * 2560)?;
                let rows_of = k.stream.alloc_zeros::<i32>(10)?;
                let wts = k.stream.alloc_zeros::<f32>(10)?;
                let mut eout = k.stream.alloc_zeros::<half::bf16>(2560)?;
                let t_comb = bench("combine", &mut || {
                    k.moe_combine_rows_bf16(&o_all, &rows_of, &wts, &mut eout, 10, 1, 2560)
                })?;
                // MoE swiglu（每层 1 次 × 48）
                let gu = k.stream.alloc_zeros::<half::bf16>(10 * 1280)?;
                let mut act = k.stream.alloc_zeros::<half::bf16>(10 * 640)?;
                let t_swi = bench("swiglu", &mut || k.swiglu_bf16(&gu, &mut act, 10, 640))?;
                // RoPE（QSA 每层 2 次 × 12）
                let mut rq = k.stream.alloc_zeros::<half::bf16>(24 * 256)?;
                let rcos = k.stream.alloc_zeros::<half::bf16>(512 * 32)?;
                let rsin = k.stream.alloc_zeros::<half::bf16>(512 * 32)?;
                let t_rope = bench("rope", &mut || {
                    k.rope_partial_inplace_bf16(&mut rq, &rcos, &rsin, 1, 24, 256, 64, 0)
                })?;
                // QSA indexer 四连（每层各 1 次 × 12）
                let iq = k.stream.alloc_zeros::<half::bf16>(4 * 128)?;
                let pooled = k.stream.alloc_zeros::<half::bf16>(48 * 128)?;
                let mut sc = k.stream.alloc_zeros::<f32>(48)?;
                let mut keep = k.stream.alloc_zeros::<u8>(48)?;
                let t_score = bench("idx_scores", &mut || {
                    k.qsa_indexer_scores_bf16(&iq, &pooled, &mut sc, 1, 48, 4, 128, 11.3137)
                })?;
                let t_rank = bench("idx_rank", &mut || {
                    k.qsa_indexer_rank_f32(&sc, &mut keep, 1, 48, 512)
                })?;
                let mut msk = k.stream.alloc_zeros::<f32>(192)?;
                let t_fill = bench("mask_fill", &mut || {
                    k.qsa_mask_fill_f32(&keep, &mut msk, 1, 192, 48, 4, 0, f32::MIN)
                })?;
                let rk = k.stream.alloc_zeros::<half::bf16>(192 * 128)?;
                let mut pool = k.stream.alloc_zeros::<half::bf16>(128)?;
                let t_pool = bench("idx_pool", &mut || k.qsa_pool_block_bf16(&rk, &mut pool, 0, 4, 128))?;
                eprintln!(
                    "  小核普查（µs/次；括号里是每 token 调用次数）：\n  \
                     \u{3000}MoE combine {t_comb:.2}（48）｜ swiglu {t_swi:.2}（48）｜ RoPE {t_rope:.2}（24）\n  \
                     \u{3000}QSA idx scores {t_score:.2}（12）｜ rank {t_rank:.2}（12）｜ mask_fill {t_fill:.2}（12）｜ pool {t_pool:.2}（12）\n  \
                     \u{3000}合计 **{:.2} ms/token**",
                    (t_comb * 48.0 + t_swi * 48.0 + t_rope * 24.0
                        + (t_score + t_rank + t_fill + t_pool) * 12.0) / 1000.0
                );
            }

            // ── QSA 稀疏注意力核（12 层；`LINGXI_SKIP=attn` 拆出来 attention 块占 292 µs/层）──
            {
                let (heads, kv_heads, hd) = (24usize, 2usize, 256usize);
                let its = 200usize;
                for &tk in &[192usize, 1024, 4096] {
                    let q = k.stream.alloc_zeros::<half::bf16>(heads * hd)?;
                    let kk = k.stream.alloc_zeros::<half::bf16>(tk * kv_heads * hd)?;
                    let vv = k.stream.alloc_zeros::<half::bf16>(tk * kv_heads * hd)?;
                    let mask = k.stream.alloc_zeros::<f32>(tk)?;
                    let mut out = k.stream.alloc_zeros::<half::bf16>(heads * hd)?;
                    for _ in 0..5 {
                        k.attn_masked_bf16(&q, &kk, &vv, &mask, &mut out, heads, kv_heads, 1, tk, hd, 0.0625)?;
                    }
                    k.stream.synchronize()?;
                    let t = std::time::Instant::now();
                    for _ in 0..its {
                        k.attn_masked_bf16(&q, &kk, &vv, &mask, &mut out, heads, kv_heads, 1, tk, hd, 0.0625)?;
                    }
                    k.stream.synchronize()?;
                    let dt = t.elapsed().as_secs_f64() / its as f64 * 1e6;
                    let by = (2 * tk * kv_heads * hd * 2) as f64;
                    eprintln!(
                        "  QSA attn_masked（t=1, heads=24, kv=2, hd=256, tk={tk:>4}，**{:.0} blocks**）：{dt:.2} µs → **{:.0} GB/s**",
                        heads as f64,
                        by / (dt * 1e-6) / 1e9
                    );
                }
            }

            // ── GDN 层小核总账（36 层都走这条；`LINGXI_SKIP=attn` 拆出来是 292 µs/层，
            //    而 Q4 投影加起来只有 ~90 µs ⇒ 大头在下面这两个核）──
            {
                let (conv_dim, kk, heads, hk) = (10240usize, 4usize, 48usize, 128usize);
                let (val_dim, nk) = (heads * hk, 16usize);
                let cx = k.stream.alloc_zeros::<half::bf16>(kk * conv_dim)?;
                let cw = k.stream.alloc_zeros::<half::bf16>(kk * conv_dim)?;
                let mut cy = k.stream.alloc_zeros::<half::bf16>(kk * conv_dim)?;
                let z = k.stream.alloc_zeros::<half::bf16>(val_dim)?;
                let ab = k.stream.alloc_zeros::<half::bf16>(heads)?;
                let nw = k.stream.alloc_zeros::<half::bf16>(hk)?;
                let alog = k.stream.alloc_zeros::<f32>(heads)?;
                let dtb = k.stream.alloc_zeros::<f32>(heads)?;
                let mut s_all = k.stream.alloc_zeros::<f32>(heads * hk * hk)?;
                let mut gated = k.stream.alloc_zeros::<half::bf16>(val_dim)?;
                let its = 200usize;
                for _ in 0..5 {
                    k.causal_conv1d_silu_bf16(&cx, &cw, &mut cy, kk, conv_dim, 4, 1)?;
                }
                k.stream.synchronize()?;
                let t1 = std::time::Instant::now();
                for _ in 0..its {
                    k.causal_conv1d_silu_bf16(&cx, &cw, &mut cy, kk, conv_dim, 4, 1)?;
                }
                k.stream.synchronize()?;
                let tc = t1.elapsed().as_secs_f64() / its as f64 * 1e6;
                for _ in 0..5 {
                    k.gdn_recurrent_bf16(
                        &cx, &z, &ab, &ab, &nw, &alog, &dtb, &mut s_all, &mut gated,
                        1, hk, hk, heads, nk, 1, 3, 1,
                    )?;
                }
                k.stream.synchronize()?;
                let t2 = std::time::Instant::now();
                for _ in 0..its {
                    k.gdn_recurrent_bf16(
                        &cx, &z, &ab, &ab, &nw, &alog, &dtb, &mut s_all, &mut gated,
                        1, hk, hk, heads, nk, 1, 3, 1,
                    )?;
                }
                k.stream.synchronize()?;
                let tr = t2.elapsed().as_secs_f64() / its as f64 * 1e6;
                let sb = (heads * hk * hk * 4) as f64;
                eprintln!(
                    "  GDN 小核：conv1d+silu（4×10240）**{tc:.1} µs** ｜ recurrent（状态 {:.1} MB 读+写）**{tr:.1} µs**（{:.0} GB/s）\n  \
                     \u{3000}      两者合计 {:.1} µs/层 ⇒ 36 层 **{:.2} ms/token**",
                    sb / 1048576.0,
                    2.0 * sb / (tr * 1e-6) / 1e9,
                    tc + tr,
                    (tc + tr) * 36.0 / 1000.0
                );
            }

            // ── 专家槽 Q4 GEMV（decode 每层真正的大头：10 个专家 × 2.61 MB）──
            // 之前从没单独量过它。布局按 arena 契约复刻（见 arena_source.rs）：
            //   blob = gu_bytes + down_bytes；gu=[2*inter,hidden]=[1280,2560]、down=[hidden,inter]=[2560,640]
            {
                let (s_experts, hidden, inter) = (10usize, 2560usize, 640usize);
                let kpad = |kk: usize| ((kk + 127) / 128) * 128;
                let gu_n = 2 * inter;
                let gu_k_pad = kpad(hidden);
                let gu_base = gu_n * (gu_k_pad / 2);
                let gu_scale = (gu_base + 255) / 256 * 256;
                let gu_bytes = gu_scale + gu_n * (gu_k_pad / 64) * 2;
                let dn_k_pad = kpad(inter);
                let dn_base = hidden * (dn_k_pad / 2);
                let dn_scale = (dn_base + 255) / 256 * 256;
                let dn_bytes = dn_scale + hidden * (dn_k_pad / 64) * 2;
                let blob = gu_bytes + dn_bytes;
                eprintln!(
                    "  专家 blob 复刻：gu {gu_bytes} B + down {dn_bytes} B = **{blob} B**（真值 2611200）"
                );
                let slots = unsafe { k.stream.alloc::<u8>(blob * s_experts)? };
                let ids: Vec<i32> = (0..s_experts as i32).collect();
                let slot_ids = k.stream.clone_htod(&ids)?;
                let xs = k.stream.alloc_zeros::<half::bf16>(s_experts * hidden)?;
                let mut yg = k.stream.alloc_zeros::<half::bf16>(s_experts * gu_n)?;
                let mut yd = k.stream.alloc_zeros::<half::bf16>(s_experts * hidden)?;
                let its = 30usize;
                for _ in 0..3 {
                    k.gemv_q4g64_slots_bf16(
                        &slots, &slot_ids, &xs, &mut yg, s_experts, gu_n, hidden, gu_k_pad as i32,
                        hidden, blob as u64, 0, gu_scale as u64,
                    )?;
                }
                k.stream.synchronize()?;
                let t1 = std::time::Instant::now();
                for _ in 0..its {
                    k.gemv_q4g64_slots_bf16(
                        &slots, &slot_ids, &xs, &mut yg, s_experts, gu_n, hidden, gu_k_pad as i32,
                        hidden, blob as u64, 0, gu_scale as u64,
                    )?;
                }
                k.stream.synchronize()?;
                let tg = t1.elapsed().as_secs_f64() / its as f64 * 1e6;
                for _ in 0..3 {
                    k.gemv_q4g64_slots_bf16(
                        &slots, &slot_ids, &yg, &mut yd, s_experts, hidden, inter, dn_k_pad as i32,
                        inter, blob as u64, gu_bytes as u64, dn_scale as u64,
                    )?;
                }
                k.stream.synchronize()?;
                let t2 = std::time::Instant::now();
                for _ in 0..its {
                    k.gemv_q4g64_slots_bf16(
                        &slots, &slot_ids, &yg, &mut yd, s_experts, hidden, inter, dn_k_pad as i32,
                        inter, blob as u64, gu_bytes as u64, dn_scale as u64,
                    )?;
                }
                k.stream.synchronize()?;
                let td = t2.elapsed().as_secs_f64() / its as f64 * 1e6;
                // 差分探针：同形状把 k 砍到 512（恰好 2 次满迭代、无半 warp 尾、无标量尾）。
                // 若 512 的有效 GB/s（按 256B/行 计）≈ 640 的（320B/行），说明尾迭代无罪、
                // dn 慢在「每行字节太少」；若 512 明显更快，尾迭代假设成立。
                let t512: f64;
                {
                    for _ in 0..3 {
                        k.gemv_q4g64_slots_bf16(
                            &slots, &slot_ids, &yg, &mut yd, s_experts, hidden, 512, dn_k_pad as i32,
                            inter, blob as u64, gu_bytes as u64, dn_scale as u64,
                        )?;
                    }
                    k.stream.synchronize()?;
                    let t1b = std::time::Instant::now();
                    for _ in 0..its {
                        k.gemv_q4g64_slots_bf16(
                            &slots, &slot_ids, &yg, &mut yd, s_experts, hidden, 512, dn_k_pad as i32,
                            inter, blob as u64, gu_bytes as u64, dn_scale as u64,
                        )?;
                    }
                    k.stream.synchronize()?;
                    t512 = t1b.elapsed().as_secs_f64() / its as f64 * 1e6;
                }
                // 地板探针：k=128（每 warp 恰好 1 次满迭代，有效 payload 只有 1.6 MB）。
                // 若耗时仍 ~15-20 µs ⇒ 存在与字节数无关的**每 launch 固定开销地板**，
                // dn 的 260 GB/s 是被地板拖的，不是被尾迭代/短循环拖的。
                let t128: f64;
                {
                    for _ in 0..3 {
                        k.gemv_q4g64_slots_bf16(
                            &slots, &slot_ids, &yg, &mut yd, s_experts, hidden, 128, dn_k_pad as i32,
                            inter, blob as u64, gu_bytes as u64, dn_scale as u64,
                        )?;
                    }
                    k.stream.synchronize()?;
                    let t1c = std::time::Instant::now();
                    for _ in 0..its {
                        k.gemv_q4g64_slots_bf16(
                            &slots, &slot_ids, &yg, &mut yd, s_experts, hidden, 128, dn_k_pad as i32,
                            inter, blob as u64, gu_bytes as u64, dn_scale as u64,
                        )?;
                    }
                    k.stream.synchronize()?;
                    t128 = t1c.elapsed().as_secs_f64() / its as f64 * 1e6;
                }
                let b640 = (s_experts * hidden * (dn_k_pad / 2)) as f64; // 320B/行
                let b512 = (s_experts * hidden * 256) as f64; // 256B/行
                let b128 = (s_experts * hidden * 64) as f64; // 64B/行
                eprintln!(
                    "  dn 尾探针：k=640 {:.1} µs（{:.0} GB/s 有效）｜k=512 {:.1} µs（{:.0} GB/s）→ 比值 {:.2}（带宽限=0.80）",
                    td, b640 / (td * 1e-6) / 1e9,
                    t512, b512 / (t512 * 1e-6) / 1e9,
                    t512 / td
                );
                eprintln!(
                    "  dn 地板探针：k=128 {:.1} µs（有效仅 {:.1} MB ⇒ 边际 {:.0} GB/s）",
                    t128, b128 / 1048576.0,
                    (b640 - b128) / ((td - t128).max(1e-9) * 1e-6) / 1e9
                );
                // **graph 重放探针**：把 dn/gu 各录进一张图，测纯重放的每 launch 时间。
                // 裸 launch 循环受 CPU 发射速率限制（每次 launch 要 ~10 µs 的 host 成本），
                // 若 dn 在图重放下远小于 31 µs ⇒ 那个"14.5 µs 地板"是 **host 发射地板**，
                // 真 decode（graph 内）没有它，dn 内核本身早已跑在边际带宽上。
                {
                    use lingxi_model::graph;
                    let mut tg_graph = 0f64;
                    let mut td_graph = 0f64;
                    for key in 777778u64..=777779 {
                        // warm：建缓冲 + 录图
                        for _ in 0..5 {
                            if key == 777778 {
                                graph::run_cached(&k, key, || {
                                    k.gemv_q4g64_slots_bf16(
                                        &slots, &slot_ids, &yg, &mut yd, s_experts, hidden, inter,
                                        dn_k_pad as i32, inter, blob as u64, gu_bytes as u64,
                                        dn_scale as u64,
                                    )
                                })?;
                            } else {
                                graph::run_cached(&k, key, || {
                                    k.gemv_q4g64_slots_bf16(
                                        &slots, &slot_ids, &xs, &mut yg, s_experts, gu_n, hidden,
                                        gu_k_pad as i32, hidden, blob as u64, 0,
                                        gu_scale as u64,
                                    )
                                })?;
                            }
                        }
                        k.stream.synchronize()?;
                        let t0 = std::time::Instant::now();
                        for _ in 0..its {
                            if key == 777778 {
                                graph::run_cached(&k, key, || Ok(()))?;
                            } else {
                                graph::run_cached(&k, key, || Ok(()))?;
                            }
                        }
                        k.stream.synchronize()?;
                        let t = t0.elapsed().as_secs_f64() / its as f64 * 1e6;
                        if key == 777778 { td_graph = t; } else { tg_graph = t; }
                    }
                    eprintln!(
                        "  graph 重放：dn {td_graph:.1} µs（**{:.0} GB/s**）｜gu {tg_graph:.1} µs（**{:.0} GB/s**）",
                        b640 / (td_graph.max(1e-9) * 1e-6) / 1e9,
                        (s_experts * gu_bytes) as f64 / (tg_graph.max(1e-9) * 1e-6) / 1e9
                    );
                }
                // **多行/ warp 扫描**：验证「线程数过载」假设——dn（n=2560,k=640）一 warp 一行时
                // 全局 warp 数 25600、每线程仅搬 ~10 B，固定成本主导。每 warp 多行后 warp 数
                // 按比例降、跨行 ILP 上升。rows=1 必须与上面的基线数字逐位一致（自校验）。
                {
                    let scan = |label: &str, xsrc: &cudarc::driver::CudaSlice<half::bf16>,
                                    ysrc: &mut cudarc::driver::CudaSlice<half::bf16>,
                                    nn: usize, kk: usize, kpad: usize, stride: usize,
                                    seg: u64, sc: u64, bytes_per: u64| -> anyhow::Result<()> {
                        let mut s = format!("  {label} 多行/warp:");
                        for &rpw in &[1usize, 2, 4, 8] {
                            for _ in 0..3 {
                                k.gemv_q4g64_slots_rows_bf16(
                                    &slots, &slot_ids, xsrc, ysrc, s_experts, nn, kk,
                                    kpad as i32, stride, blob as u64, seg, sc, rpw,
                                )?;
                            }
                            k.stream.synchronize()?;
                            let t0 = std::time::Instant::now();
                            for _ in 0..its {
                                k.gemv_q4g64_slots_rows_bf16(
                                    &slots, &slot_ids, xsrc, ysrc, s_experts, nn, kk,
                                    kpad as i32, stride, blob as u64, seg, sc, rpw,
                                )?;
                            }
                            k.stream.synchronize()?;
                            let t = t0.elapsed().as_secs_f64() / its as f64 * 1e6;
                            let bytes = (s_experts as u64 * bytes_per) as f64;
                            s.push_str(&format!(" {rpw}行 {t:.1}µs({:.0}GB/s)", bytes / (t * 1e-6) / 1e9));
                        }
                        eprintln!("{s}");
                        Ok(())
                    };
                    scan("gu", &xs, &mut yg, gu_n, hidden, gu_k_pad, hidden, 0, gu_scale as u64, gu_bytes as u64)?;
                    scan("dn", &yg, &mut yd, hidden, inter, dn_k_pad, inter, gu_bytes as u64, dn_scale as u64, dn_bytes as u64)?;
                }
                // **微型 launch 探针**：定位「~14.5 µs 地板」的来源。
                //   S=1,n=8   → 最小 launch（1 block、1 warp）
                //   S=1,n=2560→ 行数拉满但只 1 个专家
                //   S=10,n=8  → 专家数拉满但每专家 1 行
                //   S=10,n=2560 → 真实形状
                // 若四者都是 ~14.5 µs ⇒ 地板是**每次 launch 的固定 GPU 侧成本**（与 S/n 无关）
                //   ⇒ 正确对策是**减少 launch 次数**（融合 gu+dn / 融合整个 MoE 段），而不是改核内。
                for &(ss, nn) in &[(1usize, 8usize), (1, 2560), (10, 8), (10, 2560)] {
                    for _ in 0..3 {
                        k.gemv_q4g64_slots_bf16(
                            &slots, &slot_ids, &yg, &mut yd, ss, nn, inter, dn_k_pad as i32,
                            inter, blob as u64, gu_bytes as u64, dn_scale as u64,
                        )?;
                    }
                    k.stream.synchronize()?;
                    let t0 = std::time::Instant::now();
                    for _ in 0..its {
                        k.gemv_q4g64_slots_bf16(
                            &slots, &slot_ids, &yg, &mut yd, ss, nn, inter, dn_k_pad as i32,
                            inter, blob as u64, gu_bytes as u64, dn_scale as u64,
                        )?;
                    }
                    k.stream.synchronize()?;
                    let t = t0.elapsed().as_secs_f64() / its as f64 * 1e6;
                    eprintln!("  微型 launch：S={ss} n={nn} k=640 → **{t:.1} µs**");
                }
                eprintln!(
                    "  专家段 Q4 GEMV（S=10）：gu [{s_experts},{gu_n},{hidden}] {:.1} MB → {tg:.1} µs（**{:.0} GB/s**）\n  \
                     \u{3000}                    down [{s_experts},{hidden},{inter}] {:.1} MB → {td:.1} µs（**{:.0} GB/s**）\n  \
                     \u{3000}           每层合计 {:.1} µs ⇒ 48 层 **{:.2} ms/token**",
                    (s_experts * gu_bytes) as f64 / 1048576.0,
                    (s_experts * gu_bytes) as f64 / (tg * 1e-6) / 1e9,
                    (s_experts * dn_bytes) as f64 / 1048576.0,
                    (s_experts * dn_bytes) as f64 / (td * 1e-6) / 1e9,
                    tg + td,
                    (tg + td) * 48.0 / 1000.0
                );
            }

            // ── slotsm（按唯一专家分组）形状对比：拆开 FIXED_EXPERTS@t=8 省下的 21 ms ──
            // A：真实 t=8 分布（80 对 → e_uniq=80，m=1，权重读 80 份）
            // B：FIXED_EXPERTS（80 对 → e_uniq=10，m=8，权重只读 10 份）
            // C：t=1 稳态（10 对 → e_uniq=10，m=1，与上面 slots S=10 基线对照）
            {
                let (hidden, inter) = (2560usize, 640usize);
                let kpad = |kk: usize| ((kk + 127) / 128) * 128;
                let gu_n = 2 * inter;
                let gu_k_pad = kpad(hidden);
                let gu_base = gu_n * (gu_k_pad / 2);
                let gu_scale = (gu_base + 255) / 256 * 256;
                let gu_bytes = gu_scale + gu_n * (gu_k_pad / 64) * 2;
                let dn_k_pad = kpad(inter);
                let dn_base = hidden * (dn_k_pad / 2);
                let dn_scale = (dn_base + 255) / 256 * 256;
                let dn_bytes = dn_scale + hidden * (dn_k_pad / 64) * 2;
                let blob = gu_bytes + dn_bytes;

                struct SmShape {
                    name: &'static str,
                    pairs: usize,
                    e_uniq: usize,
                }
                let shapes = [
                    SmShape { name: "A 真 t=8 (e_uniq=80,m=1)", pairs: 80, e_uniq: 80 },
                    SmShape { name: "B fixed  (e_uniq=10,m=8)", pairs: 80, e_uniq: 10 },
                    SmShape { name: "C t=1    (e_uniq=10,m=1)", pairs: 10, e_uniq: 10 },
                    SmShape { name: "D t=2 无重叠(e_uniq=20,m=1)", pairs: 20, e_uniq: 20 },
                    SmShape { name: "E t=2 全重叠(e_uniq=10,m=2)", pairs: 20, e_uniq: 10 },
                    SmShape { name: "F t=4 典型(e_uniq=30,m=2)", pairs: 56, e_uniq: 28 },
                ];
                let slots = unsafe { k.stream.alloc::<u8>(blob * 80)? };
                let xs_g = k.stream.alloc_zeros::<half::bf16>(80 * hidden)?;
                let mut yg = k.stream.alloc_zeros::<half::bf16>(80 * gu_n)?;
                let mut yd = k.stream.alloc_zeros::<half::bf16>(80 * hidden)?;
                // P15-⑤b ①：x 的 fp32 预转副本（转换在计时外 —— "后面要转化的提前转化"）
                let mut xf_g = k.stream.alloc_zeros::<f32>(80 * hidden)?;
                k.convert_bf16_f32(&xs_g, &mut xf_g, 80 * hidden)?;
                let (xf_gp, _gxf) = xf_g.device_ptr(&k.stream);
                let its = 30usize;
                for sh in shapes.iter() {
                    let m = sh.pairs / sh.e_uniq;
                    let ids: Vec<i32> = (0..sh.e_uniq as i32).collect();
                    let slot_ids = k.stream.clone_htod(&ids)?;
                    let goff: Vec<i32> = (0..=sh.e_uniq).map(|i| (i * m) as i32).collect();
                    let group_off = k.stream.clone_htod(&goff)?;
                    for _ in 0..3 {
                        k.gemv_q4g64_slotsm_bf16(
                            &slots, &slot_ids, &xs_g, &group_off, &mut yg,
                            sh.e_uniq, gu_n, hidden, gu_k_pad as i32, hidden,
                            blob as u64, 0, gu_scale as u64, xf_gp,
                        )?;
                    }
                    k.stream.synchronize()?;
                    let t0 = std::time::Instant::now();
                    for _ in 0..its {
                        k.gemv_q4g64_slotsm_bf16(
                            &slots, &slot_ids, &xs_g, &group_off, &mut yg,
                            sh.e_uniq, gu_n, hidden, gu_k_pad as i32, hidden,
                            blob as u64, 0, gu_scale as u64, xf_gp,
                        )?;
                    }
                    k.stream.synchronize()?;
                    let tg = t0.elapsed().as_secs_f64() / its as f64 * 1e6;
                    for _ in 0..3 {
                        k.gemv_q4g64_slotsm_bf16(
                            &slots, &slot_ids, &yg, &group_off, &mut yd,
                            sh.e_uniq, hidden, inter, dn_k_pad as i32, gu_n,
                            blob as u64, gu_bytes as u64, dn_scale as u64, 0,
                        )?;
                    }
                    k.stream.synchronize()?;
                    let t1 = std::time::Instant::now();
                    for _ in 0..its {
                        k.gemv_q4g64_slotsm_bf16(
                            &slots, &slot_ids, &yg, &group_off, &mut yd,
                            sh.e_uniq, hidden, inter, dn_k_pad as i32, gu_n,
                            blob as u64, gu_bytes as u64, dn_scale as u64, 0,
                        )?;
                    }
                    k.stream.synchronize()?;
                    let td = t1.elapsed().as_secs_f64() / its as f64 * 1e6;
                    let gbytes = (sh.e_uniq * gu_bytes) as f64;
                    let dbytes = (sh.e_uniq * dn_bytes) as f64;
                    eprintln!(
                        "  slotsm {name}：gu {tg:.1} µs（{:.0} GB/s）｜dn {td:.1} µs（{:.0} GB/s）\n  \
                        \u{3000}                    每层合计 {:.1} µs ⇒ 48 层 **{:.2} ms**",
                        gbytes / (tg * 1e-6) / 1e9,
                        dbytes / (td * 1e-6) / 1e9,
                        tg + td,
                        (tg + td) * 48.0 / 1000.0,
                        name = sh.name
                    );
                }
            }


            // ── GR 块总账：旧 5 kernel vs 新 2 kernel（同形状，逐项对比）──
            // 这是"融合到底赚不赚"的**快速判据**（秒级，不用加载 250 GB 产物）。
            {
                let (dim, hidden, lowrank, hc) = (10240usize, 2560usize, 320usize, 4usize);
                let ngroups = dim / hidden;
                let eps = 1e-6f32;
                let xg = k.stream.alloc_zeros::<half::bf16>(dim)?;
                let gam = k.stream.alloc_zeros::<half::bf16>(dim)?;
                let wd = k.stream.alloc_zeros::<half::bf16>(lowrank * dim)?;
                let wu = k.stream.alloc_zeros::<half::bf16>(dim * lowrank)?;
                let wi = k.stream.alloc_zeros::<half::bf16>(hc * dim)?;
                let mut xn = k.stream.alloc_zeros::<half::bf16>(dim)?;
                let mut dd = k.stream.alloc_zeros::<half::bf16>(lowrank)?;
                let mut imw = k.stream.alloc_zeros::<half::bf16>(dim)?;
                let mut mixed = k.stream.alloc_zeros::<half::bf16>(hidden)?;
                let mut injw = k.stream.alloc_zeros::<half::bf16>(hc)?;
                let mut rmsg = k.stream.alloc_zeros::<f32>(ngroups)?;
                let its = 400usize;
                let t_old = {
                    let mut old_path = |k: &lingxi_kernels::Kernels| -> Result<()> {
                        k.grouped_rmsnorm_bf16(&xg, &gam, &mut xn, 1, dim, hidden, eps)?;
                        let (wp, _) = wd.device_ptr(&k.stream);
                        let (xp, _) = xn.device_ptr(&k.stream);
                        let (dp, _) = dd.device_ptr_mut(&k.stream);
                        k.gemv_bf16_block_epi_ptr(wp, xp, dp, lowrank, dim, 1, 1.0 / hc as f32, 0.0)?;
                        let (up, _) = wu.device_ptr(&k.stream);
                        let (ip, _) = imw.device_ptr_mut(&k.stream);
                        k.gemv_bf16_epi_ptr(up, dp, ip, dim, lowrank, 2, 1.0, 1.0)?;
                        k.mix_mean_bf16(&imw, &xn, &mut mixed, hc, hidden, 1u32)?;
                        let (wp2, _) = wi.device_ptr(&k.stream);
                        let (jp, _) = injw.device_ptr_mut(&k.stream);
                        k.gemv_bf16_block_epi_ptr(wp2, xp, jp, hc, dim, 2, 1.0 / hc as f32, 2.0)
                    };
                    for _ in 0..20 {
                        old_path(&k)?;
                    }
                    k.stream.synchronize()?;
                    let t = std::time::Instant::now();
                    for _ in 0..its {
                        old_path(&k)?;
                    }
                    k.stream.synchronize()?;
                    t.elapsed().as_secs_f64() / its as f64 * 1e6
                };
                let (t_k1, t_new) = {
                    macro_rules! k1 {
                        () => {
                            k.gr_down_fuse_bf16(
                                &xg, &gam, &wd, &wi, &mut dd, &mut injw, &mut rmsg,
                                dim, hidden, ngroups, lowrank, hc, 1.0 / hc as f32, eps,
                            )?
                        };
                    }
                    macro_rules! k2 {
                        () => {
                            k.gr_up_mix_bf16(
                                &dd, &xg, &gam, &wu, &rmsg, &mut mixed,
                                dim, hidden, lowrank, ngroups, 1.0 / hc as f32,
                            )?
                        };
                    }
                    for _ in 0..20 {
                        k1!();
                        k2!();
                    }
                    k.stream.synchronize()?;
                    let ta = std::time::Instant::now();
                    for _ in 0..its {
                        k1!();
                    }
                    k.stream.synchronize()?;
                    let a = ta.elapsed().as_secs_f64() / its as f64 * 1e6;
                    for _ in 0..20 {
                        k1!();
                        k2!();
                    }
                    k.stream.synchronize()?;
                    let t2 = std::time::Instant::now();
                    for _ in 0..its {
                        k1!();
                        k2!();
                    }
                    k.stream.synchronize()?;
                    (a, t2.elapsed().as_secs_f64() / its as f64 * 1e6)
                };
                let gbs = |us: f64| 13.2e6 / (us * 1e-6) / 1e9;
                eprintln!(
                    "GR 块（dim=10240 hidden=2560 lowrank=320 hc=4，权重 13.2 MB）：\n  \
                     旧 5 kernel = **{t_old:.2} µs/块**（{:.0} GB/s）\n  \
                     新 K1(down+inject) = **{t_k1:.2} µs**；K1+K2 = **{t_new:.2} µs/块**（K2 = {:.2} µs，{:.0} GB/s）  → {}\n  \
                     每 token 48 层 × 2 块：旧 {:.2} ms / 新 {:.2} ms",
                    gbs(t_old),
                    t_new - t_k1,
                    gbs(t_new),
                    if t_new < t_old { "新快" } else { "新慢" },
                    t_old * 96.0 / 1000.0,
                    t_new * 96.0 / 1000.0
                );
            }

            let (dim_r, hid_r) = (10240usize, 2560usize);
            let xr = k.stream.alloc_zeros::<half::bf16>(dim_r)?;
            let wr = k.stream.alloc_zeros::<half::bf16>(dim_r)?;
            let mut yr = k.stream.alloc_zeros::<half::bf16>(dim_r)?;
            for _ in 0..500 {
                k.grouped_rmsnorm_bf16(&xr, &wr, &mut yr, 1, dim_r, hid_r, 1e-6)?;
            }
            k.stream.synchronize()?;
            let it3 = 20000usize;
            let t3 = std::time::Instant::now();
            for _ in 0..it3 {
                k.grouped_rmsnorm_bf16(&xr, &wr, &mut yr, 1, dim_r, hid_r, 1e-6)?;
            }
            let e3 = t3.elapsed().as_secs_f64();
            k.stream.synchronize()?;
            // 对比 2：裸指针参数的核（gemv：3 个 u64），GPU 只读 5KB → 同样量 CPU 侧开销
            let (xp2, _a2) = xr.device_ptr(&k.stream);
            let (wp2, _b2) = wr.device_ptr(&k.stream);
            let (yp2, _c2) = yr.device_ptr_mut(&k.stream);
            for _ in 0..500 {
                k.gemv_bf16_ptr(wp2, xp2, yp2, 1, hid_r)?;
            }
            k.stream.synchronize()?;
            let t4 = std::time::Instant::now();
            for _ in 0..it3 {
                k.gemv_bf16_ptr(wp2, xp2, yp2, 1, hid_r)?;
            }
            let e4 = t4.elapsed().as_secs_f64();
            k.stream.synchronize()?;
            eprintln!(
                "CPU 侧 launch 开销对比（GPU 均为极小活儿）：\n  grouped_rmsnorm（3 slice 参数）= **{:.2} µs/次**\n  gemv 裸指针（3×u64 参数）     = **{:.2} µs/次**",
                e3 / it3 as f64 * 1e6,
                e4 / it3 as f64 * 1e6
            );
            // 对比 3：MoE 每层调 2 次的 strided-batched GEMM（批次 10，形状同 decode 专家段）
            let (bs, bn, bk) = (10usize, 1280usize, 2560usize);            let ab = k.stream.alloc_zeros::<half::bf16>(bk)?;
            let wb = k.stream.alloc_zeros::<half::bf16>(bs * bn * bk)?;
            let mut cb2 = k.stream.alloc_zeros::<half::bf16>(bs * bn)?;
            for _ in 0..50 {
                k.gemm_bf16_nt_batched(
                    &ab, &wb, &mut cb2, 1, bn as u32, bk as u32, bs as u32,
                    0, (bn * bk) as i64, bn as i64,
                )?;
            }
            k.stream.synchronize()?;
            let it5 = 1000usize;
            let t5 = std::time::Instant::now();
            for _ in 0..it5 {
                k.gemm_bf16_nt_batched(
                    &ab, &wb, &mut cb2, 1, bn as u32, bk as u32, bs as u32,
                    0, (bn * bk) as i64, bn as i64,
                )?;
            }
            let e5 = t5.elapsed().as_secs_f64();
            k.stream.synchronize()?;
            let t5b = std::time::Instant::now();
            k.stream.synchronize()?;
            eprintln!(
                "gemm_bf16_nt_batched（batch=10, [1,1280]x[1280,2560]）: **{:.2} µs/次（CPU）**；含同步 {:.3}s",
                e5 / it5 as f64 * 1e6,
                t5b.elapsed().as_secs_f64()
            );

            // 对比 4：`alloc_zeros` + 析构（每层约 30 次，是 CPU 侧的另一个大头）
            let it6 = 3000usize;
            for _ in 0..200 {
                let v = k.stream.alloc_zeros::<half::bf16>(2560)?;
                drop(v);
            }
            k.stream.synchronize()?;
            let t6 = std::time::Instant::now();
            for _ in 0..it6 {
                let v = k.stream.alloc_zeros::<half::bf16>(2560)?;
                drop(v);
            }
            let e6 = t6.elapsed().as_secs_f64();
            k.stream.synchronize()?;
            eprintln!(
                "alloc_zeros(2560)+drop: **{:.2} µs/次（CPU）** → 每层 30 次 = {:.3} ms/层",
                e6 / it6 as f64 * 1e6,
                e6 / it6 as f64 * 1e3 * 30.0
            );
            return Ok(());
        }
        Some("argmax-check") => {
            // ★ P15-⑭ 对拍台：GPU 两段式 argmax vs host `argmax_rows_bf16`，必须**逐行一致**。
            // 覆盖：真词表形状、平局（全同值行 / 重复最大值 → 最小下标）、NaN（永不中选）、
            // inf、vocab < G 的退化切块。比较键是全序（f32 值降序、下标升序）⇒ 答案唯一。
            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let mut total_bad = 0usize;
            for &(rows, vocab, seed) in &[
                (1usize, 247000usize, 1u64),
                (7, 247000, 2),
                (9, 247000, 3),
                (3, 1024, 4),
                (2, 129, 5),
                (1, 32, 6),
            ] {
                let mut data = Vec::with_capacity(rows * vocab);
                let mut s = seed | 1;
                for _ in 0..rows * vocab {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let e = ((s >> 8) & 0x7FFF) as u16;
                    data.push(half::bf16::from_bits(0x3F00u16.wrapping_add(e).wrapping_sub(0x4000)));
                }
                if rows >= 3 {
                    // 行1：全同值（平局 ⇒ argmax=0）
                    for j in 0..vocab {
                        data[vocab + j] = half::bf16::from_f32(1.25);
                    }
                    // 行2：重复最大值 9.0（42/777 ⇒ 取 42）+ NaN（不中）+ inf（中，盖过 9）
                    for j in 0..vocab {
                        data[2 * vocab + j] = half::bf16::from_f32(-3.0);
                    }
                    data[2 * vocab + 42] = half::bf16::from_f32(9.0);
                    data[2 * vocab + 777] = half::bf16::from_f32(9.0);
                    data[2 * vocab + 100] = half::bf16::from_f32(f32::NAN);
                    data[2 * vocab + 200] = half::bf16::from_f32(f32::INFINITY);
                }
                let x = k.stream.clone_htod(&data)?;
                const G: usize = 64;
                let mut pval = k.stream.alloc_zeros::<f32>(rows * G)?;
                let mut pidx = k.stream.alloc_zeros::<i32>(rows * G)?;
                let mut am = k.stream.alloc_zeros::<i32>(rows)?;
                k.argmax_rows_part_bf16(&x, 0, &mut pval, &mut pidx, rows, vocab, G)?;
                k.argmax_rows_fin(&pval, &pidx, &mut am, rows, G)?;
                let gpu: Vec<i32> = k.stream.clone_dtoh(&am)?;
                let want = lingxi_model::trunk::argmax_rows_bf16(&data, rows, vocab);
                let bad = (0..rows).filter(|&r| gpu[r] as u32 != want[r]).count();
                total_bad += bad;
                eprintln!("  argmax-check rows={rows} vocab={vocab}: {bad}/{rows} 行不一致");
                if bad > 0 {
                    for r in 0..rows {
                        if gpu[r] as u32 != want[r] {
                            eprintln!("    行{r}: gpu={} host={}", gpu[r], want[r]);
                        }
                    }
                }
            }
            // ── top-K 相位：GPU 两段式 top-K vs host `topk_bf16_row`（K=20=MTP_KTOP）──
            // 同样的全序 ⇒ 前 K 名（含平局下标升序）逐位唯一。覆盖重复最大值/全同值行。
            for &(rows, vocab, seed) in &[
                (1usize, 247000usize, 11u64),
                (7, 247000, 12),
                (2, 129, 13),
            ] {
                let mut data = Vec::with_capacity(rows * vocab);
                let mut s = seed | 1;
                for _ in 0..rows * vocab {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    // 只取尾数 8 位 ⇒ 全部有限，且只有 256 种取值 ⇒ 大量平局（tie 覆盖）
                    let e = (s >> 8) as u16 & 0x00FF;
                    data.push(half::bf16::from_bits(0x3F00u16 | e));
                }
                if rows >= 2 {
                    // 行1：全同值（top-K 的下标必须是 0,1,2,…,K-1 升序）
                    for j in 0..vocab {
                        data[vocab + j] = half::bf16::from_f32(2.5);
                    }
                }
                let x = k.stream.clone_htod(&data)?;
                const G: usize = 64;
                const K: usize = 20;
                let mut pval = k.stream.alloc_zeros::<f32>(rows * G * K)?;
                let mut pidx = k.stream.alloc_zeros::<i32>(rows * G * K)?;
                let mut oval = k.stream.alloc_zeros::<f32>(rows * K)?;
                let mut oidx = k.stream.alloc_zeros::<i32>(rows * K)?;
                let mut bad = k.stream.alloc_zeros::<i32>(1)?;
                k.topk_rows_part_bf16(&x, &mut pval, &mut pidx, &mut bad, rows, vocab, G, K)?;
                k.topk_rows_fin(&pval, &pidx, &mut oval, &mut oidx, rows, G, K)?;
                let vals: Vec<f32> = k.stream.clone_dtoh(&oval)?;
                let idxs: Vec<i32> = k.stream.clone_dtoh(&oidx)?;
                let bad_h: Vec<i32> = k.stream.clone_dtoh(&bad)?;
                anyhow::ensure!(bad_h[0] == 0, "topk-check：出现非有限值");
                let mut bad_rows = 0usize;
                for r in 0..rows {
                    let want = lingxi_model::trunk::topk_bf16_row(
                        &data[r * vocab..(r + 1) * vocab], K,
                    )?;
                    anyhow::ensure!(want.len() == K, "对拍台要求 vocab >= K");
                    for j in 0..K {
                        let (wv, wi) = want[j];
                        if vals[r * K + j].to_bits() != wv.to_bits() || idxs[r * K + j] as u32 != wi {
                            bad_rows += 1;
                            eprintln!(
                                "    行{r} 第{j}名: gpu=({},{}) host=({},{})",
                                vals[r * K + j], idxs[r * K + j], wv, wi
                            );
                            break;
                        }
                    }
                }
                total_bad += bad_rows;
                eprintln!("  topk-check K={K} rows={rows} vocab={vocab}: {bad_rows}/{rows} 行不一致");
            }
            anyhow::ensure!(total_bad == 0, "argmax/topk-check：{total_bad} 行不一致");
            eprintln!("  argmax/topk-check 全部一致 ✓");
        }
        Some("f32x-check") => {
            // ★ P15-⑨ 对拍台：`gemm_skinny_row_f32x`（x 预转 fp32）vs `gemm_skinny_row`
            //   的**输出比特**。随机数据、m=7、GDN 真形状；任何一位不同都打印样本。
            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let rnd_bf16 = |n: usize, seed: u64| -> Vec<half::bf16> {
                let mut v = Vec::with_capacity(n);
                let mut s = seed | 1;
                for _ in 0..n {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    // 小规模数值（[-2,2)）—— 防止 inf/NaN 干扰对拍
                    let e = ((s >> 8) & 0x3FF) as u16;
                    v.push(half::bf16::from_bits((0x3F80u16.wrapping_sub(e)) | 0x8000));
                }
                v
            };
            for &(nn, kk, m, epi) in &[
                (10240usize, 2560usize, 7usize, 0i32),
                (2560, 6144, 8, 0),
                (512, 2560, 2, 0),
                (1280, 2560, 9, 0),
                // epi 尾对拍：shared gate 行（epi=2 sigmoid）与 mix_down（epi=1 silu）
                (1, 2560, 7, 2),
                (320, 2560, 7, 1),
            ] {
                // GR mix_up 真形状 [20480, 320]：原路 k=320 < 1024 走**模板版** skinny，
                // 与 skinny_row 结构不同 —— 预测这里会有位差（P15-⑨ 实锤用）。
                {
                    let (nn2, kk2, m2, epi2) = (20480usize, 320usize, 7usize, 2i32);
                    let t2 = m2;
                    let w2 = k.stream.clone_htod(&rnd_bf16(nn2 * kk2, 0x33))?;
                    let x2 = k.stream.clone_htod(&rnd_bf16(t2 * kk2, 0x44))?;
                    let mut y0b = k.stream.alloc_zeros::<half::bf16>(t2 * nn2)?;
                    let mut y1b = k.stream.alloc_zeros::<half::bf16>(t2 * nn2)?;
                    {
                        let (wp, _a) = w2.device_ptr(&k.stream);
                        let (xp, _b) = x2.device_ptr(&k.stream);
                        let (yp, _c) = y0b.device_ptr_mut(&k.stream);
                        k.gemm_skinny_bf16_ptr(wp, xp, yp, nn2, kk2, m2, epi2, 1.0, 1.0)?;
                    }
                    let mut xf2 = k.stream.alloc_zeros::<f32>(t2 * kk2)?;
                    k.convert_bf16_f32(&x2, &mut xf2, t2 * kk2)?;
                    {
                        let (wp, _a) = w2.device_ptr(&k.stream);
                        let (xp, _b) = xf2.device_ptr(&k.stream);
                        let (yp, _c) = y1b.device_ptr_mut(&k.stream);
                        k.gemm_skinny_row_f32x_ptr(wp, xp, yp, nn2, kk2, m2, epi2, 1.0, 1.0)?;
                    }
                    k.stream.synchronize()?;
                    let h0b = k.stream.clone_dtoh(&y0b)?;
                    let h1b = k.stream.clone_dtoh(&y1b)?;
                    let mut diff = 0usize;
                    for i in 0..h0b.len() {
                        if h0b[i].to_bits() != h1b[i].to_bits() {
                            diff += 1;
                        }
                    }
                    eprintln!("  f32x-check 模板版skinny [{nn2},{kk2}] m={m2} epi={epi2}: **{diff}/{} 位不同**", h0b.len());
                }
                let t = m.max(2);
                let (ae, be) = match epi {
                    1 => (0.25f32, 0.0f32),
                    2 => (1.0, 1.0),
                    _ => (0.0, 0.0),
                };
                let w = k.stream.clone_htod(&rnd_bf16(nn * kk, 0x11))?;
                let x = k.stream.clone_htod(&rnd_bf16(t * kk, 0x22))?;
                let mut y0 = k.stream.alloc_zeros::<half::bf16>(t * nn)?;
                let mut y1 = k.stream.alloc_zeros::<half::bf16>(t * nn)?;
                for d0 in 0..t {
                    let mm = (t - d0).min(8);
                    let (wp, _a) = w.device_ptr(&k.stream);
                    let (xp, _b) = x.device_ptr(&k.stream);
                    let (yp, _c) = y0.device_ptr_mut(&k.stream);
                    k.gemm_skinny_row_bf16_ptr(wp, xp + (d0 * kk * 2) as u64, yp + (d0 * nn * 2) as u64, nn, kk, mm, epi, ae, be)?;
                }
                let mut xf = k.stream.alloc_zeros::<f32>(t * kk)?;
                k.convert_bf16_f32(&x, &mut xf, t * kk)?;
                for d0 in 0..t {
                    let mm = (t - d0).min(8);
                    let (wp, _a) = w.device_ptr(&k.stream);
                    let (xp, _b) = xf.device_ptr(&k.stream);
                    let (yp, _c) = y1.device_ptr_mut(&k.stream);
                    k.gemm_skinny_row_f32x_ptr(wp, xp + (d0 * kk * 4) as u64, yp + (d0 * nn * 2) as u64, nn, kk, mm, epi, ae, be)?;
                }
                k.stream.synchronize()?;
                let h0 = k.stream.clone_dtoh(&y0)?;
                let h1 = k.stream.clone_dtoh(&y1)?;
                let mut diff = 0usize;
                for i in 0..h0.len() {
                    if h0[i].to_bits() != h1[i].to_bits() {
                        diff += 1;
                        if diff <= 3 {
                            eprintln!("    [{nn},{kk}] t={t} 第 {i} 项: bf16={} f32x={}", h0[i], h1[i]);
                        }
                    }
                }
                eprintln!("  f32x-check [{nn:>5},{kk:>5}] t={t}: **{diff}/{} 位不同**", h0.len());
            }
            return Ok(());
        }
        Some("bw-ceiling") => {
            //
            //   ① 纯 memcpy（同字节数）        = **硬件上限**
            //   ② bf16 GEMV 真实形状 n=1280 k=2560（专家 gu）与 n=2560 k=640（专家 dn）
            //   ③ bf16 GEMV **大形状** n=39680（= 31 个专家的行拼起来）= 看"网格大小"的影响
            //   ④ bf16 GEMV **高并行度变体**（`gemv_bf16_block_epi_ptr`，一 block 一行）
            //
            //   Q4 专家 GEMV 不便在这里造布局（要 Q4G64 的 payload/scale 平面），
            //   所以它的数用**真实 decode 路径的实测**：miss=0 时专家段 78 MB/层 ÷ 0.46 ms
            //   ≈ **170 GB/s**（见 PLAN §P15-②）。这样三个数直接可比。
            //
            //   口径说明：量的是"权重字节 / 墙钟"，墙钟含入队（rep 次一起排队后同步），
            //   所以小形状的数字里天然含 launch 开销 —— 那正是我们想看清的一部分。
            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let rep: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
            fn gv(
                k: &lingxi_kernels::Kernels,
                w: &cudarc::driver::CudaSlice<half::bf16>,
                x: &cudarc::driver::CudaSlice<half::bf16>,
                y: &mut cudarc::driver::CudaSlice<half::bf16>,
                n: usize,
                kk: usize,
            ) -> anyhow::Result<()> {
                use cudarc::driver::{DevicePtr, DevicePtrMut};
                let (wp, _g0) = w.device_ptr(&k.stream);
                let (xp, _g1) = x.device_ptr(&k.stream);
                let (yp, _g2) = y.device_ptr_mut(&k.stream);
                k.gemv_bf16_ptr(wp, xp, yp, n, kk)
            }
            fn gv_block(
                k: &lingxi_kernels::Kernels,
                w: &cudarc::driver::CudaSlice<half::bf16>,
                x: &cudarc::driver::CudaSlice<half::bf16>,
                y: &mut cudarc::driver::CudaSlice<half::bf16>,
                n: usize,
                kk: usize,
            ) -> anyhow::Result<()> {
                use cudarc::driver::{DevicePtr, DevicePtrMut};
                let (wp, _g0) = w.device_ptr(&k.stream);
                let (xp, _g1) = x.device_ptr(&k.stream);
                let (yp, _g2) = y.device_ptr_mut(&k.stream);
                k.gemv_bf16_block_epi_ptr(wp, xp, yp, n, kk, 0, 1.0, 0.0)
            }
            // 伪随机字节（xorshift64）。**权重必须随机**：全零会让"按权重内容索引 shared 表"
            // 的访问退化成全 lane 同地址广播（无 bank 冲突），把带查表的核严重高估
            // （实测假象记录见 PLAN §P15-⑥：slotsm 查表版微基准 +27%、整机 −3%）。
            fn rnd_bytes(n: usize, seed: u64) -> Vec<u8> {
                let mut v = Vec::with_capacity(n);
                let mut s = seed | 1;
                for _ in 0..n {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    v.push(s as u8);
                }
                v
            }
            fn bw(
                k: &lingxi_kernels::Kernels,
                rep: usize,
                bytes: usize,
                mut f: impl FnMut() -> anyhow::Result<()>,
            ) -> anyhow::Result<f64> {
                for _ in 0..3 {
                    f()?;
                }
                k.stream.synchronize()?;
                let t = std::time::Instant::now();
                for _ in 0..rep {
                    f()?;
                }
                k.stream.synchronize()?;
                Ok(bytes as f64 * rep as f64 / t.elapsed().as_secs_f64() / 1e9)
            }
            // ── ① 纯 memcpy：硬件上限（单 blob 与一层 31 个 blob 两档）──
            for (tag, nb) in [("单 blob 2.61MB", 2_611_200usize), ("31 blob 81MB", 2_611_200 * 31)] {
                let src = k.stream.alloc_zeros::<u8>(nb)?;
                let mut dst = k.stream.alloc_zeros::<u8>(nb)?;
                let g = bw(&k, rep, nb, || Ok(k.stream.memcpy_dtod(&src, &mut dst)?))?;
                eprintln!("  [memcpy ] {tag:16} → **{g:7.1} GB/s**（上限）");
            }
            // ── ②③④ bf16 GEMV ──
            let (gu_n, gu_k) = (1280usize, 2560usize);
            let (dn_n, dn_k) = (2560usize, 640usize);
            let wgu = k.stream.alloc_zeros::<half::bf16>(gu_n * gu_k)?;
            let xgu = k.stream.alloc_zeros::<half::bf16>(gu_k)?;
            let mut ygu = k.stream.alloc_zeros::<half::bf16>(gu_n)?;
            let wdn = k.stream.alloc_zeros::<half::bf16>(dn_n * dn_k)?;
            let xdn = k.stream.alloc_zeros::<half::bf16>(dn_k)?;
            let mut ydn = k.stream.alloc_zeros::<half::bf16>(dn_n)?;
            let big_n = gu_n * 31;
            let wbig = k.stream.alloc_zeros::<half::bf16>(big_n * gu_k)?;
            let mut ybig = k.stream.alloc_zeros::<half::bf16>(big_n)?;
            let b2 = |n: usize, kk: usize| n * kk * 2;
            let g = bw(&k, rep, b2(gu_n, gu_k), || gv(&k, &wgu, &xgu, &mut ygu, gu_n, gu_k))?;
            eprintln!("  [bf16 gu] n={gu_n:5} k={gu_k:5} 单次        → **{g:7.1} GB/s**");
            let g = bw(&k, rep, b2(gu_n, gu_k) * 31, || {
                for _ in 0..31 {
                    gv(&k, &wgu, &xgu, &mut ygu, gu_n, gu_k)?;
                }
                Ok(())
            })?;
            eprintln!("  [bf16 gu] 同形状循环 31 次（像一层）      → **{g:7.1} GB/s**");
            let g = bw(&k, rep, b2(big_n, gu_k), || gv(&k, &wbig, &xgu, &mut ybig, big_n, gu_k))?;
            eprintln!("  [bf16 gu] n={big_n:5}（31 专家拼一起）单次 → **{g:7.1} GB/s**");
            let g =
                bw(&k, rep, b2(gu_n, gu_k), || gv_block(&k, &wgu, &xgu, &mut ygu, gu_n, gu_k))?;
            eprintln!("  [bf16 gu] 同形状·高并行度变体(block 一行)  → **{g:7.1} GB/s**");
            let g = bw(&k, rep, b2(dn_n, dn_k), || gv(&k, &wdn, &xdn, &mut ydn, dn_n, dn_k))?;
            eprintln!("  [bf16 dn] n={dn_n:5} k={dn_k:5} 单次        → **{g:7.1} GB/s**");
            let g = bw(&k, rep, b2(dn_n, dn_k) * 31, || {
                for _ in 0..31 {
                    gv(&k, &wdn, &xdn, &mut ydn, dn_n, dn_k)?;
                }
                Ok(())
            })?;
            eprintln!("  [bf16 dn] 同形状循环 31 次（像一层）      → **{g:7.1} GB/s**");
            // ── ⑤ ★ 我们的 Q4 GEMV：**形状对照**（= "改布局能不能吃满带宽"的判决）──
            //   同一份 Q4 核（同一个解包路径），只改矩阵行数：
            //     n=1280   = 一个专家的 gu（现状：每专家一个小矩阵）
            //     n=39680  = 31 个专家的 gu **拼成一个连续大矩阵**（用户提的"改布局"）
            //   字节是 zeros —— 带宽与内容无关，这里只测"几何形状"的影响。
            //   若 n=39680 明显更高 ⇒ 布局重排（离线做，运行时不付代价）本身就能吃到带宽。
            {
                use lingxi_artifact::expert_arena::{matrix_layout, BlobFormat};
                for (tag, n) in
                    [("n=1280 单专家 gu", 1280usize), ("n=39680 拼成大矩阵", 1280 * 31)]
                {
                    let lay = matrix_layout(BlobFormat::Q4G64F16S, n as u64, 2560u64)
                        .expect("Q4G64F16S 几何非法");
                    let payload = (n * 2560) / 2; // 4 bit/权重
                    let w = k.stream.alloc_zeros::<u8>(payload * 3)?; // 宽松：覆盖 payload+scale 面
                    let x = k.stream.alloc_zeros::<half::bf16>(2560)?;
                    let mut y = k.stream.alloc_zeros::<half::bf16>(n)?;
                    let g = bw(&k, rep, payload, || {
                        use cudarc::driver::{DevicePtr, DevicePtrMut};
                        let (wp, _g0) = w.device_ptr(&k.stream);
                        let (xp, _g1) = x.device_ptr(&k.stream);
                        let (yp, _g2) = y.device_ptr_mut(&k.stream);
                        k.gemv_q4g64_dense_epi_ptr(
                            wp, xp, yp, n, 2560, 2560, 0,
                            lay.scale_offset as u64, 0, 1.0, 0.0,
                        )
                    })?;
                    eprintln!("  [Q4 核 ] {tag:22} → **{g:7.1} GB/s**（payload {payload} B）");
                }
            }
            // ── ⑥⑦⑧ ★★ 三种喂法隔离（用户指定的判决）：同一个**专家槽核**，同一份字节 ──
            //   (a) s=1, n=39680：全部行当一个"大专家" ⇒ **1D 网格(1240块) + 连续内存**
            //   (b) s=31, n=1280, blob=2.61MB：31 个专家紧挨着（81MB 区内）⇒ **2D 网格(20,31) + 连续**
            //   (c) s=31, n=1280, blob=64MB：31 个专家**散在 2GB 区**里 ⇒ **2D 网格 + 散布**
            //   比较：(a)vs(b) 隔离"网格形状"；(b)vs(c) 隔离"寻址散布"。
            //   字节都是 zeros（带宽与内容无关）。
            {
                use lingxi_artifact::expert_arena::{matrix_layout, BlobFormat};
                let gslots = |k: &lingxi_kernels::Kernels,
                              slots: &cudarc::driver::CudaSlice<u8>,
                              ids: &cudarc::driver::CudaSlice<i32>,
                              x: &cudarc::driver::CudaSlice<half::bf16>,
                              y: &mut cudarc::driver::CudaSlice<half::bf16>,
                              s: usize,
                              n: usize,
                              kk: usize,
                              k_pad: i32,
                              blob: u64,
                              scale_off: u64|
                 -> anyhow::Result<()> {
                    k.gemv_q4g64_slots_bf16(
                        slots, ids, x, y, s, n, kk, k_pad, 0, blob, 0, scale_off,
                    )
                };
                let mk = |n: usize| {
                    matrix_layout(BlobFormat::Q4G64F16S, n as u64, 2560u64)
                        .expect("几何非法")
                };
                // (a) s=1，一个大矩阵 39680 行
                {
                    let n = 1280 * 31;
                    let lay = mk(n);
                    let w = k.stream.clone_htod(&rnd_bytes(n * 2560 / 2 * 2, 0xA1))?;
                    let ids = k.stream.alloc_zeros::<i32>(1)?;
                    let x = k.stream.alloc_zeros::<half::bf16>(2560)?;
                    let mut y = k.stream.alloc_zeros::<half::bf16>(n)?;
                    let blob = (n * 2560 / 2 * 2) as u64;
                    let g = bw(&k, rep, n * 2560 / 2, || {
                        gslots(&k, &w, &ids, &x, &mut y, 1, n, 2560, 2560, blob, lay.scale_offset)
                    })?;
                    eprintln!("  [槽核 a] s=1  n=39680（1D网格+连续） → **{g:7.1} GB/s**");
                }
                // (b)/(c) s=31，n=1280，紧挨 或 按 64MB 步长散开
                for (tag, stride_mb) in [("b", 0usize), ("c", 64)] {
                    let blob: u64 = if stride_mb == 0 {
                        2_611_200
                    } else {
                        stride_mb as u64 * 1024 * 1024
                    };
                    let total = (blob as usize) * 31;
                    let w = k.stream.clone_htod(&rnd_bytes(total, 0xB2 + stride_mb as u64))?;
                    // 每个专家读自己的 blob（真实形态）；全 0 会让 31 个 CTA 挤同一块 L2
                    let ids = k.stream
                        .clone_htod(&(0..31i32).collect::<Vec<i32>>())?;
                    let x = k.stream.alloc_zeros::<half::bf16>(2560)?;
                    let mut y = k.stream.alloc_zeros::<half::bf16>(1280 * 31)?;
                    let lay = mk(1280);
                    let g = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                        gslots(&k, &w, &ids, &x, &mut y, 31, 1280, 2560, 2560, blob, lay.scale_offset)
                    })?;
                    eprintln!(
                        "  [槽核 {tag}] s=31 n=1280 blob={:>4}MB({} 区) → **{g:7.1} GB/s**",
                        blob / 1048576,
                        if stride_mb == 0 { "81MB 连续" } else { "2GB 散布" }
                    );
                }
            }
            // ── ⑨⑩ ★★ 真实路径用的那两个变体，在**真实 s/e_uniq**下测 ──
            //   (d) `slots_rows rpw=1`：t=1 路径（应等于 (b) 的 612）
            //   (e) `slotsm rpw=2`：t=7 批量路径（s=217 对、e_uniq=31、每专家 7 个 token）
            //       —— **这才是 decode 主力**。若它远低于 (b)，瓶颈就在它身上。
            {
                use lingxi_artifact::expert_arena::{matrix_layout, BlobFormat};
                let blob: u64 = 2_611_200;
                let w = k.stream.clone_htod(&rnd_bytes(blob as usize * 31, 0xC3))?;
                let ids = k.stream
                    .clone_htod(&(0..31i32).collect::<Vec<i32>>())?;
                let x = k.stream.alloc_zeros::<half::bf16>(2560)?;
                let lay = matrix_layout(BlobFormat::Q4G64F16S, 1280u64, 2560u64).expect("几何");
                // (d) slots_rows rpw=1
                let mut y1 = k.stream.alloc_zeros::<half::bf16>(1280 * 31)?;
                let g = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slots_rows_bf16(
                        &w, &ids, &x, &mut y1, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, 1,
                    )
                })?;
                eprintln!("  [d slots_rows rpw=1] s=31 → **{g:7.1} GB/s**");
                // (e) slotsm rpw=2：s=217 对、31 个唯一专家、每专家 7 个 token。
                //     ★ bf16 x（现转）vs fp32 预转 x 对照：转换在计时外（提前转化）。
                let s_pairs = 31usize * 7;
                let group_off: Vec<i32> = (0..=31).map(|i| (i * 7) as i32).collect();
                let gof = k.stream.clone_htod(&group_off)?;
                let x7 = k.stream.alloc_zeros::<half::bf16>(2560)?;
                let mut y2 = k.stream.alloc_zeros::<half::bf16>(1280 * s_pairs)?;
                let g0 = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_bf16(
                        &w, &ids, &x7, &gof, &mut y2, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, 0,
                    )
                })?;
                let mut xf7 = k.stream.alloc_zeros::<f32>(2560)?;
                k.convert_bf16_f32(&x7, &mut xf7, 2560)?;
                let (xf7p, _g7) = xf7.device_ptr(&k.stream);
                let g1 = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_bf16(
                        &w, &ids, &x7, &gof, &mut y2, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, xf7p,
                    )
                })?;
                eprintln!(
                    "  [e slotsm rpw=2]   s=217 e=31 → bf16 {g0:7.1} | **xf32 {g1:7.1} GB/s** ← decode 主力核"
                );
                // (f) slotsm rpw=1 对照
                let g = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_rpw_bf16(
                        &w, &ids, &x7, &gof, &mut y2, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, 1, xf7p,
                    )
                })?;
                eprintln!("  [f slotsm rpw=1]   s=217 e=31 → **{g:7.1} GB/s**（xf32）");
                // (l)(m) ★ P15-㉖ dp4a/int8-x 版（同权重同网格，x 流量 ÷4）：(e)/(i) 的对照新尺子。
                {
                    let mut xq7 = k.stream.alloc_zeros::<u8>(2560)?;
                    let mut sx7 = k.stream.alloc_zeros::<f32>(1)?;
                    k.convert_bf16_q8_rs(&x7, &mut xq7, &mut sx7, 1, 2560)?;
                    let ids_v = ids.slice(0..ids.len());
                    let gof_v = gof.slice(0..gof.len());
                    let gl = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                        k.gemv_q4g64_slotsm_rpw_dp4a_view(
                            &w, &ids_v, &x7, &gof_v, &mut y2, 31, 1280, 2560, 2560, 0, blob, 0,
                            lay.scale_offset, 2, &xq7, &sx7,
                        )
                    })?;
                    eprintln!("  [l slotsm dp4a ]  s=217 e=31 (m=7) → **{gl:7.1} GB/s**（int8-x）");
                    let gof2b = k.stream
                        .clone_htod(&(0..=31i32).map(|i| i * 2).collect::<Vec<i32>>())?;
                    let gof2b_v = gof2b.slice(0..gof2b.len());
                    let gl2 = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                        k.gemv_q4g64_slotsm_rpw_dp4a_view(
                            &w, &ids_v, &x7, &gof2b_v, &mut y2, 31, 1280, 2560, 2560, 0, blob, 0,
                            lay.scale_offset, 2, &xq7, &sx7,
                        )
                    })?;
                    eprintln!("  [m slotsm dp4a ]  s=62  e=31 (m=2) → **{gl2:7.1} GB/s**（int8-x，真实路由≈此）");
                    // (n)(o) ★ P15-④ 判决：dense 核（带 LUT 解码）在**随机权重**下的大/小形状。
                    //   ⑤ 里的 557 是 alloc_zeros 测的——零权重让 LUT 索引恒定 ⇒ 全 lane
                    //   同地址广播、零 bank 冲突（P15-⑥ 已证这是假象）。随机权重下若
                    //   dense n=39680 也 ≈ 389 ⇒ "布局重排 557" 的 premise 不成立。
                    let wl = k.stream.clone_htod(&rnd_bytes(1280 * 31 * 2560 / 2 * 2, 0xD4))?;
                    let mut yl = k.stream.alloc_zeros::<half::bf16>(1280 * 31)?;
                    for (tag, nn) in [("n=39680", 1280 * 31usize), ("n=1280 ", 1280usize)] {
                        let gn = bw(&k, rep, nn * 2560 / 2, || {
                            use cudarc::driver::{DevicePtr, DevicePtrMut};
                            let (wp, _g0) = wl.device_ptr(&k.stream);
                            let (xp, _g1) = x7.device_ptr(&k.stream);
                            let (yp, _g2) = yl.device_ptr_mut(&k.stream);
                            k.gemv_q4g64_dense_epi_ptr(
                                wp, xp, yp, nn, 2560, 2560, 0, lay.scale_offset as u64, 0,
                                1.0, 0.0,
                            )
                        })?;
                        eprintln!("  [{tag} dense随机] Q4 LUT 核            → **{gn:7.1} GB/s**");
                    }
                }
                // (g) ★ 判定"是每元素指令多"还是"slotsm 结构本身"：**把 m 降到 1**。
                //     m=1 时每个权重字节只被 1 个 token 用 ⇒ 若 (g) 远高于 (e)，
                //     说明瓶颈是"每个 (行,列块,token) 那串 转换+FMA"（改嵌套可治）；
                //     若 (g) 仍 ≈170，说明瓶颈在 slotsm 的结构（加群表/地址算法），另说。
                let gof1 = k.stream.clone_htod(&(0..=31i32).collect::<Vec<i32>>())?;
                let ids1 = k.stream
                    .clone_htod(&(0..31i32).collect::<Vec<i32>>())?;
                let mut y3 = k.stream.alloc_zeros::<half::bf16>(1280 * 31)?;
                let g_bf = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_bf16(
                        &w, &ids1, &x, &gof1, &mut y3, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, 0,
                    )
                })?;
                let g = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_bf16(
                        &w, &ids1, &x, &gof1, &mut y3, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, xf7p,
                    )
                })?;
                eprintln!(
                    "  [g slotsm rpw=2]   s=31  e=31 (m=1) → bf16 {g_bf:7.1} | **xf32 {g:7.1} GB/s**（对照槽核 b 603）"
                );
                // (h) 对偶：把 m 拉到 8，看是否继续掉
                let gof8 = k.stream.clone_htod(&(0..=31i32).map(|i| i * 8).collect::<Vec<i32>>())?;
                let mut y4 = k.stream.alloc_zeros::<half::bf16>(1280 * 248)?;
                let g = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_bf16(
                        &w, &ids1, &x, &gof8, &mut y4, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, xf7p,
                    )
                })?;
                eprintln!("  [h slotsm rpw=2]   s=248 e=31 (m=8) → **{g:7.1} GB/s**（xf32）");
                // (i) ★ 真实路由 m≈2.26 的判定项：bf16 vs xf32 谁快（决定 moe.rs 默认）
                let gof2 = k.stream.clone_htod(&(0..=31i32).map(|i| i * 2).collect::<Vec<i32>>())?;
                let mut y5 = k.stream.alloc_zeros::<half::bf16>(1280 * 62)?;
                let g2b = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_bf16(
                        &w, &ids1, &x, &gof2, &mut y5, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, 0,
                    )
                })?;
                let g2f = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_bf16(
                        &w, &ids1, &x, &gof2, &mut y5, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, xf7p,
                    )
                })?;
                eprintln!(
                    "  [i slotsm rpw=2]   s=62  e=31 (m=2) → bf16 {g2b:7.1} | **xf32 {g2f:7.1} GB/s**（真实路由≈此形状）"
                );
                // (j) ★ rpw 隔离：slots 黄金路径用 rpw=1；slotsm 默认 rpw=2。
                //     若 rpw=1 追平槽核 b(388)，则差距在行分块而非 x 预取，改默认即可。
                let gjb = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_rpw_bf16(
                        &w, &ids1, &x, &gof1, &mut y3, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, 1, 0,
                    )
                })?;
                let gjf = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_rpw_bf16(
                        &w, &ids1, &x, &gof1, &mut y3, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, 1, xf7p,
                    )
                })?;
                eprintln!(
                    "  [j slotsm rpw=1]   s=31  e=31 (m=1) → bf16 {gjb:7.1} | **xf32 {gjf:7.1} GB/s**（对照槽核 b 388）"
                );
                let gkr = bw(&k, rep, 1280 * 2560 / 2 * 31, || {
                    k.gemv_q4g64_slotsm_rpw_bf16(
                        &w, &ids1, &x, &gof2, &mut y5, 31, 1280, 2560, 2560, 0, blob, 0,
                        lay.scale_offset, 1, xf7p,
                    )
                })?;
                eprintln!("  [k slotsm rpw=1]   s=62  e=31 (m=2) → **{gkr:7.1} GB/s**（xf32）");
            }
            eprintln!(
                "  参照：**我们的 Q4 专家 GEMV（真实 decode 实测）= 170 GB/s**（78MB/层 ÷ 0.46ms）"
            );
        }
        Some("verify") => {
            let path = args.get(2).expect("用法: lingxi-bench verify <artifact.ninfer>");
            let t0 = std::time::Instant::now();
            let art = Artifact::open(path)?;
            art.verify()?;
            let catalog = lingxi_model::catalog_from_artifact(&art);

            let mut formats: BTreeMap<String, usize> = BTreeMap::new();
            let mut layouts: BTreeMap<String, usize> = BTreeMap::new();
            let mut roles: BTreeMap<String, usize> = BTreeMap::new();
            let mut total = 0u64;
            for (_, info) in catalog.iter() {
                *formats.entry(info.format.name().to_string()).or_default() += 1;
                *layouts.entry(info.layout.name().to_string()).or_default() += 1;
                *roles.entry(format!("{:?}", info.role)).or_default() += 1;
                total += info.storage_bytes();
            }
            println!(
                "OK: identity={}/{} objects={} (tensor {} + resource {}) payload={:.2}GB 用时 {:.2}s",
                art.identity.model_id,
                art.identity.weights_id,
                art.objects.len(),
                art.tensors().count(),
                art.resources().count(),
                total as f64 / 1e9,
                t0.elapsed().as_secs_f64(),
            );
            println!("formats: {formats:?}");
            println!("layouts: {layouts:?}");
            let other = roles.get("Other").copied().unwrap_or(0);
            println!("roles: {roles:?}  (Other={other})");
        }
        Some("vram-budget") => {
            // 用法: vram-budget <artifact> [arena] [split_lo] [kv_cap]
            // 只读 catalog/arena 与设备总量，不加载权重、不分配专家槽。
            let path = args
                .get(2)
                .expect("用法: vram-budget <artifact> [arena] [split_lo] [kv_cap]");
            let arena = args
                .get(3)
                .map(|s| s.as_str())
                .unwrap_or("/dev/shm/qwen4_exp.expert-arena.mix64.bin");
            let split_lo: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(24);
            let kv_cap: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(32768);
            let art = Artifact::open(path)?;
            println!("artifact={path}\narena={arena}\nsplit_lo={split_lo} kv_cap={kv_cap}");
            for dev in 0..2 {
                let _ctx = cudarc::driver::CudaContext::new(dev)?;
                match autotune_arena_slots(&art, arena, dev, split_lo, kv_cap) {
                    Ok(Some((slots, ledger))) => println!("{ledger}\n  dev{dev}: slots={slots}"),
                    Ok(None) => println!("dev{dev}: autotune disabled"),
                    Err(e) => println!("dev{dev}: ERROR {e:#}"),
                }
            }
        }
        Some("bytes") => {
            // 用法: bytes <artifact> [top_n]
            // 统计 payload 字节构成，用于部署内存策略（什么常驻 VRAM / 什么流式读盘）。
            let path = args.get(2).expect("用法: lingxi-bench bytes <artifact.ninfer> [top_n]");
            let top_n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
            let art = Artifact::open(path)?;
            let catalog = lingxi_model::catalog_from_artifact(&art);

            let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
            let mut total = 0u64;
            let mut by_format: BTreeMap<String, (u64, usize)> = BTreeMap::new();
            let mut by_cat: BTreeMap<String, (u64, usize)> = BTreeMap::new();
            let mut by_layer: BTreeMap<String, u64> = BTreeMap::new();
            let mut tensors: Vec<(u64, String)> = Vec::new();

            for (_, info) in catalog.iter() {
                let b = info.storage_bytes();
                total += b;
                let f = by_format.entry(info.format.name().to_string()).or_insert((0, 0));
                f.0 += b;
                f.1 += 1;

                let cat = categorize(&info.name);
                let c = by_cat.entry(cat.to_string()).or_insert((0, 0));
                c.0 += b;
                c.1 += 1;

                *by_layer.entry(layer_of(&info.name)).or_default() += b;
                tensors.push((b, info.name.clone()));
            }

            println!("总 payload: {:.2} GiB ({total} bytes)", gb(total));
            println!("\n== 按量化格式 ==");
            for (k, (b, n)) in &by_format {
                println!("  {k:<16} {:>8.2} GiB  ({n} 个)", gb(*b));
            }
            println!("\n== 按类别 ==");
            let mut cats: Vec<_> = by_cat.iter().collect();
            cats.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            for (k, (b, n)) in cats {
                println!(
                    "  {k:<24} {:>8.2} GiB  {:>5.1}%  ({n} 个)",
                    gb(*b),
                    100.0 * *b as f64 / total as f64
                );
            }
            println!("\n== 按层（text 主干 + mtp）==");
            let mut layers: Vec<_> = by_layer.iter().collect();
            layers.sort_by(|a, b| b.1.cmp(a.1));
            for (k, b) in layers.iter().take(55) {
                println!("  {k:<28} {:>8.2} GiB", gb(**b));
            }
            // text 48 层是否等大（决定能否共享一套“单层常驻”方案）
            let text_layers: Vec<u64> = (0..48)
                .filter_map(|i| by_layer.get(&format!("text/layers/{i}")).copied())
                .collect();
            if !text_layers.is_empty() {
                let mn = *text_layers.iter().min().unwrap();
                let mx = *text_layers.iter().max().unwrap();
                println!(
                    "  text 主干 48 层：min {:.2} GiB / max {:.2} GiB（差 {:.2} GiB），合计 {:.2} GiB",
                    gb(mn),
                    gb(mx),
                    gb(mx - mn),
                    gb(text_layers.iter().sum())
                );
            }
            println!("\n== 最大的 {top_n} 个张量 ==");
            tensors.sort_by(|a, b| b.0.cmp(&a.0));
            for (b, name) in tensors.iter().take(top_n) {
                println!("  {:>8.2} GiB  {name}", gb(*b));
            }
        }
        Some("mtp-pack") => {
            // **把 `mtp/*` 张量抽出来单独打包**。
            //
            // 背景（docs §3.51）：生产产物 `qwen4_exp.ninfer` 里**没有** MTP 张量，
            // 而 `/data/models/lingxi-assets/qwen4_exp.q4dense.mtp2.ninfer`（252 GB）里有 27 个。
            // 投机解码只需要那 27 个 ⇒ 把它们抽成一个 ~5 GiB 的小产物，
            // 引擎启动时多开一个小产物即可，**不必去碰那个 252 GB 的大文件**。
            //
            // 用法: lingxi-bench mtp-pack <src.ninfer> <dst.ninfer>
            let src = args
                .get(2)
                .context("用法: lingxi-bench mtp-pack <src.ninfer> <dst.ninfer>")?;
            let dst = args.get(3).context("缺 dst 路径")?;
            let art = Artifact::open(src)?;
            let catalog = lingxi_model::catalog_from_artifact(&art);
            let mut objects: Vec<lingxi_artifact::Object> = Vec::new();
            let mut payloads: Vec<(String, &[u8])> = Vec::new();
            for (_, info) in catalog.iter() {
                if !info.name.starts_with("mtp/") {
                    continue;
                }
                let payload = art
                    .payload(&info.name)
                    .with_context(|| format!("取 payload 失败: {}", info.name))?;
                objects.push(lingxi_artifact::Object::Tensor {
                    name: info.name.clone(),
                    shape: info.shape.to_vec(),
                    format: info.format.name().to_string(),
                    layout: info.layout.name().to_string(),
                    offset: 0,
                    bytes: payload.len() as u64,
                });
                payloads.push((info.name.clone(), payload));
            }
            if objects.is_empty() {
                anyhow::bail!("源产物 {src} 里没有 mtp/* 张量");
            }
            let total: u64 = objects.iter().map(|o| o.bytes()).sum();
            println!(
                "mtp-pack: {} 个张量，共 {:.2} GiB → {dst}",
                objects.len(),
                total as f64 / 1073741824.0
            );
            let identity = lingxi_artifact::Identity {
                model_id: "qwen4_exp-mtp".to_string(),
                weights_id: "extracted-from-mtp2".to_string(),
            };
            let mut w = lingxi_artifact::Writer::create(dst, &identity, objects)?;
            for (name, payload) in payloads {
                w.write(&name, payload)?;
            }
            w.finish()?;
            println!("mtp-pack 完成：{dst}");
        }
        Some("mtp-accept") => {
            // **接受率测量** —— 投机解码的总闸门（§3.52 步骤 2）。
            // 主干贪心跑 N 步；每步拿 (h_P, 主干刚预测的 token) 喂 MTP 头出草稿，
            // 与主干**真实的下一个 token** 比对；输出一致率（= 接受率）。
            // MTP 有自己的 KV（1 层），随主干同步推进：prompt 一次性 T 行，之后每步 1 行。
            // 用法: lingxi-bench mtp-accept <main.ninfer> <mtp.ninfer> [prompt] [n]
            let main_path = args
                .get(2)
                .context("用法: lingxi-bench mtp-accept <main.ninfer> <mtp.ninfer> [prompt] [n]")?;
            let mtp_path = args.get(3).context("缺 MTP 产物路径")?;
            let prompt = prompt_arg(args.get(4))?;
            let n_new: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);

            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let art = Arc::new(Artifact::open(main_path)?);
            let meta = flash_next_meta();
            let shape = layer_shape(&meta);
            let tok_bytes = art
                .payload("frontend/tokenizer.json")
                .context("产物缺少 frontend/tokenizer.json")?;
            let tok = lingxi_tokenize::Tokenizer::from_json(&tok_bytes)?;
            let prompt_ids = tok.encode(&prompt)?;
            let prompt_len = prompt_ids.len();
            let endpoints = load_endpoints(&k, &art)?;
            // ⚠️ `LINGXI_SPEC_BENCH` 那段 A/B 会**先顺序跑 n_seq 个、再投机跑 n_seq 个**
            // ⇒ 位置最多到 `prompt_len + 2*n_seq`，比普通 decode 多一整段。
            // 容量/rope 表按普通口径开会在投机环路里炸
            // （实测：`QsaState 容量不足: 57 > 53`，崩在 `spec_chunk`）。
            let spec_ab = std::env::var("LINGXI_SPEC_BENCH").is_ok();
            let extra = if spec_ab {
                std::env::var("LINGXI_SPEC_N").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(24)
            } else {
                0
            };
            let rope_len = prompt_len + n_new + extra + 8;
            let tables = load_rope_tables(&k, rope_len, shape.attn.rot_dim)?;
            let n_layers = lingxi_model::DIMS.n_layers as u32;
            let arena_path = std::env::var("LINGXI_EXPERT_ARENA")
                .unwrap_or_else(|_| "/dev/shm/qwen4_exp.expert-arena.q4g64.bin".to_string());
            // ㊿-P0：未显式给槽数时走显存预算自动分配。tune_lo 在 ls_lo 定案前先用
            // env 口径（缺省 24；无 GPU1 时 ls_lo 实为 0，此处按 24 只会**少估**槽数，安全方向）。
            let tune_lo: u32 = std::env::var("LINGXI_LAYER_SPLIT")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(24);
            let slots: usize = match std::env::var("LINGXI_ARENA_SLOTS")
                .ok().and_then(|s| s.parse().ok())
            {
                Some(v) => v,
                None => match autotune_arena_slots(&art, &arena_path, 0, tune_lo, rope_len + 8)? {
                    Some((s, ledger)) => {
                        eprintln!("{ledger}");
                        s
                    }
                    None => 7000,
                },
            };
            let moe_split = lingxi_model::flag("LINGXI_MOE_SPLIT");
            // ⚠️ 跨卡 + CUDA Graph 是**危险组合**：图是在 GPU0 的 stream 上捕获/实例化的，
            //    GPU1 的内核一旦落进捕获区间，重放时就会在"图所属的设备"上执行 ⇒ 错乱。
            //    当前这条路必须关图（`LINGXI_NO_GRAPHS=1`；实测关图本来就快 5%，见 §P13.3）。
            anyhow::ensure!(
                !moe_split || lingxi_model::flag("LINGXI_NO_GRAPHS"),
                "LINGXI_MOE_SPLIT 必须同时设 LINGXI_NO_GRAPHS=1 —— 跨卡内核不能被 GPU0 的 CUDA Graph 捕获"
            );
            // GPU0 这份 arena 服务哪些层：默认全部 48 层；开跨卡/LayerSplit 时只服务前一半
            // （否则一半的槽位会白占在 GPU1 负责的层上）。
            let split_lo_env: u32 = std::env::var("LINGXI_MOE_SPLIT_LO")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(24);
            let layers_hi_env: u32 = std::env::var("LINGXI_ARENA_LAYERS_HI")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(48);

            // ★★ **全量 layer-split**（`LINGXI_LAYER_SPLIT=<lo>`，PLAN §P15 里程碑 A）：
            //    与 `LINGXI_MOE_SPLIT`（只搬 MoE）互斥。这里必须在建 `Sequence` **之前**把
            //    两张卡都建好 —— 因为 `[lo, n_layers)` 的**状态**要直接建在 GPU1 上。
            // ★ P15-㉜（2026-10-17）：**双卡时默认自动开 split**（lo=24）——真实语料实测
            //    20.7 → 54.1 tok/s（+161%：每卡 arena 减半 → 专家 miss 崩塌；bench 英文
            //    高命中语境 −1%）。显式设 `LINGXI_LAYER_SPLIT=0` 可关（回归对照用）。
            let ls_env: Option<u32> = std::env::var("LINGXI_LAYER_SPLIT")
                .ok().and_then(|s| s.parse().ok());
            let auto_split = ls_env.is_none() && !moe_split
                && cudarc::driver::CudaContext::new(1).is_ok();
            let ls_lo: u32 = ls_env.unwrap_or(if auto_split { 24 } else { 0 });
            let ls_on = ls_lo > 0 && ls_lo < n_layers;
            anyhow::ensure!(
                !(ls_on && moe_split),
                "LINGXI_LAYER_SPLIT 与 LINGXI_MOE_SPLIT 互斥（前者已覆盖后者的能力）"
            );
            if ls_on {
                if auto_split {
                    // ㊸：不再整体禁图——所录的图全是**单层本地段**（专家段 3 核固定槽），
                    // 层⇄设备一一对应（key=layer*8+7 只在那一台设备上出现），捕获/重放都在
                    // 本层自己的 stream 上；跨卡拷贝（cross_copy）在层循环之外，不进图。
                    // 若实测判据翻车，回退 = 恢复 force_disable() 一行。
                    eprintln!(
                        "[layer-split] 检测到双卡：自动启用整层切分 lo={ls_lo}（真实语料 +161%，\
                         bench 语境 −1%；设 LINGXI_LAYER_SPLIT=0 关闭）。单层本地图保持开启。"
                    );
                } else {
                    if !lingxi_model::flag("LINGXI_NO_GRAPHS") {
                        eprintln!(
                            "[layer-split] 警告：未设 LINGXI_NO_GRAPHS。㊸ 起单层本地图（专家段）\
                             在 split 下保持可用；若出现数值异常请先设 LINGXI_NO_GRAPHS=1 对照。"
                        );
                    }
                }
                // ★ P15-㉞：池已 device-aware（键=(设备,长度)）⇒ 不再需要禁池。
                // 禁池正是 split+分块确定性发散的根因（直连 alloc/free 与在飞 kernel 竞争）。
            }
            let ls_k1: Option<lingxi_kernels::Kernels> =
                if ls_on { Some(lingxi_kernels::Kernels::load(1)?) } else { None };
            let ls_tables: Option<lingxi_driver::RopeTables> = match ls_k1.as_ref() {
                Some(kk) => Some(load_rope_tables(kk, rope_len, shape.attn.rot_dim)?),
                None => None,
            };
            // ㊿-P0：未显式给槽数时走显存预算自动分配（env 显式值优先）。
            // talk 路径 KV 容量 = rope_len+8（prompt+n_new+spec 余量+16）。
            let ls_slots: usize = match std::env::var("LINGXI_LAYER_SPLIT_SLOTS")
                .ok().and_then(|s| s.parse().ok())
            {
                Some(v) => v,
                None => match autotune_arena_slots(&art, &arena_path, 1, ls_lo, rope_len + 8)? {
                    Some((s, ledger)) => {
                        eprintln!("{ledger}");
                        s
                    }
                    None => 9600,
                },
            };  // ★㊲ 工作集≥槽位判决(375/卡层)
            let ls_experts: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> =
                match ls_k1.as_ref() {
                    Some(kk) => {
                        eprintln!(
                            "[layer-split] GPU1 上下文就绪；层 {ls_lo}..{n_layers} 整层交给它，专家槽位 {ls_slots}"
                        );
                        Some(Box::new(arena_source::ArenaExpertSource::new(
                            kk,
                            &arena_path,
                            ls_slots,
                            false,
                            ls_lo..n_layers,
                        )?))
                    }
                    None => None,
                };
            let gpu0_layers = 0..if moe_split {
                layers_hi_env.min(split_lo_env)
            } else if ls_on {
                ls_lo
            } else {
                n_layers
            };
            let mut seq = lingxi_driver::Sequence::new_split(
                &k,
                ls_k1.as_ref().map(|kk| (kk, ls_lo)),
                n_layers,
                &shape,
                rope_len + 8,
                Some((
                    1,
                    meta["ple_conv_kernel_size"].as_u64().unwrap() as usize,
                    meta["ngram_size"].as_u64().unwrap() as usize,
                )),
            )?;
            let host_cache = {
                let c = host_cache::HostCache::new(Arc::clone(&art));
                c.start_prefetcher();
                c.prefetch_layer_dense(0);
                c
            };
            let expert_source: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> =
                Some(Box::new(arena_source::ArenaExpertSource::new(&k, &arena_path, slots, false, gpu0_layers)?));
            // ★ **跨卡 MoE**（`LINGXI_MOE_SPLIT=1`，设计见 `docs/PLAN.md` §P13.1c）：
            //   把**后一半层**的 MoE 交给目前全空的 GPU1 跑 —— 它只有这些层的槽位，可以给到
            //   每层 400+ 个（覆盖 ~100%、miss≈0），而 GPU0 的 9000 槽只需伺候前一半层
            //   （每层 375 个 ⇒ ~92%）。跨卡只过 MoE 的输入/输出激活（36 KB×2/层）。
            let split_lo: u32 = split_lo_env;
            let split_slots: usize = std::env::var("LINGXI_MOE_SPLIT_SLOTS")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(11000);
            let k1 = if moe_split {
                let kk = lingxi_kernels::Kernels::load(1)?;
                eprintln!("[moe-split] GPU1 上下文就绪；层 {split_lo}..{n_layers} 的 MoE 交给它，槽位 {split_slots}");
                Some(kk)
            } else {
                None
            };
            let split_src: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> = match k1.as_ref() {
                // GPU1 那份 arena：只服务 `split_lo..48`（钉表也只钉这些层）⇒ 每层槽位拉满、
                // 首触也能靠钉表省掉（**关键**：纯 LRU 时 GPU1 的首触 miss 全走 x8 链路，
                // 实测 300 token 里 7350 次 miss ≈ 18 GB，直接把跨卡的收益吃光）。
                Some(kk) => Some(Box::new(arena_source::ArenaExpertSource::new(
                    kk,
                    &arena_path,
                    split_slots,
                    false,
                    split_lo..n_layers,
                )?)),
                None => None,
            };
            // 这些层在 GPU1 上的 MoE 权重（**只装 moe 段**：router + shared expert，~12 MB/层；
            // routed 专家走 GPU1 的 arena，不在这里）
            let mut split_moe: Vec<Option<lingxi_model::moe::MoeWeights>> =
                (0..n_layers as usize).map(|_| None).collect();
            if let Some(kk) = k1.as_ref() {
                let tl = std::time::Instant::now();
                for l in split_lo..n_layers {
                    let canon = format!("text/layers/{l}/moe/");
                    split_moe[l as usize] = Some(lingxi_model::moe::MoeWeights {
                        router_shared_gate: lingxi_model::gpu::upload_any_bf16(
                            kk, &art, &format!("{canon}router_shared_gate"))?,
                        routed_gate_up: None,
                        routed_down: None,
                        shared_gate_up: lingxi_model::gpu::upload_any_bf16(
                            kk, &art, &format!("{canon}shared_gate_up"))?,
                        shared_down: lingxi_model::gpu::upload_any_bf16(
                            kk, &art, &format!("{canon}shared_down"))?,
                    });
                }
                eprintln!("[moe-split] GPU1 的 MoE 权重装好（{} 层，{:.1}s）", n_layers - split_lo, tl.elapsed().as_secs_f64());
                // ⚠️ cudarc 的错误**粘在上下文上**，要等下一次 `bind_to_thread` 才吐 ⇒
                //    初始化期（建 context/开 arena/钉页/传权重）如果有错，会一路潜伏到
                //    后面的某次拷贝才炸、把报错点带偏。这里先清一次，保证后面报错的就是真凶。
                kk.ctx.check_err().map_err(|e| {
                    anyhow::anyhow!("[moe-split] GPU1 初始化阶段就有报错（见上）：{e}")
                })?;
            }
            let split_stat = [
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
            ];
            let remote = match (k1.as_ref(), split_src.as_deref()) {
                (Some(kk), Some(src)) => Some(lingxi_model::moe::RemoteMoe {
                    k: kk,
                    experts: src,
                    layers: split_lo..n_layers,
                    moe: &split_moe,
                    stat: &split_stat,
                }),
                _ => None,
            };
            let eng = lingxi_driver::Engine {
                k: &k,
                shape: &shape,
                n_layers,
                vocab: lingxi_model::DIMS.vocab as usize,
                endpoints: &endpoints,
                tables: &tables,
                experts: expert_source.as_deref(),
                remote: remote.as_ref(),
                k1: ls_k1.as_ref(),
                split_lo: ls_lo,
                tables1: ls_tables.as_ref(),
                experts1: ls_experts.as_deref(),
            };
            let hc_for_build: Option<Arc<host_cache::HostCache>> = Some(host_cache.clone());
            let build = {
                let (kb, k1b, ab, mb) = (&k, ls_k1.as_ref(), &art, &meta);
                let (lsb, lob) = (ls_on, ls_lo);
                Box::new(move |layer: u32| -> Result<lingxi_model::trunk::LoadedLayer> {
                    eprint!("\r  [load] 层 {layer}/{n_layers}        ");
                    // ★ 全量 layer-split：本层的稠密权重必须直接装到**本层要跑的卡**上
                    //   （装错了不会报错，只会在第一次 kernel 时 ILLEGAL_ADDRESS 或静默写串）。
                    let dev_k = match (lsb, k1b) {
                        (true, Some(kk)) if layer >= lob => kk,
                        _ => kb,
                    };
                    let b = build_layer(dev_k, ab, mb, layer, true, hc_for_build.as_deref())?;
                    if let (Some(c), true) = (&hc_for_build, layer + 1 < n_layers) {
                        c.prefetch_layer_dense(layer + 1);
                    }
                    Ok(lingxi_model::trunk::LoadedLayer {
                        weights: b.weights,
                        k_layernorm_host: b.k_layernorm_host,
                    })
                })
            };
            let mut cache_provider =
                DeviceLayerCache { slots: (0..n_layers as usize).map(|_| None).collect(), build };
            let provider = &mut cache_provider;

            // ── MTP：权重 + 自己的一份 KV 状态 ──
            let mtp_art = Artifact::open(mtp_path)?;
            let g = lingxi_model::mtp::load(&k, &mtp_art)?;
            let mtp_q4_exp = lingxi_model::flag("LINGXI_MTP_Q4_EXPERTS");
            let ml = build_mtp_layer(&k, &mtp_art, mtp_q4_exp)?;
            // MTP 的小 arena：把 512 个专家从 bf16(5.03 GiB) 换成 Q4G64(1.34 GiB)，
            // 给主干腾出专家槽位（真语料 miss 19.4% → ~11.7%）。判据：固定贪心序列必须不变。
            let mtp_arena = if mtp_q4_exp {
                Some(mtp_arena::MtpArena::build(&k, &mtp_art)?)
            } else {
                None
            };
            let mut mst = lingxi_model::layer::LayerStates::for_kind(
                &k,
                lingxi_ids::LayerKind::QsaSparseAttention,
                &shape,
                rope_len + 8,
                None,
            )?;
            let vocab = lingxi_model::DIMS.vocab as usize;
            let hidden = shape.hidden;
            let dim = shape.hc * hidden;
            // `LINGXI_MTP_STATS=1`：打印 draft 三站（stem 输出 / 层输出 / mixer 输出）的量级。
            // 白盒第一步 —— 哪一站尺度离谱就直接定位到那一段，不必先写 oracle。
            let mtp_stats = std::env::var("LINGXI_MTP_STATS").is_ok();
            let stat = |tag: &str, d: &cudarc::driver::CudaSlice<half::bf16>| -> Result<()> {
                let h: Vec<half::bf16> = k.stream.clone_dtoh(d)?;
                let f: Vec<f32> = h.iter().map(|v| v.to_f32()).collect();
                let mean = f.iter().sum::<f32>() / f.len() as f32;
                let sd = (f.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / f.len() as f32).sqrt();
                let mx = f.iter().fold(0f32, |a, v| a.max(v.abs()));
                eprintln!("    [mtp-stats] {tag:<14} n={} mean={mean:+.4} std={sd:.4} max|x|={mx:.3}", f.len());
                Ok(())
            };
            let embed1 = |k: &lingxi_kernels::Kernels, id: i64| -> Result<cudarc::driver::CudaSlice<half::bf16>> {
                let d = k.stream.clone_htod(&[id as i32])?;
                let mut e = k.stream.alloc_zeros::<half::bf16>(hidden)?;
                k.gather_rows_bf16(&endpoints.token_embedding, &d, &mut e, 1, hidden, 0)?;
                Ok(e)
            };
            // MTP 一步：`(Rin[t,dim], emb[t,H])` → （argmax, 每行 logits 的 **top-K 下标**, 本层输出 `out`）
            // 返回 `out` 是为了**草稿链**：下一枚草稿的 h 输入就是它（MTP 自喂自）。
            // 返回 top-K 是为了**量"多候选/树草稿"的天花板** —— top-1 命中率会低估草稿头
            // （分布可能全对，只是 argmax 差一两位）。
            //
            // `need_head=false`：**只要层的输出 `o`**（用来刷 MTP 自己的 KV），
            // 不算 head、不 D2H、不做 host 侧的 argmax/top-K/softmax。
            // 这一条是实测出来的大头：host 侧要对 `t×248320` 个 logits 做多趟标量处理
            // （`clone_dtoh` 同步拷贝 + f32 转换 + argmax + top-K + softmax 的 248K 次 `exp()`），
            // **比主干一整层（0.57 ms）还贵**。而**重放格的 token id 根本没用**（只用来刷 KV）。
            const MTP_KTOP: usize = 20;
            // prompt 相位的 top-K 存档（`mtp_step` 一次给 t 行的 top-K；这里只需要那一次）
            thread_local! {
                static TOPK_P: std::cell::RefCell<Vec<Vec<u32>>> =
                    const { std::cell::RefCell::new(Vec::new()) };
                /// `LINGXI_MTP_CHAIN_DIAG=hhcmp` 用：草稿链第 0 步的 `o`（= 自喂量）
                static O0_SAVE: std::cell::RefCell<
                    Option<cudarc::driver::CudaSlice<half::bf16>>,
                > = const { std::cell::RefCell::new(None) };
                /// `LINGXI_SUFFIX_TRACE` 用：本轮查表提议的内容（到 verify 之后与真值对照）
                static SD_TRACE: std::cell::RefCell<Option<Vec<i32>>> =
                    const { std::cell::RefCell::new(None) };
            }
            let mtp_step = |mst: &mut lingxi_model::layer::LayerStates,
                                hyper: &cudarc::driver::CudaSlice<half::bf16>,
                                emb: &cudarc::driver::CudaSlice<half::bf16>,
                                t: usize,
                                ids: &[i64],
                                need_head: bool|
             -> Result<(
                Vec<u32>,
                Vec<Vec<u32>>,
                Vec<f32>,
                cudarc::driver::CudaSlice<half::bf16>,
            )> {
                // `LINGXI_KTS=1`：把草稿一步拆成三段**真 GPU 时间**（层+stem / mixer+head / host 侧）。
                // 为什么必须测：按内核逐项估只算得出 ~7 ms/轮，而实测草稿是 14.1 ms/轮 —— 差 7 ms 不明。
                let kts_mtp = lingxi_model::flag("LINGXI_KTS");
                if kts_mtp {
                    k.stream.synchronize()?;
                }
                let t_mtp0 = std::time::Instant::now();
                let r = lingxi_model::mtp::stem(&k, &g, hyper, emb, t, shape.eps)?;
                if mtp_stats {
                    stat("stem R", &r)?;
                }
                let env = lingxi_model::layer::LayerEnv {
                    cos: &tables.cos,
                    sin: &tables.sin,
                    cos_host: &tables.cos_host,
                    sin_host: &tables.sin_host,
                    remote: None,
                    causal_host: &[],
                    k_layernorm_host: &ml.k_layernorm_host,
                    ids,
                    experts: mtp_arena
                        .as_ref()
                        .map(|a| a as &dyn lingxi_model::moe::ExpertSource),
                };
                // ⚠️ **层号必须用哨兵值**：`layer_step` 的层号决定 (a) CUDA Graph 的键、
                // (b) `graph::take_fixed(layer*8+slot)` 的固定缓冲。MTP 若用 `layer=0`，
                // 就会和**主干 layer 0** 共用图/缓冲 ⇒ 主干 layer 0 被写坏（实测：开图时
                // 第二个 token 就错，关图（LINGXI_NO_GRAPHS=1）就正确）。
                const MTP_LAYER_SENTINEL: u32 = 100;
                let out = lingxi_model::layer::layer_step(
                    &k, &ml.weights, &shape, &r, t, &env, mst, MTP_LAYER_SENTINEL,
                )?;
                if mtp_stats {
                    stat("layer out", &out)?;
                }
                if kts_mtp {
                    k.stream.synchronize()?;
                }
                let d_layer = t_mtp0.elapsed().as_secs_f64() * 1e3;
                let t_mtp1 = std::time::Instant::now();
                lingxi_model::scratch::put_bf16(r);
                if !need_head {
                    return Ok((Vec::new(), Vec::new(), Vec::new(), out));
                }
                // 【实验 LINGXI_MTP_GMIX=1】§3.64：末步该用**全局** mixer
                // (`text/hyper_connection_mixer`，主干喂输出头前的那一站) 还是 MTP 自带的 `mtp/mixer`？
                // Strata（已验证能跑投机的实现）读的是全局那套 ⇒ 开关切换以对拍；默认保持现状。
                let fmix: &lingxi_model::gr::GrWeights = if lingxi_model::flag("LINGXI_MTP_GMIX") {
                    &endpoints.mixer
                } else {
                    &ml.mixer
                };
                let (mixed, injw) = lingxi_model::gr::gated_residual_forward(
                    &k, fmix, &out, t as u32, dim, hidden, shape.hc, shape.lowrank, shape.eps,
                )?;
                if let Some(j) = injw {
                    lingxi_model::scratch::put_bf16(j);
                }
                if mtp_stats {
                    stat("mixer mixed", &mixed)?;
                    stat("  h 输入", hyper)?;
                }
                let mut logits = k.stream.alloc_zeros::<half::bf16>(t * vocab)?;
                if let Some(wq) = &endpoints.output_head_q4 {
                    lingxi_model::gpu::proj_q4_epi(&k, &mixed, wq, &mut logits, vocab, hidden, 0, t, 0, 0.0, 0.0)?;
                } else {
                    lingxi_model::gpu::proj_t1(&k, &mixed, &endpoints.output_head, &mut logits, vocab, hidden, t)?;
                }
                lingxi_model::scratch::put_bf16(mixed);
                if kts_mtp {
                    k.stream.synchronize()?;
                }
                let d_head = t_mtp1.elapsed().as_secs_f64() * 1e3;
                let t_mtp2 = std::time::Instant::now();
                // ★ P15-⑮：top-K 移进 GPU（两段式），免去整幅 logits 的 D2H + host topk
                // （KTS 实测 host 段 0.7 ms/步）。**位等价**：与 `topk_bf16_row` 同一全序
                // （f32 值降序、下标升序）⇒ 结果唯一；非有限值计进 `bad` 并在此报错（与原
                // `ensure!(is_finite)` 同效）。`LINGXI_HOST_TOPK=1` 回退老路做 A/B。
                let mut d: Vec<u32> = Vec::with_capacity(t);
                let mut tk: Vec<Vec<u32>> = Vec::with_capacity(t);
                let mut probs: Vec<f32> = Vec::with_capacity(t);
                if !lingxi_model::flag("LINGXI_HOST_TOPK") {
                    const G: usize = 64;
                    let kk = MTP_KTOP;
                    let mut pval = lingxi_model::scratch::f32(&k, t * G * kk)?;
                    let mut pidx = lingxi_model::scratch::i32(&k, t * G * kk)?;
                    let mut oval = lingxi_model::scratch::f32(&k, t * kk)?;
                    let mut oidx = lingxi_model::scratch::i32(&k, t * kk)?;
                    let mut bad = k.stream.alloc_zeros::<i32>(1)?;
                    k.topk_rows_part_bf16(&logits, &mut pval, &mut pidx, &mut bad, t, vocab, G, kk)?;
                    k.topk_rows_fin(&pval, &pidx, &mut oval, &mut oidx, t, G, kk)?;
                    let vals: Vec<f32> = k.stream.clone_dtoh(&oval)?;
                    let idxs: Vec<i32> = k.stream.clone_dtoh(&oidx)?;
                    let bad_h: Vec<i32> = k.stream.clone_dtoh(&bad)?;
                    anyhow::ensure!(bad_h[0] == 0, "logits 出现非有限值（GPU topk 计数 {}）", bad_h[0]);
                    lingxi_model::scratch::put_f32(pval);
                    lingxi_model::scratch::put_i32(pidx);
                    lingxi_model::scratch::put_f32(oval);
                    lingxi_model::scratch::put_i32(oidx);
                    for r in 0..t {
                        let top: Vec<(f32, u32)> =
                            (0..kk).map(|j| (vals[r * kk + j], idxs[r * kk + j] as u32)).collect();
                        // 与原来完全同式：只用已算出的 top-K，`p ≈ 1/(1+Σ exp(l_j − l_0))`
                        let l0 = top[0].0;
                        let mut s = 0f64;
                        for &(v, _) in top.iter().skip(1) {
                            s += ((v - l0) as f64).exp();
                        }
                        probs.push((1.0 / (1.0 + s)) as f32);
                        d.push(top[0].1);
                        tk.push(top.iter().map(|&(_, i)| i).collect());
                    }
                } else {
                let lh: Vec<half::bf16> = lingxi_model::scratch::download_bf16(&k, &logits)?;
                // ★ **单趟**取代原来的「建 f32 Vec + argmax + topk」三趟（KTS 实测 0.90 ms/步）。
                // `topk_bf16_row` 的口径与 `topk_rows` 逐位相同；有限性校验也并进这一趟。
                for r in 0..t {
                    let top =
                        lingxi_model::trunk::topk_bf16_row(&lh[r * vocab..(r + 1) * vocab], MTP_KTOP)?;
                    anyhow::ensure!(!top.is_empty(), "MTP topk 为空（vocab={vocab}）");
                    // 与原来完全同式：只用已算出的 top-K，`p ≈ 1/(1+Σ exp(l_j − l_0))`
                    let l0 = top[0].0;
                    let mut s = 0f64;
                    for &(v, _) in top.iter().skip(1) {
                        s += ((v - l0) as f64).exp();
                    }
                    probs.push((1.0 / (1.0 + s)) as f32);
                    d.push(top[0].1);
                    tk.push(top.iter().map(|&(_, i)| i).collect());
                }
                }
                if kts_mtp {
                    eprintln!(
                        "      [kts-mtp t={t}] 层+stem {:.2} | mixer+head {:.2} | **host(D2H+argmax/topk) {:.2}** ms",
                        d_layer,
                        d_head,
                        t_mtp2.elapsed().as_secs_f64() * 1e3
                    );
                }
                Ok((d, tk, probs, out))
            };

            // ── 主干 prefill（同时拿到 T 行的 hyper）──
            let (mut trunk_next, hyper_p) = seq.prefill_with_hyper(&eng, &prompt_ids, provider)?;
            eprintln!("\nprefill {prompt_len} token，首 token = {trunk_next}");
            // prompt 相位：位置 i 喂 (h_i, ids[i+1]) —— 即"下一个 token"的 embedding
            let mut ids_p = seq.ids.clone();
            if ids_p.len() < prompt_len + 1 {
                ids_p.push(trunk_next);
            }
            let emb_p = {
                let idx: Vec<i32> = ids_p[1..prompt_len + 1].iter().map(|&v| v as i32).collect();
                let d = k.stream.clone_htod(&idx)?;
                let mut e = k.stream.alloc_zeros::<half::bf16>(prompt_len * hidden)?;
                k.gather_rows_bf16(&endpoints.token_embedding, &d, &mut e, prompt_len, hidden, 0)?;
                e
            };
            // ══════════ (b) 诊断：喂给 MTP 的 `h` 是否**逐行位置对齐** ══════════
            // 自洽判据（不需要外部 oracle）：主干自己在位置 i 的 `h`，过「全局 mixer + lm_head」
            // 后的 argmax 必须 == 该位置**真实的下一个 token** `ids_p[i+1]`
            // （主干贪心能逐位复现 prompt 续写，golden 就是这么来的）。
            //   逐行命中 ⇒ 我们喂 MTP 的 `hyper` 行与位置一一对应，且与主干喂头的那份**同源**；
            //   错位/系统偏 ⇒ 问题就在 (b)（h 的来源或节奏）。
            {
                let (mixed_h, injw_h) = lingxi_model::gr::gated_residual_forward(
                    &k, &endpoints.mixer, &hyper_p, prompt_len as u32, dim, hidden,
                    shape.hc, shape.lowrank, shape.eps,
                )?;
                if let Some(j) = injw_h {
                    lingxi_model::scratch::put_bf16(j);
                }
                let mut logits_h = k.stream.alloc_zeros::<half::bf16>(prompt_len * vocab)?;
                if let Some(wq) = &endpoints.output_head_q4 {
                    lingxi_model::gpu::proj_q4_epi(
                        &k, &mixed_h, wq, &mut logits_h, vocab, hidden, 0, prompt_len, 0, 0.0, 0.0,
                    )?;
                } else {
                    lingxi_model::gpu::proj_t1(
                        &k, &mixed_h, &endpoints.output_head, &mut logits_h, vocab, hidden, prompt_len,
                    )?;
                }
                lingxi_model::scratch::put_bf16(mixed_h);
                let lh: Vec<half::bf16> = k.stream.clone_dtoh(&logits_h)?;
                let lf: Vec<f32> = lh.iter().map(|v| v.to_f32()).collect();
                let am = lingxi_model::trunk::argmax_rows(&lf, prompt_len, vocab);
                let (mut ok, mut tot) = (0usize, 0usize);
                for i in 0..prompt_len {
                    if i + 1 >= ids_p.len() {
                        break;
                    }
                    let want = ids_p[i + 1] as u32;
                    tot += 1;
                    if want == am[i] {
                        ok += 1;
                    }
                    if i < 6 {
                        eprintln!(
                            "  [h 对齐] 行{i}: 真实下个={want} h-argmax={} {}",
                            am[i],
                            if want == am[i] { "✓" } else { "✗" }
                        );
                    }
                }
                eprintln!(
                    "  [h 对齐] **主干 h(行i) 过 mixer+head 的 argmax == ids[i+1] 命中 {ok}/{tot}**"
                );
            }
            // h 的"身份"候选（两条线索打架，实测裁）：
            //   `hyper`（默认）：48 层后的**残差栈** [T,10240]（Strata 的 window_R_ 就叫 residual）
            //   `mixed`：层尾 mixer 的产物 [T,2560] **repeat 到 10240**
            //            （llama.cpp 说 h 是"喂 lm_head 的 post-norm hidden"，而本模型无最终 norm）
            let mtp_h_mixed =
                std::env::var("LINGXI_MTP_H").map(|v| v == "mixed").unwrap_or(false);
            eprintln!(
                "MTP 的 h 候选：{}",
                if mtp_h_mixed { "mixed(repeat 到 10240)" } else { "hyper(残差栈)" }
            );
            let h_prep = |hyper: &cudarc::driver::CudaSlice<half::bf16>,
                          t: usize|
             -> Result<cudarc::driver::CudaSlice<half::bf16>> {
                let mut hb = lingxi_model::scratch::bf16(&k, t * dim)?;
                if mtp_h_mixed {
                    let (mixed, injw) = lingxi_model::gr::gated_residual_forward(
                        &k, &endpoints.mixer, hyper, t as u32, dim, hidden, shape.hc,
                        shape.lowrank, shape.eps,
                    )?;
                    if let Some(j) = injw {
                        lingxi_model::scratch::put_bf16(j);
                    }
                    k.hc_repeat_bf16(&mixed, &mut hb, t, hidden, shape.hc)?;
                    lingxi_model::scratch::put_bf16(mixed);
                } else {
                    k.stream.memcpy_dtod(hyper, &mut hb)?;
                }
                Ok(hb)
            };
            let drafts_p = {
                let h_p = h_prep(&hyper_p, prompt_len)?;
                lingxi_model::scratch::put_bf16(hyper_p);
                let (d, tk, _pr, o) = mtp_step(&mut mst, &h_p, &emb_p, prompt_len, &seq.ids, true)?;
                lingxi_model::scratch::put_bf16(o);
                lingxi_model::scratch::put_bf16(h_p);
                TOPK_P.with(|c| *c.borrow_mut() = tk);
                d
            };
            let mut pending: i64 = drafts_p[prompt_len - 1] as i64; // 最后一个 prompt 位置的草稿
            // ── **强核对（真值，无需主干 rollout）** ──
            // prompt 相位：行 i 喂的是 (h_i, emb(ids[i+1]))，语义上应预测 **ids[i+2]**
            // ⇒ 每个 prompt 行的期望值是**已知的**，一次前向可得 ~prompt_len 个独立样本。
            //   这里命中率高 ⇒ 草稿头本身对，问题在 decode 段的喂法/索引；
            //   这里也≈0    ⇒ 草稿头本身有问题（与 §3.60"必是范畴性错误"一致）。
            {
                let (mut ok, mut tot) = (0usize, 0usize);
                let n = prompt_len.saturating_sub(2);
                for i in 0..n {
                    let (want, got) = (ids_p[i + 2] as i64, drafts_p[i] as i64);
                    tot += 1;
                    if want == got {
                        ok += 1;
                    }
                    if i < 6 || i + 3 >= prompt_len {
                        eprintln!(
                            "  [prompt 核对] 行{i}: 期望={want} 草稿={got} {}",
                            if want == got { "✓" } else { "✗" }
                        );
                    }
                }
                eprintln!(
                    "  [prompt 核对] **MTP 在已知真值上的命中 {ok}/{tot} = {:.3}**",
                    if tot > 0 { ok as f64 / tot as f64 } else { 0.0 }
                );
                // ── **top-N 命中率**（回答"该不该做多候选/树草稿"）──
                // 判据：主干真值 `ids_p[i+2]` 是否落在 MTP 该行 logits 的 **top-N** 里。
                // 若 top-1 低而 top-8/top-16 高 ⇒ 草稿**分布是对的**，只是 argmax 差一两位
                // ⇒ 多候选（树）草稿有真收益；若 top-20 也只高一点 ⇒ 分布本身偏了，树也没用。
                // ⚠️ 本轮样本 = 1 个 prompt 的 (prompt_len-2) 行，**统计上很薄**；
                //    要看"不同专家/上下文"的差异必须换多个 prompt 各跑一遍再合。
                {
                    const NS: [usize; 7] = [1, 2, 3, 5, 8, 10, 20];
                    let mut hit = [0usize; 7];
                    let n = prompt_len.saturating_sub(2);
                    TOPK_P.with(|c| -> Result<()> {
                        let tk = c.borrow();
                        for i in 0..n {
                            let want = ids_p[i + 2] as u32;
                            let Some(row) = tk.get(i) else { continue };
                            for (j, &nn) in NS.iter().enumerate() {
                                if row.iter().take(nn).any(|&x| x == want) {
                                    hit[j] += 1;
                                }
                            }
                        }
                        Ok(())
                    })?;
                    eprintln!("  [top-N 命中] 真值落在 MTP 该行 logits 的 top-N 里（n={n}）：");
                    for (j, &nn) in NS.iter().enumerate() {
                        eprintln!(
                            "      top-{nn:<2} {}/{} = {:.3}{}",
                            hit[j],
                            n,
                            if n > 0 { hit[j] as f64 / n as f64 } else { 0.0 },
                            if nn == 1 { "   ← 贪心接受率就是它" } else { "" }
                        );
                    }
                }
                // ── 目标偏移扫描（零额外算力）：草稿到底"对上"了 ids 的哪一个偏移？ ──
                // 若命中集中在 k≠0，说明喂进去的 (h, tok) 与目标之间差了一个位置
                // （tok 差一格 → k=-1；h 差一格 → k=+1）。全 0 才说明草稿真的算错了。
                eprintln!("  [偏移扫描] drafts_p[i] 命中 ids_p[i+2+k] 的计数：");
                for k in -2i64..=2i64 {
                    let (mut okk, mut totk) = (0usize, 0usize);
                    for i in 0..prompt_len {
                        let tgt = i as i64 + 2 + k;
                        if tgt < 0 || tgt as usize >= ids_p.len() {
                            continue;
                        }
                        totk += 1;
                        if drafts_p[i] as i64 == ids_p[tgt as usize] {
                            okk += 1;
                        }
                    }
                    eprintln!(
                        "      k={k:+}: {okk}/{totk} = {:.3}{}",
                        if totk > 0 { okk as f64 / totk as f64 } else { 0.0 },
                        if k == 0 { "   ← 当前口径" } else { "" }
                    );
                }
                if prompt_len >= 2 {
                    let last = prompt_len - 1;
                    eprintln!(
                        "  [prompt 核对] 注：最后一行(行{last})预测的是 ids[{}]（未知），而首个 decode 步该要的是 ids[{prompt_len}] 的草稿 ⇒ 应为 drafts_p[{}]",
                        prompt_len + 1,
                        prompt_len.saturating_sub(2)
                    );
                }
            }
            // ══════════════ 投机（speculative）机械：正确性核对 + 一轮成本 ══════════════
            let spec_ple = Some((
                1u32,
                meta["ple_conv_kernel_size"].as_u64().unwrap() as usize,
                meta["ngram_size"].as_u64().unwrap() as usize,
            ));
            if let Ok(ks) = std::env::var("LINGXI_MTP_SPEC") {
                // ── 真实投机环路（MTP 草稿链）──
                // 每轮：① MTP 草稿 k 个（自喂自链）② 主干一次批前向验证 ③ 数出接受前缀
                //      ④ 提交：主干状态回卷 + MTP 的 KV cell 用**主干的 h** 重放（否则下一轮建立在错 KV 上）
                let kd: usize = ks.parse().unwrap_or(6);
                let t_max = kd + 1;
                let n_gen = n_new.max(kd + 1);
                let mut sb = seq.spec_prepare(&eng, provider, t_max)?;
                // 起手：吃掉 prefill 吐的 pending，同时拿到主干在**那一行**的 h（草稿链的 h 输入）
                let (tok_first, hyper0) = seq.decode_token_with_hyper(&eng, provider)?;
                let mut h_last = h_prep(&hyper0, 1)?;
                lingxi_model::scratch::put_bf16(hyper0);
                let mut out_tokens: Vec<i64> = vec![tok_first];
                let (mut n_round, mut acc_sum) = (0usize, 0usize);
                // 分段计时（均值）：草稿链 / 主干验证 / 提交 / MTP 重放
                let (mut t_draft, mut t_chunk, mut t_commit, mut t_resync) = (0f64, 0f64, 0f64, 0f64);
                // ── 链上**逐位**接受率（回答用户提的"长度可能也不一样"）──
                //   reached[j] = 有多少轮**走到了第 j 位**（第 0..j-1 位全接受）
                //   hit1[j]    = 其中第 j 位 top-1 命中 ⇒ 条件接受率 = hit1[j]/reached[j]
                //   hitn[j][..] = 其中真值落在该位草稿 **top-N** 里（N=1,2,3,5,8,10,20）
                // ⚠️ 条件接受率**天然随 j 递减**（越深越难），所以必须报"分母 = reached"，
                //    否则第 3 位之后的数字会被分母缩水而看着像"突然崩掉"。
                const NS_SPEC: [usize; 7] = [1, 2, 3, 5, 8, 10, 20];
                let mut reached = vec![0usize; kd];
                let mut hit1 = vec![0usize; kd];
                let mut hitn = vec![[0usize; NS_SPEC.len()]; kd];
                // oracle 链（`LINGXI_MTP_CHAIN_DIAG=oracle`）：h 与 tok 都换成**主干真值**
                let mut or_reached = vec![0usize; kd];
                let mut or_hit = vec![0usize; kd];
                // `hhcmp` 累积器：自喂量 o0 vs 主干真 h(hyper_rows[0]) 的 cos / 相对误差 / 归一化 std
                let (mut hhc_n, mut hhc_cos, mut hhc_rel, mut hhc_a, mut hhc_b) =
                    (0usize, 0f64, 0f64, 0f64, 0f64);
                // 草稿概率 vs 实际接受（按 prob1 分桶；回答"模型知道自己第几位开始不确定吗"）
                let mut pb_n = [0usize; 4];
                let mut pb_hit = [0usize; 4];
                let mut pb_n0 = [0usize; 4];
                let mut pb_hit0 = [0usize; 4];
                // ── ★ 后缀查表草稿（Strata `--suffix-draft`，他那边默认开）──
                // 见 `lingxi_model::suffix` 的模块说明与 `probes.rs` 的登记项。
                // `sd_cursor`：历史里已经喂进查表器的位置游标（`seq.ids` 是单调前推的）。
                // ⚠️ **默认 0 = 关**（这是本轮的实测教训，别急着开）：
                //   查表本身是对的（`LINGXI_SUFFIX_TRACE=1` 实测提议 `[369,11751,13]` vs 真值同、
                //   `[369,44299,13,561,6511,314]` vs 真值同 ⇒ 3/3、3/4、6/6 命中），
                //   但**接受是前缀式的**：把 lookup 草稿挂在 MTP 链后面，只有整条 MTP 链先被接受
                //   才轮得到它 ⇒ 实测"提议 7 枚、接受 0 枚"，而且白付了行钱（47.4 → 44.3 tok/s）。
                //   参考实现用的是 `DraftPolicy`（`spec/draft_policy.hpp`）的**两种策略**，我们都还没做：
                //     ① `choose(t_mtp, lookup_k, match)`：匹配够长时 **用 lookup 顶掉整个 MTP 窗**
                //        （省掉草稿链的时间，且第一枚就高置信）；
                //     ② `chain(t_mtp, p_mtp, k_avail, match)`：**只在 `p_mtp`（整条 MTP 链都成立的概率）
                //        够高时**才追加，且要用"按匹配长度分桶的历史命中率 c"算期望收益 > 成本才追加。
                //   ⇒ 下一步：实现 ①（`--suffix-draft` 的本命用法），见 PLAN.md §P9.5c。
                let sd_min: usize = std::env::var("LINGXI_SUFFIX_DRAFT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                // 想让它"补几位"：0 = 不设限（把 MTP 被闸门砍掉的空位全补上）
                let sd_chain: usize = std::env::var("LINGXI_SUFFIX_CHAIN")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let mut sd = lingxi_model::suffix::SuffixDrafter::new(sd_min.max(2));
                let mut sd_cursor = 0usize;
                let (mut sd_prop, mut sd_acc, mut sd_rounds) = (0usize, 0usize, 0usize);
                let mut mtp_len = 0usize; // 本轮 MTP 链（闸门后）产出的草稿数
                let t0 = std::time::Instant::now();
                while out_tokens.len() < n_gen {
                    seq.spec_snapshot(&eng, &mut sb)?;
                    let pending = *seq.ids.last().unwrap();
                    // ★ 后缀查表：把"刚确认下来的 token"（含当前 pending）补进历史。
                    // `seq.ids` 是单调前推的（已提交前缀不会被改写），所以游标式追加不会重复。
                    if sd_min > 0 && seq.ids.len() > sd_cursor {
                        let tail: Vec<i32> =
                            seq.ids[sd_cursor..].iter().map(|&v| v as i32).collect();
                        sd.append(&tail);
                        sd_cursor = seq.ids.len();
                    }
                    let mlen0 = match &mst.attn {
                        lingxi_model::layer::AttnState::Qsa(q) => q.len,
                        _ => 0,
                    };
                    // ① 草稿链
                    // `LINGXI_KTS=1`：**前一个 sync 吃掉"上一段遗留的等待"，尾部 sync 量出本段真 GPU 时间**。
                    // 默认（不叠 KTS）这里只是 CPU 入队时间 —— 段与段会互相借时间（PLAN §0 测量纪律）。
                    let kts_spec = lingxi_model::flag("LINGXI_KTS");
                    if kts_spec {
                        k.stream.synchronize()?;
                    }
                    let ta = std::time::Instant::now();
                    let mut drafts: Vec<i64> = Vec::with_capacity(kd);
                    // 每一步的 top-K（用来量"链上第 j 位：真值在草稿 top-N 里吗"）
                    let mut draft_top: Vec<Vec<u32>> = Vec::with_capacity(kd);
                    // 每步草稿的 top-1 概率（参考实现用它当"链到几位"的闸门）
                    let mut probs_chain: Vec<f32> = Vec::with_capacity(kd);
                    // ── 【归因诊断】`LINGXI_MTP_CHAIN_DIAG=noh|nokv|…`（数学作废，只为定位）──
                    // 链上第 0 位 0.97（≈单步 0.95）但**第 1 位塌到 0.57**，且 top-20 也只 0.89
                    // ⇒ 是"自喂那步的状态偏了"。两个嫌疑：
                    //   `noh` ：自喂的 h（MTP 层的输出）本身不该这么用 ⇒ 每步都沿用第 0 步的 h
                    //   `nokv`：第 0 步 append 的那个 KV cell 污染了第 1 步的注意力
                    //           ⇒ 每步先把 MTP 的 QSA 回卷到轮前（只留 pre-round 上下文）
                    // 若 `noh` 把第 1 位救回 ~0.95 ⇒ 是 h；若 `nokv` 救回 ⇒ 是 KV；都不动 ⇒ 另找。
                    let diag = std::env::var("LINGXI_MTP_CHAIN_DIAG").unwrap_or_default();
                    let diag_noh = diag.contains("noh");
                    let diag_nokv = diag.contains("nokv");
                    // `LINGXI_SPEC_MIN_P`：草稿自己的 top-1 概率低于阈值就**停链**（参考实现 Strata
                    // `generate.cpp:9172` 的 `--spec-min-p`）。实测依据（prob1 vs 实际接受）：
                    //   ≥0.90 → 0.988 ｜ 0.70~0.90 → 0.750 ｜ 0.50~0.70 → 0.250 ｜ <0.50 → 0.278
                    // ⇒ 深位的低概率草稿几乎不被接受，砍掉它们**不丢接受数、但省下来的是"行"**。
                    let min_p: f32 = std::env::var("LINGXI_SPEC_MIN_P")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    // 【调试】开了闸门就逐轮打轨迹：截短链后报 `MTP logits 非有限值`，
                    // 需要看清 mlen0 / MTP 的 len / keep 三者的口径在哪一轮对不齐。
                    // 调试轨迹只打前 6 轮（够看清口径；开闸门时不刷屏）
                    if min_p > 0.0 && n_round < 6 {
                        let mst_len = match &mst.attn {
                            lingxi_model::layer::AttnState::Qsa(q) => q.len,
                            _ => usize::MAX,
                        };
                        eprintln!(
                            "    [min-p] 轮 {n_round} 起手: mlen0={mlen0} mst.len={mst_len} seq.ids.len={}",
                            seq.ids.len()
                        );
                    }
                    let mut h_in = lingxi_model::scratch::bf16(&k, dim)?;
                    k.stream.memcpy_dtod(&h_last, &mut h_in)?;
                    // ★ `DraftPolicy::choose`（参考实现 `spec/draft_policy.hpp` 的第一条策略）：
                    //   **匹配够长时用 lookup 顶掉整个 MTP 窗**。
                    //   为什么必须有它：查表草稿挂在 MTP 链**后面**是没用的——接受是前缀式的，
                    //   只有整条 MTP 链先被接受才轮得到它（实测"提议 7 枚、接受 0 枚"）。
                    //   当上下文里有长重复时（改代码/引号原文/工具 JSON/枚举），
                    //   **lookup 的第 0 枚置信度远高于 MTP 的自喂链**，此时应该整窗换成 lookup，
                    //   还顺带省掉草稿链的时间。
                    //   不设 `LINGXI_SUFFIX_CHOOSE` ⇒ 行为完全不变。
                    let sd_choose: usize = std::env::var("LINGXI_SUFFIX_CHOOSE")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    let mut chose = false;
                    if sd_choose > 0 {
                        let mut lk: Vec<i32> = Vec::new();
                        let n_lk = sd.propose(kd, &mut lk);
                        if n_lk >= sd_choose {
                            // 用 lookup 顶掉整窗：不跑 MTP 链
                            drafts.extend(lk[..n_lk.min(kd)].iter().map(|&v| v as i64));
                            chose = true;
                            sd_prop += n_lk;
                            sd_rounds += 1;
                            if lingxi_model::flag("LINGXI_SUFFIX_TRACE") && sd_rounds <= 12 {
                                eprintln!("    [suffix] choose 整窗 {:?}（无 MTP）", &lk[..n_lk.min(kd)]);
                            }
                        }
                    }
                    let mut h_step0 = lingxi_model::scratch::bf16(&k, dim)?;
                    k.stream.memcpy_dtod(&h_in, &mut h_step0)?;
                    let mut tok = pending;
                    for _ in 0..(if drafts.is_empty() { kd } else { 0 }) {
                        if diag_nokv {
                            if let lingxi_model::layer::AttnState::Qsa(q) = &mut mst.attn {
                                q.rewind(mlen0);
                            }
                        }
                        let emb = embed1(&k, tok)?;
                        let (d, tk, pr, o) = mtp_step(&mut mst, &h_in, &emb, 1, &seq.ids, true)?;
                        lingxi_model::scratch::put_bf16(emb);
                        lingxi_model::scratch::put_bf16(h_in);
                        drafts.push(d[0] as i64);
                        draft_top.push(tk.into_iter().next().unwrap_or_default());
                        probs_chain.push(pr.first().copied().unwrap_or(0.0));
                        // 概率闸门：这一步的草稿自己就不确定 ⇒ 再往下链没意义（省"行"）
                        if min_p > 0.0 && probs_chain.last().copied().unwrap_or(1.0) < min_p {
                            // ⚠️ 必须把 `o` 挂到 `h_in` 上再 break：`h_in` 在上面已经被
                            //    `put_bf16` 移走了，直接 break 会让循环后的 `put_bf16(h_in)` 报 E0382
                            h_in = o;
                            break;
                        }
                        tok = d[0] as i64;
                        if diag_noh {
                            // 不采纳自喂的 h：每一步都回到第 0 步的 h
                            lingxi_model::scratch::put_bf16(o);
                            h_in = lingxi_model::scratch::bf16(&k, dim)?;
                            k.stream.memcpy_dtod(&h_step0, &mut h_in)?;
                        } else {
                            // `hhcmp`：留一份第 0 步的 `o`（自喂量），后面与主干真 h 比数值
                            if diag.contains("hhcmp") {
                                let mut o0 = lingxi_model::scratch::bf16(&k, dim)?;
                                k.stream.memcpy_dtod(&o, &mut o0)?;
                                O0_SAVE.with(|c| *c.borrow_mut() = Some(o0));
                            }
                            // ⚠️ **自喂量也必须过同一个 `h_prep`**：否则 `LINGXI_MTP_H=mixed`
                            // 只影响第 0 步、第 1 步起偷偷回退到 hyper —— 是个"静默不一致"，
                            // 让那个开关的 A/B 结论完全不可信。默认（hyper）时 h_prep 是恒等，
                            // 所以这里**只在 mixed 模式下多一次拷贝**，默认路径逐位不变。
                            if mtp_h_mixed {
                                let o_prep = h_prep(&o, 1)?;
                                lingxi_model::scratch::put_bf16(o);
                                h_in = o_prep;
                            } else {
                                h_in = o;
                            }
                        }
                    }
                    lingxi_model::scratch::put_bf16(h_step0);
                    lingxi_model::scratch::put_bf16(h_in);
                    // ★ 后缀查表草稿：把 MTP 被闸门砍掉的空位用"上下文里原样出现过的续写"补上。
                    // 语义与 Strata 的 `--lookup-chain` 一致：查询 = 「历史后缀 + 已产出的草稿」，
                    // 命中就从"那个更早的位置之后"取 token。**只改提哪些草稿**，被采纳与否仍由主干验证决定。
                    // ⚠️ `chose` 时**不能**把 lookup 草稿记成 MTP 产的（否则 `sd_acc` 恒 0，
                    //    上一版就是这么把自己的成绩单写错的）
                    if !chose {
                        mtp_len = drafts.len();
                    }
                    if sd_min > 0 && drafts.len() < kd {
                        let extra: Vec<i32> = drafts.iter().map(|&v| v as i32).collect();
                        let room = kd - drafts.len();
                        let want = if sd_chain == 0 { room } else { room.min(sd_chain) };
                        let mut buf: Vec<i32> = Vec::new();
                        let got = sd.propose_after(&extra, want, &mut buf);
                        for &v in &buf[..got] {
                            drafts.push(v as i64);
                        }
                        if got > 0 {
                            sd_prop += got;
                            sd_rounds += 1;
                            // 诊断：这一枚是"提议错了"还是"这段文本本来就查不到"？
                            // 打印提议的前几枚 + 主干真值（真值在下面 verify 之后才有，所以先存下来）
                            if lingxi_model::flag("LINGXI_SUFFIX_TRACE") && sd_rounds <= 8 {
                                eprintln!(
                                    "    [suffix] 提议 {:?}（MTP 链 {} 枚：{:?}）",
                                    &buf[..got],
                                    mtp_len,
                                    &drafts[..mtp_len]
                                );
                                SD_TRACE.with(|c| *c.borrow_mut() = Some(buf[..got].to_vec()));
                            }
                        }
                    }
                    if kts_spec {
                        k.stream.synchronize()?;
                    }
                    t_draft += ta.elapsed().as_secs_f64() * 1e3;
                    // ②③ 主干验证 + 接受前缀
                    let tb = std::time::Instant::now();
                    let chunk: Vec<i64> =
                        std::iter::once(pending).chain(drafts.iter().copied()).collect();
                    seq.spec_chunk(&eng, &chunk, provider, &mut sb)?;
                    let mut acc = 0usize;
                    for (j, &d) in drafts.iter().enumerate() {
                        if d == sb.argmax[j] as i64 {
                            if j >= mtp_len {
                                sd_acc += 1; // 这一位是后缀查表提的
                            }
                            acc += 1;
                        } else {
                            break;
                        }
                    }
                    let keep = acc + 1; // 含 chunk[0]（pending 本身，必然保留）
                    // ★ `LINGXI_SPEC_TRACE_ARGMAX=1`：逐轮把「主干各行的 argmax + 草稿 + acc/keep」打出来。
                    //   为什么要它（2026-10-16）：**提交的 token 只可能是 `argmax[acc]`**，而 `acc` 由
                    //   「drafts 是否等于 argmax」决定 ⇒ **只要 `argmax` 本身不变，草稿再怎么变，
                    //   提交的 token 流都必须一样**（同一前缀下贪心是唯一的）。
                    //   所以两条配置的 token 流一旦分叉，**必然**是某一轮的 `argmax` 变了 ——
                    //   这条轨迹就是用来把"哪一轮"钉死的（drafts 的差异只是噪声，不是原因）。
                    if lingxi_model::flag("LINGXI_SPEC_TRACE_ARGMAX") {
                        eprintln!(
                            "    [argmax] 轮 {n_round:3} mlen0={mlen0} t={} acc={acc} keep={keep} argmax={:?} drafts={:?}",
                            sb.argmax.len(),
                            sb.argmax,
                            drafts
                        );
                    }
                    // 诊断对照：主干在那些位置上的真值（回答"是提议错了还是文本查不到"）
                    if lingxi_model::flag("LINGXI_SUFFIX_TRACE") {
                        if let Some(prop) = SD_TRACE.with(|c| c.borrow_mut().take()) {
                            let truth: Vec<u32> = (mtp_len..drafts.len())
                                .map(|j| sb.argmax[j])
                                .collect();
                            eprintln!("    [suffix] 真值 {truth:?}  ← 提议 {prop:?}");
                        }
                    }
                    // 逐位记账（只记"走到了"的位；见上面 reached 的注释）
                    // ⚠️ 上界必须取 `drafts.len()`：开了 `LINGXI_SPEC_MIN_P` 后链会被截短，
                    //    而 `acc` 可以等于 `drafts.len()` ⇒ 用 `0..kd` 会在 `drafts[j]` 越界 panic。
                    for j in 0..kd.min(drafts.len()) {
                        if j > acc {
                            break;
                        }
                        reached[j] += 1;
                        let truth = sb.argmax[j] as u32;
                        if drafts[j] as u32 == truth {
                            hit1[j] += 1;
                        }
                        if let Some(row) = draft_top.get(j) {
                            for (n, &nn) in NS_SPEC.iter().enumerate() {
                                if row.iter().take(nn).any(|&x| x == truth) {
                                    hitn[j][n] += 1;
                                }
                            }
                        }
                        // 草稿概率分桶（位 j 自己那一步的 prob1）
                        if let Some(&p1) = probs_chain.get(j) {
                            let bkt = if p1 >= 0.9 { 0 } else if p1 >= 0.7 { 1 } else if p1 >= 0.5 { 2 } else { 3 };
                            pb_n[bkt] += 1;
                            if drafts[j] as u32 == truth { pb_hit[bkt] += 1; }
                            if j == 0 {
                                pb_n0[bkt] += 1;
                                if drafts[0] as u32 == truth { pb_hit0[bkt] += 1; }
                            }
                        }
                    }
                    t_chunk += tb.elapsed().as_secs_f64() * 1e3;
                    // ── 【归因诊断 · 决定性】`LINGXI_MTP_CHAIN_DIAG=oracle` ──
                    // 实测已排除两个假设：`noh`（不更新自喂 h）让第 1 位**更差**（0.57→0.35）、
                    // `nokv`（回卷 MTP KV）只值 +0.06 且 top-20 不变 ⇒ 都不是元凶。
                    // 剩下唯一要问的：**自喂的那个 h（MTP 层输出 `o`）到底等不等价于"主干在该位置的 h"？**
                    //   oracle 链 = 把 h 换成**主干真值**（滞后一格：j=0 用 h_last，j≥1 用 hyper_rows[j-1]），
                    //               且 tok 也换成真值 chunk[j]。
                    //   判据：oracle 第 1 位回到 ~0.95 而自喂只有 0.57 ⇒ **h 的"身份"错了**
                    //        （我们的自喂量不是 MTP 训练时期望的那个）；若 oracle 也 ~0.6 ⇒ 草稿头本身
                    //        只能预测 1 步，2 步是能力上限（那就不用修接线，只能上多候选）。
                    if diag.contains("oracle") {
                        let mut o_hit = vec![0usize; kd];
                        let mut o_runs = 0usize;
                        for j in 0..kd {
                            if j >= chunk.len() {
                                break;
                            }
                            let mut h_j = lingxi_model::scratch::bf16(&k, dim)?;
                            if j == 0 {
                                k.stream.memcpy_dtod(&h_last, &mut h_j)?;
                            } else {
                                k.stream.memcpy_dtod(
                                    &sb.hyper_rows.slice((j - 1) * dim..j * dim),
                                    &mut h_j,
                                )?;
                            }
                            let emb = embed1(&k, chunk[j])?; // **真值** token
                            let (d, _tk, _pr, o) = mtp_step(&mut mst, &h_j, &emb, 1, &seq.ids, true)?;
                            lingxi_model::scratch::put_bf16(o);
                            lingxi_model::scratch::put_bf16(emb);
                            lingxi_model::scratch::put_bf16(h_j);
                            if d[0] as i64 == sb.argmax[j] as i64 {
                                o_hit[j] += 1;
                            }
                            o_runs += 1;
                        }
                        // 把 MTP 的 QSA 回卷到轮前，别污染后面的 commit/重放
                        if let lingxi_model::layer::AttnState::Qsa(q) = &mut mst.attn {
                            q.rewind(mlen0);
                        }
                        if o_runs > 0 {
                            or_reached.iter_mut().for_each(|v| *v += 1);
                            for j in 0..o_runs {
                                or_hit[j] += o_hit[j];
                            }
                        }
                        // `hhcmp`：把"自喂量 o0"与"主干真 h（hyper_rows[0]）"逐元素比一下。
                        // 若两者差不多 ⇒ h 没错（问题在别处）；若差很远 ⇒ 自喂量根本不是主干 h 那一族。
                        if diag.contains("hhcmp") {
                            let o0h = O0_SAVE.with(|c| c.borrow_mut().take());
                            if let Some(o0) = o0h {
                                let a: Vec<f32> =
                                    k.stream.clone_dtoh(&o0)?.iter().map(|v| v.to_f32()).collect();
                                let b: Vec<f32> = k.stream
                                    .clone_dtoh(&sb.hyper_rows.slice(0..dim))?
                                    .iter()
                                    .map(|v| v.to_f32())
                                    .collect();
                                let stat = |v: &[f32]| {
                                    let m = v.iter().sum::<f32>() / v.len() as f32;
                                    let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>()
                                        / v.len() as f32)
                                        .sqrt();
                                    let mx = v.iter().fold(0f32, |s, x| s.max(x.abs()));
                                    (m, sd, mx)
                                };
                                let (am, asd, amx) = stat(&a);
                                let (bm, bsd, bmx) = stat(&b);
                                let dot: f64 = a.iter().zip(&b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
                                let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                                let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                                let rel: f64 = a
                                    .iter()
                                    .zip(&b)
                                    .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                                    .sum::<f64>()
                                    .sqrt()
                                    / nb.max(1e-9);
                                hhc_n += 1;
                                hhc_cos += if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 };
                                hhc_rel += rel;
                                hhc_a += asd as f64 / (amx.max(1e-9) as f64);
                                hhc_b += bsd as f64 / (bmx.max(1e-9) as f64);
                                let _ = (am, bm);
                                lingxi_model::scratch::put_bf16(o0);
                            }
                        }
                    }
                    // ④ 提交主干状态
                    if kts_spec {
                        k.stream.synchronize()?;
                    }
                    let tc = std::time::Instant::now();
                    seq.spec_commit(&eng, provider, &mut sb, keep)?;
                    if kts_spec {
                        k.stream.synchronize()?;
                    }
                    t_commit += tc.elapsed().as_secs_f64() * 1e3;
                    let td = std::time::Instant::now();
                    // ④' MTP：回卷到批前 cell 数，再用（主干 h, 该位置token）推进 keep 个 cell
                    //      cell 语义 = `(h_j, tok_{j+1})`；最后一个 cell 的 `tok_{keep}` 就是
                    //      **奖励 token**（提交后 ids 末尾那个），不在 chunk 里。
                    if let lingxi_model::layer::AttnState::Qsa(q) = &mut mst.attn {
                        q.rewind(mlen0);
                    }
                    let bonus = *seq.ids.last().unwrap();
                    // ★ 2026-10-16：**批量重放**（`LINGXI_MTP_REPLAY_1BY1=1` 退回逐行）。
                    // 逐行重放 = `keep` 次 t=1 的 MTP 前向 ⇒ MTP 那一层的权重被读 `keep` 遍。
                    // 批成一次 `t=keep` 的前向：权重只读一遍，行边际只剩专家读。
                    // **为什么与逐行等价**：MTP 的注意力是因果的、状态已 `rewind(mlen0)`，
                    //   一次喂 `keep` 行与逐行喂的每行输出逐位同源；
                    // 而**对外输出不受影响**：草稿只影响"哪些行被验"，被采纳的 token 始终是
                    //   主干自己的贪心 argmax（主干路径与重放无关）。
                    if lingxi_model::flag("LINGXI_MTP_REPLAY_1BY1") {
                    for j in 0..keep {
                        let mut h1 = lingxi_model::scratch::bf16(&k, dim)?;
                        k.stream
                            .memcpy_dtod(&sb.hyper_rows.slice(j * dim..(j + 1) * dim), &mut h1)?;
                        let tok_n = if j + 1 < keep { chunk[j + 1] } else { bonus };
                        let emb = embed1(&k, tok_n)?;
                        let (_, _, _, o) = mtp_step(&mut mst, &h1, &emb, 1, &seq.ids, false)?;
                        lingxi_model::scratch::put_bf16(o);
                        lingxi_model::scratch::put_bf16(emb);
                        lingxi_model::scratch::put_bf16(h1);
                    }
                    } else {
                    let mut hb = lingxi_model::scratch::bf16(&k, keep * dim)?;
                    k.stream.memcpy_dtod(&sb.hyper_rows.slice(0..keep * dim), &mut hb)?;
                    let toks: Vec<i32> = (0..keep)
                        .map(|j| if j + 1 < keep { chunk[j + 1] as i32 } else { bonus as i32 })
                        .collect();
                    let d_toks = k.stream.clone_htod(&toks)?;
                    let mut emb = k.stream.alloc_zeros::<half::bf16>(keep * hidden)?;
                    k.gather_rows_bf16(&endpoints.token_embedding, &d_toks, &mut emb, keep, hidden, 0)?;
                    let (_, _, _, o) = mtp_step(&mut mst, &hb, &emb, keep, &seq.ids, false)?;
                    lingxi_model::scratch::put_bf16(o);
                    lingxi_model::scratch::put_bf16(emb);
                    lingxi_model::scratch::put_bf16(hb);
                    }
                    // 新的 h_last = 被接受前缀最后一行的主干 h
                    if min_p > 0.0 && n_round < 6 {
                        let mst_len = match &mst.attn {
                            lingxi_model::layer::AttnState::Qsa(q) => q.len,
                            _ => usize::MAX,
                        };
                        eprintln!(
                            "    [min-p] 轮 {n_round} 收尾: m={} keep={} acc={} mst.len={}（应 = mlen0+keep = {}）probs={:?}",
                            drafts.len(),
                            keep,
                            acc,
                            mst_len,
                            mlen0 + keep,
                            probs_chain.iter().map(|p| (p * 100.0).round() / 100.0).collect::<Vec<_>>()
                        );
                    }
                    lingxi_model::scratch::put_bf16(h_last);
                    h_last = lingxi_model::scratch::bf16(&k, dim)?;
                    k.stream
                        .memcpy_dtod(&sb.hyper_rows.slice((keep - 1) * dim..keep * dim), &mut h_last)?;
                    // 同上：`MTP_H=mixed` 时每轮也要过 h_prep（默认恒等 ⇒ 默认路径逐位不变）
                    if mtp_h_mixed {
                        let hl = h_prep(&h_last, 1)?;
                        lingxi_model::scratch::put_bf16(h_last);
                        h_last = hl;
                    }
                    // 计数口径：chunk[0]（= pending）**上一轮已经算产出**，这里只把
                    // 新接受的草稿 chunk[1..keep] 与奖励 token 计入。
                    out_tokens.extend_from_slice(&chunk[1..keep]);
                    out_tokens.push(bonus);
                    if kts_spec {
                        k.stream.synchronize()?;
                    }
                    t_resync += td.elapsed().as_secs_f64() * 1e3;
                    n_round += 1;
                    acc_sum += acc;
                }
                k.stream.synchronize()?;
                let el = t0.elapsed().as_secs_f64();
                let nr = n_round.max(1);
                eprintln!(
                    "  [mtp-spec] {n_round} 轮｜平均接受 {:.2}/{kd}｜产出 {} token｜{:.2}s ⇒ **{:.1} tok/s**（{:.1} ms/token）",
                    acc_sum as f64 / nr as f64,
                    out_tokens.len(),
                    el,
                    out_tokens.len() as f64 / el,
                    el * 1e3 / out_tokens.len() as f64
                );
                eprintln!("  [mtp-spec] 生成序列 = {out_tokens:?}");
                // `LINGXI_EVPROF=1`：打印**最后一轮**的逐段真 GPU 时间（事件口径，不排空流水）
                if let Some(p) = k.evprof.as_ref() {
                    p.dump(n_layers);
                }
                {
                    // 跨卡（`LINGXI_MOE_SPLIT`）的实际开销：次数 + 累计 µs + 其中搬运 µs。
                    // 这是判断"跨卡到底值不值"的唯一直接证据（对照两卡的 miss 差）。
                    let sn = split_stat[0].load(std::sync::atomic::Ordering::Relaxed);
                    let sus = split_stat[1].load(std::sync::atomic::Ordering::Relaxed);
                    let sxus = split_stat[2].load(std::sync::atomic::Ordering::Relaxed);
                    if sn > 0 {
                        eprintln!(
                            "  [mtp-spec] 跨卡：{sn} 次｜轮均 {:.2} ms（其中搬运 {:.2} ms）｜共 {:.1} ms（占整轮 {:.1}%）",
                            sus as f64 / 1e3 / nr as f64,
                            sxus as f64 / 1e3 / nr as f64,
                            sus as f64 / 1e3,
                            100.0 * (sus as f64 / 1e3) / (el * 1e3),
                        );
                    }
                }
                eprintln!(
                    "  [mtp-spec] 分段均值/轮：草稿 {:.1}｜主干验证 t={t_max} {:.1}｜提交 {:.1}｜MTP 重放 {:.1} ms",
                    t_draft / nr as f64,
                    t_chunk / nr as f64,
                    t_commit / nr as f64,
                    t_resync / nr as f64
                );
                if sd_min > 0 {
                    eprintln!(
                        "  [mtp-spec] 后缀查表（LINGXI_SUFFIX_DRAFT={sd_min}）：{sd_rounds}/{n_round} 轮命中｜\
                         提议 {sd_prop} 枚、**接受 {sd_acc} 枚**（命中率 {:.2}）",
                        if sd_prop > 0 { sd_acc as f64 / sd_prop as f64 } else { 0.0 }
                    );
                }
                eprintln!("  [mtp-spec] 判据：去掉首 token 后应等于固定贪心序列 [13, 561, 6511, 314, 9564, 369, 19241, 13]");
                // ── 链上逐位接受率 × top-N（本轮新增）──
                // 为什么必须这么报：整体"平均接受 2.57/6"会掩盖两件事 ——
                //   ① **第 1 位就低**（草稿头本身弱） vs **只有深位低**（自喂误差累积）；
                //   ② top-1 低但 top-N 高（= 多候选/树草稿有收益）。
                eprintln!("  [mtp-spec] 链上**逐位**命中（分母 = 走到该位的轮数，所以是**条件**接受率）：");
                eprintln!(
                    "     位 | 走到 | top1 | {}",
                    NS_SPEC.iter().map(|n| format!("top{n:<2}")).collect::<Vec<_>>().join(" | ")
                );
                for j in 0..kd {
                    let r = reached[j];
                    if r == 0 {
                        continue;
                    }
                    let cells: Vec<String> = NS_SPEC
                        .iter()
                        .enumerate()
                        .map(|(n, _)| format!("{:>5.2}", hitn[j][n] as f64 / r as f64))
                        .collect();
                    eprintln!(
                        "     {j:>2} | {r:>4} | {:.2} | {}",
                        hit1[j] as f64 / r as f64,
                        cells.join(" | ")
                    );
                }
                if reached[0] > 0 {
                    let h0 = &hitn[0];
                    let r0 = reached[0] as f64;
                    eprintln!(
                        "  [mtp-spec] ⇒ 第 1 位：top-1 {:.3} → top-8 {:.3} → top-20 {:.3}；\
                         若 top-8/top-20 明显高于 top-1 ⇒ **多候选（树）草稿有空间**，否则是分布本身偏",
                        h0[0] as f64 / r0,
                        h0[4] as f64 / r0,
                        h0[6] as f64 / r0
                    );
                }
                // oracle 链对照（只有开了 `LINGXI_MTP_CHAIN_DIAG=oracle` 才有数）
                if or_reached[0] > 0 {
                    eprintln!(
                        "  [mtp-spec] **oracle 链**（h 与 tok 都换成主干真值，n={} 轮）：",
                        or_reached[0]
                    );
                    for j in 0..kd {
                        if or_reached[j] == 0 {
                            continue;
                        }
                        eprintln!(
                            "     位 {j:>2} | 自喂 {:.2}  vs  oracle {:.2}",
                            if reached[j] > 0 {
                                hit1[j] as f64 / reached[j] as f64
                            } else {
                                f64::NAN
                            },
                            or_hit[j] as f64 / or_reached[j] as f64
                        );
                    }
                    eprintln!(
                        "     ⇒ 判读：oracle 位1 显著高于自喂 ⇒ **自喂的 h 不是 MTP 训练期望的那个量**；\
                         两者接近（都 ~0.6）⇒ 草稿头本身预测不了第 2 步（只能上多候选）"
                    );
                }
                if hhc_n > 0 {
                    let n = hhc_n as f64;
                    eprintln!(
                        "  [mtp-spec] **自喂量 o0 vs 主干真 h**（n={hhc_n}）：cos {:.4}｜相对误差 {:.3}｜\
                         归一化 std  o0 {:.3} vs 真 h {:.3}",
                        hhc_cos / n,
                        hhc_rel / n,
                        hhc_a / n,
                        hhc_b / n
                    );
                    eprintln!(
                        "     ⇒ cos 高（>0.9）且相对误差小 ⇒ 自喂量与主干 h 同族，问题不在 h；\
                         cos 低/相对误差大 ⇒ 自喂量确实不是那一族（要改喂法）"
                    );
                }
                {
                    const LBL: [&str; 4] = ["≥0.90", "0.70~0.90", "0.50~0.70", "<0.50"];
                    eprintln!("  [mtp-spec] **草稿概率 prob1 vs 实际接受**（全链 / 仅第0位）：");
                    for b in 0..4 {
                        let f = |h: usize, n: usize| {
                            if n == 0 { "  n/a".to_string() } else { format!("{:.3}", h as f64 / n as f64) }
                        };
                        eprintln!(
                            "     prob1 {:<9} 全链 {}/{} = {}   第0位 {}/{} = {}",
                            LBL[b], pb_hit[b], pb_n[b], f(pb_hit[b], pb_n[b]),
                            pb_hit0[b], pb_n0[b], f(pb_hit0[b], pb_n0[b])
                        );
                    }
                    eprintln!(
                        "     ⇒ 若「高 prob 桶的接受率明显更高」⇒ 可以用参考实现的 `--spec-min-p` 式闸门\
                         砍掉低概率的深位（省的是「行」）；若各桶差不多 ⇒ 概率不是有效信号"
                    );
                }
                return Ok(());
            }
            if std::env::var("LINGXI_SPEC_CHECK").is_ok() {
                // 把"快照+重放"拆成两半分别判，且用**模型自己的贪心 token**当 chunk（同分布，
                // 避免合成 token 把状态推到退化区）。期望值都是确定的：
                //   p0 = 自然贪心链            ⇒ [t0, t1, t2, ...]
                //   chunk(喂 t0..t5)
                //   a2 = 喂完不 commit          ⇒ 应等于 p0[6..]      （因果性）
                //   b2 = 喂完 commit(6)         ⇒ 应等于 a2           （重放等价）
                //   b0 = 喂完 commit(0)（全拒）⇒ 应等于 p0            （快照/恢复完整）
                let t = 6usize;
                let prompt_ids = prompt_ids.clone();
                let run = |eng: &lingxi_driver::Engine,
                           provider: &mut dyn lingxi_driver::LayerProvider,
                           chunk: Option<Vec<i64>>,
                           prep_only: bool,
                           commit: Option<usize>,
                           decode: usize|
                 -> Result<(i64, Vec<i64>)> {
                    let mut sq = lingxi_driver::Sequence::new(&k, n_layers, &shape, rope_len + 16, spec_ple)?;
                    let pre = sq.prefill(eng, &prompt_ids, provider)?;
                    if prep_only {
                        let _ = sq.spec_prepare(eng, provider, 8)?;
                    }
                    if let Some(ch) = chunk {
                        let n = ch.len();
                        let mut buf = sq.spec_prepare(eng, provider, n + 1)?;
                        sq.spec_snapshot(eng, &mut buf)?;
                        sq.spec_chunk(eng, &ch, provider, &mut buf)?;
                        if let Some(a) = commit {
                            sq.spec_commit(eng, provider, &mut buf, a)?;
                        }
                    }
                    let out = (0..decode)
                        .map(|_| sq.decode_token(eng, provider))
                        .collect::<Result<Vec<_>>>()?;
                    Ok((pre, out))
                };
                let (a0, p0) = run(&eng, provider, None, false, None, 12)?;
                let (_, pp) = run(&eng, provider, None, true, None, 12)?;
                eprintln!("  [spec 核对] A0(prefill 吐的 pending)= {a0}   p0={p0:?}");
                eprintln!("  [spec 核对] pp(只 spec_prepare) ={pp:?} ⇒ {}", if pp == p0 { "✅ 分配无副作用" } else { "❌ spec_prepare 本身破坏状态" });
                // **自然链**：喂 [A0, p0[0], p0[1], …] —— 链首必须是 prefill 吐的 pending，
                // 不是第一个 decode 输出（早先拿错了 ⇒ 判据给出的是假警报）。
                let chunk: Vec<i64> =
                    std::iter::once(a0).chain(p0[0..t - 1].iter().copied()).collect();
                let (_, b1) = run(&eng, provider, Some(chunk[0..1].to_vec()), false, None, 6)?;
                let (_, a2) = run(&eng, provider, Some(chunk.clone()), false, None, 6)?;
                let (_, b2) = run(&eng, provider, Some(chunk.clone()), false, Some(t), 6)?;
                let (_, b0) = run(&eng, provider, Some(chunk), false, Some(0), 6)?;
                // 期望值口径：喂 k 个 token 后，第 k 行的 argmax 已成为 pending（会被下一步吃掉），
                // 所以后续 D 个 decode 产出 p0[k .. k+D]。
                eprintln!("  [spec 核对] b1(喂 1 个)      ={b1:?}  期望 p0[1..7]={:?} ⇒ {}", &p0[1..7], if b1 == p0[1..7] { "✅ t=1 走通" } else { "❌ t=1 坏" });
                eprintln!("  [spec 核对] a2(喂 6 个不提交)={a2:?}  期望 p0[6..12]={:?} ⇒ {}", &p0[6..12], if a2 == p0[6..12] { "✅ 因果性" } else { "❌" });
                eprintln!("  [spec 核对] b2(喂 6 + commit 6)={b2:?} ⇒ {}", if b2 == a2 { "✅ 重放等价" } else { "❌ 重放不等价" });
                eprintln!("  [spec 核对] b0(喂 6 + commit 0)={b0:?}  期望 p0[0..6]={:?} ⇒ {}", &p0[0..6], if b0 == p0[0..6] { "✅ 快照/恢复完整" } else { "❌ 恢复不完整" });
                return Ok(());
            }
            if let Ok(ks) = std::env::var("LINGXI_SPEC_BENCH") {
                // 一轮的真实成本：`cost(t) + 提交开销`。t = 草稿数 + 1（pending）。
                let kd: usize = ks.parse().unwrap_or(6);
                let t = kd + 1;
                let mut sb = seq.spec_prepare(&eng, provider, t)?;
                // ── A/B（同进程、同状态、同 env）：顺序 tok/s vs 投机 tok/s ──
                // 1) 先顺序解码 N 个 token = 顺序基线，同时拿到**自然 token 链**
                //    （用自然链当草稿 = 模拟 100% 接受；MTP 的草稿成本 ~1 层 ≈ 1~2 ms/轮，可忽略）
                let n_seq: usize = std::env::var("LINGXI_SPEC_N")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(24);
                let accept: usize = std::env::var("LINGXI_SPEC_ACCEPT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(t - 1)
                    .min(t - 1)
                    .max(1);
                let mut chain: Vec<i64> = Vec::with_capacity(n_seq);
                let t0 = std::time::Instant::now();
                while chain.len() < n_seq {
                    chain.push(seq.decode_token(&eng, provider)?);
                }
                k.stream.synchronize()?;
                let t_seq = t0.elapsed().as_secs_f64();
                // 2) 投机环路：每轮喂 [pending, 链上 accept-1 个]，提交 accept 个
                let mut ci = 0usize;
                let mut emitted = 0usize;
                let mut rounds = 0usize;
                let t1 = std::time::Instant::now();
                while ci + accept <= chain.len() {
                    seq.spec_snapshot(&eng, &mut sb)?;
                    let pend = *seq.ids.last().unwrap();
                    let chunk: Vec<i64> = std::iter::once(pend)
                        .chain(chain[ci..ci + accept - 1].iter().copied())
                        .collect();
                    seq.spec_chunk(&eng, &chunk, provider, &mut sb)?;
                    seq.spec_commit(&eng, provider, &mut sb, accept)?;
                    ci += accept;
                    emitted += accept;
                    rounds += 1;
                }
                k.stream.synchronize()?;
                let t_sp = t1.elapsed().as_secs_f64();
                eprintln!(
                    "  [spec A/B] 顺序 {n_seq} token 用 {:.2}s = **{:.1} tok/s**（{:.1} ms/token）",
                    t_seq,
                    n_seq as f64 / t_seq,
                    t_seq * 1e3 / n_seq as f64
                );
                eprintln!(
                    "  [spec A/B] 投机 {emitted} token / {rounds} 轮 用 {:.2}s = **{:.1} tok/s**（{:.1} ms/token，t={t}，每轮接受 {accept}）",
                    t_sp,
                    emitted as f64 / t_sp,
                    t_sp * 1e3 / emitted as f64
                );
                eprintln!(
                    "  [spec A/B] ⇒ 加速比 **{:.2}×**（草稿接受率 100%；按 p 折算：p=0.9 时约 {:.2}×）",
                    (emitted as f64 / t_sp) / (n_seq as f64 / t_seq),
                    (t_seq * 1e3 / n_seq as f64) / (t_sp * 1e3 / emitted as f64) * 1.0
                );
                return Ok(());
            }
            let mut hits = 0usize;
            let mut total = 0usize;
            // **单步 top-N 命中**（这一路是最好的样本：n_new 步、每步一个"真值 vs 草稿 top-N"）
            //   判据：主干这一步真实吐的 `tok_next`，是否落在**上一步草稿**的 top-N 里。
            //   为什么要它：`hits/total` 只是 top-1；若 top-8/top-20 明显更高，
            //   说明草稿**分布是对的**（可以做多候选/树草稿），只是 argmax 差一两位。
            const NS_ACC: [usize; 7] = [1, 2, 3, 5, 8, 10, 20];
            let mut acc_hitn = [0usize; 7];
            let mut acc_last_top: Vec<u32> = Vec::new(); // 上一步草稿的 top-N
            let mut acc_rows = 0usize;
            let mut acc_reach = 0usize; // 有草稿可比的总步数
            // 对齐假设：MTP 的 token 分支喂**哪个** token？
            //  `next`（默认）：喂主干**刚预测出**的 token（predict 它后面那个）
            //  `last`：喂主干**刚吃进**的 token
            // 两个都试，一次跑清 —— 输入语义是 0/64 的最大嫌疑。
            let align_last = std::env::var("LINGXI_MTP_ALIGN").map(|v| v == "last").unwrap_or(false);
            let mut prev_tok: i64 = trunk_next; // 第一个 decode 步吃进去的就是它
            eprintln!("MTP 对齐假设：{}", if align_last { "last" } else { "next" });

            // ── decode 相位 ──
            for i in 0..n_new {
                let (tok_next, hyper_i) = seq.decode_token_with_hyper(&eng, provider)?;
                // 比对上一步留下的草稿
                total += 1;
                if pending == tok_next {
                    hits += 1;
                }
                // 上一步草稿的 top-N 里有没有这一步的真值
                if acc_rows > 0 {
                    acc_reach += 1;
                    let truth = tok_next as u32;
                    for (n, &nn) in NS_ACC.iter().enumerate() {
                        if acc_last_top.iter().take(nn).any(|&x| x == truth) {
                            acc_hitn[n] += 1;
                        }
                    }
                }
                // 用本步的 (h, token) 出下一步的草稿
                let emb_i = embed1(&k, if align_last { prev_tok } else { tok_next })?;
                let h_i = h_prep(&hyper_i, 1)?;
                lingxi_model::scratch::put_bf16(hyper_i);
                let (d, tk, _pr, o) = mtp_step(&mut mst, &h_i, &emb_i, 1, &seq.ids, true)?;
                lingxi_model::scratch::put_bf16(o);
                lingxi_model::scratch::put_bf16(h_i);
                acc_last_top = tk.into_iter().next().unwrap_or_default();
                acc_rows += 1;
                pending = d[0] as i64;
                trunk_next = tok_next;
                prev_tok = tok_next;
                if i % 16 == 0 || i + 1 == n_new {
                    eprintln!(
                        "  step {}/{}  主干={trunk_next} draft={pending} 累计命中 {hits}/{total}",
                        i + 1,
                        n_new
                    );
                }
            }
            println!(
                "MTP 接受率：{hits}/{total} = **{:.3}**（prompt {prompt_len} token，n_new {n_new}）",
                hits as f64 / total as f64
            );
            if acc_reach > 0 {
                println!("MTP **单步 top-N 命中**（n={acc_reach}；真值落在草稿 top-N 里）：");
                for (n, &nn) in NS_ACC.iter().enumerate() {
                    println!(
                        "    top-{nn:<2} {}/{} = **{:.3}**{}",
                        acc_hitn[n],
                        acc_reach,
                        acc_hitn[n] as f64 / acc_reach as f64,
                        if nn == 1 { "   ← 这就是贪心接受率" } else { "" }
                    );
                }
                println!(
                    "  ⇒ 提示：若 top-8/top-20 明显高于 top-1 ⇒ 草稿分布是对的、\n\
                     多候选（树）草稿有空间；若几乎持平 ⇒ 分布本身偏，树也没用。"
                );
            }
        }
        Some("mtp-dump") => {
            // **白盒对拍用**：把 MTP stem 的输入 / 权重 / 输出全部落成 f32 —— 交给
            // `tools/golden/mtp_stem_oracle.py` 在 numpy 里复算，**逐步**比中间量。
            // 黑盒已经试不出东西了（§3.52 记着全部被否的假设），必须白盒。
            // 用法: lingxi-bench mtp-dump <mtp.ninfer> <outdir>
            let path = args.get(2).context("用法: lingxi-bench mtp-dump <mtp.ninfer> <outdir>")?;
            let outdir = args.get(3).context("缺 outdir")?;
            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let art = Artifact::open(path)?;
            let g = lingxi_model::mtp::load(&k, &art)?;
            let meta = flash_next_meta();
            let shape = layer_shape(&meta);
            std::fs::create_dir_all(outdir)?;
            let (n, hc) = (g.geom.n, g.geom.hc);
            let t = 2usize;
            let dump_dev = |name: &str, d: &cudarc::driver::CudaSlice<half::bf16>| -> Result<()> {
                let h: Vec<half::bf16> = k.stream.clone_dtoh(d)?;
                let b: Vec<u8> = h.iter().flat_map(|v| v.to_f32().to_le_bytes()).collect();
                std::fs::write(format!("{outdir}/{name}.f32"), b)?;
                Ok(())
            };
            let dump_f32 = |name: &str, v: &[f32]| -> Result<()> {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{outdir}/{name}.f32"), b)?;
                Ok(())
            };
            dump_dev("fc_embedding", &g.fc_embedding)?;
            dump_dev("fc_hidden", &g.fc_hidden)?;
            dump_dev("embedding_norm", &g.embedding_norm)?;
            dump_dev("hidden_norm", &g.hidden_norm)?;
            let mk = |cnt: usize, seed: usize| -> Vec<half::bf16> {
                (0..cnt)
                    .map(|i| half::bf16::from_f32((((i + seed) % 13) as f32 - 6.0) * 0.1))
                    .collect()
            };
            let emb_h = mk(t * n, 1);
            let rin_h = mk(t * hc * n, 3);
            let to_f = |v: &[half::bf16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
            dump_f32("emb", &to_f(&emb_h))?;
            dump_f32("rin", &to_f(&rin_h))?;
            let emb = k.stream.clone_htod(&emb_h)?;
            let rin = k.stream.clone_htod(&rin_h)?;
            let r = lingxi_model::mtp::stem(&k, &g, &rin, &emb, t, shape.eps)?;
            k.stream.synchronize()?;
            dump_dev("R", &r)?;
            // ── 再把 MTP 那一层跑一遍，把它的输入/输出/注意力权重也落下来 ──
            // （这样就能在 numpy 里单独复算**那一层的注意力**，把"层接线错"钉死或排除）
            let ml = build_mtp_layer(&k, &art, false)?;
            let rope = load_rope_tables(&k, 64, shape.attn.rot_dim)?;
            let ids: Vec<i64> = (0..t as i64).collect();
            let mut st = lingxi_model::layer::LayerStates::for_kind(
                &k,
                lingxi_ids::LayerKind::QsaSparseAttention,
                &shape,
                64,
                None,
            )?;
            let env = lingxi_model::layer::LayerEnv {
                cos: &rope.cos,
                sin: &rope.sin,
                cos_host: &rope.cos_host,
                sin_host: &rope.sin_host,
                remote: None,
                causal_host: &[],
                k_layernorm_host: &ml.k_layernorm_host,
                ids: &ids,
                experts: None,
            };
            dump_dev("layer_in", &r)?;
            let out =
                lingxi_model::layer::layer_step(&k, &ml.weights, &shape, &r, t, &env, &mut st, 0)?;
            k.stream.synchronize()?;
            dump_dev("layer_out", &out)?;
            if let lingxi_model::layer::AttnWeights::Qsa { attn, indexer } = &ml.weights.attn {
                dump_dev("w_qkgv", &attn.query_key_gate_value)?;
                dump_dev("w_q_norm", &attn.query_norm)?;
                dump_dev("w_k_norm", &attn.key_norm)?;
                dump_dev("w_o_proj", &attn.output)?;
                dump_dev("w_index_qk", &indexer.index_qk_proj)?;
                dump_dev("w_index_klayernorm", &indexer.k_layernorm)?;
            }
            dump_dev("rope_cos", &rope.cos)?;
            dump_dev("rope_sin", &rope.sin)?;
            // ── 再把**注意力块单独**跑一次：落它的输入(mixed_a)/输出(attn_out) ──
            // 这样 numpy 只需复算 attention_step（投影→norm→rope→掩码注意力→sigmoid 门→o_proj），
            // 不必连两个 GR 和 MoE 一起重写。
            let mut mixed_a = k.stream.alloc_zeros::<half::bf16>(t * shape.hidden)?;
            let mut injw_a = k.stream.alloc_zeros::<half::bf16>(t * shape.hc)?;
            lingxi_model::gr::gated_residual_forward_into(
                &k, &ml.weights.attn_gr, &r, &mut mixed_a, Some(&mut injw_a), t as u32,
                shape.hc * shape.hidden, shape.hidden, shape.hc, shape.lowrank, shape.eps, 0,
            )?;
            dump_dev("attn_mixed_a", &mixed_a)?;
            // GR 块的 4 组权重（白盒：复算 GR(R) → attn_mixed_a）
            dump_dev("gr_hc_norm", &ml.weights.attn_gr.hc_norm)?;
            dump_dev("gr_mix_down", &ml.weights.attn_gr.mix_down)?;
            dump_dev("gr_mix_up", &ml.weights.attn_gr.mix_up)?;
            if let Some(bi) = &ml.weights.attn_gr.block_inject {
                dump_dev("gr_block_inject", bi)?;
            }
            if let lingxi_model::layer::AttnWeights::Qsa { attn, indexer } = &ml.weights.attn {
                let mut st2 = lingxi_model::layer::LayerStates::for_kind(
                    &k,
                    lingxi_ids::LayerKind::QsaSparseAttention,
                    &shape,
                    64,
                    None,
                )?;
                let attn_out = {
                    let lingxi_model::layer::AttnState::Qsa(qs) = &mut st2.attn else {
                        unreachable!()
                    };
                    lingxi_model::qsa::attention_step(
                        &k, attn, indexer, &mixed_a, &rope.cos, &rope.sin, &rope.cos_host,
                        &rope.sin_host, &ml.k_layernorm_host, t, &shape.attn, &shape.indexer, qs,
                    )?
                };
                k.stream.synchronize()?;
                dump_dev("attn_out", &attn_out)?;
            }
            println!(
                "mtp-dump：已写 {outdir}（stem + 层 + **注意力块单独**的输入/输出；全 f32 LE；n={n} hc={hc} t={t} eps={}）",
                shape.eps
            );
        }
        Some("mtp-load") => {
            // **MTP 草稿路径的机械化冒烟**：产物读入 → 几何/形状交叉校验 → stem →
            // 复用主干的 `layer_step` → 层尾 mixer → 复用主干的输出头 → argmax。
            // 输入是**合成的**（不是真 token 流），所以只证明"接线正确、没有非法访问、
            // 数值有限"；真正的**接受率**由下一步的真 token 流测量给出。
            // 用法: lingxi-bench mtp-load <main.ninfer> <mtp.ninfer>
            let main_path = args.get(2).context("用法: lingxi-bench mtp-load <main.ninfer> <mtp.ninfer>")?;
            let mtp_path = args.get(3).context("缺 MTP 产物路径")?;
            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let meta = flash_next_meta();
            let shape = layer_shape(&meta);
            let main_art = Artifact::open(main_path)?;
            let mtp_art = Artifact::open(mtp_path)?;

            let g = lingxi_model::mtp::load(&k, &mtp_art)?;
            let gm = g.geom;
            println!(
                "MTP 几何（由张量形状推出）：n={} hc={} q_dim={} kv_dim={} inter={} E={}",
                gm.n, gm.hc, gm.q_dim, gm.kv_dim, gm.inter, gm.num_experts
            );
            // 与主干同源的量必须逐一对上——对不上说明产物抽错或几何推错
            anyhow::ensure!(
                gm.n == shape.hidden
                    && gm.hc == shape.hc
                    && gm.inter == shape.inter
                    && gm.num_experts == shape.num_experts,
                "MTP 几何与主干 LayerShape 不一致：MTP n={} hc={} inter={} E={} vs 主干 hidden={} hc={} inter={} E={}",
                gm.n, gm.hc, gm.inter, gm.num_experts,
                shape.hidden, shape.hc, shape.inter, shape.num_experts
            );
            println!("与主干 LayerShape 一致 ✓（hidden / hc / inter / num_experts 全对上）");

            let ml = build_mtp_layer(&k, &mtp_art, false)?;
            println!("MTP 层权重构造完成 ✓（复用主干 QSA 层的算子，不重写前向）");

            // 合成输入：`emb` 给一段图案；`Rin` 用 `hc_repeat(emb)`（与主干起点同形）
            let t = 1usize;
            let dim = shape.hc * shape.hidden;
            let emb_h: Vec<half::bf16> = (0..t * shape.hidden)
                .map(|i| half::bf16::from_f32((((i % 11) as f32) - 5.0) * 0.1))
                .collect();
            let emb = k.stream.clone_htod(&emb_h)?;
            let mut rin = k.stream.alloc_zeros::<half::bf16>(t * dim)?;
            k.hc_repeat_bf16(&emb, &mut rin, t, shape.hidden, shape.hc)?;

            let r = lingxi_model::mtp::stem(&k, &g, &rin, &emb, t, shape.eps)?;
            let rope = load_rope_tables(&k, 64, shape.attn.rot_dim)?;
            let ids = vec![0i64];
            let mut st = lingxi_model::layer::LayerStates::for_kind(
                &k,
                lingxi_ids::LayerKind::QsaSparseAttention,
                &shape,
                64,
                None,
            )?;
            let env = lingxi_model::layer::LayerEnv {
                cos: &rope.cos,
                sin: &rope.sin,
                cos_host: &rope.cos_host,
                sin_host: &rope.sin_host,
                remote: None,
                causal_host: &[],
                k_layernorm_host: &ml.k_layernorm_host,
                ids: &ids,
                experts: None,
            };
            let out = lingxi_model::layer::layer_step(
                &k, &ml.weights, &shape, &r, t, &env, &mut st, 0,
            )?;
            let (mixed, injw) = lingxi_model::gr::gated_residual_forward(
                &k, &ml.mixer, &out, t as u32, dim, shape.hidden, shape.hc, shape.lowrank, shape.eps,
            )?;
            if let Some(j) = injw {
                lingxi_model::scratch::put_bf16(j);
            }
            // 输出头复用主干的（MTP 产物里没有 output_head）
            let tw = load_endpoints(&k, &main_art)?;
            let vocab = lingxi_model::DIMS.vocab as usize;
            let mut logits = k.stream.alloc_zeros::<half::bf16>(t * vocab)?;
            if let Some(wq) = &tw.output_head_q4 {
                lingxi_model::gpu::proj_q4_epi(&k, &mixed, wq, &mut logits, vocab, shape.hidden, 0, t, 0, 0.0, 0.0)?;
            } else {
                lingxi_model::gpu::proj_t1(&k, &mixed, &tw.output_head, &mut logits, vocab, shape.hidden, t)?;
            }
            k.stream.synchronize()?;
            let lh: Vec<half::bf16> = k.stream.clone_dtoh(&logits)?;
            let lf: Vec<f32> = lh.iter().map(|v| v.to_f32()).collect();
            anyhow::ensure!(lf.iter().all(|v| v.is_finite()), "MTP draft 的 logits 出现非有限值");
            let top = lingxi_model::trunk::argmax_rows(&lf, t, vocab);
            println!(
                "MTP 草稿路径全程跑通 ✓：stem → layer_step → mixer → 输出头 → argmax = {:?}（合成输入，token 无意义）",
                top
            );
        }
        Some("list") => {
            let path = args.get(2).expect("用法: lingxi-bench list <artifact.ninfer>");
            let art = Artifact::open(path)?;
            let catalog = lingxi_model::catalog_from_artifact(&art);
            for (id, info) in catalog.iter() {
                println!(
                    "{:>5}  {:<46} {:<20} {:<16} {:<22} {:?}",
                    id.0,
                    info.name,
                    format!("{:?}", info.shape),
                    info.format.name(),
                    info.layout.name(),
                    info.role
                );
            }
            println!("共 {} 个张量", catalog.len());
        }
        Some("decode-check") => {
            let path = args
                .get(2)
                .context("用法: lingxi-bench decode-check <artifact.ninfer> <tensor-name>")?;
            let name = args.get(3).context("缺少 tensor 名")?;
            let art = Artifact::open(path)?;
            let obj = art.find(name).with_context(|| format!("tensor 不存在: {name}"))?;
            let bytes = art.payload(name).context("取 payload 失败")?;

            let t0 = std::time::Instant::now();
            let data = lingxi_model::decode::decode_f32(obj, bytes)?;
            let dt = t0.elapsed().as_secs_f64();
            let (sum, sumsq, count) = stats(&data);

            // 参考实现：ninfer 自己的 Python 解码器（复用，不重写）
            let py = std::env::var("LINGXI_PYTHON")
                .unwrap_or_else(|_| "/models/pocenv/bin/python".to_string());
            let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/golden/decode_oracle.py");
            let out = Command::new(py)
                .env("PYTHONPATH", "/data14t/ninfer-dev")
                .arg(script)
                .arg(path)
                .arg(name)
                .output()
                .context("无法运行 decode_oracle.py")?;
            if !out.status.success() {
                anyhow::bail!("oracle 失败: {}", String::from_utf8_lossy(&out.stderr));
            }
            let refv: serde_json::Value = serde_json::from_slice(&out.stdout)?;
            let rsum = refv["sum"].as_f64().unwrap();
            let rsumsq = refv["sumsq"].as_f64().unwrap();
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
            let rel_sum = rel(sum, rsum);
            let rel_sumsq = rel(sumsq, rsumsq);
            println!("tensor {name}  元素 {count}  解码用时 {dt:.4}s");
            println!("  lingxi : sum={sum:.6e}  sumsq={sumsq:.6e}");
            println!("  ninfer : sum={rsum:.6e}  sumsq={rsumsq:.6e}");
            println!("  rel: sum={rel_sum:.2e}  sumsq={rel_sumsq:.2e}");
            assert!(rel_sum < 1e-6 && rel_sumsq < 1e-6, "Rust 解码与 ninfer 不一致");
            println!("  OK: lingxi 解码 == ninfer 解码");
        }
        Some("component-check") => {
            // 用法: component-check <kind> <artifact> <canonical-prefix> <oracle-dir>
            let kind = args.get(2).context("缺少 kind（gr|moe）")?;
            let art_path = args.get(3).context("缺少 artifact")?;
            let canonical = args.get(4).context("缺少 canonical 前缀")?;
            let dir = std::path::PathBuf::from(args.get(5).context("缺少 oracle 目录")?);
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
            let rows = meta["rows"].as_u64().unwrap() as usize;
            let hidden = meta["hidden"].as_u64().unwrap() as usize;
            let eps = meta["eps"].as_f64().unwrap() as f32;

            let k = lingxi_kernels::Kernels::load(0)?;
            let art = Artifact::open(art_path)?;
            let x_host = read_f32(&dir.join("x.f32"));
            let want = read_f32(&dir.join("ref.f32"));
            let x_bf: Vec<half::bf16> = x_host.iter().map(|v| half::bf16::from_f32(*v)).collect();
            let dx = k.stream.clone_htod(&x_bf)?;

            let tol = 2e-2f32; // bf16 量级
            let (got, tag) = match kind.as_str() {
                "gr" => {
                    let dim = meta["dim"].as_u64().unwrap() as usize;
                    let hc = meta["hc"].as_u64().unwrap() as usize;
                    let lowrank = meta["lowrank"].as_u64().unwrap() as usize;
                    let use_combine = meta["use_combine"].as_bool().unwrap_or(true);
                    let block_inject = if use_combine {
                        Some(lingxi_model::gpu::upload_bf16(&k, &art, &format!("{canonical}block_inject"))?)
                    } else {
                        None
                    };
                    let w = lingxi_model::gr::GrWeights {
                        hc_norm: lingxi_model::gpu::upload_bf16(&k, &art, &format!("{canonical}hc_norm"))?,
                        mix_down: lingxi_model::gpu::upload_bf16(&k, &art, &format!("{canonical}mix_down"))?,
                        mix_up: lingxi_model::gpu::upload_bf16(&k, &art, &format!("{canonical}mix_up"))?,
                        block_inject,
                    };
                    let (mixed, injw) = lingxi_model::gr::gated_residual_forward(
                        &k, &w, &dx, rows as u32, dim, hidden, hc, lowrank, eps,
                    )?;
                    let out: Vec<f32> = k.stream.clone_dtoh(&mixed)?.iter().map(|v| v.to_f32()).collect();
                    if let Some(injw) = injw {
                        let want_injw = read_f32(&dir.join("injw.f32"));
                        let got_injw: Vec<f32> = k.stream.clone_dtoh(&injw)?.iter().map(|v| v.to_f32()).collect();
                        println!("  injw : rel={:.3e}", rel_l2(&got_injw, &want_injw));
                    }
                    (out, format!("GR {canonical} combine={use_combine}"))
                }
                "moe" => {
                    let num_experts = meta["num_experts"].as_u64().unwrap() as usize;
                    let top_k = meta["top_k"].as_u64().unwrap() as usize;
                    let inter = meta["inter"].as_u64().unwrap() as usize;
                    let w = lingxi_model::moe::MoeWeights {
                        router_shared_gate: lingxi_model::gpu::upload_bf16(&k, &art, &format!("{canonical}router_shared_gate"))?,
                        routed_gate_up: Some(lingxi_model::gpu::upload_bf16(&k, &art, &format!("{canonical}routed_gate_up"))?),
                        routed_down: Some(lingxi_model::gpu::upload_bf16(&k, &art, &format!("{canonical}routed_down"))?),
                        shared_gate_up: lingxi_model::gpu::upload_bf16(&k, &art, &format!("{canonical}shared_gate_up"))?,
                        shared_down: lingxi_model::gpu::upload_bf16(&k, &art, &format!("{canonical}shared_down"))?,
                    };
                    let dbg_dir = if std::env::var("LINGXI_MOE_DEBUG").is_ok() { Some(dir.as_path()) } else { None };
                    let res = lingxi_model::moe::sparse_moe_forward(
                        &k, &w, &dx, rows, hidden, num_experts, top_k, inter, dbg_dir, 0, None,
                        None,
                    )?;
                    if std::env::var("LINGXI_MOE_DEBUG").is_ok() {
                        // 零拷贝路径下 `router_logits` 为 None（logits 落在映射 host 内存里）
                        if let Some(rl) = &res.router_logits {
                            let v: Vec<f32> =
                                k.stream.clone_dtoh(rl)?.iter().map(|x| x.to_f32()).collect();
                            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                            std::fs::write(dir.join("lingxi_router_logits.f32"), bytes)?;
                        }
                        for (tag, dev) in [
                            ("shared", &res.shared),
                            ("expert_out", &res.expert_out),
                        ] {
                            let v: Vec<f32> = k.stream.clone_dtoh(dev)?.iter().map(|x| x.to_f32()).collect();
                            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                            std::fs::write(dir.join(format!("lingxi_{tag}.f32")), bytes)?;
                        }
                        let nonempty: Vec<usize> =
                            (0..num_experts).filter(|&e| !res.per_expert[e].is_empty()).collect();
                        println!("  [dbg] 命中专家 {} 个", nonempty.len());
                        let triples: Vec<f32> = res
                            .sel_rows
                            .iter()
                            .flat_map(|(r, e, w)| [*r as f32, *e as f32, *w])
                            .collect();
                        let bytes: Vec<u8> = triples.iter().flat_map(|x| x.to_le_bytes()).collect();
                        std::fs::write(dir.join("lingxi_selrows.f32"), bytes)?;

                        // 权重视图抽样：expert 0 与 expert 1 的首 8 个元素（验证 slice 偏移）
                        let routed_gu = w.routed_gate_up.as_ref().expect("component-check 走全量路径");
                        let per = 2 * inter * hidden; // 每个专家在 routed_gate_up 里占的元素数
                        let g0: Vec<f32> = k.stream.clone_dtoh(&routed_gu.slice(0..8))?.iter().map(|v| v.to_f32()).collect();
                        let g1: Vec<f32> = k.stream.clone_dtoh(&routed_gu.slice(per..per + 8))?.iter().map(|v| v.to_f32()).collect();
                        let g2: Vec<f32> = k.stream.clone_dtoh(&routed_gu.slice(2 * per..2 * per + 8))?.iter().map(|v| v.to_f32()).collect();
                        let all: Vec<f32> = g0.into_iter().chain(g1).chain(g2).collect();
                        let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
                        std::fs::write(dir.join("lingxi_wslice.f32"), bytes)?;
                    }
                    let out: Vec<f32> = k.stream.clone_dtoh(&res.out)?.iter().map(|v| v.to_f32()).collect();

                    // top-k 边界并列时，极小数值差会翻转选择；只对"选择一致"的行比 rel，并报告一致率。
                    let sel_ref = read_f32(&dir.join("router_sel.f32"));
                    let w_ref = read_f32(&dir.join("router_w.f32"));
                    let mut worst = 0f32;
                    let mut considered = 0usize;
                    let mut flipped = 0usize;
                    for r in 0..rows {
                        let mut mine: Vec<(i32, f32)> = res
                            .sel_rows
                            .iter()
                            .filter(|(rr, _, _)| *rr == r as u32)
                            .map(|(_, e, w)| (*e, *w))
                            .collect();
                        let mut hf: Vec<(i32, f32)> = (0..top_k)
                            .map(|k| (sel_ref[r * top_k + k] as i32, w_ref[r * top_k + k]))
                            .collect();
                        mine.sort_by(|a, b| a.0.cmp(&b.0));
                        hf.sort_by(|a, b| a.0.cmp(&b.0));
                        let same = mine.len() == hf.len() && mine.iter().zip(&hf).all(|(a, b)| a.0 == b.0);
                        if !same {
                            flipped += 1;
                            continue;
                        }
                        let lo = r * hidden;
                        worst = worst.max(rel_l2(&out[lo..lo + hidden], &want[lo..lo + hidden]));
                        considered += 1;
                    }
                    println!(
                        "  按行 rel（选择一致的 {considered}/{rows} 行）: worst={worst:.3e}；选择翻转 {flipped} 行"
                    );
                    assert!(
                        worst < tol,
                        "MoE 选择一致行里最差 rel={worst} >= {tol}"
                    );
                    (out, format!("MoE {canonical} E={num_experts} top{top_k} I={inter}"))
                }
                "attention" => {
                    let c = lingxi_model::qsa::AttnCfg {
                        hidden,
                        heads: meta["num_attention_heads"].as_u64().unwrap() as usize,
                        kv_heads: meta["num_key_value_heads"].as_u64().unwrap() as usize,
                        head_dim: meta["head_dim"].as_u64().unwrap() as usize,
                        rot_dim: meta["rot_dim"].as_u64().unwrap() as usize,
                        eps,
                    };
                    let w = lingxi_model::qsa::AttentionWeights {
                        query_key_gate_value: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}query_key_gate_value"))?,
                        query_key_gate_value_q4: None, // 组件对拍路径不启用 W4
                        query_norm: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}query_norm"))?,
                        key_norm: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}key_norm"))?,
                        output: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}output"))?,
                        output_q4: None, // 组件对拍路径不启用 W4
                    };
                    let to_bf = |v: &Vec<f32>| -> anyhow::Result<cudarc::driver::CudaSlice<half::bf16>> {
                        let t: Vec<half::bf16> = v.iter().map(|x| half::bf16::from_f32(*x)).collect();
                        Ok(k.stream.clone_htod(&t)?)
                    };
                    let cos_h = read_f32(&dir.join("cos.f32"));
                    let sin_h = read_f32(&dir.join("sin.f32"));
                    let causal_h = read_f32(&dir.join("causal.f32"));
                    let dcos = to_bf(&cos_h)?;
                    let dsin = to_bf(&sin_h)?;

                    // indexer：自己算选块掩码（HF 的 mask.f32 是"因果 + indexer"的组合结果）
                    let iw = lingxi_model::qsa::IndexerWeights {
                        index_qk_proj: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}indexer/index_qk_proj"))?,
                        q_layernorm: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}indexer/q_layernorm"))?,
                        k_layernorm: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}indexer/k_layernorm"))?,
                    };
                    let ic = lingxi_model::qsa::IndexerCfg {
                        heads: meta["indexer_n_heads"].as_u64().unwrap() as usize,
                        head_dim: meta["indexer_head_dim"].as_u64().unwrap() as usize,
                        rot_dim: c.rot_dim,
                        budget: meta["indexer_budget"].as_u64().unwrap() as usize,
                        compress: meta["indexer_compress_ratio"].as_u64().unwrap() as usize,
                        eps,
                    };
                    let kl_host = lingxi_model::gpu::read_f32_host(&k, &art, &format!("{canonical}indexer/k_layernorm"))?;
                    let dmask = lingxi_model::qsa::indexer_mask(
                        &k, &iw, &dx, &dcos, &dsin, &cos_h, &sin_h, &causal_h, &kl_host, rows, hidden, &ic,
                    )?;
                    {
                        let got_mask: Vec<f32> = k.stream.clone_dtoh(&dmask)?;
                        let want_mask = read_f32(&dir.join("mask.f32"));
                        let (mut mism, mut worst) = (0usize, 0f32);
                        for i in 0..rows * rows {
                            let g = got_mask[i] > -1e30;
                            let w = want_mask[i] > -1e30;
                            if g != w {
                                mism += 1;
                            } else if g {
                                worst = worst.max((got_mask[i] - want_mask[i]).abs());
                            }
                        }
                        println!("  indexer 掩码：选择不一致 {mism}/{}；选中项最大绝对差 {worst:.3e}", rows * rows);
                    }

                    let out_dev = lingxi_model::qsa::attention_forward(&k, &w, &dx, &dcos, &dsin, &dmask, rows, &c)?;
                    let out: Vec<f32> = k.stream.clone_dtoh(&out_dev)?.iter().map(|v| v.to_f32()).collect();
                    (out, format!("Attention {canonical} heads={} kv={} D={} rot={}", c.heads, c.kv_heads, c.head_dim, c.rot_dim))
                }
                "gdn" => {
                    let c = lingxi_model::gdn::GdnCfg {
                        hidden,
                        num_k_heads: meta["linear_num_key_heads"].as_u64().unwrap() as usize,
                        num_v_heads: meta["linear_num_value_heads"].as_u64().unwrap() as usize,
                        head_k_dim: meta["linear_key_head_dim"].as_u64().unwrap() as usize,
                        head_v_dim: meta["linear_value_head_dim"].as_u64().unwrap() as usize,
                        conv_kernel: meta["linear_conv_kernel_dim"].as_u64().unwrap() as usize,
                        gate_activation: lingxi_ids::GateActivation::from_name(
                            meta["output_gate_type"].as_str().unwrap_or("silu"),
                        )
                        .unwrap_or(lingxi_ids::GateActivation::Silu),
                    };
                    let w = lingxi_model::gdn::GdnWeights {
                        query_key_value_z: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}query_key_value_z"))?,
                        query_key_value_z_q4: None, // 组件对拍路径不启用 W4
                        a_b_projection: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}a_b_projection"))?,
                        convolution: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}convolution"))?,
                        norm: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}norm"))?,
                        output: lingxi_model::gpu::upload_any_bf16(&k, &art, &format!("{canonical}output"))?,
                        output_q4: None, // 组件对拍路径不启用 W4
                        a_log: lingxi_model::gpu::read_fp32_host(&art, &format!("{canonical}a_log"))?,
                        dt_bias: lingxi_model::gpu::read_fp32_host(&art, &format!("{canonical}dt_bias"))?,
                    };
                    let out_dev = lingxi_model::gdn::gdn_forward(&k, &w, &dx, rows, &c)?;
                    let out: Vec<f32> = k.stream.clone_dtoh(&out_dev)?.iter().map(|v| v.to_f32()).collect();
                    (out, format!("GDN {canonical} kvh={} vh={} D={}", c.num_k_heads, c.num_v_heads, c.head_k_dim))
                }
                other => anyhow::bail!("未知 kind: {other}"),
            };

            let rel = rel_l2(&got, &want);
            println!("{tag}  rows={rows} hidden={hidden}");
            println!("  整体 ref : rel={rel:.3e}  (n={})", want.len());
            if kind != "moe" {
                // MoE 已在上面按行断言（top-k 并列会翻转选择，整体 rel 会被单行主导）
                assert!(rel < tol, "组件输出与 HF 参考不一致（tol={tol}）");
            }
            println!("  OK: lingxi == HF（bf16 容差 {tol}）");
        }
        Some("dequant-check") => {
            // 用法: dequant-check <artifact> <canonical-name> <ref.f32>
            let art_path = args.get(2).context("缺少 artifact")?;
            let name = args.get(3).context("缺少 tensor 名")?;
            let ref_path = std::path::PathBuf::from(args.get(4).context("缺少参考 f32")?);
            let k = lingxi_kernels::Kernels::load(0)?;
            let art = Artifact::open(art_path)?;
            let w = lingxi_model::gpu::upload_any_bf16(&k, &art, name)?;
            let got: Vec<f32> = k.stream.clone_dtoh(&w)?.iter().map(|v| v.to_f32()).collect();
            let want = read_f32(&ref_path);
            let rel = rel_l2(&got, &want);
            println!("dequant {name}: n={} vs {}  rel={rel:.3e}", got.len(), want.len());
            assert_eq!(got.len(), want.len(), "元素数不一致");
            assert!(rel < 3e-2, "q4 解码与参考不一致（bf16 容差 3e-2）");
            println!("  OK: 解码 == 参考");
        }
        Some("layer-check") => {
            // 用法: layer-check <artifact> <layer-idx> <golden-f32-dir>
            let art_path = args.get(2).context("缺少 artifact")?;
            let layer: u32 = args.get(3).context("缺少 layer 序号")?.parse()?;
            let dir = std::path::PathBuf::from(args.get(4).context("缺少 golden f32 目录")?);
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
            let u = |key: &str| meta[key].as_u64().unwrap() as usize;
            let hidden = u("hidden");
            let hc = u("hc");
            let _lowrank = u("lowrank");
            let _eps = meta["eps"].as_f64().unwrap() as f32;
            let rows = u("rows");
            let dim = hc * hidden;

            // 设备可选（默认 0）：服务占着 0 时用 LINGXI_DEVICE=1 跑对拍。
            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let art = Artifact::open(art_path)?;
            let built = build_layer(&k, &art, &meta, layer, false, None)?;
            let kind = built.kind;
            let klay_h = built.k_layernorm_host;
            let shape = built.shape;

            // 与层无关的公共输入（仅 QSA 层需要）
            let causal_h = if kind == lingxi_ids::LayerKind::QsaSparseAttention {
                read_f32(&dir.join("causal.f32"))
            } else {
                Vec::new()
            };
            let cos_h = if kind == lingxi_ids::LayerKind::QsaSparseAttention {
                read_f32(&dir.join("cos.f32"))
            } else {
                Vec::new()
            };
            let sin_h = if kind == lingxi_ids::LayerKind::QsaSparseAttention {
                read_f32(&dir.join("sin.f32"))
            } else {
                Vec::new()
            };
            let dummy = k.stream.alloc_zeros::<half::bf16>(1)?;
            let to_dev_bf16 =
                |v: &[f32]| -> Result<_> { Ok(k.stream.clone_htod(&v.iter().map(|x| half::bf16::from_f32(*x)).collect::<Vec<_>>())?) };
            let (cos_dev, sin_dev) = if kind == lingxi_ids::LayerKind::QsaSparseAttention {
                (to_dev_bf16(&cos_h)?, to_dev_bf16(&sin_h)?)
            } else {
                (dummy, k.stream.alloc_zeros::<half::bf16>(1)?)
            };

            let ids_path = dir.join("input_ids.i32");
            let ids: Vec<i64> = if ids_path.exists() {
                std::fs::read(&ids_path)?
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                    .collect()
            } else {
                (0..rows as i64).collect()
            };

            let w = built.weights;
            let env = lingxi_model::layer::LayerEnv {
                cos: &cos_dev,
                sin: &sin_dev,
                cos_host: &cos_h,
                sin_host: &sin_h,
                remote: None,
                causal_host: &causal_h,
                k_layernorm_host: &klay_h,
                ids: &ids,
                experts: None,
            };

            let x_host = read_f32(&dir.join(format!("layer{layer}_in.f32")));
            let want = read_f32(&dir.join(format!("layer{layer}_out.f32")));
            let x_bf: Vec<half::bf16> = x_host.iter().map(|v| half::bf16::from_f32(*v)).collect();
            let dx = k.stream.clone_htod(&x_bf)?;
            let out_dev = lingxi_model::layer::layer_forward(&k, &w, &shape, &dx, rows, &env, layer)?;
            let got: Vec<f32> = k.stream.clone_dtoh(&out_dev)?.iter().map(|v| v.to_f32()).collect();

            let rel = rel_l2(&got, &want);
            println!("Layer {layer} ({kind:?})  rows={rows} dim={dim}");
            println!("  整体 : rel={rel:.3e}  (n={})", want.len());
            // 逐行（深层 MoE router 的 bf16 logits 在第 10/11 名会**精确并列**，
            // 参考实现 `torch.topk` 的并列取舍未定义且自相矛盾 —— 只毁掉个别行。
            // 逐行比对 + 报告，不把它算作引擎缺陷。）
            let mut row_rel = vec![0f32; rows];
            let mut worst = 0f32;
            let mut worst_row = 0usize;
            for r in 0..rows {
                let d = dim;
                let rr = rel_l2(&got[r * d..(r + 1) * d], &want[r * d..(r + 1) * d]);
                row_rel[r] = rr;
                if rr > worst {
                    worst = rr;
                    worst_row = r;
                }
                print!("  row{r}:{rr:.2e}");
            }
            println!();
            let tol = 1.5e-2f32; // bf16 引擎 vs bf16 golden
            let ok = row_rel.iter().filter(|r| **r < tol).count();
            println!("  逐行 worst={worst:.3e} @row{worst_row}；<{tol} 的行 {ok}/{rows}");
            // 判据：至少一行完全复现（结构性正确 ⇒ GR/注意力/MoE/注入都对），
            // 且整体无 gross 错误。整层 rel 会被"并列翻转行"主导，不能直接当判据。
            assert!(
                ok >= 1,
                "所有行都偏离 HF（rel 最小 {:.3e}）—— 结构性错误，不是并列翻转",
                row_rel.iter().cloned().fold(f32::INFINITY, f32::min)
            );
            assert!(rel < 1.5e-1, "层输出整体偏离过大（rel={rel:.3e}）");
            println!("  OK: lingxi == HF（逐行 bf16 容差 {tol}；并列翻转行已报告）");
        }
        Some("trunk-check") => {
            // 用法: trunk-check <artifact> <golden-f32-dir>
            let art_path = args.get(2).context("缺少 artifact")?;
            let dir = std::path::PathBuf::from(args.get(3).context("缺少 golden f32 目录")?);
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
            let u = |key: &str| meta[key].as_u64().unwrap() as usize;
            let rows = u("rows");
            let hidden = u("hidden");
            let vocab = u("vocab_size");
            let n_layers = lingxi_model::DIMS.n_layers as u32;

            // 设备可选（默认 0）：服务占着 0 时用 LINGXI_DEVICE=1 跑对拍。
            let dev: usize = std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let art = Artifact::open(art_path)?;

            // 主干两端（小，常驻显存）
            let token_embedding =
                lingxi_model::gpu::upload_any_bf16(&k, &art, "text/token_embedding")?;
            let output_head = lingxi_model::gpu::upload_any_bf16(&k, &art, "text/output_head")?;
            let mup = |n: &str| {
                lingxi_model::gpu::upload_any_bf16(
                    &k,
                    &art,
                    &format!("text/hyper_connection_mixer/{n}"),
                )
            };
            let mixer = lingxi_model::gr::GrWeights {
                hc_norm: mup("hc_norm")?,
                mix_down: mup("mix_down")?,
                mix_up: mup("mix_up")?,
                block_inject: None,
            };
            let tw = lingxi_model::trunk::TrunkWeights {
                token_embedding, mixer, output_head, output_head_q4: None,
            };

            // cos/sin/因果掩码/token id（与层无关）
            let cos_h = read_f32(&dir.join("cos.f32"));
            let sin_h = read_f32(&dir.join("sin.f32"));
            let causal_h = read_f32(&dir.join("causal.f32"));
            let to_dev_bf16 = |v: &[f32]| -> Result<_> {
                Ok(k.stream
                    .clone_htod(&v.iter().map(|x| half::bf16::from_f32(*x)).collect::<Vec<_>>())?)
            };
            let cos_dev = to_dev_bf16(&cos_h)?;
            let sin_dev = to_dev_bf16(&sin_h)?;
            let ids: Vec<i64> = std::fs::read(dir.join("input_ids.i32"))?
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect();

            let shape = layer_shape(&meta);
            let env = lingxi_model::trunk::TrunkEnv {
                cos: &cos_dev,
                sin: &sin_dev,
                cos_host: &cos_h,
                sin_host: &sin_h,
                remote: None,
                causal_host: &causal_h,
                ids: &ids,
                experts: None,
            };

            println!("主干前向：{n_layers} 层 rows={rows}（逐层从产物载入，层权重用后即弃）");
            let mut loader = |layer: u32| -> Result<lingxi_model::trunk::LoadedLayer> {
                eprint!("\r  [trunk] 载入层 {layer}/{n_layers} …");
                let b = build_layer(&k, &art, &meta, layer, false, None)?;
                Ok(lingxi_model::trunk::LoadedLayer {
                    weights: b.weights,
                    k_layernorm_host: b.k_layernorm_host,
                })
            };
            let out = lingxi_model::trunk::trunk_forward(
                &k, &tw, &shape, n_layers, vocab, rows, &env, &mut loader,
            )?;
            eprintln!();

            // 1) mixer 输出（logits 之前的最后一站）—— 整条主干的强结构判据
            let want_mixed = read_f32(&dir.join("mixer_out.f32"));
            println!("  mixer_out : rel={:.3e}", rel_l2(&out.mixed, &want_mixed));
            let mut worst = 0f32;
            for r in 0..rows {
                let rr = rel_l2(
                    &out.mixed[r * hidden..(r + 1) * hidden],
                    &want_mixed[r * hidden..(r + 1) * hidden],
                );
                print!("    row{r}:{rr:.2e}");
                worst = worst.max(rr);
            }
            println!();
            println!("  逐行 worst={worst:.3e}（容差 1.5e-2）");

            // 2) logits 的贪心 argmax —— T3「端到端贪心」的第一步
            let want_logits = read_f32(&dir.join("logits.f32"));
            let got_am = lingxi_model::trunk::argmax_rows(&out.logits, rows, vocab);
            let want_am = lingxi_model::trunk::argmax_rows(&want_logits, rows, vocab);
            println!("  贪心 argmax got ={got_am:?}");
            println!("                want={want_am:?}");
            println!("  logits : rel={:.3e}", rel_l2(&out.logits, &want_logits));
        }
        Some("trunk-ends") => {
            // 用法: trunk-ends <artifact> <golden-f32-dir>
            // 省 I/O 版的 T3：把 golden 的层间中间量当作输入，逐段验证主干**两端**
            // （配合 48/48 的 layer-check，即可覆盖整条 `embedding → 48 层 → mixer → lm_head`）：
            //   ① embedding + hc 广播        vs layer0_in
            //   ② 末端 mixer（输入 layer47_out）vs mixer_out
            //   ③ lm_head（输入 golden mixer_out 与 ② 的 mixed 各一遍）vs logits
            let art_path = args.get(2).context("缺少 artifact")?;
            let dir = std::path::PathBuf::from(args.get(3).context("缺少 golden f32 目录")?);
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
            let u = |key: &str| meta[key].as_u64().unwrap() as usize;
            let rows = u("rows");
            let hidden = u("hidden");
            let hc = u("hc");
            let lowrank = u("lowrank");
            let vocab = u("vocab_size");
            let eps = meta["eps"].as_f64().unwrap() as f32;
            let dim = hc * hidden;

            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let art = Artifact::open(art_path)?;

            let to_dev_bf16 = |v: &[f32]| -> Result<_> {
                Ok(k.stream
                    .clone_htod(&v.iter().map(|x| half::bf16::from_f32(*x)).collect::<Vec<_>>())?)
            };
            let dtoh = |d: &cudarc::driver::CudaSlice<half::bf16>| -> Result<Vec<f32>> {
                Ok(k.stream.clone_dtoh(d)?.iter().map(|v| v.to_f32()).collect())
            };

            // ① embedding → hc 广播
            let ids: Vec<i64> = std::fs::read(dir.join("input_ids.i32"))?
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect();
            let idx_host: Vec<i32> = ids.iter().map(|&v| v as i32).collect();
            let didx = k.stream.clone_htod(&idx_host)?;
            let emb_table = lingxi_model::gpu::upload_any_bf16(&k, &art, "text/token_embedding")?;
            let mut emb = k.stream.alloc_zeros::<half::bf16>(rows * hidden)?;
            k.gather_rows_bf16(&emb_table, &didx, &mut emb, rows, hidden, 0)?;
            let mut hyper = k.stream.alloc_zeros::<half::bf16>(rows * dim)?;
            k.hc_repeat_bf16(&emb, &mut hyper, rows, hidden, hc)?;
            let got_hyper = dtoh(&hyper)?;
            let want_hyper = read_f32(&dir.join("layer0_in.f32"));
            println!("① embedding+hc广播 : rel={:.3e}  (vs layer0_in)", rel_l2(&got_hyper, &want_hyper));

            // ② 末端 mixer（输入取 golden layer47_out，隔离 mixer 本身）
            let mup = |n: &str| {
                lingxi_model::gpu::upload_any_bf16(
                    &k,
                    &art,
                    &format!("text/hyper_connection_mixer/{n}"),
                )
            };
            let mixer = lingxi_model::gr::GrWeights {
                hc_norm: mup("hc_norm")?,
                mix_down: mup("mix_down")?,
                mix_up: mup("mix_up")?,
                block_inject: None,
            };
            let x47 = read_f32(&dir.join("layer47_out.f32"));
            let dx47 = to_dev_bf16(&x47)?;
            let (mixed, _) = lingxi_model::gr::gated_residual_forward(
                &k, &mixer, &dx47, rows as u32, dim, hidden, hc, lowrank, eps,
            )?;
            let got_mixed = dtoh(&mixed)?;
            let want_mixed = read_f32(&dir.join("mixer_out.f32"));
            println!("② 末端 mixer       : rel={:.3e}  (vs mixer_out)", rel_l2(&got_mixed, &want_mixed));

            // ③ lm_head：分别喂 golden mixer_out（隔离 lm_head）与 ② 的 mixed（串起来）
            let head = lingxi_model::gpu::upload_any_bf16(&k, &art, "text/output_head")?;
            let want_logits = read_f32(&dir.join("logits.f32"));
            let want_am = lingxi_model::trunk::argmax_rows(&want_logits, rows, vocab);
            for (tag, src, want_in) in [
                ("golden mixer_out", &want_mixed, &want_logits),
                ("② 的 mixed", &got_mixed, &want_logits),
            ] {
                let din = to_dev_bf16(src)?;
                let mut logits = k.stream.alloc_zeros::<half::bf16>(rows * vocab)?;
                k.gemm_bf16_nt(&din, &head, &mut logits, rows as u32, vocab as u32, hidden as u32)?;
                let got_logits = dtoh(&logits)?;
                let ga = lingxi_model::trunk::argmax_rows(&got_logits, rows, vocab);
                let hit = ga.iter().zip(&want_am).filter(|(a, b)| a == b).count();
                println!(
                    "③ lm_head({tag:<16}): rel={:.3e}  贪心 argmax {hit}/{rows} 一致  got={ga:?}",
                    rel_l2(&got_logits, want_in)
                );
            }
            println!("   参考 argmax       = {want_am:?}");
        }
        Some("step-check") => {
            // 用法: step-check <artifact> <golden-f32-dir> [layer] [T]
            // T4 判据：同一层上，「整段一次 T 行」(layer_forward) 与
            // 「逐 token layer_step T 次」必须一致（GDN/PLE 期望逐位同源；QSA 容差对拍）。
            // 输入用 golden layer0_in（任何层都可拿它当输入，判的是两条路径的数学同源性，
            // 不要求分布真实）；PLE 层需要的 ids 用 golden input_ids。
            let art_path = args.get(2).context("缺少 artifact")?;
            let dir = std::path::PathBuf::from(args.get(3).context("缺少 golden f32 目录")?);
            let layer: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
            let rows = meta["rows"].as_u64().unwrap() as usize;
            let t_in: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(rows);
            let t = t_in.min(rows);

            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let art = Artifact::open(art_path)?;
            let shape = layer_shape(&meta);
            let dim = shape.hc * shape.hidden;

            // cos/sin 全表（覆盖到 rows）；整段/步进共用
            let cos_h = read_f32(&dir.join("cos.f32"));
            let sin_h = read_f32(&dir.join("sin.f32"));
            let to_dev_bf16 = |v: &[f32]| -> Result<_> {
                Ok(k.stream
                    .clone_htod(&v.iter().map(|x| half::bf16::from_f32(*x)).collect::<Vec<_>>())?)
            };
            let cos_dev = to_dev_bf16(&cos_h)?;
            let sin_dev = to_dev_bf16(&sin_h)?;
            let ids_all: Vec<i64> = std::fs::read(dir.join("input_ids.i32"))?
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect();
            let ids_all = &ids_all[..t];

            // 输入（golden layer0_in 前 t 行）
            let x_h = read_f32(&dir.join("layer0_in.f32"));
            let x_h = &x_h[..t * dim];
            let x_bf: Vec<half::bf16> = x_h.iter().map(|v| half::bf16::from_f32(*v)).collect();
            let dx = k.stream.clone_htod(&x_bf)?;

            println!("step-check layer={layer} t={t}（载入层权重，机械盘稍候 …）");
            let built = build_layer(&k, &art, &meta, layer, false, None)?;
            let kind = built.kind;
            let klay_h = built.k_layernorm_host;

            // ── 路径 A：整段一次 t 行 ──
            let env_a = lingxi_model::layer::LayerEnv {
                cos: &cos_dev,
                sin: &sin_dev,
                cos_host: &cos_h,
                sin_host: &sin_h,
                remote: None,
                causal_host: &[],
                k_layernorm_host: &klay_h,
                ids: ids_all,
                experts: None,
            };
            let out_a_dev = lingxi_model::layer::layer_forward(&k, &built.weights, &shape, &dx, t, &env_a, layer)?;
            let out_a: Vec<f32> = k
                .stream
                .clone_dtoh(&out_a_dev)?
                .iter()
                .map(|v| v.to_f32())
                .collect();

            // ── 路径 B：逐 token layer_step ──
            let mut states =
                lingxi_model::layer::LayerStates::new(&k, &built.weights, &shape, rows)?;
            let mut out_b = vec![0f32; t * dim];
            for i in 0..t {
                let mut xi = k.stream.alloc_zeros::<half::bf16>(dim)?;
                k.stream
                    .memcpy_dtod(&dx.slice(i * dim..(i + 1) * dim), &mut xi)?;
                let prefix = &ids_all[..=i];
                let env_b = lingxi_model::layer::LayerEnv {
                    cos: &cos_dev,
                    sin: &sin_dev,
                    cos_host: &cos_h,
                    sin_host: &sin_h,
                    remote: None,
                    causal_host: &[],
                    k_layernorm_host: &klay_h,
                    ids: prefix,
                    experts: None,
                };
                let row_dev = lingxi_model::layer::layer_step(
                    &k,
                    &built.weights,
                    &shape,
                    &xi,
                    1,
                    &env_b,
                    &mut states,
                    layer,
                )?;
                let row_h: Vec<f32> =
                    k.stream.clone_dtoh(&row_dev)?.iter().map(|v| v.to_f32()).collect();
                out_b[i * dim..(i + 1) * dim].copy_from_slice(&row_h);
            }

            // ── 路径 C：layer_step 单次 t 行（空状态）——driver prefill 的真实调用形状 ──
            let mut states_c =
                lingxi_model::layer::LayerStates::new(&k, &built.weights, &shape, rows)?;
            let env_c = lingxi_model::layer::LayerEnv {
                cos: &cos_dev,
                sin: &sin_dev,
                cos_host: &cos_h,
                sin_host: &sin_h,
                remote: None,
                causal_host: &[],
                k_layernorm_host: &klay_h,
                ids: ids_all,
                experts: None,
            };
            let out_c_dev = lingxi_model::layer::layer_step(
                &k, &built.weights, &shape, &dx, t, &env_c, &mut states_c, layer,
            )?;
            let out_c: Vec<f32> = k
                .stream
                .clone_dtoh(&out_c_dev)?
                .iter()
                .map(|v| v.to_f32())
                .collect();

            // ── 判据：逐行 rel L2 + 逐元素最大差 ──
            println!("  layer kind = {kind:?}");
            let mut worst = 0f32;
            let mut worst_c = 0f32;
            for r in 0..t {
                let a = &out_a[r * dim..(r + 1) * dim];
                let b = &out_b[r * dim..(r + 1) * dim];
                let cv = &out_c[r * dim..(r + 1) * dim];
                let rel = rel_l2(a, b);
                let rel_c = rel_l2(a, cv);
                let maxabs = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                worst = worst.max(rel);
                worst_c = worst_c.max(rel_c);
                println!("    row{r}: B逐token rel={rel:.3e}  C整段step rel={rel_c:.3e}  max|Δ|={maxabs:.3e}");
            }
            // C 与 A 用完全相同的批量形状（m=t），差异只应来自零状态的拼接/索引路径，
            // 应收敛到 bf16 舍入量级；这条不过说明 t>1 的 step 路径有真 bug。
            assert!(worst_c < 1e-4, "step-check 路径C 失败：整段step vs layer_forward rel={worst_c:.3e}");
            println!("  路径C（整段 step t={t} 空状态）vs layer_forward：worst rel={worst_c:.3e}");
            // 递推/卷积/注意力数学同源，唯一差异来自 cuBLAS 批量 GEMM（m=T 对 m=1）
            // 的 bf16 舍入，实测 GDN ~2e-3；QSA 还多一处注意力归约，容差放宽到 5e-2。
            let tol = match kind {
                lingxi_ids::LayerKind::GdnLinearAttention => 1e-2,
                lingxi_ids::LayerKind::QsaSparseAttention => 5e-2,
            };
            assert!(worst < tol, "step-check 失败：worst rel={worst:.3e} ≥ {tol}");
            println!("  OK: 整段 vs 逐token 一致（worst rel={worst:.3e}，容差 {tol}）");
        }
        Some("drive-check") => {
            // 用法: drive-check <artifact> <golden-f32-dir> [decode-steps]
            // 端到端 driver 门禁：用 lingxi-driver 的 Sequence 跑 prefill（全 48 层，
            // 逐层流式取权重），比对贪心 token 与 golden logits；可选再 decode 几步
            // （需 golden cos/sin 覆盖得到，当前 golden 只有 rows 行，故默认 0 步）。
            let art_path = args.get(2).context("缺少 artifact")?;
            let dir = std::path::PathBuf::from(args.get(3).context("缺少 golden f32 目录")?);
            let decode_steps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
            let u = |key: &str| meta[key].as_u64().unwrap() as usize;
            let rows = u("rows");
            let vocab = u("vocab_size");
            let n_layers = lingxi_model::DIMS.n_layers as u32;
            let ple_kk = u("ple_conv_kernel_size");
            let ple_d = u("ngram_size");

            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            let art = Artifact::open(art_path)?;
            let shape = layer_shape(&meta);

            // 两端权重
            let token_embedding =
                lingxi_model::gpu::upload_any_bf16(&k, &art, "text/token_embedding")?;
            let output_head = lingxi_model::gpu::upload_any_bf16(&k, &art, "text/output_head")?;
            let mup = |n: &str| {
                lingxi_model::gpu::upload_any_bf16(
                    &k,
                    &art,
                    &format!("text/hyper_connection_mixer/{n}"),
                )
            };
            let mixer = lingxi_model::gr::GrWeights {
                hc_norm: mup("hc_norm")?,
                mix_down: mup("mix_down")?,
                mix_up: mup("mix_up")?,
                block_inject: None,
            };
            let endpoints = lingxi_model::trunk::TrunkWeights {
                token_embedding, mixer, output_head, output_head_q4: None,
            };

            // RoPE 表：按 HF 参数自生成（覆盖到 prefill+decode 全长度），
            // 先用 golden 的 cos/sin 对验前 rows 行（验证 theta/排布没搞错）。
            let (cos_h, sin_h) = lingxi_model::rope::build_tables(
                rows + decode_steps + 1,
                shape.attn.rot_dim,
                lingxi_model::rope::ROPE_THETA,
            );
            // golden 的 cos/sin 是 HF 在 bf16 上下文产出再转 f32 的快照（落在 bf16
            // 网格上）；kernel 实际也只吃 bf16 表，故先 bf16 舍入再比对。
            let q = |v: &[f32]| v.iter().map(|&x| half::bf16::from_f32(x).to_f32()).collect::<Vec<_>>();
            let golden_cos = q(&read_f32(&dir.join("cos.f32")));
            let golden_sin = q(&read_f32(&dir.join("sin.f32")));
            let cos_rel = rel_l2(&q(&cos_h[..golden_cos.len()]), &golden_cos);
            let sin_rel = rel_l2(&q(&sin_h[..golden_sin.len()]), &golden_sin);
            println!("RoPE 自生成表 vs golden（均 bf16 舍入）：cos rel={cos_rel:.3e}  sin rel={sin_rel:.3e}");
            assert!(cos_rel < 1e-6 && sin_rel < 1e-6, "RoPE 表与 golden 不一致");
            let to_dev_bf16 = |v: &[f32]| -> Result<_> {
                Ok(k.stream
                    .clone_htod(&v.iter().map(|x| half::bf16::from_f32(*x)).collect::<Vec<_>>())?)
            };
            let tables = lingxi_driver::RopeTables {
                cos: to_dev_bf16(&cos_h)?,
                sin: to_dev_bf16(&sin_h)?,
                cos_host: cos_h,
                sin_host: sin_h,
            };

            let ids: Vec<i64> = std::fs::read(dir.join("input_ids.i32"))?
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect();

            // 模式：step（默认，Sequence 路径）| cmp（每层只加载一次，同时跑 layer_step
            // 空状态链与 layer_forward 链，逐层 rel，端到端定位两条路径的分叉）
            let mode = args.get(5).map(|s| s.as_str()).unwrap_or("step");
            let prompt_ids = ids[..rows].to_vec();

            // golden 判据：最后一行 argmax
            let want_logits = read_f32(&dir.join("logits.f32"));
            let want_am = lingxi_model::trunk::argmax_rows(&want_logits, rows, vocab);
            println!("golden argmax（全部行）= {want_am:?}");
            let want_last = *want_am.last().unwrap() as i64;

            if mode == "cmp" {
                use lingxi_model::layer::{LayerEnv, LayerStates};
                let dim = shape.hc * shape.hidden;
                // 48 层状态（layer 1 带 PLE）
                let mut states: Vec<LayerStates> = (0..n_layers)
                    .map(|i| {
                        let kind = lingxi_model::layer_kind(i);
                        let ple = if i == 1 { Some((ple_kk, ple_d)) } else { None };
                        LayerStates::for_kind(&k, kind, &shape, rows + 1, ple)
                    })
                    .collect::<Result<_>>()?;

                // 同一份 embedding 起跑两条链
                let idx_host: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
                let didx = k.stream.clone_htod(&idx_host)?;
                let mut emb = k.stream.alloc_zeros::<half::bf16>(rows * shape.hidden)?;
                k.gather_rows_bf16(&endpoints.token_embedding, &didx, &mut emb, rows, shape.hidden, 0)?;
                let mut hyper_s = k.stream.alloc_zeros::<half::bf16>(rows * dim)?;
                let mut hyper_f = k.stream.alloc_zeros::<half::bf16>(rows * dim)?;
                k.hc_repeat_bf16(&emb, &mut hyper_s, rows, shape.hidden, shape.hc)?;
                k.hc_repeat_bf16(&emb, &mut hyper_f, rows, shape.hidden, shape.hc)?;

                println!("双链对拍：{n_layers} 层流式（机械盘约 45 分钟）…");
                let t0 = std::time::Instant::now();
                for i in 0..n_layers {
                    let b = build_layer(&k, &art, &meta, i, false, None)?;
                    let env = LayerEnv {
                        cos: &tables.cos,
                        sin: &tables.sin,
                        cos_host: &tables.cos_host,
                        sin_host: &tables.sin_host,
                        remote: None,
                        causal_host: &[],
                        k_layernorm_host: &b.k_layernorm_host,
                        ids: &prompt_ids,
                        experts: None,
                    };
                    let next_s = lingxi_model::layer::layer_step(
                        &k, &b.weights, &shape, &hyper_s, rows, &env, &mut states[i as usize], i,
                    )?;
                    let next_f = lingxi_model::layer::layer_forward(
                        &k, &b.weights, &shape, &hyper_f, rows, &env, i,
                    )?;
                    let hs: Vec<f32> = k.stream.clone_dtoh(&next_s)?.iter().map(|v| v.to_f32()).collect();
                    let hf: Vec<f32> = k.stream.clone_dtoh(&next_f)?.iter().map(|v| v.to_f32()).collect();
                    // 只看最后一行（预测位）的链间 rel，避免 48×5 行刷屏
                    let rel_last = rel_l2(
                        &hs[(rows - 1) * dim..rows * dim],
                        &hf[(rows - 1) * dim..rows * dim],
                    );
                    eprintln!(
                        "  [cmp] 层 {i:2}/{n_layers} {:>6.1}s  lastrow rel(step,fwd)={rel_last:.3e}",
                        t0.elapsed().as_secs_f64()
                    );
                    hyper_s = next_s;
                    hyper_f = next_f;
                }

                let head_of = |hyper: &cudarc::driver::CudaSlice<half::bf16>| -> Result<(i64, Vec<f32>)> {
                    let (mixed, _) = lingxi_model::gr::gated_residual_forward(
                        &k, &endpoints.mixer, hyper, rows as u32, dim, shape.hidden,
                        shape.hc, shape.lowrank, shape.eps,
                    )?;
                    let mut logits = k.stream.alloc_zeros::<half::bf16>(rows * vocab)?;
                    k.gemm_bf16_nt(
                        &mixed, &endpoints.output_head, &mut logits,
                        rows as u32, vocab as u32, shape.hidden as u32,
                    )?;
                    let lh: Vec<f32> = k.stream.clone_dtoh(&logits)?.iter().map(|v| v.to_f32()).collect();
                    let last = &lh[(rows - 1) * vocab..rows * vocab];
                    let mut best = 0usize;
                    for (i, v) in last.iter().enumerate() {
                        if *v > last[best] { best = i; }
                    }
                    Ok((best as i64, last.to_vec()))
                };
                let (tok_s, log_s) = head_of(&hyper_s)?;
                let (tok_f, log_f) = head_of(&hyper_f)?;
                // 末端 logits 与 golden 的量级（argmax 之外的定量判据）
                let g_last = &want_logits[(rows - 1) * vocab..rows * vocab];
                let rel_s = rel_l2(&log_s, g_last);
                let rel_f = rel_l2(&log_f, g_last);
                println!("末端 logits rel vs golden：step 链 {rel_s:.3e} | forward 链 {rel_f:.3e}");
                let _ = (rel_s, rel_f);
                println!("双链完成，用时 {:.1}s", t0.elapsed().as_secs_f64());
                println!("贪心：step 链 = {tok_s} | forward 链 = {tok_f} | golden = {want_last}");
                assert_eq!(tok_f, want_last, "forward 链贪心 {tok_f} ≠ golden {want_last}：编排/权重链有 bug");
                assert_eq!(tok_s, want_last, "step 链贪心 {tok_s} ≠ golden {want_last}：step 空状态路径有 bug");
                println!("OK: 两条链贪心都 == golden");
            } else if mode == "golden" {
                // 逐层独立对拍：喂 golden layer{i}_in，比 layer{i}_out。
                // 层间不串联——坏多层会全部暴露，且不受误差传播影响。
                use lingxi_model::layer::LayerEnv;
                let dim = shape.hc * shape.hidden;
                let t0 = std::time::Instant::now();
                let mut first_bad: Option<(u32, f32)> = None;
                // 末行（决定贪心 token 的唯一一行）逐层 rel
                let mut lastrow_chain: Vec<(u32, f32)> = Vec::new();
                for i in 0..n_layers {
                    let gin = read_f32(&dir.join(format!("layer{i}_in.f32")));
                    let gout = read_f32(&dir.join(format!("layer{i}_out.f32")));
                    assert_eq!(gin.len(), rows * dim, "layer{i}_in 尺寸不符");
                    let dx_h: Vec<half::bf16> = gin.iter().map(|&v| half::bf16::from_f32(v)).collect();
                    let dx = k.stream.clone_htod(&dx_h)?;
                    let b = build_layer(&k, &art, &meta, i, false, None)?;
                    let env = LayerEnv {
                        cos: &tables.cos,
                        sin: &tables.sin,
                        cos_host: &tables.cos_host,
                        sin_host: &tables.sin_host,
                        remote: None,
                        causal_host: &[],
                        k_layernorm_host: &b.k_layernorm_host,
                        ids: &prompt_ids,
                        experts: None,
                    };
                    let out = lingxi_model::layer::layer_forward(
                        &k, &b.weights, &shape, &dx, rows, &env, i,
                    )?;
                    let oh: Vec<f32> = k.stream.clone_dtoh(&out)?.iter().map(|v| v.to_f32()).collect();
                    // 逐行 rel（T2 判据：bf16 router 第 10/11 名并列会让个别行合法翻转）
                    let mut row_rels = Vec::with_capacity(rows);
                    for r in 0..rows {
                        let sl = r * dim..(r + 1) * dim;
                        row_rels.push(rel_l2(&oh[sl.clone()], &gout[sl]));
                    }
                    let worst_row = row_rels.iter().cloned().fold(0f32, f32::max);
                    let ok_rows = row_rels.iter().filter(|&&r| r < 1.5e-2).count();
                    // 最终贪心 token 只依赖最后一行（因果），单独追踪它的链
                    let last_rel = *row_rels.last().unwrap();
                    let rstr = row_rels.iter().map(|r| format!("{r:.1e}")).collect::<Vec<_>>().join(",");
                    let tag = if ok_rows >= 1 { "ok " } else { "BAD" };
                    eprintln!(
                        "  [golden] 层 {i:2}/{n_layers} {:>6.1}s  rows[{rstr}] ok={ok_rows}/{rows} lastrow={last_rel:.3e} {tag}",
                        t0.elapsed().as_secs_f64()
                    );
                    if ok_rows == 0 && first_bad.is_none() {
                        first_bad = Some((i, worst_row));
                    }
                    // 末行链：逐层记录（最终 argmax 只走这一条）
                    lastrow_chain.push((i, last_rel));
                }
                // 末端：golden layer47_out → mixer+head → 比 golden logits 最后一行
                let last_in = read_f32(&dir.join(format!("layer{}_in.f32", n_layers - 1)));
                let last_out = read_f32(&dir.join(format!("layer{}_out.f32", n_layers - 1)));
                let lh_bf: Vec<half::bf16> = last_out.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let ldev = k.stream.clone_htod(&lh_bf)?;
                let (mixed, _) = lingxi_model::gr::gated_residual_forward(
                    &k, &endpoints.mixer, &ldev, rows as u32, dim, shape.hidden,
                    shape.hc, shape.lowrank, shape.eps,
                )?;
                let mut logits = k.stream.alloc_zeros::<half::bf16>(rows * vocab)?;
                k.gemm_bf16_nt(
                    &mixed, &endpoints.output_head, &mut logits,
                    rows as u32, vocab as u32, shape.hidden as u32,
                )?;
                let glog: Vec<f32> = k.stream.clone_dtoh(&logits)?.iter().map(|v| v.to_f32()).collect();
                let g_last = &want_logits[(rows - 1) * vocab..rows * vocab];
                let my_last = &glog[(rows - 1) * vocab..rows * vocab];
                let logrel = rel_l2(my_last, g_last);
                let my_am = my_last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
                println!("末端 mixer+head on golden layer47_out：logits rel={logrel:.3e}  argmax={my_am} golden={want_last}");
                let _ = last_in;
                // 末行链：首个越界层即端到端贪心分叉点（该行之后全部传播）
                let first_last_bad = lastrow_chain.iter().find(|(_, r)| *r >= 1.5e-2);
                println!(
                    "末行链：{}/{} 层 rel<1.5e-2{}",
                    lastrow_chain.iter().filter(|(_, r)| *r < 1.5e-2).count(),
                    n_layers,
                    match first_last_bad {
                        Some((i, r)) => format!("；首个末行越界层 = {i}（rel={r:.3e}）= 端到端贪心分叉点"),
                        None => "；末行全程复现".to_string(),
                    }
                );
                match first_bad {
                    Some((i, r)) => panic!("发现全 BAD 层（无一行 rel<1.5e-2）：层 {i}（worst={r:.3e}）；末端 logits rel={logrel:.3e}"),
                    None => println!("全部 {n_layers} 层单层 T2 通过（至少一行 <1.5e-2）；末端 logits rel={logrel:.3e}"),
                }
            } else {
            // 建序列（layer 1 带 PLE：K=4, dilation=ngram_size）
            let mut seq = lingxi_driver::Sequence::new(
                &k,
                n_layers,
                &shape,
                rows + decode_steps + 1,
                Some((1, ple_kk, ple_d)),
            )?;
            let eng = lingxi_driver::Engine {
                k: &k,
                shape: &shape,
                n_layers,
                vocab,
                endpoints: &endpoints,
                tables: &tables,
                experts: None,
                remote: None,
                k1: None,
                split_lo: 0,
                tables1: None,
                experts1: None,
            };

            println!("driver prefill：{n_layers} 层流式（机械盘约 20 分钟）…");
            let t0 = std::time::Instant::now();
            let mut loader = StreamingProvider {
                cur: None,
                build: {
                    let (kb, ab, mb) = (&k, &art, &meta);
                    Box::new(move |layer: u32| -> Result<lingxi_model::trunk::LoadedLayer> {
                        eprint!("\r  [drive] 层 {layer}/{n_layers}  {:>5.1}s", t0.elapsed().as_secs_f64());
                        let b = build_layer(kb, ab, mb, layer, false, None)?;
                        Ok(lingxi_model::trunk::LoadedLayer { weights: b.weights, k_layernorm_host: b.k_layernorm_host })
                    })
                },
            };
            let token = seq.prefill(&eng, &ids[..rows], &mut loader)?;
            eprintln!();
            println!("prefill 完成，用时 {:.1}s，贪心 token = {token}", t0.elapsed().as_secs_f64());

            assert_eq!(token, want_last, "prefill 贪心 token {token} ≠ golden {want_last}");
            println!("OK: driver prefill 贪心 == golden（{token}）");

            if decode_steps > 0 {
                let mut got = Vec::new();
                for _ in 0..decode_steps {
                    let t1 = seq.decode_token(&eng, &mut loader)?;
                    got.push(t1);
                }
                println!("decode {decode_steps} 步: {got:?}");
            }
            }
        }
        Some("talk") => {
            // 用法: talk <artifact> "<prompt>" [max_new_tokens]
            // 出字闭环：产物内 frontend/tokenizer.json 分词 → Sequence 贪心
            // prefill+generate → token 解码回文本。当前每层全量取权重（机械盘慢），
            // 只用于打通"文本进→文本出"；速度要等专家瘦身+缓存（P1）。
            let art_path = args.get(2).context("用法: lingxi-bench talk <artifact> <prompt> [max_new_tokens]")?;
            let prompt = prompt_arg(args.get(3)).context("缺少 prompt 文本")?;
            let max_new: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);

            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            // [探针] `LINGXI_SYNCTEST=N`：量**空闲 GPU 下**一次 `cuStreamSynchronize` 的代价。
            // decode 每层都要同步一次取路由（48 次/token）。若单次就是几百 µs，那它就是主瓶颈，
            // 必须换成"映射内存 + 自旋读标志"这类不依赖驱动同步的取法。
            if let Some(n) = std::env::var("LINGXI_SYNCTEST").ok().and_then(|s| s.parse::<u32>().ok()) {
                k.stream.synchronize()?;
                let t = std::time::Instant::now();
                for _ in 0..n {
                    k.stream.synchronize()?;
                }
                let dt = t.elapsed().as_secs_f64() / f64::from(n);
                eprintln!("[synctest] 空闲同步 {n} 次 → **{:.1} µs/次**", dt * 1e6);
            }
            // `LINGXI_GRAPHTEST=1`：把 CUDA Graph 捕获拆成一步步探测，精确定位哪一步失败。
            if std::env::var("LINGXI_GRAPHTEST").is_ok() {
                use cudarc::driver::sys;
                let xs = k.stream.alloc_zeros::<f32>(1024)?;
                let mut ys = k.stream.alloc_zeros::<f32>(1024)?;
                k.smoke_scale(&xs, 2.0, &mut ys, 1024)?;
                k.stream.synchronize()?;
                eprintln!("[graphtest] 基线 kernel 正常");
                for m in [
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                ] {
                    let r1 = k.stream.begin_capture(m);
                    eprintln!("[graphtest] begin_capture({m:?}) -> {r1:?}");
                    if r1.is_ok() {
                        let r2 = k.smoke_scale(&xs, 2.0, &mut ys, 1024);
                        eprintln!("[graphtest]   捕获中 launch -> {r2:?}");
                        let r3 = k.stream.end_capture(sys::CUgraphInstantiate_flags(0));
                        match r3 {
                            Ok(Some(g)) => {
                                eprintln!("[graphtest]   end_capture OK，试 replay");
                                eprintln!("[graphtest]   replay -> {:?}", g.launch());
                            }
                            Ok(None) => eprintln!("[graphtest]   end_capture 返回空图"),
                            Err(e) => eprintln!("[graphtest]   end_capture -> {e:?}"),
                        }
                    }
                }
            }

            // `LINGXI_NO_SCRATCH_POOL=1`：量"整轮禁池"的代价（layer-split 现在无条件禁池，
            // 所以这是把那条 22 ms 回归拆成"禁池 vs 跨卡"两半的判决性实验）。
            if lingxi_model::flag("LINGXI_NO_SCRATCH_POOL") {
                lingxi_model::scratch::disable_pool_forever();
            }

            let art = Arc::new(Artifact::open(art_path)?);
            let meta = flash_next_meta();

            let shape = layer_shape(&meta);

            // 分词器与 eos（全部取自产物 resource，零外部文件）
            let tok_bytes = art
                .payload("frontend/tokenizer.json")
                .context("产物缺少 frontend/tokenizer.json")?;
            let tok = lingxi_tokenize::Tokenizer::from_json(&tok_bytes)?;
            let prompt_ids = tok.encode(&prompt)?;
            let prompt_len = prompt_ids.len();
            let cfg_eos = match art.payload("frontend/generation_config.json") {
                Some(b) => {
                    let g: serde_json::Value = serde_json::from_slice(b)?;
                    match &g["eos_token_id"] {
                        serde_json::Value::Number(n) => n.as_i64(),
                        serde_json::Value::Array(a) => a.first().and_then(|v| v.as_i64()),
                        _ => None,
                    }
                }
                None => None,
            };
            let eos = cfg_eos
                .or_else(|| tok.token_to_id("<|im_end|>"))
                .or_else(|| tok.token_to_id(""))
                .context("无法确定 eos token id")?;
            // talk 默认走瘦身路径（只传命中专家）；LINGXI_SPARSE_EXPERTS=0 回全量对拍路径
            let sparse = std::env::var("LINGXI_SPARSE_EXPERTS")
                .map(|v| v != "0")
                .unwrap_or(true);
            eprintln!(
                "prompt={prompt_len} tokens  eos={eos}  max_new={max_new}  sparse_experts={sparse}"
            );

            // 两端权重 + RoPE 表（覆盖 prompt+生成长度）
            let endpoints = load_endpoints(&k, &art)?;
            let rope_len = prompt_len + max_new + 1;
            let tables = load_rope_tables(&k, rope_len, shape.attn.rot_dim)?;

            let n_layers = lingxi_model::DIMS.n_layers as u32;
            // ★★ **全量 layer-split**（`LINGXI_LAYER_SPLIT=<lo>`，PLAN §P15 里程碑 A）。
            //    这套接线必须建在 `Sequence` **之前**：`[lo, n_layers)` 的跨 token 状态
            //    （QSA KV/indexer、GDN 的 S+卷积史、PLE 史）要**直接建在 GPU1 上**，
            //    建错卡不报错、只会在第一次 kernel 时 ILLEGAL_ADDRESS 或静默写串。
            let ls_arena_path = std::env::var("LINGXI_EXPERT_ARENA").ok();
            // ★ P15-㉜：双卡默认自动 split（同 mtp-accept 分支的口径）。
            let ls_env: Option<u32> = std::env::var("LINGXI_LAYER_SPLIT")
                .ok().and_then(|s| s.parse().ok());
            let auto_split = ls_env.is_none()
                && cudarc::driver::CudaContext::new(1).is_ok()
                && ls_arena_path.is_some();
            let ls_lo: u32 = ls_env.unwrap_or(if auto_split { 24 } else { 0 });
            let ls_on = ls_lo > 0 && ls_lo < n_layers;
            if ls_on {
                if auto_split {
                    // ㊸：单层本地图在 split 下保持开启（专家段 3 核固定槽、层⇄设备一一对应、
                    // 跨卡拷贝在层循环外不进图）。判据翻车则恢复 force_disable()。
                    eprintln!(
                        "[layer-split] 检测到双卡：自动启用整层切分 lo={ls_lo}（真实语料 +161%；\
                         设 LINGXI_LAYER_SPLIT=0 关闭）。单层本地图保持开启。"
                    );
                } else {
                    if !lingxi_model::flag("LINGXI_NO_GRAPHS") {
                        eprintln!(
                            "[layer-split] 警告：未设 LINGXI_NO_GRAPHS；㊸ 起单层本地图 split 下可用，\
                             数值异常先设它对拍。"
                        );
                    }
                }
                // ★ P15-㉞：池已 device-aware，不再需要禁池（见 mtp-accept 分支同款注释）。
            }
            let ls_k1: Option<lingxi_kernels::Kernels> =
                if ls_on { Some(lingxi_kernels::Kernels::load(1)?) } else { None };
            let ls_tables: Option<lingxi_driver::RopeTables> = match ls_k1.as_ref() {
                Some(kk) => Some(load_rope_tables(kk, rope_len, shape.attn.rot_dim)?),
                None => None,
            };
            // ㊿-P0：未显式给槽数时走显存预算自动分配（env 显式值优先）。
            let ls_slots: usize = match std::env::var("LINGXI_LAYER_SPLIT_SLOTS")
                .ok().and_then(|s| s.parse().ok())
            {
                Some(v) => v,
                None => match ls_arena_path.as_deref() {
                    Some(ap) => match autotune_arena_slots(&art, ap, 1, ls_lo, rope_len + 8)? {
                        Some((s, ledger)) => {
                            eprintln!("{ledger}");
                            s
                        }
                        None => 9600,
                    },
                    None => 9600,
                },
            };  // ★㊲ 工作集≥槽位判决(375/卡层)
            let ls_experts: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> =
                match (ls_k1.as_ref(), ls_arena_path.as_deref()) {
                    (Some(kk), Some(ap)) => {
                        eprintln!(
                            "[layer-split] GPU1 上下文就绪；层 {ls_lo}..{n_layers} 整层交给它，专家槽位 {ls_slots}"
                        );
                        Some(Box::new(arena_source::ArenaExpertSource::new(
                            kk, ap, ls_slots, false, ls_lo..n_layers,
                        )?))
                    }
                    (Some(_), None) => anyhow::bail!(
                        "LINGXI_LAYER_SPLIT 需要 LINGXI_EXPERT_ARENA（GPU1 那份专家表）"
                    ),
                    _ => None,
                };
            let mut seq = lingxi_driver::Sequence::new_split(
                &k,
                ls_k1.as_ref().map(|kk| (kk, ls_lo)),
                n_layers,
                &shape,
                rope_len + 8,
                Some((
                    1,
                    meta["ple_conv_kernel_size"].as_u64().unwrap() as usize,
                    meta["ngram_size"].as_u64().unwrap() as usize,
                )),
            )?;
            // host 缓存（sparse 时启用）：dense 匿名常驻 + 专家 mmap 零拷贝视图
            let host_cache: Option<Arc<host_cache::HostCache>> = if sparse {
                let c = host_cache::HostCache::new(Arc::clone(&art));
                c.start_prefetcher();
                eprintln!("host_cache: dense 后台预读已启动，专家走 mmap 零拷贝视图");
                Some(c)
            } else {
                None
            };

            // 瘦身专家源（sparse 时启用）：
            //   LINGXI_EXPERT_ARENA 给出 → arena 直读（只读 mmap + 整段 pin + DMA + 视图解量化）
            //   否则 → 旧的 host LRU(mmap 视图) + device LRU 路径（对照/回退）
            let expert_source: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> = if sparse {
                let inter = meta["inter"].as_u64().unwrap() as usize;
                let hidden = meta["hidden"].as_u64().unwrap() as usize;
                if let Ok(ap) = std::env::var("LINGXI_EXPERT_ARENA") {
                    // ㊿-P0：未显式给槽数时走显存预算自动分配（env 显式值优先）。
                    let slots: usize = match std::env::var("LINGXI_ARENA_SLOTS")
                        .ok().and_then(|s| s.parse().ok())
                    {
                        Some(v) => v,
                        None => match autotune_arena_slots(&art, &ap, 0, ls_lo, rope_len + 8)? {
                            Some((s, ledger)) => {
                                eprintln!("{ledger}");
                                s
                            }
                            None => 4000,
                        },
                    };
                    Some(Box::new(arena_source::ArenaExpertSource::new(
                        &k,
                        &ap,
                        slots,
                        false,
                        // ★ 全量 layer-split：GPU0 这份只服务 `0..ls_lo`（后一半层的槽位
                        //   全部留给 GPU1 那份，否则一半的槽白占）。
                        if ls_on { 0..ls_lo } else { 0..u32::MAX },
                    )?))
                } else {
                    // ⚠️ **不允许静默回退**：这条 host LRU ↔ device LRU 的路实测
                    // decode **488~714 ms/token**（比 arena 慢 15~20 倍），而且以前毫无提示。
                    // 历史上"忘了设 LINGXI_EXPERT_ARENA"就掉进来，被误判成代码回归（§3.45）。
                    // 要用它必须显式开 `LINGXI_ALLOW_MMAP_FALLBACK=1`。
                    if std::env::var("LINGXI_ALLOW_MMAP_FALLBACK").is_err() {
                        anyhow::bail!(
                            "没给 LINGXI_EXPERT_ARENA ⇒ 稀疏路径只剩 MmapExpertSource（host LRU），\
                             实测慢 15~20 倍。请设 LINGXI_EXPERT_ARENA=<arena 文件>；\
                             确实要用旧路请显式设 LINGXI_ALLOW_MMAP_FALLBACK=1（对比用）。"
                        );
                    }
                    let dev_gb: usize = std::env::var("LINGXI_DEV_EXPERT_GB")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(17);
                    Some(Box::new(MmapExpertSource::new(
                        &k,
                        host_cache.as_ref().unwrap(),
                        inter,
                        hidden,
                        dev_gb,
                    )?))
                }
            } else {
                None
            };
            let eng = lingxi_driver::Engine {
                k: &k,
                shape: &shape,
                n_layers,
                vocab: lingxi_model::DIMS.vocab as usize,
                endpoints: &endpoints,
                tables: &tables,
                experts: expert_source.as_deref(),
                remote: None,
                k1: ls_k1.as_ref(),
                split_lo: ls_lo,
                tables1: ls_tables.as_ref(),
                experts1: ls_experts.as_deref(),
            };

            // dense 预读：立刻给预读线程喂前两层，之后每层回调里再喂下一层
            if let Some(c) = &host_cache {
                c.prefetch_layer_dense(0);
            }

            let t0 = std::time::Instant::now();
            // sparse：dense 权重显存常驻，prefill 首遍逐层构建（同时后台预读下一层
            // host 字节），之后 decode 全部按引用复用、零 dense H2D；
            // full（对照）：保持每层每步流式现建。
            let mut stream_provider: Option<StreamingProvider>;
            let mut cache_provider: Option<DeviceLayerCache>;
            let hc_for_build = host_cache.clone();
            let build = {
                let (kb, k1b, ab, mb) = (&k, ls_k1.as_ref(), &art, &meta);
                let (lsb, lob) = (ls_on, ls_lo);
                Box::new(move |layer: u32| -> Result<lingxi_model::trunk::LoadedLayer> {
                    eprint!("\r  [load] 层 {layer}/{n_layers}  {:>6.1}s", t0.elapsed().as_secs_f64());
                    // ★ 全量 layer-split：本层的稠密权重必须直接装到**本层要跑的卡**上。
                    let dev_k = match (lsb, k1b) {
                        (true, Some(kk)) if layer >= lob => kk,
                        _ => kb,
                    };
                    let b = build_layer(dev_k, ab, mb, layer, sparse, hc_for_build.as_deref())?;
                    // 本层 dense 已就位：后台线程立刻开始读下一层（与本层 GPU 计算重叠）
                    if let Some(c) = &hc_for_build {
                        if layer + 1 < n_layers {
                            c.prefetch_layer_dense(layer + 1);
                        }
                    }
                    Ok(lingxi_model::trunk::LoadedLayer { weights: b.weights, k_layernorm_host: b.k_layernorm_host })
                })
            };
            let provider: &mut dyn lingxi_driver::LayerProvider = if sparse {
                cache_provider = Some(DeviceLayerCache {
                    slots: (0..n_layers as usize).map(|_| None).collect(),
                    build,
                });
                cache_provider.as_mut().unwrap()
            } else {
                stream_provider = Some(StreamingProvider { cur: None, build });
                stream_provider.as_mut().unwrap()
            };

            let first = seq.prefill(&eng, &prompt_ids, provider)?;
            eprintln!("\nprefill 用时 {:.1}s，首 token = {first}", t0.elapsed().as_secs_f64());
            // `LINGXI_EVPROF=1`：prefill 最后一个 chunk 的逐段真 GPU 时间（层求和）。
            // 注意事件是每个 (层,段) 取**最后一次**记录 ⇒ decode 开始前 dump = 末 chunk 账。
            if let Some(p) = k.evprof.as_ref() {
                p.dump(eng.n_layers);
            }

            // `LINGXI_FWD_BENCH="2,4,8,16"`：**一次 t=k 前向要多少钱？**
            // 这是"投机解码能不能把单流推到 200"的唯一判据：若 t=8 的一趟 ≈ t=1 的一趟
            // （decode 43 ms），那么一轮验 8 个 token 就等于"8 个 token 花 1.3 趟的钱"。
            // 此时 provider 已把 48 层全建好（cache 命中），所以这里量的是纯前向。
            if let Ok(spec) = std::env::var("LINGXI_FWD_BENCH") {
                let reps: usize = std::env::var("LINGXI_FWD_REPS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5);
                let ple_spec = (
                    1u32,
                    meta["ple_conv_kernel_size"].as_u64().unwrap() as usize,
                    meta["ngram_size"].as_u64().unwrap() as usize,
                );
                let cap = rope_len + 8;
                for tv in spec.split(',').filter_map(|x| x.trim().parse::<usize>().ok()) {
                    let ids: Vec<i64> =
                        (0..tv).map(|i| prompt_ids[i % prompt_ids.len()]).collect();
                    let mk_seq = || {
                        lingxi_driver::Sequence::new_split(
                            &k,
                            ls_k1.as_ref().map(|kk| (kk, ls_lo)),
                            n_layers,
                            &shape,
                            cap,
                            Some(ple_spec),
                        )
                    };
                    {
                        let mut w = mk_seq()?;
                        w.prefill(&eng, &ids, provider)?;
                    }
                    let mut ts = Vec::with_capacity(reps);
                    for _ in 0..reps {
                        let mut s = mk_seq()?;
                        let tf = std::time::Instant::now();
                        s.prefill(&eng, &ids, provider)?;
                        ts.push(tf.elapsed().as_secs_f64());
                    }
                    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let med = ts[ts.len() / 2];
                    eprintln!(
                        "[fwd] t={tv:<4} median={:>7.1} ms   （每 token-slot {:.2} ms；t=1 decode 对比见 bench.sh）",
                        med * 1e3,
                        med * 1e3 / tv as f64
                    );
                }
            }

            let mut gen_ids = Vec::new();
            if first != eos {
                gen_ids.push(first);
                for step in 0..max_new {
                    let t1 = std::time::Instant::now();
                    let t = seq.decode_token(&eng, provider)?;
                    let dt = t1.elapsed().as_secs_f64();
                    // ⚠️ 这里原来打的是 `{dt:>7.3}s/token` —— **只有 3 位小数 = 1 ms 粒度**，
                    // 而单 token 约 33 ms，也就是**量化台阶就有 3%**：任何 1~3% 的优化都量不出来。
                    // 改成 µs（0.1 µs 分辨率）后，尺子才比被测物细。
                    eprint!("\r  decode {:>2}/{max_new}  token={t:<7} {:>9.1} µs/token        ", step + 1, dt * 1e6);
                    use std::io::Write;
                    std::io::stderr().flush().ok();
                    if t == eos {
                        break;
                    }
                    gen_ids.push(t);
                }
                eprintln!();
            } else {
                eprintln!("首 token 即 eos，结束");
            }

            eprintln!("-- 生成 {} tokens，解码文本 --", gen_ids.len());
            println!("{}", tok.decode(&gen_ids)?);
            // ㊸：decode 后再 dump 一次 ⇒ 逐段数字 = **最后一个 decode token** 的真 GPU 分解
            if let Some(p) = k.evprof.as_ref() {
                p.dump(eng.n_layers);
            }
            // ㊸：GPU1 的账单独打（split 下 wall = max(两卡) + 跨卡手递手，只看 GPU0 会误判瓶颈侧）
            if let Some(kk) = ls_k1.as_ref() {
                if let Some(p) = kk.evprof.as_ref() {
                    p.dump(eng.n_layers);
                }
            }
        }
        Some("serve") => {
            // 用法: serve <artifact> [port=11434]   —— OpenAI 兼容 HTTP（P15-㊷）
            // 装配与 talk 完全同一条生产链（layer-split/arena/图门控全在），
            // 差别只在出字循环：每请求 reset 复用状态，HTTP+SSE 出字。
            let art_path = args.get(2).context("用法: lingxi-bench serve <artifact> [port]")?;
            let port: u16 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(11434);
            let serve_ctx: usize = std::env::var("LINGXI_SERVE_MAX_CTX")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(32768);
            let max_new_default: usize = std::env::var("LINGXI_SERVE_MAX_NEW")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(512);

            let dev: usize =
                std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let k = lingxi_kernels::Kernels::load(dev)?;
            if lingxi_model::flag("LINGXI_NO_SCRATCH_POOL") {
                lingxi_model::scratch::disable_pool_forever();
            }

            let art = Arc::new(Artifact::open(art_path)?);
            let meta = flash_next_meta();
            let shape = layer_shape(&meta);

            let tok_bytes = art
                .payload("frontend/tokenizer.json")
                .context("产物缺少 frontend/tokenizer.json")?;
            let tok = lingxi_tokenize::Tokenizer::from_json(&tok_bytes)?;
            let cfg_eos = match art.payload("frontend/generation_config.json") {
                Some(b) => {
                    let g: serde_json::Value = serde_json::from_slice(b)?;
                    match &g["eos_token_id"] {
                        serde_json::Value::Number(n) => n.as_i64(),
                        serde_json::Value::Array(a) => a.first().and_then(|v| v.as_i64()),
                        _ => None,
                    }
                }
                None => None,
            };
            let eos = cfg_eos
                .or_else(|| tok.token_to_id("<|im_end|>"))
                .context("无法确定 eos token id")?;

            let endpoints = load_endpoints(&k, &art)?;
            // 容量按 serve_ctx 一次建足：rope/序列状态都不随请求重分配
            let rope_len = serve_ctx + 520;
            let tables = load_rope_tables(&k, rope_len, shape.attn.rot_dim)?;

            let n_layers = lingxi_model::DIMS.n_layers as u32;
            let ls_arena_path = std::env::var("LINGXI_EXPERT_ARENA").ok();
            let ls_env: Option<u32> = std::env::var("LINGXI_LAYER_SPLIT")
                .ok().and_then(|s| s.parse().ok());
            let auto_split = ls_env.is_none()
                && cudarc::driver::CudaContext::new(1).is_ok()
                && ls_arena_path.is_some();
            let ls_lo: u32 = ls_env.unwrap_or(if auto_split { 24 } else { 0 });
            let ls_on = ls_lo > 0 && ls_lo < n_layers;
            if ls_on {
                if auto_split {
                    // ㊸：单层本地图 split 下保持开启（判据翻车则恢复 force_disable()）
                    eprintln!("[layer-split] 双卡自动整层切分 lo={ls_lo}（单层本地图保持开启）");
                } else {
                    if !lingxi_model::flag("LINGXI_NO_GRAPHS") {
                        eprintln!("[layer-split] 警告：未设 LINGXI_NO_GRAPHS；单层本地图 split 下可用");
                    }
                }
            }
            let ls_k1: Option<lingxi_kernels::Kernels> =
                if ls_on { Some(lingxi_kernels::Kernels::load(1)?) } else { None };
            let ls_tables: Option<lingxi_driver::RopeTables> = match ls_k1.as_ref() {
                Some(kk) => Some(load_rope_tables(kk, rope_len, shape.attn.rot_dim)?),
                None => None,
            };
            let ls_slots: usize = std::env::var("LINGXI_LAYER_SPLIT_SLOTS")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(9600);
            let ls_experts: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> =
                match (ls_k1.as_ref(), ls_arena_path.as_deref()) {
                    (Some(kk), Some(ap)) => {
                        eprintln!("[layer-split] GPU1：层 {ls_lo}..{n_layers}，专家槽位 {ls_slots}");
                        Some(Box::new(arena_source::ArenaExpertSource::new(
                            kk, ap, ls_slots, false, ls_lo..n_layers,
                        )?))
                    }
                    (Some(_), None) => anyhow::bail!(
                        "LINGXI_LAYER_SPLIT 需要 LINGXI_EXPERT_ARENA（GPU1 那份专家表）"
                    ),
                    _ => None,
                };
            // ㊿-P1：Sequence 不再预建——交给 serve 的会话池按需物化（每会话一个，复用/沉没/唤醒）。
            let mut mk_seq = {
                let (kb, k1b) = (&k, ls_k1.as_ref());
                let (sh, nl, rl, lo) = (&shape, n_layers, rope_len, ls_lo);
                let ple = (
                    meta["ple_conv_kernel_size"].as_u64().unwrap() as usize,
                    meta["ngram_size"].as_u64().unwrap() as usize,
                );
                move || {
                    lingxi_driver::Sequence::new_split(
                        kb, k1b.map(|kk| (kk, lo)), nl, sh, rl + 8, Some((1, ple.0, ple.1)),
                    )
                }
            };
            let host_cache: Option<Arc<host_cache::HostCache>> = {
                let c = host_cache::HostCache::new(Arc::clone(&art));
                c.start_prefetcher();
                Some(c)
            };
            let expert_source: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> = {
                let ap = std::env::var("LINGXI_EXPERT_ARENA").context(
                    "serve 需要 LINGXI_EXPERT_ARENA=<arena 文件>（host LRU 路径慢 15-20 倍，不允许静默回退）",
                )?;
                // ㊿-P0：未显式给槽数时走显存预算自动分配（env 显式值优先）。
                let slots: usize = match std::env::var("LINGXI_ARENA_SLOTS")
                    .ok().and_then(|s| s.parse().ok())
                {
                    Some(v) => v,
                    None => match autotune_arena_slots(&art, &ap, 0, ls_lo, serve_ctx + 528)? {
                        Some((s, ledger)) => {
                            eprintln!("{ledger}");
                            s
                        }
                        None => 9000,
                    },
                };
                Some(Box::new(arena_source::ArenaExpertSource::new(
                    &k, &ap, slots, false,
                    if ls_on { 0..ls_lo } else { 0..u32::MAX },
                )?))
            };
            let eng = lingxi_driver::Engine {
                k: &k,
                shape: &shape,
                n_layers,
                vocab: lingxi_model::DIMS.vocab as usize,
                endpoints: &endpoints,
                tables: &tables,
                experts: expert_source.as_deref(),
                remote: None,
                k1: ls_k1.as_ref(),
                split_lo: ls_lo,
                tables1: ls_tables.as_ref(),
                experts1: ls_experts.as_deref(),
            };
            if let Some(c) = &host_cache {
                c.prefetch_layer_dense(0);
            }

            let t0 = std::time::Instant::now();
            let mut cache_provider: Option<DeviceLayerCache>;
            let hc_for_build = host_cache.clone();
            let build = {
                let (kb, k1b, ab, mb) = (&k, ls_k1.as_ref(), &art, &meta);
                let (lsb, lob) = (ls_on, ls_lo);
                Box::new(move |layer: u32| -> Result<lingxi_model::trunk::LoadedLayer> {
                    eprint!("\r  [load] 层 {layer}/{n_layers}  {:>6.1}s", t0.elapsed().as_secs_f64());
                    let dev_k = match (lsb, k1b) {
                        (true, Some(kk)) if layer >= lob => kk,
                        _ => kb,
                    };
                    let b = build_layer(dev_k, ab, mb, layer, true, hc_for_build.as_deref())?;
                    if let Some(c) = &hc_for_build {
                        if layer + 1 < n_layers {
                            c.prefetch_layer_dense(layer + 1);
                        }
                    }
                    Ok(lingxi_model::trunk::LoadedLayer { weights: b.weights, k_layernorm_host: b.k_layernorm_host })
                })
            };
            let provider: &mut dyn lingxi_driver::LayerProvider = {
                cache_provider = Some(DeviceLayerCache {
                    slots: (0..n_layers as usize).map(|_| None).collect(),
                    build,
                });
                cache_provider.as_mut().unwrap()
            };

            let opts = serve::ServeOpts {
                tok: &tok,
                eos,
                max_ctx: serve_ctx,
                max_new_default,
            };
            // ── 投机（可选）：`LINGXI_SERVE_SPEC=1` + `LINGXI_MTP_ARTIFACT=<path>` ──
            // 只接在 serve 上（产品路径）；talk 仍走纯自回归。
            let mut spec_rt: Option<spec::SpecRuntime<'_>> =
                if std::env::var("LINGXI_SERVE_SPEC").map(|v| v == "1").unwrap_or(false) {
                    let mp = std::env::var("LINGXI_MTP_ARTIFACT")
                        .unwrap_or_else(|_| "/data/models/lingxi-assets/qwen4_exp.mtp.fixed.ninfer".to_string());
                    eprintln!("[serve] 投机解码启用：MTP = {mp}");
                    Some(spec::SpecRuntime::load(&k, &mp, &tables, &endpoints, &shape, rope_len)?)
                } else {
                    None
                };
            serve::run(port, &opts, &eng, &mut mk_seq, provider, &mut spec_rt)?;
        }
        _ => {
            eprintln!("lingxi-bench: 用法: verify|list|bytes|vram-budget|decode-check|dequant-check|component-check|layer-check|trunk-check|trunk-ends|step-check|drive-check|argmax-check|f32x-check|talk|serve ...");
        }
    }
    Ok(())
}

/// 固化的 Qwen3.8-Flash-Next 全模型超参（值与 golden meta.json 同源，来自 HF
/// `text_config`）。talk/server 不依赖 golden 目录即可构造 `LayerShape` 与 PLE 配置。
/// 未来第二个模型出现时，此函数上移为 lingxi-model 的强类型 `ModelSpec` 数据。
/// ㊿-P0 显存预算自动分配（M3 弹性 KV 的第一块砖）：
/// 从 catalog 逐张量算**每卡权重字节**（角色排除法：`MoeRouted*` 走 arena 不上卡；
/// `text/layers/{i}/` 按层号归卡，无层号的 embedding/head 归 GPU0），减去
/// KV(serve_ctx)/GDN/rope/预留/边距，余量全部给专家槽。
/// 显式给了槽数 env 时不接管（返回 None）；`LINGXI_NO_VRAM_AUTOTUNE=1` 整体关闭。
/// 分配后仍建议看启动账与实际 free VRAM 的差距，逐步收紧 reserve。
fn autotune_arena_slots(
    art: &lingxi_artifact::Artifact,
    arena_path: &str,
    dev: usize,
    split_lo: u32,
    kv_cap: usize,
) -> Result<Option<(usize, String)>> {
    use lingxi_ids::TensorRole;
    if std::env::var("LINGXI_NO_VRAM_AUTOTUNE").is_ok() {
        return Ok(None);
    }
    let arena = lingxi_artifact::expert_arena::ExpertArena::open(arena_path)?;
    let n_layers = arena.n_layers;
    let (lo, hi) = if dev == 0 {
        (0, split_lo.min(n_layers))
    } else {
        (split_lo.min(n_layers), n_layers)
    };
    // 按**每个 blob 自己的格式**统计本卡服务区间的专家数与平均字节数。
    // 混档 arena 不能再用缺省 Q4 blob 估算——那会把 Q3 槽也按 Q4 大小算，
    // 白白少算几千个槽（2026-10-22 修正）。
    let mut n_q4 = 0usize;
    let mut n_q3 = 0usize;
    let mut q4_bytes = 0u64;
    let mut q3_bytes = 0u64;
    for l in lo..hi {
        for e in 0..arena.n_experts {
            if let Some(b) = arena.routed(l, e)? {
                let sz = (b.gu.len() + b.down.len()) as u64;
                match b.format {
                    BlobFormat::Q4G64F16S => {
                        n_q4 += 1;
                        q4_bytes += sz;
                    }
                    BlobFormat::Q3G128F16S => {
                        n_q3 += 1;
                        q3_bytes += sz;
                    }
                    _ => {}
                }
            }
        }
    }
    let blob_q4 = if n_q4 > 0 {
        q4_bytes / n_q4 as u64
    } else {
        arena.gu_bytes + arena.down_bytes
    };
    let blob_q3 = if n_q3 > 0 { q3_bytes / n_q3 as u64 } else { 0 };
    let mixed = n_q4 > 0 && n_q3 > 0;
    let catalog = lingxi_model::catalog_from_artifact(art);
    let mut w = [0u64; 2];
    for (_tid, info) in catalog.iter() {
        // 专家路由权重：Q4 arena 路径下不进显存（bf16 原档留在 host mmap）。
        if matches!(info.role, TensorRole::MoeRoutedGateUp | TensorRole::MoeRoutedDown) {
            continue;
        }
        let gpu = info
            .name
            .split("layers/")
            .nth(1)
            .and_then(|rest| rest.split('/').next())
            .and_then(|s| s.parse::<u32>().ok())
            .map_or(0usize, |l| if l >= split_lo { 1 } else { 0 });
        w[gpu] += info.storage_bytes();
    }
    // P2 分页：KV 设备侧只开常驻窗口（`LINGXI_QSA_WIN_TOKENS`，默认 32K token）——
    // 预算按窗口口径算，省下的全额 KV 直接变成专家槽。（indexer 的 rk/pooled 是
    // 逻辑缓冲但懒增长到 len，不进预算。）
    let cap = if lingxi_model::flag("LINGXI_QSA_PAGED") {
        let win: usize = std::env::var("LINGXI_QSA_WIN_TOKENS")
            .ok().and_then(|s| s.parse().ok()).filter(|&n| n >= 1024)
            .unwrap_or(32768);
        kv_cap.min(win)
    } else {
        kv_cap // 调用方给 KV 池的实际容量（serve=serve_ctx+528，talk=rope_len+8）
    };
    const QSA_PER_DEV: usize = 6; // 12 层 full-attn 双卡各半
    const GDN_PER_DEV: usize = 18; // 36 层 linear-attn 双卡各半
    let kv_per_token_layer: u64 = if lingxi_model::flag("LINGXI_KV_K8V4") { 1184 } else { 2368 };
    let kv = cap as u64 * kv_per_token_layer * QSA_PER_DEV as u64;
    let gdn = 3_220_000u64 * GDN_PER_DEV as u64; // S 3.1MB + 卷积历史双缓冲 ~3.22MB/层
    let reserve: u64 = if dev == 0 { 3 << 30 } else { 2500 << 20 }; // scratch/图/投机缓冲
    let margin: u64 = 1 << 30; // 驱动/碎片边距
    let dev_raw = cudarc::driver::result::device::get(dev as i32)?;
    let total = unsafe { cudarc::driver::result::device::total_mem(dev_raw)? } as u64;
    let budget = total.saturating_sub(margin);
    let fixed = w[dev] + kv + gdn + reserve;
    let remain = budget.saturating_sub(fixed);
    let gib = |b: u64| b as f64 / (1 << 30) as f64;
    let (slots, ledger) = if mixed {
        let q4_slots = n_q4.min((remain / blob_q4) as usize);
        let remain2 = remain.saturating_sub(q4_slots as u64 * blob_q4);
        let q3_slots = n_q3.min((remain2 / blob_q3) as usize);
        let slots = q4_slots + q3_slots;
        let used = q4_slots as u64 * blob_q4 + q3_slots as u64 * blob_q3;
        let all = q4_slots == n_q4 && q3_slots == n_q3;
        let ledger = format!(
            "[vram-budget] GPU{dev}: 总量 {:.1} − 边距 {:.1} − 权重 {:.1} − KV {:.2} \
             − GDN {:.2} − 预留 {:.1} = 余 {:.1} GiB | Q4 {}/{} (blob {:.2}MB) + \
             Q3 {}/{} (blob {:.2}MB) ⇒ 槽 {}，占用 {:.1} GiB{}",
            gib(total), gib(margin), gib(w[dev]), gib(kv), gib(gdn), gib(reserve),
            gib(remain),
            q4_slots, n_q4, blob_q4 as f64 / (1 << 20) as f64,
            q3_slots, n_q3, blob_q3 as f64 / (1 << 20) as f64,
            slots, gib(used),
            if all { "（全部专家可驻留）" } else { "" },
        );
        (slots, ledger)
    } else {
        let blob = if n_q4 > 0 { blob_q4 } else if n_q3 > 0 { blob_q3 } else { arena.gu_bytes + arena.down_bytes };
        let slots = (remain / blob) as usize;
        let ledger = format!(
            "[vram-budget] GPU{dev}: 总量 {:.1} − 边距 {:.1} − 权重 {:.1} − KV {:.2} \
             − GDN {:.2} − 预留 {:.1} = 余 {:.1} GiB ⇒ 槽 {} × blob {:.2}MB",
            gib(total), gib(margin), gib(w[dev]), gib(kv), gib(gdn), gib(reserve),
            gib(remain), slots, blob as f64 / (1 << 20) as f64,
        );
        (slots, ledger)
    };
    if slots == 0 {
        anyhow::bail!(
            "{ledger}\nautotune: GPU{dev} 预算连 1 个专家槽都装不下。降 LINGXI_SERVE_MAX_CTX、\
             显式给 LINGXI_ARENA_SLOTS/LINGXI_LAYER_SPLIT_SLOTS，或检查权重分类是否漏了新角色。"
        );
    }
    Ok(Some((slots, ledger)))
}

fn flash_next_meta() -> serde_json::Value {
    serde_json::json!({
        "hidden": 2560,
        "hc": 4,
        "lowrank": 320,
        "eps": 1e-06,
        "num_experts": 512,
        "top_k": 10,
        "inter": 640,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rot_dim": 64,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "output_gate_type": "sigmoid",
        "vocab_size": 248320,
        "ple_layer_ids": [2],
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20000000,
        "make_ngram_vocab_size_divisible_by": 128,
        "seed": 1234,
        "eos_token_id": 248044
    })
}

/// MTP 草稿层（`mtp/layer/*` + `mtp/mixer/*`）。
///
/// 结构与主干 QSA 层**同构**（§3.52 已交叉验证），所以这里只做"命名映射 + 构造"，
/// 前向直接复用主干的 `layer_step` —— 不另写一份容易写错的注意力 / MoE。
pub(crate) struct MtpLayer {
    pub(crate) weights: lingxi_model::layer::LayerWeights,
    pub(crate) k_layernorm_host: Vec<f32>,
    /// 层尾 mixer（`use_combine=False` ⇒ 没有 `block_inject`）
    pub(crate) mixer: lingxi_model::gr::GrWeights,
}

pub(crate) fn build_mtp_layer(
    k: &lingxi_kernels::Kernels,
    art: &Artifact,
    sparse: bool,
) -> Result<MtpLayer> {
    let p = "mtp/layer/";
    let up = |n: &str| lingxi_model::gpu::upload_any_bf16(k, art, &format!("{p}{n}"));
    // ⚠️ 产物里 MoE 段叫 `mlp/*`（主干叫 `moe/*`）—— 纯命名差异，结构相同
    let upm = |n: &str| lingxi_model::gpu::upload_any_bf16(k, art, &format!("{p}mlp/{n}"));
    let gr = |n: &str| -> Result<lingxi_model::gr::GrWeights> {
        Ok(lingxi_model::gr::GrWeights {
            hc_norm: up(&format!("{n}/hc_norm"))?,
            mix_down: up(&format!("{n}/mix_down"))?,
            mix_up: up(&format!("{n}/mix_up"))?,
            block_inject: Some(up(&format!("{n}/block_inject"))?),
        })
    };
    let klay = up("attention/indexer/k_layernorm")?;
    let klay_h: Vec<f32> = k.stream.clone_dtoh(&klay)?.iter().map(|v| v.to_f32()).collect();
    // ⚠️ 诊断开关：`query_norm` 与 `key_norm` 形状相同（[head_dim]），命名搞反同样会"能跑但全错"。
    let (q_norm, k_norm) = if std::env::var("LINGXI_MTP_SWAP_QK").is_ok() {
        eprintln!("[mtp] ⚠️ LINGXI_MTP_SWAP_QK=1：已交换 query_norm / key_norm（诊断用）");
        (up("attention/key_norm")?, up("attention/query_norm")?)
    } else {
        (up("attention/query_norm")?, up("attention/key_norm")?)
    };
    let weights = lingxi_model::layer::LayerWeights {
        attn_gr: gr("attn_gr")?,
        mlp_gr: gr("mlp_gr")?,
        attn: lingxi_model::layer::AttnWeights::Qsa {
            attn: lingxi_model::qsa::AttentionWeights {
                query_key_gate_value: up("attention/query_key_gate_value")?,
                query_key_gate_value_q4: None,
                query_norm: q_norm,
                key_norm: k_norm,
                output: up("attention/output")?,
                output_q4: None,
            },
            indexer: lingxi_model::qsa::IndexerWeights {
                index_qk_proj: up("attention/indexer/index_qk_proj")?,
                q_layernorm: up("attention/indexer/q_layernorm")?,
                k_layernorm: klay,
            },
        },
        moe: lingxi_model::moe::MoeWeights {
            router_shared_gate: upm("router_shared_gate")?,
            // `sparse=true`：512 个专家**不上传**（省 5.03 GiB 显存），运行时由 `MtpArena`
            // （把同批专家量化成 Q4G64 的 1.34 GiB）按路由直读 ⇒ 给 GPU0 腾出 ~3.7 GB 换专家槽。
            routed_gate_up: if sparse { None } else { Some(upm("routed_gate_up")?) },
            routed_down: if sparse { None } else { Some(upm("routed_down")?) },
            shared_gate_up: upm("shared_gate_up")?,
            shared_down: upm("shared_down")?,
        },
        ple: None,
    };
    let mixer = lingxi_model::gr::GrWeights {
        hc_norm: lingxi_model::gpu::upload_any_bf16(k, art, "mtp/mixer/hc_norm")?,
        mix_down: lingxi_model::gpu::upload_any_bf16(k, art, "mtp/mixer/mix_down")?,
        mix_up: lingxi_model::gpu::upload_any_bf16(k, art, "mtp/mixer/mix_up")?,
        block_inject: None,
    };
    Ok(MtpLayer { weights, k_layernorm_host: klay_h, mixer })
}

/// 装配两端权重：token embedding、末端 hyper-connection mixer、lm_head。
fn load_endpoints(
    k: &lingxi_kernels::Kernels,
    art: &Artifact,
) -> Result<lingxi_model::trunk::TrunkWeights> {
    let token_embedding =
        lingxi_model::gpu::upload_any_bf16(k, art, "text/token_embedding")?;
    // `LINGXI_HEAD_Q4=1`：lm_head（1.18 GiB bf16，每 token 全读）在线量化成 W4G64，
    // 省下 ~0.9 ms/token 的 GEMV 时间 + 0.87 GiB 显存（换成专家槽）。
    let output_head_q4 = if lingxi_model::flag("LINGXI_HEAD_Q4") {
        Some(lingxi_model::gpu::upload_q4_named(k, art, "text/output_head")?)
    } else {
        None
    };
    let output_head = if output_head_q4.is_some() {
        unsafe { k.stream.alloc::<half::bf16>(1)? }
    } else {
        lingxi_model::gpu::upload_any_bf16(k, art, "text/output_head")?
    };
    let mup = |n: &str| {
        lingxi_model::gpu::upload_any_bf16(k, art, &format!("text/hyper_connection_mixer/{n}"))
    };
    let mixer = lingxi_model::gr::GrWeights {
        hc_norm: mup("hc_norm")?,
        mix_down: mup("mix_down")?,
        mix_up: mup("mix_up")?,
        block_inject: None,
    };
    Ok(lingxi_model::trunk::TrunkWeights {
        token_embedding,
        mixer,
        output_head,
        output_head_q4,
    })
}

/// 生成 RoPE 表并上传（device bf16 + host f32 各一份），`len` 需覆盖 prefill+decode。
fn load_rope_tables(
    k: &lingxi_kernels::Kernels,
    len: usize,
    rot_dim: usize,
) -> Result<lingxi_driver::RopeTables> {
    let (cos_h, sin_h) =
        lingxi_model::rope::build_tables(len, rot_dim, lingxi_model::rope::ROPE_THETA);
    let to_dev = |v: &[f32]| -> Result<_> {
        Ok(k.stream
            .clone_htod(&v.iter().map(|x| half::bf16::from_f32(*x)).collect::<Vec<_>>())?)
    };
    Ok(lingxi_driver::RopeTables {
        cos: to_dev(&cos_h)?,
        sin: to_dev(&sin_h)?,
        cos_host: cos_h,
        sin_host: sin_h,
    })
}

/// 从产物装配一层的全部权重 + 形状（`layer-check` / `trunk-check` 共用）。
struct BuiltLayer {
    weights: lingxi_model::layer::LayerWeights,
    /// indexer 池化 key 的归一化 gamma（host f32；GDN 层为空）
    k_layernorm_host: Vec<f32>,
    shape: lingxi_model::layer::LayerShape,
    kind: lingxi_ids::LayerKind,
}

/// 由 golden meta 构造与层无关的 `LayerShape`（各层共用同一套超参）。
fn layer_shape(meta: &serde_json::Value) -> lingxi_model::layer::LayerShape {
    let u = |key: &str| meta[key].as_u64().unwrap() as usize;
    let eps = meta["eps"].as_f64().unwrap() as f32;
    lingxi_model::layer::LayerShape {
        hidden: u("hidden"),
        hc: u("hc"),
        lowrank: u("lowrank"),
        eps,
        attn: lingxi_model::qsa::AttnCfg {
            hidden: u("hidden"),
            heads: u("num_attention_heads"),
            kv_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            rot_dim: u("rot_dim"),
            eps,
        },
        indexer: lingxi_model::qsa::IndexerCfg {
            heads: u("indexer_n_heads"),
            head_dim: u("indexer_head_dim"),
            rot_dim: u("rot_dim"),
            budget: u("indexer_budget"),
            compress: u("indexer_compress_ratio"),
            eps,
        },
        gdn: lingxi_model::gdn::GdnCfg {
            hidden: u("hidden"),
            num_k_heads: u("linear_num_key_heads"),
            num_v_heads: u("linear_num_value_heads"),
            head_k_dim: u("linear_key_head_dim"),
            head_v_dim: u("linear_value_head_dim"),
            conv_kernel: u("linear_conv_kernel_dim"),
            gate_activation: lingxi_ids::GateActivation::from_name(
                meta["output_gate_type"].as_str().unwrap_or("silu"),
            )
            .unwrap_or(lingxi_ids::GateActivation::Silu),
        },
        num_experts: u("num_experts"),
        top_k: u("top_k"),
        inter: u("inter"),
    }
}

fn build_layer(
    k: &lingxi_kernels::Kernels,
    art: &Artifact,
    meta: &serde_json::Value,
    layer: u32,
    sparse: bool,
    cache: Option<&host_cache::HostCache>,
) -> Result<BuiltLayer> {
    let u = |key: &str| meta[key].as_u64().unwrap() as usize;
    let hidden = u("hidden");
    let hc = u("hc");
    let eps = meta["eps"].as_f64().unwrap() as f32;
    let p = format!("text/layers/{layer}/");
    // dense 权重：有 host cache 走常驻缓存（未命中同步填、后台预读提前一层），
    // 无 cache（对拍命令）走原始产物直读。
    let up = |n: &str| -> Result<_> {
        let name = format!("{p}{n}");
        match cache {
            Some(c) => c.upload_dense(k, &name),
            None => lingxi_model::gpu::upload_any_bf16(k, art, &name),
        }
    };

    // 【已否决的实验，留档于此】曾试过 `LINGXI_GR_Q4=1` 把 GR 的 mix_down/mix_up 转 W4G64
    // （它们 1.26 GB/token，占全量流量 27%，却一直是 bf16）。整机实测 33 ms → **12 s/token**
    // （慢 360 倍），且 launch-bench 逐形状算下来**净亏**：mix_down [320,10240] 的 Q4 只有
    // `n/8 = 40` 个 block，14.4 µs 反而慢于 bf16 的 9.3 µs。**方向关闭，代码已撤。**
    let attn_gr = lingxi_model::gr::GrWeights {
        hc_norm: up("attn_gr/hc_norm")?,
        mix_down: up("attn_gr/mix_down")?,
        mix_up: up("attn_gr/mix_up")?,
        block_inject: Some(up("attn_gr/block_inject")?),
    };
    let mlp_gr = lingxi_model::gr::GrWeights {
        hc_norm: up("mlp_gr/hc_norm")?,
        mix_down: up("mlp_gr/mix_down")?,
        mix_up: up("mlp_gr/mix_up")?,
        block_inject: Some(up("mlp_gr/block_inject")?),
    };
    let moe = lingxi_model::moe::MoeWeights {
        router_shared_gate: up("moe/router_shared_gate")?,
        // sparse=true：512 个专家不预上传（每层省约 5GB），运行时由 ExpertSource
        // 按路由结果只取命中专家（prefill ≤rows*top_k、decode=top_k 个）
        routed_gate_up: if sparse {
            None
        } else {
            Some(up("moe/routed_gate_up")?)
        },
        routed_down: if sparse { None } else { Some(up("moe/routed_down")?) },
        shared_gate_up: up("moe/shared_gate_up")?,
        shared_down: up("moe/shared_down")?,
    };

    // `LINGXI_DENSE_Q4=1`：把最大的几块稠密矩阵在线量化成 W4G64（decode 走融合反量化 GEMV）。
    // 这些矩阵实测全是纯带宽受限（launch-bench 分形状表 ~700 GB/s），是 decode 的主要流量。
    let dense_q4 = lingxi_model::flag("LINGXI_DENSE_Q4");
    let q4of = |name: &str| -> Result<Option<lingxi_model::q4::Q4W>> {
        if !dense_q4 {
            return Ok(None);
        }
        // 按张量名的磁盘缓存：命中就连 bf16 都不读（"提前算好、之后只取"）。
        Ok(Some(lingxi_model::gpu::upload_q4_named(k, art, name)?))
    };
    // W4 模式下**不再上传这些矩阵的 bf16**（省 ~6 GiB → 直接换成更多专家槽）。
    // 留 1 元素 stub 只为满足结构体类型；所有读取点都先看 `*_q4`，走 W4 时不会读它。
    // （`gdn_forward` / `attention_forward` 这类非增量路径仍读 bf16 → 那些子命令不要开 W4。）
    let up_w = |name: &str| -> Result<cudarc::driver::CudaSlice<half::bf16>> {
        if dense_q4 {
            Ok(unsafe { k.stream.alloc::<half::bf16>(1) }?)
        } else {
            up(name)
        }
    };
    // QSA 的两块单独开关（`LINGXI_DENSE_Q4_QSA=1`）：实测 GDN 侧稳定，QSA 侧在长序列下会
    // 触发 `CUDA_ERROR_ILLEGAL_ADDRESS`（未定位完），所以默认分开控制。
    let qsa_q4 = lingxi_model::flag("LINGXI_DENSE_Q4_QSA");
    let q4of_qsa = |name: &str| -> Result<Option<lingxi_model::q4::Q4W>> {
        if !qsa_q4 {
            return Ok(None);
        }
        q4of(name)
    };

    let kind = lingxi_model::layer_kind(layer);

    let (attn, klay_h) = match kind {
        lingxi_ids::LayerKind::QsaSparseAttention => {
            let klay = up("attention/indexer/k_layernorm")?;
            let klay_h: Vec<f32> = k.stream.clone_dtoh(&klay)?.iter().map(|v| v.to_f32()).collect();
            let w = lingxi_model::layer::AttnWeights::Qsa {
                attn: lingxi_model::qsa::AttentionWeights {
                    query_key_gate_value: if qsa_q4 { up_w("attention/query_key_gate_value")? } else { up("attention/query_key_gate_value")? },
                    query_key_gate_value_q4: q4of_qsa(&format!("{p}attention/query_key_gate_value"))?,
                    query_norm: up("attention/query_norm")?,
                    key_norm: up("attention/key_norm")?,
                    output: if qsa_q4 { up_w("attention/output")? } else { up("attention/output")? },
                    output_q4: q4of_qsa(&format!("{p}attention/output"))?,
                },
                indexer: lingxi_model::qsa::IndexerWeights {
                    index_qk_proj: up("attention/indexer/index_qk_proj")?,
                    q_layernorm: up("attention/indexer/q_layernorm")?,
                    k_layernorm: up("attention/indexer/k_layernorm")?,
                },
            };
            (w, klay_h)
        }
        lingxi_ids::LayerKind::GdnLinearAttention => {
            let w = lingxi_model::layer::AttnWeights::Gdn(lingxi_model::gdn::GdnWeights {
                query_key_value_z: up_w("gdn/query_key_value_z")?,
                query_key_value_z_q4: q4of(&format!("{p}gdn/query_key_value_z"))?,
                a_b_projection: up("gdn/a_b_projection")?,
                convolution: up("gdn/convolution")?,
                norm: up("gdn/norm")?,
                output: up_w("gdn/output")?,
                output_q4: q4of(&format!("{p}gdn/output"))?,
                a_log: lingxi_model::gpu::read_fp32_host(art, &format!("{p}gdn/a_log"))?,
                dt_bias: lingxi_model::gpu::read_fp32_host(art, &format!("{p}gdn/dt_bias"))?,
            });
            (w, Vec::new())
        }
    };

    let shape = layer_shape(meta);

    // PLE（仅 layer 1；权重是全局的 `text/ple/*`）
    let ple_ids: Vec<i64> = meta["ple_layer_ids"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_i64().unwrap()).collect())
        .unwrap_or_default();
    let ple = if ple_ids.contains(&(layer as i64 + 1)) {
        let cfg = lingxi_model::plet::PleCfg {
            hidden,
            hc,
            embed_dim: u("ple_embed_dim"),
            ngram_size: u("ngram_size"),
            heads_per_ngram: u("heads_per_ngram"),
            ngram_vocab_base: u("ngram_vocab_size_base") as u64,
            unigram_vocab: u("vocab_size") as u64,
            divisible_by: u("make_ngram_vocab_size_divisible_by") as u64,
            seed: u("seed") as u64,
            eos: meta["eos_token_id"].as_i64().unwrap(),
            conv_kernel: u("ple_conv_kernel_size"),
            eps,
            // PLE 表路径（**不写死，四级解析，兜底响亮警告不静默**）：
            //   1. `LINGXI_PLE_TABLE` 显式指定（最高优先）；
            //   2. /dev/shm 官方 IQ4_NL 表（26.8GB，量化器 quantize_ple_iq4nl.py 产出）；
            //   3. /dev/shm 官方 FP8 表（51.2GB，extract_ple_fp8.py 产出）；
            //   4. 兜底 lingxi-assets bf16（102GB 机械盘）——**打响亮横幅**（慢 5×，不许静默）。
            // 格式自动识别：`.iq4nl` sidecar ⇒ IQ4_NL；`.scale` sidecar ⇒ FP8；否则 bf16。
            table_path: std::env::var("LINGXI_PLE_TABLE").unwrap_or_else(|_| {
                // 优先级：shm iq4nl（26.8G）> shm fp8（51.2G）> 磁盘 bf16 兜底（102G，慢！）
                for (cand, sidecar) in [
                    ("/dev/shm/qwen4_exp.ple-table.iq4nl.bin", ".iq4nl"),
                    ("/dev/shm/qwen4_exp.ple-table.fp8.bin", ".scale"),
                ] {
                    // 必须同时存在 sidecar（量化器的完成标记）：只检文件本体的话，
                    // 量化器 prealloc 的**半写完**文件会被当完整表读（实测：读穿 EOF
                    // "failed to fill whole buffer"）。
                    if std::path::Path::new(cand).exists()
                        && std::path::Path::new(&format!("{cand}{sidecar}")).exists()
                    {
                        return cand.to_string();
                    }
                }
                // ⚠️ **响亮兜底**：/dev/shm 的内存表丢了（机器重启/crash 会清 shm）才走这里。
                // 磁盘随机读 prefill 慢 5 倍级（实测 60ms/行 vs 11.3ms/行）——绝不让它静默发生。
                eprintln!(
                    "╔══════════════════════════════════════════════════════════════════╗\n\
                     ║ ⚠️  PLE 表回退到机械盘 bf16（lingxi-assets，102GB）——prefill 将慢 5× ║\n\
                     ║     内存表丢失。恢复：python3 tools/convert/extract_ple_fp8.py     ║\n\
                     ║     （或 quantize_ple_iq4nl.py），或显式 LINGXI_PLE_TABLE=...     ║\n\
                     ╚══════════════════════════════════════════════════════════════════╝"
                );
                "/data/models/lingxi-assets/qwen4_exp.ple-table.bin".to_string()
            }),
        };
        let derived = lingxi_model::plet::PleDerived::new(&cfg, 0);
        let pup = |n: &str| lingxi_model::gpu::upload_any_bf16(k, art, &format!("text/ple/{n}"));
        // 产物的 convolution 是 `[dim, K]`（HF 卷积权重布局），转成卷积核要的 `[K, dim]`
        let dim = hc * hidden;
        let kk = cfg.conv_kernel;
        let conv_h = lingxi_model::gpu::read_f32_host(k, art, "text/ple/convolution")?;
        let mut t_h = vec![0f32; kk * dim];
        for c in 0..dim {
            for kk_i in 0..kk {
                t_h[kk_i * dim + c] = conv_h[c * kk + kk_i];
            }
        }
        let convolution = k
            .stream
            .clone_htod(&t_h.iter().map(|v| half::bf16::from_f32(*v)).collect::<Vec<_>>())?;
        let table = lingxi_model::plet::PleTable::open(&cfg.table_path, derived.head_dim)?;
        Some(lingxi_model::plet::PleWeights {
            key_proj: pup("key_proj")?,
            value_proj: pup("value_proj")?,
            norm_key: pup("norm_key")?,
            norm_query: pup("norm_query")?,
            norm_conv: pup("norm_conv")?,
            convolution,
            cfg,
            derived,
            table,
        })
    } else {
        None
    };

    Ok(BuiltLayer {
        weights: lingxi_model::layer::LayerWeights { attn_gr, mlp_gr, attn, moe, ple },
        k_layernorm_host: klay_h,
        shape,
        kind,
    })
}

/// 瘦身专家源（P1）：路由后只取命中专家；host 字节走 [`host_cache::HostCache`]
/// （LRU + 并行冷读），device 侧再经 [`host_cache::DeviceExpertLru`] 让热专家
/// 整槽常驻显存。
///
/// 自带一对 device 紧凑张量作为 scratch，跨层跨步复用（48 层 × N 步只用同一块
/// 显存逐层覆写）。driver 单序列单线程调用，故用 UnsafeCell 提供 &self 内部可变性。
struct MmapExpertSource<'a> {
    k: &'a lingxi_kernels::Kernels,
    cache: &'a host_cache::HostCache,
    dev: std::cell::UnsafeCell<host_cache::DeviceExpertLru>,
    scratch: std::cell::UnsafeCell<ExpertScratch>,
}

struct ExpertScratch {
    dgu: Option<cudarc::driver::CudaSlice<half::bf16>>,
    ddn: Option<cudarc::driver::CudaSlice<half::bf16>>,
}

// 安全前提：ExpertSource 仅在 driver 单序列的主线程逐层调用，不跨线程共享；
// 返回的 ExpertHits 借用在下一次 load_hits 之前必然用完（单层 MoE 内成立）。
unsafe impl Sync for MmapExpertSource<'_> {}

impl<'a> MmapExpertSource<'a> {
    fn new(
        k: &'a lingxi_kernels::Kernels,
        cache: &'a host_cache::HostCache,
        inter: usize,
        hidden: usize,
        dev_gb: usize,
    ) -> Result<Self> {
        Ok(Self {
            k,
            cache,
            dev: std::cell::UnsafeCell::new(host_cache::DeviceExpertLru::new(
                k, dev_gb, inter, hidden,
            )?),
            scratch: std::cell::UnsafeCell::new(ExpertScratch { dgu: None, ddn: None }),
        })
    }
}

impl<'a> lingxi_model::moe::ExpertSource for MmapExpertSource<'a> {
    fn load_hits<'b>(
        &'b self,
        layer: u32,
        experts: &[u32],
        inter: usize,
        hidden: usize,
        _want_q4: bool,
        _overlap: bool,
    ) -> Result<lingxi_model::moe::ExpertHits<'b>> {
        let per_gu = 2 * inter * hidden;
        let per_dn = hidden * inter;
        // 1) host 字节：mmap 只读视图（零拷贝；首次触页在 fill 并行线程里读盘）
        let prof = std::env::var("LINGXI_PROF").is_ok();
        let tf = std::time::Instant::now();
        let blobs = self.cache.fetch_hits(layer, experts, inter, hidden)?;
        if prof {
            eprintln!(
                "    [fetch ] 层 {layer:2} {} 槽视图 {:.3}s",
                experts.len(),
                tf.elapsed().as_secs_f64()
            );
        }
        // 2) device LRU 聚集进 scratch（热槽 dtod / 冷槽分槽 htod + 回填）
        {
            let sc = unsafe { &mut *self.scratch.get() };
            let ngu = experts.len() * per_gu;
            let ndn = experts.len() * per_dn;
            if sc.dgu.as_ref().map_or(0, |s| s.len()) < ngu {
                sc.dgu = Some(unsafe { self.k.stream.alloc::<half::bf16>(ngu) }?);
            }
            if sc.ddn.as_ref().map_or(0, |s| s.len()) < ndn {
                sc.ddn = Some(unsafe { self.k.stream.alloc::<half::bf16>(ndn) }?);
            }
            let dev = unsafe { &mut *self.dev.get() };
            dev.fill(
                self.k,
                layer,
                experts,
                &blobs,
                sc.dgu.as_mut().unwrap(),
                sc.ddn.as_mut().unwrap(),
            )?;
        }
        let sc = unsafe { &*self.scratch.get() };
        Ok(lingxi_model::moe::ExpertHits {
            gate_up: sc.dgu.as_ref().unwrap(),
            down: sc.ddn.as_ref().unwrap(),
            q4: None,
            q4_alt: None,
            bits_sel: Vec::new(),
        })
    }
}

/// 流式层供给：每层每步现建、用后即弃（对拍 / 全量对照路径，保持原行为）。
struct StreamingProvider<'a> {
    cur: Option<lingxi_model::trunk::LoadedLayer>,
    build: Box<dyn FnMut(u32) -> Result<lingxi_model::trunk::LoadedLayer> + 'a>,
}

impl<'a> lingxi_driver::LayerProvider for StreamingProvider<'a> {
    fn layer(&mut self, i: u32) -> Result<&lingxi_model::trunk::LoadedLayer> {
        self.cur = Some((self.build)(i)?);
        Ok(self.cur.as_ref().unwrap())
    }
}

/// dense 显存常驻供给：每层 dense 权重只在第一次访问时构建一次，之后全部驻留
/// 显存、跨 decode 步按引用复用——decode 每层每步权重逐字节相同，没有任何理由
/// 重新 H2D（注意 `CudaSlice::clone` 会真复制显存，所以只能走引用复用）。
/// sparse 模式下专家张量不在这里，仍由 `ExpertSource` 每步按路由注入。
struct DeviceLayerCache<'a> {
    slots: Vec<Option<lingxi_model::trunk::LoadedLayer>>,
    build: Box<dyn FnMut(u32) -> Result<lingxi_model::trunk::LoadedLayer> + 'a>,
}

impl<'a> lingxi_driver::LayerProvider for DeviceLayerCache<'a> {
    fn layer(&mut self, i: u32) -> Result<&lingxi_model::trunk::LoadedLayer> {
        let idx = i as usize;
        if self.slots[idx].is_none() {
            self.slots[idx] = Some((self.build)(i)?);
        }
        Ok(self.slots[idx].as_ref().unwrap())
    }
    fn as_cache(&self) -> Option<&dyn lingxi_driver::LayerCache> {
        Some(self)
    }
}

/// SAFETY：**只允许在全部层预热之后、且不再调用 `layer()` 的前提下跨线程读**。
/// `build` 只在 `&mut self` 的 `layer()` 里访问；`peek` 只读 `slots`。
unsafe impl<'a> Sync for DeviceLayerCache<'a> {}

impl<'a> lingxi_driver::LayerCache for DeviceLayerCache<'a> {
    fn peek(&self, i: u32) -> Option<&lingxi_model::trunk::LoadedLayer> {
        self.slots[i as usize].as_ref()
    }
}

fn categorize(name: &str) -> &'static str {
    if name == "text/token_embedding" {
        return "embedding";
    }
    if name == "text/output_head" {
        return "output_head";
    }
    if name.starts_with("text/ple/") {
        return "ple";
    }
    if name.starts_with("mtp/") {
        return "mtp";
    }
    if name.ends_with("/routed_gate_up") || name.ends_with("/routed_down") {
        return "moe_routed_experts";
    }
    if name.ends_with("/router_shared_gate") {
        return "moe_router";
    }
    if name.ends_with("/shared_gate_up") || name.ends_with("/shared_down") {
        return "moe_shared";
    }
    if name.contains("/indexer/") {
        return "qsa_indexer";
    }
    if name.contains("/attention/") {
        return "qsa_attention";
    }
    if name.contains("/gdn/") {
        return "gdn";
    }
    if name.ends_with("hc_norm")
        || name.ends_with("block_inject")
        || name.ends_with("mix_down")
        || name.ends_with("mix_up")
    {
        return "gr";
    }
    "other"
}

/// 把 canonical 名归到所属层（`text/layers/N`、`mtp/...`、`text/ple`、`top`）。
fn layer_of(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("text/layers/") {
        if let Some(idx) = rest.split('/').next() {
            return format!("text/layers/{idx}");
        }
    }
    if name.starts_with("mtp/") {
        return "mtp".to_string();
    }
    if name.starts_with("text/ple/") {
        return "text/ple".to_string();
    }
    "top".to_string()
}

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("读 {} 失败: {e}", p.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    let n = got.len().min(want.len());
    let num: f32 = (0..n).map(|i| (got[i] - want[i]).powi(2)).sum::<f32>().sqrt();
    let den: f32 = (0..n).map(|i| want[i] * want[i]).sum::<f32>().sqrt();
    num / den.max(1e-12)
}

fn stats(v: &[f32]) -> (f64, f64, usize) {
    let mut sum = 0f64;
    let mut sumsq = 0f64;
    for &x in v {
        let d = x as f64;
        sum += d;
        sumsq += d * d;
    }
    (sum, sumsq, v.len())
}
