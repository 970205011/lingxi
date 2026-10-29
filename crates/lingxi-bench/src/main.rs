//! `lingxi-bench` — 压测 / 对拍 / 检查工具入口。旁路，不进热路径。
//!
//! 当前提供：
//!   `verify <artifact>` — 打开现成 ninfer v2 产物，跑结构自检 + 构建 tensor 目录
//!   `list   <artifact>` — 打印目录中的张量（名字/形状/格式/布局/角色）

use std::process::Command;

use anyhow::{Context, Result};
use cudarc::driver::DevicePtr;
use lingxi_artifact::Artifact;

mod host_cache;
mod mtp_arena;
mod arena_source;
mod doctor;
mod moe_gemm_bench;
mod serve;
mod spec;
mod vram;
mod cli;

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
    // 自报版本：任何环境（无 GPU/无产物）都能跑，支持"你跑的是哪个版本"快速定位。
    if matches!(
        args.get(1).map(|s| s.as_str()),
        Some("version") | Some("--version") | Some("-V")
    ) {
        println!(
            "lingxi-bench {} (git {}, target: {})",
            env!("CARGO_PKG_VERSION"),
            option_env!("LINGXI_BUILD_GIT").unwrap_or("unknown"),
            std::env::consts::ARCH
        );
        return Ok(());
    }
    // 环境开关校验（点名拼错/幽灵开关、拦住已知危险组合）——必须在任何 GPU 初始化之前。
    // `probes` 子命令只打印状态表，不跑别的（但**也要先过一遍校验**，方便在跑之前查配置）。
    lingxi_model::probes::check_env();
    if args.get(1).map(|s| s.as_str()) == Some("probes") {
        print!("{}", lingxi_model::probes::dump());
        return Ok(());
    }
    // export-tensor <art> <name> <out>：把 artifact 里的 bf16 张量导成裸 LE 字节
    // （2026-10-29 共享专家 GPTQ 校准链用：离线 numpy 工具读权重，免 torch 依赖）。
    if args.get(1).map(|s| s.as_str()) == Some("export-tensor") {
        let art = Artifact::open(std::path::Path::new(&args[2]))?;
        let (bits, rows, cols) = lingxi_model::gpu::read_bf16_bits(&art, &args[3])?;
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        std::fs::write(&args[4], bytes)?;
        println!("export-tensor {name}: {rows}x{cols} -> {out}",
            name = args[3], out = args[4]);
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
        Some("verify") => cli::cmd::verify::run(&args)?,
        Some("doctor") => doctor::run(&args)?,
        Some("moe-gemm-bench") => moe_gemm_bench::run(&args)?,
        Some("vram-budget") => cli::cmd::vram::run(&args)?,
        Some("bytes") => cli::cmd::bytes::run(&args)?,
        Some("mtp-pack") => cli::cmd::mtp::pack(&args)?,
        Some("mtp-accept") => cli::cmd::mtp::accept(&args)?,
        Some("mtp-dump") => cli::cmd::mtp::dump(&args)?,
        Some("mtp-load") => cli::cmd::mtp::load(&args)?,
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
                        shared_gate_up_q4: None,
                        shared_down_q4: None,
                    };
                    let dbg_dir = if std::env::var("LINGXI_MOE_DEBUG").is_ok() { Some(dir.as_path()) } else { None };
                    let res = lingxi_model::moe::sparse_moe_forward(
                        &k, &w, &dx, rows, hidden, num_experts, top_k, inter, dbg_dir, 0, None,
                        None, None,
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
            let built = cli::common::build_layer(&k, &art, &meta, layer, false, None)?;
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

            let shape = cli::common::layer_shape(&meta);
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
                let b = cli::common::build_layer(&k, &art, &meta, layer, false, None)?;
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
            let shape = cli::common::layer_shape(&meta);
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
            let built = cli::common::build_layer(&k, &art, &meta, layer, false, None)?;
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
            let shape = cli::common::layer_shape(&meta);

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
                    let b = cli::common::build_layer(&k, &art, &meta, i, false, None)?;
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
                    let b = cli::common::build_layer(&k, &art, &meta, i, false, None)?;
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
                ep: None,
                twopass_hi_on0: None,
                twopass_lo_on1: None,
                twopass_rmoe_hi1: None,
                twopass_rmoe_lo0: None,
            };

            println!("driver prefill：{n_layers} 层流式（机械盘约 20 分钟）…");
            let t0 = std::time::Instant::now();
            let mut loader = cli::common::StreamingProvider {
                cur: None,
                build: {
                    let (kb, ab, mb) = (&k, &art, &meta);
                    Box::new(move |layer: u32| -> Result<lingxi_model::trunk::LoadedLayer> {
                        eprint!("\r  [drive] 层 {layer}/{n_layers}  {:>5.1}s", t0.elapsed().as_secs_f64());
                        let b = cli::common::build_layer(kb, ab, mb, layer, false, None)?;
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
        Some("talk") => cli::cmd::talk::run(&args)?,
        Some("serve") => cli::cmd::serve_cmd::run(&args)?,
        _ => {
            eprintln!("lingxi-bench: 用法: doctor|verify|list|bytes|vram-budget|decode-check|dequant-check|component-check|layer-check|trunk-check|trunk-ends|step-check|drive-check|argmax-check|f32x-check|talk|serve ...");
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
    // ★ 2026-10-25 `LINGXI_MTP_ATTN_Q4=1`（**默认开**，`=0` 回退）：
    //   MTP（草稿）层的 `attention/query_key_gate_value`（68.2 MB bf16）与 `attention/output`
    //   （31.5 MB）在线量化成 W4G64 —— 每步省 ~74 MB 的读，草稿 ~3.79 步/轮 ⇒ 约 −0.46 ms/轮。
    //   为什么**不损质量**：贪心投机解码里草稿只决定"这一轮能接受几个"，
    //   被接受的前缀永远是主干自己的 argmax ⇒ **草稿用什么权重都不改变生成文本**
    //   （判据：md5 必须与不开时相同；只有接受率可能微动，损了就把开关关掉）。
    //   与主干同一套 `upload_q4_named`（带磁盘缓存），W4 形态 ⇒ 也顺带省 ~74 MB 显存。
    // ⚠️ 2026-10-25 **默认已翻回 0（关）**：端到端 2×2 实测（200 token，同 prompt）
    //   开 = 70.3/68.6 tok/s、49 轮、接受 149（3.04/轮）；关 = 78.4/79.0、47 轮、153（3.26/轮）。
    //   ⇒ 省下的 0.1 ms/轮被"接受率 3.26→3.04"吃成 **−10%**（生成的文本 md5 四跑全同，纯亏吞吐）。
    //   连带修正一条授权边界：**"草稿权重可以随便量化"只对 head 成立**（head 的舍入只扰动
    //   最终 argmax ~1e-7）；草稿的**注意力**权重会改变整层 hidden ⇒ 草稿分布漂移、接受率掉。
    //   `=1` 仍可开（代码与缓存都在），只作诊断/省显存用（−74 MB）。
    let mtp_attn_q4 = std::env::var("LINGXI_MTP_ATTN_Q4").map(|v| v == "1").unwrap_or(false);
    let up_m = |n: &str| -> Result<cudarc::driver::CudaSlice<half::bf16>> {
        if mtp_attn_q4 {
            // 1 元素 stub：所有读取点都先看 `*_q4`，W4 档不会读它
            Ok(unsafe { k.stream.alloc::<half::bf16>(1) }?)
        } else {
            up(n)
        }
    };
    let q4_m = |n: &str| -> Result<Option<lingxi_model::q4::Q4W>> {
        if mtp_attn_q4 {
            Ok(Some(lingxi_model::gpu::upload_q4_named(k, art, &format!("{p}{n}"))?))
        } else {
            Ok(None)
        }
    };
    let weights = lingxi_model::layer::LayerWeights {
        attn_gr: gr("attn_gr")?,
        mlp_gr: gr("mlp_gr")?,
        attn: lingxi_model::layer::AttnWeights::Qsa {
            attn: lingxi_model::qsa::AttentionWeights {
                query_key_gate_value: up_m("attention/query_key_gate_value")?,
                query_key_gate_value_q4: q4_m("attention/query_key_gate_value")?,
                query_norm: q_norm,
                key_norm: k_norm,
                output: up_m("attention/output")?,
                output_q4: q4_m("attention/output")?,
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
            shared_gate_up_q4: None,
            shared_down_q4: None,
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
