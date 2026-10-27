//! `talk <artifact> "<prompt>" [max_new_tokens]` — 出字闭环（prefill + 贪心 decode）。
//! 从 main.rs 搬出（CODE_MAP Phase 1 刀2b，只搬家不改行为）。

use std::sync::Arc;

use anyhow::{Context, Result};
use lingxi_artifact::Artifact;

pub fn run(args: &[String]) -> Result<()> {
    // 用法: talk <artifact> "<prompt>" [max_new_tokens]
    // 出字闭环：产物内 frontend/tokenizer.json 分词 → Sequence 贪心
    // prefill+generate → token 解码回文本。当前每层全量取权重（机械盘慢），
    // 只用于打通"文本进→文本出"；速度要等专家瘦身+缓存（P1）。
    let art_path = args.get(2).context("用法: lingxi-bench talk <artifact> <prompt> [max_new_tokens]")?;
    let prompt = crate::cli::common::prompt_arg(args.get(3)).context("缺少 prompt 文本")?;
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
    let meta = crate::cli::common::flash_next_meta();

    let shape = crate::cli::common::layer_shape(&meta);

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
    let endpoints = crate::cli::common::load_endpoints(&k, &art)?;
    let rope_len = prompt_len + max_new + 1;
    let tables = crate::cli::common::load_rope_tables(&k, rope_len, shape.attn.rot_dim)?;

    let n_layers = lingxi_model::DIMS.n_layers as u32;
    // ★★ **全量 layer-split**。
    // 这套接线必须建在 `Sequence` **之前**：`[lo, n_layers)` 的跨 token 状态
    // （QSA KV/indexer、GDN 的 S+卷积史、PLE 史）要**直接建在 GPU1 上**，
    // 建错卡不报错、只会在第一次 kernel 时 ILLEGAL_ADDRESS 或静默写串。
    let ls_arena_path = std::env::var("LINGXI_EXPERT_ARENA").ok();
    // ★★ **EP（专家并行，`LINGXI_EP=1`）**：两卡都跑全部 n_layers 层的稠密段，
    // 专家按 id 二分（卡 0 < half，卡 1 ≥ half），每层交换 MoE 部分和。
    // EP 与 layer-split **互斥**（都要求整层/半专家在本卡，混用会重复计算）。
    // 跨卡 kernel 不能被 CUDA Graph 捕获 ⇒ 强制 `LINGXI_NO_GRAPHS=1`。
    let ep_on = std::env::var("LINGXI_EP").map(|v| v == "1").unwrap_or(false);
    if ep_on {
        std::env::set_var("LINGXI_NO_GRAPHS", "1");
        lingxi_model::graph::force_disable();
    }
    // ★ P15-㉜：双卡默认自动 split（同 mtp-accept 分支的口径）。EP 模式下必须关掉。
    let ls_env: Option<u32> = if ep_on {
        Some(0)
    } else {
        std::env::var("LINGXI_LAYER_SPLIT").ok().and_then(|s| s.parse().ok())
    };
    let auto_split = !ep_on
        && ls_env.is_none()
        && cudarc::driver::CudaContext::new(1).is_ok()
        && ls_arena_path.is_some();
    let ls_lo: u32 = ls_env.unwrap_or(if auto_split { 24 } else { 0 });
    let ls_on = !ep_on && ls_lo > 0 && ls_lo < n_layers;
    if ls_on {
        if auto_split {
            // ㊸：单层本地图在 split 下保持开启（专家段 3 核固定槽、层⇄设备一一对应、
            // 跨卡拷贝在层循环外不进图）。判据翻车则恢复 force_disable。
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
        if ls_on || ep_on { Some(lingxi_kernels::Kernels::load(1)?) } else { None };
    let ls_tables: Option<lingxi_driver::RopeTables> = match ls_k1.as_ref() {
        Some(kk) => Some(crate::cli::common::load_rope_tables(kk, rope_len, shape.attn.rot_dim)?),
        None => None,
    };
    // ㊿-P0：未显式给槽数时走显存预算自动分配（env 显式值优先）。
    // ★ EP：两卡各装**全部** 48 层稠密权重，显存比 layer-split 更紧 ⇒ 每卡槽数默认
    // 8000（`LINGXI_EP_SLOTS` 可覆盖），不走 autotune（那是按半场层数算的）。
    let ep_slots: usize = std::env::var("LINGXI_EP_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);
    let ls_slots: usize = if ep_on {
        ep_slots
    } else {
    match std::env::var("LINGXI_LAYER_SPLIT_SLOTS")
        .ok().and_then(|s| s.parse().ok())
    {
        Some(v) => v,
        None => match ls_arena_path.as_deref() {
            Some(ap) => match crate::vram::autotune_arena_slots(&art, ap, 1, ls_lo, rope_len + 8)? {
                Some((s, ledger)) => {
                    eprintln!("{ledger}");
                    s
                }
                None => 9600,
            },
            None => 9600,
        },
    }
    };  // ★㊲ 工作集≥槽位判决(375/卡层)
    let ls_experts: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> =
        match (ls_k1.as_ref(), ls_arena_path.as_deref()) {
            (Some(kk), Some(ap)) => {
                if ep_on {
                    eprintln!(
                        "[EP] GPU1 上下文就绪；专家 id ≥ {} 由它算，专家槽位 {ep_slots}",
                        shape.num_experts / 2
                    );
                    Some(Box::new(crate::arena_source::ArenaExpertSource::new(
                        kk, ap, ep_slots, false, 0..u32::MAX,
                    )?))
                } else {
                    eprintln!(
                        "[layer-split] GPU1 上下文就绪；层 {ls_lo}..{n_layers} 整层交给它，专家槽位 {ls_slots}"
                    );
                    Some(Box::new(crate::arena_source::ArenaExpertSource::new(
                        kk, ap, ls_slots, false, ls_lo..n_layers,
                    )?))
                }
            }
            (Some(_), None) => anyhow::bail!(
                "layer-split / EP 需要 LINGXI_EXPERT_ARENA（GPU1 那份专家表）"
            ),
            _ => None,
        };
    let mut seq = if ep_on {
        // EP：两卡各一套**全部 n_layers 层**的跨 token 状态
        lingxi_driver::Sequence::new_ep(
            &k,
            ls_k1.as_ref().expect("EP 必须有 GPU1 上下文"),
            n_layers,
            &shape,
            rope_len + 8,
            Some((
                1,
                meta["ple_conv_kernel_size"].as_u64().unwrap() as usize,
                meta["ngram_size"].as_u64().unwrap() as usize,
            )),
        )?
    } else {
        lingxi_driver::Sequence::new_split(
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
        )?
    };
    // host 缓存（sparse 时启用）：dense 匿名常驻 + 专家 mmap 零拷贝视图
    let host_cache: Option<Arc<crate::host_cache::HostCache>> = if sparse {
        let c = crate::host_cache::HostCache::new(Arc::clone(&art));
        // ★ 默认**不起**后台预读线程（2026-10-23 实测反转）：prefill 会把 48 层的 dense
        // 转换请求全塞进无界队列，后台线程一直转到 decode 期，突发的大块 malloc/缺页会让
        // decode 单轮停顿 2~5.7s（实测 47 轮里 1 次，聚合 59.9→33.3 tok/s；关掉后连跑两遍
        // 逐位同分 61.5 tok/s、最大轮 95ms）。且关掉后 prefill 用时不变（4.4s，预读并没有
        // 省时间）。要对照旧行为才设 `LINGXI_PREFETCH=1`。
        if std::env::var("LINGXI_PREFETCH").map(|v| v == "1").unwrap_or(false) {
            c.start_prefetcher();
            eprintln!("host_cache: dense 后台预读已启动（LINGXI_PREFETCH=1），专家走 mmap 零拷贝视图");
        } else {
            eprintln!("host_cache: dense 后台预读已关闭（默认；LINGXI_PREFETCH=1 可开回），专家走 mmap 零拷贝视图");
        }
        Some(c)
    } else {
        None
    };

    // 瘦身专家源（sparse 时启用）：
    // LINGXI_EXPERT_ARENA 给出 → arena 直读（只读 mmap + 整段 pin + DMA + 视图解量化）
    // 否则 → 旧的 host LRU(mmap 视图) + device LRU 路径（对照/回退）
    let expert_source: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> = if sparse {
        let inter = meta["inter"].as_u64().unwrap() as usize;
        let hidden = meta["hidden"].as_u64().unwrap() as usize;
        if let Ok(ap) = std::env::var("LINGXI_EXPERT_ARENA") {
            // ㊿-P0：未显式给槽数时走显存预算自动分配（env 显式值优先）。
            let slots: usize = if ep_on {
                ep_slots
            } else {
                match std::env::var("LINGXI_ARENA_SLOTS")
                    .ok().and_then(|s| s.parse().ok())
                {
                    Some(v) => v,
                    None => match crate::vram::autotune_arena_slots(&art, &ap, 0, ls_lo, rope_len + 8)? {
                        Some((s, ledger)) => {
                            eprintln!("{ledger}");
                            s
                        }
                        None => 4000,
                    },
                }
            };
            Some(Box::new(crate::arena_source::ArenaExpertSource::new(
                &k,
                &ap,
                slots,
                false,
                // ★ 全量 layer-split：GPU0 这份只服务 `0..ls_lo`（后一半层的槽位
                // 全部留给 GPU1 那份，否则一半的槽白占）。
                if ls_on { 0..ls_lo } else { 0..u32::MAX },
            )?))
        } else {
            // ⚠️ **不允许静默回退**：这条 host LRU ↔ device LRU 的路实测
            // decode **488~714 ms/token**（比 arena 慢 15~20 倍），而且以前毫无提示。
            // 历史上"忘了设 LINGXI_EXPERT_ARENA"就掉进来，被误判成代码回归。
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
            Some(Box::new(crate::cli::common::MmapExpertSource::new(
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
    // ★ **EP**：GPU1 的稠密层缓存——**全部** n_layers 层，逐层 build 到 GPU1。
    // 先预热（`peek` 只读，预热后 `run_layers_ep` 才能每层取到权重）；
    // `build` 闭包来源与 card0 相同（同一产物、同一指标），只是目标设备不同。
    let mut ep_cache1: Option<crate::cli::common::DeviceLayerCache> = None;
    if ep_on {
        anyhow::ensure!(sparse, "LINGXI_EP=1 目前要求稀疏专家路径（LINGXI_SPARSE_EXPERTS≠0）");
        let k1 = ls_k1.as_ref().expect("EP 必须有 GPU1 上下文");
        let build1 = {
            let (ab, mb) = (&art, &meta);
            let hcb = host_cache.clone();
            let t1 = std::time::Instant::now();
            Box::new(move |layer: u32| -> Result<lingxi_model::trunk::LoadedLayer> {
                eprint!(
                    "\r  [ep-load] GPU1 层 {layer}/{n_layers}  {:>6.1}s",
                    t1.elapsed().as_secs_f64()
                );
                let b = crate::cli::common::build_layer(
                    k1, ab, mb, layer, sparse, hcb.as_deref(),
                )?;
                Ok(lingxi_model::trunk::LoadedLayer {
                    weights: b.weights,
                    k_layernorm_host: b.k_layernorm_host,
                })
            })
        };
        let mut c = crate::cli::common::DeviceLayerCache {
            slots: (0..n_layers as usize).map(|_| None).collect(),
            build: build1,
        };
        for i in 0..n_layers {
            lingxi_driver::LayerProvider::layer(&mut c, i)?;
        }
        eprintln!("\r  [ep-load] GPU1 全部 {n_layers} 层稠密权重就绪");
        ep_cache1 = Some(c);
    }
    // ★ EP 专家分配表：**每层一张** `expert -> card`。按「专家槽字节 × 历史命中频次」
    // 贪心均衡（Q4 热 2.6 MB / Q3 冷 1.92 MB），热专家自然交错到两卡。
    // 只构建一次，卡0/卡1 读同一份切片 ⇒ 路由结果不会分叉。
    let ep_assign: Option<Vec<u8>> = if ep_on {
        let src = expert_source.as_deref().expect("EP 必须有专家源");
        Some(lingxi_model::moe::build_ep_expert_assign(
            n_layers,
            shape.num_experts,
            |l, e| src.slot_bytes(l, e),
        ))
    } else {
        None
    };
    let ep_ctx: Option<lingxi_driver::EpCtx> = if ep_on {
        Some(lingxi_driver::EpCtx {
            k1: ls_k1.as_ref().expect("EP 必须有 GPU1 上下文"),
            tables1: ls_tables.as_ref().expect("EP 必须有 GPU1 RoPE 表"),
            experts1: ls_experts.as_deref().expect("EP 必须有 GPU1 专家源"),
            layers1: ep_cache1.as_ref().expect("EP 必须有 GPU1 层缓存"),
            expert_assign: ep_assign.as_deref().expect("EP 必须有分配表"),
            num_experts: shape.num_experts,
        })
    } else {
        None
    };
    if ep_on {
        eprintln!(
            "╔══════════════════════════════════════════════════════════════════════╗\n\
             ║ [EP] 专家并行已启用（LINGXI_EP=1）                                  ║\n\
             ║  · layer-split = 关闭；两卡各跑全部 {n_layers} 层稠密段           ║\n\
             ║  · 专家按「槽字节×命中频次」贪心均衡（Q4 热/Q3 冷交错；每卡槽位 {ep_slots}）║\n\
             ║  · 每层交换 MoE 部分和（t×2560 bf16/单向）；CUDA Graph 已强制关闭   ║\n\
             ╚══════════════════════════════════════════════════════════════════════╝"
        );
    }
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
        ep: ep_ctx.as_ref(),
    };

    // dense 预读：立刻给预读线程喂前两层，之后每层回调里再喂下一层
    if let Some(c) = &host_cache {
        c.prefetch_layer_dense(0);
    }

    let t0 = std::time::Instant::now();
    // sparse：dense 权重显存常驻，prefill 首遍逐层构建（同时后台预读下一层
    // host 字节），之后 decode 全部按引用复用、零 dense H2D；
    // full（对照）：保持每层每步流式现建。
    let mut stream_provider: Option<crate::cli::common::StreamingProvider>;
    let mut cache_provider: Option<crate::cli::common::DeviceLayerCache>;
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
            let b = crate::cli::common::build_layer(dev_k, ab, mb, layer, sparse, hc_for_build.as_deref())?;
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
        cache_provider = Some(crate::cli::common::DeviceLayerCache {
            slots: (0..n_layers as usize).map(|_| None).collect(),
            build,
        });
        cache_provider.as_mut().unwrap()
    } else {
        stream_provider = Some(crate::cli::common::StreamingProvider { cur: None, build });
        stream_provider.as_mut().unwrap()
    };

    // `LINGXI_TALK_SPEC=1`：把 serve 的 MTP 投机环路接到 talk 上，让权威尺子
    // （talk/bench.sh）直接量到投机吞吐。默认关 = 纯自回归不变。
    let talk_spec = std::env::var("LINGXI_TALK_SPEC").map(|v| v == "1").unwrap_or(false);
    let (first, hyper_first) = if talk_spec {
        // 投机要整段 prompt 的 hyper 行（MTP prime 用），走 prefill_with_hyper。
        let (f, h) = seq.prefill_with_hyper(&eng, &prompt_ids, provider)?;
        (f, Some(h))
    } else {
        (seq.prefill(&eng, &prompt_ids, provider)?, None)
    };
    eprintln!("\nprefill 用时 {:.1}s，首 token = {first}", t0.elapsed().as_secs_f64());

    // ══ 双会话相位重叠判决原型══
    //
    // 两个 host 线程各驱动一套 (Sequence, SpecRuntime)：重叠不靠调度器，靠的是
    // "每线程阻塞在自己那条 stream 的同步上" —— A.verify.hi 阻塞在 k1 期间，B 的
    // 草稿/verify.lo（k0）照常推进 ⇒ GPU1 的空转（一轮 ~12 ms）被第二会话填上。
    // 三个前置条件（本块强制或已落实）：
    // 1. **CUDA Graph 全关**：捕获是线程本地的，两会话并发会把对方 kernel 捕进
    // 自己的图 ⇒ 必须 force_disable。
    // 2. **层全部预热**：线程里只能 peek（DeviceLayerCache 的 unsafe Sync 契约）。
    // 3. **ArenaExpertSource 簿记串行化**（op_lock，arena_source/source.rs）：
    // miss 的 victim/insert 是 host 裸可变；scratch 双缓冲的"隔层必同步"余量
    // 是每线程路径内的论证（同一条 k0 FIFO + 每层路由同步），与并发无关。
    // 判据：①两会话各自 token 流与单会话逐位相同；②聚合 ≥ 1.3× 单流。
    if talk_spec && lingxi_model::flag("LINGXI_TALK_SPEC_BATCH") {
        anyhow::ensure!(
            provider.as_cache().is_some(),
            "LINGXI_TALK_SPEC_BATCH 需要 sparse 常驻层缓存（DeviceLayerCache）"
        );
        lingxi_model::graph::force_disable();
        std::env::set_var("LINGXI_NO_GRAPHS", "1");
        for i in 0..n_layers {
            provider.layer(i)?;
        }
        eprintln!("[talk-spec-2] {n_layers} 层预热完，建第二会话…");

        // 第二序列（**同 prompt**：两会话输出都该与单会话逐位相同 ⇒ 最强正确性对拍；
        // 生产形态不同 prompt 只是专家集不重叠，调度行为同构）。
        let mut seq_b = lingxi_driver::Sequence::new_split(
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
        let (first_b, hyper_b) = seq_b.prefill_with_hyper(&eng, &prompt_ids, provider)?;
        anyhow::ensure!(
            first_b == first,
            "同 prompt 两会话首 token 不同（{first} vs {first_b}）"
        );

        let mp = crate::cli::common::mtp_artifact_path();
        // 两份 SpecRuntime（各持一套 MTP 权重/Q4 专家/KV 状态 —— mst 是每会话状态，不能共享）。
        let mut rt_a =
            crate::spec::SpecRuntime::load(&k, ls_k1.as_ref(), &mp, &tables, &endpoints, &shape, rope_len)?;
        let mut rt_b =
            crate::spec::SpecRuntime::load(&k, ls_k1.as_ref(), &mp, &tables, &endpoints, &shape, rope_len)?;
        let kd = rt_a.kd();
        let mut sb_a = seq.spec_prepare(&eng, provider, kd + 1)?;
        let mut sb_b = seq_b.spec_prepare(&eng, provider, kd + 1)?;

        // SAFETY：DeviceLayerCache 已 unsafe impl Sync，契约即"全部层预热后跨线程只读
        // peek"。本分支末尾必然 return，主线程之后不会再碰 provider ⇒ 无实际混叠。
        let layer_cache: &dyn lingxi_driver::LayerCache =
            unsafe { std::mem::transmute(provider.as_cache().unwrap()) };

        let t_dual = std::time::Instant::now();
        // ★ 反相位 PLL（`LINGXI_TALK_SPEC_PHASE=<Δ_ms>`，默认 36；0 = 关）：
        // A 自由跑（master），B（slave）的每轮锚在 **A 的实际轮起点 + Δ** —— A 快 B 跟、
        // A 慢 B 等，相位锁不滑。固定网格（v1）实测会滑出同步（轮长 40~90ms 不均，
        // 一过格就自由漂移 ⇒ 双模式 61~105 乱跳）。调度依据（R/S）：
        // k0 是瓶颈卡（2×38.7ms/轮对），A 的 verify.hi（23.4ms 纯 k1）应与 B 的
        // 草稿+lo（33.4ms 纯 k0）对齐 ⇒ Δ ≈ A 的 pre-hi 段长。
        let offset_ns = std::env::var("LINGXI_TALK_SPEC_PHASE")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(36)
            * 1_000_000;
        let anchor = std::time::Instant::now();
        // A 的轮起点信号：ts = 第 r 轮起点（ns since anchor），ctr = 已开跑的轮数。
        let a_sig = std::sync::Arc::new(ASig {
            ts: std::sync::atomic::AtomicU64::new(0),
            ctr: std::sync::atomic::AtomicU64::new(0),
        });
        // 只让闭包拿走引用（Copy），本体留给后续路径。
        let eng_ref = &eng;
        let (ra, rb) = std::thread::scope(|scp| {
            let h_a = hyper_first.as_ref().expect("talk_spec 必有 hyper");
            let sig_a = std::sync::Arc::clone(&a_sig);
            let ta = scp.spawn(move || {
                spec_dual_session(
                    "A", &mut seq, &mut rt_a, &mut sb_a, h_a, eng_ref, layer_cache, eos, max_new,
                    first, anchor, Some((&sig_a, 0)),
                 )
            });
            let sig_b = std::sync::Arc::clone(&a_sig);
            let tb = scp.spawn(move || {
                spec_dual_session(
                    "B", &mut seq_b, &mut rt_b, &mut sb_b, &hyper_b, eng_ref, layer_cache, eos,
                    max_new, first_b, anchor,
                    Some((&sig_b, offset_ns)),
                )
            });
            (
                ta.join()
                    .unwrap_or_else(|e| panic!("会话 A 线程 panic: {e:?}")),
                tb.join()
                    .unwrap_or_else(|e| panic!("会话 B 线程 panic: {e:?}")),
            )
        });
        let (gen_a, rounds_a, acc_a) = ra?;
        let (gen_b, rounds_b, acc_b) = rb?;
        let wall = t_dual.elapsed().as_secs_f64();
        let tot = gen_a.len() + gen_b.len() + 2; // + 两个首 token
        eprintln!(
            "[talk-spec-2] 聚合：{tot} tok / {wall:.2}s = **{:.1} tok/s**｜\
             A {rounds_a} 轮/接受 {acc_a}，B {rounds_b} 轮/接受 {acc_b}（对照单流 spec_bench.sh）",
            tot as f64 / wall.max(1e-9)
        );
        for (tag, g) in [("A", &gen_a), ("B", &gen_b)] {
            let mut full = Vec::with_capacity(g.len() + 1);
            full.push(first);
            full.extend_from_slice(g);
            eprintln!("-- [{tag}] {} tokens --", full.len());
            println!("[{tag}] {}", tok.decode(&full)?);
        }
        return Ok(());
    }
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
        // ── 投机分支（`LINGXI_TALK_SPEC=1`）：与 serve 同一套 SpecRuntime 调用序 ──
        // prime → spec_prepare → decode_token_with_hyper（拿 h_last）→ round 循环。
        // 关闸逻辑同 serve：连续 3 轮"产出/验证行 < 0.5"切回本函数的自回归循环。
        let mut spec_state: Option<(
            crate::spec::SpecRuntime<'_>,
            lingxi_driver::SpecBuffers,
            cudarc::driver::CudaSlice<half::bf16>,
            std::collections::VecDeque<i64>,
            f64, // AR 基准 ms/tok（decode_token_with_hyper 实测）
        )> = None;
        if talk_spec {
            let mp = crate::cli::common::mtp_artifact_path();
            // 第二张卡（层切分的 GPU1）交给 SpecRuntime：草稿相位它整段空闲，
            // 用来把 lm_head 按 vocab 行二分。
            let mut rt = crate::spec::SpecRuntime::load(
                &k, ls_k1.as_ref(), &mp, &tables, &endpoints, &shape, rope_len,
            )?;
            let mut queue: std::collections::VecDeque<i64> = std::collections::VecDeque::new();
            let mut ok = true;
            let prime_res = (|| -> anyhow::Result<()> {
                rt.reset();
                let emb_p = rt.prompt_emb(&seq.ids)?;
                let _ = rt.prime(
                    hyper_first.as_ref().expect("talk_spec 必有 hyper"),
                    &seq.ids,
                    emb_p,
                )?;
                Ok(())
            })();
            if let Err(e) = prime_res {
                eprintln!("[talk-spec] MTP prime 失败，回退纯自回归: {e:#}");
                ok = false;
            }
            if ok {
                let kd = rt.kd();
                let init_res = (|| -> anyhow::Result<(
                    lingxi_driver::SpecBuffers,
                    cudarc::driver::CudaSlice<half::bf16>,
                    Vec<i64>,
                    f64,
                )> {
                    let sb = seq.spec_prepare(&eng, provider, kd + 1)?;
                    // AR 基准：取**第二步** decode（第一步含一次性冷成本——实测 44.3ms
                    // vs 稳态 24.4ms，会把关闸 ② 的门槛抬高近 2× 而失灵）。
                    let (tok2, _h2) = seq.decode_token_with_hyper(&eng, provider)?;
                    let t_ar = std::time::Instant::now();
                    let (tok3, h_last) = seq.decode_token_with_hyper(&eng, provider)?;
                    let ar_ms = t_ar.elapsed().as_secs_f64() * 1e3;
                    Ok((sb, h_last, vec![tok2, tok3], ar_ms))
                })();
                match init_res {
                    Ok((sb, h_last, toks, ar_ms)) => {
                        eprintln!("[talk-spec] AR 基准 {ar_ms:.1} ms/tok（第二步实测；投机每产出 tok 必须低于它）");
                        queue.extend(toks);
                        spec_state = Some((rt, sb, h_last, queue, ar_ms));
                    }
                    Err(e) => {
                        eprintln!("[talk-spec] 投机初始化失败，回退纯自回归: {e:#}");
                    }
                }
            }
        }
        if let Some((mut rt, mut sb, mut h_last, mut queue, ar_ms)) = spec_state {
            // 专家槽统计：`去重` = 本请求**装入过**的不同 (层, 专家) 数（设备快路只上报 miss，
            // 所以它是"装入去重"，不是"路由去重"）。
            // 判据：它一旦接近该卡槽位总数，就会进入淘汰螺旋，专家段会从 ~0.4 ms/层 涨到 3+ ms/层。
            crate::arena_source::expert_track_reset();
            let mut accepted = 0usize;
            let mut verified = 0usize;
            let mut rounds = 0usize;
            let mut eff: Vec<f64> = Vec::new(); // 每轮 ms/产出tok（墙上时钟）
            let t_gen = std::time::Instant::now();
            while gen_ids.len() < max_new + 1 {
                if let Some(t) = queue.pop_front() {
                    if t == eos {
                        break;
                    }
                    gen_ids.push(t);
                    continue;
                }
                // ★ 2026-10-25：**关闸①已删**。它是个「产出/验证行 < 0.5」
                // 代理判据，原注释自称"安全网（默认不会触发）"；kd=7 下它**数学上就是错的**：
                // 验证 8 行/轮 ⇒ 盈亏平衡接受率 = 3.0/轮，而中文暖机后 1.9/轮、投机比 AR
                // 快 1.6× 也会被判死（实测 42 vs 66 tok/s）。它当初防的 875-token 文档场景
                // 由关闸②（墙上时钟 ms/tok vs AR 基准）精确覆盖——②才是语义正确的判据。
                // `LINGXI_SPEC_NO_GATE=1` 可把关闸②也关掉（诊断）。
                // 关闸 ②（墙上时钟，2026-10-22 实测补）：最近 5 轮的中位 ms/产出tok
                // 高于 AR 基准 15% ⇒ 投机在帮倒忙（典型：代码内容验证批 miss 爆炸，
                // 接受率 4.3/轮、产出/验证行 0.87 很漂亮，但轮均 160ms ⇒ 13.8 tok/s
                // < AR 40.9。产出/行数判据看不见这个，必须用 wall clock）。
                // ⚠️ 前 8 轮是**冷启动**（本次内容专家槽首访 = 真 miss + dense 首转换）：
                // 2026-10-23 实测第 7 轮中位 43.7 vs AR 基准 42.3 就误判，整条回退 AR 后
                // 只有 24.6 tok/s；冷启动是一次性成本，扛过去即回到 59~61。故等热起来，
                // 且留 15% 余量 + 5 轮中位，避免单轮抖动误杀。
                if !lingxi_model::flag("LINGXI_SPEC_NO_GATE") && rounds >= 8 && eff.len() >= 5 {
                    let mut w = eff[eff.len() - 5..].to_vec();
                    w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let med = w[2];
                    if med > ar_ms * 1.15 {
                        eprintln!(
                            "\n[talk-spec] 最近 5 轮中位 {med:.1} ms/tok > AR 基准 {ar_ms:.1}×1.15 ⇒ 投机在帮倒忙，切回自回归"
                        );
                        break;
                    }
                }
                let tr = std::time::Instant::now();
                let r = rt.round(&mut seq, &eng, provider, &mut sb, &mut h_last)?;
                let ms = tr.elapsed().as_secs_f64() * 1e3;
                rounds += 1;
                accepted += r.accepted;
                verified += r.verified;
                eff.push(ms / (r.tokens.len().max(1) as f64));
                eprint!(
                    "\r  [talk-spec] 轮 {rounds:>3}: 提交 {} 枚｜验证 {} 行｜{ms:>6.1} ms/轮｜{:.1} ms/tok        ",
                    r.tokens.len(),
                    r.verified,
                    ms / (r.tokens.len().max(1) as f64),
                );
                use std::io::Write;
                std::io::stderr().flush().ok();
                queue.extend(r.tokens);
                // `LINGXI_EVPROF_TRACE`：每轮 dump 一次——一轮里最后一个主干前向
                // 是验证批（草稿/重放走 MTP 哨兵层 100，在 n_layers 之外被忽略），
                // 所以 trace 行 = 该轮验证批的逐段 GPU 账。
                if let Some(p) = k.evprof.as_ref() {
                    p.trace_token(&k.stream, 0, rounds, ms, eng.n_layers);
                }
                if let Some(kk) = ls_k1.as_ref() {
                    if let Some(p) = kk.evprof.as_ref() {
                        p.trace_token(&kk.stream, 1, rounds, ms, eng.n_layers);
                    }
                }
            }
            // 收尾：关闸/eos 后剩余步数走纯自回归（已完成则空转）。
            while gen_ids.len() < max_new + 1 {
                let t = seq.decode_token(&eng, provider)?;
                if t == eos {
                    break;
                }
                gen_ids.push(t);
            }
            let secs = t_gen.elapsed().as_secs_f64();
            eprintln!(
                "\n[talk-spec] 共 {} tok / {secs:.2}s = {:.1} tok/s｜{rounds} 轮｜接受 {accepted}（{:.2}/轮）",
                gen_ids.len(),
                gen_ids.len() as f64 / secs.max(1e-9),
                accepted as f64 / rounds.max(1) as f64,
            );
            let (eh, em, eu) = crate::arena_source::expert_track_finish();
            eprintln!(
                "  [talk-spec] 专家槽：命中 {eh}／未命中 {em}（{:.2}%）／去重 (层,专家) {eu}",
                100.0 * em as f64 / (eh + em).max(1) as f64
            );
        } else {
        for step in 0..max_new {
            let t1 = std::time::Instant::now();
            let t = seq.decode_token(&eng, provider)?;
            let dt = t1.elapsed().as_secs_f64();
            // `LINGXI_EVPROF_TRACE=<路径>`：每 token 逐段 GPU 时间落盘（JSONL，双卡各一行）。
            if let Some(p) = k.evprof.as_ref() {
                p.trace_token(&k.stream, 0, step, dt * 1e3, eng.n_layers);
            }
            if let Some(kk) = ls_k1.as_ref() {
                if let Some(p) = kk.evprof.as_ref() {
                    p.trace_token(&kk.stream, 1, step, dt * 1e3, eng.n_layers);
                }
            }
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
        } // talk_spec 的 else（纯自回归循环）收尾
    } else {
        eprintln!("首 token 即 eos，结束");
    }

    // `LINGXI_DUMP_IDS=1`：把整条 decode token 流打出来（含 prefill 首 token），
    // 供 EP ⇄ layer-split 的逐 token 对拍（配合 LINGXI_FIXED_IDS 固定输入）。
    if std::env::var("LINGXI_DUMP_IDS").is_ok() {
        eprintln!("[ids] {gen_ids:?}");
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
    Ok(())
}

/// 反相位 PLL 的共享信号：A（master）每轮起点发布 `ts`（ns since anchor）+ 递增 `ctr`；
/// B（slave）等 `ctr ≥ r+1` 后再等到 `ts + Δ`。
struct ASig {
    ts: std::sync::atomic::AtomicU64,
    ctr: std::sync::atomic::AtomicU64,
}

/// 双会话原型（`LINGXI_TALK_SPEC_BATCH=1`）的单会话驱动线程体：与 talk_spec 主循环
/// 同一套调用序（prime → decode×2 拿 ar_ms/h_last → round 循环 → 两道关闸 → AR 收尾）。
/// 层供给走 `PeekProvider`（主线程已全量预热）。只服务判决原型；生产调度器（serve）另案。
///
/// `lock`：A 传 `Some(&sig)`（每轮起点发布信号）；B 传 `Some((&sig, Δ_ns))`
///（轮 r 锚在 A 的第 r 轮起点 + Δ）；`None` = 自由跑（对照）。
#[allow(clippy::too_many_arguments)]
fn spec_dual_session(
    label: &str,
    seq: &mut lingxi_driver::Sequence,
    rt: &mut crate::spec::SpecRuntime<'_>,
    sb: &mut lingxi_driver::SpecBuffers,
    hyper: &cudarc::driver::CudaSlice<half::bf16>,
    eng: &lingxi_driver::Engine<'_>,
    cache: &dyn lingxi_driver::LayerCache,
    eos: i64,
    max_new: usize,
    first: i64,
    anchor: std::time::Instant,
    lock: Option<(&ASig, u64)>,
) -> Result<(Vec<i64>, usize, usize)> {
    let mut provider = crate::cli::common::PeekProvider { cache };
    let mut queue: std::collections::VecDeque<i64> = std::collections::VecDeque::new();
    let mut gen: Vec<i64> = Vec::new();
    rt.reset();
    let emb_p = rt.prompt_emb(&seq.ids)?;
    let _ = rt.prime(hyper, &seq.ids, emb_p)?;
    // AR 基准：取第二步 decode（第一步含一次性冷成本，同 serve 口径）。
    let (tok2, _h2) = seq.decode_token_with_hyper(eng, &mut provider)?;
    let t_ar = std::time::Instant::now();
    let (tok3, mut h_last) = seq.decode_token_with_hyper(eng, &mut provider)?;
    let ar_ms = t_ar.elapsed().as_secs_f64() * 1e3;
    queue.extend([tok2, tok3]);
    let (mut accepted, mut verified, mut rounds) = (0usize, 0usize, 0usize);
    let mut eff: Vec<f64> = Vec::new();
    let t_gen = std::time::Instant::now();
    // `gen` 不含 first（主线程拼回）⇒ 终止条件比主循环少 1。
    while gen.len() < max_new {
        if let Some(t) = queue.pop_front() {
            if t == eos {
                break;
            }
            gen.push(t);
            continue;
        }
        // 关闸 ①（产出/验证行 < 0.5）
        if rounds >= 3 && (accepted + rounds) as f64 / (verified.max(1) as f64) < 0.5 {
            eprintln!("\n[talk-spec-2] {label}: 产出/验证行 < 0.5（{rounds} 轮）⇒ 切回自回归");
            break;
        }
        // 关闸 ②（最近 5 轮中位 ms/tok > AR 基准 ×1.15）
        if rounds >= 8 && eff.len() >= 5 {
            let mut w = eff[eff.len() - 5..].to_vec();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if w[2] > ar_ms * 1.15 {
                eprintln!(
                    "\n[talk-spec-2] {label}: 最近 5 轮中位 {:.1} > AR {ar_ms:.1}×1.15 ⇒ 切回自回归",
                    w[2]
                );
                break;
            }
        }
        // 反相位 PLL：master（A）每轮起点发布信号；slave（B）等 A 的第 r 轮
        // 开跑后再等到 +Δ。A 快 B 跟、A 慢 B 等，锁不滑（固定网格 v1 会滑，实测双模式）。
        if let Some((sig, off_ns)) = lock {
            if off_ns == 0 {
                use std::sync::atomic::Ordering;
                sig.ts.store(anchor.elapsed().as_nanos() as u64, Ordering::Release);
                sig.ctr.store(rounds as u64 + 1, Ordering::Release);
            } else {
                use std::sync::atomic::Ordering;
                while sig.ctr.load(Ordering::Acquire) < rounds as u64 + 1 {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
                let t_a = sig.ts.load(Ordering::Acquire);
                while (anchor.elapsed().as_nanos() as u64) < t_a + off_ns {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            }
        }
        let r = rt.round(seq, eng, &mut provider, sb, &mut h_last)?;
        rounds += 1;
        accepted += r.accepted;
        verified += r.verified;
        eff.push(0.0); // 相位重叠下"每轮墙钟"被另一会话拉伸，ms/tok 对关闸失真，只看轮数结构。
        queue.extend(r.tokens);
    }
    // 收尾：关闸/eos 后剩余步数走纯自回归。
    while gen.len() < max_new {
        let t = seq.decode_token(eng, &mut provider)?;
        if t == eos {
            break;
        }
        gen.push(t);
    }
    let secs = t_gen.elapsed().as_secs_f64();
    eprintln!(
        "[talk-spec-2] 会话 {label}: {} tok / {secs:.2}s = {:.1} tok/s｜{rounds} 轮｜\
         接受 {accepted}（{:.2}/轮）｜首={first}",
        gen.len() + 1,
        (gen.len() + 1) as f64 / secs.max(1e-9),
        accepted as f64 / rounds.max(1) as f64,
    );
    Ok((gen, rounds, accepted))
}
