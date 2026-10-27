//! MTP 子命令（`mtp-pack` / `mtp-accept` / `mtp-dump` / `mtp-load`）。
//! 从 main.rs 搬出（CODE_MAP Phase 1 刀2c，只搬家不改行为）。

use std::sync::Arc;

use anyhow::{Context, Result};
use lingxi_artifact::Artifact;

/// **把 `mtp/*` 张量抽出来单独打包**（~5 GiB 小产物，免碰 252 GB 大文件）。
pub fn pack(args: &[String]) -> Result<()> {
    // **把 `mtp/*` 张量抽出来单独打包**。
    //
    // 背景：生产产物 `qwen4_exp.ninfer` 里**没有** MTP 张量，
    // 而 `/models/lingxi-assets/qwen4_exp.q4dense.mtp2.ninfer`（252 GB）里有 27 个。
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
    Ok(())
}

/// MTP 接受率测量（投机解码判据工具）。
pub fn accept(args: &[String]) -> Result<()> {
    // **接受率测量** —— 投机解码的总闸门（步骤 2）。
    // 主干贪心跑 N 步；每步拿 (h_P, 主干刚预测的 token) 喂 MTP 头出草稿，
    // 与主干**真实的下一个 token** 比对；输出一致率（= 接受率）。
    // MTP 有自己的 KV（1 层），随主干同步推进：prompt 一次性 T 行，之后每步 1 行。
    // 用法: lingxi-bench mtp-accept <main.ninfer> <mtp.ninfer> [prompt] [n]
    let main_path = args
        .get(2)
        .context("用法: lingxi-bench mtp-accept <main.ninfer> <mtp.ninfer> [prompt] [n]")?;
    let mtp_path = args.get(3).context("缺 MTP 产物路径")?;
    let prompt = crate::cli::common::prompt_arg(args.get(4))?;
    let n_new: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);

    let dev: usize =
        std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let k = lingxi_kernels::Kernels::load(dev)?;
    let art = Arc::new(Artifact::open(main_path)?);
    let meta = crate::cli::common::flash_next_meta();
    let shape = crate::cli::common::layer_shape(&meta);
    let tok_bytes = art
        .payload("frontend/tokenizer.json")
        .context("产物缺少 frontend/tokenizer.json")?;
    let tok = lingxi_tokenize::Tokenizer::from_json(&tok_bytes)?;
    let prompt_ids = tok.encode(&prompt)?;
    let prompt_len = prompt_ids.len();
    let endpoints = crate::cli::common::load_endpoints(&k, &art)?;
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
    let tables = crate::cli::common::load_rope_tables(&k, rope_len, shape.attn.rot_dim)?;
    let n_layers = lingxi_model::DIMS.n_layers as u32;
    let arena_path = std::env::var("LINGXI_EXPERT_ARENA")
        .unwrap_or_else(|_| "/dev/shm/qwen4_exp.expert-arena.q4g64.bin".to_string());
    // ★★ **EP（专家并行，`LINGXI_EP=1`）**：两卡各跑**全部** n_layers 层的稠密段，
    // 专家按 id 二分（卡 0 < half，卡 1 ≥ half），每层交换 MoE 部分和。
    // 与 layer-split / moe-split 互斥；跨卡 kernel 不能被 CUDA Graph 捕获 ⇒ 强制关图。
    let ep_on = std::env::var("LINGXI_EP").map(|v| v == "1").unwrap_or(false);
    if ep_on {
        std::env::set_var("LINGXI_NO_GRAPHS", "1");
        lingxi_model::graph::force_disable();
    }
    // ㊿-P0：未显式给槽数时走显存预算自动分配。tune_lo 在 ls_lo 定案前先用
    // env 口径（缺省 24；无 GPU1 时 ls_lo 实为 0，此处按 24 只会**少估**槽数，安全方向）。
    let tune_lo: u32 = std::env::var("LINGXI_LAYER_SPLIT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    // EP：两卡各装**全部** 48 层稠密权重 ⇒ 槽数默认 8000（`LINGXI_EP_SLOTS` 覆盖）。
    let ep_slots: usize = std::env::var("LINGXI_EP_SLOTS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8000);
    let slots: usize = if ep_on {
        ep_slots
    } else {
    match std::env::var("LINGXI_ARENA_SLOTS")
        .ok().and_then(|s| s.parse().ok())
    {
        Some(v) => v,
        None => match crate::vram::autotune_arena_slots(&art, &arena_path, 0, tune_lo, rope_len + 8)? {
            Some((s, ledger)) => {
                eprintln!("{ledger}");
                s
            }
            None => 7000,
        },
    }
    };
    let moe_split = lingxi_model::flag("LINGXI_MOE_SPLIT");
    // ⚠️ 跨卡 + CUDA Graph 是**危险组合**：图是在 GPU0 的 stream 上捕获/实例化的，
    // GPU1 的内核一旦落进捕获区间，重放时就会在"图所属的设备"上执行 ⇒ 错乱。
    // 当前这条路必须关图（`LINGXI_NO_GRAPHS=1`；实测关图本来就快 5%，见）。
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

    // ★★ **全量 layer-split**：
    // 与 `LINGXI_MOE_SPLIT`（只搬 MoE）互斥。这里必须在建 `Sequence` **之前**把
    // 两张卡都建好 —— 因为 `[lo, n_layers)` 的**状态**要直接建在 GPU1 上。
    // ★ P15-㉜（2026-10-17）：**双卡时默认自动开 split**（lo=24）——真实语料实测
    // 20.7 → 54.1 tok/s（+161%：每卡 arena 减半 → 专家 miss 崩塌；bench 英文
    // 高命中语境 −1%）。显式设 `LINGXI_LAYER_SPLIT=0` 可关（回归对照用）。
    let ls_env: Option<u32> = if ep_on {
        Some(0)
    } else {
        std::env::var("LINGXI_LAYER_SPLIT")
            .ok().and_then(|s| s.parse().ok())
    };
    let auto_split = !ep_on && ls_env.is_none() && !moe_split
        && cudarc::driver::CudaContext::new(1).is_ok();
    let ls_lo: u32 = ls_env.unwrap_or(if auto_split { 24 } else { 0 });
    let ls_on = !ep_on && ls_lo > 0 && ls_lo < n_layers;
    anyhow::ensure!(
        !(ep_on && moe_split),
        "LINGXI_EP 与 LINGXI_MOE_SPLIT 互斥（EP 已覆盖跨卡 MoE 的能力）"
    );
    anyhow::ensure!(
        !(ls_on && moe_split),
        "LINGXI_LAYER_SPLIT 与 LINGXI_MOE_SPLIT 互斥（前者已覆盖后者的能力）"
    );
    if ls_on {
        if auto_split {
            // ㊸：不再整体禁图——所录的图全是**单层本地段**（专家段 3 核固定槽），
            // 层⇄设备一一对应（key=layer*8+7 只在那一台设备上出现），捕获/重放都在
            // 本层自己的 stream 上；跨卡拷贝（cross_copy）在层循环之外，不进图。
            // 若实测判据翻车，回退 = 恢复 force_disable 一行。
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
        if ls_on || ep_on { Some(lingxi_kernels::Kernels::load(1)?) } else { None };
    let ls_tables: Option<lingxi_driver::RopeTables> = match ls_k1.as_ref() {
        Some(kk) => Some(crate::cli::common::load_rope_tables(kk, rope_len, shape.attn.rot_dim)?),
        None => None,
    };
    // ㊿-P0：未显式给槽数时走显存预算自动分配（env 显式值优先）。
    // talk 路径 KV 容量 = rope_len+8（prompt+n_new+spec 余量+16）。
    let ls_slots: usize = if ep_on {
        ep_slots
    } else {
    match std::env::var("LINGXI_LAYER_SPLIT_SLOTS")
        .ok().and_then(|s| s.parse().ok())
    {
        Some(v) => v,
        None => match crate::vram::autotune_arena_slots(&art, &arena_path, 1, ls_lo, rope_len + 8)? {
            Some((s, ledger)) => {
                eprintln!("{ledger}");
                s
            }
            None => 9600,
        },
    }
    };  // ★㊲ 工作集≥槽位判决(375/卡层)
    let ls_experts: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> =
        match ls_k1.as_ref() {
            Some(kk) => {
                if ep_on {
                    eprintln!(
                        "[EP] GPU1 上下文就绪；专家 id ≥ {} 由它算，专家槽位 {ep_slots}",
                        shape.num_experts / 2
                    );
                    Some(Box::new(crate::arena_source::ArenaExpertSource::new(
                        kk, &arena_path, ep_slots, false, 0..u32::MAX,
                    )?))
                } else {
                    eprintln!(
                        "[layer-split] GPU1 上下文就绪；层 {ls_lo}..{n_layers} 整层交给它，专家槽位 {ls_slots}"
                    );
                    Some(Box::new(crate::arena_source::ArenaExpertSource::new(
                        kk,
                        &arena_path,
                        ls_slots,
                        false,
                        ls_lo..n_layers,
                    )?))
                }
            }
            None => None,
        };
    let gpu0_layers = 0..if ep_on {
        n_layers
    } else if moe_split {
        layers_hi_env.min(split_lo_env)
    } else if ls_on {
        ls_lo
    } else {
        n_layers
    };
    let mut seq = if ep_on {
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
    let host_cache = {
        let c = crate::host_cache::HostCache::new(Arc::clone(&art));
        c.start_prefetcher();
        c.prefetch_layer_dense(0);
        c
    };
    let expert_source: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> =
        Some(Box::new(crate::arena_source::ArenaExpertSource::new(&k, &arena_path, slots, false, gpu0_layers)?));
    // ★ **跨卡 MoE**：
    // 把**后一半层**的 MoE 交给目前全空的 GPU1 跑 —— 它只有这些层的槽位，可以给到
    // 每层 400+ 个（覆盖 ~100%、miss≈0），而 GPU0 的 9000 槽只需伺候前一半层
    // （每层 375 个 ⇒ ~92%）。跨卡只过 MoE 的输入/输出激活（36 KB×2/层）。
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
        Some(kk) => Some(Box::new(crate::arena_source::ArenaExpertSource::new(
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
        // 初始化期（建 context/开 arena/钉页/传权重）如果有错，会一路潜伏到
        // 后面的某次拷贝才炸、把报错点带偏。这里先清一次，保证后面报错的就是真凶。
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
    // ★ **EP**：GPU1 的稠密层缓存——**全部** n_layers 层，逐层 build 到 GPU1。
    // 先预热（`peek` 只读，`run_layers_ep` 每层都要取权重）。
    let mut ep_cache1: Option<crate::cli::common::DeviceLayerCache> = None;
    if ep_on {
        let k1c = ls_k1.as_ref().expect("EP 必须有 GPU1 上下文");
        let build1 = {
            let (ab, mb) = (&art, &meta);
            let hcb = Some(host_cache.clone());
            let t1 = std::time::Instant::now();
            Box::new(move |layer: u32| -> Result<lingxi_model::trunk::LoadedLayer> {
                eprint!(
                    "\r  [ep-load] GPU1 层 {layer}/{n_layers}  {:>6.1}s",
                    t1.elapsed().as_secs_f64()
                );
                let b = crate::cli::common::build_layer(k1c, ab, mb, layer, true, hcb.as_deref())?;
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
    // ★ EP 专家分配表：每层 `expert -> card`，字节×频次贪心均衡；两卡同一份。
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
        remote: remote.as_ref(),
        k1: ls_k1.as_ref(),
        split_lo: ls_lo,
        tables1: ls_tables.as_ref(),
        experts1: ls_experts.as_deref(),
        ep: ep_ctx.as_ref(),
        // two-pass 镜像只接在 talk 路径（mtp 的 prefill 走既有编排）
        twopass_hi_on0: None,
        twopass_lo_on1: None,
        twopass_rmoe_hi1: None,
        twopass_rmoe_lo0: None,
    };
    let hc_for_build: Option<Arc<crate::host_cache::HostCache>> = Some(host_cache.clone());
    let build = {
        let (kb, k1b, ab, mb) = (&k, ls_k1.as_ref(), &art, &meta);
        let (lsb, lob) = (ls_on, ls_lo);
        Box::new(move |layer: u32| -> Result<lingxi_model::trunk::LoadedLayer> {
            eprint!("\r  [load] 层 {layer}/{n_layers}        ");
            // ★ 全量 layer-split：本层的稠密权重必须直接装到**本层要跑的卡**上
            // （装错了不会报错，只会在第一次 kernel 时 ILLEGAL_ADDRESS 或静默写串）。
            let dev_k = match (lsb, k1b) {
                (true, Some(kk)) if layer >= lob => kk,
                _ => kb,
            };
            let b = crate::cli::common::build_layer(dev_k, ab, mb, layer, true, hc_for_build.as_deref())?;
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
        crate::cli::common::DeviceLayerCache { slots: (0..n_layers as usize).map(|_| None).collect(), build };
    let provider = &mut cache_provider;

    // ── MTP：权重 + 自己的一份 KV 状态 ──
    let mtp_art = Artifact::open(mtp_path)?;
    let g = lingxi_model::mtp::load(&k, &mtp_art)?;
    let mtp_q4_exp = lingxi_model::flag("LINGXI_MTP_Q4_EXPERTS");
    let ml = crate::build_mtp_layer(&k, &mtp_art, mtp_q4_exp)?;
    // MTP 的小 arena：把 512 个专家从 bf16(5.03 GiB) 换成 Q4G64(1.34 GiB)，
    // 给主干腾出专家槽位（真语料 miss 19.4% → ~11.7%）。判据：固定贪心序列必须不变。
    let mtp_arena = if mtp_q4_exp {
        Some(crate::mtp_arena::MtpArena::build(&k, &mtp_art)?)
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
    // （`clone_dtoh` 同步拷贝 + f32 转换 + argmax + top-K + softmax 的 248K 次 `exp`），
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
        // 【实验 LINGXI_MTP_GMIX=1】：末步该用**全局** mixer
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
    // 逐行命中 ⇒ 我们喂 MTP 的 `hyper` 行与位置一一对应，且与主干喂头的那份**同源**；
    // 错位/系统偏 ⇒ 问题就在 (b)（h 的来源或节奏）。
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
    // `hyper`（默认）：48 层后的**残差栈** [T,10240]（Strata 的 window_R_ 就叫 residual）
    // `mixed`：层尾 mixer 的产物 [T,2560] **repeat 到 10240**
    // （llama.cpp 说 h 是"喂 lm_head 的 post-norm hidden"，而本模型无最终 norm）
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
    // 这里命中率高 ⇒ 草稿头本身对，问题在 decode 段的喂法/索引；
    // 这里也≈0 ⇒ 草稿头本身有问题（与 "必是范畴性错误"一致）。
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
        // 要看"不同专家/上下文"的差异必须换多个 prompt 各跑一遍再合。
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
        // ④ 提交：主干状态回卷 + MTP 的 KV cell 用**主干的 h** 重放（否则下一轮建立在错 KV 上）
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
        // reached[j] = 有多少轮**走到了第 j 位**（第 0..j-1 位全接受）
        // hit1[j] = 其中第 j 位 top-1 命中 ⇒ 条件接受率 = hit1[j]/reached[j]
        // hitn[j][..] = 其中真值落在该位草稿 **top-N** 里（N=1,2,3,5,8,10,20）
        // ⚠️ 条件接受率**天然随 j 递减**（越深越难），所以必须报"分母 = reached"，
        // 否则第 3 位之后的数字会被分母缩水而看着像"突然崩掉"。
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
        // 查表本身是对的（`LINGXI_SUFFIX_TRACE=1` 实测提议 `[369,11751,13]` vs 真值同、
        // `[369,44299,13,561,6511,314]` vs 真值同 ⇒ 3/3、3/4、6/6 命中），
        // 但**接受是前缀式的**：把 lookup 草稿挂在 MTP 链后面，只有整条 MTP 链先被接受
        // 才轮得到它 ⇒ 实测"提议 7 枚、接受 0 枚"，而且白付了行钱（47.4 → 44.3 tok/s）。
        // 参考实现用的是 `DraftPolicy`（`spec/draft_policy.hpp`）的**两种策略**，我们都还没做：
        // ① `choose(t_mtp, lookup_k, match)`：匹配够长时 **用 lookup 顶掉整个 MTP 窗**
        // （省掉草稿链的时间，且第一枚就高置信）；
        // ② `chain(t_mtp, p_mtp, k_avail, match)`：**只在 `p_mtp`（整条 MTP 链都成立的概率）
        // 够高时**才追加，且要用"按匹配长度分桶的历史命中率 c"算期望收益 > 成本才追加。
        // ⇒ 下一步：实现 ①（`--suffix-draft` 的本命用法）。
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
            // 默认（不叠 KTS）这里只是 CPU 入队时间 —— 段与段会互相借时间。
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
            // `noh`：自喂的 h（MTP 层的输出）本身不该这么用 ⇒ 每步都沿用第 0 步的 h
            // `nokv`：第 0 步 append 的那个 KV cell 污染了第 1 步的注意力
            // ⇒ 每步先把 MTP 的 QSA 回卷到轮前（只留 pre-round 上下文）
            // 若 `noh` 把第 1 位救回 ~0.95 ⇒ 是 h；若 `nokv` 救回 ⇒ 是 KV；都不动 ⇒ 另找。
            let diag = std::env::var("LINGXI_MTP_CHAIN_DIAG").unwrap_or_default();
            let diag_noh = diag.contains("noh");
            let diag_nokv = diag.contains("nokv");
            // `LINGXI_SPEC_MIN_P`：草稿自己的 top-1 概率低于阈值就**停链**（参考实现 Strata
            // `generate.cpp:9172` 的 `--spec-min-p`）。实测依据（prob1 vs 实际接受）：
            // ≥0.90 → 0.988 ｜ 0.70~0.90 → 0.750 ｜ 0.50~0.70 → 0.250 ｜ <0.50 → 0.278
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
            // **匹配够长时用 lookup 顶掉整个 MTP 窗**。
            // 为什么必须有它：查表草稿挂在 MTP 链**后面**是没用的——接受是前缀式的，
            // 只有整条 MTP 链先被接受才轮得到它（实测"提议 7 枚、接受 0 枚"）。
            // 当上下文里有长重复时（改代码/引号原文/工具 JSON/枚举），
            // **lookup 的第 0 枚置信度远高于 MTP 的自喂链**，此时应该整窗换成 lookup，
            // 还顺带省掉草稿链的时间。
            // 不设 `LINGXI_SUFFIX_CHOOSE` ⇒ 行为完全不变。
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
                    // `put_bf16` 移走了，直接 break 会让循环后的 `put_bf16(h_in)` 报 E0382
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
            // 上一版就是这么把自己的成绩单写错的）
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
            seq.spec_chunk(&eng, &chunk, provider, &mut sb, None)?;
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
            // 为什么要它（2026-10-16）：**提交的 token 只可能是 `argmax[acc]`**，而 `acc` 由
            // 「drafts 是否等于 argmax」决定 ⇒ **只要 `argmax` 本身不变，草稿再怎么变，
            // 提交的 token 流都必须一样**（同一前缀下贪心是唯一的）。
            // 所以两条配置的 token 流一旦分叉，**必然**是某一轮的 `argmax` 变了 ——
            // 这条轨迹就是用来把"哪一轮"钉死的（drafts 的差异只是噪声，不是原因）。
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
            // ⚠️ 上界必须取 `drafts.len`：开了 `LINGXI_SPEC_MIN_P` 后链会被截短，
            // 而 `acc` 可以等于 `drafts.len` ⇒ 用 `0..kd` 会在 `drafts[j]` 越界 panic。
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
            // oracle 链 = 把 h 换成**主干真值**（滞后一格：j=0 用 h_last，j≥1 用 hyper_rows[j-1]），
            // 且 tok 也换成真值 chunk[j]。
            // 判据：oracle 第 1 位回到 ~0.95 而自喂只有 0.57 ⇒ **h 的"身份"错了**
            // （我们的自喂量不是 MTP 训练时期望的那个）；若 oracle 也 ~0.6 ⇒ 草稿头本身
            // 只能预测 1 步，2 步是能力上限（那就不用修接线，只能上多候选）。
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
            // cell 语义 = `(h_j, tok_{j+1})`；最后一个 cell 的 `tok_{keep}` 就是
            // **奖励 token**（提交后 ids 末尾那个），不在 chunk 里。
            if let lingxi_model::layer::AttnState::Qsa(q) = &mut mst.attn {
                q.rewind(mlen0);
            }
            let bonus = *seq.ids.last().unwrap();
            // ★ 2026-10-16：**批量重放**（`LINGXI_MTP_REPLAY_1BY1=1` 退回逐行）。
            // 逐行重放 = `keep` 次 t=1 的 MTP 前向 ⇒ MTP 那一层的权重被读 `keep` 遍。
            // 批成一次 `t=keep` 的前向：权重只读一遍，行边际只剩专家读。
            // **为什么与逐行等价**：MTP 的注意力是因果的、状态已 `rewind(mlen0)`，
            // 一次喂 `keep` 行与逐行喂的每行输出逐位同源；
            // 而**对外输出不受影响**：草稿只影响"哪些行被验"，被采纳的 token 始终是
            // 主干自己的贪心 argmax（主干路径与重放无关）。
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
        // ① **第 1 位就低**（草稿头本身弱） vs **只有深位低**（自喂误差累积）；
        // ② top-1 低但 top-N 高（= 多候选/树草稿有收益）。
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
        // p0 = 自然贪心链 ⇒ [t0, t1, t2, ...]
        // chunk(喂 t0..t5)
        // a2 = 喂完不 commit ⇒ 应等于 p0[6..] （因果性）
        // b2 = 喂完 commit(6) ⇒ 应等于 a2 （重放等价）
        // b0 = 喂完 commit(0)（全拒）⇒ 应等于 p0 （快照/恢复完整）
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
                sq.spec_chunk(eng, &ch, provider, &mut buf, None)?;
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
        // （用自然链当草稿 = 模拟 100% 接受；MTP 的草稿成本 ~1 层 ≈ 1~2 ms/轮，可忽略）
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
            seq.spec_chunk(&eng, &chunk, provider, &mut sb, None)?;
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
    // 判据：主干这一步真实吐的 `tok_next`，是否落在**上一步草稿**的 top-N 里。
    // 为什么要它：`hits/total` 只是 top-1；若 top-8/top-20 明显更高，
    // 说明草稿**分布是对的**（可以做多候选/树草稿），只是 argmax 差一两位。
    const NS_ACC: [usize; 7] = [1, 2, 3, 5, 8, 10, 20];
    let mut acc_hitn = [0usize; 7];
    let mut acc_last_top: Vec<u32> = Vec::new(); // 上一步草稿的 top-N
    let mut acc_rows = 0usize;
    let mut acc_reach = 0usize; // 有草稿可比的总步数
    // 对齐假设：MTP 的 token 分支喂**哪个** token？
    // `next`（默认）：喂主干**刚预测出**的 token（predict 它后面那个）
    // `last`：喂主干**刚吃进**的 token
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
    Ok(())
}

/// MTP 草稿层逐张量 dump（对拍用）。
pub fn dump(args: &[String]) -> Result<()> {
    // **白盒对拍用**：把 MTP stem 的输入 / 权重 / 输出全部落成 f32 —— 交给
    // `tools/golden/mtp_stem_oracle.py` 在 numpy 里复算，**逐步**比中间量。
    // 黑盒已经试不出东西了（记着全部被否的假设），必须白盒。
    // 用法: lingxi-bench mtp-dump <mtp.ninfer> <outdir>
    let path = args.get(2).context("用法: lingxi-bench mtp-dump <mtp.ninfer> <outdir>")?;
    let outdir = args.get(3).context("缺 outdir")?;
    let dev: usize =
        std::env::var("LINGXI_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let k = lingxi_kernels::Kernels::load(dev)?;
    let art = Artifact::open(path)?;
    let g = lingxi_model::mtp::load(&k, &art)?;
    let meta = crate::cli::common::flash_next_meta();
    let shape = crate::cli::common::layer_shape(&meta);
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
    let ml = crate::build_mtp_layer(&k, &art, false)?;
    let rope = crate::cli::common::load_rope_tables(&k, 64, shape.attn.rot_dim)?;
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
    Ok(())
}

/// MTP 产物加载自检。
pub fn load(args: &[String]) -> Result<()> {
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
    let meta = crate::cli::common::flash_next_meta();
    let shape = crate::cli::common::layer_shape(&meta);
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

    let ml = crate::build_mtp_layer(&k, &mtp_art, false)?;
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
    let rope = crate::cli::common::load_rope_tables(&k, 64, shape.attn.rot_dim)?;
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
    let tw = crate::cli::common::load_endpoints(&k, &main_art)?;
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
    Ok(())
}

