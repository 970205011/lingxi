//! `serve <artifact> [port=11434]` — OpenAI 兼容 HTTP（装配与 talk 同一生产链）。
//! 从 main.rs 搬出（CODE_MAP Phase 1 刀2b，只搬家不改行为）。

use std::sync::Arc;

use anyhow::{Context, Result};
use lingxi_artifact::Artifact;

pub fn run(args: &[String]) -> Result<()> {
    // 用法: serve <artifact> [port=11434]   —— OpenAI 兼容 HTTP
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
    let meta = crate::cli::common::flash_next_meta();
    let shape = crate::cli::common::layer_shape(&meta);

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

    let endpoints = crate::cli::common::load_endpoints(&k, &art)?;
    // 容量按 serve_ctx 一次建足：rope/序列状态都不随请求重分配
    let rope_len = serve_ctx + 520;
    let tables = crate::cli::common::load_rope_tables(&k, rope_len, shape.attn.rot_dim)?;

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
            // 单层本地图 split 下保持开启（判据翻车则恢复 force_disable()）
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
        Some(kk) => Some(crate::cli::common::load_rope_tables(kk, rope_len, shape.attn.rot_dim)?),
        None => None,
    };
    let ls_slots: usize = std::env::var("LINGXI_LAYER_SPLIT_SLOTS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(9600);
    let ls_experts: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> =
        match (ls_k1.as_ref(), ls_arena_path.as_deref()) {
            (Some(kk), Some(ap)) => {
                eprintln!("[layer-split] GPU1：层 {ls_lo}..{n_layers}，专家槽位 {ls_slots}");
                Some(Box::new(crate::arena_source::ArenaExpertSource::new(
                    kk, ap, ls_slots, false, ls_lo..n_layers,
                )?))
            }
            (Some(_), None) => anyhow::bail!(
                "LINGXI_LAYER_SPLIT 需要 LINGXI_EXPERT_ARENA（GPU1 那份专家表）"
            ),
            _ => None,
        };
    // Sequence 不再预建——交给 serve 的会话池按需物化（每会话一个，复用/沉没/唤醒）。
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
    let host_cache: Option<Arc<crate::host_cache::HostCache>> = {
        let c = crate::host_cache::HostCache::new(Arc::clone(&art));
        // 默认不起后台预读（与 talk 同口径，理由见 talk.rs：会让 decode 单轮停顿数秒）。
        if std::env::var("LINGXI_PREFETCH").map(|v| v == "1").unwrap_or(false) {
            c.start_prefetcher();
        }
        Some(c)
    };
    let expert_source: Option<Box<dyn lingxi_model::moe::ExpertSource + '_>> = {
        let ap = std::env::var("LINGXI_EXPERT_ARENA").context(
            "serve 需要 LINGXI_EXPERT_ARENA=<arena 文件>（host LRU 路径慢 15-20 倍，不允许静默回退）",
        )?;
        // 未显式给槽数时走显存预算自动分配（env 显式值优先）。
        let slots: usize = match std::env::var("LINGXI_ARENA_SLOTS")
            .ok().and_then(|s| s.parse().ok())
        {
            Some(v) => v,
            None => match crate::vram::autotune_arena_slots(&art, &ap, 0, ls_lo, serve_ctx + 528)? {
                Some((s, ledger)) => {
                    eprintln!("{ledger}");
                    s
                }
                None => 9000,
            },
        };
        Some(Box::new(crate::arena_source::ArenaExpertSource::new(
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
        ep: None,
        // two-pass 镜像只接在 talk 路径（serve 的 prefill 走既有编排）
        twopass_hi_on0: None,
        twopass_lo_on1: None,
        twopass_rmoe_hi1: None,
        twopass_rmoe_lo0: None,
    };
    if let Some(c) = &host_cache {
        c.prefetch_layer_dense(0);
    }

    let t0 = std::time::Instant::now();
    let mut cache_provider: Option<crate::cli::common::DeviceLayerCache>;
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
            let b = crate::cli::common::build_layer(dev_k, ab, mb, layer, true, hc_for_build.as_deref())?;
            if let Some(c) = &hc_for_build {
                if layer + 1 < n_layers {
                    c.prefetch_layer_dense(layer + 1);
                }
            }
            Ok(lingxi_model::trunk::LoadedLayer { weights: b.weights, k_layernorm_host: b.k_layernorm_host })
        })
    };
    let provider: &mut dyn lingxi_driver::LayerProvider = {
        cache_provider = Some(crate::cli::common::DeviceLayerCache {
            slots: (0..n_layers as usize).map(|_| None).collect(),
            build,
        });
        cache_provider.as_mut().unwrap()
    };

    let opts = crate::serve::ServeOpts {
        tok: &tok,
        eos,
        max_ctx: serve_ctx,
        max_new_default,
    };
    // ── 投机（可选）：`LINGXI_SERVE_SPEC=1` + `LINGXI_MTP_ARTIFACT=<path>` ──
    // 只接在 serve 上（产品路径）；talk 仍走纯自回归。
    // ★ 2026-10-27：接入双卡——`ls_k1` 传入 SpecRuntime ⇒ 草稿 head 行二分自动生效
    // （草稿相位 GPU1 整段空闲，拿它算 lm_head 后半；serve 与 talk 同口径 96 tok/s 档）。
    let mut spec_rt: Option<crate::spec::SpecRuntime<'_>> =
        if std::env::var("LINGXI_SERVE_SPEC").map(|v| v == "1").unwrap_or(false) {
            let mp = crate::cli::common::mtp_artifact_path();
            eprintln!("[serve] 投机解码启用：MTP = {mp}（双卡：草稿 head 二分 {}）",
                if ls_k1.is_some() { "开" } else { "关（无第二张卡）" });
            Some(crate::spec::SpecRuntime::load(
                &k, ls_k1.as_ref(), &mp, &tables, &endpoints, &shape, rope_len,
            )?)
        } else {
            None
        };
    crate::serve::run(port, &opts, &eng, &mut mk_seq, provider, &mut spec_rt)?;
    Ok(())
}
