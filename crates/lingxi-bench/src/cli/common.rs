//! 共享装配助手：talk/serve/mtp 与 check 命令共用的模型构建函数
//!（CODE_MAP Phase 1 刀2b 从 main.rs 搬入，只搬家不改行为）。

use anyhow::Result;
use lingxi_artifact::Artifact;

/// 读 prompt 参数：`@<路径>` 从文件读（长上下文 prompt 可能几百 KB，超 argv 上限），否则当文本。
pub fn prompt_arg(arg: Option<&String>) -> anyhow::Result<String> {
    let a = arg.cloned().unwrap_or_else(|| "The capital of France is".to_string());
    if let Some(path) = a.strip_prefix('@') {
        Ok(std::fs::read_to_string(path)?)
    } else {
        Ok(a)
    }
}

pub fn flash_next_meta() -> serde_json::Value {
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

/// 装配两端权重：token embedding、末端 hyper-connection mixer、lm_head。
pub fn load_endpoints(
    k: &lingxi_kernels::Kernels,
    art: &Artifact,
) -> Result<lingxi_model::trunk::TrunkWeights> {
    let token_embedding =
        lingxi_model::gpu::upload_any_bf16(k, art, "text/token_embedding")?;
    // `LINGXI_HEAD_Q4=1`：lm_head（1.18 GiB bf16，每 token 全读）在线量化成 W4G64，
    // 省下 ~0.9 ms/token 的 GEMV 时间 + 0.87 GiB 显存（换成专家槽）。
    let output_head_q4 = if lingxi_model::flag_val("LINGXI_HEAD_Q4", true) {
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

/// MTP 草稿产物路径（投机链路）。
///
/// 顺序：`LINGXI_MTP_ARTIFACT` 显式 → 常见部署口径 `/models/...` → 当前目录 `./...`。
/// 为什么要两个口径：不同机器的挂载点叫法不一（容器 `/models` 常是宿主数据盘的 bind）；
/// 写死一个会在换机器/重启后直接报"打开产物失败"。两个都不在时回 `/models` 口径（报错一致）。
pub fn mtp_artifact_path() -> String {
    const CT: &str = "/models/lingxi-assets/qwen4_exp.mtp.fixed.ninfer";
    const LOCAL: &str = "./qwen4_exp.mtp.fixed.ninfer";
    if let Ok(p) = std::env::var("LINGXI_MTP_ARTIFACT") {
        return p;
    }
    for p in [CT, LOCAL] {
        if std::path::Path::new(p).exists() {
            return p.to_string();
        }
    }
    CT.to_string()
}

/// 生成 RoPE 表并上传（device bf16 + host f32 各一份），`len` 需覆盖 prefill+decode。
pub fn load_rope_tables(
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
pub struct BuiltLayer {
    pub weights: lingxi_model::layer::LayerWeights,
    /// indexer 池化 key 的归一化 gamma（host f32；GDN 层为空）
    pub k_layernorm_host: Vec<f32>,
    pub shape: lingxi_model::layer::LayerShape,
    pub kind: lingxi_ids::LayerKind,
}

/// 由 golden meta 构造与层无关的 `LayerShape`（各层共用同一套超参）。
pub fn layer_shape(meta: &serde_json::Value) -> lingxi_model::layer::LayerShape {
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

pub fn build_layer(
    k: &lingxi_kernels::Kernels,
    art: &Artifact,
    meta: &serde_json::Value,
    layer: u32,
    sparse: bool,
    cache: Option<&crate::host_cache::HostCache>,
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
    let dense_q4 = lingxi_model::flag_val("LINGXI_DENSE_Q4", true);
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
    let qsa_q4 = lingxi_model::flag_val("LINGXI_DENSE_Q4_QSA", true);
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
            // 1. `LINGXI_PLE_TABLE` 显式指定（最高优先）；
            // 2. /dev/shm 官方 IQ4_NL 表（26.8GB，量化器 quantize_ple_iq4nl.py 产出）；
            // 3. /dev/shm 官方 FP8 表（51.2GB，extract_ple_fp8.py 产出）；
            // 4. 兜底 lingxi-assets bf16（102GB 机械盘）——**打响亮横幅**（慢 5×，不许静默）。
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
                "/models/lingxi-assets/qwen4_exp.ple-table.bin".to_string()
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
pub struct MmapExpertSource<'a> {
    k: &'a lingxi_kernels::Kernels,
    cache: &'a crate::host_cache::HostCache,
    dev: std::cell::UnsafeCell<crate::host_cache::DeviceExpertLru>,
    scratch: std::cell::UnsafeCell<ExpertScratch>,
}

pub struct ExpertScratch {
    dgu: Option<cudarc::driver::CudaSlice<half::bf16>>,
    ddn: Option<cudarc::driver::CudaSlice<half::bf16>>,
}

// 安全前提：ExpertSource 仅在 driver 单序列的主线程逐层调用，不跨线程共享；
// 返回的 ExpertHits 借用在下一次 load_hits 之前必然用完（单层 MoE 内成立）。
unsafe impl Sync for MmapExpertSource<'_> {}

impl<'a> MmapExpertSource<'a> {
    pub fn new(
        k: &'a lingxi_kernels::Kernels,
        cache: &'a crate::host_cache::HostCache,
        inter: usize,
        hidden: usize,
        dev_gb: usize,
    ) -> Result<Self> {
        Ok(Self {
            k,
            cache,
            dev: std::cell::UnsafeCell::new(crate::host_cache::DeviceExpertLru::new(
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
pub struct StreamingProvider<'a> {
    pub cur: Option<lingxi_model::trunk::LoadedLayer>,
    pub build: Box<dyn FnMut(u32) -> Result<lingxi_model::trunk::LoadedLayer> + 'a>,
}

impl<'a> lingxi_driver::LayerProvider for StreamingProvider<'a> {
    fn layer(&mut self, i: u32) -> Result<&lingxi_model::trunk::LoadedLayer> {
        self.cur = Some((self.build)(i)?);
        Ok(self.cur.as_ref().unwrap())
    }
}

/// 只读 peek 供给（**多会话并发**共用一份已预热的 DeviceLayerCache）：
/// `layer` 退化为 `peek`，要求全部层已预热（契约见 `DeviceLayerCache` 的 `unsafe impl Sync`）。
pub struct PeekProvider<'a> {
    pub cache: &'a dyn lingxi_driver::LayerCache,
}

impl<'a> lingxi_driver::LayerProvider for PeekProvider<'a> {
    fn layer(&mut self, i: u32) -> Result<&lingxi_model::trunk::LoadedLayer> {
        // 复制共享引用出来，返回值的寿命可以覆盖 &'s mut self（协变收窄）。
        let c: &dyn lingxi_driver::LayerCache = self.cache;
        c.peek(i).ok_or_else(|| anyhow::anyhow!("[peek] 层 {i} 未预热"))
    }
}

/// dense 显存常驻供给：每层 dense 权重只在第一次访问时构建一次，之后全部驻留
/// 显存、跨 decode 步按引用复用——decode 每层每步权重逐字节相同，没有任何理由
/// 重新 H2D（注意 `CudaSlice::clone` 会真复制显存，所以只能走引用复用）。
/// sparse 模式下专家张量不在这里，仍由 `ExpertSource` 每步按路由注入。
pub struct DeviceLayerCache<'a> {
    pub slots: Vec<Option<lingxi_model::trunk::LoadedLayer>>,
    pub build: Box<dyn FnMut(u32) -> Result<lingxi_model::trunk::LoadedLayer> + 'a>,
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

/// SAFETY：**只允许在全部层预热之后、且不再调用 `layer` 的前提下跨线程读**。
/// `build` 只在 `&mut self` 的 `layer` 里访问；`peek` 只读 `slots`。
unsafe impl<'a> Sync for DeviceLayerCache<'a> {}

impl<'a> lingxi_driver::LayerCache for DeviceLayerCache<'a> {
    fn peek(&self, i: u32) -> Option<&lingxi_model::trunk::LoadedLayer> {
        self.slots[i as usize].as_ref()
    }
}