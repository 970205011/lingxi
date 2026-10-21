//! R4 拆分（2026-10-21）：原 serve.rs 3152 行按职责分段，本模块为其中一段。
//! 模块总览见 `serve/mod.rs`。段间共享类型一律 `pub(crate)`。

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::spec::SpecRuntime;

/// 请求体上限（256K token 的纯文本远小于这个；防止异常/恶意请求把内存打爆）。
pub(crate) const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// 请求头总上限。
pub(crate) const MAX_HEADER_BYTES: usize = 64 * 1024;

use super::cache::*;
use super::sessions::*;
use super::proto::*;


pub struct ServeOpts<'a> {
    pub tok: &'a lingxi_tokenize::Tokenizer,
    pub eos: i64,
    /// prompt + 生成的容量上限（Sequence/rope 建局时定死）
    pub max_ctx: usize,
    pub max_new_default: usize,
}

/// **只记性能、不记内容**的 JSONL（数据飞轮 M1，见 BACKLOG §6.2）。
///
/// 应用侧拿 `request_id` 与自己的对话日志 join，就能得到“真实 token 流 + 每 token 耗时”。
/// `LINGXI_SERVE_PERF_LOG` 置空可关；`LINGXI_SERVE_PERF_MAX_MB` 控制轮转（默认 64MB，留 3 份）。
struct PerfLog {
    path: Option<String>,
    max_bytes: u64,
}

impl PerfLog {
    pub(crate) fn from_env() -> Self {
        let path = std::env::var("LINGXI_SERVE_PERF_LOG").unwrap_or_else(|_| "run/perf.jsonl".into());
        let max_mb = std::env::var("LINGXI_SERVE_PERF_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(64);
        if path.is_empty() {
            return Self { path: None, max_bytes: 0 };
        }
        if let Some(dir) = std::path::Path::new(&path).parent() {
            if !dir.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(dir);
            }
        }
        Self { path: Some(path), max_bytes: max_mb.max(1) * 1024 * 1024 }
    }

    pub(crate) fn write_line(&self, line: &str) {
        let Some(path) = &self.path else { return };
        if let Ok(m) = std::fs::metadata(path) {
            if m.len() >= self.max_bytes {
                // 轮转：.1 ← 当前；.2 ← .1；.3 ← .2（只留 3 份）
                for i in (1..3).rev() {
                    let src = format!("{path}.{i}");
                    if std::path::Path::new(&src).exists() {
                        let _ = std::fs::rename(&src, format!("{path}.{}", i + 1));
                    }
                }
                let _ = std::fs::rename(path, format!("{path}.1"));
            }
        }
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// 收到本机 `/shutdown` 后置位；accept 循环看到它就把当前会话落盘再退出。
pub(crate) static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// 请求 id：`<unix_ms>-<seq>`。响应体与性能日志用同一个，方便应用侧 join。
fn next_request_id() -> String {
    use std::sync::atomic::AtomicU64;
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{ms}-{}", SEQ.fetch_add(1, Ordering::Relaxed))
}

/// 服务循环入口（v1.5）：**投机解码开 ⇒ 旧串行路径**（MTP 草稿态绑会话，与并发调度
/// 不兼容，保持原行为/原性能）；**投机关 ⇒ 并发路径**（连接线程 + 引擎线程轮转步进，
/// 多物化会话真并发——无队头阻塞，单流时间片）。两条路径共用 Sessions（多物化语义
/// 对串行路径同样有益：顺序请求切换会话不再沉/醒）。
pub fn run(
    port: u16,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    spec: &mut Option<SpecRuntime<'_>>,
) -> Result<()> {
    let listener =
        TcpListener::bind(("0.0.0.0", port)).with_context(|| format!("[serve] 绑定 :{port} 失败"))?;
    crate::serve::stats::init();
    // R2 检查点统一（2026-10-21）：paged 的不变量在启动期打响亮——dev 检查点已禁用
    // （save/restore 两头都有栅栏），前缀缓存一律落 Disk 档。
    if lingxi_model::flag("LINGXI_QSA_PAGED") {
        eprintln!(
            "[serve] QSA 分页 ON（win={}）：前缀缓存强制 Disk/host 档，dev 检查点已禁用",
            std::env::var("LINGXI_QSA_WIN_TOKENS").unwrap_or_else(|_| "32768".into())
        );
    }
    if spec.is_some() {
        eprintln!(
            "[serve] 就绪 :{port}（串行路径·投机开）  ㊿-P1/v1.5 多会话多物化（常驻上限见 LINGXI_SERVE_MATERIALIZED）  max_ctx={}",
            opts.max_ctx
        );
        run_serial(listener, opts, eng, mk_seq, provider, spec)
    } else {
        eprintln!(
            "[serve] 就绪 :{port}（v1.5 并发路径：连接线程 + 引擎线程轮转步进·单流时间片）  max_ctx={}",
            opts.max_ctx
        );
        run_concurrent(listener, opts, eng, mk_seq, provider)
    }
}

/// 串行路径（原实现不动）：单线程 accept，逐连接同步处理。投机解码只在此路径可用。
fn run_serial(
    listener: TcpListener,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    spec: &mut Option<SpecRuntime<'_>>,
) -> Result<()> {
    let mut cache = PrefillCache::default();
    let mut sessions = Sessions::new(opts.max_ctx);
    let perf = PerfLog::from_env();
    if let Some(p) = &perf.path {
        eprintln!("[serve] 性能日志（不含内容）: {p}");
    }
    for conn in listener.incoming() {
        match conn {
            Ok(s) => {
                if let Err(e) =
                    handle(s, opts, eng, mk_seq, &mut sessions, provider, &mut cache, spec, &perf)
                {
                    eprintln!("[serve] 请求处理失败: {e:#}");
                }
            }
            Err(e) => eprintln!("[serve] accept 失败: {e}"),
        }
        if SHUTDOWN.load(Ordering::Relaxed) {
            eprintln!("[serve] 收到 shutdown：把全部 live 会话落盘…");
            sessions.offload_all_live(eng, &mut cache);
            eprintln!("[serve] 已退出");
            return Ok(());
        }
    }
    Ok(())
}

fn handle(
    mut stream: TcpStream,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    sessions: &mut Sessions,
    provider: &mut dyn lingxi_driver::LayerProvider,
    cache: &mut PrefillCache,
    spec: &mut Option<SpecRuntime<'_>>,
    perf: &PerfLog,
) -> Result<()> {
    stream.set_nodelay(true).ok();
    // 单线程 accept 循环：一个只连不发的连接会把整个服务卡死 ⇒ 等**请求头**必须有超时。
    // ⚠️ 但 body 可能是一大段上下文，30s 会把它掐断（用户实测 24K token 请求被断开）⇒
    //    头读完就放宽到 10min。
    stream.set_read_timeout(Some(std::time::Duration::from_secs(60))).ok();
    stream.set_write_timeout(Some(std::time::Duration::from_secs(120))).ok();
    let mut rdr = BufReader::new(stream.try_clone()?);

    // ── 读请求行 + 头 ──
    let mut line = String::new();
    rdr.read_line(&mut line).context("读请求行")?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let http11 = parts.next().map(|v| v.eq_ignore_ascii_case("HTTP/1.1")).unwrap_or(false);
    let mut content_len = 0usize;
    let mut chunked = false;
    let mut expect_continue = false;
    let mut header_bytes = line.len();
    loop {
        let mut h = String::new();
        rdr.read_line(&mut h)?;
        header_bytes += h.len();
        if header_bytes > MAX_HEADER_BYTES {
            return respond(
                &mut stream,
                431,
                "application/json",
                br#"{"error":"request headers too large"}"#,
            );
        }
        let h = h.trim();
        if h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_len = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = lower.strip_prefix("transfer-encoding:") {
            chunked = v.trim().contains("chunked");
        } else if let Some(v) = lower.strip_prefix("expect:") {
            expect_continue = v.trim().eq_ignore_ascii_case("100-continue");
        }
    }
    // 长上下文：body 读取不再用 60s 的小超时
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(600)))
        .ok();
    if expect_continue {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        stream.flush()?;
    }

    // ── 读 body（支持 Content-Length 与 chunked）──
    if !chunked && content_len > MAX_BODY_BYTES {
        return respond(
            &mut stream,
            413,
            "application/json",
            br#"{"error":"request body too large"}"#,
        );
    }
    let body = if chunked {
        match read_chunked(&mut rdr, MAX_BODY_BYTES) {
            Ok(b) => b,
            Err(e) => {
                return respond(
                    &mut stream,
                    413,
                    "application/json",
                    format!(r#"{{"error":"invalid chunked body: {e}"}}"#).as_bytes(),
                )
            }
        }
    } else {
        let mut body = vec![0u8; content_len];
        std::io::Read::read_exact(&mut rdr, &mut body)?;
        body
    };

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => respond(&mut stream, 200, "application/json", br#"{"ok":true}"#),
        ("GET", "/stats") => respond(
            &mut stream,
            200,
            "application/json",
            crate::serve::stats::json().as_bytes(),
        ),
        ("GET", "/dashboard") => respond(
            &mut stream,
            200,
            "text/html; charset=utf-8",
            crate::serve::stats::dashboard().as_bytes(),
        ),
        ("GET", "/v1/models") => respond(
            &mut stream,
            200,
            "application/json",
            br#"{"object":"list","data":[{"id":"lingxi","object":"model","created":0,"owned_by":"lingxi"}]}"#,
        ),
        // 优雅退出：只允许本机调用（down.sh 用）；accept 循环随后会把当前会话落盘
        ("GET", "/shutdown") | ("POST", "/shutdown") => {
            let local = stream
                .peer_addr()
                .map(|a| a.ip().is_loopback())
                .unwrap_or(false);
            if !local {
                return respond(&mut stream, 403, "application/json", br#"{"error":"local only"}"#);
            }
            SHUTDOWN.store(true, Ordering::Relaxed);
            respond(&mut stream, 200, "application/json", br#"{"ok":true}"#)
        }
        ("POST", "/v1/chat/completions") => {
            chat(&mut stream, &body, http11, opts, eng, mk_seq, sessions, provider, cache, spec, perf)
        }
        _ => respond(&mut stream, 404, "application/json", br#"{"error":"not found"}"#),
    }
}

#[allow(clippy::too_many_arguments)]

fn chat(
    stream: &mut TcpStream,
    body: &[u8],
    http11: bool,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    sessions: &mut Sessions,
    provider: &mut dyn lingxi_driver::LayerProvider,
    cache: &mut PrefillCache,
    spec: &mut Option<SpecRuntime<'_>>,
    perf: &PerfLog,
) -> Result<()> {
    let v: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return respond(
                stream,
                400,
                "application/json",
                format!(r#"{{"error":"invalid json: {e}"}}"#).as_bytes(),
            )
        }
    };
    let stream_resp = v["stream"].as_bool().unwrap_or(false);
    let max_tokens = v["max_tokens"]
        .as_u64()
        .or_else(|| v["max_completion_tokens"].as_u64())
        .unwrap_or(opts.max_new_default as u64) as usize;
    let messages = match v["messages"].as_array() {
        Some(m) if !m.is_empty() => m.clone(),
        _ => {
            return respond(
                stream,
                400,
                "application/json",
                br#"{"error":"messages must be a non-empty array"}"#,
            )
        }
    };

    // ── 模板开关（与 vLLM / 官方 chat_template.jinja 同口径）──
    let thinking_kw = v["chat_template_kwargs"]["enable_thinking"].as_bool();
    let cfg = ChatCfg {
        enable_thinking: v["enable_thinking"].as_bool().or(thinking_kw).unwrap_or(true),
        preserve_thinking: v["preserve_thinking"].as_bool().unwrap_or(true),
        effort: match ReasoningEffort::parse(v["reasoning_effort"].as_str().unwrap_or("xhigh")) {
            Some(e) => e,
            None => {
                return respond(
                    stream,
                    400,
                    "application/json",
                    br#"{"error":"reasoning_effort must be one of xhigh|medium|low"}"#,
                )
            }
        },
    };
    let tools: Option<Vec<serde_json::Value>> = v["tools"].as_array().cloned();
    let rid = next_request_id();
    let stops: Vec<String> = match &v["stop"] {
        serde_json::Value::String(s) if !s.is_empty() => vec![s.clone()],
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };

    // ── chat 模板（严格对齐产物 chat_template.jinja）──
    let (prompt, boundary_chars) = build_prompt(&messages, tools.as_ref(), cfg);

    let ids = match opts.tok.encode(&prompt) {
        Ok(x) => x,
        Err(e) => {
            return respond(
                stream,
                400,
                "application/json",
                format!(r#"{{"error":"tokenize 失败: {e}"}}"#).as_bytes(),
            )
        }
    };
    let prompt_tokens = ids.len();
    // 消息边界的 token 前缀（只在它确实是 ids 的前缀时启用）
    let boundary_ids: Option<usize> = if boundary_chars > 0 && boundary_chars < prompt.len() {
        match opts.tok.encode(&prompt[..boundary_chars]) {
            Ok(b) if !b.is_empty() && b.len() < ids.len() && ids[..b.len()] == b[..] => Some(b.len()),
            _ => None,
        }
    } else {
        None
    };
    if prompt_tokens + 1 >= opts.max_ctx {
        return respond(
            stream,
            400,
            "application/json",
            format!(
                r#"{{"error":"prompt {} tokens + 生成超过容量 {}（调大 LINGXI_SERVE_MAX_CTX 重启）"}}"#,
                prompt_tokens, opts.max_ctx
            )
            .as_bytes(),
        );
    }
    // 生成 token 数不能超过剩余容量，否则 decode 会写出 KV 容量。
    let max_tokens = max_tokens.min(opts.max_ctx.saturating_sub(prompt_tokens + 1));

    // ── ㊿-P1 会话驻留：按 session_id 物化/切换（沉没的秒级唤醒；不显式传则共享 "default"）──
    let sid = v["session_id"]
        .as_str()
        .or_else(|| v["user"].as_str())
        .unwrap_or("default")
        .to_string();
    let need = (prompt_tokens + max_tokens + 64).min(opts.max_ctx);
    if let Err(e) = sessions.ensure(&sid, need, eng, mk_seq, cache, &[]) {
        return respond(
            stream,
            429,
            "application/json",
            format!(r#"{{"error":"会话准入失败: {e}"}}"#).as_bytes(),
        );
    }
    let Some(seq) = sessions.current_mut() else {
        return respond(stream, 500, "application/json", br#"{"error":"session materialize failed"}"#);
    };

    // **先把 200 头发出去**，并在 prefill 期间每 5s 写一次保活字节。
    // 否则冷 prefill（24K token 243s、256K 57min）期间零字节，客户端/网关必然超时断连。
    // 流式走 SSE 注释行；非流式（HTTP/1.1）走 chunked 的空白块。
    let chunked_resp = !stream_resp && http11;
    let keep = if stream_resp {
        sse_headers(stream, &rid)?;
        Some(KeepAlive::start(stream.try_clone()?))
    } else if chunked_resp {
        chunked_headers(stream)?;
        Some(KeepAlive::start_chunked(stream.try_clone()?))
    } else {
        None
    };

    // ── 智能前缀复用（C3）：最长匹配检查点 + 消息边界检查点 ──
    let cache_on = prefix_cache_enabled();
    // **按请求选路**：只有完全没有前缀可复用的请求才开 MTP。
    // 有缓存命中时走自回归——MTP 的草稿 KV 还没纳入检查点，混用会算错；
    // 这样冷请求吃投机提速，热请求吃前缀缓存的 TTFT≈0。
    let pre_reuse = if cache_on {
        cache.best(&ids).map(|(_, l)| l).unwrap_or(0)
    } else {
        0
    };
    let spec_on = spec.is_some() && pre_reuse == 0;
    // 本请求的专家统计（引擎内部；应用侧拿不到）
    crate::arena_source::expert_track_reset();
    let t0 = Instant::now();
    let mut hyper_p: Option<cudarc::driver::CudaSlice<half::bf16>> = None;
    let (first, restored, hit_src) = match run_prefill(
        seq, eng, provider, &ids, boundary_ids, cache, cache_on, spec_on, &mut hyper_p, true,
    ) {
        Ok(v) => v,
        Err(e) => {
            // 兜底：检查点/续跑出任何问题都退回“reset + 全量 prefill”，不把错误抛给用户
            eprintln!("[serve] 前缀续跑失败（回退全量）: {e:#}");
            cache.misses += 1;
            match run_prefill(
                seq, eng, provider, &ids, boundary_ids, cache, cache_on, spec_on, &mut hyper_p,
                false,
            ) {
                Ok(v) => v,
                Err(e) => {
                    if stream_resp {
                        // 200/SSE 头已经发出去了，只能走 SSE 错误事件
                        return sse_fail(stream, &format!("prefill 失败: {e}"));
                    }
                    let msg = format!(r#"{{"error":"prefill 失败: {e}"}}"#);
                    if chunked_resp {
                        return respond_chunked(stream, msg.as_bytes());
                    }
                    return respond(stream, 500, "application/json", msg.as_bytes());
                }
            }
        }
    };
    let ttft = t0.elapsed();
    eprintln!(
        "[serve] prompt={prompt_tokens} tok  前缀复用={restored} tok  TTFT={:.2}s  首token={first}",
        ttft.as_secs_f64()
    );
    // “prompt 刚喂完”检查点：下一轮**原样重发同一 prompt** 时命中（生成阶段的 token 不在里面）
    if cache_on && ckpt_level() >= 1 {
        save_ckpt_auto(seq, eng, prompt_tokens, CkptKind::Prompt, first, cache, "prompt");
    }

    // ── 投机：prime MTP（喂整段 prompt 的 h）→ 先 decode 一次拿 h_last ──
    let mut spec_gen: Option<SpecGen<'_, '_>> = None;
    let t_spec0 = Instant::now();
    if let (Some(rt), Some(hp)) = (spec.as_mut(), hyper_p) {
        let primed = (|| -> Result<()> {
            rt.reset();
            let emb_p = rt.prompt_emb(&seq.ids)?;
            let _ = rt.prime(&hp, &seq.ids, emb_p)?;
            Ok(())
        })();
        if let Err(e) = primed {
            eprintln!("[serve] MTP prime 失败，回退纯自回归: {e:#}");
        } else {
            let kd = rt.kd();
            match seq
                .spec_prepare(eng, provider, kd + 1)
                .and_then(|sb| Ok((sb, seq.decode_token_with_hyper(eng, provider)?)))
            {
                Ok((sb, (tok2, h_last))) => {
                    // `first` 由生成器开头直接发（与纯自回归同口径）；队列里只放它之后的 token
                    let mut queue = VecDeque::new();
                    queue.push_back(tok2);
                    spec_gen =
                        Some(SpecGen { rt, sb, h_last, queue, accepted: 0, rounds: 0, verified: 0, disabled: false });
                }
                Err(e) => eprintln!("[serve] 投机初始化失败，回退纯自回归: {e:#}"),
            }
        }
    }

    if spec_gen.is_some() {
        eprintln!("[serve] MTP prime+prepare {:.2}s", t_spec0.elapsed().as_secs_f64());
    }
    // prefill 结束：停掉 keep-alive，生成阶段只由主线程写
    drop(keep);
    let t_gen = Instant::now();
    let r = if stream_resp {
        sse_generate(stream, opts, cfg, &rid, stops.clone(), seq, eng, provider, first,
                     prompt_tokens, max_tokens, spec_gen.as_mut())
    } else {
        batch_generate(stream, opts, cfg, chunked_resp, &rid, stops.clone(), seq, eng, provider,
                       first, prompt_tokens, max_tokens, spec_gen.as_mut())
    };
    if let Some(g) = &spec_gen {
        eprintln!(
            "[serve] 投机 {} 轮｜接受 {} 枚｜{:.2}/轮｜生成 {:.2}s{}",
            g.rounds,
            g.accepted,
            if g.rounds > 0 { g.accepted as f64 / g.rounds as f64 } else { 0.0 },
            t_gen.elapsed().as_secs_f64(),
            if g.disabled { "｜已切回自回归" } else { "" }
        );
    }
    // ── 引擎侧日志：**只记应用拿不到的内部量** ──
    // 标准 OpenAI 字段（prompt/completion tokens、finish_reason、TTFT、decode 速度）
    // 应用侧从响应和 SSE 时间戳就能自己算，这里不再重复。
    {
        let (eh, em, eu) = crate::arena_source::expert_track_finish();
        let (sr, sa, sv, sdis) = match spec_gen.as_ref() {
            Some(g) => (g.rounds, g.accepted, g.verified, g.disabled),
            None => (0, 0, 0, false),
        };
        crate::serve::stats::record(
            seq.ids.len().saturating_sub(prompt_tokens),
            t_gen.elapsed().as_secs_f64(),
            restored > 0,
            eh,
            em,
            sr as u64,
            sa as u64,
        );
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let line = serde_json::json!({
            "ts": ts,
            "request_id": rid,
            "prefix_hit": restored,
            "hit_src": hit_src,
            "spec_rounds": sr,
            "spec_accepted": sa,
            "spec_verified": sv,
            "spec_disabled": sdis,
            "expert_hit": eh,
            "expert_miss": em,
            "expert_unique": eu,
            "cache_mb": (cache.cache_bytes as f64 / (1024.0 * 1024.0) * 10.0).round() / 10.0
        });
        perf.write_line(&(line.to_string() + "\n"));
    }
    // 生成结束：最后一个 token 是"已产生但没喂进 KV"的 ⇒ 去掉，存一份**末尾检查点**
    //（下一轮完全相同的请求可直接命中；投机模式不写缓存）。
    if cache_on {
        // 最后一个“已产生但没喂”的 token 就是这份检查点的 next
        let nxt = *seq.ids.last().unwrap_or(&0);
        seq.truncate_unfed();
        if ckpt_level() >= 2 {
            save_ckpt_auto(seq, eng, seq.ids.len(), CkptKind::End, nxt, cache, "末尾");
        }
        // 设备 KV 现在正好对应 seq.ids（下次判断 Dev 条目是否还有效就靠它）
        cache.device_ids = seq.ids.clone();
    }
    r
}

fn run_prefill(
    seq: &mut lingxi_driver::Sequence,
    eng: &lingxi_driver::Engine<'_>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    ids: &[i64],
    boundary: Option<usize>,
    cache: &mut PrefillCache,
    cache_on: bool,
    spec_on: bool,
    hyper_p: &mut Option<cudarc::driver::CudaSlice<half::bf16>>,
    allow_reuse: bool,
) -> Result<(i64, usize, &'static str)> {
    let mut restored = 0usize;
    let mut hit = "";
    let mut best: Option<(usize, usize)> = None;
    if allow_reuse && cache_on {
        best = cache.best(ids);
        // 没命中（或命中的是别的会话的 host 缓存）⇒ 先把设备上当前会话整段换出到内存，
        // 别让它一直占着显存；下次再命中时拉回来。
        let need_offload = match best {
            None => true,
            Some((ci, _)) => matches!(cache.entries[ci].store, CkptStore::Disk(_)),
        };
        if need_offload {
            if let Err(e) = cache.offload_resident(eng, seq) {
                eprintln!("[serve] 缓存换出失败（忽略）: {e:#}");
            }
            // 换出会删/加条目 ⇒ 之前的下标作废，必须重新找
            best = cache.best(ids);
        }
        if best.is_none() {
            cache.misses += 1;
        }
    }
    seq.reset(eng)?;
    if let Some((ci, n)) = best {
        let nxt = cache.next_at(&cache.entries[ci], n);
        let full = n == ids.len();
        let r = match &cache.entries[ci].store {
            CkptStore::Dev(cp) => seq.restore_checkpoint(eng, cp),
            CkptStore::Disk(path) => match lingxi_driver::HostCheckpoint::read_from(path) {
                Ok(cp) => seq.restore_host_checkpoint_at(eng, &cp, n),
                Err(e) => Err(e),
            },
        };
        match r {
            Ok(()) => {
                // 设备 KV 现在正好对应这份检查点（next token 还没喂进去）
                cache.device_ids = cache.entries[ci].ids[..n].to_vec();
                cache.hits += 1;
                cache.entries[ci].used = cache.tick();
                if full {
                    // **完全命中**：检查点里已经存了这个前缀的 next token，恢复状态后直接返回
                    hit = store_kind_name(&cache.entries[ci]);
                    seq.ids.push(nxt);
                    return Ok((nxt, n, hit));
                }
                hit = store_kind_name(&cache.entries[ci]);
                restored = n;
            }
            Err(e) => {
                eprintln!("[serve] 检查点恢复失败（回退全量）: {e:#}");
                seq.reset(eng)?;
                cache.misses += 1;
                cache.entries.remove(ci);
            }
        }
    }
    // 消息边界检查点：只有“从更早的位置续跑”才有意义。
    // ⚠️ spec_on 时跳过：投机要 `prefill_with_hyper` 在**空序列**上整段喂（收全段 hyper），
    //    先喂边界前缀会把 seq 弄非空 → assert 炸（2026-10-20 v1.5 验证实测命中）。
    let boundary = if allow_reuse && cache_on && !spec_on {
        boundary.filter(|&b| b > restored && b < ids.len())
    } else {
        None
    };
    if let Some(b) = boundary {
        let mut nxt: Option<i64> = None;
        if seq.ids.len() < b {
            // 先喂到边界；返回的 token 就是这个前缀的下一个 token（存进检查点，完全命中时直接复用）
            nxt = Some(if seq.ids.is_empty() {
                seq.prefill(eng, &ids[..b], provider)?
            } else {
                seq.prefill_append(eng, &ids[..b], provider)?
            });
            seq.truncate_unfed();
        }
        if let (Some(nxt), true) = (nxt, seq.ids.len() == b && ckpt_level() >= 3) {
            save_ckpt_auto(seq, eng, b, CkptKind::Boundary, nxt, cache, "边界");
        }
    }
    // 投机要求从空序列整段 prefill（要收集每行 hyper）；一旦命中过检查点就退回普通路径，
    // 调用方看到 hyper_p=None 会自动跳过 MTP。
    let first = if spec_on && restored == 0 {
        let (t, h) = seq.prefill_with_hyper(eng, ids, provider)?;
        *hyper_p = Some(h);
        t
    } else if seq.ids.is_empty() {
        seq.prefill(eng, ids, provider)?
    } else {
        seq.prefill_append(eng, ids, provider)?
    };
    Ok((first, restored, hit))
}

fn sse_generate(
    stream: &mut TcpStream,
    opts: &ServeOpts<'_>,
    cfg: ChatCfg,
    rid: &str,
    stops: Vec<String>,
    seq: &mut lingxi_driver::Sequence,
    eng: &lingxi_driver::Engine<'_>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    first: i64,
    prompt_tokens: usize,
    max_tokens: usize,
    mut spec: Option<&mut SpecGen<'_, '_>>,
) -> Result<()> {
    // 200 头与 role 块已在 prefill 之前发过（见 `sse_headers`）
    let mut n = 0usize;
    let mut finish = "length";
    let mut cur = first;
    let mut detok = Detok::new();
    let mut split = StreamSplitter::new(cfg.enable_thinking);
    let mut stop = StopFilter::new(stops);
    let mut call_idx = 0usize;
    let mut stopped = false;
    loop {
        if cur == opts.eos {
            finish = "stop";
            break;
        }
        n += 1;
        let text = detok.push(opts.tok, cur);
        let mut events = Vec::new();
        split.push(&text, &mut events);
        stopped = write_events(stream, &mut events, &mut call_idx, rid, &mut stop)?;
        if stopped {
            finish = "stop";
            break;
        }
        if n >= max_tokens {
            break;
        }
        cur = match spec.as_mut() {
            Some(g) => g.next(seq, eng, provider)?,
            None => seq.decode_token(eng, provider)?,
        };
    }
    let mut tail = Vec::new();
    split.flush(&mut tail);
    if !stopped {
        stopped = write_events(stream, &mut tail, &mut call_idx, rid, &mut stop)?;
    }
    if let Some(rest) = stop.finish() {
        stream.write_all(sse_chunk(rid, "content", &rest).as_bytes())?;
        stream.flush()?;
    }
    if stopped {
        finish = "stop";
    } else if call_idx > 0 {
        finish = "tool_calls";
    }

    let usage = serde_json::json!({
        "id": rid,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "lingxi",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n,
            "total_tokens": prompt_tokens + n
        }
    });
    stream.write_all(format!("data: {usage}\n\n").as_bytes())?;
    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()?;
    eprintln!("[serve] 完成 {n} tok  finish={finish}");
    Ok(())
}

/// 非流式：攒完全文一次返回。
#[allow(clippy::too_many_arguments)]

fn batch_generate(
    stream: &mut TcpStream,
    opts: &ServeOpts<'_>,
    cfg: ChatCfg,
    chunked: bool,
    rid: &str,
    stops: Vec<String>,
    seq: &mut lingxi_driver::Sequence,
    eng: &lingxi_driver::Engine<'_>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    first: i64,
    prompt_tokens: usize,
    max_tokens: usize,
    mut spec: Option<&mut SpecGen<'_, '_>>,
) -> Result<()> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut detok = Detok::new();
    let mut split = StreamSplitter::new(cfg.enable_thinking);
    let mut stop = StopFilter::new(stops);
    let mut n = 0usize;
    let mut finish = "length";
    let mut stopped = false;
    let mut cur = first;
    loop {
        if cur == opts.eos {
            finish = "stop";
            break;
        }
        n += 1;
        let text = detok.push(opts.tok, cur);
        let mut events = Vec::new();
        split.push(&text, &mut events);
        stopped =
            collect_events(&mut events, &mut content, &mut reasoning, &mut tool_calls, &mut stop);
        if stopped {
            finish = "stop";
            break;
        }
        if n >= max_tokens {
            break;
        }
        cur = match spec.as_mut() {
            Some(g) => g.next(seq, eng, provider)?,
            None => seq.decode_token(eng, provider)?,
        };
    }
    let mut tail = Vec::new();
    split.flush(&mut tail);
    if !stopped {
        stopped =
            collect_events(&mut tail, &mut content, &mut reasoning, &mut tool_calls, &mut stop);
    }
    if let Some(rest) = stop.finish() {
        content.push_str(&rest);
    }
    if stopped {
        finish = "stop";
    } else if !tool_calls.is_empty() {
        finish = "tool_calls";
    }
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning
    });
    if !tool_calls.is_empty() {
        message["tool_calls"] = serde_json::Value::Array(tool_calls);
    }
    let body = serde_json::json!({
        "id": rid,
        "object": "chat.completion",
        "created": 0,
        "model": "lingxi",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n,
            "total_tokens": prompt_tokens + n
        }
    })
    .to_string();
    if chunked {
        respond_chunked(stream, body.as_bytes())?;
    } else {
        respond(stream, 200, "application/json", body.as_bytes())?;
    }
    eprintln!("[serve] 完成 {n} tok  finish={finish}");
    Ok(())
}

/// 引擎线程 → 连接线程 的事件。
enum Ev {
    /// 早期非流式响应（状态行 + body；头未发过才可用）。
    Early { code: u16, body: String },
    /// 流式：先写 200/SSE 头 + role 块；之后 `Bytes` 全是 SSE 帧。
    HeadersSse { rid: String },
    /// 非流式 HTTP/1.1：chunked 头；之后 `FinalBody` 走 chunked 帧。
    HeadersChunked,
    /// 已编码好的字节（SSE 帧）。
    Bytes(Vec<u8>),
    /// 非流式收尾：整段 body（chunked 或 HTTP/1.0 整体响应）。
    FinalBody { body: String },
    /// 请求结束（连接线程收尾退出）。
    Done,
}

/// 连接线程交给引擎线程的请求（JSON 已在连接线程解析；早退错误直接本地回）。
struct Req {
    v: serde_json::Value,
    http11: bool,
    tx: std::sync::mpsc::Sender<Ev>,
}

impl Req {
    pub(crate) fn sid_of(v: &serde_json::Value) -> String {
        v["session_id"]
            .as_str()
            .or_else(|| v["user"].as_str())
            .unwrap_or("default")
            .to_string()
    }
    pub(crate) fn sid(&self) -> String {
        Self::sid_of(&self.v)
    }
}

fn send_ev(tx: &std::sync::mpsc::Sender<Ev>, ev: Ev) {
    let _ = tx.send(ev); // 客户端断连则静默丢弃
}

/// 连接线程主体：读请求 → 入队 → 转发事件到 socket。
fn conn_loop(stream: TcpStream, opts: &ServeOpts<'_>, req_tx: std::sync::mpsc::Sender<Req>) {
    if let Err(e) = conn_loop_inner(stream, opts, req_tx) {
        eprintln!("[serve] 连接处理失败: {e:#}");
    }
}

fn conn_loop_inner(
    mut stream: TcpStream,
    _opts: &ServeOpts<'_>,
    req_tx: std::sync::mpsc::Sender<Req>,
) -> Result<()> {
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(std::time::Duration::from_secs(60))).ok();
    stream.set_write_timeout(Some(std::time::Duration::from_secs(120))).ok();
    let mut rdr = BufReader::new(stream.try_clone()?);

    // ── 读请求行 + 头（与串行路径同口径）──
    let mut line = String::new();
    rdr.read_line(&mut line).context("读请求行")?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let http11 = parts.next().map(|v| v.eq_ignore_ascii_case("HTTP/1.1")).unwrap_or(false);
    let mut content_len = 0usize;
    let mut chunked = false;
    let mut expect_continue = false;
    let mut header_bytes = line.len();
    loop {
        let mut h = String::new();
        rdr.read_line(&mut h)?;
        header_bytes += h.len();
        if header_bytes > MAX_HEADER_BYTES {
            return respond(&mut stream, 431, "application/json", br#"{"error":"request headers too large"}"#);
        }
        let h = h.trim();
        if h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_len = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = lower.strip_prefix("transfer-encoding:") {
            chunked = v.trim().contains("chunked");
        } else if let Some(v) = lower.strip_prefix("expect:") {
            expect_continue = v.trim().eq_ignore_ascii_case("100-continue");
        }
    }
    stream.set_read_timeout(Some(std::time::Duration::from_secs(600))).ok();
    if expect_continue {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        stream.flush()?;
    }
    if !chunked && content_len > MAX_BODY_BYTES {
        return respond(&mut stream, 413, "application/json", br#"{"error":"request body too large"}"#);
    }
    let body = if chunked {
        match read_chunked(&mut rdr, MAX_BODY_BYTES) {
            Ok(b) => b,
            Err(e) => {
                return respond(&mut stream, 413, "application/json",
                    format!(r#"{{"error":"invalid chunked body: {e}"}}"#).as_bytes())
            }
        }
    } else {
        let mut body = vec![0u8; content_len];
        std::io::Read::read_exact(&mut rdr, &mut body)?;
        body
    };

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => respond(&mut stream, 200, "application/json", br#"{"ok":true}"#),
        ("GET", "/stats") => respond(
            &mut stream,
            200,
            "application/json",
            crate::serve::stats::json().as_bytes(),
        ),
        ("GET", "/dashboard") => respond(
            &mut stream,
            200,
            "text/html; charset=utf-8",
            crate::serve::stats::dashboard().as_bytes(),
        ),
        ("GET", "/v1/models") => respond(
            &mut stream, 200, "application/json",
            br#"{"object":"list","data":[{"id":"lingxi","object":"model","created":0,"owned_by":"lingxi"}]}"#,
        ),
        ("GET", "/shutdown") | ("POST", "/shutdown") => {
            let local = stream.peer_addr().map(|a| a.ip().is_loopback()).unwrap_or(false);
            if !local {
                return respond(&mut stream, 403, "application/json", br#"{"error":"local only"}"#);
            }
            SHUTDOWN.store(true, Ordering::Relaxed);
            respond(&mut stream, 200, "application/json", br#"{"ok":true}"#)
        }
        ("POST", "/v1/chat/completions") => {
            let v: serde_json::Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => {
                    return respond(&mut stream, 400, "application/json",
                        format!(r#"{{"error":"invalid json: {e}"}}"#).as_bytes())
                }
            };
            let (tx, rx) = std::sync::mpsc::channel::<Ev>();
            if req_tx.send(Req { v, http11, tx }).is_err() {
                return respond(&mut stream, 503, "application/json", br#"{"error":"engine gone"}"#);
            }
            // 转发事件 → socket
            let mut keep: Option<KeepAlive> = None;
            while let Ok(ev) = rx.recv() {
                match ev {
                    Ev::Early { code, body } => {
                        respond(&mut stream, code, "application/json", body.as_bytes())?;
                        break;
                    }
                    Ev::HeadersSse { rid } => {
                        sse_headers(&mut stream, &rid)?;
                        keep = Some(KeepAlive::start(stream.try_clone()?));
                    }
                    Ev::HeadersChunked => {
                        chunked_headers(&mut stream)?;
                        keep = Some(KeepAlive::start_chunked(stream.try_clone()?));
                    }
                    Ev::Bytes(b) => {
                        stream.write_all(&b)?;
                        stream.flush()?;
                    }
                    Ev::FinalBody { body } => {
                        if http11 {
                            respond_chunked(&mut stream, body.as_bytes())?;
                        } else {
                            respond(&mut stream, 200, "application/json", body.as_bytes())?;
                        }
                        break;
                    }
                    Ev::Done => break,
                }
            }
            drop(keep);
            Ok(())
        }
        _ => respond(&mut stream, 404, "application/json", br#"{"error":"not found"}"#),
    }
}

/// 一轮生成步进的结果。
enum Step {
    /// 本步产出的客户端字节（可能为空）。
    More(Vec<u8>),
    /// 生成结束：收尾字节（SSE 的 usage+[DONE] / 非流式整段 body）+ finish_reason。
    Fin { bytes: Vec<u8>, finish: &'static str },
}

/// 进行中的生成（v1.5：每会话一个，引擎线程轮转步进）。
struct Gen {
    sid: String,
    rid: String,
    stream_resp: bool,
    eos: i64,
    max_tokens: usize,
    prompt_tokens: usize,
    restored: usize,
    hit_src: &'static str,
    cur: i64,
    n: usize,
    t_start: std::time::Instant,
    detok: Detok,
    split: StreamSplitter,
    stop: StopFilter,
    call_idx: usize,
    stopped: bool,
    // 非流式累积
    content: String,
    reasoning: String,
    tool_calls: Vec<serde_json::Value>,
    tx: std::sync::mpsc::Sender<Ev>,
}

impl Gen {
    /// 收尾：冲刷 think/content 尾巴 + stop 余量 + finish/usage 帧（与非流式整段 body）。
    pub(crate) fn finish(&mut self, mut finish: &'static str) -> Step {
        let mut tail = Vec::new();
        self.split.flush(&mut tail);
        let mut out = Vec::new();
        let stopped = if self.stream_resp {
            render_events(&mut out, &mut tail, &mut self.call_idx, &self.rid, &mut self.stop)
                .unwrap_or(false)
        } else {
            collect_events(&mut tail, &mut self.content, &mut self.reasoning,
                &mut self.tool_calls, &mut self.stop)
        };
        self.stopped |= stopped;
        if let Some(rest) = self.stop.finish() {
            if self.stream_resp {
                out.extend_from_slice(sse_chunk(&self.rid, "content", &rest).as_bytes());
            } else {
                self.content.push_str(&rest);
            }
        }
        if self.stopped {
            finish = "stop";
        } else if self.stream_resp {
            if self.call_idx > 0 {
                finish = "tool_calls";
            }
        } else if !self.tool_calls.is_empty() {
            finish = "tool_calls";
        }
        let rid = self.rid.clone();
        let bytes = if self.stream_resp {
            let usage = serde_json::json!({
                "id": rid,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "lingxi",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.n,
                    "total_tokens": self.prompt_tokens + self.n
                }
            });
            out.extend_from_slice(format!("data: {usage}\n\n").as_bytes());
            out.extend_from_slice(b"data: [DONE]\n\n");
            out
        } else {
            let mut message = serde_json::json!({
                "role": "assistant",
                "content": std::mem::take(&mut self.content),
                "reasoning_content": std::mem::take(&mut self.reasoning)
            });
            if !self.tool_calls.is_empty() {
                message["tool_calls"] = serde_json::Value::Array(std::mem::take(&mut self.tool_calls));
            }
            serde_json::json!({
                "id": rid,
                "object": "chat.completion",
                "created": 0,
                "model": "lingxi",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.n,
                    "total_tokens": self.prompt_tokens + self.n
                }
            })
            .to_string()
            .into_bytes()
        };
        Step::Fin { bytes, finish }
    }
}

/// 一步：发当前 token 的增量 → 解码下一个。生成器状态全部在 `Gen` 里，
/// `seq` 由引擎线程按会话切好（`activate`）后传入。
#[allow(clippy::too_many_arguments)]

fn gen_step(
    g: &mut Gen,
    seq: &mut lingxi_driver::Sequence,
    eng: &lingxi_driver::Engine<'_>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    opts: &ServeOpts<'_>,
) -> Result<Step> {
    if g.cur == g.eos {
        return Ok(g.finish("stop"));
    }
    g.n += 1;
    let text = g.detok.push(opts.tok, g.cur);
    let mut events = Vec::new();
    g.split.push(&text, &mut events);
    let mut out = Vec::new();
    let stopped = if g.stream_resp {
        render_events(&mut out, &mut events, &mut g.call_idx, &g.rid, &mut g.stop)?
    } else {
        collect_events(&mut events, &mut g.content, &mut g.reasoning, &mut g.tool_calls, &mut g.stop)
    };
    g.stopped |= stopped;
    if g.stopped {
        return Ok(g.finish("stop"));
    }
    if g.n >= g.max_tokens {
        return Ok(g.finish("length"));
    }
    // 并发路径无投机（见 `run`）：纯自回归一步。
    g.cur = seq.decode_token(eng, provider)?;
    Ok(Step::More(out))
}

/// 生成收尾：检查点/记账（需 seq）→ 发收尾帧 → Done。
#[allow(clippy::too_many_arguments)]

fn finish_gen(
    g: Gen,
    fin: Step,
    sessions: &mut Sessions,
    cache: &mut PrefillCache,
    eng: &lingxi_driver::Engine<'_>,
    perf: &PerfLog,
) {
    // ── 检查点尾巴：最后一个 token "已产生但没喂" ⇒ 去掉 + 存末尾检查点 ──
    if prefix_cache_enabled() {
        sessions.activate(&g.sid);
        if let Some(seq) = sessions.current_mut() {
            let nxt = *seq.ids.last().unwrap_or(&0);
            seq.truncate_unfed();
            if ckpt_level() >= 2 {
                save_ckpt_auto(seq, eng, seq.ids.len(), CkptKind::End, nxt, cache, "末尾");
            }
            cache.device_ids = seq.ids.clone();
        }
    }
    // ── 引擎侧日志（应用拿不到的内部量；spec 并发路径不可用 ⇒ 恒 0）──
    {
        let (eh, em, eu) = crate::arena_source::expert_track_finish();
        let (psink, ppref) = lingxi_model::qsa::paged_stats();
        crate::serve::stats::record(
            g.n,
            g.t_start.elapsed().as_secs_f64(),
            g.restored > 0,
            eh,
            em,
            0,
            0,
        );
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let line = serde_json::json!({
            "ts": ts,
            "request_id": g.rid,
            "session": g.sid,
            "prefix_hit": g.restored,
            "hit_src": g.hit_src,
            "expert_hit": eh,
            "expert_miss": em,
            "expert_unique": eu,
            "paged_sinks": psink,
            "paged_prefetches": ppref,
            "cache_mb": (cache.cache_bytes as f64 / (1024.0 * 1024.0) * 10.0).round() / 10.0
        });
        perf.write_line(&(line.to_string() + "\n"));
    }
    let Step::Fin { bytes, finish } = fin else { unreachable!() };
    eprintln!("[serve] 完成 {} tok  finish={finish}", g.n);
    if g.stream_resp {
        send_ev(&g.tx, Ev::Bytes(bytes));
    } else {
        send_ev(&g.tx, Ev::FinalBody { body: String::from_utf8_lossy(&bytes).into_owned() });
    }
    send_ev(&g.tx, Ev::Done);
}

/// 请求预处理（引擎线程）：校验 → 会话准入 → 头发送 → prefill → 首个 token。
/// 返回 Ok(Some(Gen)) 入轮转；Ok(None) = 已 Early 回应；内部错误也已回应。
#[allow(clippy::too_many_arguments)]

fn setup_gen(
    req: Req,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    sessions: &mut Sessions,
    provider: &mut dyn lingxi_driver::LayerProvider,
    cache: &mut PrefillCache,
    protect: &[String],
) -> Result<Option<Gen>> {
    let early = |code: u16, body: String| -> Result<Option<Gen>> {
        send_ev(&req.tx, Ev::Early { code, body });
        Ok(None)
    };
    let v = &req.v;
    let stream_resp = v["stream"].as_bool().unwrap_or(false);
    let max_tokens = v["max_tokens"]
        .as_u64()
        .or_else(|| v["max_completion_tokens"].as_u64())
        .unwrap_or(opts.max_new_default as u64) as usize;
    let Some(messages) = v["messages"].as_array().filter(|m| !m.is_empty()) else {
        return early(400, r#"{"error":"messages must be a non-empty array"}"#.into());
    };
    let thinking_kw = v["chat_template_kwargs"]["enable_thinking"].as_bool();
    let cfg = ChatCfg {
        enable_thinking: v["enable_thinking"].as_bool().or(thinking_kw).unwrap_or(true),
        preserve_thinking: v["preserve_thinking"].as_bool().unwrap_or(true),
        effort: match ReasoningEffort::parse(v["reasoning_effort"].as_str().unwrap_or("xhigh")) {
            Some(e) => e,
            None => return early(400, r#"{"error":"reasoning_effort must be one of xhigh|medium|low"}"#.into()),
        },
    };
    let tools: Option<Vec<serde_json::Value>> = v["tools"].as_array().cloned();
    let rid = next_request_id();
    let stops: Vec<String> = match &v["stop"] {
        serde_json::Value::String(s) if !s.is_empty() => vec![s.clone()],
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    let (prompt, boundary_chars) = build_prompt(messages, tools.as_ref(), cfg);
    let ids = match opts.tok.encode(&prompt) {
        Ok(x) => x,
        Err(e) => return early(400, format!(r#"{{"error":"tokenize 失败: {e}"}}"#)),
    };
    let prompt_tokens = ids.len();
    let boundary_ids: Option<usize> = if boundary_chars > 0 && boundary_chars < prompt.len() {
        match opts.tok.encode(&prompt[..boundary_chars]) {
            Ok(b) if !b.is_empty() && b.len() < ids.len() && ids[..b.len()] == b[..] => Some(b.len()),
            _ => None,
        }
    } else {
        None
    };
    if prompt_tokens + 1 >= opts.max_ctx {
        return early(400, format!(
            r#"{{"error":"prompt {} tokens + 生成超过容量 {}（调大 LINGXI_SERVE_MAX_CTX 重启）"}}"#,
            prompt_tokens, opts.max_ctx
        ));
    }
    let max_tokens = max_tokens.min(opts.max_ctx.saturating_sub(prompt_tokens + 1));

    let sid = Req::sid_of(v);
    let need = (prompt_tokens + max_tokens + 64).min(opts.max_ctx);
    if let Err(e) = sessions.ensure(&sid, need, eng, mk_seq, cache, protect) {
        return early(429, format!(r#"{{"error":"会话准入失败: {e}"}}"#));
    }
    let Some(_) = sessions.current_mut() else {
        return early(500, r#"{"error":"session materialize failed"}"#.into());
    };

    // **先把头送出去**（连接线程负责 prefill 期间 keep-alive）。
    if stream_resp {
        send_ev(&req.tx, Ev::HeadersSse { rid: rid.clone() });
    } else if req.http11 {
        send_ev(&req.tx, Ev::HeadersChunked);
    }
    // prefill 失败时的按模式报错
    let fail = |msg: &str| {
        if stream_resp {
            let e = serde_json::json!({"error": {"message": msg, "type": "server_error"}});
            send_ev(&req.tx, Ev::Bytes(format!("data: {e}\n\ndata: [DONE]\n\n").into_bytes()));
        } else {
            send_ev(&req.tx, Ev::FinalBody { body: format!(r#"{{"error":"{msg}"}}"#) });
        }
        send_ev(&req.tx, Ev::Done);
    };

    let cache_on = prefix_cache_enabled();
    let _pre_reuse = if cache_on {
        cache.best(&ids).map(|(_, l)| l).unwrap_or(0)
    } else {
        0
    };
    crate::arena_source::expert_track_reset();
    let t0 = Instant::now();
    let mut hyper_p = None;
    let (first, restored, hit_src) = {
        let seq = sessions.current_mut().unwrap();
        match run_prefill(seq, eng, provider, &ids, boundary_ids, cache, cache_on,
                          false, &mut hyper_p, true)
        {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[serve] 前缀续跑失败（回退全量）: {e:#}");
                cache.misses += 1;
                match run_prefill(seq, eng, provider, &ids, boundary_ids, cache, cache_on,
                                  false, &mut hyper_p, false)
                {
                    Ok(v) => v,
                    Err(e) => {
                        fail(&format!("prefill 失败: {e}"));
                        return Ok(None);
                    }
                }
            }
        }
    };
    eprintln!(
        "[serve] prompt={prompt_tokens} tok  前缀复用={restored} tok  TTFT={:.2}s  首token={first}  会话='{sid}'",
        t0.elapsed().as_secs_f64()
    );
    if cache_on && ckpt_level() >= 1 {
        let seq = sessions.current_mut().unwrap();
        save_ckpt_auto(seq, eng, prompt_tokens, CkptKind::Prompt, first, cache, "prompt");
    }

    Ok(Some(Gen {
        sid,
        rid,
        stream_resp,
        eos: opts.eos,
        max_tokens,
        prompt_tokens,
        restored,
        hit_src,
        cur: first,
        n: 0,
        t_start: std::time::Instant::now(),
        detok: Detok::new(),
        split: StreamSplitter::new(cfg.enable_thinking),
        stop: StopFilter::new(stops),
        call_idx: 0,
        stopped: false,
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        tx: req.tx,
    }))
}

/// 并发引擎线程：收请求 → prefill 入列 → 轮转步进活跃生成。
fn engine_loop(
    req_rx: std::sync::mpsc::Receiver<Req>,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    perf: &PerfLog,
) {
    let mut sessions = Sessions::new(opts.max_ctx);
    let mut cache = PrefillCache::default();
    let mut pending: VecDeque<Req> = VecDeque::new();
    let mut active: Vec<Gen> = Vec::new();

    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            eprintln!("[serve] 收到 shutdown：把全部 live 会话落盘…");
            sessions.offload_all_live(eng, &mut cache);
            eprintln!("[serve] 已退出");
            return;
        }
        // 收新请求（非阻塞排空）。
        loop {
            match req_rx.try_recv() {
                Ok(req) => {
                    let sid = req.sid();
                    if active.iter().any(|g| g.sid == sid) {
                        pending.push_back(req); // 同会话在跑：排队保序
                        continue;
                    }
                    let protect: Vec<String> = active.iter().map(|g| g.sid.clone()).collect();
                    match setup_gen(req, opts, eng, mk_seq, &mut sessions, provider, &mut cache, &protect) {
                        Ok(Some(g)) => active.push(g),
                        Ok(None) => {}
                        Err(e) => eprintln!("[serve] 请求预处理失败: {e:#}"),
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    eprintln!("[serve] 全部连接线程退出，引擎收工");
                    return;
                }
            }
        }
        if active.is_empty() {
            // 启动排队请求（跳过与活跃生成冲突的会话）；无事可做则等下一个请求
            // （recv_timeout：shutdown 旗标要能唤醒引擎线程，不能永久阻塞在 recv 上）。
            if !try_start_pending(&mut pending, &mut active, &mut sessions, &mut cache,
                                  opts, eng, mk_seq, provider)
            {
                match req_rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(req) => pending.push_back(req),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => return,
                }
            }
            continue;
        }
        // ── 轮转步进：每个活跃生成一个 token ──
        let mut i = 0;
        while i < active.len() {
            let sid = active[i].sid.clone();
            sessions.activate(&sid);
            let Some(seq) = sessions.current_mut() else {
                let g = active.remove(i);
                eprintln!("[serve] ⚠ 会话 '{sid}' 失踪，生成中断");
                send_ev(&g.tx, Ev::Done);
                continue;
            };
            let st = match gen_step(&mut active[i], seq, eng, provider, opts) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[serve] 步进失败（会话 '{sid}'）: {e:#}");
                    let g = active.remove(i);
                    if g.stream_resp {
                        let e = serde_json::json!({"error": {"message": format!("{e}"), "type": "server_error"}});
                        send_ev(&g.tx, Ev::Bytes(format!("data: {e}\n\ndata: [DONE]\n\n").into_bytes()));
                    } else {
                        send_ev(&g.tx, Ev::FinalBody { body: r#"{"error":"decode 失败"}"#.into() });
                    }
                    finish_gen_bookkeeping_only(g, &mut sessions, &mut cache, eng, perf);
                    continue;
                }
            };
            match st {
                Step::More(bytes) => {
                    if !bytes.is_empty() {
                        send_ev(&active[i].tx, Ev::Bytes(bytes));
                    }
                    i += 1;
                }
                Step::Fin { .. } => {
                    let g = active.remove(i);
                    finish_gen(g, st, &mut sessions, &mut cache, eng, perf);
                }
            }
        }
        // 一轮步进后：起掉已释放会话的排队请求。
        let _ = try_start_pending(&mut pending, &mut active, &mut sessions, &mut cache,
                                  opts, eng, mk_seq, provider);
    }
}

/// 启动排队请求（跳过与活跃生成同会话的；prefill 在函数内联完成）。
#[allow(clippy::too_many_arguments)]

fn try_start_pending(
    pending: &mut VecDeque<Req>,
    active: &mut Vec<Gen>,
    sessions: &mut Sessions,
    cache: &mut PrefillCache,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    provider: &mut dyn lingxi_driver::LayerProvider,
) -> bool {
    let mut started = false;
    let mut i = 0;
    while i < pending.len() {
        let sid = pending[i].sid();
        if active.iter().any(|g| g.sid == sid) {
            i += 1;
            continue;
        }
        let req = pending.remove(i).unwrap();
        let protect: Vec<String> = active.iter().map(|g| g.sid.clone()).collect();
        match setup_gen(req, opts, eng, mk_seq, sessions, provider, cache, &protect) {
            Ok(Some(g)) => {
                active.push(g);
                started = true;
            }
            Ok(None) => {}
            Err(e) => eprintln!("[serve] 请求预处理失败: {e:#}"),
        }
        i += 1;
    }
    started
}

/// 步进失败路径的收尾：只做检查点/记账，不发正常收尾帧（错误帧已发）。
fn finish_gen_bookkeeping_only(
    g: Gen,
    sessions: &mut Sessions,
    cache: &mut PrefillCache,
    _eng: &lingxi_driver::Engine<'_>,
    perf: &PerfLog,
) {
    if prefix_cache_enabled() {
        sessions.activate(&g.sid);
        if let Some(seq) = sessions.current_mut() {
            seq.truncate_unfed();
            cache.device_ids = seq.ids.clone();
        }
    }
    {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let line = serde_json::json!({
            "ts": ts, "request_id": g.rid, "session": g.sid, "error": true,
        });
        perf.write_line(&(line.to_string() + "\n"));
    }
    send_ev(&g.tx, Ev::Done);
}

/// v1.5 并发路径装配：accept 线程 + 每连接线程 + 本线程 = 引擎线程。
fn run_concurrent(
    listener: TcpListener,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    provider: &mut dyn lingxi_driver::LayerProvider,
) -> Result<()> {
    let perf = PerfLog::from_env();
    if let Some(p) = &perf.path {
        eprintln!("[serve] 性能日志（不含内容）: {p}");
    }
    let (req_tx, req_rx) = std::sync::mpsc::channel::<Req>();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            // 非阻塞 accept：否则 shutdown 后线程永远卡在 `incoming()` 上不退出。
            listener.set_nonblocking(true).ok();
            loop {
                if SHUTDOWN.load(Ordering::Relaxed) {
                    break;
                }
                match listener.accept() {
                    Ok((s, _)) => {
                        let tx = req_tx.clone();
                        // 每个连接一个线程：读请求 → 入队 → 转发事件回 socket。
                        scope.spawn(move || conn_loop(s, opts, tx));
                    }
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            || e.kind() == std::io::ErrorKind::TimedOut =>
                    {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    Err(e) => {
                        eprintln!("[serve] accept 失败: {e}");
                        break;
                    }
                }
            }
        });
        engine_loop(req_rx, opts, eng, mk_seq, provider, &perf);
    });
    Ok(())
}
