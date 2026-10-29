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

/// **只记性能、不记内容**的 JSONL（数据飞轮 M1）。
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
    // ★ 2026-10-27：投机（MTP）接入 v1.5 并发路径——草稿状态拆成每请求 SpecState
    // （见 spec.rs/chat.rs），多会话时间片轮转各不干扰。默认全部走并发路径；
    // 串行路径仅作排障回退（LINGXI_SERVE_SERIAL=1）。
    if spec.is_some() && lingxi_model::flag("LINGXI_SERVE_SERIAL") {
        eprintln!(
            "[serve] 就绪 :{port}（串行路径·投机开·排障回退）  max_ctx={}",
            opts.max_ctx
        );
        run_serial(listener, opts, eng, mk_seq, provider, spec)
    } else {
        if spec.is_some() {
            eprintln!("[serve] 投机 MTP 开（v1.5 并发路径，kd={} min_p={}，草稿状态每请求隔离）",
                std::env::var("LINGXI_SERVE_SPEC_KD").unwrap_or_else(|_| "7".into()),
                std::env::var("LINGXI_SPEC_MIN_P").unwrap_or_else(|_| "0.7".into()));
        }
        eprintln!(
            "[serve] 就绪 :{port}（v1.5 并发路径：连接线程 + 引擎线程轮转步进·单流时间片）  max_ctx={}",
            opts.max_ctx
        );
        run_concurrent(listener, opts, eng, mk_seq, provider, spec)
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
        // 轻量监视面板（可选）：`LINGXI_NO_DASHBOARD=1` 关闭 `/stats` + `/dashboard`
        // （生产只暴露 /health 时用它显式关掉，而不是靠网络层挡）。
        ("GET", "/stats") if !crate::serve::stats::disabled() => respond(
            &mut stream,
            200,
            "application/json",
            crate::serve::stats::json().as_bytes(),
        ),
        ("GET", "/dashboard") if !crate::serve::stats::disabled() => respond(
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

    // ── 会话驻留：按 session_id 物化/切换（沉没的秒级唤醒；不显式传则共享 "default"）──
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
        // 理论不可达（ensure Ok ⇒ materialize 必已设 cur_sid）。真撞到说明切换竞争/
        // 物化不变量被破坏——降级 429 让客户端重试（排队口径），并响亮日志钉现场，
        // 不再拿 500 砸用户（2026-10-27：长会话运行后撞 500 的根因靠这条日志确认）。
        eprintln!("[serve] ⚠ ensure 成功但 current 为空（物化不变量破坏，sid={sid}）——降级 429");
        return respond(
            stream,
            429,
            "application/json",
            br#"{"error":"session switch race, retry"}"#,
        );
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
    // **按请求选路**（2026-10-25 改）：缓存命中不再一票否决投机。
    // 旧设计：命中 ⇒ 纯 AR（吃 TTFT≈0）。但多轮对话的每一轮都命中——于是生产的
    // 典型请求**永远**吃不到投机（实测：168 tok 走 AR 3.9s，投机只需 ~2.2s）。
    // 新判据（重 prefill 成本 vs 投机收益，实测数字：prefill ~0.5ms/tok、
    // AR ~43 tok/s、投机 ~90 tok/s ⇒ 收益 ~12.7ms/tok ⇒ 盈亏 max_tokens ≈ prompt/25）：
    //   命中但 `max_tokens ≥ max(prompt/25, 48)` ⇒ 放弃复用、整段重 prefill 开投机；
    //   短生成或超长 prompt ⇒ 保持旧路（TTFT≈0 + AR）。
    // 两条路都是 trunk 贪心 ⇒ 文本逐位相同，纯性能路由。`LINGXI_SPEC_ON_HIT=0` 回滚。
    let pre_reuse = if cache_on {
        cache.best(&ids).map(|(_, l)| l).unwrap_or(0)
    } else {
        0
    };
    let spec_on_hit = std::env::var("LINGXI_SPEC_ON_HIT")
        .map(|v| v != "0")
        .unwrap_or(true);
    let spec_on = spec.is_some()
        && (pre_reuse == 0
            || (spec_on_hit && max_tokens >= 48 && max_tokens * 25 >= prompt_tokens));
    // 本请求的专家统计（引擎内部；应用侧拿不到）
    crate::arena_source::expert_track_reset();
    let t0 = Instant::now();
    let mut hyper_p: Option<cudarc::driver::CudaSlice<half::bf16>> = None;
    let (first, restored, hit_src) = match run_prefill(
        seq, eng, provider, &ids, boundary_ids, cache, cache_on, spec_on, &mut hyper_p, true,
        &sid, 0, // 串行路径无引擎轮转，分块无意义 ⇒ 恒整块
    ) {
        Ok(PrefillOutcome::Done { first, restored, hit }) => (first, restored, hit),
        Ok(PrefillOutcome::Chunked(_)) => unreachable!("串行路径 chunk_sz=0 ⇒ 不可能分块"),
        Err(e) => {
            // 兜底：检查点/续跑出任何问题都退回“reset + 全量 prefill”，不把错误抛给用户
            eprintln!("[serve] 前缀续跑失败（回退全量）: {e:#}");
            cache.misses += 1;
            match run_prefill(
                seq, eng, provider, &ids, boundary_ids, cache, cache_on, spec_on, &mut hyper_p,
                false,
                &sid, 0,
            ) {
                Ok(PrefillOutcome::Done { first, restored, hit }) => (first, restored, hit),
                Ok(PrefillOutcome::Chunked(_)) => unreachable!("chunk_sz=0 ⇒ 不可能分块"),
                Err(e) => {
                    eprintln!("[serve] ✗ 全量 prefill 也失败: {e:#}");
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
    // prompt 检查点已改由 run_prefill 在喂入路径上落"安全尾"档（len-2，防跨段合并失效）。

    // ── 投机：prime MTP（喂整段 prompt 的 h）→ 先 decode 两次拿 h_last 与 AR 基准 ──
    let mut spec_gen: Option<SpecGen> = None;
    let t_spec0 = Instant::now();
    if let (Some(rt), Some(hp)) = (spec.as_mut(), hyper_p) {
        let mut st_out: Option<crate::spec::SpecState> = None;
        let primed = (|| -> Result<()> {
            let mut st = crate::spec::SpecState::new(eng.k, &eng.shape, opts.max_ctx + 528, rt.kd())?;
            rt.reset(&mut st);
            let emb_p = rt.prompt_emb(&seq.ids)?;
            let _ = rt.prime(&hp, &seq.ids, emb_p, &mut st)?;
            st_out = Some(st);
            Ok(())
        })();
        if let Err(e) = primed {
            eprintln!("[serve] MTP prime 失败，回退纯自回归: {e:#}");
        } else {
            let kd = rt.kd();
            let init_res = (|| -> Result<(lingxi_driver::SpecBuffers, Vec<i64>, cudarc::driver::CudaSlice<half::bf16>, f64)> {
                let sb = seq.spec_prepare(eng, provider, kd + 1)?;
                // AR 基准 ms/tok：取**第二步** decode（第一步含一次性冷成本，实测 44.3 vs
                // 稳态 24.4——抬高关闸 ② 门槛近 2×）。多花一步 decode 换可靠判据。
                let (tok2, _h2) = seq.decode_token_with_hyper(eng, provider)?;
                let t_ar = Instant::now();
                let (tok3, h_last) = seq.decode_token_with_hyper(eng, provider)?;
                Ok((sb, vec![tok2, tok3], h_last, t_ar.elapsed().as_secs_f64() * 1e3))
            })();
            match init_res {
                Ok((sb, toks, h_last, ar_ms)) => {
                    // `first` 由生成器开头直接发（与纯自回归同口径）；队列里只放它之后的 token
                    let mut queue = VecDeque::new();
                    queue.extend(toks);
                    spec_gen = Some(SpecGen {
                        st: st_out.expect("prime 成功必有 st"), sb, h_last, queue,
                        accepted: 0, rounds: 0, verified: 0, disabled: false,
                        ar_ms, eff: std::collections::VecDeque::new(),
                    });
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
                     prompt_tokens, max_tokens, spec_gen.as_mut(), spec)
    } else {
        batch_generate(stream, opts, cfg, chunked_resp, &rid, stops.clone(), seq, eng, provider,
                       first, prompt_tokens, max_tokens, spec_gen.as_mut(), spec)
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
            save_ckpt_auto(seq, eng, seq.ids.len(), CkptKind::End, nxt, cache, "末尾", &sid);
        }
        // 设备 KV 现在正好对应 seq.ids（下次判断 Dev 条目是否还有效就靠它）
        cache.device_ids = seq.ids.clone();
    }
    r
}

/// reset 重灌后清理本会话的旧 Dev 检查点：除 keep（刚恢复、KV 前缀还有效）外全删。
fn drop_stale_dev(cache: &mut PrefillCache, sid: &str, keep: Option<usize>) {
    let mut pos = 0usize;
    cache.entries.retain(|e| {
        let keep_it = !matches!(e.store, CkptStore::Dev(_))
            || e.owner.as_deref() != Some(sid)
            || Some(pos) == keep;
        pos += 1;
        keep_it
    });
}

/// serve 层分块 prefill 的**调度块大小**（`LINGXI_SERVE_PREFILL_CHUNK`，值感知，
/// **默认 0 = 关**，建议档 4096）。
///
/// 与 driver 内部 `LINGXI_PREFILL_CHUNK`（压显存峰值的**内层**粒度，默认 2048/512）
/// 正交：本开关切的是**调度块**——块之间引擎线程回到 gen 轮转，让其他会话的 decode
/// 先走，下一块轮转一圈后接续（对标 vLLM/SGLang 的 chunked prefill）。
fn serve_prefill_chunk() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LINGXI_SERVE_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Pfx 连续被跳过的**防饿死上限**（`LINGXI_SERVE_PREFILL_STALL_MS`，值感知，毫秒，
/// **默认 1000**）：decode 优先下本轮有 Gen 在跑时 Pfx 不喂；距上次喂块（或请求
/// 到达）超过本值 ⇒ 强制喂一块（哪怕有 Gen 在跑）——连续 decode 流不会把 prefill
/// 饿死。设 0 = 只要 decode 在动就永远让路（恢复旧的 1:1 轮转让路语义，无强制）。
fn serve_prefill_stall_ms() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LINGXI_SERVE_PREFILL_STALL_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1000)
    })
}

/// 无竞争时 Pfx 块长的**放大倍数**（`LINGXI_SERVE_PREFILL_BURST_MULT`，值感知，
/// **默认 8；0 = 不放大**）：active 里无其他 Gen 且有大量未喂 prompt 时，一块喂
/// `min(剩余, max(块配置×本值, 65536))`——无竞争时逼近整块（块间成本 = 权重
/// DRAM 重读地板，只付一次）；有竞争时回落到配置的调度块长。块长决策每个
/// Pfx 步现算：竞争消失即放大、出现即缩小。
fn serve_prefill_burst_mult() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LINGXI_SERVE_PREFILL_BURST_MULT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
    })
}

/// 单个 Pfx 步的**硬上限**（`LINGXI_SERVE_PREFILL_BURST_MAX`，值感知，默认 4096）：
/// burst 放大后的块长再钳到本值。为什么要有它：一个 Pfx 步是**不可抢占**的同步
/// 前向——65494 tok 的 burst 步实测阻塞循环 ~320s，期间新请求干等（burst 本意
/// 是省权重 DRAM 地板，但单步时长成了新的队头阻塞源）。上限 4096 ⇒ 单步 ≤
/// ~20s（4096 tok × ~4.9ms + 0.76s 地板），新请求准入最坏等 ~20s，大请求地板
/// 开销 119K/4096×0.76s ≈ 22s（3.8%）。调大上限 = 小请求最坏等待变长。
fn serve_prefill_burst_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LINGXI_SERVE_PREFILL_BURST_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4096)
    })
}

/// decode 优先 + 防饿死的喂块决策（纯函数，单测覆盖）：本轮没有任何 Gen 步进
/// ⇒ 喂（decode 全速时 prefill 穿插补一块）；有 Gen 步进但 Pfx 已连续被跳过
/// ≥ `stall_ms` ⇒ 强制喂一块；否则让路（跳过）。
fn pfx_should_feed(gen_moved: bool, stalled_ms: u128, stall_ms: u64) -> bool {
    !gen_moved || stalled_ms >= stall_ms as u128
}

/// 自适应块长的放量决策（纯函数，单测覆盖）：`competing`（active 里有其他 Gen）
/// 或倍数为 0 或剩余量不超过配置的调度块 ⇒ 不放大（用计划块长）；否则本步可喂
/// `min(剩余, max(chunk×mult, 65536))` token。
fn pfx_burst_tokens(chunk: usize, mult: usize, competing: bool, remaining: usize) -> Option<usize> {
    if competing || mult == 0 || remaining <= chunk {
        return None;
    }
    // 硬上限：单步不可抢占，burst 块长必须钳住（默认 4096 ⇒ 单步 ≤ ~20s）。
    Some(remaining.min(chunk.saturating_mul(mult).max(65_536)).min(serve_prefill_burst_max()))
}

/// 自适应块长的块终点（纯函数，单测覆盖）：把本步块终点从计划的 `marks[next]`
/// 放大到 `a + burst`（钳到不超过 prompt 尾），但**不能跨过检查点里程碑**——
/// `halt` = 下一个里程碑位置（恒为 marks 成员）：放大上限覆盖到里程碑 ⇒ 停在
/// 里程碑（存档时机与整块路径"先存安全尾档再补喂尾 2 token"逐字一致）；否则
/// 终点取不超过上限的最大计划块边界（里程碑恒为边界 ⇒ 仍是边界）。返回
/// (块终点 b, 本步喂完后 `next` 应设的值)。
fn pfx_block_end(
    marks: &[usize],
    next: usize,
    fed0: usize,
    burst: Option<usize>,
    halt: usize,
) -> (usize, usize) {
    let b0 = marks[next];
    let a = if next == 0 { fed0 } else { marks[next - 1] };
    match burst {
        Some(bt) if a + bt > b0 => {
            let limit = (a + bt).min(*marks.last().unwrap());
            if halt <= limit {
                // 停在里程碑：k-1 = halt 在 marks 里的下标（里程碑必为 marks 成员）。
                let k = marks.partition_point(|&m| m <= halt);
                (marks[k - 1], k)
            } else {
                let k = marks.partition_point(|&m| m <= limit);
                (marks[k - 1], k)
            }
        }
        _ => (b0, next + 1),
    }
}

/// 请求级分块选路。两条硬守卫（落到整块路径，行为与关分块时逐字相同）：
/// - **spec_on**：MTP 要整段 prompt 的 hyper（`prefill_with_hyper` 收全段），分块不收集；
/// - **LINGXI_PREFILL_TWOPASS**：two-pass 只支持空序列整段喂（`start==0` ensure），
///   分块续跑走 `prefill_append`（start>0）必炸。组合在 probes `check_env` 直接拒绝。
fn chunk_eligible(chunk_sz: usize, spec_on: bool, remaining: usize) -> bool {
    chunk_sz > 0 && !spec_on && !lingxi_driver::Sequence::prefill_twopass_enabled() && remaining > chunk_sz
}

/// 分块 prefill 的**块拆分计划**（纯函数，单测覆盖）。
///
/// 把待喂区间 `[fed, len)` 切成调度块：每 `chunk` token 一块；**检查点里程碑**
/// （安全尾 prompt 档 `cut`、消息边界档 `b`）强制成为块边界——保存时机与整块路径
/// 一致：喂到该位置、用该位置产出的 token 存档，再继续喂剩余。
///
/// 返回 `None` = 不分块（调用方走原整块路径，零行为变化）。
struct ChunkPlan {
    /// 块结束位置（绝对 token 下标，递增，末元恒为 `len`）。
    marks: Vec<usize>,
    /// 安全尾 prompt 档位置（必须是 `marks` 之一）。
    prompt_ckpt: Option<usize>,
    /// 消息边界档位置（必须是 `marks` 之一）。
    boundary_ckpt: Option<usize>,
}

fn build_chunk_plan(
    fed: usize,
    len: usize,
    chunk: usize,
    prompt_ckpt: Option<usize>,
    boundary_ckpt: Option<usize>,
) -> Option<ChunkPlan> {
    if chunk == 0 || len <= fed || len - fed <= chunk {
        return None;
    }
    // 里程碑必须落在待喂区间内部（与整块路径的 `seq.ids.len() < cut` 同口径）。
    let pc = prompt_ckpt.filter(|&c| c > fed && c < len);
    let bc = boundary_ckpt.filter(|&b| b > fed && b < len);
    let mut marks: Vec<usize> = Vec::new();
    let mut p = fed + chunk;
    while p < len {
        marks.push(p);
        p += chunk;
    }
    if let Some(c) = pc {
        marks.push(c);
    }
    if let Some(b) = bc {
        marks.push(b);
    }
    marks.push(len);
    marks.sort_unstable();
    marks.dedup();
    Some(ChunkPlan { marks, prompt_ckpt: pc, boundary_ckpt: bc })
}

/// `run_prefill` 的结果：整块完成；或超块 ⇒ 拆出分块作业（prefill 侧状态），
/// 由调用方（setup_gen）补全请求上下文后挂进 active 列表随轮转步进。
enum PrefillOutcome {
    Done { first: i64, restored: usize, hit: &'static str },
    Chunked(Box<PrefillJob>),
}

/// 分块 prefill 作业的 prefill 侧状态（请求无关部分；`run_prefill` 的产物）。
struct PrefillJob {
    /// 全 prompt（owned：块跨轮转引用，不能借调用方的切片）。
    ids: Vec<i64>,
    plan: ChunkPlan,
    /// 开始分块时的 `seq.ids.len()`（首块起点）。
    fed0: usize,
    /// 热继续来源（收尾时按整块路径口径回写 `cache.device_ids`）。
    hot: bool,
    restored: usize,
    hit: &'static str,
}

/// 分块 prefill 的未完成态：挂在引擎线程的 active 列表里，decode 优先调度——
/// 每轮所有 Gen 先走一步，本轮无 Gen 步进（或连续被跳过 ≥ stall_ms 防饿死强制）
/// 才喂一块（一块 ≈ chunk token 的前向，无竞争时按 `pfx_burst_tokens` 放宽）。
/// 块边界对 GDN 递推（状态原位连续推进）与 QSA KV（append-only）透明，
/// 首个 token 只在最后一块产出。
struct PrefillGen {
    sid: String,
    rid: String,
    stream_resp: bool,
    eos: i64,
    max_tokens: usize,
    prompt_tokens: usize,
    restored: usize,
    hit_src: &'static str,
    /// 请求到达时刻（setup_gen 起跑）：TTFT 口径 = 最后一块完成时刻 - 本时刻。
    t0: Instant,
    enable_thinking: bool,
    stops: Vec<String>,
    // ── 分块计划与进度 ──
    /// 全 prompt（owned：块跨轮转引用，不能借调用方的切片）。
    ids: Vec<i64>,
    plan: ChunkPlan,
    /// 下一块在 `plan.marks` 里的下标。
    next: usize,
    /// 开始分块时的 `seq.ids.len()`（首块起点）。
    fed0: usize,
    /// 热继续来源（收尾时按整块路径口径回写 `cache.device_ids`）。
    hot: bool,
    n_chunks: usize,
    t_feed: Duration,
    /// 块失败回退全量重灌只许一次。
    retried: bool,
    // ── decode 优先调度状态 ──
    /// 距上次喂块（或请求到达）的连续让路计时基准：超过 `serve_prefill_stall_ms`
    /// 还在被跳过 ⇒ 强制喂一块（防饿死）。每次喂块（含失败的尝试）重置。
    since: Instant,
    /// 上次喂块以来被连续跳过的轮数（日志口径；喂块时清零）。
    n_skips: usize,
    tx: std::sync::mpsc::Sender<Ev>,
}

/// 一步分块 prefill 的结果。
enum PfxStep {
    /// 本块喂完，还有剩余块。
    More,
    /// 最后一块喂完：首个 token 产出，转成正常生成。
    Done(Gen),
}

impl PrefillGen {
    /// 最后一块收尾：打 TTFT（首 token 口径）+ 分块统计，转成 `Gen` 进 decode 轮转。
    /// `&mut self` + `mem::take`：调用方手里是 `&mut`（错误路径还要读 retried/tx）。
    fn into_gen(&mut self, first: i64) -> Gen {
        let ttft = self.t0.elapsed();
        eprintln!(
            "[serve] prompt={} tok  前缀复用={} tok  TTFT={:.2}s（分块 prefill：{} 块，均 {:.2}s/块）  首token={first}  会话='{}'",
            self.prompt_tokens,
            self.restored,
            ttft.as_secs_f64(),
            self.n_chunks,
            if self.n_chunks > 0 { self.t_feed.as_secs_f64() / self.n_chunks as f64 } else { 0.0 },
            self.sid,
        );
        Gen {
            sid: std::mem::take(&mut self.sid),
            rid: std::mem::take(&mut self.rid),
            stream_resp: self.stream_resp,
            eos: self.eos,
            max_tokens: self.max_tokens,
            prompt_tokens: self.prompt_tokens,
            restored: self.restored,
            hit_src: self.hit_src,
            cur: first,
            n: 0,
            t_start: Instant::now(),
            detok: Detok::new(),
            split: StreamSplitter::new(self.enable_thinking),
            stop: StopFilter::new(std::mem::take(&mut self.stops)),
            call_idx: 0,
            stopped: false,
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            spec: None,
            tx: self.tx.clone(),
        }
    }
}

/// 分块 prefill 步进：喂**一块**（`ids[a..b]`），处理落在本块末尾的检查点里程碑。
/// 中间块产出 token 丢弃（`truncate_unfed` 保持 `seq.ids == ids[..b]`，不依赖
/// argmax 与下一 token 相等的隐含假设）；最后一块的产出即首个 token。
/// 所有权进出：Ok(More) 时调用方把 job 放回 active；Err 时 job 一并交还
/// （引擎循环要读 `retried` 决定是否就地回退全量、要拿 tx/rid 发错误帧）。
/// `burst` = 自适应块长放宽（`pfx_burst_tokens` 的产物）：None 用计划块长喂
/// 一块；Some(n) 时本步块终点放宽到 `a+n`（仍钳在计划块边界/里程碑上）。
fn prefill_step(
    mut j: PrefillGen,
    seq: &mut lingxi_driver::Sequence,
    eng: &lingxi_driver::Engine<'_>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    cache: &mut PrefillCache,
    burst: Option<usize>,
) -> (PrefillGen, Result<PfxStep>) {
    let r = prefill_step_inner(&mut j, seq, eng, provider, cache, burst);
    (j, r)
}

fn prefill_step_inner(
    j: &mut PrefillGen,
    seq: &mut lingxi_driver::Sequence,
    eng: &lingxi_driver::Engine<'_>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    cache: &mut PrefillCache,
    burst: Option<usize>,
) -> Result<PfxStep> {
    let t_blk = Instant::now();
    let a = if j.next == 0 { j.fed0 } else { j.plan.marks[j.next - 1] };
    // 下一个里程碑（安全尾 prompt 档 / 消息边界档）位置：burst 不得跨过它。
    let halt = [j.plan.prompt_ckpt, j.plan.boundary_ckpt]
        .into_iter()
        .flatten()
        .filter(|&m| m > a)
        .min()
        .unwrap_or(j.ids.len());
    let (b, next) = pfx_block_end(&j.plan.marks, j.next, j.fed0, burst, halt);
    let tok = if a == 0 {
        debug_assert!(seq.ids.is_empty(), "首块起点 0 ⇒ 序列必空");
        seq.prefill(eng, &j.ids[..b], provider)?
    } else {
        seq.prefill_append(eng, &j.ids[..b], provider)?
    };
    j.t_feed += t_blk.elapsed();
    j.n_chunks += 1;
    j.next = next;
    let last = j.next == j.plan.marks.len();
    if !last {
        seq.truncate_unfed();
    }
    // 缓存接线：保存时机与整块路径一致——全部块按序喂完之后、该位置之后的内容
    // 还没喂进来之前（里程碑被 build_chunk_plan 强制对齐为块边界）。
    if j.plan.prompt_ckpt == Some(b) {
        save_ckpt_auto(seq, eng, b, CkptKind::Prompt, tok, cache, "prompt", &j.sid);
    }
    if j.plan.boundary_ckpt == Some(b) {
        save_ckpt_auto(seq, eng, b, CkptKind::Boundary, tok, cache, "边界", &j.sid);
    }
    if !last {
        return Ok(PfxStep::More);
    }
    if j.hot {
        cache.device_ids = seq.ids.clone();
    }
    Ok(PfxStep::Done(j.into_gen(tok)))
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
    sid: &str,
    chunk_sz: usize,
) -> Result<PrefillOutcome> {
    // ── 同会话热继续（v1.5 多物化私有 KV 工作区）：新 prompt 若是当前 seq.ids 的
    //    延伸（多轮对话常态：旧 prompt + 旧回复 + 新消息），KV 还热着——不 reset、
    //    不查缓存，直接补后缀，TTFT 只付增量。spec_on 不行（投机要整段重灌收 hyper）。
    if allow_reuse && !spec_on {
        seq.truncate_unfed();
        let n = seq.ids.len();
        if n > 0 && ids.len() > n && ids[..n] == seq.ids[..] {
            // 同样落"安全尾"prompt 档（len-2，见下方主喂路径的说明），让下一轮还能接龙。
            let want_pc = cache_on && ckpt_level() >= 1 && ids.len() >= n + 6 && ids.len() >= 34;
            // 热继续的增量后缀也可能超块（多轮长文档）⇒ 同样可分块抢占。
            if chunk_eligible(chunk_sz, false, ids.len() - n) {
                let pc = if want_pc { Some(ids.len() - 2) } else { None };
                if let Some(plan) = build_chunk_plan(n, ids.len(), chunk_sz, pc, None) {
                    return Ok(PrefillOutcome::Chunked(Box::new(PrefillJob {
                        ids: ids.to_vec(),
                        plan,
                        fed0: n,
                        hot: true,
                        restored: n,
                        hit: "热继续",
                    })));
                }
            }
            let first = if want_pc {
                let cut = ids.len() - 2;
                let mid = seq.prefill_append(eng, &ids[..cut], provider)?;
                save_ckpt_auto(seq, eng, cut, CkptKind::Prompt, mid, cache, "prompt", sid);
                seq.prefill_append(eng, ids, provider)?
            } else {
                seq.prefill_append(eng, &ids[n..], provider)?
            };
            cache.device_ids = seq.ids.clone();
            return Ok(PrefillOutcome::Done { first, restored: n, hit: "热继续" });
        }
    }
    let mut restored = 0usize;
    let mut hit = "";
    let mut best: Option<(usize, usize)> = None;
    // ★ 2026-10-25：spec_on 时**不恢复**检查点——投机要整段重 prefill 收
    // hyper（命中恢复 ⇒ hyper_p=None ⇒ MTP 被静默跳过，修了选路也不生效）。
    if allow_reuse && cache_on && !spec_on {
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
    let mut keep_pos: Option<usize> = None;
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
                keep_pos = Some(ci);
                if full {
                    // **完全命中**：检查点里已经存了这个前缀的 next token，恢复状态后直接返回
                    hit = store_kind_name(&cache.entries[ci]);
                    seq.ids.push(nxt);
                    drop_stale_dev(cache, sid, keep_pos);
                    return Ok(PrefillOutcome::Done { first: nxt, restored: n, hit });
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
    // reset 重灌会覆盖本会话 KV 工作区：除刚恢复的条格外，本会话旧 Dev 档全部作废
    // （它们的 GDN/PLE 快照对应的 KV 前缀已不可靠；Disk 档自包含，不受影响）。
    drop_stale_dev(cache, sid, keep_pos);
    // 消息边界检查点：只有“从更早的位置续跑”才有意义。
    // ⚠️ spec_on 时跳过：投机要 `prefill_with_hyper` 在**空序列**上整段喂（收全段 hyper），
    //    先喂边界前缀会把 seq 弄非空 → assert 炸（2026-10-20 v1.5 验证实测命中）。
    let boundary = if allow_reuse && cache_on && !spec_on {
        boundary.filter(|&b| b > restored && b < ids.len())
    } else {
        None
    };
    // 分块选路（守卫见 `chunk_eligible`：spec_on / two-pass 永不分块）。会分块时
    // 消息边界档改作**计划里程碑**（块边界强制对齐到 b，保存时机语义不变）；
    // 不分块则保持原同步喂法。
    let fed = seq.ids.len();
    let chunk_here = chunk_eligible(chunk_sz, spec_on, ids.len() - fed);
    if !chunk_here {
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
                save_ckpt_auto(seq, eng, b, CkptKind::Boundary, nxt, cache, "边界", sid);
            }
        }
    }
    // 投机要求从空序列整段 prefill（要收集每行 hyper）；一旦命中过检查点就退回普通路径，
    // 调用方看到 hyper_p=None 会自动跳过 MTP。
    let first = if spec_on && restored == 0 {
        // spec 守卫：分块请求永不到这里（hyper 要整段收集）。
        let (t, h) = seq.prefill_with_hyper(eng, ids, provider)?;
        *hyper_p = Some(h);
        t
    } else {
        // 2026-10-28 安全尾 prompt 档：最后 2 个 token 单独补喂，喂之前存 len-2 档。
        // 多轮对话的下一轮 prompt 是本轮的延伸，但 prompt 末尾 token 会跟新内容跨段
        // 合并（"\n"+"收到" → "\n收到" 单 token）——全量档在合并点**整份失效**
        // （119K 文档会话第二轮复用仅 42 tok 实测）；len-2 档丢弃末 2 token，
        // 后续轮必能整段命中，代价只是每轮重喂末尾 2 token。
        // `cur` = 判据基点：分块时必为 fed（boundary 未同步喂）；整块时可能已喂到边界 b。
        let cur = seq.ids.len();
        let want_pc = !spec_on
            && allow_reuse
            && cache_on
            && ckpt_level() >= 1
            && ids.len() >= 34
            && ids.len() >= cur + 4;
        if chunk_here {
            let pc = if want_pc { Some(ids.len() - 2) } else { None };
            if let Some(plan) = build_chunk_plan(fed, ids.len(), chunk_sz, pc, boundary) {
                return Ok(PrefillOutcome::Chunked(Box::new(PrefillJob {
                    ids: ids.to_vec(),
                    plan,
                    fed0: fed,
                    hot: false,
                    restored,
                    hit,
                })));
            }
        }
        if want_pc {
            let cut = ids.len() - 2;
            if cur < cut {
                let mid = if cur == 0 {
                    seq.prefill(eng, &ids[..cut], provider)?
                } else {
                    seq.prefill_append(eng, &ids[..cut], provider)?
                };
                save_ckpt_auto(seq, eng, cut, CkptKind::Prompt, mid, cache, "prompt", sid);
            }
            seq.prefill_append(eng, ids, provider)?
        } else if cur == 0 {
            seq.prefill(eng, ids, provider)?
        } else {
            seq.prefill_append(eng, ids, provider)?
        }
    };
    Ok(PrefillOutcome::Done { first, restored, hit })
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
    mut spec: Option<&mut SpecGen>,
    spec_rt: &mut Option<SpecRuntime<'_>>,
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
            Some(g) => g.next(spec_rt.as_mut().expect("SpecGen 存在必有 rt"), seq, eng, provider)?,
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
    mut spec: Option<&mut SpecGen>,
    spec_rt: &mut Option<SpecRuntime<'_>>,
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
            Some(g) => g.next(spec_rt.as_mut().expect("SpecGen 存在必有 rt"), seq, eng, provider)?,
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
        ("GET", "/stats") if !crate::serve::stats::disabled() => respond(
            &mut stream,
            200,
            "application/json",
            crate::serve::stats::json().as_bytes(),
        ),
        ("GET", "/dashboard") if !crate::serve::stats::disabled() => respond(
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
    /// 投机状态（LINGXI_SERVE_SPEC=1 且 prime 成功时 Some；每请求隔离，随轮转走）。
    spec: Option<SpecGen>,
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
    spec: &mut Option<SpecRuntime<'_>>,
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
    // 投机开：SpecGen 队列优先（一轮接受多个 token，逐个出），队列空了跑下一轮；
    // 关闸/失效时 SpecGen 内部自己退回自回归一步。
    if g.spec.is_some() && spec.is_some() {
        let rt = spec.as_mut().expect("spec 同时 Some");
        let sp = g.spec.as_mut().expect("checked");
        g.cur = sp.next(rt, seq, eng, provider)?;
    } else {
        g.cur = seq.decode_token(eng, provider)?;
    }
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
                save_ckpt_auto(seq, eng, seq.ids.len(), CkptKind::End, nxt, cache, "末尾", &g.sid);
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
    if let Some(sp) = &g.spec {
        eprintln!(
            "[serve] 投机统计（会话 '{}'）：{} 轮｜接受 {} 枚｜{:.2}/轮｜{}",
            g.sid, sp.rounds, sp.accepted,
            if sp.rounds > 0 { sp.accepted as f64 / sp.rounds as f64 } else { 0.0 },
            if sp.disabled { "已切回自回归" } else { "全程投机" }
        );
    }
    eprintln!("[serve] 完成 {} tok  finish={finish}", g.n);
    if g.stream_resp {
        send_ev(&g.tx, Ev::Bytes(bytes));
    } else {
        send_ev(&g.tx, Ev::FinalBody { body: String::from_utf8_lossy(&bytes).into_owned() });
    }
    send_ev(&g.tx, Ev::Done);
}

/// 请求预处理（引擎线程）：校验 → 会话准入 → 头发送 → prefill（整块或拆成分块作业）
/// → 首个 token（整块路径）。
/// 返回 Ok(Some(Active)) 入轮转（Gen = 直接 decode；Pfx = 分块 prefill 随轮转喂块）；
/// Ok(None) = 已 Early 回应；内部错误也已回应。
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
    spec: &mut Option<SpecRuntime<'_>>,
) -> Result<Option<Active>> {
    let early = |code: u16, body: String| -> Result<Option<Active>> {
        if code >= 400 {
            eprintln!("[serve] ✗ HTTP {code} (并发路径): {body}");
        }
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
        eprintln!("[serve] ⚠ ensure 成功但 current 为空（并发路径物化不变量破坏，sid={sid}）——降级 429");
        return early(429, r#"{"error":"session switch race, retry"}"#.into());
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
    // 投机自适应阈值（与下方 prime 的门禁同一口径）：超阈值的请求 spec_on=false——
    // 恢复前缀缓存复用 + 免 hyper 收集；短请求才全速投机。
    let spec_budget = std::env::var("LINGXI_SERVE_SPEC_MAX_PROMPT")
        .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(16384);
    let spec_on = spec.is_some() && ids.len() <= spec_budget;
    let outcome = {
        let seq = sessions.current_mut().unwrap();
        match run_prefill(seq, eng, provider, &ids, boundary_ids, cache, cache_on,
                          spec_on, &mut hyper_p, true, &sid, serve_prefill_chunk())
        {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[serve] 前缀续跑失败（回退全量）: {e:#}");
                cache.misses += 1;
                // 回退重灌整块同步做（chunk_sz=0）：分块作业已在轮转中的失败由
                // 引擎循环就地回退（见 engine_loop 的 Pfx 错误分支）。
                match run_prefill(seq, eng, provider, &ids, boundary_ids, cache, cache_on,
                                  false, &mut hyper_p, false, &sid, 0)
                {
                    Ok(o) => o,
                    Err(e) => {
                        fail(&format!("prefill 失败: {e}"));
                        return Ok(None);
                    }
                }
            }
        }
    };
    // 超块 ⇒ 分块作业挂进 active 列表随轮转步进：块间让路其他会话的 decode，
    // 最后一块产出首个 token（TTFT 口径不变：首 token 时刻 - 请求到达时刻 t0）。
    // spec_on / two-pass 请求永不到这里（run_prefill 内的守卫直接走整块路径）。
    if let PrefillOutcome::Chunked(job) = outcome {
        let job = *job;
        return Ok(Some(Active::Pfx(PrefillGen {
            sid: sid.clone(),
            rid,
            stream_resp,
            eos: opts.eos,
            max_tokens,
            prompt_tokens,
            restored: job.restored,
            hit_src: job.hit,
            t0,
            enable_thinking: cfg.enable_thinking,
            stops,
            ids: job.ids,
            plan: job.plan,
            next: 0,
            fed0: job.fed0,
            hot: job.hot,
            n_chunks: 0,
            t_feed: Duration::ZERO,
            retried: false,
            // 防饿死计时从请求到达（t0）起算：decode 优先下若持续被让路，
            // 超 stall_ms 强制喂一块。
            since: t0,
            n_skips: 0,
            tx: req.tx,
        })));
    }
    let PrefillOutcome::Done { first, restored, hit: hit_src } = outcome else { unreachable!() };
    eprintln!(
        "[serve] prompt={prompt_tokens} tok  前缀复用={restored} tok  TTFT={:.2}s  首token={first}  会话='{sid}'",
        t0.elapsed().as_secs_f64()
    );
    // prompt 检查点已改由 run_prefill 在喂入路径上落"安全尾"档（len-2，防跨段合并失效）。

    // ── 投机（v1.5）：prime MTP → SpecGen（草稿状态每请求隔离，随 Gen 轮转）──
    // 门禁与 spec_on 同口径（上面已按 LINGXI_SERVE_SPEC_MAX_PROMPT 卡过；这里再卡是双保险）。
    let mut spec_gen: Option<SpecGen> = None;
    if ids.len() <= spec_budget {
        if let (Some(rt), Some(hp)) = (spec.as_mut(), hyper_p) {
        let primed = (|| -> Result<()> {
            let mut st = crate::spec::SpecState::new(eng.k, &eng.shape, opts.max_ctx + 528, rt.kd())?;
            rt.reset(&mut st);
            let emb_p = rt.prompt_emb(&sessions.current_mut().unwrap().ids)?;
            let _ = rt.prime(&hp, &sessions.current_mut().unwrap().ids, emb_p, &mut st)?;
            let kd = rt.kd();
            let init_res = (|| -> Result<(lingxi_driver::SpecBuffers, Vec<i64>, cudarc::driver::CudaSlice<half::bf16>, f64)> {
                let seq = sessions.current_mut().unwrap();
                let sb = seq.spec_prepare(eng, provider, kd + 1)?;
                let (tok2, _h2) = seq.decode_token_with_hyper(eng, provider)?;
                let t_ar = Instant::now();
                let (tok3, h_last) = seq.decode_token_with_hyper(eng, provider)?;
                Ok((sb, vec![tok2, tok3], h_last, t_ar.elapsed().as_secs_f64() * 1e3))
            })();
            match init_res {
                Ok((sb, toks, h_last, ar_ms)) => {
                    let mut queue = std::collections::VecDeque::new();
                    queue.extend(toks);
                    eprintln!("[serve] 投机初始化 OK（prime 成功，kd={kd}，AR 基准 {ar_ms:.1} ms/tok）");
                    spec_gen = Some(SpecGen {
                        st, sb, h_last, queue, accepted: 0, rounds: 0, verified: 0,
                        disabled: false, ar_ms,
                        eff: std::collections::VecDeque::new(),
                    });
                    Ok(())
                }
                Err(e) => Err(e),
            }
        })();
        if let Err(e) = primed {
            eprintln!("[serve] 投机初始化失败，回退纯自回归: {e:#}");
        }
        }
    }

    Ok(Some(Active::Gen(Gen {
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
        spec: spec_gen,
        tx: req.tx,
    })))
}

/// 引擎线程 active 列表的条目：正常生成（decode 轮转）或分块 prefill（块轮转）。
/// 两者共享同一列表 ⇒ 一个条目的"一步"粒度相同，prefill 块之间天然让路 decode。
enum Active {
    Gen(Gen),
    Pfx(PrefillGen),
}

impl Active {
    fn sid(&self) -> &str {
        match self {
            Active::Gen(g) => &g.sid,
            Active::Pfx(j) => &j.sid,
        }
    }
}

/// 并发引擎线程：收请求 → prefill 入列 → 轮转步进活跃生成。
fn engine_loop(
    req_rx: std::sync::mpsc::Receiver<Req>,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    perf: &PerfLog,
    spec: &mut Option<SpecRuntime<'_>>,
) {
    let mut sessions = Sessions::new(opts.max_ctx);
    let mut cache = PrefillCache::default();
    let mut pending: VecDeque<Req> = VecDeque::new();
    let mut active: Vec<Active> = Vec::new();

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
                    if active.iter().any(|a| a.sid() == sid) {
                        pending.push_back(req); // 同会话在跑：排队保序
                        continue;
                    }
                    let protect: Vec<String> = active.iter().map(|a| a.sid().to_string()).collect();
                    match setup_gen(req, opts, eng, mk_seq, &mut sessions, provider, &mut cache, &protect, spec) {
                        Ok(Some(a)) => active.push(a),
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
            // 启动排队请求（跳过与活跃任务冲突的会话）；无事可做则等下一个请求
            // （recv_timeout：shutdown 旗标要能唤醒引擎线程，不能永久阻塞在 recv 上）。
            if !try_start_pending(&mut pending, &mut active, &mut sessions, &mut cache,
                                  opts, eng, mk_seq, provider, spec)
            {
                match req_rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(req) => pending.push_back(req),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => return,
                }
            }
            continue;
        }
        // ── 轮转步进（decode 优先，两轮）：pass 1 先让所有 Gen 各走一步（生成 =
        //    一个 decode token），记 `gen_moved`；pass 2 再决定每个 Pfx——本轮
        //    没有 Gen 步进 ⇒ 喂一块（无竞争时还可按 `pfx_burst_tokens` 放宽块长）；
        //    有 Gen 步进但已连续被跳过 ≥ stall_ms ⇒ 强制喂一块（防饿死）；否则
        //    让路（跳过）。LINGXI_SERVE_PREFILL_CHUNK=0 时 active 里永远没有 Pfx，
        //    pass 2 为空转，pass 1 与原轮转逐字等价。──
        let stall_ms = serve_prefill_stall_ms();
        let mut gen_moved = false;
        let mut i = 0;
        while i < active.len() {
            if matches!(active[i], Active::Pfx(_)) {
                i += 1;
                continue;
            }
            let sid = active[i].sid().to_string();
            sessions.activate(&sid);
            let Some(seq) = sessions.current_mut() else {
                let a = active.remove(i);
                eprintln!("[serve] ⚠ 会话 '{sid}' 失踪，任务中断");
                match a {
                    Active::Gen(g) => send_ev(&g.tx, Ev::Done),
                    Active::Pfx(j) => send_ev(&j.tx, Ev::Done),
                }
                continue;
            };
            let st = {
                let Active::Gen(g) = &mut active[i] else { unreachable!() };
                gen_moved = true;
                gen_step(g, seq, eng, provider, opts, spec)
            };
            let st = match st {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[serve] 步进失败（会话 '{sid}'）: {e:#}");
                    let Active::Gen(g) = active.remove(i) else { unreachable!() };
                    if g.stream_resp {
                        let e = serde_json::json!({"error": {"message": format!("{e}"), "type": "server_error"}});
                        send_ev(&g.tx, Ev::Bytes(format!("data: {e}\n\ndata: [DONE]\n\n").into_bytes()));
                    } else {
                        send_ev(&g.tx, Ev::FinalBody { body: r#"{"error":"decode 失败"}"#.into() });
                    }
                    send_ev(&g.tx, Ev::Done);
                    finish_gen_bookkeeping_only(&g.rid, &g.sid, &mut sessions, &mut cache, eng, perf);
                    continue;
                }
            };
            match st {
                Step::More(bytes) => {
                    if !bytes.is_empty() {
                        let Active::Gen(g) = &active[i] else { unreachable!() };
                        send_ev(&g.tx, Ev::Bytes(bytes));
                    }
                    i += 1;
                }
                Step::Fin { .. } => {
                    let Active::Gen(g) = active.remove(i) else { unreachable!() };
                    finish_gen(g, st, &mut sessions, &mut cache, eng, perf);
                }
            }
        }
        // ── pass 2：分块 prefill 的喂块决策（decode 优先 + 防饿死 + 自适应块长）。
        //    chunk=0 时 active 里永远没有 Pfx，整段跳过（零开销路径不动）。──
        if serve_prefill_chunk() > 0 {
            let mut i = 0;
            while i < active.len() {
                if !matches!(active[i], Active::Pfx(_)) {
                    i += 1;
                    continue;
                }
                // decode 优先：本轮有 Gen 步进且未饿过阈 ⇒ 让路（跳过次数 +1）。
                let stalled_ms = {
                    let Active::Pfx(j) = &active[i] else { unreachable!() };
                    j.since.elapsed().as_millis()
                };
                if !pfx_should_feed(gen_moved, stalled_ms, stall_ms) {
                    if let Active::Pfx(j) = &mut active[i] {
                        j.n_skips += 1;
                    }
                    i += 1;
                    continue;
                }
                let forced = gen_moved; // 有 Gen 在动还喂 ⇒ 必是防饿死强制
                let sid = active[i].sid().to_string();
                sessions.activate(&sid);
                let Some(seq) = sessions.current_mut() else {
                    let a = active.remove(i);
                    eprintln!("[serve] ⚠ 会话 '{sid}' 失踪，任务中断");
                    match a {
                        Active::Gen(g) => send_ev(&g.tx, Ev::Done),
                        Active::Pfx(j) => send_ev(&j.tx, Ev::Done),
                    }
                    continue;
                };
                let Active::Pfx(j) = active.remove(i) else { unreachable!() };
                // 自适应块长：无竞争（active 里无其他 Gen）且剩余量超配置块 ⇒ 放宽
                // 到 min(剩余, max(块×mult, 65536))；有竞争 ⇒ 计划块长。每 Pfx 步现算。
                let fed_pre = if j.next == 0 { j.fed0 } else { j.plan.marks[j.next - 1] };
                let competing = active.iter().any(|e| matches!(e, Active::Gen(_)));
                let burst = pfx_burst_tokens(serve_prefill_chunk(),
                                             serve_prefill_burst_mult(),
                                             competing, j.ids.len() - fed_pre);
                let skips = j.n_skips;
                let (mut j, st) = prefill_step(j, seq, eng, provider, &mut cache, burst);
                j.n_skips = 0;
                j.since = Instant::now();
                match st {
                    Ok(st) => {
                        // 每块一行：块号/块长/剩余/被跳过次数/是否强制（TTFT 行保留在收尾处）。
                        let fed_post = if j.next == 0 { j.fed0 } else { j.plan.marks[j.next - 1] };
                        eprintln!(
                            "[serve] Pfx 块 {}/{}  块长={} tok  剩余={} tok  被跳过={} 次{}  会话='{}'",
                            j.n_chunks,
                            j.plan.marks.len(),
                            fed_post - fed_pre,
                            j.ids.len() - fed_post,
                            skips,
                            if forced { "  【强制：防饿死】" } else { "" },
                            j.sid,
                        );
                        match st {
                            PfxStep::More => {
                                active.insert(i, Active::Pfx(j));
                                i += 1;
                            }
                            PfxStep::Done(g) => {
                                // 最后一块产出首个 token：原位转成生成条目，下轮起参与 decode 轮转。
                                active.insert(i, Active::Gen(g));
                                i += 1;
                            }
                        }
                    }
                    Err(e) if !j.retried => {
                        // 块失败就地回退：reset + 全量重灌（allow_reuse=false 切断缓存假设），
                        // 与 setup_gen 的整块回退同一语义；只许一次。
                        j.retried = true;
                        eprintln!("[serve] 分块 prefill 块失败（回退全量重灌，会话 '{sid}'）: {e:#}");
                        cache.misses += 1;
                        let mut hyper = None;
                        match run_prefill(seq, eng, provider, &j.ids, None, &mut cache,
                                          prefix_cache_enabled(), false, &mut hyper, false,
                                          &j.sid, 0)
                        {
                            Ok(PrefillOutcome::Done { first, restored, hit }) => {
                                j.restored = restored;
                                j.hit_src = hit;
                                j.hot = false; // 全量重灌后 device_ids 由 finish_gen 收尾口径管
                                active.insert(i, Active::Gen(j.into_gen(first)));
                                i += 1;
                            }
                            Ok(PrefillOutcome::Chunked(_)) => unreachable!("chunk_sz=0 ⇒ 不可能分块"),
                            Err(e2) => {
                                eprintln!("[serve] ✗ 全量重灌也失败（会话 '{sid}'）: {e2:#}");
                                if j.stream_resp {
                                    let e2 = serde_json::json!({"error": {"message": format!("{e2}"), "type": "server_error"}});
                                    send_ev(&j.tx, Ev::Bytes(format!("data: {e2}\n\ndata: [DONE]\n\n").into_bytes()));
                                } else {
                                    send_ev(&j.tx, Ev::FinalBody { body: r#"{"error":"prefill 失败"}"#.into() });
                                }
                                send_ev(&j.tx, Ev::Done);
                                finish_gen_bookkeeping_only(&j.rid, &j.sid, &mut sessions, &mut cache, eng, perf);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[serve] 分块 prefill 重灌后仍失败（会话 '{sid}'）: {e:#}");
                        if j.stream_resp {
                            let e = serde_json::json!({"error": {"message": format!("{e}"), "type": "server_error"}});
                            send_ev(&j.tx, Ev::Bytes(format!("data: {e}\n\ndata: [DONE]\n\n").into_bytes()));
                        } else {
                            send_ev(&j.tx, Ev::FinalBody { body: r#"{"error":"prefill 失败"}"#.into() });
                        }
                        send_ev(&j.tx, Ev::Done);
                        finish_gen_bookkeeping_only(&j.rid, &j.sid, &mut sessions, &mut cache, eng, perf);
                    }
                }
            }
        }
        // 一轮步进后：起掉已释放会话的排队请求。
        let _ = try_start_pending(&mut pending, &mut active, &mut sessions, &mut cache,
                                  opts, eng, mk_seq, provider, spec);
    }
}

/// 启动排队请求（跳过与活跃生成同会话的；prefill 在函数内联完成）。
#[allow(clippy::too_many_arguments)]

fn try_start_pending(
    pending: &mut VecDeque<Req>,
    active: &mut Vec<Active>,
    sessions: &mut Sessions,
    cache: &mut PrefillCache,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    spec: &mut Option<SpecRuntime<'_>>,
) -> bool {
    let mut started = false;
    let mut i = 0;
    while i < pending.len() {
        let sid = pending[i].sid();
        if active.iter().any(|a| a.sid() == sid) {
            i += 1;
            continue;
        }
        let req = pending.remove(i).unwrap();
        let protect: Vec<String> = active.iter().map(|a| a.sid().to_string()).collect();
        match setup_gen(req, opts, eng, mk_seq, sessions, provider, cache, &protect, spec) {
            Ok(Some(a)) => {
                active.push(a);
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
    rid: &str,
    sid: &str,
    sessions: &mut Sessions,
    cache: &mut PrefillCache,
    _eng: &lingxi_driver::Engine<'_>,
    perf: &PerfLog,
) {
    if prefix_cache_enabled() {
        sessions.activate(sid);
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
            "ts": ts, "request_id": rid, "session": sid, "error": true,
        });
        perf.write_line(&(line.to_string() + "\n"));
    }
}

/// v1.5 并发路径装配：accept 线程 + 每连接线程 + 本线程 = 引擎线程。
fn run_concurrent(
    listener: TcpListener,
    opts: &ServeOpts<'_>,
    eng: &lingxi_driver::Engine<'_>,
    mk_seq: &mut dyn FnMut() -> anyhow::Result<lingxi_driver::Sequence>,
    provider: &mut dyn lingxi_driver::LayerProvider,
    spec: &mut Option<SpecRuntime<'_>>,
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
        engine_loop(req_rx, opts, eng, mk_seq, provider, &perf, spec);
    });
    Ok(())
}

#[cfg(test)]
mod chunk_plan_tests {
    //! 分块 prefill 的调度逻辑单测：块拆分计划 + 请求级选路守卫。
    use super::*;

    /// 生产实测形态：119K token 文档、块 4096、安全尾 prompt 档在 len-2。
    /// 期望：4096 对齐的 29 块 + 安全尾档一块（119117）+ 末尾 2 token 一块 = 31 块。
    #[test]
    fn plan_119k_4096() {
        let len = 119_119usize;
        let plan = build_chunk_plan(0, len, 4096, Some(len - 2), None).expect("119K/4096 必须分块");
        assert_eq!(plan.marks.len(), 31, "块数不对: {:?}", plan.marks);
        assert_eq!(plan.marks.last().copied(), Some(len));
        assert_eq!(plan.marks[plan.marks.len() - 2], len - 2, "安全尾档必须是独立块边界");
        assert!(plan.marks.windows(2).all(|w| w[0] < w[1]), "块边界必须严格递增");
        assert_eq!(plan.prompt_ckpt, Some(len - 2));
        // 首 29 块必须是 4096 对齐（生产口径：块 = 调度粒度）
        assert!(plan.marks[..29].iter().enumerate().all(|(k, &m)| m == (k + 1) * 4096));
        // 末块 = 2 token（安全尾单独补喂）
        assert_eq!(plan.marks[30] - plan.marks[29], 2);
    }

    /// 余数收尾：9000/4096 ⇒ 3 块（4096 + 4096 + 808），末块吃余数。
    #[test]
    fn plan_remainder_tail() {
        let plan = build_chunk_plan(0, 9000, 4096, None, None).expect("9000/4096 必须分块");
        assert_eq!(plan.marks, vec![4096, 8192, 9000]);
        assert_eq!(plan.marks.last().copied(), Some(9000));
    }

    /// 块大小 ≥ 待喂量（或开关关）⇒ 不分块，调用方走原整块路径（零行为变化）。
    #[test]
    fn plan_no_chunk_when_small() {
        assert!(build_chunk_plan(0, 4096, 4096, None, None).is_none());
        assert!(build_chunk_plan(0, 3000, 4096, None, None).is_none());
        assert!(build_chunk_plan(0, 119_119, 0, None, None).is_none());
        // 只剩一点增量（热继续后缀 ≤ 块）也不分块
        assert!(build_chunk_plan(119_000, 119_119, 4096, None, None).is_none());
    }

    /// 里程碑强制对齐块边界：消息边界 b=5000 不是 4096 的倍数 ⇒ 必须插进块列表；
    /// 里程碑与块倍数重合（8192）⇒ 去重，不多切块。
    #[test]
    fn plan_milestone_alignment() {
        let plan = build_chunk_plan(0, 10_000, 4096, None, Some(5000)).expect("应分块");
        assert_eq!(plan.marks, vec![4096, 5000, 8192, 10_000]);
        assert_eq!(plan.boundary_ckpt, Some(5000));
        let plan = build_chunk_plan(0, 20_000, 4096, None, Some(8192)).expect("应分块");
        assert_eq!(plan.marks, vec![4096, 8192, 12_288, 16_384, 20_000]);
        // 里程碑不在待喂区间（≤ fed 或 ≥ len）⇒ 忽略
        let plan = build_chunk_plan(6000, 20_000, 4096, None, Some(5000)).expect("应分块");
        assert_eq!(plan.boundary_ckpt, None);
        assert!(!plan.marks.contains(&5000));
    }

    /// 前缀续跑后的分块：fed>0 时块从 fed 起算，里程碑（安全尾 cut）仍对齐。
    #[test]
    fn plan_from_restored_prefix() {
        let len = 50_000usize;
        let plan = build_chunk_plan(42_000, len, 4096, Some(len - 2), None).expect("应分块");
        assert_eq!(plan.marks.first().copied(), Some(46_096));
        assert!(plan.marks.contains(&(len - 2)));
        assert_eq!(plan.marks.last().copied(), Some(len));
    }

    /// 请求级守卫：spec_on 请求永不分块（MTP 要整段 hyper）；开关 0 = 关；
    /// 剩余量 ≤ 块大小不分块。twopass 分支由 sequence.rs 的守卫读点把住
    /// （本测试环境未设 LINGXI_PREFILL_TWOPASS ⇒ 恒 false）。
    #[test]
    fn eligibility_spec_and_size_guards() {
        // spec_on ⇒ 拒绝分块（即使 119K 超块）
        assert!(!chunk_eligible(4096, true, 119_119));
        // 开关关
        assert!(!chunk_eligible(0, false, 119_119));
        // 不超块
        assert!(!chunk_eligible(4096, false, 4096));
        assert!(!chunk_eligible(4096, false, 100));
        // 正常超块请求 ⇒ 分块
        assert!(chunk_eligible(4096, false, 119_119));
        assert!(chunk_eligible(4096, false, 4097));
    }

    // ── decode 优先 + 防饿死 + 自适应块长的调度决策（纯函数）──

    /// decode 优先：本轮有 Gen 步进且 Pfx 未饿过阈 ⇒ 不喂（让路）。
    #[test]
    fn schedule_decode_priority_pfx_yields() {
        // Gen 在动、距上次喂块不足阈值 ⇒ 跳过
        assert!(!pfx_should_feed(true, 0, 1000));
        assert!(!pfx_should_feed(true, 999, 1000));
        // 本轮没有任何 Gen 步进 ⇒ 喂（哪怕刚喂过——无竞争穿插补块）
        assert!(pfx_should_feed(false, 0, 1000));
        assert!(pfx_should_feed(false, 999, 1000));
    }

    /// 防饿死：连续被跳过达到阈值 ⇒ 强制喂一块（哪怕有 Gen 在跑）。
    #[test]
    fn schedule_anti_starvation_forces_feed() {
        assert!(pfx_should_feed(true, 1000, 1000), "跳过满 1000ms ⇒ 强制");
        assert!(pfx_should_feed(true, 60_000, 1000), "长时间被跳过 ⇒ 强制");
        assert!(pfx_should_feed(true, 1000, 0), "阈值 0 = 有 Gen 也照喂（等价旧 1:1 轮转）");
        // 阈值 0 且无 Gen ⇒ 也喂（无竞争路径不受影响）
        assert!(pfx_should_feed(false, 0, 0));
    }

    /// 自适应块长·放量决策：无竞争且剩余超配置块 ⇒ min(剩余, max(块×mult, 65536),
    /// **硬上限 BURST_MAX(默认 4096)**)；有竞争 / mult=0 / 剩余不超块 ⇒ 不放大。
    #[test]
    fn schedule_burst_tokens_decision() {
        // 生产形态：119K、块 4096、mult 8 ⇒ 原 65536，被 BURST_MAX=4096 钳住
        // （单步不可抢占：65494 tok 的步实测阻塞循环 ~320s，期间新请求干等）
        assert_eq!(pfx_burst_tokens(4096, 8, false, 119_119), Some(4_096));
        // 剩余量小于上限 ⇒ 一块喂完剩余（用小块 1024 让 3000 处于 >chunk 区）
        assert_eq!(pfx_burst_tokens(1024, 8, false, 3_000), Some(3_000));
        assert_eq!(pfx_burst_tokens(4096, 8, false, 4097), Some(4_096));
        // 有竞争 ⇒ 回落计划块长（不放大）
        assert_eq!(pfx_burst_tokens(4096, 8, true, 119_119), None);
        // mult=0 ⇒ 关闭放大
        assert_eq!(pfx_burst_tokens(4096, 0, false, 119_119), None);
        // 剩余量不超配置块 ⇒ 按原计划收尾（无需放大）
        assert_eq!(pfx_burst_tokens(4096, 8, false, 4096), None);
        assert_eq!(pfx_burst_tokens(4096, 8, false, 100), None);
    }

    /// 自适应块长·块终点：burst=None 用计划块长；burst 放宽到块边界；
    /// **不得跨过检查点里程碑**（halt）——覆盖到里程碑 ⇒ 停在里程碑。
    #[test]
    fn schedule_burst_block_end() {
        // 9000/4096 ⇒ marks [4096, 8192, 9000]，无里程碑
        let marks = vec![4096, 8192, 9000];
        // 不放大：计划块长
        assert_eq!(pfx_block_end(&marks, 0, 0, None, 9000), (4096, 1));
        assert_eq!(pfx_block_end(&marks, 1, 0, None, 9000), (8192, 2));
        // 放宽 6000：上限 6000 内最大边界 = 4096（halt=len 不拦）
        assert_eq!(pfx_block_end(&marks, 0, 0, Some(6000), 9000), (4096, 1));
        // 放宽到覆盖剩余：一次喂完（末块吃余数）
        assert_eq!(pfx_block_end(&marks, 1, 0, Some(9000), 9000), (9000, 3));
        // burst 小于本块剩余 ⇒ 按计划块长
        assert_eq!(pfx_block_end(&marks, 0, 0, Some(100), 9000), (4096, 1));

        // 里程碑在界内：marks [4096, 5000(里程碑), 8192, 10000]，halt=5000
        let marks = vec![4096, 5000, 8192, 10_000];
        // 上限 6000 覆盖里程碑 5000 ⇒ 停在 5000（存档时机与逐块喂一致），不得跨过
        assert_eq!(pfx_block_end(&marks, 0, 0, Some(6000), 5000), (5000, 2));
        // 上限 9000 同样停在里程碑（floor 取 8192 会跨过 5000——本断言钉死该语义）
        assert_eq!(pfx_block_end(&marks, 0, 0, Some(9000), 5000), (5000, 2));
        // 上限未及里程碑 ⇒ 正常放宽
        assert_eq!(pfx_block_end(&marks, 0, 0, Some(4096), 5000), (4096, 1));
        // 里程碑已过（halt 在当前位置之后没有）⇒ 正常放宽到尾
        assert_eq!(pfx_block_end(&marks, 2, 0, Some(9000), 10_000), (10_000, 4));

        // 热继续 fed0>0：首块起点 fed0，burst 从 fed0 起算（上限 52000 内最大
        // 边界 = 50000，跨过一个完整计划块）
        let marks = vec![46_096, 50_000, 54_096, 58_192];
        assert_eq!(pfx_block_end(&marks, 0, 42_000, Some(10_000), 58_192), (50_000, 2));
        assert_eq!(pfx_block_end(&marks, 0, 42_000, None, 58_192), (46_096, 1));
    }

    /// 默认行为不变：LINGXI_SERVE_PREFILL_CHUNK=0 ⇒ 选路守卫恒拒绝分块、
    /// 计划恒 None ⇒ 引擎循环里永远没有 Pfx，调度逻辑（pass 2）整体不激活。
    #[test]
    fn schedule_inactive_when_chunk_off() {
        for remaining in [100, 4096, 4097, 119_119] {
            assert!(!chunk_eligible(0, false, remaining), "chunk=0 必不分块: {remaining}");
            assert!(build_chunk_plan(0, remaining, 0, None, None).is_none());
        }
    }
}
