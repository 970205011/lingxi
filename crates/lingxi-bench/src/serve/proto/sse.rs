//! 出站/流式侧：OutEv 输出解析、StreamSplitter、SSE/chunked 响应、Detok/StopFilter、
//! 事件渲染与写回。CODE_MAP Phase 4 收尾刀从 proto.rs 搬入，只搬家不改行为。

use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;



// ───────────────────────── 输出解析（思考 / 正文 / 工具调用）─────────────────────────

pub(crate) enum OutEv {
    Think(String),
    Content(String),
    /// 一个完整的 `<tool_call>…</tool_call>` 原文
    Call(String),
}

/// 流式解析：标签可能被切在两个 token 之间，所以尾部像“半个标签”时扣住不发。
pub(crate) struct StreamSplitter {
    pub(crate) buf: String,
    pub(crate) in_think: bool,
    /// 正在攒 `<tool_call>` 块
    pub(crate) tool: Option<String>,
    /// 已经发出几个工具调用（用于 `index`/id）
    pub calls: usize,
}

impl StreamSplitter {
    pub(crate) fn new(enable_thinking: bool) -> Self {
        Self { buf: String::new(), in_think: enable_thinking, tool: None, calls: 0 }
    }

    pub(crate) fn push(&mut self, s: &str, out: &mut Vec<OutEv>) {
        if s.is_empty() {
            return;
        }
        self.buf.push_str(s);
        loop {
            if let Some(tb) = self.tool.as_mut() {
                if let Some(pos) = self.buf.find("</tool_call>") {
                    let end = pos + "</tool_call>".len();
                    tb.push_str(&self.buf[..end]);
                    self.buf.drain(..end);
                    let block = self.tool.take().unwrap();
                    self.calls += 1;
                    out.push(OutEv::Call(block));
                    continue;
                }
                tb.push_str(&std::mem::take(&mut self.buf));
                break;
            }

            if self.in_think {
                // 正常：先收 `</think>`；异常：模型没关思考就发工具调用，也当思考结束
                let close = self.buf.find("</think>");
                let call = self.buf.find("<tool_call>");
                let (pos, len, to_call) = match (close, call) {
                    (Some(c), Some(k)) if k < c => (k, "<tool_call>".len(), true),
                    (Some(c), _) => (c, "</think>".len(), false),
                    (None, Some(k)) => (k, "<tool_call>".len(), true),
                    (None, None) => {
                        let keep = partial_tag_suffix(&self.buf, "</think>")
                            .max(partial_tag_suffix(&self.buf, "<tool_call>"));
                        if keep > 0 {
                            let cut = self.buf.len() - keep;
                            if cut > 0 {
                                out.push(OutEv::Think(self.buf[..cut].to_string()));
                                self.buf.drain(..cut);
                            }
                        } else if !self.buf.is_empty() {
                            out.push(OutEv::Think(std::mem::take(&mut self.buf)));
                        }
                        break;
                    }
                };
                if pos > 0 {
                    out.push(OutEv::Think(self.buf[..pos].to_string()));
                }
                self.buf.drain(..pos + len);
                if to_call {
                    self.in_think = false;
                    self.tool = Some("<tool_call>".to_string());
                } else {
                    self.in_think = false;
                }
                continue;
            }

            // 正文态：找 `<tool_call>` 或误出的 `<think>`
            let call = self.buf.find("<tool_call>");
            let think = self.buf.find("<think>");
            let (pos, len, is_call) = match (call, think) {
                (Some(c), Some(t)) if t < c => (t, "<think>".len(), false),
                (Some(c), _) => (c, "<tool_call>".len(), true),
                (None, Some(t)) => (t, "<think>".len(), false),
                (None, None) => {
                    let keep = partial_tag_suffix(&self.buf, "<tool_call>")
                        .max(partial_tag_suffix(&self.buf, "<think>"));
                    if keep > 0 {
                        let cut = self.buf.len() - keep;
                        if cut > 0 {
                            out.push(OutEv::Content(self.buf[..cut].to_string()));
                            self.buf.drain(..cut);
                        }
                    } else if !self.buf.is_empty() {
                        out.push(OutEv::Content(std::mem::take(&mut self.buf)));
                    }
                    break;
                }
            };
            if pos > 0 {
                out.push(OutEv::Content(self.buf[..pos].to_string()));
            }
            self.buf.drain(..pos + len);
            if is_call {
                self.tool = Some("<tool_call>".to_string());
            } else {
                self.in_think = true;
            }
        }
    }

    pub(crate) fn flush(&mut self, out: &mut Vec<OutEv>) {
        if let Some(mut tb) = self.tool.take() {
            tb.push_str(&std::mem::take(&mut self.buf));
            self.calls += 1;
            out.push(OutEv::Call(tb));
            return;
        }
        if !self.buf.is_empty() {
            let s = std::mem::take(&mut self.buf);
            out.push(if self.in_think { OutEv::Think(s) } else { OutEv::Content(s) });
        }
    }
}

/// 解析一个 `<tool_call>` 块 → `(函数名, 参数对象)`。
pub(crate) fn parse_tool_block(block: &str) -> Option<(String, serde_json::Value)> {
    let start = block.find("<function=")?;
    let rest = &block[start + "<function=".len()..];
    let name_end = rest.find('>')?;
    let name = rest[..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut body = &rest[name_end + 1..];
    let mut args = serde_json::Map::new();
    while let Some(p0) = body.find("<parameter=") {
        let after = &body[p0 + "<parameter=".len()..];
        let k_end = after.find('>')?;
        let key = after[..k_end].trim().to_string();
        let vstart = k_end + 1;
        let vend = after[vstart..].find("</parameter>")? + vstart;
        let raw = &after[vstart..vend];
        // 模板固定写成 `<parameter=键>\n值\n</parameter>`，两端的换行是格式分隔符
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        args.insert(key, json_or_string(raw));
        body = &after[vend + "</parameter>".len()..];
    }
    Some((name, serde_json::Value::Object(args)))
}

/// 参数值优先当 JSON（对象/数组/数字/布尔/null），否则按原文字符串。
pub(crate) fn json_or_string(raw: &str) -> serde_json::Value {
    let t = raw.trim();
    let looks_json = t.starts_with('{')
        || t.starts_with('[')
        || t == "true"
        || t == "false"
        || t == "null"
        || t.parse::<f64>().is_ok();
    if looks_json {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(t) {
            return v;
        }
    }
    serde_json::Value::String(raw.to_string())
}

/// 读 `Transfer-Encoding: chunked` 的请求体（部分客户端/网关发大 body 时用）。
pub(crate) fn read_chunked(rdr: &mut BufReader<TcpStream>, max: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let mut line = String::new();
        rdr.read_line(&mut line)?;
        let sz = line.trim().split(';').next().unwrap_or("0").to_string();
        let n = usize::from_str_radix(&sz, 16)
            .with_context(|| format!("chunk 长度非法: {sz:?}"))?;
        anyhow::ensure!(
            out.len().saturating_add(n) <= max,
            "chunked body 超过上限 {max} 字节"
        );
        if n == 0 {
            // 跳过 trailers
            loop {
                let mut t = String::new();
                rdr.read_line(&mut t)?;
                if t.trim().is_empty() {
                    break;
                }
            }
            break;
        }
        let start = out.len();
        out.resize(start + n, 0);
        rdr.read_exact(&mut out[start..])?;
        let mut crlf = [0u8; 2];
        rdr.read_exact(&mut crlf)?;
    }
    Ok(out)
}

/// prefill 期间给 SSE 连接定期写注释行，避免网关/客户端因长时间无字节而超时断连
/// （实测 24K token 冷 prefill 243s，期间一个字节都没有 ⇒ 客户端先超时，回写 Broken pipe）。
pub(crate) struct KeepAlive {
    pub(crate) stop: Arc<AtomicBool>,
    pub(crate) h: Option<std::thread::JoinHandle<()>>,
}

impl KeepAlive {
    pub(crate) fn start(s: TcpStream) -> Self {
        Self::start_with(s, b": keep-alive\n\n")
    }

    /// 非流式用 chunked：每 5s 发一个只含空格的块（JSON 允许前导空白）。
    pub(crate) fn start_chunked(s: TcpStream) -> Self {
        Self::start_with(s, b"1\r\n \r\n")
    }

    pub(crate) fn start_with(s: TcpStream, payload: &'static [u8]) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let st = stop.clone();
        let h = std::thread::spawn(move || {
            let mut s = s;
            loop {
                // 5s 一跳；200ms 粒度便于及时退出
                for _ in 0..25 {
                    if st.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                if s.write_all(payload).is_err() || s.flush().is_err() {
                    return;
                }
            }
        });
        Self { stop, h: Some(h) }
    }
}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
    }
}

/// 流式请求的**早期**响应头 + role 块：必须在 prefill 之前发出去。
pub(crate) fn sse_headers(stream: &mut TcpStream, rid: &str) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
         cache-control: no-cache\r\nx-accel-buffering: no\r\nconnection: close\r\n\r\n"
    )?;
    let role = serde_json::json!({
        "id": rid,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "lingxi",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    });
    stream.write_all(format!("data: {role}\n\n").as_bytes())?;
    stream.flush()?;
    Ok(())
}

/// 头已经发出去之后出错：只能走 SSE 错误事件，不能再回 500。
pub(crate) fn sse_fail(stream: &mut TcpStream, msg: &str) -> Result<()> {
    let v = serde_json::json!({"error": {"message": msg, "type": "server_error"}});
    stream.write_all(format!("data: {v}\n\n").as_bytes())?;
    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()?;
    Ok(())
}

/// 非流式请求的早期响应头（chunked，便于 prefill 期间保活）。
pub(crate) fn chunked_headers(stream: &mut TcpStream) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\n\
         cache-control: no-cache\r\nx-accel-buffering: no\r\nconnection: close\r\n\r\n"
    )?;
    stream.flush()?;
    Ok(())
}

/// 写一个 chunked 响应体并收尾。
pub(crate) fn respond_chunked(stream: &mut TcpStream, body: &[u8]) -> Result<()> {
    write!(stream, "{:x}\r\n", body.len())?;
    stream.write_all(body)?;
    stream.write_all(b"\r\n0\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}

// ─────────────────────────── ㊿-P1 会话驻留（多会话·单物化） ───────────────────────────
//
// 形态（PLAN_NEXT P1 v1）：**任意数量会话驻留，但同时只物化一个 Sequence**。
// 切换时当前会话整份 `save_host_checkpoint` 沉到 host RAM（秒级唤醒 = H2D 灌回，
// 不是重跑 prefill）；沉没层总量受 `LINGXI_SERVE_SUNK_MB`（默认 32GB）约束，超了按 LRU 丢
// （数据损失，打响亮日志）。KV 预算走 `KvBudget`（pledge = prompt+生成+余量）；
// CUDA Graph 按会话 epoch 重新录制（图体烘死旧会话地址，必须隔离）。

/// 增量反分词：逐 token 喂入，等一个字符的 UTF-8 字节补齐后再吐。
///
/// 不能直接 `decode(&[id])`：byte-level BPE 的单个 token 可能只含多字节字符的前 1~2 个字节，
/// 单独解码会得到 U+FFFD（实测 emoji 被拆成三个 `�`，中文也会碎）。
pub(crate) struct Detok {
    pub(crate) pending: Vec<i64>,
}

impl Detok {
    pub(crate) fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// 喂一个 token，返回**可以安全输出**的文本（可能为空 = 还在等后续字节）。
    pub(crate) fn push(&mut self, tok: &lingxi_tokenize::Tokenizer, id: i64) -> String {
        self.pending.push(id);
        let s = tok.decode(&self.pending).unwrap_or_default();
        // 末尾是替换符 ⇒ 字节还不完整；最多攒 8 个 token 兜底，避免坏 token 卡住输出
        if s.ends_with('\u{FFFD}') && self.pending.len() < 8 {
            return String::new();
        }
        self.pending.clear();
        s
    }
}

pub(crate) fn partial_tag_suffix(s: &str, tag: &str) -> usize {
    let max = tag.len().min(s.len());
    for k in (1..=max).rev() {
        // tag 可能是中文 ⇒ 只能按字符边界切
        if !tag.is_char_boundary(k) {
            continue;
        }
        if s.ends_with(&tag[..k]) {
            return k;
        }
    }
    0
}

/// OpenAI `stop` 序列过滤：会扣住“可能是一个 stop 的前缀”的尾巴，避免先发出去再撤回。
pub(crate) struct StopFilter {
    pub(crate) stops: Vec<String>,
    pub(crate) buf: String,
    pub hit: bool,
}

impl StopFilter {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        Self { stops, buf: String::new(), hit: false }
    }

    pub(crate) fn active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// 喂一段正文，返回**可以安全发出**的片段。
    pub(crate) fn push(&mut self, s: &str) -> Vec<String> {
        if self.hit || !self.active() {
            return if s.is_empty() { Vec::new() } else { vec![s.to_string()] };
        }
        self.buf.push_str(s);
        let mut out = Vec::new();
        if let Some(pos) = self
            .stops
            .iter()
            .filter_map(|st| self.buf.find(st.as_str()))
            .min()
        {
            if pos > 0 {
                out.push(self.buf[..pos].to_string());
            }
            self.buf.clear();
            self.hit = true;
            return out;
        }
        let hold = self
            .stops
            .iter()
            .map(|st| partial_tag_suffix(&self.buf, st))
            .max()
            .unwrap_or(0);
        let cut = self.buf.len().saturating_sub(hold);
        if cut > 0 {
            out.push(self.buf[..cut].to_string());
            self.buf.drain(..cut);
        }
        out
    }

    pub(crate) fn finish(&mut self) -> Option<String> {
        if self.hit || self.buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buf))
        }
    }
}

/// 一个 SSE data 行（`field` = `content` / `reasoning_content`）。
pub(crate) fn sse_chunk(rid: &str, field: &str, text: &str) -> String {
    let v = serde_json::json!({
        "id": rid,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "lingxi",
        "choices": [{"index": 0, "delta": {(field): text}, "finish_reason": null}]
    });
    format!("data: {v}\n\n")
}

/// 一个完整的工具调用 delta（OpenAI 格式：`arguments` 是 JSON 字符串）。
pub(crate) fn sse_tool_chunk(rid: &str, index: usize, call_id: &str, name: &str, args: &str) -> String {
    let v = serde_json::json!({
        "id": rid,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "lingxi",
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args}
            }]},
            "finish_reason": null
        }]
    });
    format!("data: {v}\n\n")
}

/// 把 OutEv 渲染成 SSE 帧字节（v1.5：引擎线程产出字节、连接线程写 socket）。
pub(crate) fn render_events(
    out: &mut Vec<u8>,
    events: &mut Vec<OutEv>,
    call_idx: &mut usize,
    rid: &str,
    stop: &mut StopFilter,
) -> Result<bool> {
    let mut stopped = false;
    for ev in events.drain(..) {
        match ev {
            OutEv::Think(t) => {
                if !t.is_empty() {
                    out.extend_from_slice(sse_chunk(rid, "reasoning_content", &t).as_bytes());
                }
            }
            OutEv::Content(t) => {
                for piece in stop.push(&t) {
                    if !piece.is_empty() {
                        out.extend_from_slice(sse_chunk(rid, "content", &piece).as_bytes());
                    }
                }
                if stop.hit {
                    stopped = true;
                }
            }
            OutEv::Call(block) => {
                if let Some((name, args)) = parse_tool_block(&block) {
                    let id = format!("call_{}", *call_idx);
                    out.extend_from_slice(
                        sse_tool_chunk(rid, *call_idx, &id, &name, &args.to_string()).as_bytes(),
                    );
                    *call_idx += 1;
                } else {
                    // 解析失败就把原文当正文发出去，不吞内容
                    out.extend_from_slice(sse_chunk(rid, "content", &block).as_bytes());
                }
            }
        }
    }
    Ok(stopped)
}

pub(crate) fn write_events(
    stream: &mut TcpStream,
    events: &mut Vec<OutEv>,
    call_idx: &mut usize,
    rid: &str,
    stop: &mut StopFilter,
) -> Result<bool> {
    let mut buf = Vec::new();
    let stopped = render_events(&mut buf, events, call_idx, rid, stop)?;
    stream.write_all(&buf)?;
    stream.flush()?;
    Ok(stopped)
}

/// 非流式：把事件累加进正文 / 思考 / 工具调用。
pub(crate) fn collect_events(
    events: &mut Vec<OutEv>,
    content: &mut String,
    reasoning: &mut String,
    tool_calls: &mut Vec<serde_json::Value>,
    stop: &mut StopFilter,
) -> bool {
    let mut stopped = false;
    for ev in events.drain(..) {
        match ev {
            OutEv::Think(t) => reasoning.push_str(&t),
            OutEv::Content(t) => {
                for piece in stop.push(&t) {
                    content.push_str(&piece);
                }
                if stop.hit {
                    stopped = true;
                }
            }
            OutEv::Call(block) => {
                if let Some((name, args)) = parse_tool_block(&block) {
                    tool_calls.push(serde_json::json!({
                        "id": format!("call_{}", tool_calls.len()),
                        "type": "function",
                        "function": {"name": name, "arguments": args.to_string()}
                    }));
                } else {
                    content.push_str(&block);
                }
            }
        }
    }
    stopped
}

/// SSE：逐 token 发 chunk，发完 `[DONE]` 关连接（HTTP/1.1 无长度 + connection:close）。
#[allow(clippy::too_many_arguments)]

pub(crate) fn respond(stream: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> Result<()> {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {code} {reason}\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}



// ═══════════════════════════ v1.5 并发路径（多物化 + 真并发） ═══════════════════════════
//
// 结构（PLAN_NEXT P1→v1.5）：
// 连接线程（每连接一个）：读 HTTP/请求体 → `Req` 送进队列 → 阻塞转发 `Ev` 字节回 socket
// 引擎线程（唯一）：收 `Req` → 物化/切换会话（`Sessions`）→ prefill → 把生成拆成
// 每会话一个 `Gen`，**轮转步进**（每轮每个活跃 Gen 一个 token）——单流时间片，
// 无队头阻塞；引擎内部（Kernels/scratch/图固定槽/路由工作区）全线程局部，
// 串行步进下天然安全。
// 会话隔离：每会话 per-session CUDA Graph epoch（切换 = epoch_switch，图集各自保留；
// 新建/唤醒 = epoch_fresh；沉没 = epoch_purge）。`LINGXI_SERVE_MATERIALIZED`（默认 2）
// 控制同时物化上限，超配把最久未动的 live 会话沉到 host（生成中的会话受 protect）。
// 投机解码与并发调度不兼容（草稿态绑会话）——投机开时整体走串行路径（见 `run`）。
