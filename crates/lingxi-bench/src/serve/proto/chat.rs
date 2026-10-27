//! 入站/配置侧：SpecGen 投机生成状态、ChatCfg/ReasoningEffort、prompt 构建、工具参数解析。
//! CODE_MAP Phase 4 收尾刀从 proto.rs 搬入，只搬家不改行为。

use anyhow::Result;
use std::collections::VecDeque;

use crate::spec::SpecRuntime;




/// 投机解码的**单请求状态**：MTP 运行时 + 本请求的验证缓冲 + 待发 token 队列。
///
/// 为什么队列：一轮会一次接受 `keep` 个 token，而 HTTP/SSE 是一个一个发；
/// 队列空了再跑下一轮（`SpecRuntime::round`）。
pub(crate) struct SpecGen<'a, 'k> {
    pub(crate) rt: &'a mut SpecRuntime<'k>,
    pub(crate) sb: lingxi_driver::SpecBuffers,
    pub(crate) h_last: cudarc::driver::CudaSlice<half::bf16>,
    pub(crate) queue: VecDeque<i64>,
    pub(crate) accepted: usize,
    pub(crate) rounds: usize,
    /// 已送进验证的行数（= 每轮 `chunk.len`）——用来算"每验证一行产出几个 token"
    pub(crate) verified: usize,
    /// 自适应关闸：长上下文/低接受率时投机反而更慢（实测 875-token 文档：20 vs 25 tok/s），
    /// 连续 3 轮 `产出/验证行 < 0.5` 就本请求切回纯自回归（KV 状态本来就是连续的，切换安全）。
    pub(crate) disabled: bool,
    /// AR 基准 ms/tok（初始化时 `decode_token_with_hyper` 实测）：投机每产出 tok
    /// 的墙上时间**必须低于它**，否则投机在帮倒忙。2026-10-22 实测：代码内容验证批
    /// miss 爆炸（轮均 160ms ⇒ 13.8 tok/s < AR 40.9），但接受率 4.3/轮、产出/验证行
    /// 0.87 很漂亮——产出/行数判据看不见这种亏损，必须 wall clock。
    pub(crate) ar_ms: f64,
    /// 最近各轮 ms/产出tok（墙上时钟），关闸 ② 用。
    pub(crate) eff: std::collections::VecDeque<f64>,
}

impl<'a, 'k> SpecGen<'a, 'k> {
    pub(crate) fn next(
        &mut self,
        seq: &mut lingxi_driver::Sequence,
        eng: &lingxi_driver::Engine<'_>,
        provider: &mut dyn lingxi_driver::LayerProvider,
    ) -> Result<i64> {
        // ⚠️ 顺序：**先发完上一轮队列里的 token**，再决定要不要关闸。
        // 否则关闸时会把"已提交但还没发"的 token 静默吞掉（ids 里已经有它们，AR 会从更后面接）。
        if let Some(t) = self.queue.pop_front() {
            return Ok(t);
        }
        if self.disabled {
            return seq.decode_token(eng, provider);
        }
        // 关闸（墙上时钟，唯一判据；2026-10-25 起）：最近 5 轮 ms/产出tok 的中位
        // 高于 AR 基准 15% ⇒ 切回自回归。
        // ★ 为什么删掉旧「产出/验证行 < 0.5」代理判据：kd=7 验证 8 行/轮
        // ⇒ 盈亏平衡接受率 3.0/轮，中文暖机 1.9/轮、投机比 AR 快 1.6× 也会被判死
        // （实测 42 vs 66 tok/s）。它自称"安全网（默认不会触发）"，防的 875-token 文档
        // 场景（轮均 160ms ⇒ 13.8 < AR 40.9）由本墙钟判据精确覆盖——这才是语义正确的量。
        // ⚠️ 前 8 轮是冷启动（该内容专家槽首访 = 真 miss + dense 首转换），实测 2026-10-23
        // 第 7 轮 43.7 vs AR 42.3 就被误判、整条回退 AR 只剩 24.6 tok/s；冷启动是一次性成本，
        // 必须扛过去，且留 15% 余量防单轮抖动。理由与 talk.rs 同口径。
        if !lingxi_model::flag("LINGXI_SPEC_NO_GATE") && self.rounds >= 8 && self.eff.len() >= 5 {
            let mut w: Vec<f64> = self.eff.iter().rev().take(5).copied().collect();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = w[w.len() / 2];
            if med > self.ar_ms * 1.15 {
                eprintln!(
                    "[serve] 投机最近 5 轮中位 {med:.1} ms/tok > AR 基准 {:.1}×1.15 ⇒ 切回自回归",
                    self.ar_ms
                );
                self.disabled = true;
                return seq.decode_token(eng, provider);
            }
        }
        let tr = std::time::Instant::now();
        let r = self.rt.round(seq, eng, provider, &mut self.sb, &mut self.h_last)?;
        let ms = tr.elapsed().as_secs_f64() * 1e3;
        self.accepted += r.accepted;
        self.verified += r.verified;
        self.rounds += 1;
        self.eff.push_back(ms / (r.tokens.len().max(1) as f64));
        if self.eff.len() > 8 {
            self.eff.pop_front();
        }
        self.queue.extend(r.tokens);
        Ok(self.queue.pop_front().expect("每轮至少提交 1 个 token"))
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ReasoningEffort {
    Xhigh,
    Medium,
    Low,
}

impl ReasoningEffort {
    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s {
            "xhigh" => Some(Self::Xhigh),
            "medium" => Some(Self::Medium),
            "low" => Some(Self::Low),
            _ => None,
        }
    }

    pub(crate) fn instruction(self) -> &'static str {
        match self {
            Self::Xhigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            Self::Low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            Self::Medium => "",
        }
    }
}


#[derive(Clone, Copy)]
pub(crate) struct ChatCfg {
    pub(crate) enable_thinking: bool,
    pub(crate) preserve_thinking: bool,
    pub(crate) effort: ReasoningEffort,
}

impl ChatCfg {
    pub(crate) fn reasoning_instruction(&self) -> &'static str {
        if self.enable_thinking {
            self.effort.instruction()
        } else {
            ""
        }
    }
}

/// 取消息正文（字符串或 OpenAI 的 content parts 数组）。
pub(crate) fn content_text(m: &serde_json::Value) -> String {
    match &m["content"] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// 历史 assistant 的 `tool_calls[].function.arguments`（可能是对象或 JSON 字符串）。
pub(crate) fn tool_arguments(tc: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    let f = tc.get("function").unwrap_or(tc);
    match f.get("arguments") {
        Some(serde_json::Value::Object(m)) => m.clone(),
        Some(serde_json::Value::String(s)) => serde_json::from_str::<serde_json::Value>(s)
            .ok()
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default(),
        _ => serde_json::Map::new(),
    }
}

pub(crate) const TOOLS_PREAMBLE: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
pub(crate) const TOOLS_SUFFIX: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// 按官方 `chat_template.jinja` 拼 prompt，返回 `(prompt, 最后一条消息的字符边界)`。
pub(crate) fn build_prompt(
    messages: &[serde_json::Value],
    tools: Option<&Vec<serde_json::Value>>,
    cfg: ChatCfg,
) -> (String, usize) {
    let instr = cfg.reasoning_instruction();
    let mut p = String::new();

    // ── 1) system 段（有 tools 时先写工具说明，再接原 system 内容）──
    if let Some(tools) = tools.filter(|t| !t.is_empty()) {
        p.push_str("<|im_start|>system\n");
        if !instr.is_empty() {
            p.push_str(instr);
            p.push_str("\n\n");
        }
        p.push_str(TOOLS_PREAMBLE);
        for t in tools {
            p.push('\n');
            p.push_str(&serde_json::to_string(t).unwrap_or_default());
        }
        p.push_str(TOOLS_SUFFIX);
        if messages.first().and_then(|m| m["role"].as_str()) == Some("system") {
            let c = content_text(&messages[0]).trim().to_string();
            if !c.is_empty() {
                p.push_str("\n\n");
                p.push_str(&c);
            }
        }
        p.push_str("<|im_end|>\n");
    } else if messages.first().and_then(|m| m["role"].as_str()) == Some("system") {
        let c = content_text(&messages[0]).trim().to_string();
        if !c.is_empty() {
            p.push_str("<|im_start|>system\n");
            if !instr.is_empty() {
                p.push_str(instr);
                p.push_str("\n\n");
            }
            p.push_str(&c);
            p.push_str("<|im_end|>\n");
        } else if !instr.is_empty() {
            p.push_str("<|im_start|>system\n");
            p.push_str(instr);
            p.push_str("<|im_end|>\n");
        }
    } else if !instr.is_empty() {
        p.push_str("<|im_start|>system\n");
        p.push_str(instr);
        p.push_str("<|im_end|>\n");
    }

    // ── 2) 最后一条“真实用户消息”的下标（preserve_thinking=false 时的分界）──
    let mut multi_step = true;
    let mut last_query = messages.len().saturating_sub(1);
    for (i, m) in messages.iter().enumerate().rev() {
        if multi_step && m["role"].as_str() == Some("user") {
            let c = content_text(m);
            let c = c.trim();
            if !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>")) {
                multi_step = false;
                last_query = i;
            }
        }
    }

    // ── 3) 逐条消息 ──
    let n = messages.len();
    let mut boundary = 0usize;
    let mut tool_run_start: Option<usize> = None;
    for (i, m) in messages.iter().enumerate() {
        let role = m["role"].as_str().unwrap_or("user");
        if i + 1 == n {
            boundary = if role == "tool" {
                tool_run_start.unwrap_or(p.len())
            } else {
                p.len()
            };
        }
        match role {
            // system 已在头部渲染，这里只校验位置
            "system" => {}
            "user" | "developer" => {
                p.push_str("<|im_start|>user\n");
                p.push_str(content_text(m).trim());
                p.push_str("<|im_end|>\n");
                tool_run_start = None;
            }
            "assistant" => {
                let content = content_text(m);
                let content = content.trim();
                let reasoning = m["reasoning_content"].as_str().unwrap_or("").trim();
                let include_think = cfg.preserve_thinking || i > last_query;
                p.push_str("<|im_start|>assistant\n");
                if include_think {
                    p.push_str("<think>\n");
                    p.push_str(reasoning);
                    p.push_str("\n</think>\n\n");
                }
                p.push_str(content);
                if let Some(tcs) = m["tool_calls"].as_array() {
                    for (ti, tc) in tcs.iter().enumerate() {
                        let f = tc.get("function").unwrap_or(tc);
                        let name = f["name"].as_str().unwrap_or("");
                        if ti == 0 {
                            if content.is_empty() {
                                p.push_str("<tool_call>\n<function=");
                            } else {
                                p.push_str("\n\n<tool_call>\n<function=");
                            }
                        } else {
                            p.push_str("\n<tool_call>\n<function=");
                        }
                        p.push_str(name);
                        p.push_str(">\n");
                        for (k, val) in tool_arguments(tc) {
                            p.push_str("<parameter=");
                            p.push_str(&k);
                            p.push_str(">\n");
                            match val {
                                serde_json::Value::String(s) => p.push_str(&s),
                                other => p.push_str(&other.to_string()),
                            }
                            p.push_str("\n</parameter>\n");
                        }
                        p.push_str("</function>\n</tool_call>");
                    }
                }
                p.push_str("<|im_end|>\n");
                tool_run_start = None;
            }
            "tool" => {
                let first_of_run = i == 0 || messages[i - 1]["role"].as_str() != Some("tool");
                if first_of_run {
                    tool_run_start = Some(p.len());
                    p.push_str("<|im_start|>user");
                }
                p.push_str("\n<tool_response>\n");
                p.push_str(content_text(m).trim());
                p.push_str("\n</tool_response>");
                let next_is_tool =
                    i + 1 < n && messages[i + 1]["role"].as_str() == Some("tool");
                if !next_is_tool {
                    p.push_str("<|im_end|>\n");
                    tool_run_start = None;
                }
            }
            _ => {
                p.push_str("<|im_start|>user\n");
                p.push_str(content_text(m).trim());
                p.push_str("<|im_end|>\n");
                tool_run_start = None;
            }
        }
    }

    // ── 4) 生成头 ──
    p.push_str("<|im_start|>assistant\n");
    if cfg.enable_thinking {
        p.push_str("<think>\n");
    } else {
        p.push_str("<think>\n\n</think>\n\n");
    }
    (p, boundary)
}
