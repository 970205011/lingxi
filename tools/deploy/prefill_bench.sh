#!/usr/bin/env bash
# 维护窗口专用：固定 prompt 长度的**冷 prefill** 基准。
# ⚠️ 会真的占用唯一序列（单实例串行），不要在用户使用中跑。
# 用法：PORT=8000 TOKENS=24000 MAXNEW=8 bash tools/deploy/prefill_bench.sh
set -euo pipefail
PORT=${PORT:-8000}
TOKENS=${TOKENS:-24000}
MAXNEW=${MAXNEW:-8}

python3 - "$PORT" "$TOKENS" "$MAXNEW" <<'PY'
import json, os, random, sys, time, urllib.request

port, tokens, maxnew = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
nonce = f"[run {random.randrange(10**9)}] "  # 随机前缀：强制冷启动，不命中前缀缓存

# 用仓库里的真实代码拼 prompt：专家路由分布比重复句子分散得多，更接近编程场景。
# 可用 BENCH_FILE=a,b,c 覆盖。
files = os.environ.get("BENCH_FILE", "").split(",") if os.environ.get("BENCH_FILE") else [
    "crates/lingxi-bench/src/main.rs",
    "crates/lingxi-bench/src/serve.rs",
    "crates/lingxi-model/src/qsa.rs",
    "crates/lingxi-driver/src/lib.rs",
]
text = ""
for p in files:
    try:
        text += open(p, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        pass
if not text:
    text = "The quick brown fox jumps over the lazy dog. "
chars = max(1, tokens * 4)
while len(text) < chars:
    text += "\n" + text
body_text = nonce + text[:chars]
payload = {
    "model": "lingxi",
    "stream": True,
    "max_tokens": maxnew,
    "enable_thinking": False,
    "messages": [{"role": "user", "content": "请只回答 OK。材料：\n" + body_text}],
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
t0 = time.time()
first = None
usage = None
with urllib.request.urlopen(req, timeout=7200) as f:
    for line in f:
        s = line.decode().strip()
        if not s.startswith("data: "):
            continue
        s = s[6:]
        if s == "[DONE]":
            break
        d = json.loads(s)
        if "choices" not in d:
            print("SERVER_ERROR:", json.dumps(d, ensure_ascii=False))
            sys.exit(1)
        ch = d["choices"][0]
        if (ch.get("delta") or {}).get("content") and first is None:
            first = time.time()
        if d.get("usage"):
            usage = d["usage"]
ttft = (first - t0) if first else -1.0
pt = usage["prompt_tokens"] if usage else -1
print(
    "prompt_tokens=%s ttft_s=%.2f prefill_tok_s=%.1f total_s=%.2f"
    % (pt, ttft, (pt / ttft if ttft > 0 else 0.0), time.time() - t0)
)
PY
