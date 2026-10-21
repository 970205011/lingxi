#!/usr/bin/env python3
"""智能前缀缓存（C3）冒烟：三轮对话，第三轮只改最后一句。

用法：python3 tools/golden/serve_cache_test.py <port> <prompt_file>
  服务端：./start.sh ./bin/lingxi-bench serve <产物> 11434   （缓存默认开）

预期日志（`[serve] prompt=… 前缀复用=… TTFT=…`）：
  T1 全量；T2 复用“上一轮末尾”；T3 复用“最后一句之前”的消息边界检查点 ⇒ TTFT 大幅下降。
"""
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
long_text = open(sys.argv[2], encoding="utf-8").read()


def chat(messages, max_tokens=4):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    v = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    dt = time.time() - t0
    n = (v.get("usage") or {}).get("completion_tokens", 0)
    return dt, n, v["choices"][0]["message"]["content"]


base = [
    {"role": "user", "content": long_text},
    {"role": "assistant", "content": "OK"},
]
t1, n1, c1 = chat(base + [{"role": "user", "content": "第一问：这段话主要在讲什么？"}])
print(f"T1（全量）      wall={t1:7.2f}s completion={n1}  {c1!r}")
t2, n2, c2 = chat(base + [{"role": "user", "content": "第二问：换成一句话概括。"}])
print(f"T2（只改末句）  wall={t2:7.2f}s completion={n2}  {c2!r}")
t3, n3, c3 = chat(base + [{"role": "user", "content": "第二问：换成一句话概括。"}])
print(f"T3（原样重发）  wall={t3:7.2f}s completion={n3}  {c3!r}")
print(f"加速：T1/T2={t1/max(t2,1e-9):.1f}x  T1/T3={t1/max(t3,1e-9):.1f}x")
