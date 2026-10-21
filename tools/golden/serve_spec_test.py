#!/usr/bin/env python3
"""serve 冒烟/测速：`python3 tmp/serve_spec_test.py <port> <max_tokens> [prompt]`"""
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
mt = int(sys.argv[2])
prompt = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"
if prompt.startswith("@"):
    prompt = open(prompt[1:], encoding="utf-8").read()
body = json.dumps(
    {"messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "stream": False}
).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=body,
    headers={"content-type": "application/json"},
)
t0 = time.time()
v = json.loads(urllib.request.urlopen(req, timeout=7200).read())
dt = time.time() - t0
u = v.get("usage") or {}
n = u.get("completion_tokens", 0)
print("wall=%.2fs  completion=%s  => %.1f tok/s" % (dt, n, n / dt if dt > 0 else 0))
print(repr(v["choices"][0]["message"]["content"]))
