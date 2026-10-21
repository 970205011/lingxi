#!/usr/bin/env python3
"""C2 跨请求前缀缓存验证（serve 多轮 TTFT + 逐位一致性）。

用法：
  # 服务端：LINGXI_SERVE_PREFIX_CACHE=1 ./start.sh ./bin/lingxi-bench serve <art> 11888
  serve_c2_check.py http://127.0.0.1:11888 /tmp/parity_prompt.txt multi
  # 再把上一轮回复喂回去当"首个请求"（服务端关缓存重启），比较首 token：
  serve_c2_check.py http://127.0.0.1:11888 /tmp/parity_prompt.txt fresh --reply '<上轮回复>'

`max_tokens=1` 的意义：生成的唯一 token **从未喂进 KV**（serve 收尾会 truncate_unfed），
所以第二轮的缓存是"纯 prefill 前缀" —— 续跑结果必须与整段 prefill **逐位一致**。
首 token 的 id 看服务端日志 `[serve] ... 首token=`。
"""
import argparse
import json
import time
import urllib.request


def chat(url: str, messages, max_tokens: int = 1):
    body = json.dumps(
        {"messages": messages, "max_tokens": max_tokens, "stream": False}
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body, headers={"content-type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        v = json.loads(r.read())
    return v["choices"][0]["message"]["content"], time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("prompt_file")
    p.add_argument("mode", choices=["multi", "fresh"])
    p.add_argument("--reply", default="", help="fresh 模式：上一轮 assistant 回复原文")
    a = p.parse_args()
    text = open(a.prompt_file, encoding="utf-8").read()

    if a.mode == "multi":
        c1, t1 = chat(a.url, [{"role": "user", "content": text}])
        print(f"T1   {t1:8.2f}s  reply={c1!r}")
        c2, t2 = chat(
            a.url,
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": c1},
                {"role": "user", "content": "OK"},
            ],
        )
        print(f"T2   {t2:8.2f}s  reply={c2!r}")
        print(f"SPEEDUP {t1 / max(t2, 1e-9):.1f}x")
    else:
        c2, t2 = chat(
            a.url,
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": a.reply},
                {"role": "user", "content": "OK"},
            ],
        )
        print(f"FRESH{t2:8.2f}s  reply={c2!r}")


if __name__ == "__main__":
    main()
