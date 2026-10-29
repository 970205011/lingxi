#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KV FP8-G32（LINGXI_KV_FP8）质量判据脚本——三组消融 + 病灶指标。

判决问题（与 K8V4 翻车同口径）：同一批测试 prompt，bf16 KV vs FP8 KV **贪心解码**，
输出差多少？本版跑 **三组消融**（残差窗的贡献拆开看）：

  1. int8+res   —— `LINGXI_KV_FP8=1`（MODE 默认 int8，残差窗默认开）
                   **安全档**：int8 g32 相对误差 ~0.4%，文献口径 4-bit per-channel 即
                   近无损，应过最严判据（mean KL < 0.005 且首 token 一致）。
  2. e4m3+res   —— + `LINGXI_KV_FP8_MODE=e4m3`（残差窗开）
  3. e4m3-nores —— + `LINGXI_KV_FP8_RESIDUAL=0`（全量量化消融：单独看残差窗的贡献）

指标（每个 prompt × 每个配置 vs bf16 基准）：

  a. mean KL        —— 逐步 next-token 分布的 KL(bf16 ‖ cfg) 均值（需要 logits 探针；
                       ⚠️ **待 GPU 窗口实测**：引擎目前不导出逐步 logits，本脚本已
                       定义好输入格式，探针接上后直接出数）。
  b. top-1 一致率   —— 贪心 argmax（= 实际输出的 token）在公共前缀上的逐位一致率。
  c. 首 token id    —— **病灶指标**：K8V4 翻车的直接表现是首 token 翻转
                       （实测"首 token 1543 应 11751"——attention sink 被量化）。
  d. 短序列 md5     —— 生成文本 md5（与回归 golden 同口径）。

用法：
  # 跑实验（需要 GPU；默认 4 个内置短 prompt，可用 --prompt-file 追加）：
  python3 tools/kvfp8_quality.py run --bench ./bin/lingxi-bench \
      --artifact /models/qwen4_exp-artifact/qwen4_exp.ninfer --tokens 128

  # 离线算指标（对已抓取的输出）：
  python3 tools/kvfp8_quality.py compare --dir run/kvfp8_quality

  # KL 需要 logits 探针按以下格式落盘（--run 时自动识别，缺则该项记 N/A）：
  #   <outdir>/<prompt_slug>/<cfg>/logprobs.jsonl  每行
  #   {"step": 0, "top": [[token_id, logp], ...]}   # top 档取 top-64 即可（其余质量忽略不计）

⚠️ 状态标注（2026-10-28）：脚本就绪（三组消融版），**尚未在 GPU 窗口实测**。跑之前确认：
  - 独占 GPU（别与其他生产实例共用）；
  - 固定随机性：贪心解码 + 确定性路由 ⇒ 同配置多次运行应逐位复现（先跑两遍 bf16
    自检 md5 全同，再开 LINGXI_KV_FP8=1 对比）；
  - spec（LINGXI_TALK_SPEC / LINGXI_SERVE_SPEC）**不要开**——K8V4 实测 FP8 KV 杀
    主干 argmax ⇒ 草稿接受率归零，spec 数字会把质量信号淹没（判据只看 AR 贪心）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_PROMPTS = [
    # 短序列（md5 口径）：覆盖枚举/事实/代码/中文四类典型生成
    "The capital of France is",
    "List the first five prime numbers.",
    "Write a Python function that reverses a string.",
    "请用三句话介绍杭州。",
]

# 三组消融配置（相对 bf16 基准）。残差窗参数可调（SINK/RECENT）做敏感性。
CFG_ENVS = {
    "int8+res": {"LINGXI_KV_FP8": "1"},
    "e4m3+res": {"LINGXI_KV_FP8": "1", "LINGXI_KV_FP8_MODE": "e4m3"},
    "e4m3-nores": {"LINGXI_KV_FP8": "1", "LINGXI_KV_FP8_MODE": "e4m3",
                   "LINGXI_KV_FP8_RESIDUAL": "0"},
}

# 判据门槛建议（见 docs/kv-fp8-design.md）：int8 安全档过最严，e4m3 放宽一档。
# 全部要求"首 token 一致"（K8V4 的病灶指标，零容忍）。
GATES = {
    "int8+res": {"kl": 0.005, "first_token": True},
    "e4m3+res": {"kl": 0.02, "first_token": True},
    # 消融档不设门槛——它存在的意义是暴露残差窗的贡献，翻车本身提供信息。
    "e4m3-nores": {"kl": None, "first_token": False},
}

IDS_RE = re.compile(r"^\[ids\]\s*\[([0-9,\s]*)\]", re.M)


def _slug(prompt: str) -> str:
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", prompt)[:32].strip("_")
    return s or "prompt"


def _parse_ids(stderr_text: str) -> list[int]:
    m = IDS_RE.search(stderr_text)
    if not m:
        return []
    return [int(x) for x in m.group(1).split(",") if x.strip()]


def cmd_run(a: argparse.Namespace) -> int:
    outdir = Path(a.dir)
    prompts = list(DEFAULT_PROMPTS)
    for pf in a.prompt_file or []:
        prompts += [l.strip() for l in Path(pf).read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.startswith("#")]
    base_env = dict(os.environ)
    # 判据纪律：AR 贪心。spec 会改变采样路径（草稿接受），把质量信号与吞吐信号搅在一起。
    for bad in ("LINGXI_TALK_SPEC", "LINGXI_MTP_SPEC", "LINGXI_KV_K8V4",
                "LINGXI_KV_FP8", "LINGXI_KV_FP8_MODE", "LINGXI_KV_FP8_RESIDUAL"):
        base_env.pop(bad, None)

    for prompt in prompts:
        slug = _slug(prompt)
        for cfg, extra in (("bf16", {}), *CFG_ENVS.items()):
            cfg_dir = outdir / slug / cfg
            cfg_dir.mkdir(parents=True, exist_ok=True)
            env = dict(base_env, LINGXI_DUMP_IDS="1", **extra)
            print(f"[run] {slug}/{cfg}: {prompt[:40]!r} ...", flush=True)
            p = subprocess.run(
                [a.bench, "talk", a.artifact, prompt, str(a.tokens)],
                capture_output=True, text=True, env=env,
            )
            (cfg_dir / "stdout.txt").write_text(p.stdout, encoding="utf-8")
            (cfg_dir / "stderr.txt").write_text(p.stderr, encoding="utf-8")
            ids = _parse_ids(p.stderr)
            (cfg_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
            if p.returncode != 0:
                print(f"  ⚠️ 退出码 {p.returncode}（判据中止前先看 stderr.txt）", flush=True)
    print(f"[run] 完成，输出在 {outdir}/；接着跑: python3 {sys.argv[0]} compare --dir {outdir}")
    return 0


def _load_ids(d: Path) -> list[int]:
    f = d / "ids.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _load_logprobs(d: Path) -> list[dict] | None:
    f = d / "logprobs.jsonl"
    if not f.exists():
        return None
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def _kl_rows(rows_a: list[dict], rows_b: list[dict]) -> float | None:
    """mean KL(bf16 ‖ cfg)：按 step 对齐，top 档外的质量忽略不计（两边同口径）。"""
    if not rows_a or not rows_b:
        return None
    kb = {r["step"]: {int(t): float(lp) for t, lp in r["top"]} for r in rows_b}
    kls = []
    for r in rows_a:
        pa = {int(t): float(lp) for t, lp in r["top"]}
        pb = kb.get(r["step"])
        if pb is None:
            continue
        # 对称化 top 集合：缺失项给一个远小于 top-64 尾部的地板 logp
        floor = min(min(pa.values()), min(pb.values())) - 20.0
        kl = 0.0
        for t, la in pa.items():
            p_a = math.exp(la)
            kl += p_a * (la - pb.get(t, floor))
        kls.append(kl)
    return sum(kls) / len(kls) if kls else None


def _verdict(cfg: str, v: dict) -> str:
    gate = GATES.get(cfg, {})
    msgs = []
    if gate.get("first_token"):
        ok = v["first_token_match"]
        msgs.append(f"首token{'✓' if ok else '✗(翻转!)'}")
    if gate.get("kl") is not None and isinstance(v["mean_kl"], float):
        ok = v["mean_kl"] < gate["kl"]
        msgs.append(f"KL {'✓' if ok else '✗'}({v['mean_kl']:.5f}<{gate['kl']})")
    return " ".join(msgs)


def cmd_compare(a: argparse.Namespace) -> int:
    outdir = Path(a.dir)
    if not outdir.is_dir():
        print(f"[compare] 目录不存在：{outdir}（先跑 run 子命令）", file=sys.stderr)
        return 1
    results = {}
    for slug_dir in sorted(p for p in outdir.iterdir() if p.is_dir()):
        bf = slug_dir / "bf16"
        if not bf.is_dir():
            continue
        ids_a = _load_ids(bf)
        text_a = (bf / "stdout.txt").read_text(encoding="utf-8") if (bf / "stdout.txt").exists() else ""
        md5_a = hashlib.md5(text_a.encode()).hexdigest()
        first_a = ids_a[0] if ids_a else None
        per_cfg = {}
        for cfg in CFG_ENVS:
            cd = slug_dir / cfg
            if not cd.is_dir():
                continue
            ids_b = _load_ids(cd)
            text_b = (cd / "stdout.txt").read_text(encoding="utf-8") if (cd / "stdout.txt").exists() else ""
            n = min(len(ids_a), len(ids_b))
            agree = sum(1 for x, y in zip(ids_a[:n], ids_b[:n]) if x == y) / n if n else 0.0
            first_div = next((i for i in range(n) if ids_a[i] != ids_b[i]), None)
            per_cfg[cfg] = {
                "tokens_cfg": len(ids_b),
                "top1_agree": agree,
                "first_token_bf16": first_a,
                "first_token_cfg": ids_b[0] if ids_b else None,
                # 病灶指标：K8V4 翻车 = 首 token 翻转（"1543 应 11751"）
                "first_token_match": (first_a is None or not ids_b or ids_b[0] == first_a),
                "first_divergence": first_div,
                "md5_same": hashlib.md5(text_b.encode()).hexdigest() == md5_a,
                "mean_kl": _kl_rows(_load_logprobs(bf) or [], _load_logprobs(cd) or [])
                           or "N/A（待接 logits 探针，GPU 窗口实测）",
            }
        results[slug_dir.name] = {"tokens_bf16": len(ids_a), "md5_bf16": md5_a, "cfg": per_cfg}
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # 逐配置门槛判决（参考 K8V4 翻车口径：首 token 翻转/乱码即判死）
    print("\n[verdict] 门槛（int8+res: KL<0.005 且首 token 一致；e4m3+res: KL<0.02；"
          " e4m3-nores 消融档不设门槛）", file=sys.stderr)
    fail = []
    for slug, v in results.items():
        for cfg, cv in v["cfg"].items():
            gate = GATES.get(cfg, {})
            bad = (gate.get("first_token") and not cv["first_token_match"]) or (
                gate.get("kl") is not None and isinstance(cv["mean_kl"], float)
                and cv["mean_kl"] >= gate["kl"])
            tag = "✗FAIL" if bad else "✓"
            if bad:
                fail.append(f"{slug}/{cfg}")
            print(f"  {tag} {slug}/{cfg}: {_verdict(cfg, cv)} "
                  f"top1={cv['top1_agree']:.3f} first_div={cv['first_divergence']}", file=sys.stderr)
    # 残差窗贡献提示：e4m3 两档（有/无残差）的分叉位置对比
    for slug, v in results.items():
        r1, r0 = v["cfg"].get("e4m3+res", {}), v["cfg"].get("e4m3-nores", {})
        if r1 and r0:
            print(f"  [残差窗贡献] {slug}: e4m3+res first_div={r1['first_divergence']} "
                  f"vs e4m3-nores first_div={r0['first_divergence']}", file=sys.stderr)
    if fail:
        print(f"[verdict] 未过门槛：{fail}", file=sys.stderr)
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="跑 bf16 + 三组 FP8 消融的贪心解码并抓取输出")
    pr.add_argument("--bench", default="./bin/lingxi-bench")
    pr.add_argument("--artifact", required=True)
    pr.add_argument("--tokens", type=int, default=128)
    pr.add_argument("--dir", default="run/kvfp8_quality")
    pr.add_argument("--prompt-file", action="append", help="额外 prompt 列表（# 注释）")
    pc = sub.add_parser("compare", help="离线算 mean KL / top-1 一致率 / 首 token / md5")
    pc.add_argument("--dir", default="run/kvfp8_quality")
    pc.add_argument("--early-ok", type=int, default=8,
                    help="前多少个 token 内分叉算 early-divergence 红线（默认 8）")
    args = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
