#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连续贪心 token golden：HF（产物真实权重、bf16）逐步贪心，产 token id 序列。

用途：对拍 lingxi-driver 的 prefill + 连续 decode_token（T3 门）。
做法（故意不用 HF KV cache）：
  - 第 0 步全序列 prefill；之后每步把已生成 token 拼回，**全序列重新前向**，
    取末行 argmax。causal 掩码下末行结果与增量 decode 数学等价，
    且不依赖 qwen4_exp 定制层（GDN 递归态/QSA 索引器）的 cache 支持，零适配风险。
  - 权重逐层流式加载、用完即释放（同 run_layer_golden.py，79GB 内存装不下 225GB 权重）。
  - 代价：每步重读全量权重（机械盘约 20+ 分钟/步），故 n_new 保持 2~4。

用法: run_gen_golden.py <out.json> [prompt] [n_new]
环境: LINGXI_ARTIFACT / LINGXI_HF_CONFIG 同 run_layer_golden.py
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", "~/ninfer")))

from contextlib import contextmanager  # noqa: E402

import torch.nn as nn  # noqa: E402

from defuse import Defuser, load_ple_table  # noqa: E402
from tools.convert.qwen4_exp import recipe as R  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: E402
    Qwen4ExpForCausalLM,
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextRotaryEmbedding,
)

DTYPE = torch.bfloat16
ARTIFACT = os.environ.get("LINGXI_ARTIFACT", "/models/qwen4_exp-artifact/qwen4_exp.ninfer")
HF_CONFIG = os.environ.get("LINGXI_HF_CONFIG", "/models/nvfp4/Qwen3.8-Flash-Next/config.json")
LAYER_PREFIX = "model.language_model.layers."


@contextmanager
def _no_embedding_init():
    original = nn.Embedding.reset_parameters
    nn.Embedding.reset_parameters = lambda self: None
    try:
        yield
    finally:
        nn.Embedding.reset_parameters = original


class _LazyLayerList:
    def __init__(self, n: int, build):
        self.n, self.build = n, build

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.n)
            for i in range(start, stop, step):
                yield self.build(i)
            return
        raise IndexError("只支持整体遍历")


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/gen_golden.json"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    torch.set_grad_enabled(False)
    tc = json.load(open(HF_CONFIG))["text_config"]
    tc.pop("architectures", None)
    tc.pop("dtype", None)
    cfg = Qwen4ExpTextConfig(**tc)
    cfg._attn_implementation = "eager"

    df = Defuser(ARTIFACT)
    recipes = list(R.RECIPE_SPECS)

    def layer_of(name: str) -> int:
        m = "text/layers/"
        return int(name[len(m):].split("/", 1)[0]) if name.startswith(m) else -1

    by_layer: dict[int, list] = {}
    for r in recipes:
        by_layer.setdefault(layer_of(r.object_name), []).append(r)
    ple_recipes = [r for r in recipes if r.object_name.startswith("text/ple/")]

    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(cfg)
    model.eval()

    df.run(by_layer.get(-1, []))
    skeleton = dict(df.state_dict())
    for name, mod in (
        ("model.language_model.embed_tokens", model.model.embed_tokens),
        ("model.language_model.hyper_connection_mixer", model.model.hyper_connection_mixer),
        ("lm_head", model.lm_head),
    ):
        prefix = name + "."
        sd = {k[len(prefix):]: v for k, v in skeleton.items() if k.startswith(prefix)}
        mod.to_empty(device="cpu")
        mod.load_state_dict(sd, strict=False)
        mod.to(DTYPE)
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(cfg)

    ple_table = None

    def build_layer(i: int):
        nonlocal ple_table
        with _no_embedding_init():
            layer = Qwen4ExpTextDecoderLayer(cfg, i)
        has_ple = getattr(layer, "ple", None) is not None
        sub = Defuser(ARTIFACT)
        sub.run(by_layer.get(i, []) + (ple_recipes if has_ple else []))
        prefix = f"{LAYER_PREFIX}{i}."
        sd = {k[len(prefix):]: v for k, v in sub.state_dict().items() if k.startswith(prefix)}
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("ple.ple_embedding.")]
        if missing or unexpected:
            print(f"  [layer {i}] missing={missing} unexpected={unexpected}", flush=True)
        if has_ple:
            if ple_table is None:
                ple_table = load_ple_table()
            layer.ple.ple_embedding.ngram_emb.weight = torch.nn.Parameter(ple_table, requires_grad=False)
        return layer.to(DTYPE)

    # tokenizer 用 HF 目录的（与产物内 frontend/tokenizer.json 同源；
    # 已验证两者对基准 prompt 产出相同 id）
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.dirname(HF_CONFIG))
    seq = tok(prompt, return_tensors="pt").input_ids
    prompt_ids = seq[0].tolist()
    print(f"[gen-golden] prompt={prompt!r} ids={prompt_ids}  n_new={n_new}", flush=True)

    gen: list[int] = []
    # n_new+1 次前向：首次=prompt 出第 1 个新 token，之后每次拼回再出一个
    for step in range(n_new + 1):
        object.__setattr__(
            model.model, "layers",
            _LazyLayerList(cfg.num_hidden_layers, lambda i: build_layer(i)),
        )
        import time
        ts = time.time()
        out = model(input_ids=seq, use_cache=False)
        nxt = int(out.logits[0, -1].argmax(-1).item())
        phase = "prefill" if step == 0 else f"decode{step}"
        print(f"[gen-golden] {phase:8s} 用时 {time.time()-ts:.0f}s  token={nxt}", flush=True)
        gen.append(nxt)
        seq = torch.cat([seq, torch.tensor([[nxt]], dtype=seq.dtype)], dim=1)
        del out
        gc.collect()

    result = {
        "prompt": prompt,
        "artifact": ARTIFACT,
        "dtype": "bfloat16",
        "prompt_ids": prompt_ids,
        # gen[0] = prefill 末行 argmax；gen[1..] = 后续 decode token
        "generated_ids": gen,
        "generated_text": tok.decode(gen),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[gen-golden] 写入 {out_path}: {gen}")
    print(f"[gen-golden] 文本: {tok.decode(gen)!r}")


if __name__ == "__main__":
    main()
