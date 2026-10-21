#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐层 golden：用 HF 的模块 + **逐层从产物取权重**（常驻一层，CPU 前向）。

流程
  1. 用 meta device 建出 `Qwen4ExpForCausalLM`（不占内存）
  2. 非层模块（embed_tokens / rotary_emb / hyper_connection_mixer / lm_head）从产物填充
  3. 主干循环逐层：meta 建层 → `to_empty(cpu)` → 从产物 load_state_dict → forward → 记录 → 释放
     （layer 1 的 PLE ngram 表用 mmap 的 102GB 外部文件，不进内存）
  4. 逐层输入/输出 + 最终 mixer 输出落 `.npz`

用法: run_layer_golden.py <out.npz> [prompt_text]
需要 PYTHONPATH 指向 ninfer-dev。
"""

from __future__ import annotations

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


@contextmanager
def _no_embedding_init():
    """构造期跳过 `nn.Embedding` 的随机初始化。

    PLE 的 ngram 表是 102GB；`reset_parameters` 会真的写满它。跳过初始化后
    `torch.empty` 的页不会被触碰，随后我们直接用 mmap 张量替换它。
    """
    original = nn.Embedding.reset_parameters
    nn.Embedding.reset_parameters = lambda self: None
    try:
        yield
    finally:
        nn.Embedding.reset_parameters = original


from tools.convert.qwen4_exp import inventory as I  # noqa: E402
from tools.convert.qwen4_exp import recipe as R  # noqa: E402

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: E402
    Qwen4ExpForCausalLM,
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextRotaryEmbedding,
)

# 引擎按 **bf16** 前向，故 golden 也必须在 bf16 下跑，逐层 rel 才有可比性。
# （产物权重是 q4→bf16；bf16 golden == 引擎的实际精度。）
DTYPE = torch.bfloat16 if os.environ.get("LINGXI_GOLDEN_BF16", "1") == "1" else torch.float32

ARTIFACT = os.environ.get("LINGXI_ARTIFACT", "/data/models/lingxi-assets/qwen4_exp.q4dense.mtp2.ninfer")
HF_CONFIG = os.environ.get("LINGXI_HF_CONFIG", "/models/nvfp4/Qwen3.8-Flash-Next/config.json")
LAYER_PREFIX = "model.language_model.layers."


class _LazyLayerList:
    """`self.layers[:N]` 的懒迭代：逐个 materialize（用完即弃）。"""

    def __init__(self, n: int, build):
        self.n = n
        self.build = build

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.iter_slice(idx)
        raise IndexError("golden runner 只支持整体遍历")

    def iter_slice(self, s: slice):
        start, stop, step = s.indices(self.n)
        for i in range(start, stop, step):
            yield self.build(i)


class _Recorder(torch.nn.Module):
    def __init__(self, inner, store: dict, idx: int):
        super().__init__()
        self.inner = inner
        self.store = store
        self.idx = idx

    def forward(self, hidden_states, **kwargs):
        out = self.inner(hidden_states, **kwargs)
        self.store[self.idx] = (
            hidden_states.detach().to(torch.float32).clone(),
            out.detach().to(torch.float32).clone(),
        )
        return out


def build_text_config() -> Qwen4ExpTextConfig:
    tc = json.load(open(HF_CONFIG))["text_config"]
    tc.pop("architectures", None)
    tc.pop("dtype", None)
    cfg = Qwen4ExpTextConfig(**tc)
    # 与引擎一致：走 float 加性掩码的 eager 路径（indexer 掩码含 f32::MIN）
    cfg._attn_implementation = "eager"
    return cfg


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/golden_layers.npz"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    torch.set_grad_enabled(False)

    cfg = build_text_config()
    print(f"[golden] layers={cfg.num_hidden_layers} ple_layer_ids={cfg.ple_layer_ids}")

    df = Defuser(ARTIFACT)
    recipes = list(R.RECIPE_SPECS)

    # ---- 每个 recipe 属于哪个层（按 canonical object 名判定；-1 = 非层） ----
    def layer_of(object_name: str) -> int:
        marker = "text/layers/"
        if object_name.startswith(marker):
            return int(object_name[len(marker):].split("/", 1)[0])
        return -1

    by_layer: dict[int, list] = {}
    for r in recipes:
        by_layer.setdefault(layer_of(r.object_name), []).append(r)
    print(f"[golden] recipes: 非层 {len(by_layer.get(-1, []))}，层 0..{cfg.num_hidden_layers - 1}")

    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(cfg)
    model.eval()

    # ---- 非层模块 ----
    df.run(by_layer.get(-1, []))
    skeleton = {k: v for k, v in df.state_dict().items()}
    for name, mod in (
        ("model.language_model.embed_tokens", model.model.embed_tokens),
        ("model.language_model.hyper_connection_mixer", model.model.hyper_connection_mixer),
        ("lm_head", model.lm_head),
    ):
        prefix = name + "."
        sd = {k[len(prefix):]: v for k, v in skeleton.items() if k.startswith(prefix)}
        mod.to_empty(device="cpu")
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        mod.to(DTYPE)
        if missing or unexpected:
            print(f"  [{name}] missing={missing[:3]} unexpected={unexpected[:3]}")
    # rotary_emb 无 checkpoint 权重：直接重建（避免 meta 上的 inv_freq）
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(cfg)

    # ---- 逐层 ----
    store: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    ple_table = None
    # PLE 权重在产物里是**全局** `text/ple/*`（不按 `text/layers/N/`），
    # 所以按 `text/layers/N/` 分组的 recipe 里没有它——必须显式并进来，
    # 否则 PLE 层的 key_proj/value_proj/norm_* 会是随机初始化！
    ple_recipes = [r for r in recipes if r.object_name.startswith("text/ple/")]

    def build_layer(i: int):
        nonlocal ple_table
        # 在 CPU 上正常构造（`__init__` 里算的 buffer 才是对的）；
        # 只 patch 掉 Embedding 随机初始化，避免写满 102GB 的 PLE 表。
        with _no_embedding_init():
            layer = Qwen4ExpTextDecoderLayer(cfg, i)
        print(f"  [layer {i}] 构造/取权重…", flush=True)
        has_ple = getattr(layer, "ple", None) is not None
        sub = Defuser(ARTIFACT)
        sub.run(by_layer.get(i, []) + (ple_recipes if has_ple else []))
        sd_all = sub.state_dict()
        prefix = f"{LAYER_PREFIX}{i}."
        sd = {k[len(prefix):]: v for k, v in sd_all.items() if k.startswith(prefix)}
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        # PLE 的 3 个 buffer（素数门表/乘子）在 `__init__` 里算出，本就不在 checkpoint 里
        missing = [m for m in missing if not m.startswith("ple.ple_embedding.")]
        if missing or unexpected:
            print(f"  [layer {i}] missing={missing} unexpected={unexpected}")
        if has_ple:
            if ple_table is None:
                ple_table = load_ple_table()
                print(f"  [layer {i}] PLE 表 mmap: {tuple(ple_table.shape)} bf16")
            emb = layer.ple.ple_embedding.ngram_embedding
            emb.weight = torch.nn.Parameter(ple_table, requires_grad=False)
        # 先在 PLE 表就位后再转 dtype：mmap 表已是 bf16，`.to(bf16)` 对它是 no-op（不复制 102GB）
        layer = layer.to(DTYPE)
        return layer

    object.__setattr__(
        model.model,
        "layers",
        _LazyLayerList(cfg.num_hidden_layers, lambda i: _Recorder(build_layer(i), store, i)),
    )

    ids = None
    tok_path = os.path.join(os.path.dirname(HF_CONFIG), "tokenizer.json")
    if os.path.exists(tok_path):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(os.path.dirname(HF_CONFIG))
        ids = tok(prompt, return_tensors="pt").input_ids
    else:
        ids = torch.arange(1, 17).unsqueeze(0)  # 退路：任意 16 个 token
    print(f"[golden] input_ids shape={tuple(ids.shape)}  prompt={prompt!r}")

    # 记录末尾 hyper_connection_mixer 的输出（主干最后一站）
    mixer_store: dict[int, tuple] = {}
    model.model.hyper_connection_mixer = _Recorder(model.model.hyper_connection_mixer, mixer_store, 0)

    out = model(input_ids=ids, use_cache=True)
    print(f"[golden] forward 完成，记录 {len(store)} 层", flush=True)

    arrays = {"input_ids": ids.numpy()}
    for i in sorted(store):
        hin, hout = store[i]
        arrays[f"layer{i}_in"] = hin.numpy()
        arrays[f"layer{i}_out"] = hout.numpy()
    if 0 in mixer_store:
        arrays["mixer_out"] = mixer_store[0][1].numpy()

    # 供 Rust 侧对拍：rope 的 cos/sin 与因果掩码（与层无关，各 dump 一次）
    from transformers.models.qwen4_exp import modeling_qwen4_exp as _M

    x_dummy = torch.zeros(1, ids.shape[1], cfg.hidden_size, dtype=DTYPE)
    pos4 = torch.arange(ids.shape[1]).view(1, 1, -1).expand(4, 1, -1)
    pos_emb = model.model.rotary_emb(x_dummy, pos4[1:])
    causal = _M.create_causal_mask(
        config=cfg, inputs_embeds=x_dummy, attention_mask=None,
        past_key_values=None, position_ids=pos4[0], allow_is_causal_skip=False,
    )
    arrays["cos"] = pos_emb[0][0].to(torch.float32).numpy()
    arrays["sin"] = pos_emb[1][0].to(torch.float32).numpy()
    arrays["causal"] = causal[0, 0].to(torch.float32).numpy()

    # 先落 npz / f32 目录（万一后面出错，40 分钟的前向不至于白跑）
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"[golden] 写入 {out_path}（{len(arrays)} 个数组）", flush=True)

    # `<name>.f32` 原始二进制，供 lingxi-bench 直接读（避免 Rust 侧依赖 npz）
    f32_dir = os.environ.get(
        "LINGXI_GOLDEN_F32_DIR",
        os.path.join(os.path.dirname(os.path.abspath(out_path)), "golden_f32"),
    )
    os.makedirs(f32_dir, exist_ok=True)

    meta = {
        "hidden": cfg.hidden_size,
        "hc": cfg.hc_count,
        "lowrank": cfg.hc_lowrank,
        "eps": cfg.rms_norm_eps,
        "num_experts": cfg.num_experts,
        "top_k": cfg.num_experts_per_tok,
        "inter": cfg.shared_expert_intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "rot_dim": int(arrays["cos"].shape[-1]),
        "indexer_n_heads": cfg.indexer_n_heads,
        "indexer_kv_heads": cfg.indexer_kv_heads,
        "indexer_head_dim": cfg.indexer_head_dim,
        "indexer_budget": cfg.indexer_budget,
        "indexer_compress_ratio": cfg.indexer_compress_ratio,
        "linear_num_key_heads": cfg.linear_num_key_heads,
        "linear_num_value_heads": cfg.linear_num_value_heads,
        "linear_key_head_dim": cfg.linear_key_head_dim,
        "linear_value_head_dim": cfg.linear_value_head_dim,
        "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
        "output_gate_type": getattr(cfg, "output_gate_type", None) or cfg.hidden_act,
        "vocab_size": cfg.vocab_size,
        # PLE（layer 1）
        "ple_layer_ids": list(cfg.ple_layer_ids),
        "ple_embed_dim": cfg.ple_embed_dim,
        "ple_conv_kernel_size": cfg.ple_conv_kernel_size,
        "ngram_size": cfg.ngram_size,
        "heads_per_ngram": cfg.heads_per_ngram,
        "ngram_vocab_size_base": cfg.ngram_vocab_size_base,
        "make_ngram_vocab_size_divisible_by": cfg.make_ngram_vocab_size_divisible_by,
        "seed": cfg.seed,
        "eos_token_id": cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id,
        "rows": int(ids.shape[1]),
    }
    json.dump(meta, open(os.path.join(f32_dir, "meta.json"), "w"), indent=1)

    for name, arr in arrays.items():
        if name == "input_ids":
            np.asarray(arr, dtype="<i4").tofile(os.path.join(f32_dir, "input_ids.i32"))
        else:
            np.asarray(arr, dtype="<f4").tofile(os.path.join(f32_dir, name + ".f32"))
    print(f"[golden] 落盘 .f32 → {f32_dir}", flush=True)

    for attr in ("last_hidden_state", "logits"):
        value = getattr(out, attr, None)
        if value is not None:
            np.savez(out_path + f".{attr}.npz", value=value.to(torch.float32).numpy())
            print(f"[golden] 另存 {attr}: {tuple(value.shape)}", flush=True)
            break


if __name__ == "__main__":
    main()
