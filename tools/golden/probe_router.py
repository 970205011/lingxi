#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时探针：抓 HF golden 路径下 layer 0/1 的 MoE 入口 x、router logits、top-k 选择。
走完整模型前向（位置/PLE kwargs 由主干自然提供），过完 layer 1 即中止。
与引擎 LINGXI_MOE_DUMP 产物对拍，定位 engine vs golden 逐层偏差根因。
用法：python3 probe_router.py <golden_f32_dir> <out_dir>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", "~/ninfer")))
sys.path.insert(0, "/models/lingxi/tools/golden")

import torch.nn as nn  # noqa: E402
from defuse import Defuser, load_ple_table  # noqa: E402
from tools.convert.qwen4_exp import recipe as R  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import ( # noqa: E402
    Qwen4ExpForCausalLM,
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextRotaryEmbedding,
)
from run_layer_golden import ( # noqa: E402
    _no_embedding_init,
    build_text_config,
    LAYER_PREFIX,
    HF_CONFIG,
)

DTYPE = torch.bfloat16
ARTIFACT = os.environ.get("LINGXI_ARTIFACT", "/models/lingxi-assets/qwen4_exp.q4dense.mtp2.ninfer")


class _StopAfterLayer1(Exception):
    pass


class _ProbeRecorder(torch.nn.Module):
    def __init__(self, idx, build, out_dir, hidden_size):
        super().__init__()
        self.idx = idx
        self.inner = build(idx)
        self.out_dir = out_dir
        H = hidden_size

        def pre(_m, args):
            x = args[0].detach().reshape(-1, H).to(torch.float32).cpu().numpy()
            np.asarray(x, dtype="<f4").tofile(os.path.join(out_dir, f"l{idx}_hf_x.f32"))

        def post(_m, _a, out):
            logits, _scores, sel = out
            np.asarray(logits.detach().to(torch.float32).cpu().numpy(), dtype="<f4").tofile(
                os.path.join(out_dir, f"l{idx}_hf_logits.f32"))
            np.asarray(sel.detach().to(torch.int32).cpu().numpy(), dtype="<i4").tofile(
                os.path.join(out_dir, f"l{idx}_hf_sel.i32"))
            print(f"  [probe] layer {idx} router 已落盘", flush=True)

        self._h1 = self.inner.mlp.gate.register_forward_pre_hook(pre)
        self._h2 = self.inner.mlp.gate.register_forward_hook(post)

        def save(name, arr):
            np.asarray(arr.detach().to(torch.float32).cpu().numpy(), dtype="<f4").tofile(
                os.path.join(out_dir, f"l{idx}_hf_{name}.f32"))

        def gr_post(_m, _a, out):
            mixed, hyper_input, injw = out
            save("mixed_attn", mixed)
            save("attn_hyper_out", hyper_input)
            save("attn_injw", injw)
            print(f"  [probe] layer {idx} attn GR 已落盘", flush=True)

        def attn_post(_m, _a, out):
            save("attn_out", out[0] if isinstance(out, tuple) else out)
            print(f"  [probe] layer {idx} attn 输出已落盘", flush=True)

        def ple_post(_m, _a, out):
            save("ple_out", out)
            print(f"  [probe] layer {idx} PLE 输出已落盘", flush=True)

        self._h3 = self.inner.attn_hyper_connection.register_forward_hook(gr_post)
        self._h4 = self.inner.linear_attn.register_forward_hook(attn_post)
        if self.inner.ple is not None:
            self._h5 = self.inner.ple.register_forward_hook(ple_post)

        # ── GDN 内部阶段（layer 0 重点排查）──
        la = self.inner.linear_attn
        def lin_hook(tag):
            def _h(_m, _a, out):
                save(f"gdn_{tag}", out)
            return _h
        self._h6 = la.in_proj_qkv.register_forward_hook(lin_hook("qkv_proj"))
        self._h7 = la.in_proj_z.register_forward_hook(lin_hook("z_proj"))
        self._h8 = la.in_proj_a.register_forward_hook(lin_hook("a_proj"))
        self._h9 = la.in_proj_b.register_forward_hook(lin_hook("b_proj"))
        def conv_raw(_m, _a, out):
            save("gdn_conv_raw", out)  # [B, C, T+K-1] 未切片/未激活
        self._h10 = la.conv1d.register_forward_hook(conv_raw)
        save("gdn_conv_w", la.conv1d.weight.squeeze(1))  # [C,K]
        save("gdn_a_log", la.A_log)
        save("gdn_dt_bias", la.dt_bias)
        save("gdn_norm_w", la.norm.weight)
        if idx == 0:
            save("gdn_w_qkv", la.in_proj_qkv.weight)
            save("gdn_w_z", la.in_proj_z.weight)
            save("gdn_w_a", la.in_proj_a.weight)
            save("gdn_w_b", la.in_proj_b.weight)

        import transformers.models.qwen4_exp.modeling_qwen4_exp as _M
        orig_chunk = _M.torch_chunk_gated_delta_rule
        def wrapped_chunk(query, key, value, g, beta, **kw):
            o, st = orig_chunk(query, key, value, g, beta, **kw)
            if idx == 0:
                save("gdn_chunk_q", query)
                save("gdn_chunk_k", key)
                save("gdn_chunk_v", value)
                save("gdn_chunk_g", g)
                save("gdn_chunk_beta", beta)
                save("gdn_chunk_out", o)
                print("  [probe] layer 0 chunk 核入口/出口已落盘", flush=True)
            return o, st
        _M.torch_chunk_gated_delta_rule = wrapped_chunk
        self._unpatch = lambda: setattr(_M, "torch_chunk_gated_delta_rule", orig_chunk)

    def forward(self, hidden_states, **kwargs):
        out = self.inner(hidden_states, **kwargs)
        if self.idx >= 1:
            raise _StopAfterLayer1()
        return out


def main():
    golden_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    cfg = build_text_config()
    H = cfg.hidden_size

    df = Defuser(ARTIFACT)
    recipes = list(R.RECIPE_SPECS)

    def layer_of(name: str) -> int:
        m = "text/layers/"
        return int(name[len(m):].split("/", 1)[0]) if name.startswith(m) else -1

    by_layer: dict[int, list] = {}
    for r in recipes:
        by_layer.setdefault(layer_of(r.object_name), []).append(r)
    ple_recipes = [r for r in recipes if r.object_name.startswith("text/ple/")]
    ple_table = None

    def build_layer(i: int):
        nonlocal ple_table
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
        missing = [m for m in missing if not m.startswith("ple.ple_embedding.")]
        if missing or unexpected:
            print(f"  [layer {i}] missing={missing} unexpected={unexpected}")
        if has_ple:
            if ple_table is None:
                ple_table = load_ple_table()
            layer.ple.ple_embedding.ngram_embedding.weight = torch.nn.Parameter(
                ple_table, requires_grad=False)
        return layer.to(DTYPE)

    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(cfg)
    model.eval()

    df.run(by_layer.get(-1, []))
    skeleton = dict(df.state_dict().items())
    for name, mod in (
        ("model.language_model.embed_tokens", model.model.embed_tokens),
        ("model.language_model.hyper_connection_mixer", model.model.hyper_connection_mixer),
        ("lm_head", model.lm_head),
    ):
        pfx = name + "."
        sd = {k[len(pfx):]: v for k, v in skeleton.items() if k.startswith(pfx)}
        mod.to_empty(device="cpu")
        mod.load_state_dict(sd, strict=False)
        mod.to(DTYPE)
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(cfg)

    class _Lazy:
        def __init__(self, n):
            self.n = n

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                s0, s1, st = idx.indices(self.n)
                for i in range(s0, s1, st):
                    yield _ProbeRecorder(i, build_layer, out_dir, H)
            else:
                raise IndexError

    object.__setattr__(model.model, "layers", _Lazy(cfg.num_hidden_layers))

    ids = np.fromfile(os.path.join(golden_dir, "input_ids.i32"), dtype=np.int32).astype(np.int64)
    ids = torch.from_numpy(ids).unsqueeze(0)
    print(f"[probe] input_ids={tuple(ids.shape)}", flush=True)
    try:
        model(input_ids=ids, use_cache=True)
    except _StopAfterLayer1:
        print("[probe] 已抓完 layer 0/1，正常中止", flush=True)


if __name__ == "__main__":
    main()
