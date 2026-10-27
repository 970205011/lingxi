#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组件参考：用 HF 的模块 + 产物里的真实权重算一个组件的前向，供 Rust 侧对拍。

用法: component_oracle.py <kind> <artifact> <canonical-prefix> <out_dir> [rows] [seed]
  kind = gr | moe
  canonical-prefix 例：text/layers/0/attn_gr/  |  text/layers/0/moe/  |  text/hyper_connection_mixer/

输出（原始 f32 小端）: out_dir/{x,ref}.f32 + out_dir/meta.json

精度：**bf16**（与引擎实际精度一致；权重本来就是 bf16）。
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", "~/ninfer")))

from defuse import Defuser  # noqa: E402

from tools.convert.qwen4_exp import recipe as R  # noqa: E402

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import ( # noqa: E402
    Qwen4ExpTextAttention,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextSparseMoeBlock,
)

HF_CONFIG = os.environ.get("LINGXI_HF_CONFIG", "/models/nvfp4/Qwen3.8-Flash-Next/config.json")


def hf_source_prefix(kind: str, canonical: str) -> str:
    """canonical 前缀 → HF 源前缀。"""
    marker = "text/layers/"
    if canonical.startswith(marker):
        rest = canonical[len(marker):]
        layer = rest.split("/")[0]
        base = f"model.language_model.layers.{layer}."
        if kind == "gr":
            kindname = rest.split("/")[1]
            module = "attn_hyper_connection" if kindname == "attn_gr" else "mlp_hyper_connection"
            return base + module + "."
        if kind == "moe":
            return base + "mlp."
        if kind == "attention":
            return base + "self_attn."
        if kind == "gdn":
            return base + "linear_attn."
        raise SystemExit(f"未知 kind: {kind}")
    if kind == "gr" and canonical.startswith("text/hyper_connection_mixer/"):
        return "model.language_model.hyper_connection_mixer."
    raise SystemExit(f"无法解析 canonical 前缀: {canonical}（kind={kind}）")


def build_module(kind: str, cfg, use_combine: bool, layer_idx: int):
    if kind == "gr":
        return Qwen4ExpTextGatedResidual(cfg, use_combine=use_combine)
    if kind == "moe":
        return Qwen4ExpTextSparseMoeBlock(cfg)
    if kind == "attention":
        return Qwen4ExpTextAttention(cfg, layer_idx)
    if kind == "gdn":
        return M.Qwen4ExpTextGatedDeltaNet(cfg, layer_idx)
    raise SystemExit(f"未知 kind: {kind}")


def layer_index_of(canonical: str) -> int:
    marker = "text/layers/"
    if canonical.startswith(marker):
        return int(canonical[len(marker):].split("/")[0])
    return -1


def main() -> None:
    kind, artifact, canonical, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 99

    tc = json.load(open(HF_CONFIG))["text_config"]
    tc.pop("architectures", None)
    tc.pop("dtype", None)
    cfg = Qwen4ExpTextConfig(**tc)

    hf_prefix = hf_source_prefix(kind, canonical)
    df = Defuser(artifact)
    recipes = [r for r in R.RECIPE_SPECS if r.object_name.startswith(canonical)]
    if not recipes:
        raise SystemExit(f"没有匹配的 recipe: {canonical}")
    df.run(recipes)
    sd = {k[len(hf_prefix):]: v for k, v in df.state_dict().items() if k.startswith(hf_prefix)}

    torch.manual_seed(seed)
    use_combine = kind == "gr" and any(r.object_name == canonical + "block_inject" for r in recipes)
    layer_idx = layer_index_of(canonical)
    if kind == "attention":
        cfg._attn_implementation = "eager"  # 用 float 加性掩码路径
    with torch.device("meta"):
        mod = build_module(kind, cfg, use_combine, layer_idx)
    mod = mod.to_empty(device="cpu")
    missing, unexpected = mod.load_state_dict(sd, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected: {unexpected}")
    if missing:
        print(f"[oracle] 未加载（应为空）: {missing}")
    mod = mod.to(torch.bfloat16).eval()

    hidden = cfg.hidden_size
    extra = {}
    if kind == "attention":
        T = rows
        x3 = torch.randn(1, T, hidden).to(torch.bfloat16)
        rot = Qwen4ExpTextRotaryEmbedding(cfg)
        pos4 = torch.arange(T).view(1, 1, -1).expand(4, 1, -1)
        with torch.no_grad():
            pos_emb = rot(x3, pos4[1:])
            causal = M.create_causal_mask(
                config=cfg, inputs_embeds=x3, attention_mask=None,
                past_key_values=None, position_ids=pos4[0], allow_is_causal_skip=False,
            )
            out, _ = mod(x3, pos_emb, causal, None)
            sel = mod.indexer(x3, pos_emb, causal, None)
            combined = causal + sel
        x = x3[0]
        ref = out[0]
        extra["cos"] = pos_emb[0][0]
        extra["sin"] = pos_emb[1][0]
        extra["mask"] = combined[0, 0]
        extra["causal"] = causal[0, 0]
    else:
        in_dim = cfg.hc_count * hidden if kind == "gr" else hidden
        with torch.no_grad():
            x = torch.randn(rows, in_dim).to(torch.bfloat16)
            if kind in ("moe", "gdn"):
                out = mod(x.unsqueeze(0))          # 这两个模块要 [B,T,H]
                ref = out[0]
            else:
                out = mod(x)
                ref = out[0] if (kind == "gr" and use_combine) else out
        if kind == "gr" and use_combine:
            extra["injw"] = out[2]

    # GDN 分段对拍用：投影 + 卷积输出
    if kind == "gdn":
        with torch.no_grad():
            hs = x
            mixed = mod.in_proj_qkv(hs.unsqueeze(0)).transpose(1, 2)
            zz = mod.in_proj_z(hs.unsqueeze(0))[0]
            extra["z"] = zz
            aa_ = mod.in_proj_a(hs.unsqueeze(0))
            bb_ = mod.in_proj_b(hs.unsqueeze(0))
            extra["a"] = aa_[0]
            extra["b"] = bb_[0]
            conv = M.causal_conv1d_fn(
                mixed, mod.conv1d.weight.squeeze(1), mod.conv1d.bias, activation=mod.activation
            )
            extra["conv"] = conv[0].transpose(0, 1)   # [T, conv_dim]
            # 逐段复现模块内部（用 HF 自己的函数），供 Rust 分段对拍
            key_dim = mod.head_k_dim * mod.num_k_heads
            value_dim = mod.head_v_dim * mod.num_v_heads
            qq, kk_, vv = torch.split(conv.transpose(1, 2), [key_dim, key_dim, value_dim], dim=-1)
            qq = qq.reshape(1, rows, -1, mod.head_k_dim)
            kk_ = kk_.reshape(1, rows, -1, mod.head_k_dim)
            vv = vv.reshape(1, rows, -1, mod.head_v_dim)
            rep = mod.num_v_heads // mod.num_k_heads
            if rep > 1:
                qq = qq.repeat_interleave(rep, dim=2)
                kk_ = kk_.repeat_interleave(rep, dim=2)
            beta_t = bb_.sigmoid()
            g_t = -mod.A_log.float().exp() * torch.nn.functional.softplus(aa_.float() + mod.dt_bias)
            core, _ = M.torch_chunk_gated_delta_rule(
                qq, kk_, vv, g=g_t, beta=beta_t, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=True,
            )
            extra["core"] = core[0].reshape(rows, -1)
            extra["gated"] = mod.norm(
                core[0].reshape(-1, mod.head_v_dim).to(torch.bfloat16),
                zz.reshape(-1, mod.head_v_dim).to(torch.bfloat16),
            ).reshape(rows, -1)
            extra["beta"] = beta_t[0]
            extra["g"] = g_t[0]

    # MoE 分段对拍用：router 选出的专家、shared 分支、专家分支
    if kind == "moe":
        with torch.no_grad():
            hs = x.view(-1, hidden)
            logits, scores, sel = mod.gate(hs)
            shared = mod.shared_expert(hs)
            shared = torch.sigmoid(mod.shared_expert_gate(hs)) * shared
            expert_out = mod.experts(hs, sel, scores)
            extra["router_logits"] = logits
            extra["router_sel"] = sel.to(torch.float32)
            extra["router_w"] = scores
            extra["shared"] = shared
            extra["expert_out"] = expert_out

    os.makedirs(out_dir, exist_ok=True)
    x.float().contiguous().numpy().tofile(os.path.join(out_dir, "x.f32"))
    ref.float().contiguous().numpy().tofile(os.path.join(out_dir, "ref.f32"))
    for name, t in extra.items():
        t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(out_dir, f"{name}.f32"))
    meta = {
        "kind": kind,
        "rows": rows,
        "hidden": hidden,
        "hc": cfg.hc_count,
        "dim": cfg.hc_count * hidden,
        "lowrank": cfg.hc_lowrank,
        "num_experts": cfg.num_experts,
        "top_k": cfg.num_experts_per_tok,
        "inter": cfg.shared_expert_intermediate_size,
        "eps": cfg.rms_norm_eps,
        "use_combine": use_combine,
        "canonical": canonical,
        "hf_prefix": hf_prefix,
        "ref_shape": list(ref.shape),
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
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
        # RMSNormGated 的 gate 激活（GDN 层用），如 "sigmoid"
        "output_gate_type": getattr(cfg, "output_gate_type", None) or cfg.hidden_act,
    }
    if "cos" in extra:
        meta["rot_dim"] = int(extra["cos"].shape[-1])
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"[oracle] OK({kind}): x{tuple(x.shape)} → ref{tuple(ref.shape)} → {out_dir}")


if __name__ == "__main__":
    main()
