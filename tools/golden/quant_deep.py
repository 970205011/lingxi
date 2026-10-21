#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两个决定性测量（回答"还有没有别的招"）：

  A) 专家模式的奇异值谱 → 判 TD-MoE / BitsMoE 的"共享基/Tucker 分解"对我们是否可行
  B) 上下文条件熵 → 判 DeepCABAC/CABAC 式上下文熵编码能比一阶熵再省多少

在 测试机 内运行：
  cd /models/lingxi && \
  PYTHONPATH=~/ninfer \
  python3.12 tools/golden/quant_deep.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", "~/ninfer")))
from tools.artifact.container import Artifact  # noqa: E402
from tools.artifact.layouts import decode_direct  # noqa: E402

ART = os.environ.get("LINGXI_ARTIFACT", "/models/qwen4_exp-artifact/qwen4_exp.ninfer")
N_EXPERTS_TOTAL = 512


def load_gu(art: Artifact, layer: int) -> torch.Tensor:
    o = art.find(f"text/layers/{layer}/moe/routed_gate_up")
    t = decode_direct(art.payload(o), o.format, o.shape)
    rows = t.shape[0] // N_EXPERTS_TOTAL
    return t.reshape(N_EXPERTS_TOTAL, rows, t.shape[1]).to(torch.float32)


def entropy(vals: torch.Tensor, n: int) -> float:
    c = torch.bincount(vals.reshape(-1), minlength=n).double()
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def joint_entropy(parts: list[torch.Tensor], ns: list[int]) -> float:
    code = torch.zeros(parts[0].numel(), dtype=torch.int64)
    mult = 1
    for v, n in zip(parts, ns):
        code = code + v.reshape(-1).to(torch.int64) * mult
        mult *= n
    return entropy(code, mult)


def main() -> None:
    art = Artifact.open(ART)
    gu = load_gu(art, 0)
    print(f"== layer0 gu {tuple(gu.shape)} ==")

    # ---- A) 专家模式奇异值谱 ----
    print("\n[A] 专家模式奇异值谱（Tucker/BitsMoE 共享基是否可行）")
    n_exp, n_cols = 32, 200_000
    g = torch.Generator().manual_seed(0)
    cols = torch.randperm(gu.shape[2], generator=g)[:n_cols]
    A = gu[:n_exp, :, cols].reshape(n_exp, -1)  # [32, 1280*200000/...]
    G = A @ A.t()
    ev = torch.linalg.eigvalsh(G.double()).flip(0).clamp_min(0)
    ev = ev / ev.sum()
    cum = ev.cumsum(0)
    for r in (1, 2, 4, 8, 16, 32):
        print(f"  前 {r:2d} 个奇异值能量占比 = {cum[r-1]*100:6.2f}%")
    print(f"  （若专家相互独立 → 谱平坦；共享基成立 → 前几个占比极高）")

    # ---- B) 上下文条件熵（CABAC/DeepCABAC 前提）----
    print("\n[B] W4 索引的条件熵（上下文熵编码的上限收益）")
    bits = 4
    W = gu[0].reshape(-1)
    Wg = W.reshape(-1, 128)
    qm = 2 ** (bits - 1) - 1
    scale = (Wg.abs().amax(1, keepdim=True) / qm).clamp_min(1e-12)
    q = torch.round(Wg / scale).clamp(-qm - 1, qm).to(torch.int64) + (qm + 1)
    n = 2**bits
    x0 = q[:, :-2].reshape(-1)
    x1 = q[:, 1:-1].reshape(-1)
    x2 = q[:, 2:].reshape(-1)
    H0 = entropy(x0, n)
    H1 = joint_entropy([x1, x0], [n, n]) - entropy(x0, n)
    H2 = joint_entropy([x2, x1, x0], [n, n, n]) - joint_entropy([x1, x0], [n, n])
    print(f"  0阶熵 H(X)        = {H0:.3f} bit")
    print(f"  1阶 H(X|X-1)      = {H1:.3f} bit  (再省 {(1 - H1 / H0) * 100:.1f}%)")
    print(f"  2阶 H(X|X-1,X-2)  = {H2:.3f} bit  (再省 {(1 - H2 / H0) * 100:.1f}%)")


if __name__ == "__main__":
    main()
