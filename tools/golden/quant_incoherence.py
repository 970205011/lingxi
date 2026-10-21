#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证两条 SOTA 路线在我们真实专家权重上是否有效：

  A) Hadamard 去相干（RHT，QuIP/QuIP#/QTIP 的第一步）后再量化
     —— 观察 outlier 抑制（kurtosis / 逐列 std 离散度）与误差改善
  B) 专家间共享结构（BitsMoE 的 shared spectral basis 前提）
     —— 测跨专家的均值/相关性，判断"用共享基代替个体数值"是否可行

在 测试机 内运行：
  cd /models/lingxi && \
  PYTHONPATH=~/ninfer \
  python3.12 tools/golden/quant_incoherence.py
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
    return t.reshape(N_EXPERTS_TOTAL, rows, t.shape[1]).to(torch.float32)  # [512,1280,2560]


def hadamard(n: int) -> torch.Tensor:
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def rht(X: torch.Tensor, block: int, seed: int = 0) -> torch.Tensor:
    """随机 Hadamard 变换，按 input 维分块（正交，Frobenius 范数不变）。"""
    out, inn = X.shape
    assert inn % block == 0
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (inn // block, block), generator=g).float() * 2 - 1)
    H = hadamard(block) / (block**0.5)
    Xb = X.reshape(out, inn // block, block) * s
    return (Xb @ H).reshape(out, inn)


def kurtosis(W: torch.Tensor) -> float:
    z = (W - W.mean()) / (W.std() + 1e-12)
    return float((z**4).mean())


def colstd_spread(W: torch.Tensor) -> float:
    cs = W.std(0)
    return float(cs.max() / cs.min())


def affine_group(W: torch.Tensor, bits: int, group: int = 128):
    out, inn = W.shape
    Wg = W.reshape(-1, group)
    qm = 2 ** (bits - 1) - 1
    scale = (Wg.abs().amax(1, keepdim=True) / qm).clamp_min(1e-12)
    q = torch.round(Wg / scale).clamp(-qm - 1, qm)
    return (q * scale).reshape(out, inn)


def relerr(W: torch.Tensor, Wq: torch.Tensor) -> float:
    return float((Wq - W).norm() / W.norm())


def main() -> None:
    art = Artifact.open(ART)
    gu = load_gu(art, 0)
    W = gu[0]
    print(f"== layer0 expert0 gu {tuple(W.shape)} ==")
    print(f"  kurtosis={kurtosis(W):.2f} (高斯=3)  逐列std离散度={colstd_spread(W):.2f}")

    print("\n[A] Hadamard 去相干前后的量化误差")
    for blk in (64, 128, 256):
        Wr = rht(W, blk, seed=1)
        print(f"  RHT block={blk}: kurtosis={kurtosis(Wr):.2f} 逐列std离散度={colstd_spread(Wr):.2f}")
    for bits in (4, 3, 2):
        e0 = relerr(W, affine_group(W, bits))
        print(f"  W{bits} 无RHT : 相对误差={e0:.4f}")
        for blk in (128, 256):
            Wr = rht(W, blk, seed=1)
            Wrq = affine_group(Wr, bits)
            # 正交变换保范数：直接在变换域比误差
            print(f"  W{bits} +RHT{blk}: 相对误差={relerr(Wr, Wrq):.4f}")

    print("\n[B] 专家间共享结构（BitsMoE 前提）")
    n = 8
    mean_e = gu[:n].mean(0)
    print(f"  跨{n}专家均值能量占比 = {(mean_e.norm()/gu[:n].norm()* (n**0.5)).item()**2:.4f}"
          f"  (‖mean‖/‖W‖={float(mean_e.norm()/W.norm()):.4f})")
    a = gu[0].reshape(-1)
    for j in (1, 2, 4):
        b = gu[j].reshape(-1)
        cc = torch.corrcoef(torch.stack([a, b]))[0, 1]
        print(f"  专家0 vs 专家{j} 逐元素相关 = {cc:.4f}")


if __name__ == "__main__":
    main()
