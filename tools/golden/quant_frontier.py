#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""压缩率↔误差前沿：在**真实专家权重**上量各量化/压缩方案。

目的：回答"能不能压到 2bit 还保住精度"。全部在单专家矩阵上量，
指标 = 权重 Frobenius 相对误差 + 随机向量 matvec 相对误差。

在 测试机 内运行：
  cd /models/lingxi && \
  PYTHONPATH=~/ninfer \
  python3.12 tools/golden/quant_frontier.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", "~/ninfer")))
from tools.artifact.container import Artifact  # noqa: E402
from tools.artifact.layouts import decode_direct  # noqa: E402

ART = os.environ.get("LINGXI_ARTIFACT", "/models/qwen4_exp-artifact/qwen4_exp.ninfer")
LAYER = int(os.environ.get("QFY_LAYER", "0"))
NEXP = int(os.environ.get("QFY_NEXP", "1"))
N_EXPERTS_TOTAL = 512
ROWS_PER_EXPERT = 1280  # gu: 2*640


def load_expert_gu(art: Artifact, layer: int, e: int) -> torch.Tensor:
    o = art.find(f"text/layers/{layer}/moe/routed_gate_up")
    t = decode_direct(art.payload(o), o.format, o.shape)  # [512*1280, 2560]
    rows = t.shape[0] // N_EXPERTS_TOTAL
    return t[e * rows : (e + 1) * rows].to(torch.float32)


# ---------------- 方案 ----------------
def affine_group(W: torch.Tensor, bits: int, group: int = 128):
    out, inn = W.shape
    assert inn % group == 0
    Wg = W.reshape(out, inn // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = (Wg.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.round(Wg / scale).clamp(-qmax - 1, qmax)
    Wq = (q * scale).reshape(out, inn)
    return Wq, bits + 16.0 / group  # scale 存 fp16


def kmeans(X: torch.Tensor, k: int, iters: int = 12, seed: int = 0, sample: int = 32768):
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    Xs = X[torch.randperm(N, generator=g)[:sample]] if N > sample else X
    C = Xs[torch.randperm(Xs.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        d = (Xs * Xs).sum(1, keepdim=True) - 2 * Xs @ C.t() + (C * C).sum(1)[None, :]
        a = d.argmin(1)
        oh = torch.zeros(Xs.shape[0], k)
        oh.scatter_(1, a[:, None], 1.0)
        C = (oh.t() @ Xs) / oh.sum(0, keepdim=True).t().clamp_min(1)
    return C


def _assign(Xb: torch.Tensor, C: torch.Tensor):
    d = (Xb * Xb).sum(1, keepdim=True) - 2 * Xb @ C.t() + (C * C).sum(1)[None, :]
    return d.argmin(1)


def vq(W: torch.Tensor, g: int, b: int, stages: int = 1, iters: int = 12):
    """码本向量量化；stages=2 即加性（AQLM 式，每块两个码字相加）。bpw = stages*b/g。"""
    out, inn = W.shape
    assert inn % g == 0
    nb = inn // g
    Xb = W.reshape(out * nb, g).clone()
    k = 2**b
    acc = torch.zeros_like(Xb)
    for s in range(stages):
        R = Xb - acc
        C = kmeans(R, k, iters=iters, seed=s)
        a = _assign(R, C)
        acc = acc + C[a]
    Wq = acc.reshape(out, inn)
    return Wq, stages * b / g


def vq_seg(W: torch.Tensor, g: int, b: int, seg: int, stages: int = 1, iters: int = 12):
    """**分块码本** VQ：input 维每 seg 个权重共享一个自己的码本（AQLM 的实际做法）。
    bpw = stages*b/g + 码本开销。"""
    out, inn = W.shape
    assert inn % seg == 0 and seg % g == 0
    nseg = inn // seg
    nb = seg // g
    Wq = torch.empty_like(W)
    for s in range(nseg):
        blk = W[:, s * seg : (s + 1) * seg].reshape(out * nb, g)
        acc = torch.zeros_like(blk)
        for st in range(stages):
            R = blk - acc
            C = kmeans(R, 2**b, iters=iters, seed=st * 131 + s)
            acc = acc + C[_assign(R, C)]
        Wq[:, s * seg : (s + 1) * seg] = acc.reshape(out, seg)
    # 码本开销：nseg 个码本，每个 2^b×g 个 fp16
    overhead = nseg * (2**b) * g * 16.0 / (out * inn)
    return Wq, stages * b / g + overhead


def report(name: str, W: torch.Tensor, Wq: torch.Tensor, bpw: float, x: torch.Tensor):
    werr = (Wq - W).norm() / W.norm()
    y0 = W @ x
    y1 = Wq @ x
    merr = (y1 - y0).norm() / y0.norm()
    print(f"  {name:24s} bpw={bpw:5.2f}  权重相对误差={werr:.4f}  matvec相对误差={merr:.4f}")


def main() -> None:
    art = Artifact.open(ART)
    torch.manual_seed(0)
    for e in range(NEXP):
        W = load_expert_gu(art, LAYER, e)
        x = torch.randn(W.shape[1])
        print(f"\n== layer {LAYER} expert {e} gu {tuple(W.shape)} "
              f"(std={W.std():.5f}, absmax={W.abs().max():.5f}) ==")
        for nm, fn in [
            ("W8 仿射 g128", lambda w: affine_group(w, 8)),
            ("W4 仿射 g128", lambda w: affine_group(w, 4)),
            ("VQ g16 b8 (0.5bpw)", lambda w: vq(w, 16, 8)),
            ("VQ g8  b8 (1.0bpw)", lambda w: vq(w, 8, 8)),
            ("VQ g4  b8 (2.0bpw)", lambda w: vq(w, 4, 8)),
            ("加性VQ g8 b8 M2 (2.0bpw)", lambda w: vq(w, 8, 8, stages=2)),
            ("加性VQ g4 b8 M2 (4.0bpw)", lambda w: vq(w, 4, 8, stages=2)),
            ("分块VQ seg256 g8 b8", lambda w: vq_seg(w, 8, 8, 256)),
            ("分块VQ seg256 g4 b8", lambda w: vq_seg(w, 4, 8, 256)),
            ("分块VQ seg128 g8 b8", lambda w: vq_seg(w, 8, 8, 128)),
            ("分块加性VQ seg256 g8 b8 M2", lambda w: vq_seg(w, 8, 8, 256, stages=2)),
            ("分块加性VQ seg256 g4 b8 M2", lambda w: vq_seg(w, 4, 8, 256, stages=2)),
            ("分块加性VQ seg256 g8 b10 M2", lambda w: vq_seg(w, 8, 10, 256, stages=2)),
        ]:
            Wq, bpw = fn(W)
            report(nm, W, Wq, bpw, x)

        print("  --- 群宽扫描（W4 对称仿射；scale 存 fp16，bpw=4+16/group） ---")
        for gsz in (32, 64, 128, 256):
            Wq, bpw = affine_group(W, 4, gsz)
            report(f"W4 g{gsz}", W, Wq, bpw, x)
        print("  --- 位宽×群宽 联合（找同误差下最小字节） ---")
        for bits in (4, 5, 6):
            for gsz in (64, 128):
                Wq, bpw = affine_group(W, bits, gsz)
                report(f"W{bits} g{gsz}", W, Wq, bpw, x)


if __name__ == "__main__":
    main()
