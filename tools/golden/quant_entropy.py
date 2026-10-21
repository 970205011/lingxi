#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高级压缩手段的实测（回答"能不能再榨"）：
  1) 量化索引的经验熵  → 熵编码/7z 式无损压缩能省多少
  2) outlier 分离      → 高幅值特殊处理的收益
  3) 低秩谱            → "用公式代替数值"是否可行
  4) 邻域相关          → 是否存在重复/规律（LZ/字典的前提）

在 测试机 内运行：
  cd /models/lingxi && \
  PYTHONPATH=~/ninfer \
  python3.12 tools/golden/quant_entropy.py
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


def load_expert_gu(art: Artifact, layer: int, e: int) -> torch.Tensor:
    o = art.find(f"text/layers/{layer}/moe/routed_gate_up")
    t = decode_direct(art.payload(o), o.format, o.shape)
    rows = t.shape[0] // N_EXPERTS_TOTAL
    return t[e * rows : (e + 1) * rows].to(torch.float32)


def qmax_of(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def entropy_bits(codes: torch.Tensor, n_levels: int) -> float:
    c = torch.bincount(codes.reshape(-1), minlength=n_levels).double()
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def quant_codes(W: torch.Tensor, bits: int, group: int = 128):
    out, inn = W.shape
    Wg = W.reshape(-1, group)
    qm = qmax_of(bits)
    scale = (Wg.abs().amax(1, keepdim=True) / qm).clamp_min(1e-12)
    q = torch.round(Wg / scale).clamp(-qm - 1, qm)
    Wq = (q * scale).reshape(out, inn)
    return q.to(torch.int64), Wq


def main() -> None:
    art = Artifact.open(ART)
    W = load_expert_gu(art, 0, 0)
    print(f"== layer0 expert0 gu {tuple(W.shape)} std={W.std():.5f} ==")

    print("\n[1] 标量量化的索引熵与误差（=> 熵编码后的等效 bpw）")
    for bits in (8, 4, 3, 2):
        q, Wq = quant_codes(W, bits)
        H = entropy_bits(q + qmax_of(bits) + 1, 2**bits)
        err = float((Wq - W).norm() / W.norm())
        ebpw = H + 16.0 / 128
        print(f"  {bits}bit 名义 {bits:.2f} bit/权重 → 熵 {H:.3f} → 等效 {ebpw:4.2f} bpw"
              f"  相对误差={err:.4f}  (熵编码省 {(1 - H / bits) * 100:.1f}%)")

    print("\n[2] outlier 分离（W4 底座 + 误差最大的 p% 追加 4bit）")
    base_bits, extra, group = 4, 4, 128
    _, Wq0 = quant_codes(W, base_bits, group)
    base_bpw = base_bits + 16.0 / group
    base_err = float((Wq0 - W).norm() / W.norm())
    print(f"  纯 W4 g128:              bpw={base_bpw:5.2f}  相对误差={base_err:.4f}")
    res = W - Wq0
    flat = res.abs().reshape(-1)
    for frac in (0.001, 0.005, 0.01, 0.02, 0.05):
        k = max(1, int(frac * flat.numel()))
        thr = flat.topk(k).values.min()
        mask = (res.abs() >= thr)
        qm = qmax_of(extra)
        sc = (res[mask].abs().amax() / qm).clamp_min(1e-12)
        qr = torch.round(res / sc).clamp(-qm - 1, qm) * mask
        Wq = Wq0 + qr * sc
        err = float((Wq - W).norm() / W.norm())
        bpw = base_bpw + frac * (extra + 1)  # 额外 4bit + 1bit 位置掩码
        print(f"  +outlier p={frac*100:4.1f}% 追加4bit: bpw={bpw:5.2f}  相对误差={err:.4f}")

    print("\n[3] 低秩谱（=> 能否用 U·V 公式代替原始矩阵）")
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    e = S**2
    e = e / e.sum()
    cum = e.cumsum(0)
    for r in (8, 16, 32, 64, 128, 256):
        r = min(r, cum.numel())
        print(f"  前 {r:4d} 个奇异值占比 = {cum[r-1]*100:6.2f}%   (秩{r}约需 bpw≈{16*r/W.shape[1]:.2f})")

    print("\n[4] 邻域相关（=> 是否有重复/规律可被字典/LZ 利用）")
    a = W[:, :-1].reshape(-1)
    b = W[:, 1:].reshape(-1)
    cc = torch.corrcoef(torch.stack([a, b]))[0, 1]
    a = W[:-1, :].reshape(-1)
    b = W[1:, :].reshape(-1)
    rr = torch.corrcoef(torch.stack([a, b]))[0, 1]
    # 逐行 std 的离散度（AWQ 型通道显著性）
    rowstd = W.std(1)
    print(f"  相邻列相关={cc:.4f}  相邻行相关={rr:.4f}")
    print(f"  逐行 std: mean={rowstd.mean():.5f} std={rowstd.std():.5f} "
          f"max/min={float(rowstd.max()/rowstd.min()):.2f}")


if __name__ == "__main__":
    main()
