#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测专家权重对象的形状/格式（自研压缩"压缩率↔误差"前沿量测的第一步）。

在 CT141 内运行：
  cd /models/lingxi && \
  PYTHONPATH=~/ninfer \
  python3.12 tools/golden/quant_probe.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", "~/ninfer")))
from tools.artifact.container import Artifact  # noqa: E402
from tools.artifact.layouts import decode_direct  # noqa: E402

ART = os.environ.get("LINGXI_ARTIFACT", "/models/qwen4_exp-artifact/qwen4_exp.ninfer")


def main() -> None:
    art = Artifact.open(ART)
    for layer in (0, 1):
        for role in ("routed_gate_up", "routed_down"):
            name = f"text/layers/{layer}/moe/{role}"
            try:
                o = art.find(name)
            except Exception as e:  # noqa: BLE001
                print(f"{name}: 未找到 ({e})")
                continue
            print(f"{name}: shape={tuple(o.shape)} format={o.format} layout={o.layout}")

    # 抽 layer0 的一个专家，量基本统计
    name = "text/layers/0/moe/routed_gate_up"
    o = art.find(name)
    payload = art.payload(o)
    t = decode_direct(payload, o.format, o.shape)
    print(f"\n{name}: tensor shape={tuple(t.shape)} dtype={t.dtype}")
    w = t[0].to(torch.float32).numpy()
    print(
        f"expert0 gu: shape={w.shape} mean={w.mean():.5f} std={w.std():.5f} "
        f"absmax={np.abs(w).max():.5f}"
    )


if __name__ == "__main__":
    main()
