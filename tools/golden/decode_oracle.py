#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""产物解码参考（复用 ninfer 自己的容器/布局实现），供 Rust 侧对拍。

用法: decode_oracle.py <artifact.ninfer> <tensor-name>
输出: 一行 JSON 统计（count/sum/sumsq/min/max/first4/last4），值按 f64 累计。
需要 PYTHONPATH 指向 ninfer-dev（以复用 tools.artifact.*）。
"""

import json
import sys

import torch
from tools.artifact.container import Artifact
from tools.artifact.layouts import decode_direct, dequantize_row_split


def main():
    art = Artifact.open(sys.argv[1])
    name = sys.argv[2]
    obj = art.find(name)
    payload = art.payload(obj)
    if obj.kind != "tensor":
        raise SystemExit("不是 tensor")
    if obj.layout == "contiguous-le-v1":
        t = decode_direct(payload, obj.format, obj.shape).to(torch.float32)
    elif obj.layout == "row-split-k128-v1":
        t = dequantize_row_split(payload, obj.format, obj.shape, device="cpu", dtype=torch.float32)
    else:
        raise SystemExit(f"未支持的布局: {obj.layout}")
    v = t.reshape(-1).to(torch.float64)
    stats = {
        "count": int(v.numel()),
        "sum": float(v.sum()),
        "sumsq": float((v * v).sum()),
        "min": float(v.min()),
        "max": float(v.max()),
        "first4": [float(x) for x in v[:4]],
        "last4": [float(x) for x in v[-4:]],
    }
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
