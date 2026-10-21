#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1 契约测试的 torch 参考生成器（同一 pocenv 环境）。

用法:  kernel_oracle.py <out_dir> <op> <json_params>
输出:  out_dir/in_<name>.f32 与 out_dir/ref.f32（小端 f32），以及 out_dir/meta.json

只做"给定的逻辑算子"的 torch 参考，与 Rust 侧 kernel 一一对应。
"""

import json
import os
import sys

import torch


def main():
    out_dir, op, params = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    os.makedirs(out_dir, exist_ok=True)
    seed = int(params.get("seed", 0))
    torch.manual_seed(seed)

    def dump(name, t):
        t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(out_dir, f"{name}.f32"))

    if op == "smoke_scale":
        n = int(params["n"])
        a = float(params["a"])
        x = torch.randn(n)
        dump("in_x", x)
        dump("ref", x * a)
    elif op == "rmsnorm":
        rows, dim = int(params["rows"]), int(params["dim"])
        eps = float(params.get("eps", 1e-6))
        x = torch.randn(rows, dim)
        gamma = torch.randn(dim) * 0.1 + 1.0
        y = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * gamma
        dump("in_x", x)
        dump("in_gamma", gamma)
        dump("ref", y)
    else:
        sys.exit(f"未知算子: {op}")

    json.dump({"op": op, "params": params}, open(os.path.join(out_dir, "meta.json"), "w"))
    print("ok")


if __name__ == "__main__":
    main()
