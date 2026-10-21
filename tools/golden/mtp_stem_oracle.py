#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MTP stem 的**白盒对拍**：用 Rust 侧 dump 的输入+权重，在 numpy 里复算 stem，与 Rust 的输出比。

为什么需要它：黑盒已经把能猜的都试过了（token 对齐、indexer、h 身份、同形状交换、基座同源…），
全部否掉，而量级又是正常的 —— 剩下只能靠**逐步比中间量**来把"算式错"和"接线错"分开。

用法:
  python3 tools/golden/mtp_stem_oracle.py <dump_dir>
其中 dump_dir 由 `lingxi-bench mtp-dump <mtp.ninfer> <dump_dir>` 生成。
"""
import os
import sys

import numpy as np

d = sys.argv[1]
def rd(name):
    return np.fromfile(os.path.join(d, name + ".f32"), dtype="<f4")

N, HC, T = 2560, 4, 2
emb = rd("emb").reshape(T, N).astype(np.float64)
rin = rd("rin").reshape(T, HC, N).astype(np.float64)
fce = rd("fc_embedding").reshape(N, N).astype(np.float64)   # [out, in]（与 proj_t1 同一读法）
fch = rd("fc_hidden").reshape(N, N).astype(np.float64)
enorm = rd("embedding_norm").astype(np.float64)
hnorm = rd("hidden_norm").astype(np.float64)
R_rust = rd("R").reshape(T, HC, N).astype(np.float64)

print(f"dump: emb{emb.shape} rin{rin.shape} fce{fce.shape} fch{fch.shape} "
      f"enorm{enorm.shape} hnorm{hnorm.shape} R{R_rust.shape}")


def rel(a, b):
    den = max(np.abs(a).max(), 1e-12)
    return np.abs(a - b).max() / den


def rmsnorm(x, g, eps=1e-6):
    ms = (x * x).mean(axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)) * g


# ── 两条分支（与 §3.52 对齐）──
en = rmsnorm(emb, enorm)                    # [T, N]
e2 = en @ fce.T                             # [T, N]

# 变体 W：整段 10240 一起 norm（Rust 现在用的）
hn_w = rmsnorm(rin.reshape(T, HC * N), hnorm).reshape(T, HC, N)
R_w = (hn_w @ fch.T) + e2[:, None, :]

# 变体 S：逐 stream norm（每 N 一片用各自的 gamma 片）
hn_s = rmsnorm(rin, hnorm.reshape(HC, N))
R_s = (hn_s @ fch.T) + e2[:, None, :]

print(f"[变体 W 整段 10240] R 相对误差 = {rel(R_w, R_rust):.3e}")
print(f"[变体 S 逐 stream ] R 相对误差 = {rel(R_s, R_rust):.3e}")

# 附带把两条分支自己的量级打出来，便于判断"是不是某个分支根本没生效"
for tag, v in (("en", en), ("e2", e2), ("hn_w", hn_w), ("hn_s", hn_s)):
    print(f"    {tag:<5} mean={v.mean():+.4f} std={v.std():.4f} max|x|={np.abs(v).max():.3f}")
print(f"    R_rust mean={R_rust.mean():+.4f} std={R_rust.std():.4f} max|x|={np.abs(R_rust).max():.3f}")
