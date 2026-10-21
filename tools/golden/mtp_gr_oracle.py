#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QSA/GR 块的 **GR 前段**独立复算（白盒第三件）。

输入来自 `lingxi-bench mtp-dump <mtp.ninfer> <dir>`：
  layer_in（GR 的输入 R）、attn_mixed_a（GR 的输出）、gr_hc_norm / gr_mix_down /
  gr_mix_up / gr_block_inject。

按规格复算（gr.rs 的 body + gr_ops.cu）：
  xn    = 逐 stream RMSNorm(R, hc_norm)            # 每 hidden 一片用各自 gamma 片
  d     = silu((mix_down @ xn) / hc)               # epi=1：silu(acc*a)，a=1/hc
  imw   = sigmoid(mix_up @ d)                      # epi=2：sigmoid(acc*a)*b，a=b=1
  mixed = mean_g(imw * xn)                         # 对 hc 求均值
用法: python3 tools/golden/mtp_gr_oracle.py <dump_dir>
"""
import os
import sys

import numpy as np

d = sys.argv[1]
rd = lambda n: np.fromfile(os.path.join(d, n + ".f32"), dtype="<f4")

HC, H, DIM, LR, T = 4, 2560, 10240, 320, 2
EPS = 1e-6

R = rd("layer_in").reshape(T, DIM).astype(np.float64)
ref = rd("attn_mixed_a").reshape(T, H).astype(np.float64)
hcn = rd("gr_hc_norm").astype(np.float64)
md = rd("gr_mix_down").reshape(LR, DIM).astype(np.float64)
mu = rd("gr_mix_up").reshape(DIM, LR).astype(np.float64)
bi = rd("gr_block_inject").reshape(HC, DIM).astype(np.float64)
print(f"dump: R{R.shape} ref{ref.shape} hcn{hcn.shape} md{md.shape} mu{mu.shape} bi{bi.shape}")

rel = lambda a, b: np.abs(a - b).max() / max(np.abs(a).max(), 1e-12)

# 1) 逐 stream RMSNorm
x = R.reshape(T, HC, H)
xn = (x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)) * hcn.reshape(HC, H)
xn = xn.reshape(T, DIM)

# 2) d = silu(mix_down @ xn / hc)（silu(t) = t * sigmoid(t)）
t = (xn @ md.T) / HC
d = t / (1.0 + np.exp(-t))

# 3) imw = sigmoid(mix_up @ d)
imw = 1.0 / (1.0 + np.exp(-(d @ mu.T)))

# 4) mixed = mean_g(imw * xn)
mixed = (imw.reshape(T, HC, H) * xn.reshape(T, HC, H)).mean(1)

# 5) injw = 2*sigmoid((block_inject @ xn) / hc)
injw = 2.0 / (1.0 + np.exp(-((xn @ bi.T) / HC)))

print(f"[GR] attn_mixed_a 相对误差 = {rel(mixed, ref):.3e}   "
      f"std(ref)={ref.std():.4f} std(mine)={mixed.std():.4f}")
print(f"[GR] injw（没落 Rust 侧做对照，仅打印）mean={injw.mean():+.4f} std={injw.std():.4f}")
