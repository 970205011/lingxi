#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MTP / QSA **注意力块**的独立复算（白盒对拍第二件）。

输入全部来自 `lingxi-bench mtp-dump <mtp.ninfer> <dir>`：
  attn_mixed_a（该块输入）、attn_out（Rust 的输出）、w_qkgv / w_q_norm / w_k_norm /
  w_o_proj / w_index_* / rope_cos / rope_sin。

按**规格**（ 与 qsa.rs 的注释）独立实现一遍：
  q|k|gate|v 四段切分 → 每头 RMSNorm → partial RoPE → 因果掩码注意力 → sigmoid 门 → o_proj
若与 Rust 的输出对不上，就说明这两者里有一个错 —— 这正是要裁的。

用法: python3 tools/golden/mtp_attn_oracle.py <dump_dir> [rot_dim]
"""
import os
import sys

import numpy as np

d = sys.argv[1]
ROT = int(sys.argv[2]) if len(sys.argv) > 2 else 64
rd = lambda n: np.fromfile(os.path.join(d, n + ".f32"), dtype="<f4")

HEADS, KV_HEADS, HD = 24, 2, 256
QD, KVD = HEADS * HD, KV_HEADS * HD          # 6144, 512
T, N = 2, 2560
EPS = 1e-6

x = rd("attn_mixed_a").reshape(T, N).astype(np.float64)
w = rd("w_qkgv").reshape(-1, N).astype(np.float64)   # [13312, N]（行 = 输出维）
qn_w = rd("w_q_norm").astype(np.float64)
kn_w = rd("w_k_norm").astype(np.float64)
o_w = rd("w_o_proj").reshape(N, QD).astype(np.float64)   # [H, q_dim]
cos = rd("rope_cos").reshape(-1, ROT).astype(np.float64)
sin = rd("rope_sin").reshape(-1, ROT).astype(np.float64)
ref = rd("attn_out").reshape(T, N).astype(np.float64)
print(f"dump: x{x.shape} w{w.shape} o{o_w.shape} cos{cos.shape} ref{ref.shape}")


def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(a).max(), 1e-12)


def rms_heads(v, g, nhead):
    v = v.reshape(T, nhead, HD)
    ms = (v * v).mean(axis=-1, keepdims=True)
    return (v / np.sqrt(ms + EPS)) * g  # g: [HD] 每头同一 gamma


def rope_partial(v, nhead):
    """前 ROT 维做 partial RoPE（NeoX 式：前一半与后一半配对）。"""
    v = v.reshape(T, nhead, HD).copy()
    r = v[:, :, :ROT].reshape(T, nhead, 2, ROT // 2)
    a, b = r[:, :, 0, :].copy(), r[:, :, 1, :].copy()
    c = cos[:T, :ROT].reshape(T, 1, 2, ROT // 2)[:,:, 0,:]
    s = sin[:T, :ROT].reshape(T, 1, 2, ROT // 2)[:,:, 0,:]
    r[:, :, 0, :] = a * c - b * s
    r[:, :, 1, :] = a * s + b * c
    return v


def attn(q, k, v):
    """标准 GQA + 因果掩码（row i 只能看 0..i）。"""
    q = q.reshape(T, HEADS, HD)
    k = k.reshape(T, KV_HEADS, HD)
    v = v.reshape(T, KV_HEADS, HD)
    rep = HEADS // KV_HEADS
    k = np.repeat(k, rep, axis=1)          # [T, HEADS, HD]
    v = np.repeat(v, rep, axis=1)
    scale = 1.0 / np.sqrt(float(HD))
    out = np.zeros((T, HEADS, HD))
    causal = np.tril(np.ones((T, T)))
    for i in range(T):
        sc = np.einsum("hd,thd->ht", q[i], k) * scale        # [HEADS, T]
        sc = np.where(causal[i][None, :] > 0, sc, -np.inf)
        m = sc.max(axis=-1, keepdims=True)
        e = np.exp(sc - m)
        e = e / e.sum(axis=-1, keepdims=True)
        out[i] = np.einsum("ht,thd->hd", e, v)
    return out.reshape(T, QD)


def run(layout):
    if layout == "qk_gv":                  # q | k | gate | v（qsa.rs 的读法）
        q = x @ w[0:QD].T
        k = x @ w[QD:QD + KVD].T
        g = x @ w[QD + KVD:QD + KVD + QD].T
        v = x @ w[QD + KVD + QD:].T
    elif layout == "qkv_g":                # q(12288 含 gate) | k | v ⇒ 再看 gate 从哪来
        q = x @ w[0:QD].T
        k = x @ w[12288:12288 + KVD].T
        g = x @ w[QD:2 * QD].T            # ← 猜：gate 在 q 之后
        v = x @ w[12288 + KVD:].T
    else:
        raise SystemExit("layout?")
    qn = rope_partial(rms_heads(q, qn_w, HEADS), HEADS)
    kn = rope_partial(rms_heads(k, kn_w, KV_HEADS), KV_HEADS)
    a = attn(qn, kn, v.reshape(T, KVD))
    g = 1.0 / (1.0 + np.exp(-g))           # sigmoid
    out = (a * g) @ o_w.T
    return out


for lay in ("qk_gv", "qkv_g"):
    o = run(lay)
    print(f"[layout {lay:<6}] attn_out 相对误差 = {rel(o, ref):.3e}   "
          f"std(ref)={ref.std():.4f} std(mine)={o.std():.4f}")
