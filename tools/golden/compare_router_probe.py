#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较引擎 LINGXI_MOE_DUMP 与 HF probe_router 的 MoE 入口 x / router logits / top-k。
用法：python3 compare_router_probe.py <engine_dir> <hf_dir>
"""
import os, sys
import numpy as np

ROWS, H, E, K = 5, 2560, 512, 10

def rd(p, dt):
    return np.fromfile(p, dtype=dt)

def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

def main():
    edir, hdir = sys.argv[1], sys.argv[2]
    for li in (0, 1):
        print(f"===== layer {li} =====")
        ex = rd(os.path.join(edir, f"l{li}", "moe000_x.f32"), np.float32)
        hx = rd(os.path.join(hdir, f"l{li}_hf_x.f32"), np.float32)
        print(f"x shapes engine={ex.shape} hf={hx.shape}")
        if ex.size == hx.size:
            ex = ex.reshape(ROWS, H); hx = hx.reshape(ROWS, H)
            for r in range(ROWS):
                print(f"  x row{r} rel={rel(ex[r], hx[r]):.3e}")
        el = rd(os.path.join(edir, f"l{li}", "moe000_logits.f32"), np.float32).reshape(ROWS, E)
        hl = rd(os.path.join(hdir, f"l{li}_hf_logits.f32"), np.float32).reshape(ROWS, E)
        for r in range(ROWS):
            print(f"  logits row{r} rel={rel(el[r], hl[r]):.3e}  "
                  f"max|d|={np.max(np.abs(el[r]-hl[r])):.3f}")
        es = rd(os.path.join(edir, f"l{li}", "moe000_sel.i32"), np.int32).reshape(ROWS, K)
        hs = rd(os.path.join(hdir, f"l{li}_hf_sel.i32"), np.int32).reshape(ROWS, K)
        # HF 风格 softmax(f32) 重算概率，看第 10/11 名间隔
        def probs(z):
            z = z.astype(np.float64); z -= z.max(-1, keepdims=True)
            p = np.exp(z); return p / p.sum(-1, keepdims=True)
        for r in range(ROWS):
            hp = probs(hl[r]); ep = probs(el[r])
            ho = np.argsort(-hp); eo = np.argsort(-ep)
            flip = sorted(set(hs[r].tolist()) ^ set(es[r].tolist()))
            gap_h = hp[ho[K-1]] - hp[ho[K]]
            gap_e = ep[eo[K-1]] - ep[eo[K]]
            print(f"  row{r}: engine top10={sorted(es[r].tolist())}")
            print(f"          hf     top10={sorted(hs[r].tolist())}")
            print(f"          对称差={flip}  hf 10/11 概率差={gap_h:.3e}  engine 10/11 差={gap_e:.3e}")
            # logits 全局最大偏差位置与量级
            d = np.abs(el[r] - hl[r])
            top = np.argsort(-d)[:3]
            print(f"          logits 最大偏差位: {[(int(i), float(el[r][i]), float(hl[r][i])) for i in top]}")

if __name__ == "__main__":
    main()
