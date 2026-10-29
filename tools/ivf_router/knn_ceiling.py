#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""k-NN 邻域一致性上界探测：任何「h 空间近邻 → 候选专家」方法（含 IVF）的分歧率下界。

对每层 held-out h，在 train h 中找 cosine k 近邻，取邻居 top-10 专家并集作候选（
不限 64——量级 ~10k），算候选内重打分 top-k 与全量 top-k 的分歧率。若此上界仍高于
判据，则 IVF（候选 ≤64）在原理上不可能达标。

用法：
  python3 knn_ceiling.py --layers 0-47 --k 1,4,16 --topk 8 --workers 8
"""
import argparse
import json
import os

import numpy as np

from measure import load_layer, topk_set


def run_layer(layer, args):
    h, logits = load_layer(args.data_dir, layer, args.samples_per_layer, args.seed + layer)
    n = h.shape[0]
    rng = np.random.default_rng(args.seed + 1000 + layer)
    perm = rng.permutation(n)
    tr, ho = perm[:int(n * args.train_frac)], perm[int(n * args.train_frac):]
    hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    sim = hn[ho] @ hn[tr].T                                  # [n_ho, n_tr]
    top10_tr = np.argpartition(logits[tr], -10, axis=1)[:, -10:]
    out = {"layer": layer}
    for k in args.k_list:
        nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        div, rec = 0, []
        for i in range(len(ho)):
            cand = np.unique(top10_tr[nn[i]].ravel())
            full = set(topk_set(logits[ho[i]], args.topk).tolist())
            rec.append(len(full & set(cand.tolist())) / args.topk)
            sub = logits[ho[i], cand]
            ivf = set(cand[np.argpartition(sub, -args.topk)[-args.topk:]].tolist())
            div += ivf != full
        out[f"k{k}"] = {"div_rate": round(div / len(ho), 6),
                        "recall_at_k": round(float(np.mean(rec)), 6)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="/path/to/router-calib")
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--k", default="1,4,16")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--samples-per-layer", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    args.k_list = sorted({int(x) for x in args.k.split(",")})
    if "-" in args.layers:
        a, b = args.layers.split("-")
        layers = list(range(int(a), int(b) + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import multiprocessing as mp
    if args.workers <= 1 or len(layers) == 1:
        rows = [run_layer(L, args) for L in layers]
    else:
        with mp.Pool(args.workers) as pool:
            rows = pool.starmap(run_layer, [(L, args) for L in layers])
        rows.sort(key=lambda r: r["layer"])
    print(f"{'层':>3}", "".join(f" | {'k='+str(k)+' div':>12} {'recall':>7}" for k in args.k_list))
    for r in rows:
        print(f"{r['layer']:>3}", "".join(
            f" | {r[f'k{k}']['div_rate']*100:>11.3f}% {r[f'k{k}']['recall_at_k']*100:>6.2f}%"
            for k in args.k_list))
    for k in args.k_list:
        divs = [r[f"k{k}"]["div_rate"] for r in rows]
        print(f"k={k}: div min/median/max = {min(divs)*100:.3f}% / "
              f"{float(np.median(divs))*100:.3f}% / {max(divs)*100:.3f}%")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"args": vars(args) | {"k_list": args.k_list},
                       "per_layer": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
