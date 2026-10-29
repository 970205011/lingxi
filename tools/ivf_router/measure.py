#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IVFFlat 路由预筛原型：逐层训练 IVF 码本 + 倒排专家表，held-out 上测 top-k 分歧率。

数据布局（<模型资产目录>/router-calib/，由 GPTQ 校准 dump 产生）：
  moeNNN_meta.json   {"layer": L, "rows": 512, "s": 5120, "topk": 10, "hidden": 2560, ...}
  moeNNN_xpair.bf16  s×hidden bf16——MoE 入口激活 h，但同一行的 h 为其 top-k 专家重复
                     gather 了 topk 次（供逐专家 Hessian），须按 moeNNN_pairs.pr.i32
                     去重（每个唯一行只取第一次出现）。
  moeNNN_logits.f32  512 行 × 512 专家 router logits（与唯一行一一对应）——真值。
  moeNNN_sel.i32     512×topk 排序后的 top-k 专家（与 logits 推出的一致率 ~95%，
                     有并列/精度差；本脚本一律以 logits 重算为准）。

方法（每层独立）：
  1. 汇总该层全部 dump 文件 → 去重 h / logits；随机切 train/held-out。
  2. h 行 L2 归一化后 k-means 聚 k 类（默认 256）。
  3. 倒排表：每类记录 train 样本 top-k 专家的频次；查询时取最近 nprobe 类的专家
     并集（按频次截断到 --cap，默认 64）作为候选集。
  4. 分歧率：held-out 上「全量 logits top-k」vs「候选集内重打分 top-k」集合不一致率
     （任一个专家不同即分歧）。

用法示例：
  python3 measure.py --data-dir <模型资产目录>/router-calib \
      --clusters 256 --nprobe 1,2,4 --topk 8 --out-json result.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

TOPK_META_FALLBACK = 8


# ---------------------------------------------------------------- 数据加载

def bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    """uint16 位型的 bf16 → float32（位保留左移 16）。u16 形状任意，返回 f32 同形。"""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_layer(data_dir: str, layer: int, max_samples: int, seed: int):
    """读某层全部 moeNNN_* dump → (h[N,hidden] f32, logits[N,512] f32)（pr 去重后）。"""
    metas = []
    for name in os.listdir(data_dir):
        if not name.endswith("_meta.json"):
            continue
        with open(os.path.join(data_dir, name)) as f:
            m = json.load(f)
        if m.get("layer") == layer:
            metas.append((name[: -len("_meta.json")], m))
    metas.sort()
    if not metas:
        raise FileNotFoundError(f"layer {layer}: 无 meta.json")
    hs, lgs = [], []
    for base, m in metas:
        s, hidden, rows = m["s"], m["hidden"], m["rows"]
        pr = np.fromfile(os.path.join(data_dir, base + "_pairs.pr.i32"), dtype=np.int32)
        assert pr.shape[0] == s, (base, pr.shape, s)
        # pr = 行号（0..rows-1，每行 topk 个 pair 重复）；meta rows = 唯一行数。
        # xpair/logits 均按 pr 值对齐：logits[pr] 是该行的 512 维 router logits（真值）。
        assert pr.min() >= 0 and pr.max() < rows, (base, pr.min(), pr.max(), rows)
        first = np.full(rows, s, dtype=np.int64)
        np.minimum.at(first, pr, np.arange(s, dtype=np.int64))
        assert (first < s).all(), (base, "pr 非连续", rows)
        xp = np.memmap(os.path.join(data_dir, base + "_xpair.bf16"),
                       dtype=np.uint16, mode="r").reshape(s, hidden)
        lg = np.memmap(os.path.join(data_dir, base + "_logits.f32"),
                       dtype=np.float32, mode="r").reshape(rows, -1)
        hs.append(bf16_u16_to_f32(np.asarray(xp[first])))
        lgs.append(np.asarray(lg))
        del xp, lg
    h = np.concatenate(hs)          # [N, hidden] f32
    logits = np.concatenate(lgs)    # [N, 512] f32
    if max_samples and max_samples < h.shape[0]:
        rng = np.random.default_rng(seed)
        keep = rng.choice(h.shape[0], size=max_samples, replace=False)
        keep.sort()
        h, logits = h[keep], logits[keep]
    return h, logits


# ---------------------------------------------------------------- k-means

def kmeans(X: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    """行已归一化的 k-means（cosine 空间 = 点积最近）。返回质心 [k, d]（归一化）。"""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # k-means++ 初始化
    centers = np.empty((k, d), dtype=np.float32)
    centers[0] = X[rng.integers(n)]
    closest = X @ centers[0]
    for i in range(1, k):
        p = np.clip(1.0 - closest, 0.0, None)
        total = p.sum()
        if total <= 0:
            centers[i] = X[rng.integers(n)]
        else:
            centers[i] = X[rng.choice(n, p=p / total)]
        closest = np.maximum(closest, X @ centers[i])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12

    assign = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        sim = X @ centers.T                    # [n, k]
        new_assign = sim.argmax(1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            mask = assign == c
            if mask.any():
                cen = X[mask].mean(0)
                centers[c] = cen / (np.linalg.norm(cen) + 1e-12)
            else:  # 空类 → 当前离自己质心最远的点
                far = (X * centers[c]).sum(1).argmin()
                centers[c] = X[far]
    return centers


def topk_set(logits_row: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(logits_row, -k)[-k:]


# ---------------------------------------------------------------- 单层流程

def run_layer(layer: int, args) -> dict:
    t0 = time.time()
    h, logits = load_layer(args.data_dir, layer, args.samples_per_layer, args.seed + layer)
    n, hidden = h.shape
    k = args.topk
    rng = np.random.default_rng(args.seed + 1000 + layer)
    perm = rng.permutation(n)
    n_train = int(n * args.train_frac)
    tr, ho = perm[:n_train], perm[n_train:]

    # 行归一化（router logit = w·h，方向即信息）
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    centers = kmeans(h[tr].astype(np.float32), args.clusters, args.kmeans_iters,
                     args.seed + layer)

    # 倒排表：每类 train 样本 top-list 专家频次 [k_cls, 512]。
    # 注意建表用满校准 topk（meta=10，比测量 topk  richer，recall 更高——已 A/B 验证）。
    n_cls = args.clusters
    list_topk = min(args.list_topk, logits.shape[1])
    freq = np.zeros((n_cls, logits.shape[1]), dtype=np.int32)
    tr_assign = (h[tr] @ centers.T).argmax(1)
    top_tr = np.argpartition(logits[tr], -list_topk, axis=1)[:, -list_topk:]
    for ci in range(n_cls):
        mask = tr_assign == ci
        if mask.any():
            freq[ci] = np.bincount(top_tr[mask].ravel(), minlength=logits.shape[1])
    # 每类候选 = 按频次排序的专家 id（预排序一次，查询时按 cap 截断）
    cls_lists = np.argsort(-freq, axis=1).astype(np.int32)      # [n_cls, 512]
    cls_counts = np.sort(freq, axis=1)[:, ::-1]                  # 对应频次

    sim_ho = h[ho] @ centers.T                                   # [n_ho, n_cls]
    result = {"layer": layer, "n_train": int(n_train), "n_heldout": int(len(ho)),
              "hidden": int(hidden), "n_clusters": n_cls, "topk": k,
              "train_sec": round(time.time() - t0, 2), "nprobe": {}}

    for nprobe in args.nprobe_list:
        nn = np.argpartition(-sim_ho, nprobe - 1, axis=1)[:, :nprobe]   # [n_ho, nprobe]
        div = 0
        cand_sizes = np.empty(len(ho), dtype=np.int32)
        recall = np.empty(len(ho), dtype=np.float32)
        for i in range(len(ho)):
            # nprobe 类并集：按频次合并去重，截断 cap
            cand, cnt = {}, {}
            for c in nn[i]:
                for j in range(args.cap):
                    e = int(cls_lists[c, j])
                    if cls_counts[c, j] == 0:
                        break
                    cand[e] = cand.get(e, 0) + int(cls_counts[c, j])
            order = sorted(cand, key=cand.get, reverse=True)[:args.cap]
            cand_set = np.array(order, dtype=np.int64)
            cand_sizes[i] = len(cand_set)
            full = set(topk_set(logits[ho[i]], k).tolist())
            recall[i] = len(full & set(order)) / k
            if len(cand_set) >= k:
                sub = logits[ho[i], cand_set]
                ivf = set(cand_set[np.argpartition(sub, -k)[-k:]].tolist())
            else:  # 候选不足 k 个：缺的按全量补（记为分歧，除非恰好同集）
                ivf = set(cand_set.tolist())
            if ivf != full:
                div += 1
        result["nprobe"][str(nprobe)] = {
            "div_rate": round(div / len(ho), 6),
            "cand_mean": round(float(cand_sizes.mean()), 2),
            "cand_p95": round(float(np.percentile(cand_sizes, 95)), 1),
            "cand_max": int(cand_sizes.max()),
            "cand_min": int(cand_sizes.min()),
            "recall_at_k_mean": round(float(recall.mean()), 6),
            "under_k_frac": round(float((cand_sizes < k).mean()), 6),
        }
    return result


def summarize(per_layer):
    out = {}
    for nprobe in per_layer[0]["nprobe"]:
        divs = [r["nprobe"][nprobe]["div_rate"] for r in per_layer]
        cms = [r["nprobe"][nprobe]["cand_mean"] for r in per_layer]
        out[nprobe] = {
            "div_min": round(min(divs), 6), "div_median": round(float(np.median(divs)), 6),
            "div_max": round(max(divs), 6),
            "cand_mean_avg": round(float(np.mean(cms)), 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="/path/to/router-calib")
    ap.add_argument("--layers", default="0-47", help="如 0-47 或 3,5,7")
    ap.add_argument("--samples-per-layer", type=int, default=0, help="0=全用（每层 ~9216 唯一行）")
    ap.add_argument("--clusters", type=int, default=256, help="每层 k-means 类数")
    ap.add_argument("--nprobe", default="1,2,4", help="逗号分隔的查询类数扫描")
    ap.add_argument("--topk", type=int, default=8, help="测量目标 top-k（引擎 meta 为 10）")
    ap.add_argument("--list-topk", type=int, default=10,
                    help="建倒排表用的 train top-k（默认 10=校准 topk）")
    ap.add_argument("--cap", type=int, default=64, help="候选集大小上限")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--kmeans-iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    args.nprobe_list = sorted({int(x) for x in args.nprobe.split(",")})
    if "-" in args.layers:
        a, b = args.layers.split("-")
        layers = list(range(int(a), int(b) + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    if args.workers <= 1 or len(layers) == 1:
        per_layer = [run_layer(L, args) for L in layers]
    else:
        import multiprocessing as mp
        # 子进程按层并行；BLAS 线程数收敛防超订
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        with mp.Pool(args.workers) as pool:
            per_layer = pool.starmap(run_layer, [(L, args) for L in layers])
        per_layer.sort(key=lambda r: r["layer"])

    summary = summarize(per_layer)
    print(f"{'层':>3} {'n_tr':>6} {'n_ho':>6}", end="")
    for nprobe in args.nprobe_list:
        print(f" | {'div@np' + str(nprobe):>10} {'cand':>6} {'recall':>7}", end="")
    print()
    for r in per_layer:
        print(f"{r['layer']:>3} {r['n_train']:>6} {r['n_heldout']:>6}", end="")
        for nprobe in args.nprobe_list:
            s = r["nprobe"][str(nprobe)]
            print(f" | {s['div_rate']*100:>9.3f}% {s['cand_mean']:>6.1f} {s['recall_at_k_mean']*100:>6.2f}%", end="")
        print()
    print("汇总 (min/median/max):")
    for nprobe, s in summary.items():
        print(f"  nprobe={nprobe}: 分歧率 {s['div_min']*100:.3f}% / {s['div_median']*100:.3f}% / "
              f"{s['div_max']*100:.3f}%  候选均值 {s['cand_mean_avg']}")

    if args.out_json:
        payload = {"args": vars(args) | {"nprobe_list": args.nprobe_list},
                   "layers": layers, "per_layer": per_layer, "summary": summary}
        with open(args.out_json, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"已写 {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
