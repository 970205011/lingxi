#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清点两份产物同名张量差异（抽样前 1MB 字节比较 + 格式/布局统计）。"""
import json, struct, mmap, collections

A = "/models/qwen4_exp-artifact/qwen4_exp.ninfer"
B = "/models/lingxi-assets/qwen4_exp.q4dense.mtp2.ninfer"

def load(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    jb = struct.unpack("<Q", mm[8:16])[0]
    d = json.loads(mm[16:16 + jb])
    po = ((16 + jb) + 4095) // 4096 * 4096
    tab = {o["name"]: o for o in d["objects"] if o.get("kind") == "tensor"}
    return mm, tab, po

mA, tA, pA = load(A)
mB, tB, pB = load(B)
shared = sorted(set(tA) & set(tB))
print("shared tensors:", len(shared))

diff = []
fmtA = collections.Counter()
fmtB = collections.Counter()
for n in shared:
    a, b = tA[n], tB[n]
    fmtA[(a["format"], a["layout"])] += 1
    fmtB[(b["format"], b["layout"])] += 1
    nb = min(a["bytes"], b["bytes"], 1 << 20)
    oa = pA + a["offset"]
    ob = pB + b["offset"]
    same = (a["bytes"] == b["bytes"] and mA[oa:oa + nb] == mB[ob:ob + nb])
    if not same:
        diff.append((n, a["format"], a["layout"], b["format"], b["layout"]))

print("A fmt:", dict(fmtA))
print("B fmt:", dict(fmtB))
print("不同张量数:", len(diff))

g = collections.Counter()
sample = {}
for n, fa, la, fb, lb in diff:
    if n.startswith("text/layers/"):
        key = "/".join(n.split("/")[:4])
    else:
        key = "/".join(n.split("/")[:2])
    g[key] += 1
    sample.setdefault(key, (fa, la, fb, lb))
for k, v in sorted(g.items()):
    fa, la, fb, lb = sample[k]
    print(f"  {v:4d}  {k:40s} A={fa}/{la}  B={fb}/{lb}")
