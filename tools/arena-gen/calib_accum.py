#!/usr/bin/env python3
"""GPTQ 校准 Hessian 离线累加（6.2-3）：消费 LINGXI_MOE_DUMP + LINGXI_MOE_DUMP_ACT 的
dump 目录，产出 hessians_L{层}.pt 分片（quantize_expert_arena.py 的 GPTQ_H 输入）。

每个 dump 调用（moe{NNN}_*）含：
  _xpair.bf16   s×hidden   每对 gather 的 MoE 入口 x（gu 的输入，K=hidden）
  _act.bf16     s×inter    每对 swiglu 中间激活（dn 的输入，K=inter）
  _pairs.pe.u32 / _pr.i32  对→(专家,行) 名单（pair 序 = 专家升序）
  _meta.json    layer/rows/s/topk/hidden/inter

累加：H_gu[(l,e)] += Σ_{对p: e_p=e} x_p x_p^T（hidden² 矩阵）
      H_dn[(l,e)] += Σ_{对p: e_p=e} act_p act_p^T（inter² 矩阵）

实现要点（2026-10-21 两轮事故后的定型版）：
  * **两遍法**：先扫 meta 按层分组，再逐层处理。必须——双卡 prefill 线程
    （LINGXI_PREFILL_THREADS=1）让 0..23 / 24..47 两条流的 dump 交错到达，
    文件名序 ≠ 层序；按到达顺序 flush 会把每层 .pt 写成残缺子集互相覆盖。
  * **GPU 常驻累加器**：每层 512×2 个 H 全在显存（13.4+0.8 GB，32G 卡够用），
    组 matmul 结果就地 +=，层尾一次性 D2H。host 往返版实测 4 min/层（瓶颈是
    ~1000 次/层的小 D2H 同步，不是算力——整层才 ~0.5 TFLOP）；GPU 版 ~1 min/层。

用法：
  python3 tools/arena-gen/calib_accum.py <dump_dir> <out_dir>
产出 <out_dir>/hessians_L{层:02d}.pt（打包器 GPTQ_H 指向目录即可，逐层懒加载）。
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

KINDS = ("gu", "dn")


def main() -> int:
    dump_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = sorted(f for f in os.listdir(dump_dir) if re.fullmatch(r"moe\d+_meta\.json", f))
    assert names, f"{dump_dir} 里没有 moeNNN_meta.json（忘了 LINGXI_MOE_DUMP_ACT=1？）"
    print(f"[calib] {len(names)} 个 dump 调用", flush=True)

    ne = 512
    n_calls = 0

    # ── 两遍法：先按层分组（原因见模块 docstring）──
    by_layer: dict[int, list[str]] = {}
    for meta_name in names:
        base = meta_name[: -len("_meta.json")]
        with open(os.path.join(dump_dir, meta_name)) as f:
            m = json.load(f)
        by_layer.setdefault(m["layer"], []).append(base)
    print("[calib] 层分布: " + " ".join(f"L{l}:{len(v)}" for l, v in sorted(by_layer.items())),
          flush=True)

    for layer in sorted(by_layer):
        metas = []
        for base in by_layer[layer]:
            with open(os.path.join(dump_dir, base + "_meta.json")) as f:
                metas.append((base, json.load(f)))
        hd = metas[0][1]["hidden"]
        it = metas[0][1]["inter"]
        assert all(m["hidden"] == hd and m["inter"] == it for _, m in metas), "层内维度不一致"
        # GPU 常驻累加器（显存 ~14 GB/层）
        H_gu = torch.zeros(ne, hd, hd, dtype=torch.float32, device=dev)
        H_dn = torch.zeros(ne, it, it, dtype=torch.float32, device=dev)
        for base, m in metas:
            s, hidden, inter = m["s"], m["hidden"], m["inter"]
            pe = np.fromfile(os.path.join(dump_dir, base + "_pairs.pe.u32"), dtype=np.uint32)
            assert pe.shape[0] == s, (base, pe.shape, s)
            xpair = np.fromfile(os.path.join(dump_dir, base + "_xpair.bf16"),
                                dtype=np.uint16).reshape(s, hidden)
            act = np.fromfile(os.path.join(dump_dir, base + "_act.bf16"),
                              dtype=np.uint16).reshape(s, inter)
            bounds = np.flatnonzero(np.diff(pe)) + 1
            starts = np.concatenate([[0], bounds])
            ends = np.concatenate([bounds, [s]])
            for kind, mat, Hall in (("gu", xpair, H_gu), ("dn", act, H_dn)):
                # uint16 位型 → int16（位保留）→ torch view bf16（与打包器同招）
                t = torch.from_numpy(mat.astype(np.int16)).view(torch.bfloat16).to(dev).float()
                for a, b in zip(starts, ends):
                    e = int(pe[a])
                    Xg = t[a:b]                    # [m, K]
                    Hall[e] += Xg.T @ Xg           # GPU 就地累加（view iadd，无 D2H）
            n_calls += 1
        # 层尾：向量化筛出非零专家（一次归约 + 一次同步，避免逐专家 sync）并 D2H 落盘
        out: dict = {}
        hit_gu = (H_gu != 0).any(dim=(1, 2)).nonzero().flatten().tolist()
        hit_dn = (H_dn != 0).any(dim=(1, 2)).nonzero().flatten().tolist()
        for e in hit_gu:
            out[(layer, e, "gu")] = H_gu[e].cpu()
        for e in hit_dn:
            out[(layer, e, "dn")] = H_dn[e].cpu()
        path = os.path.join(out_dir, f"hessians_L{layer:02d}.pt")
        torch.save(out, path)
        n_gu = sum(1 for k in out if k[2] == "gu")
        n_dn = sum(1 for k in out if k[2] == "dn")
        print(f"[calib] 层 {layer} → {path}（gu {n_gu} / dn {n_dn}）", flush=True)
        del H_gu, H_dn
        # 释放 dump 文件占的盘（CALIB_DELETE=1 时；默认保留，便于分层重跑/校验后再清）
        if os.environ.get("CALIB_DELETE") == "1":
            for base, _ in metas:
                for suffix in ("_xpair.bf16", "_act.bf16", "_pairs.pe.u32",
                               "_pairs.pr.i32", "_meta.json"):
                    os.remove(os.path.join(dump_dir, base + suffix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
