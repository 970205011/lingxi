#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修 MTP 产物的 stem norm：把 `mtp/embedding_norm` / `mtp/hidden_norm` 补上 wplus 的 `+1`。

## 为什么

本模型族（Qwen3.8-Flash-Next 系）的 safetensors 把 **RMSNorm 的 gamma 存成 `γ−1`**；
ninfer 转换的 **`wplus`** 步骤负责 `+1` 还原成真正的 γ。
但转换器的 +1 名单**漏了 role = `MtpStemNorm` 的两个张量**（层内的 `AttentionNorm` /
`HyperConnectionNorm` 都加对了）。结果 MTP 的 **stem** 拿 `γ−1` 当 γ 用：
**符号翻转 + 量级差 ~3 倍** ⇒ 整个 MTP 算错 ⇒ 草稿全错（`mtp-accept` 恒 0）。

实测（2026-10-16）：补上 +1 后 `mtp-accept` 从 **0/64 = 0.000** 跳到 **58/64 = 0.906**。

## 判据（gamma 必须全为正）

    修复前   embedding_norm mean=-0.7642（100% 为负）  hidden_norm mean=-0.3284（84.3% 为负）
    修复后   embedding_norm mean=+0.2358（ 0.0% 为负）  hidden_norm mean=+0.6703（ 0.3% 为负）

## 用法

    python3 tools/golden/mtp_fix_stem_norm.py <src.ninfer> <dst.ninfer>

幂等：若目标张量已经是"几乎全正"的 gamma，则跳过（并打印提示）。
"""

import json
import mmap
import shutil
import struct
import sys

import numpy as np

STEM = ("mtp/embedding_norm", "mtp/hidden_norm")


def open_ninfer(path, writable=False):
    f = open(path, "r+b" if writable else "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.ACCESS_WRITE if writable else mmap.PROT_READ)
    jl = struct.unpack("<Q", mm[8:16])[0]
    d = json.loads(mm[16 : 16 + jl])
    po = ((16 + jl) + 4095) // 4096 * 4096
    objs = {o["name"]: o for o in d["objects"] if o.get("kind") == "tensor"}
    return f, mm, po, objs


def as_f32(buf):
    a = np.frombuffer(bytes(buf), dtype="<u2")
    return (a.astype(np.uint32) << 16).view(np.float32)


def from_f32(v):
    """f32 → bf16 位模式，**round-half-even**（主干的约定；截断会差 ~25% 元素、每个 1 ulp）。"""
    u = np.ascontiguousarray(np.asarray(v, dtype=np.float32)).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype("<u2")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    assert src != dst, "别原地改：给一个 dst 路径"
    shutil.copyfile(src, dst)

    # 先只读地看一眼分布
    _, mm, po, objs = open_ninfer(src)
    need = []
    for name in STEM:
        o = objs.get(name)
        if o is None:
            raise SystemExit(f"产物里缺少 {name}")
        v = as_f32(mm[po + o["offset"] : po + o["offset"] + o["bytes"]])
        neg = float((v < 0).mean())
        print(f"  {name:<22} mean={v.mean():+.4f} 负值占比={neg * 100:.1f}%")
        if neg > 0.5:
            need.append(name)
    mm.close()

    if not need:
        print("两个 stem norm 都已是'几乎全正'的 gamma ⇒ 无需修复（幂等跳过）")
        return

    f, mm, po, objs = open_ninfer(dst, writable=True)
    for name in need:
        o = objs[name]
        off = po + o["offset"]
        v = as_f32(mm[off : off + o["bytes"]]) + 1.0
        newb = from_f32(v).tobytes()
        mm[off : off + o["bytes"]] = newb
        vv = as_f32(newb)
        print(f"  已 +1 {name:<22} -> mean={vv.mean():+.4f} 负值占比={(vv < 0).mean() * 100:.1f}%")
    mm.flush()
    mm.close()
    f.close()
    print(f"完成：{dst}")
    print("验证：lingxi-bench mtp-accept <main.ninfer> " + dst + " 'The capital of France is' 64")


if __name__ == "__main__":
    main()
