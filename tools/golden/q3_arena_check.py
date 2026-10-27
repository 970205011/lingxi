#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 Q3G128 专家 arena 的质量：解出专家权重，与 BF16 产物逐元素对拍。

与 q4_arena_check.py 同构；格式 Q3G128_F16S（上游 row_split_geometry 已支持）。
在 CT141 内运行：
  cd /models/lingxi && \
  PYTHONPATH=~/ninfer \
  Q3_ARENA=/dev/shm/qwen4_exp.expert-arena.q3g128.bin \
  python3.12 tools/golden/q3_arena_check.py
"""
from __future__ import annotations

import mmap
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", "~/ninfer")))
from tools.artifact.container import Artifact  # noqa: E402
from tools.artifact.layouts import ( # noqa: E402
    decode_direct,
    dequantize_row_split,
    row_split_geometry,
)

ARENA = os.environ.get("Q3_ARENA", "/models/lingxi-assets/qwen4_exp.expert-arena.q3g128.bin")
ART = os.environ.get("LINGXI_ARTIFACT", "/models/qwen4_exp-artifact/qwen4_exp.ninfer")
FMT = "Q3G128_F16S"

HDR = struct.Struct("<8s7i4x4q")  # 72 B
REF = struct.Struct("<qii")  # 16 B
NL, NE, H, FF = 48, 512, 2560, 640
ROWS_GU, ROWS_DN = 2 * FF, H  # 1280, 2560


def main() -> None:
    layers = [int(x) for x in os.environ.get("Q3_LAYERS", "0,24,47").split(",")]
    experts = [int(x) for x in os.environ.get("Q3_EXPERTS", "0,1,17,255,511").split(",")]

    gu_geo = row_split_geometry(FMT, (ROWS_GU, H))
    dn_geo = row_split_geometry(FMT, (ROWS_DN, FF))
    blob_bytes = gu_geo.payload_bytes + dn_geo.payload_bytes
    print(f"blob 几何: gu {gu_geo.payload_bytes} B + dn {dn_geo.payload_bytes} B = {blob_bytes} B"
          f"  (BF16 为 {6*FF*H*2} B → {6*FF*H*2/blob_bytes:.2f}x)")

    fd = os.open(ARENA, os.O_RDONLY)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    (magic, ver, nl, ne, nin, nff, align, flags, doff, rioff, sioff, dbytes) = HDR.unpack(
        bytes(mm[: HDR.size])
    )
    print(f"arena: {size/2**30:.2f} GiB  magic={magic} ver={ver} layers={nl} experts={ne} "
          f"n_in={nin} n_ff={nff} align={align} flags={flags}")
    assert size == dbytes + (doff - HDR.size) + HDR.size or rioff > HDR.size

    art = Artifact.open(ART)
    worst = (0.0, "")
    for L in layers:
        gu_art = decode_direct(art.payload(art.find(f"text/layers/{L}/moe/routed_gate_up")),
                               "BF16", (NE * ROWS_GU, H))
        dn_art = decode_direct(art.payload(art.find(f"text/layers/{L}/moe/routed_down")),
                               "BF16", (NE * ROWS_DN, FF))
        print(f"\nlayer{L} 专家对拍（Q3G128 vs BF16）：")
        for e in experts:
            ro, rb, fmt = REF.unpack(bytes(mm[rioff + (L * NE + e) * 16: rioff + (L * NE + e) * 16 + 16]))
            assert ro >= 0 and rb == blob_bytes, (e, ro, rb, fmt)
            base = doff + ro
            blob = bytes(mm[base: base + rb])
            gu = dequantize_row_split(blob[: gu_geo.payload_bytes], FMT, (ROWS_GU, H),
                                      dtype=torch.float32)
            dn = dequantize_row_split(blob[gu_geo.payload_bytes:], FMT, (ROWS_DN, FF),
                                      dtype=torch.float32)
            rgu = gu_art[e * ROWS_GU:(e + 1) * ROWS_GU].float()
            rdn = dn_art[e * ROWS_DN:(e + 1) * ROWS_DN].float()
            egu = float((gu - rgu).norm() / rgu.norm())
            edn = float((dn - rdn).norm() / rdn.norm())
            print(f"  expert {e:3d} (fmt={fmt}): gu 相对误差={egu:.4f}  dn 相对误差={edn:.4f}")
            for tag, v in ((f"L{L}/e{e}/gu", egu), (f"L{L}/e{e}/dn", edn)):
                if v > worst[0]:
                    worst = (v, tag)
        del gu_art, dn_art

    print(f"\nWORST rel={worst[0]:.4f} @ {worst[1]}")
    mm.close()
    os.close(fd)


if __name__ == "__main__":
    main()
