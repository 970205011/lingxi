#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 Q4G64 专家 arena 的质量：解出专家权重，与 BF16 产物逐元素对拍。

arena 格式 Q4EARN1 与编码器均来自 ninfer-dev：
  src/targets/qwen4_exp/impl/kernels/moe_expert_arena.h （Strata 借鉴，MIT）
  tools/convert/common/quantize.py + tools/artifact/layouts.py （Q4G64_F16S, row-split-k128-v1）

在 CT141 内运行：
  cd /models/lingxi && \
  PYTHONPATH=~/ninfer \
  python3.12 tools/golden/q4_arena_check.py
"""
from __future__ import annotations

import mmap
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", "~/ninfer")))
from tools.artifact.container import Artifact  # noqa: E402
from tools.artifact.layouts import ( # noqa: E402
    decode_direct,
    dequantize_row_split,
    row_split_geometry,
)

ARENA = os.environ.get("Q4_ARENA", "/models/lingxi-assets/qwen4_exp.expert-arena.q4g64.bin")
ART = os.environ.get("LINGXI_ARTIFACT", "/models/qwen4_exp-artifact/qwen4_exp.ninfer")
FMT = os.environ.get("FMT", "Q4G64_F16S")
assert FMT in ("Q4G64_F16S", "Q3G128_F16S"), FMT
# arena 索引里的格式码（Q4EARN1，见 crates/lingxi-artifact/src/expert_arena.rs）。
FMT_CODE = {"Q4G64_F16S": 1, "Q3G128_F16S": 5}[FMT]

HDR = struct.Struct("<8s7i4x4q")  # 72 B
REF = struct.Struct("<qii")  # 16 B
NL, NE, H, FF = 48, 512, 2560, 640
ROWS_GU, ROWS_DN = 2 * FF, H  # 1280, 2560


def main() -> None:
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

    art = Artifact.open(ART)
    gu_art = decode_direct(art.payload(art.find("text/layers/0/moe/routed_gate_up")),
                           "BF16", (NE * ROWS_GU, H))
    dn_art = decode_direct(art.payload(art.find("text/layers/0/moe/routed_down")),
                           "BF16", (NE * ROWS_DN, FF))

    print(f"\nlayer0 专家对拍（{FMT} vs BF16）:")
    for e in (0, 1, 17, 255, 511):
        ro, rb, fmt = REF.unpack(bytes(mm[rioff + (0 * NE + e) * 16: rioff + (0 * NE + e) * 16 + 16]))
        assert ro >= 0 and rb == blob_bytes and fmt == FMT_CODE, (e, ro, rb, fmt, FMT_CODE)
        base = doff + ro  # ArenaBlobRef::offset 相对 data 区起点
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

    mm.close()
    os.close(fd)


if __name__ == "__main__":
    main()
