#!/usr/bin/env python3
"""GPTQ 补偿效果判决（6.2-3）：同一批专家、同一份校准 Hessian，比较
RTN arena 与 GPTQ arena 的**输出空间**误差 ||(Q−W)Xᵀ||_F/||WXᵀ||_F。

权重空间 rel L2 不是判据（GPTQ 优化输出空间，自测里 rel L2 反而略升）。
抽样：默认 3 层 × 8 专家（可 env 覆盖），对每个专家取 hessians 里 (l,e,kind) 的
Cholesky 因子做高效输出误差（eigendecomposition: ||A||_H² = ||Lᵀ Aᵀ||² 不必物化 X）。

用法：
  PYTHONPATH=~/ninfer python3 tools/arena-gen/gptq_check.py \
    /models/lingxi-assets/qwen4_exp.expert-arena.q3g128.bin \
    /models/lingxi-assets/qwen4_exp.expert-arena.q3g128.gptq.bin \
    /models/lingxi-assets/gptq-hessians
"""
from __future__ import annotations

import mmap
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", os.path.expanduser("~/ninfer"))))
from tools.artifact.layouts import dequantize_row_split, row_split_geometry  # noqa: E402

HDR = struct.Struct("<8s7i4x4q")
REF = struct.Struct("<qii")
NL, NE, H, FF = 48, 512, 2560, 640
FMT = "Q3G128_F16S"
ROWS_GU, ROWS_DN = 2 * FF, H


class Arena:
    def __init__(self, path: str):
        fd = os.open(path, os.O_RDONLY)
        self.mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        (magic, ver, nl, ne, nin, nff, align, flags,
         self.doff, self.rioff, self.sioff, self.dbytes) = HDR.unpack(bytes(self.mm[: HDR.size]))
        assert magic == b"Q4EARN1\0" and nl == NL and ne == NE

    def blob(self, l: int, e: int) -> bytes:
        ro, rb, fmt = REF.unpack(bytes(self.mm[self.rioff + (l * NE + e) * 16:
                                               self.rioff + (l * NE + e) * 16 + 16]))
        assert fmt == 5, (l, e, fmt)
        return bytes(self.mm[self.doff + ro: self.doff + ro + rb])


class ArenaBf16:
    """bf16 原档 arena（routed 保 BF16 格式 0）：blob = gu[1280×2560]‖dn[2560×640] bf16。"""

    def __init__(self, path: str):
        fd = os.open(path, os.O_RDONLY)
        self.mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        (magic, ver, nl, ne, nin, nff, align, flags,
         self.doff, self.rioff, self.sioff, self.dbytes) = HDR.unpack(bytes(self.mm[: HDR.size]))
        assert magic == b"Q4EARN1\0", magic

    def mat(self, l: int, e: int, kind: str) -> torch.Tensor:
        ro, rb, fmt = REF.unpack(bytes(self.mm[self.rioff + (l * NE + e) * 16:
                                               self.rioff + (l * NE + e) * 16 + 16]))
        assert fmt == 0 and rb == 9830400, (l, e, fmt, rb)
        buf = bytes(self.mm[self.doff + ro: self.doff + ro + rb])
        t = torch.frombuffer(bytearray(buf), dtype=torch.int16).view(torch.bfloat16).float()
        gu = t[: 2 * FF * H].reshape(2 * FF, H)
        dn = t[2 * FF * H:].reshape(H, FF)
        return gu if kind == "gu" else dn


def out_err(W: torch.Tensor, Q: torch.Tensor, Hm: torch.Tensor) -> float:
    """||(Q-W) X^T||_F / ||W X^T||_F，其中 H = X^T X。用 H^{1/2}：||A X^T||² = tr(A H A^T)。"""
    d = (Q - W).float()
    num = torch.einsum("nk,kl,nl->", d, Hm, d).clamp_min(0).sqrt()
    den = torch.einsum("nk,kl,nl->", W.float(), Hm, W.float()).clamp_min(0).sqrt()
    return float(num / (den + 1e-30))


def main() -> int:
    rtn_path, gptq_path, hess_dir, bf16_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    layers = [int(x) for x in os.environ.get("QC_LAYERS", "0,24,47").split(",")]
    experts = [int(x) for x in os.environ.get("QC_EXPERTS", "0,1,17,255,511").split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a_rtn, a_g = Arena(rtn_path), Arena(gptq_path)
    a_w = ArenaBf16(bf16_path)
    geo_gu = row_split_geometry(FMT, (ROWS_GU, H))
    geo_dn = row_split_geometry(FMT, (ROWS_DN, FF))

    ratios = []
    for L in layers:
        hess = torch.load(os.path.join(hess_dir, f"hessians_L{L:02d}.pt"),
                          map_location="cpu", weights_only=True)
        for e in experts:
            if (L, e, "gu") not in hess:
                continue
            for kind, geo, rows in (("gu", geo_gu, ROWS_GU), ("dn", geo_dn, ROWS_DN)):
                Hm = hess[(L, e, kind)].to(dev)
                # W_ref：用 RTN arena 的解量化近似 bf16 原值不可用时，以"两者均值"无意义——
                # 直接以 RTN 解量化当参考 W（RTN 误差 27~32% 时该参考带偏，但同参考下同尺度
                # 比较仍反映补偿增益；严格参考需 bf16 原档）。
                br = a_rtn.blob(L, e)
                bg = a_g.blob(L, e)
                cut = geo_gu.payload_bytes
                if kind == "gu":
                    qr_ = dequantize_row_split(br[:cut], FMT, (rows, H), dtype=torch.float32)
                    qg_ = dequantize_row_split(bg[:cut], FMT, (rows, H), dtype=torch.float32)
                else:
                    qr_ = dequantize_row_split(br[cut:], FMT, (rows, FF), dtype=torch.float32)
                    qg_ = dequantize_row_split(bg[cut:], FMT, (rows, FF), dtype=torch.float32)
                W = a_w.mat(L, e, kind).to(dev)  # bf16 原档 = 金标准参考
                e_rtn = out_err(W, qr_.to(dev), Hm)
                e_gptq = out_err(W, qg_.to(dev), Hm)
                ratios.append(e_gptq / e_rtn)
                print(f"L{L}/e{e}/{kind}: RTN={e_rtn:.4f} GPTQ={e_gptq:.4f} ×{e_gptq/e_rtn:.3f}",
                      flush=True)
    r = np.array(ratios)
    print(f"\n[判决] n={len(r)} 中位 ×{np.median(r):.3f} 均值 ×{r.mean():.3f} 最大 ×{r.max():.3f}")
    print("[判决] 通过线：中位 < 0.85（砍半目标 0.5 附近）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
