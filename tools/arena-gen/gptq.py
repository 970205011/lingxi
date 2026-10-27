#!/usr/bin/env python3
"""GPTQ one-shot 误差补偿（PLAN_NEXT -3）：真实校准的 Hessian ⇒ 同 bpw 误差砍半。

背景（2026-10-19 实测）：Q3G128 纯 RTN 专家权重 rel L2 27~32%（Q4G64 为 11~12%），
pad4983 长程金丝雀翻船。GPTQ（OBQ 块状版）用校准激活的二阶信息把"已量化列的误差"
补偿进未量化列，3-bit 档输出空间误差通常降 20~40%（≈砍半目标）。

量化语义与 `tools/arena-gen/quantize_expert_arena.py` 的消费端**逐位一致**：
scale = fp16(max_abs/qmax)、reciprocal = f32(1/scale)、codes = clamp(round(w*recip), -4, 3)，
组 = 沿 K（输入）维 128 连续列、逐行 scale。编码复用上游 `encode_row_split`，
产物可直接进 arena（或先过 q3_arena_check.py 对拍）。

形状约定（与 quantize_expert_arena.py 相同）：
- W: [N, K]（gu: [1280, 2560]，dn: [2560, 640]），K 维 = 激活维，组沿 K。
- H: [K, K] = X^T X（X: [m, K] 为该专家在校准流上收集的输入激活）。

自测（合成相关激活，无需引擎）：
  python3 tools/arena-gen/gptq.py
  断言：GPTQ 输出空间误差 < RTN 输出空间误差（幅度校验 rel L2 也打印）。

校准 Hessian 的捕获（下一步，引擎侧）：见 PLAN_NEXT -3 记录。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", os.path.expanduser("~/ninfer"))))

from tools.artifact.layouts import encode_row_split, row_split_geometry  # noqa: E402
from tools.artifact.numeric import get_format  # noqa: E402
from tools.convert.common.quantize import _canonical_scale_words  # noqa: E402


def hessian_add(H: torch.Tensor | None, X: torch.Tensor) -> torch.Tensor:
    """累加 H += X^T X。X: [m, K]（该专家本次捕获的输入激活）。首次调用传 H=None。"""
    X = X.detach().float()
    if H is None:
        H = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float32, device=X.device)
    H += X.T @ X
    return H


def _group_quant(block: torch.Tensor, qmin: int, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    """RTN 量化一个列块 [N, B]：逐行 fp16 scale（上游 _canonical_scale_words 语义）。"""
    max_abs = block.abs().amax(dim=1).cpu()
    scales, reciprocal = _canonical_scale_words(max_abs, qmax)
    scales = scales.to(block.device)
    reciprocal = reciprocal.to(block.device).unsqueeze(1)
    codes = torch.clamp(torch.round(block * reciprocal), qmin, qmax).to(torch.int8)
    return codes, scales


def rtn_quant(W: torch.Tensor, group_size: int = 128, qmin: int = -4, qmax: int = 3):
    """整矩阵 RTN（参考实现，语义 = 上游 quantize_matrix；K 需被 group_size 整除）。"""
    N, K = W.shape
    assert K % group_size == 0
    Wf = W.float()
    codes = torch.empty(N, K, dtype=torch.int8, device=Wf.device)
    scales = torch.empty(N, K // group_size, dtype=torch.float16, device=Wf.device)
    for g in range(K // group_size):
        blk = Wf[:, g * group_size:(g + 1) * group_size]
        c, s = _group_quant(blk, qmin, qmax)
        codes[:, g * group_size:(g + 1) * group_size] = c
        scales[:, g] = s
    return codes, scales


def gptq_quant_groupwise(
    W: torch.Tensor,
    H: torch.Tensor,
    group_size: int = 128,
    qmin: int = -4,
    qmax: int = 3,
    percdamp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ 组版：每 128 列一组，组内列间传播 + 组尾跨组传播（官方 blocksize 口径）。

    组 scale = 组首 running 权重的逐行 max_abs（fp16 语义），组内所有列共用
    （与 RTN 的"组内 128 列共用一行 scale"完全同构）。codes 逐列从 running 权重算。
    """
    N, K = W.shape
    assert K % group_size == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = W.device if W.is_cuda else (torch.device("cuda") if torch.cuda.is_available() else W.device)
    Wf = W.detach().to(device=dev, dtype=torch.float32).clone()
    Hf = H.detach().to(device=dev, dtype=torch.float32).clone()
    Hf += (percdamp * torch.diag(Hf).mean()) * torch.eye(K, device=dev)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hf))

    codes = torch.empty(N, K, dtype=torch.int8, device=dev)
    scales = torch.empty(N, K // group_size, dtype=torch.float16, device=dev)
    G = K // group_size
    for g in range(G):
        i0, i1 = g * group_size, (g + 1) * group_size
        blk = Wf[:, i0:i1]
        # 组 scale = 组首 running 权重的逐行 max_abs（fp16 语义），组内 128 列共用（与 RTN 同构）
        max_abs = blk.abs().amax(dim=1).cpu()
        s, recip = _canonical_scale_words(max_abs, qmax)
        s = s.to(dev)
        scales[:, g] = s
        r = recip.to(dev)
        for j in range(i0, i1):
            col = Wf[:, j]
            q = torch.clamp(torch.round(col * r), qmin, qmax)
            codes[:, j] = q.to(torch.int8)
            # OBQ 逐列传播：δ = 本列量化误差（权重空间），按 H⁻¹ 比例摊到未量化列
            d = Hinv[j, j]
            delta = (col - q * s) / d
            if j + 1 < K:
                Wf[:, j + 1:] -= delta.unsqueeze(1) * Hinv[j, j + 1:].unsqueeze(0)
    return codes, scales


def gptq_quant_fast(
    W: torch.Tensor,
    H: torch.Tensor,
    group_size: int = 128,
    qmin: int = -4,
    qmax: int = 3,
    percdamp: float = 0.01,
    sub_block: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """快速版：组（128）内按 sub_block（默认 32）列做**子块量化 + 秩-sub_block 补偿**，
    跨子块/组精确传播（全逆 Schur 更新，matmul 发射）。sub_block=1 退化为逐列版
    （质量上限、慢 20×）；sub_block=128 全组一块（最快、质量略降）。实测 sub_block=32
    在真实 Hessian 上与逐列版质量差 <1%，速度 ~50ms/矩阵（全量 ~20min）。

    W: [N, K] → codes [N, K] i8、scales [N, G] f16。
    """
    N, K = W.shape
    assert K % group_size == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = W.device if W.is_cuda else (torch.device("cuda") if torch.cuda.is_available() else W.device)
    Wf = W.detach().to(device=dev, dtype=torch.float32).clone()
    Hf = H.detach().to(device=dev, dtype=torch.float32).clone()
    Hf += (percdamp * torch.diag(Hf).mean()) * torch.eye(K, device=dev)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hf))

    codes = torch.empty(N, K, dtype=torch.int8, device=dev)
    scales = torch.empty(N, K // group_size, dtype=torch.float16, device=dev)
    for g in range(K // group_size):
        i0, i1 = g * group_size, (g + 1) * group_size
        # 组 scale = 组首 running 权重的逐行 max_abs（与 RTN 同构），组内共用
        blk = Wf[:, i0:i1]
        max_abs = blk.abs().amax(dim=1).cpu()
        s, recip = _canonical_scale_words(max_abs, qmax)
        s = s.to(dev)
        r = recip.to(dev)
        scales[:, g] = s
        # 组内推进：sub_block >= group_size（默认 128）时整块量化 + 秩-128 补偿一发
        # （2026-10-21 实测：与逐列版输出误差逐位同值（×0.587 vs ×0.587），子块内传播
        # 对质量零贡献、对速度百倍拖累；子块版保留给未来更激进 Hessian 用）。
        for j0 in range(i0, i1, sub_block):
            j1 = min(j0 + sub_block, i1)
            if j1 - j0 > 1 and sub_block < group_size:
                for j in range(j0, j1 - 1):
                    col = Wf[:, j]
                    q = torch.clamp(torch.round(col * r), qmin, qmax)
                    d = Hinv[j, j]
                    delta = (col - q * s) / d
                    Wf[:, j + 1:j1] -= delta.unsqueeze(1) * Hinv[j, j + 1:j1].unsqueeze(0)
            blk2 = Wf[:, j0:j1]
            Q2 = torch.clamp(torch.round(blk2 * r.unsqueeze(1)), qmin, qmax)
            codes[:, j0:j1] = Q2.to(torch.int8)             # 整块写（省 2560 次 strided 单列写）
            err = blk2 - Q2 * s.unsqueeze(1)                # [N, sb]
            dblk = Hinv[j0:j1, j0:j1].diag()                # [sb]
            if j1 < K:
                Wf[:, j1:] -= err @ (Hinv[j0:j1, j1:] / dblk.unsqueeze(1))
    return codes, scales


def gptq_quant_fast_batched(
    Wb: torch.Tensor,
    Hb: torch.Tensor,
    group_size: int = 128,
    qmin: int = -4,
    qmax: int = 3,
    percdamp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """gptq_quant_fast（sub_block≥group 整块路径）的跨专家批量版，数学逐项相同。

    Wb: [B, N, K]（CPU/GPU 皆可），Hb: [B, K, K]（GPU）→ codes [B, N, K] i8、
    scales [B, N, G] f16。单矩阵 ~950ms（逐专家 Python/kernel 开销）→ B=16 批量
    ~40ms/矩阵（2026-10-21 重打包性能定案：瓶颈是 per-expert 固定开销，摊 16 份）。
    """
    B, N, K = Wb.shape
    assert K % group_size == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = Hb.device
    Wf = Wb.detach().to(device=dev, dtype=torch.float32).clone()
    Hf = Hb.detach().to(dtype=torch.float32).clone()
    idx = torch.arange(K, device=dev)
    Hf[:, idx, idx] += (percdamp * torch.diagonal(Hf, dim1=1, dim2=2).mean(-1)).unsqueeze(1)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hf))     # [B,K,K]

    codes = torch.empty(B, N, K, dtype=torch.int8, device=dev)
    scales = torch.empty(B, N, K // group_size, dtype=torch.float16, device=dev)
    for g in range(K // group_size):
        i0, i1 = g * group_size, (g + 1) * group_size
        blk = Wf[:, :, i0:i1]                                     # [B,N,gs]
        max_abs = blk.abs().amax(dim=2).reshape(-1).cpu()         # [B*N]，每组一次同步
        s, recip = _canonical_scale_words(max_abs, qmax)
        s = s.reshape(B, N).to(dev)
        r = recip.reshape(B, N).to(dev)
        scales[:, :, g] = s
        Q = torch.clamp(torch.round(blk * r.unsqueeze(-1)), qmin, qmax)
        codes[:, :, i0:i1] = Q.to(torch.int8)
        err = blk - Q * s.unsqueeze(-1)                           # [B,N,gs]
        dblk = Hinv[:, i0:i1, i0:i1].diagonal(dim1=1, dim2=2)     # [B,gs]
        if i1 < K:
            Wf[:, :, i1:] -= torch.bmm(err, Hinv[:, i0:i1, i1:] / dblk.unsqueeze(-1))
    return codes, scales


def gptq_and_encode(W: torch.Tensor, H: torch.Tensor, fmt: str,
                    device: str | torch.device | None = None) -> bytes:
    """GPTQ 量化 + row-split 编码（quantize_and_encode 的 1:1 替代）。

    W: [N, K]；H: [K, K] = X^T X（同形状激活二阶矩）。组尺寸/位宽取 fmt 规格。
    """
    spec = get_format(fmt)
    assert spec.group_size and spec.group_size > 1, "仅支持 grouped 格式"
    codes, scales = gptq_quant_fast(
        W, H, group_size=spec.group_size, qmin=spec.qmin, qmax=spec.qmax)
    N, K = codes.shape
    G = K // spec.group_size
    return encode_row_split(
        codes.reshape(N, G, spec.group_size).cpu(),
        scales.cpu(), spec, tuple(W.shape))


def _out_err(W: torch.Tensor, Q: torch.Tensor, S: torch.Tensor,
             X: torch.Tensor, group_size: int) -> float:
    """输出空间相对误差 ||X(Q−W)^T||_F / ||XW^T||_F。"""
    G = Q.shape[1] // group_size
    deq = (Q.float().reshape(Q.shape[0], G, group_size)
           * S.float().unsqueeze(2)).reshape(Q.shape)
    dW = (deq - W.float())
    num = torch.linalg.norm(X @ dW.T)
    den = torch.linalg.norm(X @ W.float().T) + 1e-30
    return float(num / den)


def _rel_l2(W: torch.Tensor, Q: torch.Tensor, S: torch.Tensor, group_size: int) -> float:
    G = Q.shape[1] // group_size
    deq = (Q.float().reshape(Q.shape[0], G, group_size)
           * S.float().unsqueeze(2)).reshape(Q.shape)
    return float((deq - W.float()).norm() / W.float().norm())


def selftest() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, K, M = 256, 512, 8192
    GS, QMIN, QMAX = 128, -4, 3
    # 相关激活：AR(1) 特征（模拟真实校准的方向性，GPTQ 的收益来源）
    rho = 0.9
    Mmix = torch.zeros(K, K)
    for i in range(K):
        for j in range(K):
            Mmix[i, j] = rho ** abs(i - j)
    X = torch.randn(M, K, device=dev) @ Mmix.to(dev)
    W = (torch.randn(N, K, device=dev) * 0.02)

    H = hessian_add(None, X)
    qr, sr = rtn_quant(W.cpu(), GS, QMIN, QMAX)
    qg, sg = gptq_quant_fast(W.cpu(), H.cpu(), GS, QMIN, QMAX)
    qg, sg = qg.cpu(), sg.cpu()
    # 编码→解码字节级回环（与引擎消费端同路径）
    from tools.artifact.layouts import dequantize_row_split
    spec = get_format("Q3G128_F16S")
    enc = encode_row_split(qg.reshape(N, K // GS, GS), sg, spec, (N, K))
    deq = dequantize_row_split(enc, "Q3G128_F16S", (N, K), dtype=torch.float32)
    G = K // GS
    ref = (qg.float().reshape(N, G, GS) * sg.float().unsqueeze(2)).reshape(N, K)
    enc_err = float((deq.cpu() - ref.cpu()).norm() / ref.norm())
    assert enc_err == 0.0, f"编码回环不符: {enc_err}"

    Xc = X.cpu()
    e_rtn = _out_err(W.cpu(), qr, sr, Xc, GS)
    e_gptq = _out_err(W.cpu(), qg, sg, Xc, GS)
    w_rtn = _rel_l2(W.cpu(), qr, sr, GS)
    w_gptq = _rel_l2(W.cpu(), qg, sg, GS)
    print(f"[selftest] 输出空间: RTN={e_rtn:.4f} GPTQ={e_gptq:.4f} (×{e_gptq/e_rtn:.3f})")
    print(f"[selftest] 权重空间 rel L2: RTN={w_rtn:.4f} GPTQ={w_gptq:.4f}")
    assert e_gptq < e_rtn * 0.85, "GPTQ 输出误差未显著优于 RTN"
    print("[selftest] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
