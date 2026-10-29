#!/usr/bin/env python3
"""共享专家 GPTQ 校准 → dense Q4 缓存（2026-10-29，纯 numpy 版）。

产物：**与 `crate::q4::quantize_bf16_bits` 逐字节同构**的 .q4 缓存文件，文件名 =
FNV-1a（张量全名）（与 `upload_q4_named` 的寻址一致）⇒ 引擎侧 LINGXI_SHARED_Q4=1 +
LINGXI_Q4_CACHE=<dir> 直接加载，零引擎改动。

输入：
  - 权重：lingxi-bench export-tensor 导出的裸 bf16 LE 文件（--wdir/L{层}_{gu,dn}.bin，
    形状固定 gu=[2I,H]、dn=[H,I]，由 meta 的 hidden/inter 推）；
  - 校准 dump（LINGXI_MOE_DUMP_ACT 产物，专家 GPTQ 同一份）：xpair = MoE 入口 h
    （每 token 按 topk 重复），pr = pair→行 ⇒ 去重得**每 token 唯一 h**。

数学（对齐 tools/arena-gen/gptq.py::gptq_quant_fast，换码制）：
  - 码制 = dense Q4 通道的 W4G64：qmin=-7, qmax=7，组 64 列，组 scale =
    f16(组首 running 权重逐行 amax / 7)（组内共用，与 quantize_bf16_bits 的 scale
    语义一致；码从 running 权重算 = GPTQ 传播）；
  - 误差按 H⁻¹ 比例摊到未量化列（sub_block=32 子块内逐列 + 跨子块/组整块传播）；
  - Hessian：gu 用去重 token 的 H = Σ h hᵀ（shared 看全部 token，不按路由分桶）；
    dn 用 act = swiglu(gu·h)（bf16 舍入，对齐引擎）后 Σ act actᵀ。

判据（脚本自验）：
  - 每矩阵报 rel_err = ‖W−Q·S‖/‖W‖（GPTQ vs 同格式 RTN，必须更优）；
  - 落盘前用纯 numpy 解码器回读 payload，对 codes/scales 逐位核对。

用法：
  python3 tools/arena-gen/gptq_shared.py \
      --dump <校准 dump 目录> --wdir <export-tensor 权重目录> \
      --hidden 2560 --inter 640 --out <LINGXI_Q4_CACHE 目录>
"""
import argparse
import json
import os
import re
import sys

import numpy as np

GROUP = 64
QMIN, QMAX = -7, 7
K_ALIGN = 128
PLANE_ALIGN = 256


def fnv1a(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def bf16_bits_to_f32(bits_u16: np.ndarray) -> np.ndarray:
    return (bits_u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """f32 → bf16（round-to-nearest-even），返回 u16 位型。"""
    u = x.astype(np.float32).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return (r >> 16).astype(np.uint16)


def f16_round(x: np.ndarray) -> np.ndarray:
    """f32 → f16 位型（u16），RNE（numpy float16 转换，非规格数/舍入全对）。"""
    return x.astype(np.float32).astype(np.float16).view(np.uint16)


def f16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """f16 位型（u16）→ f32 值（f16 布局 ≠ f32 高半，必须经 np.float16 视图转换）。"""
    return u16.view(np.float16).astype(np.float32)


def pack_payload(codes: np.ndarray, scales_u16: np.ndarray) -> tuple[bytes, int, int]:
    """codes [N,K] i8（∈[-7,7]）、scales_u16 [N,G] f16 位型 → quantize_bf16_bits 布局。"""
    n, k = codes.shape
    assert k % GROUP == 0
    k_pad = (k + K_ALIGN - 1) // K_ALIGN * K_ALIGN
    groups = k_pad // GROUP
    row_bytes = k_pad // 2
    scale_off = (n * row_bytes + PLANE_ALIGN - 1) // PLANE_ALIGN * PLANE_ALIGN
    out = np.zeros(scale_off + n * groups * 2, dtype=np.uint8)
    cd = codes.astype(np.int8)
    sc = scales_u16.astype(np.uint16)
    for r in range(n):
        for g in range(groups):
            c0, c1 = g * GROUP, min((g + 1) * GROUP, k)
            nib = (cd[r, c0:c1] & 0x0F).astype(np.uint8)
            pk = nib[0::2] | (nib[1::2] << 4)
            out[r * row_bytes + g * 32: r * row_bytes + g * 32 + 32] = pk
            so = scale_off + (r * groups + g) * 2
            out[so] = sc[r, g] & 0xFF
            out[so + 1] = sc[r, g] >> 8
    return out.tobytes(), k_pad, scale_off


def unpack_payload(payload: bytes, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    k_pad = (k + K_ALIGN - 1) // K_ALIGN * K_ALIGN
    groups = k_pad // GROUP
    row_bytes = k_pad // 2
    scale_off = (n * row_bytes + PLANE_ALIGN - 1) // PLANE_ALIGN * PLANE_ALIGN
    buf = np.frombuffer(payload, dtype=np.uint8)
    w16 = np.frombuffer(payload, dtype="<u2")  # 注意：uint8 << 8 会溢出回 uint8，scale 必须走 u16 视图
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, groups), dtype=np.uint16)
    for r in range(n):
        for g in range(groups):
            c0, c1 = g * GROUP, min((g + 1) * GROUP, k)
            pk = buf[r * row_bytes + g * 32: r * row_bytes + g * 32 + 32]
            nib = np.empty(c1 - c0, dtype=np.uint8)
            nib[0::2] = pk & 0x0F
            nib[1::2] = pk >> 4
            v = nib.astype(np.int8)
            v[v > 7] -= 16
            codes[r, c0:c1] = v
            so = scale_off + (r * groups + g) * 2
            scales[r, g] = w16[so // 2]
    return codes, scales


def gptq_w4g64(W: np.ndarray, H: np.ndarray, percdamp: float = 0.01,
               sub_block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """GPTQ（码制 W4G64 [-7,7]，组 scale=f16(amax/7)）。W [N,K]、H [K,K] f32。

    算法 = gptq.py::gptq_quant_fast 的 numpy 复刻（N 向量化、K 逐列推进）。
    """
    n, k = W.shape
    assert k % GROUP == 0
    Wf = W.astype(np.float32).copy()
    Hf = H.astype(np.float32)
    Hf = Hf + percdamp * np.mean(np.diag(Hf)) * np.eye(k, dtype=np.float32)
    L = np.linalg.cholesky(Hf)
    Linv = np.linalg.solve(L, np.eye(k, dtype=np.float32))
    Hinv = Linv.T @ Linv

    codes = np.empty((n, k), dtype=np.int8)
    scales = np.empty((n, k // GROUP), dtype=np.uint16)
    for g in range(k // GROUP):
        i0, i1 = g * GROUP, (g + 1) * GROUP
        blk = Wf[:, i0:i1]
        max_abs = np.abs(blk).max(axis=1)
        s = np.where(max_abs > 0, max_abs / QMAX, 1.0).astype(np.float32)
        s_u16 = f16_round(s)
        # scale 以 f16 位型为准（与 quantize_bf16_bits 一致：f16 舍入后再用）
        s_f16 = f16_bits_to_f32(s_u16)
        scales[:, g] = s_u16
        r = 1.0 / s_f16
        for j0 in range(i0, i1, sub_block):
            j1 = min(j0 + sub_block, i1)
            if j1 - j0 > 1 and sub_block < GROUP:
                for j in range(j0, j1 - 1):
                    col = Wf[:, j]
                    q = np.clip(np.round(col * r), QMIN, QMAX)
                    d = Hinv[j, j]
                    delta = (col - q * s_f16) / d
                    Wf[:, j + 1:j1] -= np.outer(delta, Hinv[j, j + 1:j1])
            blk2 = Wf[:, j0:j1]
            q2 = np.clip(np.round(blk2 * r[:, None]), QMIN, QMAX)
            codes[:, j0:j1] = q2.astype(np.int8)
            err = blk2 - q2 * s_f16[:, None]
            dblk = np.diag(Hinv[j0:j1, j0:j1])
            if j1 < k:
                Wf[:, j1:] -= err @ (Hinv[j0:j1, j1:] / dblk[:, None])
    return codes, scales


def rel_err(W: np.ndarray, codes: np.ndarray, scales_u16: np.ndarray) -> float:
    s = f16_bits_to_f32(scales_u16)
    deq = codes.astype(np.float32) * np.repeat(s, GROUP, axis=1)
    return float(np.linalg.norm(W - deq) / max(np.linalg.norm(W), 1e-12))


def out_err(W: np.ndarray, H: np.ndarray, codes: np.ndarray, scales_u16: np.ndarray) -> float:
    """输出空间误差 √(tr((W−QS) H (W−QS)ᵀ)) / √(tr(W H Wᵀ))——GPTQ 优化的真实目标。"""
    s = f16_bits_to_f32(scales_u16)
    deq = codes.astype(np.float32) * np.repeat(s, GROUP, axis=1)
    d = W - deq
    e = float(np.sqrt(np.sum(d * (d @ H))) / max(np.sqrt(np.sum(W * (W @ H))), 1e-12))
    return e


def rtn_w4g64(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, k = W.shape
    codes = np.empty((n, k), dtype=np.int8)
    scales = np.empty((n, k // GROUP), dtype=np.uint16)
    for g in range(k // GROUP):
        blk = W[:, g * GROUP:(g + 1) * GROUP].astype(np.float32)
        max_abs = np.abs(blk).max(axis=1)
        s = np.where(max_abs > 0, max_abs / QMAX, 1.0).astype(np.float32)
        s_u16 = f16_round(s)
        s_f16 = f16_bits_to_f32(s_u16)
        scales[:, g] = s_u16
        codes[:, g * GROUP:(g + 1) * GROUP] = np.clip(
            np.round(blk / s_f16[:, None]), QMIN, QMAX).astype(np.int8)
    return codes, scales


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--wdir", required=True, help="export-tensor 的裸 bf16 目录（L{层}_gu.bin / L{层}_dn.bin）")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--inter", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=48)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    H_, I = args.hidden, args.inter

    metas = sorted(f for f in os.listdir(args.dump) if re.fullmatch(r"moe\d+_meta\.json", f))
    by_layer: dict[int, list[str]] = {}
    for mn in metas:
        with open(os.path.join(args.dump, mn)) as f:
            by_layer.setdefault(json.load(f)["layer"], []).append(mn[:-len("_meta.json")])
    print(f"[shared-gptq] {len(by_layer)} 层有 dump", flush=True)

    def load_w(layer: int, role: str) -> np.ndarray:
        path = os.path.join(args.wdir, f"L{layer}_{role}.bin")
        bits = np.fromfile(path, dtype="<u2")
        # 产物布局：gu = [2I, H]、dn = [H, I]（行 = N，列 = K——与引擎 proj/GEMV 的 [N,K] 约定一致）
        cols = {"gu": H_, "dn": I}[role]
        rows = bits.size // cols
        assert bits.size == rows * cols, (path, bits.size, rows, cols)
        return bf16_bits_to_f32(bits.reshape(rows, cols))

    for layer in range(args.layers):
        if layer not in by_layer:
            print(f"[L{layer}] 无 dump，跳过", flush=True)
            continue
        gu = load_w(layer, "gu")  # [2I, H]
        dn = load_w(layer, "dn")  # [H, I]
        assert gu.shape == (2 * I, H_) and dn.shape == (H_, I), (layer, gu.shape, dn.shape)

        H_gu = np.zeros((H_, H_), dtype=np.float32)
        H_dn = np.zeros((I, I), dtype=np.float32)
        ntok = 0
        for base in by_layer[layer]:
            with open(os.path.join(args.dump, base + "_meta.json")) as f:
                m = json.load(f)
            s = m["s"]
            pr = np.fromfile(os.path.join(args.dump, base + "_pairs.pr.i32"), dtype=np.int32)
            assert pr.shape[0] == s
            xpair = np.fromfile(os.path.join(args.dump, base + "_xpair.bf16"),
                                dtype="<u2").reshape(s, m["hidden"])
            _, first_idx = np.unique(pr, return_index=True)
            h = bf16_bits_to_f32(xpair[first_idx])  # [tok, H] f32
            ntok += h.shape[0]
            H_gu += h.T @ h
            gate = h @ gu[:I].T
            up = h @ gu[I:].T
            act = f32_to_bf16_bits(gate / (1.0 + np.exp(-gate)) * up)  # swiglu → bf16 位型
            act = bf16_bits_to_f32(act)
            H_dn += act.T @ act

        codes_gu, sc_gu = gptq_w4g64(gu, H_gu)
        codes_dn, sc_dn = gptq_w4g64(dn, H_dn)
        rg, rd = rel_err(gu, codes_gu, sc_gu), rel_err(dn, codes_dn, sc_dn)
        cg, sg_ = rtn_w4g64(gu)
        cd_, sd_ = rtn_w4g64(dn)
        rg_rtn, rd_rtn = rel_err(gu, cg, sg_), rel_err(dn, cd_, sd_)
        og, od = out_err(gu, H_gu, codes_gu, sc_gu), out_err(dn, H_dn, codes_dn, sc_dn)
        og_rtn, od_rtn = out_err(gu, H_gu, cg, sg_), out_err(dn, H_dn, cd_, sd_)
        print(f"[L{layer}] tokens={ntok} gu w {rg:.5f}/{rg_rtn:.5f} out {og:.5f}/{og_rtn:.5f} | "
              f"dn w {rd:.5f}/{rd_rtn:.5f} out {od:.5f}/{od_rtn:.5f}  (gptq/rtn)", flush=True)
        # 判据 = 输出空间（GPTQ 的真实优化目标）；权重空间仅参考（GPTQ 刻意不保证）
        assert og <= og_rtn and od <= od_rtn, "GPTQ 输出空间误差必须 ≤ 同格式 RTN"

        for role, codes, scales, (n, k) in (
            ("gu", codes_gu, sc_gu, gu.shape),
            ("dn", codes_dn, sc_dn, dn.shape),
        ):
            payload, k_pad, scale_off = pack_payload(codes, scales)
            c2, s2 = unpack_payload(payload, n, k)
            assert np.array_equal(c2, codes), f"{role} codes 回读不一致"
            assert np.array_equal(s2, scales), f"{role} scales 回读不一致"
            tname = "shared_gate_up" if role == "gu" else "shared_down"
            name = f"text/layers/{layer}/moe/{tname}"
            path = os.path.join(args.out, f"n{fnv1a(name):016x}.q4")
            hdr = (n.to_bytes(8, "little") + k.to_bytes(8, "little")
                   + int(k_pad).to_bytes(4, "little", signed=True)
                   + scale_off.to_bytes(8, "little"))
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(hdr + payload)
            os.replace(tmp, path)
        print(f"[L{layer}] 已写 2 个缓存", flush=True)
    print("SHARED_Q4_GPTQ DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
