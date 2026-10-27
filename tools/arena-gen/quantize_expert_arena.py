#!/usr/bin/env python3
"""qwen4_exp 专家 arena 量化打包器（lingxi 收纳版，2026-10-19）。

从 /data14t/tmp/quantize_expert_arena.py 原样收纳并参数化：格式由 `ARENA_FMT` 环境
变量选择（默认 Q4G64_F16S；Q3G128_F16S = 3.125bpw、blob −26%，见 PLAN_NEXT ）。
编码数值与字节布局单一来源 = ninfer-dev `tools/convert/common/quantize.py::
quantize_and_encode`（3-bit 打包 = dense LSB-first 位流，128 值/48B，见上游 layouts.py
_pack_three_bit；GPU 参考反量化在上游 gsq_dequant_sm70.cu）。

格式（Q4EARN1，见 crates/lingxi-artifact/src/expert_arena.rs）：
  header(72B) | routed index [NL*NE * 16B ref] | shared index [NL * 16B]
  | pad 到 slot_align | data（每 blob slot_align 对齐）
routed blob = [encode(gate_up [1280,2560])][encode(down [2560,640])]；shared 保 BF16。

用法（本机）：
  ARENA_FMT=Q3G128_F16S PYTHONPATH=~/ninfer python3 \
    tools/arena-gen/quantize_expert_arena.py \
    /models/lingxi-assets/qwen4_exp.expert-arena.bin \
    /models/lingxi-assets/qwen4_exp.expert-arena.q3g128.bin
"""
from __future__ import annotations

import mmap
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("NINFER_REPO", os.path.expanduser("~/ninfer"))))

from tools.convert.common.quantize import quantize_and_encode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gptq  # noqa: E402  # 6.2-3：GPTQ one-shot 补偿（GPTQ_H Hessian 档启用）

# GPTQ 补偿（6.2-3）：GPTQ_H = hessians.pt，键 (layer, expert, 'gu'|'dn') → [K,K] = X^T X
# （校准流上该专家输入激活的二阶矩）。有 Hessian 的矩阵走 gptq_and_encode，其余回落 RTN。
GPTQ_H = os.environ.get("GPTQ_H", "")

FMT = os.environ.get("ARENA_FMT", "Q4G64_F16S")
assert FMT in ("Q4G64_F16S", "Q3G128_F16S", "MIXED"), FMT
FMT_CODE = {"Q4G64_F16S": 1, "Q3G128_F16S": 5}[FMT] if FMT != "MIXED" else None


def load_hot_set() -> set[tuple[int, int]]:
    """MIXED 模式的热集：HOT_CSV（layer,expert 两列，可带 count 第三列）里的 (l,e) → Q4，
    其余 → Q3。hist_spec.csv 直接可用（第三列 count 忽略，按文件行全收——调用方负责裁剪）。"""
    path = os.environ.get("HOT_CSV", "")
    assert path, "MIXED 模式必须给 HOT_CSV（热专家清单，layer,expert 两列）"
    hot: set[tuple[int, int]] = set()
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split(',')
            hot.add((int(parts[0]), int(parts[1])))
    return hot

NL, NE, H, FF = 48, 512, 2560, 640
GU_BF16 = 2 * FF * H * 2          # 6,553,600
DOWN_BF16 = H * FF * 2            # 3,276,800
BLOB_BF16 = GU_BF16 + DOWN_BF16   # 9,830,400

HDR = struct.Struct("<8s7i4x4q")  # 72B
REF = struct.Struct("<qii")       # 16B

K_BF16_RAW = 0


def read_header(mm: memoryview) -> dict:
    magic, version, nl, ne, nin, nff, align, flags, doff, rioff, sioff, dbytes = HDR.unpack(
        bytes(mm[: HDR.size])
    )
    assert magic == b"Q4EARN1\0", magic
    assert version == 1 and flags & 1, (version, flags)
    return dict(nl=nl, ne=ne, nin=nin, nff=nff, align=align, doff=doff,
                rioff=rioff, sioff=sioff, dbytes=dbytes)


def read_index(mm: memoryview, off: int, count: int) -> list[tuple[int, int, int]]:
    return [REF.unpack(bytes(mm[off + i * 16: off + (i + 1) * 16])) for i in range(count)]


def bf16_to_f32(buf: bytes) -> torch.Tensor:
    # int16 位型 view 成 bf16 再上传（位级转换留给 GPU，quantize_and_encode 内部完成）。
    return torch.frombuffer(bytearray(buf), dtype=torch.int16).view(torch.bfloat16)


def q_payload_bytes(fmt: str | None = None) -> tuple[int, int, int]:
    """从编码器本身推 routed blob 几何（gu 段 / down 段 / 合计），不自造数字。"""
    fmt = fmt or (FMT if FMT != "MIXED" else "Q4G64_F16S")
    gu = len(quantize_and_encode(torch.zeros(2 * FF, H), fmt, device="cpu"))
    dn = len(quantize_and_encode(torch.zeros(H, FF), fmt, device="cpu"))
    return gu, dn, gu + dn


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pack-q{FMT[1]}] fmt={FMT} code={FMT_CODE}\n[pack-q] src={src}\n[pack-q] dst={dst}"
          f"\n[pack-q] device={dev}", flush=True)

    hessians: dict = {}
    hess_dir = GPTQ_H if GPTQ_H and os.path.isdir(GPTQ_H) else ""
    if GPTQ_H and not hess_dir:
        hessians = torch.load(GPTQ_H, map_location="cpu", weights_only=True)
        print(f"[pack-q] GPTQ 补偿开：{GPTQ_H}（{len(hessians)} 个 Hessian）", flush=True)
    elif hess_dir:
        print(f"[pack-q] GPTQ 补偿开（逐层懒加载）：{hess_dir}", flush=True)

    fd = os.open(src, os.O_RDONLY)
    st = os.fstat(fd)
    imm = mmap.mmap(fd, st.st_size, mmap.MAP_PRIVATE, mmap.PROT_READ)
    mm = memoryview(imm)
    hdr = read_header(mm)
    assert (hdr["nl"], hdr["ne"], hdr["nin"], hdr["nff"]) == (NL, NE, H, FF), hdr
    align = hdr["align"]
    routed = read_index(mm, hdr["rioff"], NL * NE)
    shared = read_index(mm, hdr["sioff"], NL)
    assert all(o >= 0 for o, _, _ in routed), "routed index 有缺席条目，需先修复 arena"
    for l in range(NL):
        for e in range(NE):
            o, b, f = routed[l * NE + e]
            assert b == BLOB_BF16 and f == K_BF16_RAW, (l, e, b, f)

    gu_q, dn_q, blob_q = q_payload_bytes()
    slot_q = (blob_q + align - 1) // align * align
    # MIXED：双几何并存（热=Q4、冷=Q3），逐专家偏移/字节数写进 ref（ref 本就逐槽 16B 携带）。
    if FMT == "MIXED":
        hot = load_hot_set()
        gu4, dn4, blob4 = gu_q, dn_q, blob_q                      # Q4G64
        gu3, dn3, blob3 = q_payload_bytes("Q3G128_F16S")
        slot4 = (blob4 + align - 1) // align * align
        slot3 = (blob3 + align - 1) // align * align
        fmt_of = [("Q4G64_F16S" if (l, e) in hot else "Q3G128_F16S")
                  for l in range(NL) for e in range(NE)]
        slot_of = [slot4 if f == "Q4G64_F16S" else slot3 for f in fmt_of]
        layer_q_bytes = [sum(slot_of[l * NE:(l + 1) * NE]) for l in range(NL)]
        n_hot = sum(1 for f in fmt_of if f == "Q4G64_F16S")
        print(f"[pack-q] MIXED: hot(Q4)={n_hot} cold(Q3)={NL*NE-n_hot} | "
              f"slot4={slot4} slot3={slot3}", flush=True)
    else:
        blob4 = blob3 = slot4 = slot3 = None
        fmt_of = [FMT] * (NL * NE)
        layer_q_bytes = [NE * slot_q] * NL
        print(f"[pack-q] per-expert: bf16 {BLOB_BF16} -> {blob_q} "
              f"(gu {gu_q} + down {dn_q}), slot {slot_q}", flush=True)

    # ---- 输出布局 ----
    hdr_bytes = HDR.size
    routed_off = hdr_bytes
    shared_off = routed_off + REF.size * NL * NE
    data_off = (shared_off + REF.size * NL + align - 1) // align * align
    shared_part = FF * H * 2
    shared_slot = (3 * shared_part + align - 1) // align * align
    data_total = sum(layer_q_bytes)
    out_size = data_off + data_total + NL * shared_slot
    print(f"[pack-q] out size = {out_size / 2**30:.2f} GiB "
          f"(bf16 was {st.st_size / 2**30:.2f} GiB)", flush=True)

    out = os.open(dst, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(out, out_size)
    omm = mmap.mmap(out, out_size)

    omm[:HDR.size] = HDR.pack(b"Q4EARN1\0", 1, NL, NE, H, FF, align, 1,
                              data_off, routed_off, shared_off, out_size - data_off)
    sdata_base = data_off + data_total
    code_of = {"Q4G64_F16S": 1, "Q3G128_F16S": 5}
    # 整层一次顺序读（层内 min..max offset 跨度），层内偏移以 index 为准。
    # GPTQ 批量档（hess_dir）：hessian 逐层懒加载 + 后台线程预取下一层（单层 12.8GB，
    # 冷盘 ~190s，与计算流水重叠）；有 H 的专家按 GPTQ_BATCH（默认 16）批量量化
    # （数学与逐专家版逐项相同，固定开销摊 B 份：单矩阵 950ms → 批量 ~40ms/矩阵），
    # 缺 H 的回落 RTN。
    import threading
    hess_pref: dict = {"d": None, "t": None}

    def hess_load(l: int) -> None:
        # 直接解析到 **GPU**：CPU 峰值 ~0（血泪：CPU 版两层 13+13GB + W 副本 5GB +
        # span 4.7GB = 35.5GB 两次全局 OOM，dmesg oom-kill 实证）。GPU 驻留两层
        # 26GB（32G 卡）+ 计算工作集 ~2GB；prefetch 线程与当前层计算流水重叠（盘 66MB/s
        # 地板 ~190s/层被隐藏）。层尾 del 当前层字典（GPU 显存即刻归还）。
        hp = os.path.join(hess_dir, f"hessians_L{l:02d}.pt")
        hess_pref["d"] = torch.load(hp, map_location=dev, weights_only=True) \
            if os.path.exists(hp) else {}

    def hess_take() -> dict:
        if hess_pref["t"] is not None:
            hess_pref["t"].join()
            hess_pref["t"] = None
        d = hess_pref["d"]
        hess_pref["d"] = None
        return d or {}

    if hess_dir:
        hess_pref["t"] = threading.Thread(target=hess_load, args=(0,), daemon=True)
        hess_pref["t"].start()
    t0 = time.time()
    l_base = 0
    for l in range(NL):
        if hess_dir:
            hessians = hess_take()
            if l + 1 < NL:
                hess_pref["t"] = threading.Thread(target=hess_load, args=(l + 1,), daemon=True)
                hess_pref["t"].start()
        layer_blob = bytearray(layer_q_bytes[l])
        offs = [routed[l * NE + e][0] for e in range(NE)]
        span_lo, span_hi = min(offs), max(offs) + BLOB_BF16
        layer_raw = bytes(mm[hdr["doff"] + span_lo: hdr["doff"] + span_hi])
        # 预裁好每专家的 bf16 权重字节段，避免重复切片
        Wgu_of: list[torch.Tensor] = []
        Wdn_of: list[torch.Tensor] = []
        for e in range(NE):
            eb = offs[e] - span_lo
            Wgu_of.append(bf16_to_f32(layer_raw[eb: eb + GU_BF16]).reshape(2 * FF, H))
            Wdn_of.append(bf16_to_f32(layer_raw[eb + GU_BF16: eb + BLOB_BF16]).reshape(H, FF))

        # GPTQ 批量档：本层有 (gu&dn) Hessian 且同格式的专家，按批量化
        gbatch = int(os.environ.get("GPTQ_BATCH", "16"))
        gptq_done: set[int] = set()
        layer_blob_gu: dict[int, bytes] = {}
        layer_blob_dn: dict[int, bytes] = {}
        if hess_dir and hessians:
            from tools.artifact.numeric import get_format
            from tools.artifact.layouts import encode_row_split
            # GPTQ_ONLY_FMT：只对指定格式段做 GPTQ（混档主路径 = 冷 Q3 补偿、热 Q4 原样，
            # 保持已验证基线不动；Q4+GPTQ 留作后续实验）。
            only = os.environ.get("GPTQ_ONLY_FMT", "")
            es_h = [e for e in range(NE)
                    if (l, e, "gu") in hessians and (l, e, "dn") in hessians
                    and (not only or fmt_of[l * NE + e] == only)]
            # 按格式分段（批量内必须同格式）
            seg: list[list[int]] = []
            for e in es_h:
                ef_e = fmt_of[l * NE + e]
                if seg and fmt_of[l * NE + seg[-1][-1]] == ef_e:
                    seg[-1].append(e)
                else:
                    seg.append([e])
            for es in seg:
                for i0 in range(0, len(es), gbatch):
                    esb = es[i0:i0 + gbatch]
                    ef_b = fmt_of[l * NE + esb[0]]
                    spec = get_format(ef_b)
                    for kind, Wof, dims in (
                            ("gu", Wgu_of, (2 * FF, H)),
                            ("dn", Wdn_of, (H, FF))):
                        Wb = torch.stack([Wof[e] for e in esb]).to(dev)
                        Hb = torch.stack([hessians[(l, e, kind)] for e in esb])  # 已驻 GPU
                        codes, scales = gptq.gptq_quant_fast_batched(
                            Wb, Hb, group_size=spec.group_size,
                            qmin=spec.qmin, qmax=spec.qmax)
                        codes, scales = codes.cpu(), scales.cpu()
                        Bn, Nn, Kn = codes.shape
                        for i, e in enumerate(esb):
                            enc = encode_row_split(
                                codes[i].reshape(Nn, Kn // spec.group_size, spec.group_size),
                                scales[i], spec, dims)
                            if kind == "gu":
                                layer_blob_gu[e] = enc
                            else:
                                layer_blob_dn[e] = enc
                    gptq_done.update(esb)

        loff = 0
        for e in range(NE):
            ef = fmt_of[l * NE + e]
            egu, edn, eblob = (gu_q, dn_q, blob_q) if ef == FMT else (
                (gu4, dn4, blob4) if ef == "Q4G64_F16S" else (gu3, dn3, blob3))
            eslot = slot_of[l * NE + e] if FMT == "MIXED" else slot_q
            if e in gptq_done:
                gu_enc, dn_enc = layer_blob_gu[e], layer_blob_dn[e]
            else:
                gu_enc = quantize_and_encode(Wgu_of[e], ef, device=dev)
                dn_enc = quantize_and_encode(Wdn_of[e], ef, device=dev)
            assert len(gu_enc) == egu and len(dn_enc) == edn, (l, e, ef)
            layer_blob[loff: loff + egu] = gu_enc
            layer_blob[loff + egu: loff + eblob] = dn_enc
            omm[routed_off + (l * NE + e) * 16:
                routed_off + (l * NE + e + 1) * 16] = REF.pack(
                    l_base + loff, eblob, code_of[ef] if FMT == "MIXED" else FMT_CODE)
            loff += eslot
        assert loff == layer_q_bytes[l], (l, loff, layer_q_bytes[l])
        omm[data_off + l_base: data_off + l_base + layer_q_bytes[l]] = layer_blob
        l_base += layer_q_bytes[l]
        # shared 原样拷贝（BF16，格式 0）
        so, sb, _ = shared[l]
        omm[sdata_base + l * shared_slot: sdata_base + l * shared_slot + sb] = \
            mm[hdr["doff"] + so: hdr["doff"] + so + sb]
        omm[shared_off + l * 16: shared_off + (l + 1) * 16] = \
            REF.pack(sdata_base - data_off + l * shared_slot, sb, K_BF16_RAW)
        el = time.time() - t0
        print(f"[pack-q] layer {l + 1}/{NL} done | {el:.0f}s | "
              f"ETA {el / (l + 1) * (NL - l - 1):.0f}s", flush=True)
        # 层尾显式释放：hessians 12.8GB + W 副本 ~5GB 必须在下一层解析（+12.8GB）前归还，
        # 否则峰值 ~38GB 超 available → 全局 OOM（2026-10-21 05:18 实证被杀）。
        # hessians 只在 hess_dir（逐层懒加载）模式下于循环内绑定；file 模式（line 109
        # 单次加载）要跨层保留，无 GPTQ 时从未绑定——三种情况都不能无条件 del
        # （2026-10-22 实证：无 GPTQ_H 跑到第 3 层 UnboundLocalError）。
        if hess_dir:
            del hessians
        del Wgu_of, Wdn_of, layer_blob_gu, layer_blob_dn
        import gc
        gc.collect()
    omm.flush()
    omm.close()
    del mm
    imm.close()
    os.close(fd)
    os.close(out)
    print("[pack-q] arena complete:", dst, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
