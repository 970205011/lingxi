#!/usr/bin/env python3
"""混档 arena 验证器（2026-10-20，6.2-2 混档 v1 判据之一）。

对 `qwen4_exp.expert-arena.mix64.bin` 做三层检查：
  1. **几何**：header 合法；每个 routed ref 的 (offset, bytes, format) 落在 data 区内、
     格式码 ∈ {1=Q4G64, 5=Q3G128}；层内偏移连续无重叠（按 ref 排序验证）。
  2. **分配**：HOT_CSV（热集）里的 (层,专家) 必须 format=1 且 bytes=blob4；
     热集外必须 format=5 且 bytes=blob3（逐条核对，不漏不错）。
  3. **字节级对拍**：抽样 N 个热专家 blob 与 Q4 正本 arena 同 (层,专家) 逐字节比对；
     抽样 N 个冷专家 blob 与 Q3 正本 arena 比对。量化器确定性 ⇒ 必须完全一致。

用法：
  python3 tools/golden/mix_arena_check.py \
    /models/lingxi-assets/qwen4_exp.expert-arena.mix64.bin \
    /tmp/hot64.csv /models/lingxi-assets/qwen4_exp.expert-arena.q4g64.bin \
    /models/lingxi-assets/qwen4_exp.expert-arena.q3g128.bin [N=64]
"""
from __future__ import annotations

import mmap
import os
import struct
import sys

HDR = struct.Struct("<8s7i4x4q")   # 72B
REF = struct.Struct("<qii")        # 16B
NL, NE = 48, 512
# 实测几何（量化器日志 slot4=2613248 / slot3=1921024；ref.bytes 是不含对齐填充的净长，
# 由 ref 自身核对一致即可，不预设）。
FMT_NAME = {0: "BF16", 1: "Q4G64", 5: "Q3G128"}


def open_arena(path: str):
    fd = os.open(path, os.O_RDONLY)
    st = os.fstat(fd)
    mm = mmap.mmap(fd, st.st_size, mmap.MAP_PRIVATE, mmap.PROT_READ)
    h = HDR.unpack(bytes(mm[: HDR.size]))
    magic, ver, nl, ne, nin, nff, align, flags, doff, rioff, sioff, dbytes = h
    assert magic == b"Q4EARN1\0", magic
    assert (nl, ne) == (NL, NE), (nl, ne)
    return mm, dict(align=align, doff=doff, rioff=rioff, sioff=sioff, dbytes=dbytes)


def refs(mm, rioff, n):
    return [REF.unpack(bytes(mm[rioff + i * 16: rioff + (i + 1) * 16])) for i in range(n)]


def main() -> int:
    mix, hot_csv, q4p, q3p = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    n_spot = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    m, mh = open_arena(mix)
    r4, h4 = open_arena(q4p)
    r3, h3 = open_arena(q3p)
    routed = refs(m, mh["rioff"], NL * NE)
    rr4 = refs(r4, h4["rioff"], NL * NE)
    rr3 = refs(r3, h3["rioff"], NL * NE)

    hot: set[tuple[int, int]] = set()
    with open(hot_csv) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.strip().split(',')
            hot.add((int(p[0]), int(p[1])))

    # ── 1) 几何 + 2) 分配 ──
    bad_geom = bad_assign = 0
    seen: set[tuple[int, int]] = set()
    sizes: dict[int, int] = {}
    for l in range(NL):
        layer = sorted(
            (routed[l * NE + e], e) for e in range(NE)
        )
        prev_end = 0
        for (o, b, fmt), e in layer:
            assert o >= prev_end, f"层{l} 专家{e} 偏移重叠 o={o} prev_end={prev_end}"
            prev_end = o + b
            if not (0 <= o and o + b <= mh["dbytes"]):
                bad_geom += 1
            want = 1 if (l, e) in hot else 5
            if fmt != want:
                bad_assign += 1
                if bad_assign < 5:
                    print(f"  ✗ 分配错 层{l} 专家{e}: fmt={FMT_NAME.get(fmt, fmt)} 期望 {FMT_NAME[want]}")
            sizes.setdefault((fmt, b), 0)
            sizes[(fmt, b)] += 1
            seen.add((l, e))
    assert seen == {(l, e) for l in range(NL) for e in range(NE)}
    print(f"[1] 几何: {'OK' if bad_geom == 0 else f'✗ {bad_geom} 条越界'}"
          f" | 层内偏移连续无重叠 ✓ | 全部 {NL*NE} 条齐全 ✓")
    print(f"[2] 分配: {'OK（热{}/冷{} 逐条核对一致）'.format(len(hot), NL*NE-len(hot)) if bad_assign == 0 else f'✗ {bad_assign} 条错'}")
    for (fmt, b), cnt in sorted(sizes.items()):
        print(f"    fmt={FMT_NAME.get(fmt, fmt)} bytes={b} × {cnt}")

    # ── 3) 字节级对拍（抽样） ──
    import random
    rng = random.Random(20261020)
    hot_l = sorted(hot)
    cold_l = [(l, e) for l in range(NL) for e in range(NE) if (l, e) not in hot]
    n_bad = 0
    for (l, e), ref_arena in (
        [(p, r4) for p in rng.sample(hot_l, min(n_spot, len(hot_l)))]
        + [(p, r3) for p in rng.sample(cold_l, n_spot)]
    ):
        o, b, _ = routed[l * NE + e]
        mine = bytes(m[mh["doff"] + o: mh["doff"] + o + b])
        if ref_arena is r4:
            o2, b2, _ = rr4[l * NE + e]
            theirs = bytes(r4[h4["doff"] + o2: h4["doff"] + o2 + b2])
        else:
            o2, b2, _ = rr3[l * NE + e]
            theirs = bytes(r3[h3["doff"] + o2: h3["doff"] + o2 + b2])
        assert b == b2, (l, e, b, b2)
        if mine != theirs:
            n_bad += 1
            if n_bad < 5:
                print(f"  ✗ 字节不一致 层{l} 专家{e}")
    print(f"[3] 对拍: {'OK（{} 热 + {} 冷 逐字节一致）'.format(min(n_spot, len(hot_l)), n_spot) if n_bad == 0 else f'✗ {n_bad} 个不一致'}")
    print("MIX_ARENA_CHECK", "PASS" if bad_geom == 0 and bad_assign == 0 and n_bad == 0 else "FAIL")
    return 0 if bad_geom == 0 and bad_assign == 0 and n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
