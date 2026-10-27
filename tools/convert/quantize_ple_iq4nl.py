# P15-㊳：把 bf16 PLE 母表量化成官方 IQ4_NL 格式（26.8GiB）。
# 用法（在 141 内）：python3 tools/convert/quantize_ple_iq4nl.py [行数]
# 产物：/dev/shm/qwen4_exp.ple-table.iq4nl.bin（320001536 行 × 90 B）+ 空 sidecar .iq4nl
#
# 格式 = llama.cpp IQ4_NL（dequant 语义与 ninfer-dev ple_iq4nl_dequant_row 逐项对齐）：
#   每行 160 值 = 5 个 block × 32 值；block = 2B fp16 scale(d) + 16B packed nibbles(32 个 4bit 码)；
#   value = cb[code] * d；nibble j 低 4bit → 元素 j、高 4bit → 元素 j+16（split-half，非交错）。
# 量化器 = 每 block 迭代最近码 + 最小二乘重估 d（llama.cpp make_qx_quants 同思路，8 轮），
# numpy 全向量化（在 32 值维度上并行，跨 block 批处理）。
import numpy as np
import sys, os, time

SRC = "/models/lingxi-assets/qwen4_exp.ple-table.bin"
OUT = "/dev/shm/qwen4_exp.ple-table.iq4nl.bin"
ROWS = 320_001_536
ROW_ELEMS = 160
BLOCK = 32
NBLK = ROW_ELEMS // BLOCK # 5
CB = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
MID = (CB[:-1] + CB[1:]) / 2  # 15 个边界，cb 有序 ⇒ 最近码 = sum(x > mid)
NSTEP = 8
CHUNK = 262_144  # 行/chunk（= 1.31M block，每轮 ~ 0.4GB 中间量）

def quantize_chunk(x):
    """x: (n, 160) float32 → bytes (n, 90)。llama.cpp make_qx_quants 同思路：
    迭代 {最近码分配 → 逐 block 记最优(L,d) → 最小二乘重估 d}，全向量化。
    最近码 = searchsorted（cb 有序，side='left' 精确等于"大于几个边界"）。"""
    n = x.shape[0]
    xb = x.reshape(n * NBLK, BLOCK)
    nb = xb.shape[0]
    amax = np.abs(xb).max(axis=1)
    d = amax / 127.0  # llama.cpp rmax=127（码本最大幅值）
    d = np.where(d > 1e-30, d, np.float32(1e-30))
    SUB = 1_048_576  # 子块（block 数），searchsorted 无大中间量，可以放大
    best_err = np.full(nb, np.inf, dtype=np.float32)
    best_q = np.zeros((nb, BLOCK), dtype=np.uint8)
    best_d = np.ones(nb, dtype=np.float32)

    def nearest(vals):
        # vals (m,32) → 最近码 (m,32) ∈ [0,15]
        return np.searchsorted(MID, vals.ravel(), side="left").reshape(vals.shape).astype(np.uint8)

    for _ in range(NSTEP):
        for s in range(0, nb, SUB):
            e = min(s + SUB, nb)
            vals = xb[s:e] / d[s:e, None]  # (sub,32)
            q = nearest(vals)
            cbc = CB[q]
            r = xb[s:e] - d[s:e, None] * cbc
            err = np.abs(r).sum(axis=1)
            upd = err < best_err[s:e]
            best_err[s:e] = np.where(upd, err, best_err[s:e])
            best_q[s:e][upd] = q[upd]
            best_d[s:e] = np.where(upd, d[s:e], best_d[s:e])
            num = (xb[s:e] * cbc).sum(axis=1)
            den = (cbc * cbc).sum(axis=1)
            new_d = np.abs(num / np.where(den > 0, den, 1.0))
            d[s:e] = np.where(new_d > 1e-30, new_d, np.float32(1e-30))
    # fp16 舍入修正：scale 落 fp16 后再分配一次码，误差更优者胜出
    d16 = best_d.astype(np.float16)
    d_back = d16.astype(np.float32)
    for s in range(0, nb, SUB):
        e = min(s + SUB, nb)
        vals = xb[s:e] / d_back[s:e, None]
        q = nearest(vals)
        err = np.abs(xb[s:e] - d_back[s:e, None] * CB[q]).sum(axis=1)
        upd = err < best_err[s:e]
        best_err[s:e] = np.where(upd, err, best_err[s:e])
        best_q[s:e][upd] = q[upd]
        best_d[s:e] = np.where(upd, d_back[s:e], best_d[s:e])
    out = np.empty((nb, 18), dtype=np.uint8)
    lo = best_q[:, :16]
    hi = best_q[:, 16:]
    out[:, :2] = best_d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = lo | (hi << 4)
    rows = out.reshape(n, NBLK * 18)
    # 全零 block：d16=0 ⇒ 解码全 0 ✓
    zb = amax == 0
    if zb.any():
        rows.view(np.uint8).reshape(n, NBLK, 18)[zb.reshape(n, NBLK)] = 0
    return rows.tobytes()

def quantize_band(args):
    """[行带并行] 量化 [r0, r1) 行，pwrite 到 OUT 的对应偏移。每个进程独立 fd。"""
    r0, r1 = args
    import os
    t0 = time.time()
    fd_in = os.open(SRC, os.O_RDONLY)
    fd_out = os.open(OUT, os.O_WRONLY)
    done = r0
    while done < r1:
        c = min(CHUNK, r1 - done)
        raw = os.pread(fd_in, c * 320, done * 320)
        x = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        x = x.view(np.float32).reshape(c, ROW_ELEMS)
        os.pwrite(fd_out, quantize_chunk(x), done * 90)
        done += c
        if (done - r0) % (CHUNK * 16) == 0:
            el = time.time() - t0
            print(f"band {r0}: {done - r0}/{r1 - r0} 行, {el:.0f}s, ETA {(el/(done-r0))*(r1-done):.0f}s", flush=True)
    os.close(fd_in)
    os.close(fd_out)
    return r1 - r0

def main():
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else ROWS
    nrows = min(nrows, ROWS)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    # 预分配输出文件（避免并发写时互相截断）
    import os
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(fd, nrows * 90)
    os.close(fd)
    if workers <= 1:
        total = quantize_band((0, nrows))
    else:
        from multiprocessing import Pool
        band = (nrows + workers - 1) // workers
        bands = [(i * band, min((i + 1) * band, nrows)) for i in range(workers) if i * band < nrows]
        with Pool(workers) as p:
            total = sum(p.map(quantize_band, bands))
    open(OUT + ".iq4nl", "wb").close()
    print(f"quantized {total} rows in {time.time()-t0:.0f}s", flush=True)
    # 抽验 round-trip
    de_cb = CB
    def dequant_row(j):
        with open(OUT, "rb") as f:
            f.seek(j * 90); b = np.frombuffer(f.read(90), dtype=np.uint8).reshape(5, 18)
        d = b[:, :2].copy().reshape(-1).view(np.float16).astype(np.float32)
        qs = b[:, 2:]
        lo = qs & 0xF
        hi = qs >> 4
        q = np.concatenate([lo, hi], axis=1)  # (5,32)：前 16 = 元素 j，后 16 = 元素 j+16
        v = de_cb[q] * d[:, None]
        return v.reshape(160)
    def bf16_row(j):
        with open(SRC, "rb") as f:
            f.seek(j * 320)
        with open(SRC, "rb") as f:
            f.seek(j * 320); raw = f.read(320)
        h = np.frombuffer(raw, dtype=np.uint16)
        return (h.astype(np.uint32) << 16).view(np.float32)
    errs = []
    for j in [0, 1, 100, 12345, 2500012, 98765432, 320001535]:
        j = min(j, total - 1)  # 部分行数测试时钳到文件内
        a, b = bf16_row(j), dequant_row(j)
        errs.append(np.abs(a - b).sum() / (np.abs(a).sum() + 1e-9))
        print(f"row {j}: rel={errs[-1]:.4f}", flush=True)
    print("IQ4NL_DONE", total, f"max_rel={max(errs):.4f}", flush=True)

if __name__ == "__main__":
    main()
