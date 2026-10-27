# P15-㉳：把官方 NVFP4 版模型自带的 PLE FP8(E4M3) 表从 safetensors 分片抽成平铺二进制。
# 用法（在 141 内）：python3 tools/convert/extract_ple_fp8.py
# 产物：/dev/shm/qwen4_exp.ple-table.fp8.bin（51.2GB = 320001536 行 × 160 B）+ .scale sidecar（全局 bf16 标量）。
# 行序 = shard_0..shard_127 顺序拼接（已抽验：与 bf16 母表 rel err ~2.3% = FP8 量化噪声）。
import json, struct, os, shutil
SRC = os.environ.get("PLE_FP8_SRC", "/models/nvfp4/Qwen3.8-Flash-Next-Uncensored-NVFP4")
OUT = "/dev/shm/qwen4_exp.ple-table.fp8.bin"
shards = {}
for fn in sorted(os.listdir(SRC)):
    if not fn.startswith("model-plefp8-") or not fn.endswith(".safetensors"): continue
    p = os.path.join(SRC, fn)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data_base = 8 + n
    for k, v in hdr.items():
        if k == "__metadata__" or ".shard_" not in k: continue
        sid = int(k.rsplit(".shard_", 1)[1].split(".")[0])
        shards[sid] = (p, data_base + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
assert len(shards) == 128 and set(shards) == set(range(128))
total = sum(s[2] for s in shards.values())
print("total bytes:", total, flush=True)
with open(OUT, "wb") as o:
    for sid in range(128):
        p, base, ln = shards[sid]
        with open(p, "rb") as f:
            f.seek(base)
            left = ln
            while left > 0:
                chunk = f.read(min(1 << 23, left))
                o.write(chunk)
                left -= len(chunk)
        print(f"shard {sid} done ({ln} B)", flush=True)
assert os.path.getsize(OUT) == total
# sidecar: weight_scale（全局 bf16 标量）
with open(OUT + ".scale", "wb") as f:
    f.write(struct.pack("<f", 0.00019931793212890625))
print("EXTRACT_OK", total)
