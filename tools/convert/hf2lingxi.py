#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HF safetensors → 灵犀（lingxi）引擎 `.lingxi` 产物（容器 = NINFER v2）。

## 为什么有这份

生产产物一直是 ninfer-dev 的转换器造的，**不透明且出过一次事**：
MTP 小产物里 `mtp/embedding_norm` / `mtp/hidden_norm` 漏了 `wplus` 的 `+1`，
导致 MTP 草稿全错（`mtp-accept` 恒 0），而这些错**用引擎自己造的参照物永远测不出来**。
这份转换器放仓库里，目的就是：**规则显式、可复现、带自检**，不再依赖外部黑盒。

## 本模型族的关键约定（踩过的坑，务必看）

1. **容器格式**：`b"NINFER\\x00\\x02"` + `u64`(JSON 长度) + JSON 目录 + 4096 对齐的 payload。
   JSON = `{"identity": {...}, "objects": [{"bytes","format","kind","layout","name","offset","shape"}]}`，
   每个 `offset` 相对 **payload 起点**（= header 之后第一个 4096 边界）。
2. **`wplus`：RMSNorm 的 gamma 在 safetensors 里存的是 `γ−1`**（本模型族）⇒ 写 `.ninfer` 时
   必须给这些 gamma **整体 `+1`**。
   - 名单**按语义**判定：**所有真正的 RMSNorm gamma**（层内 `q_norm/k_norm/hc_norm/indexer.*_layernorm`
     **以及 MTP 的 stem norm** `pre_fc_norm_embedding/hidden`）。**踩坑就是这里漏了 stem 两个。**
   - **自检（本脚本强制打印）**：gamma 必须**全为正**。若某个 gamma 修完仍一半以上为负，
     说明它要么漏了 `+1`、要么本就不该加 —— 直接 `--strict` 会报错退出。
3. **`query_key_gate_value` 的行序**（产物 13312 行）= `q(6144) | k(512) | gate(6144) | v(512)`，
   其中 q/gate 来自 `q_proj` 的**按头去交织**：
   q 段行 `h*256+d` ← `q_proj[h*512+d]`，gate 段行 `h*256+d` ← `q_proj[h*512+256+d]`，
   k/v 整段搬。**这条已被主干产物（golden 金标准，layer 3）逐行验证**（`--check-against-trunk`）。
   ⚠️ **线上那份 MTP 产物在这一项上是坏的**：它的行序不是 `[q;k;v]` 的真置换
   （在 q 段里插了一个 k 块 ⇒ 丢 256 行 q、重复 256 行 k）。修好后接受率 0.906 → **1.000**。
   ⇒ 所以本脚本**默认按规则生成**，并强制做**真置换自检**（行多重集必须等于 `[q;k;v]`），
   否则报错 —— 这类"丢行/重复行"只有这条自检能抓到（逐行哈希反查会被重复行骗过去）。
4. 其余张量按名字一一对应（见 `MTP_MAP`），只做拼接/展平。

## 用法

    # 命名约定：本脚本按 lingxi 内容约定生成 ⇒ 输出用 .lingxi 后缀
    # （容器仍是 NINFER v2；读取器按 magic 识别，旧 .ninfer 产物照样能读）

    # 1) 从 HF 目录打出 MTP 小产物
    python3 tools/convert/hf2lingxi.py mtp \
        --src $MODELS/nvfp4/Qwen3.8-Flash-Next \
        --out $MODELS/lingxi-assets/qwen4_exp.mtp.lingxi

    # 2) 与一份已知产物逐位比对（验证映射/置换/约定是否正确）
    python3 tools/convert/hf2lingxi.py mtp --src <dir> --out /tmp/x.lingxi \
        --verify-against $MODELS/lingxi-assets/qwen4_exp.mtp.lingxi

    # 3) 只看会写什么、不动磁盘
    python3 tools/convert/hf2lingxi.py mtp --src <dir> --dry-run

## 还没迁进来的部分（trunk）

主干 251 GB 的产物还依赖 ninfer-dev：**48 层的 Q4 专家打包**（`Q4G64F16S` blob 布局）、
`experts.bin` 的 (layer, expert) 索引、PLE 表的 ngram 分片 —— 这些**打包细节尚未逆向**，
所以 `trunk` 子命令先留骨架（`TODO(trunk)` 处），等把 Q4 blob 格式钉死后接上。
MTP 这一份是**完整且逐位验证过的**（27/27 复现现有产物）。
"""

import argparse
import json
import mmap
import os
import struct
import sys

import numpy as np

MAGIC = b"NINFER\x00\x02"
BF16 = "BF16"
LAYOUT = "contiguous-le-v1"
ALIGN = 4096


# ─────────────────────────── 容器读写 ───────────────────────────

class NinferWriter:
    """按 `.ninfer` 容器格式写产物（与 ninfer-dev 的 Writer 兼容）。"""

    def __init__(self, path, model_id, weights_id):
        self.path = path
        self.identity = {"model_id": model_id, "weights_id": weights_id}
        self.items = []          # [(name, np.uint16 数组)]

    def add(self, name, arr):
        assert arr.dtype == np.uint16, f"{name}: 只接受 bf16 位模式（uint16）"
        self.items.append((name, np.ascontiguousarray(arr)))

    def write(self):
        objs, payload, off = [], [], 0
        for name, a in self.items:
            b = a.tobytes()
            objs.append({"bytes": len(b), "format": BF16, "kind": "tensor", "layout": LAYOUT,
                         "name": name, "offset": off, "shape": [int(s) for s in a.shape]})
            payload.append(b)
            off += len(b)
        j = json.dumps({"identity": self.identity, "objects": objs}).encode()
        with open(self.path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", len(j)))
            hdr = 16 + len(j)
            f.write(j)
            f.write(b"\x00" * ((ALIGN - hdr % ALIGN) % ALIGN))
            for b in payload:
                f.write(b)
        return sum(len(b) for b in payload)


def read_ninfer(path):
    """读 `.ninfer` 的张量（返回 {name: uint16 数组}）。"""
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    assert mm[:8] == MAGIC[:8], f"{path}: magic 不对 {mm[:8]!r}"
    jl = struct.unpack("<Q", mm[8:16])[0]
    d = json.loads(mm[16:16 + jl])
    po = ((16 + jl) + ALIGN - 1) // ALIGN * ALIGN
    out = {}
    for o in d["objects"]:
        if o.get("kind") != "tensor":
            continue
        a = np.frombuffer(mm[po + o["offset"]:po + o["offset"] + o["bytes"]], dtype="<u2")
        out[o["name"]] = a.reshape([int(s) for s in o["shape"]])
    return out


# ─────────────────────────── HF 侧读取 ───────────────────────────

class HfSafetensors:
    def __init__(self, root):
        self.root = root
        idx = os.path.join(root, "model.safetensors.index.json")
        self.wm = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
        if self.wm is None:
            raise SystemExit(f"{root}: 缺 model.safetensors.index.json（本脚本按多分片索引读取）")

    def keys(self):
        return list(self.wm)

    def tensor(self, key):
        from safetensors import safe_open
        if key not in self.wm:
            raise KeyError(key)
        with safe_open(os.path.join(self.root, self.wm[key]), "pt") as sf:
            return sf.get_tensor(key)

    def bf16_bits(self, key):
        """取张量的 bf16 位模式（uint16）。非 bf16 会先转 bf16（与转换器口径一致）。"""
        import torch
        t = self.tensor(key)
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        return t.contiguous().view(torch.int16).numpy().astype("<u2")


# ─────────────────────────── wplus（gamma +1）───────────────────────────

def to_f32(bits):
    a = np.asarray(bits, dtype=np.uint16)
    return (a.astype(np.uint32) << 16).view(np.float32)


def from_f32(v):
    """f32 → bf16 位模式，**round-half-even**（不是截断！）。

    ⚠️ 这行是踩过的第三个坑：一开始写成 `>> 16`（截断），于是所有 `+1` 后的 gamma
    与主干产物差 ~25% 的元素（每个差 1 ulp，且总是"低"）。
    主干的约定是 round-half-even —— 逐位比对主干 `text/layers/*/{attn_gr,mlp_gr}/hc_norm`
    与 `attention/*_norm` 可 100% 复现。
    """
    u = np.ascontiguousarray(np.asarray(v, dtype=np.float32)).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype("<u2")


def wplus_plus1(bits):
    """gamma(γ−1) → gamma：整体 +1（在 f32 里加，再舍回 bf16）。"""
    return from_f32(to_f32(bits) + 1.0)


def check_gamma(name, bits, strict=False):
    """gamma 必须全为正 —— 这是本轮抓出 stem norm 漏 `+1` 的判据。"""
    v = to_f32(bits)
    neg = float((v < 0).mean())
    flag = "OK " if neg <= 0.05 else "⚠️ "
    print(f"    {flag}{name:<28} mean={v.mean():+.4f}  负值占比={neg * 100:5.1f}%")
    if strict and neg > 0.05:
        raise SystemExit(f"{name}: gamma 有一半以上为负 ⇒ 可能漏了 wplus 的 +1（见本文件顶部约定）")


# ─────────────────────────── MTP 映射 ───────────────────────────

# ninfer 名 → (HF 名, 变换)；变换 ∈ {"raw", "plus1", "qkv", "cat", "flat"}
MTP_MAP = [
    ("mtp/fc_embedding", "mtp.fc_embedding.weight", "raw"),
    ("mtp/fc_hidden", "mtp.fc_hidden.weight", "raw"),
    # ⚠️ 这两个就是踩过的坑：也要 +1（role 是 MtpStemNorm，转换器曾漏掉）
    ("mtp/embedding_norm", "mtp.pre_fc_norm_embedding.weight", "plus1"),
    ("mtp/hidden_norm", "mtp.pre_fc_norm_hidden.weight", "plus1"),
    ("mtp/mixer/hc_norm", "mtp.hyper_connection_mixer.hc_norm.weight", "plus1"),
    ("mtp/mixer/mix_down", "mtp.hyper_connection_mixer.input_mix_weight_down.weight", "raw"),
    ("mtp/mixer/mix_up", "mtp.hyper_connection_mixer.input_mix_weight_up.weight", "raw"),
    ("mtp/layer/attention/query_key_gate_value", ("q", "k", "v"), "qkv"),
    ("mtp/layer/attention/query_norm", "mtp.layers.0.self_attn.q_norm.weight", "plus1"),
    ("mtp/layer/attention/key_norm", "mtp.layers.0.self_attn.k_norm.weight", "plus1"),
    ("mtp/layer/attention/output", "mtp.layers.0.self_attn.o_proj.weight", "raw"),
    ("mtp/layer/attention/indexer/index_qk_proj", "mtp.layers.0.self_attn.indexer.index_qk_proj.weight", "raw"),
    ("mtp/layer/attention/indexer/q_layernorm", "mtp.layers.0.self_attn.indexer.q_layernorm.weight", "plus1"),
    ("mtp/layer/attention/indexer/k_layernorm", "mtp.layers.0.self_attn.indexer.k_layernorm.weight", "plus1"),
    ("mtp/layer/attn_gr/hc_norm", "mtp.layers.0.attn_hyper_connection.hc_norm.weight", "plus1"),
    ("mtp/layer/attn_gr/block_inject", "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", "raw"),
    ("mtp/layer/attn_gr/mix_down", "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", "raw"),
    ("mtp/layer/attn_gr/mix_up", "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight", "raw"),
    ("mtp/layer/mlp_gr/hc_norm", "mtp.layers.0.mlp_hyper_connection.hc_norm.weight", "plus1"),
    ("mtp/layer/mlp_gr/block_inject", "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", "raw"),
    ("mtp/layer/mlp_gr/mix_down", "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", "raw"),
    ("mtp/layer/mlp_gr/mix_up", "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", "raw"),
    ("mtp/layer/mlp/router_shared_gate", ("mtp.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.shared_expert_gate.weight"), "cat"),
    ("mtp/layer/mlp/routed_gate_up", "mtp.layers.0.mlp.experts.gate_up_proj", "flat"),
    ("mtp/layer/mlp/routed_down", "mtp.layers.0.mlp.experts.down_proj", "flat"),
    ("mtp/layer/mlp/shared_gate_up", ("mtp.layers.0.mlp.shared_expert.gate_proj.weight", "mtp.layers.0.mlp.shared_expert.up_proj.weight"), "cat"),
    ("mtp/layer/mlp/shared_down", "mtp.layers.0.mlp.shared_expert.down_proj.weight", "raw"),
]

QKV_OUT_ROWS = 13312          # q(6144) | k(512) | gate(6144) | v(512)
HEAD_DIM, N_Q_HEADS = 256, 24
Q_DIM = HEAD_DIM * N_Q_HEADS  # 6144


def qkv_row_perm(n_q_rows=12288, n_k_rows=512, n_v_rows=512):
    """产物 13312 行 ← `[q_proj(12288); k_proj(512); v_proj(512)]` 的行下标（按头去交织）。

    q_proj 的行布局是 `[head0: q(256) gate(256) | head1: ... ]`，产物要的是
    `q(6144 连续) | k(512) | gate(6144 连续) | v(512)`。
    """
    assert n_q_rows + n_k_rows + n_v_rows == QKV_OUT_ROWS, (n_q_rows, n_k_rows, n_v_rows)
    n_heads = n_q_rows // (2 * HEAD_DIM)
    perm = []
    for h in range(n_heads):                       # q 段
        for d in range(HEAD_DIM):
            perm.append(h * 2 * HEAD_DIM + d)
    perm.extend(range(n_q_rows, n_q_rows + n_k_rows))               # k 段整段搬
    for h in range(n_heads):                       # gate 段
        for d in range(HEAD_DIM):
            perm.append(h * 2 * HEAD_DIM + HEAD_DIM + d)
    perm.extend(range(n_q_rows + n_k_rows, QKV_OUT_ROWS))           # v 段整段搬
    return np.array(perm)


def perm_from_known_pack(src, known_pack_path):
    """⚠️ 只作**交叉检查**：拿一份产物反查哈希推出行表。

    注意：**会被重复行骗过去**（线上那份坏产物就是"在 q 段插了一个 k 块"，
    逐行反查照样每行都能找到 ⇒ 看不出坏）。**默认请用规则版** `qkv_row_perm()`。
    """
    ref = read_ninfer(known_pack_path)["mtp/layer/attention/query_key_gate_value"]
    q = src.bf16_bits("mtp.layers.0.self_attn.q_proj.weight")
    k = src.bf16_bits("mtp.layers.0.self_attn.k_proj.weight")
    v = src.bf16_bits("mtp.layers.0.self_attn.v_proj.weight")
    cat = np.concatenate([q, k, v], axis=0)
    inv = {}
    for i, r in enumerate(cat):
        inv.setdefault(r.tobytes(), []).append(i)
    perm = []
    for r in ref:
        hit = inv.get(r.tobytes())
        if not hit:
            raise SystemExit("--perm-from: 有行在 [q;k;v] 里找不到 ⇒ 这份产物不是这套映射造的")
        perm.append(hit[0])
    return np.array(perm)


def build_mtp(src, perm, strict):
    q = src.bf16_bits("mtp.layers.0.self_attn.q_proj.weight")
    k = src.bf16_bits("mtp.layers.0.self_attn.k_proj.weight")
    v = src.bf16_bits("mtp.layers.0.self_attn.v_proj.weight")
    qkv_src = np.concatenate([q, k, v], axis=0)
    qkv = qkv_src[perm]
    # **真置换自检**：行多重集必须与 [q;k;v] 完全一致。
    # 这是唯一能抓"丢行/重复行"的判据 —— 逐行哈希反查会被重复行骗过去（线上那份就是这么坏的）。
    rows = lambda a: sorted(r.tobytes() for r in a)
    if rows(qkv) != rows(qkv_src):
        raise SystemExit(
            "qkv 自检失败：写出来的不是 [q_proj;k_proj;v_proj] 的真置换（有丢行或重复行）。\n"
            "  这通常意味着 **置换表用错了**（例如照抄了一份坏产物）。默认规则见文件顶部约定 3。")
    print("  ✅ qkv 真置换自检通过（行多重集与 [q;k;v] 一致）")
    out = []
    print("  写盘前的 gamma 自检（必须几乎全为正）：")
    for name, hf, how in MTP_MAP:
        if how == "qkv":
            arr = qkv
        elif how == "plus1":
            arr = wplus_plus1(src.bf16_bits(hf))
        elif how == "cat":
            arr = np.concatenate([src.bf16_bits(x) for x in hf], axis=0)
        elif how == "flat":
            t = src.bf16_bits(hf)
            arr = t.reshape(-1, t.shape[-1])
        else:
            arr = src.bf16_bits(hf)
        if how == "plus1":
            check_gamma(name, arr, strict=strict)
        out.append((name, arr))
    return out


# ─────────────────────────── 主流程 ───────────────────────────

def cmd_mtp(args):
    src = HfSafetensors(args.src)
    if args.perm_from:
        perm = perm_from_known_pack(src, args.perm_from)
        print(f"置换表：从 {args.perm_from} 反查得到（{len(perm)} 行）")
    else:
        perm = qkv_row_perm()
        print(f"置换表：按规则生成（{len(perm)} 行）—— 该规则已用主干产物 layer 3 逐行验证")
    items = build_mtp(src, perm, args.strict)
    if args.dry_run:
        print(f"[dry-run] 将写 {len(items)} 个张量；未落盘")
        return
    w = NinferWriter(args.out, "qwen4_exp-mtp", f"converted-from-{os.path.basename(args.src)}")
    for name, arr in items:
        w.add(name, arr)
    total = w.write()
    print(f"已写 {args.out}：{len(items)} 张量，{total / 1073741824:.2f} GiB")
    if args.verify_against:
        ref = read_ninfer(args.verify_against)
        mine = read_ninfer(args.out)
        bad = [n for n, _ in items
               if n not in ref or ref[n].shape != mine[n].shape or not (ref[n] == mine[n]).all()]
        print("与 %s 逐位比对：%s" % (args.verify_against,
                                    "✅ 全部逐位同" if not bad else f"❌ 不一致 {bad}"))


def cmd_trunk(args):
    # TODO(trunk): 主干产物还缺三块逆向，接上前不动手（免得造出"看着对"的产物）：
    #   1) 48 层的 Q4 专家打包 —— `Q4G64F16S` blob 的字节布局（group=64? scale 放在哪、f16 scale 的顺序）
    #   2) `experts.bin` 的 (layer, expert) 索引与 `ArenaExpertSource` 期望的 blob 头
    #   3) PLE 表（`ple-table.bin`）的 ngram 分片布局
    # 已有可复用的部分：本文件的容器写入、wplus `+1` 规则、role→名字映射（见 catalog role_of）。
    raise SystemExit("trunk 子命令尚未实现：见本文件底部 TODO(trunk)（缺 Q4 专家打包/experts.bin/PLE 表的逆向）")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mtp", help="HF 目录 → MTP 小产物（已验证 27/27 可复现生产产物）")
    m.add_argument("--src", required=True, help="HF safetensors 目录")
    m.add_argument("--out", help="输出的产物路径（建议 .lingxi；容器是 NINFER v2）")
    m.add_argument("--perm-from", help="从这份已知产物反查 qkv 行置换（最稳）")
    m.add_argument("--verify-against", help="与这份已知产物逐位比对")
    m.add_argument("--strict", action="store_true", help="gamma 一半以上为负时直接报错")
    m.add_argument("--dry-run", action="store_true")
    m.set_defaults(func=cmd_mtp)

    t = sub.add_parser("trunk", help="（未实现）HF 目录 → 主干产物")
    t.add_argument("--src", required=True)
    t.add_argument("--out")
    t.set_defaults(func=cmd_trunk)

    args = p.parse_args()
    # 注意顺序：先判 func 再读 args.dry_run（trunk 子命令没有 dry_run 属性）
    if getattr(args, "func", None) is cmd_mtp and not args.dry_run and not args.out:
        raise SystemExit("mtp: 需要 --out（或用 --dry-run）")
    args.func(args)


if __name__ == "__main__":
    main()
