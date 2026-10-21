#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""产物 → HF 命名的张量：**逆用 ninfer 的 recipe**（`tools.convert.qwen4_exp.recipe`）。

不手抄任何映射：直接遍历 ninfer 的表达式树
  SourceTensor / Slice / Reshape / Transpose / Concat / Cast
用 `view_of(expr)[...] = data` 统一填充（Reshape/Transpose/Slice 都是 ndarray 视图），
`Concat` 按轴切分后递归。这样 ninfer 的 recipe 一旦变化，反解自动跟随。

另：`patch_artifact_wplus.py` 已就地把 148 个归一化 gamma 烘成 `(1+w)`，
HF 的 `Qwen4ExpTextRMSNorm` 期望 delta `w`，故这些张量反解时 **减 1**。

需要 PYTHONPATH 指向 ninfer-dev。
"""

from __future__ import annotations

import numpy as np
import torch

from tools.artifact.container import Artifact
from tools.artifact.layouts import decode_direct, dequantize_row_split
from tools.convert.qwen3_6.common.recipe import (
    Cast,
    Concat,
    Reshape,
    Slice,
    SourceTensor,
    Transpose,
    expression_shape,
)
from tools.convert.qwen4_exp import inventory as I

# PLE 表（95GiB）不是 artifact 对象：从 ninfer 导出的外部文件 mmap 读取
PLE_TABLE_FILE = "/data/models/lingxi-assets/qwen4_exp.ple-table.bin"
PLE_TABLE_ROW_WIDTH = 160


def wplus_targets() -> set[str]:
    """`patch_artifact_wplus.py` 烘焙过 (1+w) 的对象名集合（148 个）。"""
    t: list[str] = []
    for layer in I.TEXT_LAYERS:
        t += [f"text/layers/{layer}/attn_gr/hc_norm", f"text/layers/{layer}/mlp_gr/hc_norm"]
    t += ["text/hyper_connection_mixer/hc_norm"]
    for layer in I.FULL_ATTENTION_LAYERS:
        t += [
            f"text/layers/{layer}/attention/query_norm",
            f"text/layers/{layer}/attention/key_norm",
            f"text/layers/{layer}/attention/indexer/q_layernorm",
            f"text/layers/{layer}/attention/indexer/k_layernorm",
        ]
    t += ["text/ple/norm_key", "text/ple/norm_query", "text/ple/norm_conv"]
    return set(t)


WPLUS = wplus_targets()


class Defuser:
    """按 HF 源名累积缓冲区；逐 recipe 反解填充。"""

    def __init__(self, artifact_path: str):
        self.art = Artifact.open(artifact_path)
        self.buffers: dict[str, np.ndarray] = {}

    # ---- 读取产物 ----
    def decode(self, object_name: str) -> np.ndarray:
        obj = self.art.find(object_name)
        payload = self.art.payload(obj)
        if obj.layout == "contiguous-le-v1":
            t = decode_direct(payload, obj.format, obj.shape).to(torch.float32)
        elif obj.layout == "row-split-k128-v1":
            t = dequantize_row_split(
                payload, obj.format, obj.shape, device="cpu", dtype=torch.float32
            )
        else:
            raise SystemExit(f"未支持的布局: {obj.layout} ({object_name})")
        a = t.numpy()
        if object_name in WPLUS:
            a = a - np.float32(1.0)  # 还原 HF 的 delta 形式
        return a

    # ---- 表达式视图/填充 ----
    def buffer(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name not in self.buffers:
            self.buffers[name] = np.zeros(shape, dtype=np.float32)
        buf = self.buffers[name]
        assert tuple(buf.shape) == tuple(shape), (name, buf.shape, shape)
        return buf

    def view_of(self, e):
        if isinstance(e, SourceTensor):
            return self.buffer(e.name, e.shape)
        if isinstance(e, Reshape):
            return self.view_of(e.source).reshape(e.shape)
        if isinstance(e, Transpose):
            return np.transpose(self.view_of(e.source), e.axes)
        if isinstance(e, Slice):
            src = self.view_of(e.source)
            idx = [slice(None)] * src.ndim
            idx[e.axis] = slice(e.begin, e.end)
            return src[tuple(idx)]
        if isinstance(e, Cast):
            return self.view_of(e.source)
        raise SystemExit(f"未支持的表达式: {type(e).__name__}")

    def fill(self, e, data: np.ndarray) -> None:
        """把 data 写回 e 指向的缓冲区。

        关键：不能依赖 `view_of(e)[...] = data`——`Slice` 之后再 `Reshape` 会因非连续
        而返回**拷贝**，写入静默丢失（q_proj 交错的 q/gate 两段就踩了这个）。
        因此 `Reshape`/`Transpose`/`Cast` 一律先变换 data 再向**源**递归，
        只在 `SourceTensor` 上落地；`Slice` 也对源递归后写入视图。
        """
        if isinstance(e, Concat):
            offset = 0
            for part in e.sources:
                n = expression_shape(part)[e.axis]
                idx = [slice(None)] * data.ndim
                idx[e.axis] = slice(offset, offset + n)
                self.fill(part, data[tuple(idx)])
                offset += n
            return
        if isinstance(e, Cast):
            return self.fill(e.source, data)
        if isinstance(e, Reshape):
            return self.fill(e.source, data.reshape(expression_shape(e.source)))
        if isinstance(e, Transpose):
            return self.fill(e.source, np.transpose(data, e.axes))
        if isinstance(e, Slice):
            src = self.view_of(e.source)
            if not np.shares_memory(src, self.view_of(self._root(e.source))):
                raise SystemExit(f"Slice 源视图不是可写视图（{type(e.source).__name__}）")
            idx = [slice(None)] * src.ndim
            idx[e.axis] = slice(e.begin, e.end)
            src[tuple(idx)] = data
            return
        if isinstance(e, SourceTensor):
            self.buffer(e.name, e.shape)[...] = data
            return
        raise SystemExit(f"未支持的表达式: {type(e).__name__}")

    @staticmethod
    def _root(e):
        while not isinstance(e, SourceTensor):
            e = e.source
        return e

    def apply(self, recipe) -> None:
        data = self.decode(recipe.object_name)
        expected = tuple(expression_shape(recipe.expression))
        if tuple(data.shape) != expected:
            raise SystemExit(f"{recipe.object_name}: 产物 {data.shape} != recipe {expected}")
        self.fill(recipe.expression, data)

    def run(self, recipes) -> None:
        for r in recipes:
            self.apply(r)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(v) for k, v in self.buffers.items()}


def load_ple_table() -> torch.Tensor:
    """mmap 外部 PLE 表 → `[320001536, 160]` bf16 张量（只按需换页）。"""
    raw = np.memmap(PLE_TABLE_FILE, dtype=np.uint16, mode="r")
    rows = raw.size // PLE_TABLE_ROW_WIDTH
    assert raw.size % PLE_TABLE_ROW_WIDTH == 0, "PLE 表尺寸与行宽不符"
    t = torch.from_numpy(raw.reshape(rows, PLE_TABLE_ROW_WIDTH))
    return t.view(torch.bfloat16)
