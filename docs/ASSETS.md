# ASSETS.md — 模型与数据资产清单（单一事实源）

> 2026-10-19。本文件是全引擎**所有运行时资产的总账**：路径、大小、是否易失、来源、再生命令。
> 规则：**找资产先查本表；新资产入库必须登记本表**。`$MODELS` 为模型资产根目录，
> 所有路径两个名字等价。
> 机器：本机（R720XD，251GB RAM，2×V100 32G，/dev/shm 126G tmpfs，/data 5.5T）。
>
> ⚠️ **2026-10-21 起：持久资产正本全部在 `$LINGXI_ASSETS/`**。`$MODELS/tmp/`
> 是**自动清理区**（外部同步进程会周期性删除其中的 .niner 等文件，10-21 深夜曾一次性
> 清掉全部 MTP 产物）——**任何要保留的资产禁止放 tmp**，入库一律进 lingxi-assets。
> 表中出现的旧 `$MODELS/tmp` 路径（历史文档）均应读作 lingxi-assets。

## A. 主产物（模型本体）

| 资产 | 路径 | 大小 | 易失 | 说明 |
|---|---|---|---|---|
| 主产物 | `$MODELS/qwen4_exp-artifact/qwen4_exp.ninfer` | ~110G | 否 | NINFER v2 容器，lingxi 目录约定（FORMAT.md §1-2） |
| 主产物（wplus 前备份） | `同目录 qwen4_exp.ninfer.pre-wplus.bak` | — | 否 | 历史对照，勿删 |
| MTP 小产物（修好后） | `$LINGXI_ASSETS/qwen4_exp.mtp.fixed.ninfer` | 4.9G | 否 | hf2lingxi.py 从 safetensors 重生成（FORMAT.md §6） |
| MTP 大产物源 | `$LINGXI_ASSETS/qwen4_exp.q4dense.mtp2.ninfer` | 236G | 否 | 含 mtp/* 的抽取源；另有 247G/252G 历史副本在 tmp |
| 转换元数据 | `artifact/qwen4_exp.niner.conversion.json` 同目录 json | — | 否 | wplus 还原清单 |

## B. 专家权重（Q4G64 arena 链）

| 资产 | 路径 | 大小 | 易失 | 说明 |
|---|---|---|---|---|
| **Q4G64 arena（持久正本）** | `$LINGXI_ASSETS/qwen4_exp.expert-arena.q4g64.bin` | 60.25 GiB | 否 | Q4EARN1 格式（FORMAT.md §4）；索引修复 bak 同目录 |
| **混档 arena mix64 GPTQ（持久正本）** | `$LINGXI_ASSETS/qwen4_exp.expert-arena.mix64.gptq.bin` | 46.39 GiB | 否 | 混档 GPTQ 版：热 Q4 与 mix64 字节级一致 + 冷 Q3 GPTQ 补偿；QA 3/3 + 计时无差已验收，即插即用备选，非默认档 |
| 运行时副本 | `/dev/shm/qwen4_exp.expert-arena.q4g64.bin` | 同上 | **是（重启即失）** | 从持久正本拷贝；丢失后 `cp` 恢复 |
| **混档 arena mix64（6.2-2，持久正本）** | `$LINGXI_ASSETS/qwen4_exp.expert-arena.mix64.bin` | 46.39 GiB | 否 | 热 3072×Q4G64 + 冷 21504×Q3G128 逐槽格式码（hist top-64/层，`/tmp/hot64.csv` 可由 hist_spec.csv 重建）；`mix_arena_check.py` PASS；**Stage B（prefill 双视图）落地前不驻 shm、不作默认** |
| 混档运行时副本 | `/dev/shm/qwen4_exp.expert-arena.mix64.bin` | 同上 | **是（重启即失）** | Stage B 后 `cp` 上盘并改 start.sh 默认；与 FP8 PLE 合计 97.6G < 126G ✓ |
| arena 上游源 | `$LINGXI_ASSETS/qwen4_exp.expert-arena.bin` | 226G | 否 | bf16 专家池源（索引 .bak 同目录） |
| 构建工具 | ninfer-dev `tools/convert/common/quantize.py` 等 | — | — | 本仓只读解析；写入器在上游 |
| 校验工具 | `tools/golden/q4_arena_check.py` | — | — | arena vs BF16 逐元素对拍 |

## C. PLE 查找表（prefill 每个 token 都查，**必须在内存盘**）

| 资产 | 路径 | 大小 | 易失 | 说明 |
|---|---|---|---|---|
| **FP8 运行时表（官方档，当前默认）** | `/dev/shm/qwen4_exp.ple-table.fp8.bin` (+`.scale` sidecar) | 51.2G | **是** | E4M3，rel err 2.3%，10-17 验证过（f4f6d6d）；10-19 重抽恢复（EXTRACT_OK 51200245760，抽验 400 行 mean 2.26%），QA 回归 3/3 绿 |
| ~~IQ4_NL 运行时表~~ ❌未过判据 | `$LINGXI_ASSETS/qwen4_exp.ple-table.iq4nl.bin` (+`.iq4nl`) | 26.8G | 否（已挪出 shm） | **禁用**：pad4983 首 token 3 连跑稳定 eos（max_rel 8.94%，固有 4-bit 误差，见 PLAN_NEXT 事故记录）；勿放回 /dev/shm |
| IQ4_NL 再生 | `python3 tools/convert/quantize_ple_iq4nl.py`（~50min，8  workers） | — | — | 从 bf16 母表生成；**重生成≠可用，必须过 3 基线回归才准放回 shm** |
| **FP8 源（4-bit 上游）** | `$MODELS/nvfp4/Qwen3.8-Flash-Next-Uncensored-NVFP4/model-plefp8-*.safetensors` | （目录 126G） | 否 | 官方 NVFP4 发布分片；`tools/convert/extract_ple_fp8.py` 抽 51.2G 平铺表 |
| bf16 母表 | `$MODELS/qwen4_exp-artifact/qwen4_exp.ple-table.bin` | 96 GiB | 否 | 与 `$LINGXI_ASSETS/qwen4_exp.ple-table.bin` 硬链同份；兜底档（机械盘，慢 5×） |

## D. 运行参数/预热数据

| 资产 | 路径 | 说明 |
|---|---|---|
| 专家 hist 预热表 | `docs/bench-logs/hist_spec.csv`（源码树内） | 启动预热用（默认表）；另有 tmp/hist_bench.csv、hist_json.csv 实验版 |
| 分词器 | 产物内嵌（`tokenizer.json` 于 artifact 目录） | opts.tok 加载 |

## E. 上游参照（只读）

| 路径 | 说明 |
|---|---|
| `$MODELS/ninfer/` | ninfer-dev 检出（含 LINGXI_CHANGES.md）；arena/PLE 写入器与格式的出处 |
| `$MODELS/nvfp4/Qwen3.8-Flash-Next-Uncensored-NVFP4/` | **项目自产量化**（非下载的发布版；modelopt 0.46 + 自定义 calibration_contract）。用途：① FP8 PLE 源（extract_ple_fp8.py 原料）② 专家/张量量化分配的历史参考 ③ 校准契约复用。⚠ 勿当第三方证据引用 |

## 易失资产恢复手册（机器重启后）

```bash
# 1. shm 清空了 → 拷回 arena（61G，数分钟）
cp $LINGXI_ASSETS/qwen4_exp.expert-arena.q4g64.bin /dev/shm/
# 2. PLE 表 → 重抽官方 FP8 档（51.2G，验证过的档位）；急用可先 LINGXI_PLE_TABLE 指 bf16 母表顶住
python3 tools/convert/extract_ple_fp8.py
#   （勿用 quantize_ple_iq4nl.py 顶替：4-bit 档未过判据，见 C 节与 PLAN_NEXT 事故记录）
# 3. 验证：bash tests/regression/run.sh（3 基线 11751/332/220）
```

## 教训登记（为什么有本文件）

- 2026-10-19：PLE 内存表只在 dev 机 shm 里，随机器切换"丢失"，当天所有 benchmark 慢 5×
  才被迫重生成——**shm 上的每一份都要有持久正本或一条再生命令**。
- 同日上午按文件名 `*ple-table*` 找不到 4-bit 源，实际 FP8 源藏在 NVFP4 发布目录的
  safetensors 分片里（`extract_ple_fp8.py` 的 SRC 写明）——**找资产先查工具脚本的源路径**。
