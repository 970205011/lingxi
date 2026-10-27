#!/usr/bin/env bash
# 投机（`mtp-accept`）A/B 计时 —— **草稿接受率/专家 DMA 诊断尺**。
#
# ⚠️ 分工（2026-10-22 划定，别再混）：
# 本脚本 = mtp-accept 诊断口径（接受率、逐位账、专家 DMA）——**不是生产吞吐**。
# 生产投机吞吐（serve/talk SpecRuntime 环路）→ **spec_bench.sh**。
#
# 为什么必须用它：`bench.sh` 走 `talk`，而投机环路只接在 `mtp-accept` 里；
# 且 `LINGXI_FWD_BENCH` 用重复 prompt token ⇒ 槽 100% 命中 ⇒ 天生看不到专家 DMA。
#
# 用法: bash tools/golden/spec_ab.sh <tag> <n> [kd] [env=val ...]
# kd = 0 表示**不开投机**（纯顺序对照）
# env 追加在最后，可覆盖任何默认（如 LINGXI_ARENA_PREWARM_N=145）
set -u
cd /models/lingxi
TAG=${1:?用法: spec_ab.sh <tag> <n> [kd] [env=val ...]}
N=${2:-200}
KD=${3:-5}
# 草稿概率闸门（㊿-⑦）：prob1 低于它就把链截断。实测桶接受率 ≥0.9→97%、<0.5→42%
# ⇒ 0.7 是"省行不丢接受"的档。kd 与它一起扫出来的最优约 kd=5 / 0.7（~64 tok/s）。
export LINGXI_SPEC_MIN_P=${LINGXI_SPEC_MIN_P:-0.7}
if [ $# -ge 3 ]; then shift 3; else shift $#; fi

export LINGXI_DEVICE=${LINGXI_DEVICE:-0}
# ㊿ 起默认 arena 是混档 mix64（q4g64 文件已不存在，旧默认会直接 bail）。
export LINGXI_EXPERT_ARENA=${LINGXI_EXPERT_ARENA:-/dev/shm/qwen4_exp.expert-arena.mix64.bin}
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-9000}   # ★㊲ 同 bench.sh
export LINGXI_LAYER_SPLIT_SLOTS=${LINGXI_LAYER_SPLIT_SLOTS:-9600}
export LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
export LINGXI_DENSE_Q4=${LINGXI_DENSE_Q4:-1}
export LINGXI_DENSE_Q4_QSA=${LINGXI_DENSE_Q4_QSA:-1}
export LINGXI_HEAD_Q4=${LINGXI_HEAD_Q4:-1}
export LINGXI_Q4_CACHE=${LINGXI_Q4_CACHE:-/dev/shm/lingxi-q4cache}
# 预热钉表（可选）：给一份历史 (层,专家,频次) CSV 就打开，纯 LRU 则留空。
# export LINGXI_ARENA_PREWARM=${LINGXI_ARENA_PREWARM:-<hist.csv>}
export LINGXI_ARENA_PREWARM_N=${LINGXI_ARENA_PREWARM_N:-200}
export LINGXI_NO_GRAPHS=${LINGXI_NO_GRAPHS:-1}

ART=${LINGXI_ARTIFACT:-/models/qwen4_exp-artifact/qwen4_exp.ninfer}
MTP=${LINGXI_MTP_ARTIFACT:-/models/lingxi-assets/qwen4_exp.mtp.fixed.ninfer}

# `PROMPT_FILE=<path>`：从文件读提示词 —— **长中文语料/多行内容不能走命令行引号**
#。
if [ -n "${PROMPT_FILE:-}" ] && [ -f "$PROMPT_FILE" ]; then
  PROMPT=$(cat "$PROMPT_FILE")
fi

OUT=/tmp/spec_${TAG}.log
if [ "$KD" = "0" ]; then
  # 顺序对照：mtp-accept 不带 LINGXI_MTP_SPEC ⇒ 只跑主干贪心
  env "$@" ./bin/lingxi-bench mtp-accept "$ART" "$MTP" "${PROMPT:-The capital of France is}" "$N" > "$OUT" 2>&1
  echo "== $TAG (顺序, 无投机) → $OUT"
  tr '\r' '\n' < "$OUT" | grep -aE 'tok/s|接受|µs|轮' | tail -6
else
  env "$@" LINGXI_MTP_SPEC="$KD" ./bin/lingxi-bench mtp-accept "$ART" "$MTP" "${PROMPT:-The capital of France is}" "$N" > "$OUT" 2>&1
  echo "== $TAG (kd=$KD) → $OUT"
  tr '\r' '\n' < "$OUT" | grep -aE 'tok/s|接受|轮|草稿|主干验证|重放|提交' | tail -14
fi
