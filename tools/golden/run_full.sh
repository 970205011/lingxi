#!/usr/bin/env bash
# 单流 decode 计时 + 专家槽命中率（CT141 内跑）。
# 用法：bash tools/golden/run_full.sh "<prompt>" [max_new]
set -u
cd /models/lingxi
PROMPT=${1:-"The capital of France is"}
MAX=${2:-64}
export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q4g64.bin
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-7000}
export LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
export LINGXI_ARENA_STATS=1
# 生产路径默认开三档 W4：−10.2% 且贪心序列逐位不变；载入多 ~25 s。
export LINGXI_DENSE_Q4=${LINGXI_DENSE_Q4:-1}
export LINGXI_DENSE_Q4_QSA=${LINGXI_DENSE_Q4_QSA:-1}
export LINGXI_HEAD_Q4=${LINGXI_HEAD_Q4:-1}
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer

./bin/lingxi-bench talk "$ART" "$PROMPT" "$MAX" 2>&1 \
  | tr '\r' '\n' \
  | grep -E "arena:|命中率|prefill|decode (1|8|16|24|32|40|48|56|64)/|生成" 
