#!/usr/bin/env bash
# 分层剥离耗时矩阵：用 LINGXI_SKIP 逐块关掉子块，反推每块的净耗时。
# 一律带上 `ple`——PLE 每 token 会从 95 GiB 大表按 ngram 哈希随机取行（表在机械盘上），
# 跳过子块后输出是乱码 token，会把"冷随机读"混进计时里。
# 用法（测试机 内）：bash tools/golden/skip_matrix.sh [max_new]
set -u
cd /models/lingxi
MAX=${1:-24}
export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q4g64.bin
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-7000}
export LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer

for cfg in "ple:ALL-ple" "ple,moe:GR+attn" "ple,attn:GR+moe" "ple,attn,moe:GR"; do
  sk=${cfg%%:*}; tag=${cfg##*:}
  export LINGXI_SKIP=$sk
  printf '=== SKIP=%-14s (%s) ===\n' "$sk" "$tag"
  ./bin/lingxi-bench talk "$ART" "The capital of France is" "$MAX" 2>&1 \
    | tr '\r' '\n' | sed -n 's/.*\(prefill 用时[^ ]*s，首 token = [0-9]*\).*/\1/p; s/.*\(decode 2[02]\/[0-9]*  token=[0-9]* *[0-9.]*s\/token\).*/\1/p'
done
echo "SKIP_MATRIX_DONE"
