#!/usr/bin/env bash
# ============================================================
# 生产 Q4 档投机/AR 性能尺（2026-10-22 固化，全日本机验证 57~61 tok/s）
# 用法：bash tools/golden/spec_perf_q4.sh [tokens] [R]
# 前置（容器 141 内）：
# cp /models/lingxi-assets/qwen4_exp.expert-arena.mix64.bin /dev/shm/
# （arena 必须在 tmpfs 上以启用整段 pin；放 sdb 会启动报错 PIN_GIB=0 且冷内容 1 tok/s 惨案）
# ============================================================
set -u
cd /models/lingxi
N=${1:-128}
R=${2:-2}

export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.mix64.bin
export LINGXI_ARENA_SLOTS=9600
export LINGXI_LAYER_SPLIT_SLOTS=10000
# 钉表需要历史 (层,专家,频次) CSV（本仓库不带）；有就取消下一行注释：
# export LINGXI_ARENA_PREWARM=<hist.csv>
export LINGXI_ARENA_PREWARM_N=200
export LINGXI_DENSE_Q4=1
export LINGXI_DENSE_Q4_QSA=1
export LINGXI_HEAD_Q4=1
export LINGXI_Q4_CACHE=/dev/shm/lingxi-q4cache

ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer

echo "══ AR 基线 200×3 ══"
for r in 1 2 3; do
  ./start.sh ./bin/lingxi-bench talk "$ART" "The capital of France is" 200 2>&1 \
    | grep -aoE "[0-9.]+ µs/token" | sed "s/ µs\/token//" | python3 -c "
import sys
xs=[float(l) for l in sys.stdin]
xs=xs[1:]; xs.sort(); n=len(xs)
p=lambda q: xs[min(n-1,int(n*q))]
print(f'  p10={p(0.1):.0f}us -> {1e6/p(0.1):.1f} tok/s')"
done

echo "══ spec France ${N}×${R} ══"
LINGXI_TALK_SPEC=1 bash tools/golden/spec_bench.sh q4_fr "$N" "$R" "The capital of France is"

echo "══ spec 代码 ${N}×${R}（miss 风暴对照）══"
printf "def quicksort(arr):\n    if len(arr) <= 1:" > /tmp/code_prompt.txt
LINGXI_TALK_SPEC=1 bash tools/golden/spec_bench.sh q4_code "$N" "$R" @/tmp/code_prompt.txt
