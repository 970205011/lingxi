#!/usr/bin/env bash
# ============================================================
# 全 Q3 全驻留档性能尺（2026-10-22 实验固化；与 Q4 档对比用）
# 实测结论：速度与 Q4 持平——全驻留不是墙。
# 用法：bash tools/golden/spec_perf_q3full.sh [tokens] [R]
# 前置（容器 141 内）：
# cp /models/lingxi-assets/qwen4_exp.expert-arena.q3all.bin /dev/shm/
# 显存账：LINGXI_KV_K8V4 与本档互斥（接受率归零实证）；256K KV 时槽位 ~94% 驻留。
# ============================================================
set -u
cd /models/lingxi
N=${1:-128}
R=${2:-2}

export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q3all.bin
export LINGXI_ARENA_SLOTS=12097
export LINGXI_LAYER_SPLIT_SLOTS=13775
export LINGXI_DENSE_Q4=1
export LINGXI_DENSE_Q4_QSA=1
export LINGXI_HEAD_Q4=1
export LINGXI_Q4_CACHE=/dev/shm/lingxi-q4cache
# 无 PREWARM：全驻留档不需要钉表。

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
LINGXI_TALK_SPEC=1 bash tools/golden/spec_bench.sh q3_fr "$N" "$R" "The capital of France is"

echo "══ spec 代码 ${N}×${R} ══"
printf "def quicksort(arr):\n    if len(arr) <= 1:" > /tmp/code_prompt.txt
LINGXI_TALK_SPEC=1 bash tools/golden/spec_bench.sh q3_code "$N" "$R" @/tmp/code_prompt.txt
