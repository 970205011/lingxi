#!/usr/bin/env bash
# 干净的端到端分解：`LINGXI_SKIP` 剥块只有在**路由被钉死**时才可信
# （否则跳块→输出乱码→路由乱跳→专家槽全失效→被剥的块反而更慢，见 警告）。
# 所以每一行都叠加 `LINGXI_ARENA_FIXED_EXPERTS=1`（路由钉死、槽 100% 命中，
# kernel 数/launch 数/缓冲/同步点与正常路径完全一致，只是数学作废）。
# `LINGXI_ROUTER_NOSYNC=1` 单独用来量 router 同步点（host 往返）的净代价。
# 用法（容器内）：bash tools/golden/diag_probe.sh [max_new]
set -u
cd "$(dirname "$0")/../.."
N=${1:-16}
export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q4g64.bin
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-7000}
export LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
export LINGXI_DENSE_Q4=1 LINGXI_DENSE_Q4_QSA=1
export LINGXI_FIXED_IDS="$(cat /tmp/fixed200.txt)"
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer

run() {
  local tag="$1"; shift
  local vals mn md
  vals=$(env "$@" ./bin/lingxi-bench talk "$ART" "The capital of France is" "$N" 2>&1 \
    | tr '\r' '\n' | grep -a 's/token' | tail -n $((N/2)) \
    | sed -E 's/.*[[:space:]]([0-9.]+)s\/token.*/\1/' | sort -n)
  mn=$(echo "$vals" | head -1)
  md=$(echo "$vals" | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}')
  printf '%-46s min=%-8s med=%s\n' "$tag" "$mn" "$md"
}

run "基线(固定ids)"                  LINGXI_SKIP=
run "固定专家(槽100%命中)"           LINGXI_SKIP= LINGXI_ARENA_FIXED_EXPERTS=1
run "固定专家+router去同步"          LINGXI_SKIP= LINGXI_ARENA_FIXED_EXPERTS=1 LINGXI_ROUTER_NOSYNC=1
run "固定专家+跳MoE"                 LINGXI_SKIP=ple,moe LINGXI_ARENA_FIXED_EXPERTS=1
run "固定专家+跳attn"                LINGXI_SKIP=ple,attn LINGXI_ARENA_FIXED_EXPERTS=1
run "固定专家+跳GR"                  LINGXI_SKIP=ple,gr LINGXI_ARENA_FIXED_EXPERTS=1
run "固定专家+跳MoE+GR+attn"         LINGXI_SKIP=ple,moe,gr,attn LINGXI_ARENA_FIXED_EXPERTS=1
echo DIAG_PROBE_DONE
