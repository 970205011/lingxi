#!/usr/bin/env bash
# A/B：两条**可选**上下文开关对 decode 的影响（实测均无影响，默认关闭）。
set -u
cd /models/lingxi
export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q4g64.bin
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-7000}
export LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer
run() {
  local tag="$1"; shift
  printf '=== %s ===\n' "$tag"
  env "$@" ./bin/lingxi-bench talk "$ART" "The capital of France is" 24 2>&1 \
    | tr '\r' '\n' | grep -aE "decode (8|16|24)/" | sed 's/^ *//'
}
run "A 默认（cudarc 原语义）"
run "B LINGXI_DISABLE_EVENT_TRACKING=1" LINGXI_DISABLE_EVENT_TRACKING=1
run "C LINGXI_SCHED_SPIN=1" LINGXI_SCHED_SPIN=1
run "D 两者都开" LINGXI_DISABLE_EVENT_TRACKING=1 LINGXI_SCHED_SPIN=1
run "E 同步拷贝路由 logits" LINGXI_ROUTER_SYNCCPY=1
echo AB_DONE
