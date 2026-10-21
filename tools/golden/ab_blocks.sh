#!/usr/bin/env bash
# 逐块剥离 A/B（每项独立进程；数字仅供排序，skip 会改变数值语义）
set -u
cd "$(dirname "$0")/../.."
N=${1:-48}
export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q4g64.bin
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-7000}
export LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
export LINGXI_DENSE_Q4=1 LINGXI_DENSE_Q4_QSA=1
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer
run() { # tag env...
  tag=$1; shift
  env "$@" ./bin/lingxi-bench talk "$ART" "The capital of France is" "$N" > /tmp/ab_$tag.log 2>&1
  med=$(tr '\r' '\n' < /tmp/ab_$tag.log | grep -a 's/token' \
    | sed -E 's/.*[[:space:]]([0-9]+\.[0-9]+)s\/token.*/\1/' | tail -n $((N/2)) | sort -n \
    | awk '{v[NR]=$1} END {if(NR>0) printf "%.4f", v[int((NR+1)/2)]}')
  printf "%-14s %s s/token  (%.1f tok/s)\n" "$tag" "$med" "$(awk -v m="$med" 'BEGIN{if(m>0)printf "%.1f",1/m}')"
}
run base
run skip_ple   LINGXI_SKIP=ple
run skip_attn  LINGXI_SKIP=attn
run skip_gr    LINGXI_SKIP=gr
run nomoe_dma  LINGXI_ARENA_NOH2D=1
