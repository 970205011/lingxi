#!/usr/bin/env bash
# 干净剥块：全部跑在 LINGXI_ARENA_FIXED_EXPERTS=1（路由钉死、无 DMA、kernel 数不变）下，
# 这样 skip 不会污染路由。base 也在同一模式下，差值可信。
set -u
cd /models/lingxi
N=${1:-48}
export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q4g64.bin
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-7000}
export LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
export LINGXI_DENSE_Q4=1 LINGXI_DENSE_Q4_QSA=1 LINGXI_ARENA_FIXED_EXPERTS=1
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer
run() {
  tag=$1; shift
  env "$@" ./bin/lingxi-bench talk "$ART" "The capital of France is" "$N" > /tmp/ab2_$tag.log 2>&1
  med=$(tr '\r' '\n' < /tmp/ab2_$tag.log | grep -a 's/token' \
    | sed -E 's/.*[[:space:]]([0-9]+\.[0-9]+)s\/token.*/\1/' | tail -n $((N/2)) | sort -n \
    | awk '{v[NR]=$1} END {if(NR>0) printf "%.4f", v[int((NR+1)/2)]}')
  printf "%-16s %s s/token  (%.1f tok/s)\n" "$tag" "$med" "$(awk -v m="$med" 'BEGIN{if(m>0)printf "%.1f",1/m}')"
}
run fixed
run fixed_nogr    LINGXI_SKIP=gr
run fixed_noattn  LINGXI_SKIP=attn
run fixed_nomoe   LINGXI_SKIP=moe
run fixed_nople   LINGXI_SKIP=ple
