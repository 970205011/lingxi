#!/usr/bin/env bash
# 可控 A/B：固定 token 流 + 取后段 **p10**（比中位数稳，避开偶发停顿）。
# 用法： bash tools/golden/ab.sh <N> <tag> [env=val ...] -- <tag2> ...
set -u
cd /models/lingxi
N=${1:-128}; shift
export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q4g64.bin
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-7000} LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
export LINGXI_DENSE_Q4=1 LINGXI_DENSE_Q4_QSA=1
export LINGXI_FIXED_IDS="$(cat /tmp/fixed200.txt)"
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer
cur_tag=""; cur_env=()
flush() {
  [ -z "$cur_tag" ] && return
  env ${cur_env[@]+"${cur_env[@]}"} ./bin/lingxi-bench talk "$ART" "The capital of France is" "$N" > /tmp/ab_$cur_tag.log 2>&1
  tr '\r' '\n' < /tmp/ab_$cur_tag.log | grep -a 's/token' \
    | sed -E 's/.*[[:space:]]([0-9]+\.[0-9]+)s\/token.*/\1/' | tail -n $((N*3/4)) | sort -n \
    | awk -v t="$cur_tag" '{v[NR]=$1} END {if(NR>0){p=v[int(NR/10)+1]; m=v[int((NR+1)/2)]; printf "%-16s p10=%.4f (%.1f tok/s)  med=%.4f\n", t, p, 1/p, m}}'
  cur_tag=""; cur_env=()
}
for a in "$@"; do
  if [ "$a" = "--" ]; then flush; else
    if [ -z "$cur_tag" ]; then cur_tag="$a"; else cur_env+=("$a"); fi
  fi
done
flush
