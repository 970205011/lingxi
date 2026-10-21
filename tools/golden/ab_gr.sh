#!/usr/bin/env bash
set -u
cd /models/lingxi
N=${1:-64}
export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q4g64.bin
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-7000} LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
export LINGXI_DENSE_Q4=1 LINGXI_DENSE_Q4_QSA=1
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer
run() {
  tag=$1; shift
  env "$@" ./bin/lingxi-bench talk "$ART" "The capital of France is" "$N" > /tmp/abgr_$tag.log 2>&1
  med=$(tr '\r' '\n' < /tmp/abgr_$tag.log | grep -a 's/token' \
    | sed -E 's/.*[[:space:]]([0-9]+\.[0-9]+)s\/token.*/\1/' | tail -n $((N/2)) | sort -n \
    | awk '{v[NR]=$1} END {if(NR>0) printf "%.4f", v[int((NR+1)/2)]}')
  printf "%-16s %s s/token  (%.1f tok/s)\n" "$tag" "$med" "$(awk -v m="$med" 'BEGIN{if(m>0)printf "%.1f",1/m}')"
}
run base
run base2
run grfuse        LINGXI_GR_FUSE=1
run grfuse2       LINGXI_GR_FUSE=1
run grfuse_graph  LINGXI_GR_FUSE=1 LINGXI_GRAPHS=1
