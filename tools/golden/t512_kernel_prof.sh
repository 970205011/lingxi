#!/usr/bin/env bash
# 靶子清单：t=512 单层逐核真 GPU 时间（零同步事件链，gr_prof 层末一次 sync 读回）
# 用法：bash tools/golden/t512_kernel_prof.sh [层数上限，默认 48]
set -u
cd /models/lingxi
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer
PROMPT="$(cat /tmp/parity_prompt.txt)"
N="${1:-48}"
env LINGXI_GR_PROF=1 LINGXI_GR_PROF_N="$N" LINGXI_ATTN_SYNC=1 LINGXI_MOE_PROF=1 \
    LINGXI_PREFILL_THREADS=0 ./start.sh ./bin/lingxi-bench talk "$ART" "$PROMPT" 1 \
    > /tmp/t512prof.log 2>&1
grep -E "prefill 用时|首 token" /tmp/t512prof.log | head -2
echo "== GR 逐核层均（[gr] 行）=="
grep "^\[gr\]" /tmp/t512prof.log | tail -3
echo "== attn 逐层样本 =="
grep -E "\[attn" /tmp/t512prof.log | tail -4
echo "== MoE 六事件样本 =="
grep -E "\[moe-prof|\[prof4|s4" /tmp/t512prof.log | tail -4
