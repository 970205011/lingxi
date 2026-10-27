#!/usr/bin/env bash
# prefill 剖面复测套件（，2026-10-27）——判决实验链的稳定启动指令。
#
# 用法（容器内权威口径）：
# bash tools/golden/prefill_profile.sh # 基线 + 同步探针 + 干净拷贝探针
# bash tools/golden/prefill_profile.sh kick # 追加跑 KICK 对照
# bash tools/golden/prefill_profile.sh bw # 追加跑 D2H_BW（默认已含）
#
# 判据：
# - 首 token = 332（4.2K parity prompt 逐位锚）
# - "[moe-sync-probe]" 的 host_spin = 每层路由同步等时（GPU-bound 下 ≈ 前序 GPU 工作）
# - "[d2h-bw]" 的 clean_copy 应 ≈0.1ms（满速）；若它也慢 ⇒ CE 全局问题，先修平台
# - prefill 用时基线 16~17s（软卡宿主方差 ±2s，看多次取众数）
set -u
cd /models/lingxi
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer
PROMPT="$(cat /tmp/parity_prompt.txt)"
[ -s /tmp/parity_prompt.txt ] || { echo "缺 /tmp/parity_prompt.txt（4167 token parity prompt）"; exit 1; }

run() {
    local tag="$1"; shift
    env LINGXI_PROF=1 LINGXI_PREFILL_THREADS=0 LINGXI_D2H_BW=1 "$@" \
        ./start.sh ./bin/lingxi-bench talk "$ART" "$PROMPT" 1 > "/tmp/pp_$tag.log" 2>&1
    grep -E "prefill 用时|首 token" "/tmp/pp_$tag.log" | head -2
    grep "moe-sync-probe" "/tmp/pp_$tag.log" | grep rows512 | \
        awk '{for(i=1;i<=NF;i++){if($i~/^host_spin=/)h+=substr($i,11);if($i~/^copy=/)c+=substr($i,6);if($i~/^sync=/)s+=substr($i,6)}} END{printf "  rows512 n=%d copy=%.3f sync=%.2f spin=%.2f ms(avg)\n",NR,c/NR,s/NR,h/NR}'
    grep "\[d2h-bw\]" "/tmp/pp_$tag.log" | grep rows512 | \
        awk '{for(i=1;i<=NF;i++)if($i~/^clean_copy=/)x+=substr($i,12)} END{if(NR)printf "  clean_copy n=%d avg=%.3f ms\n",NR,x/NR}'
}

echo "== 基线（含同步探针 + 干净拷贝探针）=="
run base "$@"
for extra in "$@"; do
    case "$extra" in
        kick) echo "== KICK 对照 =="; run kick LINGXI_KICK=1 ;;
        poll) echo "== EVT_POLL 对照 =="; run poll LINGXI_EVT_POLL=1 ;;
        pool) echo "== MEMPOOL_KEEP 对照 =="; run pool LINGXI_MEMPOOL_KEEP=1 ;;
        ple)  echo "== SKIP=ple（PLE 残底账）=="; run ple LINGXI_SKIP=ple ;;
    esac
done
