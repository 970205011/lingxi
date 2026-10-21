#!/usr/bin/env bash
# 单流 decode 计时（**低噪声版**）。
#
# 为什么重写：旧版只打一个"后段中位数"，而实测**相邻两次同口径测量就能差 5%**
# （min/p10/med 在 0.026~0.031 之间摆），而大多数优化的收益是 1~3% ——
# 也就是**尺子比被测物还粗**。新版做四件事：
#   1. 报完整分布（min/p10/p50/p90/max/mean/σ）并给出**离散度判据**；
#   2. 跑 R 轮，报「每轮 p50」和「R 轮里最好的 p50」——后者对系统抖动更稳；
#   3. 默认丢掉前 20% 的 token（冷启动 + PLE 冷读）；
#   4. 默认叠 `LINGXI_FIXED_IDS`（若给了 id 文件）——**A/B 必须用同一 token 流**，
#      否则采样差异会改变路由，两个配置根本不是同一负载。
#
# 用法：bash tools/golden/bench.sh <tag> <tokens> [R] [env=val ...]
#   IDS=/tmp/fixed400.txt bash tools/golden/bench.sh base 400 3
set -u
cd /models/lingxi
TAG=${1:?用法: bench.sh <tag> <tokens> [R] [env=val ...]}
N=${2:-200}
R=${3:-3}
if [ $# -ge 3 ]; then shift 3; else shift $#; fi
export LINGXI_EXPERT_ARENA=${LINGXI_EXPERT_ARENA:-/dev/shm/qwen4_exp.expert-arena.q4g64.bin}
export LINGXI_ARENA_SLOTS=${LINGXI_ARENA_SLOTS:-9600}   # ★㊿-⑧ 9600/10000：钉表 + 4K prefill 兼顾
export LINGXI_LAYER_SPLIT_SLOTS=${LINGXI_LAYER_SPLIT_SLOTS:-10000}
# 生产默认：钉表（㊿-⑤）。想量"不钉表"的基线就 LINGXI_ARENA_PREWARM_N=0。
export LINGXI_ARENA_PREWARM=${LINGXI_ARENA_PREWARM:-/models/lingxi/docs/bench-logs/hist_spec.csv}
export LINGXI_ARENA_PREWARM_N=${LINGXI_ARENA_PREWARM_N:-200}
export LINGXI_ARENA_PIN_GIB=${LINGXI_ARENA_PIN_GIB:-64}
# **生产/计时路径默认开三档 W4**（§3.49）：实测整机 p10 27,576 → 24,763 µs（−10.2%），
# 且贪心 token 序列**逐位不变**。代价：载入时多 ~25 s 的主机端在线量化（10.0s → 36.9s）。
# 想量 bf16 基线就 `LINGXI_DENSE_Q4=0`（三档一起关）。
export LINGXI_DENSE_Q4=${LINGXI_DENSE_Q4:-1}
export LINGXI_DENSE_Q4_QSA=${LINGXI_DENSE_Q4_QSA:-1}
export LINGXI_HEAD_Q4=${LINGXI_HEAD_Q4:-1}
# W4 量化产物落盘缓存：首次算+存，之后**只读**（省 ~27 s 启动量化，且读的字节少 4 倍）。
export LINGXI_Q4_CACHE=${LINGXI_Q4_CACHE:-/dev/shm/lingxi-q4cache}
ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer
WARM=$(( N / 5 ))                       # 丢前 20%
if [ -n "${IDS:-}" ] && [ -f "$IDS" ]; then
  export LINGXI_FIXED_IDS="$(cat "$IDS")"
fi

# 每轮的 p10/p50 用 >> 追加，所以必须先清空（否则会带上上一次同 tag 的残留）
rm -f "/tmp/bench_${TAG}_p10.txt" "/tmp/bench_${TAG}_p50.txt"

for r in $(seq 1 "$R"); do
  OUT=/tmp/bench_${TAG}_r$r.txt
  env "$@" ./bin/lingxi-bench talk "$ART" "The capital of France is" "$N" > "$OUT" 2>&1
  tr '\r' '\n' < "$OUT" | grep -a 'µs/token' \
    | sed -E 's/.*[[:space:]]([0-9]+\.[0-9]+) µs\/token.*/\1/' \
    | tail -n +$(( WARM + 1 )) | sort -n \
    | awk -v t="$TAG" -v r="$r" '
      { v[NR] = $1; s += $1; ss += $1*$1 }
      END {
        n = NR
        if (n == 0) { printf "%-22s r%s  无数据\n", t, r; exit }
        i10 = int(0.10*n + 0.5); if (i10 < 1) i10 = 1
        i50 = int(0.50*n + 0.5); if (i50 < 1) i50 = 1; if (i50 > n) i50 = n
        i90 = int(0.90*n + 0.5); if (i90 < 1) i90 = 1; if (i90 > n) i90 = n
        mean = s/n; sd = sqrt((ss/n) - mean*mean)
        # 离散度 = (p90-p10)/p50；>8% 说明这一轮不能用来支撑 1~3% 级别的结论
        spread = (v[i90] - v[i10]) / v[i50] * 100
        printf "%-20s r%s  min=%.0f p10=%.0f p50=%.0f p90=%.0f max=%.0f mean=%.0f sd=%.0f µs 离散=%.1f%%%s\n",
               t, r, v[1], v[i10], v[i50], v[i90], v[n], mean, sd, spread,
               (spread > 8 ? "  ⚠️离散过大" : "")
        print v[i50] >> ("/tmp/bench_" t "_p50.txt")
        print v[i10] >> ("/tmp/bench_" t "_p10.txt")
      }'
done

# **以 p10 为头条指标**：实测同一配置连跑两轮，p50 会漂 5%，而 **p10 只漂 2%** ——
# p10 代表"机器干净时"的速度（不受 PLE 冷行/专家槽 miss 这些长尾拖累），
# 更适合判断 1~3% 级别的优化。p50 一起报，作为"体验"的参考。
B10=$(sort -n "/tmp/bench_${TAG}_p10.txt" | head -1)
B50=$(sort -n "/tmp/bench_${TAG}_p50.txt" | head -1)
printf "%-20s **最好的 p10 = %.1f µs/token ⇒ %.2f tok/s**  （参考：最好的 p50 = %.1f µs ⇒ %.2f tok/s）\n" \
  "$TAG" "$B10" "$(awk -v v="$B10" 'BEGIN{print 1e6/v}')" "$B50" "$(awk -v v="$B50" 'BEGIN{print 1e6/v}')"
rm -f "/tmp/bench_${TAG}_p50.txt" "/tmp/bench_${TAG}_p10.txt" \
      "/tmp/bench_${TAG}_b10.txt" "/tmp/bench_${TAG}_b50.txt"
