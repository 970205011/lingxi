#!/usr/bin/env bash
# 「这一轮的时间到底花在哪」——**唯一权威的分段测量入口**。
#
# 与 spec_bench.sh 的分工：
# spec_bench.sh → 吞吐尺子（tok/s，eval 用）。
# seg_profile.sh → 归因（每层每段的 GPU 毫秒 + 路由 host 毫秒），**不看 tok/s**。
#
# 为什么不能用剥块（LINGXI_SKIP）来归因：跳块会改路由/改数值 ⇒ 路由乱跳、专家槽全废，
# 被剥的块反而更慢。
# 这里靠 evprof 的**带计时事件**在同一轮内逐段量，零扰动（不排空流水）。
#
# 三个探针叠加，各自回答一个问题：
# LINGXI_EVPROF=1 → 每层 6 段的真 GPU 毫秒（seg3=路由/host，seg4=专家）
# LINGXI_FULLDEV_HTIME=1 → 设备全路由的**host 侧**分解：launch/pairs/自旋/读计数
# LINGXI_ROUTE_FULLDEV_TRACE → 确认快路到底有没有被走（打了 "[fulldev] call" 就是走了）
#
# 关键判读（2026-10-23 实测锚点，V100 x8 ×2，mix64 arena，kd=5）：
# 主干 24 层/卡五段相加 ≈ 32~34 ms；seg4 专家 0.49、seg3 路由 0.33、attn 0.34、
# GR 两段 0.18、入口 ~0.10。**seg3 里 host 只占 24.6µs**（htime 汇总），其余是 GPU。
#
# 用法：
# bash tools/golden/seg_profile.sh # 默认 128 tok
# bash tools/golden/seg_profile.sh 256 # 更长的稳态
# LINGXI_ROUTE_FULLDEV=0 bash tools/golden/seg_profile.sh 128 # legacy 路由对照
set -u
cd /models/lingxi
N=${1:-128}
ART=${LINGXI_ARTIFACT:-/models/qwen4_exp-artifact/qwen4_exp.ninfer}
PROMPT=${2:-"The capital of France is"}

# 与 spec_bench.sh 同一套生产参数（seg3/seg4 的分界依赖 FULLDEV 走哪条路）
export LINGXI_ROUTE_FULLDEV=${LINGXI_ROUTE_FULLDEV:-1}
export LINGXI_SERVE_SPEC_KD=${LINGXI_SERVE_SPEC_KD:-7}
export LINGXI_SPEC_MIN_P=${LINGXI_SPEC_MIN_P:-0.7}
export LINGXI_EVPROF=1
export LINGXI_FULLDEV_HTIME=1
export LINGXI_ROUTE_FULLDEV_TRACE=1

OUT=/tmp/seg_profile_$$.log
LINGXI_TALK_SPEC=1 ./start.sh ./bin/lingxi-bench talk "$ART" "$PROMPT" "$N" > "$OUT" 2>&1
RC=$?

echo "=== 吞吐（仅参考，尺子是 spec_bench.sh） ==="
grep -a "talk-spec] 共" "$OUT" | tail -1

echo
echo "=== 每卡五段（最后一轮，逐层求和） ==="
awk '/逐段真 GPU 时间/{show=1} show&&/五段相加/{print;show=0} show&&/(入口|attn|mlp_GR|MoE:|gr_write)/{print}' "$OUT"

echo
echo "=== 路由 host 侧（稳态层均 µs；自旋≈0 说明 host 不是瓶颈） ==="
grep -a "^\[htime\]" "$OUT" | tail -4

echo
echo "=== 快路是否生效（有 [fulldev] call 才是设备全路由） ==="
FULLDEV_N=$(grep -ac "^\[fulldev\] call" "$OUT")
echo "命中快路抽样行数 = $FULLDEV_N （>0 即生效；0 表示在跑 legacy host 路）"
grep -a "^\[fulldev\] call" "$OUT" | head -3

echo
echo "=== 最慢的专家段层（诊断用） ==="
grep -a "最慢 8 层" -A9 "$OUT" | head -10

echo "原始日志：$OUT （RC=$RC）"
rm -f /tmp/seg_profile_*.log.old
