#!/usr/bin/env bash
# 双会话相位重叠判决尺。
#
# 干什么：同一进程起两个 host 线程、各驱动一套 (Sequence, SpecRuntime) 跑同一 prompt，
# 量"两会话并行聚合 tok/s"并逐字对拍 solo（正确性判据）。
# 判据（R/S 的三条）：
# ① 两会话各自文本与 solo **逐字相同**（本脚本自动 diff）；
# ② 聚合 ≥ 1.3× solo（实测封顶 **~1.1×** —— 反相位 PLL 后稳定 101~104 vs solo
# 90~94.5；行拼接/自由重叠/固定网格各自都更差。已收口：不上生产调度器）；
# ③ 无新增 Xid。
#
# 用法：
# bash tools/golden/spec_dual_bench.sh <tag> <tokens> [prompt]
# bash tools/golden/spec_dual_bench.sh dual 192
set -u
cd /models/lingxi
TAG=${1:?用法: spec_dual_bench.sh <tag> <tokens> [prompt]}
N=${2:-192}
PROMPT=${3:-"The capital of France is"}
ART=${LINGXI_ARTIFACT:-/models/qwen4_exp-artifact/qwen4_exp.ninfer}

export LINGXI_ROUTE_FULLDEV=${LINGXI_ROUTE_FULLDEV:-1}
export LINGXI_SERVE_SPEC_KD=${LINGXI_SERVE_SPEC_KD:-7}
export LINGXI_SPEC_MIN_P=${LINGXI_SPEC_MIN_P:-0.7}
# 反相位 PLL：消除自由漂移的 0.5× 塌落模式；0 = 对照（自由漂移）。
export LINGXI_TALK_SPEC_PHASE=${LINGXI_TALK_SPEC_PHASE:-36}

XID0=$(dmesg 2>/dev/null | grep -c Xid || true)

# ── solo 对照（同尺子：talk + LINGXI_TALK_SPEC=1）──
LINGXI_TALK_SPEC=1 ./start.sh ./bin/lingxi-bench talk "$ART" "$PROMPT" "$N" \
    > "/tmp/dual_${TAG}_solo.log" 2>&1
SOLO_LINE=$(grep -a "talk-spec] 共" "/tmp/dual_${TAG}_solo.log" | tail -1)
SOLO_TEXT=$(tail -1 "/tmp/dual_${TAG}_solo.log")
[ -z "$SOLO_LINE" ] && { echo "solo 跑失败，尾部："; tail -3 "/tmp/dual_${TAG}_solo.log"; exit 1; }

# ── 双会话 ──
LINGXI_TALK_SPEC=1 LINGXI_TALK_SPEC_BATCH=1 timeout 900 \
    ./start.sh ./bin/lingxi-bench talk "$ART" "$PROMPT" "$N" \
    > "/tmp/dual_${TAG}_2s.log" 2>&1
RC=$?
[ "$RC" -ne 0 ] && { echo "dual 跑失败 RC=$RC，尾部："; tail -5 "/tmp/dual_${TAG}_2s.log"; exit 1; }

AGG=$(grep -a "talk-spec-2] 聚合" "/tmp/dual_${TAG}_2s.log" | tail -1)
SA=$(grep -a "会话 A" "/tmp/dual_${TAG}_2s.log" | tail -1)
SB=$(grep -a "会话 B" "/tmp/dual_${TAG}_2s.log" | tail -1)

echo "solo   : $SOLO_LINE"
echo "dual A : $SA"
echo "dual B : $SB"
echo "dual   : $AGG"

# ── 判据 ①：逐字对拍 ──
grep -a "^\[A\] " "/tmp/dual_${TAG}_2s.log" | sed 's/^\[A\] //' > "/tmp/dual_${TAG}_a.txt"
grep -a "^\[B\] " "/tmp/dual_${TAG}_2s.log" | sed 's/^\[B\] //' > "/tmp/dual_${TAG}_b.txt"
printf '%s\n' "$SOLO_TEXT" > "/tmp/dual_${TAG}_solo.txt"
if diff -q "/tmp/dual_${TAG}_a.txt" "/tmp/dual_${TAG}_solo.txt" >/dev/null \
&& diff -q "/tmp/dual_${TAG}_b.txt" "/tmp/dual_${TAG}_solo.txt" >/dev/null; then
    echo "判据①  文本对拍：A ≡ B ≡ solo ✓（md5 $(md5sum < "/tmp/dual_${TAG}_a.txt" | cut -d' ' -f1)）"
else
    echo "判据①  **文本分叉** ✗ —— A/B 与 solo 不一致！"
    diff "/tmp/dual_${TAG}_a.txt" "/tmp/dual_${TAG}_solo.txt" | head -5
fi

# ── 判据 ③：Xid ──
XID1=$(dmesg 2>/dev/null | grep -c Xid || true)
if [ "$XID1" = "$XID0" ]; then
    echo "判据③  Xid：无新增 ✓（累计 $XID1）"
else
    echo "判据③  **新增 Xid $((XID1 - XID0)) 个** ✗（$XID0 → $XID1）"
fi
