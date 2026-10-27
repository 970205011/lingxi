#!/usr/bin/env bash
# prefill 逐位等价回归（㊿-②/③ 的守护脚本）。
#
# 为什么必须单独有它：2026-10-18 之前 `slotsm` 专家核每组只算前 8 个对，
# 长 prompt 的专家输出大面积是脏值 —— 脏值随 chunk 边界/命中态/卡配置变化，
# 表现为"首 token 随配置漂移"（16 / 79648 / 1294 / 220 / 332 / 7313 …），
# 而 48-token 短 prompt 判据一直是过的。本脚本用**固定长 prompt + 固定首 token**
# 把这一类 bug 钉住。
#
# 用法：
# bash tools/golden/prefill_equiv.sh <prompt_file> <expected_first_token> [env=val ...]
#
# 已知基准（2026-10-18，内核修复后；**分块已逐位不变，所以 1 chunk 与默认 512 同值**）：
# /tmp/p700.txt (867 token) → 96133
# /tmp/parity_prompt.txt (4167 token) → 332
# /tmp/ctx64.txt (23673 token) → 95835
# 想额外证明"分块不变"，可同一条 prompt 跑两遍：
# prefill_equiv.sh <pf> <tok>
# prefill_equiv.sh <pf> <tok> LINGXI_PREFILL_CHUNK=8192
#
# 判据：首 token 完全相等；不等就退出码 1（可直接挂 CI/提交前自检）。
set -u
cd /models/lingxi
PF=${1:?用法: prefill_equiv.sh <prompt_file> <expected_first_token> [env=val ...]}
EXPECT=${2:?缺 expected_first_token}
if [ $# -ge 2 ]; then shift 2; fi

ART=/models/qwen4_exp-artifact/qwen4_exp.ninfer
OUT=$(mktemp /tmp/prefill_equiv.XXXXXX.log)
# 走 start.sh ⇒ 与生产同一套 env（含 LINGXI_PREFILL_THREADS / LINGXI_MOE_TC）
env "$@" ./start.sh ./bin/lingxi-bench talk "$ART" "$(cat "$PF")" 1 > "$OUT" 2>&1
GOT=$(tr '\r' '\n' < "$OUT" | grep -aE '首 token' | tail -1 | sed -E 's/.*首 token = ([0-9-]+).*/\1/')
TOKS=$(tr '\r' '\n' < "$OUT" | grep -aE '^prompt=' | tail -1)
echo "== $TOKS"
if [ "$GOT" = "$EXPECT" ]; then
  echo "== ✅ prefill 等价：首 token = $GOT（期望 $EXPECT）→ $OUT"
  exit 0
else
  echo "== ❌ prefill 发散：首 token = '$GOT'（期望 $EXPECT）→ $OUT"
  exit 1
fi
