#!/usr/bin/env bash
# 长上下文（256K）单实例部署：双卡 layer-split，主入口:8000。
# 用法：PORT=8000 CTX=262144 SLOTS=7200 SLOTS1=7600 bash tools/deploy/up_long.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p run

PORT=${PORT:-8000}
CTX=${CTX:-262144}
SLOTS=${SLOTS:-7200}     # GPU0 槽位
SLOTS1=${SLOTS1:-7600}   # GPU1 槽位
# 不活跃会话的落盘缓存：目录 + 上限。默认 tmpfs（进程退出后还在，重启机器才没）。
# 想跨重启保留就指到真实磁盘，例如 /data/lingxi-prefix-cache。
CACHE_DIR=${CACHE_DIR:-/tmp/lingxi-prefix-cache}
CACHE_GIB=${CACHE_GIB:-24}
# 短会话不落盘（GDN/PLE 固定 ~113MB/份，短上下文重算更便宜）
CACHE_MIN_TOKENS=${CACHE_MIN_TOKENS:-8192}
# 长期专家频次表：真实流量累计，之后用它重建钉表（每 ~20 个 decode token 合并落盘一次）
# 放 run/（可写；docs/ 在容器里对服务用户只读，实测 Permission denied）
HIST=${HIST:-run/hist_live.csv}
# MTP 投机：冷请求（无前缀可复用）自动开，命中缓存的请求自动退回自回归。
# ⚠️ 256K + MTP 首次重启要确认显存；若顶到 32G，把 SLOTS/SLOTS1 各降 400~800。
SPEC=${SPEC:-1}
MTP=${MTP:-/models/lingxi-assets/qwen4_exp.mtp.fixed.ninfer}
ART=${LINGXI_ARTIFACT:-/models/qwen4_exp-artifact/qwen4_exp.ninfer}

# 2026-10-18 实测（33,052-token 代码类冷 prefill，MTP 开）：
# lo=24/chunk=512 336.9s
# lo=28/chunk=512 319.3s（−5.2%，但 chunk 1024 会 OOM）
# lo=24/chunk=1024 278.7s（−17.3%，miss 440k vs 760k）← **生产默认**
# lo=24/chunk=1536 OOM（GPU0 32.4G）
SPLIT_LO=${SPLIT_LO:-24}
CHUNK=${CHUNK:-1024}
EXTRA=()
[ -n "${SPLIT_LO:-}" ] && EXTRA+=(LINGXI_LAYER_SPLIT="$SPLIT_LO")
[ -n "${CHUNK:-}" ] && EXTRA+=(LINGXI_PREFILL_CHUNK="$CHUNK")

if [ -f run/serve_long.pid ] && kill -0 "$(cat run/serve_long.pid)" 2>/dev/null; then
  echo "already running (run/serve_long.pid)"; exit 1
fi

# 日志交给 logrun.py：超过 --max-mb 自动轮转，pid 文件里是它（down.sh 杀它会转发给子进程）
setsid python3 ./tools/deploy/logrun.py --log run/serve_long.log --max-mb 32 --keep 3 -- \
  env "${EXTRA[@]}" \
  LINGXI_SERVE_MAX_CTX="$CTX" \
  LINGXI_ARENA_SLOTS="$SLOTS" \
  LINGXI_LAYER_SPLIT_SLOTS="$SLOTS1" \
  LINGXI_SERVE_PREFIX_CACHE=1 \
  LINGXI_SERVE_CACHE_DIR="$CACHE_DIR" \
  LINGXI_SERVE_CACHE_GIB="$CACHE_GIB" \
  LINGXI_SERVE_CACHE_MIN_TOKENS="$CACHE_MIN_TOKENS" \
  LINGXI_ARENA_HIST="$HIST" \
  LINGXI_SERVE_SPEC="$SPEC" \
  LINGXI_MTP_ARTIFACT="$MTP" \
  ./start.sh ./bin/lingxi-bench serve \
  "$ART" "$PORT" &
echo $! > run/serve_long.pid

echo "started long-context: :$PORT  ctx=$CTX  slots=$SLOTS/$SLOTS1"
echo "log: run/serve_long.log"
