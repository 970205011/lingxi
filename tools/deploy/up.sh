#!/usr/bin/env bash
# 双实例单卡并发部署：GPU0->8001，GPU1->8002，主入口（代理）->8000。
# 用法：CTX=32768 GPU0_SLOTS=9000 GPU1_SLOTS=9000 bash tools/deploy/up.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p run

CTX=${CTX:-32768}
GPU0_SLOTS=${GPU0_SLOTS:-9000}
GPU1_SLOTS=${GPU1_SLOTS:-9000}
P0=${P0:-8001}   # GPU0 实例
P1=${P1:-8002}   # GPU1 实例
PP=${PP:-8000}   # 对外主入口（代理）

if [ -f run/serve0.pid ] && kill -0 "$(cat run/serve0.pid)" 2>/dev/null; then
  echo "already running (run/serve0.pid)"; exit 1
fi

setsid python3 ./tools/deploy/logrun.py --log run/serve0.log --max-mb 16 --keep 2 -- \
  ./tools/deploy/serve_single.sh 0 "$P0" "$GPU0_SLOTS" "$CTX" &
echo $! > run/serve0.pid
setsid python3 ./tools/deploy/logrun.py --log run/serve1.log --max-mb 16 --keep 2 -- \
  ./tools/deploy/serve_single.sh 1 "$P1" "$GPU1_SLOTS" "$CTX" &
echo $! > run/serve1.pid
setsid python3 ./tools/deploy/logrun.py --log run/proxy.log --max-mb 8 --keep 2 -- \
  python3 ./tools/deploy/proxy.py "$PP" "$P0" "$P1" &
echo $! > run/proxy.pid

echo "started: GPU0->$P0  GPU1->$P1  proxy->$PP  ctx=$CTX slots=$GPU0_SLOTS/$GPU1_SLOTS"
echo "logs: run/serve0.log run/serve1.log run/proxy.log"
