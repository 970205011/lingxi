#!/usr/bin/env bash
# 停掉 up.sh 起的全部进程（按 pid 文件，不用 pkill 以免误杀）。
set -uo pipefail
cd "$(dirname "$0")/../.."

# 先请求优雅退出：本机 /shutdown 会把**当前会话**也落盘（只允许 127.0.0.1）
for P in 8000 8001 8002; do
  curl -s -m 30 -X POST "http://127.0.0.1:$P/shutdown" >/dev/null 2>&1 || true
done

for f in run/proxy.pid run/serve1.pid run/serve0.pid run/serve_long.pid; do
  if [ -f "$f" ]; then
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      # 落盘 8GB 级检查点可能要几秒，给足 60s 再强杀
      for _ in $(seq 1 120); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$f"
  fi
done
echo "stopped"
