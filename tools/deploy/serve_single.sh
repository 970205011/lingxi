#!/usr/bin/env bash
# 单卡单实例（部署用）：serve_single.sh <gpu> <port> [slots] [max_ctx]
#   - LINGXI_LAYER_SPLIT=0：48 层全放在一张卡上
#   - LINGXI_DEVICE=<gpu>：选卡
#   - 前缀缓存默认开；槽位按 max_ctx 留给 KV 的余量调
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU=${1:?用法: serve_single.sh <gpu> <port> [slots] [max_ctx]}
PORT=${2:?}
SLOTS=${3:-9000}
CTX=${4:-32768}

export LINGXI_DEVICE="$GPU"
export LINGXI_LAYER_SPLIT=0
export LINGXI_ARENA_SLOTS="$SLOTS"
export LINGXI_SERVE_MAX_CTX="$CTX"
export LINGXI_SERVE_PREFIX_CACHE=1
# 不活跃会话的落盘缓存（每个实例独立目录/上限，避免两个进程互删文件）
export LINGXI_SERVE_CACHE_DIR="${LINGXI_SERVE_CACHE_DIR:-/tmp/lingxi-prefix-cache-$PORT}"
export LINGXI_SERVE_CACHE_GIB="${LINGXI_SERVE_CACHE_GIB:-12}"
export LINGXI_SERVE_CACHE_MIN_TOKENS="${LINGXI_SERVE_CACHE_MIN_TOKENS:-8192}"
# 长期专家频次表（每个实例独立文件；真实流量累计，用来重建钉表）
export LINGXI_ARENA_HIST="${LINGXI_ARENA_HIST:-run/hist_live-$PORT.csv}"
# MTP 投机：冷请求开，命中前缀缓存的请求自动退回自回归
export LINGXI_SERVE_SPEC="${LINGXI_SERVE_SPEC:-1}"
export LINGXI_MTP_ARTIFACT="${LINGXI_MTP_ARTIFACT:-/data/models/lingxi-assets/qwen4_exp.mtp.fixed.ninfer}"
ART=${LINGXI_ARTIFACT:-/models/qwen4_exp-artifact/qwen4_exp.ninfer}

exec ./start.sh ./bin/lingxi-bench serve \
  "$ART" "$PORT"
