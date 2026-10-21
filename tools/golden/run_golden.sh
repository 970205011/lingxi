#!/usr/bin/env bash
# golden 生成启动器（避免命令行自匹配误杀）
set -u
cd /models/lingxi
exec python3 -u tools/golden/run_layer_golden.py \
  "${1:-artifacts/golden_layers.npz}" \
  "${2:-The capital of France is}"
