# lingxi 工程纪律入口（R6，2026-10-21）
# 动机：手工长命令链是本仓 5 个工程事故的共同根因（A 表 #5）——
#   "cp 旧二进制"（构建后忘 cp）、"pkill 自杀"、重打包 OOM 无预检。
# 原则：**构建/部署/回归的权威命令只存在于这里和 start.sh**，别处不再手写。

CARGO := $(shell [ -f $$HOME/.cargo/env ] && . $$HOME/.cargo/env && command -v cargo || command -v cargo)
PY    ?= python3

.PHONY: build deploy smoke regress gates gptq-pipeline

## 构建 + 部署二进制（一条链，杜绝"改了代码跑了旧 bin"）
build:
	. $(HOME)/.cargo/env 2>/dev/null || true; \
	cargo build --release && \
	cp -v target/release/lingxi-bench bin/lingxi-bench && \
	date +%T && ls -l --time-style=+%T bin/lingxi-bench

## 判据回归（QA 套件 + 护栏 gates；GATES=0 跳过 gates）
regress: build
	bash tests/regression/run.sh

gates: build
	bash tests/regression/gates.sh

## GPTQ 一条龙（校准→校验→重打包→判决→md5 落账），参数全走 env
gptq-pipeline:
	bash tools/arena-gen/gptq_pipeline.sh all
