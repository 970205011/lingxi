# lingxi 工程纪律入口（R6，2026-10-21）
# 动机：手工长命令链是本仓 5 个工程事故的共同根因（A 表 #5）——
#   "cp 旧二进制"（构建后忘 cp）、"pkill 自杀"、重打包 OOM 无预检。
# 原则：**构建/部署/回归的权威命令只存在于这里和 start.sh**，别处不再手写。

CARGO_BIN := $(firstword $(wildcard $(HOME)/.cargo/bin/cargo) $(shell command -v cargo))
PY    ?= python3
# cargo 实际 target 目录（尊重 .cargo/config.toml 的 target-dir 覆盖，如 141 的 /dev/shm 内存盘）
TARGET_DIR := $(shell cd $(CURDIR) && $(CARGO_BIN) metadata --no-deps --format-version 1 2>/dev/null | $(PY) -c "import json,sys; print(json.load(sys.stdin)['target_directory'])")

.PHONY: build deploy smoke regress gates gptq-pipeline dist

## 构建 + 部署二进制（一条链，杜绝"改了代码跑了旧 bin"）
build:
	$(CARGO_BIN) build --release && \
	cp -v "$(TARGET_DIR)/release/lingxi-bench" bin/lingxi-bench && \
	date +%T && ls -l --time-style=+%T bin/lingxi-bench

## 判据回归（QA 套件 + 护栏 gates；GATES=0 跳过 gates）
regress: build
	bash tests/regression/run.sh

gates: build
	bash tests/regression/gates.sh

## GPTQ 一条龙（校准→校验→重打包→判决→md5 落账），参数全走 env
gptq-pipeline:
	bash tools/arena-gen/gptq_pipeline.sh all

## 发布包备料（不发布：只产出本地 tarball；git tag / Gitea Release 等用户拍板）
## 模型产物不含在包内（~252GB，按 docs/FORMAT.md 自行构建）。
DIST_DATE := $(shell date +%Y%m%d)
DIST_DIR  := dist/lingxi-$(DIST_DATE)

dist: build
	rm -rf $(DIST_DIR) dist/lingxi-$(DIST_DATE).tar.gz
	mkdir -p $(DIST_DIR)/bin $(DIST_DIR)/docs $(DIST_DIR)/tools/deploy
	cp -v bin/lingxi-bench $(DIST_DIR)/bin/
	cp -v start.sh README.md README.en.md LICENSE $(DIST_DIR)/
	cp -v docs/RELEASE-v0.2.0.md docs/FORMAT.md docs/HANDOVER.md $(DIST_DIR)/docs/
	cp -v tools/deploy/up.sh tools/deploy/down.sh tools/deploy/serve_single.sh $(DIST_DIR)/tools/deploy/
	cd dist && tar czf lingxi-$(DIST_DATE).tar.gz lingxi-$(DIST_DATE) && ls -lh lingxi-$(DIST_DATE).tar.gz
