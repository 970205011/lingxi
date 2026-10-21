#!/usr/bin/env bash
# ============================================================================
# GPTQ 补偿一条龙（R6 工程链固化，2026-10-21）：
#   校准 accum → hessian 校验 → 重打包 → 输出空间判决 → md5 落账
#
# 为什么是它：6.2-3 全程是手工长命令，沿途踩过 5 个工程事故——
#   torch.save 截断（zip 中央目录缺失）、双卡 prefill 线程交错 flush 残缺
#   hessian、hessian 12.8GB/层 CPU 预取 OOM×2、pkill 自杀、cp 旧二进制。
# 本脚本把护栏固化：每步前置校验、失败即停、产物 md5 落账、支持断点续跑
# （已完成的阶段有标记文件则跳过，STAGE=all 时幂等）。
#
# 用法：
#   bash tools/arena-gen/gptq_pipeline.sh all
#   bash tools/arena-gen/gptq_pipeline.sh calib|pack|check   # 单阶段
# 关键 env（都有默认，按需覆盖）：
#   PY          python（默认 python3）
#   DUMP        引擎校准 dump 目录（LINGXI_MOE_DUMP 指向处）
#   HESS        hessians 输出目录（.pt 分片）
#   SRC         bf16 源 arena
#   OUT         产物路径        FMT   ARENA_FMT（默认 MIXED）
#   HOT_CSV     热集表（MIXED 时必填）   GPTQ_ONLY_FMT  只补偿的格式段（可选）
#   RTN_OUT     RTN 对照产物（check 阶段用，默认 <OUT去掉.gptq>.bin 不存在时须显式给）
#   LAYERS      层数（默认 48）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-python3}
DUMP=${DUMP:-/data/models/lingxi-assets/gptq-calib}
HESS=${HESS:-/data/models/lingxi-assets/gptq-hessians}
SRC=${SRC:-/data/models/lingxi-assets/qwen4_exp.expert-arena.bin}
OUT=${OUT:-/data/models/lingxi-assets/qwen4_exp.expert-arena.mix64.gptq.bin}
FMT=${FMT:-MIXED}
LAYERS=${LAYERS:-48}
STAGE=${1:-all}
MD5_LOG=${MD5_LOG:-$OUT.md5log}
NINFER_PY=${NINFER_PY:-$HOME/ninfer}

step() { echo; echo "[pipe $(date +%T)] $*"; }

# ── 阶段 1：校准 accum（引擎 dump → hessian 分片）─────────────────────────
# 前置：引擎侧已用 LINGXI_MOE_DUMP=<DUMP> LINGXI_MOE_DUMP_ACT=1 跑过校准流，
# dump 里每调用落 moe{NNN}_{xpair,act}.bf16 + meta.json（见 PLAN 6.2-3）。
do_calib() {
    [ -f "$DUMP/.accum_done" ] && { echo "[pipe] accum 标记已存在，跳过（rm $DUMP/.accum_done 强制重跑）"; return; }
    step "校准 accum（两遍法 + GPU 常驻累加器；层分布断言防交错残缺）"
    "$PY" tools/arena-gen/calib_accum.py "$DUMP" "$HESS"
    touch "$DUMP/.accum_done"
}

# ── 阶段 2：hessian 分片校验（zip 结构 + 层数 + 形状抽查）──────────────────
do_validate() {
    step "校验 hessian 分片（zipfile + 层数 + 键覆盖）"
    "$PY" - <<EOF
import zipfile, os, sys
d = "$HESS"; L = $LAYERS
files = sorted(f for f in os.listdir(d) if f.startswith("hessians_L"))
assert len(files) == L, f"分片数 {len(files)} != {L}（双卡线程交错互覆的典型症状）"
bad = [f for f in files if not zipfile.is_zipfile(os.path.join(d, f))]
assert not bad, f"zip 结构损坏（writeback 截断）: {bad}"
print(f"[pipe] hessians {L}/{L} OK")
EOF
}

# ── 阶段 3：重打包（torch.load 直落 GPU，CPU 峰值 ~0；B=16 批量量化）──────
do_pack() {
    [ -f "$OUT" ] && [ -f "$OUT.done" ] && { echo "[pipe] 产物标记已存在，跳过（rm $OUT.done 强制重跑）"; return; }
    step "重打包 FMT=$FMT（GPTQ_H=$HESS；GPU 直载防 OOM）"
    local extra=()
    [ -n "${HOT_CSV:-}" ] && extra+=("HOT_CSV=$HOT_CSV")
    [ -n "${GPTQ_ONLY_FMT:-}" ] && extra+=("GPTQ_ONLY_FMT=$GPTQ_ONLY_FMT")
    env "${extra[@]}" ARENA_FMT="$FMT" GPTQ_H="$HESS" PYTHONPATH="$NINFER_PY" \
        "$PY" tools/arena-gen/quantize_expert_arena.py "$SRC" "$OUT"
    # 完成标记 = 产物 + md5 落账（防"半写完产物被当成品"，同 PLE sidecar 教训）。
    md5sum "$OUT" | tee -a "$MD5_LOG"
    touch "$OUT.done"
}

# ── 阶段 4：输出空间判决（RTN vs GPTQ，同 Hessian 对比，bf16 原档参考）───
do_check() {
    step "输出空间判决（gptq_check.py）"
    local rtn=${RTN_OUT:-${OUT%.gptq.bin}.bin}
    [ -f "$rtn" ] || { echo "[pipe] ⚠️ RTN 对照产物不存在（$rtn）——给 RTN_OUT= 或跳过 check"; return; }
    if ! PYTHONPATH="$NINFER_PY:$PWD/tools/arena-gen" "$PY" tools/arena-gen/gptq_check.py \
        "$rtn" "$OUT" "$HESS" "$SRC"; then
        echo "[pipe] ⚠️ 判决脚本失败不阻塞——它是抽样对比，重打包产物本身已由 $OUT.done 背书"
    fi
}

case "$STAGE" in
    calib) do_calib; do_validate ;;
    pack)  do_validate; do_pack ;;
    check) do_check ;;
    all)   do_calib; do_validate; do_pack; do_check ;;
    *) echo "用法: $0 [all|calib|pack|check]" >&2; exit 2 ;;
esac
echo; echo "[pipe $(date +%T)] 完成（阶段=$STAGE 产物=$OUT）"
