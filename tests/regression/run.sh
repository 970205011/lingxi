#!/usr/bin/env bash
# 回归套件（㊽ 判据地基 → 2026-10-19 可用性判据版）：按 golden.toml 逐 case 跑判据。
#
# 用法：
# bash tests/regression/run.sh [env=val ...] # 判据模式：不等即退出码 1
# bash tests/regression/run.sh RECORD=1 [...] # 记录模式：打印实测值，不比对
#
# 环境：
# ART=/path/model.ninfer 产物路径（默认 /models/qwen4_exp-artifact/qwen4_exp.ninfer）
# CASES="smoke48,needle12k" 只跑指定 case（逗号分隔；默认全量）
# BLOCKING=1 置 CUDA_LAUNCH_BLOCKING=1（stopgap：WIP 懒分配分支有一起
# 间歇 CUDA_ERROR_INVALID_VALUE（见 PLAN_NEXT 事故记录），
# 同步发射可关闭竞态窗口；根因修复前回归门禁默认开）
#
# 设计约束：
# * 走 ./start.sh ⇒ 与生产同一套 env；传入的 env=val 放 start.sh 之前生效
# （start.sh 内部开关一律 ${VAR:-default} 形式，可被外部覆盖——2026-10-19 教训）。
# * 每个 case 独立进程（引擎不跨 run 复用状态）。
#
# golden.toml case 字段（2026-10-19 判据升级：从"首 token 金丝雀"到"可用性判据"）：
# name case 名
# prompt / prompt_file 二选一
# tokens 期望的 prompt token 数（信息项，输出对照用）
# max_new 生成长度（默认 1）。可用性判据必须 >1，否则 eos 即"测了寂寞"
# expect_first_token 首 token 锚（可选；短 prompt 稳定锚保留，长 prompt 有损档会合法翻转）
# expect_contains 逗号分隔子串，生成文本必须**全部**包含（质量/检索判据）
# min_new 生成 token 数下限（可选，默认：设了 expect_contains 则 ≥1）
# source 记录口径（哪套环境录的期望值）
set -u
cd "$(dirname "$0")/../.."
ART=${ART:-${LINGXI_ARTIFACT:-/models/qwen4_exp-artifact/qwen4_exp.ninfer}}
[ "${BLOCKING:-0}" = 1 ] && export CUDA_LAUNCH_BLOCKING=1
RECORD=0
ENVS=()
for a in "$@"; do
    case "$a" in
        RECORD=1) RECORD=1 ;;
        *) ENVS+=("$a") ;;
    esac
done

GOLDEN=tests/regression/golden.toml
pass=0; fail=0
names=($(grep -a '^name = ' "$GOLDEN" | sed -E 's/name = "(.*)"/\1/'))
# 从 toml 抠 case 块里某个字段（行内 # 注释剥掉；缺省回退 default）
field() { echo "$1" | grep -a "^$2" | sed -E 's/#.*//' | sed -E "s/^$2 *= *\"?([^\"]*)\"?.*$/\1/" | xargs | head -1; }
run_case() {
    local name="$1"
    local block expect pfile prompt max_new contains min_new
    block=$(awk -v n="\"$name\"" '/^\[\[case\]\]/{f=0} $0 ~ "name = " n {f=1} f' "$GOLDEN")
    expect=$(field "$block" expect_first_token)
    pfile=$(field "$block" prompt_file)
    prompt=$(field "$block" prompt)
    max_new=$(field "$block" max_new);  [ -z "$max_new" ] && max_new=1
    contains=$(field "$block" expect_contains)
    min_new=$(field "$block" min_new);  [ -z "$min_new" ] && { [ -n "$contains" ] && min_new=1 || min_new=0; }
    local text
    if [ -n "$pfile" ]; then
        text=$(cat "$pfile")
    else
        text="$prompt"
    fi
    local out
    out=$(mktemp /tmp/regr.XXXXXX.log)
    env "${ENVS[@]:-DUMMY=0}" ./start.sh ./bin/lingxi-bench talk "$ART" "$text" "$max_new" > "$out" 2>&1
    # 间歇驱动级故障（P0 海森堡 bug 两形态：EINVAL 已降级；load 期 libcuda NULL+0xe8
    # segfault 未根因，见 PLAN_NEXT）会**无判决杀死进程**。检测：日志里既无"首 token"
    # 也无"失败退出" ⇒ 静默死亡 ⇒ 重试一次（第二次仍死才算 FAIL，首次记 FLAKY）。
    local flaky=""
    if ! grep -qa '首 token' "$out" && ! grep -qa '失败退出' "$out"; then
        flaky="FLAKY(进程静默死亡,已重试) "
        out2=$(mktemp /tmp/regr.XXXXXX.log)
        env "${ENVS[@]:-DUMMY=0}" ./start.sh ./bin/lingxi-bench talk "$ART" "$text" "$max_new" > "$out2" 2>&1
        if grep -qa '首 token' "$out2" || grep -qa '失败退出' "$out2"; then
            out="$out2"
        else
            echo "FAIL  $name  两轮均静默死亡（驱动级 segfault？dmesg 查 libcuda）→ $out / $out2"
            fail=$((fail+1))
            return 0
        fi
    fi
    local toks got gentok gen_text
    toks=$(tr '\r' '\n' < "$out" | grep -aE '^prompt=' | tail -1 | sed -E 's/prompt=([0-9]+).*/\1/')
    got=$(tr '\r' '\n' < "$out" | grep -aE '首 token' | tail -1 | sed -E 's/.*首 token = ([0-9-]+).*/\1/')
    gentok=$(tr '\r' '\n' < "$out" | grep -aE -- '-- 生成 [0-9]+ tokens' | tail -1 | sed -E 's/.*-- 生成 ([0-9]+) tokens.*/\1/')
    [ -z "$gentok" ] && gentok=0
    gen_text=$(tr '\r' '\n' < "$out" | sed -n '/-- 生成 [0-9]* tokens，解码文本 --/,$p' | tail -n +2)
    if [ "$RECORD" = 1 ]; then
        echo "RECORD $name tokens=$toks first_token=$got gen=$gentok  (期望槽位: $expect)"
        echo "  text: $(echo "$gen_text" | head -c 200 | tr '\n' ' ')"
        return 0
    fi
    # ── 判据逐条核 ──
    local reasons=()
    if [ -n "$expect" ] && [ "$expect" != "-1" ] && [ "$got" != "$expect" ]; then
        reasons+=("首 token='$got'≠$expect")
    fi
    if [ "$gentok" -lt "$min_new" ]; then
        reasons+=("生成 $gentok token < 下限 $min_new（可能即 eos，判据未覆盖到生成质量）")
    fi
    if [ -n "$contains" ]; then
        local miss=""
        IFS=','; for sub in $contains; do
            sub=$(echo "$sub" | xargs)
            case "$gen_text" in
                *"$sub"*) ;;
                *) miss="$miss [$sub]" ;;
            esac
        done; unset IFS
        [ -n "$miss" ] && reasons+=("缺子串:$miss")
    fi
    if [ ${#reasons[@]} = 0 ]; then
        echo "PASS  $name  tokens=$toks 首 token=$got gen=$gentok ${flaky}"
        pass=$((pass+1))
    else
        echo "FAIL  $name  tokens=$toks 首 token='$got' gen=$gentok | $(IFS=';'; echo "${reasons[*]}") → $out"
        fail=$((fail+1))
    fi
}

CASES=${CASES:-}
for n in "${names[@]}"; do
    if [ -n "$CASES" ] && ! echo ",$CASES," | grep -q ",$n,"; then
        continue
    fi
    run_case "$n"
done
echo "== 回归：$pass 过 / $fail 挂"
rc=0; [ "$fail" = 0 ] || rc=1

# 架构护栏 gates（2026-10-21 R5）：特定 bug 形态守门，见 gates.sh 头注。
# GATES=0 跳过（快速迭代时）；任何 gate 挂 ⇒ 整体退出码非零。
if [ "${GATES:-1}" != 0 ]; then
    bash tests/regression/gates.sh
    [ $? = 0 ] || rc=1
fi
exit $rc
