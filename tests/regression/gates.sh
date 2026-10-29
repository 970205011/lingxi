#!/usr/bin/env bash
# 架构护栏 gates（2026-10-21 R5）：QA toml case 之外的"特定 bug 形态"守门。
# 背景：P2 分页与混档期间的 5 个 bug 全部穿过了旧 QA 3/3 才暴露（PLAN_NEXT 架构债务 A 表），
# 这里把每个 bug 形态固化成一条 gate。run.sh 末尾自动调用；GATES=0 跳过。
#
# gate1 mix_arena_check：混档几何/热集分配/字节级对拍（10-19/10-20 两次 dn 段偏移雷的门）。
# gate2 serve_paged：serve×分页三场景（冷/Disk 缓存命中/改末句）+ perf 账本计数（10-20 两雷的门）。
# gate3 paged_bail：分页窗口 < 工作集必须响亮报错而非脏值（protect_step 的门）。
set -u
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH="${LINGXI_CUDA_LIBS:-/root/cu12/nvidia/cublas/lib:/root/cu12/nvidia/cuda_nvrtc/lib:${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ART=${ART:-${LINGXI_ARTIFACT:-/models/qwen4_exp-artifact/qwen4_exp.ninfer}}
pass=0; fail=0
ok()  { echo "GATE-PASS  $1"; pass=$((pass+1)); }
bad() { echo "GATE-FAIL  $1 | $2"; fail=$((fail+1)); }

# ── gate1：混档 arena 三层对拍（CPU，分钟级）───────────────────────────────
echo "== gate1 mix_arena_check（混档几何/分配/字节级对拍）"
# 参数：混档 arena（默认取 LINGXI_EXPERT_ARENA 所指）/ 热集 csv / Q4 正本 / Q3 正本
mix_arena=${LINGXI_EXPERT_ARENA:-/dev/shm/qwen4_exp.expert-arena.mix64.bin}
# ninfer 提供量化/布局库（仅 gate1 需要）；python 解释器可 PYTHON= 覆盖
PY=${PYTHON:-python3}
# ninfer 检出位置：可 NINFER_REPO= 覆盖；默认在常见位置里挑第一个存在的
if [ -z "${NINFER_REPO:-}" ]; then
  for d in "$HOME/ninfer" /data/models/ninfer "$HOME/ninfer-dev"; do
    [ -d "$d" ] && NINFER_REPO=$d && break
  done
fi
if PYTHONPATH="$NINFER_REPO" "$PY" tools/golden/mix_arena_check.py \
        "$mix_arena" docs/bench-logs/hist_top64.csv \
        "${Q4_ARENA:-/data/models/lingxi-assets/qwen4_exp.expert-arena.q4g64.bin}" \
        "${Q3_ARENA:-/data/models/lingxi-assets/qwen4_exp.expert-arena.q3g128.gptq.bin}" \
        > /tmp/gate1.log 2>&1 && grep -qa "MIX_ARENA_CHECK PASS" /tmp/gate1.log; then
    ok gate1
else
    bad gate1 "见 /tmp/gate1.log"
fi

# ── gate2：serve × 分页三场景 ──────────────────────────────────────────────
echo "== gate2 serve_paged（三场景 + 账本计数）"
rm -rf /tmp/lingxi-gate2-cache
LINGXI_SERVE_MAX_CTX=32768 LINGXI_QSA_PAGED=1 LINGXI_QSA_WIN_TOKENS=8192 \
    LINGXI_QSA_ADMIT_BATCH=4 LINGXI_SERVE_CACHE_DIR=/tmp/lingxi-gate2-cache \
    ./start.sh ./bin/lingxi-bench serve "$ART" 11451 > /tmp/gate2_serve.log 2>&1 &
srv=$!
for i in $(seq 1 60); do grep -qa "就绪" /tmp/gate2_serve.log && break; sleep 3; done
grep -qa "就绪" /tmp/gate2_serve.log || { bad gate2 "serve 未就绪：/tmp/gate2_serve.log"; }
if grep -qa "就绪" /tmp/gate2_serve.log; then
    ans=$(python3 - <<'EOF'
import json, urllib.request
prompt = open('tests/regression/prompts/needle12k.txt').read()
def ask(p):
    body = json.dumps({"model":"qwen4","messages":[{"role":"user","content":p}],
                       "max_tokens":400,"temperature":0}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:11451/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"}), timeout=900).read())
    m = r["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or "")
r = []
r.append("0.52" in ask(prompt))                       # 冷（写缓存）
r.append("0.52" in ask(prompt))                       # Disk 缓存全命中恢复
r.append("0.52" in ask(prompt + "\n\n再确认：接受率是多少？只答数字。"))  # 边界命中+追加
print(" ".join("T" if x else "F" for x in r))
EOF
)
    [ "$ans" = "T T T" ] && ok "gate2 三场景 needle 全对（$ans）" || bad gate2 "三场景=$ans（期望 T T T）"
    grep -qa "步进失败\|panic\|未物化" /tmp/gate2_serve.log && bad gate2 "serve 日志有错误"
    sinks=$(grep -a paged_sinks run/perf.jsonl 2>/dev/null | tail -1 | grep -o '"paged_sinks":[[:space:]]*[0-9]*' | grep -o '[0-9]*')
    [ -n "$sinks" ] && [ "$sinks" -gt 0 ] && ok "gate2 账本计数 paged_sinks=$sinks" || bad gate2 "账本未计数"
fi
kill -9 $srv 2>/dev/null; wait $srv 2>/dev/null

# ── gate3：窗口不足必须响亮报错（不许静默脏值）────────────────────────────
echo "== gate3 paged_bail（win=2048 << 14.5K 工作集）"
if LINGXI_QSA_PAGED=1 LINGXI_QSA_WIN_TOKENS=2048 LINGXI_QSA_ADMIT_BATCH=4 timeout 900 \
    ./start.sh ./bin/lingxi-bench talk "$ART" "$(cat tests/regression/prompts/needle12k.txt)" 8 \
    > /tmp/gate3.log 2>&1; then
    bad gate3 "窗口不足却正常退出（应响亮报错非零）"
else
    grep -qa "常驻窗口耗尽" /tmp/gate3.log \
        && ok "gate3 响亮报错（常驻窗口耗尽，退出码非零）" \
        || bad gate3 "挂了但报错不对：/tmp/gate3.log"
fi

echo "== gates：$pass 过 / $fail 挂"
[ "$fail" = 0 ]
