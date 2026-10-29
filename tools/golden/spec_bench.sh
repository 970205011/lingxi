#!/usr/bin/env bash
# 投机 decode 计时尺（与 bench.sh 同纪律：走 start.sh 全量最优参数 + 多轮取最好）。
#
# ⚠️ 分工（2026-10-22 划定，别再混）：
#   本脚本 = 生产投机吞吐（serve/talk SpecRuntime 环路，LINGXI_TALK_SPEC=1）。
#   草稿接受率/逐位诊断（mtp-accept 口径）→ **spec_ab.sh**。
#   AR 基线 p10 → **bench.sh**；一键全套 → **perf_suite.sh**（唯一权威入口）。
#
# 为什么单独一把尺：投机吞吐 = 聚合 tok/s（总 tok / 总墙钟），与 AR 的 p10 口径
# 不同（见 docs/longctx-decode.md §3.8 测量口径纪律）；bench.sh 解析不了 [talk-spec] 行。
#
# 用法：
#   bash tools/golden/spec_bench.sh <tag> <tokens> [R] ["prompt"]
#   bash tools/golden/spec_bench.sh france 128 3                      # 默认 France 提示
#   bash tools/golden/spec_bench.sh code 128 3 $'def quicksort(arr):\n    if len(arr) <= 1:'
set -u
cd /models/lingxi
TAG=${1:?用法: spec_bench.sh <tag> <tokens> [R] [prompt]}
N=${2:-128}
R=${3:-3}
PROMPT=${4:-"The capital of France is"}
ART=${LINGXI_ARTIFACT:-/models/qwen4_exp-artifact/qwen4_exp.ninfer}

# 2026-10-23 稳定最优组合（全部实测过，见 docs/spec-verify-kernel-campaign.md §K0e）：
#   LINGXI_ROUTE_FULLDEV=1  A 方案设备全路由：与 legacy **逐位等价**（shadow 全 0），
#                           实测 59.9 vs 59.2 tok/s（+1%）；可被外部 env 覆盖。
#   kd=7 / min_p=0.7 为默认（2026-10-25 改，docs §四十六：全驻留后每步边际成本 5.1→3.5 ms，
#   最优点从 kd=5 右移到 kd=7；容器档 91.9 → 96.0 tok/s、宿主档 88.6 → 94.9，md5 不变；
#   kd=8/9/10 回落到 85.6~88.2）。min_p 0.85 会掉到 AR（勿用）。
#
# ★ 2026-10-25 两个新默认（都在 start.sh 里，本脚本不用重复设）：
#   `LINGXI_ARENA_PRELOAD_ALL=1`  启动时把本卡服务层**全部专家**搬进显存（幂等、池满即停）。
#       实测：decode 期首触 DMA 摊 4.4 ms/轮 ⇒ 全驻留后 miss 713 → 25、轮 p50 56 → 51 ms、
#       **200 token 84~86 → 93~95 tok/s**，文本 md5 逐位不变；代价启动每卡 +1.8~2.9 s。
#   `LINGXI_VERIFY_HEAD_SPLIT=1`  verify 的 lm_head 按词表行二分到两卡（GPU1 那时空闲）。
#       +2.7%（off 82.5/82.9 vs on 85.9/83.9），逐位不变（`gemm_q4_skinny_bf16` 无 split-k）。
#   当前基线（干净态，2026-10-29 重立）：**200 token 90.8~94.0 tok/s（最好 94.0）**、42 轮、
#   接受 3.81/轮。基线配置栈 = PAGED=1 + REPIN=30000 + SHARED_Q4=1 + PRELOAD_ALL=1 +
#   槽 8000/8400 + kd=7/min_p=0.7（全部 start.sh 默认，零额外 env）。
#   ⚠️ md5 口径修正：整日志 md5 **无效**——REPIN 重钉行与逐轮进度行的**行数**随跑变化
#   （接受轮数不同 ⇒ 进度行数不同），但解码文本逐位一致。正确取法：
#     sed -n '/-- 生成/{n;p}' /tmp/spec_<tag>_r*.log | md5sum
#   当前文本 md5 `193caec1`（与 62b508f8 不同是预期的：SHARED_Q4 + 分页封印改了数值路径，
#   贪心序列在新栈下重新收敛，判据是"同配置多次跑一致"而非"与旧 md5 相同"）。
#   ⚠️ 不要开 LINGXI_Q4GEMM_S8（实测 75→101ms/轮）、不要开 VERIFY_CHUNKS=2（71→107ms/轮）、
#      不要开 LINGXI_Q4TC_M8（M≤8 MMA 稠密投影：实测 61.5→55.5 tok/s，见 docs §K2''）。
#   ⚠️ 2026-10-25：**不要开 LINGXI_MTP_ATTN_Q4**（草稿注意力量化：接受率 3.26→3.04，
#      200 token 实测 78.4/79.0 → 70.3/68.6 tok/s，**−10%**，docs §三十三 D）。
#      草稿相位唯一安全的非逐位是 head，且两档都已是**默认**：
#        `LINGXI_DRAFT_HEAD_FAST=1`   lm_head 换 16B/lane（层外 1171.7→924.5 µs）
#        `LINGXI_DRAFT_HEAD_SPLIT=1`  再把 lm_head 行二分到两张卡（924.5→716.4 µs）
#      `=0` 各自回退；主干 verify 的 head 不受影响（逐位）。
#   ⚠️ 改完代码必须 `make build`（= cargo build + cp 到 bin/），否则本脚本跑的还是旧二进制。
#   ⚠️ 看 `draft-prof-head` 必须**只取 t1 步**（`awk '$3=="t1"'`）并丢 >5 ms 的离群，
#      否则 prime/验证行的异常步会把中位抬高 200 µs（docs §三十三 C 的口径修正）。
#   ⚠️ 启动偶发 OOM 已根治（docs §三十八/§四十一 L）：槽位分配 OOM 会先等 400 ms 重试 3 次
#      再按 −6.25% 退让。看到 `[arena] ⚠️` 不是故障，是预算紧了一次。
#   ⚠️ 判 tok/s **只看逐轮分布**（本脚本已打印 p50/max 并自动作废离群跑）：约 25% 的跑会出现
#      **某一轮单点软卡 3.3 s**（把平均从 95 拉到 35），根因在宿主内存形态（Shmem 185G/251G、
#      无 swap），引擎侧修不掉 —— 见 docs §四十三 C。
#   ⚠️ A/B 前先确认两侧**生效值**不同：`start.sh` 里不少开关是 `${VAR:-1}`（默认开），
#      "不设 env" ≠ "关"。踩过一次（TC：both on 被读成"中性"），正确写法是实验组显式 `=0`。
export LINGXI_ROUTE_FULLDEV=${LINGXI_ROUTE_FULLDEV:-1}
export LINGXI_SERVE_SPEC_KD=${LINGXI_SERVE_SPEC_KD:-7}
export LINGXI_SPEC_MIN_P=${LINGXI_SPEC_MIN_P:-0.7}

rm -f "/tmp/spec_${TAG}_toks.txt"
for r in $(seq 1 "$R"); do
    OUT="/tmp/spec_${TAG}_r$r.log"
    LINGXI_TALK_SPEC=1 ./start.sh ./bin/lingxi-bench talk "$ART" "$PROMPT" "$N" > "$OUT" 2>&1
    RC=$?
    LINE=$(grep -a "talk-spec] 共" "$OUT" | tail -1)
    if [ "$RC" -ne 0 ] || [ -z "$LINE" ]; then
        echo "$TAG  r$r  失败（RC=$RC）—— 尾部日志："
        tail -3 "$OUT"
        continue
    fi
    TPS=$(echo "$LINE" | sed -E 's/.*= ([0-9.]+) tok\/s.*/\1/')
    ACC=$(echo "$LINE" | sed -E 's/.*接受 [0-9]+（([0-9.]+)\/轮）.*/\1/')
    ROUNDS=$(echo "$LINE" | sed -E 's/.*｜([0-9]+) 轮｜.*/\1/')
    # ★ 离群轮体检（2026-10-25 加，docs §四十一 F）：一轮软卡 3.4 s 就能把 200 token 的
    #   平均从 85 拉到 35（正好 2.4×），而**逐轮中位完全正常** —— 之前的"机器双峰态"
    #   就是这么误判出来的。根因是**上一个进程把 GPU 上下文搞崩**（dmesg 的 Xid 13），
    #   崩溃后的下一次跑会中途卡一次做恢复。判据：max 轮 > 3× 中位轮 ⇒ 本跑不可用于 tok/s。
    MS=$(tr '\r' '\n' < "$OUT" | grep -a -oE '[0-9.]+ ms/轮' | sed 's/ ms.*//' | sort -n)
    NMS=$(echo "$MS" | grep -c .)
    if [ "$NMS" -ge 5 ]; then
        MED=$(echo "$MS" | awk -v n="$NMS" 'NR==int((n+1)/2)||NR==int((n+2)/2){s+=$1;c++} END{printf "%.0f", s/c}')
        MAX=$(echo "$MS" | tail -1)
        if awk -v a="$MAX" -v b="$MED" 'BEGIN{exit !(a > 3*b)}'; then
            printf "%-16s r%s  %6s tok/s｜接受 %s/轮｜%s 轮  ⚠️ 含离群轮（max %.0f ms ≫ 中位 %.0f ms）⇒ **不计入最好**\n" \
                "$TAG" "$r" "$TPS" "$ACC" "$ROUNDS" "$MAX" "$MED"
            printf "     提示：先 `dmesg | grep -c Xid` 看有没有新的 Xid 13（上一个进程崩溃的余波）\n"
            continue
        fi
        printf "%-16s r%s  %6s tok/s｜接受 %s/轮｜%s 轮｜轮 p50 %s / max %s ms\n" \
            "$TAG" "$r" "$TPS" "$ACC" "$ROUNDS" "$MED" "$MAX"
    else
        printf "%-16s r%s  %6s tok/s｜接受 %s/轮｜%s 轮\n" "$TAG" "$r" "$TPS" "$ACC" "$ROUNDS"
    fi
    grep -a "专家槽：" "$OUT" | sed 's/^ */      /'
    echo "$TPS" >> "/tmp/spec_${TAG}_toks.txt"
done
if [ -s "/tmp/spec_${TAG}_toks.txt" ]; then
    BEST=$(sort -rn "/tmp/spec_${TAG}_toks.txt" | head -1)
    printf "%-16s **最好 = %.1f tok/s**（AR 基线见 bench.sh，口径不同勿直接比）\n" "$TAG" "$BEST"
fi
rm -f "/tmp/spec_${TAG}_toks.txt"
