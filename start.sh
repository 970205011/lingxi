#!/usr/bin/env bash
# ============================================================================
# 灵犀引擎生产启动器（外挂最优 env，2026-10-17 ㊶）
#
# 用法：
#   ./start.sh                    # 默认起 talk 交互（英文冒烟）
#   ./start.sh talk <产物> <提示词> [N]   # 透传给 lingxi-bench
#   ./start.sh <任意命令>         # 在固化 env 下执行任意命令
#
# 设计原则（用户点名）：
#   **最优启动参数只存在于这一个文件**，别散落在历史命令/文档里。
#   新增优化开关的落地流程：代码默认关（env 门控）→ 实测验证 → 把开关
#   加进下面的"已验证优化"段 → 更新本注释。回退 = 注释掉那一行。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ── 基础生产路径（141 容器内）────────────────────────────────────────────
# ㊿：必须 ${VAR:-default} 形式——无条件 export 会把调用方的显式指定吃掉
# （2026-10-19 实测：回归传 Q3 arena 被这里覆盖回 Q4 路径）。
export LINGXI_EXPERT_ARENA="${LINGXI_EXPERT_ARENA:-/dev/shm/qwen4_exp.expert-arena.mix64.bin}"  # ㊿：混档（热Q4/冷Q3，6.2-2）2026-10-20 起为默认；正本在 /data/models/lingxi-assets（tmp 是自动清理区，勿放持久资产）
# §AI（2026-10-26）：ct141 权威测试环境的大文件经宿主 /dev/shm bind 到 /mnt/hostshm
#（宿主 RAM 只有 251G，扛不住第三份 77GB 副本；配置在 /etc/pve/lxc/141.conf）。
# 容器内 /dev/shm 没有副本时自动回落到共享挂载，并同步指 PLE 表（引擎最高优先读 LINGXI_PLE_TABLE）。
if [ ! -f "$LINGXI_EXPERT_ARENA" ] && [ -f /mnt/hostshm/expert-arena.mix64.bin ]; then
    export LINGXI_EXPERT_ARENA=/mnt/hostshm/expert-arena.mix64.bin
    export LINGXI_PLE_TABLE="${LINGXI_PLE_TABLE:-/mnt/hostshm/ple-table.iq4nl.bin}"
fi
# ㊿-P0（2026-10-19）：槽数默认值移交 vram-autotune（main.rs autotune_arena_slots，
# 启动时按 权重+KV+GDN+预留 逐卡算账）。这里不再写死默认——显式 export 会**屏蔽** autotune。
# 要手工指定：LINGXI_ARENA_SLOTS=<n> LINGXI_LAYER_SPLIT_SLOTS=<n> ./start.sh ...（透传生效）；
# 整体关 autotune：LINGXI_NO_VRAM_AUTOTUNE=1（回落 talk/serve 路径内建默认 7000/4000/9600）。
export LINGXI_ARENA_PIN_GIB="${LINGXI_ARENA_PIN_GIB:-64}"  # ㊿：${VAR:-default} 形式，同 L19 教训
# 注：不设该变量时引擎按介质自动判定（RAM 盘整段 pin / 非 RAM 一律禁止）。
# 生产机 arena 在 /dev/shm，显式写 64 只为锁定行为；开放环境可自行 `LINGXI_ARENA_PIN_GIB=0`。
# CUDA 运行库（cublas/nvrtc）：本机 pip 版 CUDA 树不在 ld 默认路径，cudarc 动态加载需要
# （2026-10-21：从 gates.sh 收编——此前 serve 经 start.sh 起会死在这，只有 gates 能跑）。
# 可用 LINGXI_CUDA_LIBS 或 CUDA_HOME 覆盖；不存在的目录放进 LD_LIBRARY_PATH 无害。
# §AI：ct141 没有 /root/cu12，回落共享盘自洽的 CUDA 12.4 树（cublas+nvrtc 同版本成套）
if [ -z "${LINGXI_CUDA_LIBS:-}" ] && [ ! -d /root/cu12/nvidia/cublas/lib ] && [ -d /models/cuda-12.4/targets/x86_64-linux/lib ]; then
    LINGXI_CUDA_LIBS="/models/cuda-12.4/targets/x86_64-linux/lib"
else
    LINGXI_CUDA_LIBS="${LINGXI_CUDA_LIBS:-/root/cu12/nvidia/cublas/lib:/root/cu12/nvidia/cuda_nvrtc/lib:${CUDA_HOME:-/usr/local/cuda-12.8}/targets/x86_64-linux/lib}"
fi
export LD_LIBRARY_PATH="${LINGXI_CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LINGXI_Q4_CACHE=/dev/shm/lingxi-q4cache              # W4 缓存：启动 40s → 4.4s
# PLE 表：默认自动解析 shm IQ4_NL(26.8GiB) > shm FP8 > 磁盘 bf16（响亮兜底横幅）；
# 想强制某档用 LINGXI_PLE_TABLE=/path 覆盖。

# ── 稠密投影档 + 投影 TC + 设备路由（2026-10-27 引擎内默认值，这里不再 export）──
# ⚠ 这三个是**值感知**开关（lingxi_model::flag_val）：未设 = 引擎默认；
#   =0 关 / =1 开都真生效。start.sh **不要**无条件 export——会把外部的 =0 顶掉
#   （㊿ 教训；且"存在即真"的老 flag() 语义下 export X=0 反而是开，2026-10-27 踩过）。
# 引擎默认（值感知）：**Q4 三档开**（生产档，判据链全在此）+ PROJ_TC 开 + BIG 开。
# bf16 质量模式：     LINGXI_DENSE_Q4=0 LINGXI_DENSE_Q4_QSA=0 LINGXI_HEAD_Q4=0 ./start.sh ...
#                     （PROJ_TC 自动生效；nvprof 实测 GPU 时间 +22%——字节账是实的，
#                      墙钟被双峰噪声掩盖，§AP 翻案记录见 HANDOFF §5）
# 关投影 TC：         LINGXI_PROJ_TC=0 ./start.sh ...（仅 bf16 档有意义，Q4 档本就 TC）
# 关设备路由 BIG：    LINGXI_ROUTE_FULLDEV_BIG=0 ./start.sh ...

# ── 双卡 layer-split：引擎默认自动（lo=24，自动关图），无需手设 ────────────
# 回归对照才用 LINGXI_LAYER_SPLIT=0。

# ── 已验证优化开关（bit-safe 对拍过，撤掉=注释行）─────────────────────────
# prefill 按卡分线程（㊿-①）：GPU0 跑 0..24、GPU1 跑 24..48，chunk 间用 mpsc 交接。
# 判据（2026-10-18，在 ㊿-② 内核修复后重测）：48token 首 token 11751 + 贪心序列不变；
# 867token 与单卡逐位一致（96133）；4167token 顺序 46.2s → 线程 30.3s（−34%）。
# 前提：provider 必须是缓存型（sparse 路径的 DeviceLayerCache），否则会显式报错。
export LINGXI_PREFILL_THREADS=1
# 张量核专家段（㊼）：**只在投机验证批（1 < rows ≤ 16）生效**（moe.rs 里已按 rows 门控）。
# t=7 验证轮专家段 0.59→0.55 ms/层、整轮 −5%（2026-10-18 实测）；
 # t=1 顺序 decode 仍走 SIMT（t=1 用 TC 反而 −3%）。
# ㊽ 2026-10-19：prefill 大 tile 档（rows>16）也吃这个开关 ⇒ 保留默认 1，但改成可外部覆盖
#（`LINGXI_MOE_TC=0 ./start.sh ...` 做 SIMT 对照；之前无条件 export 会把外部 0 覆盖回 1，
#  导致一轮 A/B 实际两边都是 TC——教训：开关必须 ${VAR:-default} 形式）。
export LINGXI_MOE_TC=${LINGXI_MOE_TC:-1}
# 冷启动钉表（㊿-⑤，2026-10-18 实测）：把 hist 里出现过的专家**全部**钉进显存，
# 消灭 decode 的 PCIe 单次 miss（一次 miss ≈ 2.6MB DMA ≈ 350~450 µs！）。
# 实测（本机 split、talk 16 token）：N=100 30.43 ms/token → **N=200 24.36 ms/token（−20%）**，
# prefill 6.3s → 3.8s；首 token 11751、贪心文本逐位不变。
# N=200 时 hist_spec.csv 的全表 3433 个 (层,专家) 全被钉住（8.3 GiB），LRU 仍剩 5567 槽（62%）。
# 旧注释里"启动多 ~30s"作废：实测灌钉表只 **0.9 s**。
#
# ⚠️ 钉表会占掉 LRU 槽位：在 **9000/9600** 槽下，4167-token prompt 的 prefill 会从 30.4s
# 退化到 36~39s（中间工作集被挤进 thrash）；**9600/10000** 后回到 **28.7s**，decode 仍 24.3ms/token。
# 该槽位档同时容得下 serve 投机：32K ctx + MTP 时 GPU0/GPU1 = 29.9/27.8 GiB（㊿-⑧）。
# 外部显式指定的 PREWARM_N 先存一份：GPUS=1 分支要把默认值从 200 改小（见下方 GPUS=1 段），
# 但下一行的 :-200 一旦执行就丢了"用户有没有显式给"的信息。
_PREWARM_N_GIVEN="${LINGXI_ARENA_PREWARM_N:-}"
export LINGXI_ARENA_PREWARM=${LINGXI_ARENA_PREWARM:-docs/bench-logs/hist_spec.csv}
export LINGXI_ARENA_PREWARM_N=${LINGXI_ARENA_PREWARM_N:-200}
# ★ 2026-10-25（docs §四十三 E）：**启动时把本卡服务层的全部专家一次搬进显存**。
#   实测：decode 期首触 DMA 摊 4.4 ms/轮（`unique` 713 个 × 2.6 MB，30~44 轮里还完），
#   全驻留后 miss 713 → **18**、轮 p50 55→50 ms、**200 token 84~87 → 94.6~95.3 tok/s（+12%）**，
#   文本 md5 逐位不变；代价只是启动每卡 +1.8~2.9 s（tok/s 口径不含启动）。
#   分页（paged）档必须关掉它：`LINGXI_ARENA_PRELOAD_ALL=0`。
export LINGXI_ARENA_PRELOAD_ALL="${LINGXI_ARENA_PRELOAD_ALL:-1}"
# ★ 2026-10-29 转正：在线自适应重钉（f394fd0 实测真语料 +17%：38.1 vs 32.5 tok/s、
#   命中率 84.8→88.2%、逐位判据过）。窗口 30000 = 实测最优档（60000 略退 +14%）。
#   值感知默认：显式 LINGXI_ARENA_REPIN=0 可关。
export LINGXI_ARENA_REPIN="${LINGXI_ARENA_REPIN:-30000}"
# ★ 2026-10-29 prefill 许可模式转正：QSA 分页默认开——prefill 期弹性全驻留（读时尖峰
# 不可避免：O(N²) 注意力读取必须走 HBM），全部喂完"封印"整体下沉 host 锁页池（显存
# 回落，腾给专家槽）；decode 只驻留窗口+选中块，冷块 PCIe 流取（流量 ≪ 专家 DMA）。
# 生产验收：58K 文档 needle 命中、prefill wall 与长文档口径吻合、二轮前缀复用 TTFT 7.7s。
# 值感知默认：显式 LINGXI_QSA_PAGED=0 可关（R5 gate2/gate3 守门）。
export LINGXI_QSA_PAGED="${LINGXI_QSA_PAGED:-1}"
export LINGXI_QSA_WIN_TOKENS="${LINGXI_QSA_WIN_TOKENS:-32768}"
# （暂无默认开启项；LINGXI_MOE_OVERLAP=1 为冷启动场景可选，见 PLAN ㊶）

# ── 有损优化开关（㊶ 用户已批：env 门控 + 质量验证；合格前**保持注释**）───
# export LINGXI_KV_K8V4=1      # ❌ 2026-10-18 复核：**质量不过判据，禁止开**
                               #   短 prompt 首 token 1543（应 11751）、文本 `XXX XXXX`；
                               #   8K 上下文首 token 108559、64 token 全乱码；dense 路径同样失败
                               #   ⇒ 病根在 E4M3 量化本体。且 decode 零收益（8.2K 下 on/off 都 97ms/token），
                               #   prefill 还 +13%。KV 字节不是当前瓶颈，256K 用 bf16 即可（7200/7600 槽装得下）。
                               #   ⚠️ 另：与 LINGXI_ATTN_GATHER 互斥（gather 只支持 bf16 KV）。
# export LINGXI_GDN_REGS=1     # ㊶ 已撤销：实测天花板 2-4%（GDN 递推只占 token 3-6%），核已删
# export LINGXI_Q4_HMMA=1      # ㊶ N/A：HMMA 需 T≥16，spec 窗 t=7 够不到

# ── 启动 ──────────────────────────────────────────────────────────────────
# **裸 `./start.sh` = 直接起生产服务**（OpenAI 兼容 /v1/chat/completions + SSE，
# 固定端口 8000、上下文 256K，均为默认可覆盖：PORT= / MAX_CTX= ./start.sh）。
# 带参数则原样透传（talk/bench 等子命令不变）。
# v1.5（2026-10-20）：默认**并发路径**——连接线程 + 引擎线程轮转步进，多会话真并发
#   （无队头阻塞，单流时间片）；`LINGXI_SERVE_MATERIALIZED`（默认 2）= 同时物化会话数。
#   请求体 session_id（回落 user）选会话；同会话并发请求排队保序。
#   投机 LINGXI_SERVE_SPEC=1 时整体走串行路径（MTP 草稿态绑会话，与并发调度不兼容）。
# 容量用 MAX_CTX（默认 262144=256K）、单请求默认生成长度 LINGXI_SERVE_MAX_NEW（默认 512）。
#
# **部署（单卡 × 2 并发）**：`bash tools/deploy/up.sh` ⇒ 主入口 **:8000**（轮询代理），
# 两个单卡实例 :8001/:8002；`bash tools/deploy/down.sh` 停。槽位/上下文用
# `CTX=32768 GPU0_SLOTS=9000 GPU1_SLOTS=9000` 覆盖。
#
# **256K 配方**（2026-10-18 实测）：bf16 KV（**不要**开 K8V4：与 gather 互斥、质量不过判据）
#   + 智能前缀缓存默认开；大上下文吃显存 ⇒ 槽位让位（>64K ctx 时自动落 7200/7600 档，
#   可 SLOT_RECIPE=0 关闭交给 autotune）；47K 预填峰值 27.7/26.8GiB（32G 卡留 ~4GiB）；
#   冷预填 47K ≈ 286s（9600 档）/~457s（7200 档）、250K ≈ 3437s；**重复/续聊秒级**
#   （250K 全量重发 TTFT 0.00s、整请求 6.4s——前缀缓存复用 250552 tok）。
#
# 投机解码（㊿-⑦，2026-10-18）：
#   LINGXI_SERVE_SPEC=1 ./start.sh ./bin/lingxi-bench serve <产物> 11434
#   - MTP 产物默认 /data/models/lingxi-assets/qwen4_exp.mtp.fixed.ninfer（LINGXI_MTP_ARTIFACT 覆盖）；
#   - MTP 专家默认 Q4（1.3 GiB；LINGXI_MTP_Q4_EXPERTS=0 回 bf16 5 GiB，GPU0 会顶满）；
#   - 窗口/闸门默认 kd=7 + min_p=0.7（LINGXI_SERVE_SPEC_KD / LINGXI_SPEC_MIN_P）；
#     kd=7 是 2026-10-25 复扫结果（docs §四十六：全驻留专家 + 草稿 head 二分把每步边际成本
#     压到 3.5 ms ⇒ 最优点从 5 右移到 7；容器档 91.9 → 96.0 tok/s，md5 逐位不变）。
#   - 实测：13-token prompt 解码 **89 tok/s**；875-token + 128 token **38.4 vs 29.9 tok/s**，文本逐字相同；
#   - ⚠️ 投机 + 32K ctx 要降槽位：**LINGXI_ARENA_SLOTS=8800 LINGXI_LAYER_SPLIT_SLOTS=9200**
#     （9600/10000 下 875-token 的 prefill 会 OOM；8800/9200 峰值 31.3/28.6 GiB）；
#   - 与 LINGXI_SERVE_PREFIX_CACHE 互斥（spec 要整段 prompt 的 h，强制全量 prefill）；
#   - prefill 与自回归同为线程版（TTFT 875-token 7.68s vs 7.58s）；
#   - 首个请求把 48 层装进 DeviceLayerCache（TTFT ~4.4s，一次性）；prime 本身 0.04s(13tok)/0.24s(875tok)。
if [ $# -eq 0 ]; then
    ART=${LINGXI_ARTIFACT:-/models/qwen4_exp-artifact/qwen4_exp.ninfer}
    [ -x ./bin/lingxi-bench ] || { echo "未找到 ./bin/lingxi-bench：先跑 make build" >&2; exit 2; }
    [ -f "$ART" ] || {
        echo "找不到产物：$ART" >&2
        echo "用 LINGXI_ARTIFACT=/path/model.ninfer 或 ./start.sh talk <产物> 指定；从零构建见 tools/convert/hf2lingxi.py" >&2
        exit 2
    }
    [ -f "$LINGXI_EXPERT_ARENA" ] || {
        echo "找不到专家 arena：$LINGXI_EXPERT_ARENA" >&2
        echo "用 LINGXI_EXPERT_ARENA=/path/expert-arena.bin 指定（建议放 /dev/shm）；生成见 tools/arena-gen/" >&2
        exit 2
    }
    # ── 单卡档（GPUS=1，2026-10-28 实测基线）──────────────────────────────
    # 引擎默认自动双卡 layer-split（生产档）。GPUS=1 显式关掉、只用 GPU0：
    #   GPUS=1 ./start.sh            # 单卡 256K 服务
    # 实测账（ct141 单 V100 32G，MAX_CTX=262144，autotune 槽 3072Q4+4128Q3
    # ≈14.9GiB 专家上卡，显存 22.4GiB）：
    #   - AR 模式（SPEC 不开）：能服务，decode 8~10 tok/s（专家冷 miss 主导，
    #     双卡档同口径 60~90）；
    #   - SPEC=1：MTP ~2.7GiB 压顶，autotune 预算装不下会话工作区 ⇒ 准入 OOM
    #     （单卡 256K+投机 = 配方里程碑，见 HANDOFF §4.3）；
    #   - 多会话/千 token 级 prompt 在 22.4GiB 基线上余量不足 ⇒ 建议单卡档先
    #     配 MAX_CTX=65536（KV 缩到 ~1.8GiB）+ 弹性分页 KV（LINGXI_QSA_PAGED=1）。
    # 相关开关：LINGXI_ARENA_PRELOAD_ALL=0（全驻留装不下，关掉走 LRU+预取）、
    #          LINGXI_ARENA_PREFETCH=1（router lookahead 预取，实测本模型跨层
    #          命中率仅 2.4% ⇒ 自适应休眠，默认关——见 HANDOFF 负结果档）。
    if [ "${GPUS:-2}" = "1" ]; then
        export LINGXI_LAYER_SPLIT=0
        # 强制关全驻留：上面 L94 已把未设值默认成 1，这里 :-0 是死代码——
        # 单卡装不下全驻留（6659 个 Q3 ≈ 12.8GiB 会挤掉会话工作区），强制 0。
        export LINGXI_ARENA_PRELOAD_ALL=0
        # 单卡钉表档（2026-10-28）：双卡默认 N=200 会把单卡全部 7200 槽（3072 Q4 +
        # 4128 Q3 ≈ 14.9 GiB）钉满 7212 项（LRU≈0 + "槽不够"告警），prefill 工作集
        # thrash——双卡 9000 槽档的已知病灶在单卡必然命中。单卡默认改 N=64：恰好
        # 钉住 mix64 的 Q4 热集 3072 个（每层 top-64 ≈ 7.5 GiB），Q3 的 4128 个 LRU
        # 槽全留给长尾。表本身不用换：layer-split 不改路由数学（双卡验收过逐位一致），
        # hist_spec.csv 的层内专家分布与卡数无关（依据见 docs/HANDOVER.md §9）。
        # 覆盖方式：LINGXI_ARENA_PREWARM_N=<n> GPUS=1 ./start.sh ...（显式值优先）。
        export LINGXI_ARENA_PREWARM_N="${_PREWARM_N_GIVEN:-64}"
    fi
    # 大上下文显存配方：>64K 时专家槽让位给 KV。2026-10-29 起分页封印（超窗 KV 下沉
    # host 池、显存回落 ~3-4.5GiB/卡）⇒ 槽位上调一档（8000/8400，实测 58K 封印请求 +
    # 250K 峰值余量 OK；可 SLOT_RECIPE=0 关）
    MAX_CTX="${MAX_CTX:-262144}"
    if [ "$MAX_CTX" -gt 65536 ] && [ "${SLOT_RECIPE:-1}" = "1" ]; then
        export LINGXI_ARENA_SLOTS="${LINGXI_ARENA_SLOTS:-8000}"
        export LINGXI_LAYER_SPLIT_SLOTS="${LINGXI_LAYER_SPLIT_SLOTS:-8400}"
    fi
    # 投机档（SPEC=1）：v1.5 并发 + 双卡草稿 head 二分；多会话轮转、各请求草稿状态隔离。
    # 2026-10-27 起 256K 实测打通（draft KV 单层 ~0.6G + MTP ~2.7G，配方见下）⇒ 不再降 32K。
    if [ "${SPEC:-0}" = "1" ]; then
        export LINGXI_SERVE_SPEC=1
    fi
    # 分块 prefill（2026-10-28 decode 优先 + 防饿死 + 自适应 burst 调度版，默认开 512）：
    # 大 prompt 切块喂，decode 优先穿插——**对外服务口径：一个大文档不拖死其他客户**。
    # 实测账（119K 文档 + 4 并发小请求，GPTQ arena）：小请求 7.7~21.9s（不开分块要
    # 干等大请求 578s+），大请求 650.9s（+8.5% vs 整块）——两个优势同时拿到。
    # 单租户压测场景可 CHUNK=0 关回整块（纯单流最快）。burst 单步硬上限 4096
    # （LINGXI_SERVE_PREFILL_BURST_MAX）：单步不可抢占，burst 块不钳会队头阻塞。
    export LINGXI_SERVE_PREFILL_CHUNK="${LINGXI_SERVE_PREFILL_CHUNK:-512}"
    export LINGXI_SERVE_MAX_CTX="$MAX_CTX"
    # 并发物化保持引擎默认 2：会话 KV 预算是**按实际用量**核的（prompt+max_tokens，非按
    # max_ctx 顶格），两份超限会走 429 排队重试（已验证语义），不需要强行单驻留。
    export LINGXI_SERVE_MATERIALIZED="${LINGXI_SERVE_MATERIALIZED:-2}"
    PORT="${PORT:-8000}"
    exec ./bin/lingxi-bench serve "$ART" "$PORT"
fi
exec "$@"
