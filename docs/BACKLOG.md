# 灵犀引擎 BACKLOG（全量审计 · 2026-10-18）

> **这份文件解决什么问题**：仓库里已有 268 个 commit、4105 行 `PLAN.md`、6 份设计与审计文档、21 份 bench 日志。
> 为了避免“把前人做过/否决过的东西再试一遍”，这里做一次**全量去重**：
> - §1 已落地（**不要再做**，只列结论与证据）
> - §2 已否决（**不要再试**，附实测原因）
> - §3 未做/半做（**准备做**，每条给做法、依赖、验收判据、收益估计）
> - §4 建议顺序 / §5 需要用户拍板的清单
>
> 审计来源：`PLAN.md` 全文、`HANDOVER.md`、`speedup-audit.md`、`decode-speed-profile.md`、
> `weight-compression-plan.md`、`inference-framework-design.md`、`strata-mtp-expert-cache-plan.md`、
> `docs/bench-logs/*`、`git log --all`（268 commits + 1 stash）。
> 判据口径（不变量）：首 token `11751`；固定贪心序列 `13,561,6511,314,9564,369,19241,13`；
> 长 prompt `867→96133`、`4167→332`；任何新开关必须登记 `probes.rs`。

---

## 0. 结论摘要

1. **单流的“便宜项”已经挖完**：专家核到 memcpy 的 84%、launch 融合零收益、整轮图零收益、块级只剩 ~3%。
   剩余单流空间集中在两处：**去掉每层 MoE 的 host 同步（~3.9ms/token）** 与 **投机草稿质量/树**。
2. **最大结构性杠杆是并发**：单流 decode 两卡各有一半时间在等（13.3+13.1ms 串行）。
   ninfer 的 4 路聚合 94.2 t/s vs 单流 26.9（3.5×）已经证明；Strata 的 `--batch-groups` 还允许两卡各接一批。
3. **长上下文的瓶颈不是容量而是 prefill**（**2026-10-18 已大幅改善，见 §1.6**）：
   原来 23.7K 140s / 27.8K 188s / 47K 539s，47K→256K 外推约 4.5 小时（O(S²)，实测曲线吻合）。
   今天三个改动把 prefill 的注意力与选块拉成线性/近线性：
   **① 选块从 O(nblocks²) 全排名改成 O(nblocks) 基数选择；② 池化批量 launch；
   ③ 稀疏注意力改成"选中块 gather"（每行恒定 2048 token）。**
   实测：8.3K 89.3→48.8s（1.83×）、30.9K 208.4→166.1s、47K 539→285.8s（同槽位口径）。
   **256K 实测（250,552 token）冷 prefill 3436.9s = 57.3 分钟**（无 OOM，首 token 248068）；
   同 prompt 第二次 **TTFT 0.00s、整请求 6.4s（只剩 decode）**。
   按 47K→256K 实测（457s→3437s，token ×5.5、时间 ×7.5）**只剩轻度超线性（~1.37×）**，
   主体已经是线性项（每 chunk 的专家换入 + 每行计算）；配方与证据见 §1.6。
4. **服务层已补齐到可直接接 agent**（2026-10-18，见 §1.7）：
   OpenAI chat 兼容（`reasoning_content`、`tool_calls`、流式/非流式、`max_completion_tokens`、
   `enable_thinking`/`reasoning_effort`）、UTF-8 增量反分词、末尾 chunk 修复；
   前缀缓存支持**不活跃会话整段换出到 host**（多会话、LRU、默认 32 GiB 预算），
   显存只留当前会话，下次命中再拉回。

---

## 1. 已落地（不要再做）

### 1.1 权重 / 量化 / 核
| 项 | 结论与证据 |
|---|---|
| W4 三档（dense / QSA / head） | p10 −10.2%，贪心序列逐位不变（§3.49，commit `9168468`） |
| Q4 权重按张量名落盘缓存 | 启动 40.1→4.4s（§3.50） |
| 专家 Q4G64 arena（60.25GiB，48×512） | blob 2.61MB vs bf16 19.66MB（7.53×），逐专家误差 0.108–0.124（weight-compression §11） |
| `slotsm` 按唯一专家分组（一次权重喂 m token） | m=2 达 memcpy 的 84%（318/378.6 GB/s）（§㊿-④） |
| `slotsm` 每组 8 对截断修复 + MB=8 分批 | 长 prompt 首 token 漂移根因；四配置全 = 96133（§㊿-②，`fe9d0ce`） |
| 专家核 nibble→float shared LUT | gu 347→439、dn 263→315 GB/s（§3.32） |
| 专家核预取深度 4 | 专家段 16.6→7.6ms（§P15-㊸） |
| 稠密 Q4 单行/warp（非 4 行） | qkvz 369→482、qsa 354→512 GB/s（§3.33） |
| Q4 瘦长 GEMM（分块、直接吃 Q4） | t=8 157.7→99.6ms（§3.23/3.24） |
| V100 张量核 `mma.sync.m8n8k4`（专家，1<rows≤16） | t=7 验证 −3%，48token 逐位一致（§P15-㊼，`abf302f`） |
| x-fp32 预转（slotsm） | m=7 +81%；真实 m≈2.3 仅 +0.5%（§P15-⑤c） |
| 双输出 qkv/z 融合 | **零收益**（155 vs 148µs，整机无变化）→ 停放（§㊿-⑥） |

### 1.2 专家驻留 / arena
| 项 | 结论与证据 |
|---|---|
| 静态钉热专家（PREWARM） | p10 −4.2%、p90 −15%、mean −10%（§P1.2） |
| 自产直方图 + 在线重钉（REPIN） | 命中 84.8→88.2%，32.5→38.1 tok/s（§P11.5） |
| 钉表转生产默认（hist_spec + N=200） | p90 36.84→24.82ms（−32.6%）、mean −21.4%（§㊿-⑤，`ad19954`） |
| 槽位 9600/10000 | 4K prefill 36~39s→28.7s，decode 仅 −2%（§㊿-⑧，`ad3b1e8`） |
| 槽位 flat 直查表（HashMap→Vec） | ~30–50µs（§P15-㊹ B3） |
| `load_hits` 微优化 | 46–61→41–54µs/层（§P15-⑱） |
| MTP 专家 Q4 小 arena | 省 3.44GiB，槽 7000→9000（§P11.3） |

### 1.3 attn / GDN / PLE
| 项 | 结论与证据 |
|---|---|
| GDN 递推 16 批量预载 | 35.2→18.0µs，整机 +5%（§P15-㉘） |
| GDN 卷积 4 launch 融 1 | 28–30→19.7µs（§3.35/3.37） |
| attn3 行块平铺（R=4/KPF=4） | 32K 551→271ms；in-situ 29.2→18.7ms/层（§P15-㊳） |
| 跳块稀疏 attn5/attn2s | 稀疏≡密集逐字节；K 流量 ÷12（§P15-㊵） |
| KV E4M3（K8V4） | 256K KV 6.3→3.15GB；prefill +22%（§P15-㊶）。**⚠️ 2026-10-18 复核：质量不过判据**——8K 上下文首 token 108559（应 248068）、64 token 全是乱码；短 prompt 首 token 1543（应 11751）、文本 `XXX XXXXXXXXXX`；`LINGXI_QSA_DENSE=1` 同样失败 ⇒ 病根在量化本身（不是稀疏核）。**默认必须关**。**decode 也无收益**：8.2K 上下文 64 token 实测 on/off 都是 ~97ms/token（选块+扫描是主项，KV 字节不是）。 |
| QSA keep/scores 持久缓冲 | 省每 token 2 次 `alloc_zeros`（§P15-㊸） |
| **QSA 选块基数选择**（`LINGXI_QSA_KEEP_RADIX`，默认开） | 每行一个 block、3 遍直方图定阈值 + 2 遍填 keep：O(nblocks)/行，零 `t×nblocks×4B` scores 缓冲（256K 省 ≈537MB/层）。输出与旧 rank 逐字相同（4.2K/30.9K 双测）。`d4d3ded` |
| **池化批量 launch** | 原来每块 4 次 launch（256K 约 300 万次）；改成每 chunk 一次 `pool_blocks` + 一次 stride-RoPE。31K prefill 234→208s。`d4d3ded` |
| **稀疏注意力 gather**（`LINGXI_ATTN_GATHER`，默认开） | 每行只遍历 512 选中块（2048 token）+尾部块，QSA 单层 attn 在 8K/16K 上下文恒 ~28ms（旧路径 16K=143ms 且线性涨）。8.3K 89.3→48.8s、30.9K 208→166s。首 token 仍 11751。⚠️ 归约配对变了 ⇒ 长文本贪心与旧路径差 ~1ulp 级；`=0` 回逐位等价路径。`346cd14` |
| PLE IQ4_NL 26.8GiB 表进 shm | 当前生产默认（start.sh 横幅；§㉳） |
| PLE ngram 只算末尾 t 行 + 行缓存 | ~5ms/token（§3.11） |

### 1.4 两卡 / prefill / serve
| 项 | 结论与证据 |
|---|---|
| 双卡 layer-split 默认（lo=24） | 真实中文 20.7→54.1 tok/s（+161%）（§P15-㉜）。⚠️ 该数是 2026-10-17、未钉表、6 token 口径；钉表/槽位变化后**没有干净的同负载单卡 A/B**，引用需谨慎 |
| 按卡分线程 prefill | 4167 token 46.2→30.3s（−34%）；23.7K 6.03ms/行（§㊿-①/③） |
| prefill 分块（单卡 2048 / split 512） | 长 prompt 从 OOM 到跑通，逐位一致（§P15-㉝） |
| scratch 池键 `(设备,类型,长度)` | split+chunk 不再发散（§P15-㉞，`6329c10`） |
| serve（OpenAI chat + SSE + reset） | 热 TTFT 0.13s（§P15-㊷） |
| C2 精确前缀缓存（默认关） | 4187 tok 145×、23.7K 466×（§P15-㊷，`23ff0a2`） |
| serve 接 MTP 投机 | 短 prompt 解码 89 tok/s；875 上下文 38.4 vs 29.9（§P15-㊷，`dbb5af7`） |
| 投机 prefill 线程化 + 全量 hyper | 875-token TTFT 11.8→7.68s（`7a3e080`） |
| HIST 双卡只落一半层修复 | 按实例 `Cell<u64>`（§㊿-④，`d4d954d`） |

### 1.5 投机 / MTP / 路由
| 项 | 结论与证据 |
|---|---|
| MTP stem `+1` 缺失 + qkv 行序修复 | 接受率 0→0.906→1.000（HANDOVER §5.5，`c4c4c40`） |
| prob1 闸门（min_p 0.7） | 18.2→27.6 tok/s（+52%），序列不变（§P0''） |
| 重放 `need_head=false` + 批量重放 | +17% / +3.6%（§P0''/§P15-㉛） |
| GDN FAST 重放（录内部输入） | +10~12.7%，位等价（§P15-F-2） |
| prob1 top-K 20 GPU 核 | 草稿 22.9→12.7ms（§P15-⑮） |
| 钉表后 kd/min_p 重扫 | kd=5+0.7 ≈ 64 tok/s（§㊿-⑦） |
| lookup `choose`（重复负载） | 6/6 命中、57.0→62.9（§P15-㉚） |
| 路由 topk 单趟 / Argmax 两段式 / 锁页 D2H | +10% / 4.5MB→28B / 74.1→66.6ms（§P15-⑰/⑭/§8.8） |

### 1.6 智能前缀缓存 + 长上下文（2026-10-18）

| 项 | 结论与证据 |
|---|---|
| **C3 智能前缀缓存**（`LINGXI_SERVE_PREFIX_CACHE`，**默认开**） | 三类检查点（消息边界 / prompt 刚喂完 / 生成末尾）+ 每个带 `next` token；请求进来取**最长匹配**，等长直接返回（TTFT=0），否则回退到最长前缀续 prefill。检查点 = `StateCheckpoint`（QSA 长度 + GDN `s_all/k_hist/k_hist_next` + PLE `hist/hist_next` 真拷贝，≈115MB/份；`LINGXI_SERVE_CKPTS` 0~3 控份数）。实测 4193-token 多轮：全量 28.09s → 只改末句 **0.35s（81×）** → 原样重发 **0.16s / TTFT 0.00s（177×）**，三次输出逐字一致。`568f4b2` |
| 与 C2 的区别 | C2 只认"完全匹配"；C3 允许回退（真实多轮里用户改最后一句、加一句都能命中），且边界检查点让"前 N-1 条消息"整段复用。 |
| **256K 配方**（实测可启动） | `LINGXI_SERVE_MAX_CTX=262144` + `LINGXI_ARENA_SLOTS=7200 LINGXI_LAYER_SPLIT_SLOTS=7600`（bf16 KV 3.2GB/卡；K8V4 档与 gather 互斥）。启动 23.7/23.2GiB；47K 预填峰值 27.7/26.8GiB。 |
| prefill 实测（双卡、chunk 512） | 8.3K 48.8s；30.9K 166.1s（9600/10000 槽）；47.3K **285.8s**（9600/10000）/457.3s（256K 配方的 7200/7600 槽，专家命中率低是主因）；**250.6K 3436.9s（57.3min）**。⚠️ **2026-10-18 修正**：批量池化核越界写（`356ba85`）让后续 chunk 的 pooled 键陈旧，上述 47K/256K 数字偏乐观；修复后 47.3K = **368.2s**（cap 65536），256K 需重测。 |
| **256K 端到端判据** | 冷 prefill 57.3min → 同 prompt 重发 **TTFT 0.00s**（前缀复用 250552 tok）、整请求 6.4s；两次输出逐字相同（`'<think>\nThe user'`）。峰值显存 31.9/27.3GiB（未 OOM）。 |
| 选块打分 4 路 ILP（bit-exact） | 每线程 4 个独立点积累加链；47K 实测 457→475s（**噪声内，无收益**）⇒ 说明选块不是剩余主项，保留（不影响正确性）。 |
| **不活跃会话落盘缓存**（`LINGXI_SERVE_CACHE_DIR`，默认 `/tmp/lingxi-prefix-cache`；`LINGXI_SERVE_CACHE_GIB` 默认 24，0=关） | 新请求未命中时，把设备上当前会话**整段 D2H**（QSA KV + indexer `rk_dev/pooled` + GDN/PLE）写成 `.lxc` 文件，随后释放内存副本；命中时读回并 H2D 到指定前缀长度。**进程退出后缓存还在**（默认 tmpfs；指到真实磁盘可跨机器重启）。同一会话的 Boundary/Prompt 两档只额外存 GDN/PLE（KV 共享最长那份），头部带 `(len,next)`，启动只读头即可恢复。LRU 按字节上限删文件。实测：18-token 会话 **225.7MB**；A→B→A 命中 18 tok，**TTFT 4.89→0.34s**；**重启服务后**再次请求命中 `hit_src:"disk"`、复用 18 tok、输出逐字相同。`down.sh` 会先调本机 `/shutdown`，把**当前会话**也落盘（实测退出后文件在、重启后命中）。默认 tmpfs 只跨进程退出，不跨机器重启；要跨重启把 `LINGXI_SERVE_CACHE_DIR` 指到真实磁盘。**短会话不落盘**：`LINGXI_SERVE_CACHE_MIN_TOKENS` 默认 8192 —— 18 tok 的 226MB 里 KV 只有 ~0.4MB，其余全是与长度无关的 GDN/PLE 状态（~113MB/份），重算更便宜。字节账：KV 24KB/token（12 QSA 层 × 2KB），32K≈786MB、256K≈6.3GB；indexer rk 3KB/token（256K≈786MB）+ pooled 1/4；GDN/PLE 固定 ~113MB。256K 一份约 **7~9GB**，24GiB ≈ 3 个长会话。 |

### 1.7 serve / OpenAI 兼容 / 部署（2026-10-18）

| 项 | 结论与证据 |
|---|---|
| 流式收尾坏 JSON（**502 根因**） | usage chunk 多了一个 `]`（`"total_tokens":61}]}`），客户端解析失败 ⇒ 读不到 `finish_reason:"stop"`，应用层重试/报 502。已全部改用 `serde_json::json!` 构造，不再手拼大括号 |
| UTF-8 增量反分词 | 逐 token `decode` 会把多字节字符拆成 `�`（实测 emoji 碎成三个）。改成攒够一个完整字符的字节再吐；流式/非流式都走同一套 |
| 思考链字段 | `<think>…</think>` 拆到 `delta.reasoning_content` / `message.reasoning_content`，正文不再带标签；标签跨 token 也能识别 |
| 工具调用 | 请求 `tools` 注入官方 `<tools>` 段；输出 `<tool_call><function=…><parameter=…>` 解析成 OpenAI `tool_calls`（流式一次性 delta + `finish_reason:"tool_calls"`）；参数值优先按 JSON、否则字符串；历史 `tool_calls` 重放、连续 `role:"tool"` 合并成 `<tool_response>` |
| 模板对齐 | 生成头补 `<think>\n`；`enable_thinking` / `reasoning_effort`（xhigh/medium/low）/ `preserve_thinking` / `max_completion_tokens` 生效；历史 assistant 的 `reasoning_content` 回填 `<think>`。模板硬编码在 Rust 里，运行时**不读外部文件、不加依赖** |
| 健壮性 | SSE 首块带 `role`、`x-accel-buffering: no`；单线程服务：请求头 60s / body 600s / 写 120s 超时（只连不发的连接不再卡死全服务；长 body 也不会被误杀） |
| 部署 | **:8000 主入口**。短上下文并发 = `up.sh`（GPU0→:8001、GPU1→:8002、轮询代理 :8000）；长上下文 = `up_long.sh`（双卡 layer-split、`max_ctx=262144`、槽 7200/7600）。两种模式互斥，切换先 `down.sh`（按 pid 停，不 pkill） |
| **长上下文保活** | prefill **之前**先发 200 头：流式每 5s 写 `: keep-alive` 注释行；非流式（HTTP/1.1）用 chunked 空白块。实测 4,020-token 冷 prefill：**首字节 0.022s、保活 4 次、正常收尾**。修复前 24K prompt 冷 prefill 243s 期间零字节 → 客户端/网关超时断开 → 回写 Broken pipe（应用侧看到 500） |
| 请求体读取 | 请求头 60s；**body 放宽到 600s**（长上下文不再被 30s 掐断）；支持 `Transfer-Encoding: chunked` 与 `Expect: 100-continue` |
| `stop` 序列 | `stop` 字符串/数组生效：命中即截断并 `finish_reason:"stop"`；会扣住“可能是 stop 前缀”的尾巴，不会先发出去再撤回 |
| request_id | 响应 `id` = `<unix_ms>-<seq>`，与性能日志同源，应用侧可直接 join |
| **引擎侧日志（M1）** | `run/perf.jsonl`（`LINGXI_SERVE_PERF_LOG`，置空关；`LINGXI_SERVE_PERF_MAX_MB` 默认 64MB、留 3 份）。**只记应用侧拿不到的量**：`request_id / prefix_hit / hit_src(dev:boundary|dev:prompt|dev:end|disk) / spec_rounds / spec_accepted / spec_verified / spec_disabled / expert_hit / expert_miss / expert_unique / cache_mb`。标准 OpenAI 字段（prompt/completion tokens、finish_reason、TTFT、decode 速度）应用侧从响应与 SSE 时间戳自己算，引擎**不重复记**；也不落 prompt/输出内容 |
| 日志体积 | `tools/deploy/logrun.py` 作 supervisor：按行实时落盘；超过 `--max-mb` 轮转（serve 32MB×3、单卡实例 16MB×2、代理 8MB×2）；SIGTERM 转发给子进程，pid 文件里是它 |
| **长期专家频次** | `up_long.sh` / `serve_single.sh` 现在默认 `LINGXI_ARENA_HIST=run/hist_live*.csv`（**已生效**）：每 ~20 个 decode token 把本实例负责的层合并落盘（双卡按层合并 + 全局锁）。放 `run/` 是因为 `docs/` 在容器里对服务用户只读（实测 `Permission denied`）。这是“真实流量偏爱哪些专家”的长期表，之后用它重建钉表。**在线重钉 `LINGXI_ARENA_REPIN` 仍未开**——先积累频次、确认热点稳定再评估 |
| **MTP 按请求启用** | `up_long.sh` / `serve_single.sh` 默认 `LINGXI_SERVE_SPEC=1` + `LINGXI_MTP_ARTIFACT=$LINGXI_ASSETS/qwen4_exp.mtp.fixed.ninfer`。**实测（2026-10-18，256K + MTP）**：显存 25.8/23.2GiB（空闲），冷请求 decode **59.1 tok/s**（21 轮、5.0 枚/轮）；命中前缀缓存的请求自动退回 AR（实测同 prompt 第二次 **0.26s**、`spec_rounds=0`）。冷 prefill 不变 |

---

## 2. 已否决（不要再试）

### 2.1 投机 / 草稿
| 项 | 否决原因 |
|---|---|
| 自喂链封顶 / `MTP_H=mixed` / `nokv` | 位1 接受 0.57，cos(自喂,真 h)=0.135；改 mixer/nokv 无救（§P0） |
| 变长度 k 按历史接受率 | 被 prob1 闸门取代；9 组合扫描全在噪声内（§P15-㉒） |
| lookup 挂在 MTP 链后 | 提议 7 枚接受 0 枚，47.4→44.3（§P9.5c） |
| lookup 自然文本默认开 | choose=3 时 53.9 vs 57.0；中文 0 触发（§P15-㉚/㉜） |
| 扩宽 kd 到 7/8 | 48.5 / 42.0（钉表后），kd=5 最优（§㊿-⑦） |
| 完整多候选树（Medusa 式） | 旧口径“每行 +14ms DMA”必亏；**钉表后前提已变，见 §3.1 重评**（speedup-audit §1/9.4） |
| 放宽接受判据 / 真采样 | 输出不再是主干贪心；需用户拍板（见 §5） |
| attn 段进 CUDA Graph | 24.9/24.7 vs 25.6，零收益且略亏（§P15-⑪） |

### 2.2 核 / 调度
| 项 | 否决原因 |
|---|---|
| dp4a（int8-x） | m=7 301→235；m=2 372→354；且非位等价（§P15-㉖/㊾） |
| 预取深度 6 / RPW 2/4/8 / 4 行 per warp | 全平或更慢（掉占用率）（§P15-㊾附/§3.26A） |
| 四轮 smem 方案（v1~v5） | 全部差于 bf16 全局路径（§P15-⑤c） |
| GR 融合 / GR 转 Q4 | 整机 −11% / 12s per token（§3.14/§7） |
| attn split-K | 有损求和次序；真实语料 attn 不是瓶颈（§P15-㉜/㉛） |
| `a|b` 合并 GEMV | t=1 过、t=7 静默错（§P15-⑯） |
| 瘦长 GEMM 多行/块版 | 逐位等价但 m=8 慢 8–9×（§P15-⑲） |
| f32x 投影（整机） | 净亏 2.6ms；GR mix_up 漂移未解（§P15-⑨） |
| 整轮单图 / 异步出口 / 自旋 | p10 27610 vs 27707；sync 只 0.1ms/token（§P15-㊶） |
| A2 全设备路由（自旋版） | 逐位正确但 −8%（305µs/层自旋）（§P15-㊺） |
| MoE host sync→自旋 | 中性回退（§㊿-①） |
| 双输出 qkv/z 融合 | 零收益（§㊿-⑥） |
| 槽 8000/9200/9500 | 更慢或 OOM（§P11/§14） |
| 24576 槽全驻留 | 需两卡 64GB，本机不可行（§P9.4） |

### 2.3 两卡 / 系统
| 项 | 否决原因 |
|---|---|
| 双卡 Pipeline Parallel（重跨卡） | 一天崩 6 次 GPU1 驱动；净零（§P9.6/§P14） |
| MoE-only 跨卡（层 24..47 交给 GPU1） | 仅 +1.5%，噪声内（§P13.1c） |
| P2P 直连 | 5KB 49.3ms=0.0001 GB/s，IOMMU DMAR 挡死（bench-logs/peer*.log） |
| prefill outer-chunk 成对流水 | 不提速且首 token 变；默认关（§P15-㊿附，`0ca9052`） |
| KV E4M3 用于 prefill | 17K 212→258.8s（+22%）（§P15-㊶） |
| CPU 专家池兜底 miss | 20 线程 1.88GB/s，比 PCIe 慢 5×（§P10.5②） |
| CUDA Graph 扩大覆盖 | 多处实测零收益（§3.47/§5.2） |
| 机械盘放 PLE / 整段 pin arena | 随机读分钟级；曾致整机重启（§7 铁律） |
| `FWD_BENCH` 判 DMA / `SKIP` 判 ms / `KTS` 看绝对值 | 口径全部失真（§0/§3.42） |

### 2.4 量化 / 压缩
| 项 | 否决原因 |
|---|---|
| 2bit / 三值后处理 | 香农下限：W4 误差 0.13，2bit 下限 0.25（weight-compression §0/4） |
| 熵编码 / 字典 / 低秩 / 专家共享基 | 权重无记忆、专家相关≈0、谱近平坦（§3） |
| VP / Q4 用于 GR/shared/router（未测） | GR 已证更慢；shared/router 见 §3.1 待量 |
| NVFP4 当算力红利 | V100 无 FP4 硬件，只是存储格式（§P15-㊻） |
| DeepGEMM 直接使用 | sm90+，本机不可用；只借思想（已借 prenorm → §3.1） |

---

## 3. 未做 / 半做（准备做）

> 每条 = 做法 → 依赖 → 验收判据 → 收益（实测/推断）→ 风险。按“值得先做”排序。

### 3.1 单流 decode

#### A. 设备端 MoE 路由 + 槽管理（去 host 同步）— **已实现并实测，−8% ⇒ 已停放（2026-10-17）**
> 结论见 PLAN §P15-㊺：设备路由核逐位正确（`ROUTE_DEV_CHECK` 全 bit 一致、48 token 序列不变），
> 但单跑 **+2.2ms/token（−8%）**：A2 决策段（自旋）305µs/层 vs legacy 161µs/层。
> 本平台 host 链不是瓶颈，把 host 工作挪到设备端反而因"每层一次硬屏障 + 单 block 核"多出 ~70µs/层 gap。
> 复用件（`load_missing` 部分命中、设备槽位表、脏行传播、glibc expf 移植）保留。
> **不要再按旧模型（host 链 14.7ms）重做**；只有做跨卡流水/并发时才需要回头用它。

- **（历史）现状**：每层 router 要 D2H logits + host softmax/top-k + 查槽，**82µs/层 × 48 = 3.9ms/token（14%）**。
- **做法**（4 步，来自 HANDOVER §5.4 / decode-speed §3.19）：
  1. 设备端 router 核：softmax + top-k + 槽位直查，写 `slot_ids/wts`（A2 已有逐位正确版本可直接复用）；
  2. 设备端槽表 `[层,512]`，miss 时**不通知 host**，走“零拷贝直读 pinned host arena”兜底；
  3. 把“缺专家计数”改成**可选**统计（只在调试时回读），常态零同步；
  4. 接线生产路径，`LINGXI_ROUTE_DEV2=1` 门控。
- **依赖**：glibc `expf` 移植（A2 已完成，4M 输入 0 处不一致）；位等价 harness（A2 的 `ROUTE_DEV_CHECK`）。
- **验收**：首 token 11751 + 固定序列逐位不变 + `bench.sh` p10/p50/p90 全改善；miss 兜底路径单独压测。
- **收益**：−2~3ms/token（−10~13%）；顺带解锁 prefill chunk 重叠（§3.3）。
- **风险**：中；A2 的教训是“自旋等待不可取”，本方案必须做到**零等待**。

#### B. 投机再压一档（树草稿 / lookup / 新 head）
- **B1 浅层窄树** — ❌ **否决（2026-10-18，行经济学不成立）**：
  验证行就是成本单位。实测每多验证一行 ≈ 13ms ≈ **0.5 个顺序 token**；一棵深度 1、2~3 候选的浅树
  每轮多 1~2 行（+13~26ms），而 top-1 单步接受率已 0.958，top-2 只能补几个百分点 ⇒ 期望多收
  **0.05~0.1 token/轮**，远小于需要补的 1~2 个。**不要再做树拓扑**；要提产出只能提草稿质量（B3）。
  （旧结论"树更亏"的前提是每草稿行 ~14ms DMA，钉表后该前提确实变了，但这条行账没变。）
- **B2 lookup 按流量开**：`SUFFIX_DRAFT=3 + CHOOSE=6~8`，只在检测到重复率高的请求（agent/JSON/代码）启用；
  先做离线判定（最近 N token 的 ngram 命中率），再自动开关。验收：自然文本不退化，重复负载 ≥ +10%。
- **B3 新 MTP head（训练侧）**：接受率 3.19→4.3+ 是 decode 剩余最大单一杠杆，但不属推理引擎；
  记录为“需要训练侧配合”。
- **依赖**：B1 依赖验证批能上张量核（§3.4 kTiles）；B2 已有 suffix 模块。
- **收益**：B1 +15~20%；B2 场景相关 +10%；B3 +30% 以上（训练成本另计）。

#### C. 采样/头融合（linear_topk）
- 做法：greedy 时把 head GEMV 与 argmax/top-k 合成一个核（ninfer `C6`，27B 几何完全匹配）。
- 验收：top-1 与原路径逐位相同。收益 ~1%。难度中。

#### D. dense 段格式/布局
- D1 **权重连续大 blob**：先量“每张量独立 H2D/页表/预取”的固定开销，再决定是否改容器格式。
- D2 **shared / router 转 Q4**：每 token 0.6GB bf16；Q4 字节 −75% 但该形状 ALU 限（320 vs 780GB/s），先量。
- D3 ~~W3G64（3bit）~~：**用户 2026-10-18 拍板不做**（不做重训/校准；无校准的 3bit 误差会过不了判据）。
- D4 **GR 三个 rmsnorm 折进 mix_down**：借 DeepGEMM prenorm 技巧，省每层 2 次 HBM 往返；~1–2%。

#### E. 长上下文
- E1 **KV K8V4 的 decode A/B 补测** — ✅ 已测（2026-10-18）：8.2K 上下文 64 token，
  `KV_K8V4=0` 97ms/token / `=1` 97ms/token ⇒ **decode 零收益**（decode 主项是选块+全量扫描，不是 KV 字节）；
  prefill 还 +13%（45.9→52.2s）；**质量不过判据**（见 §1.3 复核）。⇒ **关闭，不再投入**。
- E2 **TBQ4 KV 移植**：K8V4 既然质量不过，先不引入第二种有损 KV；等 K8V4 修好再谈。
- E3 **消息边界检查点 / prompt-tail 检查点** — ✅ 已完成（C3，`568f4b2`，见 §1.6）。
- E4 **分页 KV + host 卸载 + chunked prefill/decode 交替**：
  **会话级落盘缓存已落地**（换出/拉回 + 跨进程恢复，见 §1.6）；分页 KV 与 chunked 交替仍留给
  **并发**场景（多 slot 共用 32K/卡）；256K 单用户容量已够，不需要分页。

### 3.2 并发 / 多卡（最大杠杆）

#### F. 并发 8（continuous batching）★
- **证据**：ninfer 4 路 94.2 t/s vs 单流 26.9（3.5×）；Strata `--batch N` 权重每窗只读一次。
- **做法**：
  1. `BatchSequence`：KV/GDN/PLE 状态按序列复制（GDN 递推已有 `n_seq` 参数，n_seq=1 逐位等价）；
  2. QSA attention 改 per-row `(kv_ptr, kv_len)` 间接寻址；indexer 的 host 循环下设备；
  3. GDN 卷积/PLE short conv 改成 per-seq launch（或一次多 seq 的核）；
  4. serve 调度器：多 slot 排队 + 合批 decode；固定 8 行形状 + 占位 mask（批内关 spec，solo 保 spec）。
- **依赖**：F 是 A 的上层；先做 A 才能让合批不被害；
- **验收**：单流性能不退化（solo 仍 11751/64 tok/s）；4 路聚合 ≥3×；8 路张量核满载。
- **收益**：2–4 路 1.7–2.5×；8 路接近 3–4×（吞吐口径）。
- **风险**：大（多周）；这是目前唯一能突破单流物理上限的路。

#### G. decode PP 窗口流水（依赖 A）
- 旧实测净 1.16×（重叠收益 ≈ chunk 重读成本）；A 去掉 host 同步后重评。
- 做法：chunk i 的后半与 chunk i+1 的前半重叠；事件回收缓冲；仅末块回传。
- 验收：首 token 不变；prefill 4167 从 28.7s → <22s。
- 收益：prefill 再 ×1.3–1.5（TTFT）。

#### H. 专家分卡（Strata SECOND_GPU 式）
- 做法：主卡放全部 48 层稠密+KV，副卡只做“第二份专家缓存”；同一层专家两家各算一半，pinned buffer 汇合。
- 理论临界路径 26.4 → 17–22ms（−17~35%，未实测）；与 layer-split 互斥，需要重排权重归属。
- 风险：大；收益不确定（汇合/负载不均）。建议在 F 之后再评估。

#### I. MTP 搬 GPU1
- 旧方案值 4.86GiB 槽；现在 MTP 已 Q4（1.34GiB），收益大部分被吃掉。**降级为“可选”**。

### 3.3 prefill / 长上下文

#### J. RouterLookahead 预取
- 做法：提前一层用 host router 预测下一层专家，DMA 藏在计算后（weight-compression §5）。
- 依赖：钉表后 miss≈0 时收益小；只在非钉表负载/长 prompt 上有效。
- 收益：256K TTFT 再砍一半以上（文档估计）。

#### K. prefill chunk 加大 — ⚠️ 结论修正（2026-10-18）：**长 prompt + 低槽位时有效**
- 旧测（8.2K prompt，9600/9200 槽，命中率高）：chunk 512 = 66.3s，1024 = 68.8s ⇒ 当时无收益。
- **新测（33K 代码类冷 prefill，7200/7600 槽）**：chunk 512 = **336.9s**，
  chunk 1024 = **278.7s（−17.3%，expert_miss 760k→440k）**；chunk 1536 **OOM**。
- 原因：槽位少 + 长 prompt 时，每个 chunk 都要重搬整层专家；加大 chunk 直接摊薄 H2D。
  短 prompt/高槽位时该瓶颈不存在，所以旧结论只适用于那种配置。
- **决策**：`up_long.sh` 默认 chunk=1024（见 §3.3 P2）；不要再加到 1536（OOM）。

#### L. 256K 全量（**2026-10-18 已落地到"可用"档**）
- 已完成：选块 O(nblocks²)→O(nblocks)；池化批量 launch；注意力改 gather（每行恒定 2048 token）。
  256K 配方（bf16 KV）实测可启动、47K 预填峰值 27.7/26.8GiB。
- ✅ **选块打分已解决（2026-10-18）**：`qsa_indexer_scores_bf16` 一次 grid 并行打分 +
  `qsa_keep_from_scores_bf16`（只读分数做直方图/填 keep）⇒ 32K select+fill 293~335 → **41ms/层**，
  覆盖 256K（nblocks=65536）。**没有剩余项**。
- 已不需要 E4（分页 KV）：256K 容量在 7200/7600 槽下装得下；分页 KV 只对并发有意义。

#### P. prefill 三项手术（2026-10-18 已全部定案）

> 共同证据（真实流量实测）：
> - 24,108-token 冷 prefill **244.93s**；`expert_miss=579,648` × 2.6MB ≈ **1.5TB H2D**，
>   摊到 245s ≈ **6.1GB/s**，正好是 PCIe x8 上限 ⇒ prefill 是 **H2D-bound**，不是 compute-bound。
> - 71,898-token prompt：冷 prefill **567.39s**（前缀复用 24,392 tok，新增 47.5K tok 约 84 tok/s）。
> - 理论下限：60GB 专家**每个只搬一次** ÷ 6GB/s ≈ **10s**；当前多搬了 ~24×
>   （token-major 分块：每 512 token 走完 48 层，每层专家在每个 chunk 都要重搬）。
> - 24K prompt 的 `expert_unique=24,504/24,576`（99.7%）：所有专家都必须流一次，
>   浪费不在“搬了哪些”，而在“每个 chunk 重复搬”。
> - prefill 时 **GPU0 0% / GPU1 100%**：x16 的 GPU0 干完自己 24 层就等 x8 的 GPU1。
>
> 验收统一口径：首 token `11751` / 固定贪心序列不变；47K prefill 与 decode tok/s 对比基线；
> 不得 OOM（每卡 <32GB）。

**P1. 专家 DMA 合并（per-layer 整段搬）**
- 现状：`load_hits` 对每个 miss 专家发一次 2.6MB H2D（每层每 chunk 几百次），
  小拷贝的 DMA 建立/槽位散射开销吃掉了带宽。
- 做法：按层把该 chunk 缺失的专家合成**一次连续传输**。
  - 先核对 arena 同层专家是否物理连续（`ExpertArena` blob 布局）；连续则直接一段 H2D；
  - 不连续则在 pinned host 侧做一次 gather 到 per-layer staging（512×2.6MB ≈ 1.33GB），
    再单次 H2D；槽位改为“每层预留 512 个连续槽”，slot_map 指向这段。
- 依赖：无（可与 P2 独立）。
- 验收：固定 prompt 逐位一致；47K 冷 prefill ≤ 0.75× 基线；decode 不退化。
- 收益估计：**1.3~2×**（per-blob 开销 + 更接近线速）。
- 风险：中；槽位布局变化、staging 多 1.33GB 显存、D2D 拷贝可能抵消收益，需实测。

**P2. 切卡 + prefill 分块（`SPLIT_LO` / `CHUNK`）— ✅ 已实测（2026-10-18），设为默认**
- 现状：自动 `lo=24`（24/24 层）、split 下 chunk 512；GPU0 是 x16、GPU1 是 x8。
- **实测（33,052-token 代码类冷 prefill，MTP 开，7200/7600 槽）**：
  - lo=24 / chunk=512：server TTFT **336.94s**（98.1 tok/s），峰值 29.2/25.5GiB；
  - lo=28 / chunk=512：**319.33s**（−5.2%），峰值 29.9/24.5GiB；
  - **lo=24 / chunk=1024：278.70s（118.6 tok/s，−17.3%）**，峰值 29.1/25.7GiB；
    expert_miss 760k → **440k（−42%）**，与“每层专家每 chunk 重搬”的推断一致；
  - lo=24 / chunk=1536：**OOM**（GPU0 32.4GiB）；
  - lo=28 / chunk=1024：**OOM**（GPU0 32.4GiB，lo=28 的 4 层稠密权重吃掉余量）。
- **决策**：`up_long.sh` 默认 `SPLIT_LO=24` + `CHUNK=1024` + MTP。chunk 1024 是本机上限。
- 依赖：无；只需环境变量 + 重启。风险：低（峰值距 32G 约 2.9GiB）。

**P3. expert-stationary / layer-major prefill — ✅ 框架已实现（2026-10-18），但实测**不划算**，默认关**
- 实现：`LINGXI_PREFILL_LM=1` 打开（默认关）。把「外层 chunk × 内层 layer」交换成
  「外层 layer × 内层 sub-chunk」，`LINGXI_PREFILL_LM_CHUNK`（默认 16384）控制外层次块；
  分卡时 GPU0 先跑完整个外层次块的 0..lo 层、整段跨卡一次，再在 GPU1 跑 lo..48；
  两个外层次块之间用双线程流水（复用 `LINGXI_PREFILL_THREADS`）。
- **逐位一致已认证**：4167 token → 首 token `332`；23,673 → `95835`；32,736 → `198`，
  LM 开/关完全相同。纯调度重排，不改任何算子/状态推进顺序。
- **实测（2026-10-18，冷 prefill，7200/7600 槽，chunk=1024）**：
  - 33,052 token：LM 开 **291.8s** vs 关 **270.6s**（**+7.8%，略亏**）；
  - 4,167 token（9600/10000 槽）：LM 开 47.1s vs 关 29.6s（+59%，小 prompt 更亏，
    因为只有一个外层次块、没有流水，且丢掉两卡重叠）。
  - `LINGXI_PREFILL_CHUNK=2048`：**OOM**（GPU0）⇒ sub-chunk 1024 仍是上限。
- **为什么没赢**（关键证据，evprof 逐段真 GPU 时间，32K 最后一个 chunk，rows=1008）：
  | 段 | GPU0 层均 | GPU1 层均 | 占比（GPU0） |
  |---|---|---|---|
  | 入口+PLE+attn_GR | 56.6ms（含 PLE 层 947ms 异常值） | 215.6ms | 17.2% |
  | attn（QSA/GDN） | **141.8ms** | 142.2ms | **43.2%** |
  | mlp_GR | 18.0ms | 17.7ms | 5.5% |
  | MoE 路由+host | 34.4ms | 26.0ms | 10.5% |
  | MoE 专家段 | **77.6ms** | 77.6ms | 23.6% |
  - **专家段 77.6ms 与 miss 数完全无关**（`miss=0` 层也是 77.5ms）⇒ 它是**纯计算**，
    不是 H2D。layer-major 把 H2D 从 ~1TB 降到 ~90GB，但这部分本来就被 GPU 计算盖住了。
  - 换句话：**chunk=1024 + 7200/7600 槽时，prefill 已经不是 H2D-bound，而是算力 bound**。
    §3.3 P 的共同证据（6.1GB/s 打满）来自 chunk=512 的旧口径，结论需要更新。
- **仍然值得保留的理由**：① 它是 expert-stationary 的前置框架（每层专家只换入一次）；
  ② 对 **≥128K** 的 prompt，H2D 总量按 `outer` 分块后 ÷(chunk 数/outer 数)，
  按 256K 的 miss 账（16TB → ~0.6TB）估计可把 57min 压到 ~30min 量级（**未实测**，需停机 A/B）。
- **结论**：调度层已无可挖项，全部归因到算力（见 §3.3 R）。


**R. prefill 算力瓶颈（2026-10-18 实测与修复记录）**
> 32K 冷 prefill 基线 = 270.6s（118 tok/s）。evprof + `LINGXI_QSA_PROF` 指出时间不在 H2D：
> QSA 段里 **gather 只有 58~62ms/层**，真正的大头是**选块 select+fill 293~335ms/层**。

| 目标 | 现场数字（rows=1008，32K 上下文，层均） | 已定位的原因 | 状态 |
|---|---|---|---|
| **QSA 选块（indexer top-k）** | select+fill **293~335ms/层** | radix 版为省 scores 缓冲，把同一份打分**重算 4 遍**（3 遍直方图 + 1 遍填 keep） | ✅ **已优化（终版）**：`qsa_indexer_scores_bf16` 一次 **grid 并行**打分落一次性 scratch + `qsa_keep_from_scores_bf16`（每行一个 block，只读 4B/块做 3 遍直方图 + 填 keep）。32K 实测 **293~335ms → 41ms/层（7~8×）**；**覆盖 256K（nblocks=65536），不再有 49K 边界**。逐位一致（332/95835/198/40.5K→198） |
| **MoE 专家段** | **77.6ms/层，且与 miss 数无关** | 1024 行 × top-10、400+ 组，走 SIMT Q4 GEMV（4bit 解包 + fp32 FMA）。**已逐项实测排除**：x 改 bf16（字节÷2）只快 1.5%、MB 8→16 无收益、x 搬 shared 的前提不成立（不是带宽瓶颈） | ✅ **已结案（2026-10-18）**：把 Q4 张量核扩到 prefill（kTiles=4、≤32 行/组、整份 expert 列表一次 launch）实测 **120.5 → 130.5s（+8%，更慢）**——fp16 x 的逐次 CVT + 小组填充浪费吃掉 MMA 收益。**代码已移除**，prefill 保持 SIMT；TC 仍只服务 1<rows≤16 的验证批。剩余 77.6ms 是模型固有算术，无 bit-safe 的更快路径 |
| **GPU1 禁池（历史遗留）** | 入口/GR 段 GPU1 是 GPU0 的 7 倍 | 线程版曾 `disable_pool_forever()`，两卡都退化成 cudaMalloc 风暴；池键早已带设备号 | ✅ **已恢复**：工作线程不再禁池（GPU1 的层仍保持 `without_pool`，A/B 开池无收益）。23.7K：137.7 → **123.4s**（−10.4%） |
| **QSA gather 的 12× 冗余搬运** | K/V 49GB/层 ≈ 816GB/s（正好打满 HBM） | 24 q-head / 2 kv-head ⇒ 12 个 q-head 共用一份 K/V，旧版按 (row,head) 各读一遍 | ✅ **已优化**：`attn_gather_body<NH>` 把同 kv-group 的 **4 个 q-head 合并进一个 block**（K/V 只读一次）。32K 实测 attn **58~62ms → 24~26ms（2.4×）**，逐位一致。NH 不能再加：分数缓冲 `span×4B×NH`，NH=6 就超 48KB |
| **gather 第 2 遍的 256× 重复 exp** | 每块 2048 token × 256 线程重复算同一份 exp | `__expf(sp[idx]-mx)` 放在"每线程固定一个 d"的循环里 | ✅ **已优化**：先由 256 线程并行算一次、就地写回 shared。23.7K：123 → 120.8s（−2%），逐位一致 |

**本轮端到端实测（同一二进制 A/B，均逐位一致）**：
- 4,167 token：30.0 → **26.7s（−11%）**，首 token 332；
- 23,673 token：137.7 → **约 123s（−10.7%）**，首 token 95835；
- 32,736 token（生产 7200/7600 + chunk1024）：270.6 → **242.4s（−10.4%）**，首 token 198；
- 40,514 token：radix 316.6s → bigpar **308.5s**（该 prompt 只有最后 4 个 chunk 越过 49K 阈值；阈值已取消，256K 全程受益）。

- 验收统一口径不变：首 token `11751` / 固定贪心序列、长 prompt `95835`/`198` 逐位一致。
- 顺序建议：**没有剩余待办了**。QSA 选块/gather、线程池化、CPU 归因、MoE 张量核都已实测定案（该修的修了，不该修的有数据）。

#### R.1 全 `.cu` 重复计算审计（2026-10-18，逐文件）

| 文件 | 检查结果 |
|---|---|
| `attn_ops.cu` | **发现并已修**：`qsa_keep_select/fill` 对同一份打分算 4 遍 → 终版改为「1 遍 grid 并行打分落 scratch + 只读分数的直方图核」（41ms/层，覆盖 256K）。其余：`attn_gather` 的 2 遍是 softmax 必需（第 1 遍分数留 shared，不重读 K）；`attn3/attn2s` 已是 K 按 R 行共享 + 分数留 shared（旧的"每个输出维度重算 QK"已不存在）；`qsa_pool_blocks`/RoPE 都是增量一次；`qsa_sel_build` 单遍 |
| `dequant_ops.cu` | `slotsm` 已是"权重行只 load 一次 + nibble 只解一次（提到 pair 循环外）"，MB=8 分批；x 按输出行重复读是该 GEMV 形态的固有代价。**实测 `LINGXI_MOE_NO_XF32=1`（x 流量减半）只快 1.5%** ⇒ x 不是瓶颈，不做 smem 重写 |
| `moe_ops.cu` | `gemm_skinny_row_multi`（R=4 行/块 + x 进 smem + swizzle）已覆盖 dense 投影；gather/combine 均单遍 |
| `gdn_ops.cu` | S 已常驻 shared（原来是全局 4 遍读写）；16 宽预载；q/k 每 token 读 2 次（范数 + 使用）只有 512B/token，可忽略。T 方向是递推依赖，不能并行 |
| `gr_ops.cu` | 5 核融合只服务 decode t==1；prefill 的 5 个小核被 t=1008 摊薄，launch 不是瓶颈 |
| `topk_ops.cu` | prefill 路由走 host 插入式 top-k。试过**按行并行（4 线程）→ 反而慢 7%**（123.8 → 132.7s）：两卡线程已在抢 CPU。已回退，不留开关 |
| `q4_tc.cu` | 张量核档是 fp16 有损（~2^-11），只覆盖 ≤16 行；**2026-10-18 实测扩到 prefill（kTiles=4）反而 +8% 慢** ⇒ 已移除，prefill 保持 SIMT |

#### R.2 CPU/host 到底占多少（2026-10-18 实测，纠正归因）

`LINGXI_PROF=1` 的 `[moe]` 行（23.7K prefill，层均，48 层 × 24 chunk）：

| host 段 | 实测/层 |
|---|---|
| 路由 softmax + top-k（1008×512） | **0.76~0.83ms** |
| `load_hits`（槽位查找，uniq≈150~210） | 0.03~0.11ms |
| 专家 GEMV 的 host 发射 | 0.29~0.42ms |
| 全部 kernel 发射 `enq` | 0.18~0.23ms |
| shared expert host 段 | 0.16~0.19ms |
| **合计** | **≈1.5ms/层** ⇒ 48×32×1.5ms ≈ **2.3s/卡 ≈ 总时间的 1%** |

- 结论：**CPU 不是瓶颈，也没有"提前做完"的空间** —— 路由依赖本层 logits，而 logits 依赖上一层输出，
  是串行依赖；可复用的 Vec/工作区（`RouteWs`）已经池化了。
- **纠正**：evprof 的 "route+host 34ms/层" 不是 CPU，而是 `stream.synchronize()` 在等
  **上游 GPU 流水排空**（attn/GR 已经入队但没跑完）⇒ 归因到 GPU 时间。
- 并行 top-k（4 线程）实测更慢（123.8 → 132.7s），也印证了这一点：本来就只有 0.8ms。

**当前状态（2026-10-18 收口）**：P1/P2/P3 与 R 的三项都已实测定案——
P2 已落地；P1 收益被 P3 证据大幅下调（H2D 不再是瓶颈）；P3 框架已实现但默认关（33K +7.8%）；
R 的选块/gather/exp/池化四项已优化并上线，MoE 张量核已实测否决。
**没有未完成的引擎侧待办**。再要提速只剩并发（F）或换数值档，需单独立项。

### 3.4 产品 / 服务

| 项 | 做法 | 验收 |
|---|---|---|
| M. 采样（temperature/top-p） | serve 增加采样路径，`temperature=0` 走现有贪心；采样时允许非逐位 | 贪心路径零回归；采样输出质量抽检 |
| N. think 标签过滤 / stop 序列 | think 拆到 `reasoning_content`、`stop` 字符串/数组截断 | ✅ 均已落地（§1.7，2026-10-18） |
| O. 投机 + 前缀缓存兼容 | **按请求选路**：冷请求（无前缀可复用）开 MTP；命中前缀缓存自动退回自回归 | ✅ 已实现并验证（这是**设计选择**：MTP 的 KV 不进检查点，避免多一份状态） |
| P. 前缀缓存默认开 | **已拍板并落地**（C3，默认开；`=0` 可关）。检查点取真拷贝，跨请求逐字一致；不活跃会话可整段落盘（§1.6）。MTP 改为**按请求选路**：冷请求开、命中缓存退回 AR（见 O） | ✅ `568f4b2` + 落盘 + 选路（2026-10-18） |
| Q. Responses API / Anthropic 协议 | 明确 YAGNI，暂不做 | 只要兼容openai的就够了 |

### 3.5 系统 / 测量

| 项 | 做法 | 收益 |
|---|---|---|
| R. ECC 关 + 锁频 1380MHz + 250W | 需停机/BIOS；文档估 +3~4% | +3~4%（未验），该项目不做，没意义，收益太低 |
| S. 每层 Vec/upload 复用 | 小而顺手 | <2ms/轮（量不出） |
| T. bench 自动回归挂 CI | `bench.sh` + `prefill_equiv.sh` + `spec_ab.sh` 串成一条命令 | 防回退 |

---

## 4. 建议执行顺序（2026-10-18 更新，已剔出结案项）

| 顺位 | 项 | 状态/理由 |
|---|---|---|
| 0 | **部署保持可用**：短上下文并发 `up.sh` / 长上下文 `up_long.sh`（§1.7） | ✅ 已落地并在跑（:8000） |
| 1 | ~~长上下文 decode 收尾~~ | ✅ 已修：47K illegal address = 批量池化核越界写（`356ba85`）；修复后 47.3K 预填 368.2s 无崩溃。decode 选块改并行直方图：30K 254→65ms/token |
| 2 | **数据飞轮（M1）**：应用侧记内容，引擎只记性能 JSON（不落 prompt/输出） | 零 GPU 成本；后面所有训练的前置（§6.2）。**引擎侧已完成**：`run/perf.jsonl` + `request_id`；剩余是应用侧按 id join |
| 3 | **草稿接受率测量（M2）**：用真实流量算 MTP 的 A 分布 | 决定要不要训小草稿模型 |
| 4 | **F 并发 8** | 单流之外唯一大杠杆；工程量大 |
| 5 | **C linear_topk / D4 GR norm 折叠 / D2 shared、router 转 Q4** | 各 ~1%，顺手做 |
| 6 | **G prefill 窗口流水** | 依赖 A 的设备路由；A 已停放 ⇒ 顺延 |
| 7 | **prefill 三项**（§3.3 P） | ✅ P2 已实测：lo=24 + CHUNK=1024 冷 prefill **−17.3%**，已设为默认；1536 OOM。下一项 P1，再 P3 |
| — | ~~A 设备端路由~~ | ✅ 已实测 −8%，停放（PLAN ㊺） |
| — | ~~B1 树草稿~~ | ❌ 否决（行经济学，§3.1 B） |
| — | ~~E1 K8V4 / E3 边界检查点~~ | ✅ 已测/已落地（§1.6、§3.1 E） |
| — | ~~K chunk 加大~~ | ⚠️ 结论修正：33K/7200 槽下 chunk 1024 **−17.3%**（§3.3 K），已设默认；1536 OOM |

## 5. 需要用户拍板的清单

| 决策 | 选项 | 影响 | 状态 |
|---|---|---|---|
| 位等价放宽 | 允许 1ulp（GDN 拆 dk / attn split-K / 设备 expf） | 换 2~5% 单流，但判据要从“逐位”改“质量等价” | 不做 |
| 3bit 权重（W3G64） | 做 / 不做 | 字节 −25%、速度 +10~15%，需重训/校准 | **不做**（2026-10-18 用户拍板：不做重训/校准） |
| 投机接受判据 | 严格 argmax / 允许 top-m | 后者更赚，但输出不再逐位复现 | 不做 |
| 前缀缓存默认 | 开 / 关 | 多轮 TTFT vs 逐位可复现 | **开**（含不活跃会话落盘，§1.6） |
| 会话缓存落盘预算 | `LINGXI_SERVE_CACHE_GIB`（默认 24）/ 0 关；`LINGXI_SERVE_CACHE_DIR`（默认 `/tmp/lingxi-prefix-cache`）；`LINGXI_SERVE_CACHE_MIN_TOKENS`（默认 8192） | 磁盘/内存占用 vs 跨重启命中率 | **24 GiB + tmpfs + 只存 ≥8192 tok**；256K 一份约 7~9GB |
| 256K 目标 | 维持“容量可用” / 投分页 KV | 后者工作量大，换来长上下文可用 | **容量可用，已部署**（`up_long.sh`）；不投分页 KV |

### 5.1 三个决策的量化依据（2026-10-18）

**① 1ulp vs 16→4bit：完全不同的两件事**

| | 16→4bit（W4，已做） | 放宽到 1ulp（未做） |
|---|---|---|
| 改的是什么 | **存储的数据**（权重精度） | **计算的顺序**（归约/分块） |
| 数值差异 | 量化误差（专家 gu 相对 0.115~0.124） | 只在被重排的归约处差 1 ulp |
| 可复现性 | 固定产物 + 固定配置下**逐位可复现** | 换 batch/上下文/卡配置都可能变 |
| 实测收益 | 字节 ÷4；端到端 **−10%**（p10 27.6→24.8ms） | GDN 拆 dk ≤0.7ms（~3%）；attn split-K 长上下文 0~10%，短上下文≈0 ⇒ **合计 2~5%** |
| 质量验证 | 一次性对拍（dense/head W4 序列逐位不变；专家 Q4 会改序列但质量测试过） | **固定 golden 失效**，只能做统计/任务级评估（仓库暂无这套 eval） |
| 风险 | 可控：产物固定、可回退 | 蝴蝶效应 + 未来所有 A/B 失去“逐位”这把尺子 |

结论：W4 是“换一份固定的权重”，1ulp 是“每次运行都可能不一样”。后者收益只有前者的 1/3，却会拆掉我们最重要的验证基础设施。

**② W3G64 不做重训/校准 ⇒ 关闭**
现有 W4 是 AWQ + GPTQ + imatrix 校准后的产物（4.12 bpw，误差 0.13）。3bit 若不做任何校准，
误差会显著大于 0.13（同族数据外推 >0.2），既过不了 11751/长题，也会让 agent 工具调用误差累积；
换来的只是 ~10~15% 速度。**已按用户拍板从 BACKLOG 移除。**

**③ top-m 放宽的代价（用实测接受率算）**
- 实测链上条件接受率（短 prompt）：第 0 位 top-1 = 1.000；**第 1 位 top-1 = 0.57、top-8 = 0.89**；更深位 top-1 0.6~1.0。
- 放宽到 top-2：第 1 位大约 0.57→0.65~0.70 ⇒ 被接受的字里约 **8~12% 不是主干 argmax**。
- 放宽到 top-8：第 1 位 0.57→0.89 ⇒ 约 **30%+ 的接受字不是主干 argmax**。
- 影响：聊天/散文通常仍连贯；**JSON、代码、工具参数**里一个 token 错就整段失败。
- 建议：全局不开；若要开，只在“自由文本 + 非工具调用”的请求上开 top-2，且用任务级 eval 验收，
  不要用它替换现有的逐位判据。


## 6. 训练侧路线：数据飞轮 + 自训草稿（2026-10-18 讨论定稿）

> 用户前提：引擎、模型权重、应用框架都是我们自己的 ⇒ 理论上可以做"数据 → 训练 → 权重"的闭环。
> 本节只记**决策与里程碑**，不展开训练实现（那是另一套栈，不在灵犀推理引擎里）。

### 6.1 硬约束（决定了什么能做、什么不能做）

- 硬件：2×V100-32G、无 NVLink、PCIe x8；**服务时两卡已被 layer-split 占满**。
  ⇒ **服务与训练不能同时占卡**（进程主权规矩：一台机器同一时刻只准一个会话指挥显卡）。
- 判据：现有 bit 级回归（首 token `11751` / 固定贪心序列）仍是准绳；任何训练产物必须过同一套。
- 权重策略：模型权重不发布；语料只在自己机器上（既定策略）。

### 6.2 三条结论

1. **不做"边服务边改权重"的在线自训练**：没有空卡，且在线改权重会让历史 A/B 全部失去可复现性。
2. **先建数据飞轮（零 GPU 成本）**。**分工：应用记内容，引擎只记性能。**
   - 应用层（已有）：对话、记忆、工具调用与结果、用户纠正/重问、按域打标；
   - 引擎层（`serve`）：只写一行性能 JSON —— `request_id / prompt_tokens / ttft_ms / decode_ms /
     tokens / 前缀命中数 / 投机接受数`，**不落 prompt/输出内容**；这些东西应用侧可以拿到，为什么要做？
   - 离线用 `request_id` join：得到"真实请求的 token 流 + 每 token 耗时"，正是训草稿的监督信号。
   **这是后面所有训练的前置，越早开越好。**
3. **训练放夜间离线窗口**（服务停机）：**主干完全冻结，只训外挂件**——
   ① **外挂小草稿模型**（0.3~1.5B，下节；不常驻显存，用时加载）
   ② **LoRA/QLoRA 适配层**（rank≤16、seq≤2K，只做"工具格式/风格"适配，不改核心知识）。
   27B 全参训练在这台机器上不做（显存/带宽都不够，也没必要）。

### 6.3 自训草稿模型（D-Flash2 式）

- 方向正确：它抬的是**每验证行产出 token 数 A**，正好打在我们的瓶颈上（验证行 ≈ 0.5 个顺序 token）。
- **判据**：A ≥ 3 才明显超过现有 MTP（A≈2.5）；草稿单轮成本 <10ms。
- "并行预测"（一次出 4~8 token，含扩散/循环结构）：验证端本来就是 t 行一遍过，多出的 token 不加行；
  **但接受率不够就会亏**（行经济学见 §3.1 B）。
- 建议顺序：先 M2（在真实流量上统计各域 A）找到 A 最低的域 → 再决定 0.3~1.5B 自回归草稿
  vs 扩散/循环结构。**不要一上来做扩散小模型**。

### 6.4 里程碑

| 编号 | 内容 | 验收 |
|---|---|---|
| M1 | serve 性能 JSON（不含内容）+ 应用侧内容日志，按 `request_id` join；每周域统计 | **引擎侧 ✅**（`run/perf.jsonl`，字段见 §1.7）；应用侧 join + 周报待做。验收：连续 7 天可 join；能出"请求数/上下文分布/decode 速度/工具成功率" |
| M2 | MTP 的 A 分布（分域） | 每域 A 的中位数与 P10；标出 A<2 的域 |
| M3 | 0.3~1.5B 草稿训练（离线） | A≥3 且草稿 <10ms/轮；接回 serve 后 TTFT/decode 都改善 |
| M4 | LoRA 夜间适配（工具格式/风格） | golden 判据不退化 + 目标任务 eval 提升 |

### 6.5 风险

- 夜间训练要占卡 ⇒ 服务需停机窗口（单用户可接受）。
- 草稿模型与主干版本强绑定：主干换权重必须重训/重验草稿。
- 域统计与日志涉及对话内容：只落本机、不发布（既定策略）。

---

## 7. 来源与复现

- 全量审计：`PLAN.md`（4105 行）、`HANDOVER.md`、`speedup-audit.md`、`decode-speed-profile.md`、
  `weight-compression-plan.md`、`inference-framework-design.md`、`strata-mtp-expert-cache-plan.md`、
  `docs/bench-logs/*`、`git log --all`（268 commits + 1 stash）。
- 常用回归：`bash tools/golden/bench.sh prod 200 2`；`bash tools/golden/spec_ab.sh spec 64 5`；
  `bash tools/golden/prefill_equiv.sh /tmp/parity_prompt.txt 332`；
  `LINGXI_SERVE_SPEC=1 ./start.sh ./bin/lingxi-bench serve <产物> 11434`。
- 探针：`LINGXI_EVPROF`（五段）、`LINGXI_KTS`（GDN 分段）、`LINGXI_GEMV_PROBE`（背靠背带宽）、
  `LINGXI_ARENA_STATS/HIST`（命中）、`LINGXI_ROUTE_DEV_CHECK`（路由逐位）。
