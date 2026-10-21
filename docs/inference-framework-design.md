# lingxi 推理框架总体设计——strata × ninfer 优势综合方案

> 事实源：
> - strata 源码（已读透，详见 [strata-mtp-expert-cache-plan.md](strata-mtp-expert-cache-plan.md)）；
> - ninfer C++ fork 源码（`$DATA/ninfer-dev`，已拉出 892 个源码文件对照阅读，
>   含其自己的 STRATA_RESEARCH.md，
>   属互相参考关系，fork 已转录过 strata 的槽位/profile 思想并按 V100 现实做了关键修正）；
> - lingxi 现有架构约束 [ARCHITECTURE.md](../ARCHITECTURE.md)（硬约束，本文不违反）。
>
> 一句话：**ninfer 给「能不能跑起来」的工程骨架（产物、专家 arena、分页 KV、调度、HTTP），
> strata 给「能不能跑快」的算法部件（profile 热度、lookahead、suffix/MTP Controller），
> lingxi 保持自己的灵魂——ids 解耦、离线零转换、bit 级确定性和 Rust 的模块纪律。**

# 1. 三引擎事实卡

| 维度 | strata | ninfer（灵犀 C++ fork） | lingxi（本项目，目标框架） |
|---|---|---|---|
| 语言 | C++/CUDA + Python server | C++/CUDA + 内嵌 HTTP | **Rust/cudarc**，unsafe 只在 kernels crate |
| 主模型 | 另厂 MoE | Qwen3.8-27B dense + qwen4_exp MoE（端口进行中） | qwen4_exp：48 层混合（12 QSA / 36 GDN），512 选 10 MoE，hc=4，PLE |
| 专家精度 | Q8_0（draft Q8_0，主模型 gguf 全档量化 MMQ） | BF16 原样镜像为主，备 W4/W5/W6/W8（Q4G64…）混档 | 同 ninfer 注册表，**离线产出 kernel 直接消费的镜像**（铁律 0.6） |
| 专家 miss 怎么算 | **CPU 池 AVX512 + GPU 双路径**，舍入不同，temp=0 也会漂移（issue #152）；byte-identical 需一串开关 | **只搬字节、永远 GPU 同一套核算**（moe_expert_cache.cuh L8-18），命中与否逐位一致 | **继承 ninfer 这条**：不做 CPU 专家池；缓存只解决「字节在哪」，不引入第二条舍入路径 |
| 调度并发 | 单/少请求，验证窗内 T 列并行 | **连续批处理**：decode 合批（权重每轮只读一次）、单 prefill lane、FIFO 队首保护 + persistent backfill 准入、KV 预算准入 | 先单流贪心跑对，再按 ninfer 同款模型实现合批；**不提前抽象**（单流都没对先不写调度框架） |
| KV | 分页 + FP16/Q8/Q4 + >64K 降 RAM | paged KV + **host KV arena 降级/恢复** + TBQ4（262K 实测可用）+ StateImage 前缀快照 | QSA 仅 12 层需要 KV（占比小）；GDN 36 层持递归 state；先 bf16，后续可借 TBQ4 |
| 投机 | MTP + suffix trigram + Controller E[提交]/ms + 在线 EMA | MTP + DFlash2（27B 模型实测 57~59 t/s，接受率 0.52） | 按 strata 方案先 M0→M3（见专家方案文档 §3.1）；DFlash2 不做（需要训练管线） |
| 长上下文 | 一般 | YaRN + MRoPE 像素轴保护，全路径无降级 | 本模型纯文本、max_pos 262144，RoPE 表已生成；YaRN/MRoPE 先不做（需要时从 ninfer 转录） |
| 服务 | Python OpenAI/Anthropic | **C++ 内嵌三套协议**：OpenAI Chat + Responses API + Anthropic Messages，SSE | axum 只做 OpenAI Chat Completions + SSE（YAGNI），Responses/Anthropic 排后 |
| sm 目标 | sm_75 起，sm_70 实验性 | **sm_70 V100 一等公民**（本家和我们同硬件） | sm_70，cuBLAS bf16 tensor core + 自研小核；无 fp4 指令 |
| 存储假设 | SSD、大内存 | SSD；qwen4 专家 arena 60GiB mmap | **机械盘**、79GB RAM、PCIe x8——三层存储更依赖离线镜像与预取设计 |
| 验证文化 | scalar 参考+pinned llama.cpp oracle+负对照，env 门控手术 | bit 级对拍文化（C 系列每条独立验证）；故障静默回退教训深（C8） | T0–T3 分级（[ARCHITECTURE.md](../ARCHITECTURE.md) §6）+ env 门控 + 每个新核配 golden |

# 2. 各自真正的强项（剔除表象后的清单）

## 2.1 ninfer 强在「工程骨架已经在同硬件上被生产验证过」

1. **专家 arena 镜像格式 Q4EARN1**（moe_expert_arena.h）：
   72B 头 + routed/shared 两张 16B 索引 + slot 对齐 blob 区；**每个 blob 自带字节数与格式字**，
   读侧强制校验「tag 与字节数一致」（挡「Q4 的标签、BF16 的长度」这类貌似合理的垃圾）；
   缺席条目 -1 支持稀疏 arena。转换期一次落盘、启动直接 mmap，**运行时零反量化**——与我们铁律 0.6 同构。
2. **mmap + `cudaHostRegisterReadOnly` 零拷贝直读**（C8 修复，8→21~26 tok/s）：
   只读 mmap 必须用 ReadOnly 注册；失败静默回退 gather 路径并打日志。
   启动用 `fadvise` 按 GB 预热页缓存（机械盘冷启动 5 分钟 fault-in 的教训）。
3. **ExpertSlotCache 的工程细节**（同文件）：pinned staging 环把 pageable 1.5ms 小拷贝降到 0.35ms；
   聚合 gather（32 个 miss blob 拼成两发 H2D + 一个设备侧 scatter 核）；
   per-call scratch 与 persistent LRU 双模式；**call_pin 保证本 call 在用的槽不被自己驱逐**；
   fill 后 verify_slot 字节读回是上线唯一凭据。
4. **三层 KV 体系**：device paged pool（paged_kv_cache.cpp）
   → host pinned KV arena，device→host demote / host→device 逐页恢复完整实现；
   StateImage 前缀快照（shared 前缀/多轮命中）；TBQ4（WHT-d256 + Lloyd-Max 4bit）质量 A/B 过。
5. **连续批处理调度**（scheduler.h）：
   FIFO 队首保护 + epoch/protection 机制，队首被资源阻塞时才借让出资源给 persistent backfill；
   prefill 单 lane、decode 多 lane 合批（`ordinary_decode_batch`，权重每轮只读一次，4 路近线性 3.5×）；
   KV 容量准入（kv_capacity.cpp）与
   materialization planner。
6. **完整服务层**：OpenAI Chat / Responses API / Anthropic Messages 三套请求-状态机 + SSE +
   请求校验/操作日志（`src/serve/`），可直接照抄协议语义（不是代码）。
7. **env 门控手术纪律**：所有高风险机制（resident/pin/route dump/量化档）默认关、env 开，
   默认路径行为字节不变，出问题可逐开关回退——与我们的确定性改造打法完全一致。

## 2.2 strata 强在「算法部件：让缓存和投机真的有命中率/接受率」

1. **STRP profile 与十折留一测量**：出厂先验（24576 对全排名）；
   实测「按专家热度 0.6447 vs 按层整组 0.0456」决定了必须行级 hit/miss 拆分；
   全局到达顺序计数器 2.97% 命中率的失败教训 → per-layer admission。
2. **自适应换入换出**：usage 衰减计数、候选/受害者迟滞门限、独立 adapt stream 窗口间换入、
   event 落地+校验后才翻 residency 表、换出先回 host 镜像。
3. **RouterLookahead**：host bf16 router 提前一层预测路由预热页面。
4. **MTP 验证窗 + 接受循环**：窗口 T=k+1 主模型一次过、最长前缀接受、输出永远由验证决定；
   MTP 层独立 K/V、<2051 cell dense 注意力；接受率 0.89/0.86/0.85 有实测。
5. **SuffixDrafter**：零权重 trigram（~20B/token），重复语料极便宜。
6. **Controller**：none/suffix/MTP 按 E[提交 token]/耗时 选路，5% 迟滞，EMA 在线学接受率，
   参数先验/cost 全部实测标定。
7. **确定性自觉**：split reduction 固定树、multi_exact（T 列 == T 次单列 bit 级）、
   STATE_HASH 跨请求状态哈希校验、byte-identical 开关组合。

## 2.3 lingxi 自己必须保的灵魂（两套 C++ 都没有的东西）

1. **lingxi-ids 唯一契约**：跨模块只传 id，换模型/换 kernel 只改注册表（铁律 1）。
2. **单向依赖 + 一个模块一件事**（铁律 2/4）：ninfer 的 qwen4 launcher 单文件 857 行、
   缓存/路由/量化/探针全混在一起——我们把它拆成 ids/plan/kernels/model/driver/server 横向层。
3. **离线零转换**（铁律 0.6）：Q4EARN1 风格镜像 + 按 kernel 访问模式预排布，运行时只读只算。
4. **T0–T3 结构验证**：ninfer/strata 的 bit 纪律靠人盯注释；我们把它固化成 bench 子命令与分级门禁。
5. **Rust 安全边界**：unsafe 集中 kernels；状态所有权让「call_pin 式生命周期保护」
   大概率能由借用检查在编译期承担一部分，而不是运行时向量+注释。

# 3. 目标架构：部件从哪来、长在哪个 crate

严格落在 [ARCHITECTURE.md](../ARCHITECTURE.md) §1 的依赖方向上；
新增职责**优先新增模块**，不膨胀现有模块。

```
lingxi-ids       （契约：补 ExpertId/ProfileId/DraftKind/SessionId 等身份，零算法）
lingxi-artifact  （ninfer v2 容器读取器，已有；增加专家 arena 镜像 Q4EARN1 读取能力
                  或新 lingxi-expert-arena crate——见 §3.4）
lingxi-plan      （把 profile/arena/槽位摊成 Span）
lingxi-kernels   （新增：dense 因果注意力（MTP）、设备侧槽位 scatter、（后续）grouped GEMM）
lingxi-model     （mtp.rs 新增；moe.rs 改 hit/miss 行集拆分；前向不做 IO 不做调度）
lingxi-driver    （专家缓存运行时、KV/state、连续批处理、draft/verify、Controller）
lingxi-server    （OpenAI Chat + SSE；分词）
lingxi-bench     （golden/trace/命中率/接受率/对拍，旁路）
```

## 3.1 权重与专家存储三层（综合 ninfer 镜像 + strata 热度）

- **盘（机械盘）**：ninfer v2 产物（模型权重）+ Q4EARN1 兼容的专家 arena 镜像
  （BF16 原样 blob + per-blob 格式字；读侧几何/字节强校验）。
  离线转换器一次产出；PLE 102GB 外表维持 mmap。
- **内存（79GB RAM）**：arena 整段 mmap + `cudaHostRegisterReadOnly`；
  启动 fadvise 预热；被自适应换出的专家回紧凑 host 镜像（strata exchange buffer，
  在我们这天然就是 arena 内原位，无需另建——镜像已含全量专家）。
- **显存（32GB）**：等大专家槽 arena（同层专家同形，不需要 strata VMM 分段）；
  residency[48*512]→槽；pinned staging 环 + 聚合 gather H2D（ninfer 实测参数先照搬：
  环深 8、聚合 32）；fill→**verify_slot 字节校验**→翻表，三步不省。
- **放什么进槽**：第一层先用 STRP 风格 profile 预填（**per-layer 名额**，
  不许用全局到达顺序）；槽位数由自己的 routing trace 十折留一曲线定，不抄 7168。
  第二层再上 strata usage 衰减自适应（独立 stream、窗口间换入、迟滞门限）。
- **hit/miss 行集拆分**：host 路由（[moe.rs](../crates/lingxi-model/src/moe.rs)
  已有 `sel_rows`）在调核前切行；hit 走槽内权重，miss 走 PCIe 直读 pin 住的 blob 上卡算；
  **两路径同一 GPU 核、同一 bf16 舍入**，输出按 router 原序 scatter 加权（现有
  `scatter_add_rows` 可复用）。不做 strata 的 CPU 池——确定性优先。
- 细节顺序与门禁见 [strata-mtp-expert-cache-plan.md](strata-mtp-expert-cache-plan.md) §3.2 E0–E2；
  其中 arena 文件格式/staging/聚合/gather/verify 的**实现以 ninfer 已验证代码为蓝本转录为 Rust**，
  不重新发明。

## 3.2 KV 与状态：按我们的架构特点裁剪

- GDN 36 层：递归 state + conv state（step 路径已对拍逐位一致），容量小、随会话常驻；
  长会话/换出时借 ninfer StateImage 思路整体快照到 host。
- QSA 12 层：paged KV（先 bf16）；多会话后引入 device page pool → host arena 降级/恢复；
  TBQ4 作为容量档选项（ninfer 已在同硬件 A/B 验证），但必须走我们的 T2/T3 门禁再默认开启。
- PLE：仅层 1，hist 状态已实现；ngram 表 mmap 直读（ninfer 的 PleReader issue/collect
  把盘延迟藏进 embedding+layer0 的手法后续转录）。
- 前缀共享：单会话先不做；多会话时上 ninfer 的 checkpoint 链思路（radix 塌缩成单链，
  我们对话型负载大概率够用）。

## 3.3 批处理与调度（ninfer 模型，分两步）

- **第一步（单会话正确）**：保持现有 driver 单流；专家缓存先让单流 decode 可用。
- **第二步（多会话吞吐）**：转录 ninfer 调度的核心结构而非框架：
  decode lane 合批（权重每轮只读一次是近线性的来源）、单 prefill lane、
  FIFO 队首保护 + backfill 准入、按 KV 预算准入新请求；最大并发默认 4、硬顶 8 先照搬。
  Rust 侧落点 `lingxi-driver::scheduler` 新模块；lane 状态用所有权表达，
  不把 ninfer 那套 64 位 epoch 校验原样搬（那是 C++ 缺少别名保护的补偿）。

## 3.4 投机解码（strata 算法，ninfer 实测数据做先验）

路线与门禁已在专家方案文档 §3.1 定死（M0 验证窗 → M1 suffix → M2 MTP 层+dense 核+golden →
M3 Controller），不重复。补充两点 ninfer 侧实证：

- MTP 在同硬件 27B 上的真实接受率约 0.52、长上下文 1K:2.34 / 8K:1.60 接受长度——
  Controller 的先验/cost 不要只信 strata 的 0.86，要**机上自校准**（strata 自己也是实测标定）。
- 确定性红线：验证窗 T 列与 T 次单列必须逐位等价（multi_exact 原则）；
  我们 step 路径已证 t=5 空状态整段 step 与 layer_forward 逐位相同，这条已经提前满足一半，
  接 MTP 后要把「窗口批量形状」也纳入同一对拍。

## 3.5 服务层

- `lingxi-server`：axum，只做 OpenAI Chat Completions（含 SSE chunk）+ 分词
  （tokenizer.json 离线随产物/模型目录）；不做计算不做调度（铁律）。
- 协议语义、SSE 事件序列、请求校验字段以 ninfer `src/serve/openai_*` 为蓝本。
- 离线依赖问题（cargo 缓存仅 44 crate）：axum/tokio 走 vendored 或本地 registry 镜像解决，
  不引重量级框架，hyper+tokio 足够。

## 3.6 模块/文件落点一览

| 新能力 | 落点（Rust） | 蓝本来源 |
|---|---|---|
| arena 镜像读（Q4EARN1 兼容） | 新 crate `lingxi-expert-arena`（纯 host、零 CUDA，符合依赖分层）或 lingxi-artifact 内子模块 | ninfer moe_expert_arena.h |
| STRP profile 读写 | `lingxi-model::expert_profile` 或 plan 层（纯数据） | strata expert_cache + make_profile.py |
| 专家槽缓存/预取/换入 | `lingxi-driver::expert_cache`（stream/event/host 环） | ninfer moe_expert_cache.cuh + strata adapt |
| routing trace 采集 | moe.rs 加 env 开关 dump；bench 子命令聚合 | ninfer ROUTE_DUMP/ROUTE_STATS 门控 |
| MoE hit/miss 拆分 | lingxi-model/moe.rs 改造（纯行集逻辑，无 IO） | strata dst_index + ninfer GPU-only |
| MTP 前向/dense 注意力 | lingxi-model/mtp.rs + lingxi-kernels 新核 | strata mtp.hpp；golden 对拍 |
| draft/verify/Controller/suffix | lingxi-driver 新模块 spec/controller/suffix | strata spec/* |
| paged KV / host KV 降级 | lingxi-driver::kv（后做） | ninfer paged_kv_cache/host_kv_arena |
| 合批调度 | lingxi-driver::scheduler（后做） | ninfer scheduler.h 模型 |
| HTTP/SSE | lingxi-server | ninfer openai_* 语义 |

# 4. 路线图（与当前实际进度对齐）

| 阶段 | 内容 | 来自 | 状态 |
|---|---|---|---|
| P0 | 48 层 prefill + 贪心 decode 主干 bit 级对拍（step/forward 已逐位一致，端到端 argmax 排查中） | lingxi | **进行中** |
| P1 | routing trace + 十折留一命中率；arena 镜像（先 BF16）；槽缓存 + verify + HostRegisterReadOnly；hit/miss 拆分；单流 decode 服务化门槛 | ninfer+strata | 下一阶段 |
| P2 | MTP M0–M3 + suffix（见专家方案文档） | strata | P1 后可并行启动 M0/M1 |
| P3 | OpenAI Chat + SSE + 分词 | ninfer 协议 | 依赖 P1（无缓存 decode 不可用） |
| P4 | 自适应换入 + RouterLookahead；多会话 paged KV/StateImage；TBQ4 容量档 | strata+ninfer | 有真实流量后 |
| P5 | decode 合批调度（4 lane 起步） | ninfer | P4 后按吞吐数据决定 |
| P6 | 长上下文（YaRN 转录）/Q4G64 专家核/VMM（若需要） | ninfer | 按需 |

# 5. 明确不做

| 项 | 原因 |
|---|---|
| CPU 专家池（strata 式 AVX 核） | 引入第二条舍入路径，破坏 bit 级确定性；Rust 侧无 CPU GEMM 基建；PCIe 直读+预取先够用 |
| DFlash2/块扩散草稿 | 需要训练蒸馏管线，超出推理框架范围 |
| Responses API / Anthropic 协议首版 | YAGNI，OpenAI Chat 覆盖当前需求 |
| VMM 分段 arena | 同层专家同形，固定槽即可 |
| 全局到达顺序填槽 | strata 实测 2.97% 命中率，反面教材 |
| 照搬 7168 槽/4105 槽等任何数值 | 精度、槽字节、硬件全不同，全部用自己 trace 重标 |
| 在单流正确性未过门提前写调度抽象 | ninfer 调度是被 27B 生产负载逼出来的；我们没有负载前抽象必错 |
| 把 ninfer launcher 式大文件搬进 Rust | 违反铁律 2；按 §3.6 拆模块转录 |

# 6. 贯穿性工程纪律（两套参照共同的血泪，写进验收）

1. **env 门控手术**：缓存/预取/量化/自适应全部默认关，开关打开才生效，默认路径逐字节不变。
2. **静默回退必须打日志**（C8：注册失败静默回退让零拷贝「看起来在跑」）。
3. **fill 必 verify**；读镜像必校验格式字+字节数+边界三重。
4. **多流拷贝 event 未落地不许翻 residency 表**；本 call 在用槽驱逐免疫（Rust 尽量用所有权表达）。
5. **新 kernel 先对 HF/scalar golden 再接线**；批量形状 vs 单列形状必须有 bit 级或容差对拍
   （multi_exact 原则）。
6. **性能参数机上标定**：trace 命中率、接受率 EMA、cost model，先验只用于冷启动。
7. T3 端到端贪心中若遇到 bf16 router 并列翻转，按 [ARCHITECTURE.md](../ARCHITECTURE.md)
   T2 附注处理：逐行归因到并列翻转，不把混沌当 bug，也不把真 bug 塞进「混沌」里蒙混。
