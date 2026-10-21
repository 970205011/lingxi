# lingxi（灵犀）

[English](README.en.md) | 简体中文

lingxi 是一个专门为 Qwen3-Next 架构写的推理引擎，用 Rust 和 CUDA 实现，目标硬件是
NVIDIA sm_70——也就是我们手头的两块 Tesla V100 32G。它能把一个 48 层（36 层 GDN 线性
注意力 + 12 层稀疏全注意力）、每 token 激活 512 选 10 的巨型 MoE 模型，在这种主流
推理栈已不再支持的老卡上跑到可交互的速度：生产配置下单流 decode 稳定在 40 到 64 tok/s
（视上下文与是否开启投机解码而定），多轮对话的第二轮靠前缀缓存可以做到亚秒级响应。
项目以 Apache-2.0 协议开源。

它不是又一个通用 serving 框架。lingxi 服务的是一个很具体的场景：一个或几个人、一块或两块
老显卡、想和这个量级的大模型流畅地对话、跑 Agent、处理长文档。所有设计都围绕
"把单请求的每一毫秒拆开、量清楚、再决定值不值得优化"展开。

## 为什么会有这个项目

故事的起点是一个硬件现实加一个模型现实。硬件现实是：vLLM、TensorRT-LLM、FlashInfer、
DeepGEMM 这些主流方案早已把底线抬到 sm_80，V100 没有 bf16 张量核、没有 FP4 指令，
连 int4 的 dp4a 都在 Volta 上跑不过 4 路 FP32 FMA。模型现实是：Qwen3-Next 这一代
架构（GDN 线性递推 + 每 4 层一次的稀疏全注意力 + 512 专家的 MoE + 95 GiB 的 PLE
外置表 + 自带的 MTP 投机头）和上一代完全不同，llama.cpp 对它的支持还在完善。
这条路并不是没人走过——strata（llama.cpp 的 fork）已经能在同款 V100 上把模型跑起来，
也验证了不少可行做法；但它不是专为 sm_70 逐形状调优的引擎，量化档位也更激进。
要把 GDN、稀疏注意力、PLE 和专家驻留逐项优化到可交互速度，仍需要一套为这张卡
专门设计的引擎。

更具体地说，这个模型的稀疏专家权重在盘上就有 225 GiB（bf16 源）。每 token 要触达
480 个专家；按生产 Q4G64 算是 1.25 GB/token。如果没有一套专门设计的驻留与搬运体系，
decode 会退化成对整个 241 GB 专家池的随机读——根本不可用。ninfer（一个同硬件上被生产
验证过的 C++ 推理项目）提供了"能跑起来"的工程骨架；strata 跑通了同一代模型，并在专家
热度画像、按层准入、投机解码等算法部件上给出了可借鉴的实现。lingxi 的做法是把两家的
成果拆开研究透：产物格式、数值格式、张量布局直接复用 ninfer 的注册表语义；专家热度
画像、按层准入、投机解码的算法框架借鉴 strata 的机制；但执行层从零写——Rust 的模块
纪律、cudarc 安全绑定、以及为 sm_70 逐形状调优的 83 个手写 CUDA 核（88 个 `__global__`
声明，含模板实例化）。

## 性能（全部实测数字）

测试机是两路 Tesla V100-PCIE-32GB（GPU0 走 PCIe x16、GPU1 走 x8，无 NVLink），
251 GB 内存。模型产物 251.5 GB，专家以 Q4G64 存于内存盘 arena。下面这些数字每一个
都出自逐位验证过的配置——这个项目的铁律是"没有回归证据的性能数字不许报"。

| 场景 | 结果 |
|---|---|
| 单流 decode（钉表 + 双卡，p10） | **43 tok/s**（23.2 ms/token） |
| decode + MTP 投机（生产档 kd=5, min_p=0.7） | **约 64 tok/s** |
| 双卡层间切分（真实中文语料 vs 单卡） | 20.7 → **54.1 tok/s**（+161%） |
| serve 短 prompt 投机解码 | **89 tok/s** |
| 875 token 前缀 + 投机 | **38.4 vs 29.9 tok/s**（投机 vs 自回归） |
| 256K 超长上下文 decode + MTP | **59.1 tok/s** |
| 首 token（875 token prompt） | 约 3.8 s（与自回归持平） |
| 首 token（4.2K token prompt，双卡线程版） | **28.7 s**（顺序版 46.2 s） |
| 23.7K token 冷 prefill（双卡） | 143.2 s（6.03 ms/token） |
| 33K token 代码类冷 prefill（调优后） | 278.7 s（调优前 336.9 s，−17.3%） |
| 256K token 冷 prefill | 3436.9 s（57.3 分钟），峰值 31.9/27.3 GiB 未 OOM |
| 同一 256K prompt 重发（前缀缓存命中） | **TTFT 0.00 s，整请求 6.4 s**，输出逐字相同 |
| 多轮对话第二轮（智能前缀缓存命中） | 28.1 s → **0.35 s（81×）** → 原样重发 **0.16 s（177×）** |

关于数字的口径值得多说两句，因为这本身就是项目的一段历史。引擎早期在英文基准
文本上跑到 48.5 tok/s，一度以为接近目标；换真实中文对话当语料后掉到 11.3 tok/s
——真实对话每层要触达 276~411 个不同专家，而英文基准只有 148 个，按基准文本钉的
热专家表覆盖率从 97% 掉到 28%。用真实语料重建钉表后回到 36 tok/s，再叠在线重钉、
钉表转生产默认、投机参数重扫，才稳定在今天的水平。所以上表全部按真实语料、
生产默认配置报告。

另一个诚实的结论：单流 200 tok/s 在这张卡上物理不可达。带宽账算得很清楚——
这个模型每 token 光权重就要搬 4.6 GB，V100 的 HBM 峰值 900 GB/s 给出 196 tok/s 的
理论上限且要求 100% 带宽效率；而 Q4 权重是发射率受限（每字节权重约 13 条解码指令），
V100 的发射能力给出 320~505 GB/s 的现实区间。再往上只有两条路：多请求并发聚合
带宽（ninfer 在同硬件实测 4 路聚合 94.2 tok/s），或者换更小的模型。这两条路的
账也都算过、写在档案里，不靠乐观。

## 容量与显存账（实测）

**单 token 的搬运账**（Q4 已按 0.5 B/参数计；来源 decode-speed-profile §3.29.1）：

| 项 | 字节/token |
|---|---|
| routed 专家（10 × 2.611 MB × 48 层） | **1.25 GB** |
| attention 投影（36 层 GDN + 12 层 QSA） | **~1.25 GB** |
| lm_head（bf16，单块最大） | **1.27 GB** |
| MoE shared + router（bf16） | **0.60 GB** |
| GDN 递推状态（读 + 写） | **0.23 GB** |
| **合计** | **≈ 4.6 GB** |

4.6 GB ÷ 900 GB/s = 5.1 ms/token = **196 tok/s**，这是 V100 的物理地板（要求 100%
带宽效率）；实测 26.7 ms 对应 **172 GB/s 有效带宽（19%）**。

**容量账**（产物头 + 实测）：

| 量 | 数值 |
|---|---|
| 专家总数 | 24,576（48 层 × 512） |
| 单个 Q4G64 blob | 2,611,200 B（2.49 MiB） |
| 全量 Q4G64 专家 | 59.8 GiB（arena 文件 60.25 GiB） |
| 混档 arena（热 Q4 + 冷 Q3） | 46.39 GiB（−23%） |
| 设备槽 | 7000 槽 = 17.0 GiB；9000 槽 = 21.9 GiB |
| 非 arena 占用 | ≈9.4 GiB（全部 dense + KV/状态 + MTP 1.3 GiB + workspace，实测反推） |
| KV | 24 KB/token：32K ≈ 786 MB、256K ≈ 6.3 GB（双卡各一半） |
| indexer | 3 KB/token：256K ≈ 786 MB，另加 pooled 1/4 |
| GDN/PLE 会话状态 | ≈113 MB/份，与上下文长度无关 |
| PLE 表 | bf16 95.4 GiB（host）/ FP8 51.2 GB（/dev/shm） |

**实测显存峰值**（nvidia-smi / 启动横幅）：

| 配置 | 槽位 | 实测显存 |
|---|---|---|
| 单卡 Q4，9000 槽满载 | 9000 | **31.3 / 32 GiB**（剩 740 MiB） |
| 双卡 32K + MTP（钉表） | 9600 / 10000 | **29.9 / 27.8 GiB** |
| 双卡 32K + 投机，875-token prefill 峰值 | 8800 / 9200 | **31.3 / 28.6 GiB** |
| 双卡 256K 配方启动 | 7200 / 7600 | **23.7 / 23.2 GiB** |
| 256K，47K prefill 峰值 | 7200 / 7600 | **27.7 / 26.8 GiB** |
| 256K，250.6K prefill 峰值 | 7200 / 7600 | **31.9 / 27.3 GiB**（未 OOM） |

显存可以按同一口径复算：

```text
显存 ≈ 槽数 × 每槽字节
     + dense/head（W4 约 2.4 GiB；bf16 约 9.2 GiB）
     + KV（24 KB/token）+ indexer（3 KB/token）
     + GDN/PLE 状态（≈113 MB/会话）
     + MTP（draft 835 MiB + Q4 专家 1.3 GiB）
     + workspace / cuBLAS / 驱动（≈2~3 GiB）
```

9000 × 2.49 MiB = 21.9 GiB，加上实测非 arena 9.4 GiB ≈ 31.3 GiB，与采样一致。

## 工程上做了什么

这一节是项目的主体。按子系统讲，每个设计决策背后都有实测对比，正结果和负结果
同样留档。

**一页收益表**（全部为同机 A/B 实测，详细记录见 docs/）：

| 改动 | 基线 → 改动后 | 结论 |
|---|---|---|
| arena 整段锁页 | p50 8.8 → **34.3 tok/s** | +290% |
| 离线钉表 N=200 | 30.43 → **24.36 ms/token** | −20% |
| 在线重钉 | 命中 84.8% → **88.2%**；32.5 → **38.1 tok/s** | +17% |
| 混档 arena（热 Q4 / 冷 Q3） | 60.3 → **46.4 GiB** | −23%，端到端计时打平 |
| MTP 概率闸门 min_p=0.7 | 18.2 → **27.6 tok/s** | +52% |
| MTP 全链（闸门 + 重放两刀） | 17.0 → **约 64 tok/s** | +276% |
| 双卡 layer-split | 20.7 → **54.1 tok/s** | +161% |
| prefill 线程版（4.2K prompt） | 46.2 → **28.7 s** | −38% |
| 多轮前缀缓存 | 28.1 → **0.35 s** | 81× |
| 原样重发 | 28.1 → **0.16 s** | 177× |
| 256K 冷 prefill | **3436.9 s（57.3 min）** | 峰值 31.9 / 27.3 GiB，未 OOM |
| 256K 前缀命中 | TTFT **0.00 s**、整请求 **6.4 s** | 复用 250,552 token |

### 模型布局与双卡切分

模型是 48 层混合架构：`i % 4 == 3` 的 12 层是 QSA 稀疏全注意力，其余 36 层是 GDN
线性递推（`crates/lingxi-model/src/lib.rs::layer_kind`，从第一版起没有变过）。层内张量
按 kernel 访问模式离线预排布（`row-split-k128-v1`，K 对齐 128、平面 256B 对齐），
运行时不做转置、不反量化、不融合。

双卡默认在 **lo=24** 处层间切分：GPU0 跑第 0–23 层并持有 embedding / head / MTP，
GPU1 跑第 24–47 层；KV 与 GDN 状态按层各持一半。槽位由 vram-autotune 逐卡算账
（生产实测 GPU0 8984 / GPU1 10218 槽）。24 不是拍脑袋：28 也试过（冷 prefill −5.2%），
但 33K 冷 prefill 从 336.9 s 降到 278.7 s（−17.3%）靠的是 **lo=24 + chunk 1024**；
同配置 chunk 1536 会 OOM（GPU0 32.4 G）。真实语料上双卡 20.7 → 54.1 tok/s。

布局相关的两次“看起来有收益”的改动都被实测否决：按专家粒度重排 arena（真实粒度
收益≈0，不动产物格式）、prefill 改 layer-major（33K 慢 7.8%，默认关）。

### 数值验证体系：从第一天起就不是"能出字就行"

引擎的每个组件——GR 门控残差、MoE、QSA 注意力与 indexer 选块、GDN 线性注意力、
95 GiB 的 PLE 表——都逐一对拍过 HuggingFace 参考实现的 bf16 前向，48 层整层
golden 对拍全过（多数层 5/5 行在 1.5e-2 以内；深层个别越界行被归因到 MoE router
的 bf16 logits 精确并列，并用同输入重跑 HF 证伪了"引擎缺陷"）。在这个地基之上，
每次性能改动都要过一个位级判据：固定 prompt 的首 token id 必须是 11751，贪心序列
逐位不变。这套判据后来固化为三级验证（T1 每 kernel 对 torch 参考、T2 逐层 golden、
T3 端到端贪心）加 4 道护栏 gates（混档字节级对拍、serve×paged 三场景、分页窗口
耗尽必须响亮报错），任何重构——包括两次千行级文件拆分——都以 gates 全绿作为
合入条件。

这个体系抓到过的最有代表性的 bug 是 MTP 投机头的移植：草稿接受率从 0/64 开始
排查，黑盒否掉九条结构假设、白盒对拍三个组件全部"通过"（后来证明那是拿同一份
错误产物权重比 Rust 的循环论证），最终用两条先验判据破局——"RMSNorm 的 gamma
必须全为正"抓到转换器漏了 MtpStemNorm 的 wplus +1（该模型族存的是 γ−1），
"qkv 行多重集必须等于 [q;k;v]"抓到行序里丢了 256 行 q、重复了 256 行 k。
两个 bug 修完，接受率从 0 到 0.906 再到 1.000。这两条判据如今固化在转换器
（tools/convert/hf2lingxi.py）里成为自检。

### 专家权重驻留：这个项目真正的主战场

MoE 推理的生死线是专家权重的容量与搬运，围绕它的一整套体系是 lingxi 投入最大的
部分：

专家权重以 Q4G64 压缩（每个 blob 2.61 MB，是 bf16 的 7.53 倍）存在内存盘的
arena 里，GPU 上只驻留一个 LRU 槽缓存。启动时 vram-autotune 按"权重 + KV + GDN
状态 + 预留"逐卡算账自动定槽数（双卡实测 GPU0 8984 槽 / GPU1 10218 槽），不用
人手调。锁页策略按介质判定：读 /proc/mounts 确认 arena 在 RAM 盘上就默认整段
锁页（实测 p50 从 8.8 提到 34.3 tok/s——锁与不锁差 4 倍，而不锁时 p50 与 p10
的离散度高达 139%），在机械盘上则一律禁止，因为整段 cudaHostRegister 会把盘
的固件压死（这是 10 月 14 日一次真实事故换来的规则，只支持"整段或 0"，中间值
启动直接报错）。

哪些专家该常驻不是拍脑袋，是数据决策：离线用真实流量统计每层的专家路由频次，
启动时把最热的一批钉进槽里（钉表转生产默认后 p90 从 36.84 ms 降到 24.82 ms，
长尾基本消失）。引擎还会在服务过程中持续累积频次表，支持在线自适应重钉
（实测命中率 84.8%→88.2%，真语料 +17%）。这里有个值得讲的数据发现：一次槽
miss 的代价不是 2.6 MB 除以 PCIe 带宽——那只有约 200 µs——而是 370~450 µs 的
host 编排与同步停顿；冷启动与稳态也必须分开看（N=16 时 miss 惩罚 11 ms/token，
稳态只剩约 1 ms）。

在 Q4 基础上还有一层"混档"：按路由频次把专家**逐槽**混布，热专家 Q4G64、冷专家
Q3G128（3-bit），整个 arena 从纯 Q4 的 60.3 GiB 压到 46.4 GiB（−23%），而端到端
计时与全 Q4 基线打平。混档不是选两个格式就完事——它有一条完整验证链：字节级
对拍 gate 保证混档 arena 与单格式正本逐字节一致（量化器是确定性的，所以必须
一致）；Q3 位流（dense LSB-first）有独立的 decode/prefill/batch 三形态内核；
消费端用双视图循环按格式段拆分调用。在此之上还做了 GPTQ 式 one-shot 离线补偿
（OBQ 逐列传播，TF32），用 Hessian 特征分解在输出空间量误差（而不是自欺欺人的
权重空间 rel L2），同 bpw 误差显著收敛；不过因为计时与 RTN 无差，默认档维持
RTN，GPTQ 版作为即插即用备选——"不静默翻生产档"是这个项目的另一条纪律。

### MTP 投机解码：从必亏到 +140%

这颗模型自带 MTP 头，但"自带"不等于"能用"。投机的经济账在项目里被算到了极致：
顺序 decode 每行 26.8 ms，而投机每多验证一行要多付一次专家 H2D（草稿全是新
token，槽 miss 率翻倍），实测投机轮 218 ms 里有 99 ms（45%）是专家搬运——
所以"消灭专家 H2D"不是优化项之一，是投机的门票。经济账还证明了一个反直觉结论：
行成本结构不变时，投机在任何草稿长度 k、任何接受率 p 下都不赢（kd=8 接受 5.8/8
反而更慢，因为每多一行的边际成本比多收的 token 贵）。

这组数字和显存容量强相关，结论只成立于"专家装不下、每多验证一行就多付一次 H2D"
的成本结构。把账算开：全部专家 Q4G64 约 60.3 GiB，加 dense/head（W4 约 2.4 GiB）、
MTP（约 2.1 GiB）、32K KV（约 1.1 GiB）、GDN 状态与工作区/驱动预留（约 3.4 GiB），
单卡约 69 GiB；256K KV 下约 77 GiB。也就是说**单卡 80 GB 是全部专家常驻的入门线**
（32K 勉强、256K 紧），96 GB 才从容；双卡层间切分则每卡约 35–41 GiB，**2×48 GB 足够**。
到了这个容量，验证行的边际成本回到纯算力，投机在任意 k/p 下都不亏——速度意义上
"无损"；输出质量本来就是无损的（概率闸门 + 零回滚，生成序列逐 token 不变）。
这时上限只剩 MTP 头本身：它只训练过一步，自喂链第 1 位接受率 0.57，所以现实收益
仍是 1.35× 量级，而不是论文里的 2×。

最终让投机翻盘的四件事：一是概率闸门（min_p=0.7，用草稿自己的 top-1 概率当
停链信号，投机 18.2 → 27.6 tok/s，且只砍无用行、生成序列逐 token 不变）；二是
概率闸门的前提——钉表把专家 DMA 压下去之后，kd=5 才反超 kd=6；三是 MTP 重放
路径的两刀（砍掉重放里的 head 计算，−59% 重放时间；prob1 改 top-K 廉价近似，
19 次 exp 顶掉 248K 次）；四是发现这颗 MTP 头只训练过一步——自喂链上第 1 位
接受率塌到 0.57（喂主干真值时是 0.93），这个事实终结了"链式加长"的幻想，
也把方向指向"投机值 1.35× 而不是论文里的 2×"的诚实预期。累计下来投机从
17.0 tok/s（比顺序还亏）做到约 64 tok/s。

### 长上下文与 prefill

长上下文的三场手术都在注意力上。第一刀是算法级 bug：稀疏注意力核内层循环对
每个输出维度整条重算点积，复杂度 O(head_dim²·Tk) 而非 O(head_dim·Tk)，
256 倍冗余，占整机时间 44%——这是"总时间对不上微基准加总"时逐核普查抓出来的
（微基准 tk=4096：32.6 ms → 1.3 ms）。第二刀是把注意力从"逐行扫全部 KV"改成
"只 gather 选中的块"（512 块 = 2048 token）：8K 上下文的单层注意力从 143 ms 且
随上下文线性涨，变成恒定约 28 ms。第三刀是 QSA 选块的流式基数选择（O(nblocks)
直方图替代 O(nblocks²) 全排名，零 scores 缓冲）加并行打分，select+fill 从
293~335 ms/层 降到 41 ms/层（7~8 倍），再叠 q-head 合并读 K/V（12 个 q-head 共用
2 份 K/V，单层省 49 GB 流量），32K 冷 prefill 再降 10.4%。

prefill 侧的关键是 95 GiB 的 PLE 表。它一度躺在机械盘上，每 token 120 次随机
pread，冷态一次行读 60 ms（暖态 11.3 ms，5.3 倍差距）——这是"prefill 越跑越
慢"的真凶。解法是把官方 FP8 档（51.2 GB）抽到 /dev/shm 全内存，加四级解析
（显式指定 > shm FP8 > shm IQ4_NL > 磁盘 bf16 响亮兜底横幅），配合分块 prefill
（峰值显存 O(chunk)，33K 代码类冷 prefill 靠 chunk 1024 + 层切分 24 调优到 278.7 s，
miss 数降 42%）和后台顺序预热线程。256K 全量 prefill 的实测成绩是 57.3 分钟、
峰值 31.9/27.3 GiB 未 OOM；而同一 prompt 重发时前缀缓存复用 250,552 个 token，
TTFT 为零，6.4 秒出完整回答。诚实的边界也写在档案里：250K 以上全额 KV 在这
64 GB 显存上 OOM，单请求建议 ≤32~64K，更长靠前缀缓存加多轮。

### 服务层

serve 是零新依赖的手写 HTTP/SSE（OpenAI 兼容 /v1/chat/completions，支持
reasoning_content、工具调用、流式增量反分词、长 prefill 保活）。跨请求前缀
缓存有两档：精确匹配档只做纯 prefill 前缀（TTFT 466 倍）；"智能"档在消息边界
回退、把 GDN/PLE 状态整段 checkpoint（约 115 MB/份），多轮对话第二轮 81 倍
提速；不活跃会话整段落盘（原子写 + LRU），重启后秒级恢复。v1.5 起支持并发
多会话：连接线程与引擎线程轮转步进，每个物化会话持有独立的 CUDA Graph epoch，
会话间切换零数据搬迁。投机解码按请求自动启用——冷请求走投机，命中前缀缓存
的请求自动退回自回归（实测同 prompt 第二次 0.26 s、spec_rounds=0）。

### 测量与证伪：这个项目的方法论资产

如果把 docs/ 里的工程档案翻完，会发现差不多同样多的笔墨花在"怎么量"和"哪些路
被证伪"上。这是刻意的：每一轮"看起来有依据"的优化（scratch 池消 alloc 风暴、
融合 GR 的 8 个核、shared-x 的 Q4 GEMV、加专家槽到 9500、双卡 P2P、CPU 专家池、
dp4a、attention 段进 CUDA Graph……）都先做完完整实现再实测，零收益或更慢的
全部留档否决，防止后人重踩。测量工具本身也被反复证伪过：SKIP 剥块会系统性
高估被剥段（它连带去掉了同步栅栏）、分段计时因排空偏差偏高 26%、重复 prompt
token 的前向基准天生测不到专家 DMA、p10 会掩盖长尾尖峰（PLE 换话题时一次
8.65 ms 的机械盘寻道，让真实均值比 p10 差 30%）、逐元素同步读回会让观测手段
本身变成竞态源。这些教训最后都固化成了规则：A/B 必须同进程同状态交错跑、
以 md5 确认换入的是目标二进制、会改数学的消融只能定"有没有"不能定"多少毫秒"。

## 与现有方案的关系

| 维度 | vLLM / TRT-LLM / FlashInfer | llama.cpp | lingxi |
|---|---|---|---|
| sm_70（V100）支持 | 不支持（要求 sm_80+） | 可运行 | 一等公民，全套内核逐形状调优 |
| Qwen3-Next 语义（GDN/PLE/MTP） | — | 仍在完善 | 完整，48 层逐层对拍验证 |
| MTP 投机解码 | 不支持该架构 | 无 | 有（概率闸门 + 零回滚提交） |
| 专家权重逐槽混布 + GPTQ 补偿 | — | — | 有，字节级对拍 gate 背书 |
| 超显存窗口 paged KV | 有（新卡路径） | 无 | 有（块表间接寻址 + 锁页块池） |
| 数值验证 | 精度测试 | 抽样对拍 | T1 kernel / T2 逐层 / T3 端到端逐位 |

### 外部对照：200 tok/s 的口径与差距

同机同模型的外部参照做过，但必须先对齐口径：strata 公开的 **200.8 tok/s 是峰值**，
同一份记录里 A/B 基准中位数 **104.5**、真实 Agent 任务中位数 **159.2**；它跑的是
IQ3_S 82.8 GB（约 3.4 bit），按参数量×位宽计每 token 搬约 **2.55 GB**。lingxi 生产档
按同一口径是 **3.3 GB/token**，真实语料双卡稳态 **54.1 tok/s**。拿 200.8 的峰值比
54.1 的稳态会夸大差距；按中位口径是 104.5 vs 54.1 ≈ **1.9×**，其中约 **1.3×** 来自
更激进的量化（少搬字节），剩下约 **1.5×** 来自投机链深度、host 调度与内核效率：
strata 能走 8 位，真实文本接受率 63~78%、重复输出可到 93~99%；而这颗 MTP 头第 1 位
就塌到 0.57，我们每轮只接受约 3.1 个 token。

补上同代上游的 V100 社区实测，避免把峰值当常态：单卡 IQ3_XXS 短/长文 **54~72 tok/s**，
Q2_0 **76.4**、IQ2_XS **71.2**、UD-IQ4_XS **26.3**；V100 + RTX 4070 双卡 **75~85**。
也就是说我们双卡 Q4 的 54.1 与 strata 单卡 IQ3_XXS 同档，200.8 是调优峰值，不是
V100 的日常水平。

真正拉开差距的是 **prefill**：strata 在 V100 上 32K prompt 约 **1,200~1,600 tok/s**，
我们 33K 冷 prefill 278.7 s ≈ **118 tok/s**，差约 10 倍。主因是他们用 FP16 张量核做
prompt attention（+15~18%）和 grouped expert GEMM（+10%），而我们的 prefill 仍是
FP32 SIMT 算力受限。另一个结构性差异是量化：2-bit 档让专家几乎全驻留，CPU/PCIe
每层只剩约 0.7 个专家要搬；我们 Q4 混档仍有约 20% 的专家要搬。

单 token 的 26.8 ms 也能拆开：专家 H2D **7.0 ms（26%）**（路由时序决定，只能重叠
不能消）、真核 **11.6 ms（43%）**、host **3.4 ms（13%）**、同步/零头 **4.8 ms（18%）**。
HBM 地板 4.6 GB/token ÷ 900 GB/s = **5.1 ms = 196 tok/s**，所以单流顺序路径物理上
到不了 200；即使核效率拉满（11.6 ms）也只有 **86 tok/s**。再往上只有加并发聚合带宽，
或者换更小的量化/模型——这两条账都写在 docs/decode-speed-profile.md 里。

我们之前在环境里把 strata 当数值 oracle 跑通过，但那次不是这次 200.8 峰值的同一
量化档/同一计时口径；**同一 prompt、同一模型、同一计时口径的两引擎对拍还没做**——
这是下一个版本的第一件事。这一版先把 32G 单卡/双卡的可交互跑稳，把工程档案和
验证体系开源。

## 生态兼容与部署形态

产物以 ninfer v2 容器为基座（magic `NINFER\0\x02`，格式规范见 docs/FORMAT.md）。
**读取侧**：上游纯 ninfer 产物（`.ninfer`）开箱可用。
**命名侧**：按 lingxi 内容约定（qwen4 张量目录、wplus +1、PLE 寻址、混档格式码）
生成/读取的产物用 `.lingxi` 后缀；容器本身不变，读取器按 magic 识别、不看后缀。
上游 ninfer 引擎能解容器，但不认识这些内容约定，所以 `.lingxi` = "给 lingxi 读的
NINFER v2"；现有 253 GB 主产物沿用 `.ninfer` 名字，不做盘上迁移。API 是 OpenAI
兼容的，现有客户端改个 base_url 就能接；分词用模型自带的 HuggingFace tokenizer.json。
离线工具链（tools/convert、tools/arena-gen）可以从 HF checkpoint 重建全部产物，
包括混档 arena 和 GPTQ 补偿的重打包管线（每步带护栏、断点续跑、md5 落账）。

部署上单卡双卡都支持。单卡 V100 32G 就是完整功能（9000 槽 / 32K 上下文 /
投机可开）；双卡默认层间切分，真实语料 +161%，配长上下文档可服务 256K 上下文。
日常用 ./start.sh 起服务，正式部署有 tools/deploy/ 下的脚本（双实例加代理、
长上下文单实例），日志轮转、pid 管理、优雅停机都齐了。

## 快速开始

```bash
make build        # cargo build --release（需 nvcc / CUDA 12.8 + rustc）
./start.sh        # 生产启动器（固化最优 env，talk 冒烟）
./start.sh ./bin/lingxi-bench serve <产物.lingxi 或 .ninfer> [端口]   # OpenAI 兼容 API
make regress      # QA 判据回归（3 项端到端判据）
make gates        # 护栏 gates（混档对拍 / serve×paged / 窗口耗尽报错）
```

```bash
curl http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "lingxi",
  "messages": [{"role": "user", "content": "The capital of France is"}]
}'
```

产物（`.lingxi`，或沿用 `.ninfer`；约 253 GB）不入库，用 `LINGXI_ARTIFACT` 或命令行参数指定路径。
全部运行参数是环境变量，集中登记在 crates/lingxi-model/src/probes.rs（180 余个
开关，带类型/默认/用途/耦合，拼错启动即点名、危险组合直接拒绝——由守卫测试
强制）。从零构建产物见 tools/convert/hf2lingxi.py。

## 项目结构

```
crates/
  lingxi-ids        标识与连接层：跨模块契约的唯一来源
  lingxi-artifact   .lingxi / .ninfer 容器（NINFER v2）读取器/写入器 + 加载期结构断言
  lingxi-plan       由 catalog 生成运行时内存布局（只算地址不做计算）
  lingxi-kernels    CUDA C++ → cubin（build.rs 调 nvcc）+ cudarc 加载/launch
  lingxi-model      前向逻辑 + 模型静态描述 + 张量角色推导 + 开关注册表
  lingxi-driver     slot 池 · 准入 · decode 循环 · MTP · 前缀共享 · KV 预算账本
  lingxi-server     HTTP/SSE（OpenAI 兼容）· 分词
  lingxi-bench      产物检查 · 压测 · golden 对拍 · serve 引擎
```

依赖方向单向无回环（ids → artifact/plan/kernels → model → driver → server），
unsafe 只允许出现在 lingxi-kernels。设计与调试纪律见 ARCHITECTURE.md。

## 文档索引

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构硬约束、模块边界、验证分级 |
| [docs/FORMAT.md](docs/FORMAT.md) | `.lingxi` / `.ninfer` 容器与数值格式手册（模型转换必读） |
| [docs/decode-speed-profile.md](docs/decode-speed-profile.md) | decode 逐项性能剖面与 A/B 记录 |
| [docs/speedup-audit.md](docs/speedup-audit.md) | 优化项逐条审计（含被证伪项） |
| [docs/inference-framework-design.md](docs/inference-framework-design.md) | 框架设计与 ninfer/strata 取舍 |
| [docs/weight-compression-plan.md](docs/weight-compression-plan.md) | W4 / 混档 / GPTQ 量化路线 |
| [docs/strata-mtp-expert-cache-plan.md](docs/strata-mtp-expert-cache-plan.md) | MTP 与专家缓存方案 |
| [docs/PLAN.md](docs/PLAN.md) / [docs/PLAN_NEXT.md](docs/PLAN_NEXT.md) | 工程决策与里程碑全记录 |
| [docs/BACKLOG.md](docs/BACKLOG.md) | 去重审计后的待办与已知问题 |
| [docs/HANDOVER.md](docs/HANDOVER.md) | 运维与排障手册 |
| [docs/ASSETS.md](docs/ASSETS.md) | 资产清单与再生命令 |
| [docs/bench-logs/](docs/bench-logs/) | 关键实测日志原始数据 |

> 文档中的 `$MODELS`、`$LINGXI_ASSETS`、`$DATA`、`$CUDA_LIBS` 是路径占位符，
> 分别指模型产物根目录、资产根目录、数据盘、CUDA 运行库目录。

## 参考与致谢

产物容器格式、数值格式注册表、张量布局和转换器复用 ninfer 的成果（lingxi-artifact
是我们自写的兼容读取器，0.01 秒打开 252 GB 产物）；专家热度画像与投机解码的
算法框架参考 strata；量化格式的对照基线来自 llama.cpp；混档补偿用的是 GPTQ/OBQ
算法；CUDA Graph 计时探针机制参考 DeepInfer；生态库用了 cudarc、axum、tokenizers。
Qwen3-Next 的模型语义来自 HF 上的 orcarouter/Qwen3.8-Flash-Next-Uncensored。

## License

Apache License 2.0（见 LICENSE）。第三方依赖各自许可证见其仓库。
