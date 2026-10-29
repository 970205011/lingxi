# 性能实测（BENCH）

本文只保留最终性能结论。所有数字均为同一测试机上的生产默认配置实测；
方法上，每个性能配置都通过位级判据链验收（T1 每 kernel 对 torch、T2 逐层
golden、T3 端到端贪心逐位一致，外加护栏 gates），优化前后生成序列逐 token 不变。

**测试环境**：2× Tesla V100-PCIE-32GB（GPU0 x16、GPU1 x8，无 NVLink），Xeon 双路
主机，251 GB 内存，CUDA 12.x。

## 端到端最终数字（生产 Q4 默认档）

| 场景 | 结果 |
|---|---|
| 单流 decode（双卡层间切分，p10） | **43 tok/s**（23.2 ms/token） |
| decode + MTP 投机（kd=7, min_p=0.7；英文列举 200 tok） | **96 tok/s**（接受 4.08/轮、39 轮、轮 p50 58 ms） |
| 同上，代码提示 128 tok / 中文提示 128 tok | **85 / 62 tok/s**（中文接受 1.47/轮；自适应 kd=7→3 + 墙钟关闸） |
| 双卡层间切分（真实中文语料 vs 单卡） | 20.7 → **54.1 tok/s**（+161%） |
| serve 短 prompt 投机（冷请求） | **89 tok/s**；warm/多轮请求按预算选路也走投机（57tok prompt + 200tok：wall 3.2s ≈ 58 tok/s） |
| 875 token 前缀 + 投机 | **38.4 vs 29.9 tok/s** |
| 256K 上下文 decode + MTP | **59.1 tok/s** |
| 首 token（4.2K prompt，双卡线程版） | **28.7 s**（顺序 46.2 s） |
| 4167 token prefill（默认路径，nvprof 校正后） | **~10 s**（≈425 tok/s；GPU kernel 每卡 4.0s，busy ≈40%，余量为双卡结构性互等） |
| 23.7K / 33K token 冷 prefill | 143.2 s（6.03 ms/tok）/ 278.7 s（−17.3% 调优后） |
| 256K token 冷 prefill | 3436.9 s（57.3 min），峰值 31.9/27.3 GiB，未 OOM |
| 同一 256K prompt 重发（前缀命中） | **TTFT 0.00 s，整请求 6.4 s** |
| 多轮对话第二轮 | 28.1 s → **0.35 s（81×）** → 原样重发 **0.16 s（177×）** |

单流 200 tok/s 在 V100 上物理不可达：每 token 需搬约 4.6 GB，900 GB/s 峰值给出
196 tok/s 的理论地板（要求 100% 带宽效率）。

## 内核层最终账（prefill 4167 tok 时间预算，每卡 192 层 pass）

| 段 | 每卡合计 | 每层层均 | 备注 |
|---|---:|---:|---|
| GR 两段（down/up ×2 + 小件） | 1.19 s | 6.2 ms | topk/路由并行化后从 15.05 ms 砍到 6.2 ms |
| attn 段（GDN 3.85 / QSA 6.4 加权） | 0.86 s | 4.5 ms | QSA gather prefill v3 重排后 attn 段全 chunk −34~36%（逐位等价） |
| MoE 专家段 | 1.34 s | 7.0 ms | q3g128 4.25 / q4_dense 1.60 / q4g64 0.73 / 其余 ~0.4 s |
| GDN rec 流水版 | — | −10.9% | 逐位等价，默认开 |
| PLE + 小核 + 拷贝 | 0.57 s | 3.0 ms | 含专家 HtoD 仅 0.47 s（驻留命中 97.8~98.4%） |
| **GPU kernel 合计** | **3.96 s** | **20.6 ms** | GPU busy ≈40%；8s 以下需改并行结构（GDN 状态并行等路线评估中） |
| host/延迟（簿记 + topk + launch 链） | 1.9 s | 5.0 ms | 层间 ~2.7ms/层，深队列化可压 |
| **墙钟** | **9.8 s** | 25.5 ms | 双峰：快峰 9.6~10.0 / 慢峰 12.8~13.1 |

## 可选实验档（默认关，全部经质量抽检）

- `LINGXI_PROJ_TC=1`：bf16 投影张量核化（Volta mma），逐核真 GPU 时间 **−97%**
  （1211 ms → 37.2 ms/次 prefill），有损（不一致位 0.04~0.17%、max_abs ≤3.9e-3、
  0 NaN/Inf）。仅 bf16 档有意义（Q4 档投影本就是张量核）。
- bf16 质量模式（`LINGXI_DENSE_Q4=0 LINGXI_DENSE_Q4_QSA=0 LINGXI_HEAD_Q4=0`）：
  权重全 bf16，质量最高、显存占用更大；spec 约 96 tok/s 与 Q4 档持平（接受 4.08）。
- `LINGXI_ROUTE_OPT=1`：路由 pair 乐观发射——**负结果留档**，默认关。
- `LINGXI_PREFILL_TWOPASS=1`：路线 A two-pass 双卡并行 prefill——段切分 + 末态
  f32 字节交接的数值判决（t28 集成测试，72 组组合含零态/随机态、变 decay、
  多种 split）两段式 vs 单段式 **逐位全绿（0 bit 不同）**；生产集成默认关，
  需双卡 layer-split 且不与 EP 同开。GPU busy≈40% 的结构性余量即其动机。

另外 2026-10-27 起量化/投影/路由 BIG 等开关全部改为**值感知语义**
（`flag_val`：未设 = 引擎默认，`=0` 真关）——此前"存在即真"语义下
`export LINGXI_X=0` 反而是开，曾导致 A/B 两边实际同档的误判；start.sh
相应不再无条件 export 这些开关。

## 显存（生产 Q4 档，实测）

| 配置 | 实测显存 | 说明 |
|---|---|---|
| 单卡 Q4，9000 槽 | 31.3 / 32 GiB | 32K 上下文 |
| 双卡 32K + MTP | 29.9 / 27.8 GiB | 9600 / 10000 槽 |
| 双卡 256K 启动 | 23.7 / 23.2 GiB | 7200 / 7600 槽 |
| 256K，250.6K prefill 峰值 | 31.9 / 27.3 GiB | 未 OOM |
| 单卡 Q4 + KV int8（32 维组缩放） | 同等槽数再省 ~50% KV 显存 | 真实路径逐位一致 + 投机接受率正常 |

## serve 调度与专家加载（2026-10 批次）

- **Chunked prefill 自适应调度器**（默认开，块 512 tok）：decode 优先 + 防饿死
  （stall >1 s 强制喂 prefill）+ 按竞争流数自适应 burst，单步硬钳 4096 tok。
  实测两个优势同时拿到：小请求等待从 578 s 降到 40 s 量级，大请求仅慢 ~8.5%。
  块间固有成本 = 每块全部层权重 DRAM 重读（与块大小无关的地板）。
- **GPTQ 校准专家权重**（生产默认）：10^5 级校准 token 逐层二阶校准后质量可见
  提升（长文档问答从乱码级复述到准确引用），投机接受率与 RTN 档一致。
- **MoE 同层 top-k 专家 write-through 预取**（`LINGXI_ARENA_TOPK_FETCH=1`，默认开）：
  专家 miss 时同层其余 top-k 专家（≤2 个）走第二 DMA 流异步补取，复用 pending
  认领机制避免重复搬运。单卡 decode 17~24 → ~20 tok/s 档内的稳定项。
- **KV FP8-G32 压缩**（`LINGXI_KV_FP8=int8|e4m3`，默认关）：每 32 维组一个 f16
  scale，sink 16 + recent 128 残差 bf16 窗（KIVI/KVSink 思路）。**int8 档经逐位
  判据转正**（量化平面真实生效 + 投机接受率正常）；e4m3 档判负——解码路径非有限
  （prime 成功但投机 logits NaN）且与 int8 同体积零收益。

## 探索留档（判负项，均实测后关闭）

- 路由 logits 预测式专家预取：真实 logits 预测命中率 10.1%，经济账不闭合，
  默认关（`LINGXI_ARENA_PREDICT`）。
- IVFFlat 路由预筛：top-8 路由熵 5.1~5.8 nats 太弥散，64 候选 k-NN 上界分歧 33%，
  原理性判负。
- Q3 全量专家档（单卡塞 512 专家）：窗口内可启动但冷路径显存超限判负；
  单卡天花板 ~20 tok/s 受限于专家字节体积（2.6 MB/个），提速路线在专家格式密度。
- `LINGXI_ROUTE_OPT=1` 路由 pair 乐观发射：负结果，默认关。
