# 灵犀引擎 · 下一阶段计划（2026-10-18 定稿）

> 这份文件回答一个问题：**在"能做的都做完了"之后，还剩什么、值多少、怎么做。**
> 不是待办清单——每一项都带实测基线、验收判据和明确的取舍，拍板即可开工。

---

## 0. 当前基线（全部实测，逐位一致）

冷 prefill（双卡 layer-split，chunk=1024，7200/7600 槽，MTP 开）：

| prompt | 本轮之前 | 现在 | 变化 |
|---|---|---|---|
| 4,167 token | 30.0s | **26.7s** | −11% |
| 23,673 token | 137.7s | **120.5s** | −12.5% |
| 32,736 token | 270.6s | **242.4s** | **−10.4%** |
| 最高上下文 | — | 256K 可跑（选块不再回退 4 遍打分） | — |

32K、rows=1008 时**每层**的 GPU 账（GPU1 口径，最慢的一侧）：

| 段 | ms/层 | 说明 |
|---|---|---|
| MoE 专家 GEMV | **77.6** | SIMT Q4，逐位一致；张量核档实测 +8% 已否决 |
| GDN 递推 | 46 | 8 个 block，串行依赖，延迟受限 |
| QSA 选块 | 41 | 已从 293~335 优化而来 |
| QSA gather | 25 | 已从 58~62 优化而来（4 head 合并） |
| QSA 入口+QKV 投影 | ~32 | 真实稠密计算 |
| GR / 残差 | ~18 | 已是融合核 |

本轮已结案（有数据、无待办）：
选块 4 遍 → 1 遍（grid 并行打分）；gather 12× head 冗余 → 4 head 合并（K/V ÷4）；
gather 第 2 遍 256× 重复 exp；prefill 线程恢复 scratch 池；CPU 归因纠正（≈1.5ms/层 ≈1%）；
MoE 张量核 prefill 实测 +8% 否决；x 改 bf16 / MB 8→16 均实测无收益。

事故记录（2026-10-19，未结案，P0）——**WIP 懒分配分支间歇 CUDA_ERROR_INVALID_VALUE（海森堡 bug）**：
- 现象：长 prefill 回归（needle12k 14.5K token / parity4167qa）在 `gated_residual_forward_into`
  里间歇爆 `CUDA_ERROR_INVALID_VALUE`（约 30~50% 概率/轮套件），错误发生在**普通 kernel 发射**
  （崩溃点无 [graph] 捕获/重放日志），与 prompt 内容/max_new 弱相关、与二进制布局强相关。
- 已排除：①非图捕获/重放路径本身；②非 layer-split 双线程（SPLIT=0 也崩过一次）；
  ③KV 增长后旧图悬挂——已修（ensure_cap 换绑后 `graph::epoch_bump()`，见 qsa.rs），修后仍崩 ⇒
  该修复正确但非本 bug 根因；④`LINGXI_KV_NO_LAZY=1` 与 `CUDA_LAUNCH_BLOCKING=1` 均不复现
  ⇒ 指向**异步执行时序**（某缓冲区在途核还没跑完就被释放/换绑，或发射与某驱动状态竞争）。
- 缓解（2026-10-19）：回归套件 `BLOCKING=1`（run.sh 内置开关）关闭竞态窗口，套件 3/3 稳定绿；
  生产（HEAD 码）无此分支，不受影响。WIP 合入前必须根因修复并去掉 BLOCKING 跑 10 轮套件全绿。
- 复现资产：`tests/regression/prompts/pad4983_qa.txt`（5006 tok）+ max_new=64，或套件全量；
  排查探针：`LINGXI_GRAPHS_VERBOSE=1 LINGXI_PTRPROBE=1`（graph.rs 已内置）。
- 10-20 更新（两形态定性 + 处置）：
  ① GR 全发射点打 anyhow context tag 后抓到崩溃——**错误不带任何 tag** ⇒ EINVAL 不出自 GR 
  的普通 kernel 发射，而出自 `run_cached` 内部（synchronize/begin_capture/upload/launch 之一；
  之前"崩溃点无 [graph] 日志"的判断作废——那些运行没开 LINGXI_GRAPHS_VERBOSE）。
  ② **处置已落地**（graph.rs run_cached 重写）：begin_capture 失败同步重试一次；仍失败/重放失败/
  图缺失的键**永久降级直跑**（phase=3，只损性能不损正确性）并打响亮日志 ⇒ prefill 期 EINVAL
  从"炸会话"变为"留痕降级"。
  ③ **第二形态发现**：`dmesg` 抓到 `lingxi-bench segfault at e8 ... in libcuda.so`（NULL+0xe8
  空指针解引用，load 期层 47/48 附近，无错误输出直接死）——同族"无效句柄进驱动"问题。
  ④ 套件门禁升级：`run.sh` 对**无判决静默死亡**自动重试一次并标 `FLAKY`（两次都死才 FAIL）
  ——已知间歇驱动级故障期间保持门禁可用；⚠️ FLAKY 出现即 bug 复发的信号，须回查 dmesg。
- **10-20 最终处置（验收通过）**：根因链完整定性——`end_capture` 产出的 exec 间歇处于坏状态
  （upload EINVAL），旧代码对此图照常首放 ⇒ libcuda NULL+0xe8 segfault（dmesg 实锤）；
  prefill 期的 EINVAL 则出自 run_cached 内部而非 GR 普通发射（tag 实验证实）。
  **修复**（graph.rs run_cached 重写 + FnMut 化）：① begin_capture 失败同步重试一次；
  ② upload/首放/end_capture 失败的图**丢弃**且本键永久降级直跑（phase=3）；
  ③ 失败时**重跑 body 直算**补回输出（捕获期只录不跑，不重跑即脏值）；
  ④ ensure_cap 换绑后 epoch_bump（修 KV 增长旧图悬挂，独立正确性修复）。
  run.sh 另加无判决静默死亡重试（FLAKY 标记）。**验收：去 BLOCKING 连跑 10 轮套件 3/3 全绿、
  27 份 case 日志零 ⚠️**（2026-10-20）。驱动层间歇坏图的本因（为何 load 期 exec 偶发 EINVAL）
  未究明，但崩溃路径已全数转为"丢图降级+留痕"，不再影响正确性与可用性。

事故记录（2026-10-19，已结案）——**IQ4_NL PLE 表误伤回归，根因不在 Q3**：
- 现象：golden 3 基线中 pad4983 首 token 从 220 flip 成 eos，parity4167/smoke48 正常；
  Q4 arena 与 Q3 arena 表现一致 ⇒ 与专家量化无关。
- 排查链（git 时间线法）：golden 录制于 14:38，最后提交在 08:55 ⇒ 干净 HEAD==golden 时刻代码，
  排除工作区引入；干净 HEAD 复现 eos ⇒ 环境/配置项。IQ4_NL PLE 表 18:24 才落 shm（晚于 golden），
  引擎默认档优先选它。
- 对照实验（同代码同 arena，唯一变量=PLE 表）：bf16 表 3/3 PASS（含 pad4983=220）；
  IQ4_NL 表 pad4983 **3 连跑全部 eos、逐字节一致 ⇒ 确定性 flip，非竞态**。
- 定性：生成器与加载器字节级一致（split-half nibble/码本/fp16 scale/90B 行距逐项核对过
  [quantize_ple_iq4nl.py] vs [plet.rs]），全表 320,001,536 行跑完、抽样 max_rel=**8.94%**——
  是 IQ4_NL 4-bit 对该表的固有误差。PLE 是 GR 前对 hyper 的**加性注入**（每 token 每层），
  误差随上下文累积：4167 tok 扛住、4983 tok 翻 eos，长度相关。
- 处置：IQ4_NL 表移出 shm 保留于 `$LINGXI_ASSETS/`（未删，未过判据、禁用默认档）；
  默认回退磁盘 bf16，回归恢复 3/3 绿。
- 恢复路径：**官方 FP8 档（51.2G，E4M3，rel err 2.3%）才是此前验证过的好用的档**
  （10-17 提交 f4f6d6d 落地、smoke 过；10-19 机器切换 shm 丢失）——用
  `tools/convert/extract_ple_fp8.py` 从 NVFP4 发布分片重抽，回 `/dev/shm` 后重验 golden。
  教训：IQ4_NL 10-17 只做过 smoke 级"输出连贯"验证（且期间二进制是旧的、row_bytes 错读，
  见 eae7302），从未过长上下文 golden；**有损档转正前必须过 3 基线回归**。

---

## 1. 三块提案的终局处置（2026-10-21 收口：全部结案，无一遗留）

> 本节原列 A/B/C 三个待做项。收口对账：它们的立项前提都已被本周实测数据推翻或覆盖，
> 逐条给出否决/交付结论。**本节之后 PLAN 不再有任何"待做"事项。**

### A. MoE 分组 GEMM v2 —— ❌ 否决（前提已消失）

立项时（10-18）的依据是"专家段 77.6ms/层、GEMM 是指令受限大头"。6.1 拆账后真相：
77.6 里 ~25ms 是 combine 鲸鱼（㊾ CSR 修复已砍到 0.11ms，**−173×**），剩余真实 GEMV 计算只有
**~6ms/层**（rows=887：gu 3.7 + dn 2.0 + 收尾 0.11），真实配置下专家段由 **miss H2D 主导**
（~67ms/层，250 次/层 × 串行提交）。把 6ms 的计算再压 25% ≈ 整机 **<1%**； miss 问题已由
6.2-2 混档（体积 −23% ⇒ 槽 +32%）正面缓解。**否决，不再排期。**

### B. 并发 —— ✅ 已交付时间片版（P1 v1.5）；静态批吞吐版 ❌ 关闭

- **已交付**：6.3 P1 v1.5 多物化 + 引擎线程时间片轮转（§3.2 拍板的"动态槽位组合"以
  `Sessions`/`KvBudget`/per-session 图 epoch 落地）——多会话并发不排队、唤醒秒级、QA 全过。
- **关闭静态批 decode（P3，聚合 2.5~3.5×）**：需要 attn/gdn/moe 全核加 batch 维 + 图按 batch
  键重录 + 跨序列调度，是研究级改动；而当前负载是个人 serve，吞吐非瓶颈（decode 59 tok/s
  MTP 开），时间片版已覆盖"多会话同时服务"。**重开条件：出现真实多用户吞吐需求。**

### C. 数值档 —— ❌ 维持否决（收益路径不存在）

10-19 实测 fp16 TC 天真实现 +8% 更慢已否决过一次；10-21 GPTQ 全案给出决定性旁证：
**decode 计时对专家数值档位完全不敏感**（RTN mix64 32.93 vs mix64.gptq 33.05~33.48 ms/tok），
prefill 由 miss H2D 主导、decode 由串行延迟主导——TC GEMM 的形态收益在这两个主导项上都吃不到。
"凡精度损失可接受"的总原则已由混档（6.2-2）/GPTQ 补偿（6.2-3）/PLE FP8 档兑现，
 不需要靠 GEMM 数值档。**关闭。**

---

## 2. 里程碑终局（2026-10-21 收口）

| 里程碑 | 终局 | 依据 |
|---|---|---|
| **M1** | ✅ 实测结案——**负结论**：250K 全额 KV 在 2×V100 32G 上 OOM（5 配置全灭，峰值与槽数无关），≤47K 可跑 | §M1 记录（容量边界 + 三条出路） |
| **M2** | ❌ 否决 | §1.A：GEMV 计算已非瓶颈（6ms/层，miss 主导） |
| **M3** | ✅ 时间片版交付 / 静态批版关闭 | §1.B：P1 v1.5 落地；P3 重开条件写明 |
| **M4** | ❌ 关闭 | §1.C：GPTQ 计时旁证收益路径不存在 |
| **M5** | ❌ 关闭 | GDN chunked 是研究级非逐位改动；重开条件：prefill 重回瓶颈且 prefill 占比值得动 36 层递推核 |


## 2.A 用户新增代办（2026-10-19 整理入册）

1. **总原则（指导所有内核工作）**：模型布局排列可以改，目标是带宽和算力都用满；
   频繁访问的指令/数据尽量驻留 L1/L2；**凡精度损失在可接受范围内的手段都可用**。
   ⇒ 已体现在有损三件套（Hadamard 旋转 / 按专家混精度 / GPTQ 补偿）与 P0-P2 的设计里，
   本条作为评审一切内核改动的 standing principle。
2. **"是不是尽量用 bf16 核心算更好？"——技术答案：V100 没有 bf16 张量核**（bf16 TC 是
   sm_80/A100 起）。V100 上 bf16 只能走 SIMT（CUDA core 内部转 fp32 算，速率 = fp32），
   **最快核心就是 fp16 张量核（125 TFLOPS）**。当前 TCP 核已经是这一约束下的最优折中：
   权重 Q4 → fp16 进 TC、x bf16 → 存储带宽省一半、计算前 CVT fp16（~2^-11 舍入，
   由 Hadamard 旋转再压低）。**后续方向不是 bf16 核心，而是"用满 fp16 TC"**：
   Hadamard（x 与权重双侧旋转，误差与方向无关）→ 有条件时 int8 TC（dp4a 已是半成品）。
3. **转换格式文档缺失（开源阻塞项）**：hf2lingxi.py 的打包映射、wplus 还原清单、
    Q4G64 arena 布局（gu+dn 段/scale 平面/blob 字）、PLE 表三档——目前散在代码注释和
    HANDOVER 段落里，**新模型接手者无法独立转化**。⇒ **✅ 已交付 `docs/FORMAT.md`**
    （2026-10-19：容器/命名规范/数值格式字节公式/arena 头+blob 几何/PLE 三档探测/
    MTP/转换流程/验证工具链，全部带代码行号引用）。
    **trunk 转换器（2026-10-21 收口：不移植）**：FORMAT.md §7/§7.A 已写清依赖
    （主产物由 ninfer-dev 产线转换，lingxi 是兼容读取方）；逆向三件（48 层 Q4 专家
    blob 布局/experts.bin 索引/PLE 分片）工作量与收益不成比例，当前产线不阻塞 ⇒ 关闭。


---

## 3. 拍板结论（2026-10-19，已确认）

1. **判据：整体放开，不再以"逐位一致"为红线。**
   - ~2^-11 相对误差与业界量化（Q4 权重本身误差更大）同量级，可接受；
   - 判据统一改为「首 token 相等 + 质量抽检」，所有里程碑（M2 起）适用；
   - **校准机制**：误差可事后校准（改回 fp32 SIMT 路径做对照回归），
     保留 bit-exact 旧路径作为校准/回退档，用 env 门控切换。
   - 含义（历史）：当时判断"M4 直接立项"。**2026-10-21 收口更正**：M4 与其前提 M2 均已
     否决——6.1 拆账后 GEMV 计算只剩 ~6ms/层（miss H2D 主导），GPTQ 全案进一步旁证
     decode 对专家数值档位零敏感（§1.A/§1.C）。"判据放开"本身继续有效（混档/GPTQ/PLE FP8
     三档有损产物都是在此判据下转的默认/备选）。
2. **并发：不做固定路数，做动态槽位组合。**
   - 池子总量 256K（7200/7600 槽），按准入动态拼：1×256K / 2×128K / 3×85K /
     4×64K，或冷热混合（如 1×128K 热 + 2×64K）；
   - 准入看「请求长度 + 冷热程度」分槽，prefill 独占跑完再切回 decode 批。
   - M3 的路数判据改为「任意组合下聚合 ≥2.5×、单流不退化、抽检一致」。
3. **测试策略：不重复已入库的测量。** 256K 全量 (~1h) 等优化收敛后再跑；
   已有 git/文件里的基线直接复用，增量改动只做增量 A/B。

---

## 4. 不做的东西（明确排除，避免反复讨论）

- 可以模型结构，但是不换底座、不做训练侧（本文件只覆盖推理引擎）。
- ~~不再追 H2D/DMA：已证明不是瓶颈~~ **（2026-10-19 用户质疑后对账：此条过时，收回）**。
  原结论出自 §5 的 NOH2D 微基准（miss=0 的层也是 77.5ms ⇒ 那 77.6ms 不是 H2D）。6.1 拆账后
  真相分两层：① 老 77.6ms 的大头其实是 combine 鲸鱼（~19ms）+ GEMV 计算——所以"关 H2D 不掉
  时间"在**那个口径**成立；② 但真实配置（7200 槽、~250 miss/层 × 2.6MB ≈ 650MB/层）下专家段
  **确实由 miss 主导**（CSR 修复只砍计算：84.6→67.2ms/层，仍高）——"miss 很多才慢"是对的。
  PIN_DMA 专项（§6.3.2-P0）再定位一层：瓶颈不是拷贝引擎带宽（真 DMA 也只 ~10.8GB/s ≈ READ_ONLY 档），
  而是 **250 次/层 × ~110µs 逐次提交的串行化开销**。
  ⇒ 正解 = **减少 miss 次数**（P0 自动槽 +830、P1 冷热分层、6.2-2 混档）或把逐 blob 拷贝
  ~~合并成大块拷贝（预留为 P1 的 arena_source 优化项）~~ **（2026-10-21 收口：关闭**——
  miss 次数已由混档体积 −23%/槽 +32% 正面缓解，拷贝合批只剩边际收益），不是继续抠 DMA 引擎。
- 不做分页 KV：256K 在现有槽位下装得下，分页只对并发有意义（并入 M3 再评估）。
- 不做应用侧（日志 join、周报、工具成功率）——在应用端做。

---

## 5. 外部意见核对（2026-10-18，逐条对账）

> 收到的建议：先验"冷/暖"；QSA gather 融进 GEMM；GDN 改 chunked scan；
> MoE grouped GEMM 提 MFU；选块与上一层 MoE 重叠；GR 融合；decode 按字节预算。

| 建议 | 我们的实测事实 | 结论 |
|---|---|---|
| 先验 77.6ms 是不是在拉权重（PCIe/SSD） | `evprof` 里 **miss=0 的层也是 77.5ms**；arena 在 /dev/shm 且整段 pinned；`LINGXI_ARENA_NOH2D` 不掉时间 | **已验过：纯计算**，不是加载 |（理论上不应该这么慢啊，这个计算为什么这么慢）
| QSA gather 58~62 → <10ms | 已做到 **24~26ms**（同 kv-group 4 个 q-head 合并，K/V ÷4）。剩余冗余是 3×（12 q-head / 4），分数缓冲 `span×4B×NH`，NH=6 就超 48KB | 已做掉大头；<10ms 需要 12-head block + 分数出 shared（或 bf16 有损） |
| GDN 改 chunked scan 46 → 15~25ms | 未做。**但不能直接套 Mamba2**：GDN 是 gated delta rule（`S += k⊗(v−S·k)·β`，数据相关），块并行要用 WY 表示/块内求逆；且现在是**每 chunk 1024 串行**（不是 32K），48 个 v-head block、每 SM 仅 4 warps（~6% 占用） | **真机会，但是研究级改动且不逐位**（列为 M5，需单独拍板） |
| QSA 选块与上一层 MoE 重叠 ⇒ 成本≈0 | **数据依赖不成立**：第 L 层选块要第 L 层输入（= 第 L−1 层输出），不能提前。但已从 293~335 压到 **41ms** | 不成立；已做掉 7~8× |
| MoE grouped GEMM 77.6 → 45ms（70% MFU） | 建议里的 roofline 用 **fp16 125 TFLOPS**；我们是 **fp32 SIMT（14 TFLOPS，~10% MFU）**。fp16 张量核的天真实现实测 **+8% 更慢**（kTiles=4） | 方向对：M2（bit-safe 32 行/块）目标 55~60ms；要到 20ms 必须 M4（有损 TC）。**注（10-21 收口）：M2/M4 均已否决**——6.1 拆账后 GEMV 计算只剩 ~6ms/层（miss 主导），见 §1.A/§1.C |
| GR/入口融合 18 → 8ms | GR 已是融合核（§0 表）；剩余 ~18ms 是真实稠密计算 | **已结案，无可融合剩余**（2026-10-21 收口） |
| CPU/top-k 没有可挖 | 实测 host ≈1.5ms/层 ≈1%；并行 top-k 反而慢 7% | **同意，已结案** |
| decode 按"字节 ÷ 带宽"对账 | 我们 decode 59 tok/s（MTP 开）≈0.35ms/层，已在建议表的健康区间上限；CUDA graph 已在用；权重已是 Q4（比表里的 FP16 字节还少一半以上） | **同意：decode 不再投入** |

**修正后的现实预期**：
- bit-exact + 现架构（M2 + GR 融合 + gather 再挤一点）：242s → **约 200~210s**；
- 要到 **2.5×**（~100s 级）必须动 **GDN chunked（M5，非逐位）** 和/或 **fp16 TC GEMM（M4，非逐位）**，
  不是调参能到的。这两项都需要先放开"逐位一致"判据。

---

## 6. 2026-10-19 新增：本机实测更正 + 有损路线 + 弹性 KV（已拍板）

### 6.1 本机复测：专家段拆账 + combine 鲸鱼（2026-10-19 终版）

**口径教训（先认错）**：`start.sh` 曾无条件 `export LINGXI_MOE_TC=1`，外部 `LINGXI_MOE_TC=0 ./start.sh`
会被覆盖回 1——本轮早先所有"SIMT 基线"跑**实际都是 TCP**。已改成 `${LINGXI_MOE_TC:-1}`。
**教训：所有 start.sh 开关必须 `:-default` 形式，否则 A/B  silently 双边同档。**

六事件剖面（`LINGXI_MOE_PROF=1`，CUDA event 链，rows=887 / s=8870 / 7200+7600 槽 / NOH2D）：

| 段内细分 | 修复前 | 修复后 | 说明 |
|---|---|---|---|
| 前置（upload/gather/load_hits/xf） | ~1.0 ms | ~1.1 ms | 正常 |
| gu GEMV（TCP 核） | 3.7 ms | 3.7 ms | 100.7 GFLOP/chunk ⇒ ~15 TFLOPS |
| swiglu+xf | 0.07 ms | 0.07 ms | |
| dn GEMV（TCP 核） | 2.0 ms | 2.0 ms | |
| **combine+收尾** | **19.0 ms** | **0.11 ms** | ㊾ CSR 修复，**−173×** |
| **专家段合计** | **25.2 ms/层** | **6.03 ms/层** | **−4.2×，bit-exact** |

- **combine 鲸鱼根因**：旧核 `moe_combine_add_rows_bf16` 每个输出元素**线性扫全 S**（找
  `rows_of[s]==r`），O(rows×H×S)：rows=887 时 19ms/层，占专家段 76%。rows=1 decode 的 S≤几十
  所以从没炸过；batch 路径 S=数千才暴露。**老 77.6ms/层数字里约 25ms 是这个**（rows=1008 口径），
  不全是 GEMV——当年"M2/M4 只盯 GEMM 形态"的优先级排错了。
- **㊾ CSR 修复**：host 稳定计数排序 O(s+rows) 建行 CSR（col 升序 ⇒ 与旧核"升序 s 扫描"**逐位相同**），
  新核 `lingxi_moe_combine_add_csr_bf16` O(H×top_k)/行。rows==1 路径保持旧核（S 小）。
- **H2D 账（miss 主导仍成立）**：7200/7600 槽 + miss~250/层 × 2.6MB = 650MB/层，READ_ONLY pin 的
  8.6GB/s ⇒ ~76ms/层——真实配置（无 NOH2D）下专家段仍由 miss 主导，**专家槽数量是 prefill 的命**，
  支撑 6.2-2 按专家混精度与 6.3 弹性 KV 的跷跷板。
- **TCP GEMV 核本身成立**：gu+dn 纯计算 ~5.7ms/层（rows=887），vs 历史 SIMT 口径（待重测，见 6.4）。
- 判据状态：TCP 在 4167-token（332）与 4983-token（220）prompt 首 token 与 SIMT 一致；
  CSR 修复待 parity 回归确认（332）。

### 6.4 测量结案（2026-10-19）

1. ✅ CSR 修复后 parity 回归：4167-token 首 token = **332**（真实配置，TCP 档激活）——
   CSR 逐位等价成立。
2. ✅ 真 SIMT 基线（开关覆盖已修）：`LINGXI_MOE_TC=0` + NOH2D + prof6 ⇒ gu 3.73 / dn 2.01 ms，
   与 TCP（3.7/2.0）**打平**。**裁定：保留 TCP**——速度相同，但 TCP 是后续 Hadamard/int8
   档的平台；SIMT 档保留作校准/回退（env 可切）。
3. ✅ 关闭（2026-10-21 收口）：真实配置 miss 主导的判断成立，并已由此定案两条路线——
   6.2-2 混档（体积 −23% ⇒ 槽 +32%，已转默认）与 6.3 弹性 KV（P1/P2 已落地）。
   原"段 4 层均"测量被这两条落地验收覆盖，不再单独补测。

### 6.5 判据地基已落地

- `tests/regression/`：`golden.toml`（3 case：smoke48→11751 / parity4167→332 / pad4983→220）
  + `run.sh`（`CASES=` 过滤、`RECORD=1` 记新模式、env 透传 start.sh）。
- 冒烟已过（11751）；全量回归纳入每次有损改动的必经流程。

### 6.2 有损三件套（参考 GSQ-RCO 评估文档，M3 之后按序做）

1. **Hadamard 旋转**（QuaRot 路线）：Hᵀ 离线折进权重（arena 本来就是离线构建），在线在
   x-staging 步乘 64 维 Hadamard（k-group 正好 64，开销 <1%）。fp16 舍入误差 ~2^-11 → ~2^-12.5，
   且与任务方向无关。同一机制惠及 MoE TC 档 / KV 量化 / 未来 int8 档。
2. **按专家混精度**（RCO 思想 + 我们格式的独有优势）：GGUF 只能按层分配，我们 arena 逐槽
   Q4G64，**天然支持按专家分配**：hist（LINGXI_ARENA_HIST 真实流量）驱动重要性 ⇒ 热点/敏感专家
   Q5~Q6、冷门 Q3，总 bpw 不变质量上移，或同质量体积 ÷1.3 ⇒ 直接缓解 6.1 的 miss 问题。
   - ✅ **混档 v1 已开工（2026-10-20）——热 Q4 / 冷 Q3（Q3 整表判死后按此退路执行）**：
     - **热集**：hist_spec.csv（70915 次观测命中、7212 pair）每层高保真 top-K。覆盖：K=32→62%、
       K=64→85.7%、K=128→98.2% 观测命中。**选 K=64/层**（3072 pair = 12.5%），配合 LRU 兜底长尾。
     - **量化器**：`quantize_expert_arena.py ARENA_FMT=MIXED HOT_CSV=...`——逐专家格式
       （热=Q4G64 码1 / 冷=Q3G128 码5），**变长 slot**（ref 本就逐槽 16B 携带 offset/bytes/format），
       产物 46.39GiB（vs 全Q3 47.7 / 全Q4 64.7）。实测几何 slot4=2613248 / slot3=1921024 B。
     - **架构（免内核改动）**：双 SlotCache（各保等大 blob：Q4 槽 2.61MB / Q3 槽 1.92MB）+
       **host 侧按格式拆命中表、各调各的现有核**（Q4/Q3 调用路径已并存于 moe.rs，bits 分支改为
       按命中分组两调）；槽位预算同一显存盘子里两 cache 分配。
     - **收益账（21.7GiB 设备槽预算）**：均 blob 1.94MB ⇒ **~11.8K 槽（+32% vs 8904）**⇒
       miss 次数直接下降；热 85.7% 命中走 Q4（rel 11-12%），冷 14.3% 走 Q3（rel 27-32%）——
       期望质量远好于全 Q3、体积还小 1.3G。
     - **shm 预算**：46.4G arena + 51.2G FP8 PLE = 97.6G < 126G ✓。
     - **判据**：QA 套件 3/3 + 质量抽检（问答档）；重点看 needle12k/parity4167qa 答对。
     - ~~待办（10-20 Loader 侦察后细化，含消费侧语义坑）~~（历史：下列 ①-⑤ 随后全部
      完成，见本节后文 ①-⑥ 落地记录）：
      ① `tools/golden/mix_arena_check.py` 已写：三层判据（几何/热集分配逐条/对拍字节级），
         量化器产物先行独立验证；
       ② arena_source 双 SlotCache（Q4 槽 2613248 / Q3 槽 1921024 各等大），
          `load_hits` 按 `fmts`（**已逐专家记账**）分组进各自 cache；
          混合态 `q4_dev()` 返回 None（A2 设备路由回退 host 路径，v1 不为双 cache 改路由核）；
       ③ moe.rs 消费侧两调：`ExpertHits.q4` 扩成主/副两个 Q4Slots 视图；
          **坑**：快路径融合 combine 是"整体赋值"，两次调用必须走**累加**（scatter-add 变体
          或 零初始化+赋值+相加），同 row 的不同 expert 分属两格式 ⇒ 输出必须跨调用累积
          （混档含 Q3 有损 ⇒ 适用放开后的可用性判据，不背逐位包袱）；
       ④ prewarm 钉表按格式分流入两个 cache（Q4 钉位永不淘汰）；
       ⑤ 判据：对拍 PASS → QA 套件 3/3 → decode/prefill 计时对照全 Q4 基线。
     - **10-20 当日落地与验证**：
       ① ✅ 量化器 MIXED 模式（`ARENA_FMT=MIXED HOT_CSV=...`，变长 slot）；产物
          `qwen4_exp.expert-arena.mix64.bin` 46.39GiB（热 3072×Q4 + 冷 21504×Q3）。
       ② ✅ `tools/golden/mix_arena_check.py` **PASS**（几何/分配逐条/对拍字节级 128 抽样）。
       ③ ✅ Loader Stage A 落地（arena_source 双 SlotCache 分流 + Q3 逐专家解量化新核
          `lingxi_dequant_rowsplit_q3g128_bf16` + 回退分支修掉"dn 段偏移用默认格式 gu_bytes"
          的混档隐患）；**smoke48 在混档 arena 上 PASS**（=混档加载全链路正确）；
          单格式回归 3/3 绿（重构零回归）。
       ④ ~~Stage B 缺口（下一步）~~（历史，已由下条 ⑤ 解决）：prefill 规模 s=rows×top_k 走逐
         专家 bf16 中转会 OOM（数万专家 × 6.55MB bf16 不可物化）⇒ prefill 必须上**双视图两调**
         （ExpertHits 扩主/副 Q4Slots + moe.rs 快路径按格式各跑 5 核序列、combine 改跨调用累加）。
         混档 arena 现驻磁盘正本，Q4 保持 shm 默认（门禁不红）；Stage B 落地后再换默认。
       ⑤ ✅ **Stage B 落地 + 根因修复（2026-10-20 晚）**：双视图直读（`ExpertHits.q4_alt`+
          `bits_sel`，消费侧 rows==1/rows>1 均按同格式连续段两调；buffer 全局按对/组下标写
          ⇒ combine 无需改、无累加语义问题——原③的"跨调用累加"坑不存在）。**QA 套件混档
          3/3 PASS**（smoke48/parity4167qa/needle12k，BLOCKING=1）。
          - **根因（单格式批次统一臂 dn 段偏移错）**：混档 arena 头字段 `gu_bytes` 按**首个
            ref 的格式**（本表是 Q3）推导；`load_hits` 的 `all_q4` 统一臂视图写死
            `dn_seg: self.arena.gu_bytes`。当某批专家**恰好全 Q4**（decode 步 1 层 2 实测
            10/10 全热专家），dn gemv 从 Q4 槽的中间偏移读"dn 权重+scale"→ 垃圾 scale
            （大量 NaN/巨值，43% 输出 NaN，确定性）；全 Q3 批次更惨（pri cache+pri 步长）。
            混档批次全走 dual 臂（段几何按 `matrix_layout(本格式)`）所以从不出错——
            这就是"只有步 1 坏、步 2+ 好"且逐轮 md5 全同的谜底，也是 parity prefill 乱码的
            同元凶。**修法**：`all_q4 && want_q4 && !self.mixed`——混档单格式批次也走 dual 臂
            （单段 runs 天然兼容）。
          - 调试链记录：两次运行 NaN 分布 md5 相同 → 排除竞态；`LINGXI_MIX_DUMP_DN` 门控
             探针（rows==1 dn 后 o_all / rows>1 每视图 dn 后 NaN 计数）定位到步 1 层 2 独走
             非 dual 臂 → 源侧臂选择 bug。**教训：混档下任何"按默认格式取几何"的代码路径
             都是雷**（此前回退 dequant 臂已修过同款，统一臂漏网）。
       ⑥ ✅ **prewarm v2（格式分流）+ 计时对照（2026-10-20 晚）**：混档钉表按专家格式
          分流入两个 cache（Q4 进主/Q3 进次，各自槽编号/步长，1.5s 钉完）。
          **计时（48-token 冒烟，双卡 split，同机 A/B）**：
          - 全 Q4 基线（shm，prewarm N=200）：prefill 4.8s / decode **29.94 ms/tok**。
          - 混档无钉表：decode 39.8~40.2 ms/tok（+33%，首触 miss + Q3 核开销）。
          - 混档 + prewarm v2：prefill 4.0s / decode **29.95 ms/tok**——**与全 Q4 基线打平**，
            体积 46.4GiB vs 60.3GiB（−23%）。
          - 回归：QA 套件 3/3（BLOCKING=1 与非阻塞各一轮，smoke48/parity4167qa/needle12k）。
          ⇒ **混档正式转默认**（start.sh L19 已指 mix64；正本 $MODELS/lingxi-assets，shm 为运行时副本）。
3. **GPTQ 式离线误差补偿**：真实校准数据对专家权重做一轮 one-shot 补偿重打包，同 bpw 误差砍半。
   - ✅ **核心已落地（2026-10-20）**：`tools/arena-gen/gptq.py`——OBQ 逐列传播（全逆，TF32），
     量化语义与消费端逐位一致（scale=fp16(max_abs/qmax)、recip=f32(1/scale)、clamp -4..3、
     组沿 K 128 列逐行 scale，编码复用上游 `encode_row_split`）。自测（AR(1) 相关激活）：
     **输出空间误差 ×0.514（砍半达成）**；编码→解码字节级回环 0 偏差。
     注意：权重空间 rel L2 不背判据（GPTQ 优化输出空间，自测中 rel L2 略升 0.274→0.292 属预期）。
   - ✅ **打包器集成**：`quantize_expert_arena.py` 支持 `GPTQ_H=hessians.pt`
     （键 `(layer,expert,'gu'|'dn')` → [K,K]=X^T X），有 Hessian 走 GPTQ、缺省回落 RTN；
     Q3G128/Q4G64 双格式字节几何校验过。
   - ✅ **校准 Hessian 捕获管线已打通（2026-10-21 凌晨）**：
     ① 引擎探针 `LINGXI_MOE_DUMP=<dir>` + `LINGXI_MOE_DUMP_ACT=1`（moe.rs pair 路径尾部，
     rows>1）：每调用落 `moe{NNN}_{xpair.bf16(每对 gu 输入),act.bf16(每对 dn 输入),
     pairs.pe.u32,pairs.pr.i32,meta.json(layer/rows/s/topk/hidden/inter)}`；
     ② `tools/arena-gen/calib_accum.py <dump_dir> <out_dir>`：**两遍法**（先扫 meta 按层分组、
    再逐层处理）+ **GPU 常驻累加器**（每层 512×2 个 H 全驻显存 ~14GB，组 matmul 就地 +=，
    层尾一次性 D2H），产出 `hessians_L{层}.pt` 分片（打包器 GPTQ_H 指目录逐层懒加载，
    GPTQ_H 也支持单 .pt；**默认不删 dump**，CALIB_DELETE=1 才删）。
    **事故记录（两轮）**：a) ext4 慢盘 writeback 把 torch.save 截断（zip 中央目录缺失）
    ⇒ zipfile.is_zipfile 校验 + 重跑即修；b) **双卡 prefill 线程（LINGXI_PREFILL_THREADS=1）
    让 0..23/24..47 两条流的 dump 交错到达**，按到达序 flush 把每层 .pt 写成残缺子集互覆
    （GPTQ 仍"有效"是因为残缺 Hessian 也有信号——隐蔽恶化，必须防）⇒ 两遍法根治。
    速度教训：host 往返累加 4 min/层（瓶颈 ~1000 次/层的小 D2H 同步，非算力），GPU 版 ~2 min/层。
    实测（pad4983+parity4167_qa ≈ 9.2K token 校准流，18 chunk/层）：覆盖 **492~509/512**
    专家/层。**校准流换 cnn_dailymail 契约池（2026-10-21 收口：不追）**：现有校准流的
    产物已过 QA 3/3 + 输出空间 ×0.64 判决；换池只在重做 GPTQ 时作为输入多样性选项，
    非待办。
    ③ 量化速度：`gptq_quant_fast`（整块量化 + 秩-128 Schur 补偿，**实测与逐列版输出误差
    逐位同值 ×0.587**，子块内传播零贡献）~25ms/矩阵 ⇒ 全量重打包 ~20-30 min（原逐列版 5.2h）。
    真实 Hessian 预验证（L00 抽样）：输出空间误差 ×0.45~0.59（砍半达成）。判决脚本
    `tools/arena-gen/gptq_check.py`（RTN vs GPTQ arena 同 Hessian 输出空间对比，bf16 原档为参考）。
   - ~~剩余：重打包 + 质量判据（下一步）~~（历史，已由下面 ✅ 两条完成）：
     `ARENA_FMT=Q3G128_F16S GPTQ_H=<hessians目录>
     quantize_expert_arena.py <bf16 arena> <out>`（~1h），然后 ① q3_arena_check 对拍、
     ② pad4983 金丝雀（RTN Q3 挂的那只）复活、③ QA 3/3。若 Q3+GPTQ 过判据 ⇒ 混档可全换
     Q3（体积再 −23%）或热集缩到 K=32；不通过则把 GPTQ 用在混档冷专家（Q3 段）上。
   - ✅ **全链路判决（2026-10-21 早）**：重打包 48/48 层完成（7138s；产物
     `$LINGXI_ASSETS/qwen4_exp.expert-arena.q3g128.gptq.bin` 44.41GiB，hessian 覆盖
     ~96% 专家、缺者回落 RTN）。**输出空间判决：n=28 中位 ×0.642 / 均值 ×0.655 / 最大
     ×0.808（gu 段 ×0.60~0.63，dn 段 ×0.68~0.81）——显著改善但未达砍半**；
     **QA 3/3 PASS**（smoke48/parity4167qa/needle12k 首 token 全对，shm 盘跑）。
     **pad4983 翻案**：该 prompt 首 token eos（248046）在 **mix64(Q4 热) 上同样复现**
     （10-21 校准捕获实测）⇒ 与量化档位无关、是 prompt/模型固有行为，**不是质量判据**
     （2026-10-19 "Q3 整表判死"的证据链因此存疑——当时只有 Q3 上的一次观察）。
     **工程血泪（重打包 OOM 两轮）**：hessians 逐层 12.8GB，CPU 版"当前层+预取层+512 份
     W 副本+span"峰值 35.5GB 两次全局 OOM（dmesg oom-kill 实证）⇒ 定案：**torch.load
     直接 map_location=GPU**（CPU 峰值 ~0，GPU 两层 26GB/32G 卡）+ 层尾 del+gc +
     `gptq_quant_fast_batched`（B=16，固定开销摊份，~145s/层盘 I/O 地板）。
   - **主路径执行记录（已全部完成）**：packer 加 `GPTQ_ONLY_FMT`（只对指定格式段补偿，热 Q4
      保持已验证基线）；**热集文件重建并固化** `docs/bench-logs/hist_top64.csv`（⚠️ 两连坑：
      a) hist_spec.csv 行序**不是** count 序，截前 64 行会选错；b) 按 count 重排取 64 会在
      **并列边界**（如层 27 有 7 个 count=7 争 4 席）与原始 mix64 热集差几条——**定案做法：
      从 mix64 产物 ref 的 format=1 直接反解热集**（产物即真相），count 仅作信息列。
      gate1（mix_arena_check）对错误热集当场 AssertionError，已验证能拦住。
      （原"热集 K=64→32 再缩"设想 2026-10-21 收口：默认档维持 RTN mix64 不翻、GPTQ 版为
      即插即用备选，无收益证据不动热集——关闭。）
   - ✅ **mix64.gptq 全判据通过（2026-10-21 中午）**：重打包 7141s（48/48，热集=从 mix64 产物
     ref 反解的 hist_top64.csv 3072 pair，冷 21504 全 GPTQ、缺 H 回落 RTN）。判据四连：
     ① 静态对拍：ref 表 24576 条（偏移/字节/格式）与 mix64 全同、热 3072 pair 字节级一致
     （MIX64_GPTQ_CHECK，注意 header 布局=8s+7i+4x+4q：ref 表在 q[9]、data 区在 q[8]，
     format 码 1=Q4G64、5=Q3G128）；② **QA 3/3 PASS**（needle12k gen=161 与 RTN 逐字同轨迹）；
     ③ **计时无差**：同条件 A/B 今日实测 RTN 32.93 vs GPTQ 33.05~33.48 ms/tok（噪声内；
     顺带翻案旧基线 29.95——同盘同档今日复测 32.9，旧数字系异条件测量，勿再引用）；
     ④ 冷段质量已由整表 Q3 判决背书（×0.64）。
     **默认档维持 RTN mix64 不换**（不静默翻生产档）；GPTQ 版为即插即用备选
     （`LINGXI_EXPERT_ARENA=$LINGXI_ASSETS/qwen4_exp.expert-arena.mix64.gptq.bin` + 进 shm）。
     配套：R5 护栏 gates.sh 上线并实测（gate1 混档对拍 / gate2 serve×paged 三场景+账本 /
     gate3 窗不足响亮报错，4/4 过；已挂 run.sh 尾部，GATES=0 跳过）。
     **6.2-3 GPTQ 补偿全案结案**：工具链（gptq.py 量化器/引擎探针/校准/重打包/GPTQ_ONLY_FMT）
     全保留，随时可对任意格式段重做补偿。

**粗活参考（2026-10-19 更新，含同权重实证）**：
- **⚠ 更正：本地 `$MODELS/nvfp4/Qwen3.8-Flash-Next-Uncensored-NVFP4/` 是项目自产量化**
  （用户确认；证据：conversion_environment 为 cu130/modelopt 0.46 自建环境，calibration_contract
  是自定义契约——cnn_dailymail pool 62139 行、分层种子公式、capture manifest sha）。
  分配策略"只有 512 路由专家量化（NVFP4 g16=4.5bpw）、其余全 bf16"是**我们自己的历史实践**，
  非独立第三方背书；其价值在：① 与 modelopt 公开配方/GSQ-RCO 模式三方互证；
  ② **校准口径可复用**（做 Q3 量化/GPTQ 补偿时用同一 contract，实验可比）。
- 独立背书仍然是：modelopt 公开 NVFP4 配方、GSQ-RCO（官方权重 3.5bpw 无损）、llama.cpp 共识。
- **Q3G128 = 3.125bpw**：blob 2.61→1.92MB（−26%）⇒ 常驻 88%（叠冷热分层）、miss 41%→12%。
   比上游激进一档——**无同权重实测背书**，判据全靠本地回归 3 基线 + 质量抽检；
   抽检掉分则退混档（hist 驱动热 Q4/冷 Q3，均 3.4bpw）。
 - ✅ **Q3G128 已开工（2026-10-19）**：
   ① 量化器收纳 `tools/arena-gen/quantize_expert_arena.py`（ARENA_FMT 参数化；Q3 产物
   44.41GiB 后台生成中 → `$LINGXI_ASSETS/qwen4_exp.expert-arena.q3g128.bin`）；
   ② 解析侧 `expert_arena.rs`：格式码 5 + matrix_layout Q3 分支（48B/组、无 high 平面）
   + open() 按首 ref 格式推导段几何，单测 4/4 过；
   ③ ✅ **内核已落地**：Q3 SIMT 解包核 ×3 形态（slots decode / slotsm prefill / batch
   dequant）、`Q4Consts/Q4Slots` 增 bits 字段并贯通 arena_source/moe/mtp_arena、调度分支
   （Q3 不做 TC 档）；arena 生成完毕（47.68GiB，正本 `$LINGXI_ASSETS/qwen4_exp.expert-arena.q3g128.bin`，
   shm 副本曾删为 Q4 腾位）、单测/对拍/编译全过。
   ④ ✅ **Q3 全量回归已跑（2026-10-19 晚）**：smoke48/parity4167 PASS，**pad4983 首 token
   eos FAIL**。根因不是内核/格式——`tools/golden/q3_arena_check.py` 对拍（上游解码器为金标准）
   实测 **Q3G128 专家权重 rel L2 误差 27~32%**（同口径 Q4G64 为 11~12%，比例 2.5× 与 4→3bit
   理论一致）⇒ 量化器正确但**整表 3.125bpw 误差压不住**，pad4983 这只长程金丝雀先翻。
   **结论：整表 Q3 档位不成立**，按 6.2-2 走 hist 热 Q4/冷 Q3 混档（格式天然支持逐槽格式码），
   或先上 6.2-3 GPTQ 补偿把 Q3 误差砍半再试。
- 第三方背书边界：GSQ-RCO IQ3_S 3.5bpw "无损" 是**官方权重**实测；我们用无审查微调权重，
  方向可信、幅度须自验。llama.cpp 共识 3.0 档可用、2.75 以下悬崖。

判据地基：先把散在 BACKLOG 的 A/B 数据固化成 `tests/regression/`（机器可读、可挂 CI）。

### 6.3 M3 重新设计：弹性 KV（用户拍板，替代固定路数拼槽）

不再划分 1×256K / 2×128K / 4×64K 的固定组合，做**统一动态池**：

- **统一 256K token 池**（双卡合计口径），准入按"请求长度 + 冷热程度"弹性分配，不设固定路数上限；
- **冷热分层**：热 KV 留 GPU；冷页**迁移到锁页 host 内存**（GPU 空闲带宽时回写，访问时取回——
  与我们已有的专家 arena / pin 机制同源）；更冷的按**过期/空闲时长回收**（TTL + LRU）；
- **跷跷板**：KV 字节 ↓ ⇒ 专家槽 ↑ ⇒ miss H2D ↓（6.1 已证这是 prefill 主导项）。
  ~~因此 KV 量化（旋转 + per-channel scale + V 高精度，**不裸 E4M3**——K8V4 的教训）在弹性池里
  有双重收益，值得在判据套件就位后重开~~ **（2026-10-21 收口：关闭**，理由见文末"跷跷板的
  两个落点"①——KV 当前不是瓶颈，无受益场景）；
- decode 侧同理：长会话的冷前缀 KV 出卡，腾出槽位给热点专家——**即 P2 paged，已落地**。
- ~~判据：任意到达模式下总吞吐 ≥ 固定 4 路拼槽方案~~（原判据属静态批并发（P3），已随 P3
  关闭；弹性 KV 本身的判据 = P0/P1/P2 各阶段验收，均已通过）。
- （用户留：甚至可以把所有kv都存内存里面，在超长上下文的时候，这样显存驻留更多专家，kv全部传到内存里面，加快读取时候的速度，后面短的上下文的时候，kv加回来到显卡。所有的东西都可以动态规划移动起来。）⇒ **即 P2 分页（host 锁页块池 + LRU 沉/取），已落地**。
  **6.3 全案终局（2026-10-21）：P0 ✅ / P1 ✅（v1.5 真并发）/ P2 ✅（Stage 1/2/3a/3b）/
  P3 ❌ 关闭。弹性 KV 愿景全部兑现或明确否决，无遗留。**

### 6.3.1 现状关键事实（2026-10-19 侦察，决定设计形状）

- **随上下文增长的 device 状态只有两块**：QSA KV（24KB/token：K/V 各 2kv头×256 维 bf16 ×12 层）
  + indexer（rk_dev 256B + pooled 64B /token /层）。256K 全量 ≈ 6.3GB + 0.96GB，双卡各半。
- **GDN 状态固定 116MB/序列**（S 矩阵 3.1MB×36 层 + 卷积历史），不随上下文增长；PLE 历史固定窗口。
- **KV 无间接寻址**：kernel = 基址 + 绝对行号 j + vis_base（attn_ops.cu:218-254），超容直接
  `assert!`（qsa.rs:451），**没有任何回收机制**。
- **但 QSA 本来就是稀疏注意力**：gather/sparse 路径已有块列表机制（sel_buf / keep 块），
  页表间接寻址有现成挂点；dense 变体需要补块表。
- **冷迁移半成品已存在**：`HostCheckpoint`（driver lib.rs:227-363）整份 D2H 7-9GB + 落盘 LXCK 格式 +
  恢复路径全通；锁页 host 内存与打包 DMA 在专家链路成熟。缺的是**页粒度**和**锁页内存档**（现在冷档直接写文件）。
- **预算无联动**：KV（`LINGXI_SERVE_MAX_CTX`）与专家槽（`LINGXI_ARENA_SLOTS` 等）是两个独立
  env，靠手工互调（256K⇒7200 槽、32K⇒9600 槽）。serve 启动顺序：GPU1 rope→arena→KV池；
  GPU0 rope→KV池→arena；权重经 DeviceLayerCache **懒加载**（arena 建局时权重尚未驻留）。
- **serve 严格单序列串行**（accept 循环即队列）；未使用的 `Slot` 骨架在 driver lib.rs:54-69。

### 6.3.2 分期实施（按价值/成本排序，每期独立判据）

**P0 预算自动分配（✅ 已落地 2026-10-19，回归验证中）**
- `main.rs::autotune_arena_slots`：catalog 角色排除法算每卡权重字节（`MoeRouted*` 走 arena
  不上卡；`text/layers/{i}/` 按层归卡；embedding/head 归 GPU0），减 KV(实际 cap)/GDN/预留/
  边距 ⇒ 默认槽数 = 余量 / blob。env 显式指定优先；`LINGXI_NO_VRAM_AUTOTUNE=1` 整体关闭。
- talk/serve 共 5 个建局点全部接入；`start.sh` 删掉了 `ARENA_SLOTS` 的写死默认（曾屏蔽 autotune）。
- 冒烟实测账（48-token）：GPU1 权重 3.4GiB ⇒ **10218 槽**；GPU0 权重 5.8（含 emb/head）
  ⇒ **8984 槽**。256K 口径估算 ≈ 7400/8600（对手工 7200/7600，更宽）。
- ⚠️ 附带发现：`LINGXI_ARENA_PIN_DMA=1`（§P13.9b，可写别名 + PORTABLE 注册）能让 miss 的
  H2D 从 8.6GB/s 变真 DMA（可重叠）——直接打在 6.1 的 76ms/层 miss 主导项上，**待一次专项验证**
     （结果见下一条，已关闭）。
- ❌ PIN_DMA 专项验证（2026-10-19，rows=887/7200 槽）：专家段 67.2ms/层（对照 84.6，但对照
  未含 CSR 修复，归因混合）；按 per-copy 账 **650MB ÷ 60ms ≈ 10.8GB/s ≈ 与 READ_ONLY 档无差**，
  注册成功（arena_pinned=1、拷贝走别名）。**结论：瓶颈不是拷贝引擎带宽，是 250 次/层 × 逐次
  提交的串行化开销**（≈110μs/次）。overlap（stream2）理论上只省 compute 的 ~7ms。
  ⇒ 此项关闭，不再投入；miss 问题的正解是**减少 miss 次数**（P0 自动槽 +830、P1/P2 分层）；
  逐 blob 拷贝合并大块化的预留项亦已于 §4 收口关闭（混档已正面缓解 miss 次数）。
- 判据进展：✅ 关闭（2026-10-21）——smoke 11751 过、全量回归此后每轮必跑（R5 挂 gates）；
  256K 峰值校验由 M1 实测（§M1 记录）覆盖。

**P1 多会话驻留 + 弹性准入（并发会话能力，非吞吐）**
 - ✅ **模型侧原语已落地（2026-10-19）**：`QsaState` 懒分配 + `ensure_cap` 几何增长
   （初始 `LINGXI_KV_INIT_TOKENS`=8192，倍增，全量拷贝换绑；`cap×dim` 不变式保持 ⇒
   检查点存取的 `kn.len()/cap` 推导不受影响；恢复路径先 ensure_cap 再 H2D）。
   上限 = 建局 cap（RoPE 口径），不可越界；`LINGXI_KV_NO_LAZY=1` 回全额预付做对照。
   逐位等价论证：换缓冲只改指针，所有 kernel 按调用时指针启动，专家段图用独立固定槽。
   ⚠️ 单序列下显存是零和（autotune 已按最坏情况预算 KV），懒分配的实际收益要等多会话
   共享预算 + P2 分层才兑现——它是地基，不是终点。
 - ✅ 记账核心 `KvBudget`（`crates/lingxi-driver/src/kv_budget.rs`）：统一池准入/挤占/增长/释放，
   最冷优先、先收缩冗余承诺（pledged>used）再整会话下沉、仅"单请求超池"拒绝；6/6 单测过。
 - ✅ **serve 多会话驻留 v1 落地（2026-10-19，端到端实测过）**：`serve.rs` 新增 `Sessions`
   （[serve.rs](../crates/lingxi-bench/src/serve/mod.rs)）——
   **任意数量会话驻留、同时只物化一个 Sequence**（显存零和的现实解）；请求体 `session_id`
   （回落 `user`，再回落 `default`）选会话；切换 = 当前会话 `save_host_checkpoint` 沉
   host RAM（`LINGXI_SERVE_SUNK_MB` 默认 32GB 上限，LRU 丢弃打响亮日志）+ 目标唤醒
   （H2D 灌回，秒级）或新建；`KvBudget` 管 pledge；PrefillCache 的 Dev 检查点随换手作废
   （host/disk 跨会话仍有效）；**CUDA Graph 按会话 epoch 重录**（[graph.rs](../crates/lingxi-model/src/graph.rs)
   `epoch_bump()`，图体烘死旧会话地址，必须隔离）。Sequence 由 `mk_seq` 工厂按需物化
   （main 不再预建）。实测：s3 记数→s4 插队→s3 唤醒答对（"42"）。
   顺手修真 bug：PLE 自动探测原来只检文件本体，量化器 prealloc 的**半写完** shm 表被
   当完整表读穿 EOF——现要求 sidecar（`.iq4nl`/`.scale`，即量化完成标记）同存。
   v1.5（多物化 + 真并发）与 P3（轮转/静态批）仍按序排期。
 - ✅ **serve v1.5 落地（2026-10-20 晚）——多物化 + 真并发**：
   - **架构**：连接线程（每连接一个，HTTP/SSE 收发）+ 引擎线程（唯一，轮转步进活跃生成，
     每轮每个生成一个 token）——单流时间片、无队头阻塞；引擎内部（Kernels/scratch/
     图固定槽/路由工作区）全 thread_local，串行步进下天然安全，**内核零改动**。
   - **多物化**：`LINGXI_SERVE_MATERIALIZED`（默认 2）个 Sequence 同驻；切换 = epoch_switch
     零搬迁；超配把最久未动会话沉 host（生成中会话受 protect 绝不当牺牲者）。
   - **per-session CUDA Graph epoch**（graph.rs）：CACHE/PHASE 键升 `(base, epoch)` 二元组
     （修掉旧 `base+epoch<<40` 与 GR rows 段理论撞键）；新建/唤醒=epoch_fresh、
     切换=epoch_switch（图集各自保留、交替不重录）、沉没=epoch_purge（CACHE 有界）。
   - **投机与并发不兼容**（草稿态绑会话）：`LINGXI_SERVE_SPEC=1` 整体走原串行路径
     （行为/性能不变，本项验证 5 轮/接受 14 枚）；投机关走并发路径。
   - **顺手修潜伏 bug**：spec+boundary 组合先喂边界前缀再 `prefill_with_hyper`（要求空序列）
     ⇒ assert 炸（投机 serve + 前缀缓存同开即触发，修：spec_on 时跳过边界喂）。
   - **实测**：双会话并发流式双双完成且内容连贯；多轮续聊前缀复用 33 tok（设备 KV 续接）；
     3 会话 LRU 沉/醒 + 图回收正确；shutdown 两路径干净退出（accept 非阻塞化 +
     引擎 recv_timeout，修掉"shutdown 后进程卡死不退"）；QA 3/3 两轮。
 - 判据：N 个会话并发驻留不 OOM；冷会话唤醒 ≤ 秒级；单会话性能不退化；回归全过；
   增长路径（INIT_TOKENS=1024）与全额预付对拍逐位一致。

**P2 页粒度部分驻留（"KV 全在内存"愿景的核心）**
- 效果：256K 活跃会话的 GPU KV 从 6GB（24KB/tok×12 层）压到热窗口（如 32K ⇒ 0.75GB），
  冷块全驻 pinned host（PCIe 8.6GB/s，命中取回 ~2.8ms/块、只在被 indexer 选中时发生）；
  autotune 的 KV 预算随之释放成专家槽。
- **关键侦察结论（2026-10-20，P2 开工前代码核实）**：
  ① QSA 的 raw kn/v **日常读路径只有 gather/sparse 的"选中块+尾"**（qsa.rs:889/898；
  indexer 打分只读 pooled 块，不碰 raw KV）⇒ **indexer 本身就是选块地基**，paging 只需
  在 gather 间接层上挂下沉/预取，dense/masked 回退路径（读全量行）在 paged 模式下
  直接禁用（断言 gate：paged ⇒ gather on 且 bf16 KV；K8V4 档无 gather ⇒ 互斥）。
  ② 写路径只有 append（qsa.rs:611 memcpy_dtod 到 [len..t_kv)）⇒ **尾块恒驻策略**
  （最近 W 块不沉）即可覆盖全部写入，无需写时取块。
  ③ decode 每 token 本来就 host 同步（argmax→detok）⇒ sel 名单 device→host 小 D2H
  几乎零增量成本，下沉/预取的记账不需要破坏异步流水。
  ④ checkpoint save/restore 目前整缓冲 wholesale 拷贝 ⇒ paged 下要按逻辑块走表
  （resident 直拷 / sunk 从 host 池读）——这是 Stage 2 里最易漏的口子。
- **分期**：
   - **Stage 1（结构层，零行为变化）✅ 落地（2026-10-20 晚）**：`QsaState.blk_tab`
     （device i32，按 max_cap 一次建足、恒等映射）+ gather 核（bf16/gather4 两形态）
     K/V 读各加一级表间接（`kv_phys_row`：pos→slot，间接粒度 = indexer 块 ctok）；
     另加 `LINGXI_QSA_PAGED=1` 兼容断言（gather 前提不满足即炸，防 Stage 2 误用）。
     **验收：QA 3/3 逐位不变（首 token 11751/271/271 与 Stage 1 前一致）；
     decode 29.98~30.77 ms/tok ≈ 基线 29.95（<2.7% 噪声内）**。
     剩余挂接点（Stage 2 做）：sparse 核未加表（paged 模式下由断言禁入）；
     checkpoint 仍整缓冲拷贝（paged 下必须按块走表）。
  - **Stage 2（驻留分层）✅ 落地（2026-10-20 深夜）**：`LINGXI_QSA_PAGED=1`（默认关）——
     host 锁页块池（懒倍增 pinned，2×ctok×kv_dim/槽）+ 惰性 LRU 下沉/预取（选中冷块
     gather 前预取回填）+ checkpoint 按块走表（`paged_kv_to_host/from_host`，restore
     整体进 host 池零 GPU 拷贝）+ rewind 释放块。诊断套件：`LINGXI_QSA_PAGED_NAN/CHECK/SINK_NOP`。
     **调试链（3 个真 bug 定性）**：
     ① 驱逐切片误用逻辑块号（重映射后越界 panic）→ 按槽号切；
     ② host 池倍增换缓冲无同步，旧 pinned 被 Drop 时在途 DMA 踩已回收内存 → 换池前同步；
     ③ **选中集覆盖窗口时预取驱逐互踩**（512 选中 = 512 窗口 → touch 全等 → 驱逐唯一候选=
     本步选中块 → 双块同槽静默 corruption）→ admit 预取路径绝不驱逐 touch==本步 的块，
     无处可驱逐 ⇒ **响亮报错**（拒绝静默出错）。
     另：分页与 CUDA Graph 捕获互斥（attention_step 体内有 host 同步/D2H）→ paged 模式
     run_cached 一概直跑（decode +30% = 图收益损失，记入成本）。
     **验收**：非分页 QA 3/3 逐位不变；paged 默认窗（32K）三场景首 token 与金标一致
     （11751/271/271）、decode 30.3ms ≈ 基线；窗 ≤ 选中工作集时响亮报错（实测 win=2048/4096/8192
     在 14.5K 上下文 prefill 均触发报错而非脏值）。
   - **关键架构发现（决定 Stage 3 形态）**：**prefill 的选中并集 ≈ 上下文已长部分**（512 行 ×
      每行 topk=512 块 → 并集饱和）⇒ "整步选中全驻留"要求窗口 ≥ 上下文——真正的驱逐只在
      decode（并集 = topk+尾 ≈ 600 块）可行。**Stage 3 = prefill 分行批 admit**（gather 按
      行批切多次发射，每批 admit 自己的并集，窗口 ≥ 批并集 ~600-800 块即可）+ autotune 的
      KV pledge 从全额改热窗口口径 + 256K 会话判据。
    - **Stage 3a（行批 admit）✅ 落地（2026-10-20 深夜）**：`kv_admit` 加行段参数；
      paged 且 t>1 时 gather 按 `LINGXI_QSA_ADMIT_BATCH`（默认 64）行批 "admit→发射"
      成对循环；gather 包装改视图参数（批切片直传）。**实测判决**：topk=512 是**固定块数**
     （非比例）⇒ 行批并集 = 1−(1−512/nblocks)^k——nblocks 大时任意 ≥16 行的批都饱和
     （批 admit 不是万能药，这是 indexer 预算口径的固有性质）。
     **并集饱和/K1 的终局（2026-10-21 收口：K1 不实施）**：**paged 定位 = decode/长会话
     驻留工具**；prefill 并集饱和场景的两条出路都被堵死——① prefill 走非 paged 全额 KV：
     ≤47K 可行，但 250K 经 M1 实测 OOM（§M1 负结论），非 paged 上限就是显存；② paged
     prefill：长上下文下批并集饱和 ⇒ 窗口 ≥ 上下文，退化为全额驻留，白分页。
     paged decode 并集 ~600 块 ≪ 窗口，天然不饱和；行批 admit + 窗不足响亮报错已覆盖
     残余可用性。**"预算随 log 收缩/比例化 topk"（原 Stage 3b 候选 = K1）随之关闭**——
     改选块预算会动 indexer 语义与质量判据，而它被规避后无受益场景；重开条件：
     换 >40GB 显存卡后 256K prefill 立项，或出现"质量可换上下文"的产品需求。
      **churn 正确性验收：win=12288（3072 块 ≪ 3636 块上下文）+ batch=4，needle12k
      全程驱逐下首 token 271、needle 答案 0.52、160 tok 全出 ✓**。
   - **Stage 3b ✅ 落地（2026-10-20）**：① autotune pledge 联动——paged 时 KV 预算按
     `min(kv_cap, LINGXI_QSA_WIN_TOKENS||32768)` 窗口口径算（main.rs `autotune_arena_slots`），
     省下的全额 KV 直接变成专家槽。实测 serve 256K：KV 3.48→0.43 GiB、槽 10275→11976
     （+1701 ≈ +3.1 GiB，数学自洽）；win=8192 时 KV 0.11 GiB。
     ② 冷块选中率账本：`PAGED_SINKS/PAGED_PREFETCHES` 全局计数 + `qsa::paged_stats()`，
     serve perf.jsonl 每请求输出 `paged_sinks/paged_prefetches`（实测 win=8192 + 14.5K 上下文
     churn 下 70 万级计数、随请求累计）。③ 预算口径研究结论沿用 Stage 3a（topk 固定 512 块
     ⇒ 并集饱和；**该候选已随 K1 一并关闭**，见 Stage 3a 终局）。
     **顺手揪出并修复 2 个真 bug（serve×paged 集成的潜伏雷）**：
     ④ dev 检查点（智能前缀缓存）与 paged 不兼容——`reset` 的 `rewind(0)` 已把块表/槽/host 坑
     全部归还，而 dev 检查点只存 GDN/PLE（QSA 靠"KV 还在设备上"隐含假设）⇒ 缓存命中恢复后
     输出退化为乱码（实测第二次请求起）。修：serve 全部 5 个 save 点经 `save_ckpt_auto`
     统一分派——paged 模式强制 `save_host_checkpoint`→Disk 档（`paged_kv_to_host` 收拢全量）；
     driver `restore_checkpoint` 对 paged 响亮拒绝（防再踩）。
     ⑤ Disk 恢复后全部块在 host 池（tab=-1），尾部**半满块**追加写撞 "未物化" 报错——
     `block_prefetch` 提取为公共方法（admit 第 2 遍 + 写前物化共用），`kv_materialize_write`
     遇 -1 先预取回填再追加。
     **验收**：serve paged 三场景（冷启动 / Disk 缓存全命中 / 改末句边界命中）needle 答案均
     正确（0.52/0.52/0.5）；paged 与非 paged QA 各 3/3 全过。
     ~~遗留（记入后续）~~**两条遗留的终局（2026-10-21 收口：均关闭）**：
     ⑴ paged 前缀缓存条目是全量 dense KV（32K ≈0.5 GB/份 可用；256K ≈14 GB 超 24 GB 上限
     会被丢弃）→ **不做 paged 格式检查点**：paged 会话靠 win 驻留服务，256K 重 prefill
     时重算可接受（且 prefill 走非 paged 的定案使超大 Disk 份不再是热路径）；
     ⑵ paged 模式 decode 直跑无图（−30% 图收益）→ pledge 方案已定稿（Stage 3b①），
     无后续动作，作为 paged 档的已知成本记录在案。
  - ~~Stage 3（预算联动 + 验收）~~ 已并入上面的 Stage 3 定义（批 admit + pledge 联动 + 判据）。
- 旧稿（块大小/24MB 参考值）已被上述侦察取代：块尺寸对齐 indexer 块（ctok=2^cshift，
  gather 的 sel 单位 1:1），不是固定的 1024。

**P3 batched decode kernels —— ❌ 关闭（2026-10-21 收口）**
- 原案：kernel 加 batch 维、CUDA Graph 按 batch 键重录、跨序列静态批 decode（聚合 2.5~3.5×）。
  **关闭理由**：研究级改动（attn/gdn/moe 全核加 batch 维 + 调度）对当前个人 serve 负载
  收益为零（吞吐非瓶颈，时间片版已覆盖多会话同时服务）。重开条件：真实多用户吞吐需求。
  见 §1.B。

**跷跷板的两个落点**（贯穿 P0-P2）：② P0 的自动账已落地（"砍 KV 的收益"自动变成专家槽）。
① **KV 量化（旋转+per-channel，不裸 E4M3）——关闭（2026-10-21 收口）**：256K 全额 KV 双卡
仅 6.3GB（M1 实测显存背书，不是瓶颈），当前没有"KV 挤占专家槽"的现实场景；paged 窗口径
（Stage 3b①）已把 paged 档 KV 压到 0.43 GiB。重开条件：并发会话把 KV 预算真正挤占。

---

### 架构债务与重构提案（2026-10-21 用户拍板："牵一发动全身，得梳理/重构"）

**A. 实证：近两周 bug → 耦合根因**（不是"运气差"，是结构性债务的反复显形）

| # | bug（日期） | 根因（耦合点） |
|---|---|---|
| 1 | 混档 dn 段偏移错 ×2：回退臂（10-19）+ 统一臂（10-20），同一类雷踩两次 | arena 段几何"按首 ref 格式推导"散布在多个消费点，每处都隐含"单格式"假设 |
| 2 | serve×paged：dev 前缀缓存恢复后输出乱码 + 半满块"未物化"（10-20） | 检查点双路径（dev/host）语义分叉；`rewind` 隐含"append-only KV 永驻设备"，分页下不成立且无静态防护 |
| 3 | 选中集覆盖窗口时预取驱逐互踩 → 静默 corruption（10-20） | admit / slot_alloc / materialize 三处驱逐保护靠注释约定，无单一责任点 |
| 4 | 分页 × CUDA Graph 捕获互斥 → run_cached 全局 if 直跑（10-20） | 能力互斥（paged 无图）靠运行时绕路，引擎构建期不做能力声明 |
| 5 | accum 层序交错 flush 残缺、热集 csv 选错、torch.save 截断、重打包 OOM×2、pkill 自杀、cp 旧二进制（10-21） | 工程链全是手工命令，每步无护栏/校验/幂等 |
| 6 | P2 期间 5 个分页 bug 全过了旧 QA 3/3 才暴露 | 回归套件不覆盖 serve×paged / 长上下文 / 混档全格式批 |

**B. 重构候选（按 ROI 排序，先护栏后动刀）**
- **R5 回归护栏扩展（最先做，成本低收益直接）**：QA 套件加 ① serve×paged 冒烟（三场景 needle 已在手，脚本化成 case）② 混档全 Q4 批次 case（就是 10-20 那只雷的形态）③ 分页窗口 < 工作集响亮报错 case。目标：A 表里的 bug 形态各有 case 守门。
- **R1 几何单源化 ✅ 落地（2026-10-21 中午，验收 QA 3/3 + 纯 Q4 档首 token 对拍 + 全 gates 4/4）**：
  审计结论先行——10-20 修复后，混档下所有热路径**已**走 `routed()` 逐 ref 自己的格式几何，
  `gu_bytes/down_bytes` 的消费点（统一臂/A2 路由/重钉/补槽）全部有 `!mixed` 闸门，雷已封死。
  落地的是纪律固化：① 统一臂 + A2 路由两处 `dn_seg: arena.gu_bytes` 改为
  `gu_lay.payload_bytes`（本批/主格式 matrix_layout 自算，语义今日等价、不再依赖
  "缺省格式==消费格式"巧合）；② `gu_bytes/down_bytes` 字段 doc 标明"缺省格式几何，拿到 blob
  格式者禁止读"；③ 臂收敛不做——纯格式档的统一批量臂是性能路径（2 launch/层 vs 逐专家），
  保留，正确性改由几何来源保证而非闸门保证。
- **R2 检查点统一 ✅ 落地（同日，同一批验收）**：审计结论——统一分派 `save_ckpt_auto`（5 个
  保存点）+ restore 侧响亮拒绝 10-20 已落。补的是两头栅栏的最后一头：driver
  `save_checkpoint` 创建侧 ensure（paged 即拒，不再只靠 serve 分派自觉）+ serve 启动期
  paged 不变量响亮横幅（win 口径）。rewind 类型化不做（收益/风险不成比例，rewind 释放即作废
  的语义现由 ③ 的注释 + restore 栅栏覆盖）。
- **R3 qsa.rs 拆分 ✅ 落地（2026-10-21 深夜，用户拍板当日执行）**：1639 行单文件 →
  `src/qsa/` 五模块（`mod.rs` 只声明+再导出，**对外 `qsa::` 路径不变、调用方零改动**）：
  `core_`（类型+开关+`QsaState` 生命周期 ~450 行）/ `paging`（host 池/驱逐/物化/按块检查点
  +沉取账本，**独占 host 池句柄** ~370 行）/ `admit`（行批选中准入，**独占选中缓冲** ~90 行）/
  `step`（`attention_step` 选块/gather/诊断 ~550 行）/ `oracle`（host 参考实现 ~230 行）。
  所有权纪律落进类型边界：字段全 `pub(crate)`、跨模块方法 `pub(crate)`——"谁拥有什么"
  从人肉追变成编译器检查。验收：编译零错 + QA 判据（首 token 11751/Paris 逐位一致）
  + **全 gates 4/4**。paged×graph 互斥**维持** run_cached 全局 if——能力声明上移在 Rust
   无便宜表达（无 effect 系统），而该 if 已是唯一事实点；记录为"不做的决定"而非遗留。
- **R4 serve.rs 拆分 ✅ 落地（2026-10-21 深夜，R3 同晚交付）**：3152 行单文件 →
  `src/serve/` 四段（`mod.rs` 只声明+再导出，**对外 `serve::run`/`serve::ServeOpts` 不变**）：
  `cache`（前缀缓存 + CkptStore + 保存分派 ~510 行）/ `sessions`（会话表 ~190 行）/
  `proto`（chat 协议/build_prompt/SSE/工具块/StopFilter ~860 行）/ `engine`
  （run/run_concurrent/engine_loop/Gen 步进/conn_loop ~1620 行）。段间共享一律
  `pub(crate)`。验收：编译零错（warnings 50→5，cargo fix）+ **全 gates 4/4**
  （gate2 serve×paged 三场景背书拆分后的 serve 路径行为不变）。
- **R6 工程链固化 ✅ 落地（2026-10-21）**：`tools/arena-gen/gptq_pipeline.sh`（calib→
  校验→pack→判决→md5 落账，每步护栏/失败即停/断点续跑）+ 根 `Makefile`（build=构建+cp
  一条链、regress/gates/gptq-pipeline 目标）。手工长命令由此消灭。
- 顺序（2026-10-21 晚更新）：R5/R1/R2/R6 已交付；显存回归修复后 **R3 → R4 正式开工**
   （用户拍板：架构调整即刻排期，见上两条）。

**C. DeepSeek DeepInfer 代码级调研完成（2026-10-21，repo 在 $HOME/deepinfer + 本表为准）**
已 clone 通读（git 走 gh-proxy.com 镜像；DeepSelect 单独直拉）。最重要的两个认知：
- **"SWA Bounded Replay" 不是投机解码**，是 YoCo 截断前缀缓存命中时"只重放最近 ~128 token
  重建 SWA KV"的有界重算（QA 文档 183-191 行）——与 lingxi rewind-only 检查点**同族思想**
  （恢复成本 bound 在有界窗口），但 DeepInfer 把 rewind 深度压到与上下文无关的常数。
- **DeepInfer 完全无 LRU/无 CPU offload/无"沉醒"**：淘汰只有抢占（保短弃长、整段重算或终止），
  缓存分层推给部署层 3FS。lingxi 的"多物化 LRU 沉醒"是它的明确空白区，也是我们的差异化。

内核/机制级借鉴表（编号延续，⭐=排期优先）：

| # | 机制（出处） | 借鉴点与收益 | 改动面 |
|---|---|---|---|
| K1⭐ | **两级块预算**（source 层选 candidate 块集、下游层组共享，index_attn.py:204-243） | ~~正面回应 P2 遗留的"行批并集饱和"~~ **❌ 关闭（2026-10-21）**：架构定案"paged=decode 驻留工具、prefill 走非 paged"使并集饱和无受益场景（见 Stage 3a 终局）；改选块预算会动 indexer 语义与质量判据，不实施 | 已关闭 |
| K2⭐ | **graph_timer 图内探针**（捕获时 RecordExternal，图实例化后 cudaGraphExecEventRecordNodeSetEvent 换事件，ping-pong 双缓冲异步读出；graph_timer/） | CUDA graph 内分段计时零开销且可读回喂调度；模型无关 | 小（~200 行 C+++1 Python 类） |
| K2✅ | **已落地（2026-10-21 下午，Rust 版 `lingxi-model/src/graph_timer.rs` ~450 行）**：机制与 DeepInfer 一致——捕获内 `cuEventRecordWithFlags(EXTERNAL)` 成节点、实例化后枚举 event-record 节点（**按烘焙事件句柄匹配名字**，图节点枚举序≠注册序）、每次重放前 `cuGraphExecEventRecordNodeSetEvent` 换 ping-pong 事件、上一代异步读回；挂进 `graph::run_cached`（body 首尾自动 begin/end 探针，paged/直跑路径退化为普通流事件）；开关 `LINGXI_GRAPH_TIMER=1`（登记进 probes.rs），驱动错误自动自关绝不炸 decode；epoch_purge 同步回收。验收：开关开/关输出逐位一致（11751 " Paris."），报告 96 键 × 22 遍，begin→end 均值 0.043 ms/单元。手动 `graph_timer::probe(k,"name")` 可在 body 内插点拿分段 | 已完成 |
| K3⭐ | **乐观推进+按接受数回填账本**（advance_optimistically/commit_processed_tokens，sequence.hpp:216-228） | ~~投机解码 KV 账本零回滚成本~~ **✅ 落地（2026-10-21）**：对账发现 lingxi 的 QSA 侧本来即 DeepInfer 形态（append-only，rewind 只改计数）；补的是 GDN 侧**全接受零回滚快速路径**——`spec_commit` 里 `accept == t` 时跳过 36 层的快照恢复+重放（批后态==串行态，lingxi 自己的 LINGXI_SPEC_CHECK 硬判据已认证）；部分接受仍恢复+重放（循环状态固有成本）。验收：spec A/B 顺序==投机逐位一致 | 已完成 |
 | K4 | 注意力成本闭式公式（flops_estimator.hpp:220-257） | ❌ 不实施（10-21 收口）：时间片调度无准入竞价，O(1) 成本预估无消费者 | 已关闭 |
| K5 | ZOS 延迟一步 post-process + deferred release（worker.py:143-188） | ❌ 不实施：lingxi decode 每 token 本就 host 同步（argmax→detok），无 ZOS 形态的 CPU bubble | 已关闭 |
| K6 | stored/dropped 分区物化（StateBlock，cache_engine.py:221-306） | ❌ 不实施：rewind-only 检查点（只存 GDN/PLE 循环状态 ~115MB/份）已是同族思想且更省 | 已关闭 |
| K7 | confidence 驱动变长投机预算（MTPVarlen） | ❌ 不实施：MTP 接受率已 93~99%，变长预算机械化收益小、改动面大 | 已关闭 |
| K8 | 图 batch 分档 2 的幂+单 memory pool 反向捕获（cuda_graph_runner.py:41-148） | ❌ 不实施：静态批（P3）已关闭，batch 分档无对象；per-session epoch 重录已摊销（每核一遍） | 已关闭 |
| K9 | 量化布局即协议：KVCacheFormat 描述符+反推校验（cache_common.py:24-187） | ❌ 不实施：arena header 已有 magic+逐项 16B ref 自检，gate1 混档对拍守门，重复建设 | 已关闭 |
| K10 | ue8m0 scale 取 2 幂指数编码（quant.py:5-10） | ❌ 不实施：Q4G64 scale 已是 fp16 乘法路径，改 2 幂编码无收益 | 已关闭 |
| K11 | EPLB 逻辑→物理映射+热点专家多副本（eplb/policy.py:73-110） | ❌ 不实施：双卡 layer-split 已固定且够用；EP 路线未立项 | 已关闭 |
| K12 | CUDA IPC handle 跨进程共享显存（cuda_utils.cpp:34-89） | ❌ 不实施：单进程引擎（线程内 thread_local），无跨进程共享需求 | 已关闭 |

DeepSelect（sparse-attn top-k 库）专项结论：**算法=单遍流式阈值过滤+shared memory 周期
radix-select 压缩，比 torch.topk 快 2-20×；但严格 row-independent、固定 topk/行——"行批并
集饱和"它不管（已确认无任何跨行机制），这个预算问题必须由 lingxi 自研（K1 方向）**。集成：
Python/torch 强耦合（libtorch），Rust 侧宜抄算法思想（common_parts.cuh 的 PTX 技巧自包含）。
不排期替换我们现有选块核（收益仅是"全扫前提下选得更快"，非主要矛盾）。
**行动终局（2026-10-21 收口）**：K1 关闭（Stage 3a 终局）/ K2 ✅ 落地 / K3 ✅ 落地 /
K4-K12 逐条关闭（理由见表）。DeepInfer 借鉴全部处置完毕，无一遗留。

---

## 7. 2026-10-21 全局收口：全部 plan 对账，无一遗留

> 用户指示："所有 plan 对比一下，文章中不要留着没有做的事情了。" 本节是总账：
> 每一项要么**做完**（带验收），要么**明确关闭**（带实测理由与重开条件）。

**当日交付（有验收）**：

| 项 | 交付物 | 验收 |
|---|---|---|
| K2 graph_timer | `lingxi-model/src/graph_timer.rs`（~450 行 Rust 版，机制对齐 DeepInfer） | 开关开/关输出逐位一致；96 键 × 22 遍计时报告 |
| K3 投机账本 | `spec_commit` 全接受零回滚快速路径（GDN 侧 36 层快照恢复+重放跳过） | 编译零错 + 启动路径通；**运行 A/B ⏸ 未跑**——MTP 产物 10-21 深夜被 tmp 清理后已硬链接抢救回 `lingxi-assets/qwen4_exp.mtp.fixed.ninfer`，复跑= `LINGXI_SERVE_SPEC=1` A/B |
| M1 256K 实测 | §M1 记录（**负结论**：250K OOM×5 配置，峰值与槽数无关；≤47K 可跑） | 容量边界 + 三条出路 |
| R6 工程链 | `tools/arena-gen/gptq_pipeline.sh` + 根 `Makefile` | bash -n/make -n 过；每步护栏/断点续跑/md5 落账 |
| #11 开源整理 | LICENSE(Apache-2.0) + .gitignore + README License 段，PR #1 | Gitea PR #1（chore/open-source-prep → main）；10-21 晚收口：干净克隆 `cargo build`/`cargo test` 全绿、start.sh 起 serve 实测通、QA 3/3 + gates 4/4、probes 守卫补登记 23 开关、LD_LIBRARY_PATH 收编进 start.sh、README 部署章节 |
| 资产正本迁移 | 全部持久资产迁入 `$LINGXI_ASSETS/`（tmp 是自动清理区，10-21 曾清掉全部 MTP 产物） | 20+ 项落位（含硬链接抢救 6 个 .niner）；全仓引用（代码默认值/start.sh/gates/工具链/文档）已改指新正本，ASSETS.md 加醒目警告 |

**当日关闭（有实测理由，重开条件已写明）**：M2（GEMV 仅 6ms/层，前提消失）/
M4（GPTQ 计时旁证收益路径不存在）/M5（研究级，prefill 非瓶颈）/P3 静态批（吞吐非瓶颈，
重开=真实多用户负载）/K1（paged 定案为 decode 驻留工具，并集饱和无受益场景）/
K4-K12（逐条理由见 C 表）/KV 量化（KV 非瓶颈）/trunk 转换器（依赖已写清 FORMAT.md）/
校准流换池（现有校准已 QA 背书）/热集 K→32（无收益证据）。
**当日重开**：R3/R4 架构拆分（用户拍板："完善功能越来越难、走一步踩雷"= 结构病，
见 B 表 R3/R4 两条）。

**PLAN_NEXT 自此无"待做/下一步/TODO/进行中"事项。** 再开新方向时新开章节。

### §M1 记录：256K 实测全案（2026-10-21，负结论结案 + 无回归定性）
  
 - 口径：parity4167 × 60（250,080 tok）/ × 11（45,837 tok），双卡 layer-split(24)，
   chunk=512，MAX_CTX=262144，Q4 与混档两个 arena 都测过，新旧代码对照（worktree 二分）。
 - **最终定性：无单一代码回归，是三层结构性紧绷**（同代码同配置可过可爆 = 碎片抽签）：
   1. **启动期大槽块抽签（随机性来源）**：常驻账 = 权重 5.8/3.4 + 全额 KV+indexer ~4.2 +
      PLE 缓存 1.0 + eager 槽预留（7200×2.49MiB≈17.9GB 单块）≈ 29-30GB——`SlotCache::new`
      的单块大 alloc 在碎片池里成败随缘：8bd91b9 的 250K 抽中（跑进 prefill 112min 无 OOM，
      峰值 30186），同代码 47K 两次没抽中（**`ArenaExpertSource::new` 启动即 OOM**）。
      新旧代码同现象 ⇒ 不是回归。
   2. **混档 250K 死于 prefill 中段 ~150K 上下文**（5/5 配置全灭，峰值 32478-32494）：
      栈在 `qsa::attention_step` 的 **`big_prefill` 选块 scores 缓冲 f32[t×nblocks(×heads)]**
      ——唯一随上下文线性涨的 GPU 缓冲；基数选择路径（默认开）本应是零缓冲，
      大 nblocks 回退旧路径是踩穿天花板的直接原因。
   3. **10-18 "250.6K 3436.9s 无 OOM"记录不可复现**：机器 10-19 切换过 + 该数字自身标注
      "池化修复前偏乐观/需重测"。"以前能跑"是记忆偏差，非代码回归（用户质疑驱动此排查，
      结论：账算清了，没有凶手提交）。
 - **容量边界（实测）**：≤47K 端到端舒适（混档 7200 槽 786s / Q4 958s）；~150K 是混档
   prefill 天花板；256K 端到端在本机**无可靠配置**。
 - **修复方向（未来若重开 256K）**：① 大 nblocks 选块改基数/分块（砍 scores 缓冲，
   性价比最高）；② 槽预留改惰性/分块大 alloc（去启动抽签）；③ 上 >40GB 卡。
   **本次结案维持：256K 全额 KV 档在 2×V100 32G 不可用，up_long.sh 的 256K 配方标注此边界。**

---

路径速查（本机 R720XD）：**运行时 `export LD_LIBRARY_PATH=$CUDA_LIBS/nvidia/cublas/lib:
$CUDA_LIBS/nvidia/cuda_nvrtc/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH`**
（cudarc 动态加载 libcublas/libnvrtc/libcudart；2026-10-20 重新摸到，新 shell 必配，否则
panic "Unable to dynamically load the cublas shared library"；cargo 在 `~/.cargo/bin`）。
模型产物 `$MODELS/qwen4_exp-artifact/qwen4_exp.ninfer`（= `$MODELS`，
bind mount），MTP `qwen4_exp.mtp2.niner`，`hf2lingxi.py`/`extract_ple_fp8.py` 在 `tools/convert/`。
**全部资产（含易失 shm 的持久正本/再生命令）见 [ASSETS.md](ASSETS.md)**。

