# strata MTP 与专家 profile 机制 → lingxi 对照方案

> 事实源：strata 源码（`expert_cache.hpp/.cpp`、`expert_source.hpp`、`mtp.hpp/.cpp`、
> `spec/controller`、`suffix_drafter.hpp/.cpp`、`generate.cpp`、`make_profile.py`），
> 以及 lingxi 本仓库代码与 Qwen3.8-Flash-Next 产物实测尺寸。
> 文中「strata」指对方 C++ 引擎；所有移植建议都给出了 lingxi 的具体落点，不照搬参数。

---

## 0. 双方事实卡

| 维度 | strata（参照系） | lingxi（本项目，现状） |
|---|---|---|
| 语言/运行时 | C++/CUDA，多线程 lookahead、多流 | Rust/cudarc，单流为主，无后台预取线程 |
| 专家权重精度 | Q8_0（投影另有取舍），约 1.38MB/专家 | **BF16，9.83MB/专家**（gate_up 1280×2560 + down 2560×640）；另有 Q4G64 产物可选 |
| 每 token 专家暴露字节 | 663.6MB（480 次专家访问，实测 PCIe ~40GB/s → 16.2ms，占 53ms token 的大头） | BF16 理论 480×9.83MB = **4.72GB/token → ~118ms/token**；Q4G64 约 1.2GB → ~30ms |
| 显存策略 | 4105 槽静态/自适应 arena + CPU 专家池（覆盖 96% 暴露）+ PCIe 直读 + VMM 分段 | **无专家级缓存**：`build_layer` 每次把整层 512 个专家（5.03GB）从盘上全读一遍 |
| MoE 计算 | GPU grouped kernel（常驻专家）+ CPU 池（漏出专家），按 router 行序合并 | GPU 上**逐专家** cuBLAS GEMM（[moe.rs](../crates/lingxi-model/src/moe.rs)），无 grouped kernel、无 CPU 计算 |
| 路由决策位置 | GPU + host bf16 router 做下一层预测 | router 在 GPU 算 logits，**softmax/top-k 已在 host 做**（天然适合采集与 hit/miss 拆分） |
| 投机解码 | 内置 MTP 层（dense 注意力）+ trigram suffix drafter + Controller 在线选路 | `n_mtp=1` 已在维度表，mtp2.ninfer 含全套 27 个张量；driver 只有贪心 decode |
| GPU | （对方环境） | 单张 V100 32GB，PCIe x8，sm_70（无 fp4 原生指令）；GPU0 常被 ninfer 占用 |
| 磁盘 | SSD 假设 | **机械盘**，48 层全读一遍约 45 分钟（与 ninfer 抢 I/O 时更慢） |

**一句话现状**：lingxi 现在 prefill 是「每层读 5GB 专家、用完即弃」，prefill 一次尚可忍，
**decode 每生成一个 token 都要重读 241GB，不做专家缓存根本无法服务**。
MTP 则相反：权重齐备、算法清晰，是纯增量功能。

---

## 1. 专家 profile 与缓存：strata 机制拆解

### 1.1 profile 二进制（出厂先验）

- 文件布局：magic `STRP` + 5 个 uint32（version=1、n_layers=48、n_expert=512、slots、n_ranked），
  随后 n_ranked 个 `(uint16 layer, uint16 expert)` 对（按路由频率降序），
  最后是 48×512 的 int32 名次表（-1 表示未排名）。出厂 profile 把 24576 个（层,专家）全排名。
- 生产方式：`--dump-routing` 在真实对话上吐 routing trace，离线 `make_profile.py` 聚合生成。
- 关键测量结论（对方实测，作为机制设计依据，数值不可照搬）：
  - **十折留一命中率**：4105 槽按专家热度 h_expert=0.6447；
  - 若按「每层 10 个专家整组常驻」h_layer 只有 0.0456——
    **绝对不能按层整组预取，必须按 router 的行级选择做 hit/miss 拆分**；
  - 每 token 专家字节 663.6MB，PCIe ~40GB/s → 16.2ms，是 token 时延最大头；
  - CPU 专家池可覆盖 96% 的暴露字节（他们有 CPU GEMM 基建）。

### 1.2 静态缓存层 ExpertCache

- 显存一块 arena，槽位等大；`residency[48*512] -> slot / -1`，**无驱逐**。
- 换槽协议：`fill_slot` 拷入后必须 `verify_slot` 与源做**逐字节比对**，不一致不许上线。
- 失败教训（写在注释里的）：早期用「全局到达顺序计数器」，槽位被前 26 层的 0 号专家
  占满，命中率仅 2.97% → 必须有 **per-layer admission（每层独立名额/准入）**。
- 其余配套：PCIe 直读 host 上 pin 住的 blob（pcie_frac、三种 pcie_mode）；
  VMM 分段 arena 支持弹性显存；双卡有 DISJOINT 去重。

### 1.3 自适应层（在线学，纠出厂先验之偏）

- 每 `adapt_every=4` 轮做一次换入换出：
  - 每层候选 = 衰减后 usage ≥ 2.0 且未常驻的专家；
  - 受害者 = 同层 usage 最低的常驻专家；
  - 仅当 `候选.usage ≥ 受害者.usage + 1.5` 才允许换（迟滞，防抖动）；
  - 全局按 gain 排序，本轮最多换 `adapt_swaps=96` 个；轮后所有 usage *= 0.7（衰减）。
- 拷贝走独立 adapt_stream，夹在两个验证窗口之间；event 未落地则下轮跳过，
  落地并校验后才翻 residency 表；被换出专家先拷回 host 紧凑镜像（exchange buffer），
  RAM 层持 GPU 缓存的补集。
- RouterLookahead 线程：用 l+1 层的 router（host bf16）对第 l 层输入预测路由，提前预热页面。
- `--expert-profile-save`：把在线学到的（resident→heat→prior→index）回写 profile。

### 1.4 hit/miss 拆分（让缓存真正省时间的核心）

验证窗口里一次要访问 480 个专家位。按 router 输出把每个 token 的 10 个路由位拆成：

- **hit（常驻）行**：GPU grouped kernel 一次算完，带 dst 路由位；
- **miss 行**：交给 CPU 池（strata），并把 GPU 上这些行置零，避免重复累加；
- `moe_combine` 最后按 **router 原始顺序**加权拼回。

只缓存、不拆分，则每组 miss 仍会把整层延迟拖回 PCIe 读全量的水平。

---

## 2. MTP 投机解码：strata 机制拆解

### 2.1 模型本体（1 个 MTP 层，权重就在 mtp2.ninfer 里）

第 i 个草稿 cell 把**主模型位置 i 的末端残差 R_i** 与**已确认的 token i+1** 配对，
RoPE 用位置 i（vLLM 约定），预测 token i+2；自身产出的残差喂给下一个 cell：

```
e = fc_embedding( rms(embedding(token_i+1)) * (1 + w_emb) )       # [2560,2560] ×2
h = fc_hidden(    rms_10240(R_i)            * (1 + w_hid) )       # 逐流，10240=hc*2560
R = h + e
  → attn gated-residual（带 block_inject）
  → QSA 注意力（独立 K/V；cell 数 <2051 时走 dense 因果注意力，不碰主模型 indexer 状态）
  → mlp gated-residual
  → MoE(512 专家, top10, shared)          # 专家中间维 640，比主专家小同尺寸其实一样
  → R += y * 2*sigmoid(inject / hc)
  → hyper mixer → lm_head → 草稿 logits
```

实测草稿 1/2/3 步接受率 0.89/0.86/0.85。
工程取舍只影响草稿质量、不影响正确性（最终输出永远由主模型验证窗决定）：
投影 Q8_0、512 专家全常驻约 708MB、draft head 只算 draft_vocab 子集，共约 0.9GB。

### 2.2 验证循环（正确性只在这里）

- 窗口 T = k+1（k 个草稿 + 1 个 bonus），主模型**一次前向**算完窗口全部位置；
- `a` = 最长前缀满足 `主模型在 a+1 位置的贪心 == 草稿 token[a+1]`，提交 a+1 个；
- 下一轮从最后接受行后续草稿，被拒 bonus 行直接覆写，不做修补；
- `spec_min_p`：草稿概率过低则提前截断窗口（少算几列）；
- 温度采样时草稿与主模型用 Philox 同随机数（coupled draft）。

### 2.3 Controller 与 suffix drafter

- 每步在 none / suffix / MTP(k≤8) 三种策略中，按
  **E[提交 token 数 / 耗时]** 最大化；须比普通 decode 快 5% 才启用投机。
- 接受率用 EMA(0.05) 在线学；先验：MTP 0.86，lookup 四档 {0.35, 0.6, 0.8, 0.92}；
  CostModel 含 dense_ms、distinct_ratio 等实测参数。
- SuffixDrafter：**零权重、零 GPU** 的 trigram 哈希表（WAYS=4、min_match=3、
  max_match=32、约 20 字节/token），在重复/代码类语料上极便宜。

---

## 3. lingxi 对照设计

### 3.1 MTP（先做：纯增量、权重齐备、可 bit 级对拍）

#### M0 验证窗与接受循环（不含任何草稿模型）

- 在 [lingxi-driver](../crates/lingxi-driver/src/lib.rs) 增加
  `propose/verify` 两段式接口：
  - `verify_window(tokens: &[i64])`：复用现有 prefill 路径一次性跑 t=k+1 行
    （`layer_step` 已验证 t=5 空状态与 `layer_forward` **逐位相同**，多 token 形状是现成的）；
  - 接受判定按 §2.2 的前缀规则；窗口必须从**最后一个已提交位置**起算，状态只提交接受部分，
    被拒列作废——状态不能整窗提交后回滚，实现上注意：先在影子状态上跑整窗，确认 a 后
    再把前 a+1 行并入正式状态（或正式状态顺序逐 token 推进到接受长度，二者数值等价后再优化）。
- 此时 propose 只返回空/单 token，先把循环、状态截断、贪心对齐三件事测对。

#### M1 suffix drafter（零权重，当天可落地）

- 新模块 `crates/lingxi-driver/src/suffix.rs`：trigram 哈希表，
  WAYS=4、min_match=3、max_match=32 先用 strata 原值（纯启发式，无需对拍权重）；
  每个已生成 token 约 20 字节表项，`Sequence` 里顺带维护。
- 定义草稿源 trait，`DraftProposer::propose(&self, ids) -> Vec<DraftToken>`，
  suffix 与后续 MTP 各实现一份；Controller 只依赖 trait。
- 正确性门禁：suffix 草稿无论对错，**verify 后的输出必须与纯贪心逐 token decode 一致**
  （随机插入垃圾草稿也不改变输出——这是验证窗实现正确性的最强测试）。

#### M2 MTP 层前向（权重落地 + HF golden bit 级对拍）

- [lingxi-ids](../crates/lingxi-ids/src/lib.rs)：`TensorRole` 目前只有
  `MtpStemProjection/MtpStemNorm`，需补 MTP 注意力/MoE/mixer 角色或复用既有角色
  （现有 `AttentionProjection/Moe*/HyperConnection*` 可直接复用，只是名字解析多一个 `mtp/` 前缀）。
- 新模块 `crates/lingxi-model/src/mtp.rs`：
  - stem：fc_embedding / fc_hidden 两个乘性偏置门 `*(1+w)`、两条 rms、`R=h+e`；
  - 复用 [layer.rs](../crates/lingxi-model/src/layer.rs) 的
    gated-residual、MoE、GR 编排；
  - **注意力是唯一新 kernel**：MTP 的 QSA 用独立 K/V、cell<2051 走 dense 因果注意力。
    主模型 QSA 常态走 indexer 稀疏路径，dense 因果 kernel 需要新写
    （kernel 落点 [attn_ops.cu](../crates/lingxi-kernels/cuda/attn_ops.cu)，
    bf16，causal 掩码直接按位置推出，与 step 路径不用 causal_host 的做法一致）。
  - 独立 `MtpState`（自己的 K/V cache，**绝不碰主模型 48 层状态**）。
- golden：扩展 [tools/golden/run_layer_golden.py](../tools/golden/run_layer_golden.py)
  dump MTP 单 cell（stem 输出、attn 后、MoE 后、最终残差、草稿 logits），
  bench 加 `mtp-check`：**单 cell rel 门禁过了才允许接验证窗**（遵循不靠日志猜、bit 级回归的经验）。
- 显存取舍（第一版不要学 Q8_0）：
  - MTP 专家 512×9.83MB = **5.03GB BF16**；stem/attn/GR/head 合计约 0.2GB；
    V100 32GB 在主模型流式权重不常驻的前提下放得下，但要和专家 arena 统一做预算（见 §3.2）。
  - draft head 第一版**直接复用主 output_head**（已常驻端点，1.27GB），不做 draft_vocab 子集；
  - 投影 Q8_0 是草稿质量/显存的优化项，等功能跑通、有接受率数据后再议
    （V100 无 fp4，但 int8 tensor core 可用）。
- 接受率门禁：真实语料上 1/2/3 步接受率应接近 strata 的 0.89/0.86/0.85 量级；
  显著偏低先查 RoPE 位置约定（位置 i，不是 i+1）与残差配对行号。

#### M3 Controller

- 新模块 `crates/lingxi-driver/src/controller.rs`：策略 none/suffix/mtp(k≤8)，
  CostModel 先给保守常数（dense 注意力耗时、distinct_ratio），接受率 EMA α=0.05、
  先验照 strata 给（MTP 0.86、lookup 四档），5% 提速门槛；
- **不做** Philox coupled sampling：第一版只支持贪心，温度采样列入 server 阶段再评估。

### 3.2 专家 profile 与缓存（后做但必须做：decode 的生死线）

#### E0 先量自己，不抄 4105 槽

- lingxi 专家 9.83MB（BF16）/约 2.4MB（Q4G64 量级，以产物实测为准），
  与 strata 的 Q8_0 完全不同预算；槽位数必须由「显存预算 ÷ 槽字节」反推。
- 在 [moe.rs](../crates/lingxi-model/src/moe.rs) host 路由处
  （已有 `sel_rows: Vec<(行, 专家, 权重)>`）加一个开关式 trace dumper，
  格式每行 `(layer, token_position, expert, weight)`，bench 子命令 `routing-trace`
  在典型语料上跑 prefill+贪心生成产出 trace。
- 离线工具（新 `crates/lingxi-bench` 子命令或 tools/ 下脚本）做**十折留一**：
  对候选槽位预算（如 1000/2000/3000 BF16 槽 ≈ 9.8/19.7/29.5GB）
  分别算 h_expert；同时算 h_layer 复现「整组常驻无效」的结论作为方案旁证。
  有了自己的命中率曲线再定 arena 大小。

#### E1 profile 格式与静态 per-layer 缓存

- Rust 版 profile：建议直接采用 STRP 兼容布局（magic/version/字段一致，n_ranked 与名次表），
  读码成本最低、可借用对方工具链经验；读写放新模块 `crates/lingxi-model/src/expert_profile.rs`。
- 新模块 `crates/lingxi-model/src/expert_cache.rs`：
  - 一块连续 arena，按层等大槽（本模型同层专家同形，**不需要 VMM 分段**）；
  - `residency[48*512] -> Option<slot>`；
  - **per-layer admission 从第一天就做**（全局到达顺序的 2.97% 教训直接吸收）：
    每层按 profile 名次给固定名额，初始填满每层 top-N；
  - `fill_slot` 后必须 `verify_slot` 逐字节比对（mmap 源 vs arena 槽），失败即报错停机，
    不允许「差不多」上线；
  - 权重源：ninfer 产物 mmap + `cudaHostRegisterReadOnly` 直读
    （ninfer-dev 的 C8 修复就在这条链路上，复用其结论），miss 时 PCIe 直读 pin 住的 host blob。
- MoE 执行改 hit/miss 拆分（对应 §1.4）：
  - host 已有每 token 路由，天然可把 `sel_rows` 切成 hit/miss；
  - hit 行 GPU 算；miss 行第一版**没有 CPU 池**（lingxi 无 CPU GEMM 基建），
    走「预取到临时槽 → GPU 算」，因此必须配合 E2 的 lookahead 预取把拷贝藏在计算后面；
  - 输出按 router 原始顺序加权合并（现有 `scatter_add_rows` 的按行累加可以直接复用，
    关键是 kernel 调用前的行集拆分）。
  - grouped GEMM：V100 上第一版可继续用「逐命中专家一次 cuBLAS」
    （常驻专家权重直接指向 arena 槽，用 slice 零拷贝），
    功能正确后再评估写 grouped kernel 省 launch 开销。

#### E2 自适应换入换出 + RouterLookahead

- 参数先照搬 strata 作初值（adapt_every=4、usage 阈值 2.0、迟滞 1.5、
  全局 96、衰减 0.7），上线后用自己的 trace/在线统计调；
- 拷贝必须在**独立 CUDA stream**，且只在验证窗口之间 join；未完成则本轮跳过、
  residency 表在 event 落地 + verify 通过后才翻转；被换出专家写回 host 紧凑镜像；
- RouterLookahead：lingxi router 是「GPU 算 logits→D2H→host 决策」，
  预取需要在 host 上提前算 l+1 层路由——把 l+1 的 router 权重（513×2560×2≈2.6MB/层）
  pin 在 host，用 host 侧 GEMM（后续选型）对第 l 层输入做路由预测，提前一个窗口发起 DMA。
  注意 host 预测与 GPU 真路由不一致时以真路由为准，预取只是 hint。

#### E3（可选，远期）Q4G64 专家与 CPU 池

- BF16 4.72GB/token 的暴露字节决定了：要达到 strata 那种 50ms/token，
  最终要么走 Q4G64 专家（V100 sm_70 无 fp4 指令，dequant  kernel 的代价要实测），
  要么建 CPU 专家池（等于在 Rust 侧重写一套 GEMM 调度，工作量大，排最后）。
- 双卡 DISJOINT 去重等 GPU0 解套后再说。

### 3.3 不移植 / 暂缓清单

| strata 机制 | lingxi 决定 | 理由 |
|---|---|---|
| VMM 分段 arena | 不做 | 同层专家等大，固定槽即可 |
| Q8_0 MTP 投影 | 暂缓 | 先 BF16 跑通拿到接受率，显存不够再量化 |
| draft_vocab 子集 | 暂缓 | 复用主 output_head，少一套权重分叉 |
| Philox coupled sampling | 暂缓 | 第一版只贪心；温度采样随 server 阶段评估 |
| CPU 专家池 | 暂缓/最后 | Rust 侧无 CPU GEMM 基建，先用 PCIe 直读+预取 |
| 双卡 DISJOINT | 暂缓 | 当前只有 GPU1 可用 |
| Controller 在线学习 | 初版给死参数 | 先验常数起步，有流量后再开 EMA |

---

## 4. 建议的落地顺序与门禁

| 阶段 | 内容 | 正确性/性能门禁 |
|---|---|---|
| M0 | driver 验证窗 + 接受循环（propose 为空） | 影子状态验证后输出与逐 token 贪心**逐 token 相同** |
| M1 | suffix trigram drafter | 随机垃圾草稿也不改变贪心输出；重复语料上实测提交率 |
| M2a | lingxi-ids 角色 + mtp.rs 前向 + dense 因果注意力 kernel | `mtp-check` 对 HF golden 单层/单 cell rel 达 bf16 量级 |
| M2b | MTP 草稿接验证窗 | 真实语料 1/2/3 步接受率 ≈ 0.89/0.86/0.85 量级 |
| M3 | Controller 三策略选路 | 端到端 tokens/s 相对纯贪心提升 >5% 才启用投机 |
| E0 | routing trace + 十折留一命中率曲线 | 输出自己的槽位—命中率表，定 arena 预算 |
| E1 | STRP profile + per-layer arena + verify_slot + hit/miss 拆分 | 常驻槽字节校验全过；decode 结果与无缓存贪心 **token 完全一致**；tokens/s 量化 |
| E2 | 独立流自适应换入换出 + RouterLookahead | 长对话多轮后命中率不低于静态 profile；输出仍完全一致 |
| E3 | Q4G64 专家 / CPU 池（按需） | 数值回归 + 吞吐收益双达标 |

顺序上 M0–M2 与 E0 可并行（一个人做 kernel/算法，trace 采集只需 bench 开关）；
E1 是 server 上线前的硬前提——没有专家缓存，机械盘上的 decode 不可用。

## 5. 主要风险

1. **dense 因果注意力 kernel** 是 MTP 唯一新算子，且 bf16 在 V100 上的归约舍入要预留容差；
   必须先单 kernel 对 HF golden，再进 MTP 整层。
2. **状态提交语义**：验证窗整窗在影子状态上计算，确认接受长度后再并入，
   避免被拒 token 污染 GDN/QSA/PLE 状态（PLE 只有主模型第 1 层，MTP 无 PLE）。
3. **PCIe x8 + 机械盘**：即使有 arena，冷启动填充与 miss 抖动仍会很明显；
   profile 出厂预填 + lookahead 缺一不可。
4. **BF16 专家预算**：32GB 卡上 arena 与 MTP 5GB 常驻、主模型工作区要同盘算账，
   槽位数以 E0 实测命中率曲线为准，不许拍脑袋。
5. 所有数值门禁延续现有规矩：**逐层/逐算子 bit 级或 bf16 容差回归，不靠日志猜机制**。
