# lingxi 架构与程序规范

> 本文是硬约束。任何模块不得违反；违反即回退重写。

## 0. 四条铁律

1. **只经 ids 通信。** 模块之间不 import 对方的业务类型、不引用对方的字面量名字/常量。
   一切跨模块身份（tensor、层、算子、kernel、模型、槽位、会话）都由 `lingxi-ids` 定义并通过它传递。
   未来换模型 / 换 kernel 实现 / 加功能 → **只改 `lingxi-ids` 的注册表与映射**，模块内部零改动。
2. **一个模块只做一件事。** 功能不穿插。模块边界 = 责任边界；
   想加别的事 → 新增模块，不改已有模块的职责（可替换设计）。
3. **正确性靠结构，不靠调英雄。** 静态规划 + 逐层 golden；禁止用"端到端能出字"作任何模块的验收。
4. **改一处只重编一处。** 改 server 不重编 kernel；改 model 不碰 kernels。
   这是编译时间可控与并行开发的前提。

## 0.5 复用优于自研（硬约束）

**在别人已经做对的地方，禁止自己重新设计、禁止重复造轮子。**
`lingxi` 只承载"我们的、特殊的"增量（ids 解耦、模块边界、可更替架构）。

具体落法：
- **产物格式 = ninfer v2 容器**（`NINFER\0\2 | json_bytes(u64) | JSON directory | pad→4096 | payload`）。
  `lingxi-artifact` 是它的**读取器**（我们自己的实现，格式完全兼容），**直接读现成 `.ninfer` 产物**，
  绝不从 BF16 重转一遍。
- **数值格式 / 张量布局**（`BF16/FP32/I32/Q3G128_F16S/Q4G64_F16S/Q5G64_F16S/Q6G64_F16S/W8G32_F16S/NVFP4/FP8…`；
  `contiguous-le-v1/row-split-k128-v1/blockscale-k16-m128x4-v1/row-scale-v1`）**照搬 ninfer 注册表语义**，
  不自造简化版（自造版在行可寻址性、K 对齐、多精度上都不如现成的）。
- **张量清单 = ninfer 的 canonical inventory**（GR/qkv/MoE 融合、PLE 外置表、资源对象），不自起一套命名。
- 需要"我们的"扩展时，**在 ninfer 的成果上增量**（例如 `role` 由 canonical 名字在加载期推导，
  不需要改产物），而不是另起一条平行管线。
- 判断标准：若自研方案在成熟度/性能/正确性上不如现成方案 → **必须复用**，只保留我们的架构增量。

## 0.6 离线优先：运行时零转换（硬约束）

**一切能提前做的，都在离线阶段做完；运行时只做"读 + 算"。**

- 离线阶段（转换器/预处理）负责：量化、融合（GR/qkv/MoE）、gamma 烘焙、**按 kernel 访问模式预排布布局**、
  外部大表（PLE/n-gram 表、专家区）生成、以及任何需要重排/重编码的工作。
- 产物里存的必须是 **kernel 可以直接消费的形态**：`row-split-k128-v1` 的行可寻址、K 对齐补零、
  平面 256B 对齐，都是为此服务——**运行时不做反量化、不做转置、不做融合、不建临时表**。
- L0 Plan 在加载期一次算死全部 offset，之后只读（`ARCHITECTURE.md` §1）。
- 目的：**GPU 的带宽与算力 100% 用于前向计算**，不被"准备数据"消耗。
  （MoE decode 只读命中专家；若运行时还要把权重重排一遍，就是纯浪费。）
- 因此：新需求优先考虑"能否离线预生成/预处理"，而不是"运行时能不能算出来"。

## 1. 依赖方向（单向，禁止回环）

```
lingxi-ids            ← 最底层，零依赖（无 IO / 无 CUDA / 无 tokio）
  ↑
lingxi-artifact       （容器格式；只依赖 ids）
lingxi-plan           （内存布局；只依赖 ids）
lingxi-kernels        （cubin + launch；只依赖 ids）
  ↑
lingxi-model          （前向；依赖 ids + plan + kernels + artifact）
  ↑
lingxi-driver         （调度/槽位/批循环）
  ↑
lingxi-server         （HTTP/SSE）
lingxi-bench          （旁路工具；依赖 plan/artifact/model，不反向）
```

## 2. `lingxi-ids`：唯一的跨模块契约

`lingxi-ids` 里定义且**只**定义"身份 + 结构"，不含任何算法：

| 类型 | 作用 |
|---|---|
| `TensorId(u32)` | 全局 tensor 身份。加载期由 `TensorCatalog` 按 manifest 顺序分配 |
| `LayerId(u32)` / `LayerKind` | 层身份与层类型（GDN / QSA / …，未来扩 `Dense` 等只在此加） |
| `OpId` | 逻辑算子身份（与实现解耦）：`RmsNorm`/`Gemv`/`QsaAttention`/`MoeExpert`/`GrWrite`… |
| `KernelId` | 具体 kernel 实现身份；`OpId → KernelId` 映射可替换 |
| `ModelId` / `Dims` / `LayerSpec` | 模型身份与静态形状（编译期常量集中地） |
| `Dtype` / `Semantic` / `TensorRole` | 共享词表 |
| `TensorCatalog` | `TensorId ↔ TensorInfo(name,shape,dtype,semantic,role)` 双向表 |

连接方式（模块只认这些）：

```rust
// plan 侧提供：全局 tensor → 设备/主机内存跨度
pub trait TensorResolver { fn span(&self, id: TensorId) -> Span; }

// kernel 侧提供：逻辑算子 → 具体 kernel 实现（换实现只改这张表）
pub trait KernelRegistry { fn kernel(&self, op: OpId, shape_tag: ShapeTag) -> KernelId; }

// 模型侧提供：模型静态描述（换模型只加一个 impl）
pub trait ModelSpec {
    fn id(&self) -> ModelId;
    fn dims(&self) -> &Dims;
    fn layers(&self) -> &[LayerSpec];
}
```

`TensorId` → 名字/形状的解析**只在 ids/artifact 层发生**；model/kernel/driver 只持有 `TensorId`，
永远不出现 `"model.language_model.layers.7.self_attn.q_proj.weight"` 这类字符串。

## 3. 各模块职责（不重叠）

- ** lingxi-artifact **：`.lingxi` 容器（`magic | version | manifest_len | JSON manifest | pad→4096 | blob`）
  的读写；加载期 T0 断言（sha1 / semantic 均值 / 布局）。**不做前向、不碰 CUDA。**
- ** lingxi-plan **：把 catalog 摊成运行时布局（权重区 / KV 区 / indexer 区 / 工作区），产出 `Span`。
  **只算地址，不做计算。**
- ** lingxi-kernels **：`build.rs` 调 nvcc 编 `cuda/*.cu` → cubin；`cudarc` 加载与 launch 封装。
  **`unsafe` 只允许出现在这里**；纯函数 `(ptr+param)→out`，stream 显式传入。
- ** lingxi-model **：前向 = `forward_layer(&Weights, &mut Ctx, LayerId)`。MTP 层是"第 49 层"。
  **只做数学编排，不做调度、不做 IO。**
- ** lingxi-driver **：slot 池、准入、静态批 decode 循环、MTP draft/verify、前缀共享、取消。**不做 HTTP。**
- ** lingxi-server **：HTTP/SSE/分词。**不做计算、不做调度决策。**
- ** lingxi-bench **：压测与对拍入口；golden 消费。**旁路，不进热路径。**

## 4. 可更替设计（未来换模型 / 换实现）

- 换 kernel：改 `KernelRegistry` 映射（`OpId→KernelId`）即可，model 不动。
- 换模型：新增一个 `impl ModelSpec` + 一个新 catalog；`lingxi-artifact`/`plan`/`kernels`/`driver` 不动。
- 换服务协议：只改 `lingxi-server`。
- 加功能（如新采样器、新注意力）：新增 `OpId` + 实现 + 注册，旧路径字节不变。

## 5. 调试纪律（血泪）

- 探针/dump 读取前必须 stream 同步；调试 buffer **启动时一次预分配**，热路径禁止 alloc（含 cudaMalloc）。
- 调试代码不得改变被测系统的分配数量/时序。
- 构建失败时禁止用旧二进制出结论——跑测试前先确认构建时间戳。
- 所有 dump 配对用**单调 call counter**，不用指针匹配。
- 每个 kernel/模块合入即配齐测试，测试先于优化。

## 6. 验证分级

| 级别 | 内容 | 基准 | 容差 |
|---|---|---|---|
| T0 | 加载期断言（sha1 + gamma 均值 + 布局） | manifest | 失败即 panic |
| T1 | 每 kernel 契约测试（随机输入） | torch 参考 | rel < 2e-3 |
| T2 | golden 逐层（固定 prompt 逐层输出） | HF 落盘 .npz（**bf16**） | 逐行；每行与"参考并列翻转行"除外后 < 1.5e-2 |
| T3 | 端到端贪心（20+ prompt） | HF 贪心序列 | 完全一致 |

**T2 说明（bf16 现实）**：golden 用 HF 模块 + 产物真实权重、**bf16** 前向落盘（与引擎精度一致）。
- 深层 MoE router 的 bf16 logits 在第 `top_k`/`top_k+1` 名会**精确并列**（gap == 0），参考
  `torch.topk` 的并列取舍**未定义且自相矛盾**（同一层不同行有时取最大下标、有时取最小）。
  这会让个别 token 行的专家选择翻转、整行 rel 变到 2e-2~8e-2。**这不是引擎缺陷**，已用
  "同一输入下重跑 HF MoE、对比选择"证伪（一致行 rel ≈ 6.7e-3）。
- 因此 T2 判据为**逐行**：至少一行完全复现（证明 GR/注意力/MoE/注入的结构正确），
  其余行若越界需能归因到并列翻转；整层 rel 不作为判据（会被翻转行主导）。
