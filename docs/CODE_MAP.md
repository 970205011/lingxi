# 代码地图与重构计划

> 目标：把“能跑的巨石”拆成有边界的模块。原则：**每刀只搬家、不改行为**；每刀后
> `cargo build --release` + 现有 gates 必须全绿；一刀一个提交。

## 1. 现在的模块边界

| crate | 职责 | 依赖 |
|---|---|---|
| `lingxi-ids` | 跨模块契约：张量/角色/格式/布局/层/算子 ID | 无 |
| `lingxi-artifact` | `.lingxi` / `.ninfer` 容器读写、结构断言、专家 arena 解析 | ids |
| `lingxi-plan` | 由 catalog 生成运行时内存布局（只算地址） | ids, artifact |
| `lingxi-kernels` | CUDA C++ → cubin + cudarc launch；唯一允许 unsafe 的地方 | ids |
| `lingxi-model` | 前向、模型静态描述、角色推导、开关注册表 | ids, artifact, plan, kernels |
| `lingxi-driver` | slot 池、准入、decode 循环、MTP、KV 预算 | model |
| `lingxi-server` | HTTP/SSE、分词 | ids |
| `lingxi-bench` | CLI、压测、golden 对拍、serve 引擎 | 全部 |

## 2. 需要拆的热点

| 文件 | 行数 | 问题 | 拆到 |
|---|---:|---|---|
| `crates/lingxi-bench/src/main.rs` | 6261 | 所有子命令、talk/serve 装配、自动调优、MTP 探针挤在一个文件 | `cli/` + `cmd/*` |
| `crates/lingxi-bench/src/arena_source.rs` | 1991 | 槽缓存、混档、预热/钉表、DMA/pin 混在一起 | `arena/{source,slot,prewarm,mix,pin}.rs` |
| `crates/lingxi-model/src/moe.rs` | 2249 | router、dispatch、GEMV/TC、远端卡路径同文件 | `moe/{router,dispatch,gemm,remote}.rs` |
| `crates/lingxi-driver/src/lib.rs` | 2879 | decode 主循环、KV 账本、前缀共享、MTP 提交同文件 | `driver/{decode,kv,prefix,spec}.rs` |
| `crates/lingxi-kernels/src/lib.rs` | 2356 | 所有 launch 封装平铺 | `kernels/{attn,moe,q4,gdn,topk,rmsnorm}.rs` |
| `crates/lingxi-bench/src/serve/proto.rs` | 877 | 请求/响应/SSE 编码 | `serve/proto/{chat,sse,tools}.rs` |

## 3. 分阶段

### Phase 1 — CLI 骨架（先做）

1. `crates/lingxi-bench/src/vram.rs`：把 `autotune_arena_slots` 和 `vram-budget` 子命令搬出
   `main.rs`；`main.rs` 只保留 dispatch。
2. `crates/lingxi-bench/src/cli/`：`mod.rs` 负责解析；`cmd/verify.rs`、`cmd/bytes.rs`、
   `cmd/vram.rs`、`cmd/talk.rs`、`cmd/serve_cmd.rs`、`cmd/mtp.rs` 各自一个入口函数。
3. 验收：`cargo build --release`；`vram-budget` 输出与搬迁前逐字节一致；
   `make regress` / `make gates` 全绿。

### Phase 2 — arena

`arena_source.rs` 拆成 `arena::source`（对外 API）、`arena::slot`（LRU/槽表）、
`arena::prewarm`（hist/钉表）、`arena::mix`（Q4/Q3 双 cache）、`arena::pin`（介质判定/DMA）。
验收：首 token 11751、贪心序列不变；混档字节级 gate 通过。

### Phase 3 — model / driver

`moe.rs` 按 router / dispatch / gemm / remote 拆；`driver/lib.rs` 按 decode / kv / prefix / spec 拆。
验收：T2 逐层 golden + T3 端到端逐位；`LINGXI_MOE_SPLIT` 路径单独跑一遍。

### Phase 4 — kernels / serve

`kernels/src/lib.rs` 按算子域拆；`serve/proto.rs` 按 chat/SSE/tools 拆。
验收：T1 全过；serve×paged 三场景 + 缓存命中 gate。

## 4. 版本节奏

- `v0.1.x`：当前公开版（V100 可交互、Q4/Q3 混档、MTP、paged KV）。
- `v0.2.0`：Phase 1–2 完成 + **prefill 张量核**（prompt attention 与 grouped expert GEMM），
  目标把 33K prefill 从 278.7 s 拉到 100 s 以内，同时 decode 不回退。
- `v0.3.0`：Phase 3–4 + KV paging 默认策略 + 自动校准（PCIe/worker/spec）。

## 5. 纪律

- 不做“顺手重构”：每刀只移动代码、不改数学；行为变更单独提交。
- 每刀附测量：首 token / 贪心序列 / gates / 关键 A/B。
- README 只留用户可见的结果；过程、负结果和实验记录放 `docs/`。
