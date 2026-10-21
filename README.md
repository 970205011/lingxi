# lingxi（灵犀）

[English](README.en.md) | 简体中文

lingxi 是一个为 **NVIDIA sm_70（Tesla V100）** 写的 Qwen3-Next 推理引擎，Rust + CUDA 实现。
它把 48 层（36 层 GDN 线性注意力 + 12 层稀疏全注意力）、512 专家、每 token 激活 10 个的
MoE 模型，在 2×V100 32G 上跑到可交互速度：单流 **43 tok/s**，MTP 投机约 **64 tok/s**，
256K 上下文可用。Apache-2.0 开源。

它不是通用 serving 框架，只服务一个具体场景：一或两个人、一两块老卡，想和这个量级的模型
流畅对话、跑 Agent、处理长文档。

[一页简介（HTML）](docs/intro.html) · [GitHub](https://github.com/970205011/lingxi) · [详细文档](#文档)

## 性能（实测）

测试机：2× Tesla V100-PCIE-32GB（GPU0 x16、GPU1 x8，无 NVLink），251 GB 内存。
所有数字均来自通过位级判据的配置。

| 场景 | 结果 |
|---|---|
| 单流 decode（双卡，p10） | **43 tok/s**（23.2 ms/token） |
| decode + MTP 投机（kd=5, min_p=0.7） | **约 64 tok/s** |
| 双卡层间切分（真实中文语料 vs 单卡） | 20.7 → **54.1 tok/s**（+161%） |
| serve 短 prompt 投机 | **89 tok/s** |
| 875 token 前缀 + 投机 | **38.4 vs 29.9 tok/s** |
| 256K 上下文 decode + MTP | **59.1 tok/s** |
| 首 token（4.2K prompt，双卡线程版） | **28.7 s**（顺序 46.2 s） |
| 23.7K token 冷 prefill | 143.2 s（6.03 ms/token） |
| 33K token 冷 prefill（调优后） | 278.7 s（−17.3%） |
| 256K token 冷 prefill | 3436.9 s（57.3 min），峰值 31.9/27.3 GiB，未 OOM |
| 同一 256K prompt 重发（前缀命中） | **TTFT 0.00 s，整请求 6.4 s** |
| 多轮对话第二轮 | 28.1 s → **0.35 s（81×）** → 原样重发 **0.16 s（177×）** |

单流 200 tok/s 在 V100 上物理不可达：每 token 需搬约 4.6 GB，900 GB/s 峰值给出 196 tok/s
的理论地板（要求 100% 带宽效率）。再往上只能靠多请求并发聚合带宽，或换更小的模型。

## 能力

- **专家驻留**：Q4G64 arena（60.25 GiB）+ 显存 LRU 槽 + 离线钉表 + 在线重钉；
  混档 arena（热 Q4 / 冷 Q3）压到 46.39 GiB，端到端计时与全 Q4 打平。
- **MTP 投机**：概率闸门（min_p=0.7）+ 零回滚提交；生成序列逐 token 不变。
- **长上下文**：QSA 稀疏注意力只 gather 选中块；paged KV（设备 32K 窗口 + host 锁页块池）；
  FP8 PLE 表驻内存；256K 冷 prefill 57.3 分钟，前缀命中 TTFT 为零。
- **服务层**：手写 OpenAI 兼容 HTTP/SSE；精确 / 智能前缀缓存；多会话并发；不活跃会话落盘。
- **数值验证**：T1 每 kernel 对 torch、T2 逐层 golden、T3 端到端贪心逐位，外加 4 道护栏 gates。
- **产物兼容**：读写 ninfer v2 容器；上游 `.ninfer` 开箱可读，lingxi 内容约定用 `.lingxi` 后缀。

## 显存与部署

| 配置 | 实测显存 | 说明 |
|---|---|---|
| 单卡 Q4，9000 槽 | 31.3 / 32 GiB | 32K 上下文 |
| 双卡 32K + MTP | 29.9 / 27.8 GiB | 9600 / 10000 槽 |
| 双卡 256K 启动 | 23.7 / 23.2 GiB | 7200 / 7600 槽 |
| 256K，250.6K prefill 峰值 | 31.9 / 27.3 GiB | 未 OOM |

显存账可由 `lingxi-bench vram-budget <artifact> <arena> 24 32768` 逐项打印（不加载权重）。
32K 下 GPU1 的 12,288 个专家可全部驻留；GPU0 差 865 个冷专家（1.58 GiB），把 KV 放 host
或收紧预留后可全驻留。256K 时用 `LINGXI_QSA_PAGED=1` 只保留 32K KV 窗口，其余在 host。

## 快速开始

```bash
make build        # 需要 rustc + nvcc / CUDA 12.8
./start.sh        # 生产启动器
./start.sh ./bin/lingxi-bench serve <产物.lingxi 或 .ninfer> 11434
make regress      # 端到端判据
make gates        # 护栏 gates
```

```bash
curl http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "lingxi",
  "messages": [{"role": "user", "content": "The capital of France is"}]
}'
```

产物（约 253 GB）不入库，用 `LINGXI_ARTIFACT` 或命令行参数指定；从 HF checkpoint 重建见
`tools/convert/hf2lingxi.py`。全部运行参数是环境变量，登记在
`crates/lingxi-model/src/probes.rs`。

## 项目结构

```
crates/
  lingxi-ids        跨模块契约的唯一来源
  lingxi-artifact   .lingxi / .ninfer 容器读取器/写入器
  lingxi-plan       由 catalog 生成运行时内存布局
  lingxi-kernels    CUDA C++ → cubin（build.rs 调 nvcc）+ cudarc
  lingxi-model      前向逻辑 + 模型静态描述 + 张量角色 + 开关注册表
  lingxi-driver     slot 池 · 准入 · decode 循环 · MTP · KV 预算
  lingxi-server     HTTP/SSE（OpenAI 兼容）· 分词
  lingxi-bench      产物检查 · 压测 · golden 对拍 · serve 引擎
```

依赖方向单向无回环（ids → artifact/plan/kernels → model → driver → server），unsafe 只允许
出现在 `lingxi-kernels`。

## 文档

- [docs/intro.html](docs/intro.html) — 一页简介
- [ARCHITECTURE.md](ARCHITECTURE.md) — 架构硬约束与模块边界
- [docs/FORMAT.md](docs/FORMAT.md) — `.lingxi` / `.ninfer` 容器与数值格式
- [docs/decode-speed-profile.md](docs/decode-speed-profile.md) — 性能剖面与全部实测
- [docs/speedup-audit.md](docs/speedup-audit.md) — 优化项审计（含被证伪项）
- [docs/PLAN.md](docs/PLAN.md) / [docs/PLAN_NEXT.md](docs/PLAN_NEXT.md) — 工程决策记录
- [docs/strata-0.2.2-v100.md](docs/strata-0.2.2-v100.md) — 同机 Strata 0.2.2 对比实测
- [docs/BACKLOG.md](docs/BACKLOG.md) — 待办与已知问题
- [docs/HANDOVER.md](docs/HANDOVER.md) — 运维与排障
- [docs/ASSETS.md](docs/ASSETS.md) — 资产清单与再生命令
- [docs/bench-logs/](docs/bench-logs/) — 关键实测日志

## License

Apache License 2.0（见 LICENSE）。第三方依赖各自许可证见其仓库。
