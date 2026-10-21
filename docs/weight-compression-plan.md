# 灵犀权重压缩方案 · 决策与归属

> 本文回答一个问题：**在 V100(sm_70) + 机械盘 + 251 GiB 内存的约束下，专家权重用什么表示最划算。**
> 所有结论都来自本仓库 `tools/golden/quant_*.py` 在**真实产物权重**上的实测，不是文献转述。
> 归属一节列明每个机制的借鉴源，供未来开源时合规引用。

---

## 0. 结论速览

| 项 | 决定 |
|---|---|
| 专家权重格式 | **W4A16（AWQ 校准 + GPTQ 补偿 + imatrix）× 随机 Hadamard 去相干（RHT）** ≈ 3.1~3.4 等效 bpw |
| 为什么不上 2bit/三值 | 香农下限：误差=2^(−R)。W4 级精度（误差 0.13）下限 **2.94 bpw**，2bit 下限误差只有 **0.25**（差 2 倍） |
| 为什么不上"更聪明的编码" | 实测：专家独立、满秩、零相关、索引无记忆 → 低秩/共享基/LZ/上下文熵编码**全部无效** |
| 内存策略 | 量化后专家 60 GiB，**全量常驻 host RAM** → 机械盘彻底出局 |
| 显存策略 | device 槽 arena + 直接从 pinned arena **DMA 填槽**（删掉 CPU 拷贝层）；MTP 常驻显存 |
| 剩余可选收益 | 0 阶熵编码 −18%；TCQ 逼近香农界，最多再 −15%（工程量大于收益，列为第二刀） |

**预期**：decode 从 ~1000 ms/token 降到 **~20–40 ms/token**（25–50×）。

---

## 1. 底账（qwen4_exp，实测）

- 专家：48 层 × 512 × 9.375 MiB（gu 6.25 + dn 3.125）= **225 GiB**；全产物 234 GiB；PLE 表 95 GiB（每 token 只随机读 16 行 ≈ 5 KB，**留盘**）
- 每 token 触达 480 专家 = **2.36 G 权重**
- GPU0 PCIe 3.0 **x16** ≈ 12 GB/s（GPU1 是 x8，且有掉线前科）；V100 HBM2 900 GB/s；FP32 ≈ 15.7 TFLOPS
- **算力 : 传输 ≈ 0.30 ms : 393 ms ≈ 1 : 1300** → 瓶颈 100% 在"每 token 传多少字节"
- 产物位于 `机械盘`（ROTA=1 **机械盘**）

---

## 2. 实测率失真前沿（layer0 真实专家）

工具：`tools/golden/quant_frontier.py`、`tools/golden/quant_entropy.py`

| 方案 | 名义 bpw | 等效 bpw（熵编码后） | 权重相对误差 |
|---|---|---|---|
| W8 仿射 g128 | 8.12 | 7.54 | 0.0072 |
| **W4 仿射 g128** | 4.12 | **3.39** | **0.1300** |
| **W4 + RHT256** | 4.12 | — | **0.1171** |
| 分块加性 VQ g4 b8 M2 | 4.05 | — | 0.1085 |
| W3 仿射 g128 | 3.12 | 2.23 | 0.3025 |
| 分块加性 VQ g8 b10 M2 | 2.90 | — | 0.2180 |
| VQ g4 b8（分块） | 2.05 | — | 0.3198 |
| VQ g8 b8（分块） | 1.10 | — | 0.5637 |
| W2 仿射 g128 | 2.12 | 0.82 | 0.7854 |

**曲线是断崖的**：3.4 bpw → 0.13；2.2 bpw → 0.30（2.3×）。

---

## 3. 逐方案判决（附实测依据）

| 方案 | 实测结果 | 依据 | 判决 |
|---|---|---|---|
| **RHT 去相干** | kurtosis 3.62→**3.02**，逐列 std 离散度 6.25→**1.19**，W4 误差 0.130→**0.117** | `quant_incoherence.py` | ✅**采用**（几乎免费） |
| **加性向量量化 VQ** | 同 bpw 比标量好 ~17%（0.1085 vs 0.1300） | `quant_frontier.py` | ⚠️ 收益有限、要码本内核，暂缓 |
| 分块码本 vs 全局码本 | 0.3198 vs 0.3239（无差异） | `quant_frontier.py` | ❌ 块间同分布，无益 |
| 0 阶熵编码 | 4bit 索引熵 4.000→**3.269**（省 18%） | `quant_entropy.py` | ⚠️ 真收益但 GPU rANS 解码复杂，第二刀 |
| **上下文熵编码**（DeepCABAC/CABAC 思想） | H(X)=3.269 vs H(X\|X−1)=**3.267** → **0.0%** | `quant_deep.py` | ❌ 索引无记忆，CABAC 那套对图像/文本有效、对权重无效 |
| outlier 分离（top-p 追加 4bit） | p=5% → 0.130→0.116，代价 +0.26 bpw | `quant_entropy.py` | ❌ 不划算 |
| **专家共享基 / Tucker 分解**（TD-MoE、BitsMoE） | 专家模式奇异值谱近平坦（top1=3.64% vs 均匀 3.13%）；跨专家逐元素相关 ≈ 0.000 | `quant_deep.py`、`quant_incoherence.py` | ❌ 我们的专家相互独立 |
| **低秩 / 公式代替** | rank 256 只占 53% 能量 | `quant_entropy.py` | ❌ 满秩随机 |
| **字典 / LZ（7z 思想）** | 相邻列相关 −0.0005、相邻行 0.0002 | `quant_entropy.py` | ❌ 零相关，无重复可利用 |
| 权重共享 / 循环复用（模型内同一权重反复用） | 需**重新训练**才能引入层间/步间共享 | 见 §6 归属 | ❌ 权重固定时不可后处理得到 |

---

## 4. 决定的数学：香农下限

RHT 后权重是标准 i.i.d. 高斯（kurtosis 3.02）。高斯源的率失真函数有闭式解

```
D(R) = σ² · 2^(−2R)      ⇒     相对误差 = 2^(−R)
```

| 目标误差 | 香农下限 bpw | 我们实测等效 bpw | 差距 |
|---|---|---|---|
| 0.0072（W8 级） | 7.12 | 7.54 | +6% |
| **0.1171（W4+RHT 级）** | **3.09** | ~3.2 | **+4%** |
| 0.1300（W4 级） | 2.94 | 3.39 | +15% |
| 0.3025（W3 级） | 1.72 | 2.23 | +30% |
| 0.7854（W2 级） | 0.35 | 0.82 | +134% |

**两条铁结论：**

1. **W4 精度档下，我们距离物理下限只有 10~15% 余量**——任何更"聪明"的编码最多再榨这么多。
2. **2 bit 的最优量化器误差下限是 0.25**（比 W4 差 2 倍）。**2-bit 不可能在权重层面追平 4-bit**；三值(BitNet)可行是因为它是 **QAT 从零训出来的**，不是压出来的。

---

## 5. 采用方案与落地步骤

1. **离线量化**：专家 → W4A16，AWQ 校准 + GPTQ 补偿 + imatrix；每层先做随机 Hadamard（RHT），推理时对激活做一次 FWHT（O(n log n)，几乎零成本）。
2. **host 常驻 pinned arena**：量化后约 60 GiB，一次性 `cudaHostRegister`，机械盘彻底出局。
3. **device 槽 arena**：GPU0 留 ~22 GiB ≈ 9400 槽 / 24576；填充 = `cudaMemcpyAsync(槽, pinned_blob)`（**一条 DMA，无 CPU 参与**）。
4. **per-layer 名额 + 路由 profile**：避免"全局到达顺序"退化（Strata 实测仅 2.97% 命中）。填完逐字节 `verify`。
5. **RouterLookahead 预取**：提前一层用 host 侧 router 预测，DMA 藏在计算后。
6. **MTP 常驻显存**（512 专家 BF16 约 5 GiB）。
7. **可选第二刀**：0 阶熵编码（−18%）；TCQ（逼近香农界）。
8. **门禁**：量化前后逐层对 HF golden 做**容差回归**（不再要求 bit 级，因为权重已变）；端到端贪心输出与 BF16 基线对比。

---

## 6. 归属 / 借鉴源（开源必备）

> 原则：**只借鉴机制与数学，不搬运代码**；下表逐项标明来源与许可线索。

| 机制 | 来源 | 性质 | 我们的用法 |
|---|---|---|---|
| **host 常驻 pinned arena（PinnedArena）** | Strata（github `Niko1221/Strata`，`include/strata/core/pinned.hpp`） | 开源工程 | 借结构：页锁定 + `cudaHostRegister`，一次装载 |
| **VRAM 专家槽 + residency 表 + per-layer 名额 + verify_slot** | Strata（`src/core/expert_cache.cpp`、`tests/core/expert_cache_per_layer_test.cpp`） | 开源工程 | 借机制与其"全局到达顺序 → 2.97%"的失败教训 |
| **DMA 批量上传 / PCIe 直读** | Strata（`include/strata/core/dma_batch.hpp`、`docs/BATCHED_DMA.md`） | 开源工程 | 借"直接从 pinned blob `cudaMemcpyAsync`" |
| **RouterLookahead 预取 / MTP 投机** | Strata（`src/core/mtp.cpp`、`src/spec/`） | 开源工程 | 借预取与投机框架 |
| **Hadamard 去相干（RHT）** | QuIP（Malinovskii et al.）/ **QuIP#**（Tseng et al., ICML'24, arXiv:2402.04396） | 学术论文 | 借 RHT 做 outlier 抑制；E8 点阵码本暂未采用 |
| **Trellis 编码量化（TCQ）** | **QTIP**（Tseng et al., NeurIPS'24, arXiv:2406.11235）；CCQ（arXiv:2507.07145）；XOR-Trellis（arXiv:2610.00432） | 学术论文 | 列为第二刀，借鉴"bitshift trellis + 状态机解耦码本与码率" |
| **AWQ 激活感知量化** | Lin et al., MLSys'24 | 学术论文 | W4A16 的校准部分 |
| **GPTQ Hessian 补偿舍入** | Frantar et al., ICLR'23 | 学术论文 | W4A16 的误差补偿部分 |
| **imatrix（重要性矩阵）** | llama.cpp（ggerganov/ggml, MIT） | 开源工程 | 校准权重 |
| **上下文自适应算术编码（评估后未采用）** | **DeepCABAC**（MPEG NNC，ISO/IEC 15938-17；arXiv:1905.08318） | 国际标准 | 实测上下文建模对 i.i.d. 权重 0% 收益，不采用 |
| **专家共享基 / Tucker 分解（评估后不适用）** | BitsMoE（arXiv:2606.00079）；TD-MoE（OpenReview H5NX243IUP）；Minima（arXiv:2602.01613） | 学术论文 | 实测我们专家独立，不适用；架构可作文档备查 |
| **QAT 极低比特（评估后不适用）** | ParetoQ（arXiv:2502.02631）；SAGE（arXiv:2610.14745） | 学术论文 | 需重训权重，与"权重固定"约束冲突 |
| 产物容器 / recipe 反解 | ninfer（本机 `$DATA/ninfer-dev`，已弃用） | 自有 | 仅保留读取工具链 |

---

## 7. 复现命令（测试环境 内）

```bash
cd $MODELS/lingxi
export PYTHONPATH=$MODELS/pocenv/lib/python3.12/site-packages:$DATA/ninfer-dev

python3.12 tools/golden/quant_probe.py        # 产物对象形状/格式
python3.12 tools/golden/quant_frontier.py     # 各量化/压缩方案的率失真前沿
python3.12 tools/golden/quant_entropy.py      # 索引熵 / outlier / 低秩 / 邻域相关
python3.12 tools/golden/quant_incoherence.py  # RHT 去相干 + 专家间共享结构
python3.12 tools/golden/quant_deep.py         # 专家模式谱（Tucker）+ 上下文条件熵
```

---

## 8. KV 缓存压缩（未来方向）

**权重没有上下文，KV 有——这是两者的根本差别，也决定了能用的方法完全不同。**

| | 权重（本方案的主体） | KV 缓存 |
|---|---|---|
| 结构 | 实测 i.i.d. 高斯、满秩、零相关 | 沿 token 维时间相关、跨层冗余、低秩 |
| 上下文熵编码 | 实测 **0.0%** 收益 | 有效 |
| 低秩 / 共享 | 无效（谱平坦） | 有效（MLA） |
| 跨层复用 | 无效（专家独立） | 有效（CSA2 / XQuant / KVAlign） |

**我们模型已经是架构级压缩的 KV**：48 层里 36 层是 **GDN（线性注意力）**——递归状态 O(1)，与序列长度无关；12 层是 **QSA**——indexer 从全上下文里选 2048 个 token。所以"上下文"在 GDN 里被压成常数大小的状态、在 QSA 里被压成稀疏选择。

进一步可借（等都做了再上）：

| 机制 | 来源 | 说明 |
|---|---|---|
| **MLA 低秩 KV 联合压缩** | DeepSeek-V2/V3 | 把 KV 压成低秩 latent，只缓存 latent |
| **DSA 稀疏注意力 / CSA / HCA** | DeepSeek-V3.2-Exp、V4 Preview | token 维压缩 + 稀疏选择 |
| **CSA2 跨层复用** | DeepSeek-V4.1-Flash（arXiv:2609.19969） | 复用前层的 global KV 与 Top-K 索引，Full/Reindex/Reuse 三模式 |
| **TurboQuant** | ICLR 2026 | 3–4 bit 近无损、与模型无关 |
| **OSCAR** | 2026 | 可部署 INT2，100K 上下文 8× 显存↓、7.83× 吞吐 |
| **XQuant / KVAlign** | arXiv:2510.11236 / Berkeley CS262A | 跨层 KV 共享（量化后相邻层更相似）+ 轻量自蒸馏 |
| 工程件 | FlashMLA / DeepGEMM / DeepEP / Engram | DeepSeek 开源工具链 |

---

## 9. 扩散/图像生成那边能借什么

大部分不可借：TASQ / FourTune / HierarchicalPrune / TMP 都是**压缩扩散模型本身**，与"压固定权重"不同构。唯一可借的想法：

- **按阶段/位置分配不同精度**（TASQ：一份最高精度缓冲 + LSB 掩码，逐层/逐去噪步选有效精度）。映射到我们：**逐层/逐角色的混合位宽分配**（例如 routed 专家 W4、shared 专家 W8——文献 RotaryQuant 正是这么做的，理由 shared 专家激活峰度更高）。这是我们唯一还没榨的权重侧杠杆，收益个位数百分比。

---

## 10. 尚未探索的权重侧杠杆（已知收益有限）

1. **逐层/逐角色混合位宽**：按敏感度分配（routed 专家 W4、shared 专家 W8、dense W8）。
2. **PLE 表量化**：95 GiB 的表，每 token 只随机读 16 行；可量化到 W8/W4 减盘上占用（对速度无影响）。
3. **0 阶熵编码**：−18%（见 §3），GPU 端变长解码复杂。
4. **TCQ**（QTIP 式）：逼近香农界，最多再 −15%。

以上合计上限约 **25~30%**，但都要新产物 + 新内核。相对"host 常驻 + DMA 直填 + arena 命中"带来的 25~50× 提速，属于第二优先级。

---

## 11. 【重要】ninfer-dev 已实现同构方案，且 Q4 产物已在盘上

排查工具链时发现：`$DATA/ninfer-dev/src/targets/qwen4_exp/` **就是我们本方案的完整 C++ 参考实现**，且已实测：

| 资产 | 位置 | 说明 |
|---|---|---|
| **Q4G64 专家 arena** | `$LINGXI_ASSETS/qwen4_exp.expert-arena.q4g64.bin`（**60.25 GiB**） | 从合格 BF16 arena 量化而来，已验证 |
| BF16 专家 arena | `$LINGXI_ASSETS/qwen4_exp.expert-arena.bin`（225.4 GiB） | 量化前镜像 |
| arena 格式 Q4EARN1 | `impl/kernels/moe_expert_arena.h` | 头部 72B + 索引 + blob；**每 blob 自带格式字**，已支持 BF16/Q4/Q5/Q6/W8 混合精度 |
| 零拷贝直读 | `impl/kernels/qwen4_moe_launcher.cu` C8 修复 | 只读 mmap + `cudaHostRegisterReadOnly`；日志 `arena_pinned=1` |
| 槽位 LRU / copier pool / Volta 低比特核 | `moe_expert_cache.cuh` / `q4_copier_pool.cpp` / `moe_expert_lowbit_volta.cuh` | 全套 |
| 量化器 | `tools/quantize_expert_arena.py` → `tools/convert/common/quantize.py::quantize_and_encode` | 与 27b/35b nvfp4/w4a4 同一条量化路径 |
| **实测性能** | `LINGXI_CHANGES.md` C8 | 单流 decode **21~26 tok/s**（即 ~38–48 ms/token） |

**归属（已在 ninfer 源码内标注）**：`moe_expert_arena.h` 首行写明 "设计参考 Strata (Niko1221/Strata, **MIT**)" —— pinned.cu 的 shared-file arena、expert_cache.hpp 的常驻表/槽位、DETAILS.md 的 `--mmap-experts/--resident-experts`。**未复用其 Q4 CPU 计算路径**（灵犀专家核全在 GPU，命中/未命中的舍入完全一致）。

**本仓库（Rust 灵犀）的验证结果**（`tools/golden/q4_arena_check.py`）：

```
arena: 60.25 GiB  magic=Q4EARN1 ver=1 layers=48 experts=512 n_in=2560 n_ff=640 align=4096
blob: gu 1,740,800 B + dn 870,400 B = 2,611,200 B  (BF16 为 19,660,800 B → 7.53x)
layer0 专家对拍（Q4G64 vs BF16）:
  expert 0/1/17/255/511: gu 相对误差 0.115~0.124, dn 相对误差 0.108~0.113
```

**结论：方案第 1 步（量化）与第 2 步（arena）已完成且有验证**；Rust 灵犀要做的是**读取该 arena + 复刻 pin/LRU/copier + Volta 低比特核**，而不是从零做量化。


