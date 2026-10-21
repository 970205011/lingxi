# 灵犀引擎提速 —— 交接文档

> 写于 2026-10-15。承接本会话（目标：**单流 decode 提到 200 tok/s**）。
> 与之配套的**详细证据档**是同目录的 `decode-speed-profile.md`（§3.1–§3.64 + §4 路线图 + §7 失败登记）——
> 本文是"读完就能接手"的入口，细节一律回溯到那份。

***

## 0. 一句话现状

**目标未达成。** 当前单流 **p10 ≈ 26.9 ms/token（≈37 tok/s；p50 29.1 ms ≈ 34 tok/s）**；首 token 恒 **11751**（判据绿）。
本会话把 **t=2 前向砍掉 49%**、t=8 砍掉 21%、启动 40.1 s → **4.4 s**；
2026-10-16 又交付两件：**① arena 锁页按介质判定（p10 +28%、p50 +290%）；② 查明并修好 MTP 0/n 的根因 —— 接受率 0/64 → 58/64 = 0.906。**

> **⭐⭐ MTP 0/n 的真正根因（已定位、已复现、已修复）**
>
> **`$LINGXI_ASSETS/qwen4_exp.mtp.ninfer` 里 `mtp/embedding_norm` 与 `mtp/hidden_norm`
> 漏了 wplus 的 `+1`。**
>
> 这个模型族的 safetensors 把 **RMSNorm 的 gamma 存成 `γ−1`**，ninfer 转换的 **`wplus` 步骤负责 +1 还原成 γ**；
> 但转换器的 +1 名单**漏掉了 role 为 `MtpStemNorm` 的这两个张量**（层内的 `AttentionNorm`/`HyperConnectionNorm` 都加对了）。
> 于是 MTP 的 **stem** 拿 `γ−1` 当 γ 用：**符号翻、量级差 ~3 倍** ⇒ 整条 MTP 算错 ⇒ 草稿全错（0/64）。
>
> **判据（一眼可辨，不必跑模型）**：gamma 必须全为正。修复前
> `hidden_norm` mean=**−0.3284**（84% 为负）、`embedding_norm` mean=**−0.7642**（**100% 为负**）；
> 补 `+1` 后变成 **+0.6703（0.3% 负）/ +0.2358（0% 负）** ⇒ 立刻是正常 gamma。
>
> **修复效果（两个 bug 叠加）**：`mtp-accept` 从 **0/64 = 0.000** →
> 只修 ① 得 **58/64 = 0.906** → ①+② 全修得 **64/64 = 1.000**
> （多 prompt 复核：`The capital of France is` 96/96、`1 2 3 4 5 6 7 8 9 10` 96/96、
> `def fibonacci(n):` 85/96 = 0.885、`Write a short poem...` 78/96 = 0.812 ⇒ 实际 0.81~1.00）。
>
> **⚠️ 第二个 bug（同一产物）**：`mtp/layer/attention/query_key_gate_value` 的**行序也错了** ——
> 它的行序**不是** `[q_proj;k_proj;v_proj]` 的真置换：在 q 段里插了一个 k 块
> ⇒ **丢了 256 行 q、重复了 256 行 k**（q 的 256 个输出维用的是 k 的权重）。
> 判据：**"行多重集必须等于 `[q;k;v]`"** —— 注意**逐行哈希反查抓不到它**（重复行照样每行都能命中，
> 我一开始就被这个骗过去了）。正确行序 = `q(6144)|k(512)|gate(6144)|v(512)`，q/gate 按头去交织
> （`q 段行 h*256+d ← q_proj[h*512+d]`，`gate 段 ← q_proj[h*512+256+d]`），
> **已用主干产物（golden 金标准，layer 3）逐行验证**。
>
> **⇒ 结论**：**不要用 `mtp-pack` 抽的那份产物**（两个 bug 都在里面）。
> 用仓库里的转换器从原始 safetensors 重新生成：
> ```
> python3 tools/convert/hf2lingxi.py mtp \
>     --src $MODELS/nvfp4/Qwen3.8-Flash-Next \
>     --out $LINGXI_ASSETS/qwen4_exp.mtp.fixed.ninfer --strict
> ```
> （自带两道自检：**gamma 必须全为正**、**qkv 必须是真置换**。两条自检正好各抓一个 bug。）
> 产物：`$LINGXI_ASSETS/qwen4_exp.mtp.fixed.ninfer`。旧的那份 `$LINGXI_ASSETS/qwen4_exp.mtp.ninfer`
> **保留未动**（未删除、未覆盖）。
>
> **为什么之前所有白盒都"通过"却查不出来**：三个 oracle（stem/attn/GR）以及所有对拍都**用同一份产物的权重**
> 去比 Rust —— **是循环论证**，权重本身错了永远测不出来；而主干侧没有 stem norm，所以主干判据也照不到。
>
> **⚠️ 现有可用产物**：`$LINGXI_ASSETS/qwen4_exp.mtp.fixed.ninfer`（= 原产物 + 2 个 stem norm 的 +1）。
> `$LINGXI_ASSETS/qwen4_exp.mtp.ninfer` **仍是坏的**（未删除，也未覆盖；要换请显式改名）。
> 重建脚本见 `tools/golden/mtp_fix_stem_norm.py`（幂等），完整 HF→ninfer 打包映射见 §5.5。
>
> **⇒ 但仍不是"能用"**：`mtp-accept` 只量**接受率**；要拿到吞吐，引擎里还必须有
> **"草稿 k 个 → 一次主干前向验证 → 接受前缀"的投机环路**（lingxi 目前**没有**实现，这是下一步）。

***

## 1. 验收判据（改任何东西前后都要过）

| 项       | 值                                                                                                                                                                 |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 首 token | **11751**                                                                                                                                                         |
| 固定流贪心序列 | `13, 561, 6511, 314, 9564, 369, 19241, 13`                                                                                                                        |
| 计时尺子    | `tools/golden/bench.sh <tag> <tokens> [R]`，**以 p10 为头条**（p50 只作参考）                                                                                                |
| 构建      | `CARGO_TARGET_DIR=/dev/shm/lingxi-target cargo build --release -p lingxi-bench`，**只看退出码**；成功后必须 `cp /dev/shm/lingxi-target/release/lingxi-bench bin/lingxi-bench` |

> ⚠️ 别用 `cargo build | grep | head && cp` —— `head` 退出码恒 0，会把**编译失败**吞掉并复制旧二进制（本会话真踩过，整轮结论作废）。

### 1.1 ★ 速度现状（2026-10-16 实测更正）

⚠️ **先纠正一个口径坑**：本文件原写的 "p10 ≈ 24.8 ms（40 tok/s）" **是"专家 arena 整段锁页"口径**
（`bench.sh` 默认 `LINGXI_ARENA_PIN_GIB=64`）。把 pin 关掉（守 §2 铁律 1）后**会慢 28%、而且方差爆炸**：

| 配置（`bench.sh`，200 token × 2 轮） | p10 | tok/s | p50 | p50 tok/s | 离散 |
| --- | --- | --- | --- | --- | --- |
| **`PIN_GIB=61`（整段 arena 锁页）** | **26,738 µs** | **37.4** | **29,002 µs** | **34.5** | 27% |
| `PIN_GIB=0`（默认，分开源） | 34,441 µs | 29.0 | 113,967 µs | **8.8** | **139%** ⚠️ |
| `PIN_GIB=0` + `LINGXI_ARENA_FIXED_EXPERTS=1` | 24,938 µs | 40.1 | — | — | 2.7% |
| `PIN_GIB=0` + `LINGXI_ARENA_NOH2D=1`（不真拷） | 24,220 µs | 41.3 | — | — | 2.8% |

**结论（三条）**：

1. **专家 blob 的 H2D 就是那 ~28% 与 139% 方差的来源**：`NOH2D=1`（只记槽不真拷）与 `FIXED_EXPERTS=1`
   两条独立消融都落到 ~24.5 ms / 41 tok/s、离散 2.7% ⇒ 与 host 路由、槽簿记无关，**就是拷贝本身**。
2. **分开源（pageable）退化成"驱动 staging + 同步"**，所以又慢又抖；**整段锁页后 p10 37.4 / p50 34.5 tok/s**。
   命中率实测只有 **85%**（且**只驻留 4442/7000 槽，没满**）⇒ 全是**首次触碰的 compulsory miss**
   （~76 个/token × 2.6 MB ≈ 198 MB/token 过 PCIe），**不是容量淘汰**（加大槽数无效，与 §3.16.1 一致）。
3. **🐞 新发现真 bug：`LINGXI_ARENA_PIN_GIB` 只支持"整段"**。给中间值（如 16）会在 decode 第 15 个 token
   崩 `arena blob H2D: DriverError(CUDA_ERROR_INVALID_VALUE)` —— **同一 mmap 只注册一部分时，
   落到注册区间之外的拷贝源会被拒**。⇒ 现状是"整段 pin 或完全不 pin，别给中间值"（代码应显式拦掉）。

> ✅ **已落地（提交 `efb50d9`，判据绿）**：铁律 1 改成**按介质判定**（代码读 `/proc/mounts` 硬判）——
> arena 在 **RAM（tmpfs/ramfs）** ⇒ 允许且**默认整段 pin**（`LINGXI_ARENA_PIN_GIB` 不设 = 整段）；
> 非 RAM 介质 ⇒ **硬禁**（要关 pin 显式 `=0`）；**只接受「整段或 0」**，中间值启动即报错。
> 实测（`bench.sh 200×2`，新默认）：**p10 26,934 µs = 37.1 tok/s，p50 29,124 µs = 34.3 tok/s**
> —— 相对旧默认（29.0 / **8.8**）是 **p10 +28%、p50 +290%**。首 token 11751 判据仍绿。

> **⇒ 当前基线（2026-10-16，新默认）**：**p10 26.9 ms = 37.1 tok/s**（p50 29.1 ms = 34.3 tok/s）。
> 距 200 tok/s 还差 5.4×，后续只能靠：**① 内核吞吐（Q4/专家 GEMV 的 2× 空间）+ ② 投机解码**。

***

## 2. 环境与常用命令

* 引擎在**宿主仓库检出** = **测试容器内的同一挂载**。一切容器操作走
  `pct exec <CTID> -- bash -lc '...'`。

* GPU：Tesla V100-PCIE-32GB（sm\_70），**容器内无 ncu/nvprof**（所以块账靠 SKIP/PROF/KTS 反推）。

* 产物：`$MODELS/qwen4_exp-artifact/qwen4_exp.ninfer`（主干）；
  MTP 小产物 `"$LINGXI_ASSETS/qwen4_exp.mtp.ninfer"`（27 张量 / 4.86 GiB，由 `mtp-pack` 生成）。

* 常用 env：

```bash
export LINGXI_EXPERT_ARENA=/dev/shm/qwen4_exp.expert-arena.q4g64.bin
export LINGXI_ARENA_SLOTS=7000                      # 设备槽数（17 GiB），9200 无增益
export LINGXI_Q4_CACHE=/dev/shm/lingxi-q4cache      # W4 权重缓存：启动 40.1s → 4.4s
# ★ 2026-10-17（P15-㉜）：双卡时 layer-split 已是**引擎默认**（lo=24，自动关 CUDA Graph），
#    不再需要手设 LINGXI_LAYER_SPLIT/LINGXI_NO_GRAPHS。依据：真实中文语料 20.7 → 54.1 tok/s
#    （+161%，每卡 arena 减半 ⇒ 专家 miss 崩塌）；bench 英文高命中语境 −7%。
#    回归对照用 LINGXI_LAYER_SPLIT=0 显式关。
# ★ 2026-10-17（P15-㉝）：长 prompt 单卡自动 prefill 分块（LINGXI_PREFILL_CHUNK，默认 2048，
#    单卡逐位认证过）；split 下默认 512（GPU1 显存余量）。
# ★ 2026-10-17（P15-㉳）：**PLE 表默认自动选 /dev/shm 官方 FP8 表**（51.2GB 全内存，启动横幅可确认）。
#    三级解析：LINGXI_PLE_TABLE 显式 > /dev/shm/qwen4_exp.ple-table.fp8.bin > lingxi-assets bf16（兜底，
#    靠 LINGXI_PLE_PREWARM 顺序预热）。格式自动识别（.scale sidecar ⇒ FP8）。
#    抽取工具 /tmp/extract_fp8.py（测试容器内），源 = 模型自带 NVFP4 版 plefp8 safetensors 分片；
#    官方另有 IQ4_NL 26.8GiB / bf16 102.4GB 两档（ninfer-dev ple_reader.h 注释），想要更省可再抽。
```

### PLE 表（2026-10-17 ㉳定盘）

| 档位 | 大小 | 说明 |
| --- | --- | --- |
| **FP8（默认）** | 51.2GB | 官方 NVFP4 版模型自带；已抽平铺到测试容器的 /dev/shm（全内存，随机读≈µs 级） |
| IQ4_NL | 26.8GiB | 官方默认档（行 90B，4bit 码本+block fp16 scale）；本机未抽取（要自写量化器或找官方表） |
| BF16 | 102.4GB | 母表，lingxi-assets 机械盘；仅作对照/兜底，靠预热 |

### 铁律（血的教训，别破）

1. **机械盘（6T）单块持续读写 ≤15 分钟**，必须切块 + 歇息。
   **专家 arena 的 pin 按「介质」判定**（2026-10-16 起由代码硬判，见 §1.1）：
   * arena 在 **RAM（tmpfs/ramfs）** ⇒ ✅ 允许且**默认整段 pin**（不设 `LINGXI_ARENA_PIN_GIB` 即整段）——
     分开源会慢 28% 且 p50 掉到 8.8 tok/s；
   * arena 在**机械盘**上 ⇒ ❌ **硬禁**（曾导致盘固件假死、整机重启、`/dev/shm` 全丢），
     要显式 `LINGXI_ARENA_PIN_GIB=0` 才能起来；
   * **只支持「整段或 0」**：给中间值（如 16）启动即报错——部分 pin 会崩
     `CUDA_ERROR_INVALID_VALUE`（同一 mmap 部分注册，区间外源被拒）。
2. **禁止删除任何文件**（无用户逐条确认）。
3. 进程主权三规矩：杀进程后**进程名单 + 显卡显存双查**确认死透；远程杀进程用**方括号技巧**防自杀；
   **一台机器同一时刻只准一个会话指挥显卡**。
4. 忘掉 `LINGXI_EXPERT_ARENA` 会掉进 `MmapExpertSource` 回退路 —— 实测**慢 15\~20 倍**
   （曾让 decode 变成 1.8 s/token，被我误判成"代码回归"）。现已**改为硬报错**，要用得显式
   `LINGXI_ALLOW_MMAP_FALLBACK=1`。

***

## 3. 本会话已落地（都已提交推送，判据绿）

| 改动                    | 收益                                   | 说明/提交                                                                                  |
| --------------------- | ------------------------------------ | -------------------------------------------------------------------------------------- |
| **cuBLAS 请出前向热路径**    | **t=2: 74.7→37.9 ms（−49%）**；t=8 −21% | `gemm_bf16_nt` 在 `n=2` 极窄 N 上选到病态内核（单层 50→500 µs）。改成自家瘦长核                              |
| **W4 三档（稠密投影 + 输出头）** | 单流 **−10%**（p10 27.6→24.8 ms）        | 数值影响标尺（§3.48）证明**贪心序列逐位不变**才转正                                                         |
| **W4 权重按张量名缓存**       | 启动 **40.1 → 4.4 s**                  | 关键：按"内容哈希"寻址会白读 6 GiB bf16（省的是**读**不是算）                                                |
| **Rust 警告清零**         | 149 → **0**                          | 用户明确要求                                                                                 |
| **探针登记表 + 启动校验**      | ——                                   | `crates/lingxi-model/src/probes.rs`：拼错开关/幽灵开关/危险组合直接报错或警告；`lingxi-bench probes` 打印全部状态 |
| **回退路禁静默**            | ——                                   | `MmapExpertSource` 慢 15\~20×，改硬报错                                                      |

***

## 4. 失败/否决登记（别再试，详见 §7）

GR 的 W4 化（整机 12 s/token）· `Q4_ROW4` 多行/warp · **专家 Q4 GEMV 的 RPW 2/4/8（单调更慢）** ·
`GR_FUSE`（+7.5%）· `ARENA_SLOTS=9200`（噪声内）· `SCHED_SPIN`/`NO_GRAPHS`（都零收益）·
双卡 Pipeline Parallel（把 GPU1 驱动搞崩）· PLE 放机械盘 · DeepGEMM（只支持 sm90+）·
**SKIP 口径的系统性高估**（剥 MoE 会连带剥掉栅栏）· **KTS 口径 +26% 膨胀**（插同步=排空流水）·
**以** **`synchronize()`** **结尾的区段会被"排空"污染**（哪怕是纯墙钟口径）。

***

## 5. 剩余工作（按优先级）

### 5.1 ✅【已修】`m>8` 崩溃（**已提交，判据绿**）

```
Error: skinny-row 只支持 1≤m≤8（给到 23）
```

**崩因更正（别记错）**：崩的不是 "MTP 的 MoE 慢路径"，而是 **MoE router 那段瘦长 GEMM**
（`moe.rs:214`，`use_map && rows>1`）。它**在 `pair_path` 判据之前**无条件执行 ⇒
**主干 prefill 一样会走到**（堆栈 `prefill_with_hyper` 就是铁证：23 token 的 prompt ⇒ rows=23）。
> 原文档把锅甩给 `src=None` 是错的：`src` 只决定**专家段**走哪条路，router 段与它无关。

**修法（已落地）**：在调用处按 ≤8 行切片（照 `proj_t1` 的 `SKINNY_MAX_M` 循环）。
⚠️ 切片时**输出偏移必须按 bf16 = 2 字节**：`m.dev + done*num_experts*2`
（与 `gpu.rs::proj_t1` 的 `yp + done*n*2` 同口径）。草稿版曾误写成 `*4`，会把第 2 片起的
logits 写错位 —— 已一并修正。

**验证**：23 token 与 38 token 的长 prompt 都能跑完（此前直接 `Error` 退出）。

### 5.2【高价值】MTP 的"真值强核对"（**已定性：草稿头本身不对**）

`mtp-accept` 现在会在 prompt 段逐行核对：行 i 喂 `(h_i, emb(ids[i+1]))`，**语义上应预测** **`ids[i+2]`**
（这是**已知真值**，不需要主干 rollout）⇒ 一次前向可得 \~prompt\_len 个独立样本。

**已扩到 30+ 样本（3 次独立跑，全部 0）**：

| prompt | 样本数 | 命中 |
| --- | --- | --- |
| 短（3 token） | 3 | **0/3** |
| 长（23 token） | 21 | **0/21** |
| 长（38 token） | 36 | **0/36** |

草稿一律是 76402/190569/228343 这类**超大 id**（而主干偏好 13/561/22 这类小 id）。
⇒ **草稿头本身就不对**（不只是 decode 段喂法），且**在已知 token 上也全错**。

> ⚠️ **跑长 prompt 的内存坑**：`LINGXI_ARENA_SLOTS=7000`（17 GiB）在 38 token prompt 下会在
> `[load] 层 47/48` 撞 `CUDA_ERROR_OUT_OF_MEMORY`（48 层常驻 + arena 顶到 32 GiB 边）。
> 做**正确性**实验时降到 `LINGXI_ARENA_SLOTS=4096` 即可（只影响专家缓存命中率，不影响数值）。

### 5.3 块级：只剩 \~3%

实测否掉了原本估的 7%（GR 减核无交集：`norm` 是生产者、`inject` 是 `[4,10240]` 的整行归约、
`mix_mean` 是逐元素）与 12%（router 搬设备只省 host 的 `expf`+选择，**每层那次 sync 省不掉**）。
⇒ 块级**已接近局部最优**。

### 5.4 【量级级】1b 设备端槽管理（用户已原则批准，会破坏位级复现）

现状：**每层一次** **`synchronize()`**（48 次/token）由 **host 侧专家槽缓存**强制 ⇒ 流水被压得很浅。
根除要**把槽管理搬设备**（设备端 LRU + 槽命中读显存、miss 过 PCIe 读锁页 host arena）。
建议拆 4 块、每块单独验证提交：
① 设备端 router（softmax+top-k）+ **对拍 harness 量专家 id 不匹配率**（⚠️ CUDA `expf` 与 Rust `f32::exp` 不同 libm，**必须先量再接线**）；
② 设备端槽表；③ miss 走映射读；④ 接线生产路径。

### 5.5【量级级】投机解码（唯一能到 200 的路）

#### ★★ 2026-10-16 定案：**0/n 的根因 = MTP 产物漏了 stem norm 的 wplus `+1`**（已修，接受率 0 → 0.906）

> ⚠️ **先撤回上一版结论**：上一版写"MTP 与主干不同源（借了 BASE 的 MTP）"——**这条是错的**。
> 实测：主干产物 `qwen4_exp.ninfer` 的张量**逐位等于** `RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored`；
> MTP 产物的 26/27 张量**逐位等于** `Qwen3.8-Flash-Next` 的 `mtp.*`。
> 而这两份的差异（专家权重 rel≈5.5e-2）**不是** 0/n 的原因。真正的原因是下面这一条。

**根因**：`$LINGXI_ASSETS/qwen4_exp.mtp.ninfer` 的 `mtp/embedding_norm` 与 `mtp/hidden_norm`
**漏了 wplus 的 `+1`**。

- 本模型族 safetensors 把 **RMSNorm 的 gamma 存成 `γ−1`**；ninfer 转换的 **`wplus`** 步骤 +1 还原成 γ。
- 转换器的 +1 名单**漏了 role = `MtpStemNorm` 的两个张量**（层内的 `AttentionNorm` /
  `HyperConnectionNorm` 都加对了 ⇒ 所以只有 stem 错）。
- 于是 stem 用 `γ−1` 当 γ：**符号翻 + 量级差 ~3 倍** ⇒ MTP 整条算错 ⇒ 草稿全错。

**判据（不用跑模型，看一眼分布即可）**：gamma 必须全为正。

| 张量 | 修复前 mean / 负值占比 | 补 +1 后 |
| --- | --- | --- |
| `mtp/embedding_norm` | **−0.7642 / 100.0%** | **+0.2358 / 0.0%** |
| `mtp/hidden_norm` | **−0.3284 / 84.3%** | **+0.6703 / 0.3%** |

**修复效果（`mtp-accept`，两个 bug 逐个修）**：`0/64 = 0.000` →
修 ① 后 **`58/64 = 0.906`** → 再修 ② 后 **`64/64 = 1.000`**
（多 prompt：96/96、96/96、0.885、0.812 ⇒ 实际 0.81~1.00）。
prompt 段真值强核对 `0/21` → `8/21`，且**偏移扫描只在 k=+0 命中** ⇒ 语义正确。

**② 第二个 bug：`mtp/layer/attention/query_key_gate_value` 的行序错了。**
它的行序**不是** `[q_proj;k_proj;v_proj]` 的真置换：在 q 段里插了一个 k 块
⇒ **丢 256 行 q、重复 256 行 k**。正确行序：`q(6144)|k(512)|gate(6144)|v(512)`，
q/gate 按头去交织（`q 段行 h*256+d ← q_proj[h*512+d]`，`gate 段 ← q_proj[h*512+256+d]`），
k/v 整段搬 —— **用主干产物（golden 金标准）layer 3 逐行验证过**。
⚠️ **判据是"行多重集 == `[q;k;v]`"**：逐行哈希反查**抓不到**它（重复行照样每行都能命中）。

**产物与工具**：
- **正确产物**：用仓库里的转换器重新生成（不要用 `mtp-pack` 抽的那份）：
  ```
  python3 tools/convert/hf2lingxi.py mtp \
      --src $MODELS/nvfp4/Qwen3.8-Flash-Next \
      --out $LINGXI_ASSETS/qwen4_exp.mtp.fixed.ninfer --strict
  ```
  自带两道自检（**gamma 全为正** / **qkv 真置换**），正好各抓一个 bug。
- 旧的坏产物 `$LINGXI_ASSETS/qwen4_exp.mtp.ninfer` **保留未动**。
- `tools/golden/mtp_fix_stem_norm.py`：只修 bug ① 的快捷补丁（幂等）。

**⚠️ 为什么以前查不出来（方法论教训）**：三个 oracle（stem/attn/GR）+ 全部对拍都是
**"拿同一份产物的权重去比 Rust"** ⇒ **循环论证**：权重本身错，oracle 永远绿。
凡"比实现"的对拍都要先问一句：**参照物是独立的吗？**（本次真正的突破口是
"用 HF 原始权重当独立参照" + "gamma 必须全为正"的先验。）

**附：HF safetensors → ninfer MTP 产物的完整映射（已逐位验证 27/27 复现原产物）**

| ninfer 名 | HF 名 | 变换 |
| --- | --- | --- |
| `mtp/fc_embedding` / `mtp/fc_hidden` | `mtp.fc_embedding` / `mtp.fc_hidden` | 原样 |
| `mtp/embedding_norm` / `mtp/hidden_norm` | `mtp.pre_fc_norm_embedding/hidden` | **+1（原来漏了）** |
| `mtp/mixer/{hc_norm,mix_down,mix_up}` | `mtp.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}` | `hc_norm` **+1** |
| `mtp/layer/attention/query_key_gate_value` | `mtp.layers.0.self_attn.{q,k,v}_proj` | **按行置换**（见下） |
| `mtp/layer/attention/{query_norm,key_norm}` | `...self_attn.{q_norm,k_norm}` | **+1** |
| `mtp/layer/attention/output` | `...self_attn.o_proj` | 原样 |
| `mtp/layer/attention/indexer/{index_qk_proj,q_layernorm,k_layernorm}` | `...indexer.*` | 两个 `*_layernorm` **+1** |
| `mtp/layer/{attn_gr,mlp_gr}/{hc_norm,mix_down,mix_up,block_inject}` | `...{attn,mlp}_hyper_connection.*` | `hc_norm` **+1**，其余原样 |
| `mtp/layer/mlp/router_shared_gate` | `mlp.gate.weight` ⊕ `mlp.shared_expert_gate.weight` | 行拼接 `[512+1, 2560]` |
| `mtp/layer/mlp/routed_gate_up` / `routed_down` | `mlp.experts.gate_up_proj` / `down_proj` | 展平 |
| `mtp/layer/mlp/shared_gate_up` | `mlp.shared_expert.{gate_proj,up_proj}` | 行拼接 |
| `mtp/layer/mlp/shared_down` | `mlp.shared_expert.down_proj` | 原样 |

- **+1 名单**（即 HF 里存 `γ−1` 的"层内 norm"）：`mixer/hc_norm`、`query_norm`、`key_norm`、
  `indexer/{q,k}_layernorm`、`{attn,mlp}_gr/hc_norm`、**以及两个 stem norm**（`embedding_norm`/`hidden_norm`）。
- **`query_key_gate_value` 的行置换**：产物 13312 行 = `[q_proj(12288); k_proj(512); v_proj(512)]` 的逐位双射置换；
  规则是"**按头去交织**"：产物行 `h*256+d`（q 段）→ `src[h*512+d]`，gate 段 → `src[h*512+256+d]`；
  k/v 段整段搬到 `[6144:6656)` / `[12800:13312)`。（脚本里用**反查哈希**一次性推出整张置换表，不靠猜规则。）

**⇒ 下一步（通往 200 tok/s 的关键）**：接受率 0.906 只是**接受率**；要有吞吐，必须在引擎里实现
**投机环路**：`草稿 k 个 → 一次主干前向验证 → 接受前缀 → 提交`。lingxi **目前没有**这条环路
（`mtp-accept` 只测接受率），这是下一步的主要工作。

MTP 移植进度：

| 部件                             | 状态                                        |
| ------------------------------ | ----------------------------------------- |
| 权重 → 独立小产物 `mtp-pack`          | ✅                                         |
| `stem`                         | ✅ 白盒 **3.9e-3**（= bf16 精度）                |
| 注意力块                           | ✅ 白盒 **1.0e-2**（并钉死四段排布为 `q\|k\|gate\|v`） |
| GR 块                           | ✅ 白盒 **3.4e-3**                           |
| GR 结构（block\_inject 有/无、注入源）   | ✅ 逐条对上 Strata `mtp.cpp`                   |
| 层与主干同构（`13312 = 2×6144+2×512`） | ✅ 交叉验证                                    |
| **序列级喂法**                      | ❌ **唯一未验**                                |

**已实测否掉 9 条**：token 对齐(next/last) · h 身份(hyper/mixed) · indexer/稀疏(`QSA_HOST`) ·
同形状张量交换(fc/qk) · 基座不同源（`diff_artifacts.py`：876 张量逐位同） ·
慢路径 bf16 累加精度（`NO_PAIR_MOE=1` 主干序列逐位不变） · `pre_fc_norm_hidden` 两变体（本模型数值等价） ·
**末步 mixer 用** **`mtp/mixer`** **还是全局** **`hyper_connection_mixer`（`LINGXI_MTP_GMIX`：两种都 0/32）** ·
**§3.63 的"尺度假设"**（见下）。

**✅ 本轮否掉第 9 条 —— §3.63 说"喂头的 hidden 尺度不对（3.2 vs 0.5~0.7，差 4~6 倍）"，实测不成立。**
零代码补测（`LINGXI_MTP_H=mixed` 会把**主干自己的全局 mixer 输出**经 `hc_repeat` 后打印成 `h 输入`，
其 std 即主干喂头的 `mixed` 尺度）：

| 量 | std | max&#124;x&#124; |
| --- | --- | --- |
| **主干** `mixed`（喂 lm_head 的 hidden） | **≈ 4.19** | 196 |
| MTP `mixer mixed`（喂 lm_head 的 hidden） | ≈ 3.24 | 21 |

⇒ 主干喂头的 hidden **比 MTP 还大**，两者同量级 ⇒ **当初那个"3.2 vs 0.5~0.7（差 4~6 倍）"
是拿 MTP 的 mixer **输出**去比残差栈**输入**，苹果比橘子**。** scale 线索作废**：
MTP 的 hidden **量级正常、内容错**（= 典型的"静默错"：某处读错张量/位置/掩码但不炸量级）。

**判据（§3.60）**：结合 §3.48（W4 化稠密+输出头序列也不变）⇒ **本模型对小数值扰动高度免疫**
⇒ **0/64 必然是"范畴性"错误**（读错张量 / 位置·掩码·缓存语义错 / 索引错位），不是精度问题。

**⇒ 下一步的搜索方向**：不要试精度微调，找"必然造成 100% 错"的结构错误。
**注意：prompt 段（批量喂 T 行）与 decode 段（t=1 逐行）两条路都 0** ⇒ 根因是**两条路共用的
"单 token 前向 + h/emb 配对"**，而不是 prompt 专属的掩码/批量接线。
最可疑仍是**序列级喂法/配对**：草稿层 KV/位置的推进节奏、`(h, tok)` 到底配哪一行的哪一个
（注意 `main.rs:1426` 的 `pending` 取的是 `drafts_p[prompt_len-1]`，而按 5.2 的语义，
首个 decode 步该要的可能是 `drafts_p[prompt_len-2]` —— **这个 off-by-one 只影响 1 个比较，解释不了 0/n**，
但仍应修）。**下一条最该做的是白盒**：`mtp-dump` 已备好（落 layer_in/out、attn_mixed_a/attn_out、
全部注意力权重、cos/sin），用 numpy 复算**该层的 MoE 段**（`mlp/*` 接线 + `top_k=10` 断言值），
把"层接线错"钉死或排除。

**资产**：`mtp-pack` / `mtp-load` / `mtp-accept`（含真值强核对 + `LINGXI_MTP_*` 开关）/ `mtp-dump`，
以及 `tools/golden/mtp_{stem,attn,gr}_oracle.py` 三个白盒对拍脚本。

**✅ 本轮又排除 3 条（对照 Strata `src/core/mtp.cpp` 权威源，宿主 `$MODELS/tmp/strata`）**：

1. **配对语义对得上**。Strata 的 `draft_first(spec, ss.R, tok, p-1)`（`generate.cpp:10655`，`tok` 取
   `tokens[pos+1]`，`cell = pos`）⇒ 配对 = `(h_pos, emb(tok_{pos+1}))` 在 **位置 pos** ⇒ 预测 `tok_{pos+2}`。
   与我们的 harness `(h_i, emb(ids[i+1])) @ i → ids[i+2]` **完全一致**。
2. **prompt 段 KV 语义对得上**。Strata 的 `prefill(R_rows, next_tokens, n, cell0)`（`mtp.cpp:1116`）
   逐行 `Rin_[i]=R_rows[i]`、`tok[i]=next_tokens[i]`、`pos=cell0+i`，且 `full=false` ⇒ **prompt 段只填 K/V、不出草稿**；
   与我们 `mtp_step(t=prompt_len)`（填 K/V + 出全部行草稿）**数值上等价**（因果注意力下批量 = 逐行）。
3. **stem 维度对得上**。Strata `mtp.cpp:684-693`：`fc_embedding`/`fc_hidden` 都是 `[N,N]`（`native_mmvq(..., N, N, nc)`），
   hidden 分支按 **每 stream 一份** 投影；`hidden_norm` 是 `[HC*N]=10240` 整段 norm（`hnorm_stream_` 关时，L686-688）。
   ⇒ 与产物 `fc_hidden [2560,2560]` + `hidden_norm [10240]` + 我方实现**逐条一致**（不是维度错）。

⇒ 单 token 前向的**语义与接线**已基本被 Strata 权威源钉死。剩下的空间很窄：

- **（i）MTP 的 MoE 段** —— 唯一还没白盒过的整段（`mlp/*` 接线 + `top_k=10`）。
- **（ii）`mtp-accept` 自身的 decode 计数口径**：`pending` 取 `drafts_p[T-1]`（`main.rs:1426`）。
  按 Strata 口径 `drafts_p[T-1]` 预测 `tok_{T+1}`；step0 的 `tok_next` 是否等于 `tok_{T+1}` 取决于
  `decode_token_with_hyper` 会不会重复产出首个 token。**这条只影响 1 个样本，解释不了 0/n**，但应核准。
- 注：0/36 是**在 §5.1 的切片修好之后**跑的（长 prompt 修前直接崩、出不来样本），故结论成立。

**✅ 本轮再排除 4 条（并列在此，避免重试）**：

4. **管线（CUDAGraph）键冲突** —— 怀疑"MTP 的 `layer_step(layer=0)` 与主干 layer 0 共用
   `graph::run_cached` 的 key 1/2/7，会重放对方的图"。零代码证伪：`LINGXI_NO_GRAPHS=1` 与默认
   **草稿逐位相同**（step1 draft=35116、0/21、0/4 全同）⇒ 图与本 bug 无关。
5. **27 个 MTP 张量的形状** —— 全部与主干 QSA 层同构且自洽（`fc_*[N,N]`、`hidden_norm[10240]`、
   `query_key_gate_value[13312,2560]`、`output[2560,6144]`、`router_shared_gate[513,2560]`、
   `routed_gate_up[655360,2560]`、`routed_down[1310720,640]`、`shared_*` 两份、GR 各 4 份）
   ⇒ **无形状/维度异常**（排除"转换把某个张量打包错"这一类）。
6. **草稿头是独立小头** —— Strata 的 `dhead_` 只是**共享头的行子集**（`--mtp-q4` 加速用，
   `full_head_env()` 时用全头）⇒ 用主干的 `text/output_head` **是对的**。
7. **因果掩码** —— `qsa.rs` 的 `_causal_host` 根本没被使用，因果性由 `build_indexer_mask`
   按**位置**构造（`j < len+i+1`）⇒ MTP 的 prompt 段掩码正确。

**⇒ 收敛判断（重要）**：MTP 单步用的是**与主干同一份代码**（`layer_step`），而主干 48 层端到端
能精确复现 golden ⇒ **共享代码路径不可能只在 MTP 上错**。因此白盒（同产物复算）**预期会通过**；
真正还能出错的只剩三类：
**(a) MTP 权重的内容本身**（需要**独立来源**比对，同产物复算是循环论证）；
**(b) 喂给 MTP 的 `h`/`tok`/KV 的来源与节奏**（`prefill_with_hyper` / `decode_token_with_hyper` 返回的
残差栈到底对应哪个位置）；
**(c) 主干 QSA 层从未在"生产路径"下走过的那条分支**（MTP 的 MoE 走 `src=None` 慢路径）。

***

## 6. 文件地图

| 文件                                  | 管什么                                                                                                                                                                              |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/lingxi-bench/src/main.rs`   | 全部子命令。`mtp-accept` 在 **1203–1458**；`mtp-dump` 1459–1584；`mtp-load` 1585–1679（⚠️ **别把这两个的** **`ml.mixer`** **搞混**）；`trunk-ends` 2218；`load_endpoints` 3162；`build_mtp_layer` 3100 |
| `crates/lingxi-model/src/probes.rs` | 探针登记表 + 启动校验（新增/改开关都要登记）                                                                                                                                                         |
| `crates/lingxi-model/src/moe.rs`    | MoE。**分派在** **`:402`**（`fast_path = pair_path \|\| (src.is_some() && rows==1 && ...)`）⇒ `src=None`（MTP）恒走慢路径                                                                     |
| `crates/lingxi-model/src/mtp.rs`    | MTP：`shape_of` / `load`（stem）/ `stem`                                                                                                                                            |
| `crates/lingxi-model/src/gpu.rs`    | 上传 + `proj_t1` 分派（`use_row8`/`use_skinny_row` 的 `k≥1024` 判据）                                                                                                                     |
| `crates/lingxi-model/src/gr.rs`     | GR（超连接）：`gated_residual_forward[_into]`                                                                                                                                          |
| `crates/lingxi-model/src/trunk.rs`  | 主干装配：`embedding → hc 广播 → 48 层 → 全局 mixer → lm_head`                                                                                                                             |
| `docs/decode-speed-profile.md`      | **详细证据档**（§3.x 逐条 + §4 路线图 + §7 失败登记）                                                                                                                                            |

***

## 7. 判活与排障速查

* **判活要多维度**：单一 `GPU util = 0` 会误判（加载期 GPU 静默、CPU 在反量化/缺页）。
  看：显存涨 / anon 涨 / `read_bytes` 涨 / 是否出现 `session is up`。

* **仪器伪影**：KTS 段值 ≈ 该段 GPU 时间 **+ 排空/重填**（四段和×48 比真值高 26%）；SKIP 剥块会
  连带剥掉栅栏 ⇒ 用"**剥块后 Δ(t=2−t=1) 是否塌缩**"做免疫判据。**以** **`bench.sh`** **的 p10 为准**。

* **W4 会打坏非增量 oracle 子命令**（它们会读到 1 元素 stub）⇒ W4 默认只在
  `bench.sh`/`run_full.sh` 里开，没翻全局默认。

***

## 8. 建议的第一步（按顺序）

1. **修 5.1**（`m>8` 切片）—— 小改，解锁长 prompt。
2. **跑 5.2 强核对到 30+ 样本** —— 一句话定性"草稿头对不对"。
3. 若草稿头**对** ⇒ 全力查 decode 段的 KV/位置/索引（含上面那个 off-by-one）；
   若草稿头**也错** ⇒ 回到白盒，把该层的 **MoE 段**用同样手法对拍（`mlp/*` 接线 + `top_k=10` 这个**断言值**）。
4. 并行可做 **5.4 的第①块**（设备端 router + 不匹配率 harness），它不依赖 MTP。

