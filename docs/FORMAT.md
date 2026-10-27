# FORMAT.md — 模型产物与二进制格式手册

> 2026-10-19。目的：让**没有参与过本项目的人**能把一个新模型转换进来、并理解引擎消费的
> 每一份二进制。所有公式均与代码逐字核对过（引用文件+行号）；发现与实现不符请以代码为准并改本文件。
>
> 读者画像：做模型转换的人读 – + ；改内核的人读  + ；排障的人读 。

---

## 1. `.ninfer` 产物容器（主产物 / MTP 产物同一格式）

```
| magic 8B "NINFER\0\x02" | json_bytes u64 LE | JSON directory | pad → 4096 | payload |
```

- JSON 目录 = `{"identity":{"model_id","weights_id"},"objects":[...]}`。
- 每个 tensor 对象的 key 集合**恰好**为 `{name,kind,shape,format,layout,offset,bytes}`；
  resource 对象为 `{name,kind:"resource",encoding:"raw-bytes-v1",offset,bytes}`。
- `offset` 相对 **payload 起点**（= `align_up(16 + json_bytes, 4096)`）；tensor 按 256B 对齐、
  resource 按 1B（`crates/lingxi-artifact/src/lib.rs` L4-L28）。
- **打开时的硬校验**（`Artifact::open`，同文件 L206-255）：
  - `bytes` 必须严格等于 `lingxi_ids::try_encoded_size(format, layout, shape)`（不等即拒）；
  - 对象按 offset 单调不重叠、各自按 alignment 对齐、`offset+bytes ≤ payload 长`；
  - format/layout 字符串必须在注册表内（ 的两个枚举）。
- Python 同款实现：`tools/convert/hf2lingxi.py` L64-L118（`NinferWriter` / `read_ninfer`）。

## 2. 张量命名规范（canonical 名字 → 角色）

名字解析**只发生在** `lingxi_model::catalog_from_artifact`（`crates/lingxi-model/src/lib.rs`
L160-175 → `role_of` L79-156）；产物本身不存角色，角色由名字后缀推导。

| 名字模式 | TensorRole |
|---|---|
| `text/token_embedding` | TokenEmbedding |
| `text/output_head` | OutputHead |
| `*/attn_gr/hc_norm`、`*/mlp_gr/hc_norm` | HyperConnectionNorm |
| `*/attn_gr/block_inject`、`*/mlp_gr/block_inject` | HyperConnectionInject |
| `*/attn_gr/mix_down`、`*/attn_gr/mix_up`、`*/mlp_gr/...` | HyperConnectionMix |
| `*/attention/query_key_gate_value` | AttentionProjection |
| `*/attention/output` | AttentionOutput |
| `*/attention/indexer/index_qk_proj` | AttentionIndexerProjection |
| `*/attention/query_norm`、`key_norm`、`*/indexer/q_layernorm`、`k_layernorm` | AttentionNorm |
| `*/gdn/norm` | LinearAttentionNorm |
| `*/gdn/a_log`、`dt_bias`、`convolution` | LinearAttentionState |
| 其它含 `/gdn/` | LinearAttentionProjection |
| `*/moe/router_shared_gate` | MoeRouter |
| `*/moe/routed_gate_up` | MoeRoutedGateUp |
| `*/moe/routed_down` | MoeRoutedDown |
| `*/moe/shared_gate_up` | MoeSharedGateUp |
| `*/moe/shared_down` | MoeSharedDown |
| `mtp/fc_embedding`、`mtp/fc_hidden` | MtpStemProjection |
| `mtp/embedding_norm`、`mtp/hidden_norm` | MtpStemNorm |
| `text/ple/*norm*` / 其它 `text/ple/*` | PleNorm / PleProjection |

**层类型**：`i % 4 == 3` 的层是 QSA（稀疏 full-attention，带 `attention/` + `attention/indexer/`），
其余是 GDN（linear-attention，带 `gdn/`）。trunk 用 `/moe/`、MTP 内部用 `/mlp/`，
MoE 角色只按后缀判定。PLE 只有 layer 1 有（全局 `text/ple/`，6 个张量）。

## 3. 数值格式（TensorFormat / TensorLayout）

枚举定义在 `crates/lingxi-ids/src/lib.rs` L14-129（语义照搬 ninfer 注册表）。

### 定长格式（`contiguous-le-v1`：行主序小端直排）

| 枚举 | 名字串 | 字节/元素 |
|---|---|---|
| `Bf16` | `BF16` | 2 |
| `Fp32` | `FP32` | 4 |
| `I32` | `I32` | 4 |

### 分组量化格式（`row-split-k128-v1`，仅 rank-2 `(n, k)`）

公共几何：`k_pad = align_up(k, 128)`，`groups = k_pad / group`；三个平面依次：

1. **base 平面** @0：每行 `groups × group/2` 字节（Q4；Q3 是 `groups × group*3/8`，W8 是
   `groups × group`）。**每字节：低 nibble = 组内偶元素、高 nibble = 奇元素**（int8 二补码）。
2. **high 平面**（仅 Q5/Q6；Q4 为 0 字节）：每 group `(bits-4)×group/8` 字节；
   `high_offset = align_up(base_bytes, 256)`。
3. **scale 平面**：`scale_offset = high_offset + align_up(high_bytes, 256)`，
   **每行 groups 个 fp16**（2B/组）；总字节 = `scale_offset + n × groups × 2`。

| 枚举 | 名字串 | (bits, group, qmin, qmax) |
|---|---|---|
| `Q3G128F16S` | `Q3G128_F16S` | (3, 128, -4, 3) |
| `Q4G64F16S` | `Q4G64_F16S` | (4, 64, -8, 7) |
| `Q5G64F16S` | `Q5G64_F16S` | (5, 64, -16, 15) |
| `Q6G64F16S` | `Q6G64_F16S` | (6, 64, -32, 31) |
| `W8G32F16S` | `W8G32_F16S` | (8, 32, -127, 127) |

**Q4G64F16S 精确公式**：`total = align_up(n × k_pad / 2, 256) + n × (k_pad/64) × 2`。
解码：`value = ((nib ^ 8) - 8) × scale`（host 参考 `expert_arena.rs::decode_q4g64_f16s` L425-452；
GPU 契约 `dequant_ops.cu` 文件头 L1-7）。

### 其它布局

- `blockscale-k16-m128x4-v1`（仅 `Nvfp4`）：code 平面 `n×k/2` 字节 @0；scale 平面
  @ `align_up(code,256)`，长 `n×k/16 + 4`。
- `row-scale-v1`（仅 `Fp8E4m3fnRowBf16s`）：`align_up(n×k,256) + n×2`。

所有 layout 的 `alignment() = 256`；`PLANE_ALIGNMENT=256`、`K_ALIGNMENT=128`。

## 4. 专家 arena（`Q4EARN1`）

文件：`crates/lingxi-artifact/src/expert_arena.rs`（只读解析；**写入器在 ninfer-dev**
`tools/convert/common/quantize.py` + `tools/artifact/layouts.py`）。

### 文件头（72B 固定，`<8s7i4x4q`）

| 偏移 | 字段 | 说明 |
|---|---|---|
| @0 | magic 8B | `Q4EARN1\0` |
| @8 | i32 version | 必须 = 1 |
| @12/@16/@20/@24 | n_layers / n_experts / n_in / n_ff | |
| @28 | i32 slot_align | data 区对齐（实测 4096） |
| @32 | i32 flags | 仅 bit0 `FLAG_BLOB_FORMATS`：置位时每个 ref 的 format 生效，否则全部按 BF16Raw |
| @40/@48/@56 | i64 data_offset / routed_index_off / shared_index_off | |
| @64 | i64 data_bytes | |

布局：`header | routed index [n_layers×n_experts × 16B] | shared index [n_layers × 16B]
| pad → slot_align | data`。`ArenaBlobRef = {i64 offset(相对 data 起点), i32 bytes, i32 format}`；
**offset < 0 = 该专家缺席**；format 码：0=BF16、1=Q4G64、2=Q5G64、3=Q6G64、4=W8G32
（⇒ 同一 arena 允许按专家混精度——P1 之后引擎侧核已支持 Q4G64，混精度档见 PLAN_NEXT ）。

### 单个 blob 内部

- routed blob = `[gate_up][down]`：gate_up `[2×n_ff, n_in]`、down `[n_in, n_ff]`，
  各按  的 row-split-k128 布局排；**`gu_bytes + down_bytes == ref.bytes`**（校验强制）。
- 段几何（`matrix_layout`）：scale 平面偏移 = `align_up(base_bytes,256) [+ align_up(high,256)]`；
  本模型实测：gu [1280,2560] → payload 1,740,800 B；dn [2560,640] → 870,400 B；
  **routed blob = 2,611,200 B ≈ 2.49 MiB**。
- 设备侧取址：`base = slots + slot×blob + seg_off; codes = base; scales = base + scale_off`
  （`q4_tc.cu` L71-73）。

## 5. PLE 表（三档自动探测）

`PleTable::open`（`crates/lingxi-model/src/plet.rs` L256-290）按序探测：

1. **FP8(E4M3)**：sidecar `<path>.scale` 恰为 4B f32；每行 `row_elems` 字节，行共享一个 scale。
2. **IQ4_NL**：`<path>.iq4nl` 存在；每行 `row_elems/32 × 18` 字节，每 18B 块 = 2B fp16 d +
   16B nibble（**低半→元素 j、高半→元素 j+16**，split-half）；码本
   `[-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]`。
3. **bf16**：每行 `row_elems × 2` 字节 u16 LE。

表尺寸 = 320,001,536 行 × 160 元素（ngram 哈希取行）。启动选档顺序（`main.rs` L5877-5903）：
`LINGXI_PLE_TABLE` env → `/dev/shm/*.iq4nl.bin` → `/dev/shm/*.fp8.bin` → 机械盘 bf16（响亮横幅）。
配套工具：`tools/convert/quantize_ple_iq4nl.py`（bf16→IQ4_NL）、`tools/convert/extract_ple_fp8.py`。

## 6. MTP 产物

- 主产物**不含** `mtp/*` 张量；MTP 是**独立 `.ninfer`**（同容器），27 个 `mtp/*` 张量 ≈ 4.9 GiB。
- `lingxi-bench mtp-pack <src> <dst>`：从大产物**原样抽取** `mtp/` 前缀张量写小产物（不重编码）。
- ⚠️ **已知坑**：历史 mtp-pack 产物带两个 bug（stem norm 漏 wplus `+1`；qkv 行序坏）——
  **必须用 `tools/convert/hf2lingxi.py mtp` 从 safetensors 重新生成**（两道自检会各抓一个）。

## 7. 转换流程（当前状态与步骤）

`tools/convert/hf2lingxi.py`：
- `mtp` 子命令**可用**：取张量（跨分片，统一 bf16 位形）→ 27 项 `MTP_MAP` 变换
  （raw / **plus1** / **qkv** / cat / flat）→ `NinferWriter` 打包（全 BF16 +
  contiguous-le-v1，magic+JSON+4096 对齐）。
- **plus1（wplus 还原）**：该模型族 safetensors 把 RMSNorm gamma 存成 `γ−1`；转换时
  **所有真正的 RMSNorm gamma 必须 +1**（f32 里做、round-half-even 舍回 bf16）。名单 =
  层内 `q_norm/k_norm/hc_norm/indexer.*_layernorm` + MTP stem norm（漏一个 = MTP 接受率
  崩到 0 或首 token 漂移，判据：**gamma 必须全为正**，`check_gamma` L174-181）。
- **qkv 真置换**：产物 `query_key_gate_value` 行序 = `q(6144) | k(512) | gate(6144) | v(512)`，
  q/gate 按头去交织。自检 = **输出行多重集必须等于 `[q;k;v]` 拼接**（逐行哈希会被重复行骗过）。
- `trunk` 子命令**未实现**（TODO：48 层 Q4 专家 blob 布局逆向、`experts.bin` 索引、
  PLE 分片）——主产物目前靠 ninfer-dev 产线转换。

## 7.A 命名与兼容策略（重要）

**容器层我们没有改，也不打算改**：magic 仍是 `NINFER\0\x02`，`lingxi-artifact` 与原格式
逐字节兼容（`crates/lingxi-artifact/src/lib.rs` L3 复用铁律）。我们的主产物就是 ninfer-dev
产线转换的，本引擎是它的兼容读取方——**继续用 NINFER 容器可以白嫖 ninfer 工具链的读取/校验**。

ninfer 引擎读不了我们的产物，**不是容器问题，是内容层**：角色表只认本文  的 qwen4 命名
（`text/ple/*`、`gdn/*`、MTP 27 项）、wplus +1 约定、PLE 表寻址都是 lingxi 自己的约定。
所以分层命名：

- **容器**：NINFER v2（与上游互操作）；
- **内容约定**："lingxi 模型目录约定"（= 本文 //）。
- **文件后缀**：上游纯 ninfer 产物用 `.ninfer`；按 lingxi 内容约定生成/读取的产物用
  `.lingxi`。读取器按 magic 识别、不依赖后缀；`.lingxi` 只是让人和工具一眼知道
  "容器是 NINFER v2，内容是 lingxi 约定"。现有 253 GB 主产物沿用 `.ninfer` 名字，
  不做盘上迁移。

**何时真改名**：当我们改动容器本身时——候选触发点：Q4EARN1 arena 引用进 JSON 目录、
按专家混精度的格式码扩展、自定义 layout。届时 bump magic 到 `\x03` 并冠名为
`LINGXI\x00\x01`，同时保留 NINFER v2 读取路径作回退。在那之前改名只会切断互操作，无收益。

## 8. 验证工具链（转换后必跑）

| 工具 | 作用 |
|---|---|
| `tools/golden/q4_arena_check.py` | arena blob vs BF16 产物逐元素对拍 |
| `crates/lingxi-artifact/tests/expert_arena_live.rs` | arena 解析真机测试 |
| `hf2lingxi.py --verify-against` | 与已知产物逐位比对 |
| `tests/regression/run.sh` | 引擎端首 token 判据（11751/332/220） |
| `tools/golden/prefill_equiv.sh` | 长 prompt 首 token 等价（332/95835/198 基线） |
