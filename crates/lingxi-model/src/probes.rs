//! **引擎环境开关的唯一权威登记表**。
//!
//! 为什么要它：这些开关是长出来的，命名又很短（`kt`/`kt2`/`kts`/`prof`），
//! 结果**已经实际误导过我们三次**（都是靠人肉回忆才发现）：
//!
//! 1. `LINGXI_ROUTER_NOSYNC` —— 文档里写着、**代码里根本不存在**，一整组对比实验作废；
//! 2. `LINGXI_KT` × `LINGXI_DENSE_Q4` —— 组合起来会 **panic**（探针引用空的 bf16 权重），
//!    连着三次「跑完没有输出」都白跑了；
//! 3. `LINGXI_SKIP` 剥块 —— **系统性高估**（去掉 MoE 也顺带去掉了每层栅栏，其它块跟着变快），
//!    差点把那 36 ms 记在 MoE 头上（实际是 GR，见 docs §3.38）。
//!
//! 本文件提供三件事：
//! - [`PROBES`]：一张表，登记每个开关的**类型 / 默认 / 用途 / 与谁耦合**；
//! - [`check_env`]：启动时校验——**拼错的开关会被点名**（以前是静默忽略），
//!   已知的危险组合**直接报错退出**（而不是崩在内核深处）；
//! - [`dump`]：把所有开关的当前状态打出来，便于把实验配置写进档案。

/// 开关的类别。分类决定了「它能不能进生产默认路径」。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// 真功能开关（改行为，可能是生产配置的一部分）。
    Feature,
    /// A/B 回退开关：用来「关掉某个优化做对照」。**跑完应清理或明确保留**。
    Ab,
    /// 诊断探针：会改时序（插同步）甚至改数学（`SKIP`/`NOH2D`/`FIXED_EXPERTS`）。
    /// **绝不能出现在生产路径里**。
    Probe,
    /// 已知失效/幽灵开关：代码里已无实现，只留文档痕迹。**用到它必然得出错结论**。
    Ghost,
}

pub struct Probe {
    pub name: &'static str,
    pub kind: Kind,
    /// 默认是否开启（只对 Feature/Probe 有意义）。
    pub default_on: bool,
    pub purpose: &'static str,
    /// 与其它开关的耦合/冲突；空串表示无已知耦合。
    pub coupling: &'static str,
}

/// ⚠️ 新增开关必须登记在这里，否则 `check_env` 会把它当拼写错误点名。
pub const PROBES: &[Probe] = &[
    // ── 路径 / 设备（Feature） ──
    Probe { name: "LINGXI_DEVICE", kind: Kind::Feature, default_on: false,
        purpose: "选 CUDA 设备号（默认 0）", coupling: "GPU1(44:00.0) 在 x8 链路，跨卡重活会触发 PCIe CmpltTO" },
    Probe { name: "LINGXI_ARTIFACT", kind: Kind::Feature, default_on: false,
        purpose: "产物路径（测试用）", coupling: "" },
    Probe { name: "LINGXI_EXPERT_ARENA", kind: Kind::Feature, default_on: false,
        purpose: "专家 arena 文件路径（**必须放内存盘 /dev/shm**；放机械盘会让 slot miss 变成随机读）",
        coupling: "与 ARENA_SLOTS / ARENA_PIN_GIB 联动" },
    Probe { name: "LINGXI_PLE_TABLE", kind: Kind::Feature, default_on: false,
        purpose: "PLE 大表路径", coupling: "" },
    Probe { name: "LINGXI_ALLOW_MMAP_FALLBACK", kind: Kind::Feature, default_on: false,
        purpose: "**显式**允许在没有 arena 时退回 MmapExpertSource（host LRU）。实测 decode 慢 15~20 倍，只做对照时开",
        coupling: "与 LINGXI_EXPERT_ARENA 互补：给了 arena 就不看它" },
    Probe { name: "LINGXI_Q4_CACHE", kind: Kind::Feature, default_on: false,
        purpose: "W4 在线量化产物的**内容寻址磁盘缓存目录**（首次算+落盘，之后只读；省掉 ~27 s 启动量化，读的字节还少 4 倍）",
        coupling: "与 DENSE_Q4/_QSA/HEAD_Q4 联动；不设则每进程重算" },

    // ── 容量 / 缓存（Feature） ──
    Probe { name: "LINGXI_ARENA_SLOTS", kind: Kind::Feature, default_on: false,
        purpose: "设备专家槽数（默认 7000 ≈ 17 GiB）",
        coupling: "加到 9500 实测无收益 §3.16.1；与显存预算（DEV_EXPERT_GB）联动" },
    Probe { name: "LINGXI_ARENA_PIN_GIB", kind: Kind::Feature, default_on: false,
        purpose: "arena 锁页：**不设 = 在 RAM(tmpfs/ramfs) 上默认整段 pin**；=0 显式关闭；**只支持整段或 0**（给中间值启动即报错）",
        coupling: "⚠️ pin 的安全性**按介质判定**（`arena_source` 读 /proc/mounts 硬判）：非 RAM 介质上一律**禁止**——注册会逼内核把整段 arena major-fault 进内存，对机械盘持续顺序读会触发固件假死（2026-10-14 事故）。另：部分 pin 已被实测证明会崩（CUDA_ERROR_INVALID_VALUE）" },
    Probe { name: "LINGXI_PLE_CACHE_MB", kind: Kind::Feature, default_on: false,
        purpose: "PLE 行缓存上限（默认 1024 MB）", coupling: "加到 16 GB 只 +5%（系统页缓存已覆盖）" },
    Probe { name: "LINGXI_DEV_EXPERT_GB", kind: Kind::Feature, default_on: false,
        purpose: "专家常驻显存预算（GiB）", coupling: "与 ARENA_SLOTS 联动" },

    // ── 量化路线（Feature） ──
    Probe { name: "LINGXI_DENSE_Q4", kind: Kind::Feature, default_on: true,
        purpose: "稠密权重走 W4G64（GDN/QSA 投影）。**p10 −6.6%，序列逐位不变**（§3.49）；`=0` 关",
        coupling: "**与 LINGXI_KT 冲突（会 panic）**；非增量 oracle 子命令不要开（会读到 1 元素 stub）" },
    Probe { name: "LINGXI_DENSE_Q4_QSA", kind: Kind::Feature, default_on: true,
        purpose: "QSA 投影也走 W4G64。与 DENSE_Q4 同开（§3.49）；`=0` 关",
        coupling: "历史记过「长序列 ILLEGAL_ADDRESS」，2026-10-15 400 token 复测无错，暂视为已消" },
    Probe { name: "LINGXI_HEAD_Q4", kind: Kind::Feature, default_on: true,
        purpose: "lm_head 走 W4G64（1.18 GiB→0.31 GiB）。**再 −3.7%，序列逐位不变**（§3.48/§3.49）；`=0` 关",
        coupling: "输出头对长链路质量最敏感 ⇒ 8 token 之外仍需长链验证" },
    Probe { name: "LINGXI_QSA_DENSE", kind: Kind::Feature, default_on: false,
        purpose: "跳块稀疏注意力回退密集掩码路径（㊵ 对拍开关；默认走稀疏，只算 keep 选中块的点积）",
        coupling: "稀疏路径与被跳过位置的 dense 贡献（恰 0）逐位等价，含退化整行无可见位情形" },
    Probe { name: "LINGXI_QSA_PAGED", kind: Kind::Feature, default_on: false,
        purpose: "P2 页粒度部分驻留：KV 块表间接 + host 锁页块池 + LRU 下沉/预取（常驻窗口 LINGXI_QSA_WIN_TOKENS，默认 32K）",
        coupling: "要求 gather 路径（bf16 KV + 2 的幂 compress + head_dim=256），与 K8V4/dense 互斥" },
    Probe { name: "LINGXI_QSA_WIN_TOKENS", kind: Kind::Feature, default_on: false,
        purpose: "P2 分页的 GPU 常驻窗口（token 数，ctok 对齐；默认 32768）", coupling: "" },
    Probe { name: "LINGXI_QSA_PAGED_NAN", kind: Kind::Feature, default_on: false,
        purpose: "P2 分页诊断：gather 后 NaN 计数 + admit/tab 状态打印", coupling: "" },
    Probe { name: "LINGXI_QSA_PAGED_CHECK", kind: Kind::Feature, default_on: false,
        purpose: "P2 分页诊断：预取后回读 GPU 槽与 host 池逐字节比对", coupling: "" },
    Probe { name: "LINGXI_QSA_PAGED_SINK_NOP", kind: Kind::Feature, default_on: false,
        purpose: "P2 分页判决开关：跳过下沉拷贝（池留垃圾），区分'读路径走池'与'破坏在别处'", coupling: "" },
    Probe { name: "LINGXI_QSA_ADMIT_BATCH", kind: Kind::Feature, default_on: false,
        purpose: "P2 分页 prefill 行批 admit 的批大小（行数，默认 64）", coupling: "" },
    Probe { name: "LINGXI_SPARSE_EXPERTS", kind: Kind::Feature, default_on: false,
        purpose: "走稀疏专家（arena）而非全量权重", coupling: "" },

    // ── 调度器（Feature） ──
    Probe { name: "LINGXI_SCHED_SPIN", kind: Kind::Feature, default_on: false,
        purpose: "调度器自旋等待", coupling: "" },
    Probe { name: "LINGXI_DISABLE_EVENT_TRACKING", kind: Kind::Feature, default_on: false,
        purpose: "关 cudarc 的 event 追踪（省 CPU）", coupling: "" },
    Probe { name: "LINGXI_FIXED_IDS", kind: Kind::Feature, default_on: false,
        purpose: "decode 时强制固定 token 序列（**A/B 必须开**：否则采样差异会改变路由）", coupling: "" },

    // ── A/B 回退（Ab）——排除法用，跑完要清理 ──
    Probe { name: "LINGXI_NO_GRAPHS", kind: Kind::Ab, default_on: false,
        purpose: "关 CUDA Graph 做对照", coupling: "" },
    Probe { name: "LINGXI_MTP_REPLAY_1BY1", kind: Kind::Ab, default_on: false,
        purpose: "**MTP 重放退回逐行**（默认走批量重放）：逐行 = `keep` 次 t=1 的 MTP 前向、权重读 `keep` 遍；\
                  批量 = 一次 `t=keep` 前向、权重只读一遍。用来量「批量重放」省了多少",
        coupling: "与 LINGXI_MTP_SPEC 一起用；只影响 MTP 状态刷新速度，**不改主干输出**" },
    Probe { name: "LINGXI_MTP_HNORM", kind: Kind::Feature, default_on: false,
        purpose: "MTP stem 的 hidden 分支 **norm 分组**：默认 = `stream`（逐 hyper-connection stream，每 2560 一段，\
                  各归一化 + 各用 gamma 的一片）；`=pooled` = 整段 HC*N(10240) 一起 RMS。\
                  取自 Strata `--mtp-hnorm`（`src/core/mtp.cpp:686`，其注释明写 stream 才是 \
                  **llama.cpp qwen4exp MTP 图的做法**）；同机同模型的「顺子」开着它。\
                  实测（交错 A/B，n=200）：stream 3.67/6 接受、41.0 tok/s ｜ pooled 3.47/6、38.6；\
                  位 1 条件接受率 0.79 vs 0.68",
        coupling: "只改 MTP 草稿质量，**不该改主干输出**；若生成序列变了说明我们在草稿上耦合了主干路径，要查" },
    Probe { name: "LINGXI_SUFFIX_DRAFT", kind: Kind::Feature, default_on: false,
        purpose: "**后缀查表草稿**（= Strata 的 `--suffix-draft`，值 = 最小匹配长度，**默认 0 = 关**）。\
                  零权重、零 GPU、微秒级：把「历史后缀 + MTP 已产出的草稿」当查询，\
                  从上下文里找最长的更早出现，把它后面跟着的 token 当草稿补进窗口。\
                  ⚠️ **查表本身实测是对的**（`LINGXI_SUFFIX_TRACE=1`：提议 `[369,11751,13]`、\
                  `[369,44299,13,561,6511,314]` 与真值逐 token 相同 ⇒ 3/3、3/4、6/6），\
                  但**挂错位置**了：接受是前缀式的，挂在 MTP 链后面 ⇒ 只有整条链先被接受才轮得到它，\
                  实测「提议 7 枚、接受 0 枚」且白付行钱（47.4→44.3 tok/s）。\
                  要生效必须先做参考实现的 `DraftPolicy::choose`（匹配够长时**用 lookup 顶掉整个 MTP 窗**），\
                  见 PLAN.md §P9.5c",
        coupling: "只改「提哪些草稿」，被采纳的 token 仍由主干验证 ⇒ 不改对外输出" },
    Probe { name: "LINGXI_SUFFIX_CHAIN", kind: Kind::Probe, default_on: false,
        purpose: "后缀查表一次最多补几枚（0 = 把 MTP 被闸门砍掉的空位全补上）",
        coupling: "与 LINGXI_SUFFIX_DRAFT 配合" },
    Probe { name: "LINGXI_SUFFIX_TRACE", kind: Kind::Probe, default_on: false,
        purpose: "逐次打印后缀查表的提议内容与真值（回答「是提议错了还是这段文本本来就查不到」）",
        coupling: "只打印，不改数学" },
    Probe { name: "LINGXI_ARENA_LAYERS_HI", kind: Kind::Feature, default_on: false,
        purpose: "只给 `< HI` 的层分槽（默认 48=全部）。配合 LINGXI_MOE_SPLIT：开了跨卡后 GPU0 的表\
                  不该再给后一半层留槽，否则一半钉位白占；配合把 PREWARM_N 调大 ⇒ 每层覆盖 83.8%→~98%",
        coupling: "与 MOE_SPLIT / PREWARM_N 联动" },
    Probe { name: "LINGXI_MOE_SPLIT", kind: Kind::Feature, default_on: false,
        purpose: "**跨卡 MoE**：把后一半层的 MoE 交给 GPU1（它只有这些层的槽位 ⇒ 每层 400+ ⇒ miss≈0），\
                  跨卡只过激活（36 KB×2/层）⇒ 直接砍掉一半的专家 miss PCIe（占整轮 43%）。\
                  位等价：GPU1 上是同一内核/同一权重字节/同一输入 ⇒ 判据仍是首 token + 固定序列",
        coupling: "与 LINGXI_MOE_SPLIT_LO / _SLOTS 联动；跨卡只能过 host（P2P 被 IOMMU 挡死）" },
    Probe { name: "LINGXI_MOE_SPLIT_LO", kind: Kind::Feature, default_on: false,
        purpose: "跨卡 MoE 的起始层号（默认 24 ⇒ 后一半层给 GPU1）", coupling: "见 LINGXI_MOE_SPLIT" },
    Probe { name: "LINGXI_MOE_SPLIT_SLOTS", kind: Kind::Feature, default_on: false,
        purpose: "GPU1 上的专家槽数（默认 11000）", coupling: "见 LINGXI_MOE_SPLIT" },
    Probe { name: "LINGXI_LAYER_SPLIT", kind: Kind::Feature, default_on: false,
        purpose: "**全量 layer-split**（值 = 切分点层号 `lo`）：`[lo, 48)` 的**整层**（稠密+注意力+\
                  MoE+跨 token 状态）都跑在 GPU1 上，残差流在边界走锁页 staging 换卡。\
                  里程碑 A 的目标是**逐位等价**（0 提速，把两卡的地基铺好）；\
                  里程碑 B 再把两块 chunk 铺到两卡做两段流水（那时才是 ~2×）。\
                  与 LINGXI_MOE_SPLIT 互斥；必须配 LINGXI_NO_GRAPHS=1（跨卡内核不能被图捕获）。",
        coupling: "LINGXI_LAYER_SPLIT_SLOTS = GPU1 的专家槽数（默认 9000）；\
                   GPU0 那份 arena 只服务 `0..lo`（自动）" },
    Probe { name: "LINGXI_LAYER_SPLIT_SLOTS", kind: Kind::Feature, default_on: false,
        purpose: "全量 layer-split 时 GPU1 的专家槽数（默认 9000）", coupling: "见 LINGXI_LAYER_SPLIT" },
    Probe { name: "LINGXI_GDN_REPLAY_ZERO", kind: Kind::Probe, default_on: false,
        purpose: "重放时把 `qkv/z/a/b` 四个内部输入缓冲**零初始化**——判「重放读到了没被完整写过/\
                  残留的内存」是不是那处 run-to-run 竞态的原因",
        coupling: "只改分配（零填），不改算式；需配合 LINGXI_GDN_REPLAY_FAST=1" },
    Probe { name: "LINGXI_GDN_REPLAY_NOPOOL", kind: Kind::Probe, default_on: false,
        purpose: "让投机提交的**快速重放整段绕开 scratch 池**（分配直连）——判定「同长度缓冲在池里\
                  被复用」是不是那处 run-to-run 竞态的元凶",
        coupling: "只改分配策略、不改数学；需配合 LINGXI_GDN_REPLAY_FAST=1" },
    Probe { name: "LINGXI_PREWARM_GLOBAL", kind: Kind::Probe, default_on: false,
        purpose: "预热钉表改**全局 count 降序贪心**取槽（用与旧口径相同的槽数），替代\
                  「每层 top-N 一刀切」——后者的层配额是平均分的，`distinct` 少的层会把槽花在\
                  `count=1` 长尾上、`distinct` 多的层反而钉不上（实测各层 distinct 182~411）",
        coupling: "需配合 LINGXI_ARENA_PREWARM=<csv>；只改「钉哪些」不改「钉多少」" },
    Probe { name: "LINGXI_GDN_DUMP_FULL", kind: Kind::Probe, default_on: false,
        purpose: "把层 0 的**整个 `s_all`**（约 2 MB f32）按时刻（postbatch/restored/legacy/fast）\
                  写成 `<值>/<tag>.bin`（只留第 0 轮）——用来逐字节 diff 判\
                  「老路提交后 == 快照恢复后」是「真没推进」还是「那几个元素恰好不动」",
        coupling: "只写文件；每次约 2 MB D2H，只对层 0" },
    Probe { name: "LINGXI_GDN_DUMP_S", kind: Kind::Probe, default_on: false,
        purpose: "提交完每层 GDN 后，把 `s_all`/`k_hist` 头 3 个值打到 stderr（定位快速/老重放的\
                  1 ulp 从第几轮、哪一层开始分叉；两条路的日志可直接 diff）",
        coupling: "只打印、极慢（每元素一次同步），只适合 n≈20 的定位跑" },
    Probe { name: "LINGXI_GDN_REPLAY_LEGACY", kind: Kind::Probe, default_on: false,
        purpose: "**回退开关**：投机提交回到「按 `t=accept` 重算 PLE+attn_GR+`gdn_step`」的老路。\
                  默认（不设本开关）走**快速重放**：只跑「卷积+递推」、跳过投影，用**批处理当时的投影字节**\
                  ⇒ 前 `accept` 行与批量逐位同源，实测 +10~12.7%（50.1→55.1 tok/s），\
                  且「全接受时提交态 == 批后态」这条硬判据**通过**（整块拷贝读+逐字节比、同二进制 3 跑全同）。\
                  本开关只为 A/B 与排障保留。",
        coupling: "需要 `spec_prepare` 分配约 5 MB 录像缓冲（gdn_qkv/z/a/b）" },
    Probe { name: "LINGXI_ARENA_SYNC_IDS", kind: Kind::Probe, default_on: false,
        purpose: "覆写跨层复用的 `ids_pin`/`slot_ids` 之前先同步（判决性诊断：判分叉是否来自这两个\
                  复用缓冲的主机侧时序）",
        coupling: "只加一次同步、不改数学；每层一次" },
    Probe { name: "LINGXI_ARENA_SYNC_H2D", kind: Kind::Probe, default_on: false,
        purpose: "每次 miss 的 blob H2D **立刻同步**（判决性诊断：把「H2D 未真正可见」这条可能一刀排除）",
        coupling: "只改同步、不改数学；会显著变慢（每 miss 一次排空）" },
    Probe { name: "LINGXI_SPEC_TRACE_ARGMAX", kind: Kind::Probe, default_on: false,
        purpose: "逐轮打印主干各行的 argmax + 草稿 + acc/keep（定位「两条配置的 token 流从哪一轮分叉」）",
        coupling: "只打印，不改数学" },
    Probe { name: "LINGXI_PREWARM_REVERSE", kind: Kind::Probe, default_on: false,
        purpose: "只**反转预热槽位映射**、不改钉住的集合（诊断：判专家 GEMV 是否依赖槽位号）",
        coupling: "仅在 LINGXI_ARENA_PREWARM 生效时有效；只改排布不改数学" },
    Probe { name: "LINGXI_NO_SCRATCH_POOL", kind: Kind::Probe, default_on: false,
        purpose: "量「整轮禁用 scratch 池」的代价（`disable_pool_forever`）。\
                  layer-split 目前无条件禁池，所以这是把「跨卡回归」拆成 **禁池 vs 跨卡** 两半的\
                  判决性实验：单卡 + 本开关 = 纯禁池代价。", coupling: "只改分配策略，不改数学" },
    Probe { name: "LINGXI_MOE_OVERLAP", kind: Kind::Feature, default_on: false,
        purpose: "**专家段 hit/miss 重叠**（PLAN §P4）：把 miss 的 blob H2D 挪到第二流上飞，\
                  compute 流先算命中组（每层 ~95%）⇒ 两边并行。收益口径：`ARENA_NOH2D` 实测\
                  这一趟值 **28 ms/轮**（40.9 → 48.4 tok/s）。只在 `rows>1` 的组列表路径生效。",
        coupling: "**必须同时关图**（`LINGXI_NO_GRAPHS=1`；跨流+事件在 capture 里语义不同）\
                   —— 只设本开关不关图时会告警并**自动退回老路径**。数学不变（位等价）。" },
    Probe { name: "LINGXI_ARENA_PIN_DMA", kind: Kind::Feature, default_on: false,
        purpose: "PLAN §P13.9b 的那条实验：再开一份**可写** `MAP_SHARED` 别名（同一份 tmpfs 页，\
                  只注册不写），用 `CU_MEMHOSTREGISTER_PORTABLE` 注册，把专家 H2D 从 zero-copy\
                  路径改成 copy-engine DMA。\
                  ❌ **实测已否决（§P13.9c）**：注册确实成功了（`arena_pinned=1`），但专家段\
                  \"每 miss 增量\" 完全没变（2.31 → 2.25 ms），tok/s 也在噪声内（40.2 → 40.9）。\
                  ⇒ 留着只作 A/B 杠杆，**别指望它提速**。",
        coupling: "需要 arena 文件**可写**（/dev/shm 上是）；默认关 = 老行为（READ_ONLY 注册）" },
    Probe { name: "LINGXI_NO_VERIFY_PIPE", kind: Kind::Feature, default_on: false,
        purpose: "回退开关：关掉 spec 验证切 chunk 时的两卡两段流水（LINGXI_LAYER_SPLIT 下），退回顺序 chunk",
        coupling: "" },
    Probe { name: "LINGXI_VERIFY_CHUNKS", kind: Kind::Feature, default_on: false,
        purpose: "把一轮验证的 t 行**切成 n 个 chunk 顺序跑**（P15 里程碑 B 第 6 步）。\
                  目的**不是提速**，而是先证明\"切分不改数学\"（QSA 的 KV append-only、GDN 行串行、\
                  MoE 每行归并仍按专家升序 ⇒ 应逐位不变）；证明之后才能把两块 chunk 铺到两卡做\
                  两段流水（那才是 ~2×）。默认 1 = 老行为。", coupling: "只改调度，不改算式" },
    Probe { name: "LINGXI_PLE_PREWARM", kind: Kind::Feature, default_on: true,
        purpose: "**PLE 表后台顺序预热**（默认开，=0 关）：开线程顺序通读 102GB 表喂 page cache。\
                  为什么默认开：PLE 行读是随机的，冷行=机械盘寻道（实测 17K prefill 从 11.3ms/行\
                  劣化到 60ms/行、单层事件间隔被拉 6.5s）；顺序预热与模型加载重叠，暖页全进程受益。",
        coupling: "纯 I/O 不改数值；抢磁盘带宽约 9 分钟（与模型加载并行）" },
    Probe { name: "LINGXI_NO_ATTN3", kind: Kind::Ab, default_on: false,
        purpose: "**禁用 attn3 行块平铺**（回退 attn2）——attn3 对拍/回归用。attn3 把 8 行并到一个 \
                  block 共享 K/V 读（流量 ÷8），每行归约次序与 attn2 逐项相同 ⇒ 输出应逐位一致。",
        coupling: "只影响 t≥8 且 head_dim=256 的注意力" },
    Probe { name: "LINGXI_PREFILL_TIMING", kind: Kind::Probe, default_on: false,
        purpose: "**prefill 分块进度计时**（driver 逐 chunk 打印「累计行数/总行数 + 累计秒 + 均 ms/行」）。\
                  只 eprintln，不改数学；长上下文 TTFT 外推的测量口径。",
        coupling: "只打印" },
    Probe { name: "LINGXI_EVPROF", kind: Kind::Probe, default_on: false,
        purpose: "**逐段真 GPU 计时**（`cuEventRecord` + `cuEventElapsedTime`，每层五段：\
                  入口+PLE+attn_GR / attn / mlp_GR / MoE / gr_write）。为什么要新做一个：\
                  `LINGXI_KTS` 的同步括号**结尾会排空流水**（量的是排空时间，偏 +25%），\
                  而 `ARENA_NOH2D` 这类**会改数学**的消融量的是另一个负载（已验证不可用）。\
                  事件只插标记、不等待 ⇒ 口径干净。跑完在 `mtp-spec` 汇总处打印最后一轮的逐段账。",
        coupling: "只加 ~1~2 µs/段的记录开销；不影响数学（不等待、不改次序）" },
    Probe { name: "LINGXI_MOE_OV_TRACE", kind: Kind::Probe, default_on: false,
        purpose: "打印前 200 层的「命中/未命中 + miss 字节数」—— 判断 miss 是**散布**（重叠有效）\
                  还是**突发**（少数层一次几十个，命中组的 GEMV 盖不住）", coupling: "只打印，不改数学" },
    Probe { name: "LINGXI_ARENA_REPIN", kind: Kind::Feature, default_on: false,
        purpose: "**在线自适应重钉**（单位 = load_hits 次数；0=关）：每窗口按「最近热度」重排每层 top-N\
                  （新进 top-N 搬进来钉住、掉出解钉还给 LRU），热度计数减半（滑动窗口）。\
                  为什么需要：钉表离线静态，实测同配置 n=80 有 46 tok/s、**n=300 掉到 29.6**\
                  —— 生成超出采集话题后覆盖率就掉 ⇒ 这条对应「应该跟着话题自动调整」",
        coupling: "复用 LINGXI_ARENA_PREWARM_N 作为每层 top-N；与 ARENA_SLOTS 要留出余量" },
    Probe { name: "LINGXI_ARENA_REPIN_CAP", kind: Kind::Feature, default_on: false,
        purpose: "在线自适应重钉时**每次最多改多少个**（默认 96）。为什么要上限：实测 take(600) + 短窗口\
                  会灌进 ~17 GB 额外 H2D，比它省下的 miss 还贵（第一版就是这么从 32.5 掉到 19.7 的）",
        coupling: "只在 LINGXI_ARENA_REPIN>0 时有效" },
    Probe { name: "LINGXI_MTP_Q4_EXPERTS", kind: Kind::Feature, default_on: false,
        purpose: "**MTP 的 512 个 routed 专家改成 Q4G64 小 arena**（bf16 5.03 GiB → Q4G64 1.34 GiB）—— \
                  给主干腾出 ~3.7 GB 专家槽位（真语料 miss 19.4% → ~11.7%）。\
                  机制：`build_mtp_layer(sparse=true)` 不上传 routed 张量，改由 `mtp_arena::MtpArena` \
                  直读（全部常驻、零 DMA、零 LRU）。**零精度风险**：MTP 只产草稿，对外 token 由主干验证决定 ⇒ \
                  改它不可能改变输出，只可能改变接受率。判据：首 token + 固定贪心序列不变",
        coupling: "与 LINGXI_ARENA_SLOTS 叠加（腾出的显存应该再换成槽位才吃到收益）" },
    Probe { name: "LINGXI_SUFFIX_CHOOSE", kind: Kind::Probe, default_on: false,
        purpose: "**`DraftPolicy::choose`**（参考实现 `spec/draft_policy.hpp` 第一条）：\
                  后缀匹配长度 ≥ 本值（token 数）时，**用 lookup 顶掉整个 MTP 窗**（不跑草稿链）。\
                  0/不设 = 行为不变。**实测（2026-10-16）在这段「首都枚举」基准文本上是负的**：\
                  8/48 轮命中、提议 31 枚、**接受 0 枚**，端到端 44.7/47.4 → 42.3/45.7 tok/s。\
                  原因：该文本是「模板 + 变槽」（`The capital of X is Y`），\
                  **对后缀查表是最不利的输入**（更早那次出现的续写里换了国家名）。\
                  要判它死活必须换 **agent/工具调用 JSON/代码改写** 语料",
        coupling: "只改「提哪些草稿」⇒ 不改对外输出；需与 LINGXI_SUFFIX_DRAFT 一起设" },
    Probe { name: "LINGXI_Q4_SLOTSM_RPW", kind: Kind::Feature, default_on: false,
        purpose: "**批量专家核**（`slotsm`，t>1 路径）的每 warp 行数：**默认 2**（实测甜点，见下）；\
                  `=1` 退回逐行（与历史逐位一致）；`=4` 实测更差。\
                  交错 A/B（各 2 轮，n=200）：RPW=1 → 44.0/46.1 tok/s、主干验证 66.8/63.8 ms；\
                  **RPW=2 → 48.2/48.5 tok/s、60.7/60.3 ms**；RPW=4 → 46.5/45.3、63.9/65.7。\
                  依据：专家段只有 ~270 GB/s（顺序路径 790），`dn` 段 k=640 每 warp 仅 ~2.5 次迭代、\
                  HBM ~600 cyc 延迟盖不住；t=1 路径的 `LINGXI_Q4_SLOTS_RPW` 早就靠同一手法解决过",
        coupling: "**位安全**（每行的列集合与累加次序不变）；与 LINGXI_Q4_SLOTS_RPW 各自独立" },
    Probe { name: "LINGXI_NO_SKINNY", kind: Kind::Ghost, default_on: false,
        purpose: "🚫 **已删除**：t>1 回退 cuBLAS 那条路整条移除（实测 t=2 64.7→405.6 ms，
            且 cuBLAS 在窄 N 上会选到病态内核，见 §3.45/§3.46）。现在 t>1 一律走自研瘦长 GEMM",
        coupling: "设了它 = 什么都没变" },
    Probe { name: "LINGXI_NO_SKINNY_ROW", kind: Kind::Ab, default_on: false,
        purpose: "瘦长 GEMM 回退旧核（并行度差 8 倍）", coupling: "" },
    Probe { name: "LINGXI_NO_Q4GEMM", kind: Kind::Ab, default_on: false,
        purpose: "t>1 的 Q4 回退「解量化成 bf16 再 GEMM」", coupling: "" },
    Probe { name: "LINGXI_NO_ROW8", kind: Kind::Ab, default_on: false,
        purpose: "bf16 GEMV 关闭「一行一 block + 128 位 load」", coupling: "" },
    Probe { name: "LINGXI_NO_PAIR_MOE", kind: Kind::Ab, default_on: false,
        purpose: "MoE 关闭 (专家,行) 对路径（**长 prompt 会 OOM**，仅调试）", coupling: "" },
    Probe { name: "LINGXI_Q4_ROW4", kind: Kind::Ab, default_on: false,
        purpose: "稠密 Q4 GEMV 回到「一 warp 4 行」做对照（**默认已翻成单行**：LUT 之后 4 行版被 shared 挤爆，见 §3.33）", coupling: "" },
    Probe { name: "LINGXI_Q4_SLOTS_RPW", kind: Kind::Ab, default_on: false,
        purpose: "专家 Q4 GEMV 的每 warp 行数（1/2/4/8）；1 = 历史逐行版、与历史逐位一致",
        coupling: "只影响 MoE 专家段；目标是 dn 段 k=640（每行仅 320 B Q4）那种窄行形状" },
    Probe { name: "LINGXI_MTP_SWAP_FC", kind: Kind::Ab, default_on: false,
        purpose: "诊断：交换 MTP stem 的 fc_embedding / fc_hidden（两者同形状 [N,N]，用来裁「产物命名搞反」）",
        coupling: "只影响 mtp-accept / mtp-load" },
    Probe { name: "LINGXI_MTP_SWAP_QK", kind: Kind::Ab, default_on: false,
        purpose: "诊断：交换 MTP 层的 query_norm / key_norm（同形状 [head_dim]）",
        coupling: "只影响 mtp-accept / mtp-load" },
    Probe { name: "LINGXI_ROUTER_NODMAP", kind: Kind::Ab, default_on: false,
        purpose: "router logits 回退 D2H 取回（对照零拷贝映射）", coupling: "" },
    Probe { name: "LINGXI_GDN_HOST", kind: Kind::Ab, default_on: false,
        purpose: "GDN 递推回退 host 参考实现（对拍用）", coupling: "" },
    Probe { name: "LINGXI_QSA_HOST", kind: Kind::Ab, default_on: false,
        purpose: "QSA indexer 选块回退 host 参考（对拍用）", coupling: "" },
    Probe { name: "LINGXI_QSA_KEEP_RADIX", kind: Kind::Feature, default_on: true,
        purpose: "QSA 选块走**流式基数选择**（每行一个 block、3 遍直方图定阈值 + 2 遍填 keep）：\
                  O(nblocks)/行、零 scores 缓冲，替代旧 rank 的 O(nblocks²) 与 t×nblocks×4B 显存",
        coupling: "`=0` 回旧 scores+rank 路径做 A/B（输出逐位相同）" },
    Probe { name: "LINGXI_QSA_FUSED_MAX", kind: Kind::Feature, default_on: false,
        purpose: "**回退杠杆**：nblocks ≤ 该值时用 `qsa_keep_fused_bf16`（一次打分落 shared）。\
                  默认 **0 = 不用**（prefill 统一走「并行打分落全局 + 只读分数选块」，32K 实测\
                  select+fill 41ms/层 vs 融合核 68ms vs radix 293~335ms）。设正值可做 A/B",
        coupling: "`=9216` 恢复 2026-10-18 之前的融合档" },
    Probe { name: "LINGXI_QSA_BIGPAR", kind: Kind::Feature, default_on: true,
        purpose: "**大 prefill 的并行直方图选块**（默认开）：1 遍 grid 并行打分落 scratch\
                  + 每行一个 block 的 3 遍直方图 + 填 keep，替代 radix 的 4 遍打分。\
                  覆盖 256K（nblocks=65536），分数缓冲走 scratch 池化复用，峰值一块",
        coupling: "`=0` 回退 radix（256K 下每层约 2.4s 的重复打分）" },
    Probe { name: "LINGXI_QSA_PROF_ALL", kind: Kind::Probe, default_on: false,
        purpose: "`[qsa-prof]` 不再只打前 40 次，而是每次 QSA 调用都打（按上下文长度看 pool/select/attn 伸缩）",
        coupling: "需同时开 `LINGXI_QSA_PROF`" },
    Probe { name: "LINGXI_ATTN_GATHER", kind: Kind::Feature, default_on: true,
        purpose: "QSA 稀疏注意力走**选中块 gather**（只遍历每行的 2048 token 列表）而不是\
                  逐 key 扫全量 Tk ⇒ prefill 注意力与上下文长度解耦",
        coupling: "`=0` 回旧 attn_sparse 路径；仅 bf16 KV（K8V4 档仍走旧路径）" },
    Probe { name: "LINGXI_ATTN_GATHER_H", kind: Kind::Feature, default_on: true,
        purpose: "gather 核的 **q-head 合并数**（默认 4）：本模型 24 q-head / 2 kv-head ⇒ 12 个\
                  q-head 共用同一份 K/V，旧版按 (row,head) 各读一遍（49GB/层 ≈ 打满 HBM）。\
                  合并 4 个后 32K 实测 attn 58~62ms/层 → **24~26ms/层**，逐位一致",
        coupling: "`=0` 回逐 head 版；smem = span×4B×NH（NH=4 时 ≈33KB，故不再往上加）" },
    Probe { name: "LINGXI_QSA_PROF", kind: Kind::Probe, default_on: false,
        purpose: "QSA 层内三段真 GPU 计时（pool / select+fill / attn），**插同步括号**\
                  ⇒ 会打乱双卡流水，只用于归因，不要在生产计时用",
        coupling: "只打印（前 400 次调用）" },
    Probe { name: "LINGXI_GR_FUSE", kind: Kind::Ab, default_on: false,
        purpose: "GR 的 5 核融成 2 核（**仅 rows==1**；历史实测整机反而慢 11%，故默认关 §3.14）", coupling: "与 gr.rs 的图捕获路径互斥语义" },

    // ── MTP 草稿头（Ab）——都只影响 mtp-accept / mtp-load ──
    // ⚠️ 这一组是 2026-10-16 补登记的：**它们早就被代码读了，却一直没进本表**
    //    （`check_env` 只在"环境里设了它"时才 warn，且只 warn ⇒ 漏登记是静默的）。
    //    现在由本文件底部的静态守卫测试兜住。
    Probe { name: "LINGXI_MTP_ALIGN", kind: Kind::Ab, default_on: false,
        purpose: "MTP 对齐诊断：`=last` 时改用最后一行对齐（与默认口径对拍）", coupling: "" },
    Probe { name: "LINGXI_MTP_GMIX", kind: Kind::Ab, default_on: false,
        purpose: "MTP 末步改用**全局** mixer（`text/hyper_connection_mixer`）而非自带的 `mtp/mixer`（§3.64 的遗留对拍开关）",
        coupling: "" },
    Probe { name: "LINGXI_MTP_H", kind: Kind::Ab, default_on: false,
        purpose: "MTP 输入 h 的来源诊断：`=mixed` 时改用 mixer 之后那一路", coupling: "" },
    Probe { name: "LINGXI_MTP_STATS", kind: Kind::Ab, default_on: false,
        purpose: "打印 MTP 三站（stem 输出 / 层输出 / mixer 输出）的均值·标准差·max|x|", coupling: "" },
    Probe { name: "LINGXI_MTP_CHAIN_DIAG", kind: Kind::Probe, default_on: false,
        purpose: "🗡 **草稿链归因**（值可取 `noh` / `nokv` / `oracle` / `hhcmp`，可逗号组合）：\
                  `noh` = 每步沿用第 0 步的 h（不更新自喂）；`nokv` = 每步回卷 MTP 的 QSA 到轮前；\
                  `oracle` = h 与 tok 都换成主干真值（**判「该不该自喂」的决定性对照**）；\
                  `hhcmp` = 逐元素对比「自喂量 o0」与「主干真 h」（cos/相对误差）",
        coupling: "⚠️ **数学作废、只为归因**；与 `LINGXI_MTP_SPEC` 一起用" },
    Probe { name: "LINGXI_SPEC_MIN_P", kind: Kind::Feature, default_on: false,
        purpose: "**草稿链的概率闸门**（值 = 阈值，0 = 关）：某一步草稿的 top-1 概率低于阈值就停链。\
                  取自参考实现 Strata `generate.cpp:9172` 的 `--spec-min-p`（`while T<S_mtp && pr[...]>=spec_min_p`）。\
                  实测（n=200 干净 A/B，叠 NO_GRAPHS）：关 → **18.2 tok/s**；0.9 → 25.6；**0.7 → 27.6（+52%）**；0.5 → 22.9。\
                  原理：prob1 ≥0.90 的草稿接受率 0.988、<0.50 只有 0.278 ⇒ 砍掉低概率深位几乎不丢接受数，\
                  但省下的是「行」—— 主干验证段实测 −39%（148.0→89.7 ms/轮）",
        coupling: "⚠️ **必须同时设 LINGXI_NO_GRAPHS=1**（截短链 × 图 会在几轮后崩；已在 check_env 里快速失败）。\
                   图对整机零成本（§8.6.2），所以不亏。默认不设 = 行为完全不变" },

    // ── 投机解码（Probe）——改的是"怎么测"，也会改时序；不进生产 ──
    Probe { name: "LINGXI_MTP_SPEC", kind: Kind::Probe, default_on: false,
        purpose: "**真实投机环路**：值 = kd（草稿数）。每轮 ①MTP 自喂草稿链 ②主干一次 t=kd+1 前向验证 ③数接受前缀 ④提交（状态回卷 + MTP 用主干 h 重放）",
        coupling: "需要 MTP 产物（`mtp-pack`）；与 talk 路径**尚未接线**（只在 mtp-accept 里）。⚠️ 会打印误导性的 `tok/s`：分母含未接线的额外开销" },
    Probe { name: "LINGXI_SPEC_BENCH", kind: Kind::Probe, default_on: false,
        purpose: "同进程/同状态 A/B：顺序 tok/s vs 投机 tok/s（草稿 = 主干自然链 ⇒ 接受率 100%）",
        coupling: "值与 LINGXI_SPEC_N 配套；会**多跑一整段序列** ⇒ 容量/rope 表已按 SPEC_N 放宽（否则 `QsaState 容量不足`）" },
    Probe { name: "LINGXI_SPEC_CHECK", kind: Kind::Probe, default_on: false,
        purpose: "投机机械判据：把「快照+批前向+接受+提交回卷」拆成 5 半分别对拍（pp/b0/b2/b1/a2）",
        coupling: "⚠️ **必须叠 LINGXI_NO_GRAPHS=1**：图键只有 `(层, t)` 且烘死状态指针 ⇒ 一个进程里跑第二条序列会写进第一条的状态" },
    Probe { name: "LINGXI_SPEC_N", kind: Kind::Probe, default_on: false,
        purpose: "SPEC_BENCH 的每臂 token 数（默认 24）", coupling: "同时用来放宽 QsaState/rope 容量" },
    Probe { name: "LINGXI_SPEC_ACCEPT", kind: Kind::Probe, default_on: false,
        purpose: "SPEC_BENCH 每轮强制接受的草稿数（默认 t-1）", coupling: "" },
    Probe { name: "LINGXI_SPEC_NOREC", kind: Kind::Probe, default_on: false,
        purpose: "🚫 **诊断用：关掉 GDN 层输入的录像**（用来二分「拷贝本身是否破坏状态」）",
        coupling: "⚠️ **不是可选项就关**：`spec_commit` 的部分接受重放就靠这份录像 ⇒ 关掉后接受率崩到 0.21/6、主干 742 ms/轮，**数学作废**" },
    Probe { name: "LINGXI_SPEC_PREP_NOALLOC", kind: Kind::Probe, default_on: false,
        purpose: "spec_prepare 不预分配（二分「预分配本身是否出问题」）", coupling: "" },
    Probe { name: "LINGXI_SPEC_PREP_NOPROV", kind: Kind::Probe, default_on: false,
        purpose: "spec_prepare 不预取层权重（二分「预取是否污染状态」）", coupling: "" },

    // ── 诊断探针（Probe）——会改时序甚至改数学，绝不进生产 ──
    Probe { name: "LINGXI_PROF", kind: Kind::Probe, default_on: false,
        purpose: "[step]/[moe]/[gdn]/[head] 逐层分块（**CPU 侧时间**，除非叠 KTS）", coupling: "" },
    Probe { name: "LINGXI_KTS", kind: Kind::Probe, default_on: false,
        purpose: "给 [step]/[moe]/[gdn] 的分段**插同步括号** ⇒ 变成真 GPU 时间（会破坏流水，偏大）", coupling: "与所有 prof 段协同；默认关" },
    Probe { name: "LINGXI_PREFILL_PIPE", kind: Kind::Feature, default_on: false,
        purpose: "prefill outer-chunk 成对流水（chunk i 后半 ∥ chunk i+1 前半）；实测不提速且有正确性差异，默认关", coupling: "需 LINGXI_LAYER_SPLIT 生效" },
    Probe { name: "LINGXI_PREFILL_THREADS", kind: Kind::Feature, default_on: false,
        purpose: "prefill 按卡分线程（GPU0/GPU1 各一线程，mpsc 交接）；需缓存型 provider（全部层预热）", coupling: "优先于 LINGXI_PREFILL_PIPE" },
    Probe { name: "LINGXI_PREFILL_LM", kind: Kind::Feature, default_on: false,
        purpose: "**layer-major prefill**（2026-10-18）：外层整层、内层 sub-chunk，每层专家在整段 prompt 上只换入一次。\
                  逐位等价已认证（332/95835/198）；但 33K 实测 291.8s vs 270.6s（chunk=1024 时已是算力 bound）⇒ **默认关**，\
                  留给 ≥128K 超长 prompt 与 expert-stationary 后续工作",
        coupling: "配合 `LINGXI_PREFILL_LM_CHUNK`；split 下自动复用 `LINGXI_PREFILL_THREADS` 做外层流水；\
                   外层次块需常驻 `outer×10240×2B` 激活（默认 16384 ≈ 320MiB/卡）" },
    Probe { name: "LINGXI_PREFILL_LM_CHUNK", kind: Kind::Feature, default_on: false,
        purpose: "layer-major 的外层次块 token 数（默认 16384；0 = 整段 prompt 一块）；越大人越省专家换入但激活常驻越大",
        coupling: "仅 `LINGXI_PREFILL_LM=1` 时生效" },
    Probe { name: "LINGXI_PLE_PROF", kind: Kind::Probe, default_on: false,
        purpose: "[ple-prof] 打印 PLE 每 chunk 的 host 查表+反量化 / f32→bf16+H2D 两段耗时（2026-10-18 实测 512 行约 15~20ms + 15~20ms）",
        coupling: "只打印，不改数学" },
    Probe { name: "LINGXI_PIPE_PROF", kind: Kind::Probe, default_on: false,
        purpose: "[pipeprof] 逐 chunk 打 lo / 跨卡搬运 / hi 三段 host 时间", coupling: "配合 LINGXI_PREFILL_PIPE" },
    Probe { name: "LINGXI_PIPE_DUMP", kind: Kind::Probe, default_on: false,
        purpose: "[pipe-dump] 线程版逐 chunk 回读两半中间结果，打 NaN 数/最大绝对值/和（定位数值发散点，会插同步）",
        coupling: "配合 LINGXI_PREFILL_THREADS" },
    Probe { name: "LINGXI_PREWARM_ALL", kind: Kind::Probe, default_on: false,
        purpose: "顺序路径也先建出全部 48 层——二分「线程版差异」是线程引起还是预热改变了分配历史",
        coupling: "与 LINGXI_PREFILL_THREADS 对照用" },
    Probe { name: "LINGXI_SERVE_MAX_CTX", kind: Kind::Feature, default_on: false,
        purpose: "`serve` 的 prompt+生成容量上限（默认 32768）；建局时一次定死 rope/Sequence",
        coupling: "调大只增 KV/GDN/PLE 状态显存" },
    Probe { name: "LINGXI_SERVE_MAX_NEW", kind: Kind::Feature, default_on: false,
        purpose: "`serve` 单请求默认最大生成长度（默认 512；请求体 max_tokens 可覆盖）", coupling: "" },
    Probe { name: "LINGXI_SERVE_PREFIX_CACHE", kind: Kind::Feature, default_on: true,
        purpose: "`serve` 跨请求智能前缀缓存（C3）：最长匹配检查点 + 消息边界检查点 ⇒ 多轮只 prefill 新增 token",
        coupling: "**默认开**（`=0` 才关）；与 `LINGXI_SERVE_SPEC=1` 互斥（MTP 的 KV 还没纳入检查点）" },
    Probe { name: "LINGXI_SERVE_CKPTS", kind: Kind::Feature, default_on: false,
        purpose: "`serve` 检查点级别（默认 3 = 消息边界+prompt+末尾，≈345MB；2 = prompt+末尾；1 = 只 prompt；0 = 不存）",
        coupling: "" },
    Probe { name: "LINGXI_SERVE_SPEC", kind: Kind::Feature, default_on: false,
        purpose: "`serve` 启用 MTP 投机解码（`spec.rs`）：顺序 43.8 → 约 60 tok/s（kd=5 + min_p 0.7）",
        coupling: "需要 `LINGXI_MTP_ARTIFACT`；与 `LINGXI_SERVE_PREFIX_CACHE` 互斥（spec 要整段 prompt 的 h）；\
                  MTP 专家默认 Q4（`LINGXI_MTP_Q4_EXPERTS=0` 可关）" },
    Probe { name: "LINGXI_MTP_ARTIFACT", kind: Kind::Feature, default_on: false,
        purpose: "`serve` 的 MTP 草稿产物路径（默认 /data/models/lingxi-assets/qwen4_exp.mtp.fixed.ninfer）", coupling: "" },
    Probe { name: "LINGXI_SERVE_SPEC_KD", kind: Kind::Feature, default_on: false,
        purpose: "`serve` 投机窗口（默认 5；§㊿-⑦ 扫描：kd=5 最快、kd=6 最稳）", coupling: "" },
    Probe { name: "LINGXI_MOE_OUT_DUMP", kind: Kind::Probe, default_on: false,
        purpose: "[moe-out] 打**最后一层** MoE 的 out/shared 全和与最大绝对值——定位同输入同路由却输出不同的段落",
        coupling: "与 LINGXI_PIPE_DUMP 配合定位分层/线程差异" },
    Probe { name: "LINGXI_MOE_ZERO_OUT", kind: Kind::Probe, default_on: false,
        purpose: "专家段输出（gu/act/o）先清零：判「是否有行没被写出」——清零后两配置一致即证明差异来自未初始化行",
        coupling: "诊断用，会多三次 memset" },
    Probe { name: "LINGXI_MOE_NAN_FILL", kind: Kind::Probe, default_on: false,
        purpose: "专家段输出先填 NaN：跑完数 NaN 即可**逐元素定位**哪些位置没被写出（正确性判据）",
        coupling: "必须配合 LINGXI_MOE_OUT_DUMP 看 nan=/first_nan=" },
    Probe { name: "LINGXI_ARENA_SLOT_CHECK", kind: Kind::Probe, default_on: false,
        purpose: "取值=层号：回读该层前 3 个 miss 槽的 64 B 与 arena 源字节比对，并核对设备 slot_ids 回读（会插同步）",
        coupling: "定位「同权重同路由、专家输出却不同」" },
    Probe { name: "LINGXI_KT", kind: Kind::Probe, default_on: false,
        purpose: "GDN 的 qkvz 单 kernel GPU 时间（只打前 3 层）",
        coupling: "⚠️ **与 LINGXI_DENSE_Q4 冲突 ⇒ 会 panic**（引用空的 bf16 query_key_value_z）；现已加守卫，但组合仍会 panic" },
    Probe { name: "LINGXI_KT2", kind: Kind::Probe, default_on: false,
        purpose: "逐层 GPU+CPU 独占时间（每层前后各一次 sync）", coupling: "实测几乎不改变总时间，是可信的逐层仪器" },
    Probe { name: "LINGXI_PTRPROBE", kind: Kind::Probe, default_on: false,
        purpose: "打印逻辑缓冲的设备地址（查 scratch 池配对/地址稳定性）", coupling: "图捕获要求地址稳定，二者一起看" },
    Probe { name: "LINGXI_SYNCTEST", kind: Kind::Probe, default_on: false,
        purpose: "空闲 GPU 下单次 cuStreamSynchronize 的代价（实测 2.0 µs）", coupling: "" },
    Probe { name: "LINGXI_GRAPHTEST", kind: Kind::Probe, default_on: false,
        purpose: "逐步探测 CUDA Graph 捕获失败在哪一步", coupling: "" },
    Probe { name: "LINGXI_GRAPHS_VERBOSE", kind: Kind::Probe, default_on: false,
        purpose: "打印图捕获/重放日志", coupling: "" },
    Probe { name: "LINGXI_GRAPH_TIMER", kind: Kind::Probe, default_on: false,
        purpose: "K2 图内探针：run_cached 每个捕获单元首尾 event-record 节点分段计时（DeepInfer graph_timer 借鉴），收尾打印各键 avg/max ms",
        coupling: "只计时不改数值；驱动调用失败自动自关。paged/直跑路径退化为普通流事件计时" },
    Probe { name: "LINGXI_PREFILL_CHUNK", kind: Kind::Feature, default_on: false,
        purpose: "prefill 分块粒度（默认 2048；0/1=不分块老路）", coupling: "" },
    Probe { name: "LINGXI_SERVE_CACHE_DIR", kind: Kind::Feature, default_on: false,
        purpose: "serve 前缀缓存落盘目录", coupling: "与 SERVE_CACHE_GIB/MIN_TOKENS 联动" },
    Probe { name: "LINGXI_SERVE_CACHE_GIB", kind: Kind::Feature, default_on: false,
        purpose: "serve 落盘缓存上限 GiB（默认 24；0=关）", coupling: "" },
    Probe { name: "LINGXI_SERVE_CACHE_MIN_TOKENS", kind: Kind::Feature, default_on: false,
        purpose: "短会话不落盘阈值（默认 8192 token）", coupling: "" },
    Probe { name: "LINGXI_SKIP", kind: Kind::Probe, default_on: false,
        purpose: "剥块（attn/moe/gr/ple）",
        coupling: "⚠️ **会改数学 ⇒ 路由乱跳 ⇒ 槽全失效**，系统性高估被剥的块。**必须叠 LINGXI_ARENA_FIXED_EXPERTS=1**，否则结论作废（§3.29/§3.36）" },
    Probe { name: "LINGXI_ARENA_FIXED_EXPERTS", kind: Kind::Probe, default_on: false,
        purpose: "把路由钉死成固定专家（数学作废，但 kernel/launch/缓冲/同步点与正常路径一致、槽 100% 命中）",
        coupling: "**与 SKIP 组合才可信**；叠 FIXED_IDS 更稳" },
    Probe { name: "LINGXI_ARENA_NOH2D", kind: Kind::Probe, default_on: false,
        purpose: "只记槽不真拷 DMA（量 DMA 净代价，数学作废）", coupling: "" },
    Probe { name: "LINGXI_ARENA_STATS", kind: Kind::Probe, default_on: false,
        purpose: "专家槽命中率日志（每 480 次 load_hits 打一次）", coupling: "" },
    Probe { name: "LINGXI_ARENA_HIST", kind: Kind::Probe, default_on: false,
        purpose: "值 = 落盘路径：**每 (层, 专家) 的请求计数直方图**（每 480 次 load_hits 覆写一次）。\
                  用途 = 离线决定「每层钉住最热 N 个专家」的 N，以及看路由到底有多集中（P1 钉热专家的前提）",
        coupling: "与 ARENA_STATS 同频；不设则零开销（不记表）" },
    Probe { name: "LINGXI_ARENA_ROUTE", kind: Kind::Probe, default_on: false,
        purpose: "一次性打印 MoE 实际走的路（all_q4/want_q4/s）", coupling: "" },
    Probe { name: "LINGXI_ARENA_PREWARM", kind: Kind::Feature, default_on: false,
        purpose: "值 = `LINGXI_ARENA_HIST` 产出的直方图 csv：**启动时把每层最热的 N 个专家搬进显存并钉住**\
                  （不进 LRU）。依据：实测工作集≈槽数 ⇒ 纯 LRU 只有 93% 命中，而每层 top-145 覆盖 99.65% 的请求",
        coupling: "N 由 LINGXI_ARENA_PREWARM_N 定（默认 100）；**钉太多会把尾巴挤没**（N=145 只剩 40 槽）。不设 = 行为完全不变" },
    Probe { name: "LINGXI_ARENA_PREWARM_N", kind: Kind::Feature, default_on: false,
        purpose: "上面每层钉几个（默认 100）。参考覆盖率：bench 负载 top-100→97.3%、top-145→99.65%；\
                  JSON 负载 top-100→75%、top-145→85.5% ⇒ **强依赖工作负载**",
        coupling: "与 ARENA_PREWARM 配套；总占用 = 48×N 槽，必须 < ARENA_SLOTS" },
    Probe { name: "LINGXI_FWD_BENCH", kind: Kind::Probe, default_on: false,
        purpose: "量一次 t=k 前向的墙钟（逗号分隔多个 k）", coupling: "与 FWD_REPS 配套" },
    Probe { name: "LINGXI_FWD_REPS", kind: Kind::Probe, default_on: false,
        purpose: "FWD_BENCH 的重复次数（默认 5）", coupling: "" },
    Probe { name: "LINGXI_MOE_DUMP", kind: Kind::Probe, default_on: false,
        purpose: "落 MoE 入口 x / router logits / top-k 到目录", coupling: "" },
    Probe { name: "LINGXI_MOE_DUMP_ACT", kind: Kind::Feature, default_on: false,
        purpose: "6.2-3 GPTQ 校准：在 LINGXI_MOE_DUMP 目录追加每对 x/act + 对名单（rows>1）",
        coupling: "需同设 LINGXI_MOE_DUMP=<dir>" },
    Probe { name: "LINGXI_MOE_NO_XF32", kind: Kind::Feature, default_on: false,
        purpose: "关掉 slotsm 的 **x-fp32 预转**（P15-⑤b ①），回退 bf16 现转路径做 A/B",
        coupling: "不设 = 默认预转（位等价：bf16→f32 精确、z 次序不变）" },
    Probe { name: "LINGXI_MOE_DP4A", kind: Kind::Feature, default_on: false,
        purpose: "slotsm 改走 **int8 行量化 x + __dp4a 内积**（P15-㉖）：x 的 L1 流量 ÷4，\
                  目标 (e) m=7 279→600+ GB/s",
        coupling: "**默认关：非位等价**（x 量化 int8，误差 ~2^-8，比 Q4 权重的 2^-4 小 16 倍）⇒ \
                  必须过判据（首 token + 固定贪心序列逐位不变 + spec 接受率）才转正；与 xf32 互斥" },
    Probe { name: "LINGXI_MOE_TC", kind: Kind::Feature, default_on: false,
        purpose: "专家段走张量核 `mma.sync.m8n8k4`（P15-㊼）：`1 < rows ≤ 16` 走 QPN 小档（投机验证批），\
                  `rows > 16` 走 ㊽ prefill 大 tile 档（CTA 64 行×128 列、x 进 shared、M 轴进 grid.z，\
                  目标 MoE 77.6 → ≤20 ms/层）",
        coupling: "**有损档**（fp16 权重×scale + x bf16→fp16，~2^-11，远小于 Q4 权重 2^-4）⇒ 判据 = \
                  首 token 相等 + 质量抽检（PLAN_NEXT §3，2026-10-19 拍板放开逐位红线）；t=1 decode 仍走 SIMT" },
    Probe { name: "LINGXI_KV_NO_LAZY", kind: Kind::Feature, default_on: false,
        purpose: "弹性 KV 关闭懒分配：QsaState 按建局 cap 全额预付（㊿-P1 的回归对照档）",
        coupling: "默认懒分配（初始 8192 token，按需几何增长）对数学逐位无影响；判据翻车用本开关对拍" },
    Probe { name: "LINGXI_MOE_PROF", kind: Kind::Probe, default_on: false,
        purpose: "段 4 细分剖面：CUDA event 链打 gu / swiglu+xf / dn 的真实 GPU 时间\
                  （[moe-prof4] 行；事件只记不同步）",
        coupling: "纯诊断；不影响执行次序与数值" },
    Probe { name: "LINGXI_GEMV_PROBE", kind: Kind::Probe, default_on: false,
        purpose: "非 0 即开：把**前 2 个 GDN 层**的 qkv+z 投影背靠背连发 8 次，量无 CPU 空档的\
                  每发成本/GB/s（区分「核真的慢」与「KTS 单发把 launch 排空算进去」）",
        coupling: "只在 t=1 且该层走 Q4 稠密路径时生效；会多做 8 次投影（数学不变）" },
    Probe { name: "LINGXI_F32X_PROJ", kind: Kind::Feature, default_on: false,
        purpose: "启用 bf16 投影的 **x-fp32 预转**（P15-⑨ gemm_skinny_row_f32x）",
        coupling: "**默认关**：整机 fwd t=7 实测 +2.6 ms 净亏（x fp32 使每 block 的 x L2 读翻倍）；\
                  微基准 m=8 带宽 350→700 GB/s 不在整机兑现；GR mix_up 另有未解的整机漂移（见 PLAN P15-⑨）" },
    Probe { name: "LINGXI_HOST_ARGMAX", kind: Kind::Feature, default_on: false,
        purpose: "回退开关：argmax 走 host（锁页 D2H t×vocab + host 遍历），关掉 GPU 两段式 argmax 核（A/B 对照）",
        coupling: "" },
    Probe { name: "LINGXI_HOST_TOPK", kind: Kind::Feature, default_on: false,
        purpose: "回退开关：MTP 草稿的 top-K 走 host（D2H 整幅 logits + topk_bf16_row），关掉 GPU 两段式 top-K 核（A/B 对照）",
        coupling: "" },

    Probe { name: "LINGXI_NO_GR_GRAPH", kind: Kind::Feature, default_on: false,
        purpose: "只禁 GR 的 CUDA Graph（重放改逐 kernel 执行）",
        coupling: "二分诊断开关（P15-⑩）；默认开图" },
    Probe { name: "LINGXI_NO_MOE_GRAPH", kind: Kind::Feature, default_on: false,
        purpose: "只禁 rows==1 专家段的 CUDA Graph",
        coupling: "二分诊断开关（P15-⑩）；默认开图" },
    Probe { name: "LINGXI_MOE_DEBUG", kind: Kind::Probe, default_on: false,
        purpose: "MoE 调试输出", coupling: "" },
    Probe { name: "LINGXI_GDN_DEBUG", kind: Kind::Probe, default_on: false,
        purpose: "dump GDN 的 conv/z/a/b", coupling: "" },
    Probe { name: "LINGXI_LAYER_DEBUG_DIR", kind: Kind::Probe, default_on: false,
        purpose: "落逐层中间张量到目录", coupling: "" },
    Probe { name: "LINGXI_Q4G_DBG", kind: Kind::Probe, default_on: false,
        purpose: "Q4 瘦长 GEMM 调试", coupling: "" },
    Probe { name: "LINGXI_Q4G_TEST", kind: Kind::Probe, default_on: false,
        purpose: "Q4 瘦长 GEMM 自测", coupling: "" },
    Probe { name: "LINGXI_PYTHON", kind: Kind::Probe, default_on: false,
        purpose: "测试用的 python 解释器路径", coupling: "" },

    // ── serve / 会话（Feature，2026-10-21 守卫测试清账补登记） ──
    Probe { name: "LINGXI_SERVE_PERF_LOG", kind: Kind::Feature, default_on: true,
        purpose: "serve 性能日志路径（默认 run/perf.jsonl，不含内容）", coupling: "" },
    Probe { name: "LINGXI_SERVE_PERF_MAX_MB", kind: Kind::Feature, default_on: false,
        purpose: "性能日志文件大小上限（MB，轮转用）", coupling: "" },
    Probe { name: "LINGXI_SERVE_SUNK_MB", kind: Kind::Feature, default_on: false,
        purpose: "sunk（已完成未释放）会话的 host 内存上限（MB）", coupling: "" },
    Probe { name: "LINGXI_SERVE_MATERIALIZED", kind: Kind::Feature, default_on: false,
        purpose: "v1.5 同时物化的 Sequence 数上限（KV 预算外显存零和由 MAX_CTX pledge 口径管）", coupling: "" },

    // ── 容量 / autotune（Feature） ──
    Probe { name: "LINGXI_NO_VRAM_AUTOTUNE", kind: Kind::Feature, default_on: false,
        purpose: "整体关 vram-autotune（启动逐卡算账定槽位），回落 talk/serve 内建默认 7000/4000/9600", coupling: "" },
    Probe { name: "LINGXI_KV_K8V4", kind: Kind::Feature, default_on: false,
        purpose: "KV 走 8-bit（k8v4，每 token 每层 1184B vs bf16 2368B）量化路径", coupling: "与 QSA_PAGED 互斥" },
    Probe { name: "LINGXI_KV_INIT_TOKENS", kind: Kind::Feature, default_on: false,
        purpose: "paged 模式下 KV 初始物化 token 数", coupling: "只在 QSA_PAGED 生效" },
    Probe { name: "LINGXI_ARENA_MIX_Q4_SLOTS", kind: Kind::Feature, default_on: false,
        purpose: "混档 arena 的热 Q4 槽位数（arena_source 对拍/自检用）", coupling: "" },
    Probe { name: "LINGXI_MTP_PRIME_CHUNK", kind: Kind::Feature, default_on: false,
        purpose: "MTP prime 的分块行数（默认 128）", coupling: "" },

    // ── A/B 回退 ──
    Probe { name: "LINGXI_GDN_SMEM", kind: Kind::Ab, default_on: false,
        purpose: "GDN 核 sm_70 dynamic smem 路径（smem ≤96KB 时可用）；实测默认关，仅留档 A/B", coupling: "" },
    Probe { name: "LINGXI_F32X_SITES", kind: Kind::Ab, default_on: false,
        purpose: "F32X_PROJ 的位点掩码（按位关掉某几处 gemm 的 x-fp32 预转做二分定位，默认 0b111）",
        coupling: "只在 LINGXI_F32X_PROJ=1 时有意义" },

    // ── 诊断探针（混档 MoE / 路由 / 计时） ──
    Probe { name: "LINGXI_MIX_DEBUG", kind: Kind::Probe, default_on: false,
        purpose: "混档双视图调试输出", coupling: "" },
    Probe { name: "LINGXI_MIX_ONLY", kind: Kind::Probe, default_on: false,
        purpose: "只跑混档的单个格式段（=3/4；数学不完整，只看该段输出是否 NaN）", coupling: "" },
    Probe { name: "LINGXI_MIX_SYNC_DN", kind: Kind::Probe, default_on: false,
        purpose: "混档 dn 前全流同步——判步 1 输出坏是否 H2D 未就绪竞态", coupling: "会改时序" },
    Probe { name: "LINGXI_MIX_DUMP_DN", kind: Kind::Probe, default_on: false,
        purpose: "dn 循环刚结束立刻回读 o_all——区分 dn gemv 本身坏 vs 后续 merge 覆写", coupling: "" },
    Probe { name: "LINGXI_MIX_FORCE_DUAL", kind: Kind::Probe, default_on: false,
        purpose: "单格式 arena 也走双视图循环的对照（=1 整段 / =2 两段拆）；判双调机制本身是否有 bug", coupling: "" },
    Probe { name: "LINGXI_MOE_OUT_DUMP_LAYER", kind: Kind::Probe, default_on: false,
        purpose: "MOE_OUT_DUMP 的层号过滤器（只 dump 指定层）", coupling: "只在 LINGXI_MOE_OUT_DUMP=1 时生效" },
    Probe { name: "LINGXI_ROUTE_DEV", kind: Kind::Probe, default_on: false,
        purpose: "rows==1 时路由核走 device 参考路径（判据未过前默认关）", coupling: "" },
    Probe { name: "LINGXI_ROUTE_DEV_SYNC", kind: Kind::Probe, default_on: false,
        purpose: "路由核后强制同步（诊断）", coupling: "会改时序" },
    Probe { name: "LINGXI_ROUTE_DEV_TRACE", kind: Kind::Probe, default_on: false,
        purpose: "每 480 次（10 token × 48 层）打印路由命中率", coupling: "" },
    Probe { name: "LINGXI_ROUTE_DEV_CHECK", kind: Kind::Probe, default_on: false,
        purpose: "逐位核对路由核 vs host 参考（会同步，仅诊断）", coupling: "" },
    Probe { name: "LINGXI_LAYER_PROF", kind: Kind::Probe, default_on: false,
        purpose: "每层 host wall 计时（含内部同步等待）——定位 GPU 闲置挂在哪些层", coupling: "会改时序" },
    Probe { name: "LINGXI_DUMP_PREFILL_STATE", kind: Kind::Probe, default_on: false,
        purpose: "prefill 后 dump 引擎中间态（只读打印，不影响数值）", coupling: "" },

    // ── 已知失效 / 幽灵（Ghost）──
    Probe { name: "LINGXI_ROUTER_NOSYNC", kind: Kind::Ghost, default_on: false,
        purpose: "🚫 **代码里已无实现**（全仓只剩注释）——文档里仍在引用它，害得一整组对比实验作废",
        coupling: "用到它 = 什么都没变" },
    Probe { name: "LINGXI_Q4_ROW1", kind: Kind::Ghost, default_on: false,
        purpose: "🚫 已失效：稠密 Q4 的默认**已翻成单行版**（§3.33），本开关不再有任何效果",
        coupling: "改用 LINGXI_Q4_ROW4 做反向对照" },
    Probe { name: "LINGXI_GRAPHS", kind: Kind::Ghost, default_on: false,
        purpose: "🚫 **反向命名的遗留**：图一直是**默认开**，代码只读 `LINGXI_NO_GRAPHS`；\
                 设 `LINGXI_GRAPHS=1` 什么都没变（2026-10-16 由 probes 静态守卫查出来的）",
        coupling: "要关图请用 LINGXI_NO_GRAPHS=1" },
    Probe { name: "LINGXI_GR_Q4", kind: Kind::Ghost, default_on: false,
        purpose: "🚫 **已否决并撤除**：GR 的 mix_down/mix_up 转 W4G64。实测 `mix_down[320,10240]` 的 Q4 **14.2 µs 慢于** bf16 row8 **8.6 µs**（Q4 只有 `n/8=40` 个 block），逐形状算下来净亏；代码已撤，全仓只剩注释",
        coupling: "设了它 = 什么都没变；结论要引用否决表" },
];

/// 启动校验：**点名拼错的开关**、**拦住已知的危险组合**。
///
/// 以前拼错（或用了幽灵开关）是**静默忽略**——你只会得到一个"没变化"的结论，
/// 然后误以为"这个优化没用"。这个函数就是为了让这种错误立刻可见。
pub fn check_env() {
    let mut unknown: Vec<String> = Vec::new();
    let mut ghosts: Vec<&'static str> = Vec::new();
    for (k, _) in std::env::vars() {
        if !k.starts_with("LINGXI_") {
            continue;
        }
        match PROBES.iter().find(|p| p.name == k) {
            None => unknown.push(k),
            Some(p) if p.kind == Kind::Ghost => ghosts.push(p.name),
            Some(_) => {}
        }
    }
    if !unknown.is_empty() {
        eprintln!(
            "[probes] ⚠️ 未登记的 LINGXI_* 开关（多半是拼错，**会被静默忽略**）：{}",
            unknown.join(", ")
        );
        eprintln!("[probes]    已登记的开关见 crates/lingxi-model/src/probes.rs 的 PROBES 表");
    }
    for g in ghosts {
        eprintln!("[probes] ⚠️ 幽灵开关 {} 已被设置，但代码里没有实现 ⇒ 本次实验没有任何变量被改变！", g);
    }

    // **回退开关（Ab）一旦被打开就大声喊**：这些开关的存在意义只是"做一次 A/B 对照"，
    // 它们关掉的都是实测更差的路。历史上多次发生"忘了关某个回退开关，
    // 于是最优路径被默认回退了却不知道"。这里把实测代价直接打出来。
    for p in PROBES.iter().filter(|p| p.kind == Kind::Ab) {
        if std::env::var(p.name).is_ok() {
            eprintln!(
                "[probes] ⚠️ 回退开关 {} = on ⇒ **最优路径已被关掉**，本次数字不能代表生产。用途：{}",
                p.name, p.purpose
            );
        }
    }

    // 危险组合：直接报错退出，而不是崩在内核深处 / 得出错结论。
    let on = |n: &str| std::env::var(n).is_ok();
    if on("LINGXI_KT") && on("LINGXI_DENSE_Q4") {
        panic!(
            "[probes] LINGXI_KT 与 LINGXI_DENSE_Q4 不能同时开：KT 探针引用的 bf16 weight 在 W4 路径下是空的，\
             会在 gdn_step 里 unwrap 到 None 直接崩（cudarc core.rs:2010）。\
             想量 GDN 分段请改用 LINGXI_KTS。"
        );
    }
    if on("LINGXI_SKIP") && !on("LINGXI_ARENA_FIXED_EXPERTS") {
        eprintln!(
            "[probes] ⚠️ 只开 LINGXI_SKIP 而未叠 LINGXI_ARENA_FIXED_EXPERTS：\
             剥块会改数学 ⇒ 路由乱跳 ⇒ 专家槽全失效，**被剥的块会被系统性高估**（§3.29/§3.36）。\
             结论仅在同时叠加 FIXED_EXPERTS（+ FIXED_IDS）时才可信。"
        );
    }
    if on("LINGXI_ARENA_PIN_GIB") {
        eprintln!(
            "[probes] 提示：arena 锁页策略 = **按介质判定** —— arena 在 RAM(tmpfs/ramfs) 上默认**整段** pin；\
             非 RAM 介质一律禁止（2026-10-14 事故）；只接受「整段或 0」，中间值启动即报错。\
             整段 pin 是实测的正收益项：p10 34.4→26.7 ms（29→37.4 tok/s），p50 114→29 ms（8.8→34.5 tok/s）。"
        );
    }
    // ★ P15-⑩（2026-10-17）：**spec × CUDA Graph 的 NaN 守卫正式解除**。
    // 原守卫（2026-10-16）：spec+graph 几轮后 NaN ⇒ 强制 NO_GRAPHS。
    // 根因是**四个叠加的"图内指针漂移"**（全部修复，见 PLAN §P15-⑩）：
    //   1. MoE rows==1 图的 x/gu/act/o 用 scratch（t=7 验证批打乱借还序列）⇒ 改 take_fixed+stage
    //   2. MoE 图的 slot_ids 双缓冲（层间交替）⇒ 图外 stage 进固定槽
    //   3. GR rows==1 与 rows=7 共用 gkey 固定槽互相覆盖 ⇒ 统一按 rows 分键
    //   4. layer.rs 输出缓冲键不带 t + mixed_m/injw_m 的 give 用旧键 ⇒ 键带 `(1+t)<<40`
    // 修复后判据：spec+min_p+graph 接受率 4.14/6 与 OFF 逐位一致；t=1 decode 图 ON 快 21%。
}

/// 把所有开关的当前状态打出来（便于把一次实验的配置原样写进档案）。
pub fn dump() -> String {
    let mut s = String::from("=== LINGXI 开关状态 ===\n");
    for p in PROBES {
        let v = std::env::var(p.name).ok();
        let st = match (&p.kind, &v) {
            (Kind::Ghost, Some(_)) => "🚫幽灵已设".to_string(),
            (Kind::Ghost, None) => "🚫已失效".to_string(),
            (_, Some(val)) => format!("= {val}"),
            (Kind::Feature, None) => if p.default_on { "默认开".to_string() } else { "关".to_string() },
            (_, None) => "关".to_string(),
        };
        let tag = match p.kind {
            Kind::Feature => "F",
            Kind::Ab => "A",
            Kind::Probe => "P",
            Kind::Ghost => "X",
        };
        s.push_str(&format!("  [{tag}] {:<32} {}\n", p.name, st));
    }
    s.push_str("  F=功能 A=A/B回退 P=诊断探针 X=已失效\n");
    s
}

#[cfg(test)]
mod guard {
    //! **静态守卫**：把"漏登记开关"从**静默**变成**红测试**。
    //!
    //! 为什么必须补这一道：`check_env()` 只能发现「**环境里设了**但没登记」的开关，
    //! 而且只 `eprintln!` 一声 warn。于是一个「代码读了、但谁都没设过」的开关可以永远
    //! 悄悄活着 —— 2026-10-16 一查就抓到 **12 个**（整套 `LINGXI_SPEC_*` / `LINGXI_MTP_SPEC`），
    //! 而它们正是当轮所有提速结论的依据。
    //!
    //! 判据：**凡是出现在 `env::var("…")` / `flag("…")` / `on("…")` 里的 `LINGXI_*` 字面量，
    //! 都必须在本文件的 `PROBES` 表里有一行**（Ghost 也算登记）。

    use super::PROBES;

    /// 从一份 .rs 源码里抽出"真的被读"的开关名（而不是注释里提一嘴的那种）。
    fn read_switches(src: &str) -> Vec<String> {
        const CALLS: [&str; 3] = ["env::var(\"", "flag(\"", "on(\""];
        let b = src.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        while let Some(rel) = src[i..].find("LINGXI_") {
            let start = i + rel;
            // 取名字
            let mut end = start;
            while end < b.len() && (b[end].is_ascii_uppercase() || b[end].is_ascii_digit() || b[end] == b'_')
            {
                end += 1;
            }
            // 往前约 24 字节里必须紧跟着一个"读"调用（窗口起点要对齐到字符边界，
            // 否则中文注释里的多字节字符会让 `is_char_boundary` 失败、白白漏判）
            let mut w = start.saturating_sub(24);
            while w < start && !src.is_char_boundary(w) {
                w += 1;
            }
            let pre = &src[w..start];
            if CALLS.iter().any(|c| pre.ends_with(c)) {
                out.push(src[start..end].to_string());
            }
            i = end.max(start + 1);
        }
        out
    }

    #[test]
    fn every_read_switch_is_registered() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/lingxi-model 的父目录")
            .parent()
            .expect("仓库根")
            .to_path_buf();
        let mut stack = vec![root.join("crates")];
        let mut missing: Vec<String> = Vec::new();
        let mut files = 0usize;
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                if p.extension().and_then(|s| s.to_str()) != Some("rs") {
                    continue;
                }
                files += 1;
                let Ok(txt) = std::fs::read_to_string(&p) else { continue };
                for name in read_switches(&txt) {
                    if !PROBES.iter().any(|pr| pr.name == name) && !missing.contains(&name) {
                        missing.push(name);
                    }
                }
            }
        }
        assert!(files > 20, "只扫到 {files} 个 .rs，路径大概不对");
        assert!(
            missing.is_empty(),
            "以下 LINGXI_* 开关**被代码真的读了**但没登记进 PROBES（漏登记是静默的，\
             以前靠人肉回忆，吃过三次亏）：{missing:?}\n\
             ⇒ 请补进 crates/lingxi-model/src/probes.rs 的 PROBES 表（含 Kind / purpose / coupling）。"
        );
    }

    /// 表里不许有重名（重名会让 `dump()` 打两份、`check_env` 判断歧义）。
    #[test]
    fn probe_names_are_unique() {
        for (i, a) in PROBES.iter().enumerate() {
            for b in &PROBES[i + 1..] {
                assert_ne!(a.name, b.name, "PROBES 里 {} 登记了两次", a.name);
            }
        }
    }
}
