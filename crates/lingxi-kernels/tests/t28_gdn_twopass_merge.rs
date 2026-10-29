//! ** 路线 A two-pass 双卡并行 prefill 的 GDN 数值判决**（2026-10-27）。
//!
//! 背景：prefill 双卡 24 层切分在
//! chunk 内硬串行，每卡 GPU busy 仅 ~40%。处方 = GDN 状态并行 two-pass：卡 A 跑半段
//! prompt 的全部 48 层、卡 B 用 A 的末态重跑另半段（delta rule 的 chunked 可结合性）。
//! **核心风险 = 数值等价**：段切分改变了递推的 launch 结构，必须判决两段式 vs 单段式
//! 是否逐位一致（本仓铁律：递推核 dk 升序链是 det 锚）。
//!
//! 数学事实（本测试验证的对象）：GDN 每步对状态 S（hk×hv，f32）是**仿射**的
//! （gdn_ops.cu 头注）：S_i = A_i·S_{i-1} + B_i，A_i = decay_i·(I − β_i·k̂_i k̂_iᵀ)，
//! B_i = β_i·k̂_i v_iᵀ。仿射映射复合仍仿射 ⇒ 段合并**实数意义下精确**；而递推核的
//! 每 token 算术（dk 升序链、RMSNorm shuffle 树、门 expf 链）**不含任何 T 依赖项**
//! （循环只用 i 做行寻址）⇒ 同一内核以「半段末态 S_A 作初值重跑后半段」与「一次跑
//! 全段」逐位相同 —— 条件是状态交接走 f32 位精确拷贝（D2D），且两段的 k̂/β/decay
//! 输入逐位相同（它们只依赖本 token 输入，与历史无关，天然满足）。
//!
//! 判据：
//! 1. **主判据（逐位）**：split ∈ {256, 200, 128, 1, 128×4}，两种初态（零态 = prefill
//! 真实形态 / 随机态 = 任意历史跨卡交接），两种门配置（decay≈1 / 变 decay），
//! 两段式的 s_all 与全部 gated 行 vs 单段式 **0 bit 不同**。
//! 2. **非空洞校验**：f64 黄金参考（CPU 逐 token 复刻核数学）与内核输出 max_abs 在
//! bf16 输出量化量级；NaN/Inf = 0。
//! 3. **有损替代路径量级**（设计不需要它，仅留档对比）：伴随矩阵合并
//! S_total = M_B·S_A + C_B（f32 CPU，vh=0）vs f32 顺序参考 vs f64 黄金 —— 给出
//! 「若走伴随矩阵式」的误差量级，论证重跑式（逐位）是正确选择。
//!
//! 派发覆盖：本测试只走生产公开 wrapper `gdn_recurrent_bf16`（= 生产派发点）。
//! `LINGXI_NO_GDN_REC_REG` / `LINGXI_GDN_PIPE` / `LINGXI_GDN_PREFETCH` 在其 OnceCell
//! 首次读取时定型 ⇒ 用 env 组合分进程跑：
//! 默认（reg+prefetch+pipe，生产默认路径）/ LINGXI_NO_GDN_REC_REG=1（旧核）。
//!
//! `bash
//! cargo test --release -p lingxi-kernels --features gpu --test t28_gdn_twopass_merge \
//! -- --nocapture --test-threads=1
//! LINGXI_NO_GDN_REC_REG=1 cargo test --release -p lingxi-kernels --features gpu \
//! --test t28_gdn_twopass_merge -- --nocapture --test-threads=1
//! `

#![cfg(feature = "gpu")]

use half::bf16;
use lingxi_kernels::Kernels;

const NVH: usize = 48;
const NKH: usize = 16;
const HK: usize = 128;
const HV: usize = 128;
const T: usize = 512; // 单段全长（生产 chunk 512 同档）

const KEY_DIM: usize = NKH * HK;
const VALUE_DIM: usize = NVH * HV;
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16777216.0 - 0.5) * scale
        })
        .collect()
}

fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|x| bf16::from_f32(*x)).collect()
}

struct Data {
    conv: Vec<bf16>, // [T, CONV_DIM] 已过 silu 的因果卷积输出（q|k|v 交织）
    z: Vec<bf16>,    // [T, VALUE_DIM]
    a: Vec<bf16>,    // [T, NVH]
    b: Vec<bf16>,    // [T, NVH]
    nw: Vec<bf16>,   // [HV]
    s0: Vec<f32>,    // [NVH*HK*HV] 初态（零态 或 随机态）
}

fn mk_data(seed: u32, _alog: &[f32], zero_init: bool) -> Data {
    Data {
        conv: to_bf16(&rnd(T * CONV_DIM, seed, 0.25)),
        z: to_bf16(&rnd(T * VALUE_DIM, seed + 1, 0.5)),
        a: to_bf16(&rnd(T * NVH, seed + 2, 0.25)),
        b: to_bf16(&rnd(T * NVH, seed + 3, 0.25)),
        nw: to_bf16(&rnd(HV, seed + 4, 0.2)),
        s0: if zero_init {
            vec![0.0; NVH * HK * HV]
        } else {
            rnd(NVH * HK * HV, seed + 5, 0.1)
        },
    }
}

/// 生产派发（`gdn_recurrent_bf16`，reg/旧核由 env 门控），输入为完整设备缓冲的行偏移视图。
fn run_range(
    k: &Kernels,
    conv: &cudarc::driver::CudaSlice<bf16>,
    z: &cudarc::driver::CudaSlice<bf16>,
    a: &cudarc::driver::CudaSlice<bf16>,
    b: &cudarc::driver::CudaSlice<bf16>,
    nw: &cudarc::driver::CudaSlice<bf16>,
    alog: &cudarc::driver::CudaSlice<f32>,
    dtb: &cudarc::driver::CudaSlice<f32>,
    s: &mut cudarc::driver::CudaSlice<f32>,
    gated: &mut cudarc::driver::CudaSlice<bf16>,
    rows: usize,
) {
    k.gdn_recurrent_bf16(conv, z, a, b, nw, alog, dtb, s, gated, rows, HK, HV, NVH, NKH, 1, 0, 1)
        .unwrap();
}

fn ndiff_f32(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn ndiff_bf16(a: &[bf16], b: &[bf16]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn maxabs_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

fn maxabs_f32f64(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y).abs())
        .fold(0.0, f64::max)
}

fn maxabs_bf16(a: &[bf16], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() as f64 - *y).abs())
        .fold(0.0, f64::max)
}

/// 主判据：两段式（[0,split) + 末态交接 + [split,T)）vs 单段式，s_all 与 gated 全行逐位对比。
#[test]
fn twopass_bitwise_vs_singlepass() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    // 门配置：① t16 标准（a_log=-10 ⇒ decay≈1，递推链最不稳定性考验）；
    // ② 变 decay（a_log 跨 -6.5..-3.5，softplus/sigmoid 全链 exercised）。
    let cfgs: [(&str, Vec<f32>); 2] = [
        ("decay≈1", vec![-10.0f32; NVH]),
        (
            "变decay",
            (0..NVH).map(|i| -6.5 + 3.0 * i as f32 / NVH as f32).collect(),
        ),
    ];
    // split 档（每档为完整分段序列）：256+256 ≈ 路线 A 半段；200+312 非对齐；
    // 1+511 / 511+1 极端边界；384+128 混合；128×4 = 生产 chunking 复刻。
    let splits: [&[usize]; 6] = [
        &[256, 256],
        &[200, 312],
        &[1, 511],
        &[511, 1],
        &[384, 128],
        &[128, 128, 128, 128],
    ];

    for (cname, alog_v) in cfgs.iter() {
        for zero_init in [true, false] {
            let d = mk_data(11, alog_v, zero_init);
            let alog = k.stream.clone_htod(alog_v).unwrap();
            let dtb = k.stream.clone_htod(&vec![0.0f32; NVH]).unwrap();
            // 参考：单段全长，一次 launch。
            let mut s_ref = k.stream.clone_htod(&d.s0).unwrap();
            let mut g_ref = k.stream.alloc_zeros::<bf16>(T * VALUE_DIM).unwrap();
            {
                let conv = k.stream.clone_htod(&d.conv).unwrap();
                let z = k.stream.clone_htod(&d.z).unwrap();
                let a = k.stream.clone_htod(&d.a).unwrap();
                let b = k.stream.clone_htod(&d.b).unwrap();
                let nw = k.stream.clone_htod(&d.nw).unwrap();
                run_range(&k, &conv, &z, &a, &b, &nw, &alog, &dtb, &mut s_ref, &mut g_ref, T);
                k.stream.synchronize().unwrap();
            }
            let hs_ref: Vec<f32> = k.stream.clone_dtoh(&s_ref).unwrap();
            let hg_ref: Vec<bf16> = k.stream.clone_dtoh(&g_ref).unwrap();

            for sp in splits {
                // 逐段上传该段的行切片（字节与全缓冲行视图完全一致），段间只交接 s_all。
                let nw = k.stream.clone_htod(&d.nw).unwrap();
                let mut s_tp = k.stream.clone_htod(&d.s0).unwrap();
                let mut hg_tp = vec![bf16::from_f32(0.0); T * VALUE_DIM];
                let mut off = 0usize; // 已交接的 token 数
                for &seg in sp.iter() {
                    let conv = k
                        .stream
                        .clone_htod(&d.conv[off * CONV_DIM..(off + seg) * CONV_DIM])
                        .unwrap();
                    let z = k
                        .stream
                        .clone_htod(&d.z[off * VALUE_DIM..(off + seg) * VALUE_DIM])
                        .unwrap();
                    let a = k.stream.clone_htod(&d.a[off * NVH..(off + seg) * NVH]).unwrap();
                    let b = k.stream.clone_htod(&d.b[off * NVH..(off + seg) * NVH]).unwrap();
                    let mut g_seg = k.stream.alloc_zeros::<bf16>(seg * VALUE_DIM).unwrap();
                    run_range(
                        &k, &conv, &z, &a, &b, &nw, &alog, &dtb, &mut s_tp, &mut g_seg, seg,
                    );
                    hg_tp[off * VALUE_DIM..(off + seg) * VALUE_DIM]
                        .copy_from_slice(&k.stream.clone_dtoh(&g_seg).unwrap());
                    off += seg;
                }
                assert_eq!(off, T, "split 档不覆盖全长");
                k.stream.synchronize().unwrap();
                let hs_tp: Vec<f32> = k.stream.clone_dtoh(&s_tp).unwrap();
                let nds = ndiff_f32(&hs_ref, &hs_tp);
                let ndg = ndiff_bf16(&hg_ref, &hg_tp);
                let nan = hg_tp.iter().filter(|x| !x.to_f32().is_finite()).count();
                println!(
                    "[t28-twopass] cfg{cname}/零态={zero_init}/split={sp:?}：s_all {nds}/{} 位不同 \
                     ｜gated {ndg}/{} 位不同｜NaN {nan}",
                    hs_ref.len(),
                    hg_ref.len()
                );
                assert_eq!(nan, 0, "gated 出现 NaN/Inf");
                assert_eq!(nds, 0, "两段式 s_all 与单段式不逐位相同（split={sp:?}）");
                assert_eq!(ndg, 0, "两段式 gated 与单段式不逐位相同（split={sp:?}）");
            }
        }
    }
}

/// f64 黄金参考：CPU 逐 token 复刻核数学（dk 升序、先 decay 后 kv_mem、更新后 S 算 acc）。
/// 只用于证明主判据非空洞（数据真实 exercising 算术），并给「有损路径」一个真值锚。
struct Golden {
    s: Vec<f64>,          // [NVH*HK*HV]
    gated: Vec<f64>,      // [T*VALUE_DIM]
}

fn golden(d: &Data, alog_v: &[f32]) -> Golden {
    let mut s = vec![0.0f64; NVH * HK * HV];
    for i in 0..NVH * HK * HV {
        s[i] = d.s0[i] as f64;
    }
    let mut gated = vec![0.0f64; T * VALUE_DIM];
    let c2f = |x: bf16| x.to_f32() as f64;
    for vh in 0..NVH {
        let hh = vh / (NVH / NKH);
        let al = (alog_v[vh] as f64).exp();
        for i in 0..T {
            let conv_i = &d.conv[i * CONV_DIM..];
            let q: Vec<f64> = (0..HK).map(|u| c2f(conv_i[hh * HK + u])).collect();
            let kk: Vec<f64> = (0..HK).map(|u| c2f(conv_i[KEY_DIM + hh * HK + u])).collect();
            let vv: Vec<f64> = (0..HV)
                .map(|u| c2f(conv_i[2 * KEY_DIM + vh * HV + u]))
                .collect();
            let (mut qss, mut kss) = (0.0f64, 0.0f64);
            for u in 0..HK {
                qss += q[u] * q[u];
                kss += kk[u] * kk[u];
            }
            let qscale = (1.0 / (qss + 1e-6).sqrt()) / (HK as f64).sqrt();
            let kscale = 1.0 / (kss + 1e-6).sqrt();
            let qn: Vec<f64> = q.iter().map(|x| x * qscale).collect();
            let kn: Vec<f64> = kk.iter().map(|x| x * kscale).collect();
            let av = c2f(d.a[i * NVH + vh]);
            let sp = if av > 20.0 { av } else { (1.0 + av.exp()).ln() };
            let decay = (-al * sp).exp();
            let beta = 1.0 / (1.0 + (-c2f(d.b[i * NVH + vh])).exp());
            let base = vh * HK * HV;
            // S *= decay；kv_mem[dv] = Σ_dk (S·decay)[dk][dv]·k̂[dk]（逐列 dk 升序，镜像核）
            let mut kv = vec![0.0f64; HV];
            for dv in 0..HV {
                let mut acc = 0.0f64;
                for dk in 0..HK {
                    let sd = s[base + dk * HV + dv] * decay;
                    s[base + dk * HV + dv] = sd;
                    acc += sd * kn[dk];
                }
                kv[dv] = acc;
            }
            let mut delta = vec![0.0f64; HV];
            for dv in 0..HV {
                delta[dv] = (vv[dv] - kv[dv]) * beta;
            }
            // S += k̂⊗delta；acc[dv] = Σ S_new·q̂；RMSNorm 跨 dv（Σacc²/hv）后乘门
            let mut accs = vec![0.0f64; HV];
            for dv in 0..HV {
                let mut acc = 0.0f64;
                for dk in 0..HK {
                    let su = s[base + dk * HV + dv] + kn[dk] * delta[dv];
                    s[base + dk * HV + dv] = su;
                    acc += su * qn[dk];
                }
                accs[dv] = acc;
            }
            let ssq: f64 = accs.iter().map(|a| a * a).sum();
            let r = 1.0 / (ssq / HV as f64 + 1e-6).sqrt();
            for dv in 0..HV {
                let zz = c2f(d.z[i * VALUE_DIM + vh * HV + dv]);
                let actv = 1.0 / (1.0 + (-zz).exp()); // gate_act=1 sigmoid
                gated[i * VALUE_DIM + vh * HV + dv] = accs[dv] * r * c2f(d.nw[dv]) * actv;
            }
        }
    }
    Golden { s, gated }
}

/// 非空洞校验：内核单段输出 vs f64 黄金 —— max_abs 须在 bf16 输出量化量级，NaN=0。
#[test]
fn f64_golden_sanity() {
    let k = Kernels::load(0).expect("CUDA 初始化失败");
    let alog_v = vec![-10.0f32; NVH];
    for zero_init in [true, false] {
        let d = mk_data(11, &alog_v, zero_init);
        let alog = k.stream.clone_htod(&alog_v).unwrap();
        let dtb = k.stream.clone_htod(&vec![0.0f32; NVH]).unwrap();
        let conv = k.stream.clone_htod(&d.conv).unwrap();
        let z = k.stream.clone_htod(&d.z).unwrap();
        let a = k.stream.clone_htod(&d.a).unwrap();
        let b = k.stream.clone_htod(&d.b).unwrap();
        let nw = k.stream.clone_htod(&d.nw).unwrap();
        let mut s = k.stream.clone_htod(&d.s0).unwrap();
        let mut g = k.stream.alloc_zeros::<bf16>(T * VALUE_DIM).unwrap();
        run_range(&k, &conv, &z, &a, &b, &nw, &alog, &dtb, &mut s, &mut g, T);
        k.stream.synchronize().unwrap();
        let hs: Vec<f32> = k.stream.clone_dtoh(&s).unwrap();
        let hg: Vec<bf16> = k.stream.clone_dtoh(&g).unwrap();
        let gold = golden(&d, &alog_v);
        let nan = hg.iter().filter(|x| !x.to_f32().is_finite()).count();
        let gmax = maxabs_bf16(&hg, &gold.gated);
        let smax = maxabs_f32f64(&hs, &gold.s);
        // bf16 输出与 f64 黄金「同一 bf16 桶」的比率（真值落舍入边界 ±1ulp 内即视为同桶）。
        let bitmatch = hg
            .iter()
            .zip(gold.gated.iter())
            .filter(|(x, y)| {
                let xf = x.to_f32() as f64;
                let bucket = (*y * 256.0).round() / 256.0; // bf16 尾数 8 位的粗桶
                (xf - bucket).abs() <= 4.0 / 256.0
            })
            .count();
        println!(
            "[t28-golden] 零态={zero_init}：NaN {nan}｜gated max_abs {gmax:.3e}｜\
             s_all max_abs {smax:.3e}｜gated 与 f64 黄金同桶 {bitmatch}/{}（{:.1}%）",
            hg.len(),
            bitmatch as f64 / hg.len() as f64 * 100.0
        );
        assert_eq!(nan, 0, "内核输出 NaN/Inf");
        assert!(gmax < 2e-2, "gated 与 f64 黄金偏差过大（{gmax:.3e}）—— 数据/实现有洞");
    }
}

/// 有损替代路径留档：伴随矩阵合并 S_total = M_B·S_A + C_B（f32，vh=0，split=256）。
/// 设计结论选重跑式（逐位），本用例只给「若走伴随矩阵式」的 f32 误差量级作对比。
#[test]
fn companion_merge_lossy_magnitude() {
    let alog_v = vec![-6.0f32; NVH];
    let d = mk_data(23, &alog_v, false);
    let split = 256usize;
    let vh = 0usize;
    let hh = vh / (NVH / NKH);
    let c2f = |x: bf16| x.to_f32();
    // f32 顺序递推（vh=0，dk 升序，镜像核次序）
    let seq_step = |s: &mut [f32], i: usize| {
        let conv_i = &d.conv[i * CONV_DIM..];
        let (mut qss, mut kss) = (0.0f32, 0.0f32);
        for u in 0..HK {
            let qv = c2f(conv_i[hh * HK + u]);
            let kv = c2f(conv_i[KEY_DIM + hh * HK + u]);
            qss += qv * qv;
            kss += kv * kv;
        }
        let _qscale = (1.0f32 / (qss + 1e-6).sqrt()) / (HK as f32).sqrt();
        let kscale = 1.0f32 / (kss + 1e-6).sqrt();
        let av = c2f(d.a[i * NVH + vh]);
        let sp = if av > 20.0 { av } else { (1.0f32 + av.exp()).ln() };
        let decay = (-alog_v[vh].exp() * sp).exp();
        let beta = 1.0f32 / (1.0f32 + (-c2f(d.b[i * NVH + vh])).exp());
        let mut kv_mem = vec![0.0f32; HV];
        for dv in 0..HV {
            let mut acc = 0.0f32;
            for dk in 0..HK {
                let kn = c2f(conv_i[KEY_DIM + hh * HK + dk]) * kscale;
                let sd = s[dk * HV + dv] * decay;
                s[dk * HV + dv] = sd;
                acc += sd * kn;
            }
            kv_mem[dv] = acc;
        }
        for dv in 0..HV {
            let v = c2f(conv_i[2 * KEY_DIM + vh * HV + dv]);
            let delta = (v - kv_mem[dv]) * beta;
            for dk in 0..HK {
                let kn = c2f(conv_i[KEY_DIM + hh * HK + dk]) * kscale;
                s[dk * HV + dv] += kn * delta;
            }
        }
    };
    let mut s_seq = vec![0.0f32; HK * HV];
    for i in 0..T {
        seq_step(&mut s_seq, i);
    }
    // f32 两段：S_A（[0,256) 从初态=0）⇒ 用 S_A 初值重跑 [256,512)
    let mut s_a = vec![0.0f32; HK * HV];
    for i in 0..split {
        seq_step(&mut s_a, i);
    }
    let mut s_b = s_a.clone();
    for i in split..T {
        seq_step(&mut s_b, i);
    }
    let d_s = maxabs_f32(&s_seq, &s_b);
    println!("[t28-companion] f32 两段(状态交接重跑, vh0) vs f32 单段：max_abs {d_s:.3e}");

    // 伴随矩阵式：段 B 传播 M（128×128）与局部终态 C，S_merge = M·S_A + C（全 f32）
    let mut m = vec![0.0f32; HK * HK];
    for r in 0..HK {
        m[r * HK + r] = 1.0;
    }
    let mut c_loc = vec![0.0f32; HK * HV];
    for i in split..T {
        let conv_i = &d.conv[i * CONV_DIM..];
        let mut kss = 0.0f32;
        for u in 0..HK {
            let kv = c2f(conv_i[KEY_DIM + hh * HK + u]);
            kss += kv * kv;
        }
        let kscale = 1.0f32 / (kss + 1e-6).sqrt();
        let av = c2f(d.a[i * NVH + vh]);
        let sp = if av > 20.0 { av } else { (1.0f32 + av.exp()).ln() };
        let decay = (-alog_v[vh].exp() * sp).exp();
        let beta = 1.0f32 / (1.0f32 + (-c2f(d.b[i * NVH + vh])).exp());
        let kn: Vec<f32> = (0..HK).map(|dk| c2f(conv_i[KEY_DIM + hh * HK + dk]) * kscale).collect();
        // w[c] = Σ k̂[dk]·M[dk][c]；M ← decay·(M − β·k̂·w)
        let mut w = vec![0.0f32; HK];
        for c in 0..HK {
            for dk in 0..HK {
                w[c] += kn[dk] * m[dk * HK + c];
            }
        }
        for dk in 0..HK {
            for cc in 0..HK {
                m[dk * HK + cc] = decay * (m[dk * HK + cc] - beta * kn[dk] * w[cc]);
            }
        }
        // wc[c] = Σ k̂[dk]·C[dk][c]；C ← decay·(C − β·k̂·wc) + β·k̂·v
        let mut wc = vec![0.0f32; HV];
        for c in 0..HV {
            for dk in 0..HK {
                wc[c] += kn[dk] * c_loc[dk * HV + c];
            }
        }
        for dk in 0..HK {
            for cc in 0..HV {
                let v = c2f(conv_i[2 * KEY_DIM + vh * HV + cc]);
                c_loc[dk * HV + cc] =
                    decay * (c_loc[dk * HV + cc] - beta * kn[dk] * wc[cc]) + beta * kn[dk] * v;
            }
        }
    }
    // S_merge = M·S_A + C（f32）
    let mut s_merge = vec![0.0f32; HK * HV];
    for r in 0..HK {
        for cc in 0..HV {
            let mut acc = c_loc[r * HV + cc];
            for dk in 0..HK {
                acc += m[r * HK + dk] * s_a[dk * HV + cc];
            }
            s_merge[r * HV + cc] = acc;
        }
    }
    let d_merge = maxabs_f32(&s_seq, &s_merge);
    let scale = s_seq.iter().map(|x| x.abs()).fold(0.0f32, f32::max) as f64;
    println!(
        "[t28-companion] f32 伴随矩阵合并(vh0, split={split}) vs f32 单段：max_abs {d_merge:.3e}｜\
         状态量级 {scale:.3e}｜相对 ~{:.3e}",
        d_merge / scale.max(1e-30)
    );
    // 伴随矩阵式只作留档对比，不设硬断言（设计已选逐位重跑式）；
    // 只挡数量级失控（f32 归约次序差异的典型量级应 ≪ 1e-2）。
    assert!(d_merge / scale.max(1e-30) < 1e-2, "伴随矩阵合并误差异常大");
    assert_eq!(d_s, 0.0, "f32 顺序镜像的两段重跑不应有差异（CPU 同序对照）");
}
