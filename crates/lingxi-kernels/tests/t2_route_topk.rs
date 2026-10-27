//! `lingxi_route_topk_rows_bf16`（rows>1 批量路由）压力/位级回归。
//!
//! 背景：2026-10-23 生产投机 128 token 跑到末轮时偶发
//! `CUDA_ERROR_ILLEGAL_ADDRESS`，先在 `LINGXI_ROUTE_DEV=1` 的 **legacy 路径**上复现
//! （与批量设备路由无关）⇒ 目标锁定本核。本测试用真实形状（E=512、top_k=10、
//! rows 1..16）+ 若干极端数据分布反复打，并在每一轮与 host 参考**逐位**核对。
//!
//! 门控：`cargo test -p lingxi-kernels --features gpu --test t2_route_topk -- --nocapture`
//! 可用 `LINGXI_T2_ITERS` 调迭代数（默认 4000）。

#![cfg(feature = "gpu")]

use half::bf16;

/// host 参考：与 `topk_bf16_row` 同语义 —— softmax（升序串行和、IEEE 除法）后按
/// 全序 (p 降序, 下标升序) 取前 k，再按**专家升序**输出 (id, w)。
fn host_ref_row(logits: &[f32], top_k: usize) -> (Vec<i32>, Vec<f32>) {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
    let mut sum = 0.0f32;
    for v in &p {
        sum += *v;
    }
    let p: Vec<f32> = p.iter().map(|v| v / sum).collect();
    // 全序排名
    let e = p.len();
    let mut rank = vec![0usize; e];
    for i in 0..e {
        let mut c = 0usize;
        for j in 0..e {
            if p[j] > p[i] || (p[j] == p[i] && j < i) {
                c += 1;
            }
        }
        rank[i] = c;
    }
    let mut chosen: Vec<usize> = (0..e).filter(|&i| rank[i] < top_k).collect();
    chosen.sort_unstable(); // 专家升序
    let mut norm = 0.0f32;
    let mut sel = chosen.clone();
    sel.sort_by_key(|&i| rank[i]);
    for i in sel {
        norm += p[i];
    }
    let ids = chosen.iter().map(|&i| i as i32).collect();
    let wts = chosen.iter().map(|&i| p[i] / norm).collect();
    (ids, wts)
}

fn pattern(kind: usize, e: usize, seed: u64, row: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(row as u64 * 1442695040888963407);
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 40.0 - 20.0
    };
    match kind {
        // 均匀随机
        0 => (0..e).map(|_| rnd()).collect(),
        // 全部相等（大量并列 ⇒ 排名退化压力）
        1 => vec![0.5f32; e],
        // bf16 粗粒度并列：量化到 3 个值
        2 => (0..e).map(|_| (rnd() * 4.0).round() * 0.25).collect(),
        // 含 -inf（softmax 后 p=0，排名里一堆 0 并列）
        3 => (0..e)
            .map(|i| if i % 7 == 0 { f32::NEG_INFINITY } else { rnd() })
            .collect(),
        // 极端量级（跨 80 个数量级，压 expf 的范围分支）
        4 => (0..e).map(|_| rnd() * 40.0).collect(),
        // 含 NaN（比较全假）
        _ => (0..e)
            .map(|i| if i % 13 == 0 { f32::NAN } else { rnd() })
            .collect(),
    }
}

#[test]
fn route_topk_rows_stress() {
    let iters: usize = std::env::var("LINGXI_T2_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);
    let k = lingxi_kernels::Kernels::load(0).expect("CUDA 初始化失败");
    let e = 512usize;
    let top_k = 10usize;
    let mut d_logits = unsafe { k.stream.alloc::<bf16>(16 * e) }.unwrap();
    let mut d_ids = unsafe { k.stream.alloc::<i32>(16 * top_k) }.unwrap();
    let mut d_wts = unsafe { k.stream.alloc::<f32>(16 * top_k) }.unwrap();
    let mut n_bad = 0usize;
    for it in 0..iters {
        let rows = 1 + it % 16;
        let kind = (it / 16) % 6;
        let mut host = Vec::with_capacity(rows * e);
        for r in 0..rows {
            host.extend(pattern(kind, e, it as u64 + 1, r));
        }
        let host_bf: Vec<bf16> = host.iter().map(|v| bf16::from_f32(*v)).collect();
        k.stream.memcpy_htod(&host_bf, &mut d_logits).unwrap();
        k.route_topk_rows_bf16(&d_logits, &mut d_ids, &mut d_wts, rows, e, top_k).unwrap();
        let got_ids: Vec<i32> = k.stream.clone_dtoh(&d_ids).unwrap();
        let got_wts: Vec<f32> = k.stream.clone_dtoh(&d_wts).unwrap();
        for r in 0..rows {
            let lr: Vec<f32> = host_bf[r * e..(r + 1) * e].iter().map(|v| v.to_f32()).collect();
            let gi = &got_ids[r * top_k..(r + 1) * top_k];
            let gw = &got_wts[r * top_k..(r + 1) * top_k];
            // 任何行（含 NaN 行）都必须给出**可查表**的 ids：越界 ids 会让下游 pair 核
            // 按 slot_map[ids] 越界读（2026-10-23 生产事故的直接触发点）。
            let ids_in_range = gi.iter().all(|&v| v >= 0 && (v as usize) < e);
            assert!(ids_in_range, "it={it} kind={kind} row={r} ids 越界: {gi:?}");
            if lr.iter().any(|v| v.is_nan()) {
                // NaN 行：不参与位级比对，只要求 ids 合法且权重可安全参与累加。
                assert!(gw.iter().all(|v| v.is_finite()), "it={it} row={r} NaN 行 wts 非有限");
                continue;
            }
            let (want_ids, want_wts) = host_ref_row(&lr, top_k);
            let ids_ok = gi == want_ids.as_slice();
            let wts_ok = gw.iter().zip(&want_wts).all(|(a, b)| a.to_bits() == b.to_bits());
            if !ids_ok || !wts_ok {
                n_bad += 1;
                if n_bad <= 3 {
                    eprintln!("[t2] it={it} kind={kind} row={r} ids {gi:?} vs {want_ids:?} wts_ok={wts_ok}");
                }
            }
        }
    }
    assert_eq!(n_bad, 0, "route_topk_rows 位级不一致 {n_bad} 行");
    println!("[t2] {iters} 轮 × rows≤16 全通过（E={e} top_k={top_k}）");
}
