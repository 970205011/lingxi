//! 真机对拍：Q4EARN1 arena 的专家 vs `.ninfer` 产物里的 BF16 专家。
//!
//! 仅在给出环境变量时运行（60 GiB 文件，不进 CI）：
//!   LINGXI_EXPERT_ARENA=/data/models/lingxi-assets/qwen4_exp.expert-arena.q4g64.bin
//!   LINGXI_ARTIFACT=/models/qwen4_exp-artifact/qwen4_exp.ninfer
//! 在 测试机 内：cd /models/lingxi && \
//!   LINGXI_EXPERT_ARENA=... LINGXI_ARTIFACT=... cargo test -p lingxi-artifact --test expert_arena_live -- --nocapture

use lingxi_artifact::expert_arena::{decode_q4g64_f16s, BlobFormat, ExpertArena};
use lingxi_artifact::Artifact;

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn relerr(num: f64, den: f64) -> f64 {
    (num / den).sqrt()
}

#[test]
fn q4_arena_matches_bf16_artifact() {
    let (Ok(arena_path), Ok(art_path)) =
        (std::env::var("LINGXI_EXPERT_ARENA"), std::env::var("LINGXI_ARTIFACT"))
    else {
        eprintln!("跳过：未设置 LINGXI_EXPERT_ARENA / LINGXI_ARTIFACT");
        return;
    };
    let arena = ExpertArena::open(&arena_path).unwrap();
    println!(
        "arena {} | {} layers x {} experts, n_in={} n_ff={} align={} flags={}",
        arena.path.display(),
        arena.n_layers,
        arena.n_experts,
        arena.n_in,
        arena.n_ff,
        arena.slot_align,
        arena.flags
    );
    assert_eq!((arena.n_layers, arena.n_experts, arena.n_in, arena.n_ff), (48, 512, 2560, 640));

    let art = Artifact::open(&art_path).unwrap();
    let rows_gu = 1280usize;
    let rows_dn = 2560usize;
    let (h, ff) = (2560usize, 640usize);
    let gu_art = art.payload("text/layers/0/moe/routed_gate_up").unwrap();
    let dn_art = art.payload("text/layers/0/moe/routed_down").unwrap();

    for e in [0u32, 1, 17, 255, 511] {
        let blob = arena.routed(0, e).unwrap().expect("专家缺席");
        assert_eq!(blob.format, BlobFormat::Q4G64F16S, "期望 Q4G64_F16S");
        let gu = decode_q4g64_f16s(arena.bytes(&blob.gu), rows_gu as u64, h as u64).unwrap();
        let dn = decode_q4g64_f16s(arena.bytes(&blob.down), rows_dn as u64, ff as u64).unwrap();

        let (mut ngu, mut dgu, mut ndn, mut ddn) = (0f64, 0f64, 0f64, 0f64);
        for r in 0..rows_gu {
            for c in 0..h {
                let o = ((e as usize) * rows_gu + r) * h + c;
                let refv = bf16_to_f32(u16::from_le_bytes([gu_art[o * 2], gu_art[o * 2 + 1]]));
                let d = (gu[r * h + c] - refv) as f64;
                ngu += d * d;
                dgu += (refv as f64) * (refv as f64);
            }
        }
        for r in 0..rows_dn {
            for c in 0..ff {
                let o = ((e as usize) * rows_dn + r) * ff + c;
                let refv = bf16_to_f32(u16::from_le_bytes([dn_art[o * 2], dn_art[o * 2 + 1]]));
                let d = (dn[r * ff + c] - refv) as f64;
                ndn += d * d;
                ddn += (refv as f64) * (refv as f64);
            }
        }
        let (egu, edn) = (relerr(ngu, dgu), relerr(ndn, ddn));
        println!("  expert {e:3}: gu relerr={egu:.4}  dn relerr={edn:.4}");
        assert!(egu < 0.20 && edn < 0.20, "误差超出门禁: gu={egu} dn={edn}");
    }
}
