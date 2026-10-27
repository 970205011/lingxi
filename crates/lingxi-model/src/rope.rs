//! RoPE 频率表生成（模型知识，纯 host f32，无 GPU）。
//!
//! 数学等价 HF `Qwen4ExpTextRotaryEmbedding`（rope_type="default"）：
//! `text
//! inv_freq[i] = 1 / theta^(2i/rot_dim) i ∈ [0, rot_dim/2)
//! freqs[p, i] = p * inv_freq[i]
//! cos[p] = concat(cos(freqs[p]), cos(freqs[p])) # HF recomposition：cat 两份
//! sin[p] = concat(sin(freqs[p]), sin(freqs[p]))
//! `
//! 末尾的「复制两份」与 NEOX 布局（前/后半旋转）配对，正是
//! `lingxi_rope_partial_inplace_bf16` kernel 里 `(d, d+rot_dim/2)` 的读法。
//!
//! 注意：纯文本序列下 mRoPE 的三个 grid 位置相同，退化成普通 RoPE。

/// 本模型 `text_config.rope_parameters.rope_theta`。
pub const ROPE_THETA: f64 = 10_000_000.0;

/// 生成 `[max_pos, rot_dim]` 行主序的 cos/sin 表（f32）。频率用 f64 计算后转 f32，
/// 与 HF 的 float32 表逐位对齐。
pub fn build_tables(max_pos: usize, rot_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = rot_dim / 2;
    let mut inv = vec![0f64; half];
    for (i, v) in inv.iter_mut().enumerate() {
        *v = 1.0 / theta.powf((2 * i) as f64 / rot_dim as f64);
    }
    let mut cos = vec![0f32; max_pos * rot_dim];
    let mut sin = vec![0f32; max_pos * rot_dim];
    for p in 0..max_pos {
        for i in 0..half {
            let f = p as f64 * inv[i];
            let (c, s) = (f.cos() as f32, f.sin() as f32);
            // concat 两份
            cos[p * rot_dim + i] = c;
            cos[p * rot_dim + half + i] = c;
            sin[p * rot_dim + i] = s;
            sin[p * rot_dim + half + i] = s;
        }
    }
    (cos, sin)
}
