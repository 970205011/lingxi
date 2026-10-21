#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GDN 分阶段对拍：引擎 /tmp/lingxi_gdn_*.f32 vs HF /tmp/hfprobe/l0_hf_gdn_*."""
import numpy as np

T = 5
NK, NV, HK, HV = 16, 48, 128, 128
KEY_DIM = NK * HK          # 2048
VAL_DIM = NV * HV          # 6144
CONV_DIM = 2 * KEY_DIM + VAL_DIM  # 10240
REP = NV // NK

def rd(p):
    return np.fromfile(p, dtype=np.float32)

def rel(a, b):
    return float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)) /
                 (np.linalg.norm(b.reshape(-1)) + 1e-12))

# 1) z / a / b 投影（GEMM 级，期望 ~1e-3）
ez = rd("/tmp/lingxi_gdn_z.f32").reshape(T, VAL_DIM)
hz = rd("/tmp/hfprobe/l0_hf_gdn_z_proj.f32").reshape(T, VAL_DIM)
ea = rd("/tmp/lingxi_gdn_a.f32").reshape(T, NV)
ha = rd("/tmp/hfprobe/l0_hf_gdn_a_proj.f32").reshape(T, NV)
eb = rd("/tmp/lingxi_gdn_b.f32").reshape(T, NV)
hb = rd("/tmp/hfprobe/l0_hf_gdn_b_proj.f32").reshape(T, NV)
print(f"z proj rel={rel(ez,hz):.3e}")
print(f"a proj rel={rel(ea,ha):.3e}")
print(f"b proj rel={rel(eb,hb):.3e}")

# 2a) 卷积前 qkv 直接比（GEMM 级）
eqkv0 = rd("/tmp/lingxi_gdn_qkv_preconv.f32").reshape(T, CONV_DIM)
hqkv = rd("/tmp/hfprobe/l0_hf_gdn_qkv_proj.f32").reshape(T, CONV_DIM)
print(f"qkv preconv rel={rel(eqkv0,hqkv):.3e}")
for name, off, ln in [("q", 0, KEY_DIM), ("k", KEY_DIM, KEY_DIM), ("v", 2*KEY_DIM, VAL_DIM)]:
    print(f"  preconv[{name}] rel={rel(eqkv0[:,off:off+ln], hqkv[:,off:off+ln]):.3e}")

# 2b) conv：用 torch bf16 精确复刻 causal_conv1d_fn（F.conv1d 左 pad → silu）
import torch
wconv = rd("/tmp/hfprobe/l0_hf_gdn_conv_w.f32").reshape(CONV_DIM, 4)  # [C,K]
K = wconv.shape[1]
x = torch.from_numpy(hqkv).to(torch.bfloat16).T.reshape(1, CONV_DIM, T)
wt = torch.from_numpy(wconv).to(torch.bfloat16).unsqueeze(1)
co = torch.nn.functional.conv1d(x, wt, padding=K - 1, groups=CONV_DIM)[:, :, :T]
co = torch.nn.functional.silu(co)
hconv = co.to(torch.float32).numpy().T.reshape(T, CONV_DIM)
econv = rd("/tmp/lingxi_gdn_conv.f32").reshape(T, CONV_DIM)
print(f"conv(silu) rel={rel(econv,hconv):.3e}  (f64 近似重建 HF)")
# 分段：q / k / v
for name, off, ln in [("q", 0, KEY_DIM), ("k", KEY_DIM, KEY_DIM), ("v", 2*KEY_DIM, VAL_DIM)]:
    print(f"  conv[{name}] rel={rel(econv[:,off:off+ln], hconv[:,off:off+ln]):.3e}")

# 3) qn / kn：HF chunk 入口 q/k 是 [B,T,NK,HK] 未归一化，repeat_interleave 到 NV
hq = rd("/tmp/hfprobe/l0_hf_gdn_chunk_q.f32").reshape(T, NV, HK)
hk_ = rd("/tmp/hfprobe/l0_hf_gdn_chunk_k.f32").reshape(T, NV, HK)
def l2n(x):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
hqn = l2n(hq) / np.sqrt(HK)
hkn = l2n(hk_)
eqn = rd("/tmp/lingxi_gdn_qn.f32").reshape(T, NV, HK)
ekn = rd("/tmp/lingxi_gdn_kn.f32").reshape(T, NV, HK)
print(f"qn rel={rel(eqn,hqn):.3e}")
print(f"kn rel={rel(ekn,hkn):.3e}")

# 4) beta / g（HF 落盘原始形状 [1,T,NV]）
hbeta = rd("/tmp/hfprobe/l0_hf_gdn_chunk_beta.f32").reshape(T, NV)
ebeta = rd("/tmp/lingxi_gdn_beta.f32").reshape(T, NV)
print(f"beta rel={rel(ebeta,hbeta):.3e}")
hg_log = rd("/tmp/hfprobe/l0_hf_gdn_chunk_g.f32").reshape(T, NV)
hg_decay = np.exp(hg_log.astype(np.float64)).astype(np.float32)
eg = rd("/tmp/lingxi_gdn_g.f32").reshape(T, NV)
print(f"g(decay) rel={rel(eg,hg_decay):.3e}  "
      f"max|d|={np.max(np.abs(eg-hg_decay)):.3e}")

# 5) 递推输出（norm 前）
hout = rd("/tmp/hfprobe/l0_hf_gdn_chunk_out.f32").reshape(T, NV, HV)
eout = rd("/tmp/lingxi_gdn_out.f32").reshape(T, NV, HV)
print(f"recurrent out(pre-norm) rel={rel(eout,hout):.3e}")
for t in range(T):
    print(f"   row{t} rel={rel(eout[t],hout[t]):.3e}")

# 6) 用引擎 conv 喂 numpy 参考递推（f32），验证引擎 host 核心自身
conv = econv
v = conv[:, 2*KEY_DIM:].reshape(T, NV, HV)
q = eqn; k = ekn
S = np.zeros((NV, HK, HV), dtype=np.float64)
ref = np.zeros((T, NV, HV), dtype=np.float64)
beta = ebeta.astype(np.float64); decay = eg.astype(np.float64)
for t in range(T):
    for h in range(NV):
        S[h] *= decay[t, h]
        kv = S[h].T @ k[t, h].astype(np.float64)
        delta = (v[t, h].astype(np.float64) - kv) * beta[t, h]
        S[h] += np.outer(k[t, h].astype(np.float64), delta)
        ref[t, h] = S[h].T @ q[t, h].astype(np.float64)
print(f"引擎自身递推 vs engine dump out rel={rel(ref.astype(np.float32), eout):.3e}")
