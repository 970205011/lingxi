# lingxi

English | [简体中文](README.md)

lingxi is an inference engine written specifically for the Qwen3-Next architecture in Rust and
CUDA. Its target hardware is NVIDIA sm_70 — in our case two Tesla V100 32GB cards. It runs a
48-layer model (36 GDN linear-attention layers + 12 sparse full-attention layers, 512-expert
top-10 MoE) at interactive speed on a GPU generation that mainstream inference stacks no longer
support: 40–64 tok/s single-stream decode in production (depending on context and whether
speculative decoding is enabled), and sub-second second turns in multi-turn chat via prefix
caching. The project is released under Apache-2.0.

It is not another general-purpose serving framework. lingxi serves a very specific setting: one
or a few people, one or two older GPUs, and the desire to hold a fluent conversation with a model
of this size, run agents, and process long documents. Every design decision is organized around
one question: break each request's milliseconds apart, measure them, and only then decide whether
an optimization is worth it.

## Why this project exists

The story starts with one hardware fact and one model fact. The hardware fact: vLLM, TensorRT-LLM,
FlashInfer, and DeepGEMM have all raised their floor to sm_80; V100 has no bf16 tensor cores and
no FP4 instructions, and even int4 dp4a loses to four-way FP32 FMA on Volta. The model fact:
Qwen3-Next (GDN linear recurrence + sparse full attention every fourth layer + a 512-expert MoE +
a 95 GiB external PLE table + a built-in MTP speculative head) is a very different architecture,
and llama.cpp support for it is still maturing. This path is not unexplored: strata (a llama.cpp
fork) already runs this model on the same V100 and validates several useful techniques. But it is
not an engine tuned shape-by-shape for sm_70, and its quantization choices are more aggressive. To
optimize GDN, sparse attention, PLE, and expert residency to interactive speed on this card, a
purpose-built engine is still needed.

More concretely: the model's sparse expert weights occupy 225 GiB on disk (bf16 source). Each
token touches 480 experts, which is 1.25 GB/token in production Q4G64. Without a purpose-built
residency and transfer system, decode degenerates into random reads over the entire 241 GB expert
pool — unusable. ninfer (a production-proven C++ inference project on the same hardware) provides
the engineering skeleton that makes it run; strata runs the same model family and offers reusable
implementations of expert-hotness profiles, per-layer admission, and speculative decoding. lingxi
studies both, reuses ninfer's container format, numeric-format registry, and tensor layouts,
borrows strata's expert-hotness, admission, and speculation mechanisms, and writes the execution
layer from scratch — Rust module discipline, safe cudarc bindings, and 83 hand-written CUDA
kernels (88 `__global__` declarations including template instantiations) tuned shape by shape for
sm_70.

## Performance (all measured)

Test machine: two Tesla V100-PCIE-32GB cards (GPU0 on PCIe x16, GPU1 on x8, no NVLink), 251 GB
RAM. The model artifact is 251.5 GB, with experts stored in a RAM-disk arena in Q4G64. Every
number below comes from a bit-exactly verified configuration. The project's iron rule is: no
performance number without regression evidence.

| Scenario | Result |
|---|---|
| Single-stream decode (pinned table + dual GPU, p10) | **43 tok/s** (23.2 ms/token) |
| Decode + MTP speculation (production: kd=5, min_p=0.7) | **about 64 tok/s** |
| Dual-GPU layer split (real Chinese corpus vs single GPU) | 20.7 → **54.1 tok/s** (+161%) |
| serve, short prompt, speculative decoding | **89 tok/s** |
| 875-token prefix + speculation | **38.4 vs 29.9 tok/s** (speculative vs autoregressive) |
| 256K-context decode + MTP | **59.1 tok/s** |
| First token (875-token prompt) | about 3.8 s (on par with autoregressive) |
| First token (4.2K-token prompt, dual-GPU threaded) | **28.7 s** (sequential: 46.2 s) |
| 23.7K-token cold prefill (dual GPU) | 143.2 s (6.03 ms/token) |
| 33K-token code cold prefill (tuned) | 278.7 s (336.9 s before tuning, −17.3%) |
| 256K-token cold prefill | 3436.9 s (57.3 min), peak 31.9/27.3 GiB, no OOM |
| Re-send the same 256K prompt (prefix-cache hit) | **TTFT 0.00 s, 6.4 s total**, identical output |
| Second turn of a multi-turn chat (smart prefix-cache hit) | 28.1 s → **0.35 s (81×)** → exact re-send **0.16 s (177×)** |

The measurement methodology deserves a note, because it is part of the project's history. Early
on, the engine hit 48.5 tok/s on an English benchmark and looked close to the target; switching
to real Chinese conversations dropped it to 11.3 tok/s. Real dialogue touches 276–411 distinct
experts per layer, while the English benchmark touched only 148, so the hot-expert table built
from benchmark text fell from 97% coverage to 28%. Rebuilding the pinned table from real corpus
restored 36 tok/s; online re-pinning, making the pinned table the production default, and
re-sweeping speculation parameters brought it to today's level. All numbers above are reported
on real corpus with production defaults.

Another honest conclusion: 200 tok/s single-stream is physically unreachable on this card. The
bandwidth ledger is clear: each token must move 4.6 GB of weights, and V100's 900 GB/s HBM peak
gives a 196 tok/s theoretical ceiling at 100% bandwidth efficiency. Q4 weights are issue-limited
(about 13 decode instructions per weight byte), so V100's issue capability yields a realistic
320–505 GB/s range. Beyond that there are only two paths: aggregate bandwidth across concurrent
requests (ninfer measured 94.2 tok/s with four concurrent requests on the same hardware), or use a
smaller model. The ledger for both paths is written down in the docs; no optimism.

## Capacity and VRAM ledger (measured)

**Per-token traffic ledger** (Q4 counted at 0.5 B/parameter; source: decode-speed-profile §3.29.1):

| Item | Bytes/token |
|---|---|
| Routed experts (10 × 2.611 MB × 48 layers) | **1.25 GB** |
| Attention projections (36 GDN + 12 QSA layers) | **~1.25 GB** |
| lm_head (bf16, the single largest block) | **1.27 GB** |
| MoE shared + router (bf16) | **0.60 GB** |
| GDN recurrent state (read + write) | **0.23 GB** |
| **Total** | **≈ 4.6 GB** |

4.6 GB ÷ 900 GB/s = 5.1 ms/token = **196 tok/s**, the physical floor on V100 (at 100% bandwidth
efficiency); the measured 26.7 ms corresponds to **172 GB/s effective bandwidth (19%)**.

**Capacity ledger** (artifact headers + per-item `vram-budget`; not a total-minus-arena subtraction):

| Item | Value |
|---|---|
| Total experts | 24,576 (48 layers × 512) |
| One Q4G64 blob | 2,611,200 B (2.49 MiB) |
| One Q3G128 blob (mix64 cold experts) | 1,920,000 B (1.83 MiB) |
| All experts in Q4G64 | 59.8 GiB (arena file 60.25 GiB) |
| Mixed-precision arena (hot Q4 + cold Q3) | 46.39 GiB (−23%) |
| Q4-only device slots | 7000 slots = 17.0 GiB; 9000 slots = 21.9 GiB |
| Non-expert fixed footprint (32K, autotune ledger) | GPU0 **10.3 GiB** = weights 5.8 + KV 0.43 + GDN 0.05 + reserve 3.0 + margin 1.0; GPU1 **7.3 GiB** = 3.4 + 0.43 + 0.05 + 2.4 + 1.0 |
| Device expert slots (mix64, 32K) | GPU0 **11,423** (Q4 1536 + Q3 9,887/10,752); GPU1 **12,288** (all resident) |
| KV (logical / device window) | logical 24 KB/token: 32K ≈ 786 MB, 256K ≈ 6.3 GB; with `LINGXI_QSA_PAGED=1` the device keeps only a 32K window (0.43 GiB/card in the autotune ledger), the rest lives in the host pinned block pool |
| Indexer | 3 KB/token: 256K ≈ 786 MB, plus 1/4 for pooled |
| GDN/PLE session state | ≈113 MB per session, independent of context length |
| PLE table | bf16 95.4 GiB (host) / FP8 51.2 GB (/dev/shm) |

The table is printed by the new `lingxi-bench vram-budget <artifact> <arena> 24 32768`:
weights come from the catalog tensor by tensor, expert slots from each arena blob's actual
Q4/Q3 format, and KV/GDN/reserve/margin from code constants — no weights are loaded and no
VRAM is allocated. This also fixes the mixed-arena slot estimate: the old autotune used the
default Q4 blob (2.49 MB) for every slot, over-sizing Q3 slots and **undercounting by about
2,000–2,500 slots per card**.

**The boundary for full expert residency**:

- GPU1 already holds all of its 12,288 experts (23.0 GiB, fully resident).
- GPU0 is short by **865 cold experts (1.58 GiB)**. Moving the 32K KV to host saves 0.43 GiB;
  tightening reserve/margin by about 1.2 GiB, or demoting some hot Q4 experts to Q3, lets GPU0
  hold everything too.
- At 256K the logical KV is **3.5 GiB per card**; `LINGXI_QSA_PAGED=1` must keep only a 32K
  resident window, otherwise KV crowds out expert slots. Paging is already implemented as a
  block table + host pinned block pool + LRU sink/prefetch; not all KV has to live in VRAM.
- To be honest: short-prompt hit rate is already **100%**; full residency mainly helps long-tail
  and diverse traffic. It will not turn 33 ms/token into 10 ms. The real speed gap is prefill
  tensor cores and host scheduling, discussed below.

**Measured VRAM peaks** (nvidia-smi / startup banners):

| Configuration | Slots | Measured VRAM |
|---|---|---|
| Single GPU, Q4, 9000 slots fully loaded | 9000 | **31.3 / 32 GiB** (740 MiB free) |
| Dual GPU, 32K + MTP (pinned table) | 9600 / 10000 | **29.9 / 27.8 GiB** |
| Dual GPU, 32K + speculation, 875-token prefill peak | 8800 / 9200 | **31.3 / 28.6 GiB** |
| Dual GPU, 256K recipe at startup | 7200 / 7600 | **23.7 / 23.2 GiB** |
| 256K, 47K prefill peak | 7200 / 7600 | **27.7 / 26.8 GiB** |
| 256K, 250.6K prefill peak | 7200 / 7600 | **31.9 / 27.3 GiB** (no OOM) |

The same ledger can be recomputed as:

```text
VRAM ≈ slots × bytes per slot
     + dense/head (W4 ≈ 2.4 GiB; bf16 ≈ 9.2 GiB)
     + KV (24 KB/token) + indexer (3 KB/token)
     + GDN/PLE state (≈113 MB/session)
     + MTP (draft 835 MiB + Q4 experts 1.3 GiB)
     + workspace / cuBLAS / driver (≈2–3 GiB)
```

9000 × 2.49 MiB = 21.9 GiB, plus the measured non-arena 9.4 GiB ≈ 31.3 GiB, matching the sample.

## What we built

This section is the core of the project. Each subsystem is described with measured comparisons,
and negative results are documented alongside positive ones.

**One-page results table** (all same-machine A/B measurements; details in docs/):

| Change | Baseline → after | Result |
|---|---|---|
| Pin the whole arena | p50 8.8 → **34.3 tok/s** | +290% |
| Offline pinned table N=200 | 30.43 → **24.36 ms/token** | −20% |
| Online re-pinning | hit rate 84.8% → **88.2%**; 32.5 → **38.1 tok/s** | +17% |
| Mixed-precision arena (hot Q4 / cold Q3) | 60.3 → **46.4 GiB** | −23%, end-to-end timing unchanged |
| MTP probability gate min_p=0.7 | 18.2 → **27.6 tok/s** | +52% |
| Full MTP chain (gate + two replay cuts) | 17.0 → **about 64 tok/s** | +276% |
| Dual-GPU layer split | 20.7 → **54.1 tok/s** | +161% |
| Threaded prefill (4.2K prompt) | 46.2 → **28.7 s** | −38% |
| Multi-turn prefix cache | 28.1 → **0.35 s** | 81× |
| Exact re-send | 28.1 → **0.16 s** | 177× |
| 256K cold prefill | **3436.9 s (57.3 min)** | peak 31.9 / 27.3 GiB, no OOM |
| 256K prefix hit | TTFT **0.00 s**, **6.4 s** total | 250,552 tokens reused |

### Model layout and dual-GPU split

The model is a 48-layer hybrid: layers where `i % 4 == 3` (12 layers) use QSA sparse full
attention, the other 36 use GDN linear recurrence (`crates/lingxi-model/src/lib.rs::layer_kind`,
unchanged since the first version). Intra-layer tensors are pre-arranged offline for the kernels'
access patterns (`row-split-k128-v1`, K aligned to 128, planes aligned to 256 B). Runtime does no
transpose, no dequantization, and no fusion.

The default dual-GPU split is at **lo=24**: GPU0 runs layers 0–23 and holds embedding / head /
MTP, GPU1 runs layers 24–47; KV and GDN state are split by layer. Slot counts are computed per
card by vram-autotune (production measurement: GPU0 8984 / GPU1 10218 slots). 24 is not arbitrary:
28 was also tried (cold prefill −5.2%), but the 33K cold prefill improvement from 336.9 s to
278.7 s (−17.3%) comes from **lo=24 + chunk 1024**; chunk 1536 OOMs in the same configuration
(GPU0 32.4 G). On real corpus the dual-GPU path goes from 20.7 to 54.1 tok/s.

Two layout changes that looked promising were rejected by measurement: reordering the arena at
expert granularity (real gain ≈0; artifact format unchanged), and layer-major prefill (33K is 7.8%
slower; off by default).

### Numerical verification: "it emits tokens" was never the bar

Every component — GR gated residual, MoE, QSA attention and indexer block selection, GDN linear
attention, the 95 GiB PLE table — was cross-checked against the HuggingFace reference bf16 forward,
and all 48 layers passed the layer-wise golden comparison (most layers within 1.5e-2 on 5/5 rows;
a few out-of-range rows in deeper layers were attributed to exact bf16 logit ties in the MoE
router, and re-running HF on the same input disproved an engine defect). On top of that, every
performance change must pass a bit-level criterion: the first token id for a fixed prompt must be
11751, and the greedy sequence must be bit-identical. This hardened into three verification levels
(T1 per-kernel vs. torch reference, T2 layer-wise golden, T3 end-to-end greedy) plus four guard
gates (mixed-arena byte-level equality, three serve×paged scenarios, and a loud error when the
paged window is exhausted). Every refactor — including two thousand-line file splits — ships only
with all gates green.

The most representative bug this system caught was the MTP speculative head port. Draft acceptance
started at 0/64; nine structural hypotheses were rejected black-box, and white-box comparison of
three components all "passed" (later shown to be circular reasoning: comparing Rust against the
same wrong artifact weights). Two prior criteria broke the case: "RMSNorm gamma must be all
positive" caught the converter missing the `wplus +1` on MtpStemNorm (this model family stores
γ−1), and "the qkv row multiset must equal [q;k;v]" caught a row order that dropped 256 q rows and
duplicated 256 k rows. With both fixed, acceptance went 0 → 0.906 → 1.000. Both criteria are now
built-in self-checks in the converter (`tools/convert/hf2lingxi.py`).

### Expert-weight residency: the real battleground

The life-or-death issue in MoE inference is expert capacity and movement; this is where lingxi
invests the most:

Expert weights are stored in a RAM-disk arena compressed as Q4G64 (2.61 MB per blob, 7.53× smaller
than bf16), and the GPU keeps only an LRU slot cache. At startup, vram-autotune computes slots per
card from "weights + KV + GDN state + reserve" (measured dual-GPU: GPU0 8984 / GPU1 10218 slots),
with no hand tuning. Pinning is decided by medium: if /proc/mounts says the arena is on a RAM disk,
the whole arena is pinned by default (measured p50 from 8.8 to 34.3 tok/s — a 4× difference; without
pinning, p50/p10 dispersion reaches 139%). On spinning disks it is forbidden, because a whole-arena
`cudaHostRegister` can wedge the drive firmware (a rule paid for by a real incident on October 14;
only "whole arena or 0" is accepted — intermediate values fail loudly at startup).

Which experts stay resident is a data decision, not a guess: offline real-traffic statistics of
per-layer routing frequency determine the hottest set pinned at startup (making the pinned table
the production default cut p90 from 36.84 ms to 24.82 ms and largely removed the tail). The engine
also accumulates frequency tables while serving and supports online adaptive re-pinning (measured
hit rate 84.8% → 88.2%; +17% on real corpus). One data finding worth noting: the cost of a slot
miss is not 2.6 MB divided by PCIe bandwidth (about 200 µs), but 370–450 µs of host orchestration
and synchronization stalls; cold start and steady state must be measured separately (at N=16 a
miss costs 11 ms/token, while steady state is about 1 ms).

On top of Q4 there is a mixed-precision layer: experts are placed **slot by slot** by routing
frequency — hot experts Q4G64, cold experts Q3G128 (3-bit) — shrinking the arena from 60.3 GiB
(all Q4) to 46.4 GiB (−23%) with end-to-end timing equal to the all-Q4 baseline. Mixed precision
is not just picking two formats: it has a full verification chain. A byte-level gate guarantees
the mixed arena is byte-identical to the single-format originals (the quantizer is deterministic,
so it must be). The Q3 bitstream (dense LSB-first) has separate decode/prefill/batch kernels. The
consumer uses a dual-view loop to dispatch by format segment. On top of that, a one-shot GPTQ-style
offline compensation (OBQ column propagation, TF32) measures error in output space via Hessian
eigendecomposition (not self-deceiving weight-space relative L2), significantly reducing error at
the same bpw. But since timing is identical to RTN, the default stays RTN and GPTQ remains a
drop-in option — "never silently flip the production default" is another project rule.

### MTP speculative decoding: from a guaranteed loss to +140%

The model ships with an MTP head, but "shipped" does not mean "usable." The economics of
speculation were pushed to the limit: sequential decode costs 26.8 ms per row, while each extra
verification row in speculation pays another expert H2D (drafts are all new tokens, so slot miss
rate doubles). In a measured speculative round, 99 of 218 ms (45%) was expert movement — so
"eliminate expert H2D" is not one optimization among many; it is the ticket to speculation. The
ledger also proved a counterintuitive result: with the row-cost structure unchanged, speculation
loses at any draft length k and any acceptance rate p (kd=8 with 5.8/8 acceptance was slower,
because the marginal cost of an extra row exceeds the tokens it wins).

These numbers are strongly VRAM-dependent; the conclusion holds only under the cost structure
where experts do not fit and each extra verification row pays another H2D. The ledger: all experts
Q4G64 ≈ 60.3 GiB, plus dense/head (W4 ≈ 2.4 GiB), MTP (≈2.1 GiB), 32K KV (≈1.1 GiB), GDN state
and workspace/driver reserve (≈3.4 GiB) — about 69 GiB on one GPU, or about 77 GiB with 256K KV.
So **a single 80 GB card is the entry point for full expert residency** (tight at 32K, very tight
at 256K); 96 GB is comfortable. A dual-GPU layer split needs about 35–41 GiB per card, so
**2×48 GB is enough**. At that capacity, the marginal cost of a verification row returns to pure
compute, and speculation never loses at any k/p — "lossless" in the speed sense. Output quality is
lossless anyway (probability gate + zero-rollback commit; the generated sequence is token-for-token
identical). The remaining ceiling is the MTP head itself: it was trained for only one step, and
first-position acceptance in the self-fed chain collapses to 0.57, so the realistic gain is still
around 1.35×, not the 2× in the paper.

Four things ultimately turned speculation around: (1) a probability gate (min_p=0.7, using the
draft's own top-1 probability as the stop signal; 18.2 → 27.6 tok/s, cutting only useless rows and
keeping the generated sequence token-for-token identical); (2) the precondition for that gate —
after the pinned table suppressed expert DMA, kd=5 finally beat kd=6; (3) two cuts in the MTP
replay path (drop the head computation during replay, −59% replay time; replace prob1 with a cheap
top-K approximation, 19 exponentials instead of 248K); (4) discovering that the MTP head was
trained for only one step — first-position acceptance in the self-fed chain drops to 0.57 (0.93
when fed the trunk's ground truth), which ended the "longer chain" fantasy and set the honest
expectation of 1.35× rather than the paper's 2×. Cumulatively, speculation went from 17.0 tok/s
(worse than sequential) to about 64 tok/s.

### Long context and prefill

Three long-context surgeries all target attention. The first was an algorithmic bug: the sparse
attention kernel's inner loop recomputed the dot product for every output dimension, O(head_dim²·Tk)
instead of O(head_dim·Tk), a 256× redundancy that consumed 44% of total time. It was found by
per-kernel inspection when "total time did not match the sum of microbenchmarks" (microbenchmark
tk=4096: 32.6 ms → 1.3 ms). The second changed attention from "scan all KV row by row" to "gather
only the selected blocks" (512 blocks = 2048 tokens): single-layer attention at 8K context went
from 143 ms and growing linearly with context to a constant ~28 ms. The third replaced QSA block
selection with streaming radix selection (O(nblocks) histogram instead of O(nblocks²) full ranking,
zero scores buffer) plus parallel scoring, cutting select+fill from 293–335 ms/layer to 41 ms/layer
(7–8×); then q-head merging for K/V reads (12 q-heads share 2 K/V copies, saving 49 GB of traffic
per layer) cut 32K cold prefill by another 10.4%.

The key on the prefill side is the 95 GiB PLE table. It once lived on a spinning disk, with 120
random preads per token and 60 ms per cold row read (11.3 ms warm, a 5.3× gap) — the real cause of
"prefill gets slower as it runs." The fix extracts the official FP8 tier (51.2 GB) into /dev/shm,
adds four-level resolution (explicit > shm FP8 > shm IQ4_NL > disk bf16 with a loud fallback
banner), and combines chunked prefill (peak VRAM O(chunk); 33K code cold prefill tuned to 278.7 s
with chunk 1024 + layer split 24, 42% fewer misses) with a background sequential prewarm thread.
The measured 256K full prefill is 57.3 minutes with peak 31.9/27.3 GiB and no OOM. Re-sending the
same prompt reuses 250,552 tokens from the prefix cache, giving zero TTFT and a full answer in 6.4
s. The honest boundary is documented too: full KV above 250K OOMs on this 64 GB of VRAM; a single
request should stay ≤32–64K, and longer contexts rely on prefix caching and multiple turns.

### Serving layer

serve is a dependency-free hand-written HTTP/SSE server (OpenAI-compatible
/v1/chat/completions, with reasoning_content, tool calls, streaming incremental detokenization,
and long-prefill keepalive). Cross-request prefix caching has two modes: exact matching does pure
prefill prefixes only (466× TTFT); "smart" mode falls back at message boundaries and checkpoints
the whole GDN/PLE state (~115 MB per copy), giving 81× on the second turn of multi-turn chat.
Inactive sessions spill to disk (atomic writes + LRU) and recover in seconds after restart. Since
v1.5, concurrent multi-session serving is supported: a connection thread and an engine thread step
in rotation, each materialized session owns an independent CUDA Graph epoch, and switching sessions
moves no data. Speculative decoding is enabled per request — cold requests speculate, while
prefix-cache hits fall back to autoregressive automatically (measured: same prompt second time
0.26 s, spec_rounds=0).

### Measurement and falsification: the project's methodology asset

Reading through the engineering docs, roughly as much ink is spent on "how to measure" and "which
paths were falsified" as on the wins. This is deliberate: every optimization that "looked
justified" (scratch pools to kill alloc storms, fusing GR's 8 kernels, shared-x Q4 GEMV, raising
expert slots to 9500, dual-GPU P2P, CPU expert pools, dp4a, putting attention into CUDA Graph, …)
was fully implemented and measured first; anything with zero gain or slower was documented and
rejected so nobody repeats it. The measurement tools themselves were repeatedly falsified too:
SKIP block-stripping systematically overestimates the stripped segment (it removes synchronization
barriers along with it); segmented timing is 26% high due to drain bias; forward benchmarks with
repeated prompt tokens cannot see expert DMA; p10 masks long-tail spikes (one 8.65 ms disk seek
when PLE changed topics made the true mean 30% worse than p10); and element-wise synchronous
readback turns the observation itself into a race source. These lessons hardened into rules: A/B
must run interleaved in the same process and state, md5 must confirm the target binary was swapped
in, and ablations that change the math can only establish "whether," not "how many milliseconds."

## Relationship to existing solutions

| Dimension | vLLM / TRT-LLM / FlashInfer | llama.cpp | lingxi |
|---|---|---|---|
| sm_70 (V100) support | Not supported (requires sm_80+) | Runs | First-class, all kernels tuned per shape |
| Qwen3-Next semantics (GDN/PLE/MTP) | — | Still maturing | Complete, 48 layers verified layer by layer |
| MTP speculative decoding | Architecture unsupported | None | Yes (probability gate + zero-rollback commit) |
| Per-slot mixed-precision experts + GPTQ compensation | — | — | Yes, backed by byte-level comparison gates |
| Paged KV beyond VRAM window | Yes (new-GPU path) | None | Yes (block-table indirection + pinned block pool) |
| Numerical verification | Accuracy tests | Sampled comparison | T1 per-kernel / T2 per-layer / T3 end-to-end bit-exact |

### External reference: what the 200 tok/s number means

We did build an external reference on the same machine and model, but the measurement basis must
be aligned first: strata's published 200.8 tok/s is a **peak**; in the same record the A/B
benchmark median is **104.5** and the real agent-task median is **159.2**. It runs IQ3_S at
82.8 GB (~3.4 bit), moving about **2.55 GB/token** by parameter count × bit width. lingxi's
production configuration is **3.3 GB/token** by the same accounting, with a real-corpus dual-GPU
steady state of **54.1 tok/s**. Comparing 200.8 peak with 54.1 steady overstates the gap; on
medians it is 104.5 vs 54.1 ≈ **1.9×**, of which about **1.3×** comes from more aggressive
quantization (fewer bytes) and the remaining **1.5×** from speculative-chain depth, host
scheduling, and kernel efficiency: strata runs an 8-deep chain at 63–78% acceptance on real text
(93–99% on repetitive output), while our MTP head collapses to 0.57 at position 1 and accepts only
about 3.1 tokens per round.

Current upstream V100 community runs put the peak in context: one card with IQ3_XXS decodes at
**54–72 tok/s**, Q2_0 at **76.4**, IQ2_XS at **71.2**, UD-IQ4_XS at **26.3**; a V100 + RTX 4070
layer split reaches **75–85**. Our dual-V100 Q4 result of 54.1 is therefore in the same class as
strata's single-V100 IQ3_XXS, and 200.8 is a tuned peak, not a typical V100 number.

The real gap is **prefill**: strata reads a 32K prompt at about **1,200–1,600 tok/s** on a V100,
while our 33K cold prefill takes 278.7 s ≈ **118 tok/s**, roughly 10× slower. The main reasons are
its FP16 tensor-core prompt attention (+15–18%) and grouped expert GEMM (+10%), whereas our prefill
is still FP32 SIMT compute-bound. We rebuilt strata 0.2.2 locally and ran its
`qsa_prompt_attn_parity` on a V100 (int8 KV, 32K context, 2048 queries): **FP32 25.7 ms/chunk →
tensor cores 12.5 ms (2.06×)**, with a 2.3e-6 error against FP64. The other structural difference
is quantization: a 2-bit file leaves almost all experts resident, with only about 0.7 experts per
layer coming from CPU/PCIe; our Q4 mixed arena still moves about 20% of the experts.

The 26.8 ms per token also decomposes: expert H2D **7.0 ms (26%)** (routing-timing-bound; can
only be overlapped, not removed), real kernels **11.6 ms (43%)**, host **3.4 ms (13%)**,
synchronization/overhead **4.8 ms (18%)**. The HBM floor is 4.6 GB/token ÷ 900 GB/s = **5.1 ms
= 196 tok/s**, so the sequential single-stream path physically cannot reach 200; even with
perfect kernel efficiency (11.6 ms) it is only **86 tok/s**. Beyond that the only paths are
aggregating bandwidth across concurrency or using a smaller quantization/model — both ledgers
are in docs/decode-speed-profile.md.

We previously ran strata in this environment as a numerical oracle, but that was not the same
quantization tier or timing basis as the 200.8 peak. **A two-engine, same-prompt, same-model,
same-timing head-to-head has not been done yet** — it is the first task for the next version.
This version focuses on making single/dual 32G interactive serving solid and open-sourcing the
engineering record and verification system.

## Ecosystem compatibility and deployment

Artifacts use the ninfer v2 container as their base (magic `NINFER\0\x02`; format spec in
docs/FORMAT.md). **Reading**: upstream pure-ninfer artifacts (`.ninfer`) work out of the box.
**Naming**: artifacts generated or consumed under lingxi content conventions (qwen4 tensor
directory, wplus +1, PLE addressing, mixed-precision format codes) use the `.lingxi` suffix. The
container itself is unchanged; the reader identifies artifacts by magic, not by suffix. Upstream
ninfer can parse the container but does not understand these content conventions, so `.lingxi`
means "a NINFER v2 artifact meant for lingxi." The existing 253 GB main artifact keeps its
`.ninfer` name; no on-disk migration. The API is OpenAI-compatible, so existing clients only need
a new base_url; tokenization uses the model's own HuggingFace tokenizer.json. The offline toolchain
(tools/convert, tools/arena-gen) can rebuild all artifacts from an HF checkpoint, including the
mixed-precision arena and the GPTQ-compensated repacking pipeline (guards at every step, resumable,
md5 recorded).

Deployment supports one or two GPUs. A single V100 32G is fully functional (9000 slots / 32K
context / speculation available); dual GPU defaults to a layer split, +161% on real corpus, and
can serve 256K context with the long-context profile. Day-to-day use `./start.sh`; production
scripts live in tools/deploy/ (dual instances with a proxy, single long-context instance), with
log rotation, pid management, and graceful shutdown.

## Quick start

```bash
make build        # cargo build --release (requires nvcc / CUDA 12.8 + rustc)
./start.sh        # production launcher (hardened env, talk smoke test)
./start.sh ./bin/lingxi-bench serve <artifact.lingxi or .ninfer> [port]   # OpenAI-compatible API
make regress      # QA criteria regression (3 end-to-end criteria)
make gates        # guard gates (mixed-arena comparison / serve×paged / window-exhaustion error)
```

```bash
curl http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "lingxi",
  "messages": [{"role": "user", "content": "The capital of France is"}]
}'
```

Artifacts (`.lingxi`, or legacy `.ninfer`; about 253 GB) are not checked in; specify the path via
`LINGXI_ARTIFACT` or a command-line argument. All runtime parameters are environment variables,
registered in crates/lingxi-model/src/probes.rs (180+ switches with type/default/purpose/coupling;
a typo is named at startup and dangerous combinations are rejected — enforced by guard tests).
Building artifacts from scratch: tools/convert/hf2lingxi.py.

## Project structure

```
crates/
  lingxi-ids        identity and connection layer: the single source of cross-module contracts
  lingxi-artifact   .lingxi / .ninfer container (NINFER v2) reader/writer + load-time structural assertions
  lingxi-plan       builds runtime memory layout from the catalog (addresses only, no compute)
  lingxi-kernels    CUDA C++ → cubin (build.rs invokes nvcc) + cudarc load/launch
  lingxi-model      forward logic + static model description + tensor-role derivation + switch registry
  lingxi-driver     slot pool · admission · decode loop · MTP · prefix sharing · KV budget ledger
  lingxi-server     HTTP/SSE (OpenAI-compatible) · tokenization
  lingxi-bench      artifact inspection · benchmarks · golden comparison · serve engine
```

Dependencies flow one way with no cycles (ids → artifact/plan/kernels → model → driver → server);
unsafe is allowed only in lingxi-kernels. Design and debugging discipline: ARCHITECTURE.md.

## Documentation index

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture constraints, module boundaries, verification levels |
| [docs/FORMAT.md](docs/FORMAT.md) | `.lingxi` / `.ninfer` container and numeric-format manual (required for model conversion) |
| [docs/decode-speed-profile.md](docs/decode-speed-profile.md) | Per-item decode profile and A/B records |
| [docs/speedup-audit.md](docs/speedup-audit.md) | Item-by-item optimization audit (including falsified ideas) |
| [docs/inference-framework-design.md](docs/inference-framework-design.md) | Framework design and ninfer/strata trade-offs |
| [docs/weight-compression-plan.md](docs/weight-compression-plan.md) | W4 / mixed-precision / GPTQ roadmap |
| [docs/strata-mtp-expert-cache-plan.md](docs/strata-mtp-expert-cache-plan.md) | MTP and expert-cache plan |
| [docs/PLAN.md](docs/PLAN.md) / [docs/PLAN_NEXT.md](docs/PLAN_NEXT.md) | Full engineering decisions and milestones |
| [docs/BACKLOG.md](docs/BACKLOG.md) | Deduplicated backlog and known issues |
| [docs/HANDOVER.md](docs/HANDOVER.md) | Operations and troubleshooting manual |
| [docs/ASSETS.md](docs/ASSETS.md) | Asset inventory and regeneration commands |
| [docs/bench-logs/](docs/bench-logs/) | Raw logs for key measurements |

> `$MODELS`, `$LINGXI_ASSETS`, `$DATA`, and `$CUDA_LIBS` in the docs are path placeholders for
> the model-artifact root, asset root, data disk, and CUDA runtime library directory.

## References and acknowledgements

The artifact container format, numeric-format registry, tensor layouts, and converter reuse
ninfer's work (lingxi-artifact is our own compatible reader; it opens a 252 GB artifact in 0.01 s).
Expert-hotness profiling and speculative-decoding algorithms reference strata. The quantization
format baseline comes from llama.cpp. Mixed-precision compensation uses GPTQ/OBQ algorithms. The
CUDA Graph timing-probe mechanism references DeepInfer. Ecosystem libraries: cudarc, axum,
tokenizers. Qwen3-Next model semantics come from orcarouter/Qwen3.8-Flash-Next-Uncensored on HF.

## License

Apache License 2.0 (see LICENSE). Third-party dependencies retain their own licenses.
