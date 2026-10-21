# Strata 0.2.2 on 2×V100 — local benchmark (2026-10-22)

This is the local, same-machine measurement used for the README/intro comparison. It is not a
bit-exact cross-engine comparison: Strata ran IQ3_XXS (about 3-bit) while lingxi runs a Q4/Q3
mixed arena, and the two benchmarks use different output modes.

## Setup

- Hardware: 2× Tesla V100-PCIE-32GB (GPU0 x16, GPU1 x8), 251 GB RAM, Xeon E5-2680 v2.
- Strata: source `origin/main` (0.2.2, engine 0.1.47), built with
  `-DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_ISA_FLOOR=avx -DCMAKE_CUDA_ARCHITECTURES=70`
  (the CPU has AVX but no AVX2; the older-CPU build is required).
- Model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (47.0 GB, two shards).
- Engine args: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --max-context 32768 --kv int8 --kv-resident 32768 --ple-io ram --mmap-experts
  --layer-split auto` on GPUs 0 and 1.
- The two model shards were pre-warmed into the page cache in chunks before the run
  (100% resident), so start-up did not do a cold 70 GB read.
- Server: `serve/server.py`, OpenAI-compatible `/v1/chat/completions`, port 8081.

At “session is up” the engine reported **100% of experts resident**, 12,288 slots per card,
6,612 MiB / 3,934 MiB VRAM free, 32K context, int8 KV, MTP draft 178.4 MiB.

## Results

Streaming requests, temperature 0, `max_tokens=128`, `reasoning_content` counted as output.
TTFT is the first content/reasoning delta; steady rate is completion tokens divided by
(total time − TTFT).

| Prompt | TTFT | Total | Output | Steady |
|---|---:|---:|---:|---:|
| Short (`The capital of France is`), run 1 | 0.139 s | 0.624 s | 42 | 86.5 tok/s |
| Short, run 2 | 0.138 s | 0.602 s | 42 | 90.5 tok/s |
| 4K document (`parity4167`), cold | 6.040 s | 7.548 s | 128 | 84.9 tok/s |
| Same 4K, prefix cache hit | 0.140 s | 1.645 s | 128 | 85.1 tok/s |
| 14K document (`needle12k`), cold | 13.488 s | 14.653 s | 128 | 109.8 tok/s |

Approximate prompt reading from the two cold runs: **~690 tok/s at 4K**, **~1,080 tok/s at 14K**.

## Lingxi reference (same box)

| Metric | lingxi (Q4/Q3 mix) | Strata 0.2.2 (IQ3_XXS) |
|---|---:|---:|
| Short decode | 43 tok/s (no MTP), ~64 (MTP) | 86–90 |
| 4K prefill | 28.7 s (4.2K, threaded) | 6.0 s |
| 33K prefill | 278.7 s | — |
| 14K prefill | — | 13.5 s |
| Expert residency | GPU1 100%, GPU0 95% | 100% on both |

## Why Strata is faster here

1. **Prompt path uses tensor cores.** The local `qsa_prompt_attn_parity` on the same V100
   measured FP32 25.7 ms/chunk → tensor-core 12.5 ms (2.06×) for int8 KV at 32K. Strata also
   uses grouped FP16 tensor-core expert GEMM for the prompt (+10% in its own report).
2. **More aggressive quantization.** IQ3_XXS leaves all experts resident and reduces bytes per
   token; lingxi's Q4/Q3 mix still moves roughly 20% of experts in diverse traffic.
3. **Host/launch overhead.** Strata's 0.1.39 ready-made engine was 2.3× its 0.1.38 source build
   on an old dual-socket box, mostly from host launches and round trips.
4. **Its long-context KV is streamed** (`--kv-resident 32768`), leaving more VRAM for experts.

## Caveats

- Different quantization and different output modes (Strata's benchmark produced reasoning
  content; lingxi's `talk`/`serve` numbers are from the documented production runs).
- Single run per cell except the short prompt; Strata's own docs note ±20% run-to-run variance.
- The CPU is an AVX-only E5-2680 v2, so Strata ran its experimental older-CPU build; its CPU
  expert kernels are not representative of a Haswell/Zen host.
- This is a reference, not a claim that the two engines are bit-comparable.
