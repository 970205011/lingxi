# lingxi

English | [简体中文](README.md)

lingxi is an inference engine for the **NVIDIA sm_70 (Tesla V100)**, written in Rust and CUDA. It
runs a 48-layer Qwen3-Next model (36 GDN linear-attention layers + 12 sparse full-attention
layers, 512 experts, top-10 per token) at interactive speed on 2×V100 32G: **43 tok/s**
single-stream decode, about **64 tok/s** with MTP speculation, and 256K context. Apache-2.0.

It is not a general-purpose serving framework. It serves one concrete setting: one or two people,
one or two older GPUs, holding a fluent conversation with a model of this size, running agents,
and processing long documents.

[One-page introduction (HTML)](docs/intro.html) · [GitHub](https://github.com/970205011/lingxi) · [Documentation](#documentation)

## Performance (measured)

Test machine: 2× Tesla V100-PCIE-32GB (GPU0 x16, GPU1 x8, no NVLink), 251 GB RAM. Every number
comes from a configuration that passed the bit-level criteria.

| Scenario | Result |
|---|---|
| Single-stream decode (dual GPU, p10) | **43 tok/s** (23.2 ms/token) |
| Decode + MTP (kd=5, min_p=0.7) | **about 64 tok/s** |
| Dual-GPU layer split (real Chinese corpus vs single GPU) | 20.7 → **54.1 tok/s** (+161%) |
| serve, short prompt, speculative | **89 tok/s** |
| 875-token prefix + speculation | **38.4 vs 29.9 tok/s** |
| 256K context decode + MTP | **59.1 tok/s** |
| First token (4.2K prompt, dual-GPU threaded) | **28.7 s** (sequential 46.2 s) |
| 23.7K-token cold prefill | 143.2 s (6.03 ms/token) |
| 33K-token cold prefill (tuned) | 278.7 s (−17.3%) |
| 256K-token cold prefill | 3436.9 s (57.3 min), peak 31.9/27.3 GiB, no OOM |
| Same 256K prompt re-sent (prefix hit) | **TTFT 0.00 s, 6.4 s total** |
| Second turn of multi-turn chat | 28.1 s → **0.35 s (81×)** → exact re-send **0.16 s (177×)** |

Single-stream 200 tok/s is physically unreachable on V100: each token moves about 4.6 GB, and the
900 GB/s peak gives a 196 tok/s theoretical floor at 100% bandwidth efficiency. Beyond that only
concurrency (aggregating bandwidth) or a smaller model helps.

## Capabilities

- **Expert residency**: Q4G64 arena (60.25 GiB) + GPU LRU slots + offline pinned hot table +
  online re-pinning; a mixed arena (hot Q4 / cold Q3) shrinks it to 46.39 GiB with identical
  end-to-end timing.
- **MTP speculation**: probability gate (min_p=0.7) + zero-rollback commit; generated sequence is
  token-for-token identical.
- **Long context**: QSA sparse attention gathers only selected blocks; paged KV (32K device
  window + host pinned block pool); FP8 PLE table in RAM; 256K cold prefill in 57.3 minutes and
  zero TTFT on a prefix hit.
- **Serving**: hand-written OpenAI-compatible HTTP/SSE; exact and smart prefix caching;
  concurrent sessions; inactive sessions spill to disk.
- **Numerical verification**: T1 per-kernel vs torch, T2 layer-wise golden, T3 end-to-end greedy
  bit-exact, plus four guard gates.
- **Artifact compatibility**: reads and writes the ninfer v2 container; upstream `.ninfer` works
  out of the box, lingxi content conventions use the `.lingxi` suffix.

## VRAM and deployment

| Configuration | Measured VRAM | Notes |
|---|---|---|
| Single GPU, Q4, 9000 slots | 31.3 / 32 GiB | 32K context |
| Dual GPU, 32K + MTP | 29.9 / 27.8 GiB | 9600 / 10000 slots |
| Dual GPU, 256K startup | 23.7 / 23.2 GiB | 7200 / 7600 slots |
| 256K, 250.6K prefill peak | 31.9 / 27.3 GiB | no OOM |

The ledger can be printed item by item with
`lingxi-bench vram-budget <artifact> <arena> 24 32768` (no weights loaded). At 32K, all 12,288
experts on GPU1 are resident; GPU0 is short by 865 cold experts (1.58 GiB), and moving KV to the
host or tightening the reserve lets GPU0 hold everything too. At 256K, use
`LINGXI_QSA_PAGED=1` to keep only a 32K KV window and leave the rest in host memory.

## Quick start

```bash
make build        # requires rustc + nvcc / CUDA 12.8
./start.sh        # production launcher
./start.sh ./bin/lingxi-bench serve <artifact.lingxi or .ninfer> 11434
make regress      # end-to-end criteria
make gates        # guard gates
```

```bash
curl http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "lingxi",
  "messages": [{"role": "user", "content": "The capital of France is"}]
}'
```

Artifacts (about 253 GB) are not checked in; specify the path with `LINGXI_ARTIFACT` or a CLI
argument. Rebuilding from an HF checkpoint: `tools/convert/hf2lingxi.py`. All runtime parameters
are environment variables registered in `crates/lingxi-model/src/probes.rs`.

## Monitoring dashboard

serve ships a zero-dependency, read-only dashboard:

```text
http://127.0.0.1:11434/dashboard
```

It polls `/stats` every 2 seconds and shows average/last tok/s, request count, prefix-cache hit
rate, expert-cache hit rate, speculative acceptance, generated tokens, uptime, and per-GPU VRAM
and utilization. The dashboard and `/stats` expose counters and hardware state only — **never
prompts or output content**. In production you can expose only `/health` if you prefer.

## Project structure

```
crates/
  lingxi-ids        single source of cross-module contracts
  lingxi-artifact   .lingxi / .ninfer container reader/writer
  lingxi-plan       runtime memory layout generated from the catalog
  lingxi-kernels    CUDA C++ → cubin (build.rs invokes nvcc) + cudarc
  lingxi-model      forward logic + static model description + tensor roles + switches
  lingxi-driver     slot pool · admission · decode loop · MTP · KV budget
  lingxi-server     HTTP/SSE (OpenAI-compatible) · tokenization
  lingxi-bench      artifact inspection · benchmarks · golden comparison · serve engine
```

Dependencies flow one way with no cycles (ids → artifact/plan/kernels → model → driver → server);
unsafe is allowed only in `lingxi-kernels`.

## Documentation

- [docs/intro.html](docs/intro.html) — one-page introduction
- [ARCHITECTURE.md](ARCHITECTURE.md) — architecture constraints and module boundaries
- [docs/FORMAT.md](docs/FORMAT.md) — `.lingxi` / `.ninfer` container and numeric formats
- [docs/decode-speed-profile.md](docs/decode-speed-profile.md) — performance profile and all measurements
- [docs/speedup-audit.md](docs/speedup-audit.md) — optimization audit (including falsified ideas)
- [docs/PLAN.md](docs/PLAN.md) / [docs/PLAN_NEXT.md](docs/PLAN_NEXT.md) — engineering decision record
- [docs/strata-0.2.2-v100.md](docs/strata-0.2.2-v100.md) — same-machine Strata 0.2.2 comparison
- [docs/CODE_MAP.md](docs/CODE_MAP.md) — module map and phased refactor plan
- [docs/BACKLOG.md](docs/BACKLOG.md) — backlog and known issues
- [docs/HANDOVER.md](docs/HANDOVER.md) — operations and troubleshooting
- [docs/ASSETS.md](docs/ASSETS.md) — asset inventory and regeneration commands
- [docs/bench-logs/](docs/bench-logs/) — key measurement logs

## License

Apache License 2.0 (see LICENSE). Third-party dependencies retain their own licenses.
