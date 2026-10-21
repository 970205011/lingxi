# Contributing to lingxi

Thanks for taking a look. This project is deliberately narrow: one model family, one GPU
generation (NVIDIA sm_70 / V100), and one goal — making that combination fast and predictable.

## Before you start

- Runtime target: Tesla V100 (sm_70), CUDA 12.8+, Linux x86_64.
- Build: Rust 1.80+ and `nvcc` (the build script defaults to `/usr/local/cuda-12.8/bin/nvcc`;
  override with `QE_NVCC=/path/to/nvcc`).
- Model artifacts are not in the repository (about 253 GB). Conversion and format details are in
  [docs/FORMAT.md](docs/FORMAT.md) and [tools/convert/hf2lingxi.py](tools/convert/hf2lingxi.py).

## Build and check

```bash
make build          # cargo build --release + copy to bin/lingxi-bench
make regress        # end-to-end criteria; needs model artifacts + expert arena
make gates          # guard gates; needs the assets listed in docs/ASSETS.md
```

If you do not have the artifacts, at least run:

```bash
cargo build --release
```

## Code rules

These are hard constraints, not style preferences (see [ARCHITECTURE.md](ARCHITECTURE.md)):

1. Modules communicate only through `lingxi-ids`. Do not import another module's business types.
2. `unsafe` is allowed only inside `lingxi-kernels`.
3. Correctness is structural: use layer-wise golden checks, not "it emits tokens".
4. Any performance change must pass the bit-level criteria: fixed prompt first token and greedy
   sequence must stay identical, unless the change is explicitly documented as lossy and gated by
   an environment switch.
5. Do not silently change a production default. New optimizations start opt-in, get measured, and
   only then become defaults.

## Pull requests

- Keep changes focused; include the measurement or regression evidence for performance changes.
- State which GPU, driver, CUDA version, and model artifact you tested with.
- Negative results are welcome and useful. If an optimization did not work, documenting it prevents
  someone else from repeating it.

## License

By contributing, you agree that your contributions are licensed under Apache-2.0.
