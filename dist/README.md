# Prebuilt binary (Tesla V100 / sm_70)

| Item | Value |
|---|---|
| File | `lingxi-bench-v0.1.0-linux-x86_64-sm70` |
| Source commit | `6c4caf3` (public `open-source` branch) |
| Built | 2026-10-21 |
| Rust | 1.99.1 |
| nvcc | CUDA 12.9 compiler, kernels target `sm_70` |
| glibc | built on glibc 2.41; requires **glibc >= 2.39** |
| SHA256 | see `SHA256SUMS` |

All CUDA kernels are embedded into the binary as `sm_70` cubins; no separate `.cubin`
files are needed. The binary links `libc` / `libm` / `libgcc_s` dynamically and loads
`libcuda`, `libcublas`, and `libnvrtc` at runtime, so a matching NVIDIA driver and CUDA
runtime libraries must be present.

Model artifacts are **not** included (about 253 GB). Build or obtain them separately;
see `docs/FORMAT.md` and `tools/convert/hf2lingxi.py`.

## Usage

```bash
chmod +x lingxi-bench-v0.1.0-linux-x86_64-sm70
./lingxi-bench-v0.1.0-linux-x86_64-sm70 --help

# In the repo, start.sh looks for ./bin/lingxi-bench:
mkdir -p bin
cp lingxi-bench-v0.1.0-linux-x86_64-sm70 bin/lingxi-bench
LINGXI_ARTIFACT=/path/to/model.lingxi ./start.sh
```

If your distribution has glibc < 2.39 (for example Ubuntu 22.04), build from source with
`make build` instead.
