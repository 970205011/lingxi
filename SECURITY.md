# Security policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Use GitHub's private vulnerability reporting:

https://github.com/970205011/lingxi/security/advisories/new

Include the affected version/commit, a minimal reproduction, and the impact you believe it has.
We will acknowledge the report and coordinate a fix and disclosure timeline.

## Scope

In scope:

- The engine, runtime, HTTP server, and artifact reader in this repository.
- Memory-safety issues in Rust/CUDA boundary code.
- Crashes or silent corruption triggered by malformed `.lingxi` / `.ninfer` artifacts.

Out of scope:

- Model weights (not distributed in this repository).
- Vulnerabilities in third-party dependencies; report them upstream, but feel free to notify us.
- Issues that require an already-compromised host or GPU driver.
