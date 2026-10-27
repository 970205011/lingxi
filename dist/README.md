# Prebuilt binary (Tesla V100 / sm_70)

Prebuilt `lingxi-bench` binaries are attached to Releases, built by CI
(`.github/workflows/release.yml`, Ubuntu 22.04, glibc 2.35).

> 2026-10-27 开源分支刷新时移除了此前随仓库提交的 v0.1.0 预编译二进制：它落后于
> 当前 0.2.0 代码，且内嵌的默认路径是本机路径。请用 CI Release 附件或
> `make build && make dist` 自行打包（`make dist` 产出含 bin/ + docs/ + 部署脚本的
> 完整发布包）。
