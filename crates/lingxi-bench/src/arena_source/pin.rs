//! arena 介质判定（pin 安全性硬判据，CODE_MAP Phase 2 刀P2-1 搬入）。

/// arena 所在**介质**的 fs 类型（读 `/proc/mounts` 取最长前缀匹配的挂载点）。
///
/// 为什么要它：pin 的安全性**完全取决于介质**（见下面的策略注释），而环境变量本身
/// 无法表达"这块盘是不是机械盘"。用 fs 类型做硬判据，把"按介质判定"落到代码里。
pub(crate) fn backing_fs_type(path: &str) -> Option<String> {
    let canon = std::fs::canonicalize(path).ok()?;
    let mounts = std::fs::read_to_string("/proc/mounts").ok()?;
    let mut best: Option<(usize, String)> = None;
    for line in mounts.lines() {
        let mut it = line.split_whitespace();
        let (Some(_dev), Some(mnt), Some(fstype)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let m = std::path::Path::new(mnt);
        if canon.starts_with(m) {
            let depth = m.components().count();
            if best.as_ref().map_or(true, |(d, _)| depth >= *d) {
                best = Some((depth, fstype.to_string()));
            }
        }
    }
    best.map(|(_, t)| t)
}
