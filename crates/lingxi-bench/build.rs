//! 构建期注入 git 溯源（`lingxi-bench version` 子命令用）。
//! 支持场景："你跑的是哪个版本"——二进制自己答，不需要翻部署目录。

fn git(args: &[&str]) -> Option<std::process::Output> {
    std::process::Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
}

fn main() {
    let hash = git(&["rev-parse", "--short", "HEAD"])
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let dirty = git(&["diff", "--quiet", "HEAD"]).is_none(); // diff --quiet 成功=干净
    println!(
        "cargo:rustc-env=LINGXI_BUILD_GIT={}{}",
        hash,
        if dirty { "-dirty" } else { "" }
    );
    // hash 随提交变：HEAD 动了就重跑 build.rs。
    let head = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.git/HEAD");
    if head.exists() {
        println!("cargo:rerun-if-changed={}", head.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}
