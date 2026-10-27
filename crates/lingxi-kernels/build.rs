//! build.rs — 把 `cuda/*.cu` 编译为 `$OUT_DIR/*.cubin`。
//! 纯函数 kernel：无全局态、无隐式 stream 依赖；host 侧 launch 封装后续加 cudarc。

use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let nvcc = std::env::var("QE_NVCC")
        .unwrap_or_else(|_| "/usr/local/cuda-12.8/bin/nvcc".to_string());
    let arch = std::env::var("QE_CUDA_ARCH").unwrap_or_else(|_| "sm_70".to_string());

    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=QE_NVCC");
    println!("cargo:rerun-if-env-changed=QE_CUDA_ARCH");

    let cuda_dir = Path::new("cuda");
    let mut any = false;
    for entry in std::fs::read_dir(cuda_dir).expect("缺少 cuda/ 目录") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        any = true;
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        let cubin = Path::new(&out_dir).join(format!("{stem}.cubin"));
        let status = Command::new(&nvcc)
            .arg("-cubin")
            .arg(format!("-arch={arch}"))
            .arg("-O3")
            .arg("--std=c++17")
            .arg("-Xptxas")
            .arg("-O3")
            .arg("-o")
            .arg(&cubin)
            .arg(&path)
            .status()
            .unwrap_or_else(|e| panic!("无法启动 nvcc ({nvcc}): {e}"));
        assert!(status.success(), "nvcc 编译失败: {}", path.display());
        println!("cargo:rerun-if-changed={}", path.display());
    }
    assert!(any, "cuda/ 目录下没有 .cu 文件");
}
