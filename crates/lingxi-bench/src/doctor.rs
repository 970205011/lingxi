//! `lingxi-bench doctor`：开箱即用预检。只读文件头/设备信息，不加载模型、不分配显存。

use anyhow::{bail, Result};
use std::io::Read;
use std::net::TcpListener;
use std::path::Path;

fn ok(msg: &str) { println!("  [OK]   {msg}"); }
fn warn(msg: &str) { println!("  [WARN] {msg}"); }
fn fail(msg: &str) { println!("  [FAIL] {msg}"); }

pub(crate) fn run(args: &[String]) -> Result<()> {
    let artifact = args
        .get(2)
        .cloned()
        .or_else(|| std::env::var("LINGXI_ARTIFACT").ok())
        .filter(|s| !s.is_empty());
    let arena = args
        .get(3)
        .cloned()
        .or_else(|| std::env::var("LINGXI_EXPERT_ARENA").ok())
        .filter(|s| !s.is_empty());
    let port: u16 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(11434);

    let mut fails = 0usize;
    println!("lingxi doctor");
    println!("=============");

    println!("\n[1] GPU");
    match std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader",
        ])
        .output()
    {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let mut n = 0;
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                n += 1;
                let cols: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                let name = cols.get(1).copied().unwrap_or("?");
                let mem = cols.get(2).copied().unwrap_or("?");
                let cc = cols.get(3).copied().unwrap_or("?");
                if cc == "7.0" {
                    ok(&format!("GPU{} {name} · {mem} · sm_70", cols[0]));
                } else {
                    warn(&format!(
                        "GPU{} {name} · {mem} · compute {cc}（本引擎为 sm_70 编译，非 V100 可能无法启动）",
                        cols[0]
                    ));
                }
            }
            if n == 0 {
                fail("nvidia-smi 没有返回 GPU");
                fails += 1;
            }
        }
        _ => {
            fail("nvidia-smi 不可用；先装 NVIDIA 驱动并确认容器直通 GPU");
            fails += 1;
        }
    }

    println!("\n[2] 产物");
    match &artifact {
        None => {
            fail("未指定产物；用法: lingxi-bench doctor <artifact> [arena] [port] 或设 LINGXI_ARTIFACT");
            fails += 1;
        }
        Some(path) => {
            let p = Path::new(path);
            if !p.is_file() {
                fail(&format!("找不到产物 {path}"));
                fails += 1;
            } else {
                let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                let mut head = [0u8; 8];
                let magic = std::fs::File::open(p)
                    .and_then(|mut f| f.read_exact(&mut head))
                    .map(|_| &head[..])
                    .unwrap_or(&[]);
                if magic.starts_with(b"NINFER") || magic.starts_with(b"LINGXI") {
                    ok(&format!("{path} · {:.1} GiB · 容器头有效", size as f64 / (1u64 << 30) as f64));
                } else {
                    fail(&format!("{path} 的 magic 不是 NINFER/LINGXI"));
                    fails += 1;
                }
            }
        }
    }

    println!("\n[3] 专家 arena");
    match &arena {
        None => warn("未指定 arena；MoE 模型需要 LINGXI_EXPERT_ARENA"),
        Some(path) => {
            let p = Path::new(path);
            if p.is_file() {
                let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                ok(&format!("{path} · {:.1} GiB", size as f64 / (1u64 << 30) as f64));
            } else {
                fail(&format!("找不到 arena {path}"));
                fails += 1;
            }
        }
    }

    println!("\n[4] /dev/shm");
    match std::process::Command::new("df").args(["-Pk", "/dev/shm"]).output() {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = text.lines().nth(1) {
                let cols: Vec<&str> = line.split_whitespace().collect();
                if let Some(kb) = cols.get(3).and_then(|s| s.parse::<u64>().ok()) {
                    let gib = kb as f64 / (1024.0 * 1024.0);
                    if gib >= 32.0 { ok(&format!("/dev/shm 可用 {gib:.1} GiB")); }
                    else { warn(&format!("/dev/shm 只剩 {gib:.1} GiB；PLE/arena 可能放不下")); }
                }
            }
        }
        Err(_) => warn("无法读取 /dev/shm"),
    }

    println!("\n[5] CUDA 运行库");
    let ldconfig = std::process::Command::new("ldconfig").arg("-p").output();
    let libs = ldconfig.map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let lp = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let lp_has = lp.split(':').filter(|s| !s.is_empty()).any(|d| {
        Path::new(d).join("libcublas.so").exists() || Path::new(d).join("libnvrtc.so").exists()
    });
    let has_cublas = libs.contains("libcublas.so") || lp_has;
    let has_nvrtc = libs.contains("libnvrtc.so") || lp_has;
    if has_cublas && has_nvrtc {
        ok("libcublas / libnvrtc 可见");
    } else {
        warn("ldconfig 看不到 libcublas/libnvrtc；若启动报库缺失，设 LD_LIBRARY_PATH 指向 CUDA 12.8 的 lib64");
    }

    println!("\n[6] 端口 {port}");
    match TcpListener::bind(("127.0.0.1", port)) {
        Ok(_) => ok(&format!("127.0.0.1:{port} 可绑定")),
        Err(e) => warn(&format!("127.0.0.1:{port} 不可绑定：{e}（可能已有 serve 在跑）")),
    }

    println!();
    if fails > 0 { bail!("doctor: {fails} 项失败"); }
    println!("doctor: 通过。下一步：./start.sh");
    Ok(())
}
