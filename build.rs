// build.rs - compiles CUDA kernels to PTX for cudarc loading
use std::path::PathBuf;
use std::process::Command;

fn find_nvcc() -> Option<String> {
    if let Ok(p) = std::env::var("NVCC") {
        return Some(p);
    }
    let candidates = [
        "nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.1/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
    ];
    for c in candidates {
        if c == "nvcc" {
            if Command::new("nvcc").arg("--version").output().is_ok() {
                return Some("nvcc".into());
            }
        } else if PathBuf::from(c).exists() {
            return Some(c.into());
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/decay.cu");
    println!("cargo:rerun-if-changed=kernels/decay.ptx");

    // Prefer recompiling; fall back to checked-in PTX if nvcc missing.
    let Some(nvcc_cmd) = find_nvcc() else {
        if PathBuf::from("kernels/decay.ptx").exists() {
            println!("cargo:warning=nvcc not found; using existing kernels/decay.ptx");
            return;
        }
        panic!("nvcc not found and kernels/decay.ptx missing — install CUDA toolkit");
    };

    // Compile to PTX for GB10 (sm_90/sm_121 family)
    let status = Command::new(&nvcc_cmd)
        .args([
            "-arch=sm_90",
            "--ptx",
            "kernels/decay.cu",
            "-o",
            "kernels/decay.ptx",
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=Compiled decay.ptx with {nvcc_cmd}");
        }
        Ok(s) => {
            if PathBuf::from("kernels/decay.ptx").exists() {
                println!("cargo:warning=nvcc exit {s}; using existing kernels/decay.ptx");
            } else {
                panic!("CUDA kernel compilation failed: {s}");
            }
        }
        Err(e) => {
            if PathBuf::from("kernels/decay.ptx").exists() {
                println!("cargo:warning=nvcc error ({e}); using existing kernels/decay.ptx");
            } else {
                panic!("nvcc failed: {e}");
            }
        }
    }
}
