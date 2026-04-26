// build.rs - compiles CUDA kernels to PTX for cudarc loading
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/decay.cu");

    // Compile to PTX for GB10 (sm_90 Blackwell)
    let nvcc_cmd = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let status = Command::new(nvcc_cmd)
        .args([
            "-arch=sm_90",
            "--ptx",
            "kernels/decay.cu",
            "-o",
            "kernels/decay.ptx",
        ])
        .status()
        .expect("nvcc failed - install CUDA toolkit");

    if !status.success() {
        panic!("CUDA kernel compilation failed");
    }
    println!("cargo:warning=Compiled decay.ptx for GB10 Grace-Blackwell");
}