use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/shader/CREATE2.cu");

    // Only compile CUDA if we're on a system with nvcc
    if Command::new("nvcc").arg("--version").output().is_ok() {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let cuda_dir = PathBuf::from("src/shader");

        // Compile CUDA to PTX
        let status = Command::new("nvcc")
            .args(&[
                // "-ccbin=C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\bin\\Hostx64\\x64\\cl.exe",
                "--ptx",
                "-arch=sm_50", // Minimum compute capability
                "-o",
                &out_dir.join("CREATE2.ptx").to_str().unwrap(),
                &cuda_dir.join("CREATE2.cu").to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc");

        if !status.success() {
            panic!("Failed to compile CUDA shader");
        }

        // Copy PTX file to shader directory
        std::fs::copy(out_dir.join("CREATE3.ptx"), cuda_dir.join("CREATE3.ptx"))
            .expect("Failed to copy PTX file");

        // Compile CUDA to PTX
        let status = Command::new("nvcc")
            .args(&[
                // "-ccbin=C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\bin\\Hostx64\\x64\\cl.exe",
                "--ptx",
                "-arch=sm_50", // Minimum compute capability
                "-o",
                &out_dir.join("CREATE3.ptx").to_str().unwrap(),
                &cuda_dir.join("CREATE3.cu").to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc");

        if !status.success() {
            panic!("Failed to compile CUDA shader");
        }

        // Copy PTX file to shader directory
        std::fs::copy(out_dir.join("CREATE3.ptx"), cuda_dir.join("CREATE3.ptx"))
            .expect("Failed to copy PTX file");
    } else {
        println!("cargo:warning=nvcc not found, skipping CUDA compilation");
    }
}
