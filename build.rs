use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_path = manifest_dir.join("third-party").join("llama.cpp");
    let build_path = llama_path.join("build");
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    let target_dir = manifest_dir.join("target").join(&profile);

    let building_gguf = std::env::var("CARGO_FEATURE_GGUF").is_ok();

    if building_gguf {
        let llama_lib = build_path.join("src/Release/llama.lib");
        let mtmd_lib = build_path.join("tools/mtmd/Release/mtmd.lib");

        if !llama_lib.exists() || !mtmd_lib.exists() {
            println!("cargo:warning=[GGUF] llama.cpp libs not found, building...");

            let status = Command::new("cmake")
                .args([
                    "-B", "build",
                    "-DBUILD_SHARED_LIBS=ON",
                    "-DGGML_CUDA=ON",
                    "-DCMAKE_BUILD_TYPE=Release",
                ])
                .current_dir(&llama_path)
                .status()
                .expect("Failed to run cmake configure for llama.cpp");
            if !status.success() {
                panic!("cmake configure for llama.cpp failed");
            }

            let status = Command::new("cmake")
                .args(["--build", "build", "--config", "Release"])
                .current_dir(&llama_path)
                .status()
                .expect("Failed to run cmake build for llama.cpp");
            if !status.success() {
                panic!("cmake build for llama.cpp failed");
            }
        }

        // Copy DLLs to target directory so the exe can find them at runtime
        let dll_dir = build_path.join("bin/Release");
        for entry in std::fs::read_dir(&dll_dir).expect("Failed to read llama.cpp DLL directory") {
            let entry = entry.unwrap();
            let src = entry.path();
            if src.extension().map_or(false, |ext| ext == "dll") {
                let dest = target_dir.join(src.file_name().unwrap());
                std::fs::copy(&src, &dest).unwrap_or_else(|e| {
                    panic!("Failed to copy {} -> {}: {}", src.display(), dest.display(), e);
                });
            }
        }

        println!("cargo:rustc-link-search=native={}", build_path.join("src/Release").display());
        println!("cargo:rustc-link-search=native={}", build_path.join("tools/mtmd/Release").display());
        println!("cargo:rustc-link-lib=llama");
        println!("cargo:rustc-link-lib=mtmd");
    }

    println!("cargo:rerun-if-changed=build.rs");
    if building_gguf {
        println!("cargo:rerun-if-changed={}", llama_path.join("include/llama.h").display());
        println!("cargo:rerun-if-changed={}", llama_path.join("tools/mtmd/mtmd.h").display());
        println!("cargo:rerun-if-changed=src/backend/llama_cpp_sys.rs");
    }
}
