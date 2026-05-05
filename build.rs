use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    let target_dir = manifest_dir.join("target").join(&profile);

    let building_gguf = std::env::var("CARGO_FEATURE_GGUF").is_ok();

    if building_gguf {
        let skip_build = std::env::var("GGUF_SKIP_BUILD")
            .map(|v| v.trim().to_ascii_lowercase())
            .map(|v| v == "1" || v == "true" || v == "yes")
            .unwrap_or(false);

        if skip_build {
            // Copy prebuilt DLLs from GGUF_PREBUILT_DLL_DIR if set
            if let Ok(dll_dir) = std::env::var("GGUF_PREBUILT_DLL_DIR") {
                let dll_dir = PathBuf::from(&dll_dir);
                if dll_dir.is_dir() {
                    for entry in std::fs::read_dir(&dll_dir).unwrap_or_else(|e| {
                        panic!("Failed to read GGUF_PREBUILT_DLL_DIR '{}': {}", dll_dir.display(), e);
                    }) {
                        let entry = entry.unwrap();
                        let src = entry.path();
                        if src.extension().map_or(false, |ext| ext == "dll") {
                            let dest = target_dir.join(src.file_name().unwrap());
                            std::fs::copy(&src, &dest).unwrap_or_else(|e| {
                                panic!("Failed to copy {} -> {}: {}", src.display(), dest.display(), e);
                            });
                        }
                    }
                }
            }
            println!("cargo:warning=[GGUF] Skipping llama.cpp build (GGUF_SKIP_BUILD set)");
        } else {
            // Use custom llama.cpp source dir if provided
            let llama_src = std::env::var("GGUF_LLAMA_SRC")
                .map(PathBuf::from)
                .unwrap_or_else(|_| manifest_dir.join("third-party").join("llama.cpp"));

            let build_path = llama_src.join("build");
            let llama_lib = build_path.join("src/Release/llama.lib");
            let mtmd_lib = build_path.join("tools/mtmd/Release/mtmd.lib");

            if !llama_lib.exists() || !mtmd_lib.exists() {
                println!("cargo:warning=[GGUF] llama.cpp libs not found, building from {}...", llama_src.display());

                let status = Command::new("cmake")
                    .args([
                        "-B", "build",
                        "-DBUILD_SHARED_LIBS=ON",
                        "-DGGML_CUDA=ON",
                        "-DCMAKE_BUILD_TYPE=Release",
                    ])
                    .current_dir(&llama_src)
                    .status()
                    .expect("Failed to run cmake configure for llama.cpp");
                if !status.success() {
                    panic!("cmake configure for llama.cpp failed");
                }

                let status = Command::new("cmake")
                    .args(["--build", "build", "--config", "Release"])
                    .current_dir(&llama_src)
                    .status()
                    .expect("Failed to run cmake build for llama.cpp");
                if !status.success() {
                    panic!("cmake build for llama.cpp failed");
                }
            }

            // Copy DLLs to target directory so the exe can find them at runtime
            let dll_dir = build_path.join("bin/Release");
            if dll_dir.is_dir() {
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
            }
        }

        // NOTE: No more cargo:rustc-link-lib — libraries are loaded explicitly
        // at runtime via libloading (see src/backend/llama_cpp_loader.rs).
    }

    println!("cargo:rerun-if-changed=build.rs");
    if building_gguf {
        println!("cargo:rerun-if-changed=src/backend/llama_cpp_sys.rs");
        println!("cargo:rerun-if-changed=src/backend/llama_cpp_loader.rs");
    }
}
