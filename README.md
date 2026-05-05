# GLM OCR Rust

Rust inference for GLM-OCR, powered by [llama.cpp](https://github.com/ggml-org/llama.cpp) GGUF, [aha](https://github.com/jhqxxx/aha), ONNX Runtime, and Candle backends with full CUDA optimization.

## Quick Start

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/caojiachen1/glmocr_rs.git
cd glmocr_rs

# Place GGUF models in GLM-OCR-GGUF/ (GLM-OCR-Q8_0.gguf + mmproj-GLM-OCR-Q8_0.gguf)
# Or safetensors weights in GLM-OCR/ (model.safetensors, tokenizer.json, config.json)
# Place a test image as test.png

# Run (default: gguf backend, GPU)
cargo run --release

# CPU mode
cargo run --release -- --cpu

# Other backends
cargo run --release -- --backend aha
cargo run --release -- --backend onnx
cargo run --release -- --backend native
```

## CLI

```
glmocr_rs [--backend <gguf|aha|onnx|native>] [--model <path>] [--image <path>]
          [--cpu] [--timing] [--onnx-quantized]
          [--gguf-llama-dll <path>] [--gguf-mtmd-dll <path>]
          [--verbose|--no-verbose]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-b, --backend` | `gguf` | Inference backend |
| `-m, --model` | auto | Model root path |
| `-i, --image` | `test.png` | Input image |
| `--cpu` | off | Force CPU-only inference |
| `--timing` | off | Output `ELAPSED=... TOKENS=...` |
| `--onnx-quantized` | off | Use quantized ONNX model files |
| `--gguf-llama-dll` | auto | Path to `llama.dll` for GGUF backend |
| `--gguf-mtmd-dll` | auto | Path to `mtmd.dll` for GGUF backend |
| `-v, --verbose` | on | Progress and timing logs |

## Environment Variables

### GGUF Backend

| Variable | Description |
|----------|-------------|
| `GGUF_LLAMA_DLL` | Path to `llama.dll` (overridden by `--gguf-llama-dll`) |
| `GGUF_MTMD_DLL` | Path to `mtmd.dll` (overridden by `--gguf-mtmd-dll`) |
| `GGUF_LLAMA_SRC` | Custom path to llama.cpp source directory (build time) |
| `GGUF_SKIP_BUILD` | Set to `1` to skip building llama.cpp from source |
| `GGUF_PREBUILT_DLL_DIR` | Directory containing prebuilt `llama.dll` and `mtmd.dll` (build time) |

### General

| Variable | Description |
|----------|-------------|
| `OCR_BACKEND` | Backend name (`gguf`, `aha`, `onnx`, `native`) |
| `OCR_MODEL_ROOT` | Model directory path |
| `OCR_IMAGE` | Input image path |
| `OCR_VERBOSE` | `1` (default) / `0` for progress logging |
| `OCR_CPU` | `1` for CPU-only inference |
| `OCR_MIN_PIXELS` | Minimum pixel count (default: `12544`) |
| `OCR_MAX_PIXELS` | Maximum pixel count (default: `1048576`) |
| `OCR_MAX_NEW_TOKENS` | Max tokens to generate (default: `2048`) |
| `OCR_STREAM_DECODE` | `1` (default) / `0` for streaming output |

### Using Custom GGML Libraries

```bash
# Use custom DLLs at runtime
GGUF_LLAMA_DLL=/path/to/custom/llama.dll \
GGUF_MTMD_DLL=/path/to/custom/mtmd.dll \
cargo run --release

# Or via CLI flags
cargo run --release -- --gguf-llama-dll /path/to/custom/llama.dll --gguf-mtmd-dll /path/to/custom/mtmd.dll

# Skip building from source, use prebuilt DLLs
GGUF_SKIP_BUILD=1 \
GGUF_PREBUILT_DLL_DIR=/path/to/prebuilt/dlls \
cargo build --release

# Use custom llama.cpp source location
GGUF_LLAMA_SRC=/path/to/custom/llama.cpp \
cargo build --release
```

## Crate API

```toml
[dependencies]
glmocr_rs = { git = "https://github.com/caojiachen1/glmocr_rs" }
```

### One-Shot Inference

```rust
use glmocr_rs::{recognize, OcrConfig, BackendType};

// Simple — uses gguf GPU backend by default
let result = recognize(&OcrConfig {
    image_path: "scan.png".into(),
    ..Default::default()
})?;
println!("{}", result.text);

// Custom backend
let result = recognize(&OcrConfig {
    model_root: "GLM-OCR".into(),
    image_path: "scan.png".into(),
    backend: BackendType::Aha,
    cpu: false,
    ..Default::default()
})?;
```

### Persistent Model (GGUF)

The GGUF backend keeps the model loaded across calls — ideal for multi-image batches
or server scenarios where you want to avoid reloading the model file each time.

```rust
use std::path::Path;
use std::sync::Arc;
use glmocr_rs::backend::gguf_backend::GgufBackend;
use glmocr_rs::backend::llama_cpp_loader::{LlamaCppLib, GgufLibConfig};

// Load llama.cpp DLLs (once per process)
let lib = Arc::new(LlamaCppLib::load(&GgufLibConfig::default())?);

// Create backend — model is NOT loaded yet
let mut backend = GgufBackend::new(Arc::clone(&lib), false);

// First call — loads model (one-time cost)
let r1 = backend.infer(
    Path::new("GLM-OCR-GGUF"),
    Path::new("scan1.png"),
    12544, 1_048_576,
)?;

// Second call — model reused, only KV cache cleared
let r2 = backend.infer(
    Path::new("GLM-OCR-GGUF"),
    Path::new("scan2.png"),
    12544, 1_048_576,
)?;

// Explicitly unload when done (optional — Drop handles it)
backend.unload();
```

### Custom Library Paths (via API)

```rust
use glmocr_rs::backend::llama_cpp_loader::GgufLibConfig;

let config = GgufLibConfig {
    llama_dll: Some("/custom/path/llama.dll".into()),
    mtmd_dll: Some("/custom/path/mtmd.dll".into()),
};
let lib = Arc::new(LlamaCppLib::load(&config)?);
let mut backend = GgufBackend::new(lib, false);
```

### API Overview

| Export | Description |
|--------|-------------|
| `recognize(config)` | One-shot OCR inference |
| `create_backend(config)` | Create backend instance for reuse |
| `OcrConfig` | Configuration struct (`Default` available) |
| `BackendType` | `Gguf` / `Aha` / `Onnx` / `Native` |
| `OcrBackend` | Trait for custom backends |
| `InferResult` | `{ text, token_count }` |
| `GgufLibConfig` | GGUF library path config (`llama_dll`, `mtmd_dll`) |
| `LlamaCppLib` | Dynamically-loaded llama.cpp function pointers |
| `GgufBackend` | GGUF backend with persistent model (`is_loaded()`, `unload()`) |

## Model Setup

| Backend | Directory | Files |
|---------|-----------|-------|
| gguf (default) | `GLM-OCR-GGUF/` | `GLM-OCR-Q8_0.gguf`, `mmproj-GLM-OCR-Q8_0.gguf` |
| aha | `GLM-OCR/` | `model.safetensors`, `tokenizer.json`, `*.json` |
| onnx | `GLM-OCR-ONNX/` | `onnx/*.onnx`, `tokenizer.json` |
| native | `GLM-OCR/` | `model.safetensors`, `tokenizer.json`, `*.json` |

## Features

- **GGUF** — llama.cpp submodule, explicit runtime library loading via `libloading`, persistent model across calls, auto-build via cmake with `GGUF_SKIP_BUILD` / `GGUF_PREBUILT_DLL_DIR` overrides, Flash Attention, KV cache Q8_0, CUDA graph
- **aha** — [aha](https://github.com/jhqxxx/aha) crate, OpenAI-compatible streaming API, auto device detection with CUDA priority
- **ONNX** — ONNX Runtime with CUDA execution provider
- **Native** — Candle (safetensors) pure Rust backend

## Benchmark

```bash
# Test all backends
benchmark.bat

# Single backend
benchmark.bat --backend gguf --runs 10
```
