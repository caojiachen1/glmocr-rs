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
          [--cpu] [--timing] [--verbose|--no-verbose]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-b, --backend` | `gguf` | Inference backend |
| `-m, --model` | auto | Model root path |
| `-i, --image` | `test.png` | Input image |
| `--cpu` | off | Force CPU-only inference |
| `--timing` | off | Output `ELAPSED=... TOKENS=...` |
| `-v, --verbose` | on | Progress and timing logs |

## Crate API

```toml
[dependencies]
glmocr_rs = { git = "https://github.com/caojiachen1/glmocr_rs" }
```

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

### API Overview

| Export | Description |
|--------|-------------|
| `recognize(config)` | One-shot OCR inference |
| `create_backend(config)` | Create backend instance for reuse |
| `OcrConfig` | Configuration struct (`Default` available) |
| `BackendType` | `Gguf` / `Aha` / `Onnx` / `Native` |
| `OcrBackend` | Trait for custom backends |
| `InferResult` | `{ text, token_count }` |

## Model Setup

| Backend | Directory | Files |
|---------|-----------|-------|
| gguf (default) | `GLM-OCR-GGUF/` | `GLM-OCR-Q8_0.gguf`, `mmproj-GLM-OCR-Q8_0.gguf` |
| aha | `GLM-OCR/` | `model.safetensors`, `tokenizer.json`, `*.json` |
| onnx | `GLM-OCR-ONNX/` | `onnx/*.onnx`, `tokenizer.json` |
| native | `GLM-OCR/` | `model.safetensors`, `tokenizer.json`, `*.json` |

## Features

- **GGUF** — llama.cpp submodule, auto-build via cmake, Flash Attention, KV cache Q8_0, CUDA graph
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
