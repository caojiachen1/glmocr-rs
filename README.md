# GLM OCR ONNX Rust Inference

This project provides a Rust OCR pipeline for GLM-OCR with two backends:

- `onnx` (default)
- `native` (safetensors + Candle)

## Features

- Image preprocessing in Rust (resize + normalize + patchify)
- ONNX Runtime backend using:
  - `GLM-OCR-ONNX/onnx/vision_encoder.onnx`
  - `GLM-OCR-ONNX/onnx/embed_tokens.onnx`
  - `GLM-OCR-ONNX/onnx/decoder_model_merged.onnx`
- Native backend using `GLM-OCR/model.safetensors`
- Autoregressive text decoding
- Output written to `output.md`

## Run

From repository root:

```text
cargo run --release
```

By default this uses:

- backend: `onnx`
- model root: `GLM-OCR-ONNX`
- image: `test.png`

## CLI Arguments

```text
--backend, -b <onnx|native>   Select inference backend
--model, -m <path>            Model root path
--image, -i <path>            Input image path
--verbose, -v                 Enable verbose logs (default)
--no-verbose                  Disable verbose logs; stream OCR text only
--help, -h                    Show help
```

Examples:

```text
cargo run --release -- --image test.png
cargo run --release -- --backend native --model GLM-OCR --image test.png
cargo run --release -- --backend onnx --model GLM-OCR-ONNX --image test.png
cargo run --release -- --no-verbose --image test.png
```

## Environment Variables

- `OCR_BACKEND`
- `OCR_MODEL_ROOT`
- `OCR_IMAGE`
- `OCR_VERBOSE` (default: enabled)
- `OCR_MIN_PIXELS` (default: `12544`)
- `OCR_MAX_PIXELS` (default: `1048576`)
- `OCR_MAX_NEW_TOKENS` (default: `512`)
- `OCR_STREAM_DECODE` (default: enabled)

Resolution order:

- Backend: `--backend` > `OCR_BACKEND` > default `onnx`
- Model root: `--model` > `OCR_MODEL_ROOT` > backend default (`GLM-OCR-ONNX` for `onnx`, `GLM-OCR` for `native`)
- Image: `--image` > `OCR_IMAGE` > `test.png`
- Verbose: `--verbose`/`--no-verbose` > `OCR_VERBOSE` > default enabled

## Notes

- First build can be slow because dependencies must be downloaded and compiled.
- If inference is slow in the vision stage, try lowering `OCR_MAX_PIXELS`.
