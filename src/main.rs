use std::path::PathBuf;
use std::time::Instant;
use std::{fs, io};

use anyhow::{anyhow, bail, Result};

mod backend;

use backend::{ort_backend::OrtBackend, native_backend::NativeBackend, OcrBackend};

const DEFAULT_MIN_PIXELS: usize = 12_544;
const DEFAULT_MAX_PIXELS: usize = 1_048_576;

#[derive(Debug, Clone, Copy)]
enum BackendChoice {
    Onnx,
    Native,
}

impl BackendChoice {
    fn model_root(self) -> PathBuf {
        match self {
            Self::Onnx => PathBuf::from("GLM-OCR-ONNX"),
            Self::Native => PathBuf::from("GLM-OCR"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Native => "native",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "onnx" | "ort" => Some(Self::Onnx),
            "native" | "nat" | "n" => Some(Self::Native),
            _ => None,
        }
    }
}

fn log_info(message: impl AsRef<str>) {
    eprintln!("[OCR][INFO] {}", message.as_ref());
}

fn read_env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v.parse::<usize>().unwrap_or(default),
        Err(_) => default,
    }
}

fn print_help(binary: &str) {
    println!("GLM OCR Rust Inference");
    println!();
    println!("Usage:");
    println!("  {binary} [--backend <onnx|native>] [--model <path>] [--image <path>] [--onnx-cpu] [--onnx-quantized] [--verbose|--no-verbose]");
    println!();
    println!("Arguments:");
    println!("  -b, --backend <name>   Select inference backend (onnx | native)");
    println!("  -m, --model <path>     Model root path");
    println!("  -i, --image <path>     Input image path");
    println!("      --onnx-cpu         Force ONNX Runtime to use CPUExecutionProvider only");
    println!("      --onnx-quantized   Use quantized ONNX model files (*_quantized.onnx)");
    println!("  -v, --verbose          Enable verbose logs (default)");
    println!("      --no-verbose       Disable verbose logs; stream OCR text only");
    println!("  -h, --help             Show help");
    println!();
    println!("Resolution order:");
    println!("  Backend:");
    println!("    1) CLI argument --backend");
    println!("    2) Environment variable OCR_BACKEND");
    println!("    3) Default: onnx");
    println!("  Model root:");
    println!("    1) CLI argument --model");
    println!("    2) Environment variable OCR_MODEL_ROOT");
    println!("    3) Default by backend: GLM-OCR-ONNX (onnx) / GLM-OCR (native)");
    println!("  Image:");
    println!("    1) CLI argument --image");
    println!("    2) Environment variable OCR_IMAGE");
    println!("    3) Default: test.png");
    println!("  Verbose:");
    println!("    1) CLI argument --verbose/--no-verbose");
    println!("    2) Environment variable OCR_VERBOSE");
    println!("    3) Default: enabled");
    println!("  ONNX CPU mode:");
    println!("    1) CLI argument --onnx-cpu");
    println!("    2) Environment variable OCR_ONNX_CPU");
    println!("    3) Default: disabled (auto CUDA->CPU fallback)");
    println!("  ONNX quantized mode:");
    println!("    1) CLI argument --onnx-quantized");
    println!("    2) Environment variable OCR_ONNX_QUANTIZED");
    println!("    3) Default: disabled (use non-quantized ONNX)");
}

fn parse_cli_options(
) -> Result<(
    BackendChoice,
    &'static str,
    PathBuf,
    &'static str,
    PathBuf,
    &'static str,
    bool,
    &'static str,
    bool,
    &'static str,
    bool,
    &'static str,
)> {
    let mut args = std::env::args();
    let binary = args.next().unwrap_or_else(|| "glm_ocr_onnx_rust".to_string());
    let mut cli_backend: Option<String> = None;
    let mut cli_model_root: Option<String> = None;
    let mut cli_image: Option<String> = None;
    let mut cli_verbose: Option<bool> = None;
    let mut cli_onnx_cpu: Option<bool> = None;
    let mut cli_onnx_quantized: Option<bool> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help(&binary);
                std::process::exit(0);
            }
            "-b" | "--backend" => {
                let v = args.next().ok_or_else(|| {
                    anyhow!("Missing value for {arg}; allowed values: onnx | native")
                })?;
                cli_backend = Some(v);
            }
            "-m" | "--model" => {
                let v = args
                    .next()
                    .ok_or_else(|| anyhow!("Missing value for {arg}; expected a model root path"))?;
                cli_model_root = Some(v);
            }
            "-i" | "--image" => {
                let v = args
                    .next()
                    .ok_or_else(|| anyhow!("Missing value for {arg}; expected an image path"))?;
                cli_image = Some(v);
            }
            "-v" | "--verbose" => {
                cli_verbose = Some(true);
            }
            "--no-verbose" => {
                cli_verbose = Some(false);
            }
            "--onnx-cpu" => {
                cli_onnx_cpu = Some(true);
            }
            "--no-onnx-cpu" => {
                cli_onnx_cpu = Some(false);
            }
            "--onnx-quantized" => {
                cli_onnx_quantized = Some(true);
            }
            "--no-onnx-quantized" => {
                cli_onnx_quantized = Some(false);
            }
            _ if arg.starts_with("--backend=") => {
                let v = arg.trim_start_matches("--backend=").trim().to_string();
                if v.is_empty() {
                    bail!("Missing value for --backend=; allowed values: onnx | native");
                }
                cli_backend = Some(v);
            }
            _ if arg.starts_with("--model=") => {
                let v = arg.trim_start_matches("--model=").trim().to_string();
                if v.is_empty() {
                    bail!("Missing value for --model=; expected a model root path");
                }
                cli_model_root = Some(v);
            }
            _ if arg.starts_with("--image=") => {
                let v = arg.trim_start_matches("--image=").trim().to_string();
                if v.is_empty() {
                    bail!("Missing value for --image=; expected an image path");
                }
                cli_image = Some(v);
            }
            _ if arg.starts_with("--verbose=") => {
                let v = arg.trim_start_matches("--verbose=").trim().to_ascii_lowercase();
                let enabled = match v.as_str() {
                    "1" | "true" | "on" | "yes" => true,
                    "0" | "false" | "off" | "no" => false,
                    _ => bail!("Invalid value for --verbose={v}; use true/false"),
                };
                cli_verbose = Some(enabled);
            }
            _ if arg.starts_with("--onnx-cpu=") => {
                let v = arg.trim_start_matches("--onnx-cpu=").trim().to_ascii_lowercase();
                let enabled = match v.as_str() {
                    "1" | "true" | "on" | "yes" => true,
                    "0" | "false" | "off" | "no" => false,
                    _ => bail!("Invalid value for --onnx-cpu={v}; use true/false"),
                };
                cli_onnx_cpu = Some(enabled);
            }
            _ if arg.starts_with("--onnx-quantized=") => {
                let v = arg
                    .trim_start_matches("--onnx-quantized=")
                    .trim()
                    .to_ascii_lowercase();
                let enabled = match v.as_str() {
                    "1" | "true" | "on" | "yes" => true,
                    "0" | "false" | "off" | "no" => false,
                    _ => bail!("Invalid value for --onnx-quantized={v}; use true/false"),
                };
                cli_onnx_quantized = Some(enabled);
            }
            _ => {
                bail!(
                    "Unknown argument: {arg}. Available arguments: -b/--backend, -m/--model, -i/--image, --onnx-cpu, --onnx-quantized, -v/--verbose, --no-verbose, -h/--help"
                );
            }
        }
    }

    let (backend, backend_source) = if let Some(v) = cli_backend {
        let backend = BackendChoice::parse(&v).ok_or_else(|| {
            anyhow!("Unsupported backend: {v}; allowed values: onnx | native")
        })?;
        (backend, "CLI(--backend)")
    } else if let Ok(v) = std::env::var("OCR_BACKEND") {
        let backend = BackendChoice::parse(&v).ok_or_else(|| {
            anyhow!("Invalid OCR_BACKEND value: {v}; allowed values: onnx | native")
        })?;
        (backend, "ENV(OCR_BACKEND)")
    } else {
        (BackendChoice::Onnx, "DEFAULT")
    };

    let (model_root, model_source) = if let Some(v) = cli_model_root {
        (PathBuf::from(v), "CLI(--model)")
    } else if let Ok(v) = std::env::var("OCR_MODEL_ROOT") {
        (PathBuf::from(v), "ENV(OCR_MODEL_ROOT)")
    } else {
        (backend.model_root(), "DEFAULT(from-backend)")
    };

    let (image_path, image_source) = if let Some(v) = cli_image {
        (PathBuf::from(v), "CLI(--image)")
    } else if let Ok(v) = std::env::var("OCR_IMAGE") {
        (PathBuf::from(v), "ENV(OCR_IMAGE)")
    } else {
        (PathBuf::from("test.png"), "DEFAULT")
    };

    let (verbose, verbose_source) = if let Some(v) = cli_verbose {
        (v, "CLI(--verbose/--no-verbose)")
    } else if let Ok(v) = std::env::var("OCR_VERBOSE") {
        let v = v.trim().to_ascii_lowercase();
        let enabled = !(v == "0" || v == "false" || v == "off" || v == "no");
        (enabled, "ENV(OCR_VERBOSE)")
    } else {
        (true, "DEFAULT")
    };

    let (onnx_cpu, onnx_cpu_source) = if let Some(v) = cli_onnx_cpu {
        (v, "CLI(--onnx-cpu/--no-onnx-cpu)")
    } else if let Ok(v) = std::env::var("OCR_ONNX_CPU") {
        let v = v.trim().to_ascii_lowercase();
        let enabled = !(v == "0" || v == "false" || v == "off" || v == "no");
        (enabled, "ENV(OCR_ONNX_CPU)")
    } else {
        (false, "DEFAULT")
    };

    let (onnx_quantized, onnx_quantized_source) = if let Some(v) = cli_onnx_quantized {
        (v, "CLI(--onnx-quantized/--no-onnx-quantized)")
    } else if let Ok(v) = std::env::var("OCR_ONNX_QUANTIZED") {
        let v = v.trim().to_ascii_lowercase();
        let enabled = !(v == "0" || v == "false" || v == "off" || v == "no");
        (enabled, "ENV(OCR_ONNX_QUANTIZED)")
    } else {
        (false, "DEFAULT")
    };

    Ok((
        backend,
        backend_source,
        model_root,
        model_source,
        image_path,
        image_source,
        verbose,
        verbose_source,
        onnx_cpu,
        onnx_cpu_source,
        onnx_quantized,
        onnx_quantized_source,
    ))
}

fn main() -> Result<()> {
    let (
        backend_choice,
        backend_source,
        model_root,
        model_source,
        input_image_path,
        image_source,
        verbose,
        verbose_source,
        onnx_cpu,
        onnx_cpu_source,
        onnx_quantized,
        onnx_quantized_source,
    ) = parse_cli_options()?;

    std::env::set_var("OCR_VERBOSE", if verbose { "1" } else { "0" });
    if !verbose {
        std::env::set_var("OCR_STREAM_DECODE", "1");
    }

    let min_pixels = read_env_usize("OCR_MIN_PIXELS", DEFAULT_MIN_PIXELS);
    let max_pixels = read_env_usize("OCR_MAX_PIXELS", DEFAULT_MAX_PIXELS);
    if verbose {
        log_info(format!(
            "Model root: {} (source: {})",
            model_root.display(),
            model_source
        ));
        log_info(format!(
            "Input image: {} (source: {})",
            input_image_path.display(),
            image_source
        ));
        log_info(format!(
            "Backend selection: {} (source: {})",
            backend_choice.as_str(),
            backend_source
        ));
        log_info(format!("Verbose: {} (source: {})", verbose, verbose_source));
        log_info(format!(
            "ONNX CPU mode: {} (source: {})",
            onnx_cpu, onnx_cpu_source
        ));
        log_info(format!(
            "ONNX quantized mode: {} (source: {})",
            onnx_quantized, onnx_quantized_source
        ));
        log_info(format!(
            "Pixel constraints: OCR_MIN_PIXELS={}, OCR_MAX_PIXELS={}",
            min_pixels, max_pixels
        ));
    }

    if !input_image_path.exists() {
        bail!("Input image not found: {}", input_image_path.display());
    }

    let infer_start = Instant::now();
    let mut backend: Box<dyn OcrBackend> = match backend_choice {
        BackendChoice::Onnx => Box::new(OrtBackend::new(onnx_cpu, onnx_quantized)),
        BackendChoice::Native => Box::new(NativeBackend::new()),
    };
    if verbose {
        log_info(format!("Active backend: {}", backend.name()));
    }
    let decoded = backend.infer(&model_root, &input_image_path, min_pixels, max_pixels)?;
    if verbose {
        log_info(format!(
            "Main inference stage elapsed: {:.3}s",
            infer_start.elapsed().as_secs_f64()
        ));
    }

    let output_path = PathBuf::from("output.md");
    fs::write(&output_path, decoded.trim())
        .map_err(|e: io::Error| anyhow!("Failed to write output file {}: {}", output_path.display(), e))?;
    if verbose {
        log_info(format!("Result written to: {}", output_path.display()));
    }

    if verbose {
        println!("==== OCR Inference Result ====");
        println!("{}", decoded.trim());
    } else {
        println!();
    }

    Ok(())
}