use std::path::PathBuf;
use std::time::Instant;
use std::{fs, io};

use anyhow::{anyhow, bail, Result};

mod backend;

use backend::{ort_backend::OrtBackend, native_backend::NativeBackend, OcrBackend, log_info};
#[cfg(feature = "gguf")]
use backend::gguf_backend::GgufBackend;

const DEFAULT_MIN_PIXELS: usize = 12_544;
const DEFAULT_MAX_PIXELS: usize = 1_048_576;

#[derive(Debug, Clone, Copy)]
enum BackendChoice {
    Onnx,
    Native,
    #[cfg(feature = "gguf")]
    Gguf,
}

impl BackendChoice {
    fn model_root(self) -> PathBuf {
        match self {
            Self::Onnx => PathBuf::from("GLM-OCR-ONNX"),
            Self::Native => PathBuf::from("GLM-OCR"),
            #[cfg(feature = "gguf")]
            Self::Gguf => PathBuf::from("GLM-OCR-GGUF"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Native => "native",
            #[cfg(feature = "gguf")]
            Self::Gguf => "gguf",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "onnx" | "ort" => Some(Self::Onnx),
            "native" | "nat" | "n" => Some(Self::Native),
            #[cfg(feature = "gguf")]
            "gguf" | "llama" | "ggml" => Some(Self::Gguf),
            _ => None,
        }
    }
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
    #[cfg(feature = "gguf")]
    println!("Usage:");
    #[cfg(feature = "gguf")]
    println!("  {binary} [--backend <onnx|native|gguf>] [--model <path>] [--image <path>] [--cpu] [--onnx-quantized] [--timing] [--verbose|--no-verbose]");
    #[cfg(not(feature = "gguf"))]
    println!("Usage:");
    #[cfg(not(feature = "gguf"))]
    println!("  {binary} [--backend <onnx|native>] [--model <path>] [--image <path>] [--cpu] [--onnx-quantized] [--timing] [--verbose|--no-verbose]");
    println!();
    println!("Arguments:");
    #[cfg(feature = "gguf")]
    println!("  -b, --backend <name>   Select inference backend (onnx | native | gguf)");
    #[cfg(not(feature = "gguf"))]
    println!("  -b, --backend <name>   Select inference backend (onnx | native)");
    println!("  -m, --model <path>     Model root path");
    println!("  -i, --image <path>     Input image path");
    println!("      --cpu              Force CPU mode (all backends: ONNX CPU-only, Native CPU-only, GGUF CPU-only)");
    println!("      --onnx-quantized   Use quantized ONNX model files (*_quantized.onnx)");
    println!("      --timing           Print structured timing data to stdout (ELAPSED=... TOKENS=...)");
    println!("  -v, --verbose          Enable verbose logs (default)");
    println!("      --no-verbose       Disable verbose logs; stream OCR text only");
    println!("  -h, --help             Show help");
}

struct CliOptions {
    backend: BackendChoice,
    backend_source: &'static str,
    model_root: PathBuf,
    model_source: &'static str,
    image_path: PathBuf,
    image_source: &'static str,
    verbose: bool,
    verbose_source: &'static str,
    cpu: bool,
    cpu_source: &'static str,
    onnx_quantized: bool,
    onnx_quantized_source: &'static str,
    timing: bool,
}

fn parse_cli_options() -> Result<CliOptions> {
    let mut args = std::env::args();
    let binary = args.next().unwrap_or_else(|| "glm_ocr_onnx_rust".to_string());
    let mut cli_backend: Option<String> = None;
    let mut cli_model_root: Option<String> = None;
    let mut cli_image: Option<String> = None;
    let mut cli_verbose: Option<bool> = None;
    let mut cli_cpu: Option<bool> = None;
    let mut cli_onnx_quantized: Option<bool> = None;
    let mut cli_timing = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help(&binary);
                std::process::exit(0);
            }
            "-b" | "--backend" => {
                let v = args.next().ok_or_else(|| {
                    #[cfg(feature = "gguf")]
                    { anyhow!("Missing value for {arg}; allowed values: onnx | native | gguf") }
                    #[cfg(not(feature = "gguf"))]
                    { anyhow!("Missing value for {arg}; allowed values: onnx | native") }
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
            "--cpu" => {
                cli_cpu = Some(true);
            }
            "--no-cpu" => {
                cli_cpu = Some(false);
            }
            "--onnx-quantized" => {
                cli_onnx_quantized = Some(true);
            }
            "--no-onnx-quantized" => {
                cli_onnx_quantized = Some(false);
            }
            "--timing" => {
                cli_timing = true;
            }
            _ if arg.starts_with("--backend=") => {
                let v = arg.trim_start_matches("--backend=").trim().to_string();
                if v.is_empty() {
                    #[cfg(feature = "gguf")]
                    { bail!("Missing value for --backend=; allowed values: onnx | native | gguf"); }
                    #[cfg(not(feature = "gguf"))]
                    { bail!("Missing value for --backend=; allowed values: onnx | native"); }
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
            _ if arg.starts_with("--cpu=") => {
                let v = arg.trim_start_matches("--cpu=").trim().to_ascii_lowercase();
                let enabled = match v.as_str() {
                    "1" | "true" | "on" | "yes" => true,
                    "0" | "false" | "off" | "no" => false,
                    _ => bail!("Invalid value for --cpu={v}; use true/false"),
                };
                cli_cpu = Some(enabled);
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
                    "Unknown argument: {arg}. Available arguments: -b/--backend, -m/--model, -i/--image, --cpu, --onnx-quantized, --timing, -v/--verbose, --no-verbose, -h/--help"
                );
            }
        }
    }

    let (backend, backend_source) = if let Some(v) = cli_backend {
        let backend = BackendChoice::parse(&v).ok_or_else(|| {
            #[cfg(feature = "gguf")]
            { anyhow!("Unsupported backend: {v}; allowed values: onnx | native | gguf") }
            #[cfg(not(feature = "gguf"))]
            { anyhow!("Unsupported backend: {v}; allowed values: onnx | native") }
        })?;
        (backend, "CLI(--backend)")
    } else if let Ok(v) = std::env::var("OCR_BACKEND") {
        let backend = BackendChoice::parse(&v).ok_or_else(|| {
            #[cfg(feature = "gguf")]
            { anyhow!("Invalid OCR_BACKEND value: {v}; allowed values: onnx | native | gguf") }
            #[cfg(not(feature = "gguf"))]
            { anyhow!("Invalid OCR_BACKEND value: {v}; allowed values: onnx | native") }
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

    let (cpu, cpu_source) = if let Some(v) = cli_cpu {
        (v, "CLI(--cpu/--no-cpu)")
    } else if let Ok(v) = std::env::var("OCR_CPU") {
        let v = v.trim().to_ascii_lowercase();
        let enabled = !(v == "0" || v == "false" || v == "off" || v == "no");
        (enabled, "ENV(OCR_CPU)")
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

    Ok(CliOptions {
        backend,
        backend_source,
        model_root,
        model_source,
        image_path,
        image_source,
        verbose,
        verbose_source,
        cpu,
        cpu_source,
        onnx_quantized,
        onnx_quantized_source,
        timing: cli_timing,
    })
}

fn main() -> Result<()> {
    let opts = parse_cli_options()?;

    // --timing mode: suppress all output except the final timing line
    let verbose = if opts.timing { false } else { opts.verbose };

    std::env::set_var("OCR_VERBOSE", if verbose { "1" } else { "0" });
    if opts.timing {
        std::env::set_var("OCR_STREAM_DECODE", "0");
    } else if !verbose {
        std::env::set_var("OCR_STREAM_DECODE", "1");
    }

    let min_pixels = read_env_usize("OCR_MIN_PIXELS", DEFAULT_MIN_PIXELS);
    let max_pixels = read_env_usize("OCR_MAX_PIXELS", DEFAULT_MAX_PIXELS);
    if verbose {
        log_info("MAIN", format!(
            "Model root: {} (source: {})",
            opts.model_root.display(),
            opts.model_source
        ));
        log_info("MAIN", format!(
            "Input image: {} (source: {})",
            opts.image_path.display(),
            opts.image_source
        ));
        log_info("MAIN", format!(
            "Backend selection: {} (source: {})",
            opts.backend.as_str(),
            opts.backend_source
        ));
        log_info("MAIN", format!("Verbose: {} (source: {})", verbose, opts.verbose_source));
        log_info("MAIN", format!(
            "CPU mode: {} (source: {})",
            opts.cpu, opts.cpu_source
        ));
        log_info("MAIN", format!(
            "ONNX quantized mode: {} (source: {})",
            opts.onnx_quantized, opts.onnx_quantized_source
        ));
        log_info("MAIN", format!(
            "Pixel constraints: OCR_MIN_PIXELS={}, OCR_MAX_PIXELS={}",
            min_pixels, max_pixels
        ));
    }

    if !opts.image_path.exists() {
        bail!("Input image not found: {}", opts.image_path.display());
    }

    let infer_start = Instant::now();
    let mut backend: Box<dyn OcrBackend> = match opts.backend {
        BackendChoice::Onnx => Box::new(OrtBackend::new(opts.cpu, opts.onnx_quantized)),
        BackendChoice::Native => Box::new(NativeBackend::new(opts.cpu)),
        #[cfg(feature = "gguf")]
        BackendChoice::Gguf => Box::new(GgufBackend::new(opts.cpu)),
    };
    if verbose {
        log_info("MAIN", format!("Active backend: {}", backend.name()));
    }
    let result = backend.infer(&opts.model_root, &opts.image_path, min_pixels, max_pixels)?;
    let elapsed = infer_start.elapsed().as_secs_f64();

    // --timing: print structured data to stdout and exit
    if opts.timing {
        println!("BACKEND={} ELAPSED={:.3} TOKENS={}", opts.backend.as_str(), elapsed, result.token_count);
        return Ok(());
    }

    if verbose {
        let tokens_per_second = if elapsed > 0.0 {
            result.token_count as f64 / elapsed
        } else {
            0.0
        };
        log_info("MAIN", format!(
            "Main inference stage elapsed: {:.3}s",
            elapsed
        ));
        log_info("MAIN", format!(
            "Decode speed: {:.2} tokens/s ({} tokens in {:.3}s)",
            tokens_per_second, result.token_count, elapsed
        ));
    }
    let decoded = result.text;

    let output_path = PathBuf::from("output.md");
    fs::write(&output_path, decoded.trim())
        .map_err(|e: io::Error| anyhow!("Failed to write output file {}: {}", output_path.display(), e))?;
    if verbose {
        log_info("MAIN", format!("Result written to: {}", output_path.display()));
    }

    if verbose {
        println!("==== OCR Inference Result ====");
        println!("{}", decoded.trim());
    } else {
        println!();
    }

    Ok(())
}
