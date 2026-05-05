pub mod backend;

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::Result;

#[cfg(feature = "gguf")]
use backend::gguf_backend::GgufBackend;
#[cfg(feature = "gguf")]
use backend::llama_cpp_loader::LlamaCppLib;
#[cfg(feature = "aha")]
use backend::aha_backend::AhaBackend;
use backend::ort_backend::OrtBackend;
use backend::native_backend::NativeBackend;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use backend::{
    InferResult,
    OcrBackend,
    is_verbose,
    log_info,
    log_stage_start,
    log_stage_end,
    log_stream,
};

#[cfg(feature = "gguf")]
pub use backend::llama_cpp_loader::GgufLibConfig;

/// Supported OCR backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// llama.cpp GGUF backend (GPU + CPU, fastest).
    #[cfg(feature = "gguf")]
    Gguf,
    /// aha crate backend (GPU + CPU, streaming via HTTP-compatible API).
    #[cfg(feature = "aha")]
    Aha,
    /// ONNX Runtime backend (GPU + CPU).
    Onnx,
    /// Candle native Rust backend (GPU + CPU).
    Native,
}

impl BackendType {
    /// Human-readable name for this backend.
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "gguf")]
            Self::Gguf => "gguf",
            #[cfg(feature = "aha")]
            Self::Aha => "aha",
            Self::Onnx => "onnx",
            Self::Native => "native",
        }
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for OCR inference.
///
/// # Example
///
/// ```no_run
/// use glmocr_rs::{OcrConfig, BackendType};
///
/// let config = OcrConfig {
///     model_root: "GLM-OCR-GGUF".into(),
///     image_path: "scan.png".into(),
///     backend: BackendType::Gguf,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// Path to the model directory (e.g. `GLM-OCR-GGUF`, `GLM-OCR-ONNX`).
    pub model_root: PathBuf,
    /// Path to the input image file.
    pub image_path: PathBuf,
    /// Which inference backend to use.
    pub backend: BackendType,
    /// Force CPU-only inference (no GPU offload).
    pub cpu: bool,
    /// Use quantized ONNX model files (`*_quantized.onnx`). Only relevant for the ONNX backend.
    pub onnx_quantized: bool,
    /// Minimum pixels for image preprocessing (prevents downscaling below this).
    pub min_pixels: usize,
    /// Maximum pixels for image preprocessing (downscale if image exceeds this).
    pub max_pixels: usize,
    /// Print progress and timing to stderr.
    pub verbose: bool,
    /// GGUF-specific: custom DLL paths for llama.cpp libraries.
    /// When `None` (default), libraries are resolved via env vars then default OS search path.
    #[cfg(feature = "gguf")]
    pub gguf_lib: Option<GgufLibConfig>,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "gguf")]
            model_root: PathBuf::from("GLM-OCR-GGUF"),
            #[cfg(not(feature = "gguf"))]
            model_root: PathBuf::from("GLM-OCR-ONNX"),
            image_path: PathBuf::from("test.png"),
            #[cfg(feature = "gguf")]
            backend: BackendType::Gguf,
            #[cfg(not(feature = "gguf"))]
            backend: BackendType::Onnx,
            cpu: false,
            onnx_quantized: false,
            min_pixels: 12_544,
            max_pixels: 1_048_576,
            verbose: true,
            #[cfg(feature = "gguf")]
            gguf_lib: None,
        }
    }
}

impl OcrConfig {
    /// Create a new config with defaults, overriding the backend.
    pub fn with_backend(backend: BackendType) -> Self {
        let model_root = match backend {
            #[cfg(feature = "gguf")]
            BackendType::Gguf => PathBuf::from("GLM-OCR-GGUF"),
            #[cfg(feature = "aha")]
            BackendType::Aha => PathBuf::from("GLM-OCR"),
            BackendType::Onnx => PathBuf::from("GLM-OCR-ONNX"),
            BackendType::Native => PathBuf::from("GLM-OCR"),
        };
        Self {
            model_root,
            backend,
            ..Default::default()
        }
    }
}

// ── API ───────────────────────────────────────────────────────────────────────

#[cfg(feature = "gguf")]
static CACHED_GGUF_LIB: OnceLock<Arc<LlamaCppLib>> = OnceLock::new();

#[cfg(feature = "gguf")]
fn get_or_load_gguf_lib(config: &OcrConfig) -> Result<Arc<LlamaCppLib>> {
    if let Some(ref lib_config) = config.gguf_lib {
        // Custom config — always load fresh (not cached)
        return Ok(Arc::new(LlamaCppLib::load(lib_config)?));
    }
    // Default config — cache for reuse across calls
    Ok(CACHED_GGUF_LIB
        .get_or_init(|| {
            Arc::new(LlamaCppLib::load(&GgufLibConfig::default())
                .expect("Failed to load default GGUF libraries"))
        })
        .clone())
}

/// Create a backend instance from the given backend type.
///
/// The returned boxed trait object implements [`OcrBackend`] and can be used
/// directly if you need fine-grained control.
pub fn create_backend(config: &OcrConfig) -> Result<Box<dyn OcrBackend>> {
    match config.backend {
        #[cfg(feature = "gguf")]
        BackendType::Gguf => {
            let lib = get_or_load_gguf_lib(config)?;
            Ok(Box::new(GgufBackend::new(lib, config.cpu)))
        }
        #[cfg(feature = "aha")]
        BackendType::Aha => Ok(Box::new(AhaBackend::new(config.cpu))),
        BackendType::Onnx => Ok(Box::new(OrtBackend::new(config.cpu, config.onnx_quantized))),
        BackendType::Native => Ok(Box::new(NativeBackend::new(config.cpu))),
    }
}

/// Run OCR inference with the given configuration.
///
/// This is the primary entry point. It creates a backend, runs inference,
/// and returns the recognized text together with token count and timing.
///
/// # Example
///
/// ```no_run
/// use glmocr_rs::{recognize, OcrConfig, BackendType};
///
/// let config = OcrConfig {
///     model_root: "GLM-OCR-GGUF".into(),
///     image_path: "document.png".into(),
///     backend: BackendType::Gguf,
///     ..Default::default()
/// };
///
/// let result = recognize(&config)?;
/// println!("{}", result.text);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn recognize(config: &OcrConfig) -> Result<InferResult> {
    // Set verbosity so backend logging respects the config
    std::env::set_var("OCR_VERBOSE", if config.verbose { "1" } else { "0" });

    if !config.image_path.exists() {
        anyhow::bail!("Input image not found: {}", config.image_path.display());
    }

    let mut backend = create_backend(config)?;
    backend.infer(
        &config.model_root,
        &config.image_path,
        config.min_pixels,
        config.max_pixels,
    )
}
