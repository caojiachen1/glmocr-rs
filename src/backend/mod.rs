use std::path::Path;
use std::time::Instant;

use anyhow::Result;

pub mod ort_backend;
pub mod native_backend;
#[cfg(feature = "gguf")]
pub mod gguf_backend;
#[cfg(feature = "gguf")]
pub mod llama_cpp_sys;

// ── Shared logging ──────────────────────────────────────────────────────────

pub fn is_verbose() -> bool {
    let v = std::env::var("OCR_VERBOSE").unwrap_or_else(|_| "1".to_string());
    let v = v.trim().to_ascii_lowercase();
    !(v == "0" || v == "false" || v == "off" || v == "no")
}

pub fn log_info(tag: &str, message: impl AsRef<str>) {
    if is_verbose() {
        eprintln!("[OCR][{}][INFO] {}", tag, message.as_ref());
    }
}

pub fn log_stage_start(tag: &str, stage: impl AsRef<str>) {
    if is_verbose() {
        eprintln!("[OCR][{}][STAGE] >>> {}", tag, stage.as_ref());
    }
}

pub fn log_stage_end(tag: &str, stage: impl AsRef<str>, started_at: Instant) {
    if is_verbose() {
        eprintln!(
            "[OCR][{}][STAGE] <<< {} (elapsed: {:.3}s)",
            tag,
            stage.as_ref(),
            started_at.elapsed().as_secs_f64()
        );
    }
}

pub fn log_stream(tag: &str, step: usize, piece: &str) {
    if is_verbose() {
        eprintln!("[OCR][{}][STREAM] step={}, piece={}", tag, step, piece.replace('\n', "\\n"));
    }
}

// ── Backend trait ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct InferResult {
    pub text: String,
    pub token_count: usize,
}

pub trait OcrBackend {
    fn name(&self) -> &'static str;

    fn infer(
        &mut self,
        model_root: &Path,
        image_path: &Path,
        min_pixels: usize,
        max_pixels: usize,
    ) -> Result<InferResult>;
}
