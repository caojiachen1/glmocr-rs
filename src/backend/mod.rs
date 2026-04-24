use std::path::Path;

use anyhow::Result;

pub mod ort_backend;
pub mod native_backend;

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
