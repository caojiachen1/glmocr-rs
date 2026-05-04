use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use futures::StreamExt;

use aha::models::common::model_mapping::WhichModel;
use aha::models::{load_model, GenerateModel};
use aha::params::chat::{
    ChatCompletionParameters, ChatMessage, ChatMessageContent, ChatMessageContentPart,
    ChatMessageImageContentPart, ChatMessageTextContentPart, DeltaChatMessage, ImageUrlType,
};

use super::{InferResult, OcrBackend, is_verbose, log_info, log_stage_end, log_stage_start, log_stream};

const TAG: &str = "AHA";

fn delta_text(delta: &DeltaChatMessage) -> Option<&str> {
    match delta {
        DeltaChatMessage::Assistant { content, .. } | DeltaChatMessage::Untagged { content, .. } => {
            match content {
                Some(ChatMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            }
        }
        _ => None,
    }
}

pub struct AhaBackend {
    force_cpu: bool,
}

impl AhaBackend {
    pub fn new(force_cpu: bool) -> Self {
        Self { force_cpu }
    }
}

impl OcrBackend for AhaBackend {
    fn name(&self) -> &'static str {
        if self.force_cpu {
            "aha (CPU forced)"
        } else {
            "aha"
        }
    }

    fn infer(
        &mut self,
        model_root: &Path,
        image_path: &Path,
        _min_pixels: usize,
        _max_pixels: usize,
    ) -> Result<InferResult> {
        let total_start = Instant::now();
        log_stage_start(TAG, "GLM OCR aha inference");

        if self.force_cpu {
            log_info(TAG, "Execution mode: CPU only");
        } else {
            log_info(TAG, "Execution mode: auto device selection");
        }

        let verbose = is_verbose();
        let stream_decode = std::env::var("OCR_STREAM_DECODE")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "false" || v == "off" || v == "no")
            })
            .unwrap_or(true);

        // --- Load model ---
        let stage = "Load model";
        let stage_start = Instant::now();
        log_stage_start(TAG, stage);

        let (device, dtype) = if self.force_cpu {
            log_info(TAG, "Forcing CPU mode (F32)");
            (Some(aha::Device::Cpu), Some(aha::DType::F32))
        } else {
            let device = match aha::Device::new_cuda(0) {
                Ok(cuda_device) => {
                    log_info(TAG, "CUDA device 0 initialized");
                    Some(cuda_device)
                }
                Err(e) => {
                    log_info(TAG, format!("CUDA unavailable ({}), falling back to auto-select", e));
                    Some(aha::utils::get_device(None))
                }
            };
            (device, None)
        };
        log_info(TAG, format!(
            "Device: {}",
            device.as_ref().map(|d| format!("{:?}", d)).unwrap_or_else(|| "CPU".to_string())
        ));

        let mut model = load_model(
            WhichModel::GlmOCR,
            model_root.to_str().ok_or_else(|| anyhow!("Invalid model root path"))?,
            device.as_ref(),
            dtype,
        )
        .map_err(|e| anyhow!("Failed to load aha model: {e}"))?;

        log_stage_end(TAG, stage, stage_start);

        // --- Build request params ---
        let image_path_abs = if image_path.is_absolute() {
            image_path.to_path_buf()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(image_path)
        };
        // Strip Windows verbatim prefix (\\?\) and normalize separators
        let path_str = image_path_abs.display().to_string();
        let path_str = path_str.strip_prefix(r"\\?\").unwrap_or(&path_str);
        let image_url = format!("file:///{}", path_str.replace('\\', "/"));
        log_info(TAG, format!("Input image: {}", image_url));

        let params = ChatCompletionParameters {
            messages: vec![ChatMessage::User {
                content: ChatMessageContent::ContentPart(vec![
                    ChatMessageContentPart::Image(ChatMessageImageContentPart {
                        r#type: "image_url".into(),
                        image_url: ImageUrlType {
                            url: image_url,
                            detail: None,
                        },
                    }),
                    ChatMessageContentPart::Text(ChatMessageTextContentPart {
                        r#type: "text".into(),
                        text: "Text Recognition:".into(),
                    }),
                ]),
                name: None,
            }],
            model: "glm-ocr".into(),
            max_tokens: Some(2048),
            temperature: Some(0.6),
            top_p: Some(0.9),
            ..Default::default()
        };

        // --- Autoregressive decode loop ---
        let stage = "Autoregressive decode loop";
        let decode_stage_start = Instant::now();
        log_stage_start(TAG, stage);

        let mut generated_text = String::new();
        let mut token_count = 0usize;

        let stream = model
            .generate_stream(params)
            .map_err(|e| anyhow!("Failed to start stream: {e}"))?;

        futures::executor::block_on(async {
            let mut stream = std::pin::pin!(stream);
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(c) => {
                        if let Some(choice) = c.choices.first() {
                            if let Some(content) = delta_text(&choice.delta) {
                                if !content.is_empty() {
                                    if stream_decode {
                                        if verbose {
                                            log_stream(TAG, token_count, content);
                                        } else {
                                            print!("{}", content);
                                            let _ = std::io::stdout().flush();
                                        }
                                    }
                                    generated_text.push_str(content);
                                    token_count += 1;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if verbose {
                            log_info(TAG, format!("Stream error: {e}"));
                        }
                        break;
                    }
                }
            }
        });

        log_stage_end(TAG, stage, decode_stage_start);

        log_info(TAG, format!("Generated token count: {token_count}"));
        log_info(TAG, format!(
            "Total elapsed: {:.3}s",
            total_start.elapsed().as_secs_f64()
        ));
        log_stage_end(TAG, "GLM OCR aha inference", total_start);

        Ok(InferResult {
            text: generated_text,
            token_count,
        })
    }
}
