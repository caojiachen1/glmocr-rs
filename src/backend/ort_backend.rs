use std::path::{Path, PathBuf};
use std::time::Instant;
use std::io::Write;

use anyhow::{anyhow, bail, Context, Result};
use image::{imageops::FilterType, DynamicImage};
use ndarray::{s, Array, Array2, Array3, Array4, ArrayD, Axis, Ix2, Ix3, IxDyn};
use ort::{
    ep::{get_gpu_device, CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider},
    session::{Session, SessionInputValue},
    value::Tensor,
};
use tokenizers::Tokenizer;

use super::{OcrBackend, InferResult};

const IMAGE_TOKEN_ID: i64 = 59280;
const EOS_TOKEN_IDS: [i64; 2] = [59246, 59253];
const HIDDEN_SIZE: usize = 1536;
const NUM_LAYERS: usize = 16;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const PATCH_SIZE: usize = 14;
const TEMPORAL_PATCH_SIZE: usize = 2;
const MERGE_SIZE: usize = 2;

fn log_info(message: impl AsRef<str>) {
    if is_verbose() {
        eprintln!("[OCR][INFO] {}", message.as_ref());
    }
}

fn log_stage_start(stage: impl AsRef<str>) {
    if is_verbose() {
        eprintln!("[OCR][STAGE] >>> {}", stage.as_ref());
    }
}

fn log_stage_end(stage: impl AsRef<str>, started_at: Instant) {
    if is_verbose() {
        eprintln!(
            "[OCR][STAGE] <<< {} (elapsed: {:.3}s)",
            stage.as_ref(),
            started_at.elapsed().as_secs_f64()
        );
    }
}

fn is_verbose() -> bool {
    let v = std::env::var("OCR_VERBOSE").unwrap_or_else(|_| "1".to_string());
    let v = v.trim().to_ascii_lowercase();
    !(v == "0" || v == "false" || v == "off" || v == "no")
}

fn ok<T, E: std::fmt::Display>(res: std::result::Result<T, E>) -> Result<T> {
    res.map_err(|e| anyhow!(e.to_string()))
}

fn create_session_with_cuda_fallback(
    model_path: &Path,
    session_name: &str,
    force_cpu: bool,
) -> Result<Session> {
    let t = Instant::now();

    if force_cpu {
        let cpu_builder = ok(Session::builder())?;
        let mut cpu_builder = ok(
            cpu_builder.with_execution_providers([CPUExecutionProvider::default().build()]),
        )?;
        let session = ok(cpu_builder.commit_from_file(model_path)).with_context(|| {
            format!(
                "{} CPU-only initialization failed: {}",
                session_name,
                model_path.display()
            )
        })?;
        log_info(format!(
            "{} initialized (CPU only mode, elapsed: {:.3}s)",
            session_name,
            t.elapsed().as_secs_f64()
        ));
        return Ok(session);
    }

    let cuda_ep = CUDAExecutionProvider::default();
    if !cuda_ep.supported_by_platform() {
        log_info(format!(
            "{} CUDAExecutionProvider is not supported on this platform, falling back to CPU",
            session_name
        ));
    } else {
        match cuda_ep.is_available() {
            Ok(true) => {
                let cuda_builder = ok(Session::builder())?;
                match ok(cuda_builder.with_execution_providers([
                    CUDAExecutionProvider::default().build().error_on_failure(),
                ])) {
                    Ok(mut cuda_builder) => match ok(cuda_builder.commit_from_file(model_path)) {
                        Ok(session) => {
                            let device_msg = match get_gpu_device() {
                                Ok(id) => format!("device_id={id}"),
                                Err(e) => format!("device_id=unknown ({e})"),
                            };
                            log_info(format!(
                                "{} initialized (CUDAExecutionProvider enabled, {}, elapsed: {:.3}s)",
                                session_name,
                                device_msg,
                                t.elapsed().as_secs_f64()
                            ));
                            return Ok(session);
                        }
                        Err(cuda_err) => {
                            log_info(format!(
                                "{} failed to create CUDA session, falling back to CPU: {}",
                                session_name, cuda_err
                            ));
                        }
                    },
                    Err(cuda_err) => {
                        log_info(format!(
                            "{} failed to register CUDA provider, falling back to CPU: {}",
                            session_name, cuda_err
                        ));
                    }
                }
            }
            Ok(false) => {
                log_info(format!(
                    "{} CUDAExecutionProvider is unavailable (ORT build/dependencies), falling back to CPU",
                    session_name
                ));
            }
            Err(e) => {
                log_info(format!(
                    "{} failed to check CUDAExecutionProvider availability, falling back to CPU: {}",
                    session_name, e
                ));
            }
        }
    }

    let cpu_builder = ok(Session::builder())?;
    let mut cpu_builder = ok(
        cpu_builder.with_execution_providers([CPUExecutionProvider::default().build()]),
    )?;
    let session = ok(cpu_builder.commit_from_file(model_path)).with_context(|| {
        format!(
            "{} CPU fallback initialization failed: {}",
            session_name,
            model_path.display()
        )
    })?;
    log_info(format!(
        "{} initialized (CPU fallback, elapsed: {:.3}s)",
        session_name,
        t.elapsed().as_secs_f64()
    ));
    Ok(session)
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let mut h_bar = ((height as f64 / factor as f64).round() as usize).max(1) * factor;
    let mut w_bar = ((width as f64 / factor as f64).round() as usize).max(1) * factor;

    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        w_bar = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
    }

    (h_bar, w_bar)
}

fn preprocess_image(
    image_path: &Path,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(Array2<f32>, Array2<i64>)> {
    let stage = "Image preprocessing";
    let stage_start = Instant::now();
    log_stage_start(stage);
    log_info(format!("Loading image: {}", image_path.display()));

    let image = image::open(image_path)
        .with_context(|| format!("Failed to open input image: {}", image_path.display()))?;
    let rgb = to_rgb(image);
    let (orig_w, orig_h) = rgb.dimensions();
    log_info(format!("Original resolution: {}x{}", orig_w, orig_h));

    let (target_h, target_w) = smart_resize(
        orig_h as usize,
        orig_w as usize,
        PATCH_SIZE * MERGE_SIZE,
        min_pixels,
        max_pixels,
    );
    log_info(format!("Resized resolution: {}x{}", target_w, target_h));

    let resized = image::imageops::resize(&rgb, target_w as u32, target_h as u32, FilterType::CatmullRom);

    let image_mean = [0.48145466f32, 0.4578275f32, 0.40821073f32];
    let image_std = [0.26862954f32, 0.26130258f32, 0.27577711f32];

    let mut chw = Array3::<f32>::zeros((3, target_h, target_w));
    for y in 0..target_h {
        for x in 0..target_w {
            let p = resized.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                chw[[c, y, x]] = (v - image_mean[c]) / image_std[c];
            }
        }
    }

    let mut frames = Array4::<f32>::zeros((TEMPORAL_PATCH_SIZE, 3, target_h, target_w));
    for t in 0..TEMPORAL_PATCH_SIZE {
        frames.slice_mut(s![t, .., .., ..]).assign(&chw);
    }

    let grid_h = target_h / PATCH_SIZE;
    let grid_w = target_w / PATCH_SIZE;
    log_info(format!("Patch grid: grid_h={}, grid_w={}", grid_h, grid_w));

    let mut pixel_values = Vec::<f32>::with_capacity(grid_h * grid_w * 1176);
    for gh_block in 0..(grid_h / MERGE_SIZE) {
        for gw_block in 0..(grid_w / MERGE_SIZE) {
            for gh_inner in 0..MERGE_SIZE {
                for gw_inner in 0..MERGE_SIZE {
                    let gh = gh_block * MERGE_SIZE + gh_inner;
                    let gw = gw_block * MERGE_SIZE + gw_inner;
                    for c in 0..3 {
                        for t in 0..TEMPORAL_PATCH_SIZE {
                            for ph in 0..PATCH_SIZE {
                                for pw in 0..PATCH_SIZE {
                                    let y = gh * PATCH_SIZE + ph;
                                    let x = gw * PATCH_SIZE + pw;
                                    pixel_values.push(frames[[t, c, y, x]]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let num_patches = grid_h * grid_w;
    if num_patches > 4096 {
        log_info(format!(
            "Warning: num_patches={} is large; vision encoding may be slow. Consider lowering OCR_MAX_PIXELS",
            num_patches
        ));
    }
    let pixel_values = Array2::from_shape_vec((num_patches, 1176), pixel_values)
        .context("Failed to build pixel_values")?;
    let image_grid_thw = Array2::from_shape_vec((1, 3), vec![1i64, grid_h as i64, grid_w as i64])
        .context("Failed to build image_grid_thw")?;

    log_info(format!(
        "pixel_values shape={:?}, image_grid_thw={:?}",
        pixel_values.shape(),
        image_grid_thw
    ));
    log_stage_end(stage, stage_start);

    Ok((pixel_values, image_grid_thw))
}

fn to_rgb(img: DynamicImage) -> image::RgbImage {
    img.to_rgb8()
}

fn build_prompt() -> String {
    "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>\nText Recognition:\n<|assistant|>\n"
        .to_string()
}

fn build_glm_mrope_positions(
    image_pos: usize,
    text_seq_len: usize,
    image_grid_thw: &Array2<i64>,
    image_token_count: usize,
    merged_seq_len: usize,
) -> Result<(Array3<i64>, i64)> {
    if image_grid_thw.shape() != [1, 3] {
        bail!("Invalid image_grid_thw shape: {:?}", image_grid_thw.shape());
    }

    let t = image_grid_thw[[0, 0]];
    let h = image_grid_thw[[0, 1]];
    let w = image_grid_thw[[0, 2]];
    let spatial_merge = MERGE_SIZE as i64;
    let llm_grid_t = t;
    let llm_grid_h = h / spatial_merge;
    let llm_grid_w = w / spatial_merge;

    let expected_image_tokens = (llm_grid_t * llm_grid_h * llm_grid_w) as usize;
    if expected_image_tokens != image_token_count {
        bail!(
            "Image token count mismatch: expected={}, actual={}"
,            expected_image_tokens,
            image_token_count
        );
    }

    let mut rows = [
        Vec::<i64>::with_capacity(merged_seq_len),
        Vec::<i64>::with_capacity(merged_seq_len),
        Vec::<i64>::with_capacity(merged_seq_len),
    ];

    let mut current_pos = 0i64;

    for i in 0..image_pos {
        let v = current_pos + i as i64;
        rows[0].push(v);
        rows[1].push(v);
        rows[2].push(v);
    }
    current_pos += image_pos as i64;

    for tt in 0..llm_grid_t {
        for hh in 0..llm_grid_h {
            for ww in 0..llm_grid_w {
                rows[0].push(current_pos + tt);
                rows[1].push(current_pos + hh);
                rows[2].push(current_pos + ww);
            }
        }
    }

    current_pos += (h.max(w)) / spatial_merge;

    let suffix_len = text_seq_len - image_pos - 1;
    for i in 0..suffix_len {
        let v = current_pos + i as i64;
        rows[0].push(v);
        rows[1].push(v);
        rows[2].push(v);
    }

    if rows[0].len() != merged_seq_len || rows[1].len() != merged_seq_len || rows[2].len() != merged_seq_len {
        bail!(
            "position_ids length mismatch: got=({}, {}, {}), expected={}"
,            rows[0].len(),
            rows[1].len(),
            rows[2].len(),
            merged_seq_len
        );
    }

    let max_pos = rows
        .iter()
        .flat_map(|r| r.iter().copied())
        .max()
        .unwrap_or(0);
    let mrope_delta = max_pos + 1 - merged_seq_len as i64;

    let mut position_ids = Array3::<i64>::zeros((3, 1, merged_seq_len));
    for axis in 0..3 {
        for i in 0..merged_seq_len {
            position_ids[[axis, 0, i]] = rows[axis][i];
        }
    }

    Ok((position_ids, mrope_delta))
}

fn pick_next_token(logits: &[f32], step: usize) -> i64 {
    if logits.is_empty() {
        return EOS_TOKEN_IDS[0];
    }

    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if step < 5 {
        for (idx, _) in &pairs {
            let token = *idx as i64;
            if !EOS_TOKEN_IDS.contains(&token) {
                return token;
            }
        }
    }

    pairs.first().map(|(idx, _)| *idx as i64).unwrap_or(EOS_TOKEN_IDS[0])
}

pub struct OrtBackend {
    force_cpu: bool,
    quantized: bool,
}

impl OrtBackend {
    pub fn new(force_cpu: bool, quantized: bool) -> Self {
        Self {
            force_cpu,
            quantized,
        }
    }

    fn model_paths(model_root: &Path, quantized: bool) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
        let onnx_dir = model_root.join("onnx");
        let (vision_file, embed_file, decoder_file) = if quantized {
            (
                "vision_encoder_quantized.onnx",
                "embed_tokens_quantized.onnx",
                "decoder_model_merged_quantized.onnx",
            )
        } else {
            (
                "vision_encoder.onnx",
                "embed_tokens.onnx",
                "decoder_model_merged.onnx",
            )
        };
        (
            model_root.join("tokenizer.json"),
            onnx_dir.join(vision_file),
            onnx_dir.join(embed_file),
            onnx_dir.join(decoder_file),
        )
    }
}

impl OcrBackend for OrtBackend {
    fn name(&self) -> &'static str {
        if self.force_cpu {
            "ort (CPU forced)"
        } else {
            "ort"
        }
    }

    fn infer(
        &mut self,
        model_root: &Path,
        image_path: &Path,
        min_pixels: usize,
        max_pixels: usize,
    ) -> Result<InferResult> {
        let total_start = Instant::now();
        log_stage_start("GLM OCR ONNX Rust inference");

        if self.force_cpu {
            log_info("ONNX execution mode: CPU only");
        } else {
            log_info("ONNX execution mode: auto CUDA -> CPU fallback");
        }
        log_info(format!(
            "ONNX model precision mode: {}",
            if self.quantized { "quantized" } else { "default" }
        ));

        let (tokenizer_path, vision_path, embed_path, decoder_path) =
            Self::model_paths(model_root, self.quantized);

        for p in [&tokenizer_path, &vision_path, &embed_path, &decoder_path] {
            if !p.exists() {
                bail!("Missing model file: {}", p.display());
            }
        }
        log_info("File check passed: tokenizer + vision + embed + decoder all exist");

        let stage = "Load ONNX sessions";
        let stage_start = Instant::now();
        log_stage_start(stage);

        let mut vision_session =
            create_session_with_cuda_fallback(&vision_path, "vision_session", self.force_cpu)?;
        let mut embed_session =
            create_session_with_cuda_fallback(&embed_path, "embed_session", self.force_cpu)?;
        let mut decoder_session = create_session_with_cuda_fallback(
            &decoder_path,
            "decoder_session",
            self.force_cpu,
        )?;

        log_stage_end(stage, stage_start);

        let stage = "Load tokenizer";
        let stage_start = Instant::now();
        log_stage_start(stage);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!(e.to_string()))?;
        log_stage_end(stage, stage_start);

        let (pixel_values, image_grid_thw) = preprocess_image(image_path, min_pixels, max_pixels)?;

        let stage = "Vision encoding (vision_encoder)";
        let stage_start = Instant::now();
        log_stage_start(stage);
        let image_features = {
            log_info("Preparing vision input tensors (pixel_values / image_grid_thw)");
            let t_build = Instant::now();
            let pixel_values_tensor = ok(Tensor::from_array(pixel_values.clone()))?;
            let image_grid_thw_tensor = ok(Tensor::from_array(image_grid_thw.clone()))?;
            log_info(format!(
                "Vision input tensors prepared (elapsed: {:.3}s)",
                t_build.elapsed().as_secs_f64()
            ));

            log_info("Running vision_session.run(...)");
            let t_run = Instant::now();
            let vision_outputs = ok(vision_session.run(ort::inputs! {
                "pixel_values" => pixel_values_tensor,
                "image_grid_thw" => image_grid_thw_tensor,
            }))?;
            log_info(format!(
                "vision_session.run finished (elapsed: {:.3}s)",
                t_run.elapsed().as_secs_f64()
            ));
            ok(vision_outputs[0].try_extract_array::<f32>())?
                .into_dimensionality::<Ix2>()
                .context("vision_encoder output is not 2D")?
                .to_owned()
        };
        log_info(format!("image_features shape={:?}", image_features.shape()));
        log_stage_end(stage, stage_start);

        let stage = "Build prompt and text embeddings";
        let stage_start = Instant::now();
        log_stage_start(stage);
        let prompt = build_prompt();
        log_info(format!("Prompt length: {} chars", prompt.len()));
        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!(e.to_string()))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let image_pos = input_ids
            .iter()
            .position(|&x| x == IMAGE_TOKEN_ID)
            .ok_or_else(|| anyhow!("<|image|> token ({IMAGE_TOKEN_ID}) not found in prompt"))?;
        log_info(format!(
            "input_ids length={}, image_token position={}",
            input_ids.len(),
            image_pos
        ));

        let text_ids_array = Array2::from_shape_vec((1, input_ids.len()), input_ids.clone())?;
        let text_embeds = {
            let t = Instant::now();
            let embed_outputs = ok(embed_session.run(ort::inputs! {
                "input_ids" => ok(Tensor::from_array(text_ids_array.clone()))?,
            }))?;
            log_info(format!("Initial embed_session.run finished (elapsed: {:.3}s)", t.elapsed().as_secs_f64()));
            ok(embed_outputs[0].try_extract_array::<f32>())?
                .into_dimensionality::<Ix3>()
                .context("embed_tokens output is not 3D")?
                .to_owned()
        };
        log_info(format!("text_embeds shape={:?}", text_embeds.shape()));
        log_stage_end(stage, stage_start);

        let prefix = text_embeds.slice(s![0, 0..image_pos, ..]).to_owned();
        let suffix = text_embeds
            .slice(s![0, (image_pos + 1)..text_embeds.shape()[1], ..])
            .to_owned();

        let seq_len = prefix.shape()[0] + image_features.shape()[0] + suffix.shape()[0];
        let mut full_embeds = Array3::<f32>::zeros((1, seq_len, HIDDEN_SIZE));
        log_info(format!(
            "prefix_len={}, image_feature_tokens={}, suffix_len={}, full_seq_len={}",
            prefix.shape()[0],
            image_features.shape()[0],
            suffix.shape()[0],
            seq_len
        ));

        let mut cursor = 0usize;
        if prefix.shape()[0] > 0 {
            let n = prefix.shape()[0];
            full_embeds
                .slice_mut(s![0, cursor..(cursor + n), ..])
                .assign(&prefix);
            cursor += n;
        }

        let img_n = image_features.shape()[0];
        full_embeds
            .slice_mut(s![0, cursor..(cursor + img_n), ..])
            .assign(&image_features);
        cursor += img_n;

        if suffix.shape()[0] > 0 {
            let n = suffix.shape()[0];
            full_embeds
                .slice_mut(s![0, cursor..(cursor + n), ..])
                .assign(&suffix);
        }

        let (initial_position_ids, mrope_position_delta) = build_glm_mrope_positions(
            image_pos,
            input_ids.len(),
            &image_grid_thw,
            img_n,
            seq_len,
        )?;
        log_info(format!("mRoPE delta = {}", mrope_position_delta));

        let mut past_kv: Vec<ArrayD<f32>> = (0..NUM_LAYERS * 2)
            .map(|_| ArrayD::<f32>::zeros(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM])))
            .collect();

        let mut generated_ids: Vec<i64> = Vec::new();
        log_stage_start("Autoregressive decode loop");
        let decode_stage_start = Instant::now();
        let verbose = is_verbose();
        let stream_decode = if verbose {
            std::env::var("OCR_STREAM_DECODE")
                .ok()
                .map(|v| {
                    let v = v.trim().to_ascii_lowercase();
                    !(v == "0" || v == "false" || v == "off" || v == "no")
                })
                .unwrap_or(true)
        } else {
            true
        };

        let mut step = 0usize;
        loop {
            let step_start = Instant::now();
            let (current_embeds, current_position_ids) = if step == 0 {
                (full_embeds.clone(), initial_position_ids.clone())
            } else {
                let last = *generated_ids.last().unwrap();
                let ids = Array2::from_shape_vec((1, 1), vec![last])?;
                let embed = {
                    let t = Instant::now();
                    let out = ok(embed_session.run(ort::inputs! {
                        "input_ids" => ok(Tensor::from_array(ids.clone()))?,
                    }))?;
                    log_info(format!(
                        "step={} incremental embed finished (elapsed: {:.3}s)",
                        step,
                        t.elapsed().as_secs_f64()
                    ));
                    ok(out[0].try_extract_array::<f32>())?
                        .into_dimensionality::<Ix3>()
                        .context("Incremental embed output is not 3D")?
                        .to_owned()
                };

                let pos = past_kv
                    .first()
                    .and_then(|kv| kv.shape().get(2).copied())
                    .unwrap_or(0) as i64
                    + mrope_position_delta;
                let mut p = Array3::<i64>::zeros((3, 1, 1));
                p[[0, 0, 0]] = pos;
                p[[1, 0, 0]] = pos;
                p[[2, 0, 0]] = pos;
                (embed, p)
            };

            let past_len = past_kv
                .first()
                .and_then(|kv| kv.shape().get(2).copied())
                .unwrap_or(0);
            let total_len = past_len + current_embeds.shape()[1];
            let attention_mask = Array2::<i64>::ones((1, total_len));
            let num_logits_to_keep = Array::from_shape_vec(IxDyn(&[]), vec![1i64])?;

            let mut decoder_inputs: Vec<(std::borrow::Cow<'_, str>, SessionInputValue<'_>)> = ort::inputs! {
                "inputs_embeds" => ok(Tensor::from_array(current_embeds.clone()))?,
                "attention_mask" => ok(Tensor::from_array(attention_mask.clone()))?,
                "position_ids" => ok(Tensor::from_array(current_position_ids.clone()))?,
                "num_logits_to_keep" => ok(Tensor::from_array(num_logits_to_keep.clone()))?,
            };

            for i in 0..NUM_LAYERS {
                let k_name = format!("past_key_values.{i}.key");
                let v_name = format!("past_key_values.{i}.value");
                decoder_inputs.push((
                    k_name.into(),
                    ok(Tensor::from_array(past_kv[2 * i].clone()))?.into(),
                ));
                decoder_inputs.push((
                    v_name.into(),
                    ok(Tensor::from_array(past_kv[2 * i + 1].clone()))?.into(),
                ));
            }

            if step < 5 || step % 10 == 0 {
                log_info(format!(
                    "step={} preparing decoder_session.run, attention_total_len={}",
                    step, total_len
                ));
            }
            let t = Instant::now();
            let outputs = ok(decoder_session.run(decoder_inputs))?;
            if step < 5 || step % 10 == 0 {
                log_info(format!(
                    "step={} decoder_session.run finished (elapsed: {:.3}s)",
                    step,
                    t.elapsed().as_secs_f64()
                ));
            }

            let logits = ok(outputs[0].try_extract_array::<f32>())?
                .into_dimensionality::<Ix3>()
                .context("Decoder logits output is not 3D")?
                .to_owned();

            let vocab_slice = logits.index_axis(Axis(0), 0).index_axis(Axis(0), 0).to_owned();
            let next = pick_next_token(vocab_slice.as_slice().unwrap_or(&[]), step);
            if step < 5 || step % 10 == 0 {
                log_info(format!(
                    "step={} next_token_id={}, step_elapsed={:.3}s",
                    step,
                    next,
                    step_start.elapsed().as_secs_f64()
                ));
            }

            generated_ids.push(next);

            if stream_decode {
                let piece = tokenizer
                    .decode(&[next as u32], true)
                    .map_err(|e| anyhow!(e.to_string()))?;
                if !piece.is_empty() {
                    if verbose {
                        log_info(format!(
                            "[STREAM] step={}, piece={}",
                            step,
                            piece.replace('\n', "\\n")
                        ));
                    } else {
                        print!("{}", piece);
                        let _ = std::io::stdout().flush();
                    }
                }
            }

            if EOS_TOKEN_IDS.contains(&next) {
                log_info(format!("step={} hit EOS token={}, stopping decode early", step, next));
                break;
            }

            let mut new_past = Vec::<ArrayD<f32>>::with_capacity(NUM_LAYERS * 2);
            for i in 0..(NUM_LAYERS * 2) {
                let t = ok(outputs[i + 1].try_extract_array::<f32>())?.to_owned();
                new_past.push(t);
            }
            past_kv = new_past;
            step += 1;
        }
        log_stage_end("Autoregressive decode loop", decode_stage_start);

        let gen_u32: Vec<u32> = generated_ids.iter().map(|&x| x as u32).collect();
        let decoded = tokenizer
            .decode(&gen_u32, true)
            .map_err(|e| anyhow!(e.to_string()))?;

        log_info(format!("Generated token count: {}", generated_ids.len()));
        log_info(format!("Total elapsed: {:.3}s", total_start.elapsed().as_secs_f64()));
        log_stage_end("GLM OCR ONNX Rust inference", total_start);

        Ok(InferResult {
            text: decoded,
            token_count: generated_ids.len(),
        })
    }
}
