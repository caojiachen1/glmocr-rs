use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Instant;
use std::io::Write;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_core::utils::cuda_is_available;
use candle_nn::ops;
use image::{imageops::FilterType, DynamicImage};
use ndarray::{s, Array2, Array3, Array4};
use serde::Deserialize;
use tokenizers::Tokenizer;
use rayon::ThreadPoolBuilder;

use super::{OcrBackend, InferResult, is_verbose, log_info, log_stage_start, log_stage_end, log_stream};

const TAG: &str = "NATIVE";
const IMAGE_TOKEN_ID: i64 = 59280;
const EOS_TOKEN_IDS: [i64; 2] = [59246, 59253];
const PATCH_SIZE: usize = 14;
const TEMPORAL_PATCH_SIZE: usize = 2;
const MERGE_SIZE: usize = 2;

/// Convert candle error to anyhow error - inline for performance
#[inline(always)]
fn c<T>(r: candle_core::Result<T>) -> Result<T> {
    r.map_err(|e| anyhow!(e.to_string()))
}

static RT_INIT: Once = Once::new();

/// Detect and return the best available device (CUDA > CPU)
/// If force_cpu is true, always return CPU device
fn get_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        log_info(TAG, "CPU mode forced: using CPU device");
        return Ok(Device::Cpu);
    }
    
    if cuda_is_available() {
        log_info(TAG, "CUDA is available, using GPU device");
        match Device::new_cuda(0) {
            Ok(device) => {
                log_info(TAG, format!("CUDA device initialized: {:?}", device));
                Ok(device)
            }
            Err(e) => {
                log_info(TAG, format!("Failed to initialize CUDA device: {}, falling back to CPU", e));
                Ok(Device::Cpu)
            }
        }
    } else {
        log_info(TAG, "CUDA is not available, using CPU device");
        Ok(Device::Cpu)
    }
}

fn init_runtime() {
    RT_INIT.call_once(|| {
        if let Ok(nz) = std::thread::available_parallelism() {
            let n = nz.get();
            let _ = std::env::set_var("RAYON_NUM_THREADS", n.to_string());
            let _ = ThreadPoolBuilder::new().num_threads(n).build_global();
            log_info(TAG, format!("Initialized CPU thread pool: rayon_threads={}", n));
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let avx = std::is_x86_feature_detected!("avx");
            let avx2 = std::is_x86_feature_detected!("avx2");
            let fma = std::is_x86_feature_detected!("fma");
            let avx512f = std::is_x86_feature_detected!("avx512f");
            let bf16 = std::is_x86_feature_detected!("avx512bf16");
            log_info(TAG, format!(
                "CPU features: avx={}, avx2={}, fma={}, avx512f={}, avx512bf16={}",
                avx, avx2, fma, avx512f, bf16
            )); 
        }
    });
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParams {
    #[serde(default)]
    mrope_section: Vec<usize>,
    #[serde(default = "default_theta")]
    rope_theta: f64,
}

fn default_theta() -> f64 {
    10000.0
}

#[derive(Debug, Clone, Deserialize)]
struct TextConfig {
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_parameters: RopeParams,
}

#[derive(Debug, Clone, Deserialize)]
struct VisionConfig {
    hidden_size: usize,
    depth: usize,
    num_heads: usize,
    out_hidden_size: usize,
    rms_norm_eps: f64,
    spatial_merge_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelConfig {
    text_config: TextConfig,
    vision_config: VisionConfig,
    image_token_id: i64,
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

#[inline(always)]
fn to_rgb(img: DynamicImage) -> image::RgbImage {
    img.to_rgb8()
}

/// Optimized image preprocessing using parallel pixel processing
fn preprocess_image(
    image_path: &Path,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(Array2<f32>, Array2<i64>)> {
    let image = image::open(image_path)
        .with_context(|| format!("Failed to open input image: {}", image_path.display()))?;
    let rgb = to_rgb(image);

    let (orig_w, orig_h) = rgb.dimensions();
    let (target_h, target_w) = smart_resize(
        orig_h as usize,
        orig_w as usize,
        PATCH_SIZE * MERGE_SIZE,
        min_pixels,
        max_pixels,
    );

    let resized = image::imageops::resize(
        &rgb,
        target_w as u32,
        target_h as u32,
        FilterType::CatmullRom,
    );

    const IMAGE_MEAN: [f32; 3] = [0.48145466f32, 0.4578275f32, 0.40821073f32];
    const IMAGE_STD: [f32; 3] = [0.26862954f32, 0.26130258f32, 0.27577711f32];

    // Optimized pixel normalization
    let mut chw = Array3::<f32>::zeros((3, target_h, target_w));
    for c in 0..3 {
        for y in 0..target_h {
            for x in 0..target_w {
                let p = resized.get_pixel(x as u32, y as u32).0;
                let v = p[c] as f32 / 255.0;
                chw[[c, y, x]] = (v - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }
    }

    let mut frames = Array4::<f32>::zeros((TEMPORAL_PATCH_SIZE, 3, target_h, target_w));
    for t in 0..TEMPORAL_PATCH_SIZE {
        frames.slice_mut(s![t, .., .., ..]).assign(&chw);
    }

    let grid_h = target_h / PATCH_SIZE;
    let grid_w = target_w / PATCH_SIZE;

    // Pre-allocate with exact capacity to avoid reallocations
    let num_patches = grid_h * grid_w;
    let patch_elements = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE; // 1176
    let mut pixel_values = Vec::<f32>::with_capacity(num_patches * patch_elements);
    
    // Optimized patch extraction with better memory access pattern
    for gh_block in 0..(grid_h / MERGE_SIZE) {
        for gw_block in 0..(grid_w / MERGE_SIZE) {
            for gh_inner in 0..MERGE_SIZE {
                for gw_inner in 0..MERGE_SIZE {
                    let gh = gh_block * MERGE_SIZE + gh_inner;
                    let gw = gw_block * MERGE_SIZE + gw_inner;
                    for c in 0..3 {
                        for t in 0..TEMPORAL_PATCH_SIZE {
                            let base_y = gh * PATCH_SIZE;
                            let base_x = gw * PATCH_SIZE;
                            for ph in 0..PATCH_SIZE {
                                for pw in 0..PATCH_SIZE {
                                    let y = base_y + ph;
                                    let x = base_x + pw;
                                    pixel_values.push(frames[[t, c, y, x]]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let pixel_values = Array2::from_shape_vec((num_patches, patch_elements), pixel_values)
        .context("Failed to build pixel_values")?;
    let image_grid_thw = Array2::from_shape_vec((1, 3), vec![1i64, grid_h as i64, grid_w as i64])
        .context("Failed to build image_grid_thw")?;
    Ok((pixel_values, image_grid_thw))
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
            "Image token count mismatch: expected={}, actual={}",
            expected_image_tokens,
            image_token_count
        );
    }

    let mut rows = [
        Vec::<i64>::with_capacity(merged_seq_len),
        Vec::<i64>::with_capacity(merged_seq_len),
        Vec::<i64>::with_capacity(merged_seq_len),
    ];

    let mut current_pos = 0i64;

    // Prefix positions
    for i in 0..image_pos {
        let v = current_pos + i as i64;
        rows[0].push(v);
        rows[1].push(v);
        rows[2].push(v);
    }
    current_pos += image_pos as i64;

    // Image positions
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

    // Suffix positions
    let suffix_len = text_seq_len - image_pos - 1;
    for i in 0..suffix_len {
        let v = current_pos + i as i64;
        rows[0].push(v);
        rows[1].push(v);
        rows[2].push(v);
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

#[inline(always)]
fn pick_next_token(logits: &[f32]) -> i64 {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i64)
        .unwrap_or(EOS_TOKEN_IDS[0])
}

/// Optimized linear layer - inspired by vllm.rs Linear::forward
/// Uses broadcast_left for efficient batch matmul and avoids unnecessary clones
fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    // Ensure compatible dtypes
    let x = if x.dtype() != w.dtype() {
        c(x.to_dtype(w.dtype()))?
    } else {
        x.clone()
    };
    
    // Transpose weight once
    let wt = c(w.t())?;
    
    // Optimized matmul based on input dimensions - matching vllm.rs strategy
    // For 3D inputs (batch, seq, hidden), use broadcast_matmul
    // For 2D inputs (seq, hidden), use regular matmul
    let y = if x.dims().len() == 3 {
        c(x.broadcast_matmul(&wt))
    } else {
        c(x.matmul(&wt))
    }?;
    
    match b {
        Some(bias) => c(y.broadcast_add(bias)),
        None => Ok(y),
    }
}

/// Optimized RMS norm - fused operations, reduced allocations
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    // Only convert to f32 if necessary
    let x_f = if dtype == DType::F32 {
        x.clone()
    } else {
        c(x.to_dtype(DType::F32))?
    };
    
    let var = c(x_f.sqr())?;
    let var = c(var.mean_keepdim(D::Minus1))?;
    let inv = c(c(c(&var + eps)?.sqrt())?.recip())?;
    let normed = c(x_f.broadcast_mul(&inv))?;
    
    // Convert back to original dtype if needed
    let normed = if dtype != DType::F32 {
        c(normed.to_dtype(dtype))?
    } else {
        normed
    };
    
    c(normed.broadcast_mul(weight))
}

/// Optimized layer norm
fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f = c(x.to_dtype(DType::F32))?;
    let mean = c(x_f.mean_keepdim(D::Minus1))?;
    let centered = c(x_f.broadcast_sub(&mean))?;
    let var = c(c(centered.sqr())?.mean_keepdim(D::Minus1))?;
    let inv = c(c(c(&var + eps)?.sqrt())?.recip())?;
    let y = c(centered.broadcast_mul(&inv))?;
    let y = c(y.broadcast_mul(weight))?;
    c(y.broadcast_add(bias))
}

/// Optimized rotate_half for vision - reduces allocations
fn rotate_half_vision(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1).map_err(|e| anyhow!(e.to_string()))?;
    let half = d / 2;
    let x1 = c(x.narrow(D::Minus1, 0, half))?;
    let x2 = c(x.narrow(D::Minus1, half, half))?;
    c(Tensor::cat(&[&c(x2.neg())?, &x1], D::Minus1))
}

/// Optimized rotate_half for text - reduces allocations
fn rotate_half_text(x: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4().map_err(|e| anyhow!(e.to_string()))?;
    let x = c(x.reshape((b, h, s, d / 2, 2)))?;
    let x_even = c(x.i((.., .., .., .., 0)))?;
    let x_odd = c(x.i((.., .., .., .., 1)))?;
    let x_odd = c(x_odd.neg())?;
    let y = c(Tensor::stack(&[&x_odd, &x_even], D::Minus1))?;
    c(y.flatten_from(D::Minus2))
}

/// Pre-computed text RoPE cos/sin with optimized vector operations
fn compute_text_cos_sin(cfg: &TextConfig, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
    let (_, b, seq) = position_ids.dims3().map_err(|e| anyhow!(e.to_string()))?;
    if b != 1 {
        bail!("Only batch=1 is currently supported");
    }
    
    let dim = cfg.head_dim;
    let inv_len = dim / 2;
    let rope_theta = cfg.rope_parameters.rope_theta as f32;
    
    // Pre-compute inv_freq once
    let inv_freq: Vec<f32> = (0..inv_len)
        .map(|i| 1f32 / rope_theta.powf((2 * i) as f32 / dim as f32))
        .collect();

    let pos = c(position_ids.to_dtype(DType::I64))?.to_vec3::<i64>().map_err(|e| anyhow!(e.to_string()))?;

    // Pre-allocate freqs3 with exact dimensions
    let mut freqs3 = vec![vec![vec![0f32; inv_len]; seq]; 3];

    for axis in 0..3 {
        for s in 0..seq {
            let p = pos[axis][0][s] as f32;
            let freq_row = &mut freqs3[axis][s];
            for i in 0..inv_len {
                freq_row[i] = p * inv_freq[i];
            }
        }
    }

    let mut sections = cfg.rope_parameters.mrope_section.clone();
    if sections.is_empty() {
        sections = vec![16, 24, 24];
    }

    // Build mrope frequencies
    let mut freq_mrope = vec![0f32; seq * inv_len];
    let mut offset = 0usize;
    for (i, sec) in sections.iter().enumerate() {
        let axis = i % 3;
        for s in 0..seq {
            let base = s * inv_len;
            let src_row = &freqs3[axis][s];
            for j in 0..*sec {
                freq_mrope[base + offset + j] = src_row[offset + j];
            }
        }
        offset += *sec;
    }

    if offset != inv_len {
        bail!(
            "Invalid mrope_section total length: sum={} != head_dim/2={}",
            offset,
            inv_len
        );
    }

    // Build cos/sin with repeat_interleave(2) pattern
    let mut cos = vec![0f32; seq * dim];
    let mut sin = vec![0f32; seq * dim];
    for s in 0..seq {
        let base_freq = s * inv_len;
        let base_out = s * dim;
        for i in 0..inv_len {
            let v = freq_mrope[base_freq + i];
            let c = v.cos();
            let s_val = v.sin();
            cos[base_out + 2 * i] = c;
            cos[base_out + 2 * i + 1] = c;
            sin[base_out + 2 * i] = s_val;
            sin[base_out + 2 * i + 1] = s_val;
        }
    }

    let dev = position_ids.device();
    let cos = c(Tensor::from_vec(cos, (1, seq, dim), dev))?;
    let sin = c(Tensor::from_vec(sin, (1, seq, dim), dev))?;
    Ok((cos, sin))
}

/// Optimized text rotary application - reduces intermediate allocations
fn apply_text_rotary(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let cos = c(cos.unsqueeze(1))?;
    let sin = c(sin.unsqueeze(1))?;
    let q_rot = c(c(q.broadcast_mul(&cos))? + c(rotate_half_text(q)?.broadcast_mul(&sin))?)?;
    let k_rot = c(c(k.broadcast_mul(&cos))? + c(rotate_half_text(k)?.broadcast_mul(&sin))?)?;
    Ok((q_rot, k_rot))
}

/// Optimized causal mask - pre-allocate and fill efficiently
fn causal_mask(device: &Device, q_len: usize, k_len: usize, past: usize) -> Result<Tensor> {
    let mut v = vec![0f32; q_len * k_len];
    for i in 0..q_len {
        let allowed = past + i;
        let row_start = i * k_len;
        for j in (allowed + 1)..k_len {
            v[row_start + j] = f32::NEG_INFINITY;
        }
    }
    c(Tensor::from_vec(v, (1, 1, q_len, k_len), device))
}

/// Optimized grouped KV attention - inspired by vllm.rs
/// Reduces clones, uses more efficient tensor operations
fn attention_grouped_kv(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kv_repeat: usize,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, n_heads, _q_len, head_dim) = q.dims4().map_err(|e| anyhow!(e.to_string()))?;
    let (bk, n_kv_heads, k_len, k_head_dim) = k.dims4().map_err(|e| anyhow!(e.to_string()))?;
    let (_, n_kv_heads_v, v_len, v_head_dim) = v.dims4().map_err(|e| anyhow!(e.to_string()))?;

    if b != bk || n_kv_heads != n_kv_heads_v || k_len != v_len || head_dim != k_head_dim || head_dim != v_head_dim {
        bail!(
            "attention_grouped_kv dimension mismatch: q={:?}, k={:?}, v={:?}",
            q.dims(),
            k.dims(),
            v.dims()
        );
    }
    if n_heads != n_kv_heads * kv_repeat {
        bail!(
            "attention_grouped_kv head count mismatch: n_heads={} != n_kv_heads({})*kv_repeat({})",
            n_heads,
            n_kv_heads,
            kv_repeat
        );
    }

    // Process each KV head group
    let mut parts = Vec::<Tensor>::with_capacity(n_kv_heads);
    for kvh in 0..n_kv_heads {
        let h_start = kvh * kv_repeat;
        
        // Extract q group - avoid unnecessary contiguous when possible
        let q_g = c(q.narrow(1, h_start, kv_repeat))?;
        let k_g = c(k.narrow(1, kvh, 1))?;
        let v_g = c(v.narrow(1, kvh, 1))?;
        
        // Broadcast k and v to match q's head count, ensure contiguous for CUDA matmul
        let k_g = c(c(k_g.broadcast_as((b, kv_repeat, k_len, head_dim)))?.contiguous())?;
        let v_g = c(c(v_g.broadcast_as((b, kv_repeat, v_len, head_dim)))?.contiguous())?;

        // Compute attention scores
        let k_t = c(c(k_g.transpose(2, 3))?.contiguous())?;
        let mut scores = c(c(q_g.matmul(&k_t))? * scale)?;
        
        if let Some(attn_mask) = mask {
            scores = c(scores.broadcast_add(attn_mask))?;
        }

        let probs = ops::softmax_last_dim(&scores).map_err(|e| anyhow!(e.to_string()))?;
        let ctx_g = c(probs.matmul(&v_g))?;
        parts.push(ctx_g);
    }

    let part_refs: Vec<&Tensor> = parts.iter().collect();
    c(Tensor::cat(&part_refs, 1))
}

#[derive(Default)]
struct LayerCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

struct TextLayerWeights {
    in_ln: Tensor,
    pa_ln: Tensor,
    psa_ln: Tensor,
    pm_ln: Tensor,
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor,
    gate_up: Tensor,
    down: Tensor,
}

struct VisionLayerWeights {
    n1: Tensor,
    n2: Tensor,
    qkv_w: Tensor,
    qkv_b: Option<Tensor>,
    qn: Tensor,
    kn: Tensor,
    proj_w: Tensor,
    proj_b: Option<Tensor>,
    gate_w: Tensor,
    gate_b: Option<Tensor>,
    up_w: Tensor,
    up_b: Option<Tensor>,
    down_w: Tensor,
    down_b: Option<Tensor>,
}

struct VisionWeights {
    patch_w: Tensor,
    patch_b: Option<Tensor>,
    blocks: Vec<VisionLayerWeights>,
    post_norm: Tensor,
    downsample_w: Tensor,
    downsample_b: Option<Tensor>,
    merger_proj: Tensor,
    merger_post_ln_w: Tensor,
    merger_post_ln_b: Tensor,
    merger_gate_w: Tensor,
    merger_up_w: Tensor,
    merger_down_w: Tensor,
}

struct SafetensorsModel {
    cfg: ModelConfig,
    device: Device,
    text_layers: Vec<TextLayerWeights>,
    vision: VisionWeights,
    embed_tokens_w: Tensor,
    text_norm_w: Tensor,
    lm_head_w: Tensor,
    text_inv_freq: Vec<f32>,
    kv_repeat: usize,
}

impl SafetensorsModel {
    fn load(model_root: &Path, force_cpu: bool) -> Result<Self> {
        let total_t = Instant::now();
        log_stage_start(TAG, "Load safetensors model");

        let config_path = model_root.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config: {}", config_path.display()))?;
        let cfg: ModelConfig = serde_json::from_str(&config_str).context("Failed to parse config.json")?;
        log_info(TAG, format!("Config loaded: {}", config_path.display()));

        let weights_path = model_root.join("model.safetensors");
        let device = get_device(force_cpu)?;
        log_info(TAG, format!("Loading weights: {}", weights_path.display()));
        let tensors = c(candle_core::safetensors::load(&weights_path, &device))
            .with_context(|| format!("Failed to load safetensors: {}", weights_path.display()))?;
        log_info(TAG, format!("Weights loaded, total tensors: {}", tensors.len()));

        // Helper closures for loading weights
        let get_required_f32 = |name: &str| -> Result<Tensor> {
            let t = tensors
                .get(name)
                .ok_or_else(|| anyhow!("Missing weight: {name}"))?
                .clone();
            if t.dtype() == DType::F32 {
                Ok(t)
            } else {
                c(t.to_dtype(DType::F32))
            }
        };

        let get_optional_f32 = |name: &str| -> Result<Option<Tensor>> {
            match tensors.get(name) {
                Some(t) => {
                    let t = t.clone();
                    if t.dtype() == DType::F32 {
                        Ok(Some(t))
                    } else {
                        Ok(Some(c(t.to_dtype(DType::F32))?))
                    }
                }
                None => Ok(None),
            }
        };

        // Load text layers
        let mut text_layers = Vec::with_capacity(cfg.text_config.num_hidden_layers);
        for layer in 0..cfg.text_config.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{layer}");
            let lw = TextLayerWeights {
                in_ln: get_required_f32(&format!("{prefix}.input_layernorm.weight"))?,
                pa_ln: get_required_f32(&format!("{prefix}.post_attention_layernorm.weight"))?,
                psa_ln: get_required_f32(&format!("{prefix}.post_self_attn_layernorm.weight"))?,
                pm_ln: get_required_f32(&format!("{prefix}.post_mlp_layernorm.weight"))?,
                q_w: get_required_f32(&format!("{prefix}.self_attn.q_proj.weight"))?,
                k_w: get_required_f32(&format!("{prefix}.self_attn.k_proj.weight"))?,
                v_w: get_required_f32(&format!("{prefix}.self_attn.v_proj.weight"))?,
                o_w: get_required_f32(&format!("{prefix}.self_attn.o_proj.weight"))?,
                gate_up: get_required_f32(&format!("{prefix}.mlp.gate_up_proj.weight"))?,
                down: get_required_f32(&format!("{prefix}.mlp.down_proj.weight"))?,
            };
            text_layers.push(lw);
        }

        // Load vision blocks
        let mut vision_blocks = Vec::with_capacity(cfg.vision_config.depth);
        for layer in 0..cfg.vision_config.depth {
            let prefix = format!("model.visual.blocks.{layer}");
            vision_blocks.push(VisionLayerWeights {
                n1: get_required_f32(&format!("{prefix}.norm1.weight"))?,
                n2: get_required_f32(&format!("{prefix}.norm2.weight"))?,
                qkv_w: get_required_f32(&format!("{prefix}.attn.qkv.weight"))?,
                qkv_b: get_optional_f32(&format!("{prefix}.attn.qkv.bias"))?,
                qn: get_required_f32(&format!("{prefix}.attn.q_norm.weight"))?,
                kn: get_required_f32(&format!("{prefix}.attn.k_norm.weight"))?,
                proj_w: get_required_f32(&format!("{prefix}.attn.proj.weight"))?,
                proj_b: get_optional_f32(&format!("{prefix}.attn.proj.bias"))?,
                gate_w: get_required_f32(&format!("{prefix}.mlp.gate_proj.weight"))?,
                gate_b: get_optional_f32(&format!("{prefix}.mlp.gate_proj.bias"))?,
                up_w: get_required_f32(&format!("{prefix}.mlp.up_proj.weight"))?,
                up_b: get_optional_f32(&format!("{prefix}.mlp.up_proj.bias"))?,
                down_w: get_required_f32(&format!("{prefix}.mlp.down_proj.weight"))?,
                down_b: get_optional_f32(&format!("{prefix}.mlp.down_proj.bias"))?,
            });
        }

        let vision = VisionWeights {
            patch_w: get_required_f32("model.visual.patch_embed.proj.weight")?,
            patch_b: get_optional_f32("model.visual.patch_embed.proj.bias")?,
            blocks: vision_blocks,
            post_norm: get_required_f32("model.visual.post_layernorm.weight")?,
            downsample_w: get_required_f32("model.visual.downsample.weight")?,
            downsample_b: get_optional_f32("model.visual.downsample.bias")?,
            merger_proj: get_required_f32("model.visual.merger.proj.weight")?,
            merger_post_ln_w: get_required_f32("model.visual.merger.post_projection_norm.weight")?,
            merger_post_ln_b: get_required_f32("model.visual.merger.post_projection_norm.bias")?,
            merger_gate_w: get_required_f32("model.visual.merger.gate_proj.weight")?,
            merger_up_w: get_required_f32("model.visual.merger.up_proj.weight")?,
            merger_down_w: get_required_f32("model.visual.merger.down_proj.weight")?,
        };

        let embed_tokens_w = get_required_f32("model.language_model.embed_tokens.weight")?;
        let text_norm_w = get_required_f32("model.language_model.norm.weight")?;
        let lm_head_w = get_required_f32("lm_head.weight")?;

        // Pre-compute text inv_freq for decode
        let dim = cfg.text_config.head_dim;
        let inv_len = dim / 2;
        let rope_theta = cfg.text_config.rope_parameters.rope_theta as f32;
        let text_inv_freq: Vec<f32> = (0..inv_len)
            .map(|i| 1f32 / rope_theta.powf((2 * i) as f32 / dim as f32))
            .collect();

        let kv_repeat = cfg.text_config.num_attention_heads / cfg.text_config.num_key_value_heads;

        log_stage_end(TAG, "Load safetensors model", total_t);

        Ok(Self {
            cfg,
            device,
            text_layers,
            vision,
            embed_tokens_w,
            text_norm_w,
            lm_head_w,
            text_inv_freq,
            kv_repeat,
        })
    }

    /// Optimized embedding lookup
    fn embed_ids(&self, ids: &[i64]) -> Result<Tensor> {
        let ids_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        let ids_t = c(Tensor::from_vec(ids_u32, (ids.len(),), &self.device))?;
        let emb = c(self.embed_tokens_w.index_select(&ids_t, 0))?;
        c(emb.unsqueeze(0))
    }

    /// Optimized single position cos/sin for decode - uses pre-computed inv_freq
    fn compute_text_cos_sin_single_pos(&self, pos: i64) -> Result<(Tensor, Tensor)> {
        let dim = self.cfg.text_config.head_dim;
        let inv_len = dim / 2;
        let p = pos as f32;

        let mut cos = vec![0f32; dim];
        let mut sin = vec![0f32; dim];
        
        // Use pre-computed inv_freq
        for i in 0..inv_len {
            let v = p * self.text_inv_freq[i];
            let c_val = v.cos();
            let s_val = v.sin();
            cos[2 * i] = c_val;
            cos[2 * i + 1] = c_val;
            sin[2 * i] = s_val;
            sin[2 * i + 1] = s_val;
        }

        let cos = c(Tensor::from_vec(cos, (1, 1, dim), &self.device))?;
        let sin = c(Tensor::from_vec(sin, (1, 1, dim), &self.device))?;
        Ok((cos, sin))
    }

    /// Optimized vision forward - reduced allocations, better memory layout
    fn vision_forward(&self, pixel_values: &Array2<f32>, image_grid_thw: &Array2<i64>) -> Result<Tensor> {
        let vt = &self.cfg.vision_config;
        let vw = &self.vision;
        let dev = &self.device;
        
        log_info(TAG, format!(
            "vision_forward: num_patches={}, hidden_size={}, depth={}",
            pixel_values.shape()[0],
            vt.hidden_size,
            vt.depth
        ));

        // Direct tensor creation from slice - avoid collecting iterator
        let pv_slice = pixel_values.as_slice().unwrap();
        let x = c(Tensor::from_vec(
            pv_slice.to_vec(),
            (pixel_values.shape()[0], pixel_values.shape()[1]),
            dev,
        ))?;

        // Patch embedding - reshape weight once
        let patch_w = c(vw.patch_w.reshape((vt.hidden_size, 3 * 2 * 14 * 14)))?;
        let patch_b = vw.patch_b.as_ref();
        let mut hidden = linear(&x, &patch_w, patch_b)?;

        // 2D rotary for vision
        let grid_h = image_grid_thw[[0, 1]] as usize;
        let grid_w = image_grid_thw[[0, 2]] as usize;
        let merge = vt.spatial_merge_size;

        // Pre-allocate position vectors
        let seq = (grid_h / merge) * (grid_w / merge) * merge * merge;
        let mut h_pos = Vec::<usize>::with_capacity(seq);
        let mut w_pos = Vec::<usize>::with_capacity(seq);
        
        for gh_block in 0..(grid_h / merge) {
            for gw_block in 0..(grid_w / merge) {
                for gh_inner in 0..merge {
                    for gw_inner in 0..merge {
                        h_pos.push(gh_block * merge + gh_inner);
                        w_pos.push(gw_block * merge + gw_inner);
                    }
                }
            }
        }

        let head_dim = vt.hidden_size / vt.num_heads;
        let inv_len = (head_dim / 2) / 2;
        
        // Pre-compute inv_freq for vision
        let inv_freq: Vec<f32> = (0..inv_len)
            .map(|i| 1f32 / 10000f32.powf((2 * i) as f32 / (head_dim / 2) as f32))
            .collect();

        let max_grid = grid_h.max(grid_w);
        let mut table = vec![0f32; max_grid * inv_len];
        for p in 0..max_grid {
            let base = p * inv_len;
            for i in 0..inv_len {
                table[base + i] = p as f32 * inv_freq[i];
            }
        }

        // Build cos/sin tables
        let mut cos = vec![0f32; seq * head_dim];
        let mut sin = vec![0f32; seq * head_dim];
        for i in 0..seq {
            let base = i * head_dim;
            for j in 0..inv_len {
                let hv = table[h_pos[i] * inv_len + j];
                let wv = table[w_pos[i] * inv_len + j];
                cos[base + j] = hv.cos();
                cos[base + inv_len + j] = wv.cos();
                sin[base + j] = hv.sin();
                sin[base + inv_len + j] = wv.sin();
                // duplicate for full head_dim
                cos[base + head_dim / 2 + j] = hv.cos();
                cos[base + head_dim / 2 + inv_len + j] = wv.cos();
                sin[base + head_dim / 2 + j] = hv.sin();
                sin[base + head_dim / 2 + inv_len + j] = wv.sin();
            }
        }
        
        let cos_t = c(Tensor::from_vec(cos, (seq, head_dim), dev))?;
        let sin_t = c(Tensor::from_vec(sin, (seq, head_dim), dev))?;

        // Vision transformer blocks
        for layer in 0..vt.depth {
            let layer_t = Instant::now();
            let lw = &vw.blocks[layer];

            // Attention path
            let a_in = rms_norm(&hidden, &lw.n1, vt.rms_norm_eps)?;

            let mut qkv = linear(&a_in, &lw.qkv_w, lw.qkv_b.as_ref())?;
            qkv = c(qkv.reshape((seq, 3, vt.num_heads, head_dim)))?;
            
            // Extract q, k, v and ensure contiguous
            let mut q = c(qkv.i((.., 0, .., ..)).and_then(|t| t.contiguous()))?;
            let mut k = c(qkv.i((.., 1, .., ..)).and_then(|t| t.contiguous()))?;
            let v = c(qkv.i((.., 2, .., ..)).and_then(|t| t.contiguous()))?;

            // Apply per-head norm
            q = rms_norm(&q, &lw.qn, vt.rms_norm_eps)?;
            k = rms_norm(&k, &lw.kn, vt.rms_norm_eps)?;

            // Apply rotary embeddings
            let cos_u = c(cos_t.unsqueeze(D::Minus2))?;
            let sin_u = c(sin_t.unsqueeze(D::Minus2))?;
            q = c(c(q.broadcast_mul(&cos_u))? + c(rotate_half_vision(&q)?.broadcast_mul(&sin_u))?)?;
            k = c(c(k.broadcast_mul(&cos_u))? + c(rotate_half_vision(&k)?.broadcast_mul(&sin_u))?)?;

            // Reshape for attention: [seq, heads, dim] -> [1, heads, seq, dim]
            let q = c(c(q.transpose(0, 1))?.unsqueeze(0))?;
            let q = c(q.contiguous())?;
            let k = c(c(k.transpose(0, 1))?.unsqueeze(0))?;
            let k = c(k.contiguous())?;
            let v = c(c(v.transpose(0, 1))?.unsqueeze(0))?;
            let v = c(v.contiguous())?;

            // Compute attention: each query token should attend to ALL key/value tokens
            // Fixed: removed the buggy chunked attention that only attended within chunks
            let k_t = c(c(k.transpose(2, 3))?.contiguous())?;
            let scores = c(c(q.matmul(&k_t))? * (1.0 / (head_dim as f64).sqrt()))?;
            let probs = ops::softmax_last_dim(&scores).map_err(|e| anyhow!(e.to_string()))?;
            let attn = c(probs.matmul(&v))?;
            let attn_chunks = vec![attn];
            
            let attn = if attn_chunks.len() == 1 {
                attn_chunks.into_iter().next().unwrap()
            } else {
                c(Tensor::cat(&attn_chunks.iter().collect::<Vec<_>>(), 2))?
            };
            
            // Reshape back: [1, heads, seq, dim] -> [seq, heads*dim]
            let attn = c(c(attn.squeeze(0))?.transpose(0, 1))?;
            let attn = c(attn.reshape((seq, vt.hidden_size)))?;

            let attn_out = linear(&attn, &lw.proj_w, lw.proj_b.as_ref())?;

            // Residual connection
            let h1 = c(&hidden + &attn_out)?;
            
            // MLP path
            let m_in = rms_norm(&h1, &lw.n2, vt.rms_norm_eps)?;
            let gate = c(linear(&m_in, &lw.gate_w, lw.gate_b.as_ref())?.silu())?;
            let up = linear(&m_in, &lw.up_w, lw.up_b.as_ref())?;
            let mlp = linear(&c(&gate * &up)?, &lw.down_w, lw.down_b.as_ref())?;

            hidden = c(&h1 + &mlp)?;

            if layer < 2 || layer + 1 == vt.depth || (layer + 1) % 4 == 0 {
                log_info(TAG, format!(
                    "vision block progress: {}/{} (layer elapsed: {:.3}s)",
                    layer + 1,
                    vt.depth,
                    layer_t.elapsed().as_secs_f64()
                ));
            }
        }

        // Post-norm
        hidden = rms_norm(&hidden, &vw.post_norm, vt.rms_norm_eps)?;

        // Downsample conv2d(kernel=2,stride=2) == linear over flattened 2x2x1024
        let n_groups = seq / 4;
        let hidden = c(hidden.reshape((n_groups, 2, 2, vt.hidden_size)))?;
        let hidden = c(hidden.permute((0, 3, 1, 2)))?;
        let hidden = c(hidden.reshape((n_groups, vt.hidden_size * 4)))?;
        let ds_w = c(vw.downsample_w.reshape((vt.out_hidden_size, vt.hidden_size * 4)))?;
        let ds_b = vw.downsample_b.as_ref();
        let mut hidden = linear(&hidden, &ds_w, ds_b)?;

        // Merger
        hidden = linear(&hidden, &vw.merger_proj, None)?;
        hidden = c(layer_norm(&hidden, &vw.merger_post_ln_w, &vw.merger_post_ln_b, 1e-5)?.gelu())?;

        let gate = c(linear(&hidden, &vw.merger_gate_w, None)?.silu())?;
        let up = linear(&hidden, &vw.merger_up_w, None)?;
        hidden = linear(&c(&gate * &up)?, &vw.merger_down_w, None)?;

        Ok(hidden)
    }

    /// Optimized text prefill - reduced allocations, better tensor reuse
    fn text_forward_prefill(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        cache: &mut [LayerCache],
    ) -> Result<Tensor> {
        let tc = &self.cfg.text_config;
        let (b, seq, _) = inputs_embeds.dims3().map_err(|e| anyhow!(e.to_string()))?;
        if b != 1 {
            bail!("Only batch=1 is currently supported");
        }
        
        let mut h = inputs_embeds.clone();
        let (cos, sin) = compute_text_cos_sin(tc, position_ids)?;
        let scale = 1.0 / (tc.head_dim as f64).sqrt();

        for layer in 0..tc.num_hidden_layers {
            let lw = &self.text_layers[layer];

            // Self-attention path
            let residual = h.clone();
            let x = rms_norm(&h, &lw.in_ln, tc.rms_norm_eps)?;

            // QKV projections - reshape in one go
            let q = c(linear(&x, &lw.q_w, None)?.reshape((1, seq, tc.num_attention_heads, tc.head_dim)))?;
            let k = c(linear(&x, &lw.k_w, None)?.reshape((1, seq, tc.num_key_value_heads, tc.head_dim)))?;
            let v = c(linear(&x, &lw.v_w, None)?.reshape((1, seq, tc.num_key_value_heads, tc.head_dim)))?;

            // Transpose for attention: [b, seq, heads, dim] -> [b, heads, seq, dim]
            let q = c(q.transpose(1, 2))?;
            let k = c(k.transpose(1, 2))?;
            let v = c(v.transpose(1, 2))?;

            // Apply rotary embeddings
            let (q, k) = apply_text_rotary(&q, &k, &cos, &sin)?;

            // Store in cache
            cache[layer].k = Some(k.clone());
            cache[layer].v = Some(v.clone());

            // Compute attention with causal mask
            let mask = causal_mask(&self.device, seq, seq, 0)?;
            let ctx = attention_grouped_kv(&q, &k, &v, self.kv_repeat, scale, Some(&mask))?;
            
            // Reshape and project
            let ctx = c(c(c(ctx.transpose(1, 2))?.contiguous())?.reshape((1, seq, tc.num_attention_heads * tc.head_dim)))?;
            let attn = linear(&ctx, &lw.o_w, None)?;
            let attn = rms_norm(&attn, &lw.psa_ln, tc.rms_norm_eps)?;
            h = c(&residual + &attn)?;

            // MLP path
            let residual2 = h.clone();
            let m_in = rms_norm(&h, &lw.pa_ln, tc.rms_norm_eps)?;
            let gu = linear(&m_in, &lw.gate_up, None)?;
            let d = gu.dim(D::Minus1).map_err(|e| anyhow!(e.to_string()))?;
            let half = d / 2;
            let gate = c(gu.narrow(D::Minus1, 0, half))?;
            let up = c(gu.narrow(D::Minus1, half, half))?;
            let mlp = linear(&c(c(gate.silu())? * up)?, &lw.down, None)?;
            let mlp = rms_norm(&mlp, &lw.pm_ln, tc.rms_norm_eps)?;
            h = c(&residual2 + &mlp)?;
        }

        // Final norm and project last token
        h = rms_norm(&h, &self.text_norm_w, tc.rms_norm_eps)?;
        let last_h = c(h.narrow(1, seq - 1, 1))?;
        linear(&last_h, &self.lm_head_w, None)
    }

    /// Optimized single token decode - minimal allocations
    fn text_forward_decode_one(
        &self,
        token_id: i64,
        pos: i64,
        cache: &mut [LayerCache],
    ) -> Result<Tensor> {
        let tc = &self.cfg.text_config;
        let mut h = self.embed_ids(&[token_id])?;

        let (cos, sin) = self.compute_text_cos_sin_single_pos(pos)?;
        let scale = 1.0 / (tc.head_dim as f64).sqrt();

        for layer in 0..tc.num_hidden_layers {
            let lw = &self.text_layers[layer];

            // Self-attention path
            let residual = h.clone();
            let x = rms_norm(&h, &lw.in_ln, tc.rms_norm_eps)?;

            // QKV for single token
            let q = c(linear(&x, &lw.q_w, None)?.reshape((1, 1, tc.num_attention_heads, tc.head_dim)))?;
            let k = c(linear(&x, &lw.k_w, None)?.reshape((1, 1, tc.num_key_value_heads, tc.head_dim)))?;
            let v = c(linear(&x, &lw.v_w, None)?.reshape((1, 1, tc.num_key_value_heads, tc.head_dim)))?;

            let q = c(q.transpose(1, 2))?;
            let k = c(k.transpose(1, 2))?;
            let v = c(v.transpose(1, 2))?;

            let (q, k) = apply_text_rotary(&q, &k, &cos, &sin)?;

            // Concatenate with cache
            let k_cat = match &cache[layer].k {
                None => k,
                Some(prev) => c(Tensor::cat(&[prev, &k], 2))?,
            };
            let v_cat = match &cache[layer].v {
                None => v,
                Some(prev) => c(Tensor::cat(&[prev, &v], 2))?,
            };

            cache[layer].k = Some(k_cat.clone());
            cache[layer].v = Some(v_cat.clone());

            // Attention without mask for decode (causal is implicit via cache)
            let ctx = attention_grouped_kv(&q, &k_cat, &v_cat, self.kv_repeat, scale, None)?;
            let ctx = c(c(c(ctx.transpose(1, 2))?.contiguous())?.reshape((1, 1, tc.num_attention_heads * tc.head_dim)))?;
            let attn = linear(&ctx, &lw.o_w, None)?;
            let attn = rms_norm(&attn, &lw.psa_ln, tc.rms_norm_eps)?;
            h = c(&residual + &attn)?;

            // MLP path
            let residual2 = h.clone();
            let m_in = rms_norm(&h, &lw.pa_ln, tc.rms_norm_eps)?;
            let gu = linear(&m_in, &lw.gate_up, None)?;
            let d = gu.dim(D::Minus1).map_err(|e| anyhow!(e.to_string()))?;
            let half = d / 2;
            let gate = c(gu.narrow(D::Minus1, 0, half))?;
            let up = c(gu.narrow(D::Minus1, half, half))?;
            let mlp = linear(&c(c(gate.silu())? * up)?, &lw.down, None)?;
            let mlp = rms_norm(&mlp, &lw.pm_ln, tc.rms_norm_eps)?;
            h = c(&residual2 + &mlp)?;
        }

        h = rms_norm(&h, &self.text_norm_w, tc.rms_norm_eps)?;
        linear(&h, &self.lm_head_w, None)
    }
}

pub struct NativeBackend {
    force_cpu: bool,
}

impl NativeBackend {
    pub fn new(force_cpu: bool) -> Self {
        Self { force_cpu }
    }

    fn model_paths(model_root: &Path) -> (PathBuf, PathBuf, PathBuf) {
        (
            model_root.join("tokenizer.json"),
            model_root.join("config.json"),
            model_root.join("model.safetensors"),
        )
    }
}

impl OcrBackend for NativeBackend {
    fn name(&self) -> &'static str {
        if self.force_cpu {
            "native (CPU forced)"
        } else {
            "native"
        }
    }

    fn infer(
        &mut self,
        model_root: &Path,
        image_path: &Path,
        min_pixels: usize,
        max_pixels: usize,
    ) -> Result<InferResult> {
        init_runtime();
        if cfg!(debug_assertions) {
            log_info(TAG, "Debug build detected; CPU inference can be much slower. Consider running `cargo run --release`");
        }
        let total_t = Instant::now();
        log_stage_start(TAG, "safetensors OCR inference");

        let (tokenizer_path, config_path, weights_path) = Self::model_paths(model_root);
        for p in [&tokenizer_path, &config_path, &weights_path] {
            if !p.exists() {
                bail!("Missing file: {}", p.display());
            }
        }
        log_info(TAG, "File check passed: tokenizer + config + safetensors all exist");

        let stage = "Load tokenizer";
        let stage_t = Instant::now();
        log_stage_start(TAG, stage);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!(e.to_string()))?;
        log_stage_end(TAG, stage, stage_t);

        let stage = "Load safetensors weights";
        let stage_t = Instant::now();
        log_stage_start(TAG, stage);
        let model = SafetensorsModel::load(model_root, self.force_cpu)?;
        log_stage_end(TAG, stage, stage_t);
        if model.cfg.image_token_id != IMAGE_TOKEN_ID {
            bail!(
                "image_token_id mismatch: config={}, code={}",
                model.cfg.image_token_id,
                IMAGE_TOKEN_ID
            );
        }

        let stage = "Image preprocessing + vision encoding";
        let stage_t = Instant::now();
        log_stage_start(TAG, stage);
        let (pixel_values, image_grid_thw) = preprocess_image(image_path, min_pixels, max_pixels)?;
        let image_features = model.vision_forward(&pixel_values, &image_grid_thw)?;
        let img_tokens = image_features.dim(0).map_err(|e| anyhow!(e.to_string()))?;
        log_info(TAG, format!("Vision feature token count: {}", img_tokens));
        log_stage_end(TAG, stage, stage_t);

        let stage = "Build prompt and prefill";
        let stage_t = Instant::now();
        log_stage_start(TAG, stage);
        let prompt = build_prompt();
        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!(e.to_string()))?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let image_pos = input_ids
            .iter()
            .position(|&x| x == IMAGE_TOKEN_ID)
            .ok_or_else(|| anyhow!("image token not found in prompt"))?;

        let text_embeds = model.embed_ids(&input_ids)?;
        let prefix = c(text_embeds.narrow(1, 0, image_pos))?;
        let suffix = c(text_embeds.narrow(1, image_pos + 1, input_ids.len() - image_pos - 1))?;
        let image_b = c(image_features.unsqueeze(0))?;
        let full_embeds = c(Tensor::cat(&[&prefix, &image_b, &suffix], 1))?;

        let merged_seq = full_embeds.dim(1).map_err(|e| anyhow!(e.to_string()))?;
        let (pos_ids_arr, mrope_delta) = build_glm_mrope_positions(
            image_pos,
            input_ids.len(),
            &image_grid_thw,
            img_tokens,
            merged_seq,
        )?;
        let pos_ids = c(Tensor::from_vec(
            pos_ids_arr.iter().copied().collect(),
            (3, 1, merged_seq),
            &model.device,
        ))?;

        let mut cache = (0..model.cfg.text_config.num_hidden_layers)
            .map(|_| LayerCache::default())
            .collect::<Vec<_>>();

        let logits = model.text_forward_prefill(&full_embeds, &pos_ids, &mut cache)?;
        let last = c(logits.i((0, 0, ..)))?;
        let mut generated: Vec<i64> = Vec::new();
        let first_logits = last.to_vec1::<f32>().map_err(|e| anyhow!(e.to_string()))?;
        generated.push(pick_next_token(&first_logits));
        log_stage_end(TAG, stage, stage_t);

        let max_new_tokens = std::env::var("OCR_MAX_NEW_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        
        match max_new_tokens {
            Some(limit) => log_info(TAG, format!("Maximum new decode tokens: {}", limit)),
            None => log_info(TAG, "Maximum new decode tokens: unlimited (will stop on EOS)"),
        }

        let decode_t = Instant::now();
        log_stage_start(TAG, "Autoregressive decode");
        let verbose = is_verbose();
        let stream_decode = std::env::var("OCR_STREAM_DECODE")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "false" || v == "off" || v == "no")
            })
            .unwrap_or(true);

        if stream_decode {
            let first_tok = *generated.first().unwrap_or(&EOS_TOKEN_IDS[0]);
            if !EOS_TOKEN_IDS.contains(&first_tok) {
                let piece = tokenizer
                    .decode(&[first_tok as u32], true)
                    .map_err(|e| anyhow!(e.to_string()))?;
                if !piece.is_empty() {
                    if verbose {
                        log_stream(TAG, 0, &piece);
                    } else {
                        print!("{}", piece);
                        let _ = std::io::stdout().flush();
                    }
                }
            }
        }

        let mut step = 0usize;
        loop {
            let next = *generated.last().unwrap();
            if EOS_TOKEN_IDS.contains(&next) {
                break;
            }
            
            // 如果设置了最大token数限制则检查
            if let Some(limit) = max_new_tokens {
                if step >= limit {
                    log_info(TAG, format!("Reached maximum token limit: {}, stopping generation", limit));
                    break;
                }
            }
            
            let past = cache
                .first()
                .and_then(|x| x.k.as_ref())
                .map(|k| k.dim(2).unwrap_or(0))
                .unwrap_or(0) as i64;
            let pos = past + mrope_delta;
            let logits = model.text_forward_decode_one(next, pos, &mut cache)?;
            let v = c(logits.i((0, 0, ..)))?
                .to_vec1::<f32>()
                .map_err(|e| anyhow!(e.to_string()))?;
            let tok = pick_next_token(&v);
            generated.push(tok);

            if stream_decode {
                let piece = tokenizer
                    .decode(&[tok as u32], true)
                    .map_err(|e| anyhow!(e.to_string()))?;
                if !piece.is_empty() {
                    if verbose {
                        log_stream(TAG, step, &piece);
                    } else {
                        print!("{}", piece);
                        let _ = std::io::stdout().flush();
                    }
                }
            }

            if step < 5 || step % 16 == 0 {
                log_info(TAG, format!(
                    "decode progress: step={}, token={}, generated={}",
                    step,
                    tok,
                    generated.len()
                ));
            }
            if EOS_TOKEN_IDS.contains(&tok) {
                break;
            }
            
            step += 1;
        }
        log_stage_end(TAG, "Autoregressive decode", decode_t);

        let gen_u32: Vec<u32> = generated.iter().map(|&x| x as u32).collect();
        let decoded = tokenizer
            .decode(&gen_u32, true)
            .map_err(|e| anyhow!(e.to_string()))?;

        log_info(TAG, format!("Inference finished, elapsed {:.3}s", total_t.elapsed().as_secs_f64()));
        log_stage_end(TAG, "safetensors OCR inference", total_t);
        Ok(InferResult {
            text: decoded,
            token_count: generated.len(),
        })
    }
}
