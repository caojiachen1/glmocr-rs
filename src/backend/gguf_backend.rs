use std::ffi::{CStr, CString};
use std::io::Write;
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use super::llama_cpp_sys::*;
use super::{InferResult, OcrBackend, is_verbose, log_info, log_stage_end, log_stage_start, log_stream};

const TAG: &str = "GGUF";
const EOS_TOKEN_IDS: [LlamaToken; 4] = [59246, 59253, 59252, 59251];
// GLM-OCR GGUF has 59392 tokens (from tokenizer.ggml.tokens metadata)
const N_VOCAB: usize = 59392;

static GGUF_VERBOSE: AtomicBool = AtomicBool::new(true);

unsafe extern "C" fn llama_log_callback(level: c_int, text: *const c_char, _user_data: *mut c_void) {
    if !GGUF_VERBOSE.load(Ordering::Relaxed) {
        return;
    }
    if text.is_null() {
        return;
    }
    let text = match CStr::from_ptr(text).to_str() {
        Ok(s) => s.trim(),
        Err(_) => return,
    };
    if text.is_empty() {
        return;
    }
    let level_name = match level {
        1 => "DEBUG",
        3 => "WARN",
        4 => "ERROR",
        _ => "INFO",
    };
    eprintln!("[OCR][GGUF][LLAMA][{}] {}", level_name, text);
}

fn install_log_hooks() {
    GGUF_VERBOSE.store(is_verbose(), Ordering::Relaxed);
    unsafe {
        llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
        mtmd_log_set(Some(llama_log_callback), std::ptr::null_mut());
        mtmd_helper_log_set(Some(llama_log_callback), std::ptr::null_mut());
    }
}

pub struct GgufBackend {
    force_cpu: bool,
}

impl GgufBackend {
    pub fn new(force_cpu: bool) -> Self {
        Self { force_cpu }
    }

    fn model_paths(model_root: &Path) -> (PathBuf, PathBuf) {
        let text_model = model_root.join("GLM-OCR-Q8_0.gguf");
        let mmproj = model_root.join("mmproj-GLM-OCR-Q8_0.gguf");
        (text_model, mmproj)
    }
}

struct LlamaState {
    model: *mut LlamaModel,
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    mtmd_ctx: *mut MtmdContext,
}

impl Drop for LlamaState {
    fn drop(&mut self) {
        unsafe {
            if !self.mtmd_ctx.is_null() {
                mtmd_free(self.mtmd_ctx);
            }
            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
            llama_backend_free();
        }
    }
}

fn init_llama(text_model_path: &Path, mmproj_path: &Path, force_cpu: bool) -> Result<LlamaState> {
    unsafe { llama_backend_init(); }
    install_log_hooks();

    let n_threads = std::thread::available_parallelism()
        .map(|nz| nz.get() as c_int)
        .unwrap_or(4);

    // Load text model with full GPU offload
    let model_path_c = CString::new(text_model_path.to_str().unwrap())
        .map_err(|_| anyhow!("Invalid model path"))?;

    let mut mparams = unsafe { llama_model_default_params() };
    mparams.n_gpu_layers = if force_cpu { 0 } else { -1 }; // -1 = offload ALL layers to GPU
    mparams.use_mmap = true;

    log_info(TAG, format!("Loading text model: {}", text_model_path.display()));
    let model = unsafe { llama_model_load_from_file(model_path_c.as_ptr(), mparams) };
    if model.is_null() {
        bail!("Failed to load text model from {}", text_model_path.display());
    }
    log_info(TAG, "Text model loaded successfully");

    // Create context with maximum CUDA optimizations
    let mut cparams = unsafe { llama_context_default_params() };
    cparams.n_ctx = 8192;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.flash_attn_type = if force_cpu { LlamaFlashAttnType::Disabled } else { LlamaFlashAttnType::Enabled };
    cparams.offload_kqv = !force_cpu;
    // KV cache quantization (Q8_0) requires flash attention — disable on CPU
    if !force_cpu {
        cparams.type_k = GgmlType::Q8_0;
        cparams.type_v = GgmlType::Q8_0;
    }
    cparams.no_perf = false;
    cparams.op_offload = !force_cpu;

    log_info(TAG, format!(
        "Context: n_ctx={}, n_batch={}, n_ubatch={}, flash_attn={}, offload_kqv={}, type_k/v={}, threads={}",
        if force_cpu { "default" } else { "Q8_0" },
        cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, !force_cpu, !force_cpu, n_threads
    ));

    let ctx = unsafe { llama_init_from_model(model, cparams) };
    if ctx.is_null() {
        unsafe { llama_model_free(model); }
        bail!("Failed to create llama context");
    }
    log_info(TAG, "Llama context created successfully");

    // Initialize multimodal context
    let mmproj_c = CString::new(mmproj_path.to_str().unwrap())
        .map_err(|_| anyhow!("Invalid mmproj path"))?;

    let mut mtmd_p = unsafe { mtmd_context_params_default() };
    mtmd_p.use_gpu = !force_cpu;
    mtmd_p.n_threads = n_threads;
    mtmd_p.flash_attn_type = if force_cpu { LlamaFlashAttnType::Disabled } else { LlamaFlashAttnType::Enabled };
    mtmd_p.warmup = true;

    log_info(TAG, format!("Loading mmproj: {}", mmproj_path.display()));
    let mtmd_ctx = unsafe { mtmd_init_from_file(mmproj_c.as_ptr(), model, mtmd_p) };
    if mtmd_ctx.is_null() {
        unsafe { llama_free(ctx); llama_model_free(model); }
        bail!("Failed to load mmproj from {}", mmproj_path.display());
    }
    log_info(TAG, "Multimodal context initialized successfully");

    let vocab = unsafe { llama_model_get_vocab(model) };
    if vocab.is_null() {
        unsafe { llama_free(ctx); llama_model_free(model); }
        bail!("Failed to get vocab from model");
    }

    Ok(LlamaState { model, ctx, vocab, mtmd_ctx })
}

fn token_to_string(vocab: *const LlamaVocab, token: LlamaToken) -> Result<String> {
    let mut buf = [0u8; 512];
    let n = unsafe {
        llama_token_to_piece(vocab, token, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int, 0, true)
    };
    if n < 0 {
        bail!("Failed to convert token {} to string (returned {})", token, n);
    }
    if n == 0 {
        return Ok(String::new());
    }
    let n = n as usize;
    if n >= buf.len() {
        bail!("Token {} produced {} bytes, buffer is {}", token, n, buf.len());
    }
    let s = std::str::from_utf8(&buf[..n])
        .map_err(|_| anyhow!("Non-UTF-8 token output for token {}", token))?
        .to_string();
    Ok(s)
}

/// Manual argmax over logits - avoids sampler chain entirely
fn argmax(logits: &[f32]) -> LlamaToken {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as LlamaToken)
        .unwrap_or(EOS_TOKEN_IDS[0])
}

impl OcrBackend for GgufBackend {
    fn name(&self) -> &'static str {
        if self.force_cpu { "gguf (CPU forced)" } else { "gguf" }
    }

    fn infer(
        &mut self,
        model_root: &Path,
        image_path: &Path,
        _min_pixels: usize,
        _max_pixels: usize,
    ) -> Result<InferResult> {
        let total_t = Instant::now();
        log_stage_start(TAG, "GGUF OCR inference");

        let (text_model_path, mmproj_path) = Self::model_paths(model_root);
        for p in [&text_model_path, &mmproj_path] {
            if !p.exists() { bail!("Missing file: {}", p.display()); }
        }
        log_info(TAG, "File check passed: text model + mmproj both exist");

        // Initialize
        let stage = "Initialize llama.cpp";
        let stage_t = Instant::now();
        log_stage_start(TAG, stage);
        let state = init_llama(&text_model_path, &mmproj_path, self.force_cpu)?;
        log_stage_end(TAG, stage, stage_t);

        // Load image
        let stage = "Load and encode image";
        let stage_t = Instant::now();
        log_stage_start(TAG, stage);

        let image_path_c = CString::new(image_path.to_str().unwrap())
            .map_err(|_| anyhow!("Invalid image path"))?;
        let bitmap = unsafe { mtmd_helper_bitmap_init_from_file(state.mtmd_ctx, image_path_c.as_ptr()) };
        if bitmap.is_null() { bail!("Failed to load image: {}", image_path.display()); }
        log_info(TAG, format!("Image loaded: {}", image_path.display()));

        // Build prompt
        let marker = unsafe { CStr::from_ptr(mtmd_default_marker()) };
        let marker_str = marker.to_str().unwrap_or("<__media__>");
        let prompt = format!(
            "[gMASK]<sop><|user|>\n<|begin_of_image|>{}<|end_of_image|>\nText Recognition:\n<|assistant|>\n",
            marker_str
        );
        log_info(TAG, format!("Prompt: {}", prompt));

        let prompt_c = CString::new(prompt.as_str()).map_err(|_| anyhow!("Invalid prompt"))?;
        let input_text = MtmdInputText {
            text: prompt_c.as_ptr(),
            add_special: false,
            parse_special: true,
        };

        // Tokenize
        let bitmaps_arr: [*const MtmdBitmap; 1] = [bitmap as *const MtmdBitmap];
        let chunks = unsafe { mtmd_input_chunks_init() };
        if chunks.is_null() {
            unsafe { mtmd_bitmap_free(bitmap); }
            bail!("Failed to create input chunks");
        }

        let res = unsafe { mtmd_tokenize(state.mtmd_ctx, chunks, &input_text, bitmaps_arr.as_ptr(), 1) };
        unsafe { mtmd_bitmap_free(bitmap); }
        if res != 0 {
            unsafe { mtmd_input_chunks_free(chunks); }
            bail!("mtmd_tokenize failed with code {}", res);
        }

        let n_chunks = unsafe { mtmd_input_chunks_size(chunks) };
        let total_tokens = unsafe { mtmd_helper_get_n_tokens(chunks) };
        let total_pos = unsafe { mtmd_helper_get_n_pos(chunks) };
        log_info(TAG, format!("Tokenized: {} chunks, {} tokens, {} positions", n_chunks, total_tokens, total_pos));

        // Eval all chunks (encode vision + decode text) via helper
        let n_batch = 512i32;
        let mut new_n_past: LlamaPos = 0;
        let eval_res = unsafe {
            mtmd_helper_eval_chunks(
                state.mtmd_ctx, state.ctx, chunks,
                0, 0, n_batch, true, &mut new_n_past,
            )
        };
        unsafe { mtmd_input_chunks_free(chunks); }

        if eval_res != 0 {
            bail!("mtmd_helper_eval_chunks failed with code {}", eval_res);
        }
        log_info(TAG, format!("Eval complete, n_past={}", new_n_past));
        log_stage_end(TAG, stage, stage_t);

        // ── Get logits for the first generated token ──
        let logits_ptr = unsafe { llama_get_logits(state.ctx) };
        if logits_ptr.is_null() {
            bail!("llama_get_logits returned null");
        }
        let logits_slice = unsafe { std::slice::from_raw_parts(logits_ptr, N_VOCAB) };
        let first_token = argmax(logits_slice);

        // ── Autoregressive decode loop ──
        let stage = "Autoregressive decode";
        let stage_t = Instant::now();
        log_stage_start(TAG, stage);

        let verbose = is_verbose();
        let stream_decode = std::env::var("OCR_STREAM_DECODE")
            .ok()
            .map(|v| { let v = v.trim().to_ascii_lowercase(); !(v == "0" || v == "false" || v == "off" || v == "no") })
            .unwrap_or(true);

        let max_new_tokens: usize = std::env::var("OCR_MAX_NEW_TOKENS")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(2048);

        let mut generated_tokens: Vec<LlamaToken> = vec![first_token];
        let mut n_past = new_n_past;

        // Stream first token
        if stream_decode {
            let piece = token_to_string(state.vocab, first_token)?;
            if !piece.is_empty() {
                if verbose { log_stream(TAG, 0, &piece); }
                else { print!("{}", piece); let _ = std::io::stdout().flush(); }
            }
        }

        let mut step = 0usize;
        loop {
            if step >= max_new_tokens {
                log_info(TAG, format!("Reached max token limit: {}", max_new_tokens));
                break;
            }

            let current_token = *generated_tokens.last().unwrap();
            if EOS_TOKEN_IDS.contains(&current_token) {
                log_info(TAG, format!("Hit EOS token {} at step {}", current_token, step));
                break;
            }

            // Build batch for single token
            let mut batch = unsafe { llama_batch_init(1, 0, 1) };
            batch.n_tokens = 1;
            unsafe {
                *batch.token = current_token;
                *batch.pos = n_past;
                *batch.n_seq_id = 1;
                *(*batch.seq_id).add(0) = 0;
                *batch.logits = 1;
            }

            let dec_res = unsafe { llama_decode(state.ctx, batch) };
            unsafe { llama_batch_free(batch); }
            if dec_res != 0 {
                bail!("llama_decode failed at step {} with code {}", step, dec_res);
            }

            // Get logits and pick next token
            let logits_ptr = unsafe { llama_get_logits(state.ctx) };
            if logits_ptr.is_null() {
                bail!("llama_get_logits returned null at step {}", step);
            }
            let logits_slice = unsafe { std::slice::from_raw_parts(logits_ptr, N_VOCAB) };
            let next_token = argmax(logits_slice);

            generated_tokens.push(next_token);
            n_past += 1;

            if stream_decode {
                let piece = token_to_string(state.vocab, next_token)?;
                if !piece.is_empty() {
                    if verbose { log_stream(TAG, step + 1, &piece); }
                    else { print!("{}", piece); let _ = std::io::stdout().flush(); }
                }
            }

            if step < 5 || step % 16 == 0 {
                log_info(TAG, format!(
                    "decode: step={}, token={}, generated={}, n_past={}",
                    step, next_token, generated_tokens.len(), n_past
                ));
            }

            step += 1;
        }

        log_stage_end(TAG, stage, stage_t);
        log_stage_end(TAG, "GGUF OCR inference", total_t);

        // Decode all generated tokens to text
        let mut output = String::new();
        for &tok in &generated_tokens {
            if EOS_TOKEN_IDS.contains(&tok) { break; }
            let piece = token_to_string(state.vocab, tok)?;
            output.push_str(&piece);
        }

        log_info(TAG, format!(
            "Inference finished, elapsed {:.3}s, {} tokens",
            total_t.elapsed().as_secs_f64(),
            generated_tokens.len()
        ));

        Ok(InferResult {
            text: output,
            token_count: generated_tokens.len(),
        })
    }
}
