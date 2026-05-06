use std::ffi::{CStr, CString};
use std::io::Write;
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use super::llama_cpp_loader::LlamaCppLib;
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

fn install_log_hooks(lib: &LlamaCppLib) {
    GGUF_VERBOSE.store(is_verbose(), Ordering::Relaxed);
    unsafe {
        (lib.llama_log_set)(Some(llama_log_callback), std::ptr::null_mut());
        (lib.mtmd_log_set)(Some(llama_log_callback), std::ptr::null_mut());
        (lib.mtmd_helper_log_set)(Some(llama_log_callback), std::ptr::null_mut());
    }
}

pub struct GgufBackend {
    lib: Arc<LlamaCppLib>,
    force_cpu: bool,
    // Persistent model state — loaded lazily on first infer(), stays alive across calls.
    loaded: bool,
    model: *mut LlamaModel,
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    mtmd_ctx: *mut MtmdContext,
    n_threads: c_int,
    loaded_model_root: Option<PathBuf>,
}

impl GgufBackend {
    pub fn new(lib: Arc<LlamaCppLib>, force_cpu: bool) -> Self {
        Self {
            lib,
            force_cpu,
            loaded: false,
            model: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
            vocab: std::ptr::null(),
            mtmd_ctx: std::ptr::null_mut(),
            n_threads: 4,
            loaded_model_root: None,
        }
    }

    fn model_paths(model_root: &Path) -> (PathBuf, PathBuf) {
        let text_model = model_root.join("GLM-OCR-Q8_0.gguf");
        let mmproj = model_root.join("mmproj-GLM-OCR-Q8_0.gguf");
        (text_model, mmproj)
    }

    /// Returns true if model is loaded and ready for inference.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Explicitly unload the model, freeing GPU/CPU memory.
    /// The backend remains usable — next `infer()` will reload automatically.
    pub fn unload(&mut self) {
        if !self.loaded {
            return;
        }
        let lib = &self.lib;
        unsafe {
            if !self.mtmd_ctx.is_null() {
                (lib.mtmd_free)(self.mtmd_ctx);
            }
            if !self.ctx.is_null() {
                (lib.llama_free)(self.ctx);
            }
            if !self.model.is_null() {
                (lib.llama_model_free)(self.model);
            }
            (lib.llama_backend_free)();
        }
        self.loaded = false;
        self.model = std::ptr::null_mut();
        self.ctx = std::ptr::null_mut();
        self.vocab = std::ptr::null();
        self.mtmd_ctx = std::ptr::null_mut();
        self.loaded_model_root = None;
    }

    fn ensure_loaded(&mut self, model_root: &Path) -> Result<()> {
        if self.loaded {
            // If model root changed, reload
            if self.loaded_model_root.as_deref() != Some(model_root) {
                log_info(TAG, "Model root changed, reloading...");
                self.unload();
            } else {
                return Ok(());
            }
        }

        let lib = &self.lib;
        let (text_model_path, mmproj_path) = Self::model_paths(model_root);
        for p in [&text_model_path, &mmproj_path] {
            if !p.exists() {
                bail!("Missing file: {}", p.display());
            }
        }

        let stage = "Load model (persistent)";
        let stage_t = Instant::now();
        log_stage_start(TAG, stage);

        unsafe { (lib.llama_backend_init)(); }
        install_log_hooks(lib);

        let n_threads = std::thread::available_parallelism()
            .map(|nz| nz.get() as c_int)
            .unwrap_or(4);

        // Load text model
        let model_path_c = CString::new(text_model_path.to_str().unwrap())
            .map_err(|_| anyhow!("Invalid model path"))?;

        let mut mparams = unsafe { (lib.llama_model_default_params)() };
        mparams.n_gpu_layers = if self.force_cpu { 0 } else { -1 };
        mparams.use_mmap = true;

        log_info(TAG, format!("Loading text model: {}", text_model_path.display()));
        let model = unsafe { (lib.llama_model_load_from_file)(model_path_c.as_ptr(), mparams) };
        if model.is_null() {
            bail!("Failed to load text model from {}", text_model_path.display());
        }
        log_info(TAG, "Text model loaded successfully");

        // Create context
        let mut cparams = unsafe { (lib.llama_context_default_params)() };
        cparams.n_ctx = 8192;
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;
        cparams.n_threads = n_threads;
        cparams.n_threads_batch = n_threads;
        cparams.flash_attn_type = if self.force_cpu { LlamaFlashAttnType::Disabled } else { LlamaFlashAttnType::Enabled };
        cparams.offload_kqv = !self.force_cpu;
        if !self.force_cpu {
            cparams.type_k = GgmlType::Q8_0;
            cparams.type_v = GgmlType::Q8_0;
        }
        cparams.no_perf = false;
        cparams.op_offload = !self.force_cpu;

        let ctx = unsafe { (lib.llama_init_from_model)(model, cparams) };
        if ctx.is_null() {
            unsafe { (lib.llama_model_free)(model); }
            bail!("Failed to create llama context");
        }
        log_info(TAG, "Llama context created successfully");

        // Initialize multimodal context
        let mmproj_c = CString::new(mmproj_path.to_str().unwrap())
            .map_err(|_| anyhow!("Invalid mmproj path"))?;

        let mut mtmd_p = unsafe { (lib.mtmd_context_params_default)() };
        mtmd_p.use_gpu = !self.force_cpu;
        mtmd_p.n_threads = n_threads;
        mtmd_p.flash_attn_type = if self.force_cpu { LlamaFlashAttnType::Disabled } else { LlamaFlashAttnType::Enabled };
        mtmd_p.warmup = true;

        log_info(TAG, format!("Loading mmproj: {}", mmproj_path.display()));
        let mtmd_ctx = unsafe { (lib.mtmd_init_from_file)(mmproj_c.as_ptr(), model, mtmd_p) };
        if mtmd_ctx.is_null() {
            unsafe { (lib.llama_free)(ctx); (lib.llama_model_free)(model); }
            bail!("Failed to load mmproj from {}", mmproj_path.display());
        }
        log_info(TAG, "Multimodal context initialized successfully");

        let vocab = unsafe { (lib.llama_model_get_vocab)(model) };
        if vocab.is_null() {
            unsafe { (lib.llama_free)(ctx); (lib.llama_model_free)(model); }
            bail!("Failed to get vocab from model");
        }

        self.model = model;
        self.ctx = ctx;
        self.vocab = vocab;
        self.mtmd_ctx = mtmd_ctx;
        self.n_threads = n_threads;
        self.loaded = true;
        self.loaded_model_root = Some(model_root.to_path_buf());

        log_stage_end(TAG, stage, stage_t);
        log_info(TAG, "Model loaded — will persist across infer() calls");
        Ok(())
    }
}

impl Drop for GgufBackend {
    fn drop(&mut self) {
        self.unload();
    }
}

// ── Standalone helpers ─────────────────────────────────────────────────────────

fn token_to_string(lib: &LlamaCppLib, vocab: *const LlamaVocab, token: LlamaToken) -> Result<String> {
    let mut buf = [0u8; 512];
    let n = unsafe {
        (lib.llama_token_to_piece)(vocab, token, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int, 0, true)
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

// ── OcrBackend impl ────────────────────────────────────────────────────────────

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

        // ── Ensure model is loaded ──
        self.ensure_loaded(model_root)?;
        let lib = &self.lib;
        let ctx = self.ctx;
        let vocab = self.vocab;
        let mtmd_ctx = self.mtmd_ctx;

        // Clear KV cache for fresh inference (seq_id=-1, p0=-1, p1=-1 removes all)
        let mem = unsafe { (lib.llama_get_memory)(ctx) };
        unsafe { (lib.llama_memory_seq_rm)(mem, -1, -1, -1); }

        // ── Load and encode image ──
        let stage = "Load and encode image";
        let stage_t = Instant::now();
        log_stage_start(TAG, stage);

        let image_path_c = CString::new(image_path.to_str().unwrap())
            .map_err(|_| anyhow!("Invalid image path"))?;
        let bitmap = unsafe { (lib.mtmd_helper_bitmap_init_from_file)(mtmd_ctx, image_path_c.as_ptr()) };
        if bitmap.is_null() { bail!("Failed to load image: {}", image_path.display()); }
        log_info(TAG, format!("Image loaded: {}", image_path.display()));

        // Build prompt
        let marker = unsafe { CStr::from_ptr((lib.mtmd_default_marker)()) };
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
        let chunks = unsafe { (lib.mtmd_input_chunks_init)() };
        if chunks.is_null() {
            unsafe { (lib.mtmd_bitmap_free)(bitmap); }
            bail!("Failed to create input chunks");
        }

        let res = unsafe { (lib.mtmd_tokenize)(mtmd_ctx, chunks, &input_text, bitmaps_arr.as_ptr(), 1) };
        unsafe { (lib.mtmd_bitmap_free)(bitmap); }
        if res != 0 {
            unsafe { (lib.mtmd_input_chunks_free)(chunks); }
            bail!("mtmd_tokenize failed with code {}", res);
        }

        let n_chunks = unsafe { (lib.mtmd_input_chunks_size)(chunks) };
        let total_tokens = unsafe { (lib.mtmd_helper_get_n_tokens)(chunks) };
        let total_pos = unsafe { (lib.mtmd_helper_get_n_pos)(chunks) };
        log_info(TAG, format!("Tokenized: {} chunks, {} tokens, {} positions", n_chunks, total_tokens, total_pos));

        // Eval all chunks (encode vision + decode text) via helper
        let n_batch = 512i32;
        let mut new_n_past: LlamaPos = 0;
        let eval_res = unsafe {
            (lib.mtmd_helper_eval_chunks)(
                mtmd_ctx, ctx, chunks,
                0, 0, n_batch, true, &mut new_n_past,
            )
        };
        unsafe { (lib.mtmd_input_chunks_free)(chunks); }

        if eval_res != 0 {
            bail!("mtmd_helper_eval_chunks failed with code {}", eval_res);
        }
        log_info(TAG, format!("Eval complete, n_past={}", new_n_past));
        log_stage_end(TAG, stage, stage_t);

        // ── Get logits for the first generated token ──
        let logits_ptr = unsafe { (lib.llama_get_logits)(ctx) };
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
            let piece = token_to_string(lib, vocab, first_token)?;
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
            let mut batch = unsafe { (lib.llama_batch_init)(1, 0, 1) };
            batch.n_tokens = 1;
            unsafe {
                *batch.token = current_token;
                *batch.pos = n_past;
                *batch.n_seq_id = 1;
                *(*batch.seq_id).add(0) = 0;
                *batch.logits = 1;
            }

            let dec_res = unsafe { (lib.llama_decode)(ctx, batch) };
            unsafe { (lib.llama_batch_free)(batch); }
            if dec_res != 0 {
                bail!("llama_decode failed at step {} with code {}", step, dec_res);
            }

            // Get logits and pick next token
            let logits_ptr = unsafe { (lib.llama_get_logits)(ctx) };
            if logits_ptr.is_null() {
                bail!("llama_get_logits returned null at step {}", step);
            }
            let logits_slice = unsafe { std::slice::from_raw_parts(logits_ptr, N_VOCAB) };
            let next_token = argmax(logits_slice);

            generated_tokens.push(next_token);
            n_past += 1;

            if stream_decode {
                let piece = token_to_string(lib, vocab, next_token)?;
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
            let piece = token_to_string(lib, vocab, tok)?;
            output.push_str(&piece);
        }

        log_info(TAG, format!(
            "Inference finished, elapsed {:.3}s, {} tokens",
            total_t.elapsed().as_secs_f64(),
            generated_tokens.len()
        ));

        // NOTE: model/context stay alive for next call

        Ok(InferResult {
            text: output,
            token_count: generated_tokens.len(),
        })
    }
}
