use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::path::PathBuf;

use anyhow::{Context, Result};
use libloading::Library;

use super::llama_cpp_sys::*;

// ── Configuration ───────────────────────────────────────────────────────────────

/// Specifies which DLL paths to load for the GGUF backend.
#[derive(Debug, Clone)]
pub struct GgufLibConfig {
    /// Path to `llama.dll` (or `libllama.so` / `libllama.dylib` on other platforms).
    /// Falls back to `GGUF_LLAMA_DLL` env var, then `llama.dll`.
    pub llama_dll: Option<PathBuf>,
    /// Path to `mtmd.dll` (or `libmtmd.so` / `libmtmd.dylib` on other platforms).
    /// Falls back to `GGUF_MTMD_DLL` env var, then `mtmd.dll`.
    pub mtmd_dll: Option<PathBuf>,
}

impl Default for GgufLibConfig {
    fn default() -> Self {
        Self {
            llama_dll: None,
            mtmd_dll: None,
        }
    }
}

impl GgufLibConfig {
    /// Resolve the effective llama DLL path: config → env → default.
    pub fn resolve_llama_dll(&self) -> PathBuf {
        if let Some(ref p) = self.llama_dll {
            return p.clone();
        }
        if let Ok(p) = std::env::var("GGUF_LLAMA_DLL") {
            return PathBuf::from(p);
        }
        PathBuf::from("llama.dll")
    }

    /// Resolve the effective mtmd DLL path: config → env → default.
    pub fn resolve_mtmd_dll(&self) -> PathBuf {
        if let Some(ref p) = self.mtmd_dll {
            return p.clone();
        }
        if let Ok(p) = std::env::var("GGUF_MTMD_DLL") {
            return PathBuf::from(p);
        }
        PathBuf::from("mtmd.dll")
    }
}

// ── Loaded library ──────────────────────────────────────────────────────────────

/// Holds dynamically-loaded llama.cpp function pointers.
///
/// Load once with [`LlamaCppLib::load`], then share via `Arc<LlamaCppLib>`.
pub struct LlamaCppLib {
    // Library handles kept alive — function pointers are valid as long as these live.
    _llama_lib: Library,
    _mtmd_lib: Library,

    // ── Backend ──
    pub llama_backend_init: unsafe extern "C" fn(),
    pub llama_backend_free: unsafe extern "C" fn(),

    // ── Logging ──
    pub llama_log_set: unsafe extern "C" fn(GgmlLogCallback, *mut c_void),
    pub mtmd_log_set: unsafe extern "C" fn(GgmlLogCallback, *mut c_void),
    pub mtmd_helper_log_set: unsafe extern "C" fn(GgmlLogCallback, *mut c_void),

    // ── Default params ──
    pub llama_model_default_params: unsafe extern "C" fn() -> LlamaModelParams,
    pub llama_context_default_params: unsafe extern "C" fn() -> LlamaContextParams,
    pub llama_sampler_chain_default_params: unsafe extern "C" fn() -> LlamaSamplerChainParams,

    // ── Model ──
    pub llama_model_load_from_file:
        unsafe extern "C" fn(*const c_char, LlamaModelParams) -> *mut LlamaModel,
    pub llama_model_free: unsafe extern "C" fn(*mut LlamaModel),
    pub llama_model_n_embd: unsafe extern "C" fn(*const LlamaModel) -> c_int,
    pub llama_model_n_embd_inp: unsafe extern "C" fn(*const LlamaModel) -> c_int,
    pub llama_model_n_layer: unsafe extern "C" fn(*const LlamaModel) -> c_int,
    pub llama_model_rope_type: unsafe extern "C" fn(*const LlamaModel) -> c_int,

    // ── Context ──
    pub llama_init_from_model:
        unsafe extern "C" fn(*mut LlamaModel, LlamaContextParams) -> *mut LlamaContext,
    pub llama_free: unsafe extern "C" fn(*mut LlamaContext),
    pub llama_n_ctx: unsafe extern "C" fn(*const LlamaContext) -> c_uint,
    pub llama_get_model: unsafe extern "C" fn(*const LlamaContext) -> *const LlamaModel,
    pub llama_get_memory: unsafe extern "C" fn(*const LlamaContext) -> LlamaMemoryT,
    pub llama_model_get_vocab: unsafe extern "C" fn(*const LlamaModel) -> *const LlamaVocab,

    // ── Threading ──
    pub llama_set_n_threads: unsafe extern "C" fn(*mut LlamaContext, c_int, c_int),

    // ── Attention ──
    pub llama_set_causal_attn: unsafe extern "C" fn(*mut LlamaContext, bool),
    pub llama_set_embeddings: unsafe extern "C" fn(*mut LlamaContext, bool),

    // ── Batch ──
    pub llama_batch_init: unsafe extern "C" fn(c_int, c_int, c_int) -> LlamaBatch,
    pub llama_batch_free: unsafe extern "C" fn(LlamaBatch),
    pub llama_batch_get_one: unsafe extern "C" fn(*mut LlamaToken, c_int) -> LlamaBatch,

    // ── Decode ──
    pub llama_encode: unsafe extern "C" fn(*mut LlamaContext, LlamaBatch) -> c_int,
    pub llama_decode: unsafe extern "C" fn(*mut LlamaContext, LlamaBatch) -> c_int,

    // ── Logits ──
    pub llama_get_logits: unsafe extern "C" fn(*mut LlamaContext) -> *mut c_float,
    pub llama_get_logits_ith: unsafe extern "C" fn(*mut LlamaContext, c_int) -> *mut c_float,

    // ── Vocab ──
    pub llama_vocab_n_tokens: unsafe extern "C" fn(*const LlamaVocab) -> c_int,

    // ── KV cache ──
    pub llama_kv_cache_clear: unsafe extern "C" fn(*mut LlamaContext),

    // ── Token ──
    pub llama_tokenize: unsafe extern "C" fn(
        *const LlamaVocab, *const c_char, c_int,
        *mut LlamaToken, c_int, bool, bool,
    ) -> c_int,
    pub llama_token_to_piece: unsafe extern "C" fn(
        *const LlamaVocab, LlamaToken,
        *mut c_char, c_int, c_int, bool,
    ) -> c_int,
    pub llama_vocab_is_eog: unsafe extern "C" fn(*const LlamaVocab, LlamaToken) -> bool,

    // ── Sampler ──
    pub llama_sampler_chain_init:
        unsafe extern "C" fn(LlamaSamplerChainParams) -> *mut LlamaSampler,
    pub llama_sampler_chain_add:
        unsafe extern "C" fn(*mut LlamaSampler, *mut LlamaSampler),
    pub llama_sampler_init_greedy: unsafe extern "C" fn() -> *mut LlamaSampler,
    pub llama_sampler_sample:
        unsafe extern "C" fn(*mut LlamaSampler, *mut LlamaContext, c_int) -> LlamaToken,
    pub llama_sampler_accept: unsafe extern "C" fn(*mut LlamaSampler, LlamaToken),
    pub llama_sampler_free: unsafe extern "C" fn(*mut LlamaSampler),

    // ── Memory ──
    pub llama_memory_clear: unsafe extern "C" fn(LlamaMemoryT, bool),

    // ── mtmd ──
    pub mtmd_context_params_default: unsafe extern "C" fn() -> MtmdContextParams,
    pub mtmd_init_from_file: unsafe extern "C" fn(
        *const c_char, *const LlamaModel, MtmdContextParams,
    ) -> *mut MtmdContext,
    pub mtmd_free: unsafe extern "C" fn(*mut MtmdContext),
    pub mtmd_default_marker: unsafe extern "C" fn() -> *const c_char,
    pub mtmd_bitmap_init: unsafe extern "C" fn(c_uint, c_uint, *const u8) -> *mut MtmdBitmap,
    pub mtmd_bitmap_free: unsafe extern "C" fn(*mut MtmdBitmap),
    pub mtmd_input_chunks_init: unsafe extern "C" fn() -> *mut MtmdInputChunks,
    pub mtmd_input_chunks_size: unsafe extern "C" fn(*const MtmdInputChunks) -> usize,
    pub mtmd_input_chunks_get: unsafe extern "C" fn(
        *const MtmdInputChunks, usize,
    ) -> *const MtmdInputChunk,
    pub mtmd_input_chunks_free: unsafe extern "C" fn(*mut MtmdInputChunks),
    pub mtmd_input_chunk_get_type:
        unsafe extern "C" fn(*const MtmdInputChunk) -> MtmdInputChunkType,
    pub mtmd_input_chunk_get_tokens_text:
        unsafe extern "C" fn(*const MtmdInputChunk, *mut usize) -> *const LlamaToken,
    pub mtmd_input_chunk_get_n_tokens:
        unsafe extern "C" fn(*const MtmdInputChunk) -> usize,
    pub mtmd_input_chunk_get_n_pos:
        unsafe extern "C" fn(*const MtmdInputChunk) -> LlamaPos,
    pub mtmd_tokenize: unsafe extern "C" fn(
        *mut MtmdContext, *mut MtmdInputChunks, *const MtmdInputText,
        *const *const MtmdBitmap, usize,
    ) -> c_int,
    pub mtmd_encode_chunk:
        unsafe extern "C" fn(*mut MtmdContext, *const MtmdInputChunk) -> c_int,
    pub mtmd_get_output_embd: unsafe extern "C" fn(*mut MtmdContext) -> *mut c_float,
    pub mtmd_decode_use_non_causal:
        unsafe extern "C" fn(*mut MtmdContext, *const MtmdInputChunk) -> bool,
    pub mtmd_decode_use_mrope: unsafe extern "C" fn(*mut MtmdContext) -> bool,

    // ── mtmd helper ──
    pub mtmd_helper_bitmap_init_from_file:
        unsafe extern "C" fn(*mut MtmdContext, *const c_char) -> *mut MtmdBitmap,
    pub mtmd_helper_eval_chunks: unsafe extern "C" fn(
        *mut MtmdContext, *mut LlamaContext, *const MtmdInputChunks,
        LlamaPos, LlamaSeqId, c_int, bool, *mut LlamaPos,
    ) -> c_int,
    pub mtmd_helper_get_n_tokens:
        unsafe extern "C" fn(*const MtmdInputChunks) -> usize,
    pub mtmd_helper_get_n_pos:
        unsafe extern "C" fn(*const MtmdInputChunks) -> LlamaPos,
}

// ── Loading ─────────────────────────────────────────────────────────────────────

/// Load a symbol via libloading and transmute to the struct field's function pointer type.
///
/// # Safety
///
/// The caller must ensure the target field's type matches the actual function signature.
/// On all platforms supported by libloading, function pointers are address-sized, making
/// the transmute sound as long as the signature is correct at the call site.
macro_rules! load_fn {
    ($lib:expr, $name:literal) => {{
        let sym: libloading::Symbol<unsafe extern "C" fn()> = unsafe { $lib.get($name.as_bytes()) }
            .with_context(|| format!("symbol '{}' not found", $name))?;
        let ptr: *const () = *sym as *const ();
        // Transmute to the target function pointer type determined by struct field
        unsafe { std::mem::transmute_copy::<*const (), _>(&ptr) }
    }};
}

impl LlamaCppLib {
    /// Load llama.cpp DLLs according to `config`.
    ///
    /// DLL path resolution order (first wins):
    /// 1. `config.llama_dll` / `config.mtmd_dll`
    /// 2. `GGUF_LLAMA_DLL` / `GGUF_MTMD_DLL` env vars
    /// 3. Default `llama.dll` / `mtmd.dll` (OS loader search path)
    pub fn load(config: &GgufLibConfig) -> Result<Self> {
        let llama_path = config.resolve_llama_dll();
        let mtmd_path = config.resolve_mtmd_dll();

        let llama_lib = unsafe { Library::new(&llama_path) }
            .with_context(|| format!("loading {}", llama_path.display()))?;

        let mtmd_lib = unsafe { Library::new(&mtmd_path) }
            .with_context(|| format!("loading {}", mtmd_path.display()))?;

        Ok(Self {
            llama_backend_init: load_fn!(llama_lib, "llama_backend_init"),
            llama_backend_free: load_fn!(llama_lib, "llama_backend_free"),

            llama_log_set: load_fn!(llama_lib, "llama_log_set"),
            mtmd_log_set: load_fn!(mtmd_lib, "mtmd_log_set"),
            mtmd_helper_log_set: load_fn!(mtmd_lib, "mtmd_helper_log_set"),

            llama_model_default_params: load_fn!(llama_lib, "llama_model_default_params"),
            llama_context_default_params: load_fn!(llama_lib, "llama_context_default_params"),
            llama_sampler_chain_default_params: load_fn!(
                llama_lib, "llama_sampler_chain_default_params"
            ),

            llama_model_load_from_file: load_fn!(llama_lib, "llama_model_load_from_file"),
            llama_model_free: load_fn!(llama_lib, "llama_model_free"),
            llama_model_n_embd: load_fn!(llama_lib, "llama_model_n_embd"),
            llama_model_n_embd_inp: load_fn!(llama_lib, "llama_model_n_embd_inp"),
            llama_model_n_layer: load_fn!(llama_lib, "llama_model_n_layer"),
            llama_model_rope_type: load_fn!(llama_lib, "llama_model_rope_type"),

            llama_init_from_model: load_fn!(llama_lib, "llama_init_from_model"),
            llama_free: load_fn!(llama_lib, "llama_free"),
            llama_n_ctx: load_fn!(llama_lib, "llama_n_ctx"),
            llama_get_model: load_fn!(llama_lib, "llama_get_model"),
            llama_get_memory: load_fn!(llama_lib, "llama_get_memory"),
            llama_model_get_vocab: load_fn!(llama_lib, "llama_model_get_vocab"),

            llama_set_n_threads: load_fn!(llama_lib, "llama_set_n_threads"),

            llama_set_causal_attn: load_fn!(llama_lib, "llama_set_causal_attn"),
            llama_set_embeddings: load_fn!(llama_lib, "llama_set_embeddings"),

            llama_batch_init: load_fn!(llama_lib, "llama_batch_init"),
            llama_batch_free: load_fn!(llama_lib, "llama_batch_free"),
            llama_batch_get_one: load_fn!(llama_lib, "llama_batch_get_one"),

            llama_encode: load_fn!(llama_lib, "llama_encode"),
            llama_decode: load_fn!(llama_lib, "llama_decode"),

            llama_get_logits: load_fn!(llama_lib, "llama_get_logits"),
            llama_get_logits_ith: load_fn!(llama_lib, "llama_get_logits_ith"),

            llama_vocab_n_tokens: load_fn!(llama_lib, "llama_vocab_n_tokens"),

            llama_kv_cache_clear: load_fn!(llama_lib, "llama_kv_cache_clear"),

            llama_tokenize: load_fn!(llama_lib, "llama_tokenize"),
            llama_token_to_piece: load_fn!(llama_lib, "llama_token_to_piece"),
            llama_vocab_is_eog: load_fn!(llama_lib, "llama_vocab_is_eog"),

            llama_sampler_chain_init: load_fn!(llama_lib, "llama_sampler_chain_init"),
            llama_sampler_chain_add: load_fn!(llama_lib, "llama_sampler_chain_add"),
            llama_sampler_init_greedy: load_fn!(llama_lib, "llama_sampler_init_greedy"),
            llama_sampler_sample: load_fn!(llama_lib, "llama_sampler_sample"),
            llama_sampler_accept: load_fn!(llama_lib, "llama_sampler_accept"),
            llama_sampler_free: load_fn!(llama_lib, "llama_sampler_free"),

            llama_memory_clear: load_fn!(llama_lib, "llama_memory_clear"),

            mtmd_context_params_default: load_fn!(mtmd_lib, "mtmd_context_params_default"),
            mtmd_init_from_file: load_fn!(mtmd_lib, "mtmd_init_from_file"),
            mtmd_free: load_fn!(mtmd_lib, "mtmd_free"),
            mtmd_default_marker: load_fn!(mtmd_lib, "mtmd_default_marker"),
            mtmd_bitmap_init: load_fn!(mtmd_lib, "mtmd_bitmap_init"),
            mtmd_bitmap_free: load_fn!(mtmd_lib, "mtmd_bitmap_free"),
            mtmd_input_chunks_init: load_fn!(mtmd_lib, "mtmd_input_chunks_init"),
            mtmd_input_chunks_size: load_fn!(mtmd_lib, "mtmd_input_chunks_size"),
            mtmd_input_chunks_get: load_fn!(mtmd_lib, "mtmd_input_chunks_get"),
            mtmd_input_chunks_free: load_fn!(mtmd_lib, "mtmd_input_chunks_free"),
            mtmd_input_chunk_get_type: load_fn!(mtmd_lib, "mtmd_input_chunk_get_type"),
            mtmd_input_chunk_get_tokens_text: load_fn!(
                mtmd_lib, "mtmd_input_chunk_get_tokens_text"
            ),
            mtmd_input_chunk_get_n_tokens: load_fn!(
                mtmd_lib, "mtmd_input_chunk_get_n_tokens"
            ),
            mtmd_input_chunk_get_n_pos: load_fn!(mtmd_lib, "mtmd_input_chunk_get_n_pos"),
            mtmd_tokenize: load_fn!(mtmd_lib, "mtmd_tokenize"),
            mtmd_encode_chunk: load_fn!(mtmd_lib, "mtmd_encode_chunk"),
            mtmd_get_output_embd: load_fn!(mtmd_lib, "mtmd_get_output_embd"),
            mtmd_decode_use_non_causal: load_fn!(mtmd_lib, "mtmd_decode_use_non_causal"),
            mtmd_decode_use_mrope: load_fn!(mtmd_lib, "mtmd_decode_use_mrope"),

            mtmd_helper_bitmap_init_from_file: load_fn!(
                mtmd_lib, "mtmd_helper_bitmap_init_from_file"
            ),
            mtmd_helper_eval_chunks: load_fn!(mtmd_lib, "mtmd_helper_eval_chunks"),
            mtmd_helper_get_n_tokens: load_fn!(mtmd_lib, "mtmd_helper_get_n_tokens"),
            mtmd_helper_get_n_pos: load_fn!(mtmd_lib, "mtmd_helper_get_n_pos"),

            _llama_lib: llama_lib,
            _mtmd_lib: mtmd_lib,
        })
    }
}
