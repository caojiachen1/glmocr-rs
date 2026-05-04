use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};

// Opaque types
pub enum LlamaModel {}
pub enum LlamaContext {}
pub enum LlamaVocab {}
pub enum LlamaSampler {}
pub enum LlamaMemoryT {}
pub enum MtmdContext {}
pub enum MtmdBitmap {}
pub enum MtmdInputChunk {}
pub enum MtmdInputChunks {}

// Basic types
pub type LlamaToken = i32;
pub type LlamaPos = i32;
pub type LlamaSeqId = i32;

// ggml_type enum (subset we need)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    BF16 = 30,
}

// llama_rope_type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaRopeType {
    None = -1,
    Norm = 0,
    Mrope = 6,
}

// llama_split_mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaSplitMode {
    None = 0,
    Layer = 1,
    Row = 2,
    Tensor = 3,
}

// llama_flash_attn_type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaFlashAttnType {
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

// llama_pooling_type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaPoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
}

// llama_attention_type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaAttentionType {
    Unspecified = -1,
    Causal = 0,
    NonCausal = 1,
}

// llama_batch
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlamaBatch {
    pub n_tokens: c_int,
    pub token: *mut LlamaToken,
    pub embd: *mut c_float,
    pub pos: *mut LlamaPos,
    pub n_seq_id: *mut c_int,
    pub seq_id: *mut *mut LlamaSeqId,
    pub logits: *mut i8,
}

// llama_model_params
#[repr(C)]
pub struct LlamaModelParams {
    pub devices: *mut *mut c_void,        // ggml_backend_dev_t
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: c_int,
    pub split_mode: LlamaSplitMode,
    pub main_gpu: c_int,
    pub tensor_split: *const c_float,
    pub progress_callback: *const c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

// llama_context_params
#[repr(C)]
pub struct LlamaContextParams {
    pub n_ctx: c_uint,
    pub n_batch: c_uint,
    pub n_ubatch: c_uint,
    pub n_seq_max: c_uint,
    pub n_threads: c_int,
    pub n_threads_batch: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: LlamaPoolingType,
    pub attention_type: LlamaAttentionType,
    pub flash_attn_type: LlamaFlashAttnType,
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: c_uint,
    pub defrag_thold: c_float,
    pub cb_eval: *const c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub abort_callback: *const c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *const c_void,
    pub n_samplers: usize,
}

// llama_sampler_chain_params
#[repr(C)]
pub struct LlamaSamplerChainParams {
    pub no_perf: bool,
}

// mtmd_input_text
#[repr(C)]
pub struct MtmdInputText {
    pub text: *const c_char,
    pub add_special: bool,
    pub parse_special: bool,
}

// mtmd_context_params
#[repr(C)]
pub struct MtmdContextParams {
    pub use_gpu: bool,
    pub print_timings: bool,
    pub n_threads: c_int,
    pub image_marker: *const c_char,
    pub media_marker: *const c_char,
    pub flash_attn_type: LlamaFlashAttnType,
    pub warmup: bool,
    pub image_min_tokens: c_int,
    pub image_max_tokens: c_int,
    pub cb_eval: *const c_void,
    pub cb_eval_user_data: *mut c_void,
}

// mtmd_input_chunk_type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtmdInputChunkType {
    Text = 0,
    Image = 1,
    Audio = 2,
}

// mtmd_decoder_pos
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdDecoderPos {
    pub t: c_uint,
    pub x: c_uint,
    pub y: c_uint,
}

// ggml_log_level
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlLogLevel {
    None = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Cont = 5,
}

pub type GgmlLogCallback = Option<unsafe extern "C" fn(level: c_int, text: *const c_char, user_data: *mut c_void)>;

extern "C" {
    // Backend init/free
    pub fn llama_backend_init();
    pub fn llama_backend_free();

    // Logging
    pub fn llama_log_set(log_callback: GgmlLogCallback, user_data: *mut c_void);
    pub fn mtmd_log_set(log_callback: GgmlLogCallback, user_data: *mut c_void);
    pub fn mtmd_helper_log_set(log_callback: GgmlLogCallback, user_data: *mut c_void);

    // Default params
    pub fn llama_model_default_params() -> LlamaModelParams;
    pub fn llama_context_default_params() -> LlamaContextParams;
    pub fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;

    // Model
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: LlamaModelParams,
    ) -> *mut LlamaModel;
    pub fn llama_model_free(model: *mut LlamaModel);
    pub fn llama_model_n_embd(model: *const LlamaModel) -> c_int;
    pub fn llama_model_n_embd_inp(model: *const LlamaModel) -> c_int;
    pub fn llama_model_n_layer(model: *const LlamaModel) -> c_int;
    pub fn llama_model_rope_type(model: *const LlamaModel) -> c_int;

    // Context
    pub fn llama_init_from_model(
        model: *mut LlamaModel,
        params: LlamaContextParams,
    ) -> *mut LlamaContext;
    pub fn llama_free(ctx: *mut LlamaContext);
    pub fn llama_n_ctx(ctx: *const LlamaContext) -> c_uint;
    pub fn llama_get_model(ctx: *const LlamaContext) -> *const LlamaModel;
    pub fn llama_get_memory(ctx: *const LlamaContext) -> LlamaMemoryT;
    pub fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;

    // Threading
    pub fn llama_set_n_threads(ctx: *mut LlamaContext, n_threads: c_int, n_threads_batch: c_int);

    // Attention
    pub fn llama_set_causal_attn(ctx: *mut LlamaContext, causal_attn: bool);
    pub fn llama_set_embeddings(ctx: *mut LlamaContext, embeddings: bool);

    // Batch
    pub fn llama_batch_init(n_tokens: c_int, embd: c_int, n_seq_max: c_int) -> LlamaBatch;
    pub fn llama_batch_free(batch: LlamaBatch);
    pub fn llama_batch_get_one(tokens: *mut LlamaToken, n_tokens: c_int) -> LlamaBatch;

    // Decode
    pub fn llama_encode(ctx: *mut LlamaContext, batch: LlamaBatch) -> c_int;
    pub fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> c_int;

    // Logits
    pub fn llama_get_logits(ctx: *mut LlamaContext) -> *mut c_float;
    pub fn llama_get_logits_ith(ctx: *mut LlamaContext, i: c_int) -> *mut c_float;

    // Vocab
    pub fn llama_vocab_n_tokens(vocab: *const LlamaVocab) -> c_int;

    // Token operations
    pub fn llama_tokenize(
        vocab: *const LlamaVocab,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut LlamaToken,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;

    pub fn llama_token_to_piece(
        vocab: *const LlamaVocab,
        token: LlamaToken,
        buf: *mut c_char,
        length: c_int,
        lstrip: c_int,
        special: bool,
    ) -> c_int;

    pub fn llama_vocab_is_eog(vocab: *const LlamaVocab, token: LlamaToken) -> bool;

    // Sampler
    pub fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;
    pub fn llama_sampler_chain_add(chain: *mut LlamaSampler, smpl: *mut LlamaSampler);
    pub fn llama_sampler_init_greedy() -> *mut LlamaSampler;
    pub fn llama_sampler_sample(
        smpl: *mut LlamaSampler,
        ctx: *mut LlamaContext,
        idx: c_int,
    ) -> LlamaToken;
    pub fn llama_sampler_accept(smpl: *mut LlamaSampler, token: LlamaToken);
    pub fn llama_sampler_free(smpl: *mut LlamaSampler);

    // Memory
    pub fn llama_memory_clear(mem: LlamaMemoryT, data: bool);

    // mtmd functions
    pub fn mtmd_context_params_default() -> MtmdContextParams;
    pub fn mtmd_init_from_file(
        mmproj_fname: *const c_char,
        text_model: *const LlamaModel,
        ctx_params: MtmdContextParams,
    ) -> *mut MtmdContext;
    pub fn mtmd_free(ctx: *mut MtmdContext);

    pub fn mtmd_default_marker() -> *const c_char;

    pub fn mtmd_bitmap_init(
        nx: c_uint,
        ny: c_uint,
        data: *const u8,
    ) -> *mut MtmdBitmap;
    pub fn mtmd_bitmap_free(bitmap: *mut MtmdBitmap);

    pub fn mtmd_input_chunks_init() -> *mut MtmdInputChunks;
    pub fn mtmd_input_chunks_size(chunks: *const MtmdInputChunks) -> usize;
    pub fn mtmd_input_chunks_get(
        chunks: *const MtmdInputChunks,
        idx: usize,
    ) -> *const MtmdInputChunk;
    pub fn mtmd_input_chunks_free(chunks: *mut MtmdInputChunks);

    pub fn mtmd_input_chunk_get_type(chunk: *const MtmdInputChunk) -> MtmdInputChunkType;
    pub fn mtmd_input_chunk_get_tokens_text(
        chunk: *const MtmdInputChunk,
        n_tokens_output: *mut usize,
    ) -> *const LlamaToken;
    pub fn mtmd_input_chunk_get_n_tokens(chunk: *const MtmdInputChunk) -> usize;
    pub fn mtmd_input_chunk_get_n_pos(chunk: *const MtmdInputChunk) -> LlamaPos;

    pub fn mtmd_tokenize(
        ctx: *mut MtmdContext,
        output: *mut MtmdInputChunks,
        text: *const MtmdInputText,
        bitmaps: *const *const MtmdBitmap,
        n_bitmaps: usize,
    ) -> c_int;

    pub fn mtmd_encode_chunk(
        ctx: *mut MtmdContext,
        chunk: *const MtmdInputChunk,
    ) -> c_int;

    pub fn mtmd_get_output_embd(ctx: *mut MtmdContext) -> *mut c_float;
    pub fn mtmd_decode_use_non_causal(
        ctx: *mut MtmdContext,
        chunk: *const MtmdInputChunk,
    ) -> bool;
    pub fn mtmd_decode_use_mrope(ctx: *mut MtmdContext) -> bool;

    // Helper functions from mtmd-helper.h
    pub fn mtmd_helper_bitmap_init_from_file(
        ctx: *mut MtmdContext,
        fname: *const c_char,
    ) -> *mut MtmdBitmap;

    pub fn mtmd_helper_eval_chunks(
        ctx: *mut MtmdContext,
        lctx: *mut LlamaContext,
        chunks: *const MtmdInputChunks,
        n_past: LlamaPos,
        seq_id: LlamaSeqId,
        n_batch: c_int,
        logits_last: bool,
        new_n_past: *mut LlamaPos,
    ) -> c_int;

    pub fn mtmd_helper_get_n_tokens(chunks: *const MtmdInputChunks) -> usize;
    pub fn mtmd_helper_get_n_pos(chunks: *const MtmdInputChunks) -> LlamaPos;
}
