pub const GPT2_VOCAB_SIZE: usize = llama2_tokenizer::VOCAB_SIZE;
include!(concat!(env!("OUT_DIR"), "/gpt2_shape.rs"));

pub const GPT2_TOKEN_ROWS: usize = GPT2_BATCH_SIZE * GPT2_SEQ_LEN;
pub const GPT2_CONTEXT_LEN: usize = GPT2_SEQ_LEN;

pub const GPT2_LAYER_NORM_EPSILON: f32 = 1.0e-5;

pub const GPT2_MLP: usize = 4 * GPT2_N_EMBD;
pub const GPT2_MLP_ROUTE_TILE: usize = 128;
pub const GPT2_MLP_ROUTE_FEATURE_TILES: usize = GPT2_MLP / GPT2_MLP_ROUTE_TILE;
pub const GPT2_MLP_ROUTE_TOKEN_TILES: usize = GPT2_TOKEN_ROWS / GPT2_MLP_ROUTE_TILE;
pub const GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES: usize = 3 * GPT2_MLP_ROUTE_FEATURE_TILES / 4;
pub const GPT2_MLP_ROUTE_SCORES: usize = GPT2_MLP_ROUTE_TOKEN_TILES * GPT2_MLP_ROUTE_FEATURE_TILES;
pub const GPT2_MLP_ROUTE_MASKS: usize = GPT2_MLP_ROUTE_TOKEN_TILES + GPT2_MLP_ROUTE_FEATURE_TILES;
pub const GPT2_MLP_DENSE_PREFIX_LAYERS: usize = 1;
pub const GPT2_VALUE_RESIDUAL_LAYERS: usize = 3 * GPT2_N_LAYER / 8;
pub const GPT2_VALUE_RESIDUAL_START_LAYER: usize = GPT2_N_LAYER - GPT2_VALUE_RESIDUAL_LAYERS;
pub const GPT2_FULL_ATTENTION_QKV: usize = 3 * GPT2_N_EMBD;
pub const GPT2_FULL_ATTENTION_GATED_QKV: usize =
    align_up(GPT2_FULL_ATTENTION_QKV + GPT2_N_HEAD, 128);
pub const GPT2_QK_SCALE_STORAGE: usize = 16;
pub const GPT2_XSA_ALPHA_STORAGE: usize = GPT2_N_LAYER * GPT2_N_HEAD;
pub const GPT2_ATTENTION_BACKWARD_TILE_BUDGET: f32 = 12.0;
pub const GPT2_KDA_ACTIVE_QKV: usize = 4 * GPT2_N_EMBD + GPT2_N_HEAD;
pub const GPT2_QKV: usize = align_kda_qkv(GPT2_KDA_ACTIVE_QKV);
pub const GPT2_EMBEDDING_DIM: u32 = GPT2_N_EMBD as u32;
pub const GPT2_MLP_DIM: u32 = GPT2_MLP as u32;
pub const GPT2_TOKEN_ROWS_U32: u32 = GPT2_TOKEN_ROWS as u32;
pub const GPT2_VOCAB_DIM: u32 = GPT2_VOCAB_SIZE as u32;
pub const GPT2_Q_OFFSET: usize = 0;
pub const GPT2_K_OFFSET: usize = GPT2_N_EMBD;
pub const GPT2_V_OFFSET: usize = 2 * GPT2_N_EMBD;
pub const GPT2_KDA_G_OFFSET: usize = 3 * GPT2_N_EMBD;
pub const GPT2_KDA_BETA_OFFSET: usize = 4 * GPT2_N_EMBD;
pub const GPT2_FULL_ATTENTION_GATE_OFFSET: usize = GPT2_FULL_ATTENTION_QKV;
pub const GPT2_KDA_GATE_OFFSET: usize = GPT2_KDA_BETA_OFFSET + GPT2_N_HEAD;
pub const KDA_CHUNK_SIZE: usize = 64;
pub const KDA_DECAY_SCALE: f32 = 0.01;
pub const NEXTLAT_INPUT: usize = 2 * GPT2_N_EMBD;
pub const NEXTLAT_HIDDEN: usize = NEXTLAT_INPUT;
pub const NEXTLAT_INPUT_DIM: u32 = NEXTLAT_INPUT as u32;
pub const NEXTLAT_HIDDEN_DIM: u32 = NEXTLAT_HIDDEN as u32;

pub const KIMI_FULL_ATTENTION_PERIOD: usize = 4;

const _: () = {
    assert!(GPT2_MLP_ROUTE_TILE == 128);
    assert!(GPT2_MLP.is_multiple_of(GPT2_MLP_ROUTE_TILE));
    assert!(GPT2_TOKEN_ROWS.is_multiple_of(GPT2_MLP_ROUTE_TILE));
    assert!(GPT2_MLP_ROUTE_FEATURE_TILES == 64);
    assert!(GPT2_MLP_ROUTE_TOKEN_TILES == 64);
    assert!(GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES == 48);
    assert!(GPT2_MLP_ROUTE_SCORES == 4096);
    assert!(GPT2_VALUE_RESIDUAL_LAYERS > 0);
    assert!(GPT2_VALUE_RESIDUAL_START_LAYER > 0);
};

const fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

const fn align_kda_qkv(value: usize) -> usize {
    align_up(value, 128)
}

pub const fn uses_full_attention(block_index: usize) -> bool {
    block_index % KIMI_FULL_ATTENTION_PERIOD == KIMI_FULL_ATTENTION_PERIOD - 1
}

pub const fn uses_block_topk_mlp(block_index: usize) -> bool {
    block_index >= GPT2_MLP_DENSE_PREFIX_LAYERS
}

pub const fn uses_value_residual(block_index: usize) -> bool {
    block_index >= GPT2_VALUE_RESIDUAL_START_LAYER
}

pub fn layer_norm_scale(block_index: usize) -> f32 {
    1.0 / ((block_index + 1) as f32).sqrt()
}

pub fn qk_norm_initial_scale() -> f32 {
    let visible_tokens = GPT2_FULL_ATTENTION_WINDOW as f32;
    (visible_tokens * visible_tokens - visible_tokens).log2()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2Config;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionDims {
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

impl AttentionDims {
    pub fn new(use_full_attention: bool) -> Self {
        let mut qkv_dim = Gpt2Config::attention_qkv_dim(use_full_attention);
        if use_full_attention && attention_headwise_gate_enabled() {
            qkv_dim = GPT2_FULL_ATTENTION_GATED_QKV;
        }
        Self {
            embedding_dim: GPT2_EMBEDDING_DIM,
            qkv_dim: qkv_dim as u32,
            head_count: GPT2_N_HEAD as u32,
            head_dim: Gpt2Config::head_dim() as u32,
        }
    }
}

pub fn attention_headwise_gate_enabled() -> bool {
    env_bool("TRAIN_ATTENTION_HEADWISE_GATE", true)
}

pub fn exclusive_self_attention_enabled() -> bool {
    env_bool("TRAIN_XSA", true)
}

pub fn exclusive_self_attention_kda_enabled() -> bool {
    env_bool("TRAIN_XSA_KDA", true)
}

pub fn uses_exclusive_self_attention(use_full_attention: bool) -> bool {
    exclusive_self_attention_enabled()
        && (use_full_attention || exclusive_self_attention_kda_enabled())
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub const fn attention_headwise_gate_offset(use_full_attention: bool) -> usize {
    if use_full_attention {
        GPT2_FULL_ATTENTION_GATE_OFFSET
    } else {
        GPT2_KDA_GATE_OFFSET
    }
}

pub fn attention_trainable_qkv_dim(use_full_attention: bool) -> usize {
    if use_full_attention && attention_headwise_gate_enabled() {
        GPT2_FULL_ATTENTION_QKV + GPT2_N_HEAD
    } else {
        Gpt2Config::attention_qkv_dim(use_full_attention)
    }
}

impl Gpt2Config {
    pub const fn gpt2_124m() -> Self {
        Self
    }

    pub const fn vocab_size(self) -> usize {
        GPT2_VOCAB_SIZE
    }

    pub const fn max_seq_len(self) -> usize {
        GPT2_SEQ_LEN
    }

    pub const fn head_dim() -> usize {
        GPT2_N_EMBD / GPT2_N_HEAD
    }

    pub const fn mlp_hidden(self) -> usize {
        GPT2_MLP
    }

    pub const fn attention_qkv_dim(use_full_attention: bool) -> usize {
        if use_full_attention {
            GPT2_FULL_ATTENTION_QKV
        } else {
            GPT2_QKV
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GPT2_FULL_ATTENTION_WINDOW, GPT2_VALUE_RESIDUAL_LAYERS, GPT2_VALUE_RESIDUAL_START_LAYER,
        layer_norm_scale, qk_norm_initial_scale, uses_value_residual,
    };

    #[test]
    fn layer_norm_scale_uses_one_indexed_inverse_square_root() {
        assert_eq!(layer_norm_scale(0), 1.0);
        assert_eq!(layer_norm_scale(3), 0.5);
        assert_eq!(layer_norm_scale(15), 0.25);
    }

    #[test]
    fn sparse_value_residual_uses_the_final_three_eighths() {
        assert_eq!(GPT2_VALUE_RESIDUAL_LAYERS, 6);
        assert_eq!(GPT2_VALUE_RESIDUAL_START_LAYER, 10);
        assert!(!uses_value_residual(0));
        assert!(!uses_value_residual(9));
        assert!(uses_value_residual(10));
        assert!(uses_value_residual(15));
    }

    #[test]
    fn qk_norm_scale_uses_the_visible_attention_length() {
        let length = GPT2_FULL_ATTENTION_WINDOW as f32;
        assert_eq!(qk_norm_initial_scale(), (length * length - length).log2());
    }
}
