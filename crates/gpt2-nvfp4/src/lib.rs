mod activations;
mod backward;
mod config;
mod random;
mod tensor;
mod types;

pub use activations::{
    AttentionLogSumExp, HiddenState, Logits, MlpActivation, NextLatHiddenActivation,
    NextLatInputActivation, QkvActivation,
};

pub use backward::{
    AttentionBackwardModules, AttentionBackwardSeeds, AttentionCProjBackwardArgs,
    AttentionCProjScratch, AttentionCoreBackwardArgs, AttentionCoreScratch,
    AttentionCoreScratchBuffers, AttentionQkvBackwardArgs, AttentionQkvScratch,
    BlockAttentionBackwardArgs, BlockAttentionBackwardModules, BlockAttentionBackwardScratch,
    BlockAttentionBackwardSeeds, BlockMlpBackwardArgs, BlockMlpBackwardModules,
    FinalHeadBackwardArgs, FinalHeadBackwardModules, FinalHeadBackwardScratch,
    FinalHeadBackwardSeeds, Gpt2BackwardArgs, Gpt2BackwardModules, Gpt2BackwardScratch,
    Gpt2BackwardSeeds, Gpt2BackwardWeights, Gpt2LayerNormBackwardAddAmaxArgs,
    Gpt2LayerNormBackwardAmaxArgs, Gpt2LayerNormBackwardArgs,
    Gpt2LayerNormBackwardInputAddAmaxArgs, Gpt2LayerNormBackwardInputAmaxArgs,
    Gpt2LayerNormBackwardInputArgs, Gpt2LayerNormBackwardParamArgs, LinearScratch, MlpBackwardArgs,
    MlpBackwardGrads, MlpBackwardModules, MlpBackwardScratch, MlpBackwardSeeds,
    attention_c_proj_backward, attention_side_backward, causal_attention_backward,
    final_head_backward, gpt2_backward, layer_norm_backward, layer_norm_backward_add_amax,
    layer_norm_backward_amax, layer_norm_backward_input, layer_norm_backward_input_add_amax,
    layer_norm_backward_input_amax, layer_norm_backward_params, mlp_backward, mlp_side_backward,
    qkv_projection_backward,
};

pub use config::AttentionDims;
pub use config::{
    GPT2_ATTENTION_BACKWARD_TILE_BUDGET, GPT2_BATCH_SIZE, GPT2_CONTEXT_LEN,
    GPT2_FULL_ATTENTION_QKV, GPT2_FULL_ATTENTION_WINDOW, GPT2_K_OFFSET, GPT2_KDA_BETA_OFFSET,
    GPT2_KDA_G_OFFSET, GPT2_LAYER_NORM_EPSILON, GPT2_MLP, GPT2_MLP_DENSE_PREFIX_LAYERS,
    GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES, GPT2_MLP_ROUTE_FEATURE_TILES, GPT2_MLP_ROUTE_MASKS,
    GPT2_MLP_ROUTE_SCORES, GPT2_MLP_ROUTE_TILE, GPT2_MLP_ROUTE_TOKEN_TILES, GPT2_N_EMBD,
    GPT2_N_HEAD, GPT2_N_LAYER, GPT2_Q_OFFSET, GPT2_QKV, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS,
    GPT2_V_OFFSET, GPT2_VALUE_RESIDUAL_LAYERS, GPT2_VALUE_RESIDUAL_START_LAYER, GPT2_VOCAB_SIZE,
    Gpt2Config, KDA_CHUNK_SIZE, KDA_DECAY_SCALE, KIMI_FULL_ATTENTION_PERIOD, NEXTLAT_HIDDEN,
    NEXTLAT_INPUT, layer_norm_scale, uses_block_topk_mlp, uses_full_attention, uses_value_residual,
};
pub use config::{
    GPT2_EMBEDDING_DIM, GPT2_MLP_DIM, GPT2_TOKEN_ROWS_U32, GPT2_VOCAB_DIM, NEXTLAT_HIDDEN_DIM,
    NEXTLAT_INPUT_DIM,
};

pub use tensor::{FixedBytes, Nvfp4Shape, Nvfp4Tensor};

pub use random::Gpt2Rng;

pub use types::{
    AttentionForwardArgs, AttentionForwardTape, AttentionProjectionTensors, AttentionWeights,
    BlockBackwardGrads, BlockForwardArgs, BlockForwardSaved, BlockForwardTape, Gpt2,
    Gpt2BackwardContext, Gpt2BackwardGrads, Gpt2BlockWeights, Gpt2ForwardArgs, Gpt2ForwardSaved,
    Gpt2ForwardTape, Gpt2Weights, HiddenStateDevice, HiddenStateNvfp4, LayerNormForwardArgs,
    LayerNormGrads, LayerNormSaved, LayerNormTape, LayerNormTensors, LayerNormWeights,
    LinearWeights, MlpActivationNvfp4, MlpDownTensors, MlpForwardArgs, MlpForwardTape,
    MlpProjectionTensors, MlpScratch, MlpUpTensors, MlpWeights, NextLatWeights,
    RowwiseNvfp4Buffers, RowwiseNvfp4Scratch, RowwiseNvfp4Tape, TokenEmbeddingArgs,
};

pub use types::{
    HiddenVectorShape, MlpDownWeightShape, MlpUpWeightShape, MlpVectorShape, NextLatHiddenShape,
    NextLatInputShape, NextLatOutWeightShape, NextLatProjectionWeightShape,
    NextLatTransitionWeightShape, QkvVectorShape, QkvWeightShape, ResidualWeightShape,
    TokenEmbeddingShape,
};

pub use types::{
    LayerNormTensor, MlpDownLinear, MlpUpLinear, QkvLinear, ResidualLinear, TokenEmbedding,
};

pub use types::{nvfp4_bytes, nvfp4_scales};
