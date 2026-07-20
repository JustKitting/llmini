use gpt2_nvfp4::{
    GPT2_ATTENTION_BACKWARD_TILE_BUDGET, GPT2_BATCH_SIZE, GPT2_FULL_ATTENTION_WINDOW, GPT2_N_EMBD,
    GPT2_N_HEAD, GPT2_N_LAYER, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS, GPT2_VALUE_RESIDUAL_LAYERS,
    GPT2_VALUE_RESIDUAL_START_LAYER,
};
use rust_kernels_cuda::optimizer::{MUON_COOPERATIVE_BLOCKS, MUON_MATRIX_PHASES};

use super::super::TrainConfig;

pub(in crate::training::launch) fn build_run_info(dataset: &str, config: &TrainConfig) -> String {
    let mut info = String::new();
    push_info(&mut info, "dataset", dataset);
    push_info(&mut info, "training_launcher", "burn");
    push_info(&mut info, "metric_logger", "burn_file");
    push_info(&mut info, "tokenizer", llama2_tokenizer::TOKENIZER_NAME);
    push_info(
        &mut info,
        "tokenizer_vocab_size",
        llama2_tokenizer::TOKENIZER_VOCAB_SIZE,
    );
    push_info(
        &mut info,
        "model_vocab_size",
        llama2_tokenizer::MODEL_VOCAB_SIZE,
    );
    push_info(
        &mut info,
        "unused_model_vocab_rows",
        llama2_tokenizer::MODEL_VOCAB_SIZE - llama2_tokenizer::TOKENIZER_VOCAB_SIZE,
    );
    push_info(
        &mut info,
        "document_boundaries",
        llama2_tokenizer::DOCUMENT_BOUNDARIES,
    );
    push_info(&mut info, "gpt2_seq_len", GPT2_SEQ_LEN);
    push_info(&mut info, "gpt2_batch_size", GPT2_BATCH_SIZE);
    push_info(&mut info, "gpt2_token_rows", GPT2_TOKEN_ROWS);
    push_info(&mut info, "gpt2_n_layer", GPT2_N_LAYER);
    push_info(&mut info, "gpt2_n_head", GPT2_N_HEAD);
    push_info(&mut info, "gpt2_n_embd", GPT2_N_EMBD);
    push_info(
        &mut info,
        "gpt2_value_residual_layers",
        GPT2_VALUE_RESIDUAL_LAYERS,
    );
    push_info(
        &mut info,
        "gpt2_value_residual_start_layer",
        GPT2_VALUE_RESIDUAL_START_LAYER,
    );
    push_info(
        &mut info,
        "gpt2_full_attention_window",
        GPT2_FULL_ATTENTION_WINDOW,
    );
    push_info(
        &mut info,
        "gpt2_attention_backward_tile_budget",
        GPT2_ATTENTION_BACKWARD_TILE_BUDGET,
    );
    push_info(
        &mut info,
        "nextlat_loss_weight",
        crate::training::next_latent::loss_weight(),
    );
    push_info(
        &mut info,
        "muon_cooperative_blocks",
        MUON_COOPERATIVE_BLOCKS,
    );
    push_info(&mut info, "muon_matrix_phases", MUON_MATRIX_PHASES);
    push_info(
        &mut info,
        "hyperball_enabled",
        crate::training::optimizer_muon::hyperball_enabled(),
    );
    push_info(
        &mut info,
        "hyperball_lr",
        crate::training::optimizer_muon::hyperball_learning_rate(),
    );
    push_info(
        &mut info,
        "hyperball_amuse",
        crate::training::optimizer_muon::hyperball_uses_schedule_free(),
    );
    push_info(
        &mut info,
        "hyperball_polar_period",
        crate::training::optimizer_muon::hyperball_polar_period(),
    );
    push_info(
        &mut info,
        "hyperball_polar_numerator",
        crate::training::optimizer_muon::hyperball_polar_numerator(),
    );
    push_info(
        &mut info,
        "muon_vs_enabled",
        crate::training::optimizer_muon::muon_vs_enabled(),
    );
    push_info(
        &mut info,
        "ember_enabled",
        crate::training::optimizer_apply::ember::enabled(),
    );
    push_info(
        &mut info,
        "ember_learning_rate",
        crate::training::optimizer_apply::ember::learning_rate(100),
    );
    push_info(
        &mut info,
        "ember_beta2",
        crate::training::optimizer_apply::ember::beta2(),
    );
    push_info(
        &mut info,
        "ember_weight_decay",
        crate::training::optimizer_apply::ember::weight_decay(),
    );
    push_info(
        &mut info,
        "symexp_lin_enabled",
        crate::training::symexp_lin::enabled(),
    );
    push_info(
        &mut info,
        "symexp_lin_beta",
        crate::training::symexp_lin::beta(),
    );
    push_info(
        &mut info,
        "symexp_lin_learn_scales",
        crate::training::symexp_lin::learn_scales(),
    );
    push_info(
        &mut info,
        "attention_headwise_gate_enabled",
        gpt2_nvfp4::attention_headwise_gate_enabled(),
    );
    push_info(
        &mut info,
        "exclusive_self_attention_enabled",
        gpt2_nvfp4::exclusive_self_attention_enabled(),
    );
    push_info(
        &mut info,
        "exclusive_self_attention_kda_enabled",
        gpt2_nvfp4::exclusive_self_attention_kda_enabled(),
    );
    push_info(
        &mut info,
        "partial_key_offset_enabled",
        gpt2_nvfp4::partial_key_offset_enabled(),
    );
    push_info(
        &mut info,
        "selective_attention_enabled",
        gpt2_nvfp4::selective_attention_enabled(),
    );
    push_info(
        &mut info,
        "stable_mask_gamma",
        gpt2_nvfp4::stable_mask_gamma(),
    );
    push_info(
        &mut info,
        "canon_ac_enabled",
        gpt2_nvfp4::canon_ac_enabled(),
    );
    push_info(&mut info, "step_cap", config.step_cap);
    push_info(&mut info, "log_interval", config.log_interval);
    push_info(&mut info, "max_seconds", config.max_seconds);
    let eval_interval = config
        .eval_interval
        .map_or_else(|| "none".to_string(), |value| value.to_string());
    push_info(&mut info, "eval_interval", eval_interval);
    push_info(&mut info, "seed", format!("{:#x}", config.seed));
    push_run_env(&mut info);
    info
}

fn push_run_env(info: &mut String) {
    for name in [
        "CUDA_DEVICE_INDEX",
        "TRAIN_DATASET",
        "TRAIN_LOAD_MODEL",
        "TRAIN_SAVE_MODEL",
        "TRAIN_STEPS",
        "TRAIN_LOG_INTERVAL",
        "TRAIN_EVAL_INTERVAL",
        "TRAIN_MAX_SECONDS",
        "TRAIN_REPEAT_BATCH",
        "TRAIN_SEED",
        "TRAIN_LR_SCALE",
        "TRAIN_ADAM_LR_SCALE",
        "TRAIN_NEXTLAT_LR_SCALE",
        "TRAIN_LR_WARMUP_STEPS",
        "TRAIN_LR_START_RATIO",
        "TRAIN_AMUSE_BETA1",
        "TRAIN_AMUSE_RHO",
        "TRAIN_HYPERBALL",
        "TRAIN_HYPERBALL_LR",
        "TRAIN_HYPERBALL_AMUSE",
        "TRAIN_HYPERBALL_POLAR_PERIOD",
        "TRAIN_HYPERBALL_POLAR_NUMERATOR",
        "TRAIN_MUON_VS",
        "TRAIN_EMBER",
        "TRAIN_EMBER_LR",
        "TRAIN_EMBER_BETA2",
        "TRAIN_EMBER_WEIGHT_DECAY",
        "TRAIN_EMBER_EPS",
        "TRAIN_SYMEXP_LIN",
        "TRAIN_SYMEXP_LIN_BETA",
        "TRAIN_SYMEXP_LIN_LEARN_SCALES",
        "TRAIN_SYMEXP_LIN_ANNEAL_STEPS",
        "TRAIN_SYMEXP_LIN_REUSE_AMAX",
        "TRAIN_ATTENTION_HEADWISE_GATE",
        "TRAIN_XSA",
        "TRAIN_XSA_KDA",
        "TRAIN_PARTIAL_KEY_OFFSET",
        "TRAIN_SELECTIVE_ATTENTION",
        "TRAIN_CANON_AC",
        "TRAIN_NEXTLAT_LOSS_WEIGHT",
        "TRAIN_SKIP_UNSTABLE_UPDATES",
        "TRAIN_SKIP_ROLLING_INTERVAL",
        "TRAIN_SKIP_SIGMA_FACTOR",
        "TRAIN_SKIP_USE_LOSS",
        "TRAIN_SKIP_USE_GRAD_NORM",
        "TRAIN_GENERATE_PROMPT",
        "TRAIN_GENERATE_TOKENS",
        "TRAIN_GENERATE_TEMPERATURE",
        "TRAIN_GENERATE_TOP_K",
        "TRAIN_GENERATE_TOP_P",
    ] {
        if let Ok(value) = std::env::var(name) {
            push_info(info, name, value);
        }
    }
}

fn push_info(info: &mut String, name: &str, value: impl std::fmt::Display) {
    use std::fmt::Write;
    let _ = writeln!(info, "{name}={value}");
}
