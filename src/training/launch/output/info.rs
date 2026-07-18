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
    push_info(&mut info, "vocab_size", llama2_tokenizer::VOCAB_SIZE);
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
