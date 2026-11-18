use std::{env, fs, path::PathBuf};

#[path = "../build_support.rs"]
mod build_support;

use build_support::{Baseline, emit_rerun_metadata, env_usize};

fn main() {
    let baseline = Baseline::load();
    let bounded_swiglu = build_support::bounded_swiglu();
    let mlp_initializer = match env::var("GPT2_MLP_INITIALIZER") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => "legacy".into(),
        Err(error) => panic!("invalid GPT2_MLP_INITIALIZER: {error}"),
    };
    let mlp_reference_normal_init = match mlp_initializer.as_str() {
        "legacy" => false,
        "reference-normal-nvfp4" => true,
        other => panic!("unsupported GPT2_MLP_INITIALIZER={other}"),
    };
    assert!(
        !mlp_reference_normal_init || bounded_swiglu,
        "reference-normal-nvfp4 initialization requires bounded-swiglu"
    );
    let value_rms = build_support::value_rms();
    let full_kda_backward = build_support::full_kda_backward();
    let stationary_rope = build_support::stationary_rope();
    let kda_rational_output_gate = match env::var("GPT2_KDA_OUTPUT_GATE").as_deref() {
        Ok("sigmoid") | Err(env::VarError::NotPresent) => false,
        Ok("rational") => true,
        other => panic!("unsupported GPT2_KDA_OUTPUT_GATE={other:?}; use sigmoid or rational"),
    };
    assert!(
        !(stationary_rope && (bounded_swiglu || value_rms)) || full_kda_backward,
        "stationary RoPE with numerical components requires the complete full-credit Engram/mHC graph"
    );
    assert!(
        !(bounded_swiglu && value_rms) || full_kda_backward,
        "bounded SwiGLU with value RMS requires the complete full-credit Engram/mHC graph"
    );
    let seq_len = env_usize("GPT2_SEQ_LEN")
        .or_else(|| baseline.usize("GPT2_SEQ_LEN"))
        .unwrap_or(2048);
    let batch_size = env_usize("GPT2_BATCH_SIZE")
        .or_else(|| baseline.usize("GPT2_BATCH_SIZE"))
        .unwrap_or(4);
    let n_layer = env_usize("GPT2_N_LAYER")
        .or_else(|| baseline.usize("GPT2_N_LAYER"))
        .unwrap_or(16);
    let n_head = env_usize("GPT2_N_HEAD")
        .or_else(|| baseline.usize("GPT2_N_HEAD"))
        .unwrap_or(32);
    let n_embd = env_usize("GPT2_N_EMBD")
        .or_else(|| baseline.usize("GPT2_N_EMBD"))
        .unwrap_or(2048);
    let full_attention_window = env_usize("GPT2_FULL_ATTENTION_WINDOW")
        .or_else(|| baseline.usize("GPT2_FULL_ATTENTION_WINDOW"))
        .unwrap_or(seq_len);
    let budget_name = "GPT2_ATTENTION_BACKWARD_TILE_BUDGET";
    let budget_override = env::var(budget_name).ok();
    let backward_tile_budget = budget_override
        .as_deref()
        .or_else(|| baseline.value(budget_name))
        .unwrap_or("12")
        .parse::<f32>()
        .expect("GPT2_ATTENTION_BACKWARD_TILE_BUDGET must be a number");
    let dense_value_gradient = match env::var("GPT2_FULL_ATTN_DENSE_DV").as_deref() {
        Ok("0") | Err(env::VarError::NotPresent) => false,
        Ok("1") => true,
        other => panic!("unsupported GPT2_FULL_ATTN_DENSE_DV={other:?}; use 0 or 1"),
    };

    assert!(
        seq_len >= 2048,
        "GPT2_SEQ_LEN must be >= 2048 for the pretraining target"
    );
    assert!(batch_size > 0, "GPT2_BATCH_SIZE must be > 0");
    assert!(
        n_layer >= 16,
        "GPT2_N_LAYER must be >= 16 for the 1B target"
    );
    assert!(
        n_embd >= 2048,
        "GPT2_N_EMBD must be >= 2048 for the 1B target"
    );
    assert!(n_head >= 32, "GPT2_N_HEAD must be >= 32 for the 1B target");
    assert_eq!(
        n_embd % n_head,
        0,
        "GPT2_N_EMBD must be divisible by GPT2_N_HEAD"
    );
    assert!(
        (1..=seq_len).contains(&full_attention_window),
        "GPT2_FULL_ATTENTION_WINDOW must be in 1..=GPT2_SEQ_LEN"
    );
    assert_eq!(
        full_attention_window % 128,
        0,
        "GPT2_FULL_ATTENTION_WINDOW must be aligned to the 128-token attention CTA"
    );
    assert!(
        backward_tile_budget.is_finite() && backward_tile_budget >= 0.0,
        "GPT2_ATTENTION_BACKWARD_TILE_BUDGET must be finite and nonnegative"
    );

    emit_rerun_metadata(&[
        "GPT2_MLP_KIND",
        "GPT2_MLP_INITIALIZER",
        "GPT2_VALUE_RMS",
        "GPT2_KDA_BACKWARD",
        "GPT2_KDA_OUTPUT_GATE",
        "GPT2_POSITION_KIND",
        "GPT2_SEQ_LEN",
        "GPT2_BATCH_SIZE",
        "GPT2_N_LAYER",
        "GPT2_N_HEAD",
        "GPT2_N_EMBD",
        "GPT2_FULL_ATTENTION_WINDOW",
        budget_name,
        "GPT2_FULL_ATTN_DENSE_DV",
    ]);

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    fs::write(
        out.join("gpt2_shape.rs"),
        format!(
            "pub const GPT2_SEQ_LEN: usize = {seq_len};\n\
             pub const GPT2_BATCH_SIZE: usize = {batch_size};\n\
             pub const GPT2_N_LAYER: usize = {n_layer};\n\
             pub const GPT2_N_HEAD: usize = {n_head};\n\
             pub const GPT2_N_EMBD: usize = {n_embd};\n\
             pub const GPT2_FULL_ATTENTION_WINDOW: usize = {full_attention_window};\n\
             pub const GPT2_ATTENTION_BACKWARD_TILE_BUDGET: f32 = {backward_tile_budget:?};\n\
             pub const GPT2_FULL_ATTN_DENSE_DV: bool = {dense_value_gradient};\n\
             pub const GPT2_BOUNDED_SWIGLU: bool = {bounded_swiglu};\n\
             pub const GPT2_MLP_REFERENCE_NORMAL_INIT: bool = {mlp_reference_normal_init};\n\
             pub const GPT2_VALUE_RMS: bool = {value_rms};\n\
             pub const GPT2_KDA_FULL_STATE_BACKWARD: bool = {full_kda_backward};\n\
             pub const GPT2_KDA_RATIONAL_OUTPUT_GATE: bool = {kda_rational_output_gate};\n\
             pub const GPT2_STATIONARY_ROPE: bool = {stationary_rope};\n"
        ),
    )
    .expect("failed to write generated GPT-2 shape");
}
