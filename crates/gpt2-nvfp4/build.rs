use std::{env, fs, path::PathBuf};

#[path = "../build_support.rs"]
mod build_support;

use build_support::{Baseline, emit_rerun_metadata, env_usize};

fn main() {
    let baseline = Baseline::load();
    let seq_len = env_usize("GPT2_SEQ_LEN")
        .or_else(|| baseline.usize("GPT2_SEQ_LEN"))
        .unwrap_or(8192);
    let batch_size = env_usize("GPT2_BATCH_SIZE")
        .or_else(|| baseline.usize("GPT2_BATCH_SIZE"))
        .unwrap_or(1);
    let n_layer = env_usize("GPT2_N_LAYER")
        .or_else(|| baseline.usize("GPT2_N_LAYER"))
        .unwrap_or(16);
    let n_head = env_usize("GPT2_N_HEAD")
        .or_else(|| baseline.usize("GPT2_N_HEAD"))
        .unwrap_or(32);
    let n_embd = env_usize("GPT2_N_EMBD")
        .or_else(|| baseline.usize("GPT2_N_EMBD"))
        .unwrap_or(2048);

    assert!(
        seq_len >= 8192,
        "GPT2_SEQ_LEN must be >= 8192 for the 8K target"
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

    emit_rerun_metadata(&[
        "GPT2_SEQ_LEN",
        "GPT2_BATCH_SIZE",
        "GPT2_N_LAYER",
        "GPT2_N_HEAD",
        "GPT2_N_EMBD",
    ]);

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    fs::write(
        out.join("gpt2_shape.rs"),
        format!(
            "pub const GPT2_SEQ_LEN: usize = {seq_len};\n\
             pub const GPT2_BATCH_SIZE: usize = {batch_size};\n\
             pub const GPT2_N_LAYER: usize = {n_layer};\n\
             pub const GPT2_N_HEAD: usize = {n_head};\n\
             pub const GPT2_N_EMBD: usize = {n_embd};\n"
        ),
    )
    .expect("failed to write generated GPT-2 shape");
}
