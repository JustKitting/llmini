use std::{env, fs, path::PathBuf};

struct TokenizerConfig {
    name: &'static str,
    asset_subdir: &'static str,
    tokenizer_vocab_size: usize,
    model_vocab_size: usize,
    bos_token: u32,
    eos_token: u32,
    document_prefix: &'static [u32],
    document_suffix: &'static [u32],
    fineweb_shard_prefix: &'static str,
    fineweb_marker: &'static str,
    synth_shard_prefix: &'static str,
    synth_marker: &'static str,
    document_boundaries: &'static str,
}

fn main() {
    println!("cargo:rerun-if-env-changed=TOKENIZER_VARIANT");
    let variant = env::var("TOKENIZER_VARIANT").unwrap_or_else(|_| "mistral_v01".to_string());
    let config = match variant.as_str() {
        "llama2" => TokenizerConfig {
            name: "llama2",
            asset_subdir: "llama2",
            tokenizer_vocab_size: 32_000,
            model_vocab_size: 32_000,
            bos_token: 1,
            eos_token: 2,
            document_prefix: &[1],
            document_suffix: &[2],
            fineweb_shard_prefix: "fineweb_llama2",
            fineweb_marker: ".fineweb_llama2_v1",
            synth_shard_prefix: "synth_llama2",
            synth_marker: ".synth_llama2_v1",
            document_boundaries: "bos_eos",
        },
        "mistral_v01" => TokenizerConfig {
            name: "mistral_v01",
            asset_subdir: "mistral_v01",
            tokenizer_vocab_size: 32_000,
            model_vocab_size: 32_000,
            bos_token: 1,
            eos_token: 2,
            document_prefix: &[1],
            document_suffix: &[2],
            fineweb_shard_prefix: "fineweb_mistral_v01",
            fineweb_marker: ".fineweb_mistral_v01_v1",
            synth_shard_prefix: "synth_mistral_v01",
            synth_marker: ".synth_mistral_v01_v1",
            document_boundaries: "bos_eos",
        },
        other => panic!("unknown TOKENIZER_VARIANT={other}; expected llama2 or mistral_v01"),
    };

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    fs::write(
        out.join("training_tokenizer_config.rs"),
        format!(
            "pub const TOKENIZER_NAME: &str = {name:?};\n\
             pub const TOKENIZER_ASSET_SUBDIR: &str = {asset_subdir:?};\n\
             pub const TOKENIZER_VOCAB_SIZE: usize = {tokenizer_vocab_size};\n\
             pub const MODEL_VOCAB_SIZE: usize = {model_vocab_size};\n\
             pub const BOS_TOKEN: u32 = {bos_token};\n\
             pub const EOS_TOKEN: u32 = {eos_token};\n\
             pub const DOCUMENT_PREFIX: &[u32] = &{document_prefix:?};\n\
             pub const DOCUMENT_SUFFIX: &[u32] = &{document_suffix:?};\n\
             pub const FINEWEB_SHARD_FILE_PREFIX: &str = {fineweb_shard_prefix:?};\n\
             pub const FINEWEB_TOKENIZATION_MARKER: &str = {fineweb_marker:?};\n\
             pub const SYNTH_SHARD_FILE_PREFIX: &str = {synth_shard_prefix:?};\n\
             pub const SYNTH_TOKENIZATION_MARKER: &str = {synth_marker:?};\n\
             pub const DOCUMENT_BOUNDARIES: &str = {document_boundaries:?};\n",
            name = config.name,
            asset_subdir = config.asset_subdir,
            tokenizer_vocab_size = config.tokenizer_vocab_size,
            model_vocab_size = config.model_vocab_size,
            bos_token = config.bos_token,
            eos_token = config.eos_token,
            document_prefix = config.document_prefix,
            document_suffix = config.document_suffix,
            fineweb_shard_prefix = config.fineweb_shard_prefix,
            fineweb_marker = config.fineweb_marker,
            synth_shard_prefix = config.synth_shard_prefix,
            synth_marker = config.synth_marker,
            document_boundaries = config.document_boundaries,
        ),
    )
    .expect("failed to write tokenizer build configuration");
}
