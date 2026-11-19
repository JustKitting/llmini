use std::{env, fs, path::PathBuf};

#[path = "build_nvfp4_config.rs"]
mod build_nvfp4_config;
#[path = "../build_support.rs"]
mod build_support;

use build_support::{Baseline, emit_rerun_metadata, env_usize};

fn main() {
    let baseline = Baseline::load();
    let cooperative_blocks = env_usize("MUON_COOPERATIVE_BLOCKS")
        .or_else(|| baseline.usize("MUON_COOPERATIVE_BLOCKS"))
        .unwrap_or(125);
    let matrix_phases = env_usize("MUON_MATRIX_PHASES")
        .or_else(|| baseline.usize("MUON_MATRIX_PHASES"))
        .unwrap_or(16);

    assert!(
        cooperative_blocks > 0,
        "MUON_COOPERATIVE_BLOCKS must be > 0"
    );
    assert!(matrix_phases > 0, "MUON_MATRIX_PHASES must be > 0");

    emit_rerun_metadata(&[
        "MUON_COOPERATIVE_BLOCKS",
        "MUON_MATRIX_PHASES",
        "GPT2_N_EMBD",
        "GPT2_MLP_KIND",
        "GPT2_VALUE_RMS",
        "GPT2_POSITION_KIND",
        "GPT2_KDA_BACKWARD",
        "GPT2_ACTIVATION_QUANT",
    ]);

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    build_nvfp4_config::emit_nvfp4_config(&out);
    fs::write(
        out.join("nvfp4_activation_config.rs"),
        format!(
            "pub const ACTIVATION_MAE_FOUR_SIX: bool = {};\n",
            build_support::activation_mae_four_six()
        ),
    )
    .expect("write NVFP4 activation policy");
    let mlp_kind = build_support::mlp_kind();
    let bounded_swiglu = mlp_kind == build_support::MlpKind::BoundedSwiGlu;
    let adaptive_relu2 = mlp_kind == build_support::MlpKind::AdaptiveRelu2;
    let value_rms = build_support::value_rms();
    let stationary_rope = build_support::stationary_rope();
    assert!(
        !(stationary_rope && (bounded_swiglu || value_rms)) || build_support::full_kda_backward(),
        "stationary RoPE with numerical components requires the complete full-credit Engram/mHC graph"
    );
    fs::write(
        out.join("position_config.rs"),
        format!("pub const STATIONARY_ROPE: bool = {stationary_rope};\n"),
    )
    .expect("write positional graph configuration");
    assert!(
        !(bounded_swiglu && value_rms) || build_support::full_kda_backward(),
        "bounded SwiGLU with value RMS requires the complete full-credit Engram/mHC graph"
    );
    fs::write(
        out.join("value_config.rs"),
        format!("pub const VALUE_RMS:bool={value_rms};\n"),
    )
    .expect("write value graph configuration");
    fs::write(out.join("mlp_config.rs"), format!(
        "pub const BOUNDED_SWIGLU: bool = {bounded_swiglu};\npub const ADAPTIVE_RELU2: bool = {adaptive_relu2};\npub const FEATURE_TILES: u32 = {};\npub const ACTIVE_FEATURE_TILES: u32 = {};\n",
        if bounded_swiglu { 43 } else { 64 }, if bounded_swiglu { 32 } else { 48 },
    )).expect("write MLP graph configuration");
    let activation_fan_in = env_usize("GPT2_N_EMBD")
        .or_else(|| baseline.usize("GPT2_N_EMBD"))
        .unwrap_or(2048);
    // The actual initializer samples 0 (p=1/4), +/-0.01 (p=3/4).
    // Symmetry gives E[ReLU(Z)^2]=Var(Z)/2 for normalized input.
    let activation_mean = activation_fan_in as f32 * 0.000_037_5;
    fs::write(
        out.join("activation_config.rs"),
        format!("pub const RELU2_INITIAL_MEAN:f32={activation_mean:?};\n"),
    )
    .expect("write activation config");
    fs::write(
        out.join("optimizer_config.rs"),
        format!(
            "pub const MUON_COOPERATIVE_BLOCKS: usize = {cooperative_blocks};\n\
             pub const MUON_MATRIX_PHASES: usize = {matrix_phases};\n"
        ),
    )
    .expect("failed to write generated optimizer config");
}
