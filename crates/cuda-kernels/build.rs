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

    emit_rerun_metadata(&["MUON_COOPERATIVE_BLOCKS", "MUON_MATRIX_PHASES"]);

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    build_nvfp4_config::emit_nvfp4_config(&out);
    fs::write(
        out.join("optimizer_config.rs"),
        format!(
            "pub const MUON_COOPERATIVE_BLOCKS: usize = {cooperative_blocks};\n\
             pub const MUON_MATRIX_PHASES: usize = {matrix_phases};\n"
        ),
    )
    .expect("failed to write generated optimizer config");
}
