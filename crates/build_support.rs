use std::{env, fs, path::PathBuf};

pub fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|value| value.parse().ok())
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    Relu2,
    AdaptiveRelu2,
    BoundedSwiGlu,
}

pub fn mlp_kind() -> MlpKind {
    match env::var("GPT2_MLP_KIND").as_deref().unwrap_or("relu2") {
        "relu2" => MlpKind::Relu2,
        "adaptive-relu2" => MlpKind::AdaptiveRelu2,
        "bounded-swiglu" => MlpKind::BoundedSwiGlu,
        other => {
            panic!("unsupported GPT2_MLP_KIND={other}; use relu2, adaptive-relu2 or bounded-swiglu")
        }
    }
}

pub fn value_rms() -> bool {
    match env::var("GPT2_VALUE_RMS").as_deref().unwrap_or("0") {
        "0" => false,
        "1" => true,
        other => panic!("unsupported GPT2_VALUE_RMS={other}; use0 or1"),
    }
}

pub fn activation_mae_four_six() -> bool {
    match env::var("GPT2_ACTIVATION_QUANT")
        .as_deref()
        .unwrap_or("six")
    {
        "six" => false,
        "mae-4-6" => true,
        other => panic!("unsupported GPT2_ACTIVATION_QUANT={other}; use six or mae-4-6"),
    }
}

pub fn full_kda_backward() -> bool {
    match env::var("GPT2_KDA_BACKWARD")
        .as_deref()
        .unwrap_or("detached")
    {
        "detached" => false,
        "full" => true,
        other => panic!("unsupported GPT2_KDA_BACKWARD={other}; use detached or full"),
    }
}

pub fn stationary_rope() -> bool {
    match env::var("GPT2_POSITION_KIND")
        .as_deref()
        .unwrap_or("full-rope")
    {
        "full-rope" => false,
        "stationary-rope" => true,
        other => panic!("unsupported GPT2_POSITION_KIND={other}; use full-rope or stationary-rope"),
    }
}

pub fn emit_rerun_metadata(env_names: &[&str]) {
    for name in env_names {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!(
        "cargo:rerun-if-changed={}",
        baseline_path().to_string_lossy()
    );
}

pub struct Baseline {
    text: String,
}

impl Baseline {
    pub fn load() -> Self {
        Self {
            text: fs::read_to_string(baseline_path()).unwrap_or_default(),
        }
    }

    pub fn usize(&self, name: &str) -> Option<usize> {
        self.value(name).and_then(|value| value.parse().ok())
    }

    pub fn value(&self, name: &str) -> Option<&str> {
        self.text.lines().find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == name).then_some(value.trim())
        })
    }
}

fn baseline_path() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"))
        .join("../../notes/sweep_baseline.env")
}
