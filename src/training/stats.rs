use super::diagnostics;

#[derive(Default)]
pub struct TrainStats {
    pub tokens: usize,
    pub logits: usize,
    pub finite: bool,
    pub nonzero: bool,
    pub loss: f32,
    pub forward_ms: f64,
    pub backward_enqueue_ms: f64,
    pub loss_host_wait_ms: f64,
    pub optimizer_ms: f64,
    pub optimizer: OptimizerTrace,
    pub diagnostics: Option<diagnostics::TrainingDiagnostics>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OptimizerTrace {
    pub embedding_lookup_ms: f64,
    pub token_embedding_ms: f64,
    pub final_norm_ms: f64,
    pub blocks_ms: f64,
    pub muon_ms: f64,
    pub kda_clip_ms: f64,
    pub adam_ms: f64,
    pub adam_lr: f32,
    pub muon_lr: f32,
    pub grad_norm: f32,
    pub update_skipped: bool,
    pub skip_loss_spike: bool,
    pub skip_grad_norm_spike: bool,
    pub skip_non_finite: bool,
}
