use cuda_core::CudaStream;

use super::super::{
    OptimizerTrace, TokenBatch, diagnostics::TrainingDiagnostics, grad_clip::GradientClipBuffers,
    grads::BackwardBuffers, next_latent::NextLatGradBuffers, optimizer::OptimizerScratch,
    optimizer_muon::MuonPointerTables, optimizer_state::OptimizerStateBuffers,
    optimizer_tc_scratch::MuonScratchBuffers, runtime::Runtime, tape::ForwardTapeBuffers,
};
use crate::upload::UploadedModel;

pub struct WeightUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub runtime: &'a Runtime,
    pub batch: &'a TokenBatch,
    pub uploaded: &'a mut UploadedModel,
    pub grads: &'a mut BackwardBuffers,
    pub next_latent_grads: &'a NextLatGradBuffers,
    pub observed_loss: Option<f32>,
    pub scratch: &'a mut OptimizerScratch,
    pub state: &'a mut OptimizerStateBuffers,
    pub muon: &'a mut MuonScratchBuffers,
    pub muon_tables: &'a MuonPointerTables,
    pub tape: &'a ForwardTapeBuffers,
    pub grad_clip: &'a mut GradientClipBuffers,
}

pub struct WeightUpdateResult {
    pub trace: OptimizerTrace,
    pub diagnostics: Option<TrainingDiagnostics>,
}
