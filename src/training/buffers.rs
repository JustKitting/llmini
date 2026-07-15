use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionLogSumExp, GPT2_TOKEN_ROWS, HiddenState, Logits, MlpActivation, QkvActivation,
    RowwiseNvfp4Buffers,
};

use super::device_buffer::zero;
use super::forward_tma::ForwardTmaBuffers;
use super::grad_clip::GradientClipBuffers;
use super::grads::BackwardBuffers;
use super::next_latent::{NextLatBuffers, NextLatGradBuffers, NextLatScratchBuffers};
use super::optimizer::OptimizerScratch;
use super::optimizer_muon::MuonPointerTables;
use super::optimizer_state::OptimizerStateBuffers;
use super::optimizer_tc_scratch::MuonScratchBuffers;
use super::scratch::BackwardScratchBuffers;
use super::tape::ForwardTapeBuffers;
use crate::training::runtime::Runtime;
use crate::upload::UploadedModel;

pub struct TrainBuffers {
    pub residual: DeviceBuffer<f32>,
    pub normalized: DeviceBuffer<f32>,
    pub normalized_amax: DeviceBuffer<f32>,
    pub mean: DeviceBuffer<f32>,
    pub inv_std: DeviceBuffer<f32>,
    pub hidden_nvfp4: RowwiseNvfp4Buffers,
    pub mlp_pre: DeviceBuffer<f32>,
    pub mlp_act: DeviceBuffer<f32>,
    pub mlp_activation_nvfp4: RowwiseNvfp4Buffers,
    pub qkv: DeviceBuffer<f32>,
    pub log_sum_exp: DeviceBuffer<f32>,
    pub logits: DeviceBuffer<f32>,
    pub forward_tma: ForwardTmaBuffers,
    pub next_latent: NextLatBuffers,
    pub next_latent_grads: NextLatGradBuffers,
    pub next_latent_scratch: NextLatScratchBuffers,
    pub tape: ForwardTapeBuffers,
    pub backward: BackwardBuffers,
    pub scratch: BackwardScratchBuffers,
    pub optimizer: OptimizerScratch,
    pub optimizer_state: OptimizerStateBuffers,
    pub muon: MuonScratchBuffers,
    pub muon_tables: MuonPointerTables,
    pub grad_clip: GradientClipBuffers,
}

impl TrainBuffers {
    pub fn new(
        stream: &CudaStream,
        runtime: &Runtime,
        uploaded: &UploadedModel,
    ) -> Result<Self, DriverError> {
        let backward = BackwardBuffers::new(stream)?;
        let next_latent_grads = NextLatGradBuffers::new(stream)?;
        let optimizer_state = OptimizerStateBuffers::new(stream, &runtime.decode, uploaded)?;
        let muon_tables = MuonPointerTables::new(
            stream,
            uploaded,
            &backward,
            &next_latent_grads,
            &optimizer_state,
        )?;
        let grad_clip = GradientClipBuffers::new(stream, &backward, &next_latent_grads)?;

        Ok(Self {
            residual: zero(stream, HiddenState::LEN)?,
            normalized: zero(stream, HiddenState::LEN)?,
            normalized_amax: zero(stream, GPT2_TOKEN_ROWS)?,
            mean: zero(stream, GPT2_TOKEN_ROWS)?,
            inv_std: zero(stream, GPT2_TOKEN_ROWS)?,
            hidden_nvfp4: RowwiseNvfp4Buffers::gpt2_rows(stream, HiddenState::LEN)?,
            mlp_pre: zero(stream, MlpActivation::LEN)?,
            mlp_act: zero(stream, MlpActivation::LEN)?,
            mlp_activation_nvfp4: RowwiseNvfp4Buffers::gpt2_rows(stream, MlpActivation::LEN)?,
            qkv: zero(stream, QkvActivation::LEN)?,
            log_sum_exp: zero(stream, AttentionLogSumExp::LEN)?,
            logits: zero(stream, Logits::LEN)?,
            forward_tma: ForwardTmaBuffers::new(stream)?,
            next_latent: NextLatBuffers::new(stream)?,
            next_latent_grads,
            next_latent_scratch: NextLatScratchBuffers::new(stream)?,
            tape: ForwardTapeBuffers::new(stream)?,
            backward,
            scratch: BackwardScratchBuffers::new(stream)?,
            optimizer: OptimizerScratch::new(stream)?,
            optimizer_state,
            muon: MuonScratchBuffers::new(stream)?,
            muon_tables,
            grad_clip,
        })
    }
}
