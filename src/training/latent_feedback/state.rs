use super::{EPSILON, Weights, zero};
use crate::{
    AppResult,
    training::{
        grad_clip::GradientClipBuffers,
        optimizer::OptimizerScratch,
        optimizer_muon::MuonPointerTables,
        optimizer_state::{MuonState, StateInit},
        runtime::Runtime,
        schedule_free::Materializer,
    },
};
use cuda_core::{DeviceBuffer, DriverError, LaunchConfig};
use gpt2_nvfp4::{
    GPT2_N_EMBD, GPT2_TOKEN_ROWS, Gpt2Rng, HiddenStateDevice, LinearScratch, RowwiseNvfp4Buffers,
};
use rust_kernels_cuda::{
    linear_backward::{
        LinearBackwardInputTranspose, LinearBackwardMsEdenArgs, LinearBackwardWeightTranspose,
    },
    nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors,
    residual::ResidualGradAccumulateArgs,
};

pub(in crate::training) struct Feedback {
    pub state: MuonState,
    pub grad: DeviceBuffer<f32>,
    pub top: DeviceBuffer<f32>,
    pub input: DeviceBuffer<f32>,
    pub input_q: RowwiseNvfp4Buffers,
    pub projected: DeviceBuffer<f32>,
    pub detached_gradient: DeviceBuffer<f32>,
    row_amax: DeviceBuffer<f32>,
    tma: TmaNvfp4DeviceScaleDescriptors,
    linear: LinearScratch,
    pub module: rust_kernels_cuda::latent_feedback::kernels::LoadedModule,
    pub active: bool,
    pub n: u32,
    rng: Gpt2Rng,
}
impl Feedback {
    #[cfg(test)]
    pub(in crate::training) fn integrity_metadata(&self) -> serde_json::Value {
        // Enumerate the object so a new field cannot silently escape this audit.
        let Self {
            state: _,
            grad: _,
            top: _,
            input: _,
            input_q: _,
            projected: _,
            detached_gradient: _,
            row_amax: _,
            tma: _,
            linear: _,
            module: _,
            active,
            n,
            rng,
        } = self;
        serde_json::json!({"active":active,"rows":n,"rng":format!("{rng:?}")})
    }

    pub fn new(r: &Runtime, w: &Weights) -> Result<Self, DriverError> {
        let s = r.stream.as_ref();
        let d = GPT2_N_EMBD;
        let total = GPT2_TOKEN_ROWS * d;
        Ok(Self {
            state: MuonState::new(
                StateInit::new(
                    s,
                    &r.decode,
                    &r.optimizer,
                    crate::training::symexp_lin::beta(),
                ),
                &w.weight,
                d,
            )?,
            grad: DeviceBuffer::zeroed(s, d * d)?,
            top: DeviceBuffer::zeroed(s, total)?,
            input: DeviceBuffer::zeroed(s, total)?,
            input_q: RowwiseNvfp4Buffers::gpt2_rows(s, total)?,
            projected: DeviceBuffer::zeroed(s, total)?,
            detached_gradient: DeviceBuffer::zeroed(s, total)?,
            row_amax: DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS)?,
            tma: TmaNvfp4DeviceScaleDescriptors::new(s)?,
            linear: LinearScratch::new(s, d, d)?,
            module: r.latent_feedback_module()?,
            active: false,
            n: 0,
            rng: Gpt2Rng::new(0x4c4f524d6a910735),
        })
    }
    pub fn named_gradients(&self) -> Vec<(&str, &DeviceBuffer<f32>)> {
        vec![
            ("latent_feedback.fusion.weight", &self.grad),
            (
                "latent_feedback.fusion.weight.exponential",
                &self.state.symexp_lin.exponential.grad,
            ),
            (
                "latent_feedback.fusion.weight.linear",
                &self.state.symexp_lin.linear.grad,
            ),
            (
                "latent_feedback.fusion.weight.curvature",
                &self.state.symexp_lin.curvature.grad,
            ),
        ]
    }
    pub fn begin(
        &mut self,
        r: &Runtime,
        tables: &mut MuonPointerTables,
        clip: &GradientClipBuffers,
        passes: usize,
        rows: u32,
    ) -> AppResult {
        self.active = passes > 1;
        self.n = rows;
        tables.set_conditional_matrix_enabled(self.grad.cu_deviceptr(), self.active)?;
        let s = r.stream.as_ref();
        for (_, g) in self.named_gradients() {
            zero(s, g)?;
        }
        if crate::training::symexp_lin::learn_scales() {
            use rust_kernels_cuda::nvfp4_tma_matmul::kernels::{TILE_M, TILE_N};
            let i = tables
                .all
                .host_symexp_lin_slots
                .iter()
                .position(|v| v.grad == self.grad.cu_deviceptr())
                .ok_or("feedback matrix missing from native Muon table")?;
            let desc = tables.all.host_symexp_lin_slots[i];
            let len = (desc.cols.div_ceil(TILE_M) * desc.rows.div_ceil(TILE_N)) as usize;
            unsafe {
                cuda_core::memory::memset_d8_async(
                    clip.matrix_sums.cu_deviceptr() + 4 * u64::from(clip.matrix_sum_offsets[i]),
                    0,
                    len * 4,
                    s.cu_stream(),
                )?;
            }
        }
        Ok(())
    }
    pub fn mix(
        &mut self,
        r: &Runtime,
        w: &Weights,
        h: &mut HiddenStateDevice<'_>,
    ) -> Result<(), DriverError> {
        let s = r.stream.as_ref();
        let n = h.row_count;
        let d = GPT2_N_EMBD as u32;
        self.module.latent_feedback_shifted_rms_kernel(
            s,
            LaunchConfig {
                grid_dim: (n, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &self.top,
            &mut self.input,
            n,
            d,
            h.seq_len,
            EPSILON,
        )?;
        self.input_q.scratch().quantize_row_amax(
            &r.quant,
            s,
            &self.input,
            &mut self.row_amax,
            n,
            d,
        )?;
        let input = self.input_q.rowwise();
        r.tma_gemm.prepare_tma_nvfp4_csf_weight_into(
            s,
            input.bytes,
            input.scales,
            &w.weight.bytes,
            &w.weight.scales,
            n,
            d,
            d,
            &mut self.tma,
        )?;
        r.tma_gemm
            .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer(
                s,
                &self.tma,
                &mut self.projected,
                n,
                d,
                d,
                input.global_scales,
                &w.weight.global_scale,
                None,
            )?;
        r.residual.grad_accumulate(ResidualGradAccumulateArgs {
            stream: s,
            branch: &self.projected,
            out: h.residual,
            len: n * d,
        })
    }
    pub fn capture_top(
        &mut self,
        r: &Runtime,
        h: &HiddenStateDevice<'_>,
    ) -> Result<(), DriverError> {
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                self.top.cu_deviceptr(),
                h.residual.cu_deviceptr(),
                h.row_count as usize * GPT2_N_EMBD * 4,
                r.stream.cu_stream(),
            )
        }
    }
    pub fn backward(
        &mut self,
        r: &Runtime,
        w: &Weights,
        e: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if !self.active {
            return Ok(());
        }
        r.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
            stream: r.stream.as_ref(),
            quant_module: &r.quant,
            e,
            weight_t: LinearBackwardWeightTranspose::Nvfp4(w.weight.device()),
            input_t: LinearBackwardInputTranspose::RowwiseNvfp4(self.input_q.rowwise()),
            scratch: self.linear.parts(),
            dinput: &mut self.detached_gradient,
            dweight: &mut self.grad,
            dbias: None,
            token_count: self.n,
            input_dim: GPT2_N_EMBD as u32,
            output_dim: GPT2_N_EMBD as u32,
            sign_seed: self.rng.next_u32(),
            scale_seed: self.rng.next_u32(),
            precomputed_e_amax_chunks: None,
        })
        // The computed input derivative belongs to the detached previous pass and is not accumulated.
    }
    pub fn materialize_training(
        &mut self,
        r: &Runtime,
        w: &mut Weights,
        scratch: &mut OptimizerScratch,
        beta: f32,
    ) -> Result<(), DriverError> {
        Materializer::new(
            r.stream.as_ref(),
            &r.optimizer,
            &r.f32_ops,
            scratch,
            beta,
            crate::training::symexp_lin::beta(),
            false,
            false,
        )
        .muon(&mut w.weight, &self.state)
    }
    pub fn materialize_evaluation(
        &mut self,
        r: &Runtime,
        w: &mut Weights,
        scratch: &mut OptimizerScratch,
    ) -> Result<(), DriverError> {
        Materializer::new(
            r.stream.as_ref(),
            &r.optimizer,
            &r.f32_ops,
            scratch,
            0.0,
            crate::training::symexp_lin::beta(),
            false,
            false,
        )
        .symexp_lin_muon_master(
            &mut w.weight,
            &self.state.x_master,
            &self.state.symexp_lin,
        )
    }
}
