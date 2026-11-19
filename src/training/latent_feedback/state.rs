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
    latent_feedback::FUSED_ROW_WIDTH,
    linear_backward::{LinearBackwardInputTranspose, LinearBackwardMsEdenWeightArgs},
    nvfp4::Nvfp4RowwiseDecodeTransposeArgs,
    nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors,
    residual::ResidualGradAccumulateArgs,
};

pub(in crate::training) struct Feedback {
    pub state: MuonState,
    pub grad: DeviceBuffer<f32>,
    unpacked_input: Option<UnpackedInput>,
    pub input_q: RowwiseNvfp4Buffers,
    pub projected: DeviceBuffer<f32>,
    fusion_error: DeviceBuffer<f32>,
    tma: TmaNvfp4DeviceScaleDescriptors,
    linear: LinearScratch,
    pub module: rust_kernels_cuda::latent_feedback::kernels::LoadedModule,
    pub active: bool,
    pub n: u32,
    next_training_attempt: u64,
    rng: Gpt2Rng,
}

struct UnpackedInput {
    normalized: DeviceBuffer<f32>,
    row_amax: DeviceBuffer<f32>,
}

impl Feedback {
    #[cfg(test)]
    pub(in crate::training) fn integrity_metadata(&self) -> serde_json::Value {
        // Enumerate the object so a new field cannot silently escape this audit.
        let Self {
            state: _,
            grad: _,
            unpacked_input: _,
            input_q: _,
            projected: _,
            fusion_error: _,
            tma: _,
            linear: _,
            module: _,
            active,
            n,
            next_training_attempt,
            rng,
        } = self;
        serde_json::json!({"active":active,"rows":n,"next_training_attempt":next_training_attempt,"rng":format!("{rng:?}")})
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
            unpacked_input: if d == FUSED_ROW_WIDTH {
                None
            } else {
                Some(UnpackedInput {
                    normalized: DeviceBuffer::zeroed(s, total)?,
                    row_amax: DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS)?,
                })
            },
            input_q: RowwiseNvfp4Buffers::gpt2_rows(s, total)?,
            projected: DeviceBuffer::zeroed(s, total)?,
            fusion_error: DeviceBuffer::zeroed(s, total)?,
            tma: TmaNvfp4DeviceScaleDescriptors::new(s)?,
            linear: LinearScratch::new(s, d, d)?,
            module: r.latent_feedback_module()?,
            active: false,
            n: 0,
            next_training_attempt: 0,
            rng: Gpt2Rng::new(0x4c4f524d6a910735),
        })
    }

    pub(in crate::training) fn next_training_passes(&mut self, seed: u64) -> usize {
        super::sample_training_passes(seed, &mut self.next_training_attempt)
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
    pub fn prepare_input(
        &mut self,
        r: &Runtime,
        h: &HiddenStateDevice<'_>,
    ) -> Result<(), DriverError> {
        let n = h.row_count;
        let stream = r.stream.as_ref();
        let launch = LaunchConfig {
            grid_dim: (n, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Consume the preceding residual here, before the next embedding
        // overwrites it. Only the packed input is needed by the projection.
        let mut packed = self.input_q.scratch();
        if let Some(scratch) = &mut self.unpacked_input {
            self.module.latent_feedback_shifted_rms_kernel(
                stream,
                launch,
                h.residual,
                &mut scratch.normalized,
                n,
                GPT2_N_EMBD as u32,
                h.seq_len,
                EPSILON,
            )?;
            packed.quantize_row_amax(
                &r.quant,
                stream,
                &scratch.normalized,
                &mut scratch.row_amax,
                n,
                GPT2_N_EMBD as u32,
            )
        } else {
            self.module.latent_feedback_shifted_rms_pack_kernel(
                stream,
                launch,
                h.residual,
                packed.bytes,
                packed.scales,
                packed.global_scales,
                n,
                h.seq_len,
                EPSILON,
            )
        }
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
    /// Diagnostics read the same represented values as the fusion GEMM.
    pub(in crate::training) fn decoded_input_transpose(
        &self,
        r: &Runtime,
        rows: usize,
    ) -> Result<Vec<f32>, DriverError> {
        let stream = r.stream.as_ref();
        let mut decoded = DeviceBuffer::zeroed(stream, rows * GPT2_N_EMBD)?;
        r.decode.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
            stream,
            input: self.input_q.rowwise(),
            output: &mut decoded,
            rows: rows as u32,
            cols: GPT2_N_EMBD as u32,
        })?;
        decoded.to_host_vec(stream)
    }

    pub fn backward(
        &mut self,
        r: &Runtime,
        e: &DeviceBuffer<f32>,
        seq_len: usize,
    ) -> Result<(), DriverError> {
        if !self.active {
            return Ok(());
        }
        let stream = r.stream.as_ref();
        let rows = self.n as usize;
        let row_bytes = GPT2_N_EMBD * size_of::<f32>();
        if seq_len == 0
            || !rows.is_multiple_of(seq_len)
            || e.len() < rows * GPT2_N_EMBD
            || self.fusion_error.len() < rows * GPT2_N_EMBD
        {
            return Err(DriverError(
                cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        // The fusion branch is absent at each sequence start. Mask its
        // upstream errors before MS-EDEN mixes token lanes and derives amax.
        // The direct embedding residual gradient still uses the original e.
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                self.fusion_error.cu_deviceptr(),
                e.cu_deviceptr(),
                rows * row_bytes,
                stream.cu_stream(),
            )?;
            for row in (0..rows).step_by(seq_len) {
                cuda_core::memory::memset_d8_async(
                    self.fusion_error.cu_deviceptr() + (row * row_bytes) as u64,
                    0,
                    row_bytes,
                    stream.cu_stream(),
                )?;
            }
        }
        r.linear
            .backward_ms_eden_weight(LinearBackwardMsEdenWeightArgs {
                stream,
                quant_module: &r.quant,
                e: &self.fusion_error,
                input_t: LinearBackwardInputTranspose::RowwiseNvfp4(self.input_q.rowwise()),
                scratch: self.linear.parts(),
                dweight: &mut self.grad,
                token_count: self.n,
                input_dim: GPT2_N_EMBD as u32,
                output_dim: GPT2_N_EMBD as u32,
                sign_seed: self.rng.next_u32(),
                scale_seed: self.rng.next_u32(),
            })
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
