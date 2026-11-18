//! Exact NVFP4 affine recomputation and native decay-factor summaries.
use crate::{
    AppResult,
    training::{Trainer, runtime::Runtime},
};
use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{AttentionDims, AttentionProjectionTensors, project_attention_qkv};
use rust_kernels_cuda::{
    kda_decay_audit::DecayAuditModule,
    nvfp4::Nvfp4RowwiseDeviceTensor,
    nvfp4_tma_matmul::{
        scale_layout::sm120_scale_padded_mn_extent, tma::TmaNvfp4DeviceScaleDescriptors,
    },
};
use serde_json::{Value, json};
pub(super) struct Inspector {
    module: DecayAuditModule,
    raw: DeviceBuffer<f32>,
    padded: DeviceBuffer<u8>,
    descriptors: TmaNvfp4DeviceScaleDescriptors,
}
impl Inspector {
    pub(super) fn new(t: &Trainer, rows: usize) -> AppResult<Self> {
        let r = &t.runtime;
        let s = r.stream.as_ref();
        let dims = AttentionDims::new(false);
        let module = r
            .kda_decay_audit_module()
            .map_err(|e| format!("KDA decay inspection requires matching diagnostic PTX: {e}"))?;
        Ok(Self {
            module,
            raw: DeviceBuffer::zeroed(s, rows * dims.qkv_dim as usize)?,
            padded: DeviceBuffer::zeroed(
                s,
                sm120_scale_padded_mn_extent(dims.qkv_dim as usize) * dims.embedding_dim as usize
                    / 2,
            )?,
            descriptors: TmaNvfp4DeviceScaleDescriptors::new(s)?,
        })
    }
    pub(super) fn inspect(
        &mut self,
        r: &Runtime,
        input: Nvfp4RowwiseDeviceTensor<'_>,
        weights: AttentionProjectionTensors<'_>,
        tape: &DeviceBuffer<u16>,
        rows: u32,
        seq: u32,
    ) -> AppResult<Value> {
        let s = r.stream.as_ref();
        let dims = AttentionDims::new(false);
        project_attention_qkv(
            s,
            &r.tma_gemm,
            &r.tma_pad,
            &mut self.descriptors,
            &mut self.padded,
            input,
            weights,
            &mut self.raw,
            rows,
            dims,
        )?;
        let data = self.module.inspect(
            s,
            &self.raw,
            tape,
            rows,
            seq,
            dims.embedding_dim,
            dims.qkv_dim,
        )?;
        let mut counts = [0u64; 8];
        let mut extrema = [0f32; 4];
        for row in data.counts.chunks_exact(8) {
            for i in 0..8 {
                counts[i] += u64::from(row[i]);
            }
        }
        for row in data.extrema.chunks_exact(4) {
            extrema[0] = extrema[0].min(row[0]);
            for i in 1..4 {
                extrema[i] = extrema[i].max(row[i]);
            }
        }
        let token_features = u64::from(rows) * u64::from(dims.embedding_dim);
        let chunk_features = data.cells as u64;
        let pair_features = chunk_features * 2080;
        let fraction = |count: u64, total: u64| count as f64 / total as f64;
        let passed = counts[3] == 0 && counts[4] == 0 && counts[7] == 0;
        Ok(
            json!({"passed":passed,"source":"exact_native_NVFP4_affine_recomputation_from_saved_input_and_checkpoint_weights",
            "raw_decay_coordinate":"original_FP32_before_softplus","saved_FP16_coordinate_mismatches":counts[3],
            "nonfinite_raw_coordinates":counts[4],"nonfinite_prefix_coordinates":counts[7],
            "token_feature_coordinates":token_features,"chunk_feature_coordinates":chunk_features,
            "causal_pair_feature_coordinates":pair_features,"decay_scale":data.decay_scale,"factor_exponent_limit":data.exponent_limit,
            "prefix_capped":counts[0],"prefix_capped_fraction":fraction(counts[0],token_features),
            "exit_key_relative_capped":counts[1],"exit_key_relative_capped_fraction":fraction(counts[1],token_features),
            "terminal_carry_capped":counts[2],"terminal_carry_capped_fraction":fraction(counts[2],chunk_features),
            "potential_readout_factor_overweight_pairs":counts[5],"potential_readout_factor_overweight_pair_fraction":fraction(counts[5],pair_features),
            "terminal_readout_vs_exit_key_argument_mismatches":counts[6],
            "terminal_readout_vs_exit_key_argument_mismatch_fraction":fraction(counts[6],token_features),
            "minimum_cumulative_log_decay":extrema[0],"maximum_prefix_cap_log_gap":extrema[1],
            "maximum_readout_factor_log_overweight":extrema[2],"maximum_terminal_readout_exit_key_argument_gap":extrema[3],
            "limits":"Factor arguments/counts, not a measured output error or loss attribution; invalid-coordinate counts must be zero."}),
        )
    }
}
