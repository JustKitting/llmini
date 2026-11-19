use cuda_core::DriverError;
use rust_kernels_cuda::canon::CanonQuantizeArgs;

use super::args::BlockForwardArgs;
use super::weights::Gpt2BlockWeights;
use crate::types::{
    AttentionForwardArgs, AttentionWeights, HiddenStateDevice, LayerNormWeights, MlpForwardArgs,
    MlpScratch, MlpWeights,
};

impl Gpt2BlockWeights {
    pub fn forward<'a, 'scratch>(
        &self,
        args: BlockForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.forward_with_boundaries(args, &mut |_, _, _| Ok(false))
    }

    pub fn forward_with_boundaries<'a, 'scratch>(
        &self,
        args: BlockForwardArgs<'a, 'scratch>,
        boundary: &mut dyn FnMut(
            usize,
            bool,
            &mut HiddenStateDevice<'_>,
        ) -> Result<bool, DriverError>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let qkv = args.qkv;
        let value_residual = args.value_residual;
        let attention_log_sum_exp = args.attention_log_sum_exp;
        let mlp_activation = args.mlp_activation;
        let mlp_route_scores = args.mlp_route_scores;
        let mlp_route_masks = args.mlp_route_masks;
        let mut hidden_nvfp4 = args.hidden_nvfp4;
        let tma_descriptors = args.tma_descriptors;
        let tma_weight_bytes_padded = args.tma_weight_bytes_padded;
        let mut tape = args.tape;
        let layer_norm_scale = crate::layer_norm_scale(args.block_index);
        let mut hidden = args.hidden;
        let branch = args.block_index * 2;
        let isolated_attention = boundary(branch, true, &mut hidden)?;

        let ln_1 = LayerNormWeights::input_from_block(
            args.layer_norm_module,
            args.ln_1,
            hidden,
            layer_norm_scale,
        );
        let hidden = self
            .ln_1
            .forward_with_tape(ln_1, tape.as_mut().map(|tape| &mut tape.ln_1))?;
        let canon_a_enabled = crate::canon_ac_enabled();
        if canon_a_enabled {
            if let Some(tape) = tape.as_mut() {
                tape.qkv_input_nvfp4.swap_storage(&mut hidden_nvfp4);
            }
            args.canon_module.forward_quantized(CanonQuantizeArgs {
                stream: hidden.stream,
                input: &*hidden.normalized,
                weight: args.canon_a.weight,
                amax: &mut *hidden.normalized_amax,
                out_fp4: &mut *hidden_nvfp4.bytes,
                out_scales: &mut *hidden_nvfp4.scales,
                out_global_scale: &mut *hidden_nvfp4.global_scales,
                row_count: hidden.row_count,
                seq_len: hidden.seq_len,
                width: crate::GPT2_EMBEDDING_DIM,
            })?;
        }

        let attention_tape = tape.as_mut().map(|tape| tape.attention_forward());

        let attention_args = AttentionForwardArgs {
            value_rms: args.value_rms,
            value_projection_observer: args.value_projection_observer,
            block_index: args.block_index,
            use_full_attention: args.use_full_attention,
            module: args.attention_module,
            tc_module: args.attention_tc_module,
            tma_module: args.tma_module,
            tma_pad: args.tma_pad,
            quant_module: args.quant_module,
            input_nvfp4: hidden_nvfp4.reborrow(),
            tc_scratch: args.attention_tc_scratch,
            tma_descriptors: &mut *tma_descriptors,
            tma_weight_bytes_padded: &mut *tma_weight_bytes_padded,
            projections: args.projections,
            qkv: &mut *qkv,
            value_residual: &mut *value_residual,
            attention_log_sum_exp: &mut *attention_log_sum_exp,
            input_prequantized: canon_a_enabled,
            hidden,
            tape: attention_tape,
        };
        let mut hidden = if isolated_attention {
            AttentionWeights::forward_branch(attention_args)?
        } else {
            AttentionWeights::forward(attention_args)?
        };
        boundary(branch, false, &mut hidden)?;

        if let Some(tape) = tape.as_mut() {
            tape.save_attention_log_sum_exp(hidden.stream, attention_log_sum_exp)?;
        }

        let isolated_mlp = boundary(branch + 1, true, &mut hidden)?;
        let ln_2 = LayerNormWeights::input_from_block(
            args.layer_norm_module,
            args.ln_2,
            hidden,
            layer_norm_scale,
        );
        let hidden = self
            .ln_2
            .forward_with_tape(ln_2, tape.as_mut().map(|tape| &mut tape.ln_2))?;
        let canon_c_enabled = crate::canon_ac_enabled();
        if canon_c_enabled {
            if let Some(tape) = tape.as_mut() {
                tape.mlp_up_input_nvfp4.swap_storage(&mut hidden_nvfp4);
            }
            args.canon_module.forward_quantized(CanonQuantizeArgs {
                stream: hidden.stream,
                input: &*hidden.normalized,
                weight: args.canon_c.weight,
                amax: &mut *hidden.normalized_amax,
                out_fp4: &mut *hidden_nvfp4.bytes,
                out_scales: &mut *hidden_nvfp4.scales,
                out_global_scale: &mut *hidden_nvfp4.global_scales,
                row_count: hidden.row_count,
                seq_len: hidden.seq_len,
                width: crate::GPT2_EMBEDDING_DIM,
            })?;
        }

        let mlp_tape = tape.as_mut().map(|tape| tape.mlp_forward());

        let mlp_args = MlpForwardArgs {
            projection_observer: args.mlp_projection_observer,
            block_index: args.block_index,
            module: args.mlp_module,
            tma_module: args.tma_module,
            quant_module: args.quant_module,
            scratch: MlpScratch {
                paired: args.mlp_paired,
                input_nvfp4: hidden_nvfp4.reborrow(),
                activation_nvfp4: args.mlp_activation_nvfp4,
                activation: &mut *mlp_activation,
                route_scores: &mut *mlp_route_scores,
                route_masks: &mut *mlp_route_masks,
                tma_descriptors: &mut *tma_descriptors,
            },
            projections: args.mlp,
            input_prequantized: canon_c_enabled,
            hidden,
            tape: mlp_tape,
        };
        let mut hidden = if isolated_mlp {
            MlpWeights::forward_branch(mlp_args)?
        } else {
            MlpWeights::forward(mlp_args)?
        };
        boundary(branch + 1, false, &mut hidden)?;
        Ok(hidden)
    }
}
