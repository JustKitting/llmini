use cuda_core::DriverError;

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
        let qkv = args.qkv;
        let attention_log_sum_exp = args.attention_log_sum_exp;
        let mlp_activation = args.mlp_activation;
        let mut hidden_nvfp4 = args.hidden_nvfp4;
        let tma_descriptors = args.tma_descriptors;
        let tma_input_scale_packed = args.tma_input_scale_packed;
        let tma_wide_input_scale_packed = args.tma_wide_input_scale_packed;
        let tma_weight_scale_packed = args.tma_weight_scale_packed;
        let tma_weight_bytes_padded = args.tma_weight_bytes_padded;
        let mut tape = args.tape;

        let ln_1 =
            LayerNormWeights::input_from_block(args.layer_norm_module, args.ln_1, args.hidden);
        let hidden = self
            .ln_1
            .forward_with_tape(ln_1, tape.as_mut().map(|tape| &mut tape.ln_1))?;

        let attention_tape = tape.as_mut().map(|tape| tape.attention_forward());

        let hidden = AttentionWeights::forward(AttentionForwardArgs {
            use_full_attention: args.use_full_attention,
            module: args.attention_module,
            tc_module: args.attention_tc_module,
            tma_module: args.tma_module,
            tma_scale_pack: args.tma_scale_pack,
            tma_pad: args.tma_pad,
            quant_module: args.quant_module,
            input_nvfp4: hidden_nvfp4.reborrow(),
            tc_scratch: args.attention_tc_scratch,
            tma_descriptors: &mut *tma_descriptors,
            tma_input_scale_packed: &mut *tma_input_scale_packed,
            tma_weight_scale_packed: &mut *tma_weight_scale_packed,
            tma_weight_bytes_padded: &mut *tma_weight_bytes_padded,
            projections: args.projections,
            qkv: &mut *qkv,
            attention_log_sum_exp: &mut *attention_log_sum_exp,
            hidden,
            tape: attention_tape,
        })?;

        if let Some(tape) = tape.as_mut() {
            tape.save_attention_log_sum_exp(hidden.stream, attention_log_sum_exp)?;
        }

        let ln_2 = LayerNormWeights::input_from_block(args.layer_norm_module, args.ln_2, hidden);
        let hidden = self
            .ln_2
            .forward_with_tape(ln_2, tape.as_mut().map(|tape| &mut tape.ln_2))?;

        let mlp_tape = tape.as_mut().map(|tape| tape.mlp_forward());

        let hidden = MlpWeights::forward(MlpForwardArgs {
            module: args.mlp_module,
            tma_module: args.tma_module,
            tma_scale_pack: args.tma_scale_pack,
            quant_module: args.quant_module,
            scratch: MlpScratch {
                input_nvfp4: hidden_nvfp4.reborrow(),
                activation_nvfp4: args.mlp_activation_nvfp4,
                activation: &mut *mlp_activation,
                tma_descriptors: &mut *tma_descriptors,
                tma_input_scale_packed: &mut *tma_input_scale_packed,
                tma_wide_input_scale_packed: &mut *tma_wide_input_scale_packed,
                tma_weight_scale_packed: &mut *tma_weight_scale_packed,
            },
            projections: args.mlp,
            hidden,
            tape: mlp_tape,
        })?;

        Ok(hidden)
    }
}
