use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::lm_head::{LmHeadModule, LmHeadTmaArgs};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

use crate::types::{
    Gpt2ForwardTape, HiddenStateDevice, HiddenStateNvfp4, LayerNormTensors, LayerNormWeights,
};

pub(super) struct FinalForwardArgs<'a, 'w> {
    pub ln_f_weights: &'w LayerNormWeights,
    pub layer_norm_module: &'a LayerNormModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub lm_head_module: &'a LmHeadModule,
    pub lm_head_tma_module: &'a Nvfp4GemmModule,
    pub lm_head_tma_scale_pack: &'a Sm120ScalePackModule,
    pub lm_head_tma_descriptors: &'a mut TmaNvfp4DeviceScaleDescriptors,
    pub lm_head_input_scale_packed: &'a mut DeviceBuffer<u8>,
    pub ln_f: LayerNormTensors<'a>,
    pub hidden_nvfp4: HiddenStateNvfp4<'a>,
    pub lm_head_weight_device: Nvfp4DeviceTensor<'a>,
    pub lm_head_weight_scale_packed: &'a mut DeviceBuffer<u8>,
    pub logits: &'a mut DeviceBuffer<f32>,
    pub tape: Option<Gpt2ForwardTape<'a>>,
    pub hidden: HiddenStateDevice<'a>,
}

pub(super) fn finish_forward<'a>(
    args: FinalForwardArgs<'a, '_>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut hidden_nvfp4 = args.hidden_nvfp4;
    let mut tape = args.tape;
    let ln_f =
        LayerNormWeights::input_from_block(args.layer_norm_module, args.ln_f, args.hidden, 1.0);
    let hidden = args
        .ln_f_weights
        .forward_with_tape(ln_f, tape.as_mut().map(|tape| &mut tape.final_norm))?;

    let input = hidden_nvfp4.quantize_hidden_precomputed(
        args.quant_module,
        &hidden,
        crate::GPT2_EMBEDDING_DIM,
    )?;
    if let Some(tape) = tape.as_mut() {
        tape.lm_head_input_nvfp4.save(hidden.stream, input)?;
    }

    args.lm_head_module.logits_tma(LmHeadTmaArgs {
        stream: hidden.stream,
        tma: args.lm_head_tma_module,
        scale_pack: args.lm_head_tma_scale_pack,
        descriptors: args.lm_head_tma_descriptors,
        input_scale_packed: args.lm_head_input_scale_packed,
        input,
        weight: args.lm_head_weight_device,
        weight_scale_packed: args.lm_head_weight_scale_packed,
        logits: &mut *args.logits,
        token_count: hidden.row_count,
        input_dim: crate::GPT2_EMBEDDING_DIM,
        vocab_size: crate::GPT2_VOCAB_DIM,
    })?;

    Ok(hidden)
}
