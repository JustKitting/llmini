use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::canon::CanonBackwardArgs;
use rust_kernels_cuda::canon::CanonModule;
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::residual::ResidualBackwardModule;

use super::layer_norm::{Gpt2LayerNormBackwardAddAmaxArgs, layer_norm_backward_add_amax};
use super::mlp::{
    MlpBackwardArgs, MlpBackwardGrads, MlpBackwardModules, MlpBackwardScratch, MlpBackwardSeeds,
    backward as mlp_backward,
};
use crate::types::{BlockBackwardGrads, BlockForwardSaved};
use crate::{CanonTensors, LayerNormTensors, MlpProjectionTensors};

#[derive(Clone, Copy)]
pub struct BlockMlpBackwardModules<'a> {
    pub residual: &'a ResidualBackwardModule,
    pub layer_norm: &'a LayerNormBackwardModule,
    pub canon: &'a CanonModule,
    pub mlp: MlpBackwardModules<'a>,
}

pub struct BlockMlpBackwardArgs<'a, 'scratch, 'out> {
    pub block_index: usize,
    pub stream: &'a CudaStream,
    pub modules: BlockMlpBackwardModules<'a>,
    pub saved: BlockForwardSaved<'a>,
    pub ln_2: LayerNormTensors<'a>,
    pub canon_c: CanonTensors<'a>,
    pub mlp_projections: MlpProjectionTensors<'a>,
    pub d_residual_out: &'scratch DeviceBuffer<f32>,
    pub precomputed_d_residual_amax_chunks: Option<u32>,
    pub d_residual_after_attention: &'scratch mut DeviceBuffer<f32>,
    pub d_residual_after_attention_chunk_amax: &'scratch mut DeviceBuffer<f32>,
    pub d_hidden: &'scratch mut DeviceBuffer<f32>,
    pub d_mlp_up: &'scratch mut DeviceBuffer<f32>,
    pub grads: BlockBackwardGrads<'out>,
    pub scratch: MlpBackwardScratch<'scratch>,
    pub seeds: MlpBackwardSeeds,
}

pub fn mlp_side_backward(args: BlockMlpBackwardArgs<'_, '_, '_>) -> Result<u32, DriverError> {
    let BlockMlpBackwardArgs {
        block_index,
        stream,
        modules,
        saved,
        ln_2,
        canon_c,
        mlp_projections,
        d_residual_out,
        precomputed_d_residual_amax_chunks,
        d_residual_after_attention,
        d_residual_after_attention_chunk_amax,
        d_hidden,
        d_mlp_up,
        grads,
        scratch,
        seeds,
    } = args;
    let BlockBackwardGrads {
        ln_2: mut ln_2_grads,
        d_canon_c_weight,
        d_mlp_c_fc_weight,
        d_mlp_c_fc_bias,
        d_mlp_c_proj_weight,
        d_mlp_c_proj_bias,
        ..
    } = grads;
    mlp_backward(MlpBackwardArgs {
        block_index,
        stream,
        modules: modules.mlp,
        saved,
        projections: mlp_projections,
        d_residual_out,
        precomputed_d_residual_amax_chunks,
        grads: MlpBackwardGrads {
            d_mlp_up: &mut *d_mlp_up,
            d_ln_2_normalized: &mut *d_hidden,
            d_c_proj_weight: d_mlp_c_proj_weight,
            d_c_proj_bias: d_mlp_c_proj_bias,
            d_c_fc_weight: d_mlp_c_fc_weight,
            d_c_fc_bias: d_mlp_c_fc_bias,
        },
        scratch,
        seeds,
    })?;

    let d_ln_2_normalized = if crate::canon_ac_enabled() {
        modules.canon.backward(CanonBackwardArgs {
            stream,
            residual_f16: saved.ln_2.residual,
            mean: saved.ln_2.mean,
            inv_std: saved.ln_2.inv_std,
            norm_weight: ln_2.weight,
            norm_bias: ln_2.bias,
            canon_weight: canon_c.weight,
            d_output: &*d_hidden,
            d_input: &mut *d_mlp_up,
            d_weight: d_canon_c_weight,
            row_count: saved.row_count,
            seq_len: saved.seq_len,
            width: crate::GPT2_EMBEDDING_DIM,
            norm_output_scale: crate::layer_norm_scale(block_index),
        })?;
        &*d_mlp_up
    } else {
        &*d_hidden
    };

    layer_norm_backward_add_amax(Gpt2LayerNormBackwardAddAmaxArgs {
        stream,
        module: modules.layer_norm,
        weights: ln_2,
        saved: saved.ln_2,
        grads: ln_2_grads.reborrow(),
        d_normalized: d_ln_2_normalized,
        direct: d_residual_out,
        d_residual: d_residual_after_attention,
        chunk_amax: d_residual_after_attention_chunk_amax,
        output_scale: crate::layer_norm_scale(block_index),
    })
}
