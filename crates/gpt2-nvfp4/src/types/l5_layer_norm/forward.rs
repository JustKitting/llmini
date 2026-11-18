use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::layer_norm::{
    GptLayerNormArgs, GptLayerNormSaveResidualF16Args, GptLayerNormSaveXhatF16Args, LayerNormModule,
};

use super::args::{LayerNormForwardArgs, LayerNormTensors};
use super::weights::LayerNormWeights;
use crate::types::{HiddenStateDevice, LayerNormTape};
use crate::{GPT2_EMBEDDING_DIM, GPT2_LAYER_NORM_EPSILON};

impl LayerNormWeights {
    pub fn input_from_block<'a>(
        module: &'a LayerNormModule,
        tensors: LayerNormTensors<'a>,
        hidden: HiddenStateDevice<'a>,
        output_scale: f32,
    ) -> LayerNormForwardArgs<'a> {
        LayerNormForwardArgs {
            module,
            tensors,
            hidden,
            output_scale,
        }
    }

    pub fn forward<'a>(
        &self,
        args: LayerNormForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let hidden = args.hidden;

        args.module.gpt_layer_norm(GptLayerNormArgs {
            stream: hidden.stream,
            residual: &mut *hidden.residual,
            weight: args.tensors.weight,
            bias: args.tensors.bias,
            normalized: &mut *hidden.normalized,
            normalized_amax: &mut *hidden.normalized_amax,
            mean: &mut *hidden.mean,
            inv_std: &mut *hidden.inv_std,
            row_count: hidden.row_count,
            embedding_dim: GPT2_EMBEDDING_DIM,
            epsilon: GPT2_LAYER_NORM_EPSILON,
            output_scale: args.output_scale,
        })?;

        Ok(hidden)
    }

    pub fn forward_with_tape<'a>(
        &self,
        args: LayerNormForwardArgs<'a>,
        tape: Option<&mut LayerNormTape<'_>>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        if let Some(tape) = tape {
            std::mem::swap(args.hidden.mean, tape.mean);
            std::mem::swap(args.hidden.inv_std, tape.inv_std);
            let hidden = self.forward_save_xhat_f16(args, &mut *tape.residual)?;
            std::mem::swap(hidden.mean, tape.mean);
            std::mem::swap(hidden.inv_std, tape.inv_std);
            Ok(hidden)
        } else {
            self.forward(args)
        }
    }

    pub fn forward_save_residual_f16<'a>(
        &self,
        args: LayerNormForwardArgs<'a>,
        residual_f16: &mut DeviceBuffer<u16>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let hidden = args.hidden;

        args.module
            .gpt_layer_norm_save_residual_f16(GptLayerNormSaveResidualF16Args {
                stream: hidden.stream,
                residual: &mut *hidden.residual,
                weight: args.tensors.weight,
                bias: args.tensors.bias,
                normalized: &mut *hidden.normalized,
                normalized_amax: &mut *hidden.normalized_amax,
                mean: &mut *hidden.mean,
                inv_std: &mut *hidden.inv_std,
                residual_f16,
                row_count: hidden.row_count,
                embedding_dim: GPT2_EMBEDDING_DIM,
                epsilon: GPT2_LAYER_NORM_EPSILON,
                output_scale: args.output_scale,
            })?;

        Ok(hidden)
    }

    pub fn forward_save_xhat_f16<'a>(
        &self,
        args: LayerNormForwardArgs<'a>,
        xhat_f16: &mut DeviceBuffer<u16>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let hidden = args.hidden;

        args.module
            .gpt_layer_norm_save_xhat_f16(GptLayerNormSaveXhatF16Args {
                stream: hidden.stream,
                residual: &mut *hidden.residual,
                weight: args.tensors.weight,
                bias: args.tensors.bias,
                normalized: &mut *hidden.normalized,
                normalized_amax: &mut *hidden.normalized_amax,
                mean: &mut *hidden.mean,
                inv_std: &mut *hidden.inv_std,
                xhat_f16,
                row_count: hidden.row_count,
                embedding_dim: GPT2_EMBEDDING_DIM,
                epsilon: GPT2_LAYER_NORM_EPSILON,
                output_scale: args.output_scale,
            })?;

        Ok(hidden)
    }
}
