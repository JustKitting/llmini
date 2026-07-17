use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::layer_norm_backward::{
    LayerNormBackwardInputAddAmaxArgs, LayerNormBackwardInputAddArgs,
    LayerNormBackwardInputAmaxArgs, LayerNormBackwardInputArgs, LayerNormBackwardModule,
    LayerNormBackwardParamArgs,
};

use crate::GPT2_EMBEDDING_DIM;
use crate::{LayerNormGrads, LayerNormSaved, LayerNormTensors};

pub struct Gpt2LayerNormBackwardInputArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub output_scale: f32,
}

pub struct Gpt2LayerNormBackwardInputAddArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub direct: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub output_scale: f32,
}

pub struct Gpt2LayerNormBackwardInputAmaxArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub chunk_amax: &'out mut DeviceBuffer<f32>,
    pub output_scale: f32,
}

pub struct Gpt2LayerNormBackwardInputAddAmaxArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub direct: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub chunk_amax: &'out mut DeviceBuffer<f32>,
    pub output_scale: f32,
}

pub struct Gpt2LayerNormBackwardParamArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub d_weight: &'out mut DeviceBuffer<f32>,
    pub d_bias: &'out mut DeviceBuffer<f32>,
    pub output_scale: f32,
}

pub struct Gpt2LayerNormBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub grads: LayerNormGrads<'out>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub output_scale: f32,
}

pub struct Gpt2LayerNormBackwardAddArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub grads: LayerNormGrads<'out>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub direct: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub output_scale: f32,
}

pub struct Gpt2LayerNormBackwardAmaxArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub grads: LayerNormGrads<'out>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub chunk_amax: &'out mut DeviceBuffer<f32>,
    pub output_scale: f32,
}

pub struct Gpt2LayerNormBackwardAddAmaxArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub grads: LayerNormGrads<'out>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub direct: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub chunk_amax: &'out mut DeviceBuffer<f32>,
    pub output_scale: f32,
}

pub fn layer_norm_backward_input(
    args: Gpt2LayerNormBackwardInputArgs<'_, '_>,
) -> Result<(), DriverError> {
    args.module.backward_input(LayerNormBackwardInputArgs {
        stream: args.stream,
        residual: args.saved.residual,
        d_normalized: args.d_normalized,
        mean: args.saved.mean,
        inv_std: args.saved.inv_std,
        weight: args.weights.weight,
        d_residual: args.d_residual,
        output_scale: args.output_scale,
        row_count: args.saved.row_count,
        embedding_dim: GPT2_EMBEDDING_DIM,
    })
}

pub fn layer_norm_backward_input_add(
    args: Gpt2LayerNormBackwardInputAddArgs<'_, '_>,
) -> Result<(), DriverError> {
    args.module
        .backward_input_add(LayerNormBackwardInputAddArgs {
            stream: args.stream,
            residual: args.saved.residual,
            d_normalized: args.d_normalized,
            mean: args.saved.mean,
            inv_std: args.saved.inv_std,
            weight: args.weights.weight,
            direct: args.direct,
            d_residual: args.d_residual,
            output_scale: args.output_scale,
            row_count: args.saved.row_count,
            embedding_dim: GPT2_EMBEDDING_DIM,
        })
}

pub fn layer_norm_backward_input_amax(
    args: Gpt2LayerNormBackwardInputAmaxArgs<'_, '_>,
) -> Result<u32, DriverError> {
    args.module
        .backward_input_amax(LayerNormBackwardInputAmaxArgs {
            stream: args.stream,
            residual: args.saved.residual,
            d_normalized: args.d_normalized,
            mean: args.saved.mean,
            inv_std: args.saved.inv_std,
            weight: args.weights.weight,
            d_residual: args.d_residual,
            chunk_amax: args.chunk_amax,
            output_scale: args.output_scale,
            row_count: args.saved.row_count,
            embedding_dim: GPT2_EMBEDDING_DIM,
        })
}

pub fn layer_norm_backward_input_add_amax(
    args: Gpt2LayerNormBackwardInputAddAmaxArgs<'_, '_>,
) -> Result<u32, DriverError> {
    args.module
        .backward_input_add_amax(LayerNormBackwardInputAddAmaxArgs {
            stream: args.stream,
            residual: args.saved.residual,
            d_normalized: args.d_normalized,
            mean: args.saved.mean,
            inv_std: args.saved.inv_std,
            weight: args.weights.weight,
            direct: args.direct,
            d_residual: args.d_residual,
            chunk_amax: args.chunk_amax,
            output_scale: args.output_scale,
            row_count: args.saved.row_count,
            embedding_dim: GPT2_EMBEDDING_DIM,
        })
}

pub fn layer_norm_backward_params(
    args: Gpt2LayerNormBackwardParamArgs<'_, '_>,
) -> Result<(), DriverError> {
    args.module.backward_params(LayerNormBackwardParamArgs {
        stream: args.stream,
        residual: args.saved.residual,
        d_normalized: args.d_normalized,
        mean: args.saved.mean,
        inv_std: args.saved.inv_std,
        d_weight: args.d_weight,
        d_bias: args.d_bias,
        output_scale: args.output_scale,
        row_count: args.saved.row_count,
        embedding_dim: GPT2_EMBEDDING_DIM,
    })
}

pub fn layer_norm_backward(args: Gpt2LayerNormBackwardArgs<'_, '_>) -> Result<(), DriverError> {
    let grads = args.grads;

    layer_norm_backward_params(Gpt2LayerNormBackwardParamArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        d_normalized: args.d_normalized,
        d_weight: grads.d_weight,
        d_bias: grads.d_bias,
        output_scale: args.output_scale,
    })?;
    layer_norm_backward_input(Gpt2LayerNormBackwardInputArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        weights: args.weights,
        d_normalized: args.d_normalized,
        d_residual: args.d_residual,
        output_scale: args.output_scale,
    })
}

pub fn layer_norm_backward_amax(
    args: Gpt2LayerNormBackwardAmaxArgs<'_, '_>,
) -> Result<u32, DriverError> {
    let grads = args.grads;

    layer_norm_backward_params(Gpt2LayerNormBackwardParamArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        d_normalized: args.d_normalized,
        d_weight: grads.d_weight,
        d_bias: grads.d_bias,
        output_scale: args.output_scale,
    })?;
    layer_norm_backward_input_amax(Gpt2LayerNormBackwardInputAmaxArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        weights: args.weights,
        d_normalized: args.d_normalized,
        d_residual: args.d_residual,
        chunk_amax: args.chunk_amax,
        output_scale: args.output_scale,
    })
}

pub fn layer_norm_backward_add(
    args: Gpt2LayerNormBackwardAddArgs<'_, '_>,
) -> Result<(), DriverError> {
    let grads = args.grads;

    layer_norm_backward_params(Gpt2LayerNormBackwardParamArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        d_normalized: args.d_normalized,
        d_weight: grads.d_weight,
        d_bias: grads.d_bias,
        output_scale: args.output_scale,
    })?;
    layer_norm_backward_input_add(Gpt2LayerNormBackwardInputAddArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        weights: args.weights,
        d_normalized: args.d_normalized,
        direct: args.direct,
        d_residual: args.d_residual,
        output_scale: args.output_scale,
    })
}

pub fn layer_norm_backward_add_amax(
    args: Gpt2LayerNormBackwardAddAmaxArgs<'_, '_>,
) -> Result<u32, DriverError> {
    let grads = args.grads;

    layer_norm_backward_params(Gpt2LayerNormBackwardParamArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        d_normalized: args.d_normalized,
        d_weight: grads.d_weight,
        d_bias: grads.d_bias,
        output_scale: args.output_scale,
    })?;
    layer_norm_backward_input_add_amax(Gpt2LayerNormBackwardInputAddAmaxArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        weights: args.weights,
        d_normalized: args.d_normalized,
        direct: args.direct,
        d_residual: args.d_residual,
        chunk_amax: args.chunk_amax,
        output_scale: args.output_scale,
    })
}
