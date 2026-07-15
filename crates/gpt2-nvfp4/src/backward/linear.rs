use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::{
    LinearBackwardInputTranspose, LinearBackwardModule, LinearBackwardMsEdenArgs,
    LinearBackwardMsEdenScratch, LinearBackwardWeightTranspose,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

pub(super) struct LinearBackwardCall<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LinearBackwardModule,
    pub quant: &'a Nvfp4QuantModule,
    pub e: &'a DeviceBuffer<f32>,
    pub weight_t: LinearBackwardWeightTranspose<'a>,
    pub input_t: LinearBackwardInputTranspose<'a>,
    pub scratch: LinearBackwardMsEdenScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: Option<&'out mut DeviceBuffer<f32>>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
    pub precomputed_e_amax_chunks: Option<u32>,
}

pub(super) struct RowwiseLinearBackwardPass<'a, 'scratch, 'out> {
    pub e: &'a DeviceBuffer<f32>,
    pub saved_input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub scratch: LinearBackwardMsEdenScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
    pub precomputed_e_amax_chunks: Option<u32>,
}

pub(super) fn nvfp4_weight_t(
    weight: Nvfp4FourSixMmaWeightTensor<'_>,
) -> LinearBackwardWeightTranspose<'_> {
    LinearBackwardWeightTranspose::Nvfp4(Nvfp4DeviceTensor::new(
        weight.bytes,
        weight.scales,
        weight.global_scale,
    ))
}

pub(super) fn run_rowwise_linear_backward(
    module: &LinearBackwardModule,
    quant: &Nvfp4QuantModule,
    stream: &CudaStream,
    pass: RowwiseLinearBackwardPass<'_, '_, '_>,
) -> Result<(), DriverError> {
    run_linear_backward(LinearBackwardCall {
        stream,
        module,
        quant,
        e: pass.e,
        weight_t: nvfp4_weight_t(pass.weight),
        input_t: LinearBackwardInputTranspose::RowwiseNvfp4(pass.saved_input),
        scratch: pass.scratch,
        dinput: pass.dinput,
        dweight: pass.dweight,
        dbias: Some(pass.dbias),
        token_count: pass.row_count,
        input_dim: pass.input_dim,
        output_dim: pass.output_dim,
        sign_seed: pass.sign_seed,
        scale_seed: pass.scale_seed,
        precomputed_e_amax_chunks: pass.precomputed_e_amax_chunks,
    })
}

pub(super) fn run_linear_backward(call: LinearBackwardCall<'_, '_, '_>) -> Result<(), DriverError> {
    call.module.backward_ms_eden(LinearBackwardMsEdenArgs {
        stream: call.stream,
        quant_module: call.quant,
        e: call.e,
        weight_t: call.weight_t,
        input_t: call.input_t,
        scratch: call.scratch,
        dinput: call.dinput,
        dweight: call.dweight,
        dbias: call.dbias,
        token_count: call.token_count,
        input_dim: call.input_dim,
        output_dim: call.output_dim,
        sign_seed: call.sign_seed,
        scale_seed: call.scale_seed,
        precomputed_e_amax_chunks: call.precomputed_e_amax_chunks,
    })
}
