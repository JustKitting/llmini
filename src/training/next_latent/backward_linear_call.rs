use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::{
    LinearBackwardInputTranspose, LinearBackwardModule, LinearBackwardMsEdenArgs,
    LinearBackwardWeightTranspose,
};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use crate::upload::UploadedLinear;
use gpt2_nvfp4::LinearScratch;

pub(super) struct LinearCall<'a, 'scratch, 'out> {
    pub linear: &'a LinearBackwardModule,
    pub quant: &'a Nvfp4QuantModule,
    pub stream: &'a CudaStream,
    pub e: &'a DeviceBuffer<f32>,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: &'a UploadedLinear,
    pub scratch: &'scratch mut LinearScratch,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub(super) fn run_linear(call: LinearCall<'_, '_, '_>) -> Result<(), DriverError> {
    let scratch = call.scratch.parts();
    call.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
        stream: call.stream,
        quant_module: call.quant,
        e: call.e,
        weight_t: LinearBackwardWeightTranspose::Nvfp4(call.weight.weight.device()),
        input_t: LinearBackwardInputTranspose::RowwiseNvfp4(call.input),
        scratch,
        dinput: call.dinput,
        dweight: call.dweight,
        dbias: Some(call.dbias),
        token_count: call.row_count,
        input_dim: call.input_dim,
        output_dim: call.output_dim,
        sign_seed: call.sign_seed,
        scale_seed: call.scale_seed,
        precomputed_e_amax_chunks: None,
    })
}
