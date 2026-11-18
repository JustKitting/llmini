use cuda_core::CudaStream;

use crate::nvfp4_quant::Nvfp4QuantModule;
use crate::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;

pub(in crate::linear_backward::ms_eden) struct QuantizeContext<'a> {
    pub(in crate::linear_backward::ms_eden::quantize) module: &'a Nvfp4QuantModule,
    pub(in crate::linear_backward::ms_eden::quantize) stream: &'a CudaStream,
    pub(in crate::linear_backward::ms_eden::quantize) token_count: u32,
    pub(in crate::linear_backward::ms_eden::quantize) input_dim: u32,
    pub(in crate::linear_backward::ms_eden::quantize) output_dim: u32,
    pub(in crate::linear_backward::ms_eden::quantize) output_k: u32,
    pub(in crate::linear_backward::ms_eden::quantize) token_k: u32,
    pub(in crate::linear_backward::ms_eden::quantize) sign_seed: u32,
    pub(in crate::linear_backward::ms_eden::quantize) scale_seed: u32,
}

impl<'a> QuantizeContext<'a> {
    pub(in crate::linear_backward::ms_eden) fn new(
        stream: &'a CudaStream,
        module: &'a Nvfp4QuantModule,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        sign_seed: u32,
        scale_seed: u32,
    ) -> Self {
        Self {
            module,
            stream,
            token_count,
            input_dim,
            output_dim,
            output_k: nvfp4_tc_matmul_padded_k(output_dim),
            token_k: nvfp4_tc_matmul_padded_k(token_count),
            sign_seed,
            scale_seed,
        }
    }
}
