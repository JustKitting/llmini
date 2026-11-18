use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionTcScratch};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tma_matmul::launcher::Nvfp4GemmModule;
use rust_kernels_cuda::nvfp4_tma_matmul::pad::TmaMatrixPadModule;
use rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors;

use super::tape::AttentionForwardTape;
use crate::types::{HiddenStateDevice, HiddenStateNvfp4};

/// Optional independent native mixed-value replay observer. Production uses None.
pub trait ValueProjectionObserver {
    fn observe(
        &self,
        block: usize,
        stream: &cuda_core::CudaStream,
        qkv: &DeviceBuffer<f32>,
        rows: usize,
        width: usize,
        stride: usize,
        recomputed: bool,
    ) -> Result<(), cuda_core::DriverError>;

    /// Zero-update verification hook; production observers are absent.
    fn represented_point(
        &self,
        _block: usize,
        _stream: &cuda_core::CudaStream,
        _raw: &DeviceBuffer<f32>,
        _stored: &DeviceBuffer<u16>,
        _rows: usize,
        _width: usize,
        _stride: usize,
        _kda: bool,
    ) -> Result<(), cuda_core::DriverError> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct AttentionProjectionTensors<'a> {
    pub qkv_weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub qkv_weight_device: Nvfp4DeviceTensor<'a>,
    pub qkv_bias: Nvfp4DeviceTensor<'a>,
    pub qk_scale: Nvfp4DeviceTensor<'a>,
    pub xsa_alphas: Nvfp4DeviceTensor<'a>,
    pub xsa_alpha_offset: u32,
    pub c_proj_weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub c_proj_weight_device: Nvfp4DeviceTensor<'a>,
    pub c_proj_bias: Nvfp4DeviceTensor<'a>,
}

pub struct AttentionForwardArgs<'a, 'scratch> {
    pub value_rms: &'a rust_kernels_cuda::value_rms::ValueRmsModule,
    pub value_projection_observer: Option<&'a dyn ValueProjectionObserver>,
    pub block_index: usize,
    pub use_full_attention: bool,
    pub module: &'a AttentionModule,
    pub tc_module: &'a F16TcMatmulModule,
    pub tma_module: &'a Nvfp4GemmModule,
    pub tma_pad: &'a TmaMatrixPadModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub input_nvfp4: HiddenStateNvfp4<'scratch>,
    pub tc_scratch: CausalAttentionTcScratch<'scratch>,
    pub tma_descriptors: &'scratch mut TmaNvfp4DeviceScaleDescriptors,
    pub tma_weight_bytes_padded: &'scratch mut DeviceBuffer<u8>,
    pub projections: AttentionProjectionTensors<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub value_residual: &'scratch mut DeviceBuffer<f32>,
    pub attention_log_sum_exp: &'scratch mut DeviceBuffer<f32>,
    pub input_prequantized: bool,
    pub hidden: HiddenStateDevice<'a>,
    pub tape: Option<AttentionForwardTape<'scratch>>,
}
