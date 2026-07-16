mod amax;
mod args;
mod config;
mod four_six;
mod global_scale;
pub(crate) mod kernels;
mod launcher;
mod ms_eden;
mod ms_eden_fp32_transpose;
mod ms_eden_pair;
mod ms_eden_quartet;
mod ms_eden_transpose;
mod ms_eden_transpose_rowwise;
mod shape;

pub use args::{
    MsEdenDeviceScaleQuantArgs, MsEdenPairDeviceScaleQuantArgs, MsEdenQuantArgs,
    MsEdenTransposeDeviceScaleQuantArgs, Nvfp4QuantArgs, Nvfp4QuantPaddedArgs,
    Nvfp4QuantPairTransposeExactArgs, Nvfp4QuantRowwiseArgs, Nvfp4QuantRowwiseDerivedAmaxArgs,
    Nvfp4QuantTransposePaddedArgs, Nvfp4TransposeMsEdenDeviceScaleQuantArgs,
    QuartetBackwardMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenQuantArgs, RowAmaxArgs,
    RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs, TensorAmaxArgs,
};
pub use launcher::Nvfp4QuantModule;

pub const NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK: usize =
    kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK as usize;

#[inline]
pub fn nvfp4_tensor_amax_chunks(element_count: usize) -> usize {
    element_count.div_ceil(NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK)
}
