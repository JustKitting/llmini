const HADAMARD_DIM: u32 = 32;
const GROUP_SIZE: u32 = 16;
const INV_SQRT_32: f32 = 0.176_776_69;
const FP4_MAX: f32 = 6.0;
const AMAX_WARPS_PER_BLOCK: u32 = crate::nvfp4_quant::config::WARPS_PER_BLOCK;

#[path = "ms_eden/amax.rs"]
pub(crate) mod amax;
#[path = "ms_eden/body.rs"]
mod body;
#[path = "ms_eden/fp32.rs"]
pub(crate) mod fp32;
#[path = "ms_eden/fp32_transpose.rs"]
pub(crate) mod fp32_transpose;
#[path = "ms_eden/input.rs"]
mod input;
#[path = "ms_eden/input/padded.rs"]
mod input_padded;
#[path = "ms_eden/input/values.rs"]
mod input_values;
#[path = "ms_eden/nvfp4_transpose.rs"]
pub(crate) mod nvfp4_transpose;
#[path = "ms_eden/pack.rs"]
mod pack;
#[path = "ms_eden/random.rs"]
mod random;
#[path = "ms_eden/rounding.rs"]
mod rounding;
#[path = "ms_eden/rowwise_transpose.rs"]
pub(crate) mod rowwise_transpose;
#[path = "ms_eden/transpose_kernels.rs"]
mod transpose_kernels;

#[path = "ms_eden/csf.rs"]
pub(crate) mod csf;

#[path = "ms_eden/input/position.rs"]
mod input_position;
