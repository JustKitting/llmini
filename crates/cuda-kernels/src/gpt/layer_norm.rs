pub const ROW_SIZE: usize = 32;
const WARPS_PER_BLOCK: u32 = 8;
const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * ROW_SIZE as u32;
const GPT_LAYER_NORM_THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const GPT_LAYER_NORM_WARPS_PER_BLOCK: u32 = GPT_LAYER_NORM_THREADS_PER_BLOCK / WARP_SIZE;

#[path = "layer_norm/body.rs"]
mod body;
#[path = "layer_norm/kernels.rs"]
mod kernels;
#[path = "layer_norm/launcher.rs"]
mod launcher;
pub use launcher::{
    GptLayerNormArgs, GptLayerNormSaveResidualF16Args, GptLayerNormSaveXhatF16Args, LayerNormArgs,
    LayerNormModule,
};
