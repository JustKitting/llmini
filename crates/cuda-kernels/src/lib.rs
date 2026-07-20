#![allow(static_mut_refs)]

#[path = "utils/amax.rs"]
pub(crate) mod amax;
#[path = "utils/atomic.rs"]
pub mod atomic;
#[path = "gpt/attention/mod.rs"]
pub mod attention;
#[path = "utils/block_reduce.rs"]
pub(crate) mod block_reduce;
#[path = "gpt/canon.rs"]
pub mod canon;
#[path = "utils/device_ptr.rs"]
pub(crate) mod device_ptr;
#[path = "gpt/embedding.rs"]
pub mod embedding;
#[path = "utils/f16_tc_matmul/mod.rs"]
pub mod f16_tc_matmul;
#[path = "utils/f32_matrix_ops.rs"]
pub mod f32_matrix_ops;
#[path = "utils/float_ptx.rs"]
pub mod float_ptx;
#[path = "utils/kda_backward.rs"]
pub(crate) mod kda_backward;
#[path = "utils/kda_common.rs"]
pub(crate) mod kda_common;
#[path = "utils/kda_elementwise.rs"]
pub(crate) mod kda_elementwise;
#[path = "utils/kda_launch.rs"]
pub(crate) mod kda_launch;
#[path = "utils/kda_tc.rs"]
pub(crate) mod kda_tc;
#[path = "utils/launch.rs"]
pub(crate) mod launch;
#[path = "gpt/layer_norm.rs"]
pub mod layer_norm;
#[path = "gpt/layer_norm_backward/mod.rs"]
pub mod layer_norm_backward;
#[path = "utils/layer_norm_reduce.rs"]
pub(crate) mod layer_norm_reduce;
#[path = "utils/layer_norm.rs"]
pub mod layer_norm_utils;
#[path = "gpt/linear_backward.rs"]
pub mod linear_backward;
#[path = "gpt/lm_head.rs"]
pub mod lm_head;
#[path = "gpt/logits/mod.rs"]
pub mod logits;
#[path = "gpt/loss.rs"]
pub mod loss;
#[path = "gpt/mlp/mod.rs"]
pub mod mlp;
#[path = "utils/mma/mod.rs"]
pub mod mma;
#[path = "gpt/next_latent/mod.rs"]
pub mod next_latent;
#[path = "utils/nvfp4/mod.rs"]
pub mod nvfp4;
#[path = "utils/nvfp4/cast.rs"]
pub mod nvfp4_cast;
#[path = "utils/nvfp4/quant/mod.rs"]
pub mod nvfp4_quant;
#[path = "utils/nvfp4_tc_matmul/mod.rs"]
pub mod nvfp4_tc_matmul;
#[path = "utils/nvfp4_tma_matmul/mod.rs"]
pub mod nvfp4_tma_matmul;
#[path = "gpt/optimizer/mod.rs"]
pub mod optimizer;
#[path = "gpt/projection_postop.rs"]
pub mod projection_postop;
#[path = "utils/quartet.rs"]
pub mod quartet;
#[path = "gpt/residual.rs"]
pub mod residual;
#[path = "utils/shuffle.rs"]
pub mod shuffle;
#[path = "utils/transpose.rs"]
pub mod transpose;
#[path = "utils/warp_reduce.rs"]
pub mod warp_reduce;

pub mod gpt {
    pub use crate::{
        attention, canon, embedding, layer_norm, layer_norm_backward, linear_backward, lm_head,
        logits, loss, mlp, next_latent, optimizer, projection_postop, residual,
    };
}

pub mod utils {
    pub use crate::{
        atomic, f16_tc_matmul, f32_matrix_ops, float_ptx, layer_norm_utils, mma, nvfp4, nvfp4_cast,
        nvfp4_quant, nvfp4_tc_matmul, nvfp4_tma_matmul, quartet, shuffle, transpose, warp_reduce,
    };
}

pub const CUDA_OXIDE_PTX_PATH: &str = "rust_kernels_cuda.ptx";
