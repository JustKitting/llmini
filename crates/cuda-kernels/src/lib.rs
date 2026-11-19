#![allow(static_mut_refs)]

#[path = "gpt/attention/kda_decay_audit.rs"]
pub mod kda_decay_audit;
#[path = "utils/kda_exact_boundary_vjp.rs"]
pub mod kda_exact_boundary_vjp;
#[path = "utils/kda_exact_factors.rs"]
pub mod kda_exact_factors;
#[path = "utils/kda_exact_forward.rs"]
pub mod kda_exact_forward;
#[path = "utils/kda_exact_grad_matmul.rs"]
pub mod kda_exact_grad_matmul;
#[path = "utils/kda_exact_inverse_vjp.rs"]
pub mod kda_exact_inverse_vjp;
#[path = "utils/kda_exact_pair_vjp.rs"]
pub mod kda_exact_pair_vjp;
#[path = "utils/kda_exact_pairs.rs"]
pub mod kda_exact_pairs;
#[path = "utils/kda_exact_prepare.rs"]
pub mod kda_exact_prepare;
#[path = "utils/kda_exact_raw_vjp.rs"]
pub mod kda_exact_raw_vjp;
#[path = "utils/kda_exact_state_carry_vjp.rs"]
pub mod kda_exact_state_carry_vjp;
#[path = "gpt/mhc_coefficients.rs"]
pub mod mhc_coefficients;
#[path = "gpt/mhc_mixing.rs"]
pub mod mhc_mixing;
#[path = "gpt/mhc_post_reuse.rs"]
pub mod mhc_post_reuse;
#[path = "gpt/mhc_reverse.rs"]
pub mod mhc_reverse;
#[path = "gpt/mhc_rms_quant.rs"]
pub mod mhc_rms_quant;
#[path = "gpt/mhc_window_post.rs"]
pub mod mhc_window_post;
#[path = "utils/module_exports.rs"]
pub(crate) mod module_exports;
#[path = "utils/position.rs"]
pub mod position;
#[path = "gpt/value_rms.rs"]
pub mod value_rms;

#[path = "utils/activation.rs"]
pub mod activation;

#[path = "utils/amax.rs"]
pub(crate) mod amax;
#[path = "utils/atomic.rs"]
pub mod atomic;
#[path = "gpt/attention/mod.rs"]
pub mod attention;
#[path = "utils/block_reduce.rs"]
pub(crate) mod block_reduce;
#[path = "gpt/bounded_swiglu.rs"]
pub mod bounded_swiglu;
#[path = "gpt/canon.rs"]
pub mod canon;
#[path = "utils/device_ptr.rs"]
pub(crate) mod device_ptr;
#[path = "gpt/embedding.rs"]
pub mod embedding;
#[path = "gpt/engram.rs"]
pub mod engram;
#[path = "gpt/engram_multi.rs"]
pub mod engram_multi;
#[path = "gpt/hnet_byte_head.rs"]
pub mod hnet_byte_head;
#[path = "gpt/hnet_chunk.rs"]
pub mod hnet_chunk;
#[path = "gpt/hnet_optimizer.rs"]
pub mod hnet_optimizer;
#[path = "gpt/hnet_router.rs"]
pub mod hnet_router;
#[path = "gpt/hnet_stage.rs"]
pub mod hnet_stage;
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
#[path = "gpt/latent_feedback.rs"]
pub mod latent_feedback;
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
#[path = "gpt/mhc.rs"]
pub mod mhc;
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
