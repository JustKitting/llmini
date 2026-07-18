use cuda_core::DriverError;
use rust_kernels_cuda::attention::CausalAttentionBackwardTcArgs;

use super::types::AttentionCoreBackwardArgs;
use crate::{AttentionDims, GPT2_ATTENTION_BACKWARD_TILE_BUDGET, GPT2_FULL_ATTENTION_WINDOW};

pub fn causal_attention_backward(
    args: AttentionCoreBackwardArgs<'_, '_, '_>,
) -> Result<Option<u32>, DriverError> {
    assert!(
        !args.use_full_attention || args.reuse_forward_probs,
        "the GPT-2 full-attention backward scratch requires saved forward probabilities"
    );
    let dims = AttentionDims::new(args.use_full_attention);
    let tc_args = CausalAttentionBackwardTcArgs {
        reuse_forward_probs: args.reuse_forward_probs,
        forward_probs_f16: args.saved.attention_probs,
        stream: args.stream,
        tc_module: args.tc_module,
        qkv: args.saved.qkv,
        attention_out: args.saved.attention_out,
        kda_v_new: args.saved.kda_v_new,
        kda_akk_inv: args.saved.kda_akk_inv,
        kda_w: args.saved.kda_w,
        kda_aqk: args.saved.kda_aqk,
        d_out: args.d_attention_out,
        log_sum_exp: args.saved.attention_log_sum_exp,
        softmax_d: args.scratch.softmax_d,
        qk_norm_max: args.scratch.qk_norm_max,
        d_qkv: args.d_qkv,
        d_qkv_chunk_amax: args.d_qkv_chunk_amax,
        scratch: args.scratch.tc,
        row_count: args.saved.row_count,
        seq_len: args.saved.seq_len,
        batch_size: args.saved.batch_size,
        embedding_dim: dims.embedding_dim,
        qkv_dim: dims.qkv_dim,
        head_count: dims.head_count,
        head_dim: dims.head_dim,
        attention_window: if args.use_full_attention {
            GPT2_FULL_ATTENTION_WINDOW as u32
        } else {
            args.saved.seq_len
        },
        qk_norm_offset: (args.block_index as u32) * 2 * dims.head_count,
        backward_mask_seed: args.backward_mask_seed,
        backward_tile_budget: if args.use_full_attention {
            GPT2_ATTENTION_BACKWARD_TILE_BUDGET
        } else {
            0.0
        },
    };
    if args.use_full_attention {
        args.module.causal_attention_backward_tc(tc_args).map(Some)
    } else {
        args.module
            .kda_attention_backward_tc_detached_chunk_state(tc_args)
            .map(Some)
    }
}
