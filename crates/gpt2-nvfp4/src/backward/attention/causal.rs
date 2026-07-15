use cuda_core::DriverError;
use rust_kernels_cuda::attention::CausalAttentionBackwardTcArgs;

use super::types::AttentionCoreBackwardArgs;
use crate::AttentionDims;

pub fn causal_attention_backward(
    args: AttentionCoreBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    let dims = AttentionDims::new(args.use_full_attention);
    let tc_args = CausalAttentionBackwardTcArgs {
        reuse_forward_probs: args.reuse_forward_probs,
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
        d_qkv: args.d_qkv,
        scratch: args.scratch.tc,
        row_count: args.saved.row_count,
        seq_len: args.saved.seq_len,
        batch_size: args.saved.batch_size,
        embedding_dim: dims.embedding_dim,
        qkv_dim: dims.qkv_dim,
        head_count: dims.head_count,
        head_dim: dims.head_dim,
    };
    if args.use_full_attention {
        args.module.causal_attention_backward_tc(tc_args)
    } else {
        args.module.kda_attention_backward_tc(tc_args)
    }
}
