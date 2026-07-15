use cuda_core::DriverError;

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::launch_config::attention_config;
use super::launch_grads::run_grad_matmuls;
use super::launch_scores::run_pair_scores;
use super::matmul::AttentionTcMatmulContext;
use super::types::CausalAttentionBackwardTcArgs;
use crate::attention::AttentionModule;
use crate::launch::linear_config;

impl AttentionModule {
    pub fn causal_attention_backward_tc(
        &self,
        args: CausalAttentionBackwardTcArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        let params = args.params();
        let CausalAttentionBackwardTcArgs {
            stream,
            tc_module,
            qkv,
            attention_out,
            kda_v_new: _,
            kda_akk_inv: _,
            d_out,
            log_sum_exp,
            softmax_d,
            d_qkv,
            scratch,
            row_count: _,
            seq_len,
            batch_size,
            embedding_dim: _,
            qkv_dim: _,
            head_count,
            head_dim,
        } = args;
        let batch_head = batch_size * head_count;
        let tc_ctx = AttentionTcMatmulContext {
            stream,
            tc_module,
            batch_head,
            seq_len,
            head_dim,
        };
        let mut scratch = scratch;
        let linear = |n| linear_config(n, TC_BACKWARD_THREADS_PER_BLOCK);
        let kernels = &self.causal_attention_backward_tc.base;

        kernels.softmax_d_f16_kernel(
            stream,
            attention_config(seq_len, head_count, batch_size),
            attention_out,
            d_out,
            softmax_d,
            params,
        )?;
        kernels.gather_qkv_dout_kernel(
            stream,
            linear(batch_head * seq_len * head_dim),
            qkv,
            d_out,
            scratch.q,
            scratch.k,
            scratch.v,
            scratch.d_out,
            params,
        )?;
        run_pair_scores(&tc_ctx, &mut scratch)?;
        kernels.attention_prob_ds_f16_kernel(
            stream,
            linear(batch_head * seq_len * seq_len),
            scratch.scores,
            scratch.dot,
            log_sum_exp,
            softmax_d,
            scratch.p_half,
            scratch.ds_half,
            params,
        )?;
        run_grad_matmuls(&tc_ctx, &mut scratch)?;
        kernels.scatter_dqkv_kernel(
            stream,
            linear(batch_head * seq_len * head_dim),
            scratch.d_q,
            scratch.d_k,
            scratch.d_v,
            d_qkv,
            params,
        )
    }
}
