use cuda_core::DriverError;

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::kernels::KDA_NORM_REDUCE_THREADS_PER_BLOCK;
use super::launch_config::attention_config;
use super::launch_grads::run_grad_matmuls;
use super::launch_scores::{run_ds_scores, run_pair_scores};
use super::matmul::AttentionTcMatmulContext;
use super::types::CausalAttentionBackwardTcArgs;
use crate::attention::AttentionModule;
use crate::launch::{grid_x_config, linear_config};

impl AttentionModule {
    pub fn causal_attention_backward_tc(
        &self,
        args: CausalAttentionBackwardTcArgs<'_, '_, '_>,
    ) -> Result<u32, DriverError> {
        let params = args.params();
        let CausalAttentionBackwardTcArgs {
            reuse_forward_probs,
            forward_probs_f16,
            stream,
            tc_module,
            qkv,
            attention_out,
            kda_v_new: _,
            kda_akk_inv: _,
            kda_w: _,
            kda_aqk: _,
            d_out,
            log_sum_exp,
            softmax_d,
            qk_norm_max,
            d_qkv,
            d_qkv_chunk_amax,
            scratch,
            row_count: _,
            seq_len,
            batch_size,
            embedding_dim: _,
            qkv_dim: _,
            head_count,
            head_dim,
            attention_window: _,
            qk_norm_offset,
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
        if head_dim == 64 {
            kernels.gather_qkv_dout_norms_kernel(
                stream,
                linear(batch_head * seq_len * head_dim),
                qkv,
                d_out,
                scratch.q,
                scratch.k,
                scratch.v,
                scratch.d_out,
                scratch.q_f32,
                scratch.k_f32,
                params,
            )?;
            self.causal_attention_backward_tc
                .kda_elementwise
                .reduce_kda_qk_norm_max_kernel(
                    stream,
                    grid_x_config(head_count, KDA_NORM_REDUCE_THREADS_PER_BLOCK),
                    &*scratch.q_f32,
                    &*scratch.k_f32,
                    qk_norm_max,
                    qk_norm_offset,
                    params.row_count,
                    head_count,
                )?;
        } else {
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
        }
        if reuse_forward_probs {
            match forward_probs_f16 {
                Some(probs_half) => {
                    run_ds_scores(
                        &tc_ctx,
                        &*scratch.d_out,
                        &*scratch.v,
                        probs_half,
                        softmax_d,
                        &mut *scratch.ds_half,
                    )?;
                    run_grad_matmuls(&tc_ctx, &mut scratch, Some(probs_half))?;
                }
                None => {
                    run_ds_scores(
                        &tc_ctx,
                        &*scratch.d_out,
                        &*scratch.v,
                        &*scratch.p_half,
                        softmax_d,
                        &mut *scratch.ds_half,
                    )?;
                    run_grad_matmuls(&tc_ctx, &mut scratch, None)?;
                }
            }
        } else {
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
            run_grad_matmuls(&tc_ctx, &mut scratch, None)?;
        }
        assert_eq!(
            params.embedding_dim,
            params.head_count * params.head_dim,
            "full-attention embedding must equal head_count * head_dim"
        );
        assert_eq!(
            params.qkv_dim,
            3 * params.embedding_dim,
            "full-attention output must contain contiguous Q, K, and V sections"
        );
        let chunk_count = 3 * params.row_count;
        assert!(d_qkv_chunk_amax.len() >= chunk_count as usize);
        kernels.scatter_dqkv_amax_kernel(
            stream,
            grid_x_config(chunk_count, TC_BACKWARD_THREADS_PER_BLOCK),
            scratch.d_q,
            scratch.d_k,
            scratch.d_v,
            d_qkv,
            d_qkv_chunk_amax,
            params,
        )?;
        Ok(chunk_count)
    }
}
