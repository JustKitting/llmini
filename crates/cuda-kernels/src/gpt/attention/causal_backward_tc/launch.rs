use cuda_core::DriverError;

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::kernels::KDA_NORM_REDUCE_THREADS_PER_BLOCK;
use super::launch_config::attention_config;
use super::launch_grads::{run_grad_matmuls, run_grad_matmuls_sparse};
use super::launch_scores::{run_ds_scores, run_ds_scores_sparse, run_pair_scores};
use super::matmul::AttentionTcMatmulContext;
use super::sparse_probs::SPARSE_PROB_THREADS_PER_BLOCK;
use super::types::CausalAttentionBackwardTcArgs;
use crate::attention::AttentionModule;
use crate::f16_tc_matmul::cta_tile::CTA_M;
use crate::launch::launch_config;
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
            qk_scale,
            log_sum_exp,
            softmax_d,
            qk_norm_max,
            d_qkv,
            d_qkv_chunk_amax,
            d_qk_scale,
            accumulate_value_grad,
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
            backward_mask_seed,
            backward_tile_budget,
        } = args;
        let batch_head = batch_size * head_count;
        let tc_ctx = AttentionTcMatmulContext {
            stream,
            tc_module,
            batch_head,
            seq_len,
            head_dim,
            attention_window: params.attention_window,
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
                qk_scale.bytes,
                qk_scale.scales,
                qk_scale.global_scale,
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
            let probs_half = forward_probs_f16.unwrap_or(&*scratch.p_half);
            if backward_tile_budget > 0.0 {
                let tiles = seq_len.div_ceil(CTA_M);
                kernels.sparsify_attention_probs_f16_kernel(
                    stream,
                    launch_config((tiles, tiles, batch_head), SPARSE_PROB_THREADS_PER_BLOCK),
                    probs_half,
                    &mut *scratch.p,
                    backward_mask_seed,
                    backward_tile_budget,
                    params,
                )?;
                run_ds_scores_sparse(
                    &tc_ctx,
                    &*scratch.d_out,
                    &*scratch.v,
                    probs_half,
                    softmax_d,
                    &*scratch.p,
                    &mut *scratch.ds_half,
                )?;
                run_grad_matmuls_sparse(
                    &tc_ctx,
                    &*scratch.ds_half,
                    &*scratch.q,
                    &*scratch.k,
                    &*scratch.d_out,
                    probs_half,
                    &*scratch.p,
                    &mut *scratch.d_q,
                    &mut *scratch.d_k,
                    &mut *scratch.d_v,
                )?;
            } else {
                run_ds_scores(
                    &tc_ctx,
                    &*scratch.d_out,
                    &*scratch.v,
                    probs_half,
                    softmax_d,
                    &mut *scratch.ds_half,
                )?;
                run_grad_matmuls(&tc_ctx, &mut scratch, forward_probs_f16)?;
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
        assert!(
            params.qkv_dim == 3 * params.embedding_dim
                || params.qkv_dim >= 3 * params.embedding_dim + params.head_count,
            "full-attention output must contain contiguous Q, K, V, optional head-gate, and padding sections"
        );
        let chunk_count = 3 * params.row_count;
        assert!(d_qkv_chunk_amax.len() >= chunk_count as usize);
        if head_dim == 64 {
            kernels.scatter_qknorm_dqkv_amax_kernel(
                stream,
                grid_x_config(chunk_count, TC_BACKWARD_THREADS_PER_BLOCK),
                qkv,
                &*scratch.q_f32,
                &*scratch.k_f32,
                scratch.d_q,
                scratch.d_k,
                scratch.d_v,
                qk_scale.bytes,
                qk_scale.scales,
                qk_scale.global_scale,
                d_qkv,
                &mut *scratch.g_f32,
                d_qkv_chunk_amax,
                u32::from(accumulate_value_grad),
                params,
            )?;
            kernels.reduce_qk_scale_grad_kernel(
                stream,
                grid_x_config(1, TC_BACKWARD_THREADS_PER_BLOCK),
                &*scratch.g_f32,
                d_qk_scale,
                params.row_count,
            )?;
        } else {
            kernels.scatter_dqkv_amax_kernel(
                stream,
                grid_x_config(chunk_count, TC_BACKWARD_THREADS_PER_BLOCK),
                scratch.d_q,
                scratch.d_k,
                scratch.d_v,
                d_qkv,
                d_qkv_chunk_amax,
                u32::from(accumulate_value_grad),
                params,
            )?;
            kernels.reduce_qk_scale_grad_kernel(
                stream,
                grid_x_config(1, TC_BACKWARD_THREADS_PER_BLOCK),
                &*scratch.g_f32,
                d_qk_scale,
                0,
            )?;
        }
        Ok(chunk_count)
    }
}
