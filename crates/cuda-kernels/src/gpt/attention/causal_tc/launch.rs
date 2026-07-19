use cuda_core::DriverError;

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use super::selective::{SELECTIVE_APPLY_THREADS_PER_BLOCK, SELECTIVE_ATTENTION_THREADS_PER_BLOCK};
use super::types::CausalAttentionTcArgs;
use crate::attention::AttentionModule;
use crate::f16_tc_matmul::{
    F16TcMatmulF32Args, F16TcMatmulF32WindowArgs, F16TcMatmulHalfRhsArgs,
    F16TcMatmulHalfRhsWindowArgs,
};
use crate::launch::{grid_x_config, launch_config, linear_config};

impl AttentionModule {
    pub fn causal_attention_tc(
        &self,
        args: CausalAttentionTcArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        let params = args.params();
        let batch_head = args.batch_size * args.head_count;
        let scratch = args.scratch;
        let probs_half = args.forward_probs_f16.unwrap_or(scratch.probs_half);
        let selection_mask_tape = args.qkv_f16;

        let gather_config = linear_config(
            batch_head * args.seq_len * args.head_dim,
            TC_FORWARD_THREADS_PER_BLOCK,
        );
        if args.head_dim == 64 {
            self.causal_attention_tc
                .base
                .gather_qknorm_v_f16_forward_kernel(
                    args.stream,
                    gather_config,
                    args.qkv,
                    args.qk_scale.bytes,
                    args.qk_scale.scales,
                    args.qk_scale.global_scale,
                    &mut *scratch.q,
                    &mut *scratch.k,
                    &mut *scratch.chunk_states,
                    params,
                )?;
        } else {
            self.causal_attention_tc
                .base
                .gather_qk_v_f16_forward_kernel(
                    args.stream,
                    gather_config,
                    args.qkv,
                    &mut *scratch.q,
                    &mut *scratch.k,
                    &mut *scratch.chunk_states,
                    params,
                )?;
        }
        if params.attention_window == args.seq_len {
            args.tc_module
                .batched_matmul_f32_input_lower(F16TcMatmulF32Args {
                    stream: args.stream,
                    a: &*scratch.q,
                    b_t: &*scratch.k,
                    out: &mut *scratch.scores,
                    batch_count: batch_head,
                    m: args.seq_len,
                    n: args.seq_len,
                    k: args.head_dim,
                })?;
        } else {
            args.tc_module
                .batched_matmul_f32_input_windowed_lower(F16TcMatmulF32WindowArgs {
                    stream: args.stream,
                    a: &*scratch.q,
                    b_t: &*scratch.k,
                    out: &mut *scratch.scores,
                    batch_count: batch_head,
                    m: args.seq_len,
                    n: args.seq_len,
                    k: args.head_dim,
                    window: params.attention_window,
                })?;
        }
        if args.selective_attention {
            let packed_mask_elements = args.batch_size * args.seq_len * args.attention_window;
            assert!(
                scratch.probs.len() >= packed_mask_elements as usize,
                "selective attention requires one packed f32 mask per visible score"
            );
            let config = grid_x_config(
                args.batch_size * args.seq_len.div_ceil(SELECTIVE_ATTENTION_THREADS_PER_BLOCK),
                SELECTIVE_ATTENTION_THREADS_PER_BLOCK,
            );
            if let Some(selection_mask_tape) = selection_mask_tape {
                let required_tape_elements = args.row_count as usize * args.qkv_dim as usize
                    + (args.batch_size * args.seq_len * args.attention_window) as usize;
                assert!(
                    selection_mask_tape.len() >= required_tape_elements,
                    "selective attention requires QKV tape tail for the ReLU mask"
                );
                self.causal_attention_tc
                    .base
                    .selective_attention_mask_save_tape_kernel(
                        args.stream,
                        config,
                        &*scratch.scores,
                        &mut *scratch.probs,
                        selection_mask_tape,
                        params,
                    )?;
            } else {
                self.causal_attention_tc
                    .base
                    .selective_attention_mask_kernel(
                        args.stream,
                        config,
                        &*scratch.scores,
                        &mut *scratch.probs,
                        params,
                    )?;
            }
            self.causal_attention_tc
                .base
                .apply_selective_attention_mask_kernel(
                    args.stream,
                    linear_config(packed_mask_elements, SELECTIVE_APPLY_THREADS_PER_BLOCK),
                    &mut *scratch.scores,
                    &*scratch.probs,
                    params,
                )?;
        }
        self.causal_attention_tc
            .base
            .attention_softmax_forward_f16_kernel(
                args.stream,
                launch_config(
                    (args.seq_len, args.head_count, args.batch_size),
                    TC_FORWARD_THREADS_PER_BLOCK,
                ),
                &*scratch.scores,
                &mut *probs_half,
                args.log_sum_exp,
                params,
            )?;
        if params.attention_window == args.seq_len {
            args.tc_module
                .batched_matmul_half_rhs_lower_a(F16TcMatmulHalfRhsArgs {
                    stream: args.stream,
                    a: &*probs_half,
                    rhs: &*scratch.chunk_states,
                    out: &mut *scratch.compact_out,
                    batch_count: batch_head,
                    m: args.seq_len,
                    n: args.head_dim,
                    k: args.seq_len,
                })?;
        } else {
            args.tc_module.batched_matmul_half_rhs_windowed_lower_a(
                F16TcMatmulHalfRhsWindowArgs {
                    stream: args.stream,
                    a: &*probs_half,
                    rhs: &*scratch.chunk_states,
                    out: &mut *scratch.compact_out,
                    batch_count: batch_head,
                    m: args.seq_len,
                    n: args.head_dim,
                    k: args.seq_len,
                    window: params.attention_window,
                },
            )?;
        }
        let config = linear_config(
            batch_head * args.seq_len * args.head_dim,
            TC_FORWARD_THREADS_PER_BLOCK,
        );
        if let Some(attention_out_f16) = args.attention_out_f16 {
            return self
                .causal_attention_tc
                .base
                .scatter_attention_forward_save_f16_kernel(
                    args.stream,
                    config,
                    &*scratch.compact_out,
                    args.out,
                    attention_out_f16,
                    params,
                );
        }

        self.causal_attention_tc
            .base
            .scatter_attention_forward_kernel(
                args.stream,
                config,
                &*scratch.compact_out,
                args.out,
                params,
            )
    }
}
