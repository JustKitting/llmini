use cuda_core::DriverError;

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use super::types::CausalAttentionTcArgs;
use crate::attention::AttentionModule;
use crate::kda_launch::{self, KDA_HEAD_DIM};
use crate::launch::{grid_x_config, linear_config};

const KDA_RECURRENT_THREADS_PER_BLOCK: u32 = 512;

impl AttentionModule {
    pub fn kda_attention_tc(
        &self,
        args: CausalAttentionTcArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(
            args.head_dim, KDA_HEAD_DIM,
            "KDA path currently expects head_dim=64"
        );
        let params = args.params();
        let dims = kda_launch::LaunchDims::new(
            args.batch_size,
            args.head_count,
            args.seq_len,
            args.head_dim,
            params.chunk_size,
        );
        let mm = kda_launch::MatmulRunner::new(args.stream, args.tc_module, dims.chunk_batch);
        let scratch = args.scratch;
        let kda = &self.causal_attention_tc.kda;
        let stream = args.stream;
        let threads = TC_FORWARD_THREADS_PER_BLOCK;
        let linear = |n| linear_config(n, threads);
        let batch_cfg = grid_x_config(dims.batch_head, KDA_RECURRENT_THREADS_PER_BLOCK);
        let chunk_cfg = kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads);
        let matrix_cfg = grid_x_config(dims.chunk_batch, threads);
        macro_rules! kda_kernel {
            ($kernel:ident($config:expr; $($arg:expr),* $(,)?)) => {
                kda.$kernel(stream, $config, $($arg,)* params)?;
            };
        }
        if let Some(qkv_f16) = args.qkv_f16 {
            assert!(qkv_f16.len() >= (args.row_count * args.qkv_dim) as usize);
            kda_kernel!(prepare_kda_forward_save_f16_kernel(linear(dims.batch_head * args.seq_len * 32); args.qkv, qkv_f16, &mut *scratch.q, &mut *scratch.k, &mut *scratch.v, &mut *scratch.scores, &mut *args.log_sum_exp));
        } else {
            kda_kernel!(prepare_kda_forward_kernel(linear(dims.batch_head * args.seq_len * 32); args.qkv, &mut *scratch.q, &mut *scratch.k, &mut *scratch.v, &mut *scratch.scores, &mut *args.log_sum_exp));
        }
        kda_kernel!(chunk_cumsum_kda_g_kernel(chunk_cfg; &mut *scratch.scores));
        kda_kernel!(make_kda_qg_kneg_kg_kpos_vbeta_kernel(linear(dims.compact_elems); &mut *scratch.q, &mut *scratch.k, &mut *scratch.v, &*scratch.scores, &*args.log_sum_exp, &mut *scratch.compact_out, &mut *scratch.probs));
        kda_kernel!(store_kda_chunk_g_last_kernel(linear(dims.batch_head * dims.chunks * args.head_dim); &*scratch.scores, &mut *args.log_sum_exp));
        let reuses_initial_kneg = args.kda_w.is_some();
        let w = {
            let akk_inv = match args.kda_akk_inv {
                Some(akk_inv) => akk_inv,
                None => &mut *scratch.scores,
            };
            mm.f32_input_strict_causal(
                &*scratch.probs,
                &*scratch.compact_out,
                &mut *akk_inv,
                dims.cch(),
            )?;
            kda_kernel!(solve_kda_akk_inv_kernel(matrix_cfg; &mut *akk_inv));
            let w = match args.kda_w {
                Some(w) => {
                    mm.f32_rhs(&*akk_inv, &*scratch.probs, &mut *w, dims.chc())?;
                    &*w
                }
                None => {
                    mm.f32_rhs(
                        &*akk_inv,
                        &*scratch.probs,
                        &mut *scratch.compact_out,
                        dims.chc(),
                    )?;
                    &*scratch.compact_out
                }
            };
            mm.f32_rhs(&*akk_inv, &*scratch.v, &mut *scratch.probs, dims.chc())?;
            w
        };
        let kneg = if reuses_initial_kneg {
            &*scratch.compact_out
        } else {
            kda_kernel!(make_kda_kneg_from_kg_kernel(linear(dims.compact_elems); &*scratch.k, &*args.log_sum_exp, &mut *scratch.v));
            &*scratch.v
        };
        let aqk = match args.kda_aqk {
            Some(aqk) => {
                mm.f32_input_causal(&*scratch.q, kneg, &mut *aqk, dims.cch())?;
                &*aqk
            }
            None => {
                mm.f32_input_causal(&*scratch.q, kneg, &mut *scratch.scores, dims.cch())?;
                &*scratch.scores
            }
        };

        let v_new = match args.kda_v_new {
            Some(v_new) => v_new,
            None => &mut *scratch.v,
        };
        macro_rules! kda_output {
            ($chunk_states:expr) => {{
                kda_kernel!(chunk_kda_state_save_kernel(batch_cfg; &*scratch.k, &mut *v_new, w, &*scratch.probs, &*args.log_sum_exp, &mut *$chunk_states));
                kda_kernel!(chunk_kda_output_from_state_kernel(chunk_cfg; &*scratch.q, &*v_new, aqk, args.out, &*$chunk_states));
            }};
        }
        if let Some(chunk_states) = args.attention_out_f16 {
            kda_output!(chunk_states);
        } else {
            kda_output!(scratch.chunk_states);
        }
        kda.zero_kda_f32_kernel(
            stream,
            linear(dims.batch_head * args.seq_len),
            &mut *args.log_sum_exp,
            dims.batch_head * args.seq_len,
        )
    }
}
