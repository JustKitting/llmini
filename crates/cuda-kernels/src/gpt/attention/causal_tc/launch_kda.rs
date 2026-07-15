use cuda_core::DriverError;

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use super::types::CausalAttentionTcArgs;
use crate::attention::AttentionModule;
use crate::f16_tc_matmul::F16ConvertArgs;
use crate::kda_launch::{self, KDA_HEAD_DIM};
use crate::launch::{grid_x_config, linear_config};

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
        let batch_cfg = grid_x_config(dims.batch_head, threads);
        let chunk_cfg = kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads);
        let matrix_cfg = grid_x_config(dims.chunk_batch, threads);
        macro_rules! kda_kernel {
            ($kernel:ident($config:expr; $($arg:expr),* $(,)?)) => {
                kda.$kernel(stream, $config, $($arg,)* params)?;
            };
        }
        macro_rules! mm_in_scratch {
            ($a:ident, $b:ident, $out:ident, $shape:expr) => {
                mm.f32_input(&*scratch.$a, &*scratch.$b, &mut *scratch.$out, $shape)?;
            };
        }
        if let Some(qkv_f16) = args.qkv_f16 {
            args.tc_module.fp32_to_f16(F16ConvertArgs {
                stream,
                src: args.qkv,
                dst: qkv_f16,
                element_count: args.row_count * args.qkv_dim,
            })?;
        }

        kda_kernel!(prepare_kda_forward_kernel(linear(dims.batch_head * args.seq_len * 32); args.qkv, &mut *scratch.q, &mut *scratch.k, &mut *scratch.v, &mut *scratch.scores, &mut *args.log_sum_exp));
        kda_kernel!(chunk_cumsum_kda_g_kernel(chunk_cfg; &mut *scratch.scores));
        kda_kernel!(make_kda_qg_kneg_kernel(linear(dims.compact_elems); &mut *scratch.q, &*scratch.k, &*scratch.scores, &mut *scratch.compact_out));
        kda_kernel!(make_kda_kg_kpos_vbeta_kernel(linear(dims.compact_elems); &mut *scratch.k, &mut *scratch.v, &*scratch.scores, &*args.log_sum_exp, &mut *scratch.probs));
        kda_kernel!(store_kda_chunk_g_last_kernel(linear(dims.batch_head * dims.chunks * args.head_dim); &*scratch.scores, &mut *args.log_sum_exp));
        let w = {
            let akk_inv = match args.kda_akk_inv {
                Some(akk_inv) => akk_inv,
                None => &mut *scratch.scores,
            };
            mm.f32_input(
                &*scratch.probs,
                &*scratch.compact_out,
                &mut *akk_inv,
                dims.cch(),
            )?;
            kda_kernel!(mask_kda_akk_kernel(matrix_cfg; &mut *akk_inv));
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
        kda_kernel!(make_kda_kneg_from_kg_kernel(linear(dims.compact_elems); &*scratch.k, &*args.log_sum_exp, &mut *scratch.v));
        let aqk = match args.kda_aqk {
            Some(aqk) => {
                mm.f32_input(&*scratch.q, &*scratch.v, &mut *aqk, dims.cch())?;
                kda_kernel!(mask_kda_aqk_kernel(linear(dims.chunk_matrix_elems); &mut *aqk));
                &*aqk
            }
            None => {
                mm_in_scratch!(q, v, scores, dims.cch());
                kda_kernel!(mask_kda_aqk_kernel(linear(dims.chunk_matrix_elems); &mut *scratch.scores));
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
