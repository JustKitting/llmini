use cuda_core::DriverError;

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use super::kda::KDA_STATE_COLUMN_PARTITIONS;
use super::types::CausalAttentionTcArgs;
use crate::attention::AttentionModule;
use crate::f16_tc_matmul::cta_tile::CTA_WIDE_THREADS;
use crate::kda_launch::{self, KDA_HEAD_DIM};
use crate::launch::{launch_config, linear_config};

const KDA_WIDE_THREADS_PER_BLOCK: u32 = 512;

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
        let state_cfg = launch_config(
            (dims.batch_head, KDA_STATE_COLUMN_PARTITIONS, 1),
            CTA_WIDE_THREADS,
        );
        let tc_chunk_cfg =
            kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, KDA_WIDE_THREADS_PER_BLOCK);
        macro_rules! kda_kernel {
            ($kernel:ident($config:expr; $($arg:expr),* $(,)?)) => {
                kda.$kernel(stream, $config, $($arg,)* params)?;
            };
        }
        // The shared attention scratch allocations have no live FP32 values
        // here. KDA writes their bytes in the consumer's native format directly.
        // AQK must use the original causal key. Reconstructing it through
        // Kg * exp(-g_last) introduces future chunk-end decay and rounding.
        let half_len = dims.compact_elems as usize;
        assert!(
            scratch.probs.len() >= half_len,
            "KDA Kpos/U arena must hold a complete compact FP32 tensor"
        );
        let mut qg = unsafe { kda_launch::BorrowedHalfScratch::new(&*scratch.q, half_len) };
        let mut kg = unsafe { kda_launch::BorrowedHalfScratch::new(&*scratch.k, half_len) };
        let mut vbeta = unsafe { kda_launch::BorrowedHalfScratch::new(&*scratch.v, half_len) };
        let mut kneg =
            unsafe { kda_launch::BorrowedHalfScratch::new(&*scratch.compact_out, half_len) };
        if let Some(qkv_f16) = args.qkv_f16 {
            assert!(qkv_f16.len() >= (args.row_count * args.qkv_dim) as usize);
            kda_kernel!(prepare_transform_kda_forward_save_f16_kernel(kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads); args.qkv, qkv_f16, &mut *qg, &mut *kg, &mut *vbeta, &mut *kneg, &mut *scratch.probs, &mut *args.log_sum_exp));
        } else {
            kda_kernel!(prepare_transform_kda_forward_kernel(kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads); args.qkv, &mut *qg, &mut *kg, &mut *vbeta, &mut *kneg, &mut *scratch.probs, &mut *args.log_sum_exp));
        }
        let mut w_scratch;
        let w = {
            let akk_inv = match args.kda_akk_inv {
                Some(akk_inv) => akk_inv,
                None => &mut *scratch.probs_half,
            };
            mm.native_inverse(&*scratch.probs, &*kneg, &mut *akk_inv)?;
            let w = match args.kda_w {
                Some(w) => {
                    mm.kda_w_u(&*akk_inv, &*vbeta, &mut *scratch.probs, &mut *w)?;
                    &*w
                }
                None => {
                    // Keep Kneg intact for AQK. Dead scores storage holds W
                    // and AQK in disjoint native-half ranges through recurrence.
                    w_scratch = unsafe {
                        kda_launch::BorrowedHalfScratch::range(&*scratch.scores, 0, half_len)
                    };
                    mm.kda_w_u(&*akk_inv, &*vbeta, &mut *scratch.probs, &mut w_scratch)?;
                    &*w_scratch
                }
            };
            w
        };
        let aqk_kneg = &*kneg;
        let mut aqk_scratch;
        let aqk = match args.kda_aqk {
            Some(aqk) => {
                mm.half_causal_half_out(&*qg, aqk_kneg, &mut *aqk, dims.cch())?;
                &*aqk
            }
            None => {
                aqk_scratch = unsafe {
                    kda_launch::BorrowedHalfScratch::range(
                        &*scratch.scores,
                        half_len,
                        dims.chunk_matrix_elems as usize,
                    )
                };
                mm.half_causal_half_out(&*qg, aqk_kneg, &mut aqk_scratch, dims.cch())?;
                &*aqk_scratch
            }
        };

        let v_new = match args.kda_v_new {
            Some(v_new) => v_new,
            None => &mut *scratch.probs_half,
        };
        assert!(v_new.len() >= dims.compact_elems as usize);
        macro_rules! kda_output {
            ($chunk_states:expr) => {{
                kda_kernel!(chunk_kda_state_save_kernel(state_cfg; &*kg, &mut *v_new, w, &*scratch.probs, &*args.log_sum_exp, &mut *$chunk_states));
                kda_kernel!(chunk_kda_output_from_state_kernel(tc_chunk_cfg; &*qg, &*v_new, aqk, args.out, &*$chunk_states));
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
