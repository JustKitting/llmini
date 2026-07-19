use cuda_core::DriverError;

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::types::{CausalAttentionBackwardTcArgs, CausalAttentionBackwardTcScratch};
use crate::attention::AttentionModule;
use crate::f16_tc_matmul::cta_tile::CTA_ULTRA_WIDE_THREADS;
use crate::kda_launch::{self, KDA_HEAD_DIM};
use crate::launch::{grid_x_config, linear_config};

const KDA_WIDE_THREADS_PER_BLOCK: u32 = 512;

use super::kernels::KDA_NORM_REDUCE_THREADS_PER_BLOCK;

impl AttentionModule {
    pub fn kda_attention_backward_tc(
        &self,
        args: CausalAttentionBackwardTcArgs<'_, '_, '_>,
    ) -> Result<u32, DriverError> {
        self.kda_attention_backward_tc_impl(args, false)
    }

    /// Runs the exact KDA backward inside each chunk while treating the saved
    /// recurrent state entering that chunk as a backward constant.
    ///
    /// The forward recurrence is unchanged, and every KDA projection still
    /// receives its within-chunk parameter gradient.
    pub fn kda_attention_backward_tc_detached_chunk_state(
        &self,
        args: CausalAttentionBackwardTcArgs<'_, '_, '_>,
    ) -> Result<u32, DriverError> {
        self.kda_attention_backward_tc_impl(args, true)
    }

    fn kda_attention_backward_tc_impl(
        &self,
        args: CausalAttentionBackwardTcArgs<'_, '_, '_>,
        detach_chunk_state: bool,
    ) -> Result<u32, DriverError> {
        assert_eq!(
            args.head_dim, KDA_HEAD_DIM,
            "KDA path currently expects head_dim=64"
        );
        let params = args.params();
        let CausalAttentionBackwardTcArgs {
            reuse_forward_probs: _,
            forward_probs_f16: _,
            stream,
            tc_module,
            qkv,
            attention_out: chunk_states,
            kda_v_new,
            kda_akk_inv,
            kda_w,
            kda_aqk,
            d_out,
            qk_scale: _,
            log_sum_exp: _log_sum_exp,
            softmax_d: beta,
            qk_norm_max,
            d_qkv,
            d_qkv_chunk_amax,
            d_qk_scale: _,
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
            partial_key_offset: _,
            selective_attention: _,
            qk_norm_offset,
            backward_mask_seed: _,
            backward_tile_budget: _,
        } = args;
        let dims = kda_launch::LaunchDims::new(
            batch_size,
            head_count,
            seq_len,
            head_dim,
            params.chunk_size,
        );
        let mm = kda_launch::MatmulRunner::new(stream, tc_module, dims.chunk_batch);
        let CausalAttentionBackwardTcScratch {
            q_f32: qg,
            k_f32: kg,
            v_f32: vbeta,
            g_f32: g,
            q: _q_half,
            k: _k_half,
            v: _v_half,
            d_out: _d_out_half,
            scores: chunk_matrix,
            dot: aqk_or_dm,
            p: local_grad,
            ds: dkg_from_state,
            p_half: _p_half,
            ds_half: _ds_half,
            d_q: kneg_vnew_dqg_dv,
            d_k: kpos_u_dw,
            d_v: w_du_dq,
            kda_d_q: dka_dg,
            kda_d_k: d_kneg_from_inverse,
            kda_d_v: dout_daqk_dvbeta,
            kda_d_g: dh_states_or_kneg,
            kda_d_beta: d_beta,
        } = scratch;
        let bwd_elementwise = &self.causal_attention_backward_tc.kda_elementwise;
        let bwd_tc = &self.causal_attention_backward_tc.kda_tc;
        let fwd = &self.causal_attention_tc.kda;
        let threads = TC_BACKWARD_THREADS_PER_BLOCK;
        let chunkwise_cfg = grid_x_config(dims.batch_head, CTA_ULTRA_WIDE_THREADS);
        let chunk_cfg = kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads);
        let tc_chunk_cfg =
            kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, KDA_WIDE_THREADS_PER_BLOCK);
        let matrix_cfg = grid_x_config(dims.chunk_batch, threads);
        let finish_element_count = dims.batch_head * seq_len * 32;
        let finish_chunk_count = finish_element_count.div_ceil(threads);
        assert!(d_qkv_chunk_amax.len() >= finish_chunk_count as usize);
        macro_rules! launch {
            ($target:ident.$kernel:ident($config:expr; $($arg:expr),* $(,)?)) => {
                $target.$kernel(stream, $config, $($arg,)* params)?;
            };
        }
        macro_rules! mm {
            ($method:ident($a:expr, $b:expr, $out:expr, $shape:expr)) => {
                mm.$method($a, $b, $out, $shape)?;
            };
        }

        launch!(bwd_elementwise.prepare_kda_backward_inputs_kernel(linear_config(dims.batch_head * seq_len * 32, threads); qkv, qg, kg, vbeta, g, beta));
        launch!(bwd_elementwise.chunk_cumsum_kda_backward_g_kernel(chunk_cfg; g));
        if kda_v_new.is_some() && kda_akk_inv.is_some() && kda_w.is_some() && kda_aqk.is_some() {
            launch!(fwd.make_kda_qg_kg_vbeta_kernel(linear_config(dims.compact_elems, threads); qg, kg, vbeta, g, beta));
        } else {
            launch!(fwd.make_kda_qg_kneg_kg_kpos_vbeta_kernel(linear_config(dims.compact_elems, threads); qg, kg, vbeta, g, beta, kneg_vnew_dqg_dv, kpos_u_dw));
        }
        if kda_akk_inv.is_none() {
            mm!(f32_input_strict_causal(
                kpos_u_dw,
                kneg_vnew_dqg_dv,
                chunk_matrix,
                dims.cch()
            ));
            launch!(fwd.solve_kda_akk_inv_kernel(matrix_cfg; chunk_matrix));
        }
        let w = {
            let akk_inv = kda_akk_inv.unwrap_or(&*chunk_matrix);
            match kda_w {
                Some(w) => w,
                None => {
                    mm!(f32_rhs(akk_inv, kpos_u_dw, w_du_dq, dims.chc()));
                    &*w_du_dq
                }
            }
        };
        if kda_v_new.is_none() {
            let akk_inv = kda_akk_inv.unwrap_or(&*chunk_matrix);
            mm!(f32_rhs(akk_inv, vbeta, kpos_u_dw, dims.chc()));
            launch!(bwd_tc.chunk_kda_vnew_from_state_kernel(tc_chunk_cfg; w, kpos_u_dw, chunk_states, kneg_vnew_dqg_dv));
        }
        let aqk = match kda_aqk {
            Some(aqk) => aqk,
            None => {
                mm!(f32_input_causal(
                    qg,
                    kneg_vnew_dqg_dv,
                    aqk_or_dm,
                    dims.cch()
                ));
                &*aqk_or_dm
            }
        };
        launch!(bwd_elementwise.gather_kda_dout_kernel(linear_config(dims.compact_elems, threads); d_out, dout_daqk_dvbeta));
        {
            let v_new = kda_v_new.unwrap_or(&*kneg_vnew_dqg_dv);
            mm!(f32_input_causal(
                dout_daqk_dvbeta,
                v_new,
                local_grad,
                dims.cch()
            ));
        }
        mm!(f32_a_transposed_rhs(
            aqk,
            dout_daqk_dvbeta,
            kpos_u_dw,
            dims.chc()
        ));
        if detach_chunk_state {
            fwd.zero_kda_f32_kernel(
                stream,
                linear_config(dims.compact_elems, threads),
                &mut *dkg_from_state,
                dims.compact_elems,
            )?;
        } else {
            launch!(bwd_tc.chunkwise_kda_backward_kernel(chunkwise_cfg; qg, kg, kpos_u_dw, w, aqk, g, chunk_states, d_out, dh_states_or_kneg, local_grad));
            let v_new = kda_v_new.unwrap_or(&*kneg_vnew_dqg_dv);
            launch!(bwd_tc.chunk_kda_dkg_from_vnew_dh_kernel(tc_chunk_cfg; v_new, dh_states_or_kneg, dkg_from_state));
        }
        launch!(bwd_tc.chunk_kda_dw_dqg_from_state_kernel(tc_chunk_cfg; kpos_u_dw, dout_daqk_dvbeta, chunk_states, w_du_dq, kneg_vnew_dqg_dv));
        launch!(bwd_elementwise.make_kda_backward_kneg_from_kg_kernel(linear_config(dims.compact_elems, threads); kg, g, dh_states_or_kneg));
        mm!(f32_input_accumulate(
            local_grad,
            dh_states_or_kneg,
            kneg_vnew_dqg_dv,
            dims.chc()
        ));
        mm!(f32_a_transposed_rhs(local_grad, qg, dka_dg, dims.chc()));
        {
            let akk_inv = kda_akk_inv.unwrap_or(&*chunk_matrix);
            mm!(f32_a_transposed_rhs(
                akk_inv,
                w_du_dq,
                dh_states_or_kneg,
                dims.chc()
            ));
            mm!(f32_a_transposed_rhs(
                akk_inv,
                kpos_u_dw,
                dout_daqk_dvbeta,
                dims.chc()
            ));
        }
        launch!(bwd_tc.chunk_intra_kda_dm_kernel(tc_chunk_cfg; kg, vbeta, g, beta, kpos_u_dw, w_du_dq, aqk_or_dm));
        match kda_akk_inv {
            Some(akk_inv) => {
                mm!(f32_input(aqk_or_dm, akk_inv, local_grad, dims.ccc()));
                mm!(f32_a_transposed_rhs_strict_neg(
                    akk_inv,
                    local_grad,
                    chunk_matrix,
                    dims.ccc()
                ));
            }
            None => {
                mm!(f32_input(aqk_or_dm, chunk_matrix, local_grad, dims.ccc()));
                mm!(f32_a_transposed_rhs(
                    chunk_matrix,
                    local_grad,
                    aqk_or_dm,
                    dims.ccc()
                ));
                launch!(bwd_elementwise.make_kda_strict_neg_matrix_kernel(linear_config(dims.chunk_matrix_elems, threads); aqk_or_dm, chunk_matrix));
            }
        }
        launch!(bwd_elementwise.make_kda_backward_kneg_kpos_from_kg_kernel(linear_config(dims.compact_elems, threads); kg, g, beta, kpos_u_dw, aqk_or_dm));
        mm!(f32_rhs(
            chunk_matrix,
            kpos_u_dw,
            d_kneg_from_inverse,
            dims.chc()
        ));
        mm!(f32_a_transposed_rhs(
            chunk_matrix,
            aqk_or_dm,
            local_grad,
            dims.chc()
        ));
        launch!(bwd_tc.chunk_intra_kda_backward_kernel(chunk_cfg; qg, kg, vbeta, g, beta, kneg_vnew_dqg_dv, dkg_from_state, dka_dg, dh_states_or_kneg, dout_daqk_dvbeta, d_kneg_from_inverse, local_grad, w_du_dq, chunk_matrix, d_beta));
        // After chunk_intra_kda_backward_kernel these reused buffers hold final compact gradients.
        // qg and kg are dead after the final gradient pass, so reuse their leading
        // batch-head rows for the post-SiLU norms already computed by that pass.
        launch!(bwd_elementwise.finish_kda_backward_kernel(grid_x_config(finish_chunk_count, threads); qkv, w_du_dq, chunk_matrix, kneg_vnew_dqg_dv, dka_dg, d_beta, qg, kg, d_qkv, d_qkv_chunk_amax, u32::from(accumulate_value_grad)));
        bwd_elementwise.reduce_kda_qk_norm_max_kernel(
            stream,
            grid_x_config(head_count, KDA_NORM_REDUCE_THREADS_PER_BLOCK),
            &*qg,
            &*kg,
            qk_norm_max,
            qk_norm_offset,
            params.row_count,
            head_count,
        )?;
        Ok(finish_chunk_count)
    }
}
