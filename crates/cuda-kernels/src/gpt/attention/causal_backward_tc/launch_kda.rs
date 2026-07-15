use cuda_core::DriverError;

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::types::{CausalAttentionBackwardTcArgs, CausalAttentionBackwardTcScratch};
use crate::attention::AttentionModule;
use crate::kda_launch::{self, KDA_HEAD_DIM};
use crate::launch::{grid_x_config, linear_config};

impl AttentionModule {
    pub fn kda_attention_backward_tc(
        &self,
        args: CausalAttentionBackwardTcArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
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
            log_sum_exp: _log_sum_exp,
            softmax_d: beta,
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
        let batch_cfg = grid_x_config(dims.batch_head, threads);
        let chunk_cfg = kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads);
        let matrix_cfg = grid_x_config(dims.chunk_batch, threads);
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
        launch!(fwd.make_kda_qg_kneg_kernel(linear_config(dims.compact_elems, threads); qg, kg, g, kneg_vnew_dqg_dv));
        launch!(fwd.make_kda_kg_kpos_vbeta_kernel(linear_config(dims.compact_elems, threads); kg, vbeta, g, beta, kpos_u_dw));
        if kda_akk_inv.is_none() {
            mm!(f32_input(
                kpos_u_dw,
                kneg_vnew_dqg_dv,
                chunk_matrix,
                dims.cch()
            ));
            launch!(fwd.mask_kda_akk_kernel(matrix_cfg; chunk_matrix));
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
            launch!(bwd_tc.chunk_kda_vnew_from_state_kernel(chunk_cfg; w, kpos_u_dw, chunk_states, kneg_vnew_dqg_dv));
        }
        let aqk = match kda_aqk {
            Some(aqk) => aqk,
            None => {
                mm!(f32_input(qg, kneg_vnew_dqg_dv, aqk_or_dm, dims.cch()));
                launch!(fwd.mask_kda_aqk_kernel(linear_config(dims.chunk_matrix_elems, threads); aqk_or_dm));
                &*aqk_or_dm
            }
        };
        launch!(bwd_elementwise.gather_kda_dout_kernel(linear_config(dims.compact_elems, threads); d_out, dout_daqk_dvbeta));
        {
            let v_new = kda_v_new.unwrap_or(&*kneg_vnew_dqg_dv);
            mm!(f32_input(dout_daqk_dvbeta, v_new, local_grad, dims.cch()));
        }
        launch!(fwd.mask_kda_aqk_kernel(linear_config(dims.chunk_matrix_elems, threads); local_grad));
        mm!(f32_a_transposed_rhs(
            aqk,
            dout_daqk_dvbeta,
            kpos_u_dw,
            dims.chc()
        ));
        launch!(bwd_tc.chunkwise_kda_backward_kernel(batch_cfg; qg, kg, kpos_u_dw, w, aqk, g, chunk_states, d_out, dh_states_or_kneg, local_grad));
        {
            let v_new = kda_v_new.unwrap_or(&*kneg_vnew_dqg_dv);
            launch!(bwd_tc.chunk_kda_dkg_from_vnew_dh_kernel(chunk_cfg; v_new, dh_states_or_kneg, dkg_from_state));
        }
        launch!(bwd_tc.chunk_kda_dw_from_du_state_kernel(chunk_cfg; kpos_u_dw, chunk_states, w_du_dq));
        launch!(bwd_tc.chunk_kda_dqg_from_dout_state_kernel(chunk_cfg; dout_daqk_dvbeta, chunk_states, kneg_vnew_dqg_dv));
        launch!(bwd_elementwise.make_kda_backward_kneg_from_kg_kernel(linear_config(dims.compact_elems, threads); kg, g, dh_states_or_kneg));
        mm!(f32_input(
            local_grad,
            dh_states_or_kneg,
            dout_daqk_dvbeta,
            dims.chc()
        ));
        launch!(bwd_elementwise.add_kda_compact_kernel(linear_config(dims.compact_elems, threads); kneg_vnew_dqg_dv, dout_daqk_dvbeta));
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
        launch!(bwd_tc.chunk_intra_kda_dm_kernel(chunk_cfg; kg, vbeta, g, beta, kpos_u_dw, w_du_dq, aqk_or_dm));
        {
            let akk_inv = kda_akk_inv.unwrap_or(&*chunk_matrix);
            mm!(f32_input(aqk_or_dm, akk_inv, local_grad, dims.ccc()));
            mm!(f32_a_transposed_rhs(
                akk_inv,
                local_grad,
                aqk_or_dm,
                dims.ccc()
            ));
        }
        launch!(bwd_elementwise.make_kda_strict_neg_matrix_kernel(linear_config(dims.chunk_matrix_elems, threads); aqk_or_dm, chunk_matrix));
        launch!(bwd_elementwise.make_kda_backward_kneg_from_kg_kernel(linear_config(dims.compact_elems, threads); kg, g, kpos_u_dw));
        mm!(f32_rhs(
            chunk_matrix,
            kpos_u_dw,
            d_kneg_from_inverse,
            dims.chc()
        ));
        launch!(bwd_elementwise.make_kda_backward_kpos_from_kg_kernel(linear_config(dims.compact_elems, threads); kg, g, beta, kpos_u_dw));
        mm!(f32_a_transposed_rhs(
            chunk_matrix,
            kpos_u_dw,
            local_grad,
            dims.chc()
        ));
        launch!(bwd_tc.chunk_intra_kda_backward_kernel(chunk_cfg; qg, kg, vbeta, g, beta, kneg_vnew_dqg_dv, dkg_from_state, dka_dg, dh_states_or_kneg, dout_daqk_dvbeta, d_kneg_from_inverse, local_grad, w_du_dq, chunk_matrix, d_beta));
        // After chunk_intra_kda_backward_kernel these reused buffers hold final compact gradients.
        launch!(bwd_elementwise.finish_kda_backward_kernel(linear_config(dims.batch_head * seq_len * 32, threads); qkv, w_du_dq, chunk_matrix, kneg_vnew_dqg_dv, dka_dg, d_beta, d_qkv));
        Ok(())
    }
}
