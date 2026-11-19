use cuda_core::DriverError;

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::kda::KDA_INTRA_THREADS_PER_BLOCK;
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
        assert_eq!(
            args.seq_len % KDA_HEAD_DIM,
            0,
            "KDA path requires full 64-token chunks"
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
            q: vnew_half,
            k: w_half,
            v: aqk_half,
            d_out: dout_half,
            scores: _chunk_matrix,
            dot: daqk_storage,
            p: local_grad,
            ds: dkg_from_state,
            p_half: _p_half,
            ds_half: inverse_half,
            d_q: kneg_vnew_dqg_dv,
            d_k: kpos_u_dw,
            d_v: w_du_dq,
            kda_d_q: dka_dg,
            kda_d_k: d_kneg_from_inverse,
            kda_d_v: dout_daqk_dvbeta,
            kda_d_g: dh_states_or_kneg,
            kda_d_beta: state_decay_grad,
        } = scratch;
        // p_half retains zero masked entries for the final full-attention
        // layer's tile-mass sampling across iterations. Its bytes are not dead.
        // Dot has no live FP32 value in KDA, so dAQK owns native-half bytes here.
        let mut daqk_half = unsafe {
            kda_launch::BorrowedHalfScratch::new(&*daqk_storage, dims.chunk_matrix_elems as usize)
        };
        let bwd_elementwise = &self.causal_attention_backward_tc.kda_elementwise;
        let bwd_tc = &self.causal_attention_backward_tc.kda_tc;
        let fwd = &self.causal_attention_tc.kda;
        let threads = TC_BACKWARD_THREADS_PER_BLOCK;
        let chunkwise_cfg = grid_x_config(dims.batch_head, CTA_ULTRA_WIDE_THREADS);
        let intra_chunk_cfg =
            kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, KDA_INTRA_THREADS_PER_BLOCK);
        let cumsum_cfg = kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, KDA_HEAD_DIM);
        let tc_chunk_cfg =
            kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, KDA_WIDE_THREADS_PER_BLOCK);
        let finish_element_count = dims.batch_head * seq_len * 32;
        let finish_chunk_count = finish_element_count.div_ceil(threads);
        assert_eq!(
            finish_chunk_count,
            dims.chunk_batch * (threads / 32),
            "KDA fused backward amax layout requires full 64-token chunks"
        );
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

        // These upper halves are dead before preparation. Native Kpos/Kneg
        // remain read-only through the fused producer-to-scalar kernel.
        let mut kneg_half = unsafe {
            kda_launch::BorrowedHalfScratch::range(
                &*local_grad,
                dims.compact_elems as usize,
                dims.compact_elems as usize,
            )
        };
        let mut kpos_half = unsafe {
            kda_launch::BorrowedHalfScratch::range(
                &*w_du_dq,
                dims.compact_elems as usize,
                dims.compact_elems as usize,
            )
        };
        let has_cached_forward_state =
            kda_v_new.is_some() && kda_akk_inv.is_some() && kda_w.is_some() && kda_aqk.is_some();
        if has_cached_forward_state {
            launch!(bwd_elementwise.prepare_transform_kda_backward_cached_kernel(kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, KDA_WIDE_THREADS_PER_BLOCK); qkv, qg, kg, vbeta, g, beta, &mut *kpos_half, &mut *kneg_half));
        } else {
            launch!(bwd_elementwise.prepare_kda_backward_inputs_kernel(linear_config(dims.batch_head * seq_len * 32, threads); qkv, qg, kg, vbeta, g, beta));
            launch!(bwd_elementwise.chunk_cumsum_kda_backward_g_kernel(cumsum_cfg; g));
            launch!(fwd.make_kda_qg_kneg_kg_kpos_vbeta_kernel(linear_config(dims.compact_elems, threads); qg, kg, vbeta, g, beta, kneg_vnew_dqg_dv, kpos_u_dw, &mut *kpos_half, &mut *kneg_half));
        }
        if kda_akk_inv.is_none() {
            mm.fused_inverse(kpos_u_dw, kneg_vnew_dqg_dv, &mut *inverse_half, dims.cch())?;
        }
        let w = {
            let akk_inv = kda_akk_inv.unwrap_or(&*inverse_half);
            match kda_w {
                Some(w) => w,
                None => {
                    mm.half_a_f32_rhs_half_out(akk_inv, kpos_u_dw, &mut *w_half, dims.chc())?;
                    &*w_half
                }
            }
        };
        // kneg must remain live until AQK is reconstructed below. Native FP16
        // v_new uses the otherwise idle half buffer instead of overwriting it.
        if kda_v_new.is_none() {
            let akk_inv = kda_akk_inv.unwrap_or(&*inverse_half);
            mm!(half_a_f32_rhs(akk_inv, vbeta, kpos_u_dw, dims.chc()));
            launch!(bwd_tc.chunk_kda_vnew_from_state_kernel(tc_chunk_cfg; w, kpos_u_dw, chunk_states, &mut *vnew_half));
        }
        let aqk = match kda_aqk {
            Some(aqk) => aqk,
            None => {
                mm.f32_causal_half_out(qg, kneg_vnew_dqg_dv, &mut *aqk_half, dims.cch())?;
                &*aqk_half
            }
        };
        launch!(bwd_elementwise.gather_kda_dout_kernel(linear_config(dims.compact_elems, threads); d_out, &mut *dout_half));
        {
            let v_new = kda_v_new.unwrap_or(&*vnew_half);
            mm!(half_causal_half_out(
                &*dout_half,
                v_new,
                &mut *daqk_half,
                dims.cch()
            ));
        }
        // All three views occupy existing arenas only after their previous
        // values die. Exported gradients are born as native half; the two
        // recurrent carries and all scalar gradient math stay FP32.
        let mut du_half = unsafe {
            kda_launch::BorrowedHalfScratch::new(&*kpos_u_dw, dims.compact_elems as usize)
        };
        let mut dw_half =
            unsafe { kda_launch::BorrowedHalfScratch::new(&*w_du_dq, dims.compact_elems as usize) };
        let mut dh_half = unsafe {
            kda_launch::BorrowedHalfScratch::new(&*dh_states_or_kneg, dims.compact_elems as usize)
        };
        if detach_chunk_state {
            // The actual trainer uses this API: keep the projection parallel
            // across chunks and consume its native dU in the same CTA.
            launch!(bwd_tc.chunk_kda_project_du_dw_dqg_kernel(tc_chunk_cfg; aqk, &*dout_half, chunk_states, &mut *du_half, &mut *dw_half, kneg_vnew_dqg_dv));
        } else {
            launch!(bwd_tc.chunkwise_kda_backward_kernel(chunkwise_cfg; qg, kg, &mut *du_half, w, g, aqk, &*dout_half, &mut *dh_half));
            // dG_last also receives the direct exp(G_last)*H_entry carry.
            // Preserve it before dH's arena becomes the inverse backprojection.
            let fix = self.full_state_decay_module()?;
            assert!(state_decay_grad.len() >= (dims.chunk_batch * head_dim) as usize);
            let capture_cfg = kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads);
            launch!(fix.capture_full_state_decay_grad_kernel(capture_cfg; chunk_states, &*dh_half, g, &mut *state_decay_grad));
            let v_new = kda_v_new.unwrap_or(&*vnew_half);
            launch!(bwd_tc.chunk_kda_dkg_from_vnew_dh_kernel(tc_chunk_cfg; v_new, &*dh_half, dkg_from_state));
            launch!(bwd_tc.chunk_kda_dw_dqg_from_state_kernel(tc_chunk_cfg; &*du_half, &*dout_half, chunk_states, &mut *dw_half, kneg_vnew_dqg_dv));
        }
        // dAQK[token,source] multiplies Kneg[source,feature]. The generic
        // half_input_accumulate API takes B-transposed storage and silently
        // computes the wrong product here when both axes happen to be64.
        // Stage the native RHS with its actual layout in both backward modes.
        let repair = self.full_state_decay_module()?;
        let query_cfg =
            kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, KDA_WIDE_THREADS_PER_BLOCK);
        launch!(repair.accumulate_kda_dqg_rhs_kernel(query_cfg; &*daqk_half, &*kneg_half, kneg_vnew_dqg_dv));
        mm!(half_a_transposed_f32_rhs(
            &*daqk_half,
            qg,
            dka_dg,
            dims.chc()
        ));
        {
            let akk_inv = kda_akk_inv.unwrap_or(&*inverse_half);
            mm.half_a_transposed_dual_rhs(
                akk_inv,
                &*dw_half,
                &*du_half,
                dh_states_or_kneg,
                dout_daqk_dvbeta,
                dims.chc(),
            )?;
        }
        // dM and both inverse products stay in one native-half shared arena.
        // Both backprojections remain FP32 inside the same CTA and become
        // its prefix scratch in place. Q norms use dead dAQK/dot bytes and K
        // norms use dead dedicated dK bytes. Both are disjoint from Kpos/Kneg;
        // writing them into M here would race another chunk's live M reads.
        let akk_inv = kda_akk_inv.unwrap_or(&*inverse_half);
        launch!(bwd_tc.chunk_intra_kda_backward_kernel(intra_chunk_cfg; qkv, qg, kg, vbeta, g, beta, kneg_vnew_dqg_dv, dkg_from_state, dka_dg, dh_states_or_kneg, dout_daqk_dvbeta, akk_inv, &*du_half, &*dw_half, &*kneg_half, &*kpos_half, daqk_storage, d_kneg_from_inverse, d_qkv, d_qkv_chunk_amax, u32::from(accumulate_value_grad), u32::from(detach_chunk_state)));
        if !detach_chunk_state {
            let fix = self.full_state_decay_module()?;
            let correction_cfg =
                kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads);
            launch!(fix.add_full_state_decay_grad_and_amax_kernel(correction_cfg; qkv, &*state_decay_grad, &mut *d_qkv, &mut *d_qkv_chunk_amax));
        }
        // The fused intra kernel stores the final Q/K norms in its otherwise-dead
        // compact Q/K scratch buffers after producing dQKV directly.
        bwd_elementwise.reduce_kda_qk_norm_max_kernel(
            stream,
            grid_x_config(head_count, KDA_NORM_REDUCE_THREADS_PER_BLOCK),
            &*daqk_storage,
            &*d_kneg_from_inverse,
            qk_norm_max,
            qk_norm_offset,
            params.row_count,
            head_count,
        )?;
        Ok(finish_chunk_count)
    }
}
