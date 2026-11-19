//! Rebuild complete mHC windows from exact FP32 stream checkpoints and saved
//! heavy branch outputs. Projectors and packing retain their native NVFP4
//! forward; no backward seed is consumed and no heavy branch is re-executed.
use super::*;
use crate::training::runtime::Runtime;
use gpt2_nvfp4::GPT2_N_EMBD;
use rust_kernels_cuda::mhc_window_post::PostWindowArgs;
impl Mhc {
    pub(super) fn prepare_tape(
        &mut self,
        index: usize,
        r: &Runtime,
        w: &Weights,
        _reconstructed: bool,
    ) -> Result<(), DriverError> {
        let s = r.stream.as_ref();
        let n = self.n;
        let slot = tape_slot(index);
        let stream_input = boundary_input(&self.checkpoints, &self.replay_inputs, index);
        let projection = &mut self.projections[index];
        let weights = &w.units[index];
        let packed = projection.input_q.scratch();
        self.rms_quant
            .forward(rust_kernels_cuda::mhc_rms_quant::RmsQuantArgs {
                stream: s,
                input: stream_input,
                inverse: &mut projection.inverse,
                amax: &mut self.row_amax,
                payload: packed.bytes,
                scales: packed.scales,
                global_scales: packed.global_scales,
                rows: n,
            })?;
        let input = projection.input_q.rowwise();
        r.tma_gemm.prepare_tma_nvfp4_csf_weight_into(
            s,
            input.bytes,
            input.scales,
            &weights.phi.bytes,
            &weights.phi.scales,
            n,
            WIDTH as u32,
            PAD as u32,
            &mut self.tma,
        )?;
        r.tma_gemm
            .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf(
                s,
                &self.tma,
                &mut projection.raw,
                n,
                WIDTH as u32,
                PAD as u32,
                input.global_scales,
                &weights.phi.global_scale,
                None,
            )?;
        self.coefficients
            .forward(rust_kernels_cuda::mhc_coefficients::ForwardArgs {
                stream: s,
                raw: &projection.raw,
                bias: &weights.bias.weight,
                alpha: &weights.alpha.weight,
                out: &mut projection.coefficients,
                rows: n,
                stride: PAD as u32,
                head: weights.head,
            })?;
        self.tape_units[slot] = Some(index);
        #[cfg(test)]
        if let Some(verification) = &mut self.recompute_verification {
            verification.observe(
                s,
                index,
                _reconstructed,
                stream_input,
                projection,
                n as usize,
            )?;
        }
        Ok(())
    }
    pub(super) fn ensure_window(
        &mut self,
        index: usize,
        r: &Runtime,
        _w: &Weights,
    ) -> Result<(), DriverError> {
        let start = window_start(index);
        if self.active_window == Some(start) {
            return Ok(());
        }
        let s = r.stream.as_ref();
        let checkpoint = &self.checkpoints[checkpoint_slot(start)];
        self.tape_units = [None; WINDOW];
        let end = window_end(start);
        let fused = self.window_post.as_ref().filter(|_| end - start == WINDOW);
        if let Some(fused) = fused {
            let [first, second, third] = self.replay_inputs.as_mut_slice() else {
                unreachable!("mHC allocates three replay inputs")
            };
            fused.rebuild(PostWindowArgs {
                stream: s,
                input: &checkpoint.input,
                branches: std::array::from_fn(|i| {
                    self.units[start + i]
                        .branch
                        .as_ref()
                        .expect("branch successor")
                }),
                coefficients: std::array::from_fn(|i| &self.projections[start + i].coefficients),
                out: [first, second, third],
                rows: self.n,
                dim: GPT2_N_EMBD as u32,
            })?;
        }
        for boundary in start..end {
            let slot = tape_slot(boundary);
            self.tape_units[slot] = Some(boundary);
            #[cfg(test)]
            if let Some(verification) = &mut self.recompute_verification {
                verification.observe(
                    s,
                    boundary,
                    true,
                    boundary_input(&self.checkpoints, &self.replay_inputs, boundary),
                    &self.projections[boundary],
                    self.n as usize,
                )?;
            }
            if fused.is_none() && boundary + 1 < end {
                let (previous, next) = self.replay_inputs.split_at_mut(slot);
                let input = boundary_input(&self.checkpoints, previous, boundary);
                // `current` holds the reverse stream gradient here. Rebuild
                // directly into the next tape so that gradient survives.
                post_mix(
                    self.post_reuse.as_ref(),
                    &self.module,
                    PostArgs {
                        stream: s,
                        input,
                        branch: self.units[boundary]
                            .branch
                            .as_ref()
                            .expect("branch successor"),
                        coefficients: &self.projections[boundary].coefficients,
                        out: &mut next[0],
                        rows: self.n,
                        dim: GPT2_N_EMBD as u32,
                    },
                )?;
            }
        }
        self.active_window = Some(start);
        Ok(())
    }
}
