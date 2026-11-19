use super::*;
use crate::training::runtime::Runtime;
use gpt2_nvfp4::{GPT2_N_EMBD, HiddenStateDevice};
impl Mhc {
    pub fn forward_boundary(
        &mut self,
        index: usize,
        before: bool,
        r: &Runtime,
        w: &Weights,
        h: &mut HiddenStateDevice<'_>,
    ) -> Result<bool, DriverError> {
        let s = r.stream.as_ref();
        let n = h.row_count;
        let d = GPT2_N_EMBD as u32;
        let slot = self.replay_windows.tape_slot(index);
        if before {
            self.n = n;
            let input = if slot == 0 {
                self.tape_units = [None; WINDOW];
                &mut self.checkpoints[self.replay_windows.checkpoint_slot(index)].input
            } else {
                &mut self.replay_inputs[slot - 1]
            };
            if index == 0 {
                self.active_window = None;
                #[cfg(test)]
                if let Some(v) = &mut self.recompute_verification {
                    v.reset();
                }
                self.module.mhc_expand_kernel(
                    s,
                    grid(n * WIDTH as u32),
                    h.residual,
                    input,
                    n,
                    d,
                )?;
            } else if index == BOUNDARIES {
                copy(s, &self.current, input, n as usize * WIDTH)?;
            } else {
                // A new window saves this buffer directly. The displaced
                // checkpoint or replay buffer becomes the next post destination.
                std::mem::swap(&mut self.current, input);
            }
            self.prepare_tape(index, r, w, false)?;
            let input = self
                .replay_windows
                .input(&self.checkpoints, &self.replay_inputs, index);
            self.module.mhc_pre_kernel(
                s,
                grid(n * d),
                input,
                &self.projections[index].coefficients,
                h.residual,
                n,
                d,
            )?;
            if index == BOUNDARIES {
                self.active_window = Some(self.replay_windows.range(index).start);
                #[cfg(test)]
                if self.recompute_verification.is_some() {
                    // The fixture checks this window by rebuilding it too.
                    self.active_window = None;
                }
            }
            Ok(index < BOUNDARIES)
        } else {
            debug_assert_eq!(self.tape_units[slot], Some(index));
            let branch = self.units[index].branch.as_mut().expect("branch post map");
            // Keep this result for backward. The next pre map overwrites
            // the recycled buffer before the next branch reads it.
            std::mem::swap(h.residual, branch);
            let input = self
                .replay_windows
                .input(&self.checkpoints, &self.replay_inputs, index);
            post_mix(
                self.post_reuse.as_ref(),
                &self.module,
                PostArgs {
                    stream: s,
                    input,
                    branch,
                    coefficients: &self.projections[index].coefficients,
                    out: &mut self.current,
                    rows: n,
                    dim: d,
                },
            )?;
            Ok(false)
        }
    }
}
