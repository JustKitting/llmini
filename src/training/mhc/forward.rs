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
        let slot = tape_slot(index);
        if before {
            self.n = n;
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
                    &mut self.tapes[slot].input,
                    n,
                    d,
                )?;
            } else if index == BOUNDARIES {
                copy(
                    s,
                    &self.current,
                    &self.tapes[slot].input,
                    n as usize * WIDTH,
                )?;
            } else {
                // The previous post map produced this boundary's input. The
                // reused slot is dead; make it the next post map's destination
                // without copying the complete four-stream activation.
                std::mem::swap(&mut self.current, &mut self.tapes[slot].input);
            }
            if index == window_start(index) {
                self.tape_units = [None; WINDOW];
                let checkpoint = self
                    .checkpoints
                    .iter()
                    .find(|c| c.index == index)
                    .expect("mHC window checkpoint exists");
                copy(
                    s,
                    &self.tapes[slot].input,
                    &checkpoint.input,
                    n as usize * WIDTH,
                )?;
            }
            self.prepare_tape(index, r, w, false)?;
            let tape = &self.tapes[slot];
            self.module.mhc_pre_kernel(
                s,
                grid(n * d),
                &tape.input,
                &self.projections[index].coefficients,
                h.residual,
                n,
                d,
            )?;
            if index == BOUNDARIES {
                self.active_window = Some(window_start(index));
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
            copy(s, h.residual, branch, (n * d) as usize)?;
            let tape = &self.tapes[slot];
            post_mix(
                self.post_reuse.as_ref(),
                &self.module,
                PostArgs {
                    stream: s,
                    input: &tape.input,
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
