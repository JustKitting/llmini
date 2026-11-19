use gpt2_nvfp4::{
    GPT2_N_LAYER, KIMI_FULL_ATTENTION_PERIOD as GROWTH,
    depth_growth::{GrowthTransition, LayerLayout},
};

use super::{
    Trainer,
    optimizer_state::relocate::{copy_fp32, copy_muon},
};
use crate::AppResult;

impl Trainer {
    pub(in crate::training) fn active_depth(&self) -> usize {
        self.buffers.optimizer_state.active_depth
    }

    pub(in crate::training) fn layer_layout(&self) -> LayerLayout {
        LayerLayout {
            depth: self.active_depth(),
            inherit_block_policy: self.inherit_growth_block_policy,
        }
    }

    /// Growth is monotone after the initial prefix is selected. The same
    /// trainer keeps its applied update, running history, averaging and RNG.
    pub(in crate::training) fn select_training_depth(&mut self, depth: usize) -> AppResult {
        self.ensure_ready()?;
        let previous = self.active_depth();
        if depth == previous {
            self.inherit_growth_block_policy |= depth == GROWTH;
            return Ok(());
        }
        if depth > GPT2_N_LAYER {
            return Err("growth depth exceeds the allocated model".into());
        }
        let initial_prefix =
            previous == GPT2_N_LAYER && depth == GROWTH && self.buffers.optimizer_state.step == 0;
        let transition = if initial_prefix {
            None
        } else {
            Some(GrowthTransition::new(previous, depth).ok_or("invalid native depth transition")?)
        };

        let stream = self.runtime.stream.as_ref();
        stream.synchronize()?;
        // If device copying or rebuilding the tables fails, stop before any
        // subsequent operation can read inconsistent optimizer ownership.
        self.architecture_ready = false;
        if let Some(transition) = transition {
            self.buffers
                .optimizer_state
                .copy_growth_blocks(stream, transition)?;
            if let Some(mhc) = &mut self.mhc {
                for (source, destination) in transition.copies() {
                    let (before, after) = mhc.units.split_at_mut(destination * 2);
                    for (from, to) in before[source * 2..source * 2 + 2]
                        .iter()
                        .zip(after.iter_mut().take(2))
                    {
                        copy_muon(stream, &from.phi, &mut to.phi)?;
                        for (from, to) in from.fp.iter().zip(&mut to.fp) {
                            copy_fp32(stream, from, to)?;
                        }
                    }
                }
            }
        }

        self.buffers.optimizer_state.active_depth = depth;
        self.inherit_growth_block_policy = true;
        if let Some(mhc) = &mut self.mhc {
            mhc.active_boundaries = depth * 2;
        }
        if let Some(engram) = &mut self.engram {
            engram.active_depth = depth;
        }
        self.buffers.configure_optimizer_graph(
            stream,
            &self.uploaded,
            self.engram.as_ref(),
            self.latent_feedback.as_ref(),
            self.mhc.as_ref(),
        )?;
        stream.synchronize()?;
        self.architecture_ready = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
