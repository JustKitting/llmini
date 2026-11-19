use std::path::Path;

use super::Trainer;
use super::optimizer_state::OptimizerStateBuffers;
use crate::AppResult;
use crate::checkpoint::{load_uploaded_model, save_uploaded_model};

impl Trainer {
    pub fn save_model(&self, path: &Path) -> AppResult {
        self.ensure_ready()?;
        save_uploaded_model(
            self.runtime.stream.as_ref(),
            &self.uploaded,
            self.layer_layout(),
            path,
        )
    }

    pub fn load_model(&mut self, path: &Path) -> AppResult {
        let stream = self.runtime.stream.as_ref();
        let (uploaded, layout) =
            if let Some(reload) = crate::checkpoint::prepare_reload(&mut self.uploaded, path)? {
                stream.synchronize()?;
                self.architecture_ready = false;
                (None, reload.apply(stream)?)
            } else {
                let (model, layout) = load_uploaded_model(stream, path)?;
                (Some(model), layout)
            };
        // Multi-stream tapes are large. Release the old optional graph before
        // constructing the replacement; a failed partial load stays unusable.
        stream.synchronize()?;
        self.architecture_ready = false;
        self.mhc = None;
        self.engram = None;
        self.latent_feedback = None;
        self.buffers.optimizer_state = OptimizerStateBuffers::new(
            stream,
            &self.runtime.decode,
            &self.runtime.optimizer,
            uploaded.as_ref().unwrap_or(&self.uploaded),
        )?;
        if let Some(uploaded) = uploaded {
            self.uploaded = uploaded;
        }
        let depth = layout.depth;
        self.buffers.optimizer_state.active_depth = depth;
        self.inherit_growth_block_policy = layout.inherit_block_policy;
        let mut engram = self
            .uploaded
            .engram
            .as_ref()
            .map(|w| super::engram::Engram::new(&self.runtime, w))
            .transpose()?;
        let latent_feedback = self
            .uploaded
            .latent_feedback
            .as_ref()
            .map(|w| super::latent_feedback::Feedback::new(&self.runtime, w))
            .transpose()?;
        let mut mhc = self
            .uploaded
            .mhc
            .as_ref()
            .map(|w| super::mhc::Mhc::new(&self.runtime, w))
            .transpose()?;
        if let Some(engram) = &mut engram {
            engram.active_depth = depth;
        }
        if let Some(mhc) = &mut mhc {
            mhc.active_boundaries = depth * 2;
        }
        self.buffers.configure_optimizer_graph(
            stream,
            &self.uploaded,
            engram.as_ref(),
            latent_feedback.as_ref(),
            mhc.as_ref(),
        )?;
        self.engram = engram;
        self.latent_feedback = latent_feedback;
        self.mhc = mhc;
        self.architecture_ready = true;
        Ok(())
    }
}
