use std::path::Path;

use super::optimizer_muon::MuonPointerTables;
use super::optimizer_state::OptimizerStateBuffers;
use super::Trainer;
use crate::checkpoint::{load_uploaded_model, save_uploaded_model};
use crate::AppResult;

impl Trainer {
    pub fn save_model(&self, path: &Path) -> AppResult {
        self.ensure_ready()?;
        save_uploaded_model(self.runtime.stream.as_ref(), &self.uploaded, path)
    }

    pub fn load_model(&mut self, path: &Path) -> AppResult {
        let stream = self.runtime.stream.as_ref();
        let uploaded =
            if let Some(reload) = crate::checkpoint::prepare_reload(&mut self.uploaded, path)? {
                stream.synchronize()?;
                self.architecture_ready = false;
                reload.apply(stream)?;
                None
            } else {
                Some(load_uploaded_model(stream, path)?)
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
        let engram = self
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
        let mhc = self
            .uploaded
            .mhc
            .as_ref()
            .map(|w| super::mhc::Mhc::new(&self.runtime, w))
            .transpose()?;
        let muon_tables = MuonPointerTables::new_with_architectures(
            stream,
            &self.uploaded,
            &self.buffers.backward,
            &self.buffers.next_latent_grads,
            &self.buffers.optimizer_state,
            engram.as_ref(),
            latent_feedback.as_ref(),
            mhc.as_ref(),
        )?;
        let mut extra = engram
            .as_ref()
            .map_or_else(Vec::new, |e| e.named_gradients());
        if let Some(f) = &latent_feedback {
            extra.extend(f.named_gradients());
        }
        if let Some(m) = &mhc {
            extra.extend(m.named_gradients());
        }
        let recompute = mhc.as_ref().map_or_else(Vec::new, |m| {
            m.units.iter().map(|u| u.grad.cu_deviceptr()).collect()
        });
        self.buffers.grad_clip =
            super::grad_clip::GradientClipBuffers::new_with_recomputed_matrices(
                stream,
                &self.buffers.backward,
                &self.buffers.next_latent_grads,
                &muon_tables,
                &extra,
                &recompute,
            )?;
        self.engram = engram;
        self.latent_feedback = latent_feedback;
        self.mhc = mhc;
        self.buffers.muon_tables = muon_tables;
        self.architecture_ready = true;
        Ok(())
    }
}
