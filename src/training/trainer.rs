use std::path::Path;

use cuda_core::sys;
use gpt2_nvfp4::{GPT2_N_LAYER, GPT2_SEQ_LEN, Gpt2, Gpt2Rng};

use super::{ReusableTokenBatch, TokenBatch, buffers, runtime::Runtime};
use crate::{AppResult, upload::UploadedModel};

pub struct Trainer {
    #[cfg(test)]
    pub(in crate::training) update_rejection_audit: Option<super::update_rejection_tests::Audit>,
    #[cfg(test)]
    pub(in crate::training) value_projection_audit: Option<super::value_rms_tests::ValueAudit>,
    #[cfg(test)]
    pub(in crate::training) bounded_projection_audit:
        Option<super::bounded_swiglu_tests::RecomputeAudit>,
    pub(in crate::training) mhc: Option<super::mhc::Mhc>,
    pub(in crate::training) architecture_ready: bool,
    pub(in crate::training) engram: Option<super::engram::Engram>,
    pub(in crate::training) latent_feedback: Option<super::latent_feedback::Feedback>,
    pub(in crate::training) base_seed: u64,
    pub(in crate::training) runtime: Runtime,
    pub(in crate::training) model: Gpt2,
    pub(in crate::training) uploaded: UploadedModel,
    pub(in crate::training) buffers: buffers::TrainBuffers,
    pub(in crate::training) rng: Gpt2Rng,
}

impl Trainer {
    pub fn new(seed: u64) -> AppResult<Self> {
        let memory = gpt2_nvfp4::GPT2_REQUIRES_ENGRAM_MHC || super::engram::multistream_enabled();
        Self::new_architecture(seed, false, memory || super::mhc::enabled(), memory)
    }

    /// Open saved evaluation weights directly, before allocating trainer state.
    pub(in crate::training) fn from_checkpoint(seed: u64, path: &Path) -> AppResult<Self> {
        let runtime = Runtime::new()?;
        let (uploaded, depth) = crate::checkpoint::load_uploaded_model(&runtime.stream, path)?;
        let mut model = Gpt2::new();
        model.init(seed);
        Self::with_uploaded_model(seed, runtime, model, uploaded, depth)
    }

    pub fn new_feedback(seed: u64) -> AppResult<Self> {
        Self::new_architecture(seed, true, super::mhc::enabled(), false)
    }

    pub fn new_mhc(seed: u64) -> AppResult<Self> {
        Self::new_architecture(seed, false, true, false)
    }
    pub fn new_bounded_swiglu(seed: u64) -> AppResult<Self> {
        if !gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
            return Err(
                "bounded SwiGLU requires the native GPT2_MLP_KIND=bounded-swiglu build".into(),
            );
        }
        Self::new_architecture(seed, false, false, false)
    }
    pub fn new_value_rms(seed: u64) -> AppResult<Self> {
        if !gpt2_nvfp4::GPT2_VALUE_RMS {
            return Err("value RMS requires the native GPT2_VALUE_RMS=1 build".into());
        }
        Self::new_architecture(seed, false, false, false)
    }
    pub fn new_full_kda(seed: u64) -> AppResult<Self> {
        if !gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
            return Err("full KDA backward requires a GPT2_KDA_BACKWARD=full native build".into());
        }
        Self::new_architecture(seed, false, false, false)
    }
    pub fn new_engram_mhc(seed: u64) -> AppResult<Self> {
        Self::new_architecture(seed, false, true, true)
    }
    pub fn new_stationary_rope(seed: u64) -> AppResult<Self> {
        if !gpt2_nvfp4::GPT2_STATIONARY_ROPE {
            return Err(
                "stationary RoPE requires a GPT2_POSITION_KIND=stationary-rope native build".into(),
            );
        }
        Self::new_architecture(seed, false, false, false)
    }
    pub fn new_engram_mhc_with_table_budget(seed: u64, requested: usize) -> AppResult<Self> {
        super::engram::Config::multistream_with_table_budget(Some(requested))?;
        Self::new_architecture_with_table_budget(seed, false, true, true, Some(requested), false)
    }
    pub fn new_engram_mhc_with_four_grams(seed: u64, requested: Option<usize>) -> AppResult<Self> {
        super::engram::Config::multistream_with_ngram_budget(requested, true)?;
        Self::new_architecture_with_table_budget(seed, false, true, true, requested, true)
    }
    fn new_architecture(
        seed: u64,
        feedback: bool,
        use_mhc: bool,
        multi_memory: bool,
    ) -> AppResult<Self> {
        Self::new_architecture_with_table_budget(seed, feedback, use_mhc, multi_memory, None, false)
    }
    fn new_architecture_with_table_budget(
        seed: u64,
        feedback: bool,
        use_mhc: bool,
        multi_memory: bool,
        table_budget: Option<usize>,
        four_grams: bool,
    ) -> AppResult<Self> {
        let bounded_combined = gpt2_nvfp4::GPT2_BOUNDED_SWIGLU_ENGRAM_MHC;
        if gpt2_nvfp4::GPT2_REQUIRES_ENGRAM_MHC && !(use_mhc && multi_memory && !feedback) {
            return Err("the composite full-credit build requires the complete four-stream Engram/mHC graph".into());
        }
        if ((!bounded_combined && gpt2_nvfp4::GPT2_BOUNDED_SWIGLU)
            || (gpt2_nvfp4::GPT2_VALUE_RMS && !gpt2_nvfp4::GPT2_VALUE_RMS_ENGRAM_MHC)
            || (gpt2_nvfp4::GPT2_STATIONARY_ROPE && !gpt2_nvfp4::GPT2_STATIONARY_ROPE_ENGRAM_MHC))
            && (feedback
                || use_mhc
                || multi_memory
                || std::env::var("TRAIN_ENGRAM").is_ok_and(|s| s == "1"))
        {
            return Err("bounded SwiGLU, value RMS and stationary RoPE each select separate complete native trials".into());
        }
        if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
            && (feedback || use_mhc || multi_memory || super::engram::enabled())
            && !(use_mhc && multi_memory && !feedback)
        {
            return Err("full recurrent KDA credit supports the standalone backbone or complete four-stream Engram/mHC graph".into());
        }
        if feedback && super::engram::multistream_enabled() {
            return Err(
                "four-stream Engram and latent feedback select separate native graphs".into(),
            );
        }
        if use_mhc
            && (feedback
                || (!multi_memory && std::env::var("TRAIN_ENGRAM").is_ok_and(|s| s == "1")))
        {
            return Err("select full mHC as a separate native graph with TRAIN_ENGRAM=0 and no latent feedback; the single-stream memory hook is not a four-stream integration".into());
        }
        let runtime = Runtime::new()?;
        let mut model = Gpt2::new();
        model.init(seed);
        let weights = model.weights().expect("Gpt2::init must create weights");

        let mut uploaded = UploadedModel::new_with_engram(
            &runtime,
            weights,
            !feedback
                && !use_mhc
                && !gpt2_nvfp4::GPT2_BOUNDED_SWIGLU
                && !gpt2_nvfp4::GPT2_VALUE_RMS
                && !gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
                && !gpt2_nvfp4::GPT2_STATIONARY_ROPE
                && super::engram::enabled(),
        )?;
        if feedback {
            uploaded.latent_feedback = Some(super::latent_feedback::Weights::new(&runtime, seed)?);
        }
        if use_mhc {
            uploaded.mhc = Some(super::mhc::Weights::new(&runtime, seed)?);
        }
        if multi_memory {
            uploaded.engram = Some(if four_grams {
                super::engram::Weights::new_multistream_with_ngram_budget(
                    &runtime,
                    table_budget,
                    true,
                )?
            } else if table_budget.is_some() {
                super::engram::Weights::new_multistream_with_table_budget(&runtime, table_budget)?
            } else {
                super::engram::Weights::new_multistream(&runtime)?
            });
        }
        Self::with_uploaded_model(seed, runtime, model, uploaded, GPT2_N_LAYER)
    }

    fn with_uploaded_model(
        seed: u64,
        runtime: Runtime,
        model: Gpt2,
        uploaded: UploadedModel,
        depth: usize,
    ) -> AppResult<Self> {
        let stream = runtime.stream.as_ref();
        let mut buffers = buffers::TrainBuffers::new(stream, &runtime, &uploaded)?;
        buffers.optimizer_state.active_depth = depth;
        let mut engram = uploaded
            .engram
            .as_ref()
            .map(|weights| super::engram::Engram::new(&runtime, weights))
            .transpose()?;
        let latent_feedback = uploaded
            .latent_feedback
            .as_ref()
            .map(|w| super::latent_feedback::Feedback::new(&runtime, w))
            .transpose()?;
        let mut mhc = uploaded
            .mhc
            .as_ref()
            .map(|w| super::mhc::Mhc::new(&runtime, w))
            .transpose()?;
        if let Some(engram) = &mut engram {
            engram.active_depth = depth;
        }
        if let Some(mhc) = &mut mhc {
            mhc.active_boundaries = depth * 2;
        }
        if depth != GPT2_N_LAYER || engram.is_some() || latent_feedback.is_some() || mhc.is_some() {
            buffers.configure_optimizer_graph(
                stream,
                &uploaded,
                engram.as_ref(),
                latent_feedback.as_ref(),
                mhc.as_ref(),
            )?;
        }

        if std::env::var_os("TRAIN_REPORT_MEMORY").is_some() {
            stream.synchronize()?;
            let mut free_bytes = 0usize;
            let mut total_bytes = 0usize;
            let result = unsafe { sys::cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes) };
            if result != 0 {
                return Err(cuda_core::DriverError(result).into());
            }
            println!(
                "device_memory total_bytes={total_bytes} free_bytes={free_bytes} used_bytes={}",
                total_bytes - free_bytes
            );
        }

        Ok(Self {
            #[cfg(test)]
            update_rejection_audit: None,
            #[cfg(test)]
            value_projection_audit: None,
            #[cfg(test)]
            bounded_projection_audit: None,
            mhc,
            architecture_ready: true,
            latent_feedback,
            base_seed: seed,
            engram,
            uploaded,
            buffers,
            runtime,
            model,
            rng: Gpt2Rng::new(seed ^ 0xa047_0a91),
        })
    }

    pub(in crate::training) fn ensure_ready(&self) -> AppResult {
        if !self.architecture_ready {
            return Err("native model loading did not complete; reload a valid checkpoint before using the trainer".into());
        }
        Ok(())
    }

    pub fn reusable_default_batch(&self) -> AppResult<ReusableTokenBatch> {
        ReusableTokenBatch::default(self.runtime.stream.as_ref())
    }

    pub fn upload_default_batch<'a>(
        &self,
        batch: &'a mut ReusableTokenBatch,
        tokens: &[u16],
    ) -> AppResult<&'a TokenBatch> {
        batch.upload(self.runtime.stream.as_ref(), tokens)
    }

    pub fn batch_from_windows(&self, tokens: &[u16], batch_size: usize) -> AppResult<TokenBatch> {
        TokenBatch::from_flat_windows(
            self.runtime.stream.as_ref(),
            tokens,
            batch_size,
            GPT2_SEQ_LEN,
        )
    }
}
