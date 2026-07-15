use cuda_core::sys;
use gpt2_nvfp4::{GPT2_SEQ_LEN, Gpt2, Gpt2Rng};

use super::{ReusableTokenBatch, TokenBatch, buffers, runtime::Runtime};
use crate::{AppResult, upload::UploadedModel};

pub struct Trainer {
    pub(in crate::training) runtime: Runtime,
    pub(in crate::training) model: Gpt2,
    pub(in crate::training) uploaded: UploadedModel,
    pub(in crate::training) buffers: buffers::TrainBuffers,
    pub(in crate::training) rng: Gpt2Rng,
}

impl Trainer {
    pub fn new(seed: u64) -> AppResult<Self> {
        let runtime = Runtime::new()?;
        let stream = runtime.stream.as_ref();
        let mut model = Gpt2::new();
        model.init(seed);
        let weights = model.weights().expect("Gpt2::init must create weights");

        let uploaded = UploadedModel::new(stream, weights)?;
        let buffers = buffers::TrainBuffers::new(stream, &runtime, &uploaded)?;

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
            uploaded,
            buffers,
            runtime,
            model,
            rng: Gpt2Rng::new(seed ^ 0xa047_0a91),
        })
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
