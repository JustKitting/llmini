use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use llama2_tokenizer::{TOKENIZER_VOCAB_SIZE, TrainingTokenizer};
use rust_kernels_cuda::logits::{LOGITS_TOP_K, LogitsTopKArgs};

use super::{TokenBatch, Trainer};
use crate::AppResult;

mod sampling;

use sampling::sample_top_k;

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Trainer {
    pub fn generate_sampled(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        config: SamplingConfig,
    ) -> AppResult<String> {
        let tokenizer = TrainingTokenizer::from_default_assets()?;
        let mut tokens = tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            tokens.push(tokenizer.bos_token());
        }

        let stream = self.runtime.stream.clone();
        let top_k = config.top_k.clamp(1, LOGITS_TOP_K);
        let mut top_tokens_dev = DeviceBuffer::<u32>::zeroed(stream.as_ref(), top_k)?;
        let mut top_logits_dev = DeviceBuffer::<f32>::zeroed(stream.as_ref(), top_k)?;

        let bos_token = tokenizer.bos_token();
        let eos_token = tokenizer.eos_token();

        for _ in 0..max_new_tokens {
            let (windows, row) = generation_batch(&tokens, bos_token)?;
            let batch = TokenBatch::from_default_batch(stream.as_ref(), &windows)?;
            self.forward_step(&batch)?;
            self.runtime.logits.top_k(LogitsTopKArgs {
                stream: stream.as_ref(),
                logits: &self.buffers.logits,
                out_tokens: &mut top_tokens_dev,
                out_values: &mut top_logits_dev,
                row,
                vocab_size: TOKENIZER_VOCAB_SIZE as u32,
                k: top_k as u32,
            })?;
            let top_tokens = top_tokens_dev.to_host_vec(stream.as_ref())?;
            let top_logits = top_logits_dev.to_host_vec(stream.as_ref())?;
            let next = sample_top_k(
                &top_tokens,
                &top_logits,
                config.temperature,
                config.top_p,
                &mut self.rng,
            );
            tokens.push(next);
            if next == eos_token {
                break;
            }
        }

        tokenizer.decode(&tokens)
    }
}

fn generation_batch(tokens: &[u32], pad_token: u32) -> AppResult<(Vec<u16>, u32)> {
    let context_len = tokens.len().min(GPT2_SEQ_LEN);
    let context_start = tokens.len() - context_len;
    let row = context_len.saturating_sub(1) as u32;
    let window_len = GPT2_SEQ_LEN + 1;
    let pad = u16::try_from(pad_token)?;
    let mut one = vec![pad; window_len];

    for (dst, &token) in one.iter_mut().zip(tokens[context_start..].iter()) {
        *dst = u16::try_from(token)?;
    }

    Ok((one.repeat(GPT2_BATCH_SIZE), row))
}
