use std::time::Instant;

use burn::train::{SupervisedTrainingEventProcessor, ValidLoader};

use super::super::super::Trainer;
use super::super::{
    CudaLearningComponents, data_loader::CudaValidationInput, metrics::CudaValidOutput,
};
use super::events::process_valid_step;
use crate::AppResult;

pub(super) fn process_validation(
    trainer: &mut Trainer,
    processor: &mut SupervisedTrainingEventProcessor<CudaLearningComponents>,
    validation: &CudaValidationInput,
    step: usize,
    completed_steps: usize,
) -> AppResult<CudaValidOutput> {
    let eval_start = Instant::now();
    trainer.materialize_evaluation_weights()?;
    let val_loss = trainer.eval_loss_windows(&validation.tokens, validation.window_count)?;
    let target_token_count = validation.window_count * gpt2_nvfp4::GPT2_SEQ_LEN;
    let val_bits_per_byte = f64::from(val_loss) * target_token_count as f64
        / validation.target_byte_count as f64
        / std::f64::consts::LN_2;
    let output = CudaValidOutput {
        val_loss,
        val_bits_per_byte,
        eval_elapsed_s: eval_start.elapsed().as_secs_f64(),
        window_count: validation.window_count,
        target_token_count,
        target_byte_count: validation.target_byte_count,
        completed_steps,
    };
    Ok(process_valid_step(processor, step, output))
}

pub(super) fn validation_input(
    dataloader_valid: ValidLoader<CudaLearningComponents>,
) -> AppResult<CudaValidationInput> {
    match dataloader_valid.iter().next() {
        Some(Ok(input)) => Ok(input),
        Some(Err(err)) => Err(format!("validation dataloader failed: {err}").into()),
        None => Err("validation dataloader produced no windows".into()),
    }
}
