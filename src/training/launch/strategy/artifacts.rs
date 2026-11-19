use super::super::config::{generate_prompt, generate_tokens, sampling_config};
use super::super::output::{RunOutput, ensure_parent, save_model_path, write_generated_text};
use crate::AppResult;
use crate::training::Trainer;

pub(super) fn finish_training_artifacts(
    trainer: &mut Trainer,
    dataset: &str,
    train_elapsed_s: f64,
    run_output: &RunOutput,
) -> AppResult {
    if let Some(path) = save_model_path(run_output) {
        ensure_parent(&path)?;
        trainer.save_model(&path)?;
        println!("saved_model={}", path.display());
    }

    if let Some(prompt) = generate_prompt(dataset, train_elapsed_s) {
        let text = trainer.generate_sampled(&prompt, generate_tokens(), sampling_config())?;
        let generated_path = write_generated_text(run_output, &text)?;
        println!("generated_text={}", generated_path.display());
        println!("generated_text_begin");
        println!("{text}");
        println!("generated_text_end");
    }

    Ok(())
}
