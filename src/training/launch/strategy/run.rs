use burn::train::{
    EventProcessorTraining, Interrupter, LearnerEvent, SupervisedTrainingEventProcessor,
    TrainLoader, ValidLoader,
};

use super::{
    CudaTrainingStrategy, TRAIN_EPOCH,
    artifacts::finish_training_artifacts,
    budget::WallClockBudget,
    events::process_train_step,
    validation::{process_validation, validation_input},
};
use crate::AppResult;
use crate::training::launch::{
    CudaLearningComponents,
    config::{load_model_path, should_eval_step, should_log_step},
};
use crate::training::{Trainer, debug_metrics};

impl CudaTrainingStrategy {
    pub(super) fn run_training(
        &self,
        dataloader_train: TrainLoader<CudaLearningComponents>,
        dataloader_valid: ValidLoader<CudaLearningComponents>,
        processor: &mut SupervisedTrainingEventProcessor<CudaLearningComponents>,
        interrupter: &Interrupter,
    ) -> AppResult {
        let mut trainer = Trainer::new(self.config.seed)?;
        if let Some(path) = load_model_path() {
            trainer.load_model(&path)?;
            println!("loaded_model={}", path.display());
        }

        let validation = validation_input(dataloader_valid)?;
        let mut train_batch = trainer.reusable_default_batch()?;
        let mut debug_logger = debug_metrics::DebugTraceLogger::new(self.run_output.path("debug"))?;
        let wall_clock = WallClockBudget::new(self.config.max_seconds);
        let mut completed_steps = 0usize;
        let mut train_iter = dataloader_train.iter();

        while let Some(item) = train_iter.next() {
            let step = completed_steps;
            let log_step = should_log_step(step, self.config.step_cap, self.config.log_interval);
            let window = item.map_err(|err| format!("training dataloader failed: {err}"))?;
            let batch = trainer.upload_default_batch(&mut train_batch, &window.tokens)?;
            let stats = trainer.train_step(batch, log_step)?;
            completed_steps = step + 1;

            if log_step {
                process_train_step(
                    &mut debug_logger,
                    processor,
                    step,
                    &window,
                    stats,
                    train_iter.progress(),
                )?;
            }

            if should_eval_step(step, self.config.step_cap, self.config.eval_interval) {
                process_validation(&mut trainer, processor, &validation, step, completed_steps)?;
            }

            if let Some(reason) = stop_reason(interrupter, &wall_clock) {
                println!(
                    "stopped_by_{reason}=true elapsed_s={:.3} completed_steps={completed_steps}",
                    wall_clock.elapsed_seconds(),
                );
                break;
            }
        }
        let train_elapsed_s = wall_clock.elapsed_seconds();

        let final_eval = process_validation(
            &mut trainer,
            processor,
            &validation,
            completed_steps.saturating_sub(1),
            completed_steps,
        )?;
        processor.process_train(LearnerEvent::EndEpoch(TRAIN_EPOCH));
        processor.process_valid(LearnerEvent::EndEpoch(TRAIN_EPOCH));
        println!(
            "heldout_eval split=val val_loss={:.6} val_bits_per_byte={:.6} val_target_tokens={} val_target_bytes={} train_elapsed_s={:.3} eval_elapsed_s={:.3} completed_steps={completed_steps}",
            final_eval.val_loss,
            final_eval.val_bits_per_byte,
            final_eval.target_token_count,
            final_eval.target_byte_count,
            train_elapsed_s,
            final_eval.eval_elapsed_s,
        );
        if let Some(summary) = crate::training::symexp_lin::scale_summary(&trainer)? {
            println!("{summary}");
        }

        finish_training_artifacts(
            &mut trainer,
            &self.dataset,
            train_elapsed_s,
            &self.run_output,
        )
    }
}

fn stop_reason(interrupter: &Interrupter, wall_clock: &WallClockBudget) -> Option<&'static str> {
    if interrupter.should_stop() {
        Some("burn_interrupter")
    } else if wall_clock.expired() {
        Some("wall_clock")
    } else {
        None
    }
}
