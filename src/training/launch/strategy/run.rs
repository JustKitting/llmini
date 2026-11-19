use burn::train::{
    EventProcessorTraining, Interrupter, LearnerEvent, SupervisedTrainingEventProcessor,
    TrainLoader, ValidLoader,
};
use gpt2_nvfp4::{GPT2_N_LAYER, depth_growth::LayerLayout};
use std::time::Instant;

use super::{
    CudaTrainingStrategy, TRAIN_EPOCH,
    artifacts::finish_training_artifacts,
    budget::WallClockBudget,
    events::process_train_step,
    validation::{process_validation, validation_input},
};
use crate::AppResult;
use crate::training::launch::{
    CudaLearningComponents, TrainConfig,
    config::{load_model_path, should_eval_step, should_log_step},
};
use crate::training::{Trainer, debug_metrics, engram, mhc};

impl TrainConfig {
    pub(in crate::training) fn create_trainer(&self) -> AppResult<Trainer> {
        let layout = self.growth_plan()?.map_or_else(
            || LayerLayout::from(GPT2_N_LAYER),
            |plan| LayerLayout::grown(plan.depth(0)),
        );
        if gpt2_nvfp4::GPT2_REQUIRES_ENGRAM_MHC && !self.engram_mhc {
            return Err(
                "this composite full-credit build requires the complete Engram/mHC factory".into(),
            );
        }
        if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
            && (self.latent_feedback || (self.mhc && !self.engram_mhc))
        {
            return Err("full recurrent KDA credit supports the standalone backbone or complete four-stream Engram/mHC graph".into());
        }
        let (feedback, mhc, multi_memory) = if self.engram_mhc {
            (false, true, true)
        } else if gpt2_nvfp4::GPT2_STATIONARY_ROPE
            || gpt2_nvfp4::GPT2_VALUE_RMS
            || gpt2_nvfp4::GPT2_BOUNDED_SWIGLU
            || gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
        {
            (false, false, false)
        } else if self.mhc {
            (false, true, false)
        } else if self.latent_feedback {
            (true, mhc::enabled(), false)
        } else {
            let memory = engram::multistream_enabled();
            (false, memory || mhc::enabled(), memory)
        };
        Trainer::new_architecture_at_layout(
            self.seed,
            feedback,
            mhc,
            multi_memory,
            if self.engram_mhc {
                self.engram_table_params
            } else {
                None
            },
            layout,
        )
    }
}

impl CudaTrainingStrategy {
    pub(super) fn run_training(
        &self,
        dataloader_train: TrainLoader<CudaLearningComponents>,
        dataloader_valid: ValidLoader<CudaLearningComponents>,
        processor: &mut SupervisedTrainingEventProcessor<CudaLearningComponents>,
        interrupter: &Interrupter,
    ) -> AppResult {
        crate::training::diagnostics::kda_clip::initialize(self.run_output.dir())?;
        let gradient_interval =
            crate::training::env::env_usize("TRAIN_GRADIENT_COMPONENT_INTERVAL")
                .filter(|&interval| interval > 0);
        let mut gradient_log = gradient_interval
            .map(|_| std::fs::File::create(self.run_output.path("gradient_components.jsonl")))
            .transpose()?;
        let growth = self.config.growth_plan()?;
        let mut trainer = self.config.create_trainer()?;
        if growth.is_some() {
            let depth = trainer.active_depth();
            println!("depth_growth active_depth={depth} completed_attempts=0");
        }
        if let Some(path) = load_model_path() {
            trainer.load_model(&path)?;
            if trainer.active_depth() != gpt2_nvfp4::GPT2_N_LAYER {
                return Err("a partial growth checkpoint stores evaluation weights without a live optimizer".into());
            }
            println!("loaded_model={}", path.display());
        }

        let validation = validation_input(dataloader_valid)?;
        let mut train_batch = trainer.reusable_default_batch()?;
        let mut debug_logger = debug_metrics::DebugTraceLogger::new(self.run_output.path("debug"))?;
        let mut wall_clock = WallClockBudget::new(self.config.max_seconds);
        let mut completed_steps = 0usize;
        let mut train_iter = dataloader_train.iter();

        while let Some(item) = train_iter.next() {
            let step = completed_steps;
            if let Some(plan) = growth {
                let depth = plan.depth(step);
                if depth != trainer.active_depth() {
                    trainer.select_training_depth(depth)?;
                    println!("depth_growth active_depth={depth} completed_attempts={step}");
                }
            }
            let log_step = should_log_step(step, self.config.step_cap, self.config.log_interval);
            let window = item.map_err(|err| format!("training dataloader failed: {err}"))?;
            let batch = trainer.upload_default_batch(&mut train_batch, &window.tokens)?;
            let stats = trainer.train_step(batch, log_step)?;
            completed_steps = step + 1;

            let numerical_failure = super::super::config::env_bool("TRAIN_STOP_ON_NONFINITE")
                .unwrap_or(false)
                && (!stats.finite || stats.optimizer.skip_non_finite);
            if numerical_failure {
                eprintln!("numerical_failure_capture_started completed_steps={completed_steps}");
                // Capture before publishing the bad metric: external monitors must
                // not terminate the CUDA context while its state is being read.
                trainer.write_numerical_failure_dump(
                    batch,
                    &self.run_output.path("numerical_failure_state.tsv"),
                )?;
                trainer.save_model(&self.run_output.path("numerical_failure_weights.ckpt"))?;
                eprintln!("numerical_failure_capture_complete completed_steps={completed_steps}");
            }

            if let (Some(interval), Some(file)) = (gradient_interval, gradient_log.as_mut()) {
                if completed_steps == 1
                    || completed_steps % interval == 0
                    || completed_steps == self.config.step_cap
                    || stats.optimizer.update_skipped
                {
                    use std::io::Write;
                    let mut snapshot = trainer.completed_gradient_components(
                        completed_steps,
                        stats.optimizer.grad_norm,
                    )?;
                    snapshot["source_file"] = serde_json::json!(window.source);
                    snapshot["window_offset"] = serde_json::json!(window.offset);
                    snapshot["update_skipped"] = serde_json::json!(stats.optimizer.update_skipped);
                    snapshot["skip_grad_norm_spike"] =
                        serde_json::json!(stats.optimizer.skip_grad_norm_spike);
                    snapshot["skip_loss_spike"] =
                        serde_json::json!(stats.optimizer.skip_loss_spike);
                    snapshot["skip_non_finite"] =
                        serde_json::json!(stats.optimizer.skip_non_finite);
                    snapshot["capture_reason"] =
                        serde_json::json!(if stats.optimizer.update_skipped {
                            "rejected_update"
                        } else {
                            "periodic"
                        });
                    serde_json::to_writer(&mut *file, &snapshot)?;
                    writeln!(file)?;
                    file.flush()?;
                    // Accounting results stay in the diagnostic record. Actual
                    // nonfinite training uses the stop-and-capture path above.
                }
            }

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

            if numerical_failure {
                processor.process_train(LearnerEvent::EndEpoch(TRAIN_EPOCH));
                return Err(format!(
                    "nonfinite training at attempted step {completed_steps}; state captured"
                )
                .into());
            }

            if should_eval_step(step, self.config.step_cap, self.config.eval_interval) {
                let eval_start = Instant::now();
                process_validation(
                    &mut trainer,
                    processor,
                    &validation,
                    step,
                    completed_steps,
                    self.config.eval_repeats,
                )?;
                wall_clock.exclude(eval_start.elapsed());
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
            self.config.eval_repeats,
        )?;
        processor.process_train(LearnerEvent::EndEpoch(TRAIN_EPOCH));
        processor.process_valid(LearnerEvent::EndEpoch(TRAIN_EPOCH));
        println!(
            "heldout_eval split=val val_loss={:.6} val_loss_min={:.6} val_loss_max={:.6} val_loss_spread={:.6} val_bits_per_byte={:.6} val_target_tokens={} val_target_bytes={} train_elapsed_s={:.3} eval_elapsed_s={:.3} eval_repeats={} completed_steps={completed_steps}",
            final_eval.val_loss,
            final_eval.val_loss_min,
            final_eval.val_loss_max,
            final_eval.val_loss_spread,
            final_eval.val_bits_per_byte,
            final_eval.target_token_count,
            final_eval.target_byte_count,
            train_elapsed_s,
            final_eval.eval_elapsed_s,
            self.config.eval_repeats,
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
