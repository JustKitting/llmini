use std::sync::{Arc, Mutex};

use burn::data::dataloader::DataLoader;
use burn::train::logger::FileMetricLogger;
use burn::train::{Interrupter, Learner, SupervisedTraining, TrainingStrategy};

use super::CudaLearningComponents;
use super::burn_shim::{
    BurnBackend, BurnInnerBackend, CudaBurnModel, CudaNoopOptimizer, CudaTrainInput, CudaValidInput,
};
use super::config::TrainConfig;
use super::data_loader::{CudaTrainDataLoader, CudaValidDataLoader};
use super::loss_plot::write_loss_plot;
use super::metrics::register_cuda_metrics;
use super::output::{RunOutput, build_run_info};
use super::render::{BoxedMetricsRenderer, default_renderer};
use super::strategy::CudaTrainingStrategy;
use crate::AppResult;
use crate::training::data::{VALIDATION_WINDOWS, validation_target_byte_count};
use crate::training::{TokenDataLoader, debug_metrics};

pub(crate) fn launch_from_env() -> AppResult {
    let (dataset, data) = TokenDataLoader::from_training_dataset()?;
    let config = TrainConfig::from_env();
    let interrupter = Interrupter::new();
    let run_label = format!("{}s", config.max_seconds.round() as u64);
    let run_output = RunOutput::new(&dataset, &run_label)?;
    let metrics_dir = run_output.path("metrics");
    println!("run_dir={}", run_output.dir().display());
    println!("metrics_dir={}", metrics_dir.display());

    let training_tokens = data.token_count();
    let validation_tokens = data.validation_tokens()?;
    let validation_target_bytes = validation_target_byte_count(&validation_tokens)?;

    run_output.write_info(&build_run_info(&dataset, &config))?;
    println!(
        "training_tokens={} max_seconds={:.3} step_cap={}",
        training_tokens, config.max_seconds, config.step_cap
    );
    println!(
        "validation_tokens={} validation_target_bytes={validation_target_bytes}",
        VALIDATION_WINDOWS * gpt2_nvfp4::GPT2_SEQ_LEN,
    );

    let train_loader: Arc<dyn DataLoader<BurnBackend, CudaTrainInput>> =
        Arc::new(CudaTrainDataLoader::new(data, config.step_cap));
    let valid_loader: Arc<dyn DataLoader<BurnInnerBackend, CudaValidInput>> =
        Arc::new(CudaValidDataLoader::new(
            validation_tokens,
            VALIDATION_WINDOWS,
            validation_target_bytes,
        ));
    let strategy_result = Arc::new(Mutex::new(None));
    let strategy = Arc::new(CudaTrainingStrategy::new(
        dataset,
        config,
        run_output.clone(),
        Arc::clone(&strategy_result),
    ));

    let training = SupervisedTraining::<CudaLearningComponents>::new(
        run_output.dir(),
        train_loader,
        valid_loader,
    )
    .with_interrupter(interrupter.clone())
    .with_metric_logger(FileMetricLogger::new(metrics_dir))
    .with_application_logger(None)
    .renderer(BoxedMetricsRenderer::new(default_renderer(interrupter)))
    .with_training_strategy(TrainingStrategy::Custom(strategy));
    let training = debug_metrics::register_burn_metrics(register_cuda_metrics(training));
    let burn_device = Default::default();
    let learner = Learner::new(CudaBurnModel::new(&burn_device), CudaNoopOptimizer, 0.0);
    let _learning_result = training.launch(learner);
    if let Some(path) = write_loss_plot(&run_output)? {
        println!("loss_plot={}", path.display());
    }

    match strategy_result.lock().unwrap().take() {
        Some(Ok(())) => Ok(()),
        Some(Err(err)) => Err(err.into()),
        None => Err("Burn custom training strategy did not report a result".into()),
    }
}
