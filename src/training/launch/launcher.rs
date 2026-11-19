use std::sync::{Arc, Mutex};

use burn::data::dataloader::DataLoader;
use burn::train::logger::FileMetricLogger;
use burn::train::{Interrupter, Learner, SupervisedTraining, TrainingStrategy};
use clap::Parser;

use super::CudaLearningComponents;
use super::burn_shim::{
    BurnBackend, BurnInnerBackend, CudaBurnModel, CudaNoopOptimizer, CudaTrainInput, CudaValidInput,
};
use super::cli::TrainArgs;
use super::data_loader::{CudaTrainDataLoader, CudaValidDataLoader};
use super::loss_plot::write_loss_plot;
use super::metrics::register_cuda_metrics;
use super::output::{RunOutput, build_run_info};
use super::render::{BoxedMetricsRenderer, default_renderer};
use super::strategy::CudaTrainingStrategy;
use crate::AppResult;
use crate::training::data::{VALIDATION_WINDOWS, validation_target_byte_count};
use crate::training::{TokenDataLoader, debug_metrics};

pub(crate) fn launch_from_args() -> AppResult {
    // Validate arguments before loading a dataset or initializing CUDA.
    let args = TrainArgs::parse();
    if let Some(checkpoint) = args.inspect_feedback_depths.as_deref() {
        return crate::training::feedback_depth_inspection::run(
            checkpoint,
            args.feedback_depth_output.as_deref(),
            args.replay_provenance.as_deref(),
        );
    }
    if let Some(checkpoint) = args.inspect_gradients.as_deref() {
        return crate::training::gradient_inspection::run(
            checkpoint,
            args.gradient_output.as_deref(),
            args.replay_provenance.as_deref(),
            args.gradient_shard.as_deref(),
            args.gradient_token_offset,
            args.gradient_rng_attempt,
        );
    }
    if let Some(checkpoint) = args.inspect_model.as_deref() {
        return crate::training::activation_inspection::run(
            checkpoint,
            args.inspection_output.as_deref(),
            args.replay_provenance.as_deref(),
            args.inspect_kda_decay,
        );
    }
    if let Some(checkpoint) = args.evaluate_model.as_deref() {
        if let Some(answers) = args.validation_answers.as_deref() {
            return crate::training::reference_eval::run_answers(
                checkpoint,
                answers,
                args.evaluation_output.as_deref(),
                args.replay_provenance.as_deref(),
            );
        }
        return crate::training::reference_eval::run(
            checkpoint,
            args.validation_documents
                .as_deref()
                .expect("clap requires validation documents"),
            args.evaluation_output.as_deref(),
            args.replay_provenance.as_deref(),
            args.feedback_evaluation_passes.map(usize::from),
        );
    }
    let config = args.config();
    if args.bounded_swiglu_engram_mhc != gpt2_nvfp4::GPT2_BOUNDED_SWIGLU_ENGRAM_MHC {
        return Err("select --bounded-swiglu-engram-mhc --full-kda-backward with the matching GPT2_MLP_KIND=bounded-swiglu GPT2_KDA_BACKWARD=full native build".into());
    }
    if args.value_rms_engram_mhc != gpt2_nvfp4::GPT2_VALUE_RMS_ENGRAM_MHC {
        return Err("select --value-rms-engram-mhc --full-kda-backward with the matching GPT2_VALUE_RMS=1 GPT2_KDA_BACKWARD=full native build".into());
    }
    if args.stationary_rope_engram_mhc != gpt2_nvfp4::GPT2_STATIONARY_ROPE_ENGRAM_MHC {
        return Err("select --stationary-rope-engram-mhc --full-kda-backward with the matching GPT2_POSITION_KIND=stationary-rope GPT2_KDA_BACKWARD=full native build".into());
    }
    if (args.stationary_rope || args.stationary_rope_engram_mhc) != gpt2_nvfp4::GPT2_STATIONARY_ROPE
    {
        return Err("select --stationary-rope with the matching GPT2_POSITION_KIND=stationary-rope native build".into());
    }
    if args.stationary_rope {
        if config.mhc
            || config.engram_mhc
            || config.latent_feedback
            || std::env::var("TRAIN_ENGRAM").is_ok_and(|s| s == "1")
        {
            return Err("stationary RoPE selects a separate complete native trial".into());
        }
        if super::config::load_model_path().is_some() {
            return Err("stationary RoPE requires fresh model, optimizer, RNG and warmup state; TRAIN_LOAD_MODEL is not a continuation".into());
        }
    }
    if gpt2_nvfp4::GPT2_STATIONARY_ROPE
        && super::config::env_bool("TRAIN_PARTIAL_KEY_OFFSET") == Some(false)
    {
        return Err("stationary RoPE requires its stationary-half key offset".into());
    }
    if config.engram_table_params.is_some() || config.engram_four_grams {
        crate::training::engram::Config::multistream_with_ngram_budget(
            config.engram_table_params,
            config.engram_four_grams,
        )?;
        if super::config::load_model_path().is_some() {
            return Err("an explicit Engram table budget or four-gram lookup requires fresh model, optimizer, RNG and warmup state; TRAIN_LOAD_MODEL is not a continuation".into());
        }
    }
    if args.full_kda_backward != gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
        return Err(
            "select --full-kda-backward with the matching GPT2_KDA_BACKWARD=full native build"
                .into(),
        );
    }
    if args.full_kda_backward
        && (config.latent_feedback
            || (!config.engram_mhc
                && (config.mhc || std::env::var("TRAIN_ENGRAM").is_ok_and(|s| s == "1"))))
    {
        return Err("full recurrent KDA credit supports the standalone backbone or complete --engram-mhc graph".into());
    }
    if args.full_kda_backward && super::config::load_model_path().is_some() {
        return Err("full recurrent KDA trial requires fresh model, optimizer, RNG and warmup state; TRAIN_LOAD_MODEL is not a continuation".into());
    }
    if (args.value_rms || args.value_rms_engram_mhc) != gpt2_nvfp4::GPT2_VALUE_RMS {
        return Err("select --value-rms with the matching GPT2_VALUE_RMS=1 native build".into());
    }
    if args.value_rms
        && (config.mhc
            || config.engram_mhc
            || config.latent_feedback
            || std::env::var("TRAIN_ENGRAM").is_ok_and(|s| s == "1"))
    {
        return Err("value RMS selects a separate complete native architecture trial".into());
    }
    if (args.bounded_swiglu || args.bounded_swiglu_engram_mhc) != gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
        return Err("select --bounded-swiglu with a GPT2_MLP_KIND=bounded-swiglu native build; the default build and graph use ReLU-squared".into());
    }
    if args.bounded_swiglu
        && !args.bounded_swiglu_engram_mhc
        && (config.mhc
            || config.engram_mhc
            || config.latent_feedback
            || std::env::var("TRAIN_ENGRAM").is_ok_and(|s| s == "1"))
    {
        return Err("bounded-SwiGLU selects a separate complete native architecture trial".into());
    }
    if config.mhc && config.latent_feedback {
        return Err(
            "mHC and latent feedback currently select separate native training graphs".into(),
        );
    }
    if config.mhc && !config.engram_mhc && std::env::var("TRAIN_ENGRAM").is_ok_and(|s| s == "1") {
        return Err("four-stream mHC cannot use the single-stream Engram hook; unset TRAIN_ENGRAM or select TRAIN_ENGRAM=0".into());
    }
    let (dataset, data, manifest_info) = if let Some(path) = args.training_shards.as_deref() {
        let (data, sha256) = TokenDataLoader::from_training_manifest(path)?;
        ("manifest".to_owned(), data, Some((path, sha256)))
    } else {
        let (dataset, data) = match args.dataset.as_deref() {
            Some(dataset) => TokenDataLoader::from_dataset(dataset)?,
            None => TokenDataLoader::from_training_dataset()?,
        };
        (dataset, data, None)
    };
    let interrupter = Interrupter::new();
    let run_label = format!("{}s", config.max_seconds.round() as u64);
    let run_output = RunOutput::new(&dataset, &run_label)?;
    let metrics_dir = run_output.path("metrics");
    println!("run_dir={}", run_output.dir().display());
    println!("metrics_dir={}", metrics_dir.display());

    let training_tokens = data.token_count();
    let validation_tokens = data.validation_tokens()?;
    let validation_target_bytes = validation_target_byte_count(&validation_tokens)?;

    let mut info = build_run_info(&dataset, &config);
    if let Some((path, sha256)) = manifest_info {
        info.push_str(&format!(
            "training_shards_manifest={}\ntraining_shards_manifest_sha256={sha256}\nvalidation_dataset=fineweb\n",
            path.display()
        ));
    }
    run_output.write_info(&info)?;
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
