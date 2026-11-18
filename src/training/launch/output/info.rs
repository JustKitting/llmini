use gpt2_nvfp4::{
    GPT2_ATTENTION_BACKWARD_TILE_BUDGET, GPT2_BATCH_SIZE, GPT2_BOUNDED_SWIGLU,
    GPT2_FULL_ATTENTION_WINDOW, GPT2_MLP, GPT2_MLP_UP, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_N_LAYER,
    GPT2_SEQ_LEN, GPT2_TOKEN_ROWS, GPT2_VALUE_RESIDUAL_LAYERS, GPT2_VALUE_RESIDUAL_START_LAYER,
};
use rust_kernels_cuda::optimizer::{MUON_COOPERATIVE_BLOCKS, MUON_MATRIX_PHASES};

use super::super::TrainConfig;

pub(in crate::training::launch) fn build_run_info(dataset: &str, config: &TrainConfig) -> String {
    let mut info = String::new();
    push_info(&mut info, "dataset", dataset);
    push_info(&mut info, "training_launcher", "burn");
    push_info(&mut info, "metric_logger", "burn_file");
    push_info(&mut info, "muon_storage_layout", "physical_output_by_input");
    push_info(
        &mut info,
        "kda_output_gate",
        if gpt2_nvfp4::GPT2_KDA_RATIONAL_OUTPUT_GATE {
            "rational"
        } else {
            "sigmoid"
        },
    );
    push_info(
        &mut info,
        "nvfp4_activation_quantization",
        if rust_kernels_cuda::nvfp4_quant::ACTIVATION_MAE_FOUR_SIX {
            "mae-4-6"
        } else {
            "six"
        },
    );
    push_info(
        &mut info,
        "muon_storage_layout_version",
        rust_kernels_cuda::optimizer::MUON_STORAGE_LAYOUT_VERSION,
    );
    let gradient_interval = crate::training::env::env_usize("TRAIN_GRADIENT_COMPONENT_INTERVAL")
        .filter(|&interval| interval > 0);
    push_info(
        &mut info,
        "gradient_component_interval",
        gradient_interval.unwrap_or(0),
    );
    push_info(
        &mut info,
        "gradient_component_capture",
        if gradient_interval.is_some() {
            "periodic_and_rejected"
        } else {
            "disabled"
        },
    );
    push_info(&mut info, "tokenizer", llama2_tokenizer::TOKENIZER_NAME);
    push_info(
        &mut info,
        "tokenizer_vocab_size",
        llama2_tokenizer::TOKENIZER_VOCAB_SIZE,
    );
    push_info(
        &mut info,
        "model_vocab_size",
        llama2_tokenizer::MODEL_VOCAB_SIZE,
    );
    push_info(
        &mut info,
        "unused_model_vocab_rows",
        llama2_tokenizer::MODEL_VOCAB_SIZE - llama2_tokenizer::TOKENIZER_VOCAB_SIZE,
    );
    push_info(
        &mut info,
        "document_boundaries",
        llama2_tokenizer::DOCUMENT_BOUNDARIES,
    );
    push_info(&mut info, "gpt2_seq_len", GPT2_SEQ_LEN);
    push_info(&mut info, "gpt2_batch_size", GPT2_BATCH_SIZE);
    push_info(&mut info, "gpt2_token_rows", GPT2_TOKEN_ROWS);
    push_info(&mut info, "gpt2_n_layer", GPT2_N_LAYER);
    push_info(&mut info, "gpt2_n_head", GPT2_N_HEAD);
    push_info(&mut info, "gpt2_n_embd", GPT2_N_EMBD);
    push_info(
        &mut info,
        "KDA_local_query_adjoint",
        "native_token_feature_RHS",
    );
    push_info(
        &mut info,
        "KDA_backward",
        if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
            "full_recurrent_state_adjoint"
        } else {
            "detached_between_64_token_chunks"
        },
    );
    push_info(
        &mut info,
        "KDA_decay_adjoint",
        "direct_normalized_K_factors_and_exact_clamp_masks",
    );
    push_info(&mut info, "value_rms_enabled", gpt2_nvfp4::GPT2_VALUE_RMS);
    if gpt2_nvfp4::GPT2_VALUE_RMS {
        push_info(&mut info, "value_rms_epsilon", "1e-6");
        push_info(&mut info, "value_rms_precision_policy_version", 2);
        push_info(&mut info, "value_rms_forward_native_affine_input_bits", 32);
        push_info(&mut info, "value_rms_reverse_SiLU_input_bits", 16);
        push_info(&mut info, "value_rms_reverse_normalized_direction_bits", 16);
        push_info(&mut info, "value_rms_normalization_control_bits", 32);
        push_info(
            &mut info,
            "value_rms_order",
            "fixed_raw_share_then_KDA_SiLU_then_head_RMS_then_beta",
        );
        push_info(
            &mut info,
            "value_rms_tape",
            "normalized_V_with_exact_NVFP4_raw_projection_recomputation",
        );
        push_info(
            &mut info,
            "value_rms_shared_cache",
            "immutable_F32_raw_cache_with_separate_gradient",
        );
    }
    push_info(
        &mut info,
        "mlp_kind",
        if GPT2_BOUNDED_SWIGLU {
            "bounded-swiglu"
        } else {
            "relu2"
        },
    );
    push_info(
        &mut info,
        "mlp_initializer",
        gpt2_nvfp4::MlpWeights::initializer_name(),
    );
    push_info(&mut info, "mlp_initializer_weight_storage", "NVFP4");
    if gpt2_nvfp4::MlpWeights::initializer_name() == "reference-normal-nvfp4" {
        push_info(&mut info, "mlp_initializer_source_stddev", 0.02);
    }
    if GPT2_BOUNDED_SWIGLU {
        push_info(&mut info, "mlp_activation_input_bits", 16);
        push_info(
            &mut info,
            "mlp_activation_input_rounding",
            "rn_even_fp16_forward_and_recomputed_reverse",
        );
        push_info(
            &mut info,
            "mlp_input_quantizer_adjoint",
            "straight_through_at_represented_input",
        );
        push_info(&mut info, "mlp_gate_bound", 4);
        push_info(&mut info, "mlp_up_bound", 25);
        push_info(&mut info, "mlp_sigmoid_input", "raw_gate");
        push_info(&mut info, "mlp_derivative", "stable_analytic_all_factors");
        push_info(
            &mut info,
            "mlp_raw_paired_tape",
            "exact_native_projection_recomputation",
        );
        push_info(
            &mut info,
            "mlp_paired_optimizer",
            "one_common_Muon_SymExpLin_matrix",
        );
    }
    push_info(
        &mut info,
        "mlp_route_feature_tiles",
        gpt2_nvfp4::GPT2_MLP_ROUTE_FEATURE_TILES,
    );
    push_info(
        &mut info,
        "mlp_route_active_tiles",
        gpt2_nvfp4::GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES,
    );
    push_info(
        &mut info,
        "engram_enabled",
        // A validated explicit complete graph owns its memory selection.
        // Standalone exclusions apply to the implicit single-stream hook.
        config.engram_mhc
            || (!GPT2_BOUNDED_SWIGLU
                && !gpt2_nvfp4::GPT2_VALUE_RMS
                && !gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
                && !gpt2_nvfp4::GPT2_STATIONARY_ROPE
                && !config.latent_feedback
                && !config.mhc
                && crate::training::engram::enabled()),
    );
    push_info(
        &mut info,
        "engram_streams",
        if config.engram_mhc { 4 } else { 1 },
    );
    push_info(&mut info, "engram_mhc_enabled", config.engram_mhc);
    push_info(
        &mut info,
        "stationary_rope_enabled",
        gpt2_nvfp4::GPT2_STATIONARY_ROPE,
    );
    if gpt2_nvfp4::GPT2_STATIONARY_ROPE {
        push_info(&mut info, "position_rotating_dimensions", "0..32");
        push_info(&mut info, "position_stationary_dimensions", "32..64");
        push_info(&mut info, "position_frequency_base", 1024);
        push_info(
            &mut info,
            "position_frequency_endpoints",
            "inclusive_1_to_1/1024",
        );
        push_info(&mut info, "position_rotation_sign", "negative");
        push_info(&mut info, "position_key_offset", "stationary_half_only");
        push_info(
            &mut info,
            "position_norm_order",
            "native_rotation_then_QK_norm",
        );
        push_info(&mut info, "position_trigonometry", "native_FP32");
    }
    push_info(
        &mut info,
        "engram_keys_per_module",
        if config.engram_mhc { 4 } else { 1 },
    );
    push_info(&mut info, "engram_values_per_module", 1);
    push_info(
        &mut info,
        "engram_checkpoint_graph_version",
        if config.engram_mhc { 4 } else { 3 },
    );
    push_info(&mut info, "engram_implementation", "full_native_nvfp4");
    let memory_config = if config.engram_mhc {
        crate::training::engram::Config::multistream_with_table_budget(config.engram_table_params)
            .expect("launcher validates the Engram table budget before writing run info")
    } else {
        std::array::from_fn(crate::training::engram::Config::reference)
    };
    if let Some(requested) = config.engram_table_params {
        push_info(&mut info, "engram_table_parameters_requested", requested);
    }
    push_info(
        &mut info,
        "engram_parameters_total",
        memory_config
            .iter()
            .map(|c| c.trainable_parameters())
            .sum::<usize>(),
    );
    push_info(
        &mut info,
        "engram_module_count",
        crate::training::engram::LAYERS.len(),
    );
    push_info(&mut info, "engram_insert_before_blocks", "1,5");
    push_info(&mut info, "engram_embedding_adam_lr_multiplier", 5);
    push_info(
        &mut info,
        "engram_initialization",
        "reference_normal_table_uniform_linear_cuda_nvfp4_packing",
    );
    push_info(
        &mut info,
        "engram_insert_before_block",
        crate::training::engram::LAYER,
    );
    push_info(&mut info, "engram_ngram_orders", "2,3");
    push_info(
        &mut info,
        "engram_heads_per_order",
        crate::training::engram::HEADS,
    );
    push_info(
        &mut info,
        "engram_head_dimension",
        crate::training::engram::HEAD_DIM,
    );
    push_info(
        &mut info,
        "engram_memory_width",
        crate::training::engram::MEMORY,
    );
    push_info(
        &mut info,
        "engram_table_parameters",
        memory_config.iter().map(|c| c.table_len()).sum::<usize>(),
    );
    push_info(
        &mut info,
        "engram_convolution",
        "depthwise_kernel4_dilation3_rmsnorm_silu",
    );
    push_info(
        &mut info,
        "engram_optimizer",
        "native_Muon_SymExpLin_projections_Adam_AMUSE_lookup_bias_controls",
    );
    push_info(
        &mut info,
        "engram_checkpoint",
        "same_model_file_with_architecture_descriptor_and_compressed_id_map",
    );
    push_info(&mut info, "mlp_hidden", GPT2_MLP);
    push_info(&mut info, "mlp_up_dim", GPT2_MLP_UP);
    push_info(
        &mut info,
        "gpt2_value_residual_layers",
        GPT2_VALUE_RESIDUAL_LAYERS,
    );
    push_info(
        &mut info,
        "gpt2_value_residual_start_layer",
        GPT2_VALUE_RESIDUAL_START_LAYER,
    );
    push_info(
        &mut info,
        "gpt2_full_attention_window",
        GPT2_FULL_ATTENTION_WINDOW,
    );
    push_info(
        &mut info,
        "gpt2_attention_backward_tile_budget",
        GPT2_ATTENTION_BACKWARD_TILE_BUDGET,
    );
    push_info(
        &mut info,
        "nextlat_loss_weight",
        crate::training::next_latent::loss_weight(),
    );
    push_info(
        &mut info,
        "lr_scale",
        crate::training::learning_rate::scale(),
    );
    push_info(
        &mut info,
        "adam_lr_scale",
        crate::training::learning_rate::adam_scale(),
    );
    push_info(
        &mut info,
        "nextlat_lr_scale",
        crate::training::learning_rate::next_latent_scale(),
    );
    push_info(
        &mut info,
        "lr_warmup_steps",
        crate::training::learning_rate::warmup_steps(),
    );
    push_info(
        &mut info,
        "lr_start_ratio",
        crate::training::learning_rate::start_ratio(),
    );
    push_info(
        &mut info,
        "amuse_beta1",
        crate::training::learning_rate::amuse_beta1(),
    );
    push_info(
        &mut info,
        "amuse_rho",
        crate::training::learning_rate::amuse_rho(),
    );
    push_info(
        &mut info,
        "muon_cooperative_blocks",
        MUON_COOPERATIVE_BLOCKS,
    );
    push_info(&mut info, "muon_matrix_phases", MUON_MATRIX_PHASES);
    push_info(
        &mut info,
        "hyperball_enabled",
        crate::training::optimizer_muon::hyperball_enabled(),
    );
    push_info(
        &mut info,
        "hyperball_lr",
        crate::training::optimizer_muon::hyperball_learning_rate(),
    );
    push_info(
        &mut info,
        "hyperball_amuse",
        crate::training::optimizer_muon::hyperball_uses_schedule_free(),
    );
    push_info(
        &mut info,
        "hyperball_polar_period",
        crate::training::optimizer_muon::hyperball_polar_period(),
    );
    push_info(
        &mut info,
        "hyperball_polar_numerator",
        crate::training::optimizer_muon::hyperball_polar_numerator(),
    );
    push_info(
        &mut info,
        "muon_vs_enabled",
        crate::training::optimizer_muon::muon_vs_enabled(),
    );
    push_info(
        &mut info,
        "ember_enabled",
        crate::training::optimizer_apply::ember::enabled(),
    );
    push_info(
        &mut info,
        "ember_learning_rate",
        crate::training::optimizer_apply::ember::learning_rate(100),
    );
    push_info(
        &mut info,
        "ember_beta2",
        crate::training::optimizer_apply::ember::beta2(),
    );
    push_info(
        &mut info,
        "ember_weight_decay",
        crate::training::optimizer_apply::ember::weight_decay(),
    );
    push_info(
        &mut info,
        "symexp_lin_enabled",
        crate::training::symexp_lin::enabled(),
    );
    push_info(
        &mut info,
        "symexp_lin_beta",
        crate::training::symexp_lin::beta(),
    );
    push_info(
        &mut info,
        "symexp_lin_learn_scales",
        crate::training::symexp_lin::learn_scales(),
    );
    push_info(
        &mut info,
        "attention_headwise_gate_enabled",
        gpt2_nvfp4::attention_headwise_gate_enabled(),
    );
    push_info(
        &mut info,
        "exclusive_self_attention_enabled",
        gpt2_nvfp4::exclusive_self_attention_enabled(),
    );
    push_info(
        &mut info,
        "exclusive_self_attention_kda_enabled",
        gpt2_nvfp4::exclusive_self_attention_kda_enabled(),
    );
    push_info(
        &mut info,
        "partial_key_offset_enabled",
        gpt2_nvfp4::partial_key_offset_enabled(),
    );
    push_info(
        &mut info,
        "selective_attention_enabled",
        gpt2_nvfp4::selective_attention_enabled(),
    );
    push_info(
        &mut info,
        "stable_mask_gamma",
        gpt2_nvfp4::stable_mask_gamma(),
    );
    push_info(
        &mut info,
        "canon_ac_enabled",
        gpt2_nvfp4::canon_ac_enabled(),
    );
    push_info(&mut info, "step_cap", config.step_cap);
    push_info(&mut info, "log_interval", config.log_interval);
    push_info(&mut info, "max_seconds", config.max_seconds);
    let eval_interval = config
        .eval_interval
        .map_or_else(|| "none".to_string(), |value| value.to_string());
    push_info(&mut info, "eval_interval", eval_interval);
    push_info(&mut info, "eval_repeats", config.eval_repeats);
    push_info(&mut info, "mhc_enabled", config.mhc);
    if config.mhc {
        use crate::training::mhc::{BOUNDARIES, PAD, STREAMS, WIDTH};
        push_info(
            &mut info,
            "mhc_post_reuse",
            crate::training::mhc::post_reuse_enabled(),
        );
        push_info(
            &mut info,
            "mhc_window_post",
            crate::training::mhc::window_post_enabled(),
        );
        push_info(
            &mut info,
            "mhc_implementation",
            "four_stream_native_nvfp4_finite_sinkhorn_complete_adjoint",
        );
        push_info(&mut info, "mhc_streams", STREAMS);
        push_info(&mut info, "mhc_branch_boundaries", BOUNDARIES);
        push_info(&mut info, "mhc_hyperhead", "learned_before_final_norm");
        push_info(
            &mut info,
            "mhc_extra_parameters",
            BOUNDARIES * (WIDTH * 24 + 27) + WIDTH * 4 + 5,
        );
        push_info(&mut info, "mhc_projection_physical_rows", PAD);
        push_info(
            &mut info,
            "mhc_projection_padding",
            "structural_zero_not_trainable",
        );
        push_info(&mut info, "mhc_sinkhorn_column_normalizations", 20);
        push_info(&mut info, "mhc_rms_epsilon", 1e-6);
        push_info(
            &mut info,
            "mhc_initialization",
            "paper_branch_alpha_0.01_library_phi_normal_0.02_zero_bias_hyperhead_alpha_1",
        );
        push_info(&mut info, "mhc_branch_alpha_init", 0.01);
        push_info(&mut info, "mhc_hyperhead_alpha_init", 1);
        push_info(&mut info, "mhc_checkpoint_initialization_version", 2);
        push_info(
            &mut info,
            "mhc_activation_policy",
            "exact_stream_checkpoints_and_three_recomputed_boundary_inputs",
        );
        push_info(
            &mut info,
            "mhc_backward_rng_during_recomputation",
            "unchanged_forward_packing_consumes_no_backward_seeds",
        );
        push_info(
            &mut info,
            "mhc_optimizer",
            "common_Muon_SymExpLin_and_AMUSE_vector_updates_global_clip_and_skip",
        );
    }
    push_info(&mut info, "latent_feedback_enabled", config.latent_feedback);
    if config.latent_feedback {
        push_info(
            &mut info,
            "latent_feedback_implementation",
            "full_native_detached_recurrence",
        );
        push_info(
            &mut info,
            "latent_feedback_one_pass_probability",
            crate::training::latent_feedback::ONE_PASS_PROBABILITY,
        );
        push_info(
            &mut info,
            "latent_feedback_evaluation_passes",
            crate::training::latent_feedback::EVAL_PASSES,
        );
        push_info(
            &mut info,
            "latent_feedback_extra_parameters",
            gpt2_nvfp4::GPT2_N_EMBD * gpt2_nvfp4::GPT2_N_EMBD + 3,
        );
        push_info(
            &mut info,
            "latent_feedback_optimizer",
            "common Muon/SymExpLin, conditional full-state skip on one-pass updates",
        );
    }
    push_info(&mut info, "seed", format!("{:#x}", config.seed));
    push_run_env(&mut info);
    info
}

fn push_run_env(info: &mut String) {
    for name in [
        "CUDA_DEVICE_INDEX",
        "CUDA_OXIDE_PTX_PATH",
        "GPT2_MLP_KIND",
        "GPT2_MLP_INITIALIZER",
        "TRAIN_DATASET",
        "TRAIN_RUN_DIR",
        "TRAIN_LOAD_MODEL",
        "TRAIN_SAVE_MODEL",
        "TRAIN_STEPS",
        "TRAIN_LOG_INTERVAL",
        "TRAIN_KDA_CLIP_LOG_INTERVAL",
        "TRAIN_GRADIENT_COMPONENT_INTERVAL",
        "TRAIN_EVAL_INTERVAL",
        "TRAIN_EVAL_REPEATS",
        "TRAIN_MAX_SECONDS",
        "TRAIN_REPEAT_BATCH",
        "TRAIN_SEED",
        "TRAIN_MHC",
        "TRAIN_ENGRAM_MHC",
        "TRAIN_ENGRAM",
        "TRAIN_LATENT_FEEDBACK",
        "TRAIN_LR_SCALE",
        "TRAIN_ADAM_LR_SCALE",
        "TRAIN_NEXTLAT_LR_SCALE",
        "TRAIN_LR_WARMUP_STEPS",
        "TRAIN_LR_START_RATIO",
        "TRAIN_AMUSE_BETA1",
        "TRAIN_AMUSE_RHO",
        "TRAIN_HYPERBALL",
        "TRAIN_HYPERBALL_LR",
        "TRAIN_HYPERBALL_AMUSE",
        "TRAIN_HYPERBALL_POLAR_PERIOD",
        "TRAIN_HYPERBALL_POLAR_NUMERATOR",
        "TRAIN_MUON_VS",
        "TRAIN_EMBER",
        "TRAIN_EMBER_LR",
        "TRAIN_EMBER_BETA2",
        "TRAIN_EMBER_WEIGHT_DECAY",
        "TRAIN_EMBER_EPS",
        "TRAIN_SYMEXP_LIN",
        "TRAIN_SYMEXP_LIN_BETA",
        "TRAIN_SYMEXP_LIN_LEARN_SCALES",
        "TRAIN_SYMEXP_LIN_ANNEAL_STEPS",
        "TRAIN_SYMEXP_LIN_REUSE_AMAX",
        "TRAIN_ATTENTION_HEADWISE_GATE",
        "TRAIN_XSA",
        "TRAIN_XSA_KDA",
        "TRAIN_PARTIAL_KEY_OFFSET",
        "TRAIN_SELECTIVE_ATTENTION",
        "TRAIN_STABLE_MASK",
        "TRAIN_CANON_AC",
        "TRAIN_NEXTLAT_LOSS_WEIGHT",
        "TRAIN_SKIP_UNSTABLE_UPDATES",
        "TRAIN_SKIP_ROLLING_INTERVAL",
        "TRAIN_SKIP_SIGMA_FACTOR",
        "TRAIN_SKIP_USE_LOSS",
        "TRAIN_SKIP_USE_GRAD_NORM",
        "TRAIN_SKIP_CAP_GRAD_HISTORY",
        "TRAIN_GENERATE_PROMPT",
        "TRAIN_GENERATE",
        "TRAIN_GENERATE_TOKENS",
        "TRAIN_GENERATE_TEMPERATURE",
        "TRAIN_GENERATE_TOP_K",
        "TRAIN_GENERATE_TOP_P",
    ] {
        if let Ok(value) = std::env::var(name) {
            push_info(info, name, value);
        }
    }
}

fn push_info(info: &mut String, name: &str, value: impl std::fmt::Display) {
    use std::fmt::Write;
    let _ = writeln!(info, "{name}={value}");
}
