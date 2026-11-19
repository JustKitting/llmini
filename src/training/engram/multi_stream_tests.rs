//! Declared before host integration: complete graph/lifecycle, not a loss screen.
use crate::training::optimizer_apply::{WeightUpdateArgs, apply_weight_updates};
use crate::{AppResult, training::Trainer};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_SEQ_LEN};

#[test]
fn four_stream_descriptor_rejects_silent_legacy_broadcast() -> AppResult {
    for index in 0..2 {
        let cfg = super::Config::reference_multistream(index);
        assert_eq!(cfg.streams(), 4);
        assert_eq!(cfg.nv_names().len(), 11);
        assert_eq!(cfg.projector_indices(), vec![1, 3, 5, 7, 9]);
        assert_eq!(cfg.adam_indices(), vec![0, 2, 4, 6, 8, 10]);
        let parsed = super::Config::parse(&cfg.descriptor(), false)?;
        assert_eq!(parsed.streams(), 4);
        assert_eq!(parsed.primes, cfg.primes);
        assert_eq!(parsed.multipliers, cfg.multipliers);
        assert_eq!(parsed.gate_rms_epsilon(), f32::EPSILON);
        assert!(super::Config::parse(&cfg.descriptor(), true).is_err());
        let old = super::Config::reference(index);
        assert_eq!(super::Config::parse(&old.descriptor(), false)?.streams(), 1);
        assert_eq!(old.nv_names()[1], "key.weight");
    }
    Ok(())
}

#[test]
#[ignore = "full native B8/L16/d2048 four-stream Engram +32 mHC boundaries + learned head"]
fn native_four_stream_engram_complete_graph_checkpoint_and_causality() -> AppResult {
    verify_complete_graph(None)
}

#[test]
#[ignore = "complete native B8/L16/d2048 four-stream graph with500M lookup values; released GPU required"]
fn native_budgeted_four_stream_engram_complete_graph_checkpoint_and_causality() -> AppResult {
    verify_complete_graph(Some(500_000_000))
}

#[test]
#[ignore = "actual launcher factory + complete B8 full-credit KDA/Engram/mHC graph, all gradients, checkpoint and causality"]
fn native_full_credit_engram_mhc_launcher_complete_graph_checkpoint_and_causality() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD);
    let config = crate::training::launch::TrainConfig {
        depth_growth: false,
        seed: 0x47505432,
        latent_feedback: false,
        mhc: true,
        engram_mhc: true,
        engram_table_params: None,
        step_cap: 20000,
        log_interval: 1,
        eval_interval: Some(1000),
        eval_repeats: 1,
        max_seconds: 43200.,
    };
    let trainer = config.create_trainer()?;
    verify_complete_trainer(trainer, None)
}

fn verify_complete_graph(table_budget: Option<usize>) -> AppResult {
    let t = if let Some(budget) = table_budget {
        Trainer::new_engram_mhc_with_table_budget(0x47505432, budget)?
    } else {
        Trainer::new_engram_mhc(0x47505432)?
    };
    verify_complete_trainer(t, table_budget)
}

pub(in crate::training) fn verify_complete_trainer(
    mut t: Trainer,
    table_budget: Option<usize>,
) -> AppResult {
    t.buffers
        .muon_tables
        .verify_native_storage_layout(&t.uploaded);
    if gpt2_nvfp4::MlpWeights::initializer_name() == "reference-normal-nvfp4" {
        super::super::bounded_swiglu_tests::verify_source_normal_upload(&t)?;
    }
    let actual_tables = t
        .uploaded
        .engram
        .as_ref()
        .unwrap()
        .modules
        .iter()
        .map(|m| m.cfg.table_len())
        .sum::<usize>();
    if let Some(requested) = table_budget {
        assert!((requested..requested + 1_000_000).contains(&actual_tables));
    }
    assert!(t.latent_feedback.is_none());
    assert_eq!(t.mhc.as_ref().unwrap().named_gradients().len(), 198);
    let w = t.uploaded.engram.as_ref().unwrap();
    assert_eq!(w.tensor_count(), 35);
    assert_eq!(w.modules.len(), 2);
    let ranges = t.buffers.muon_tables.matrix_grad_ranges();
    for (i, module) in w.modules.iter().enumerate() {
        assert_eq!(module.cfg.layer, [1, 5][i]);
        assert_eq!(module.cfg.streams(), 4);
        assert_eq!(module.cfg.memory(), 1024);
        assert_eq!(module.nv.len(), 11);
        for f in &module.fp[..3] {
            assert_eq!(f.weight.len(), 4 * GPT2_N_EMBD);
        }
        assert_eq!(module.fp[3].weight.len(), 16 * GPT2_N_EMBD);
        let e = &t.engram.as_ref().unwrap().modules[i];
        assert_eq!(e.named_gradients().len(), 30);
        assert_eq!(e.projector_states.len(), 5);
        assert_eq!(e.nv_states.len(), 6);
        for index in module.cfg.projector_indices() {
            assert_eq!(
                ranges
                    .iter()
                    .filter(|(p, _)| *p == e.ng[index].cu_deviceptr())
                    .count(),
                1,
                "independent K and shared V each participate in Muon exactly once"
            );
        }
    }
    let windows = (0..GPT2_BATCH_SIZE * (GPT2_SEQ_LEN + 1))
        .map(|i| {
            if i % (GPT2_SEQ_LEN + 1) == 0 {
                1
            } else {
                (32 + i * 37 % 28000) as u16
            }
        })
        .collect::<Vec<_>>();
    let b = t.batch_from_windows(&windows, GPT2_BATCH_SIZE)?;
    for _ in 0..2 {
        let result = t.train_step(&b, true)?;
        assert!(result.finite && result.optimizer.grad_norm.is_finite());
        assert!(!result.optimizer.update_skipped);
    }
    t.materialize_training_weights()?;
    if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
        t.enable_bounded_projection_verification();
    }
    if gpt2_nvfp4::GPT2_VALUE_RMS {
        t.enable_value_projection_verification()?;
    }
    t.mhc.as_mut().unwrap().verify_recomputation();
    t.forward_step(&b)?;
    let first_value_hash = if gpt2_nvfp4::GPT2_VALUE_RMS {
        Some(crate::training::value_rms_tests::hash(
            &t.runtime.stream,
            &t.buffers.value_residual,
        )?)
    } else {
        None
    };
    t.enqueue_numeric_probe_backward(&b)?;
    if let Some(expected) = first_value_hash {
        let (raw, represented) = t.value_projection_recomputation_verified();
        assert_eq!(raw, vec![true; 16]);
        assert_eq!(represented, vec![true; 16]);
        assert_eq!(
            expected,
            crate::training::value_rms_tests::hash(&t.runtime.stream, &t.buffers.value_residual)?,
            "complete graph overwrote immutable raw-first-value cache"
        );
        t.value_projection_audit = None;
    }
    if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
        assert_eq!(
            t.bounded_projection_recomputation_verified(),
            vec![true; 16]
        );
        let h = gpt2_nvfp4::GPT2_MLP;
        for block in 0..16 {
            let gradients = &t.buffers.backward.blocks[block];
            for (name, gradient, half_len) in [
                (
                    "paired_weight",
                    &gradients.d_mlp_c_fc_weight,
                    h * GPT2_N_EMBD,
                ),
                ("paired_bias", &gradients.d_mlp_c_fc_bias, h),
            ] {
                let values = gradient.to_host_vec(&t.runtime.stream)?;
                assert_eq!(values.len(), 2 * half_len);
                for (half, values) in values.chunks_exact(half_len).enumerate() {
                    assert!(
                        values.iter().all(|value| value.is_finite()),
                        "nonfinite layer={block} {name} half={half}"
                    );
                    assert!(
                        values.iter().any(|value| *value != 0.),
                        "gradient-dead layer={block} {name} half={half}"
                    );
                }
            }
        }
    }
    assert_eq!(
        t.mhc
            .as_ref()
            .unwrap()
            .recompute_verification
            .as_ref()
            .unwrap()
            .reconstructed,
        vec![true; 33],
        "memory insertion boundaries must survive exact mHC recomputation"
    );
    for (name, gradient) in t
        .engram
        .as_ref()
        .unwrap()
        .named_gradients()
        .into_iter()
        .chain(t.mhc.as_ref().unwrap().named_gradients())
    {
        let values = gradient.to_host_vec(&t.runtime.stream)?;
        assert!(values.iter().all(|x| x.is_finite()), "nonfinite {name}");
        assert!(values.iter().any(|x| *x != 0.), "gradient-dead {name}");
    }
    let clip = t
        .buffers
        .grad_clip
        .clip(&t.runtime.stream, &t.runtime.optimizer)?;
    assert!(clip.norm.is_finite());
    let components = t.buffers.grad_clip.norm_components(
        &t.runtime.stream,
        &t.buffers.backward,
        &t.buffers.next_latent_grads,
        &t.buffers.muon_tables,
    )?;
    for layer in 0..16 {
        for suffix in [
            "attn_qkv.weight",
            "attn_c_proj.weight",
            "mlp_up.weight",
            "mlp_down.weight",
        ] {
            let name = format!("blocks.{layer}.{suffix}");
            let c = components
                .iter()
                .find(|c| c["name"].as_str() == Some(&name))
                .unwrap();
            assert!(
                c["finite"] == true && c["l2"].as_f64().unwrap() > 0.,
                "inactive {name}"
            );
        }
    }
    for suffix in ["input_projection", "transition", "output_projection"] {
        let name = format!("next_latent.{suffix}.weight");
        let c = components
            .iter()
            .find(|c| c["name"].as_str() == Some(&name))
            .unwrap();
        assert!(c["finite"] == true && c["l2"].as_f64().unwrap() > 0.);
    }
    for layer in [1, 5] {
        assert_eq!(
            components
                .iter()
                .filter(|c| c["name"]
                    .as_str()
                    .unwrap()
                    .starts_with(&format!("engram.layers.{layer}.")))
                .count(),
            30
        );
    }
    // A bad memory contribution must reject the complete shared model update.
    let bad = &t.engram.as_ref().unwrap().modules[1].ng[0];
    unsafe {
        cuda_core::memory::memset_d8_async(
            bad.cu_deviceptr(),
            255,
            4,
            t.runtime.stream.cu_stream(),
        )?;
    }
    let step_before = t.buffers.optimizer_state.step;
    let rejected = apply_weight_updates(WeightUpdateArgs {
        stream: &t.runtime.stream,
        runtime: &t.runtime,
        batch: &b,
        uploaded: &mut t.uploaded,
        grads: &mut t.buffers.backward,
        next_latent_grads: &mut t.buffers.next_latent_grads,
        observed_loss: Some(5.),
        scratch: &mut t.buffers.optimizer,
        state: &mut t.buffers.optimizer_state,
        muon: &mut t.buffers.muon,
        muon_tables: &t.buffers.muon_tables,
        tape: &t.buffers.tape,
        qk_norm_max: t.buffers.scratch.attention_core.qk_norm_max(),
        grad_clip: &mut t.buffers.grad_clip,
    })?;
    assert!(rejected.trace.update_skipped && rejected.trace.skip_non_finite);
    assert_eq!(step_before, t.buffers.optimizer_state.step);
    t.materialize_evaluation_weights()?;
    let before = t.eval_token_losses(&b)?;
    assert!(before.iter().all(|x| x.is_finite()));
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        if gpt2_nvfp4::GPT2_STATIONARY_ROPE_ENGRAM_MHC {
            format!(
                "target/checks/20261116/stationary-engram-{}-{}-{}-roundtrip.ckpt",
                if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
                    "bounded"
                } else {
                    "relu"
                },
                if gpt2_nvfp4::GPT2_VALUE_RMS {
                    "value"
                } else {
                    "plain"
                },
                if table_budget.is_some() {
                    "budgeted"
                } else {
                    "default"
                }
            )
        } else if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD && gpt2_nvfp4::GPT2_VALUE_RMS {
            format!(
                "target/checks/20261116/value-engram-{}-{}-roundtrip.ckpt",
                if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
                    "bounded"
                } else {
                    "relu"
                },
                if table_budget.is_some() {
                    "budgeted"
                } else {
                    "default"
                }
            )
        } else if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD && gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
            if table_budget.is_some() {
                "target/checks/20261116/bounded-engram-budgeted-roundtrip.ckpt"
            } else {
                "target/checks/20261116/bounded-engram-default-roundtrip.ckpt"
            }
            .to_string()
        } else if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
            "target/checks/20261116/full-credit-engram-mhc-roundtrip.ckpt".to_string()
        } else if table_budget.is_some() {
            "target/checks/20261115/engram-mhc-budgeted-full-roundtrip.ckpt".to_string()
        } else {
            "target/checks/20261115/engram-mhc-full-roundtrip.ckpt".to_string()
        },
    );
    t.save_model(&path)?;
    let header = crate::checkpoint::CheckpointReader::new(std::io::BufReader::new(
        std::fs::File::open(&path)?,
    ))
    .read_header()?;
    assert_eq!(
        header,
        387 + u32::from(gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD)
            + u32::from(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU)
            + u32::from(gpt2_nvfp4::GPT2_VALUE_RMS)
            + u32::from(gpt2_nvfp4::GPT2_STATIONARY_ROPE)
    );
    let expected = t.eval_token_losses(&b)?;
    let allocation_identity = || {
        [
            t.uploaded.token_embedding.bytes.cu_deviceptr(),
            t.uploaded.engram.as_ref().unwrap().modules[0].nv[0]
                .bytes
                .cu_deviceptr(),
            t.uploaded.engram.as_ref().unwrap().modules[1].nv[0]
                .bytes
                .cu_deviceptr(),
            t.uploaded.mhc.as_ref().unwrap().units[0]
                .phi
                .bytes
                .cu_deviceptr(),
        ]
    };
    let original_allocations = allocation_identity();
    t.load_model(&path)?;
    if table_budget.is_some() && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
        assert_eq!(
            original_allocations,
            [
                t.uploaded.token_embedding.bytes.cu_deviceptr(),
                t.uploaded.engram.as_ref().unwrap().modules[0].nv[0]
                    .bytes
                    .cu_deviceptr(),
                t.uploaded.engram.as_ref().unwrap().modules[1].nv[0]
                    .bytes
                    .cu_deviceptr(),
                t.uploaded.mhc.as_ref().unwrap().units[0]
                    .phi
                    .bytes
                    .cu_deviceptr(),
            ],
            "same-graph reload must reuse NVFP4 parameter allocations",
        );
    }
    let actual = t.eval_token_losses(&b)?;
    assert_eq!(
        expected, actual,
        "averaged complete checkpoint must replay bit exactly"
    );
    if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
        let legacy = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/runs/20261116_023313Z_fineweb_150s/model.ckpt");
        assert!(
            t.load_model(&legacy).is_err(),
            "detached combined semantics silently loaded into full credit"
        );
        assert_eq!(
            actual,
            t.eval_token_losses(&b)?,
            "rejected legacy load mutated the full combined graph"
        );
        if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU
            || gpt2_nvfp4::GPT2_VALUE_RMS
            || gpt2_nvfp4::GPT2_STATIONARY_ROPE
        {
            let relu_full = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("target/runs/20261116_062746Z_fineweb_43200s/model.ckpt");
            assert!(
                t.load_model(&relu_full).is_err(),
                "full-credit checkpoint without the selected numerical descriptors was accepted"
            );
            assert_eq!(
                actual,
                t.eval_token_losses(&b)?,
                "rejected checkpoint without numerical descriptors mutated the selected graph"
            );
        }
    }
    if gpt2_nvfp4::GPT2_VALUE_RMS && !gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
        let old_point = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/checks/20261115/value-rms-full-roundtrip.ckpt");
        let error = t.load_model(&old_point).unwrap_err().to_string();
        assert!(
            error.contains("checkpoint value-normalization graph does not match"),
            "old value-RMS point must reject for its numerical descriptor: {error}"
        );
        assert_eq!(
            actual,
            t.eval_token_losses(&b)?,
            "rejected FP32 reverse-point checkpoint mutated the complete graph"
        );
        println!(
            "historical_value_RMS_reverse_policy_rejected complete_graph=unchanged descriptor_checked_before_upload=true"
        );
    }
    assert_eq!(t.engram.as_ref().unwrap().named_gradients().len(), 60);
    assert_eq!(t.mhc.as_ref().unwrap().named_gradients().len(), 198);
    for cut in [1, 31, 63, 64, 127, 128, 1023, 2015, 2047] {
        let mut altered = windows.clone();
        for sequence in altered.chunks_exact_mut(GPT2_SEQ_LEN + 1) {
            sequence[cut + 1..].fill(28001);
        }
        let changed = t.batch_from_windows(&altered, GPT2_BATCH_SIZE)?;
        let other = t.eval_token_losses(&changed)?;
        for sequence in 0..GPT2_BATCH_SIZE {
            let start = sequence * GPT2_SEQ_LEN;
            assert_eq!(&actual[start..start + cut], &other[start..start + cut]);
        }
    }
    let mut independent = windows.clone();
    independent[..GPT2_SEQ_LEN + 1].fill(28002);
    let independent_batch = t.batch_from_windows(&independent, GPT2_BATCH_SIZE)?;
    let independent_loss = t.eval_token_losses(&independent_batch)?;
    assert_eq!(
        &actual[GPT2_SEQ_LEN..],
        &independent_loss[GPT2_SEQ_LEN..],
        "memory/convolution/mHC must not mix independent sequences"
    );
    println!(
        "full_four_stream_engram_verified modules=2 memory=1024 independent_keys=8 shared_values=2 streams=4 boundaries=32 extra_gradient_groups=258 checkpoint=bit_exact causality=true prefix_cases=9 cross_sequence=bit_exact fixture_updates=2"
    );
    println!(
        "full_credit_combined_lifecycle_verified full_recurrent_credit={} checkpoint_tensors={header}",
        gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
    );
    if gpt2_nvfp4::GPT2_VALUE_RMS {
        println!(
            "complete_value_engram_verified raw_V_replays=16 represented_V_points=16 raw_first_cache=unchanged reverse_input_bits=16 reverse_direction_bits=16 full_credit=true bounded_MLP={} table_parameters={actual_tables} checkpoint_tensors={header}",
            gpt2_nvfp4::GPT2_BOUNDED_SWIGLU
        );
    }
    if gpt2_nvfp4::GPT2_STATIONARY_ROPE_ENGRAM_MHC {
        println!(
            "complete_stationary_engram_verified position_rotating=0..32 stationary=32..64 key_offset=stationary_half_only full_credit=true bounded_MLP={} value_RMS={} table_parameters={actual_tables} checkpoint_tensors={header}",
            gpt2_nvfp4::GPT2_BOUNDED_SWIGLU,
            gpt2_nvfp4::GPT2_VALUE_RMS
        );
    }
    if let Some(requested) = table_budget {
        println!(
            "budgeted_four_stream_engram_verified requested={requested} actual_table_parameters={actual_tables} streams=4 fixture_updates=2"
        );
    }
    Ok(())
}
