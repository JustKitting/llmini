//! Real complete-model rejection semantics; declared before adding the test hook.
mod feedback;
mod fingerprint;
mod first_update_replay;
use super::Trainer;
use crate::AppResult;
use fingerprint::Snapshot;
use gpt2_nvfp4::{GPT2_BATCH_SIZE as B, GPT2_N_EMBD as D, GPT2_SEQ_LEN as S};
use serde_json::json;

#[derive(Clone, Copy, Debug)]
pub(in crate::training) enum BadGradient {
    Table,
    Key3Bias,
    ConvNorm3,
    HyperHead,
    HyperHeadCurvature,
    HyperHeadBias,
    BoundaryScale,
    FeedbackExponential,
    FeedbackCurvature,
    FeedbackLateBias,
}
impl BadGradient {
    fn name(self) -> &'static str {
        match self {
            Self::Table => "late_table",
            Self::Key3Bias => "late_key3_bias",
            Self::ConvNorm3 => "late_convolution_norm3",
            Self::HyperHead => "hyperhead_projection",
            Self::HyperHeadCurvature => "hyperhead_curvature",
            Self::HyperHeadBias => "hyperhead_bias",
            Self::BoundaryScale => "late_boundary_scale",
            Self::FeedbackExponential => "feedback_exponential",
            Self::FeedbackCurvature => "feedback_curvature",
            Self::FeedbackLateBias => "feedback_late_backbone_bias",
        }
    }
}
pub(in crate::training) struct Audit {
    case: Option<BadGradient>,
    expected: Option<Snapshot>,
}
impl Audit {
    pub fn before_update(&mut self, t: &mut Trainer) -> AppResult {
        assert!(self.expected.is_none());
        self.expected = Some(Snapshot::capture(t)?);
        let Some(case) = self.case else {
            return Ok(());
        };
        let gradient = match case {
            BadGradient::Table => &t.engram.as_ref().unwrap().modules[1].ng[0],
            BadGradient::Key3Bias => {
                let e = &t.engram.as_ref().unwrap().modules[1];
                &e.ng[e.cfg.key_index(3) + 1]
            }
            BadGradient::ConvNorm3 => &t.engram.as_ref().unwrap().modules[1].fg[2],
            BadGradient::HyperHead => &t.mhc.as_ref().unwrap().units[32].grad,
            BadGradient::HyperHeadCurvature => {
                &t.mhc.as_ref().unwrap().units[32]
                    .phi
                    .symexp_lin
                    .curvature
                    .grad
            }
            BadGradient::HyperHeadBias => &t.mhc.as_ref().unwrap().units[32].fg[0],
            BadGradient::BoundaryScale => &t.mhc.as_ref().unwrap().units[31].fg[1],
            BadGradient::FeedbackExponential => {
                &t.latent_feedback
                    .as_ref()
                    .unwrap()
                    .state
                    .symexp_lin
                    .exponential
                    .grad
            }
            BadGradient::FeedbackCurvature => {
                &t.latent_feedback
                    .as_ref()
                    .unwrap()
                    .state
                    .symexp_lin
                    .curvature
                    .grad
            }
            BadGradient::FeedbackLateBias => &t.buffers.backward.blocks[15].d_mlp_c_fc_bias,
        };
        let index = if matches!(case, BadGradient::HyperHead) {
            0
        } else {
            gradient.len() - 1
        };
        unsafe {
            cuda_core::memory::memset_d8_async(
                gradient.cu_deviceptr() + (index * 4) as u64,
                255,
                4,
                t.runtime.stream.cu_stream(),
            )?;
        }
        Ok(())
    }
}

#[test]
#[ignore = "complete B8/L16/d2048 Engram/mHC actual train_step; all-state rejection fingerprints; correctness only"]
fn native_complete_engram_global_rejection_preserves_all_training_state() -> AppResult {
    verify(None, false)
}
#[test]
#[ignore = "same complete train_step rejection proof with500M lookup values; released GPU required"]
fn native_budgeted_engram_global_rejection_preserves_all_training_state() -> AppResult {
    verify(Some(500_000_000), false)
}

#[test]
#[ignore = "complete standalone B8/L16/d2048 mHC actual train_step; Ember/all-state rejection and recovery"]
fn native_standalone_mhc_global_rejection_preserves_all_training_state() -> AppResult {
    verify(None, true)
}

#[test]
#[ignore = "complete full-credit Engram/mHC actual train_step; five faults, all persistent AMUSE/Ember state and clean recovery"]
fn native_full_credit_engram_global_rejection_preserves_all_training_state() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD);
    verify(None, false)
}

#[test]
#[ignore = "complete bounded-SwiGLU/Engram/mHC/full-credit actual train_step; both table budgets, five faults, Ember/all-state recovery"]
fn native_bounded_engram_full_credit_global_rejection_preserves_all_training_state() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD);
    verify(None, false)?;
    verify(Some(500_000_000), false)
}

#[test]
#[ignore = "complete value-RMS/Engram/mHC/full-credit actual train_step; both budgets, original five faults, Ember and all-state recovery"]
fn native_value_engram_full_credit_global_rejection_preserves_all_training_state() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_VALUE_RMS && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD);
    verify(None, false)?;
    verify(Some(500_000_000), false)
}

#[test]
#[ignore = "complete stationary-position/Engram/mHC/full-credit actual train_step; both budgets, selected value/MLP, five faults, original Ember and all-state recovery"]
fn native_stationary_engram_full_credit_global_rejection_preserves_all_training_state() -> AppResult
{
    assert!(gpt2_nvfp4::GPT2_STATIONARY_ROPE && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD);
    verify(None, false)?;
    verify(Some(500_000_000), false)
}

fn verify(budget: Option<usize>, standalone_mhc: bool) -> AppResult {
    assert!(
        (!gpt2_nvfp4::GPT2_STATIONARY_ROPE || gpt2_nvfp4::GPT2_STATIONARY_ROPE_ENGRAM_MHC)
            && (!gpt2_nvfp4::GPT2_BOUNDED_SWIGLU || gpt2_nvfp4::GPT2_BOUNDED_SWIGLU_ENGRAM_MHC)
            && (!gpt2_nvfp4::GPT2_VALUE_RMS || gpt2_nvfp4::GPT2_VALUE_RMS_ENGRAM_MHC)
    );
    assert!(!gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD || !standalone_mhc);
    assert_eq!((B, S, D, gpt2_nvfp4::GPT2_N_LAYER), (8, 2048, 2048, 16));
    assert!(!standalone_mhc || budget.is_none());
    let mut t = if standalone_mhc {
        Trainer::new_mhc(0x47505432)?
    } else if let Some(requested) = budget {
        Trainer::new_engram_mhc_with_table_budget(0x47505432, requested)?
    } else {
        Trainer::new_engram_mhc(0x47505432)?
    };
    assert!(t.latent_feedback.is_none());
    assert_eq!(t.engram.is_none(), standalone_mhc);
    assert!(
        matches!(
            &t.buffers.optimizer_state.token_embedding,
            super::optimizer_state::TokenEmbeddingState::Ember(_)
        ),
        "verify the active Ember embedding; never replace it with Adam for the fixture"
    );
    let windows = (0..B * (S + 1))
        .map(|i| {
            if i % (S + 1) == 0 {
                1
            } else {
                (32 + i * 37 % 28000) as u16
            }
        })
        .collect::<Vec<_>>();
    let b = t.batch_from_windows(&windows, B)?;
    for _ in 0..2 {
        let stats = t.train_step(&b, true)?;
        assert!(stats.finite && !stats.optimizer.update_skipped);
    }
    assert_eq!(t.buffers.optimizer_state.step, 2);
    let table_values = t
        .uploaded
        .engram
        .as_ref()
        .iter()
        .flat_map(|e| &e.modules)
        .map(|w| w.cfg.table_len())
        .sum::<usize>();
    assert_eq!(
        table_values,
        if standalone_mhc {
            0
        } else if budget.is_some() {
            500_339_072
        } else {
            328_044_672
        }
    );
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        if gpt2_nvfp4::GPT2_STATIONARY_ROPE_ENGRAM_MHC {
            format!(
                "target/checks/20261116/stationary-engram-{}-{}-state-integrity-{table_values}",
                if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
                    "bounded"
                } else {
                    "relu"
                },
                if gpt2_nvfp4::GPT2_VALUE_RMS {
                    "value"
                } else {
                    "plain"
                }
            )
        } else if gpt2_nvfp4::GPT2_VALUE_RMS_ENGRAM_MHC {
            format!(
                "target/checks/20261116/value-engram-{}-state-integrity-{table_values}",
                if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
                    "bounded"
                } else {
                    "relu"
                }
            )
        } else if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU_ENGRAM_MHC {
            format!("target/checks/20261116/bounded-engram-state-integrity-{table_values}")
        } else if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
            format!("target/checks/20261116/full-credit-engram-state-integrity-{table_values}")
        } else if standalone_mhc {
            "target/checks/20261115/mhc-state-integrity".to_string()
        } else {
            format!("target/checks/20261115/engram-state-integrity-{table_values}")
        },
    );
    std::fs::create_dir_all(&out)?;
    let mut results = Vec::new();
    let cases: &[BadGradient] = if standalone_mhc {
        &[
            BadGradient::HyperHead,
            BadGradient::HyperHeadCurvature,
            BadGradient::HyperHeadBias,
            BadGradient::BoundaryScale,
        ]
    } else {
        &[
            BadGradient::Table,
            BadGradient::Key3Bias,
            BadGradient::ConvNorm3,
            BadGradient::HyperHead,
            BadGradient::HyperHeadCurvature,
        ]
    };
    for &case in cases {
        t.update_rejection_audit = Some(Audit {
            case: Some(case),
            expected: None,
        });
        let stats = t.train_step(&b, true)?;
        let expected = t.update_rejection_audit.take().unwrap().expected.unwrap();
        {
            let snapshot =
                t.completed_gradient_components(results.len() + 3, stats.optimizer.grad_norm)?;
            assert_eq!(snapshot["scope"], "actual_live_training_update");
            assert_eq!(snapshot["applied_updates"], 2);
            assert!(snapshot["components"].as_array().unwrap().len() > 600);
            if stats.optimizer.grad_norm.is_finite() {
                assert_eq!(
                    snapshot["global_gradient_norm"].as_f64(),
                    Some(f64::from(stats.optimizer.grad_norm))
                );
            } else {
                assert!(snapshot["global_gradient_norm"].is_null());
            }
            println!(
                "rejected_gradient_components_available case={} table_values={} applied_updates=2 extra_backward=false",
                case.name(),
                table_values
            );
        }
        let actual = Snapshot::capture(&t)?;
        let before = expected.json();
        let after = actual.json();
        std::fs::write(
            out.join(format!("{}-before.json", case.name())),
            serde_json::to_vec_pretty(&before)?,
        )?;
        std::fs::write(
            out.join(format!("{}-after.json", case.name())),
            serde_json::to_vec_pretty(&after)?,
        )?;
        assert!(
            stats.finite && stats.optimizer.update_skipped && stats.optimizer.skip_non_finite,
            "gradient fault not rejected after a finite forward: {case:?}"
        );
        assert_eq!(
            expected, actual,
            "persistent training state changed during {case:?}"
        );
        assert_eq!(t.buffers.optimizer_state.step, 2);
        results.push(json!({"case":case.name(),"passed":true,"nonfinite_rejected":true,"before":before["root_sha256"],"after":after["root_sha256"],"buffers":before["device_buffers"],"bytes":before["device_bytes"],"counts":before["counts"]}));
        println!(
            "complete_engram_state_rejection case={} table_values={} buffers={} bytes={} state=bit_exact applied_updates=2",
            case.name(),
            table_values,
            before["device_buffers"],
            before["device_bytes"]
        );
    }
    let recovery = t.train_step(&b, true)?;
    assert!(
        recovery.finite
            && !recovery.optimizer.update_skipped
            && recovery.optimizer.grad_norm.is_finite()
    );
    assert_eq!(t.buffers.optimizer_state.step, 3);
    {
        let snapshot =
            t.completed_gradient_components(cases.len() + 3, recovery.optimizer.grad_norm)?;
        assert_eq!(snapshot["passed"], true);
        assert_eq!(snapshot["applied_updates"], 3);
        println!(
            "recovered_gradient_components_verified table_values={table_values} applied_updates=3 extra_backward=false"
        );
    }
    let result = json!({"passed":true,"graph":if standalone_mhc{"standalone_mhc"}else{"engram_mhc"},"full_recurrent_credit":gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD,"stationary_RoPE":gpt2_nvfp4::GPT2_STATIONARY_ROPE,"value_RMS":gpt2_nvfp4::GPT2_VALUE_RMS,"value_RMS_precision_policy":if gpt2_nvfp4::GPT2_VALUE_RMS {Some([2_u32,32,16,16,32])} else {None},"bounded_SwiGLU":gpt2_nvfp4::GPT2_BOUNDED_SWIGLU,"embedding_optimizer":"Ember","table_values":table_values,"shape":{"batch":B,"context":S,"layers":16,"width":D,"heads":32},"cases":results,"fixture_applied_updates":3,"injected_rejections":cases.len(),"clean_recovery":true,"dense_forward_backward":"NVFP4","CPU_training":false,"host_fingerprint_workspace_bytes":fingerprint::CHUNK_BYTES,"loss_evidence":false});
    std::fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    if standalone_mhc {
        println!(
            "complete_standalone_mhc_all_training_state_verified cases=4 clean_recovery=true applied_updates=3 embedding_optimizer=Ember loss_evidence=false"
        );
    } else {
        println!(
            "complete_engram_all_training_state_verified table_values={table_values} cases=5 clean_recovery=true applied_updates=3 loss_evidence=false"
        );
    }
    Ok(())
}
