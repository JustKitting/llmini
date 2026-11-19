use cuda_core::CudaStream;

use super::super::OptimizerTrace;
use super::super::grad_clip::first_non_finite_gradient;
use super::super::grads::BackwardBuffers;
use super::super::next_latent::NextLatGradBuffers;
use super::super::update_skip::UpdateSkipDecision;
use crate::AppResult;

pub(super) fn record_skip_decision(
    stream: &CudaStream,
    grads: &BackwardBuffers,
    next_latent_grads: &NextLatGradBuffers,
    candidate_step: u32,
    trace: &mut OptimizerTrace,
    skip: UpdateSkipDecision,
    extra_non_finite: Option<(String, usize, f32)>,
) -> AppResult<bool> {
    trace.update_skipped = skip.skipped;
    trace.skip_loss_spike = skip.loss_spike;
    trace.skip_grad_norm_spike = skip.grad_norm_spike;
    trace.skip_non_finite = skip.non_finite;
    if skip.non_finite {
        if let Some((name, index, value)) = extra_non_finite {
            eprintln!(
                "non_finite_gradient optimizer_step_candidate={candidate_step} tensor={name} index={index} value={value:.9e}"
            );
        } else {
            log_non_finite_gradient(stream, grads, next_latent_grads, candidate_step)?;
        }
    }
    Ok(skip.skipped)
}

fn log_non_finite_gradient(
    stream: &CudaStream,
    grads: &BackwardBuffers,
    next_latent_grads: &NextLatGradBuffers,
    candidate_step: u32,
) -> AppResult {
    if let Some(bad) = first_non_finite_gradient(stream, grads, next_latent_grads)? {
        eprintln!(
            "non_finite_gradient optimizer_step_candidate={candidate_step} tensor={} index={} value={:.9e}",
            bad.name, bad.index, bad.value
        );
    } else {
        eprintln!("non_finite_gradient optimizer_step_candidate={candidate_step} tensor=unknown");
    }
    Ok(())
}
