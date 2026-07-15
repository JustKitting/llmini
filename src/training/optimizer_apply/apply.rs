use super::super::OptimizerTrace;
use super::super::diagnostics::PendingTrainingDiagnostics;
use super::super::optimizer_aurora::aurora_learning_rate;
use super::adam::adam_learning_rate;
use super::aurora::update_aurora_groups;
use super::base::{BaseAdamUpdateArgs, update_base_adam};
use super::block::{BlockUpdateArgs, update_blocks};
use super::embedding::add_embedding_lookup_grad;
use super::kda_clip::apply_kda_aurora_clip;
use super::skip::record_skip_decision;
use super::timed_ms;
use super::types::{WeightUpdateArgs, WeightUpdateResult};
use crate::AppResult;

pub fn apply_weight_updates(args: WeightUpdateArgs<'_>) -> AppResult<WeightUpdateResult> {
    let WeightUpdateArgs {
        stream,
        runtime,
        batch,
        uploaded,
        grads,
        next_latent_grads,
        observed_loss,
        scratch,
        state,
        aurora,
        aurora_tables,
        tape,
        grad_clip,
    } = args;
    let optimizer = &runtime.optimizer;
    let mut trace = OptimizerTrace::default();
    let candidate_step = state.next_step();
    trace.adam_lr = adam_learning_rate(candidate_step);
    trace.aurora_lr = aurora_learning_rate(candidate_step);
    trace.embedding_lookup_ms =
        timed_ms(|| add_embedding_lookup_grad(stream, optimizer, batch, grads, next_latent_grads))?;

    let grad_clip = grad_clip.clip(stream, optimizer)?;
    trace.grad_norm = grad_clip.norm;

    let skip_decision = state.should_skip_update(observed_loss, grad_clip.norm);
    if record_skip_decision(
        stream,
        grads,
        next_latent_grads,
        candidate_step,
        &mut trace,
        skip_decision,
    )? {
        return Ok(WeightUpdateResult {
            trace,
            diagnostics: None,
        });
    }

    let step = state.advance();
    debug_assert_eq!(step, candidate_step);
    let average_coefficient = state.schedule_free_average_coefficient(step);

    let diagnostics = super::super::diagnostics::enabled()
        .then(|| {
            PendingTrainingDiagnostics::collect(
                stream,
                uploaded,
                grads,
                state,
                step,
                average_coefficient,
                grad_clip.scale,
            )
        })
        .transpose()?;

    update_base_adam(BaseAdamUpdateArgs {
        stream,
        optimizer,
        uploaded,
        grads,
        next_latent_grads,
        scratch,
        state,
        step,
        average_coefficient,
        grad_scale: grad_clip.scale,
        trace: &mut trace,
    })?;

    update_blocks(BlockUpdateArgs {
        stream,
        optimizer,
        uploaded,
        grads,
        scratch,
        state,
        step,
        average_coefficient,
        grad_scale: grad_clip.scale,
        trace: &mut trace,
    })?;

    update_aurora_groups(
        stream,
        runtime,
        aurora_tables,
        aurora,
        step,
        average_coefficient,
        grad_clip.scale,
        &mut trace,
    )?;
    apply_kda_aurora_clip(stream, runtime, uploaded, tape, scratch, state, &mut trace)?;

    let diagnostics = diagnostics
        .map(|pending| pending.finish(stream, uploaded))
        .transpose()?;

    Ok(WeightUpdateResult { trace, diagnostics })
}
