//! Native CUDA forward/backward on saved averaged weights, with zero updates.
use super::{TokenDataLoader, Trainer, data::VALIDATION_WINDOWS};
use crate::{AppResult, reference_validation};
use serde_json::{Value, json};
use std::{fs, path::Path};

#[cfg(test)]
mod tests;
mod window;
#[cfg(test)]
mod window_tests;

pub(super) fn run(
    checkpoint: &Path,
    output: Option<&Path>,
    replay_record: Option<&Path>,
    shard: Option<&Path>,
    token_offset: Option<usize>,
) -> AppResult {
    let selected = match (shard, token_offset) {
        (None, None) => None,
        (Some(shard), Some(offset)) => Some(window::load(
            shard,
            offset,
            gpt2_nvfp4::GPT2_BATCH_SIZE,
            gpt2_nvfp4::GPT2_SEQ_LEN,
        )?),
        _ => return Err("gradient shard and token offset must be provided together".into()),
    };
    let result = inspect_checkpoint_with_replay(checkpoint, replay_record, selected.as_ref())?;
    let output = output.map(Path::to_path_buf).unwrap_or_else(|| {
        checkpoint
            .parent()
            .unwrap_or(Path::new("."))
            .join("gradient_inspection.json")
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_string_pretty(&result)? + "\n")?;
    println!(
        "gradient_inspection_result={} norm={} replay_exact={} updates={}",
        output.display(),
        result["global_gradient_norm"],
        result["ce_replay_bit_exact"],
        result["optimizer_updates"]
    );
    if result["passed"] != true {
        return Err("gradient inspection failed; report saved".into());
    }
    Ok(())
}

#[cfg(test)]
fn inspect_checkpoint(checkpoint: &Path) -> AppResult<Value> {
    inspect_checkpoint_with_replay(checkpoint, None, None)
}
fn inspect_checkpoint_with_replay(
    checkpoint: &Path,
    replay_record: Option<&Path>,
    selected: Option<&window::InputWindow>,
) -> AppResult<Value> {
    let sha = reference_validation::sha256(checkpoint)?;
    let validation = TokenDataLoader::fineweb_validation_tokens()?;
    let mut trainer = Trainer::new(0x47505432)?;
    trainer.load_model(checkpoint)?;
    let validation_batch = trainer.batch_from_windows(&validation, VALIDATION_WINDOWS)?;
    let before = trainer.eval_token_losses(&validation_batch)?;
    let mean_ce = before.iter().map(|&x| f64::from(x)).sum::<f64>() / before.len() as f64;
    let (expected, verified_replay_record) =
        super::reference_eval::replay_expectation(checkpoint, replay_record)?;
    if before.iter().any(|v| !v.is_finite()) || expected.is_some_and(|v| (v - mean_ce).abs() > 1e-4)
    {
        return Err(
            "gradient probe checkpoint replay disagrees with recorded validation CE".into(),
        );
    }
    let selected_batch = selected
        .map(|s| trainer.batch_from_windows(&s.tokens, gpt2_nvfp4::GPT2_BATCH_SIZE))
        .transpose()?;
    let batch = selected_batch.as_ref().unwrap_or(&validation_batch);
    let probe_before = if selected.is_some() {
        trainer.eval_token_losses(batch)?
    } else {
        before.clone()
    };
    if probe_before.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite corpus-window checkpoint losses before gradient inspection".into());
    }
    let probe_ce =
        probe_before.iter().map(|&v| f64::from(v)).sum::<f64>() / probe_before.len() as f64;
    trainer.forward_step(batch)?;
    trainer.enqueue_numeric_probe_backward(batch)?;
    // Match the normal update's tied input/output-embedding accumulation, without applying it.
    super::optimizer_apply::add_embedding_lookup_grad(
        trainer.runtime.stream.as_ref(),
        &trainer.runtime.optimizer,
        batch,
        &mut trainer.buffers.backward,
        &trainer.buffers.next_latent_grads,
    )?;
    let stream = trainer.runtime.stream.as_ref();
    let clip = trainer
        .buffers
        .grad_clip
        .clip(stream, &trainer.runtime.optimizer)?;
    let components = trainer.buffers.grad_clip.norm_components(
        stream,
        &trainer.buffers.backward,
        &trainer.buffers.next_latent_grads,
        &trainer.buffers.muon_tables,
    )?;
    let all_finite = components.iter().all(|c| c["finite"] == true);
    let sum = components
        .iter()
        .filter_map(|c| c["squared_l2"].as_f64())
        .sum::<f64>();
    let reconciles = clip.norm.is_finite()
        && (sum.sqrt() - f64::from(clip.norm)).abs() < 5e-5 * f64::from(clip.norm).max(1.0);
    let independent_matrix_checks = independent_matrix_checks(&trainer, &components)?;
    let mut grouped = std::collections::BTreeMap::<String, f64>::new();
    for component in &components {
        *grouped
            .entry(component["kind"].as_str().unwrap().to_owned())
            .or_default() += component["squared_l2"].as_f64().unwrap_or(0.0);
    }
    let groups = grouped.into_iter().map(|(kind, square)|
        json!({"kind":kind,"squared_l2":square,"l2":square.sqrt(),"share_of_squared_norm":square/sum}))
        .collect::<Vec<_>>();
    let probe_after = trainer.eval_token_losses(batch)?;
    let probe_replay = probe_before == probe_after;
    let validation_replay = if selected.is_some() {
        before == trainer.eval_token_losses(&validation_batch)?
    } else {
        before == probe_after
    };
    let replay = probe_replay && validation_replay;
    let updates = trainer.buffers.optimizer_state.step;
    let after_sha = reference_validation::sha256(checkpoint)?;
    let passed = replay
        && updates == 0
        && all_finite
        && reconciles
        && sha == after_sha
        && independent_matrix_checks
            .iter()
            .all(|v| v["passed"] == true);
    Ok(
        json!({"passed":passed,"checkpoint":checkpoint,"checkpoint_sha256":sha,
        "checkpoint_unchanged":sha==after_sha,"model":"native approximately1B B8/S2048/L16/d2048/h32",
        "input":if selected.is_some(){"selected native corpus window"}else{"native held-out validation windows"},
        "input_window":selected.map(|s|&s.metadata),"target_tokens":batch.token_count,
        "base_seed":"0x47505432", "native_tokenwise_ce":mean_ce,"recorded_native_ce":expected,
        "native_validation_ce":mean_ce,"native_validation_target_tokens":validation_batch.token_count,
        "probe_tokenwise_ce":probe_ce,"probe_replay_bit_exact":probe_replay,
        "native_validation_replay_bit_exact":validation_replay,
        "global_gradient_norm":clip.norm,"clip_scale_diagnostic_only":clip.scale,
        "producer_recorded_native_ce":super::reference_eval::expected_native_ce(checkpoint)?,
        "verified_replay_provenance":verified_replay_record,
        "component_norm":sum.sqrt(),"global_norm_reconciles":reconciles,
        "full_attention_backward_tile_budget":gpt2_nvfp4::GPT2_ATTENTION_BACKWARD_TILE_BUDGET,
        "components":components,"groups":groups,"all_components_finite":all_finite,
        "independent_matrix_checks":independent_matrix_checks,
        "optimizer_updates":updates,"optimizer_state_replayed":false,
        "engram_present":trainer.engram.is_some(),"mhc_present":trainer.mhc.is_some(),"ce_replay_bit_exact":replay,
        "limits":"Frozen averaged evaluation checkpoint; native CUDA forward/backward and existing GPU clip reductions. No optimizer update, training-weight materialization, CPU training or dense precision fallback. Original unsaved optimizer/RNG/training-view state is not replayed; this probe localizes contributions at the saved model, not the original training event. Native validation replay and selected corpus-window CE are reported separately."}),
    )
}

fn inspected_engram_projectors(cfg: &super::engram::Config) -> Vec<(String, usize)> {
    let names = cfg.nv_names();
    cfg.projector_indices()
        .into_iter()
        .map(|index| (format!("{}.{}", cfg.prefix(), names[index]), index))
        .collect()
}

fn independent_matrix_checks(trainer: &Trainer, components: &[Value]) -> AppResult<Vec<Value>> {
    let d = gpt2_nvfp4::GPT2_N_EMBD;
    let mut selected = vec![
        (
            "next_latent.output_projection.weight".to_string(),
            &trainer.buffers.next_latent_grads.d_output_projection_weight,
            gpt2_nvfp4::NEXTLAT_HIDDEN * d,
        ),
        (
            "blocks.0.attn_c_proj.weight".to_string(),
            &trainer.buffers.backward.blocks[0].d_attn_c_proj_weight,
            d * d,
        ),
        (
            "blocks.15.attn_c_proj.weight".to_string(),
            &trainer.buffers.backward.blocks[15].d_attn_c_proj_weight,
            d * d,
        ),
    ];
    if let Some(e) = &trainer.engram {
        for e in &e.modules {
            for (name, index) in inspected_engram_projectors(&e.cfg) {
                selected.push((name, &e.ng[index], e.ng[index].len()));
            }
        }
    }
    if let Some(mhc) = &trainer.mhc {
        for index in [0, mhc.units.len() - 1] {
            let unit = &mhc.units[index];
            selected.push((unit.names[0].clone(), &unit.grad, unit.grad.len()));
        }
    }
    let mut result = Vec::new();
    for (name, buffer, len) in selected {
        let data = buffer.to_host_vec(trainer.runtime.stream.as_ref())?;
        let oracle = data[..len]
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>();
        // mHC's physical padding is cleared after MS-EDEN, so its matrix norm
        // deliberately scans the completed gradient through the ordinary
        // reduction path. Other selected matrices use cached Muon tile sums.
        let expected_kind = if name.starts_with("mhc.") {
            "ordinary_parameter"
        } else {
            "muon_matrix"
        };
        let component = components
            .iter()
            .find(|v| v["name"] == name && v["kind"] == expected_kind)
            .ok_or_else(|| format!("missing matrix norm component {name}"))?;
        let reduced = component["squared_l2"].as_f64().unwrap_or(f64::NAN);
        let passed = oracle.is_finite() && (oracle - reduced).abs() < 5e-5 * oracle.max(1.0);
        result.push(json!({"name":name,"kind":expected_kind,"values":len,"actual_gradient_f64_sum_sq":oracle,
            "gpu_tile_sum_sq":reduced,"passed":passed,
            "scope":"Read-only host reduction of the actual CUDA backward output; not CPU training."}));
    }
    Ok(result)
}

impl Trainer {
    pub(in crate::training) fn completed_gradient_components(
        &self,
        attempted_step: usize,
        norm: f32,
    ) -> AppResult<Value> {
        let components = self.buffers.grad_clip.norm_components(
            self.runtime.stream.as_ref(),
            &self.buffers.backward,
            &self.buffers.next_latent_grads,
            &self.buffers.muon_tables,
        )?;
        let all_finite = components.iter().all(|v| v["finite"] == true);
        let square = components
            .iter()
            .filter_map(|v| v["squared_l2"].as_f64())
            .sum::<f64>();
        let reconciles = norm.is_finite()
            && (square.sqrt() - f64::from(norm)).abs() < 5e-5 * f64::from(norm).max(1.0);
        Ok(
            json!({"passed":all_finite && reconciles,"scope":"actual_live_training_update",
            "attempted_step":attempted_step,"applied_updates":self.buffers.optimizer_state.step,
            "global_gradient_norm":norm,"component_norm":square.sqrt(),
            "global_norm_reconciles":reconciles,"all_components_finite":all_finite,
            "components":components,
            "source":"Existing pre-update CUDA clipping reductions, retained after the optimizer returns. No additional forward/backward or optimizer update."}),
        )
    }
}
