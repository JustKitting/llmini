//! Exact trained native checkpoint reproducer, declared before a repair.
use super::*;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_SEQ_LEN};
fn prefix(t: &Trainer, b: &cuda_core::DeviceBuffer<f32>, n: usize) -> AppResult<Vec<f32>> {
    let mut v = vec![0.; n];
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            v.as_mut_ptr(),
            b.cu_deviceptr(),
            n * 4,
            t.runtime.stream.cu_stream(),
        )?;
    }
    t.runtime.stream.synchronize()?;
    Ok(v)
}
fn diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max)
}
#[test]
#[ignore = "exact trained30k CUDA causality replay; zero optimizer updates"]
fn native_trained_latent_feedback_causality_replay() -> AppResult {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("target/runs/20261115_044432Z_fineweb_43200s/model.ckpt");
    crate::reference_validation::require_hash(
        &path,
        "04bac2eee6ef32d15a144fe41ed87dbdadefd33358266c617c7094ca4bf2818a",
    )?;
    let windows = input::load(&root.join("data/fineweb/reference/documents.bin"))?;
    let flat = windows[..GPT2_BATCH_SIZE]
        .iter()
        .flat_map(|w| w.ids.iter().copied())
        .collect::<Vec<_>>();
    let mut t = Trainer::new_feedback(0x47505432)?;
    t.load_model(&path)?;
    let b = t.batch_from_windows(&flat, GPT2_BATCH_SIZE)?;
    let mut maximum = 0f32;
    for passes in [1, 2, 4] {
        for cut in [1, 64, 128, 1023] {
            let a = t.eval_token_losses_for_passes(&b, passes)?;
            let f = t.latent_feedback.as_ref().unwrap();
            let top = prefix(&t, &t.buffers.residual, cut * GPT2_N_EMBD)?;
            let norm = prefix(&t, &f.input, cut * GPT2_N_EMBD)?;
            let projected = prefix(&t, &f.projected, cut * GPT2_N_EMBD)?;
            let mut changed = flat.clone();
            for window in changed.chunks_exact_mut(GPT2_SEQ_LEN + 1) {
                for (i, v) in window.iter_mut().enumerate().skip(cut) {
                    *v = if i % 2 == 0 { 22557 } else { 28808 };
                }
            }
            let mut cb = t.batch_from_windows(&changed, GPT2_BATCH_SIZE)?;
            cb.targets = t.batch_from_windows(&flat, GPT2_BATCH_SIZE)?.targets;
            let z = t.eval_token_losses_for_passes(&cb, passes)?;
            let f = t.latent_feedback.as_ref().unwrap();
            let mut loss_max = 0f32;
            let mut first = None;
            for seq in 0..GPT2_BATCH_SIZE {
                for j in 0..cut {
                    let i = seq * GPT2_SEQ_LEN + j;
                    let error = (a[i] - z[i]).abs();
                    loss_max = loss_max.max(error);
                    if first.is_none() && error > 1e-5 {
                        first = Some((seq, j, a[i], z[i]));
                    }
                }
            }
            maximum = maximum.max(loss_max);
            println!(
                "trained_causality passes={passes} cut={cut} loss_max={loss_max} top_prefix_diff={} feedback_input_prefix_diff={} feedback_projection_prefix_diff={} first={first:?}",
                diff(&top, &prefix(&t, &t.buffers.residual, cut * GPT2_N_EMBD)?),
                diff(&norm, &prefix(&t, &f.input, cut * GPT2_N_EMBD)?),
                diff(&projected, &prefix(&t, &f.projected, cut * GPT2_N_EMBD)?)
            );
        }
    }
    assert_eq!(t.buffers.optimizer_state.step, 0);
    crate::reference_validation::require_hash(
        &path,
        "04bac2eee6ef32d15a144fe41ed87dbdadefd33358266c617c7094ca4bf2818a",
    )?;
    assert!(
        maximum <= 1e-5,
        "exact trained checkpoint future-prefix error {maximum}"
    );
    let v = TokenDataLoader::fineweb_validation_tokens()?;
    let native = t.eval_loss_windows(&v, super::super::data::VALIDATION_WINDOWS)?;
    println!("corrected_native_validation_ce={native:.9}");
    let full = causality::verify(&mut t, &windows)?;
    println!("corrected_full_causality={full}");
    let certificate = serde_json::json!({"type":"native_kda_causal_repair_replay_v1","passed":true,"checkpoint_sha256":crate::reference_validation::sha256(&path)?,"checkpoint_unchanged":true,"ptx_sha256":crate::reference_validation::sha256(std::path::Path::new(&super::super::runtime::ptx_path()))?,"reference_documents_sha256":crate::reference_validation::DATA_SHA,"producer_recorded_native_ce":expected_native_ce(&path)?,"corrected_native_ce":native,"causality":full,"optimizer_updates":t.buffers.optimizer_state.step,"method":"AQK consumes original causal Kneg; removed future chunk-end decay reconstruction","scope":"Read-only replay of unchanged trained weights under the repaired graph, not a new training trial"});
    std::fs::write(
        path.parent().unwrap().join("causal_repair_replay.json"),
        serde_json::to_string_pretty(&certificate)? + "\n",
    )?;
    Ok(())
}
