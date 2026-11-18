//! Explicit, hash-bound replay landmarks for a verified numerical graph repair.
//! The default historical-CE gate remains unchanged without a supplied record.
use super::*;
pub(in crate::training) fn expectation(
    checkpoint: &Path,
    record: Option<&Path>,
) -> AppResult<(Option<f64>, Option<serde_json::Value>)> {
    let original = super::expected_native_ce(checkpoint)?;
    let Some(path) = record else {
        return Ok((original, None));
    };
    let v: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let causality = &v["causality"];
    let correct = v["corrected_native_ce"]
        .as_f64()
        .filter(|x| x.is_finite() && *x > 0.);
    let producer = v["producer_recorded_native_ce"].as_f64();
    let causal = [
        "repeat_max_difference",
        "future_padding_max_difference",
        "future_prefix_max_difference",
        "cross_sequence_max_difference",
    ]
    .into_iter()
    .all(|key| {
        causality[key]
            .as_f64()
            .is_some_and(|x| x.is_finite() && x >= 0. && x <= 1e-5)
    });
    let ptx = super::super::runtime::ptx_path();
    if v["type"] != "native_kda_causal_repair_replay_v1"
        || v["passed"] != true
        || v["optimizer_updates"] != 0
        || v["checkpoint_unchanged"] != true
        || v["checkpoint_sha256"] != reference::sha256(checkpoint)?
        || v["ptx_sha256"] != reference::sha256(Path::new(&ptx))?
        || v["reference_documents_sha256"] != reference::DATA_SHA
        || causality["prefix_cases"] != 9
        || causality["batch_size"] != GPT2_BATCH_SIZE
        || !causal
        || correct.is_none()
        || original.is_none()
        || producer.is_none()
        || (original.unwrap() - producer.unwrap()).abs() > 1e-4
    {
        return Err("invalid or incompatible causal-repair replay provenance".into());
    }
    Ok((correct, Some(v)))
}
