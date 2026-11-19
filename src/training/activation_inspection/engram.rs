//! Read-only summaries of the real Engram CUDA forward, not a CPU model replay.
//! Failure cases: stale unused tape rows, missing modules, invalid gate stats,
//! zero denominators, and sampled transfer buffers dropped before stream sync.
use super::Moments;
use crate::{AppResult, training::Trainer};
use cuda_core::{CudaStream, DeviceBuffer, memory};
use gpt2_nvfp4::GPT2_N_EMBD;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub(super) fn sample_rows(
    stream: &CudaStream,
    buffer: &DeviceBuffer<f32>,
    rows: &[usize],
    width: usize,
) -> AppResult<Vec<f32>> {
    if rows.iter().any(|row| (row + 1) * width > buffer.len()) {
        return Err("activation inspection row exceeds its native buffer".into());
    }
    let mut values = vec![0.; rows.len() * width];
    let copies = rows.iter().enumerate().try_for_each(|(index, row)| unsafe {
        memory::memcpy_dtoh_async(
            values.as_mut_ptr().add(index * width),
            buffer.cu_deviceptr() + (row * width * 4) as u64,
            width * 4,
            stream.cu_stream(),
        )
    });
    // Drain already-enqueued copies even if a subsequent enqueue failed.
    let synchronized = stream.synchronize();
    copies?;
    synchronized?;
    Ok(values)
}

pub(super) fn distribution(values: &mut [f64]) -> Value {
    if values.is_empty() {
        return json!({"values":0});
    }
    values.sort_by(f64::total_cmp);
    let last = values.len() - 1;
    json!({"values":values.len(),"min":values[0],"p01":values[last/100],
        "median":values[values.len()/2],"p99":values[last*99/100],"max":values[last]})
}

pub(super) fn inspect(trainer: &Trainer, rows: &[usize]) -> AppResult<(Value, usize)> {
    let (Some(state), Some(weights)) = (&trainer.engram, &trainer.uploaded.engram) else {
        if trainer.engram.is_some() || trainer.uploaded.engram.is_some() {
            return Err("Engram inspection state/weight mismatch".into());
        }
        return Ok((Value::Null, 0));
    };
    if state.modules.len() != weights.modules.len() {
        return Err("Engram inspection module count mismatch".into());
    }
    let stream = trainer.runtime.stream.as_ref();
    let mut modules = Vec::new();
    let mut nonfinite = 0;
    for (module, weight) in state.modules.iter().zip(&weights.modules) {
        if module.cfg.layer >= trainer.active_depth() {
            continue;
        }
        let n = module.n as usize;
        let stream_count = weight.cfg.streams();
        if n == 0 || rows.iter().any(|row| *row >= n) {
            return Err("Engram inspection requires the current native forward tape".into());
        }
        let stats = module.stats.to_host_vec(stream)?;
        let token_map = weight.token_map.to_host_vec(stream)?;
        let map_bytes = token_map
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>();
        let token_map_sha256 = format!("{:x}", Sha256::digest(&map_bytes));
        let compressed_token_classes = token_map.iter().copied().collect::<BTreeSet<_>>().len();
        if stats.len() < n * stream_count * 4 {
            return Err("Engram inspection has a truncated gate tape".into());
        }
        let mut gate = Vec::new();
        let mut raw = Vec::new();
        let mut query_inv = Vec::new();
        let mut key_inv = Vec::new();
        let mut clamped = 0;
        let mut near_closed = 0;
        let mut near_open = 0;
        for r in 0..n * stream_count {
            let values = &stats[r * 4..r * 4 + 4];
            nonfinite += values.iter().filter(|x| !x.is_finite()).count();
            if values.iter().any(|x| !x.is_finite()) {
                continue;
            }
            if values[0] <= 0. || values[1] <= 0. || !(0. ..=1.).contains(&values[3]) {
                return Err("Engram inspection found invalid native gate statistics".into());
            }
            query_inv.push(f64::from(values[0]));
            key_inv.push(f64::from(values[1]));
            raw.push(f64::from(values[2]));
            gate.push(f64::from(values[3]));
            clamped += usize::from(values[2].abs() < 1e-6);
            near_closed += usize::from(values[3] < 0.01);
            near_open += usize::from(values[3] > 0.99);
        }
        let mut tensors = Vec::new();
        let mut sampled_square = Vec::new();
        for (name, buffer, width) in [
            (
                "hidden_before_memory",
                &module.hidden,
                stream_count * GPT2_N_EMBD,
            ),
            ("projected_key", &module.key, stream_count * GPT2_N_EMBD),
            ("projected_value", &module.value, GPT2_N_EMBD),
            ("gated_value", &module.gated, stream_count * GPT2_N_EMBD),
            (
                "convolution_preactivation",
                &module.pre,
                stream_count * GPT2_N_EMBD,
            ),
        ] {
            let sample = sample_rows(stream, buffer, rows, width)?;
            let mut moments = Moments::default();
            for value in sample {
                moments.add(f64::from(value));
            }
            nonfinite += moments.nonfinite;
            sampled_square.push(moments.square);
            tensors.push(json!({"tensor":name,"sampled":moments.json()}));
        }
        let ids = module.ids.to_host_vec(stream)?;
        let heads = weight.cfg.primes.len();
        if ids.len() < n * heads {
            return Err("Engram inspection has truncated lookup addresses".into());
        }
        let mut offset = 0;
        let mut lookups = Vec::new();
        for (head, prime) in weight.cfg.primes.iter().enumerate() {
            let unique = rows
                .iter()
                .map(|row| ids[row * heads + head])
                .collect::<BTreeSet<_>>();
            if unique
                .iter()
                .any(|id| *id < offset || *id >= offset + prime)
            {
                return Err("Engram inspection found an address outside its hash head".into());
            }
            lookups.push(
                json!({"head":head,"table_rows":prime,"selected_lookup_events":rows.len(),
                "unique_selected_rows":unique.len()}),
            );
            offset += prime;
        }
        let injection_ratio =
            (sampled_square[0] > 0.).then(|| (sampled_square[3] / sampled_square[0]).sqrt());
        modules.push(json!({"layer":weight.cfg.layer,"graph_version":weight.cfg.descriptor()[0],"streams":stream_count,
            "token_id_map_sha256":token_map_sha256,"compressed_token_classes":compressed_token_classes,
            "gate_rms_epsilon":weight.cfg.gate_rms_epsilon(),"convolution_rms_epsilon":1e-5,
            "native_forward_rows":n,"native_gate_rows":n*stream_count,"gate":distribution(&mut gate),
            "normalized_dot_before_signed_root":distribution(&mut raw),
            "query_inverse_rms":distribution(&mut query_inv),"key_inverse_rms":distribution(&mut key_inv),
            "signed_root_clamped_rows":clamped,"gate_below_0_01_rows":near_closed,"gate_above_0_99_rows":near_open,
            "selected_token_rows":rows,"sampled_tensors":tensors,
            "sampled_gated_value_over_input_l2":injection_ratio,"selected_lookup_usage":lookups,
            "limits":"Actual CUDA outputs; host transfers and scalar summaries only. Lookup usage is the fixed token-row sample, not corpus-wide coverage. The L2 ratio measures the gated value, not its causal effect or the complete post-convolution residual. No alternate epsilon is replayed and no optimizer state is reconstructed."}));
    }
    Ok((json!(modules), nonfinite))
}
