//! Summaries of the existing native HyperHead tapes; no CPU model replay.
use super::{
    Moments,
    engram::{distribution, sample_rows},
};
use crate::{
    AppResult,
    training::{
        Trainer,
        mhc::{BOUNDARIES, PAD, STREAMS, WIDTH},
    },
};
use gpt2_nvfp4::GPT2_N_EMBD;
use serde_json::{Value, json};

pub(super) fn inspect(trainer: &Trainer, rows: &[usize]) -> AppResult<(Value, usize)> {
    let (Some(state), Some(weights)) = (&trainer.mhc, &trainer.uploaded.mhc) else {
        if trainer.mhc.is_some() || trainer.uploaded.mhc.is_some() {
            return Err("mHC inspection state/weight mismatch".into());
        }
        return Ok((Value::Null, 0));
    };
    if state.n == 0 || rows.iter().any(|&row| row >= state.n as usize) {
        return Err("mHC inspection requires the current native forward tape".into());
    }
    let slot = state
        .tape_units
        .iter()
        .position(|&unit| unit == Some(BOUNDARIES))
        .ok_or("mHC inspection is missing the current HyperHead input tape")?;
    let stream = trainer.runtime.stream.as_ref();
    let input = sample_rows(stream, &state.tapes[slot].input, rows, WIDTH)?;
    let raw = state.projections[BOUNDARIES].raw.to_host_vec(stream)?;
    // All coefficient rows use stride24, including the head's four active entries.
    let coefficients = state.projections[BOUNDARIES]
        .coefficients
        .to_host_vec(stream)?;
    let active_rows = state.n as usize;
    if raw.len() < active_rows * PAD || coefficients.len() < active_rows * 24 {
        return Err("mHC inspection has a truncated HyperHead tape".into());
    }
    let alpha = weights.units[BOUNDARIES].alpha.weight.to_host_vec(stream)?;
    let bias = weights.units[BOUNDARIES].bias.weight.to_host_vec(stream)?;
    let mut inputs = Moments::default();
    for &x in &input {
        inputs.add(f64::from(x));
    }
    let mut projections = Moments::default();
    let mut gates = Vec::with_capacity(active_rows * STREAMS);
    let mut projection_values = Vec::with_capacity(active_rows * STREAMS);
    let mut nonfinite = alpha.iter().chain(&bias).filter(|x| !x.is_finite()).count();
    for (raw, coefficients) in raw[..active_rows * PAD]
        .chunks_exact(PAD)
        .zip(coefficients[..active_rows * 24].chunks_exact(24))
    {
        for i in 0..STREAMS {
            projections.add(f64::from(raw[i]));
            if raw[i].is_finite() {
                projection_values.push(f64::from(raw[i]));
            }
            if coefficients[i].is_finite() {
                gates.push(f64::from(coefficients[i]));
            } else {
                nonfinite += 1;
            }
        }
    }
    let mut cosines = Vec::with_capacity(rows.len() * STREAMS * (STREAMS - 1) / 2);
    let mut centered_cosines = Vec::with_capacity(cosines.capacity());
    let mut norms = Vec::with_capacity(rows.len() * STREAMS);
    let mut undefined_pairs = 0;
    let mut undefined_centered_pairs = 0;
    let mut zero_streams = 0;
    for row in input.chunks_exact(WIDTH) {
        // Host reductions describe copied native values; they do not feed the model.
        let squares: [f64; STREAMS] = std::array::from_fn(|i| {
            row[i * GPT2_N_EMBD..(i + 1) * GPT2_N_EMBD]
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum()
        });
        let means: [f64; STREAMS] = std::array::from_fn(|i| {
            row[i * GPT2_N_EMBD..(i + 1) * GPT2_N_EMBD]
                .iter()
                .map(|&x| f64::from(x))
                .sum::<f64>()
                / GPT2_N_EMBD as f64
        });
        let centered_squares: [f64; STREAMS] = std::array::from_fn(|i| {
            row[i * GPT2_N_EMBD..(i + 1) * GPT2_N_EMBD]
                .iter()
                .map(|&x| (f64::from(x) - means[i]).powi(2))
                .sum()
        });
        for &square in &squares {
            zero_streams += usize::from(square == 0.0);
            if square.is_finite() {
                norms.push(square.sqrt());
            }
        }
        for i in 0..STREAMS {
            for j in i + 1..STREAMS {
                let a = &row[i * GPT2_N_EMBD..(i + 1) * GPT2_N_EMBD];
                let b = &row[j * GPT2_N_EMBD..(j + 1) * GPT2_N_EMBD];
                let denom = (squares[i] * squares[j]).sqrt();
                if denom > 0.0 && denom.is_finite() {
                    let dot: f64 = a
                        .iter()
                        .zip(b)
                        .map(|(&x, &y)| f64::from(x) * f64::from(y))
                        .sum();
                    cosines.push((dot / denom).clamp(-1.0, 1.0));
                } else {
                    undefined_pairs += 1;
                }
                let centered_denom = (centered_squares[i] * centered_squares[j]).sqrt();
                if centered_denom > 0.0 && centered_denom.is_finite() {
                    let dot: f64 = a
                        .iter()
                        .zip(b)
                        .map(|(&x, &y)| (f64::from(x) - means[i]) * (f64::from(y) - means[j]))
                        .sum();
                    centered_cosines.push((dot / centered_denom).clamp(-1.0, 1.0));
                } else {
                    undefined_centered_pairs += 1;
                }
            }
        }
    }
    nonfinite += inputs.nonfinite + projections.nonfinite;
    Ok((
        json!({
            "stream_similarity_token_rows": rows.len(), "coefficient_token_rows": active_rows,
            "streams": STREAMS,
            "input_coordinate": "native_FP32_four_stream_tape_before_HyperHead_RMS_projection_and_collapse",
            "input_values": inputs.json(), "stream_l2": distribution(&mut norms),
            "zero_stream_vectors": zero_streams, "undefined_stream_pairs": undefined_pairs,
            "stream_pair_cosine": distribution(&mut cosines),
            "centered_stream_pair_cosine": distribution(&mut centered_cosines),
            "undefined_centered_stream_pairs": undefined_centered_pairs,
            "raw_projection": distribution(&mut projection_values),
            "native_coefficients_including_epsilon": distribution(&mut gates),
            "dynamic_scale": alpha, "bias": bias,
            "limits": "Coefficients cover all active native validation rows; stream similarity uses the selected token rows. Values are descriptive, not acceptance thresholds or evidence of the cause of an unsaved earlier training event."
        }),
        nonfinite,
    ))
}
