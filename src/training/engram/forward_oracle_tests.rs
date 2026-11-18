//! Native reference-sized Engram at B8/S2048/d2048, zero updates.
//! Host f64 is an independent verification oracle for stored NVFP4 operands.
use super::*;
use super::{Module as Engram, ModuleWeights as Weights};
use crate::{AppResult, training::runtime::Runtime, upload::UploadedNvfp4};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_SEQ_LEN, HiddenStateDevice};
use std::collections::{BTreeSet, HashMap};

fn code(scales: &[u8], row: usize, group: usize, k: usize, offset: usize) -> u8 {
    let tile = row / 128 * (k / 128) + group / 8;
    let r = row % 128;
    let shift = (group % 4) * 6;
    let at = offset
        + tile * 896
        + (r / 16) * 96
        + (r % 8) * 12
        + (r % 16 / 8) * 3
        + (group % 8 / 4) * 6
        + shift / 8;
    let word = scales[at] as u16 | ((scales[at + 1] as u16) << 8);
    let high = (scales[offset + tile * 896 + 768 + r] >> (group % 8)) & 1;
    (((word >> (shift % 8)) & 63) as u8) | (high << 6)
}
fn decode(bytes: &[u8], scale: u8, global: f32, i: usize) -> f64 {
    let nibble = (bytes[i / 2] >> (4 * (i % 2))) & 15;
    let mag = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(nibble & 7) as usize];
    let exp = (scale >> 3) as i32;
    let mantissa = (scale & 7) as f64;
    let sf = if exp == 0 {
        mantissa * 2f64.powi(-9)
    } else {
        (1. + mantissa / 8.) * 2f64.powi(exp - 7)
    };
    (if nibble & 8 == 0 { mag } else { -mag }) * sf * f64::from(global)
}
fn unpack(s: &CudaStream, t: &UploadedNvfp4) -> AppResult<Vec<f64>> {
    let bytes = t.bytes.to_host_vec(s)?;
    let scales = t.scales.to_host_vec(s)?;
    let global = t.global_scale.to_host_vec(s)?[0];
    let k = t.scale_k as usize;
    Ok((0..t.len)
        .map(|i| {
            let sf = if scales[8] == 255 {
                scales[128 + i / 16]
            } else {
                code(&scales, i / k, i % k / 16, k, 128)
            };
            decode(&bytes, sf, global, i)
        })
        .collect())
}
fn row(s: &CudaStream, b: &DeviceBuffer<f32>, index: usize, d: usize) -> AppResult<Vec<f32>> {
    let mut values = vec![0.; d];
    unsafe {
        memory::memcpy_dtoh_async(
            values.as_mut_ptr(),
            b.cu_deviceptr() + (index * d * 4) as u64,
            d * 4,
            s.cu_stream(),
        )?;
    }
    s.synchronize()?;
    Ok(values)
}
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

#[test]
#[ignore = "native full-size Engram forward oracle, zero optimizer updates"]
fn native_engram_full_quantized_forward_matches_independent_oracle() -> AppResult {
    for i in 0..LAYERS.len() {
        check_layer(Config::reference(i))?;
        check_layer(Config::reference_multistream(i))?;
    }
    Ok(())
}
fn check_layer(cfg: Config) -> AppResult {
    assert_eq!(
        (GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_N_EMBD),
        (8, 2048, 2048)
    );
    let r = Runtime::new()?;
    let s = r.stream.as_ref();
    let mut w = Weights::new_with_config(&r, cfg)?;
    let streams = w.cfg.streams();
    // Heterogeneous nonzero biases expose the vector/matrix scale contract.
    for index in w.cfg.adam_indices().into_iter().skip(1) {
        let tensor = &mut w.nv[index];
        tensor.bytes = DeviceBuffer::from_host(s, &vec![0x22; tensor.len / 2])?;
        let scales = (0..tensor.len / 16)
            .map(|g| 0x38 + (g % 7) as u8)
            .collect::<Vec<_>>();
        tensor.scales = DeviceBuffer::from_host(
            s,
            &rust_kernels_cuda::nvfp4::csf::encode_host(
                &scales,
                tensor.scale_rows as usize,
                tensor.scale_k as usize,
            ),
        )?;
        tensor.global_scale = DeviceBuffer::from_host(s, &[0.125])?;
    }
    let mut e = Engram::new(&r, &w)?;
    let n = GPT2_BATCH_SIZE * GPT2_SEQ_LEN;
    let d = GPT2_N_EMBD;
    let mut tokens = (0..n)
        .map(|i| (32 + i * 37 % 28000) as u32)
        .collect::<Vec<_>>();
    for b in 0..GPT2_BATCH_SIZE {
        tokens[b * GPT2_SEQ_LEN] = llama2_tokenizer::BOS_TOKEN;
        tokens[b * GPT2_SEQ_LEN + 7] = llama2_tokenizer::EOS_TOKEN;
        tokens[b * GPT2_SEQ_LEN + 8] = llama2_tokenizer::BOS_TOKEN;
    }
    let input = (0..n * streams * d)
        .map(|i| ((i * 17 % 101) as f32 - 50.) * 0.007)
        .collect::<Vec<_>>();
    let gamma = [
        (0..streams * d)
            .map(|j| 0.85 + (j % 7) as f32 * 0.04)
            .collect::<Vec<_>>(),
        (0..streams * d)
            .map(|j| 0.9 + (j % 9) as f32 * 0.02)
            .collect::<Vec<_>>(),
        (0..streams * d)
            .map(|j| 1.1 - (j % 11) as f32 * 0.01)
            .collect::<Vec<_>>(),
        (0..streams * d * 4)
            .map(|j| ((j * 3 % 13) as f32 - 6.) * 0.013)
            .collect::<Vec<_>>(),
    ];
    for (t, values) in w.fp.iter_mut().zip(&gamma) {
        t.weight = DeviceBuffer::from_host(s, values)?;
    }
    let td = DeviceBuffer::from_host(s, &tokens)?;
    let mut residual = DeviceBuffer::from_host(s, &input)?;
    let mut normalized = DeviceBuffer::zeroed(s, 1)?;
    let mut amax = DeviceBuffer::zeroed(s, 1)?;
    let mut mean = DeviceBuffer::zeroed(s, 1)?;
    let mut inv = DeviceBuffer::zeroed(s, 1)?;
    e.forward(
        &r,
        &w,
        &td,
        &mut HiddenStateDevice {
            stream: s,
            batch_size: GPT2_BATCH_SIZE as u32,
            seq_len: GPT2_SEQ_LEN as u32,
            row_count: n as u32,
            residual: &mut residual,
            normalized: &mut normalized,
            normalized_amax: &mut amax,
            mean: &mut mean,
            inv_std: &mut inv,
        },
    )?;
    let outputs = [0, 1, 7, 8, 9, 10, 2047, 2048, 2057, 8191, 8192, 16383];
    let needed: BTreeSet<_> = outputs
        .into_iter()
        .flat_map(|p| {
            [0, 3, 6, 9]
                .into_iter()
                .filter_map(move |lag| (p % GPT2_SEQ_LEN >= lag).then(|| p - lag))
        })
        .collect();
    let map = w.token_map.to_host_vec(s)?;
    let table = unpack(s, &w.nv[0])?;
    let key_weights = (0..streams)
        .map(|j| unpack(s, &w.nv[w.cfg.key_index(j)]))
        .collect::<AppResult<Vec<_>>>()?;
    let key_biases = (0..streams)
        .map(|j| unpack(s, &w.nv[w.cfg.key_index(j) + 1]))
        .collect::<AppResult<Vec<_>>>()?;
    let value_weight = unpack(s, &w.nv[w.cfg.value_index()])?;
    let value_bias = unpack(s, &w.nv[w.cfg.value_index() + 1])?;
    let q = e.memory_q.rowwise();
    let qb = q.bytes.to_host_vec(s)?;
    let qs = q.scales.to_host_vec(s)?;
    let qg = q.global_scales.to_host_vec(s)?;
    let mut expected_u = HashMap::new();
    let mut expected_un = HashMap::new();
    let mut max_projection_error = 0f64;
    let mut max_output_error = 0f64;
    for p in needed.iter().copied() {
        let gathered = row(s, &e.memory, p, MEMORY)?;
        let mut offset = 0;
        for head in 0..HEADS * 2 {
            let order = 2 + head / HEADS;
            let mut mix = 0u64;
            let mut valid = true;
            for lag in 0..order {
                let mut token = llama2_tokenizer::EOS_TOKEN;
                if valid && lag <= p % GPT2_SEQ_LEN {
                    let raw = tokens[p - lag];
                    if lag > 0
                        && [llama2_tokenizer::BOS_TOKEN, llama2_tokenizer::EOS_TOKEN].contains(&raw)
                    {
                        valid = false;
                    } else {
                        token = map[raw as usize];
                        if [llama2_tokenizer::BOS_TOKEN, llama2_tokenizer::EOS_TOKEN].contains(&raw)
                        {
                            valid = false;
                        }
                    }
                }
                mix ^= u64::from(token).wrapping_mul(w.cfg.multipliers[lag]);
            }
            let id = offset + (mix % u64::from(w.cfg.primes[head])) as usize;
            for j in 0..HEAD_DIM {
                // Exact stored-value decode rounded to the FP32 gather output.
                // An absolute1e-7 bound rejects correct values above2 by <0.5ULP.
                assert_eq!(
                    gathered[head * HEAD_DIM + j].to_bits(),
                    (table[id * HEAD_DIM + j] as f32).to_bits(),
                    "hash/gather at row {p}, head {head}, feature {j}"
                );
            }
            offset += w.cfg.primes[head] as usize;
        }
        let memory = (0..MEMORY)
            .map(|j| decode(&qb, code(&qs, p, j / 16, MEMORY, 0), qg[p], p * MEMORY + j))
            .collect::<Vec<_>>();
        for stream_index in 0..streams {
            let mut projections = Vec::new();
            for (weights, bias, actual) in [
                (
                    &key_weights[stream_index],
                    &key_biases[stream_index],
                    &e.key,
                ),
                (&value_weight, &value_bias, &e.value),
            ] {
                let expected = (0..d)
                    .map(|j| {
                        bias[j]
                            + (0..MEMORY)
                                .map(|k| weights[j * MEMORY + k] * memory[k])
                                .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let actual_row = if std::ptr::eq(actual, &e.value) {
                    p
                } else {
                    p * streams + stream_index
                };
                for (a, b) in row(s, actual, actual_row, d)?.into_iter().zip(&expected) {
                    max_projection_error = max_projection_error.max((f64::from(a) - b).abs());
                }
                projections.push(expected);
            }
            let gamma_base = stream_index * d;
            let h = &input[(p * streams + stream_index) * d..(p * streams + stream_index + 1) * d];
            let key = &projections[0];
            let value = &projections[1];
            let gate_epsilon = f64::from(w.cfg.gate_rms_epsilon());
            let ih = (h.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / d as f64
                + gate_epsilon)
                .sqrt()
                .recip();
            let ik = (key.iter().map(|x| x * x).sum::<f64>() / d as f64 + gate_epsilon)
                .sqrt()
                .recip();
            let dot = (0..d)
                .map(|j| {
                    f64::from(h[j])
                        * ih
                        * f64::from(gamma[0][gamma_base + j])
                        * key[j]
                        * ik
                        * f64::from(gamma[1][gamma_base + j])
                })
                .sum::<f64>()
                / (d as f64).sqrt();
            let signed_root = if dot == 0. {
                0.
            } else {
                dot.signum() * dot.abs().max(1e-6).sqrt()
            };
            let gate = sigmoid(signed_root);
            let u = value.iter().map(|v| gate * v).collect::<Vec<_>>();
            let iu = (u.iter().map(|v| v * v).sum::<f64>() / d as f64 + 1e-5)
                .sqrt()
                .recip();
            let un = (0..d)
                .map(|j| u[j] * iu * f64::from(gamma[2][gamma_base + j]))
                .collect::<Vec<_>>();
            expected_u.insert((p, stream_index), u);
            expected_un.insert((p, stream_index), un);
        }
    }
    for p in outputs {
        for stream_index in 0..streams {
            let actual = row(s, &residual, p * streams + stream_index, d)?;
            for j in 0..d {
                let conv = (0..4)
                    .filter_map(|c| {
                        let lag = 3 * (3 - c);
                        (p % GPT2_SEQ_LEN >= lag).then(|| {
                            f64::from(gamma[3][(stream_index * d + j) * 4 + c])
                                * expected_un[&(p - lag, stream_index)][j]
                        })
                    })
                    .sum::<f64>();
                let expected = f64::from(input[(p * streams + stream_index) * d + j])
                    + expected_u[&(p, stream_index)][j]
                    + conv * sigmoid(conv);
                max_output_error = max_output_error.max((f64::from(actual[j]) - expected).abs());
            }
        }
    }
    assert!(
        max_projection_error < 3e-5,
        "NVFP4 dense projection error {max_projection_error}"
    );
    assert!(
        max_output_error < 3e-5,
        "complete Engram output error {max_output_error}"
    );
    println!(
        "{}",
        serde_json::json!({"passed":true,"batch":GPT2_BATCH_SIZE,"context":GPT2_SEQ_LEN,"width":d,"streams":streams,"layer":w.cfg.layer,
        "checked_output_rows":outputs,"checked_projection_rows":needed.len(),"max_projection_abs_error":max_projection_error,
        "max_complete_output_abs_error":max_output_error,"optimizer_updates":0,"dense_products":"native NVFP4 TMA","oracle":"host f64 verification only"})
    );
    if streams == 4 {
        // Observe each real native dense adjoint before the host loop reuses
        // its workspace; independently check complete accumulation/scatter.
        e.memory_adjoint_verification =
            Some(super::adjoint_verification::MemoryAdjointVerification::new());
        let upstream = (0..n * streams * d)
            .map(|i| ((i * 11 % 73) as f32 - 36.) * 0.002)
            .collect::<Vec<_>>();
        let mut direct = DeviceBuffer::from_host(s, &upstream)?;
        let mut scratch =
            super::backward_scratch::BackwardScratch::new(s, w.cfg.memory(), streams)?;
        e.backward(&r, &w, &mut direct, &mut scratch)?;
        e.memory_adjoint_verification.as_ref().unwrap().verify(
            s,
            &scratch.d_mem_key,
            &e.ng[0],
            &e.ids,
            n,
            w.cfg.head_dim,
            &w.cfg.projector_indices(),
        )?;
        let names = e.named_gradients();
        // Learned optimizer controls belong to the common Muon preparation;
        // this module-only oracle checks every actual memory model parameter.
        for (name, gradient) in names.into_iter().take(e.ng.len() + e.fg.len()) {
            let values = gradient.to_host_vec(s)?;
            assert!(values.iter().all(|x| x.is_finite()), "nonfinite {name}");
            assert!(values.iter().any(|x| *x != 0.), "inactive {name}");
        }
        let values = direct.to_host_vec(s)?;
        assert!(values.iter().all(|x| x.is_finite()));
        assert_ne!(
            values, upstream,
            "complete direct/query adjoint must be active"
        );
        println!(
            "four_stream_module_native_backward layer={} groups=15 independent_keys=4 shared_value_backward=1 updates=0",
            w.cfg.layer
        );
    }
    Ok(())
}
