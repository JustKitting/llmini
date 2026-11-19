use super::*;
use crate::{
    AppResult,
    training::{Trainer, runtime::Runtime},
};
use cuda_core::{DeviceBuffer, LaunchConfig};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_SEQ_LEN};

#[test]
#[ignore = "CUDA real-width causal RMS oracle; no optimizer updates"]
fn native_feedback_shifted_rms_matches_independent_oracle() -> AppResult {
    let r = Runtime::new()?;
    let s = r.stream.as_ref();
    let d = GPT2_N_EMBD;
    let seq = 4;
    let n = 12;
    let top = (0..n * d)
        .map(|i| ((i * 13 % 97) as f32 - 48.0) * 0.03125)
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(s, &top)?;
    let mut out = DeviceBuffer::zeroed(s, n * d)?;
    r.latent_feedback_module()?
        .latent_feedback_shifted_rms_kernel(
            s,
            LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &input,
            &mut out,
            n as u32,
            d as u32,
            seq as u32,
            EPSILON,
        )?;
    let got = out.to_host_vec(s)?;
    for row in 0..n {
        let inverse = if row % seq == 0 {
            0.0
        } else {
            let square = top[(row - 1) * d..row * d]
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                / d as f64;
            1.0 / (square + f64::from(EPSILON)).sqrt()
        };
        for j in 0..d {
            let expected = if row % seq == 0 {
                0.0
            } else {
                f64::from(top[(row - 1) * d + j]) * inverse
            };
            assert!((f64::from(got[row * d + j]) - expected).abs() < 2e-6);
        }
    }
    Ok(())
}

pub(super) fn assert_projection(t: &Trainer) -> AppResult {
    let f = t.latent_feedback.as_ref().unwrap();
    let w = &t.uploaded.latent_feedback.as_ref().unwrap().weight;
    let s = t.runtime.stream.as_ref();
    let d = GPT2_N_EMBD;
    let wb = w.bytes.to_host_vec(s)?;
    let ws = w.scales.to_host_vec(s)?;
    let wg = f64::from(w.global_scale.to_host_vec(s)?[0]);
    let input = f.input_q.rowwise();
    let ab = input.bytes.to_host_vec(s)?;
    let raw_scales = input.scales.to_host_vec(s)?;
    let ag = input.global_scales.to_host_vec(s)?;
    let mut scales = vec![0u8; 128];
    scales[..4].copy_from_slice(&(d as u32).to_le_bytes());
    scales[4..8].copy_from_slice(&(ag.len() as u32).to_le_bytes());
    scales[8] = (d / 16).trailing_zeros() as u8;
    scales[9] = 7;
    scales.extend_from_slice(&raw_scales);
    for row in [
        0,
        1,
        GPT2_SEQ_LEN - 1,
        GPT2_SEQ_LEN,
        GPT2_SEQ_LEN + 1,
        4 * GPT2_SEQ_LEN,
        4 * GPT2_SEQ_LEN + 7,
        8 * GPT2_SEQ_LEN - 1,
    ] {
        let mut got = vec![0.0f32; d];
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                got.as_mut_ptr(),
                f.projected.cu_deviceptr() + (row * d * 4) as u64,
                d * 4,
                s.cu_stream(),
            )?;
        }
        s.synchronize()?;
        for col in [0, 1, 15, 31, 127, 511, 1023, 2047] {
            let mut expected = 0.0;
            for k in 0..d {
                let ai = row * d + k;
                let bi = col * d + k;
                let a = decode(&ab, &scales, ai, f64::from(ag[row]));
                let b = decode(&wb, &ws, bi, wg);
                expected += a * b;
            }
            assert!(
                (f64::from(got[col]) - expected).abs() < 5e-5 * (1.0 + expected.abs()),
                "native feedback projection row={row} col={col} got={} oracle={expected}",
                got[col]
            );
        }
    }
    Ok(())
}
fn decode(bytes: &[u8], scales: &[u8], index: usize, global: f64) -> f64 {
    let byte = bytes[index / 2];
    let code = if index % 2 == 0 { byte & 15 } else { byte >> 4 };
    let magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = magnitudes[(code & 7) as usize] * if code & 8 != 0 { -1.0 } else { 1.0 };
    let scale = rust_kernels_cuda::nvfp4::csf::scale_at(scales, index / 16);
    let e = (scale >> 3) & 15;
    let m = scale & 7;
    let fp8 = if e == 0 {
        f64::from(m) / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + f64::from(m) / 8.0) * 2f64.powi(i32::from(e) - 7)
    };
    value * fp8 * global
}
