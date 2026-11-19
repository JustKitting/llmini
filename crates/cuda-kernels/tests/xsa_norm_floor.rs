//! Native XSA floor adjoints against finite differences of the full operator.
//! Host f64 evaluates only the independent verification objective.
use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::attention::{
    AttentionModule, ExclusiveSelfAttentionBackwardArgs, ExclusiveSelfAttentionForwardArgs,
};
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, csf};
use std::error::Error;
mod common;

fn half(x: f32) -> u16 {
    // Fixtures are exact binary powers and their small integer multiples.
    let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
    let x = x.abs();
    if x < 2f32.powi(-14) {
        sign | ((x * 16777216.).round() as u16)
    } else {
        let bits = x.to_bits();
        sign | (((((bits >> 23) & 255) - 112) as u16) << 10) | ((bits >> 13) & 1023) as u16
    }
}
fn objective(v: &[f64], out: &[f64], g: &[f64], alpha: f64, gate: f64, kda: bool) -> f64 {
    let values = v
        .iter()
        .map(|&x| if kda { x / (1. + (-x).exp()) } else { x })
        .collect::<Vec<_>>();
    let norm = values
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(f64::from(1e-4f32));
    let projection = out
        .iter()
        .zip(&values)
        .map(|(o, v)| o * v / norm)
        .sum::<f64>();
    let scale = 1. / (1. + (-gate).exp());
    out.iter()
        .zip(&values)
        .zip(g)
        .map(|((&o, &v), &e)| e * (o - alpha.tanh() * projection * v / norm) * scale)
        .sum()
}
fn close(actual: f32, expected: f64, label: &str) {
    assert!(
        actual.is_finite() && (f64::from(actual) - expected).abs() <= 2e-4 + expected.abs() * 2e-4,
        "{label}: actual={actual:e} finite_difference={expected:e} error={:e}",
        (f64::from(actual) - expected).abs()
    );
}

#[test]
#[ignore = "native XSA tiny/zero/ordinary norms,64-wide and general heads, full/KDA activation and all adjoints; zero updates"]
fn native_xsa_norm_floor_matches_full_operator_finite_differences() -> Result<(), Box<dyn Error>> {
    let (_, s, m) = common::cuda_test_module(AttentionModule::from_module)?;
    for head_dim in [64, 32, 96] {
        for amp in [0., 2f32.powi(-17), 2f32.powi(-10)] {
            for kda in [false, true] {
                check(&s, &m, head_dim, amp, kda)?;
            }
        }
    }
    Ok(())
}
fn check(
    s: &CudaStream,
    m: &AttentionModule,
    h: usize,
    amp: f32,
    kda: bool,
) -> Result<(), Box<dyn Error>> {
    let rows = 3;
    let heads = 2;
    let d = h * heads;
    let stride = 3 * d + heads + 4;
    let gate_offset = 3 * d;
    let mut qkv = vec![0.; rows * stride];
    for r in 0..rows {
        for head in 0..heads {
            for i in 0..h {
                qkv[r * stride + 2 * d + head * h + i] =
                    amp * [1., -0.5, 0.25, 0.5][(i + r + head) % 4];
            }
            qkv[r * stride + gate_offset + head] = if head == 0 { 0.5 } else { -0.5 };
        }
    }
    let raw = (0..rows * d)
        .map(|i| [0.125, -0.25, 0.5, 0.25][(i * 3 + i / d) % 4])
        .collect::<Vec<_>>();
    let grad = (0..rows * d)
        .map(|i| [0.25, 0.5, -0.125, 0.75][(i * 7 + i / d) % 4])
        .collect::<Vec<_>>();
    let q = DeviceBuffer::from_host(s, &qkv)?;
    let mut q16 = DeviceBuffer::from_host(s, &qkv.iter().copied().map(half).collect::<Vec<_>>())?;
    let mut o16 = DeviceBuffer::zeroed(s, rows * d)?;
    let mut output = DeviceBuffer::from_host(s, &raw)?;
    let mut payload = vec![0; 64];
    payload[0] = 0xa2;
    let ab = DeviceBuffer::from_host(s, &payload)?;
    let asc = DeviceBuffer::from_host(s, &csf::encode_host(&[0x38], 1, 128))?;
    let ag = DeviceBuffer::from_host(s, &[0.25])?;
    let alpha = Nvfp4DeviceTensor::new(&ab, &asc, &ag);
    m.exclusive_self_attention_forward(ExclusiveSelfAttentionForwardArgs {
        stream: s,
        qkv: &q,
        out: &mut output,
        xsa_alphas: alpha,
        qkv_f16: Some(&mut q16),
        raw_out_f16: Some(&mut o16),
        row_count: rows as u32,
        embedding_dim: d as u32,
        qkv_dim: stride as u32,
        head_count: heads as u32,
        head_dim: h as u32,
        value_offset: (2 * d) as u32,
        gate_offset: gate_offset as u32,
        alpha_offset: 0,
        gate: rust_kernels_cuda::attention::AttentionOutputGate::Sigmoid,
        kda_value_activation: kda,
    })?;
    let mut dout = DeviceBuffer::from_host(s, &grad)?;
    let mut dq = DeviceBuffer::from_host(s, &vec![-17.; rows * stride])?;
    let mut da = DeviceBuffer::zeroed(s, 128)?;
    let mut am = DeviceBuffer::zeroed(s, 128)?;
    m.exclusive_self_attention_backward(ExclusiveSelfAttentionBackwardArgs {
        stream: s,
        qkv_f16: &q16,
        raw_out_f16: &o16,
        d_out: &mut dout,
        d_qkv: &mut dq,
        d_qkv_chunk_amax: &mut am,
        d_xsa_alphas: &mut da,
        xsa_alphas: alpha,
        row_count: rows as u32,
        embedding_dim: d as u32,
        qkv_dim: stride as u32,
        head_count: heads as u32,
        head_dim: h as u32,
        value_offset: (2 * d) as u32,
        gate_offset: gate_offset as u32,
        gate_amax_offset: 0,
        alpha_offset: 0,
        gate: rust_kernels_cuda::attention::AttentionOutputGate::Sigmoid,
        kda_value_activation: kda,
    })?;
    let dx = dq.to_host_vec(s)?;
    let dr = dout.to_host_vec(s)?;
    let dalpha = da.to_host_vec(s)?;
    let actual_forward = output.to_host_vec(s)?;
    let mut expected_alpha = [0.; 2];
    for r in 0..rows {
        for head in 0..heads {
            let vb = r * stride + 2 * d + head * h;
            let ob = r * d + head * h;
            let v = qkv[vb..vb + h]
                .iter()
                .map(|x| f64::from(*x))
                .collect::<Vec<_>>();
            let o = raw[ob..ob + h]
                .iter()
                .map(|x| f64::from(*x))
                .collect::<Vec<_>>();
            let g = grad[ob..ob + h]
                .iter()
                .map(|x| f64::from(*x))
                .collect::<Vec<_>>();
            let a = if head == 0 { 0.25 } else { -0.25 };
            let gate = f64::from(qkv[r * stride + gate_offset + head]);
            let native_objective = actual_forward[ob..ob + h]
                .iter()
                .zip(&g)
                .map(|(&y, &e)| f64::from(y) * e)
                .sum::<f64>();
            close(
                native_objective as f32,
                objective(&v, &o, &g, a, gate, kda),
                "full forward objective",
            );
            let step = if amp == 0. {
                2f64.powi(-30)
            } else {
                f64::from(amp) * 0.0005
            };
            for i in 0..h {
                let mut plus = v.clone();
                let mut minus = v.clone();
                plus[i] += step;
                minus[i] -= step;
                let expected = (objective(&plus, &o, &g, a, gate, kda)
                    - objective(&minus, &o, &g, a, gate, kda))
                    / (2. * step);
                close(
                    dx[vb + i],
                    expected,
                    &format!("V h={h} amp={amp:e} kda={kda} r={r} head={head} i={i}"),
                );
                if h == 64 && amp == 2f32.powi(-17) && !kda && r == 1 && head == 0 && i == 0 {
                    println!(
                        "original_training64_floor_counterexample actual={} independent_finite_difference={expected} abs_error={}",
                        dx[vb + i],
                        (f64::from(dx[vb + i]) - expected).abs()
                    );
                }
                let mut plus = o.clone();
                let mut minus = o.clone();
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                close(
                    dr[ob + i],
                    (objective(&v, &plus, &g, a, gate, kda)
                        - objective(&v, &minus, &g, a, gate, kda))
                        / 2e-5,
                    "output adjoint",
                );
            }
            expected_alpha[head] += (objective(&v, &o, &g, a + 1e-5, gate, kda)
                - objective(&v, &o, &g, a - 1e-5, gate, kda))
                / 2e-5;
            close(
                dx[r * stride + gate_offset + head],
                (objective(&v, &o, &g, a, gate + 1e-5, kda)
                    - objective(&v, &o, &g, a, gate - 1e-5, kda))
                    / 2e-5,
                "gate adjoint",
            );
        }
    }
    for head in 0..heads {
        close(dalpha[head], expected_alpha[head], "alpha adjoint");
    }
    println!(
        "native_xsa_floor_verified rows={rows} heads={heads} head_dim={h} raw_amplitude={amp:e} kda={kda} all_value_output_gate_alpha_adjoints=finite_difference updates=0"
    );
    Ok(())
}
