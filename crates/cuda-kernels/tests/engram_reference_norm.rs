//! Reference query/key RMS counterexamples, declared before the kernel repair.
//! CUDA computes the actual forward/backward; host arithmetic is an independent
//! scalar verification oracle, with zero optimizer updates or CPU training.
use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::engram::kernels;
use std::error::Error;
mod common;

const N: usize = 8;
const D: usize = 2048;
type Inputs = [Vec<f64>; 5]; // hidden, key, value, query gamma, key gamma

fn gate(a: &Inputs, row: usize, epsilon: f64) -> [f64; 4] {
    let b = row * D;
    let ih = (a[0][b..b + D].iter().map(|x| x * x).sum::<f64>() / D as f64 + epsilon)
        .sqrt()
        .recip();
    let ik = (a[1][b..b + D].iter().map(|x| x * x).sum::<f64>() / D as f64 + epsilon)
        .sqrt()
        .recip();
    let raw = (0..D)
        .map(|j| a[0][b + j] * ih * a[3][j] * a[1][b + j] * ik * a[4][j])
        .sum::<f64>()
        / (D as f64).sqrt();
    let root = if raw == 0. {
        0.
    } else {
        raw.signum() * raw.abs().max(1e-6).sqrt()
    };
    [ih, ik, raw, 1. / (1. + (-root).exp())]
}

fn objective(a: &Inputs, upstream: &[f64], epsilon: f64) -> f64 {
    (0..N)
        .map(|r| {
            let g = gate(a, r, epsilon)[3];
            (0..D)
                .map(|j| g * a[2][r * D + j] * upstream[r * D + j])
                .sum::<f64>()
        })
        .sum()
}

fn rows() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[test]
#[ignore = "native sm_120a reference/legacy RMS forward and finite-difference backward"]
fn reference_and_legacy_gate_rms_low_amplitude_backward() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(kernels::from_module)?;
    let mut input: Inputs = [
        vec![0.; N * D],
        vec![0.; N * D],
        (0..N * D).map(|i| (i % 17) as f64 * 0.013 - 0.1).collect(),
        (0..D).map(|j| 0.9 + (j % 7) as f64 * 0.03).collect(),
        (0..D).map(|j| 1.1 - (j % 5) as f64 * 0.02).collect(),
    ];
    input[3][0] = 1.;
    input[4][0] = 1.;
    for row in 1..4 {
        for j in 0..D {
            let pattern = (j % 7) as f64 / 3. - 1.;
            input[0][row * D + j] = pattern * if row == 3 { 0.1 } else { 3e-4 };
            input[1][row * D + j] = pattern * if row == 2 { -2e-4 } else { 2e-4 };
        }
    }
    // Stay on both sides of the signed-root clamp, away from its kink.
    for row in 4..N {
        let target = if row < 6 { 0.5e-6 } else { 2e-6 };
        let root_d = (D as f64).sqrt();
        let square = target * f64::from(f32::EPSILON) * root_d / (1. - target / root_d);
        input[0][row * D] = square.sqrt();
        input[1][row * D] = square.sqrt() * if row % 2 == 0 { 1. } else { -1. };
    }
    // Freeze the exact FP32 inputs consumed by the kernels for the oracle.
    let host: [Vec<f32>; 5] = std::array::from_fn(|i| input[i].iter().map(|x| *x as f32).collect());
    let input: Inputs = std::array::from_fn(|i| host[i].iter().map(|x| f64::from(*x)).collect());
    let upstream_host = host[2]
        .iter()
        .map(|x| *x * 0.01 + 0.001)
        .collect::<Vec<_>>();
    let upstream = upstream_host
        .iter()
        .map(|x| f64::from(*x))
        .collect::<Vec<_>>();
    let buffers = host
        .iter()
        .map(|x| DeviceBuffer::from_host(&stream, x))
        .collect::<Result<Vec<_>, _>>()?;
    let du = DeviceBuffer::from_host(&stream, &upstream_host)?;
    let zero = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let inv_zero = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    assert!((gate(&input, 1, f64::from(f32::EPSILON))[3] - gate(&input, 1, 1e-5)[3]).abs() > 0.1);

    for epsilon in [1e-5_f32, f32::EPSILON] {
        let mut stats = DeviceBuffer::zeroed(&stream, N * 4)?;
        let mut output = DeviceBuffer::zeroed(&stream, N * D)?;
        module.engram_gate_kernel(
            &stream,
            rows(),
            &buffers[0],
            &buffers[1],
            &buffers[2],
            &buffers[3],
            &buffers[4],
            &mut stats,
            &mut output,
            N as u32,
            D as u32,
            epsilon,
        )?;
        let actual_stats = stats.to_host_vec(&stream)?;
        let actual_output = output.to_host_vec(&stream)?;
        for row in 0..N {
            let expected = gate(&input, row, f64::from(epsilon));
            for j in 0..4 {
                let got = f64::from(actual_stats[row * 4 + j]);
                assert!(
                    (got - expected[j]).abs() < 2e-6 * expected[j].abs() + 2e-7,
                    "epsilon={epsilon} row={row} stat={j}: {got} != {}",
                    expected[j]
                );
            }
            for j in 0..D {
                let expected_output = expected[3] * input[2][row * D + j];
                assert!((f64::from(actual_output[row * D + j]) - expected_output).abs() < 2e-7);
            }
        }
        let direct_host = vec![0.125_f32; N * D];
        let mut direct = DeviceBuffer::from_host(&stream, &direct_host)?;
        let mut dk = DeviceBuffer::zeroed(&stream, N * D)?;
        let mut dv = DeviceBuffer::zeroed(&stream, N * D)?;
        let mut d_dot = DeviceBuffer::zeroed(&stream, N)?;
        module.engram_gate_backward_kernel(
            &stream,
            rows(),
            &buffers[0],
            &buffers[1],
            &buffers[2],
            &buffers[3],
            &buffers[4],
            &stats,
            &du,
            &mut direct,
            &mut dk,
            &mut dv,
            &mut d_dot,
            N as u32,
            D as u32,
        )?;
        let mut dgh = DeviceBuffer::zeroed(&stream, D)?;
        let mut dgk = DeviceBuffer::zeroed(&stream, D)?;
        let mut dgu = DeviceBuffer::zeroed(&stream, D)?;
        let mut dw = DeviceBuffer::zeroed(&stream, D * 4)?;
        let parameters = LaunchConfig {
            grid_dim: ((D * 16 / 256) as u32, 1, 1),
            ..rows()
        };
        module.engram_parameter_backward_kernel(
            &stream,
            parameters,
            &buffers[0],
            &buffers[1],
            &buffers[3],
            &buffers[4],
            &stats,
            &d_dot,
            &output,
            &zero,
            &inv_zero,
            &zero,
            &zero,
            &mut dgh,
            &mut dgk,
            &mut dgu,
            &mut dw,
            N as u32,
            D as u32,
            N as u32,
        )?;
        let mut gradients = [
            direct.to_host_vec(&stream)?,
            dk.to_host_vec(&stream)?,
            dv.to_host_vec(&stream)?,
            dgh.to_host_vec(&stream)?,
            dgk.to_host_vec(&stream)?,
        ];
        for v in &mut gradients[0] {
            *v -= 0.125;
        }
        assert!(gradients.iter().flatten().all(|v| v.is_finite()));
        assert!(
            dgu.to_host_vec(&stream)?
                .iter()
                .chain(dw.to_host_vec(&stream)?.iter())
                .all(|v| *v == 0.)
        );
        for which in 0..5 {
            let indices = if which < 3 {
                vec![0, D, D + 17, 6 * D, 7 * D]
            } else {
                vec![0, 1, 17, D - 1]
            };
            for index in indices {
                let delta = if which < 2 {
                    input[which][index].abs().max(1e-7) * 1e-3
                } else {
                    1e-5
                };
                let mut perturbed = input.clone();
                perturbed[which][index] += delta;
                let plus = objective(&perturbed, &upstream, f64::from(epsilon));
                perturbed[which][index] -= 2. * delta;
                let minus = objective(&perturbed, &upstream, f64::from(epsilon));
                let expected = (plus - minus) / (2. * delta);
                let got = f64::from(gradients[which][index]);
                assert!(
                    (got - expected).abs() < 2e-4 + expected.abs() * 8e-3,
                    "epsilon={epsilon} tensor={which} index={index}: {got} != {expected}"
                );
            }
        }
        println!(
            "native gate RMS epsilon={epsilon}: zero/tiny/outlier/signed-root-clamp forward and five backward groups verified; zero updates"
        );
    }
    Ok(())
}
