//! Actual B8/S2048/d2048 value normalization; f64 is verification only.
use cuda_core::DeviceBuffer;
use rust_kernels_cuda::value_rms::ValueRmsModule;
mod common;
#[path = "common/f16_round.rs"]
mod half_input;
const N: usize = 8 * 2048;
const D: usize = 2048;
const H: usize = 64;
fn input(head: usize, j: usize) -> f32 {
    let pattern = [1., -0.5, 0.25, 0.5][j % 4];
    match head % 12 {
        0 => 0.,
        1 => pattern * 1e-12,
        2 => pattern * 0.125,
        3 => pattern * 1e6,
        4 => pattern * 1e30,
        5 => 1e8,
        6 => -100.,
        7 => -110.,
        8 => pattern * 32.,
        9 => pattern * 1e-38,
        10 => pattern * 0.001,
        _ => -1.2784646 + pattern * 0.0001,
    }
}
fn incoming(head: usize, j: usize) -> f32 {
    if [5, 6, 7].contains(&(head % 12)) {
        0.25
    } else {
        [0.125, -0.25, 0.5, 0.25][(j + head) % 4]
    }
}
fn value(x: f64, kda: bool) -> f64 {
    if kda { x / (1. + (-x).exp()) } else { x }
}
fn derivative(x: f64, kda: bool) -> f64 {
    if !kda {
        return 1.;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 1. } else { 0. };
    }
    let e = (-x.abs()).exp();
    let s = if x >= 0. { 1. / (1. + e) } else { e / (1. + e) };
    s + x * e / (1. + e).powi(2)
}
fn close(actual: f32, expected: f64, label: &str) -> f64 {
    let error = (f64::from(actual) - expected).abs();
    let tolerance = if expected.abs() < 1e-30 {
        3e-44 + expected.abs() * 3e-5
    } else {
        3e-6 + expected.abs() * 3e-5
    };
    assert!(
        actual.is_finite() && error <= tolerance,
        "{label} actual={actual:e} oracle={expected:e} error={error:e} tolerance={tolerance:e}"
    );
    error
}
#[test]
#[ignore = "full B8/S2048/d2048 native value-RMS, full/KDA maps and exact epsilon/tail adjoints; zero optimizer updates"]
fn native_full_value_rms_and_complete_reverse_map() -> Result<(), Box<dyn std::error::Error>> {
    let (_, s, m) = common::cuda_test_module(ValueRmsModule::from_module)?;
    for kda in [false, true] {
        let stride = if kda { 8320 } else { 6176 };
        let mut original = vec![0.03125; N * stride];
        let mut grad = vec![0.0625; N * stride];
        for row in 0..N {
            for head in 0..D / H {
                for j in 0..H {
                    let index = row * stride + 2 * D + head * H + j;
                    let case = row * (D / H) + head;
                    original[index] = input(case, j);
                    grad[index] = incoming(case, j);
                }
            }
        }
        let raw = DeviceBuffer::from_host(&s, &original)?;
        let mut qkv = DeviceBuffer::from_host(&s, &original)?;
        let mut d = DeviceBuffer::from_host(&s, &grad)?;
        m.forward(&s, &mut qkv, N as u32, D as u32, stride as u32, kda)?;
        m.backward(&s, &raw, &mut d, N as u32, D as u32, stride as u32, kda)?;
        let actual = qkv.to_host_vec(&s)?;
        let reverse = d.to_host_vec(&s)?;
        let mut retained = 0usize;
        let mut max_forward: f64 = 0.;
        let mut max_reverse: f64 = 0.;
        for row in 0..N {
            for col in 0..stride {
                if col < 2 * D || col >= 3 * D {
                    let i = row * stride + col;
                    assert_eq!(
                        actual[i].to_bits(),
                        original[i].to_bits(),
                        "non-value forward coordinate changed"
                    );
                    assert_eq!(
                        reverse[i].to_bits(),
                        grad[i].to_bits(),
                        "non-value backward coordinate changed"
                    );
                }
            }
            for head in 0..D / H {
                let base = row * stride + 2 * D + head * H;
                let values = (0..H)
                    .map(|j| value(f64::from(original[base + j]), kda))
                    .collect::<Vec<_>>();
                let square = values.iter().map(|v| v * v).sum::<f64>() / H as f64;
                let epsilon = f64::from(1e-6f32);
                let inverse = (square + epsilon).sqrt().recip();
                let represented = (0..H)
                    .map(|j| f64::from(half_input::round(actual[base + j])))
                    .collect::<Vec<_>>();
                let direction_square = represented.iter().map(|v| v * v).sum::<f64>() / H as f64;
                let dot = (0..H)
                    .map(|j| represented[j] * f64::from(grad[base + j]))
                    .sum::<f64>()
                    / H as f64;
                for j in 0..H {
                    let g = f64::from(grad[base + j]);
                    let v = values[j];
                    let dv = if direction_square > 0. {
                        let radial = represented[j] * (dot / direction_square);
                        ((g - radial) + radial * (epsilon / (square + epsilon))) * inverse
                    } else {
                        g * inverse
                    };
                    let expected =
                        dv * derivative(f64::from(half_input::round(original[base + j])), kda);
                    max_forward = max_forward.max(close(
                        actual[base + j],
                        v * inverse,
                        "native value RMS forward",
                    ));
                    max_reverse = max_reverse.max(close(
                        reverse[base + j],
                        expected,
                        "complete value RMS/SILU adjoint",
                    ));
                    assert!(actual[base + j].abs() <= 8.00001);
                    if expected.abs() > 1e-43 && expected.abs() < 1e-25 {
                        assert_ne!(reverse[base + j], 0., "representable derivative erased");
                        retained += 1;
                    }
                }
            }
        }
        assert!(retained > 0);
        println!(
            "native_value_rms_verified B=8 S=2048 d=2048 head=64 kda={kda} epsilon=1e-6 reverse_input_bits=16 reverse_direction_bits=16 scalar_control_bits=32 all_value_entries={} outside_V=bit_exact raw_through=1e30 retained_small_adjoints={retained} max_forward_error={max_forward:e} max_reverse_error={max_reverse:e} updates=0",
            N * D
        );
    }
    Ok(())
}

#[test]
#[ignore = "same-half-cell native value-RMS counterexample; frozen old PTX only for zero-update RED verification"]
fn native_value_rms_reverse_cannot_distinguish_same_half_inputs_with_identical_controls()
-> Result<(), Box<dyn std::error::Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    // Deliberate direct binding: the production enabled loader rejects the old
    // precision signature. This isolated witness must expose its original math.
    let kernels = rust_kernels_cuda::value_rms::kernels::from_module(ptx)?;
    let cases = [
        (1.00001f32, 1.00002f32),
        (1e6, 1e6 + 1.),
        (-100.00001, -100.00002),
    ];
    let rows = 2 * cases.len();
    let stride = 8320;
    let mut raw = vec![0.03125; rows * stride];
    let mut upstream = vec![0.0625; rows * stride];
    for (case, &(a, b)) in cases.iter().enumerate() {
        assert_eq!(
            half_input::round(a).to_bits(),
            half_input::round(b).to_bits()
        );
        for swap in 0..2 {
            for head in 0..D / H {
                for j in 0..H {
                    let index = (case * 2 + swap) * stride + 2 * D + head * H + j;
                    raw[index] = if (j / 32 + swap) % 2 == 0 { a } else { b };
                    upstream[index] = [0.125, -0.25, 0.5, 0.25][(j + head) % 4];
                }
            }
        }
    }
    let raw_gpu = DeviceBuffer::from_host(&stream, &raw)?;
    let mut forward_gpu = DeviceBuffer::from_host(&stream, &raw)?;
    let mut gradient_gpu = DeviceBuffer::from_host(&stream, &upstream)?;
    let launch = cuda_core::LaunchConfig {
        grid_dim: ((rows * (D / H)).div_ceil(8) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    kernels.value_rms_forward_kernel(
        &stream,
        launch,
        &mut forward_gpu,
        rows as u32,
        D as u32,
        stride as u32,
        1,
    )?;
    kernels.value_rms_backward_kernel(
        &stream,
        launch,
        &raw_gpu,
        &mut gradient_gpu,
        rows as u32,
        D as u32,
        stride as u32,
        1,
    )?;
    let forward = forward_gpu.to_host_vec(&stream)?;
    let gradient = gradient_gpu.to_host_vec(&stream)?;
    for case in 0..cases.len() {
        for j in 0..D {
            let a = (case * 2) * stride + 2 * D + j;
            let b = a + stride;
            assert_eq!(
                half_input::round(forward[a]).to_bits(),
                half_input::round(forward[b]).to_bits(),
                "witness must preserve actual represented forward direction"
            );
            assert!(gradient[a].is_finite() && gradient[b].is_finite());
            assert_eq!(
                gradient[a].to_bits(),
                gradient[b].to_bits(),
                "value-RMS retained forbidden extra reverse precision case={case} feature={j}"
            );
        }
    }
    println!(
        "value_RMS_represented_precision_witness_verified cases={} width={D} ordinary_overflow_and_negative_tail=true adjoints=bit_exact inverse_controls=pair_permutation_identical updates=0",
        cases.len()
    );
    Ok(())
}
