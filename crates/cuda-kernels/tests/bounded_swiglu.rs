//! Full5504-wide vector contract declared before implementation; zero updates.
use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::bounded_swiglu::kernels;
mod common;
#[path = "common/f16_round.rs"]
mod half_input;
const N: usize = 256;
const H: usize = 5504;

fn oracle(g: f64, v: f64, dy: f64) -> (f64, f64, f64) {
    let s = if g >= 0. {
        1. / (1. + (-g).exp())
    } else {
        let e = g.exp();
        e / (1. + e)
    };
    let es = (-g.abs()).exp();
    let ds = es / (1. + es).powi(2);
    let tg = (g / 4.).tanh();
    let tu = (v / 25.).tanh();
    let eg = (-2. * (g / 4.).abs()).exp();
    let eu = (-2. * (v / 25.).abs()).exp();
    let dtg = 4. * eg / (1. + eg).powi(2);
    let dtu = 4. * eu / (1. + eu).powi(2);
    (
        4. * tg * s * 25. * tu,
        dy * 25. * tu * (dtg * s + 4. * tg * ds),
        dy * 4. * tg * s * dtu,
    )
}

#[test]
#[ignore = "native complete5504-wide bounded activation and both routed adjoints; host oracle only"]
fn native_full_width_bounded_swiglu_raw_sigmoid_tails_and_both_route_halves(
) -> Result<(), Box<dyn std::error::Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = kernels::from_module(ptx)?;
    let gates = [
        -1e6f32, -100., -40., -20., -4., -0.5, -1e-7, -1e-12, -0., 0., 1e-12, 1e-7, 0.5, 4., 20.,
        40., 100., 1e6,
    ];
    let ups = [
        -1e6f32, -300., -25., -1., -1e-8, 0., 1e-8, 1., 25., 300., 1e6,
    ];
    let mut raw = vec![0.; N * 2 * H];
    let mut upstream = vec![0.; N * H];
    for r in 0..N {
        for j in 0..H {
            raw[r * 2 * H + j] = gates[(r + j) % gates.len()];
            raw[r * 2 * H + H + j] = ups[(j / gates.len() + r) % ups.len()];
            upstream[r * H + j] = if (r + j) % 2 == 0 { 0.37 } else { -0.61 };
        }
    }
    let source = DeviceBuffer::from_host(&stream, &raw)?;
    let dy = DeviceBuffer::from_host(&stream, &upstream)?;
    let mut output = DeviceBuffer::zeroed(&stream, N * H)?;
    let mut gradient = DeviceBuffer::from_host(&stream, &raw)?;
    let masks = DeviceBuffer::from_host(
        &stream,
        &[(1u64 << 32) - 1, ((1u64 << 43) - 1) ^ ((1u64 << 11) - 1)],
    )?;
    let grid = LaunchConfig {
        grid_dim: ((N * H).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    module.bounded_swiglu_forward_kernel(
        &stream,
        grid,
        &source,
        &mut output,
        N as u32,
        H as u32,
    )?;
    module.bounded_swiglu_backward_kernel(
        &stream,
        grid,
        &mut gradient,
        &dy,
        &masks,
        N as u32,
        H as u32,
        true,
    )?;
    let output = output.to_host_vec(&stream)?;
    let gradient = gradient.to_host_vec(&stream)?;
    let mut maximum_error = 0f64;
    let mut retained_tails = 0;
    for r in 0..N {
        for j in 0..H {
            let g = half_input::round(raw[r * 2 * H + j]);
            let v = half_input::round(raw[r * 2 * H + H + j]);
            let (expected, dg, dv) = oracle(g.into(), v.into(), upstream[r * H + j].into());
            let a = f64::from(output[r * H + j]);
            assert!(a.is_finite() && a.abs() <= 100.00001);
            let error = (a - expected).abs();
            assert!(
                error <= expected.abs() * 4e-6 + 1e-12,
                "forward g={g} v={v} actual={a} expected={expected}"
            );
            maximum_error = maximum_error.max(error);
            let active = if r < 128 { j / 128 < 32 } else { j / 128 >= 11 };
            for (actual, expected) in [
                (gradient[r * 2 * H + j], dg),
                (gradient[r * 2 * H + H + j], dv),
            ] {
                assert!(actual.is_finite());
                if active {
                    assert!(
                        (f64::from(actual) - expected).abs() <= expected.abs() * 1e-5 + 1e-31,
                        "adjoint g={g} v={v} actual={actual} expected={expected}"
                    );
                    if expected.abs() > 1e-30 && expected.abs() < 1e-8 {
                        assert_ne!(
                            actual, 0.,
                            "rounded activation must not erase a representable analytic tail"
                        );
                        retained_tails += 1;
                    }
                } else {
                    assert_eq!(actual, 0., "both routed halves must be exactly zero");
                }
            }
        }
    }
    assert!(retained_tails > 1000);
    println!(
        "bounded_swiglu_vector_verified rows={N} width={H} paired_width={} raw_sigmoid=true two_caps=4,25 finite_raw_through=1000000 both_route_halves=true retained_tail_cases={retained_tails} max_forward_error={maximum_error} updates=0",
        2 * H
    );
    Ok(())
}

#[test]
#[ignore = "native same-half-cell precision witnesses; full5504-wide forward and both adjoints, zero updates"]
fn native_bounded_swiglu_cannot_distinguish_inputs_inside_the_same_half_cell(
) -> Result<(), Box<dyn std::error::Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = kernels::from_module(ptx)?;
    let cases = [
        (1.00001f32, 3.00001f32),
        (1.00002, 3.00002),
        (-0.50003, -25.0001),
        (-0.50004, -25.0002),
        (1e6, 1e6),
        (2e6, 2e6),
        (1e-12, 1.00001),
        (2e-12, 1.00002),
    ];
    let n = cases.len();
    let mut raw = vec![0.; n * 2 * H];
    for (row, (gate, up)) in cases.iter().copied().enumerate() {
        raw[row * 2 * H..row * 2 * H + H].fill(gate);
        raw[row * 2 * H + H..(row + 1) * 2 * H].fill(up);
    }
    let source = DeviceBuffer::from_host(&stream, &raw)?;
    let mut gradient = DeviceBuffer::from_host(&stream, &raw)?;
    let upstream = DeviceBuffer::from_host(&stream, &vec![0.37f32; n * H])?;
    let masks = DeviceBuffer::from_host(&stream, &[u64::MAX])?;
    let mut output = DeviceBuffer::zeroed(&stream, n * H)?;
    let grid = LaunchConfig {
        grid_dim: ((n * H).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    module.bounded_swiglu_forward_kernel(
        &stream,
        grid,
        &source,
        &mut output,
        n as u32,
        H as u32,
    )?;
    module.bounded_swiglu_backward_kernel(
        &stream,
        grid,
        &mut gradient,
        &upstream,
        &masks,
        n as u32,
        H as u32,
        false,
    )?;
    let output = output.to_host_vec(&stream)?;
    let gradient = gradient.to_host_vec(&stream)?;
    for row in (0..n).step_by(2) {
        let (g, v) = (
            half_input::round(cases[row].0),
            half_input::round(cases[row].1),
        );
        assert_eq!(g.to_bits(), half_input::round(cases[row + 1].0).to_bits());
        assert_eq!(v.to_bits(), half_input::round(cases[row + 1].1).to_bits());
        let expected = oracle(f64::from(g), f64::from(v), f64::from(0.37f32));
        for feature in 0..H {
            assert_eq!(
                output[row * H + feature].to_bits(),
                output[(row + 1) * H + feature].to_bits(),
                "forward retained forbidden extra input precision row={row} feature={feature}"
            );
            for half in 0..2 {
                assert_eq!(gradient[row * 2 * H + half * H + feature].to_bits(), gradient[(row + 1) * 2 * H + half * H + feature].to_bits(), "adjoint retained forbidden extra input precision row={row} half={half} feature={feature}");
            }
            for (actual, expected) in [
                (output[row * H + feature], expected.0),
                (gradient[row * 2 * H + feature], expected.1),
                (gradient[row * 2 * H + H + feature], expected.2),
            ] {
                assert!(actual.is_finite());
                assert!((f64::from(actual) - expected).abs() <= expected.abs() * 1e-5 + 1e-31);
            }
        }
    }
    println!("bounded_swiglu_half_input_cells_verified rows={n} width={H} forward_and_both_adjoints=bit_exact input_bits=16 updates=0");
    Ok(())
}
