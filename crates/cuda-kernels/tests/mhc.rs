//! Full finite-Sinkhorn boundary contracts, declared before native implementation.
//! These are vector/coefficient correctness fixtures, not a trained mHC model.
//! Full 32-boundary integration, NVFP4 projectors and long-run quality are separate gates.
use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::mhc::kernels;
use std::error::Error;
mod common;
const N: usize = 8;
const D: usize = 2048;
const STRIDE: usize = 128;
const EPS: f64 = 1e-6;
type Inputs = [Vec<f64>; 5]; // projected coefficients, biases, scales, streams, branch output
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn coefficient(a: &Inputs, r: usize) -> [f64; 24] {
    let mut c = [0.; 24];
    for i in 0..24 {
        let group = if i < 4 {
            0
        } else if i < 8 {
            1
        } else {
            2
        };
        c[i] = a[0][r * STRIDE + i] * a[2][group] + a[1][i];
    }
    for i in 0..4 {
        c[i] = sigmoid(c[i]) + EPS;
        c[4 + i] = 2. * sigmoid(c[4 + i]);
    }
    for j in 0..4 {
        let m = (0..4)
            .map(|k| c[8 + j * 4 + k])
            .fold(f64::NEG_INFINITY, f64::max);
        let s = (0..4).map(|k| (c[8 + j * 4 + k] - m).exp()).sum::<f64>();
        for k in 0..4 {
            c[8 + j * 4 + k] = (c[8 + j * 4 + k] - m).exp() / s + EPS;
        }
    }
    for stage in 1..40 {
        for g in 0..4 {
            let at = |j: usize| {
                if stage % 2 == 1 {
                    8 + j * 4 + g
                } else {
                    8 + g * 4 + j
                }
            };
            let denom = (0..4).map(|j| c[at(j)]).sum::<f64>() + EPS;
            for j in 0..4 {
                c[at(j)] /= denom;
            }
        }
    }
    c
}
fn objective(a: &Inputs, du: &[f64], g: &[f64]) -> f64 {
    let mut result = 0.;
    for r in 0..N {
        let c = coefficient(a, r);
        for d in 0..D {
            let u = (0..4)
                .map(|j| c[j] * a[3][(r * 4 + j) * D + d])
                .sum::<f64>();
            result += u * du[r * D + d];
            for k in 0..4 {
                let y = (0..4)
                    .map(|j| c[8 + j * 4 + k] * a[3][(r * 4 + j) * D + d])
                    .sum::<f64>()
                    + c[4 + k] * a[4][r * D + d];
                result += y * g[(r * 4 + k) * D + d];
            }
        }
    }
    result
}
fn rows() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn grid(len: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (len.div_ceil(256) as u32, 1, 1),
        ..rows()
    }
}
#[test]
#[ignore = "native four-stream/d2048 finite20-round forward and complete boundary adjoint"]
fn finite_sinkhorn_boundary_forward_and_all_adjoint_paths() -> Result<(), Box<dyn Error>> {
    let (_, stream, m) = common::cuda_test_module(kernels::from_module)?;
    let values: Inputs = [
        (0..N * STRIDE)
            .map(|i| ((i * 17 % 73) as f64 - 36.) * 0.071)
            .collect(),
        (0..24)
            .map(|i| ((i * 11 % 17) as f64 - 8.) * 0.043)
            .collect(),
        vec![0.41, 0.73, 0.57],
        (0..N * 4 * D)
            .map(|i| ((i * 19 % 101) as f64 - 50.) * 0.013)
            .collect(),
        (0..N * D)
            .map(|i| ((i * 7 % 89) as f64 - 44.) * 0.017)
            .collect(),
    ];
    let host: [Vec<f32>; 5] =
        std::array::from_fn(|i| values[i].iter().map(|x| *x as f32).collect());
    let values: Inputs = std::array::from_fn(|i| host[i].iter().map(|x| f64::from(*x)).collect());
    let du_host = (0..N * D)
        .map(|i| ((i * 3 % 23) as f32 - 11.) * 0.009)
        .collect::<Vec<_>>();
    let g_host = (0..N * 4 * D)
        .map(|i| ((i * 13 % 31) as f32 - 15.) * 0.007)
        .collect::<Vec<_>>();
    let du = du_host.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let g = g_host.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let b = host
        .iter()
        .map(|x| DeviceBuffer::from_host(&stream, x))
        .collect::<Result<Vec<_>, _>>()?;
    let dud = DeviceBuffer::from_host(&stream, &du_host)?;
    let gd = DeviceBuffer::from_host(&stream, &g_host)?;
    let mut c = DeviceBuffer::zeroed(&stream, N * 24)?;
    let mut u = DeviceBuffer::zeroed(&stream, N * D)?;
    let mut y = DeviceBuffer::zeroed(&stream, N * 4 * D)?;
    m.mhc_coefficients_kernel(
        &stream,
        rows(),
        &b[0],
        &b[1],
        &b[2],
        &mut c,
        N as u32,
        STRIDE as u32,
        false,
    )?;
    m.mhc_pre_kernel(&stream, grid(N * D), &b[3], &c, &mut u, N as u32, D as u32)?;
    m.mhc_post_kernel(
        &stream,
        grid(N * 4 * D),
        &b[3],
        &b[4],
        &c,
        &mut y,
        N as u32,
        D as u32,
    )?;
    let ch = c.to_host_vec(&stream)?;
    let uh = u.to_host_vec(&stream)?;
    let yh = y.to_host_vec(&stream)?;
    assert!(
        (ch[9] - ch[12]).abs() > 1e-4,
        "fixture must expose mixer transposition"
    );
    for r in 0..N {
        let expected = coefficient(&values, r);
        for i in 0..24 {
            assert!((f64::from(ch[r * 24 + i]) - expected[i]).abs() < 3e-6);
        }
        for j in 0..4 {
            let row = (0..4)
                .map(|k| f64::from(ch[r * 24 + 8 + j * 4 + k]))
                .sum::<f64>();
            let col = (0..4)
                .map(|k| f64::from(ch[r * 24 + 8 + k * 4 + j]))
                .sum::<f64>();
            assert!((row - 1.).abs() < 1e-4 && (col - 1.).abs() < 1e-4);
        }
        for d in 0..D {
            let expected_u = (0..4)
                .map(|j| expected[j] * values[3][(r * 4 + j) * D + d])
                .sum::<f64>();
            assert!((f64::from(uh[r * D + d]) - expected_u).abs() < 4e-6);
            for k in 0..4 {
                let expected_y = (0..4)
                    .map(|j| expected[8 + j * 4 + k] * values[3][(r * 4 + j) * D + d])
                    .sum::<f64>()
                    + expected[4 + k] * values[4][r * D + d];
                assert!((f64::from(yh[(r * 4 + k) * D + d]) - expected_y).abs() < 4e-6);
            }
        }
    }
    let mut df = DeviceBuffer::zeroed(&stream, N * D)?;
    let mut dx = DeviceBuffer::zeroed(&stream, N * 4 * D)?;
    let mut dc = DeviceBuffer::zeroed(&stream, N * 24)?;
    let mut dr = DeviceBuffer::zeroed(&stream, N * STRIDE)?;
    let mut db = DeviceBuffer::zeroed(&stream, 24)?;
    let mut da = DeviceBuffer::zeroed(&stream, 3)?;
    m.mhc_post_backward_kernel(
        &stream,
        rows(),
        &b[3],
        &b[4],
        &c,
        &gd,
        &mut df,
        &mut dx,
        &mut dc,
        N as u32,
        D as u32,
    )?;
    m.mhc_pre_backward_kernel(
        &stream,
        rows(),
        &b[3],
        &c,
        &dud,
        &mut dx,
        &mut dc,
        N as u32,
        D as u32,
        false,
    )?;
    m.mhc_coefficients_backward_kernel(
        &stream,
        rows(),
        &b[0],
        &b[1],
        &b[2],
        &dc,
        &mut dr,
        &mut db,
        &mut da,
        N as u32,
        STRIDE as u32,
        false,
    )?;
    let actual = [
        dr.to_host_vec(&stream)?,
        db.to_host_vec(&stream)?,
        da.to_host_vec(&stream)?,
        dx.to_host_vec(&stream)?,
        df.to_host_vec(&stream)?,
    ];
    assert!(actual.iter().flatten().all(|x| x.is_finite()));
    for r in 0..N {
        assert!(
            actual[0][r * STRIDE + 24..(r + 1) * STRIDE]
                .iter()
                .all(|x| *x == 0.),
            "padding gradients must be structural zero"
        );
    }
    for which in 0..5 {
        let indices = match which {
            0 => vec![0, 3, 4, 7, 8, 9, 12, 23, STRIDE + 11],
            1 => (0..24).collect(),
            2 => vec![0, 1, 2],
            3 => vec![0, D + 17, 2 * D + 3, 3 * D + 127, N * 4 * D - 1],
            _ => vec![0, 17, D, N * D - 1],
        };
        for i in indices {
            let mut a = values.clone();
            let delta = 1e-4;
            a[which][i] += delta;
            let plus = objective(&a, &du, &g);
            a[which][i] -= 2. * delta;
            let minus = objective(&a, &du, &g);
            let expected = (plus - minus) / (2. * delta);
            let got = f64::from(actual[which][i]);
            assert!(
                (got - expected).abs() < 3e-4 + expected.abs() * 0.004,
                "tensor={which} index={i}: {got} != {expected}"
            );
        }
    }
    println!(
        "native finite20 mHC boundary: nonsymmetric orientation, all maps and finite-program adjoints verified; no optimizer updates"
    );
    Ok(())
}

#[test]
#[ignore = "native flattened RMS, four-way expansion/reduction and learned HyperHead"]
fn expansion_flattened_rms_and_hyperhead_adjoint() -> Result<(), Box<dyn Error>> {
    let (_, stream, m) = common::cuda_test_module(kernels::from_module)?;
    let embedding = (0..N * D)
        .map(|i| {
            if i < D {
                0.
            } else {
                ((i * 29 % 97) as f32 - 48.) * 0.003
            }
        })
        .collect::<Vec<_>>();
    let upstream = (0..N * 4 * D)
        .map(|i| ((i * 7 % 41) as f32 - 20.) * 0.013)
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&stream, &embedding)?;
    let dy = DeviceBuffer::from_host(&stream, &upstream)?;
    let mut x = DeviceBuffer::zeroed(&stream, N * 4 * D)?;
    let mut normalized = DeviceBuffer::zeroed(&stream, N * 4 * D)?;
    let mut inv = DeviceBuffer::zeroed(&stream, N)?;
    let mut reduced = DeviceBuffer::zeroed(&stream, N * D)?;
    m.mhc_expand_kernel(&stream, grid(N * 4 * D), &input, &mut x, N as u32, D as u32)?;
    m.mhc_expand_backward_kernel(&stream, grid(N * D), &dy, &mut reduced, N as u32, D as u32)?;
    let streams = x.to_host_vec(&stream)?;
    let reduction = reduced.to_host_vec(&stream)?;
    for r in 0..N {
        for d in 0..D {
            for j in 0..4 {
                assert_eq!(streams[(r * 4 + j) * D + d], embedding[r * D + d]);
            }
            let expected = (0..4).map(|j| upstream[(r * 4 + j) * D + d]).sum::<f32>();
            assert!((reduction[r * D + d] - expected).abs() < 1e-6);
        }
    }
    m.mhc_rms_kernel(
        &stream,
        rows(),
        &x,
        &mut normalized,
        &mut inv,
        N as u32,
        (4 * D) as u32,
    )?;
    let zero = vec![0.125_f32; N * 4 * D];
    let mut dx = DeviceBuffer::from_host(&stream, &zero)?;
    m.mhc_rms_backward_kernel(
        &stream,
        rows(),
        &x,
        &inv,
        &dy,
        &mut dx,
        N as u32,
        (4 * D) as u32,
    )?;
    let norm = normalized.to_host_vec(&stream)?;
    let invh = inv.to_host_vec(&stream)?;
    let grad = dx.to_host_vec(&stream)?;
    for r in 0..N {
        let begin = r * 4 * D;
        let end = (r + 1) * 4 * D;
        let square = streams[begin..end]
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            / (4 * D) as f64;
        let expected = (square + EPS).sqrt().recip();
        let dot = streams[begin..end]
            .iter()
            .zip(&upstream[begin..end])
            .map(|(x, g)| f64::from(*x) * f64::from(*g))
            .sum::<f64>()
            / (4 * D) as f64;
        assert!((f64::from(invh[r]) - expected).abs() < expected * 2e-6);
        for i in begin..end {
            assert!((f64::from(norm[i]) - f64::from(streams[i]) * expected).abs() < 3e-6);
            let derivative = expected
                * (f64::from(upstream[i]) - f64::from(streams[i]) * expected * expected * dot);
            assert!(
                (f64::from(grad[i]) - 0.125 - derivative).abs() < 3e-5 + derivative.abs() * 3e-6
            );
        }
    }
    // A learned head is not a fixed mean: both its dynamic map and direct stream
    // derivatives must be live, including all three surviving parameter groups.
    let raw = (0..N * STRIDE)
        .map(|i| ((i * 11 % 37) as f32 - 18.) * 0.03)
        .collect::<Vec<_>>();
    let base = [0.1_f32, -0.2, 0.3, -0.4];
    let alpha = [0.7_f32];
    let rd = DeviceBuffer::from_host(&stream, &raw)?;
    let bd = DeviceBuffer::from_host(&stream, &base)?;
    let ad = DeviceBuffer::from_host(&stream, &alpha)?;
    let head_upstream = (0..N * D)
        .map(|i| ((i * 3 % 19) as f32 - 9.) * 0.02)
        .collect::<Vec<_>>();
    let hd = DeviceBuffer::from_host(&stream, &head_upstream)?;
    let mut coeff = DeviceBuffer::zeroed(&stream, N * 24)?;
    let mut out = DeviceBuffer::zeroed(&stream, N * D)?;
    let mut dc = DeviceBuffer::zeroed(&stream, N * 24)?;
    let mut dr = DeviceBuffer::zeroed(&stream, N * STRIDE)?;
    let mut db = DeviceBuffer::zeroed(&stream, 4)?;
    let mut da = DeviceBuffer::zeroed(&stream, 1)?;
    m.mhc_coefficients_kernel(
        &stream,
        rows(),
        &rd,
        &bd,
        &ad,
        &mut coeff,
        N as u32,
        STRIDE as u32,
        true,
    )?;
    m.mhc_pre_kernel(
        &stream,
        grid(N * D),
        &x,
        &coeff,
        &mut out,
        N as u32,
        D as u32,
    )?;
    m.mhc_pre_backward_kernel(
        &stream,
        rows(),
        &x,
        &coeff,
        &hd,
        &mut dx,
        &mut dc,
        N as u32,
        D as u32,
        true,
    )?;
    m.mhc_coefficients_backward_kernel(
        &stream,
        rows(),
        &rd,
        &bd,
        &ad,
        &dc,
        &mut dr,
        &mut db,
        &mut da,
        N as u32,
        STRIDE as u32,
        true,
    )?;
    let co = coeff.to_host_vec(&stream)?;
    let ho = out.to_host_vec(&stream)?;
    let dg = dx.to_host_vec(&stream)?;
    let projected = dr.to_host_vec(&stream)?;
    let biases = db.to_host_vec(&stream)?;
    let scales = da.to_host_vec(&stream)?;
    let mut expected_bias = [0_f64; 4];
    let mut expected_scale = 0.;
    for r in 0..N {
        for j in 0..4 {
            let s =
                sigmoid(f64::from(raw[r * STRIDE + j]) * f64::from(alpha[0]) + f64::from(base[j]));
            assert!((f64::from(co[r * 24 + j]) - (s + EPS)).abs() < 1e-7);
            let gate_grad = (0..D)
                .map(|d| {
                    f64::from(head_upstream[r * D + d]) * f64::from(streams[(r * 4 + j) * D + d])
                })
                .sum::<f64>()
                * s
                * (1. - s);
            expected_bias[j] += gate_grad;
            expected_scale += gate_grad * f64::from(raw[r * STRIDE + j]);
            assert!(
                (f64::from(projected[r * STRIDE + j]) - gate_grad * f64::from(alpha[0])).abs()
                    < 1e-5
            );
            for d in 0..D {
                assert!(
                    (dg[(r * 4 + j) * D + d] - co[r * 24 + j] * head_upstream[r * D + d]).abs()
                        < 1e-7
                );
            }
        }
    }
    for r in 0..N {
        for d in 0..D {
            let expected = (0..4)
                .map(|j| co[r * 24 + j] * streams[(r * 4 + j) * D + d])
                .sum::<f32>();
            assert!((ho[r * D + d] - expected).abs() < 1e-6);
        }
    }
    for j in 0..4 {
        assert!((f64::from(biases[j]) - expected_bias[j]).abs() < 2e-5);
    }
    assert!((f64::from(scales[0]) - expected_scale).abs() < 2e-5);
    println!(
        "native four-stream expansion sum-adjoint, flattened RMS and learned HyperHead forward/backward verified; zero updates"
    );
    Ok(())
}

// Extreme-logit stability and both finite-round residuals were part of the
// predeclared contract. Compare the actual finite program, not a fixed point.
#[test]
#[ignore = "native finite Sinkhorn extreme-logit stability and all coefficient adjoints"]
fn finite_sinkhorn_extreme_logits_and_coefficient_adjoints() -> Result<(), Box<dyn Error>> {
    let (_, stream, m) = common::cuda_test_module(kernels::from_module)?;
    let magnitude = [10000_f32, 1000., 0., 80., 40., 1e-8, 500., 1.];
    let raw = (0..N * STRIDE)
        .map(|i| {
            let r = i / STRIDE;
            let c = i % STRIDE;
            let positive = if c < 8 {
                c % 2 == 0
            } else {
                let matrix_row = (c - 8) / 4;
                let matrix_col = (c - 8) % 4;
                matrix_col == if r % 3 == 0 { 0 } else { matrix_row % 3 }
            };
            if positive {
                magnitude[r]
            } else {
                -magnitude[r]
            }
        })
        .collect::<Vec<_>>();
    let base = (0..24).map(|i| (i as f32 - 11.) * 0.03).collect::<Vec<_>>();
    let alpha = vec![1., 0.75, 1.25];
    let host = [raw, base, alpha];
    let values: Inputs = std::array::from_fn(|i| {
        if i < 3 {
            host[i].iter().map(|&x| f64::from(x)).collect()
        } else {
            Vec::new()
        }
    });
    let buffers: Vec<_> = host
        .iter()
        .map(|v| DeviceBuffer::from_host(&stream, v))
        .collect::<Result<_, _>>()?;
    let upstream = (0..N * 24)
        .map(|i| ((i * 13 % 31) as f32 - 15.) * 0.03125)
        .collect::<Vec<_>>();
    let dc = DeviceBuffer::from_host(&stream, &upstream)?;
    let mut coefficients = DeviceBuffer::zeroed(&stream, N * 24)?;
    let mut dr = DeviceBuffer::zeroed(&stream, N * STRIDE)?;
    let mut db = DeviceBuffer::zeroed(&stream, 24)?;
    let mut da = DeviceBuffer::zeroed(&stream, 3)?;
    m.mhc_coefficients_kernel(
        &stream,
        rows(),
        &buffers[0],
        &buffers[1],
        &buffers[2],
        &mut coefficients,
        N as u32,
        STRIDE as u32,
        false,
    )?;
    m.mhc_coefficients_backward_kernel(
        &stream,
        rows(),
        &buffers[0],
        &buffers[1],
        &buffers[2],
        &dc,
        &mut dr,
        &mut db,
        &mut da,
        N as u32,
        STRIDE as u32,
        false,
    )?;
    let actual = coefficients.to_host_vec(&stream)?;
    let mut row_residual = 0_f64;
    let mut column_residual = 0_f64;
    for r in 0..N {
        let oracle = coefficient(&values, r);
        for i in 0..24 {
            assert!(
                actual[r * 24 + i].is_finite(),
                "nonfinite coefficient r={r} i={i}"
            );
            assert!(
                (f64::from(actual[r * 24 + i]) - oracle[i]).abs() < 5e-6,
                "finite Sinkhorn mismatch r={r} i={i}"
            );
        }
        for j in 0..4 {
            let row = (0..4)
                .map(|k| f64::from(actual[r * 24 + 8 + j * 4 + k]))
                .sum::<f64>();
            let col = (0..4)
                .map(|k| f64::from(actual[r * 24 + 8 + k * 4 + j]))
                .sum::<f64>();
            row_residual = row_residual.max((row - 1.).abs());
            column_residual = column_residual.max((col - 1.).abs());
        }
        assert!(actual[r * 24 + 8..(r + 1) * 24].iter().all(|x| *x > 0.));
    }
    let derivatives = [
        dr.to_host_vec(&stream)?,
        db.to_host_vec(&stream)?,
        da.to_host_vec(&stream)?,
    ];
    assert!(derivatives.iter().flatten().all(|x| x.is_finite()));
    let objective = |v: &Inputs| -> f64 {
        (0..N)
            .map(|r| {
                let c = coefficient(v, r);
                (0..24)
                    .map(|i| c[i] * f64::from(upstream[r * 24 + i]))
                    .sum::<f64>()
            })
            .sum()
    };
    for which in 0..3 {
        let indices = if which == 0 {
            (0..N)
                .flat_map(|r| (0..24).map(move |i| r * STRIDE + i))
                .collect::<Vec<_>>()
        } else {
            (0..host[which].len()).collect()
        };
        for index in indices {
            let mut shifted = values.clone();
            let delta = 1e-4;
            shifted[which][index] += delta;
            let positive = objective(&shifted);
            shifted[which][index] -= 2. * delta;
            let negative = objective(&shifted);
            let expected = (positive - negative) / (2. * delta);
            let got = f64::from(derivatives[which][index]);
            assert!(
                (got - expected).abs() < 2e-4 + expected.abs() * 0.004,
                "extreme coefficient derivative tensor={which} index={index}: {got} != {expected}"
            );
        }
    }
    for r in 0..N {
        assert!(
            derivatives[0][r * STRIDE + 24..(r + 1) * STRIDE]
                .iter()
                .all(|x| *x == 0.)
        );
    }
    println!(
        "native finite20 extreme Sinkhorn verified max_abs_logit=12500 row_sum_max_residual={row_residual} column_sum_max_residual={column_residual} positive=true adjoints=true updates=0"
    );
    Ok(())
}
