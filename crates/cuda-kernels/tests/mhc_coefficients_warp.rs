//! Declared before implementation: parallelize the same finite20-round mHC
//! program, preserving forward/raw-adjoint bits, all physical padding, real
//! bias/alpha accumulation, extreme logits, HyperHead, partial final blocks,
//! and independent two-step finite differences. No dense precision change.
use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::mhc::kernels as legacy;
use rust_kernels_cuda::mhc_coefficients::{BackwardArgs, ForwardArgs, MhcCoefficientsModule};
use std::error::Error;
mod common;
const STRIDE: usize = 128;
const GUARD: usize = 257;
fn raw(row: usize, i: usize) -> f32 {
    match row % 8 {
        0 => 0.0,
        1 => (((row * 7 + i * 11) % 17) as i32 - 8) as f32 / 131072.0,
        2 => {
            if i % 4 == 0 {
                10000.0
            } else {
                -10000.0
            }
        }
        3 => ((i % 4) as i32 - 2) as f32 * 5000.0,
        _ => (((row * 17 + i * 13) % 73) as i32 - 36) as f32 * 0.071,
    }
}
fn bias(i: usize) -> f32 {
    ((i * 11 % 17) as i32 - 8) as f32 * 0.043
}
fn upstream(row: usize, i: usize) -> f32 {
    (((row * 7 + i * 3) % 19) as i32 - 9) as f32 / 64.0
}
fn coefficients(raw: &[f64], bias: &[f64], alpha: &[f64], head: bool) -> [f64; 24] {
    let mut c = [0.0; 24];
    for i in 0..4 {
        c[i] = 1.0 / (1.0 + (-(raw[i] * alpha[0] + bias[i])).exp()) + 1e-6;
    }
    if head {
        return c;
    }
    for i in 4..8 {
        c[i] = 2.0 / (1.0 + (-(raw[i] * alpha[1] + bias[i])).exp());
    }
    for r in 0..4 {
        let values =
            std::array::from_fn::<_, 4, _>(|j| raw[8 + r * 4 + j] * alpha[2] + bias[8 + r * 4 + j]);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = values.iter().map(|x| (x - max).exp()).sum::<f64>();
        for j in 0..4 {
            c[8 + r * 4 + j] = (values[j] - max).exp() / sum + 1e-6;
        }
    }
    for stage in 1..40 {
        for group in 0..4 {
            let at = |j: usize| -> usize {
                if stage % 2 == 1 {
                    8 + j * 4 + group
                } else {
                    8 + group * 4 + j
                }
            };
            let denom = (0..4).map(|j| c[at(j)]).sum::<f64>() + 1e-6;
            for j in 0..4 {
                c[at(j)] /= denom;
            }
        }
    }
    c
}
#[test]
#[ignore = "native B8/S2048 mHC coefficient/adjoint parity, tails and independent FD"]
fn native_mhc_warp_coefficients_preserve_finite_program_and_all_gradients()
-> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let old = legacy::from_module(ptx.clone())?;
    let new = MhcCoefficientsModule::from_module(ptx)?;
    let alpha_host = [0.41f32, 0.73, 0.57];
    let bias_host = (0..24).map(bias).collect::<Vec<_>>();
    let alpha = DeviceBuffer::from_host(&stream, &alpha_host)?;
    let base = DeviceBuffer::from_host(&stream, &bias_host)?;
    let mut compared = 0usize;
    for n in [1usize, 7, 8, 9, 15, 16, 17, 33, 16384] {
        for head in [false, true] {
            let mut input = vec![f32::NAN; n * STRIDE];
            let mut dc = vec![0.0f32; n * 24];
            for r in 0..n {
                for i in 0..24 {
                    input[r * STRIDE + i] = raw(r, i);
                    dc[r * 24 + i] = upstream(r, i);
                }
            }
            let x = DeviceBuffer::from_host(&stream, &input)?;
            let dy = DeviceBuffer::from_host(&stream, &dc)?;
            let mut old_c = DeviceBuffer::from_host(&stream, &vec![0.375f32; n * 24 + GUARD])?;
            let mut new_c = DeviceBuffer::from_host(&stream, &vec![0.375f32; n * 24 + GUARD])?;
            let mut old_dx = DeviceBuffer::from_host(&stream, &vec![0.375f32; n * STRIDE + GUARD])?;
            let mut new_dx = DeviceBuffer::from_host(&stream, &vec![0.375f32; n * STRIDE + GUARD])?;
            let mut old_db = DeviceBuffer::from_host(&stream, &[0.125f32; 24 + GUARD])?;
            let mut new_db = DeviceBuffer::from_host(&stream, &[0.125f32; 24 + GUARD])?;
            let mut old_da = DeviceBuffer::from_host(&stream, &[0.25f32; 3 + GUARD])?;
            let mut new_da = DeviceBuffer::from_host(&stream, &[0.25f32; 3 + GUARD])?;
            let config = LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            old.mhc_coefficients_kernel(
                &stream,
                config,
                &x,
                &base,
                &alpha,
                &mut old_c,
                n as u32,
                STRIDE as u32,
                head,
            )?;
            old.mhc_coefficients_backward_kernel(
                &stream,
                config,
                &x,
                &base,
                &alpha,
                &dy,
                &mut old_dx,
                &mut old_db,
                &mut old_da,
                n as u32,
                STRIDE as u32,
                head,
            )?;
            new.forward(ForwardArgs {
                stream: &stream,
                raw: &x,
                bias: &base,
                alpha: &alpha,
                out: &mut new_c,
                rows: n as u32,
                stride: STRIDE as u32,
                head,
            })?;
            new.backward(BackwardArgs {
                stream: &stream,
                raw: &x,
                bias: &base,
                alpha: &alpha,
                dc: &dy,
                draw: &mut new_dx,
                dbias: &mut new_db,
                dalpha: &mut new_da,
                rows: n as u32,
                stride: STRIDE as u32,
                head,
            })?;
            let c0 = old_c.to_host_vec(&stream)?;
            let c1 = new_c.to_host_vec(&stream)?;
            let g0 = old_dx.to_host_vec(&stream)?;
            let g1 = new_dx.to_host_vec(&stream)?;
            for (i, (a, b)) in c0.iter().zip(&c1).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "forward n={n} head={head} i={i} old={a:e} new={b:e}"
                );
            }
            for (i, (a, b)) in g0.iter().zip(&g1).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "adjoint n={n} head={head} i={i} old={a:e} new={b:e}"
                );
            }
            let db0 = old_db.to_host_vec(&stream)?;
            let db1 = new_db.to_host_vec(&stream)?;
            let da0 = old_da.to_host_vec(&stream)?;
            let da1 = new_da.to_host_vec(&stream)?;
            let count = if head { 4 } else { 24 };
            let groups = if head { 1 } else { 3 };
            for i in 0..24 {
                let group = if i < 4 {
                    0
                } else if i < 8 {
                    1
                } else {
                    2
                };
                let abs_sum = (0..n)
                    .map(|r| (f64::from(g0[r * STRIDE + i]) / f64::from(alpha_host[group])).abs())
                    .sum::<f64>();
                let bound = 2e-5 * abs_sum + 2e-6;
                assert!(
                    (f64::from(db0[i]) - f64::from(db1[i])).abs() <= bound,
                    "bias n={n} i={i}"
                );
                if i >= count {
                    assert_eq!(db1[i], 0.125);
                }
            }
            for group in 0..3 {
                let range = if group == 0 {
                    0..4
                } else if group == 1 {
                    4..8
                } else {
                    8..24
                };
                let mut magnitude = 0.0;
                for r in 0..n {
                    for i in range.clone() {
                        magnitude += (f64::from(g0[r * STRIDE + i]) / f64::from(alpha_host[group])
                            * f64::from(input[r * STRIDE + i]))
                        .abs();
                    }
                }
                assert!(
                    (f64::from(da0[group]) - f64::from(da1[group])).abs()
                        <= 2e-5 * magnitude + 2e-5,
                    "alpha n={n} head={head} group={group} old={} new={} bound={}",
                    da0[group],
                    da1[group],
                    2e-5 * magnitude + 2e-5
                );
                if group >= groups {
                    assert_eq!(da1[group], 0.25);
                }
            }
            assert!(c1[n * 24..].iter().all(|&x| x == 0.375));
            assert!(g1[n * STRIDE..].iter().all(|&x| x == 0.375));
            assert!(db1[24..].iter().all(|&x| x == 0.125));
            assert!(da1[3..].iter().all(|&x| x == 0.25));
            compared += n * (24 + STRIDE);
        }
    }
    let raw0 = (0..STRIDE)
        .map(|i| if i < 24 { raw(5, i) } else { f32::NAN })
        .collect::<Vec<_>>();
    let dc0 = (0..24).map(|i| upstream(5, i)).collect::<Vec<_>>();
    let raw_dev = DeviceBuffer::from_host(&stream, &raw0)?;
    let dy = DeviceBuffer::from_host(&stream, &dc0)?;
    let mut dx = DeviceBuffer::zeroed(&stream, STRIDE + GUARD)?;
    let mut db = DeviceBuffer::zeroed(&stream, 24 + GUARD)?;
    let mut da = DeviceBuffer::zeroed(&stream, 3 + GUARD)?;
    let mut fd_max = 0.0f64;
    for head in [false, true] {
        let mut db = DeviceBuffer::zeroed(&stream, 24 + GUARD)?;
        let mut da = DeviceBuffer::zeroed(&stream, 3 + GUARD)?;
        new.backward(BackwardArgs {
            stream: &stream,
            raw: &raw_dev,
            bias: &base,
            alpha: &alpha,
            dc: &dy,
            draw: &mut dx,
            dbias: &mut db,
            dalpha: &mut da,
            rows: 1,
            stride: STRIDE as u32,
            head,
        })?;
        let grads = [
            dx.to_host_vec(&stream)?,
            db.to_host_vec(&stream)?,
            da.to_host_vec(&stream)?,
        ];
        let original = [
            raw0.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            bias_host.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            alpha_host.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        ];
        for which in 0..3 {
            for i in 0..if which == 2 { 3 } else { 24 } {
                for step in [1e-5, 5e-6] {
                    let mut x = original.clone();
                    x[which][i] += step;
                    let plus = coefficients(&x[0], &x[1], &x[2], head)
                        .iter()
                        .zip(&dc0)
                        .map(|(c, g)| c * f64::from(*g))
                        .sum::<f64>();
                    x[which][i] -= 2.0 * step;
                    let minus = coefficients(&x[0], &x[1], &x[2], head)
                        .iter()
                        .zip(&dc0)
                        .map(|(c, g)| c * f64::from(*g))
                        .sum::<f64>();
                    let expected = (plus - minus) / (2.0 * step);
                    let actual = f64::from(grads[which][i]);
                    let error = (actual - expected).abs();
                    fd_max = fd_max.max(error);
                    assert!(
                        error <= 3e-5 + 2e-3 * expected.abs(),
                        "FD head={head} which={which} i={i} actual={actual:e} expected={expected:e}"
                    );
                }
            }
        }
    }
    let mut out = DeviceBuffer::from_host(&stream, &[0.375f32; 24 + GUARD])?;
    for (rows, stride) in [(0, 128), (u32::MAX, 128), (1, 3), (1, u32::MAX)] {
        assert!(
            new.forward(ForwardArgs {
                stream: &stream,
                raw: &raw_dev,
                bias: &base,
                alpha: &alpha,
                out: &mut out,
                rows,
                stride,
                head: false
            })
            .is_err()
        );
        assert!(
            new.backward(BackwardArgs {
                stream: &stream,
                raw: &raw_dev,
                bias: &base,
                alpha: &alpha,
                dc: &dy,
                draw: &mut dx,
                dbias: &mut db,
                dalpha: &mut da,
                rows,
                stride,
                head: false
            })
            .is_err()
        );
    }
    assert!(out.to_host_vec(&stream)?.iter().all(|&x| x == 0.375));
    println!(
        "native_mhc_warp_coefficients_verified max_rows=16384 row_counts=9 head_modes=2 forward_and_raw_adjoint_bit_exact=true finite_rounds=20 FD_coordinates=102 FD_steps=2 compared_values={compared} FD_max={fd_max:e}"
    );
    Ok(())
}
