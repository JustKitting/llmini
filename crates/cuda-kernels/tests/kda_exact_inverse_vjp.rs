//! Declared before inverse-control VJP implementation: both implicit solves use
//! original FP32 L, not the rounded FP16 inverse. All8192chunks, eight signed/
//! physical systems, unique batch scales, sub-FP16 upstreams, ignored constant
//! inverse upper/diagonal adjoints, independent dense inverse-product oracle,
//! full writes, guards and rejected shapes. No model/optimizer updates.
use cuda_core::DeviceBuffer;
use rust_kernels_cuda::kda_exact_inverse_vjp::{KdaExactInverseVjpModule, VjpArgs};
use std::error::Error;
mod common;
const D: usize = 64;
const CHUNKS: usize = 8192;
const GUARD: usize = 257;
fn l(case: usize, r: usize, c: usize) -> f32 {
    if c >= r {
        return f32::NAN;
    }
    match case {
        0 => 0.0,
        1 => {
            if r == c + 1 {
                0.5
            } else {
                0.0
            }
        }
        2 => {
            if r == c + 1 {
                -0.5
            } else {
                0.0
            }
        }
        3 => ((r * 17 + c * 13) % 9) as f32 / 512.0 - 1.0 / 128.0,
        4 => {
            ((r % 5) as f32 - 2.0) * ((c % 5) as f32 - 2.0) / 64.0
                * (-0.03 * (r - c) as f64).exp() as f32
        }
        5 => {
            if (r + c) % 2 == 0 {
                0.25
            } else {
                -0.25
            }
        }
        6 => {
            if (r + c) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
        }
        7 => 1.0,
        _ => unreachable!(),
    }
}
fn dy(case: usize, r: usize, c: usize) -> f32 {
    if r <= c {
        return f32::INFINITY;
    }
    let exponent = if case % 2 == 0 { -28 } else { -8 };
    (((r * 11 + c * 7 + case) % 9) as i32 - 4) as f32 * 2.0f32.powi(exponent)
}
fn reference(case: usize) -> Vec<f64> {
    let mut inverse = vec![0.0; D * D];
    for row in 0..D {
        for col in 0..=row {
            let mut v = f64::from(row == col);
            for k in 0..row {
                v -= f64::from(l(case, row, k)) * inverse[k * D + col];
            }
            inverse[row * D + col] = v;
        }
    }
    // Independent ordinary dense products -I^T dI I^T. The GPU implements two
    // direct triangular solves and never multiplies by a rounded inverse.
    let mut temp = vec![0.0; D * D];
    let mut result = vec![0.0; D * D];
    for r in 0..D {
        for c in 0..D {
            for u in c + 1..D {
                temp[r * D + c] += inverse[u * D + r] * f64::from(dy(case, u, c));
            }
        }
    }
    for r in 0..D {
        for c in 0..r {
            for v in 0..=c {
                result[r * D + c] -= temp[r * D + v] * inverse[c * D + v];
            }
        }
    }
    result
}
#[test]
#[ignore = "native full8192 FP32 triangular-control inverse VJP and sanitizers"]
fn native_kda_exact_inverse_vjp_original_control_and_tiny_gradients() -> Result<(), Box<dyn Error>>
{
    let (_, stream, module) = common::cuda_test_module(KdaExactInverseVjpModule::from_module)?;
    let mut lower = Vec::with_capacity(CHUNKS * D * D);
    let mut gradient = Vec::with_capacity(CHUNKS * D * D);
    for chunk in 0..CHUNKS {
        let scale = (chunk + 1) as f32 / 8192.0;
        for r in 0..D {
            for c in 0..D {
                lower.push(l(chunk % 8, r, c));
                gradient.push(dy(chunk % 8, r, c) * scale);
            }
        }
    }
    let ld = DeviceBuffer::from_host(&stream, &lower)?;
    let gd = DeviceBuffer::from_host(&stream, &gradient)?;
    let sentinel = vec![f32::from_bits(0x7fc01234); lower.len() + GUARD];
    let mut out = DeviceBuffer::from_host(&stream, &sentinel)?;
    module.vjp(VjpArgs {
        stream: &stream,
        lower: &ld,
        inverse_grad: &gd,
        out: &mut out,
        chunk_batch: CHUNKS as u32,
    })?;
    let actual = out.to_host_vec(&stream)?;
    let refs = (0..8).map(reference).collect::<Vec<_>>();
    let mut max_error = 0.0f64;
    let mut tiny_nonzero = 0usize;
    for chunk in 0..CHUNKS {
        let scale = (chunk + 1) as f64 / 8192.0;
        for i in 0..D * D {
            let expected = refs[chunk % 8][i] * scale;
            let value = f64::from(actual[chunk * D * D + i]);
            let error = (value - expected).abs();
            max_error = max_error.max(error);
            let abs_tolerance = if chunk % 2 == 0 { 2e-13 } else { 2e-7 };
            assert!(
                value.is_finite() && error <= abs_tolerance + 3e-5 * expected.abs(),
                "inverse VJP chunk={chunk} row={} col={} actual={value} expected={expected} error={error}",
                i / D,
                i % D
            );
            if i % D >= i / D {
                assert_eq!(actual[chunk * D * D + i].to_bits(), 0);
            }
            if chunk % 2 == 0 && expected.abs() > 1e-12 {
                assert_ne!(value, 0.0, "lost sub-FP16 control adjoint");
                tiny_nonzero += 1;
            }
        }
    }
    assert!(tiny_nonzero > 1_000_000);
    assert!(
        actual[lower.len()..]
            .iter()
            .all(|x| x.to_bits() == 0x7fc01234)
    );
    for chunk_batch in [0, u32::MAX] {
        assert!(
            module
                .vjp(VjpArgs {
                    stream: &stream,
                    lower: &ld,
                    inverse_grad: &gd,
                    out: &mut out,
                    chunk_batch
                })
                .is_err()
        );
    }
    let short = DeviceBuffer::from_host(&stream, &[0.0f32])?;
    assert!(
        module
            .vjp(VjpArgs {
                stream: &stream,
                lower: &short,
                inverse_grad: &gd,
                out: &mut out,
                chunk_batch: 1
            })
            .is_err()
    );
    assert!(
        module
            .vjp(VjpArgs {
                stream: &stream,
                lower: &ld,
                inverse_grad: &short,
                out: &mut out,
                chunk_batch: 1
            })
            .is_err()
    );
    assert!(
        out.to_host_vec(&stream)?
            .iter()
            .zip(&actual)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    println!(
        "native_exact_KDA_inverse_VJP_verified chunk_batch={CHUNKS} pattern_cases=8 original_FP32_control=true tiny_nonzero={tiny_nonzero} max_error={max_error:e} guards=unchanged optimizer_updates=0"
    );
    Ok(())
}
