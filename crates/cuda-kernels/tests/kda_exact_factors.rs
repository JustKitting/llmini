//! Failure cases declared before the formed-L entry used by complete exact KDA:
//! all B8/S2048/H32 chunks, independent signed/dense triangular systems,
//! ignored diagonal/upper garbage, batch addressing, output guards and invalid
//! host shapes. This is an attention-operator oracle, not CPU model training.
use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::kda_exact_factors::{FormedInverseArgs, KdaExactFactorModule};

mod common;
#[allow(dead_code)]
#[path = "causal_attention_backward_tc/f16.rs"]
mod half_reference;

const D: usize = 64;
const CHUNKS: usize = 8 * 32 * (2048 / D);
const CASES: usize = 8;
const GUARD: usize = 257;

fn coefficient(case: usize, row: usize, col: usize) -> f32 {
    if col >= row {
        return if (row + col) % 2 == 0 {
            f32::NAN
        } else {
            f32::INFINITY
        };
    }
    let distance = row - col;
    match case {
        0 => 0.0,
        1 => {
            if distance == 1 {
                0.5
            } else {
                0.0
            }
        }
        2 => {
            if distance == 1 {
                -0.5
            } else {
                0.0
            }
        }
        3 => {
            if distance == 1 {
                if row % 2 == 0 { 0.25 } else { -0.25 }
            } else {
                0.0
            }
        }
        4 => {
            let beta = (row % 3 + 1) as f64 / 4.0;
            let kr = (row as i32 % 5 - 2) as f64 / 8.0;
            let kc = (col as i32 % 5 - 2) as f64 / 8.0;
            (beta * kr * kc * (-0.03 * distance as f64).exp()) as f32
        }
        5 => ((row * 17 + col * 13) as i32 % 9 - 4) as f32 / 1024.0,
        6 => {
            let sign = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
            sign * (-0.02 * distance as f64).exp() as f32
        }
        7 => 1.0,
        _ => unreachable!(),
    }
}

// Independent row-major forward substitution. The native implementation uses
// one GPU thread per transposed inverse column in shared memory.
fn reference(lower: &[f32]) -> Vec<f64> {
    let mut inverse = vec![0.0; D * D];
    for row in 0..D {
        for rhs in 0..D {
            let mut value = f64::from(row == rhs);
            for previous in 0..row {
                value -= f64::from(lower[row * D + previous]) * inverse[previous * D + rhs];
            }
            inverse[row * D + rhs] = value;
        }
    }
    inverse
}

fn half_ulp(value: f64) -> f64 {
    if value.abs() < 2.0_f64.powi(-14) {
        2.0_f64.powi(-24)
    } else {
        2.0_f64.powi(value.abs().log2().floor() as i32 - 10)
    }
}

#[test]
#[ignore = "native B8/S2048/H32 formed-L inverse; run with complete exact-factor PTX and sanitizers"]
fn native_kda_formed_inverse_all_chunks_and_guards() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(KdaExactFactorModule::from_module)?;
    let patterns = (0..CASES)
        .map(|case| {
            (0..D * D)
                .map(|i| coefficient(case, i / D, i % D))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let references = patterns
        .iter()
        .map(|lower| reference(lower))
        .collect::<Vec<_>>();
    let mut lower = Vec::with_capacity(CHUNKS * D * D);
    for chunk in 0..CHUNKS {
        lower.extend_from_slice(&patterns[chunk % CASES]);
        // A unique last-row/first-column perturbation detects bad batch strides.
        // For a unit-lower system it changes only inverse[63,0], by -delta.
        lower[chunk * D * D + (D - 1) * D] += (chunk + 1) as f32 / 131072.0;
    }
    let count = lower.len();
    let input = DeviceBuffer::from_host(&stream, &lower)?;
    let mut sentinel = vec![0x7e00u16; count + GUARD];
    sentinel[count..].fill(0x3555);
    let mut output = DeviceBuffer::from_host(&stream, &sentinel)?;
    module.formed_inverse(FormedInverseArgs {
        stream: &stream,
        lower: &input,
        out: &mut output,
        chunk_batch: CHUNKS as u32,
    })?;
    let actual = output.to_host_vec(&stream)?;
    assert_eq!(
        &actual[count..],
        &sentinel[count..],
        "inverse overwrote its output guard"
    );
    let mut max_error = 0.0f64;
    let mut max_residual = 0.0f64;
    for chunk in 0..CHUNKS {
        let case = chunk % CASES;
        let base = chunk * D * D;
        let delta = f64::from(lower[base + (D - 1) * D]) - f64::from(patterns[case][(D - 1) * D]);
        for row in 0..D {
            for col in 0..D {
                let bits = actual[base + row * D + col];
                let value = f64::from(half_reference::f16_bits_to_f32(bits));
                assert!(
                    value.is_finite(),
                    "unwritten/nonfinite chunk={chunk} row={row} col={col}"
                );
                if row == col {
                    assert_eq!(bits, 0x3c00);
                }
                if col > row {
                    assert_eq!(bits, 0);
                }
                let expected = references[case][row * D + col]
                    - if row == D - 1 && col == 0 { delta } else { 0.0 };
                let error = (value - expected).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 5e-7 + 0.56 * half_ulp(expected),
                    "formed inverse chunk={chunk} row={row} col={col} actual={value} expected={expected} error={error}"
                );
                if (case == 1 || case == 2) && row > col && !(row == D - 1 && col == 0) {
                    let distance = row - col;
                    if distance <= 24 {
                        let magnitude = if distance <= 14 {
                            ((15 - distance) as u16) << 10
                        } else {
                            1u16 << (24 - distance)
                        };
                        let sign = if case == 1 && distance % 2 == 1 {
                            0x8000
                        } else {
                            0
                        };
                        assert_eq!(bits, magnitude | sign, "closed-form bidiagonal inverse");
                    } else {
                        assert_eq!(bits & 0x7fff, 0);
                    }
                }
            }
        }
        // Independent product identity on both small and large batch signatures.
        if chunk < CASES || chunk >= CHUNKS - CASES {
            for row in 0..D {
                for col in 0..D {
                    let mut product = f64::from(half_reference::f16_bits_to_f32(
                        actual[base + row * D + col],
                    ));
                    for k in 0..row {
                        product += f64::from(lower[base + row * D + k])
                            * f64::from(half_reference::f16_bits_to_f32(
                                actual[base + k * D + col],
                            ));
                    }
                    let error = (product - f64::from(row == col)).abs();
                    max_residual = max_residual.max(error);
                    assert!(
                        error < 8e-4,
                        "inverse identity chunk={chunk} row={row} col={col} residual={error}"
                    );
                }
            }
        }
    }
    for chunk_batch in [0, u32::MAX] {
        assert!(
            module
                .formed_inverse(FormedInverseArgs {
                    stream: &stream,
                    lower: &input,
                    out: &mut output,
                    chunk_batch,
                })
                .is_err()
        );
    }
    let short = DeviceBuffer::from_host(&stream, &[0.0f32])?;
    assert!(
        module
            .formed_inverse(FormedInverseArgs {
                stream: &stream,
                lower: &short,
                out: &mut output,
                chunk_batch: 1,
            })
            .is_err()
    );
    assert_eq!(
        output.to_host_vec(&stream)?,
        actual,
        "rejected shapes launched or mutated output"
    );
    println!(
        "native_exact_KDA_formed_inverse_verified B=8 S=2048 H=32 chunk_batch={CHUNKS} pattern_cases={CASES} max_inverse_error={max_error:e} max_identity_residual={max_residual:e} output_guard=unchanged optimizer_updates=0"
    );
    Ok(())
}
