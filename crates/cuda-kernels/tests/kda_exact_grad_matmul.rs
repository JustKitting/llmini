//! Declared before implementation: the connected KDA reverse must retain FP16
//! tensor-core multiplicands. Local row scaling must preserve small upstreams
//! without a global FP32 state/operand tape or a changed optimizer loss scale.
//! Exercise all transpose/store/accumulation combinations at8192 native chunks,
//! dense64-feature coupling, independently rounded operands, subnormal through
//! large FP32 cotangents, poisoned masked cells, guards, and invalid shapes.
use cuda_core::DeviceBuffer;
use rust_kernels_cuda::kda_exact_grad_matmul::{GradMatmulArgs, KdaExactGradMatmulModule};
use std::error::Error;
mod common;
const BATCH: usize = 8192;
const AREA: usize = 4096;
const GUARD: usize = 257;
fn half(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    let unit = 2.0f64.powi((x.abs().log2().floor() as i32).max(-14) - 10);
    (x / unit).round_ties_even() * unit
}
fn half_bits(x: f32) -> u16 {
    if x == 0.0 {
        return if x.is_sign_negative() { 0x8000 } else { 0 };
    }
    let sign = if x < 0.0 { 0x8000 } else { 0 };
    let a = x.abs();
    let exp = a.log2().floor() as i32;
    if exp < -14 {
        return sign | (a * 16777216.0).round_ties_even() as u16;
    }
    sign | (((exp + 15) as u16) << 10)
        | ((a / 2.0f32.powi(exp) - 1.0) * 1024.0).round_ties_even() as u16
}
fn grad(batch: usize, r: usize, c: usize) -> f32 {
    if batch % 8 == 6 || r % 13 == 0 {
        return 0.0;
    }
    let e = [-140, -28, -12, 0, 40, 100, 0, -70][batch % 8] + ((r % 5) as i32 - 2) * 3;
    let signature = 1.0 + (batch / 8) as f64 / 2048.0;
    let x = (((r * 7 + c * 11 + batch % 8) % 17) as i32 - 8) as f64;
    (x * signature * 2.0f64.powi(e)) as f32
}
fn a(r: usize) -> f64 {
    (((r * 3) % 17) as i32 - 8) as f64 / 8.0
}
fn b(c: usize) -> f64 {
    (((c * 5) % 19) as i32 - 9) as f64 / 16.0
}
fn arbitrary(batch: usize) -> bool {
    batch < 8 || batch >= BATCH - 8
}
fn operand(batch: usize, r: usize, c: usize) -> f64 {
    if arbitrary(batch) {
        (((r * 7 + c * 11 + r * c + batch) % 17) as i32 - 8) as f64 / 16.0
    } else {
        a(r) * b(c) + if r == c { 0.125 } else { 0.0 }
    }
}
fn prior(batch: usize, r: usize, c: usize) -> f32 {
    (batch as f32 + 1.0) / 32768.0 + ((r * 3 + c * 7) % 11) as f32 / 128.0
}
fn host_scale(values: &[f64]) -> (f64, f64) {
    let max = values.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    if max == 0.0 {
        return (1.0, 1.0);
    }
    let shift = 4 - max.log2().floor() as i32;
    let first = shift.clamp(-120, 120);
    (2.0f64.powi(first), 2.0f64.powi(shift - first))
}
#[test]
#[ignore = "native full8192-chunk locally scaled FP16 tensor-core cotangent products"]
fn native_kda_exact_grad_matmul_scaled_half_operands_and_all_layouts() -> Result<(), Box<dyn Error>>
{
    let (_, stream, module) = common::cuda_test_module(KdaExactGradMatmulModule::from_module)?;
    let n = BATCH * AREA;
    let mut gradients = vec![0.0f32; n];
    let mut operands = vec![0u16; n];
    for batch in 0..BATCH {
        for r in 0..64 {
            for c in 0..64 {
                gradients[batch * AREA + r * 64 + c] = grad(batch, r, c);
                operands[batch * AREA + r * 64 + c] = half_bits(operand(batch, r, c) as f32);
            }
        }
    }
    let gd = DeviceBuffer::from_host(&stream, &gradients)?;
    let hd = DeviceBuffer::from_host(&stream, &operands)?;
    drop(gradients);
    drop(operands);
    let mut nonzero_tiny = 0usize;
    let mut max_normalized_error = 0.0f64;
    for bits in 0..8u32 {
        for accumulate in [false, true] {
            let tg = bits & 1 != 0;
            let th = bits & 2 != 0;
            let to = bits & 4 != 0;
            let mut before = vec![f32::from_bits(0x7fc01234); n + GUARD];
            for batch in 0..BATCH {
                for r in 0..64 {
                    for c in 0..64 {
                        before[batch * AREA + r * 64 + c] = prior(batch, r, c);
                    }
                }
            }
            let mut out = DeviceBuffer::from_host(&stream, &before)?;
            module.matmul(GradMatmulArgs {
                stream: &stream,
                grad: &gd,
                operand: &hd,
                out: &mut out,
                chunk_batch: BATCH as u32,
                transpose_grad: tg,
                transpose_operand: th,
                transpose_output: to,
                accumulate,
                alpha: -0.75,
                rows: 64,
                cols: 64,
                inner: 64,
            })?;
            let actual = out.to_host_vec(&stream)?;
            for batch in 0..BATCH {
                for r in 0..64 {
                    let g = (0..64)
                        .map(|k| {
                            f64::from(if tg {
                                grad(batch, k, r)
                            } else {
                                grad(batch, r, k)
                            })
                        })
                        .collect::<Vec<_>>();
                    let (s1, s2) = host_scale(&g);
                    let rounded = g.iter().map(|x| half((x * s1) * s2)).collect::<Vec<_>>();
                    let dot_a = (0..64).map(|k| rounded[k] * a(k)).sum::<f64>();
                    let dot_b = (0..64).map(|k| rounded[k] * b(k)).sum::<f64>();
                    let abs_a = (0..64).map(|k| (rounded[k] * a(k)).abs()).sum::<f64>();
                    let abs_b = (0..64).map(|k| (rounded[k] * b(k)).abs()).sum::<f64>();
                    for c in 0..64 {
                        let (sum, magnitude) = if arbitrary(batch) {
                            let terms = (0..64).map(|k| {
                                rounded[k]
                                    * if th {
                                        operand(batch, c, k)
                                    } else {
                                        operand(batch, k, c)
                                    }
                            });
                            (terms.clone().sum::<f64>(), terms.map(f64::abs).sum::<f64>())
                        } else if th {
                            (
                                a(c) * dot_b + 0.125 * rounded[c],
                                a(c).abs() * abs_b + 0.125 * rounded[c].abs(),
                            )
                        } else {
                            (
                                b(c) * dot_a + 0.125 * rounded[c],
                                b(c).abs() * abs_a + 0.125 * rounded[c].abs(),
                            )
                        };
                        let (or, oc) = if to { (c, r) } else { (r, c) };
                        let index = batch * AREA + or * 64 + oc;
                        let base = if accumulate {
                            f64::from(before[index])
                        } else {
                            0.0
                        };
                        let expected = -0.75 * (sum / s1) / s2 + base;
                        let bound = 2.0e-5 * ((magnitude / s1) / s2) * 0.75
                            + 3.0e-7 * base.abs()
                            + f64::from(f32::from_bits(1)) * 8.0;
                        let error = (f64::from(actual[index]) - expected).abs();
                        assert!(
                            actual[index].is_finite() && error <= bound,
                            "bits={bits} add={accumulate} batch={batch} r={r} c={c} actual={} expected={expected:e} error={error:e} bound={bound:e}",
                            actual[index]
                        );
                        if bound > 0.0 {
                            max_normalized_error = max_normalized_error.max(error / bound);
                        }
                        if !accumulate && batch % 8 == 0 && actual[index] != 0.0 {
                            nonzero_tiny += 1;
                        }
                    }
                }
            }
            assert!(actual[n..].iter().all(|x| x.to_bits() == 0x7fc01234));
        }
    }
    assert!(nonzero_tiny > 1000000, "tiny responses={nonzero_tiny}");
    // Poison all masked source cells. The orientations change physical
    // strides, so rebuild each layout rather than truncate a flat allocation.
    for bits in 0..8u32 {
        let (rows, cols, inner) = (17usize, 31usize, 15usize);
        let tg = bits & 1 != 0;
        let th = bits & 2 != 0;
        let to = bits & 4 != 0;
        let mut g = vec![f32::NAN; AREA];
        let mut h = vec![0x7e12u16; AREA];
        for r in 0..rows {
            for k in 0..inner {
                g[if tg { k * 64 + r } else { r * 64 + k }] = grad(1, r, k);
            }
        }
        for k in 0..inner {
            for c in 0..cols {
                h[if th { c * 64 + k } else { k * 64 + c }] = half_bits(operand(1, k, c) as f32);
            }
        }
        let gg = DeviceBuffer::from_host(&stream, &g)?;
        let hh = DeviceBuffer::from_host(&stream, &h)?;
        let mut oo = DeviceBuffer::from_host(&stream, &vec![0.375f32; AREA + GUARD])?;
        module.matmul(GradMatmulArgs {
            stream: &stream,
            grad: &gg,
            operand: &hh,
            out: &mut oo,
            chunk_batch: 1,
            transpose_grad: tg,
            transpose_operand: th,
            transpose_output: to,
            accumulate: false,
            alpha: 1.0,
            rows: rows as u32,
            cols: cols as u32,
            inner: inner as u32,
        })?;
        let v = oo.to_host_vec(&stream)?;
        for r in 0..64 {
            for c in 0..64 {
                let at = if to { c * 64 + r } else { r * 64 + c };
                if r >= rows || c >= cols {
                    assert_eq!(v[at].to_bits(), 0);
                    continue;
                }
                let row = (0..inner)
                    .map(|k| f64::from(grad(1, r, k)))
                    .collect::<Vec<_>>();
                let (s1, s2) = host_scale(&row);
                let terms = (0..inner).map(|k| half((row[k] * s1) * s2) * operand(1, k, c));
                let expected = terms.clone().sum::<f64>() / s1 / s2;
                let bound = 2e-5 * terms.map(f64::abs).sum::<f64>() / s1 / s2 + 1e-30;
                assert!((f64::from(v[at]) - expected).abs() <= bound);
            }
        }
        assert!(v[AREA..].iter().all(|&x| x == 0.375));
    }
    let one = DeviceBuffer::from_host(&stream, &[1.0f32])?;
    let oneh = DeviceBuffer::from_host(&stream, &[0x3c00u16])?;
    let mut guard = DeviceBuffer::from_host(&stream, &[0.375f32; AREA + GUARD])?;
    for (count, rows, cols, inner, alpha) in [
        (0, 64, 64, 64, 1.0),
        (u32::MAX, 64, 64, 64, 1.0),
        (1, 0, 64, 64, 1.0),
        (1, 65, 64, 64, 1.0),
        (1, 64, 0, 64, 1.0),
        (1, 64, 64, 65, 1.0),
        (1, 64, 64, 64, f32::NAN),
        (1, 64, 64, 64, 1.0),
    ] {
        assert!(
            module
                .matmul(GradMatmulArgs {
                    stream: &stream,
                    grad: &one,
                    operand: &oneh,
                    out: &mut guard,
                    chunk_batch: count,
                    transpose_grad: false,
                    transpose_operand: false,
                    transpose_output: false,
                    accumulate: false,
                    alpha,
                    rows,
                    cols,
                    inner
                })
                .is_err()
        );
    }
    assert!(guard.to_host_vec(&stream)?.iter().all(|&x| x == 0.375));
    // Nonfinite upstreams must remain observable by the normal global guard.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut g = vec![0.0; AREA];
        g[0] = bad;
        let gg = DeviceBuffer::from_host(&stream, &g)?;
        let hh = DeviceBuffer::from_host(&stream, &vec![0x3c00u16; AREA])?;
        module.matmul(GradMatmulArgs {
            stream: &stream,
            grad: &gg,
            operand: &hh,
            out: &mut guard,
            chunk_batch: 1,
            transpose_grad: false,
            transpose_operand: false,
            transpose_output: false,
            accumulate: false,
            alpha: 1.0,
            rows: 64,
            cols: 64,
            inner: 64,
        })?;
        assert!(
            guard.to_host_vec(&stream)?[..64]
                .iter()
                .any(|x| !x.is_finite())
        );
    }
    println!(
        "native_exact_KDA_scaled_grad_matmul_verified chunk_batch=8192 layouts=8 accumulation_modes=2 FP16_multiplicands=true FP32_state_tape=false tiny_nonzero={nonzero_tiny} max_fraction_of_error_bound={max_normalized_error:e}"
    );
    Ok(())
}
