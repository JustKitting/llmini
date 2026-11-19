use super::*;
use rust_kernels_cuda::nvfp4_quant::nvfp4_tensor_amax_chunks;
use rust_kernels_cuda::nvfp4_tma_matmul::scale_layout::{
    sm120_scale_byte_offset, sm120_scale_packed_len, sm120_scale_padded_mn_extent,
};

fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|i| (0..k).map(|j| a[i / n * k + j] * b[j * n + i % n]).sum())
        .collect()
}
fn ideal(mut x: Vec<f64>, m: usize, n: usize, factored: bool) -> Vec<f64> {
    for iter in 0..POLAR_ITERATIONS {
        let xt: Vec<f64> = (0..n * m).map(|i| x[i % m * n + i / m]).collect();
        let mut g = matmul(&x, &xt, m, n, m);
        let bound = g.iter().copied().map(f64::abs).fold(1.0_f64, f64::max);
        x.iter_mut().for_each(|v| *v /= bound.sqrt());
        g.iter_mut().for_each(|v| *v /= bound);
        let c = muon_polar_coefficients(iter);
        if factored {
            let gg = matmul(&g, &g, m, m, m);
            let b: Vec<f64> = g
                .iter()
                .zip(gg)
                .map(|(g, gg)| c.b as f64 * g + c.c as f64 * gg)
                .collect();
            let bx = matmul(&b, &x, m, m, n);
            x.iter_mut()
                .zip(bx)
                .for_each(|(x, bx)| *x = c.a as f64 * *x + bx);
        } else {
            let gx = matmul(&g, &x, m, m, n);
            let ggx = matmul(&g, &gx, m, m, n);
            for i in 0..x.len() {
                x[i] = c.a as f64 * x[i] + c.b as f64 * gx[i] + c.c as f64 * ggx[i];
            }
        }
    }
    let bound = x
        .chunks_exact(n)
        .map(|r| r.iter().map(|v| v * v).sum::<f64>())
        .fold(1.0_f64, f64::max);
    x.iter_mut().for_each(|v| *v /= bound.sqrt());
    x
}
#[test]
fn gram_square_matches_direct_f64_quintic_polynomial() {
    let x: Vec<f64> = (0..16 * 64)
        .map(|i| (((i * 17 + 3) % 61) as f64 - 30.0) / 1024.0)
        .collect();
    let a = ideal(x.clone(), 16, 64, false);
    let b = ideal(x, 16, 64, true);
    let error = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
        / a.iter().map(|v| v * v).sum::<f64>();
    assert!(error.sqrt() < 1e-11, "{error}");
}
fn operand(stream: &CudaStream, m: usize, k: usize) -> MuonTmaOperandScratch {
    MuonTmaOperandScratch {
        bytes: DeviceBuffer::zeroed(stream, m * k / 2).unwrap(),
        scales: DeviceBuffer::zeroed(stream, m * k / 16).unwrap(),
        scale_packed: DeviceBuffer::zeroed(
            stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(m), k),
        )
        .unwrap(),
        global_scale: DeviceBuffer::zeroed(stream, 1).unwrap(),
        amax: DeviceBuffer::zeroed(stream, 1).unwrap(),
        chunk_amax: DeviceBuffer::zeroed(stream, nvfp4_tensor_amax_chunks(m * k)).unwrap(),
    }
}
fn scratch(stream: &CudaStream, input: &[f32], m: u32, n: u32) -> MuonScratchBuffers {
    let d = TmaDims::new(m, n, m);
    let k = ceil_to(n, TILE_K) as usize;
    let mm = d.m as usize;
    let nn = d.n as usize;
    let z = |len| DeviceBuffer::zeroed(stream, len).unwrap();
    MuonScratchBuffers {
        oriented: z(1),
        polar_next: z(input.len()),
        polar_x: DeviceBuffer::from_host(stream, input).unwrap(),
        polar_gram: z((m * m) as usize),
        polar_ax: z(input.len().max((m * m) as usize)),
        polar_chunks: z(1),
        normuon_factors: z(1),
        normuon_chunks: z(1),
        hyperball_chunks: z(1),
        tma: MuonTmaScratch {
            out_padded: z((mm * nn).max(mm * mm)),
            a: operand(stream, mm, k.max(d.k as usize)),
            b: operand(stream, nn.max(mm), d.k as usize),
            bound_amax: z(1),
            descriptors: TmaNvfp4DeviceScaleDescriptors::new(stream).unwrap(),
        },
    }
}
fn decode(stream: &CudaStream, p: &MuonTmaOperandScratch, m: usize, k: usize) -> (Vec<f64>, f32) {
    let bytes = p.bytes.to_host_vec(stream).unwrap();
    let scales = p.scale_packed.to_host_vec(stream).unwrap();
    let e2m1 = [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let data = (0..m * k)
        .map(|i| {
            let q = (bytes[i / 2] >> ((i % 2) * 4)) & 15;
            let s = scales
                [sm120_scale_byte_offset(i / k, (i % k) / 16, sm120_scale_padded_mn_extent(m), k)];
            let exp = (s >> 3) & 15;
            let mant = (s & 7) as f64;
            let scale = if exp == 0 {
                mant * 2.0_f64.powi(-9)
            } else {
                (1.0 + mant / 8.0) * 2.0_f64.powi(exp as i32 - 7)
            };
            e2m1[(q & 7) as usize] * if q & 8 != 0 { -scale } else { scale }
        })
        .collect();
    (data, p.global_scale.to_host_vec(stream).unwrap()[0])
}
#[path = "../../../crates/cuda-kernels/tests/optimizer/polar_reference/rounding.rs"]
mod rounding;

#[test]
#[ignore = "requires CUDA; original numerical counterexamples and decoded native-FP4 round oracle"]
fn gram_square_original_counterexamples_and_decoded_rounds() {
    let runtime = Runtime::new().unwrap();
    let stream = runtime.stream.as_ref();
    for (m, n) in [
        (16_usize, 64_usize),
        (2048, 8192),
        (2048, 6176),
        (128, 160),
        (64, 128),
        (32, 128),
    ] {
        let mut input = vec![0.0_f32; m * n];
        if m == 16 {
            for i in 0..input.len() {
                let u = ((i / n * 17 + 3) % 29) as f32 - 14.0;
                let v = ((i % n * 11 + 7) % 31) as f32 - 15.0;
                input[i] = u * v;
            }
            let norm = input
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                * 1.01
                + 1e-7;
            input
                .iter_mut()
                .for_each(|v| *v = rounding::round_f16_to_f32(*v / norm as f32));
        } else if m != 32 {
            for i in 0..m - 1 {
                input[i * n + i] = if m == 64 {
                    i as f32 / 63.0
                } else if i == 0 {
                    0.98
                } else {
                    0.003
                };
            }
        }
        let expected = if m <= 64 {
            ideal(input.iter().map(|v| *v as f64).collect(), m, n, false)
        } else {
            let mut s: Vec<f64> = (0..m).map(|i| input[i * n + i] as f64).collect();
            for iter in 0..POLAR_ITERATIONS {
                let bound = s.iter().map(|v| v * v).fold(1.0_f64, f64::max);
                let c = muon_polar_coefficients(iter);
                for v in &mut s {
                    *v /= bound.sqrt();
                    let g = *v * *v;
                    *v *= c.a as f64 + c.b as f64 * g + c.c as f64 * g * g;
                }
            }
            let bound = s.iter().map(|v| v * v).fold(1.0_f64, f64::max);
            let mut x = vec![0.0_f64; m * n];
            for i in 0..m {
                x[i * n + i] = s[i] / bound.sqrt();
            }
            x
        };
        let mut s = scratch(stream, &input, m as u32, n as u32);
        let initial_amax = input.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        s.tma.a.amax = DeviceBuffer::from_host(stream, &[initial_amax]).unwrap();
        let desc = MuonSlotDescriptor {
            grad: 0,
            momentum: 0,
            variance: 0,
            second_momentum: 0,
            z_master: 0,
            x_master: 0,
            schedule_amax: 0,
            bytes: 0,
            scales: 0,
            global_scale: 0,
            rows: m as u32,
            cols: n as u32,
            learning_rate_multiplier: 1.0,
            qk_clip_factor_offset: 0,
            qk_clip_head_dim: 0,
        };
        let defer = can_defer_polar_bounds(m as u32, n as u32);
        let dims = TmaDims::new(m as u32, n as u32, m as u32);
        for iter in 0..POLAR_ITERATIONS {
            let (source, target) = if iter % 2 == 0 {
                (&mut s.polar_x, &mut s.polar_next)
            } else {
                (&mut s.polar_next, &mut s.polar_x)
            };
            run_tma_polar_iteration(
                stream,
                &runtime,
                source,
                target,
                &mut s.polar_gram,
                &mut s.polar_ax,
                tma_refs(&mut s.tma),
                m as u32,
                n as u32,
                iter,
                0,
                TmaTraceConfig {
                    enabled: false,
                    slot: None,
                },
                desc,
                defer,
            )
            .unwrap();
            let source = source.to_host_vec(stream).unwrap();
            let out = target.to_host_vec(stream).unwrap();
            let gram = s.polar_gram.to_host_vec(stream).unwrap();
            if iter + 1 < POLAR_ITERATIONS {
                let expected_amax = out.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
                assert_eq!(
                    s.tma.b.amax.to_host_vec(stream).unwrap()[0],
                    expected_amax,
                    "stale next-X maximum {m}x{n} round{iter}"
                );
            }
            if defer {
                let factor = s.polar_ax.to_host_vec(stream).unwrap();
                let expected = factor[..m * m]
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0_f32, f32::max);
                assert_eq!(
                    s.tma.a.amax.to_host_vec(stream).unwrap()[0],
                    expected,
                    "stale factor maximum {m}x{n} round{iter}"
                );
            }
            let bound = if defer {
                gram.iter().copied().map(f32::abs).fold(1.0_f32, f32::max)
            } else {
                1.0
            };
            let inv = 1.0 / bound.sqrt();
            let (a, ag) = decode(stream, &s.tma.a, dims.m as usize, dims.k as usize);
            if iter + 1 == POLAR_ITERATIONS && !defer {
                // The final materialized bound replaces A with packed next-X.
                // Its observable contract is the freshly recomputed Gram and
                // bound, rather than the now-consumed polynomial factor.
                let k = ceil_to(n as u32, TILE_K) as usize;
                let (next, global) = decode(stream, &s.tma.a, dims.m as usize, k);
                let actual_bound = s.tma.bound_amax.to_host_vec(stream).unwrap()[0];
                let mut expected_bound = 0.0_f32;
                for row in 0..m {
                    for col in 0..m {
                        let dot = (0..k)
                            .map(|j| next[row * k + j] * next[col * k + j])
                            .sum::<f64>();
                        let want = (dot as f32) * (global * global);
                        expected_bound = expected_bound.max(want.abs());
                        let normalized = want / actual_bound.max(1.0);
                        assert!(
                            (gram[row * m + col] - normalized).abs()
                                <= 2e-5 * normalized.abs().max(1.0),
                            "final Gram {m}x{n} ({row},{col}): {} vs {normalized}",
                            gram[row * m + col]
                        );
                    }
                }
                assert!((actual_bound - expected_bound).abs() <= 2e-5 * expected_bound.max(1.0));
                assert!(out.iter().all(|v| v.is_finite()));
                println!(
                    "gram_square_final_bound rows={m} cols={n} actual={actual_bound:.9e} decoded={expected_bound:.9e}"
                );
                std::mem::swap(&mut s.tma.a.amax, &mut s.tma.b.amax);
                continue;
            }
            let (b, bg) = decode(stream, &s.tma.b, dims.n as usize, dims.k as usize);
            let src_amax = source.iter().copied().map(f32::abs).fold(0.0_f32, f32::max) * inv;
            let expected_global = if src_amax == 0.0 {
                1.0
            } else {
                src_amax / 1536.0
            };
            assert!(
                (bg - expected_global).abs() <= expected_global.abs() * 3e-6,
                "X transpose scale changed: {bg} vs {expected_global}"
            );
            let final_scale = if iter + 1 == POLAR_ITERATIONS && !defer {
                1.0 / s.tma.bound_amax.to_host_vec(stream).unwrap()[0]
                    .max(1.0)
                    .sqrt()
            } else {
                1.0
            };
            let c = muon_polar_coefficients(iter);
            let samples: Vec<usize> = if m <= 64 {
                (0..m * n).collect()
            } else {
                (0..m)
                    .map(|i| i * n + i)
                    .chain((0..1024).map(|i| (i * 1009 + 13) % (m * n)))
                    .collect()
            };
            let mut max_error = 0.0_f64;
            for i in samples {
                let r = i / n;
                let col = i % n;
                let dot = (0..dims.k as usize)
                    .map(|k| a[r * dims.k as usize + k] * b[col * dims.k as usize + k])
                    .sum::<f64>();
                let product = (dot as f32) * (ag * bg);
                let want = c.a.mul_add(source[i] * inv, product) * final_scale;
                max_error = max_error.max((out[i] - want).abs() as f64);
                assert!(
                    (out[i] - want).abs() <= 2e-5 * want.abs().max(1.0),
                    "decoded {m}x{n} iter{iter} index{i}: {} vs {want}",
                    out[i]
                );
            }
            assert!(out.iter().all(|v| v.is_finite()));
            println!("gram_square_decoded rows={m} cols={n} iter={iter} max_abs={max_error:.9e}");
            std::mem::swap(&mut s.tma.a.amax, &mut s.tma.b.amax);
        }
        let out = s.polar_next.to_host_vec(stream).unwrap();
        let raw_bound = if defer {
            s.tma.a.amax.to_host_vec(stream).unwrap()[0]
        } else {
            s.tma.bound_amax.to_host_vec(stream).unwrap()[0]
        };
        if defer {
            let expected = out
                .chunks_exact(n)
                .map(|r| r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>())
                .fold(0.0_f64, f64::max);
            assert!(
                (raw_bound as f64 - expected).abs() <= 2e-6 * expected.max(1.0),
                "final norm scalar ownership {m}x{n}: {raw_bound} vs {expected}"
            );
        }
        let bound = raw_bound.max(1.0);
        let inv = if defer { 1.0 / bound.sqrt() } else { 1.0 };
        let mut err = 0.0_f64;
        let mut norm = 0.0_f64;
        for i in 0..out.len() {
            err += (out[i] as f64 * inv as f64 - expected[i]).powi(2);
            norm += expected[i].powi(2);
            if m != 16 && i / n != i % n {
                assert_eq!(out[i], 0.0, "rank-deficient off-diagonal {i}");
            }
        }
        if m == 32 {
            assert!(out.iter().all(|v| *v == 0.0));
        }
        // Native FP4 association changes numerical error. Report distance to the
        // ideal polynomial; the declared 200/450 training gates judge quality.
        println!(
            "gram_square_ideal rows={m} cols={n} rel_l2={:.9e}",
            (err / norm.max(1e-30)).sqrt()
        );
    }
}
