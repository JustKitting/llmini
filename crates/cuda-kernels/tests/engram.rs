//! Independent CUDA oracle declared before implementation of native Engram.
use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::engram::kernels;
use std::error::Error;
mod common;
fn grid(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn rows(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn oracle(
    h: &[f64],
    k: &[f64],
    v: &[f64],
    gh: &[f64],
    gk: &[f64],
    gu: &[f64],
    w: &[f64],
    n: usize,
    d: usize,
    s: usize,
) -> Vec<f64> {
    let mut u = vec![0.; n * d];
    let mut un = vec![0.; n * d];
    for r in 0..n {
        let b = r * d;
        let ih = (h[b..b + d].iter().map(|x| x * x).sum::<f64>() / d as f64 + 1e-5)
            .sqrt()
            .recip();
        let ik = (k[b..b + d].iter().map(|x| x * x).sum::<f64>() / d as f64 + 1e-5)
            .sqrt()
            .recip();
        let dot = (0..d)
            .map(|j| h[b + j] * ih * gh[j] * k[b + j] * ik * gk[j])
            .sum::<f64>()
            / (d as f64).sqrt();
        let signed_root = if dot == 0. {
            0.
        } else {
            dot.signum() * dot.abs().max(1e-6).sqrt()
        };
        let g = sigmoid(signed_root);
        for j in 0..d {
            u[b + j] = g * v[b + j];
        }
        let iu = (u[b..b + d].iter().map(|x| x * x).sum::<f64>() / d as f64 + 1e-5)
            .sqrt()
            .recip();
        for j in 0..d {
            un[b + j] = u[b + j] * iu * gu[j];
        }
    }
    let mut y = vec![0.; n * d];
    for r in 0..n {
        for j in 0..d {
            let z = (0..4)
                .filter_map(|c| {
                    let lag = 3 * (3 - c);
                    (r % s >= lag).then(|| w[j * 4 + c] * un[(r - lag) * d + j])
                })
                .sum::<f64>();
            y[r * d + j] = h[r * d + j] + u[r * d + j] + z * sigmoid(z);
        }
    }
    y
}
#[test]
#[ignore = "requires native sm_120a CUDA"]
fn full_engram_gate_convolution_reverse_and_causality() -> Result<(), Box<dyn Error>> {
    let (_, stream, m) = common::cuda_test_module(kernels::from_module)?;
    let (n, d, s) = (256usize, 128usize, 128usize);
    let l = n * d;
    let h = (0..l)
        .map(|i| ((i * 13 % 97) as f64 - 48.) * 0.019)
        .collect::<Vec<_>>();
    let k = (0..l)
        .map(|i| ((i * 29 % 89) as f64 - 44.) * 0.017)
        .collect::<Vec<_>>();
    let v = (0..l)
        .map(|i| ((i * 11 % 83) as f64 - 41.) * 0.021)
        .collect::<Vec<_>>();
    let gh = (0..d)
        .map(|i| 0.8 + (i % 7) as f64 * 0.04)
        .collect::<Vec<_>>();
    let gk = (0..d)
        .map(|i| 1.1 - (i % 9) as f64 * 0.02)
        .collect::<Vec<_>>();
    let gu = (0..d)
        .map(|i| 0.9 + (i % 5) as f64 * 0.03)
        .collect::<Vec<_>>();
    let w = (0..d * 4)
        .map(|i| ((i * 7 % 19) as f64 - 9.) * 0.023)
        .collect::<Vec<_>>();
    let e = (0..l)
        .map(|i| ((i * 3 % 23) as f64 - 11.) * 0.011)
        .collect::<Vec<_>>();
    let upload = |a: &[f64]| {
        DeviceBuffer::from_host(&stream, &a.iter().map(|x| *x as f32).collect::<Vec<_>>())
    };
    let hd = upload(&h)?;
    let kd = upload(&k)?;
    let vd = upload(&v)?;
    let ghd = upload(&gh)?;
    let gkd = upload(&gk)?;
    let gud = upload(&gu)?;
    let wd = upload(&w)?;
    let mut out = upload(&h)?;
    let mut stats = DeviceBuffer::zeroed(&stream, n * 4)?;
    let mut u = DeviceBuffer::zeroed(&stream, l)?;
    let mut un = DeviceBuffer::zeroed(&stream, l)?;
    let mut iu = DeviceBuffer::zeroed(&stream, n)?;
    let mut z = DeviceBuffer::zeroed(&stream, l)?;
    m.engram_gate_kernel(
        &stream,
        rows(n as u32),
        &hd,
        &kd,
        &vd,
        &ghd,
        &gkd,
        &mut stats,
        &mut u,
        n as u32,
        d as u32,
        1e-5,
    )?;
    m.engram_norm_kernel(
        &stream,
        rows(n as u32),
        &u,
        &gud,
        &mut un,
        &mut iu,
        n as u32,
        d as u32,
    )?;
    m.engram_conv_add_kernel(
        &stream,
        grid(l as u32),
        &u,
        &un,
        &wd,
        &mut z,
        &mut out,
        n as u32,
        d as u32,
        s as u32,
    )?;
    let expected = oracle(&h, &k, &v, &gh, &gk, &gu, &w, n, d, s);
    let actual = out.to_host_vec(&stream)?;
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        assert!((f64::from(a) - b).abs() < 2e-5, "forward {i}: {a} {b}");
    }
    let mut ed = upload(&e)?;
    let mut dun = DeviceBuffer::zeroed(&stream, l)?;
    let mut du = DeviceBuffer::zeroed(&stream, l)?;
    let mut dk = DeviceBuffer::zeroed(&stream, l)?;
    let mut dv = DeviceBuffer::zeroed(&stream, l)?;
    let mut dg = DeviceBuffer::zeroed(&stream, n)?;
    m.engram_conv_delta_kernel(&stream, grid(l as u32), &ed, &mut z, l as u32)?;
    m.engram_conv_input_backward_kernel(
        &stream,
        grid(l as u32),
        &z,
        &wd,
        &mut dun,
        n as u32,
        d as u32,
        s as u32,
    )?;
    m.engram_norm_backward_kernel(
        &stream,
        rows(n as u32),
        &u,
        &dun,
        &gud,
        &iu,
        &ed,
        &mut du,
        n as u32,
        d as u32,
    )?;
    m.engram_gate_backward_kernel(
        &stream,
        rows(n as u32),
        &hd,
        &kd,
        &vd,
        &ghd,
        &gkd,
        &stats,
        &du,
        &mut ed,
        &mut dk,
        &mut dv,
        &mut dg,
        n as u32,
        d as u32,
    )?;
    let mut dgh = DeviceBuffer::zeroed(&stream, d)?;
    let mut dgk = DeviceBuffer::zeroed(&stream, d)?;
    let mut dgu = DeviceBuffer::zeroed(&stream, d)?;
    let mut dw = DeviceBuffer::zeroed(&stream, d * 4)?;
    m.engram_parameter_backward_kernel(
        &stream,
        grid(d as u32 * 16),
        &hd,
        &kd,
        &ghd,
        &gkd,
        &stats,
        &dg,
        &u,
        &un,
        &iu,
        &dun,
        &z,
        &mut dgh,
        &mut dgk,
        &mut dgu,
        &mut dw,
        n as u32,
        d as u32,
        s as u32,
    )?;
    let grad_arrays = [
        ed.to_host_vec(&stream)?,
        dk.to_host_vec(&stream)?,
        dv.to_host_vec(&stream)?,
        dgh.to_host_vec(&stream)?,
        dgk.to_host_vec(&stream)?,
        dgu.to_host_vec(&stream)?,
        dw.to_host_vec(&stream)?,
    ];
    let originals = [
        h.clone(),
        k.clone(),
        v.clone(),
        gh.clone(),
        gk.clone(),
        gu.clone(),
        w.clone(),
    ];
    for which in 0..7 {
        for index in [0, 1, 17, 127, 128, originals[which].len() - 1] {
            if index >= originals[which].len() {
                continue;
            }
            let mut a = originals.clone();
            let eps = 1e-4;
            a[which][index] += eps;
            let yp = oracle(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], n, d, s);
            a[which][index] -= 2. * eps;
            let ym = oracle(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], n, d, s);
            let fd = yp
                .iter()
                .zip(&ym)
                .zip(&e)
                .map(|((p, m), e)| (p - m) * e / (2. * eps))
                .sum::<f64>();
            let got = f64::from(grad_arrays[which][index]);
            assert!(
                (got - fd).abs() < 2e-4 + fd.abs() * 2e-3,
                "gradient tensor={which} index={index} got={got} fd={fd}"
            );
        }
    }
    // Changing the entire second sequence must leave the first sequence exact.
    let mut changed = h.clone();
    for x in &mut changed[s * d..] {
        *x += 100.;
    }
    let hd2 = upload(&changed)?;
    let mut out2 = upload(&changed)?;
    m.engram_gate_kernel(
        &stream,
        rows(n as u32),
        &hd2,
        &kd,
        &vd,
        &ghd,
        &gkd,
        &mut stats,
        &mut u,
        n as u32,
        d as u32,
        1e-5,
    )?;
    m.engram_norm_kernel(
        &stream,
        rows(n as u32),
        &u,
        &gud,
        &mut un,
        &mut iu,
        n as u32,
        d as u32,
    )?;
    m.engram_conv_add_kernel(
        &stream,
        grid(l as u32),
        &u,
        &un,
        &wd,
        &mut z,
        &mut out2,
        n as u32,
        d as u32,
        s as u32,
    )?;
    assert_eq!(&out2.to_host_vec(&stream)?[..s * d], &actual[..s * d]);
    println!(
        "full native Engram forward, 42 gradient finite differences and sequence isolation passed"
    );
    Ok(())
}
#[test]
#[ignore = "requires native sm_120a CUDA"]
fn hash_gather_collision_scatter_and_sequence_boundaries() -> Result<(), Box<dyn Error>> {
    for dim in [32, 64] {
        hash_case(dim)?;
    }
    Ok(())
}
fn hash_case(dim: u32) -> Result<(), Box<dyn Error>> {
    let (_, stream, m) = common::cuda_test_module(kernels::from_module)?;
    let seq = 8u32;
    let heads = 8u32;
    let n = 16u32;
    let primes = [
        17u32, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
    ];
    let mut offsets = Vec::new();
    let mut total = 0;
    for &p in &primes {
        offsets.push(total);
        total += p;
    }
    let elems = (total * dim).div_ceil(128) * 128;
    let tokens = vec![1u32, 5, 6, 5, 6, 2, 7, 8, 1, 5, 6, 5, 6, 2, 7, 8];
    let compression = (0..32u32)
        .map(|x| if x == 6 { 5 } else { x })
        .collect::<Vec<_>>();
    let td = DeviceBuffer::from_host(&stream, &tokens)?;
    let cd = DeviceBuffer::from_host(&stream, &compression)?;
    let pd = DeviceBuffer::from_host(&stream, &primes)?;
    let od = DeviceBuffer::from_host(&stream, &offsets)?;
    let multipliers = [36313u64, 27191, 52711];
    let md = DeviceBuffer::from_host(&stream, &multipliers)?;
    let mut ids = DeviceBuffer::zeroed(&stream, (n * heads * 2) as usize)?;
    m.engram_hash_kernel(
        &stream,
        grid(n * heads * 2),
        &td,
        &cd,
        &pd,
        &od,
        &md,
        &mut ids,
        n,
        seq,
        heads,
        1,
        2,
        2,
    )?;
    let got = ids.to_host_vec(&stream)?;
    for r in 0..n as usize {
        for head in 0..16usize {
            let order = 2 + head / 8;
            let mut hist = [2u32; 3];
            let mut valid = true;
            for lag in 0..order {
                if lag > r % seq as usize || !valid {
                    break;
                }
                let raw = tokens[r - lag] as u32;
                if lag > 0 && (raw == 1 || raw == 2) {
                    break;
                }
                hist[lag] = compression[raw as usize];
                if raw == 1 || raw == 2 {
                    valid = false;
                }
            }
            let mix = (0..order)
                .map(|j| u64::from(hist[j]).wrapping_mul(multipliers[j]))
                .fold(0, |a, b| a ^ b);
            assert_eq!(
                got[r * 16 + head],
                offsets[head] + (mix % u64::from(primes[head])) as u32
            );
        }
    }
    let bytes = vec![0x22u8; (elems / 2) as usize];
    let raw_scales = vec![0x38u8; (elems / 16) as usize];
    let scales =
        rust_kernels_cuda::nvfp4::csf::encode_host(&raw_scales, (elems / 128) as usize, 128);
    let bd = DeviceBuffer::from_host(&stream, &bytes)?;
    let sd = DeviceBuffer::from_host(&stream, &scales)?;
    let gd = DeviceBuffer::from_host(&stream, &[0.5f32])?;
    let mut emb = DeviceBuffer::zeroed(&stream, (n * heads * 2 * dim) as usize)?;
    m.engram_gather_kernel(
        &stream,
        grid(n * heads * 2 * dim),
        &ids,
        &bd,
        &sd,
        &gd,
        &mut emb,
        n,
        heads * 2,
        dim,
    )?;
    assert!(emb.to_host_vec(&stream)?.iter().all(|&x| x == 0.5));
    let a = DeviceBuffer::from_host(&stream, &vec![1f32; (n * heads * 2 * dim) as usize])?;
    let b = DeviceBuffer::from_host(&stream, &vec![2f32; (n * heads * 2 * dim) as usize])?;
    let mut grad = DeviceBuffer::zeroed(&stream, elems as usize)?;
    m.engram_scatter_kernel(
        &stream,
        grid(n * heads * 2 * dim),
        &ids,
        &a,
        &b,
        &mut grad,
        n,
        heads * 2,
        dim,
    )?;
    let actual = grad.to_host_vec(&stream)?;
    let mut expected = vec![0f32; elems as usize];
    for id in got {
        for j in 0..dim as usize {
            expected[id as usize * dim as usize + j] += 3.;
        }
    }
    assert_eq!(actual, expected);
    Ok(())
}
