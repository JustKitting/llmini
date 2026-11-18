//! Declared before boundary reverse: all8192chunks/eight gate/beta patterns,
//! ordinary/sub-FP16 cotangents, accumulation into pair gradients, direct
//! interval-support oracle, two-step STE FD, multi-sequence65/127tails,
//! exact gradient-prefix support, poisoned padding, guards and shape rejection.
use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::kda_exact_boundary_vjp::{BoundaryVjpArgs, KdaExactBoundaryVjpModule};
use std::error::Error;
mod common;
const D: usize = 64;
const N: usize = 4096;
const CHUNKS: usize = 8192;
const GUARD: usize = 257;
#[derive(Clone)]
struct Fixture {
    q: Vec<f64>,
    k: Vec<f64>,
    g: Vec<f64>,
    b: Vec<f64>,
    uq: Vec<f64>,
    uk: Vec<f64>,
    ub: Vec<f64>,
    ue: Vec<f64>,
    dq: Vec<f64>,
    dk: Vec<f64>,
    dg: Vec<f64>,
    db: Vec<f64>,
}
impl Fixture {
    fn empty() -> Self {
        Self {
            q: Vec::new(),
            k: Vec::new(),
            g: Vec::new(),
            b: Vec::new(),
            uq: Vec::new(),
            uk: Vec::new(),
            ub: Vec::new(),
            ue: Vec::new(),
            dq: Vec::new(),
            dk: Vec::new(),
            dg: Vec::new(),
            db: Vec::new(),
        }
    }
    fn new(case: usize, features: usize, tiny: bool) -> Self {
        let mut f = Self {
            q: vec![0.0; N],
            k: vec![0.0; N],
            g: vec![0.0; N],
            b: vec![0.0; D],
            uq: vec![0.0; N],
            uk: vec![0.0; N],
            ub: vec![0.0; N],
            ue: vec![0.0; D],
            dq: vec![0.0; N],
            dk: vec![0.0; N],
            dg: vec![0.0; N],
            db: vec![0.0; D],
        };
        let scale = 2.0f64.powi(if tiny { -28 } else { -8 });
        for t in 0..D {
            f.b[t] = match case {
                6 => 0.0,
                7 => 1.0,
                _ => (1 + t % 4) as f64 / 8.0,
            };
            f.db[t] = ((t + case) % 5) as f64 * scale / 8.0;
            for h in 0..features {
                let i = t * D + h;
                f.q[i] = ((t * 13 + h * 7 + case) as i32 % 9 - 4) as f64 / 256.0;
                f.k[i] = ((t * 5 + h * 3 + case) as i32 % 5 - 2) as f64 / 16.0;
                f.g[i] = f64::from(match case {
                    0 => -0.01f32,
                    1 => -0.3,
                    2 => {
                        if t == 0 {
                            -600.0
                        } else {
                            -0.0001
                        }
                    }
                    3 => -8.0,
                    4 => 0.0,
                    _ => -0.015 * (1 + (t + h * 3) % 7) as f32,
                });
                f.uq[i] = (((t * 3 + h * 11 + case) % 9) as i32 - 4) as f64 * scale;
                f.uk[i] = (((t * 13 + h * 7 + case) % 11) as i32 - 5) as f64 * scale;
                f.ub[i] = (((t * 7 + h * 3 + case) % 7) as i32 - 3) as f64 * scale;
                f.dq[i] = ((t + h + case) % 5) as f64 * scale / 8.0;
                f.dk[i] = -(((t * 3 + h + case) % 5) as f64) * scale / 16.0;
                f.dg[i] = ((t + h * 3 + case) % 3) as f64 * scale / 32.0;
            }
        }
        for h in 0..features {
            f.ue[h] = ((h + case) as i32 % 7 - 3) as f64 * scale;
        }
        f
    }
    fn append(&mut self, f: &Self, scale: f64) {
        self.q.extend(&f.q);
        self.k.extend(&f.k);
        self.g.extend(&f.g);
        self.b.extend(&f.b);
        for (dst, src) in [
            (&mut self.uq, &f.uq),
            (&mut self.uk, &f.uk),
            (&mut self.ub, &f.ub),
            (&mut self.ue, &f.ue),
            (&mut self.dq, &f.dq),
            (&mut self.dk, &f.dk),
            (&mut self.dg, &f.dg),
            (&mut self.db, &f.db),
        ] {
            dst.extend(src.iter().map(|&x| x * scale));
        }
    }
    fn zero_initial(&mut self) {
        self.dq.fill(0.0);
        self.dk.fill(0.0);
        self.dg.fill(0.0);
        self.db.fill(0.0);
    }
}
fn prefix(f: &Fixture, t: usize, h: usize) -> f64 {
    let mut x = 0.0f32;
    for s in 0..=t {
        x += f.g[s * D + h] as f32;
    }
    f64::from(x)
}
fn suffix(f: &Fixture, t: usize, h: usize, valid: usize) -> f64 {
    let mut x = 0.0f32;
    for s in (t + 1..valid).rev() {
        x += f.g[s * D + h] as f32;
    }
    f64::from(x)
}
fn reference(f: &Fixture, valid: usize, features: usize) -> [Vec<f64>; 4] {
    let mut out = [f.dq.clone(), f.dk.clone(), f.dg.clone(), f.db.clone()];
    for h in 0..features {
        let carry = prefix(f, valid - 1, h).exp() * f.ue[h];
        for s in 0..valid {
            out[2][s * D + h] += carry;
        }
        for t in 0..valid {
            let i = t * D + h;
            let ep = prefix(f, t, h).exp();
            let es = suffix(f, t, h, valid).exp();
            let b = f.b[t];
            out[0][i] += ep * f.uq[i];
            out[1][i] += b * ep * f.ub[i] + es * f.uk[i];
            out[3][t] += f.k[i] * ep * f.ub[i];
            let gp = f.q[i] * ep * f.uq[i] + b * f.k[i] * ep * f.ub[i];
            let gs = f.k[i] * es * f.uk[i];
            for s in 0..=t {
                out[2][s * D + h] += gp;
            }
            for s in t + 1..valid {
                out[2][s * D + h] += gs;
            }
        }
    }
    out
}
// The objective is linear in its rounded boundary outputs. Their frozen cast
// residuals cancel in a central difference, leaving the smooth STE expression.
fn objective(changed: &Fixture, base: &Fixture, valid: usize, features: usize) -> f64 {
    let mut result = 0.0;
    for h in 0..features {
        let mut total_delta = 0.0;
        for s in 0..valid {
            total_delta += changed.g[s * D + h] - base.g[s * D + h];
        }
        result += base.ue[h] * (prefix(base, valid - 1, h) + total_delta).exp();
        for t in 0..valid {
            let i = t * D + h;
            let pd = (0..=t)
                .map(|s| changed.g[s * D + h] - base.g[s * D + h])
                .sum::<f64>();
            let sd = (t + 1..valid)
                .map(|s| changed.g[s * D + h] - base.g[s * D + h])
                .sum::<f64>();
            let ep = (prefix(base, t, h) + pd).exp();
            let es = (suffix(base, t, h, valid) + sd).exp();
            result += (base.uq[i] * changed.q[i] + base.ub[i] * changed.b[t] * changed.k[i]) * ep
                + base.uk[i] * changed.k[i] * es;
        }
    }
    result
}
struct DeviceInputs {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    uq: DeviceBuffer<f32>,
    uk: DeviceBuffer<f32>,
    ub: DeviceBuffer<f32>,
    ue: DeviceBuffer<f32>,
}
struct Grads {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
}
fn upload(
    stream: &CudaStream,
    v: &[f64],
    guard: usize,
) -> Result<DeviceBuffer<f32>, cuda_core::DriverError> {
    let mut data = v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    data.extend(std::iter::repeat_n(f32::from_bits(0x7fc01234), guard));
    DeviceBuffer::from_host(stream, &data)
}
impl DeviceInputs {
    fn new(stream: &CudaStream, f: &Fixture) -> Result<Self, cuda_core::DriverError> {
        Ok(Self {
            q: upload(stream, &f.q, 0)?,
            k: upload(stream, &f.k, 0)?,
            g: upload(stream, &f.g, 0)?,
            b: upload(stream, &f.b, 0)?,
            uq: upload(stream, &f.uq, 0)?,
            uk: upload(stream, &f.uk, 0)?,
            ub: upload(stream, &f.ub, 0)?,
            ue: upload(stream, &f.ue, 0)?,
        })
    }
}
impl Grads {
    fn new(stream: &CudaStream, f: &Fixture) -> Result<Self, cuda_core::DriverError> {
        Ok(Self {
            q: upload(stream, &f.dq, GUARD)?,
            k: upload(stream, &f.dk, GUARD)?,
            g: upload(stream, &f.dg, GUARD)?,
            b: upload(stream, &f.db, GUARD)?,
        })
    }
    fn run(
        &mut self,
        m: &KdaExactBoundaryVjpModule,
        s: &CudaStream,
        f: &DeviceInputs,
        bh: u32,
        seq_len: u32,
    ) -> Result<(), cuda_core::DriverError> {
        m.vjp(BoundaryVjpArgs {
            stream: s,
            q: &f.q,
            k: &f.k,
            g: &f.g,
            beta: &f.b,
            dqg: &f.uq,
            dkg: &f.uk,
            dincoming: &f.ub,
            dexit: &f.ue,
            dq: &mut self.q,
            dk: &mut self.k,
            dg: &mut self.g,
            dbeta: &mut self.b,
            batch_heads: bh,
            seq_len,
        })
    }
    fn host(&self, s: &CudaStream) -> Result<[Vec<f32>; 4], cuda_core::DriverError> {
        Ok([
            self.q.to_host_vec(s)?,
            self.k.to_host_vec(s)?,
            self.g.to_host_vec(s)?,
            self.b.to_host_vec(s)?,
        ])
    }
}

#[test]
#[ignore = "native8192 boundary VJP accumulation, two-step STE FD and multi-sequence tails"]
fn native_kda_exact_boundary_vjp_accumulation_intervals_and_tails() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(KdaExactBoundaryVjpModule::from_module)?;
    let cases = (0..8)
        .map(|i| Fixture::new(i, 64, i % 2 == 0))
        .collect::<Vec<_>>();
    let refs = cases
        .iter()
        .map(|f| reference(f, 64, 64))
        .collect::<Vec<_>>();
    let mut all = Fixture::empty();
    for chunk in 0..CHUNKS {
        all.append(&cases[chunk % 8], (chunk + 1) as f64 / 8192.0);
    }
    let inputs = DeviceInputs::new(&stream, &all)?;
    let mut out = Grads::new(&stream, &all)?;
    drop(all);
    out.run(&module, &stream, &inputs, 256, 2048)?;
    let full = out.host(&stream)?;
    let mut max_errors = [0.0f64; 4];
    let mut tiny_nonzero = 0usize;
    for chunk in 0..CHUNKS {
        let scale = (chunk + 1) as f64 / 8192.0;
        for plane in 0..4 {
            let count = if plane == 3 { D } else { N };
            for i in 0..count {
                let target = refs[chunk % 8][plane][i] * scale;
                let value = f64::from(full[plane][chunk * count + i]);
                let error = (value - target).abs();
                max_errors[plane] = max_errors[plane].max(error);
                let tol = if chunk % 2 == 0 { 2e-12 } else { 2e-6 };
                assert!(
                    value.is_finite() && error <= tol + 2e-4 * target.abs(),
                    "boundary VJP chunk={chunk} plane={plane} i={i} actual={value} target={target} error={error}"
                );
                if chunk % 2 == 0 && target.abs() > 1e-12 {
                    assert_ne!(value, 0.0);
                    tiny_nonzero += 1;
                }
            }
        }
    }
    assert!(tiny_nonzero > 1_000_000);
    for plane in 0..4 {
        let count = if plane == 3 { CHUNKS * D } else { CHUNKS * N };
        assert!(
            full[plane][count..]
                .iter()
                .all(|x| x.to_bits() == 0x7fc01234)
        );
    }
    // Each of two sequences has two chunks. Repack each prefix into its actual
    // padded chunk stride and poison only the nonexistent tail tokens.
    for cut in [1usize, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 127] {
        let mut f = Fixture::empty();
        for chunk in 0..4 {
            let mut a = Fixture::new(chunk, 64, false);
            a.zero_initial();
            a.uk.fill(0.0);
            a.ue.fill(0.0);
            f.append(&a, 1.0);
        }
        for bh in 0..2 {
            for token in cut..128 {
                let i = bh * 2 * N + token * D;
                f.uq[i..i + D].fill(0.0);
                f.ub[i..i + D].fill(0.0);
            }
        }
        let initial = DeviceInputs::new(&stream, &f)?;
        let mut g = Grads::new(&stream, &f)?;
        g.run(&module, &stream, &initial, 2, 128)?;
        let original = g.host(&stream)?;
        for bh in 0..2 {
            for token in cut..128 {
                let i = bh * 2 * N + token * D;
                f.q[i..i + D].fill(0.375);
                f.k[i..i + D].fill(-0.25);
                f.g[i..i + D].fill(-31.0);
                f.b[bh * 128 + token] = 0.25;
            }
        }
        let changed = DeviceInputs::new(&stream, &f)?;
        let mut cg = Grads::new(&stream, &f)?;
        cg.run(&module, &stream, &changed, 2, 128)?;
        let after = cg.host(&stream)?;
        for bh in 0..2 {
            for plane in 0..4 {
                let width = if plane == 3 { 1 } else { D };
                let at = bh * 128 * width;
                assert!(
                    original[plane][at..at + cut * width]
                        .iter()
                        .zip(&after[plane][at..at + cut * width])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "boundary future prefix cut={cut} bh={bh} plane={plane}"
                );
                assert!(
                    after[plane][at + cut * width..at + 128 * width]
                        .iter()
                        .all(|x| x.to_bits() == 0),
                    "boundary future support cut={cut} bh={bh} plane={plane}"
                );
            }
        }
        let chunks = cut.div_ceil(64);
        let mut prefix_f = Fixture::empty();
        for bh in 0..2 {
            for chunk in 0..chunks {
                let start = (bh * 2 + chunk) * N;
                let scalar = (bh * 2 + chunk) * D;
                let piece = Fixture {
                    q: f.q[start..start + N].to_vec(),
                    k: f.k[start..start + N].to_vec(),
                    g: f.g[start..start + N].to_vec(),
                    b: f.b[scalar..scalar + D].to_vec(),
                    uq: f.uq[start..start + N].to_vec(),
                    uk: f.uk[start..start + N].to_vec(),
                    ub: f.ub[start..start + N].to_vec(),
                    ue: f.ue[scalar..scalar + D].to_vec(),
                    dq: vec![0.0; N],
                    dk: vec![0.0; N],
                    dg: vec![0.0; N],
                    db: vec![0.0; D],
                };
                prefix_f.append(&piece, 1.0);
            }
        }
        for bh in 0..2 {
            for token in cut..chunks * D {
                let at = (bh * chunks * D + token) * D;
                for v in [
                    &mut prefix_f.q,
                    &mut prefix_f.k,
                    &mut prefix_f.g,
                    &mut prefix_f.uq,
                    &mut prefix_f.uk,
                    &mut prefix_f.ub,
                ] {
                    v[at..at + D].fill(f64::NAN);
                }
                prefix_f.b[bh * chunks * D + token] = f64::NAN;
            }
        }
        let pd = DeviceInputs::new(&stream, &prefix_f)?;
        let mut pg = Grads::new(&stream, &prefix_f)?;
        pg.run(&module, &stream, &pd, 2, cut as u32)?;
        let got = pg.host(&stream)?;
        for bh in 0..2 {
            for plane in 0..4 {
                let width = if plane == 3 { 1 } else { D };
                let source = bh * 128 * width;
                let at = bh * chunks * D * width;
                assert!(
                    got[plane][at..at + cut * width]
                        .iter()
                        .zip(&original[plane][source..source + cut * width])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "boundary truncated prefix cut={cut} bh={bh} plane={plane}"
                );
                assert!(
                    got[plane][at + cut * width..at + chunks * D * width]
                        .iter()
                        .all(|x| x.to_bits() == 0),
                    "boundary poisoned tail cut={cut}"
                );
            }
        }
    }
    let coords = [
        (0, 0, 1),
        (0, 16, 2),
        (0, 32, 3),
        (0, 63, 0),
        (1, 0, 2),
        (1, 15, 1),
        (1, 31, 3),
        (1, 63, 0),
        (2, 0, 0),
        (2, 1, 1),
        (2, 16, 2),
        (2, 31, 3),
        (2, 32, 1),
        (2, 63, 2),
        (3, 16, 0),
        (3, 63, 0),
    ];
    let mut max_fd_error = 0.0f64;
    let mut max_native_fd_error = 0.0f64;
    for case in [0usize, 1, 2, 5] {
        let mut f = Fixture::new(case, 4, false);
        f.zero_initial();
        let r = reference(&f, 64, 4);
        let input = DeviceInputs::new(&stream, &f)?;
        let mut grads = Grads::new(&stream, &f)?;
        grads.run(&module, &stream, &input, 1, 64)?;
        let native = grads.host(&stream)?;
        for &(plane, t, h) in &coords {
            let i = if plane == 3 { t } else { t * D + h };
            let target = r[plane][i];
            for step in [1e-4, 5e-5] {
                let mut plus = f.clone();
                let mut minus = f.clone();
                let a = match plane {
                    0 => &mut plus.q,
                    1 => &mut plus.k,
                    2 => &mut plus.g,
                    _ => &mut plus.b,
                };
                a[i] += step;
                let b = match plane {
                    0 => &mut minus.q,
                    1 => &mut minus.k,
                    2 => &mut minus.g,
                    _ => &mut minus.b,
                };
                b[i] -= step;
                let fd =
                    (objective(&plus, &f, 64, 4) - objective(&minus, &f, 64, 4)) / (2.0 * step);
                max_fd_error = max_fd_error.max((fd - target).abs());
                max_native_fd_error =
                    max_native_fd_error.max((fd - f64::from(native[plane][i])).abs());
                assert!(
                    (fd - target).abs() <= 2e-8 + 1e-5 * target.abs(),
                    "boundary independent FD case={case} plane={plane} t={t} h={h} fd={fd} expected={target}"
                );
                assert!(
                    (fd - f64::from(native[plane][i])).abs() <= 2e-6 + 5e-4 * fd.abs(),
                    "boundary native FD"
                );
            }
        }
    }
    for (bh, seq_len) in [(0, 2048), (256, 0), (u32::MAX, 2048), (256, u32::MAX)] {
        assert!(out.run(&module, &stream, &inputs, bh, seq_len).is_err());
    }
    let short = DeviceInputs::new(&stream, &cases[0])?;
    assert!(out.run(&module, &stream, &short, 256, 2048).is_err());
    let same = out.host(&stream)?;
    for plane in 0..4 {
        assert!(
            same[plane]
                .iter()
                .zip(&full[plane])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    println!(
        "native_exact_KDA_boundary_VJP_verified chunk_batch={CHUNKS} pattern_cases=8 accumulation=true tiny_nonzero={tiny_nonzero} multisequence_prefix_cuts=14 literal_zero_future=true STE_coordinates=64 STE_step_sizes=2 max_errors={max_errors:?} max_analytic_FD_error={max_fd_error:e} max_native_FD_error={max_native_fd_error:e} guards=unchanged optimizer_updates=0"
    );
    Ok(())
}
