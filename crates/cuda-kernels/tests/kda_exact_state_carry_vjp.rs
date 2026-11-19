//! Pre-implementation contract: consume the true native recurrent accumulator
//! for carry gradients, replaying cached FP16 Kg/R into shared memory. Never
//! save a global FP32 entry-state tape. Verify every32-chunk boundary against
//! actual native prefix finals, with one-hot probes recovering exact bits and
//! independent dense vector cotangents/scales, zero/nonzero initial states,
//! partial sequences, guards and shape rejection. FP32 prefix finals below are
//! diagnostic snapshots from the previously verified forward API, not training
//! storage or a replacement forward backend.
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::kda_exact_forward::{ForwardArgs, ForwardScratch, KdaExactForwardModule};
use rust_kernels_cuda::kda_exact_state_carry_vjp::{CarryVjpArgs, KdaExactStateCarryVjpModule};
use std::error::Error;
mod common;
#[allow(dead_code)]
#[path = "causal_attention_backward_tc/f16.rs"]
mod half_reference;
const GUARD: usize = 257;
struct Forward {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    lower: DeviceBuffer<f32>,
    bu: DeviceBuffer<f32>,
    qg: DeviceBuffer<u16>,
    kg: DeviceBuffer<u16>,
    vb: DeviceBuffer<u16>,
    j: DeviceBuffer<u16>,
    w: DeviceBuffer<u16>,
    c: DeviceBuffer<u16>,
    r: DeviceBuffer<u16>,
    states: DeviceBuffer<u16>,
    last: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
}
impl Forward {
    fn new(stream: &CudaStream, b: usize, s: usize, h: usize) -> Result<Self, DriverError> {
        let n = b * h * s.div_ceil(64) * 4096;
        let fp = || DeviceBuffer::zeroed(stream, n);
        let hp = || DeviceBuffer::zeroed(stream, n);
        Ok(Self {
            q: fp()?,
            k: fp()?,
            g: fp()?,
            beta: DeviceBuffer::zeroed(stream, n / 64)?,
            lower: fp()?,
            bu: fp()?,
            qg: hp()?,
            kg: hp()?,
            vb: hp()?,
            j: hp()?,
            w: hp()?,
            c: hp()?,
            r: hp()?,
            states: hp()?,
            last: DeviceBuffer::zeroed(stream, b * h * 4096)?,
            out: fp()?,
        })
    }
    fn run(
        &mut self,
        module: &KdaExactForwardModule,
        stream: &CudaStream,
        qkv: &DeviceBuffer<f32>,
        initial: Option<&DeviceBuffer<f32>>,
        b: usize,
        s: usize,
        h: usize,
    ) -> Result<(), DriverError> {
        module.forward(ForwardArgs {
            stream,
            qkv,
            qkv_f16: None,
            initial_state: initial,
            out: &mut self.out,
            scratch: ForwardScratch {
                q: &mut self.q,
                k: &mut self.k,
                g: &mut self.g,
                beta: &mut self.beta,
                lower: &mut self.lower,
                incoming_rhs_u: &mut self.bu,
                qg: &mut self.qg,
                kg: &mut self.kg,
                vbeta: &mut self.vb,
                inverse: &mut self.j,
                w: &mut self.w,
                readout: &mut self.c,
                residual: &mut self.r,
                chunk_states: &mut self.states,
                final_state: &mut self.last,
            },
            batch_size: b as u32,
            seq_len: s as u32,
            head_count: h as u32,
            qkv_dim: (h * 260) as u32,
        })
    }
}
fn input(b: usize, s: usize, h: usize) -> Vec<f32> {
    let d = h * 64;
    let stride = h * 260;
    let mut out = vec![-0.375; b * s * stride];
    for batch in 0..b {
        for t in 0..s {
            for head in 0..h {
                let case = (batch * h + head) % 8;
                let row = (batch * s + t) * stride;
                for k in 0..64 {
                    out[row + head * 64 + k] =
                        (((t * 7 + k * 3 + case) % 17) as i32 - 8) as f32 / 16.0;
                    out[row + d + head * 64 + k] =
                        (((t * 11 + k * 5 + case) % 19) as i32 - 9) as f32 / 16.0;
                    out[row + 2 * d + head * 64 + k] =
                        (((t * 3 + k * 7 + case) % 13) as i32 - 6) as f32 / 32.0;
                    out[row + 3 * d + head * 64 + k] = match case {
                        0 => -100.0,
                        1 => -4.600166,
                        2 => 8.0,
                        3 => 1000.0,
                        4 => {
                            if t % 64 == 0 {
                                60000.0
                            } else {
                                -4.600166
                            }
                        }
                        _ => ((t + k * 3) % 9) as f32 / 4.0,
                    };
                }
                out[row + 4 * d + head] = match case {
                    6 => -100.0,
                    7 => 100.0,
                    _ => (t as i32 % 7 - 3) as f32 / 4.0,
                };
            }
        }
    }
    out
}
fn initial(b: usize, h: usize) -> Vec<f32> {
    (0..b * h * 4096)
        .map(|i| {
            let case = i / 4096 % 8;
            let k = i / 64 % 64;
            let v = i % 64;
            if case % 2 == 0 {
                0.0
            } else {
                (((k * 3 + v * 7 + case) % 17) as i32 - 8) as f32 / 4096.0 + 0.00000037
            }
        })
        .collect()
}
fn half(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let negative = x.is_sign_negative();
    let a = x.abs();
    let bits = a.to_bits();
    let exp = ((bits >> 23) & 255) as i32 - 127;
    let raw = if exp < -14 {
        (a * 16777216.0).round_ties_even() as u32
    } else {
        ((exp + 15) as u32 * 1024)
            + ((a / 2.0f32.powi(exp) - 1.0) * 1024.0).round_ties_even() as u32
    };
    (raw as u16) | if negative { 0x8000 } else { 0 }
}
fn run_case(
    stream: &CudaStream,
    forward: &KdaExactForwardModule,
    module: &KdaExactStateCarryVjpModule,
    b: usize,
    s: usize,
    h: usize,
    with_initial: bool,
) -> Result<(usize, f64), Box<dyn Error>> {
    let chunks = s.div_ceil(64);
    let bh = b * h;
    let n = bh * chunks * 4096;
    let controls = n / 64;
    let raw = input(b, s, h);
    let raw_dev = DeviceBuffer::from_host(stream, &raw)?;
    let init = initial(b, h);
    let init_dev = DeviceBuffer::from_host(stream, &init)?;
    let initial_arg = if with_initial { Some(&init_dev) } else { None };
    let mut buffers = Forward::new(stream, b, s, h)?;
    buffers.run(forward, stream, &raw_dev, initial_arg, b, s, h)?;
    let kg = DeviceBuffer::from_host(stream, &buffers.kg.to_host_vec(stream)?)?;
    let residual = DeviceBuffer::from_host(stream, &buffers.r.to_host_vec(stream)?)?;
    let carry = DeviceBuffer::from_host(stream, &buffers.lower.to_host_vec(stream)?[..controls])?;
    let mut prefix_states = Vec::<Vec<f32>>::with_capacity(chunks);
    prefix_states.push(if with_initial {
        init.clone()
    } else {
        vec![0.0; bh * 4096]
    });
    for chunk in 1..chunks {
        let len = chunk * 64;
        let stride = h * 260;
        let mut prefix = Vec::with_capacity(b * len * stride);
        for batch in 0..b {
            prefix.extend_from_slice(&raw[batch * s * stride..batch * s * stride + len * stride]);
        }
        let prefix_dev = DeviceBuffer::from_host(stream, &prefix)?;
        buffers.run(forward, stream, &prefix_dev, initial_arg, b, len, h)?;
        prefix_states.push(buffers.last.to_host_vec(stream)?);
    }
    let mut distinguishing_entries = 0usize;
    let mut maximum_error = 0.0f64;
    for mode in 0..3 {
        let mut cot = vec![0u16; n];
        let mut unscale = vec![1.0f32; controls * 2];
        for head in 0..bh {
            for chunk in 0..chunks {
                for k in 0..64 {
                    let row = (head * chunks + chunk) * 64 + k;
                    if mode < 2 {
                        let v = (head / 8 + mode * 32) % 64;
                        cot[row * 64 + v] = 0x3c00;
                    } else {
                        // Tiny final cotangents are carried by native-half payloads
                        // plus row controls, never by a full FP32 state plane.
                        unscale[row * 2] = 2.0f32.powi(-80 + ((head + chunk + k) % 5) as i32);
                        for v in 0..64 {
                            cot[row * 64 + v] = half(
                                (((head * 3 + chunk + k * 7 + v * 11) % 13) as i32 - 6) as f32
                                    / 64.0,
                            );
                        }
                    }
                }
            }
        }
        let cot_dev = DeviceBuffer::from_host(stream, &cot)?;
        let scale_dev = DeviceBuffer::from_host(stream, &unscale)?;
        let mut parts = DeviceBuffer::from_host(
            stream,
            &vec![f32::from_bits(0x7fc01234); controls * 2 + GUARD],
        )?;
        let mut out =
            DeviceBuffer::from_host(stream, &vec![f32::from_bits(0x7fc01234); controls + GUARD])?;
        module.vjp(CarryVjpArgs {
            stream,
            kg: &kg,
            residual: &residual,
            carry: &carry,
            state_grad: &cot_dev,
            state_grad_unscale: &scale_dev,
            initial_state: initial_arg,
            partials: &mut parts,
            out: &mut out,
            batch_size: b as u32,
            seq_len: s as u32,
            head_count: h as u32,
        })?;
        let actual = out.to_host_vec(stream)?;
        let parts_host = parts.to_host_vec(stream)?;
        for head in 0..bh {
            for chunk in 0..chunks {
                for k in 0..64 {
                    let row = (head * chunks + chunk) * 64 + k;
                    let state =
                        &prefix_states[chunk][head * 4096 + k * 64..head * 4096 + (k + 1) * 64];
                    if mode < 2 {
                        let v = (head / 8 + mode * 32) % 64;
                        let expected = state[v];
                        assert_eq!(
                            actual[row].to_bits(),
                            expected.to_bits(),
                            "initial={with_initial} mode={mode} head={head} chunk={chunk} k={k}"
                        );
                        if half_reference::f16_bits_to_f32(half(expected)).to_bits()
                            != expected.to_bits()
                        {
                            distinguishing_entries += 1;
                        }
                    } else {
                        let terms = (0..64).map(|v| {
                            f64::from(state[v])
                                * f64::from(half_reference::f16_bits_to_f32(cot[row * 64 + v]))
                        });
                        let expected = terms.clone().sum::<f64>()
                            * f64::from(unscale[row * 2])
                            * f64::from(unscale[row * 2 + 1]);
                        let bound = 2e-6
                            * terms.map(f64::abs).sum::<f64>()
                            * f64::from(unscale[row * 2])
                            * f64::from(unscale[row * 2 + 1])
                            + 1e-40;
                        let error = (f64::from(actual[row]) - expected).abs();
                        maximum_error = maximum_error.max(error);
                        assert!(
                            actual[row].is_finite() && error <= bound,
                            "head={head} chunk={chunk} k={k} got={:e} expected={expected:e} error={error:e} bound={bound:e}",
                            actual[row]
                        );
                    }
                }
            }
        }
        assert!(actual[controls..].iter().all(|x| x.to_bits() == 0x7fc01234));
        assert!(
            parts_host[controls * 2..]
                .iter()
                .all(|x| x.to_bits() == 0x7fc01234)
        );
    }
    Ok((distinguishing_entries, maximum_error))
}
#[test]
#[ignore = "native B8/S2048 true carry VJP via shared-state TC replay, no FP32 state tape"]
fn native_kda_exact_state_carry_vjp_shared_replay_all_boundaries_and_tails()
-> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let forward = KdaExactForwardModule::from_module(ptx.clone())?;
    let module = KdaExactStateCarryVjpModule::from_module(ptx)?;
    let mut witnesses = 0usize;
    let mut error = 0.0f64;
    for with_initial in [false, true] {
        let (w, e) = run_case(&stream, &forward, &module, 8, 2048, 32, with_initial)?;
        witnesses += w;
        error = error.max(e);
    }
    assert!(
        witnesses > 10000,
        "insufficient full-accumulator witnesses={witnesses}"
    );
    for s in [1, 15, 16, 17, 63, 64, 65, 127] {
        let (w, e) = run_case(&stream, &forward, &module, 2, s, 2, true)?;
        witnesses += w;
        error = error.max(e);
    }
    let hp = DeviceBuffer::from_host(&stream, &[0u16])?;
    let fp = DeviceBuffer::from_host(&stream, &[1.0f32])?;
    let mut parts = DeviceBuffer::from_host(&stream, &[0.375f32; GUARD])?;
    let mut out = DeviceBuffer::from_host(&stream, &[0.375f32; GUARD])?;
    for (b, s, h) in [
        (0, 2048, 32),
        (8, 0, 32),
        (8, 2048, 0),
        (u32::MAX, 2048, 32),
        (8, u32::MAX, 32),
        (8, 2048, 32),
    ] {
        assert!(
            module
                .vjp(CarryVjpArgs {
                    stream: &stream,
                    kg: &hp,
                    residual: &hp,
                    carry: &fp,
                    state_grad: &hp,
                    state_grad_unscale: &fp,
                    initial_state: Some(&fp),
                    partials: &mut parts,
                    out: &mut out,
                    batch_size: b,
                    seq_len: s,
                    head_count: h
                })
                .is_err()
        );
    }
    assert!(parts.to_host_vec(&stream)?.iter().all(|&x| x == 0.375));
    assert!(out.to_host_vec(&stream)?.iter().all(|&x| x == 0.375));
    println!(
        "native_exact_KDA_state_carry_VJP_verified B=8 S=2048 H=32 chunk_batch=8192 initial_modes=2 partial_lengths=8 actual_native_prefix_finals=true FP16_state_gradient_payload=true FP32_state_tape=false distinguishing_entries={witnesses} max_dense_error={error:e}"
    );
    Ok(())
}
