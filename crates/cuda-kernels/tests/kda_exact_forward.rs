//! Complete-forward failure cases declared before native orchestration/state:
//! intact B8/S2048/H32, all32recurrent chunks, nonzero initial states, original
//! raw activations/gates, no/crossed/strong decay, beta limits, independent
//! recurrent output/entry/exit-state oracle, zero-state dispatch equivalence,
//! exact truncated-prefix causality, sequence boundaries and output guards.
//! This verifies a CUDA attention operator; the host oracle never trains.
use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::kda_exact_forward::{ForwardArgs, ForwardScratch, KdaExactForwardModule};
use std::error::Error;
mod common;
#[allow(dead_code)]
#[path = "causal_attention_backward_tc/f16.rs"]
mod half_reference;
const B: usize = 8;
const S: usize = 2048;
const H: usize = 32;
const D: usize = 2048;
const QKV: usize = 8320;
const GUARD: usize = 257;

fn raw(bh: usize, t: usize, h: usize, plane: usize) -> f32 {
    let case = bh % 8;
    match plane {
        0 | 1 => ((t * (7 + plane * 4) + h * (3 + plane * 2) + bh) as i32 % 17 - 8) as f32 / 16.0,
        2 => ((t * 3 + h * 7 + bh) as i32 % 13 - 6) as f32 / 32.0,
        3 => match case {
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
            _ => ((t + h * 3) % 9) as f32 / 4.0,
        },
        _ => unreachable!(),
    }
}
fn beta_raw(bh: usize, t: usize) -> f32 {
    match bh % 8 {
        6 => -100.0,
        7 => 100.0,
        _ => (t as i32 % 7 - 3) as f32 / 4.0,
    }
}
fn initial(bh: usize, k: usize, v: usize) -> f32 {
    if bh % 2 == 0 {
        ((bh + k * 3 + v * 7) % 17) as f32 / 4096.0 - 0.002
    } else {
        0.0
    }
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}
fn log_gate(x: f64) -> f64 {
    -0.01 * if x > 20.0 { x } else { x.exp().ln_1p() }
}

struct Buffers {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    lower: DeviceBuffer<f32>,
    bu: DeviceBuffer<f32>,
    qg: DeviceBuffer<u16>,
    kg: DeviceBuffer<u16>,
    vb: DeviceBuffer<u16>,
    inv: DeviceBuffer<u16>,
    w: DeviceBuffer<u16>,
    c: DeviceBuffer<u16>,
    r: DeviceBuffer<u16>,
    states: DeviceBuffer<u16>,
    out: DeviceBuffer<f32>,
    final_state: DeviceBuffer<f32>,
    tape: DeviceBuffer<u16>,
}
impl Buffers {
    fn new(stream: &CudaStream) -> Result<Self, cuda_core::DriverError> {
        let n = B * H * S * 64;
        let fp = || DeviceBuffer::from_host(stream, &vec![f32::from_bits(0x7fc01234); n + GUARD]);
        let half = || DeviceBuffer::from_host(stream, &vec![0x7e12u16; n + GUARD]);
        Ok(Self {
            q: fp()?,
            k: fp()?,
            g: fp()?,
            beta: DeviceBuffer::from_host(stream, &vec![f32::NAN; n / 64 + GUARD])?,
            lower: fp()?,
            bu: fp()?,
            qg: half()?,
            kg: half()?,
            vb: half()?,
            inv: half()?,
            w: half()?,
            c: half()?,
            r: half()?,
            states: half()?,
            out: fp()?,
            final_state: DeviceBuffer::from_host(
                stream,
                &vec![f32::from_bits(0x7fc01234); B * H * 4096 + GUARD],
            )?,
            tape: DeviceBuffer::from_host(stream, &vec![0x7e12u16; B * S * QKV + GUARD])?,
        })
    }
    fn run(
        &mut self,
        module: &KdaExactForwardModule,
        stream: &CudaStream,
        qkv: &DeviceBuffer<f32>,
        init: Option<&DeviceBuffer<f32>>,
        s: usize,
        save: bool,
    ) -> Result<(), cuda_core::DriverError> {
        module.forward(ForwardArgs {
            stream,
            qkv,
            qkv_f16: if save { Some(&mut self.tape) } else { None },
            initial_state: init,
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
                inverse: &mut self.inv,
                w: &mut self.w,
                readout: &mut self.c,
                residual: &mut self.r,
                chunk_states: &mut self.states,
                final_state: &mut self.final_state,
            },
            batch_size: B as u32,
            seq_len: s as u32,
            head_count: H as u32,
            qkv_dim: QKV as u32,
        })
    }
}

#[test]
#[ignore = "complete native B8/S2048 exact-factor KDA forward and recurrent oracle"]
fn native_kda_exact_complete_forward_recurrence_states_and_causality() -> Result<(), Box<dyn Error>>
{
    let (_, stream, module) = common::cuda_test_module(KdaExactForwardModule::from_module)?;
    let mut input = vec![-0.375f32; B * S * QKV];
    for b in 0..B {
        for t in 0..S {
            for h in 0..H {
                let bh = b * H + h;
                for d in 0..64 {
                    for p in 0..4 {
                        input[(b * S + t) * QKV + p * D + h * 64 + d] = raw(bh, t, d, p);
                    }
                }
                input[(b * S + t) * QKV + 4 * D + h] = beta_raw(bh, t);
            }
        }
    }
    let init = (0..B * H * 4096)
        .map(|i| initial(i / 4096, i / 64 % 64, i % 64))
        .collect::<Vec<_>>();
    let init_dev = DeviceBuffer::from_host(&stream, &init)?;
    let input_dev = DeviceBuffer::from_host(&stream, &input)?;
    let mut out = Buffers::new(&stream)?;
    out.run(&module, &stream, &input_dev, Some(&init_dev), S, true)?;
    let actual = out.out.to_host_vec(&stream)?;
    let states = out.states.to_host_vec(&stream)?;
    let final_state = out.final_state.to_host_vec(&stream)?;
    let n = B * S * D;
    assert!(actual[..n].iter().all(|x| x.is_finite()));
    assert!(actual[n..].iter().all(|x| x.to_bits() == 0x7fc01234));
    assert!(
        final_state[B * H * 4096..]
            .iter()
            .all(|x| x.to_bits() == 0x7fc01234)
    );
    assert!(states[B * H * S * 64..].iter().all(|&x| x == 0x7e12));
    let mut output_error2 = 0.0;
    let mut output_ref2 = 0.0;
    let mut state_error2 = 0.0;
    let mut state_ref2 = 0.0;
    let mut max_output_error = 0.0f64;
    let mut max_state_error = 0.0f64;
    for bh in 0..B * H {
        let b = bh / H;
        let head = bh % H;
        let mut state = init[bh * 4096..(bh + 1) * 4096]
            .iter()
            .map(|&x| f64::from(x))
            .collect::<Vec<_>>();
        for t in 0..S {
            if t % 64 == 0 {
                let base = (bh * (S / 64) + t / 64) * 4096;
                for i in 0..4096 {
                    let value = f64::from(half_reference::f16_bits_to_f32(states[base + i]));
                    let error = (value - state[i]).abs();
                    max_state_error = max_state_error.max(error);
                    state_error2 += error * error;
                    state_ref2 += state[i] * state[i];
                    assert!(
                        value.is_finite() && error <= 1e-2,
                        "entry state bh={bh} chunk={} i={i} error={error}",
                        t / 64
                    );
                }
            }
            let mut q = [0.0f64; 64];
            let mut k = [0.0f64; 64];
            let mut v = [0.0f64; 64];
            let mut decay = [0.0f64; 64];
            let mut qss = 0.0;
            let mut kss = 0.0;
            for d in 0..64 {
                q[d] = silu(f64::from(raw(bh, t, d, 0)));
                k[d] = silu(f64::from(raw(bh, t, d, 1)));
                v[d] = silu(f64::from(raw(bh, t, d, 2)));
                decay[d] = log_gate(f64::from(raw(bh, t, d, 3))).exp();
                qss += q[d] * q[d];
                kss += k[d] * k[d];
            }
            let qinv = 0.125 / ((qss + 1e-6).sqrt() + 1e-6);
            let kinv = 1.0 / ((kss + 1e-6).sqrt() + 1e-6);
            for d in 0..64 {
                q[d] *= qinv;
                k[d] *= kinv;
            }
            let beta = sigmoid(f64::from(beta_raw(bh, t)));
            let mut prediction = [0.0f64; 64];
            for h in 0..64 {
                for d in 0..64 {
                    state[h * 64 + d] *= decay[h];
                    prediction[d] += k[h] * state[h * 64 + d];
                }
            }
            let mut residual = [0.0f64; 64];
            for d in 0..64 {
                residual[d] = beta * (v[d] - prediction[d]);
            }
            let mut expected = [0.0f64; 64];
            for h in 0..64 {
                for d in 0..64 {
                    state[h * 64 + d] += k[h] * residual[d];
                    expected[d] += q[h] * state[h * 64 + d];
                }
            }
            for d in 0..64 {
                let value = f64::from(actual[(b * S + t) * D + head * 64 + d]);
                let error = (value - expected[d]).abs();
                max_output_error = max_output_error.max(error);
                output_error2 += error * error;
                output_ref2 += expected[d] * expected[d];
                assert!(
                    error <= 2e-3,
                    "complete recurrent output bh={bh} t={t} d={d} value={value} expected={} error={error}",
                    expected[d]
                );
            }
        }
        for i in 0..4096 {
            let value = f64::from(final_state[bh * 4096 + i]);
            let error = (value - state[i]).abs();
            max_state_error = max_state_error.max(error);
            state_error2 += error * error;
            state_ref2 += state[i] * state[i];
            assert!(
                value.is_finite() && error <= 1e-2,
                "final state bh={bh} i={i} error={error}"
            );
        }
    }
    let output_relative_l2 = (output_error2 / output_ref2.max(1e-30)).sqrt();
    let state_relative_l2 = (state_error2 / state_ref2.max(1e-30)).sqrt();
    assert!(
        output_relative_l2 < 0.03 && state_relative_l2 < 0.03,
        "independent recurrent error output={output_relative_l2} state={state_relative_l2}"
    );
    // The complete forward must preserve exact native prefix outputs. Packing
    // each input sequence to its actual truncated length catches padded-stride
    // bugs that a one-sequence or whole64-chunk-only fixture would miss.
    for cut in [1usize, 15, 16, 17, 63, 64, 65, 127] {
        let mut prefix = Vec::with_capacity(B * cut * QKV);
        for b in 0..B {
            prefix.extend_from_slice(&input[b * S * QKV..(b * S + cut) * QKV]);
        }
        let prefix_dev = DeviceBuffer::from_host(&stream, &prefix)?;
        out.run(&module, &stream, &prefix_dev, Some(&init_dev), cut, false)?;
        let got = out.out.to_host_vec(&stream)?;
        for b in 0..B {
            for t in 0..cut {
                for d in 0..D {
                    assert_eq!(
                        got[(b * cut + t) * D + d].to_bits(),
                        actual[(b * S + t) * D + d].to_bits(),
                        "complete forward prefix cut={cut} b={b} t={t} d={d}"
                    );
                }
            }
        }
    }
    let zeros = DeviceBuffer::<f32>::zeroed(&stream, B * H * 4096)?;
    out.run(&module, &stream, &input_dev, None, S, false)?;
    let zero_out = out.out.to_host_vec(&stream)?;
    let zero_exit = out.final_state.to_host_vec(&stream)?;
    out.run(&module, &stream, &input_dev, Some(&zeros), S, false)?;
    assert!(
        out.out
            .to_host_vec(&stream)?
            .iter()
            .zip(&zero_out)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        out.final_state
            .to_host_vec(&stream)?
            .iter()
            .zip(&zero_exit)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    let short = DeviceBuffer::from_host(&stream, &[0.0f32])?;
    assert!(
        out.run(&module, &stream, &short, Some(&init_dev), S, false)
            .is_err()
    );
    assert!(
        out.run(&module, &stream, &input_dev, Some(&short), S, false)
            .is_err()
    );
    assert!(
        out.run(&module, &stream, &input_dev, None, 0, false)
            .is_err()
    );
    assert!(
        out.out
            .to_host_vec(&stream)?
            .iter()
            .zip(&zero_out)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "shape rejection mutated complete output"
    );
    println!(
        "native_exact_KDA_complete_forward_verified B=8 S=2048 H=32 chunks_per_sequence=32 independent_recurrence=true nonzero_initial_state=true prefix_cuts=8 zero_initial_dispatch=bit_exact max_output_error={max_output_error:e} max_state_error={max_state_error:e} output_relative_L2={output_relative_l2:e} state_relative_L2={state_relative_l2:e} guards=unchanged optimizer_updates=0"
    );
    Ok(())
}
