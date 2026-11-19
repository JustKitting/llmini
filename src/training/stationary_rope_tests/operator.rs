//! Real native QKV -> positional map -> full attention -> raw-coordinate VJP.
//! The independent f64 code below is a mathematical oracle, never CPU training.
use crate::{AppResult, training::runtime::Runtime};
use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionCoreScratchBuffers, AttentionDims, GPT2_BATCH_SIZE as B, GPT2_N_EMBD as D,
    GPT2_N_HEAD as HEADS, GPT2_SEQ_LEN as S,
};
use rust_kernels_cuda::{
    attention::{ApplyRopeArgs, CausalAttentionBackwardTcArgs, CausalAttentionTcArgs},
    nvfp4::{Nvfp4DeviceTensor, csf},
};
const H: usize = 64;
const N: usize = B * S;
const READOUT: [usize; 8] = [1, 63, 64, 127, 128, 1023, 2015, 2047];

fn raw(b: usize, t: usize, section: usize, head: usize, j: usize) -> f64 {
    (((t * 7 + j * 3 + section * 5 + b * 11 + head) % 31) as f64 - 15.) / 16.
}
fn incoming(j: usize) -> f64 {
    [0.25, -0.125, 0.5, 0.0625][j % 4]
}
type Change = (usize, usize, usize, f64);
fn vector(b: usize, t: usize, section: usize, change: Option<Change>) -> [f64; H] {
    std::array::from_fn(|j| {
        raw(b, t, section, 0, j)
            + change
                .filter(|&(s, u, k, _)| (s, u, k) == (section, t, j))
                .map_or(0., |x| x.3)
    })
}
fn rotate(mut x: [f64; H], t: usize) -> [f64; H] {
    for pair in 0..16 {
        let angle = t as f64 * (1. / 1024f64).powf(pair as f64 / 15.);
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (x[2 * pair], x[2 * pair + 1]);
        x[2 * pair] = a * cos + b * sin;
        x[2 * pair + 1] = b * cos - a * sin;
    }
    x
}
fn normalized(x: [f64; H]) -> [f64; H] {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-6);
    x.map(|v| v / norm)
}
fn oracle(b: usize, change: Option<Change>) -> f64 {
    let query = READOUT[b];
    let q = normalized(rotate(vector(b, query, 0, change), query));
    let keys = (0..=query)
        .map(|t| normalized(rotate(vector(b, t, 1, change), t)))
        .collect::<Vec<_>>();
    let mut scores = Vec::with_capacity(query + 1);
    let mut readouts = Vec::with_capacity(query + 1);
    for t in 0..=query {
        scores.push(
            (0..H)
                .map(|j| q[j] * keys[if j < 32 { t } else { t.saturating_sub(1) }][j])
                .sum::<f64>(),
        );
        readouts.push(
            vector(b, t, 2, change)
                .iter()
                .enumerate()
                .map(|(j, x)| incoming(j) * x)
                .sum::<f64>(),
        );
    }
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
    weights
        .iter()
        .zip(readouts)
        .map(|(p, v)| p * v)
        .sum::<f64>()
        / weights.iter().sum::<f64>()
}
fn close(actual: f32, expected: f64, label: &str, abs: f64, rel: f64) {
    assert!(
        actual.is_finite() && (f64::from(actual) - expected).abs() <= abs + rel * expected.abs(),
        "{label} actual={actual:e} oracle={expected:e}"
    );
}

#[test]
#[ignore = "real B8/S2048/d2048 stationary-RoPE forward/tapes/offset/norm/attention/VJP and independent finite differences; zero updates"]
fn native_stationary_rope_complete_attention_operator() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_STATIONARY_ROPE);
    let r = Runtime::new()?;
    let s = r.stream.as_ref();
    let mut scratch = AttentionCoreScratchBuffers::new(s)?;
    let stride = AttentionDims::new(true).qkv_dim as usize;
    let values = (0..N * stride)
        .map(|i| {
            let row = i / stride;
            let col = i % stride;
            raw(row / S, row % S, col / D, (col % D) / H, col % H) as f32
        })
        .collect::<Vec<_>>();
    let mut qkv = DeviceBuffer::from_host(s, &values)?;
    let mut tape = DeviceBuffer::<u16>::zeroed(s, N * stride)?;
    r.attention.apply_rope(ApplyRopeArgs {
        stream: s,
        qkv: &mut qkv,
        qkv_f16: Some(&mut tape),
        row_count: N as u32,
        seq_len: S as u32,
        batch_size: B as u32,
        embedding_dim: D as u32,
        qkv_dim: stride as u32,
        head_count: HEADS as u32,
        head_dim: H as u32,
    })?;
    let rotated = qkv.to_host_vec(s)?;
    for row in 0..N {
        for col in 0..stride {
            if col >= 2 * D || col % H >= 32 || row % S == 0 {
                assert_eq!(
                    rotated[row * stride + col].to_bits(),
                    values[row * stride + col].to_bits(),
                    "identity row={row} col={col}"
                );
            }
        }
        for head in [0, HEADS - 1] {
            for section in 0..2 {
                let mut expected = std::array::from_fn(|j| raw(row / S, row % S, section, head, j));
                expected = rotate(expected, row % S);
                for j in 0..32 {
                    close(
                        rotated[row * stride + section * D + head * H + j],
                        expected[j],
                        "native rotation",
                        4e-4,
                        3e-5,
                    );
                }
            }
        }
    }
    let mut no_tape = DeviceBuffer::from_host(s, &values)?;
    r.attention.apply_rope(ApplyRopeArgs {
        stream: s,
        qkv: &mut no_tape,
        qkv_f16: None,
        row_count: N as u32,
        seq_len: S as u32,
        batch_size: B as u32,
        embedding_dim: D as u32,
        qkv_dim: stride as u32,
        head_count: HEADS as u32,
        head_dim: H as u32,
    })?;
    assert_eq!(rotated, no_tape.to_host_vec(s)?, "saved/unsaved rotation");
    drop(no_tape);
    drop(values);
    drop(rotated);
    let payload = DeviceBuffer::from_host(s, &[0x22u8; 64])?;
    let scales = DeviceBuffer::from_host(s, &csf::encode_host(&[0x38; 8], 1, 128))?;
    let global = DeviceBuffer::from_host(s, &[1f32])?;
    let qk = Nvfp4DeviceTensor::new(&payload, &scales, &global);
    let mut output = DeviceBuffer::<f32>::zeroed(s, N * D)?;
    let mut states = DeviceBuffer::<u16>::zeroed(s, N * D)?;
    let mut probs = DeviceBuffer::<u16>::zeroed(s, B * HEADS * S * S)?;
    let mut lse = DeviceBuffer::<f32>::zeroed(s, N * HEADS)?;
    let mut first = None;
    for saved in [true, false] {
        r.attention.causal_attention_tc(CausalAttentionTcArgs {
            stream: s,
            tc_module: &r.f16_tc_matmul,
            qkv: &qkv,
            qk_scale: qk,
            out: &mut output,
            qkv_f16: saved.then_some(&mut tape),
            attention_out_f16: saved.then_some(&mut states),
            forward_probs_f16: saved.then_some(&mut probs),
            kda_v_new: None,
            kda_akk_inv: None,
            kda_w: None,
            kda_aqk: None,
            log_sum_exp: &mut lse,
            scratch: scratch.forward_tc(),
            row_count: N as u32,
            seq_len: S as u32,
            batch_size: B as u32,
            embedding_dim: D as u32,
            qkv_dim: stride as u32,
            head_count: HEADS as u32,
            head_dim: H as u32,
            attention_window: S as u32,
            partial_key_offset: true,
            selective_attention: false,
            stable_mask_gamma: 0.,
        })?;
        let actual = output.to_host_vec(s)?;
        if let Some(ref original) = first {
            assert_eq!(original, &actual, "saved/unsaved full attention");
        }
        for b in 0..B {
            let off = (b * S + READOUT[b]) * D;
            let native = (0..H)
                .map(|j| f64::from(actual[off + j]) * incoming(j))
                .sum::<f64>();
            close(
                native as f32,
                oracle(b, None),
                "full attention readout",
                2e-4,
                0.02,
            );
        }
        first = Some(actual);
    }
    let mut upstream = vec![0f32; N * D];
    for b in 0..B {
        for head in 0..HEADS {
            for j in 0..H {
                upstream[(b * S + READOUT[b]) * D + head * H + j] = incoming(j) as f32;
            }
        }
    }
    let upstream = DeviceBuffer::from_host(s, &upstream)?;
    let mut dqkv = DeviceBuffer::<f32>::zeroed(s, N * stride)?;
    let mut amax = DeviceBuffer::<f32>::zeroed(s, N * 4)?;
    let mut d_scale = DeviceBuffer::<f32>::zeroed(s, 16)?;
    let mut checks = 0;
    let mut norm_coupled_witness = 0f32;
    let mut first_key_witness = 0f32;
    // The model's saved-probability path only reserves a compact dot workspace.
    // This operator also exercises generic uncached backward, whose DOut*V^T
    // output contains one FP32 entry for every query/key pair.
    let mut uncached_dot = DeviceBuffer::<f32>::zeroed(s, B * HEADS * S * S)?;
    for cached in [true, false] {
        let mut core = scratch.args();
        core.tc.dot = &mut uncached_dot;
        r.attention
            .causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
                stream: s,
                tc_module: &r.f16_tc_matmul,
                qkv: &tape,
                attention_out: &states,
                kda_v_new: None,
                kda_akk_inv: None,
                kda_w: None,
                kda_aqk: None,
                d_out: &upstream,
                qk_scale: qk,
                log_sum_exp: &lse,
                softmax_d: core.softmax_d,
                qk_norm_max: core.qk_norm_max,
                d_qkv: &mut dqkv,
                d_qkv_chunk_amax: &mut amax,
                d_qk_scale: &mut d_scale,
                accumulate_value_grad: false,
                scratch: core.tc,
                row_count: N as u32,
                seq_len: S as u32,
                batch_size: B as u32,
                embedding_dim: D as u32,
                qkv_dim: stride as u32,
                head_count: HEADS as u32,
                head_dim: H as u32,
                attention_window: S as u32,
                partial_key_offset: true,
                selective_attention: false,
                qk_norm_offset: 0,
                backward_mask_seed: 0,
                backward_tile_budget: 0.,
                reuse_forward_probs: cached,
                forward_probs_f16: cached.then_some(&probs),
            })?;
        let grads = dqkv.to_host_vec(s)?;
        assert!(grads.iter().all(|x| x.is_finite()));
        for b in 0..B {
            let q = READOUT[b];
            for token in q + 1..S {
                assert!(
                    grads[(b * S + token) * stride..(b * S + token) * stride + 3 * D]
                        .iter()
                        .all(|x| *x == 0.),
                    "future gradient b={b} token={token}"
                );
            }
            let mut tokens = vec![0, 1, q.saturating_sub(1), q];
            tokens.sort_unstable();
            tokens.dedup();
            for section in 0..3 {
                for &t in &tokens {
                    if section == 0 && t != q {
                        continue;
                    }
                    for j in [0, 31, 32, 63] {
                        let fd = |eps: f64| {
                            (oracle(b, Some((section, t, j, eps)))
                                - oracle(b, Some((section, t, j, -eps))))
                                / (2. * eps)
                        };
                        let expected = fd(1e-4);
                        let refined = fd(5e-5);
                        assert!(
                            (expected - refined).abs() <= 1e-7 + 0.002 * refined.abs(),
                            "independent FD self convergence"
                        );
                        let actual = grads[(b * S + t) * stride + section * D + j];
                        close(
                            actual,
                            refined,
                            &format!(
                                "raw adjoint cached={cached} b={b} token={t} section={section} j={j}"
                            ),
                            2e-6,
                            0.04,
                        );
                        if section == 1 && t == q && j >= 32 {
                            norm_coupled_witness = norm_coupled_witness.max(actual.abs());
                        }
                        if section == 1 && t == 0 {
                            first_key_witness = first_key_witness.max(actual.abs());
                        }
                        checks += 1;
                    }
                }
            }
        }
        let maxima = amax.to_host_vec(s)?;
        for row in 0..N {
            for section in 0..3 {
                let actual = grads[row * stride + section * D..row * stride + (section + 1) * D]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0f32, f32::max);
                assert_eq!(
                    actual.to_bits(),
                    maxima[row * 3 + section].to_bits(),
                    "actual gradient maximum"
                );
            }
        }
    }
    assert!(norm_coupled_witness > 1e-6 && first_key_witness > 1e-6);
    println!(
        "stationary_rope_operator_verified B={B} S={S} d={D} fd_checks={checks} saved_unsaved=bit_exact stationary_identity=bit_exact future_gradients=zero norm_coupled_witness={norm_coupled_witness:e} first_key_witness={first_key_witness:e} actual_gradient_amax=bit_exact optimizer_updates=0"
    );
    Ok(())
}
