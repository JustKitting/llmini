//! Independent clamped chunk objective, state trace and finite differences.
//! Host f64 is a mathematical oracle only; all returned gradients are CUDA.
use super::{B, D, HEADS, S, STRIDE, effective_value, incoming, value};
use crate::AppResult;
const H: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Family {
    Query,
    Key,
    Value,
    Decay,
    Beta,
}
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}
fn norm(raw: [f64; H], scale: f64) -> [f64; H] {
    let a = raw.map(silu);
    let denominator = (a.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt() + 1e-6;
    a.map(|v| scale * v / denominator)
}
fn exp(x: f64) -> f64 {
    x.clamp(-3., 3.).exp()
}
struct Trace {
    objective: f64,
    entry: Vec<[f64; H]>,
}

// Solve Vnew directly from the triangular chunk equations, rather than
// reproducing the CUDA inverse/backprojection code or its reverse equations.
fn objective(family: Family, token: usize, dim: usize, delta: f64) -> Trace {
    let qraw = std::array::from_fn(|j| {
        super::coupled::raw_query(j)
            + if family == Family::Query && token == S - 1 && dim == j {
                delta
            } else {
                0.
            }
    });
    let q = norm(qraw, 0.125);
    let key_raw = std::array::from_fn(super::coupled::raw_key);
    let key = norm(key_raw, 1.);
    let mut changed_raw = key_raw;
    changed_raw[dim] += delta;
    let changed_key = norm(changed_raw, 1.);
    let readout = (0..H)
        .map(|j| effective_value(f64::from(value(j))) * f64::from(incoming(j)))
        .sum::<f64>();
    let mut h = [0f64; H];
    let mut entries = Vec::with_capacity(S / H);
    let mut output = 0.;
    for chunk in 0..S / H {
        entries.push(h);
        let mut keys = [key; H];
        let mut pos = [[0f64; H]; H];
        let mut neg = [[0f64; H]; H];
        let mut cumulative = [[0f64; H]; H];
        let mut beta = [0f64; H];
        let mut v = [readout; H];
        let mut prefix = [0f64; H];
        for t in 0..H {
            let global = chunk * H + t;
            if family == Family::Key && global == token {
                keys[t] = changed_key;
            }
            beta[t] = sigmoid(
                -7. + if family == Family::Beta && global == token {
                    delta
                } else {
                    0.
                },
            );
            if family == Family::Value && global == token {
                v[t] += (effective_value(f64::from(value(dim)) + delta)
                    - effective_value(f64::from(value(dim))))
                    * f64::from(incoming(dim));
            }
            for j in 0..H {
                let raw = 6.
                    + if family == Family::Decay && global == token && j == dim {
                        delta
                    } else {
                        0.
                    };
                prefix[j] += -0.01 * (1. + raw.exp()).ln();
                cumulative[t][j] = prefix[j];
                pos[t][j] = beta[t] * keys[t][j] * exp(prefix[j]);
                neg[t][j] = keys[t][j] * exp(-prefix[j]);
            }
        }
        let mut vnew = [0f64; H];
        for t in 0..H {
            let mut result = beta[t] * v[t] - (0..H).map(|j| pos[t][j] * h[j]).sum::<f64>();
            for earlier in 0..t {
                let lower = (0..H).map(|j| pos[t][j] * neg[earlier][j]).sum::<f64>();
                result -= lower * vnew[earlier];
            }
            vnew[t] = result;
        }
        if chunk == S / H - 1 {
            let qg = std::array::from_fn::<_, H, _>(|j| q[j] * exp(cumulative[H - 1][j]));
            output = (0..H).map(|j| qg[j] * h[j]).sum::<f64>();
            for t in 0..H {
                output += (0..H).map(|j| qg[j] * neg[t][j]).sum::<f64>() * vnew[t];
            }
        }
        let mut next = [0f64; H];
        for j in 0..H {
            next[j] = exp(cumulative[H - 1][j]) * h[j];
            for t in 0..H {
                next[j] += keys[t][j] * exp(cumulative[H - 1][j] - cumulative[t][j]) * vnew[t];
            }
        }
        h = next;
    }
    Trace {
        objective: output,
        entry: entries,
    }
}
fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1. } else { -1. };
    let e = ((bits >> 10) & 31) as i32;
    let f = f64::from(bits & 1023);
    sign * if e == 0 {
        f * 2f64.powi(-24)
    } else {
        (1. + f / 1024.) * 2f64.powi(e - 15)
    }
}
fn close(actual: f64, expected: f64, context: impl std::fmt::Debug) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= 2e-8 + 0.03 * expected.abs(),
        "clamped recurrent {context:?} actual={actual:e} expected={expected:e} error={:e}",
        (actual - expected).abs()
    );
}
pub(super) fn verify(
    full: &[f32],
    detached: &[f32],
    wrapper: &[f32],
    forward: &[f32],
    states: &[u16],
) -> AppResult {
    assert!(
        states.iter().all(|v| v & 0x7c00 != 0x7c00),
        "nonfinite native recurrent state"
    );
    let reference = objective(Family::Value, 0, 0, 0.);
    let mut state_checks = 0;
    for batch in 0..B {
        for head in 0..HEADS {
            let native = (0..H)
                .map(|j| {
                    f64::from(forward[(batch * S + S - 1) * D + head * H + j])
                        * f64::from(incoming(j))
                })
                .sum::<f64>();
            close(
                native,
                reference.objective,
                ("terminal objective", batch, head),
            );
            for chunk in 0..S / H {
                for key in 0..H {
                    let base = (((batch * HEADS + head) * (S / H) + chunk) * H + key) * H;
                    let native = (0..H)
                        .map(|j| half(states[base + j]) * f64::from(incoming(j)))
                        .sum::<f64>();
                    close(
                        native,
                        reference.entry[chunk][key],
                        ("state", batch, head, chunk, key),
                    );
                    state_checks += 1;
                }
            }
        }
    }
    let mut fd_checks = 0;
    let mut native_checks = 0;
    for family in [
        Family::Query,
        Family::Key,
        Family::Value,
        Family::Decay,
        Family::Beta,
    ] {
        for token in [0, 63, 64, 1920, 1983, 1984, 2000, 2033, S - 1] {
            if family == Family::Query && token != S - 1 {
                continue;
            }
            for dim in [0, 17] {
                if family == Family::Beta && dim != 0 {
                    continue;
                }
                let derivative = |e: f64| {
                    (objective(family, token, dim, e).objective
                        - objective(family, token, dim, -e).objective)
                        / (2. * e)
                };
                let expected = derivative(1e-4);
                let half_step = derivative(5e-5);
                assert!(
                    (expected - half_step).abs() <= 2e-9 + 1e-3 * expected.abs(),
                    "unstable finite difference {family:?} token={token} dim={dim} e={expected:e} e_half={half_step:e}"
                );
                fd_checks += 1;
                let offset = match family {
                    Family::Query => 0,
                    Family::Key => D,
                    Family::Value => 2 * D,
                    Family::Decay => 3 * D,
                    Family::Beta => 4 * D,
                };
                for batch in 0..B {
                    for head in 0..HEADS {
                        let index = (batch * S + token) * STRIDE
                            + offset
                            + if family == Family::Beta {
                                head
                            } else {
                                head * H + dim
                            };
                        close(
                            f64::from(full[index]),
                            expected,
                            (family, token, dim, batch, head),
                        );
                        if token < S - H {
                            assert_eq!(detached[index], 0.);
                        }
                        if family == Family::Beta && token == 1983 {
                            assert!(
                                expected.abs() > 1e-8 && full[index] != 0.,
                                "missing cross-boundary clamped beta credit"
                            );
                        }
                        native_checks += 1;
                    }
                }
            }
        }
    }
    let selected = if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
        full
    } else {
        detached
    };
    let mismatch = wrapper
        .iter()
        .zip(selected)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits());
    assert!(
        mismatch.is_none(),
        "clamped recurrent wrapper mismatch={mismatch:?}"
    );
    println!(
        "native_full_KDA_clamped_state_adjoints_verified B={B} S={S} d={D} state_checks={state_checks} finite_difference_cases={fd_checks} perturbation_scales=2 native_Q_K_V_G_beta_checks={native_checks} full32_chunks=true optimizer_updates=0"
    );
    Ok(())
}
