//! Independent dense-head recurrent objective and all coupled adjoint families.
use super::{
    B, D, HEADS, S, STRIDE, effective_value, effective_value_grad, incoming, sigmoid, value,
};
use crate::AppResult;
pub(super) fn raw_query(j: usize) -> f64 {
    [1., 0.5, -0.25, 0.75][j % 4]
}
pub(super) fn raw_key(j: usize) -> f64 {
    [0.5, -0.5, 1., 0.25][j % 4]
}
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}
fn prime(x: f64) -> f64 {
    let s = sigmoid(x);
    s + x * s * (1. - s)
}
fn normalized(raw: &[f64], scale: f64) -> Vec<f64> {
    let a = raw.iter().copied().map(silu).collect::<Vec<_>>();
    let denom = (a.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt() + 1e-6;
    a.iter().map(|x| scale * x / denom).collect()
}
#[derive(Clone, Copy, Debug)]
enum Family {
    Query,
    Key,
    Value,
    Decay,
    Beta,
}

// Contract value columns with the fixed readout gradient first. This gives an
// exact64-dimensional serial recurrence for the independent scalar objective.
fn objective(family: Family, token: usize, dim: usize, delta: f64) -> f64 {
    let mut qraw = (0..64).map(raw_query).collect::<Vec<_>>();
    if matches!(family, Family::Query) && token == S - 1 {
        qraw[dim] += delta;
    }
    let q = normalized(&qraw, 0.125);
    let kraw = (0..64).map(raw_key).collect::<Vec<_>>();
    let k = normalized(&kraw, 1.);
    let mut changed_k = kraw.clone();
    changed_k[dim] += delta;
    let changed_k = normalized(&changed_k, 1.);
    let v = (0..64)
        .map(|j| effective_value(f64::from(value(j))) * f64::from(incoming(j)))
        .sum::<f64>();
    let alpha = (-0.01 * (1. + (-3f64).exp()).ln()).exp();
    let changed_alpha = (-0.01 * (1. + (-3. + delta).exp()).ln()).exp();
    let mut h = [0f64; 64];
    for t in 0..S {
        let kt = if matches!(family, Family::Key) && t == token {
            &changed_k
        } else {
            &k
        };
        let beta = sigmoid(
            -7. + if matches!(family, Family::Beta) && t == token {
                delta
            } else {
                0.
            },
        );
        for j in 0..64 {
            h[j] *= if matches!(family, Family::Decay) && t == token && j == dim {
                changed_alpha
            } else {
                alpha
            };
        }
        let prediction = (0..64).map(|j| kt[j] * h[j]).sum::<f64>();
        let vt = if matches!(family, Family::Value) && t == token {
            v + (effective_value(f64::from(value(dim)) + delta)
                - effective_value(f64::from(value(dim))))
                * f64::from(incoming(dim))
        } else {
            v
        };
        for j in 0..64 {
            h[j] += beta * kt[j] * (vt - prediction);
        }
    }
    q.iter().zip(h).map(|(q, h)| q * h).sum()
}
struct Oracle {
    u: Vec<f64>,
    q: Vec<f64>,
    beta: f64,
    alpha: f64,
    square: f64,
    qu: f64,
    carry: f64,
    readout: f64,
}
impl Oracle {
    fn new() -> Self {
        let u = normalized(&(0..64).map(raw_key).collect::<Vec<_>>(), 1.);
        let q = normalized(&(0..64).map(raw_query).collect::<Vec<_>>(), 0.125);
        let beta = sigmoid(-7.);
        let alpha = (-0.01 * (1. + (-3f64).exp()).ln()).exp();
        let square = u.iter().map(|x| x * x).sum::<f64>();
        let qu = u.iter().zip(&q).map(|(k, q)| k * q).sum::<f64>();
        let carry = alpha * (1. - beta * square);
        let readout = (0..64)
            .map(|j| effective_value(f64::from(value(j))) * f64::from(incoming(j)))
            .sum::<f64>();
        Self {
            u,
            q,
            beta,
            alpha,
            square,
            qu,
            carry,
            readout,
        }
    }
    fn scalar_state(&self, steps: usize) -> f64 {
        self.beta * (1. - self.carry.powi(steps as i32)) / (1. - self.carry)
    }
    fn derivative(&self, family: Family, token: usize, dim: usize) -> f64 {
        let lag = S - 1 - token;
        let p = (0..64)
            .map(|j| {
                self.alpha.powi(lag as i32)
                    * (self.q[j]
                        + self.u[j] * self.qu / self.square
                            * ((1. - self.beta * self.square).powi(lag as i32) - 1.))
            })
            .collect::<Vec<_>>();
        let pu = self.qu * self.carry.powi(lag as i32);
        let hp = self.scalar_state(token);
        let residual = (1. - self.alpha * self.square * hp) * self.readout;
        match family {
            Family::Value => {
                self.beta
                    * pu
                    * f64::from(incoming(dim))
                    * effective_value_grad(f64::from(value(dim)))
            }
            Family::Beta => pu * residual * self.beta * (1. - self.beta),
            Family::Decay => {
                (p[dim] - self.beta * pu * self.u[dim])
                    * (-0.01 * sigmoid(-3.) * self.alpha)
                    * self.u[dim]
                    * hp
                    * self.readout
            }
            Family::Key => {
                let grad = (0..64)
                    .map(|j| {
                        self.beta
                            * (p[j] * residual - self.u[j] * pu * self.alpha * hp * self.readout)
                    })
                    .collect::<Vec<_>>();
                norm_reverse(&grad, &(0..64).map(raw_key).collect::<Vec<_>>(), 1., dim)
            }
            Family::Query => {
                if token != S - 1 {
                    return 0.;
                }
                let grad = self
                    .u
                    .iter()
                    .map(|u| u * self.scalar_state(S) * self.readout)
                    .collect::<Vec<_>>();
                norm_reverse(
                    &grad,
                    &(0..64).map(raw_query).collect::<Vec<_>>(),
                    0.125,
                    dim,
                )
            }
        }
    }
}
fn norm_reverse(grad: &[f64], raw: &[f64], scale: f64, dim: usize) -> f64 {
    let a = raw.iter().copied().map(silu).collect::<Vec<_>>();
    let norm = (a.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
    let denom = norm + 1e-6;
    let dot = grad.iter().zip(&a).map(|(g, a)| g * a).sum::<f64>();
    scale * (grad[dim] / denom - a[dim] * dot / (norm * denom * denom)) * prime(raw[dim])
}
fn close(native: f32, expected: f64, family: Family, token: usize, dim: usize) {
    let error = (f64::from(native) - expected).abs();
    assert!(
        native.is_finite() && error <= 2e-8 + 0.03 * expected.abs(),
        "coupled recurrent {family:?} token={token} dim={dim} native={native:e} oracle={expected:e} error={error:e}"
    );
}
pub(super) fn verify_saturated(full: &[f32], detached: &[f32], actual: &[f32]) -> AppResult {
    let o = Oracle::new();
    let gstep = -0.01 * (1. + 6f64.exp()).ln();
    let exp_g = |token: usize| (gstep * (token + 1) as f64).clamp(-3., 3.).exp();
    let lambda = 1. - o.beta * o.square;
    let mut states = [0f64; 65];
    for t in 0..64 {
        states[t + 1] = lambda * states[t] + o.beta * o.readout / exp_g(t);
    }
    let mut checks = 0;
    for gradient in [full, detached, actual] {
        for batch in 0..B {
            for head in 0..HEADS {
                for token in [0, 1, 31, 47, 48, 49, 50, 59, 60, 61, 62, 63] {
                    let carry = lambda.powi((63 - token) as i32);
                    for dim in 0..64 {
                        let expected = o.qu * exp_g(63) * carry * o.beta / exp_g(token)
                            * f64::from(incoming(dim))
                            * effective_value_grad(f64::from(value(dim)));
                        close(
                            gradient[(batch * S + token) * STRIDE + 2 * D + head * 64 + dim],
                            expected,
                            Family::Value,
                            token,
                            dim,
                        );
                        checks += 1;
                    }
                    let expected = o.qu
                        * exp_g(63)
                        * carry
                        * (o.readout / exp_g(token) - o.square * states[token])
                        * o.beta
                        * (1. - o.beta);
                    close(
                        gradient[(batch * S + token) * STRIDE + 4 * D + head],
                        expected,
                        Family::Beta,
                        token,
                        0,
                    );
                    checks += 1;
                }
                let grad =
                    o.u.iter()
                        .map(|u| u * exp_g(63) * states[64])
                        .collect::<Vec<_>>();
                for dim in 0..64 {
                    let expected = norm_reverse(
                        &grad,
                        &(0..64).map(raw_query).collect::<Vec<_>>(),
                        0.125,
                        dim,
                    );
                    close(
                        gradient[(batch * S + 63) * STRIDE + head * 64 + dim],
                        expected,
                        Family::Query,
                        63,
                        dim,
                    );
                    checks += 1;
                }
            }
        }
    }
    println!(
        "native_saturated_KDA_independent_V_beta_Q_adjoints_verified native_checks={checks} all_modes=true optimizer_updates=0"
    );
    Ok(())
}
pub(super) fn verify(
    full: &[f32],
    detached: &[f32],
    actual: &[f32],
    forward: &[f32],
    states: &[u16],
    aqk: &[u16],
    w: &[u16],
    parts: &[Vec<f32>],
) -> AppResult {
    let o = Oracle::new();
    fn half(bits: u16) -> f64 {
        let exp = (bits >> 10) & 31;
        let frac = bits & 1023;
        let sign = if bits & 0x8000 == 0 { 1. } else { -1. };
        sign * if exp == 0 {
            f64::from(frac) * 2f64.powi(-24)
        } else {
            (1. + f64::from(frac) / 1024.) * 2f64.powi(i32::from(exp) - 15)
        }
    }
    let chunk = S / 64 - 1;
    let base = chunk * 4096;
    for dim in [0, 1, 2, 3, 17] {
        let h = (0..64)
            .map(|j| half(states[base + dim * 64 + j]) * f64::from(incoming(j)))
            .sum::<f64>();
        let ideal_h = o.u[dim] * o.scalar_state(chunk * 64) * o.readout;
        let qg = o.q[dim] * o.alpha.powi(64);
        let aqkw = (0..64)
            .map(|t| half(aqk[base + 63 * 64 + t]) * half(w[(chunk * 64 + t) * 64 + dim]))
            .sum::<f64>();
        let stored = -0.01 * sigmoid(-3.) * h * (qg - aqkw);
        let mut readout_chain = 0.;
        let mut inverse_chain = 0.;
        let mut qchain = 0.;
        let mut kchain = 0.;
        let glast = f64::from(parts[2][63 * 64 + dim]);
        for t in 0..64 {
            let i = t * 64 + dim;
            let gt = f64::from(parts[2][i]);
            let kval = f64::from(parts[1][i]) * (gt - glast).exp();
            let kneg = kval * (-gt).exp();
            let kpos = o.beta * kval * gt.exp();
            readout_chain +=
                f64::from(parts[3][i]) * f64::from(parts[0][i]) - f64::from(parts[4][i]) * kneg;
            inverse_chain += f64::from(parts[5][i]) * kpos;
            qchain += f64::from(parts[3][i]) * f64::from(parts[0][i]);
            kchain += f64::from(parts[4][i]) * kneg;
        }
        let native_pre =
            f64::from(full[chunk * 64 * STRIDE + 3 * D + dim]) / (-0.01 * sigmoid(-3.));
        eprintln!(
            "decay_readout_parts dim={dim} qchain={qchain:e} kchain={kchain:e} qg_last_native={} qg_last_ideal={qg:e} dQg_last={} H_expected={h:e} direct_Hqg={:e}",
            parts[0][63 * 64 + dim],
            parts[3][63 * 64 + dim],
            h * f64::from(parts[0][63 * 64 + dim])
        );
        eprintln!(
            "decay_chain_parts dim={dim} readout_chain={readout_chain:e} inverse_chain={inverse_chain:e} inferred_inverse_M_residual={:e} native_prefix={native_pre:e} gauge_reference={:e}",
            native_pre - readout_chain - inverse_chain,
            h * (qg - aqkw)
        );
        eprintln!(
            "decay_stored_operand_probe dim={dim} native={} ideal={} H_native={h:e} H_ideal={ideal_h:e} stored_product_reference={stored:e}",
            full[chunk * 64 * STRIDE + 3 * D + dim],
            o.derivative(Family::Decay, chunk * 64, dim)
        );
    }
    let final_scale = o.scalar_state(S) * o.qu;
    eprintln!(
        "coupled_forward_probe native={} ideal={}",
        forward[(S - 1) * D],
        final_scale * effective_value(f64::from(value(0)))
    );
    let eps = 1e-4;
    let mut fd_checks = 0;
    for family in [Family::Key, Family::Value, Family::Decay, Family::Beta] {
        for token in [63, 64, 128, S - 1] {
            for dim in [0, 17] {
                let analytic = o.derivative(family, token, dim);
                let fd = (objective(family, token, dim, eps) - objective(family, token, dim, -eps))
                    / (2. * eps);
                assert!(
                    (fd - analytic).abs() <= 2e-9 + analytic.abs() * 1e-5,
                    "independent oracle failed {family:?} token={token} dim={dim} analytic={analytic:e} finite_difference={fd:e}"
                );
                fd_checks += 1;
            }
        }
    }
    for dim in [0, 17] {
        let analytic = o.derivative(Family::Query, S - 1, dim);
        let fd = (objective(Family::Query, S - 1, dim, eps)
            - objective(Family::Query, S - 1, dim, -eps))
            / (2. * eps);
        assert!((fd - analytic).abs() <= 2e-9 + analytic.abs() * 1e-5);
        fd_checks += 1;
    }
    let probes = [
        0, 1, 62, 63, 64, 65, 126, 127, 128, 129, 1983, 1984, 2046, 2047,
    ];
    let mut checks = 0;
    for batch in 0..B {
        for head in 0..HEADS {
            for token in probes {
                for dim in 0..64 {
                    let start = (batch * S + token) * STRIDE + head * 64 + dim;
                    for (family, offset) in [
                        (Family::Key, D),
                        (Family::Value, 2 * D),
                        (Family::Decay, 3 * D),
                    ] {
                        close(
                            full[start + offset],
                            o.derivative(family, token, dim),
                            family,
                            token,
                            dim,
                        );
                        if token >= S - 64 {
                            close(
                                detached[start + offset],
                                o.derivative(family, token, dim),
                                family,
                                token,
                                dim,
                            );
                        }
                        if token < 1984 {
                            assert_eq!(detached[start + offset], 0.);
                        }
                        checks += 1;
                    }
                    if token == S - 1 {
                        close(
                            full[start],
                            o.derivative(Family::Query, token, dim),
                            Family::Query,
                            token,
                            dim,
                        );
                        close(
                            detached[start],
                            o.derivative(Family::Query, token, dim),
                            Family::Query,
                            token,
                            dim,
                        );
                        checks += 1;
                    }
                }
                let index = (batch * S + token) * STRIDE + 4 * D + head;
                close(
                    full[index],
                    o.derivative(Family::Beta, token, 0),
                    Family::Beta,
                    token,
                    0,
                );
                if token >= S - 64 {
                    close(
                        detached[index],
                        o.derivative(Family::Beta, token, 0),
                        Family::Beta,
                        token,
                        0,
                    );
                }
                checks += 1;
            }
        }
    }
    let final_scale = o.scalar_state(S) * o.qu;
    for j in 0..64 {
        close(
            forward[(S - 1) * D + j],
            final_scale * effective_value(f64::from(value(j))),
            Family::Value,
            S - 1,
            j,
        );
    }
    let expected_wrapper = if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
        full
    } else {
        detached
    };
    let mismatch = actual
        .iter()
        .zip(expected_wrapper)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits());
    assert!(
        mismatch.is_none(),
        "coupled real-wrapper mismatch={mismatch:?}"
    );
    println!(
        "native_full_KDA_coupled_adjoints_verified B={B} S={S} d={D} finite_difference_oracle_checks={fd_checks} native_Q_K_V_G_beta_checks={checks} full32_chunk_credit=true wrapper=bit_exact optimizer_updates=0"
    );
    Ok(())
}
