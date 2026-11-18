//! Pre-repair native reproducer for the reached shared Q/K normalizer. The
//! forward denominator is sqrt(sum(SiLU(raw)^2)+eps)+eps, whose radial VJP
//! denominator is norm*(norm+eps)^2. Test actual quantized factors under STE.
use super::*;
pub(super) fn raw_query(head: usize) -> f32 {
    [1.0 / 1024.0, 1.0 / 4096.0, -1.0 / 1024.0, 0.0][head % 4]
}
pub(super) fn raw_key(head: usize) -> f32 {
    [1.0 / 1024.0, -1.0 / 1024.0, 1.0 / 4096.0, 1.0 / 1024.0][head % 4]
}
fn half(x: u16) -> f64 {
    let sign = if x & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((x >> 10) & 31) as i32;
    let fraction = (x & 1023) as f64;
    if exponent == 0 {
        sign * fraction * 2.0f64.powi(-24)
    } else {
        sign * (1.0 + fraction / 1024.0) * 2.0f64.powi(exponent - 15)
    }
}
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}
fn silu_grad(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}
fn normalized(raw: &[f64; 64], scale: f64) -> [f64; 64] {
    let a = raw.map(silu);
    let n = (a.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
    a.map(|x| scale * x / (n + 1e-6))
}
pub(super) fn verify(
    full: &[f32],
    detached: &[f32],
    wrapper: &[f32],
    qg: &[u16],
    kneg: &[u16],
    residual: &[u16],
    uncached: bool,
) -> AppResult {
    let mut max_native_error = 0.0f64;
    let mut max_fd_error = 0.0f64;
    for batch in 0..B {
        for head in 0..HEADS {
            let compact = (batch * HEADS + head) * S * 64;
            let qraw = f64::from(raw_query(head));
            let kraw = f64::from(raw_key(head));
            let qa = silu(qraw);
            let ka = silu(kraw);
            let qn = (64.0 * qa * qa + 1e-6).sqrt();
            let kn = (64.0 * ka * ka + 1e-6).sqrt();
            let qd = qn + 1e-6;
            let kd = kn + 1e-6;
            let upstream = (0..64)
                .map(|v| 0.25 * half(residual[compact + v]))
                .sum::<f64>();
            let dq = (0..64)
                .map(|h| upstream * half(kneg[compact + h]))
                .collect::<Vec<_>>();
            let dk = (0..64)
                .map(|h| upstream * half(qg[compact + h]))
                .collect::<Vec<_>>();
            let qdot = dq.iter().sum::<f64>() * qa;
            let kdot = dk.iter().sum::<f64>() * ka;
            for dim in 0..64 {
                let expected_q =
                    0.125 * (dq[dim] / qd - qa * qdot / (qn * qd * qd)) * silu_grad(qraw);
                let expected_k = (dk[dim] / kd - ka * kdot / (kn * kd * kd)) * silu_grad(kraw);
                for (mode, gradient) in
                    [("full", full), ("detached", detached), ("wrapper", wrapper)]
                {
                    for (plane, expected) in [(0, expected_q), (1, expected_k)] {
                        let actual =
                            f64::from(gradient[(batch * S) * STRIDE + plane * D + head * 64 + dim]);
                        let error = (actual - expected).abs();
                        max_native_error = max_native_error.max(error);
                        assert!(
                            actual.is_finite() && error <= 5e-4 + 0.01 * expected.abs(),
                            "KDA_qk_norm_exact_denominator uncached={uncached} mode={mode} batch={batch} head={head} plane={plane} dim={dim} actual={actual} expected={expected} error={error}"
                        );
                    }
                }
            }
            if batch == 0 && head < 4 {
                let base_q = [qraw; 64];
                let base_k = [kraw; 64];
                let norm_q = normalized(&base_q, 0.125);
                let norm_k = normalized(&base_k, 1.0);
                for plane in 0..2 {
                    let dim = 17;
                    let target = if plane == 0 {
                        0.125 * (dq[dim] / qd - qa * qdot / (qn * qd * qd)) * silu_grad(qraw)
                    } else {
                        (dk[dim] / kd - ka * kdot / (kn * kd * kd)) * silu_grad(kraw)
                    };
                    for step in [2.0f64.powi(-18), 2.0f64.powi(-19)] {
                        let objective = |delta: f64| {
                            let mut qr = base_q;
                            let mut kr = base_k;
                            if plane == 0 {
                                qr[dim] += delta;
                            } else {
                                kr[dim] += delta;
                            }
                            let q = normalized(&qr, 0.125);
                            let k = normalized(&kr, 1.0);
                            (0..64)
                                .map(|h| {
                                    upstream
                                        * (q[h] + half(qg[compact + h]) - norm_q[h])
                                        * (k[h] + half(kneg[compact + h]) - norm_k[h])
                                })
                                .sum::<f64>()
                        };
                        let fd = (objective(step) - objective(-step)) / (2.0 * step);
                        let error = (fd - target).abs();
                        max_fd_error = max_fd_error.max(error);
                        assert!(
                            error <= 2e-5 + 1e-4 * target.abs(),
                            "independent native-factor STE finite difference head={head} plane={plane} fd={fd} expected={target} error={error}"
                        );
                    }
                }
            }
        }
    }
    println!(
        "native_KDA_small_norm_derivative_verified B={B} S={S} H={HEADS} uncached={uncached} full_detached_wrapper=true first_token_uniform_raw_cases=4 independent_native_factor_STE=true step_sizes=2 max_native_error={max_native_error:e} max_FD_error={max_fd_error:e} amax=actual_completed_gradient optimizer_updates=0"
    );
    Ok(())
}
