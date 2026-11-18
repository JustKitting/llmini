//! Native diagnostic and factorization counterexample, declared before implementation.
use crate::{AppResult, training::runtime::Runtime};
use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionCoreScratchBuffers, GPT2_BATCH_SIZE as B, GPT2_N_EMBD as D, GPT2_N_HEAD as HEADS,
    GPT2_SEQ_LEN as S,
};
const N: usize = B * S;
const STRIDE: usize = 8320;
const RAW: [f32; 8] = [-3., 0., 6., 32., 6.001, -10., -0.5, 32.001];
const HALF: [u16; 8] = [0xc200, 0, 0x4600, 0x5000, 0x4600, 0xc900, 0xb800, 0x5000];
fn clamp(x: f64) -> f64 {
    x.clamp(-3., 3.)
}
fn oracle(raw: f32) -> ([u32; 8], [f64; 4]) {
    let delta = -0.01 * (1. + f64::from(raw).exp()).ln();
    let prefix = std::array::from_fn::<_, 64, _>(|t| (t + 1) as f64 * delta);
    let mut c = [0u32; 8];
    let mut mismatch = 0f64;
    for (t, &g) in prefix.iter().enumerate() {
        c[0] += u32::from(g < -3.);
        c[1] += u32::from(prefix[63] - g < -3.);
        let gap = (clamp(prefix[63]) - clamp(g) - clamp(prefix[63] - g)).abs();
        c[6] += u32::from(gap != 0.);
        mismatch = mismatch.max(gap);
        for &earlier in &prefix[..=t] {
            c[5] += u32::from((clamp(g) - clamp(earlier) - (g - earlier)).abs() > 1e-12);
        }
    }
    c[2] = u32::from(prefix[63] < -3.);
    (
        c,
        [
            prefix[63],
            (-3. - prefix[63]).max(0.),
            clamp(prefix[63]) - clamp(prefix[0]) - (prefix[63] - prefix[0]),
            mismatch,
        ],
    )
}
#[test]
#[ignore = "full B8/S2048/d2048 CUDA decay counters, actual FP32 inputs, tape binding, nonfinite/shape checks; zero updates"]
fn native_kda_decay_audit_counts_and_original_FP32_binding() -> AppResult {
    let r = Runtime::new()?;
    let s = r.stream.as_ref();
    let audit = r.kda_decay_audit_module()?;
    let mut raw = vec![0f32; N * STRIDE];
    let mut half = vec![0u16; N * STRIDE];
    for row in 0..N {
        for j in 0..D {
            raw[row * STRIDE + 3 * D + j] = RAW[j % 8];
            half[row * STRIDE + 3 * D + j] = HALF[j % 8];
        }
    }
    let mut raw_device = DeviceBuffer::from_host(s, &raw)?;
    let mut tape = DeviceBuffer::from_host(s, &half)?;
    let data = audit.inspect(
        s,
        &raw_device,
        &tape,
        N as u32,
        S as u32,
        D as u32,
        STRIDE as u32,
    )?;
    assert_eq!(data.cells, N / 64 * D);
    let expected = RAW.map(oracle);
    for cell in 0..data.cells {
        let pattern = cell % 64 % 8;
        assert_eq!(
            &data.counts[cell * 8..cell * 8 + 8],
            &expected[pattern].0,
            "cell={cell}"
        );
        for j in 0..4 {
            let actual = f64::from(data.extrema[cell * 4 + j]);
            let reference = expected[pattern].1[j];
            assert!(
                (actual - reference).abs() <= 3e-5 + 2e-6 * reference.abs(),
                "cell={cell} statistic={j} actual={actual:e} expected={reference:e}"
            );
        }
    }
    let corrupt = (N - 1) * STRIDE + 4 * D - 1;
    half[corrupt] ^= 1;
    tape = DeviceBuffer::from_host(s, &half)?;
    let corrupted = audit.inspect(
        s,
        &raw_device,
        &tape,
        N as u32,
        S as u32,
        D as u32,
        STRIDE as u32,
    )?;
    assert_eq!(
        corrupted
            .counts
            .chunks_exact(8)
            .map(|v| u64::from(v[3]))
            .sum::<u64>(),
        1
    );
    assert_eq!(
        corrupted
            .counts
            .chunks_exact(8)
            .map(|v| u64::from(v[4]))
            .sum::<u64>(),
        0
    );
    raw[3 * D] = f32::INFINITY;
    half[3 * D] = 0x7c00;
    half[corrupt] ^= 1;
    raw_device = DeviceBuffer::from_host(s, &raw)?;
    tape = DeviceBuffer::from_host(s, &half)?;
    let invalid = audit.inspect(
        s,
        &raw_device,
        &tape,
        N as u32,
        S as u32,
        D as u32,
        STRIDE as u32,
    )?;
    assert_eq!(
        invalid
            .counts
            .chunks_exact(8)
            .map(|v| u64::from(v[4]))
            .sum::<u64>(),
        1
    );
    assert_eq!(
        invalid
            .counts
            .chunks_exact(8)
            .map(|v| u64::from(v[7]))
            .sum::<u64>(),
        64
    );
    assert!(
        audit
            .inspect(s, &raw_device, &tape, N as u32, 63, D as u32, STRIDE as u32)
            .is_err()
    );
    assert!(
        audit
            .inspect(
                s,
                &raw_device,
                &tape,
                (N + S) as u32,
                S as u32,
                D as u32,
                STRIDE as u32
            )
            .is_err()
    );
    println!(
        "native_KDA_decay_audit_verified B={B} S={S} d={D} cells={} patterns=8 FP32_source=true corrupted_tape_mismatches=1 nonfinite_raw_detected=1 invalid_prefixes=64 optimizer_updates=0",
        data.cells
    );
    Ok(())
}

#[test]
#[ignore = "real full-size native KDA factorization witness against independent recurrence; finite output and zero updates"]
fn native_kda_clamped_factorization_differs_from_recurrent_reference() -> AppResult {
    use rust_kernels_cuda::{
        attention::CausalAttentionTcArgs,
        nvfp4::{Nvfp4DeviceTensor, csf},
    };
    let r = Runtime::new()?;
    let s = r.stream.as_ref();
    let mut scratch = AttentionCoreScratchBuffers::new(s)?;
    let mut raw = vec![0f32; N * STRIDE];
    for row in 0..N {
        for head in 0..HEADS {
            raw[row * STRIDE + head * 64] = 1.;
            if row % S == 0 {
                raw[row * STRIDE + D + head * 64] = 1.;
                raw[row * STRIDE + 2 * D + head * 64] = 1.;
            }
            for j in 0..64 {
                raw[row * STRIDE + 3 * D + head * 64 + j] = if row % S < 64 { 6. } else { -100. };
            }
        }
    }
    let qkv = DeviceBuffer::from_host(s, &raw)?;
    let payload = DeviceBuffer::from_host(s, &[0x22u8; 64])?;
    let scales = DeviceBuffer::from_host(s, &csf::encode_host(&[0x38; 8], 1, 128))?;
    let global = DeviceBuffer::from_host(s, &[1f32])?;
    let mut out = DeviceBuffer::<f32>::zeroed(s, N * D)?;
    let mut lse = DeviceBuffer::<f32>::zeroed(s, N * HEADS)?;
    r.attention.kda_attention_tc(CausalAttentionTcArgs {
        stream: s,
        tc_module: &r.f16_tc_matmul,
        qkv: &qkv,
        qk_scale: Nvfp4DeviceTensor::new(&payload, &scales, &global),
        out: &mut out,
        qkv_f16: None,
        attention_out_f16: None,
        forward_probs_f16: None,
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
        qkv_dim: STRIDE as u32,
        head_count: HEADS as u32,
        head_dim: 64,
        attention_window: S as u32,
        partial_key_offset: false,
        selective_attention: false,
        stable_mask_gamma: 0.,
    })?;
    let actual = out.to_host_vec(s)?;
    assert!(actual.iter().all(|x| x.is_finite()));
    let activation = 1. / (1. + (-1f64).exp());
    let norm = (activation * activation + 1e-6).sqrt() + 1e-6;
    let amplitude = 0.125 * (activation / norm).powi(2) * activation * 0.5;
    let delta = -0.01 * (1. + 6f64.exp()).ln();
    let capped = amplitude * (-3. - delta).exp();
    let recurrent = amplitude * (63. * delta).exp();
    let next = amplitude * (-3f64).exp();
    for b in 0..B {
        for head in 0..HEADS {
            let terminal = f64::from(actual[(b * S + 63) * D + head * 64]);
            let after = f64::from(actual[(b * S + 64) * D + head * 64]);
            assert!(
                (terminal - capped).abs() <= 2e-5 + 0.02 * capped.abs(),
                "capped native={terminal:e} expected={capped:e}"
            );
            assert!(
                (after - next).abs() <= 2e-5 + 0.02 * next.abs(),
                "state native={after:e} expected={next:e}"
            );
            assert!((terminal - recurrent).abs() > 0.2 * recurrent.abs());
            assert!((terminal - after).abs() > 0.03 * after.abs());
        }
    }
    println!(
        "native_KDA_factor_clamp_witness_verified B={B} S={S} d={D} capped_reference={capped:e} recurrent_reference={recurrent:e} state_reference={next:e} unclamped_ratio={} finite=true optimizer_updates=0",
        capped / recurrent
    );
    Ok(())
}
