//! Native full-context credit assignment contract declared before dispatch change.
mod clamped;
mod coupled;
mod model;
mod normalization;
use super::runtime::Runtime;
use crate::AppResult;
use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionCoreBackwardArgs, AttentionCoreScratchBuffers, BlockForwardSaved,
    GPT2_BATCH_SIZE as B, GPT2_N_EMBD as D, GPT2_N_HEAD as HEADS, GPT2_SEQ_LEN as S,
    LayerNormSaved,
};
use rust_kernels_cuda::{
    attention::{CausalAttentionBackwardTcArgs, CausalAttentionTcArgs},
    nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor, csf},
};
const N: usize = B * S;
const STRIDE: usize = 8320;
fn value(j: usize) -> f32 {
    [0.125, -0.25, 0.5, -0.125][j % 4]
}
fn incoming(j: usize) -> f32 {
    [0.25, 0.125, -0.5, 0.0625][j % 4]
}
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
// The value-RMS graph supplies already-normalized effective values to KDA.
// Q/K retain their own SiLU map. This is an interface oracle,not CPU training.
fn effective_value(x: f64) -> f64 {
    if gpt2_nvfp4::GPT2_VALUE_RMS {
        x
    } else {
        x * sigmoid(x)
    }
}
fn effective_value_grad(x: f64) -> f64 {
    if gpt2_nvfp4::GPT2_VALUE_RMS {
        1.
    } else {
        let s = sigmoid(x);
        s + x * s * (1. - s)
    }
}
fn native_half_range(
    stream: &cuda_core::CudaStream,
    buffer: &DeviceBuffer<f32>,
    offset: usize,
) -> AppResult<Vec<u16>> {
    let mut out = vec![0u16; N * D];
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            out.as_mut_ptr(),
            buffer.cu_deviceptr() + offset as u64,
            N * D * 2,
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()?;
    Ok(out)
}

#[test]
#[ignore = "actual B8/S2048/d2048 KDA; root wrapper must preserve full32-chunk adjoint; zero updates"]
fn native_full_kda_wrapper_carries_credit_across_all_32_chunks() -> AppResult {
    run_credit_case(CreditCase::Sparse)
}

#[test]
#[ignore = "actual B8/S2048/d2048 full recurrent Q/K/V/decay/beta native adjoints and independent finite differences; zero updates"]
fn native_full_kda_coupled_recurrent_adjoints() -> AppResult {
    run_credit_case(CreditCase::Dense)
}

#[test]
#[ignore = "actual B8/S2048/d2048 uncached KDA reconstruction; same independent coupled adjoints, causality and chunk-amax contract; zero updates"]
fn native_full_kda_uncached_recurrent_adjoints() -> AppResult {
    run_credit_case(CreditCase::DenseUncached)
}

#[test]
#[ignore = "actual B8/S2048/d2048 native saturated forward plateau and full/detached/wrapper decay adjoints; zero updates"]
fn native_kda_saturated_decay_adjoint_matches_native_plateau() -> AppResult {
    run_credit_case(CreditCase::SaturatedPlateau)
}

#[test]
#[ignore = "actual B8/S2048/d2048 clamped32-chunk native state trace and Q/K/V/G/beta finite differences; zero updates"]
fn native_full_kda_clamped_recurrent_state_adjoints() -> AppResult {
    run_credit_case(CreditCase::SaturatedRecurrent)
}

#[test]
#[ignore = "same complete clamped32-chunk native adjoint gate with uncached reconstruction; zero updates"]
fn native_full_kda_uncached_clamped_recurrent_state_adjoints() -> AppResult {
    run_credit_case(CreditCase::SaturatedRecurrentUncached)
}

#[test]
#[ignore = "actual B8 KDA Q/K normalization epsilon derivative through cached/full/detached/wrapper CUDA; zero updates"]
fn native_kda_small_norm_exact_forward_derivative() -> AppResult {
    run_credit_case(CreditCase::SmallNorm)
}

#[test]
#[ignore = "same native original small-norm denominator and two-step STE derivative through uncached reconstruction; zero updates"]
fn native_kda_small_norm_uncached_exact_forward_derivative() -> AppResult {
    run_credit_case(CreditCase::SmallNormUncached)
}

#[derive(Clone, Copy)]
enum CreditCase {
    Sparse,
    Dense,
    DenseUncached,
    SaturatedPlateau,
    SaturatedRecurrent,
    SaturatedRecurrentUncached,
    SmallNorm,
    SmallNormUncached,
}

fn run_credit_case(case: CreditCase) -> AppResult {
    let dense_qk = !matches!(case, CreditCase::Sparse);
    let uncached = matches!(
        case,
        CreditCase::DenseUncached
            | CreditCase::SaturatedRecurrentUncached
            | CreditCase::SmallNormUncached
    );
    let small_norm = matches!(case, CreditCase::SmallNorm | CreditCase::SmallNormUncached);
    let saturated = matches!(
        case,
        CreditCase::SaturatedPlateau
            | CreditCase::SaturatedRecurrent
            | CreditCase::SaturatedRecurrentUncached
    );
    let runtime = Runtime::new()?;
    let stream = runtime.stream.as_ref();
    let mut scratch = AttentionCoreScratchBuffers::new(stream)?;
    let mut values = vec![0f32; N * STRIDE];
    let mut upstream = vec![0f32; N * D];
    for row in 0..N {
        for head in 0..HEADS {
            values[row * STRIDE + head * 64] = 1.;
            values[row * STRIDE + D + head * 64] = 1.;
            for j in 0..64 {
                if dense_qk {
                    values[row * STRIDE + head * 64 + j] = if small_norm {
                        normalization::raw_query(head)
                    } else {
                        coupled::raw_query(j) as f32
                    };
                    values[row * STRIDE + D + head * 64 + j] = if small_norm {
                        normalization::raw_key(head)
                    } else {
                        coupled::raw_key(j) as f32
                    };
                }
                values[row * STRIDE + 2 * D + head * 64 + j] =
                    if small_norm { 0.125 } else { value(j) };
                values[row * STRIDE + 3 * D + head * 64 + j] = if small_norm {
                    -100.0
                } else if saturated {
                    6.
                } else {
                    -3.
                };
                if row % S
                    == if small_norm {
                        0
                    } else if matches!(case, CreditCase::SaturatedPlateau) {
                        63
                    } else {
                        S - 1
                    }
                {
                    upstream[row * D + head * 64 + j] = if small_norm { 0.25 } else { incoming(j) };
                }
            }
            values[row * STRIDE + 4 * D + head] = if small_norm { 0.0 } else { -7. };
        }
    }
    let qkv = DeviceBuffer::from_host(stream, &values)?;
    drop(values);
    let dout = DeviceBuffer::from_host(stream, &upstream)?;
    drop(upstream);
    let mut tape = DeviceBuffer::<u16>::zeroed(stream, N * STRIDE)?;
    let mut states = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut lse = DeviceBuffer::<f32>::zeroed(stream, N * HEADS)?;
    let mut vnew = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
    let mut inverse = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
    let mut w = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
    let mut aqk = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
    let bytes = DeviceBuffer::<u8>::zeroed(stream, 64)?;
    let scales = DeviceBuffer::from_host(stream, &csf::encode_host(&[0x38; 8], 1, 128))?;
    let global = DeviceBuffer::from_host(stream, &[1f32])?;
    let scale = Nvfp4DeviceTensor::new(&bytes, &scales, &global);
    runtime.attention.kda_attention_tc(CausalAttentionTcArgs {
        stream,
        tc_module: &runtime.f16_tc_matmul,
        qkv: &qkv,
        qk_scale: scale,
        out: &mut output,
        qkv_f16: Some(&mut tape),
        attention_out_f16: Some(&mut states),
        forward_probs_f16: None,
        kda_v_new: Some(&mut vnew),
        kda_akk_inv: Some(&mut inverse),
        kda_w: Some(&mut w),
        kda_aqk: Some(&mut aqk),
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
    let forward = output.to_host_vec(stream)?;
    assert!(forward.iter().all(|x| x.is_finite()));
    let forward_kneg = if saturated {
        Some(native_half_range(
            stream,
            scratch.forward_tc().compact_out,
            0,
        )?)
    } else {
        None
    };
    let normalization_factors = if small_norm {
        Some((
            native_half_range(stream, scratch.forward_tc().q, 0)?,
            native_half_range(stream, scratch.forward_tc().compact_out, 0)?,
            vnew.to_host_vec(stream)?,
        ))
    } else {
        None
    };
    let dummy_nv = Nvfp4RowwiseDeviceTensor::new(&bytes, &scales, &global);
    let norm = LayerNormSaved {
        row_count: N as u32,
        residual: &tape,
        mean: &global,
        inv_std: &global,
    };
    // Only the truthful attention-core fields are consumed by this wrapper.
    let saved = BlockForwardSaved {
        batch_size: B as u32,
        seq_len: S as u32,
        row_count: N as u32,
        ln_1: norm,
        qkv_input_nvfp4: dummy_nv,
        qkv: &tape,
        attention_out: &states,
        headwise_gate_input: None,
        attention_probs: None,
        kda_v_new: if uncached { None } else { Some(&vnew) },
        kda_akk_inv: if uncached { None } else { Some(&inverse) },
        kda_w: if uncached { None } else { Some(&w) },
        kda_aqk: if uncached { None } else { Some(&aqk) },
        attention_log_sum_exp: &lse,
        c_proj_input_nvfp4: dummy_nv,
        ln_2: norm,
        mlp_up_input_nvfp4: dummy_nv,
        mlp_up: &tape,
        mlp_down_input_nvfp4: dummy_nv,
        mlp_route_masks: &DeviceBuffer::<u64>::zeroed(stream, 1)?,
    };
    let mut dscale = DeviceBuffer::<f32>::zeroed(stream, 128)?;
    let mut full = None;
    let mut truncated = None;
    let mut actual = None;
    for mode in 0..3 {
        let mut dqkv = DeviceBuffer::<f32>::zeroed(stream, N * STRIDE)?;
        let mut amax = DeviceBuffer::<f32>::zeroed(stream, 4 * N)?;
        let core = scratch.args();
        if mode == 2 {
            gpt2_nvfp4::causal_attention_backward(AttentionCoreBackwardArgs {
                block_index: 0,
                use_full_attention: false,
                reuse_forward_probs: false,
                stream,
                module: &runtime.attention,
                tc_module: &runtime.f16_tc_matmul,
                saved,
                d_attention_out: &dout,
                qk_scale: scale,
                d_qkv: &mut dqkv,
                d_qkv_chunk_amax: &mut amax,
                d_qk_scale: &mut dscale,
                accumulate_value_grad: false,
                scratch: core,
                backward_mask_seed: 0,
            })?;
        } else {
            let args = CausalAttentionBackwardTcArgs {
                stream,
                tc_module: &runtime.f16_tc_matmul,
                qkv: &tape,
                attention_out: &states,
                kda_v_new: if uncached { None } else { Some(&vnew) },
                kda_akk_inv: if uncached { None } else { Some(&inverse) },
                kda_w: if uncached { None } else { Some(&w) },
                kda_aqk: if uncached { None } else { Some(&aqk) },
                d_out: &dout,
                qk_scale: scale,
                log_sum_exp: &lse,
                softmax_d: core.softmax_d,
                qk_norm_max: core.qk_norm_max,
                d_qkv: &mut dqkv,
                d_qkv_chunk_amax: &mut amax,
                d_qk_scale: &mut dscale,
                accumulate_value_grad: false,
                scratch: core.tc,
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
                qk_norm_offset: 0,
                backward_mask_seed: 0,
                backward_tile_budget: 0.,
                reuse_forward_probs: false,
                forward_probs_f16: None,
            };
            if mode == 0 {
                runtime.attention.kda_attention_backward_tc(args)?;
            } else {
                runtime
                    .attention
                    .kda_attention_backward_tc_detached_chunk_state(args)?;
            }
        }
        if let Some(reference) = &forward_kneg {
            let derived = native_half_range(stream, scratch.args().tc.p, N * D * 2)?;
            let mismatch = reference
                .iter()
                .zip(&derived)
                .enumerate()
                .find(|(_, (a, b))| a != b);
            assert!(
                mismatch.is_none(),
                "backward Kneg differs from actual native forward factors mode={mode} first_mismatch={mismatch:?}"
            );
        }
        let g = dqkv.to_host_vec(stream)?;
        assert!(g.iter().all(|x| x.is_finite()));
        let chunks = amax.to_host_vec(stream)?;
        for batch in 0..B {
            for head in 0..HEADS {
                for chunk in 0..S / 64 {
                    let mut expected = 0f32;
                    for t in 0..64 {
                        let row = batch * S + chunk * 64 + t;
                        for section in 0..4 {
                            for dim in 0..64 {
                                expected = expected
                                    .max(g[row * STRIDE + section * D + head * 64 + dim].abs());
                            }
                        }
                        expected = expected.max(g[row * STRIDE + 4 * D + head].abs());
                    }
                    let start = ((batch * HEADS + head) * (S / 64) + chunk) * 8;
                    assert_eq!(
                        chunks[start].to_bits(),
                        expected.to_bits(),
                        "actual post-adjoint chunk amax mode={mode} batch={batch} head={head} chunk={chunk}"
                    );
                    assert!(chunks[start + 1..start + 8].iter().all(|x| *x == 0.));
                }
            }
        }
        match mode {
            0 => full = Some(g),
            1 => truncated = Some(g),
            _ => actual = Some(g),
        }
    }
    assert!(
        forward
            .iter()
            .zip(output.to_host_vec(stream)?)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "backward mode mutated forward output"
    );
    let full = full.unwrap();
    let truncated = truncated.unwrap();
    let actual = actual.unwrap();
    if let Some((qg, kneg, residual)) = normalization_factors {
        return normalization::verify(&full, &truncated, &actual, &qg, &kneg, &residual, uncached);
    }
    if matches!(
        case,
        CreditCase::SaturatedRecurrent | CreditCase::SaturatedRecurrentUncached
    ) {
        return clamped::verify(
            &full,
            &truncated,
            &actual,
            &forward,
            &states.to_host_vec(stream)?,
        );
    }
    if saturated {
        let mut perturbed = qkv.to_host_vec(stream)?;
        for dim in [0, 17] {
            for delta in [-0.125f32, 0.125] {
                for batch in 0..B {
                    for head in 0..HEADS {
                        perturbed[(batch * S + 60) * STRIDE + 3 * D + head * 64 + dim] = 6. + delta;
                    }
                }
                let probe = DeviceBuffer::from_host(stream, &perturbed)?;
                runtime.attention.kda_attention_tc(CausalAttentionTcArgs {
                    stream,
                    tc_module: &runtime.f16_tc_matmul,
                    qkv: &probe,
                    qk_scale: scale,
                    out: &mut output,
                    qkv_f16: Some(&mut tape),
                    attention_out_f16: Some(&mut states),
                    forward_probs_f16: None,
                    kda_v_new: Some(&mut vnew),
                    kda_akk_inv: Some(&mut inverse),
                    kda_w: Some(&mut w),
                    kda_aqk: Some(&mut aqk),
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
                let observed = output.to_host_vec(stream)?;
                for batch in 0..B {
                    for j in 0..D {
                        let i = (batch * S + 63) * D + j;
                        assert_eq!(
                            forward[i].to_bits(),
                            observed[i].to_bits(),
                            "native saturated forward plateau dim={dim} delta={delta} batch={batch} feature={j}"
                        );
                    }
                }
                for batch in 0..B {
                    for head in 0..HEADS {
                        perturbed[(batch * S + 60) * STRIDE + 3 * D + head * 64 + dim] = 6.;
                    }
                }
            }
        }
        println!(
            "native_saturated_KDA_forward_plateau_verified B={B} S={S} d={D} objective_token=63 perturbed_token=60 features=0,17 signed_perturbation=0.125 bit_exact=true optimizer_updates=0"
        );
        for (mode, gradient) in [
            ("full", &full),
            ("detached", &truncated),
            ("wrapper", &actual),
        ] {
            for batch in 0..B {
                for head in 0..HEADS {
                    for token in 60..64 {
                        for dim in 0..64 {
                            let index = (batch * S + token) * STRIDE + 3 * D + head * 64 + dim;
                            assert_eq!(
                                gradient[index], 0.,
                                "raw decay adjoint through proven native plateau mode={mode} batch={batch} head={head} token={token} dim={dim}"
                            );
                        }
                    }
                }
            }
        }
        println!("native_saturated_KDA_decay_adjoint_verified all_modes=true optimizer_updates=0");
        coupled::verify_saturated(&full, &truncated, &actual)?;
        return Ok(());
    }
    if dense_qk {
        fn chunk(
            stream: &cuda_core::CudaStream,
            buffer: &DeviceBuffer<f32>,
        ) -> AppResult<Vec<f32>> {
            let mut out = vec![0f32; 4096];
            unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    out.as_mut_ptr(),
                    buffer.cu_deviceptr() + ((S / 64 - 1) * 4096 * 4) as u64,
                    4096 * 4,
                    stream.cu_stream(),
                )?;
            }
            stream.synchronize()?;
            Ok(out)
        }
        let core = scratch.args();
        let parts = [
            chunk(stream, core.tc.q_f32)?,
            chunk(stream, core.tc.k_f32)?,
            chunk(stream, core.tc.g_f32)?,
            chunk(stream, core.tc.d_q)?,
            chunk(stream, core.tc.kda_d_q)?,
            chunk(stream, core.tc.kda_d_g)?,
        ];
        return coupled::verify(
            &full,
            &truncated,
            &actual,
            &forward,
            &states.to_host_vec(stream)?,
            &aqk.to_host_vec(stream)?,
            &w.to_host_vec(stream)?,
            &parts,
        );
    }
    let beta = sigmoid(-7.);
    let q = sigmoid(1.);
    let k = q / ((q * q + 1e-6).sqrt() + 1e-6);
    let decay = (-0.01 * (1. + (-3f64).exp()).ln()).exp();
    let carry = decay * (1. - beta * k * k);
    let coefficient = 0.125 * k * k * beta;
    let probes = [
        0, 1, 62, 63, 64, 65, 126, 127, 128, 129, 1983, 1984, 2046, 2047,
    ];
    let mut checked = 0;
    for batch in 0..B {
        for head in 0..HEADS {
            for token in probes {
                for j in 0..64 {
                    let x = f64::from(value(j));
                    let deriv = effective_value_grad(x);
                    let expected = f64::from(incoming(j))
                        * coefficient
                        * carry.powi((S - 1 - token) as i32)
                        * deriv;
                    let index = (batch * S + token) * STRIDE + 2 * D + head * 64 + j;
                    let error = (f64::from(full[index]) - expected).abs();
                    assert!(
                        full[index] != 0. && error <= 2e-8 + 0.03 * expected.abs(),
                        "full recurrent raw-V oracle batch={batch} head={head} token={token} j={j} native={} expected={expected:e} error={error:e}",
                        full[index]
                    );
                    if token < 1984 {
                        assert_eq!(truncated[index], 0., "truncated predecessor derivative");
                    }
                    checked += 1;
                }
            }
        }
    }
    let index = 2 * D;
    println!(
        "native_KDA_full_state_counterexample B={B} S={S} d={D} full_V0={} detached_V0={} wrapper_V0={} oracle_checks={checked} forward_unchanged=true optimizer_updates=0",
        full[index], truncated[index], actual[index]
    );
    let expected_wrapper = if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
        &full
    } else {
        &truncated
    };
    let mismatch = actual
        .iter()
        .zip(expected_wrapper)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits());
    assert!(
        mismatch.is_none(),
        "real trainer wrapper violates its compiled recurrent-credit contract; first mismatch={mismatch:?}"
    );
    Ok(())
}
