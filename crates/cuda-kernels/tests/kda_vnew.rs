//! Immutable execution goldens captured before changing KDA tape storage.
//! Recurrent chunks, full training dimensions, padding, value-residual
//! accumulation and uncached reconstruction must survive native storage and
//! producer/consumer fusion without changing tensor-core operand bits.
use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{
    AttentionModule, CausalAttentionBackwardTcArgs, CausalAttentionTcArgs,
};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use std::error::Error;
mod common;
#[path = "causal_attention_backward_tc/scratch.rs"]
mod scratch;
#[path = "causal_attention_backward_tc/shape.rs"]
mod shape;

#[test]
#[ignore = "requires generated sm_120a PTX"]
fn kda_tapes_preserve_cached_goldens_and_reconstruction_gradients() -> Result<(), Box<dyn Error>> {
    for (batch, seq, heads, cached, accumulate) in [
        (1, 128, 2, true, false),
        (2, 192, 3, true, true),
        (1, 128, 2, false, false),
        (4, 2048, 32, true, false),
        (2, 192, 3, false, true),
    ] {
        check(batch, seq, heads, cached, accumulate, false)?;
    }
    check(4, 2048, 32, true, false, true)?;
    Ok(())
}

fn hash(words: impl IntoIterator<Item = u32>) -> u64 {
    words
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .fold(0xcbf29ce484222325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        })
}
fn check(
    batch: usize,
    seq: usize,
    heads: usize,
    cached: bool,
    accumulate: bool,
    detach: bool,
) -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let attn = AttentionModule::from_module(ptx.clone())?;
    let tc = F16TcMatmulModule::from_module(ptx)?;
    let width = heads * 64;
    let rows = batch * seq;
    let qkv_dim = (4 * width + heads).div_ceil(128) * 128;
    let compact = rows * width;
    let qkv_values: Vec<f32> = (0..rows * qkv_dim)
        .map(|i| {
            let col = i % qkv_dim;
            if col >= 3 * width && col < 4 * width {
                -3.0 + ((i * 7 % 13) as f32 - 6.0) * 0.03125
            } else {
                ((i * 73 % 257) as f32 - 128.0) * 0.00390625
            }
        })
        .collect();
    let dout_values: Vec<f32> = (0..compact)
        .map(|i| ((i * 37 % 97) as f32 - 48.0) * 0.0078125)
        .collect();
    let qkv = DeviceBuffer::from_host(&stream, &qkv_values)?;
    let mut qkv_tape = DeviceBuffer::<u16>::zeroed(&stream, qkv_values.len())?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, compact)?;
    let mut states = DeviceBuffer::<u16>::zeroed(&stream, compact)?;
    let mut vnew = DeviceBuffer::<u16>::zeroed(&stream, compact)?;
    let mut inverse = DeviceBuffer::<u16>::zeroed(&stream, compact)?;
    let mut w = DeviceBuffer::<u16>::zeroed(&stream, compact)?;
    let mut aqk = DeviceBuffer::<u16>::zeroed(&stream, compact)?;
    let mut lse = DeviceBuffer::<f32>::zeroed(&stream, rows * heads)?;
    let mut fs = scratch::TcForwardScratchBuffers::new(&stream, batch * heads, seq, 64)?;
    let qs = DeviceBuffer::<u8>::zeroed(&stream, 128)?;
    let qb = DeviceBuffer::<u8>::zeroed(&stream, 8)?;
    let qg = DeviceBuffer::from_host(&stream, &[1.0f32])?;
    let scale = Nvfp4DeviceTensor::new(&qb, &qs, &qg);
    let context_references = std::sync::Arc::strong_count(stream.context());
    let mut first_forward = None;
    for save_tape in [true, false] {
        attn.kda_attention_tc(CausalAttentionTcArgs {
            stream: &stream,
            tc_module: &tc,
            qkv: &qkv,
            qk_scale: scale,
            out: &mut out,
            qkv_f16: save_tape.then_some(&mut qkv_tape),
            attention_out_f16: Some(&mut states),
            forward_probs_f16: None,
            kda_v_new: cached.then_some(&mut vnew),
            kda_akk_inv: cached.then_some(&mut inverse),
            kda_w: cached.then_some(&mut w),
            kda_aqk: cached.then_some(&mut aqk),
            log_sum_exp: &mut lse,
            scratch: fs.args(),
            row_count: rows as u32,
            seq_len: seq as u32,
            batch_size: batch as u32,
            embedding_dim: width as u32,
            qkv_dim: qkv_dim as u32,
            head_count: heads as u32,
            head_dim: 64,
            attention_window: seq as u32,
            partial_key_offset: false,
            selective_attention: false,
            stable_mask_gamma: 0.0,
        })?;
        let actual = (
            hash(out.to_host_vec(&stream)?.iter().map(|x| x.to_bits())),
            hash(states.to_host_vec(&stream)?.into_iter().map(u32::from)),
        );
        if let Some(expected) = first_forward {
            assert_eq!(
                actual, expected,
                "no-QKV-tape native preparation changed forward math"
            );
        } else {
            first_forward = Some(actual);
        }
    }
    assert_eq!(
        std::sync::Arc::strong_count(stream.context()),
        context_references,
        "borrowed scratch must neither retain nor release context references"
    );
    let dout = DeviceBuffer::from_host(&stream, &dout_values)?;
    let initial: Vec<f32> = (0..rows * qkv_dim)
        .map(|i| {
            if accumulate && i % qkv_dim >= 2 * width && i % qkv_dim < 3 * width {
                0.125
            } else {
                0.0
            }
        })
        .collect();
    let mut dqkv = DeviceBuffer::from_host(&stream, &initial)?;
    let mut amax = DeviceBuffer::<f32>::zeroed(&stream, rows * heads)?;
    let mut qkmax = DeviceBuffer::<f32>::zeroed(&stream, 2 * heads)?;
    let mut dscale = DeviceBuffer::<f32>::zeroed(&stream, 16)?;
    let mut beta = DeviceBuffer::<f32>::zeroed(&stream, rows * heads)?;
    let mut bs = scratch::TcScratchBuffers::new_for_shape(&stream, batch * heads, seq, 64)?;
    let backward_context_references = std::sync::Arc::strong_count(stream.context());
    let backward = if detach {
        AttentionModule::kda_attention_backward_tc_detached_chunk_state
    } else {
        AttentionModule::kda_attention_backward_tc
    };
    let count = backward(
        &attn,
        CausalAttentionBackwardTcArgs {
            stream: &stream,
            tc_module: &tc,
            qkv: &qkv_tape,
            attention_out: &states,
            kda_v_new: cached.then_some(&vnew),
            kda_akk_inv: cached.then_some(&inverse),
            kda_w: cached.then_some(&w),
            kda_aqk: cached.then_some(&aqk),
            d_out: &dout,
            qk_scale: scale,
            log_sum_exp: &lse,
            softmax_d: &mut beta,
            qk_norm_max: &mut qkmax,
            d_qkv: &mut dqkv,
            d_qkv_chunk_amax: &mut amax,
            d_qk_scale: &mut dscale,
            accumulate_value_grad: accumulate,
            scratch: bs.args(),
            row_count: rows as u32,
            seq_len: seq as u32,
            batch_size: batch as u32,
            embedding_dim: width as u32,
            qkv_dim: qkv_dim as u32,
            head_count: heads as u32,
            head_dim: 64,
            attention_window: seq as u32,
            partial_key_offset: false,
            selective_attention: false,
            qk_norm_offset: 0,
            backward_mask_seed: 0,
            backward_tile_budget: 0.0,
            reuse_forward_probs: false,
            forward_probs_f16: None,
        },
    )?;
    assert_eq!(
        std::sync::Arc::strong_count(stream.context()),
        backward_context_references,
        "native backward scratch must neither retain nor release context references"
    );
    let output = out.to_host_vec(&stream)?;
    let gradients = dqkv.to_host_vec(&stream)?;
    assert!(output.iter().all(|x| x.is_finite()) && output.iter().any(|x| *x != 0.0));
    assert!(gradients.iter().all(|x| x.is_finite()) && gradients.iter().any(|x| *x != 0.0));
    let digests = [
        hash(output.iter().map(|x| x.to_bits())),
        hash(gradients.iter().map(|x| x.to_bits())),
        hash(states.to_host_vec(&stream)?.into_iter().map(u32::from)),
        hash(qkmax.to_host_vec(&stream)?.iter().map(|x| x.to_bits())),
        hash(
            amax.to_host_vec(&stream)?[..count as usize]
                .iter()
                .map(|x| x.to_bits()),
        ),
        if cached {
            hash(vnew.to_host_vec(&stream)?.into_iter().map(u32::from))
        } else {
            0
        },
    ];
    let expected = if detach {
        assert_eq!(
            (batch, seq, heads, cached, accumulate),
            (4, 2048, 32, true, false)
        );
        // Captured on the accepted PTX before editing the native state-gradient path.
        [
            9563841761746704474,
            10033616252352943118,
            2757280212317737445,
            9083439839750663205,
            17469897835847952426,
            18150118550938871583,
        ]
    } else {
        match (batch, seq, heads, cached, accumulate) {
            (1, 128, 2, true, false) => [
                8282605232789635786,
                533124665179712792,
                17767849051932164127,
                7261168746017876470,
                1415461560531672912,
                18316054953920683695,
            ],
            (2, 192, 3, true, true) => [
                8574949503207710090,
                2568326936220084594,
                12940826935046376462,
                12289443832680002269,
                7103229523045245190,
                12122343449041632122,
            ],
            (1, 128, 2, false, false) => [
                8282605232789635786,
                533124665179712792,
                17767849051932164127,
                7261168746017876470,
                1415461560531672912,
                0,
            ],
            (2, 192, 3, false, true) => [
                8574949503207710090,
                2568326936220084594,
                12940826935046376462,
                12289443832680002269,
                7103229523045245190,
                0,
            ],
            (4, 2048, 32, true, false) => [
                9563841761746704474,
                12266833003598672454,
                2757280212317737445,
                9083439839750663205,
                2053666187632218242,
                18150118550938871583,
            ],
            _ => unreachable!(),
        }
    };
    // Before this change, uncached backward wrote reconstructed v_new over
    // kneg, then used that corrupted kneg to reconstruct AQK. Preserve the
    // original counterexample hashes and require the independent cached
    // gradient/amax goldens instead. Forward and state goldens stay unchanged.
    if !cached {
        // The old uncached forward reconstructed Kneg through future g_last.
        // It must now match the independent, unchanged cached forward golden.
        let old_noncausal_forward = if batch == 1 {
            1139824175716846630
        } else {
            13823825657414252170
        };
        assert_ne!(
            digests[0], old_noncausal_forward,
            "noncausal AQK reconstruction returned"
        );
        let old_wrong_gradient = if batch == 1 {
            6394258566756902918
        } else {
            7721587015442054023
        };
        assert_ne!(
            digests[1], old_wrong_gradient,
            "reconstruction alias returned"
        );
    }
    assert_eq!(
        digests, expected,
        "KDA v_new storage changed an operand or gradient"
    );
    println!(
        "KDA_GOLDEN batch={batch} seq={seq} heads={heads} cached={cached} accumulate={accumulate} detached={detach} count={count} digests={digests:?}"
    );
    Ok(())
}
