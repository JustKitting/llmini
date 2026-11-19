use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{GPT2_MLP_ROUTE_TILE, GPT2_N_EMBD, Gpt2Rng};
use rust_kernels_cuda::{
    hnet_stage::{RatioObjective, StageShape},
    linear_backward::LinearBackwardMsEdenScratchBuffers,
    nvfp4::csf,
    nvfp4_quant::MsEdenPairDeviceScaleQuantArgs,
    nvfp4_tc_matmul::{QUARTET_MS_EDEN_SCALE_OVERRIDE, nvfp4_tc_matmul_padded_k},
};

use super::{Trainer, outer::OuterBlockGradients};
use crate::{AppResult, training::runtime::Runtime, upload::UploadedNvfp4};

pub(super) fn assert_gradient_pack_matches_rescan(
    runtime: &Runtime,
    gradient: &DeviceBuffer<f32>,
    actual: &LinearBackwardMsEdenScratchBuffers,
    rows: u32,
    columns: u32,
    seeds: [u32; 2],
) -> AppResult {
    let stream = &runtime.stream;
    let mut bytes = DeviceBuffer::zeroed(stream, actual.e_h.bytes.len())?;
    let mut scales = DeviceBuffer::zeroed(stream, actual.e_h.scales.len())?;
    let mut transpose_bytes = DeviceBuffer::zeroed(stream, actual.e_t_h.bytes.len())?;
    let mut transpose_scales = DeviceBuffer::zeroed(stream, actual.e_t_h.scales.len())?;
    let mut chunks = DeviceBuffer::zeroed(stream, actual.e_h.chunk_amax.len())?;
    let mut global = DeviceBuffer::zeroed(stream, 1)?;
    runtime
        .quant
        .fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
            MsEdenPairDeviceScaleQuantArgs {
                stream,
                x: gradient,
                out_fp4: &mut bytes,
                out_scales: &mut scales,
                transpose_out_fp4: &mut transpose_bytes,
                transpose_out_scales: &mut transpose_scales,
                out_chunk_amax: &mut chunks,
                out_global_scale: &mut global,
                row_count: rows,
                src_row_len: columns,
                dst_row_len: nvfp4_tc_matmul_padded_k(columns),
                transpose_dst_row_len: nvfp4_tc_matmul_padded_k(rows),
                scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                sign_seed: seeds[0],
                scale_seed: seeds[1],
                transpose_scale_seed: seeds[1] ^ 0x85eb_ca6b,
                precomputed_chunk_count: None,
            },
        )?;
    assert_eq!(
        actual.e_h.global_scale.to_host_vec(stream)?[0].to_bits(),
        global.to_host_vec(stream)?[0].to_bits(),
        "gradient global scale"
    );
    for (name, observed, expected) in [
        ("row payload", &actual.e_h.bytes, &bytes),
        ("row scales", &actual.e_h.scales, &scales),
        ("transpose payload", &actual.e_t_h.bytes, &transpose_bytes),
        ("transpose scales", &actual.e_t_h.scales, &transpose_scales),
    ] {
        assert!(
            observed.to_host_vec(stream)? == expected.to_host_vec(stream)?,
            "gradient {name} differs from a full rescan"
        );
    }
    Ok(())
}

fn packed_matrix(
    stream: &CudaStream,
    rows: usize,
    width: usize,
    rng: &mut Gpt2Rng,
) -> AppResult<UploadedNvfp4> {
    let bytes: Vec<u8> = (0..rows * width / 2)
        .map(|_| rng.next_u32() as u8)
        .collect();
    let scales = csf::encode_host(&vec![0x38; rows * width / 16], rows, width);
    UploadedNvfp4::from_host(stream, &bytes, &scales, 0.013, rows * width)
}

fn unit_norm_scale(stream: &CudaStream, width: usize) -> AppResult<UploadedNvfp4> {
    let scales = csf::encode_host(&vec![0x38; width / 16], 1, width);
    UploadedNvfp4::from_host(stream, &vec![0x22; width / 2], &scales, 1.0, width)
}

struct TestBlockGradients {
    qkv: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    norm_scale: DeviceBuffer<f32>,
    norm_bias: DeviceBuffer<f32>,
}

impl TestBlockGradients {
    fn poisoned(stream: &CudaStream, qkv: usize, width: usize) -> AppResult<Self> {
        let buffer = |len| DeviceBuffer::from_host(stream, &vec![f32::NAN; len]);
        Ok(Self {
            qkv: buffer(qkv * width)?,
            output: buffer(width * width)?,
            norm_scale: buffer(width)?,
            norm_bias: buffer(width)?,
        })
    }

    fn view(&mut self) -> OuterBlockGradients<'_> {
        OuterBlockGradients {
            qkv: &mut self.qkv,
            output: &mut self.output,
            norm_scale: &mut self.norm_scale,
            norm_bias: &mut self.norm_bias,
        }
    }
}

#[test]
#[ignore = "requires sm_120a PTX and the native NVFP4 H-Net outer block"]
fn outer_hnet_block_is_causal_and_preserves_recurrent_adjoint() -> AppResult {
    use super::outer::{OuterScratch, OuterWeights};

    const BATCH: usize = 2;
    const SEQUENCE: usize = 128;
    const WIDTH: usize = 512;
    const QKV: usize = 2176;
    let runtime = Runtime::new()?;
    let stream = &runtime.stream;
    let mut rng = Gpt2Rng::new(911);
    let qkv = packed_matrix(stream, QKV, WIDTH, &mut rng)?;
    let output = packed_matrix(stream, WIDTH, WIDTH, &mut rng)?;
    let scales = csf::encode_host(&[0x38; WIDTH / 16], 1, WIDTH);
    let norm_scale = UploadedNvfp4::from_host(stream, &[0x22; WIDTH / 2], &scales, 1.0, WIDTH)?;
    let norm_bias = UploadedNvfp4::from_host(stream, &[0; WIDTH / 2], &scales, 1.0, WIDTH)?;
    let weights = OuterWeights {
        qkv: &qkv,
        output: &output,
        norm_scale: &norm_scale,
        norm_bias: &norm_bias,
    };
    let mut scratch = OuterScratch::new(&runtime, BATCH as u32, SEQUENCE as u32, WIDTH as u32)?;
    let input: Vec<_> = (0..BATCH * SEQUENCE * WIDTH)
        .map(|_| (rng.next_u32() % 65) as f32 / 32.0 - 1.0)
        .collect();
    let prefix = 93 * WIDTH;
    let mut altered = input.clone();
    for value in &mut altered[prefix..SEQUENCE * WIDTH] {
        *value = -*value;
    }
    let input_dev = DeviceBuffer::from_host(stream, &input)?;
    let altered_dev = DeviceBuffer::from_host(stream, &altered)?;
    let (output, tape) = scratch.forward(&runtime, weights, &input_dev)?;
    let (altered_output, _) = scratch.forward(&runtime, weights, &altered_dev)?;
    let output = output.to_host_vec(stream)?;
    let altered_output = altered_output.to_host_vec(stream)?;
    assert!(output.iter().all(|x| x.is_finite()));
    assert_ne!(output[..prefix], input[..prefix]);
    assert_eq!(output[..prefix], altered_output[..prefix]);
    assert_eq!(
        output[SEQUENCE * WIDTH..],
        altered_output[SEQUENCE * WIDTH..]
    );

    let mut gradient = vec![0.0; input.len()];
    for (i, value) in gradient[(SEQUENCE - 1) * WIDTH..SEQUENCE * WIDTH]
        .iter_mut()
        .enumerate()
    {
        *value = [0.001, -0.002, 0.003][i % 3];
    }
    let gradient = DeviceBuffer::from_host(stream, &gradient)?;
    // The optimizer reuses these destinations between batches. The reverse
    // kernels must overwrite every entry, including padded projection columns.
    let mut adjoint = TestBlockGradients::poisoned(stream, QKV, WIDTH)?;
    let d_input = tape.backward(&runtime, &mut scratch, &gradient, adjoint.view(), 17, 19)?;
    scratch.assert_qkv_pack_matches_rescan(&runtime, [18, 20])?;
    let d_input = d_input.to_host_vec(stream)?;
    assert!(d_input.iter().all(|value| value.is_finite()));
    assert!(d_input[..64 * WIDTH].iter().any(|value| *value != 0.0));
    assert!(
        d_input[SEQUENCE * WIDTH..]
            .iter()
            .all(|value| *value == 0.0)
    );
    let d_qkv = adjoint.qkv.to_host_vec(stream)?;
    let active = (4 * WIDTH + WIDTH / 64) * WIDTH;
    assert!(d_qkv[..active].iter().any(|value| *value != 0.0));
    assert!(d_qkv[active..].iter().all(|value| *value == 0.0));
    for parameter in [
        d_qkv,
        adjoint.output.to_host_vec(stream)?,
        adjoint.norm_scale.to_host_vec(stream)?,
        adjoint.norm_bias.to_host_vec(stream)?,
    ] {
        assert!(parameter.iter().all(|value| value.is_finite()));
        assert!(parameter.iter().any(|value| *value != 0.0));
    }
    Ok(())
}

#[test]
#[ignore = "uses the CUDA outer blocks, routed inner backbone, and byte head"]
fn native_hnet_hierarchy_reverses_masked_byte_loss() -> AppResult {
    use rust_kernels_cuda::hnet_byte_head::{BYTE_BOS, BYTE_EOS, BYTE_LOGIT_STRIDE};

    use super::{
        bytes::{ByteDocuments, MAGIC},
        hierarchy::{HierarchyScratch, HierarchyWeights, OuterGradients},
        outer::OuterWeights,
        projection::StageWeights,
    };

    const STRIDE: usize = 128;
    const WIDTH: usize = 512;
    const QKV: usize = 2176;
    let lengths = [113, 57];
    let mut trainer = Trainer::new_full_kda(0x484e4554)?;
    let stream = trainer.runtime.stream.clone();
    let stream = stream.as_ref();
    let mut rng = Gpt2Rng::new(19);
    let scales = csf::encode_host(&[0x38; WIDTH / 16], 1, WIDTH);
    let norm_scale = UploadedNvfp4::from_host(stream, &[0x22; WIDTH / 2], &scales, 1.0, WIDTH)?;
    let norm_bias = UploadedNvfp4::from_host(stream, &[0; WIDTH / 2], &scales, 1.0, WIDTH)?;
    let encoder_mixer = packed_matrix(stream, QKV, WIDTH, &mut rng)?;
    let encoder_output = packed_matrix(stream, WIDTH, WIDTH, &mut rng)?;
    let decoder_mixer = packed_matrix(stream, QKV, WIDTH, &mut rng)?;
    let decoder_output = packed_matrix(stream, WIDTH, WIDTH, &mut rng)?;
    let encoder = [OuterWeights {
        qkv: &encoder_mixer,
        output: &encoder_output,
        norm_scale: &norm_scale,
        norm_bias: &norm_bias,
    }];
    let decoder = [OuterWeights {
        qkv: &decoder_mixer,
        output: &decoder_output,
        norm_scale: &norm_scale,
        norm_bias: &norm_bias,
    }];
    let query = packed_matrix(stream, WIDTH, WIDTH, &mut rng)?;
    let key = packed_matrix(stream, WIDTH, WIDTH, &mut rng)?;
    let residual = packed_matrix(stream, WIDTH, WIDTH, &mut rng)?;
    let byte_readout = packed_matrix(stream, BYTE_LOGIT_STRIDE as usize, WIDTH, &mut rng)?;
    let byte_embedding = packed_matrix(stream, (BYTE_BOS + 1) as usize, WIDTH, &mut rng)?;
    let padding = DeviceBuffer::zeroed(stream, GPT2_N_EMBD - WIDTH)?;
    let mut scratch = HierarchyScratch::new(&trainer.runtime, 2, STRIDE as u32, WIDTH as u32)?;
    let mut framed = MAGIC.to_vec();
    framed.extend(2_u32.to_le_bytes());
    for length in lengths {
        let bytes = length - 1_u32;
        framed.extend(bytes.to_le_bytes());
        framed.push(1);
        framed.extend((0..bytes).map(|i| (i * 73 % 256) as u8));
    }
    let mut source = ByteDocuments::new(framed.as_slice())?;
    let batch = source.next_batch(2, STRIDE as u32)?.unwrap();
    assert_eq!(batch.lengths, lengths);
    let mut encoder_adjoints =
        TestBlockGradients::poisoned(stream, encoder_mixer.scale_rows as usize, WIDTH)?;
    let mut decoder_adjoints =
        TestBlockGradients::poisoned(stream, decoder_mixer.scale_rows as usize, WIDTH)?;
    let mut byte_rng = rng.clone();
    let byte_seeds = [byte_rng.next_u32(), byte_rng.next_u32()];
    let (adjoints, d_lookup) = trainer.backward_hnet_bytes(
        &mut scratch,
        &batch,
        &byte_embedding,
        HierarchyWeights {
            encoder: &encoder,
            decoder: &decoder,
            encoder_norm: &norm_scale,
            backbone_norm: &unit_norm_scale(stream, GPT2_N_EMBD)?,
            decoder_norm: &norm_scale,
            stage: StageWeights {
                query: &query,
                key: &key,
                residual: &residual,
                residual_bias: &norm_bias,
            },
            width_padding: Some(&padding),
            byte_readout: &byte_readout,
        },
        OuterGradients {
            encoder: vec![encoder_adjoints.view()],
            decoder: vec![decoder_adjoints.view()],
        },
        RatioObjective {
            target: 4.0,
            weight: 0.03,
        },
        &mut rng,
    )?;
    scratch.assert_byte_pack_matches_rescan(&trainer.runtime, byte_seeds)?;
    assert!(adjoints.ratio_loss.to_host_vec(stream)?[0].is_finite());
    let losses = adjoints.byte_losses.to_host_vec(stream)?;
    let d_embeddings = adjoints.embeddings.to_host_vec(stream)?;
    for (sequence, length) in lengths.into_iter().enumerate() {
        let first = sequence * STRIDE;
        let last = first + length as usize;
        let end = first + STRIDE;
        assert!(
            losses[first..last]
                .iter()
                .all(|loss| loss.is_finite() && *loss > 0.0)
        );
        assert!(losses[last..end].iter().all(|loss| *loss == 0.0));
        assert!(
            d_embeddings[last * WIDTH..end * WIDTH]
                .iter()
                .all(|value| *value == 0.0)
        );
    }
    let d_readout = adjoints.byte_readout.to_host_vec(stream)?;
    assert!(
        d_readout[(BYTE_EOS as usize + 1) * WIDTH..]
            .iter()
            .all(|value| *value == 0.0)
    );
    let d_lookup = d_lookup.to_host_vec(stream)?;
    let eos = BYTE_EOS as usize * WIDTH;
    let bos = BYTE_BOS as usize * WIDTH;
    assert!(d_lookup.iter().all(|g| g.is_finite()));
    assert!(d_lookup[eos..bos].iter().all(|g| *g == 0.0));
    assert!(d_lookup[bos..].iter().any(|g| *g != 0.0));
    for matrix in [
        &encoder_adjoints.qkv,
        &decoder_adjoints.qkv,
        &adjoints.stage.query,
        &adjoints.stage.key,
        &adjoints.stage.residual,
        &adjoints.stage.residual_bias,
        &adjoints.encoder_norm,
        &adjoints.backbone_norm,
        &adjoints.decoder_norm,
        adjoints.width_padding.as_ref().unwrap(),
    ] {
        let values = matrix.to_host_vec(stream)?;
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != 0.0));
    }
    Ok(())
}

#[test]
#[ignore = "runs one complete H-Net update on CUDA"]
fn native_hnet_step_updates_the_outer_and_inner_parameters() -> AppResult {
    use super::{
        bytes::{ByteDocuments, MAGIC},
        hierarchy::HierarchyScratch,
        parameters::{OUTER_WIDTH, Parameters},
        snapshot::Snapshot,
    };

    let mut trainer = Trainer::new_full_kda(720)?;
    let stream = trainer.runtime.stream.clone();
    let mut parameters = Parameters::new(&mut trainer, 1, 4.0, 92)?;
    let mut scratch = HierarchyScratch::new(&trainer.runtime, 2, 128, OUTER_WIDTH as u32)?;
    let before = trainer.buffers.optimizer_state.blocks[0]
        .ln_1
        .bias
        .z_master
        .to_host_vec(&stream)?;

    let mut framed = MAGIC.to_vec();
    framed.extend(2_u32.to_le_bytes());
    for bytes in [
        b"abcdef abcdef abcdef".as_slice(),
        b"The training loss follows the byte targets.".as_slice(),
    ] {
        framed.extend((bytes.len() as u32).to_le_bytes());
        framed.push(1);
        framed.extend(bytes);
    }
    let batch = ByteDocuments::new(framed.as_slice())?
        .next_batch(2, 128)?
        .unwrap();
    let step = trainer.train_hnet_step(
        &mut scratch,
        &mut parameters,
        &batch,
        RatioObjective {
            target: 4.0,
            weight: 0.03,
        },
        &mut Gpt2Rng::new(66),
    )?;
    assert!(!step.optimizer.update_skipped);
    assert!(step.byte_ce > 0.0);
    assert!(!step.has_nonfinite());
    assert_eq!(trainer.buffers.optimizer_state.step, 1);
    assert_ne!(
        trainer.buffers.optimizer_state.blocks[0]
            .ln_1
            .bias
            .z_master
            .to_host_vec(&stream)?,
        before
    );

    trainer.materialize_evaluation_weights()?;
    parameters.materialize(&trainer.runtime, &mut trainer.buffers.optimizer, 1.0)?;
    let averaged = parameters.weights();
    let losses = trainer.score_hnet_bytes(
        &mut scratch,
        &batch,
        averaged.embedding,
        averaged.hierarchy(),
        RatioObjective {
            target: 4.0,
            weight: 0.03,
        },
    )?;
    assert_eq!(trainer.buffers.optimizer_state.step, 1);
    let mut outer_checkpoint = Vec::new();
    parameters.write_weights(&stream, &mut outer_checkpoint)?;
    for (&target, &loss) in batch.targets.iter().zip(&losses) {
        if target == u32::MAX {
            assert_eq!(loss, 0.0);
        } else {
            assert!(loss.is_finite() && loss > 0.0);
        }
    }
    let weights = averaged.hierarchy();
    assert!(
        weights
            .width_padding
            .unwrap()
            .to_host_vec(&stream)?
            .iter()
            .any(|&bits| bits & 0x7fff != 0)
    );
    for tensor in [
        weights.stage.residual,
        weights.encoder[0].norm_bias,
        weights.decoder[0].norm_bias,
    ] {
        assert!(tensor.global_scale_to_host(&stream)? > 0.0);
        assert!(
            tensor
                .bytes
                .to_host_vec(&stream)?
                .into_iter()
                .any(|byte| byte & 0x77 != 0)
        );
    }

    let mut backbone_checkpoint = Vec::new();
    crate::checkpoint::write_uploaded_model_to(
        &stream,
        &trainer.uploaded,
        &mut backbone_checkpoint,
    )?;
    drop(parameters);
    drop(scratch);
    drop(trainer);
    drop(stream);

    let mut reopened =
        Trainer::from_hnet_backbone_checkpoint(0x47505432, backbone_checkpoint.as_slice())?;
    drop(backbone_checkpoint);
    let snapshot = Snapshot::read(&reopened.runtime, outer_checkpoint.as_slice(), 1)?;
    let mut scratch = HierarchyScratch::new(&reopened.runtime, 2, 128, OUTER_WIDTH as u32)?;
    let saved = snapshot.weights();
    let saved_losses = reopened.score_hnet_bytes(
        &mut scratch,
        &batch,
        saved.embedding,
        saved.hierarchy(),
        RatioObjective {
            target: 4.0,
            weight: 0.03,
        },
    )?;
    assert_eq!(losses, saved_losses);
    assert_eq!(reopened.buffers.optimizer_state.step, 0);
    Ok(())
}

#[test]
#[ignore = "requires sm_120a PTX and the full native KDA backbone"]
fn native_routed_backbone_executes_and_reverses_real_blocks() -> AppResult {
    let mut trainer = Trainer::new_full_kda(0x47505432)?;
    let stream = trainer.runtime.stream.clone();
    let extra = DeviceBuffer::from_host(&stream, &[3.0_f32, 4.0])?;
    trainer.prepare_hnet_optimizer(&[("hnet.test_outer_gradient", &extra)])?;
    let unused_gradient = [f32::NAN];
    for buffer in [
        &trainer.buffers.backward.d_lm_head_weight,
        &trainer.buffers.next_latent_grads.d_transition_weight,
    ] {
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                buffer.cu_deviceptr(),
                unused_gradient.as_ptr(),
                size_of::<f32>(),
                stream.cu_stream(),
            )?;
        }
    }
    stream.synchronize()?;
    let stage = trainer.runtime.hnet_stage_module()?;
    const BATCH: usize = 2;
    const OUTER: usize = 256;
    const WIDTH: usize = 512;
    let lengths = [192, 91];
    let queries = [0x3c00_u16, 0].repeat(BATCH * OUTER);
    let mut keys = [0x3c00_u16, 0].repeat(BATCH * OUTER);
    // A negative cosine selects these positions. An orthogonal key is a
    // continuation under the released router's tie rule.
    for row in [41, 98, OUTER + 32] {
        keys[2 * row..2 * row + 2].copy_from_slice(&[0xbc00, 0x3c00]);
    }
    let encoded: Vec<_> = (0..BATCH * OUTER * WIDTH)
        .map(|index| [0x3000_u16, 0xb000, 0x3400, 0x2800][(index + index / WIDTH) % 4])
        .collect();
    let q = DeviceBuffer::from_host(&stream, &queries)?;
    let k = DeviceBuffer::from_host(&stream, &keys)?;
    let valid = DeviceBuffer::from_host(&stream, &lengths)?;
    let encoded = DeviceBuffer::from_host(&stream, &encoded)?;
    let plan = stage.route(
        &stream,
        &q,
        &k,
        &valid,
        StageShape {
            batch: BATCH as u32,
            outer_stride: OUTER as u32,
            features: WIDTH as u32,
            router_width: 2,
            inner_alignment: GPT2_MLP_ROUTE_TILE as u32,
        },
        RatioObjective {
            target: 4.0,
            weight: 0.03,
        },
    )?;
    assert_eq!(plan.inner_lengths().to_host_vec(&stream)?, [3, 2]);
    assert_eq!(plan.inner_stride(), GPT2_MLP_ROUTE_TILE as u32);
    let gathered = stage.gather(&plan, &encoded)?.to_host_vec(&stream)?;
    let padding = DeviceBuffer::zeroed(&stream, GPT2_N_EMBD - WIDTH)?;
    let rms = trainer.runtime.hnet_rms_norm_module()?;
    let scale = unit_norm_scale(&stream, GPT2_N_EMBD)?;
    let tape = trainer.forward_routed_backbone(
        &stage,
        &plan,
        &encoded,
        Some(&padding),
        &rms,
        scale.device(),
    )?;
    let native = tape.output().to_host_vec(&stream)?;
    assert_ne!(native, gathered);
    for (sequence, valid) in [3, 2].into_iter().enumerate() {
        let row_base = sequence * plan.inner_stride() as usize;
        let real = &native[row_base * WIDTH..(row_base + valid) * WIDTH];
        assert!(real.iter().all(|&bits| bits & 0x7c00 != 0x7c00));
        let tail =
            &native[(row_base + valid) * WIDTH..(row_base + plan.inner_stride() as usize) * WIDTH];
        assert!(tail.iter().all(|&bits| bits == 0));
    }

    let output_gradient = (0..BATCH * OUTER * WIDTH)
        .map(|i| [0.0001_f32, -0.0002, 0.0003][i % 3])
        .collect::<Vec<_>>();
    let output_gradient = DeviceBuffer::from_host(&stream, &output_gradient)?;
    let smoothed = stage.smooth_chunks(&plan, tape.output())?;
    let reverse = stage.backward(&smoothed, &output_gradient)?;
    drop(smoothed);
    let (inner_adjoint, scale_adjoint) = tape.backward(&reverse.d_inner)?;
    let scale_adjoint = scale_adjoint.to_host_vec(&stream)?;
    assert!(scale_adjoint[..WIDTH].iter().any(|grad| *grad != 0.0));
    assert!(scale_adjoint[WIDTH..].iter().all(|grad| *grad == 0.0));
    let d_padding = inner_adjoint.d_padding.unwrap().to_host_vec(&stream)?;
    assert_eq!(d_padding.len(), GPT2_N_EMBD - WIDTH);
    assert!(d_padding.iter().all(|grad| grad.is_finite()));
    assert!(d_padding.iter().any(|grad| *grad != 0.0));
    let encoder_adjoint = stage
        .scatter(&plan, &inner_adjoint.d_features)?
        .to_host_vec(&stream)?;
    assert!(encoder_adjoint.iter().all(|value| value.is_finite()));
    assert!(
        encoder_adjoint
            .chunks_exact(WIDTH)
            .enumerate()
            .all(|(row, grad)| {
                let selected = [0, 41, 98, OUTER, OUTER + 32].contains(&row);
                selected == grad.iter().any(|value| *value != 0.0)
            })
    );
    assert_eq!(trainer.buffers.optimizer_state.step, 0);
    let state = &trainer.buffers.optimizer_state;
    let before_block_affine = state.blocks[0].ln_1.bias.z_master.to_host_vec(&stream)?;
    let before_final_affine = state.ln_f.weight.z_master.to_host_vec(&stream)?;
    let before_next_latent_scale = state
        .next_latent
        .transition
        .weight_muon
        .schedule_amax
        .to_host_vec(&stream)?;
    let before_embedding_scale = trainer
        .uploaded
        .token_embedding
        .global_scale_to_host(&stream)?;
    let update = trainer.update_hnet_backbone(plan.batch() * plan.inner_stride(), None)?;
    assert!(!update.trace.update_skipped);
    assert!(update.trace.grad_norm.is_finite() && update.trace.grad_norm >= 5.0);
    let state = &trainer.buffers.optimizer_state;
    assert_eq!(state.step, 1);
    assert_ne!(
        state.blocks[0].ln_1.bias.z_master.to_host_vec(&stream)?,
        before_block_affine
    );
    assert_eq!(
        state.ln_f.weight.z_master.to_host_vec(&stream)?,
        before_final_affine
    );
    assert_eq!(
        state
            .next_latent
            .transition
            .weight_muon
            .schedule_amax
            .to_host_vec(&stream)?,
        before_next_latent_scale
    );
    assert_eq!(
        trainer
            .uploaded
            .token_embedding
            .global_scale_to_host(&stream)?,
        before_embedding_scale
    );
    Ok(())
}
