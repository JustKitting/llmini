use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{GPT2_MLP_ROUTE_TILE, GPT2_N_EMBD};
use rust_kernels_cuda::hnet_stage::{RatioObjective, StageShape};

use super::Trainer;
use crate::AppResult;

#[test]
#[ignore = "requires sm_120a PTX and the native NVFP4 H-Net outer block"]
fn outer_hnet_block_is_causal_and_preserves_recurrent_adjoint() -> AppResult {
    use gpt2_nvfp4::Gpt2Rng;
    use rust_kernels_cuda::nvfp4::csf;

    use super::outer::{OuterScratch, OuterWeights};
    use crate::{training::runtime::Runtime, upload::UploadedNvfp4};

    const BATCH: usize = 2;
    const SEQUENCE: usize = 128;
    const WIDTH: usize = 512;
    const QKV: usize = 2176;
    let runtime = Runtime::new()?;
    let stream = &runtime.stream;
    let mut rng = Gpt2Rng::new(911);
    let mut matrix = |rows| -> AppResult<UploadedNvfp4> {
        let bytes: Vec<u8> = (0..rows * WIDTH / 2)
            .map(|_| rng.next_u32() as u8)
            .collect();
        let scales = csf::encode_host(&vec![0x38; rows * WIDTH / 16], rows, WIDTH);
        UploadedNvfp4::from_host(stream, &bytes, &scales, 0.013, rows * WIDTH)
    };
    let qkv = matrix(QKV)?;
    let output = matrix(WIDTH)?;
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
    let adjoint = tape.backward(&runtime, &mut scratch, &gradient, 17, 19)?;
    let d_input = adjoint.input.to_host_vec(stream)?;
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
#[ignore = "requires sm_120a PTX and the full native KDA backbone"]
fn native_routed_backbone_executes_and_reverses_real_blocks() -> AppResult {
    let mut trainer = Trainer::new_full_kda(0x47505432)?;
    let stream = trainer.runtime.stream.clone();
    let stage = trainer.runtime.hnet_stage_module()?;
    const BATCH: usize = 2;
    const OUTER: usize = 256;
    const WIDTH: usize = 512;
    let lengths = [192, 91];
    let mut queries = [0x3c00_u16, 0].repeat(BATCH * OUTER);
    for row in [41, 98, OUTER + 32] {
        queries[2 * row..2 * row + 2].copy_from_slice(&[0, 0x3c00]);
    }
    let keys = [0x3c00_u16, 0].repeat(BATCH * OUTER);
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
    let tape = trainer.forward_routed_backbone(&stage, &plan, &encoded, Some(&padding))?;
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
    let expanded = stage.expand(&plan, tape.output())?;
    let reverse = stage.backward(&expanded, &output_gradient)?;
    drop(expanded);
    let inner_adjoint = tape.backward(&reverse.d_inner)?;
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
    Ok(())
}
