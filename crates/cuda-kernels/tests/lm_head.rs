use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::lm_head::{LmHeadArgs, LmHeadModule, LmHeadTmaArgs};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule,
    scale_layout::{sm120_scale_packed_len, sm120_scale_padded_mn_extent},
    scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

mod common;

use common::nvfp4::{one_scales, set_e2m1_one};

const TOKEN_COUNT: usize = 2;
const INPUT_DIM: usize = 64;
const VOCAB_SIZE: usize = 16;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn lm_head_projects_rowwise_nvfp4_hidden_to_logits() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(LmHeadModule::from_module)?;

    let mut input_bytes = vec![0_u8; TOKEN_COUNT * INPUT_DIM / 2];
    set_e2m1_one(&mut input_bytes, 0);
    set_e2m1_one(&mut input_bytes, 1);
    set_e2m1_one(&mut input_bytes, INPUT_DIM + 2);

    let mut weight_bytes = vec![0_u8; VOCAB_SIZE * INPUT_DIM / 2];
    set_e2m1_one(&mut weight_bytes, 0);
    set_e2m1_one(&mut weight_bytes, INPUT_DIM + 1);
    set_e2m1_one(&mut weight_bytes, 2 * INPUT_DIM + 2);

    let input_bytes_dev = DeviceBuffer::from_host(&stream, &input_bytes)?;
    let input_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(TOKEN_COUNT * INPUT_DIM))?;
    let input_global_scales_dev = DeviceBuffer::from_host(&stream, &[1.0_f32; TOKEN_COUNT])?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &weight_bytes)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(VOCAB_SIZE * INPUT_DIM))?;
    let weight_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let mut logits_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * VOCAB_SIZE)?;

    module.logits(LmHeadArgs {
        stream: &stream,
        input: Nvfp4RowwiseDeviceTensor::new(
            &input_bytes_dev,
            &input_scales_dev,
            &input_global_scales_dev,
        ),
        weight: Nvfp4FourSixMmaWeightTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        logits: &mut logits_dev,
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        vocab_size: VOCAB_SIZE as u32,
    })?;

    let logits = logits_dev.to_host_vec(&stream)?;
    common::assert_close(logits[0], 1.0, TOLERANCE);
    common::assert_close(logits[1], 1.0, TOLERANCE);
    common::assert_close(logits[2], 0.0, TOLERANCE);
    common::assert_close(logits[VOCAB_SIZE], 0.0, TOLERANCE);
    common::assert_close(logits[VOCAB_SIZE + 1], 0.0, TOLERANCE);
    common::assert_close(logits[VOCAB_SIZE + 2], 1.0, TOLERANCE);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn lm_head_tma_matches_old_projection() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 128;
    const K: usize = 128;
    const N: usize = 128;

    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LmHeadModule::from_module(ptx.clone())?;
    let tma = Nvfp4GemmModule::from_module(ptx.clone())?;
    let scale_pack = Sm120ScalePackModule::from_module(ptx)?;

    let mut input_bytes = vec![0_u8; ROWS * K / 2];
    for row in 0..ROWS {
        for col in 0..K {
            if (row * 13 + col * 7) % 17 == 0 {
                set_e2m1_one(&mut input_bytes, row * K + col);
            }
        }
    }

    let mut weight_bytes = vec![0_u8; N * K / 2];
    for row in 0..N {
        for col in 0..K {
            if (row * 11 + col * 5 + 3) % 19 == 0 {
                set_e2m1_one(&mut weight_bytes, row * K + col);
            }
        }
    }

    let mut input_scales = one_scales(ROWS * K);
    for (index, scale) in input_scales.iter_mut().enumerate() {
        *scale = [0x30, 0x38, 0x40, 0x34][index & 3];
    }
    let mut weight_scales = one_scales(N * K);
    for (index, scale) in weight_scales.iter_mut().enumerate() {
        *scale = [0x38, 0x3c, 0x34, 0x30][index & 3];
    }
    let input_globals: Vec<f32> = (0..ROWS)
        .map(|row| 0.5 + (row % 7) as f32 * 0.125)
        .collect();

    let input_bytes_dev = DeviceBuffer::from_host(&stream, &input_bytes)?;
    let input_scales_dev = DeviceBuffer::from_host(&stream, &input_scales)?;
    let input_global_scales_dev = DeviceBuffer::from_host(&stream, &input_globals)?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &weight_bytes)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &weight_scales)?;
    let weight_global_scale_dev = DeviceBuffer::from_host(&stream, &[0.75_f32])?;
    let mut old_logits_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * N)?;
    let mut tma_logits_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * N)?;
    let padded_rows = sm120_scale_padded_mn_extent(ROWS);
    let padded_cols = sm120_scale_padded_mn_extent(N);
    let mut input_scale_packed =
        DeviceBuffer::<u8>::zeroed(&stream, sm120_scale_packed_len(padded_rows, K))?;
    let mut weight_scale_packed =
        DeviceBuffer::<u8>::zeroed(&stream, sm120_scale_packed_len(padded_cols, K))?;
    let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&stream)?;

    let input = Nvfp4RowwiseDeviceTensor::new(
        &input_bytes_dev,
        &input_scales_dev,
        &input_global_scales_dev,
    );
    module.logits(LmHeadArgs {
        stream: &stream,
        input,
        weight: Nvfp4FourSixMmaWeightTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        logits: &mut old_logits_dev,
        token_count: ROWS as u32,
        input_dim: K as u32,
        vocab_size: N as u32,
    })?;

    module.logits_tma(LmHeadTmaArgs {
        stream: &stream,
        tma: &tma,
        scale_pack: &scale_pack,
        descriptors: &mut descriptors,
        input_scale_packed: &mut input_scale_packed,
        input,
        weight: Nvfp4DeviceTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        weight_scale_packed: &mut weight_scale_packed,
        logits: &mut tma_logits_dev,
        token_count: ROWS as u32,
        input_dim: K as u32,
        vocab_size: N as u32,
    })?;

    let old_logits = old_logits_dev.to_host_vec(&stream)?;
    let tma_logits = tma_logits_dev.to_host_vec(&stream)?;
    common::assert_slice_close(&tma_logits, &old_logits, 1.0e-5);
    Ok(())
}
