use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::optimizer::{
    MUON_COOPERATIVE_BLOCKS, MuonSlotDescriptor, MuonTmaFinishArgs, OptimizerModule,
};

use crate::common;

const ROWS: usize = 32;
const COLS: usize = 32;
const LEN: usize = ROWS * COLS;
const SCHEDULE_BETA: f32 = 0.4;

pub fn run_schedule_amax_case() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let z: Vec<_> = (0..LEN).map(|i| (i as f32 - 511.0) / 997.0).collect();
    let x: Vec<_> = (0..LEN).map(|i| (307.0 - i as f32) / 613.0).collect();
    let expected = z
        .iter()
        .zip(x.iter())
        .map(|(&z, &x)| (z + SCHEDULE_BETA * (x - z)).abs())
        .fold(0.0_f32, f32::max);

    let grad = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let momentum = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let second_momentum = DeviceBuffer::<f32>::zeroed(&stream, COLS)?;
    let z_master = DeviceBuffer::from_host(&stream, &z)?;
    let x_master = DeviceBuffer::from_host(&stream, &x)?;
    let schedule_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let bytes = DeviceBuffer::<u8>::zeroed(&stream, LEN / 2)?;
    let scales = DeviceBuffer::<u8>::zeroed(&stream, LEN / 16)?;
    let global_scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let descriptor = MuonSlotDescriptor {
        grad: grad.cu_deviceptr(),
        momentum: momentum.cu_deviceptr(),
        second_momentum: second_momentum.cu_deviceptr(),
        z_master: z_master.cu_deviceptr(),
        x_master: x_master.cu_deviceptr(),
        schedule_amax: schedule_amax.cu_deviceptr(),
        bytes: bytes.cu_deviceptr(),
        scales: scales.cu_deviceptr(),
        global_scale: global_scale.cu_deviceptr(),
        rows: ROWS as u32,
        cols: COLS as u32,
        learning_rate_multiplier: 1.0,
        qk_clip_factor_offset: u32::MAX,
        qk_clip_head_dim: 0,
    };
    let slots = DeviceBuffer::from_host(&stream, &[descriptor])?;
    let polar_update = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let polar_bound_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut polar_chunks = DeviceBuffer::<f32>::zeroed(&stream, 2 * MUON_COOPERATIVE_BLOCKS)?;
    let mut normuon_factors = DeviceBuffer::<f32>::zeroed(&stream, COLS)?;
    let mut normuon_chunks = DeviceBuffer::<f32>::zeroed(&stream, 2 * COLS)?;
    let qk_clip_factors = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    module.muon_tma_finish_update(MuonTmaFinishArgs {
        stream: &stream,
        slots: &slots,
        polar_update: &polar_update,
        polar_bound_amax: &polar_bound_amax,
        polar_chunks: &mut polar_chunks,
        normuon_factors: &mut normuon_factors,
        normuon_chunks: &mut normuon_chunks,
        qk_clip_factors: &qk_clip_factors,
        slot_index: 0,
        matrix_len: LEN as u32,
        polar_cols: COLS as u32,
        learning_rate: 0.0,
        weight_decay: 0.0,
        average_coefficient: 0.0,
        schedule_beta: SCHEDULE_BETA,
        apply_polar_sqrt_bound: 0,
    })?;

    let actual = schedule_amax.to_host_vec(&stream)?[0];
    assert!(
        (actual - expected).abs() <= 1.0e-7,
        "actual={actual} expected={expected}"
    );
    common::assert_slice_close(&z_master.to_host_vec(&stream)?, &z, 0.0);
    common::assert_slice_close(&x_master.to_host_vec(&stream)?, &x, 0.0);
    Ok(())
}

pub fn run_qk_clip_case() -> Result<(), Box<dyn Error>> {
    const CLIP_ROWS: usize = 2048;
    const CLIP_COLS: usize = 64;
    const CLIP_LEN: usize = CLIP_ROWS * CLIP_COLS;
    const HEAD_DIM: usize = 32;
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let z = vec![1.0_f32; CLIP_LEN];
    let x = vec![2.0_f32; CLIP_LEN];
    let momentum_values = vec![3.0_f32; CLIP_LEN];
    let grad = DeviceBuffer::<f32>::zeroed(&stream, CLIP_LEN)?;
    let momentum = DeviceBuffer::from_host(&stream, &momentum_values)?;
    let second_momentum = DeviceBuffer::<f32>::zeroed(&stream, CLIP_ROWS)?;
    let z_master = DeviceBuffer::from_host(&stream, &z)?;
    let x_master = DeviceBuffer::from_host(&stream, &x)?;
    let schedule_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let bytes = DeviceBuffer::<u8>::zeroed(&stream, CLIP_LEN / 2)?;
    let scales = DeviceBuffer::<u8>::zeroed(&stream, CLIP_LEN / 16)?;
    let global_scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let descriptor = MuonSlotDescriptor {
        grad: grad.cu_deviceptr(),
        momentum: momentum.cu_deviceptr(),
        second_momentum: second_momentum.cu_deviceptr(),
        z_master: z_master.cu_deviceptr(),
        x_master: x_master.cu_deviceptr(),
        schedule_amax: schedule_amax.cu_deviceptr(),
        bytes: bytes.cu_deviceptr(),
        scales: scales.cu_deviceptr(),
        global_scale: global_scale.cu_deviceptr(),
        rows: CLIP_ROWS as u32,
        cols: CLIP_COLS as u32,
        learning_rate_multiplier: 1.0,
        qk_clip_factor_offset: 0,
        qk_clip_head_dim: HEAD_DIM as u32,
    };
    let slots = DeviceBuffer::from_host(&stream, &[descriptor])?;
    let polar_update = DeviceBuffer::<f32>::zeroed(&stream, CLIP_LEN)?;
    let polar_bound_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut polar_chunks = DeviceBuffer::<f32>::zeroed(&stream, 2 * CLIP_COLS)?;
    let mut normuon_factors = DeviceBuffer::<f32>::zeroed(&stream, CLIP_ROWS)?;
    let mut normuon_chunks = DeviceBuffer::<f32>::zeroed(&stream, 2 * CLIP_ROWS)?;
    let qk_clip_factors = DeviceBuffer::from_host(&stream, &[0.5_f32, 0.25_f32])?;

    module.muon_tma_finish_update_deferred_quantization(MuonTmaFinishArgs {
        stream: &stream,
        slots: &slots,
        polar_update: &polar_update,
        polar_bound_amax: &polar_bound_amax,
        polar_chunks: &mut polar_chunks,
        normuon_factors: &mut normuon_factors,
        normuon_chunks: &mut normuon_chunks,
        qk_clip_factors: &qk_clip_factors,
        slot_index: 0,
        matrix_len: CLIP_LEN as u32,
        polar_cols: CLIP_ROWS as u32,
        learning_rate: 0.0,
        weight_decay: 0.0,
        average_coefficient: 0.0,
        schedule_beta: SCHEDULE_BETA,
        apply_polar_sqrt_bound: 0,
    })?;

    let mut expected_z = z;
    let mut expected_x = x;
    let mut expected_momentum = momentum_values;
    for col in 0..CLIP_COLS {
        let factor = if col < HEAD_DIM { 0.5 } else { 0.25 };
        for index in col * CLIP_ROWS..(col + 1) * CLIP_ROWS {
            expected_z[index] *= factor;
            expected_x[index] *= factor;
            expected_momentum[index] *= factor;
        }
    }
    assert_eq!(z_master.to_host_vec(&stream)?, expected_z);
    assert_eq!(x_master.to_host_vec(&stream)?, expected_x);
    assert_eq!(momentum.to_host_vec(&stream)?, expected_momentum);
    assert_eq!(
        schedule_amax.to_host_vec(&stream)?[0],
        0.5 * (1.0 + SCHEDULE_BETA)
    );
    Ok(())
}

pub fn run_normuon_variance_case() -> Result<(), Box<dyn Error>> {
    const NORM_ROWS: usize = 32;
    const NORM_COLS: usize = 32;
    const NORM_LEN: usize = NORM_ROWS * NORM_COLS;
    const LEARNING_RATE: f32 = 0.01;
    const BETA2: f32 = 0.95;

    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let update: Vec<_> = (0..NORM_LEN)
        .map(|index| {
            let row = index / NORM_COLS;
            let col = index % NORM_COLS;
            (row + 1) as f32 * (col % 7 + 1) as f32 / 128.0
        })
        .collect();
    let old_second: Vec<_> = (0..NORM_COLS)
        .map(|col| 0.01 + col as f32 / 10_000.0)
        .collect();

    let mut expected_second = Vec::with_capacity(NORM_COLS);
    let mut expected_factors = Vec::with_capacity(NORM_COLS);
    let mut raw_sumsq = 0.0_f32;
    let mut scaled_sumsq = 0.0_f32;
    for row in 0..NORM_ROWS {
        let row_sumsq = (0..NORM_COLS)
            .map(|col| update[row * NORM_COLS + col].powi(2))
            .sum::<f32>();
        let mean = row_sumsq / NORM_COLS as f32;
        let next = BETA2 * old_second[row] + (1.0 - BETA2) * mean;
        let factor = next.max(1.0e-10).sqrt().recip();
        expected_second.push(next);
        expected_factors.push(factor);
        raw_sumsq += row_sumsq;
        scaled_sumsq += row_sumsq * factor * factor;
    }
    let expected_scale = (raw_sumsq / scaled_sumsq.max(1.0e-10)).sqrt();

    let grad = DeviceBuffer::<f32>::zeroed(&stream, NORM_LEN)?;
    let momentum = DeviceBuffer::<f32>::zeroed(&stream, NORM_LEN)?;
    let second_momentum = DeviceBuffer::from_host(&stream, &old_second)?;
    let z_master = DeviceBuffer::<f32>::zeroed(&stream, NORM_LEN)?;
    let x_master = DeviceBuffer::<f32>::zeroed(&stream, NORM_LEN)?;
    let schedule_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let bytes = DeviceBuffer::<u8>::zeroed(&stream, NORM_LEN / 2)?;
    let scales = DeviceBuffer::<u8>::zeroed(&stream, NORM_LEN / 16)?;
    let global_scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let descriptor = MuonSlotDescriptor {
        grad: grad.cu_deviceptr(),
        momentum: momentum.cu_deviceptr(),
        second_momentum: second_momentum.cu_deviceptr(),
        z_master: z_master.cu_deviceptr(),
        x_master: x_master.cu_deviceptr(),
        schedule_amax: schedule_amax.cu_deviceptr(),
        bytes: bytes.cu_deviceptr(),
        scales: scales.cu_deviceptr(),
        global_scale: global_scale.cu_deviceptr(),
        rows: NORM_ROWS as u32,
        cols: NORM_COLS as u32,
        learning_rate_multiplier: 1.0,
        qk_clip_factor_offset: u32::MAX,
        qk_clip_head_dim: 0,
    };
    let slots = DeviceBuffer::from_host(&stream, &[descriptor])?;
    let polar_update = DeviceBuffer::from_host(&stream, &update)?;
    let polar_bound_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut polar_chunks = DeviceBuffer::<f32>::zeroed(&stream, 2 * MUON_COOPERATIVE_BLOCKS)?;
    let mut normuon_factors = DeviceBuffer::<f32>::zeroed(&stream, NORM_COLS)?;
    let mut normuon_chunks = DeviceBuffer::<f32>::zeroed(&stream, 2 * NORM_COLS)?;
    let qk_clip_factors = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    module.muon_tma_finish_update_deferred_quantization(MuonTmaFinishArgs {
        stream: &stream,
        slots: &slots,
        polar_update: &polar_update,
        polar_bound_amax: &polar_bound_amax,
        polar_chunks: &mut polar_chunks,
        normuon_factors: &mut normuon_factors,
        normuon_chunks: &mut normuon_chunks,
        qk_clip_factors: &qk_clip_factors,
        slot_index: 0,
        matrix_len: NORM_LEN as u32,
        polar_cols: NORM_COLS as u32,
        learning_rate: LEARNING_RATE,
        weight_decay: 0.0,
        average_coefficient: 1.0,
        schedule_beta: SCHEDULE_BETA,
        apply_polar_sqrt_bound: 0,
    })?;

    let actual_second = second_momentum.to_host_vec(&stream)?;
    let actual_factors = normuon_factors.to_host_vec(&stream)?;
    let actual_chunks = normuon_chunks.to_host_vec(&stream)?;
    common::assert_slice_close(&actual_second, &expected_second, 2.0e-6);
    common::assert_slice_close(&actual_factors, &expected_factors, 2.0e-5);
    assert!(
        (actual_chunks[0] - expected_scale).abs() <= 2.0e-6,
        "actual_scale={} expected_scale={expected_scale}",
        actual_chunks[0]
    );

    let muon_scale = 0.2 * (NORM_COLS as f32).sqrt();
    let expected_z: Vec<_> = update
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            -LEARNING_RATE
                * muon_scale
                * value
                * expected_factors[index / NORM_COLS]
                * expected_scale
        })
        .collect();
    common::assert_slice_close(&z_master.to_host_vec(&stream)?, &expected_z, 2.0e-6);
    common::assert_slice_close(&x_master.to_host_vec(&stream)?, &expected_z, 2.0e-6);

    let transformed_sumsq = update
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let transformed = value * actual_factors[index / NORM_COLS] * actual_chunks[0];
            transformed * transformed
        })
        .sum::<f32>();
    assert!(
        (transformed_sumsq - raw_sumsq).abs() <= raw_sumsq * 2.0e-5,
        "transformed_sumsq={transformed_sumsq} raw_sumsq={raw_sumsq}"
    );
    Ok(())
}

pub fn run_split_matches_reference_case() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let split = run_finish(&stream, &module, false)?;
    let cooperative = run_finish(&stream, &module, true)?;
    assert_eq!(split.z, cooperative.z);
    assert_eq!(split.x, cooperative.x);
    assert_eq!(split.schedule_amax, cooperative.schedule_amax);
    assert_eq!(split.bytes, cooperative.bytes);
    assert_eq!(split.scales, cooperative.scales);
    assert_eq!(split.global_scale, cooperative.global_scale);
    Ok(())
}

struct FinishOutput {
    z: Vec<f32>,
    x: Vec<f32>,
    schedule_amax: Vec<f32>,
    bytes: Vec<u8>,
    scales: Vec<u8>,
    global_scale: Vec<f32>,
}

fn run_finish(
    stream: &cuda_core::CudaStream,
    module: &OptimizerModule,
    cooperative: bool,
) -> Result<FinishOutput, Box<dyn Error>> {
    let z: Vec<_> = (0..LEN).map(|i| (i as f32 - 511.0) / 997.0).collect();
    let x: Vec<_> = (0..LEN).map(|i| (307.0 - i as f32) / 613.0).collect();
    let update: Vec<_> = (0..LEN)
        .map(|i| ((i * 73 % 257) as f32 - 128.0) / 211.0)
        .collect();
    let grad = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
    let momentum = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
    let second_momentum = DeviceBuffer::<f32>::zeroed(stream, COLS)?;
    let z_master = DeviceBuffer::from_host(stream, &z)?;
    let x_master = DeviceBuffer::from_host(stream, &x)?;
    let schedule_amax = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let bytes = DeviceBuffer::<u8>::zeroed(stream, LEN / 2)?;
    let scales = DeviceBuffer::<u8>::zeroed(stream, LEN / 16)?;
    let global_scale = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let descriptor = MuonSlotDescriptor {
        grad: grad.cu_deviceptr(),
        momentum: momentum.cu_deviceptr(),
        second_momentum: second_momentum.cu_deviceptr(),
        z_master: z_master.cu_deviceptr(),
        x_master: x_master.cu_deviceptr(),
        schedule_amax: schedule_amax.cu_deviceptr(),
        bytes: bytes.cu_deviceptr(),
        scales: scales.cu_deviceptr(),
        global_scale: global_scale.cu_deviceptr(),
        rows: ROWS as u32,
        cols: COLS as u32,
        learning_rate_multiplier: 0.75,
        qk_clip_factor_offset: u32::MAX,
        qk_clip_head_dim: 0,
    };
    let slots = DeviceBuffer::from_host(stream, &[descriptor])?;
    let polar_update = DeviceBuffer::from_host(stream, &update)?;
    let polar_bound_amax = DeviceBuffer::from_host(stream, &[4.0_f32])?;
    let mut polar_chunks = DeviceBuffer::<f32>::zeroed(stream, 2 * MUON_COOPERATIVE_BLOCKS)?;
    let mut normuon_factors = DeviceBuffer::<f32>::zeroed(stream, COLS)?;
    let mut normuon_chunks = DeviceBuffer::<f32>::zeroed(stream, 2 * COLS)?;
    let qk_clip_factors = DeviceBuffer::from_host(stream, &[1.0_f32])?;
    let args = MuonTmaFinishArgs {
        stream,
        slots: &slots,
        polar_update: &polar_update,
        polar_bound_amax: &polar_bound_amax,
        polar_chunks: &mut polar_chunks,
        normuon_factors: &mut normuon_factors,
        normuon_chunks: &mut normuon_chunks,
        qk_clip_factors: &qk_clip_factors,
        slot_index: 0,
        matrix_len: LEN as u32,
        polar_cols: COLS as u32,
        learning_rate: 0.03,
        weight_decay: 0.1,
        average_coefficient: 0.2,
        schedule_beta: SCHEDULE_BETA,
        apply_polar_sqrt_bound: 1,
    };
    if cooperative {
        module.muon_tma_finish_update_cooperative_reference(args)?;
    } else {
        module.muon_tma_finish_update(args)?;
    }
    Ok(FinishOutput {
        z: z_master.to_host_vec(stream)?,
        x: x_master.to_host_vec(stream)?,
        schedule_amax: schedule_amax.to_host_vec(stream)?,
        bytes: bytes.to_host_vec(stream)?,
        scales: scales.to_host_vec(stream)?,
        global_scale: global_scale.to_host_vec(stream)?,
    })
}
