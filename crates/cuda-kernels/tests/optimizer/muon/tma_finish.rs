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
    let z_master = DeviceBuffer::from_host(&stream, &z)?;
    let x_master = DeviceBuffer::from_host(&stream, &x)?;
    let schedule_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let bytes = DeviceBuffer::<u8>::zeroed(&stream, LEN / 2)?;
    let scales = DeviceBuffer::<u8>::zeroed(&stream, LEN / 16)?;
    let global_scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let descriptor = MuonSlotDescriptor {
        grad: grad.cu_deviceptr(),
        momentum: momentum.cu_deviceptr(),
        z_master: z_master.cu_deviceptr(),
        x_master: x_master.cu_deviceptr(),
        schedule_amax: schedule_amax.cu_deviceptr(),
        bytes: bytes.cu_deviceptr(),
        scales: scales.cu_deviceptr(),
        global_scale: global_scale.cu_deviceptr(),
        rows: ROWS as u32,
        cols: COLS as u32,
        learning_rate_multiplier: 1.0,
    };
    let slots = DeviceBuffer::from_host(&stream, &[descriptor])?;
    let polar_update = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let polar_bound_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut polar_chunks = DeviceBuffer::<f32>::zeroed(&stream, 2 * MUON_COOPERATIVE_BLOCKS)?;

    module.muon_tma_finish_update(MuonTmaFinishArgs {
        stream: &stream,
        slots: &slots,
        polar_update: &polar_update,
        polar_bound_amax: &polar_bound_amax,
        polar_chunks: &mut polar_chunks,
        slot_index: 0,
        matrix_len: LEN as u32,
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
    let z_master = DeviceBuffer::from_host(stream, &z)?;
    let x_master = DeviceBuffer::from_host(stream, &x)?;
    let schedule_amax = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let bytes = DeviceBuffer::<u8>::zeroed(stream, LEN / 2)?;
    let scales = DeviceBuffer::<u8>::zeroed(stream, LEN / 16)?;
    let global_scale = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let descriptor = MuonSlotDescriptor {
        grad: grad.cu_deviceptr(),
        momentum: momentum.cu_deviceptr(),
        z_master: z_master.cu_deviceptr(),
        x_master: x_master.cu_deviceptr(),
        schedule_amax: schedule_amax.cu_deviceptr(),
        bytes: bytes.cu_deviceptr(),
        scales: scales.cu_deviceptr(),
        global_scale: global_scale.cu_deviceptr(),
        rows: ROWS as u32,
        cols: COLS as u32,
        learning_rate_multiplier: 0.75,
    };
    let slots = DeviceBuffer::from_host(stream, &[descriptor])?;
    let polar_update = DeviceBuffer::from_host(stream, &update)?;
    let polar_bound_amax = DeviceBuffer::from_host(stream, &[4.0_f32])?;
    let mut polar_chunks = DeviceBuffer::<f32>::zeroed(stream, 2 * MUON_COOPERATIVE_BLOCKS)?;
    let args = MuonTmaFinishArgs {
        stream,
        slots: &slots,
        polar_update: &polar_update,
        polar_bound_amax: &polar_bound_amax,
        polar_chunks: &mut polar_chunks,
        slot_index: 0,
        matrix_len: LEN as u32,
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
