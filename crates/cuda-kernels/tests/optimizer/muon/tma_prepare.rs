use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::optimizer::{
    MUON_COOPERATIVE_BLOCKS, MuonSlotDescriptor, MuonTmaPrepareArgs, OptimizerModule,
};

use crate::common;

pub fn run_split_matches_reference_case() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    compare_shape(&stream, &module, 64, 128)?;
    compare_shape(&stream, &module, 128, 64)
}

fn compare_shape(
    stream: &CudaStream,
    module: &OptimizerModule,
    rows: usize,
    cols: usize,
) -> Result<(), Box<dyn Error>> {
    let split = run_prepare(stream, module, rows, cols, false)?;
    let cooperative = run_prepare(stream, module, rows, cols, true)?;
    assert_eq!(split.momentum, cooperative.momentum);
    assert_eq!(split.oriented, cooperative.oriented);
    assert_eq!(split.polar_x, cooperative.polar_x);
    assert_eq!(split.polar_chunks, cooperative.polar_chunks);
    Ok(())
}

struct PrepareOutput {
    momentum: Vec<f32>,
    oriented: Vec<f32>,
    polar_x: Vec<f32>,
    polar_chunks: Vec<f32>,
}

fn run_prepare(
    stream: &CudaStream,
    module: &OptimizerModule,
    rows: usize,
    cols: usize,
    cooperative: bool,
) -> Result<PrepareOutput, Box<dyn Error>> {
    let len = rows * cols;
    let grad: Vec<_> = (0..len)
        .map(|i| ((i * 73 % 257) as f32 - 128.0) / 211.0)
        .collect();
    let momentum: Vec<_> = (0..len)
        .map(|i| ((i * 29 % 131) as f32 - 65.0) / 173.0)
        .collect();
    let grad = DeviceBuffer::from_host(stream, &grad)?;
    let momentum = DeviceBuffer::from_host(stream, &momentum)?;
    let z_master = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let x_master = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let schedule_amax = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let bytes = DeviceBuffer::<u8>::zeroed(stream, 1)?;
    let scales = DeviceBuffer::<u8>::zeroed(stream, 1)?;
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
        rows: rows as u32,
        cols: cols as u32,
        learning_rate_multiplier: 1.0,
    };
    let slots = DeviceBuffer::from_host(stream, &[descriptor])?;
    let mut oriented = DeviceBuffer::<f32>::zeroed(stream, len)?;
    let mut polar_x = DeviceBuffer::<f32>::zeroed(stream, len)?;
    let mut polar_chunks = DeviceBuffer::<f32>::zeroed(stream, 2 * MUON_COOPERATIVE_BLOCKS)?;
    let args = MuonTmaPrepareArgs {
        stream,
        slots: &slots,
        oriented: &mut oriented,
        polar_x: &mut polar_x,
        polar_chunks: &mut polar_chunks,
        slot_index: 0,
        matrix_len: len as u32,
        mu: 0.9,
        grad_scale: 0.125,
    };
    if cooperative {
        module.muon_tma_prepare_polar_cooperative_reference(args)?;
    } else {
        module.muon_tma_prepare_polar(args)?;
    }
    Ok(PrepareOutput {
        momentum: momentum.to_host_vec(stream)?,
        oriented: oriented.to_host_vec(stream)?,
        polar_x: polar_x.to_host_vec(stream)?,
        polar_chunks: polar_chunks.to_host_vec(stream)?,
    })
}
