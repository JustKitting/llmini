use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::optimizer::{
    MUON_COOPERATIVE_BLOCKS, MuonSlotDescriptor, MuonTmaPrepareArgs, OptimizerModule,
};

use crate::common;

pub fn run_split_matches_reference_case() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    compare_shape(&stream, &module, 64, 128, true, false)?;
    compare_shape(&stream, &module, 128, 64, true, false)?;
    compare_shape(&stream, &module, 64, 128, false, false)?;
    compare_shape(&stream, &module, 128, 64, false, false)
}

pub fn run_muon_vs_reference_case() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    compare_shape(&stream, &module, 64, 128, true, true)?;
    compare_shape(&stream, &module, 128, 64, true, true)
}

fn compare_shape(
    stream: &CudaStream,
    module: &OptimizerModule,
    rows: usize,
    cols: usize,
    nesterov: bool,
    variance_adaptive: bool,
) -> Result<(), Box<dyn Error>> {
    let split = run_prepare(
        stream,
        module,
        rows,
        cols,
        false,
        nesterov,
        variance_adaptive,
    )?;
    let cooperative = run_prepare(
        stream,
        module,
        rows,
        cols,
        true,
        nesterov,
        variance_adaptive,
    )?;
    assert_eq!(split.momentum, cooperative.momentum);
    assert_eq!(split.variance, cooperative.variance);
    assert_eq!(split.oriented, cooperative.oriented);
    assert_eq!(split.polar_x, cooperative.polar_x);
    assert_eq!(split.polar_chunks, cooperative.polar_chunks);
    for (chunk, &got) in split.polar_x_chunk_amax.iter().enumerate() {
        let start = chunk * 2048;
        let end = (start + 2048).min(split.polar_x.len());
        let expected = split.polar_x[start..end]
            .iter()
            .fold(0.0_f32, |amax, value| amax.max(value.abs()));
        assert_eq!(got, expected, "chunk {chunk}");
    }
    Ok(())
}

struct PrepareOutput {
    momentum: Vec<f32>,
    variance: Vec<u16>,
    oriented: Vec<f32>,
    polar_x: Vec<f32>,
    polar_chunks: Vec<f32>,
    polar_x_chunk_amax: Vec<f32>,
}

fn run_prepare(
    stream: &CudaStream,
    module: &OptimizerModule,
    rows: usize,
    cols: usize,
    cooperative: bool,
    nesterov: bool,
    variance_adaptive: bool,
) -> Result<PrepareOutput, Box<dyn Error>> {
    let len = rows * cols;
    let grad_values: Vec<_> = (0..len)
        .map(|i| ((i * 73 % 257) as f32 - 128.0) / 211.0)
        .collect();
    let momentum_values: Vec<_> = (0..len)
        .map(|i| ((i * 29 % 131) as f32 - 65.0) / 173.0)
        .collect();
    let variance_values: Vec<_> = (0..len)
        .map(|i| 0.01 + (i * 17 % 97) as f32 / 503.0)
        .collect();
    let variance_bits = common::f32_slice_to_bf16_bits(&variance_values);
    let stored_variance_values = common::bf16_bits_slice_to_f32(&variance_bits);
    let grad = DeviceBuffer::from_host(stream, &grad_values)?;
    let momentum = DeviceBuffer::from_host(stream, &momentum_values)?;
    let variance = DeviceBuffer::from_host(stream, &variance_bits)?;
    let z_master = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let x_master = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let schedule_amax = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let bytes = DeviceBuffer::<u8>::zeroed(stream, 1)?;
    let scales = DeviceBuffer::<u8>::zeroed(stream, 1)?;
    let global_scale = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let descriptor = MuonSlotDescriptor {
        grad: grad.cu_deviceptr(),
        momentum: momentum.cu_deviceptr(),
        variance: variance.cu_deviceptr(),
        second_momentum: 0,
        z_master: z_master.cu_deviceptr(),
        x_master: x_master.cu_deviceptr(),
        schedule_amax: schedule_amax.cu_deviceptr(),
        bytes: bytes.cu_deviceptr(),
        scales: scales.cu_deviceptr(),
        global_scale: global_scale.cu_deviceptr(),
        rows: rows as u32,
        cols: cols as u32,
        learning_rate_multiplier: 1.0,
        qk_clip_factor_offset: u32::MAX,
        qk_clip_head_dim: 0,
    };
    let slots = DeviceBuffer::from_host(stream, &[descriptor])?;
    let mut oriented = DeviceBuffer::<f32>::zeroed(stream, len)?;
    let mut polar_x = DeviceBuffer::<f32>::zeroed(stream, len)?;
    let mut polar_chunks = DeviceBuffer::<f32>::zeroed(stream, 2 * MUON_COOPERATIVE_BLOCKS)?;
    let mut polar_x_chunk_amax = DeviceBuffer::<f32>::zeroed(stream, len.div_ceil(2048))?;
    let args = MuonTmaPrepareArgs {
        stream,
        slots: &slots,
        oriented: &mut oriented,
        polar_x: &mut polar_x,
        polar_chunks: &mut polar_chunks,
        polar_x_chunk_amax: &mut polar_x_chunk_amax,
        slot_index: 0,
        matrix_len: len as u32,
        mu: 0.9,
        grad_scale: 0.125,
        nesterov: nesterov as u32,
        variance_adaptive: variance_adaptive as u32,
        bias_correction_inv: if variance_adaptive {
            1.0 / (1.0 - 0.9_f32.powi(3))
        } else {
            1.0
        },
    };
    if cooperative {
        module.muon_tma_prepare_polar_cooperative_reference(args)?;
    } else {
        module.muon_tma_prepare_polar(args)?;
    }
    let actual_momentum = momentum.to_host_vec(stream)?;
    let actual_variance = variance.to_host_vec(stream)?;
    let actual_oriented = oriented.to_host_vec(stream)?;
    let mut expected_momentum = vec![0.0_f32; len];
    let mut expected_variance = stored_variance_values.clone();
    let mut expected_oriented = vec![0.0_f32; len];
    let bias_correction_inv = 1.0 / (1.0 - 0.9_f32.powi(3));
    for index in 0..len {
        let g = grad_values[index] * 0.125;
        let next = 0.9 * momentum_values[index] + 0.1 * g;
        let update = if variance_adaptive {
            let innovation = momentum_values[index] - g;
            let next_variance =
                0.9 * stored_variance_values[index] + 0.9 * 0.1 * innovation * innovation;
            expected_variance[index] =
                common::bf16_bits_to_f32(common::f32_to_bf16_bits(next_variance));
            let corrected_momentum = next * bias_correction_inv;
            let numerator = if nesterov {
                g + 9.0 * corrected_momentum
            } else {
                corrected_momentum
            };
            numerator / ((next_variance * bias_correction_inv).sqrt() + 1.0e-8)
        } else if nesterov {
            0.9 * next + 0.1 * g
        } else {
            next
        };
        expected_momentum[index] = next;
        let row = index / cols;
        let col = index % cols;
        let dst = if rows > cols { col * rows + row } else { index };
        expected_oriented[dst] = update;
    }
    common::assert_slice_close(&actual_momentum, &expected_momentum, 1.0e-6);
    common::assert_slice_close(
        &common::bf16_bits_slice_to_f32(&actual_variance),
        &expected_variance,
        0.0,
    );
    common::assert_slice_close(
        &actual_oriented,
        &expected_oriented,
        if variance_adaptive { 5.0e-5 } else { 1.0e-6 },
    );

    Ok(PrepareOutput {
        momentum: actual_momentum,
        variance: actual_variance,
        oriented: actual_oriented,
        polar_x: polar_x.to_host_vec(stream)?,
        polar_chunks: polar_chunks.to_host_vec(stream)?,
        polar_x_chunk_amax: polar_x_chunk_amax.to_host_vec(stream)?,
    })
}
