use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::optimizer::{AuroraMegaUpdateArgs, OptimizerModule};

use crate::{common, polar_vector};

use super::buffers::{SLOT_COUNT, Scratch, Slots, descriptors};

const ROWS: usize = 32;
const COLS: usize = 64;
const LEN: usize = ROWS * COLS;
const MU: f32 = 0.95;
const LEARNING_RATE: f32 = 0.25;
const WEIGHT_DECAY: f32 = 0.1;
const ITERATIONS: u32 = 5;
const GRAD_SCALE: f32 = 0.5;

pub fn run_wide_case() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let grad = gradient();
    let mut slots = Slots::new(&stream, &grad)?;
    let mut scratch = Scratch::new(&stream, LEN, ROWS)?;
    let slot_descriptors = descriptors(&slots, ROWS, COLS);
    let slot_descriptors = DeviceBuffer::from_host(&stream, &slot_descriptors)?;

    module.aurora_mega_update(AuroraMegaUpdateArgs {
        stream: &stream,
        slots: &slot_descriptors,
        oriented: &mut scratch.oriented,
        polar_next: &mut scratch.polar_next,
        polar_x: &mut scratch.polar_x,
        polar_gram: &mut scratch.polar_gram,
        polar_ax: &mut scratch.polar_ax,
        polar_chunks: &mut scratch.polar_chunks,
        slot_count: SLOT_COUNT as u32,
        max_len: LEN as u32,
        max_ax_len: LEN as u32,
        max_dim: ROWS as u32,
        mu: MU,
        grad_scale: GRAD_SCALE,
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        average_coefficient: 1.0,
        iterations: ITERATIONS,
    })?;

    let scaled_grad: Vec<f32> = grad.iter().map(|value| value * GRAD_SCALE).collect();
    let expected = polar_vector::first_iteration_update(
        &scaled_grad,
        ROWS,
        COLS,
        MU,
        LEARNING_RATE,
        WEIGHT_DECAY,
        ITERATIONS as usize,
    );
    common::assert_slice_close(
        &slots.x_masters.remove(0).to_host_vec(&stream)?,
        &expected,
        2.0e-3,
    );
    common::assert_slice_close(
        &slots.z_masters.remove(0).to_host_vec(&stream)?,
        &expected,
        2.0e-3,
    );
    Ok(())
}

fn gradient() -> Vec<f32> {
    (0..LEN)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.0003 + ((i / COLS) as f32) * 0.00001)
        .collect()
}
