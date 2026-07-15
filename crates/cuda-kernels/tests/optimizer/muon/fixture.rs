use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::optimizer::{MuonMegaUpdateArgs, OptimizerModule};

use crate::assertions::assert_update_matches;
use crate::common;
use crate::polar_reference::polar_first_iteration_scalar;

use super::buffers::{SLOT_COUNT, Scratch, Slots, assert_quantized_slot_matches, descriptors};

pub fn run_first_iteration_case(row_count: usize, col_count: usize) -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let len = row_count * col_count;
    let gram_dim = row_count.min(col_count);
    let mut slots = Slots::with_repeated_grad(&stream, GRAD_VALUE, len)?;
    let mut scratch = Scratch::new(&stream, len, gram_dim)?;
    let slot_descriptors = descriptors(&slots, row_count, col_count);
    let slot_descriptors = DeviceBuffer::from_host(&stream, &slot_descriptors)?;

    module.muon_mega_update(MuonMegaUpdateArgs {
        stream: &stream,
        slots: &slot_descriptors,
        oriented: &mut scratch.oriented,
        polar_next: &mut scratch.polar_next,
        polar_x: &mut scratch.polar_x,
        polar_gram: &mut scratch.polar_gram,
        polar_ax: &mut scratch.polar_ax,
        polar_chunks: &mut scratch.polar_chunks,
        slot_count: SLOT_COUNT as u32,
        max_len: len as u32,
        max_ax_len: len as u32,
        max_dim: gram_dim as u32,
        mu: MU,
        grad_scale: GRAD_SCALE,
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        average_coefficient: 1.0,
        iterations: 1,
    })?;

    let expected = expected_update(row_count, col_count);
    assert_update_matches(
        &stream,
        slots.x_masters.remove(0),
        slots.z_masters.remove(0),
        expected,
    )?;
    assert_quantized_slot_matches(&stream, slots, expected)
}

const GRAD_VALUE: f32 = 0.5;
const GRAD_SCALE: f32 = 0.5;
const LEARNING_RATE: f32 = 0.25;
const MU: f32 = 0.95;
const WEIGHT_DECAY: f32 = 0.1;

fn expected_update(rows: usize, cols: usize) -> f32 {
    let nesterov = (1.0 - MU) * (1.0 + MU) * GRAD_VALUE * GRAD_SCALE;
    let update = polar_first_iteration_scalar(nesterov, rows, cols);
    let scale = 0.2 * (rows.max(cols) as f32).sqrt();
    (1.0 - LEARNING_RATE * WEIGHT_DECAY) - LEARNING_RATE * update * scale
}
