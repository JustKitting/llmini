use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::optimizer::{EmberUpdateArgs, OptimizerModule, ember_column_partial_len};

mod common;

use common::{assert_slice_close, cuda_test_module};

const ROWS: usize = 4;
const COLS: usize = 32;
const LEN: usize = ROWS * COLS;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn ember_matches_factored_second_moment_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = cuda_test_module(OptimizerModule::from_module)?;
    let grad = (0..LEN)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            0.01 * (row + 1) as f32 * (col + 2) as f32
        })
        .collect::<Vec<_>>();
    let mut z_master = DeviceBuffer::from_host(&stream, &vec![1.0_f32; LEN])?;
    let mut x_master = DeviceBuffer::from_host(&stream, &vec![1.0_f32; LEN])?;
    let grad_device = DeviceBuffer::from_host(&stream, &grad)?;
    let mut row_second_moment = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    let mut column_second_moment = DeviceBuffer::<f32>::zeroed(&stream, COLS)?;
    let mut column_partials =
        DeviceBuffer::<f32>::zeroed(&stream, ember_column_partial_len(ROWS as u32, COLS as u32))?;
    let mut normalizer = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    let grad_scale = 0.5;
    let beta2 = 0.5;
    let beta2_correction = 0.5;
    let learning_rate = 0.01;
    let weight_decay = 0.1;
    let average_coefficient = 0.25;
    module.apply_ember_update(EmberUpdateArgs {
        stream: &stream,
        z_master: &mut z_master,
        x_master: &mut x_master,
        grad: &grad_device,
        row_second_moment: &mut row_second_moment,
        column_second_moment: &mut column_second_moment,
        column_partials: &mut column_partials,
        normalizer: &mut normalizer,
        rows: ROWS as u32,
        cols: COLS as u32,
        grad_scale,
        learning_rate,
        weight_decay,
        beta2,
        beta2_correction,
        eps: 1.0e-8,
        average_coefficient,
    })?;

    let scaled = grad
        .iter()
        .map(|value| value * grad_scale)
        .collect::<Vec<_>>();
    let row_hat = (0..ROWS)
        .map(|row| {
            scaled[row * COLS..(row + 1) * COLS]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / COLS as f32
        })
        .collect::<Vec<_>>();
    let column_hat = (0..COLS)
        .map(|col| {
            (0..ROWS)
                .map(|row| {
                    let value = scaled[row * COLS + col];
                    value * value
                })
                .sum::<f32>()
                / ROWS as f32
        })
        .collect::<Vec<_>>();
    let scale = ((row_hat.iter().sum::<f32>() / ROWS as f32)
        * (column_hat.iter().sum::<f32>() / COLS as f32))
        .sqrt();
    let expected_z = (0..LEN)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            let second = row_hat[row] * column_hat[col] / scale;
            let update = scaled[index] / (second.sqrt() + 1.0e-8);
            (1.0 - learning_rate * weight_decay) - learning_rate * update
        })
        .collect::<Vec<_>>();
    let expected_x = expected_z
        .iter()
        .map(|next| 1.0 + average_coefficient * (next - 1.0))
        .collect::<Vec<_>>();
    let expected_row_state = row_hat
        .iter()
        .map(|value| value * beta2_correction)
        .collect::<Vec<_>>();
    let expected_column_state = column_hat
        .iter()
        .map(|value| value * beta2_correction)
        .collect::<Vec<_>>();

    assert_slice_close(
        &row_second_moment.to_host_vec(&stream)?,
        &expected_row_state,
        2.0e-6,
    );
    assert_slice_close(
        &column_second_moment.to_host_vec(&stream)?,
        &expected_column_state,
        2.0e-6,
    );
    assert_slice_close(&z_master.to_host_vec(&stream)?, &expected_z, 2.0e-6);
    assert_slice_close(&x_master.to_host_vec(&stream)?, &expected_x, 2.0e-6);
    Ok(())
}
