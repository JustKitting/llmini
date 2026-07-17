use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{
    AccumulateValueResidualGradArgs, AttentionModule, CaptureValueResidualArgs,
    FinishValueResidualGradArgs, InitializeValueResidualGradArgs, MixValueResidualArgs,
};

mod common;

const ROWS: usize = 3;
const EMBEDDING_DIM: usize = 4;
const QKV_DIM: usize = 20;
const VALUE_OFFSET: usize = 2 * EMBEDDING_DIM;

fn qkv_values(base: f32) -> Vec<f32> {
    (0..ROWS * QKV_DIM)
        .map(|index| base + index as f32 * 0.25)
        .collect()
}

fn value_at(values: &[f32], row: usize, col: usize) -> f32 {
    values[row * QKV_DIM + VALUE_OFFSET + col]
}

fn assert_non_value_unchanged(actual: &[f32], expected: &[f32]) {
    for row in 0..ROWS {
        for col in 0..QKV_DIM {
            if (VALUE_OFFSET..VALUE_OFFSET + EMBEDDING_DIM).contains(&col) {
                continue;
            }
            let index = row * QKV_DIM + col;
            assert_eq!(actual[index], expected[index], "index={index}");
        }
    }
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn value_residual_forward_and_backward_match_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(AttentionModule::from_module)?;

    let first_qkv = qkv_values(-3.0);
    let first_qkv_dev = DeviceBuffer::from_host(&stream, &first_qkv)?;
    let mut first_value_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * EMBEDDING_DIM)?;
    module.capture_value_residual(CaptureValueResidualArgs {
        stream: &stream,
        qkv: &first_qkv_dev,
        first_value: &mut first_value_dev,
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
    })?;

    let first_value = first_value_dev.to_host_vec(&stream)?;
    for row in 0..ROWS {
        for col in 0..EMBEDDING_DIM {
            assert_eq!(
                first_value[row * EMBEDDING_DIM + col],
                value_at(&first_qkv, row, col)
            );
        }
    }

    let current_qkv = qkv_values(11.0);
    let mut current_qkv_dev = DeviceBuffer::from_host(&stream, &current_qkv)?;
    module.mix_value_residual(MixValueResidualArgs {
        stream: &stream,
        qkv: &mut current_qkv_dev,
        first_value: &first_value_dev,
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
    })?;

    let mixed = current_qkv_dev.to_host_vec(&stream)?;
    assert_non_value_unchanged(&mixed, &current_qkv);
    for row in 0..ROWS {
        for col in 0..EMBEDDING_DIM {
            let expected =
                0.5 * value_at(&first_qkv, row, col) + 0.5 * value_at(&current_qkv, row, col);
            assert_eq!(value_at(&mixed, row, col), expected);
        }
    }

    let top_grad = qkv_values(30.0);
    let mut top_grad_dev = DeviceBuffer::from_host(&stream, &top_grad)?;
    module.initialize_value_residual_grad(InitializeValueResidualGradArgs {
        stream: &stream,
        d_qkv: &mut top_grad_dev,
        d_first_value: &mut first_value_dev,
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
    })?;
    let top_routed = top_grad_dev.to_host_vec(&stream)?;
    assert_non_value_unchanged(&top_routed, &top_grad);

    let middle_grad = qkv_values(20.0);
    let mut middle_grad_dev = DeviceBuffer::from_host(&stream, &middle_grad)?;
    module.accumulate_value_residual_grad(AccumulateValueResidualGradArgs {
        stream: &stream,
        d_qkv: &mut middle_grad_dev,
        d_first_value: &mut first_value_dev,
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
    })?;
    let middle_routed = middle_grad_dev.to_host_vec(&stream)?;
    assert_non_value_unchanged(&middle_routed, &middle_grad);

    let lower_grad = qkv_values(10.0);
    let mut lower_grad_dev = DeviceBuffer::from_host(&stream, &lower_grad)?;
    module.accumulate_value_residual_grad(AccumulateValueResidualGradArgs {
        stream: &stream,
        d_qkv: &mut lower_grad_dev,
        d_first_value: &mut first_value_dev,
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
    })?;
    let lower_routed = lower_grad_dev.to_host_vec(&stream)?;
    assert_non_value_unchanged(&lower_routed, &lower_grad);

    let bottom_grad = qkv_values(1.0);
    let mut bottom_grad_dev = DeviceBuffer::from_host(&stream, &bottom_grad)?;
    module.finish_value_residual_grad(FinishValueResidualGradArgs {
        stream: &stream,
        d_qkv: &mut bottom_grad_dev,
        d_first_value: &first_value_dev,
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
    })?;
    let bottom_routed = bottom_grad_dev.to_host_vec(&stream)?;
    assert_non_value_unchanged(&bottom_routed, &bottom_grad);

    for row in 0..ROWS {
        for col in 0..EMBEDDING_DIM {
            let top = value_at(&top_grad, row, col);
            let middle = value_at(&middle_grad, row, col);
            let lower = value_at(&lower_grad, row, col);
            let bottom = value_at(&bottom_grad, row, col);
            assert_eq!(value_at(&top_routed, row, col), 0.5 * top);
            assert_eq!(value_at(&middle_routed, row, col), 0.5 * middle);
            assert_eq!(value_at(&lower_routed, row, col), 0.5 * lower);
            assert_eq!(
                value_at(&bottom_routed, row, col),
                bottom + 0.5 * (top + middle + lower)
            );
        }
    }

    Ok(())
}
