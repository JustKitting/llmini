use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::next_latent::{NextLatConcatArgs, NextLatModule, NextLatSmoothL1Args};

mod common;

#[path = "next_latent/reference.rs"]
mod reference;

const BATCH_SIZE: usize = 2;
const SEQ_LEN: usize = 3;
const ROW_COUNT: usize = BATCH_SIZE * SEQ_LEN;
const EMBED: usize = 4;
const LAMBDA: f32 = 2.0;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nextlat_concat_and_shifted_smooth_l1_match_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(NextLatModule::from_module)?;

    let next_token_embeddings = reference::values(0.25);
    let current_states = reference::values(-0.5);
    let predicted = reference::values(0.125);
    let losses = vec![1.0_f32; ROW_COUNT];

    let next_token_dev = DeviceBuffer::from_host(&stream, &next_token_embeddings)?;
    let current_dev = DeviceBuffer::from_host(&stream, &current_states)?;
    let predicted_dev = DeviceBuffer::from_host(&stream, &predicted)?;
    let mut concat_dev = DeviceBuffer::<f32>::zeroed(&stream, ROW_COUNT * EMBED * 2)?;
    let mut losses_dev = DeviceBuffer::from_host(&stream, &losses)?;
    let mut d_pred_dev = DeviceBuffer::<f32>::zeroed(&stream, ROW_COUNT * EMBED)?;

    module.concat_input(NextLatConcatArgs {
        stream: &stream,
        next_token_embeddings: &next_token_dev,
        current_states: &current_dev,
        out: &mut concat_dev,
        row_count: ROW_COUNT as u32,
        embedding_dim: EMBED as u32,
    })?;
    module.smooth_l1(NextLatSmoothL1Args {
        stream: &stream,
        predicted_next_states: &predicted_dev,
        target_states: &current_dev,
        losses: &mut losses_dev,
        d_predicted_next_states: &mut d_pred_dev,
        batch_size: BATCH_SIZE as u32,
        seq_len: SEQ_LEN as u32,
        embedding_dim: EMBED as u32,
        lambda: LAMBDA,
    })?;

    let expected_concat = reference::concat(&next_token_embeddings, &current_states);
    let (expected_losses, expected_grad) = reference::smooth_l1(&predicted, &current_states);

    common::assert_slice_close(
        &concat_dev.to_host_vec(&stream)?,
        &expected_concat,
        TOLERANCE,
    );
    common::assert_slice_close(
        &losses_dev.to_host_vec(&stream)?,
        &expected_losses,
        TOLERANCE,
    );
    common::assert_slice_close(&d_pred_dev.to_host_vec(&stream)?, &expected_grad, TOLERANCE);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nextlat_smooth_l1_preserves_nonfinite_gradient_signal() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(NextLatModule::from_module)?;
    let mut predicted = [0.0_f32; ROW_COUNT * EMBED];
    let mut target = [0.0_f32; ROW_COUNT * EMBED];
    predicted[..EMBED].copy_from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX]);
    target[EMBED + 3] = -f32::MAX;
    predicted[EMBED..2 * EMBED].copy_from_slice(&[0.25, -0.5, 2.0, -2.0]);
    predicted[2 * EMBED..3 * EMBED].fill(f32::NAN);
    target[4 * EMBED..5 * EMBED].copy_from_slice(&[
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
    ]);
    predicted[5 * EMBED..6 * EMBED].fill(f32::INFINITY);

    let predicted = DeviceBuffer::from_host(&stream, &predicted)?;
    let target = DeviceBuffer::from_host(&stream, &target)?;
    let mut losses = DeviceBuffer::<f32>::zeroed(&stream, ROW_COUNT)?;
    let mut gradient = DeviceBuffer::<f32>::zeroed(&stream, ROW_COUNT * EMBED)?;
    module.smooth_l1(NextLatSmoothL1Args {
        stream: &stream,
        predicted_next_states: &predicted,
        target_states: &target,
        losses: &mut losses,
        d_predicted_next_states: &mut gradient,
        batch_size: BATCH_SIZE as u32,
        seq_len: SEQ_LEN as u32,
        embedding_dim: EMBED as u32,
        lambda: LAMBDA,
    })?;

    let losses = losses.to_host_vec(&stream)?;
    let gradient = gradient.to_host_vec(&stream)?;
    let rows = gradient.chunks_exact(EMBED).collect::<Vec<_>>();
    assert!(rows[0][0].is_nan() && losses[0].is_nan());
    assert_eq!(
        rows[0][1..],
        [f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY]
    );
    assert_eq!(rows[1], [0.03125, -0.0625, 0.125, -0.125]);
    assert_eq!(losses[1], 1.578125);
    assert!(rows[3][0].is_nan() && losses[3].is_nan());
    assert_eq!(rows[3][1..], [f32::NEG_INFINITY, f32::INFINITY, 0.0]);
    for row in [2, 5] {
        assert_eq!(rows[row], [0.0; EMBED]);
        assert_eq!(losses[row], 0.0);
    }
    Ok(())
}
