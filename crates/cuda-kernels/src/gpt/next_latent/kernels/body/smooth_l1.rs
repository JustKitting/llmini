use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::float_ptx::abs_f32;
use crate::layer_norm_reduce::layer_norm_block_reduce;
use crate::warp_reduce::{thread_lane_warp, warp_sum_f32};

use super::super::super::args::NextLatShape;
use super::THREADS_PER_BLOCK;

const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / WARP_SIZE;

pub(crate) fn nextlat_smooth_l1_body(
    predicted_next_states: &[f32],
    target_states: &[f32],
    losses: &mut DisjointSlice<f32>,
    d_predicted_next_states: &mut DisjointSlice<f32>,
    shape: NextLatShape,
) {
    static mut SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

    let batch = thread::blockIdx_x();
    let pos = thread::blockIdx_y();
    let row = batch * shape.seq_len + pos;
    let valid = batch < shape.batch_size && pos + 1 < shape.seq_len;
    let (thread, lane, warp_in_block) = thread_lane_warp();
    let valid_rows = shape.batch_size * (shape.seq_len - 1);
    let grad_scale = shape.lambda / (valid_rows * shape.embedding_dim) as f32;

    let mut local = 0.0;
    let mut col = thread;
    while col < shape.embedding_dim {
        let offset = (row * shape.embedding_dim + col) as usize;
        if valid {
            let target_offset = ((row + 1) * shape.embedding_dim + col) as usize;
            let diff = predicted_next_states[offset] - target_states[target_offset];
            let abs = abs_f32(diff);
            let grad = if abs < 1.0 {
                local += 0.5 * diff * diff;
                diff
            } else {
                local += abs - 0.5;
                if abs <= f32::MAX {
                    if diff < 0.0 { -1.0 } else { 1.0 }
                } else {
                    // Preserve invalid differences for steps without loss readback.
                    diff
                }
            };
            unsafe {
                *d_predicted_next_states.get_unchecked_mut(offset) = grad * grad_scale;
            }
        } else {
            unsafe {
                *d_predicted_next_states.get_unchecked_mut(offset) = 0.0;
            }
        }
        col += THREADS_PER_BLOCK;
    }

    let sum = layer_norm_block_reduce!(
        SUMS,
        WARPS_PER_BLOCK,
        local,
        lane,
        warp_in_block,
        warp_sum_f32
    );
    let row_loss = if valid {
        shape.lambda * sum / shape.embedding_dim as f32
    } else {
        0.0
    };
    if warp_in_block == 0 && lane == 0 {
        unsafe {
            *losses.get_unchecked_mut(row as usize) = row_loss;
        }
    }
}
