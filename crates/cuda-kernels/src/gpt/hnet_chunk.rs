//! H-Net causal selection, smoothing and expansion (arXiv:2507.07955, §2.2).
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};

use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::warp_reduce::warp_sum_f32;

const INVALID: u32 = u32::MAX;

#[inline(always)]
fn causal_layout(
    valid: u32,
    mut probability: impl FnMut(u32) -> f32,
    mut write: impl FnMut(u32, u32, bool),
) -> u32 {
    let mut count = 0;
    for row in 0..valid {
        // Forced start, including a start after splitting a document.
        let boundary = row == 0 || probability(row) >= 0.5;
        if boundary {
            count += 1;
        }
        write(row, count - 1, boundary);
    }
    count
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Probabilities and indices have layout [batch, outer_stride]. The count
    /// is the actual inner length, including rows whose final chunk is partial.
    #[kernel]
    pub fn hnet_causal_layout_kernel(
        probability: &[u16],
        lengths: &[u32],
        mut row_to_chunk: DisjointSlice<u32>,
        mut chunk_to_row: DisjointSlice<u32>,
        mut counts: DisjointSlice<u32>,
        batch: u32,
        outer_stride: u32,
    ) {
        let sequence = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if sequence >= batch {
            return;
        }
        let valid = lengths[sequence as usize].min(outer_stride);
        let offset = sequence as usize * outer_stride as usize;
        let count = causal_layout(
            valid,
            |row| cvt_f32_f16(probability[offset + row as usize]),
            |row, chunk, boundary| unsafe {
                *row_to_chunk.get_unchecked_mut(offset + row as usize) = chunk;
                if boundary {
                    *chunk_to_row.get_unchecked_mut(offset + chunk as usize) = row;
                }
            },
        );
        unsafe {
            *counts.get_unchecked_mut(sequence as usize) = count;
            for row in valid..outer_stride {
                *row_to_chunk.get_unchecked_mut(offset + row as usize) = INVALID;
            }
            for chunk in count..outer_stride {
                *chunk_to_row.get_unchecked_mut(offset + chunk as usize) = INVALID;
            }
        }
    }

    /// Gather selected encoder rows without re-encoding their FP16 values.
    /// inner_stride is the maximum of the returned counts for this batch.
    #[kernel]
    pub fn hnet_gather_chunk_kernel(
        encoded: &[u16],
        chunk_to_row: &[u32],
        mut chunks: DisjointSlice<u16>,
        batch: u32,
        outer_stride: u32,
        inner_stride: u32,
        features: u32,
    ) {
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let inner_stride = inner_stride as usize;
        let outer_stride = outer_stride as usize;
        let features = features as usize;
        if idx >= batch as usize * inner_stride * features {
            return;
        }
        let sequence = idx / (inner_stride * features);
        let chunk = idx / features % inner_stride;
        let row = chunk_to_row[sequence * outer_stride + chunk];
        let value = if row == INVALID {
            0
        } else {
            encoded[(sequence * outer_stride + row as usize) * features + idx % features]
        };
        unsafe { *chunks.get_unchecked_mut(idx) = value };
    }

    /// Transpose of the gather. Each selected outer row owns one chunk;
    /// the remaining outer rows receive their gradient from the residual path.
    #[kernel]
    pub fn hnet_gather_chunk_backward_kernel(
        d_chunks: &[f32],
        row_to_chunk: &[u32],
        chunk_to_row: &[u32],
        mut d_encoder: DisjointSlice<f32>,
        batch: u32,
        outer_stride: u32,
        inner_stride: u32,
        features: u32,
    ) {
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let outer_stride = outer_stride as usize;
        let features = features as usize;
        if idx >= batch as usize * outer_stride * features {
            return;
        }
        let row = idx / features;
        let sequence = row / outer_stride;
        let chunk = row_to_chunk[row];
        let gradient = if chunk != INVALID
            && chunk_to_row[sequence * outer_stride + chunk as usize] as usize == row % outer_stride
        {
            d_chunks
                [(sequence * inner_stride as usize + chunk as usize) * features + idx % features]
        } else {
            0.0
        };
        unsafe { *d_encoder.get_unchecked_mut(idx) = gradient };
    }

    /// Smooth the inner model outputs using the selected boundary probabilities.
    /// Store the represented FP16 state at every hop, as for the outer tape.
    #[kernel]
    pub fn hnet_smooth_chunk_kernel(
        inner: &[u16],
        probability: &[u16],
        chunk_to_row: &[u32],
        counts: &[u32],
        mut smoothed: DisjointSlice<u16>,
        batch: u32,
        outer_stride: u32,
        inner_stride: u32,
        features: u32,
    ) {
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let features = features as usize;
        if idx >= batch as usize * features {
            return;
        }
        let sequence = idx / features;
        let inner_stride = inner_stride as usize;
        let inner_offset = sequence * inner_stride * features + idx % features;
        let outer_offset = sequence * outer_stride as usize;
        let valid = counts[sequence] as usize;
        let mut previous = 0;
        for chunk in 0..valid {
            let index = inner_offset + chunk * features;
            let row = chunk_to_row[outer_offset + chunk] as usize;
            let value = inner[index];
            let next = if chunk == 0 {
                value
            } else {
                let p = cvt_f32_f16(probability[outer_offset + row]);
                cvt_rn_f16_f32(p * cvt_f32_f16(value) + (1.0 - p) * cvt_f32_f16(previous))
            };
            unsafe { *smoothed.get_unchecked_mut(index) = next };
            previous = next;
        }
        for chunk in valid..inner_stride {
            unsafe { *smoothed.get_unchecked_mut(inner_offset + chunk * features) = 0 };
        }
    }

    /// Repeat only the most recent causal chunk. STE confidence is one in forward.
    #[kernel]
    pub fn hnet_lift_chunk_kernel(
        smoothed: &[u16],
        row_to_chunk: &[u32],
        mut decoded: DisjointSlice<u16>,
        batch: u32,
        outer_stride: u32,
        inner_stride: u32,
        features: u32,
    ) {
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let outer_stride = outer_stride as usize;
        let features = features as usize;
        if idx >= batch as usize * outer_stride * features {
            return;
        }
        let sequence = idx / (outer_stride * features);
        let chunk = row_to_chunk[idx / features];
        let value = if chunk == INVALID {
            0
        } else {
            smoothed
                [(sequence * inner_stride as usize + chunk as usize) * features + idx % features]
        };
        unsafe { *decoded.get_unchecked_mut(idx) = value };
    }

    /// Backpropagate repeat and EMA. Each lane owns one feature and visits
    /// each outer row once. The local EMA adjoint is reused by the router.
    #[kernel]
    pub fn hnet_smooth_chunk_backward_kernel(
        d_decoder_input: &[f32],
        probability: &[u16],
        chunk_to_row: &[u32],
        counts: &[u32],
        lengths: &[u32],
        mut d_smoothed: DisjointSlice<f32>,
        mut d_inner: DisjointSlice<f32>,
        batch: u32,
        outer_stride: u32,
        inner_stride: u32,
        features: u32,
    ) {
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let features = features as usize;
        if idx >= batch as usize * features {
            return;
        }
        let sequence = idx / features;
        let inner_stride = inner_stride as usize;
        let inner_offset = sequence * inner_stride * features + idx % features;
        let outer_offset = sequence * outer_stride as usize;
        let valid = counts[sequence] as usize;
        let mut end = lengths[sequence].min(outer_stride) as usize;
        let mut carry = 0.0;
        for chunk in (0..valid).rev() {
            let row = chunk_to_row[outer_offset + chunk] as usize;
            let mut adjoint = carry;
            for pos in row..end {
                adjoint += d_decoder_input[(outer_offset + pos) * features + idx % features];
            }
            let p = if chunk == 0 {
                1.0
            } else {
                cvt_f32_f16(probability[outer_offset + row])
            };
            let index = inner_offset + chunk * features;
            unsafe {
                *d_smoothed.get_unchecked_mut(index) = adjoint;
                *d_inner.get_unchecked_mut(index) = p * adjoint;
            }
            carry = (1.0 - p) * adjoint;
            end = row;
        }
        for chunk in valid..inner_stride {
            let index = inner_offset + chunk * features;
            unsafe {
                *d_smoothed.get_unchecked_mut(index) = 0.0;
                *d_inner.get_unchecked_mut(index) = 0.0;
            }
        }
    }

    /// Probability VJP from both confidence STE and the represented EMA.
    /// Launch with a whole number of warps per block, one warp per outer row.
    #[kernel]
    pub fn hnet_lift_probability_backward_kernel(
        d_decoder_input: &[f32],
        inner: &[u16],
        smoothed: &[u16],
        d_smoothed: &[f32],
        row_to_chunk: &[u32],
        chunk_to_row: &[u32],
        mut d_probability: DisjointSlice<f32>,
        batch: u32,
        outer_stride: u32,
        inner_stride: u32,
        features: u32,
    ) {
        let thread = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let lane = warp::lane_id() as usize;
        let row = (thread / 32) as usize;
        let outer_stride = outer_stride as usize;
        if row >= batch as usize * outer_stride {
            return;
        }
        let chunk = row_to_chunk[row];
        let position = row % outer_stride;
        let mut adjoint = 0.0;
        if position != 0 && chunk != INVALID {
            let sequence = row / outer_stride;
            let chunk = chunk as usize;
            let is_boundary = chunk_to_row[sequence * outer_stride + chunk] as usize == position;
            let features = features as usize;
            let chunk_offset = (sequence * inner_stride as usize + chunk) * features;
            let direction = if is_boundary { 1.0 } else { -1.0 };
            for feature in (lane..features).step_by(32) {
                let index = chunk_offset + feature;
                adjoint += direction
                    * d_decoder_input[row * features + feature]
                    * cvt_f32_f16(smoothed[index]);
                if is_boundary {
                    let previous = cvt_f32_f16(smoothed[index - features]);
                    adjoint += d_smoothed[index] * (cvt_f32_f16(inner[index]) - previous);
                }
            }
        }
        let adjoint = warp_sum_f32(adjoint);
        if lane == 0 {
            unsafe { *d_probability.get_unchecked_mut(row) = adjoint };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping(p: &[f32]) -> (Vec<u32>, Vec<u32>) {
        let mut selection = Vec::new();
        let mut lift = Vec::new();
        let count = causal_layout(
            p.len() as u32,
            |row| p[row as usize],
            |row, chunk, boundary| {
                lift.push(chunk);
                if boundary {
                    selection.push(row);
                }
            },
        );
        assert_eq!(selection.len(), count as usize);
        (selection, lift)
    }

    #[test]
    fn causal_selection_retains_first_and_threshold_ties() {
        let p = [0.0, 0.4, 0.5, 0.1, 0.9];
        assert_eq!(mapping(&p), (vec![0, 2, 4], vec![0, 0, 1, 1, 2]));
        assert_eq!(mapping(&[]), (vec![], vec![]));
        assert_eq!(mapping(&[0.2]), (vec![0], vec![0]));
    }

    #[test]
    fn later_probabilities_cannot_change_an_earlier_row() {
        let p = [0.0, 0.51, 0.1, 0.1, 0.51, 0.2];
        let (full, lift) = mapping(&p);
        for prefix in 0..=p.len() {
            let (selected, prefix_lift) = mapping(&p[..prefix]);
            assert_eq!(prefix_lift, lift[..prefix]);
            assert_eq!(
                selected,
                full.iter()
                    .copied()
                    .take_while(|&row| (row as usize) < prefix)
                    .collect::<Vec<_>>()
            );
        }
    }
}
