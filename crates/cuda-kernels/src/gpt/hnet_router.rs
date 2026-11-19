//! Cosine boundary routing for an H-Net stage (arXiv:2507.07955, §2.2).
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};

use crate::{
    f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32},
    float_ptx::{abs_f32, fma_f32, max_f32, sqrt_f32},
    warp_reduce::warp_sum_f32,
};

const NORM_EPS: f32 = 1.0e-12;

#[inline(always)]
fn ratio_loss(selected: f32, mean_probability: f32, target_ratio: f32) -> (f32, f32) {
    let n_minus_one = target_ratio - 1.0;
    let weight = target_ratio / n_minus_one;
    let unselected = 1.0 - selected;
    let loss = weight
        * (n_minus_one * selected * mean_probability + unselected * (1.0 - mean_probability));
    (loss, weight * (target_ratio * selected - 1.0))
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[inline(always)]
    fn vector_pair(
        q: &[u16],
        k: &[u16],
        query_offset: usize,
        key_offset: usize,
        lane: usize,
        width: usize,
    ) -> (f32, f32, f32) {
        let mut dot = 0.0;
        let mut q_sq = 0.0;
        let mut k_sq = 0.0;
        for feature in (lane..width).step_by(32) {
            let x = cvt_f32_f16(q[query_offset + feature]);
            let y = cvt_f32_f16(k[key_offset + feature]);
            dot = fma_f32(x, y, dot);
            q_sq = fma_f32(x, x, q_sq);
            k_sq = fma_f32(y, y, k_sq);
        }
        let q_norm = sqrt_f32(warp_sum_f32(q_sq));
        let k_norm = sqrt_f32(warp_sum_f32(k_sq));
        let cosine = warp_sum_f32(dot) / (max_f32(q_norm, NORM_EPS) * max_f32(k_norm, NORM_EPS));
        (cosine, q_norm, k_norm)
    }

    /// One warp per row. The caller supplies represented outputs of the two
    /// NVFP4 router projections. The released router pairs the previous
    /// query with the current key and forces the first probability to 1.
    #[kernel]
    pub fn hnet_route_cosine_kernel(
        q: &[u16],
        k: &[u16],
        lengths: &[u32],
        mut probability: DisjointSlice<u16>,
        batch: u32,
        stride: u32,
        width: u32,
    ) {
        let thread = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let lane = warp::lane_id() as usize;
        let row = (thread / 32) as usize;
        let stride = stride as usize;
        if row >= batch as usize * stride {
            return;
        }
        let sequence = row / stride;
        let position = row % stride;
        let p = if position >= lengths[sequence] as usize {
            0.0
        } else if position == 0 {
            1.0
        } else {
            let width = width as usize;
            let (cosine, _, _) = vector_pair(q, k, (row - 1) * width, row * width, lane, width);
            let cosine = if cosine > 1.0 {
                1.0
            } else if cosine < -1.0 {
                -1.0
            } else {
                cosine
            };
            0.5 * (1.0 - cosine)
        };
        if lane == 0 {
            unsafe { *probability.get_unchecked_mut(row) = cvt_rn_f16_f32(p) };
        }
    }

    /// Each routing probability consumes one K row and its predecessor's Q.
    /// Each projection row has at most one owner, so reverse needs no atomics.
    #[kernel]
    pub fn hnet_route_cosine_backward_kernel(
        q: &[u16],
        k: &[u16],
        lengths: &[u32],
        d_probability: &[f32],
        ratio_gradient: &[f32],
        mut d_q: DisjointSlice<f32>,
        mut d_k: DisjointSlice<f32>,
        batch: u32,
        stride: u32,
        width: u32,
    ) {
        let thread = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let lane = warp::lane_id() as usize;
        let row = (thread / 32) as usize;
        let stride = stride as usize;
        let width = width as usize;
        if row >= batch as usize * stride {
            return;
        }
        let sequence = row / stride;
        let position = row % stride;
        let valid = (lengths[sequence] as usize).min(stride);
        let current = row * width;
        if position != 0 && position < valid {
            let previous = (row - 1) * width;
            let (cosine, q_norm, k_norm) = vector_pair(q, k, previous, current, lane, width);
            let q_inv = 1.0 / max_f32(q_norm, NORM_EPS);
            let k_inv = 1.0 / max_f32(k_norm, NORM_EPS);
            let scale = if abs_f32(cosine) > 1.0 {
                0.0
            } else {
                -0.5 * (d_probability[row] + ratio_gradient[0])
            };
            let q_correction = if q_norm > NORM_EPS {
                cosine * q_inv
            } else {
                0.0
            };
            let k_correction = if k_norm > NORM_EPS {
                cosine * k_inv
            } else {
                0.0
            };
            for feature in (lane..width).step_by(32) {
                let x = cvt_f32_f16(q[previous + feature]) * q_inv;
                let y = cvt_f32_f16(k[current + feature]) * k_inv;
                unsafe {
                    *d_q.get_unchecked_mut(previous + feature) =
                        scale * (y * q_inv - x * q_correction);
                    *d_k.get_unchecked_mut(current + feature) =
                        scale * (x * k_inv - y * k_correction);
                }
            }
        } else {
            for feature in (lane..width).step_by(32) {
                unsafe { *d_k.get_unchecked_mut(current + feature) = 0.0 };
            }
        }
        if position + 1 >= valid {
            for feature in (lane..width).step_by(32) {
                unsafe { *d_q.get_unchecked_mut(current + feature) = 0.0 };
            }
        }
    }

    /// One warp computes valid-row means and the weighted per-row adjoint.
    /// The hard routing fraction is treated as fixed.
    #[kernel]
    pub fn hnet_ratio_loss_kernel(
        probability: &[u16],
        lengths: &[u32],
        counts: &[u32],
        mut loss: DisjointSlice<f32>,
        mut gradient: DisjointSlice<f32>,
        batch: u32,
        stride: u32,
        target_ratio: f32,
        weight: f32,
    ) {
        if thread::blockIdx_x() != 0 || thread::threadIdx_x() >= 32 {
            return;
        }
        let lane = warp::lane_id() as usize;
        let stride = stride as usize;
        let mut p_sum = 0.0;
        let mut valid = 0.0;
        let mut retained = 0.0;
        for sequence in 0..batch as usize {
            let length = (lengths[sequence] as usize).min(stride);
            for position in (lane..length).step_by(32) {
                p_sum += cvt_f32_f16(probability[sequence * stride + position]);
            }
            if lane == 0 {
                valid += length as f32;
                retained += counts[sequence] as f32;
            }
        }
        let p_sum = warp_sum_f32(p_sum);
        if lane == 0 {
            let (value, derivative) = if valid == 0.0 {
                (0.0, 0.0)
            } else {
                let (value, derivative) = ratio_loss(retained / valid, p_sum / valid, target_ratio);
                (weight * value, weight * derivative / valid)
            };
            unsafe {
                *loss.get_unchecked_mut(0) = value;
                *gradient.get_unchecked_mut(0) = derivative;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ratio_loss;

    #[test]
    fn held_boundary_fraction_gives_the_expected_probability_adjoint() {
        assert_eq!(ratio_loss(0.25, 0.25, 4.0), (1.0, 0.0));
        for retained in [0.125, 0.5, 1.0] {
            let p = 0.4;
            let (_, derivative) = ratio_loss(retained, p, 4.0);
            let difference = (ratio_loss(retained, p + 0.001, 4.0).0
                - ratio_loss(retained, p - 0.001, 4.0).0)
                / 0.002;
            assert!((difference - derivative).abs() < 0.001);
            assert_eq!(derivative.is_sign_positive(), retained > 0.25);
        }
    }
}
