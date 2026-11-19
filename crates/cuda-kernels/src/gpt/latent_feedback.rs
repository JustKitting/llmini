//! Causal RMS input of latent feedback. Dense products use shared NVFP4 TMA.
use crate::{
    block_reduce::block_sum_shared_f32, f16_tc_matmul::convert::load_f32x2_global,
    float_ptx::sqrt_f32, nvfp4_quant::kernels::four_six::module::pack_cached_rows,
};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

pub const FUSED_ROW_WIDTH: usize = 2048;

#[cuda_module]
pub mod kernels {
    use super::*;

    static mut REDUCE: SharedArray<f32, 8> = SharedArray::UNINIT;

    #[inline(always)]
    fn rms_inverse(top: &[f32], base: u32, width: u32, tid: u32, epsilon: f32) -> f32 {
        let mut square = 0.0;
        let mut column = tid;
        while column < width {
            let value = top[(base + column) as usize];
            square += value * value;
            column += 256;
        }
        let square = unsafe { block_sum_shared_f32(&mut REDUCE, square, tid % 32, tid / 32) };
        1.0 / sqrt_f32(square / width as f32 + epsilon)
    }

    #[kernel]
    pub fn latent_feedback_shifted_rms_kernel(
        top: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        d: u32,
        seq: u32,
        epsilon: f32,
    ) {
        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if row >= n {
            return;
        }
        if row % seq == 0 {
            let mut column = tid;
            while column < d {
                unsafe { *out.get_unchecked_mut((row * d + column) as usize) = 0.0 };
                column += 256;
            }
            return;
        }
        let base = (row - 1) * d;
        let inverse = rms_inverse(top, base, d, tid, epsilon);
        let mut column = tid;
        while column < d {
            unsafe {
                *out.get_unchecked_mut((row * d + column) as usize) =
                    top[(base + column) as usize] * inverse;
            }
            column += 256;
        }
    }

    #[kernel]
    pub fn latent_feedback_shifted_rms_pack_kernel(
        top: &[f32],
        payload: DisjointSlice<u8>,
        scales: DisjointSlice<u8>,
        global: DisjointSlice<f32>,
        n: u32,
        seq: u32,
        epsilon: f32,
    ) {
        let row = thread::blockIdx_x();
        if row >= n {
            return;
        }
        let starts_sequence = row % seq == 0;
        let base = row.saturating_sub(1) * FUSED_ROW_WIDTH as u32;
        let inverse = if starts_sequence {
            0.0
        } else {
            rms_inverse(
                top,
                base,
                FUSED_ROW_WIDTH as u32,
                thread::threadIdx_x(),
                epsilon,
            )
        };
        unsafe {
            pack_cached_rows::<FUSED_ROW_WIDTH, 2>(
                |_, column| {
                    if starts_sequence {
                        [0.0; 4]
                    } else {
                        let index = base as usize + column;
                        let (a, b) = load_f32x2_global(top.as_ptr(), index);
                        let (c, d) = load_f32x2_global(top.as_ptr(), index + 2);
                        [a * inverse, b * inverse, c * inverse, d * inverse]
                    }
                },
                None,
                payload,
                scales,
                global,
                n,
                1.0,
                &mut REDUCE,
            );
        }
    }
}
