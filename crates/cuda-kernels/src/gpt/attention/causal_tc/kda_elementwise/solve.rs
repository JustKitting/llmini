use cuda_device::{DisjointSlice, SharedArray, thread};

use super::super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::float_ptx::fma_f32;
use crate::kda_common::{
    KDA_MATRIX_ELEMS, KDA_MAX_HEAD_DIM, batch_head, chunk_count, chunk_matrix_elems,
};

pub(in super::super) fn solve_akk_inv_body(
    mut akk: DisjointSlice<f32>,
    params: CausalAttentionParams,
    raw: &mut SharedArray<f32, KDA_MATRIX_ELEMS>,
) {
    let batch_chunk = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    let chunks = chunk_count(&params);
    let matrix_elems = chunk_matrix_elems(&params);
    if batch_chunk >= batch_head(&params) * chunks || params.chunk_size > KDA_MAX_HEAD_DIM as u32 {
        return;
    }

    let base = (batch_chunk * matrix_elems) as usize;
    let mut idx = tid;
    while idx < matrix_elems {
        raw[idx as usize] = unsafe { *akk.get_unchecked_mut(base + idx as usize) };
        idx += TC_FORWARD_THREADS_PER_BLOCK;
    }
    thread::sync_threads();

    let col = tid;
    if col < params.chunk_size {
        let mut row = 0;
        while row < params.chunk_size {
            let value = if col > row {
                0.0
            } else if col == row {
                1.0
            } else {
                let mut sum = 0.0;
                let mut mid = col;
                while mid < row {
                    sum = fma_f32(
                        raw[(row * params.chunk_size + mid) as usize],
                        raw[(col * params.chunk_size + mid) as usize],
                        sum,
                    );
                    mid += 1;
                }
                -sum
            };
            let matrix_index = row * params.chunk_size + col;
            if col <= row {
                raw[(col * params.chunk_size + row) as usize] = value;
            }
            unsafe {
                *akk.get_unchecked_mut(base + matrix_index as usize) = value;
            }
            row += 1;
        }
    }
}
