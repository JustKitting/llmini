use cuda_device::thread;

use crate::attention::CausalAttentionParams;

pub(crate) const KDA_MAX_HEAD_DIM: usize = 64;
pub(crate) const KDA_STATE_ELEMS: usize = KDA_MAX_HEAD_DIM * KDA_MAX_HEAD_DIM;
pub(crate) const KDA_DENOM_EPS: f32 = 1.0e-6;

#[inline(always)]
pub(crate) fn kda_tc_shape(params: &CausalAttentionParams) -> bool {
    params.head_dim == KDA_MAX_HEAD_DIM as u32 && params.chunk_size == KDA_MAX_HEAD_DIM as u32
}

#[inline(always)]
pub(crate) fn compact_elems(params: &CausalAttentionParams) -> u32 {
    params.batch_size * params.head_count * params.seq_len * params.head_dim
}

#[inline(always)]
pub(crate) fn batch_head(params: &CausalAttentionParams) -> u32 {
    params.batch_size * params.head_count
}

#[inline(always)]
pub(crate) fn chunk_count(params: &CausalAttentionParams) -> u32 {
    params.seq_len.div_ceil(params.chunk_size)
}

#[inline(always)]
pub(crate) fn chunk_end_token(chunk: u32, params: &CausalAttentionParams) -> u32 {
    params.seq_len.min((chunk + 1) * params.chunk_size) - 1
}

#[inline(always)]
pub(crate) fn chunk_matrix_elems(params: &CausalAttentionParams) -> u32 {
    params.chunk_size * params.chunk_size
}

#[inline(always)]
pub(crate) fn state_elems(params: &CausalAttentionParams) -> u32 {
    params.head_dim * params.head_dim
}

#[inline(always)]
pub(crate) fn linear_thread_index(threads_per_block: u32, total: u32) -> Option<u32> {
    let index = thread::blockIdx_x() * threads_per_block + thread::threadIdx_x();
    if index < total { Some(index) } else { None }
}
