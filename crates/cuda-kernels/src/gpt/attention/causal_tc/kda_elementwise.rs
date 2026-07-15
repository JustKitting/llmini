mod chunk_g;
mod mask;
mod prepare;
mod solve;
mod transform;

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::kda_common::linear_thread_index;

pub(super) use chunk_g::{chunk_cumsum_g_body, store_chunk_g_last_body};
pub(super) use mask::{mask_akk_body, mask_aqk_body};
pub(super) use prepare::{prepare_kda_body, zero_f32_body};
pub(super) use solve::solve_akk_inv_body;
pub(super) use transform::{
    make_kg_kpos_vbeta_body, make_kneg_from_kg_body, make_qg_kg_vbeta_body, make_qg_kneg_body,
    make_qg_kneg_kg_kpos_vbeta_body,
};

#[inline(always)]
fn thread_index(element_count: u32) -> Option<u32> {
    linear_thread_index(TC_FORWARD_THREADS_PER_BLOCK, element_count)
}
