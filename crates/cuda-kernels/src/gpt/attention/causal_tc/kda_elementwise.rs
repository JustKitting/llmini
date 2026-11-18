mod mask;
mod prepare;
mod transform;

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::kda_common::linear_thread_index;

pub(super) use mask::{mask_akk_body, mask_aqk_body};
pub(super) use prepare::{
    KDA_FUSED_PREPARE_BETA_ELEMS, KDA_FUSED_PREPARE_TILE_ELEMS, prepare_transform_kda_body,
    zero_f32_body,
};
pub(super) use transform::{
    make_kg_kpos_vbeta_body, make_qg_kg_vbeta_body, make_qg_kneg_body,
    make_qg_kneg_kg_kpos_vbeta_body,
};

#[inline(always)]
fn thread_index(element_count: u32) -> Option<u32> {
    linear_thread_index(TC_FORWARD_THREADS_PER_BLOCK, element_count)
}
