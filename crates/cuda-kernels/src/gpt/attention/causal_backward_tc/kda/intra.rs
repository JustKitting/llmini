use cuda_device::{SharedArray, thread};

mod compact;
mod prefix;
mod projection;
pub(crate) use projection::{
    project_kda_intra_tiles, stage_backprojection, stage_chain_native, stage_chain_shared,
    stage_chain_vbeta, store_backprojection, store_chain_half,
};

use compact::update_compact_grads;
use prefix::reverse_prefix_dg;

use crate::attention::CausalAttentionParams;
use crate::float_ptx::max_f32;

pub(crate) const KDA_INTRA_DG_TILE_ELEMS: usize = 64 * 64;
pub(crate) const KDA_INTRA_DKG_KG_TILE_ELEMS: usize = 64 * 64;
pub(crate) const KDA_INTRA_THREADS_PER_BLOCK: u32 = 512;
pub(crate) const KDA_INTRA_TOKEN_SLOTS: usize = (KDA_INTRA_THREADS_PER_BLOCK / 64) as usize;
pub(crate) const KDA_INTRA_PAIR_VALUE_ELEMS: usize = KDA_INTRA_TOKEN_SLOTS * 6 * 32;
pub(crate) const KDA_INTRA_PAIR_STAT_ELEMS: usize = KDA_INTRA_TOKEN_SLOTS * 4;

#[derive(Clone, Copy)]
pub(crate) struct KdaIntraInputs<'a> {
    pub(crate) qkv: &'a [u16],
    pub(crate) qg: &'a [f32],
    pub(crate) kg: &'a [f32],
    pub(crate) vbeta: &'a [f32],
    pub(crate) g: &'a [f32],
    pub(crate) beta_chunk: &'a SharedArray<f32, 64>,
    pub(crate) g_last_chunk: &'a SharedArray<f32, 64>,
    pub(crate) detached_chunk_state: u32,
    pub(crate) d_kg: &'a [f32],
    pub(crate) d_kpos_m: &'a [f32],
    pub(crate) d_vbeta_m: &'a [f32],
    pub(crate) d_qg: &'a [f32],
    pub(crate) d_k_a: &'a [f32],
}

#[derive(Clone, Copy)]
pub(crate) struct KdaIntraGrads {
    pub(crate) q_norms: *mut f32,
    pub(crate) k_norms: *mut f32,
    pub(crate) d_qkv: *mut f32,
}

#[derive(Clone, Copy)]
struct KdaIntraCtx<'a> {
    params: &'a CausalAttentionParams,
    batch: u32,
    head: u32,
    start: u32,
    compact_base: u32,
}

impl KdaIntraCtx<'_> {
    fn compact(self, token: u32, dim: u32) -> usize {
        (self.compact_base + (token - self.start) * 64 + dim) as usize
    }
}

pub(crate) fn chunk_intra_kda_backward_body(
    inputs: KdaIntraInputs<'_>,
    grads: KdaIntraGrads,
    dg_tile: &mut SharedArray<f32, KDA_INTRA_DG_TILE_ELEMS>,
    dkg_kg_tile: &mut SharedArray<f32, KDA_INTRA_DKG_KG_TILE_ELEMS>,
    pair_values: &mut SharedArray<f32, KDA_INTRA_PAIR_VALUE_ELEMS>,
    pair_stats: &mut SharedArray<f32, KDA_INTRA_PAIR_STAT_ELEMS>,
    accumulate_value_grad: u32,
    params: &CausalAttentionParams,
) -> f32 {
    // Both public KDA backward APIs enforce head_dim=64 and complete chunks;
    // their exact grid covers every batch-head and chunk once.
    let bh = thread::blockIdx_x();
    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;
    let start = thread::blockIdx_y() * 64;
    let tid = thread::threadIdx_x();
    let ctx = KdaIntraCtx {
        params,
        batch,
        head,
        start,
        compact_base: (bh * params.seq_len + start) * 64,
    };

    let mut local_amax = update_compact_grads(
        inputs,
        grads,
        dg_tile,
        dkg_kg_tile,
        pair_values,
        pair_stats,
        ctx,
        tid,
        accumulate_value_grad,
    );
    thread::sync_threads();
    local_amax = max_f32(
        local_amax,
        reverse_prefix_dg(inputs.qkv, dg_tile, dkg_kg_tile, grads.d_qkv, ctx, tid),
    );
    local_amax
}
