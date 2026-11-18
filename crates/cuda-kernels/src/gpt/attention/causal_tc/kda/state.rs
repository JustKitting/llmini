mod exact;
mod phase;
pub(crate) use exact::{exact_kda_state_body, exact_kda_state_carry_body};

use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::kda_common::KDA_MAX_HEAD_DIM;
use crate::kda_tc::{CtaTiles, KdaDecayTile, KdaStateSaveTile};

use phase::{compute_kg_vnew_add_state, compute_ws_to_vnew, decay_state};

pub(in super::super) const KDA_STATE_COLUMN_PARTITIONS: u32 = 2;
pub(super) const KDA_STATE_PARTITION_COLS: u32 = 32;
pub(super) const KDA_STATE_DIM: u32 = KDA_MAX_HEAD_DIM as u32;
pub(super) const KDA_STATE_SHARED_STRIDE: u32 = KDA_STATE_DIM + 7;
const KDA_STATE_ELEMS: u32 = KDA_STATE_DIM * KDA_STATE_DIM;

#[derive(Clone, Copy)]
pub(super) struct KdaStateCtx {
    tile: CtaTile,
    bh: u32,
    start: u32,
    end: u32,
    seq_len: u32,
    chunks: u32,
}

impl KdaStateCtx {
    #[inline(always)]
    pub(super) fn compact_index(self, token: u32, dim: u32) -> usize {
        ((self.bh * self.seq_len + token) * KDA_STATE_DIM + dim) as usize
    }

    #[inline(always)]
    pub(super) fn chunk_g_last_index(self, chunk: u32, dim: u32) -> usize {
        ((self.bh * self.chunks + chunk) * KDA_STATE_DIM + dim) as usize
    }
}

#[derive(Clone, Copy)]
pub(in super::super) struct KdaStateSaveInputs<'a> {
    pub(in super::super) kg: &'a [u16],
    pub(in super::super) w: &'a [u16],
    pub(in super::super) u: &'a [f32],
    pub(in super::super) chunk_g_last: &'a [f32],
}

pub(in super::super) fn chunk_kda_state_save_body(
    inputs: KdaStateSaveInputs<'_>,
    mut v_new: DisjointSlice<u16>,
    mut chunk_states: DisjointSlice<u16>,
    params: CausalAttentionParams,
    state: &mut KdaStateSaveTile,
    decay: &mut KdaDecayTile,
    vnew_tile: &mut crate::kda_tc::KdaVNewTile,
    tiles: CtaTiles<'_>,
) {
    let (a_tile, b_tile) = tiles;
    let bh = thread::blockIdx_x();
    let column_partition = thread::blockIdx_y();
    let tid = thread::threadIdx_x();
    if bh >= params.batch_size * params.head_count
        || column_partition >= KDA_STATE_COLUMN_PARTITIONS
        || params.head_dim != KDA_STATE_DIM
        || params.chunk_size != KDA_STATE_DIM
    {
        return;
    }

    let seq_len = params.seq_len;
    let chunks = seq_len.div_ceil(KDA_STATE_DIM);
    let column_start = column_partition * KDA_STATE_PARTITION_COLS;
    let partition_elems = KDA_STATE_DIM * KDA_STATE_PARTITION_COLS;
    if chunks == 0 {
        return;
    }
    let kg = inputs.kg.as_ptr();
    let w = inputs.w.as_ptr();
    let u = inputs.u.as_ptr();
    let chunk_g_last = inputs.chunk_g_last.as_ptr();
    let v_new = v_new.as_mut_ptr();
    let chunk_states = chunk_states.as_mut_ptr();
    let first_state_base = (bh * chunks * KDA_STATE_ELEMS) as usize;
    let mut linear = tid;
    while linear < partition_elems {
        let k_dim = linear / KDA_STATE_PARTITION_COLS;
        let v_dim = column_start + linear % KDA_STATE_PARTITION_COLS;
        let state_index = k_dim * KDA_STATE_DIM + v_dim;
        state[(k_dim * KDA_STATE_SHARED_STRIDE + v_dim) as usize] = 0.0;
        unsafe {
            *chunk_states.add(first_state_base + state_index as usize) = 0;
        }
        linear += thread::blockDim_x();
    }
    thread::sync_threads();

    let tile = CtaTile::from_column_partition(tid, column_start);
    let mut chunk = 0;
    while chunk < chunks {
        let start = chunk * KDA_STATE_DIM;
        let end = seq_len.min(start + KDA_STATE_DIM);
        let ctx = KdaStateCtx {
            tile,
            bh,
            start,
            end,
            seq_len,
            chunks,
        };

        if chunk != 0 {
            linear = tid;
            while linear < partition_elems {
                let k_dim = linear / KDA_STATE_PARTITION_COLS;
                let v_dim = column_start + linear % KDA_STATE_PARTITION_COLS;
                let state_index = k_dim * KDA_STATE_DIM + v_dim;
                let base = ((bh * chunks + chunk) * KDA_STATE_ELEMS) as usize;
                unsafe {
                    *chunk_states.add(base + state_index as usize) =
                        cvt_rn_f16_f32(state[(k_dim * KDA_STATE_SHARED_STRIDE + v_dim) as usize]);
                }
                linear += thread::blockDim_x();
            }
        }

        compute_ws_to_vnew(
            w,
            u,
            v_new,
            vnew_tile.as_mut_ptr(),
            state,
            a_tile,
            b_tile,
            ctx,
        );

        decay_state(state, decay, chunk_g_last, chunk, tid, ctx);
        compute_kg_vnew_add_state(kg, vnew_tile.as_mut_ptr(), state, a_tile, b_tile, ctx);
        if chunk + 1 < chunks {
            thread::sync_threads();
        }

        chunk += 1;
    }
}
