//! Exact bounded carry with the established FP16 tensor-core W*S/Kg*r phases.
//! Compact factors are padded per64-token chunk; hidden rows remain unpadded.
use super::phase::{compute_kg_vnew_add_state, compute_ws_to_vnew};
use super::{
    KDA_STATE_COLUMN_PARTITIONS, KDA_STATE_DIM, KDA_STATE_ELEMS, KDA_STATE_PARTITION_COLS,
    KDA_STATE_SHARED_STRIDE, KdaStateCtx,
};
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::float_ptx::fma_f32;
use crate::kda_tc::{CtaTiles, KdaDecayTile, KdaStateSaveTile, KdaVNewTile};
use cuda_device::{DisjointSlice, thread};

pub(crate) fn exact_kda_state_body(
    kg: &[u16],
    mut v_new: DisjointSlice<u16>,
    w: &[u16],
    u: &[f32],
    exit_decay: &[f32],
    mut chunk_states: DisjointSlice<u16>,
    initial_state: *const f32,
    final_state: *mut f32,
    p: CausalAttentionParams,
    state: &mut KdaStateSaveTile,
    _decay: &mut KdaDecayTile,
    vnew_tile: &mut KdaVNewTile,
    tiles: CtaTiles<'_>,
) {
    let bh = thread::blockIdx_x();
    let partition = thread::blockIdx_y();
    let tid = thread::threadIdx_x();
    if bh >= p.batch_size * p.head_count
        || partition >= KDA_STATE_COLUMN_PARTITIONS
        || p.head_dim != 64
        || p.chunk_size != 64
    {
        return;
    }
    let chunks = p.seq_len.div_ceil(64);
    let padded_seq_len = chunks * 64;
    let column_start = partition * KDA_STATE_PARTITION_COLS;
    let partition_elems = 64 * KDA_STATE_PARTITION_COLS;
    let tile = CtaTile::from_column_partition(tid, column_start);
    let (a_tile, b_tile) = tiles;
    let mut i = tid;
    while i < partition_elems {
        let k = i / KDA_STATE_PARTITION_COLS;
        let v = column_start + i % KDA_STATE_PARTITION_COLS;
        let index = k * 64 + v;
        let value = if initial_state.is_null() {
            0.0
        } else {
            unsafe { *initial_state.add((bh * KDA_STATE_ELEMS + index) as usize) }
        };
        state[(k * KDA_STATE_SHARED_STRIDE + v) as usize] = value;
        i += thread::blockDim_x();
    }
    thread::sync_threads();
    let mut chunk = 0;
    while chunk < chunks {
        let start = chunk * 64;
        let end = p.seq_len.min(start + 64);
        let ctx = KdaStateCtx {
            tile,
            bh,
            start,
            end,
            seq_len: padded_seq_len,
            chunks,
        };
        i = tid;
        while i < partition_elems {
            let k = i / KDA_STATE_PARTITION_COLS;
            let v = column_start + i % KDA_STATE_PARTITION_COLS;
            unsafe {
                *chunk_states.get_unchecked_mut(
                    ((bh * chunks + chunk) * KDA_STATE_ELEMS + k * 64 + v) as usize,
                ) = cvt_rn_f16_f32(state[(k * KDA_STATE_SHARED_STRIDE + v) as usize]);
            }
            i += thread::blockDim_x();
        }
        // Initialize only the masked residual tail, preserving complete native
        // tape contents without reading nonexistent input tokens.
        i = tid;
        while i < partition_elems {
            let token = i / KDA_STATE_PARTITION_COLS;
            let v = column_start + i % KDA_STATE_PARTITION_COLS;
            if start + token >= end {
                unsafe {
                    *v_new.get_unchecked_mut(
                        ((bh * chunks + chunk) * 4096 + token * 64 + v) as usize,
                    ) = 0;
                }
            }
            i += thread::blockDim_x();
        }
        thread::sync_threads();
        compute_ws_to_vnew(
            w.as_ptr(),
            u.as_ptr(),
            v_new.as_mut_ptr(),
            vnew_tile.as_mut_ptr(),
            state,
            a_tile,
            b_tile,
            ctx,
        );
        // This factor was produced from the original negative increments.
        // Applying it directly preserves natural underflow to zero.
        i = tid;
        while i < partition_elems {
            let k = i / KDA_STATE_PARTITION_COLS;
            let v = column_start + i % KDA_STATE_PARTITION_COLS;
            state[(k * KDA_STATE_SHARED_STRIDE + v) as usize] *=
                exit_decay[((bh * chunks + chunk) * 64 + k) as usize];
            i += thread::blockDim_x();
        }
        // Subsequent TC staging reads state only after this CTA is coherent.
        thread::sync_threads();
        compute_kg_vnew_add_state(
            kg.as_ptr(),
            vnew_tile.as_mut_ptr(),
            state,
            a_tile,
            b_tile,
            ctx,
        );
        thread::sync_threads();
        chunk += 1;
    }
    i = tid;
    while i < partition_elems {
        let k = i / KDA_STATE_PARTITION_COLS;
        let v = column_start + i % KDA_STATE_PARTITION_COLS;
        unsafe {
            *final_state.add((bh * KDA_STATE_ELEMS + k * KDA_STATE_DIM + v) as usize) =
                state[(k * KDA_STATE_SHARED_STRIDE + v) as usize];
        }
        i += thread::blockDim_x();
    }
}

/// Replay only the established Kg^T*R state update. The true incoming state
/// remains in the working shared accumulator and is consumed by a vector-dot
/// carry VJP before the next update. Saved state cotangents are FP16; row-scale
/// controls are applied after the two32-value partial reductions are joined.
/// There is no global FP32 recurrent-state trace or higher-precision GEMM.
pub(crate) fn exact_kda_state_carry_body(
    kg: &[u16],
    residual: &[u16],
    carry: &[f32],
    state_grad: &[u16],
    mut partials: DisjointSlice<f32>,
    initial_state: *const f32,
    p: CausalAttentionParams,
    state: &mut KdaStateSaveTile,
    _decay: &mut KdaDecayTile,
    vnew_tile: &mut KdaVNewTile,
    tiles: CtaTiles<'_>,
) {
    let bh = thread::blockIdx_x();
    let partition = thread::blockIdx_y();
    let tid = thread::threadIdx_x();
    if bh >= p.batch_size * p.head_count
        || partition >= KDA_STATE_COLUMN_PARTITIONS
        || p.head_dim != 64
        || p.chunk_size != 64
    {
        return;
    }
    let chunks = p.seq_len.div_ceil(64);
    let column_start = partition * KDA_STATE_PARTITION_COLS;
    let partition_elems = 64 * KDA_STATE_PARTITION_COLS;
    let tile = CtaTile::from_column_partition(tid, column_start);
    let (a_tile, b_tile) = tiles;
    let mut i = tid;
    while i < partition_elems {
        let k = i / KDA_STATE_PARTITION_COLS;
        let v = column_start + i % KDA_STATE_PARTITION_COLS;
        state[(k * KDA_STATE_SHARED_STRIDE + v) as usize] = if initial_state.is_null() {
            0.0
        } else {
            unsafe { *initial_state.add((bh * KDA_STATE_ELEMS + k * 64 + v) as usize) }
        };
        i += thread::blockDim_x();
    }
    thread::sync_threads();
    let mut chunk = 0;
    while chunk < chunks {
        let base = (bh * chunks + chunk) * 4096;
        if tid < 64 {
            let k = tid;
            let mut sum = 0.0;
            let mut local_v = 0;
            while local_v < KDA_STATE_PARTITION_COLS {
                let v = column_start + local_v;
                sum = fma_f32(
                    cvt_f32_f16(state_grad[(base + k * 64 + v) as usize]),
                    state[(k * KDA_STATE_SHARED_STRIDE + v) as usize],
                    sum,
                );
                local_v += 1;
            }
            unsafe {
                *partials.get_unchecked_mut(
                    (((bh * chunks + chunk) * 64 + k) * 2 + partition) as usize,
                ) = sum;
            }
        }
        // All vector-dot readers must finish before any thread modifies the
        // shared incoming accumulator for the recurrent update.
        thread::sync_threads();
        let start = chunk * 64;
        let end = p.seq_len.min(start + 64);
        i = tid;
        while i < partition_elems {
            let token = i / KDA_STATE_PARTITION_COLS;
            let v = column_start + i % KDA_STATE_PARTITION_COLS;
            vnew_tile[(token as usize) * crate::kda_tc::KDA_VNEW_SHARED_STRIDE
                + (v - column_start) as usize] = if start + token < end {
                residual[(base + token * 64 + v) as usize]
            } else {
                0
            };
            let k = i / KDA_STATE_PARTITION_COLS;
            state[(k * KDA_STATE_SHARED_STRIDE + v) as usize] *=
                carry[((bh * chunks + chunk) * 64 + k) as usize];
            i += thread::blockDim_x();
        }
        thread::sync_threads();
        compute_kg_vnew_add_state(
            kg.as_ptr(),
            vnew_tile.as_mut_ptr(),
            state,
            a_tile,
            b_tile,
            KdaStateCtx {
                tile,
                bh,
                start,
                end,
                seq_len: chunks * 64,
                chunks,
            },
        );
        thread::sync_threads();
        chunk += 1;
    }
}
