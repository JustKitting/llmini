use super::{KdaChunkwiseGrads, KdaChunkwiseInputs};
use crate::f16_tc_matmul::convert::{load_f16_global_bits_read_only, store_f16x2_shared};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_K};
use crate::kda_backward::{stage_compact_t_a, store_dh_quads};
use crate::kda_tc::{
    CompactTileCtx, CtaATile, CtaBTile, KdaStateTile, mma_accumulate, stage_compact_a,
    stage_compact_half_b_t, stage_compact_half_b_t_disjoint, stage_shared_state_b_t,
    store_compact_half_quad, tc_stage_loop,
};
use cuda_device::{DisjointSlice, thread};

// Both products retain separate, zero-initialized FP32 accumulators. Only their
// dOut RHS tile and synchronization are shared; neither MMA sum is reordered.
pub(super) fn project_du_and_state_seed(
    inputs: KdaChunkwiseInputs<'_>,
    tiles: (&mut CtaATile, &mut CtaATile, &mut CtaBTile),
    ctx: CompactTileCtx,
) -> ([f32; 4], [f32; 4]) {
    let (a, q, b) = tiles;
    let mut du = [[0.0_f32; 4]; 1];
    let mut seed = [[0.0_f32; 4]; 1];
    let mut k = 0;
    while k < ctx.chunk_size {
        stage_aqk_transposed(inputs.aqk, a, ctx, k);

        stage_compact_t_a(inputs.qg, q, ctx, k, 1.0);

        stage_compact_half_b_t(inputs.d_out, b, ctx, k);
        thread::sync_threads();
        mma_accumulate(ctx.tile, a, b, &mut du);

        mma_accumulate(ctx.tile, q, b, &mut seed);

        thread::sync_threads();
        k += CTA_K;
    }
    (du[0], seed[0])
}

fn stage_aqk_transposed(src: &[u16], dst: &mut CtaATile, ctx: CompactTileCtx, k: u32) {
    let chunks = ctx.seq_len.div_ceil(ctx.chunk_size);
    let chunk = ctx.start / ctx.chunk_size;
    let bh = ctx.batch * ctx.head_count + ctx.head;
    let base = (bh * chunks + chunk) * ctx.chunk_size * ctx.chunk_size;
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let query = k + col;
        let key = ctx.tile.row_base + row;
        let lo = if query < ctx.end - ctx.start && key < ctx.end - ctx.start {
            load_f16_global_bits_read_only(
                src.as_ptr(),
                (base + query * ctx.chunk_size + key) as usize,
            )
        } else {
            0
        };
        let hi = if query + 1 < ctx.end - ctx.start && key < ctx.end - ctx.start {
            load_f16_global_bits_read_only(
                src.as_ptr(),
                (base + (query + 1) * ctx.chunk_size + key) as usize,
            )
        } else {
            0
        };
        store_f16x2_shared(
            dst.as_mut_ptr(),
            crate::kda_tc::swizzle_tc_offset(pair) as usize,
            u32::from(lo) | (u32::from(hi) << 16),
        );
        pair += thread::blockDim_x() * 2;
    }
}

pub(super) fn finish_du(
    kg: &[f32],
    d_u: &mut DisjointSlice<u16>,
    d_h_next: &KdaStateTile,
    initial: [f32; 4],
    tiles: (&mut CtaATile, &mut CtaBTile),
    ctx: CompactTileCtx,
) {
    let (a, b) = tiles;
    let mut kg_dh = [[0.0_f32; 4]; 1];

    tc_stage_loop!(ctx.tile, a, b, kg_dh; k < ctx.head_dim; {
        stage_compact_a(kg, a, ctx, k);
    } {
        stage_shared_state_b_t(d_h_next, b, ctx, k);
    });

    let mut value = initial;

    let mut i = 0;
    while i < 4 {
        value[i] = initial[i] + kg_dh[0][i];
        i += 1;
    }

    store_compact_half_quad(value, d_u, ctx, ctx.tile.warp_n0, 1.0);
}

pub(super) fn compute_prev_dh_tc(
    inputs: KdaChunkwiseInputs<'_>,
    grads: &mut KdaChunkwiseGrads<'_>,
    seed: [f32; 4],
    states: (&KdaStateTile, &mut KdaStateTile),
    tiles: (&mut CtaATile, &mut CtaBTile),
    ctx: CompactTileCtx,
) {
    let (d_h_next, d_h) = states;
    let (a, b) = tiles;
    let mut acc = [seed];
    tc_stage_loop!(ctx.tile, a, b, acc; k < ctx.chunk_size; {
        crate::kda_backward::stage_half_compact_t_a_neg(inputs.w, a, ctx, k);
    } {
        stage_compact_half_b_t_disjoint(&mut grads.d_u, b, ctx, k);
    });
    store_dh_quads(acc, d_h_next, d_h, inputs.g, ctx);
}
