use cuda_device::{SharedArray, thread};

use super::super::finish::finish_kda_g_value;
use super::{KDA_INTRA_DG_TILE_ELEMS, KDA_INTRA_DKG_KG_TILE_ELEMS, KdaIntraCtx};
use crate::float_ptx::max_f32;
use crate::kda_elementwise::KdaWarpCtx;

pub(super) fn reverse_prefix_dg(
    qkv: &[u16],
    dg_tile: &SharedArray<f32, KDA_INTRA_DG_TILE_ELEMS>,
    dkg_kg_tile: &mut SharedArray<f32, KDA_INTRA_DKG_KG_TILE_ELEMS>,
    d_qkv: *mut f32,
    ctx: KdaIntraCtx<'_>,
    tid: u32,
) -> f32 {
    const PREFIX_SEGMENTS: u32 = 4;
    const PREFIX_DIMS: u32 = 64;

    let segment = tid / PREFIX_DIMS;
    let dim = tid - segment * PREFIX_DIMS;
    let valid = segment < PREFIX_SEGMENTS;
    let tokens_per_segment = 64 / PREFIX_SEGMENTS;
    let segment_start = ctx.start + segment * tokens_per_segment;
    let segment_end = segment_start + tokens_per_segment;

    let mut dg_last_value = 0.0;
    if valid && segment == PREFIX_SEGMENTS - 1 {
        let mut source = 0;
        while source < 64 {
            dg_last_value += dkg_kg_tile[(source * 64 + dim) as usize];
            source += 1;
        }
    }

    let mut segment_total = dg_last_value;
    if valid {
        let mut token = segment_end;
        while token > segment_start {
            token -= 1;
            segment_total += dg_tile[((token - ctx.start) * 64 + dim) as usize];
        }
    }

    // The product tile is dead once the last segment has accumulated dG_last.
    // Reuse its first four rows for the per-segment reverse-prefix totals.
    thread::sync_threads();
    dkg_kg_tile[(segment * PREFIX_DIMS + dim) as usize] = segment_total;
    thread::sync_threads();

    if valid {
        let mut acc = dg_last_value;
        let mut later_segment = segment + 1;
        while later_segment < PREFIX_SEGMENTS {
            acc += dkg_kg_tile[(later_segment * PREFIX_DIMS + dim) as usize];
            later_segment += 1;
        }

        let mut local_amax = 0.0;
        let mut token = segment_end;
        while token > segment_start {
            token -= 1;
            acc += dg_tile[((token - ctx.start) * 64 + dim) as usize];
            let row = ctx.batch * ctx.params.seq_len + token;
            local_amax = max_f32(
                local_amax,
                finish_kda_g_value(
                    qkv,
                    d_qkv,
                    KdaWarpCtx {
                        lane: tid & 31,
                        row,
                        batch: ctx.batch,
                        token,
                        head: ctx.head,
                        valid: true,
                    },
                    dim,
                    acc,
                    ctx.params,
                ),
            );
        }
        local_amax
    } else {
        0.0
    }
}
