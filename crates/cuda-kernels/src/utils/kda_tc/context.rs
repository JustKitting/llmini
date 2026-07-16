use cuda_device::thread;

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::kda_common::{batch_head, chunk_count, kda_tc_shape};

#[derive(Clone, Copy)]
pub(crate) struct CompactTileCtx<'a> {
    pub(crate) tile: CtaTile,
    pub(crate) batch: u32,
    pub(crate) head: u32,
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) params: &'a CausalAttentionParams,
}

impl<'a> CompactTileCtx<'a> {
    pub(crate) fn new(
        tile: CtaTile,
        batch_head: (u32, u32),
        token_span: (u32, u32),
        params: &'a CausalAttentionParams,
    ) -> Self {
        let (batch, head) = batch_head;
        let (start, end) = token_span;
        Self {
            tile,
            batch,
            head,
            start,
            end,
            params,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MatrixTileCtx<'a> {
    pub(crate) tile: CtaTile,
    pub(crate) bh: u32,
    pub(crate) chunk: u32,
    pub(crate) chunk_tokens: u32,
    pub(crate) params: &'a CausalAttentionParams,
}

#[derive(Clone, Copy)]
pub(crate) struct KdaChunkTileCtx<'a> {
    pub(crate) bh: u32,
    pub(crate) chunk: u32,
    pub(crate) compact: CompactTileCtx<'a>,
    pub(crate) matrix: MatrixTileCtx<'a>,
}

impl<'a> KdaChunkTileCtx<'a> {
    pub(crate) fn from_block(params: &'a CausalAttentionParams) -> Option<Self> {
        Self::from_block_tile(params, CtaTile::from_tile(thread::threadIdx_x(), 0, 0, 0))
    }

    pub(crate) fn from_wide_block(params: &'a CausalAttentionParams) -> Option<Self> {
        Self::from_block_tile(
            params,
            CtaTile::from_wide_tile(thread::threadIdx_x(), 0, 0, 0),
        )
    }

    fn from_block_tile(params: &'a CausalAttentionParams, tile: CtaTile) -> Option<Self> {
        let bh = thread::blockIdx_x();
        let chunk = thread::blockIdx_y();
        let chunks = chunk_count(params);
        if bh >= batch_head(params) || chunk >= chunks || !kda_tc_shape(params) {
            return None;
        }
        let batch = bh / params.head_count;
        let head = bh - batch * params.head_count;
        let start = chunk * params.chunk_size;
        let end = params.seq_len.min(start + params.chunk_size);
        Some(Self {
            bh,
            chunk,
            compact: CompactTileCtx::new(tile, (batch, head), (start, end), params),
            matrix: MatrixTileCtx {
                tile,
                bh,
                chunk,
                chunk_tokens: end - start,
                params,
            },
        })
    }
}
