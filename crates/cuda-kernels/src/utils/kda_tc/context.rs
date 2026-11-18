use cuda_device::thread;

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::kda_common::{batch_head, chunk_count, kda_tc_shape};

#[derive(Clone, Copy)]
pub(crate) struct CompactTileCtx {
    pub(crate) tile: CtaTile,
    pub(crate) batch: u32,
    pub(crate) head: u32,
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) row_count: u32,
    pub(crate) seq_len: u32,
    pub(crate) embedding_dim: u32,
    pub(crate) head_count: u32,
    pub(crate) head_dim: u32,
    pub(crate) chunk_size: u32,
}

impl CompactTileCtx {
    pub(crate) fn new(
        tile: CtaTile,
        batch_head: (u32, u32),
        token_span: (u32, u32),
        params: &CausalAttentionParams,
    ) -> Self {
        let (batch, head) = batch_head;
        let (start, end) = token_span;
        Self {
            tile,
            batch,
            head,
            start,
            end,
            row_count: params.row_count,
            seq_len: params.seq_len,
            embedding_dim: params.embedding_dim,
            head_count: params.head_count,
            head_dim: params.head_dim,
            chunk_size: params.chunk_size,
        }
    }

    #[inline(always)]
    pub(crate) fn compact_index(self, token: u32, dim: u32) -> usize {
        (((self.batch * self.head_count + self.head) * self.seq_len + token) * self.head_dim + dim)
            as usize
    }

    #[inline(always)]
    pub(crate) fn hidden_index(self, token: u32, dim: u32) -> usize {
        ((self.batch * self.seq_len + token) * self.embedding_dim + self.head * self.head_dim + dim)
            as usize
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MatrixTileCtx {
    pub(crate) tile: CtaTile,
    pub(crate) bh: u32,
    pub(crate) chunk: u32,
    pub(crate) chunk_tokens: u32,
    pub(crate) chunk_size: u32,
    pub(crate) chunk_count: u32,
}

impl MatrixTileCtx {
    #[inline(always)]
    pub(crate) fn index(self, row: u32, col: u32) -> usize {
        (((self.bh * self.chunk_count + self.chunk) * self.chunk_size + row) * self.chunk_size
            + col) as usize
    }
}

#[derive(Clone, Copy)]
pub(crate) struct KdaChunkTileCtx {
    pub(crate) bh: u32,
    pub(crate) chunk: u32,
    pub(crate) compact: CompactTileCtx,
    pub(crate) matrix: MatrixTileCtx,
}

impl KdaChunkTileCtx {
    pub(crate) fn from_block(params: &CausalAttentionParams) -> Option<Self> {
        Self::from_block_tile(params, CtaTile::from_tile(thread::threadIdx_x(), 0, 0, 0))
    }

    pub(crate) fn from_wide_block(params: &CausalAttentionParams) -> Option<Self> {
        Self::from_block_tile(
            params,
            CtaTile::from_wide_tile(thread::threadIdx_x(), 0, 0, 0),
        )
    }

    fn from_block_tile(params: &CausalAttentionParams, tile: CtaTile) -> Option<Self> {
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
                chunk_size: params.chunk_size,
                chunk_count: chunks,
            },
        })
    }

    #[inline(always)]
    pub(crate) fn state_index(self, state_index: u32) -> usize {
        ((self.bh * self.matrix.chunk_count + self.chunk)
            * self.compact.head_dim
            * self.compact.head_dim
            + state_index) as usize
    }
}
