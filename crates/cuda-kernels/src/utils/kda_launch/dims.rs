use cuda_core::LaunchConfig;

use crate::launch::launch_config;

pub(crate) const KDA_HEAD_DIM: u32 = 64;

#[derive(Clone, Copy)]
pub(crate) struct LaunchDims {
    pub(crate) batch_head: u32,
    pub(crate) chunks: u32,
    pub(crate) chunk_batch: u32,
    pub(crate) compact_elems: u32,
    pub(crate) chunk_matrix_elems: u32,
    chunk_size: u32,
    head_dim: u32,
}

impl LaunchDims {
    pub(crate) fn new(
        batch_size: u32,
        head_count: u32,
        seq_len: u32,
        head_dim: u32,
        chunk_size: u32,
    ) -> Self {
        let batch_head = batch_size * head_count;
        let chunks = seq_len.div_ceil(chunk_size);
        Self {
            batch_head,
            chunks,
            chunk_batch: batch_head * chunks,
            compact_elems: batch_head * seq_len * head_dim,
            chunk_matrix_elems: batch_head * chunks * chunk_size * chunk_size,
            chunk_size,
            head_dim,
        }
    }

    pub(crate) fn cch(self) -> MatmulShape {
        MatmulShape(self.chunk_size, self.chunk_size, self.head_dim)
    }

    pub(crate) fn chc(self) -> MatmulShape {
        MatmulShape(self.chunk_size, self.head_dim, self.chunk_size)
    }
}

pub(crate) fn chunk_dim_config(
    batch_head: u32,
    chunks: u32,
    threads_per_block: u32,
) -> LaunchConfig {
    launch_config((batch_head, chunks, 1), threads_per_block)
}

#[derive(Clone, Copy)]
pub(crate) struct MatmulShape(pub(crate) u32, pub(crate) u32, pub(crate) u32);
