use cuda_device::thread;

pub(crate) const CTA_M: u32 = 64;
pub(crate) const CTA_N: u32 = 64;
pub(crate) const CTA_K: u32 = 16;
pub(crate) const CTA_THREADS: u32 = 256;
pub(crate) const CTA_WIDE_THREADS: u32 = 512;
pub(crate) const CTA_A_ELEMS: usize = CTA_M as usize * CTA_K as usize;
pub(crate) const CTA_B_ELEMS: usize = CTA_N as usize * CTA_K as usize;

#[derive(Clone, Copy)]
pub(crate) struct CtaMatmulDims {
    pub(crate) batch_count: u32,
    pub(crate) m: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
}

impl CtaMatmulDims {
    #[inline(always)]
    pub(crate) fn new(batch_count: u32, m: u32, n: u32, k: u32) -> Self {
        Self {
            batch_count,
            m,
            n,
            k,
        }
    }

    #[inline(always)]
    pub(crate) fn aligned(self) -> bool {
        self.m.is_multiple_of(CTA_M) && self.n.is_multiple_of(CTA_N) && self.k.is_multiple_of(CTA_K)
    }
}

#[inline(always)]
pub(super) fn active_tile(batch_count: u32) -> Option<CtaTile> {
    let thread_id = thread::threadIdx_x();
    if thread_id >= CTA_THREADS || thread::blockIdx_z() >= batch_count {
        return None;
    }
    Some(CtaTile::new(thread_id))
}

#[inline(always)]
pub(super) fn active_wide_tile(batch_count: u32) -> Option<CtaTile> {
    let thread_id = thread::threadIdx_x();
    if thread_id >= CTA_WIDE_THREADS || thread::blockIdx_z() >= batch_count {
        return None;
    }
    Some(CtaTile::from_wide_tile(
        thread_id,
        thread::blockIdx_y(),
        thread::blockIdx_x(),
        thread::blockIdx_z(),
    ))
}

#[derive(Clone, Copy)]
pub(crate) struct CtaTile {
    pub(crate) batch: u32,
    pub(crate) row_base: u32,
    pub(crate) col_base: u32,
    pub(crate) warp_m: u32,
    pub(crate) warp_n0: u32,
    pub(crate) group: u32,
    pub(crate) thread_in_group: u32,
}

impl CtaTile {
    pub(super) fn new(thread_id: u32) -> Self {
        Self::from_tile(
            thread_id,
            thread::blockIdx_y(),
            thread::blockIdx_x(),
            thread::blockIdx_z(),
        )
    }

    pub(crate) fn from_tile(thread_id: u32, tile_row: u32, tile_col: u32, batch: u32) -> Self {
        let lane = thread_id & 31;
        let warp = thread_id >> 5;
        Self {
            batch,
            row_base: tile_row * CTA_M,
            col_base: tile_col * CTA_N,
            warp_m: warp >> 1,
            warp_n0: (warp & 1) << 2,
            group: lane >> 2,
            thread_in_group: lane & 0x3,
        }
    }

    pub(crate) fn from_wide_tile(thread_id: u32, tile_row: u32, tile_col: u32, batch: u32) -> Self {
        let lane = thread_id & 31;
        let warp = thread_id >> 5;
        Self {
            batch,
            row_base: tile_row * CTA_M,
            col_base: tile_col * CTA_N,
            warp_m: warp >> 2,
            warp_n0: (warp & 0x3) << 1,
            group: lane >> 2,
            thread_in_group: lane & 0x3,
        }
    }

    #[inline(always)]
    pub(crate) fn accumulator_coords(self, warp_n: u32, acc_index: usize) -> (u32, u32) {
        let row = self.row_base + self.warp_m * 16 + self.group + if acc_index < 2 { 0 } else { 8 };
        let col = self.col_base + warp_n * 8 + self.thread_in_group * 2 + (acc_index as u32 & 1);
        (row, col)
    }
}
