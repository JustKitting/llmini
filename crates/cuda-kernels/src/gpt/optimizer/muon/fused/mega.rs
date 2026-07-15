use cuda_device::{
    DisjointSlice, SharedArray, cooperative_launch, cuda_module, grid, kernel, thread,
};

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};
use crate::optimizer::MuonSlotDescriptor;

use super::super::super::MUON_MATRIX_PHASES;
use super::super::super::threads::WARPS_PER_BLOCK;
use super::types::{MuonMatrixScratch, MuonMatrixTiles, MuonUpdateScalars};

mod slot;

#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    #[cooperative_launch]
    pub fn muon_mega_update_cooperative_kernel(
        slots: &[MuonSlotDescriptor],
        mut oriented: DisjointSlice<f32>,
        mut polar_next: DisjointSlice<f32>,
        mut polar_x: DisjointSlice<f32>,
        mut polar_gram: DisjointSlice<f32>,
        mut polar_ax: DisjointSlice<f32>,
        mut polar_chunks: DisjointSlice<f32>,
        slot_count: u32,
        max_len: u32,
        max_ax_len: u32,
        max_dim: u32,
        mu: f32,
        grad_scale: f32,
        learning_rate: f32,
        weight_decay: f32,
        average_coefficient: f32,
        iterations: u32,
    ) {
        static mut A_TILE: SharedArray<u16, CTA_A_ELEMS> = SharedArray::UNINIT;
        static mut B_TILE: SharedArray<u16, CTA_B_ELEMS> = SharedArray::UNINIT;
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
        static mut WARP_MAX_PAIRS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let scratch = MuonMatrixScratch {
            oriented: oriented.as_mut_ptr(),
            polar_next: polar_next.as_mut_ptr(),
            polar_x: polar_x.as_mut_ptr(),
            polar_gram: polar_gram.as_mut_ptr(),
            polar_ax: polar_ax.as_mut_ptr(),
            polar_chunks: polar_chunks.as_mut_ptr(),
        };
        let layout = slot::MegaScratchLayout {
            max_len,
            max_ax_len,
            max_dim,
        };
        let scalars = MuonUpdateScalars {
            mu,
            grad_scale,
            learning_rate,
            weight_decay,
            average_coefficient,
            iterations,
        };
        let matrix = thread::blockIdx_y();
        let matrix_count = thread::gridDim_y();
        let mut phase = 0;
        while phase < MUON_MATRIX_PHASES as u32 {
            let slot = phase * matrix_count + matrix;
            if slot < slot_count {
                unsafe {
                    slot::launch_slot(
                        slot,
                        matrix,
                        slots,
                        scratch,
                        layout,
                        MuonMatrixTiles {
                            a_tile: &mut A_TILE,
                            b_tile: &mut B_TILE,
                            warp_sums: &mut WARP_SUMS,
                            warp_max_pairs: &mut WARP_MAX_PAIRS,
                        },
                        scalars,
                    );
                }
            }
            grid::sync();
            phase += 1;
        }
    }
}
