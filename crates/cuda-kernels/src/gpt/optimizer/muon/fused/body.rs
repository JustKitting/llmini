use cuda_device::grid;

use super::super::super::work_grid::WorkGrid;
use super::momentum::momentum_orient;
use super::polar_step::run_polar_step;
use super::quant::quantize_updated_master;
use super::types::{
    MuonMatrixScratch, MuonMatrixShape, MuonMatrixState, MuonMatrixTiles, MuonUpdateScalars,
};
use super::update::update_master_chunks;

pub(super) fn muon_matrix_update_body(
    state: MuonMatrixState,
    scratch: MuonMatrixScratch,
    tiles: MuonMatrixTiles<'_>,
    work: WorkGrid,
    shape: MuonMatrixShape,
    scalars: MuonUpdateScalars,
) {
    let len = shape.len();
    let transposed = shape.polar_transposed();

    momentum_orient(
        state.grad,
        state.momentum,
        scratch.oriented,
        work,
        shape,
        scalars.mu,
        scalars.grad_scale,
        transposed,
    );
    grid::sync();

    let polar_rows = if transposed { shape.cols } else { shape.rows };
    let polar_cols = if transposed { shape.rows } else { shape.cols };
    let polar_update = run_polar_step(
        scratch.oriented,
        scratch.polar_next,
        scratch.polar_x,
        scratch.polar_gram,
        scratch.polar_ax,
        scratch.polar_chunks,
        tiles.a_tile,
        tiles.b_tile,
        tiles.warp_sums,
        work,
        polar_rows,
        polar_cols,
        false,
        scalars.iterations,
    );

    // Polar Express returns after a grid-wide sync; update can consume its result directly.
    update_master_chunks(
        polar_update,
        state.z_master,
        state.x_master,
        state.momentum,
        scratch.polar_chunks,
        core::ptr::null(),
        u32::MAX,
        0,
        shape.rows,
        shape.cols,
        len,
        shape.master_transposed(),
        1.0,
        scalars.learning_rate,
        scalars.weight_decay,
        scalars.average_coefficient,
        1.0,
        tiles.warp_sums,
        tiles.warp_max_pairs,
        work,
    );
    grid::sync();

    quantize_updated_master(
        state.x_master,
        scratch.polar_chunks,
        state.schedule_amax,
        state.out_fp4,
        state.out_scales,
        state.out_global_scale,
        len,
        tiles.warp_sums,
        tiles.warp_max_pairs,
        work,
    );
}
