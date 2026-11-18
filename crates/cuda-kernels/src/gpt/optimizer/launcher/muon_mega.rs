use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::launch::{grid_x_config, launch_config};
use crate::nvfp4_quant::NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK;

use super::super::args::{
    MuonMegaUpdateArgs, MuonTmaFinishArgs, MuonTmaHyperballFinishArgs, MuonTmaPrepareArgs,
    MuonTmaSignUpdateArgs,
};
use super::super::{MUON_COOPERATIVE_BLOCKS, MUON_MATRIX_PHASES};
use super::OptimizerModule;

const MUON_NVFP4_VALUES_PER_GROUP: u32 = 16;
const MUON_NVFP4_THREADS_PER_GROUP: u32 = 4;
const NORMUON_RECTANGULAR_THREADS_PER_BLOCK: u32 = 1024;
const NORMUON_RECTANGULAR_COLS_PER_BLOCK: u32 = 32;
const REDUCE_PARALLEL_MIN_CHUNKS: u32 = 4096;
const REDUCE_PARALLEL_MAX_BLOCKS: u32 = 128;
const MUON_TRANSPOSE_TILE_ROWS: u32 = 32;
const MUON_TRANSPOSE_TILE_COLS: u32 = 64;

impl OptimizerModule {
    pub fn muon_mega_update(&self, args: MuonMegaUpdateArgs<'_>) -> Result<(), DriverError> {
        assert_mega_args(&args);
        let matrix_count = args.slot_count / MUON_MATRIX_PHASES as u32;
        self.apply.muon.mega.muon_mega_update_cooperative_kernel(
            args.stream,
            launch_config(
                (MUON_COOPERATIVE_BLOCKS as u32, matrix_count, 1),
                CTA_THREADS,
            ),
            args.slots,
            args.oriented,
            args.polar_next,
            args.polar_x,
            args.polar_gram,
            args.polar_ax,
            args.polar_chunks,
            args.slot_count,
            args.max_len,
            args.max_ax_len,
            args.max_dim,
            args.mu,
            args.grad_scale,
            args.learning_rate,
            args.weight_decay,
            args.average_coefficient,
            args.iterations,
        )
    }

    pub fn muon_tma_prepare_polar(&self, args: MuonTmaPrepareArgs<'_>) -> Result<u32, DriverError> {
        assert!(args.slot_index < args.slots.len() as u32);
        assert!(args.matrix_len > 0);
        let amax_chunk_count = args
            .matrix_len
            .div_ceil(NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32);
        assert!(args.polar_x_chunk_amax.len() >= amax_chunk_count as usize);

        let elements_per_thread = if args.variance_adaptive != 0 { 2 } else { 1 };
        let momentum_chunk_count = args.matrix_len.div_ceil(elements_per_thread * CTA_THREADS);
        assert!(args.polar_chunks.len() >= momentum_chunk_count as usize);
        self.apply.muon.tma_split.muon_tma_momentum_orient_kernel(
            args.stream,
            grid_x_config(momentum_chunk_count, CTA_THREADS),
            args.slots,
            &mut *args.oriented,
            &mut *args.polar_chunks,
            args.slot_index,
            args.mu,
            args.grad_scale,
            args.nesterov,
            args.variance_adaptive,
            args.bias_correction_inv,
        )?;

        let (source_partial_count, source_chunk_stride) = reduction_geometry(momentum_chunk_count);
        if source_partial_count > 1 {
            self.apply
                .muon
                .tma_split
                .muon_tma_reduce_source_norm_partials_kernel(
                    args.stream,
                    grid_x_config(source_partial_count, CTA_THREADS),
                    &mut *args.polar_chunks,
                    momentum_chunk_count,
                    source_chunk_stride,
                )?;
        } else {
            self.apply
                .muon
                .tma_split
                .muon_tma_reduce_source_norm_kernel(
                    args.stream,
                    grid_x_config(1, CTA_THREADS),
                    &mut *args.polar_chunks,
                    momentum_chunk_count,
                )?;
        }

        self.apply
            .muon
            .tma_split
            .muon_tma_scale_source_to_x_kernel(
                args.stream,
                grid_x_config(amax_chunk_count, CTA_THREADS),
                &*args.oriented,
                args.polar_x,
                &*args.polar_chunks,
                args.polar_x_chunk_amax,
                args.matrix_len,
            )?;
        Ok(amax_chunk_count)
    }

    pub fn muon_tma_prepare_polar_cooperative_reference(
        &self,
        args: MuonTmaPrepareArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.slot_index < args.slots.len() as u32);
        assert!(args.matrix_len > 0);
        self.apply.muon.tma_split.muon_tma_prepare_polar_kernel(
            args.stream,
            launch_config((MUON_COOPERATIVE_BLOCKS as u32, 1, 1), CTA_THREADS),
            args.slots,
            args.oriented,
            args.polar_x,
            args.polar_chunks,
            args.slot_index,
            args.mu,
            args.grad_scale,
            args.nesterov,
            args.variance_adaptive,
            args.bias_correction_inv,
        )
    }

    pub fn muon_tma_finish_update(
        &self,
        mut args: MuonTmaFinishArgs<'_>,
    ) -> Result<(), DriverError> {
        self.muon_tma_normuon(&mut args)?;
        self.muon_tma_update_master_and_amax(&mut args)?;

        let groups_per_block = CTA_THREADS / MUON_NVFP4_THREADS_PER_GROUP;
        let group_count = args.matrix_len / MUON_NVFP4_VALUES_PER_GROUP;
        self.apply
            .muon
            .tma_split
            .muon_tma_encode_updated_master_kernel(
                args.stream,
                grid_x_config(group_count.div_ceil(groups_per_block), CTA_THREADS),
                args.slots,
                args.slot_index,
            )
    }

    pub fn muon_tma_finish_update_deferred_quantization(
        &self,
        mut args: MuonTmaFinishArgs<'_>,
    ) -> Result<(), DriverError> {
        self.muon_tma_normuon(&mut args)?;
        self.muon_tma_update_master_and_amax(&mut args)
    }

    pub fn muon_tma_finish_hyperball_update_deferred_quantization(
        &self,
        mut args: MuonTmaHyperballFinishArgs<'_>,
    ) -> Result<(), DriverError> {
        self.muon_tma_hyperball_normuon(&mut args)?;
        let chunk_count = args
            .finish
            .matrix_len
            .div_ceil(NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32);
        assert!(args.finish.polar_chunks.len() >= 2 * chunk_count as usize);

        let tiled_transpose = args.finish.matrix_rows > args.finish.matrix_cols
            && args
                .finish
                .matrix_rows
                .is_multiple_of(MUON_TRANSPOSE_TILE_ROWS)
            && args
                .finish
                .matrix_cols
                .is_multiple_of(MUON_TRANSPOSE_TILE_COLS);
        if tiled_transpose {
            self.apply
                .muon
                .tma_split
                .muon_tma_hyperball_project_transposed_tiles_kernel(
                    args.finish.stream,
                    grid_x_config(chunk_count, CTA_THREADS),
                    args.finish.slots,
                    args.finish.symexp_lin_slots,
                    args.finish.polar_update,
                    &mut *args.finish.polar_chunks,
                    &*args.finish.normuon_factors,
                    &*args.finish.normuon_chunks,
                    &*args.hyperball_chunks,
                    args.finish.qk_clip_factors,
                    args.finish.slot_index,
                    args.finish.learning_rate,
                    args.finish.average_coefficient,
                    args.finish.schedule_beta,
                    args.finish.symexp_lin_beta,
                    args.use_schedule_free as u32,
                )?;
        } else {
            self.apply
                .muon
                .tma_split
                .muon_tma_hyperball_project_chunks_kernel(
                    args.finish.stream,
                    grid_x_config(chunk_count, CTA_THREADS),
                    args.finish.slots,
                    args.finish.symexp_lin_slots,
                    args.finish.polar_update,
                    &mut *args.finish.polar_chunks,
                    &*args.finish.normuon_factors,
                    &*args.finish.normuon_chunks,
                    &*args.hyperball_chunks,
                    args.finish.qk_clip_factors,
                    args.finish.slot_index,
                    args.finish.learning_rate,
                    args.finish.average_coefficient,
                    args.finish.schedule_beta,
                    args.finish.symexp_lin_beta,
                    args.use_schedule_free as u32,
                )?;
        }
        self.reduce_update_amax(
            args.finish.stream,
            args.finish.slots,
            args.finish.polar_chunks,
            args.finish.slot_index,
            chunk_count,
        )
    }

    pub fn muon_tma_sign_update_deferred_quantization(
        &self,
        args: MuonTmaSignUpdateArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.slot_index < args.slots.len() as u32);
        assert!(args.matrix_len > 0);
        assert_eq!(args.matrix_len % MUON_NVFP4_VALUES_PER_GROUP, 0);
        let chunk_count = args
            .matrix_len
            .div_ceil(NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32);
        assert!(args.update_chunks.len() >= 2 * chunk_count as usize);

        self.apply
            .muon
            .tma_split
            .muon_tma_sign_update_master_chunks_kernel(
                args.stream,
                grid_x_config(chunk_count, CTA_THREADS),
                args.slots,
                args.symexp_lin_slots,
                &mut *args.update_chunks,
                args.qk_clip_factors,
                args.slot_index,
                args.mu,
                args.grad_scale,
                args.variance_adaptive,
                args.learning_rate,
                args.weight_decay,
                args.average_coefficient,
                args.schedule_beta,
                args.symexp_lin_beta,
            )?;

        self.reduce_update_amax(
            args.stream,
            args.slots,
            args.update_chunks,
            args.slot_index,
            chunk_count,
        )
    }

    fn muon_tma_normuon(&self, args: &mut MuonTmaFinishArgs<'_>) -> Result<(), DriverError> {
        assert!(args.slot_index < args.slots.len() as u32);
        assert!(args.matrix_len > 0);
        assert!(args.polar_cols > 0);
        assert_eq!(args.matrix_len % args.polar_cols, 0);
        let polar_rows = args.matrix_len / args.polar_cols;
        let square = polar_rows == args.polar_cols;
        let threads_per_block = if square {
            CTA_THREADS
        } else {
            NORMUON_RECTANGULAR_THREADS_PER_BLOCK
        };
        let chunk_count = if square {
            args.polar_cols
        } else {
            args.polar_cols.div_ceil(NORMUON_RECTANGULAR_COLS_PER_BLOCK)
        };
        assert!(args.normuon_factors.len() >= args.polar_cols as usize);
        assert!(args.normuon_chunks.len() >= 2 * chunk_count as usize);

        self.apply.muon.tma_split.muon_tma_normuon_stats_kernel(
            args.stream,
            grid_x_config(chunk_count, threads_per_block),
            args.slots,
            args.polar_update,
            &mut *args.normuon_factors,
            &mut *args.normuon_chunks,
            &mut *args.polar_chunks,
            args.slot_index,
            args.matrix_len,
            args.polar_cols,
            0,
        )?;
        self.apply
            .muon
            .tma_split
            .muon_tma_normuon_reduce_scale_kernel(
                args.stream,
                grid_x_config(1, CTA_THREADS),
                &mut *args.normuon_chunks,
                chunk_count,
            )
    }

    fn muon_tma_hyperball_normuon(
        &self,
        args: &mut MuonTmaHyperballFinishArgs<'_>,
    ) -> Result<(), DriverError> {
        let finish = &mut args.finish;
        assert!(finish.slot_index < finish.slots.len() as u32);
        assert!(finish.matrix_len > 0);
        assert!(finish.polar_cols > 0);
        assert_eq!(finish.matrix_len % finish.polar_cols, 0);
        let polar_rows = finish.matrix_len / finish.polar_cols;
        let square = polar_rows == finish.polar_cols;
        let threads_per_block = if square {
            CTA_THREADS
        } else {
            NORMUON_RECTANGULAR_THREADS_PER_BLOCK
        };
        let chunk_count = if square {
            finish.polar_cols
        } else {
            finish
                .polar_cols
                .div_ceil(NORMUON_RECTANGULAR_COLS_PER_BLOCK)
        };
        assert!(finish.normuon_factors.len() >= finish.polar_cols as usize);
        assert!(finish.normuon_chunks.len() >= 2 * chunk_count as usize);
        assert!(args.hyperball_chunks.len() >= 2 * chunk_count as usize);

        self.apply.muon.tma_split.muon_tma_normuon_stats_kernel(
            finish.stream,
            grid_x_config(chunk_count, threads_per_block),
            finish.slots,
            finish.polar_update,
            &mut *finish.normuon_factors,
            &mut *finish.normuon_chunks,
            &mut *args.hyperball_chunks,
            finish.slot_index,
            finish.matrix_len,
            finish.polar_cols,
            1,
        )?;
        self.apply
            .muon
            .tma_split
            .muon_tma_normuon_reduce_scale_kernel(
                finish.stream,
                grid_x_config(1, CTA_THREADS),
                &mut *finish.normuon_chunks,
                chunk_count,
            )?;
        self.apply
            .muon
            .tma_split
            .muon_tma_hyperball_reduce_parameter_stats_kernel(
                finish.stream,
                grid_x_config(1, CTA_THREADS),
                &mut *args.hyperball_chunks,
                chunk_count,
            )
    }

    fn muon_tma_update_master_and_amax(
        &self,
        args: &mut MuonTmaFinishArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.slot_index < args.slots.len() as u32);
        assert!(args.matrix_len > 0);
        assert_eq!(args.matrix_len % MUON_NVFP4_VALUES_PER_GROUP, 0);
        let chunk_count = args
            .matrix_len
            .div_ceil(NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32);
        assert!(args.polar_chunks.len() >= 2 * chunk_count as usize);

        self.apply
            .muon
            .tma_split
            .muon_tma_update_master_chunks_kernel(
                args.stream,
                grid_x_config(chunk_count, CTA_THREADS),
                args.slots,
                args.symexp_lin_slots,
                args.polar_update,
                args.polar_bound_amax,
                &mut *args.polar_chunks,
                &*args.normuon_factors,
                &*args.normuon_chunks,
                args.qk_clip_factors,
                args.slot_index,
                args.learning_rate,
                args.weight_decay,
                args.average_coefficient,
                args.schedule_beta,
                args.symexp_lin_beta,
                args.apply_polar_sqrt_bound,
            )?;

        self.reduce_update_amax(
            args.stream,
            args.slots,
            args.polar_chunks,
            args.slot_index,
            chunk_count,
        )
    }

    fn reduce_update_amax(
        &self,
        stream: &CudaStream,
        slots: &DeviceBuffer<super::super::MuonSlotDescriptor>,
        polar_chunks: &mut DeviceBuffer<f32>,
        slot_index: u32,
        chunk_count: u32,
    ) -> Result<(), DriverError> {
        let (partial_count, chunk_stride) = reduction_geometry(chunk_count);
        if partial_count > 1 {
            return self
                .apply
                .muon
                .tma_split
                .muon_tma_reduce_update_amax_partials_kernel(
                    stream,
                    grid_x_config(partial_count, CTA_THREADS),
                    slots,
                    &mut *polar_chunks,
                    slot_index,
                    chunk_count,
                    chunk_stride,
                );
        }
        self.apply
            .muon
            .tma_split
            .muon_tma_reduce_update_amax_kernel(
                stream,
                grid_x_config(1, CTA_THREADS),
                slots,
                &*polar_chunks,
                slot_index,
                chunk_count,
            )
    }

    pub fn muon_tma_finish_update_cooperative_reference(
        &self,
        mut args: MuonTmaFinishArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.slot_index < args.slots.len() as u32);
        assert!(args.matrix_len > 0);
        assert!(args.polar_chunks.len() >= 2 * MUON_COOPERATIVE_BLOCKS);
        self.muon_tma_normuon(&mut args)?;
        self.apply.muon.tma_split.muon_tma_finish_update_kernel(
            args.stream,
            launch_config((MUON_COOPERATIVE_BLOCKS as u32, 1, 1), CTA_THREADS),
            args.slots,
            args.symexp_lin_slots,
            args.polar_update,
            args.polar_bound_amax,
            args.polar_chunks,
            args.normuon_factors,
            args.normuon_chunks,
            args.slot_index,
            args.learning_rate,
            args.weight_decay,
            args.average_coefficient,
            args.schedule_beta,
            args.symexp_lin_beta,
            args.apply_polar_sqrt_bound,
        )
    }
}

fn reduction_geometry(chunk_count: u32) -> (u32, u32) {
    if chunk_count < REDUCE_PARALLEL_MIN_CHUNKS {
        return (1, 1);
    }
    let partial_count = chunk_count
        .div_ceil(CTA_THREADS)
        .min(REDUCE_PARALLEL_MAX_BLOCKS);
    (partial_count, chunk_count.div_ceil(partial_count))
}

fn assert_mega_args(args: &MuonMegaUpdateArgs<'_>) {
    let slots = args.slot_count as usize;
    let matrix_count = args.slot_count as usize / MUON_MATRIX_PHASES;
    assert_eq!(args.slot_count as usize % MUON_MATRIX_PHASES, 0);
    assert!(args.slots.len() >= slots);
    assert!(args.oriented.len() >= args.max_len as usize * matrix_count);
    assert!(args.polar_next.len() >= args.max_len as usize * matrix_count);
    assert!(args.polar_x.len() >= args.max_len as usize * matrix_count);
    assert!(args.polar_ax.len() >= args.max_ax_len as usize * matrix_count);
    assert!(args.polar_gram.len() >= args.max_dim as usize * args.max_dim as usize * matrix_count);
    assert!(args.polar_chunks.len() >= 2 * MUON_COOPERATIVE_BLOCKS * matrix_count);
}
