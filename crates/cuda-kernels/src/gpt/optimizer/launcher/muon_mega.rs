use cuda_core::DriverError;

use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::launch::{grid_x_config, launch_config};
use crate::nvfp4_quant::NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK;

use super::super::args::{MuonMegaUpdateArgs, MuonTmaFinishArgs, MuonTmaPrepareArgs};
use super::super::{MUON_COOPERATIVE_BLOCKS, MUON_MATRIX_PHASES};
use super::OptimizerModule;

const MUON_NVFP4_VALUES_PER_GROUP: u32 = 16;
const MUON_NVFP4_THREADS_PER_GROUP: u32 = 4;

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
        assert!(args.polar_chunks.len() >= MUON_COOPERATIVE_BLOCKS);
        let amax_chunk_count = args
            .matrix_len
            .div_ceil(NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32);
        assert!(args.polar_x_chunk_amax.len() >= amax_chunk_count as usize);

        self.apply.muon.tma_split.muon_tma_momentum_orient_kernel(
            args.stream,
            grid_x_config(args.matrix_len.div_ceil(CTA_THREADS), CTA_THREADS),
            args.slots,
            &mut *args.oriented,
            args.slot_index,
            args.mu,
            args.grad_scale,
        )?;

        self.apply
            .muon
            .tma_split
            .muon_tma_source_sumsq_chunks_kernel(
                args.stream,
                grid_x_config(MUON_COOPERATIVE_BLOCKS as u32, CTA_THREADS),
                &*args.oriented,
                &mut *args.polar_chunks,
                args.matrix_len,
            )?;

        self.apply
            .muon
            .tma_split
            .muon_tma_reduce_source_norm_kernel(
                args.stream,
                grid_x_config(1, CTA_THREADS),
                &mut *args.polar_chunks,
                MUON_COOPERATIVE_BLOCKS as u32,
            )?;

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
        )
    }

    pub fn muon_tma_finish_update(
        &self,
        mut args: MuonTmaFinishArgs<'_>,
    ) -> Result<(), DriverError> {
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
        self.muon_tma_update_master_and_amax(&mut args)
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
                args.polar_update,
                args.polar_bound_amax,
                &mut *args.polar_chunks,
                args.qk_clip_factors,
                args.slot_index,
                args.learning_rate,
                args.weight_decay,
                args.average_coefficient,
                args.schedule_beta,
                args.apply_polar_sqrt_bound,
            )?;

        self.apply
            .muon
            .tma_split
            .muon_tma_reduce_update_amax_kernel(
                args.stream,
                grid_x_config(1, CTA_THREADS),
                args.slots,
                &*args.polar_chunks,
                args.slot_index,
                chunk_count,
            )
    }

    pub fn muon_tma_finish_update_cooperative_reference(
        &self,
        args: MuonTmaFinishArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.slot_index < args.slots.len() as u32);
        assert!(args.matrix_len > 0);
        assert!(args.polar_chunks.len() >= 2 * MUON_COOPERATIVE_BLOCKS);
        self.apply.muon.tma_split.muon_tma_finish_update_kernel(
            args.stream,
            launch_config((MUON_COOPERATIVE_BLOCKS as u32, 1, 1), CTA_THREADS),
            args.slots,
            args.polar_update,
            args.polar_bound_amax,
            args.polar_chunks,
            args.slot_index,
            args.learning_rate,
            args.weight_decay,
            args.average_coefficient,
            args.schedule_beta,
            args.apply_polar_sqrt_bound,
        )
    }
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
