use cuda_core::DriverError;

use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::launch::launch_config;

use super::super::args::{AuroraMegaUpdateArgs, AuroraTmaFinishArgs, AuroraTmaPrepareArgs};
use super::super::{AURORA_COOPERATIVE_BLOCKS, AURORA_MATRIX_PHASES};
use super::OptimizerModule;

impl OptimizerModule {
    pub fn aurora_mega_update(&self, args: AuroraMegaUpdateArgs<'_>) -> Result<(), DriverError> {
        assert_mega_args(&args);
        let matrix_count = args.slot_count / AURORA_MATRIX_PHASES as u32;
        self.apply
            .aurora
            .mega
            .aurora_mega_update_cooperative_kernel(
                args.stream,
                launch_config(
                    (AURORA_COOPERATIVE_BLOCKS as u32, matrix_count, 1),
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

    pub fn aurora_tma_prepare_polar(
        &self,
        args: AuroraTmaPrepareArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.slot_index < args.slots.len() as u32);
        self.apply.aurora.tma_split.aurora_tma_prepare_polar_kernel(
            args.stream,
            launch_config((AURORA_COOPERATIVE_BLOCKS as u32, 1, 1), CTA_THREADS),
            args.slots,
            args.oriented,
            args.polar_x,
            args.polar_chunks,
            args.slot_index,
            args.mu,
            args.grad_scale,
        )
    }

    pub fn aurora_tma_finish_update(
        &self,
        args: AuroraTmaFinishArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.slot_index < args.slots.len() as u32);
        self.apply.aurora.tma_split.aurora_tma_finish_update_kernel(
            args.stream,
            launch_config((AURORA_COOPERATIVE_BLOCKS as u32, 1, 1), CTA_THREADS),
            args.slots,
            args.polar_update,
            args.polar_chunks,
            args.slot_index,
            args.learning_rate,
            args.weight_decay,
            args.average_coefficient,
        )
    }
}

fn assert_mega_args(args: &AuroraMegaUpdateArgs<'_>) {
    let slots = args.slot_count as usize;
    let matrix_count = args.slot_count as usize / AURORA_MATRIX_PHASES;
    assert_eq!(args.slot_count as usize % AURORA_MATRIX_PHASES, 0);
    assert!(args.slots.len() >= slots);
    assert!(args.oriented.len() >= args.max_len as usize * matrix_count);
    assert!(args.polar_next.len() >= args.max_len as usize * matrix_count);
    assert!(args.polar_x.len() >= args.max_len as usize * matrix_count);
    assert!(args.polar_ax.len() >= args.max_ax_len as usize * matrix_count);
    assert!(args.polar_gram.len() >= args.max_dim as usize * args.max_dim as usize * matrix_count);
    assert!(args.polar_chunks.len() >= AURORA_COOPERATIVE_BLOCKS * matrix_count);
}
