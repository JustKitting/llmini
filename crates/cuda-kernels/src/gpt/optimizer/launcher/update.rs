use cuda_core::DriverError;

use super::super::args::{ScheduleFreeMaterializeArgs, ScheduleFreeMaterializePrecomputedArgs};
use super::super::threads::APPLY_THREADS_PER_BLOCK;
use super::OptimizerModule;
use crate::launch::{grid_x_config, launch_config};
use crate::nvfp4_quant::NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK;

const SCHEDULE_FREE_GROUP_SIZE: u32 = 16;

impl OptimizerModule {
    pub fn materialize_schedule_free_precomputed(
        &self,
        args: ScheduleFreeMaterializePrecomputedArgs<'_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.z_master.len() >= args.len as usize);
        assert!(args.x_master.len() >= args.len as usize);
        assert!(!args.amax.is_empty());
        assert!(args.bytes.len() >= args.len as usize / 2);
        assert!(args.scales.len() >= args.len as usize / 16);

        let groups_per_block = APPLY_THREADS_PER_BLOCK / SCHEDULE_FREE_GROUP_SIZE;
        self.apply.schedule_free.schedule_free_four_six_kernel(
            args.stream,
            launch_config(
                ((args.len / 16).div_ceil(groups_per_block), 1, 1),
                APPLY_THREADS_PER_BLOCK,
            ),
            args.z_master,
            args.x_master,
            args.amax,
            args.bytes,
            args.scales,
            args.global_scale,
            args.beta,
        )
    }

    pub fn materialize_schedule_free(
        &self,
        args: ScheduleFreeMaterializeArgs<'_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.z_master.len() >= args.len as usize);
        assert!(args.x_master.len() >= args.len as usize);
        assert!(args.bytes.len() >= args.len as usize / 2);
        assert!(args.scales.len() >= args.len as usize / 16);

        let chunk_count = args.len.div_ceil(NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32);
        self.apply.schedule_free.schedule_free_chunk_amax_kernel(
            args.stream,
            grid_x_config(chunk_count, APPLY_THREADS_PER_BLOCK),
            args.z_master,
            args.x_master,
            args.chunk_amax,
            args.beta,
            args.len,
        )?;

        self.quant.tensor_amax_from_chunks_f32(
            args.stream,
            &*args.chunk_amax,
            args.amax,
            chunk_count,
        )?;

        let groups_per_block = APPLY_THREADS_PER_BLOCK / SCHEDULE_FREE_GROUP_SIZE;
        self.apply.schedule_free.schedule_free_four_six_kernel(
            args.stream,
            launch_config(
                ((args.len / 16).div_ceil(groups_per_block), 1, 1),
                APPLY_THREADS_PER_BLOCK,
            ),
            args.z_master,
            args.x_master,
            &*args.amax,
            args.bytes,
            args.scales,
            args.global_scale,
            args.beta,
        )
    }
}
