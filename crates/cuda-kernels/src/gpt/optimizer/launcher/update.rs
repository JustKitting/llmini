use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::super::args::{
    ScheduleFreeMaterializeArgs, ScheduleFreeMaterializePrecomputedArgs, SymExpLinScaleRefs,
};
use super::super::threads::APPLY_THREADS_PER_BLOCK;
use super::OptimizerModule;
use crate::launch::{grid_x_config, launch_config};
use crate::nvfp4_quant::{NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK, Nvfp4QuantArgs};

const SCHEDULE_FREE_THREADS_PER_GROUP: u32 = 4;

fn zero_amax(stream: &CudaStream, output_amax: &DeviceBuffer<f32>) -> Result<(), DriverError> {
    unsafe {
        cuda_core::memory::memset_d8_async(
            output_amax.cu_deviceptr(),
            0,
            std::mem::size_of::<f32>(),
            stream.cu_stream(),
        )
    }
}

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
        assert!(
            args.scales.len()
                >= crate::nvfp4::csf::encoded_len(args.scale_rows as usize, args.scale_k as usize)
        );
        assert!(u64::from(args.scale_rows) * u64::from(args.scale_k) >= u64::from(args.len));
        if let Some(values) = args.precomputed_values {
            assert!(values.len() >= args.len as usize);
            return self.quant.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
                stream: args.stream,
                x: values,
                amax: args.amax,
                out_fp4: args.bytes,
                out_scales: args.scales,
                out_global_scale: args.global_scale,
                group_count: args.len / 16,
            });
        }
        let scale_ptrs = symexp_lin_scale_ptrs(args.symexp_lin_scales);

        let groups_per_block = APPLY_THREADS_PER_BLOCK / SCHEDULE_FREE_THREADS_PER_GROUP;
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
            args.symexp_lin_beta,
            scale_ptrs[0],
            scale_ptrs[1],
            scale_ptrs[2],
            scale_ptrs[3],
            scale_ptrs[4],
            scale_ptrs[5],
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
        assert!(
            args.scales.len()
                >= crate::nvfp4::csf::encoded_len(args.scale_rows as usize, args.scale_k as usize)
        );
        assert!(u64::from(args.scale_rows) * u64::from(args.scale_k) >= u64::from(args.len));
        let scale_ptrs = symexp_lin_scale_ptrs(args.symexp_lin_scales);

        let chunk_count = args.len.div_ceil(NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32);
        zero_amax(args.stream, &*args.amax)?;
        self.apply.schedule_free.schedule_free_amax_kernel(
            args.stream,
            grid_x_config(chunk_count, APPLY_THREADS_PER_BLOCK),
            args.z_master,
            args.x_master,
            &*args.amax,
            args.beta,
            args.symexp_lin_beta,
            scale_ptrs[0],
            scale_ptrs[1],
            scale_ptrs[2],
            scale_ptrs[3],
            scale_ptrs[4],
            scale_ptrs[5],
            args.len,
        )?;

        let groups_per_block = APPLY_THREADS_PER_BLOCK / SCHEDULE_FREE_THREADS_PER_GROUP;
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
            args.symexp_lin_beta,
            scale_ptrs[0],
            scale_ptrs[1],
            scale_ptrs[2],
            scale_ptrs[3],
            scale_ptrs[4],
            scale_ptrs[5],
        )
    }
}

fn symexp_lin_scale_ptrs(scales: Option<SymExpLinScaleRefs<'_>>) -> [u64; 6] {
    scales.map_or([0; 6], |scales| {
        [
            scales.exponential_z.cu_deviceptr(),
            scales.exponential_x.cu_deviceptr(),
            scales.linear_z.cu_deviceptr(),
            scales.linear_x.cu_deviceptr(),
            scales.curvature_z.cu_deviceptr(),
            scales.curvature_x.cu_deviceptr(),
        ]
    })
}
