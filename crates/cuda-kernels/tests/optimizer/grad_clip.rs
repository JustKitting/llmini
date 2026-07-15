use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::optimizer::{GradientClipArgs, OptimizerModule};

use crate::common::{self, assert_slice_close};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn global_clip_scales_all_gradient_buffers_together() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;

    let first = DeviceBuffer::from_host(&stream, &[3.0_f32, 4.0, 0.0, 0.0])?;
    let second = DeviceBuffer::from_host(&stream, &[12.0_f32, 0.0, 0.0, 0.0])?;
    let chunk_ptrs =
        DeviceBuffer::from_host(&stream, &[first.cu_deviceptr(), second.cu_deviceptr()])?;
    let chunk_lens = DeviceBuffer::from_host(&stream, &[4_u32, 4])?;
    let mut chunk_sums = DeviceBuffer::<f32>::zeroed(&stream, 2)?;
    let mut scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut norm = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    module.clip_gradients(GradientClipArgs {
        stream: &stream,
        chunk_ptrs: &chunk_ptrs,
        chunk_lens: &chunk_lens,
        chunk_sums: &mut chunk_sums,
        scale: &mut scale,
        norm: &mut norm,
        chunk_count: 2,
        max_norm: 6.5,
        apply: true,
    })?;

    assert_slice_close(&first.to_host_vec(&stream)?, &[1.5, 2.0, 0.0, 0.0], 1.0e-6);
    assert_slice_close(&second.to_host_vec(&stream)?, &[6.0, 0.0, 0.0, 0.0], 1.0e-6);
    assert_slice_close(&scale.to_host_vec(&stream)?, &[0.5], 1.0e-6);
    assert_slice_close(&norm.to_host_vec(&stream)?, &[13.0], 1.0e-6);
    Ok(())
}
