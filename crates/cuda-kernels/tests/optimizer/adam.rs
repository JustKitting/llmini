use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::optimizer::{AdamWUpdateArgs, OptimizerModule};

use crate::common::nvfp4::{one_pair_bytes, one_scales};
use crate::common::{self, assert_all_close};

const LEN: usize = 32;
const GRAD_SCALE: f32 = 0.5;

struct AdamFixture {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    z_master: DeviceBuffer<f32>,
    x_master: DeviceBuffer<f32>,
    grad: DeviceBuffer<f32>,
    first: DeviceBuffer<f32>,
    second: DeviceBuffer<f32>,
    amax: DeviceBuffer<f32>,
    chunk_amax: DeviceBuffer<f32>,
    global_scale: DeviceBuffer<f32>,
}

impl AdamFixture {
    fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            bytes: DeviceBuffer::from_host(stream, &one_pair_bytes(LEN))?,
            scales: DeviceBuffer::from_host(stream, &one_scales(LEN))?,
            z_master: DeviceBuffer::from_host(stream, &[1.0_f32; LEN])?,
            x_master: DeviceBuffer::from_host(stream, &[1.0_f32; LEN])?,
            grad: DeviceBuffer::from_host(stream, &[0.5_f32; LEN])?,
            first: DeviceBuffer::<f32>::zeroed(stream, LEN)?,
            second: DeviceBuffer::<f32>::zeroed(stream, LEN)?,
            amax: DeviceBuffer::<f32>::zeroed(stream, 1)?,
            chunk_amax: DeviceBuffer::<f32>::zeroed(stream, 1)?,
            global_scale: DeviceBuffer::from_host(stream, &[1.0_f32])?,
        })
    }

    fn apply(
        &mut self,
        stream: &CudaStream,
        module: &OptimizerModule,
        average_coefficient: f32,
    ) -> Result<(), Box<dyn Error>> {
        module.apply_adamw_update(AdamWUpdateArgs {
            stream,
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scale: &mut self.global_scale,
            z_master: &mut self.z_master,
            x_master: &mut self.x_master,
            grad: &self.grad,
            grad_scale: GRAD_SCALE,
            first_moment: &mut self.first,
            second_moment: &mut self.second,
            amax: &mut self.amax,
            chunk_amax: &mut self.chunk_amax,
            len: LEN as u32,
            learning_rate: 0.25,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            beta1_correction: 0.1,
            beta2_correction: 0.05,
            eps: 1.0e-10,
            average_coefficient,
        })?;
        Ok(())
    }
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_adamw_update_tracks_moments_and_requantizes() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let mut fixture = AdamFixture::new(&stream)?;

    fixture.apply(&stream, &module, 1.0)?;
    assert_all_close(&fixture.z_master.to_host_vec(&stream)?, 0.725, 1.0e-6);
    assert_all_close(&fixture.x_master.to_host_vec(&stream)?, 0.725, 1.0e-6);
    assert_all_close(&fixture.first.to_host_vec(&stream)?, 0.025, 1.0e-6);
    assert_all_close(&fixture.second.to_host_vec(&stream)?, 0.003125, 1.0e-6);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_adamw_update_applies_schedule_free_average() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let mut fixture = AdamFixture::new(&stream)?;

    fixture.apply(&stream, &module, 0.25)?;
    assert_all_close(&fixture.z_master.to_host_vec(&stream)?, 0.725, 1.0e-6);
    assert_all_close(&fixture.x_master.to_host_vec(&stream)?, 0.93125, 1.0e-6);
    Ok(())
}
