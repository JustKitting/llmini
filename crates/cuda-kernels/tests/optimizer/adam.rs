use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::optimizer::{AdamWUpdateArgs, OptimizerModule, SymExpLinSlotDescriptor};

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
            grad: &mut self.grad,
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
            schedule_beta: 0.0,
        })?;
        Ok(())
    }
}

struct ScalarAdamFixture {
    z_master: DeviceBuffer<f32>,
    x_master: DeviceBuffer<f32>,
    grad: DeviceBuffer<f32>,
    first: DeviceBuffer<f32>,
    second: DeviceBuffer<f32>,
}

impl ScalarAdamFixture {
    fn new(stream: &CudaStream, grad: f32) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            z_master: DeviceBuffer::from_host(stream, &[1.0])?,
            x_master: DeviceBuffer::from_host(stream, &[1.0])?,
            grad: DeviceBuffer::from_host(stream, &[grad])?,
            first: DeviceBuffer::from_host(stream, &[0.0])?,
            second: DeviceBuffer::from_host(stream, &[0.0])?,
        })
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

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn deferred_adamw_update_precomputes_next_schedule_amax() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let mut fixture = AdamFixture::new(&stream)?;
    let schedule_beta = 0.4;

    module.apply_adamw_update_deferred_quantization(AdamWUpdateArgs {
        stream: &stream,
        bytes: &mut fixture.bytes,
        scales: &mut fixture.scales,
        global_scale: &mut fixture.global_scale,
        z_master: &mut fixture.z_master,
        x_master: &mut fixture.x_master,
        grad: &mut fixture.grad,
        grad_scale: GRAD_SCALE,
        first_moment: &mut fixture.first,
        second_moment: &mut fixture.second,
        amax: &mut fixture.amax,
        chunk_amax: &mut fixture.chunk_amax,
        len: LEN as u32,
        learning_rate: 0.25,
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        beta1_correction: 0.1,
        beta2_correction: 0.05,
        eps: 1.0e-10,
        average_coefficient: 0.25,
        schedule_beta,
    })?;

    let z = fixture.z_master.to_host_vec(&stream)?[0];
    let x = fixture.x_master.to_host_vec(&stream)?[0];
    let expected_amax = (z + schedule_beta * (x - z)).abs();
    assert_all_close(&fixture.amax.to_host_vec(&stream)?, expected_amax, 1.0e-6);
    assert_all_close(&fixture.grad.to_host_vec(&stream)?, expected_amax, 1.0e-6);

    let pointers = DeviceBuffer::from_host(&stream, &[fixture.amax.cu_deviceptr()])?;
    module.zero_f32_pointer_table(&stream, &pointers, 1)?;
    assert_eq!(fixture.amax.to_host_vec(&stream)?, vec![0.0]);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn symexp_lin_batched_scalar_adam_matches_independent_updates() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(OptimizerModule::from_module)?;
    let exponential = ScalarAdamFixture::new(&stream, 0.5)?;
    let linear = ScalarAdamFixture::new(&stream, -0.25)?;
    let curvature = ScalarAdamFixture::new(&stream, 0.75)?;
    let descriptor = SymExpLinSlotDescriptor {
        grad: 0,
        z_master: 0,
        x_master: 0,
        exponential_z: exponential.z_master.cu_deviceptr(),
        exponential_x: exponential.x_master.cu_deviceptr(),
        exponential_grad: exponential.grad.cu_deviceptr(),
        exponential_first: exponential.first.cu_deviceptr(),
        exponential_second: exponential.second.cu_deviceptr(),
        linear_z: linear.z_master.cu_deviceptr(),
        linear_x: linear.x_master.cu_deviceptr(),
        linear_grad: linear.grad.cu_deviceptr(),
        linear_first: linear.first.cu_deviceptr(),
        linear_second: linear.second.cu_deviceptr(),
        curvature_z: curvature.z_master.cu_deviceptr(),
        curvature_x: curvature.x_master.cu_deviceptr(),
        curvature_grad: curvature.grad.cu_deviceptr(),
        curvature_first: curvature.first.cu_deviceptr(),
        curvature_second: curvature.second.cu_deviceptr(),
        rows: 0,
        cols: 0,
    };
    let slots = DeviceBuffer::from_host(&stream, &[descriptor])?;

    module.symexp_lin_scale_adamw_update(
        &stream, &slots, 1, 0.5, 0.25, 0.1, 0.1, 0.9, 0.95, 0.1, 0.05, 1.0e-10, 0.25,
    )?;
    assert_all_close(&exponential.z_master.to_host_vec(&stream)?, 0.725, 1.0e-6);
    assert_all_close(&exponential.x_master.to_host_vec(&stream)?, 0.93125, 1.0e-6);
    assert_all_close(&linear.z_master.to_host_vec(&stream)?, 1.225, 1.0e-6);
    assert_all_close(&linear.x_master.to_host_vec(&stream)?, 1.05625, 1.0e-6);
    assert_all_close(&curvature.z_master.to_host_vec(&stream)?, 0.9, 1.0e-6);
    assert_all_close(&curvature.x_master.to_host_vec(&stream)?, 0.975, 1.0e-6);

    module.symexp_lin_zero_scale_grads(&stream, &slots, 1)?;
    assert_eq!(exponential.grad.to_host_vec(&stream)?, vec![0.0]);
    assert_eq!(linear.grad.to_host_vec(&stream)?, vec![0.0]);
    assert_eq!(curvature.grad.to_host_vec(&stream)?, vec![0.0]);
    Ok(())
}
