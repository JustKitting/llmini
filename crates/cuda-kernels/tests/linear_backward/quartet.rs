use cuda_core::DeviceBuffer;
use rust_kernels_cuda::linear_backward::{LinearBackwardArgs, LinearBackwardModule};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::common;
use super::{INPUT_DIM, OUTPUT_DIM, TOKEN_COUNT, TOLERANCE, TestResult};

use common::nvfp4::{first_col_one_bytes, first_row_one_bytes, one_pair_bytes, one_scales};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn linear_backward_computes_dinput_and_dweight_from_quartet_operands() -> TestResult {
    let (_, stream, module) = common::cuda_test_module(LinearBackwardModule::from_module)?;

    let e_h_bytes_dev =
        DeviceBuffer::from_host(&stream, &first_col_one_bytes(TOKEN_COUNT, OUTPUT_DIM))?;
    let e_h_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(TOKEN_COUNT * OUTPUT_DIM))?;
    let e_h_global_scales_dev = DeviceBuffer::from_host(&stream, &vec![1.0_f32; TOKEN_COUNT])?;
    let weight_t_h_bytes_dev =
        DeviceBuffer::from_host(&stream, &first_col_one_bytes(INPUT_DIM, OUTPUT_DIM))?;
    let weight_t_h_scales_dev = DeviceBuffer::from_host(
        &stream,
        &rust_kernels_cuda::nvfp4::csf::encode_host(
            &one_scales(INPUT_DIM * OUTPUT_DIM),
            INPUT_DIM,
            OUTPUT_DIM,
        ),
    )?;
    let weight_t_h_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let e_t_h_bytes_dev =
        DeviceBuffer::from_host(&stream, &first_row_one_bytes(OUTPUT_DIM, TOKEN_COUNT))?;
    let e_t_h_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(OUTPUT_DIM * TOKEN_COUNT))?;
    let e_t_h_global_scales_dev = DeviceBuffer::from_host(&stream, &vec![1.0_f32; OUTPUT_DIM])?;
    let input_t_h_bytes_dev =
        DeviceBuffer::from_host(&stream, &one_pair_bytes(INPUT_DIM * TOKEN_COUNT))?;
    let input_t_h_scales_dev = DeviceBuffer::from_host(
        &stream,
        &rust_kernels_cuda::nvfp4::csf::encode_host(
            &one_scales(INPUT_DIM * TOKEN_COUNT),
            INPUT_DIM,
            TOKEN_COUNT,
        ),
    )?;
    let input_t_h_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let mut dinput_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * INPUT_DIM)?;
    let mut dweight_dev = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM * INPUT_DIM)?;

    module.backward(LinearBackwardArgs {
        stream: &stream,
        e_h: Nvfp4RowwiseDeviceTensor::new(&e_h_bytes_dev, &e_h_scales_dev, &e_h_global_scales_dev),
        weight_t_h: Nvfp4FourSixMmaWeightTensor::new(
            &weight_t_h_bytes_dev,
            &weight_t_h_scales_dev,
            &weight_t_h_global_scale_dev,
        ),
        e_t_h: Nvfp4RowwiseDeviceTensor::new(
            &e_t_h_bytes_dev,
            &e_t_h_scales_dev,
            &e_t_h_global_scales_dev,
        ),
        input_t_h: Nvfp4FourSixMmaWeightTensor::new(
            &input_t_h_bytes_dev,
            &input_t_h_scales_dev,
            &input_t_h_global_scale_dev,
        ),
        dinput: &mut dinput_dev,
        dweight: &mut dweight_dev,
        dbias: None,
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        output_dim: OUTPUT_DIM as u32,
    })?;

    let dinput = dinput_dev.to_host_vec(&stream)?;
    let dweight = dweight_dev.to_host_vec(&stream)?;

    assert!(dinput.iter().all(|value| (*value - 1.0).abs() <= TOLERANCE));

    for row in 0..OUTPUT_DIM {
        for col in 0..INPUT_DIM {
            let expected = if row == 0 { TOKEN_COUNT as f32 } else { 0.0 };
            common::assert_close(dweight[row * INPUT_DIM + col], expected, TOLERANCE);
        }
    }

    Ok(())
}
