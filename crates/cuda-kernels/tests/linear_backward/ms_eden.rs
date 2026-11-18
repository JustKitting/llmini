use cuda_core::DeviceBuffer;
use rust_kernels_cuda::linear_backward::{
    LinearBackwardInputTranspose, LinearBackwardModule, LinearBackwardMsEdenArgs,
    LinearBackwardMsEdenScratchBuffers, LinearBackwardWeightTranspose,
};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use super::common;
use super::{TOLERANCE, TestResult};
const INPUT_DIM: usize = 128;
const OUTPUT_DIM: usize = 128;
const TOKEN_COUNT: usize = 128;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn linear_backward_ms_eden_quantizes_before_gemms() -> TestResult {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LinearBackwardModule::from_module(ptx.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(ptx)?;

    let e = patterned_matrix(TOKEN_COUNT, OUTPUT_DIM, 0.015625);
    let weight_t = patterned_matrix(INPUT_DIM, OUTPUT_DIM, 0.03125);
    let input_t = patterned_matrix(INPUT_DIM, TOKEN_COUNT, 0.03125);

    let e_dev = DeviceBuffer::from_host(&stream, &e)?;
    let weight_t_dev = DeviceBuffer::from_host(&stream, &weight_t)?;
    let input_t_dev = DeviceBuffer::from_host(&stream, &input_t)?;
    let mut scratch =
        LinearBackwardMsEdenScratchBuffers::new(&stream, TOKEN_COUNT, INPUT_DIM, OUTPUT_DIM)?;

    let mut dinput_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * INPUT_DIM)?;
    let mut dweight_dev = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM * INPUT_DIM)?;
    let mut dbias_dev = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM)?;

    module.backward_ms_eden(LinearBackwardMsEdenArgs {
        stream: &stream,
        quant_module: &quant_module,
        e: &e_dev,
        weight_t: LinearBackwardWeightTranspose::Fp32(&weight_t_dev),
        input_t: LinearBackwardInputTranspose::Fp32(&input_t_dev),
        scratch: scratch.as_args(),
        dinput: &mut dinput_dev,
        dweight: &mut dweight_dev,
        dbias: Some(&mut dbias_dev),
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        output_dim: OUTPUT_DIM as u32,
        sign_seed: 0x1234_5678,
        scale_seed: 0x9abc_def0,
        precomputed_e_amax_chunks: None,
    })?;

    let dinput = dinput_dev.to_host_vec(&stream)?;
    let dweight = dweight_dev.to_host_vec(&stream)?;
    let dbias = dbias_dev.to_host_vec(&stream)?;
    let e_quant = scratch.e_h.bytes.to_host_vec(&stream)?;
    let e_amax = scratch.e_h.chunk_amax.to_host_vec(&stream)?;
    let generated_scales = [
        scratch.e_h.global_scale.to_host_vec(&stream)?[0],
        scratch.weight_t_h.global_scale.to_host_vec(&stream)?[0],
        scratch.input_t_h.global_scale.to_host_vec(&stream)?[0],
    ];

    assert!(dinput.iter().all(|value| value.is_finite()));
    assert!(dweight.iter().all(|value| value.is_finite()));
    assert!(dinput.iter().any(|value| value.abs() > TOLERANCE));
    assert!(dweight.iter().any(|value| value.abs() > TOLERANCE));
    for col in 0..OUTPUT_DIM {
        let expected = (0..TOKEN_COUNT)
            .map(|row| e[row * OUTPUT_DIM + col])
            .sum::<f32>();
        common::assert_close(dbias[col], expected, TOLERANCE);
    }
    assert!(e_quant.iter().any(|byte| *byte != 0));
    assert!(e_amax.iter().all(|amax| amax.is_finite()));
    assert!(e_amax.iter().any(|amax| *amax > 0.0));
    assert!(
        generated_scales
            .iter()
            .all(|scale| *scale > 0.0 && scale.is_finite())
    );

    Ok(())
}

fn patterned_matrix(rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            ((row as f32 * 0.5) + (col as f32 - 31.5)) * scale
        })
        .collect()
}
