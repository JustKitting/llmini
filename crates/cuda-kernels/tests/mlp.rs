use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::mlp::{MlpModule, Relu2BackwardF16Args, relu2_backward_amax_chunks};

mod common;

const LEN: usize = 4097;
const CHUNK_LEN: usize = 2048;

#[ignore = "requires the adaptive-ReLU2 Blackwell build"]
#[test]
fn adaptive_relu2_centers_each_row_and_preserves_routed_off_adjoints() -> Result<(), Box<dyn Error>>
{
    let (_, stream, module) = common::cuda_test_module(MlpModule::from_module)?;
    let input = [-2., -1., 1., 2., 3., 3., 3., 3., -3., -2., -1., -0.5];
    let mut values = DeviceBuffer::from_host(&stream, &input)?;
    let mut tape = DeviceBuffer::zeroed(&stream, input.len())?;
    module.adaptive_relu2_forward(&stream, &mut values, Some(&mut tape), 3, 4)?;
    assert_eq!(
        values.to_host_vec(&stream)?,
        [-1.25, -1.25, -0.25, 2.75, 0., 0., 0., 0., 0., 0., 0., 0.],
    );
    let masks = DeviceBuffer::from_host(&stream, &[1_u64])?;
    let mut gradient =
        DeviceBuffer::from_host(&stream, &[0., 0., 0., 1., 2., 2., 2., 2., 1., 2., 3., 4.])?;
    module.adaptive_relu2_backward(&stream, &mut gradient, &tape, &masks, 3, 4, false)?;
    assert_eq!(
        gradient.to_host_vec(&stream)?,
        [0., 0., -0.5, 3., 0., 0., 0., 0., 0., 0., 0., 0.],
    );

    const WIDTH: usize = 256;
    let mut z = vec![1.0; 2 * WIDTH];
    z[WIDTH / 2..WIDTH + WIDTH / 2].fill(2.0);
    let mut values = DeviceBuffer::from_host(&stream, &z)?;
    let mut tape = DeviceBuffer::zeroed(&stream, z.len())?;
    module.adaptive_relu2_forward(&stream, &mut values, Some(&mut tape), 2, WIDTH as u32)?;
    let mut dy = vec![100.0; z.len()];
    dy[..WIDTH / 2].fill(2.0);
    dy[WIDTH..WIDTH + WIDTH / 2].fill(-4.0);
    let mut gradient = DeviceBuffer::from_host(&stream, &dy)?;
    module.adaptive_relu2_backward(&stream, &mut gradient, &tape, &masks, 2, WIDTH as u32, true)?;
    let actual = gradient.to_host_vec(&stream)?;
    for (part, expected) in actual.chunks_exact(WIDTH / 2).zip([2., -4., -8., 4.]) {
        common::assert_all_close(part, expected, 0.);
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn relu2_backward_f16_writes_exact_output_and_chunk_amax() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(MlpModule::from_module)?;
    let (pre_bits, pre_values): (Vec<_>, Vec<_>) = (0..LEN).map(pre_activation).unzip();
    let d_out: Vec<f32> = (0..LEN)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.125)
        .collect();
    let expected: Vec<f32> = pre_values
        .iter()
        .zip(&d_out)
        .map(|(pre, grad)| grad * 2.0 * pre.max(0.0))
        .collect();
    let expected_chunk_amax: Vec<f32> = expected
        .chunks(CHUNK_LEN)
        .map(|chunk| chunk.iter().copied().map(f32::abs).fold(0.0, f32::max))
        .collect();

    let pre_dev = DeviceBuffer::from_host(&stream, &pre_bits)?;
    let d_out_dev = DeviceBuffer::from_host(&stream, &d_out)?;
    let mut d_pre_dev = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut chunk_amax_dev = DeviceBuffer::<f32>::zeroed(&stream, expected_chunk_amax.len())?;

    let chunk_count = module.relu2_backward_f16(Relu2BackwardF16Args {
        stream: &stream,
        pre_activation: &pre_dev,
        d_out: &d_out_dev,
        d_pre_activation: &mut d_pre_dev,
        d_pre_activation_chunk_amax: &mut chunk_amax_dev,
        len: LEN as u32,
    })?;

    assert_eq!(chunk_count, relu2_backward_amax_chunks(LEN as u32));
    assert_eq!(chunk_count as usize, expected_chunk_amax.len());
    assert_eq!(
        float_bits(&d_pre_dev.to_host_vec(&stream)?),
        float_bits(&expected)
    );
    assert_eq!(
        float_bits(&chunk_amax_dev.to_host_vec(&stream)?),
        float_bits(&expected_chunk_amax)
    );
    Ok(())
}

fn pre_activation(index: usize) -> (u16, f32) {
    match index % 5 {
        0 => (0x3c00, 1.0),
        1 => (0xbc00, -1.0),
        2 => (0x4000, 2.0),
        3 => (0x3800, 0.5),
        _ => (0x0000, 0.0),
    }
}

fn float_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}
