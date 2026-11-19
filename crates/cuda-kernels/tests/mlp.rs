use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::{
    mlp::{
        ADAPTIVE_RELU2_PACK_WIDTH, AdaptiveRelu2PackArgs, MlpModule, Relu2BackwardF16Args,
        relu2_backward_amax_chunks,
    },
    nvfp4::csf,
    nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseDerivedAmaxArgs},
};

mod common;

const LEN: usize = 4097;
const CHUNK_LEN: usize = 2048;

#[test]
#[ignore = "uses the native Blackwell adaptive MLP and NVFP4 activation packer"]
fn fused_adaptive_relu2_matches_separate_center_and_pack_bitwise() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let mlp = MlpModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let rows = 129;
    let width = ADAPTIVE_RELU2_PACK_WIDTH as usize;
    let len = rows * width;
    let scale_len = csf::rowwise_encoded_len(rows, width);
    let z: Vec<f32> = (0..len)
        .map(|index| match (index / width) % 7 {
            0 => 0.0,
            1 => -2.25,
            2 => 0.375,
            3 if index % width == 1 => 60_000.0,
            3 => 0.000_13,
            4 => 0.015_625 * ((index % 7) as f32 - 3.0),
            5 if index % 2 == 0 => -0.0,
            5 => f32::from_bits(1),
            _ => 0.013 * ((index.wrapping_mul(691) % 1291) as f32 - 701.0),
        })
        .collect();

    struct Packed {
        amax: DeviceBuffer<f32>,
        bytes: DeviceBuffer<u8>,
        scales: DeviceBuffer<u8>,
        globals: DeviceBuffer<f32>,
    }
    let allocate = || -> Result<Packed, Box<dyn Error>> {
        Ok(Packed {
            amax: DeviceBuffer::zeroed(&stream, rows)?,
            bytes: DeviceBuffer::zeroed(&stream, len / 2)?,
            scales: DeviceBuffer::zeroed(&stream, scale_len)?,
            globals: DeviceBuffer::zeroed(&stream, rows)?,
        })
    };

    for with_tape in [true, false] {
        let mut separate_values = DeviceBuffer::from_host(&stream, &z)?;
        let mut fused_values = DeviceBuffer::from_host(&stream, &z)?;
        let mut separate_tape = DeviceBuffer::zeroed(&stream, len)?;
        let mut fused_tape = DeviceBuffer::zeroed(&stream, len)?;
        let mut separate = allocate()?;
        let mut fused = allocate()?;
        mlp.adaptive_relu2_forward(
            &stream,
            &mut separate_values,
            with_tape.then_some(&mut separate_tape),
            rows as u32,
            width as u32,
        )?;
        quant.fp32_to_nvfp4_four_six_rowwise_derived_amax(Nvfp4QuantRowwiseDerivedAmaxArgs {
            stream: &stream,
            x: &separate_values,
            amax: &mut separate.amax,
            out_fp4: &mut separate.bytes,
            out_scales: &mut separate.scales,
            out_global_scale: &mut separate.globals,
            row_count: rows as u32,
            row_len: width as u32,
        })?;
        mlp.adaptive_relu2_pack(AdaptiveRelu2PackArgs {
            stream: &stream,
            values: &mut fused_values,
            tape: with_tape.then_some(&mut fused_tape),
            amax: &mut fused.amax,
            bytes: &mut fused.bytes,
            scales: &mut fused.scales,
            globals: &mut fused.globals,
            rows: rows as u32,
        })?;
        for (reference, candidate) in [
            (&separate_values, &fused_values),
            (&separate.amax, &fused.amax),
            (&separate.globals, &fused.globals),
        ] {
            assert_eq!(
                float_bits(&reference.to_host_vec(&stream)?),
                float_bits(&candidate.to_host_vec(&stream)?)
            );
        }
        assert_eq!(
            separate.bytes.to_host_vec(&stream)?,
            fused.bytes.to_host_vec(&stream)?
        );
        assert_eq!(
            separate.scales.to_host_vec(&stream)?,
            fused.scales.to_host_vec(&stream)?
        );
        assert_eq!(
            separate_tape.to_host_vec(&stream)?,
            fused_tape.to_host_vec(&stream)?
        );
    }
    Ok(())
}

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
    module.adaptive_relu2_backward(&stream, &mut gradient, &tape, &masks, 3, 4, false, None)?;
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
    module.adaptive_relu2_backward(
        &stream,
        &mut gradient,
        &tape,
        &masks,
        2,
        WIDTH as u32,
        true,
        None,
    )?;
    let actual = gradient.to_host_vec(&stream)?;
    for (part, expected) in actual.chunks_exact(WIDTH / 2).zip([2., -4., -8., 4.]) {
        common::assert_all_close(part, expected, 0.);
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn adaptive_relu2_backward_amax_matches_unfused_gradient_and_scan() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(MlpModule::from_module)?;
    let rows = 129;
    let width = ADAPTIVE_RELU2_PACK_WIDTH;
    let len = (rows * width) as usize;
    let dy: Vec<f32> = (0..len)
        .map(|index| ((index.wrapping_mul(691) % 1291) as f32 - 251.0) * 0.013)
        .collect();
    let tape: Vec<u16> = (0..len).map(|index| pre_activation(index).0).collect();
    let tape = DeviceBuffer::from_host(&stream, &tape)?;
    let masks =
        DeviceBuffer::from_host(&stream, &[0x0f1f_f00f_000f_aaaa_u64, 0x8300_9ff0_300f_5555])?;
    let chunk_count = relu2_backward_amax_chunks(len as u32);
    let mut chunk_amax = DeviceBuffer::zeroed(&stream, chunk_count as usize)?;

    for routed in [false, true] {
        let mut reference = DeviceBuffer::from_host(&stream, &dy)?;
        let mut fused = DeviceBuffer::from_host(&stream, &dy)?;
        assert_eq!(
            module.adaptive_relu2_backward(
                &stream,
                &mut reference,
                &tape,
                &masks,
                rows,
                width,
                routed,
                None,
            )?,
            None
        );
        assert_eq!(
            module.adaptive_relu2_backward(
                &stream,
                &mut fused,
                &tape,
                &masks,
                rows,
                width,
                routed,
                Some(&mut chunk_amax),
            )?,
            Some(chunk_count)
        );
        let reference = reference.to_host_vec(&stream)?;
        assert_eq!(
            float_bits(&fused.to_host_vec(&stream)?),
            float_bits(&reference)
        );
        let expected_amax: Vec<f32> = reference
            .chunks_exact(CHUNK_LEN)
            .map(|chunk| chunk.iter().copied().map(f32::abs).fold(0.0, f32::max))
            .collect();
        assert_eq!(
            float_bits(&chunk_amax.to_host_vec(&stream)?),
            float_bits(&expected_amax),
        );
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
