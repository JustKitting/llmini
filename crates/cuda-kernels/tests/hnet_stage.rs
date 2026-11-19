use cuda_core::DeviceBuffer;
use rust_kernels_cuda::hnet_stage::{HnetStage, RatioObjective, StageShape};

mod common;

const HALF_INTEGERS: [u16; 13] = [
    0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980,
    0x4a00,
];

fn fp16(values: &[i8]) -> Vec<u16> {
    values
        .iter()
        .map(|&value| {
            HALF_INTEGERS[value.unsigned_abs() as usize] | if value < 0 { 1 << 15 } else { 0 }
        })
        .collect()
}

#[test]
#[ignore = "uses the CUDA H-Net RMSNorm and width-truncation adjoint"]
fn hnet_rms_uses_the_full_width_and_ignores_padding() -> Result<(), Box<dyn std::error::Error>> {
    use rust_kernels_cuda::{
        hnet_rms_norm::{RmsForward, RmsNormModule, RmsShape},
        nvfp4::{Nvfp4DeviceTensor, csf},
    };

    const WIDTH: usize = 128;
    let (_, stream, norm) = common::cuda_test_module(RmsNormModule::from_module)?;
    let mut rows = [1.0_f32; 2 * WIDTH];
    rows[WIDTH - 1] = 3.0;
    rows[WIDTH..].fill(f32::NAN);
    let input = DeviceBuffer::from_host(&stream, &rows)?;
    let lengths = DeviceBuffer::from_host(&stream, &[1_u32])?;
    let bytes = DeviceBuffer::from_host(&stream, &[0x22_u8; WIDTH / 2])?;
    let scales =
        DeviceBuffer::from_host(&stream, &csf::encode_host(&[0x38; WIDTH / 16], 1, WIDTH))?;
    let global = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let (value, tape) = norm.residual(RmsForward {
        stream: &stream,
        residual: &input,
        lengths: &lengths,
        scale: Nvfp4DeviceTensor::new(&bytes, &scales, &global),
        shape: RmsShape {
            batch: 1,
            stride: 2,
            width: WIDTH as u32,
            output_width: 2,
        },
    })?;
    let inverse = (136.0_f32 / WIDTH as f32 + 1e-5).sqrt().recip();
    common::assert_slice_close(
        &value.to_host_vec(&stream)?,
        &[inverse, inverse, 0.0, 0.0],
        1e-6,
    );

    let gradient = DeviceBuffer::from_host(&stream, &[1.0_f32, 0.0, f32::NAN, f32::NAN])?;
    let (d_input, d_scale) = norm.backward(&stream, tape, &gradient)?;
    let d_input = d_input.to_host_vec(&stream)?;
    let mean = inverse * inverse / WIDTH as f32;
    common::assert_close(d_input[0], inverse * (1.0 - mean), 2e-5);
    common::assert_close(d_input[WIDTH - 1], -3.0 * inverse * mean, 3e-5);
    assert!(d_input[WIDTH..].iter().all(|&grad| grad == 0.0));
    let d_scale = d_scale.to_host_vec(&stream)?;
    common::assert_close(d_scale[0], inverse, 3e-4);
    assert!(d_scale[1..].iter().all(|&grad| grad == 0.0));
    Ok(())
}

#[test]
#[ignore = "uses the CUDA H-Net stage and its generated sm_120a PTX"]
fn ragged_hnet_stage_matches_causal_forward_and_backward() -> Result<(), Box<dyn std::error::Error>>
{
    let (_, stream, stage) = common::cuda_test_module(HnetStage::from_module)?;
    let mut q = [1, 0, 0, 0].repeat(10);
    q[4..8].copy_from_slice(&[1, 1, 1, 1]);
    let mut k = [1, 0, 0, 0].repeat(10);
    k[4..8].copy_from_slice(&[0, 1, 0, 0]);
    k[8..12].copy_from_slice(&[-1, -1, -1, 1]);
    k[16] = -1;
    let q = fp16(&q);
    let k = fp16(&k);
    let encoded = fp16(&[2, 4, 9, 9, 6, 8, 9, 9, 10, 12, 1, 2, 9, 9, 9, 9, 9, 9, 9, 9]);
    let lengths = [5, 2];
    let d_expanded = [1.0; 20];
    let q_dev = DeviceBuffer::from_host(&stream, &q)?;
    let k_dev = DeviceBuffer::from_host(&stream, &k)?;
    let encoded_dev = DeviceBuffer::from_host(&stream, &encoded)?;
    let length_dev = DeviceBuffer::from_host(&stream, &lengths)?;
    let d_expanded_dev = DeviceBuffer::from_host(&stream, &d_expanded)?;
    let plan = stage.route(
        &stream,
        &q_dev,
        &k_dev,
        &length_dev,
        StageShape {
            batch: 2,
            outer_stride: 5,
            features: 2,
            router_width: 4,
            inner_alignment: 1,
        },
        RatioObjective {
            target: 4.0,
            weight: 0.03,
        },
    )?;
    assert_eq!(plan.inner_stride(), 3);
    assert_eq!(plan.inner_lengths().to_host_vec(&stream)?, [3, 1]);
    common::assert_slice_close(
        &plan.ratio_loss().to_host_vec(&stream)?,
        &[0.03 * 79.0 / 49.0],
        1e-6,
    );

    let gathered = stage.gather(&plan, &encoded_dev)?;
    assert_eq!(
        gathered.to_host_vec(&stream)?,
        fp16(&[2, 4, 6, 8, 10, 12, 1, 2, 0, 0, 0, 0])
    );
    let padding = DeviceBuffer::from_host(&stream, &fp16(&[3, 5]))?;
    let mut residual = DeviceBuffer::zeroed(&stream, 24)?;
    stage.write_residual(&plan, &gathered, 4, Some(&padding), &mut residual)?;
    assert_eq!(
        residual.to_host_vec(&stream)?,
        [
            2., 4., 3., 5., 6., 8., 3., 5., 10., 12., 3., 5., 1., 2., 3., 5., 0., 0., 0., 0., 0.,
            0., 0., 0.,
        ]
    );
    assert_eq!(
        stage
            .read_residual(&plan, &residual, 4)?
            .to_host_vec(&stream)?,
        gathered.to_host_vec(&stream)?
    );
    let d_narrow = DeviceBuffer::from_host(&stream, &[1.0; 12])?;
    let d_wide = stage.widen_output_adjoint(&plan, &d_narrow, 4)?;
    assert_eq!(
        d_wide.to_host_vec(&stream)?,
        [
            1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.,
        ]
    );
    let d_wide = DeviceBuffer::from_host(&stream, &[1.0, 2.0, 3.0, 4.0].repeat(6))?;
    let width_adjoint = stage.split_input_adjoint(&plan, &d_wide, 4)?;
    assert_eq!(
        width_adjoint.d_features.to_host_vec(&stream)?,
        [1., 2., 1., 2., 1., 2., 1., 2., 0., 0., 0., 0.]
    );
    assert_eq!(
        width_adjoint.d_padding.unwrap().to_host_vec(&stream)?,
        [12., 16.]
    );

    // Use the identity as the inner model to isolate the hierarchy operators.
    let expanded = stage.expand(&plan, &gathered)?;
    assert_eq!(
        expanded.values.to_host_vec(&stream)?,
        fp16(&[2, 4, 2, 4, 5, 7, 5, 7, 10, 12, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0,])
    );
    let reverse = stage.backward(&expanded, &d_expanded_dev)?;
    assert_eq!(
        reverse.d_inner.to_host_vec(&stream)?,
        [2.5, 2.5, 1.5, 1.5, 1., 1., 2., 2., 0., 0., 0., 0.]
    );
    let d_ratio = 0.03 * (4.0 / 3.0) * (16.0 / 7.0 - 1.0) / 7.0;
    // The tied pair continues the previous chunk, so its STE adjoint is
    // negative. The selected pair gets both the STE and smoothing terms.
    let d_tied_cosine = -0.5 * (-6.0 + d_ratio);
    let d_selected_cosine = -0.5 * (28.0 + d_ratio);
    let mut d_q = [0.0; 40];
    let mut d_k = [0.0; 40];
    d_q[1] = d_tied_cosine;
    d_k[4] = d_tied_cosine;
    for feature in 0..3 {
        d_q[4 + feature] = -0.125 * d_selected_cosine;
        d_k[8 + feature] = 0.125 * d_selected_cosine;
    }
    d_q[7] = 0.375 * d_selected_cosine;
    d_k[11] = 0.375 * d_selected_cosine;
    common::assert_slice_close(&reverse.d_queries.to_host_vec(&stream)?, &d_q, 1e-5);
    common::assert_slice_close(&reverse.d_keys.to_host_vec(&stream)?, &d_k, 1e-5);
    assert_eq!(
        stage
            .scatter(&plan, &reverse.d_inner)?
            .to_host_vec(&stream)?,
        [
            2.5, 2.5, 0., 0., 1.5, 1.5, 0., 0., 1., 1., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0.,
        ]
    );
    Ok(())
}
