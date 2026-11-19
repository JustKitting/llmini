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
#[ignore = "uses the CUDA H-Net stage and its generated sm_120a PTX"]
fn ragged_hnet_stage_matches_causal_forward_and_backward() -> Result<(), Box<dyn std::error::Error>>
{
    let (_, stream, stage) = common::cuda_test_module(HnetStage::from_module)?;
    let q = fp16(&[
        1, 0, 1, 0, 0, 1, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 0, 1, -1, 0,
    ]);
    let k = fp16(&[1, 0].repeat(10));
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
            router_width: 2,
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
        &[0.03 * 10.0 / 7.0],
        1e-6,
    );

    let gathered = stage.gather(&plan, &encoded_dev)?;
    assert_eq!(
        gathered.to_host_vec(&stream)?,
        fp16(&[2, 4, 6, 8, 10, 12, 1, 2, 0, 0, 0, 0])
    );
    // Use the identity as the inner model to isolate the hierarchy operators.
    let expanded = stage.expand(&plan, &gathered)?;
    assert_eq!(
        expanded.values.to_host_vec(&stream)?,
        fp16(&[2, 4, 2, 4, 4, 6, 4, 6, 10, 12, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0,])
    );
    let reverse = stage.backward(&expanded, &d_expanded_dev)?;
    assert_eq!(
        reverse.d_inner.to_host_vec(&stream)?,
        [3., 3., 1., 1., 1., 1., 2., 2., 0., 0., 0., 0.]
    );
    let d_ratio = 0.03 * (4.0 / 3.0) * (16.0 / 7.0 - 1.0) / 7.0;
    let d_cosine = -0.5 * (26.0 + d_ratio);
    let mut d_q = [0.0; 20];
    let mut d_k = [0.0; 20];
    d_q[4] = d_cosine;
    d_k[3] = d_cosine;
    common::assert_slice_close(&reverse.d_queries.to_host_vec(&stream)?, &d_q, 1e-5);
    common::assert_slice_close(&reverse.d_keys.to_host_vec(&stream)?, &d_k, 1e-5);
    assert_eq!(
        stage
            .scatter(&plan, &reverse.d_inner)?
            .to_host_vec(&stream)?,
        [
            3., 3., 0., 0., 1., 1., 0., 0., 1., 1., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0.,
        ]
    );
    Ok(())
}
