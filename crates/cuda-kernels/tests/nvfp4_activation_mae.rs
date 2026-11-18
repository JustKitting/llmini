//! Native counterexamples declared before the activation MAE4/6 implementation.
//! Failure modes: six-only selection, MSE substituted for MAE, wrong tie,
//! subgroup/row addressing, cached producers skipped, or weight policy changed.
use cuda_core::DeviceBuffer;
use rust_kernels_cuda::{
    nvfp4::csf,
    nvfp4_quant::{
        Nvfp4QuantArgs, Nvfp4QuantModule, Nvfp4QuantRowwiseArgs, Nvfp4QuantRowwiseDerivedAmaxArgs,
    },
};
use std::error::Error;

mod common;

fn examples(groups: usize) -> Vec<f32> {
    let mut values = vec![0.; groups * 16];
    for (group, block) in values.chunks_exact_mut(16).enumerate() {
        block[0] = 6.;
        match group % 4 {
            0 => block[1] = 4.5,      // MAE: four=0, six=.5.
            1 => block[1..].fill(1.), // MAE: four=3.75, six=0.
            2 => {
                block[1] = 4.5;
                block[2..5].fill(1.); // MAE chooses six; MSE chooses four.
            }
            _ => {} // Exact tie must choose six.
        }
    }
    values
}

fn expected(groups: usize, adaptive: bool) -> (Vec<u8>, Vec<u8>) {
    let mut payload = vec![0; groups * 8];
    let mut scales = vec![0x78; groups]; // E4M3 256, effective scale=1.
    for (group, block) in payload.chunks_exact_mut(8).enumerate() {
        match group % 4 {
            0 if adaptive => {
                block[0] = 0x56;
                scales[group] = 0x7c; // E4M3 384, effective scale=1.5.
            }
            0 => block[0] = 0x67,
            1 => {
                block.fill(0x22);
                block[0] = 0x27;
            }
            2 => block[..3].copy_from_slice(&[0x67, 0x22, 0x02]),
            _ => block[0] = 0x07,
        }
    }
    (payload, scales)
}

#[test]
#[ignore = "native CUDA activation MAE4/6 build; no training updates"]
fn activation_mae_four_six_uses_l1_ties_and_all_rowwise_producers() -> Result<(), Box<dyn Error>> {
    let (_, stream, quant) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;
    //16 exercises a partially active warp;5504 is the non-power-of-two path;
    //2048/8192 exercise their cached producers, including the MLP width.
    for cols in [16usize, 64, 2048, 5504, 8192] {
        let rows = if cols == 16 { 4 } else { 3 };
        let groups = rows * cols / 16;
        let x = DeviceBuffer::from_host(&stream, &examples(groups))?;
        let (expected_payload, expected_scales) = expected(groups, true);
        let expected_csf = csf::encode_rowwise_host(&expected_scales, rows, cols);
        for derived in [false, true] {
            let mut amax =
                DeviceBuffer::from_host(&stream, &vec![if derived { 0f32 } else { 6. }; rows])?;
            let mut payload = DeviceBuffer::<u8>::zeroed(&stream, groups * 8)?;
            let mut scales =
                DeviceBuffer::<u8>::zeroed(&stream, csf::rowwise_encoded_len(rows, cols))?;
            let mut global = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
            if derived {
                quant.fp32_to_nvfp4_four_six_rowwise_derived_amax(
                    Nvfp4QuantRowwiseDerivedAmaxArgs {
                        stream: &stream,
                        x: &x,
                        amax: &mut amax,
                        out_fp4: &mut payload,
                        out_scales: &mut scales,
                        out_global_scale: &mut global,
                        row_count: rows as u32,
                        row_len: cols as u32,
                    },
                )?;
            } else {
                quant.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
                    stream: &stream,
                    x: &x,
                    amax: &amax,
                    out_fp4: &mut payload,
                    out_scales: &mut scales,
                    out_global_scale: &mut global,
                    group_count: groups as u32,
                    row_len: cols as u32,
                })?;
            }
            assert_eq!(
                payload.to_host_vec(&stream)?,
                expected_payload,
                "cols={cols} derived={derived} payload"
            );
            assert_eq!(
                scales.to_host_vec(&stream)?,
                expected_csf,
                "cols={cols} derived={derived} scales"
            );
            assert_eq!(global.to_host_vec(&stream)?, vec![1f32 / 256.; rows]);
            assert_eq!(amax.to_host_vec(&stream)?, vec![6f32; rows]);
        }
    }
    Ok(())
}

#[test]
#[ignore = "native CUDA activation MAE4/6 build; weights remain six-only"]
fn activation_mae_four_six_preserves_tensor_weight_quantization() -> Result<(), Box<dyn Error>> {
    let (_, stream, quant) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;
    let groups = 8;
    let x = DeviceBuffer::from_host(&stream, &examples(groups))?;
    let amax = DeviceBuffer::from_host(&stream, &[6f32])?;
    let mut payload = DeviceBuffer::<u8>::zeroed(&stream, groups * 8)?;
    let empty = csf::encode_host(&vec![0; groups], 1, groups * 16);
    let mut scales = DeviceBuffer::from_host(&stream, &empty)?;
    let mut global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    quant.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
        stream: &stream,
        x: &x,
        amax: &amax,
        out_fp4: &mut payload,
        out_scales: &mut scales,
        out_global_scale: &mut global,
        group_count: groups as u32,
    })?;
    let (expected_payload, expected_scales) = expected(groups, false);
    assert_eq!(payload.to_host_vec(&stream)?, expected_payload);
    assert_eq!(
        scales.to_host_vec(&stream)?,
        csf::encode_host(&expected_scales, 1, groups * 16)
    );
    assert_eq!(global.to_host_vec(&stream)?, vec![1f32 / 256.]);
    Ok(())
}
