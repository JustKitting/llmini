use cuda_core::DeviceBuffer;
use rust_kernels_cuda::hnet_byte_head::{
    BYTE_EOS, BYTE_LOGIT_STRIDE, BYTE_PADDING_TARGET, ByteHeadModule, ByteLossArgs,
};

mod common;

#[test]
#[ignore = "requires the CUDA byte head"]
fn masked_byte_loss_scores_eos_and_ignores_projection_padding()
-> Result<(), Box<dyn std::error::Error>> {
    let (_, stream, head) = common::cuda_test_module(ByteHeadModule::from_module)?;
    let stride = BYTE_LOGIT_STRIDE as usize;
    let classes = (BYTE_EOS + 1) as usize;
    let rows = 3;
    // Unused projection columns have higher logits than any actual byte.
    let mut values = vec![1000.0; stride * rows];
    values[..classes].fill(0.0);
    values[stride..2 * stride].fill(f32::NAN);
    values[2 * stride..2 * stride + classes].fill(-32.0);
    values[2 * stride + b'A' as usize] = 32.0;
    let targets = [BYTE_EOS, BYTE_PADDING_TARGET, b'A' as u32];
    let logits = DeviceBuffer::from_host(&stream, &values)?;
    let targets = DeviceBuffer::from_host(&stream, &targets)?;
    let mut loss = DeviceBuffer::zeroed(&stream, rows)?;
    let mut dlogits = DeviceBuffer::zeroed(&stream, stride * rows)?;
    let mut row_amax = DeviceBuffer::zeroed(&stream, rows)?;
    head.loss(ByteLossArgs {
        stream: &stream,
        logits: &logits,
        targets: &targets,
        losses: &mut loss,
        dlogits: &mut dlogits,
        row_amax: &mut row_amax,
        rows: rows as u32,
        valid_targets: 2,
    })?;
    let loss = loss.to_host_vec(&stream)?;
    let dlogits = dlogits.to_host_vec(&stream)?;
    let row_amax = row_amax.to_host_vec(&stream)?;
    common::assert_close(loss[0], (classes as f32).ln(), 1e-6);
    assert_eq!(loss[1], 0.0);
    let tiny = 256.0_f32 * (-64.0_f32).exp();
    common::assert_close(loss[2] / tiny, 1.0, 2e-5);
    common::assert_close(
        dlogits[BYTE_EOS as usize],
        -256.0 / (2.0 * classes as f32),
        1e-6,
    );
    common::assert_close(
        dlogits[2 * stride + b'A' as usize] / (-0.5 * tiny),
        1.0,
        2e-5,
    );
    assert_eq!(row_amax[1], 0.0);
    assert!(dlogits[stride..2 * stride].iter().all(|d| *d == 0.0));
    for row in dlogits.chunks_exact(stride) {
        assert!(row[classes..].iter().all(|d| *d == 0.0));
    }
    Ok(())
}

#[test]
#[ignore = "requires the CUDA NVFP4 readout"]
fn byte_projection_preserves_the_physical_output_stride() -> Result<(), Box<dyn std::error::Error>>
{
    use common::nvfp4::{one_scales, set_e2m1_one};
    use rust_kernels_cuda::{
        hnet_byte_head::ByteProjection,
        nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor, csf},
        nvfp4_tma_matmul::{launcher::Nvfp4GemmModule, tma::TmaNvfp4DeviceScaleDescriptors},
    };

    const ROWS: usize = 128;
    const WIDTH: usize = 128;
    const STRIDE: usize = BYTE_LOGIT_STRIDE as usize;
    let (_, stream, ptx) = common::cuda_test_context()?;
    let head = ByteHeadModule::from_module(ptx.clone())?;
    let tma = Nvfp4GemmModule::from_module(ptx)?;
    let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&stream)?;
    let mut values = vec![0; ROWS * WIDTH / 2];
    for row in 0..ROWS {
        set_e2m1_one(&mut values, row * WIDTH + row);
    }
    let mut weights = vec![0; STRIDE * WIDTH / 2];
    for class in 0..STRIDE {
        set_e2m1_one(&mut weights, class * WIDTH + class % WIDTH);
    }
    let input_scales = one_scales(ROWS * WIDTH);
    let input_globals = vec![1.0; ROWS];
    let weight_scales = csf::encode_host(&one_scales(STRIDE * WIDTH), STRIDE, WIDTH);
    let weight_global = [1.0];

    let input_bytes = DeviceBuffer::from_host(&stream, &values)?;
    let input_scales = DeviceBuffer::from_host(&stream, &input_scales)?;
    let input_globals = DeviceBuffer::from_host(&stream, &input_globals)?;
    let weight_bytes = DeviceBuffer::from_host(&stream, &weights)?;
    let weight_scales = DeviceBuffer::from_host(&stream, &weight_scales)?;
    let weight_global = DeviceBuffer::from_host(&stream, &weight_global)?;
    let mut logits = DeviceBuffer::zeroed(&stream, ROWS * STRIDE)?;
    head.project(ByteProjection {
        stream: &stream,
        tma: &tma,
        descriptors: &mut descriptors,
        decoder_values: Nvfp4RowwiseDeviceTensor::new(&input_bytes, &input_scales, &input_globals),
        weights: Nvfp4DeviceTensor::new(&weight_bytes, &weight_scales, &weight_global),
        logits: &mut logits,
        rows: ROWS as u32,
        width: WIDTH as u32,
    })?;
    let values = logits.to_host_vec(&stream)?;
    for (row, logits) in values.chunks_exact(STRIDE).enumerate() {
        for (class, &value) in logits.iter().enumerate() {
            assert_eq!(value, if row == class % WIDTH { 1.0 } else { 0.0 });
        }
    }
    Ok(())
}
