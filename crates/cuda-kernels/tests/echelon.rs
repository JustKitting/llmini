use cuda_core::DeviceBuffer;
use rust_kernels_cuda::optimizer::echelon::{
    EchelonModule, FrameModule, Shape, TailArgs, TailScratch,
};

mod common;

#[test]
#[ignore = "runs native thin-frame Householder QR on CUDA"]
fn echelon_householder_frame_preserves_the_column_space() -> Result<(), Box<dyn std::error::Error>>
{
    let (_, stream, module) = common::cuda_test_module(FrameModule::from_module)?;
    let input = [
        2.0, -1.0, 0.0, 1.0, 1.0, 0.0, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
    ];
    let mut factors = DeviceBuffer::from_host(&stream, &input)?;
    let mut frames = DeviceBuffer::from_host(&stream, &[0.0; 16])?;
    module.orthonormalize(&stream, &mut factors, &mut frames, 2, 2, 4)?;
    let output = frames.to_host_vec(&stream)?;
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    for (source, frame) in input.chunks_exact(8).zip(output.chunks_exact(8)) {
        let (q0, q1) = frame.split_at(4);
        common::assert_close(dot(q0, q0), 1.0, 2e-6);
        common::assert_close(dot(q1, q1), 1.0, 2e-6);
        common::assert_close(dot(q0, q1), 0.0, 2e-6);
        for vector in source.chunks_exact(4) {
            let (c0, c1) = (dot(q0, vector), dot(q1, vector));
            for (col, &element) in vector.iter().enumerate() {
                common::assert_close(c0 * q0[col] + c1 * q1[col], element, 2e-6);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "runs the native Echelon tail operations on CUDA"]
fn echelon_tail_removes_the_head_and_reuses_its_warm_vector()
-> Result<(), Box<dyn std::error::Error>> {
    let (_, stream, module) = common::cuda_test_module(EchelonModule::from_module)?;
    let shape = Shape::new(3, 4, 1)?;
    let mut state = module.new_tail_state(&stream, shape, 61)?;
    let mut scratch = TailScratch::new(&stream, shape)?;
    let momentum = DeviceBuffer::from_host(
        &stream,
        &[4.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )?;
    let projection = [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut residual = DeviceBuffer::from_host(&stream, &projection)?;
    module.residual(&stream, &momentum, &mut residual, shape)?;
    let mut estimate = |residual| -> Result<_, Box<dyn std::error::Error>> {
        Ok(module
            .estimate_tail(TailArgs {
                stream: &stream,
                residual,
                state: &mut state,
                scratch: &mut scratch,
            })?
            .to_host_vec(&stream)?[0])
    };
    // The retained head singular value is 4 and the residual singular values
    // are 2 and 1. Two iterations from the equal-magnitude seed have a closed form.
    let cold = estimate(&residual)?;
    common::assert_close(cold, 1.0 / (1.05 * (65.0_f32 / 17.0).sqrt()), 1e-6);
    let warm = estimate(&residual)?;
    assert!(warm < cold);
    common::assert_close(warm, (4097.0_f32 / 16385.0).sqrt() / 1.05, 1e-6);

    let zero_tail = DeviceBuffer::from_host(&stream, &projection)?;
    let mut zero_residual = DeviceBuffer::from_host(&stream, &projection)?;
    module.residual(&stream, &zero_tail, &mut zero_residual, shape)?;
    assert_eq!(estimate(&zero_residual)?, 0.0);
    Ok(())
}
