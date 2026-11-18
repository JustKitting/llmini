//! Declared before reverse vector fusion: preserve the full original CUDA
//! post/pre/RMS adjoint chain, including its reduction and addition order.
use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::mhc::kernels as legacy;
use rust_kernels_cuda::mhc_reverse::{FinishArgs, MhcReverseModule, PostArgs, PreArgs};
use std::error::Error;
mod common;
const GUARD: usize = 257;
fn value(i: usize, salt: usize) -> f32 {
    match i % 97 {
        0 => -0.0,
        1 => 1e-30,
        2 => -1e-30,
        3 => 100.0,
        _ => (((i * salt + 17) % 113) as i32 - 56) as f32 * 0.003,
    }
}
fn same(
    stream: &cuda_core::CudaStream,
    old: &DeviceBuffer<f32>,
    new: &DeviceBuffer<f32>,
    label: &str,
) -> Result<usize, Box<dyn Error>> {
    let old = old.to_host_vec(stream)?;
    let new = new.to_host_vec(stream)?;
    assert_eq!(old.len(), new.len());
    for (i, (a, b)) in old.iter().zip(&new).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label} i={i} old={a:e} new={b:e}"
        );
    }
    Ok(old.len())
}
#[test]
#[ignore = "native full B8/S2048/d2048 fused mHC reverse vector chain and tails"]
fn native_mhc_reverse_matches_original_post_pre_rms_chain() -> Result<(), Box<dyn Error>> {
    let (ctx, stream, ptx) = common::cuda_test_context()?;
    let reference = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles/20261116/mhc-vector-batch.ptx");
    let old = legacy::from_module(ctx.load_module_from_file(reference.to_str().unwrap())?)?;
    let new = MhcReverseModule::from_module(ptx)?;
    let mut compared = 0usize;
    for (n, d) in [
        (16384usize, 2048usize),
        (1, 2048),
        (7, 2048),
        (1, 1),
        (3, 31),
        (7, 255),
        (9, 257),
        (2, 2048),
    ] {
        let wide = n * d * 4;
        let nd = n * d;
        let input = DeviceBuffer::from_host(
            &stream,
            &(0..wide).map(|i| value(i, 19)).collect::<Vec<_>>(),
        )?;
        let incoming = DeviceBuffer::from_host(
            &stream,
            &(0..wide).map(|i| value(i, 23)).collect::<Vec<_>>(),
        )?;
        let branch =
            DeviceBuffer::from_host(&stream, &(0..nd).map(|i| value(i, 29)).collect::<Vec<_>>())?;
        let branch_grad =
            DeviceBuffer::from_host(&stream, &(0..nd).map(|i| value(i, 31)).collect::<Vec<_>>())?;
        let normalized_grad = DeviceBuffer::from_host(
            &stream,
            &(0..wide).map(|i| value(i, 37)).collect::<Vec<_>>(),
        )?;
        let coefficients = DeviceBuffer::from_host(
            &stream,
            &(0..n * 24)
                .map(|i| ((i * 7 % 23) as i32 - 11) as f32 * 0.043)
                .collect::<Vec<_>>(),
        )?;
        let config = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut normalized = DeviceBuffer::zeroed(&stream, wide)?;
        let mut inverse = DeviceBuffer::zeroed(&stream, n)?;
        old.mhc_rms_kernel(
            &stream,
            config,
            &input,
            &mut normalized,
            &mut inverse,
            n as u32,
            (4 * d) as u32,
        )?;
        drop(normalized);
        for head in [false, true] {
            let mut old_df = DeviceBuffer::from_host(&stream, &vec![0.375; nd + GUARD])?;
            let mut new_df = DeviceBuffer::from_host(&stream, &vec![0.375; nd + GUARD])?;
            let mut old_dx = DeviceBuffer::from_host(&stream, &vec![0.375; wide + GUARD])?;
            let mut old_dc = DeviceBuffer::from_host(&stream, &vec![0.375; n * 24 + GUARD])?;
            let mut new_dc = DeviceBuffer::from_host(&stream, &vec![0.375; n * 24 + GUARD])?;
            let mut current_values = (0..wide).map(|i| value(i, 23)).collect::<Vec<_>>();
            current_values.resize(wide + GUARD, 0.375);
            let mut current = DeviceBuffer::from_host(&stream, &current_values)?;
            drop(current_values);
            if !head {
                old.mhc_post_backward_kernel(
                    &stream,
                    config,
                    &input,
                    &branch,
                    &coefficients,
                    &incoming,
                    &mut old_df,
                    &mut old_dx,
                    &mut old_dc,
                    n as u32,
                    d as u32,
                )?;
                new.post(PostArgs {
                    stream: &stream,
                    input: &input,
                    branch: &branch,
                    coefficients: &coefficients,
                    incoming: &current,
                    df: &mut new_df,
                    dc: &mut new_dc,
                    rows: n as u32,
                    dim: d as u32,
                })?;
                compared += same(&stream, &old_df, &new_df, "post branch adjoint and guards")?;
                compared += same(
                    &stream,
                    &old_dc,
                    &new_dc,
                    "post coefficient adjoints and guards",
                )?;
            }
            old.mhc_pre_backward_kernel(
                &stream,
                config,
                &input,
                &coefficients,
                &branch_grad,
                &mut old_dx,
                &mut old_dc,
                n as u32,
                d as u32,
                head,
            )?;
            new.pre(PreArgs {
                stream: &stream,
                input: &input,
                coefficients: &coefficients,
                gradient: &branch_grad,
                dc: &mut new_dc,
                rows: n as u32,
                dim: d as u32,
            })?;
            compared += same(
                &stream,
                &old_dc,
                &new_dc,
                "all coefficient adjoints and guards",
            )?;
            old.mhc_rms_backward_kernel(
                &stream,
                config,
                &input,
                &inverse,
                &normalized_grad,
                &mut old_dx,
                n as u32,
                (4 * d) as u32,
            )?;
            new.finish(FinishArgs {
                stream: &stream,
                input: &input,
                inverse: &inverse,
                coefficients: &coefficients,
                normalized_gradient: &normalized_grad,
                branch_gradient: &branch_grad,
                current: &mut current,
                rows: n as u32,
                dim: d as u32,
                head,
            })?;
            compared += same(
                &stream,
                &old_dx,
                &current,
                "complete reverse output and guards",
            )?;
            let short = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
            assert!(
                new.finish(FinishArgs {
                    stream: &stream,
                    input: &short,
                    inverse: &inverse,
                    coefficients: &coefficients,
                    normalized_gradient: &normalized_grad,
                    branch_gradient: &branch_grad,
                    current: &mut current,
                    rows: n as u32,
                    dim: d as u32,
                    head
                })
                .is_err()
            );
            assert!(
                new.pre(PreArgs {
                    stream: &stream,
                    input: &input,
                    coefficients: &coefficients,
                    gradient: &branch_grad,
                    dc: &mut new_dc,
                    rows: 0,
                    dim: d as u32
                })
                .is_err()
            );
        }
    }
    println!(
        "native_mhc_reverse_verified full_rows=16384 dim=2048 head_and_ordinary=true shapes=8 compared_f32={compared} coefficient_and_branch_adjoints_exact=true ordered_additions_exact=true in_place_guards_exact=true"
    );
    Ok(())
}
