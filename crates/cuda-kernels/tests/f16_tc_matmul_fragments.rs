use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f16_tc_matmul::{
    F16ConvertArgs, F16TcMatmulF32ATransposedRhsArgs, F16TcMatmulF32Args, F16TcMatmulHalfRhsArgs,
    F16TcMatmulModule,
};

mod common;

const M: usize = 64;
const N: usize = 64;
const K: usize = 64;
const TOLERANCE: f32 = 1.0e-6;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nonuniform_fragments_match_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(F16TcMatmulModule::from_module)?;
    let a = matrix(M, K, 3);
    let rhs = matrix(K, N, 11);
    let a_t = transpose(&a, M, K);
    let b_t = transpose(&rhs, K, N);

    let expected = reference(&a, &rhs, |_, _| true);
    let expected_lower = reference(&a, &rhs, |row, k| k <= row);
    let expected_transposed_lower = reference(&a, &rhs, |row, k| k >= row);

    let a_device = DeviceBuffer::from_host(&stream, &a)?;
    let a_t_device = DeviceBuffer::from_host(&stream, &a_t)?;
    let rhs_device = DeviceBuffer::from_host(&stream, &rhs)?;
    let b_t_device = DeviceBuffer::from_host(&stream, &b_t)?;

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, M * N)?;
    module.batched_matmul_f32_input(F16TcMatmulF32Args {
        stream: &stream,
        a: &a_device,
        b_t: &b_t_device,
        out: &mut out,
        batch_count: 1,
        m: M as u32,
        n: N as u32,
        k: K as u32,
    })?;
    common::assert_slice_close(&out.to_host_vec(&stream)?, &expected, TOLERANCE);

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, M * N)?;
    module.batched_matmul_f32_a_transposed_rhs(F16TcMatmulF32ATransposedRhsArgs {
        stream: &stream,
        a: &a_t_device,
        rhs: &rhs_device,
        out: &mut out,
        batch_count: 1,
        m: M as u32,
        n: N as u32,
        k: K as u32,
    })?;
    common::assert_slice_close(&out.to_host_vec(&stream)?, &expected, TOLERANCE);

    let mut a_half = DeviceBuffer::<u16>::zeroed(&stream, M * K)?;
    let mut a_t_half = DeviceBuffer::<u16>::zeroed(&stream, K * M)?;
    let mut rhs_half = DeviceBuffer::<u16>::zeroed(&stream, K * N)?;
    for (src, dst) in [
        (&a_device, &mut a_half),
        (&a_t_device, &mut a_t_half),
        (&rhs_device, &mut rhs_half),
    ] {
        module.fp32_to_f16(F16ConvertArgs {
            stream: &stream,
            src,
            dst,
            element_count: (M * K) as u32,
        })?;
    }

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, M * N)?;
    module.batched_matmul_half_rhs_lower_a(F16TcMatmulHalfRhsArgs {
        stream: &stream,
        a: &a_half,
        rhs: &rhs_half,
        out: &mut out,
        batch_count: 1,
        m: M as u32,
        n: N as u32,
        k: K as u32,
    })?;
    common::assert_slice_close(&out.to_host_vec(&stream)?, &expected_lower, TOLERANCE);

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, M * N)?;
    module.batched_matmul_half_a_transposed_rhs_lower_a(F16TcMatmulHalfRhsArgs {
        stream: &stream,
        a: &a_t_half,
        rhs: &rhs_half,
        out: &mut out,
        batch_count: 1,
        m: M as u32,
        n: N as u32,
        k: K as u32,
    })?;
    common::assert_slice_close(
        &out.to_host_vec(&stream)?,
        &expected_transposed_lower,
        TOLERANCE,
    );
    Ok(())
}

fn matrix(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index - row * cols;
            (((row * 7 + col * 11 + seed) % 15) as i32 - 7) as f32 / 8.0
        })
        .collect()
}

fn transpose(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = input[row * cols + col];
        }
    }
    output
}

fn reference(a: &[f32], rhs: &[f32], include: impl Fn(usize, usize) -> bool) -> Vec<f32> {
    let mut output = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut sum = 0.0;
            for k in 0..K {
                if include(row, k) {
                    sum += a[row * K + k] * rhs[k * N + col];
                }
            }
            output[row * N + col] = sum;
        }
    }
    output
}
