use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f16_tc_matmul::{
    F16ConvertArgs, F16TcMatmulF32ATransposedRhsArgs, F16TcMatmulF32Args, F16TcMatmulF32WindowArgs,
    F16TcMatmulHalfDsSparseWindowArgs, F16TcMatmulHalfDsWindowArgs, F16TcMatmulHalfRhsArgs,
    F16TcMatmulHalfRhsSparseWindowArgs, F16TcMatmulHalfRhsWindowArgs, F16TcMatmulModule,
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

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn windowed_attention_tiles_match_materialized_reference() -> Result<(), Box<dyn Error>> {
    const SEQ: usize = 256;
    const WIDTH: usize = 64;
    const WINDOW: usize = 128;
    const WINDOW_TOLERANCE: f32 = 1.0e-4;

    let (_, stream, module) = common::cuda_test_module(F16TcMatmulModule::from_module)?;

    let q = matrix_shape(SEQ, WIDTH, 17);
    let k = matrix_shape(SEQ, WIDTH, 23);
    let expected_scores = score_reference(&q, &k, SEQ, WIDTH, WINDOW);
    let q_device = DeviceBuffer::from_host(&stream, &q)?;
    let k_device = DeviceBuffer::from_host(&stream, &k)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(&stream, SEQ * SEQ)?;
    module.batched_matmul_f32_input_windowed_lower(F16TcMatmulF32WindowArgs {
        stream: &stream,
        a: &q_device,
        b_t: &k_device,
        out: &mut scores,
        batch_count: 1,
        m: SEQ as u32,
        n: SEQ as u32,
        k: WIDTH as u32,
        window: WINDOW as u32,
    })?;
    common::assert_slice_close(
        &scores.to_host_vec(&stream)?,
        &expected_scores,
        WINDOW_TOLERANCE,
    );

    let a = matrix_shape(SEQ, SEQ, 29);
    let rhs = matrix_shape(SEQ, WIDTH, 31);
    let a_t = transpose_shape(&a, SEQ, SEQ);
    let expected = reference_shape(&a, &rhs, SEQ, WIDTH, SEQ, |row, k| {
        k <= row && row - k < WINDOW
    });
    let expected_transposed = reference_shape(&a, &rhs, SEQ, WIDTH, SEQ, |row, k| {
        k >= row && k - row < WINDOW
    });
    let a_device = DeviceBuffer::from_host(&stream, &a)?;
    let a_t_device = DeviceBuffer::from_host(&stream, &a_t)?;
    let rhs_device = DeviceBuffer::from_host(&stream, &rhs)?;
    let mut a_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * SEQ)?;
    let mut a_t_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * SEQ)?;
    let mut rhs_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * WIDTH)?;
    for (src, dst, len) in [
        (&a_device, &mut a_half, SEQ * SEQ),
        (&a_t_device, &mut a_t_half, SEQ * SEQ),
        (&rhs_device, &mut rhs_half, SEQ * WIDTH),
    ] {
        module.fp32_to_f16(F16ConvertArgs {
            stream: &stream,
            src,
            dst,
            element_count: len as u32,
        })?;
    }

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, SEQ * WIDTH)?;
    module.batched_matmul_half_rhs_windowed_lower_a(F16TcMatmulHalfRhsWindowArgs {
        stream: &stream,
        a: &a_half,
        rhs: &rhs_half,
        out: &mut out,
        batch_count: 1,
        m: SEQ as u32,
        n: WIDTH as u32,
        k: SEQ as u32,
        window: WINDOW as u32,
    })?;
    common::assert_slice_close(&out.to_host_vec(&stream)?, &expected, WINDOW_TOLERANCE);

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, SEQ * WIDTH)?;
    module.batched_matmul_half_a_transposed_rhs_windowed_lower_a(F16TcMatmulHalfRhsWindowArgs {
        stream: &stream,
        a: &a_t_half,
        rhs: &rhs_half,
        out: &mut out,
        batch_count: 1,
        m: SEQ as u32,
        n: WIDTH as u32,
        k: SEQ as u32,
        window: WINDOW as u32,
    })?;
    common::assert_slice_close(
        &out.to_host_vec(&stream)?,
        &expected_transposed,
        WINDOW_TOLERANCE,
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn sparse_attention_backward_tiles_match_materialized_reference() -> Result<(), Box<dyn Error>> {
    const SEQ: usize = 256;
    const WIDTH: usize = 64;
    const WINDOW: usize = 128;
    const TILES: usize = SEQ / 64;
    const SPARSE_TOLERANCE: f32 = 1.0e-4;

    let (_, stream, module) = common::cuda_test_module(F16TcMatmulModule::from_module)?;
    let a = matrix_shape(SEQ, SEQ, 37);
    let rhs = matrix_shape(SEQ, WIDTH, 41);
    let mut tile_scales = vec![0.0_f32; TILES * TILES];
    for query_tile in 0..TILES {
        for key_tile in 0..TILES {
            let visible = key_tile <= query_tile && query_tile <= key_tile + WINDOW / 64;
            if visible && (query_tile + 2 * key_tile) % 3 != 0 {
                tile_scales[query_tile * TILES + key_tile] = 2.0;
            }
        }
    }
    let mut a_scaled = vec![0.0_f32; a.len()];
    for query in 0..SEQ {
        for key in 0..SEQ {
            let scale = tile_scales[(query / 64) * TILES + key / 64];
            a_scaled[query * SEQ + key] = a[query * SEQ + key] * scale;
        }
    }
    let expected = reference_shape(&a_scaled, &rhs, SEQ, WIDTH, SEQ, |query, key| {
        key <= query && query - key < WINDOW
    });
    let expected_transposed = reference_transposed_attention(&a_scaled, &rhs, SEQ, WIDTH, WINDOW);

    let a_device = DeviceBuffer::from_host(&stream, &a)?;
    let a_scaled_device = DeviceBuffer::from_host(&stream, &a_scaled)?;
    let rhs_device = DeviceBuffer::from_host(&stream, &rhs)?;
    let tile_scales_device = DeviceBuffer::from_host(&stream, &tile_scales)?;
    let mut a_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * SEQ)?;
    let mut a_scaled_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * SEQ)?;
    let mut rhs_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * WIDTH)?;
    for (src, dst, len) in [
        (&a_device, &mut a_half, SEQ * SEQ),
        (&a_scaled_device, &mut a_scaled_half, SEQ * SEQ),
        (&rhs_device, &mut rhs_half, SEQ * WIDTH),
    ] {
        module.fp32_to_f16(F16ConvertArgs {
            stream: &stream,
            src,
            dst,
            element_count: len as u32,
        })?;
    }

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, SEQ * WIDTH)?;
    module.batched_matmul_half_rhs_windowed_lower_a_sparse(F16TcMatmulHalfRhsSparseWindowArgs {
        stream: &stream,
        a: &a_scaled_half,
        rhs: &rhs_half,
        tile_scales: &tile_scales_device,
        out: &mut out,
        batch_count: 1,
        m: SEQ as u32,
        n: WIDTH as u32,
        k: SEQ as u32,
        window: WINDOW as u32,
    })?;
    common::assert_slice_close(&out.to_host_vec(&stream)?, &expected, SPARSE_TOLERANCE);

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, SEQ * WIDTH)?;
    module.batched_matmul_half_a_transposed_rhs_windowed_lower_a_sparse(
        F16TcMatmulHalfRhsSparseWindowArgs {
            stream: &stream,
            a: &a_scaled_half,
            rhs: &rhs_half,
            tile_scales: &tile_scales_device,
            out: &mut out,
            batch_count: 1,
            m: SEQ as u32,
            n: WIDTH as u32,
            k: SEQ as u32,
            window: WINDOW as u32,
        },
    )?;
    common::assert_slice_close(
        &out.to_host_vec(&stream)?,
        &expected_transposed,
        SPARSE_TOLERANCE,
    );

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, SEQ * WIDTH)?;
    module.batched_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_scaled(
        F16TcMatmulHalfRhsSparseWindowArgs {
            stream: &stream,
            a: &a_half,
            rhs: &rhs_half,
            tile_scales: &tile_scales_device,
            out: &mut out,
            batch_count: 1,
            m: SEQ as u32,
            n: WIDTH as u32,
            k: SEQ as u32,
            window: WINDOW as u32,
        },
    )?;
    common::assert_slice_close(
        &out.to_host_vec(&stream)?,
        &expected_transposed,
        SPARSE_TOLERANCE,
    );

    let d_out = matrix_shape(SEQ, WIDTH, 43);
    let values = matrix_shape(SEQ, WIDTH, 47);
    let probs = vec![0.25_f32; SEQ * SEQ];
    let mut scaled_probs = vec![0.0_f32; probs.len()];
    for query in 0..SEQ {
        for key in 0..SEQ {
            scaled_probs[query * SEQ + key] =
                probs[query * SEQ + key] * tile_scales[(query / 64) * TILES + key / 64];
        }
    }
    let d_out_device = DeviceBuffer::from_host(&stream, &d_out)?;
    let values_device = DeviceBuffer::from_host(&stream, &values)?;
    let probs_device = DeviceBuffer::from_host(&stream, &probs)?;
    let scaled_probs_device = DeviceBuffer::from_host(&stream, &scaled_probs)?;
    let mut d_out_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * WIDTH)?;
    let mut values_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * WIDTH)?;
    let mut probs_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * SEQ)?;
    let mut scaled_probs_half = DeviceBuffer::<u16>::zeroed(&stream, SEQ * SEQ)?;
    for (src, dst, len) in [
        (&d_out_device, &mut d_out_half, SEQ * WIDTH),
        (&values_device, &mut values_half, SEQ * WIDTH),
        (&probs_device, &mut probs_half, SEQ * SEQ),
        (&scaled_probs_device, &mut scaled_probs_half, SEQ * SEQ),
    ] {
        module.fp32_to_f16(F16ConvertArgs {
            stream: &stream,
            src,
            dst,
            element_count: len as u32,
        })?;
    }
    let softmax_d = DeviceBuffer::from_host(&stream, &vec![0.0_f32; SEQ])?;
    let mut dense_ds = DeviceBuffer::<u16>::zeroed(&stream, SEQ * SEQ)?;
    module.batched_matmul_half_input_windowed_lower_ds(F16TcMatmulHalfDsWindowArgs {
        stream: &stream,
        a: &d_out_half,
        b_t: &values_half,
        probs: &scaled_probs_half,
        softmax_d: &softmax_d,
        out: &mut dense_ds,
        batch_count: 1,
        m: SEQ as u32,
        n: SEQ as u32,
        k: WIDTH as u32,
        window: WINDOW as u32,
    })?;
    let mut sparse_ds = DeviceBuffer::<u16>::zeroed(&stream, SEQ * SEQ)?;
    module.batched_matmul_half_input_windowed_lower_ds_sparse(
        F16TcMatmulHalfDsSparseWindowArgs {
            stream: &stream,
            a: &d_out_half,
            b_t: &values_half,
            probs: &probs_half,
            softmax_d: &softmax_d,
            tile_scales: &tile_scales_device,
            out: &mut sparse_ds,
            batch_count: 1,
            m: SEQ as u32,
            n: SEQ as u32,
            k: WIDTH as u32,
            window: WINDOW as u32,
        },
    )?;
    let sparse_ds = sparse_ds.to_host_vec(&stream)?;
    let dense_ds = dense_ds.to_host_vec(&stream)?;
    for (index, (&sparse, &dense)) in sparse_ds.iter().zip(&dense_ds).enumerate() {
        let both_zero = sparse & 0x7fff == 0 && dense & 0x7fff == 0;
        assert!(
            sparse == dense || both_zero,
            "dS mismatch at {index}: sparse={sparse:#06x}, dense={dense:#06x}"
        );
    }
    Ok(())
}

fn matrix(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
    matrix_shape(rows, cols, seed)
}

fn matrix_shape(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index - row * cols;
            (((row * 7 + col * 11 + seed) % 15) as i32 - 7) as f32 / 8.0
        })
        .collect()
}

fn transpose(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    transpose_shape(input, rows, cols)
}

fn transpose_shape(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = input[row * cols + col];
        }
    }
    output
}

fn reference(a: &[f32], rhs: &[f32], include: impl Fn(usize, usize) -> bool) -> Vec<f32> {
    reference_shape(a, rhs, M, N, K, include)
}

fn reference_shape(
    a: &[f32],
    rhs: &[f32],
    m: usize,
    n: usize,
    k_len: usize,
    include: impl Fn(usize, usize) -> bool,
) -> Vec<f32> {
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for k in 0..k_len {
                if include(row, k) {
                    sum += a[row * k_len + k] * rhs[k * n + col];
                }
            }
            output[row * n + col] = sum;
        }
    }
    output
}

fn reference_transposed_attention(
    a: &[f32],
    rhs: &[f32],
    seq: usize,
    width: usize,
    window: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; seq * width];
    for key in 0..seq {
        for col in 0..width {
            let mut sum = 0.0;
            for query in key..seq.min(key + window) {
                sum += a[query * seq + key] * rhs[query * width + col];
            }
            out[key * width + col] = sum;
        }
    }
    out
}

fn score_reference(q: &[f32], k: &[f32], seq: usize, width: usize, window: usize) -> Vec<f32> {
    let mut output = vec![0.0; seq * seq];
    let window_tiles = window / 64;
    for row in 0..seq {
        for col in 0..seq {
            let tile_row = row / 64;
            let tile_col = col / 64;
            if tile_col > tile_row || tile_row > tile_col + window_tiles {
                continue;
            }
            output[row * seq + col] = (0..width)
                .map(|i| q[row * width + i] * k[col * width + i])
                .sum();
        }
    }
    output
}
