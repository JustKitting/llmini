use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f32_matrix_ops::{
    F32Linear3SqrtBoundAmaxArgs, F32Linear3SqrtBoundArgs, F32Linear3SqrtBoundRowSumsqArgs,
    F32MatrixOpsModule, F32ScaleInPlaceByAmaxArgs,
};
use rust_kernels_cuda::nvfp4_quant::{
    MsEdenQuantArgs, Nvfp4QuantArgs, Nvfp4QuantModule, Nvfp4QuantPaddedArgs,
    Nvfp4QuantPairTransposeExactArgs, Nvfp4QuantRowwiseArgs, Nvfp4QuantRowwiseDerivedAmaxArgs,
    Nvfp4QuantTransposePaddedArgs, RowAmaxArgs, TensorAmaxArgs, nvfp4_tensor_amax_chunks,
};
use rust_kernels_cuda::nvfp4_tma_matmul::scale_layout::pack_sm120_scale_plane_compact;
use rust_kernels_cuda::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

mod common;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_to_nvfp4_four_six_writes_quantized_outputs() -> Result<(), Box<dyn Error>> {
    let x = [
        -3.25f32, -2.0, -1.25, -0.5, -0.125, 0.0, 0.25, 0.75, 1.0, 1.5, 2.25, 3.0, 4.0, 5.0, 6.5,
        8.0,
    ];
    let amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];

    let (_, stream, module) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let mut fp4_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut scales_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut global_scale_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    module.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut fp4_dev,
        out_scales: &mut scales_dev,
        out_global_scale: &mut global_scale_dev,
        group_count: 1,
    })?;

    let fp4 = fp4_dev.to_host_vec(&stream)?;
    let scales = scales_dev.to_host_vec(&stream)?;
    let global_scale = global_scale_dev.to_host_vec(&stream)?;

    common::assert_nvfp4_buffers_nonzero(&fp4, &scales);
    assert!((global_scale[0] - 8.0 / (256.0 * 6.0)).abs() <= 1.0e-8);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fused_row_amax_four_six_matches_two_pass_rowwise_quantization() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 17;
    const COLS: usize = 2048;
    let x = (0..ROWS * COLS)
        .map(|index| {
            let signed = (index % 509) as f32 - 254.0;
            signed * (1.0 + (index / COLS) as f32 * 0.03125) * 0.0078125
        })
        .collect::<Vec<_>>();
    let (_, stream, module) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;

    let mut reference_amax = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    let mut reference_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_global = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    module.row_amax_f32(RowAmaxArgs {
        stream: &stream,
        x: &x_dev,
        out: &mut reference_amax,
        row_count: ROWS as u32,
        row_len: COLS as u32,
    })?;
    module.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
        stream: &stream,
        x: &x_dev,
        amax: &reference_amax,
        out_fp4: &mut reference_fp4,
        out_scales: &mut reference_scales,
        out_global_scale: &mut reference_global,
        group_count: (x.len() / 16) as u32,
        row_len: COLS as u32,
    })?;

    let mut fused_amax = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    let mut fused_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut fused_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut fused_global = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    module.fp32_to_nvfp4_four_six_rowwise_derived_amax(Nvfp4QuantRowwiseDerivedAmaxArgs {
        stream: &stream,
        x: &x_dev,
        amax: &mut fused_amax,
        out_fp4: &mut fused_fp4,
        out_scales: &mut fused_scales,
        out_global_scale: &mut fused_global,
        row_count: ROWS as u32,
        row_len: COLS as u32,
    })?;

    assert_eq!(
        fused_amax.to_host_vec(&stream)?,
        reference_amax.to_host_vec(&stream)?
    );
    assert_eq!(
        fused_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        fused_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        fused_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn bounded_amax_four_six_matches_rescanned_input() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 64;
    const COLS: usize = 64;
    let x = (0..ROWS * COLS)
        .map(|index| ((index % 257) as f32 - 128.0) * 0.137)
        .collect::<Vec<_>>();
    let original_amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];

    let (_, stream, ptx) = common::cuda_test_context()?;
    let quant = Nvfp4QuantModule::from_module(ptx.clone())?;
    let f32_ops = F32MatrixOpsModule::from_module(ptx)?;
    let mut bounded = DeviceBuffer::from_host(&stream, &x)?;
    let original_amax_dev = DeviceBuffer::from_host(&stream, &original_amax)?;
    f32_ops.scale_in_place_by_amax_bound(F32ScaleInPlaceByAmaxArgs {
        stream: &stream,
        x: &mut bounded,
        amax: &original_amax_dev,
        len: x.len() as u32,
    })?;

    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, nvfp4_tensor_amax_chunks(x.len()))?;
    let mut rescanned_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    quant.tensor_amax_f32(TensorAmaxArgs {
        stream: &stream,
        x: &bounded,
        chunk_amax: &mut chunk_amax,
        out: &mut rescanned_amax,
        element_count: x.len() as u32,
    })?;

    let mut reference_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut bounded_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut bounded_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut bounded_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let unbounded = DeviceBuffer::from_host(&stream, &x)?;
    let mut lazy_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut lazy_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut lazy_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    quant.fp32_to_nvfp4_four_six_padded(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &bounded,
        amax: &rescanned_amax,
        out_fp4: &mut reference_fp4,
        out_scales: &mut reference_scales,
        out_global_scale: &mut reference_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_four_six_exact_bounded_amax(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &bounded,
        amax: &original_amax_dev,
        out_fp4: &mut bounded_fp4,
        out_scales: &mut bounded_scales,
        out_global_scale: &mut bounded_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_four_six_exact_lazy_bounded_amax(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &unbounded,
        amax: &original_amax_dev,
        out_fp4: &mut lazy_fp4,
        out_scales: &mut lazy_scales,
        out_global_scale: &mut lazy_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;

    assert_eq!(
        bounded_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        bounded_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        bounded_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn linear3_sqrt_bound_row_sumsq_matches_elementwise_kernel() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 8;
    const COLS: usize = 512;
    let a = (0..ROWS * COLS)
        .map(|index| ((index % 211) as f32 - 105.0) * 0.0037)
        .collect::<Vec<_>>();
    let b = (0..ROWS * COLS)
        .map(|index| ((index % 157) as f32 - 78.0) * 0.0029)
        .collect::<Vec<_>>();
    let c = (0..ROWS * COLS)
        .map(|index| ((index % 97) as f32 - 48.0) * 0.0041)
        .collect::<Vec<_>>();
    let bound = [3.7_f32];

    let (_, stream, ptx) = common::cuda_test_context()?;
    let f32_ops = F32MatrixOpsModule::from_module(ptx)?;
    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let bound_dev = DeviceBuffer::from_host(&stream, &bound)?;
    let mut reference = DeviceBuffer::from_host(&stream, &c)?;
    let mut fused = DeviceBuffer::from_host(&stream, &c)?;
    let mut row_sumsq = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;

    f32_ops.linear3_sqrt_bound_a(F32Linear3SqrtBoundArgs {
        stream: &stream,
        a: &a_dev,
        b: &b_dev,
        c_out: &mut reference,
        bound_amax: &bound_dev,
        len: (ROWS * COLS) as u32,
        a_scale: 2.3,
        b_scale: -1.7,
        c_scale: 0.41,
    })?;
    f32_ops.linear3_sqrt_bound_a_row_sumsq(F32Linear3SqrtBoundRowSumsqArgs {
        stream: &stream,
        a: &a_dev,
        b: &b_dev,
        c_out: &mut fused,
        bound_amax: &bound_dev,
        row_sumsq: &mut row_sumsq,
        rows: ROWS as u32,
        cols: COLS as u32,
        a_scale: 2.3,
        b_scale: -1.7,
        c_scale: 0.41,
    })?;

    let reference = reference.to_host_vec(&stream)?;
    let fused = fused.to_host_vec(&stream)?;
    let row_sumsq = row_sumsq.to_host_vec(&stream)?;
    assert_eq!(fused, reference);
    for row in 0..ROWS {
        let expected = fused[row * COLS..(row + 1) * COLS]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let error = (row_sumsq[row] - expected).abs();
        assert!(
            error <= expected.abs().max(1.0) * 2.0e-6,
            "row {row}: got {}, expected {expected}, error {error}",
            row_sumsq[row]
        );
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn linear3_sqrt_bound_amax_matches_elementwise_kernel() -> Result<(), Box<dyn Error>> {
    const LEN: usize = 4103;
    let a = (0..LEN)
        .map(|index| ((index % 211) as f32 - 105.0) * 0.0037)
        .collect::<Vec<_>>();
    let b = (0..LEN)
        .map(|index| ((index % 157) as f32 - 78.0) * 0.0029)
        .collect::<Vec<_>>();
    let c = (0..LEN)
        .map(|index| ((index % 97) as f32 - 48.0) * 0.0041)
        .collect::<Vec<_>>();
    let bound = [3.7_f32];

    let (_, stream, ptx) = common::cuda_test_context()?;
    let f32_ops = F32MatrixOpsModule::from_module(ptx)?;
    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let bound_dev = DeviceBuffer::from_host(&stream, &bound)?;
    let mut reference = DeviceBuffer::from_host(&stream, &c)?;
    let mut fused = DeviceBuffer::from_host(&stream, &c)?;
    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, nvfp4_tensor_amax_chunks(LEN))?;

    f32_ops.linear3_sqrt_bound_a(F32Linear3SqrtBoundArgs {
        stream: &stream,
        a: &a_dev,
        b: &b_dev,
        c_out: &mut reference,
        bound_amax: &bound_dev,
        len: LEN as u32,
        a_scale: 2.3,
        b_scale: -1.7,
        c_scale: 0.41,
    })?;
    let chunk_count = f32_ops.linear3_sqrt_bound_a_with_amax(F32Linear3SqrtBoundAmaxArgs {
        stream: &stream,
        a: &a_dev,
        b: &b_dev,
        c_out: &mut fused,
        bound_amax: &bound_dev,
        chunk_amax: &mut chunk_amax,
        len: LEN as u32,
        a_scale: 2.3,
        b_scale: -1.7,
        c_scale: 0.41,
    })?;

    let reference = reference.to_host_vec(&stream)?;
    let fused = fused.to_host_vec(&stream)?;
    let chunk_amax = chunk_amax.to_host_vec(&stream)?;
    assert_eq!(fused, reference);
    assert_eq!(chunk_count as usize, chunk_amax.len());
    for (chunk, &got) in chunk_amax.iter().enumerate() {
        let start = chunk * 2048;
        let end = (start + 2048).min(fused.len());
        let expected = fused[start..end]
            .iter()
            .fold(0.0_f32, |amax, value| amax.max(value.abs()));
        assert_eq!(got, expected, "chunk {chunk}");
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tiled_four_six_transpose_matches_explicit_transpose() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 16;
    const COLS: usize = 64;
    let x = (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            ((row * 17 + col * 13) as f32 - 512.0) * 0.03125
        })
        .collect::<Vec<_>>();
    let mut x_t = vec![0.0f32; x.len()];
    for row in 0..ROWS {
        for col in 0..COLS {
            x_t[col * ROWS + row] = x[row * COLS + col];
        }
    }
    let amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];

    let (_, stream, module) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let x_t_dev = DeviceBuffer::from_host(&stream, &x_t)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let mut tiled_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut tiled_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut tiled_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut reference_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    module.fp32_transpose_to_nvfp4_four_six_padded(Nvfp4QuantTransposePaddedArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut tiled_fp4,
        out_scales: &mut tiled_scales,
        out_global_scale: &mut tiled_global,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
        padded_rows: COLS as u32,
        padded_cols: ROWS as u32,
    })?;
    module.fp32_to_nvfp4_four_six_padded(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &x_t_dev,
        amax: &amax_dev,
        out_fp4: &mut reference_fp4,
        out_scales: &mut reference_scales,
        out_global_scale: &mut reference_global,
        rows: COLS as u32,
        cols: ROWS as u32,
        padded_rows: COLS as u32,
        padded_cols: ROWS as u32,
    })?;

    assert_eq!(
        tiled_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        tiled_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        tiled_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn direct_tma_scale_layout_quantizers_match_logical_scale_pack() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 128;
    const COLS: usize = 256;
    let x = (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            ((row * 37 + col * 19) as f32 - 2048.0) * 0.0078125
        })
        .collect::<Vec<_>>();
    let amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];
    let (_, stream, module) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax)?;

    let mut row_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut row_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut row_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut transpose_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut transpose_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut transpose_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_pair_to_nvfp4_four_six_exact_pow2_tiled(Nvfp4QuantPairTransposeExactArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut row_fp4,
        out_scales: &mut row_scales,
        out_global_scale: &mut row_global,
        transpose_out_fp4: &mut transpose_fp4,
        transpose_out_scales: &mut transpose_scales,
        transpose_out_global_scale: &mut transpose_global,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
    })?;

    let row_scale_reference =
        pack_sm120_scale_plane_compact(&row_scales.to_host_vec(&stream)?, ROWS, COLS);
    let transpose_scale_reference =
        pack_sm120_scale_plane_compact(&transpose_scales.to_host_vec(&stream)?, COLS, ROWS);
    let mut packed_row_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut packed_row_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut packed_row_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut packed_transpose_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut packed_transpose_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut packed_transpose_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_pair_to_nvfp4_four_six_exact_pow2_tiled_packed_scales(
        Nvfp4QuantPairTransposeExactArgs {
            stream: &stream,
            x: &x_dev,
            amax: &amax_dev,
            out_fp4: &mut packed_row_fp4,
            out_scales: &mut packed_row_scales,
            out_global_scale: &mut packed_row_global,
            transpose_out_fp4: &mut packed_transpose_fp4,
            transpose_out_scales: &mut packed_transpose_scales,
            transpose_out_global_scale: &mut packed_transpose_global,
            source_rows: ROWS as u32,
            source_cols: COLS as u32,
        },
    )?;
    assert_eq!(
        packed_row_fp4.to_host_vec(&stream)?,
        row_fp4.to_host_vec(&stream)?
    );
    assert_eq!(packed_row_scales.to_host_vec(&stream)?, row_scale_reference);
    assert_eq!(
        packed_row_global.to_host_vec(&stream)?,
        row_global.to_host_vec(&stream)?
    );
    assert_eq!(
        packed_transpose_fp4.to_host_vec(&stream)?,
        transpose_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        packed_transpose_scales.to_host_vec(&stream)?,
        transpose_scale_reference
    );
    assert_eq!(
        packed_transpose_global.to_host_vec(&stream)?,
        transpose_global.to_host_vec(&stream)?
    );

    let mut bounded_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut bounded_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut bounded_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_to_nvfp4_four_six_exact_lazy_bounded_amax(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut bounded_fp4,
        out_scales: &mut bounded_scales,
        out_global_scale: &mut bounded_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;
    let bounded_scale_reference =
        pack_sm120_scale_plane_compact(&bounded_scales.to_host_vec(&stream)?, ROWS, COLS);
    let mut packed_bounded_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut packed_bounded_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut packed_bounded_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_to_nvfp4_four_six_exact_lazy_bounded_amax_packed_scales(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut packed_bounded_fp4,
        out_scales: &mut packed_bounded_scales,
        out_global_scale: &mut packed_bounded_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;
    assert_eq!(
        packed_bounded_fp4.to_host_vec(&stream)?,
        bounded_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        packed_bounded_scales.to_host_vec(&stream)?,
        bounded_scale_reference
    );
    assert_eq!(
        packed_bounded_global.to_host_vec(&stream)?,
        bounded_global.to_host_vec(&stream)?
    );

    let mut transposed_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut transposed_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut transposed_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_transpose_to_nvfp4_four_six_exact_packed_scales(Nvfp4QuantTransposePaddedArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut transposed_fp4,
        out_scales: &mut transposed_scales,
        out_global_scale: &mut transposed_global,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
        padded_rows: COLS as u32,
        padded_cols: ROWS as u32,
    })?;
    assert_eq!(
        transposed_fp4.to_host_vec(&stream)?,
        transpose_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        transposed_scales.to_host_vec(&stream)?,
        transpose_scale_reference
    );
    assert_eq!(
        transposed_global.to_host_vec(&stream)?,
        transpose_global.to_host_vec(&stream)?
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn paired_four_six_exact_matches_independent_layouts_and_lazy_bound() -> Result<(), Box<dyn Error>>
{
    const ROWS: usize = 128;
    const COLS: usize = 128;
    let x = (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            ((row * 37 + col * 19) as f32 - 2048.0) * 0.0078125
        })
        .collect::<Vec<_>>();
    let amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];
    let sqrt_bound = [3.7_f32];

    let (_, stream, module) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let sqrt_bound_dev = DeviceBuffer::from_host(&stream, &sqrt_bound)?;

    let mut row_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut row_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut row_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut transpose_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut transpose_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut transpose_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_pair_to_nvfp4_four_six_exact_pow2_tiled(Nvfp4QuantPairTransposeExactArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut row_fp4,
        out_scales: &mut row_scales,
        out_global_scale: &mut row_global,
        transpose_out_fp4: &mut transpose_fp4,
        transpose_out_scales: &mut transpose_scales,
        transpose_out_global_scale: &mut transpose_global,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
    })?;

    let mut reference_row_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_row_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_row_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_to_nvfp4_four_six_padded(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut reference_row_fp4,
        out_scales: &mut reference_row_scales,
        out_global_scale: &mut reference_row_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;
    let mut reference_transpose_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_transpose_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_transpose_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_transpose_to_nvfp4_four_six_padded(Nvfp4QuantTransposePaddedArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut reference_transpose_fp4,
        out_scales: &mut reference_transpose_scales,
        out_global_scale: &mut reference_transpose_global,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
        padded_rows: COLS as u32,
        padded_cols: ROWS as u32,
    })?;

    assert_eq!(
        row_fp4.to_host_vec(&stream)?,
        reference_row_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        row_scales.to_host_vec(&stream)?,
        reference_row_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        row_global.to_host_vec(&stream)?,
        reference_row_global.to_host_vec(&stream)?
    );
    assert_eq!(
        transpose_fp4.to_host_vec(&stream)?,
        reference_transpose_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        transpose_scales.to_host_vec(&stream)?,
        reference_transpose_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        transpose_global.to_host_vec(&stream)?,
        reference_transpose_global.to_host_vec(&stream)?
    );

    module.rebase_four_six_global_scale_sqrt_bound(
        &stream,
        &amax_dev,
        &sqrt_bound_dev,
        &mut transpose_global,
    )?;
    let mut lazy_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut lazy_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut lazy_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_transpose_to_nvfp4_four_six_exact_lazy_sqrt_bounded_amax(
        Nvfp4QuantTransposePaddedArgs {
            stream: &stream,
            x: &x_dev,
            amax: &amax_dev,
            out_fp4: &mut lazy_fp4,
            out_scales: &mut lazy_scales,
            out_global_scale: &mut lazy_global,
            source_rows: ROWS as u32,
            source_cols: COLS as u32,
            padded_rows: COLS as u32,
            padded_cols: ROWS as u32,
        },
        &sqrt_bound_dev,
    )?;
    let reused_fp4 = transpose_fp4.to_host_vec(&stream)?;
    let lazy_fp4 = lazy_fp4.to_host_vec(&stream)?;
    let fp4_mismatches = reused_fp4
        .iter()
        .zip(&lazy_fp4)
        .filter(|(reused, lazy)| reused != lazy)
        .count();
    let reused_scales = transpose_scales.to_host_vec(&stream)?;
    let lazy_scales = lazy_scales.to_host_vec(&stream)?;
    let scale_mismatches = reused_scales
        .iter()
        .zip(&lazy_scales)
        .filter(|(reused, lazy)| reused != lazy)
        .count();
    assert!(
        fp4_mismatches * 100 <= reused_fp4.len(),
        "too many lazy-bound payload differences: {fp4_mismatches}/{}",
        reused_fp4.len()
    );
    assert!(
        scale_mismatches * 100 <= reused_scales.len(),
        "too many lazy-bound scale differences: {scale_mismatches}/{}",
        reused_scales.len()
    );
    assert_eq!(
        transpose_global.to_host_vec(&stream)?,
        lazy_global.to_host_vec(&stream)?
    );

    let mut reference_bound_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_bound_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_bound_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut reference_rebased_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_to_nvfp4_four_six_exact_lazy_bounded_amax_packed_scales(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &x_dev,
        amax: &sqrt_bound_dev,
        out_fp4: &mut reference_bound_fp4,
        out_scales: &mut reference_bound_scales,
        out_global_scale: &mut reference_bound_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;
    module.rebase_four_six_global_scale_sqrt_bound(
        &stream,
        &amax_dev,
        &sqrt_bound_dev,
        &mut reference_rebased_global,
    )?;

    let mut fused_bound_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut fused_bound_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut fused_bound_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut fused_rebased_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    module.fp32_to_nvfp4_four_six_exact_lazy_bounded_amax_packed_scales_rebase(
        Nvfp4QuantPaddedArgs {
            stream: &stream,
            x: &x_dev,
            amax: &sqrt_bound_dev,
            out_fp4: &mut fused_bound_fp4,
            out_scales: &mut fused_bound_scales,
            out_global_scale: &mut fused_bound_global,
            rows: ROWS as u32,
            cols: COLS as u32,
            padded_rows: ROWS as u32,
            padded_cols: COLS as u32,
        },
        &amax_dev,
        &mut fused_rebased_global,
    )?;
    assert_eq!(
        reference_bound_fp4.to_host_vec(&stream)?,
        fused_bound_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        reference_bound_scales.to_host_vec(&stream)?,
        fused_bound_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        reference_bound_global.to_host_vec(&stream)?,
        fused_bound_global.to_host_vec(&stream)?
    );
    assert_eq!(
        reference_rebased_global.to_host_vec(&stream)?,
        fused_rebased_global.to_host_vec(&stream)?
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn sqrt_bounded_amax_transpose_matches_rescanned_input() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 16;
    const COLS: usize = 64;
    let x = (0..ROWS * COLS)
        .map(|index| ((index % 173) as f32 - 86.0) * 0.119)
        .collect::<Vec<_>>();
    let original_amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];
    let sqrt_bound_amax = [3.7_f32];

    let (_, stream, ptx) = common::cuda_test_context()?;
    let quant = Nvfp4QuantModule::from_module(ptx.clone())?;
    let f32_ops = F32MatrixOpsModule::from_module(ptx)?;
    let mut bounded = DeviceBuffer::from_host(&stream, &x)?;
    let original_amax_dev = DeviceBuffer::from_host(&stream, &original_amax)?;
    let sqrt_bound_amax_dev = DeviceBuffer::from_host(&stream, &sqrt_bound_amax)?;
    f32_ops.scale_in_place_by_sqrt_amax_bound(F32ScaleInPlaceByAmaxArgs {
        stream: &stream,
        x: &mut bounded,
        amax: &sqrt_bound_amax_dev,
        len: x.len() as u32,
    })?;

    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, nvfp4_tensor_amax_chunks(x.len()))?;
    let mut rescanned_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    quant.tensor_amax_f32(TensorAmaxArgs {
        stream: &stream,
        x: &bounded,
        chunk_amax: &mut chunk_amax,
        out: &mut rescanned_amax,
        element_count: x.len() as u32,
    })?;

    let mut reference_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut bounded_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut bounded_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut bounded_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let unbounded = DeviceBuffer::from_host(&stream, &x)?;
    let mut lazy_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut lazy_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut lazy_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    quant.fp32_transpose_to_nvfp4_four_six_padded(Nvfp4QuantTransposePaddedArgs {
        stream: &stream,
        x: &bounded,
        amax: &rescanned_amax,
        out_fp4: &mut reference_fp4,
        out_scales: &mut reference_scales,
        out_global_scale: &mut reference_global,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
        padded_rows: COLS as u32,
        padded_cols: ROWS as u32,
    })?;
    quant.fp32_transpose_to_nvfp4_four_six_exact_sqrt_bounded_amax(
        Nvfp4QuantTransposePaddedArgs {
            stream: &stream,
            x: &bounded,
            amax: &original_amax_dev,
            out_fp4: &mut bounded_fp4,
            out_scales: &mut bounded_scales,
            out_global_scale: &mut bounded_global,
            source_rows: ROWS as u32,
            source_cols: COLS as u32,
            padded_rows: COLS as u32,
            padded_cols: ROWS as u32,
        },
        &sqrt_bound_amax_dev,
    )?;
    quant.fp32_transpose_to_nvfp4_four_six_exact_lazy_sqrt_bounded_amax(
        Nvfp4QuantTransposePaddedArgs {
            stream: &stream,
            x: &unbounded,
            amax: &original_amax_dev,
            out_fp4: &mut lazy_fp4,
            out_scales: &mut lazy_scales,
            out_global_scale: &mut lazy_global,
            source_rows: ROWS as u32,
            source_cols: COLS as u32,
            padded_rows: COLS as u32,
            padded_cols: ROWS as u32,
        },
        &sqrt_bound_amax_dev,
    )?;

    assert_eq!(
        bounded_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        bounded_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        bounded_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_to_nvfp4_ms_eden_writes_rotated_quantized_outputs() -> Result<(), Box<dyn Error>> {
    let x = (0..64)
        .map(|index| (index as f32 - 31.5) * 0.03125)
        .collect::<Vec<_>>();

    let (_, stream, module) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut fp4_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut scales_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut global_scales_dev = DeviceBuffer::<f32>::zeroed(&stream, 2)?;
    let mut chunk_amax_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len() / 32)?;

    module.fp32_to_nvfp4_ms_eden(MsEdenQuantArgs {
        stream: &stream,
        x: &x_dev,
        out_fp4: &mut fp4_dev,
        out_scales: &mut scales_dev,
        out_global_scales: &mut global_scales_dev,
        out_chunk_amax: &mut chunk_amax_dev,
        row_count: 2,
        src_row_len: 32,
        dst_row_len: 32,
        global_scale: 1.0,
        scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
        sign_seed: 0x1234_5678,
        scale_seed: 0x9abc_def0,
    })?;

    let fp4 = fp4_dev.to_host_vec(&stream)?;
    let scales = scales_dev.to_host_vec(&stream)?;
    let global_scales = global_scales_dev.to_host_vec(&stream)?;
    let chunk_amax = chunk_amax_dev.to_host_vec(&stream)?;

    common::assert_nvfp4_buffers_nonzero(&fp4, &scales);
    assert!(
        global_scales
            .iter()
            .all(|scale| (*scale - 1.0).abs() <= 1.0e-8)
    );
    assert!(
        chunk_amax
            .iter()
            .all(|amax| *amax > 0.0 && amax.is_finite())
    );
    Ok(())
}
