use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::attention::{AttentionModule, CProjArgs, QkvProjectionArgs};
use rust_kernels_cuda::f16_tc_matmul::{F16ConvertArgs, F16TcMatmulModule};
use rust_kernels_cuda::f32_matrix_ops::{F32Linear3SqrtBoundAmaxArgs, F32MatrixOpsModule};
use rust_kernels_cuda::lm_head::{LmHeadArgs, LmHeadModule};
use rust_kernels_cuda::mlp::{
    MlpDownResidualArgs, MlpModule, MlpUpRelu2Args, Relu2BackwardF16Args,
    relu2_backward_amax_chunks,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_tma_matmul::{
    kernels::{tma_nvfp4_output_amax_chunks, tma_nvfp4_symmetric_output_amax_chunks},
    launcher::Nvfp4GemmModule,
    pad::{TmaMatrixPadModule, U4RowPadArgs},
    scale_layout::{sm120_scale_packed_len, sm120_scale_padded_mn_extent},
    scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

mod common;

use common::nvfp4::{one_scales, set_e2m1_one};

const ROWS: usize = 128;
const K: usize = 128;
const N: usize = 160;
const TOLERANCE: f32 = 1.0e-5;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_raw_padded_output_matches_old_lm_head_projection() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, N)?;
    let mut old_out = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * N)?;
    let mut tma_out = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * N)?;

    fixture.old_raw(&mut old_out)?;
    fixture.tma_raw(&mut tma_out)?;

    common::assert_slice_close(
        &tma_out.to_host_vec(&fixture.stream)?,
        &old_out.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_output_amax_matches_stored_output() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, 128)?;
    let mut plain = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut fused = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let chunk_count = tma_nvfp4_output_amax_chunks(ROWS as u32, 128);
    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&fixture.stream, chunk_count as usize)?;

    let launched_chunks =
        fixture.tma_scalar_with_output_amax(&mut plain, &mut fused, &mut chunk_amax)?;
    assert_eq!(launched_chunks, chunk_count);

    let plain = plain.to_host_vec(&fixture.stream)?;
    let fused = fused.to_host_vec(&fixture.stream)?;
    assert_eq!(fused, plain, "amax epilogue changed the stored GEMM output");

    let output_amax = fused.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let fused_amax = chunk_amax
        .to_host_vec(&fixture.stream)?
        .into_iter()
        .fold(0.0_f32, f32::max);
    assert_eq!(fused_amax, output_amax);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_relu2_backward_f16_amax_matches_standalone_epilogue() -> Result<(), Box<dyn Error>> {
    const DIM: usize = 128;
    const LEN: usize = ROWS * DIM;
    let fixture = Fixture::new(ROWS, K, DIM)?;
    let pre_f32 = DeviceBuffer::from_host(&fixture.stream, &pre_activation_values(LEN))?;
    let mut pre_f16 = DeviceBuffer::<u16>::zeroed(&fixture.stream, LEN)?;
    let mut plain = DeviceBuffer::<f32>::zeroed(&fixture.stream, LEN)?;
    let mut standalone = DeviceBuffer::<f32>::zeroed(&fixture.stream, LEN)?;
    let mut fused = DeviceBuffer::<f32>::zeroed(&fixture.stream, LEN)?;
    let standalone_chunk_count = relu2_backward_amax_chunks(LEN as u32);
    let fused_chunk_count = tma_nvfp4_output_amax_chunks(ROWS as u32, DIM as u32);
    assert_eq!(standalone_chunk_count, fused_chunk_count);
    let mut standalone_chunks =
        DeviceBuffer::<f32>::zeroed(&fixture.stream, standalone_chunk_count as usize)?;
    let mut fused_chunks =
        DeviceBuffer::<f32>::zeroed(&fixture.stream, fused_chunk_count as usize)?;

    fixture.f16.fp32_to_f16(F16ConvertArgs {
        stream: &fixture.stream,
        src: &pre_f32,
        dst: &mut pre_f16,
        element_count: LEN as u32,
    })?;
    let launched_chunks = fixture.tma_relu2_backward_f16_with_output_amax(
        &pre_f16,
        &mut plain,
        &mut fused,
        &mut fused_chunks,
    )?;
    assert_eq!(launched_chunks, fused_chunk_count);
    fixture.mlp.relu2_backward_f16(Relu2BackwardF16Args {
        stream: &fixture.stream,
        pre_activation: &pre_f16,
        d_out: &plain,
        d_pre_activation: &mut standalone,
        d_pre_activation_chunk_amax: &mut standalone_chunks,
        len: LEN as u32,
    })?;

    let standalone = standalone.to_host_vec(&fixture.stream)?;
    let fused = fused.to_host_vec(&fixture.stream)?;
    assert_eq!(
        fused, standalone,
        "fused ReLU2 backward epilogue changed the stored gradient"
    );
    let output_amax = fused.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let fused_amax = fused_chunks
        .to_host_vec(&fixture.stream)?
        .into_iter()
        .fold(0.0_f32, f32::max);
    assert_eq!(fused_amax, output_amax);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_symmetric_output_and_amax_match_full_self_product() -> Result<(), Box<dyn Error>> {
    const DIM: usize = 256;
    let fixture = Fixture::new(DIM, K, DIM)?;
    let mut plain = DeviceBuffer::<f32>::zeroed(&fixture.stream, DIM * DIM)?;
    let mut symmetric = DeviceBuffer::<f32>::zeroed(&fixture.stream, DIM * DIM)?;
    let chunk_count = tma_nvfp4_symmetric_output_amax_chunks(DIM as u32);
    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&fixture.stream, chunk_count as usize)?;

    let launched_chunks =
        fixture.tma_self_symmetric_with_output_amax(&mut plain, &mut symmetric, &mut chunk_amax)?;
    assert_eq!(launched_chunks, chunk_count);

    let plain = plain.to_host_vec(&fixture.stream)?;
    let symmetric = symmetric.to_host_vec(&fixture.stream)?;
    assert_eq!(
        symmetric, plain,
        "triangular self-product changed the stored Gram matrix"
    );

    let output_amax = symmetric
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    let fused_amax = chunk_amax
        .to_host_vec(&fixture.stream)?
        .into_iter()
        .fold(0.0_f32, f32::max);
    assert_eq!(fused_amax, output_amax);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_linear3_amax_epilogue_matches_standalone_update() -> Result<(), Box<dyn Error>> {
    const DIM: usize = 128;
    const LEN: usize = ROWS * DIM;
    const A_SCALE: f32 = 3.4445;
    const B_SCALE: f32 = -4.775;
    const C_SCALE: f32 = 2.0315;
    let fixture = Fixture::new(ROWS, K, DIM)?;
    let source = DeviceBuffer::from_host(&fixture.stream, &residual_values(LEN))?;
    let action = DeviceBuffer::from_host(&fixture.stream, &action_values(LEN))?;
    let bound_amax = DeviceBuffer::from_host(&fixture.stream, &[1.5625_f32])?;
    let mut standalone = DeviceBuffer::<f32>::zeroed(&fixture.stream, LEN)?;
    let mut fused = DeviceBuffer::<f32>::zeroed(&fixture.stream, LEN)?;
    let mut standalone_chunks = DeviceBuffer::<f32>::zeroed(&fixture.stream, LEN)?;
    let chunk_count = tma_nvfp4_output_amax_chunks(ROWS as u32, DIM as u32);
    let mut fused_chunks = DeviceBuffer::<f32>::zeroed(&fixture.stream, chunk_count as usize)?;

    let launched_chunks = fixture.tma_linear3_with_output_amax(
        &source,
        &action,
        &bound_amax,
        &mut standalone,
        &mut standalone_chunks,
        &mut fused,
        &mut fused_chunks,
        A_SCALE,
        B_SCALE,
        C_SCALE,
    )?;
    assert_eq!(launched_chunks, chunk_count);

    let standalone = standalone.to_host_vec(&fixture.stream)?;
    let fused = fused.to_host_vec(&fixture.stream)?;
    assert_eq!(
        fused, standalone,
        "fused linear3 epilogue changed the Muon update"
    );
    let output_amax = fused.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let fused_amax = fused_chunks
        .to_host_vec(&fixture.stream)?
        .into_iter()
        .fold(0.0_f32, f32::max);
    assert_eq!(fused_amax, output_amax);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_affine_padded_output_matches_old_qkv_projection() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, N)?;
    let mut old_out = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * N)?;
    let mut tma_out = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * N)?;

    fixture.attention.qkv_projection(QkvProjectionArgs {
        stream: &fixture.stream,
        input: fixture.input(),
        weight: fixture.weight_mma(),
        bias: fixture.bias_device(),
        out: &mut old_out,
        token_count: ROWS as u32,
        input_dim: K as u32,
        output_dim: N as u32,
    })?;
    fixture.tma_affine(&mut tma_out)?;

    common::assert_slice_close(
        &tma_out.to_host_vec(&fixture.stream)?,
        &old_out.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_residual_matches_old_projection_residual_add() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, 128)?;
    let residual = residual_values(ROWS * 128);
    let mut old_residual = DeviceBuffer::from_host(&fixture.stream, &residual)?;
    let mut tma_residual = DeviceBuffer::from_host(&fixture.stream, &residual)?;

    fixture.attention.c_proj(CProjArgs {
        stream: &fixture.stream,
        input: fixture.input(),
        weight: fixture.weight_mma(),
        bias: fixture.bias_device(),
        residual: &mut old_residual,
        token_count: ROWS as u32,
        embedding_dim: 128,
    })?;
    fixture.tma_residual(&mut tma_residual)?;

    common::assert_slice_close(
        &tma_residual.to_host_vec(&fixture.stream)?,
        &old_residual.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_relu2_matches_old_mlp_up_projection() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, 128)?;
    let mut old_pre = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut old_act = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut tma_pre = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut tma_pre_f16 = DeviceBuffer::<u16>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut compact_pre_f16 = DeviceBuffer::<u16>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut reference_pre_f16 = DeviceBuffer::<u16>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut tma_act = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut compact_act = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;

    fixture.mlp.up_relu2(MlpUpRelu2Args {
        stream: &fixture.stream,
        input: fixture.input(),
        weight: fixture.weight_mma(),
        bias: fixture.bias_device(),
        pre_activation: &mut old_pre,
        out: &mut old_act,
        token_count: ROWS as u32,
        input_dim: K as u32,
        output_dim: 128,
    })?;
    fixture.tma_relu2(&mut tma_pre, Some(&mut tma_pre_f16), &mut tma_act)?;
    fixture.tma_relu2_compact(Some(&mut compact_pre_f16), &mut compact_act)?;
    fixture.f16.fp32_to_f16(F16ConvertArgs {
        stream: &fixture.stream,
        src: &tma_pre,
        dst: &mut reference_pre_f16,
        element_count: (ROWS * 128) as u32,
    })?;

    common::assert_slice_close(
        &tma_pre.to_host_vec(&fixture.stream)?,
        &old_pre.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    common::assert_slice_close(
        &tma_act.to_host_vec(&fixture.stream)?,
        &old_act.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    assert_eq!(
        compact_act.to_host_vec(&fixture.stream)?,
        tma_act.to_host_vec(&fixture.stream)?,
        "compact ReLU2 epilogue changed the activation"
    );
    assert_eq!(
        tma_pre_f16.to_host_vec(&fixture.stream)?,
        reference_pre_f16.to_host_vec(&fixture.stream)?,
        "fused ReLU2 tape must match the standalone FP32-to-FP16 conversion"
    );
    assert_eq!(
        compact_pre_f16.to_host_vec(&fixture.stream)?,
        reference_pre_f16.to_host_vec(&fixture.stream)?,
        "compact ReLU2 tape must match the standalone FP32-to-FP16 conversion"
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_mlp_down_residual_matches_old_projection() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, 256, 128)?;
    let residual = residual_values(ROWS * 128);
    let mut old_residual = DeviceBuffer::from_host(&fixture.stream, &residual)?;
    let mut tma_residual = DeviceBuffer::from_host(&fixture.stream, &residual)?;

    fixture.mlp.down_residual(MlpDownResidualArgs {
        stream: &fixture.stream,
        input: fixture.input(),
        weight: fixture.weight_mma(),
        bias: fixture.bias_device(),
        residual: &mut old_residual,
        token_count: ROWS as u32,
        input_dim: 256,
        output_dim: 128,
    })?;
    fixture.tma_residual(&mut tma_residual)?;

    common::assert_slice_close(
        &tma_residual.to_host_vec(&fixture.stream)?,
        &old_residual.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    Ok(())
}

struct Fixture {
    stream: std::sync::Arc<CudaStream>,
    attention: AttentionModule,
    lm_head: LmHeadModule,
    mlp: MlpModule,
    f16: F16TcMatmulModule,
    f32: F32MatrixOpsModule,
    tma: Nvfp4GemmModule,
    scale_pack: Sm120ScalePackModule,
    pad: TmaMatrixPadModule,
    rows: usize,
    k: usize,
    n: usize,
    input_bytes: DeviceBuffer<u8>,
    input_scales: DeviceBuffer<u8>,
    input_globals: DeviceBuffer<f32>,
    weight_bytes: DeviceBuffer<u8>,
    weight_scales: DeviceBuffer<u8>,
    weight_global: DeviceBuffer<f32>,
    bias_bytes: DeviceBuffer<u8>,
    bias_scales: DeviceBuffer<u8>,
    bias_global: DeviceBuffer<f32>,
}

impl Fixture {
    fn new(rows: usize, k: usize, n: usize) -> Result<Self, Box<dyn Error>> {
        let (_, stream, ptx) = common::cuda_test_context()?;
        Ok(Self {
            attention: AttentionModule::from_module(ptx.clone())?,
            lm_head: LmHeadModule::from_module(ptx.clone())?,
            mlp: MlpModule::from_module(ptx.clone())?,
            f16: F16TcMatmulModule::from_module(ptx.clone())?,
            f32: F32MatrixOpsModule::from_module(ptx.clone())?,
            tma: Nvfp4GemmModule::from_module(ptx.clone())?,
            scale_pack: Sm120ScalePackModule::from_module(ptx.clone())?,
            pad: TmaMatrixPadModule::from_module(ptx.clone())?,
            input_bytes: DeviceBuffer::from_host(&stream, &sparse_bytes(rows, k, 13, 7, 0))?,
            input_scales: DeviceBuffer::from_host(&stream, &pattern_scales(rows * k, 1))?,
            input_globals: DeviceBuffer::from_host(&stream, &row_globals(rows))?,
            weight_bytes: DeviceBuffer::from_host(&stream, &sparse_bytes(n, k, 11, 5, 3))?,
            weight_scales: DeviceBuffer::from_host(&stream, &pattern_scales(n * k, 2))?,
            weight_global: DeviceBuffer::from_host(&stream, &[0.75_f32])?,
            bias_bytes: DeviceBuffer::from_host(&stream, &sparse_bytes(1, n, 3, 1, 2))?,
            bias_scales: DeviceBuffer::from_host(&stream, &pattern_scales(n, 3))?,
            bias_global: DeviceBuffer::from_host(&stream, &[0.25_f32])?,
            stream,
            rows,
            k,
            n,
        })
    }

    fn input(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&self.input_bytes, &self.input_scales, &self.input_globals)
    }

    fn weight_mma(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor::new(
            &self.weight_bytes,
            &self.weight_scales,
            &self.weight_global,
        )
    }

    fn bias_device(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor::new(&self.bias_bytes, &self.bias_scales, &self.bias_global)
    }

    fn tma_raw(&self, raw: &mut DeviceBuffer<f32>) -> Result<(), Box<dyn Error>> {
        let padded_n = sm120_scale_padded_mn_extent(self.n);
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed =
            DeviceBuffer::zeroed(&self.stream, sm120_scale_packed_len(padded_n, self.k))?;
        let mut weight_bytes_padded = DeviceBuffer::zeroed(&self.stream, padded_n * self.k / 2)?;
        let weight_bytes = if padded_n == self.n {
            &self.weight_bytes
        } else {
            self.pad.pad_u4_rows(U4RowPadArgs {
                stream: &self.stream,
                input: &self.weight_bytes,
                output: &mut weight_bytes_padded,
                rows: self.n as u32,
                padded_rows: padded_n as u32,
                cols_u4: self.k as u32,
            })?;
            &weight_bytes_padded
        };
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            padded_n as u32,
            &mut descriptors,
        )?;
        self.tma.gemm_tma_nvfp4_rowwise_a_scale_padded_output(
            &self.stream,
            &descriptors,
            raw,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            padded_n as u32,
            &self.input_globals,
            &self.weight_global,
        )?;
        Ok(())
    }

    fn tma_scalar_with_output_amax(
        &self,
        plain: &mut DeviceBuffer<f32>,
        fused: &mut DeviceBuffer<f32>,
        chunk_amax: &mut DeviceBuffer<f32>,
    ) -> Result<u32, Box<dyn Error>> {
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.n), self.k),
        )?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &mut descriptors,
        )?;
        self.tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
                &self.stream,
                &descriptors,
                plain,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
            )?;
        Ok(self
            .tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_with_output_amax(
                &self.stream,
                &descriptors,
                fused,
                chunk_amax,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
            )?)
    }

    fn tma_relu2_backward_f16_with_output_amax(
        &self,
        pre_activation: &DeviceBuffer<u16>,
        plain: &mut DeviceBuffer<f32>,
        fused: &mut DeviceBuffer<f32>,
        chunk_amax: &mut DeviceBuffer<f32>,
    ) -> Result<u32, Box<dyn Error>> {
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.n), self.k),
        )?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &mut descriptors,
        )?;
        self.tma
            .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer(
                &self.stream,
                &descriptors,
                plain,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
            )?;
        Ok(self
            .tma
            .gemm_tma_nvfp4_rowwise_a_scale_relu2_backward_f16_with_output_amax(
                &self.stream,
                &descriptors,
                pre_activation,
                fused,
                chunk_amax,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
            )?)
    }

    fn tma_self_symmetric_with_output_amax(
        &self,
        plain: &mut DeviceBuffer<f32>,
        symmetric: &mut DeviceBuffer<f32>,
        chunk_amax: &mut DeviceBuffer<f32>,
    ) -> Result<u32, Box<dyn Error>> {
        assert_eq!(self.rows, self.n);
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.input_bytes,
            &input_scale_packed,
            self.rows as u32,
            self.k as u32,
            self.rows as u32,
            &mut descriptors,
        )?;
        self.tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
                &self.stream,
                &descriptors,
                plain,
                self.rows as u32,
                self.k as u32,
                self.rows as u32,
                &self.input_globals,
                &self.input_globals,
            )?;
        Ok(self
            .tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_symmetric_with_output_amax(
                &self.stream,
                &descriptors,
                symmetric,
                chunk_amax,
                self.rows as u32,
                self.k as u32,
                &self.input_globals,
            )?)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "test compares explicit standalone and fused operands"
    )]
    fn tma_linear3_with_output_amax(
        &self,
        source: &DeviceBuffer<f32>,
        action: &DeviceBuffer<f32>,
        bound_amax: &DeviceBuffer<f32>,
        standalone: &mut DeviceBuffer<f32>,
        standalone_chunks: &mut DeviceBuffer<f32>,
        fused: &mut DeviceBuffer<f32>,
        fused_chunks: &mut DeviceBuffer<f32>,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
    ) -> Result<u32, Box<dyn Error>> {
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.n), self.k),
        )?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &mut descriptors,
        )?;
        self.tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
                &self.stream,
                &descriptors,
                standalone,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
            )?;
        self.f32
            .linear3_sqrt_bound_a_with_amax(F32Linear3SqrtBoundAmaxArgs {
                stream: &self.stream,
                a: source,
                b: action,
                c_out: standalone,
                bound_amax,
                chunk_amax: standalone_chunks,
                len: (self.rows * self.n) as u32,
                a_scale,
                b_scale,
                c_scale,
            })?;
        Ok(self
            .tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_linear3_with_output_amax(
                &self.stream,
                &descriptors,
                source,
                action,
                fused,
                bound_amax,
                fused_chunks,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
                a_scale,
                b_scale,
                c_scale,
            )?)
    }

    fn tma_affine(&self, out: &mut DeviceBuffer<f32>) -> Result<(), Box<dyn Error>> {
        let padded_n = sm120_scale_padded_mn_extent(self.n);
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed =
            DeviceBuffer::zeroed(&self.stream, sm120_scale_packed_len(padded_n, self.k))?;
        let mut weight_bytes_padded = DeviceBuffer::zeroed(&self.stream, padded_n * self.k / 2)?;
        let weight_bytes = if padded_n == self.n {
            &self.weight_bytes
        } else {
            self.pad.pad_u4_rows(U4RowPadArgs {
                stream: &self.stream,
                input: &self.weight_bytes,
                output: &mut weight_bytes_padded,
                rows: self.n as u32,
                padded_rows: padded_n as u32,
                cols_u4: self.k as u32,
            })?;
            &weight_bytes_padded
        };
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            padded_n as u32,
            &mut descriptors,
        )?;
        self.tma
            .gemm_tma_nvfp4_rowwise_a_scale_affine_padded_output(
                &self.stream,
                &descriptors,
                out,
                self.bias_device(),
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                padded_n as u32,
                &self.input_globals,
                &self.weight_global,
            )?;
        Ok(())
    }

    fn old_raw(&self, out: &mut DeviceBuffer<f32>) -> Result<(), Box<dyn Error>> {
        self.lm_head.logits(LmHeadArgs {
            stream: &self.stream,
            input: self.input(),
            weight: self.weight_mma(),
            logits: out,
            token_count: self.rows as u32,
            input_dim: self.k as u32,
            vocab_size: self.n as u32,
        })?;
        Ok(())
    }

    fn tma_residual(&self, residual: &mut DeviceBuffer<f32>) -> Result<(), Box<dyn Error>> {
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed =
            DeviceBuffer::zeroed(&self.stream, sm120_scale_packed_len(self.n, self.k))?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &mut descriptors,
        )?;
        self.tma.gemm_tma_nvfp4_rowwise_a_scale_residual(
            &self.stream,
            &descriptors,
            residual,
            self.bias_device(),
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &self.input_globals,
            &self.weight_global,
        )?;
        Ok(())
    }

    fn tma_relu2(
        &self,
        pre_activation: &mut DeviceBuffer<f32>,
        pre_activation_f16: Option<&mut DeviceBuffer<u16>>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        let padded_n = sm120_scale_padded_mn_extent(self.n);
        assert_eq!(padded_n, self.n, "fused ReLU2 requires exact output tiles");
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed =
            DeviceBuffer::zeroed(&self.stream, sm120_scale_packed_len(padded_n, self.k))?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            padded_n as u32,
            &mut descriptors,
        )?;
        self.tma.gemm_tma_nvfp4_rowwise_a_scale_relu2(
            &self.stream,
            &descriptors,
            pre_activation,
            pre_activation_f16,
            out,
            self.bias_device(),
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &self.input_globals,
            &self.weight_global,
        )?;
        Ok(())
    }

    fn tma_relu2_compact(
        &self,
        pre_activation_f16: Option<&mut DeviceBuffer<u16>>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        let padded_n = sm120_scale_padded_mn_extent(self.n);
        assert_eq!(padded_n, self.n, "fused ReLU2 requires exact output tiles");
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed =
            DeviceBuffer::zeroed(&self.stream, sm120_scale_packed_len(padded_n, self.k))?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            padded_n as u32,
            &mut descriptors,
        )?;
        self.tma.gemm_tma_nvfp4_rowwise_a_scale_relu2_compact(
            &self.stream,
            &descriptors,
            pre_activation_f16,
            out,
            self.bias_device(),
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &self.input_globals,
            &self.weight_global,
        )?;
        Ok(())
    }
}

fn sparse_bytes(rows: usize, cols: usize, row_mul: usize, col_mul: usize, add: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; rows * cols / 2];
    for row in 0..rows {
        for col in 0..cols {
            if (row * row_mul + col * col_mul + add) % 17 == 0 {
                set_e2m1_one(&mut bytes, row * cols + col);
            }
        }
    }
    bytes
}

fn pattern_scales(elements: usize, offset: usize) -> Vec<u8> {
    let mut scales = one_scales(elements);
    for (index, scale) in scales.iter_mut().enumerate() {
        *scale = [0x30, 0x34, 0x38, 0x3c, 0x40][(index + offset) % 5];
    }
    scales
}

fn row_globals(rows: usize) -> Vec<f32> {
    (0..rows)
        .map(|row| 0.5 + (row % 7) as f32 * 0.125)
        .collect()
}

fn residual_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (index % 23) as f32 * 0.03125 - 0.25)
        .collect()
}

fn action_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (index % 29) as f32 * -0.0234375 + 0.375)
        .collect()
}

fn pre_activation_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (index % 31) as f32 * 0.03125 - 0.4375)
        .collect()
}
