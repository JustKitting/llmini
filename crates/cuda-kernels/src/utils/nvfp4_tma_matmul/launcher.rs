use std::sync::Arc;

use cuda_core::{
    CudaContext, CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig, memory,
    sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
};
use cuda_device::TmaDescriptor;

use super::cute::{KMajorU4, Sm120KMajorSwizzle, Sm120ScaleLayout};
use super::kernels::{
    Nvfp4GemmParams, TILE_K, TILE_M, TILE_N, TMA_NVFP4_THREADS_PER_BLOCK, module,
    tma_nvfp4_output_amax_chunks,
};
use super::scale_layout::sm120_scale_tma_shape_padded;
use super::tma::{TmaNvfp4DeviceScaleDescriptors, encode_u4_tiled_layout, encode_u16_tiled};
use crate::nvfp4::Nvfp4DeviceTensor;

const PACKS_PER_ROW: u32 = TILE_K / 8;
type Nvfp4TmaOperandLayout = KMajorU4<PACKS_PER_ROW, Sm120KMajorSwizzle<PACKS_PER_ROW>>;

pub struct Nvfp4GemmModule {
    module: module::LoadedModule,
}

impl Nvfp4GemmModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: module::from_module(module)?,
        })
    }

    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        let loaded = module::load(ctx).map_err(|_| DriverError(500))?;
        Self::from_module(loaded.as_cuda_module().clone())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TMA descriptors need explicit operands"
    )]
    pub fn prepare_tma_nvfp4_device_scales(
        &self,
        stream: &CudaStream,
        input_bytes: &DeviceBuffer<u8>,
        input_scale_data: &DeviceBuffer<u8>,
        weight_bytes: &DeviceBuffer<u8>,
        weight_scale_data: &DeviceBuffer<u8>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<TmaNvfp4DeviceScaleDescriptors, DriverError> {
        let packed_row_stride = (input_dim / 2) as u64;
        let a = encode_u4_tiled_layout::<Nvfp4TmaOperandLayout>(
            input_bytes.cu_deviceptr() as usize as *mut _,
            input_dim as u64,
            token_count as u64,
            packed_row_stride,
            TILE_M,
        )?;
        let b = encode_u4_tiled_layout::<Nvfp4TmaOperandLayout>(
            weight_bytes.cu_deviceptr() as usize as *mut _,
            input_dim as u64,
            output_dim as u64,
            packed_row_stride,
            TILE_N,
        )?;
        let a_scale_shape = sm120_scale_tma_shape_padded(
            token_count as usize,
            input_dim as usize,
            TILE_M as usize,
            TILE_K as usize,
        );
        let b_scale_shape = sm120_scale_tma_shape_padded(
            output_dim as usize,
            input_dim as usize,
            TILE_N as usize,
            TILE_K as usize,
        );
        let a_scales = encode_u16_tiled(
            input_scale_data.cu_deviceptr() as usize as *mut _,
            a_scale_shape.width_u16,
            a_scale_shape.height,
            a_scale_shape.row_stride_bytes,
            a_scale_shape.tile_width_u16,
            a_scale_shape.tile_height,
        )?;
        let b_scales = encode_u16_tiled(
            weight_scale_data.cu_deviceptr() as usize as *mut _,
            b_scale_shape.width_u16,
            b_scale_shape.height,
            b_scale_shape.row_stride_bytes,
            b_scale_shape.tile_width_u16,
            b_scale_shape.tile_height,
        )?;

        Ok(TmaNvfp4DeviceScaleDescriptors {
            a: DeviceBuffer::from_host(stream, &[a])?,
            b: DeviceBuffer::from_host(stream, &[b])?,
            a_scales: DeviceBuffer::from_host(stream, &[a_scales])?,
            b_scales: DeviceBuffer::from_host(stream, &[b_scales])?,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TMA descriptors need explicit operands"
    )]
    pub fn prepare_tma_nvfp4_device_scales_into(
        &self,
        stream: &CudaStream,
        input_bytes: &DeviceBuffer<u8>,
        input_scale_data: &DeviceBuffer<u8>,
        weight_bytes: &DeviceBuffer<u8>,
        weight_scale_data: &DeviceBuffer<u8>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        out: &mut TmaNvfp4DeviceScaleDescriptors,
    ) -> Result<(), DriverError> {
        let packed_row_stride = (input_dim / 2) as u64;
        let a = encode_u4_tiled_layout::<Nvfp4TmaOperandLayout>(
            input_bytes.cu_deviceptr() as usize as *mut _,
            input_dim as u64,
            token_count as u64,
            packed_row_stride,
            TILE_M,
        )?;
        let b = encode_u4_tiled_layout::<Nvfp4TmaOperandLayout>(
            weight_bytes.cu_deviceptr() as usize as *mut _,
            input_dim as u64,
            output_dim as u64,
            packed_row_stride,
            TILE_N,
        )?;
        let a_scale_shape = sm120_scale_tma_shape_padded(
            token_count as usize,
            input_dim as usize,
            TILE_M as usize,
            TILE_K as usize,
        );
        let b_scale_shape = sm120_scale_tma_shape_padded(
            output_dim as usize,
            input_dim as usize,
            TILE_N as usize,
            TILE_K as usize,
        );
        let a_scales = encode_u16_tiled(
            input_scale_data.cu_deviceptr() as usize as *mut _,
            a_scale_shape.width_u16,
            a_scale_shape.height,
            a_scale_shape.row_stride_bytes,
            a_scale_shape.tile_width_u16,
            a_scale_shape.tile_height,
        )?;
        let b_scales = encode_u16_tiled(
            weight_scale_data.cu_deviceptr() as usize as *mut _,
            b_scale_shape.width_u16,
            b_scale_shape.height,
            b_scale_shape.row_stride_bytes,
            b_scale_shape.tile_width_u16,
            b_scale_shape.tile_height,
        )?;

        copy_descriptor(stream, &mut out.a, &a)?;
        copy_descriptor(stream, &mut out.b, &b)?;
        copy_descriptor(stream, &mut out.a_scales, &a_scales)?;
        copy_descriptor(stream, &mut out.b_scales, &b_scales)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TMA GEMM launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        out: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scale: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = Nvfp4GemmParams {
            token_count,
            input_dim,
            output_dim,
            global_scale_mode: 1,
            weight_global_scale: 1.0,
            a_global_scale: a_global_scale.cu_deviceptr(),
            b_global_scale: b_global_scale.cu_deviceptr(),
        };

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_kernel(
            stream,
            config,
            tma.a.cu_deviceptr() as *const TmaDescriptor,
            tma.b.cu_deviceptr() as *const TmaDescriptor,
            tma.a_scales.cu_deviceptr() as *const TmaDescriptor,
            tma.b_scales.cu_deviceptr() as *const TmaDescriptor,
            out,
            params,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TMA GEMM launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_device_scales_and_global_scale_buffers_with_output_amax(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        out: &mut DeviceBuffer<f32>,
        output_chunk_amax: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scale: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
    ) -> Result<u32, DriverError> {
        let chunk_count = tma_nvfp4_output_amax_chunks(token_count, output_dim);
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
            || output_chunk_amax.len() < chunk_count as usize
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = Nvfp4GemmParams {
            token_count,
            input_dim,
            output_dim,
            global_scale_mode: 1,
            weight_global_scale: 1.0,
            a_global_scale: a_global_scale.cu_deviceptr(),
            b_global_scale: b_global_scale.cu_deviceptr(),
        };

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_amax_kernel(
            stream,
            config,
            tma.a.cu_deviceptr() as *const TmaDescriptor,
            tma.b.cu_deviceptr() as *const TmaDescriptor,
            tma.a_scales.cu_deviceptr() as *const TmaDescriptor,
            tma.b_scales.cu_deviceptr() as *const TmaDescriptor,
            out,
            output_chunk_amax,
            params,
        )?;
        Ok(chunk_count)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TMA GEMM launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        out: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = Nvfp4GemmParams {
            token_count,
            input_dim,
            output_dim,
            global_scale_mode: 2,
            weight_global_scale: 1.0,
            a_global_scale: a_global_scales.cu_deviceptr(),
            b_global_scale: b_global_scale.cu_deviceptr(),
        };

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_kernel(
            stream,
            config,
            tma.a.cu_deviceptr() as *const TmaDescriptor,
            tma.b.cu_deviceptr() as *const TmaDescriptor,
            tma.a_scales.cu_deviceptr() as *const TmaDescriptor,
            tma.b_scales.cu_deviceptr() as *const TmaDescriptor,
            out,
            params,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fused TMA affine launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_rowwise_a_scale_affine_padded_output(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        out: &mut DeviceBuffer<f32>,
        bias: Nvfp4DeviceTensor<'_>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        padded_output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if token_count % TILE_M != 0
            || padded_output_dim % TILE_N != 0
            || output_dim > padded_output_dim
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = Nvfp4GemmParams {
            token_count,
            input_dim,
            output_dim,
            global_scale_mode: 2,
            weight_global_scale: 1.0,
            a_global_scale: a_global_scales.cu_deviceptr(),
            b_global_scale: b_global_scale.cu_deviceptr(),
        };

        let config = LaunchConfig {
            grid_dim: (
                padded_output_dim.div_ceil(TILE_N),
                token_count.div_ceil(TILE_M),
                1,
            ),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_affine_kernel(
            stream,
            config,
            tma.a.cu_deviceptr() as *const TmaDescriptor,
            tma.b.cu_deviceptr() as *const TmaDescriptor,
            tma.a_scales.cu_deviceptr() as *const TmaDescriptor,
            tma.b_scales.cu_deviceptr() as *const TmaDescriptor,
            out,
            bias.bytes,
            bias.scales,
            bias.global_scale,
            params,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fused TMA residual launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_rowwise_a_scale_residual(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        residual: &mut DeviceBuffer<f32>,
        bias: Nvfp4DeviceTensor<'_>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = Nvfp4GemmParams {
            token_count,
            input_dim,
            output_dim,
            global_scale_mode: 2,
            weight_global_scale: 1.0,
            a_global_scale: a_global_scales.cu_deviceptr(),
            b_global_scale: b_global_scale.cu_deviceptr(),
        };

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_residual_kernel(
            stream,
            config,
            tma.a.cu_deviceptr() as *const TmaDescriptor,
            tma.b.cu_deviceptr() as *const TmaDescriptor,
            tma.a_scales.cu_deviceptr() as *const TmaDescriptor,
            tma.b_scales.cu_deviceptr() as *const TmaDescriptor,
            residual,
            bias.bytes,
            bias.scales,
            bias.global_scale,
            params,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fused TMA ReLU2 launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_rowwise_a_scale_relu2(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        pre_activation: &mut DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        bias: Nvfp4DeviceTensor<'_>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = Nvfp4GemmParams {
            token_count,
            input_dim,
            output_dim,
            global_scale_mode: 2,
            weight_global_scale: 1.0,
            a_global_scale: a_global_scales.cu_deviceptr(),
            b_global_scale: b_global_scale.cu_deviceptr(),
        };

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_relu2_kernel(
            stream,
            config,
            tma.a.cu_deviceptr() as *const TmaDescriptor,
            tma.b.cu_deviceptr() as *const TmaDescriptor,
            tma.a_scales.cu_deviceptr() as *const TmaDescriptor,
            tma.b_scales.cu_deviceptr() as *const TmaDescriptor,
            pre_activation,
            out,
            bias.bytes,
            bias.scales,
            bias.global_scale,
            params,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TMA GEMM launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_rowwise_a_scale_padded_output(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        out: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        padded_output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if token_count % TILE_M != 0
            || padded_output_dim % TILE_N != 0
            || output_dim > padded_output_dim
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = Nvfp4GemmParams {
            token_count,
            input_dim,
            output_dim,
            global_scale_mode: 2,
            weight_global_scale: 1.0,
            a_global_scale: a_global_scales.cu_deviceptr(),
            b_global_scale: b_global_scale.cu_deviceptr(),
        };

        let config = LaunchConfig {
            grid_dim: (
                padded_output_dim.div_ceil(TILE_N),
                token_count.div_ceil(TILE_M),
                1,
            ),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_kernel(
            stream,
            config,
            tma.a.cu_deviceptr() as *const TmaDescriptor,
            tma.b.cu_deviceptr() as *const TmaDescriptor,
            tma.a_scales.cu_deviceptr() as *const TmaDescriptor,
            tma.b_scales.cu_deviceptr() as *const TmaDescriptor,
            out,
            params,
        )
    }
}

fn copy_descriptor(
    stream: &CudaStream,
    dst: &mut DeviceBuffer<[u64; 16]>,
    src: &[u64; 16],
) -> Result<(), DriverError> {
    debug_assert_eq!(dst.len(), 1);
    unsafe {
        memory::memcpy_htod_async(
            dst.cu_deviceptr(),
            src as *const [u64; 16],
            std::mem::size_of_val(src),
            stream.cu_stream(),
        )
    }
}
