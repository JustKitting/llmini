use std::sync::Arc;

use cuda_core::{
    CudaContext, CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig,
    sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
};
use cuda_device::TmaDescriptor;

use super::cute::{KMajorU4, Sm120KMajorSwizzle, Sm120ScaleLayout};
use super::kernels::polar;
use super::kernels::{
    Nvfp4GemmParams, ROUTE_FEATURE_M_TOKEN_K, ROUTE_FEATURE_N_TOKEN_K, ROUTE_NONE,
    ROUTE_TOKEN_M_FEATURE_K, ROUTE_TOKEN_M_FEATURE_N, TILE_K, TILE_M, TILE_N,
    TMA_NVFP4_RESIDUAL_THREADS_PER_BLOCK, TMA_NVFP4_THREADS_PER_BLOCK, module,
    tma_nvfp4_output_amax_chunks,
};
use super::scale_layout::sm120_scale_tma_shape_padded;
use super::tma::{
    TmaNvfp4DeviceScaleDescriptorKey, TmaNvfp4DeviceScaleDescriptors, encode_u4_tiled_layout,
    encode_u16_tiled_swizzle_64b,
};
use crate::nvfp4::Nvfp4DeviceTensor;

const PACKS_PER_ROW: u32 = TILE_K / 8;
const ROW_SUMSQ_REDUCE_THREADS: u32 = 64;
type Nvfp4TmaOperandLayout = KMajorU4<PACKS_PER_ROW, Sm120KMajorSwizzle<PACKS_PER_ROW>>;

fn zero_f32_scalar(stream: &CudaStream, scalar: &DeviceBuffer<f32>) -> Result<(), DriverError> {
    unsafe {
        cuda_core::memory::memset_d8_async(
            scalar.cu_deviceptr(),
            0,
            std::mem::size_of::<f32>(),
            stream.cu_stream(),
        )
    }
}

#[derive(Clone, Copy)]
pub struct Nvfp4GemmRoute<'a> {
    mode: u32,
    masks: &'a DeviceBuffer<u64>,
    token_tiles: u32,
    feature_tiles: u32,
    active_tiles: u32,
}

impl<'a> Nvfp4GemmRoute<'a> {
    fn new(
        mode: u32,
        masks: &'a DeviceBuffer<u64>,
        token_tiles: u32,
        feature_tiles: u32,
        active_tiles: u32,
    ) -> Self {
        Self {
            mode,
            masks,
            token_tiles,
            feature_tiles,
            active_tiles,
        }
    }

    pub fn token_m_feature_n(
        masks: &'a DeviceBuffer<u64>,
        token_tiles: u32,
        feature_tiles: u32,
        active_tiles: u32,
    ) -> Self {
        Self::new(
            ROUTE_TOKEN_M_FEATURE_N,
            masks,
            token_tiles,
            feature_tiles,
            active_tiles,
        )
    }

    pub fn token_m_feature_k(
        masks: &'a DeviceBuffer<u64>,
        token_tiles: u32,
        feature_tiles: u32,
        active_tiles: u32,
    ) -> Self {
        Self::new(
            ROUTE_TOKEN_M_FEATURE_K,
            masks,
            token_tiles,
            feature_tiles,
            active_tiles,
        )
    }

    pub fn feature_m_token_k(
        masks: &'a DeviceBuffer<u64>,
        token_tiles: u32,
        feature_tiles: u32,
        active_tiles: u32,
    ) -> Self {
        Self::new(
            ROUTE_FEATURE_M_TOKEN_K,
            masks,
            token_tiles,
            feature_tiles,
            active_tiles,
        )
    }

    pub fn feature_n_token_k(
        masks: &'a DeviceBuffer<u64>,
        token_tiles: u32,
        feature_tiles: u32,
        active_tiles: u32,
    ) -> Self {
        Self::new(
            ROUTE_FEATURE_N_TOKEN_K,
            masks,
            token_tiles,
            feature_tiles,
            active_tiles,
        )
    }

    fn is_valid_for(self, m: u32, k: u32, n: u32) -> bool {
        if self.token_tiles == 0
            || self.feature_tiles == 0
            || self.token_tiles > 128
            || self.feature_tiles > u64::BITS
            || self.active_tiles == 0
            || self.active_tiles > self.feature_tiles
            || self.masks.len() < (self.token_tiles + self.feature_tiles * self.token_tiles.div_ceil(64)) as usize
        {
            return false;
        }

        match self.mode {
            ROUTE_TOKEN_M_FEATURE_N => {
                m / TILE_M == self.token_tiles && n / 128 == self.feature_tiles
            }
            ROUTE_TOKEN_M_FEATURE_K => {
                m / TILE_M == self.token_tiles && k / TILE_K == self.feature_tiles
            }
            ROUTE_FEATURE_M_TOKEN_K => {
                m / TILE_M == self.feature_tiles && k / TILE_K == self.token_tiles
            }
            ROUTE_FEATURE_N_TOKEN_K => {
                n / 128 == self.feature_tiles && k / TILE_K == self.token_tiles
            }
            _ => false,
        }
    }
}

fn gemm_params(
    token_count: u32,
    input_dim: u32,
    output_dim: u32,
    global_scale_mode: u32,
    weight_global_scale: f32,
    a_global_scale: u64,
    b_global_scale: u64,
    route: Option<Nvfp4GemmRoute<'_>>,
) -> Nvfp4GemmParams {
    let (route_mode, route_token_tiles, route_feature_tiles, route_active_tiles, route_masks) =
        route.map_or((ROUTE_NONE, 0, 0, 0, 0), |route| {
            (
                route.mode,
                route.token_tiles,
                route.feature_tiles,
                route.active_tiles,
                route.masks.cu_deviceptr(),
            )
        });
    Nvfp4GemmParams {
        token_count,
        input_dim,
        output_dim,
        global_scale_mode,
        weight_global_scale,
        a_global_scale,
        b_global_scale,
        route_mode,
        route_token_tiles,
        route_feature_tiles,
        route_active_tiles,
        route_masks,
    }
}

pub struct Nvfp4GemmModule {
    module: module::LoadedModule,
    polar: polar::module::LoadedModule,
}

impl Nvfp4GemmModule {
    /// Explicit initialization for standalone calls without a fused producer.
    pub fn clear_output_amax(
        &self,
        stream: &CudaStream,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if output.is_empty() {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }
        zero_f32_scalar(stream, output)
    }

    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            polar: polar::module::from_module(module.clone())?,
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
        let mut out = TmaNvfp4DeviceScaleDescriptors::new(stream)?;
        self.prepare_tma_nvfp4_device_scales_into(
            stream,
            input_bytes,
            input_scale_data,
            weight_bytes,
            weight_scale_data,
            token_count,
            input_dim,
            output_dim,
            &mut out,
        )?;
        Ok(out)
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
        self.prepare_tma_scales_impl::<TILE_N>(
            stream,
            input_bytes,
            input_scale_data,
            weight_bytes,
            weight_scale_data,
            token_count,
            input_dim,
            output_dim,
            out,
            0,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TMA descriptors need explicit operands"
    )]
    pub fn prepare_tma_nvfp4_polar_scales_into(
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
        self.prepare_tma_scales_impl::<{ polar::TILE_N }>(
            stream,
            input_bytes,
            input_scale_data,
            weight_bytes,
            weight_scale_data,
            token_count,
            input_dim,
            output_dim,
            out,
            0,
        )
    }

    pub fn prepare_tma_nvfp4_csf_scales_into(
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
        assert_eq!((TILE_M, TILE_K), (128, 128));
        assert!(TILE_N >= 16 && 128 % TILE_N == 0);
        self.prepare_tma_scales_impl::<TILE_N>(
            stream,
            input_bytes,
            input_scale_data,
            weight_bytes,
            weight_scale_data,
            token_count,
            input_dim,
            output_dim,
            out,
            3,
        )
    }

    pub fn prepare_tma_nvfp4_csf_weight_into(
        &self,
        stream: &CudaStream,
        input_bytes: &DeviceBuffer<u8>,
        input_scales: &DeviceBuffer<u8>,
        weight_bytes: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<u8>,
        m: u32,
        k: u32,
        n: u32,
        out: &mut TmaNvfp4DeviceScaleDescriptors,
    ) -> Result<(), DriverError> {
        self.prepare_tma_scales_impl::<TILE_N>(
            stream,
            input_bytes,
            input_scales,
            weight_bytes,
            weight_scales,
            m,
            k,
            n,
            out,
            2,
        )
    }

    fn prepare_tma_scales_impl<const BOX_N: u32>(
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
        csf: u8,
    ) -> Result<(), DriverError> {
        let mut key = TmaNvfp4DeviceScaleDescriptorKey::new(
            [
                input_bytes.cu_deviceptr(),
                weight_bytes.cu_deviceptr(),
                input_scale_data.cu_deviceptr(),
                weight_scale_data.cu_deviceptr()
                    + if csf == 2 {
                        crate::nvfp4::csf::HEADER_BYTES as u64
                    } else {
                        0
                    },
            ],
            token_count,
            input_dim,
            output_dim,
        );
        key.csf = csf;
        key.tile_n = BOX_N;
        if out.activate(key) {
            return Ok(());
        }

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
            BOX_N,
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
            BOX_N as usize,
            TILE_K as usize,
        );
        let a_scales = if csf != 0 {
            super::tma::encode_csf_scales(
                input_scale_data.cu_deviceptr() as usize as *mut _,
                token_count,
                input_dim,
            )?
        } else {
            encode_u16_tiled_swizzle_64b(
                input_scale_data.cu_deviceptr() as usize as *mut _,
                a_scale_shape.width_u16,
                a_scale_shape.height,
                a_scale_shape.row_stride_bytes,
                a_scale_shape.tile_width_u16,
                a_scale_shape.tile_height,
            )?
        };
        let b_scales = if csf != 0 {
            super::tma::encode_csf_scales(
                (weight_scale_data.cu_deviceptr()
                    + if csf == 2 {
                        crate::nvfp4::csf::HEADER_BYTES as u64
                    } else {
                        0
                    }) as usize as *mut _,
                output_dim,
                input_dim,
            )?
        } else {
            encode_u16_tiled_swizzle_64b(
                weight_scale_data.cu_deviceptr() as usize as *mut _,
                b_scale_shape.width_u16,
                b_scale_shape.height,
                b_scale_shape.row_stride_bytes,
                b_scale_shape.tile_width_u16,
                b_scale_shape.tile_height,
            )?
        };
        let (a_metadata, b_metadata) = if csf != 0 {
            (
                super::tma::encode_csf_metadata(
                    (input_scale_data.cu_deviceptr() + 768) as usize as *mut _,
                    token_count,
                    input_dim,
                )?,
                super::tma::encode_csf_metadata(
                    (weight_scale_data.cu_deviceptr()
                        + if csf == 2 {
                            crate::nvfp4::csf::HEADER_BYTES as u64
                        } else {
                            0
                        }
                        + 768) as usize as *mut _,
                    output_dim,
                    input_dim,
                )?,
            )
        } else {
            ([0u64; 16], [0u64; 16])
        };
        out.insert(
            stream,
            key,
            [a, b, a_scales, a_metadata, b_scales, b_metadata],
        )
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

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            1,
            1.0,
            a_global_scale.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            None,
        );
        let csf = tma.is_csf();

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        if csf {
            if params.route_mode == ROUTE_NONE {
                self.module.nvfp4_gemm_tma_csf_dense_exact_kernel(
                    stream,
                    config,
                    tma.a_deviceptr() as *const TmaDescriptor,
                    tma.b_deviceptr() as *const TmaDescriptor,
                    tma.a_scales_deviceptr() as *const TmaDescriptor,
                    tma.b_scales_deviceptr() as *const TmaDescriptor,
                    out,
                    params,
                )
            } else {
                self.module.nvfp4_gemm_tma_csf_exact_kernel(
                    stream,
                    config,
                    tma.a_deviceptr() as *const TmaDescriptor,
                    tma.b_deviceptr() as *const TmaDescriptor,
                    tma.a_scales_deviceptr() as *const TmaDescriptor,
                    tma.b_scales_deviceptr() as *const TmaDescriptor,
                    out,
                    params,
                )
            }
        } else {
            self.module.nvfp4_gemm_tma_exact_kernel(
                stream,
                config,
                tma.a_deviceptr() as *const TmaDescriptor,
                tma.b_deviceptr() as *const TmaDescriptor,
                tma.a_scales_deviceptr() as *const TmaDescriptor,
                tma.b_scales_deviceptr() as *const TmaDescriptor,
                out,
                params,
            )
        }
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
        output_amax: &mut DeviceBuffer<f32>,
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
            || output_amax.is_empty()
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        zero_f32_scalar(stream, output_amax)?;

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            1,
            1.0,
            a_global_scale.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            None,
        );

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_amax_kernel(
            stream,
            config,
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
            out,
            &*output_amax,
            params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_tma_nvfp4_rowwise_a_scale_relu2_backward_f16_with_output_amax(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        pre_activation: &DeviceBuffer<u16>,
        out: &mut DeviceBuffer<f32>,
        output_chunk_amax: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
        route: Option<Nvfp4GemmRoute<'_>>,
    ) -> Result<u32, DriverError> {
        let output_len = token_count as usize * output_dim as usize;
        let chunk_count = tma_nvfp4_output_amax_chunks(token_count, output_dim);
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
            || pre_activation.len() < output_len
            || out.len() < output_len
            || output_chunk_amax.len() < chunk_count as usize
            || route.is_some_and(|route| !route.is_valid_for(token_count, input_dim, output_dim))
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            if a_global_scales.len() == 1 { 1 } else { 2 },
            1.0,
            a_global_scales.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            route,
        );
        let config = LaunchConfig {
            grid_dim: (output_dim / TILE_N, token_count / TILE_M, 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module
            .nvfp4_gemm_tma_csf_relu2_backward_f16_amax_kernel(
                stream,
                config,
                tma.a_deviceptr() as *const TmaDescriptor,
                tma.b_deviceptr() as *const TmaDescriptor,
                tma.a_scales_deviceptr() as *const TmaDescriptor,
                tma.b_scales_deviceptr() as *const TmaDescriptor,
                pre_activation,
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
    pub fn gemm_tma_nvfp4_device_scales_and_global_scale_buffers_symmetric_with_precleared_output_amax(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        out: &mut DeviceBuffer<f32>,
        output_amax: &mut DeviceBuffer<f32>,
        dim: u32,
        input_dim: u32,
        a_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if TILE_M % TILE_N != 0
            || dim % TILE_M != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
            || output_amax.is_empty()
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            dim,
            input_dim,
            dim,
            1,
            1.0,
            a_global_scale.cu_deviceptr(),
            a_global_scale.cu_deviceptr(),
            None,
        );
        let tiles = dim / TILE_M;
        let triangular_tiles = tiles * (tiles + 1) / 2 * (TILE_M / TILE_N);
        let config = LaunchConfig {
            grid_dim: (triangular_tiles, 1, 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_symmetric_amax_kernel(
            stream,
            config,
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
            out,
            &*output_amax,
            params,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TMA GEMM launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_device_scales_and_global_scale_buffers_gram_factor_with_precleared_output_amax(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        gram: &DeviceBuffer<f32>,
        bound_amax: &DeviceBuffer<f32>,
        logical_dim: u32,
        b_scale: f32,
        c_scale: f32,
        apply_bound: bool,
        out: &mut DeviceBuffer<f32>,
        output_amax: &mut DeviceBuffer<f32>,
        dim: u32,
        input_dim: u32,
        a_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if TILE_M % TILE_N != 0
            || dim % TILE_M != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
            || logical_dim == 0
            || logical_dim > dim
            || logical_dim % 2 != 0
            || bound_amax.is_empty()
            || gram.len() < (logical_dim as usize).pow(2)
            || out.len() < (logical_dim as usize).pow(2)
            || output_amax.is_empty()
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            dim,
            input_dim,
            dim,
            1,
            1.0,
            a_global_scale.cu_deviceptr(),
            a_global_scale.cu_deviceptr(),
            None,
        );
        let tiles = dim / TILE_M;
        let triangular_tiles = tiles * (tiles + 1) / 2 * (TILE_M / TILE_N);
        let config = LaunchConfig {
            grid_dim: (triangular_tiles, 1, 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_gram_factor_amax_kernel(
            stream,
            config,
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
            gram,
            bound_amax,
            logical_dim,
            b_scale,
            c_scale,
            u32::from(apply_bound),
            out,
            &*output_amax,
            params,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fused Muon TMA launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_device_scales_and_global_scale_buffers_polar_step_with_precleared_output_amax(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        source: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        bound_amax: &DeviceBuffer<f32>,
        output_amax: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scale: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
        a_scale: f32,
        logical_rows: u32,
        logical_cols: u32,
        apply_bound: bool,
    ) -> Result<(), DriverError> {
        let output_len = logical_rows as usize * logical_cols as usize;
        if token_count % polar::TILE_M != 0
            || output_dim % polar::TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % polar::TILE_K != 0
            || input_dim == 0
            || logical_rows == 0
            || logical_cols == 0
            || logical_rows > token_count
            || logical_cols > output_dim
            || logical_cols % 2 != 0
            || source.len() < output_len
            || out.len() < output_len
            || bound_amax.is_empty()
            || output_amax.is_empty()
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            1,
            1.0,
            a_global_scale.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            None,
        );
        let config = LaunchConfig {
            grid_dim: (output_dim / polar::TILE_N, token_count / polar::TILE_M, 1),
            block_dim: (polar::TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.polar.nvfp4_gemm_tma_polar_step_amax_kernel(
            stream,
            config,
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
            source,
            out,
            bound_amax,
            &*output_amax,
            params,
            a_scale,
            logical_rows,
            logical_cols,
            u32::from(apply_bound),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fused Muon TMA launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_device_scales_and_global_scale_buffers_polar_step_with_precleared_row_sumsq(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        source: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        source_bound_amax: &DeviceBuffer<f32>,
        output_bound_amax: &mut DeviceBuffer<f32>,
        row_sumsq_partials: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scale: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
        a_scale: f32,
        logical_rows: u32,
        logical_cols: u32,
        apply_bound: bool,
    ) -> Result<(), DriverError> {
        let output_len = logical_rows as usize * logical_cols as usize;
        let partials_per_row = output_dim / polar::TILE_N;
        let partial_count = logical_rows as usize * partials_per_row as usize;
        if token_count % polar::TILE_M != 0
            || output_dim % polar::TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % polar::TILE_K != 0
            || input_dim == 0
            || logical_rows == 0
            || logical_cols == 0
            || logical_rows > token_count
            || logical_cols > output_dim
            || logical_cols % 2 != 0
            || source.len() < output_len
            || out.len() < output_len
            || source_bound_amax.is_empty()
            || output_bound_amax.is_empty()
            || source_bound_amax.cu_deviceptr() == output_bound_amax.cu_deviceptr()
            || row_sumsq_partials.len() < partial_count
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            1,
            1.0,
            a_global_scale.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            None,
        );
        let config = LaunchConfig {
            grid_dim: (output_dim / polar::TILE_N, token_count / polar::TILE_M, 1),
            block_dim: (polar::TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        self.polar.nvfp4_gemm_tma_polar_step_row_sumsq_kernel(
            stream,
            config,
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
            source,
            out,
            source_bound_amax,
            row_sumsq_partials,
            params,
            a_scale,
            logical_rows,
            logical_cols,
            u32::from(apply_bound),
        )?;
        self.module.nvfp4_gemm_tma_row_sumsq_reduce_kernel(
            stream,
            LaunchConfig {
                grid_dim: (logical_rows, 1, 1),
                block_dim: (ROW_SUMSQ_REDUCE_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            &*row_sumsq_partials,
            &*output_bound_amax,
            logical_rows,
            partials_per_row,
        )
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
        route: Option<Nvfp4GemmRoute<'_>>,
    ) -> Result<(), DriverError> {
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
            || route.is_some_and(|route| !route.is_valid_for(token_count, input_dim, output_dim))
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            2,
            1.0,
            a_global_scales.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            route,
        );
        let csf = tma.is_csf();

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        if csf {
            if params.route_mode == ROUTE_NONE {
                self.module.nvfp4_gemm_tma_csf_dense_exact_kernel(
                    stream,
                    config,
                    tma.a_deviceptr() as *const TmaDescriptor,
                    tma.b_deviceptr() as *const TmaDescriptor,
                    tma.a_scales_deviceptr() as *const TmaDescriptor,
                    tma.b_scales_deviceptr() as *const TmaDescriptor,
                    out,
                    params,
                )
            } else {
                self.module.nvfp4_gemm_tma_csf_exact_kernel(
                    stream,
                    config,
                    tma.a_deviceptr() as *const TmaDescriptor,
                    tma.b_deviceptr() as *const TmaDescriptor,
                    tma.a_scales_deviceptr() as *const TmaDescriptor,
                    tma.b_scales_deviceptr() as *const TmaDescriptor,
                    out,
                    params,
                )
            }
        } else {
            self.module.nvfp4_gemm_tma_exact_kernel(
                stream,
                config,
                tma.a_deviceptr() as *const TmaDescriptor,
                tma.b_deviceptr() as *const TmaDescriptor,
                tma.a_scales_deviceptr() as *const TmaDescriptor,
                tma.b_scales_deviceptr() as *const TmaDescriptor,
                out,
                params,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        out: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
        route: Option<Nvfp4GemmRoute<'_>>,
    ) -> Result<(), DriverError> {
        self.gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf_impl(
            stream,
            tma,
            out,
            token_count,
            input_dim,
            output_dim,
            a_global_scales,
            b_global_scale,
            route,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf_gradient(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        out: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
        route: Option<Nvfp4GemmRoute<'_>>,
        gradient: super::gradient::GemmGradientTransform,
    ) -> Result<(), DriverError> {
        self.gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf_impl(
            stream,
            tma,
            out,
            token_count,
            input_dim,
            output_dim,
            a_global_scales,
            b_global_scale,
            route,
            Some(gradient),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf_impl(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        out: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
        route: Option<Nvfp4GemmRoute<'_>>,
        gradient: Option<super::gradient::GemmGradientTransform>,
    ) -> Result<(), DriverError> {
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
            || out.len() < token_count as usize * output_dim as usize
            || route.is_some_and(|route| !route.is_valid_for(token_count, input_dim, output_dim))
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            if a_global_scales.len() == 1 { 1 } else { 2 },
            1.0,
            a_global_scales.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            route,
        );

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        if let Some(gradient) = gradient {
            assert!(gradient.len <= token_count * output_dim);
            assert_eq!(
                gradient.len.div_ceil(output_dim).div_ceil(TILE_M),
                token_count.div_ceil(TILE_M),
                "fused gradient includes the active prefix and its final padded tile"
            );
            assert!(gradient.beta > 0.0 && gradient.beta.is_finite());
            if params.route_mode == ROUTE_NONE {
                return self.module.nvfp4_gemm_tma_csf_dense_gradient_kernel(
                    stream,
                    config,
                    tma.a_deviceptr() as *const TmaDescriptor,
                    tma.b_deviceptr() as *const TmaDescriptor,
                    tma.a_scales_deviceptr() as *const TmaDescriptor,
                    tma.b_scales_deviceptr() as *const TmaDescriptor,
                    out,
                    params,
                    gradient,
                );
            }
            return self.module.nvfp4_gemm_tma_csf_gradient_kernel(
                stream,
                config,
                tma.a_deviceptr() as *const TmaDescriptor,
                tma.b_deviceptr() as *const TmaDescriptor,
                tma.a_scales_deviceptr() as *const TmaDescriptor,
                tma.b_scales_deviceptr() as *const TmaDescriptor,
                out,
                params,
                gradient,
            );
        }
        if params.route_mode == ROUTE_NONE {
            self.module.nvfp4_gemm_tma_csf_dense_exact_kernel(
                stream,
                config,
                tma.a_deviceptr() as *const TmaDescriptor,
                tma.b_deviceptr() as *const TmaDescriptor,
                tma.a_scales_deviceptr() as *const TmaDescriptor,
                tma.b_scales_deviceptr() as *const TmaDescriptor,
                out,
                params,
            )
        } else {
            self.module.nvfp4_gemm_tma_csf_exact_kernel(
                stream,
                config,
                tma.a_deviceptr() as *const TmaDescriptor,
                tma.b_deviceptr() as *const TmaDescriptor,
                tma.a_scales_deviceptr() as *const TmaDescriptor,
                tma.b_scales_deviceptr() as *const TmaDescriptor,
                out,
                params,
            )
        }
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

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            2,
            1.0,
            a_global_scales.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            None,
        );

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
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
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
        route: Option<Nvfp4GemmRoute<'_>>,
    ) -> Result<(), DriverError> {
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
            || route.is_some_and(|route| !route.is_valid_for(token_count, input_dim, output_dim))
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            2,
            1.0,
            a_global_scales.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            route,
        );

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_RESIDUAL_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        self.module.nvfp4_gemm_tma_residual_kernel(
            stream,
            config,
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
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
        pre_activation_f16: Option<&mut DeviceBuffer<u16>>,
        out: &mut DeviceBuffer<f32>,
        bias: Nvfp4DeviceTensor<'_>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if let Some(pre_activation_f16) = pre_activation_f16.as_ref() {
            assert!(pre_activation_f16.len() >= (token_count * output_dim) as usize);
        }
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            2,
            1.0,
            a_global_scales.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            None,
        );

        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let pre_activation_f16 = pre_activation_f16
            .map(|values| values.cu_deviceptr() as *mut u16)
            .unwrap_or(core::ptr::null_mut());

        self.module.nvfp4_gemm_tma_relu2_kernel(
            stream,
            config,
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
            pre_activation,
            pre_activation_f16,
            out,
            bias.bytes,
            bias.scales,
            bias.global_scale,
            params,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "compact fused TMA ReLU2 launch uses explicit buffers"
    )]
    pub fn gemm_tma_nvfp4_rowwise_a_scale_relu2_compact(
        &self,
        stream: &CudaStream,
        tma: &TmaNvfp4DeviceScaleDescriptors,
        pre_activation_f16: Option<&mut DeviceBuffer<u16>>,
        out: &mut DeviceBuffer<f32>,
        bias: Nvfp4DeviceTensor<'_>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        a_global_scales: &DeviceBuffer<f32>,
        b_global_scale: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if let Some(pre_activation_f16) = pre_activation_f16.as_ref() {
            assert!(pre_activation_f16.len() >= (token_count * output_dim) as usize);
        }
        if token_count % TILE_M != 0
            || output_dim % TILE_N != 0
            || input_dim % Sm120ScaleLayout::K_ATOM != 0
            || input_dim % TILE_K != 0
            || input_dim == 0
        {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            2,
            1.0,
            a_global_scales.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            None,
        );
        let config = LaunchConfig {
            grid_dim: (output_dim.div_ceil(TILE_N), token_count.div_ceil(TILE_M), 1),
            block_dim: (TMA_NVFP4_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let pre_activation_f16 = pre_activation_f16
            .map(|values| values.cu_deviceptr() as *mut u16)
            .unwrap_or(core::ptr::null_mut());

        self.module.nvfp4_gemm_tma_relu2_compact_kernel(
            stream,
            config,
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
            pre_activation_f16,
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

        let params = gemm_params(
            token_count,
            input_dim,
            output_dim,
            2,
            1.0,
            a_global_scales.cu_deviceptr(),
            b_global_scale.cu_deviceptr(),
            None,
        );

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
            tma.a_deviceptr() as *const TmaDescriptor,
            tma.b_deviceptr() as *const TmaDescriptor,
            tma.a_scales_deviceptr() as *const TmaDescriptor,
            tma.b_scales_deviceptr() as *const TmaDescriptor,
            out,
            params,
        )
    }
}
