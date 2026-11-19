use cuda_core::{
    CudaStream, DeviceBuffer, DriverError, sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
};

use crate::launch::{grid_x_config, launch_config};
use crate::mma::{
    NVFP4_PROJECTION_CTA_THREADS, NVFP4_PROJECTION_THREADS_PER_BLOCK,
    Nvfp4DeviceScaleMmaWeightTensor, Nvfp4ProjectionParams, projection_cta_grid_dim,
    projection_cta_row_pair_tile_count, projection_cta_shape_aligned, projection_grid_dim,
};
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;
use crate::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;
use crate::nvfp4_tma_matmul::{
    kernels::{TILE_K, TILE_M, TILE_N},
    tma::TmaNvfp4DeviceScaleDescriptors,
};

use super::{
    LinearBackwardDeviceScaleArgs, LinearBackwardModule, LinearBackwardRoute,
    LinearBackwardTmaScratch,
};

macro_rules! device_scale_projection {
    ($this:expr, $args:ident, $input:ident, $weight:ident, $out:ident, rows: $rows:expr, k: $k:expr) => {
        $this
            .module
            .projection
            .linear_backward_projection_device_scale_kernel(
                $args.stream,
                launch_config(
                    projection_grid_dim($rows, $args.input_dim),
                    NVFP4_PROJECTION_THREADS_PER_BLOCK,
                ),
                $args.$input.bytes,
                $args.$input.scales,
                $args.$input.global_scales,
                $args.$weight.bytes,
                $args.$weight.scales,
                $args.$weight.global_scale,
                $args.$out,
                Nvfp4ProjectionParams::new($rows, $k, $args.input_dim).with_global_scales(1.0, 0.0),
            )
    };
}

impl LinearBackwardModule {
    pub fn backward_device_scale_tma(
        &self,
        args: LinearBackwardDeviceScaleArgs<'_, '_>,
        tma: LinearBackwardTmaScratch<'_>,
    ) -> Result<(), DriverError> {
        self.backward_device_scale_tma_impl(args, tma, None, None)
            .map(|_| ())
    }

    pub(crate) fn backward_device_scale_tma_routed(
        &self,
        args: LinearBackwardDeviceScaleArgs<'_, '_>,
        tma: LinearBackwardTmaScratch<'_>,
        route: LinearBackwardRoute<'_>,
    ) -> Result<(), DriverError> {
        self.backward_device_scale_tma_impl(args, tma, None, Some(route))
            .map(|_| ())
    }

    pub(crate) fn backward_device_scale_tma_relu2_backward_f16(
        &self,
        args: LinearBackwardDeviceScaleArgs<'_, '_>,
        tma: LinearBackwardTmaScratch<'_>,
        pre_activation: &DeviceBuffer<u16>,
        output_chunk_amax: &mut DeviceBuffer<f32>,
    ) -> Result<u32, DriverError> {
        self.backward_device_scale_tma_impl(
            args,
            tma,
            Some((pre_activation, output_chunk_amax)),
            None,
        )?
        .ok_or(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE))
    }

    pub(crate) fn backward_device_scale_tma_relu2_backward_f16_routed(
        &self,
        args: LinearBackwardDeviceScaleArgs<'_, '_>,
        tma: LinearBackwardTmaScratch<'_>,
        pre_activation: &DeviceBuffer<u16>,
        output_chunk_amax: &mut DeviceBuffer<f32>,
        route: LinearBackwardRoute<'_>,
    ) -> Result<u32, DriverError> {
        self.backward_device_scale_tma_impl(
            args,
            tma,
            Some((pre_activation, output_chunk_amax)),
            Some(route),
        )?
        .ok_or(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE))
    }

    fn backward_device_scale_tma_impl(
        &self,
        args: LinearBackwardDeviceScaleArgs<'_, '_>,
        tma: LinearBackwardTmaScratch<'_>,
        relu2_backward_f16: Option<(&DeviceBuffer<u16>, &mut DeviceBuffer<f32>)>,
        route: Option<LinearBackwardRoute<'_>>,
    ) -> Result<Option<u32>, DriverError> {
        let dinput_k = nvfp4_tc_matmul_padded_k(args.output_dim);
        let dweight_k = nvfp4_tc_matmul_padded_k(args.token_count);
        let dinput_tma = tma_shape_aligned(args.token_count, dinput_k, args.input_dim);
        let dweight_tma = tma_shape_aligned(args.output_dim, dweight_k, args.input_dim);
        if !dinput_tma || !dweight_tma {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }
        self.tma.prepare_tma_nvfp4_csf_scales_into(
            args.stream,
            args.e_h.bytes,
            args.e_h.scales,
            args.weight_t_h.bytes,
            args.weight_t_h.scales,
            args.token_count,
            dinput_k,
            args.input_dim,
            tma.descriptors,
        )?;
        let dinput_chunk_count =
            if let Some((pre_activation, output_chunk_amax)) = relu2_backward_f16 {
                Some(
                    self.tma
                        .gemm_tma_nvfp4_rowwise_a_scale_relu2_backward_f16_with_output_amax(
                            args.stream,
                            &*tma.descriptors,
                            pre_activation,
                            args.dinput,
                            output_chunk_amax,
                            args.token_count,
                            dinput_k,
                            args.input_dim,
                            args.e_h.global_scales,
                            args.weight_t_h.global_scale,
                            route.map(LinearBackwardRoute::dinput),
                        )?,
                )
            } else {
                self.tma
                    .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf(
                        args.stream,
                        &*tma.descriptors,
                        args.dinput,
                        args.token_count,
                        dinput_k,
                        args.input_dim,
                        args.e_h.global_scales,
                        args.weight_t_h.global_scale,
                        route.map(LinearBackwardRoute::dinput),
                    )?;
                None
            };
        self.backward_weight_device_scale_tma(
            args.stream,
            args.e_t_h,
            args.input_t_h,
            args.dweight,
            args.token_count,
            args.input_dim,
            args.output_dim,
            tma.descriptors,
            route,
        )?;
        Ok(dinput_chunk_count)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "explicit CUDA projection operands"
    )]
    pub(super) fn backward_weight_device_scale_tma(
        &self,
        stream: &CudaStream,
        error_t: Nvfp4RowwiseDeviceTensor<'_>,
        input_t: Nvfp4DeviceScaleMmaWeightTensor<'_>,
        dweight: &mut DeviceBuffer<f32>,
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
        descriptors: &mut TmaNvfp4DeviceScaleDescriptors,
        route: Option<LinearBackwardRoute<'_>>,
    ) -> Result<(), DriverError> {
        let k = nvfp4_tc_matmul_padded_k(token_count);
        if !tma_shape_aligned(output_dim, k, input_dim) {
            return Err(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }
        self.tma.prepare_tma_nvfp4_csf_scales_into(
            stream,
            error_t.bytes,
            error_t.scales,
            input_t.bytes,
            input_t.scales,
            output_dim,
            k,
            input_dim,
            descriptors,
        )?;
        let transform = self
            .weight_gradient_transforms
            .borrow()
            .get(&dweight.cu_deviceptr())
            .copied();
        let route = route.map(LinearBackwardRoute::dweight);
        if let Some(transform) = transform {
            self.tma
                .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf_gradient(
                    stream,
                    descriptors,
                    dweight,
                    output_dim,
                    k,
                    input_dim,
                    error_t.global_scales,
                    input_t.global_scale,
                    route,
                    transform,
                )
        } else {
            self.tma
                .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf(
                    stream,
                    descriptors,
                    dweight,
                    output_dim,
                    k,
                    input_dim,
                    error_t.global_scales,
                    input_t.global_scale,
                    route,
                )
        }
    }

    pub fn backward_device_scale(
        &self,
        args: LinearBackwardDeviceScaleArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let dinput_k = nvfp4_tc_matmul_padded_k(args.output_dim);
        let dweight_k = nvfp4_tc_matmul_padded_k(args.token_count);

        device_scale_projection!(self, args, e_h, weight_t_h, dinput, rows: args.token_count, k: dinput_k)?;
        device_scale_projection!(self, args, e_t_h, input_t_h, dweight, rows: args.output_dim, k: dweight_k)
    }

    pub fn backward_device_scale_cta(
        &self,
        args: LinearBackwardDeviceScaleArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let dinput_k = nvfp4_tc_matmul_padded_k(args.output_dim);
        let dweight_k = nvfp4_tc_matmul_padded_k(args.token_count);
        if !projection_cta_shape_aligned(args.token_count, dinput_k, args.input_dim)
            || !projection_cta_shape_aligned(args.output_dim, dweight_k, args.input_dim)
        {
            return self.backward_device_scale(args);
        }
        let dinput_grid = projection_cta_grid_dim(args.token_count, args.input_dim);
        let dweight_grid = projection_cta_grid_dim(args.output_dim, args.input_dim);
        assert!(dinput_grid.0.is_power_of_two());
        assert!(dweight_grid.0.is_power_of_two());
        let dinput_tiles = projection_cta_row_pair_tile_count(args.token_count, args.input_dim);
        let dweight_tiles = projection_cta_row_pair_tile_count(args.output_dim, args.input_dim);

        self.module
            .projection
            .linear_backward_projection_pair_cta_device_scale_kernel(
                args.stream,
                grid_x_config(dinput_tiles + dweight_tiles, NVFP4_PROJECTION_CTA_THREADS),
                args.e_h.bytes,
                args.e_h.scales,
                args.e_h.global_scales,
                args.weight_t_h.bytes,
                args.weight_t_h.scales,
                args.weight_t_h.global_scale,
                args.dinput,
                dinput_grid.0 - 1,
                dinput_grid.0.trailing_zeros(),
                dinput_tiles,
                args.e_t_h.bytes,
                args.e_t_h.scales,
                args.e_t_h.global_scales,
                args.input_t_h.bytes,
                args.input_t_h.scales,
                args.input_t_h.global_scale,
                args.dweight,
                dweight_grid.0 - 1,
                dweight_grid.0.trailing_zeros(),
                Nvfp4ProjectionParams::new(args.token_count, dinput_k, args.input_dim)
                    .with_global_scales(1.0, 0.0),
                Nvfp4ProjectionParams::new(args.output_dim, dweight_k, args.input_dim)
                    .with_global_scales(1.0, 0.0),
            )
    }
}

fn tma_shape_aligned(m: u32, k: u32, n: u32) -> bool {
    m.is_multiple_of(TILE_M) && k.is_multiple_of(TILE_K) && n.is_multiple_of(TILE_N)
}
