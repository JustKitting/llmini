use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::args::{
    MlpBlockTopKRouteArgs, MlpDownResidualArgs, MlpUpRelu2Args, Relu2BackwardArgs,
    Relu2BackwardF16Args,
};
use super::kernels;
use crate::launch::{launch_config, linear_config};
use crate::mma::{
    NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionParams, projection_cta_launch_grid_dim,
};

pub struct MlpModule {
    module: kernels::LoadedModule,
}

macro_rules! projection_launcher {
    ($method:ident, $args:ty, $kernel:ident, $residual_add:expr, outputs($($out:ident),+)) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            self.module.$kernel(
                args.stream,
                projection_config(args.token_count, args.input_dim, args.output_dim),
                args.input.bytes, args.input.scales, args.input.global_scales,
                args.weight.bytes, args.weight.scales, args.bias.bytes, args.bias.scales,
                args.weight.global_scale, args.bias.global_scale, $(args.$out,)*
                Nvfp4ProjectionParams::new(args.token_count, args.input_dim, args.output_dim).with_residual_add($residual_add),
            )
        }
    };
}

macro_rules! relu2_backward_launcher {
    ($method:ident, $args:ty, $kernel:ident) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            self.module.$kernel(
                args.stream,
                linear_config(args.len, kernels::RELU2_THREADS_PER_BLOCK),
                args.pre_activation,
                args.d_out,
                args.d_pre_activation,
                args.len,
            )
        }
    };
}

impl MlpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    projection_launcher!(
        up_relu2,
        MlpUpRelu2Args<'_, '_>,
        mlp_projection_relu2_kernel,
        0,
        outputs(pre_activation, out)
    );
    projection_launcher!(
        down_residual,
        MlpDownResidualArgs<'_, '_>,
        mlp_projection_kernel,
        1,
        outputs(residual)
    );
    relu2_backward_launcher!(
        relu2_backward,
        Relu2BackwardArgs<'_, '_>,
        relu2_backward_kernel
    );
    pub fn relu2_backward_f16(
        &self,
        args: Relu2BackwardF16Args<'_, '_>,
    ) -> Result<u32, DriverError> {
        let chunk_count = relu2_backward_amax_chunks(args.len);
        assert!(args.d_pre_activation_chunk_amax.len() >= chunk_count as usize);
        self.module.relu2_backward_f16_kernel(
            args.stream,
            launch_config((chunk_count, 1, 1), kernels::RELU2_THREADS_PER_BLOCK),
            args.pre_activation,
            args.d_out,
            args.d_pre_activation,
            args.d_pre_activation_chunk_amax,
            args.len,
        )?;
        Ok(chunk_count)
    }

    pub fn block_topk_route(&self, args: MlpBlockTopKRouteArgs<'_, '_>) -> Result<(), DriverError> {
        let token_tiles = args.token_count / kernels::BLOCK_TOPK_TILE;
        let feature_tiles = args.feature_count / kernels::BLOCK_TOPK_TILE;
        assert_eq!(token_tiles, kernels::BLOCK_TOPK_TOKEN_TILES);
        assert_eq!(feature_tiles, kernels::BLOCK_TOPK_FEATURE_TILES);
        assert!(
            args.scores.len() >= (token_tiles * feature_tiles) as usize,
            "block-Top-K score scratch is too small"
        );
        assert!(
            args.masks.len() >= (token_tiles + feature_tiles) as usize,
            "block-Top-K dual-layout mask buffer is too small"
        );

        self.module.mlp_block_topk_scores_nvfp4_kernel(
            args.stream,
            launch_config(
                (feature_tiles, token_tiles, 1),
                kernels::BLOCK_TOPK_SCORE_THREADS,
            ),
            args.activation.bytes,
            args.activation.scales,
            args.activation.global_scales,
            args.scores,
            args.token_count,
            args.feature_count,
        )?;
        self.module.mlp_block_topk_masks_kernel(
            args.stream,
            launch_config((1, 1, 1), kernels::BLOCK_TOPK_MASK_THREADS),
            &*args.scores,
            args.masks,
            token_tiles,
            feature_tiles,
            kernels::BLOCK_TOPK_ACTIVE_FEATURE_TILES,
        )
    }
}

pub const fn relu2_backward_amax_chunks(len: u32) -> u32 {
    len.div_ceil(kernels::RELU2_VALUES_PER_BLOCK)
}

fn projection_config(token_count: u32, input_dim: u32, output_dim: u32) -> cuda_core::LaunchConfig {
    launch_config(
        projection_cta_launch_grid_dim(token_count, input_dim, output_dim),
        NVFP4_PROJECTION_CTA_THREADS,
    )
}
