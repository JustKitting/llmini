use std::sync::{Arc, OnceLock};

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};

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
    bounded: Option<crate::bounded_swiglu::kernels::LoadedModule>,
    graph: Option<crate::bounded_swiglu::graph::LoadedModule>,
    precision: Option<crate::bounded_swiglu::precision_graph::LoadedModule>,
    graph_verified: OnceLock<bool>,
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
        let bounded = if super::config::BOUNDED_SWIGLU {
            Some(crate::bounded_swiglu::kernels::from_module(module.clone())?)
        } else {
            None
        };
        let graph = crate::bounded_swiglu::graph::from_module(module.clone()).ok();
        let precision = if super::config::BOUNDED_SWIGLU {
            if !crate::module_exports::contains(
                &module,
                b"bounded_swiglu_precision_signature_kernel",
            )? {
                eprintln!(
                    "bounded SwiGLU requires the FP16-input precision signature; refusing the old FP32-input prototype"
                );
                return Err(DriverError(1));
            }
            Some(crate::bounded_swiglu::precision_graph::from_module(
                module.clone(),
            )?)
        } else {
            None
        };
        Ok(Self {
            module: kernels::from_module(module)?,
            bounded,
            graph,
            precision,
            graph_verified: OnceLock::new(),
        })
    }

    pub fn verify_graph(&self, stream: &CudaStream) -> Result<(), DriverError> {
        let verified = *self.graph_verified.get_or_init(|| {
            let Some(graph) = &self.graph else {
                return !super::config::BOUNDED_SWIGLU
                    && !crate::nvfp4_quant::ACTIVATION_MAE_FOUR_SIX;
            };
            let result = (|| -> Result<Vec<u32>, DriverError> {
                let mut out = DeviceBuffer::zeroed(stream, 4)?;
                graph.mlp_graph_signature_kernel(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut out,
                )?;
                let mut actual = out.to_host_vec(stream)?;
                if let Some(precision) = &self.precision {
                    let mut input_policy = DeviceBuffer::zeroed(stream, 2)?;
                    precision.bounded_swiglu_precision_signature_kernel(
                        stream,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut input_policy,
                    )?;
                    actual.extend(input_policy.to_host_vec(stream)?);
                }
                Ok(actual)
            })();
            let mut expected = vec![
                super::config::BOUNDED_SWIGLU as u32,
                super::config::FEATURE_TILES,
                super::config::ACTIVE_FEATURE_TILES,
                crate::nvfp4_quant::ACTIVATION_MAE_FOUR_SIX as u32,
            ];
            if super::config::BOUNDED_SWIGLU {
                expected.extend([
                    crate::bounded_swiglu::PRECISION_POLICY_VERSION,
                    crate::bounded_swiglu::ACTIVATION_INPUT_BITS,
                ]);
            }
            match result {
                Ok(actual) if actual == expected => true,
                other => {
                    eprintln!(
                        "MLP executable/PTX graph mismatch expected={expected:?} observed={other:?}"
                    );
                    false
                }
            }
        });
        if verified {
            Ok(())
        } else {
            Err(DriverError(1))
        }
    }

    pub fn bounded_forward(
        &self,
        stream: &CudaStream,
        paired: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        n: u32,
        h: u32,
    ) -> Result<(), DriverError> {
        self.verify_graph(stream)?;
        if paired.len() < (n * 2 * h) as usize || out.len() < (n * h) as usize {
            return Err(DriverError(1));
        }
        self.bounded
            .as_ref()
            .ok_or(DriverError(1))?
            .bounded_swiglu_forward_kernel(stream, linear_config(n * h, 256), paired, out, n, h)
    }

    pub fn bounded_backward(
        &self,
        stream: &CudaStream,
        paired: &mut DeviceBuffer<f32>,
        upstream: &DeviceBuffer<f32>,
        masks: &DeviceBuffer<u64>,
        n: u32,
        h: u32,
        routed: bool,
    ) -> Result<(), DriverError> {
        self.verify_graph(stream)?;
        if paired.len() < (n * 2 * h) as usize
            || upstream.len() < (n * h) as usize
            || (routed && masks.len() < n.div_ceil(128) as usize)
        {
            return Err(DriverError(1));
        }
        self.bounded
            .as_ref()
            .ok_or(DriverError(1))?
            .bounded_swiglu_backward_kernel(
                stream,
                linear_config(n * h, 256),
                paired,
                upstream,
                masks,
                n,
                h,
                routed,
            )
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
        self.verify_graph(args.stream)?;
        assert!(args.seq_len > 0 && args.seq_len % kernels::BLOCK_TOPK_TILE == 0);
        assert_eq!(args.token_count % args.seq_len, 0);
        let token_tiles = args.token_count / kernels::BLOCK_TOPK_TILE;
        let feature_tiles = args.feature_count / kernels::BLOCK_TOPK_TILE;
        assert!(token_tiles > 0 && token_tiles <= kernels::BLOCK_TOPK_TOKEN_TILES);
        assert_eq!(feature_tiles, kernels::BLOCK_TOPK_FEATURE_TILES);
        assert!(
            args.scores.len() >= (token_tiles * feature_tiles) as usize,
            "block-Top-K score scratch is too small"
        );
        assert!(
            args.masks.len() >= (token_tiles + feature_tiles * token_tiles.div_ceil(64)) as usize,
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
            args.seq_len,
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
