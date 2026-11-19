//! One H-Net stage between the outer encoder/decoder and the inner model.
use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};

use crate::{
    hnet_chunk::kernels as chunk,
    hnet_router::kernels as router,
    launch::{grid_x_config, linear_config},
    nvfp4::Nvfp4DeviceTensor,
};

const THREADS: u32 = 128;

#[derive(Clone, Copy)]
pub struct StageShape {
    pub batch: u32,
    pub outer_stride: u32,
    pub features: u32,
    pub router_width: u32,
    pub inner_alignment: u32,
}

#[derive(Clone, Copy)]
pub struct RatioObjective {
    pub target: f32,
    pub weight: f32,
}

pub struct HnetStage {
    chunk: chunk::LoadedModule,
    router: router::LoadedModule,
}

pub struct StagePlan<'a> {
    shape: StageShape,
    stream: &'a CudaStream,
    queries: &'a DeviceBuffer<u16>,
    keys: &'a DeviceBuffer<u16>,
    lengths: &'a DeviceBuffer<u32>,
    valid_chunks: DeviceBuffer<u32>,
    inner_stride: u32,
    probability: DeviceBuffer<u16>,
    row_to_chunk: DeviceBuffer<u32>,
    chunk_to_row: DeviceBuffer<u32>,
    ratio_loss: DeviceBuffer<f32>,
    ratio_gradient: DeviceBuffer<f32>,
}

pub struct SmoothedChunks<'plan, 'a> {
    plan: &'plan StagePlan<'a>,
    inner: &'plan DeviceBuffer<u16>,
    values: DeviceBuffer<u16>,
}

pub struct StageBackward {
    pub d_inner: DeviceBuffer<f32>,
    pub d_queries: DeviceBuffer<f32>,
    pub d_keys: DeviceBuffer<f32>,
}

pub struct WidthAdjoints {
    pub d_features: DeviceBuffer<f32>,
    pub d_padding: Option<DeviceBuffer<f32>>,
}

fn require<T>(stream: &CudaStream, buffer: &DeviceBuffer<T>, len: u32) -> Result<(), DriverError> {
    if buffer.len() != len as usize || !Arc::ptr_eq(stream.context(), buffer.context()) {
        return Err(DriverError(1));
    }
    Ok(())
}

fn area(rows: u32, cols: u32) -> Result<u32, DriverError> {
    rows.checked_mul(cols)
        .filter(|&size| size > 0)
        .ok_or(DriverError(1))
}

impl StagePlan<'_> {
    pub fn batch(&self) -> u32 {
        self.shape.batch
    }

    pub fn features(&self) -> u32 {
        self.shape.features
    }

    pub fn uses_stream(&self, stream: &CudaStream) -> bool {
        std::ptr::eq(stream, self.stream)
    }

    pub fn inner_stride(&self) -> u32 {
        self.inner_stride
    }

    pub fn inner_lengths(&self) -> &DeviceBuffer<u32> {
        &self.valid_chunks
    }

    pub fn ratio_loss(&self) -> &DeviceBuffer<f32> {
        &self.ratio_loss
    }

    pub fn into_ratio_loss(self) -> DeviceBuffer<f32> {
        self.ratio_loss
    }

    fn inner_values(&self) -> u32 {
        self.shape.batch * self.inner_stride * self.shape.features
    }

    fn outer_values(&self) -> u32 {
        self.shape.batch * self.shape.outer_stride * self.shape.features
    }

    fn residual_values(&self, width: u32) -> Result<u32, DriverError> {
        if width < self.shape.features {
            return Err(DriverError(1));
        }
        area(self.shape.batch * self.inner_stride, width)
    }
}

impl HnetStage {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            chunk: chunk::from_module(module.clone())?,
            router: router::from_module(module)?,
        })
    }

    /// Build causal routing on the device. Only the per-sequence counts are
    /// read on the host, so the inner launch can use the actual compressed
    /// capacity while the device retains lengths, probabilities and maps.
    pub fn route<'a>(
        &self,
        stream: &'a CudaStream,
        queries: &'a DeviceBuffer<u16>,
        keys: &'a DeviceBuffer<u16>,
        lengths: &'a DeviceBuffer<u32>,
        shape: StageShape,
        objective: RatioObjective,
    ) -> Result<StagePlan<'a>, DriverError> {
        let StageShape {
            batch,
            outer_stride,
            features,
            router_width,
            inner_alignment,
        } = shape;
        let outer_rows = area(batch, outer_stride)?;
        let route_values = area(outer_rows, router_width)?;
        let warp_work = area(outer_rows, 32)?;
        area(outer_rows, features)?;
        require(stream, queries, route_values)?;
        require(stream, keys, route_values)?;
        require(stream, lengths, batch)?;
        if !objective.target.is_finite()
            || objective.target <= 1.0
            || !objective.weight.is_finite()
            || objective.weight < 0.0
            || inner_alignment == 0
            || !outer_stride.is_multiple_of(inner_alignment)
        {
            return Err(DriverError(1));
        }

        let mut probability = DeviceBuffer::zeroed(stream, outer_rows as usize)?;
        let mut row_to_chunk = DeviceBuffer::zeroed(stream, outer_rows as usize)?;
        let mut chunk_to_row = DeviceBuffer::zeroed(stream, outer_rows as usize)?;
        let mut valid_chunks = DeviceBuffer::zeroed(stream, batch as usize)?;
        let mut ratio_loss = DeviceBuffer::zeroed(stream, 1)?;
        let mut ratio_gradient = DeviceBuffer::zeroed(stream, 1)?;
        self.router.hnet_route_cosine_kernel(
            stream,
            linear_config(warp_work, THREADS),
            queries,
            keys,
            lengths,
            &mut probability,
            batch,
            outer_stride,
            router_width,
        )?;
        self.chunk.hnet_causal_layout_kernel(
            stream,
            linear_config(batch, THREADS),
            &probability,
            lengths,
            &mut row_to_chunk,
            &mut chunk_to_row,
            &mut valid_chunks,
            batch,
            outer_stride,
        )?;
        let maximum = valid_chunks
            .to_host_vec(stream)?
            .into_iter()
            .max()
            .unwrap_or(0);
        if maximum == 0 || maximum > outer_stride {
            return Err(DriverError(1));
        }
        let inner_stride = maximum.next_multiple_of(inner_alignment);
        self.router.hnet_ratio_loss_kernel(
            stream,
            grid_x_config(1, 32),
            &probability,
            lengths,
            &valid_chunks,
            &mut ratio_loss,
            &mut ratio_gradient,
            batch,
            outer_stride,
            objective.target,
            objective.weight,
        )?;
        Ok(StagePlan {
            shape,
            stream,
            queries,
            keys,
            lengths,
            valid_chunks,
            inner_stride,
            probability,
            row_to_chunk,
            chunk_to_row,
            ratio_loss,
            ratio_gradient,
        })
    }

    pub fn gather(
        &self,
        plan: &StagePlan<'_>,
        encoded: &DeviceBuffer<u16>,
    ) -> Result<DeviceBuffer<u16>, DriverError> {
        let StageShape {
            batch,
            outer_stride,
            features,
            ..
        } = plan.shape;
        require(plan.stream, encoded, batch * outer_stride * features)?;
        let inner_values = plan.inner_values();
        let mut chunks = DeviceBuffer::zeroed(plan.stream, inner_values as usize)?;
        self.chunk.hnet_gather_chunk_kernel(
            plan.stream,
            linear_config(inner_values, THREADS),
            encoded,
            &plan.chunk_to_row,
            &mut chunks,
            batch,
            outer_stride,
            plan.inner_stride,
            features,
        )?;
        Ok(chunks)
    }

    pub fn write_residual(
        &self,
        plan: &StagePlan<'_>,
        represented: &DeviceBuffer<u16>,
        inner_width: u32,
        learned_padding: Option<&DeviceBuffer<u16>>,
        residual: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        let values = plan.residual_values(inner_width)?;
        require(plan.stream, represented, plan.inner_values())?;
        let padding = match learned_padding {
            Some(padding) => {
                require(plan.stream, padding, inner_width - plan.shape.features)?;
                padding
            }
            None if inner_width == plan.shape.features => represented,
            None => return Err(DriverError(1)),
        };
        if residual.len() < values as usize
            || !Arc::ptr_eq(plan.stream.context(), residual.context())
        {
            return Err(DriverError(1));
        }
        self.chunk.hnet_inner_to_residual_kernel(
            plan.stream,
            linear_config(values, THREADS),
            represented,
            padding,
            &plan.valid_chunks,
            residual,
            values,
            plan.inner_stride,
            plan.shape.features,
            inner_width,
        )
    }

    pub fn read_residual(
        &self,
        plan: &StagePlan<'_>,
        residual: &DeviceBuffer<f32>,
        inner_width: u32,
    ) -> Result<DeviceBuffer<u16>, DriverError> {
        let values = plan.inner_values();
        if residual.len() < plan.residual_values(inner_width)? as usize
            || !Arc::ptr_eq(plan.stream.context(), residual.context())
        {
            return Err(DriverError(1));
        }
        let mut represented = DeviceBuffer::zeroed(plan.stream, values as usize)?;
        self.chunk.hnet_residual_to_inner_kernel(
            plan.stream,
            linear_config(values, THREADS),
            residual,
            &plan.valid_chunks,
            &mut represented,
            values,
            plan.inner_stride,
            plan.shape.features,
            inner_width,
        )?;
        Ok(represented)
    }

    pub fn widen_output_adjoint(
        &self,
        plan: &StagePlan<'_>,
        d_represented: &DeviceBuffer<f32>,
        inner_width: u32,
    ) -> Result<DeviceBuffer<f32>, DriverError> {
        require(plan.stream, d_represented, plan.inner_values())?;
        let values = plan.residual_values(inner_width)?;
        let mut output = DeviceBuffer::zeroed(plan.stream, values as usize)?;
        self.chunk.hnet_widen_output_adjoint_kernel(
            plan.stream,
            linear_config(values, THREADS),
            d_represented,
            &plan.valid_chunks,
            &mut output,
            values,
            plan.inner_stride,
            plan.shape.features,
            inner_width,
        )?;
        Ok(output)
    }

    pub fn split_input_adjoint(
        &self,
        plan: &StagePlan<'_>,
        d_residual: &DeviceBuffer<f32>,
        inner_width: u32,
    ) -> Result<WidthAdjoints, DriverError> {
        if d_residual.len() < plan.residual_values(inner_width)? as usize
            || !Arc::ptr_eq(plan.stream.context(), d_residual.context())
        {
            return Err(DriverError(1));
        }
        let values = plan.inner_values();
        let mut d_features = DeviceBuffer::zeroed(plan.stream, values as usize)?;
        self.chunk.hnet_split_input_adjoint_kernel(
            plan.stream,
            linear_config(values, THREADS),
            d_residual,
            &plan.valid_chunks,
            &mut d_features,
            values,
            plan.inner_stride,
            plan.shape.features,
            inner_width,
        )?;
        let width = inner_width - plan.shape.features;
        let d_padding = if width == 0 {
            None
        } else {
            let mut gradient = DeviceBuffer::zeroed(plan.stream, width as usize)?;
            self.chunk.hnet_padding_adjoint_kernel(
                plan.stream,
                linear_config(width, THREADS),
                d_residual,
                &plan.valid_chunks,
                &mut gradient,
                plan.batch(),
                plan.inner_stride,
                plan.shape.features,
                inner_width,
            )?;
            Some(gradient)
        };
        Ok(WidthAdjoints {
            d_features,
            d_padding,
        })
    }

    pub fn smooth_chunks<'plan, 'a>(
        &self,
        plan: &'plan StagePlan<'a>,
        inner: &'plan DeviceBuffer<u16>,
    ) -> Result<SmoothedChunks<'plan, 'a>, DriverError> {
        let StageShape {
            batch,
            outer_stride,
            features,
            ..
        } = plan.shape;
        let stream = plan.stream;
        let inner_values = plan.inner_values();
        require(stream, inner, inner_values)?;
        let mut values = DeviceBuffer::zeroed(stream, inner_values as usize)?;
        self.chunk.hnet_smooth_chunk_kernel(
            stream,
            linear_config(batch * features, THREADS),
            inner,
            &plan.probability,
            &plan.chunk_to_row,
            &plan.valid_chunks,
            &mut values,
            batch,
            outer_stride,
            plan.inner_stride,
            features,
        )?;
        Ok(SmoothedChunks {
            plan,
            inner,
            values,
        })
    }

    /// Reverse the expansion and differentiate the router, including its
    /// ratio objective. The inner model's reverse consumes the returned
    /// d_inner, then scatter sends its input gradient back to the outer rows.
    pub fn backward(
        &self,
        chunks: &SmoothedChunks<'_, '_>,
        d_expanded: &DeviceBuffer<f32>,
    ) -> Result<StageBackward, DriverError> {
        let plan = chunks.plan;
        let StageShape {
            batch,
            outer_stride,
            features,
            router_width,
            ..
        } = plan.shape;
        let stream = plan.stream;
        let outer_rows = batch * outer_stride;
        require(stream, d_expanded, outer_rows * features)?;
        let mut d_inner = DeviceBuffer::zeroed(stream, plan.inner_values() as usize)?;
        let mut d_smoothed = DeviceBuffer::zeroed(stream, plan.inner_values() as usize)?;
        let mut d_probability = DeviceBuffer::zeroed(stream, outer_rows as usize)?;
        let mut d_queries = DeviceBuffer::zeroed(stream, (outer_rows * router_width) as usize)?;
        let mut d_keys = DeviceBuffer::zeroed(stream, d_queries.len())?;
        self.chunk.hnet_smooth_chunk_backward_kernel(
            stream,
            linear_config(batch * features, THREADS),
            d_expanded,
            &plan.probability,
            &plan.chunk_to_row,
            &plan.valid_chunks,
            plan.lengths,
            &mut d_smoothed,
            &mut d_inner,
            batch,
            outer_stride,
            plan.inner_stride,
            features,
        )?;
        let warp_grid = linear_config(outer_rows * 32, THREADS);
        self.chunk.hnet_lift_probability_backward_kernel(
            stream,
            warp_grid,
            d_expanded,
            chunks.inner,
            &chunks.values,
            &d_smoothed,
            &plan.probability,
            &plan.row_to_chunk,
            &plan.chunk_to_row,
            &mut d_probability,
            batch,
            outer_stride,
            plan.inner_stride,
            features,
        )?;
        self.router.hnet_route_cosine_backward_kernel(
            stream,
            warp_grid,
            plan.queries,
            plan.keys,
            plan.lengths,
            &d_probability,
            &plan.ratio_gradient,
            &mut d_queries,
            &mut d_keys,
            batch,
            outer_stride,
            router_width,
        )?;
        Ok(StageBackward {
            d_inner,
            d_queries,
            d_keys,
        })
    }

    /// Repeat the latest causal chunk directly into the decoder's residual sum.
    pub fn decoder_input(
        &self,
        chunks: &SmoothedChunks<'_, '_>,
        projected_skip: &DeviceBuffer<f32>,
        skip_bias: Nvfp4DeviceTensor<'_>,
    ) -> Result<DeviceBuffer<f32>, DriverError> {
        let plan = chunks.plan;
        let values = plan.outer_values();
        if !plan.features().is_multiple_of(16)
            || skip_bias.bytes.len() < plan.features() as usize / 2
            || skip_bias.scales.len() < crate::nvfp4::csf::encoded_len(1, plan.features() as usize)
            || skip_bias.global_scale.is_empty()
        {
            return Err(DriverError(1));
        }
        require(plan.stream, projected_skip, values)?;
        let mut input = DeviceBuffer::zeroed(plan.stream, values as usize)?;
        self.chunk.hnet_decoder_input_kernel(
            plan.stream,
            linear_config(values, THREADS),
            &chunks.values,
            &plan.row_to_chunk,
            projected_skip,
            skip_bias.bytes,
            skip_bias.scales,
            skip_bias.global_scale,
            &mut input,
            values,
            plan.shape.outer_stride,
            plan.inner_stride,
            plan.shape.features,
        )?;
        Ok(input)
    }

    /// Both inputs of the sum receive this masked gradient.
    pub fn mask_decoder_input_adjoint(
        &self,
        plan: &StagePlan<'_>,
        gradient: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        let values = plan.outer_values();
        require(plan.stream, gradient, values)?;
        self.chunk.hnet_decoder_input_adjoint_kernel(
            plan.stream,
            linear_config(values, THREADS),
            plan.lengths,
            gradient,
            values,
            plan.shape.outer_stride,
            plan.shape.features,
        )
    }

    pub fn scatter(
        &self,
        plan: &StagePlan<'_>,
        d_chunks: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, DriverError> {
        let StageShape {
            batch,
            outer_stride,
            features,
            ..
        } = plan.shape;
        let stream = plan.stream;
        require(stream, d_chunks, plan.inner_values())?;
        let outer_values = batch * outer_stride * features;
        let mut d_encoder = DeviceBuffer::zeroed(stream, outer_values as usize)?;
        self.chunk.hnet_gather_chunk_backward_kernel(
            stream,
            linear_config(outer_values, THREADS),
            d_chunks,
            &plan.row_to_chunk,
            &plan.chunk_to_row,
            &mut d_encoder,
            batch,
            outer_stride,
            plan.inner_stride,
            features,
        )?;
        Ok(d_encoder)
    }
}
