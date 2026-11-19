use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{GPT2_MLP_ROUTE_TILE, Gpt2Rng, RowwiseNvfp4Buffers};
use rust_kernels_cuda::{
    hnet_byte_head::{BYTE_LOGIT_STRIDE, ByteHeadModule, ByteLossArgs, ByteProjection},
    hnet_stage::{HnetStage, RatioObjective, StageShape},
    linear_backward::{
        LinearBackwardInputTranspose, LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratchBuffers,
        LinearBackwardWeightTranspose,
    },
    nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors,
};

use super::{
    Trainer,
    outer::{OuterParameterAdjoints, OuterScratch, OuterTape, OuterWeights},
    projection::{StageParameterAdjoints, StageProjectionScratch, StageWeights},
};
use crate::{AppResult, training::runtime::Runtime, upload::UploadedNvfp4};

pub(super) struct HierarchyBatch<'a> {
    pub embeddings: &'a DeviceBuffer<f32>,
    pub valid_lengths: &'a DeviceBuffer<u32>,
    pub targets: &'a DeviceBuffer<u32>,
    pub scored_targets: u32,
}

pub(super) struct HierarchyWeights<'a> {
    pub encoder: &'a [OuterWeights<'a>],
    pub decoder: &'a [OuterWeights<'a>],
    pub stage: StageWeights<'a>,
    pub width_padding: Option<&'a DeviceBuffer<u16>>,
    pub byte_readout: &'a UploadedNvfp4,
}

pub(super) struct HierarchyAdjoints {
    pub backbone_rows: u32,
    pub byte_losses: DeviceBuffer<f32>,
    pub ratio_loss: DeviceBuffer<f32>,
    pub embeddings: DeviceBuffer<f32>,
    pub encoder: Vec<OuterParameterAdjoints>,
    pub decoder: Vec<OuterParameterAdjoints>,
    pub stage: StageParameterAdjoints,
    pub width_padding: Option<DeviceBuffer<f32>>,
    pub byte_readout: DeviceBuffer<f32>,
}

pub(super) struct HierarchyScratch {
    shape: StageShape,
    stage: HnetStage,
    stage_projection: StageProjectionScratch,
    outer: OuterScratch,
    byte: ByteReadoutScratch,
}

struct ByteReadoutScratch {
    byte_head: ByteHeadModule,
    byte_descriptors: TmaNvfp4DeviceScaleDescriptors,
    byte_backward: LinearBackwardMsEdenScratchBuffers,
    row_amax: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    d_logits: DeviceBuffer<f32>,
}

impl HierarchyScratch {
    pub fn new(runtime: &Runtime, batch: u32, stride: u32, width: u32) -> AppResult<Self> {
        let shape = StageShape {
            batch,
            outer_stride: stride,
            features: width,
            router_width: width,
            inner_alignment: GPT2_MLP_ROUTE_TILE as u32,
        };
        let rows = batch
            .checked_mul(stride)
            .ok_or("H-Net byte rows overflow")?;
        let logit_values = rows
            .checked_mul(BYTE_LOGIT_STRIDE)
            .ok_or("H-Net byte logits overflow")? as usize;
        let stream = &runtime.stream;
        Ok(Self {
            shape,
            stage: runtime.hnet_stage_module()?,
            stage_projection: StageProjectionScratch::new(runtime, shape)?,
            outer: OuterScratch::new(runtime, batch, stride, width)?,
            byte: ByteReadoutScratch {
                byte_head: runtime.hnet_byte_head_module()?,
                byte_descriptors: TmaNvfp4DeviceScaleDescriptors::new(stream)?,
                byte_backward: LinearBackwardMsEdenScratchBuffers::new(
                    stream,
                    rows as usize,
                    width as usize,
                    BYTE_LOGIT_STRIDE as usize,
                )?,
                row_amax: DeviceBuffer::zeroed(stream, rows as usize)?,
                logits: DeviceBuffer::zeroed(stream, logit_values)?,
                d_logits: DeviceBuffer::zeroed(stream, logit_values)?,
            },
        })
    }
}

impl ByteReadoutScratch {
    fn readout(
        &mut self,
        runtime: &Runtime,
        decoded: &DeviceBuffer<f32>,
        batch: &HierarchyBatch<'_>,
        weights: &UploadedNvfp4,
        width: u32,
        rng: &mut Gpt2Rng,
    ) -> AppResult<(DeviceBuffer<f32>, DeviceBuffer<f32>, DeviceBuffer<f32>)> {
        let rows = self.row_amax.len() as u32;
        if weights.scale_k != width || weights.scale_rows != BYTE_LOGIT_STRIDE {
            return Err("H-Net byte readout has the wrong dimensions".into());
        }
        let stream = &runtime.stream;
        let mut saved = RowwiseNvfp4Buffers::new(stream, decoded.len(), rows as usize)?;
        saved.scratch().quantize_row_amax(
            &runtime.quant,
            stream,
            decoded,
            &mut self.row_amax,
            rows,
            width,
        )?;
        self.byte_head.project(ByteProjection {
            stream,
            tma: &runtime.tma_gemm,
            descriptors: &mut self.byte_descriptors,
            decoder_values: saved.rowwise(),
            weights: weights.device(),
            logits: &mut self.logits,
            rows,
            width,
        })?;
        let mut losses = DeviceBuffer::zeroed(stream, rows as usize)?;
        self.byte_head.loss(ByteLossArgs {
            stream,
            logits: &self.logits,
            targets: batch.targets,
            losses: &mut losses,
            dlogits: &mut self.d_logits,
            row_amax: &mut self.row_amax,
            rows,
            valid_targets: batch.scored_targets,
        })?;
        let mut d_weight = DeviceBuffer::zeroed(stream, (width * BYTE_LOGIT_STRIDE) as usize)?;
        let mut d_decoder = DeviceBuffer::zeroed(stream, decoded.len())?;
        runtime.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
            stream,
            quant_module: &runtime.quant,
            e: &self.d_logits,
            weight_t: LinearBackwardWeightTranspose::Nvfp4(weights.device()),
            input_t: LinearBackwardInputTranspose::RowwiseNvfp4(saved.rowwise()),
            scratch: self.byte_backward.as_args(),
            dinput: &mut d_decoder,
            dweight: &mut d_weight,
            dbias: None,
            token_count: rows,
            input_dim: width,
            output_dim: BYTE_LOGIT_STRIDE,
            sign_seed: rng.next_u32(),
            scale_seed: rng.next_u32(),
            precomputed_e_amax_chunks: None,
        })?;
        Ok((losses, d_decoder, d_weight))
    }
}

impl Trainer {
    /// Differentiate the native byte hierarchy. The trunk gradients stay in
    /// the trainer's existing optimizer buffers, and the outer adjoints are
    /// returned in the same order as the parameter slices.
    pub(super) fn backward_hnet(
        &mut self,
        scratch: &mut HierarchyScratch,
        batch: HierarchyBatch<'_>,
        weights: HierarchyWeights<'_>,
        ratio: RatioObjective,
        rng: &mut Gpt2Rng,
    ) -> AppResult<HierarchyAdjoints> {
        let (encoded, encoder) = forward_stack(
            &self.runtime,
            &mut scratch.outer,
            weights.encoder,
            batch.embeddings,
        )?;
        let projections =
            scratch
                .stage_projection
                .forward(&self.runtime, weights.stage, &encoded)?;
        let stream = self.runtime.stream.clone();
        let stage = &scratch.stage;
        let plan = stage.route(
            &stream,
            &projections.query,
            &projections.key,
            batch.valid_lengths,
            scratch.shape,
            ratio,
        )?;
        let backbone_rows = plan.batch() * plan.inner_stride();
        let inner = self.forward_routed_backbone(
            stage,
            &plan,
            &projections.encoded,
            weights.width_padding,
        )?;
        let expanded = stage.expand(&plan, inner.output())?;
        let decoder_input = stage.decoder_input(&expanded, &projections.residual)?;
        let (decoded, decoder) = forward_stack(
            &inner.trainer.runtime,
            &mut scratch.outer,
            weights.decoder,
            &decoder_input,
        )?;
        // The decoder tape is independent of the byte loss and projection scratch.
        let (byte_losses, d_decoded, byte_readout) = scratch.byte.readout(
            &inner.trainer.runtime,
            &decoded,
            &batch,
            weights.byte_readout,
            scratch.shape.features,
            rng,
        )?;
        let (mut d_skip, decoder) = backward_stack(
            &inner.trainer.runtime,
            &mut scratch.outer,
            decoder,
            d_decoded,
            rng,
        )?;
        stage.mask_decoder_input_adjoint(&plan, &mut d_skip)?;
        let route_adjoint = stage.backward(&expanded, &d_skip)?;
        drop(expanded);
        let inner_adjoint = inner.backward(&route_adjoint.d_inner)?;
        let mut d_encoded = stage.scatter(&plan, &inner_adjoint.d_features)?;
        let ratio_loss = plan.into_ratio_loss();
        let stage = scratch.stage_projection.backward(
            &self.runtime,
            projections,
            &route_adjoint,
            &d_skip,
            &mut d_encoded,
            [rng.next_u32(), rng.next_u32()],
        )?;
        let (embeddings, encoder) =
            backward_stack(&self.runtime, &mut scratch.outer, encoder, d_encoded, rng)?;
        Ok(HierarchyAdjoints {
            backbone_rows,
            byte_losses,
            ratio_loss,
            embeddings,
            encoder,
            decoder,
            stage,
            width_padding: inner_adjoint.d_padding,
            byte_readout,
        })
    }
}

fn forward_stack<'a>(
    runtime: &Runtime,
    scratch: &mut OuterScratch,
    weights: &[OuterWeights<'a>],
    input: &DeviceBuffer<f32>,
) -> AppResult<(DeviceBuffer<f32>, Vec<OuterTape<'a>>)> {
    let (first, remaining) = weights
        .split_first()
        .ok_or("H-Net outer stack has no blocks")?;
    let (mut values, first_tape) = scratch.forward(runtime, *first, input)?;
    let mut tapes = Vec::with_capacity(weights.len());
    tapes.push(first_tape);
    for &weight in remaining {
        let (result, tape) = scratch.forward(runtime, weight, &values)?;
        values = result;
        tapes.push(tape);
    }
    Ok((values, tapes))
}

fn backward_stack(
    runtime: &Runtime,
    scratch: &mut OuterScratch,
    tapes: Vec<OuterTape<'_>>,
    mut gradient: DeviceBuffer<f32>,
    rng: &mut Gpt2Rng,
) -> AppResult<(DeviceBuffer<f32>, Vec<OuterParameterAdjoints>)> {
    let mut parameters = Vec::with_capacity(tapes.len());
    for tape in tapes.into_iter().rev() {
        let (input, adjoints) =
            tape.backward(runtime, scratch, &gradient, rng.next_u32(), rng.next_u32())?;
        gradient = input;
        parameters.push(adjoints);
    }
    parameters.reverse();
    Ok((gradient, parameters))
}
