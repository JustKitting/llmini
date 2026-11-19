use cuda_core::DeviceBuffer;
use gpt2_nvfp4::RowwiseNvfp4Buffers;
use rust_kernels_cuda::{
    f16_tc_matmul::F16ConvertArgs,
    hnet_stage::{StageBackward, StageShape},
    linear_backward::{
        LinearBackwardInputTranspose, LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratchBuffers,
        LinearBackwardWeightTranspose,
    },
    lm_head::LmHeadTmaArgs,
    nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors,
    residual::ResidualGradAccumulateArgs,
};

use crate::{AppResult, training::runtime::Runtime, upload::UploadedNvfp4};

#[derive(Clone, Copy)]
pub(super) struct StageWeights<'a> {
    pub query: &'a UploadedNvfp4,
    pub key: &'a UploadedNvfp4,
    pub residual: &'a UploadedNvfp4,
}

pub(super) struct StageProjections<'a> {
    pub encoded: DeviceBuffer<u16>,
    pub query: DeviceBuffer<u16>,
    pub key: DeviceBuffer<u16>,
    pub residual: DeviceBuffer<f32>,
    saved: RowwiseNvfp4Buffers,
    weights: StageWeights<'a>,
}

pub(super) struct StageParameterAdjoints {
    pub query: DeviceBuffer<f32>,
    pub key: DeviceBuffer<f32>,
    pub residual: DeviceBuffer<f32>,
}

pub(super) struct StageProjectionScratch {
    rows: u32,
    width: u32,
    descriptors: TmaNvfp4DeviceScaleDescriptors,
    linear: LinearBackwardMsEdenScratchBuffers,
    intermediate: DeviceBuffer<f32>,
    row_amax: DeviceBuffer<f32>,
}

impl StageProjectionScratch {
    pub fn new(runtime: &Runtime, shape: StageShape) -> AppResult<Self> {
        let rows = shape
            .batch
            .checked_mul(shape.outer_stride)
            .ok_or("outer rows overflow")?;
        let width = shape.features;
        if rows == 0
            || width == 0
            || !rows.is_multiple_of(128)
            || !width.is_multiple_of(128)
            || shape.router_width != width
        {
            return Err(
                "H-Net router and residual projections need aligned outer rows and width".into(),
            );
        }
        let values = rows
            .checked_mul(width)
            .ok_or("outer projection size overflows")? as usize;
        width
            .checked_mul(width)
            .ok_or("outer parameter size overflows")?;
        let stream = &runtime.stream;
        Ok(Self {
            rows,
            width,
            descriptors: TmaNvfp4DeviceScaleDescriptors::new(stream)?,
            linear: LinearBackwardMsEdenScratchBuffers::new(
                stream,
                rows as usize,
                width as usize,
                width as usize,
            )?,
            intermediate: DeviceBuffer::zeroed(stream, values)?,
            row_amax: DeviceBuffer::zeroed(stream, rows as usize)?,
        })
    }

    pub fn forward<'a>(
        &mut self,
        runtime: &Runtime,
        weights: StageWeights<'a>,
        encoded: &DeviceBuffer<f32>,
    ) -> AppResult<StageProjections<'a>> {
        let stream = &runtime.stream;
        let values = self.rows * self.width;
        if encoded.len() != values as usize
            || [weights.query, weights.key, weights.residual]
                .iter()
                .any(|weight| weight.scale_k != self.width || weight.scale_rows != self.width)
        {
            return Err("H-Net projection input or weight has the wrong dimensions".into());
        }
        let mut saved = RowwiseNvfp4Buffers::new(stream, values as usize, self.rows as usize)?;
        saved.scratch().quantize_row_amax(
            &runtime.quant,
            stream,
            encoded,
            &mut self.row_amax,
            self.rows,
            self.width,
        )?;
        let mut represented = DeviceBuffer::zeroed(stream, values as usize)?;
        let mut query = DeviceBuffer::zeroed(stream, values as usize)?;
        let mut key = DeviceBuffer::zeroed(stream, values as usize)?;
        let mut residual = DeviceBuffer::zeroed(stream, values as usize)?;
        runtime.f16_tc_matmul.fp32_to_f16(F16ConvertArgs {
            stream,
            src: encoded,
            dst: &mut represented,
            element_count: values,
        })?;
        for (weight, output) in [(weights.query, &mut query), (weights.key, &mut key)] {
            runtime.lm_head.logits_tma(LmHeadTmaArgs {
                stream,
                tma: &runtime.tma_gemm,
                descriptors: &mut self.descriptors,
                input: saved.rowwise(),
                weight: weight.device(),
                logits: &mut self.intermediate,
                token_count: self.rows,
                input_dim: self.width,
                vocab_size: self.width,
            })?;
            runtime.f16_tc_matmul.fp32_to_f16(F16ConvertArgs {
                stream,
                src: &self.intermediate,
                dst: output,
                element_count: values,
            })?;
        }
        runtime.lm_head.logits_tma(LmHeadTmaArgs {
            stream,
            tma: &runtime.tma_gemm,
            descriptors: &mut self.descriptors,
            input: saved.rowwise(),
            weight: weights.residual.device(),
            logits: &mut residual,
            token_count: self.rows,
            input_dim: self.width,
            vocab_size: self.width,
        })?;
        Ok(StageProjections {
            encoded: represented,
            query,
            key,
            residual,
            saved,
            weights,
        })
    }

    /// Add the router and skip projection paths to the scattered encoder adjoint.
    pub fn backward(
        &mut self,
        runtime: &Runtime,
        tape: StageProjections<'_>,
        route: &StageBackward,
        d_skip: &DeviceBuffer<f32>,
        d_encoded: &mut DeviceBuffer<f32>,
        seeds: [u32; 2],
    ) -> AppResult<StageParameterAdjoints> {
        let stream = &runtime.stream;
        let values = self.rows * self.width;
        if [d_skip, d_encoded, &route.d_queries, &route.d_keys]
            .iter()
            .any(|buffer| buffer.len() != values as usize)
            || tape.encoded.len() != values as usize
        {
            return Err("H-Net projection adjoints have the wrong dimensions".into());
        }
        let weight_values = (self.width * self.width) as usize;
        let mut adjoints = StageParameterAdjoints {
            query: DeviceBuffer::zeroed(stream, weight_values)?,
            key: DeviceBuffer::zeroed(stream, weight_values)?,
            residual: DeviceBuffer::zeroed(stream, weight_values)?,
        };
        let paths = [
            (tape.weights.query, &route.d_queries, &mut adjoints.query),
            (tape.weights.key, &route.d_keys, &mut adjoints.key),
            (tape.weights.residual, d_skip, &mut adjoints.residual),
        ];
        for (index, (weight, output_gradient, weight_gradient)) in paths.into_iter().enumerate() {
            runtime.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
                stream,
                quant_module: &runtime.quant,
                e: output_gradient,
                weight_t: LinearBackwardWeightTranspose::Nvfp4(weight.device()),
                input_t: LinearBackwardInputTranspose::RowwiseNvfp4(tape.saved.rowwise()),
                scratch: self.linear.as_args(),
                dinput: &mut self.intermediate,
                dweight: weight_gradient,
                dbias: None,
                token_count: self.rows,
                input_dim: self.width,
                output_dim: self.width,
                sign_seed: seeds[0].wrapping_add(index as u32),
                scale_seed: seeds[1].wrapping_add(index as u32),
                precomputed_e_amax_chunks: None,
            })?;
            runtime
                .residual
                .grad_accumulate(ResidualGradAccumulateArgs {
                    stream,
                    branch: &self.intermediate,
                    out: d_encoded,
                    len: values,
                })?;
        }
        Ok(adjoints)
    }
}
