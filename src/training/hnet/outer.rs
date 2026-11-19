use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionCoreScratchBuffers, GPT2_LAYER_NORM_EPSILON, KDA_CHUNK_SIZE, RowwiseNvfp4Buffers,
};
use rust_kernels_cuda::{
    attention::{CausalAttentionBackwardTcArgs, CausalAttentionTcArgs},
    layer_norm::GptLayerNormSaveXhatF16Args,
    layer_norm_backward::{LayerNormBackwardInputAddArgs, LayerNormBackwardParamArgs},
    linear_backward::{
        LinearBackwardInputTranspose, LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratchBuffers,
        LinearBackwardWeightTranspose,
    },
    lm_head::LmHeadTmaArgs,
    nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors,
    residual::ResidualGradAccumulateArgs,
};

use crate::{AppResult, training::runtime::Runtime, upload::UploadedNvfp4};

const HEAD_DIM: u32 = KDA_CHUNK_SIZE as u32;
const PROJECTION_TILE: u32 = 128;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape {
    batch: u32,
    sequence: u32,
    rows: u32,
    width: u32,
    heads: u32,
    qkv: u32,
}

/// Shared launch storage for the encoder and decoder blocks at one level.
pub(super) struct OuterScratch {
    shape: Shape,
    attention: AttentionCoreScratchBuffers,
    projection: TmaNvfp4DeviceScaleDescriptors,
    qkv_backward: LinearBackwardMsEdenScratchBuffers,
    output_backward: LinearBackwardMsEdenScratchBuffers,
    features: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    d_qkv: DeviceBuffer<f32>,
    row_amax: DeviceBuffer<f32>,
    qkv_amax: DeviceBuffer<f32>,
    log_sum_exp: DeviceBuffer<f32>,
    unused_qk_scale_grad: DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
pub(super) struct OuterWeights<'a> {
    pub qkv: &'a UploadedNvfp4,
    pub output: &'a UploadedNvfp4,
    pub norm_scale: &'a UploadedNvfp4,
    pub norm_bias: &'a UploadedNvfp4,
}

pub(super) struct OuterTape<'a> {
    shape: Shape,
    weights: OuterWeights<'a>,
    normalized: RowwiseNvfp4Buffers,
    mixed: RowwiseNvfp4Buffers,
    xhat: DeviceBuffer<u16>,
    mean: DeviceBuffer<f32>,
    inv_std: DeviceBuffer<f32>,
    qkv: DeviceBuffer<u16>,
    chunk_states: DeviceBuffer<u16>,
}

pub(super) struct OuterAdjoints {
    pub input: DeviceBuffer<f32>,
    pub qkv: DeviceBuffer<f32>,
    pub output: DeviceBuffer<f32>,
    pub norm_scale: DeviceBuffer<f32>,
    pub norm_bias: DeviceBuffer<f32>,
}

impl OuterScratch {
    pub fn new(runtime: &Runtime, batch: u32, sequence: u32, width: u32) -> AppResult<Self> {
        let rows = batch.checked_mul(sequence).ok_or("outer batch overflows")?;
        if rows == 0
            || !rows.is_multiple_of(PROJECTION_TILE)
            || !width.is_multiple_of(PROJECTION_TILE)
            || width == 0
            || !sequence.is_multiple_of(HEAD_DIM)
        {
            return Err(
                "H-Net outer blocks require full KDA chunks and aligned NVFP4 projections".into(),
            );
        }
        let heads = width / HEAD_DIM;
        let qkv = width
            .checked_mul(4)
            .and_then(|value| value.checked_add(heads))
            .and_then(|value| value.checked_next_multiple_of(PROJECTION_TILE))
            .ok_or("outer projection width overflows")?;
        let qkv_values = rows
            .checked_mul(qkv)
            .ok_or("outer projection size overflows")? as usize;
        width
            .checked_mul(qkv)
            .ok_or("outer parameter size overflows")?;
        let attention_rows = (rows * heads) as usize;
        let stream = &runtime.stream;
        Ok(Self {
            shape: Shape {
                batch,
                sequence,
                rows,
                width,
                heads,
                qkv,
            },
            attention: AttentionCoreScratchBuffers::for_kda(stream, batch, sequence, heads)?,
            projection: TmaNvfp4DeviceScaleDescriptors::new(stream)?,
            qkv_backward: LinearBackwardMsEdenScratchBuffers::new(
                stream,
                rows as usize,
                width as usize,
                qkv as usize,
            )?,
            output_backward: LinearBackwardMsEdenScratchBuffers::new(
                stream,
                rows as usize,
                width as usize,
                width as usize,
            )?,
            features: DeviceBuffer::zeroed(stream, (rows * width) as usize)?,
            qkv: DeviceBuffer::zeroed(stream, qkv_values)?,
            // The recurrence writes active Q/K/V/G/beta columns. Keep the
            // unused columns at zero for the projection's weight adjoint.
            d_qkv: DeviceBuffer::zeroed(stream, qkv_values)?,
            row_amax: DeviceBuffer::zeroed(stream, rows as usize)?,
            qkv_amax: DeviceBuffer::zeroed(stream, attention_rows)?,
            log_sum_exp: DeviceBuffer::zeroed(stream, attention_rows)?,
            unused_qk_scale_grad: DeviceBuffer::zeroed(stream, heads as usize)?,
        })
    }

    pub fn forward<'a>(
        &mut self,
        runtime: &Runtime,
        weights: OuterWeights<'a>,
        input: &DeviceBuffer<f32>,
    ) -> AppResult<(DeviceBuffer<f32>, OuterTape<'a>)> {
        let shape = self.shape;
        let stream = &runtime.stream;
        let features = (shape.rows * shape.width) as usize;
        if input.len() != features {
            return Err("H-Net outer input does not match the block shape".into());
        }
        for (weight, rows) in [
            (weights.qkv, shape.qkv),
            (weights.output, shape.width),
            (weights.norm_scale, 1),
            (weights.norm_bias, 1),
        ] {
            if weight.scale_k != shape.width || weight.scale_rows != rows {
                return Err("H-Net outer parameter does not match the block shape".into());
            }
        }
        let mut tape = OuterTape {
            shape,
            weights,
            normalized: RowwiseNvfp4Buffers::new(stream, features, shape.rows as usize)?,
            mixed: RowwiseNvfp4Buffers::new(stream, features, shape.rows as usize)?,
            xhat: DeviceBuffer::zeroed(stream, features)?,
            mean: DeviceBuffer::zeroed(stream, shape.rows as usize)?,
            inv_std: DeviceBuffer::zeroed(stream, shape.rows as usize)?,
            qkv: DeviceBuffer::zeroed(stream, (shape.rows * shape.qkv) as usize)?,
            chunk_states: DeviceBuffer::zeroed(stream, features)?,
        };
        runtime
            .layer_norm
            .gpt_layer_norm_save_xhat_f16(GptLayerNormSaveXhatF16Args {
                stream,
                residual: input,
                weight: weights.norm_scale.device(),
                bias: weights.norm_bias.device(),
                normalized: &mut self.features,
                normalized_amax: &mut self.row_amax,
                xhat_f16: &mut tape.xhat,
                mean: &mut tape.mean,
                inv_std: &mut tape.inv_std,
                row_count: shape.rows,
                embedding_dim: shape.width,
                epsilon: GPT2_LAYER_NORM_EPSILON,
                output_scale: 1.0,
            })?;
        tape.normalized.scratch().quantize_precomputed_amax(
            &runtime.quant,
            stream,
            &self.features,
            &self.row_amax,
            shape.rows,
            shape.width,
        )?;
        runtime.lm_head.logits_tma(LmHeadTmaArgs {
            stream,
            tma: &runtime.tma_gemm,
            descriptors: &mut self.projection,
            input: tape.normalized.rowwise(),
            weight: weights.qkv.device(),
            logits: &mut self.qkv,
            token_count: shape.rows,
            input_dim: shape.width,
            vocab_size: shape.qkv,
        })?;
        runtime.attention.kda_attention_tc(CausalAttentionTcArgs {
            stream,
            tc_module: &runtime.f16_tc_matmul,
            qkv: &self.qkv,
            qk_scale: weights.norm_scale.device(),
            out: &mut self.features,
            qkv_f16: Some(&mut tape.qkv),
            attention_out_f16: Some(&mut tape.chunk_states),
            forward_probs_f16: None,
            kda_v_new: None,
            kda_akk_inv: None,
            kda_w: None,
            kda_aqk: None,
            log_sum_exp: &mut self.log_sum_exp,
            scratch: self.attention.forward_tc(),
            row_count: shape.rows,
            seq_len: shape.sequence,
            batch_size: shape.batch,
            embedding_dim: shape.width,
            qkv_dim: shape.qkv,
            head_count: shape.heads,
            head_dim: HEAD_DIM,
            attention_window: shape.sequence,
            partial_key_offset: false,
            selective_attention: false,
            stable_mask_gamma: 0.0,
        })?;
        tape.mixed.scratch().quantize_row_amax(
            &runtime.quant,
            stream,
            &self.features,
            &mut self.row_amax,
            shape.rows,
            shape.width,
        )?;
        let mut output = DeviceBuffer::zeroed(stream, features)?;
        runtime.lm_head.logits_tma(LmHeadTmaArgs {
            stream,
            tma: &runtime.tma_gemm,
            descriptors: &mut self.projection,
            input: tape.mixed.rowwise(),
            weight: weights.output.device(),
            logits: &mut output,
            token_count: shape.rows,
            input_dim: shape.width,
            vocab_size: shape.width,
        })?;
        runtime
            .residual
            .grad_accumulate(ResidualGradAccumulateArgs {
                stream,
                branch: input,
                out: &mut output,
                len: features as u32,
            })?;
        Ok((output, tape))
    }
}

impl OuterTape<'_> {
    pub fn backward(
        self,
        runtime: &Runtime,
        scratch: &mut OuterScratch,
        gradient: &DeviceBuffer<f32>,
        sign_seed: u32,
        scale_seed: u32,
    ) -> AppResult<OuterAdjoints> {
        let shape = self.shape;
        let features = (shape.rows * shape.width) as usize;
        if scratch.shape != shape || gradient.len() != features {
            return Err("H-Net outer gradient does not match the forward block shape".into());
        }
        let stream = &runtime.stream;
        let width = shape.width as usize;
        let weights = self.weights;
        let mut adjoints = OuterAdjoints {
            input: DeviceBuffer::zeroed(stream, features)?,
            qkv: DeviceBuffer::zeroed(stream, width * shape.qkv as usize)?,
            output: DeviceBuffer::zeroed(stream, width * width)?,
            norm_scale: DeviceBuffer::zeroed(stream, width)?,
            norm_bias: DeviceBuffer::zeroed(stream, width)?,
        };
        runtime.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
            stream,
            quant_module: &runtime.quant,
            e: gradient,
            weight_t: LinearBackwardWeightTranspose::Nvfp4(weights.output.device()),
            input_t: LinearBackwardInputTranspose::RowwiseNvfp4(self.mixed.rowwise()),
            scratch: scratch.output_backward.as_args(),
            dinput: &mut scratch.features,
            dweight: &mut adjoints.output,
            dbias: None,
            token_count: shape.rows,
            input_dim: shape.width,
            output_dim: shape.width,
            sign_seed,
            scale_seed,
            precomputed_e_amax_chunks: None,
        })?;
        let attention = scratch.attention.args();
        runtime
            .attention
            .kda_attention_backward_tc(CausalAttentionBackwardTcArgs {
                stream,
                tc_module: &runtime.f16_tc_matmul,
                reuse_forward_probs: false,
                forward_probs_f16: None,
                qkv: &self.qkv,
                attention_out: &self.chunk_states,
                kda_v_new: None,
                kda_akk_inv: None,
                kda_w: None,
                kda_aqk: None,
                d_out: &scratch.features,
                qk_scale: weights.norm_scale.device(),
                log_sum_exp: &scratch.log_sum_exp,
                softmax_d: attention.softmax_d,
                qk_norm_max: attention.qk_norm_max,
                d_qkv: &mut scratch.d_qkv,
                d_qkv_chunk_amax: &mut scratch.qkv_amax,
                d_qk_scale: &mut scratch.unused_qk_scale_grad,
                accumulate_value_grad: false,
                scratch: attention.tc,
                row_count: shape.rows,
                seq_len: shape.sequence,
                batch_size: shape.batch,
                embedding_dim: shape.width,
                qkv_dim: shape.qkv,
                head_count: shape.heads,
                head_dim: HEAD_DIM,
                attention_window: shape.sequence,
                partial_key_offset: false,
                selective_attention: false,
                qk_norm_offset: 0,
                backward_mask_seed: 0,
                backward_tile_budget: 0.0,
            })?;
        runtime.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
            stream,
            quant_module: &runtime.quant,
            e: &scratch.d_qkv,
            weight_t: LinearBackwardWeightTranspose::Nvfp4(weights.qkv.device()),
            input_t: LinearBackwardInputTranspose::RowwiseNvfp4(self.normalized.rowwise()),
            scratch: scratch.qkv_backward.as_args(),
            dinput: &mut scratch.features,
            dweight: &mut adjoints.qkv,
            dbias: None,
            token_count: shape.rows,
            input_dim: shape.width,
            output_dim: shape.qkv,
            sign_seed: sign_seed.wrapping_add(1),
            scale_seed: scale_seed.wrapping_add(1),
            precomputed_e_amax_chunks: None,
        })?;
        runtime
            .layer_norm_backward
            .backward_params_xhat_f16(LayerNormBackwardParamArgs {
                stream,
                residual: &self.xhat,
                d_normalized: &scratch.features,
                mean: &self.mean,
                inv_std: &self.inv_std,
                d_weight: &mut adjoints.norm_scale,
                d_bias: &mut adjoints.norm_bias,
                output_scale: 1.0,
                row_count: shape.rows,
                embedding_dim: shape.width,
            })?;
        runtime
            .layer_norm_backward
            .backward_input_add_xhat_f16(LayerNormBackwardInputAddArgs {
                stream,
                residual: &self.xhat,
                d_normalized: &scratch.features,
                mean: &self.mean,
                inv_std: &self.inv_std,
                weight: weights.norm_scale.device(),
                direct: gradient,
                d_residual: &mut adjoints.input,
                output_scale: 1.0,
                row_count: shape.rows,
                embedding_dim: shape.width,
            })?;
        Ok(adjoints)
    }
}
