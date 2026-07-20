use gpt2_nvfp4::{GPT2_VOCAB_SIZE, Gpt2ForwardArgs, TokenEmbeddingArgs};

use super::next_latent::{
    NextLatForwardArgs, forward as next_latent_forward, loss_weight as next_latent_loss_weight,
};
use super::{TokenBatch, TrainStats, Trainer};
use crate::AppResult;

impl Trainer {
    pub fn forward_step(&mut self, batch: &TokenBatch) -> AppResult<TrainStats> {
        let stream = self.runtime.stream.as_ref();
        let buffers = &mut self.buffers;
        let uploaded = &self.uploaded;

        let hidden = self.model.forward(Gpt2ForwardArgs {
            embeddings: TokenEmbeddingArgs {
                module: &self.runtime.embedding,
                stream,
                tokens: &batch.tokens,
                token_embedding: uploaded.token_embedding.device(),
                batch_size: batch.batch_size as u32,
                seq_len: batch.seq_len as u32,
                row_count: batch.token_count as u32,
                residual: &mut buffers.residual,
                normalized: &mut buffers.normalized,
                normalized_amax: &mut buffers.normalized_amax,
                mean: &mut buffers.mean,
                inv_std: &mut buffers.inv_std,
            },
            attention_module: &self.runtime.attention,
            canon_module: &self.runtime.canon,
            attention_tc_module: &self.runtime.f16_tc_matmul,
            quant_module: &self.runtime.quant,
            layer_norm_module: &self.runtime.layer_norm,
            mlp_module: &self.runtime.mlp,
            lm_head_module: &self.runtime.lm_head,
            tma_module: &self.runtime.tma_gemm,
            tma_scale_pack: &self.runtime.tma_scale_pack,
            tma_pad: &self.runtime.tma_pad,
            tma_descriptors: &mut buffers.forward_tma.descriptors,
            tma_input_scale_packed: &mut buffers.forward_tma.input_scales,
            tma_wide_input_scale_packed: &mut buffers.forward_tma.wide_input_scales,
            tma_weight_scale_packed: &mut buffers.forward_tma.weight_scales,
            tma_weight_bytes_padded: &mut buffers.forward_tma.weight_bytes_padded,
            hidden_nvfp4: buffers.hidden_nvfp4.scratch(),
            attention_tc_scratch: buffers.scratch.attention_core.forward_tc(),
            mlp_activation_nvfp4: buffers.mlp_activation_nvfp4.scratch(),
            attention: std::array::from_fn(|i| {
                uploaded.blocks[i].attention_tensors(
                    uploaded.xsa_alphas.device(),
                    (i * gpt2_nvfp4::GPT2_N_HEAD) as u32,
                )
            }),
            block_ln_1: std::array::from_fn(|i| uploaded.blocks[i].ln_1.tensors()),
            block_canon_a: std::array::from_fn(|i| uploaded.blocks[i].canon_a.tensors()),
            block_ln_2: std::array::from_fn(|i| uploaded.blocks[i].ln_2.tensors()),
            block_canon_c: std::array::from_fn(|i| uploaded.blocks[i].canon_c.tensors()),
            mlp: std::array::from_fn(|i| uploaded.blocks[i].mlp_tensors()),
            ln_f: uploaded.ln_f.tensors(),
            attention_qkv: &mut buffers.qkv,
            attention_value_residual: &mut buffers.value_residual,
            attention_log_sum_exp: &mut buffers.log_sum_exp,
            mlp_activation: &mut buffers.mlp_act,
            mlp_route_scores: &mut buffers.mlp_route_scores,
            mlp_route_masks: &mut buffers.mlp_route_masks,
            logits: &mut buffers.logits,
            tape: Some(buffers.tape.tape()),
        })?;

        next_latent_forward(NextLatForwardArgs {
            stream,
            embedding: &self.runtime.embedding,
            layer_norm: &self.runtime.layer_norm,
            quant: &self.runtime.quant,
            tma: &self.runtime.tma_gemm,
            tma_scale_pack: &self.runtime.tma_scale_pack,
            next_latent: &self.runtime.next_latent,
            token_embedding: uploaded.token_embedding.device(),
            weights: &uploaded.next_latent,
            targets: &batch.targets,
            current_states: hidden.normalized,
            buffers: &mut buffers.next_latent,
            batch_size: batch.batch_size as u32,
            seq_len: batch.seq_len as u32,
            row_count: batch.token_count as u32,
            lambda: next_latent_loss_weight(),
        })?;

        Ok(TrainStats {
            tokens: batch.token_count,
            logits: batch.token_count * GPT2_VOCAB_SIZE,
            finite: true,
            ..TrainStats::default()
        })
    }
}
