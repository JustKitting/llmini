use super::*;
use crate::training::runtime::Runtime;
use cuda_core::{DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, HiddenStateDevice};
impl Module {
    pub fn forward(
        &mut self,
        r: &Runtime,
        w: &ModuleWeights,
        tokens: &DeviceBuffer<u32>,
        h: &mut HiddenStateDevice<'_>,
    ) -> Result<(), DriverError> {
        self.forward_streams(r, w, tokens, h.residual, h.row_count, h.seq_len)
    }
    pub fn forward_streams(
        &mut self,
        r: &Runtime,
        w: &ModuleWeights,
        tokens: &DeviceBuffer<u32>,
        residual: &mut DeviceBuffer<f32>,
        n: u32,
        seq: u32,
    ) -> Result<(), DriverError> {
        let memory = w.cfg.memory();
        let s = r.stream.as_ref();
        let d = GPT2_N_EMBD as u32;
        let streams = self.cfg.streams() as u32;
        if residual.len() < (n * d * streams) as usize {
            return Err(DriverError(1));
        }
        self.n = n;
        self.seq = seq;
        copy(s, residual, &self.hidden, (n * d * streams) as usize)?;
        let hash_heads = w.cfg.primes.len() as u32;
        macro_rules! launch_hash {
            ($module:expr, $kernel:ident) => {
                $module.$kernel(
                    s,
                    grid(n * hash_heads),
                    tokens,
                    &w.token_map,
                    &self.primes,
                    &self.offsets,
                    &self.multipliers,
                    &mut self.ids,
                    n,
                    seq,
                    HEADS as u32,
                    llama2_tokenizer::BOS_TOKEN,
                    llama2_tokenizer::EOS_TOKEN,
                    llama2_tokenizer::EOS_TOKEN,
                )
            };
        }
        if let Some(hash) = &self.four_gram_hash {
            launch_hash!(hash, engram_four_gram_hash_kernel)?;
        } else {
            launch_hash!(self.module, engram_hash_kernel)?;
        }
        self.module.engram_gather_kernel(
            s,
            grid(n * memory as u32),
            &self.ids,
            &w.nv[0].bytes,
            &w.nv[0].scales,
            &w.nv[0].global_scale,
            &mut self.memory,
            n,
            hash_heads,
            w.cfg.head_dim as u32,
        )?;
        self.memory_q.scratch().quantize_row_amax(
            &r.quant,
            s,
            &self.memory,
            &mut self.row_amax,
            n,
            memory as u32,
        )?;
        for wi in self.cfg.projector_indices() {
            let is_value = wi == self.cfg.value_index();
            let bi = wi + 1;
            let input = self.memory_q.rowwise();
            r.tma_gemm.prepare_tma_nvfp4_csf_weight_into(
                s,
                input.bytes,
                input.scales,
                &w.nv[wi].bytes,
                &w.nv[wi].scales,
                n,
                memory as u32,
                d,
                &mut self.tma,
            )?;
            if !is_value
                && streams == 4
                && r.tma_gemm
                    .try_gemm_tma_nvfp4_rowwise_a_scale_affine_strided(
                        s,
                        &self.tma,
                        &mut self.key,
                        w.nv[bi].device(),
                        n,
                        memory as u32,
                        d,
                        d * streams,
                        ((wi - 1) / 2) as u32 * d,
                        input.global_scales,
                        &w.nv[wi].global_scale,
                    )?
            {
                continue;
            }
            let out = if is_value {
                &mut self.value
            } else if streams == 4 {
                &mut self.projection
            } else {
                &mut self.key
            };
            r.tma_gemm
                .gemm_tma_nvfp4_rowwise_a_scale_affine_padded_output(
                    s,
                    &self.tma,
                    out,
                    w.nv[bi].device(),
                    n,
                    memory as u32,
                    d,
                    d,
                    input.global_scales,
                    &w.nv[wi].global_scale,
                )?;
            if !is_value && streams == 4 {
                self.multi.as_ref().unwrap().engram4_stream_copy_kernel(
                    s,
                    grid(n * d),
                    &self.projection,
                    &mut self.key,
                    n,
                    d,
                    ((wi - 1) / 2) as u32,
                    false,
                )?;
            }
        }
        if let Some(multi) = &self.multi {
            multi.engram4_gate_kernel(
                s,
                rows(n * 4),
                &self.hidden,
                &self.key,
                &self.value,
                &w.fp[0].weight,
                &w.fp[1].weight,
                &mut self.stats,
                &mut self.gated,
                n,
                d,
                w.cfg.gate_rms_epsilon(),
            )?;
            multi.engram4_norm_kernel(
                s,
                rows(n * 4),
                &self.gated,
                &w.fp[2].weight,
                &mut self.normalized,
                &mut self.inv_u,
                n,
                d,
            )?;
            return multi.engram4_conv_add_kernel(
                s,
                grid(n * d * 4),
                &self.gated,
                &self.normalized,
                &w.fp[3].weight,
                &mut self.pre,
                residual,
                n,
                d,
                seq,
            );
        }
        self.module.engram_gate_kernel(
            s,
            rows(n),
            &self.hidden,
            &self.key,
            &self.value,
            &w.fp[0].weight,
            &w.fp[1].weight,
            &mut self.stats,
            &mut self.gated,
            n,
            d,
            w.cfg.gate_rms_epsilon(),
        )?;
        self.module.engram_norm_kernel(
            s,
            rows(n),
            &self.gated,
            &w.fp[2].weight,
            &mut self.normalized,
            &mut self.inv_u,
            n,
            d,
        )?;
        self.module.engram_conv_add_kernel(
            s,
            grid(n * d),
            &self.gated,
            &self.normalized,
            &w.fp[3].weight,
            &mut self.pre,
            residual,
            n,
            d,
            seq,
        )
    }
}
