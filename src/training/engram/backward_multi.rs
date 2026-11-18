//! Complete four-query/shared-value adjoint, all dense products native NVFP4.
use super::*;
use crate::training::runtime::Runtime;
use gpt2_nvfp4::GPT2_N_EMBD;
use rust_kernels_cuda::{
    linear_backward::{
        LinearBackwardInputTranspose, LinearBackwardMsEdenArgs, LinearBackwardWeightTranspose,
    },
    residual::ResidualGradAccumulateArgs,
};

impl Module {
    pub fn backward_multistream(
        &mut self,
        r: &Runtime,
        w: &ModuleWeights,
        dres: &mut DeviceBuffer<f32>,
        scratch: &mut super::backward_scratch::BackwardScratch,
    ) -> Result<(), DriverError> {
        let s = r.stream.as_ref();
        let n = self.n;
        let d = GPT2_N_EMBD as u32;
        let memory = self.cfg.memory() as u32;
        let multi = self.multi.as_ref().expect("four-stream CUDA module");
        for g in self.ng.iter().chain(&self.fg) {
            zero(s, g)?;
        }
        // pre becomes d(convolution preactivation) after its forward SiLU.
        self.module
            .engram_conv_delta_kernel(s, grid(n * d * 4), dres, &mut self.pre, n * d * 4)?;
        multi.engram4_conv_input_backward_kernel(
            s,
            grid(n * d * 4),
            &self.pre,
            &w.fp[3].weight,
            &mut scratch.d_un,
            n,
            d,
            self.seq,
        )?;
        multi.engram4_norm_backward_kernel(
            s,
            rows(n * 4),
            &self.gated,
            &scratch.d_un,
            &w.fp[2].weight,
            &self.inv_u,
            dres,
            &mut scratch.d_u,
            n,
            d,
        )?;
        // One token block sums all four dV contributions before shared V's
        // only dense backward. Direct residual/query gradients add in place.
        multi.engram4_gate_backward_kernel(
            s,
            rows(n),
            &self.hidden,
            &self.key,
            &self.value,
            &w.fp[0].weight,
            &w.fp[1].weight,
            &self.stats,
            &scratch.d_u,
            dres,
            &mut scratch.d_key,
            &mut scratch.d_value,
            &mut scratch.d_dot,
            n,
            d,
        )?;
        let [gh, gk, gu, gw] = &mut self.fg;
        multi.engram4_parameter_backward_kernel(
            s,
            grid(d * 4 * 16),
            &self.hidden,
            &self.key,
            &w.fp[0].weight,
            &w.fp[1].weight,
            &self.stats,
            &scratch.d_dot,
            &self.gated,
            &self.normalized,
            &self.inv_u,
            &scratch.d_un,
            &self.pre,
            gh,
            gk,
            gu,
            gw,
            n,
            d,
            self.seq,
        )?;

        for wi in self.cfg.projector_indices() {
            let is_value = wi == self.cfg.value_index();
            if !is_value {
                multi.engram4_stream_copy_kernel(
                    s,
                    grid(n * d),
                    &scratch.d_key,
                    &mut self.projection,
                    n,
                    d,
                    ((wi - 1) / 2) as u32,
                    true,
                )?;
            }
            let e = if is_value {
                &scratch.d_value
            } else {
                &self.projection
            };
            let sign_seed = self.rng.next_u32();
            let scale_seed = self.rng.next_u32();
            let (before_bias, bias_and_after) = self.ng.split_at_mut(wi + 1);
            r.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
                stream: s,
                quant_module: &r.quant,
                e,
                weight_t: LinearBackwardWeightTranspose::Nvfp4(w.nv[wi].device()),
                input_t: LinearBackwardInputTranspose::RowwiseNvfp4(self.memory_q.rowwise()),
                scratch: scratch.linear.parts(),
                dinput: &mut scratch.d_mem_value,
                dweight: &mut before_bias[wi],
                dbias: Some(&mut bias_and_after[0]),
                token_count: n,
                input_dim: memory,
                output_dim: d,
                sign_seed,
                scale_seed,
                precomputed_e_amax_chunks: None,
            })?;
            #[cfg(test)]
            if let Some(verification) = &mut self.memory_adjoint_verification {
                verification.observe(s, wi, &scratch.d_mem_value, (n * memory) as usize)?;
            }
            if wi == self.cfg.key_index(0) {
                copy(
                    s,
                    &scratch.d_mem_value,
                    &scratch.d_mem_key,
                    (n * memory) as usize,
                )?;
            } else {
                r.residual.grad_accumulate(ResidualGradAccumulateArgs {
                    stream: s,
                    branch: &scratch.d_mem_value,
                    out: &mut scratch.d_mem_key,
                    len: n * memory,
                })?;
            }
        }
        // Shared addressing is scattered once, after all K and V adjoints.
        multi.engram4_memory_scatter_kernel(
            s,
            grid(n * memory),
            &self.ids,
            &scratch.d_mem_key,
            &mut self.ng[0],
            n,
            (HEADS * 2) as u32,
            self.cfg.head_dim as u32,
        )
    }
}
