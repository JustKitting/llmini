use super::*;
use crate::training::runtime::Runtime;
use cuda_core::{DeviceBuffer, DriverError};
use gpt2_nvfp4::GPT2_N_EMBD;
use rust_kernels_cuda::linear_backward::{
    LinearBackwardInputTranspose, LinearBackwardMsEdenArgs, LinearBackwardWeightTranspose,
};
impl Module {
    pub fn backward(
        &mut self,
        r: &Runtime,
        w: &ModuleWeights,
        dres: &mut DeviceBuffer<f32>,
        scratch: &mut super::backward_scratch::BackwardScratch,
    ) -> Result<(), DriverError> {
        if self.cfg.streams() == 4 {
            return self.backward_multistream(r, w, dres, scratch);
        }
        let memory = w.cfg.memory();
        let sign_seed = self.rng.next_u32();
        let scale_seed = self.rng.next_u32();
        let s = r.stream.as_ref();
        let n = self.n;
        let d = GPT2_N_EMBD as u32;
        for g in self.ng.iter().chain(self.fg.iter()) {
            zero(s, g)?;
        }
        self.module
            .engram_conv_delta_kernel(s, grid(n * d), dres, &mut self.pre, n * d)?;
        self.module.engram_conv_input_backward_kernel(
            s,
            grid(n * d),
            &self.pre,
            &w.fp[3].weight,
            &mut scratch.d_un,
            n,
            d,
            self.seq,
        )?;
        self.module.engram_norm_backward_kernel(
            s,
            rows(n),
            &self.gated,
            &scratch.d_un,
            &w.fp[2].weight,
            &self.inv_u,
            dres,
            &mut scratch.d_u,
            n,
            d,
        )?;
        self.module.engram_gate_backward_kernel(
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
        self.module.engram_parameter_backward_kernel(
            s,
            grid(d * 16),
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
        let [_, gkw, gkb, gvw, gvb] = self.ng.as_mut_slice() else {
            unreachable!()
        };
        for (wi, e, dx, dw, db, seed) in [
            (
                1,
                &scratch.d_key,
                &mut scratch.d_mem_key,
                gkw,
                gkb,
                sign_seed,
            ),
            (
                3,
                &scratch.d_value,
                &mut scratch.d_mem_value,
                gvw,
                gvb,
                sign_seed ^ 0x91b2a3c4,
            ),
        ] {
            r.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
                stream: s,
                quant_module: &r.quant,
                e,
                weight_t: LinearBackwardWeightTranspose::Nvfp4(w.nv[wi].device()),
                input_t: LinearBackwardInputTranspose::RowwiseNvfp4(self.memory_q.rowwise()),
                scratch: scratch.linear.parts(),
                dinput: dx,
                dweight: dw,
                dbias: Some(db),
                token_count: n,
                input_dim: memory as u32,
                output_dim: d,
                sign_seed: seed,
                scale_seed,
                precomputed_e_amax_chunks: None,
            })?;
        }
        self.module.engram_scatter_kernel(
            s,
            grid(n * memory as u32),
            &self.ids,
            &scratch.d_mem_key,
            &scratch.d_mem_value,
            &mut self.ng[0],
            n,
            (HEADS * 2) as u32,
            w.cfg.head_dim as u32,
        )
    }
}
