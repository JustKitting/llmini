use super::*;
use crate::training::{
    optimizer::OptimizerScratch,
    optimizer_state::{AdamState, Fp32AdamState, MuonState, StateInit},
    runtime::Runtime,
};
use cuda_core::{DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_TOKEN_ROWS, RowwiseNvfp4Buffers};
use rust_kernels_cuda::{engram::kernels, nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors};
pub(in crate::training) struct Module {
    #[cfg(test)]
    pub memory_adjoint_verification: Option<super::adjoint_verification::MemoryAdjointVerification>,
    pub cfg: Config,
    pub names: Vec<String>,
    pub module: kernels::LoadedModule,
    pub four_gram_hash: Option<rust_kernels_cuda::engram::four_gram_hash::LoadedModule>,
    pub multi: Option<rust_kernels_cuda::engram_multi::kernels::LoadedModule>,
    pub nv_states: Vec<AdamState>,
    pub projector_states: Vec<MuonState>,
    pub fp_states: [Fp32AdamState; 4],
    pub ng: Vec<DeviceBuffer<f32>>,
    pub fg: [DeviceBuffer<f32>; 4],
    pub ids: DeviceBuffer<u32>,
    pub primes: DeviceBuffer<u32>,
    pub offsets: DeviceBuffer<u32>,
    pub multipliers: DeviceBuffer<u64>,
    pub memory: DeviceBuffer<f32>,
    pub memory_q: RowwiseNvfp4Buffers,
    pub row_amax: DeviceBuffer<f32>,
    pub tma: TmaNvfp4DeviceScaleDescriptors,
    pub hidden: DeviceBuffer<f32>,
    pub key: DeviceBuffer<f32>,
    pub value: DeviceBuffer<f32>,
    pub projection: DeviceBuffer<f32>,
    pub gated: DeviceBuffer<f32>,
    pub normalized: DeviceBuffer<f32>,
    pub inv_u: DeviceBuffer<f32>,
    pub pre: DeviceBuffer<f32>,
    pub stats: DeviceBuffer<f32>,
    pub n: u32,
    pub seq: u32,
    pub rng: gpt2_nvfp4::Gpt2Rng,
}
impl Module {
    pub fn new(runtime: &Runtime, w: &ModuleWeights) -> Result<Self, DriverError> {
        let memory = w.cfg.memory();
        let s = runtime.stream.as_ref();
        let init = StateInit::new(s, &runtime.decode, &runtime.optimizer, 0.);
        let single = GPT2_TOKEN_ROWS * GPT2_N_EMBD;
        let streams = w.cfg.streams();
        let total = single * streams;
        let mem = GPT2_TOKEN_ROWS * memory;
        let mut offsets = Vec::new();
        let mut at = 0;
        for &p in &w.cfg.primes {
            offsets.push(at);
            at += p;
        }
        let nv_states = w
            .cfg
            .adam_indices()
            .into_iter()
            .map(|i| AdamState::new(init, &w.nv[i]))
            .collect::<Result<Vec<_>, _>>()?;
        let matrix_init = StateInit::new(
            s,
            &runtime.decode,
            &runtime.optimizer,
            crate::training::symexp_lin::beta(),
        );
        let projector_states = w
            .cfg
            .projector_indices()
            .into_iter()
            .map(|i| MuonState::new(matrix_init, &w.nv[i], GPT2_N_EMBD.max(memory)))
            .collect::<Result<Vec<_>, _>>()?;
        let fp_states =
            w.fp.iter()
                .map(|v| Fp32AdamState::new(init, v))
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .ok()
                .unwrap();
        let ng =
            w.nv.iter()
                .map(|v| DeviceBuffer::zeroed(s, v.len))
                .collect::<Result<Vec<_>, _>>()?;
        let fg =
            w.fp.iter()
                .map(|v| DeviceBuffer::zeroed(s, v.weight.len()))
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .ok()
                .unwrap();
        macro_rules! z {
            ($n:expr) => {
                DeviceBuffer::zeroed(s, $n)?
            };
        }
        let state = (|| -> Result<Self, DriverError> {
            Ok(Self {
                #[cfg(test)]
                memory_adjoint_verification: None,
                cfg: w.cfg.clone(),
                names: w
                    .cfg
                    .nv_names()
                    .into_iter()
                    .chain(
                        ["query_norm", "key_norm", "convolution_norm", "convolution"]
                            .map(String::from),
                    )
                    .chain(w.cfg.projector_indices().into_iter().flat_map(|index| {
                        let name = w.cfg.nv_names()[index].clone();
                        ["exponential", "linear", "curvature"]
                            .map(|suffix| format!("{name}.{suffix}"))
                    }))
                    .map(|n| format!("{}.{n}", w.cfg.prefix()))
                    .collect(),
                module: runtime.engram_module()?,
                four_gram_hash: (w.cfg.max_ngram() == 4)
                    .then(|| runtime.engram_four_gram_hash_module())
                    .transpose()?,
                multi: (streams == 4)
                    .then(|| runtime.engram_multi_module())
                    .transpose()?,
                nv_states,
                projector_states,
                fp_states,
                ng,
                fg,
                ids: z!(GPT2_TOKEN_ROWS * w.cfg.primes.len()),
                primes: DeviceBuffer::from_host(s, &w.cfg.primes)?,
                offsets: DeviceBuffer::from_host(s, &offsets)?,
                multipliers: DeviceBuffer::from_host(s, &w.cfg.multipliers)?,
                memory: z!(mem),
                memory_q: RowwiseNvfp4Buffers::gpt2_rows(s, mem)?,
                row_amax: z!(GPT2_TOKEN_ROWS),
                tma: TmaNvfp4DeviceScaleDescriptors::new(s)?,
                hidden: z!(total),
                key: z!(total),
                value: z!(single),
                projection: z!(if streams == 4 { single } else { 1 }),
                gated: z!(total),
                normalized: z!(total),
                inv_u: z!(GPT2_TOKEN_ROWS * streams),
                pre: z!(total),
                stats: z!(GPT2_TOKEN_ROWS * streams * 4),
                n: 0,
                seq: 0,
                rng: gpt2_nvfp4::Gpt2Rng::new(
                    0x6a91_0735_7234_21ac
                        ^ if w.cfg.legacy {
                            0
                        } else {
                            (w.cfg.initialization_layer() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        },
                ),
            })
        })();
        // Keep the local offsets allocation alive until its device copy ends.
        let synchronized = s.synchronize();
        let state = state?;
        synchronized?;
        Ok(state)
    }
    pub fn gradients(&self) -> Vec<&DeviceBuffer<f32>> {
        self.named_gradients()
            .into_iter()
            .map(|(_, gradient)| gradient)
            .collect()
    }

    pub fn named_gradients(&self) -> Vec<(&str, &DeviceBuffer<f32>)> {
        self.names
            .iter()
            .map(String::as_str)
            .zip(
                self.ng
                    .iter()
                    .chain(self.fg.iter())
                    .chain(self.projector_states.iter().flat_map(|s| {
                        [
                            &s.symexp_lin.exponential.grad,
                            &s.symexp_lin.linear.grad,
                            &s.symexp_lin.curvature.grad,
                        ]
                    })),
            )
            .collect()
    }
    pub fn materialize_training(
        &mut self,
        r: &Runtime,
        w: &mut ModuleWeights,
        scratch: &mut OptimizerScratch,
        beta: f32,
    ) -> Result<(), DriverError> {
        scratch.ensure_nvfp4_capacity(r.stream.as_ref(), w.cfg.table_len())?;
        let mut materializer = crate::training::schedule_free::Materializer::new(
            r.stream.as_ref(),
            &r.optimizer,
            &r.f32_ops,
            scratch,
            beta,
            crate::training::symexp_lin::beta(),
            false,
            false,
        );
        for (index, state) in self.cfg.adam_indices().into_iter().zip(&self.nv_states) {
            materializer.adam(&mut w.nv[index], state, &self.ng[index])?;
        }
        for (index, state) in self
            .cfg
            .projector_indices()
            .into_iter()
            .zip(&self.projector_states)
        {
            materializer.muon(&mut w.nv[index], state)?;
        }
        for (tensor, state) in w.fp.iter_mut().zip(&self.fp_states) {
            materializer.fp32_adam(tensor, state)?;
        }
        Ok(())
    }

    /// Match the base trainer: evaluate every parameter at its averaged x view.
    /// A schedule beta of zero selects z in native NVFP4 materialization.
    pub fn materialize_evaluation(
        &mut self,
        r: &Runtime,
        w: &mut ModuleWeights,
        scratch: &mut OptimizerScratch,
    ) -> Result<(), DriverError> {
        scratch.ensure_nvfp4_capacity(r.stream.as_ref(), w.cfg.table_len())?;
        let mut materializer = crate::training::schedule_free::Materializer::new(
            r.stream.as_ref(),
            &r.optimizer,
            &r.f32_ops,
            scratch,
            0.0,
            crate::training::symexp_lin::beta(),
            false,
            false,
        );
        for (index, state) in self.cfg.adam_indices().into_iter().zip(&self.nv_states) {
            materializer.master(&mut w.nv[index], &state.x_master)?;
        }
        for (index, state) in self
            .cfg
            .projector_indices()
            .into_iter()
            .zip(&self.projector_states)
        {
            materializer.symexp_lin_muon_master(
                &mut w.nv[index],
                &state.x_master,
                &state.symexp_lin,
            )?;
        }
        for (tensor, state) in w.fp.iter_mut().zip(&self.fp_states) {
            materializer.fp32_master(tensor, &state.x_master)?;
        }
        Ok(())
    }
    pub fn update(
        &mut self,
        r: &Runtime,
        w: &mut ModuleWeights,
        scratch: &mut OptimizerScratch,
        step: u32,
        average: f32,
        scale: f32,
    ) -> Result<(), DriverError> {
        scratch.ensure_nvfp4_capacity(r.stream.as_ref(), w.cfg.table_len())?;
        use rust_kernels_cuda::optimizer::{AdamWUpdateArgs, Fp32AdamWUpdateArgs};
        let s = r.stream.as_ref();
        let c = crate::training::optimizer_apply::adam_debug_config(step);
        for (index, st) in self.cfg.adam_indices().into_iter().zip(&mut self.nv_states) {
            let t = &mut w.nv[index];
            let grad = &mut self.ng[index];
            zero(s, &st.schedule_amax)?;
            r.optimizer
                .apply_adamw_update_deferred_quantization(AdamWUpdateArgs {
                    stream: s,
                    bytes: &mut t.bytes,
                    scales: &mut t.scales,
                    global_scale: &mut t.global_scale,
                    z_master: &mut st.z_master,
                    x_master: &mut st.x_master,
                    grad,
                    grad_scale: scale,
                    first_moment: &mut st.first,
                    second_moment: &mut st.second,
                    amax: &mut st.schedule_amax,
                    chunk_amax: &mut scratch.chunk_amax,
                    len: t.len as u32,
                    // Paper section4.1 / AppendixA: the sparse embedding alone
                    // uses five times the ordinary Adam rate, with no decay.
                    learning_rate: c.learning_rate * if index == 0 { 5.0 } else { 1.0 },
                    weight_decay: 0.,
                    beta1: c.beta1,
                    beta2: c.beta2,
                    beta1_correction: c.beta1_correction,
                    beta2_correction: c.beta2_correction,
                    eps: c.eps,
                    average_coefficient: average,
                    schedule_beta: crate::training::learning_rate::schedule_free_beta(step + 1),
                })?;
        }
        for (st, grad) in self.fp_states.iter_mut().zip(&self.fg) {
            r.optimizer.apply_fp32_adamw_update(Fp32AdamWUpdateArgs {
                stream: s,
                z_master: &mut st.z_master,
                x_master: &mut st.x_master,
                grad,
                grad_scale: scale,
                first_moment: &mut st.first,
                second_moment: &mut st.second,
                len: grad.len() as u32,
                learning_rate: c.learning_rate,
                weight_decay: 0.,
                beta1: c.beta1,
                beta2: c.beta2,
                beta1_correction: c.beta1_correction,
                beta2_correction: c.beta2_correction,
                eps: c.eps,
                average_coefficient: average,
            })?;
        }
        Ok(())
    }
}
