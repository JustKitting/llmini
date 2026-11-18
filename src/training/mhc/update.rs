use super::*;
use crate::training::{optimizer::OptimizerScratch, runtime::Runtime, schedule_free::Materializer};
impl Mhc {
    pub fn materialize(
        &mut self,
        r: &Runtime,
        w: &mut Weights,
        scratch: &mut OptimizerScratch,
        beta: f32,
        evaluation: bool,
    ) -> Result<(), DriverError> {
        let mut m = Materializer::new(
            &r.stream,
            &r.optimizer,
            &r.f32_ops,
            scratch,
            beta,
            crate::training::symexp_lin::beta(),
            false,
            false,
        );
        for (unit, weights) in self.units.iter().zip(&mut w.units) {
            if evaluation {
                m.symexp_lin_muon_master(
                    &mut weights.phi,
                    &unit.phi.x_master,
                    &unit.phi.symexp_lin,
                )?;
                m.fp32_master(&mut weights.bias, &unit.fp[0].x_master)?;
                m.fp32_master(&mut weights.alpha, &unit.fp[1].x_master)?;
            } else {
                m.muon(&mut weights.phi, &unit.phi)?;
                m.fp32_adam(&mut weights.bias, &unit.fp[0])?;
                m.fp32_adam(&mut weights.alpha, &unit.fp[1])?;
            }
        }
        Ok(())
    }
    pub fn update_vectors(
        &mut self,
        r: &Runtime,
        step: u32,
        average: f32,
        scale: f32,
    ) -> Result<(), DriverError> {
        let c = crate::training::optimizer_apply::adam_debug_config(step);
        for unit in &mut self.units {
            for (state, gradient) in unit.fp.iter_mut().zip(&unit.fg) {
                r.optimizer.apply_fp32_adamw_update(
                    rust_kernels_cuda::optimizer::Fp32AdamWUpdateArgs {
                        stream: &r.stream,
                        z_master: &mut state.z_master,
                        x_master: &mut state.x_master,
                        grad: gradient,
                        grad_scale: scale,
                        first_moment: &mut state.first,
                        second_moment: &mut state.second,
                        len: gradient.len() as u32,
                        learning_rate: c.learning_rate,
                        weight_decay: 0.,
                        beta1: c.beta1,
                        beta2: c.beta2,
                        beta1_correction: c.beta1_correction,
                        beta2_correction: c.beta2_correction,
                        eps: c.eps,
                        average_coefficient: average,
                    },
                )?;
            }
        }
        Ok(())
    }
}
