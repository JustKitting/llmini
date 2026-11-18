use super::*;
use crate::training::runtime::Runtime;
use gpt2_nvfp4::GPT2_N_EMBD;
use rust_kernels_cuda::linear_backward::{
    LinearBackwardInputTranspose, LinearBackwardMsEdenArgs, LinearBackwardWeightTranspose,
};
impl Mhc {
    fn coefficient_backward(
        &mut self,
        index: usize,
        r: &Runtime,
        w: &Weights,
        gradient: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        let s = r.stream.as_ref();
        let n = self.n;
        let unit = &mut self.units[index];
        let tape = &self.tapes[tape_slot(index)];
        let projection = &self.projections[index];
        let weights = &w.units[index];
        let [bias_grad, alpha_grad] = &mut unit.fg;
        self.coefficients
            .backward(rust_kernels_cuda::mhc_coefficients::BackwardArgs {
                stream: s,
                raw: &projection.raw,
                bias: &weights.bias.weight,
                alpha: &weights.alpha.weight,
                dc: &self.dc,
                draw: &mut self.draw,
                dbias: bias_grad,
                dalpha: alpha_grad,
                rows: n,
                stride: PAD as u32,
                head: weights.head,
            })?;
        let sign_seed = self.rng.next_u32();
        let scale_seed = self.rng.next_u32();
        r.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
            stream: s,
            quant_module: &r.quant,
            e: &self.draw,
            weight_t: LinearBackwardWeightTranspose::Nvfp4(weights.phi.device()),
            input_t: LinearBackwardInputTranspose::RowwiseNvfp4(projection.input_q.rowwise()),
            scratch: self.linear.parts(),
            // Keep the incoming stream gradient until the fused final write.
            dinput: &mut self.dx,
            dweight: &mut unit.grad,
            dbias: None,
            token_count: n,
            input_dim: WIDTH as u32,
            output_dim: PAD as u32,
            sign_seed,
            scale_seed,
            precomputed_e_amax_chunks: None,
        })?;
        // The true output has24 (or4) rows. MS-EDEN rounding may otherwise
        // introduce noise into the physical zero padding; these are not parameters.
        let active = weights.active_rows() * WIDTH;
        unsafe {
            memory::memset_d8_async(
                unit.grad.cu_deviceptr() + (active * 4) as u64,
                0,
                (PAD * WIDTH - active) * 4,
                s.cu_stream(),
            )?;
        }
        self.reverse
            .finish(rust_kernels_cuda::mhc_reverse::FinishArgs {
                stream: s,
                input: &tape.input,
                inverse: &projection.inverse,
                coefficients: &projection.coefficients,
                normalized_gradient: &self.dx,
                branch_gradient: gradient,
                current: &mut self.current,
                rows: n,
                dim: GPT2_N_EMBD as u32,
                head: weights.head,
            })?;
        Ok(())
    }
    pub fn backward_boundary(
        &mut self,
        index: usize,
        before: bool,
        r: &Runtime,
        w: &Weights,
        gradient: &mut DeviceBuffer<f32>,
    ) -> Result<bool, DriverError> {
        self.ensure_window(index, r, w)?;
        let s = r.stream.as_ref();
        let n = self.n;
        let d = GPT2_N_EMBD as u32;
        if index == BOUNDARIES {
            for unit in &self.units {
                for (_, g) in unit.gradients() {
                    zero(s, g)?;
                }
            }
            let tape = &self.tapes[tape_slot(index)];
            self.reverse.pre(rust_kernels_cuda::mhc_reverse::PreArgs {
                stream: s,
                input: &tape.input,
                coefficients: &self.projections[index].coefficients,
                gradient,
                dc: &mut self.dc,
                rows: n,
                dim: d,
            })?;
            self.coefficient_backward(index, r, w, gradient)?;
        } else if before {
            let unit = &self.units[index];
            let tape = &self.tapes[tape_slot(index)];
            self.reverse
                .post(rust_kernels_cuda::mhc_reverse::PostArgs {
                    stream: s,
                    input: &tape.input,
                    branch: unit.branch.as_ref().unwrap(),
                    coefficients: &self.projections[index].coefficients,
                    incoming: &self.current,
                    df: gradient,
                    dc: &mut self.dc,
                    rows: n,
                    dim: d,
                })?;
        } else {
            let tape = &self.tapes[tape_slot(index)];
            self.reverse.pre(rust_kernels_cuda::mhc_reverse::PreArgs {
                stream: s,
                input: &tape.input,
                coefficients: &self.projections[index].coefficients,
                gradient,
                dc: &mut self.dc,
                rows: n,
                dim: d,
            })?;
            self.coefficient_backward(index, r, w, gradient)?;
            if index == 0 {
                self.module.mhc_expand_backward_kernel(
                    s,
                    grid(n * d),
                    &self.current,
                    gradient,
                    n,
                    d,
                )?;
            }
        }
        Ok(true)
    }
}
