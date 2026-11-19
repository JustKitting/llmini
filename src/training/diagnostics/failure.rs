//! Failure-only copies. Nothing is read back on a healthy training step.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use cuda_core::{CudaStream, DeviceBuffer};

use crate::AppResult;
use crate::training::optimizer_state::{
    AdamState, Fp32AdamState, LayerNormState, LinearState, MuonState,
};
use crate::training::{TokenBatch, Trainer};

#[derive(Default, Debug)]
struct Summary {
    count: usize,
    nonfinite: usize,
    first_bad: Option<usize>,
    max_abs: f32,
    max_index: usize,
    max_value: f32,
    sum_sq: f64,
}

impl Summary {
    fn from_values(values: impl Iterator<Item = f32>) -> Self {
        let mut result = Self::default();
        for (index, value) in values.enumerate() {
            result.count += 1;
            if !value.is_finite() {
                result.nonfinite += 1;
                result.first_bad.get_or_insert(index);
            } else {
                result.sum_sq += f64::from(value) * f64::from(value);
                if value.abs() > result.max_abs {
                    result.max_abs = value.abs();
                    result.max_index = index;
                    result.max_value = value;
                }
            }
        }
        result
    }
}

struct Dump<'a> {
    stream: &'a CudaStream,
    file: BufWriter<File>,
}

impl Dump<'_> {
    fn row(&mut self, name: &str, values: impl Iterator<Item = f32>) -> AppResult {
        let s = Summary::from_values(values);
        let finite = s.count - s.nonfinite;
        writeln!(
            self.file,
            "{name}\t{}\t{}\t{}\t{:.9e}\t{}\t{:.9e}\t{:.9e}",
            s.count,
            s.nonfinite,
            s.first_bad.map_or_else(|| "-".into(), |i| i.to_string()),
            s.max_abs,
            s.max_index,
            s.max_value,
            (s.sum_sq / finite.max(1) as f64).sqrt()
        )?;
        // Preserve completed rows even if a later device copy fails.
        self.file.flush()?;
        Ok(())
    }

    fn f32(&mut self, name: &str, buffer: &DeviceBuffer<f32>) -> AppResult {
        self.row(name, buffer.to_host_vec(self.stream)?.into_iter())
    }

    fn bf16(&mut self, name: &str, buffer: &DeviceBuffer<u16>) -> AppResult {
        self.row(
            name,
            buffer
                .to_host_vec(self.stream)?
                .into_iter()
                .map(|bits| f32::from_bits(u32::from(bits) << 16)),
        )
    }

    fn f16(&mut self, name: &str, buffer: &DeviceBuffer<u16>) -> AppResult {
        self.row(
            name,
            buffer.to_host_vec(self.stream)?.into_iter().map(decode_f16),
        )
    }

    fn adam(&mut self, name: &str, state: &AdamState) -> AppResult {
        self.f32(&format!("{name}.z"), &state.z_master)?;
        self.f32(&format!("{name}.x"), &state.x_master)?;
        self.f32(&format!("{name}.first"), &state.first)?;
        self.f32(&format!("{name}.second"), &state.second)
    }

    fn canon(&mut self, name: &str, state: &Fp32AdamState) -> AppResult {
        self.f32(&format!("{name}.z"), &state.z_master)?;
        self.f32(&format!("{name}.x"), &state.x_master)?;
        self.f32(&format!("{name}.first"), &state.first)?;
        self.f32(&format!("{name}.second"), &state.second)
    }

    fn norm(&mut self, name: &str, state: &LayerNormState) -> AppResult {
        self.adam(&format!("{name}.weight"), &state.weight)?;
        self.adam(&format!("{name}.bias"), &state.bias)
    }

    fn linear(&mut self, name: &str, state: &LinearState) -> AppResult {
        self.muon(name, &state.weight_muon)?;
        self.adam(&format!("{name}.bias"), &state.bias)
    }

    fn muon(&mut self, name: &str, m: &MuonState) -> AppResult {
        self.f32(&format!("{name}.weight.z"), &m.z_master)?;
        self.f32(&format!("{name}.weight.x"), &m.x_master)?;
        self.bf16(&format!("{name}.weight.momentum"), &m.momentum)?;
        self.bf16(&format!("{name}.weight.variance"), &m.variance)?;
        self.f32(&format!("{name}.weight.normuon_second"), &m.second_momentum)?;
        for (label, scalar) in [
            ("exponential", &m.symexp_lin.exponential),
            ("linear", &m.symexp_lin.linear),
            ("curvature", &m.symexp_lin.curvature),
        ] {
            for (field, buffer) in [
                ("z", &scalar.z_master),
                ("x", &scalar.x_master),
                ("first", &scalar.first),
                ("second", &scalar.second),
                ("grad", &scalar.grad),
            ] {
                self.f32(&format!("{name}.weight.{label}.{field}"), buffer)?;
            }
        }
        Ok(())
    }
}

impl Trainer {
    pub(in crate::training) fn write_numerical_failure_dump(
        &self,
        batch: &TokenBatch,
        path: &Path,
    ) -> AppResult {
        let stream = self.runtime.stream.as_ref();
        stream.synchronize()?;
        let mut dump = Dump {
            stream,
            file: BufWriter::new(File::create(path)?),
        };
        let state = &self.buffers.optimizer_state;
        writeln!(dump.file, "# accepted_optimizer_updates={}", state.step)?;
        writeln!(
            dump.file,
            "name\tcount\tnonfinite\tfirst_bad\tmax_finite_abs\tmax_finite_index\tmax_finite_value\tfinite_rms"
        )?;
        dump.f32("token_embedding.z", state.token_embedding.z_master())?;
        dump.f32("token_embedding.x", state.token_embedding.x_master())?;
        dump.adam("xsa_alphas", &state.xsa_alphas)?;
        dump.norm("ln_f", &state.ln_f)?;
        for (i, b) in state.blocks.iter().enumerate() {
            let n = format!("blocks.{i}");
            dump.norm(&format!("{n}.ln_1"), &b.ln_1)?;
            dump.canon(&format!("{n}.canon_a"), &b.canon_a)?;
            dump.linear(&format!("{n}.attn_qkv"), &b.attn_qkv)?;
            dump.adam(&format!("{n}.attn_qk_scale"), &b.attn_qk_scale)?;
            dump.linear(&format!("{n}.attn_c_proj"), &b.attn_c_proj)?;
            dump.norm(&format!("{n}.ln_2"), &b.ln_2)?;
            dump.canon(&format!("{n}.canon_c"), &b.canon_c)?;
            dump.linear(&format!("{n}.mlp_up"), &b.mlp_up)?;
            dump.linear(&format!("{n}.mlp_down"), &b.mlp_down)?;
        }
        let n = &state.next_latent;
        dump.norm("next_latent.norm", &n.norm)?;
        dump.linear("next_latent.input_projection", &n.input_projection)?;
        dump.linear("next_latent.transition", &n.transition)?;
        dump.linear("next_latent.output_projection", &n.output_projection)?;
        if let Some(e) = &self.engram {
            for e in &e.modules {
                let gradients = e.named_gradients();
                for (name, gradient) in &gradients {
                    dump.f32(&format!("{name}.gradient"), gradient)?;
                }
                for (index, state) in e.cfg.adam_indices().into_iter().zip(&e.nv_states) {
                    dump.adam(gradients[index].0, state)?;
                }
                for (name, state) in e
                    .cfg
                    .projector_indices()
                    .into_iter()
                    .map(|index| gradients[index].0)
                    .zip(&e.projector_states)
                {
                    dump.muon(&name, state)?;
                }
                for ((name, _), state) in
                    gradients.iter().skip(e.ng.len()).take(4).zip(&e.fp_states)
                {
                    dump.canon(name, state)?;
                }
                for (name, buffer) in [
                    ("input", &e.hidden),
                    ("key", &e.key),
                    ("value", &e.value),
                    ("gated_value", &e.gated),
                    ("gate_stats", &e.stats),
                    ("convolution_pre_or_delta", &e.pre),
                ] {
                    dump.f32(&format!("forward.{}.{name}", e.cfg.prefix()), buffer)?;
                }
            }
        }
        if let Some(f) = &self.latent_feedback {
            dump.muon("latent_feedback.fusion", &f.state)?;
            dump.f32("latent_feedback.fusion.gradient", &f.grad)?;
            dump.f32("forward.latent_feedback.top", &self.buffers.residual)?;
            dump.f32("forward.latent_feedback.input", &f.input)?;
            dump.f32("forward.latent_feedback.projected", &f.projected)?;
        }
        if let Some(mhc) = &self.mhc {
            for unit in &mhc.units {
                dump.muon(&unit.names[0], &unit.phi)?;
                for (name, gradient) in unit.gradients() {
                    dump.f32(name, gradient)?;
                }
                for (name, state) in unit.names[1..3].iter().zip(&unit.fp) {
                    dump.canon(name, state)?;
                }
                if let Some(branch) = &unit.branch {
                    dump.f32(&format!("forward.{}.branch", unit.names[0]), branch)?;
                }
            }
            for checkpoint in &mhc.checkpoints {
                dump.f32(
                    &format!("mhc.checkpoint.{}.input", checkpoint.index),
                    &checkpoint.input,
                )?;
            }
            for (index, input) in mhc.replay_inputs.iter().enumerate() {
                let slot = index + 1;
                let resident =
                    mhc.tape_units[slot].map_or_else(|| "unused".into(), |i| i.to_string());
                dump.f32(
                    &format!("mhc.tape_slot.{slot}.boundary.{resident}.input"),
                    input,
                )?;
            }
            for (index, projection) in mhc.projections.iter().enumerate() {
                for (label, buffer) in [
                    ("inverse_rms", &projection.inverse),
                    ("raw_coefficients", &projection.raw),
                    ("coefficients", &projection.coefficients),
                ] {
                    dump.f32(&format!("mhc.projection.boundary.{index}.{label}"), buffer)?;
                }
            }
            dump.f32("mhc.stream_work_forward_or_reverse_gradient", &mhc.current)?;
            dump.f32(
                "mhc.alternate_stream_gradient_or_coefficient_dnormalized_work",
                &mhc.dx,
            )?;
        }
        let saved = self.buffers.tape.saved(
            &batch.tokens,
            batch.batch_size as u32,
            batch.seq_len as u32,
            batch.token_count as u32,
            &self.buffers.logits,
        );
        dump.f16(
            "forward.final_norm.standardized_input",
            saved.final_norm.residual,
        )?;
        for (i, b) in saved.blocks.iter().enumerate() {
            dump.f16(
                &format!("forward.blocks.{i}.ln_1.standardized_input"),
                b.ln_1.residual,
            )?;
            dump.f16(
                &format!("forward.blocks.{i}.ln_2.standardized_input"),
                b.ln_2.residual,
            )?;
            dump.f16(&format!("forward.blocks.{i}.qkv"), b.qkv)?;
            dump.f16(
                &format!("forward.blocks.{i}.attention_out"),
                b.attention_out,
            )?;
            dump.f16(&format!("forward.blocks.{i}.mlp_up"), b.mlp_up)?;
        }
        if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
            dump.f32(
                "mlp.paired_recompute_workspace",
                &self.buffers.scratch.mlp_paired,
            )?;
        }
        dump.f32("forward.logits", &self.buffers.logits)?;
        if gpt2_nvfp4::GPT2_VALUE_RMS {
            dump.f32(
                "value_rms.immutable_raw_first_value",
                &self.buffers.value_residual,
            )?;
            dump.f32(
                "value_rms.shared_raw_gradient",
                &self.buffers.scratch.value_first_grad,
            )?;
            dump.f32(
                "value_rms.raw_V_recompute_workspace_qkv_stride",
                &self.buffers.scratch.value_qkv_recompute,
            )?;
        }
        Ok(())
    }
}

fn decode_f16(bits: u16) -> f32 {
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    let value = match exponent {
        0 => f32::from(fraction) * (1.0 / 16_777_216.0),
        31 => {
            if fraction == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => {
            f32::from_bits((u32::from(exponent) + 112) << 23) * (1.0 + f32::from(fraction) / 1024.0)
        }
    };
    if bits & 0x8000 == 0 { value } else { -value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA; only small diagnostic buffers, no training"]
    fn failure_dump_reads_actual_device_special_values() -> AppResult {
        let runtime = crate::training::runtime::Runtime::new()?;
        let stream = runtime.stream.as_ref();
        let floats = DeviceBuffer::from_host(stream, &[3.0_f32, f32::NAN, f32::INFINITY, -42.0])?;
        let bfloats = DeviceBuffer::from_host(stream, &[0x3f80_u16, 0x7fc0, 0xff80, 0xc228])?;
        let halves = DeviceBuffer::from_host(stream, &[0x0001_u16, 0x7e00, 0x7c00, 0xfbff])?;
        let path = std::env::var_os("CODEX_TMP_SESSION")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("nvfp4-numerical-failure-dump-smoke.tsv");
        let mut dump = Dump {
            stream,
            file: BufWriter::new(File::create(&path)?),
        };
        dump.f32("test.f32", &floats)?;
        dump.bf16("test.bf16", &bfloats)?;
        dump.f16("test.f16", &halves)?;
        drop(dump);
        let text = std::fs::read_to_string(&path)?;
        assert!(
            text.contains("test.f32\t4\t2\t1\t4.200000000e1\t3\t-4.200000000e1"),
            "{text}"
        );
        assert!(
            text.contains("test.bf16\t4\t2\t1\t4.200000000e1\t3\t-4.200000000e1"),
            "{text}"
        );
        assert!(
            text.contains("test.f16\t4\t2\t1\t6.550400000e4\t3\t-6.550400000e4"),
            "{text}"
        );
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn failure_summary_preserves_nonfinite_locations_and_finite_extrema() {
        let s = Summary::from_values([0.0, f32::NAN, -42.0, f32::INFINITY, 3.0].into_iter());
        assert_eq!((s.count, s.nonfinite, s.first_bad), (5, 2, Some(1)));
        assert_eq!((s.max_abs, s.max_index, s.max_value), (42.0, 2, -42.0));
        assert_eq!(s.sum_sq, 1773.0);
        assert_eq!(decode_f16(0x0001), 2.0_f32.powi(-24));
        assert_eq!(decode_f16(0xfbff), -65504.0);
        assert!(decode_f16(0x7c00).is_infinite());
        assert!(decode_f16(0x7e00).is_nan());
    }
}
