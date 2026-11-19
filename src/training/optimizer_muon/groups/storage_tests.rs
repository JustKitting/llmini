//! Cross-layer audit of actual allocated model weights and optimizer tables.
use super::MuonPointerTables;
use crate::upload::{UploadedModel, UploadedNvfp4};

impl MuonPointerTables {
    pub(in crate::training) fn verify_native_storage_layout(&self, model: &UploadedModel) {
        let mut seen = std::collections::HashSet::new();
        {
            let mut check = |name: &str, weight: &UploadedNvfp4, active_rows: Option<u32>| {
                let slots: Vec<_> = self
                    .all
                    .host_slots
                    .iter()
                    .filter(|slot| slot.rows != 0 && slot.bytes == weight.bytes.cu_deviceptr())
                    .collect();
                assert_eq!(slots.len(), 1, "{name}: unique real matrix slot");
                let slot = slots[0];
                let rows = active_rows.unwrap_or(weight.scale_rows);
                assert!(rows <= weight.scale_rows, "{name}: active rows fit storage");
                assert_eq!(
                    (slot.rows, slot.cols),
                    (rows, weight.scale_k),
                    "{name}: optimizer must interpret physical output-by-input storage"
                );
                assert!(u64::from(slot.rows) * u64::from(slot.cols) <= weight.len as u64);
                let scales: Vec<_> = self
                    .all
                    .host_symexp_lin_slots
                    .iter()
                    .filter(|scale| scale.grad == slot.grad)
                    .collect();
                assert_eq!(scales.len(), 1, "{name}: unique scale parameter slot");
                assert_eq!((scales[0].rows, scales[0].cols), (slot.rows, slot.cols));
                assert!(
                    seen.insert(slot.bytes),
                    "{name}: duplicate weight ownership"
                );
            };
            for (index, block) in model.blocks.iter().enumerate() {
                check(
                    &format!("blocks.{index}.attn_qkv"),
                    &block.attn_qkv.weight,
                    Some(
                        gpt2_nvfp4::attention_trainable_qkv_dim(gpt2_nvfp4::uses_full_attention(
                            index,
                        )) as u32,
                    ),
                );
                check(
                    &format!("blocks.{index}.attn_c_proj"),
                    &block.attn_c_proj.weight,
                    None,
                );
                check(
                    &format!("blocks.{index}.mlp_up"),
                    &block.mlp_up.weight,
                    None,
                );
                check(
                    &format!("blocks.{index}.mlp_down"),
                    &block.mlp_down.weight,
                    None,
                );
            }
            for (name, weight) in [
                (
                    "next_latent.input",
                    &model.next_latent.input_projection.weight,
                ),
                (
                    "next_latent.transition",
                    &model.next_latent.transition.weight,
                ),
                (
                    "next_latent.output",
                    &model.next_latent.output_projection.weight,
                ),
            ] {
                check(name, weight, None);
            }
            if let Some(engram) = &model.engram {
                for (index, module) in engram.modules.iter().enumerate() {
                    for tensor in module.cfg.projector_indices() {
                        check(
                            &format!("engram.{index}.projector.{tensor}"),
                            &module.nv[tensor],
                            None,
                        );
                    }
                }
            }
            if let Some(mhc) = &model.mhc {
                for (index, unit) in mhc.units.iter().enumerate() {
                    check(&format!("mhc.{index}.phi"), &unit.phi, None);
                }
            }
            if let Some(feedback) = &model.latent_feedback {
                check("latent_feedback", &feedback.weight, None);
            }
        }
        assert_eq!(
            seen.len(),
            self.all
                .host_slots
                .iter()
                .filter(|slot| slot.rows != 0)
                .count(),
            "every active matrix slot must match an actual model weight"
        );
        println!(
            "\nmuon_storage_layout_verified matrices={} physical=output_by_input active_projection_tails=preserved optimizer_updates=0",
            seen.len()
        );
    }
}
