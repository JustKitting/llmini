use super::Builder;
use crate::{
    upload::{UploadedModel, UploadedPair},
    AppResult,
};

fn pair(builder: &mut Builder, prefix: &str, pair: &UploadedPair) -> AppResult {
    builder.nv(&format!("{prefix}.weight"), &pair.weight, None)?;
    builder.nv(&format!("{prefix}.bias"), &pair.bias, None)
}

pub(super) fn bind(builder: &mut Builder, model: &UploadedModel) -> AppResult {
    builder.nv("token_embedding", &model.token_embedding, None)?;
    builder.nv("xsa_alphas", &model.xsa_alphas, None)?;
    for (index, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blocks.{index}");
        for (suffix, value) in [
            ("ln_1", &block.ln_1),
            ("attn_qkv", &block.attn_qkv),
            ("attn_c_proj", &block.attn_c_proj),
            ("ln_2", &block.ln_2),
            ("mlp_up", &block.mlp_up),
            ("mlp_down", &block.mlp_down),
        ] {
            pair(builder, &format!("{prefix}.{suffix}"), value)?;
        }
        builder.fp(&format!("{prefix}.canon_a.weight"), &block.canon_a.weight)?;
        builder.fp(&format!("{prefix}.canon_c.weight"), &block.canon_c.weight)?;
        builder.nv(
            &format!("{prefix}.attn_qk_scale"),
            &block.attn_qk_scale,
            None,
        )?;
    }
    pair(builder, "ln_f", &model.ln_f)?;
    for (suffix, value) in [
        ("norm", &model.next_latent.norm),
        ("input_projection", &model.next_latent.input_projection),
        ("transition", &model.next_latent.transition),
        ("output_projection", &model.next_latent.output_projection),
    ] {
        pair(builder, &format!("next_latent.{suffix}"), value)?;
    }
    for (index, unit) in model.mhc.as_ref().unwrap().units.iter().enumerate() {
        let prefix = if unit.head {
            "mhc.head".to_string()
        } else {
            format!("mhc.boundaries.{index}")
        };
        builder.nv(
            &format!("{prefix}.projection.weight"),
            &unit.phi,
            Some(unit.active_rows() * crate::training::mhc::WIDTH / 2),
        )?;
        builder.fp(&format!("{prefix}.bias"), &unit.bias.weight)?;
        builder.fp(&format!("{prefix}.scale"), &unit.alpha.weight)?;
    }
    for module in &model.engram.as_ref().unwrap().modules {
        let prefix = module.cfg.prefix();
        for (name, tensor) in module.cfg.nv_names().into_iter().zip(&module.nv) {
            builder.nv(&format!("{prefix}.{name}"), tensor, None)?;
        }
        for (name, tensor) in ["query_norm", "key_norm", "convolution_norm", "convolution"]
            .into_iter()
            .zip(&module.fp)
        {
            builder.fp(&format!("{prefix}.{name}"), &tensor.weight)?;
        }
        builder.token_map(&format!("{prefix}.token_id_map"), &module.token_map)?;
    }
    Ok(())
}
