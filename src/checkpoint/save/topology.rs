use cuda_core::CudaStream;

use super::{super::format::CheckpointWriter, tensor};
use crate::{
    AppResult,
    upload::{UploadedBlock, UploadedModel, UploadedNextLat, UploadedPair},
};

pub(super) fn write_model(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    model: &UploadedModel,
) -> AppResult {
    tensor::write(writer, stream, "token_embedding", &model.token_embedding)?;
    tensor::write(writer, stream, "xsa_alphas", &model.xsa_alphas)?;
    for (index, block) in model.blocks.iter().enumerate() {
        write_block(writer, stream, index, block)?;
    }
    write_uploaded_pair(writer, stream, "ln_f", &model.ln_f)?;
    write_next_latent(writer, stream, &model.next_latent)
}

fn write_next_latent(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    next_latent: &UploadedNextLat,
) -> AppResult {
    let pairs = [
        ("norm", &next_latent.norm),
        ("input_projection", &next_latent.input_projection),
        ("transition", &next_latent.transition),
        ("output_projection", &next_latent.output_projection),
    ];
    write_pairs(writer, stream, "next_latent", pairs)
}

fn write_block(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    index: usize,
    block: &UploadedBlock,
) -> AppResult {
    let prefix = format!("blocks.{index}");
    let pairs = [
        ("ln_1", &block.ln_1),
        ("attn_qkv", &block.attn_qkv),
        ("attn_c_proj", &block.attn_c_proj),
        ("ln_2", &block.ln_2),
        ("mlp_up", &block.mlp_up),
        ("mlp_down", &block.mlp_down),
    ];
    write_pairs(writer, stream, &prefix, pairs)?;
    tensor::write_fp32(
        writer,
        stream,
        &format!("{prefix}.canon_a.weight"),
        &block.canon_a,
    )?;
    tensor::write_fp32(
        writer,
        stream,
        &format!("{prefix}.canon_c.weight"),
        &block.canon_c,
    )?;
    tensor::write(
        writer,
        stream,
        &format!("{prefix}.attn_qk_scale"),
        &block.attn_qk_scale,
    )
}

fn write_uploaded_pair(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    prefix: &str,
    pair: &UploadedPair,
) -> AppResult {
    tensor::write(writer, stream, &format!("{prefix}.weight"), &pair.weight)?;
    tensor::write(writer, stream, &format!("{prefix}.bias"), &pair.bias)
}

fn write_pairs<const N: usize>(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    prefix: &str,
    pairs: [(&str, &UploadedPair); N],
) -> AppResult {
    for (suffix, pair) in pairs {
        write_uploaded_pair(writer, stream, &format!("{prefix}.{suffix}"), pair)?;
    }
    Ok(())
}
