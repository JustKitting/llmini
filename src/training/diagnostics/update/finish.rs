use cuda_core::CudaStream;

use crate::AppResult;
use crate::upload::{UploadedModel, UploadedNvfp4, UploadedPair};

use super::snapshot::PendingTensorUpdateDiagnostics;
use super::stats::tensor_update_stats;
use crate::training::diagnostics::TensorUpdateDiagnostics;

pub(in crate::training::diagnostics) fn finish_update_snapshots(
    stream: &CudaStream,
    uploaded: &UploadedModel,
    pending: Vec<PendingTensorUpdateDiagnostics>,
) -> AppResult<Vec<TensorUpdateDiagnostics>> {
    let mut updates = Vec::with_capacity(pending.len());
    let mut iter = pending.into_iter();

    finish_update(
        &mut updates,
        stream,
        &uploaded.token_embedding,
        iter.next().unwrap(),
    )?;
    if gpt2_nvfp4::exclusive_self_attention_enabled() {
        finish_update(
            &mut updates,
            stream,
            &uploaded.xsa_alphas,
            iter.next().unwrap(),
        )?;
    }
    finish_pair(&mut updates, stream, &uploaded.ln_f, &mut iter)?;

    for (index, block) in uploaded.blocks.iter().enumerate() {
        finish_pair(&mut updates, stream, &block.ln_1, &mut iter)?;
        finish_pair(&mut updates, stream, &block.attn_qkv, &mut iter)?;
        if gpt2_nvfp4::uses_full_attention(index) {
            finish_update(
                &mut updates,
                stream,
                &block.attn_qk_scale,
                iter.next().unwrap(),
            )?;
        }
        finish_pair(&mut updates, stream, &block.attn_c_proj, &mut iter)?;
        finish_pair(&mut updates, stream, &block.ln_2, &mut iter)?;
        finish_pair(&mut updates, stream, &block.mlp_up, &mut iter)?;
        finish_pair(&mut updates, stream, &block.mlp_down, &mut iter)?;
    }
    assert!(
        iter.next().is_none(),
        "update snapshot topology must consume every tensor"
    );

    Ok(updates)
}

fn finish_pair(
    updates: &mut Vec<TensorUpdateDiagnostics>,
    stream: &CudaStream,
    pair: &UploadedPair,
    iter: &mut impl Iterator<Item = PendingTensorUpdateDiagnostics>,
) -> AppResult {
    finish_update(updates, stream, &pair.weight, iter.next().unwrap())?;
    finish_update(updates, stream, &pair.bias, iter.next().unwrap())
}

fn finish_update(
    updates: &mut Vec<TensorUpdateDiagnostics>,
    stream: &CudaStream,
    tensor: &UploadedNvfp4,
    pending: PendingTensorUpdateDiagnostics,
) -> AppResult {
    let after_bytes = tensor.bytes.to_host_vec(stream)?;
    let after_scales = tensor.scales.to_host_vec(stream)?;
    let after_global = tensor.global_scale_to_host(stream)?;
    updates.push(tensor_update_stats(
        pending,
        after_bytes,
        after_scales,
        after_global,
    ));
    Ok(())
}
