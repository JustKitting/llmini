//! Saved activation policy identity; metadata only, no model parameters.
use super::format::{CheckpointTensor, CheckpointWriter};
use crate::AppResult;
use std::{collections::HashMap, io::Write};

const DESCRIPTOR: [f32; 8] = [1., 1., 16., 4., 8., 256., 1., 1.];

pub(super) fn extra_tensors() -> u32 {
    rust_kernels_cuda::nvfp4_quant::ACTIVATION_MAE_FOUR_SIX as u32
}

pub(super) fn write(writer: &mut CheckpointWriter<impl Write>) -> AppResult {
    if rust_kernels_cuda::nvfp4_quant::ACTIVATION_MAE_FOUR_SIX {
        let bytes = DESCRIPTOR
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        writer.write_tensor("activation_quant.config", DESCRIPTOR.len(), 1., &bytes, &[])?;
    }
    Ok(())
}

pub(super) fn take_and_validate(tensors: &mut HashMap<String, CheckpointTensor>) -> AppResult {
    let enabled = rust_kernels_cuda::nvfp4_quant::ACTIVATION_MAE_FOUR_SIX;
    match tensors.remove("activation_quant.config") {
        None if !enabled => Ok(()),
        Some(t)
            if enabled
                && t.len == DESCRIPTOR.len()
                && t.bytes.len() == DESCRIPTOR.len() * 4
                && t.scales.is_empty()
                && t.global_scale == 1.
                && t.bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .eq(DESCRIPTOR) =>
        {
            Ok(())
        }
        _ => Err(
            "checkpoint activation quantization policy does not match this native executable"
                .into(),
        ),
    }
}
