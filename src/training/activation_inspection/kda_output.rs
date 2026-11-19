use cuda_core::CudaStream;
use gpt2_nvfp4::{
    BlockForwardSaved, GPT2_KDA_BETA_OFFSET, GPT2_KDA_GATE_OFFSET, GPT2_N_EMBD, GPT2_N_HEAD,
    GPT2_QKV, attention_headwise_gate_enabled,
};
use rust_kernels_cuda::nvfp4::csf;
use serde_json::{Value, json};

use super::{Moments, distribution, f16, fp4, fp8, rowwise_scale_header};
use crate::AppResult;

pub(super) fn inspect(
    stream: &CudaStream,
    block: &BlockForwardSaved<'_>,
    qkv: &[u16],
    rows: &[usize],
) -> AppResult<Value> {
    // KDA's attention_out is the chunk-state tape. The XSA/head-gate tape
    // saves the actual output before either operation.
    let raw_output = block
        .headwise_gate_input
        .map(|source| source.to_host_vec(stream))
        .transpose()?;
    let input = block.c_proj_input_nvfp4;
    let packed = input.bytes.to_host_vec(stream)?;
    let globals = input.global_scales.to_host_vec(stream)?;
    let scales = rowwise_scale_header(
        GPT2_N_EMBD,
        globals.len(),
        &input.scales.to_host_vec(stream)?,
    )?;

    let mut raw = Moments::default();
    let mut represented = Moments::default();
    let mut beta_logits = Vec::new();
    let mut gate_logits = Vec::new();
    let gate_enabled = attention_headwise_gate_enabled();
    for &row in rows {
        let hidden = row * GPT2_N_EMBD;
        if let Some(output) = &raw_output {
            for &bits in &output[hidden..hidden + GPT2_N_EMBD] {
                raw.add(f64::from(f16(bits)));
            }
        }
        for group in hidden / 16..(hidden + GPT2_N_EMBD) / 16 {
            let scale = f64::from(globals[row]) * fp8(csf::scale_at(&scales, group));
            for &byte in &packed[group * 8..(group + 1) * 8] {
                represented.add(fp4(byte & 15) * scale);
                represented.add(fp4(byte >> 4) * scale);
            }
        }
        let qkv_row = row * GPT2_QKV;
        for head in 0..GPT2_N_HEAD {
            beta_logits.push(f64::from(f16(qkv[qkv_row + GPT2_KDA_BETA_OFFSET + head])));
            if gate_enabled {
                gate_logits.push(f64::from(f16(qkv[qkv_row + GPT2_KDA_GATE_OFFSET + head])));
            }
        }
    }

    Ok(json!({
        "output_before_xsa_and_gate_fp16": raw_output.is_some().then(|| raw.json()),
        "output_projection_input_nvfp4": represented.json(),
        "beta_logits_fp16": distribution(&mut beta_logits),
        "head_gate_logits_fp16": gate_enabled.then(|| distribution(&mut gate_logits)),
    }))
}
