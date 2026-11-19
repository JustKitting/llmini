//! Actual full-model GPU upload proof for the opt-in packed initializer.
use crate::{AppResult, training::Trainer, upload::UploadedNvfp4};
use cuda_core::CudaStream;
use gpt2_nvfp4::{MlpDownWeightShape, MlpUpWeightShape, Nvfp4Shape, Nvfp4Tensor};

fn uploaded_matches<S: Nvfp4Shape>(
    stream: &CudaStream,
    host: &Nvfp4Tensor<S>,
    device: &UploadedNvfp4,
    scale_k: usize,
    label: &str,
) -> AppResult<Vec<u8>> {
    assert_eq!(device.len, host.len(), "{label}: logical upload length");
    let bytes = device.bytes.to_host_vec(stream)?;
    assert!(
        bytes == host.bytes.as_ref(),
        "{label}: packed GPU payload differs"
    );
    let expected_scales = rust_kernels_cuda::nvfp4::csf::encode_host(
        host.scales.as_ref(),
        host.len().div_ceil(scale_k),
        scale_k,
    );
    assert!(
        device.scales.to_host_vec(stream)? == expected_scales,
        "{label}: compressed GPU scales differ"
    );
    assert_eq!(
        device.global_scale_to_host(stream)?.to_bits(),
        host.global_scale.to_bits(),
        "{label}: GPU global scale differs"
    );
    Ok(bytes)
}
fn source_rms(bytes: &[u8], label: &str) {
    const SQUARE: [f64; 8] = [0., 0.25, 1., 2.25, 4., 9., 16., 36.];
    let square = bytes
        .iter()
        .map(|byte| SQUARE[usize::from(byte & 7)] + SQUARE[usize::from((byte >> 4) & 7)])
        .sum::<f64>();
    let rms = (square / (bytes.len() * 2) as f64).sqrt() * 0.02;
    assert!(
        (0.019..=0.021).contains(&rms),
        "{label}: GPU source-normal RMS={rms}"
    );
}
pub(in crate::training) fn verify_source_normal_upload(t: &Trainer) -> AppResult {
    assert_eq!(
        gpt2_nvfp4::MlpWeights::initializer_name(),
        "reference-normal-nvfp4"
    );
    assert!(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU);
    assert_eq!(t.uploaded.blocks.len(), 16);
    let s = t.runtime.stream.as_ref();
    let initial = t.model.weights().expect("fresh initialized model");
    for (block, (host, device)) in initial.h.iter().zip(&t.uploaded.blocks).enumerate() {
        for (part, bytes) in [
            (
                "paired",
                uploaded_matches(
                    s,
                    &host.mlp.c_fc.weight,
                    &device.mlp_up.weight,
                    MlpUpWeightShape::ROWS,
                    &format!("block{block}.paired"),
                )?,
            ),
            (
                "down",
                uploaded_matches(
                    s,
                    &host.mlp.c_proj.weight,
                    &device.mlp_down.weight,
                    MlpDownWeightShape::ROWS,
                    &format!("block{block}.down"),
                )?,
            ),
        ] {
            if part == "paired" {
                let half = bytes.len() / 2;
                source_rms(&bytes[..half], &format!("block{block}.gate"));
                source_rms(&bytes[half..], &format!("block{block}.up"));
            } else {
                source_rms(&bytes, &format!("block{block}.down"));
            }
        }
        assert_eq!(host.mlp.c_fc.weight.global_scale, 0.02);
        assert_eq!(host.mlp.c_proj.weight.global_scale, 0.02);
        let up_bias = uploaded_matches(
            s,
            &host.mlp.c_fc.bias,
            &device.mlp_up.bias,
            gpt2_nvfp4::GPT2_MLP_UP.max(128).next_power_of_two(),
            &format!("block{block}.paired_bias"),
        )?;
        let down_bias = uploaded_matches(
            s,
            &host.mlp.c_proj.bias,
            &device.mlp_down.bias,
            gpt2_nvfp4::GPT2_N_EMBD.max(128).next_power_of_two(),
            &format!("block{block}.down_bias"),
        )?;
        assert!(up_bias.iter().chain(&down_bias).all(|byte| *byte == 0));
    }
    println!(
        "source_normal_full_model_upload_verified blocks=16 tensors=64 weight_moment_checks=48 packed_payload_scales_global=bit_exact zero_bias=true optimizer_updates=0 dense=NVFP4"
    );
    Ok(())
}
