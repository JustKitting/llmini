pub(super) fn tensor_count(block_count: usize) -> u32 {
    const TOKEN_EMBEDDING: u32 = 1;
    const XSA_ALPHAS: u32 = 1;
    const FINAL_NORM: u32 = 2;
    const NEXT_LATENT: u32 = 8;
    const BLOCK: u32 = 15;

    TOKEN_EMBEDDING + XSA_ALPHAS + FINAL_NORM + NEXT_LATENT + block_count as u32 * BLOCK
}

pub(super) fn supported_tensor_count(block_count: usize, count: u32) -> bool {
    count
        .checked_sub(
            super::activation_quant::extra_tensors() + super::kda_output_gate::extra_tensors(),
        )
        .is_some_and(|count| supported_graph_tensor_count(block_count, count))
}

fn supported_graph_tensor_count(block_count: usize, count: u32) -> bool {
    let base = tensor_count(block_count);
    let mhc = 1 + 3 * (2 * block_count as u32 + 1);
    let four_stream_engram = 1
        + (0..crate::training::engram::LAYERS.len())
            .map(|i| {
                crate::training::engram::Config::reference_multistream(i)
                    .nv_names()
                    .len() as u32
                    + 6
            })
            .sum::<u32>();
    count == base
        || count == base + 1 // Explicit standalone numerical or backward-graph descriptor.
        || count == base + mhc
        || count == base + mhc + four_stream_engram
        || count == base + mhc + four_stream_engram + 1 // Full KDA credit descriptor.
        || (gpt2_nvfp4::GPT2_REQUIRES_ENGRAM_MHC && count == base + mhc + four_stream_engram + 2)
        || (gpt2_nvfp4::GPT2_REQUIRES_ENGRAM_MHC
            && (u32::from(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU) + u32::from(gpt2_nvfp4::GPT2_VALUE_RMS) + u32::from(gpt2_nvfp4::GPT2_STATIONARY_ROPE)) >= 2
            && count == base + mhc + four_stream_engram + 3)
        || (gpt2_nvfp4::GPT2_BOUNDED_SWIGLU_ENGRAM_MHC && gpt2_nvfp4::GPT2_VALUE_RMS_ENGRAM_MHC && gpt2_nvfp4::GPT2_STATIONARY_ROPE_ENGRAM_MHC
            && count == base + mhc + four_stream_engram + 4)
        || (gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
            && count == base + mhc + four_stream_engram
                + super::kda_backward::extra_tensors()
                + super::mlp::extra_tensors()
                + super::value_rms::extra_tensors()
                + super::stationary_rope::extra_tensors()
                + super::latent_feedback::TENSOR_COUNT)
        || count == base + 11
        || count == base + super::latent_feedback::TENSOR_COUNT
        || (count > base + 1
            && (count - base - 1) % 11 == 0
            && (count - base - 1) / 11 <= block_count as u32)
}

#[cfg(test)]
mod tests {
    use super::{supported_graph_tensor_count as supported_tensor_count, tensor_count};
    #[test]
    fn complete_four_stream_engram_mhc_header_matches_native_writer() {
        let layers = 16;
        let base = tensor_count(layers);
        let mhc = 1 + 3 * (2 * layers as u32 + 1);
        let engram = 1
            + (0..crate::training::engram::LAYERS.len())
                .map(|i| {
                    crate::training::engram::Config::reference_multistream(i)
                        .nv_names()
                        .len() as u32
                        + 6
                })
                .sum::<u32>();
        assert_eq!(
            (base, mhc, engram, base + mhc + engram),
            (252, 100, 35, 387)
        );
        assert!(
            supported_tensor_count(layers, base + mhc + engram),
            "native writer's complete version4 Engram+mHC header was rejected before any tensors could load"
        );
        for count in [base, base + 2, base + 11, base + 23, base + mhc] {
            assert!(
                supported_tensor_count(layers, count),
                "legacy graph header {count}"
            );
        }
        for count in [0, base - 1, base + 3, base + mhc + engram + 7, u32::MAX] {
            assert!(
                !supported_tensor_count(layers, count),
                "unknown header {count}"
            );
        }
    }
    #[test]
    fn full_credit_combined_header_matches_the_real_writer_count() {
        //252 backbone +100 mHC +35 Engram +1 KDA semantic descriptor.
        assert!(
            supported_tensor_count(16, 388),
            "complete full-credit Engram/mHC writer header"
        );
        assert_eq!(
            supported_tensor_count(16, 389),
            gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
                && (gpt2_nvfp4::GPT2_BOUNDED_SWIGLU
                    || gpt2_nvfp4::GPT2_VALUE_RMS
                    || gpt2_nvfp4::GPT2_STATIONARY_ROPE)
        );
        assert_eq!(
            supported_tensor_count(16, 390),
            gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
                && (u32::from(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU)
                    + u32::from(gpt2_nvfp4::GPT2_VALUE_RMS)
                    + u32::from(gpt2_nvfp4::GPT2_STATIONARY_ROPE))
                    >= 2
        );
        assert_eq!(
            supported_tensor_count(16, 391),
            gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
                && gpt2_nvfp4::GPT2_BOUNDED_SWIGLU
                && gpt2_nvfp4::GPT2_VALUE_RMS
                && gpt2_nvfp4::GPT2_STATIONARY_ROPE
        );
        let feedback_count = 387
            + super::super::kda_backward::extra_tensors()
            + super::super::mlp::extra_tensors()
            + super::super::value_rms::extra_tensors()
            + super::super::stationary_rope::extra_tensors()
            + super::super::latent_feedback::TENSOR_COUNT;
        if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
            assert!(supported_tensor_count(16, feedback_count));
        }
        assert!(!supported_tensor_count(16, 394));
    }
}
