use cuda_core::DriverError;
use rust_kernels_cuda::attention::{
    ApplyRopeArgs, CaptureValueResidualArgs, CausalAttentionTcArgs,
    ExclusiveSelfAttentionForwardArgs, HeadwiseAttentionGateForwardArgs, MixValueResidualArgs,
};
use rust_kernels_cuda::nvfp4_tma_matmul::{
    pad::U4RowPadArgs, scale_layout::sm120_scale_padded_mn_extent,
};

use super::tensors::AttentionForwardArgs;
use crate::types::HiddenStateDevice;
use crate::{
    AttentionDims, GPT2_FULL_ATTENTION_WINDOW, attention_headwise_gate_enabled,
    attention_headwise_gate_offset, partial_key_offset_enabled, selective_attention_enabled,
    uses_exclusive_self_attention,
};

pub(super) fn forward<'a, 'scratch>(
    args: AttentionForwardArgs<'a, 'scratch>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut input_nvfp4 = args.input_nvfp4;
    let mut tape = args.tape;
    let dims = AttentionDims::new(args.use_full_attention);
    let hidden = args.hidden;

    let input =
        input_nvfp4.quantize_hidden_precomputed(args.quant_module, &hidden, dims.embedding_dim)?;
    if let Some(tape) = tape.as_mut() {
        tape.save_qkv_input(hidden.stream, input)?;
    }

    let padded_qkv_dim = sm120_scale_padded_mn_extent(dims.qkv_dim as usize) as u32;
    args.tma_scale_pack.pack(
        hidden.stream,
        input.scales,
        args.tma_input_scale_packed,
        hidden.row_count,
        dims.embedding_dim,
    )?;
    args.tma_scale_pack.pack(
        hidden.stream,
        args.projections.qkv_weight_device.scales,
        args.tma_weight_scale_packed,
        dims.qkv_dim,
        dims.embedding_dim,
    )?;
    if padded_qkv_dim != dims.qkv_dim {
        args.tma_pad.pad_u4_rows(U4RowPadArgs {
            stream: hidden.stream,
            input: args.projections.qkv_weight_device.bytes,
            output: args.tma_weight_bytes_padded,
            rows: dims.qkv_dim,
            padded_rows: padded_qkv_dim,
            cols_u4: dims.embedding_dim,
        })?;
    }
    let qkv_weight_bytes = if padded_qkv_dim == dims.qkv_dim {
        args.projections.qkv_weight_device.bytes
    } else {
        &*args.tma_weight_bytes_padded
    };
    args.tma_module.prepare_tma_nvfp4_device_scales_into(
        hidden.stream,
        input.bytes,
        args.tma_input_scale_packed,
        qkv_weight_bytes,
        args.tma_weight_scale_packed,
        hidden.row_count,
        dims.embedding_dim,
        padded_qkv_dim,
        args.tma_descriptors,
    )?;
    args.tma_module
        .gemm_tma_nvfp4_rowwise_a_scale_affine_padded_output(
            hidden.stream,
            args.tma_descriptors,
            args.qkv,
            args.projections.qkv_bias,
            hidden.row_count,
            dims.embedding_dim,
            dims.qkv_dim,
            padded_qkv_dim,
            input.global_scales,
            args.projections.qkv_weight_device.global_scale,
        )?;

    if args.block_index == 0 {
        args.module
            .capture_value_residual(CaptureValueResidualArgs {
                stream: hidden.stream,
                qkv: &*args.qkv,
                first_value: args.value_residual,
                row_count: hidden.row_count,
                embedding_dim: dims.embedding_dim,
                qkv_dim: dims.qkv_dim,
            })?;
    } else if crate::uses_value_residual(args.block_index) {
        args.module.mix_value_residual(MixValueResidualArgs {
            stream: hidden.stream,
            qkv: args.qkv,
            first_value: &*args.value_residual,
            row_count: hidden.row_count,
            embedding_dim: dims.embedding_dim,
            qkv_dim: dims.qkv_dim,
        })?;
    }

    if args.use_full_attention {
        args.module.apply_rope(ApplyRopeArgs {
            stream: hidden.stream,
            qkv: args.qkv,
            qkv_f16: tape.as_mut().map(|tape| &mut *tape.qkv_f16),
            row_count: hidden.row_count,
            seq_len: hidden.seq_len,
            batch_size: hidden.batch_size,
            embedding_dim: dims.embedding_dim,
            qkv_dim: dims.qkv_dim,
            head_count: dims.head_count,
            head_dim: dims.head_dim,
        })?;
    }

    let (qkv_f16, attention_out_f16, attention_probs_f16, kda_v_new, kda_akk_inv, kda_w, kda_aqk) =
        match tape.as_mut() {
            Some(tape) if args.use_full_attention => (
                Some(&mut *tape.qkv_f16),
                Some(&mut *tape.attention_out_f16),
                tape.attention_probs_f16
                    .as_mut()
                    .map(|buffer| &mut **buffer),
                None,
                None,
                None,
                None,
            ),
            Some(tape) => (
                Some(&mut *tape.qkv_f16),
                Some(&mut *tape.attention_out_f16),
                None,
                tape.kda_v_new.as_mut().map(|buffer| &mut **buffer),
                tape.kda_akk_inv.as_mut().map(|buffer| &mut **buffer),
                tape.kda_w.as_mut().map(|buffer| &mut **buffer),
                tape.kda_aqk.as_mut().map(|buffer| &mut **buffer),
            ),
            None => (None, None, None, None, None, None, None),
        };

    let attention_args = CausalAttentionTcArgs {
        stream: hidden.stream,
        tc_module: args.tc_module,
        qkv: &*args.qkv,
        qk_scale: args.projections.qk_scale,
        out: &mut *hidden.normalized,
        qkv_f16,
        attention_out_f16,
        forward_probs_f16: attention_probs_f16,
        kda_v_new,
        kda_akk_inv,
        kda_w,
        kda_aqk,
        log_sum_exp: args.attention_log_sum_exp,
        scratch: args.tc_scratch,
        row_count: hidden.row_count,
        seq_len: hidden.seq_len,
        batch_size: hidden.batch_size,
        embedding_dim: dims.embedding_dim,
        qkv_dim: dims.qkv_dim,
        head_count: dims.head_count,
        head_dim: dims.head_dim,
        attention_window: if args.use_full_attention {
            GPT2_FULL_ATTENTION_WINDOW as u32
        } else {
            hidden.seq_len
        },
        partial_key_offset: args.use_full_attention && partial_key_offset_enabled(),
        selective_attention: args.use_full_attention && selective_attention_enabled(),
    };
    if args.use_full_attention {
        args.module.causal_attention_tc(attention_args)?;
    } else {
        args.module.kda_attention_tc(attention_args)?;
    }

    if uses_exclusive_self_attention(args.use_full_attention) {
        let (qkv_f16, raw_out_f16) = match tape.as_mut() {
            Some(tape) if args.use_full_attention => {
                (Some(&mut *tape.qkv_f16), Some(&mut *tape.attention_out_f16))
            }
            Some(tape) => (
                Some(&mut *tape.qkv_f16),
                Some(
                    &mut **tape
                        .headwise_gate_input_f16
                        .as_mut()
                        .expect("KDA XSA requires a raw-output tape"),
                ),
            ),
            None => (None, None),
        };
        args.module
            .exclusive_self_attention_forward(ExclusiveSelfAttentionForwardArgs {
                stream: hidden.stream,
                qkv: &*args.qkv,
                out: &mut *hidden.normalized,
                xsa_alphas: args.projections.xsa_alphas,
                qkv_f16,
                raw_out_f16,
                row_count: hidden.row_count,
                embedding_dim: dims.embedding_dim,
                qkv_dim: dims.qkv_dim,
                head_count: dims.head_count,
                head_dim: dims.head_dim,
                value_offset: 2 * dims.embedding_dim,
                gate_offset: attention_headwise_gate_offset(args.use_full_attention) as u32,
                alpha_offset: args.projections.xsa_alpha_offset,
                apply_headwise_gate: attention_headwise_gate_enabled(),
                kda_value_activation: !args.use_full_attention,
            })?;
    } else if attention_headwise_gate_enabled() {
        let (qkv_f16, raw_out_f16) = match tape.as_mut() {
            Some(tape) if args.use_full_attention => {
                (Some(&mut *tape.qkv_f16), Some(&mut *tape.attention_out_f16))
            }
            Some(tape) => (
                Some(&mut *tape.qkv_f16),
                Some(
                    &mut **tape
                        .headwise_gate_input_f16
                        .as_mut()
                        .expect("KDA headwise gate requires a raw-output tape"),
                ),
            ),
            None => (None, None),
        };
        args.module
            .headwise_attention_gate_forward(HeadwiseAttentionGateForwardArgs {
                stream: hidden.stream,
                qkv: &*args.qkv,
                out: &mut *hidden.normalized,
                qkv_f16,
                raw_out_f16,
                row_count: hidden.row_count,
                embedding_dim: dims.embedding_dim,
                qkv_dim: dims.qkv_dim,
                head_count: dims.head_count,
                head_dim: dims.head_dim,
                gate_offset: attention_headwise_gate_offset(args.use_full_attention) as u32,
            })?;
    }

    input_nvfp4.quantize_row_amax(
        args.quant_module,
        hidden.stream,
        &*hidden.normalized,
        &mut *hidden.normalized_amax,
        hidden.row_count,
        dims.embedding_dim,
    )?;

    let input = input_nvfp4.device();
    if let Some(tape) = tape.as_mut() {
        tape.save_c_proj_input(hidden.stream, input)?;
    }

    args.tma_scale_pack.pack(
        hidden.stream,
        input.scales,
        args.tma_input_scale_packed,
        hidden.row_count,
        dims.embedding_dim,
    )?;
    args.tma_scale_pack.pack(
        hidden.stream,
        args.projections.c_proj_weight_device.scales,
        args.tma_weight_scale_packed,
        dims.embedding_dim,
        dims.embedding_dim,
    )?;
    args.tma_module.prepare_tma_nvfp4_device_scales_into(
        hidden.stream,
        input.bytes,
        args.tma_input_scale_packed,
        args.projections.c_proj_weight_device.bytes,
        args.tma_weight_scale_packed,
        hidden.row_count,
        dims.embedding_dim,
        dims.embedding_dim,
        args.tma_descriptors,
    )?;
    args.tma_module.gemm_tma_nvfp4_rowwise_a_scale_residual(
        hidden.stream,
        args.tma_descriptors,
        &mut *hidden.residual,
        args.projections.c_proj_bias,
        hidden.row_count,
        dims.embedding_dim,
        dims.embedding_dim,
        input.global_scales,
        args.projections.c_proj_weight_device.global_scale,
        None,
    )?;

    Ok(hidden)
}
