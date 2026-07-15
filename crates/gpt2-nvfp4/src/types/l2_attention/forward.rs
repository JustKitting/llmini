use cuda_core::DriverError;
use rust_kernels_cuda::attention::{ApplyRopeArgs, CausalAttentionTcArgs};
use rust_kernels_cuda::nvfp4_tma_matmul::{
    pad::U4RowPadArgs, scale_layout::sm120_scale_padded_mn_extent,
};
use rust_kernels_cuda::projection_postop::ProjectionResidualArgs;

use super::tensors::AttentionForwardArgs;
use crate::AttentionDims;
use crate::types::HiddenStateDevice;

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

    let (qkv_f16, attention_out_f16) = match tape.as_mut() {
        Some(tape) if args.use_full_attention => (None, Some(&mut *tape.attention_out_f16)),
        Some(tape) => (Some(&mut *tape.qkv_f16), Some(&mut *tape.attention_out_f16)),
        None => (None, None),
    };

    let attention_args = CausalAttentionTcArgs {
        stream: hidden.stream,
        tc_module: args.tc_module,
        qkv: &*args.qkv,
        out: &mut *hidden.normalized,
        qkv_f16,
        attention_out_f16,
        log_sum_exp: args.attention_log_sum_exp,
        scratch: args.tc_scratch,
        row_count: hidden.row_count,
        seq_len: hidden.seq_len,
        batch_size: hidden.batch_size,
        embedding_dim: dims.embedding_dim,
        qkv_dim: dims.qkv_dim,
        head_count: dims.head_count,
        head_dim: dims.head_dim,
    };
    if args.use_full_attention {
        args.module.causal_attention_tc(attention_args)?;
    } else {
        args.module.kda_attention_tc(attention_args)?;
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
    args.tma_module
        .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer(
            hidden.stream,
            args.tma_descriptors,
            args.tma_residual,
            hidden.row_count,
            dims.embedding_dim,
            dims.embedding_dim,
            input.global_scales,
            args.projections.c_proj_weight_device.global_scale,
        )?;
    args.projection_postop
        .residual_add(ProjectionResidualArgs {
            stream: hidden.stream,
            raw: &*args.tma_residual,
            bias: args.projections.c_proj_bias,
            residual: &mut *hidden.residual,
            rows: hidden.row_count,
            cols: dims.embedding_dim,
        })?;

    Ok(hidden)
}
