use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::f32_matrix_ops::{
    F32Linear3Args, F32Linear3SqrtBoundAmaxArgs, F32Linear3SqrtBoundRowSumsqArgs,
    F32ScaleInPlaceByAmaxArgs,
};
use rust_kernels_cuda::nvfp4_quant::{
    Nvfp4QuantPaddedArgs, Nvfp4QuantPairTransposeExactArgs, Nvfp4QuantTransposePaddedArgs,
    TensorAmaxArgs,
};
use rust_kernels_cuda::nvfp4_tma_matmul::kernels::{TILE_K, TILE_M, TILE_N};
use rust_kernels_cuda::nvfp4_tma_matmul::pad::F32CropArgs;
use rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors;
use rust_kernels_cuda::optimizer::{
    MuonSlotDescriptor, MuonTmaFinishArgs, MuonTmaPrepareArgs, muon_polar_coefficients,
};

use super::{MU, MUON_WEIGHT_DECAY, MuonGroupTable, POLAR_ITERATIONS, muon_learning_rate};
use crate::training::env::{env_bool, env_usize};
use crate::training::optimizer_tc_scratch::{
    MuonScratchBuffers, MuonTmaOperandScratch, MuonTmaScratch,
};
use crate::training::runtime::Runtime;

pub(in crate::training) struct MuonTmaArgs<'a> {
    pub(in crate::training) runtime: &'a Runtime,
    pub(in crate::training) table: &'a MuonGroupTable,
    pub(in crate::training) scratch: &'a mut MuonScratchBuffers,
    pub(in crate::training) slot_count: usize,
    pub(in crate::training) step: u32,
    pub(in crate::training) average_coefficient: f32,
    pub(in crate::training) grad_scale: f32,
}

pub(in crate::training) fn apply_muon_tma(args: MuonTmaArgs<'_>) -> Result<(), DriverError> {
    let stream = args.runtime.stream.as_ref();
    let learning_rate = muon_learning_rate(args.step);
    let schedule_beta = super::super::learning_rate::schedule_free_beta(args.step + 1);
    let trace = TmaTraceConfig::from_env();
    for slot_index in 0..args.slot_count {
        let desc = args.table.host_slots[slot_index];
        if desc.rows == 0 || desc.cols == 0 {
            continue;
        }

        let polar_x_amax_chunks =
            args.runtime
                .optimizer
                .muon_tma_prepare_polar(MuonTmaPrepareArgs {
                    stream,
                    slots: &args.table.slots,
                    oriented: &mut args.scratch.oriented,
                    polar_x: &mut args.scratch.polar_x,
                    polar_chunks: &mut args.scratch.polar_chunks,
                    polar_x_chunk_amax: &mut args.scratch.tma.a.chunk_amax,
                    slot_index: slot_index as u32,
                    matrix_len: desc.rows * desc.cols,
                    mu: MU,
                    grad_scale: args.grad_scale,
                })?;
        args.runtime.quant.tensor_amax_from_chunks_f32(
            stream,
            &args.scratch.tma.a.chunk_amax,
            &mut args.scratch.tma.a.amax,
            polar_x_amax_chunks,
        )?;

        let (polar_rows, polar_cols) = polar_shape(desc);
        let defer_bounds = can_defer_polar_bounds(polar_rows, polar_cols);
        trace_buffer(
            stream,
            trace,
            slot_index,
            "prepared_x",
            &args.scratch.polar_x,
            polar_rows * polar_cols,
            None,
            desc,
        )?;

        run_tma_polar_loop(
            stream,
            args.runtime,
            args.scratch,
            desc,
            slot_index,
            trace,
            defer_bounds,
        )?;

        if POLAR_ITERATIONS & 1 == 0 {
            args.runtime
                .optimizer
                .muon_tma_finish_update_deferred_quantization(MuonTmaFinishArgs {
                    stream,
                    slots: &args.table.slots,
                    polar_update: &args.scratch.polar_x,
                    polar_bound_amax: &args.scratch.tma.bound_amax,
                    polar_chunks: &mut args.scratch.polar_chunks,
                    slot_index: slot_index as u32,
                    matrix_len: desc.rows * desc.cols,
                    learning_rate,
                    weight_decay: MUON_WEIGHT_DECAY,
                    average_coefficient: args.average_coefficient,
                    schedule_beta,
                    apply_polar_sqrt_bound: defer_bounds as u32,
                })?;
        } else {
            args.runtime
                .optimizer
                .muon_tma_finish_update_deferred_quantization(MuonTmaFinishArgs {
                    stream,
                    slots: &args.table.slots,
                    polar_update: &args.scratch.polar_next,
                    polar_bound_amax: &args.scratch.tma.bound_amax,
                    polar_chunks: &mut args.scratch.polar_chunks,
                    slot_index: slot_index as u32,
                    matrix_len: desc.rows * desc.cols,
                    learning_rate,
                    weight_decay: MUON_WEIGHT_DECAY,
                    average_coefficient: args.average_coefficient,
                    schedule_beta,
                    apply_polar_sqrt_bound: defer_bounds as u32,
                })?;
        }
    }
    Ok(())
}

fn run_tma_polar_loop(
    stream: &CudaStream,
    runtime: &Runtime,
    scratch: &mut MuonScratchBuffers,
    desc: MuonSlotDescriptor,
    slot_index: usize,
    trace: TmaTraceConfig,
    defer_bounds: bool,
) -> Result<(), DriverError> {
    let (polar_rows, polar_cols) = polar_shape(desc);
    for iter in 0..POLAR_ITERATIONS {
        if iter & 1 == 0 {
            run_tma_polar_iteration(
                stream,
                runtime,
                &mut scratch.polar_x,
                &mut scratch.polar_next,
                &mut scratch.polar_gram,
                &mut scratch.polar_ax,
                tma_refs(&mut scratch.tma),
                polar_rows,
                polar_cols,
                iter,
                slot_index,
                trace,
                desc,
                defer_bounds,
            )?;
        } else {
            run_tma_polar_iteration(
                stream,
                runtime,
                &mut scratch.polar_next,
                &mut scratch.polar_x,
                &mut scratch.polar_gram,
                &mut scratch.polar_ax,
                tma_refs(&mut scratch.tma),
                polar_rows,
                polar_cols,
                iter,
                slot_index,
                trace,
                desc,
                defer_bounds,
            )?;
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "polar loop has explicit matrix buffers"
)]
fn run_tma_polar_iteration(
    stream: &CudaStream,
    runtime: &Runtime,
    source: &mut DeviceBuffer<f32>,
    target: &mut DeviceBuffer<f32>,
    gram: &mut DeviceBuffer<f32>,
    ax: &mut DeviceBuffer<f32>,
    tma: TmaScratchRefs<'_>,
    polar_rows: u32,
    polar_cols: u32,
    iter: u32,
    slot_index: usize,
    trace: TmaTraceConfig,
    desc: MuonSlotDescriptor,
    defer_bounds: bool,
) -> Result<(), DriverError> {
    let mut tma = tma;
    tma_matmul_self_transpose(
        stream,
        runtime,
        source,
        gram,
        tma.reborrow(),
        polar_rows,
        polar_cols,
        iter == 0 || defer_bounds,
        defer_bounds,
    )?;
    trace_buffer(
        stream,
        trace,
        slot_index,
        "gram_xxt",
        gram,
        polar_rows * polar_rows,
        Some(iter),
        desc,
    )?;
    bound_source_and_gram(
        stream,
        runtime,
        source,
        gram,
        tma.reborrow(),
        polar_rows,
        polar_cols,
        defer_bounds,
    )?;
    trace_buffer(
        stream,
        trace,
        slot_index,
        if defer_bounds {
            "source_bound_deferred"
        } else {
            "source_bounded"
        },
        source,
        polar_rows * polar_cols,
        Some(iter),
        desc,
    )?;
    trace_buffer(
        stream,
        trace,
        slot_index,
        if defer_bounds {
            "gram_bound_deferred"
        } else {
            "gram_bounded"
        },
        gram,
        polar_rows * polar_rows,
        Some(iter),
        desc,
    )?;

    let coeffs = muon_polar_coefficients(iter);
    let action_dims = TmaDims::new(polar_rows, polar_cols, polar_rows);
    prepare_tma_a_padded(
        stream,
        runtime,
        gram,
        &mut tma,
        action_dims,
        polar_rows,
        polar_rows,
        defer_bounds,
    )?;
    if defer_bounds {
        runtime.quant.rebase_four_six_global_scale_sqrt_bound(
            stream,
            &*tma.a.amax,
            &*tma.bound_amax,
            tma.b.global_scale,
        )?;
    } else {
        prepare_tma_b_sqrt_bounded_transposed(
            stream,
            runtime,
            source,
            &mut tma,
            action_dims,
            polar_rows,
            polar_cols,
            false,
        )?;
    }
    run_tma_gemm_prepared_and_b_amax(stream, runtime, ax, &mut tma, action_dims)?;
    trace_buffer(
        stream,
        trace,
        slot_index,
        "ax_gx",
        ax,
        polar_rows * polar_cols,
        Some(iter),
        desc,
    )?;
    prepare_tma_b_transposed_precomputed_amax(
        stream,
        runtime,
        ax,
        &mut tma,
        action_dims,
        polar_rows,
        polar_cols,
    )?;
    run_tma_gemm_prepared(stream, runtime, target, &mut tma, action_dims)?;
    trace_buffer(
        stream,
        trace,
        slot_index,
        "target_ggx",
        target,
        polar_rows * polar_cols,
        Some(iter),
        desc,
    )?;
    let final_iteration = iter + 1 == POLAR_ITERATIONS;
    if defer_bounds && final_iteration {
        runtime
            .f32_ops
            .linear3_sqrt_bound_a_row_sumsq(F32Linear3SqrtBoundRowSumsqArgs {
                stream,
                a: source,
                b: ax,
                c_out: target,
                bound_amax: &*tma.bound_amax,
                row_sumsq: tma.b.chunk_amax,
                rows: polar_rows,
                cols: polar_cols,
                a_scale: coeffs.a,
                b_scale: coeffs.b,
                c_scale: coeffs.c,
            })?;
        // For a Gram matrix XX^T, every off-diagonal magnitude is bounded by
        // the largest diagonal, so its tensor amax is the maximum row sumsq.
        runtime.quant.tensor_amax_from_chunks_f32(
            stream,
            &*tma.b.chunk_amax,
            tma.bound_amax,
            polar_rows,
        )?;
    } else if defer_bounds {
        let chunk_count =
            runtime
                .f32_ops
                .linear3_sqrt_bound_a_with_amax(F32Linear3SqrtBoundAmaxArgs {
                    stream,
                    a: source,
                    b: ax,
                    c_out: target,
                    bound_amax: &*tma.bound_amax,
                    chunk_amax: tma.a.chunk_amax,
                    len: polar_rows * polar_cols,
                    a_scale: coeffs.a,
                    b_scale: coeffs.b,
                    c_scale: coeffs.c,
                })?;
        runtime.quant.tensor_amax_from_chunks_f32(
            stream,
            &*tma.a.chunk_amax,
            tma.a.amax,
            chunk_count,
        )?;
    } else {
        runtime.f32_ops.linear3(F32Linear3Args {
            stream,
            a: source,
            b: ax,
            c_out: target,
            len: polar_rows * polar_cols,
            a_scale: coeffs.a,
            b_scale: coeffs.b,
            c_scale: coeffs.c,
        })?;
    }
    trace_buffer(
        stream,
        trace,
        slot_index,
        "next",
        target,
        polar_rows * polar_cols,
        Some(iter),
        desc,
    )?;
    if final_iteration && !defer_bounds {
        tma_matmul_self_transpose(
            stream,
            runtime,
            target,
            gram,
            tma.reborrow(),
            polar_rows,
            polar_cols,
            false,
            false,
        )?;
        trace_buffer(
            stream,
            trace,
            slot_index,
            "final_gram_xxt",
            gram,
            polar_rows * polar_rows,
            Some(iter),
            desc,
        )?;
        bound_source_and_gram(
            stream,
            runtime,
            target,
            gram,
            tma.reborrow(),
            polar_rows,
            polar_cols,
            false,
        )?;
        trace_buffer(
            stream,
            trace,
            slot_index,
            "final_next_bounded",
            target,
            polar_rows * polar_cols,
            Some(iter),
            desc,
        )?;
    } else if final_iteration {
        trace_buffer(
            stream,
            trace,
            slot_index,
            "final_next_bound_deferred",
            target,
            polar_rows * polar_cols,
            Some(iter),
            desc,
        )?;
    }
    Ok(())
}

fn bound_source_and_gram(
    stream: &CudaStream,
    runtime: &Runtime,
    source: &mut DeviceBuffer<f32>,
    gram: &mut DeviceBuffer<f32>,
    tma: TmaScratchRefs<'_>,
    polar_rows: u32,
    polar_cols: u32,
    defer_bounds: bool,
) -> Result<(), DriverError> {
    let tma = tma;
    if defer_bounds {
        return Ok(());
    }
    runtime
        .f32_ops
        .scale_in_place_by_sqrt_amax_bound(F32ScaleInPlaceByAmaxArgs {
            stream,
            x: source,
            amax: &*tma.bound_amax,
            len: polar_rows * polar_cols,
        })?;
    runtime
        .f32_ops
        .scale_in_place_by_amax_bound(F32ScaleInPlaceByAmaxArgs {
            stream,
            x: gram,
            amax: &*tma.bound_amax,
            len: polar_rows * polar_rows,
        })
}

fn tma_matmul_self_transpose(
    stream: &CudaStream,
    runtime: &Runtime,
    x: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    mut tma: TmaScratchRefs<'_>,
    rows: u32,
    k: u32,
    source_amax_precomputed: bool,
    prepare_source_transpose: bool,
) -> Result<(), DriverError> {
    let dims = TmaDims::new(rows, rows, k);
    if source_amax_precomputed && prepare_source_transpose {
        quantize_operand_pair_padded_with_amax(
            stream,
            runtime,
            x,
            tma.a.reborrow(),
            tma.b.reborrow(),
            rows,
            k,
            dims.m,
            dims.k,
        )?;
    } else if source_amax_precomputed {
        quantize_operand_padded_with_amax(
            stream,
            runtime,
            x,
            tma.a.reborrow(),
            rows,
            k,
            dims.m,
            dims.k,
        )?;
    } else {
        quantize_operand_padded(
            stream,
            runtime,
            x,
            tma.a.reborrow(),
            rows,
            k,
            dims.m,
            dims.k,
        )?;
    }
    run_tma_gemm_self_prepared_and_bound_amax(stream, runtime, out, &mut tma, dims)
}

fn prepare_tma_a_padded(
    stream: &CudaStream,
    runtime: &Runtime,
    input: &DeviceBuffer<f32>,
    tma: &mut TmaScratchRefs<'_>,
    dims: TmaDims,
    rows: u32,
    cols: u32,
    defer_bound: bool,
) -> Result<(), DriverError> {
    if rows != dims.m || cols != dims.k {
        return quantize_operand_padded(
            stream,
            runtime,
            input,
            tma.a.reborrow(),
            rows,
            cols,
            dims.m,
            dims.k,
        );
    }

    let args = Nvfp4QuantPaddedArgs {
        stream,
        x: input,
        amax: &*tma.bound_amax,
        out_fp4: tma.a.bytes,
        out_scales: tma.a.scale_packed,
        out_global_scale: tma.a.global_scale,
        rows,
        cols,
        padded_rows: dims.m,
        padded_cols: dims.k,
    };
    if defer_bound {
        runtime
            .quant
            .fp32_to_nvfp4_four_six_exact_lazy_bounded_amax_packed_scales(args)
    } else {
        runtime
            .quant
            .fp32_to_nvfp4_four_six_exact_bounded_amax_packed_scales(args)
    }
}

fn prepare_tma_b_transposed(
    stream: &CudaStream,
    runtime: &Runtime,
    input: &DeviceBuffer<f32>,
    tma: &mut TmaScratchRefs<'_>,
    dims: TmaDims,
    rows: u32,
    cols: u32,
) -> Result<(), DriverError> {
    quantize_operand_transposed_padded(
        stream,
        runtime,
        input,
        tma.b.reborrow(),
        rows,
        cols,
        dims.n,
        dims.k,
    )
}

fn prepare_tma_b_transposed_precomputed_amax(
    stream: &CudaStream,
    runtime: &Runtime,
    input: &DeviceBuffer<f32>,
    tma: &mut TmaScratchRefs<'_>,
    dims: TmaDims,
    rows: u32,
    cols: u32,
) -> Result<(), DriverError> {
    quantize_operand_transposed_padded_with_amax(
        stream,
        runtime,
        input,
        tma.b.reborrow(),
        rows,
        cols,
        dims.n,
        dims.k,
    )
}

fn prepare_tma_b_sqrt_bounded_transposed(
    stream: &CudaStream,
    runtime: &Runtime,
    input: &DeviceBuffer<f32>,
    tma: &mut TmaScratchRefs<'_>,
    dims: TmaDims,
    rows: u32,
    cols: u32,
    defer_bound: bool,
) -> Result<(), DriverError> {
    if cols != dims.n
        || rows != dims.k
        || !rows.is_power_of_two()
        || !rows.is_multiple_of(16)
        || !cols.is_multiple_of(64)
    {
        return prepare_tma_b_transposed(stream, runtime, input, tma, dims, rows, cols);
    }

    let args = Nvfp4QuantTransposePaddedArgs {
        stream,
        x: input,
        amax: &*tma.a.amax,
        out_fp4: tma.b.bytes,
        out_scales: tma.b.scales,
        out_global_scale: tma.b.global_scale,
        source_rows: rows,
        source_cols: cols,
        padded_rows: dims.n,
        padded_cols: dims.k,
    };
    if defer_bound {
        runtime
            .quant
            .fp32_transpose_to_nvfp4_four_six_exact_lazy_sqrt_bounded_amax(
                args,
                &*tma.bound_amax,
            )?;
    } else {
        runtime
            .quant
            .fp32_transpose_to_nvfp4_four_six_exact_sqrt_bounded_amax(args, &*tma.bound_amax)?;
    }
    runtime.optimizer.tma_scale_pack().pack(
        stream,
        &*tma.b.scales,
        tma.b.scale_packed,
        dims.n,
        dims.k,
    )
}

fn run_tma_gemm_prepared(
    stream: &CudaStream,
    runtime: &Runtime,
    out: &mut DeviceBuffer<f32>,
    tma: &mut TmaScratchRefs<'_>,
    dims: TmaDims,
) -> Result<(), DriverError> {
    if dims.is_exact() {
        runtime
            .optimizer
            .tma_gemm()
            .prepare_tma_nvfp4_device_scales_into(
                stream,
                &*tma.a.bytes,
                &*tma.a.scale_packed,
                &*tma.b.bytes,
                &*tma.b.scale_packed,
                dims.m,
                dims.k,
                dims.n,
                tma.descriptors,
            )?;
        return runtime
            .optimizer
            .tma_gemm()
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
                stream,
                &*tma.descriptors,
                out,
                dims.m,
                dims.k,
                dims.n,
                &*tma.a.global_scale,
                &*tma.b.global_scale,
            );
    }
    runtime
        .optimizer
        .tma_gemm()
        .prepare_tma_nvfp4_device_scales_into(
            stream,
            &*tma.a.bytes,
            &*tma.a.scale_packed,
            &*tma.b.bytes,
            &*tma.b.scale_packed,
            dims.m,
            dims.k,
            dims.n,
            tma.descriptors,
        )?;
    runtime
        .optimizer
        .tma_gemm()
        .gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
            stream,
            &*tma.descriptors,
            tma.out_padded,
            dims.m,
            dims.k,
            dims.n,
            &*tma.a.global_scale,
            &*tma.b.global_scale,
        )?;
    crop_tma_out(stream, runtime, out, tma.out_padded, dims)
}

fn run_tma_gemm_prepared_and_b_amax(
    stream: &CudaStream,
    runtime: &Runtime,
    out: &mut DeviceBuffer<f32>,
    tma: &mut TmaScratchRefs<'_>,
    dims: TmaDims,
) -> Result<(), DriverError> {
    if !dims.is_exact() {
        run_tma_gemm_prepared(stream, runtime, out, tma, dims)?;
        return runtime.quant.tensor_amax_f32(TensorAmaxArgs {
            stream,
            x: out,
            chunk_amax: tma.b.chunk_amax,
            out: tma.b.amax,
            element_count: dims.logical_m * dims.logical_n,
        });
    }

    runtime
        .optimizer
        .tma_gemm()
        .prepare_tma_nvfp4_device_scales_into(
            stream,
            &*tma.a.bytes,
            &*tma.a.scale_packed,
            &*tma.b.bytes,
            &*tma.b.scale_packed,
            dims.m,
            dims.k,
            dims.n,
            tma.descriptors,
        )?;
    let chunk_count = runtime
        .optimizer
        .tma_gemm()
        .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_with_output_amax(
            stream,
            &*tma.descriptors,
            out,
            tma.b.chunk_amax,
            dims.m,
            dims.k,
            dims.n,
            &*tma.a.global_scale,
            &*tma.b.global_scale,
        )?;
    runtime
        .quant
        .tensor_amax_from_chunks_f32(stream, &*tma.b.chunk_amax, tma.b.amax, chunk_count)
}

fn run_tma_gemm_self_prepared(
    stream: &CudaStream,
    runtime: &Runtime,
    out: &mut DeviceBuffer<f32>,
    tma: &mut TmaScratchRefs<'_>,
    dims: TmaDims,
) -> Result<(), DriverError> {
    if dims.is_exact() {
        runtime
            .optimizer
            .tma_gemm()
            .prepare_tma_nvfp4_device_scales_into(
                stream,
                &*tma.a.bytes,
                &*tma.a.scale_packed,
                &*tma.a.bytes,
                &*tma.a.scale_packed,
                dims.m,
                dims.k,
                dims.n,
                tma.descriptors,
            )?;
        return runtime
            .optimizer
            .tma_gemm()
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
                stream,
                &*tma.descriptors,
                out,
                dims.m,
                dims.k,
                dims.n,
                &*tma.a.global_scale,
                &*tma.a.global_scale,
            );
    }
    runtime
        .optimizer
        .tma_gemm()
        .prepare_tma_nvfp4_device_scales_into(
            stream,
            &*tma.a.bytes,
            &*tma.a.scale_packed,
            &*tma.a.bytes,
            &*tma.a.scale_packed,
            dims.m,
            dims.k,
            dims.n,
            tma.descriptors,
        )?;
    runtime
        .optimizer
        .tma_gemm()
        .gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
            stream,
            &*tma.descriptors,
            tma.out_padded,
            dims.m,
            dims.k,
            dims.n,
            &*tma.a.global_scale,
            &*tma.a.global_scale,
        )?;
    crop_tma_out(stream, runtime, out, tma.out_padded, dims)
}

fn run_tma_gemm_self_prepared_and_bound_amax(
    stream: &CudaStream,
    runtime: &Runtime,
    out: &mut DeviceBuffer<f32>,
    tma: &mut TmaScratchRefs<'_>,
    dims: TmaDims,
) -> Result<(), DriverError> {
    if !dims.is_exact() {
        run_tma_gemm_self_prepared(stream, runtime, out, tma, dims)?;
        return runtime.quant.tensor_amax_f32(TensorAmaxArgs {
            stream,
            x: out,
            chunk_amax: tma.b.chunk_amax,
            out: tma.bound_amax,
            element_count: dims.logical_m * dims.logical_n,
        });
    }

    runtime
        .optimizer
        .tma_gemm()
        .prepare_tma_nvfp4_device_scales_into(
            stream,
            &*tma.a.bytes,
            &*tma.a.scale_packed,
            &*tma.a.bytes,
            &*tma.a.scale_packed,
            dims.m,
            dims.k,
            dims.n,
            tma.descriptors,
        )?;
    let chunk_count = runtime
        .optimizer
        .tma_gemm()
        .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_with_output_amax(
            stream,
            &*tma.descriptors,
            out,
            tma.b.chunk_amax,
            dims.m,
            dims.k,
            dims.n,
            &*tma.a.global_scale,
            &*tma.a.global_scale,
        )?;
    runtime.quant.tensor_amax_from_chunks_f32(
        stream,
        &*tma.b.chunk_amax,
        tma.bound_amax,
        chunk_count,
    )
}

fn crop_tma_out(
    stream: &CudaStream,
    runtime: &Runtime,
    out: &mut DeviceBuffer<f32>,
    out_padded: &DeviceBuffer<f32>,
    dims: TmaDims,
) -> Result<(), DriverError> {
    runtime.optimizer.tma_pad().crop_f32(F32CropArgs {
        stream,
        input: out_padded,
        output: out,
        rows: dims.logical_m,
        cols: dims.logical_n,
        input_cols: dims.n,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "paired exact quantization uses explicit dimensions"
)]
fn quantize_operand_pair_padded_with_amax(
    stream: &CudaStream,
    runtime: &Runtime,
    input: &DeviceBuffer<f32>,
    row: OperandScratchRefs<'_>,
    transpose: OperandScratchRefs<'_>,
    rows: u32,
    cols: u32,
    padded_rows: u32,
    padded_cols: u32,
) -> Result<(), DriverError> {
    assert_eq!(rows, padded_rows);
    assert_eq!(cols, padded_cols);
    assert!(rows.is_power_of_two());
    assert!(rows.is_multiple_of(16));
    assert!(cols.is_multiple_of(64));

    runtime
        .quant
        .fp32_pair_to_nvfp4_four_six_exact_pow2_tiled_packed_scales(
            Nvfp4QuantPairTransposeExactArgs {
                stream,
                x: input,
                amax: &*row.amax,
                out_fp4: row.bytes,
                out_scales: row.scale_packed,
                out_global_scale: row.global_scale,
                transpose_out_fp4: transpose.bytes,
                transpose_out_scales: transpose.scale_packed,
                transpose_out_global_scale: transpose.global_scale,
                source_rows: rows,
                source_cols: cols,
            },
        )
}

fn quantize_operand_padded(
    stream: &CudaStream,
    runtime: &Runtime,
    input: &DeviceBuffer<f32>,
    scratch: OperandScratchRefs<'_>,
    rows: u32,
    cols: u32,
    padded_rows: u32,
    padded_cols: u32,
) -> Result<(), DriverError> {
    let mut scratch = scratch;
    let elements = rows * cols;
    runtime.quant.tensor_amax_f32(TensorAmaxArgs {
        stream,
        x: input,
        chunk_amax: &mut *scratch.chunk_amax,
        out: &mut *scratch.amax,
        element_count: elements,
    })?;
    quantize_operand_padded_with_amax(
        stream,
        runtime,
        input,
        scratch.reborrow(),
        rows,
        cols,
        padded_rows,
        padded_cols,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "padded quantization uses explicit dimensions"
)]
fn quantize_operand_padded_with_amax(
    stream: &CudaStream,
    runtime: &Runtime,
    input: &DeviceBuffer<f32>,
    scratch: OperandScratchRefs<'_>,
    rows: u32,
    cols: u32,
    padded_rows: u32,
    padded_cols: u32,
) -> Result<(), DriverError> {
    runtime
        .quant
        .fp32_to_nvfp4_four_six_padded(Nvfp4QuantPaddedArgs {
            stream,
            x: input,
            amax: scratch.amax,
            out_fp4: scratch.bytes,
            out_scales: scratch.scales,
            out_global_scale: scratch.global_scale,
            rows,
            cols,
            padded_rows,
            padded_cols,
        })?;
    runtime.optimizer.tma_scale_pack().pack(
        stream,
        &*scratch.scales,
        scratch.scale_packed,
        padded_rows,
        padded_cols,
    )
}

fn quantize_operand_transposed_padded(
    stream: &CudaStream,
    runtime: &Runtime,
    input: &DeviceBuffer<f32>,
    scratch: OperandScratchRefs<'_>,
    rows: u32,
    cols: u32,
    padded_rows: u32,
    padded_cols: u32,
) -> Result<(), DriverError> {
    let elements = rows * cols;
    runtime.quant.tensor_amax_f32(TensorAmaxArgs {
        stream,
        x: input,
        chunk_amax: scratch.chunk_amax,
        out: scratch.amax,
        element_count: elements,
    })?;
    quantize_operand_transposed_padded_with_amax(
        stream,
        runtime,
        input,
        scratch,
        rows,
        cols,
        padded_rows,
        padded_cols,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "padded transpose quantization uses explicit dimensions"
)]
fn quantize_operand_transposed_padded_with_amax(
    stream: &CudaStream,
    runtime: &Runtime,
    input: &DeviceBuffer<f32>,
    scratch: OperandScratchRefs<'_>,
    rows: u32,
    cols: u32,
    padded_rows: u32,
    padded_cols: u32,
) -> Result<(), DriverError> {
    if cols == padded_rows
        && rows == padded_cols
        && rows.is_power_of_two()
        && rows.is_multiple_of(64)
        && cols.is_multiple_of(128)
    {
        return runtime
            .quant
            .fp32_transpose_to_nvfp4_four_six_exact_packed_scales(Nvfp4QuantTransposePaddedArgs {
                stream,
                x: input,
                amax: scratch.amax,
                out_fp4: scratch.bytes,
                out_scales: scratch.scale_packed,
                out_global_scale: scratch.global_scale,
                source_rows: rows,
                source_cols: cols,
                padded_rows,
                padded_cols,
            });
    }
    runtime
        .quant
        .fp32_transpose_to_nvfp4_four_six_padded(Nvfp4QuantTransposePaddedArgs {
            stream,
            x: input,
            amax: scratch.amax,
            out_fp4: scratch.bytes,
            out_scales: scratch.scales,
            out_global_scale: scratch.global_scale,
            source_rows: rows,
            source_cols: cols,
            padded_rows,
            padded_cols,
        })?;
    runtime.optimizer.tma_scale_pack().pack(
        stream,
        &*scratch.scales,
        scratch.scale_packed,
        padded_rows,
        padded_cols,
    )
}

fn polar_shape(desc: MuonSlotDescriptor) -> (u32, u32) {
    if desc.rows <= desc.cols {
        (desc.rows, desc.cols)
    } else {
        (desc.cols, desc.rows)
    }
}

fn can_defer_polar_bounds(rows: u32, cols: u32) -> bool {
    let dims = TmaDims::new(rows, cols, rows);
    rows == dims.m
        && rows == dims.k
        && cols == dims.n
        && rows.is_power_of_two()
        && rows.is_multiple_of(16)
        && cols.is_multiple_of(64)
}

#[derive(Clone, Copy)]
struct TmaDims {
    logical_m: u32,
    logical_n: u32,
    m: u32,
    n: u32,
    k: u32,
}

impl TmaDims {
    fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            logical_m: m,
            logical_n: n,
            m: ceil_to(m, TILE_M),
            n: ceil_to(n, TILE_N),
            k: ceil_to(k, TILE_K),
        }
    }

    fn is_exact(self) -> bool {
        self.logical_m == self.m && self.logical_n == self.n
    }
}

struct TmaScratchRefs<'a> {
    out_padded: &'a mut DeviceBuffer<f32>,
    a: OperandScratchRefs<'a>,
    b: OperandScratchRefs<'a>,
    bound_amax: &'a mut DeviceBuffer<f32>,
    descriptors: &'a mut TmaNvfp4DeviceScaleDescriptors,
}

impl<'a> TmaScratchRefs<'a> {
    fn reborrow(&mut self) -> TmaScratchRefs<'_> {
        TmaScratchRefs {
            out_padded: &mut *self.out_padded,
            a: self.a.reborrow(),
            b: self.b.reborrow(),
            bound_amax: &mut *self.bound_amax,
            descriptors: &mut *self.descriptors,
        }
    }
}

struct OperandScratchRefs<'a> {
    bytes: &'a mut DeviceBuffer<u8>,
    scales: &'a mut DeviceBuffer<u8>,
    scale_packed: &'a mut DeviceBuffer<u8>,
    global_scale: &'a mut DeviceBuffer<f32>,
    amax: &'a mut DeviceBuffer<f32>,
    chunk_amax: &'a mut DeviceBuffer<f32>,
}

impl<'a> OperandScratchRefs<'a> {
    fn reborrow(&mut self) -> OperandScratchRefs<'_> {
        OperandScratchRefs {
            bytes: &mut *self.bytes,
            scales: &mut *self.scales,
            scale_packed: &mut *self.scale_packed,
            global_scale: &mut *self.global_scale,
            amax: &mut *self.amax,
            chunk_amax: &mut *self.chunk_amax,
        }
    }
}

fn tma_refs(scratch: &mut MuonTmaScratch) -> TmaScratchRefs<'_> {
    TmaScratchRefs {
        out_padded: &mut scratch.out_padded,
        a: operand_refs(&mut scratch.a),
        b: operand_refs(&mut scratch.b),
        bound_amax: &mut scratch.bound_amax,
        descriptors: &mut scratch.descriptors,
    }
}

fn operand_refs(scratch: &mut MuonTmaOperandScratch) -> OperandScratchRefs<'_> {
    OperandScratchRefs {
        bytes: &mut scratch.bytes,
        scales: &mut scratch.scales,
        scale_packed: &mut scratch.scale_packed,
        global_scale: &mut scratch.global_scale,
        amax: &mut scratch.amax,
        chunk_amax: &mut scratch.chunk_amax,
    }
}

fn ceil_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

#[derive(Clone, Copy)]
struct TmaTraceConfig {
    enabled: bool,
    slot: Option<usize>,
}

impl TmaTraceConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool("MUON_TMA_TRACE").unwrap_or(false),
            slot: env_usize("MUON_TMA_TRACE_SLOT"),
        }
    }

    fn should_trace(self, slot_index: usize) -> bool {
        self.enabled && self.slot.is_none_or(|slot| slot == slot_index)
    }
}

fn trace_buffer(
    stream: &CudaStream,
    trace: TmaTraceConfig,
    slot_index: usize,
    stage: &str,
    buffer: &DeviceBuffer<f32>,
    len: u32,
    iter: Option<u32>,
    desc: MuonSlotDescriptor,
) -> Result<(), DriverError> {
    if !trace.should_trace(slot_index) {
        return Ok(());
    }
    let values = buffer.to_host_vec(stream)?;
    let active_len = (len as usize).min(values.len());
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut first_bad = None;
    let mut first_bad_value = 0.0f32;
    for (index, value) in values.iter().take(active_len).enumerate() {
        if !value.is_finite() && first_bad.is_none() {
            first_bad = Some(index);
            first_bad_value = *value;
        }
        let abs = value.abs();
        if abs.is_finite() {
            max_abs = max_abs.max(abs);
            sum_sq += (*value as f64) * (*value as f64);
        }
    }
    let rms = (sum_sq / active_len.max(1) as f64).sqrt() as f32;
    eprintln!(
        "muon_tma_trace slot={} shape={}x{} polar={}x{} iter={} stage={} len={} finite={} rms={:.9e} max_abs={:.9e} first_bad={} first_bad_value={:.9e}",
        slot_index,
        desc.rows,
        desc.cols,
        polar_shape(desc).0,
        polar_shape(desc).1,
        iter.map_or_else(|| "-".to_string(), |value| value.to_string()),
        stage,
        active_len,
        first_bad.is_none(),
        rms,
        max_abs,
        first_bad.map_or_else(|| "-".to_string(), |value| value.to_string()),
        first_bad_value,
    );
    Ok(())
}
