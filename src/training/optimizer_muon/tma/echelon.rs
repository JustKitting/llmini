use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::f32_matrix_ops::{F32Linear2Args, F32Linear3Args};
use rust_kernels_cuda::optimizer::echelon::{
    EchelonModule, FrameModule, Shape, TailArgs, TailScratch, TailState,
};

use super::{
    MuonGroupTable, MuonTmaOperandScratch, Runtime, TmaDims, TmaScratchRefs, crop_tma_out,
    operand_refs, polar_shape, quantize_operand_padded, quantize_operand_transposed_padded,
    run_tma_gemm_self_prepared,
};

const HEAD_RANK: u32 = 128;
const COLD_REFRESHES: usize = 3;
const HEAD_ITERATIONS: usize = 5;
const HEAD_LINEAR: f32 = 3.4445;
const HEAD_CUBIC: f32 = -4.7750;
const HEAD_QUINTIC: f32 = 2.0315;

pub(in crate::training) struct Echelon {
    kernels: EchelonModule,
    qr: FrameModule,
    slots: Vec<Option<Slot>>,
    scratch: Scratch,
}

struct Slot {
    shape: Shape,
    frame: DeviceBuffer<u16>,
    tail: TailState,
    seed: u32,
    refreshed: bool,
}

struct Scratch {
    q: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    factor: DeviceBuffer<f32>,
    head: DeviceBuffer<f32>,
    polar: DeviceBuffer<f32>,
    next_polar: DeviceBuffer<f32>,
    gram: DeviceBuffer<f32>,
    gram_squared: DeviceBuffer<f32>,
    polynomial: DeviceBuffer<f32>,
    norm_chunks: DeviceBuffer<f32>,
    tail: TailScratch,
    momentum_transpose: MuonTmaOperandScratch,
}

impl Echelon {
    pub(in crate::training) fn new(
        runtime: &Runtime,
        table: &MuonGroupTable,
        seed: u64,
    ) -> Result<Self, DriverError> {
        Self::for_shapes(
            runtime,
            table.host_slots.iter().copied().map(polar_shape),
            seed,
        )
    }

    fn for_shapes(
        runtime: &Runtime,
        shapes: impl Iterator<Item = (u32, u32)>,
        seed: u64,
    ) -> Result<Self, DriverError> {
        let stream = runtime.stream.as_ref();
        let (kernels, qr) = runtime.echelon_modules()?;
        let mut slots = Vec::new();
        let (mut max_short, mut max_long, mut max_rank) = (0, 0, 0);
        for (index, (short, long)) in shapes.enumerate() {
            if short == 0 || long == 0 {
                slots.push(None);
                continue;
            }
            let rank = HEAD_RANK.min(short / 2);
            let shape = Shape::new(short, long, rank)?;
            max_short = max_short.max(short);
            max_long = max_long.max(long);
            max_rank = max_rank.max(rank);
            let seed = seed as u32 ^ (seed >> 32) as u32 ^ (index as u32).wrapping_mul(0x9e37_79b9);
            slots.push(Some(Slot {
                shape,
                frame: DeviceBuffer::zeroed(stream, (rank * short) as usize)?,
                tail: kernels.new_tail_state(stream, shape, seed ^ 0x21f0_aaad)?,
                seed,
                refreshed: false,
            }));
        }
        let capacity = Shape::new(max_short, max_long, max_rank)?;
        let allocate = |len: u32| DeviceBuffer::zeroed(stream, len as usize);
        let thin = max_long * max_rank;
        let square = max_rank * max_rank;
        let momentum_dims = TmaDims::new(max_rank, max_long, max_short);
        let scratch = Scratch {
            q: allocate(max_short * max_rank)?,
            v: allocate(thin)?,
            factor: allocate(thin)?,
            head: allocate(thin)?,
            polar: allocate(thin)?,
            next_polar: allocate(thin)?,
            gram: allocate(square)?,
            gram_squared: allocate(square)?,
            polynomial: allocate(square)?,
            norm_chunks: allocate(thin.div_ceil(128))?,
            tail: TailScratch::new(stream, capacity)?,
            momentum_transpose: MuonTmaOperandScratch::new(
                stream,
                momentum_dims.n as usize * momentum_dims.k as usize,
                momentum_dims.n as usize,
                momentum_dims.k as usize,
            )?,
        };
        Ok(Self {
            kernels,
            qr,
            slots,
            scratch,
        })
    }

    pub(super) fn apply(
        &mut self,
        runtime: &Runtime,
        slot_index: usize,
        x: &DeviceBuffer<f32>,
        residual: &mut DeviceBuffer<f32>,
        update: &mut DeviceBuffer<f32>,
        tma: &mut TmaScratchRefs<'_>,
    ) -> Result<(), DriverError> {
        let stream = runtime.stream.as_ref();
        let slot = self
            .slots
            .get_mut(slot_index)
            .and_then(Option::as_mut)
            .ok_or(DriverError(1))?;
        let (short, long, rank) = slot.shape.dimensions();
        let s = &mut self.scratch;
        let cold = !slot.refreshed;
        let (sample, refresh) =
            pack_momentum(runtime, tma, &mut s.momentum_transpose, x, slot.shape)?;
        self.qr
            .load_warm(stream, &slot.frame, &mut s.q, slot.seed, short, cold)?;
        for _ in 0..if cold { COLD_REFRESHES } else { 1 } {
            // QᵀX then VᵀXᵀ. Each factor is orthonormalized before
            // the next half-step, following the two-sided power iteration.
            product_with_momentum_transpose(
                runtime,
                tma,
                &mut s.momentum_transpose,
                &s.q,
                &mut s.factor,
                sample,
                short,
            )?;
            self.qr
                .orthonormalize(stream, &mut s.factor, &mut s.v, 1, rank, long)?;
            product(
                runtime,
                tma,
                (&s.v, false),
                None,
                &mut s.factor,
                refresh,
                long,
            )?;
            self.qr
                .orthonormalize(stream, &mut s.factor, &mut s.q, 1, rank, short)?;
        }
        self.qr.store_warm(stream, &s.q, &mut slot.frame)?;
        product_with_momentum_transpose(
            runtime,
            tma,
            &mut s.momentum_transpose,
            &s.q,
            &mut s.head,
            sample,
            short,
        )?;
        let lift = TmaDims::new(short, long, rank);
        product(
            runtime,
            tma,
            (&s.q, true),
            Some((&s.head, true)),
            residual,
            lift,
            rank,
        )?;
        self.kernels.residual(stream, x, residual, slot.shape)?;
        let inverse = self.kernels.estimate_tail(TailArgs {
            stream,
            residual,
            state: &mut slot.tail,
            scratch: &mut s.tail,
        })?;

        self.kernels.normalize_head(
            stream,
            &s.head,
            &mut s.polar,
            &mut s.norm_chunks,
            slot.shape,
        )?;
        let gram_dims = TmaDims::new(rank, rank, long);
        let square_dims = TmaDims::new(rank, rank, rank);
        let head_dims = TmaDims::new(rank, long, rank);
        for _ in 0..HEAD_ITERATIONS {
            // P <- aP + (bG + cG²)P, with G = PPᵀ. Only the head
            // participates in the quintic; all products use NVFP4.
            self_product(runtime, tma, &s.polar, &mut s.gram, gram_dims, long)?;
            self_product(
                runtime,
                tma,
                &s.gram,
                &mut s.gram_squared,
                square_dims,
                rank,
            )?;
            runtime.f32_ops.linear2(F32Linear2Args {
                stream,
                a: &s.gram,
                b: &s.gram_squared,
                out: &mut s.polynomial,
                len: rank * rank,
                a_scale: HEAD_CUBIC,
                b_scale: HEAD_QUINTIC,
            })?;
            product(
                runtime,
                tma,
                (&s.polynomial, false),
                Some((&s.polar, true)),
                &mut s.next_polar,
                head_dims,
                rank,
            )?;
            runtime.f32_ops.linear3(F32Linear3Args {
                stream,
                a: &s.polar,
                b: &s.polar,
                c_out: &mut s.next_polar,
                len: rank * long,
                a_scale: HEAD_LINEAR,
                b_scale: 0.0,
                c_scale: 1.0,
            })?;
            std::mem::swap(&mut s.polar, &mut s.next_polar);
        }
        product(
            runtime,
            tma,
            (&s.q, true),
            Some((&s.polar, true)),
            update,
            lift,
            rank,
        )?;
        self.kernels
            .add_tail(stream, inverse, residual, update, slot.shape)?;
        slot.refreshed = true;
        Ok(())
    }
}

/// These two layouts are read-only across every refresh and the final head
/// projection of this slot. Keep the ordinary packers and their exact extents.
fn pack_momentum(
    runtime: &Runtime,
    tma: &mut TmaScratchRefs<'_>,
    transpose: &mut MuonTmaOperandScratch,
    momentum: &DeviceBuffer<f32>,
    shape: Shape,
) -> Result<(TmaDims, TmaDims), DriverError> {
    let stream = runtime.stream.as_ref();
    let (short, long, rank) = shape.dimensions();
    let sample = TmaDims::new(rank, long, short);
    let refresh = TmaDims::new(rank, short, long);
    quantize_operand_transposed_padded(
        stream,
        runtime,
        momentum,
        operand_refs(transpose),
        short,
        long,
        sample.n,
        sample.k,
    )?;
    quantize_operand_padded(
        stream,
        runtime,
        momentum,
        tma.b.reborrow(),
        short,
        long,
        refresh.n,
        refresh.k,
        tma.bound_amax,
    )?;
    Ok((sample, refresh))
}

fn product_with_momentum_transpose(
    runtime: &Runtime,
    tma: &mut TmaScratchRefs<'_>,
    transpose: &mut MuonTmaOperandScratch,
    frame: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    dims: TmaDims,
    inner: u32,
) -> Result<(), DriverError> {
    let mut tma = tma.reborrow();
    tma.b = operand_refs(transpose);
    product(runtime, &mut tma, (frame, false), None, output, dims, inner)
}

fn self_product(
    runtime: &Runtime,
    tma: &mut TmaScratchRefs<'_>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    dims: TmaDims,
    inner: u32,
) -> Result<(), DriverError> {
    let stream = runtime.stream.as_ref();
    quantize_operand_padded(
        stream,
        runtime,
        input,
        tma.a.reborrow(),
        dims.logical_m,
        inner,
        dims.m,
        dims.k,
        tma.bound_amax,
    )?;
    run_tma_gemm_self_prepared(stream, runtime, output, tma, dims)
}

/// Pack NVFP4 rows of length `inner` unless the right operand is already
/// packed. The TMA product contracts those rows and crops the output padding.
fn product(
    runtime: &Runtime,
    tma: &mut TmaScratchRefs<'_>,
    a: (&DeviceBuffer<f32>, bool),
    b: Option<(&DeviceBuffer<f32>, bool)>,
    output: &mut DeviceBuffer<f32>,
    dims: TmaDims,
    inner: u32,
) -> Result<(), DriverError> {
    let stream = runtime.stream.as_ref();
    for (operand, scratch, rows, padded_rows) in [
        (Some(a), tma.a.reborrow(), dims.logical_m, dims.m),
        (b, tma.b.reborrow(), dims.logical_n, dims.n),
    ] {
        let Some((input, transposed)) = operand else {
            continue;
        };
        if transposed {
            quantize_operand_transposed_padded(
                stream,
                runtime,
                input,
                scratch,
                inner,
                rows,
                padded_rows,
                dims.k,
            )?;
        } else {
            quantize_operand_padded(
                stream,
                runtime,
                input,
                scratch,
                rows,
                inner,
                padded_rows,
                dims.k,
                tma.bound_amax,
            )?;
        }
    }
    let gemm = runtime.optimizer.tma_gemm();
    gemm.prepare_tma_nvfp4_device_scales_into(
        stream,
        tma.a.bytes,
        tma.a.scale_packed,
        tma.b.bytes,
        tma.b.scale_packed,
        dims.m,
        dims.k,
        dims.n,
        tma.descriptors,
    )?;
    let out = if dims.is_exact() {
        &mut *output
    } else {
        &mut *tma.out_padded
    };
    gemm.gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
        stream,
        tma.descriptors,
        out,
        dims.m,
        dims.k,
        dims.n,
        tma.a.global_scale,
        tma.b.global_scale,
    )?;
    if !dims.is_exact() {
        crop_tma_out(stream, runtime, output, tma.out_padded, dims)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::optimizer_tc_scratch::MuonScratchBuffers;

    #[test]
    #[ignore = "uses the Blackwell NVFP4 TMA optimizer path"]
    fn cached_momentum_matches_repacking_in_both_orientations() -> crate::AppResult {
        let runtime = Runtime::new()?;
        let stream = &runtime.stream;
        let mut muon = MuonScratchBuffers::new(stream)?;
        let mut tma = super::super::tma_refs(&mut muon.tma);
        let capacity = TmaDims::new(8, 256, 128);
        let mut transpose = MuonTmaOperandScratch::new(
            stream,
            (capacity.n * capacity.k) as usize,
            capacity.n as usize,
            capacity.k as usize,
        )?;
        let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        for (case, (short, long, rank)) in [(128, 256, 8), (3, 5, 1), (128, 256, 8)]
            .into_iter()
            .enumerate()
        {
            let make = |len: u32, offset: u32| {
                let values = (0..len)
                    .map(|i| ((i.wrapping_mul(13) + offset) % 37) as f32 * 0.0625 - 1.0)
                    .collect::<Vec<_>>();
                DeviceBuffer::from_host(stream, &values)
            };
            let momentum = make(short * long, case as u32 + 1)?;
            let query = make(rank * short, 11)?;
            let frame = make(rank * long, 19)?;
            let mut actual_right = DeviceBuffer::zeroed(stream, (rank * long) as usize)?;
            let mut expected_right = DeviceBuffer::zeroed(stream, (rank * long) as usize)?;
            let mut actual_left = DeviceBuffer::zeroed(stream, (rank * short) as usize)?;
            let mut expected_left = DeviceBuffer::zeroed(stream, (rank * short) as usize)?;
            let (sample, refresh) = pack_momentum(
                &runtime,
                &mut tma,
                &mut transpose,
                &momentum,
                Shape::new(short, long, rank)?,
            )?;
            for _ in 0..2 {
                product_with_momentum_transpose(
                    &runtime,
                    &mut tma,
                    &mut transpose,
                    &query,
                    &mut actual_right,
                    sample,
                    short,
                )?;
                product(
                    &runtime,
                    &mut tma,
                    (&frame, false),
                    None,
                    &mut actual_left,
                    refresh,
                    long,
                )?;
            }
            for (basis, transposed, output, dims, inner) in [
                (&query, true, &mut expected_right, sample, short),
                (&frame, false, &mut expected_left, refresh, long),
            ] {
                product(
                    &runtime,
                    &mut tma,
                    (basis, false),
                    Some((&momentum, transposed)),
                    output,
                    dims,
                    inner,
                )?;
            }
            assert_eq!(
                bits(actual_right.to_host_vec(stream)?),
                bits(expected_right.to_host_vec(stream)?),
            );
            assert_eq!(
                bits(actual_left.to_host_vec(stream)?),
                bits(expected_left.to_host_vec(stream)?),
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "uses the Blackwell NVFP4 TMA optimizer path"]
    fn echelon_polar_preserves_the_tail_and_zero_update() -> crate::AppResult {
        let runtime = Runtime::new()?;
        let stream = runtime.stream.as_ref();
        let mut muon = MuonScratchBuffers::new(stream)?;
        let mut echelon = Echelon::for_shapes(&runtime, [(3, 5)].into_iter(), 61)?;
        let x = DeviceBuffer::from_host(
            stream,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 0.0, 0.0,
            ],
        )?;
        let mut residual = DeviceBuffer::zeroed(stream, 15)?;
        let mut update = DeviceBuffer::zeroed(stream, 15)?;
        for _ in 0..2 {
            echelon.apply(
                &runtime,
                0,
                &x,
                &mut residual,
                &mut update,
                &mut super::super::tma_refs(&mut muon.tma),
            )?;
            let actual = update.to_host_vec(stream)?;
            let mut dominant = 1.0_f32;
            for _ in 0..HEAD_ITERATIONS {
                let square = dominant * dominant;
                dominant *= HEAD_LINEAR + HEAD_CUBIC * square + HEAD_QUINTIC * square * square;
            }
            for (index, value) in actual.into_iter().enumerate() {
                let expected = match index {
                    0 => dominant,
                    6 => 1.0 / 1.05,
                    12 => 0.5 / 1.05,
                    _ => 0.0,
                };
                assert!(
                    (value - expected).abs() < 0.04,
                    "index={index} actual={value} expected={expected}",
                );
            }
        }
        let zero = DeviceBuffer::zeroed(stream, 15)?;
        echelon.apply(
            &runtime,
            0,
            &zero,
            &mut residual,
            &mut update,
            &mut super::super::tma_refs(&mut muon.tma),
        )?;
        assert_eq!(update.to_host_vec(stream)?, [0.0; 15]);
        Ok(())
    }
}
