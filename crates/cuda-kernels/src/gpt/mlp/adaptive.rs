use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use super::{
    MlpModule,
    kernels::{BLOCK_TOPK_TILE, RELU2_VALUES_PER_BLOCK},
};
use crate::{
    activation::unshifted_relu2,
    block_reduce::block_sum_shared_f32,
    f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32},
    float_ptx::{abs_f32, max_f32},
    launch::grid_x_config,
    warp_reduce::{thread_lane_warp, warp_max_nonnegative_f32},
};

const THREADS: u32 = 256;
const WARPS: usize = (THREADS / 32) as usize;
pub const ADAPTIVE_RELU2_PACK_WIDTH: u32 = 8192;
const AMAX_CHUNKS_PER_ROW: u32 = ADAPTIVE_RELU2_PACK_WIDTH / RELU2_VALUES_PER_BLOCK;

pub struct AdaptiveRelu2PackArgs<'a> {
    pub stream: &'a CudaStream,
    pub values: &'a mut DeviceBuffer<f32>,
    pub tape: Option<&'a mut DeviceBuffer<u16>>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub globals: &'a mut DeviceBuffer<f32>,
    pub rows: u32,
}

impl MlpModule {
    /// Center and pack in one CTA per token. Still write the centered view
    /// into the input workspace for the native activation inspector.
    pub fn adaptive_relu2_pack(&self, args: AdaptiveRelu2PackArgs<'_>) -> Result<(), DriverError> {
        let rows = args.rows as usize;
        let width = ADAPTIVE_RELU2_PACK_WIDTH as usize;
        let len = args
            .rows
            .checked_mul(ADAPTIVE_RELU2_PACK_WIDTH)
            .filter(|&n| n != 0)
            .ok_or(DriverError(1))? as usize;
        if args.values.len() < len
            || args.tape.as_ref().is_some_and(|tape| tape.len() < len)
            || args.amax.len() < rows
            || args.globals.len() < rows
            || args.bytes.len() < len / 2
            || args.scales.len() < crate::nvfp4::csf::rowwise_encoded_len(rows, width)
        {
            return Err(DriverError(1));
        }
        let tape = args.tape.map_or(core::ptr::null_mut(), |tape| {
            tape.cu_deviceptr() as *mut u16
        });
        self.adaptive
            .as_ref()
            .ok_or(DriverError(1))?
            .adaptive_relu2_pack_8192_kernel(
                args.stream,
                grid_x_config(args.rows, THREADS),
                args.values,
                tape,
                args.amax,
                args.bytes,
                args.scales,
                args.globals,
                args.rows,
            )
    }

    /// Replace the raw affine output by ReLU² minus its per-token feature mean.
    /// Reverse uses the ordinary FP16 preactivation tape.
    pub fn adaptive_relu2_forward(
        &self,
        stream: &CudaStream,
        values: &mut DeviceBuffer<f32>,
        tape: Option<&mut DeviceBuffer<u16>>,
        rows: u32,
        width: u32,
    ) -> Result<(), DriverError> {
        let len = rows
            .checked_mul(width)
            .filter(|&n| n != 0)
            .ok_or(DriverError(1))? as usize;
        if values.len() < len || tape.as_ref().is_some_and(|saved| saved.len() < len) {
            return Err(DriverError(1));
        }
        let tape = tape.map_or(core::ptr::null_mut(), |saved| {
            saved.cu_deviceptr() as *mut u16
        });
        self.adaptive
            .as_ref()
            .ok_or(DriverError(1))?
            .adaptive_relu2_forward_kernel(
                stream,
                grid_x_config(rows, THREADS),
                values,
                tape,
                width,
            )
    }

    /// Apply the coupled centering adjoint in place to the down-projection
    /// input gradient. A route masks only the direct term of that gradient.
    /// At width 8192, optionally emit the up projection's MS-EDEN maxima.
    pub fn adaptive_relu2_backward(
        &self,
        stream: &CudaStream,
        gradient: &mut DeviceBuffer<f32>,
        tape: &DeviceBuffer<u16>,
        masks: &DeviceBuffer<u64>,
        rows: u32,
        width: u32,
        routed: bool,
        chunk_amax: Option<&mut DeviceBuffer<f32>>,
    ) -> Result<Option<u32>, DriverError> {
        let len = rows
            .checked_mul(width)
            .filter(|&n| n != 0)
            .ok_or(DriverError(1))? as usize;
        if gradient.len() < len
            || tape.len() < len
            || (routed
                && (width > 64 * BLOCK_TOPK_TILE
                    || masks.len() < rows.div_ceil(BLOCK_TOPK_TILE) as usize))
        {
            return Err(DriverError(1));
        }
        let adaptive = self.adaptive.as_ref().ok_or(DriverError(1))?;
        let launch = grid_x_config(rows, THREADS);
        if let Some(chunk_amax) = chunk_amax.filter(|_| width == ADAPTIVE_RELU2_PACK_WIDTH) {
            let chunk_count = rows * AMAX_CHUNKS_PER_ROW;
            if chunk_amax.len() < chunk_count as usize {
                return Err(DriverError(1));
            }
            adaptive.adaptive_relu2_backward_amax_8192_kernel(
                stream,
                launch,
                gradient,
                tape,
                masks,
                chunk_amax,
                routed as u32,
            )?;
            return Ok(Some(chunk_count));
        }
        adaptive.adaptive_relu2_backward_kernel(
            stream,
            launch,
            gradient,
            tape,
            masks,
            width,
            routed as u32,
        )?;
        Ok(None)
    }
}

#[cuda_module]
pub(super) mod kernels {
    use super::*;

    #[inline(always)]
    fn row_mean(
        row: *const f32,
        tape: *mut u16,
        base: usize,
        width: u32,
        storage: &mut SharedArray<f32, WARPS>,
    ) -> f32 {
        let (tid, lane, warp) = thread_lane_warp();
        let mut sum = 0.0;
        for col in (tid..width).step_by(THREADS as usize) {
            let z = unsafe { *row.add(col as usize) };
            if !tape.is_null() {
                unsafe { *tape.add(base + col as usize) = cvt_rn_f16_f32(z) };
            }
            sum += unshifted_relu2(z);
        }
        block_sum_shared_f32(storage, sum, lane, warp) / width as f32
    }

    #[kernel]
    pub fn adaptive_relu2_forward_kernel(
        mut values: DisjointSlice<f32>,
        tape: *mut u16,
        width: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let base = (thread::blockIdx_x() * width) as usize;
        let tid = thread::threadIdx_x();
        let row = unsafe { values.as_mut_ptr().add(base) };
        let mean = unsafe { row_mean(row, tape, base, width, &mut REDUCTION) };
        for col in (tid..width).step_by(THREADS as usize) {
            let value = unsafe { &mut *row.add(col as usize) };
            *value = unshifted_relu2(*value) - mean;
        }
    }

    #[kernel]
    pub fn adaptive_relu2_pack_8192_kernel(
        mut values: DisjointSlice<f32>,
        tape: *mut u16,
        amax: DisjointSlice<f32>,
        bytes: DisjointSlice<u8>,
        scales: DisjointSlice<u8>,
        globals: DisjointSlice<f32>,
        rows: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let base = (thread::blockIdx_x() * ADAPTIVE_RELU2_PACK_WIDTH) as usize;
        let values = values.as_mut_ptr();
        let mean = unsafe {
            row_mean(
                values.add(base),
                tape,
                base,
                ADAPTIVE_RELU2_PACK_WIDTH,
                &mut REDUCTION,
            )
        };
        crate::nvfp4_quant::kernels::four_six::module::rowwise_derived_amax_8192_cached_body::<
            false,
            true,
        >(
            values,
            amax,
            bytes,
            scales,
            globals,
            rows,
            crate::nvfp4_quant::SCALE_OVERRIDE,
            mean,
            unsafe { &mut REDUCTION },
        );
    }

    #[inline(always)]
    fn adaptive_relu2_backward_body(
        mut gradient: DisjointSlice<f32>,
        tape: &[u16],
        masks: &[u64],
        width: u32,
        routed: u32,
        chunk_amax: Option<DisjointSlice<f32>>,
        reduction: &mut SharedArray<f32, WARPS>,
    ) {
        let row = thread::blockIdx_x();
        let base = (row * width) as usize;
        let (tid, lane, warp) = thread_lane_warp();
        let mask = if routed != 0 {
            masks[(row / BLOCK_TOPK_TILE) as usize]
        } else {
            u64::MAX
        };
        let gradient = unsafe { gradient.as_mut_ptr().add(base) };
        let direct = |col: u32| {
            if mask & (1_u64 << ((col / BLOCK_TOPK_TILE) & 63)) != 0 {
                unsafe { *gradient.add(col as usize) }
            } else {
                0.0
            }
        };
        let mut sum = 0.0;
        for col in (tid..width).step_by(THREADS as usize) {
            sum += direct(col);
        }
        let mean = block_sum_shared_f32(reduction, sum, lane, warp) / width as f32;
        let write_adjoint = |col: u32| {
            let positive = max_f32(cvt_f32_f16(tape[base + col as usize]), 0.0);
            let adjoint = (direct(col) - mean) * 2.0 * positive;
            unsafe { *gradient.add(col as usize) = adjoint };
            adjoint
        };
        if let Some(mut chunk_amax) = chunk_amax {
            static mut WARP_MAXIMA: SharedArray<f32, { WARPS * AMAX_CHUNKS_PER_ROW as usize }> =
                SharedArray::UNINIT;

            // Preserve the scan's 2048-element chunks and eight values per thread.
            for chunk in 0..AMAX_CHUNKS_PER_ROW {
                let start = chunk * RELU2_VALUES_PER_BLOCK + tid;
                let mut local = abs_f32(write_adjoint(start));
                for col in (start + THREADS..(chunk + 1) * RELU2_VALUES_PER_BLOCK)
                    .step_by(THREADS as usize)
                {
                    local = max_f32(local, abs_f32(write_adjoint(col)));
                }
                let maximum = warp_max_nonnegative_f32(local);
                if lane == 0 {
                    unsafe { WARP_MAXIMA[chunk as usize * WARPS + warp as usize] = maximum };
                }
            }
            thread::sync_threads();
            if warp < AMAX_CHUNKS_PER_ROW {
                let partial = if lane < WARPS as u32 {
                    unsafe { WARP_MAXIMA[warp as usize * WARPS + lane as usize] }
                } else {
                    0.0
                };
                let maximum = warp_max_nonnegative_f32(partial);
                if lane == 0 {
                    let index = (row * AMAX_CHUNKS_PER_ROW + warp) as usize;
                    unsafe { *chunk_amax.get_unchecked_mut(index) = maximum };
                }
            }
        } else {
            for col in (tid..width).step_by(THREADS as usize) {
                write_adjoint(col);
            }
        }
    }

    #[kernel]
    pub fn adaptive_relu2_backward_kernel(
        gradient: DisjointSlice<f32>,
        tape: &[u16],
        masks: &[u64],
        width: u32,
        routed: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        unsafe {
            adaptive_relu2_backward_body(
                gradient,
                tape,
                masks,
                width,
                routed,
                None,
                &mut REDUCTION,
            );
        }
    }

    #[kernel]
    pub fn adaptive_relu2_backward_amax_8192_kernel(
        gradient: DisjointSlice<f32>,
        tape: &[u16],
        masks: &[u64],
        chunk_amax: DisjointSlice<f32>,
        routed: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        unsafe {
            adaptive_relu2_backward_body(
                gradient,
                tape,
                masks,
                ADAPTIVE_RELU2_PACK_WIDTH,
                routed,
                Some(chunk_amax),
                &mut REDUCTION,
            );
        }
    }
}
