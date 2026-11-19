//! Device-side residual power iteration and vector operations for Echelon.
use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::{
    block_reduce::block_sum_shared_f32,
    f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32},
    float_ptx::{fma_f32, max_f32, sqrt_f32},
    launch::{grid_x_config, linear_config},
    warp_reduce::thread_lane_warp,
};

mod frame;
pub use frame::FrameModule;

const THREADS: u32 = 128;
const WARPS: usize = (THREADS / 32) as usize;
const TAIL_ITERATIONS: usize = 2;
const TAIL_MARGIN: f32 = 1.05;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    short: u32,
    long: u32,
    rank: u32,
}

unsafe impl DeviceCopy for Shape {}

impl Shape {
    pub fn new(short: u32, long: u32, rank: u32) -> Result<Self, DriverError> {
        if rank == 0 || rank >= short || short > long || short.checked_mul(long).is_none() {
            return Err(DriverError(1));
        }
        Ok(Self { short, long, rank })
    }

    pub fn dimensions(self) -> (u32, u32, u32) {
        (self.short, self.long, self.rank)
    }
}

pub struct TailState {
    shape: Shape,
    warm: DeviceBuffer<u16>,
    seed: u32,
}

pub struct TailScratch {
    z: DeviceBuffer<f32>,
    next_z: DeviceBuffer<f32>,
    w: DeviceBuffer<f32>,
    inverse: DeviceBuffer<f32>,
}

impl TailScratch {
    pub fn new(stream: &CudaStream, capacity: Shape) -> Result<Self, DriverError> {
        Ok(Self {
            z: DeviceBuffer::zeroed(stream, capacity.long as usize)?,
            next_z: DeviceBuffer::zeroed(stream, capacity.long as usize)?,
            w: DeviceBuffer::zeroed(stream, capacity.short as usize)?,
            inverse: DeviceBuffer::zeroed(stream, 1)?,
        })
    }
}

pub struct TailArgs<'a, 'scratch> {
    pub stream: &'a CudaStream,
    /// X - QH, in the orientation with the short dimension first.
    pub residual: &'a DeviceBuffer<f32>,
    pub state: &'scratch mut TailState,
    pub scratch: &'scratch mut TailScratch,
}

pub struct EchelonModule(kernels::LoadedModule);

impl EchelonModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self(kernels::from_module(module)?))
    }

    pub fn new_tail_state(
        &self,
        stream: &CudaStream,
        shape: Shape,
        seed: u32,
    ) -> Result<TailState, DriverError> {
        let mut warm = DeviceBuffer::zeroed(stream, shape.long as usize)?;
        self.0.echelon_init_tail_kernel(
            stream,
            linear_config(shape.long, THREADS),
            &mut warm,
            shape.long,
            seed,
        )?;
        Ok(TailState { shape, warm, seed })
    }

    /// Estimate the scale of T through its action and its transpose.
    /// The return value is the reciprocal of the estimated norm times 1.05.
    /// Exact-zero residuals return zero so the caller can use the head alone.
    pub fn estimate_tail<'scratch>(
        &self,
        args: TailArgs<'_, 'scratch>,
    ) -> Result<&'scratch DeviceBuffer<f32>, DriverError> {
        let TailArgs {
            stream,
            residual,
            state,
            scratch,
        } = args;
        let shape = state.shape;
        if residual.len() < (shape.short * shape.long) as usize
            || scratch.z.len() < shape.long as usize
            || scratch.w.len() < shape.short as usize
        {
            return Err(DriverError(1));
        }
        let vec_grid = grid_x_config(1, THREADS);
        self.0.echelon_load_tail_kernel(
            stream,
            vec_grid,
            &state.warm,
            &mut scratch.z,
            shape.long,
            state.seed,
        )?;
        for iteration in 0..TAIL_ITERATIONS {
            self.0.echelon_thin_vector_kernel(
                stream,
                grid_x_config(shape.short, THREADS),
                residual,
                &scratch.z,
                &mut scratch.w,
                shape.long,
            )?;
            if iteration + 1 == TAIL_ITERATIONS {
                self.0.echelon_tail_scale_kernel(
                    stream,
                    vec_grid,
                    &scratch.w,
                    &mut scratch.inverse,
                    shape.short,
                )?;
            }
            self.0.echelon_tail_transpose_kernel(
                stream,
                linear_config(shape.long, THREADS),
                residual,
                &scratch.w,
                &mut scratch.next_z,
                shape,
            )?;
            self.0.echelon_normalize_tail_kernel(
                stream,
                vec_grid,
                &scratch.next_z,
                &mut scratch.z,
                shape.long,
                state.seed,
            )?;
        }
        self.0.echelon_store_tail_kernel(
            stream,
            linear_config(shape.long, THREADS),
            &scratch.z,
            &mut state.warm,
            shape.long,
        )?;
        Ok(&scratch.inverse)
    }

    /// Overwrite the quantized lift QH with X - QH. The tail estimation
    /// and the final update must use this same rounded residual.
    pub fn residual(
        &self,
        stream: &CudaStream,
        momentum: &DeviceBuffer<f32>,
        projection: &mut DeviceBuffer<f32>,
        shape: Shape,
    ) -> Result<(), DriverError> {
        let len = shape.short * shape.long;
        self.0.echelon_residual_kernel(
            stream,
            linear_config(len, THREADS),
            momentum,
            projection,
            len,
        )
    }

    pub fn add_tail(
        &self,
        stream: &CudaStream,
        inverse: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        head_update: &mut DeviceBuffer<f32>,
        shape: Shape,
    ) -> Result<(), DriverError> {
        let len = shape.short * shape.long;
        self.0.echelon_add_tail_kernel(
            stream,
            linear_config(len, THREADS),
            inverse,
            residual,
            head_update,
            len,
        )
    }

    pub fn normalize_head(
        &self,
        stream: &CudaStream,
        head: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        chunks: &mut DeviceBuffer<f32>,
        shape: Shape,
    ) -> Result<(), DriverError> {
        let len = shape.rank * shape.long;
        let blocks = len.div_ceil(THREADS * 8);
        if chunks.len() < blocks as usize {
            return Err(DriverError(1));
        }
        self.0.echelon_head_sumsq_kernel(
            stream,
            grid_x_config(blocks, THREADS),
            head,
            chunks,
            len,
            blocks,
        )?;
        self.0.echelon_head_inverse_norm_kernel(
            stream,
            grid_x_config(1, THREADS),
            chunks,
            blocks,
        )?;
        self.0.echelon_scale_head_kernel(
            stream,
            linear_config(len, THREADS),
            head,
            chunks,
            out,
            len,
        )
    }
}

#[inline(always)]
fn signed_unit(index: u32, seed: u32, width: u32) -> f32 {
    let mut key = index ^ seed;
    key ^= key >> 16;
    key = key.wrapping_mul(0x7feb_352d);
    key ^= key >> 15;
    key = key.wrapping_mul(0x846c_a68b);
    key ^= key >> 16;
    let sign = if key & 1 == 0 { 1.0 } else { -1.0 };
    sign / sqrt_f32(width as f32)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn normalize(
        width: u32,
        seed: u32,
        mut read: impl FnMut(u32) -> f32,
        mut output: DisjointSlice<f32>,
        shared: &mut SharedArray<f32, WARPS>,
    ) {
        let (tid, lane, warp) = thread_lane_warp();
        let mut square = 0.0;
        for i in (tid..width).step_by(THREADS as usize) {
            let value = read(i);
            square = fma_f32(value, value, square);
        }
        let norm = sqrt_f32(block_sum_shared_f32(shared, square, lane, warp));
        for i in (tid..width).step_by(THREADS as usize) {
            let value = if norm == 0.0 {
                signed_unit(i, seed, width)
            } else {
                read(i) / norm
            };
            unsafe { *output.get_unchecked_mut(i as usize) = value };
        }
    }

    #[kernel]
    pub fn echelon_init_tail_kernel(mut warm: DisjointSlice<u16>, width: u32, seed: u32) {
        let index = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        if index < width {
            unsafe {
                *warm.get_unchecked_mut(index as usize) =
                    cvt_rn_f16_f32(signed_unit(index, seed, width))
            };
        }
    }

    #[kernel]
    pub fn echelon_load_tail_kernel(warm: &[u16], z: DisjointSlice<f32>, width: u32, seed: u32) {
        static mut SHARED: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        normalize(width, seed, |i| cvt_f32_f16(warm[i as usize]), z, unsafe {
            &mut SHARED
        });
    }

    #[kernel]
    pub fn echelon_normalize_tail_kernel(
        input: &[f32],
        output: DisjointSlice<f32>,
        width: u32,
        seed: u32,
    ) {
        static mut SHARED: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        normalize(width, seed, |i| input[i as usize], output, unsafe {
            &mut SHARED
        });
    }

    #[kernel]
    pub fn echelon_store_tail_kernel(z: &[f32], mut warm: DisjointSlice<u16>, width: u32) {
        let index = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        if index < width {
            unsafe { *warm.get_unchecked_mut(index as usize) = cvt_rn_f16_f32(z[index as usize]) };
        }
    }

    #[kernel]
    pub fn echelon_thin_vector_kernel(
        matrix: &[f32],
        vector: &[f32],
        mut output: DisjointSlice<f32>,
        width: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        let (tid, lane, warp) = thread_lane_warp();
        let mut sum = 0.0;
        for column in (tid..width).step_by(THREADS as usize) {
            sum = fma_f32(
                matrix[(row * width + column) as usize],
                vector[column as usize],
                sum,
            );
        }
        let sum = unsafe { block_sum_shared_f32(&mut REDUCTION, sum, lane, warp) };
        if tid == 0 {
            unsafe { *output.get_unchecked_mut(row as usize) = sum };
        }
    }

    #[kernel]
    pub fn echelon_tail_transpose_kernel(
        residual: &[f32],
        w: &[f32],
        mut z: DisjointSlice<f32>,
        shape: Shape,
    ) {
        let column = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        if column < shape.long {
            let mut sum = 0.0;
            for row in 0..shape.short {
                sum = fma_f32(
                    residual[(row * shape.long + column) as usize],
                    w[row as usize],
                    sum,
                );
            }
            unsafe { *z.get_unchecked_mut(column as usize) = sum };
        }
    }

    #[kernel]
    pub fn echelon_tail_scale_kernel(w: &[f32], mut inverse: DisjointSlice<f32>, rows: u32) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let (tid, lane, warp) = thread_lane_warp();
        let mut square = 0.0;
        for row in (tid..rows).step_by(THREADS as usize) {
            let value = w[row as usize];
            square = fma_f32(value, value, square);
        }
        let square = unsafe { block_sum_shared_f32(&mut REDUCTION, square, lane, warp) };
        if tid == 0 {
            let tau = TAIL_MARGIN * sqrt_f32(square);
            unsafe { *inverse.get_unchecked_mut(0) = if tau == 0.0 { 0.0 } else { 1.0 / tau } };
        }
    }

    #[kernel]
    pub fn echelon_residual_kernel(x: &[f32], mut qh: DisjointSlice<f32>, len: u32) {
        let idx = (thread::blockIdx_x() * THREADS + thread::threadIdx_x()) as usize;
        if idx < len as usize {
            let value = unsafe { qh.get_unchecked_mut(idx) };
            *value = x[idx] - *value;
        }
    }

    #[kernel]
    pub fn echelon_add_tail_kernel(
        inverse: &[f32],
        residual: &[f32],
        mut update: DisjointSlice<f32>,
        len: u32,
    ) {
        let idx = (thread::blockIdx_x() * THREADS + thread::threadIdx_x()) as usize;
        if idx < len as usize {
            let value = unsafe { update.get_unchecked_mut(idx) };
            *value = fma_f32(inverse[0], residual[idx], *value);
        }
    }

    #[kernel]
    pub fn echelon_head_sumsq_kernel(
        head: &[f32],
        mut chunks: DisjointSlice<f32>,
        len: u32,
        blocks: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let (tid, lane, warp) = thread_lane_warp();
        let block = thread::blockIdx_x();
        let mut square = 0.0;
        for index in (block * THREADS + tid..len).step_by((blocks * THREADS) as usize) {
            let value = head[index as usize];
            square = fma_f32(value, value, square);
        }
        let square = unsafe { block_sum_shared_f32(&mut REDUCTION, square, lane, warp) };
        if tid == 0 {
            unsafe { *chunks.get_unchecked_mut(block as usize) = square };
        }
    }

    #[kernel]
    pub fn echelon_head_inverse_norm_kernel(mut chunks: DisjointSlice<f32>, count: u32) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let (tid, lane, warp) = thread_lane_warp();
        let mut sum = 0.0;
        let values = chunks.as_mut_ptr();
        for index in (tid..count).step_by(THREADS as usize) {
            sum += unsafe { *values.add(index as usize) };
        }
        let sum = unsafe { block_sum_shared_f32(&mut REDUCTION, sum, lane, warp) };
        if tid == 0 {
            unsafe { *chunks.get_unchecked_mut(0) = 1.0 / max_f32(sqrt_f32(sum), 1.0e-7) };
        }
    }

    #[kernel]
    pub fn echelon_scale_head_kernel(
        head: &[f32],
        inverse_norm: &[f32],
        mut out: DisjointSlice<f32>,
        len: u32,
    ) {
        let idx = (thread::blockIdx_x() * THREADS + thread::threadIdx_x()) as usize;
        if idx < len as usize {
            unsafe { *out.get_unchecked_mut(idx) = head[idx] * inverse_norm[0] };
        }
    }
}
