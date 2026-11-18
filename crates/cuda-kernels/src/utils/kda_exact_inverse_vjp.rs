//! VJP of the original FP32 unit-lower control inverse before its FP16 cast.
//! The FP16 cast uses the declared straight-through convention. This is the
//! small triangular attention-control solve, not a dense projection backend.
use crate::float_ptx::fma_f32;
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use std::sync::Arc;
const STRIDE: usize = 66;
const TILE: usize = 64 * STRIDE;
pub struct KdaExactInverseVjpModule {
    kernels: kernels::LoadedModule,
}
pub struct VjpArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub lower: &'a DeviceBuffer<f32>,
    pub inverse_grad: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub chunk_batch: u32,
}
impl KdaExactInverseVjpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        if !crate::module_exports::contains(&module, b"kda_exact_inverse_vjp_kernel")? {
            return Err(DriverError(1));
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn vjp(&self, args: VjpArgs<'_, '_>) -> Result<(), DriverError> {
        let n = args
            .chunk_batch
            .checked_mul(4096)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))? as usize;
        if args.lower.len() < n || args.inverse_grad.len() < n || args.out.len() < n {
            return Err(DriverError(1));
        }
        self.kernels.kda_exact_inverse_vjp_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.chunk_batch, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            args.lower,
            args.inverse_grad,
            args.out,
        )
    }
}

#[inline(always)]
fn vjp_body(
    lower: &[f32],
    dy: &[f32],
    mut out: DisjointSlice<f32>,
    raw: &mut SharedArray<f32, { 2 * TILE }>,
) {
    let batch = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    let base = batch * 4096;
    let a = raw.as_mut_ptr();
    let x = unsafe { a.add(TILE) };
    let mut i = tid;
    while i < 4096 {
        let row = i / 64;
        let col = i % 64;
        let at = (row as usize) * STRIDE + col as usize;
        unsafe {
            *a.add(at) = if col < row {
                lower[(base + i) as usize]
            } else {
                0.0
            };
            // The inverse's diagonal is constant1 and its upper triangle is
            // constant0. Their adjoints cannot influence a strict-lower input.
            *x.add(at) = if col < row {
                dy[(base + i) as usize]
            } else {
                0.0
            };
        }
        i += thread::blockDim_x();
    }
    thread::sync_threads();
    // X=solve(A^T,dI). Each thread owns a complete RHS column.
    if tid < 64 {
        let col = tid as usize;
        let mut end = 64;
        while end > 0 {
            end -= 1;
            let row = end as usize;
            let mut value = unsafe { *x.add(row * STRIDE + col) };
            let mut k = row + 1;
            while k < 64 {
                unsafe {
                    value = fma_f32(-*a.add(k * STRIDE + row), *x.add(k * STRIDE + col), value);
                }
                k += 1;
            }
            unsafe {
                *x.add(row * STRIDE + col) = value;
            }
        }
    }
    thread::sync_threads();
    // Y*A^T=X, hence dL=-Y. A new row ownership starts only after every
    // column solve is coherent. Reuse each row in place with forward columns.
    if tid < 64 {
        let row = tid as usize;
        let mut col = 0;
        while col < 64 {
            let mut value = unsafe { *x.add(row * STRIDE + col) };
            let mut k = 0;
            while k < col {
                unsafe {
                    value = fma_f32(-*x.add(row * STRIDE + k), *a.add(col * STRIDE + k), value);
                }
                k += 1;
            }
            unsafe {
                *x.add(row * STRIDE + col) = value;
                *out.get_unchecked_mut((base as usize) + row * 64 + col) =
                    if col < row { -value } else { 0.0 };
            }
            col += 1;
        }
    }
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_exact_inverse_vjp_kernel(lower: &[f32], dy: &[f32], out: DisjointSlice<f32>) {
        static mut RAW: SharedArray<f32, { 2 * TILE }> = SharedArray::UNINIT;
        unsafe {
            vjp_body(lower, dy, out, &mut RAW);
        }
    }
}
