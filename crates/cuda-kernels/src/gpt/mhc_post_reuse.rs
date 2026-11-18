//! Reuse each feature's four stream inputs for all four post-mix outputs.
//! Each output preserves branch multiplication then h=0,1,2,3 accumulation.
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use std::sync::Arc;

pub struct MhcPostReuseModule {
    kernels: kernels::LoadedModule,
}
pub struct PostArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub branch: &'a DeviceBuffer<f32>,
    pub coefficients: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub dim: u32,
}
impl MhcPostReuseModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        if !crate::module_exports::contains(&module, b"mhc_post_reuse_kernel")? {
            return Err(DriverError(1));
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn post(&self, a: PostArgs<'_, '_>) -> Result<(), DriverError> {
        if a.rows == 0 || a.dim == 0 {
            return Err(DriverError(1));
        }
        let nd = a.rows.checked_mul(a.dim).ok_or(DriverError(1))?;
        let wide = nd.checked_mul(4).ok_or(DriverError(1))?;
        let coeff = a.rows.checked_mul(24).ok_or(DriverError(1))?;
        if a.input.len() < wide as usize
            || a.branch.len() < nd as usize
            || a.coefficients.len() < coeff as usize
            || a.out.len() < wide as usize
        {
            return Err(DriverError(1));
        }
        self.kernels.mhc_post_reuse_kernel(
            a.stream,
            LaunchConfig {
                grid_dim: (nd.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            a.input,
            a.branch,
            a.coefficients,
            a.out,
            a.rows,
            a.dim,
        )
    }
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[inline(always)]
    fn mix(c: &[f32], cb: usize, k: usize, branch: f32, x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
        let mut value = c[cb + 4 + k] * branch;
        value += c[cb + 8 + k] * x0;
        value += c[cb + 12 + k] * x1;
        value += c[cb + 16 + k] * x2;
        value += c[cb + 20 + k] * x3;
        value
    }
    #[kernel]
    pub fn mhc_post_reuse_kernel(
        input: &[f32],
        branch: &[f32],
        c: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i >= n * d {
            return;
        }
        let r = i / d;
        let j = i % d;
        let base = ((r * 4) * d + j) as usize;
        let dim = d as usize;
        let cb = (r * 24) as usize;
        let x0 = input[base];
        let x1 = input[base + dim];
        let x2 = input[base + 2 * dim];
        let x3 = input[base + 3 * dim];
        let f = branch[i as usize];
        unsafe {
            *out.get_unchecked_mut(base) = mix(c, cb, 0, f, x0, x1, x2, x3);
            *out.get_unchecked_mut(base + dim) = mix(c, cb, 1, f, x0, x1, x2, x3);
            *out.get_unchecked_mut(base + 2 * dim) = mix(c, cb, 2, f, x0, x1, x2, x3);
            *out.get_unchecked_mut(base + 3 * dim) = mix(c, cb, 3, f, x0, x1, x2, x3);
        }
    }
}
