//! Fuse independent mHC vector adjoints and batch their scalar reductions.
//! Preserve the original per-thread feature order and both XOR reduction
//! stages. No projection arithmetic, gradient branch, RNG or parameter changes.
use crate::warp_reduce::warp_sum_f32;
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use std::sync::Arc;
pub struct MhcMixingModule {
    kernels: kernels::LoadedModule,
}
pub struct PostBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub branch: &'a DeviceBuffer<f32>,
    pub coefficients: &'a DeviceBuffer<f32>,
    pub gradient: &'a DeviceBuffer<f32>,
    pub df: &'out mut DeviceBuffer<f32>,
    pub dx: &'out mut DeviceBuffer<f32>,
    pub dc: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub dim: u32,
}
pub struct PreBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub coefficients: &'a DeviceBuffer<f32>,
    pub gradient: &'a DeviceBuffer<f32>,
    pub dx: &'out mut DeviceBuffer<f32>,
    pub dc: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub dim: u32,
    pub overwrite: bool,
}
fn shape(rows: u32, dim: u32) -> Result<(usize, usize, usize), DriverError> {
    if rows == 0 || dim == 0 {
        return Err(DriverError(1));
    }
    let nd = rows.checked_mul(dim).ok_or(DriverError(1))?;
    let wide = nd.checked_mul(4).ok_or(DriverError(1))?;
    let coeff = rows.checked_mul(24).ok_or(DriverError(1))?;
    Ok((nd as usize, wide as usize, coeff as usize))
}
fn config(rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
impl MhcMixingModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        for name in [
            b"mhc_post_backward_batched_kernel".as_slice(),
            b"mhc_pre_backward_batched_kernel".as_slice(),
        ] {
            if !crate::module_exports::contains(&module, name)? {
                return Err(DriverError(1));
            }
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn post_backward(&self, a: PostBackwardArgs<'_, '_>) -> Result<(), DriverError> {
        let (nd, wide, coeff) = shape(a.rows, a.dim)?;
        if a.input.len() < wide
            || a.branch.len() < nd
            || a.coefficients.len() < coeff
            || a.gradient.len() < wide
            || a.df.len() < nd
            || a.dx.len() < wide
            || a.dc.len() < coeff
        {
            return Err(DriverError(1));
        }
        self.kernels.mhc_post_backward_batched_kernel(
            a.stream,
            config(a.rows),
            a.input,
            a.branch,
            a.coefficients,
            a.gradient,
            a.df,
            a.dx,
            a.dc,
            a.rows,
            a.dim,
        )
    }
    pub fn pre_backward(&self, a: PreBackwardArgs<'_, '_>) -> Result<(), DriverError> {
        let (nd, wide, coeff) = shape(a.rows, a.dim)?;
        if a.input.len() < wide
            || a.coefficients.len() < coeff
            || a.gradient.len() < nd
            || a.dx.len() < wide
            || a.dc.len() < coeff
        {
            return Err(DriverError(1));
        }
        self.kernels.mhc_pre_backward_batched_kernel(
            a.stream,
            config(a.rows),
            a.input,
            a.coefficients,
            a.gradient,
            a.dx,
            a.dc,
            a.rows,
            a.dim,
            a.overwrite,
        )
    }
}
type PostSums = SharedArray<f32, 160>;
type PreSums = SharedArray<f32, 32>;
#[inline(always)]
fn publish<const N: usize>(
    s: &mut SharedArray<f32, N>,
    value: f32,
    metric: usize,
    lane: u32,
    warp: u32,
) {
    let total = warp_sum_f32(value);
    if lane == 0 {
        s[metric * 8 + warp as usize] = total;
    }
}
#[inline(always)]
fn dot4(a0: f32, a1: f32, a2: f32, a3: f32, b0: f32, b1: f32, b2: f32, b3: f32) -> f32 {
    let mut sum = 0.0;
    sum += a0 * b0;
    sum += a1 * b1;
    sum += a2 * b2;
    sum += a3 * b3;
    sum
}
#[inline(always)]
fn pre_write(out: *mut f32, index: usize, coefficient: f32, gradient: f32, overwrite: bool) {
    let contribution = coefficient * gradient;
    unsafe {
        *out.add(index) = if overwrite {
            contribution
        } else {
            *out.add(index) + contribution
        };
    }
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn mhc_post_backward_batched_kernel(
        input: &[f32],
        branch: &[f32],
        c: &[f32],
        gradient: &[f32],
        mut df: DisjointSlice<f32>,
        mut dx: DisjointSlice<f32>,
        mut dc: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        post_backward_body::<true>(input, branch, c, gradient, df, dx.as_mut_ptr(), dc, n, d);
    }
    #[inline(always)]
    pub(crate) fn post_backward_body<const WRITE_DX: bool>(
        input: &[f32],
        branch: &[f32],
        c: &[f32],
        gradient: &[f32],
        mut df: DisjointSlice<f32>,
        dx: *mut f32,
        mut dc: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        let r = thread::blockIdx_x();
        if r >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let lane = t & 31;
        let warp = t / 32;
        let base = (r * 4 * d) as usize;
        let dim = d as usize;
        let cb = (r * 24) as usize;
        let mut b0 = 0.0;
        let mut b1 = 0.0;
        let mut b2 = 0.0;
        let mut b3 = 0.0;
        let mut m00 = 0.0;
        let mut m01 = 0.0;
        let mut m02 = 0.0;
        let mut m03 = 0.0;
        let mut m10 = 0.0;
        let mut m11 = 0.0;
        let mut m12 = 0.0;
        let mut m13 = 0.0;
        let mut m20 = 0.0;
        let mut m21 = 0.0;
        let mut m22 = 0.0;
        let mut m23 = 0.0;
        let mut m30 = 0.0;
        let mut m31 = 0.0;
        let mut m32 = 0.0;
        let mut m33 = 0.0;
        let mut j = t;
        while j < d {
            let i = base + j as usize;
            let g0 = gradient[i];
            let g1 = gradient[i + dim];
            let g2 = gradient[i + 2 * dim];
            let g3 = gradient[i + 3 * dim];
            let x0 = input[i];
            let x1 = input[i + dim];
            let x2 = input[i + 2 * dim];
            let x3 = input[i + 3 * dim];
            let f = branch[(r * d + j) as usize];
            unsafe {
                *df.get_unchecked_mut((r * d + j) as usize) =
                    dot4(c[cb + 4], c[cb + 5], c[cb + 6], c[cb + 7], g0, g1, g2, g3);
                if WRITE_DX {
                    *dx.add(i) = dot4(c[cb + 8], c[cb + 9], c[cb + 10], c[cb + 11], g0, g1, g2, g3);
                    *dx.add(i + dim) = dot4(
                        c[cb + 12],
                        c[cb + 13],
                        c[cb + 14],
                        c[cb + 15],
                        g0,
                        g1,
                        g2,
                        g3,
                    );
                    *dx.add(i + 2 * dim) = dot4(
                        c[cb + 16],
                        c[cb + 17],
                        c[cb + 18],
                        c[cb + 19],
                        g0,
                        g1,
                        g2,
                        g3,
                    );
                    *dx.add(i + 3 * dim) = dot4(
                        c[cb + 20],
                        c[cb + 21],
                        c[cb + 22],
                        c[cb + 23],
                        g0,
                        g1,
                        g2,
                        g3,
                    );
                }
            }
            b0 += g0 * f;
            b1 += g1 * f;
            b2 += g2 * f;
            b3 += g3 * f;
            m00 += g0 * x0;
            m01 += g1 * x0;
            m02 += g2 * x0;
            m03 += g3 * x0;
            m10 += g0 * x1;
            m11 += g1 * x1;
            m12 += g2 * x1;
            m13 += g3 * x1;
            m20 += g0 * x2;
            m21 += g1 * x2;
            m22 += g2 * x2;
            m23 += g3 * x2;
            m30 += g0 * x3;
            m31 += g1 * x3;
            m32 += g2 * x3;
            m33 += g3 * x3;
            j += 256;
        }
        static mut SUMS: PostSums = SharedArray::UNINIT;
        let sums = unsafe { &mut SUMS };
        publish(sums, b0, 0, lane, warp);
        publish(sums, b1, 1, lane, warp);
        publish(sums, b2, 2, lane, warp);
        publish(sums, b3, 3, lane, warp);
        publish(sums, m00, 4, lane, warp);
        publish(sums, m01, 5, lane, warp);
        publish(sums, m02, 6, lane, warp);
        publish(sums, m03, 7, lane, warp);
        publish(sums, m10, 8, lane, warp);
        publish(sums, m11, 9, lane, warp);
        publish(sums, m12, 10, lane, warp);
        publish(sums, m13, 11, lane, warp);
        publish(sums, m20, 12, lane, warp);
        publish(sums, m21, 13, lane, warp);
        publish(sums, m22, 14, lane, warp);
        publish(sums, m23, 15, lane, warp);
        publish(sums, m30, 16, lane, warp);
        publish(sums, m31, 17, lane, warp);
        publish(sums, m32, 18, lane, warp);
        publish(sums, m33, 19, lane, warp);
        thread::sync_threads();
        let mut metric = warp;
        while metric < 20 {
            let partial = if lane < 8 {
                sums[(metric * 8 + lane) as usize]
            } else {
                0.0
            };
            let total = warp_sum_f32(partial);
            if lane == 0 {
                unsafe {
                    *dc.get_unchecked_mut(cb + 4 + metric as usize) = total;
                }
            }
            metric += 8;
        }
    }
    #[kernel]
    pub fn mhc_pre_backward_batched_kernel(
        input: &[f32],
        c: &[f32],
        gradient: &[f32],
        mut dx: DisjointSlice<f32>,
        mut dc: DisjointSlice<f32>,
        n: u32,
        d: u32,
        overwrite: bool,
    ) {
        pre_backward_body::<true>(input, c, gradient, dx.as_mut_ptr(), dc, n, d, overwrite);
    }
    #[inline(always)]
    pub(crate) fn pre_backward_body<const WRITE_DX: bool>(
        input: &[f32],
        c: &[f32],
        gradient: &[f32],
        out: *mut f32,
        mut dc: DisjointSlice<f32>,
        n: u32,
        d: u32,
        overwrite: bool,
    ) {
        let r = thread::blockIdx_x();
        if r >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let lane = t & 31;
        let warp = t / 32;
        let base = (r * 4 * d) as usize;
        let dim = d as usize;
        let cb = (r * 24) as usize;
        let mut a0 = 0.0;
        let mut a1 = 0.0;
        let mut a2 = 0.0;
        let mut a3 = 0.0;
        let mut j = t;
        while j < d {
            let i = base + j as usize;
            let g = gradient[(r * d + j) as usize];
            a0 += input[i] * g;
            a1 += input[i + dim] * g;
            a2 += input[i + 2 * dim] * g;
            a3 += input[i + 3 * dim] * g;
            if WRITE_DX {
                pre_write(out, i, c[cb], g, overwrite);
                pre_write(out, i + dim, c[cb + 1], g, overwrite);
                pre_write(out, i + 2 * dim, c[cb + 2], g, overwrite);
                pre_write(out, i + 3 * dim, c[cb + 3], g, overwrite);
            }
            j += 256;
        }
        static mut SUMS: PreSums = SharedArray::UNINIT;
        let sums = unsafe { &mut SUMS };
        publish(sums, a0, 0, lane, warp);
        publish(sums, a1, 1, lane, warp);
        publish(sums, a2, 2, lane, warp);
        publish(sums, a3, 3, lane, warp);
        thread::sync_threads();
        if warp < 4 {
            let partial = if lane < 8 {
                sums[(warp * 8 + lane) as usize]
            } else {
                0.0
            };
            let total = warp_sum_f32(partial);
            if lane == 0 {
                unsafe {
                    *dc.get_unchecked_mut(cb + warp as usize) = total;
                }
            }
        }
    }
}
