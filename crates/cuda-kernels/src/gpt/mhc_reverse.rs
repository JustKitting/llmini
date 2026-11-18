//! Defer direct vector adjoints until the RMS adjoint's final write. Keep the
//! original control reductions and ordered post + pre + radial additions.
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use std::sync::Arc;
pub struct MhcReverseModule {
    kernels: kernels::LoadedModule,
}
pub struct PostArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub branch: &'a DeviceBuffer<f32>,
    pub coefficients: &'a DeviceBuffer<f32>,
    pub incoming: &'a DeviceBuffer<f32>,
    pub df: &'out mut DeviceBuffer<f32>,
    pub dc: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub dim: u32,
}
pub struct PreArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub coefficients: &'a DeviceBuffer<f32>,
    pub gradient: &'a DeviceBuffer<f32>,
    pub dc: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub dim: u32,
}
pub struct FinishArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub inverse: &'a DeviceBuffer<f32>,
    pub coefficients: &'a DeviceBuffer<f32>,
    pub normalized_gradient: &'a DeviceBuffer<f32>,
    pub branch_gradient: &'a DeviceBuffer<f32>,
    pub current: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub dim: u32,
    pub head: bool,
}
fn shape(rows: u32, dim: u32) -> Result<(usize, usize, usize), DriverError> {
    if rows == 0 || dim == 0 {
        return Err(DriverError(1));
    }
    let nd = rows.checked_mul(dim).ok_or(DriverError(1))?;
    let wide = nd.checked_mul(4).ok_or(DriverError(1))?;
    let c = rows.checked_mul(24).ok_or(DriverError(1))?;
    Ok((nd as usize, wide as usize, c as usize))
}
fn config(rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
impl MhcReverseModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        for entry in [
            b"mhc_reverse_post_prepare_kernel".as_slice(),
            b"mhc_reverse_pre_prepare_kernel".as_slice(),
            b"mhc_reverse_finish_kernel".as_slice(),
        ] {
            if !crate::module_exports::contains(&module, entry)? {
                return Err(DriverError(1));
            }
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn post(&self, a: PostArgs<'_, '_>) -> Result<(), DriverError> {
        let (nd, wide, c) = shape(a.rows, a.dim)?;
        if a.input.len() < wide
            || a.incoming.len() < wide
            || a.branch.len() < nd
            || a.df.len() < nd
            || a.coefficients.len() < c
            || a.dc.len() < c
        {
            return Err(DriverError(1));
        }
        self.kernels.mhc_reverse_post_prepare_kernel(
            a.stream,
            config(a.rows),
            a.input,
            a.branch,
            a.coefficients,
            a.incoming,
            a.df,
            a.dc,
            a.rows,
            a.dim,
        )
    }
    pub fn pre(&self, a: PreArgs<'_, '_>) -> Result<(), DriverError> {
        let (nd, wide, c) = shape(a.rows, a.dim)?;
        if a.input.len() < wide
            || a.gradient.len() < nd
            || a.coefficients.len() < c
            || a.dc.len() < c
        {
            return Err(DriverError(1));
        }
        self.kernels.mhc_reverse_pre_prepare_kernel(
            a.stream,
            config(a.rows),
            a.input,
            a.coefficients,
            a.gradient,
            a.dc,
            a.rows,
            a.dim,
        )
    }
    pub fn finish(&self, a: FinishArgs<'_, '_>) -> Result<(), DriverError> {
        let (nd, wide, c) = shape(a.rows, a.dim)?;
        if a.input.len() < wide
            || a.normalized_gradient.len() < wide
            || a.current.len() < wide
            || a.branch_gradient.len() < nd
            || a.inverse.len() < a.rows as usize
            || a.coefficients.len() < c
        {
            return Err(DriverError(1));
        }
        self.kernels.mhc_reverse_finish_kernel(
            a.stream,
            config(a.rows),
            a.input,
            a.inverse,
            a.coefficients,
            a.normalized_gradient,
            a.branch_gradient,
            a.current,
            a.rows,
            a.dim,
            a.head,
        )
    }
}
#[inline(always)]
fn dot4(c: &[f32], at: usize, g0: f32, g1: f32, g2: f32, g3: f32) -> f32 {
    let mut result = 0.0;
    result += c[at] * g0;
    result += c[at + 1] * g1;
    result += c[at + 2] * g2;
    result += c[at + 3] * g3;
    result
}
#[derive(Clone, Copy)]
struct Pair {
    input: f32,
    gradient: f32,
}
#[inline(always)]
fn pair(input: &[f32], gradient: &[f32], at: usize) -> Pair {
    Pair {
        input: input[at],
        gradient: gradient[at],
    }
}
#[inline(always)]
fn finish_cached_feature(
    out: *mut f32,
    c: &[f32],
    branch_gradient: &[f32],
    row: u32,
    feature: u32,
    inv: f32,
    mean: f32,
    head: bool,
    a: Pair,
    b: Pair,
    e: Pair,
    f: Pair,
) {
    let i = (row * 8192 + feature) as usize;
    let cb = (row * 24) as usize;
    let (g0, g1, g2, g3) = if head {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        unsafe {
            (
                *out.add(i),
                *out.add(i + 2048),
                *out.add(i + 4096),
                *out.add(i + 6144),
            )
        }
    };
    let bg = branch_gradient[(row * 2048 + feature) as usize];
    macro_rules! write {
        ($stream:expr,$pair:ident) => {
            let pre = c[cb + $stream] * bg;
            let direct = if head {
                pre
            } else {
                dot4(c, cb + 8 + $stream * 4, g0, g1, g2, g3) + pre
            };
            unsafe {
                *out.add(i + $stream * 2048) = direct + inv * ($pair.gradient - $pair.input * mean);
            }
        };
    }
    write!(0, a);
    write!(1, b);
    write!(2, e);
    write!(3, f);
}
#[inline(always)]
fn finish_cached_2048(
    input: &[f32],
    inverse: &[f32],
    c: &[f32],
    gradient: &[f32],
    branch_gradient: &[f32],
    mut current: DisjointSlice<f32>,
    row: u32,
    head: bool,
) {
    let t = thread::threadIdx_x();
    let at = (row * 8192 + t) as usize;
    // Literal bindings retain all64 values in registers. A dynamically indexed
    // local array would introduce a local-memory tape and defeat this saving.
    macro_rules! load8 {
        ($a:ident,$b:ident,$e:ident,$f:ident,$g:ident,$h:ident,$i:ident,$j:ident,$offset:expr) => {
            let $a = pair(input, gradient, at + $offset);
            let $b = pair(input, gradient, at + $offset + 256);
            let $e = pair(input, gradient, at + $offset + 512);
            let $f = pair(input, gradient, at + $offset + 768);
            let $g = pair(input, gradient, at + $offset + 1024);
            let $h = pair(input, gradient, at + $offset + 1280);
            let $i = pair(input, gradient, at + $offset + 1536);
            let $j = pair(input, gradient, at + $offset + 1792);
        };
    }
    load8!(a0, a1, a2, a3, a4, a5, a6, a7, 0);
    load8!(b0, b1, b2, b3, b4, b5, b6, b7, 2048);
    load8!(e0, e1, e2, e3, e4, e5, e6, e7, 4096);
    load8!(f0, f1, f2, f3, f4, f5, f6, f7, 6144);
    let mut dot = 0.0;
    macro_rules! sum8 {
        ($a:ident,$b:ident,$e:ident,$f:ident,$g:ident,$h:ident,$i:ident,$j:ident) => {
            dot += $a.input * $a.gradient;
            dot += $b.input * $b.gradient;
            dot += $e.input * $e.gradient;
            dot += $f.input * $f.gradient;
            dot += $g.input * $g.gradient;
            dot += $h.input * $h.gradient;
            dot += $i.input * $i.gradient;
            dot += $j.input * $j.gradient;
        };
    }
    sum8!(a0, a1, a2, a3, a4, a5, a6, a7);
    sum8!(b0, b1, b2, b3, b4, b5, b6, b7);
    sum8!(e0, e1, e2, e3, e4, e5, e6, e7);
    sum8!(f0, f1, f2, f3, f4, f5, f6, f7);
    static mut SUM: SharedArray<f32, 8> = SharedArray::UNINIT;
    let total = crate::block_reduce::block_sum_shared_f32(unsafe { &mut SUM }, dot, t % 32, t / 32);
    let inv = inverse[row as usize];
    let mean = total * inv * inv / 8192.0;
    let out = current.as_mut_ptr();
    macro_rules! finish {
        ($offset:expr,$a:ident,$b:ident,$e:ident,$f:ident) => {
            finish_cached_feature(
                out,
                c,
                branch_gradient,
                row,
                t + $offset,
                inv,
                mean,
                head,
                $a,
                $b,
                $e,
                $f,
            );
        };
    }
    finish!(0, a0, b0, e0, f0);
    finish!(256, a1, b1, e1, f1);
    finish!(512, a2, b2, e2, f2);
    finish!(768, a3, b3, e3, f3);
    finish!(1024, a4, b4, e4, f4);
    finish!(1280, a5, b5, e5, f5);
    finish!(1536, a6, b6, e6, f6);
    finish!(1792, a7, b7, e7, f7);
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn mhc_reverse_post_prepare_kernel(
        input: &[f32],
        branch: &[f32],
        c: &[f32],
        gradient: &[f32],
        df: DisjointSlice<f32>,
        dc: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        crate::mhc_mixing::kernels::post_backward_body::<false>(
            input,
            branch,
            c,
            gradient,
            df,
            core::ptr::null_mut(),
            dc,
            n,
            d,
        );
    }
    #[kernel]
    pub fn mhc_reverse_pre_prepare_kernel(
        input: &[f32],
        c: &[f32],
        gradient: &[f32],
        dc: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        crate::mhc_mixing::kernels::pre_backward_body::<false>(
            input,
            c,
            gradient,
            core::ptr::null_mut(),
            dc,
            n,
            d,
            false,
        );
    }
    #[kernel]
    pub fn mhc_reverse_finish_kernel(
        input: &[f32],
        inverse: &[f32],
        c: &[f32],
        gradient: &[f32],
        branch_gradient: &[f32],
        mut current: DisjointSlice<f32>,
        n: u32,
        d: u32,
        head: bool,
    ) {
        let row = thread::blockIdx_x();
        if row >= n {
            return;
        }
        if d == 2048 {
            finish_cached_2048(
                input,
                inverse,
                c,
                gradient,
                branch_gradient,
                current,
                row,
                head,
            );
            return;
        }
        let t = thread::threadIdx_x();
        let base = (row * 4 * d) as usize;
        let dim = d as usize;
        let cb = (row * 24) as usize;
        let mut dot = 0.0;
        let mut j = t;
        // Preserve the original radial-dot thread assignment and XOR tree.
        while j < 4 * d {
            dot += input[base + j as usize] * gradient[base + j as usize];
            j += 256;
        }
        static mut SUM: SharedArray<f32, 8> = SharedArray::UNINIT;
        let total =
            crate::block_reduce::block_sum_shared_f32(unsafe { &mut SUM }, dot, t % 32, t / 32);
        let inv = inverse[row as usize];
        let mean = total * inv * inv / (4 * d) as f32;
        let out = current.as_mut_ptr();
        j = t;
        while j < d {
            let i = base + j as usize;
            // One thread owns all four streams of this feature. Read every
            // incoming value before writing; no other thread needs them.
            let (g0, g1, g2, g3) = if head {
                (0.0, 0.0, 0.0, 0.0)
            } else {
                unsafe {
                    (
                        *out.add(i),
                        *out.add(i + dim),
                        *out.add(i + 2 * dim),
                        *out.add(i + 3 * dim),
                    )
                }
            };
            let bg = branch_gradient[(row * d + j) as usize];
            let mut h = 0usize;
            while h < 4 {
                let at = i + h * dim;
                let pre = c[cb + h] * bg;
                let direct = if head {
                    pre
                } else {
                    dot4(c, cb + 8 + h * 4, g0, g1, g2, g3) + pre
                };
                unsafe {
                    *out.add(at) = direct + inv * (gradient[at] - input[at] * mean);
                }
                h += 1;
            }
            j += 256;
        }
    }
}
