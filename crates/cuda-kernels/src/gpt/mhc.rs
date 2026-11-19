//! Full four-stream mHC vector/control maps and exact finite-round adjoints.
//! Dense coefficient projections and their adjoints belong to the NVFP4 path.
use crate::{
    atomic::atomic_add_f32,
    block_reduce::block_sum_shared_f32,
    float_ptx::{exp_f32, max_f32, sqrt_f32},
};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
const T: u32 = 256;
const EPS: f32 = 1e-6;
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1. / (1. + exp_f32(-x))
}
#[inline(always)]
fn sum(s: &mut SharedArray<f32, 8>, x: f32) -> f32 {
    let t = thread::threadIdx_x();
    block_sum_shared_f32(s, x, t % 32, t / 32)
}
#[inline(always)]
fn normalize(matrix: &mut [f32; 16], columns: bool) {
    let mut group = 0;
    while group < 4 {
        let mut denom = 0.;
        let mut j = 0;
        while j < 4 {
            let i = if columns {
                j * 4 + group
            } else {
                group * 4 + j
            };
            denom += matrix[i];
            j += 1;
        }
        denom += EPS;
        j = 0;
        while j < 4 {
            let i = if columns {
                j * 4 + group
            } else {
                group * 4 + j
            };
            matrix[i] /= denom;
            j += 1;
        }
        group += 1;
    }
}
#[inline(always)]
fn initial_residual(raw: &[f32], base: &[f32], alpha: f32, start: usize) -> [f32; 16] {
    let mut matrix = [0.; 16];
    let mut j = 0;
    while j < 4 {
        let mut maximum = f32::NEG_INFINITY;
        let mut k = 0;
        while k < 4 {
            let i = j * 4 + k;
            matrix[i] = raw[start + 8 + i] * alpha + base[8 + i];
            maximum = max_f32(maximum, matrix[i]);
            k += 1;
        }
        let mut total = 0.;
        k = 0;
        while k < 4 {
            matrix[j * 4 + k] = exp_f32(matrix[j * 4 + k] - maximum);
            total += matrix[j * 4 + k];
            k += 1;
        }
        k = 0;
        while k < 4 {
            matrix[j * 4 + k] = matrix[j * 4 + k] / total + EPS;
            k += 1;
        }
        j += 1;
    }
    matrix
}
#[cuda_module]
pub mod kernels {
    use super::*;
    static mut REDUCE: SharedArray<f32, 8> = SharedArray::UNINIT;

    #[kernel]
    pub fn mhc_coefficients_kernel(
        raw: &[f32],
        base: &[f32],
        alpha: &[f32],
        mut coefficients: DisjointSlice<f32>,
        n: u32,
        stride: u32,
        head: bool,
    ) {
        let r = thread::blockIdx_x();
        if r >= n || thread::threadIdx_x() != 0 {
            return;
        }
        let start = (r * stride) as usize;
        let b = (r * 24) as usize;
        let mut i = 0;
        while i < 24 {
            unsafe {
                *coefficients.get_unchecked_mut(b + i) = 0.;
            }
            i += 1;
        }
        i = 0;
        while i < 4 {
            unsafe {
                *coefficients.get_unchecked_mut(b + i) =
                    sigmoid(raw[start + i] * alpha[0] + base[i]) + EPS;
            }
            i += 1;
        }
        if head {
            return;
        }
        i = 0;
        while i < 4 {
            unsafe {
                *coefficients.get_unchecked_mut(b + 4 + i) =
                    2. * sigmoid(raw[start + 4 + i] * alpha[1] + base[4 + i]);
            }
            i += 1;
        }
        let mut matrix = initial_residual(raw, base, alpha[2], start);
        let mut stage = 1;
        while stage < 40 {
            normalize(&mut matrix, stage % 2 == 1);
            stage += 1;
        }
        i = 0;
        while i < 16 {
            unsafe {
                *coefficients.get_unchecked_mut(b + 8 + i) = matrix[i];
            }
            i += 1;
        }
    }

    #[kernel]
    pub fn mhc_coefficients_backward_kernel(
        raw: &[f32],
        base: &[f32],
        alpha: &[f32],
        dc: &[f32],
        mut draw: DisjointSlice<f32>,
        mut dbase: DisjointSlice<f32>,
        mut dalpha: DisjointSlice<f32>,
        n: u32,
        stride: u32,
        head: bool,
    ) {
        let r = thread::blockIdx_x();
        if r >= n || thread::threadIdx_x() != 0 {
            return;
        }
        let start = (r * stride) as usize;
        let b = (r * 24) as usize;
        let mut derivative = [0.; 24];
        let mut i = 0;
        while i < 4 {
            let s = sigmoid(raw[start + i] * alpha[0] + base[i]);
            derivative[i] = dc[b + i] * s * (1. - s);
            i += 1;
        }
        if !head {
            i = 0;
            while i < 4 {
                let s = sigmoid(raw[start + 4 + i] * alpha[1] + base[4 + i]);
                derivative[4 + i] = dc[b + 4 + i] * 2. * s * (1. - s);
                i += 1;
            }
            // Stage0 is row-softmax + epsilon; stages1..39 alternate column/row,
            // ending on columns. Reverse this exact finite program, not a fixed point.
            let mut history = [0_f32; 640];
            let mut matrix = initial_residual(raw, base, alpha[2], start);
            i = 0;
            while i < 16 {
                history[i] = matrix[i];
                i += 1;
            }
            let mut stage = 1;
            while stage < 40 {
                normalize(&mut matrix, stage % 2 == 1);
                i = 0;
                while i < 16 {
                    unsafe {
                        *history.get_unchecked_mut(stage * 16 + i) = matrix[i];
                    }
                    i += 1;
                }
                stage += 1;
            }
            let mut g = [0.; 16];
            i = 0;
            while i < 16 {
                g[i] = dc[b + 8 + i];
                i += 1;
            }
            stage = 39;
            while stage > 0 {
                let columns = stage % 2 == 1;
                let mut group = 0;
                while group < 4 {
                    let mut denom = 0.;
                    let mut dot = 0.;
                    let mut j = 0;
                    while j < 4 {
                        let at = if columns {
                            j * 4 + group
                        } else {
                            group * 4 + j
                        };
                        unsafe {
                            denom += *history.get_unchecked((stage - 1) * 16 + at);
                            dot += g[at] * *history.get_unchecked(stage * 16 + at);
                        }
                        j += 1;
                    }
                    denom += EPS;
                    j = 0;
                    while j < 4 {
                        let at = if columns {
                            j * 4 + group
                        } else {
                            group * 4 + j
                        };
                        g[at] = (g[at] - dot) / denom;
                        j += 1;
                    }
                    group += 1;
                }
                stage -= 1;
            }
            let mut j = 0;
            while j < 4 {
                let mut dot = 0.;
                let mut k = 0;
                while k < 4 {
                    let at = j * 4 + k;
                    dot += g[at] * (history[at] - EPS);
                    k += 1;
                }
                k = 0;
                while k < 4 {
                    let at = j * 4 + k;
                    derivative[8 + at] = (history[at] - EPS) * (g[at] - dot);
                    k += 1;
                }
                j += 1;
            }
        }
        i = 0;
        while i < stride as usize {
            unsafe {
                *draw.get_unchecked_mut(start + i) = 0.;
            }
            i += 1;
        }
        let count = if head { 4 } else { 24 };
        let mut scale_grad = [0.; 3];
        i = 0;
        while i < count {
            let group = if i < 4 {
                0
            } else if i < 8 {
                1
            } else {
                2
            };
            unsafe {
                *draw.get_unchecked_mut(start + i) = derivative[i] * alpha[group];
                atomic_add_f32(dbase.as_mut_ptr().add(i), derivative[i]);
            }
            scale_grad[group] += derivative[i] * raw[start + i];
            i += 1;
        }
        let groups = if head { 1 } else { 3 };
        i = 0;
        while i < groups {
            unsafe {
                atomic_add_f32(dalpha.as_mut_ptr().add(i), scale_grad[i]);
            }
            i += 1;
        }
    }

    #[kernel]
    pub fn mhc_expand_kernel(input: &[f32], mut out: DisjointSlice<f32>, n: u32, d: u32) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i < n * 4 * d {
            let r = i / (4 * d);
            let j = i % d;
            unsafe {
                *out.get_unchecked_mut(i as usize) = input[(r * d + j) as usize];
            }
        }
    }
    #[kernel]
    pub fn mhc_expand_backward_kernel(input: &[f32], mut out: DisjointSlice<f32>, n: u32, d: u32) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i < n * d {
            let r = i / d;
            let j = i % d;
            let mut value = 0.;
            let mut k = 0;
            while k < 4 {
                value += input[((r * 4 + k) * d + j) as usize];
                k += 1;
            }
            unsafe {
                *out.get_unchecked_mut(i as usize) = value;
            }
        }
    }
    #[kernel]
    pub fn mhc_rms_kernel(
        input: &[f32],
        mut out: DisjointSlice<f32>,
        mut inv: DisjointSlice<f32>,
        n: u32,
        width: u32,
    ) {
        let r = thread::blockIdx_x();
        if r >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let s = unsafe { &mut REDUCE };
        let base = (r * width) as usize;
        let mut square = 0.;
        let mut j = t;
        while j < width {
            let x = input[base + j as usize];
            square += x * x;
            j += T;
        }
        let inverse = 1. / sqrt_f32(sum(s, square) / width as f32 + EPS);
        if t == 0 {
            unsafe {
                *inv.get_unchecked_mut(r as usize) = inverse;
            }
        }
        j = t;
        while j < width {
            unsafe {
                *out.get_unchecked_mut(base + j as usize) = input[base + j as usize] * inverse;
            }
            j += T;
        }
    }
    #[kernel]
    pub fn mhc_rms_backward_kernel(
        input: &[f32],
        inv: &[f32],
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        width: u32,
    ) {
        let r = thread::blockIdx_x();
        if r >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let s = unsafe { &mut REDUCE };
        let base = (r * width) as usize;
        let mut dot = 0.;
        let mut j = t;
        while j < width {
            dot += input[base + j as usize] * gradient[base + j as usize];
            j += T;
        }
        let inverse = inv[r as usize];
        let mean = sum(s, dot) * inverse * inverse / width as f32;
        j = t;
        while j < width {
            let at = base + j as usize;
            unsafe {
                *out.get_unchecked_mut(at) += inverse * (gradient[at] - input[at] * mean);
            }
            j += T;
        }
    }

    #[kernel]
    pub fn mhc_pre_kernel(
        input: &[f32],
        coefficients: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i >= n * d {
            return;
        }
        let r = i / d;
        let j = i % d;
        let mut value = 0.;
        let mut k = 0;
        while k < 4 {
            value += coefficients[(r * 24 + k) as usize] * input[((r * 4 + k) * d + j) as usize];
            k += 1;
        }
        unsafe {
            *out.get_unchecked_mut(i as usize) = value;
        }
    }
    #[kernel]
    pub fn mhc_post_kernel(
        input: &[f32],
        branch: &[f32],
        coefficients: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i >= n * 4 * d {
            return;
        }
        let r = i / (4 * d);
        let k = (i / d) % 4;
        let j = i % d;
        let mut value = coefficients[(r * 24 + 4 + k) as usize] * branch[(r * d + j) as usize];
        let mut h = 0;
        while h < 4 {
            value += coefficients[(r * 24 + 8 + h * 4 + k) as usize]
                * input[((r * 4 + h) * d + j) as usize];
            h += 1;
        }
        unsafe {
            *out.get_unchecked_mut(i as usize) = value;
        }
    }
    #[kernel]
    pub fn mhc_post_backward_kernel(
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
        let r = thread::blockIdx_x();
        if r >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let s = unsafe { &mut REDUCE };
        let mut j = t;
        while j < d {
            let mut value = 0.;
            let mut k = 0;
            while k < 4 {
                value += c[(r * 24 + 4 + k) as usize] * gradient[((r * 4 + k) * d + j) as usize];
                k += 1;
            }
            unsafe {
                *df.get_unchecked_mut((r * d + j) as usize) = value;
            }
            let mut h = 0;
            while h < 4 {
                let mut direct = 0.;
                k = 0;
                while k < 4 {
                    direct += c[(r * 24 + 8 + h * 4 + k) as usize]
                        * gradient[((r * 4 + k) * d + j) as usize];
                    k += 1;
                }
                unsafe {
                    *dx.get_unchecked_mut(((r * 4 + h) * d + j) as usize) = direct;
                }
                h += 1;
            }
            j += T;
        }
        let mut k = 0;
        while k < 4 {
            let mut local = 0.;
            j = t;
            while j < d {
                local += gradient[((r * 4 + k) * d + j) as usize] * branch[(r * d + j) as usize];
                j += T;
            }
            let total = sum(s, local);
            if t == 0 {
                unsafe {
                    *dc.get_unchecked_mut((r * 24 + 4 + k) as usize) = total;
                }
            }
            let mut h = 0;
            while h < 4 {
                local = 0.;
                j = t;
                while j < d {
                    local += gradient[((r * 4 + k) * d + j) as usize]
                        * input[((r * 4 + h) * d + j) as usize];
                    j += T;
                }
                let total = sum(s, local);
                if t == 0 {
                    unsafe {
                        *dc.get_unchecked_mut((r * 24 + 8 + h * 4 + k) as usize) = total;
                    }
                }
                h += 1;
            }
            k += 1;
        }
    }
    #[kernel]
    pub fn mhc_pre_backward_kernel(
        input: &[f32],
        c: &[f32],
        gradient: &[f32],
        mut dx: DisjointSlice<f32>,
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
        let s = unsafe { &mut REDUCE };
        let mut h = 0;
        while h < 4 {
            let mut dot = 0.;
            let mut j = t;
            while j < d {
                let at = ((r * 4 + h) * d + j) as usize;
                let g = gradient[(r * d + j) as usize];
                dot += input[at] * g;
                unsafe {
                    let contribution = c[(r * 24 + h) as usize] * g;
                    if overwrite {
                        *dx.get_unchecked_mut(at) = contribution;
                    } else {
                        *dx.get_unchecked_mut(at) += contribution;
                    }
                }
                j += T;
            }
            let total = sum(s, dot);
            if t == 0 {
                unsafe {
                    *dc.get_unchecked_mut((r * 24 + h) as usize) = total;
                }
            }
            h += 1;
        }
    }
}
