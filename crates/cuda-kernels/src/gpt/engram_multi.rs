//! Full four-stream Engram vector graph; shared lookup/value, independent keys,
//! query/key/conv norms and depthwise channel groups. Dense products stay NVFP4.
use crate::{
    atomic::atomic_add_f32,
    block_reduce::block_sum_shared_f32,
    float_ptx::{abs_f32, exp_f32, max_f32, sqrt_f32},
};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
const T: u32 = 256;
#[inline(always)]
fn sig(x: f32) -> f32 {
    1.0 / (1.0 + exp_f32(-x))
}
#[inline(always)]
fn sum(s: &mut SharedArray<f32, 8>, x: f32) -> f32 {
    let t = thread::threadIdx_x();
    block_sum_shared_f32(s, x, t % 32, t / 32)
}
#[cuda_module]
pub mod kernels {
    use super::*;
    static mut REDUCE: SharedArray<f32, 8> = SharedArray::UNINIT;
    #[kernel]
    pub fn engram4_gate_kernel(
        h: &[f32],
        k: &[f32],
        v: &[f32],
        gh: &[f32],
        gk: &[f32],
        mut stats: DisjointSlice<f32>,
        mut u: DisjointSlice<f32>,
        n: u32,
        d: u32,
        gate_rms_epsilon: f32,
    ) {
        let r = thread::blockIdx_x();
        if r >= n * 4 {
            return;
        }
        let t = thread::threadIdx_x();
        let base = (r * d) as usize;
        let gamma_base = (r % 4 * d) as usize;
        let value_base = (r / 4 * d) as usize;
        let s = unsafe { &mut REDUCE };
        let mut sh = 0.;
        let mut sk = 0.;
        let mut dot = 0.;
        let mut j = t;
        while j < d {
            let x = h[base + j as usize];
            let y = k[base + j as usize];
            sh += x * x;
            sk += y * y;
            dot += x * gh[gamma_base + j as usize] * y * gk[gamma_base + j as usize];
            j += T;
        }
        let ih = 1.0 / sqrt_f32(sum(s, sh) / d as f32 + gate_rms_epsilon);
        let ik = 1.0 / sqrt_f32(sum(s, sk) / d as f32 + gate_rms_epsilon);
        let raw = sum(s, dot) * ih * ik / sqrt_f32(d as f32);
        let signed = if raw > 0. {
            1.
        } else if raw < 0. {
            -1.
        } else {
            0.
        };
        let g = sig(signed * sqrt_f32(max_f32(abs_f32(raw), 1e-6)));
        if t == 0 {
            unsafe {
                *stats.get_unchecked_mut((r * 4) as usize) = ih;
                *stats.get_unchecked_mut((r * 4 + 1) as usize) = ik;
                *stats.get_unchecked_mut((r * 4 + 2) as usize) = raw;
                *stats.get_unchecked_mut((r * 4 + 3) as usize) = g;
            }
        }
        j = t;
        while j < d {
            unsafe {
                *u.get_unchecked_mut(base + j as usize) = g * v[value_base + j as usize];
            }
            j += T;
        }
    }
    #[kernel]
    pub fn engram4_norm_kernel(
        u: &[f32],
        gamma: &[f32],
        mut out: DisjointSlice<f32>,
        mut inv: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        let r = thread::blockIdx_x();
        if r >= n * 4 {
            return;
        }
        let t = thread::threadIdx_x();
        let b = (r * d) as usize;
        let gamma_base = (r % 4 * d) as usize;
        let s = unsafe { &mut REDUCE };
        let mut a = 0.;
        let mut j = t;
        while j < d {
            a += u[b + j as usize] * u[b + j as usize];
            j += T;
        }
        let v = 1. / sqrt_f32(sum(s, a) / d as f32 + 1e-5);
        if t == 0 {
            unsafe {
                *inv.get_unchecked_mut(r as usize) = v;
            }
        }
        j = t;
        while j < d {
            unsafe {
                *out.get_unchecked_mut(b + j as usize) =
                    u[b + j as usize] * v * gamma[gamma_base + j as usize];
            }
            j += T;
        }
    }
    #[kernel]
    pub fn engram4_conv_add_kernel(
        u: &[f32],
        un: &[f32],
        w: &[f32],
        mut pre: DisjointSlice<f32>,
        mut residual: DisjointSlice<f32>,
        n: u32,
        d: u32,
        seq: u32,
        dilation: u32,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i >= n * 4 * d {
            return;
        }
        let r = i / d;
        let j = i % d;
        let mut z = 0.;
        let mut c = 0;
        while c < 4 {
            let lag = dilation * (3 - c);
            if r / 4 % seq >= lag {
                z += w[((r % 4 * d + j) * 4 + c) as usize] * un[((r - lag * 4) * d + j) as usize];
            }
            c += 1;
        }
        unsafe {
            *pre.get_unchecked_mut(i as usize) = z;
            *residual.get_unchecked_mut(i as usize) += u[i as usize] + z * sig(z);
        }
    }
    #[kernel]
    pub fn engram4_conv_input_backward_kernel(
        delta: &[f32],
        w: &[f32],
        mut dun: DisjointSlice<f32>,
        n: u32,
        d: u32,
        seq: u32,
        dilation: u32,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i >= n * 4 * d {
            return;
        }
        let r = i / d;
        let j = i % d;
        let mut a = 0.;
        let mut c = 0;
        while c < 4 {
            let lag = dilation * (3 - c);
            if r / 4 % seq + lag < seq && r + lag * 4 < n * 4 {
                a +=
                    w[((r % 4 * d + j) * 4 + c) as usize] * delta[((r + lag * 4) * d + j) as usize];
            }
            c += 1;
        }
        unsafe {
            *dun.get_unchecked_mut(i as usize) = a;
        }
    }
    #[kernel]
    pub fn engram4_norm_backward_kernel(
        u: &[f32],
        dun: &[f32],
        gamma: &[f32],
        inv: &[f32],
        direct: &[f32],
        mut du: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        let r = thread::blockIdx_x();
        if r >= n * 4 {
            return;
        }
        let t = thread::threadIdx_x();
        let b = (r * d) as usize;
        let gamma_base = (r % 4 * d) as usize;
        let s = unsafe { &mut REDUCE };
        let mut a = 0.;
        let mut j = t;
        while j < d {
            a += u[b + j as usize] * dun[b + j as usize] * gamma[gamma_base + j as usize];
            j += T;
        }
        let iv = inv[r as usize];
        let m = sum(s, a) * iv * iv / d as f32;
        j = t;
        while j < d {
            let i = b + j as usize;
            unsafe {
                *du.get_unchecked_mut(i) =
                    direct[i] + iv * (dun[i] * gamma[gamma_base + j as usize] - u[i] * m);
            }
            j += T;
        }
    }
    #[kernel]
    pub fn engram4_gate_backward_kernel(
        h: &[f32],
        k: &[f32],
        v: &[f32],
        gh: &[f32],
        gk: &[f32],
        stats: &[f32],
        du: &[f32],
        mut direct: DisjointSlice<f32>,
        mut dk: DisjointSlice<f32>,
        mut dv: DisjointSlice<f32>,
        mut d_dot: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        let token = thread::blockIdx_x();
        if token >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let s = unsafe { &mut REDUCE };
        let value_base = (token * d) as usize;
        // A block owns one token and all four streams. Accumulate shared dV
        // in stream order, then run its dense backward once at the host level.
        let mut stream = 0;
        while stream < 4 {
            let r = token * 4 + stream;
            let b = (r * d) as usize;
            let gamma_base = (stream * d) as usize;
            let mut a = 0.;
            let mut j = t;
            while j < d {
                a += du[b + j as usize] * v[value_base + j as usize];
                j += T;
            }
            let ih = stats[(r * 4) as usize];
            let ik = stats[(r * 4 + 1) as usize];
            let raw = stats[(r * 4 + 2) as usize];
            let gate = stats[(r * 4 + 3) as usize];
            let root_grad = if abs_f32(raw) >= 1e-6 {
                0.5 / sqrt_f32(abs_f32(raw))
            } else {
                0.
            };
            let g = sum(s, a) * gate * (1. - gate) * root_grad / sqrt_f32(d as f32);
            if t == 0 {
                unsafe {
                    *d_dot.get_unchecked_mut(r as usize) = g;
                }
            }
            let mut ph = 0.;
            let mut pk = 0.;
            j = t;
            while j < d {
                let i = b + j as usize;
                let dhn = g * k[i] * ik * gk[gamma_base + j as usize] * gh[gamma_base + j as usize];
                let dkn = g * h[i] * ih * gh[gamma_base + j as usize] * gk[gamma_base + j as usize];
                ph += dhn * h[i];
                pk += dkn * k[i];
                j += T;
            }
            let mh = sum(s, ph) * ih * ih / d as f32;
            let mk = sum(s, pk) * ik * ik / d as f32;
            j = t;
            while j < d {
                let i = b + j as usize;
                unsafe {
                    *direct.get_unchecked_mut(i) += ih
                        * (g * k[i]
                            * ik
                            * gk[gamma_base + j as usize]
                            * gh[gamma_base + j as usize]
                            - h[i] * mh);
                    *dk.get_unchecked_mut(i) = ik
                        * (g * h[i]
                            * ih
                            * gh[gamma_base + j as usize]
                            * gk[gamma_base + j as usize]
                            - k[i] * mk);
                    if stream == 0 {
                        *dv.get_unchecked_mut(value_base + j as usize) = du[i] * gate;
                    } else {
                        *dv.get_unchecked_mut(value_base + j as usize) += du[i] * gate;
                    }
                }
                j += T;
            }

            stream += 1;
        }
    }
    #[kernel]
    pub fn engram4_parameter_backward_kernel(
        h: &[f32],
        k: &[f32],
        gh: &[f32],
        gk: &[f32],
        stats: &[f32],
        d_dot: &[f32],
        u: &[f32],
        un: &[f32],
        inv_u: &[f32],
        dun: &[f32],
        delta: &[f32],
        mut dgh: DisjointSlice<f32>,
        mut dgk: DisjointSlice<f32>,
        mut dgu: DisjointSlice<f32>,
        mut dw: DisjointSlice<f32>,
        n: u32,
        d: u32,
        seq: u32,
        dilation: u32,
    ) {
        let index = thread::blockIdx_x() * T + thread::threadIdx_x();
        if index >= d * 4 * 16 {
            return;
        }
        let j = index % d;
        let part = index / d % 16;
        let stream = index / (d * 16);
        let gamma_index = (stream * d + j) as usize;
        let mut ah = 0.;
        let mut ak = 0.;
        let mut au = 0.;
        let mut w0 = 0.;
        let mut w1 = 0.;
        let mut w2 = 0.;
        let mut w3 = 0.;
        let first_lag = 3 * dilation;
        let second_lag = 2 * dilation;
        let mut r = part;
        while r < n {
            let i = ((r * 4 + stream) * d + j) as usize;
            let h0 = h[i] * stats[((r * 4 + stream) * 4) as usize];
            let k0 = k[i] * stats[((r * 4 + stream) * 4 + 1) as usize];
            let g = d_dot[(r * 4 + stream) as usize];
            ah += g * k0 * gk[gamma_index] * h0;
            ak += g * h0 * gh[gamma_index] * k0;
            au += dun[i] * u[i] * inv_u[(r * 4 + stream) as usize];
            let e = delta[i];
            let position = r % seq;
            if position >= first_lag {
                w0 += e * un[(((r - first_lag) * 4 + stream) * d + j) as usize];
            }
            if position >= second_lag {
                w1 += e * un[(((r - second_lag) * 4 + stream) * d + j) as usize];
            }
            if position >= dilation {
                w2 += e * un[(((r - dilation) * 4 + stream) * d + j) as usize];
            }
            w3 += e * un[i];
            r += 16;
        }
        unsafe {
            atomic_add_f32(dgh.as_mut_ptr().add(gamma_index), ah);
            atomic_add_f32(dgk.as_mut_ptr().add(gamma_index), ak);
            atomic_add_f32(dgu.as_mut_ptr().add(gamma_index), au);
            atomic_add_f32(dw.as_mut_ptr().add(((stream * d + j) * 4) as usize), w0);
            atomic_add_f32(dw.as_mut_ptr().add(((stream * d + j) * 4 + 1) as usize), w1);
            atomic_add_f32(dw.as_mut_ptr().add(((stream * d + j) * 4 + 2) as usize), w2);
            atomic_add_f32(dw.as_mut_ptr().add(((stream * d + j) * 4 + 3) as usize), w3);
        }
    }

    #[kernel]
    pub fn engram4_stream_copy_kernel(
        input: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        d: u32,
        stream_index: u32,
        extract: bool,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i >= n * d {
            return;
        }
        let packed = ((i / d * 4 + stream_index) * d + i % d) as usize;
        unsafe {
            if extract {
                *out.get_unchecked_mut(i as usize) = input[packed];
            } else {
                *out.get_unchecked_mut(packed) = input[i as usize];
            }
        }
    }
    #[kernel]
    pub fn engram4_memory_scatter_kernel(
        ids: &[u32],
        gradient: &[f32],
        mut table: DisjointSlice<f32>,
        n: u32,
        heads: u32,
        hd: u32,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i >= n * heads * hd {
            return;
        }
        let row = ids[(i / hd) as usize];
        unsafe {
            atomic_add_f32(
                table.as_mut_ptr().add((row * hd + i % hd) as usize),
                gradient[i as usize],
            );
        }
    }
}
