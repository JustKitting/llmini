//! Native Engram elementwise graph. Dense projections use the shared NVFP4 TMA path.
use crate::{
    atomic::atomic_add_f32,
    block_reduce::block_sum_shared_f32,
    float_ptx::{abs_f32, exp_f32, max_f32, sqrt_f32},
    nvfp4::nvfp4_value,
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
#[inline(always)]
fn hash_ngrams(
    tokens: &[u32],
    map: &[u32],
    primes: &[u32],
    offsets: &[u32],
    multipliers: &[u64],
    ids: &mut DisjointSlice<u32>,
    n: u32,
    seq: u32,
    heads: u32,
    bos: u32,
    eos: u32,
    pad: u32,
    orders: u32,
) {
    let i = thread::blockIdx_x() * T + thread::threadIdx_x();
    if i >= n * heads * orders {
        return;
    }
    let r = i / (heads * orders);
    let h = i % (heads * orders);
    let order = 2 + h / heads;
    let mut mix = 0u64;
    let mut valid = true;
    let mut lag = 0;
    while lag < order {
        let mut token = pad;
        if valid && lag <= r % seq {
            let raw = tokens[(r - lag) as usize] as u32;
            if lag > 0 && (raw == bos || raw == eos) {
                valid = false;
            } else {
                token = map[raw as usize];
                if raw == bos || raw == eos {
                    valid = false;
                }
            }
        }
        mix ^= (token as u64).wrapping_mul(multipliers[lag as usize]);
        lag += 1;
    }
    unsafe {
        *ids.get_unchecked_mut(i as usize) =
            offsets[h as usize] + (mix % primes[h as usize] as u64) as u32;
    }
}

#[cuda_module]
pub mod four_gram_hash {
    use super::*;
    #[kernel]
    pub fn engram_four_gram_hash_kernel(
        tokens: &[u32],
        map: &[u32],
        primes: &[u32],
        offsets: &[u32],
        multipliers: &[u64],
        mut ids: DisjointSlice<u32>,
        n: u32,
        seq: u32,
        heads: u32,
        bos: u32,
        eos: u32,
        pad: u32,
    ) {
        hash_ngrams(
            tokens,
            map,
            primes,
            offsets,
            multipliers,
            &mut ids,
            n,
            seq,
            heads,
            bos,
            eos,
            pad,
            3,
        );
    }
}

#[cuda_module]
pub mod kernels {
    static mut REDUCE: SharedArray<f32, 8> = SharedArray::UNINIT;
    use super::*;
    #[kernel]
    pub fn engram_hash_kernel(
        tokens: &[u32],
        map: &[u32],
        primes: &[u32],
        offsets: &[u32],
        multipliers: &[u64],
        mut ids: DisjointSlice<u32>,
        n: u32,
        seq: u32,
        heads: u32,
        bos: u32,
        eos: u32,
        pad: u32,
    ) {
        hash_ngrams(
            tokens,
            map,
            primes,
            offsets,
            multipliers,
            &mut ids,
            n,
            seq,
            heads,
            bos,
            eos,
            pad,
            2,
        );
    }
    #[kernel]
    pub fn engram_gather_kernel(
        ids: &[u32],
        bytes: &[u8],
        scales: &[u8],
        global: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        heads: u32,
        dim: u32,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i < n * heads * dim {
            let id = ids[(i / dim) as usize];
            unsafe {
                *out.get_unchecked_mut(i as usize) =
                    nvfp4_value(bytes, scales, global[0], (id * dim + i % dim) as usize);
            }
        }
    }
    #[kernel]
    pub fn engram_scatter_kernel(
        ids: &[u32],
        a: &[f32],
        b: &[f32],
        mut grad: DisjointSlice<f32>,
        n: u32,
        heads: u32,
        dim: u32,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i < n * heads * dim {
            let index = ids[(i / dim) as usize] * dim + i % dim;
            unsafe {
                atomic_add_f32(
                    grad.as_mut_ptr().add(index as usize),
                    a[i as usize] + b[i as usize],
                );
            }
        }
    }
    #[kernel]
    pub fn engram_gate_kernel(
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
        if r >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let base = (r * d) as usize;
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
            dot += x * gh[j as usize] * y * gk[j as usize];
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
                *u.get_unchecked_mut(base + j as usize) = g * v[base + j as usize];
            }
            j += T;
        }
    }
    #[kernel]
    pub fn engram_norm_kernel(
        u: &[f32],
        gamma: &[f32],
        mut out: DisjointSlice<f32>,
        mut inv: DisjointSlice<f32>,
        n: u32,
        d: u32,
    ) {
        let r = thread::blockIdx_x();
        if r >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let b = (r * d) as usize;
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
                *out.get_unchecked_mut(b + j as usize) = u[b + j as usize] * v * gamma[j as usize];
            }
            j += T;
        }
    }
    #[kernel]
    pub fn engram_conv_add_kernel(
        u: &[f32],
        un: &[f32],
        w: &[f32],
        mut pre: DisjointSlice<f32>,
        mut residual: DisjointSlice<f32>,
        n: u32,
        d: u32,
        seq: u32,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i >= n * d {
            return;
        }
        let r = i / d;
        let j = i % d;
        let mut z = 0.;
        let mut c = 0;
        while c < 4 {
            let lag = 3 * (3 - c);
            if r % seq >= lag {
                z += w[(j * 4 + c) as usize] * un[((r - lag) * d + j) as usize];
            }
            c += 1;
        }
        unsafe {
            *pre.get_unchecked_mut(i as usize) = z;
            *residual.get_unchecked_mut(i as usize) += u[i as usize] + z * sig(z);
        }
    }
    #[kernel]
    pub fn engram_conv_delta_kernel(dout: &[f32], mut pre: DisjointSlice<f32>, len: u32) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i < len {
            unsafe {
                let z = *pre.get_unchecked_mut(i as usize);
                let a = sig(z);
                *pre.get_unchecked_mut(i as usize) = dout[i as usize] * a * (1. + z * (1. - a));
            }
        }
    }
    #[kernel]
    pub fn engram_conv_input_backward_kernel(
        delta: &[f32],
        w: &[f32],
        mut dun: DisjointSlice<f32>,
        n: u32,
        d: u32,
        seq: u32,
    ) {
        let i = thread::blockIdx_x() * T + thread::threadIdx_x();
        if i >= n * d {
            return;
        }
        let r = i / d;
        let j = i % d;
        let mut a = 0.;
        let mut c = 0;
        while c < 4 {
            let lag = 3 * (3 - c);
            if r % seq + lag < seq && r + lag < n {
                a += w[(j * 4 + c) as usize] * delta[((r + lag) * d + j) as usize];
            }
            c += 1;
        }
        unsafe {
            *dun.get_unchecked_mut(i as usize) = a;
        }
    }
    #[kernel]
    pub fn engram_norm_backward_kernel(
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
        if r >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let b = (r * d) as usize;
        let s = unsafe { &mut REDUCE };
        let mut a = 0.;
        let mut j = t;
        while j < d {
            a += u[b + j as usize] * dun[b + j as usize] * gamma[j as usize];
            j += T;
        }
        let iv = inv[r as usize];
        let m = sum(s, a) * iv * iv / d as f32;
        j = t;
        while j < d {
            let i = b + j as usize;
            unsafe {
                *du.get_unchecked_mut(i) = direct[i] + iv * (dun[i] * gamma[j as usize] - u[i] * m);
            }
            j += T;
        }
    }
    #[kernel]
    pub fn engram_gate_backward_kernel(
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
        let r = thread::blockIdx_x();
        if r >= n {
            return;
        }
        let t = thread::threadIdx_x();
        let b = (r * d) as usize;
        let s = unsafe { &mut REDUCE };
        let mut a = 0.;
        let mut j = t;
        while j < d {
            a += du[b + j as usize] * v[b + j as usize];
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
            let dhn = g * k[i] * ik * gk[j as usize] * gh[j as usize];
            let dkn = g * h[i] * ih * gh[j as usize] * gk[j as usize];
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
                *direct.get_unchecked_mut(i) +=
                    ih * (g * k[i] * ik * gk[j as usize] * gh[j as usize] - h[i] * mh);
                *dk.get_unchecked_mut(i) =
                    ik * (g * h[i] * ih * gh[j as usize] * gk[j as usize] - k[i] * mk);
                *dv.get_unchecked_mut(i) = du[i] * gate;
            }
            j += T;
        }
    }
    #[kernel]
    pub fn engram_parameter_backward_kernel(
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
    ) {
        let index = thread::blockIdx_x() * T + thread::threadIdx_x();
        if index >= d * 16 {
            return;
        }
        let j = index % d;
        let part = index / d;
        let mut ah = 0.;
        let mut ak = 0.;
        let mut au = 0.;
        let mut w0 = 0.;
        let mut w1 = 0.;
        let mut w2 = 0.;
        let mut w3 = 0.;
        let mut r = part;
        while r < n {
            let i = (r * d + j) as usize;
            let h0 = h[i] * stats[(r * 4) as usize];
            let k0 = k[i] * stats[(r * 4 + 1) as usize];
            let g = d_dot[r as usize];
            ah += g * k0 * gk[j as usize] * h0;
            ak += g * h0 * gh[j as usize] * k0;
            au += dun[i] * u[i] * inv_u[r as usize];
            let e = delta[i];
            if r % seq >= 9 {
                w0 += e * un[((r - 9) * d + j) as usize];
            }
            if r % seq >= 6 {
                w1 += e * un[((r - 6) * d + j) as usize];
            }
            if r % seq >= 3 {
                w2 += e * un[((r - 3) * d + j) as usize];
            }
            w3 += e * un[i];
            r += 16;
        }
        unsafe {
            atomic_add_f32(dgh.as_mut_ptr().add(j as usize), ah);
            atomic_add_f32(dgk.as_mut_ptr().add(j as usize), ak);
            atomic_add_f32(dgu.as_mut_ptr().add(j as usize), au);
            atomic_add_f32(dw.as_mut_ptr().add((j * 4) as usize), w0);
            atomic_add_f32(dw.as_mut_ptr().add((j * 4 + 1) as usize), w1);
            atomic_add_f32(dw.as_mut_ptr().add((j * 4 + 2) as usize), w2);
            atomic_add_f32(dw.as_mut_ptr().add((j * 4 + 3) as usize), w3);
        }
    }
}
