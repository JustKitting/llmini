//! Causal bounded-interval KDA pair factors. Diagonal16-token tiles use direct
//! negative intervals; six off-diagonal tiles use FP16 tensor-core operands
//! anchored at the first query token. No default attention dispatch is changed.
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use std::sync::Arc;

use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::float_ptx::{exp_f32, fma_f32};

pub struct KdaExactPairModule {
    kernels: kernels::LoadedModule,
}

/// Inputs are original FP32 normalized/scaled q, normalized k and individual
/// nonpositive log-gate increments, in [batch_head,ceil(S/64),64,64] order.
/// beta is [batch_head,ceil(S/64),64]. Invalid padded tokens are never read.
/// L is FP32 strict lower; C is inclusive-lower FP16, both padded64x64/chunk.
pub struct PairArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub q: &'a DeviceBuffer<f32>,
    pub k: &'a DeviceBuffer<f32>,
    pub g: &'a DeviceBuffer<f32>,
    pub beta: &'a DeviceBuffer<f32>,
    pub lower: &'out mut DeviceBuffer<f32>,
    pub readout: &'out mut DeviceBuffer<u16>,
    pub batch_heads: u32,
    pub seq_len: u32,
}

impl KdaExactPairModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        for name in [
            b"kda_exact_diagonal_pairs_kernel".as_slice(),
            b"kda_exact_between_pairs_kernel".as_slice(),
        ] {
            if !crate::module_exports::contains(&module, name)? {
                return Err(DriverError(1));
            }
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }

    pub fn pairs(&self, args: PairArgs<'_, '_>) -> Result<(), DriverError> {
        let chunks = args
            .seq_len
            .checked_add(63)
            .map(|x| x / 64)
            .filter(|&x| x > 0)
            .ok_or(DriverError(1))?;
        let batches = args
            .batch_heads
            .checked_mul(chunks)
            .filter(|&x| x > 0)
            .ok_or(DriverError(1))?;
        let count = batches.checked_mul(4096).ok_or(DriverError(1))? as usize;
        if args.q.len() < count
            || args.k.len() < count
            || args.g.len() < count
            || args.beta.len() < count / 64
            || args.lower.len() < count
            || args.readout.len() < count
        {
            return Err(DriverError(1));
        }
        self.kernels.kda_exact_diagonal_pairs_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (batches, 4, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            args.q,
            args.k,
            args.g,
            args.beta,
            args.lower,
            args.readout,
            chunks,
            args.seq_len,
        )?;
        self.kernels.kda_exact_between_pairs_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (batches, 6, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            args.q,
            args.k,
            args.g,
            args.beta,
            args.lower,
            args.readout,
            chunks,
            args.seq_len,
        )
    }
}

#[inline(always)]
fn valid_tokens(batch: u32, chunks: u32, seq_len: u32) -> u32 {
    let start = (batch % chunks) * 64;
    (seq_len - start).min(64)
}

#[inline(always)]
fn diagonal_body(
    q: &[f32],
    k: &[f32],
    g: &[f32],
    beta: &[f32],
    mut lower: DisjointSlice<f32>,
    mut readout: DisjointSlice<u16>,
    chunks: u32,
    seq_len: u32,
) {
    let batch = thread::blockIdx_x();
    let first = thread::blockIdx_y() * 16;
    let tid = thread::threadIdx_x();
    let lane = tid & 31;
    let mut t = first + tid / 32;
    let valid = valid_tokens(batch, chunks, seq_len);
    let base = batch * 4096;
    while t < first + 16 {
        // These stores also own the padded rows and all future-tile columns.
        // No masked positive exponential or padded input is evaluated.
        let mut j = if t < valid { t + 1 } else { first };
        while j < 64 {
            if lane == 0 {
                unsafe {
                    *lower.get_unchecked_mut((base + t * 64 + j) as usize) = 0.0;
                    *readout.get_unchecked_mut((base + t * 64 + j) as usize) = 0;
                }
            }
            j += 1;
        }
        if t < valid {
            let ti = (base + t * 64 + lane) as usize;
            let q0 = q[ti];
            let q1 = q[ti + 32];
            let k0 = k[ti];
            let k1 = k[ti + 32];
            let bt = beta[(batch * 64 + t) as usize];
            let mut g0 = 0.0;
            let mut g1 = 0.0;
            let mut key = t;
            loop {
                let ji = (base + key * 64 + lane) as usize;
                let e0 = exp_f32(g0);
                let e1 = exp_f32(g1);
                let kj0 = k[ji] * e0;
                let kj1 = k[ji + 32] * e1;
                let c = crate::warp_reduce::warp_sum_f32(fma_f32(q0, kj0, q1 * kj1));
                let l = crate::warp_reduce::warp_sum_f32(fma_f32(k0, kj0, k1 * kj1));
                if lane == 0 {
                    unsafe {
                        *lower.get_unchecked_mut((base + t * 64 + key) as usize) =
                            if key < t { bt * l } else { 0.0 };
                        *readout.get_unchecked_mut((base + t * 64 + key) as usize) =
                            cvt_rn_f16_f32(c);
                    }
                }
                if key == first {
                    break;
                }
                // Interval for the next earlier key includes this key's gate.
                g0 += g[ji];
                g1 += g[ji + 32];
                key -= 1;
            }
        }
        t += 4;
    }
}

#[inline(always)]
fn packed(src: *const u16, row: u32, col: u32) -> u32 {
    unsafe { *(src.add((row * 64 + col) as usize) as *const u32) }
}
#[inline(always)]
fn a_fragment(src: *const u16, lane: u32, h: u32) -> [u32; 4] {
    let r = lane / 4;
    let c = (lane % 4) * 2 + h;
    [
        packed(src, r, c),
        packed(src, r + 8, c),
        packed(src, r, c + 8),
        packed(src, r + 8, c + 8),
    ]
}
#[inline(always)]
fn b_fragment(src: *const u16, lane: u32, h: u32, col_tile: u32) -> [u32; 2] {
    let r = col_tile * 8 + lane / 4;
    let c = (lane % 4) * 2 + h;
    [packed(src, r, c), packed(src, r, c + 8)]
}

#[inline(always)]
fn between_body(
    q: &[f32],
    k: &[f32],
    g: &[f32],
    beta: &[f32],
    mut lower: DisjointSlice<f32>,
    mut readout: DisjointSlice<u16>,
    chunks: u32,
    seq_len: u32,
    raw: &mut SharedArray<u16, { 3 * 16 * 64 }>,
) {
    let batch = thread::blockIdx_x();
    let pair = thread::blockIdx_y();
    let (query_tile, key_tile) = match pair {
        0 => (1, 0),
        1 => (2, 0),
        2 => (2, 1),
        3 => (3, 0),
        4 => (3, 1),
        _ => (3, 2),
    };
    let anchor = query_tile * 16;
    let valid = valid_tokens(batch, chunks, seq_len);
    let lane = thread::threadIdx_x();
    let base = batch * 4096;
    let qa = raw.as_mut_ptr();
    let ka = unsafe { qa.add(1024) };
    let kb = unsafe { ka.add(1024) };
    // One rounding, directly from original FP32 q/k/g/beta. The key-side
    // interval ends at this query tile's causal first token, never a midpoint.
    let mut i = lane;
    while i < 1024 {
        let local = i / 64;
        let h = i % 64;
        let t = anchor + local;
        let j = key_tile * 16 + local;
        let (aq, ak) = if t < valid {
            let mut log = 0.0;
            let mut s = anchor + 1;
            while s <= t {
                log += g[(base + s * 64 + h) as usize];
                s += 1;
            }
            let decay = exp_f32(log);
            let at = (base + t * 64 + h) as usize;
            (
                cvt_rn_f16_f32(q[at] * decay),
                cvt_rn_f16_f32((beta[(batch * 64 + t) as usize] * k[at]) * decay),
            )
        } else {
            (0, 0)
        };
        // An absent query tile must not read its future anchor/padded gates.
        let bk = if anchor < valid && j < valid {
            let mut log = 0.0;
            let mut s = j + 1;
            while s <= anchor {
                log += g[(base + s * 64 + h) as usize];
                s += 1;
            }
            cvt_rn_f16_f32(k[(base + j * 64 + h) as usize] * exp_f32(log))
        } else {
            0
        };
        unsafe {
            *qa.add(i as usize) = aq;
            *ka.add(i as usize) = ak;
            *kb.add(i as usize) = bk;
        }
        i += 32;
    }
    thread::sync_threads();
    let mut c0 = [0.0f32; 4];
    let mut c1 = [0.0f32; 4];
    let mut l0 = [0.0f32; 4];
    let mut l1 = [0.0f32; 4];
    let mut h = 0;
    while h < 64 {
        let aq = a_fragment(qa, lane, h);
        let ak = a_fragment(ka, lane, h);
        let b0 = b_fragment(kb, lane, h, 0);
        let b1 = b_fragment(kb, lane, h, 1);
        crate::mma::mma_m16n8k16_f16_f16_f32(aq, b0, &mut c0);
        crate::mma::mma_m16n8k16_f16_f16_f32(aq, b1, &mut c1);
        crate::mma::mma_m16n8k16_f16_f16_f32(ak, b0, &mut l0);
        crate::mma::mma_m16n8k16_f16_f16_f32(ak, b1, &mut l1);
        h += 16;
    }
    let mut element = 0;
    while element < 4 {
        let t = anchor + lane / 4 + if element < 2 { 0 } else { 8 };
        let j = key_tile * 16 + (lane % 4) * 2 + (element as u32 & 1);
        let at = (base + t * 64 + j) as usize;
        // Absent query rows are still written; their operands were all zero.
        unsafe {
            *lower.get_unchecked_mut(at) = l0[element];
            *lower.get_unchecked_mut(at + 8) = l1[element];
            *readout.get_unchecked_mut(at) = cvt_rn_f16_f32(c0[element]);
            *readout.get_unchecked_mut(at + 8) = cvt_rn_f16_f32(c1[element]);
        }
        element += 1;
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_exact_diagonal_pairs_kernel(
        q: &[f32],
        k: &[f32],
        g: &[f32],
        beta: &[f32],
        lower: DisjointSlice<f32>,
        readout: DisjointSlice<u16>,
        chunks: u32,
        seq_len: u32,
    ) {
        diagonal_body(q, k, g, beta, lower, readout, chunks, seq_len);
    }
    #[kernel]
    pub fn kda_exact_between_pairs_kernel(
        q: &[f32],
        k: &[f32],
        g: &[f32],
        beta: &[f32],
        lower: DisjointSlice<f32>,
        readout: DisjointSlice<u16>,
        chunks: u32,
        seq_len: u32,
    ) {
        static mut RAW: SharedArray<u16, { 3 * 16 * 64 }> = SharedArray::UNINIT;
        unsafe {
            between_body(q, k, g, beta, lower, readout, chunks, seq_len, &mut RAW);
        }
    }
}
