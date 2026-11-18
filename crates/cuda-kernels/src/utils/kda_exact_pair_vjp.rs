//! Complete control VJP of the actual diagonal/direct and between-tile/FP16
//! factors. FP32 cotangents avoid an extra unscaled half cast. Dense projection
//! arithmetic is unchanged; this is an explicitly loaded attention derivative.
use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::float_ptx::{exp_f32, fma_f32};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use std::sync::Arc;
const AREA: usize = 4096;
type ControlTile = SharedArray<f32, { 3 * AREA }>;
pub struct KdaExactPairVjpModule {
    kernels: kernels::LoadedModule,
}
pub struct PairVjpArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub q: &'a DeviceBuffer<f32>,
    pub k: &'a DeviceBuffer<f32>,
    pub g: &'a DeviceBuffer<f32>,
    pub beta: &'a DeviceBuffer<f32>,
    pub lower_grad: &'a DeviceBuffer<f32>,
    pub readout_grad: &'a DeviceBuffer<f32>,
    pub dq: &'out mut DeviceBuffer<f32>,
    pub dk: &'out mut DeviceBuffer<f32>,
    pub dg: &'out mut DeviceBuffer<f32>,
    pub dbeta: &'out mut DeviceBuffer<f32>,
    pub batch_heads: u32,
    pub seq_len: u32,
}
impl KdaExactPairVjpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        if !crate::module_exports::contains(&module, b"kda_exact_pair_vjp_kernel")? {
            return Err(DriverError(1));
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn vjp(&self, a: PairVjpArgs<'_, '_>) -> Result<(), DriverError> {
        let chunks = a
            .seq_len
            .checked_add(63)
            .map(|x| x / 64)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))?;
        let batches = a
            .batch_heads
            .checked_mul(chunks)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))?;
        let n = batches.checked_mul(4096).ok_or(DriverError(1))? as usize;
        if [
            a.q.len(),
            a.k.len(),
            a.g.len(),
            a.lower_grad.len(),
            a.readout_grad.len(),
            a.dq.len(),
            a.dk.len(),
            a.dg.len(),
        ]
        .iter()
        .any(|&len| len < n)
            || a.beta.len() < n / 64
            || a.dbeta.len() < n / 64
        {
            return Err(DriverError(1));
        }
        self.kernels.kda_exact_pair_vjp_kernel(
            a.stream,
            LaunchConfig {
                grid_dim: (batches, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            a.q,
            a.k,
            a.g,
            a.beta,
            a.lower_grad,
            a.readout_grad,
            a.dq,
            a.dk,
            a.dg,
            a.dbeta,
            chunks,
            a.seq_len,
        )
    }
}

// XOR each feature's64-token row. Both per-feature updates and the final
// per-token beta reduction spread contiguous warp accesses across shared banks.
#[inline(always)]
fn index(h: u32, t: u32) -> usize {
    (h * 64 + (t ^ h)) as usize
}
#[inline(always)]
fn add(tile: &mut ControlTile, at: usize, value: f32) {
    tile[at] += value;
}
#[inline(always)]
fn add_interval(
    tile: &mut ControlTile,
    h: u32,
    first: u32,
    last: u32,
    value: f32,
    support: &mut u64,
) {
    if first > last || value == 0.0 {
        return;
    }
    add(tile, AREA + index(h, first), value);
    if last + 1 < 64 {
        add(tile, AREA + index(h, last + 1), -value);
    }
    let hi = if last == 63 {
        u64::MAX
    } else {
        (1u64 << (last + 1)) - 1
    };
    *support |= hi & (u64::MAX << first);
}
#[inline(always)]
fn rounded(value: f32) -> f32 {
    cvt_f32_f16(cvt_rn_f16_f32(value))
}

#[inline(always)]
fn vjp_body(
    q: &[f32],
    k: &[f32],
    g: &[f32],
    beta: &[f32],
    dl: &[f32],
    dc: &[f32],
    mut dq: DisjointSlice<f32>,
    mut dk: DisjointSlice<f32>,
    mut dg: DisjointSlice<f32>,
    mut dbeta: DisjointSlice<f32>,
    chunks: u32,
    seq_len: u32,
    tile: &mut ControlTile,
) {
    let batch = thread::blockIdx_x();
    let h = thread::threadIdx_x();
    let base = batch * 4096;
    let valid = (seq_len - (batch % chunks) * 64).min(64);
    let mut t = 0;
    while t < 64 {
        let at = index(h, t);
        tile[at] = 0.0;
        tile[AREA + at] = 0.0;
        tile[2 * AREA + at] = 0.0;
        unsafe {
            *dq.get_unchecked_mut((base + t * 64 + h) as usize) = 0.0;
        }
        t += 1;
    }
    let mut support = 0u64;
    t = 0;
    while t < valid {
        let ti = (base + t * 64 + h) as usize;
        let qt = q[ti];
        let kt = k[ti];
        let bt = beta[(batch * 64 + t) as usize];
        let anchor = t / 16 * 16;
        let mut qlog = 0.0;
        let mut s = anchor + 1;
        while s <= t {
            qlog += g[(base + s * 64 + h) as usize];
            s += 1;
        }
        let eq = exp_f32(qlog);
        let pre_q = qt * eq;
        let pre_l = (bt * kt) * eq;
        let aq = rounded(pre_q);
        let al = rounded(pre_l);
        let mut dq_t = 0.0;
        let mut db_t = 0.0;
        let mut diagonal_log = 0.0;
        let mut key_end = t + 1;
        while key_end > 0 {
            key_end -= 1;
            let j = key_end;
            let d_c = dc[(base + t * 64 + j) as usize];
            // The diagonal of L is not a parameter and must not be evaluated.
            let d_l = if j < t {
                dl[(base + t * 64 + j) as usize]
            } else {
                0.0
            };
            if d_c != 0.0 || d_l != 0.0 {
                let kj = k[(base + j * 64 + h) as usize];
                if j >= anchor {
                    let e = exp_f32(diagonal_log);
                    let key_e = kj * e;
                    dq_t = fma_f32(d_c, key_e, dq_t);
                    add(tile, index(h, t), (d_l * bt) * key_e);
                    add(tile, index(h, j), fma_f32(d_c, qt, (d_l * bt) * kt) * e);
                    db_t = fma_f32(d_l * kt, key_e, db_t);
                    let log_grad = fma_f32(d_c * qt, kj, (d_l * bt * kt) * kj) * e;
                    add_interval(tile, h, j + 1, t, log_grad, &mut support);
                } else {
                    // Reproduce the forward ascending interval exactly. A
                    // reverse sum can change the rounded FP16 operand.
                    let mut key_log = 0.0;
                    let mut s = j + 1;
                    while s <= anchor {
                        key_log += g[(base + s * 64 + h) as usize];
                        s += 1;
                    }
                    let ek = exp_f32(key_log);
                    let pre_k = kj * ek;
                    let bk = rounded(pre_k);
                    let daq = d_c * bk;
                    let dal = d_l * bk;
                    let dbk = fma_f32(d_c, aq, d_l * al);
                    dq_t = fma_f32(eq, daq, dq_t);
                    add(tile, index(h, t), (bt * eq) * dal);
                    add(tile, index(h, j), ek * dbk);
                    db_t = fma_f32(kt * eq, dal, db_t);
                    let query_log_grad = fma_f32(pre_q, daq, pre_l * dal);
                    let key_log_grad = pre_k * dbk;
                    add_interval(tile, h, anchor + 1, t, query_log_grad, &mut support);
                    add_interval(tile, h, j + 1, anchor, key_log_grad, &mut support);
                }
            }
            // The next key's direct interval includes this key's increment.
            if j > anchor {
                diagonal_log += g[(base + j * 64 + h) as usize];
            }
        }
        unsafe {
            *dq.get_unchecked_mut(ti) = dq_t;
        }
        tile[2 * AREA + index(h, t)] = db_t;
        t += 1;
    }
    let mut cumulative = 0.0;
    t = 0;
    while t < 64 {
        cumulative += tile[AREA + index(h, t)];
        let at = (base + t * 64 + h) as usize;
        unsafe {
            *dk.get_unchecked_mut(at) = tile[index(h, t)];
            *dg.get_unchecked_mut(at) = if (support & (1u64 << t)) != 0 {
                cumulative
            } else {
                0.0
            };
        }
        t += 1;
    }
    thread::sync_threads();
    // One thread now owns each token's sum of all64 feature contributions.
    let token = h;
    let mut sum = 0.0;
    let mut feature = 0;
    while feature < 64 {
        sum += tile[2 * AREA + index(feature, token)];
        feature += 1;
    }
    unsafe {
        *dbeta.get_unchecked_mut((batch * 64 + token) as usize) = sum;
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_exact_pair_vjp_kernel(
        q: &[f32],
        k: &[f32],
        g: &[f32],
        beta: &[f32],
        dl: &[f32],
        dc: &[f32],
        dq: DisjointSlice<f32>,
        dk: DisjointSlice<f32>,
        dg: DisjointSlice<f32>,
        dbeta: DisjointSlice<f32>,
        chunks: u32,
        seq_len: u32,
    ) {
        static mut RAW: ControlTile = SharedArray::UNINIT;
        unsafe {
            vjp_body(
                q, k, g, beta, dl, dc, dq, dk, dg, dbeta, chunks, seq_len, &mut RAW,
            );
        }
    }
}
