//! Complete VJP of bounded Qg/B/Kg/carry factors. Accumulate into pair
//! cotangents with the exact inclusive-prefix/exclusive-suffix support.
use crate::float_ptx::{exp_f32, fma_f32};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use std::sync::Arc;
const AREA: usize = 4096;
type ControlTile = SharedArray<f32, { 3 * AREA }>;
pub struct KdaExactBoundaryVjpModule {
    kernels: kernels::LoadedModule,
}
pub struct BoundaryVjpArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub q: &'a DeviceBuffer<f32>,
    pub k: &'a DeviceBuffer<f32>,
    pub g: &'a DeviceBuffer<f32>,
    pub beta: &'a DeviceBuffer<f32>,
    pub dqg: &'a DeviceBuffer<f32>,
    pub dkg: &'a DeviceBuffer<f32>,
    pub dincoming: &'a DeviceBuffer<f32>,
    pub dexit: &'a DeviceBuffer<f32>,
    pub dq: &'out mut DeviceBuffer<f32>,
    pub dk: &'out mut DeviceBuffer<f32>,
    pub dg: &'out mut DeviceBuffer<f32>,
    pub dbeta: &'out mut DeviceBuffer<f32>,
    pub batch_heads: u32,
    pub seq_len: u32,
}
impl KdaExactBoundaryVjpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        if !crate::module_exports::contains(&module, b"kda_exact_boundary_vjp_kernel")? {
            return Err(DriverError(1));
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn vjp(&self, a: BoundaryVjpArgs<'_, '_>) -> Result<(), DriverError> {
        let chunks = a
            .seq_len
            .checked_add(63)
            .map(|n| n / 64)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))?;
        let batch = a
            .batch_heads
            .checked_mul(chunks)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))?;
        let n = batch.checked_mul(4096).ok_or(DriverError(1))? as usize;
        if [
            a.q.len(),
            a.k.len(),
            a.g.len(),
            a.dqg.len(),
            a.dkg.len(),
            a.dincoming.len(),
            a.dq.len(),
            a.dk.len(),
            a.dg.len(),
        ]
        .iter()
        .any(|&len| len < n)
            || a.beta.len() < n / 64
            || a.dexit.len() < n / 64
            || a.dbeta.len() < n / 64
        {
            return Err(DriverError(1));
        }
        self.kernels.kda_exact_boundary_vjp_kernel(
            a.stream,
            LaunchConfig {
                grid_dim: (batch, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            a.q,
            a.k,
            a.g,
            a.beta,
            a.dqg,
            a.dkg,
            a.dincoming,
            a.dexit,
            a.dq,
            a.dk,
            a.dg,
            a.dbeta,
            chunks,
            a.seq_len,
        )
    }
}
#[inline(always)]
fn index(h: u32, t: u32) -> usize {
    (h * 64 + (t ^ h)) as usize
}
#[inline(always)]
fn body(
    q: &[f32],
    k: &[f32],
    g: &[f32],
    beta: &[f32],
    dqg: &[f32],
    dkg: &[f32],
    dincoming: &[f32],
    dexit: &[f32],
    mut dq: DisjointSlice<f32>,
    mut dk: DisjointSlice<f32>,
    mut dg: DisjointSlice<f32>,
    mut dbeta: DisjointSlice<f32>,
    chunks: u32,
    seq_len: u32,
    tile: &mut ControlTile,
) {
    let chunk = thread::blockIdx_x();
    let h = thread::threadIdx_x();
    let base = chunk * 4096;
    let valid = (seq_len - (chunk % chunks) * 64).min(64);
    let mut t = 0;
    while t < 64 {
        let at = index(h, t);
        tile[at] = 0.0;
        tile[AREA + at] = 0.0;
        tile[2 * AREA + at] = 0.0;
        t += 1;
    }
    let mut prefix = 0.0;
    t = 0;
    while t < valid {
        let i = (base + t * 64 + h) as usize;
        let b = beta[(chunk * 64 + t) as usize];
        prefix += g[i];
        let e = exp_f32(prefix);
        let uq = dqg[i];
        let ub = dincoming[i];
        unsafe {
            let previous = *dq.get_unchecked_mut(i);
            *dq.get_unchecked_mut(i) = fma_f32(e, uq, previous);
            let previous = *dk.get_unchecked_mut(i);
            *dk.get_unchecked_mut(i) = fma_f32(b * e, ub, previous);
        }
        tile[index(h, t)] = fma_f32(q[i] * e, uq, ((b * k[i]) * e) * ub);
        tile[2 * AREA + index(h, t)] = (k[i] * e) * ub;
        t += 1;
    }
    let carry_grad = exp_f32(prefix) * dexit[(chunk * 64 + h) as usize];
    // Match the original exclusive-suffix producer's reverse summation order.
    let mut suffix = 0.0;
    let mut end = valid;
    while end > 0 {
        end -= 1;
        let i = (base + end * 64 + h) as usize;
        let e = exp_f32(suffix);
        let uk = dkg[i];
        unsafe {
            let previous = *dk.get_unchecked_mut(i);
            *dk.get_unchecked_mut(i) = fma_f32(e, uk, previous);
        }
        tile[AREA + index(h, end)] = (k[i] * e) * uk;
        suffix += g[i];
    }
    let mut accumulated = 0.0;
    let mut end = valid;
    while end > 0 {
        end -= 1;
        accumulated += tile[index(h, end)];
        let i = (base + end * 64 + h) as usize;
        unsafe {
            *dg.get_unchecked_mut(i) += accumulated + carry_grad;
        }
    }
    // Exclusive prefix of the suffix-factor cotangents: Kg[t] has no g[t]
    // derivative. No subtraction of two accumulated prefixes is introduced.
    accumulated = 0.0;
    t = 0;
    while t < valid {
        unsafe {
            *dg.get_unchecked_mut((base + t * 64 + h) as usize) += accumulated;
        }
        accumulated += tile[AREA + index(h, t)];
        t += 1;
    }
    thread::sync_threads();
    let token = h;
    let mut feature = 0;
    let mut sum = 0.0;
    while feature < 64 {
        sum += tile[2 * AREA + index(feature, token)];
        feature += 1;
    }
    unsafe {
        *dbeta.get_unchecked_mut((chunk * 64 + token) as usize) += sum;
    }
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_exact_boundary_vjp_kernel(
        q: &[f32],
        k: &[f32],
        g: &[f32],
        beta: &[f32],
        dqg: &[f32],
        dkg: &[f32],
        dincoming: &[f32],
        dexit: &[f32],
        dq: DisjointSlice<f32>,
        dk: DisjointSlice<f32>,
        dg: DisjointSlice<f32>,
        dbeta: DisjointSlice<f32>,
        chunks: u32,
        seq_len: u32,
    ) {
        static mut RAW: ControlTile = SharedArray::UNINIT;
        unsafe {
            body(
                q, k, g, beta, dqg, dkg, dincoming, dexit, dq, dk, dg, dbeta, chunks, seq_len,
                &mut RAW,
            );
        }
    }
}
