//! Original native KDA activation/gates and bounded causal boundary factors.
//! Called explicitly by complete exact-factor preparation; default dispatch is
//! unchanged. The affine input already includes the ordinary raw-V sharing.
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::float_ptx::exp_f32;
use crate::kda_common::{
    beta_index, beta_offset, g_offset, k_offset, q_offset, qkv_index, safe_denom, sigmoid,
    softplus, v_offset, value_activation,
};
use crate::kda_elementwise::{KdaQkNormAcc, read_qk_act};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use std::sync::Arc;

pub struct KdaExactPrepareModule {
    kernels: kernels::LoadedModule,
}
pub struct PrepareArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    /// Original FP32 projection result, after any raw-value residual mix.
    pub qkv: &'a DeviceBuffer<f32>,
    pub qkv_f16: Option<&'out mut DeviceBuffer<u16>>,
    pub q: &'out mut DeviceBuffer<f32>,
    pub k: &'out mut DeviceBuffer<f32>,
    pub g: &'out mut DeviceBuffer<f32>,
    pub beta: &'out mut DeviceBuffer<f32>,
    pub vbeta: &'out mut DeviceBuffer<u16>,
    pub batch_size: u32,
    pub seq_len: u32,
    pub head_count: u32,
    pub qkv_dim: u32,
}
pub struct FactorArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub q: &'a DeviceBuffer<f32>,
    pub k: &'a DeviceBuffer<f32>,
    pub g: &'a DeviceBuffer<f32>,
    pub beta: &'a DeviceBuffer<f32>,
    pub qg: &'out mut DeviceBuffer<u16>,
    pub kg: &'out mut DeviceBuffer<u16>,
    /// Existing per-chunk4096-float U arena. Only its first4096 half words
    /// contain B initially; the original joint W/U TC consumer overwrites U.
    pub incoming_rhs_u: &'out mut DeviceBuffer<f32>,
    /// True FP32 exp(sum g), never the legacy exp(-3) saturation floor.
    pub exit_decay: &'out mut DeviceBuffer<f32>,
    pub batch_heads: u32,
    pub seq_len: u32,
}
fn layout(batch_heads: u32, seq_len: u32) -> Result<(u32, u32), DriverError> {
    let chunks = seq_len
        .checked_add(63)
        .map(|x| x / 64)
        .filter(|&x| x > 0)
        .ok_or(DriverError(1))?;
    let count = batch_heads
        .checked_mul(chunks)
        .and_then(|x| x.checked_mul(4096))
        .filter(|&x| x > 0)
        .ok_or(DriverError(1))?;
    Ok((chunks, count))
}
impl KdaExactPrepareModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        for name in [
            b"kda_exact_prepare_kernel".as_slice(),
            b"kda_exact_prepare_save_kernel".as_slice(),
            b"kda_exact_boundary_factors_kernel".as_slice(),
        ] {
            if !crate::module_exports::contains(&module, name)? {
                return Err(DriverError(1));
            }
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn prepare(&self, args: PrepareArgs<'_, '_>) -> Result<(), DriverError> {
        let bh = args
            .batch_size
            .checked_mul(args.head_count)
            .ok_or(DriverError(1))?;
        let (chunks, count) = layout(bh, args.seq_len)?;
        let dim = args.head_count.checked_mul(64).ok_or(DriverError(1))?;
        let active = dim
            .checked_mul(4)
            .and_then(|x| x.checked_add(args.head_count))
            .ok_or(DriverError(1))?;
        let max_stride = args
            .head_count
            .checked_mul(32)
            .and_then(|x| x.checked_add(active))
            .ok_or(DriverError(1))?;
        let rows = args
            .batch_size
            .checked_mul(args.seq_len)
            .ok_or(DriverError(1))?;
        let raw_count = rows.checked_mul(args.qkv_dim).ok_or(DriverError(1))? as usize;
        let n = count as usize;
        if args.qkv_dim < active
            || args.qkv_dim > max_stride
            || args.qkv.len() < raw_count
            || args.q.len() < n
            || args.k.len() < n
            || args.g.len() < n
            || args.beta.len() < n / 64
            || args.vbeta.len() < n
            || args.qkv_f16.as_ref().is_some_and(|t| t.len() < raw_count)
        {
            return Err(DriverError(1));
        }
        let params = CausalAttentionParams::new(
            rows,
            args.seq_len,
            args.batch_size,
            dim,
            args.qkv_dim,
            args.head_count,
            64,
        );
        let config = LaunchConfig {
            grid_dim: (bh, chunks, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        if let Some(tape) = args.qkv_f16 {
            self.kernels.kda_exact_prepare_save_kernel(
                args.stream,
                config,
                args.qkv,
                tape,
                args.q,
                args.k,
                args.g,
                args.beta,
                args.vbeta,
                params,
            )
        } else {
            self.kernels.kda_exact_prepare_kernel(
                args.stream,
                config,
                args.qkv,
                args.q,
                args.k,
                args.g,
                args.beta,
                args.vbeta,
                params,
            )
        }
    }
    pub fn factors(&self, args: FactorArgs<'_, '_>) -> Result<(), DriverError> {
        let (chunks, count) = layout(args.batch_heads, args.seq_len)?;
        let n = count as usize;
        if args.q.len() < n
            || args.k.len() < n
            || args.g.len() < n
            || args.beta.len() < n / 64
            || args.qg.len() < n
            || args.kg.len() < n
            || args.incoming_rhs_u.len() < n
            || args.exit_decay.len() < n / 64
        {
            return Err(DriverError(1));
        }
        self.kernels.kda_exact_boundary_factors_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (count / 4096, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            args.q,
            args.k,
            args.g,
            args.beta,
            args.qg,
            args.kg,
            args.incoming_rhs_u,
            args.exit_decay,
            chunks,
            args.seq_len,
        )
    }
}

#[inline(always)]
fn prepare_body<const SAVE: bool>(
    qkv: &[f32],
    save: *mut u16,
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut g: DisjointSlice<f32>,
    mut beta: DisjointSlice<f32>,
    mut vbeta: DisjointSlice<u16>,
    p: CausalAttentionParams,
) {
    let bh = thread::blockIdx_x();
    let chunk = thread::blockIdx_y();
    let chunks = p.seq_len.div_ceil(64);
    let batch = bh / p.head_count;
    let head = bh % p.head_count;
    let lane = thread::threadIdx_x() & 31;
    let mut t = thread::threadIdx_x() / 32;
    let base = (bh * chunks + chunk) * 4096;
    while t < 64 {
        let token = chunk * 64 + t;
        let row = batch * p.seq_len + token;
        let at = (base + t * 64 + lane) as usize;
        if token < p.seq_len {
            // Reuse exactly the original FP32 SiLU and warp-norm producer.
            let qk0 = read_qk_act(qkv, row, head, lane, &p);
            let qk1 = read_qk_act(qkv, row, head, lane + 32, &p);
            let mut norm = KdaQkNormAcc::zero();
            norm.add(qk0);
            norm.add(qk1);
            let (qn, kn) = norm.norms();
            let qi = p.scale / safe_denom(qn);
            let ki = 1.0 / safe_denom(kn);
            let bi = beta_index(row, head, &p);
            let raw_b = if lane == 0 { qkv[bi] } else { 0.0 };
            let b = cuda_device::warp::shuffle_f32(if lane == 0 { sigmoid(raw_b) } else { 0.0 }, 0);
            if lane == 0 {
                unsafe {
                    *beta.get_unchecked_mut(((bh * chunks + chunk) * 64 + t) as usize) = b;
                    if SAVE {
                        *save.add(bi) = cvt_rn_f16_f32(raw_b);
                    }
                }
            }
            unsafe {
                *q.get_unchecked_mut(at) = qk0.q_act() * qi;
                *q.get_unchecked_mut(at + 32) = qk1.q_act() * qi;
                *k.get_unchecked_mut(at) = qk0.k_act() * ki;
                *k.get_unchecked_mut(at + 32) = qk1.k_act() * ki;
            }
            let mut part = 0;
            while part < 2 {
                let dim = lane + part * 32;
                let oi = at + part as usize * 32;
                let vi = qkv_index(row, head, dim, v_offset(&p), &p);
                let gi = qkv_index(row, head, dim, g_offset(&p), &p);
                let rv = qkv[vi];
                let rg = qkv[gi];
                unsafe {
                    *g.get_unchecked_mut(oi) = -p.decay_scale * softplus(rg);
                    *vbeta.get_unchecked_mut(oi) = cvt_rn_f16_f32(value_activation(rv) * b);
                    if SAVE {
                        *save.add(qkv_index(row, head, dim, q_offset(&p), &p)) =
                            cvt_rn_f16_f32(if part == 0 { qk0.raw_q } else { qk1.raw_q });
                        *save.add(qkv_index(row, head, dim, k_offset(&p), &p)) =
                            cvt_rn_f16_f32(if part == 0 { qk0.raw_k } else { qk1.raw_k });
                        *save.add(vi) = cvt_rn_f16_f32(rv);
                        *save.add(gi) = cvt_rn_f16_f32(rg);
                    }
                }
                part += 1;
            }
            let tail = beta_offset(&p) + p.head_count + head * 32 + lane;
            if SAVE && tail < p.qkv_dim {
                let index = (row * p.qkv_dim + tail) as usize;
                unsafe {
                    *save.add(index) = cvt_rn_f16_f32(qkv[index]);
                }
            }
        } else {
            unsafe {
                *q.get_unchecked_mut(at) = 0.0;
                *q.get_unchecked_mut(at + 32) = 0.0;
                *k.get_unchecked_mut(at) = 0.0;
                *k.get_unchecked_mut(at + 32) = 0.0;
                *g.get_unchecked_mut(at) = 0.0;
                *g.get_unchecked_mut(at + 32) = 0.0;
                *vbeta.get_unchecked_mut(at) = 0;
                *vbeta.get_unchecked_mut(at + 32) = 0;
                if lane == 0 {
                    *beta.get_unchecked_mut(((bh * chunks + chunk) * 64 + t) as usize) = 0.0;
                }
            }
        }
        t += 4;
    }
}

#[inline(always)]
fn factors_body(
    q: &[f32],
    k: &[f32],
    g: &[f32],
    beta: &[f32],
    mut qg: DisjointSlice<u16>,
    mut kg: DisjointSlice<u16>,
    mut incoming_rhs_u: DisjointSlice<f32>,
    mut exit_decay: DisjointSlice<f32>,
    chunks: u32,
    seq_len: u32,
) {
    let chunk = thread::blockIdx_x();
    let h = thread::threadIdx_x();
    let valid = (seq_len - (chunk % chunks) * 64).min(64);
    let base = chunk * 4096;
    let packed = incoming_rhs_u.as_mut_ptr() as *mut u16;
    let mut prefix = 0.0;
    let mut t = 0;
    while t < 64 {
        let i = (base + t * 64 + h) as usize;
        let (qbits, bbits) = if t < valid {
            prefix += g[i];
            let decay = exp_f32(prefix);
            (
                cvt_rn_f16_f32(q[i] * decay),
                cvt_rn_f16_f32((beta[(chunk * 64 + t) as usize] * k[i]) * decay),
            )
        } else {
            (0, 0)
        };
        unsafe {
            *qg.get_unchecked_mut(i) = qbits;
            *packed.add(crate::kda_common::kpos_half_arena_index(i)) = bbits;
        }
        t += 1;
    }
    unsafe {
        *exit_decay.get_unchecked_mut((chunk * 64 + h) as usize) = exp_f32(prefix);
    }
    // Exclusive suffix uses individual increments. Never subtract two large
    // prefix sums: a severe early decay must not erase a fresh later key.
    let mut suffix = 0.0;
    let mut end = 64;
    while end > 0 {
        end -= 1;
        let i = (base + end * 64 + h) as usize;
        let bits = if end < valid {
            let value = cvt_rn_f16_f32(k[i] * exp_f32(suffix));
            suffix += g[i];
            value
        } else {
            0
        };
        unsafe {
            *kg.get_unchecked_mut(i) = bits;
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_exact_prepare_kernel(
        qkv: &[f32],
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        g: DisjointSlice<f32>,
        beta: DisjointSlice<f32>,
        vbeta: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        prepare_body::<false>(qkv, 0 as *mut u16, q, k, g, beta, vbeta, params);
    }
    #[kernel]
    pub fn kda_exact_prepare_save_kernel(
        qkv: &[f32],
        mut tape: DisjointSlice<u16>,
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        g: DisjointSlice<f32>,
        beta: DisjointSlice<f32>,
        vbeta: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        prepare_body::<true>(qkv, tape.as_mut_ptr(), q, k, g, beta, vbeta, params);
    }
    #[kernel]
    pub fn kda_exact_boundary_factors_kernel(
        q: &[f32],
        k: &[f32],
        g: &[f32],
        beta: &[f32],
        qg: DisjointSlice<u16>,
        kg: DisjointSlice<u16>,
        incoming_rhs_u: DisjointSlice<f32>,
        exit_decay: DisjointSlice<f32>,
        chunks: u32,
        seq_len: u32,
    ) {
        factors_body(
            q,
            k,
            g,
            beta,
            qg,
            kg,
            incoming_rhs_u,
            exit_decay,
            chunks,
            seq_len,
        );
    }
}
