//! Read-only statistics of the original native FP32 decay plane.
use crate::{
    f16_tc_matmul::convert::cvt_rn_f16_f32,
    float_ptx::{abs_f32, max_f32},
    kda_common::{KDA_DECAY_EXP_LIMIT, softplus},
};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use std::sync::Arc;
pub struct DecayAuditModule {
    kernels: kernels::LoadedModule,
}
pub struct DecayAuditData {
    pub counts: Vec<u32>,
    pub extrema: Vec<f32>,
    pub cells: usize,
    pub decay_scale: f32,
    pub exponent_limit: f32,
}
impl DecayAuditModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn inspect(
        &self,
        s: &CudaStream,
        raw: &DeviceBuffer<f32>,
        saved: &DeviceBuffer<u16>,
        rows: u32,
        seq: u32,
        d: u32,
        stride: u32,
    ) -> Result<DecayAuditData, DriverError> {
        if rows == 0
            || seq == 0
            || seq % 64 != 0
            || rows % seq != 0
            || d == 0
            || d % 64 != 0
            || rows.checked_mul(stride).is_none()
            || stride < 4 * d + d / 64
            || raw.len() < (rows as usize) * (stride as usize)
            || saved.len() < (rows as usize) * (stride as usize)
        {
            return Err(DriverError(1));
        }
        let cells = (rows as usize / 64) * d as usize;
        let decay_scale = crate::attention::CausalAttentionParams::new(
            rows,
            seq,
            rows / seq,
            d,
            stride,
            d / 64,
            64,
        )
        .decay_scale;
        let mut counts = DeviceBuffer::<u32>::zeroed(s, cells * 8)?;
        let mut extrema = DeviceBuffer::<f32>::zeroed(s, cells * 4)?;
        self.kernels.kda_decay_audit_kernel(
            s,
            LaunchConfig {
                grid_dim: ((cells / 64) as u32, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            raw,
            saved,
            &mut counts,
            &mut extrema,
            rows,
            seq,
            d,
            stride,
            decay_scale,
        )?;
        Ok(DecayAuditData {
            counts: counts.to_host_vec(s)?,
            extrema: extrema.to_host_vec(s)?,
            cells,
            decay_scale,
            exponent_limit: KDA_DECAY_EXP_LIMIT,
        })
    }
}
#[inline(always)]
fn finite(x: f32) -> bool {
    x.to_bits() & 0x7f800000 != 0x7f800000
}
#[inline(always)]
fn clamp(x: f32) -> f32 {
    if x < -KDA_DECAY_EXP_LIMIT {
        -KDA_DECAY_EXP_LIMIT
    } else if x > KDA_DECAY_EXP_LIMIT {
        KDA_DECAY_EXP_LIMIT
    } else {
        x
    }
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_decay_audit_kernel(
        raw: &[f32],
        saved: &[u16],
        mut counts: DisjointSlice<u32>,
        mut extrema: DisjointSlice<f32>,
        rows: u32,
        seq: u32,
        d: u32,
        stride: u32,
        decay_scale: f32,
    ) {
        static mut PREFIX: SharedArray<f32, 4096> = SharedArray::UNINIT;
        let j = thread::threadIdx_x();
        let group = thread::blockIdx_x();
        let chunks = seq / 64;
        let chunk = group % chunks;
        let bh = group / chunks;
        let heads = d / 64;
        let batch = bh / heads;
        let head = bh % heads;
        if j >= 64 || batch * seq >= rows {
            return;
        }
        let mut acc = 0f32;
        let mut c = [0u32; 8];
        let mut run = 0u32;
        let mut previous = 0f32;
        let mut first_excess = 0f32;
        let mut last_excess = 0f32;
        for t in 0..64 {
            let row = batch * seq + chunk * 64 + t;
            let index = (row * stride + 3 * d + head * 64 + j) as usize;
            let g = raw[index];
            c[3] += u32::from(cvt_rn_f16_f32(g) != saved[index]);
            c[4] += u32::from(!finite(g));
            acc += -decay_scale * softplus(g);
            unsafe {
                PREFIX[(t * 64 + j) as usize] = acc;
            }
            c[7] += u32::from(!finite(acc));
            if finite(acc) {
                c[0] += u32::from(acc < -KDA_DECAY_EXP_LIMIT || acc > KDA_DECAY_EXP_LIMIT);
                let excess = acc - clamp(acc);
                if t == 0 {
                    first_excess = excess;
                }
                if t == 0 || excess != previous {
                    run = 1;
                } else {
                    run += 1;
                }
                c[5] += t + 1 - run;
                previous = excess;
                last_excess = excess;
            }
        }
        let last = acc;
        let mut max_mismatch = 0f32;
        if finite(last) {
            c[2] = u32::from(last < -KDA_DECAY_EXP_LIMIT || last > KDA_DECAY_EXP_LIMIT);
            for t in 0..64 {
                let g = unsafe { PREFIX[(t * 64 + j) as usize] };
                if finite(g) {
                    let relative = last - g;
                    c[1] += u32::from(
                        relative < -KDA_DECAY_EXP_LIMIT || relative > KDA_DECAY_EXP_LIMIT,
                    );
                    let mismatch = abs_f32(clamp(last) - clamp(g) - clamp(relative));
                    c[6] += u32::from(mismatch != 0.);
                    max_mismatch = max_f32(max_mismatch, mismatch);
                }
            }
        }
        let cell = (group * 64 + j) as usize;
        for i in 0..8 {
            unsafe {
                *counts.get_unchecked_mut(cell * 8 + i) = c[i];
            }
        }
        let values = if finite(last) {
            [
                last,
                max_f32(-KDA_DECAY_EXP_LIMIT - last, 0.),
                max_f32(first_excess - last_excess, 0.),
                max_mismatch,
            ]
        } else {
            [0.; 4]
        };
        for i in 0..4 {
            unsafe {
                *extrema.get_unchecked_mut(cell * 4 + i) = values[i];
            }
        }
    }
}
