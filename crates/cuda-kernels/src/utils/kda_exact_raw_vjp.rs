//! Original-FP32 raw KDA VJP, using the actual shared trainer math helpers.
//! No default runtime dispatch is changed and no projection backend is added.
use crate::attention::{CausalAttentionParams, exact_kda_raw_finish_body};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};
use std::sync::Arc;
pub struct KdaExactRawVjpModule {
    kernels: kernels::LoadedModule,
}
pub struct RawVjpArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<f32>,
    pub dq: &'a DeviceBuffer<f32>,
    pub dk: &'a DeviceBuffer<f32>,
    pub dg: &'a DeviceBuffer<f32>,
    pub dvbeta: &'a DeviceBuffer<f32>,
    pub dbeta: &'a DeviceBuffer<f32>,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub d_qkv_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub accumulate_value_grad: bool,
    pub batch_size: u32,
    pub seq_len: u32,
    pub head_count: u32,
    pub qkv_dim: u32,
}
impl KdaExactRawVjpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        if !crate::module_exports::contains(&module, b"kda_exact_raw_finish_kernel")? {
            return Err(DriverError(1));
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn vjp(&self, a: RawVjpArgs<'_, '_>) -> Result<u32, DriverError> {
        let chunks = a
            .seq_len
            .checked_add(63)
            .map(|n| n / 64)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))?;
        let bh = a
            .batch_size
            .checked_mul(a.head_count)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))?;
        let batch = bh.checked_mul(chunks).ok_or(DriverError(1))?;
        let n = batch.checked_mul(4096).ok_or(DriverError(1))? as usize;
        let dim = a.head_count.checked_mul(64).ok_or(DriverError(1))?;
        let active = dim
            .checked_mul(4)
            .and_then(|n| n.checked_add(a.head_count))
            .ok_or(DriverError(1))?;
        let max_stride = a
            .head_count
            .checked_mul(32)
            .and_then(|n| n.checked_add(active))
            .ok_or(DriverError(1))?;
        let rows = a.batch_size.checked_mul(a.seq_len).ok_or(DriverError(1))?;
        let raw = rows.checked_mul(a.qkv_dim).ok_or(DriverError(1))? as usize;
        let amax_count = batch.checked_mul(8).ok_or(DriverError(1))?;
        if a.qkv_dim < active
            || a.qkv_dim > max_stride
            || a.qkv.len() < raw
            || a.d_qkv.len() < raw
            || [a.dq.len(), a.dk.len(), a.dg.len(), a.dvbeta.len()]
                .iter()
                .any(|&len| len < n)
            || a.dbeta.len() < n / 64
            || a.d_qkv_chunk_amax.len() < amax_count as usize
        {
            return Err(DriverError(1));
        }
        let params = CausalAttentionParams::new(
            rows,
            a.seq_len,
            a.batch_size,
            dim,
            a.qkv_dim,
            a.head_count,
            64,
        );
        self.kernels.kda_exact_raw_finish_kernel(
            a.stream,
            LaunchConfig {
                grid_dim: (bh, chunks, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            a.qkv,
            a.dq,
            a.dk,
            a.dg,
            a.dvbeta,
            a.dbeta,
            a.d_qkv,
            a.d_qkv_chunk_amax,
            u32::from(a.accumulate_value_grad),
            params,
        )?;
        Ok(amax_count)
    }
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_exact_raw_finish_kernel(
        qkv: &[f32],
        dq: &[f32],
        dk: &[f32],
        dg: &[f32],
        dvbeta: &[f32],
        dbeta: &[f32],
        out: DisjointSlice<f32>,
        amax: DisjointSlice<f32>,
        accumulate_value_grad: u32,
        params: CausalAttentionParams,
    ) {
        static mut AMAX: SharedArray<f32, 4> = SharedArray::UNINIT;
        unsafe {
            exact_kda_raw_finish_body(
                qkv,
                dq,
                dk,
                dg,
                dvbeta,
                dbeta,
                out,
                amax,
                accumulate_value_grad,
                params,
                &mut AMAX,
            );
        }
    }
}
