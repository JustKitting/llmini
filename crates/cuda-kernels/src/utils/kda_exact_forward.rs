//! Complete bounded-interval forward operator. Projection inputs remain native
//! NVFP4 accumulators; attention uses original FP16 TC/FP32-control arithmetic.
//! This module is explicitly loaded and never changes default model dispatch.
use crate::attention::{CausalAttentionParams, exact_kda_output_body, exact_kda_state_body};
use crate::f16_tc_matmul::{F16TcMatmulKdaWuArgs, F16TcMatmulModule};
use crate::kda_exact_factors::{FormedInverseArgs, KdaExactFactorModule};
use crate::kda_exact_pairs::{KdaExactPairModule, PairArgs};
use crate::kda_exact_prepare::{FactorArgs, KdaExactPrepareModule, PrepareArgs};
use crate::kda_tc::{with_kda_tiles, with_tc_ab_tiles};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};
use std::sync::Arc;

pub struct KdaExactForwardModule {
    prepare: KdaExactPrepareModule,
    pairs: KdaExactPairModule,
    inverse: KdaExactFactorModule,
    tc: F16TcMatmulModule,
    kernels: kernels::LoadedModule,
}
/// All compact tensors are [batch_head,ceil(S/64),64,64]. beta and exit_decay
/// use one feature/token respectively; the original B/U packed alias is reused.
pub struct ForwardScratch<'a> {
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub g: &'a mut DeviceBuffer<f32>,
    pub beta: &'a mut DeviceBuffer<f32>,
    pub lower: &'a mut DeviceBuffer<f32>,
    pub incoming_rhs_u: &'a mut DeviceBuffer<f32>,
    pub qg: &'a mut DeviceBuffer<u16>,
    pub kg: &'a mut DeviceBuffer<u16>,
    pub vbeta: &'a mut DeviceBuffer<u16>,
    pub inverse: &'a mut DeviceBuffer<u16>,
    pub w: &'a mut DeviceBuffer<u16>,
    pub readout: &'a mut DeviceBuffer<u16>,
    pub residual: &'a mut DeviceBuffer<u16>,
    pub chunk_states: &'a mut DeviceBuffer<u16>,
    pub final_state: &'a mut DeviceBuffer<f32>,
}
pub struct ForwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<f32>,
    pub qkv_f16: Option<&'out mut DeviceBuffer<u16>>,
    pub initial_state: Option<&'a DeviceBuffer<f32>>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub scratch: ForwardScratch<'scratch>,
    pub batch_size: u32,
    pub seq_len: u32,
    pub head_count: u32,
    pub qkv_dim: u32,
}
impl KdaExactForwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        for name in [
            b"kda_exact_state_zero_kernel".as_slice(),
            b"kda_exact_state_initial_kernel".as_slice(),
            b"kda_exact_readout_kernel".as_slice(),
        ] {
            if !crate::module_exports::contains(&module, name)? {
                return Err(DriverError(1));
            }
        }
        Ok(Self {
            prepare: KdaExactPrepareModule::from_module(module.clone())?,
            pairs: KdaExactPairModule::from_module(module.clone())?,
            inverse: KdaExactFactorModule::from_module(module.clone())?,
            tc: F16TcMatmulModule::from_module(module.clone())?,
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn forward(&self, args: ForwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
        let chunks = args
            .seq_len
            .checked_add(63)
            .map(|x| x / 64)
            .filter(|&x| x > 0)
            .ok_or(DriverError(1))?;
        let bh = args
            .batch_size
            .checked_mul(args.head_count)
            .filter(|&x| x > 0)
            .ok_or(DriverError(1))?;
        let chunk_batch = bh
            .checked_mul(chunks)
            .filter(|&x| x <= 65535)
            .ok_or(DriverError(1))?;
        let n = chunk_batch.checked_mul(4096).ok_or(DriverError(1))? as usize;
        let dim = args.head_count.checked_mul(64).ok_or(DriverError(1))?;
        let rows = args
            .batch_size
            .checked_mul(args.seq_len)
            .ok_or(DriverError(1))?;
        let hidden = rows.checked_mul(dim).ok_or(DriverError(1))? as usize;
        let raw = rows.checked_mul(args.qkv_dim).ok_or(DriverError(1))? as usize;
        let state = bh.checked_mul(4096).ok_or(DriverError(1))? as usize;
        let s = args.scratch;
        if args.qkv.len() < raw
            || args.out.len() < hidden
            || args.qkv_f16.as_ref().is_some_and(|x| x.len() < raw)
            || args.initial_state.is_some_and(|x| x.len() < state)
            || s.final_state.len() < state
            || s.beta.len() < n / 64
            || [
                s.q.len(),
                s.k.len(),
                s.g.len(),
                s.lower.len(),
                s.incoming_rhs_u.len(),
                s.qg.len(),
                s.kg.len(),
                s.vbeta.len(),
                s.inverse.len(),
                s.w.len(),
                s.readout.len(),
                s.residual.len(),
                s.chunk_states.len(),
            ]
            .iter()
            .any(|&x| x < n)
        {
            return Err(DriverError(1));
        }
        // raw prepare validates the remaining affine-stride bounds before any
        // launch. Beta stays live through the pair and boundary producers.
        self.prepare.prepare(PrepareArgs {
            stream: args.stream,
            qkv: args.qkv,
            qkv_f16: args.qkv_f16,
            q: &mut *s.q,
            k: &mut *s.k,
            g: &mut *s.g,
            beta: &mut *s.beta,
            vbeta: &mut *s.vbeta,
            batch_size: args.batch_size,
            seq_len: args.seq_len,
            head_count: args.head_count,
            qkv_dim: args.qkv_dim,
        })?;
        self.pairs.pairs(PairArgs {
            stream: args.stream,
            q: &*s.q,
            k: &*s.k,
            g: &*s.g,
            beta: &*s.beta,
            lower: &mut *s.lower,
            readout: &mut *s.readout,
            batch_heads: bh,
            seq_len: args.seq_len,
        })?;
        // lower is dead after inversion. Its complete FP32 allocation can own
        // the bounded terminal vector while B/U retains its established alias.
        self.inverse.formed_inverse(FormedInverseArgs {
            stream: args.stream,
            lower: &*s.lower,
            out: &mut *s.inverse,
            chunk_batch,
        })?;
        self.prepare.factors(FactorArgs {
            stream: args.stream,
            q: &*s.q,
            k: &*s.k,
            g: &*s.g,
            beta: &*s.beta,
            qg: &mut *s.qg,
            kg: &mut *s.kg,
            incoming_rhs_u: &mut *s.incoming_rhs_u,
            exit_decay: &mut *s.lower,
            batch_heads: bh,
            seq_len: args.seq_len,
        })?;
        self.tc.kda_wu_inplace(F16TcMatmulKdaWuArgs {
            stream: args.stream,
            inverse: &*s.inverse,
            vbeta: &*s.vbeta,
            kpos_u: &mut *s.incoming_rhs_u,
            w: &mut *s.w,
            batch_count: chunk_batch,
        })?;
        let params = CausalAttentionParams::new(
            rows,
            args.seq_len,
            args.batch_size,
            dim,
            args.qkv_dim,
            args.head_count,
            64,
        );
        let state_config = LaunchConfig {
            grid_dim: (bh, 2, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        if let Some(initial) = args.initial_state {
            self.kernels.kda_exact_state_initial_kernel(
                args.stream,
                state_config,
                &*s.kg,
                &mut *s.residual,
                &*s.w,
                &*s.incoming_rhs_u,
                &*s.lower,
                &mut *s.chunk_states,
                initial,
                &mut *s.final_state,
                params,
            )?;
        } else {
            self.kernels.kda_exact_state_zero_kernel(
                args.stream,
                state_config,
                &*s.kg,
                &mut *s.residual,
                &*s.w,
                &*s.incoming_rhs_u,
                &*s.lower,
                &mut *s.chunk_states,
                &mut *s.final_state,
                params,
            )?;
        }
        self.kernels.kda_exact_readout_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (bh, chunks, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            },
            &*s.qg,
            &*s.residual,
            &*s.readout,
            args.out,
            &*s.chunk_states,
            params,
        )
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_exact_state_zero_kernel(
        kg: &[u16],
        residual: DisjointSlice<u16>,
        w: &[u16],
        u: &[f32],
        carry: &[f32],
        states: DisjointSlice<u16>,
        mut final_state: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_kda_tiles!(state exact_kda_state_body;kg,residual,w,u,carry,states,0 as *const f32,final_state.as_mut_ptr(),params);
    }
    #[kernel]
    pub fn kda_exact_state_initial_kernel(
        kg: &[u16],
        residual: DisjointSlice<u16>,
        w: &[u16],
        u: &[f32],
        carry: &[f32],
        states: DisjointSlice<u16>,
        initial: &[f32],
        mut final_state: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_kda_tiles!(state exact_kda_state_body;kg,residual,w,u,carry,states,initial.as_ptr(),final_state.as_mut_ptr(),params);
    }
    #[kernel]
    pub fn kda_exact_readout_kernel(
        qg: &[u16],
        residual: &[u16],
        readout: &[u16],
        out: DisjointSlice<f32>,
        states: &[u16],
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(exact_kda_output_body;qg,residual,readout,out,states,params);
    }
}
