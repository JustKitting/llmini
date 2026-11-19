//! True carry VJP from native shared-state replay. Saved operands/cotangents
//! remain FP16. Only scalar partial reductions and row controls occupy FP32
//! global storage; no FP32 entry-state plane is allocated or exposed.
use crate::attention::{CausalAttentionParams, exact_kda_state_carry_body};
use crate::kda_tc::with_kda_tiles;
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm, thread};
use std::sync::Arc;
pub struct KdaExactStateCarryVjpModule {
    kernels: kernels::LoadedModule,
}
pub struct CarryVjpArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub kg: &'a DeviceBuffer<u16>,
    pub residual: &'a DeviceBuffer<u16>,
    pub carry: &'a DeviceBuffer<f32>,
    pub state_grad: &'a DeviceBuffer<u16>,
    /// Two power-of-two inverse scales per64-value state-gradient row.
    pub state_grad_unscale: &'a DeviceBuffer<f32>,
    /// Optional input to the working accumulator, matching the forward API.
    /// The training graph's default initial state is zero.
    pub initial_state: Option<&'a DeviceBuffer<f32>>,
    /// Two scalar32-value partials per output row, not a recurrent-state tape.
    pub partials: &'out mut DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_size: u32,
    pub seq_len: u32,
    pub head_count: u32,
}
impl KdaExactStateCarryVjpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        for name in [
            b"kda_exact_state_carry_zero_kernel".as_slice(),
            b"kda_exact_state_carry_initial_kernel".as_slice(),
            b"kda_exact_state_carry_reduce_kernel".as_slice(),
        ] {
            if !crate::module_exports::contains(&module, name)? {
                return Err(DriverError(1));
            }
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn vjp(&self, a: CarryVjpArgs<'_, '_>) -> Result<(), DriverError> {
        let chunks = a
            .seq_len
            .checked_add(63)
            .map(|s| s / 64)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))?;
        let bh = a
            .batch_size
            .checked_mul(a.head_count)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))?;
        let n = bh
            .checked_mul(chunks)
            .and_then(|n| n.checked_mul(4096))
            .ok_or(DriverError(1))?;
        let controls = n / 64;
        let state = bh.checked_mul(4096).ok_or(DriverError(1))? as usize;
        let rows = a.batch_size.checked_mul(a.seq_len).ok_or(DriverError(1))?;
        let dim = a.head_count.checked_mul(64).ok_or(DriverError(1))?;
        if [a.kg.len(), a.residual.len(), a.state_grad.len()]
            .iter()
            .any(|&len| len < n as usize)
            || a.carry.len() < controls as usize
            || a.state_grad_unscale.len() < (controls * 2) as usize
            || a.partials.len() < (controls * 2) as usize
            || a.out.len() < controls as usize
            || a.initial_state.is_some_and(|s| s.len() < state)
        {
            return Err(DriverError(1));
        }
        // No affine planes are accessed by this operator.
        let params =
            CausalAttentionParams::new(rows, a.seq_len, a.batch_size, dim, 0, a.head_count, 64);
        let config = LaunchConfig {
            grid_dim: (bh, 2, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        if let Some(initial) = a.initial_state {
            self.kernels.kda_exact_state_carry_initial_kernel(
                a.stream,
                config,
                a.kg,
                a.residual,
                a.carry,
                a.state_grad,
                a.partials,
                initial,
                params,
            )?;
        } else {
            self.kernels.kda_exact_state_carry_zero_kernel(
                a.stream,
                config,
                a.kg,
                a.residual,
                a.carry,
                a.state_grad,
                a.partials,
                params,
            )?;
        }
        self.kernels.kda_exact_state_carry_reduce_kernel(
            a.stream,
            LaunchConfig {
                grid_dim: (controls.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &*a.partials,
            a.state_grad_unscale,
            a.out,
            controls,
        )
    }
}
#[inline(always)]
fn multiply(a: f32, b: f32) -> f32 {
    let value: f32;
    unsafe {
        ptx_asm!("mul.rn.f32 %0, %1, %2;",out("=f") value,in("f") a,in("f") b,options(register_only));
    }
    value
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_exact_state_carry_zero_kernel(
        kg: &[u16],
        residual: &[u16],
        carry: &[f32],
        state_grad: &[u16],
        partials: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_kda_tiles!(state exact_kda_state_carry_body;kg,residual,carry,state_grad,partials,0 as *const f32,params);
    }
    #[kernel]
    pub fn kda_exact_state_carry_initial_kernel(
        kg: &[u16],
        residual: &[u16],
        carry: &[f32],
        state_grad: &[u16],
        partials: DisjointSlice<f32>,
        initial: &[f32],
        params: CausalAttentionParams,
    ) {
        with_kda_tiles!(state exact_kda_state_carry_body;kg,residual,carry,state_grad,partials,initial.as_ptr(),params);
    }
    #[kernel]
    pub fn kda_exact_state_carry_reduce_kernel(
        partials: &[f32],
        unscale: &[f32],
        mut out: DisjointSlice<f32>,
        count: u32,
    ) {
        let row = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if row < count {
            let i = (row * 2) as usize;
            let value = multiply(
                multiply(partials[i] + partials[i + 1], unscale[i]),
                unscale[i + 1],
            );
            unsafe {
                *out.get_unchecked_mut(row as usize) = value;
            }
        }
    }
}
