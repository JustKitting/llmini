//! One positional geometry for native rotation, inverse VJP and key routing.
use crate::float_ptx::{exp_f32, sincos_f32};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use std::sync::{Arc, OnceLock};
include!(concat!(env!("OUT_DIR"), "/position_config.rs"));

#[inline(always)]
pub const fn stationary_dim(dim: u32, head_dim: u32) -> bool {
    STATIONARY_ROPE && dim >= head_dim / 2
}
#[inline(always)]
pub const fn key_offset_dim(dim: u32, head_dim: u32) -> bool {
    if STATIONARY_ROPE {
        dim >= head_dim / 2
    } else {
        (dim >= head_dim / 4 && dim < head_dim / 2) || dim >= 3 * head_dim / 4
    }
}
#[inline(always)]
pub(crate) fn rotation_factors(token: u32, dim: u32, head_dim: u32) -> (f32, f32) {
    if stationary_dim(dim, head_dim) {
        return (0., 1.);
    }
    let even = dim & !1;
    let frequency = if STATIONARY_ROPE {
        if even == 0 {
            1.
        } else if even == head_dim / 2 - 2 {
            1. / 1024.
        } else {
            exp_f32(-6.931_472 * even as f32 / (head_dim / 2 - 2) as f32)
        }
    } else {
        exp_f32(-9.210_340_5 * even as f32 / head_dim as f32)
    };
    let (sin, cos) = sincos_f32(token as f32 * frequency);
    (if STATIONARY_ROPE { -sin } else { sin }, cos)
}
const fn signature() -> [u32; 7] {
    [
        1,
        STATIONARY_ROPE as u32,
        64,
        if STATIONARY_ROPE { 32 } else { 64 },
        if STATIONARY_ROPE {
            1024f32.to_bits()
        } else {
            10000f32.to_bits()
        },
        STATIONARY_ROPE as u32,
        if STATIONARY_ROPE { 2 } else { 1 },
    ]
}
pub struct PositionModule {
    graph: Option<graph::LoadedModule>,
    verified: OnceLock<bool>,
}
impl PositionModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        // Legacy default PTX predates the signature kernel. Discover exports
        // without a failing cuModuleGetFunction probe, which is a real CUDA API
        // error under memcheck even when the host later handles it.
        let graph = if crate::module_exports::contains(&module, b"position_graph_signature_kernel")?
        {
            Some(graph::from_module(module)?)
        } else {
            None
        };
        if STATIONARY_ROPE && graph.is_none() {
            return Err(DriverError(1));
        }
        Ok(Self {
            graph,
            verified: OnceLock::new(),
        })
    }
    pub fn verify_graph(&self, stream: &CudaStream) -> Result<(), DriverError> {
        let ok = *self.verified.get_or_init(|| {
            let Some(graph) = &self.graph else {
                return !STATIONARY_ROPE;
            };
            let result = (|| -> Result<Vec<u32>, DriverError> {
                let mut out = DeviceBuffer::zeroed(stream, 7)?;
                graph.position_graph_signature_kernel(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut out,
                )?;
                out.to_host_vec(stream)
            })();
            match result {
                Ok(actual) if actual == signature() => true,
                other => {
                    eprintln!(
                        "positional executable/PTX graph mismatch expected={:?} observed={other:?}",
                        signature()
                    );
                    false
                }
            }
        });
        if ok { Ok(()) } else { Err(DriverError(1)) }
    }
}

#[cuda_module]
pub mod graph {
    use super::*;
    #[kernel]
    pub fn position_graph_signature_kernel(mut out: DisjointSlice<u32>) {
        let i = thread::threadIdx_x() as usize;
        if i < 7 {
            unsafe {
                *out.get_unchecked_mut(i) = signature()[i];
            }
        }
    }
}
