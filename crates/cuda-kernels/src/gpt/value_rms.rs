//! Unweighted per-head value RMS with complete raw-coordinate reverse map.
use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::float_ptx::{abs_f32, exp_preserve_subnormal_f32, max_f32, sqrt_f32};
use crate::warp_reduce::{warp_max_f32, warp_sum_f32};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};
use std::sync::{Arc, OnceLock};
pub mod config {
    include!(concat!(env!("OUT_DIR"), "/value_config.rs"));
}
pub const EPSILON: f32 = 1e-6;
pub const PRECISION_POLICY_VERSION: u32 = 2;
pub const FORWARD_INPUT_BITS: u32 = 32;
pub const REVERSE_SILU_INPUT_BITS: u32 = 16;
pub const REVERSE_DIRECTION_BITS: u32 = 16;
pub const CONTROL_BITS: u32 = 32;
pub const PRECISION_POLICY: [u32; 5] = [
    PRECISION_POLICY_VERSION,
    FORWARD_INPUT_BITS,
    REVERSE_SILU_INPUT_BITS,
    REVERSE_DIRECTION_BITS,
    CONTROL_BITS,
];

pub struct ValueRmsModule {
    kernels: Option<kernels::LoadedModule>,
    graph: Option<graph::LoadedModule>,
    precision: Option<precision_graph::LoadedModule>,
    point_check: Option<point_verification::LoadedModule>,
    verified: OnceLock<bool>,
}
impl ValueRmsModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        // Never issue a missing-export CUDA call, even for a disabled optional
        // graph. New default hosts must still inspect legacy PTX cleanly.
        let graph =
            if crate::module_exports::contains(&module, b"value_rms_graph_signature_kernel")? {
                Some(graph::from_module(module.clone())?)
            } else {
                None
            };
        let precision = if config::VALUE_RMS {
            if !crate::module_exports::contains(&module, b"value_rms_precision_signature_kernel")? {
                eprintln!(
                    "value RMS requires the represented FP16 reverse precision signature; refusing the old FP32-point prototype"
                );
                return Err(DriverError(1));
            }
            Some(precision_graph::from_module(module.clone())?)
        } else {
            None
        };
        let point_check = if config::VALUE_RMS {
            if !crate::module_exports::contains(&module, b"value_rms_stored_point_check_kernel")? {
                return Err(DriverError(1));
            }
            Some(point_verification::from_module(module.clone())?)
        } else {
            None
        };
        let kernels = if crate::module_exports::contains(&module, b"value_rms_forward_kernel")?
            && crate::module_exports::contains(&module, b"value_rms_backward_kernel")?
        {
            Some(kernels::from_module(module)?)
        } else {
            None
        };
        if config::VALUE_RMS && (graph.is_none() || kernels.is_none()) {
            return Err(DriverError(1));
        }
        Ok(Self {
            kernels,
            graph,
            precision,
            point_check,
            verified: OnceLock::new(),
        })
    }
    pub fn verify_graph(&self, s: &CudaStream) -> Result<(), DriverError> {
        let ok=*self.verified.get_or_init(|| {
            let Some(g)=&self.graph else{return !config::VALUE_RMS;};
            let result=(||->Result<Vec<u32>,DriverError>{
                let mut out=DeviceBuffer::zeroed(s,3)?;
                g.value_rms_graph_signature_kernel(s,LaunchConfig{grid_dim:(1,1,1),block_dim:(32,1,1),shared_mem_bytes:0},&mut out)?;
                let mut actual=out.to_host_vec(s)?;
                if let Some(p)=&self.precision {
                    let mut policy=DeviceBuffer::zeroed(s,5)?;
                    p.value_rms_precision_signature_kernel(s,LaunchConfig{grid_dim:(1,1,1),block_dim:(32,1,1),shared_mem_bytes:0},&mut policy)?;
                    actual.extend(policy.to_host_vec(s)?);
                }
                Ok(actual)
            })();
            let mut expected=vec![config::VALUE_RMS as u32,64,EPSILON.to_bits()];
            if config::VALUE_RMS {expected.extend(PRECISION_POLICY);}
            match result {Ok(actual)if actual==expected=>true,other=>{eprintln!("value-RMS executable/PTX graph mismatch expected={expected:?} observed={other:?}");false}}
        });
        if ok { Ok(()) } else { Err(DriverError(1)) }
    }
    pub fn forward(
        &self,
        s: &CudaStream,
        qkv: &mut DeviceBuffer<f32>,
        n: u32,
        d: u32,
        stride: u32,
        kda: bool,
    ) -> Result<(), DriverError> {
        self.verify_graph(s)?;
        validate(qkv.len(), n, d, stride)?;
        self.kernels
            .as_ref()
            .ok_or(DriverError(1))?
            .value_rms_forward_kernel(s, launch(n, d), qkv, n, d, stride, kda as u32)
    }
    pub fn backward(
        &self,
        s: &CudaStream,
        raw: &DeviceBuffer<f32>,
        dqkv: &mut DeviceBuffer<f32>,
        n: u32,
        d: u32,
        stride: u32,
        kda: bool,
    ) -> Result<(), DriverError> {
        self.verify_graph(s)?;
        validate(raw.len(), n, d, stride)?;
        validate(dqkv.len(), n, d, stride)?;
        self.kernels
            .as_ref()
            .ok_or(DriverError(1))?
            .value_rms_backward_kernel(s, launch(n, d), raw, dqkv, n, d, stride, kda as u32)
    }

    /// Native verification only: compare the reverse reconstruction with the
    /// actual FP16 forward tape. No optimizer or persistent state is touched.
    pub fn check_stored_point(
        &self,
        s: &CudaStream,
        raw: &DeviceBuffer<f32>,
        stored: &DeviceBuffer<u16>,
        mismatches: &mut DeviceBuffer<u32>,
        n: u32,
        d: u32,
        stride: u32,
        kda: bool,
    ) -> Result<(), DriverError> {
        self.verify_graph(s)?;
        validate(raw.len(), n, d, stride)?;
        validate(stored.len(), n, d, stride)?;
        if mismatches.len() < (n as usize) * (d as usize / 64) {
            return Err(DriverError(1));
        }
        self.point_check
            .as_ref()
            .ok_or(DriverError(1))?
            .value_rms_stored_point_check_kernel(
                s,
                launch(n, d),
                raw,
                stored,
                mismatches,
                n,
                d,
                stride,
                kda as u32,
            )
    }
}
fn validate(len: usize, n: u32, d: u32, stride: u32) -> Result<(), DriverError> {
    if n == 0 || d == 0 || d % 64 != 0 || stride < 3 * d || len < (n as usize) * (stride as usize) {
        Err(DriverError(1))
    } else {
        Ok(())
    }
}
fn launch(n: u32, d: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n * (d / 64)).div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cuda_module]
pub mod graph {
    use super::*;
    #[kernel]
    pub fn value_rms_graph_signature_kernel(mut out: DisjointSlice<u32>) {
        if thread::threadIdx_x() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = config::VALUE_RMS as u32;
                *out.get_unchecked_mut(1) = 64;
                *out.get_unchecked_mut(2) = EPSILON.to_bits();
            }
        }
    }
}

#[cuda_module]
pub mod precision_graph {
    use super::*;
    #[kernel]
    pub fn value_rms_precision_signature_kernel(mut out: DisjointSlice<u32>) {
        if thread::threadIdx_x() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = PRECISION_POLICY_VERSION;
                *out.get_unchecked_mut(1) = FORWARD_INPUT_BITS;
                *out.get_unchecked_mut(2) = REVERSE_SILU_INPUT_BITS;
                *out.get_unchecked_mut(3) = REVERSE_DIRECTION_BITS;
                *out.get_unchecked_mut(4) = CONTROL_BITS;
            }
        }
    }
}

// Multiplying the requested scale before the two half-exponent factors keeps
// representable normalized SiLU tails from disappearing in an intermediate.
#[inline(always)]
fn silu_scaled(x: f32, scale: f32) -> f32 {
    let ehalf = exp_preserve_subnormal_f32(-0.5 * abs_f32(x));
    let e = ehalf * ehalf;
    let denom = 1.0 + e;
    if ehalf == 0.0 {
        return if x >= 0.0 { scale * x } else { 0.0 };
    }
    if x >= 0.0 {
        (scale * x) / denom
    } else {
        ((scale * x * ehalf) * ehalf) / denom
    }
}
#[inline(always)]
fn silu_grad_scaled(x: f32, scale: f32) -> f32 {
    let ehalf = exp_preserve_subnormal_f32(-0.5 * abs_f32(x));
    let e = ehalf * ehalf;
    let denom = 1.0 + e;
    if ehalf == 0.0 {
        return if x >= 0.0 { scale } else { 0.0 };
    }
    let sigmoid_part = if x >= 0.0 {
        scale / denom
    } else {
        ((scale * ehalf) * ehalf) / denom
    };
    sigmoid_part + ((scale * x * ehalf) * ehalf) / (denom * denom)
}
#[inline(always)]
fn effective(raw: f32, scale: f32, kda: u32) -> f32 {
    if kda != 0 {
        silu_scaled(raw, scale)
    } else {
        raw * scale
    }
}
#[derive(Clone, Copy)]
struct Stats {
    z0: f32,
    z1: f32,
    square: f32,
    epsilon: f32,
    inverse: f32,
    denom: f32,
}
#[inline(always)]
fn stats(raw0: f32, raw1: f32, kda: u32) -> Stats {
    let a0 = effective(raw0, 1.0, kda);
    let a1 = effective(raw1, 1.0, kda);
    let max = warp_max_f32(max_f32(abs_f32(a0), abs_f32(a1)));
    let scale = max_f32(max, 0.001);
    let inv_scale = 1.0 / scale;
    let z0 = effective(raw0, inv_scale, kda);
    let z1 = effective(raw1, inv_scale, kda);
    let square = warp_sum_f32(z0 * z0 + z1 * z1) * (1.0 / 64.0);
    let epsilon = (EPSILON / scale) / scale;
    let denom = square + epsilon;
    let inverse_norm = 1.0 / sqrt_f32(denom);
    Stats {
        z0,
        z1,
        square,
        epsilon,
        inverse: inv_scale * inverse_norm,
        denom: inverse_norm,
    }
}
#[inline(always)]
fn reverse_one(
    raw_half: u16,
    direction: f32,
    g: f32,
    dot: f32,
    direction_square: f32,
    a: Stats,
    kda: u32,
) -> f32 {
    let projection = if direction_square > 0. {
        direction * (dot / direction_square)
    } else {
        0.
    };
    // Keep the real epsilon radial derivative, including constant heads.
    // a contains FP32 scalar normalization controls; reverse directions and
    // nonlinear inputs have already discarded all non-FP16 input bits.
    let tangent = (g - projection) + projection * (a.epsilon / (a.square + a.epsilon));
    let scaled = tangent * a.inverse;
    if kda != 0 {
        silu_grad_scaled(cvt_f32_f16(raw_half), scaled)
    } else {
        scaled
    }
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn value_rms_forward_kernel(
        mut qkv: DisjointSlice<f32>,
        n: u32,
        d: u32,
        stride: u32,
        kda: u32,
    ) {
        let head = thread::blockIdx_x() * 8 + thread::threadIdx_x() / 32;
        if head >= n * (d / 64) {
            return;
        }
        let row = head / (d / 64);
        let h = head % (d / 64);
        let lane = warp::lane_id();
        let i = (row * stride + 2 * d + h * 64 + lane) as usize;
        let x0 = unsafe { *qkv.get_unchecked_mut(i) };
        let x1 = unsafe { *qkv.get_unchecked_mut(i + 32) };
        let a = stats(x0, x1, kda);
        unsafe {
            *qkv.get_unchecked_mut(i) = a.z0 * a.denom;
            *qkv.get_unchecked_mut(i + 32) = a.z1 * a.denom;
        }
    }
    #[kernel]
    pub fn value_rms_backward_kernel(
        raw: &[f32],
        mut dqkv: DisjointSlice<f32>,
        n: u32,
        d: u32,
        stride: u32,
        kda: u32,
    ) {
        let head = thread::blockIdx_x() * 8 + thread::threadIdx_x() / 32;
        if head >= n * (d / 64) {
            return;
        }
        let row = head / (d / 64);
        let h = head % (d / 64);
        let lane = warp::lane_id();
        let i = (row * stride + 2 * d + h * 64 + lane) as usize;
        let x0 = raw[i];
        let x1 = raw[i + 32];
        let a = stats(x0, x1, kda);
        let g0 = unsafe { *dqkv.get_unchecked_mut(i) };
        let g1 = unsafe { *dqkv.get_unchecked_mut(i + 32) };
        // Reconstruct precisely the existing normalized-V FP16 tape point,
        // rather than retaining the prototype's FP32 reverse directions.
        let y0 = cvt_f32_f16(cvt_rn_f16_f32(a.z0 * a.denom));
        let y1 = cvt_f32_f16(cvt_rn_f16_f32(a.z1 * a.denom));
        let direction_square = warp_sum_f32(y0 * y0 + y1 * y1) * (1.0 / 64.0);
        let dot = warp_sum_f32(g0 * y0 + g1 * y1) * (1.0 / 64.0);
        let raw0 = if kda != 0 { cvt_rn_f16_f32(x0) } else { 0 };
        let raw1 = if kda != 0 { cvt_rn_f16_f32(x1) } else { 0 };
        let dx0 = reverse_one(raw0, y0, g0, dot, direction_square, a, kda);
        let dx1 = reverse_one(raw1, y1, g1, dot, direction_square, a, kda);
        unsafe {
            *dqkv.get_unchecked_mut(i) = dx0;
            *dqkv.get_unchecked_mut(i + 32) = dx1;
        }
    }
}

#[cuda_module]
pub mod point_verification {
    use super::*;
    #[kernel]
    pub fn value_rms_stored_point_check_kernel(
        raw: &[f32],
        stored: &[u16],
        mut mismatches: DisjointSlice<u32>,
        n: u32,
        d: u32,
        stride: u32,
        kda: u32,
    ) {
        let head = thread::blockIdx_x() * 8 + thread::threadIdx_x() / 32;
        if head >= n * (d / 64) {
            return;
        }
        let lane = warp::lane_id();
        let row = head / (d / 64);
        let h = head % (d / 64);
        let i = (row * stride + 2 * d + h * 64 + lane) as usize;
        let a = stats(raw[i], raw[i + 32], kda);
        let first = cvt_rn_f16_f32(a.z0 * a.denom);
        let second = cvt_rn_f16_f32(a.z1 * a.denom);
        let failures = warp_sum_f32(
            (first != stored[i]) as u32 as f32 + (second != stored[i + 32]) as u32 as f32,
        );
        if lane == 0 {
            unsafe {
                *mismatches.get_unchecked_mut(head as usize) = failures as u32;
            }
        }
    }
}
