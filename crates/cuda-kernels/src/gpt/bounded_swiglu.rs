//! Published two-cap/raw-sigmoid activation, with stable analytic tail factors.
use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::float_ptx::{abs_f32, exp_preserve_subnormal_f32};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

pub const ACTIVATION_INPUT_BITS: u32 = 16;
pub const PRECISION_POLICY_VERSION: u32 = 2;

#[inline(always)]
fn activation_input(value: f32) -> f32 {
    // Preserve the established preactivation precision. FP32 is the original
    // NVFP4 affine accumulator; neither forward nor recomputed reverse may
    // consume extra precision from it at the nonlinear input boundary.
    cvt_f32_f16(cvt_rn_f16_f32(value))
}

// A separate export preserves the old three-word ReLU2 signature ABI. New
// bounded hosts require this tag and reject the untrained FP32-input prototype.
#[cuda_module]
pub mod precision_graph {
    use super::*;
    #[kernel]
    pub fn bounded_swiglu_precision_signature_kernel(mut out: DisjointSlice<u32>) {
        if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = PRECISION_POLICY_VERSION;
                *out.get_unchecked_mut(1) = ACTIVATION_INPUT_BITS;
            }
        }
    }
}

#[cuda_module]
pub mod graph {
    use super::*;
    #[kernel]
    pub fn mlp_graph_signature_kernel(mut out: DisjointSlice<u32>) {
        if thread::blockIdx_x() != 0 || thread::threadIdx_x() != 0 {
            return;
        }
        unsafe {
            *out.get_unchecked_mut(0) = crate::mlp::config::BOUNDED_SWIGLU as u32;
            *out.get_unchecked_mut(1) = crate::mlp::config::FEATURE_TILES;
            *out.get_unchecked_mut(2) = crate::mlp::config::ACTIVE_FEATURE_TILES;
            // Preserve the original three-word ABI for archived hosts.
            if out.len() > 3 {
                *out.get_unchecked_mut(3) = crate::nvfp4_quant::ACTIVATION_MAE_FOUR_SIX as u32;
            }
        }
    }
}

#[inline(always)]
fn tanh_value(x: f32) -> f32 {
    let a = abs_f32(x);
    if a < 0.125 {
        // Avoid subtracting nearly equal exponentials at tiny inputs. The next
        // Taylor term is below an FP32 ULP throughout this small interval.
        let z = x * x;
        x * (1. + z * (-1. / 3. + z * (2. / 15. - z * (17. / 315.))))
    } else {
        let e = exp_preserve_subnormal_f32(-2. * a);
        let value = (1. - e) / (1. + e);
        if x < 0. { -value } else { value }
    }
}
#[inline(always)]
fn sigmoid_value(x: f32) -> f32 {
    let e = exp_preserve_subnormal_f32(-abs_f32(x));
    if x >= 0. { 1. / (1. + e) } else { e / (1. + e) }
}
#[inline(always)]
fn sigmoid_derivative(x: f32) -> f32 {
    let e = exp_preserve_subnormal_f32(-abs_f32(x));
    e / ((1. + e) * (1. + e))
}
#[inline(always)]
fn tanh_derivative(x: f32) -> f32 {
    let e = exp_preserve_subnormal_f32(-2. * abs_f32(x));
    4. * e / ((1. + e) * (1. + e))
}

#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn bounded_swiglu_forward_kernel(
        gate_up: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        h: u32,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i >= n * h {
            return;
        }
        let base = (i / h * (2 * h) + i % h) as usize;
        let gate = activation_input(gate_up[base]);
        let up = activation_input(gate_up[base + h as usize]);
        let value =
            (4. * tanh_value(gate / 4.)) * sigmoid_value(gate) * (25. * tanh_value(up / 25.));
        unsafe {
            *output.get_unchecked_mut(i as usize) = value;
        }
    }
    #[kernel]
    pub fn bounded_swiglu_backward_kernel(
        mut gate_up: DisjointSlice<f32>,
        upstream: &[f32],
        route_masks: &[u64],
        n: u32,
        h: u32,
        routed: bool,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i >= n * h {
            return;
        }
        let row = i / h;
        let feature = i % h;
        let base = (row * (2 * h) + feature) as usize;
        // One thread owns both positions of its gate/up pair. Read both before
        // replacing them, without aliasing a separate compact gradient buffer.
        unsafe {
            let pointer = gate_up.as_mut_ptr();
            if routed && route_masks[(row / 128) as usize] & (1u64 << (feature / 128)) == 0 {
                *pointer.add(base) = 0.;
                *pointer.add(base + h as usize) = 0.;
                return;
            }
            let g = activation_input(*pointer.add(base));
            let v = activation_input(*pointer.add(base + h as usize));
            let tg = tanh_value(g / 4.);
            let tu = tanh_value(v / 25.);
            let s = sigmoid_value(g);
            let dy = upstream[i as usize];
            // These equivalent exponential forms retain representable tails
            // after rounded tanh/sigmoid outputs have saturated to1.
            *pointer.add(base) =
                dy * (25. * tu) * (tanh_derivative(g / 4.) * s + (4. * tg) * sigmoid_derivative(g));
            *pointer.add(base + h as usize) = dy * (4. * tg) * s * tanh_derivative(v / 25.);
        }
    }
}
