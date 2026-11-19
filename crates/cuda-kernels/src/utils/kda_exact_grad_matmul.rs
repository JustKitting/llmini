//! Locally scaled FP16-TC products for attention cotangents. FP32 is consumed
//! only by operand packing and used for accumulation/output, never as an MMA
//! multiplicand. No global packed operand or recurrent-state tape is allocated.
//! This explicitly loaded module does not alter the default training graph.
use crate::f16_tc_matmul::convert::{cvt_rn_f16_f32, store_f16x2_shared};
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::kda_tc::{CtaATile, CtaBTile, mma_accumulate, swizzle_tc_offset};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, ptx_asm, thread};
use std::sync::Arc;
const STRIDE: usize = 66;
type GradTile = SharedArray<u16, { 64 * STRIDE }>;
type ScaleTile = SharedArray<f32, 128>;
pub struct KdaExactGradMatmulModule {
    kernels: kernels::LoadedModule,
}
pub struct GradMatmulArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub grad: &'a DeviceBuffer<f32>,
    pub operand: &'a DeviceBuffer<u16>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub chunk_batch: u32,
    pub transpose_grad: bool,
    pub transpose_operand: bool,
    pub transpose_output: bool,
    pub accumulate: bool,
    pub alpha: f32,
    pub rows: u32,
    pub cols: u32,
    pub inner: u32,
}
impl KdaExactGradMatmulModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        if !crate::module_exports::contains(&module, b"kda_exact_scaled_grad_matmul_kernel")? {
            return Err(DriverError(1));
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn matmul(&self, a: GradMatmulArgs<'_, '_>) -> Result<(), DriverError> {
        let n = a
            .chunk_batch
            .checked_mul(4096)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))? as usize;
        if a.grad.len() < n
            || a.operand.len() < n
            || a.out.len() < n
            || !a.alpha.is_finite()
            || [a.rows, a.cols, a.inner].iter().any(|&x| x == 0 || x > 64)
        {
            return Err(DriverError(1));
        }
        let flags = u32::from(a.transpose_grad)
            | (u32::from(a.transpose_operand) << 1)
            | (u32::from(a.transpose_output) << 2)
            | (u32::from(a.accumulate) << 3);
        self.kernels.kda_exact_scaled_grad_matmul_kernel(
            a.stream,
            LaunchConfig {
                grid_dim: (a.chunk_batch, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            },
            a.grad,
            a.operand,
            a.out,
            flags,
            a.alpha,
            a.rows,
            a.cols,
            a.inner,
        )
    }
}
// Explicit non-FTZ multiplication also covers representable FP32 subnormals.
#[inline(always)]
fn multiply(a: f32, b: f32) -> f32 {
    let value: f32;
    unsafe {
        ptx_asm!("mul.rn.f32 %0, %1, %2;",out("=f") value,in("f") a,in("f") b,options(register_only));
    }
    value
}
#[inline(always)]
fn power(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}
#[inline(always)]
fn inverse_power(value: f32) -> f32 {
    f32::from_bits((254 - (value.to_bits() >> 23)) << 23)
}
#[inline(always)]
fn grad_index(base: u32, r: u32, k: u32, flags: u32) -> usize {
    (base
        + if flags & 1 != 0 {
            k * 64 + r
        } else {
            r * 64 + k
        }) as usize
}
#[inline(always)]
fn scaled_grad_matmul_body(
    grad: &[f32],
    operand: &[u16],
    mut out: DisjointSlice<f32>,
    flags: u32,
    alpha: f32,
    rows: u32,
    cols: u32,
    inner: u32,
    packed: &mut GradTile,
    scales: &mut ScaleTile,
    a: &mut CtaATile,
    b: &mut CtaBTile,
) {
    let batch = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    let base = batch * 4096;
    if tid < 64 {
        let row = tid;
        let mut max_bits = 0u32;
        let mut k = 0;
        if row < rows {
            while k < inner {
                max_bits =
                    max_bits.max(grad[grad_index(base, row, k, flags)].to_bits() & 0x7fffffff);
                k += 1;
            }
        }
        let shift = if max_bits == 0 || max_bits >= 0x7f800000 {
            0
        } else {
            let encoded = max_bits >> 23;
            let exponent = if encoded == 0 {
                let mut mantissa = max_bits;
                let mut exponent = -149;
                while mantissa > 1 {
                    mantissa >>= 1;
                    exponent += 1;
                }
                exponent
            } else {
                encoded as i32 - 127
            };
            4 - exponent
        };
        let first = shift.max(-120).min(120);
        let s1 = power(first);
        let s2 = power(shift - first);
        scales[(row * 2) as usize] = s1;
        scales[(row * 2 + 1) as usize] = s2;
        k = 0;
        while k < 64 {
            packed[row as usize * STRIDE + k as usize] = if row < rows && k < inner {
                cvt_rn_f16_f32(multiply(
                    multiply(grad[grad_index(base, row, k, flags)], s1),
                    s2,
                ))
            } else {
                0
            };
            k += 1;
        }
    }
    thread::sync_threads();
    let tile = CtaTile::from_wide_tile(tid, 0, 0, 0);
    let mut acc = [[0.0f32; 4]; 2];
    let mut start = 0;
    while start < 64 {
        let mut pair = tid * 2;
        while pair < 1024 {
            let row = pair / 16;
            let k = start + pair % 16;
            let packed_a =
                unsafe { *(packed.as_ptr().add(row as usize * STRIDE + k as usize) as *const u32) };
            let lo = if row < cols && k < inner {
                operand[(base
                    + if flags & 2 != 0 {
                        row * 64 + k
                    } else {
                        k * 64 + row
                    }) as usize]
            } else {
                0
            };
            let hi = if row < cols && k + 1 < inner {
                operand[(base
                    + if flags & 2 != 0 {
                        row * 64 + k + 1
                    } else {
                        (k + 1) * 64 + row
                    }) as usize]
            } else {
                0
            };
            store_f16x2_shared(a.as_mut_ptr(), swizzle_tc_offset(pair) as usize, packed_a);
            store_f16x2_shared(
                b.as_mut_ptr(),
                swizzle_tc_offset(pair) as usize,
                u32::from(lo) | (u32::from(hi) << 16),
            );
            pair += thread::blockDim_x() * 2;
        }
        thread::sync_threads();
        mma_accumulate(tile, a, b, &mut acc);
        thread::sync_threads();
        start += 16;
    }
    let mut repeat = 0;
    while repeat < 2 {
        let mut fragment = 0;
        while fragment < 4 {
            let (r, c) = tile.accumulator_coords(tile.warp_n0 + repeat as u32, fragment);
            let offset = base
                + if flags & 4 != 0 {
                    c * 64 + r
                } else {
                    r * 64 + c
                };
            let value = if r < rows && c < cols {
                let inv1 = inverse_power(scales[(r * 2) as usize]);
                let inv2 = inverse_power(scales[(r * 2 + 1) as usize]);
                multiply(multiply(multiply(acc[repeat][fragment], alpha), inv1), inv2)
            } else {
                0.0
            };
            unsafe {
                let destination = out.get_unchecked_mut(offset as usize);
                *destination = if flags & 8 != 0 {
                    *destination + value
                } else {
                    value
                };
            }
            fragment += 1;
        }
        repeat += 1;
    }
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn kda_exact_scaled_grad_matmul_kernel(
        grad: &[f32],
        operand: &[u16],
        out: DisjointSlice<f32>,
        flags: u32,
        alpha: f32,
        rows: u32,
        cols: u32,
        inner: u32,
    ) {
        static mut PACKED: GradTile = SharedArray::UNINIT;
        static mut SCALES: ScaleTile = SharedArray::UNINIT;
        static mut A: CtaATile = SharedArray::UNINIT;
        static mut B: CtaBTile = SharedArray::UNINIT;
        unsafe {
            scaled_grad_matmul_body(
                grad,
                operand,
                out,
                flags,
                alpha,
                rows,
                cols,
                inner,
                &mut PACKED,
                &mut SCALES,
                &mut A,
                &mut B,
            );
        }
    }
}
