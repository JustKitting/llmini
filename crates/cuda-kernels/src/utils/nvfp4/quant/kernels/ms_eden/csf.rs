//! Producers emit the canonical compressed scales inside their original small
//! quantization tiles. No global native scale intermediate or packing pass.
use super::super::convert::{nonzero_global_scale, nonzero_scale, nvfp4_inv_scale};
use super::random::{apply_sign_mask, random_sign_mask};
use super::rounding::{covering_scale_bits, stochastic_pair};
use crate::atomic::atomic_add_f32;
use crate::float_ptx::{abs_f32, max_f32};
use crate::nvfp4_cast::e4m3_value;
use crate::warp_reduce::{half_warp_max_nonnegative_f32, warp_sum_f32};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, thread, warp};

#[inline(always)]
fn payload_pair(
    mut v0: f32,
    mut v1: f32,
    out: &mut DisjointSlice<u8>,
    c0: u32,
    c1: u32,
    g: f32,
    inv: f32,
    factor: f32,
    pair_math: bool,
    round_seed: u32,
) -> (u8, u8) {
    let lane = warp::lane_id();
    macro_rules! h {
        ($d:expr) => {{
            let p0 = warp::shuffle_xor_f32_sync(0xffffffff, v0, $d);
            let p1 = warp::shuffle_xor_f32_sync(0xffffffff, v1, $d);
            if lane & $d == 0 {
                v0 += p0;
                v1 += p1;
            } else {
                v0 = p0 - v0;
                v1 = p1 - v1;
            }
        }};
    }
    h!(1);
    h!(2);
    h!(4);
    h!(8);
    h!(16);
    v0 *= super::INV_SQRT_32;
    v1 *= super::INV_SQRT_32;
    let mask = if lane < 16 { 0xffff } else { 0xffff0000 };
    let a0 = half_warp_max_nonnegative_f32(abs_f32(v0), mask);
    let a1 = half_warp_max_nonnegative_f32(abs_f32(v1), mask);
    let f = factor * inv;
    let q0 = if pair_math {
        a0 * f
    } else {
        (a0 * factor) * inv
    };
    let q1 = if pair_math {
        a1 * f
    } else {
        (a1 * factor) * inv
    };
    let scales = u16::from(covering_scale_bits(a0, g, q0))
        | (u16::from(covering_scale_bits(a1, g, q1)) << 8);
    // Each lane owns one complete adjacent FP4 pair. Neighboring lanes
    // choose different rows, so one scale reciprocal serves both values.
    let peer0 = warp::shuffle_xor_f32_sync(0xffffffff, v0, 1);
    let peer1 = warp::shuffle_xor_f32_sync(0xffffffff, v1, 1);
    let odd = lane & 1 != 0;
    let low = if odd { peer1 } else { v0 };
    let high = if odd { v1 } else { peer0 };
    let sf = if odd { scales >> 8 } else { scales & 255 };
    let inverse = nvfp4_inv_scale(nonzero_scale(e4m3_value(sf)), g);
    let chunk = if odd { c1 } else { c0 };
    let packed = stochastic_pair(
        high * inverse,
        low * inverse,
        round_seed,
        chunk * 32 + (lane & !1),
    );
    unsafe {
        *out.get_unchecked_mut((chunk * 16 + lane / 2) as usize) = packed;
    }
    let s0 = scales as u8;
    let s1 = (scales >> 8) as u8;
    (s0, s1)
}
#[inline(always)]
fn stage_pair(scales: *mut u8, r0: u32, r1: u32, kg: u32, bits: (u8, u8)) {
    let lane = warp::lane_id();
    if lane == 0 || lane == 16 {
        unsafe {
            *scales.add((r0 * 8 + kg + lane / 16) as usize) = bits.0;
            *scales.add((r1 * 8 + kg + lane / 16) as usize) = bits.1;
        }
    }
}

// Eight adjacent values keep three butterfly axes in registers. A four-lane
// group owns one complete 32-value transform and two adjacent scale groups.
#[inline(always)]
fn payload_oct(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
    global: f32,
    quant_factor: f32,
    round_seed: u32,
    round_base: u32,
) -> (u32, u8) {
    let lane = warp::lane_id();
    let ab0 = a + b;
    let ab1 = a - b;
    let cd0 = c + d;
    let cd1 = c - d;
    let ef0 = e + f;
    let ef1 = e - f;
    let gh0 = g + h;
    let gh1 = g - h;
    let lo0 = ab0 + cd0;
    let lo1 = ab1 + cd1;
    let lo2 = ab0 - cd0;
    let lo3 = ab1 - cd1;
    let hi0 = ef0 + gh0;
    let hi1 = ef1 + gh1;
    let hi2 = ef0 - gh0;
    let hi3 = ef1 - gh1;
    let mut v0 = lo0 + hi0;
    let mut v1 = lo1 + hi1;
    let mut v2 = lo2 + hi2;
    let mut v3 = lo3 + hi3;
    let mut v4 = lo0 - hi0;
    let mut v5 = lo1 - hi1;
    let mut v6 = lo2 - hi2;
    let mut v7 = lo3 - hi3;
    macro_rules! one {
        ($value:ident,$delta:expr) => {{
            let peer = warp::shuffle_xor_f32_sync(0xffffffff, $value, $delta);
            if lane & $delta == 0 {
                $value += peer;
            } else {
                $value = peer - $value;
            }
        }};
    }
    macro_rules! h {
        ($delta:expr) => {{
            one!(v0, $delta);
            one!(v1, $delta);
            one!(v2, $delta);
            one!(v3, $delta);
            one!(v4, $delta);
            one!(v5, $delta);
            one!(v6, $delta);
            one!(v7, $delta);
        }};
    }
    h!(1);
    h!(2);
    v0 *= super::INV_SQRT_32;
    v1 *= super::INV_SQRT_32;
    v2 *= super::INV_SQRT_32;
    v3 *= super::INV_SQRT_32;
    v4 *= super::INV_SQRT_32;
    v5 *= super::INV_SQRT_32;
    v6 *= super::INV_SQRT_32;
    v7 *= super::INV_SQRT_32;
    let maximum = max_f32(
        max_f32(
            max_f32(abs_f32(v0), abs_f32(v1)),
            max_f32(abs_f32(v2), abs_f32(v3)),
        ),
        max_f32(
            max_f32(abs_f32(v4), abs_f32(v5)),
            max_f32(abs_f32(v6), abs_f32(v7)),
        ),
    );
    let amax = max_f32(maximum, warp::shuffle_xor_f32_sync(0xffffffff, maximum, 1));
    let scale = covering_scale_bits(amax, global, amax * quant_factor);
    let inverse = nvfp4_inv_scale(nonzero_scale(e4m3_value(scale as u16)), global);
    let p0 = stochastic_pair(v1 * inverse, v0 * inverse, round_seed, round_base + 0);
    let p1 = stochastic_pair(v3 * inverse, v2 * inverse, round_seed, round_base + 2);
    let p2 = stochastic_pair(v5 * inverse, v4 * inverse, round_seed, round_base + 4);
    let p3 = stochastic_pair(v7 * inverse, v6 * inverse, round_seed, round_base + 6);
    let packed = p0 as u32 | ((p1 as u32) << 8) | ((p2 as u32) << 16) | ((p3 as u32) << 24);
    (packed, scale)
}

#[inline(always)]
fn oct_signs(seed: u32, base: u32) -> u32 {
    let lane = warp::lane_id();
    let first = (lane & 3) * 8;
    let mut packed =
        (warp::ballot_sync(0xffffffff, random_sign_mask(seed, base + lane) >> 31 != 0) >> first)
            & 255;
    packed |= ((warp::ballot_sync(
        0xffffffff,
        random_sign_mask(seed, base + 32 + lane) >> 31 != 0,
    ) >> first)
        & 255)
        << 8;
    packed |= ((warp::ballot_sync(
        0xffffffff,
        random_sign_mask(seed, base + 64 + lane) >> 31 != 0,
    ) >> first)
        & 255)
        << 16;
    packed |= ((warp::ballot_sync(
        0xffffffff,
        random_sign_mask(seed, base + 96 + lane) >> 31 != 0,
    ) >> first)
        & 255)
        << 24;
    packed
}

#[inline(always)]
fn stage_oct(scales: *mut u8, row: u32, kg: u32, scale: u8) {
    let lane = warp::lane_id();
    if lane & 1 == 0 {
        unsafe {
            *scales.add((row * 8 + kg + (lane & 3) / 2) as usize) = scale;
        }
    }
}
#[inline(always)]
fn emit_warp(
    staged: *const u8,
    out: *mut u8,
    rows: u32,
    k: u32,
    rb: u32,
    kg: u32,
    local_warp: u32,
) {
    let row_local = local_warp * 32 + warp::lane_id();
    let at = row_local * 8;
    let lo = unsafe { *staged.add(at as usize).cast::<u32>() };
    let hi = unsafe { *staged.add((at + 4) as usize).cast::<u32>() };
    let row = rb + row_local;
    unsafe { crate::nvfp4::csf::write_record(out, rows, k, row, kg, lo, hi) };
}
#[inline(always)]
fn context(g: f32) -> (f32, f32) {
    let safe = nonzero_global_scale(g);
    (safe, nvfp4_inv_scale(6., safe))
}
#[inline(always)]
fn load_f32_tile(x: &[f32], tile: *mut f32, offset: usize, rb: u32, cb: u32, rows: u32, cols: u32) {
    let tid = thread::threadIdx_x();
    let mut p = tid;
    while p < 512 {
        let r = rb + p / 16;
        let c = cb + (p % 16) * 2;
        let (a, b) = if r < rows && c + 1 < cols {
            crate::f16_tc_matmul::convert::load_f32x2_global_read_only(
                x.as_ptr(),
                (r * cols + c) as usize,
            )
        } else {
            (0., 0.)
        };
        unsafe {
            *tile.add(offset + (p / 16 * 33 + (p % 16) * 2) as usize) = a;
            *tile.add(offset + (p / 16 * 33 + (p % 16) * 2 + 1) as usize) = b;
        }
        p += thread::blockDim_x();
    }
}

// A bijective 64x32 tile. Row, transpose and original bias ownership are
// bank-disjoint; each group of four consecutive columns occupies one 16B word.
#[inline(always)]
fn quad_tile_index(row: u32, col: u32) -> usize {
    let banks = ((row >> 3) & 3)
        | ((((row >> 5) ^ (row >> 2)) & 1) << 2)
        | ((row & 1) << 3)
        | ((row & 2) << 3);
    (row * 32 + (col ^ banks)) as usize
}

#[inline(always)]
fn payload_f32_oct(
    tile: *const f32,
    offset: usize,
    row: u32,
    col: u32,
    row_step: u32,
    col_step: u32,
    signs: u32,
    global: f32,
    quant_factor: f32,
    round_seed: u32,
    round_base: u32,
) -> (u32, u8) {
    macro_rules! value {
        ($index:expr) => {{
            let r = row + $index * row_step;
            let c = col + $index * col_step;
            let v = unsafe { *tile.add(offset + quad_tile_index(r, c)) };
            apply_sign_mask(v, ((signs >> $index) & 1) << 31)
        }};
    }
    payload_oct(
        value!(0),
        value!(1),
        value!(2),
        value!(3),
        value!(4),
        value!(5),
        value!(6),
        value!(7),
        global,
        quant_factor,
        round_seed,
        round_base,
    )
}
#[inline(always)]
fn load_quad_f32_tile(
    x: &[f32],
    tile: *mut f32,
    offset: usize,
    rb: u32,
    cb: u32,
    rows: u32,
    cols: u32,
) {
    let p = thread::threadIdx_x();
    let row = p / 8;
    let col = (p % 8) * 4;
    let (mut a, mut b, mut c, mut d) = (0f32, 0f32, 0f32, 0f32);
    if rb + row < rows && cb + col + 3 < cols {
        let bits: u128;
        unsafe {
            cuda_device::ptx_asm!(
                r"{ .reg .b32 a,b,c,d;
                     ld.global.nc.L2::128B.v4.u32 {a,b,c,d},[%1];
                     mov.b128 %0,{a,b,c,d}; }",
                out("=q") bits,
                in("l") x.as_ptr().add(((rb + row) * cols + cb + col) as usize) as u64,
                options(register_only),
            );
        }
        a = f32::from_bits(bits as u32);
        b = f32::from_bits((bits >> 32) as u32);
        c = f32::from_bits((bits >> 64) as u32);
        d = f32::from_bits((bits >> 96) as u32);
    }
    let index = quad_tile_index(row, col);
    if index & 1 != 0 {
        (a, b) = (b, a);
        (c, d) = (d, c);
    }
    if index & 2 != 0 {
        (a, c) = (c, a);
        (b, d) = (d, b);
    }
    unsafe {
        cuda_device::ptx_asm!(
            r"{ .reg .u64 addr;
                 cvta.to.shared.u64 addr,%0;
                 st.shared.v4.f32 [addr],{%1,%2,%3,%4}; }",
            in("l") tile.add(offset + (index & !3)) as u64,
            in("f") a, in("f") b, in("f") c, in("f") d,
        );
    }
}

#[inline(always)]
fn transpose_half_word_index(row: u32, word_col: u32) -> usize {
    let banks = ((row >> 3) & 3) | ((row & 7) << 2);
    (row * 32 + (word_col ^ banks)) as usize
}

#[inline(always)]
fn load_nvfp4_transpose_half_tile<const ROWWISE: bool>(
    bytes: &[u8],
    input_scales: &[u8],
    input_globals: &[f32],
    tile: *mut u16,
    globals: *mut f32,
    buffer: u32,
    chunk: u32,
    col_base: u32,
    rows: u32,
    cols: u32,
    storage_rows: u32,
) {
    let tid = thread::threadIdx_x();
    let local_row = tid / 8;
    let local_col = (tid % 8) * 8;
    let row = chunk * 32 + local_row;
    let col = col_base + local_col;
    if ROWWISE && tid % 8 == 0 {
        let g = if row < rows {
            unsafe { *input_globals.get_unchecked(row as usize) }
        } else {
            0.0
        };
        unsafe {
            *globals.add((buffer * 32 + local_row) as usize) = g;
        }
    }
    let halves = if row < rows && col + 7 < cols {
        let scale = unsafe {
            crate::nvfp4::csf::matrix_scale_at_data(
                input_scales.as_ptr().add(if ROWWISE {
                    0
                } else {
                    crate::nvfp4::csf::HEADER_BYTES
                }),
                storage_rows,
                if ROWWISE {
                    cols.div_ceil(128) * 128
                } else {
                    cols
                },
                row,
                col / 16,
            )
        };
        let packed: u32;
        unsafe {
            cuda_device::ptx_asm!("ld.global.nc.u32 %0,[%1];",out("=r")packed,
                in("l")bytes.as_ptr().add(((row * cols + col) / 2) as usize) as u64,
                clobber("memory"));
        }
        crate::nvfp4_cast::scaled_e2m1_half8(packed, scale)
    } else {
        0u128
    };
    let (mut a, mut b, mut c, mut d) = (
        halves as u32,
        (halves >> 32) as u32,
        (halves >> 64) as u32,
        (halves >> 96) as u32,
    );
    let index = transpose_half_word_index(local_row, local_col / 2);
    if index & 1 != 0 {
        (a, b) = (b, a);
        (c, d) = (d, c);
    }
    if index & 2 != 0 {
        (a, c) = (c, a);
        (b, d) = (d, b);
    }
    unsafe {
        cuda_device::ptx_asm!(r"{.reg .u64 dst;cvta.to.shared.u64 dst,%0;
            st.shared.v4.u32 [dst],{%1,%2,%3,%4};}",
            in("l")tile.add(buffer as usize * 2048 + (index & !3) * 2) as u64,
            in("r")a,in("r")b,in("r")c,in("r")d);
    }
}

#[inline(always)]
fn transposed_half_value<const ROWWISE: bool>(
    tile: *const u16,
    globals: *const f32,
    buffer: u32,
    row: u32,
    col: u32,
    scalar: f32,
) -> f32 {
    let half = unsafe {
        *tile.add(
            buffer as usize * 2048
                + transpose_half_word_index(row, col / 2) * 2
                + (col % 2) as usize,
        )
    };
    let g = if ROWWISE {
        unsafe { *globals.add((buffer * 32 + row) as usize) }
    } else {
        scalar
    };
    let value = crate::nvfp4_cast::f16_value(half);
    let result: f32;
    unsafe {
        // Preserve the original rounded FP32 product before the butterfly.
        cuda_device::ptx_asm!("mul.rn.f32 %0,%1,%2;",out("=f")result,in("f")value,in("f")g,options(register_only));
    }
    result
}

#[cuda_module]
pub(crate) mod module {
    use super::*;
    #[kernel]
    pub fn ms_eden_fp32_csf_kernel(
        x: &[f32],
        mut out: DisjointSlice<u8>,
        mut scales: DisjointSlice<u8>,
        global: &[f32],
        rows: u32,
        cols: u32,
        k: u32,
        seed: u32,
        round_seed: u32,
        factor: f32,
    ) {
        static mut S: SharedArray<u8, 256, 4> = SharedArray::UNINIT;
        let s = unsafe { (&mut *(&raw mut S)).as_mut_ptr() };
        let kt = thread::blockIdx_x();
        let rb = thread::blockIdx_y() * 32;
        let lane = warp::lane_id();
        let wid = thread::threadIdx_x() / 32;
        let g = global[0];
        let (safe, inv) = context(g);
        let mut cx = 0;
        while cx < 4 {
            let chunk = kt * 4 + cx;
            let sign = random_sign_mask(seed, chunk * 32 + lane);
            let mut lr = wid;
            while lr < 32 {
                let r0 = rb + lr;
                let r1 = r0 + 8;
                let col = chunk * 32 + lane;
                let v0 = if col < cols {
                    x[(r0 * cols + col) as usize]
                } else {
                    0.
                };
                let v1 = if col < cols {
                    x[(r1 * cols + col) as usize]
                } else {
                    0.
                };
                let bits = payload_pair(
                    apply_sign_mask(v0, sign),
                    apply_sign_mask(v1, sign),
                    &mut out,
                    r0 * (k / 32) + chunk,
                    r1 * (k / 32) + chunk,
                    safe,
                    inv,
                    factor,
                    false,
                    round_seed,
                );
                stage_pair(s, lr, lr + 8, cx * 2, bits);
                lr += 16;
            }
            cx += 1;
        }
        thread::sync_threads();
        if wid == 0 {
            emit_warp(s, scales.as_mut_ptr(), rows, k, rb, kt * 8, 0);
        }
    }
    #[kernel]
    pub fn ms_eden_fp32_transpose_csf_kernel(
        x: &[f32],
        mut out: DisjointSlice<u8>,
        mut scales: DisjointSlice<u8>,
        global: &[f32],
        rows: u32,
        cols: u32,
        k: u32,
        seed: u32,
        round_seed: u32,
        factor: f32,
    ) {
        static mut TILE: SharedArray<f32, { 32 * 33 }> = SharedArray::UNINIT;
        static mut S: SharedArray<u8, 256, 4> = SharedArray::UNINIT;
        let tile = unsafe { (&mut *(&raw mut TILE)).as_mut_ptr() };
        let s = unsafe { (&mut *(&raw mut S)).as_mut_ptr() };
        let kt = thread::blockIdx_x();
        let rb = thread::blockIdx_y() * 32;
        let lane = warp::lane_id();
        let wid = thread::threadIdx_x() / 32;
        let g = global[0];
        let (safe, inv) = context(g);
        let mut cx = 0;
        while cx < 4 {
            let chunk = kt * 4 + cx;
            load_f32_tile(x, tile, 0, chunk * 32, rb, rows, cols);
            thread::sync_threads();
            let sign = random_sign_mask(seed, chunk * 32 + lane);
            let mut lr = wid;
            while lr < 32 {
                let r0 = rb + lr;
                let r1 = r0 + 8;
                let v0 = unsafe { *tile.add((lane * 33 + lr) as usize) };
                let v1 = unsafe { *tile.add((lane * 33 + lr + 8) as usize) };
                let bits = payload_pair(
                    apply_sign_mask(v0, sign),
                    apply_sign_mask(v1, sign),
                    &mut out,
                    r0 * (k / 32) + chunk,
                    r1 * (k / 32) + chunk,
                    safe,
                    inv,
                    factor,
                    false,
                    round_seed,
                );
                stage_pair(s, lr, lr + 8, cx * 2, bits);
                lr += 16;
            }
            thread::sync_threads();
            cx += 1;
        }
        if wid == 0 {
            emit_warp(s, scales.as_mut_ptr(), cols, k, rb, kt * 8, 0);
        }
    }
    #[inline(always)]
    fn ms_eden_nvfp4_transpose_csf_body<const ROWWISE: bool>(
        bytes: &[u8],
        input_scales: &[u8],
        input_globals: &[f32],
        mut out: DisjointSlice<u8>,
        mut scales: DisjointSlice<u8>,
        global: &[f32],
        rows: u32,
        cols: u32,
        storage_rows: u32,
        seed: u32,
        round_seed: u32,
        factor: f32,
    ) {
        static mut TILE: SharedArray<u16, { 2 * 32 * 64 }, 16> = SharedArray::UNINIT;
        static mut G: SharedArray<f32, { 2 * 32 }> = SharedArray::UNINIT;
        static mut S: SharedArray<u8, 512, 4> = SharedArray::UNINIT;
        static mut CTX: SharedArray<f32, 2> = SharedArray::UNINIT;
        let tile = unsafe { (&mut *(&raw mut TILE)).as_mut_ptr() };
        let globals = unsafe { (&mut *(&raw mut G)).as_mut_ptr() };
        let staged = unsafe { (&mut *(&raw mut S)).as_mut_ptr() };
        let kt = thread::blockIdx_x();
        let rb = thread::blockIdx_y() * 64;
        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let wid = tid / 32;
        if tid == 0 {
            let (safe, inv) = context(global[0]);
            unsafe {
                CTX[0] = safe;
                CTX[1] = factor * inv;
            }
        }
        let input_scalar = if ROWWISE { 0.0 } else { input_globals[0] };
        let k = thread::gridDim_x() * 128;
        load_nvfp4_transpose_half_tile::<ROWWISE>(
            bytes,
            input_scales,
            input_globals,
            tile,
            globals,
            0,
            kt * 4,
            rb,
            rows,
            cols,
            storage_rows,
        );
        thread::sync_threads();
        let (safe, quant_factor) = unsafe { (CTX[0], CTX[1]) };
        let signs = oct_signs(seed, kt * 128);
        let first = (lane & 3) * 8;
        let mut cx = 0;
        while cx < 4 {
            let buffer = cx & 1;
            if cx + 1 < 4 {
                load_nvfp4_transpose_half_tile::<ROWWISE>(
                    bytes,
                    input_scales,
                    input_globals,
                    tile,
                    globals,
                    (cx + 1) & 1,
                    kt * 4 + cx + 1,
                    rb,
                    rows,
                    cols,
                    storage_rows,
                );
            }
            {
                let local_row = wid + (lane / 4) * 8;
                let v0 = transposed_half_value::<ROWWISE>(
                    tile,
                    globals,
                    buffer,
                    first,
                    local_row,
                    input_scalar,
                );
                let v1 = transposed_half_value::<ROWWISE>(
                    tile,
                    globals,
                    buffer,
                    first + 1,
                    local_row,
                    input_scalar,
                );
                let v2 = transposed_half_value::<ROWWISE>(
                    tile,
                    globals,
                    buffer,
                    first + 2,
                    local_row,
                    input_scalar,
                );
                let v3 = transposed_half_value::<ROWWISE>(
                    tile,
                    globals,
                    buffer,
                    first + 3,
                    local_row,
                    input_scalar,
                );
                let v4 = transposed_half_value::<ROWWISE>(
                    tile,
                    globals,
                    buffer,
                    first + 4,
                    local_row,
                    input_scalar,
                );
                let v5 = transposed_half_value::<ROWWISE>(
                    tile,
                    globals,
                    buffer,
                    first + 5,
                    local_row,
                    input_scalar,
                );
                let v6 = transposed_half_value::<ROWWISE>(
                    tile,
                    globals,
                    buffer,
                    first + 6,
                    local_row,
                    input_scalar,
                );
                let v7 = transposed_half_value::<ROWWISE>(
                    tile,
                    globals,
                    buffer,
                    first + 7,
                    local_row,
                    input_scalar,
                );
                let chunk = (rb + local_row) * (k / 32) + kt * 4 + cx;
                let s = signs >> (cx * 8);
                let (packed, bits) = payload_oct(
                    apply_sign_mask(v0, (s & 1) << 31),
                    apply_sign_mask(v1, ((s >> 1) & 1) << 31),
                    apply_sign_mask(v2, ((s >> 2) & 1) << 31),
                    apply_sign_mask(v3, ((s >> 3) & 1) << 31),
                    apply_sign_mask(v4, ((s >> 4) & 1) << 31),
                    apply_sign_mask(v5, ((s >> 5) & 1) << 31),
                    apply_sign_mask(v6, ((s >> 6) & 1) << 31),
                    apply_sign_mask(v7, ((s >> 7) & 1) << 31),
                    safe,
                    quant_factor,
                    round_seed,
                    chunk * 32 + (lane & 3) * 8,
                );
                unsafe {
                    *out.as_mut_ptr()
                        .add((chunk * 16 + (lane & 3) * 4) as usize)
                        .cast::<u32>() = packed;
                }
                stage_oct(staged, local_row, cx * 2, bits);
            }
            thread::sync_threads();
            cx += 1;
        }
        if wid < 2 {
            emit_warp(staged, scales.as_mut_ptr(), cols, k, rb, kt * 8, wid);
        }
    }
    #[kernel]
    #[launch_bounds(256, 6)]
    pub fn ms_eden_nvfp4_transpose_csf_kernel(
        bytes: &[u8],
        input_scales: &[u8],
        input_globals: &[f32],
        out: DisjointSlice<u8>,
        scales: DisjointSlice<u8>,
        global: &[f32],
        rows: u32,
        cols: u32,
        storage_rows: u32,
        seed: u32,
        round_seed: u32,
        factor: f32,
    ) {
        ms_eden_nvfp4_transpose_csf_body::<false>(
            bytes,
            input_scales,
            input_globals,
            out,
            scales,
            global,
            rows,
            cols,
            storage_rows,
            seed,
            round_seed,
            factor,
        );
    }
    #[kernel]
    #[launch_bounds(256, 6)]
    pub fn ms_eden_rowwise_nvfp4_transpose_csf_kernel(
        bytes: &[u8],
        input_scales: &[u8],
        input_globals: &[f32],
        out: DisjointSlice<u8>,
        scales: DisjointSlice<u8>,
        global: &[f32],
        rows: u32,
        cols: u32,
        storage_rows: u32,
        seed: u32,
        round_seed: u32,
        factor: f32,
    ) {
        ms_eden_nvfp4_transpose_csf_body::<true>(
            bytes,
            input_scales,
            input_globals,
            out,
            scales,
            global,
            rows,
            cols,
            storage_rows,
            seed,
            round_seed,
            factor,
        );
    }
    #[kernel]
    #[launch_bounds(512, 3)]
    pub fn ms_eden_fp32_pair_csf_kernel(
        x: &[f32],
        mut out: DisjointSlice<u8>,
        mut scales: DisjointSlice<u8>,
        mut trans_out: DisjointSlice<u8>,
        mut trans_scales: DisjointSlice<u8>,
        mut bias: DisjointSlice<f32>,
        global: &[f32],
        rows: u32,
        cols: u32,
        k: u32,
        trans_k: u32,
        seed: u32,
        round_seed: u32,
        transpose_round_seed: u32,
        factor: f32,
    ) {
        static mut TILE: SharedArray<f32, { 2 * 64 * 32 }, 16> = SharedArray::UNINIT;
        static mut S: SharedArray<u8, 2048, 4> = SharedArray::UNINIT;
        static mut CTX: SharedArray<f32, 2> = SharedArray::UNINIT;
        let tile = unsafe { (&mut *(&raw mut TILE)).as_mut_ptr() };
        let staged = unsafe { (&mut *(&raw mut S)).as_mut_ptr() };
        let row_tiles = rows / 128;
        let part = thread::blockIdx_x() % row_tiles;
        let ct = thread::blockIdx_x() / row_tiles;
        let tid = thread::threadIdx_x();
        let wid = tid / 32;
        let lane = warp::lane_id();
        if tid == 0 {
            let (safe, inv) = context(global[0]);
            unsafe {
                CTX[0] = safe;
                CTX[1] = inv;
            }
        }
        load_quad_f32_tile(x, tile, 0, part * 128, ct * 128, rows, cols);
        thread::sync_threads();
        let (safe, inv) = unsafe { (CTX[0], CTX[1]) };
        let quant_factor = factor * inv;
        let (mut bias0, mut bias1) = (0f32, 0f32);
        let sign_base = if wid < 8 { ct * 128 } else { part * 128 };
        let signs = oct_signs(seed, sign_base);

        let mut work = 0;
        while work < 8 {
            let rx = work % 2;
            let cx = work / 2;
            let rb = part * 128 + rx * 64;
            let cb = ct * 128 + cx * 32;
            let off = (work as usize & 1) * 64 * 32;
            if work + 1 < 8 {
                let next = work + 1;
                let nrow = part * 128 + (next % 2) * 64;
                let ncol = ct * 128 + (next / 2) * 32;
                load_quad_f32_tile(
                    x,
                    tile,
                    ((next as usize) & 1) * 64 * 32,
                    nrow,
                    ncol,
                    rows,
                    cols,
                );
            }
            let first = (lane & 3) * 8;
            let transpose_warp = wid & 7;
            let is_transpose = wid >= 8;
            let local_row = wid + (lane / 4) * 8;
            let transpose_row = (transpose_warp & 3) + (lane / 4) * 4;
            let source_row = if is_transpose {
                first + (transpose_warp / 4) * 32
            } else {
                local_row
            };
            let source_col = if is_transpose { transpose_row } else { first };
            let row_step = is_transpose as u32;
            let col_step = 1 - row_step;
            let sign_chunk = if is_transpose {
                rx * 2 + transpose_warp / 4
            } else {
                cx
            };
            let round_chunk = if is_transpose {
                (cb + transpose_row) * (trans_k / 32) + rb / 32 + transpose_warp / 4
            } else {
                (rb + local_row) * (k / 32) + cb / 32
            };
            let (packed, bits) = payload_f32_oct(
                tile,
                off,
                source_row,
                source_col,
                row_step,
                col_step,
                signs >> (sign_chunk * 8),
                safe,
                quant_factor,
                if is_transpose {
                    transpose_round_seed
                } else {
                    round_seed
                },
                round_chunk * 32 + (lane & 3) * 8,
            );
            if !is_transpose {
                let chunk = (rb + local_row) * (k / 32) + cb / 32;
                unsafe {
                    *out.as_mut_ptr()
                        .add((chunk * 16 + (lane & 3) * 4) as usize)
                        .cast::<u32>() = packed;
                }
                stage_oct(staged, rx * 64 + local_row, cx * 2, bits);
            } else {
                let chunk = (cb + transpose_row) * (trans_k / 32) + rb / 32 + transpose_warp / 4;
                unsafe {
                    *trans_out
                        .as_mut_ptr()
                        .add((chunk * 16 + (lane & 3) * 4) as usize)
                        .cast::<u32>() = packed;
                }
                stage_oct(
                    unsafe { staged.add(1024) },
                    cx * 32 + transpose_row,
                    rx * 4 + (transpose_warp / 4) * 2,
                    bits,
                );
            }
            // Preserve the original lane and sequential32-row additions.
            bias0 += unsafe { *tile.add(off + quad_tile_index(lane, wid)) };
            bias1 += unsafe { *tile.add(off + quad_tile_index(lane, wid + 16)) };
            bias0 += unsafe { *tile.add(off + quad_tile_index(lane + 32, wid)) };
            bias1 += unsafe { *tile.add(off + quad_tile_index(lane + 32, wid + 16)) };
            thread::sync_threads();
            if work % 2 == 1 {
                if !bias.is_empty() {
                    let a = warp_sum_f32(bias0);
                    let b = warp_sum_f32(bias1);
                    if lane == 0 {
                        unsafe {
                            atomic_add_f32(
                                bias.as_mut_ptr().add((ct * 128 + cx * 32 + wid) as usize),
                                a,
                            );
                            atomic_add_f32(
                                bias.as_mut_ptr()
                                    .add((ct * 128 + cx * 32 + wid + 16) as usize),
                                b,
                            );
                        }
                    }
                }
                bias0 = 0.;
                bias1 = 0.;
            }
            if work == 7 && wid < 8 {
                if wid < 4 {
                    emit_warp(
                        staged,
                        scales.as_mut_ptr(),
                        rows,
                        k,
                        part * 128,
                        ct * 8,
                        wid,
                    );
                } else {
                    emit_warp(
                        unsafe { staged.add(1024) },
                        trans_scales.as_mut_ptr(),
                        cols,
                        trans_k,
                        ct * 128,
                        part * 8,
                        wid - 4,
                    );
                }
            }
            work += 1;
        }
    }
}
