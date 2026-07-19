use cuda_device::{
    DisjointSlice, convert::cvt_f16x2_f32, ptx_asm, tcgen05::cvt_f32x2_bf16x2, thread,
};

use super::kernels::F16_THREADS_PER_BLOCK;

pub(super) fn fp32_to_f16_body(src: &[f32], mut dst: DisjointSlice<u16>, element_count: u32) {
    let pair = (thread::blockIdx_x() * F16_THREADS_PER_BLOCK + thread::threadIdx_x()) * 2;
    if pair + 1 < element_count {
        let packed = cvt_f16x2_f32(src[pair as usize], src[pair as usize + 1]);
        unsafe {
            *dst.get_unchecked_mut(pair as usize) = (packed & 0xffff) as u16;
            *dst.get_unchecked_mut(pair as usize + 1) = (packed >> 16) as u16;
        }
    } else if pair < element_count {
        unsafe {
            *dst.get_unchecked_mut(pair as usize) = cvt_rn_f16_f32(src[pair as usize]);
        }
    }
}

#[inline(always)]
pub(crate) fn cvt_rn_f16_f32(value: f32) -> u16 {
    (cvt_f16x2_f32(value, 0.0) & 0xffff) as u16
}

#[inline(always)]
pub(crate) fn cvt_f32_f16(bits: u16) -> f32 {
    let value: f32;
    unsafe {
        ptx_asm!(
            "cvt.f32.f16 %0, %1;",
            out("=f") value,
            in("h") bits,
            options(register_only),
        );
    }
    value
}

#[inline(always)]
pub(crate) fn cvt_f32_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
pub(crate) fn load_f16x2_global_bits(src: *const u16, index: usize) -> u32 {
    let packed: u32;
    unsafe {
        ptx_asm!(
            "ld.global.u32 %0, [%1];",
            out("=r") packed,
            in("l") src.add(index) as u64,
            options(register_only),
        );
    }
    packed
}

#[inline(always)]
pub(crate) fn load_f16_global_bits_read_only(src: *const u16, index: usize) -> u16 {
    let bits: u16;
    unsafe {
        ptx_asm!(
            "ld.global.nc.L2::128B.u16 %0, [%1];",
            out("=h") bits,
            in("l") src.add(index) as u64,
            options(register_only),
        );
    }
    bits
}

#[inline(always)]
pub(crate) fn load_f16x2_global_bits_read_only(src: *const u16, index: usize) -> u32 {
    let packed: u32;
    unsafe {
        ptx_asm!(
            "ld.global.nc.L2::128B.u32 %0, [%1];",
            out("=r") packed,
            in("l") src.add(index) as u64,
            options(register_only),
        );
    }
    packed
}

#[inline(always)]
pub(crate) fn load_f16x2_global(src: *const u16, index: usize) -> (f32, f32) {
    let packed = load_f16x2_global_bits(src, index);
    (
        cvt_f32_f16(packed as u16),
        cvt_f32_f16((packed >> 16) as u16),
    )
}

#[inline(always)]
pub(crate) fn load_bf16_global(src: *const u16, index: usize) -> f32 {
    let bits = unsafe { *src.add(index) };
    cvt_f32_bf16(bits)
}

#[inline(always)]
pub(crate) fn load_bf16x2_global(src: *const u16, index: usize) -> (f32, f32) {
    let packed = load_f16x2_global_bits(src, index);
    (
        cvt_f32_bf16(packed as u16),
        cvt_f32_bf16((packed >> 16) as u16),
    )
}

#[inline(always)]
pub(crate) fn load_f32x2_global(src: *const f32, index: usize) -> (f32, f32) {
    let packed: u64;
    unsafe {
        ptx_asm!(
            "ld.global.u64 %0, [%1];",
            out("=l") packed,
            in("l") src.add(index) as u64,
            options(register_only),
        );
    }
    (
        f32::from_bits(packed as u32),
        f32::from_bits((packed >> 32) as u32),
    )
}

#[inline(always)]
pub(crate) fn load_f32_global_read_only(src: *const f32, index: usize) -> f32 {
    let bits: u32;
    unsafe {
        ptx_asm!(
            "ld.global.nc.L2::128B.u32 %0, [%1];",
            out("=r") bits,
            in("l") src.add(index) as u64,
            options(register_only),
        );
    }
    f32::from_bits(bits)
}

#[inline(always)]
pub(crate) fn load_f32x2_global_read_only(src: *const f32, index: usize) -> (f32, f32) {
    let packed: u64;
    unsafe {
        ptx_asm!(
            "ld.global.nc.L2::128B.u64 %0, [%1];",
            out("=l") packed,
            in("l") src.add(index) as u64,
            options(register_only),
        );
    }
    (
        f32::from_bits(packed as u32),
        f32::from_bits((packed >> 32) as u32),
    )
}

#[inline(always)]
pub(crate) fn store_f32x2_global(dst: *mut f32, index: usize, lo: f32, hi: f32) {
    unsafe {
        ptx_asm!(
            "st.global.v2.f32 [%0], {%1, %2};",
            in("l") dst.add(index) as u64,
            in("f") lo,
            in("f") hi,
        );
    }
}

#[inline(always)]
pub(crate) fn store_bf16_global(dst: *mut u16, index: usize, value: f32) {
    let packed = cvt_f32x2_bf16x2(value, 0.0);
    unsafe {
        *dst.add(index) = packed as u16;
    }
}

#[inline(always)]
pub(crate) fn store_bf16x2_global(dst: *mut u16, index: usize, lo: f32, hi: f32) {
    let packed = cvt_f32x2_bf16x2(lo, hi);
    unsafe {
        ptx_asm!(
            "st.global.u32 [%0], %1;",
            in("l") dst.add(index) as u64,
            in("r") packed,
        );
    }
}

#[inline(always)]
pub(crate) fn store_f16x2_global(dst: *mut u16, index: usize, packed: u32) {
    unsafe {
        ptx_asm!(
            "st.global.u32 [%0], %1;",
            in("l") dst.add(index) as u64,
            in("r") packed,
        );
    }
}

#[inline(always)]
pub(crate) fn load_f32x2_shared(src: *const f32, index: usize) -> (f32, f32) {
    let packed = unsafe { *(src.add(index) as *const u64) };
    (
        f32::from_bits(packed as u32),
        f32::from_bits((packed >> 32) as u32),
    )
}

#[inline(always)]
pub(crate) fn store_f32x2_shared(dst: *mut f32, index: usize, lo: f32, hi: f32) {
    let packed = lo.to_bits() as u64 | ((hi.to_bits() as u64) << 32);
    unsafe {
        *(dst.add(index) as *mut u64) = packed;
    }
}

#[inline(always)]
pub(crate) fn store_f16x2_shared(dst: *mut u16, index: usize, packed: u32) {
    unsafe {
        *(dst.add(index) as *mut u32) = packed;
    }
}
