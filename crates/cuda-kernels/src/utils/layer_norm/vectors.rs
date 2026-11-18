use cuda_device::{DisjointSlice, ptx_asm};
#[inline(always)]
pub fn load_f32_quad(values: &[f32], index: usize) -> [f32; 4] {
    unsafe { load_f32_quad_raw(values.as_ptr(), index) }
}
#[inline(always)]
pub unsafe fn load_f32_quad_raw(values: *const f32, index: usize) -> [f32; 4] {
    let bits: u128;
    unsafe {
        ptx_asm!(r"{.reg .b32 a,b,c,d;ld.global.nc.v4.u32 {a,b,c,d},[%1];mov.b128 %0,{a,b,c,d};}",out("=q")bits,in("l")values.add(index)as u64,options(register_only));
    }
    [
        f32::from_bits(bits as u32),
        f32::from_bits((bits >> 32) as u32),
        f32::from_bits((bits >> 64) as u32),
        f32::from_bits((bits >> 96) as u32),
    ]
}
#[inline(always)]
pub fn load_f16_quad(values: &[u16], index: usize) -> [f32; 4] {
    let bits: u64;
    unsafe {
        ptx_asm!("ld.global.nc.u64 %0,[%1];",out("=l")bits,in("l")values.as_ptr().add(index)as u64,options(register_only));
    }
    [
        crate::f16_tc_matmul::convert::cvt_f32_f16(bits as u16),
        crate::f16_tc_matmul::convert::cvt_f32_f16((bits >> 16) as u16),
        crate::f16_tc_matmul::convert::cvt_f32_f16((bits >> 32) as u16),
        crate::f16_tc_matmul::convert::cvt_f32_f16((bits >> 48) as u16),
    ]
}
#[inline(always)]
pub fn store_f32_quad(out: &mut DisjointSlice<f32>, index: usize, x: [f32; 4]) {
    unsafe {
        ptx_asm!("st.global.v4.f32 [%0],{%1,%2,%3,%4};",in("l")out.as_mut_ptr().add(index)as u64,in("f")x[0],in("f")x[1],in("f")x[2],in("f")x[3],clobber("memory"));
    }
}
#[inline(always)]
pub fn store_f16_quad(out: &mut DisjointSlice<u16>, index: usize, x: [f32; 4]) {
    let lo = cuda_device::convert::cvt_f16x2_f32(x[0], x[1]);
    let hi = cuda_device::convert::cvt_f16x2_f32(x[2], x[3]);
    let bits = u64::from(lo) | (u64::from(hi) << 32);
    unsafe {
        ptx_asm!("st.global.u64 [%0],%1;",in("l")out.as_mut_ptr().add(index)as u64,in("l")bits,clobber("memory"));
    }
}
