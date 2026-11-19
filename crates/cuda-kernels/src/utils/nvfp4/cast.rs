use cuda_device::ptx_asm;

#[inline(always)]
pub fn e2m1_value(bits: u8) -> f32 {
    let value: f32;
    let packed = bits as u16;

    unsafe {
        ptx_asm!(
            "{ .reg .b8 e2; .reg .b32 h2; .reg .b16 lo; cvt.u8.u16 e2, %1; cvt.rn.f16x2.e2m1x2 h2, e2; cvt.u16.u32 lo, h2; cvt.f32.f16 %0, lo; }",
            out("=f") value,
            in("h") packed,
            options(register_only),
        );
    }
    value
}

#[inline(always)]
pub fn e2m1_values(bits: u8) -> (f32, f32) {
    let halfs: u32;
    let packed = bits as u16;

    unsafe {
        ptx_asm!(
            "{ .reg .b8 e2; cvt.u8.u16 e2, %1; cvt.rn.f16x2.e2m1x2 %0, e2; }",
            out("=r") halfs,
            in("h") packed,
            options(register_only),
        );
    }
    (f16_value(halfs as u16), f16_value((halfs >> 16) as u16))
}

#[inline(always)]
pub(crate) fn f16_value(bits: u16) -> f32 {
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
pub fn e4m3_value(bits: u16) -> f32 {
    let value: f32;

    unsafe {
        ptx_asm!(
            "{ .reg .b32 h2; .reg .b16 lo; cvt.rn.f16x2.e4m3x2 h2, %1; cvt.u16.u32 lo, h2; cvt.f32.f16 %0, lo; }",
            out("=f") value,
            in("h") bits,
            options(register_only),
        );
    }
    value
}

#[inline(always)]
pub(crate) fn e4m3_values(bits: u16) -> (f32, f32) {
    let halves: u32;
    unsafe {
        ptx_asm!("cvt.rn.f16x2.e4m3x2 %0,%1;",out("=r")halves,in("h")bits,options(register_only));
    }
    (f16_value(halves as u16), f16_value((halves >> 16) as u16))
}

/// E2M1 times E4M3 has at most six significant bits and fits binary16's
/// exponent range. This packed product is exact before the FP32 global scale.
#[inline(always)]
pub(crate) fn scaled_e2m1_half2(bits: u8, scale: u8) -> u32 {
    let out: u32;
    let scales = (scale as u16) * 257;
    unsafe {
        ptx_asm!(r"{
        .reg .b8 x;
        .reg .b32 p,s;
        cvt.u8.u16 x,%1;
        cvt.rn.f16x2.e2m1x2 p,x;
        cvt.rn.f16x2.e4m3x2 s,%2;
        mul.f16x2 %0,p,s;
    }",out("=r")out,in("h")bits as u16,in("h")scales,options(register_only));
    }
    out
}

#[inline(always)]
pub(crate) fn scaled_e2m1_half4(bits: u16, scale: u8) -> u64 {
    let result: u64;
    let scales = u16::from(scale) * 257;
    unsafe {
        ptx_asm!(r"{
        .reg .b8 x0,x1;
        .reg .b32 hi;
        .reg .b32 p0,p1,s;
        cvt.u8.u16 x0,%1;cvt.u32.u16 hi,%1;shr.b32 hi,hi,8;cvt.u8.u32 x1,hi;
        cvt.rn.f16x2.e2m1x2 p0,x0;cvt.rn.f16x2.e2m1x2 p1,x1;
        cvt.rn.f16x2.e4m3x2 s,%2;
        mul.f16x2 p0,p0,s;mul.f16x2 p1,p1,s;
        mov.b64 %0,{p0,p1};
    }",out("=l")result,in("h")bits,in("h")scales,options(register_only));
    }
    result
}

#[inline(always)]
pub(crate) fn scaled_e2m1_half8(bits: u32, scale: u8) -> u128 {
    let result: u128;
    let scales = u16::from(scale) * 257;
    unsafe {
        ptx_asm!(r"{
        .reg .b8 x0,x1,x2,x3;
        .reg .b32 t,p0,p1,p2,p3,s;
        cvt.u8.u32 x0,%1;shr.b32 t,%1,8;cvt.u8.u32 x1,t;
        shr.b32 t,%1,16;cvt.u8.u32 x2,t;shr.b32 t,%1,24;cvt.u8.u32 x3,t;
        cvt.rn.f16x2.e4m3x2 s,%2;
        cvt.rn.f16x2.e2m1x2 p0,x0;cvt.rn.f16x2.e2m1x2 p1,x1;
        cvt.rn.f16x2.e2m1x2 p2,x2;cvt.rn.f16x2.e2m1x2 p3,x3;
        mul.f16x2 p0,p0,s;mul.f16x2 p1,p1,s;mul.f16x2 p2,p2,s;mul.f16x2 p3,p3,s;
        mov.b128 %0,{p0,p1,p2,p3};
    }",out("=q")result,in("r")bits,in("h")scales,options(register_only));
    }
    result
}
