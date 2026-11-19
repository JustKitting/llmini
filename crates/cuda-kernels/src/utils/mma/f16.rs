#[inline(always)]
pub fn mma_m16n8k16_f16_f16_f32(a: [u32; 4], b: [u32; 2], acc: &mut [f32; 4]) {
    unsafe {
        let result: (f32, f32, f32, f32) = cuda_device::ptx::__ptx_asm_out_10(
            b"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
              {$0, $1, $2, $3},
              {$4, $5, $6, $7},
              {$8, $9},
              {$10, $11, $12, $13};",
            b"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
            b"register_only",
            a[0],
            a[1],
            a[2],
            a[3],
            b[0],
            b[1],
            acc[0],
            acc[1],
            acc[2],
            acc[3],
        );
        acc[0] = result.0;
        acc[1] = result.1;
        acc[2] = result.2;
        acc[3] = result.3;
    }
}
