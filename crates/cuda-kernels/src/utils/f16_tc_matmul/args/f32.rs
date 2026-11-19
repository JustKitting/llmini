use cuda_core::{CudaStream, DeviceBuffer};

macro_rules! f32_args {
    ($name:ident, $rhs:ident: $rhs_ty:ty) => { f32_args!(@typed $name, f32, $rhs:$rhs_ty, f32); };
    (@typed $name:ident, $lhs_ty:ty, $rhs:ident:$rhs_ty:ty, $out_ty:ty) => {
        pub struct $name<'a, 'out> {
            pub stream: &'a CudaStream,
            pub a: &'a DeviceBuffer<$lhs_ty>,
            pub $rhs: &'a DeviceBuffer<$rhs_ty>,
            pub out: &'out mut DeviceBuffer<$out_ty>,
            pub batch_count: u32,
            pub m: u32,
            pub n: u32,
            pub k: u32,
        }
    };
}

f32_args!(F16TcMatmulF32Args, b_t: f32);
f32_args!(F16TcMatmulF32HalfArgs, b_t: u16);

pub struct F16TcMatmulF32WindowArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b_t: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub window: u32,
}

f32_args!(F16TcMatmulF32RhsArgs, rhs: f32);
f32_args!(F16TcMatmulF32HalfRhsArgs, rhs: u16);
f32_args!(F16TcMatmulF32ATransposedRhsArgs, rhs: f32);
f32_args!(F16TcMatmulF32ATransposedHalfRhsArgs, rhs: u16);

pub struct F16TcMatmulHalfATransposedDualRhsArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<u16>,
    pub rhs0: &'a DeviceBuffer<u16>,
    pub rhs1: &'a DeviceBuffer<u16>,
    pub out0: &'out mut DeviceBuffer<f32>,
    pub out1: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

f32_args!(@typed F16TcMatmulF32CausalHalfOutArgs, f32, b_t:f32, u16);
f32_args!(@typed F16TcMatmulHalfATransposedF32RhsArgs, u16, rhs:f32, f32);

f32_args!(@typed F16TcMatmulHalfAF32RhsArgs,u16,rhs:f32,f32);
f32_args!(@typed F16TcMatmulHalfAF32RhsHalfOutArgs,u16,rhs:f32,u16);

/// KDA's Kpos input is overwritten by U only after its whole chunk is read.
pub struct F16TcMatmulKdaWuArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub inverse: &'a DeviceBuffer<u16>,
    pub vbeta: &'a DeviceBuffer<u16>,
    pub kpos_u: &'out mut DeviceBuffer<f32>,
    pub w: &'out mut DeviceBuffer<u16>,
    pub batch_count: u32,
}

/// Forward Kpos is native FP16 in the first half of each FP32-U chunk arena.
pub struct F16TcMatmulKdaInverseArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub kpos_u: &'a DeviceBuffer<f32>,
    pub kneg: &'a DeviceBuffer<u16>,
    pub out: &'out mut DeviceBuffer<u16>,
    pub batch_count: u32,
}
f32_args!(@typed F16TcMatmulHalfCausalHalfOutArgs, u16, b_t:u16, u16);
