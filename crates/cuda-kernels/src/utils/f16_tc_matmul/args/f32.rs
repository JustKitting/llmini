use cuda_core::{CudaStream, DeviceBuffer};

macro_rules! f32_args {
    ($name:ident, $rhs:ident: $rhs_ty:ty) => {
        pub struct $name<'a, 'out> {
            pub stream: &'a CudaStream,
            pub a: &'a DeviceBuffer<f32>,
            pub $rhs: &'a DeviceBuffer<$rhs_ty>,
            pub out: &'out mut DeviceBuffer<f32>,
            pub batch_count: u32,
            pub m: u32,
            pub n: u32,
            pub k: u32,
        }
    };
}

f32_args!(F16TcMatmulF32Args, b_t: f32);

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
