use cuda_core::{CudaStream, DeviceBuffer};

pub const F16_TC_MATMUL_K_ALIGNMENT: u32 = 16;

pub struct F16TcMatmulScratch<'a> {
    pub a_padded: &'a mut DeviceBuffer<f32>,
    pub b_t_padded: &'a mut DeviceBuffer<f32>,
    pub a_halves: &'a mut DeviceBuffer<u16>,
    pub b_t_halves: &'a mut DeviceBuffer<u16>,
}

pub struct F16ConvertArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub src: &'a DeviceBuffer<f32>,
    pub dst: &'out mut DeviceBuffer<u16>,
    pub element_count: u32,
}

pub struct F16TcMatmulArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b_t: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub scratch: F16TcMatmulScratch<'scratch>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub struct F16TcMatmulHalfArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<u16>,
    pub b_t: &'a DeviceBuffer<u16>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub struct F16TcMatmulHalfDsArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<u16>,
    pub b_t: &'a DeviceBuffer<u16>,
    pub probs: &'a DeviceBuffer<u16>,
    pub softmax_d: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<u16>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub struct F16TcMatmulHalfRhsArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<u16>,
    pub rhs: &'a DeviceBuffer<u16>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl<'a> F16TcMatmulScratch<'a> {
    pub fn reborrow(&mut self) -> F16TcMatmulScratch<'_> {
        F16TcMatmulScratch {
            a_padded: self.a_padded,
            b_t_padded: self.b_t_padded,
            a_halves: self.a_halves,
            b_t_halves: self.b_t_halves,
        }
    }
}

pub fn f16_tc_matmul_padded_k(k: u32) -> u32 {
    k.div_ceil(F16_TC_MATMUL_K_ALIGNMENT) * F16_TC_MATMUL_K_ALIGNMENT
}

pub fn f16_tc_matmul_elements(rows: u32, k: u32) -> usize {
    rows as usize * f16_tc_matmul_padded_k(k) as usize
}
