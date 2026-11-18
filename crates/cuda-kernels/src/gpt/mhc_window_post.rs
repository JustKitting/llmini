//! Reconstruct three mHC tape inputs while retaining each mix in registers.
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use std::sync::Arc;

pub struct MhcWindowPostModule {
    kernels: kernels::LoadedModule,
}

pub struct PostWindowArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub branches: [&'a DeviceBuffer<f32>; 3],
    pub coefficients: [&'a DeviceBuffer<f32>; 3],
    pub out: [&'out mut DeviceBuffer<f32>; 3],
    pub rows: u32,
    pub dim: u32,
}

impl MhcWindowPostModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }

    pub fn rebuild(&self, args: PostWindowArgs<'_, '_>) -> Result<(), DriverError> {
        let features = args.rows.checked_mul(args.dim).ok_or(DriverError(1))?;
        let stream_values = features.checked_mul(4).ok_or(DriverError(1))? as usize;
        let coefficients = args.rows.checked_mul(24).ok_or(DriverError(1))? as usize;
        if features == 0
            || args.input.len() < stream_values
            || args.out.iter().any(|out| out.len() < stream_values)
            || args
                .branches
                .iter()
                .any(|branch| branch.len() < features as usize)
            || args.coefficients.iter().any(|c| c.len() < coefficients)
        {
            return Err(DriverError(1));
        }
        let [first, second, third] = args.out;
        self.kernels.mhc_rebuild_window_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (features.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            args.input,
            args.branches[0],
            args.branches[1],
            args.branches[2],
            args.coefficients[0],
            args.coefficients[1],
            args.coefficients[2],
            first,
            second,
            third,
            args.rows,
            args.dim,
        )
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[inline(always)]
    fn mix_output(
        coefficients: &[f32],
        offset: usize,
        branch: f32,
        [x0, x1, x2, x3]: [f32; 4],
    ) -> f32 {
        // Keep the original post map's branch product and accumulation order.
        let mut value = coefficients[offset + 4] * branch;
        value += coefficients[offset + 8] * x0;
        value += coefficients[offset + 12] * x1;
        value += coefficients[offset + 16] * x2;
        value += coefficients[offset + 20] * x3;
        value
    }

    #[inline(always)]
    fn mix(c: &[f32], offset: usize, branch: f32, input: [f32; 4]) -> [f32; 4] {
        [
            mix_output(c, offset, branch, input),
            mix_output(c, offset + 1, branch, input),
            mix_output(c, offset + 2, branch, input),
            mix_output(c, offset + 3, branch, input),
        ]
    }

    #[inline(always)]
    fn write_tape(
        out: &mut DisjointSlice<f32>,
        base: usize,
        dim: usize,
        [x0, x1, x2, x3]: [f32; 4],
    ) {
        unsafe {
            *out.get_unchecked_mut(base) = x0;
            *out.get_unchecked_mut(base + dim) = x1;
            *out.get_unchecked_mut(base + 2 * dim) = x2;
            *out.get_unchecked_mut(base + 3 * dim) = x3;
        }
    }

    #[kernel]
    pub fn mhc_rebuild_window_kernel(
        input: &[f32],
        branch0: &[f32],
        branch1: &[f32],
        branch2: &[f32],
        coefficients0: &[f32],
        coefficients1: &[f32],
        coefficients2: &[f32],
        mut first: DisjointSlice<f32>,
        mut second: DisjointSlice<f32>,
        mut third: DisjointSlice<f32>,
        rows: u32,
        dim: u32,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i >= rows * dim {
            return;
        }
        let row = i / dim;
        let base = (row * 4 * dim + i % dim) as usize;
        let offset = (row * 24) as usize;
        let dim = dim as usize;
        let i = i as usize;
        let x = [
            input[base],
            input[base + dim],
            input[base + 2 * dim],
            input[base + 3 * dim],
        ];
        let y0 = mix(coefficients0, offset, branch0[i], x);
        write_tape(&mut first, base, dim, y0);
        let y1 = mix(coefficients1, offset, branch1[i], y0);
        write_tape(&mut second, base, dim, y1);
        let y2 = mix(coefficients2, offset, branch2[i], y1);
        write_tape(&mut third, base, dim, y2);
    }
}
