use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::f16_tc_matmul::{
    F16TcMatmulF32ATransposedRhsArgs, F16TcMatmulF32Args, F16TcMatmulF32RhsArgs, F16TcMatmulModule,
};

use super::dims::MatmulShape;

pub(crate) struct MatmulRunner<'a> {
    stream: &'a CudaStream,
    module: &'a F16TcMatmulModule,
    batch_count: u32,
}

macro_rules! matmul_method {
    ($name:ident, $call:ident, $args:ident, $rhs:ident) => {
        pub(crate) fn $name(
            &self,
            a: &DeviceBuffer<f32>,
            $rhs: &DeviceBuffer<f32>,
            out: &mut DeviceBuffer<f32>,
            shape: MatmulShape,
        ) -> Result<(), DriverError> {
            let MatmulShape(m, n, k) = shape;
            self.module.$call($args {
                stream: self.stream,
                a,
                $rhs,
                out,
                batch_count: self.batch_count,
                m,
                n,
                k,
            })
        }
    };
}

impl<'a> MatmulRunner<'a> {
    pub(crate) fn new(
        stream: &'a CudaStream,
        module: &'a F16TcMatmulModule,
        batch_count: u32,
    ) -> Self {
        Self {
            stream,
            module,
            batch_count,
        }
    }

    matmul_method!(f32_input, batched_matmul_f32_input, F16TcMatmulF32Args, b_t);
    matmul_method!(
        f32_input_accumulate,
        batched_matmul_f32_input_accumulate,
        F16TcMatmulF32Args,
        b_t
    );
    matmul_method!(
        f32_input_causal,
        batched_matmul_f32_input_causal,
        F16TcMatmulF32Args,
        b_t
    );
    matmul_method!(
        f32_input_strict_causal,
        batched_matmul_f32_input_strict_causal,
        F16TcMatmulF32Args,
        b_t
    );
    matmul_method!(f32_rhs, batched_matmul_f32_rhs, F16TcMatmulF32RhsArgs, rhs);
    matmul_method!(
        f32_a_transposed_rhs,
        batched_matmul_f32_a_transposed_rhs,
        F16TcMatmulF32ATransposedRhsArgs,
        rhs
    );
    matmul_method!(
        f32_a_transposed_rhs_strict_neg,
        batched_matmul_f32_a_transposed_rhs_strict_neg,
        F16TcMatmulF32ATransposedRhsArgs,
        rhs
    );
}
