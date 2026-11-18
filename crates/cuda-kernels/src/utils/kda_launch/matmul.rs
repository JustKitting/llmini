use crate::f16_tc_matmul::{
    F16TcMatmulHalfCausalHalfOutArgs, F16TcMatmulKdaInverseArgs, F16TcMatmulKdaWuArgs,
};
use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::f16_tc_matmul::{
    F16TcMatmulHalfATransposedDualRhsArgs, F16TcMatmulHalfArgs, F16TcMatmulModule,
};

use super::dims::MatmulShape;
use crate::f16_tc_matmul::{F16TcMatmulF32CausalHalfOutArgs, F16TcMatmulHalfATransposedF32RhsArgs};
use crate::f16_tc_matmul::{F16TcMatmulHalfAF32RhsArgs, F16TcMatmulHalfAF32RhsHalfOutArgs};

pub(crate) struct MatmulRunner<'a> {
    stream: &'a CudaStream,
    module: &'a F16TcMatmulModule,
    batch_count: u32,
}

impl<'a> MatmulRunner<'a> {
    pub(crate) fn native_inverse(
        &self,
        kpos_u: &DeviceBuffer<f32>,
        kneg: &DeviceBuffer<u16>,
        out: &mut DeviceBuffer<u16>,
    ) -> Result<(), DriverError> {
        self.module.kda_native_inverse(F16TcMatmulKdaInverseArgs {
            stream: self.stream,
            kpos_u,
            kneg,
            out,
            batch_count: self.batch_count,
        })
    }
    pub(crate) fn half_causal_half_out(
        &self,
        a: &DeviceBuffer<u16>,
        b_t: &DeviceBuffer<u16>,
        out: &mut DeviceBuffer<u16>,
        shape: MatmulShape,
    ) -> Result<(), DriverError> {
        let MatmulShape(m, n, k) = shape;
        self.module
            .batched_matmul_half_causal_half_out(F16TcMatmulHalfCausalHalfOutArgs {
                stream: self.stream,
                a,
                b_t,
                out,
                batch_count: self.batch_count,
                m,
                n,
                k,
            })
    }

    pub(crate) fn kda_w_u(
        &self,
        inverse: &DeviceBuffer<u16>,
        vbeta: &DeviceBuffer<u16>,
        kpos_u: &mut DeviceBuffer<f32>,
        w: &mut DeviceBuffer<u16>,
    ) -> Result<(), DriverError> {
        self.module.kda_wu_inplace(F16TcMatmulKdaWuArgs {
            stream: self.stream,
            inverse,
            vbeta,
            kpos_u,
            w,
            batch_count: self.batch_count,
        })
    }

    pub(crate) fn fused_inverse(
        &self,
        a: &DeviceBuffer<f32>,
        b_t: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<u16>,
        shape: MatmulShape,
    ) -> Result<(), DriverError> {
        let MatmulShape(m, n, k) = shape;
        self.module
            .batched_kda_fused_inverse(F16TcMatmulF32CausalHalfOutArgs {
                stream: self.stream,
                a,
                b_t,
                out,
                batch_count: self.batch_count,
                m,
                n,
                k,
            })
    }
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

    pub(crate) fn half_a_f32_rhs(
        &self,
        a: &DeviceBuffer<u16>,
        rhs: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        shape: MatmulShape,
    ) -> Result<(), DriverError> {
        let MatmulShape(m, n, k) = shape;
        self.module
            .batched_matmul_half_a_f32_rhs(F16TcMatmulHalfAF32RhsArgs {
                stream: self.stream,
                a,
                rhs,
                out,
                batch_count: self.batch_count,
                m,
                n,
                k,
            })
    }
    pub(crate) fn half_a_f32_rhs_half_out(
        &self,
        a: &DeviceBuffer<u16>,
        rhs: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<u16>,
        shape: MatmulShape,
    ) -> Result<(), DriverError> {
        let MatmulShape(m, n, k) = shape;
        self.module
            .batched_matmul_half_a_f32_rhs_half_out(F16TcMatmulHalfAF32RhsHalfOutArgs {
                stream: self.stream,
                a,
                rhs,
                out,
                batch_count: self.batch_count,
                m,
                n,
                k,
            })
    }
    pub(crate) fn f32_causal_half_out(
        &self,
        a: &DeviceBuffer<f32>,
        b_t: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<u16>,
        shape: MatmulShape,
    ) -> Result<(), DriverError> {
        let MatmulShape(m, n, k) = shape;
        self.module
            .batched_matmul_f32_causal_half_out(F16TcMatmulF32CausalHalfOutArgs {
                stream: self.stream,
                a,
                b_t,
                out,
                batch_count: self.batch_count,
                m,
                n,
                k,
            })
    }

    pub(crate) fn half_a_transposed_f32_rhs(
        &self,
        a: &DeviceBuffer<u16>,
        rhs: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        shape: MatmulShape,
    ) -> Result<(), DriverError> {
        let MatmulShape(m, n, k) = shape;
        self.module
            .batched_matmul_half_a_transposed_f32_rhs(F16TcMatmulHalfATransposedF32RhsArgs {
                stream: self.stream,
                a,
                rhs,
                out,
                batch_count: self.batch_count,
                m,
                n,
                k,
            })
    }

    pub(crate) fn half_a_transposed_dual_rhs(
        &self,
        a: &DeviceBuffer<u16>,
        rhs0: &DeviceBuffer<u16>,
        rhs1: &DeviceBuffer<u16>,
        out0: &mut DeviceBuffer<f32>,
        out1: &mut DeviceBuffer<f32>,
        shape: MatmulShape,
    ) -> Result<(), DriverError> {
        let MatmulShape(m, n, k) = shape;
        self.module.batched_matmul_half_a_transposed_dual_rhs(
            F16TcMatmulHalfATransposedDualRhsArgs {
                stream: self.stream,
                a,
                rhs0,
                rhs1,
                out0,
                out1,
                batch_count: self.batch_count,
                m,
                n,
                k,
            },
        )
    }

    pub(crate) fn half_input_accumulate(
        &self,
        a: &DeviceBuffer<u16>,
        b_t: &DeviceBuffer<u16>,
        out: &mut DeviceBuffer<f32>,
        shape: MatmulShape,
    ) -> Result<(), DriverError> {
        let MatmulShape(m, n, k) = shape;
        self.module
            .batched_matmul_half_input_accumulate(F16TcMatmulHalfArgs {
                stream: self.stream,
                a,
                b_t,
                out,
                batch_count: self.batch_count,
                m,
                n,
                k,
            })
    }
}

/// Nonowning typed metadata for a scratch allocation whose previous value is
/// dead. No allocation, numeric conversion, copy or context-reference change.
pub(crate) struct BorrowedHalfScratch<'a> {
    buffer: std::mem::ManuallyDrop<DeviceBuffer<u16>>,
    _owner: std::marker::PhantomData<&'a DeviceBuffer<f32>>,
}
impl<'a> BorrowedHalfScratch<'a> {
    /// The owner remains allocated, and callers serialize all uses of its bytes
    /// on one CUDA stream. No access through the old FP32 interpretation may
    /// overlap the queued half-precision value's live interval.
    pub(crate) unsafe fn new(owner: &'a DeviceBuffer<f32>, len: usize) -> Self {
        unsafe { Self::range(owner, 0, len) }
    }
    /// The selected half-element range must be dead and disjoint from every
    /// other live view of the owner. It is consumed on the same stream.
    pub(crate) unsafe fn range(owner: &'a DeviceBuffer<f32>, offset: usize, len: usize) -> Self {
        assert!(
            offset
                .checked_add(len)
                .is_some_and(|end| end <= owner.len() * 2)
        );
        // Borrow the context handle bitwise. ManuallyDrop suppresses both the
        // allocation free and the Arc release; PhantomData enforces its life.
        let context = unsafe { std::ptr::read(owner.context()) };
        let buffer = unsafe {
            DeviceBuffer::from_raw_parts(owner.cu_deviceptr() + (offset * 2) as u64, len, context)
        };
        Self {
            buffer: std::mem::ManuallyDrop::new(buffer),
            _owner: std::marker::PhantomData,
        }
    }
}
impl std::ops::Deref for BorrowedHalfScratch<'_> {
    type Target = DeviceBuffer<u16>;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
impl std::ops::DerefMut for BorrowedHalfScratch<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}
