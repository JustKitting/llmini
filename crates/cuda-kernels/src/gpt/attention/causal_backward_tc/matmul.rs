#![expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]

use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::f16_tc_matmul::{
    F16TcMatmulHalfArgs, F16TcMatmulHalfDsArgs, F16TcMatmulHalfDsWindowArgs,
    F16TcMatmulHalfRhsArgs, F16TcMatmulHalfRhsWindowArgs, F16TcMatmulModule,
};

pub(super) struct AttentionTcMatmulContext<'a> {
    pub stream: &'a CudaStream,
    pub tc_module: &'a F16TcMatmulModule,
    pub batch_head: u32,
    pub seq_len: u32,
    pub head_dim: u32,
    pub attention_window: u32,
}

pub(super) fn run_tc_matmul(
    stream: &CudaStream,
    tc_module: &F16TcMatmulModule,
    a: &DeviceBuffer<u16>,
    b_t: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<f32>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DriverError> {
    tc_module.batched_matmul_half_input_lower(F16TcMatmulHalfArgs {
        stream,
        a,
        b_t,
        out,
        batch_count,
        m,
        n,
        k,
    })
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn run_tc_matmul_lower_ds(
    stream: &CudaStream,
    tc_module: &F16TcMatmulModule,
    a: &DeviceBuffer<u16>,
    b_t: &DeviceBuffer<u16>,
    probs: &DeviceBuffer<u16>,
    softmax_d: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<u16>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
    attention_window: u32,
) -> Result<(), DriverError> {
    if attention_window == m {
        tc_module.batched_matmul_half_input_lower_ds(F16TcMatmulHalfDsArgs {
            stream,
            a,
            b_t,
            probs,
            softmax_d,
            out,
            batch_count,
            m,
            n,
            k,
        })
    } else {
        tc_module.batched_matmul_half_input_windowed_lower_ds(F16TcMatmulHalfDsWindowArgs {
            stream,
            a,
            b_t,
            probs,
            softmax_d,
            out,
            batch_count,
            m,
            n,
            k,
            window: attention_window,
        })
    }
}

pub(super) fn run_tc_matmul_rhs(
    stream: &CudaStream,
    tc_module: &F16TcMatmulModule,
    a: &DeviceBuffer<u16>,
    rhs: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<f32>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
    attention_window: u32,
) -> Result<(), DriverError> {
    if attention_window == k {
        tc_module.batched_matmul_half_rhs_lower_a(F16TcMatmulHalfRhsArgs {
            stream,
            a,
            rhs,
            out,
            batch_count,
            m,
            n,
            k,
        })
    } else {
        tc_module.batched_matmul_half_rhs_windowed_lower_a(F16TcMatmulHalfRhsWindowArgs {
            stream,
            a,
            rhs,
            out,
            batch_count,
            m,
            n,
            k,
            window: attention_window,
        })
    }
}

pub(super) fn run_tc_matmul_a_transposed_rhs(
    stream: &CudaStream,
    tc_module: &F16TcMatmulModule,
    a: &DeviceBuffer<u16>,
    rhs: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<f32>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
    attention_window: u32,
) -> Result<(), DriverError> {
    if attention_window == k {
        tc_module.batched_matmul_half_a_transposed_rhs_lower_a(F16TcMatmulHalfRhsArgs {
            stream,
            a,
            rhs,
            out,
            batch_count,
            m,
            n,
            k,
        })
    } else {
        tc_module.batched_matmul_half_a_transposed_rhs_windowed_lower_a(
            F16TcMatmulHalfRhsWindowArgs {
                stream,
                a,
                rhs,
                out,
                batch_count,
                m,
                n,
                k,
                window: attention_window,
            },
        )
    }
}
