use cuda_core::{CudaStream, DriverError};

use super::QuantScratch;

macro_rules! assert_buffer_fields_eq {
    ($stream:expr, $actual:expr, $expected:expr, [$($field:ident),+ $(,)?]) => {{
        $(assert_eq!(
            $actual.$field.to_host_vec($stream)?,
            $expected.$field.to_host_vec($stream)?
        );)+
        Ok(())
    }};
}

impl QuantScratch {
    pub(crate) fn assert_payload_eq(
        &self,
        stream: &CudaStream,
        expected: &Self,
    ) -> Result<(), DriverError> {
        assert_buffer_fields_eq!(stream, self, expected, [bytes, scales, global_scales])
    }

    pub(crate) fn assert_ms_eden_eq(
        &self,
        stream: &CudaStream,
        expected: &Self,
    ) -> Result<(), DriverError> {
        assert_buffer_fields_eq!(
            stream,
            self,
            expected,
            [bytes, scales, global_scales, chunk_amax]
        )
    }

    pub(crate) fn assert_quartet_eq(
        &self,
        stream: &CudaStream,
        expected: &Self,
    ) -> Result<(), DriverError> {
        self.assert_ms_eden_eq(stream, expected)?;
        assert_buffer_fields_eq!(stream, self, expected, [global_scale])
    }

    pub(crate) fn assert_no_chunk_quartet_eq(
        &self,
        stream: &CudaStream,
        expected: &Self,
    ) -> Result<(), DriverError> {
        assert_buffer_fields_eq!(
            stream,
            self,
            expected,
            [bytes, scales, global_scales, global_scale]
        )
    }
}
