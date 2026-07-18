use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use crate::types::RowwiseNvfp4Tape;

pub struct MlpForwardTape<'scratch> {
    pub up_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
    pub pre_activation_f16: &'scratch mut DeviceBuffer<u16>,
    pub down_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
    pub route_masks: &'scratch mut DeviceBuffer<u64>,
}

impl<'scratch> MlpForwardTape<'scratch> {
    pub(crate) fn save_up_input(
        &mut self,
        stream: &CudaStream,
        input: Nvfp4RowwiseDeviceTensor<'_>,
    ) -> Result<(), DriverError> {
        self.up_input_nvfp4.save(stream, input)
    }

    pub(crate) fn save_down_input(
        &mut self,
        stream: &CudaStream,
        input: Nvfp4RowwiseDeviceTensor<'_>,
    ) -> Result<(), DriverError> {
        self.down_input_nvfp4.save(stream, input)
    }
}
