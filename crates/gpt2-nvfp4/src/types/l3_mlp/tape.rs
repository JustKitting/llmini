use cuda_core::DeviceBuffer;

use crate::types::RowwiseNvfp4Tape;

pub struct MlpForwardTape<'scratch> {
    pub up_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
    pub pre_activation_f16: &'scratch mut DeviceBuffer<u16>,
    pub down_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
    pub route_masks: &'scratch mut DeviceBuffer<u64>,
}
