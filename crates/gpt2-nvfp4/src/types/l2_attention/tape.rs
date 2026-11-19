use cuda_core::DeviceBuffer;

use crate::types::RowwiseNvfp4Tape;

pub struct AttentionForwardTape<'scratch> {
    pub qkv_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
    pub qkv_f16: &'scratch mut DeviceBuffer<u16>,
    pub attention_out_f16: &'scratch mut DeviceBuffer<u16>,
    pub headwise_gate_input_f16: Option<&'scratch mut DeviceBuffer<u16>>,
    pub attention_probs_f16: Option<&'scratch mut DeviceBuffer<u16>>,
    pub kda_v_new: Option<&'scratch mut DeviceBuffer<u16>>,
    pub kda_akk_inv: Option<&'scratch mut DeviceBuffer<u16>>,
    pub kda_w: Option<&'scratch mut DeviceBuffer<u16>>,
    pub kda_aqk: Option<&'scratch mut DeviceBuffer<u16>>,
    pub c_proj_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
}
