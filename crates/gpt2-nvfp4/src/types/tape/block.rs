use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::f16_tc_matmul::{F16ConvertArgs, F16TcMatmulModule};

use super::device_copy::copy_device;
use super::types::BlockForwardTape;
use crate::types::{AttentionForwardTape, BlockForwardSaved, MlpForwardTape};

impl<'a> BlockForwardTape<'a> {
    pub(super) fn saved(
        &self,
        batch_size: u32,
        seq_len: u32,
        row_count: u32,
    ) -> BlockForwardSaved<'_> {
        BlockForwardSaved {
            batch_size,
            seq_len,
            row_count,
            ln_1: self.ln_1.saved(row_count),
            qkv_input_nvfp4: self.qkv_input_nvfp4.saved(),
            qkv: &*self.qkv,
            attention_out: &*self.attention_out,
            kda_v_new: self.kda_v_new.as_ref().map(|buffer| &**buffer),
            kda_akk_inv: self.kda_akk_inv.as_ref().map(|buffer| &**buffer),
            kda_w: self.kda_w.as_ref().map(|buffer| &**buffer),
            kda_aqk: self.kda_aqk.as_ref().map(|buffer| &**buffer),
            attention_log_sum_exp: &*self.attention_log_sum_exp,
            c_proj_input_nvfp4: self.c_proj_input_nvfp4.saved(),
            ln_2: self.ln_2.saved(row_count),
            mlp_up_input_nvfp4: self.mlp_up_input_nvfp4.saved(),
            mlp_up: &*self.mlp_up,
            mlp_down_input_nvfp4: self.mlp_down_input_nvfp4.saved(),
        }
    }

    pub(super) fn reborrow(&mut self) -> BlockForwardTape<'_> {
        BlockForwardTape {
            ln_1: self.ln_1.reborrow(),
            qkv_input_nvfp4: self.qkv_input_nvfp4.reborrow(),
            qkv: &mut *self.qkv,
            attention_out: &mut *self.attention_out,
            kda_v_new: self.kda_v_new.as_mut().map(|buffer| &mut **buffer),
            kda_akk_inv: self.kda_akk_inv.as_mut().map(|buffer| &mut **buffer),
            kda_w: self.kda_w.as_mut().map(|buffer| &mut **buffer),
            kda_aqk: self.kda_aqk.as_mut().map(|buffer| &mut **buffer),
            attention_log_sum_exp: &mut *self.attention_log_sum_exp,
            c_proj_input_nvfp4: self.c_proj_input_nvfp4.reborrow(),
            ln_2: self.ln_2.reborrow(),
            mlp_up_input_nvfp4: self.mlp_up_input_nvfp4.reborrow(),
            mlp_up: &mut *self.mlp_up,
            mlp_down_input_nvfp4: self.mlp_down_input_nvfp4.reborrow(),
        }
    }

    pub(crate) fn attention_forward(&mut self) -> AttentionForwardTape<'_> {
        AttentionForwardTape {
            qkv_input_nvfp4: self.qkv_input_nvfp4.reborrow(),
            qkv_f16: &mut *self.qkv,
            attention_out_f16: &mut *self.attention_out,
            kda_v_new: self.kda_v_new.as_mut().map(|buffer| &mut **buffer),
            kda_akk_inv: self.kda_akk_inv.as_mut().map(|buffer| &mut **buffer),
            kda_w: self.kda_w.as_mut().map(|buffer| &mut **buffer),
            kda_aqk: self.kda_aqk.as_mut().map(|buffer| &mut **buffer),
            c_proj_input_nvfp4: self.c_proj_input_nvfp4.reborrow(),
        }
    }

    pub(crate) fn mlp_forward(&mut self) -> MlpForwardTape<'_> {
        MlpForwardTape {
            up_input_nvfp4: self.mlp_up_input_nvfp4.reborrow(),
            down_input_nvfp4: self.mlp_down_input_nvfp4.reborrow(),
        }
    }

    pub(crate) fn save_attention_log_sum_exp(
        &mut self,
        stream: &CudaStream,
        log_sum_exp: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, log_sum_exp, self.attention_log_sum_exp)
    }

    pub(crate) fn save_mlp_up_f16(
        &mut self,
        stream: &CudaStream,
        module: &F16TcMatmulModule,
        activation: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        let element_count = self.mlp_up.len() as u32;
        module.fp32_to_f16(F16ConvertArgs {
            stream,
            src: activation,
            dst: self.mlp_up,
            element_count,
        })
    }
}
