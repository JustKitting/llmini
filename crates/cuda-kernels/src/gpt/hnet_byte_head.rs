//! NVFP4 byte readout with explicit document endings and a masked loss.
use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::{
    block_reduce::{block_max_leader_f32, block_max_shared_f32_for_warps, block_sum_shared_f32},
    float_ptx::{abs_f32, exp_preserve_subnormal_f32, log1p_f32, max_f32},
    launch::grid_x_config,
    lm_head::{LmHeadModule, LmHeadTmaArgs},
    nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor},
    nvfp4_tma_matmul::{launcher::Nvfp4GemmModule, tma::TmaNvfp4DeviceScaleDescriptors},
    warp_reduce::thread_lane_warp,
};

pub const BYTE_EOS: u32 = 256;
pub const BYTE_BOS: u32 = 257;
pub const BYTE_PADDING_TARGET: u32 = u32::MAX;
pub const BYTE_LOGIT_STRIDE: u32 = 384;
const CLASSES: u32 = BYTE_EOS + 1;
const THREADS: u32 = 128;
const WARPS: usize = (THREADS / 32) as usize;

pub struct ByteHeadModule {
    loss: kernels::LoadedModule,
    projection: LmHeadModule,
}

pub struct ByteProjection<'a, 'out> {
    pub stream: &'a CudaStream,
    pub tma: &'a Nvfp4GemmModule,
    pub descriptors: &'out mut TmaNvfp4DeviceScaleDescriptors,
    pub decoder_values: Nvfp4RowwiseDeviceTensor<'a>,
    pub weights: Nvfp4DeviceTensor<'a>,
    pub logits: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub width: u32,
}

pub struct ByteLossArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub logits: &'a DeviceBuffer<f32>,
    pub targets: &'a DeviceBuffer<u32>,
    pub losses: &'out mut DeviceBuffer<f32>,
    pub dlogits: &'out mut DeviceBuffer<f32>,
    pub row_amax: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub valid_targets: u32,
}

impl ByteHeadModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            loss: kernels::from_module(module.clone())?,
            projection: LmHeadModule::from_module(module)?,
        })
    }

    pub fn project(&self, args: ByteProjection<'_, '_>) -> Result<(), DriverError> {
        let values = args
            .rows
            .checked_mul(BYTE_LOGIT_STRIDE)
            .ok_or(DriverError(1))?;
        if values == 0 || args.logits.len() < values as usize || args.width == 0 {
            return Err(DriverError(1));
        }
        self.projection.logits_tma(LmHeadTmaArgs {
            stream: args.stream,
            tma: args.tma,
            descriptors: args.descriptors,
            input: args.decoder_values,
            weight: args.weights,
            logits: args.logits,
            token_count: args.rows,
            input_dim: args.width,
            vocab_size: BYTE_LOGIT_STRIDE,
        })
    }

    pub fn loss(&self, args: ByteLossArgs<'_, '_>) -> Result<(), DriverError> {
        let values = args
            .rows
            .checked_mul(BYTE_LOGIT_STRIDE)
            .ok_or(DriverError(1))? as usize;
        let rows = args.rows as usize;
        if args.valid_targets == 0
            || args.valid_targets > args.rows
            || args.logits.len() < values
            || args.dlogits.len() < values
            || args.targets.len() < rows
            || args.losses.len() < rows
            || args.row_amax.len() < rows
        {
            return Err(DriverError(1));
        }
        self.loss.hnet_byte_loss_kernel(
            args.stream,
            grid_x_config(args.rows, THREADS),
            args.logits,
            args.targets,
            args.losses,
            args.dlogits,
            args.row_amax,
            args.valid_targets,
        )
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn hnet_byte_loss_kernel(
        logits: &[f32],
        targets: &[u32],
        mut losses: DisjointSlice<f32>,
        mut dlogits: DisjointSlice<f32>,
        mut row_amax: DisjointSlice<f32>,
        valid_targets: u32,
    ) {
        static mut REDUCE: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let row = thread::blockIdx_x() as usize;
        let (lane_thread, lane, warp_index) = thread_lane_warp();
        let base = row * BYTE_LOGIT_STRIDE as usize;
        let target = targets[row];
        if target >= CLASSES {
            for col in (lane_thread..BYTE_LOGIT_STRIDE).step_by(THREADS as usize) {
                unsafe { *dlogits.get_unchecked_mut(base + col as usize) = 0.0 };
            }
            if lane_thread == 0 {
                unsafe {
                    *losses.get_unchecked_mut(row) = if target == BYTE_PADDING_TARGET {
                        0.0
                    } else {
                        f32::NAN
                    };
                    *row_amax.get_unchecked_mut(row) = 0.0;
                }
            }
            return;
        }

        let mut maximum = f32::NEG_INFINITY;
        for col in (lane_thread..CLASSES).step_by(THREADS as usize) {
            maximum = max_f32(maximum, logits[base + col as usize]);
        }
        let maximum = unsafe {
            block_max_shared_f32_for_warps(
                &mut REDUCE,
                WARPS as u32,
                maximum,
                lane,
                warp_index,
                f32::NEG_INFINITY,
            )
        };
        let mut argmax = f32::NEG_INFINITY;
        for col in (lane_thread..CLASSES).step_by(THREADS as usize) {
            if logits[base + col as usize] == maximum {
                argmax = max_f32(argmax, -(col as f32));
            }
        }
        let maximum_col = -unsafe {
            block_max_shared_f32_for_warps(
                &mut REDUCE,
                WARPS as u32,
                argmax,
                lane,
                warp_index,
                f32::NEG_INFINITY,
            )
        } as i32;
        let mut nonmaximum_sum = 0.0;
        for col in (lane_thread..CLASSES).step_by(THREADS as usize) {
            if col != maximum_col as u32 {
                nonmaximum_sum += exp_preserve_subnormal_f32(logits[base + col as usize] - maximum);
            }
        }
        let nonmaximum_sum =
            unsafe { block_sum_shared_f32(&mut REDUCE, nonmaximum_sum, lane, warp_index) };
        let denominator = 1.0 + nonmaximum_sum;
        let mut nontarget_grad = 0.0;
        let mut local_amax = 0.0;
        for col in (lane_thread..BYTE_LOGIT_STRIDE).step_by(THREADS as usize) {
            if col == target {
                continue;
            }
            let grad = if col < CLASSES {
                let p =
                    exp_preserve_subnormal_f32(logits[base + col as usize] - maximum) / denominator;
                p / valid_targets as f32
            } else {
                0.0
            };
            nontarget_grad += grad;
            local_amax = max_f32(local_amax, abs_f32(grad));
            unsafe { *dlogits.get_unchecked_mut(base + col as usize) = grad };
        }
        let target_grad =
            -unsafe { block_sum_shared_f32(&mut REDUCE, nontarget_grad, lane, warp_index) };
        if lane_thread == 0 {
            unsafe {
                *dlogits.get_unchecked_mut(base + target as usize) = target_grad;
                *losses.get_unchecked_mut(row) =
                    (maximum - logits[base + target as usize]) + log1p_f32(nonmaximum_sum);
            }
            local_amax = max_f32(local_amax, abs_f32(target_grad));
        }
        if let Some(amax) =
            unsafe { block_max_leader_f32(&mut REDUCE, local_amax, lane, warp_index) }
        {
            unsafe { *row_amax.get_unchecked_mut(row) = amax };
        }
    }
}
