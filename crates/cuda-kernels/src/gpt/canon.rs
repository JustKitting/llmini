use std::sync::Arc;

use std::mem::size_of;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, memory};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::atomic::atomic_add_f32;
use crate::block_reduce::block_max_shared_f32_for_warps;
use crate::float_ptx::{abs_f32, max_f32};
use crate::launch::{grid_x_config, linear_config};
use crate::layer_norm_utils::{f16_column, nvfp4_column};
use crate::nvfp4::Nvfp4DeviceTensor;
use crate::nvfp4_quant::kernels::four_six::helpers::{
    GROUP_SIZE, GROUP_THREADS, four_six_global_scale, four_six_lane, six_grid_group_scale,
    store_four_six_payload_word,
};
use crate::warp_reduce::thread_lane_warp;

const THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / WARP_SIZE;
const CANON_KERNEL_SIZE: u32 = 4;
const WEIGHT_ROW_PARTITIONS: u32 = 16;
const WEIGHT_ROW_STRIDE: u32 = WARPS_PER_BLOCK * WEIGHT_ROW_PARTITIONS;
const CANON_QUANT_MAX_WIDTH: usize = 2048;

pub struct CanonForwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub weight: &'a DeviceBuffer<f32>,
    pub output: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub seq_len: u32,
    pub width: u32,
}

pub struct CanonQuantizeArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub weight: &'a DeviceBuffer<f32>,
    pub amax: &'out mut DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub seq_len: u32,
    pub width: u32,
}

pub struct CanonBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual_f16: &'a DeviceBuffer<u16>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub norm_weight: Nvfp4DeviceTensor<'a>,
    pub norm_bias: Nvfp4DeviceTensor<'a>,
    pub canon_weight: &'a DeviceBuffer<f32>,
    pub d_output: &'a DeviceBuffer<f32>,
    pub d_input: &'out mut DeviceBuffer<f32>,
    pub d_weight: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub seq_len: u32,
    pub width: u32,
    pub norm_output_scale: f32,
}

pub struct CanonModule {
    module: kernels::LoadedModule,
}

impl CanonModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn forward(&self, args: CanonForwardArgs<'_, '_>) -> Result<(), DriverError> {
        let element_count = args.row_count * args.width;
        assert!(args.input.len() >= element_count as usize);
        assert!(args.weight.len() >= (args.width * CANON_KERNEL_SIZE) as usize);
        assert!(args.output.len() >= element_count as usize);
        assert!(args.seq_len > 0);
        assert_eq!(args.row_count % args.seq_len, 0);

        self.module.canon_forward_f32_kernel(
            args.stream,
            linear_config(element_count, THREADS_PER_BLOCK),
            args.input,
            args.weight,
            args.output,
            args.row_count,
            args.seq_len,
            args.width,
        )
    }

    pub fn forward_quantized(&self, args: CanonQuantizeArgs<'_, '_>) -> Result<(), DriverError> {
        let element_count = args.row_count * args.width;
        assert!(args.input.len() >= element_count as usize);
        assert!(args.weight.len() >= (args.width * CANON_KERNEL_SIZE) as usize);
        assert!(args.amax.len() >= args.row_count as usize);
        assert!(args.out_fp4.len() >= element_count as usize / 2);
        assert!(args.out_scales.len() >= element_count as usize / GROUP_SIZE);
        assert!(args.out_global_scale.len() >= args.row_count as usize);
        assert!(args.seq_len > 0);
        assert_eq!(args.row_count % args.seq_len, 0);
        assert!(args.width as usize <= CANON_QUANT_MAX_WIDTH);
        assert!(args.width.is_multiple_of(GROUP_SIZE as u32));

        self.module.canon_forward_quantized_f32_kernel(
            args.stream,
            grid_x_config(args.row_count, THREADS_PER_BLOCK),
            args.input,
            args.weight,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.row_count,
            args.seq_len,
            args.width,
        )
    }

    pub fn backward(&self, args: CanonBackwardArgs<'_, '_>) -> Result<(), DriverError> {
        let element_count = args.row_count * args.width;
        let weight_count = args.width * CANON_KERNEL_SIZE;
        assert!(args.residual_f16.len() >= element_count as usize);
        assert!(args.mean.len() >= args.row_count as usize);
        assert!(args.inv_std.len() >= args.row_count as usize);
        assert!(args.canon_weight.len() >= weight_count as usize);
        assert!(args.d_output.len() >= element_count as usize);
        assert!(args.d_input.len() >= element_count as usize);
        assert!(args.d_weight.len() >= weight_count as usize);
        assert!(args.seq_len > 0);
        assert_eq!(args.row_count % args.seq_len, 0);

        unsafe {
            memory::memset_d8_async(
                args.d_weight.cu_deviceptr(),
                0,
                weight_count as usize * size_of::<f32>(),
                args.stream.cu_stream(),
            )?;
        }
        self.module.canon_backward_weight_f32_kernel(
            args.stream,
            grid_x_config(
                args.width.div_ceil(WARP_SIZE) * WEIGHT_ROW_PARTITIONS,
                THREADS_PER_BLOCK,
            ),
            args.residual_f16,
            args.mean,
            args.inv_std,
            args.norm_weight.bytes,
            args.norm_weight.scales,
            args.norm_bias.bytes,
            args.norm_bias.scales,
            args.norm_weight.global_scale,
            args.norm_bias.global_scale,
            args.d_output,
            args.canon_weight,
            args.d_input,
            args.d_weight,
            args.row_count,
            args.seq_len,
            args.width,
            args.norm_output_scale,
        )
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    static mut WEIGHT_0_PARTIALS: SharedArray<f32, { THREADS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;
    static mut WEIGHT_1_PARTIALS: SharedArray<f32, { THREADS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;
    static mut WEIGHT_2_PARTIALS: SharedArray<f32, { THREADS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;
    static mut WEIGHT_3_PARTIALS: SharedArray<f32, { THREADS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;
    static mut CANON_QUANT_ROW: SharedArray<f32, CANON_QUANT_MAX_WIDTH> = SharedArray::UNINIT;
    static mut CANON_QUANT_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;

    #[kernel]
    pub fn canon_forward_f32_kernel(
        input: &[f32],
        weight: &[f32],
        mut output: DisjointSlice<f32>,
        row_count: u32,
        seq_len: u32,
        width: u32,
    ) {
        let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
        let element_count = row_count * width;
        if index >= element_count {
            return;
        }

        let row = index / width;
        let col = index % width;
        let position = row % seq_len;
        let row_base = row as usize * width as usize;
        let weight_base = col as usize * CANON_KERNEL_SIZE as usize;
        let mut value = input[row_base + col as usize];
        let mut lag = 0;
        while lag < CANON_KERNEL_SIZE {
            if lag <= position {
                let source_row = row - lag;
                let source = input[source_row as usize * width as usize + col as usize];
                value += weight[weight_base + lag as usize] * source;
            }
            lag += 1;
        }

        unsafe {
            *output.get_unchecked_mut(index as usize) = value;
        }
    }

    #[kernel]
    pub fn canon_forward_quantized_f32_kernel(
        input: &[f32],
        weight: &[f32],
        mut amax: DisjointSlice<f32>,
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        row_count: u32,
        seq_len: u32,
        width: u32,
    ) {
        let row = thread::blockIdx_x();
        let (thread_id, lane, warp_in_block) = thread_lane_warp();
        if row >= row_count {
            return;
        }

        let position = row % seq_len;
        let row_base = row as usize * width as usize;
        let mut local_amax = 0.0f32;
        let mut col = thread_id;
        while col < width {
            let weight_base = col as usize * CANON_KERNEL_SIZE as usize;
            let mut value = input[row_base + col as usize];
            let mut lag = 0;
            while lag < CANON_KERNEL_SIZE {
                if lag <= position {
                    let source_row = row - lag;
                    value += weight[weight_base + lag as usize]
                        * input[source_row as usize * width as usize + col as usize];
                }
                lag += 1;
            }
            unsafe {
                CANON_QUANT_ROW[col as usize] = value;
            }
            local_amax = max_f32(local_amax, abs_f32(value));
            col += THREADS_PER_BLOCK;
        }

        let tensor_amax = unsafe {
            block_max_shared_f32_for_warps(
                &mut CANON_QUANT_AMAX,
                WARPS_PER_BLOCK,
                local_amax,
                lane,
                warp_in_block,
                0.0,
            )
        };
        let global_scale = four_six_global_scale(tensor_amax, 1.0);
        if thread_id == 0 {
            unsafe {
                *amax.get_unchecked_mut(row as usize) = tensor_amax;
                *out_global_scale.get_unchecked_mut(row as usize) = global_scale;
            }
        }

        let (lane_in_group, group_mask, group_leader) = four_six_lane();
        let groups_per_row = width / GROUP_SIZE as u32;
        let groups_per_block = THREADS_PER_BLOCK / GROUP_THREADS as u32;
        let mut group_in_row = thread_id / GROUP_THREADS as u32;
        while group_in_row < groups_per_row {
            let local_base = group_in_row as usize * GROUP_SIZE;
            let lane_base = local_base + 4 * lane_in_group;
            let (value_0, value_1, value_2, value_3) = unsafe {
                (
                    CANON_QUANT_ROW[lane_base],
                    CANON_QUANT_ROW[lane_base + 1],
                    CANON_QUANT_ROW[lane_base + 2],
                    CANON_QUANT_ROW[lane_base + 3],
                )
            };
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                1.0,
                group_mask,
                group_leader,
                lane_in_group,
            );
            let group = row * groups_per_row + group_in_row;
            let output_base = group as usize * GROUP_SIZE;
            unsafe {
                if lane_in_group == 0 {
                    *out_scales.get_unchecked_mut(group as usize) = scale_bits;
                }
                store_four_six_payload_word(
                    out_fp4.as_mut_ptr(),
                    output_base,
                    lane_in_group,
                    payload_word,
                );
            }
            group_in_row += groups_per_block;
        }
    }

    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    #[kernel]
    pub fn canon_backward_weight_f32_kernel(
        residual_f16: &[u16],
        mean: &[f32],
        inv_std: &[f32],
        norm_weight_bytes: &[u8],
        norm_weight_scales: &[u8],
        norm_bias_bytes: &[u8],
        norm_bias_scales: &[u8],
        norm_weight_global_scale: &[f32],
        norm_bias_global_scale: &[f32],
        d_output: &[f32],
        canon_weight: &[f32],
        mut d_input: DisjointSlice<f32>,
        mut d_weight: DisjointSlice<f32>,
        row_count: u32,
        seq_len: u32,
        width: u32,
        norm_output_scale: f32,
    ) {
        let block = thread::blockIdx_x();
        let partition = block % WEIGHT_ROW_PARTITIONS;
        let col_tile = block / WEIGHT_ROW_PARTITIONS;
        let (tid, lane, warp_in_block) = thread_lane_warp();
        let col = col_tile * WARP_SIZE + lane;
        let mut grad_0 = 0.0f32;
        let mut grad_1 = 0.0f32;
        let mut grad_2 = 0.0f32;
        let mut grad_3 = 0.0f32;

        if col < width {
            let norm_gamma = nvfp4_column(
                norm_weight_bytes,
                norm_weight_scales,
                norm_weight_global_scale[0],
                0,
                col,
                width,
            );
            let norm_beta = nvfp4_column(
                norm_bias_bytes,
                norm_bias_scales,
                norm_bias_global_scale[0],
                0,
                col,
                width,
            );
            let canon_weight_base = col as usize * CANON_KERNEL_SIZE as usize;
            let canon_weight_0 = canon_weight[canon_weight_base];
            let canon_weight_1 = canon_weight[canon_weight_base + 1];
            let canon_weight_2 = canon_weight[canon_weight_base + 2];
            let canon_weight_3 = canon_weight[canon_weight_base + 3];
            let mut source_row = partition * WARPS_PER_BLOCK + warp_in_block;
            while source_row < row_count {
                let source_base = source_row as usize * width as usize;
                let residual = f16_column(residual_f16, source_base, col, width);
                let row_mean = warp::shuffle_f32(
                    if lane == 0 {
                        mean[source_row as usize]
                    } else {
                        0.0
                    },
                    0,
                );
                let row_inv_std = warp::shuffle_f32(
                    if lane == 0 {
                        inv_std[source_row as usize]
                    } else {
                        0.0
                    },
                    0,
                );
                let normalized = ((residual - row_mean) * row_inv_std * norm_gamma + norm_beta)
                    * norm_output_scale;
                let d_base = source_base + col as usize;
                let d_0 = d_output[d_base];
                let mut input_grad = d_0 + canon_weight_0 * d_0;
                grad_0 += d_0 * normalized;
                let position = source_row % seq_len;
                if position + 1 < seq_len {
                    let d_1 = d_output[d_base + width as usize];
                    input_grad += canon_weight_1 * d_1;
                    grad_1 += d_1 * normalized;
                }
                if position + 2 < seq_len {
                    let d_2 = d_output[d_base + 2 * width as usize];
                    input_grad += canon_weight_2 * d_2;
                    grad_2 += d_2 * normalized;
                }
                if position + 3 < seq_len {
                    let d_3 = d_output[d_base + 3 * width as usize];
                    input_grad += canon_weight_3 * d_3;
                    grad_3 += d_3 * normalized;
                }
                unsafe {
                    *d_input.get_unchecked_mut(d_base) = input_grad;
                }
                source_row += WEIGHT_ROW_STRIDE;
            }
        }

        unsafe {
            WEIGHT_0_PARTIALS[tid as usize] = grad_0;
            WEIGHT_1_PARTIALS[tid as usize] = grad_1;
            WEIGHT_2_PARTIALS[tid as usize] = grad_2;
            WEIGHT_3_PARTIALS[tid as usize] = grad_3;
        }
        thread::sync_threads();

        if warp_in_block == 0 && col < width {
            let mut sum_0 = 0.0f32;
            let mut sum_1 = 0.0f32;
            let mut sum_2 = 0.0f32;
            let mut sum_3 = 0.0f32;
            let mut partial_warp = 0;
            while partial_warp < WARPS_PER_BLOCK {
                let partial = (partial_warp * WARP_SIZE + lane) as usize;
                unsafe {
                    sum_0 += WEIGHT_0_PARTIALS[partial];
                    sum_1 += WEIGHT_1_PARTIALS[partial];
                    sum_2 += WEIGHT_2_PARTIALS[partial];
                    sum_3 += WEIGHT_3_PARTIALS[partial];
                }
                partial_warp += 1;
            }
            let base = col as usize * CANON_KERNEL_SIZE as usize;
            unsafe {
                atomic_add_f32(d_weight.as_mut_ptr().add(base), sum_0);
                atomic_add_f32(d_weight.as_mut_ptr().add(base + 1), sum_1);
                atomic_add_f32(d_weight.as_mut_ptr().add(base + 2), sum_2);
                atomic_add_f32(d_weight.as_mut_ptr().add(base + 3), sum_3);
            }
        }
    }
}
