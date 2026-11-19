use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, memory};
use gpt2_nvfp4::{GPT2_N_HEAD, depth_growth::GrowthTransition};

use super::{
    AdamState, BlockState, Fp32AdamState, LayerNormState, LinearState, MuonState,
    OptimizerStateBuffers,
    tensor::{SymExpLinScalarState, SymExpLinState},
};

/// Copy existing representations on their training stream. Destination
/// allocations stay in place so uploaded pointer tables still refer to them.
fn copy<T>(
    stream: &CudaStream,
    source: &DeviceBuffer<T>,
    destination: &mut DeviceBuffer<T>,
) -> Result<(), DriverError> {
    if source.len() != destination.len()
        || !Arc::ptr_eq(source.context(), destination.context())
        || !Arc::ptr_eq(source.context(), stream.context())
    {
        return Err(DriverError(1));
    }
    if source.is_empty() {
        return Ok(());
    }
    // Separate optimizer state objects own disjoint allocations of equal size.
    unsafe {
        memory::memcpy_dtod_async(
            destination.cu_deviceptr(),
            source.cu_deviceptr(),
            source.num_bytes(),
            stream.cu_stream(),
        )
    }
}

macro_rules! copy_buffers {
    ($vis:vis $name:ident, $type:ident, [$($field:ident),+ $(,)?] $(, skip [$($skip:ident),+ $(,)?])?) => {
        $vis fn $name(
            stream: &CudaStream,
            source: &$type,
            destination: &mut $type,
        ) -> Result<(), DriverError> {
            let $type { $($field: _,)+ $($($skip: _,)+)? } = source;
            $(copy(stream, &source.$field, &mut destination.$field)?;)+
            Ok(())
        }
    };
}

copy_buffers!(
    adam,
    AdamState,
    [z_master, x_master, first, second, schedule_amax]
);
copy_buffers!(
    pub(in crate::training) copy_fp32,
    Fp32AdamState,
    [z_master, x_master, first, second]
);
copy_buffers!(
    scalar,
    SymExpLinScalarState,
    [z_master, x_master, first, second],
    skip[grad]
);

pub(in crate::training) fn copy_muon(
    stream: &CudaStream,
    source: &MuonState,
    destination: &mut MuonState,
) -> Result<(), DriverError> {
    let MuonState {
        z_master,
        x_master,
        momentum,
        variance,
        second_momentum,
        schedule_amax,
        symexp_lin,
    } = source;
    copy(stream, z_master, &mut destination.z_master)?;
    copy(stream, x_master, &mut destination.x_master)?;
    copy(stream, momentum, &mut destination.momentum)?;
    copy(stream, variance, &mut destination.variance)?;
    copy(stream, second_momentum, &mut destination.second_momentum)?;
    copy(stream, schedule_amax, &mut destination.schedule_amax)?;
    let SymExpLinState {
        exponential,
        linear,
        curvature,
    } = symexp_lin;
    scalar(stream, exponential, &mut destination.symexp_lin.exponential)?;
    scalar(stream, linear, &mut destination.symexp_lin.linear)?;
    scalar(stream, curvature, &mut destination.symexp_lin.curvature)
}

fn norm(
    stream: &CudaStream,
    source: &LayerNormState,
    destination: &mut LayerNormState,
) -> Result<(), DriverError> {
    let LayerNormState { weight, bias } = source;
    adam(stream, weight, &mut destination.weight)?;
    adam(stream, bias, &mut destination.bias)
}

fn linear(
    stream: &CudaStream,
    source: &LinearState,
    destination: &mut LinearState,
) -> Result<(), DriverError> {
    let LinearState { weight_muon, bias } = source;
    copy_muon(stream, weight_muon, &mut destination.weight_muon)?;
    adam(stream, bias, &mut destination.bias)
}

fn block(
    stream: &CudaStream,
    source: &BlockState,
    destination: &mut BlockState,
) -> Result<(), DriverError> {
    let BlockState {
        ln_1,
        canon_a,
        attn_qkv,
        attn_qk_scale,
        attn_c_proj,
        ln_2,
        canon_c,
        mlp_up,
        mlp_down,
    } = source;
    norm(stream, ln_1, &mut destination.ln_1)?;
    copy_fp32(stream, canon_a, &mut destination.canon_a)?;
    linear(stream, attn_qkv, &mut destination.attn_qkv)?;
    adam(stream, attn_qk_scale, &mut destination.attn_qk_scale)?;
    linear(stream, attn_c_proj, &mut destination.attn_c_proj)?;
    norm(stream, ln_2, &mut destination.ln_2)?;
    copy_fp32(stream, canon_c, &mut destination.canon_c)?;
    linear(stream, mlp_up, &mut destination.mlp_up)?;
    linear(stream, mlp_down, &mut destination.mlp_down)
}

fn xsa(
    stream: &CudaStream,
    state: &mut AdamState,
    source: usize,
    destination: usize,
) -> Result<(), DriverError> {
    let bytes = GPT2_N_HEAD * std::mem::size_of::<f32>();
    for field in [
        &mut state.z_master,
        &mut state.x_master,
        &mut state.first,
        &mut state.second,
    ] {
        if field.num_bytes() < (destination + 1) * bytes
            || !Arc::ptr_eq(field.context(), stream.context())
        {
            return Err(DriverError(1));
        }
        // Consecutive heads belonging to different blocks never overlap.
        unsafe {
            memory::memcpy_dtod_async(
                field.cu_deviceptr() + (destination * bytes) as u64,
                field.cu_deviceptr() + (source * bytes) as u64,
                bytes,
                stream.cu_stream(),
            )?;
        }
    }
    Ok(())
}

impl OptimizerStateBuffers {
    pub(in crate::training) fn copy_growth_blocks(
        &mut self,
        stream: &CudaStream,
        growth: GrowthTransition,
    ) -> Result<(), DriverError> {
        if growth.depth() > self.blocks.len() {
            return Err(DriverError(1));
        }
        stream.context().bind_to_thread()?;
        for (source, destination) in growth.copies() {
            let (before, after) = self.blocks.split_at_mut(destination);
            block(stream, &before[source], &mut after[0])?;
            xsa(stream, &mut self.xsa_alphas, source, destination)?;
        }
        self.invalidate_training_weights();
        self.invalidate_scheduled_values();
        Ok(())
    }
}
