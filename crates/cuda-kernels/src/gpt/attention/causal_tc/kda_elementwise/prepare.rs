use cuda_device::DisjointSlice;

use super::super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use super::thread_index;
use crate::attention::CausalAttentionParams;
use crate::kda_elementwise::{KdaPrepareOutputs, prepare_kda_inputs_body};

pub(in super::super) fn prepare_kda_body(
    qkv: &[f32],
    qkv_f16: *mut u16,
    q: DisjointSlice<f32>,
    k: DisjointSlice<f32>,
    v: DisjointSlice<f32>,
    g: DisjointSlice<f32>,
    beta: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    prepare_kda_inputs_body(
        qkv,
        KdaPrepareOutputs { q, k, v, g, beta },
        qkv_f16,
        params,
        TC_FORWARD_THREADS_PER_BLOCK,
    );
}

pub(in super::super) fn zero_f32_body(mut values: DisjointSlice<f32>, element_count: u32) {
    let Some(index) = thread_index(element_count) else {
        return;
    };
    unsafe {
        *values.get_unchecked_mut(index as usize) = 0.0;
    }
}
