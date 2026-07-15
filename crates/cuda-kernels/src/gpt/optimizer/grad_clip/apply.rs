use cuda_device::thread;

use crate::device_ptr::{read_f32, write_f32};

use super::{APPLY_UNROLL, THREADS_PER_BLOCK, VALUES_PER_CHUNK};

pub(super) fn grad_clip_apply_body(
    chunk_ptrs: &[u64],
    chunk_lens: &[u32],
    scale: &[f32],
    chunk_count: u32,
) {
    let chunk = thread::blockIdx_x();
    if chunk >= chunk_count {
        return;
    }

    let ptr = chunk_ptrs[chunk as usize] as *mut f32;
    let len = chunk_lens[chunk as usize];
    let multiplier = scale[0];
    let mut offset = thread::threadIdx_x();

    while offset < VALUES_PER_CHUNK {
        apply_one(ptr, len, offset, multiplier);
        apply_one(ptr, len, offset + THREADS_PER_BLOCK, multiplier);
        apply_one(ptr, len, offset + THREADS_PER_BLOCK * 2, multiplier);
        apply_one(ptr, len, offset + THREADS_PER_BLOCK * 3, multiplier);
        offset += THREADS_PER_BLOCK * APPLY_UNROLL;
    }
}

#[inline(always)]
fn apply_one(ptr: *mut f32, len: u32, index: u32, multiplier: f32) {
    if index < len {
        let value = read_f32(ptr as *const f32, index) * multiplier;
        write_f32(ptr, index, value);
    }
}
