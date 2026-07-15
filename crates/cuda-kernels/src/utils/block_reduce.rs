#![allow(unused_unsafe)]

macro_rules! block_reduce_f32 {
    ($storage:ident, $warps:expr, $local:expr, $lane:expr, $warp:expr, $op:path, $identity:expr) => {{
        let warp_total = $op($local);
        if $lane == 0 {
            unsafe {
                $storage[$warp as usize] = warp_total;
            }
        }
        cuda_device::thread::sync_threads();

        let partial = if $warp == 0 && $lane < $warps {
            unsafe { $storage[$lane as usize] }
        } else {
            $identity
        };
        let block_total = $op(partial);
        if $warp == 0 && $lane == 0 {
            unsafe {
                $storage[0] = block_total;
            }
        }
        cuda_device::thread::sync_threads();

        let block_total = unsafe { $storage[0] };
        cuda_device::thread::sync_threads();
        block_total
    }};
}

pub(crate) use block_reduce_f32;

macro_rules! block_max_store_f32 {
    ($storage:ident, $out:ident[$index:expr], $local:expr, $lane:expr, $warp:expr) => {
        if let Some(block_value) = unsafe {
            crate::block_reduce::block_max_leader_f32(&mut $storage, $local, $lane, $warp)
        } {
            unsafe {
                *$out.get_unchecked_mut($index as usize) = block_value;
            }
        }
    };
}

pub(crate) use block_max_store_f32;

use cuda_device::SharedArray;

use crate::warp_reduce::{warp_max_finite_f32, warp_max_nonnegative_f32, warp_sum_f32};

#[inline(always)]
pub(crate) fn block_sum_shared_f32<const WARPS: usize>(
    storage: &mut SharedArray<f32, WARPS>,
    local: f32,
    lane: u32,
    warp: u32,
) -> f32 {
    block_sum_shared_f32_for_warps(storage, WARPS as u32, local, lane, warp)
}

#[inline(always)]
pub(crate) fn block_sum_shared_f32_for_warps<const WARPS: usize>(
    storage: &mut SharedArray<f32, WARPS>,
    active_warps: u32,
    local: f32,
    lane: u32,
    warp: u32,
) -> f32 {
    block_reduce_f32!(storage, active_warps, local, lane, warp, warp_sum_f32, 0.0)
}

#[inline(always)]
pub(crate) fn block_max_shared_f32_for_warps<const WARPS: usize>(
    storage: &mut SharedArray<f32, WARPS>,
    active_warps: u32,
    local: f32,
    lane: u32,
    warp: u32,
    identity: f32,
) -> f32 {
    block_reduce_f32!(
        storage,
        active_warps,
        local,
        lane,
        warp,
        warp_max_finite_f32,
        identity
    )
}

#[inline(always)]
pub(crate) fn block_max_leader_f32<const WARPS: usize>(
    storage: &mut SharedArray<f32, WARPS>,
    local: f32,
    lane: u32,
    warp: u32,
) -> Option<f32> {
    let warp_value = warp_max_nonnegative_f32(local);
    if lane == 0 {
        unsafe {
            storage[warp as usize] = warp_value;
        }
    }

    cuda_device::thread::sync_threads();

    if warp != 0 {
        return None;
    }

    let partial = if lane < WARPS as u32 {
        unsafe { storage[lane as usize] }
    } else {
        0.0
    };
    let block_value = warp_max_nonnegative_f32(partial);
    if lane == 0 { Some(block_value) } else { None }
}

#[inline(always)]
pub(crate) fn block_max_pair_shared_f32<const WARPS: usize>(
    first_storage: &mut SharedArray<f32, WARPS>,
    second_storage: &mut SharedArray<f32, WARPS>,
    first_local: f32,
    second_local: f32,
    lane: u32,
    warp: u32,
) -> (f32, f32) {
    let first_warp = warp_max_nonnegative_f32(first_local);
    let second_warp = warp_max_nonnegative_f32(second_local);
    if lane == 0 {
        unsafe {
            first_storage[warp as usize] = first_warp;
            second_storage[warp as usize] = second_warp;
        }
    }
    cuda_device::thread::sync_threads();

    let first_partial = if warp == 0 && lane < WARPS as u32 {
        unsafe { first_storage[lane as usize] }
    } else {
        0.0
    };
    let second_partial = if warp == 0 && lane < WARPS as u32 {
        unsafe { second_storage[lane as usize] }
    } else {
        0.0
    };
    let first_block = warp_max_nonnegative_f32(first_partial);
    let second_block = warp_max_nonnegative_f32(second_partial);
    if warp == 0 && lane == 0 {
        unsafe {
            first_storage[0] = first_block;
            second_storage[0] = second_block;
        }
    }
    cuda_device::thread::sync_threads();

    let result = unsafe { (first_storage[0], second_storage[0]) };
    cuda_device::thread::sync_threads();
    result
}

#[inline(always)]
pub(crate) fn block_max_pair_leader_f32<const WARPS: usize>(
    first_storage: &mut SharedArray<f32, WARPS>,
    second_storage: &mut SharedArray<f32, WARPS>,
    first_local: f32,
    second_local: f32,
    lane: u32,
    warp: u32,
) -> Option<(f32, f32)> {
    let first_warp = warp_max_nonnegative_f32(first_local);
    let second_warp = warp_max_nonnegative_f32(second_local);
    if lane == 0 {
        unsafe {
            first_storage[warp as usize] = first_warp;
            second_storage[warp as usize] = second_warp;
        }
    }
    cuda_device::thread::sync_threads();

    if warp != 0 {
        return None;
    }
    let first_partial = if lane < WARPS as u32 {
        unsafe { first_storage[lane as usize] }
    } else {
        0.0
    };
    let second_partial = if lane < WARPS as u32 {
        unsafe { second_storage[lane as usize] }
    } else {
        0.0
    };
    let first_block = warp_max_nonnegative_f32(first_partial);
    let second_block = warp_max_nonnegative_f32(second_partial);
    if lane == 0 {
        Some((first_block, second_block))
    } else {
        None
    }
}
