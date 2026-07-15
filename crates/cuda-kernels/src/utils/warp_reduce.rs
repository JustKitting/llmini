use cuda_device::{ptx_asm, thread, warp};

use crate::float_ptx::max_f32;
use crate::shuffle;

const FULL_WARP_MASK: u32 = 0xffff_ffff;

#[inline(always)]
pub fn thread_lane_warp() -> (u32, u32, u32) {
    let thread_id = thread::threadIdx_x();
    (thread_id, warp::lane_id(), thread_id / 32)
}

#[inline(always)]
pub fn warp_sum_f32(mut value: f32) -> f32 {
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 16);
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 8);
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 4);
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 2);
    value + shuffle::xor_f32_sync(FULL_WARP_MASK, value, 1)
}

#[inline(always)]
pub fn warp_max_f32(mut value: f32) -> f32 {
    value = max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 16));
    value = max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 8));
    value = max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 4));
    value = max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 2));
    max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 1))
}

#[inline(always)]
pub fn warp_max_nonnegative_f32(value: f32) -> f32 {
    redux_max_u32(value.to_bits(), FULL_WARP_MASK)
}

#[inline(always)]
pub fn half_warp_sum_f32(mut value: f32, mask: u32) -> f32 {
    value += shuffle::xor_f32_sync(mask, value, 8);
    value += shuffle::xor_f32_sync(mask, value, 4);
    value += shuffle::xor_f32_sync(mask, value, 2);
    value + shuffle::xor_f32_sync(mask, value, 1)
}

#[inline(always)]
pub fn half_warp_max_f32(mut value: f32, mask: u32) -> f32 {
    value = max_f32(value, shuffle::xor_f32_sync(mask, value, 8));
    value = max_f32(value, shuffle::xor_f32_sync(mask, value, 4));
    value = max_f32(value, shuffle::xor_f32_sync(mask, value, 2));
    max_f32(value, shuffle::xor_f32_sync(mask, value, 1))
}

#[inline(always)]
pub fn half_warp_max_nonnegative_f32(value: f32, mask: u32) -> f32 {
    redux_max_u32(value.to_bits(), mask)
}

#[inline(always)]
fn redux_max_u32(value: u32, mask: u32) -> f32 {
    let result: u32;
    unsafe {
        ptx_asm!(
            "redux.sync.max.u32 %0, %1, %2;",
            out("=r") result,
            in("r") value,
            in("r") mask,
            options(register_only),
        );
    }
    f32::from_bits(result)
}
