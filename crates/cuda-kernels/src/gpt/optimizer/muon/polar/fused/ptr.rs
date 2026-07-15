#[inline(always)]
pub(super) fn source_ptr(iter: u32, x: *mut f32, next: *mut f32) -> *const f32 {
    if iter & 1 == 0 { x } else { next }
}

#[inline(always)]
pub(super) fn target_ptr(iter: u32, x: *mut f32, next: *mut f32) -> *mut f32 {
    if iter & 1 == 0 { next } else { x }
}
