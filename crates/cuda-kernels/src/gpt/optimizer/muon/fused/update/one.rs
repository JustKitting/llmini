use crate::float_ptx::abs_f32;

use crate::device_ptr::read_f32;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_one(
    u: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    rows: u32,
    cols: u32,
    len: u32,
    transposed: bool,
    scale: f32,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    index: u32,
) -> f32 {
    if index >= len {
        return 0.0;
    }

    let row = index / cols;
    let col = index - row * cols;
    let update_index = if transposed { col * rows + row } else { index };
    let muon_update = scale * read_f32(u, update_index);
    let decay = 1.0 - learning_rate * weight_decay;

    unsafe {
        let z = z_master.add(index as usize);
        let x = x_master.add(index as usize);
        let next_z = *z * decay - learning_rate * muon_update;
        let next_x = *x + average_coefficient * (next_z - *x);
        *z = next_z;
        *x = next_x;
        abs_f32(next_x)
    }
}
