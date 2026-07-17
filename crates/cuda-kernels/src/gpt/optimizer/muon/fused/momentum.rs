use super::super::super::work_grid::WorkGrid;
use super::types::MuonMatrixShape;
use crate::device_ptr::write_f32;

pub(super) fn momentum_orient(
    grad: *const f32,
    momentum: *mut f32,
    oriented: *mut f32,
    work: WorkGrid,
    shape: MuonMatrixShape,
    mu: f32,
    grad_scale: f32,
    transposed: bool,
    nesterov: bool,
) {
    let len = shape.len();
    let mut index = work.thread();
    while index < len {
        let row = index / shape.cols;
        let col = index - row * shape.cols;
        let g = unsafe { *grad.add(index as usize) } * grad_scale;
        unsafe {
            let momentum_ptr = momentum.add(index as usize);
            let next_momentum = mu * *momentum_ptr + (1.0 - mu) * g;
            let update = if nesterov {
                mu * next_momentum + (1.0 - mu) * g
            } else {
                next_momentum
            };
            *momentum_ptr = next_momentum;
            let dst = if transposed {
                col * shape.rows + row
            } else {
                index
            };
            write_f32(oriented, dst, update);
        }
        index += work.stride();
    }
}
