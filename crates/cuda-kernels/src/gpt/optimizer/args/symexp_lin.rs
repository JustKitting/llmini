use cuda_core::DeviceCopy;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SymExpLinSlotDescriptor {
    pub grad: u64,
    pub z_master: u64,
    pub x_master: u64,
    pub exponential_z: u64,
    pub exponential_x: u64,
    pub exponential_grad: u64,
    pub linear_z: u64,
    pub linear_x: u64,
    pub linear_grad: u64,
    pub curvature_z: u64,
    pub curvature_x: u64,
    pub curvature_grad: u64,
    pub rows: u32,
    pub cols: u32,
}

unsafe impl DeviceCopy for SymExpLinSlotDescriptor {}
