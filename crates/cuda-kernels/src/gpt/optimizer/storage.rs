//! Physical row-major [output, input] matrix policy shared by host and PTX.
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

pub const MUON_STORAGE_LAYOUT_VERSION: u32 = 1;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn muon_storage_layout_signature_kernel(mut output: DisjointSlice<u32>) {
        if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
            unsafe {
                *output.get_unchecked_mut(0) = MUON_STORAGE_LAYOUT_VERSION;
            }
        }
    }
}
