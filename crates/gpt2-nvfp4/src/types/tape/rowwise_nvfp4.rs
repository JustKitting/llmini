use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::types::RowwiseNvfp4Tape;
use crate::types::RowwiseNvfp4Scratch;

impl<'a> RowwiseNvfp4Tape<'a> {
    pub(crate) fn saved(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&*self.bytes, &*self.scales, &*self.global_scales)
    }

    pub(crate) fn reborrow(&mut self) -> RowwiseNvfp4Tape<'_> {
        RowwiseNvfp4Tape {
            bytes: &mut *self.bytes,
            scales: &mut *self.scales,
            global_scales: &mut *self.global_scales,
        }
    }

    /// Exchanging handles keeps the quantizer's destination stable without
    /// copying FP4 payload, compressed scales, or row globals on the GPU.
    pub(crate) fn swap_storage(&mut self, scratch: &mut RowwiseNvfp4Scratch<'_>) {
        assert_eq!(self.bytes.len(), scratch.bytes.len());
        assert_eq!(self.scales.len(), scratch.scales.len());
        assert_eq!(self.global_scales.len(), scratch.global_scales.len());
        std::mem::swap(self.bytes, scratch.bytes);
        std::mem::swap(self.scales, scratch.scales);
        std::mem::swap(self.global_scales, scratch.global_scales);
    }
}
