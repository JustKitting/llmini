use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::mma::Nvfp4DeviceScaleMmaWeightTensor;
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;
use crate::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;
use crate::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors;

pub struct MsEdenOperandScratch<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,

    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub global_scale: &'a mut DeviceBuffer<f32>,
}

impl<'a> MsEdenOperandScratch<'a> {
    pub(crate) fn rowwise(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&*self.bytes, &*self.scales, &*self.global_scale)
    }

    pub(crate) fn device_scale_mma_weight(&self) -> Nvfp4DeviceScaleMmaWeightTensor<'_> {
        Nvfp4DeviceScaleMmaWeightTensor::new(&*self.bytes, &*self.scales, &*self.global_scale)
    }
}

pub struct MsEdenTransposeScratch<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
}
pub struct MsEdenTransposeScratchBuffer {
    pub bytes: DeviceBuffer<u8>,
    pub scales: DeviceBuffer<u8>,
}
impl MsEdenTransposeScratchBuffer {
    fn new(stream: &CudaStream, elements: usize, rows: usize) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, elements.div_ceil(2))?,
            scales: DeviceBuffer::zeroed(stream, packed_scale_len(rows, elements / rows))?,
        })
    }
    fn as_arg(&mut self) -> MsEdenTransposeScratch<'_> {
        MsEdenTransposeScratch {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
        }
    }
}

pub struct LinearBackwardMsEdenScratch<'a> {
    pub e_h: MsEdenOperandScratch<'a>,
    pub weight_t_h: MsEdenOperandScratch<'a>,
    pub e_t_h: MsEdenTransposeScratch<'a>,
    pub input_t_h: MsEdenOperandScratch<'a>,
    pub tma: LinearBackwardTmaScratch<'a>,
}

pub struct LinearBackwardTmaScratch<'a> {
    pub descriptors: &'a mut TmaNvfp4DeviceScaleDescriptors,
}

pub struct MsEdenOperandScratchBuffer {
    pub bytes: DeviceBuffer<u8>,
    pub scales: DeviceBuffer<u8>,

    pub chunk_amax: DeviceBuffer<f32>,
    pub global_scale: DeviceBuffer<f32>,
}

impl MsEdenOperandScratchBuffer {
    pub fn new(stream: &CudaStream, elements: usize, rows: usize) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, elements.div_ceil(2))?,
            scales: DeviceBuffer::zeroed(stream, packed_scale_len(rows, elements / rows))?,

            chunk_amax: DeviceBuffer::zeroed(stream, elements.div_ceil(32))?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub fn as_arg(&mut self) -> MsEdenOperandScratch<'_> {
        MsEdenOperandScratch {
            bytes: &mut self.bytes,
            scales: &mut self.scales,

            chunk_amax: &mut self.chunk_amax,
            global_scale: &mut self.global_scale,
        }
    }
}

pub struct LinearBackwardMsEdenScratchBuffers {
    pub e_h: MsEdenOperandScratchBuffer,
    pub weight_t_h: MsEdenOperandScratchBuffer,
    pub e_t_h: MsEdenTransposeScratchBuffer,
    pub input_t_h: MsEdenOperandScratchBuffer,
    pub tma: LinearBackwardTmaScratchBuffers,
}

pub struct LinearBackwardTmaScratchBuffers {
    pub descriptors: TmaNvfp4DeviceScaleDescriptors,
}

impl LinearBackwardMsEdenScratchBuffers {
    pub fn new(
        stream: &CudaStream,
        token_count: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DriverError> {
        let output_k = nvfp4_tc_matmul_padded_k(output_dim as u32) as usize;
        let token_k = nvfp4_tc_matmul_padded_k(token_count as u32) as usize;

        Ok(Self {
            e_h: MsEdenOperandScratchBuffer::new(stream, token_count * output_k, token_count)?,
            weight_t_h: MsEdenOperandScratchBuffer::new(stream, input_dim * output_k, input_dim)?,
            e_t_h: MsEdenTransposeScratchBuffer::new(stream, output_dim * token_k, output_dim)?,
            input_t_h: MsEdenOperandScratchBuffer::new(stream, input_dim * token_k, input_dim)?,
            tma: LinearBackwardTmaScratchBuffers::new(stream)?,
        })
    }

    pub fn as_args(&mut self) -> LinearBackwardMsEdenScratch<'_> {
        LinearBackwardMsEdenScratch {
            e_h: self.e_h.as_arg(),
            weight_t_h: self.weight_t_h.as_arg(),
            e_t_h: self.e_t_h.as_arg(),
            input_t_h: self.input_t_h.as_arg(),
            tma: self.tma.as_arg(),
        }
    }
}

impl LinearBackwardTmaScratchBuffers {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            descriptors: TmaNvfp4DeviceScaleDescriptors::new(stream)?,
        })
    }

    fn as_arg(&mut self) -> LinearBackwardTmaScratch<'_> {
        LinearBackwardTmaScratch {
            descriptors: &mut self.descriptors,
        }
    }
}

fn packed_scale_len(rows: usize, k: usize) -> usize {
    crate::nvfp4_tma_matmul::scale_layout::sm120_csf_scale_storage_len(rows, k)
}
