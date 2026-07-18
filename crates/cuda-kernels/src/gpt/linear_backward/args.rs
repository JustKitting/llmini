#[path = "args/ms_eden.rs"]
mod ms_eden;

use cuda_core::{CudaStream, DeviceBuffer};

use crate::mma::{Nvfp4DeviceScaleMmaWeightTensor, Nvfp4FourSixMmaWeightTensor};
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;
use crate::nvfp4_tma_matmul::launcher::Nvfp4GemmRoute;

pub use ms_eden::{
    LinearBackwardInputTranspose, LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratch,
    LinearBackwardMsEdenScratchBuffers, LinearBackwardTmaScratch, LinearBackwardWeightTranspose,
    MsEdenOperandScratch, MsEdenOperandScratchBuffer,
};

#[derive(Clone, Copy)]
enum LinearBackwardRouteLayout {
    MlpDown,
    MlpUp,
}

#[derive(Clone, Copy)]
pub struct LinearBackwardRoute<'a> {
    layout: LinearBackwardRouteLayout,
    masks: &'a DeviceBuffer<u64>,
    token_tiles: u32,
    feature_tiles: u32,
    active_tiles: u32,
}

impl<'a> LinearBackwardRoute<'a> {
    fn new(
        layout: LinearBackwardRouteLayout,
        masks: &'a DeviceBuffer<u64>,
        token_tiles: u32,
        feature_tiles: u32,
        active_tiles: u32,
    ) -> Self {
        Self {
            layout,
            masks,
            token_tiles,
            feature_tiles,
            active_tiles,
        }
    }

    pub fn mlp_down(
        masks: &'a DeviceBuffer<u64>,
        token_tiles: u32,
        feature_tiles: u32,
        active_tiles: u32,
    ) -> Self {
        Self::new(
            LinearBackwardRouteLayout::MlpDown,
            masks,
            token_tiles,
            feature_tiles,
            active_tiles,
        )
    }

    pub fn mlp_up(
        masks: &'a DeviceBuffer<u64>,
        token_tiles: u32,
        feature_tiles: u32,
        active_tiles: u32,
    ) -> Self {
        Self::new(
            LinearBackwardRouteLayout::MlpUp,
            masks,
            token_tiles,
            feature_tiles,
            active_tiles,
        )
    }

    pub(crate) fn dinput(self) -> Nvfp4GemmRoute<'a> {
        match self.layout {
            LinearBackwardRouteLayout::MlpDown => Nvfp4GemmRoute::token_m_feature_n(
                self.masks,
                self.token_tiles,
                self.feature_tiles,
                self.active_tiles,
            ),
            LinearBackwardRouteLayout::MlpUp => Nvfp4GemmRoute::token_m_feature_k(
                self.masks,
                self.token_tiles,
                self.feature_tiles,
                self.active_tiles,
            ),
        }
    }

    pub(crate) fn dweight(self) -> Nvfp4GemmRoute<'a> {
        match self.layout {
            LinearBackwardRouteLayout::MlpDown => Nvfp4GemmRoute::feature_n_token_k(
                self.masks,
                self.token_tiles,
                self.feature_tiles,
                self.active_tiles,
            ),
            LinearBackwardRouteLayout::MlpUp => Nvfp4GemmRoute::feature_m_token_k(
                self.masks,
                self.token_tiles,
                self.feature_tiles,
                self.active_tiles,
            ),
        }
    }
}

pub struct LinearBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub e_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub e_t_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub input_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: Option<&'out mut DeviceBuffer<f32>>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct LinearBackwardDeviceScaleArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub e_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight_t_h: Nvfp4DeviceScaleMmaWeightTensor<'a>,
    pub e_t_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub input_t_h: Nvfp4DeviceScaleMmaWeightTensor<'a>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}
