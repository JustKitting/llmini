use cuda_core::DeviceBuffer;

pub struct LayerNormGrads<'a> {
    pub d_weight: &'a mut DeviceBuffer<f32>,
    pub d_bias: &'a mut DeviceBuffer<f32>,
}

impl<'a> LayerNormGrads<'a> {
    pub fn reborrow(&mut self) -> LayerNormGrads<'_> {
        LayerNormGrads {
            d_weight: &mut *self.d_weight,
            d_bias: &mut *self.d_bias,
        }
    }
}
