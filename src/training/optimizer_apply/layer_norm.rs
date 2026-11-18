use cuda_core::DriverError;

use crate::upload::UploadedLayerNorm;

use super::super::grad_block::LayerNormGradBuffers;
use super::super::optimizer_state::LayerNormState;
use super::adam::AdamUpdate;
use super::timed_ms;

pub(super) fn update_layer_norm_timed(
    adam: &mut AdamUpdate<'_, '_>,
    layer_norm: &mut UploadedLayerNorm,
    grads: &mut LayerNormGradBuffers,
    state: &mut LayerNormState,
) -> Result<f64, DriverError> {
    timed_ms(|| {
        adam.update(
            &mut layer_norm.weight,
            &mut grads.d_weight,
            &mut state.weight,
        )?;
        adam.update(&mut layer_norm.bias, &mut grads.d_bias, &mut state.bias)
    })
}
