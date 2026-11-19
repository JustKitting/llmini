use cuda_core::{CudaStream, DriverError};

use super::views::{HostGradView, parameter_gradient_views};
use crate::training::{grads::BackwardBuffers, next_latent::NextLatGradBuffers};

pub(in crate::training) struct NonFiniteGradient {
    pub(in crate::training) name: String,
    pub(in crate::training) index: usize,
    pub(in crate::training) value: f32,
}

pub(in crate::training) fn first_non_finite_gradient(
    stream: &CudaStream,
    grads: &BackwardBuffers,
    next_latent: &NextLatGradBuffers,
) -> Result<Option<NonFiniteGradient>, DriverError> {
    first_non_finite_view(stream, parameter_gradient_views(grads, Some(next_latent)))
}

pub(in crate::training) fn first_non_finite_backbone_gradient(
    stream: &CudaStream,
    grads: &BackwardBuffers,
) -> Result<Option<NonFiniteGradient>, DriverError> {
    first_non_finite_view(stream, parameter_gradient_views(grads, None))
}

fn first_non_finite_view<'a>(
    stream: &CudaStream,
    views: Vec<HostGradView<'a>>,
) -> Result<Option<NonFiniteGradient>, DriverError> {
    for view in views {
        let values = view.buffer.to_host_vec(stream)?;
        if let Some((index, value)) = values
            .iter()
            .take(view.len)
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Ok(Some(NonFiniteGradient {
                name: view.name,
                index,
                value,
            }));
        }
    }
    Ok(None)
}
