#![expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[path = "body/padded.rs"]
mod padded;
pub(super) use padded::{fp32_to_nvfp4_ms_eden_body, fp32_transpose_to_nvfp4_ms_eden_body};
