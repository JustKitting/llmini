use crate::{AppResult, training::runtime::Runtime, upload::UploadedNvfp4};
use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{GPT2_N_EMBD, Gpt2Rng};
pub(crate) struct Weights {
    pub(crate) weight: UploadedNvfp4,
}
impl Weights {
    pub(in crate::training) fn new(r: &Runtime, seed: u64) -> AppResult<Self> {
        let s = r.stream.as_ref();
        let d = GPT2_N_EMBD;
        let len = d * d;
        // Only initialization is performed on the host; native GPU packing uses actual amax.
        let mut rng = Gpt2Rng::new(seed ^ 0x4c4f524d20261115);
        let mut gaussian = Vec::with_capacity(len);
        while gaussian.len() < len {
            let u = (rng.next_u32() as f32 + 1.0) / 4294967296.0;
            let angle = (rng.next_u32() as f32 / 4294967296.0) * std::f32::consts::TAU;
            let radius = (-2.0 * u.ln()).sqrt() * 0.02;
            let (sin, cos) = angle.sin_cos();
            gaussian.push(radius * cos);
            if gaussian.len() < len {
                gaussian.push(radius * sin);
            }
        }
        let master = DeviceBuffer::from_host(s, &gaussian)?;
        let scales = rust_kernels_cuda::nvfp4::csf::encode_host(&vec![0; len / 16], d, d);
        let mut weight = UploadedNvfp4::from_host(s, &vec![0; len / 2], &scales, 0.0, len)?;
        let mut amax = DeviceBuffer::zeroed(s, 1)?;
        let mut chunks = DeviceBuffer::zeroed(s, len.div_ceil(128))?;
        r.optimizer.materialize_schedule_free(
            rust_kernels_cuda::optimizer::ScheduleFreeMaterializeArgs {
                stream: s,
                bytes: &mut weight.bytes,
                scales: &mut weight.scales,
                global_scale: &mut weight.global_scale,
                z_master: &master,
                x_master: &master,
                amax: &mut amax,
                chunk_amax: &mut chunks,
                len: len as u32,
                scale_rows: d as u32,
                scale_k: d as u32,
                beta: 0.0,
                symexp_lin_beta: 0.0,
                symexp_lin_scales: None,
            },
        )?;
        s.synchronize()?;
        Ok(Self { weight })
    }
}
