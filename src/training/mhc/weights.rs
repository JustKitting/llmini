use super::*;
use crate::{
    AppResult,
    training::runtime::Runtime,
    upload::{UploadedCanon, UploadedNvfp4},
};
use gpt2_nvfp4::Gpt2Rng;
pub(crate) struct UnitWeights {
    pub phi: UploadedNvfp4,
    pub bias: UploadedCanon,
    pub alpha: UploadedCanon,
    pub head: bool,
}
impl UnitWeights {
    pub fn active_rows(&self) -> usize {
        if self.head { 4 } else { 24 }
    }
}
pub(crate) struct Weights {
    pub units: Vec<UnitWeights>,
    pub initialization_version: u32,
}
impl Weights {
    pub fn new(r: &Runtime, seed: u64) -> AppResult<Self> {
        let s = r.stream.as_ref();
        let mut units = Vec::new();
        for index in 0..=BOUNDARIES {
            let head = index == BOUNDARIES;
            let active = if head { 4 } else { 24 };
            let mut rng = Gpt2Rng::new(
                seed ^ 0x4d48432020261115 ^ (index as u64).wrapping_mul(0x9e3779b97f4a7c15),
            );
            let mut values = vec![0.; PAD * WIDTH];
            let mut at = 0;
            while at < active * WIDTH {
                let u = (rng.next_u32() as f32 + 1.) / 4294967296.;
                let v = rng.next_u32() as f32 / 4294967296.;
                let rad = (-2. * u.ln()).sqrt() * 0.02;
                let (sin, cos) = (v * std::f32::consts::TAU).sin_cos();
                values[at] = rad * cos;
                at += 1;
                if at < active * WIDTH {
                    values[at] = rad * sin;
                    at += 1;
                }
            }
            let master = DeviceBuffer::from_host(s, &values)?;
            let bytes = vec![0; PAD * WIDTH / 2];
            let scales =
                rust_kernels_cuda::nvfp4::csf::encode_host(&vec![0; PAD * WIDTH / 16], PAD, WIDTH);
            let mut phi = UploadedNvfp4::from_host(s, &bytes, &scales, 0., PAD * WIDTH)?;
            let mut amax = DeviceBuffer::zeroed(s, 1)?;
            let mut chunks = DeviceBuffer::zeroed(s, (PAD * WIDTH).div_ceil(128))?;
            r.optimizer.materialize_schedule_free(
                rust_kernels_cuda::optimizer::ScheduleFreeMaterializeArgs {
                    stream: s,
                    bytes: &mut phi.bytes,
                    scales: &mut phi.scales,
                    global_scale: &mut phi.global_scale,
                    z_master: &master,
                    x_master: &master,
                    amax: &mut amax,
                    chunk_amax: &mut chunks,
                    len: (PAD * WIDTH) as u32,
                    scale_rows: PAD as u32,
                    scale_k: WIDTH as u32,
                    beta: 0.,
                    symexp_lin_beta: 0.,
                    symexp_lin_scales: None,
                },
            )?;
            s.synchronize()?;
            units.push(UnitWeights {
                phi,
                bias: UploadedCanon::from_host(s, &vec![0.; active])?,
                alpha: UploadedCanon::from_host(
                    s,
                    &vec![if head { 1. } else { 0.01 }; if head { 1 } else { 3 }],
                )?,
                head,
            });
        }
        Ok(Self {
            units,
            initialization_version: 2,
        })
    }
    pub fn tensor_count(&self) -> u32 {
        1 + 3 * self.units.len() as u32
    }
}
