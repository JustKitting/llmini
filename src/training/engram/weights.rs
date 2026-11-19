use super::*;
use crate::{
    AppResult,
    training::runtime::Runtime,
    upload::{UploadedCanon, UploadedNvfp4},
};
use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{GPT2_N_EMBD, Gpt2Rng};
pub(crate) struct ModuleWeights {
    pub cfg: Config,
    pub nv: Vec<UploadedNvfp4>,
    pub fp: [UploadedCanon; 4],
    pub token_map: DeviceBuffer<u32>,
}
impl ModuleWeights {
    #[cfg(test)]
    pub fn new(r: &Runtime) -> AppResult<Self> {
        Self::new_with_config(r, Config::reference(0))
    }
    pub fn new_with_config(r: &Runtime, cfg: Config) -> AppResult<Self> {
        let s = r.stream.as_ref();
        let memory = cfg.memory();
        let streams = cfg.streams();
        let seed_offset = 10007 * cfg.initialization_layer() as u64;
        let map = llama2_tokenizer::TrainingTokenizer::from_default_assets()?.memory_id_map()?;
        let mut nv = vec![init_nv(
            r,
            cfg.table_len(),
            128,
            memory,
            true,
            13 + seed_offset,
        )?];
        for index in 0..=streams {
            // Preserve every legacy seed. Extra keys have independent seeds;
            // V remains one shared projection and keeps its original seed.
            let (ws, bs) = if index == streams {
                (29, 31)
            } else if index == 0 {
                (19, 23)
            } else {
                (101 + 2 * index as u64, 102 + 2 * index as u64)
            };
            nv.push(init_nv(
                r,
                memory * GPT2_N_EMBD,
                memory,
                memory,
                false,
                ws + seed_offset,
            )?);
            nv.push(init_nv(
                r,
                GPT2_N_EMBD,
                GPT2_N_EMBD,
                memory,
                false,
                bs + seed_offset,
            )?);
        }
        let uploaded = Self {
            nv,
            fp: [
                UploadedCanon::from_host(s, &vec![1.; streams * GPT2_N_EMBD])?,
                UploadedCanon::from_host(s, &vec![1.; streams * GPT2_N_EMBD])?,
                UploadedCanon::from_host(s, &vec![1.; streams * GPT2_N_EMBD])?,
                UploadedCanon::from_host(s, &vec![0.; streams * GPT2_N_EMBD * 4])?,
            ],
            token_map: DeviceBuffer::from_host(s, &map)?,
            cfg,
        };
        // The addressing map is a local host allocation, not persistent state.
        s.synchronize()?;
        Ok(uploaded)
    }
}
fn init_nv(
    r: &Runtime,
    len: usize,
    k: usize,
    memory: usize,
    gaussian: bool,
    seed: u64,
) -> AppResult<UploadedNvfp4> {
    let s = r.stream.as_ref();
    let mut rng = Gpt2Rng::new(seed);
    // Match the explicit reference nn.Embedding / nn.Linear distributions.
    // Host FP32 initialization only; real amax and NVFP4 packing run on CUDA.
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        let u = (rng.next_u32() as f32 + 1.) / 4294967296.;
        let v = rng.next_u32() as f32 / 4294967296.;
        if gaussian {
            let radius = (-2. * u.ln()).sqrt();
            let (sin, cos) = (v * std::f32::consts::TAU).sin_cos();
            values.push(radius * cos);
            if values.len() < len {
                values.push(radius * sin);
            }
        } else {
            let bound = 1. / (memory as f32).sqrt();
            values.push((2. * u - 1.) * bound);
            if values.len() < len {
                values.push((2. * v - 1.) * bound);
            }
        }
    }
    let master = DeviceBuffer::from_host(s, &values)?;
    let bytes = vec![0; len / 2];
    let scales = rust_kernels_cuda::nvfp4::csf::encode_host(&vec![0; len / 16], len.div_ceil(k), k);
    let mut out = UploadedNvfp4::from_host(s, &bytes, &scales, 0., len)?;
    let mut amax = DeviceBuffer::zeroed(s, 1)?;
    let mut chunks = DeviceBuffer::zeroed(s, len.div_ceil(128))?;
    r.optimizer.materialize_schedule_free(
        rust_kernels_cuda::optimizer::ScheduleFreeMaterializeArgs {
            stream: s,
            bytes: &mut out.bytes,
            scales: &mut out.scales,
            global_scale: &mut out.global_scale,
            z_master: &master,
            x_master: &master,
            amax: &mut amax,
            chunk_amax: &mut chunks,
            len: len as u32,
            scale_rows: len.div_ceil(k) as u32,
            scale_k: k as u32,
            beta: 0.,
            symexp_lin_beta: 0.,
            symexp_lin_scales: None,
        },
    )?;
    s.synchronize()?;
    Ok(out)
}
