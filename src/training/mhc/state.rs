use super::*;
use crate::training::{
    optimizer_state::{Fp32AdamState, MuonState, StateInit},
    runtime::Runtime,
};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_TOKEN_ROWS, Gpt2Rng, LinearScratch, RowwiseNvfp4Buffers};
use rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors;
pub(in crate::training) struct Unit {
    pub phi: MuonState,
    pub fp: [Fp32AdamState; 2],
    pub grad: DeviceBuffer<f32>,
    pub fg: [DeviceBuffer<f32>; 2],
    pub branch: Option<DeviceBuffer<f32>>,
    pub names: [String; 6],
}
impl Unit {
    pub fn new(r: &Runtime, w: &UnitWeights, index: usize) -> Result<Self, DriverError> {
        let s = r.stream.as_ref();
        let init = StateInit::new(
            s,
            &r.decode,
            &r.optimizer,
            crate::training::symexp_lin::beta(),
        );
        let name = if w.head {
            "mhc.head".to_string()
        } else {
            format!("mhc.boundaries.{index}")
        };
        Ok(Self {
            phi: MuonState::new(init, &w.phi, WIDTH)?,
            fp: [
                Fp32AdamState::new(init, &w.bias)?,
                Fp32AdamState::new(init, &w.alpha)?,
            ],
            grad: DeviceBuffer::zeroed(s, PAD * WIDTH)?,
            fg: [
                DeviceBuffer::zeroed(s, w.active_rows())?,
                DeviceBuffer::zeroed(s, if w.head { 1 } else { 3 })?,
            ],
            branch: (!w.head)
                .then(|| DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS * GPT2_N_EMBD))
                .transpose()?,
            names: [
                "projection.weight",
                "bias",
                "scale",
                "projection.weight.exponential",
                "projection.weight.linear",
                "projection.weight.curvature",
            ]
            .map(|part| format!("{name}.{part}")),
        })
    }
    pub fn gradients(&self) -> Vec<(&str, &DeviceBuffer<f32>)> {
        self.names
            .iter()
            .map(String::as_str)
            .zip([
                &self.grad,
                &self.fg[0],
                &self.fg[1],
                &self.phi.symexp_lin.exponential.grad,
                &self.phi.symexp_lin.linear.grad,
                &self.phi.symexp_lin.curvature.grad,
            ])
            .collect()
    }
}
// Keep native FP4 operands and small projection controls from the original
// forward pass. Only inputs after the saved window start are reconstructed.
pub(in crate::training) struct ProjectionTape {
    pub input_q: RowwiseNvfp4Buffers,
    pub inverse: DeviceBuffer<f32>,
    pub raw: DeviceBuffer<f32>,
    pub coefficients: DeviceBuffer<f32>,
}
impl ProjectionTape {
    fn new(s: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            input_q: RowwiseNvfp4Buffers::gpt2_rows(s, GPT2_TOKEN_ROWS * WIDTH)?,
            inverse: DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS)?,
            raw: DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS * PAD)?,
            coefficients: DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS * 24)?,
        })
    }
}
pub(in crate::training) struct Checkpoint {
    pub index: usize,
    pub input: DeviceBuffer<f32>,
}
pub(in crate::training) struct Mhc {
    pub active_boundaries: usize,
    pub units: Vec<Unit>,
    pub module: rust_kernels_cuda::mhc::kernels::LoadedModule,
    pub coefficients: rust_kernels_cuda::mhc_coefficients::MhcCoefficientsModule,
    pub reverse: rust_kernels_cuda::mhc_reverse::MhcReverseModule,
    pub post_reuse: Option<MhcPostReuseModule>,
    pub window_post: Option<rust_kernels_cuda::mhc_window_post::MhcWindowPostModule>,
    pub rms_quant: rust_kernels_cuda::mhc_rms_quant::MhcRmsQuantModule,
    pub current: DeviceBuffer<f32>,
    pub dx: DeviceBuffer<f32>,
    pub checkpoints: Vec<Checkpoint>,
    pub replay_inputs: Vec<DeviceBuffer<f32>>,
    pub projections: Vec<ProjectionTape>,
    pub tape_units: [Option<usize>; WINDOW],
    pub active_window: Option<usize>,
    #[cfg(test)]
    pub recompute_verification: Option<super::recompute_verification::RecomputeVerification>,
    pub dc: DeviceBuffer<f32>,
    pub draw: DeviceBuffer<f32>,
    pub row_amax: DeviceBuffer<f32>,
    pub tma: TmaNvfp4DeviceScaleDescriptors,
    pub linear: LinearScratch,
    pub n: u32,
    pub rng: Gpt2Rng,
}
impl Mhc {
    pub fn new(r: &Runtime, w: &Weights) -> Result<Self, DriverError> {
        let s = r.stream.as_ref();
        let total = GPT2_TOKEN_ROWS * WIDTH;
        if WIDTH != 8192 {
            return Err(DriverError(1));
        }
        // Require the matching coefficient entries before allocating mHC state.
        let coefficients = r.mhc_coefficients_module()?;
        let reverse = r.mhc_reverse_module()?;
        let post_reuse = post_reuse_enabled()
            .then(|| r.mhc_post_reuse_module())
            .transpose()?;
        let window_post = window_post_enabled()
            .then(|| r.mhc_window_post_module())
            .transpose()?;
        let rms_quant = r.mhc_rms_quant_module()?;
        Ok(Self {
            active_boundaries: BOUNDARIES,
            units: w
                .units
                .iter()
                .enumerate()
                .map(|(i, w)| Unit::new(r, w, i))
                .collect::<Result<_, _>>()?,
            module: r.mhc_module()?,
            coefficients,
            reverse,
            post_reuse,
            window_post,
            rms_quant,
            current: DeviceBuffer::zeroed(s, total)?,
            dx: DeviceBuffer::zeroed(s, total)?,
            checkpoints: (0..=BOUNDARIES)
                .filter(|&index| index == window_start(index))
                .map(|index| {
                    Ok(Checkpoint {
                        index,
                        input: DeviceBuffer::zeroed(s, total)?,
                    })
                })
                .collect::<Result<_, DriverError>>()?,
            replay_inputs: (1..WINDOW)
                .map(|_| DeviceBuffer::zeroed(s, total))
                .collect::<Result<_, _>>()?,
            projections: (0..=BOUNDARIES)
                .map(|_| ProjectionTape::new(s))
                .collect::<Result<_, _>>()?,
            tape_units: [None; WINDOW],
            active_window: None,
            #[cfg(test)]
            recompute_verification: None,
            dc: DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS * 24)?,
            draw: DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS * PAD)?,
            row_amax: DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS)?,
            tma: TmaNvfp4DeviceScaleDescriptors::new(s)?,
            linear: LinearScratch::new(s, WIDTH, PAD)?,
            n: 0,
            rng: Gpt2Rng::new(0x4d48436a910735),
        })
    }
    pub fn named_gradients(&self) -> Vec<(&str, &DeviceBuffer<f32>)> {
        self.active_unit_indices()
            .flat_map(|index| self.units[index].gradients())
            .collect()
    }

    pub fn active_unit_indices(&self) -> impl Iterator<Item = usize> + use<> {
        (0..self.active_boundaries).chain(std::iter::once(BOUNDARIES))
    }
}
