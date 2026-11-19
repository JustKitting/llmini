use super::*;
use crate::{
    AppResult,
    training::{optimizer::OptimizerScratch, runtime::Runtime},
};
pub(crate) struct Weights {
    pub modules: Vec<ModuleWeights>,
}
impl Weights {
    pub fn new(r: &Runtime) -> AppResult<Self> {
        Ok(Self {
            modules: (0..LAYERS.len())
                .map(|i| ModuleWeights::new_with_config(r, Config::reference(i)))
                .collect::<AppResult<_>>()?,
        })
    }
    #[cfg(test)]
    pub fn new_multistream(r: &Runtime) -> AppResult<Self> {
        Self::new_multistream_with_options(r, MultistreamOptions::default())
    }

    pub fn new_multistream_with_options(r: &Runtime, options: MultistreamOptions) -> AppResult<Self> {
        Ok(Self {
            modules: options.configs()?
                .into_iter()
                .map(|cfg| ModuleWeights::new_with_config(r, cfg))
                .collect::<AppResult<_>>()?,
        })
    }
    pub fn streams(&self) -> Option<usize> {
        let streams = self.modules.first()?.cfg.streams();
        if streams == 4 {
            let [first, second] = self.modules.as_slice() else {
                return None;
            };
            if !config::valid_multistream_layers(first.cfg.layer, second.cfg.layer) {
                return None;
            }
        }
        self.modules
            .iter()
            .all(|w| w.cfg.streams() == streams)
            .then_some(streams)
    }
    pub fn tensor_count(&self) -> u32 {
        if self.modules.len() == 1 && self.modules[0].cfg.legacy {
            11
        } else {
            1 + self
                .modules
                .iter()
                .map(|w| w.nv.len() as u32 + 6)
                .sum::<u32>()
        }
    }
}
pub(in crate::training) struct Engram {
    pub modules: Vec<Module>,
    backward_scratch: super::backward_scratch::BackwardScratch,
}
impl Engram {
    pub fn new(r: &Runtime, w: &Weights) -> Result<Self, DriverError> {
        Ok(Self {
            backward_scratch: super::backward_scratch::BackwardScratch::new(
                &r.stream,
                w.modules
                    .iter()
                    .map(|w| w.cfg.memory())
                    .max()
                    .unwrap_or(MEMORY),
                w.modules.iter().map(|w| w.cfg.streams()).max().unwrap_or(1),
            )?,
            modules: w
                .modules
                .iter()
                .map(|w| Module::new(r, w))
                .collect::<Result<_, _>>()?,
        })
    }
    pub fn named_gradients(&self) -> Vec<(&str, &DeviceBuffer<f32>)> {
        self.modules
            .iter()
            .flat_map(|e| e.named_gradients())
            .collect()
    }
    pub fn forward_at(
        &mut self,
        index: usize,
        r: &Runtime,
        w: &Weights,
        tokens: &DeviceBuffer<u32>,
        hidden: &mut gpt2_nvfp4::HiddenStateDevice<'_>,
    ) -> Result<(), DriverError> {
        for (e, w) in self.modules.iter_mut().zip(&w.modules) {
            if e.cfg.layer == index {
                e.forward(r, w, tokens, hidden)?;
            }
        }
        Ok(())
    }
    pub fn backward_at(
        &mut self,
        index: usize,
        r: &Runtime,
        w: &Weights,
        gradient: &mut DeviceBuffer<f32>,
    ) -> Result<bool, DriverError> {
        let mut applied = false;
        for (e, w) in self.modules.iter_mut().zip(&w.modules).rev() {
            if e.cfg.layer == index {
                e.backward(r, w, gradient, &mut self.backward_scratch)?;
                applied = true;
            }
        }
        Ok(applied)
    }
    pub fn forward_at_streams(
        &mut self,
        index: usize,
        r: &Runtime,
        w: &Weights,
        tokens: &DeviceBuffer<u32>,
        streams: &mut DeviceBuffer<f32>,
        n: u32,
        seq: u32,
    ) -> Result<(), DriverError> {
        for (e, w) in self.modules.iter_mut().zip(&w.modules) {
            if e.cfg.layer == index {
                e.forward_streams(r, w, tokens, streams, n, seq)?;
            }
        }
        Ok(())
    }
    pub fn materialize_training(
        &mut self,
        r: &Runtime,
        w: &mut Weights,
        s: &mut OptimizerScratch,
        beta: f32,
    ) -> Result<(), DriverError> {
        for (e, w) in self.modules.iter_mut().zip(&mut w.modules) {
            e.materialize_training(r, w, s, beta)?;
        }
        Ok(())
    }
    pub fn materialize_evaluation(
        &mut self,
        r: &Runtime,
        w: &mut Weights,
        s: &mut OptimizerScratch,
    ) -> Result<(), DriverError> {
        for (e, w) in self.modules.iter_mut().zip(&mut w.modules) {
            e.materialize_evaluation(r, w, s)?;
        }
        Ok(())
    }
    pub fn update(
        &mut self,
        r: &Runtime,
        w: &mut Weights,
        s: &mut OptimizerScratch,
        step: u32,
        average: f32,
        scale: f32,
    ) -> Result<(), DriverError> {
        for (e, w) in self.modules.iter_mut().zip(&mut w.modules) {
            e.update(r, w, s, step, average, scale)?;
        }
        Ok(())
    }
}
