use cuda_core::DriverError;

use super::args::Gpt2ForwardArgs;
use super::weights::Gpt2Weights;
use crate::Gpt2Rng;
use crate::depth_growth::LayerLayout;
use crate::types::{HiddenStateDevice, TokenEmbeddingArgs};

#[derive(Clone, Debug, Default)]
pub struct Gpt2 {
    weights: Option<Gpt2Weights>,
}

impl Gpt2 {
    /// Run an active prefix of a preallocated stack. The readout boundary is
    /// reported immediately after its last block.
    pub fn forward_at_depth<'a>(
        &self,
        layout: impl Into<LayerLayout>,
        args: Gpt2ForwardArgs<'a>,
        hook: &mut dyn FnMut(usize, &mut HiddenStateDevice<'_>) -> Result<(), DriverError>,
        readout: bool,
        boundary: &mut dyn FnMut(
            usize,
            bool,
            &mut HiddenStateDevice<'_>,
        ) -> Result<bool, DriverError>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        super::forward::forward_at_depth(
            self.weights().expect("Gpt2 must be initialized"),
            layout.into(),
            args,
            hook,
            readout,
            boundary,
        )
    }

    pub fn forward_with_boundaries<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
        hook: &mut dyn FnMut(usize, &mut HiddenStateDevice<'_>) -> Result<(), DriverError>,
        readout: bool,
        boundary: &mut dyn FnMut(
            usize,
            bool,
            &mut HiddenStateDevice<'_>,
        ) -> Result<bool, DriverError>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        super::forward::forward_with_boundaries(
            self.weights().expect("Gpt2 must be initialized"),
            args,
            hook,
            readout,
            boundary,
        )
    }
    pub const fn new() -> Self {
        Self { weights: None }
    }

    pub fn init(&mut self, seed: u64) {
        let mut rng = Gpt2Rng::new(seed);
        self.init_from_rng(&mut rng);
    }

    pub fn init_from_rng(&mut self, rng: &mut Gpt2Rng) {
        self.weights = Some(Gpt2Weights::init(rng, false));
    }

    /// Initialize the starting stack of a fresh LIDAS run.
    pub fn init_for_depth_growth(&mut self, seed: u64) {
        let mut rng = Gpt2Rng::new(seed);
        self.weights = Some(Gpt2Weights::init(&mut rng, true));
    }

    pub fn weights(&self) -> Option<&Gpt2Weights> {
        self.weights.as_ref()
    }

    pub fn weights_mut(&mut self) -> Option<&mut Gpt2Weights> {
        self.weights.as_mut()
    }

    pub fn forward_embeddings<'a>(
        &self,
        args: TokenEmbeddingArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.weights()
            .expect("Gpt2::init must be called before forward_embeddings")
            .forward_embeddings(args)
    }

    pub fn forward<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.weights()
            .expect("Gpt2::init must be called before forward")
            .forward(args)
    }

    pub fn forward_with_hook<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
        hook: &mut dyn FnMut(usize, &mut HiddenStateDevice<'_>) -> Result<(), DriverError>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.weights()
            .expect("Gpt2 must be initialized")
            .forward_with_hook(args, hook)
    }

    /// Run embeddings and every block without final normalization/head (detached preliminary pass).
    pub fn forward_hidden_with_hook<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
        hook: &mut dyn FnMut(usize, &mut HiddenStateDevice<'_>) -> Result<(), DriverError>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        super::forward::forward_impl(
            self.weights().expect("Gpt2 must be initialized"),
            args,
            hook,
            false,
        )
    }
}
