use std::sync::Arc;

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::BackendTypes;

use super::super::{BurnInnerBackend, CudaValidInput};

#[derive(Clone)]
pub(in crate::training) struct CudaValidationInput {
    pub(in crate::training::launch) tokens: Arc<Vec<u16>>,
    pub(in crate::training::launch) window_count: usize,
    pub(in crate::training::launch) target_byte_count: usize,
}

#[derive(Clone)]
pub(in crate::training::launch) struct CudaValidDataLoader {
    input: CudaValidationInput,
}

impl CudaValidDataLoader {
    pub(in crate::training::launch) fn new(
        tokens: Vec<u16>,
        window_count: usize,
        target_byte_count: usize,
    ) -> Self {
        Self {
            input: CudaValidationInput {
                tokens: Arc::new(tokens),
                window_count,
                target_byte_count,
            },
        }
    }
}

impl DataLoader<BurnInnerBackend, CudaValidInput> for CudaValidDataLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<CudaValidInput> + 'a> {
        Box::new(CudaValidIterator {
            input: self.input.clone(),
            produced: false,
        })
    }

    fn num_items(&self) -> usize {
        self.input.window_count
    }

    fn to_device(
        &self,
        _device: &<BurnInnerBackend as BackendTypes>::Device,
    ) -> Arc<dyn DataLoader<BurnInnerBackend, CudaValidInput>> {
        Arc::new(self.clone())
    }

    fn slice(
        &self,
        _start: usize,
        _end: usize,
    ) -> Arc<dyn DataLoader<BurnInnerBackend, CudaValidInput>> {
        Arc::new(self.clone())
    }
}

struct CudaValidIterator {
    input: CudaValidationInput,
    produced: bool,
}

impl Iterator for CudaValidIterator {
    type Item = CudaValidInput;

    fn next(&mut self) -> Option<Self::Item> {
        if self.produced {
            return None;
        }
        self.produced = true;
        Some(Ok(self.input.clone()))
    }
}

impl DataLoaderIterator<CudaValidInput> for CudaValidIterator {
    fn progress(&self) -> Progress {
        Progress::new(usize::from(self.produced), 1)
    }
}
