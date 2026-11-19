//! Prevalidated, byte-exact reload of the same complete Engram/mHC topology.
//! The exclusive model borrow keeps destinations alive until copies complete.
use std::{collections::HashMap, fs::File, marker::PhantomData, path::Path};

use super::super::format::CheckpointTensor;
use crate::{
    AppResult,
    upload::{UploadedModel, UploadedNvfp4},
};
use cuda_core::{CudaStream, DeviceBuffer};

mod bindings;

struct Copy {
    destination: u64,
    bytes: Vec<u8>,
}

pub(crate) struct PreparedReload<'a> {
    copies: Vec<Copy>,
    _model: PhantomData<&'a mut UploadedModel>,
}

impl PreparedReload<'_> {
    pub(crate) fn apply(self, stream: &CudaStream) -> AppResult {
        let copied = (|| -> AppResult {
            for copy in &self.copies {
                // All lengths/layouts and every destination have been validated.
                // The plan owns every source and exclusively borrows the model.
                unsafe {
                    cuda_core::memory::memcpy_htod_async(
                        copy.destination,
                        copy.bytes.as_ptr(),
                        copy.bytes.len(),
                        stream.cu_stream(),
                    )?;
                }
            }
            Ok(())
        })();
        // Drain earlier enqueues even when a later copy failed, before any
        // owned host source is dropped. No GPU allocation or conversion occurs.
        let synchronized = stream.synchronize();
        copied?;
        synchronized?;
        Ok(())
    }
}

struct Builder {
    tensors: HashMap<String, CheckpointTensor>,
    copies: Vec<Copy>,
    compatible_layout: bool,
}

impl Builder {
    fn take(&mut self, name: &str) -> AppResult<CheckpointTensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| format!("checkpoint is missing tensor {name}").into())
    }

    fn nv(
        &mut self,
        name: &str,
        target: &UploadedNvfp4,
        padding_start: Option<usize>,
    ) -> AppResult {
        let tensor = self.take(name)?;
        super::tensor::validate(name, &tensor)?;
        if tensor.len != target.len || tensor.bytes.len() != target.bytes.len() {
            return Err(format!("{name}: reload parameter shape mismatch").into());
        }
        if let Some(start) = padding_start {
            if tensor.bytes[start..].iter().any(|b| b & 0x77 != 0) {
                return Err(format!("mHC projection has nonzero physical padding: {name}").into());
            }
        }
        let k = u32::from_le_bytes(tensor.scales[0..4].try_into()?);
        let rows = u32::from_le_bytes(tensor.scales[4..8].try_into()?);
        if tensor.scales.len() != target.scales.len()
            || k != target.scale_k
            || rows != target.scale_rows
        {
            // Older supported bias layouts require the existing migration
            // loader. No copy is committed until every binding is checked.
            self.compatible_layout = false;
            return Ok(());
        }
        self.copies.push(Copy {
            destination: target.bytes.cu_deviceptr(),
            bytes: tensor.bytes,
        });
        self.copies.push(Copy {
            destination: target.scales.cu_deviceptr(),
            bytes: tensor.scales,
        });
        self.copies.push(Copy {
            destination: target.global_scale.cu_deviceptr(),
            bytes: tensor.global_scale.to_le_bytes().to_vec(),
        });
        Ok(())
    }

    fn fp(&mut self, name: &str, target: &DeviceBuffer<f32>) -> AppResult {
        let tensor = self.take(name)?;
        if tensor.len != target.len()
            || tensor.bytes.len() != target.len() * 4
            || !tensor.scales.is_empty()
        {
            return Err(format!("{name}: reload control shape mismatch").into());
        }
        self.copies.push(Copy {
            destination: target.cu_deviceptr(),
            bytes: tensor.bytes,
        });
        Ok(())
    }

    fn token_map(&mut self, name: &str, target: &DeviceBuffer<u32>) -> AppResult {
        let tensor = self.take(name)?;
        if tensor.len != target.len()
            || tensor.bytes.len() != target.len() * 4
            || !tensor.scales.is_empty()
        {
            return Err(format!("{name}: invalid Engram compressed ID map").into());
        }
        let mut bytes = Vec::with_capacity(tensor.bytes.len());
        for raw in tensor.bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(raw.try_into()?);
            if !value.is_finite()
                || value < 0.
                || value.fract() != 0.
                || value >= target.len() as f32
            {
                return Err(format!("{name}: invalid Engram compressed ID map").into());
            }
            bytes.extend_from_slice(&(value as u32).to_le_bytes());
        }
        self.copies.push(Copy {
            destination: target.cu_deviceptr(),
            bytes,
        });
        Ok(())
    }
}

pub(crate) fn prepare_reload<'a>(
    model: &'a mut UploadedModel,
    path: &Path,
) -> AppResult<Option<PreparedReload<'a>>> {
    let (Some(engram), Some(mhc)) = (&model.engram, &model.mhc) else {
        return Ok(None);
    };
    if model.latent_feedback.is_some()
        || engram
            .modules
            .iter()
            .any(|m| m.cfg.legacy || m.cfg.streams() != 4)
    {
        return Ok(None);
    }
    let mut tensors = super::read_tensors(File::open(path)?)?;
    let mut descriptors = vec![
        (
            "mhc.config".to_string(),
            crate::training::mhc::descriptor(mhc.initialization_version),
        ),
        (
            "engram.layers".to_string(),
            engram.modules.iter().map(|m| m.cfg.layer as f32).collect(),
        ),
    ];
    for module in &engram.modules {
        descriptors.push((
            format!("{}.config", module.cfg.prefix()),
            module.cfg.descriptor(),
        ));
    }
    for (name, expected) in &descriptors {
        let Some(tensor) = tensors.get(name) else {
            return Ok(None);
        };
        if !tensor.scales.is_empty()
            || tensor.len != expected.len()
            || tensor.bytes.len() != expected.len() * 4
            || !tensor
                .bytes
                .chunks_exact(4)
                .zip(expected)
                .all(|(raw, value)| raw == value.to_le_bytes())
        {
            return Ok(None);
        }
    }
    for (name, _) in descriptors {
        tensors.remove(&name);
    }
    let mut builder = Builder {
        tensors,
        copies: Vec::new(),
        compatible_layout: true,
    };
    bindings::bind(&mut builder, model)?;
    if !builder.tensors.is_empty() {
        return Err("checkpoint has unrecognized tensors".into());
    }
    if !builder.compatible_layout {
        return Ok(None);
    }
    Ok(Some(PreparedReload {
        copies: builder.copies,
        _model: PhantomData,
    }))
}
