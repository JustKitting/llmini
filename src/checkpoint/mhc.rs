use super::{
    format::{CheckpointTensor, CheckpointWriter},
    load::tensor::{take_uploaded, take_uploaded_fp32_len},
    save::tensor::{write as write_nv, write_fp32},
};
use crate::{
    AppResult,
    training::mhc::{self, UnitWeights, Weights},
};
use cuda_core::CudaStream;
use std::{collections::HashMap, io::Write};
pub(super) fn write(
    writer: &mut CheckpointWriter<impl Write>,
    stream: &CudaStream,
    w: &Weights,
) -> AppResult {
    let config = mhc::descriptor(w.initialization_version);
    let bytes = config
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    writer.write_tensor("mhc.config", config.len(), 1., &bytes, &[])?;
    for (index, u) in w.units.iter().enumerate() {
        let prefix = if u.head {
            "mhc.head".to_string()
        } else {
            format!("mhc.boundaries.{index}")
        };
        write_nv(
            writer,
            stream,
            &format!("{prefix}.projection.weight"),
            &u.phi,
        )?;
        write_fp32(writer, stream, &format!("{prefix}.bias"), &u.bias)?;
        write_fp32(writer, stream, &format!("{prefix}.scale"), &u.alpha)?;
    }
    Ok(())
}
pub(super) fn load(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
) -> AppResult<Weights> {
    let config = tensors
        .remove("mhc.config")
        .ok_or("missing mHC graph descriptor")?;
    let initialization_version = [1, 2]
        .into_iter()
        .find(|&version| {
            let bytes = mhc::descriptor(version)
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            config.len == bytes.len() / 4 && config.scales.is_empty() && config.bytes == bytes
        })
        .ok_or("unsupported mHC graph descriptor")?;
    let mut units = Vec::new();
    for index in 0..=mhc::BOUNDARIES {
        let head = index == mhc::BOUNDARIES;
        let active = if head { 4 } else { 24 };
        let prefix = if head {
            "mhc.head".to_string()
        } else {
            format!("mhc.boundaries.{index}")
        };
        let phi = take_uploaded(stream, tensors, &format!("{prefix}.projection.weight"))?;
        if phi.len != mhc::PAD * mhc::WIDTH
            || phi.scale_rows as usize != mhc::PAD
            || phi.scale_k as usize != mhc::WIDTH
        {
            return Err(format!("invalid mHC projection shape {prefix}").into());
        }
        let payload = phi.bytes.to_host_vec(stream)?;
        if payload[active * mhc::WIDTH / 2..]
            .iter()
            .any(|b| b & 0x77 != 0)
        {
            return Err(format!("mHC projection has nonzero physical padding: {prefix}").into());
        }
        units.push(UnitWeights {
            phi,
            bias: take_uploaded_fp32_len(stream, tensors, &format!("{prefix}.bias"), active)?,
            alpha: take_uploaded_fp32_len(
                stream,
                tensors,
                &format!("{prefix}.scale"),
                if head { 1 } else { 3 },
            )?,
            head,
        });
    }
    Ok(Weights {
        units,
        initialization_version,
    })
}
