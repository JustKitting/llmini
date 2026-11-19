//! All layers, exact configurations and tokenizer maps live in the native model file.
use super::{
    format::{CheckpointTensor, CheckpointWriter},
    load::tensor::{take_uploaded, take_uploaded_fp32_len},
    save::tensor::{write as write_nv, write_fp32},
};
use crate::{
    AppResult,
    training::engram::{Config, ModuleWeights, Weights},
};
use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::GPT2_N_EMBD;
use std::{collections::HashMap, io::Write};
const FP: [&str; 4] = ["query_norm", "key_norm", "convolution_norm", "convolution"];
fn write_values(
    writer: &mut CheckpointWriter<impl Write>,
    name: &str,
    values: &[f32],
) -> AppResult {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    writer.write_tensor(name, values.len(), 1., &bytes, &[])
}
fn read_values(tensors: &mut HashMap<String, CheckpointTensor>, name: &str) -> AppResult<Vec<f32>> {
    let t = tensors
        .remove(name)
        .ok_or_else(|| format!("missing {name}"))?;
    if !t.scales.is_empty() || t.bytes.len() != t.len * 4 {
        return Err(format!("invalid {name}").into());
    }
    Ok(t.bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
pub(super) fn write(
    writer: &mut CheckpointWriter<impl Write>,
    s: &CudaStream,
    w: &Weights,
) -> AppResult {
    if !(w.modules.len() == 1 && w.modules[0].cfg.legacy) {
        write_values(
            writer,
            "engram.layers",
            &w.modules
                .iter()
                .map(|w| w.cfg.layer as f32)
                .collect::<Vec<_>>(),
        )?;
    }
    for w in &w.modules {
        let prefix = w.cfg.prefix();
        for (name, t) in w.cfg.nv_names().into_iter().zip(&w.nv) {
            write_nv(writer, s, &format!("{prefix}.{name}"), t)?;
        }
        for (name, t) in FP.into_iter().zip(&w.fp) {
            write_fp32(writer, s, &format!("{prefix}.{name}"), t)?;
        }
        write_values(
            writer,
            &format!("{prefix}.token_id_map"),
            &w.token_map
                .to_host_vec(s)?
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>(),
        )?;
        write_values(writer, &format!("{prefix}.config"), &w.cfg.descriptor())?;
    }
    Ok(())
}
pub(super) fn load(
    s: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
) -> AppResult<Weights> {
    let legacy = tensors.contains_key("engram.config");
    let layers = if legacy {
        vec![1.]
    } else {
        read_values(tensors, "engram.layers")?
    };
    if layers.is_empty()
        || layers.len() > gpt2_nvfp4::GPT2_N_LAYER
        || layers.iter().any(|v| {
            !v.is_finite() || *v < 0. || v.fract() != 0. || *v >= gpt2_nvfp4::GPT2_N_LAYER as f32
        })
    {
        return Err("invalid Engram layers".into());
    }
    let mut modules = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut primes = std::collections::HashSet::new();
    for layer in layers {
        if !seen.insert(layer as usize) {
            return Err("duplicate Engram insertion point".into());
        }
        let prefix = if legacy {
            "engram".to_string()
        } else {
            format!("engram.layers.{}", layer as usize)
        };
        let cfg = Config::parse(&read_values(tensors, &format!("{prefix}.config"))?, legacy)?;
        if cfg.layer != layer as usize || cfg.primes.iter().any(|p| !primes.insert(*p)) {
            return Err("inconsistent Engram layer/hash tables".into());
        }
        let mut nv = Vec::new();
        for (index, name) in cfg.nv_names().into_iter().enumerate() {
            let len = if index == 0 {
                cfg.table_len()
            } else if index % 2 == 1 {
                cfg.memory() * GPT2_N_EMBD
            } else {
                GPT2_N_EMBD
            };
            let mut t = take_uploaded(s, tensors, &format!("{prefix}.{name}"))?;
            if t.len != len {
                return Err(format!("Engram {name} shape mismatch").into());
            }
            if name.ends_with(".bias") && t.scale_rows != 1 {
                let old = t.scales.to_host_vec(s)?;
                let logical = (0..t.len / 16)
                    .map(|g| rust_kernels_cuda::nvfp4::csf::scale_at(&old, g))
                    .collect::<Vec<_>>();
                let vector = rust_kernels_cuda::nvfp4::csf::encode_host(&logical, 1, t.len);
                eprintln!(
                    "engram_bias_scale_layout_migrated tensor={prefix}.{name} from_rows={} from_k={} to_rows=1 to_k={} payload_unchanged=true",
                    t.scale_rows, t.scale_k, t.len
                );
                t.scales = DeviceBuffer::from_host(s, &vector)?;
                s.synchronize()?;
                t.scale_rows = 1;
                t.scale_k = t.len as u32;
            }
            nv.push(t);
        }
        let mut fp = Vec::new();
        for (name, len) in FP.into_iter().zip(
            [GPT2_N_EMBD, GPT2_N_EMBD, GPT2_N_EMBD, GPT2_N_EMBD * 4].map(|n| n * cfg.streams()),
        ) {
            fp.push(take_uploaded_fp32_len(
                s,
                tensors,
                &format!("{prefix}.{name}"),
                len,
            )?);
        }
        let values = read_values(tensors, &format!("{prefix}.token_id_map"))?;
        if values.len() != llama2_tokenizer::MODEL_VOCAB_SIZE
            || values
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || v.fract() != 0. || *v >= values.len() as f32)
        {
            return Err("invalid Engram compressed ID map".into());
        }
        let ids = values.iter().map(|v| *v as u32).collect::<Vec<_>>();
        let token_map = DeviceBuffer::from_host(s, &ids)?;
        s.synchronize()?;
        modules.push(ModuleWeights {
            cfg,
            nv,
            fp: fp.try_into().ok().unwrap(),
            token_map,
        });
    }
    Ok(Weights { modules })
}
