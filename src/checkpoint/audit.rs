//! Inspection of stored checkpoint values; no training or optimizer updates.
use super::format::{CheckpointReader, CheckpointTensor};
use crate::AppResult;
use rust_kernels_cuda::nvfp4::csf;
use std::{fs::File, io::BufReader, path::Path};

#[derive(Default)]
struct Stats {
    zeros: usize,
    nonfinite: usize,
    count: usize,
    payload_zeros: usize,
    zero_scale_groups: usize,
    sum: f64,
    sumsq: f64,
    amax: f64,
}
impl Stats {
    fn value(&mut self, value: f64) {
        self.count += 1;
        if !value.is_finite() {
            self.nonfinite += 1;
            return;
        }
        self.zeros += usize::from(value == 0.0);
        self.sum += value;
        self.sumsq += value * value;
        self.amax = self.amax.max(value.abs());
    }
    fn json(&self) -> serde_json::Value {
        let finite = self.count - self.nonfinite;
        serde_json::json!({
            "values": self.count, "zero_values": self.zeros,
            "zero_fraction": self.zeros as f64 / self.count.max(1) as f64,
            "nonfinite_values": self.nonfinite,
            "mean_finite_values": self.sum / finite.max(1) as f64,
            "rms_finite_values": (self.sumsq / finite.max(1) as f64).sqrt(),
            "max_abs_finite_value": self.amax,
            "zero_fp4_payload_codes": self.payload_zeros,
            "zero_local_scale_groups": self.zero_scale_groups,
        })
    }
}

pub(crate) fn audit_model(path: &Path) -> AppResult<serde_json::Value> {
    let mut reader = CheckpointReader::new(BufReader::new(File::open(path)?));
    let count = reader.read_header()?;
    let mut tensors = Vec::new();
    let mut total = 0;
    let mut nonfinite = 0;
    for _ in 0..count {
        let tensor = reader.read_tensor()?;
        let stats = inspect(&tensor)?;
        total += stats.count;
        nonfinite += stats.nonfinite;
        tensors.push(serde_json::json!({
            "name": tensor.name, "allocated_values": tensor.len,
            "format": if tensor.scales.is_empty() { "existing_fp32_tensor" } else { "nvfp4_csf" },
            "csf_k": if tensor.scales.is_empty() { None } else { Some(u32::from_le_bytes(tensor.scales[..4].try_into()?)) },
            "csf_rows": if tensor.scales.is_empty() { None } else { Some(u32::from_le_bytes(tensor.scales[4..8].try_into()?)) },
            "global_scale": tensor.global_scale,
            "statistics": stats.json(),
        }));
    }
    Ok(serde_json::json!({
        "checkpoint": path, "tensor_count": count, "allocated_values": total,
        "nonfinite_values": nonfinite, "tensors": tensors,
        "purpose": "Inspect represented checkpoint values only; no training, no optimizer updates.",
    }))
}

fn inspect(tensor: &CheckpointTensor) -> AppResult<Stats> {
    let mut stats = Stats::default();
    if tensor.scales.is_empty() {
        if tensor.bytes.len() != tensor.len * 4 {
            return Err("invalid existing FP32 tensor length".into());
        }
        for bytes in tensor.bytes.chunks_exact(4) {
            stats.value(f64::from(f32::from_le_bytes(bytes.try_into()?)));
        }
        return Ok(stats);
    }
    if tensor.len % 16 != 0 || tensor.bytes.len() != tensor.len / 2 || tensor.scales.len() < 128 {
        return Err("invalid NVFP4 checkpoint tensor length".into());
    }
    let scales = &tensor.scales;
    let k = u32::from_le_bytes(scales[..4].try_into()?) as usize;
    let rows = u32::from_le_bytes(scales[4..8].try_into()?) as usize;
    if k < 128
        || !k.is_multiple_of(128)
        || rows == 0
        || rows.checked_mul(k).is_none_or(|n| n < tensor.len)
    {
        return Err("invalid CSF checkpoint shape".into());
    }
    let needed = if rows == 1 {
        if scales[8] != 255 {
            return Err("invalid vector CSF header".into());
        }
        128 + k / 16
    } else {
        if ![7, 8].contains(&scales[9]) || scales[8] != (k / 16).trailing_zeros() as u8 {
            return Err("invalid matrix CSF header".into());
        }
        128 + rows.div_ceil(128) * (k / 128) * (768 + 128 * usize::from(scales[9] - 6))
    };
    if scales.len() < needed {
        return Err("truncated CSF checkpoint scales".into());
    }
    let fp4 = [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut fp8 = [0.0_f64; 256];
    for (code, value) in fp8.iter_mut().enumerate() {
        let exponent = (code >> 3) & 15;
        let mantissa = code & 7;
        let sign = if code & 128 == 0 { 1.0 } else { -1.0 };
        *value = if exponent == 15 && mantissa == 7 {
            f64::NAN
        } else if exponent == 0 {
            sign * mantissa as f64 / 512.0
        } else {
            sign * (1.0 + mantissa as f64 / 8.0) * 2.0_f64.powi(exponent as i32 - 7)
        };
    }
    for (group, packed) in tensor.bytes.chunks_exact(8).enumerate() {
        let scale = f64::from(tensor.global_scale) * fp8[usize::from(csf::scale_at(scales, group))];
        stats.zero_scale_groups += usize::from(scale == 0.0);
        for &byte in packed {
            for nibble in [byte & 15, byte >> 4] {
                let sign = if nibble & 8 == 0 { 1.0 } else { -1.0 };
                stats.payload_zeros += usize::from(nibble & 7 == 0);
                stats.value(sign * fp4[usize::from(nibble & 7)] * scale);
            }
        }
    }
    Ok(stats)
}
