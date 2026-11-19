//! Read the existing CUDA clip reductions; never change gradients or optimizer state.
use std::collections::{BTreeMap, HashSet};

use cuda_core::CudaStream;
use gpt2_nvfp4::{GPT2_N_EMBD, attention_trainable_qkv_dim};
use rust_kernels_cuda::nvfp4_tma_matmul::kernels::{TILE_M, TILE_N};
use serde_json::{Value, json};

use super::{GradientClipBuffers, layout};
use crate::{
    AppResult,
    training::{
        grads::BackwardBuffers, next_latent::NextLatGradBuffers, optimizer_muon::MuonPointerTables,
    },
};

impl GradientClipBuffers {
    pub(in crate::training) fn uses_cached_matrix_norm(&self, gradient: u64) -> bool {
        self.matrix_sum_count != 0 && !self.recomputed_matrix_ptrs.contains(&gradient)
    }

    pub(in crate::training) fn norm_components(
        &self,
        stream: &CudaStream,
        grads: &BackwardBuffers,
        next_latent: &NextLatGradBuffers,
        tables: &MuonPointerTables,
    ) -> AppResult<Vec<Value>> {
        let mut ranges = layout::named_parameter_ranges(grads, next_latent, self.active_depth);
        ranges.extend(self.extra_gradients.iter().cloned());
        let mut control_ptrs = HashSet::new();
        for desc in &tables.all.host_symexp_lin_slots {
            if desc.rows == 0 || desc.cols == 0 {
                continue;
            }
            let name = find_name(&ranges, desc.grad, (desc.rows * desc.cols) as usize)?;
            for (suffix, ptr) in [
                ("exponential", desc.exponential_grad),
                ("linear", desc.linear_grad),
                ("curvature", desc.curvature_grad),
            ] {
                if ptr != 0 {
                    control_ptrs.insert(ptr);
                    if !ranges.iter().any(|(_, start, _)| *start == ptr) {
                        ranges.push((format!("{name}.{suffix}"), ptr, 1));
                    }
                }
            }
        }
        let ptrs = self.chunk_ptrs.to_host_vec(stream)?;
        let lens = self.chunk_lens.to_host_vec(stream)?;
        let sums = self.chunk_sums.to_host_vec(stream)?;
        let matrix_sums = self.matrix_sums.to_host_vec(stream)?;
        let mut totals: BTreeMap<(String, &'static str), (usize, f64)> = BTreeMap::new();
        let mut full_attention_qkv_sections = BTreeMap::new();
        for ((ptr, len), sum) in ptrs.into_iter().zip(lens).zip(sums) {
            let name = find_name(&ranges, ptr, len as usize)?;
            let kind = if control_ptrs.contains(&ptr) {
                "symexp_control"
            } else {
                "ordinary_parameter"
            };
            let total = totals.entry((name, kind)).or_default();
            total.0 += len as usize;
            total.1 += f64::from(sum);
        }
        if self.matrix_sum_count != 0 {
            for (slot, desc) in tables.all.host_symexp_lin_slots.iter().enumerate() {
                if desc.rows == 0 || desc.cols == 0 || !self.uses_cached_matrix_norm(desc.grad) {
                    continue;
                }
                let len = (desc.rows * desc.cols) as usize;
                let name = find_name(&ranges, desc.grad, len)?;
                let start = self.matrix_sum_offsets[slot] as usize;
                let count = (desc.cols.div_ceil(TILE_M) * desc.rows.div_ceil(TILE_N)) as usize;
                let tile_sums = &matrix_sums[start..start + count];
                let square = sum_tiles(tile_sums);
                if name.ends_with(".attn_qkv.weight")
                    && desc.rows as usize == attention_trainable_qkv_dim(true)
                    && desc.cols as usize == GPT2_N_EMBD
                    && let Some(sections) = full_attention_qkv_squared_l2(tile_sums)
                {
                    full_attention_qkv_sections.insert(name.clone(), sections);
                }
                let total = totals.entry((name, "muon_matrix")).or_default();
                total.0 += len;
                total.1 += square;
            }
        }
        let square = totals.values().map(|v| v.1).sum::<f64>();
        let mut result = totals
            .into_iter()
            .map(|((name, kind), (values, sum))| {
                let mut component =
                    json!({"name":name, "kind":kind, "values":values, "squared_l2":sum,
                    "l2":sum.sqrt(), "share_of_squared_norm":sum/square,
                    "finite":sum.is_finite() && sum >= 0.0});
                if kind == "muon_matrix"
                    && let Some(sections) = full_attention_qkv_sections.remove(&name)
                {
                    component["qkv_sections_squared_l2"] = sections;
                }
                component
            })
            .collect::<Vec<_>>();
        result.sort_by(|a, b| {
            b["squared_l2"]
                .as_f64()
                .unwrap_or(f64::INFINITY)
                .total_cmp(&a["squared_l2"].as_f64().unwrap_or(f64::INFINITY))
        });
        Ok(result)
    }
}

fn sum_tiles(sums: &[f32]) -> f64 {
    sums.iter().map(|&value| f64::from(value)).sum()
}

fn full_attention_qkv_squared_l2(sums: &[f32]) -> Option<Value> {
    // Fused dW dispatch maps weight rows to grid Y and input columns to grid X.
    // Store order is blockIdx.y * gridDim.x + blockIdx.x.
    let tile_height = TILE_M as usize;
    let tiles_across = GPT2_N_EMBD.div_ceil(TILE_N as usize);
    let projection_tiles = (GPT2_N_EMBD / tile_height) * tiles_across;
    let total_tiles = attention_trainable_qkv_dim(true).div_ceil(tile_height) * tiles_across;
    if !GPT2_N_EMBD.is_multiple_of(tile_height) || sums.len() != total_tiles {
        return None;
    }

    let (query, rest) = sums.split_at(projection_tiles);
    let (key, rest) = rest.split_at(projection_tiles);
    let (value, head_gate) = rest.split_at(projection_tiles);
    Some(json!({
        "query": sum_tiles(query),
        "key": sum_tiles(key),
        "value": sum_tiles(value),
        "head_gate": (!head_gate.is_empty()).then(|| sum_tiles(head_gate)),
    }))
}

fn find_name(ranges: &[(String, u64, usize)], ptr: u64, len: usize) -> AppResult<String> {
    ranges
        .iter()
        .find_map(|(name, start, count)| {
            (ptr >= *start && ptr + len as u64 * 4 <= *start + *count as u64 * 4)
                .then(|| name.clone())
        })
        .ok_or_else(|| format!("unmapped native gradient range ptr={ptr:#x} len={len}").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_attention_qkv_sections_follow_dw_tile_rows() {
        let tile_height = TILE_M as usize;
        if !GPT2_N_EMBD.is_multiple_of(tile_height) {
            assert_eq!(full_attention_qkv_squared_l2(&[]), None);
            return;
        }
        let columns = GPT2_N_EMBD.div_ceil(TILE_N as usize);
        let rows = attention_trainable_qkv_dim(true).div_ceil(tile_height);
        let projection_rows = GPT2_N_EMBD / tile_height;
        let mut sums = vec![0.0; rows * columns];
        sums[columns - 1] = 1.0;
        sums[projection_rows * columns] = 2.0;
        sums[(3 * projection_rows - 1) * columns] = 4.0;
        let mut expected = json!({"query": 1.0, "key": 2.0, "value": 4.0, "head_gate": null});
        if rows > 3 * projection_rows {
            sums[3 * projection_rows * columns] = 8.0;
            expected["head_gate"] = json!(8.0);
        }
        assert_eq!(full_attention_qkv_squared_l2(&sums), Some(expected));
        assert_eq!(full_attention_qkv_squared_l2(&sums[1..]), None);
    }
}
