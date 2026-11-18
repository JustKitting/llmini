//! Exhaustive persistent update-state traversal; bounded readback, no GPU clones.
use super::super::{
    Trainer,
    optimizer_state::{
        AdamState, BlockState, EmberState, Fp32AdamState, LayerNormState, LinearState, MuonState,
        NextLatState, TokenEmbeddingState,
    },
};
use crate::{
    AppResult,
    upload::{UploadedLayerNorm, UploadedLinear, UploadedNvfp4},
};
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
pub(super) const CHUNK_BYTES: usize = 8 * 1024 * 1024;
#[derive(Debug, PartialEq)]
struct Entry {
    ptr: u64,
    bytes: usize,
    sha: [u8; 32],
}
#[derive(Debug, PartialEq)]
pub(super) struct Snapshot {
    entries: BTreeMap<String, Entry>,
    // NV weights, FP32 weights, Adam, FP32 Adam, Muon, SEL scalar states, Ember.
    counts: [usize; 7],
    metadata: Value,
}
struct Reader<'a> {
    stream: &'a CudaStream,
    chunk: Vec<u8>,
    snapshot: Snapshot,
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
impl Reader<'_> {
    fn buffer<T: DeviceCopy>(&mut self, name: &str, b: &DeviceBuffer<T>) -> AppResult {
        let mut hash = Sha256::new();
        let mut offset = 0;
        while offset < b.num_bytes() {
            let count = (b.num_bytes() - offset).min(self.chunk.len());
            unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    self.chunk.as_mut_ptr(),
                    b.cu_deviceptr() + offset as u64,
                    count,
                    self.stream.cu_stream(),
                )?;
            }
            self.stream.synchronize()?;
            hash.update(&self.chunk[..count]);
            offset += count;
        }
        let entry = Entry {
            ptr: b.cu_deviceptr(),
            bytes: b.num_bytes(),
            sha: hash.finalize().into(),
        };
        assert!(
            self.snapshot.entries.insert(name.into(), entry).is_none(),
            "duplicate fingerprint name {name}"
        );
        Ok(())
    }
    fn nv(&mut self, name: &str, w: &UploadedNvfp4) -> AppResult {
        self.snapshot.counts[0] += 1;
        self.buffer(&format!("weight.{name}.payload"), &w.bytes)?;
        self.buffer(&format!("weight.{name}.scales"), &w.scales)?;
        self.buffer(&format!("weight.{name}.global_scale"), &w.global_scale)
    }
    fn fp_weight(&mut self, name: &str, w: &DeviceBuffer<f32>) -> AppResult {
        self.snapshot.counts[1] += 1;
        self.buffer(&format!("weight.{name}"), w)
    }
    fn adam(&mut self, name: &str, state: &AdamState) -> AppResult {
        self.snapshot.counts[2] += 1;
        let AdamState {
            z_master,
            x_master,
            first,
            second,
            schedule_amax,
        } = state;
        for (suffix, b) in [
            ("z", z_master),
            ("x", x_master),
            ("first", first),
            ("second", second),
            ("schedule_amax", schedule_amax),
        ] {
            self.buffer(&format!("optimizer.{name}.{suffix}"), b)?;
        }
        Ok(())
    }
    fn fp32(&mut self, name: &str, state: &Fp32AdamState) -> AppResult {
        self.snapshot.counts[3] += 1;
        let Fp32AdamState {
            z_master,
            x_master,
            first,
            second,
        } = state;
        for (suffix, b) in [
            ("z", z_master),
            ("x", x_master),
            ("first", first),
            ("second", second),
        ] {
            self.buffer(&format!("optimizer.{name}.{suffix}"), b)?;
        }
        Ok(())
    }
    fn ember(&mut self, name: &str, state: &EmberState) -> AppResult {
        self.snapshot.counts[6] += 1;
        let EmberState {
            z_master,
            x_master,
            row_second_moment,
            column_second_moment,
            column_partials,
            normalizer,
        } = state;
        for (suffix, buffer) in [
            ("z", z_master),
            ("x", x_master),
            ("row_second_moment", row_second_moment),
            ("column_second_moment", column_second_moment),
            ("column_partials", column_partials),
            ("normalizer", normalizer),
        ] {
            self.buffer(&format!("optimizer.{name}.{suffix}"), buffer)?;
        }
        Ok(())
    }
    fn muon(&mut self, name: &str, state: &MuonState) -> AppResult {
        self.snapshot.counts[4] += 1;
        let MuonState {
            z_master,
            x_master,
            momentum,
            variance,
            second_momentum,
            schedule_amax,
            symexp_lin,
        } = state;
        for (suffix, b) in [
            ("z", z_master),
            ("x", x_master),
            ("second_momentum", second_momentum),
            ("schedule_amax", schedule_amax),
        ] {
            self.buffer(&format!("optimizer.{name}.{suffix}"), b)?;
        }
        self.buffer(&format!("optimizer.{name}.momentum"), momentum)?;
        self.buffer(&format!("optimizer.{name}.variance"), variance)?;
        for (suffix, state) in [
            ("exponential", &symexp_lin.exponential),
            ("linear", &symexp_lin.linear),
            ("curvature", &symexp_lin.curvature),
        ] {
            self.snapshot.counts[5] += 1;
            for (field, b) in [
                ("z", &state.z_master),
                ("x", &state.x_master),
                ("first", &state.first),
                ("second", &state.second),
            ] {
                self.buffer(&format!("optimizer.{name}.{suffix}.{field}"), b)?;
            }
        }
        Ok(())
    }
    fn norm(&mut self, name: &str, w: &UploadedLayerNorm, state: &LayerNormState) -> AppResult {
        let LayerNormState { weight, bias } = state;
        self.nv(&format!("{name}.weight"), &w.weight)?;
        self.nv(&format!("{name}.bias"), &w.bias)?;
        self.adam(&format!("{name}.weight"), weight)?;
        self.adam(&format!("{name}.bias"), bias)
    }
    fn linear(&mut self, name: &str, w: &UploadedLinear, state: &LinearState) -> AppResult {
        let LinearState { weight_muon, bias } = state;
        self.nv(&format!("{name}.weight"), &w.weight)?;
        self.nv(&format!("{name}.bias"), &w.bias)?;
        self.muon(&format!("{name}.weight"), weight_muon)?;
        self.adam(&format!("{name}.bias"), bias)
    }
}
impl Snapshot {
    pub(super) fn capture(t: &Trainer) -> AppResult<Self> {
        let stream = t.runtime.stream.as_ref();
        stream.context().bind_to_thread()?;
        stream.synchronize()?;
        let state = &t.buffers.optimizer_state;
        let w = &t.uploaded;
        let mut r = Reader {
            stream,
            chunk: vec![0; CHUNK_BYTES],
            snapshot: Self {
                entries: BTreeMap::new(),
                counts: [0; 7],
                metadata: json!({}),
            },
        };
        r.nv("token_embedding", &w.token_embedding)?;
        match &state.token_embedding {
            TokenEmbeddingState::Adam(a) => r.adam("token_embedding", a)?,
            TokenEmbeddingState::Ember(e) => r.ember("token_embedding", e)?,
        }
        r.nv("xsa_alphas", &w.xsa_alphas)?;
        r.adam("xsa_alphas", &state.xsa_alphas)?;
        r.norm("ln_f", &w.ln_f, &state.ln_f)?;
        let NextLatState {
            norm,
            input_projection,
            transition,
            output_projection,
        } = &state.next_latent;
        r.norm("next_latent.norm", &w.next_latent.norm, norm)?;
        r.linear(
            "next_latent.input_projection",
            &w.next_latent.input_projection,
            input_projection,
        )?;
        r.linear(
            "next_latent.transition",
            &w.next_latent.transition,
            transition,
        )?;
        r.linear(
            "next_latent.output_projection",
            &w.next_latent.output_projection,
            output_projection,
        )?;
        for (i, (w, s)) in w.blocks.iter().zip(&state.blocks).enumerate() {
            let BlockState {
                ln_1,
                canon_a,
                attn_qkv,
                attn_qk_scale,
                attn_c_proj,
                ln_2,
                canon_c,
                mlp_up,
                mlp_down,
            } = s;
            let p = format!("blocks.{i}");
            r.norm(&format!("{p}.ln_1"), &w.ln_1, ln_1)?;
            r.fp_weight(&format!("{p}.canon_a"), &w.canon_a.weight)?;
            r.fp32(&format!("{p}.canon_a"), canon_a)?;
            r.linear(&format!("{p}.attn_qkv"), &w.attn_qkv, attn_qkv)?;
            r.nv(&format!("{p}.attn_qk_scale"), &w.attn_qk_scale)?;
            r.adam(&format!("{p}.attn_qk_scale"), attn_qk_scale)?;
            r.linear(&format!("{p}.attn_c_proj"), &w.attn_c_proj, attn_c_proj)?;
            r.norm(&format!("{p}.ln_2"), &w.ln_2, ln_2)?;
            r.fp_weight(&format!("{p}.canon_c"), &w.canon_c.weight)?;
            r.fp32(&format!("{p}.canon_c"), canon_c)?;
            r.linear(&format!("{p}.mlp_up"), &w.mlp_up, mlp_up)?;
            r.linear(&format!("{p}.mlp_down"), &w.mlp_down, mlp_down)?;
        }
        assert_eq!(t.engram.is_some(), w.engram.is_some());
        if let (Some(engram), Some(memory)) = (&t.engram, &w.engram) {
            assert_eq!((engram.modules.len(), memory.modules.len()), (2, 2));
            for (w, e) in memory.modules.iter().zip(&engram.modules) {
                assert_eq!(w.cfg.streams(), 4);
                let p = w.cfg.prefix();
                let names = w.cfg.nv_names();
                for (name, tensor) in names.iter().zip(&w.nv) {
                    r.nv(&format!("{p}.{name}"), tensor)?;
                }
                for (index, state) in w.cfg.adam_indices().into_iter().zip(&e.nv_states) {
                    r.adam(&format!("{p}.{}", names[index]), state)?;
                }
                for (index, state) in w
                    .cfg
                    .projector_indices()
                    .into_iter()
                    .zip(&e.projector_states)
                {
                    r.muon(&format!("{p}.{}", names[index]), state)?;
                }
                for (i, (tensor, state)) in w.fp.iter().zip(&e.fp_states).enumerate() {
                    r.fp_weight(&format!("{p}.fp.{i}"), &tensor.weight)?;
                    r.fp32(&format!("{p}.fp.{i}"), state)?;
                }
                r.buffer(&format!("{p}.token_map"), &w.token_map)?;
                r.buffer(&format!("{p}.primes"), &e.primes)?;
                r.buffer(&format!("{p}.offsets"), &e.offsets)?;
                r.buffer(&format!("{p}.multipliers"), &e.multipliers)?;
            }
        }
        assert_eq!(t.mhc.is_some(), w.mhc.is_some());
        if let (Some(mhc), Some(mw)) = (&t.mhc, &w.mhc) {
            assert_eq!((mhc.units.len(), mw.units.len()), (33, 33));
            for (i, (w, e)) in mw.units.iter().zip(&mhc.units).enumerate() {
                let p = format!("mhc.units.{i}");
                r.nv(&format!("{p}.phi"), &w.phi)?;
                r.muon(&format!("{p}.phi"), &e.phi)?;
                for (suffix, tensor, state) in
                    [("bias", &w.bias, &e.fp[0]), ("alpha", &w.alpha, &e.fp[1])]
                {
                    r.fp_weight(&format!("{p}.{suffix}"), &tensor.weight)?;
                    r.fp32(&format!("{p}.{suffix}"), state)?;
                }
            }
        }
        assert_eq!(t.latent_feedback.is_some(), w.latent_feedback.is_some());
        if let (Some(f), Some(fw)) = (&t.latent_feedback, &w.latent_feedback) {
            assert!(t.mhc.is_none() && t.engram.is_none());
            r.nv("latent_feedback.fusion.weight", &fw.weight)?;
            r.muon("latent_feedback.fusion.weight", &f.state)?;
        }
        r.buffer(
            "optimizer.adam_schedule_amax_ptrs",
            state.adam_schedule_amax_table().0,
        )?;
        r.snapshot.metadata = json!({"step":state.step,"average_weight_sum_bits":state.schedule_free_weight_sum.to_bits(),
            "training_weights_ready":state.training_weights_ready(),"scheduled_values_valid":state.scheduled_values_valid(),
            "base_seed":t.base_seed,"base_rng":format!("{:?}",t.rng),"mhc_rng":t.mhc.as_ref().map(|m|format!("{:?}",m.rng)),
            "latent_feedback":t.latent_feedback.as_ref().map(|f|f.integrity_metadata()),
            "engram_rng":t.engram.as_ref().map(|e|e.modules.iter().map(|m|format!("{:?}",m.rng)).collect::<Vec<_>>()),
            "engram_descriptors":w.engram.as_ref().map(|e|e.modules.iter().map(|m|m.cfg.descriptor().iter().map(|v|v.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>()),
            "embedding_optimizer":if matches!(&state.token_embedding,TokenEmbeddingState::Ember(_)){"Ember"}else{"Adam"},
            "mhc_initialization_version":w.mhc.as_ref().map(|m|m.initialization_version)});
        let has_memory = t.engram.is_some();
        let has_feedback = t.latent_feedback.is_some();
        assert!(t.mhc.is_some() || has_feedback);
        let ember = matches!(&state.token_embedding, TokenEmbeddingState::Ember(_));
        let mut expected = if has_memory {
            [275, 106, 164, 106, 110, 330, 1]
        } else if has_feedback {
            [221, 32, 152, 32, 68, 204, 1]
        } else {
            [253, 98, 152, 98, 100, 300, 1]
        };
        if !ember {
            expected[2] += 1;
            expected[6] = 0;
        }
        assert_eq!(
            r.snapshot.counts, expected,
            "persistent update-state coverage count"
        );
        assert_eq!(
            r.snapshot.entries.len(),
            (if has_memory {
                4170
            } else if has_feedback {
                2814
            } else {
                3816
            }) - usize::from(!ember),
            "every expected persistent buffer must be fingerprinted"
        );
        Ok(r.snapshot)
    }
    pub(super) fn assert_parameter_unchanged(&self, after: &Self, name: &str) -> usize {
        let weight = format!("weight.{name}.");
        let optimizer = format!("optimizer.{name}.");
        let mut checked = 0;
        for (key, entry) in &self.entries {
            if key.starts_with(&weight) || key.starts_with(&optimizer) {
                assert_eq!(
                    Some(entry),
                    after.entries.get(key),
                    "changed inactive {key}"
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
        checked
    }
    pub(super) fn json(&self) -> Value {
        let mut hash = Sha256::new();
        hash.update(serde_json::to_vec(&self.metadata).unwrap());
        let mut entries = serde_json::Map::new();
        let mut bytes = 0usize;
        for (name, e) in &self.entries {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(e.ptr.to_le_bytes());
            hash.update((e.bytes as u64).to_le_bytes());
            hash.update(e.sha);
            bytes += e.bytes;
            entries.insert(
                name.clone(),
                json!({"ptr":e.ptr,"bytes":e.bytes,"sha256":hex(&e.sha)}),
            );
        }
        json!({"root_sha256":hex(&hash.finalize()),"device_buffers":self.entries.len(),"device_bytes":bytes,"counts":{
            "NVFP4_tensors":self.counts[0],"FP32_parameter_tensors":self.counts[1],"Adam_states":self.counts[2],
            "FP32_Adam_states":self.counts[3],"Muon_states":self.counts[4],"SymExpLin_scalar_states":self.counts[5],"Ember_states":self.counts[6]},
            "metadata":self.metadata,"buffers":entries})
    }
}

#[test]
fn inactive_parameter_fingerprint_includes_packed_weight_and_optimizer_buffers() {
    fn sample() -> Snapshot {
        let name = "latent_feedback.fusion.weight";
        let mut entries = BTreeMap::new();
        let optimizer = [
            "z",
            "x",
            "second_momentum",
            "schedule_amax",
            "momentum",
            "variance",
        ]
        .into_iter()
        .map(|suffix| format!("optimizer.{name}.{suffix}"))
        .chain(
            ["exponential", "linear", "curvature"]
                .into_iter()
                .flat_map(|control| {
                    ["z", "x", "first", "second"]
                        .into_iter()
                        .map(move |field| format!("optimizer.{name}.{control}.{field}"))
                }),
        );
        let weights = ["payload", "scales", "global_scale"]
            .into_iter()
            .map(|field| format!("weight.{name}.{field}"));
        for (i, key) in weights.chain(optimizer).enumerate() {
            entries.insert(
                key,
                Entry {
                    ptr: 1024 + i as u64,
                    bytes: 4,
                    sha: [0; 32],
                },
            );
        }
        entries.insert(
            "weight.other.payload".into(),
            Entry {
                ptr: 9000,
                bytes: 4,
                sha: [0; 32],
            },
        );
        Snapshot {
            entries,
            counts: [0; 7],
            metadata: json!({}),
        }
    }
    let before = sample();
    assert_eq!(
        before.assert_parameter_unchanged(&sample(), "latent_feedback.fusion.weight"),
        21
    );
    for field in ["payload", "scales", "global_scale"] {
        let mut changed = sample();
        changed
            .entries
            .get_mut(&format!("weight.latent_feedback.fusion.weight.{field}"))
            .unwrap()
            .sha[0] = 1;
        assert!(
            std::panic::catch_unwind(|| {
                before.assert_parameter_unchanged(&changed, "latent_feedback.fusion.weight");
            })
            .is_err(),
            "packed weight mutation escaped fingerprint: {field}"
        );
    }
}
