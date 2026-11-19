//! Native complete-weight serialization E2E; no training or reduced model.
use crate::{AppResult, training::runtime::Runtime};
use sha2::{Digest, Sha256};
use std::io::Write;

struct HashOutput<'a>(&'a mut Sha256);
impl Write for HashOutput<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
#[ignore = "actual full1B baseline + complete version4 Engram +32 mHC boundary/head weight serialization, CUDA, zero updates"]
fn native_complete_engram_mhc_checkpoint_387_tensors_roundtrip() -> AppResult {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let baseline = root.join("target/runs/20261114_080934Z_fineweb_21600s/model.ckpt");
    crate::reference_validation::require_hash(
        &baseline,
        "0342618e641c51ce1a077e397695db35254beb4c3b1ec342602a749aed68a896",
    )?;
    let runtime = Runtime::new()?;
    let stream = runtime.stream.as_ref();
    let (mut model, depth) = super::load_uploaded_model(stream, &baseline)?;
    assert_eq!(depth, gpt2_nvfp4::GPT2_N_LAYER.into());
    assert!(model.engram.is_none() && model.mhc.is_none() && model.latent_feedback.is_none());
    model.mhc = Some(crate::training::mhc::Weights::new(&runtime, 0x47505432)?);
    model.engram = Some(crate::training::engram::Weights::new_multistream(&runtime)?);
    let path = root.join("target/checks/20261115/engram-mhc-complete-weight-serialization.ckpt");
    super::save_uploaded_model(stream, &model, depth, &path)?;
    let original_hash = crate::reference_validation::sha256(&path)?;
    drop(model);
    let (loaded, restored_depth) = super::load_uploaded_model(stream, &path)?;
    assert_eq!(restored_depth, depth);
    assert!(loaded.latent_feedback.is_none());
    let memory = loaded.engram.as_ref().unwrap();
    assert_eq!(memory.streams(), Some(4));
    assert_eq!(memory.tensor_count(), 35);
    assert_eq!(
        memory
            .modules
            .iter()
            .map(|w| w.cfg.layer)
            .collect::<Vec<_>>(),
        vec![1, 5]
    );
    assert_eq!(loaded.mhc.as_ref().unwrap().tensor_count(), 100);
    let mut hash = Sha256::new();
    super::save::write_uploaded_model_to(stream, &loaded, depth, HashOutput(&mut hash))?;
    let restored_hash = format!("{:x}", hash.finalize());
    assert_eq!(
        original_hash, restored_hash,
        "every base/memory/mHC payload, scale, graph descriptor and saved compression map must replay byte exactly"
    );
    crate::reference_validation::require_hash(
        &baseline,
        "0342618e641c51ce1a077e397695db35254beb4c3b1ec342602a749aed68a896",
    )?;
    println!(
        "complete_native_engram_mhc_checkpoint_verified tensors=387 base=252 memory=35 mhc=100 all_payload_scales_descriptors_maps=byte_exact sha256={original_hash} optimizer_updates=0 dense_training=false"
    );
    Ok(())
}
