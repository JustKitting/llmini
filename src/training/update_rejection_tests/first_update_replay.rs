use std::io::{BufWriter, Write};
use std::path::PathBuf;

use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_LAYER, GPT2_SEQ_LEN};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::{AppResult, Snapshot, Trainer};
use crate::training::TokenDataLoader;

#[test]
#[ignore = "CUDA replay of the first differing FineWeb training update"]
fn native_first_update_replay() -> AppResult {
    assert_eq!(
        (GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_N_LAYER, GPT2_N_EMBD),
        (8, 2048, 16, 2048)
    );
    let output = PathBuf::from(
        std::env::var_os("NVFP4_FIRST_UPDATE_REPORT").ok_or("set NVFP4_FIRST_UPDATE_REPORT")?,
    );
    let mut trainer = Trainer::new_engram_mhc_with_table_budget(0x47505432, 500_000_000)?;
    let mut loader = TokenDataLoader::from_fineweb()?;
    let mut upload = trainer.reusable_default_batch()?;
    let first = loader.next_batch()?;
    let second = loader.next_batch()?;
    assert_eq!(first.source, second.source);
    assert_eq!((first.offset, second.offset), (0, 16384));
    assert!(
        first
            .source
            .ends_with("fineweb_mistral_v01_train_000001.bin")
    );

    let batch = trainer.upload_default_batch(&mut upload, &first.tokens)?;
    let first_stats = trainer.train_step(batch, true)?;
    assert!(first_stats.finite && !first_stats.optimizer.update_skipped);
    let components = trainer.completed_gradient_components(1, first_stats.optimizer.grad_norm)?;
    assert_eq!(components["passed"], true);
    let update = Snapshot::capture(&trainer)?.json();

    let batch = trainer.upload_default_batch(&mut upload, &second.tokens)?;
    let second_stats = trainer.train_step(batch, true)?;
    assert!(second_stats.finite && !second_stats.optimizer.update_skipped);
    let window_hash = |tokens: &[u16]| {
        let mut hash = Sha256::new();
        for &id in tokens {
            hash.update(id.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    };
    let report = json!({
        "seed": "0x47505432",
        "fineweb_shard": first.source,
        "offsets": [first.offset, second.offset],
        "input_window_sha256": [window_hash(&first.tokens), window_hash(&second.tokens)],
        "first_loss_bits": first_stats.loss.to_bits(),
        "second_loss_bits": second_stats.loss.to_bits(),
        "first_loss": first_stats.loss,
        "second_loss": second_stats.loss,
        "first_update_snapshot": update,
        "first_backward_components": components,
    });
    let mut output = BufWriter::new(std::fs::File::create_new(output)?);
    serde_json::to_writer(&mut output, &report)?;
    output.flush()?;
    Ok(())
}
