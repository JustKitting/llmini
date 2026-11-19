use cuda_core::{CudaStream, DeviceBuffer, memory};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_LAYER, GPT2_SEQ_LEN, depth_growth::GrowthTransition};

use super::Trainer;
use crate::{AppResult, training::TokenDataLoader};

fn prefix<T: Default + Copy>(stream: &CudaStream, buffer: &DeviceBuffer<T>) -> AppResult<[T; 8]> {
    assert!(buffer.len() >= 8);
    let mut values = [T::default(); 8];
    stream.context().bind_to_thread()?;
    // This fixed host array remains borrowed until the stream is synchronized.
    unsafe {
        memory::memcpy_dtoh_async(
            values.as_mut_ptr(),
            buffer.cu_deviceptr(),
            size_of::<[T; 8]>(),
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()?;
    Ok(values)
}

fn block_state(trainer: &Trainer, index: usize) -> AppResult<Vec<u32>> {
    let stream = trainer.runtime.stream.as_ref();
    let block = &trainer.buffers.optimizer_state.blocks[index];
    let matrix = &block.attn_qkv.weight_muon;
    let mut bits = Vec::new();
    for tensor in [
        &matrix.z_master,
        &matrix.x_master,
        &matrix.second_momentum,
        &block.ln_1.bias.first,
        &block.ln_1.bias.second,
        &trainer.mhc.as_ref().unwrap().units[2 * index].phi.x_master,
    ] {
        bits.extend(prefix(stream, tensor)?.map(f32::to_bits));
    }
    for tensor in [&matrix.momentum, &matrix.variance] {
        bits.extend(prefix(stream, tensor)?.map(u32::from));
    }
    Ok(bits)
}

fn optimizer_clock(trainer: &Trainer) -> (u32, u32, String) {
    let state = &trainer.buffers.optimizer_state;
    (
        state.step,
        state.schedule_free_weight_sum.to_bits(),
        format!("{:?}", trainer.rng),
    )
}

#[test]
#[ignore = "uses the native B8/S2048 FineWeb/Engram/mHC CUDA trainer"]
fn native_depth_growth_copies_state_and_reduces_only_active_parameters() -> AppResult {
    assert_eq!((GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_N_LAYER), (8, 2048, 16));
    let mut trainer = Trainer::new_engram_mhc_with_table_budget(0x47505432, 500_000_000)?;
    let mut loader = TokenDataLoader::from_fineweb()?;
    let mut upload = trainer.reusable_default_batch()?;
    let mut previous = 0;
    for depth in [4, 8, 12, 16] {
        let clock = optimizer_clock(&trainer);
        let copies = GrowthTransition::new(previous, depth)
            .into_iter()
            .flat_map(GrowthTransition::copies)
            .map(|(source, destination)| Ok((destination, block_state(&trainer, source)?)))
            .collect::<AppResult<Vec<_>>>()?;
        trainer.select_training_depth(depth)?;
        assert_eq!(optimizer_clock(&trainer), clock);
        for (slot, expected) in copies {
            assert_eq!(
                block_state(&trainer, slot)?,
                expected,
                "depth {depth}, slot {slot}"
            );
        }
        if depth != 16 {
            let error = trainer
                .save_model(std::path::Path::new("partial.ckpt"))
                .unwrap_err();
            assert!(error.to_string().contains("partial growth stage"));
        }

        let window = loader.next_batch()?;
        let batch = trainer.upload_default_batch(&mut upload, &window.tokens)?;
        let stats = trainer.train_step(batch, true)?;
        assert!(stats.finite && !stats.optimizer.skip_non_finite);
        assert_eq!(
            trainer.buffers.optimizer_state.step,
            clock.0 + u32::from(!stats.optimizer.update_skipped)
        );
        let components =
            trainer.completed_gradient_components(depth / 4, stats.optimizer.grad_norm)?;
        assert_eq!(components["passed"], true);
        let components = components["components"].as_array().unwrap();
        let count = |prefix: &str| {
            components
                .iter()
                .filter(|c| {
                    c["name"]
                        .as_str()
                        .is_some_and(|name| name.starts_with(prefix))
                })
                .count()
        };
        assert!(count(&format!("blocks.{}.", depth - 1)) > 0);
        assert_eq!(count(&format!("blocks.{depth}.")), 0);
        assert!(count(&format!("mhc.boundaries.{}.", 2 * depth - 1)) > 0);
        assert_eq!(count(&format!("mhc.boundaries.{}.", 2 * depth)), 0);
        assert!(count("mhc.head.") > 0);
        previous = depth;
    }
    Ok(())
}
