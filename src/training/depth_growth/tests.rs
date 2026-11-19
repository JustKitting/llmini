use cuda_core::{CudaStream, DeviceBuffer, memory};
use gpt2_nvfp4::{
    GPT2_BATCH_SIZE, GPT2_N_HEAD, GPT2_N_LAYER, GPT2_SEQ_LEN, depth_growth::GrowthTransition,
    exclusive_self_attention_enabled,
};

use super::Trainer;
use crate::training::launch::TrainConfig;
use crate::{AppResult, training::TokenDataLoader};

fn configured_growth_trainer() -> AppResult<Trainer> {
    TrainConfig {
        depth_growth: true,
        seed: 0x47505432,
        latent_feedback: false,
        mhc: true,
        engram_mhc: true,
        engram_table_params: Some(500_000_000),
        engram_four_grams: false,
        step_cap: 20_000,
        log_interval: 1,
        eval_interval: None,
        eval_repeats: 1,
        max_seconds: 43_200.,
    }
    .create_trainer()
}

fn prefix<T: Default + Copy>(stream: &CudaStream, buffer: &DeviceBuffer<T>) -> AppResult<[T; 8]> {
    read_span(stream, buffer, 0)
}

fn read_span<T: Default + Copy, const N: usize>(
    stream: &CudaStream,
    buffer: &DeviceBuffer<T>,
    offset: usize,
) -> AppResult<[T; N]> {
    assert!(buffer.len() >= offset + N);
    let mut values = [T::default(); N];
    stream.context().bind_to_thread()?;
    // This fixed host array remains borrowed until the stream is synchronized.
    unsafe {
        memory::memcpy_dtoh_async(
            values.as_mut_ptr(),
            buffer.cu_deviceptr() + (offset * size_of::<T>()) as u64,
            size_of::<[T; N]>(),
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
        &block.attn_c_proj.weight_muon.z_master,
        &block.attn_c_proj.weight_muon.x_master,
        &block.mlp_down.weight_muon.z_master,
        &block.mlp_down.weight_muon.x_master,
        &block.ln_1.bias.first,
        &block.ln_1.bias.second,
        &trainer.mhc.as_ref().unwrap().units[2 * index].phi.x_master,
    ] {
        bits.extend(prefix(stream, tensor)?.map(f32::to_bits));
    }
    for tensor in [&matrix.momentum, &matrix.variance] {
        bits.extend(prefix(stream, tensor)?.map(u32::from));
    }
    let xsa = &trainer.buffers.optimizer_state.xsa_alphas;
    for tensor in [&xsa.z_master, &xsa.x_master, &xsa.first, &xsa.second] {
        bits.extend(
            read_span::<_, GPT2_N_HEAD>(stream, tensor, index * GPT2_N_HEAD)?.map(f32::to_bits),
        );
    }
    Ok(bits)
}

fn seed_dormant_xsa_moments(trainer: &mut Trainer, depth: usize) -> AppResult<()> {
    let stream = &trainer.runtime.stream;
    let xsa = &mut trainer.buffers.optimizer_state.xsa_alphas;
    let byte_offset = (depth * GPT2_N_HEAD * size_of::<f32>()) as u64;
    for (state, value) in [(&mut xsa.first, 0.25_f32), (&mut xsa.second, 0.5_f32)] {
        unsafe {
            memory::memcpy_htod_async(
                state.cu_deviceptr() + byte_offset,
                &value,
                size_of::<f32>(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
    }
    Ok(())
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
    let mut trainer = configured_growth_trainer()?;
    assert_eq!(trainer.layer_layout(), super::LayerLayout::grown(4));
    assert_eq!(trainer.mhc.as_ref().unwrap().active_boundaries, 8);
    assert_eq!(trainer.engram.as_ref().unwrap().active_depth, 4);
    let expected = 0.02 / 8.0_f32.sqrt();
    for block in &trainer.uploaded.blocks[..4] {
        for weight in [&block.attn_c_proj.weight, &block.mlp_down.weight] {
            assert_eq!(
                weight.global_scale.to_host_vec(&trainer.runtime.stream)?,
                [expected]
            );
        }
        for weight in [&block.attn_qkv.weight, &block.mlp_up.weight] {
            assert_eq!(
                weight.global_scale.to_host_vec(&trainer.runtime.stream)?,
                [0.02]
            );
        }
    }
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
        assert_eq!(trainer.layer_layout(), super::LayerLayout::grown(depth));
        assert_eq!(optimizer_clock(&trainer), clock);
        for (slot, expected) in copies {
            assert_eq!(
                block_state(&trainer, slot)?,
                expected,
                "depth {depth}, slot {slot}"
            );
        }
        // An all-zero optimizer state would hide an update to an inactive head.
        // Set nonzero moments after the initial prefix has been selected.
        if depth == 4 && exclusive_self_attention_enabled() {
            seed_dormant_xsa_moments(&mut trainer, depth)?;
        }
        let dormant = (depth < GPT2_N_LAYER)
            .then(|| block_state(&trainer, depth))
            .transpose()?;
        let window = loader.next_batch()?;
        let batch = trainer.upload_default_batch(&mut upload, &window.tokens)?;
        let stats = trainer.train_step(batch, true)?;
        assert!(stats.finite && !stats.optimizer.skip_non_finite);
        if depth == 4 {
            assert!(
                !stats.optimizer.update_skipped,
                "first update must exercise the optimizer"
            );
        }
        if let Some(expected) = dormant {
            assert_eq!(
                block_state(&trainer, depth)?,
                expected,
                "dormant depth {depth}"
            );
        }
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

#[test]
#[ignore = "uses the native NVFP4 CUDA evaluator and a partial Engram/mHC checkpoint"]
fn partial_growth_checkpoint_replays_the_active_prefix() -> AppResult {
    let mut trainer = configured_growth_trainer()?;
    let windows = TokenDataLoader::fineweb_validation_tokens()?;
    let batch = trainer.batch_from_windows(&windows, crate::training::data::VALIDATION_WINDOWS)?;
    trainer.select_training_depth(8)?;
    trainer.materialize_evaluation_weights()?;
    let expected = trainer.eval_token_losses(&batch)?;
    assert!(expected.iter().all(|loss| loss.is_finite()));

    let directory =
        std::env::temp_dir().join(format!("nvfp4-growth-replay-{}", std::process::id()));
    std::fs::create_dir(&directory)?;
    let path = directory.join("model.ckpt");
    let result = (|| -> AppResult {
        trainer.save_model(&path)?;
        trainer.select_training_depth(12)?;
        assert_eq!(trainer.active_depth(), 12);
        trainer.load_model(&path)?;
        assert_eq!(trainer.layer_layout(), super::LayerLayout::grown(8));
        assert_eq!(trainer.mhc.as_ref().unwrap().active_boundaries, 16);
        assert_eq!(trainer.engram.as_ref().unwrap().active_depth, 8);
        assert_eq!(trainer.eval_token_losses(&batch)?, expected);
        assert_eq!(trainer.buffers.optimizer_state.step, 0);

        drop(batch);
        drop(trainer);
        let mut reopened = Trainer::from_checkpoint(0x47505432, &path)?;
        let batch =
            reopened.batch_from_windows(&windows, crate::training::data::VALIDATION_WINDOWS)?;
        assert_eq!(reopened.layer_layout(), super::LayerLayout::grown(8));
        assert_eq!(reopened.mhc.as_ref().unwrap().active_boundaries, 16);
        assert_eq!(reopened.engram.as_ref().unwrap().active_depth, 8);
        assert_eq!(reopened.eval_token_losses(&batch)?, expected);
        assert_eq!(reopened.buffers.optimizer_state.step, 0);
        Ok(())
    })();
    let cleanup = std::fs::remove_dir_all(&directory);
    result?;
    cleanup?;
    Ok(())
}
