# rust-kernels

This repository is an experiment in cost-constrained language-model training.
The project goal is to fully train a small model, including pre-training and
post-training, and measure how well it can perform on standard benchmarks such
as GSM8K after roughly `$1` of training compute.

For this project, `$1` currently means about one hour on a single RTX 6000
Blackwell-class GPU. The target is not maximum tokens per dollar in the abstract;
the target is useful model capability after a fixed wall-clock budget.

## Current Focus

The active work is the pretraining path. The current model path is
GPT-2-shaped, trained on SYNTH shards, with NVFP4-heavy CUDA kernels and fused
training operations. The active comparison point is a fixed-wall training run
that reports:

```text
heldout_eval split=val val_loss=... train_elapsed_s=... completed_steps=...
```

Kernel and runtime changes are judged by held-out validation loss after the
fixed training budget. Short runs, profiler output, tokens/s, and isolated
kernel timings are diagnostics; they are useful for deciding what to try next,
but they are not the objective by themselves.

## Why Quantized And Fused Kernels

This repo is intentionally not a clean reference implementation of GPT-2
training. Unlike Karpathy-style fixed-token training comparisons, wall-clock
time is the optimization target here. That makes quantization, custom CUDA
kernels, kernel fusion, layout changes, and optimizer-specific fast paths
acceptable and expected when they improve the fixed-budget result.

The code already leans heavily on:

- NVFP4 quantization and tensor-core paths.
- Fused CUDA kernels for projection, attention, loss, optimizer, and tape saves.
- Schedule-free AMUSE with Muon matrix updates.
- Coupled sweeps over model and optimizer parameters when the architecture or
  math changes enough to justify them.

## Post-Training Direction

Post-training is a planned part of the full `$1` training target. New kernels and
runtime paths should be added for post-training methods when they become the
active focus, including supervised tuning, preference/RL-style updates, or other
benchmark-oriented training loops.

The eventual success metric is benchmark capability under the same cost budget,
with GSM8K as an example target benchmark. The current repository state is still
focused on making the pretraining step fast and useful enough to be a good base
for that later post-training work.

## Repository Layout

- `src/` - training application, wall-clock run loop, checkpointing, generation,
  sweep orchestration, and logging.
- `crates/gpt2-nvfp4/` - GPT-2-shaped model components, forward/backward graph,
  tapes, scratch buffers, and model-level tests.
- `crates/cuda-kernels/` - CUDA device kernels and kernel tests for attention,
  projection, layer norm, loss, NVFP4 quantization, and optimizer paths.
- `crates/synth-prep/` - SYNTH data download, tokenization, and shard creation.
- `crates/llama2-tokenizer/` - local tokenizer wrapper and assets.
- `notes/` - optimization rules, baseline state, sweep notes, and measured
  experiment history.

## Basic Commands

Build the CUDA kernels for Blackwell:

```bash
cargo oxide build --arch sm_120a
```

Run the default training loop:

```bash
CUDA_DEVICE_INDEX=0 TRAIN_DATASET=synth TRAIN_MAX_SECONDS=900 cargo run --release --bin run_rebuilt
```

Run a short screen:

```bash
CUDA_DEVICE_INDEX=0 TRAIN_DATASET=synth TRAIN_MAX_SECONDS=30 cargo run --release --bin run_rebuilt
```

Run a coupled sweep:

```bash
CUDA_DEVICE_INDEX=0 cargo run --release --bin sweep -- --trials 12 --max-seconds 900 --screen-max-seconds 30
```

Run focused CUDA tests, for example:

```bash
CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1
CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture --test-threads=1
```

## Optimization Rule

For kernel/runtime work, promotion requires the full fixed-wall validation gate.
Build success, unit tests, one-step launches, short screens, and profiler wins
only justify continuing. The current rules and active baseline live in:

- `notes/optimization_rules.md`
- `notes/sweep_baseline.env`
- `notes/optimization_experiments.md`

Do not retest old optimization failures, and do not use failed historical
experiments as inspiration for new candidates. The hill-climb must start from
current kernels, current profiles, and fresh source analysis. Historical failed
entries are records of what not to try again.
