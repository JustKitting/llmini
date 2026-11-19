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

Run the native executable directly:

```bash
./target/release/rust-kernels --dataset fineweb --max-seconds 900
```

Run a sustained stability session:

```bash
./target/release/rust-kernels --dataset fineweb --steps 30000 --max-seconds 21600 --eval-interval 1000
```

The default CUDA device is GPU 0. Command-line options override their corresponding
`TRAIN_*` environment settings. Existing environment settings remain supported.
Runs use the normal timestamped `target/runs` directories.

Inspect a saved checkpoint's activation tails and NVFP4 representation through
the CUDA forward path, with zero optimizer updates:

```bash
CUDA_DEVICE_INDEX=0 ./target/release/rust-kernels --inspect-model target/runs/RUN/model.ckpt
```

Attribute a saved checkpoint's gradient norm through native CUDA forward/backward,
with zero optimizer updates:

```bash
CUDA_DEVICE_INDEX=0 ./target/release/rust-kernels --inspect-gradients target/runs/RUN/model.ckpt
```

This writes `gradient_inspection.json`, using the existing GPU clipping reductions
to separate matrix, learned SymExpLin control and ordinary parameter contributions.
It verifies exact checkpoint replay before/after the probe. The checkpoint contains
averaged model weights; this does not replay an unsaved training/optimizer state.

Set `TRAIN_GRADIENT_COMPONENT_INTERVAL=1000` on a training command to record
live pre-update gradient contributions in `gradient_components.jsonl` at the
first attempt, every1000 attempts and the last attempt. This reads the existing
CUDA clipping reductions and performs no additional training updates.

Evaluate a saved native checkpoint on the reserved NanoGPT reference text:

```bash
./target/release/rust-kernels --evaluate-model target/runs/RUN/model.ckpt --validation-documents data/fineweb/reference/documents.bin
```

This performs CUDA evaluation with zero optimizer updates and writes
`reference_validation.json` beside the checkpoint. It checks native held-out
replay, causality and every scored target before reporting BPB. The result states
the context, sliding stride and comparison limits; it does not replace the
training run's fixed validation metric. Run it after training releases GPU 0.
When the checkpoint has Source/Window_offset metrics, the result also includes a
fixed 32-stratum sample of already-seen training batches, scored with the frozen
checkpoint and reported separately from held-out metrics.

For a numerical graph repair that changes an old checkpoint's replay CE, the
native checkpoint actions accept `--replay-provenance PATH`. This requires a
verified native replay record bound to the unchanged checkpoint, current PTX,
original producer CE, reference documents and strict causality checks. Reports
retain the original producer result separately from repaired-graph scoring.
Without that option the historical CE replay gate remains unchanged. See
[the trained latent-feedback causality repair](notes/latent_feedback_causality_20261115.md).

Prepare those documents with the project's Rust data preparer:

```bash
cargo build --release --bin prepare_fineweb
./target/release/prepare_fineweb --reference-tokens /path/to/fineweb_val_000000.bin --reference-vocab /path/to/gpt2-vocab.json --reference-output data/fineweb/reference
```

Inputs are the [official token shard](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin)
and [GPT-2 vocabulary](https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json).
The preparer verifies the recorded source, vocabulary and tokenizer SHA256
identities, preserves exactly 46,515,256 raw benchmark bytes and verifies the
prepared document identity. Text/EOS targets are scored once; added BOS and
padding are excluded. The final partial document receives no invented EOS.

Cargo also selects the native training executable by default:

```bash
cargo run --release -- --dataset fineweb --steps 30000 --max-seconds 21600 --eval-interval 1000
./target/release/rust-kernels --help
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
