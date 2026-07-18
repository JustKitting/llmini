# GPT-2 NVFP4 Optimization Rules

These are active rules for comparing training-kernel and optimizer changes.
They are not historical notes.

## Primary Objective

Optimize for the lowest held-out validation loss after the fixed 450-second
single-GPU candidate gate:

- For research-scale optimizer, architecture, objective, or numerical changes,
  30 seconds is only a bring-up health check: verify launchability,
  finite/nonzero metrics, real updates, zero unexpected skips, and absence of
  immediate divergence. Do not accept or reject one of these larger changes
  from its 30-second loss delta.
- A schedule parameter that is mathematically dormant for the entire
  30-second window must instead use a matched fixed-step diagnostic after the
  parameter activates. For the current 83-step warmup, use 200 completed
  optimizer steps for both control and candidate. The warmup itself remains
  step-gated; do not convert it to elapsed-time logic. This diagnostic still
  cannot replace the 450-second loss gate.
- 450 seconds is the mandatory sustained stability, regression, and held-out
  quality gate before a passing change is committed in JJ.
- The 450-second duration is an iteration gate, not the final training budget
  or a redefinition of the longer-run loss target.

### Structural-Method Characterization and Optimization

For a research-scale optimizer, architecture, objective, numerical, or
nonlinearity-layout method, do not treat the first working implementation as
the final performance of the method:

- The 30-second run is a correctness and health check only. It verifies that
  the implementation launches, updates the intended intact model, remains
  finite, and does not immediately diverge. Its fixed-time loss is not
  acceptance or rejection evidence for a structural method.
- The first sustained 450-second run of an unoptimized structural
  implementation is a characterization run. Record same-step convergence,
  fixed-wall-clock convergence, operation and kernel timings, memory use, and
  stability separately. Do not reject the method merely because recoverable
  implementation overhead causes fewer completed steps or a worse fixed-time
  endpoint.
- When same-step held-out or training quality improves, or remains competitive
  enough to preserve a credible quality signal, identify whether the measured
  slowdown is inherent to the method or is caused by recoverable implementation
  overhead such as extra launches, materialized intermediates, redundant
  quantization, unfused pointwise work, poor layout, or avoidable memory
  traffic.
- If the quality signal is credible and the slowdown is plausibly recoverable,
  perform at least one justified profiling and optimization pass before making
  a keep/reject decision. This applies even when the unoptimized
  characterization run loses the fixed-time gate.
- Reject before optimization only when the algorithmic signal itself is
  clearly bad or the added cost is mathematically inherent and too large to
  recover. After an optimization pass, rejection is appropriate when the
  remaining overhead is measured to be inherent or the optimized method still
  lacks a credible quality benefit.
- A characterization run never promotes a change. The optimized implementation
  must still pass a fresh 30-second health check and the normal 450-second
  fixed-wall-clock held-out gate before it may be committed in JJ.

This rule prevents an unoptimized implementation from being mistaken for an
algorithmic failure without weakening the final fixed-time acceptance
criterion.

Use this validation line as the comparable endpoint:

```text
heldout_eval split=val val_loss=... train_elapsed_s=... completed_steps=...
```

Held-out evaluation must materialize the schedule-free averaged `x_master`
weights. `materialize_training_weights` emits the `z/x` interpolation for the
next training step and is not a valid evaluation substitute. This distinction
is part of the baseline convention, not a tunable optimization.

Training loss, one-step runs, tokens/s, step time, memory use, and isolated
profiler timings are diagnostics. Tokens/s is a useful explanation for quality
movement because it controls training exposure, but it is not a hard objective:
a slower architecture may win if it produces better held-out and downstream
quality. Diagnostic metrics do not replace the 450-second gate.

## Model-Size Invariant

The active control is the approximately 1B-effective-parameter shape. Its
pretraining baseline uses four independent 2048-token sequences per optimizer
step:

```text
GPT2_SEQ_LEN=2048
GPT2_BATCH_SIZE=4
GPT2_N_LAYER=16
GPT2_N_EMBD=2048
GPT2_N_HEAD=32
```

This has 964376960 effective parameters and 984571904 allocated parameter
slots. Kernel, runtime, optimizer, dataset, and tokenizer experiments must not
reduce the model below 16 layers, width 2048, or 32 heads. A smaller model may
be used only for an explicitly labelled diagnostic; its timing or loss can
never become the active baseline, promotion evidence, or a commit gate.

The 2048-token sequence length is the pretraining context, not a reduction in
the model-size floor or the final context target. Any later long-context stage
must preserve optimizer state and earn its own matched held-out validation;
model-only checkpoint reloads that reset AMUSE and Muon state are not valid
stage-transition evidence.

### Model-Integrity and Task Invariant

The gate measures improvements to the intact model on the intact task. It is
invalid to lower the endpoint by making part of the model task-gradient-dead:

- Do not remove, bypass, freeze, or zero-weight an active branch, including
  NextLat, while retaining its allocations only to satisfy a nominal parameter
  count.
- Shape and layer-width redistribution is allowed, as are optimizer,
  quantization, initialization, and algorithmic changes, but the resulting
  model must retain approximately 1B effective trainable parameters and all
  declared active sections must receive a real training signal.
- A paper-defined normalization architecture may replace the existing
  LayerNorm family. This is an allowed model change, not an invalid branch
  bypass, provided the replacement's normalization, interpolation, and scale
  parameters are genuinely active and trainable. Such a candidate must still
  preserve every attention/KDA, MLP, value-residual, and NextLat computation,
  all 16 layers, width 2048, 32 heads, the intact task, and approximately 1B
  effective trainable parameters. Dead retained LayerNorm allocations do not
  count toward that parameter total.
- A paper-defined MLP activation and projection layout may replace ReLU-squared
  when the MLP remains an active, trainable nonlinear branch in every layer.
  Match the approximately 1B parameter budget by adjusting the intermediate
  width rather than deleting the branch or silently inflating the model enough
  to make the comparison a model-size experiment.
- The matched gate uses the same FineWeb source, Llama-2 tokenizer, validation
  stream, context, and sampling convention. Do not simplify the corpus,
  alphabet, vocabulary, or validation problem to lower raw cross-entropy.
- Dataset or tokenizer studies must be labelled as separate experiments and
  compared with a tokenizer-independent metric such as bits per byte plus
  downstream evaluation. Their raw token cross-entropies cannot replace this
  baseline.

The 450-second gate is a screening instrument for genuine model improvements,
not an objective that may be made easier.

## Kernel/Runtime Acceptance Rule

The active phase is loss-focused. A kernel/runtime candidate is promoted only
when its 450-second held-out validation loss is lower than the current baseline.
Completed steps and throughput can explain the loss result, but cannot promote
a flat or worse endpoint. The former `+/-1%` speed-within-noise exception is not
active.

### Minimum Whole-Step Impact

Do not implement or benchmark a kernel/runtime candidate unless its credible
mathematical ceiling can reduce total step time by at least `0.5%`. Compute the
threshold from the active baseline whenever that baseline changes:

```text
minimum_step_saving = (TRAIN_ELAPSED_S / COMPLETED_STEPS) * 0.005
```

For the current matched 450-second baseline,
`450.193 / 1451 = 0.310263956` seconds per step, so a candidate batch must
credibly be able to save at least `1.551320 ms/step`
before a rebuild, GPU test, or training screen. Multiply a per-launch saving by
the launch count per step and compare that aggregate saving with the threshold.

The threshold applies to the complete compatible batch, not every atomic edit.
Several clear savings may be implemented and profiled together when their
credible combined ceiling clears `0.5%`. A component that has already measured
a real but sub-threshold speedup may remain in an unproven batch while another
compatible saving is added; do not discard known-good work merely because it
misses `0.5%` alone. Do not run the 30-second or 450-second gates until the
combined profile clears the whole-step threshold.

After implementation, the focused profile must measure at least the same
aggregate `0.5%` whole-step saving before proceeding to the 30-second screen or
450-second validation gate. This aggregate rule applies equally to a fusion or
to a batch of independent compatible wins.

### Memory-Capacity Wins

Actively look for clear, measured reductions in peak GPU memory as a separate
optimization path. The `0.5%` whole-step speed filter does not exclude a
candidate whose primary benefit is lower peak VRAM. A memory-only candidate may
be kept when it preserves the active model math, held-out loss, and stability,
and does not introduce a meaningful step-time regression.

Report the exact matched-run peak-memory reduction and the concrete capacity it
could unlock, such as a larger batch or more tokens per step. Validate any
claimed throughput gain separately at the larger configuration; lower memory by
itself is not evidence of higher tokens/s. Memory-only candidates still require
the normal correctness checks plus the 30-second and 450-second gates before
commit.

## Active Baseline

NextLat is part of the active model path. Do not protect or compare against
pre-NextLat validation results when evaluating current NextLat work.

`notes/sweep_baseline.env` is the mutable baseline for the active model lineage.
For current work, that means a 450-second result from the active 16-layer,
width-2048, 32-head, batch-4, 2K-pretraining-context NextLat model, current
dataset, and current tokenizer, not a result from an older or smaller
architecture.

The restored pre-regression FP16 staging path plus the later accepted kernel
and memory wins, positive `0.5` NextLat auxiliary coefficient, retuned AMUSE
initial interpolation and Adam learning rate, NorMuon variance reduction,
period-2 SignMuon, fixed 50/50 ResFormer value residuals, LayerNorm Scaling
without depth-scaled residual initialization, detached KDA chunk-state
backward, later-only value-residual routing, and probability-mass sampled
64x64 full-attention backward is the current matched 450-second control:
1451 steps in 450.193 seconds with held-out loss 4.738475. Do not compare future 450-second
candidates with historical 900-second endpoints.

## Sweep Rule

Do not run a new hyperparameter sweep for same-math kernel/runtime edits. Use
profiling and fixed 450-second validation for those changes.

Run a multivariable sweep only after a major math or architecture change, or
when explicitly requested.

Sweep baseline promotion is stricter than kernel/runtime acceptance: promote a
hyperparameter candidate only when its 450-second held-out validation loss is
lower than the current sweep baseline. Do not promote a sweep candidate only
because it completed more steps inside the `+/-1%` noise band.

Stability is a Bayesian-style survival prior for screening and acquisition. It
helps predict whether a candidate is likely to survive the longer validation
run, and it can increase uncertainty/exploration pressure. It is not a separate
objective and must not promote a candidate by itself.

## Promotion Rule

Do not call a candidate promoted, accepted, or commit-worthy from:

- build success,
- one-step launch checks,
- 100-step screens,
- short profiler runs,
- tokens/s improvement alone,
- kernel timing improvement alone.

Those checks can justify continuing to the 450-second gate. They cannot replace
the 450-second held-out validation gate.
