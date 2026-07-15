# GPT-2 NVFP4 Optimization Rules

These are active rules for comparing training-kernel and optimizer changes.
They are not historical notes.

## Primary Objective

Optimize for held-out validation loss after the fixed 15-minute wall-clock
training run.

Use this validation line as the comparable endpoint:

```text
heldout_eval split=val val_loss=... train_elapsed_s=... completed_steps=...
```

Training loss, one-step runs, 100-step runs, tokens/s, and isolated profiler
timings are diagnostics. They do not prove that an optimization should be
promoted.

## Model-Size Invariant

The active target is the approximately 1B-parameter shape:

```text
GPT2_N_LAYER=16
GPT2_N_EMBD=2048
GPT2_N_HEAD=32
```

Kernel, runtime, optimizer, dataset, and tokenizer experiments must not reduce
the model below 16 layers. A smaller shape may be used only for an explicitly
labelled diagnostic; its timing or loss can never become the active baseline,
promotion evidence, or a commit gate for the 1B target.

## Kernel/Runtime Acceptance Rule

The current kernel/runtime acceptance rule is:

- Accept if 900-second held-out validation loss improves.
- Accept if 900-second held-out validation loss is within `+/-1%` of the current
  baseline and completed step count increases.
- Reject if validation loss worsens by more than `1%`, even if profiler numbers
  or completed step count improve.

The `+/-1%` band is an active noise band, not an old rule and not a weaker
objective. Seed variance can move validation loss within that band, so higher
completed step count inside the band can be a valid long-run improvement.

This rule is for math-preserving kernel/runtime changes. It does not apply to
hyperparameter sweep promotion.

### Minimum Whole-Step Impact

Do not implement or benchmark a kernel/runtime candidate unless its credible
mathematical ceiling can reduce total step time by at least `0.5%`. Compute the
threshold from the active baseline whenever that baseline changes:

```text
minimum_step_saving = (TRAIN_ELAPSED_S / COMPLETED_STEPS) * 0.005
```

For the current baseline, `900.070 / 1073 = 0.838835042` seconds per step, so a
candidate must credibly be able to save at least `4.194175 ms/step` before any
code edit, rebuild, GPU test, or training screen. Multiply a per-launch saving
by the launch count per step and compare that aggregate saving with the
threshold; do not pursue sub-threshold micro-optimizations.

After implementation, the focused profile must measure at least the same
aggregate `0.5%` whole-step saving before proceeding to the 30-second screen or
900-second validation gate. A larger fusion may qualify based on the aggregate
time it removes even when each constituent operation would not qualify alone.

## Active Baseline

NextLat is part of the active model path. Do not protect or compare against
pre-NextLat validation results when evaluating current NextLat work.

`notes/sweep_baseline.env` is the mutable baseline for the active model lineage.
For current work, that means a 900-second result from the active 16-layer
NextLat model, current dataset, and current tokenizer, not a result from an
older or smaller architecture.

## Sweep Rule

Do not run a new hyperparameter sweep for same-math kernel/runtime edits. Use
profiling and fixed 900-second validation for those changes.

Run a multivariable sweep only after a major math or architecture change, or
when explicitly requested.

Sweep baseline promotion is stricter than kernel/runtime acceptance: promote a
hyperparameter candidate only when its 900-second held-out validation loss is
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

Those checks can justify continuing to the 900-second gate. They cannot replace
the 900-second held-out validation gate.
