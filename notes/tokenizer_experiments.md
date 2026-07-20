# Tokenizer Experiments

Tokenizer experiments are separate from same-tokenizer model and kernel
experiments. Raw cross-entropy is measured per token and cannot be compared
across tokenizers. The primary cross-tokenizer validation metric is:

```text
bits_per_byte = cross_entropy_nats_per_token
              * target_token_count
              / target_byte_count
              / ln(2)
```

All runs below retained the complete B4/S2048/L16/d2048/h32 model, active
objective, optimizer, and architecture. `TRAIN_GDN2_TIED_G_ERASE=0` excluded
the unrelated working-copy GDN2 candidate.

## 2026-07-20: GPT-2 and Mistral v0.1

Sources:

- GPT-2 tokenizer:
  <https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py>
- Mistral v0.1 tokenizer:
  <https://huggingface.co/mistralai/Mistral-7B-v0.1>
- Modded-NanoGPT reference:
  <https://github.com/KellerJordan/modded-nanogpt>

Fresh Llama-2 control:

```text
run: target/runs/20260720_064550Z_fineweb_450s
tokenizer_vocab_size: 32000
model_vocab_size: 32000
completed_steps: 1232
train_elapsed_s: 450.292
validation_loss: 4.22384262085
validation_target_tokens: 8192
validation_target_bytes: 33700
validation_bits_per_byte: 1.481297568
```

The control predates direct BPB logging. Its target-byte count was measured
with the unchanged Llama validation shard and the value above was derived from
the logged loss using the documented formula.

GPT-2 used the official 50,257-token encoding and NanoGPT-compatible 50,304-row
model softmax. This added 37,486,592 tied embedding/head parameters relative
to the 32K model.

```text
health_run: target/runs/20260720_075740Z_fineweb_30s
completed_steps: 83
train_elapsed_s: 30.243
validation_loss: 6.442910
validation_bits_per_byte: 2.026990

promotion_run: target/runs/20260720_080027Z_fineweb_450s
completed_steps: 1214
train_elapsed_s: 450.281
validation_loss: 4.843128
validation_target_bytes: 37566
validation_bits_per_byte: 1.523686
```

Against Llama-2, GPT-2 completed 1.461039% fewer steps and regressed BPB by
2.861574%. Although its fixed 4,096-document compression sample encoded
4.452237794 bytes/token, the increased byte exposure did not overcome the
larger head and worse sustained predictive BPB. Reject GPT-2 and remove its
candidate asset/configuration from the retained source.

Mistral v0.1 used the official general-purpose 32K byte-fallback BPE. It kept
the model vocabulary, tied embedding/head parameter count, BOS/EOS document
boundaries, and kernel shapes unchanged.

```text
health_run: target/runs/20260720_081604Z_fineweb_30s
completed_steps: 84
train_elapsed_s: 30.154
validation_loss: 6.123139
validation_bits_per_byte: 2.086036

promotion_run: target/runs/20260720_081648Z_fineweb_450s
completed_steps: 1229
train_elapsed_s: 450.240
validation_loss: 4.317792
validation_target_tokens: 8192
validation_target_bytes: 34691
validation_bits_per_byte: 1.470989
```

Against Llama-2, Mistral completed 0.243506% fewer steps and improved BPB by
0.695915%. Its raw token loss was 2.224263% higher, which is not a regression:
the held-out stream encoded 2.940653% more bytes per token. The fixed
4,096-document compression sample encoded 4.055563409 bytes/token.

Decision: accept Mistral v0.1 as the default tokenizer and retain Llama-2 as a
compile-time control via `TOKENIZER_VARIANT=llama2`. This is a small passing
fixed-time quality result with essentially unchanged throughput and model
capacity.

The current shard builder splits validation and training after a fixed token
count, so tokenizer variants begin their training shard at slightly different
raw-text offsets. The source parquet and distribution are the same, but this is
a remaining confound. Require a fixed raw-document split before treating a
similarly small tokenizer delta as conclusive in a downstream tokenizer study.

## 2026-07-20: Exact T-FREE 8K hashed byte trigrams

Source: <https://arxiv.org/abs/2406.19223> and the released
<https://github.com/Aleph-Alpha/trigrams> implementation at commit
`bc7c2ca598a0ad7d0afe20116cbd63a936d1a1fc`.

This reran the earlier local word-trigram direction with the paper's selected
1B representation and released mapping: exactly 8,000 features, ten unique
MD5-mapped activations per byte trigram, and no lowercase-only activations.
It used separate 8K input/output tables, weighted multi-label BCE, the released
normalized-sigmoid dictionary decoder, a fixed 100,329-word dictionary, and
an 8,704-wide MLP to preserve the approximately 1B parameter budget. Every
accepted model section, including NextLat, remained active.

The published `positive_weight=300` was suboptimal for this short 8K run.
Identical 201-step tuning found a reproducible minimum at 75-100:

```text
weight 75:
  run: target/runs/20260720_161023Z_fineweb_900s
  elapsed_s: 77.884
  released_decoder_bits_per_byte: 3.504237

weight 100:
  run: target/runs/20260720_161143Z_fineweb_900s
  elapsed_s: 78.228
  released_decoder_bits_per_byte: 3.503966

weight 300:
  run: target/runs/20260720_161303Z_fineweb_900s
  elapsed_s: 78.269
  released_decoder_bits_per_byte: 3.517562
```

Weight 100 improves T-FREE's own released decoder BPB by 0.386518% at matched
exposure. It nevertheless does not beat the accepted Canon/Mistral 201-step
BPB of 1.890499. The released decoder score is bounded and not numerically
equivalent to categorical CE; a second calibrated Bernoulli dictionary
distribution was also evaluated and was materially worse, so neither
normalization supplied a cross-tokenizer quality win.

The exact layout did produce a clear memory result: 41,989.188 MiB allocated
versus 43,233.188 MiB for the clean Canon/Mistral parent, saving exactly
1,244 MiB (2.877419%). It was also 7.961744% slower per update at 201 steps.

Decision: reject this T-FREE integration for the fixed-time likelihood target,
skip the 450-second gate, remove its source, and retain only the measurements.
The detailed representation, kernels, correctness tests, evaluation caveats,
and tuning sweep are recorded in `optimization_experiments.md`.
