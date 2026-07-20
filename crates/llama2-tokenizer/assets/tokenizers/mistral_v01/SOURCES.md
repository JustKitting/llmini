# Mistral v0.1 tokenizer

The tokenizer is the official `tokenizer.json` from
`mistralai/Mistral-7B-v0.1`:

https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json

Retrieved 2026-07-20. SHA-256:

```text
11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720
```

The upstream model card identifies it as a 32,000-token byte-fallback BPE
tokenizer. This experiment uses the upstream BOS token ID 1 and EOS token ID 2
around each FineWeb document, matching the existing Llama tokenizer boundary
convention.
