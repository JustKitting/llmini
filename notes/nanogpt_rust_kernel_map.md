# nanoGPT Inference To Rust Kernel Map

Source reference:

- <https://github.com/karpathy/nanoGPT/blob/master/sample.py>
- <https://github.com/karpathy/nanoGPT/blob/master/model.py>
- <https://github.com/karpathy/nanoGPT/blob/master/README.md>

This note maps the upstream nanoGPT inference path into the Rust/CUDA kernel
work we need to replace PyTorch calls one at a time.

## PyTorch Entry Commands

README examples:

```sh
python sample.py --out_dir=out-shakespeare-char
```

```sh
python sample.py \
  --init_from=gpt2-xl \
  --start="What is the answer to life, the universe, and everything?" \
  --num_samples=5 --max_new_tokens=100
```

## Init/Inference Skeleton

Without imports/config boilerplate, nanoGPT does:

```python
# load model
if init_from == "resume":
    checkpoint = torch.load(out_dir + "/ckpt.pt", map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint["model"])
elif init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

# tokenize prompt
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# generate
with torch.no_grad():
    with autocast_ctx:
        y = model.generate(x, max_new_tokens, temperature, top_k)
```

`generate()` is:

```python
for _ in range(max_new_tokens):
    idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :] / temperature

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -inf

    probs = softmax(logits)
    idx_next = multinomial(probs, 1)
    idx = cat(idx, idx_next)
```

## Forward Graph

The actual model path we need to replace:

```python
tok_emb = wte(idx)
pos_emb = wpe(pos)
x = tok_emb + pos_emb

for block in blocks:
    x = x + attention(layernorm_1(x))
    x = x + mlp(layernorm_2(x))

x = final_layernorm(x)
logits = lm_head(x[:, [-1], :])
```

Attention block:

```python
q, k, v = c_attn(x).split(n_embd)
q, k, v = reshape_to_heads(q, k, v)

y = causal_attention(q, k, v)

y = merge_heads(y)
y = c_proj(y)
```

MLP block:

```python
x = c_fc(x)
x = gelu(x)
x = c_proj(x)
```

## Rust Kernel Map

What we need, in order:

```text
1. checkpoint/config loader
   - GPTConfig
   - weight tensors
   - GPT-2 transpose handling for imported HF weights

2. tokenizer path
   - initially can stay host-side
   - produce token ids

3. embedding kernel
   - gather token embedding
   - add positional embedding
   - output residual stream

4. LN -> NVFP4 activation kernel
   - this is the layernorm kernel we were moving toward
   - eventually hidden width = n_embd, not fixed 32
   - emits e2m1 payload + e4m3 local scales + uses supplied/global scale

5. NVFP4 GEMM kernels
   - QKV projection: ln_1(x) @ c_attn.weight
   - attention output projection: attn_out @ c_proj.weight
   - MLP up projection: ln_2(x) @ c_fc.weight
   - MLP down projection: gelu_out @ c_proj.weight
   - final lm_head: ln_f(x_last) @ tied embedding weight

6. attention kernels
   - QK^T causal score
   - softmax
   - P @ V
   - probably separate from the first LN/GEMM work

7. GELU -> NVFP4 kernel
   - apply GELU after MLP up projection
   - quantize activation tile for MLP down projection

8. residual add kernels
   - x = x + attn_out
   - x = x + mlp_out
   - good place to emit next mean/rstd or amax stats

9. logits sampling
   - top-k
   - softmax
   - multinomial
   - append token
```

One important note: nanoGPT inference recomputes the whole context each
generated token. It does not use a KV cache. For a faithful first Rust port, we
can do the same. For performance, KV cache becomes a later divergence from the
reference.

## Training Optimizer Notes

For Muon-style polar updates, dense optimizer GEMMs should use the same
NVFP4 tensor-core path as model GEMMs:

```text
fp32 operand -> Quartet-II/MS-EDEN downcast -> NVFP4 TC GEMM -> fp32 output
```

The reusable TC matmul primitive pads incompatible `k` dimensions internally
before MS-EDEN downcast, including vocab-width tied-head paths. More optimized
tiling/layout work should come later after the polar update path is
functionally wired and tested.
