---
license: mit
datasets:
  - HuggingFaceFW/fineweb
language:
  - en
pipeline_tag: text-generation
tags:
  - llama
  - from-scratch
  - gqa
  - rope
library_name: pytorch
---

# llama-124m-fineweb

A 124.67M-parameter LLaMA-style decoder-only transformer **written from scratch in
PyTorch** — RoPE, grouped-query attention, RMSNorm and SwiGLU all implemented directly
rather than imported — and trained on 5.27B tokens of FineWeb.

Code, training script and the two smaller tiers that preceded this one:
**https://github.com/CloudedLeopard17/llama-from-scratch**

## Results

| | |
|---|---|
| val loss | **3.1375** |
| perplexity | 23.0 |
| bits per char | 0.987 |
| train–val gap | +0.031 |

Validation loss fell monotonically across all 107 checkpoints, with the best value at the
very last one — the run was data/compute-limited rather than capacity-limited.

```
    step        lr   train   valid
    1000  6.00e-04  5.7140  4.6102
   20000  5.58e-04  3.4624  3.4897
   50000  3.62e-04  3.3067  3.3376
   80000  1.42e-04  3.1738  3.2088
  107000  6.00e-05  3.0982  3.1375
```

## Architecture

| | |
|---|---|
| params | 124.67 M |
| dim | 768 |
| layers | 12 |
| heads (query / kv) | 12 / 4 (GQA) |
| FFN hidden | 2,048 (SwiGLU) |
| context | 1,024 |
| vocab | 32,000 (ByteLevel BPE) |
| norm | RMSNorm, pre-norm, fp32 |
| biases | none |
| embeddings | untied |

RoPE uses the half-split convention (`rotate_half`, as in HuggingFace) rather than Meta's
interleaved `view_as_complex`, with base 10,000.

## Training

| | |
|---|---|
| data | FineWeb, 5.27 B tokens (~42 tokens/param, ~2x Chinchilla) |
| steps | 107,123 |
| global batch | 49,152 tokens (24 x 1024 per rank, 2 ranks) |
| optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 |
| lr | 6e-4 -> 6e-5, cosine with 1,000-step linear warmup |
| grad clip | 1.0 |
| dropout | 0.0 |
| precision | bf16 autocast, fp32 norms |
| hardware | 2 x RTX A5000 (no NVLink), DDP |
| wall clock | 11.6 h |

Init is `N(0, 0.02)` throughout, with the two residual-writing projections (`w_o`, `w2`)
scaled by `1/sqrt(2 * n_layers)` to keep residual-stream variance depth-independent.

## Usage

This is **not** a `transformers` architecture, so `from_pretrained` will not work. Fetch
the model class, the weights and the tokenizer:

```python
import torch
from dataclasses import dataclass
from typing import Optional
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

repo = "Neeraj17/llama-124m-fineweb"
import importlib.util
spec = importlib.util.spec_from_file_location("m", hf_hub_download(repo, "model.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

@dataclass
class CFG:
    vocab_size: int = 32000; max_seq_len: int = 1024
    dim: int = 768; n_layers: int = 12
    n_heads_q: int = 12; n_heads_kv: Optional[int] = 4
    dropout: float = 0.0; multiple_of: int = 256
    ffn_multiplier: Optional[int] = None
    norm_eps: float = 1e-6; init_std: float = 0.02

tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
model = m.Transformer(CFG())
sd = torch.load(hf_hub_download(repo, "best_model.pth"), map_location="cpu")
model.load_state_dict(sd["model_state_dict"])
model.eval().cuda()

ids = tok.encode("The French Revolution began in").ids
x = torch.tensor([ids], device="cuda")
pos = torch.arange(x.size(1), device="cuda").unsqueeze(0)
with torch.no_grad():
    logits = model(x, pos, mask=None)
print(tok.decode([logits[0, -1].argmax().item()]))
```

`forward(x, pos, mask)` takes explicit position ids; pass `mask=None` for causal attention.

## Important notes

- **Token id 0 (`<|beginoftext|>`) is untrained.** The corpus was packed with
  `<|endoftext|>` (id 1) separators only, so row 0 of the embedding never received a
  gradient. Use **id 1** as the unconditional seed.
- **1,024 tokens is a hard ceiling** on prompt + generation; there is no sliding window,
  and exceeding it raises an assertion.
- Bits-per-char is comparable only within a dataset. The 41M TinyStories model in the
  GitHub repo reaches 0.40 bpc, which is *not* evidence it is the better model —
  synthetic children's stories are a far simpler distribution than open web text.

## Limitations

A 124M model trained on 5.3B tokens, with no instruction tuning or alignment of any kind.
Expect repetition loops, drift away from the prompt, and confidently stated false facts
(it dated the Declaration of the Rights of Man to 1794). It reproduces the biases of
FineWeb. Suitable for research and education, not for production or for factual queries.
