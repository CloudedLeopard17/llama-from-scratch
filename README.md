# LLaMA from scratch

A LLaMA-style decoder-only transformer written from scratch in PyTorch, trained in three
tiers of increasing scale — character-level Shakespeare, then TinyStories, then
FineWeb — on a 2×RTX A5000 workstation.

Every architectural component (RoPE, grouped-query attention, RMSNorm, SwiGLU) is
implemented directly rather than imported, and each was verified numerically against a
reference implementation before training.

**Weights for the 124M FineWeb model:**
[huggingface.co/Neeraj17/llama-124m-fineweb](https://huggingface.co/Neeraj17/llama-124m-fineweb)
— checkpoint, tokenizer and a runnable quickstart.

## The three tiers

| | tier 0 | tier 1 | tier 2 |
|---|---|---|---|
| dataset | tiny-shakespeare | TinyStories V2 (GPT-4) | FineWeb |
| tokenizer | char-level | 16k ByteLevel BPE | 32k ByteLevel BPE |
| vocab | 65 | 16,000 | 32,000 |
| params | 10.67 M | 41.56 M | **124.67 M** |
| dim / layers | 384 / 6 | 512 / 8 | 768 / 12 |
| heads (q/kv) | 6 / 6 (MHA) | 8 / 4 (GQA) | 12 / 4 (GQA) |
| FFN hidden | 1,024 | 1,536 | 2,048 |
| context | 256 | 512 | 1,024 |
| train tokens | ~1.1 M chars | 534 M | **5.27 B** |
| best val loss | 1.1314 | 1.1347 | 3.1375 |
| wall clock | minutes | ~2 h | **11.6 h (2 GPU)** |

Grouped-query attention arrives at tier 1; tier 0 is plain multi-head.

## Tier-2 result (FineWeb, 124.67M params)

107,123 steps at a global batch of 49,152 tokens — 5.27 B tokens, roughly 42 tokens per
parameter, so about 2× the Chinchilla-optimal budget.

| | |
|---|---|
| final / best val loss | **3.1375** (step 107,000 — the last validation) |
| perplexity | 23.0 |
| bits per char | 0.987 |
| true train–val gap | **+0.031** |

```
    step        lr   train   valid
    1000  6.00e-04  5.7140  4.6102
   20000  5.58e-04  3.4624  3.4897
   50000  3.62e-04  3.3067  3.3376
   80000  1.42e-04  3.1738  3.2088
  107000  6.00e-05  3.0982  3.1375
```

Monotonic across all 107 validation points, with the best value at the very last one — the
run was data/compute-limited rather than capacity-limited, so it stopped too early rather
than too late. The cosine back half contributed **0.189** of the total improvement
(3.326 at step 53k → 3.138), which is most of the case for using a schedule at all.

Note that bits-per-char is only comparable *within* a dataset. Tier 1 reaches 0.40 bpc on
TinyStories, far below tier 2's 0.987 on FineWeb, because synthetic children's stories are
a much simpler distribution than arbitrary web text — not because the smaller model is
better.

## Architecture

- **RoPE** using the half-split convention (`rotate_half`, as in HuggingFace) rather than
  Meta's interleaved `view_as_complex`. Cos/sin tables are precomputed as non-persistent
  buffers and shared across all layers. Matched against a reference to 1.8e-07.
- **Grouped-query attention** via `repeat_interleave` on K/V, dispatched through
  `F.scaled_dot_product_attention` with `is_causal`. Matched to 2.1e-07.
- **RMSNorm** computed in fp32 and cast back, with eps *inside* the `rsqrt`. Bit-identical
  to `nn.RMSNorm` at every eps tested.
- **SwiGLU FFN** with hidden dim rounded up to a multiple of 256, matching all five
  published Llama configs.
- Pre-norm residual blocks, no biases anywhere, untied input/output embeddings.
- Init: `N(0, 0.02)` everywhere, with the two residual-writing projections (`w_o`, `w2`)
  scaled by `1/sqrt(2·n_layers)` so residual-stream variance stays depth-independent.

## Distributed training

Measured on 2×RTX A5000 **without an NVLink bridge**, so gradients cross PCIe:

| | ms/step | tok/s | peak mem | projected |
|---|---|---|---|---|
| 1 GPU | 320.9 | 76,595 | 17,018 MiB | 19.1 h |
| 2 GPU DDP | 371.5 | 132,311 | 17,472 MiB | 11.1 h |

**1.73×, not 2×** — per-step time rises ~16% from the all-reduce, partly hidden by DDP's
overlap with the backward pass. The real run came in at 11.6 h (398 ms/step wall); the
extra ~5% was the 1.5 GB best-checkpoint write firing on *every* validation, since the
curve never regressed.

```bash
python -m torch.distributed.run --nproc_per_node=2 train.py
```

## Inference

A KV cache is not unconditionally faster. Measured on the 41M tier-1 model:

- **Batch 1: a cache is a wash or worse (0.50×).** A 1-token forward costs 6.71 ms against
  7.19 ms for 512 tokens, so single-stream decoding is kernel-launch-latency bound, not
  compute bound — there is nothing for the cache to save.
- **It wins once batch and context grow: 3.72× at B=96, n=512.** Batching alone gives 13.3×.
- Combined batched + cached peaked at **28,817 tok/s at B=128, n=512** (784 MiB).

The cache in `inference-kv-cache.ipynb` is verified against a full uncached forward to
within 1.1e-05 max logit deviation in fp32, for both the parallel-prefill and
token-by-token paths.

## Layout

```
model.py                     extracted model classes (tier 2)
train.py                     DDP training script
tokenizer.ipynb              trains the 32k FineWeb BPE
tokenize_dataset.ipynb       packs documents into a uint16 .bin
inference-kv-cache.ipynb     KV-cache generation, single + batched
tokenizer/                   32k BPE (tier 2)
fineweb/                     TensorBoard logs for the tier-2 run
Tiny stories/                tier-1: notebooks + 16k BPE
tiny shakespeare/            tier-0: notebook
```

Tokenized `.bin` files (11 GB) and checkpoints (122 MB – 1.4 GB) are gitignored. Data is
regenerated by `tokenize_dataset.ipynb`; the tokenizers needed to do so are committed.
The tier-2 weights are published separately on
[the Hub](https://huggingface.co/Neeraj17/llama-124m-fineweb).

## Reproducing

```bash
# 1. train a tokenizer (or use the committed one)   -> tokenizer.ipynb
# 2. pack the corpus into a uint16 .bin             -> tokenize_dataset.ipynb
# 3. train
python -m torch.distributed.run --nproc_per_node=2 train.py
# 4. generate                                       -> inference-kv-cache.ipynb
```

Data is stored as a flat `uint16` array of token ids and read with `np.memmap`, with
documents separated by `<|endoftext|>` and sampled as non-overlapping windows.

## Gotchas worth knowing

Bugs found and fixed during this build, each of which produced plausible-looking output
while being wrong:

- **RMSNorm with eps outside the `rsqrt`** gives NaN forward *and* NaN gradients on
  zero input — silent until it isn't.
- **A KV cache without `@torch.no_grad()`** chains the autograd graph through the cached
  tensors across every decode step. Peak memory went 16,705 MiB → 445 MiB from that one
  decorator. Without a cache the leak doesn't compound, so it only appears once you add one.
- **A KV cache with no prefill** still produces fluent text — only the last prompt token
  gets cached, and the model papers over it. Verify numerically, not by reading samples.
- **Greedy argmax is not reproducible across batch shapes in bf16**; near-ties flip (batch
  noise ~6e-2 against tie gaps of 0.0000). Run equivalence tests in fp32.
- **`DDP(torch.compile(model))` has no `_orig_mod`.** Unwrap both layers before
  `state_dict()`, or checkpointing crashes at the first save.
- **`torch.load` without `map_location='cpu'`** makes every rank deserialize the
  checkpoint onto `cuda:0`; the caching allocator never gives it back, and rank 0 OOMs.
- **Plain `shuffle=True` instead of `DistributedSampler`** does not give both ranks
  identical batches (the seeds differ per process) — it leaves ~25% of the corpus unseen
  and ~25% seen twice.
- **Token id 0 (`<|beginoftext|>`) is untrained** in tiers 1 and 2: the data is packed with
  `<|endoftext|>` separators only, so that embedding row never received a gradient. Use
  id 1 as the unconditional seed.
