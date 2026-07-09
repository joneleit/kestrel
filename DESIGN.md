# Design notes

Decisions, the math behind them, and the tradeoffs I'd defend in review.

## 1. Why paged KV

A contiguous per-sequence KV allocation must reserve `max_model_len` upfront.
At Llama-3-8B scale (fp16, 32 layers, 8 KV heads, head_dim 128) KV costs
`2 * 32 * 8 * 128 * 2 = 128 KiB per token`, so one reserved 4k-context slot
is 512 MiB whether or not tokens exist. Paging (16-token blocks = 2 MiB each)
allocates on demand, eliminating internal fragmentation beyond the last
partial block — the same argument as OS virtual memory, which is exactly
where vLLM took it from.

Block size 16 balances two pressures: smaller blocks → less waste in the last
block and finer prefix-cache granularity, but more block-table indirection and
worse kernel memory coalescing; larger blocks → the reverse. 16–32 is the
industry sweet spot; it's a config knob here (`EngineConfig.block_size`), and
the tests deliberately run at block_size=4 to stress boundary conditions.

## 2. Scheduling: one flat batch, token budget, chunked prefill

Each step assembles up to `max_num_batched_tokens` tokens: running sequences
first (decode = 1 token; unfinished prefills continue), then waiting
sequences. Prefill chunking means a 3,000-token prompt cannot stall everyone
else's inter-token latency: it is sliced across steps to fit the budget. This
is the vLLM v1 / Sarathi-style design; the older "prefill batch vs decode
batch" split creates head-of-line blocking that shows up directly as p99 TTFT.

Admission is FCFS. When the queue head doesn't fit the remaining budget, we
stop admitting rather than skip ahead — skipping starves long prompts under
sustained short-prompt load. Fairness-aware policies are ROADMAP exercise 6.

## 3. Preemption: recompute, not swap

When a decode can't get a block, the scheduler preempts from the back of the
running list (latest arrival loses), frees its blocks, and requeues it at the
front of the waiting queue. Two ways to preserve the victim's work:

- **Swap** KV to host RAM: preserves compute, costs PCIe bandwidth both ways
  (~25 GB/s effective on PCIe 4 x16 vs ~2 TB/s HBM), and pins host memory.
- **Recompute** on readmission: costs one prefill over prompt+generated
  tokens, needs no extra memory tier.

Recompute wins here for a subtle reason: full blocks were already published
to the prefix cache when first computed, and freeing a block with a hash
keeps its contents in an LRU "evictable" pool rather than clearing it. On
readmission the victim usually re-matches its own blocks and recomputes only
the tail. `test_preempted_output_survives_recompute` proves output
equivalence under forced preemption; the free/evictable/active state machine
has its own invariant checker (`BlockAllocator.check_invariants`).

## 4. Prefix caching: chain hashes + collision safety

Full blocks are indexed by `hash((parent_hash, block_tokens))` — a rolling
chain, so a hash commits to the *entire* prefix, not just one block's tokens.
Equal hash + equal stored tokens ⇒ safe reuse (token comparison on lookup
makes a hash collision degrade to a miss, never wrong output). Generated
tokens are hashed too, not just prompts — that's what makes preemption
recovery and multi-turn reuse cheap.

Edge case that bit me in testing: when a prompt is an exact multiple of
block_size and fully cached, "nothing left to compute" would mean no logits
to sample from. The scheduler releases the last matched block so at least one
token runs through the model (`Scheduler._match_prefix`).

## 5. Attention: reference path vs Triton kernel

The model has one weight set and two attention paths:

- **Reference** (CPU + oracle): gather a sequence's KV from its block table
  into contiguous tensors, run `scaled_dot_product_attention` with an explicit
  causal offset mask. Readable, obviously correct, used by the parity tests.
- **Triton decode kernel** (GPU): one program per (sequence, head); walks the
  block table with an online-softmax accumulator (running max `m`, denominator
  `l`, accumulator `acc`, flash-attention-style rescaling), never
  materializing the score vector. GQA maps query head h → KV head
  `h // n_rep`. Verified against the reference on Llama-3-8B shapes
  (`tests/test_triton_kernel.py`, CUDA-only).

Prefill stays on SDPA-over-gathered-KV even on GPU: prefill is compute-bound
(matmul-dominated) and SDPA already dispatches to flash kernels; decode is
memory-bound and block-table indirection is where a custom kernel pays.

## 6. Speculative decoding: exactness first

`speculative_generate` implements Leviathan et al. rejection sampling: accept
proposal x with prob `min(1, p_t(x)/p_d(x))`, resample rejections from
`normalize(max(0, p_t - p_d))`, bonus token when all k accepted. The
procedure samples *exactly* from the target distribution — speculation is a
latency optimization with zero quality tradeoff, which is why the tests can
assert greedy token-identity and distributional equality instead of
eyeballing outputs.

It intentionally runs standalone (dense forwards, re-encoding context each
round). Engine integration needs multi-token verify (a decode with
query_len = k+1 — the paged forward already supports it via `SeqMeta`) plus
KV rollback on rejection (truncate `num_computed_tokens`, keep block refs).
That's ROADMAP exercise 7, left open on purpose.

## 7. INT8 weight-only quant

Symmetric per-output-channel: `scale = max|row| / 127`. No calibration data
needed (activations stay float), quality loss negligible at this granularity
for decoder LLMs. Implementation dequantizes in `forward` — that realizes the
memory win (which converts directly into more KV blocks, i.e. more
concurrency) but not the compute win; the fused int8 GEMM with epilogue
rescale is exercise 9. `lm_head` is skipped by default: it feeds the softmax
directly and is the most error-sensitive layer.

## 8. Streaming detokenization

Decoding a token-id *prefix* can end mid-UTF-8 sequence; Python renders the
partial bytes as U+FFFD, and those characters can retroactively merge into a
different character once the next token's bytes arrive (b"\xe2\xe2" is two
replacement chars now; the second \xe2 may become € later). The streamer
therefore withholds the entire trailing U+FFFD run until it resolves or the
stream ends. Found by `test_streaming_matches_nonstreaming`, which asserts
stream concatenation equals the non-streaming decode byte-for-byte — worth
keeping because Llama-family tokenizers emit byte-fallback tokens that hit
this in production, not just random test models.

## 9. Serving metrics

Four signals drive operations: output tokens/s (throughput), TTFT p50/p99
(queueing + prefill health), step latency (inter-token latency proxy), and
KV utilization (the true saturation signal — the EKS HPA scales on it rather
than CPU, which is meaningless for a GPU-bound server). Preemption rate is
the early-warning: sustained preemptions mean the KV pool is undersized for
the workload's context-length mix.
