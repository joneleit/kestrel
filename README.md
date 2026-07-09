# kestrel

An LLM inference engine built from scratch in PyTorch: paged KV cache,
continuous batching, prefix caching, chunked prefill, speculative decoding,
INT8 weight-only quantization, a Triton paged-attention decode kernel, and an
OpenAI-compatible server with Prometheus metrics.

No vLLM, no TGI, no HF `generate()`. The point is to own every layer between
an HTTP request and a CUDA kernel.

## Correctness before speed

The engine's core invariant: **paged, batched, cache-reusing decoding is
token-for-token identical to naive dense decoding.** The test suite enforces
this exactly (greedy match) across mixed batch shapes, chunked prefill with
tiny token budgets, prefix-cache hits, full-block edge cases, and
preemption-with-recompute under artificial memory pressure:

```
$ pytest
41 passed, 3 skipped (Triton kernel tests require CUDA)
```

Speculative decoding is verified against the Leviathan et al. (2023)
guarantee: greedy spec output equals greedy target-only output, and the
empirical one-step token distribution matches plain target sampling within
5-sigma binomial tolerance.

## Architecture

```
HTTP (FastAPI, SSE streaming)                server/api.py
  └─ EngineRunner thread ── step loop        core/engine.py
       ├─ Scheduler: continuous batching,    core/scheduler.py
       │    chunked prefill, preemption
       ├─ BlockAllocator: refcounts, LRU     core/block_allocator.py
       │    eviction, hash prefix cache
       ├─ LlamaModel.forward_paged           models/llama.py
       │    ├─ CPU/reference: gather + SDPA  (paged==dense oracle)
       │    └─ GPU: Triton decode kernel     kernels/paged_attention.py
       ├─ PagedKVCache: block tensors,       core/kv_cache.py
       │    slot scatter / block-table gather
       └─ Sampler: temp / top-k / top-p      core/sampler.py
```

One `engine.step()` = one flat token batch mixing prefill chunks and decodes
under a token budget (vLLM-v1-style). Preemption is recompute-based; because
full blocks are published to the prefix cache, a preempted sequence usually
re-matches its own blocks on readmission and recomputes almost nothing. See
[DESIGN.md](DESIGN.md) for the memory math and policy tradeoffs.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                   # CPU, ~5s
python scripts/serve.py --tiny           # smoke-test server, no downloads

# real model on a GPU box
python scripts/download_weights.py meta-llama/Llama-3.2-1B ~/models/llama-3.2-1b
python scripts/serve.py --model-dir ~/models/llama-3.2-1b --device cuda --dtype float16

curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 32, "stream": true}'
```

## Benchmarks

Continuous batching on CPU with the test model (2 x vCPU, fp32), synthetic
32-token prompts, 16 generated tokens each — the scaling shape is the point,
absolute numbers are CPU-bound:

| concurrency | tokens/s | mean step | p99 step |
|---:|---:|---:|---:|
| 1  | 362  | 2.7 ms  | 4.2 ms  |
| 4  | 1010 | 3.9 ms  | 5.1 ms  |
| 16 | 1219 | 13.1 ms | 18.7 ms |

Reproduce: `python benchmarks/bench_engine.py --tiny --csv results/cpu_tiny.csv`.

GPU benchmarks (`benchmarks/bench_http.py`) run the same streaming workload
against kestrel and vLLM serving the identical checkpoint for TTFT p50/p99 and
aggregate throughput; `deploy/` has Terraform for a g5.xlarge spot instance
(~$0.30–0.45/hr), an EKS manifest that autoscales on KV-cache utilization
instead of CPU, and a Grafana dashboard for the four serving signals that
matter (tokens/s, TTFT, step latency, KV pressure).

## What's deliberately not here (yet)

Tensor parallelism, CUDA graphs, W8A8/FP8, a fused prefill kernel, and
engine-integrated speculative decoding are tracked as exercises in
[ROADMAP.md](ROADMAP.md) with pointers into the code. The speculative
generator currently runs standalone over (draft, target) dense forwards;
integrating it into the batched scheduler (multi-token verify = `query_len k+1`)
is the single best exercise in the repo.

## License

MIT
