# Roadmap / exercises

Ordered roughly by value-per-effort. Each item says where to start in the
code and what "done" means. Interview-style questions worth being able to
answer from memory are marked ❓.

## 1. GPU bring-up and real benchmarks
Run `deploy/terraform` (g5.xlarge spot), serve Llama-3.2-1B, run
`benchmarks/bench_http.py` against kestrel and vLLM on the same checkpoint.
Done = a results table + two graphs (tokens/s vs concurrency, TTFT p99 vs
load) committed to `results/`, and an honest paragraph on where vLLM wins.
❓ Why does vLLM's throughput advantage grow with batch size?

## 2. Wire the Triton kernel into `forward_paged`
`models/llama.py::_paged_attention` currently always uses the gather+SDPA
reference. Add a dispatch: CUDA + all query_len==1 → build padded block-table
/ context-len tensors once per step, call `paged_attention_decode`. Run
`tests/test_triton_kernel.py` on the GPU box first.
Done = decode steps hit the kernel (assert via a counter), parity suite still
green on GPU. ❓ Why pad block tables to max_blocks instead of ragged?

## 3. Profile before optimizing
`torch.profiler` around `engine.step()` under the bench workload; export a
chrome trace. Find the top-3 hotspots. Candidates you should *expect*: block
gather copies, per-sequence attention loop, sampler `.item()` syncs.
Done = trace committed + one fix with before/after numbers.
❓ What does a `.item()` call do to CUDA stream utilization?

## 4. Incremental prefill for the draft model (speculative)
`core/speculative.py` re-runs both models over the full context every round
(O(n²)). Give the draft its own KV cache and make target verify use one
forward. Done = same test suite passes (it must — the tests encode the
correctness guarantee), plus a measured acceptance-rate vs k curve.

## 5. Batched sampler
`core/sampler.py` loops rows because per-row params differ. Bucket rows by
identical (temperature, top_k, top_p), apply each bucket vectorized.
Done = `test_seeded_sampling_deterministic` still passes; microbenchmark
shows the win at batch 64+.

## 6. Fairness / priority scheduling
`core/scheduler.py` is FCFS with back-of-list preemption. Implement one
alternative (priority classes, or deficit round-robin on token counts) behind
a config flag. Done = a test demonstrating starvation under FCFS that the new
policy fixes. ❓ Who should you preempt when all sequences have equal priority
— newest or oldest, and why?

## 7. Engine-integrated speculative decoding (the big one)
Multi-token verify is already expressible: schedule the target with
query_len = k+1 (`SeqMeta` supports it — that's how chunked prefill works).
Missing pieces: draft model management in the engine, KV rollback on
rejection (truncate `num_computed_tokens`; blocks past the truncation stay
allocated and get overwritten via slot_mapping), and scheduler awareness so
budgets account for k+1-token decodes.
Done = spec-decode through the HTTP server, greedy-parity test at the engine
level, measured speedup on a real model pair (e.g. 1B draft / 8B target).
❓ Why must the KV written for rejected tokens be discarded, and where exactly
does kestrel overwrite it?

## 8. Fused prefill kernel over paged KV
Triton flash-attention forward reading K/V directly through block tables
(skip the gather). Compare against SDPA-on-gathered at prompt lengths 512–8k.
❓ At what arithmetic intensity does gather cost stop mattering?

## 9. Fused INT8 GEMM
`core/quant.py` dequantizes in forward. Write the Triton int8 matmul with
per-channel rescale in the epilogue; benchmark tokens/s and accuracy delta
vs fp16 on a real checkpoint. Then compare against AWQ-style activation-aware
scaling. ❓ Why does weight-only quant help memory-bound decode but barely
help compute-bound prefill?

## 10. Tensor parallelism
Row/column-parallel linears + all-reduce after o_proj and down_proj, NCCL
process group, KV cache sharded by head. This is a multi-week project; do it
after 1–7. ❓ Why do attention heads shard cleanly but the MLP needs the
column-then-row split?

## Commit discipline

Work these top-to-bottom, one branch per exercise, PR-style messages
explaining the *why*, benchmarks in the PR description. That history — with
its failed experiments and reverts — is the artifact that proves the work is
yours.

