# Serving-engine test strategy

What to test in an LLM serving engine, which invariants and metrics matter for each subsystem, and why. The simulator in this repo implements every check below so the harness can be exercised without GPUs; on real hardware the same checks run against engine debug hooks and the replay client.

## Principles

1. **Correctness is output equality against a golden, not "it didn't crash".** Most serving bugs (prefix collisions, use-after-free on KV blocks, lost chunks) produce plausible text. Run greedy decoding and compare token-for-token with a reference path that shares no scheduling or caching code: the same model with prefix caching off, batch size 1, no preemption.
2. **Audit state every step, not only at the end.** A ref-count drift found at step 40 points at the code that caused it. The same drift found as a leak at shutdown points at nothing.
3. **Prove the test reached the path.** A pressure test that never preempted passes vacuously. Every scenario asserts its own coverage: preemptions > N, prefix-hit tokens > N, aborts landing in each queue state.
4. **Performance tests assert behaviour, with a trade-off stated.** "Disaggregation cuts ITL p99 ≥3×" and "…at a TTFT cost" are both asserted. A report that only shows the win hides the cost a customer will see.

## Prefill / decode disaggregation

| Check | Why |
|---|---|
| Outputs after the KV transfer equal colocated outputs | The transfer is a copy of block contents through two different block tables; an off-by-one in the block mapping is silent |
| Abort mid-transfer frees both sides | The request is held by two pools at once; the classic leak |
| A full decode pool back-pressures transfers, never drops | Decode admission is the real capacity limit in disaggregated serving |
| ITL p99 under a prefill-heavy mix, colocated vs disaggregated at equal GPU count | The reason to disaggregate: decode steps stop carrying prefill work |
| TTFT vs link bandwidth | TTFT includes the KV hop; a misconfigured transport (TCP fallback instead of RDMA) shows up here first |

Metrics: TTFT (includes transfer), ITL p99 and max, transfer time, transfer queue wait, decode-pool free blocks.

## KV-cache management (paged blocks, prefix caching, eviction)

| Invariant | Why |
|---|---|
| ref count == number of block tables holding the block, every step | Catches a reference taken or dropped once too often at the step it happens |
| ref count ≥ 0; a referenced block is never reallocated | Use-after-free on the GPU means another request's KV is overwritten |
| No leaked blocks or swap slots at quiescence (after finish **and** abort) | Leaks shrink effective capacity until the engine thrashes |
| Prefix-cache hit output == cold-run output | The block hash must cover the whole prefix (parent-chained), plus any extra keys (LoRA id, multimodal hashes) |
| LRU evicts least-recently-freed cached blocks first; an evicted hash leaves the table | Stale hash entries are a wrong-answer bug |

Workloads that matter: shared system prompts (should hit), the same document at different positions (must not hit), memory pressure with sharing plus swap (shared blocks during preemption).

Metrics: prefix-hit tokens / rate, evictions, blocks used, TTFT on shared-prefix traffic.

## Dynamic (continuous) batching and scheduling

| Invariant | Why |
|---|---|
| Tokens per step ≤ `max_num_batched_tokens`; sequences ≤ `max_num_seqs` | Activation buffers and CUDA-graph buckets are sized from these |
| Preemption (recompute and swap) neither loses nor duplicates a token | Recompute re-prefills prompt + generated tokens; an off-by-one repeats or skips one |
| No request starves or stalls | A queue head that can never fit blocks everything behind it |

| Behaviour | Expectation |
|---|---|
| Throughput vs batch cap | Rises steeply (memory-bound decode amortises the weight read) while per-token latency rises slightly |
| Long prompt without chunked prefill | One step of max ITL ≈ the whole prefill; every running decode stalls |
| Chunked prefill | Removes the stall; raises the ITL floor because many steps carry a chunk |
| KV pressure | Preemption storm: preemptions spike, ITL max and E2E explode, throughput drops. An admission watermark trades TTFT for stability |

Metrics: preemptions (and per-step), max step tokens, ITL max (not only p99: stalls hide in the tail), goodput under SLO.

## Release gating

- Every engine release runs the full invariant scenario matrix (`servesim qualify`); any violation is NO-GO.
- Every release runs a replay of representative traces (Poisson and bursty arrivals, ShareGPT-like lengths, a shared-prefix mix) against the real endpoint, gated by an SLO file on TTFT/ITL/TPOT percentiles, error rate, `usage` accuracy and goodput.
- SLO files are set from a baseline run on the target hardware, not copied between models.
