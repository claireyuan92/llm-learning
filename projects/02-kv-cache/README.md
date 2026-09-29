# KV Cache

Implement a KV cache for autoregressive transformer decoding from scratch.

## Goal
Understand why decoding is memory-bound and how caching K/V avoids recomputation —
the classic compute-vs-memory tradeoff.

## Milestones
- [ ] Naive decode loop without cache (baseline)
- [ ] Add KV cache; measure per-token latency before / after
- [ ] Estimate memory footprint vs sequence length and batch size
