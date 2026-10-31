# KV Cache

Implement a KV cache for autoregressive transformer decoding from scratch.

## Goal
Understand why decoding is memory-bound and how caching K/V avoids recomputation —
the classic compute-vs-memory tradeoff.

## Milestones
- [x] Naive decode loop without cache (baseline)
- [x] Add KV cache; measure per-token latency before / after
- [x] Estimate memory footprint vs sequence length and batch size

## Run the example

Requires Python 3 and NumPy:

```bash
python3 -m pip install numpy
python3 test_kv_cache.py
python3 __main__.py
```

The demo compares NumPy decoding with and without a KV cache, prints per-token
latency, verifies that both paths produce identical output, and estimates cache
memory.
