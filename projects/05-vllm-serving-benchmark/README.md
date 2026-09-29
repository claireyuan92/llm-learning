# vLLM Serving Benchmark

Serve an LLM with vLLM and benchmark it.

## Goal
Learn what production inference serving looks like: continuous batching,
paged attention, throughput vs latency.

## Milestones
- [ ] Serve a 7B-class model with vLLM
- [ ] Benchmark tokens/sec vs concurrent requests
- [ ] Sweep max_num_seqs / max_num_batched_tokens; find the knee
